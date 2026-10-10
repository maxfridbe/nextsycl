//! The talker and its code predictor (transformers' Qwen3TTSTalkerForConditionalGeneration): two Qwen3 decoders - the
//! talker (1.7B: 28 layers of 2,048, 16 query / 8 key-value heads of 128, a 6,144 SwiGLU) and the predictor (5 layers
//! of 1,024, 16 / 8 heads of 128, 3,072) - with q / k RMS norms and rotary positions (θ 1e6; the talker's M-RoPE is
//! plain RoPE for text: its three position rows are equal).
//!
//! A frame (12.5 a second), as the reference's `generate` makes it:
//!
//! ```text
//!   c0 = draw(codec_head(h))                      h: the talker's last hidden state; repetition penalty over the c0s
//!   predictor over [proj(h), proj(E0(c0))]: c1 = draw(head_0(.)); then for i in 2..16:
//!                  + proj(E_{i-1}(c_{i-1})):        c_i = draw(head_{i-1}(.))      (its cache restarts each frame)
//!   h' = talker(E0(c0) + sum_i E_i(c_i) + the next text row, or tts_pad when the text has run out)
//! ```
//!
//! The frames' codes (16 codebooks of 2,048) go to the codec (codec.rs). Sampling is on the host (sample.rs).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, Mat, Ops, Shards};
use crate::sample::{self, Draw, Rng};

const HEAD: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

/// A Qwen3 decoder (its sizes read from its tensors)
pub struct Lm {
    layers: Vec<Layer>,
    norm: DevBuf,
    pub hidden: usize,
    heads: usize,
    kv: usize,
    ffn: usize,
    inv_freq: DevBuf,
}

/// One decoder's cache and work buffers for up to `rows` rows a pass and `t` positions
pub struct Cache {
    t: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    xh: DevBuf,
    /// an int8 matrix expanded to half for a wide product (int8 weights only)
    wh: Option<DevBuf>,
    part: DevBuf,
    pub out: DevBuf,
}

impl Lm {
    fn load(ops: &Ops, f: &Shards, p: &str, int8: bool) -> Result<Lm> {
        let mut layers = Vec::new();
        for l in 0.. {
            let pre = format!("{p}layers.{l}.");
            if f.find(&format!("{pre}input_layernorm.weight")).is_err() {
                break;
            }
            let n = |s: &str| format!("{pre}{s}.weight");
            let (q, k) = (f.shape(&n("self_attn.q_proj"))?[0], f.shape(&n("self_attn.k_proj"))?[0]);
            let ffn = f.shape(&n("mlp.gate_proj"))?[0];
            layers.push(Layer {
                in_norm: f.dev_f32(ops, &n("input_layernorm"))?,
                post_norm: f.dev_f32(ops, &n("post_attention_layernorm"))?,
                q_norm: f.dev_f32(ops, &n("self_attn.q_norm"))?,
                k_norm: f.dev_f32(ops, &n("self_attn.k_norm"))?,
                qkv: f.mat(ops, &[(&n("self_attn.q_proj"), 0, q), (&n("self_attn.k_proj"), 0, k), (&n("self_attn.v_proj"), 0, k)], int8)?,
                o: f.mat1(ops, &n("self_attn.o_proj"), int8)?,
                gu: f.mat(ops, &[(&n("mlp.gate_proj"), 0, ffn), (&n("mlp.up_proj"), 0, ffn)], int8)?,
                down: f.mat1(ops, &n("mlp.down_proj"), int8)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers in the checkpoint"))) };
        let hidden = first.o.n;
        let (heads, kv, ffn) = (first.o.k / HEAD, (first.qkv.n - first.o.k) / 2 / HEAD, first.gu.n / 2);
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| 1.0 / THETA.powf((2 * i) as f32 / HEAD as f32)).collect();
        Ok(Lm { norm: f.dev_f32(ops, &format!("{p}norm.weight"))?, layers, hidden, heads, kv, ffn, inv_freq: DevBuf::from_f32(&ops.gpu, &inv)? })
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv) * HEAD
    }

    pub fn cache(&self, ops: &Ops, rows: usize, t: usize) -> Result<Cache> {
        let g = &ops.gpu;
        let kv = || -> Result<Vec<DevBuf>> { (0..self.layers.len()).map(|_| DevBuf::new(g, self.kv * t * HEAD * 2)).collect() };
        let (r, w) = (rows.max(2), self.qkv_width());
        let widest = self.hidden.max(self.heads * HEAD).max(2 * self.ffn);
        Ok(Cache {
            t,
            k: kv()?,
            v: kv()?,
            x: DevBuf::f32(g, r * self.hidden)?,
            hb: DevBuf::f32(g, r * self.hidden)?,
            qkv: DevBuf::f32(g, r * w)?,
            att: DevBuf::f32(g, r * self.heads * HEAD)?,
            gu: DevBuf::f32(g, r * 2 * self.ffn)?,
            act: DevBuf::f32(g, r * self.ffn)?,
            tmp: DevBuf::f32(g, r * self.hidden)?,
            xh: DevBuf::new(g, r * widest * 2)?,
            wh: match self.layers.first().is_some_and(|l| l.qkv.scale.is_some()) {
                true => Some(DevBuf::new(g, self.layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.n * m.k).max().unwrap_or(0) * 2)?),
                false => None,
            },
            part: DevBuf::f32(g, ops.attn_scratch(1, r, self.heads, HEAD, t).max(1))?,
            out: DevBuf::f32(g, r * self.hidden)?,
        })
    }

    /// x [r, k] . W^T added into `into` (a few rows: in one pass; more: through `tmp`)
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize) -> Result<()> {
        if r <= 8 {
            return ops.gemv(x, r, m.k, m, nul(), c.x.fp(), m.n, true);
        }
        m.apply(ops, nsd, x, r, Some(&c.xh), c.wh.as_ref(), nul(), c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions (c.x, from position p0) through every layer; the normed output in c.out
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t {
            return Err(Error(format!("the cache holds {} positions, not {}", c.t, p0 + n)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.in_norm.ptr(), EPS, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, c.hb.fp(), n, Some(&c.xh), c.wh.as_ref(), nul(), c.qkv.fp())?;
            ops.qk_norm_rope(c.qkv.fp(), w, n, n, self.heads, &l.q_norm, EPS, &self.inv_freq, p0)?;
            ops.qk_norm_rope(fp(&c.qkv, q), w, n, n, self.kv, &l.k_norm, EPS, &self.inv_freq, p0)?;
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, 1, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], 1, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), n)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.post_norm.ptr(), EPS, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, c.hb.fp(), n, Some(&c.xh), c.wh.as_ref(), nul(), c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, n, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), n)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, self.norm.ptr(), EPS, none(), none(), none(), c.out.ptr(), Dt::F32)
    }
}

/// The talker, its predictor, their embeddings and heads, the text projection
pub struct Talker {
    pub lm: Lm,
    pub pred: Lm,
    /// half [codec vocab + 15 x predictor vocab, hidden]: the talker's codec embedding, then the predictor's tables
    table: DevBuf,
    pub vocab: usize,
    pub pvocab: usize,
    pub groups: usize,
    head: Mat,
    /// predictor: talker hidden -> predictor hidden (with bias), and its fifteen heads
    proj: Mat,
    proj_b: DevBuf,
    heads: Vec<Mat>,
    /// the text projection's two layers (with biases)
    fc1: Mat,
    fc1_b: DevBuf,
    fc2: Mat,
    fc2_b: DevBuf,
    pub bytes: usize,
}

/// A generation's caches and the hidden state each frame starts from
pub struct Session {
    pub lm: Cache,
    pred: Cache,
    last: DevBuf,
    logits: DevBuf,
    e: DevBuf,
    /// [rows, text hidden] work for the text projection
    th: DevBuf,
    tp: DevBuf,
    rows: usize,
    /// a forced run's disagreements: frames whose first code / whose other codes the model would not have taken
    pub misses: std::cell::Cell<[usize; 2]>,
}

/// What a frame's draws follow
pub struct Drawing {
    pub talker: Draw,
    pub predictor: Draw,
    pub repetition_penalty: f32,
    /// the end of speech, and the ids a draw may not take (the codec vocabulary's control ids but the end)
    pub eos: i32,
    pub suppress_from: usize,
    /// the end is held back for this many frames
    pub min_frames: usize,
}

impl Talker {
    pub fn load(ops: &Ops, f: &Shards, int8: bool, log: &mut dyn FnMut(String)) -> Result<Talker> {
        let t0 = std::time::Instant::now();
        let lm = Lm::load(ops, f, "talker.model.", int8)?;
        log(format!("talker: {} layers of {} ({} / {} heads, {} SwiGLU) in {:.1} s", lm.layers.len(), lm.hidden, lm.heads, lm.kv, lm.ffn, t0.elapsed().as_secs_f64()));
        let pred = Lm::load(ops, f, "talker.code_predictor.model.", int8)?;
        let groups = (0..).take_while(|i| f.find(&format!("talker.code_predictor.lm_head.{i}.weight")).is_ok()).count() + 1;
        let vocab = f.shape("talker.model.codec_embedding.weight")?[0];
        let pvocab = f.shape("talker.code_predictor.model.codec_embedding.0.weight")?[0];
        let mut parts: Vec<(String, usize)> = vec![("talker.model.codec_embedding.weight".into(), vocab)];
        parts.extend((0..groups - 1).map(|i| (format!("talker.code_predictor.model.codec_embedding.{i}.weight"), pvocab)));
        let refs: Vec<(&str, usize, usize)> = parts.iter().map(|(n, r)| (n.as_str(), 0, *r)).collect();
        let table = f.mat(ops, &refs, false)?.w;
        let heads = (0..groups - 1).map(|i| f.mat1(ops, &format!("talker.code_predictor.lm_head.{i}.weight"), int8)).collect::<Result<Vec<_>>>()?;
        let t = Talker {
            head: f.mat1(ops, "talker.codec_head.weight", int8)?,
            proj: f.mat1(ops, "talker.code_predictor.small_to_mtp_projection.weight", false)?,
            proj_b: f.dev_f32(ops, "talker.code_predictor.small_to_mtp_projection.bias")?,
            fc1: f.mat1(ops, "talker.text_projection.linear_fc1.weight", false)?,
            fc1_b: f.dev_f32(ops, "talker.text_projection.linear_fc1.bias")?,
            fc2: f.mat1(ops, "talker.text_projection.linear_fc2.weight", false)?,
            fc2_b: f.dev_f32(ops, "talker.text_projection.linear_fc2.bias")?,
            bytes: 0,
            lm,
            pred,
            table,
            vocab,
            pvocab,
            groups,
            heads,
        };
        ops.gpu.sync()?;
        log(format!("talker and code predictor ({} layers, {groups} codebooks of {pvocab}) in {:.1} s", t.pred.layers.len(), t0.elapsed().as_secs_f64()));
        Ok(t)
    }

    /// Caches for a prompt of `prompt` rows and up to `frames` frames
    pub fn session(&self, ops: &Ops, prompt: usize, frames: usize) -> Result<Session> {
        let g = &ops.gpu;
        let h = self.lm.hidden;
        Ok(Session {
            lm: self.lm.cache(ops, prompt, prompt + frames + 2)?,
            pred: self.pred.cache(ops, 2, self.groups + 1)?,
            last: DevBuf::f32(g, h)?,
            logits: DevBuf::f32(g, self.vocab.max(self.pvocab))?,
            e: DevBuf::f32(g, 2 * h)?,
            th: DevBuf::f32(g, prompt.max(2) * h)?,
            tp: DevBuf::f32(g, prompt.max(2) * h)?,
            rows: prompt.max(2),
            misses: std::cell::Cell::new([0, 0]),
        })
    }

    /// Text rows [n, text hidden] (the text embedding's) through the text projection: [n, hidden] on the host (in
    /// pieces of the session's rows)
    pub fn project_text(&self, ops: &Ops, nsd: &Nsd, s: &Session, rows: &[f32]) -> Result<Vec<f32>> {
        let k = self.fc1.k;
        let mut out = Vec::with_capacity(rows.len() / k * self.fc2.n);
        for part in rows.chunks(s.rows * k) {
            let n = part.len() / k;
            s.th.write(0, &bytes(part))?;
            self.fc1.apply(ops, nsd, s.th.fp(), n, Some(&s.lm.xh), None, self.fc1_b.ptr(), s.tp.fp())?;
            ops.silu(s.tp.fp(), n * self.fc1.n)?;
            self.fc2.apply(ops, nsd, s.tp.fp(), n, Some(&s.lm.xh), None, self.fc2_b.ptr(), s.th.fp())?;
            out.extend_from_slice(&s.th.to_f32()?[..n * self.fc2.n]);
        }
        Ok(out)
    }

    /// Rows of the combined codec table (see `index`) summed into `out` [hidden] (the kernel sums 8 at a time)
    fn sum_into(&self, ops: &Ops, s: &Session, idx: &[i32], out: *mut f32) -> Result<()> {
        let h = self.lm.hidden;
        for (i, part) in idx.chunks(8).enumerate() {
            if i == 0 {
                ops.embed(&self.table, h, part, 1.0, out, h, 1)?;
            } else {
                ops.embed(&self.table, h, part, 1.0, s.e.fp(), h, 1)?;
                ops.add(out, s.e.fp(), h)?;
            }
        }
        Ok(())
    }

    /// The same, read back: one [hidden] row on the host
    pub fn codec_sum(&self, ops: &Ops, s: &Session, idx: &[i32]) -> Result<Vec<f32>> {
        self.sum_into(ops, s, idx, s.tp.fp())?;
        Ok(s.tp.to_f32()?[..self.lm.hidden].to_vec())
    }

    /// The combined table's row of codebook `g`'s `code` (g 0: the talker's own)
    pub fn index(&self, g: usize, code: i32) -> i32 {
        if g == 0 { code } else { (self.vocab + (g - 1) * self.pvocab) as i32 + code }
    }

    /// The prompt's rows [L, hidden] through the talker; the last hidden state kept
    pub fn prefill(&self, ops: &Ops, nsd: &Nsd, s: &Session, rows: &[f32]) -> Result<()> {
        let h = self.lm.hidden;
        let l = rows.len() / h;
        s.lm.x.write(0, &bytes(rows))?;
        self.lm.pass(ops, nsd, &s.lm, l, 0)?;
        s.last.copy_within(0, &s.lm.out, (l - 1) * h * 4, h * 4)?;
        ops.gpu.sync()
    }

    /// The first codebook's logits from the last hidden state, read back
    pub fn logits0(&self, ops: &Ops, s: &Session) -> Result<Vec<f32>> {
        ops.gemv(s.last.fp(), 1, self.lm.hidden, &self.head, nul(), s.logits.fp(), self.head.n, false)?;
        Ok(s.logits.to_f32()?[..self.head.n].to_vec())
    }

    /// One frame: c0 drawn (or forced), the predictor's codes, then the talker fed `text` (the next text row, a
    /// device [hidden]) with the frame's embeddings. None: the end of speech. `seen`: the c0s so far.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(&self, ops: &Ops, nsd: &Nsd, s: &Session, pos: usize, text: *const f32, d: &Drawing, seen: &[i32], rng: &mut Rng,
                 force: Option<&[i32]>) -> Result<Option<Vec<i32>>> {
        let h = self.lm.hidden;
        let mut l0 = self.logits0(ops, s)?;
        sample::repetition_penalty(&mut l0, seen, d.repetition_penalty);
        for (i, v) in l0.iter_mut().enumerate().skip(d.suppress_from) {
            if i as i32 != d.eos {
                *v = f32::NEG_INFINITY;
            }
        }
        if seen.len() < d.min_frames {
            l0[d.eos as usize] = f32::NEG_INFINITY;
        }
        let c0 = match force {
            Some(f) => {
                if f[0] != d.eos && sample::pick(&l0, Draw::GREEDY, rng) as i32 != f[0] {
                    let m = s.misses.get();
                    s.misses.set([m[0] + 1, m[1]]);
                }
                f[0]
            }
            None => sample::pick(&l0, d.talker, rng) as i32,
        };
        if c0 == d.eos {
            return Ok(None);
        }
        let mut codes = vec![c0];
        // the predictor: [proj(h), proj(E0(c0))] at positions 0, 1, then one position a codebook
        let ph = self.pred.hidden;
        ops.gemv(s.last.fp(), 1, h, &self.proj, self.proj_b.ptr(), s.pred.x.fp(), ph, false)?;
        ops.embed(&self.table, h, &[c0], 1.0, s.e.fp(), h, 1)?;
        ops.gemv(s.e.fp(), 1, h, &self.proj, self.proj_b.ptr(), fp(&s.pred.x, ph), ph, false)?;
        self.pred.pass(ops, nsd, &s.pred, 2, 0)?;
        let mut row = 1;
        for g in 1..self.groups {
            let m = &self.heads[g - 1];
            ops.gemv(fp(&s.pred.out, row * ph), 1, ph, m, nul(), s.logits.fp(), m.n, false)?;
            let code = match force {
                Some(f) => {
                    if sample::pick(&s.logits.to_f32()?[..m.n], Draw::GREEDY, rng) as i32 != f[g] {
                        let m = s.misses.get();
                        s.misses.set([m[0], m[1] + 1]);
                    }
                    f[g]
                }
                None => sample::pick(&s.logits.to_f32()?[..m.n], d.predictor, rng) as i32,
            };
            codes.push(code);
            if g + 1 < self.groups {
                ops.embed(&self.table, h, &[self.index(g, code)], 1.0, s.e.fp(), h, 1)?;
                ops.gemv(s.e.fp(), 1, h, &self.proj, self.proj_b.ptr(), s.pred.x.fp(), ph, false)?;
                self.pred.pass(ops, nsd, &s.pred, 1, g + 1)?;
                row = 0;
            }
        }
        // the frame fed back: its embeddings summed, plus the text row
        let idx: Vec<i32> = codes.iter().enumerate().map(|(g, c)| self.index(g, *c)).collect();
        self.sum_into(ops, s, &idx, s.lm.x.fp())?;
        ops.add(s.lm.x.fp(), text, h)?;
        self.lm.pass(ops, nsd, &s.lm, 1, pos)?;
        s.last.copy_within(0, &s.lm.out, 0, h * 4)?;
        Ok(Some(codes))
    }
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
