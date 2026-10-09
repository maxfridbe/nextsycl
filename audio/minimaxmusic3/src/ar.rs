//! The autoregressive stage: frame by frame (25 a second), the global language model (Qwen3-8B's shape: 36 layers of
//! 4,096, 32 query / 8 key-value heads of 128, a 12,288 SwiGLU) samples the frame's semantic code with
//! classifier-free guidance, and the depth decoder (4 layers of 4,096, 16 heads of 256, causal over at most 8
//! positions) samples the seven residual codes after it. The frame's eight codes, embedded and summed, are the
//! language model's next input; the hidden states that chose them condition the flow-matching stage.
//!
//! Two rows run through both models together: the prompt and its classifier-free twin (`prompt::unconditional`).
//! A frame, as the reference computes it:
//!
//! ```text
//!   logits = lm_head(h)  over <|audio_end|> and the 16,384 semantic codes; c0 = guided top-50 draw (1.5)
//!   seq = [proj(h), proj(embed(c0))] + pos;  for i in 1..8: c_i = head_i(depth(seq)[-1]), guided top-50 (1.5);
//!                                            seq += proj(audio_embed(c_i)) + pos
//!   h' = lm(8^-0.5 (embed(c0) + sum_i audio_embed(c_i)))        the cache grows a position
//!   frame = [h | depth's hidden for c1..c7] (the conditional row) -> mixed by the condition encoder's layer weights
//! ```
//!
//! The first frame only steps past <|audio_start|>: its codes are fed back, its hidden states are not kept.
//! Sampling is on the host (sample.rs): the logits come back each step.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, Mat, Ops, Shards};
use crate::sample::{self, Rng};

pub const HIDDEN: usize = 4096;
const HEADS: usize = 32;
const KV: usize = 8;
const HEAD: usize = 128;
const FFN: usize = 12288;
const QKV: usize = (HEADS + 2 * KV) * HEAD;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
pub const AUDIO_END: u32 = 151670;
pub const CODE_OFFSET: u32 = 151675;
pub const SEMANTIC: usize = 16384;
pub const CODEBOOKS: usize = 8;
pub const AUDIO_VOCAB: usize = 1024;
/// the guidance of both models' draws (the reference recipe's, fixed)
pub const CFG: f32 = 1.5;
/// the logits the semantic step reads: <|audio_end|>, then the codes
const SEM_LOGITS: usize = SEMANTIC + 1;
// the depth decoder
const D_HEADS: usize = 16;
const D_HEAD: usize = 256;
const D_FFN: usize = 6144;
const D_QKV: usize = 3 * HIDDEN;
const D_T: usize = 8;
/// prompt tokens a prefill pass (both rows: twice the rows)
const CHUNK: usize = 256;

struct LmLayer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

struct DLayer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Ar {
    lm: Vec<LmLayer>,
    norm: DevBuf,
    /// lm_head's rows for <|audio_end|> and the semantic codes
    head: Mat,
    /// half [SEMANTIC + 7 x 1024, HIDDEN]: the semantic codes' embedding rows, then the residual codebooks'
    table: DevBuf,
    inv_freq: DevBuf,
    /// the language model's files: the prompt's embedding rows are read from them
    file: Shards,
    dl: Vec<DLayer>,
    dnorm: DevBuf,
    proj: Mat,
    pos: DevBuf,
    heads: Vec<Mat>,
    pub int8: bool,
    pub bytes: usize,
}

/// A request's caches and work buffers
pub struct Session {
    t: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    dk: Vec<DevBuf>,
    dv: Vec<DevBuf>,
    x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    xh: DevBuf,
    /// an int8 matrix expanded to half for the prompt's wide products (int8 only)
    wh: Option<DevBuf>,
    part: DevBuf,
    out: DevBuf,
    /// the hidden state each frame starts from [2, HIDDEN]
    pub last: DevBuf,
    logits: DevBuf,
    e: DevBuf,
    dx: DevBuf,
    dlast: DevBuf,
    /// the frame's eight conditioning rows (the conditional row's)
    pub stage: DevBuf,
}

/// What a check reads back (the reference's dumps: prefill.npy, c0_logits.npy, lm_hidden.npy, depth.npy)
#[derive(Default)]
pub struct Probe {
    pub prefill: Vec<f32>,
    pub c0: Vec<f32>,
    pub lm: Vec<Vec<f32>>,
    pub depth: Vec<Vec<f32>>,
}

impl Ar {
    /// The two models from their files: the language model's shards and the depth decoder's (int8: their matrices
    /// int8 with a scale per row, else half)
    pub fn load(ops: &Ops, lm_files: Shards, depth: &Shards, int8: bool, log: &mut dyn FnMut(String)) -> Result<Ar> {
        let t0 = std::time::Instant::now();
        let f = &lm_files;
        let mut bytes = 0;
        let mut lm = Vec::new();
        for l in 0.. {
            let p = format!("model.layers.{l}.");
            if f.find(&format!("{p}input_layernorm.weight")).is_err() {
                break;
            }
            let n = |s: &str| format!("{p}{s}.weight");
            let layer = LmLayer {
                in_norm: f.dev_f32(ops, &n("input_layernorm"))?,
                post_norm: f.dev_f32(ops, &n("post_attention_layernorm"))?,
                q_norm: f.dev_f32(ops, &n("self_attn.q_norm"))?,
                k_norm: f.dev_f32(ops, &n("self_attn.k_norm"))?,
                qkv: f.mat(ops, &[(&n("self_attn.q_proj"), 0, HEADS * HEAD), (&n("self_attn.k_proj"), 0, KV * HEAD), (&n("self_attn.v_proj"), 0, KV * HEAD)], int8)?,
                o: f.mat1(ops, &n("self_attn.o_proj"), int8)?,
                gu: f.mat(ops, &[(&n("mlp.gate_proj"), 0, FFN), (&n("mlp.up_proj"), 0, FFN)], int8)?,
                down: f.mat1(ops, &n("mlp.down_proj"), int8)?,
            };
            bytes += layer.qkv.bytes() + layer.o.bytes() + layer.gu.bytes() + layer.down.bytes();
            lm.push(layer);
            if l % 6 == 5 {
                log(format!("language model: {} layers ({:.1} GiB) in {:.0} s", l + 1, bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
            }
        }
        if lm.len() != 36 || f.shape("model.layers.0.self_attn.q_proj.weight")? != [HEADS * HEAD, HIDDEN] {
            return Err(Error(format!("the language model: {} layers, not MiniMax Music 3's 36 of {HIDDEN}", lm.len())));
        }
        let head = f.mat(ops, &[("lm_head.weight", AUDIO_END as usize, AUDIO_END as usize + 1), ("lm_head.weight", CODE_OFFSET as usize, CODE_OFFSET as usize + SEMANTIC)],
                         int8)?;
        let sem = f.mat(ops, &[("model.embed_tokens.weight", CODE_OFFSET as usize, CODE_OFFSET as usize + SEMANTIC)], false)?;
        let aud = depth.mat1(ops, "audio_embeddings.weight", false)?;
        if aud.n != (CODEBOOKS - 1) * AUDIO_VOCAB {
            return Err(Error(format!("the depth decoder's audio embeddings: {} rows, not {}", aud.n, (CODEBOOKS - 1) * AUDIO_VOCAB)));
        }
        let table = DevBuf::new(&ops.gpu, (sem.n + aud.n) * HIDDEN * 2)?;
        table.copy_within(0, &sem.w, 0, sem.w.len)?;
        table.copy_within(sem.w.len, &aud.w, 0, aud.w.len)?;
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| 1.0 / THETA.powf((2 * i) as f32 / HEAD as f32)).collect();
        let mut dl = Vec::new();
        for l in 0..4 {
            let n = |s: &str| format!("layers.{l}.{s}.weight");
            dl.push(DLayer {
                in_norm: depth.dev_f32(ops, &n("input_layernorm"))?,
                post_norm: depth.dev_f32(ops, &n("post_attention_layernorm"))?,
                qkv: depth.mat(ops, &[(&n("attn.to_q"), 0, HIDDEN), (&n("attn.to_k"), 0, HIDDEN), (&n("attn.to_v"), 0, HIDDEN)], int8)?,
                o: depth.mat1(ops, &n("attn.to_out"), int8)?,
                gu: depth.mat(ops, &[(&n("gate_proj"), 0, D_FFN), (&n("up_proj"), 0, D_FFN)], int8)?,
                down: depth.mat1(ops, &n("down_proj"), int8)?,
            });
        }
        let heads = (0..CODEBOOKS - 1).map(|i| depth.mat1(ops, &format!("audio_heads.{i}.weight"), int8)).collect::<Result<Vec<_>>>()?;
        let ar = Ar {
            lm,
            norm: f.dev_f32(ops, "model.norm.weight")?,
            head,
            table,
            inv_freq: DevBuf::from_f32(&ops.gpu, &inv)?,
            dl,
            dnorm: depth.dev_f32(ops, "norm.weight")?,
            proj: depth.mat1(ops, "projection.weight", int8)?,
            pos: depth.dev_f32(ops, "pos_embedding.weight")?,
            heads,
            int8,
            bytes,
            file: lm_files,
        };
        ops.gpu.sync()?;
        log(format!("autoregressive stage ({}) in {:.0} s", if int8 { "int8" } else { "half" }, t0.elapsed().as_secs_f64()));
        Ok(ar)
    }

    /// Caches for a prompt of `prompt` tokens and up to `frames` frames, and the work buffers
    pub fn session(&self, ops: &Ops, prompt: usize, frames: usize) -> Result<Session> {
        let g = &ops.gpu;
        let t = prompt + frames + 2;
        let kv = |n: usize| -> Result<Vec<DevBuf>> { (0..n).map(|_| DevBuf::new(g, 2 * KV * t * HEAD * 2)).collect() };
        let dkv = |n: usize| -> Result<Vec<DevBuf>> { (0..n).map(|_| DevBuf::new(g, 2 * D_HEADS * D_T * D_HEAD * 2)).collect() };
        let r = 2 * CHUNK;
        let part = ops.attn_scratch(2, 1, HEADS, HEAD, t).max(ops.attn_scratch(2, 1, D_HEADS, D_HEAD, D_T)).max(1);
        Ok(Session {
            t,
            k: kv(self.lm.len())?,
            v: kv(self.lm.len())?,
            dk: dkv(self.dl.len())?,
            dv: dkv(self.dl.len())?,
            x: DevBuf::f32(g, r * HIDDEN)?,
            hb: DevBuf::f32(g, r * HIDDEN)?,
            qkv: DevBuf::f32(g, r * D_QKV.max(QKV))?,
            att: DevBuf::f32(g, r * HIDDEN)?,
            gu: DevBuf::f32(g, r * 2 * FFN)?,
            act: DevBuf::f32(g, r * FFN)?,
            tmp: DevBuf::f32(g, r * HIDDEN)?,
            xh: DevBuf::new(g, r * FFN * 2)?,
            wh: if self.int8 { Some(DevBuf::new(g, 2 * FFN * HIDDEN * 2)?) } else { None },
            part: DevBuf::f32(g, part)?,
            out: DevBuf::f32(g, r * HIDDEN)?,
            last: DevBuf::f32(g, 2 * HIDDEN)?,
            logits: DevBuf::f32(g, 2 * SEM_LOGITS)?,
            e: DevBuf::f32(g, 2 * HIDDEN)?,
            dx: DevBuf::f32(g, 4 * HIDDEN)?,
            dlast: DevBuf::f32(g, 2 * HIDDEN)?,
            stage: DevBuf::f32(g, CODEBOOKS * HIDDEN)?,
        })
    }

    /// x [r, k] . W^T added into `into` (a few rows: in one pass; more: through `tmp`)
    #[allow(clippy::too_many_arguments)]
    fn residual(&self, ops: &Ops, nsd: &Nsd, s: &Session, m: &Mat, x: *const f32, r: usize, into: &DevBuf) -> Result<()> {
        if r <= 8 {
            return ops.gemv(x, r, m.k, m, nul(), into.fp(), m.n, true);
        }
        m.apply(ops, nsd, x, r, Some(&s.xh), s.wh.as_ref(), nul(), s.tmp.fp())?;
        ops.add(into.fp(), s.tmp.fp(), r * m.n)
    }

    /// The language model over `n` new positions of both rows (s.x: rows b * n + i, from position p0), the last
    /// layer's normed output into s.out
    fn lm_pass(&self, ops: &Ops, nsd: &Nsd, s: &Session, n: usize, p0: usize) -> Result<()> {
        let r = 2 * n;
        for (li, l) in self.lm.iter().enumerate() {
            nsd.rms_norm_mod(s.x.ptr(), Dt::F32, r, HIDDEN, l.in_norm.ptr(), EPS, none(), none(), none(), s.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, s.hb.fp(), r, Some(&s.xh), s.wh.as_ref(), nul(), s.qkv.fp())?;
            ops.qk_norm_rope(s.qkv.fp(), QKV, r, n, HEADS, &l.q_norm, EPS, &self.inv_freq, p0)?;
            ops.qk_norm_rope(fp(&s.qkv, HEADS * HEAD), QKV, r, n, KV, &l.k_norm, EPS, &self.inv_freq, p0)?;
            ops.kv_store(fp(&s.qkv, HEADS * HEAD), fp(&s.qkv, (HEADS + KV) * HEAD), QKV, 2, n, KV, HEAD, s.t, p0, &s.k[li], &s.v[li])?;
            ops.attn(s.qkv.fp(), QKV, &s.k[li], &s.v[li], 2, n, HEADS, KV, HEAD, s.t, p0, s.att.fp(), s.part.fp())?;
            self.residual(ops, nsd, s, &l.o, s.att.fp(), r, &s.x)?;
            nsd.rms_norm_mod(s.x.ptr(), Dt::F32, r, HIDDEN, l.post_norm.ptr(), EPS, none(), none(), none(), s.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, s.hb.fp(), r, Some(&s.xh), s.wh.as_ref(), nul(), s.gu.fp())?;
            nsd.swiglu(s.gu.ptr(), Dt::F32, r, FFN, s.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, s, &l.down, s.act.fp(), r, &s.x)?;
        }
        nsd.rms_norm_mod(s.x.ptr(), Dt::F32, r, HIDDEN, self.norm.ptr(), EPS, none(), none(), none(), s.out.ptr(), Dt::F32)
    }

    /// The depth decoder over `n` new positions of both rows (s.dx), from position p0; normed output in s.out
    fn depth_pass(&self, ops: &Ops, nsd: &Nsd, s: &Session, n: usize, p0: usize) -> Result<()> {
        let r = 2 * n;
        for (li, l) in self.dl.iter().enumerate() {
            nsd.rms_norm_mod(s.dx.ptr(), Dt::F32, r, HIDDEN, l.in_norm.ptr(), EPS, none(), none(), none(), s.hb.ptr(), Dt::F32)?;
            ops.gemv(s.hb.fp(), r, HIDDEN, &l.qkv, nul(), s.qkv.fp(), D_QKV, false)?;
            ops.kv_store(fp(&s.qkv, HIDDEN), fp(&s.qkv, 2 * HIDDEN), D_QKV, 2, n, D_HEADS, D_HEAD, D_T, p0, &s.dk[li], &s.dv[li])?;
            ops.attn(s.qkv.fp(), D_QKV, &s.dk[li], &s.dv[li], 2, n, D_HEADS, D_HEADS, D_HEAD, D_T, p0, s.att.fp(), s.part.fp())?;
            ops.gemv(s.att.fp(), r, HIDDEN, &l.o, nul(), s.dx.fp(), HIDDEN, true)?;
            nsd.rms_norm_mod(s.dx.ptr(), Dt::F32, r, HIDDEN, l.post_norm.ptr(), EPS, none(), none(), none(), s.hb.ptr(), Dt::F32)?;
            ops.gemv(s.hb.fp(), r, HIDDEN, &l.gu, nul(), s.gu.fp(), 2 * D_FFN, false)?;
            nsd.swiglu(s.gu.ptr(), Dt::F32, r, D_FFN, s.act.ptr(), Dt::F32)?;
            ops.gemv(s.act.fp(), r, D_FFN, &l.down, nul(), s.dx.fp(), HIDDEN, true)?;
        }
        nsd.rms_norm_mod(s.dx.ptr(), Dt::F32, r, HIDDEN, self.dnorm.ptr(), EPS, none(), none(), none(), s.out.ptr(), Dt::F32)
    }

    /// The prompt (its ids and its twin's) through the language model: s.last holds both rows' last hidden state
    pub fn prefill(&self, ops: &Ops, nsd: &Nsd, s: &Session, ids: &[u32], unc: &[u32]) -> Result<()> {
        let mut rows = std::collections::BTreeMap::new();
        for id in ids.iter().chain(unc) {
            if !rows.contains_key(id) {
                rows.insert(*id, self.file.rows_f32("model.embed_tokens.weight", *id as usize, *id as usize + 1)?);
            }
        }
        let p = ids.len();
        let mut at = 0;
        while at < p {
            let n = CHUNK.min(p - at);
            let mut x = Vec::with_capacity(2 * n * HIDDEN);
            for list in [ids, unc] {
                for id in &list[at..at + n] {
                    x.extend_from_slice(&rows[id]);
                }
            }
            s.x.write(0, &bytes(&x))?;
            self.lm_pass(ops, nsd, s, n, at)?;
            if at + n == p {
                for b in 0..2 {
                    s.last.copy_within(b * HIDDEN * 4, &s.out, (b * n + n - 1) * HIDDEN * 4, HIDDEN * 4)?;
                }
            }
            at += n;
        }
        ops.gpu.sync()
    }

    /// Both rows' logits from `x` [2, HIDDEN] through `m`, read back: (conditional, unconditional)
    fn logits(&self, ops: &Ops, s: &Session, x: *const f32, m: &Mat) -> Result<(Vec<f32>, Vec<f32>)> {
        ops.gemv(x, 2, HIDDEN, m, nul(), s.logits.fp(), m.n, false)?;
        let mut v = vec![0u8; 2 * m.n * 4];
        s.logits.read(0, &mut v)?;
        let f: Vec<f32> = v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((f[..m.n].to_vec(), f[m.n..].to_vec()))
    }

    /// One frame from s.last: its codes drawn (or `force`d: a check's), the residual codes' hidden states into
    /// s.stage rows 1..8 and s.last into row 0, then the frame fed back (s.last advances). None: the model ended
    /// the song (<|audio_end|> drawn).
    #[allow(clippy::too_many_arguments)]
    pub fn frame(&self, ops: &Ops, nsd: &Nsd, s: &Session, pos: usize, rng: &mut Rng, force: Option<&[i32]>, probe: Option<&mut Probe>)
                 -> Result<Option<[i32; CODEBOOKS]>> {
        let h4 = HIDDEN * 4;
        let (c, u) = self.logits(ops, s, s.last.fp(), &self.head)?;
        let mut probe = probe;
        if let Some(p) = probe.as_deref_mut() {
            if p.c0.is_empty() {
                p.c0 = sample::guide(&c, &u, CFG);
            }
            p.lm.push(s.last.to_f32()?);
        }
        let pick = sample::semantic(&c, &u, CFG, rng);
        let c0 = match force {
            Some(f) => f[0],
            None if pick == 0 => return Ok(None),
            None => pick as i32 - 1,
        };
        let mut codes = [c0, 0, 0, 0, 0, 0, 0, 0];
        s.stage.copy_within(0, &s.last, 0, h4)?;
        // the depth sequence's first two positions: the hidden state and the semantic code, projected, + positions
        ops.gemv(s.last.fp(), 2, HIDDEN, &self.proj, nul(), s.dx.fp(), 2 * HIDDEN, false)?;
        ops.embed(&self.table, HIDDEN, &[c0], 1.0, s.e.fp(), HIDDEN, 2)?;
        ops.gemv(s.e.fp(), 2, HIDDEN, &self.proj, nul(), fp(&s.dx, HIDDEN), 2 * HIDDEN, false)?;
        for row in 0..4 {
            ops.add(fp(&s.dx, row * HIDDEN), fp(&self.pos, (row % 2) * HIDDEN), HIDDEN)?;
        }
        self.depth_pass(ops, nsd, s, 2, 0)?;
        let mut n = 2;
        for i in 1..CODEBOOKS {
            // the last position's rows
            for b in 0..2 {
                s.dlast.copy_within(b * h4, &s.out, (b * n + n - 1) * h4, h4)?;
            }
            s.stage.copy_within(i * h4, &s.dlast, 0, h4)?;
            let (c, u) = self.logits(ops, s, s.dlast.fp(), &self.heads[i - 1])?;
            let code = match force {
                Some(f) => f[i],
                None => sample::top_k(&sample::guide(&c, &u, CFG), rng) as i32,
            };
            codes[i] = code;
            if i + 1 < CODEBOOKS {
                ops.embed(&self.table, HIDDEN, &[(SEMANTIC + (i - 1) * AUDIO_VOCAB) as i32 + code], 1.0, s.e.fp(), HIDDEN, 2)?;
                ops.gemv(s.e.fp(), 2, HIDDEN, &self.proj, nul(), s.dx.fp(), HIDDEN, false)?;
                for b in 0..2 {
                    ops.add(fp(&s.dx, b * HIDDEN), fp(&self.pos, (i + 1) * HIDDEN), HIDDEN)?;
                }
                self.depth_pass(ops, nsd, s, 1, i + 1)?;
                n = 1;
            }
        }
        if let Some(p) = probe {
            let st = s.stage.to_f32()?;
            p.depth.push(st[HIDDEN..].to_vec());
        }
        // the frame fed back: its eight embeddings summed, scaled
        let mut idx = [0i32; CODEBOOKS];
        idx[0] = codes[0];
        for i in 1..CODEBOOKS {
            idx[i] = (SEMANTIC + (i - 1) * AUDIO_VOCAB) as i32 + codes[i];
        }
        ops.embed(&self.table, HIDDEN, &idx, (CODEBOOKS as f32).powf(-0.5), s.x.fp(), HIDDEN, 2)?;
        self.lm_pass(ops, nsd, s, 1, pos)?;
        s.last.copy_within(0, &s.out, 0, 2 * h4)?;
        Ok(Some(codes))
    }

    /// Reads the prefill's last hidden state (a check)
    pub fn probe_prefill(&self, s: &Session, p: &mut Probe) -> Result<()> {
        p.prefill = s.last.to_f32()?;
        Ok(())
    }

    pub fn check_codes(codes: &[i32]) -> Result<()> {
        if codes.len() != CODEBOOKS || codes[0] < 0 || codes[0] as usize >= SEMANTIC || codes[1..].iter().any(|c| *c < 0 || *c as usize >= AUDIO_VOCAB) {
            return Err(Error(format!("codes {codes:?}: not a frame's")));
        }
        Ok(())
    }
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
