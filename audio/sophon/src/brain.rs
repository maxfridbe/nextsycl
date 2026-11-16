//! The brain's language model: Qwen3-32B as DuplexJev carries it (`language_model.*` in the thinker's shards; 64 layers
//! of 5,120, 64 query / 8 key-value heads of 128, a 25,600 SwiGLU, q / k RMS norms, rotary positions at θ 5e6 - the
//! three M-RoPE axes equal for text and audio, so plain RoPE).
//!
//! The checkpoint's matrices are FP8 E4M3 with a scale an output row (compressed-tensors, per channel); they are made
//! into one of the engine's layouts on the card as they load (sp.h): `q4` (4-bit codes with a half scale and offset
//! every 32 weights, 5 bits a weight: ~19.5 GB, one card) or `q8` (int8 with a half scale every 32: ~33 GB, both
//! cards - the close one, for checks). The output head (bf16) is q8 either way; the token embedding stays in the file
//! (a row read a token).
//!
//! ```text
//!   prompt rows (S at a time, at most CHUNK): rms -> half -> [q|k|v] (oneDNN half GEMM over the matrix expanded a
//!     slice at a time) -> q / k norm + RoPE -> the cache (half, [T, 8 x 128] a layer) -> attention (sp.h) -> o -> +x
//!     -> rms -> half -> [gate|up] -> SwiGLU -> down -> +x
//!   a token (S <= 4): the same with the quantized matrices' own product (sp.h gemv)
//! ```
//!
//! A cache slot remembers what each of its positions holds (a token, or an audio row by its clip's hash), so the next
//! prompt's longest common prefix is not computed again (CONTRIBUTING rule 1's prompt cache, within a conversation).
//! Layers may be split over two cards (`split`): the residual rows hop through the host at the split.

use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Ops, Shards};

pub const HEAD: usize = 128;
const EPS: f32 = 1e-6;
/// prompt rows a pass at most
pub const CHUNK: usize = 512;
/// a wide product's matrix expanded to half this much at a time (bytes)
/// (512 MiB: every matrix of the 32B whole - gate|up is 500 MiB in half - so a product is one oneDNN call, its output
/// in place; slices stitched back by 2-D copies cost a 32-row pass ~1.2 s of its 1.6)
const SLICE_BYTES: usize = 512 << 20;
const P: &str = "language_model.model.";

/// The engine's weight layouts (sp.h)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Q8 = 0,
    Q4 = 1,
}

impl Layout {
    pub fn parse(s: &str) -> Result<Layout> {
        match s {
            "4" | "q4" => Ok(Layout::Q4),
            "8" | "q8" => Ok(Layout::Q8),
            _ => Err(Error(format!("bits {s:?}: 4 or 8"))),
        }
    }
    /// bytes of codes and scales a weight
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            Layout::Q8 => 1.0 + 2.0 / 32.0,
            Layout::Q4 => 0.5 + 4.0 / 32.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Layout::Q8 => "q8",
            Layout::Q4 => "q4",
        }
    }
}

/// A matrix [n, k] in a layout, on one card
pub struct QMat {
    pub q: DevBuf,
    pub s: DevBuf,
    pub n: usize,
    pub k: usize,
    pub wt: Layout,
}

impl QMat {
    fn alloc(gpu: &Arc<Gpu>, wt: Layout, n: usize, k: usize) -> Result<QMat> {
        let (qb, sb) = match wt {
            Layout::Q8 => (n * k, n * k / 32 * 2),
            Layout::Q4 => (n * k / 2, n * k / 32 * 4),
        };
        Ok(QMat { q: DevBuf::new(gpu, qb)?, s: DevBuf::new(gpu, sb)?, n, k, wt })
    }

    /// Byte offsets of row r in the codes and the scales
    fn at(&self, r: usize) -> (usize, usize) {
        match self.wt {
            Layout::Q8 => (r * self.k, r * self.k / 32 * 2),
            Layout::Q4 => (r * self.k / 2, r * self.k / 32 * 4),
        }
    }

    /// The parts' rows stacked (FP8 with their row scales, or bf16), made into the layout through `stage`
    fn load(ops: &Ops, f: &Shards, parts: &[&str], wt: Layout, stage: &DevBuf, sc: &DevBuf) -> Result<QMat> {
        let mut n = 0;
        let mut k = 0;
        for p in parts {
            let s = f.shape(&format!("{p}.weight"))?;
            if s.len() != 2 || (k != 0 && s[1] != k) {
                return Err(Error(format!("{p}: shape {s:?}")));
            }
            k = s[1];
            n += s[0];
        }
        if k % 32 != 0 {
            return Err(Error(format!("{}: {k} columns, not a multiple of 32", parts[0])));
        }
        let m = QMat::alloc(&ops.gpu, wt, n, k)?;
        let mut row = 0;
        for p in parts {
            let name = format!("{p}.weight");
            let (file, t) = f.find(&name)?;
            let rows = t.shape[0] as usize;
            let es = match t.dtype.as_str() {
                "F8_E4M3" => 1,
                "BF16" => 2,
                d => return Err(Error(format!("{name}: {d}, not F8_E4M3 or BF16"))),
            };
            let scales = if es == 1 {
                let s = f.f32(&format!("{p}.weight_scale"))?;
                if s.len() != rows {
                    return Err(Error(format!("{p}.weight_scale: {} values for {rows} rows (a scale per output channel expected)", s.len())));
                }
                Some(s)
            } else {
                None
            };
            let per = (stage.len / (k * es)).max(1);
            let mut buf = vec![0u8; per.min(rows) * k * es];
            let mut r0 = 0;
            while r0 < rows {
                let r1 = (r0 + per).min(rows);
                let b = &mut buf[..(r1 - r0) * k * es];
                file.read_into(t, (r0 * k * es) as u64, b).map_err(|e| Error(e.0))?;
                stage.write(0, b)?;
                let fp8 = match &scales {
                    Some(s) => {
                        sc.write(0, &crate::f32_bytes(&s[r0..r1]))?;
                        Some(sc.fp() as *const f32)
                    }
                    None => None,
                };
                let (qa, sa) = m.at(row + r0);
                ops.quant(wt as i32, stage.ptr(), fp8, r1 - r0, k, m.q.ptr().wrapping_byte_add(qa), m.s.ptr().wrapping_byte_add(sa))?;
                ops.gpu.sync()?;
                r0 = r1;
            }
            row += rows;
        }
        Ok(m)
    }

    pub fn bytes(&self) -> usize {
        self.q.len + self.s.len
    }

    /// out[m] (+)= x[m] . W^T, m <= 4
    fn gemv(&self, ops: &Ops, x: *const f32, m: usize, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        ops.qgemv(self.wt as i32, self.q.ptr(), self.s.ptr(), x, m, self.k, self.n, out, ldo, acc)
    }

    /// out [rows, n] f32 = xh [rows, k] half . W^T: the matrix expanded to half a slice of its rows at a time
    fn wide(&self, d: &Dev, xh: &DevBuf, rows: usize, out: &DevBuf) -> Result<()> {
        let per = slice_rows(self.n, self.k);
        let mut r0 = 0;
        while r0 < self.n {
            let r = per.min(self.n - r0);
            d.ops.dequant(self.wt as i32, self.q.ptr(), self.s.ptr(), self.k, r0, r, d.wh.ptr())?;
            if r == self.n {
                return d.nsd.linear(xh.ptr(), Dt::F16, rows, self.k, d.wh.ptr(), r, none(), out.ptr(), Dt::F32);
            }
            d.nsd.linear(xh.ptr(), Dt::F16, rows, self.k, d.wh.ptr(), r, none(), d.slice.ptr(), Dt::F32)?;
            out.copy_2d(r0 * 4, self.n * 4, &d.slice, 0, r * 4, r * 4, rows)?;
            r0 += r;
        }
        Ok(())
    }
}

struct Layer {
    dev: usize,
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: QMat,
    o: QMat,
    gu: QMat,
    down: QMat,
}

/// One card's kernels and work buffers (for CHUNK rows)
pub struct Dev {
    pub ops: Ops,
    pub nsd: Nsd,
    inv_freq: DevBuf,
    x: DevBuf,
    hb: DevBuf,
    xh: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    wh: DevBuf,
    slice: DevBuf,
    part: DevBuf,
}

/// A cache slot: each layer's keys and values (on its layer's card) and what its positions hold
pub struct Slot {
    pub t: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    /// what position i holds (crate::prompt::Prompt::keys); its length is how far the cache is filled
    pub keys: Vec<u64>,
}

pub struct Brain {
    pub devs: Vec<Dev>,
    layers: Vec<Layer>,
    norm: DevBuf,
    head: QMat,
    pub hidden: usize,
    pub heads: usize,
    pub kv: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub layout: Layout,
    /// the first layer on the second card (layers.len(): one card)
    pub split: usize,
    shards: Arc<Shards>,
    logits: DevBuf,
    pub bytes: Vec<usize>,
    /// a check's taps: the layers after which the residual rows are read back, and what was read
    pub taps: std::sync::Mutex<Taps>,
}

/// A check's taps: the layers to read after, and (layer, rows) read
pub type Taps = (Vec<usize>, Vec<(usize, Vec<f32>)>);

/// The rows a prompt's positions start from: token ids, and spans of rows given (audio)
pub struct Rows<'a> {
    pub ids: &'a [u32],
    /// (first position, rows [n, hidden]) - those positions take these rows instead of their token's
    pub spans: &'a [(usize, &'a [f32])],
}

impl Brain {
    /// Bytes on each card: the layers [0, split) on the first, the rest, the norm and the head on the last
    pub fn plan(shards: &Shards, layout: Layout, split: usize, q8: &[usize]) -> Result<(usize, Vec<u64>)> {
        let n_layer = (0..).take_while(|l| shards.has(&format!("{P}layers.{l}.input_layernorm.weight"))).count();
        let mut weights = 0f64;
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            let s = shards.shape(&format!("{P}layers.0.{name}.weight"))?;
            weights += (s[0] * s[1]) as f64;
        }
        let per = |l: usize| weights * if q8.contains(&l) { Layout::Q8 } else { layout }.bytes_per_weight();
        let head = shards.shape("language_model.lm_head.weight")?;
        let head_b = (head[0] * head[1]) as f64 * Layout::Q8.bytes_per_weight();
        let split = split.min(n_layer);
        let a: f64 = (0..split).map(per).sum();
        let b: f64 = (split..n_layer).map(per).sum::<f64>() + head_b;
        Ok((n_layer, if split == n_layer { vec![(a + head_b) as u64] } else { vec![a as u64, b as u64] }))
    }

    /// The brain on `gpus` (one, or two with the layers from `split` on the second), the layers in `q8` in q8 whatever
    /// the layout; `log` hears its progress
    pub fn load(shards: Arc<Shards>, gpus: &[Arc<Gpu>], layout: Layout, split: usize, q8: &[usize], theta: f64, log: &mut dyn FnMut(String)) -> Result<Brain> {
        let t0 = Instant::now();
        let f = &*shards;
        let n_layer = (0..).take_while(|l| f.has(&format!("{P}layers.{l}.input_layernorm.weight"))).count();
        if n_layer == 0 {
            return Err(Error("the thinker's shards hold no language_model layers".into()));
        }
        let split = if gpus.len() < 2 { n_layer } else { split.clamp(1, n_layer) };
        let emb = f.shape(&format!("{P}embed_tokens.weight"))?;
        let (vocab, hidden) = (emb[0], emb[1]);
        let q = f.shape(&format!("{P}layers.0.self_attn.q_proj.weight"))?[0];
        let kvr = f.shape(&format!("{P}layers.0.self_attn.k_proj.weight"))?[0];
        let ffn = f.shape(&format!("{P}layers.0.mlp.gate_proj.weight"))?[0];
        let (heads, kv) = (q / HEAD, kvr / HEAD);
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / HEAD as f64)) as f32).collect();
        let used: Vec<Arc<Gpu>> = if split == n_layer { vec![gpus[0].clone()] } else { gpus[..2].to_vec() };
        let mut devs = Vec::new();
        for g in &used {
            devs.push(Dev::new(g, hidden, heads, kv, ffn, &inv)?);
        }
        // a stage for a slice of a matrix's rows (bytes) and their scales
        let stages: Vec<(DevBuf, DevBuf)> = used.iter().map(|g| Ok((DevBuf::new(g, 64 << 20)?, DevBuf::f32(g, 65536)?))).collect::<Result<_>>()?;
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let dev = if l < split { 0 } else { 1 };
            let layout = if q8.contains(&l) { Layout::Q8 } else { layout };
            let (ops, (stage, sc)) = (&devs[dev].ops, &stages[dev]);
            let p = format!("{P}layers.{l}.");
            let lp = |s: &str| format!("{p}{s}");
            layers.push(Layer {
                dev,
                in_norm: f.dev_f32(ops, &lp("input_layernorm.weight"))?,
                post_norm: f.dev_f32(ops, &lp("post_attention_layernorm.weight"))?,
                q_norm: f.dev_f32(ops, &lp("self_attn.q_norm.weight"))?,
                k_norm: f.dev_f32(ops, &lp("self_attn.k_norm.weight"))?,
                qkv: QMat::load(ops, f, &[&lp("self_attn.q_proj"), &lp("self_attn.k_proj"), &lp("self_attn.v_proj")], layout, stage, sc)?,
                o: QMat::load(ops, f, &[&lp("self_attn.o_proj")], layout, stage, sc)?,
                gu: QMat::load(ops, f, &[&lp("mlp.gate_proj"), &lp("mlp.up_proj")], layout, stage, sc)?,
                down: QMat::load(ops, f, &[&lp("mlp.down_proj")], layout, stage, sc)?,
            });
            if l % 8 == 7 || l + 1 == n_layer {
                log(format!("brain: {} of {n_layer} layers ({}) in {:.0} s", l + 1, layout.name(), t0.elapsed().as_secs_f64()));
            }
        }
        let last = devs.len() - 1;
        let (stage, sc) = &stages[last];
        let head = QMat::load(&devs[last].ops, f, &["language_model.lm_head"], Layout::Q8, stage, sc)?;
        let norm = f.dev_f32(&devs[last].ops, &format!("{P}norm.weight"))?;
        let logits = DevBuf::f32(&devs[last].ops.gpu, vocab)?;
        drop(stages);
        let mut bytes = vec![0usize; devs.len()];
        for l in &layers {
            bytes[l.dev] += l.qkv.bytes() + l.o.bytes() + l.gu.bytes() + l.down.bytes();
        }
        bytes[last] += head.bytes();
        log(format!("brain: Qwen3 {n_layer} x {hidden} ({heads} / {kv} heads, {ffn} SwiGLU) {} in {:.0} s: {}", layout.name(), t0.elapsed().as_secs_f64(),
                    bytes.iter().zip(&devs).map(|(b, d)| format!("{:.1} GiB on {}", *b as f64 / (1u64 << 30) as f64, d.ops.gpu.name)).collect::<Vec<_>>().join(", ")));
        Ok(Brain { devs, layers, norm, head, hidden, heads, kv, ffn, vocab, layout, split, shards, logits, bytes, taps: Default::default() })
    }

    /// A cache slot of `t` positions
    pub fn slot(&self, t: usize) -> Result<Slot> {
        let hd = self.kv * HEAD;
        let mut k = Vec::new();
        let mut v = Vec::new();
        for l in &self.layers {
            let g = &self.devs[l.dev].ops.gpu;
            k.push(DevBuf::new(g, t * hd * 2)?);
            v.push(DevBuf::new(g, t * hd * 2)?);
        }
        Ok(Slot { t, k, v, keys: Vec::new() })
    }

    /// Bytes a slot of `t` positions takes on each card
    pub fn slot_bytes(&self, t: usize) -> Vec<usize> {
        let mut out = vec![0; self.devs.len()];
        for l in &self.layers {
            out[l.dev] += 2 * t * self.kv * HEAD * 2;
        }
        out
    }

    /// The token embedding's rows of `ids` (read from the file, a row a token), as float32
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &i in ids {
            if i as usize >= self.vocab {
                return Err(Error(format!("token {i} past the vocabulary ({})", self.vocab)));
            }
            out.extend(self.shards.rows_f32(&format!("{P}embed_tokens.weight"), i as usize, i as usize + 1)?);
        }
        Ok(out)
    }

    /// The rows of positions [a, b) of a prompt
    fn rows(&self, r: &Rows, a: usize, b: usize) -> Result<Vec<f32>> {
        let h = self.hidden;
        let mut out = self.embed(&r.ids[a..b])?;
        for (start, rows) in r.spans {
            let n = rows.len() / h;
            for p in a.max(*start)..b.min(start + n) {
                out[(p - a) * h..(p - a + 1) * h].copy_from_slice(&rows[(p - start) * h..(p - start + 1) * h]);
            }
        }
        Ok(out)
    }

    /// The prompt `keys` (and its rows) into `slot`, reusing the cached common prefix; returns the last position's
    /// logits (and, `all`, every computed row's argmax: a teacher-forced check) and the positions computed
    pub fn prompt(&self, slot: &mut Slot, keys: &[u64], r: &Rows, all: bool) -> Result<(Vec<f32>, Vec<u32>, usize)> {
        if keys.len() != r.ids.len() || keys.is_empty() {
            return Err(Error("a prompt's keys and ids differ in length (or are empty)".into()));
        }
        if keys.len() > slot.t {
            return Err(Error(format!("a prompt of {} positions; the context holds {}", keys.len(), slot.t)));
        }
        let mut same = slot.keys.iter().zip(keys).take_while(|(a, b)| a == b).count();
        if same == keys.len() {
            same -= 1;
        }
        slot.keys.truncate(same);
        let mut argmax = Vec::new();
        let mut last = Vec::new();
        let mut p = same;
        while p < keys.len() {
            let n = CHUNK.min(keys.len() - p);
            let rows = self.rows(r, p, p + n)?;
            let (l, am) = self.pass(slot, &rows, n, p, all, p + n == keys.len())?;
            argmax.extend(am);
            if !l.is_empty() {
                last = l;
            }
            slot.keys.extend_from_slice(&keys[p..p + n]);
            p += n;
        }
        Ok((last, argmax, keys.len() - same))
    }

    /// One token more (its key and id) into `slot`: the next logits
    pub fn step(&self, slot: &mut Slot, key: u64, id: u32) -> Result<Vec<f32>> {
        if slot.keys.len() >= slot.t {
            return Err(Error(format!("the context ({} positions) is full", slot.t)));
        }
        let rows = self.embed(&[id])?;
        let p = slot.keys.len();
        let (l, _) = self.pass(slot, &rows, 1, p, false, true)?;
        slot.keys.push(key);
        Ok(l)
    }

    /// Drop positions past `n` (a cancelled answer's), keeping the cache's prefix
    pub fn truncate(&self, slot: &mut Slot, n: usize) {
        slot.keys.truncate(n);
    }

    /// `n` rows (host, [n, hidden]) at positions p0.. through every layer: the last row's logits (`want_last`) and,
    /// `all`, every row's argmax
    fn pass(&self, slot: &Slot, rows: &[f32], n: usize, p0: usize, all: bool, want_last: bool) -> Result<(Vec<f32>, Vec<u32>)> {
        let h = self.hidden;
        let wide = n > 4;
        // rows bucketed to powers of two (oneDNN makes a primitive a shape; `warm` makes them at load): the extra rows
        // are zeros and their results unused
        let nb = if wide { n.next_power_of_two().clamp(32, CHUNK) } else { n };
        let mut cur = 0;
        self.devs[0].x.write(0, &crate::f32_bytes(rows))?;
        if nb > n {
            self.devs[0].x.view(n * h * 4, (nb - n) * h * 4)?.fill(0)?;
        }
        for (li, l) in self.layers.iter().enumerate() {
            if l.dev != cur {
                // the residual rows hop to the next card through the host
                let mut b = vec![0u8; nb * h * 4];
                self.devs[cur].x.read(0, &mut b)?;
                self.devs[l.dev].x.write(0, &b)?;
                cur = l.dev;
            }
            let d = &self.devs[cur];
            self.layer(d, l, &slot.k[li], &slot.v[li], n, nb, p0, wide)?;
            let mut taps = self.taps.lock().unwrap_or_else(|e| e.into_inner());
            if taps.0.contains(&li) {
                let x = d.x.to_f32()?[..n * h].to_vec();
                taps.1.push((li, x));
            }
        }
        let d = &self.devs[cur];
        let mut last = Vec::new();
        let mut am = Vec::new();
        if all {
            d.nsd.rms_norm_mod(d.x.ptr(), Dt::F32, n, h, self.norm.ptr(), EPS, none(), none(), none(), d.hb.ptr(), Dt::F32)?;
            for r in 0..n {
                self.head.gemv(&d.ops, fp(&d.hb, r * h), 1, self.logits.fp(), self.vocab, false)?;
                let lg = self.logits.to_f32()?;
                am.push(argmax(&lg[..self.vocab]) as u32);
                if r + 1 == n && want_last {
                    last = lg[..self.vocab].to_vec();
                }
            }
        } else if want_last {
            d.nsd.rms_norm_mod(fp(&d.x, (n - 1) * h).cast(), Dt::F32, 1, h, self.norm.ptr(), EPS, none(), none(), none(), d.hb.ptr(), Dt::F32)?;
            self.head.gemv(&d.ops, d.hb.fp(), 1, self.logits.fp(), self.vocab, false)?;
            last = self.logits.to_f32()?[..self.vocab].to_vec();
        }
        Ok((last, am))
    }

    /// Every row bucket's products made once (oneDNN's primitives for each matrix and bucket: seconds the first time)
    /// on `slot`'s first positions, so no prompt waits for them
    pub fn warm(&self, slot: &Slot) -> Result<()> {
        let mut b = 32;
        while b <= CHUNK.min(slot.t) {
            let rows = vec![0f32; b * self.hidden];
            self.pass(slot, &rows, b, 0, false, true)?;
            b *= 2;
        }
        Ok(())
    }

    /// Times of the pieces of a pass on layer 0's matrices (a check's `bench` stage): the expansion to half, oneDNN's
    /// half product for a few row counts, the quantized product for 1 and 4 rows, the whole pass for a few row counts
    pub fn bench(&self, slot: &Slot, log: &mut dyn FnMut(String)) -> Result<()> {
        let l = &self.layers[0];
        let d = &self.devs[l.dev];
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            f()?;
            d.ops.gpu.sync()?;
            let t = Instant::now();
            for _ in 0..5 {
                f()?;
            }
            d.ops.gpu.sync()?;
            Ok(t.elapsed().as_secs_f64() / 5.0)
        };
        for (name, m) in [("qkv", &l.qkv), ("o", &l.o), ("gate|up", &l.gu), ("down", &l.down)] {
            let per = slice_rows(m.n, m.k);
            let deq = time(&mut || d.ops.dequant(m.wt as i32, m.q.ptr(), m.s.ptr(), m.k, 0, per, d.wh.ptr()))? * (m.n / per) as f64;
            let gb = (m.n * m.k * 2) as f64 / 1e9;
            let mut line = format!("{name:<8} [{} x {}] {}: expand {:.2} ms ({:.0} GB/s written)", m.n, m.k, m.wt.name(), deq * 1e3, gb / deq);
            for rows in [32usize, 128, 512] {
                // (out into gu: [CHUNK, 2 x ffn] floats, the largest output)
                let t = time(&mut || d.nsd.linear(d.xh.ptr(), Dt::F16, rows, m.k, d.wh.ptr(), per, none(), d.gu.ptr(), Dt::F32))? * (m.n / per) as f64;
                line += &format!(", half GEMM {rows}: {:.2} ms", t * 1e3);
            }
            for rows in [1usize, 4] {
                let t = time(&mut || m.gemv(&d.ops, d.hb.fp(), rows, d.qkv.fp(), m.n, false))?;
                line += &format!(", q gemv {rows}: {:.2} ms ({:.0} GB/s)", t * 1e3, m.bytes() as f64 / 1e9 / t);
            }
            log(line);
        }
        for rows in [1usize, 4, 32, 128, 512] {
            let x = vec![0f32; rows * self.hidden];
            let t = Instant::now();
            self.pass(slot, &x, rows, 0, false, true)?;
            log(format!("a pass of {rows} rows: {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
        }
        Ok(())
    }

    /// The residual stream after a pass (rows [n, hidden] on the last card), for a check
    pub fn residual(&self, n: usize) -> Result<Vec<f32>> {
        let d = self.devs.last().ok_or_else(|| Error("no card".into()))?;
        Ok(d.x.to_f32()?[..n * self.hidden].to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, d: &Dev, l: &Layer, kc: &DevBuf, vc: &DevBuf, n: usize, nb: usize, p0: usize, wide: bool) -> Result<()> {
        let (h, q, hd) = (self.hidden, self.heads * HEAD, self.kv * HEAD);
        let w = q + 2 * hd;
        let (ops, nsd) = (&d.ops, &d.nsd);
        nsd.rms_norm_mod(d.x.ptr(), Dt::F32, nb, h, l.in_norm.ptr(), EPS, none(), none(), none(), d.hb.ptr(), Dt::F32)?;
        if wide {
            ops.to_half(d.hb.fp(), d.xh.ptr(), nb * h)?;
            l.qkv.wide(d, &d.xh, nb, &d.qkv)?;
        } else {
            l.qkv.gemv(ops, d.hb.fp(), n, d.qkv.fp(), w, false)?;
        }
        ops.qk_norm_rope(d.qkv.fp(), w, n, n, self.heads, &l.q_norm, EPS, &d.inv_freq, p0)?;
        ops.qk_norm_rope(fp(&d.qkv, q), w, n, n, self.kv, &l.k_norm, EPS, &d.inv_freq, p0)?;
        ops.sp_kv_store(fp(&d.qkv, q), fp(&d.qkv, q + hd), w, n, hd, p0, kc.ptr(), vc.ptr())?;
        ops.sp_attn(d.qkv.fp(), w, kc.ptr(), vc.ptr(), n, self.heads, self.kv, HEAD, p0, d.att.fp(), d.part.fp())?;
        if wide {
            if nb > n {
                d.att.view(n * q * 4, (nb - n) * q * 4)?.fill(0)?;
            }
            ops.to_half(d.att.fp(), d.xh.ptr(), nb * q)?;
            l.o.wide(d, &d.xh, nb, &d.tmp)?;
            ops.add(d.x.fp(), d.tmp.fp(), nb * h)?;
        } else {
            l.o.gemv(ops, d.att.fp(), n, d.x.fp(), h, true)?;
        }
        nsd.rms_norm_mod(d.x.ptr(), Dt::F32, nb, h, l.post_norm.ptr(), EPS, none(), none(), none(), d.hb.ptr(), Dt::F32)?;
        if wide {
            ops.to_half(d.hb.fp(), d.xh.ptr(), nb * h)?;
            l.gu.wide(d, &d.xh, nb, &d.gu)?;
        } else {
            l.gu.gemv(ops, d.hb.fp(), n, d.gu.fp(), 2 * self.ffn, false)?;
        }
        nsd.swiglu(d.gu.ptr(), Dt::F32, nb, self.ffn, d.act.ptr(), Dt::F32)?;
        if wide {
            ops.to_half(d.act.fp(), d.xh.ptr(), nb * self.ffn)?;
            l.down.wide(d, &d.xh, nb, &d.tmp)?;
            ops.add(d.x.fp(), d.tmp.fp(), nb * h)?;
        } else {
            l.down.gemv(ops, d.act.fp(), n, d.x.fp(), h, true)?;
        }
        Ok(())
    }
}

impl Dev {
    fn new(g: &Arc<Gpu>, hidden: usize, heads: usize, kv: usize, ffn: usize, inv: &[f32]) -> Result<Dev> {
        let ops = Ops::new(g)?;
        let nsd = Nsd::new(g)?;
        let r = CHUNK;
        let q = heads * HEAD;
        let w = q + 2 * kv * HEAD;
        let widest_k = hidden.max(q).max(ffn);
        // the expanded matrix: the largest whole when it fits SLICE_BYTES (then no slices to stitch), else a slice
        let biggest = [w * hidden, hidden * q, 2 * ffn * hidden, hidden * ffn].into_iter().max().unwrap_or(0) * 2;
        let wh_bytes = biggest.min(SLICE_BYTES) + (64 << 10);
        let slice_rows = if biggest <= SLICE_BYTES { 1 } else { (SLICE_BYTES / (hidden.min(q).min(ffn) * 2)).max(64) };
        // decode attention splits the keys (scratch): at most 32 chunks a head for up to 4 rows
        let part = ops.sp_attn_scratch(1, heads, kv, HEAD, 1 << 20).max(ops.sp_attn_scratch(4, heads, kv, HEAD, 1 << 20)).max(1);
        Ok(Dev {
            inv_freq: DevBuf::from_f32(g, inv)?,
            x: DevBuf::f32(g, r * hidden)?,
            hb: DevBuf::f32(g, r * hidden)?,
            xh: DevBuf::new(g, r * widest_k * 2)?,
            qkv: DevBuf::f32(g, r * w)?,
            att: DevBuf::f32(g, r * q)?,
            gu: DevBuf::f32(g, r * 2 * ffn)?,
            act: DevBuf::f32(g, r * ffn)?,
            tmp: DevBuf::f32(g, r * hidden)?,
            wh: DevBuf::new(g, wh_bytes)?,
            slice: DevBuf::f32(g, r * slice_rows)?,
            part: DevBuf::f32(g, part)?,
            ops,
            nsd,
        })
    }
}

/// Rows a slice of an [n, k] matrix expanded to half: at most SLICE_BYTES, and an even share of n (a multiple of
/// 16) - every slice one shape, so oneDNN makes one primitive a matrix and row bucket
pub fn slice_rows(n: usize, k: usize) -> usize {
    let mut c = (n * k * 2).div_ceil(SLICE_BYTES).max(1);
    while c < n && (!n.is_multiple_of(c) || !(n / c).is_multiple_of(16)) {
        c += 1;
    }
    n / c
}

/// The index of the largest value
pub fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0
}
