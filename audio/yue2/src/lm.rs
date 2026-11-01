//! The backbone on the card: YuE2's Mixture of Transformers (modeling_yue2.py `Backbone`), 28 layers of 2048 with two
//! sets of weights each - the AR path (input_layernorm, self_attn, post_attention_layernorm, mlp) for the prompt, the
//! score and the semantic codes, the NAR path (the nar_ twins) for the flow stage's latent rows - one embedding table, one
//! final norm, an untied head, and the flow heads (vae2llm, llm2vae, the timestep MLP, the latent position table).
//!
//! ```text
//!   AR rows (a prompt a chunk, or a token a step for one or two sequences: the song and its classifier-free twin)
//!     x = embed(ids);  per layer: x += o(attn_causal(rope(qn(q)), rope(kn(k)), v));  x += down(silu(gate) * up)
//!     every row's k / v (normed, rotated) kept in its sequence's cache
//!   NAR rows (the flow stage: [LATENT_START, frames, LATENT_END] after the chunk's AR tokens)
//!     x = vae2llm(x_t, 0 at the ends) + time(t) + pos[local];  the NAR weights; attention over every AR key of the
//!     cache and every NAR row (theirs written after the AR rows, overwritten each pass); v = llm2vae(norm(x))
//! ```
//!
//! The matrices stay in the GGUF's Q8_0 blocks on the card (q | k | v and gate | up stacked into one matrix each: the
//! same rows, one product): a prefill or a flow pass multiplies through nsd_linear_q (a slice made bf16 / half, one
//! oneDNN GEMM), a decode step reads the blocks straight (ns_audio_yue2_gemv_q8). The residual stream is float32, the
//! products' inputs bf16 (the reference's own type) or half (the attention's output), the cache half.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi::{self, Api};

pub type P = *const c_void;
pub type M = *mut c_void;

pub const HIDDEN: usize = 2048;
pub const LAYERS: usize = 28;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HEAD: usize = 128;
pub const FFN: usize = 6144;
const QW: usize = HEADS * HEAD;
const KW: usize = KV_HEADS * HEAD;
const QKV: usize = QW + 2 * KW;
pub const LATENT: usize = 64;
pub const VOCAB: usize = 184_704;
/// the model's context (max_position_embeddings, the latent position table's length)
pub const CONTEXT: usize = 24_576;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
/// the timestep embedder's sinusoid width
const FREQ: usize = 256;
const Q8_0: u32 = 8;
/// bytes of a Q8_0 block of 32
const Q8_BLOCK: usize = 34;
/// rows a prefill pass (the prompt read in chunks of these)
pub const CHUNK: usize = 512;
/// rows a block of the per-row work (the projections, the MLP) of a pass: the MLP's float32 [rows, 12288] bounded
const ROWS: usize = 2048;

/// A matrix [n, k], Q8_0 as stored
pub struct Mat {
    pub w: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    /// rows [r0, ..) of it
    fn rows_at(&self, r0: usize) -> P {
        self.w.ptr().cast::<u8>().wrapping_add(r0 * self.k / 32 * Q8_BLOCK).cast_const().cast()
    }
}

/// One path's weights in a layer (the AR or the NAR)
struct Path {
    norm1: DevBuf,
    qkv: Mat,
    qn: DevBuf,
    kn: DevBuf,
    o: Mat,
    norm2: DevBuf,
    gu: Mat,
    down: Mat,
}

struct Layer {
    ar: Path,
    nar: Path,
}

/// One sequence's keys and values, every layer's: half [cap, 8 x 128], rows by position
pub struct Cache {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub cap: usize,
    /// the AR rows held (positions [0, len)); a flow pass writes its rows after them
    pub len: usize,
}

impl Cache {
    /// bytes a cache of `cap` rows takes
    pub fn bytes(cap: usize) -> u64 {
        (LAYERS * 2 * cap * KW * 2) as u64
    }
}

/// The work buffers by name, grown as asked (a song's longest pass sets their size)
#[derive(Default)]
pub struct Scratch {
    bufs: BTreeMap<&'static str, DevBuf>,
}

impl Scratch {
    pub fn get(&mut self, gpu: &Arc<Gpu>, name: &'static str, bytes: usize) -> Result<M> {
        if self.bufs.get(name).is_none_or(|b| b.len < bytes) {
            // the old one freed before the new one is made
            self.bufs.remove(name);
            self.bufs.insert(name, DevBuf::new(gpu, bytes.max(16))?);
        }
        Ok(self.bufs[name].ptr())
    }

    pub fn buf(&self, name: &str) -> &DevBuf {
        &self.bufs[name]
    }

    pub fn clear(&mut self) {
        self.bufs.clear();
    }

    pub fn bytes(&self) -> usize {
        self.bufs.values().map(|b| b.len).sum()
    }
}

/// A pointer `bytes` into a device allocation
pub fn at(p: M, bytes: usize) -> M {
    p.cast::<u8>().wrapping_add(bytes).cast()
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// RoPE's (cos, sin), [CONTEXT, 64, 2]: upstream's float32 arithmetic - inv_freq = 1 / theta^(2 i / 128), angle = position *
/// inv_freq - with the cosine and sine of that float32 angle
fn rope_table() -> Vec<f32> {
    let inv: Vec<f32> = (0..HEAD / 2).map(|i| 1.0 / THETA.powf((2 * i) as f32 / HEAD as f32)).collect();
    let mut cs = Vec::with_capacity(CONTEXT * HEAD);
    for t in 0..CONTEXT {
        for f in &inv {
            let a = t as f32 * f;
            cs.push(a.cos());
            cs.push(a.sin());
        }
    }
    cs
}

/// A float tensor of the file (F32, BF16, F16) on the host
pub fn host_vec(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{name}: a {} tensor where float was expected", other.name()))),
    })
}

pub struct Lm {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static Api,
    layers: Vec<Layer>,
    /// the embedding table, Q8_0 [184704, 2048]
    embed: DevBuf,
    head: Mat,
    norm: DevBuf,
    v2l_w: DevBuf,
    v2l_b: DevBuf,
    l2v_w: DevBuf,
    l2v_b: DevBuf,
    /// the timestep MLP (float32 on the host: a product a flow evaluation)
    t1: (Vec<f32>, Vec<f32>),
    t2: (Vec<f32>, Vec<f32>),
    /// the latent position table, bfloat16 bits [24576, 2048] on the host (a chunk's rows go up once)
    pos: Vec<u16>,
    /// RoPE's (cos, sin), float32 [CONTEXT, 64, 2]: upstream's float32 angles (position * inv_freq)
    cs: DevBuf,
    pub bytes: u64,
}

impl Lm {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Lm> {
        let t0 = std::time::Instant::now();
        if f.architecture() != "yue2" {
            return Err(Error(format!("{}: a {} file; YuE2's language model is arch yue2", f.paths[0].display(), f.architecture())));
        }
        let n = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        for (k, want) in [("embedding_length", HIDDEN), ("block_count", LAYERS), ("attention.head_count", HEADS), ("attention.head_count_kv", KV_HEADS),
                          ("feed_forward_length", FFN), ("latent_dim", LATENT), ("vocab_size", VOCAB)] {
            if n(k) != Some(want) {
                return Err(Error(format!("{}: yue2.{k} {:?}, {want} expected (YuE2-3B)", f.paths[0].display(), n(k))));
            }
        }
        let nsd = Nsd::new(gpu)?;
        let k = ffi::api()?;
        let bytes = std::cell::Cell::new(0u64);
        let up = |b: &[u8]| -> Result<DevBuf> {
            let d = DevBuf::new(gpu, b.len())?;
            d.write(0, b)?;
            bytes.set(bytes.get() + b.len() as u64);
            Ok(d)
        };
        // Q8_0 matrices stacked along their rows (one matrix of all their outputs)
        let mats = |names: &[String]| -> Result<Mat> {
            let mut all = Vec::new();
            let (mut rows, mut kk) = (0, 0);
            for name in names {
                let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
                if t.ty.code() != Q8_0 {
                    return Err(Error(format!("{name}: {} (this engine reads YuE2's Q8_0 file)", t.ty.name())));
                }
                let (r, c) = (t.shape[0] as usize, t.shape[1] as usize);
                if kk != 0 && c != kk {
                    return Err(Error(format!("{name}: {c} inputs, {kk} expected")));
                }
                kk = c;
                rows += r;
                all.extend(f.read(t).map_err(ge)?);
            }
            Ok(Mat { w: up(&all)?, n: rows, k: kk })
        };
        let vecd = |name: &str| -> Result<DevBuf> {
            let v = host_vec(f, name)?;
            bytes.set(bytes.get() + v.len() as u64 * 4);
            DevBuf::from_f32(gpu, &v)
        };
        let path = |p: &str, attn: &str, ffn: &str| -> Result<Path> {
            let qkv = mats(&[format!("{p}.{attn}_q.weight"), format!("{p}.{attn}_k.weight"), format!("{p}.{attn}_v.weight")])?;
            let gu = mats(&[format!("{p}.{ffn}_gate.weight"), format!("{p}.{ffn}_up.weight")])?;
            let (o, down) = (mats(&[format!("{p}.{attn}_output.weight")])?, mats(&[format!("{p}.{ffn}_down.weight")])?);
            if qkv.n != QKV || qkv.k != HIDDEN || gu.n != 2 * FFN || o.n != HIDDEN || o.k != QW || down.k != FFN {
                return Err(Error(format!("{p}: the projections' shapes are not YuE2-3B's")));
            }
            Ok(Path { norm1: vecd(&format!("{p}.{attn}_norm.weight"))?, qkv, qn: vecd(&format!("{p}.{attn}_q_norm.weight"))?,
                      kn: vecd(&format!("{p}.{attn}_k_norm.weight"))?, o, norm2: vecd(&format!("{p}.{ffn}_norm.weight"))?, gu, down })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("blk.{i}");
            layers.push(Layer { nar: path(&p, "nar_attn", "nar_ffn")?, ar: path(&p, "attn", "ffn")? });
        }
        let embed = mats(&["token_embd.weight".to_string()])?;
        let head = mats(&["output.weight".to_string()])?;
        if embed.n != VOCAB || head.n != VOCAB {
            return Err(Error("token_embd / output: not 184704 rows".into()));
        }
        let pos_t = f.tensor("latent_pos_embed.weight").ok_or_else(|| Error("no latent_pos_embed.weight".into()))?;
        if pos_t.ty != GType::BF16 || pos_t.shape != [CONTEXT as u64, HIDDEN as u64] {
            return Err(Error("latent_pos_embed.weight: bf16 [24576, 2048] expected".into()));
        }
        let pos = f.read(pos_t).map_err(ge)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let lm = Lm {
            norm: vecd("output_norm.weight")?,
            v2l_w: vecd("vae2llm.weight")?,
            v2l_b: vecd("vae2llm.bias")?,
            l2v_w: vecd("llm2vae.weight")?,
            l2v_b: vecd("llm2vae.bias")?,
            t1: (host_vec(f, "time_embd.0.weight")?, host_vec(f, "time_embd.0.bias")?),
            t2: (host_vec(f, "time_embd.1.weight")?, host_vec(f, "time_embd.1.bias")?),
            pos,
            cs: DevBuf::from_f32(gpu, &rope_table())?,
            embed: embed.w,
            head,
            layers,
            k,
            bytes: bytes.get(),
            gpu: gpu.clone(),
            nsd,
        };
        gpu.sync()?;
        log(format!("yue2: the language model {:.2} GiB on {} in {:.1} s", lm.bytes as f64 / (1u64 << 30) as f64, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(lm)
    }

    /// An empty cache of `cap` rows
    pub fn cache(&self, cap: usize) -> Result<Cache> {
        let mut k = Vec::with_capacity(LAYERS);
        let mut v = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            k.push(DevBuf::new(&self.gpu, cap * KW * 2)?);
            v.push(DevBuf::new(&self.gpu, cap * KW * 2)?);
        }
        Ok(Cache { k, v, cap, len: 0 })
    }

    fn g(&self) -> M {
        self.gpu.raw()
    }

    fn rms(&self, x: P, m: usize, w: &DevBuf, out: M, out_dt: Dt) -> Result<()> {
        self.nsd.rms_norm_mod(x, Dt::F32, m, HIDDEN, w.ptr(), EPS, none(), none(), none(), out, out_dt)
    }

    fn lin(&self, x: P, dt: Dt, m: usize, w: &Mat, out: M) -> Result<()> {
        self.nsd.linear_q(x, dt, m, w.k, w.w.ptr(), Q8_0, w.n, none(), out, Dt::F32, 0)
    }

    fn add(&self, x: M, y: P, m: usize) -> Result<()> {
        self.nsd.gate_add(x, Dt::F32, m, HIDDEN, y, Dt::F32, none(), none())
    }

    fn gemv(&self, x: P, m: usize, w: P, k: usize, n: usize, out: M, ldo: usize, acc: bool) -> Result<()> {
        // SAFETY: x holds m x k floats, w n Q8_0 rows of k, out m rows of ldo floats.
        ffi::check(unsafe { (self.k.gemv_q8)(self.g(), x.cast(), m as i64, k as i64, w, n as i64, out.cast(), ldo as i64, acc as i32) }, "q8 product")
    }

    /// Token rows of the embedding table, float32 [n, 2048] into `out`
    fn embed(&self, s: &mut Scratch, ids: &[u32], out: M) -> Result<()> {
        if let Some(t) = ids.iter().find(|t| **t as usize >= VOCAB) {
            return Err(Error(format!("token {t} past the vocabulary")));
        }
        let d = s.get(&self.gpu, "ids", ids.len() * 4)?;
        let b: Vec<u8> = ids.iter().flat_map(|t| (*t as i32).to_le_bytes()).collect();
        s.buf("ids").write(0, &b)?;
        // SAFETY: the table [VOCAB, HIDDEN] Q8_0, the ids on the card, out n x HIDDEN floats.
        ffi::check(unsafe { (self.k.embed_q8)(self.g(), self.embed.ptr(), HIDDEN as i64, d.cast(), ids.len() as i64, out.cast()) }, "embedding")
    }

    /// `m` rows of the residual x (float32 [m, 2048] on the card) at positions [pos0, pos0 + m) through every layer of
    /// one path: the AR (causal, its keys and values kept in `c` at those rows) or the NAR (the flow's rows: attending
    /// to every AR row of the cache and to each other, theirs written after the AR rows)
    fn layers(&self, s: &mut Scratch, c: &Cache, x: M, m: usize, pos0: usize, nar: bool) -> Result<()> {
        if pos0 + m > c.cap {
            return Err(Error(format!("{} rows at {pos0}: past the cache's {}", m, c.cap)));
        }
        let g = &self.gpu;
        let rb = m.min(ROWS);
        let q = s.get(g, "q", m * QW * 2)?;
        let o = s.get(g, "o", m * QW * 2)?;
        let h = s.get(g, "h", rb * FFN * 2)?;
        let qkv = s.get(g, "qkv", rb * QKV * 4)?;
        let gu = s.get(g, "gu", rb * 2 * FFN * 4)?;
        let y = s.get(g, "y", rb * HIDDEN * 4)?;
        // the products' inputs: bf16, the reference's own type (no range risk in the MLP's down projection)
        let pdt = Dt::BF16;
        for (li, l) in self.layers.iter().enumerate() {
            let p = if nar { &l.nar } else { &l.ar };
            let (kc, vc) = (c.k[li].ptr(), c.v[li].ptr());
            for b0 in (0..m).step_by(rb) {
                let bm = rb.min(m - b0);
                self.rms(at(x, b0 * HIDDEN * 4), bm, &p.norm1, h, pdt)?;
                self.lin(h, pdt, bm, &p.qkv, qkv)?;
                // SAFETY: qkv holds bm rows of 4096, q m rows of 2048 half, the caches cap rows (checked above).
                ffi::check(unsafe { (self.k.qkv)(self.g(), qkv.cast(), bm as i64, p.qn.fp(), p.kn.fp(), EPS, self.cs.fp(), (pos0 + b0) as i64,
                                                 at(q, b0 * QW * 2), kc, vc, (pos0 + b0) as i64) }, "q / k / v")?;
            }
            if nar {
                // the shared attention's query and key heads are one to one: the 16 query heads as two rows of 8
                let qg = s.get(g, "qg", m * QW * 2)?;
                let og = s.get(g, "og", m * QW * 2)?;
                // SAFETY: q and qg hold m rows of 2048 half.
                ffi::check(unsafe { (self.k.heads)(self.g(), q, qg, m as i64, 1) }, "the query heads")?;
                self.nsd.attention_qk(qg, 2 * m, KW, kc, vc, pos0 + m, KW, Dt::F16, KV_HEADS, HEAD, og, Dt::F16)?;
                // SAFETY: og and o hold m rows of 2048 half.
                ffi::check(unsafe { (self.k.heads)(self.g(), og, o, m as i64, 0) }, "the query heads")?;
            } else {
                // the query table is read as if it began `pos0` rows earlier (nsd takes row0 of a table from row 0)
                let q0 = q.cast::<u8>().wrapping_sub(pos0 * QW * 2).cast_const().cast();
                self.nsd.attention_causal_rows(q0, kc, vc, Dt::F16, pos0, m, HEADS, KV_HEADS, HEAD, QW, KW, o)?;
            }
            for b0 in (0..m).step_by(rb) {
                let bm = rb.min(m - b0);
                let xb = at(x, b0 * HIDDEN * 4);
                self.lin(at(o, b0 * QW * 2), Dt::F16, bm, &p.o, y)?;
                self.add(xb, y, bm)?;
                self.rms(xb, bm, &p.norm2, h, pdt)?;
                self.lin(h, pdt, bm, &p.gu, gu)?;
                self.nsd.swiglu(gu, Dt::F32, bm, FFN, h, pdt)?;
                self.lin(h, pdt, bm, &p.down, y)?;
                self.add(xb, y, bm)?;
            }
        }
        Ok(())
    }

    /// The tokens `ids` appended to `c` (a prompt, a chunk of CHUNK rows a pass): the residual rows at `keep` (indexes
    /// into ids) back, float32 [keep.len(), 2048] on the host (the last row's: its logits come from it)
    pub fn prefill(&self, s: &mut Scratch, c: &mut Cache, ids: &[u32], keep: &[usize]) -> Result<Vec<f32>> {
        if c.len + ids.len() > c.cap {
            return Err(Error(format!("a prompt of {} tokens after {}: past the cache's {}", ids.len(), c.len, c.cap)));
        }
        let mut out = vec![0f32; keep.len() * HIDDEN];
        for (ci, chunk) in ids.chunks(CHUNK).enumerate() {
            let m = chunk.len();
            let x = s.get(&self.gpu, "px", m * HIDDEN * 4)?;
            self.embed(s, chunk, x)?;
            self.layers(s, c, x, m, c.len, false)?;
            for (i, r) in keep.iter().enumerate() {
                if (ci * CHUNK..ci * CHUNK + m).contains(r) {
                    let mut b = vec![0u8; HIDDEN * 4];
                    s.buf("px").read((r - ci * CHUNK) * HIDDEN * 4, &mut b)?;
                    for (j, v) in b.chunks_exact(4).enumerate() {
                        out[i * HIDDEN + j] = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                    }
                }
            }
            c.len += m;
        }
        Ok(out)
    }

    /// One token for each sequence (`toks[i]` appended to `caches[i]`, at its own position): the residual rows, float32
    /// [n, 2048] on the card in the scratch's "dx"
    pub fn step(&self, s: &mut Scratch, caches: &mut [&mut Cache], toks: &[u32]) -> Result<()> {
        let n = caches.len();
        if toks.len() != n || n == 0 || n > 8 {
            return Err(Error("a token a sequence, one to eight".into()));
        }
        if let Some(c) = caches.iter().find(|c| c.len >= c.cap) {
            return Err(Error(format!("the cache is full ({} rows)", c.cap)));
        }
        let g = &self.gpu;
        let x = s.get(g, "dx", n * HIDDEN * 4)?;
        let h = s.get(g, "dh", n * HIDDEN * 4)?;
        let qkv = s.get(g, "dqkv", n * QKV * 4)?;
        let q = s.get(g, "dq", n * QW * 2)?;
        let o = s.get(g, "do", n * QW * 4)?;
        let gu = s.get(g, "dgu", n * 2 * FFN * 4)?;
        let ff = s.get(g, "dff", n * FFN * 4)?;
        let longest = caches.iter().map(|c| c.len + 1).max().unwrap_or(1);
        // SAFETY: a pure function of the length.
        let part_n = unsafe { (self.k.attn_scratch)(longest as i64) } as usize;
        let part = s.get(g, "dpart", part_n.max(1) * 4)?;
        self.embed(s, toks, x)?;
        for (li, l) in self.layers.iter().enumerate() {
            let p = &l.ar;
            self.rms(x, n, &p.norm1, h, Dt::F32)?;
            self.gemv(h, n, p.qkv.w.ptr(), HIDDEN, QKV, qkv, QKV, false)?;
            for (i, c) in caches.iter().enumerate() {
                let (kc, vc) = (c.k[li].ptr(), c.v[li].ptr());
                // SAFETY: row i of qkv; the cache has row c.len (checked above).
                ffi::check(unsafe { (self.k.qkv)(self.g(), at(qkv, i * QKV * 4).cast(), 1, p.qn.fp(), p.kn.fp(), EPS, self.cs.fp(), c.len as i64,
                                                 at(q, i * QW * 2), kc, vc, c.len as i64) }, "q / k / v")?;
                // SAFETY: the cache holds c.len + 1 rows now; part sized for the longest.
                ffi::check(unsafe { (self.k.attn)(self.g(), at(q, i * QW * 2), kc, vc, (c.len + 1) as i64, at(o, i * QW * 4).cast(), part.cast()) },
                           "attention")?;
            }
            self.gemv(o, n, p.o.w.ptr(), QW, HIDDEN, x, HIDDEN, true)?;
            self.rms(x, n, &p.norm2, h, Dt::F32)?;
            self.gemv(h, n, p.gu.w.ptr(), HIDDEN, 2 * FFN, gu, 2 * FFN, false)?;
            self.nsd.swiglu(gu, Dt::F32, n, FFN, ff, Dt::F32)?;
            self.gemv(ff, n, p.down.w.ptr(), FFN, HIDDEN, x, HIDDEN, true)?;
        }
        for c in caches.iter_mut() {
            c.len += 1;
        }
        Ok(())
    }

    /// The logits over the head's rows [r0, r0 + n) of `m` residual rows (float32 [m, 2048] on the card, m <= 8),
    /// float32 [m, n] on the host
    pub fn logits(&self, s: &mut Scratch, x: P, m: usize, r0: usize, n: usize) -> Result<Vec<f32>> {
        let g = &self.gpu;
        let hn = s.get(g, "hn", m * HIDDEN * 4)?;
        let lo = s.get(g, "lo", m * n * 4)?;
        self.rms(x, m, &self.norm, hn, Dt::F32)?;
        self.gemv(hn, m, self.head.rows_at(r0), HIDDEN, n, lo, n, false)?;
        let mut b = vec![0u8; m * n * 4];
        s.buf("lo").read(0, &mut b)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The logits of host residual rows (a prefill's kept rows): as `logits`
    pub fn logits_of(&self, s: &mut Scratch, rows: &[f32], r0: usize, n: usize) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for c in rows.chunks(8 * HIDDEN) {
            let x = s.get(&self.gpu, "hx", c.len() * 4)?;
            s.buf("hx").write(0, crate::f32_bytes(c))?;
            out.extend(self.logits(s, x, c.len() / HIDDEN, r0, n)?);
        }
        Ok(out)
    }

    /// The timestep embedding of the solver's raw t (`_shift_t_value` with shift 1: sigmoid(raw), in float32), as
    /// TimestepEmbedder: [cos, sin](t e^(-ln 10^4 i / 128)) -> linear -> SiLU -> linear
    pub fn time_embedding(&self, raw: f64) -> Vec<f32> {
        let t = 1.0 / (1.0 + (-(raw as f32)).exp());
        let half = FREQ / 2;
        let mut e = vec![0f32; FREQ];
        for i in 0..half {
            let f = (-(10000f64.ln() as f32) * i as f32 / half as f32).exp();
            let a = t * f;
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        let lin = |w: &[f32], b: &[f32], x: &[f32]| -> Vec<f32> {
            let k = x.len();
            w.chunks_exact(k).zip(b).map(|(r, bi)| (r.iter().zip(x).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>() + *bi as f64) as f32).collect()
        };
        let h: Vec<f32> = lin(&self.t1.0, &self.t1.1, &e).into_iter().map(|v| v / (1.0 + (-v).exp())).collect();
        lin(&self.t2.0, &self.t2.1, &h)
    }

    /// The latent position table's rows [0, n) (local positions, clamped to the table) as float32 on the card
    pub fn positions(&self, n: usize) -> Result<DevBuf> {
        let mut v = Vec::with_capacity(n * HIDDEN);
        for r in 0..n {
            let r = r.min(CONTEXT - 1);
            v.extend(self.pos[r * HIDDEN..(r + 1) * HIDDEN].iter().map(|b| f32::from_bits((*b as u32) << 16)));
        }
        DevBuf::from_f32(&self.gpu, &v)
    }

    /// The flow's velocity at state x_t (float32 [frames, 64] on the card) under the chunk's AR rows in `c`: `temb` the
    /// time embedding (float32 [2048] on the card), `pos` the chunk's position rows [frames + 2, 2048]; out float32
    /// [frames, 64] on the card (`nar_velocity` / CachedNAR.velocity)
    pub fn velocity(&self, s: &mut Scratch, c: &Cache, xt: &DevBuf, frames: usize, temb: &DevBuf, pos: &DevBuf, out: M) -> Result<()> {
        let g = &self.gpu;
        let f2 = frames + 2;
        let xin = s.get(g, "nin", f2 * LATENT * 4)?;
        let x = s.get(g, "nx", f2 * HIDDEN * 4)?;
        let hn = s.get(g, "nh", frames * HIDDEN * 4)?;
        // [0, x_t, 0]: LATENT_START and LATENT_END take zeros
        let b = s.buf("nin");
        b.fill(0)?;
        b.copy_within(LATENT * 4, xt, 0, frames * LATENT * 4)?;
        self.nsd.linear(xin, Dt::F32, f2, LATENT, self.v2l_w.ptr(), HIDDEN, self.v2l_b.ptr(), x, Dt::F32)?;
        // SAFETY: x and pos hold f2 x 2048 floats, temb 2048.
        ffi::check(unsafe { (self.k.add_vec_rows)(self.g(), x.cast(), f2 as i64, HIDDEN as i64, temb.fp(), pos.fp()) }, "the latent rows")?;
        self.layers(s, c, x, f2, c.len, true)?;
        self.rms(at(x, HIDDEN * 4), frames, &self.norm, hn, Dt::F32)?;
        self.nsd.linear(hn, Dt::F32, frames, HIDDEN, self.l2v_w.ptr(), LATENT, self.l2v_b.ptr(), out, Dt::F32)
    }

    /// out = a + s b over n floats on the card
    pub fn axpy(&self, out: M, a: P, b: P, n: usize, s: f32) -> Result<()> {
        // SAFETY: the three hold n floats.
        ffi::check(unsafe { (self.k.axpy)(self.g(), out.cast(), a.cast(), b.cast(), n as i64, s) }, "axpy")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_rope_table_holds_upstreams_float32_angles() {
        let cs = super::rope_table();
        assert_eq!(cs.len(), super::CONTEXT * super::HEAD);
        assert_eq!((cs[0], cs[1]), (1.0, 0.0));
        // position 1000, pair 0: inv_freq 1, so the angle is the position
        let row = 1000 * super::HEAD;
        assert_eq!((cs[row], cs[row + 1]), (1000f32.cos(), 1000f32.sin()));
    }
}
