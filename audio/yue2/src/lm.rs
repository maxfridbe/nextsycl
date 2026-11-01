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

/// A matrix [n, k], Q8_0 as stored, or (`scales` Some: the silo's) repacked: quants from its start, scales from that offset
pub struct Mat {
    pub w: DevBuf,
    pub n: usize,
    pub k: usize,
    pub scales: Option<usize>,
}

impl Mat {
    /// rows [r0, ..) of it as stored
    fn rows_at(&self, r0: usize) -> P {
        self.w.ptr().cast::<u8>().wrapping_add(r0 * self.k / 32 * Q8_BLOCK).cast_const().cast()
    }

    /// rows [r0, ..) of it repacked: (quants, scales)
    fn packed_at(&self, r0: usize, sc: usize) -> (P, P) {
        let p = self.w.ptr().cast::<u8>().cast_const();
        (p.wrapping_add(r0 * self.k).cast(), p.wrapping_add(sc + r0 * (self.k / 32) * 2).cast())
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
    /// the products' inputs: half (the default) or bf16 (NS_YUE2_ACT=bf16); oneDNN's product is in that type
    act: Dt,
    /// the silo: its matrices repacked (None: as stored)
    silo: Option<&'static ffi::Silo>,
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
        // SAFETY: the GPU is open; the check reads its properties.
        let silo = ffi::silo()?.filter(|s| unsafe { (s.supported)(gpu.raw(), HIDDEN as i64) } == 1);
        let bytes = std::cell::Cell::new(0u64);
        let up = |b: &[u8]| -> Result<DevBuf> {
            let d = DevBuf::new(gpu, b.len())?;
            d.write(0, b)?;
            bytes.set(bytes.get() + b.len() as u64);
            Ok(d)
        };
        // the matrix repacked by the silo, the stored copy let go; the embedding table stays as stored
        let pack = |m: Mat| -> Result<Mat> {
            let Some(si) = silo else { return Ok(m) };
            // SAFETY: pure functions of the shape; the buffer holds their bytes, m.w the rows as stored.
            let (bytes_n, sc) = unsafe { ((si.bytes)(m.n as i64, m.k as i64), (si.scales_at)(m.n as i64, m.k as i64)) };
            let d = DevBuf::new(gpu, bytes_n)?;
            ffi::check(unsafe { (si.pack)(gpu.raw(), m.w.ptr(), m.n as i64, m.k as i64, d.ptr()) }, "repacking a matrix")?;
            gpu.sync()?;
            bytes.set(bytes.get() + bytes_n as u64 - m.w.len as u64);
            Ok(Mat { w: d, n: m.n, k: m.k, scales: Some(sc) })
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
            let m = Mat { w: up(&all)?, n: rows, k: kk, scales: None };
            pack(m)
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
        // the table as stored: rows are read one at a time (ns_audio_yue2_embed_q8)
        let embed = {
            let t = f.tensor("token_embd.weight").ok_or_else(|| Error("no token_embd.weight".into()))?;
            Mat { w: up(&f.read(t).map_err(ge)?)?, n: t.shape[0] as usize, k: t.shape[1] as usize, scales: None }
        };
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
            silo,
            act: if std::env::var("NS_YUE2_ACT").is_ok_and(|v| v.trim() == "bf16") { Dt::BF16 } else { Dt::F16 },
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

    fn lin(&self, s: &mut Scratch, x: P, dt: Dt, m: usize, w: &Mat, out: M) -> Result<()> {
        match (self.silo, w.scales) {
            (Some(si), Some(sc)) => {
                // the whole matrix as half (the largest, gate | up, is 50 MB), one GEMM
                let h = s.get(&self.gpu, "wh", w.n * w.k * 2)?;
                let (q, sv) = w.packed_at(0, sc);
                // SAFETY: the matrix's quants and scales, h its n x k halves.
                ffi::check(unsafe { (si.half)(self.g(), q, sv, w.n as i64, w.k as i64, h) }, "the matrix as half")?;
                self.nsd.linear(x, dt, m, w.k, h, w.n, none(), out, Dt::F32)
            }
            _ => self.nsd.linear_q(x, dt, m, w.k, w.w.ptr(), Q8_0, w.n, none(), out, Dt::F32, 0),
        }
    }

    /// A decode product against a matrix's rows [r0, r0 + n)
    fn mv(&self, x: P, m: usize, w: &Mat, r0: usize, n: usize, out: M, ldo: usize, acc: bool) -> Result<()> {
        match (self.silo, w.scales) {
            (Some(si), Some(sc)) => {
                let (q, sv) = w.packed_at(r0, sc);
                // SAFETY: x m x k floats, the rows' quants and scales, out m rows of ldo floats.
                ffi::check(unsafe { (si.gemv)(self.g(), x.cast(), m as i64, w.k as i64, q, sv, n as i64, out.cast(), ldo as i64, acc as i32) }, "q8 product")
            }
            _ => self.gemv(x, m, w.rows_at(r0), w.k, n, out, ldo, acc),
        }
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
        let pdt = self.act;
        for (li, l) in self.layers.iter().enumerate() {
            let p = if nar { &l.nar } else { &l.ar };
            let (kc, vc) = (c.k[li].ptr(), c.v[li].ptr());
            for b0 in (0..m).step_by(rb) {
                let bm = rb.min(m - b0);
                self.rms(at(x, b0 * HIDDEN * 4), bm, &p.norm1, h, pdt)?;
                self.lin(s, h, pdt, bm, &p.qkv, qkv)?;
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
                self.lin(s, at(o, b0 * QW * 2), Dt::F16, bm, &p.o, y)?;
                self.add(xb, y, bm)?;
                self.rms(xb, bm, &p.norm2, h, pdt)?;
                self.lin(s, h, pdt, bm, &p.gu, gu)?;
                self.nsd.swiglu(gu, Dt::F32, bm, FFN, h, pdt)?;
                self.lin(s, h, pdt, bm, &p.down, y)?;
                self.add(xb, y, bm)?;
            }
        }
        Ok(())
    }

    /// `layers` for the self-check, the slow arithmetic that makes the yardstick: every product float32 from the Q8_0
    /// blocks (ns_audio_yue2_gemv_q8), and - `exact_attn` - each query row's attention summed in float32 against the
    /// cache (ns_audio_yue2_attn; else the shared kernel `layers` uses)
    fn layers_exact(&self, s: &mut Scratch, c: &Cache, x: M, m: usize, pos0: usize, nar: bool, exact_attn: bool) -> Result<()> {
        if pos0 + m > c.cap {
            return Err(Error(format!("{m} rows at {pos0}: past the cache's {}", c.cap)));
        }
        const R: usize = 8;
        let g = &self.gpu;
        let h = s.get(g, "eh", R * HIDDEN * 4)?;
        let qkv = s.get(g, "eqkv", R * QKV * 4)?;
        let gu = s.get(g, "egu", R * 2 * FFN * 4)?;
        let ff = s.get(g, "eff", R * FFN * 4)?;
        let q = s.get(g, "eq", m * QW * 2)?;
        let o = s.get(g, "eo", m * QW * 2)?;
        let of = s.get(g, "eof", m * QW * 4)?;
        let qg = s.get(g, "eqg", m * QW * 2)?;
        let og = s.get(g, "eog", m * QW * 2)?;
        // SAFETY: a pure function of the length.
        let part_n = unsafe { (self.k.attn_scratch)((pos0 + m) as i64) } as usize;
        let part = s.get(g, "epart", part_n.max(1) * 4)?;
        for (li, l) in self.layers.iter().enumerate() {
            let p = if nar { &l.nar } else { &l.ar };
            let (kc, vc) = (c.k[li].ptr(), c.v[li].ptr());
            for b0 in (0..m).step_by(R) {
                let bm = R.min(m - b0);
                self.rms(at(x, b0 * HIDDEN * 4), bm, &p.norm1, h, Dt::F32)?;
                self.mv(h, bm, &p.qkv, 0, QKV, qkv, QKV, false)?;
                // SAFETY: as in `layers`.
                ffi::check(unsafe { (self.k.qkv)(self.g(), qkv.cast(), bm as i64, p.qn.fp(), p.kn.fp(), EPS, self.cs.fp(), (pos0 + b0) as i64,
                                                 at(q, b0 * QW * 2), kc, vc, (pos0 + b0) as i64) }, "q / k / v")?;
            }
            if exact_attn {
                for r in 0..m {
                    let n = if nar { pos0 + m } else { pos0 + r + 1 };
                    // SAFETY: q row r, the cache's n rows, of row r, part sized for the longest.
                    ffi::check(unsafe { (self.k.attn)(self.g(), at(q, r * QW * 2), kc, vc, n as i64, at(of, r * QW * 4).cast(), part.cast()) }, "attention")?;
                }
            } else {
                if nar {
                    // SAFETY: q, qg, og, o hold m rows of 2048 half.
                    ffi::check(unsafe { (self.k.heads)(self.g(), q, qg, m as i64, 1) }, "the query heads")?;
                    self.nsd.attention_qk(qg, 2 * m, KW, kc, vc, pos0 + m, KW, Dt::F16, KV_HEADS, HEAD, og, Dt::F16)?;
                    // SAFETY: as above.
                    ffi::check(unsafe { (self.k.heads)(self.g(), og, o, m as i64, 0) }, "the query heads")?;
                } else {
                    let q0 = q.cast::<u8>().wrapping_sub(pos0 * QW * 2).cast_const().cast();
                    self.nsd.attention_causal_rows(q0, kc, vc, Dt::F16, pos0, m, HEADS, KV_HEADS, HEAD, QW, KW, o)?;
                }
                // SAFETY: o m rows of half, of as floats.
                ffi::check(unsafe { (self.k.to_f32)(self.g(), o, (m * QW) as i64, of.cast()) }, "float32")?;
            }
            for b0 in (0..m).step_by(R) {
                let bm = R.min(m - b0);
                let xb = at(x, b0 * HIDDEN * 4);
                self.mv(at(of, b0 * QW * 4), bm, &p.o, 0, HIDDEN, xb, HIDDEN, true)?;
                self.rms(xb, bm, &p.norm2, h, Dt::F32)?;
                self.mv(h, bm, &p.gu, 0, 2 * FFN, gu, 2 * FFN, false)?;
                self.nsd.swiglu(gu, Dt::F32, bm, FFN, ff, Dt::F32)?;
                self.mv(ff, bm, &p.down, 0, HIDDEN, xb, HIDDEN, true)?;
            }
        }
        Ok(())
    }

    /// The prompt pass in the slow arithmetic: as `prefill`, a chunk of one block of rows
    pub fn prefill_exact(&self, s: &mut Scratch, c: &mut Cache, ids: &[u32], keep: &[usize]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; keep.len() * HIDDEN];
        for (ci, chunk) in ids.chunks(CHUNK).enumerate() {
            let m = chunk.len();
            let x = s.get(&self.gpu, "px", m * HIDDEN * 4)?;
            self.embed(s, chunk, x)?;
            self.layers_exact(s, c, x, m, c.len, false, true)?;
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

    /// Each decode kernel alone, `n` runs each with the queue drained, for the self-check: (name, microseconds a run)
    pub fn decode_probe(&self, s: &mut Scratch, c: &Cache, n: usize, log: &mut dyn FnMut(String)) -> Result<()> {
        let g = self.gpu.clone();
        let p = &self.layers[0].ar;
        let x = s.get(&g, "bx", 2 * HIDDEN * 4)?;
        s.buf("bx").fill(0)?;
        let qkv = s.get(&g, "bqkv", 2 * QKV * 4)?;
        let gu = s.get(&g, "bgu", 2 * 2 * FFN * 4)?;
        let ff = s.get(&g, "bff", 2 * FFN * 4)?;
        let oo = s.get(&g, "boo", 2 * QW * 4)?;
        let q = s.get(&g, "bq", 2 * QW * 2)?;
        let keys = c.len;
        // SAFETY: a pure function of the length.
        let part_n = unsafe { (self.k.attn_scratch)(keys as i64) } as usize;
        let part = s.get(&g, "bpart", part_n.max(1) * 4)?;
        let mut t = |name: &str, f: &mut dyn FnMut() -> Result<()>| -> Result<()> {
            g.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f()?;
                g.sync()?;
            }
            log(format!("decode kernel {name:<28} {:.0} us", t0.elapsed().as_secs_f64() * 1e6 / n as f64));
            Ok(())
        };
        t("an empty sync", &mut || Ok(()))?;
        t("rms norm 2 rows", &mut || self.rms(x, 2, &p.norm1, x, Dt::F32))?;
        t("q8 qkv 2 rows", &mut || self.mv(x, 2, &p.qkv, 0, QKV, qkv, QKV, false))?;
        t("q8 gate+up 2 rows", &mut || self.mv(x, 2, &p.gu, 0, 2 * FFN, gu, 2 * FFN, false))?;
        t("q8 gate+up 1 row", &mut || self.mv(x, 1, &p.gu, 0, 2 * FFN, gu, 2 * FFN, false))?;
        t("q8 down 2 rows", &mut || self.mv(ff, 2, &p.down, 0, HIDDEN, x, HIDDEN, false))?;
        t("swiglu 2 rows", &mut || self.nsd.swiglu(gu, Dt::F32, 2, FFN, ff, Dt::F32))?;
        // SAFETY: as in `step`; one row at the cache's last position, its row rewritten as it was (a probe only).
        t("attention 1 row", &mut || ffi::check(unsafe { (self.k.attn)(self.g(), q, c.k[0].ptr(), c.v[0].ptr(), keys as i64, oo.cast(), part.cast()) }, "attention"))?;
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
            self.mv(h, n, &p.qkv, 0, QKV, qkv, QKV, false)?;
            for (i, c) in caches.iter().enumerate() {
                let (kc, vc) = (c.k[li].ptr(), c.v[li].ptr());
                // SAFETY: row i of qkv; the cache has row c.len (checked above).
                ffi::check(unsafe { (self.k.qkv)(self.g(), at(qkv, i * QKV * 4).cast(), 1, p.qn.fp(), p.kn.fp(), EPS, self.cs.fp(), c.len as i64,
                                                 at(q, i * QW * 2), kc, vc, c.len as i64) }, "q / k / v")?;
                // SAFETY: the cache holds c.len + 1 rows now; part sized for the longest.
                ffi::check(unsafe { (self.k.attn)(self.g(), at(q, i * QW * 2), kc, vc, (c.len + 1) as i64, at(o, i * QW * 4).cast(), part.cast()) },
                           "attention")?;
            }
            self.mv(o, n, &p.o, 0, HIDDEN, x, HIDDEN, true)?;
            self.rms(x, n, &p.norm2, h, Dt::F32)?;
            self.mv(h, n, &p.gu, 0, 2 * FFN, gu, 2 * FFN, false)?;
            self.nsd.swiglu(gu, Dt::F32, n, FFN, ff, Dt::F32)?;
            self.mv(ff, n, &p.down, 0, HIDDEN, x, HIDDEN, true)?;
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
        self.mv(hn, m, &self.head, r0, n, lo, n, false)?;
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
        self.velocity_as(s, c, xt, frames, temb, pos, out, 0)
    }

    /// `velocity` in an arithmetic: 0 the engine's, 1 the products float32 and the shared attention, 2 all float32 (the self-check)
    pub fn velocity_as(&self, s: &mut Scratch, c: &Cache, xt: &DevBuf, frames: usize, temb: &DevBuf, pos: &DevBuf, out: M, how: u8) -> Result<()> {
        let g = &self.gpu;
        let f2 = frames + 2;
        let xin = s.get(g, "nin", f2 * LATENT * 4)?;
        let x = s.get(g, "nx", f2 * HIDDEN * 4)?;
        let hn = s.get(g, "nh", frames * HIDDEN * 4)?;
        // [0, x_t, 0]: LATENT_START and LATENT_END take zeros
        let b = s.buf("nin");
        b.fill(0)?;
        b.copy_within(LATENT * 4, xt, 0, frames * LATENT * 4)?;
        // SAFETY: xin f2 x 64 floats, the head [2048, 64] and its bias, x f2 x 2048.
        ffi::check(unsafe { (self.k.linear_f32)(self.g(), xin.cast(), f2 as i64, LATENT as i64, self.v2l_w.fp(), HIDDEN as i64, self.v2l_b.fp(), x.cast()) }, "vae2llm")?;
        // SAFETY: x and pos hold f2 x 2048 floats, temb 2048.
        ffi::check(unsafe { (self.k.add_vec_rows)(self.g(), x.cast(), f2 as i64, HIDDEN as i64, temb.fp(), pos.fp()) }, "the latent rows")?;
        match how {
            0 => self.layers(s, c, x, f2, c.len, true)?,
            _ => self.layers_exact(s, c, x, f2, c.len, true, how == 2)?,
        }
        self.rms(at(x, HIDDEN * 4), frames, &self.norm, hn, Dt::F32)?;
        // SAFETY: hn frames x 2048 floats, the head [64, 2048] and its bias, out frames x 64.
        ffi::check(unsafe { (self.k.linear_f32)(self.g(), hn.cast(), frames as i64, HIDDEN as i64, self.l2v_w.fp(), LATENT as i64, self.l2v_b.fp(), out.cast()) }, "llm2vae")
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
