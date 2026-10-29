//! MiniMax-M3.1-Flash (`minimax-m3`, llama.cpp's src/models/minimax-m3.cpp): 60 pre-norm layers. Attention is GQA
//! (64 query heads, 4 key/value heads of 128), q and k RMS-normed per head, NEOX RoPE on the first 64 values; on
//! every layer but the first three (llama.cpp's reading: docs/ports/minimax-m31.md) MiniMax Sparse Attention picks,
//! per key/value group, the 16 best blocks of 128 positions by an indexer of its own (4 heads of 128, normed and
//! rotated like q and k) and the block holding the query, and attends only those. The feed-forward is a dense
//! swiglu-oai block (layers 0-2) or 4 of 128 experts (sigmoid gate, a bias to choose with, weights renormalized and
//! x2) plus a shared expert, all swiglu-oai (alpha 1.702, limit 7).
//!
//! This is the exact bring-up path: float32 activations, every matrix dequantized on the GPU into a scratch (fp32,
//! or fp16 for the experts) and multiplied by oneMKL; the dense weights stay in their GGUF blocks on the GPU, the
//! experts are read from the file for each pass (the routed ones only).
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  h = x + attn(attn_norm(x));  x = h + ffn(ffn_norm(h))
//!   logits = output(output_norm(x))
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Gpu};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_llm::{Error, GpuInfo, Result, Sampler, Tap};

use crate::ffi;
use crate::model::Geometry;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

/// Tokens a prompt pass takes (NS_MM31_CHUNK)
pub fn prefill_chunk() -> usize {
    std::env::var("NS_MM31_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(512)
}

/// NS_MM31_SPARSE_ALL=1: the first three layers take the indexer too (the model's config says every layer is sparse;
/// llama.cpp's graph leaves them dense - the reference this engine is checked against)
fn sparse_all() -> bool {
    std::env::var("NS_MM31_SPARSE_ALL").is_ok_and(|v| v == "1")
}

/// A matrix [rows (outputs), cols (inputs)] in its stored blocks (or float32), on the GPU
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

/// pinned host memory is taken in slabs of this many bytes
const SLAB: usize = 1 << 30;

/// An expert's [gate | up | down] bytes where a kernel reads them: VRAM, or pinned host memory (read in place over PCIe)
struct Slot {
    buf: DevBuf,
}

/// The experts kept between passes, first come first kept while the budgets last
struct ExpertCache {
    map: HashMap<(usize, usize), Arc<Slot>>,
    vram_left: usize,
    host_left: usize,
    slabs: Vec<(nextsycl_core::HostBuf, usize)>,
}

/// One layer's routed experts in the file: [experts][rows][cols] in `ty`
struct Experts {
    name: String,
    ty: GType,
    rows: usize,
    cols: usize,
    /// bytes an expert takes
    stride: usize,
}

impl Experts {
    /// the room a matrix takes in a slot (256-byte aligned)
    fn slot_bytes(&self) -> usize {
        self.stride.next_multiple_of(256)
    }
}

struct Dense {
    gate: Mat,
    up: Mat,
    down: Mat,
}

struct Moe {
    router: Mat,
    bias: Vec<f32>,
    gate: Experts,
    up: Experts,
    down: Experts,
    shared: Dense,
}

enum Ffn {
    Dense(Dense),
    Moe(Box<Moe>),
}

struct Indexer {
    q: Mat,
    k: Mat,
    qn: DevBuf,
    kn: DevBuf,
}

struct Layer {
    attn_norm: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    qn: DevBuf,
    kn: DevBuf,
    idx: Option<Indexer>,
    ffn_norm: DevBuf,
    ffn: Ffn,
}

/// A conversation: per layer the key / value cache ([position][4][128] fp16) and, on the sparse layers, the indexer's
/// keys ([position][128] fp16)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    ik: Vec<Option<DevBuf>>,
    last_rows: usize,
}

pub struct Checkpoint {
    pub pos: usize,
    pub bytes: usize,
    /// per layer: k, v, ik (empty where none), the first `pos` positions
    parts: Vec<Vec<u8>>,
}

impl Checkpoint {
    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        w.write_all(b"MM31CK1\0")?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&(self.parts.len() as u64).to_le_bytes())?;
        for p in &self.parts {
            w.write_all(&(p.len() as u64).to_le_bytes())?;
            w.write_all(p)?;
        }
        Ok(())
    }
    pub fn read_from(r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"MM31CK1\0" {
            return Err(bad("not a minimaxm3 checkpoint"));
        }
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let pos = u64::from_le_bytes(b8) as usize;
        r.read_exact(&mut b8)?;
        let n = u64::from_le_bytes(b8) as usize;
        if n > 1024 {
            return Err(bad("too many parts"));
        }
        let mut parts = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            r.read_exact(&mut b8)?;
            let len = u64::from_le_bytes(b8) as usize;
            let mut p = vec![0u8; len];
            r.read_exact(&mut p)?;
            bytes += len;
            parts.push(p);
        }
        Ok(Checkpoint { pos, bytes, parts })
    }
}

/// Generation from a prompt: either the logits to draw the first token from, or a token committed but not fed
pub struct Decoder {
    logits: Option<Vec<f32>>,
    next: Option<u32>,
}

impl Decoder {
    pub fn new(logits: Vec<f32>) -> Decoder {
        Decoder { logits: Some(logits), next: None }
    }
    pub fn after(next: u32) -> Decoder {
        Decoder { logits: None, next: Some(next) }
    }
    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
}

pub struct MiniMax<'g> {
    file: &'g Gguf,
    gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    pub g: Geometry,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    embd: ConstTensor,
    head: Mat,
    arena: Arena,
    pub load_seconds: f64,
    pub load_bytes: u64,
    work: std::sync::Mutex<()>,
    cache: std::sync::Mutex<ExpertCache>,
    /// seconds in the routed experts' reads, and in all of their products
    prof: std::sync::Mutex<[f64; 4]>,
    /// NS_PROFILE=1: per layer, seconds by category (`CATS`), each timed to the GPU's completion (a sync at every lap)
    pf: bool,
    lay: std::sync::Mutex<Vec<[f64; NCAT]>>,
    /// passes of one token, and the seconds they took (profile)
    pf_tokens: std::sync::Mutex<(usize, f64)>,
}

const NCAT: usize = 8;
const CATS: [&str; NCAT] = ["attn proj", "indexer", "attend", "route", "expert read", "expert products", "shared/dense", "other"];

/// A tensor read from the file a row at a time (the token embedding)
struct ConstTensor {
    name: String,
    ty: GType,
    cols: usize,
}

/// MemAvailable, bytes
fn mem_available() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()))
        .map_or(0, |kb| kb * 1024)
}

fn mat(f: &Gguf, gpu: &Arc<Gpu>, name: &str, bytes: &mut u64) -> Result<Mat> {
    let t = f.tensor(name).ok_or_else(|| e(format!("{name}: missing")))?;
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    let raw = f.read(t).map_err(|x| e(x.0))?;
    let buf = DevBuf::new(gpu, raw.len())?;
    buf.write(0, &raw)?;
    *bytes += raw.len() as u64;
    Ok(Mat { buf, ty: t.ty, rows, cols })
}

fn vecf(f: &Gguf, gpu: &Arc<Gpu>, name: &str, n: usize) -> Result<DevBuf> {
    let t = f.tensor(name).ok_or_else(|| e(format!("{name}: missing")))?;
    if t.ty != GType::F32 || t.elements() as usize != n {
        return Err(e(format!("{name}: float32 [{n}] expected")));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    let b = DevBuf::new(gpu, raw.len())?;
    b.write(0, &raw)?;
    Ok(b)
}

fn experts(f: &Gguf, name: String, n_expert: usize) -> Result<Experts> {
    let t = f.tensor(&name).ok_or_else(|| e(format!("{name}: missing")))?;
    // the reader's shape is outermost first: [experts, rows, cols]
    let (ne, rows, cols) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
    if ne != n_expert {
        return Err(e(format!("{name}: {ne} experts, {n_expert} expected")));
    }
    let stride = t.ty.bytes((rows * cols) as u64).ok_or_else(|| e(format!("{name}: row size")))? as usize;
    Ok(Experts { name, ty: t.ty, rows, cols, stride })
}

impl<'g> MiniMax<'g> {
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], kv: (usize, usize), log: &mut dyn FnMut(String)) -> Result<MiniMax<'g>> {
        let t0 = Instant::now();
        let g = Geometry::from_gguf(f).map_err(|x| e(x.0))?;
        g.check(f).map_err(|x| e(x.0))?;
        let gpu = gpus.first().ok_or_else(|| e("no GPU"))?.clone();
        let k = ffi::api()?;
        let (total, free) = gpu.memory()?;
        log(format!("{}: {:.1} GiB, {:.1} GiB free", gpu.name, total as f64 / (1u64 << 30) as f64, free.unwrap_or(total) as f64 / (1u64 << 30) as f64));
        let mut bytes = 0u64;
        let eps_dim = g.n_embd as usize;
        let mut layers = Vec::with_capacity(g.n_layer as usize);
        for l in 0..g.n_layer {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let sparse = l >= g.n_dense || sparse_all();
            let idx = if sparse {
                let kl = g.indexer.key_len as usize;
                Some(Indexer {
                    q: mat(f, &gpu, &n("indexer.q_proj"), &mut bytes)?,
                    k: mat(f, &gpu, &n("indexer.k_proj"), &mut bytes)?,
                    qn: vecf(f, &gpu, &n("indexer.q_norm"), kl)?,
                    kn: vecf(f, &gpu, &n("indexer.k_norm"), kl)?,
                })
            } else {
                None
            };
            let ffn = if l < g.n_dense {
                Ffn::Dense(Dense {
                    gate: mat(f, &gpu, &n("ffn_gate"), &mut bytes)?,
                    up: mat(f, &gpu, &n("ffn_up"), &mut bytes)?,
                    down: mat(f, &gpu, &n("ffn_down"), &mut bytes)?,
                })
            } else {
                let bt = f.tensor(&format!("blk.{l}.exp_probs_b.bias")).ok_or_else(|| e("exp_probs_b missing"))?;
                let braw = f.read(bt).map_err(|x| e(x.0))?;
                let bias: Vec<f32> = braw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let ne = g.n_expert as usize;
                Ffn::Moe(Box::new(Moe {
                    router: mat(f, &gpu, &n("ffn_gate_inp"), &mut bytes)?,
                    bias,
                    gate: experts(f, n("ffn_gate_exps"), ne)?,
                    up: experts(f, n("ffn_up_exps"), ne)?,
                    down: experts(f, n("ffn_down_exps"), ne)?,
                    shared: Dense {
                        gate: mat(f, &gpu, &n("ffn_gate_shexp"), &mut bytes)?,
                        up: mat(f, &gpu, &n("ffn_up_shexp"), &mut bytes)?,
                        down: mat(f, &gpu, &n("ffn_down_shexp"), &mut bytes)?,
                    },
                }))
            };
            layers.push(Layer {
                attn_norm: vecf(f, &gpu, &n("attn_norm"), eps_dim)?,
                q: mat(f, &gpu, &n("attn_q"), &mut bytes)?,
                k: mat(f, &gpu, &n("attn_k"), &mut bytes)?,
                v: mat(f, &gpu, &n("attn_v"), &mut bytes)?,
                o: mat(f, &gpu, &n("attn_output"), &mut bytes)?,
                qn: vecf(f, &gpu, &n("attn_q_norm"), g.head_dim as usize)?,
                kn: vecf(f, &gpu, &n("attn_k_norm"), g.head_dim as usize)?,
                idx,
                ffn_norm: vecf(f, &gpu, &n("ffn_norm"), eps_dim)?,
                ffn,
            });
            if l % 10 == 9 {
                log(format!("layers 0-{l} loaded ({:.1} GiB on the GPU)", bytes as f64 / (1u64 << 30) as f64));
            }
        }
        let out_norm = vecf(f, &gpu, "output_norm.weight", eps_dim)?;
        let head = mat(f, &gpu, "output.weight", &mut bytes)?;
        let et = f.tensor("token_embd.weight").ok_or_else(|| e("token_embd.weight missing"))?;
        let embd = ConstTensor { name: "token_embd.weight".into(), ty: et.ty, cols: g.n_embd as usize };
        let arena = Arena::new(&gpu, 1536 << 20)?;
        // the experts kept between passes: VRAM left after the sessions' caches and a spare, then pinned host memory
        let gib = |n: usize| n << 30;
        let kv_tokens = if kv.0 > 0 { kv.0 } else { 32768 };
        let n_sparse = layers.iter().filter(|l| l.idx.is_some()).count();
        let per_tok = layers.len() * 2 * (g.n_kv_head * g.head_dim) as usize * 2 + n_sparse * g.indexer.key_len as usize * 2;
        let (total, free) = gpu.memory()?;
        let reserve = kv_tokens * per_tok + gib(1) + (3 << 29);
        let vram_left = match std::env::var("NS_MM31_VRAM_GIB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => (v * (1u64 << 30) as f64) as usize,
            None => (free.unwrap_or(total) as usize).saturating_sub(reserve),
        };
        let host_left = match std::env::var("NS_MM31_PIN_GIB").ok().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => (v * (1u64 << 30) as f64) as usize,
            None => (mem_available() / 3).min(gib(24)),
        };
        log(format!("experts kept: up to {:.1} GiB in VRAM, {:.1} GiB pinned in host memory; the rest read from the file", vram_left as f64 / (1u64 << 30) as f64,
                    host_left as f64 / (1u64 << 30) as f64));
        let cache = ExpertCache { map: HashMap::new(), vram_left, host_left, slabs: Vec::new() };
        Ok(MiniMax { file: f, gpu, k, g, layers, out_norm, embd, head, arena, load_seconds: t0.elapsed().as_secs_f64(), load_bytes: bytes,
                     work: std::sync::Mutex::new(()), cache: std::sync::Mutex::new(cache), prof: std::sync::Mutex::new([0.0; 4]),
                     pf: std::env::var("NS_PROFILE").is_ok_and(|v| v == "1"), lay: std::sync::Mutex::new(vec![[0.0; NCAT]; 64]), pf_tokens: std::sync::Mutex::new((0, 0.0)) })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo { index: self.gpu.index, name: self.gpu.name.clone(), pci: None, total, free, layers: (0, self.g.n_layer), expert_slots: 0, host_slots: 0 }]
    }

    pub fn profile_lines(&self) -> Vec<String> {
        let p = self.prof.lock().unwrap();
        let mut v = vec![format!("[experts: {:.2} s reading the file, {:.2} s in their products]", p[0], p[1])];
        if self.pf {
            let lay = self.lay.lock().unwrap();
            let (n, secs) = *self.pf_tokens.lock().unwrap();
            v.push(format!("[profile: {n} one-token passes, {secs:.2} s; seconds by layer and category, each lap waits for the GPU]"));
            let mut head = String::from("layer");
            for c in CATS {
                head += &format!(",{c}");
            }
            v.push(head);
            let mut tot = [0.0; NCAT];
            for (l, row) in lay.iter().enumerate().take(self.layers.len()) {
                let mut line = format!("{l}");
                for (i, x) in row.iter().enumerate() {
                    line += &format!(",{x:.4}");
                    tot[i] += x;
                }
                v.push(line);
            }
            let sum: f64 = tot.iter().sum();
            let mut line = String::from("total");
            for x in tot {
                line += &format!(",{x:.3} ({:.0}%)", 100.0 * x / sum.max(1e-9));
            }
            v.push(line);
        }
        v
    }

    /// Profile lap: waits for the GPU and adds the time since `t` to layer `l`'s category `cat` (no-op unless NS_PROFILE=1)
    fn lap(&self, l: usize, cat: usize, t: &mut Instant) {
        if self.pf {
            let _ = self.gpu.sync();
            self.lay.lock().unwrap()[l][cat] += t.elapsed().as_secs_f64();
            *t = Instant::now();
        }
    }

    // ---- kernels

    fn raw(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }

    fn rms(&self, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: x, y hold rows * c floats and w c floats (the callers' sizes).
        let rc = unsafe { (self.k.rms_norm)(self.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.rms_eps as f32) };
        ffi::check(rc, "rms_norm")
    }

    fn add(&self, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: both hold n floats.
        ffi::check(unsafe { (self.k.add)(self.raw(), y.fp(), x.fp(), n as i64) }, "add")
    }

    /// y [t, rows] = x [t, cols] . W^T: the matrix dequantized to float32 a chunk of rows at a time
    fn matmul(&self, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        if x.floats() < t * w.cols || y.floats() < t * w.rows {
            return Err(e(format!("matmul {}x{}: buffers of {} and {} floats for {t} rows", w.rows, w.cols, x.floats(), y.floats())));
        }
        if w.ty == GType::F32 {
            // SAFETY: sizes checked above; w holds rows * cols floats (loaded from a shaped tensor).
            let rc = unsafe { (self.k.gemm)(self.raw(), t as i64, w.rows as i64, w.cols as i64, x.fp(), w.cols as i64, w.buf.fp(), y.fp(), w.rows as i64, 0) };
            return ffi::check(rc, "gemm");
        }
        if t <= 8 && self.mmvq_ok(w.ty, w.cols) {
            // decode width: the products straight from the stored blocks
            let m = self.arena.mark();
            let xq = self.quantize(x, t, w.cols)?;
            self.mmvq_at(w.ty, w.buf.ptr().cast::<u8>().cast_const(), &xq, w.cols, w.rows, t, y)?;
            self.arena.rewind(m);
            return Ok(());
        }
        let row_bytes = w.ty.bytes(w.cols as u64).ok_or_else(|| e("row bytes"))? as usize;
        let chunk = (24 << 20) / w.cols;
        let m = self.arena.mark();
        let tmp = self.arena.f32(chunk.min(w.rows) * w.cols)?;
        let mut r0 = 0;
        while r0 < w.rows {
            let nr = chunk.min(w.rows - r0);
            // SAFETY: the byte range [r0 * row_bytes, (r0 + nr) * row_bytes) is inside w.buf; tmp holds nr * cols floats;
            // y has the t * rows floats checked above.
            unsafe {
                let src = w.buf.ptr().cast::<u8>().add(r0 * row_bytes).cast();
                ffi::check((self.k.dequant)(self.raw(), w.ty.code() as i32, src, nr * w.cols, tmp.fp()), "dequant")?;
                ffi::check((self.k.gemm)(self.raw(), t as i64, nr as i64, w.cols as i64, x.fp(), w.cols as i64, tmp.fp(), y.fp().add(r0), w.rows as i64, 0), "gemm")?;
            }
            r0 += nr;
        }
        self.arena.rewind(m);
        Ok(())
    }

    fn embed(&self, tokens: &[u32]) -> Result<DevBuf> {
        let c = &self.embd;
        let row = c.ty.bytes(c.cols as u64).ok_or_else(|| e("embedding row"))? as usize;
        let t = self.file.tensor(&c.name).ok_or_else(|| e("token_embd missing"))?;
        let mut host = vec![0u8; tokens.len() * row];
        for (i, &tok) in tokens.iter().enumerate() {
            if tok as u64 >= self.g.n_vocab {
                return Err(e(format!("token {tok} outside the vocabulary of {}", self.g.n_vocab)));
            }
            self.file.read_into(t, tok as u64 * row as u64, &mut host[i * row..(i + 1) * row]).map_err(|x| e(x.0))?;
        }
        let raw = DevBuf::new(&self.gpu, host.len())?;
        raw.write(0, &host)?;
        let x = DevBuf::f32(&self.gpu, tokens.len() * c.cols)?;
        // SAFETY: raw holds tokens * row bytes = tokens * cols values of the type; x tokens * cols floats.
        let rc = unsafe { (self.k.dequant)(self.raw(), c.ty.code() as i32, raw.ptr().cast_const(), tokens.len() * c.cols, x.fp()) };
        ffi::check(rc, "dequant (embedding)")?;
        self.gpu.sync()?;
        Ok(x)
    }

    // ---- the layer

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: usize, ly: &Layer, s: &Session, xn: &DevBuf, t: usize, p0: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.g;
        let (h, kv, hd) = (g.n_head as usize, g.n_kv_head as usize, g.head_dim as usize);
        let a = &self.arena;
        let mut tm = Instant::now();
        let q = a.f32(t * h * hd)?;
        let k = a.f32(t * kv * hd)?;
        let v = a.f32(t * kv * hd)?;
        self.matmul(&ly.q, xn, t, &q)?;
        self.matmul(&ly.k, xn, t, &k)?;
        self.matmul(&ly.v, xn, t, &v)?;
        self.rms(&q, &ly.qn, &q, t * h, hd)?;
        self.rms(&k, &ly.kn, &k, t * kv, hd)?;
        tap(&format!("Qcur_normed-{l}"), &q)?;
        tap(&format!("Kcur_normed-{l}"), &k)?;
        let n_rot = g.n_rot as i64;
        // SAFETY: q and k hold the rows the extents name.
        unsafe {
            ffi::check((self.k.rope)(self.raw(), q.fp(), t as i64, h as i64, (h * hd) as i64, hd as i64, n_rot, p0 as i64, g.rope_base as f32), "rope q")?;
            ffi::check((self.k.rope)(self.raw(), k.fp(), t as i64, kv as i64, (kv * hd) as i64, hd as i64, n_rot, p0 as i64, g.rope_base as f32), "rope k")?;
        }
        tap(&format!("Qcur-{l}"), &q)?;
        tap(&format!("Kcur-{l}"), &k)?;
        // the caches, fp16
        let (kc, vc) = (&s.k[l], &s.v[l]);
        let off = p0 * kv * hd;
        kc.bounds(off * 2, t * kv * hd * 2)?;
        // SAFETY: the destination ranges are inside the caches (checked above for k; v has the same extent).
        unsafe {
            ffi::check((self.k.to_f16)(self.raw(), k.fp(), kc.ptr().cast::<u16>().add(off), (t * kv * hd) as i64), "cache k")?;
            ffi::check((self.k.to_f16)(self.raw(), v.fp(), vc.ptr().cast::<u16>().add(off), (t * kv * hd) as i64), "cache v")?;
        }
        self.lap(l, 0, &mut tm);
        let out = a.f32(t * h * hd)?;
        let n_pos = p0 + t;
        let scale = 1.0 / (hd as f32).sqrt();
        match (&ly.idx, &s.ik[l]) {
            (Some(ix), Some(ic)) => {
                let (ih, il) = (g.indexer.n_head as usize, g.indexer.key_len as usize);
                let iq = a.f32(t * ih * il)?;
                let ik = a.f32(t * il)?;
                self.matmul(&ix.q, xn, t, &iq)?;
                self.matmul(&ix.k, xn, t, &ik)?;
                self.rms(&iq, &ix.qn, &iq, t * ih, il)?;
                self.rms(&ik, &ix.kn, &ik, t, il)?;
                // SAFETY: iq, ik hold the rows named; the cache range is checked just before the conversion.
                unsafe {
                    ffi::check((self.k.rope)(self.raw(), iq.fp(), t as i64, ih as i64, (ih * il) as i64, il as i64, n_rot, p0 as i64, g.rope_base as f32), "rope iq")?;
                    ffi::check((self.k.rope)(self.raw(), ik.fp(), t as i64, 1, il as i64, il as i64, n_rot, p0 as i64, g.rope_base as f32), "rope ik")?;
                    ic.bounds(p0 * il * 2, t * il * 2)?;
                    ffi::check((self.k.to_f16)(self.raw(), ik.fp(), ic.ptr().cast::<u16>().add(p0 * il), (t * il) as i64), "cache ik")?;
                }
                self.lap(l, 1, &mut tm);
                let blk = g.indexer.block_size as usize;
                let nblk = n_pos.div_ceil(blk);
                let kk = (g.indexer.top_k as usize).min(nblk);
                let bs = a.f32(t * ih * nblk)?;
                let sel = a.bytes(t * ih * kk * 4)?;
                // SAFETY: bs holds t * ih * nblk floats, sel t * ih * kk ints, ic the keys of n_pos positions.
                unsafe {
                    ffi::check((self.k.idx_scores)(self.raw(), iq.fp(), ic.ptr().cast::<u16>().cast_const(), bs.fp(), t as i64, ih as i64, nblk as i64, p0 as i64,
                                                   g.indexer.local_blocks as i64), "idx_scores")?;
                    tap(&format!("msa_bs-{l}"), &bs)?;
                    ffi::check((self.k.idx_topk)(self.raw(), bs.fp(), sel.ptr().cast::<i32>(), t as i64, ih as i64, nblk as i64, kk as i64), "idx_topk")?;
                    self.lap(l, 1, &mut tm);
                    ffi::check((self.k.attend)(self.raw(), q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), sel.ptr().cast::<i32>().cast_const(),
                                               kk as i64, out.fp(), t as i64, h as i64, kv as i64, p0 as i64, scale), "attend")?;
                }
            }
            _ => {
                // SAFETY: caches hold n_pos positions; out t * h * hd floats.
                unsafe {
                    ffi::check((self.k.attend)(self.raw(), q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), std::ptr::null(), 0, out.fp(),
                                               t as i64, h as i64, kv as i64, p0 as i64, scale), "attend")?;
                }
            }
        }
        self.lap(l, 2, &mut tm);
        tap(&format!("kqv_out-{l}"), &out)?;
        let o = a.f32(t * g.n_embd as usize)?;
        self.matmul(&ly.o, &out, t, &o)?;
        self.lap(l, 0, &mut tm);
        Ok(o)
    }

    fn dense(&self, d: &Dense, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let a = &self.arena;
        let f = d.gate.rows;
        let gt = a.f32(t * f)?;
        let up = a.f32(t * f)?;
        self.matmul(&d.gate, x, t, &gt)?;
        self.matmul(&d.up, x, t, &up)?;
        // SAFETY: gt, up hold t * f floats.
        ffi::check(unsafe { (self.k.swiglu_oai)(self.raw(), gt.fp(), up.fp(), gt.fp(), (t * f) as i64, 1.702, 7.0) }, "swiglu_oai")?;
        let out = a.f32(t * d.down.rows)?;
        self.matmul(&d.down, &gt, t, &out)?;
        Ok(out)
    }

    fn moe(&self, l: usize, m: &Moe, x: &DevBuf, t: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.g;
        let (ne, used, c) = (g.n_expert as usize, g.n_expert_used as usize, g.n_embd as usize);
        let a = &self.arena;
        let mut tm = Instant::now();
        let logits = a.f32(t * ne)?;
        self.matmul(&m.router, x, t, &logits)?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let lg = logits.to_f32()?;
        // routing on the host: sigmoid probabilities, the bias added to choose, the chosen probabilities renormalized and scaled
        let mut per: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
        let mut wts = Vec::with_capacity(t * used);
        for tok in 0..t {
            let probs: Vec<f32> = lg[tok * ne..(tok + 1) * ne].iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&i, &j| (probs[j] + m.bias[j]).total_cmp(&(probs[i] + m.bias[i])).then(i.cmp(&j)));
            let pick = &order[..used];
            let mut sum: f32 = pick.iter().map(|&i| probs[i]).sum();
            if g.expert_norm {
                sum = sum.max(6.103_515_6e-5);
            } else {
                sum = 1.0;
            }
            for &i in pick {
                let w = probs[i] / sum * g.expert_scale;
                wts.push(w);
                per[i].push((tok as i32, w));
            }
        }
        self.lap(l, 3, &mut tm);
        let wbuf = DevBuf::from_f32(&self.gpu, &wts)?;
        tap(&format!("ffn_moe_weights_scaled-{l}"), &wbuf)?;
        let y = a.f32(t * c)?;
        y.fill(0)?;
        let tr = Instant::now();
        let mut read_s = 0.0;
        let fe = m.gate.rows;
        let off_up = m.gate.slot_bytes();
        let off_down = off_up + m.up.slot_bytes();
        let total = off_down + m.down.slot_bytes();
        for (ex, list) in per.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let n = list.len();
            let mk = a.mark();
            let t0 = Instant::now();
            let slot = self.expert_slot(l, ex, m, total, off_up, off_down)?;
            read_s += t0.elapsed().as_secs_f64();
            let base = slot.buf.ptr().cast::<u8>().cast_const();
            let idx: Vec<u8> = list.iter().flat_map(|(i, _)| i.to_le_bytes()).collect();
            let wv: Vec<u8> = list.iter().flat_map(|(_, w)| w.to_le_bytes()).collect();
            let ib = a.bytes(idx.len())?;
            ib.write(0, &idx)?;
            let wb = a.bytes(wv.len())?;
            wb.write(0, &wv)?;
            let eo = a.f32(n * c)?;
            let mmvq = n <= 8 && [&m.gate, &m.up, &m.down].iter().all(|w| self.mmvq_ok(w.ty, w.cols));
            let gt = a.f32(n * fe)?;
            if mmvq {
                // decode width: the products straight from the stored blocks against Q8_1 activations
                let xe = a.f32(n * c)?;
                // SAFETY: x holds t rows of c floats, the indices are tokens < t; xe holds n * c floats.
                ffi::check(unsafe { (self.k.gather)(self.raw(), x.fp(), ib.ptr().cast::<i32>().cast_const(), xe.fp(), n as i64, c as i64) }, "gather")?;
                let xq = self.quantize(&xe, n, c)?;
                let up = a.f32(n * fe)?;
                self.mmvq_at(m.gate.ty, base, &xq, c, fe, n, &gt)?;
                // SAFETY: base + off_up is inside the slot (total bytes).
                self.mmvq_at(m.up.ty, unsafe { base.add(off_up) }, &xq, c, fe, n, &up)?;
                // SAFETY: gt, up hold n * fe floats.
                ffi::check(unsafe { (self.k.swiglu_oai)(self.raw(), gt.fp(), up.fp(), gt.fp(), (n * fe) as i64, 1.702, 7.0) }, "swiglu_oai")?;
                let hq = self.quantize(&gt, n, fe)?;
                // SAFETY: base + off_down is inside the slot.
                self.mmvq_at(m.down.ty, unsafe { base.add(off_down) }, &hq, fe, c, n, &eo)?;
            } else {
                let xe = a.bytes(n * c * 2)?;
                // SAFETY: x holds t rows of c floats and the indices are tokens < t; xe holds n * c halves.
                ffi::check(unsafe { (self.k.gather_f16)(self.raw(), x.fp(), ib.ptr().cast::<i32>().cast_const(), xe.ptr().cast::<u16>(), n as i64, c as i64) }, "gather_f16")?;
                let up = a.f32(n * fe)?;
                self.half_product(&m.gate, base, &xe, n, &gt)?;
                // SAFETY: inside the slot.
                self.half_product(&m.up, unsafe { base.add(off_up) }, &xe, n, &up)?;
                // SAFETY: gt, up hold n * fe floats.
                ffi::check(unsafe { (self.k.swiglu_oai)(self.raw(), gt.fp(), up.fp(), gt.fp(), (n * fe) as i64, 1.702, 7.0) }, "swiglu_oai")?;
                let h16 = a.bytes(n * fe * 2)?;
                // SAFETY: gt holds n * fe floats, h16 n * fe halves.
                ffi::check(unsafe { (self.k.to_f16)(self.raw(), gt.fp(), h16.ptr().cast::<u16>(), (n * fe) as i64) }, "to_f16")?;
                // SAFETY: inside the slot.
                self.half_product(&m.down, unsafe { base.add(off_down) }, &h16, n, &eo)?;
            }
            // SAFETY: y holds t rows of c floats; eo n rows; idx and w n entries.
            ffi::check(unsafe { (self.k.scatter_add)(self.raw(), y.fp(), eo.fp(), ib.ptr().cast::<i32>().cast_const(), wb.fp(), n as i64, c as i64) }, "scatter_add")?;
            self.gpu.sync()?;
            drop(slot);
            a.rewind(mk);
        }
        {
            let mut p = self.prof.lock().unwrap();
            p[0] += read_s;
            p[1] += tr.elapsed().as_secs_f64();
            if self.pf {
                let mut lay = self.lay.lock().unwrap();
                lay[l][4] += read_s;
                lay[l][5] += tr.elapsed().as_secs_f64() - read_s;
            }
        }
        tm = Instant::now();
        tap(&format!("ffn_moe_out-{l}"), &y)?;
        let sh = self.dense(&m.shared, x, t)?;
        tap(&format!("ffn_shexp-{l}"), &sh)?;
        self.add(&y, &sh, t * c)?;
        self.lap(l, 6, &mut tm);
        Ok(y)
    }

    /// Whether a matrix of this type and width has a decode-width kernel
    fn mmvq_ok(&self, ty: GType, cols: usize) -> bool {
        cols.is_multiple_of(256) && ty != GType::F32 && unsafe { (self.k.mmvq_supported)(ty.code() as i32) } != 0
    }

    /// x [n, cols] as Q8_1 blocks (n at most 8 columns)
    fn quantize(&self, x: &DevBuf, n: usize, cols: usize) -> Result<DevBuf> {
        // SAFETY: plain size query.
        let bytes = unsafe { (self.k.q8_1_bytes)(cols as i64, n as i64) };
        let q = self.arena.bytes(bytes)?;
        // SAFETY: x holds n * cols floats (the callers'), q the blocks.
        ffi::check(unsafe { (self.k.quantize_q8_1)(self.raw(), x.fp(), q.ptr(), cols as i64, n as i64) }, "quantize_q8_1")?;
        Ok(q)
    }

    /// y [n, rows] = the matrix at `w` (stored blocks of `ty`, `rows` rows of `cols`) . the n columns of `xq`
    #[allow(clippy::too_many_arguments)]
    fn mmvq_at(&self, ty: GType, w: *const u8, xq: &DevBuf, cols: usize, rows: usize, n: usize, y: &DevBuf) -> Result<()> {
        y.bounds(0, n * rows * 4)?;
        // SAFETY: w is a device matrix of rows * cols values (the callers' slices), xq the n columns, y checked.
        let rc = unsafe { (self.k.mmvq)(self.raw(), ty.code() as i32, w.cast(), xq.ptr().cast_const(), y.fp(), cols as i64, rows as i64, n as i64) };
        ffi::check(rc, "mmvq")
    }

    /// y [n, rows] = x16 [n, cols] . W^T for a matrix of stored blocks at `w`: dequantized to fp16 on the GPU, multiplied by oneMKL
    fn half_product(&self, w: &Experts, wp: *const u8, x16: &DevBuf, n: usize, y: &DevBuf) -> Result<()> {
        let m = self.arena.mark();
        let nv = w.rows * w.cols;
        let w16 = self.arena.bytes(nv * 2)?;
        // SAFETY: wp holds the matrix's rows * cols values in the type; w16 holds as many halves; x16 n * cols, y n * rows.
        unsafe {
            ffi::check((self.k.dequant_f16)(self.raw(), w.ty.code() as i32, wp.cast(), nv as i64, w16.ptr().cast::<u16>()), "dequant_f16")?;
            ffi::check((self.k.gemm_f16)(self.raw(), n as i64, w.rows as i64, w.cols as i64, x16.ptr().cast::<u16>().cast_const(), w.cols as i64,
                                         w16.ptr().cast::<u16>().cast_const(), y.fp(), w.rows as i64, 0), "gemm_f16")?;
        }
        self.gpu.sync()?;
        self.arena.rewind(m);
        Ok(())
    }

    /// An expert's three matrices [gate | up | down] as one device-readable block: from the cache (VRAM, or pinned host
    /// memory the kernels read in place), else read from the file - and kept while the budgets last
    fn expert_slot(&self, l: usize, ex: usize, m: &Moe, total: usize, off_up: usize, off_down: usize) -> Result<Arc<Slot>> {
        if let Some(s) = self.cache.lock().unwrap().map.get(&(l, ex)) {
            return Ok(s.clone());
        }
        let mut host = vec![0u8; total];
        for (w, at) in [(&m.gate, 0), (&m.up, off_up), (&m.down, off_down)] {
            let t = self.file.tensor(&w.name).ok_or_else(|| e("expert tensor missing"))?;
            self.file.read_into(t, (ex * w.stride) as u64, &mut host[at..at + w.stride]).map_err(|x| e(x.0))?;
        }
        let mut c = self.cache.lock().unwrap();
        let slot = if c.vram_left >= total {
            let buf = DevBuf::new(&self.gpu, total)?;
            buf.write(0, &host)?;
            c.vram_left -= total;
            Arc::new(Slot { buf })
        } else if c.host_left >= total {
            if c.slabs.last().is_none_or(|(_, used)| SLAB - used < total) {
                let hb = nextsycl_core::HostBuf::new(&self.gpu, SLAB)?;
                c.slabs.push((hb, 0));
            }
            let (hb, used) = c.slabs.last_mut().unwrap();
            let at = *used;
            hb.as_mut_slice()[at..at + total].copy_from_slice(&host);
            *used = (at + total).next_multiple_of(256);
            let view = hb.device_view(at, total)?;
            c.host_left -= total;
            Arc::new(Slot { buf: view })
        } else {
            let buf = DevBuf::new(&self.gpu, total)?;
            buf.write(0, &host)?;
            return Ok(Arc::new(Slot { buf }));
        };
        c.map.insert((l, ex), slot.clone());
        Ok(slot)
    }

    // ---- passes

    fn pass(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let _w = self.work.lock().unwrap();
        let t = tokens.len();
        if t == 0 {
            return Err(e("no tokens"));
        }
        if s.pos + t > s.max_ctx {
            return Err(e(format!("the context is full ({} of {})", s.pos + t, s.max_ctx)));
        }
        if n_out > t {
            return Err(e("more rows asked than tokens"));
        }
        let c = self.g.n_embd as usize;
        let p0 = s.pos;
        let pass_t0 = Instant::now();
        let mut x = self.embed(tokens)?;
        tap("inp_embd", &x)?;
        for (l, ly) in self.layers.iter().enumerate() {
            let m0 = self.arena.mark();
            let mut tm = Instant::now();
            let xn = self.arena.f32(t * c)?;
            self.rms(&x, &ly.attn_norm, &xn, t, c)?;
            tap(&format!("attn_norm-{l}"), &xn)?;
            self.lap(l, 7, &mut tm);
            let att = self.attention(l, ly, s, &xn, t, p0, &mut *tap)?;
            self.add(&att, &x, t * c)?;
            tap(&format!("ffn_inp-{l}"), &att)?;
            let hn = self.arena.f32(t * c)?;
            self.rms(&att, &ly.ffn_norm, &hn, t, c)?;
            tap(&format!("ffn_norm-{l}"), &hn)?;
            self.lap(l, 7, &mut tm);
            let f = match &ly.ffn {
                Ffn::Dense(d) => {
                    let o = self.dense(d, &hn, t)?;
                    self.lap(l, 6, &mut tm);
                    o
                }
                Ffn::Moe(m) => {
                    let o = self.moe(l, m, &hn, t, &mut *tap)?;
                    tm = Instant::now();
                    o
                }
            };
            tap(&format!("ffn_out-{l}"), &f)?;
            self.add(&f, &att, t * c)?;
            tap(&format!("l_out-{l}"), &f)?;
            let nx = DevBuf::f32(&self.gpu, t * c)?;
            nx.copy_within(0, &f, 0, t * c * 4)?;
            self.gpu.sync()?;
            x = nx;
            self.arena.rewind(m0);
            self.lap(l, 7, &mut tm);
            // NS_MM31_FORCE=DIR (Debug): the layer's output replaced by the reference dump's, so the next layer is
            // compared on the reference's own input (exactness per layer, not the noise of 60 layers added up)
            if let Some(dir) = std::env::var_os("NS_MM31_FORCE") {
                let path = std::path::Path::new(&dir).join(format!("l_out-{l}.f32"));
                if let Ok(raw) = std::fs::read(&path) {
                    if raw.len() == t * c * 4 {
                        x.write(0, &raw)?;
                    }
                }
            }
        }
        // the head, for the last n_out rows
        let xr = x.view((t - n_out) * c * 4, n_out * c * 4)?;
        let xn = DevBuf::f32(&self.gpu, n_out * c)?;
        self.rms(&xr, &self.out_norm, &xn, n_out, c)?;
        tap("result_norm", &xn)?;
        let logits = DevBuf::f32(&self.gpu, n_out * self.head.rows)?;
        self.matmul(&self.head, &xn, n_out, &logits)?;
        tap("result_output", &logits)?;
        let all = logits.to_f32()?;
        if self.pf && t == 1 {
            let mut p = self.pf_tokens.lock().unwrap();
            p.0 += 1;
            p.1 += pass_t0.elapsed().as_secs_f64();
        }
        s.pos += t;
        s.last_rows = t;
        let v = self.head.rows;
        Ok((0..n_out).map(|i| all[i * v..(i + 1) * v].to_vec()).collect())
    }

    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.g;
        let (kv, hd) = (g.n_kv_head as usize, g.head_dim as usize);
        let il = g.indexer.key_len as usize;
        let n_sparse = self.layers.iter().filter(|l| l.idx.is_some()).count();
        let need = (self.layers.len() * 2 * max_ctx * kv * hd * 2 + n_sparse * max_ctx * il * 2) as u64;
        let (total, free) = self.gpu.memory()?;
        let free = free.unwrap_or(total);
        if need + (1536 << 20) > free {
            return Err(e(format!("a session of {max_ctx} tokens needs {:.2} GiB; {:.2} GiB are free (1.5 GiB kept spare)", need as f64 / (1u64 << 30) as f64,
                                 free as f64 / (1u64 << 30) as f64)));
        }
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut ik = Vec::new();
        for ly in &self.layers {
            k.push(DevBuf::new(&self.gpu, max_ctx * kv * hd * 2)?);
            v.push(DevBuf::new(&self.gpu, max_ctx * kv * hd * 2)?);
            ik.push(if ly.idx.is_some() { Some(DevBuf::new(&self.gpu, max_ctx * il * 2)?) } else { None });
        }
        Ok(Session { pos: 0, max_ctx, k, v, ik, last_rows: 0 })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.pos = 0;
        s.last_rows = 0;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if src.pos > dst.max_ctx {
            return Err(e(format!("a session of {} tokens into one of {}", src.pos, dst.max_ctx)));
        }
        let g = &self.g;
        let (kv, hd, il) = (g.n_kv_head as usize, g.head_dim as usize, g.indexer.key_len as usize);
        for l in 0..self.layers.len() {
            dst.k[l].copy_within(0, &src.k[l], 0, src.pos * kv * hd * 2)?;
            dst.v[l].copy_within(0, &src.v[l], 0, src.pos * kv * hd * 2)?;
            if let (Some(d), Some(sr)) = (&dst.ik[l], &src.ik[l]) {
                d.copy_within(0, sr, 0, src.pos * il * 2)?;
            }
        }
        self.gpu.sync()?;
        dst.pos = src.pos;
        dst.last_rows = 0;
        Ok(())
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let mut logits = Vec::new();
        let mut done = 0;
        for c in tokens.chunks(prefill_chunk()) {
            if done > 0 && stop(done) {
                break;
            }
            logits = self.pass(s, c, 1, &mut *tap)?.pop().unwrap_or_default();
            done += c.len();
        }
        Ok((done, logits))
    }

    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        self.pass(s, tokens, n_out, tap)
    }

    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        if keep > s.last_rows {
            return Err(e("rollback past the last pass"));
        }
        s.pos = s.pos - s.last_rows + keep;
        s.last_rows = keep;
        Ok(())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(tokens.len());
        for (s, t) in sessions.iter_mut().zip(tokens) {
            out.push(self.pass(s, &[*t], 1, &mut *tap)?.pop().unwrap_or_default());
        }
        Ok(out)
    }

    /// The next committed token: drawn from the prompt's logits, or after feeding the one committed before
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let y = match (d.logits.take(), d.next.take()) {
            (Some(l), _) => smp.sample(&l),
            (None, Some(x)) => {
                let l = self.pass(s, &[x], 1, tap)?.pop().unwrap_or_default();
                smp.sample(&l)
            }
            (None, None) => return Err(e("a decoder with nothing to draw from")),
        };
        d.next = Some(y);
        Ok(vec![y])
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let g = &self.g;
        let (kv, hd, il) = (g.n_kv_head as usize, g.head_dim as usize, g.indexer.key_len as usize);
        let mut parts = Vec::new();
        let mut bytes = 0;
        for l in 0..self.layers.len() {
            for (b, w) in [(&s.k[l], kv * hd * 2), (&s.v[l], kv * hd * 2)] {
                let mut h = vec![0u8; s.pos * w];
                b.read(0, &mut h)?;
                bytes += h.len();
                parts.push(h);
            }
            let mut h = Vec::new();
            if let Some(b) = &s.ik[l] {
                h = vec![0u8; s.pos * il * 2];
                b.read(0, &mut h)?;
            }
            bytes += h.len();
            parts.push(h);
        }
        Ok(Checkpoint { pos: s.pos, bytes, parts })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        if ck.pos > s.max_ctx {
            return Err(e(format!("a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        if ck.parts.len() != self.layers.len() * 3 {
            return Err(e("a checkpoint of another layout"));
        }
        for l in 0..self.layers.len() {
            s.k[l].write(0, &ck.parts[l * 3])?;
            s.v[l].write(0, &ck.parts[l * 3 + 1])?;
            if let Some(b) = &s.ik[l] {
                b.write(0, &ck.parts[l * 3 + 2])?;
            }
        }
        s.pos = ck.pos;
        s.last_rows = 0;
        Ok(())
    }
}
