//! MiniMax-M3.1-Flash (`minimax-m3`, llama.cpp's src/models/minimax-m3.cpp): 60 pre-norm layers. Attention is GQA
//! (64 query heads, 4 key/value heads of 128), q and k RMS-normed per head, NEOX RoPE on the first 64 values; on
//! every layer but the first three (llama.cpp's reading: docs/ports/minimax-m31.md) MiniMax Sparse Attention picks,
//! per key/value group, the 16 best blocks of 128 positions by an indexer of its own (4 heads of 128, normed and
//! rotated like q and k) and the block holding the query, and attends only those. The feed-forward is a dense
//! swiglu-oai block (layers 0-2) or 4 of 128 experts (sigmoid gate, a bias to choose with, weights renormalized and
//! x2) plus a shared expert, all swiglu-oai (alpha 1.702, limit 7).
//!
//! Float32 activations; decode-width products straight from the stored blocks (mmvq), wider passes dequantize on the
//! GPU into a scratch and multiply by oneMKL. The dense weights stay in their GGUF blocks on the GPUs: the layers are
//! split over the cards (the hidden state hops at the boundary). The routed experts (~110 GiB) do not fit the cards:
//! each card keeps what its VRAM and its share of the pinned host budget hold (placed at load, the hottest first by
//! the expert profile when there is one), the rest are read from the file for the pass - O_DIRECT, many reads in
//! flight, into pinned staging memory the kernels read in place - while the GPU works on the experts it already has.
//!
//! ```text
//!   x = embed(tokens)
//!   per layer:  h = x + attn(attn_norm(x));  x = h + ffn(ffn_norm(h))
//!   logits = output(output_norm(x))
//! ```

use std::collections::HashMap;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Gpu, HostBuf};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_llm::{Error, ExpertStats, GpuInfo, Result, Sampler, Tap};

use crate::ffi;
use crate::model::Geometry;

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

/// Tokens a prompt pass takes (NS_MM31_CHUNK)
pub fn prefill_chunk() -> usize {
    std::env::var("NS_MM31_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(4096)
}

/// NS_MM31_SPARSE_ALL=1: the first three layers take the indexer too (the model's config says every layer is sparse;
/// llama.cpp's graph leaves them dense - the reference this engine is checked against)
fn sparse_all() -> bool {
    std::env::var("NS_MM31_SPARSE_ALL").is_ok_and(|v| v == "1")
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok())
}

const GIB: usize = 1 << 30;

/// A matrix [rows (outputs), cols (inputs)] in its stored blocks (or float32), on the GPU
pub struct Mat {
    pub buf: DevBuf,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

/// pinned host memory is taken in slabs of this many bytes, device memory for experts in slabs of this many
const SLAB: usize = 1 << 30;

/// Experts a wave of file reads holds (two waves are in the staging ring: one read while the other computes)
const WAVE: usize = 8;
/// VRAM slots a copied expert lands in (the copy lane runs up to LA experts ahead of the products)
const NX: usize = 4;
const LA: usize = 2;

/// NS_MM31_SPLIT_ATTEND=0: the decode passes run the attention kernel of the prompt passes or the shared one
fn no_split() -> bool {
    std::env::var("NS_MM31_SPLIT_ATTEND").is_ok_and(|v| v == "0")
}

/// Which copy lane (0 or 1) the expert copies use (NS_MM31_LANE)
fn copy_lane() -> i32 {
    std::env::var("NS_MM31_LANE").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(0, 1)
}

/// Reads in flight while a card's experts are placed at load
const LOAD_THREADS: usize = 12;

/// Slots for experts read ahead of the layer that will use them
const PF_SLOTS: usize = 12;

/// Experts of the next layer guessed and read ahead (NS_MM31_AHEAD; 0 = none)
fn ahead() -> usize {
    std::env::var("NS_MM31_AHEAD").ok().and_then(|v| v.parse().ok()).unwrap_or(3).min(PF_SLOTS)
}

// ---- the file's reads

/// A buffer aligned for O_DIRECT, one a thread
struct Bounce {
    ptr: *mut u8,
    cap: usize,
}

impl Bounce {
    fn grow(&mut self, n: usize) {
        if n > self.cap {
            self.free();
            let cap = n.next_multiple_of(1 << 20);
            // SAFETY: a non-zero size and a power-of-two alignment; freed in `free` with the same layout.
            self.ptr = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(cap, 4096).unwrap()) };
            self.cap = cap;
        }
    }
    fn free(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated in `grow` with this capacity and alignment.
            unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.cap, 4096).unwrap()) };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Bounce {
    fn drop(&mut self) {
        self.free();
    }
}

thread_local! {
    static BOUNCE: std::cell::RefCell<Bounce> = const { std::cell::RefCell::new(Bounce { ptr: std::ptr::null_mut(), cap: 0 }) };
}

/// A destination pointer a reader thread may write through (the owner keeps the memory and waits for the thread)
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
// SAFETY: plain memory the owner hands to exactly one reader and joins before use.
unsafe impl Send for SendPtr {}

/// The model's shards, open twice each: buffered and (when the file system allows) O_DIRECT. Experts are read with
/// O_DIRECT so that 100 GB of them do not churn the page cache and the reads go to the drive at the depth asked for.
struct Reader {
    files: Vec<(std::fs::File, Option<std::fs::File>)>,
    bytes: AtomicU64,
}

/// O_DIRECT's value on Linux x86-64 / aarch64 (no libc crate in the workspace for one constant)
const O_DIRECT: i32 = 0o40000;

impl Reader {
    fn open(paths: &[std::path::PathBuf]) -> Result<Reader> {
        let direct = std::env::var("NS_MM31_DIRECT").map_or(true, |v| v != "0");
        let mut files = Vec::new();
        for p in paths {
            let b = std::fs::File::open(p).map_err(|x| e(format!("{}: {x}", p.display())))?;
            let d = if direct { std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(p).ok() } else { None };
            files.push((b, d));
        }
        Ok(Reader { files, bytes: AtomicU64::new(0) })
    }

    /// `len` bytes at file offset `off` of shard `shard` into `dst`
    fn read(&self, shard: usize, off: u64, dst: *mut u8, len: usize) -> std::io::Result<()> {
        self.bytes.fetch_add(len as u64, Relaxed);
        // SAFETY: the caller owns dst[..len] and no one else touches it until this returns.
        let out = unsafe { std::slice::from_raw_parts_mut(dst, len) };
        let (b, d) = &self.files[shard];
        let Some(d) = d else { return b.read_exact_at(out, off) };
        let a0 = off & !4095;
        let end = off + len as u64;
        let a1 = end.div_ceil(4096) * 4096;
        let need = (end - a0) as usize;
        let ok = BOUNCE.with(|bb| {
            let mut bb = bb.borrow_mut();
            bb.grow((a1 - a0) as usize);
            // SAFETY: the buffer holds at least a1 - a0 bytes.
            let buf = unsafe { std::slice::from_raw_parts_mut(bb.ptr, (a1 - a0) as usize) };
            let mut got = 0usize;
            while got < need {
                // a read past the end of the file returns what is there; a short unaligned one ends the direct path
                match d.read_at(&mut buf[got..], a0 + got as u64) {
                    Ok(0) | Err(_) => return false,
                    Ok(r) => got += r,
                }
                if got < need && !got.is_multiple_of(4096) {
                    return false;
                }
            }
            out.copy_from_slice(&buf[(off - a0) as usize..(off - a0) as usize + len]);
            true
        });
        if ok {
            Ok(())
        } else {
            b.read_exact_at(out, off)
        }
    }
}

/// Where an expert's three matrices are in the files, and where each goes in a slot: (shard, file offset, bytes, offset in the slot)
#[derive(Clone, Copy)]
struct Src {
    mats: [(usize, u64, usize, usize); 3],
}

type Reads = Vec<JoinHandle<std::io::Result<()>>>;

/// Experts being read ahead: (layer, expert) -> (slot, its reads); the free slots
#[derive(Default)]
struct PfState {
    free: Vec<usize>,
    map: HashMap<(usize, usize), (usize, Reads)>,
}

fn join_all(hs: Reads) -> Result<()> {
    let mut first = None;
    for h in hs {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(x)) => first = first.or(Some(e(format!("expert read: {x}")))),
            Err(_) => first = first.or(Some(e("expert read: a reader panicked"))),
        }
    }
    first.map_or(Ok(()), Err)
}

fn spawn_reads(r: &Arc<Reader>, src: Src, dst: *mut u8) -> Reads {
    src.mats
        .iter()
        .map(|&(shard, off, len, at)| {
            let r = r.clone();
            // SAFETY: at + len is inside the slot dst points to (the slot sizes are computed from these).
            let p = SendPtr(unsafe { dst.add(at) });
            std::thread::spawn(move || {
                let p = p;
                r.read(shard, off, p.0, len)
            })
        })
        .collect()
}

// ---- the model's parts

/// An expert's [gate | up | down] bytes where a kernel reads them: VRAM, or pinned host memory (read in place over PCIe)
struct Resident {
    buf: DevBuf,
    vram: bool,
}

/// One layer's routed experts in the file: [experts][rows][cols] in `ty`
struct Experts {
    ty: GType,
    rows: usize,
    cols: usize,
    /// bytes an expert takes
    stride: usize,
    shard: usize,
    offset: u64,
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
    /// where an expert's bytes are in the stream file (the second drive's copy of the experts that are not on the cards), or MAX
    stream: Vec<u64>,
    stream_shard: usize,
}

impl Moe {
    /// an expert's slot: [gate | up | down], each 256-aligned: (offset of up, offset of down, total bytes)
    fn layout(&self) -> (usize, usize, usize) {
        let off_up = self.gate.slot_bytes();
        let off_down = off_up + self.up.slot_bytes();
        (off_up, off_down, off_down + self.down.slot_bytes())
    }
    fn src(&self, ex: usize) -> Src {
        let (u, d, _) = self.layout();
        if let Some(&base) = self.stream.get(ex).filter(|b| **b != u64::MAX) {
            let m = |w: &Experts, at: usize| (self.stream_shard, base + at as u64, w.stride, at);
            return Src { mats: [m(&self.gate, 0), m(&self.up, u), m(&self.down, d)] };
        }
        let m = |w: &Experts, at: usize| (w.shard, w.offset + (ex * w.stride) as u64, w.stride, at);
        Src { mats: [m(&self.gate, 0), m(&self.up, u), m(&self.down, d)] }
    }
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

/// One GPU: its layers, scratch, the experts it keeps and the ring its file reads land in
struct Card {
    gpu: Arc<Gpu>,
    arena: Arena,
    layers: std::ops::Range<usize>,
    /// [layer - first][expert]
    res: Vec<Vec<Option<Resident>>>,
    /// staging ring for streamed experts: 2 waves of WAVE slots of `slot` bytes (pinned, read in place by the kernels)
    stage: HostBuf,
    slot: usize,
    /// VRAM scratch of two slots: a pinned expert is copied here (a queued copy, ~20 GB/s) and read from VRAM; products reading
    /// pinned memory in place ran at ~3 GB/s (5.8 ms an expert against 0.30 ms from VRAM)
    xfer: DevBuf,
    pfs: Mutex<PfState>,
    /// slabs the resident experts live in (kept for their views)
    _vram: Vec<DevBuf>,
    _host: Vec<HostBuf>,
    /// bytes of experts resident in VRAM and in pinned memory
    kept: (usize, usize),
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

/// Counters of the expert store since load
#[derive(Default)]
struct Counts {
    vram: AtomicU64,
    pinned: AtomicU64,
    file: AtomicU64,
    /// experts read ahead, and those of them the next layer used
    ahead: AtomicU64,
    ahead_used: AtomicU64,
    /// experts the next layer needed that were not on the card (the guess's target)
    ahead_wanted: AtomicU64,
}

const NCAT: usize = 8;
const CATS: [&str; NCAT] = ["attn proj", "indexer", "attend", "route", "expert read wait", "expert products", "shared/dense", "other"];

pub struct MiniMax<'g> {
    file: &'g Gguf,
    cards: Vec<Card>,
    /// the card each layer runs on
    owner: Vec<usize>,
    k: &'static ffi::Api,
    pub g: Geometry,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    embd: ConstTensor,
    head: Mat,
    pub load_seconds: f64,
    pub load_bytes: u64,
    work: Mutex<()>,
    reader: Arc<Reader>,
    counts: Counts,
    /// decode requests per (layer, expert) since load (the expert profile)
    usage: Vec<AtomicU64>,
    /// NS_PROFILE=1: per layer, seconds by category (`CATS`), each timed to the GPU's completion (a sync at every lap);
    /// one table for one-token passes, one for wider ones
    pf: bool,
    lay: Mutex<[Vec<[f64; NCAT]>; 2]>,
    cur_wide: std::sync::atomic::AtomicBool,
    stage_ok: std::sync::atomic::AtomicBool,
    /// profile: one-token expert stages (gate, up, swiglu + quantize, down, scatter), seconds and bytes of weights read
    stage_t: Mutex<[f64; 6]>,
    /// passes of one token, and the seconds they took (profile)
    pf_tokens: Mutex<(usize, f64)>,
}

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

fn gib(n: usize) -> f64 {
    n as f64 / GIB as f64
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
    Ok(Experts { ty: t.ty, rows, cols, stride, shard: t.shard, offset: t.offset })
}

/// The expert profile (NS_EXPERT_PROFILE: "layer expert weight" lines)
fn read_profile(path: &std::path::Path) -> Option<HashMap<(usize, usize), f64>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut m = HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split_whitespace();
        if let (Some(l), Some(x), Some(w)) = (it.next(), it.next(), it.next()) {
            if let (Ok(l), Ok(x), Ok(w)) = (l.parse(), x.parse(), w.parse()) {
                m.insert((l, x), w);
            }
        }
    }
    Some(m)
}

/// One card's weights: its layers
struct Part {
    layers: Vec<Layer>,
    bytes: u64,
}

fn load_part(f: &Gguf, g: &Geometry, gpu: &Arc<Gpu>, range: std::ops::Range<usize>, log: &mut dyn FnMut(String)) -> Result<Part> {
    let mut bytes = 0u64;
    let eps_dim = g.n_embd as usize;
    let mut layers = Vec::with_capacity(range.len());
    for l in range.clone() {
        let n = |s: &str| format!("blk.{l}.{s}.weight");
        let sparse = l as u64 >= g.n_dense || sparse_all();
        let idx = if sparse {
            let kl = g.indexer.key_len as usize;
            Some(Indexer {
                q: mat(f, gpu, &n("indexer.q_proj"), &mut bytes)?,
                k: mat(f, gpu, &n("indexer.k_proj"), &mut bytes)?,
                qn: vecf(f, gpu, &n("indexer.q_norm"), kl)?,
                kn: vecf(f, gpu, &n("indexer.k_norm"), kl)?,
            })
        } else {
            None
        };
        let ffn = if (l as u64) < g.n_dense {
            Ffn::Dense(Dense {
                gate: mat(f, gpu, &n("ffn_gate"), &mut bytes)?,
                up: mat(f, gpu, &n("ffn_up"), &mut bytes)?,
                down: mat(f, gpu, &n("ffn_down"), &mut bytes)?,
            })
        } else {
            let bt = f.tensor(&format!("blk.{l}.exp_probs_b.bias")).ok_or_else(|| e("exp_probs_b missing"))?;
            let braw = f.read(bt).map_err(|x| e(x.0))?;
            let bias: Vec<f32> = braw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let ne = g.n_expert as usize;
            Ffn::Moe(Box::new(Moe {
                router: mat(f, gpu, &n("ffn_gate_inp"), &mut bytes)?,
                bias,
                gate: experts(f, n("ffn_gate_exps"), ne)?,
                up: experts(f, n("ffn_up_exps"), ne)?,
                down: experts(f, n("ffn_down_exps"), ne)?,
                stream: Vec::new(),
                stream_shard: 0,
                shared: Dense {
                    gate: mat(f, gpu, &n("ffn_gate_shexp"), &mut bytes)?,
                    up: mat(f, gpu, &n("ffn_up_shexp"), &mut bytes)?,
                    down: mat(f, gpu, &n("ffn_down_shexp"), &mut bytes)?,
                },
            }))
        };
        layers.push(Layer {
            attn_norm: vecf(f, gpu, &n("attn_norm"), eps_dim)?,
            q: mat(f, gpu, &n("attn_q"), &mut bytes)?,
            k: mat(f, gpu, &n("attn_k"), &mut bytes)?,
            v: mat(f, gpu, &n("attn_v"), &mut bytes)?,
            o: mat(f, gpu, &n("attn_output"), &mut bytes)?,
            qn: vecf(f, gpu, &n("attn_q_norm"), g.head_dim as usize)?,
            kn: vecf(f, gpu, &n("attn_k_norm"), g.head_dim as usize)?,
            idx,
            ffn_norm: vecf(f, gpu, &n("ffn_norm"), eps_dim)?,
            ffn,
        });
        if (l + 1) % 10 == 0 || l + 1 == range.end {
            log(format!("{}: layers {}-{l} loaded ({:.1} GiB)", gpu.name, range.start, bytes as f64 / GIB as f64));
        }
    }
    Ok(Part { layers, bytes })
}

/// Where a resident expert goes: slab index and offset, in device or pinned memory
struct Place {
    vram: bool,
    slab: usize,
    off: usize,
    layer: usize,
    ex: usize,
}

impl<'g> MiniMax<'g> {
    pub fn load(f: &'g Gguf, gpus: &[Arc<Gpu>], o: &nextsycl_llm::LoadOptions, log: &mut dyn FnMut(String)) -> Result<MiniMax<'g>> {
        let t0 = Instant::now();
        let kv = o.kv;
        let g = Geometry::from_gguf(f).map_err(|x| e(x.0))?;
        g.check(f).map_err(|x| e(x.0))?;
        if gpus.is_empty() {
            return Err(e("no GPU"));
        }
        let k = ffi::api()?;
        let n_layer = g.n_layer as usize;
        let ncard = gpus.len().min(n_layer);
        let gpus = &gpus[..ncard];
        // layers over the cards: even by default (NS_MM31_SPLIT=N: N layers on the first of two)
        let mut bounds: Vec<usize> = (0..=ncard).map(|i| n_layer * i / ncard).collect();
        if let Some(n) = std::env::var("NS_MM31_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| ncard == 2 && *n > 0 && *n < n_layer) {
            bounds[1] = n;
        }
        let mut owner = vec![0usize; n_layer];
        for c in 0..ncard {
            for o in owner.iter_mut().take(bounds[c + 1]).skip(bounds[c]) {
                *o = c;
            }
        }
        for gpu in gpus {
            let (total, free) = gpu.memory()?;
            log(format!("{}: {:.1} GiB, {:.1} GiB free", gpu.name, gib(total as usize), gib(free.unwrap_or(total) as usize)));
        }
        // each card's weights load at once, a thread each (NS_LOAD_SEQ=1: one after the other)
        let seq = std::env::var("NS_LOAD_SEQ").is_ok_and(|v| v == "1");
        let ranges: Vec<std::ops::Range<usize>> = (0..ncard).map(|c| bounds[c]..bounds[c + 1]).collect();
        let parts: Vec<Result<(Part, Vec<String>)>> = if seq {
            ranges
                .iter()
                .zip(gpus)
                .map(|(r, gpu)| {
                    let mut ls = Vec::new();
                    load_part(f, &g, gpu, r.clone(), &mut |s| ls.push(s)).map(|p| (p, ls))
                })
                .collect()
        } else {
            std::thread::scope(|sc| {
                let hs: Vec<_> = ranges
                    .iter()
                    .zip(gpus)
                    .map(|(r, gpu)| {
                        let (g, r) = (&g, r.clone());
                        sc.spawn(move || {
                            let mut ls = Vec::new();
                            load_part(f, g, gpu, r, &mut |s| ls.push(s)).map(|p| (p, ls))
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(e("a part's load panicked")))).collect()
            })
        };
        let mut layers = Vec::with_capacity(n_layer);
        let mut bytes = 0u64;
        for p in parts {
            let (p, ls) = p?;
            for l in ls {
                log(l);
            }
            bytes += p.bytes;
            layers.extend(p.layers);
        }
        let eps_dim = g.n_embd as usize;
        let last = gpus.last().unwrap();
        let out_norm = vecf(f, last, "output_norm.weight", eps_dim)?;
        let head = mat(f, last, "output.weight", &mut bytes)?;
        let et = f.tensor("token_embd.weight").ok_or_else(|| e("token_embd.weight missing"))?;
        let embd = ConstTensor { name: "token_embd.weight".into(), ty: et.ty, cols: g.n_embd as usize };

        // ---- what each card keeps of the experts
        let reader = Arc::new(Reader::open(&f.paths)?);
        let slot_max = layers.iter().filter_map(|l| if let Ffn::Moe(m) = &l.ffn { Some(m.layout().2) } else { None }).max().unwrap_or(256);
        let kv_tokens = if kv.0 > 0 { kv.0 } else { 32768 };
        let n_sparse_of = |r: &std::ops::Range<usize>| layers[r.clone()].iter().filter(|l| l.idx.is_some()).count();
        let per_tok = |r: &std::ops::Range<usize>| r.len() * 2 * (g.n_kv_head * g.head_dim) as usize * 2 + n_sparse_of(r) * g.indexer.key_len as usize * 2;
        let arena_bytes = 1536usize << 20;
        let mut vram_left = Vec::new();
        for (c, gpu) in gpus.iter().enumerate() {
            let (total, free) = gpu.memory()?;
            let free = free.unwrap_or(total) as usize;
            // the arena, the sessions' caches, and 1.5 GiB kept free (a spare of 1 GiB more: the driver's own use)
            let reserve = arena_bytes + kv_tokens * per_tok(&ranges[c]) + GIB + (3 << 29);
            let room = free.saturating_sub(reserve);
            let v = match env_f64("NS_MM31_VRAM_GIB").or(o.expert_bytes.map(|b| b as f64 / GIB as f64)) {
                Some(v) => ((v * GIB as f64) as usize).min(room),
                None => room,
            };
            vram_left.push(v);
        }
        // the experts each card could keep: its layers' (for the pinned memory's share)
        let exp_bytes: Vec<usize> = ranges
            .iter()
            .map(|r| layers[r.clone()].iter().map(|l| if let Ffn::Moe(m) = &l.ffn { m.layout().2.next_multiple_of(256) * g.n_expert as usize } else { 0 }).sum())
            .collect();
        let stage_bytes = (2 * WAVE + PF_SLOTS) * slot_max;
        let host_total = match env_f64("NS_MM31_PIN_GIB").or(o.mirror_bytes.map(|b| b as f64 / GIB as f64)) {
            Some(v) => (v * GIB as f64) as usize,
            // what is available less 16 GiB for the system, the page cache and the other programs, at most 32 GiB
            None => mem_available().saturating_sub(16 * GIB + ncard * stage_bytes).min(32 * GIB),
        };
        // pinned memory in proportion to what the cards cannot hold in VRAM
        let short: Vec<usize> = exp_bytes.iter().zip(&vram_left).map(|(a, b)| a.saturating_sub(*b)).collect();
        let short_sum: usize = short.iter().sum::<usize>().max(1);
        let host_left: Vec<usize> = short.iter().map(|s| (host_total as f64 * *s as f64 / short_sum as f64) as usize).map(|h| h / SLAB * SLAB).collect();
        let profile = std::env::var("NS_EXPERT_PROFILE").ok().and_then(|p| read_profile(std::path::Path::new(&p)));
        log(format!("experts: {:.1} GiB in all; kept up to {} in VRAM, {:.1} GiB pinned in host memory{}; the rest read from the file ({})",
                    gib(exp_bytes.iter().sum()), vram_left.iter().map(|v| format!("{:.1} GiB", gib(*v))).collect::<Vec<_>>().join(" + "), gib(host_left.iter().sum()),
                    if profile.is_some() { ", hottest first by the expert profile" } else { "" },
                    if reader.files.iter().all(|f| f.1.is_some()) { "O_DIRECT" } else { "buffered" }));

        // the placement: rank the experts of a card's layers (the profile's weight; without one, the same pseudo-random
        // order in every layer, so each layer keeps the same share), VRAM for the first that fit, then pinned memory
        let mut cards: Vec<Card> = Vec::new();
        for (c, gpu) in gpus.iter().enumerate() {
            let range = ranges[c].clone();
            let mut ranked: Vec<(usize, usize, f64)> = Vec::new();
            for l in range.clone() {
                if let Ffn::Moe(_) = &layers[l].ffn {
                    for x in 0..g.n_expert as usize {
                        let w = match &profile {
                            Some(p) => p.get(&(l, x)).copied().unwrap_or(0.0),
                            None => -(((x * 37 + l * 11) % g.n_expert as usize) as f64),
                        };
                        ranked.push((l, x, w));
                    }
                }
            }
            ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
            let (mut vleft, mut hleft) = (vram_left[c], host_left[c]);
            let (mut vslabs, mut hslabs): (Vec<usize>, Vec<usize>) = (vec![0], vec![0]);
            let mut places: Vec<Place> = Vec::new();
            let (mut vk, mut hk) = (0usize, 0usize);
            for &(l, x, _) in &ranked {
                let Ffn::Moe(m) = &layers[l].ffn else { continue };
                let tot = m.layout().2.next_multiple_of(256);
                if vleft >= tot {
                    vleft -= tot;
                    if *vslabs.last().unwrap() + tot > SLAB {
                        vslabs.push(0);
                    }
                    let (si, off) = (vslabs.len() - 1, *vslabs.last().unwrap());
                    *vslabs.last_mut().unwrap() += tot;
                    places.push(Place { vram: true, slab: si, off, layer: l, ex: x });
                    vk += tot;
                } else if hleft >= tot {
                    hleft -= tot;
                    if *hslabs.last().unwrap() + tot > SLAB {
                        hslabs.push(0);
                    }
                    let (si, off) = (hslabs.len() - 1, *hslabs.last().unwrap());
                    *hslabs.last_mut().unwrap() += tot;
                    places.push(Place { vram: false, slab: si, off, layer: l, ex: x });
                    hk += tot;
                }
            }
            let arena = Arena::new(gpu, arena_bytes)?;
            let dev: Vec<DevBuf> = vslabs.iter().filter(|n| **n > 0).map(|n| DevBuf::new(gpu, *n)).collect::<Result<_>>()?;
            let host: Vec<HostBuf> = hslabs.iter().filter(|n| **n > 0).map(|n| HostBuf::new(gpu, *n)).collect::<Result<_>>()?;
            let stage = HostBuf::new(gpu, stage_bytes)?;
            let mut res: Vec<Vec<Option<Resident>>> = range.clone().map(|_| (0..g.n_expert as usize).map(|_| None).collect()).collect();
            for p in &places {
                let Ffn::Moe(m) = &layers[p.layer].ffn else { continue };
                let tot = m.layout().2;
                let buf = if p.vram { dev[p.slab].view(p.off, tot)? } else { host[p.slab].device_view(p.off, tot)? };
                res[p.layer - range.start][p.ex] = Some(Resident { buf, vram: p.vram });
            }
            let xfer = DevBuf::new(gpu, NX * slot_max)?;
            cards.push(Card { gpu: gpu.clone(), arena, layers: range, res, stage, slot: slot_max, xfer, pfs: Mutex::new(PfState { free: (0..PF_SLOTS).collect(), map: HashMap::new() }), _vram: dev, _host: host, kept: (vk, hk) });
        }

        // the second drive: a copy of the experts the cards do not keep (a share of them), read beside the model's own file,
        // so the misses of a pass are read from two drives at once (NS_MM31_STREAM_DIR on the other drive)
        let mut reader = reader;
        if let Some(dir) = std::env::var_os("NS_MM31_STREAM_DIR").map(std::path::PathBuf::from) {
            let share = env_f64("NS_MM31_STREAM_SHARE").unwrap_or(50.0).clamp(0.0, 100.0) as u64;
            let mut list: Vec<(usize, usize)> = Vec::new();
            for c in &cards {
                for l in c.layers.clone() {
                    if let Ffn::Moe(_) = &layers[l].ffn {
                        for x in 0..g.n_expert as usize {
                            if c.res[l - c.layers.start][x].is_none() && ((l * 2654435761 + x * 40503) >> 3) as u64 % 100 < share {
                                list.push((l, x));
                            }
                        }
                    }
                }
            }
            let region = |l: usize| -> usize { if let Ffn::Moe(m) = &layers[l].ffn { m.layout().2.next_multiple_of(4096) } else { 0 } };
            let mut offs: Vec<u64> = Vec::with_capacity(list.len());
            let mut at = 0u64;
            for &(l, _) in &list {
                offs.push(at);
                at += region(l) as u64;
            }
            // the name carries the list and the model files' sizes: a different placement or model is a different file
            let mut h = 0xcbf29ce484222325u64;
            let mut mix = |v: u64| {
                h ^= v;
                h = h.wrapping_mul(0x100000001b3);
            };
            for p in &f.paths {
                mix(std::fs::metadata(p).map_or(0, |m| m.len()));
            }
            for &(l, x) in &list {
                mix((l * 128 + x) as u64);
            }
            let path = dir.join(format!("mm31-stream-{h:016x}.bin"));
            let ready = std::fs::metadata(&path).is_ok_and(|m| m.len() == at);
            if !ready {
                let t_b = Instant::now();
                log(format!("stream file {}: {:.1} GiB of {} experts, written once", path.display(), gib(at as usize), list.len()));
                std::fs::create_dir_all(&dir).map_err(|x| e(format!("{}: {x}", dir.display())))?;
                let part = path.with_extension("part");
                let out = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&part).map_err(|x| e(format!("{}: {x}", part.display())))?;
                out.set_len(at).map_err(e)?;
                let next = AtomicUsize::new(0);
                let errs: Mutex<Vec<Error>> = Mutex::new(Vec::new());
                std::thread::scope(|sc| {
                    for _ in 0..LOAD_THREADS {
                        sc.spawn(|| {
                            let mut tmp: Vec<u8> = Vec::new();
                            loop {
                                let i = next.fetch_add(1, Relaxed);
                                let Some(&(l, x)) = list.get(i) else { break };
                                let Ffn::Moe(m) = &layers[l].ffn else { continue };
                                let r = region(l);
                                tmp.clear();
                                tmp.resize(r, 0);
                                let res = (|| -> Result<()> {
                                    for &(shard, off, len, at) in &m.src(x).mats {
                                        reader.read(shard, off, tmp[at..].as_mut_ptr(), len).map_err(e)?;
                                    }
                                    out.write_all_at(&tmp, offs[i]).map_err(e)
                                })();
                                if let Err(x) = res {
                                    errs.lock().unwrap().push(x);
                                    break;
                                }
                            }
                        });
                    }
                });
                if let Some(x) = errs.into_inner().unwrap().into_iter().next() {
                    return Err(x);
                }
                out.sync_all().map_err(e)?;
                std::fs::rename(&part, &path).map_err(e)?;
                log(format!("stream file written in {:.0} s", t_b.elapsed().as_secs_f64()));
            }
            let mut paths = f.paths.clone();
            paths.push(path);
            for (i, &(l, x)) in list.iter().enumerate() {
                if let Ffn::Moe(m) = &mut layers[l].ffn {
                    if m.stream.is_empty() {
                        m.stream = vec![u64::MAX; g.n_expert as usize];
                        m.stream_shard = f.paths.len();
                    }
                    m.stream[x] = offs[i];
                }
            }
            reader = Arc::new(Reader::open(&paths)?);
            log(format!("{} experts off the cards are also read from the stream file on its drive", list.len()));
        }

        // fill them: the cards at once, LOAD_THREADS reads in flight each
        let t_fill = Instant::now();
        let fill = |card: &Card| -> Result<()> {
            let next = AtomicUsize::new(0);
            let items: Vec<(usize, usize)> = card
                .layers
                .clone()
                .flat_map(|l| (0..g.n_expert as usize).map(move |x| (l, x)))
                .filter(|&(l, x)| card.res[l - card.layers.start][x].is_some())
                .collect();
            let errs: Mutex<Vec<Error>> = Mutex::new(Vec::new());
            std::thread::scope(|sc| {
                for _ in 0..LOAD_THREADS {
                    sc.spawn(|| {
                        let mut tmp: Vec<u8> = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Relaxed);
                            let Some(&(l, x)) = items.get(i) else { break };
                            let Ffn::Moe(m) = &layers[l].ffn else { continue };
                            let r = card.res[l - card.layers.start][x].as_ref().unwrap();
                            let (_, _, tot) = m.layout();
                            let src = m.src(x);
                            let res: Result<()> = (|| {
                                if r.vram {
                                    tmp.resize(tot, 0);
                                    for &(shard, off, len, at) in &src.mats {
                                        reader.read(shard, off, tmp[at..].as_mut_ptr(), len).map_err(e)?;
                                    }
                                    r.buf.write(0, &tmp)
                                } else {
                                    for &(shard, off, len, at) in &src.mats {
                                        // SAFETY: at + len is inside this expert's slot, which no one else writes.
                                        reader.read(shard, off, unsafe { r.buf.ptr().cast::<u8>().add(at) }, len).map_err(e)?;
                                    }
                                    Ok(())
                                }
                            })();
                            if let Err(x) = res {
                                errs.lock().unwrap().push(x);
                                break;
                            }
                        }
                    });
                }
            });
            let mut errs = errs.into_inner().unwrap();
            if errs.is_empty() {
                Ok(())
            } else {
                Err(errs.remove(0))
            }
        };
        if seq {
            for c in &cards {
                fill(c)?;
            }
        } else {
            let rs: Vec<Result<()>> = std::thread::scope(|sc| {
                let hs: Vec<_> = cards.iter().map(|c| sc.spawn(|| fill(c))).collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(e("a fill panicked")))).collect()
            });
            for r in rs {
                r?;
            }
        }
        let kept: usize = cards.iter().map(|c| c.kept.0 + c.kept.1).sum();
        log(format!("experts placed: {:.1} GiB in {:.1} s ({})", gib(kept), t_fill.elapsed().as_secs_f64(),
                    cards.iter().map(|c| format!("{}: {:.1} GiB VRAM + {:.1} GiB pinned", c.gpu.name, gib(c.kept.0), gib(c.kept.1))).collect::<Vec<_>>().join("; ")));
        Ok(MiniMax {
            file: f,
            cards,
            owner,
            k,
            g,
            layers,
            out_norm,
            embd,
            head,
            load_seconds: t0.elapsed().as_secs_f64(),
            load_bytes: bytes + kept as u64,
            work: Mutex::new(()),
            reader,
            counts: Counts::default(),
            usage: (0..64 * 128).map(|_| AtomicU64::new(0)).collect(),
            pf: std::env::var("NS_PROFILE").is_ok_and(|v| v == "1"),
            lay: Mutex::new([vec![[0.0; NCAT]; 64], vec![[0.0; NCAT]; 64]]),
            cur_wide: false.into(),
            stage_ok: (std::env::var("NS_PROFILE").is_ok_and(|v| v == "1")).into(),
            stage_t: Mutex::new([0.0; 6]),
            pf_tokens: Mutex::new((0, 0.0)),
        })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        let slot = self.layers.iter().filter_map(|l| if let Ffn::Moe(m) = &l.ffn { Some(m.layout().2) } else { None }).max().unwrap_or(1).max(1);
        self.cards
            .iter()
            .map(|c| {
                let (total, free) = c.gpu.memory().unwrap_or((0, None));
                GpuInfo { index: c.gpu.index, name: c.gpu.name.clone(), pci: c.gpu.pci.clone(), total, free, layers: (c.layers.start as u64, c.layers.end as u64),
                          expert_slots: c.kept.0 / slot, host_slots: c.kept.1 / slot }
            })
            .collect()
    }

    pub fn expert_stats(&self) -> ExpertStats {
        ExpertStats {
            vram_hits: self.counts.vram.load(Relaxed),
            read_direct: self.counts.pinned.load(Relaxed),
            swapped_in: self.counts.file.load(Relaxed),
            from_host: 0,
            prefetched: self.counts.ahead.load(Relaxed),
            prefetch_used: self.counts.ahead_used.load(Relaxed),
        }
    }

    pub fn profile_lines(&self) -> Vec<String> {
        let mut v = Vec::new();
        let c = &self.counts;
        let (a, b, d) = (c.vram.load(Relaxed), c.pinned.load(Relaxed), c.file.load(Relaxed));
        v.push(format!("[experts used: {a} from VRAM, {b} from pinned host memory, {d} read from the file ({:.1}% on the cards); {:.1} GiB read from the file in all]",
                       100.0 * (a + b) as f64 / (a + b + d).max(1) as f64, self.reader.bytes.load(Relaxed) as f64 / GIB as f64));
        v.push(format!("[read ahead: {} experts guessed and read for the next layer, {} of the {} off-card experts it needed were among them]", c.ahead.load(Relaxed),
                       c.ahead_used.load(Relaxed), c.ahead_wanted.load(Relaxed)));
        if self.pf {
            let sg = self.stage_t.lock().unwrap();
            v.push(format!("[one-token expert stages, seconds in all: gate {:.3}, up {:.3}, swiglu+quantize {:.3}, down {:.3}, scatter {:.3}]", sg[0], sg[1], sg[2], sg[3], sg[4]));
            let lay = self.lay.lock().unwrap();
            let (n, secs) = *self.pf_tokens.lock().unwrap();
            for (name, tab) in [("one-token passes", &lay[0]), ("wider passes", &lay[1])] {
                v.push(format!("[profile, {name}{}: seconds by layer and category, each lap waits for the GPU]", if name.starts_with("one") { format!(" ({n} passes, {secs:.2} s)") } else { String::new() }));
                let mut head = String::from("layer");
                for c in CATS {
                    head += &format!(",{c}");
                }
                v.push(head);
                let mut tot = [0.0; NCAT];
                for (l, row) in tab.iter().enumerate().take(self.layers.len()) {
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
        }
        v
    }

    /// The decode requests per expert since load blended into the profile at `path` (old weights halved)
    pub fn save_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let mut prof: HashMap<(usize, usize), f64> = read_profile(path).unwrap_or_default();
        for v in prof.values_mut() {
            *v *= 0.5;
        }
        let mut n = 0;
        for l in 0..self.layers.len() {
            for x in 0..128 {
                let u = self.usage[l * 128 + x].load(Relaxed);
                if u > 0 {
                    *prof.entry((l, x)).or_insert(0.0) += u as f64;
                    n += 1;
                }
            }
        }
        if n == 0 {
            return Ok(0);
        }
        let mut v: Vec<_> = prof.into_iter().filter(|(_, w)| *w >= 0.5).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = String::from("# nextsycl expert profile: layer expert weight (decode requests, older halved each save)\n");
        for ((l, x), w) in &v {
            out += &format!("{l} {x} {w:.1}\n");
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, path)?;
        Ok(v.len())
    }

    /// Profile lap: waits for the GPU and adds the time since `t` to layer `l`'s category `cat` (no-op unless NS_PROFILE=1)
    fn lap(&self, cd: &Card, l: usize, cat: usize, t: &mut Instant) {
        if self.pf {
            let _ = cd.gpu.sync();
            let w = self.cur_wide.load(Relaxed) as usize;
            self.lay.lock().unwrap()[w][l][cat] += t.elapsed().as_secs_f64();
            *t = Instant::now();
        }
    }

    // ---- kernels

    fn rms(&self, cd: &Card, x: &DevBuf, w: &DevBuf, y: &DevBuf, rows: usize, c: usize) -> Result<()> {
        // SAFETY: x, y hold rows * c floats and w c floats (the callers' sizes).
        let rc = unsafe { (self.k.rms_norm)(cd.gpu.raw(), x.fp(), w.fp(), y.fp(), rows as i64, c as i64, self.g.rms_eps as f32) };
        ffi::check(rc, "rms_norm")
    }

    fn add(&self, cd: &Card, y: &DevBuf, x: &DevBuf, n: usize) -> Result<()> {
        // SAFETY: both hold n floats.
        ffi::check(unsafe { (self.k.add)(cd.gpu.raw(), y.fp(), x.fp(), n as i64) }, "add")
    }

    /// y [t, rows] = x [t, cols] . W^T: the matrix dequantized to float32 a chunk of rows at a time
    fn matmul(&self, cd: &Card, w: &Mat, x: &DevBuf, t: usize, y: &DevBuf) -> Result<()> {
        if x.floats() < t * w.cols || y.floats() < t * w.rows {
            return Err(e(format!("matmul {}x{}: buffers of {} and {} floats for {t} rows", w.rows, w.cols, x.floats(), y.floats())));
        }
        let raw = cd.gpu.raw();
        if w.ty == GType::F32 {
            // SAFETY: sizes checked above; w holds rows * cols floats (loaded from a shaped tensor).
            let rc = unsafe { (self.k.gemm)(raw, t as i64, w.rows as i64, w.cols as i64, x.fp(), w.cols as i64, w.buf.fp(), y.fp(), w.rows as i64, 0) };
            return ffi::check(rc, "gemm");
        }
        if t <= 8 && self.mmvq_ok(w.ty, w.cols) {
            // decode width: the products straight from the stored blocks
            let m = cd.arena.mark();
            let xq = self.quantize(cd, x, t, w.cols)?;
            self.mmvq_at(cd, w.ty, w.buf.ptr().cast::<u8>().cast_const(), &xq, w.cols, w.rows, t, y)?;
            cd.arena.rewind(m);
            return Ok(());
        }
        let row_bytes = w.ty.bytes(w.cols as u64).ok_or_else(|| e("row bytes"))? as usize;
        let chunk = (24 << 20) / w.cols;
        let m = cd.arena.mark();
        let tmp = cd.arena.f32(chunk.min(w.rows) * w.cols)?;
        let mut r0 = 0;
        while r0 < w.rows {
            let nr = chunk.min(w.rows - r0);
            // SAFETY: the byte range [r0 * row_bytes, (r0 + nr) * row_bytes) is inside w.buf; tmp holds nr * cols floats;
            // y has the t * rows floats checked above.
            unsafe {
                let src = w.buf.ptr().cast::<u8>().add(r0 * row_bytes).cast();
                ffi::check((self.k.dequant)(raw, w.ty.code() as i32, src, nr * w.cols, tmp.fp()), "dequant")?;
                ffi::check((self.k.gemm)(raw, t as i64, nr as i64, w.cols as i64, x.fp(), w.cols as i64, tmp.fp(), y.fp().add(r0), w.rows as i64, 0), "gemm")?;
            }
            r0 += nr;
        }
        cd.arena.rewind(m);
        Ok(())
    }

    fn embed(&self, cd: &Card, tokens: &[u32]) -> Result<DevBuf> {
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
        let raw = DevBuf::new(&cd.gpu, host.len())?;
        raw.write(0, &host)?;
        let x = DevBuf::f32(&cd.gpu, tokens.len() * c.cols)?;
        // SAFETY: raw holds tokens * row bytes = tokens * cols values of the type; x tokens * cols floats.
        let rc = unsafe { (self.k.dequant)(cd.gpu.raw(), c.ty.code() as i32, raw.ptr().cast_const(), tokens.len() * c.cols, x.fp()) };
        ffi::check(rc, "dequant (embedding)")?;
        cd.gpu.sync()?;
        Ok(x)
    }

    // ---- the layer

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, cd: &Card, l: usize, ly: &Layer, s: &Session, xn: &DevBuf, t: usize, p0: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.g;
        let (h, kv, hd) = (g.n_head as usize, g.n_kv_head as usize, g.head_dim as usize);
        let a = &cd.arena;
        let raw = cd.gpu.raw();
        let mut tm = Instant::now();
        let q = a.f32(t * h * hd)?;
        let k = a.f32(t * kv * hd)?;
        let v = a.f32(t * kv * hd)?;
        self.matmul(cd, &ly.q, xn, t, &q)?;
        self.matmul(cd, &ly.k, xn, t, &k)?;
        self.matmul(cd, &ly.v, xn, t, &v)?;
        self.rms(cd, &q, &ly.qn, &q, t * h, hd)?;
        self.rms(cd, &k, &ly.kn, &k, t * kv, hd)?;
        tap(&format!("Qcur_normed-{l}"), &q)?;
        tap(&format!("Kcur_normed-{l}"), &k)?;
        let n_rot = g.n_rot as i64;
        // SAFETY: q and k hold the rows the extents name.
        unsafe {
            ffi::check((self.k.rope)(raw, q.fp(), t as i64, h as i64, (h * hd) as i64, hd as i64, n_rot, p0 as i64, g.rope_base as f32), "rope q")?;
            ffi::check((self.k.rope)(raw, k.fp(), t as i64, kv as i64, (kv * hd) as i64, hd as i64, n_rot, p0 as i64, g.rope_base as f32), "rope k")?;
        }
        tap(&format!("Qcur-{l}"), &q)?;
        tap(&format!("Kcur-{l}"), &k)?;
        // the caches, fp16
        let (kc, vc) = (&s.k[l], &s.v[l]);
        let off = p0 * kv * hd;
        kc.bounds(off * 2, t * kv * hd * 2)?;
        // SAFETY: the destination ranges are inside the caches (checked above for k; v has the same extent).
        unsafe {
            ffi::check((self.k.to_f16)(raw, k.fp(), kc.ptr().cast::<u16>().add(off), (t * kv * hd) as i64), "cache k")?;
            ffi::check((self.k.to_f16)(raw, v.fp(), vc.ptr().cast::<u16>().add(off), (t * kv * hd) as i64), "cache v")?;
        }
        self.lap(cd, l, 0, &mut tm);
        let out = a.f32(t * h * hd)?;
        let n_pos = p0 + t;
        let scale = 1.0 / (hd as f32).sqrt();
        match (&ly.idx, &s.ik[l]) {
            (Some(ix), Some(ic)) => {
                let (ih, il) = (g.indexer.n_head as usize, g.indexer.key_len as usize);
                let iq = a.f32(t * ih * il)?;
                let ik = a.f32(t * il)?;
                self.matmul(cd, &ix.q, xn, t, &iq)?;
                self.matmul(cd, &ix.k, xn, t, &ik)?;
                self.rms(cd, &iq, &ix.qn, &iq, t * ih, il)?;
                self.rms(cd, &ik, &ix.kn, &ik, t, il)?;
                // SAFETY: iq, ik hold the rows named; the cache range is checked just before the conversion.
                unsafe {
                    ffi::check((self.k.rope)(raw, iq.fp(), t as i64, ih as i64, (ih * il) as i64, il as i64, n_rot, p0 as i64, g.rope_base as f32), "rope iq")?;
                    ffi::check((self.k.rope)(raw, ik.fp(), t as i64, 1, il as i64, il as i64, n_rot, p0 as i64, g.rope_base as f32), "rope ik")?;
                    ic.bounds(p0 * il * 2, t * il * 2)?;
                    ffi::check((self.k.to_f16)(raw, ik.fp(), ic.ptr().cast::<u16>().add(p0 * il), (t * il) as i64), "cache ik")?;
                }
                let blk = g.indexer.block_size as usize;
                let nblk = n_pos.div_ceil(blk);
                let kk = (g.indexer.top_k as usize).min(nblk);
                let bs = a.f32(t * ih * nblk)?;
                let sel = a.bytes(t * ih * kk * 4)?;
                // SAFETY: bs holds t * ih * nblk floats, sel t * ih * kk ints, ic the keys of n_pos positions.
                unsafe {
                    ffi::check((self.k.idx_scores)(raw, iq.fp(), ic.ptr().cast::<u16>().cast_const(), bs.fp(), t as i64, ih as i64, nblk as i64, p0 as i64,
                                                   g.indexer.local_blocks as i64), "idx_scores")?;
                    tap(&format!("msa_bs-{l}"), &bs)?;
                    ffi::check((self.k.idx_topk)(raw, bs.fp(), sel.ptr().cast::<i32>(), t as i64, ih as i64, nblk as i64, kk as i64), "idx_topk")?;
                    self.lap(cd, l, 1, &mut tm);
                    match self.k.silo_attend_split {
                        Some(f) if t < 8 && h == kv * 16 && hd == 128 && !no_split() => {
                            let s = (kk * 128).div_ceil(128);
                            let part = a.f32(t * kv * s * 16 * 130)?;
                            ffi::check(f(raw, q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), sel.ptr().cast::<i32>().cast_const(), kk as i64,
                                         out.fp(), t as i64, h as i64, kv as i64, p0 as i64, scale, s as i64, part.fp()), "attend split")?;
                        }
                        _ => ffi::check((self.attend_fn(t, h, kv, hd))(raw, q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), sel.ptr().cast::<i32>().cast_const(),
                                                   kk as i64, out.fp(), t as i64, h as i64, kv as i64, p0 as i64, scale), "attend")?,
                    }
                }
            }
            _ => {
                // SAFETY: caches hold n_pos positions; out t * h * hd floats.
                unsafe {
                    match self.k.silo_attend_split {
                        Some(f) if t < 8 && h == kv * 16 && hd == 128 && !no_split() => {
                            let s = (p0 + t).div_ceil(128);
                            let part = a.f32(t * kv * s * 16 * 130)?;
                            ffi::check(f(raw, q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), std::ptr::null(), 0, out.fp(), t as i64, h as i64,
                                         kv as i64, p0 as i64, scale, s as i64, part.fp()), "attend split")?;
                        }
                        _ => ffi::check((self.attend_fn(t, h, kv, hd))(raw, q.fp(), kc.ptr().cast::<u16>().cast_const(), vc.ptr().cast::<u16>().cast_const(), std::ptr::null(), 0, out.fp(),
                                                   t as i64, h as i64, kv as i64, p0 as i64, scale), "attend")?,
                    }
                }
            }
        }
        self.lap(cd, l, 2, &mut tm);
        tap(&format!("kqv_out-{l}"), &out)?;
        let o = a.f32(t * g.n_embd as usize)?;
        self.matmul(cd, &ly.o, &out, t, &o)?;
        self.lap(cd, l, 0, &mut tm);
        Ok(o)
    }

    /// The attention kernel for `t` query rows: the silo's for a prompt pass (its tile loop needs rows to share), else the shared one
    fn attend_fn(&self, t: usize, h: usize, kv: usize, hd: usize) -> ffi::AttendFn {
        match self.k.silo_attend {
            Some(f) if t >= 8 && h == kv * 16 && hd == 128 => f,
            _ => self.k.attend,
        }
    }

    fn dense(&self, cd: &Card, d: &Dense, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let a = &cd.arena;
        let f = d.gate.rows;
        let gt = a.f32(t * f)?;
        let up = a.f32(t * f)?;
        self.matmul(cd, &d.gate, x, t, &gt)?;
        self.matmul(cd, &d.up, x, t, &up)?;
        // SAFETY: gt, up hold t * f floats.
        ffi::check(unsafe { (self.k.swiglu_oai)(cd.gpu.raw(), gt.fp(), up.fp(), gt.fp(), (t * f) as i64, 1.702, 7.0) }, "swiglu_oai")?;
        let out = a.f32(t * d.down.rows)?;
        self.matmul(cd, &d.down, &gt, t, &out)?;
        Ok(out)
    }

    /// The one-token form of `expert_compute`: the activation quantized once for the layer (`xq`), the expert's weight and rank
    /// read by the kernel from `meta` (ids, weights and ranks of the layer's picks, one upload), so an expert is five launches
    #[allow(clippy::too_many_arguments)]
    fn expert_compute1(&self, cd: &Card, m: &Moe, base: *const u8, xq: &DevBuf, ys: &DevBuf, plane: usize, meta: &DevBuf, used: usize, rank: usize) -> Result<()> {
        let c = self.g.n_embd as usize;
        let fe = m.gate.rows;
        let (off_up, off_down, _) = m.layout();
        let a = &cd.arena;
        let raw = cd.gpu.raw();
        let mk = a.mark();
        let gt = a.f32(fe)?;
        let up = a.f32(fe)?;
        let eo = a.f32(c)?;
        let mut tt = Instant::now();
        let mut lapx = |i: usize| {
            if self.pf && self.stage_ok.load(Relaxed) {
                let _ = cd.gpu.sync();
                self.stage_t.lock().unwrap()[i] += tt.elapsed().as_secs_f64();
                tt = Instant::now();
            }
        };
        self.mmvq_at(cd, m.gate.ty, base, xq, c, fe, 1, &gt)?;
        lapx(0);
        // SAFETY: base + off_up is inside the slot (total bytes).
        self.mmvq_at(cd, m.up.ty, unsafe { base.add(off_up) }, xq, c, fe, 1, &up)?;
        lapx(1);
        // SAFETY: gt, up hold fe floats.
        ffi::check(unsafe { (self.k.swiglu_oai)(raw, gt.fp(), up.fp(), gt.fp(), fe as i64, 1.702, 7.0) }, "swiglu_oai")?;
        let hq = self.quantize(cd, &gt, 1, fe)?;
        lapx(2);
        // SAFETY: base + off_down is inside the slot.
        self.mmvq_at(cd, m.down.ty, unsafe { base.add(off_down) }, &hq, fe, c, 1, &eo)?;
        lapx(3);
        // meta: `used` ids (all 0), `used` weights, `used` ranks; this pick's entry of each
        // SAFETY: the three arrays hold `used` entries each and rank < used; ys holds `used` planes.
        unsafe {
            let p = meta.ptr().cast::<u8>();
            let (ids, ws, rks) = (p.cast::<i32>().cast_const(), p.add(used * 4).cast::<f32>().cast_const(), p.add(used * 8).cast::<i32>().cast_const());
            ffi::check((self.k.scatter_rank)(raw, ys.fp(), eo.fp(), ids, ws.add(rank), rks.add(rank), 1, c as i64, plane as i64), "scatter_rank")?;
        }
        lapx(4);
        a.rewind(mk);
        Ok(())
    }

    /// `dense` for one token whose activation is already quantized (`xq`): the gate and up products straight from it
    fn dense_q(&self, cd: &Card, d: &Dense, x: &DevBuf, xq: &DevBuf) -> Result<DevBuf> {
        let a = &cd.arena;
        let f = d.gate.rows;
        let gt = a.f32(f)?;
        let up = a.f32(f)?;
        let c = d.gate.cols;
        self.mmvq_at(cd, d.gate.ty, d.gate.buf.ptr().cast::<u8>().cast_const(), xq, c, f, 1, &gt)?;
        self.mmvq_at(cd, d.up.ty, d.up.buf.ptr().cast::<u8>().cast_const(), xq, c, f, 1, &up)?;
        let _ = x;
        // SAFETY: gt, up hold f floats.
        ffi::check(unsafe { (self.k.swiglu_oai)(cd.gpu.raw(), gt.fp(), up.fp(), gt.fp(), f as i64, 1.702, 7.0) }, "swiglu_oai")?;
        let out = a.f32(d.down.rows)?;
        self.matmul(cd, &d.down, &gt, 1, &out)?;
        Ok(out)
    }

    /// One expert's three products for the `list` of (token, weight), added into `y` (queued on the card; no wait)
    #[allow(clippy::too_many_arguments)]
    fn expert_compute(&self, cd: &Card, m: &Moe, base: *const u8, x: &DevBuf, ys: &DevBuf, plane: usize, list: &[(i32, f32, i32)]) -> Result<()> {
        let c = self.g.n_embd as usize;
        let fe = m.gate.rows;
        let (off_up, off_down, _) = m.layout();
        let a = &cd.arena;
        let raw = cd.gpu.raw();
        let n = list.len();
        let mk = a.mark();
        let idx: Vec<u8> = list.iter().flat_map(|(i, _, _)| i.to_le_bytes()).collect();
        let wv: Vec<u8> = list.iter().flat_map(|(_, w, _)| w.to_le_bytes()).collect();
        let rk: Vec<u8> = list.iter().flat_map(|(_, _, r)| r.to_le_bytes()).collect();
        let ib = a.bytes(idx.len())?;
        ib.write_async(0, &idx)?;
        let wb = a.bytes(wv.len())?;
        wb.write_async(0, &wv)?;
        let rb = a.bytes(rk.len())?;
        rb.write_async(0, &rk)?;
        let eo = a.f32(n * c)?;
        let mmvq = n <= 8 && [&m.gate, &m.up, &m.down].iter().all(|w| self.mmvq_ok(w.ty, w.cols));
        let gt = a.f32(n * fe)?;
        if mmvq {
            // decode width: the products straight from the stored blocks against Q8_1 activations
            let xe = a.f32(n * c)?;
            // SAFETY: x holds t rows of c floats, the indices are tokens < t; xe holds n * c floats.
            ffi::check(unsafe { (self.k.gather)(raw, x.fp(), ib.ptr().cast::<i32>().cast_const(), xe.fp(), n as i64, c as i64) }, "gather")?;
            let xq = self.quantize(cd, &xe, n, c)?;
            let up = a.f32(n * fe)?;
            self.mmvq_at(cd, m.gate.ty, base, &xq, c, fe, n, &gt)?;
            // SAFETY: base + off_up is inside the slot (total bytes).
            self.mmvq_at(cd, m.up.ty, unsafe { base.add(off_up) }, &xq, c, fe, n, &up)?;
            // SAFETY: gt, up hold n * fe floats.
            ffi::check(unsafe { (self.k.swiglu_oai)(raw, gt.fp(), up.fp(), gt.fp(), (n * fe) as i64, 1.702, 7.0) }, "swiglu_oai")?;
            let hq = self.quantize(cd, &gt, n, fe)?;
            // SAFETY: base + off_down is inside the slot.
            self.mmvq_at(cd, m.down.ty, unsafe { base.add(off_down) }, &hq, fe, c, n, &eo)?;
        } else {
            let xe = a.bytes(n * c * 2)?;
            // SAFETY: x holds t rows of c floats and the indices are tokens < t; xe holds n * c halves.
            ffi::check(unsafe { (self.k.gather_f16)(raw, x.fp(), ib.ptr().cast::<i32>().cast_const(), xe.ptr().cast::<u16>(), n as i64, c as i64) }, "gather_f16")?;
            let up = a.f32(n * fe)?;
            self.half_product(cd, &m.gate, base, &xe, n, &gt)?;
            // SAFETY: inside the slot.
            self.half_product(cd, &m.up, unsafe { base.add(off_up) }, &xe, n, &up)?;
            // SAFETY: gt, up hold n * fe floats.
            ffi::check(unsafe { (self.k.swiglu_oai)(raw, gt.fp(), up.fp(), gt.fp(), (n * fe) as i64, 1.702, 7.0) }, "swiglu_oai")?;
            let h16 = a.bytes(n * fe * 2)?;
            // SAFETY: gt holds n * fe floats, h16 n * fe halves.
            ffi::check(unsafe { (self.k.to_f16)(raw, gt.fp(), h16.ptr().cast::<u16>(), (n * fe) as i64) }, "to_f16")?;
            // SAFETY: inside the slot.
            self.half_product(cd, &m.down, unsafe { base.add(off_down) }, &h16, n, &eo)?;
        }
        // SAFETY: ys holds `used` planes of t rows of c floats; eo n rows; idx, w and rank n entries, ranks below `used`.
        ffi::check(unsafe { (self.k.scatter_rank)(raw, ys.fp(), eo.fp(), ib.ptr().cast::<i32>().cast_const(), wb.fp(), rb.ptr().cast::<i32>().cast_const(), n as i64, c as i64, plane as i64) },
                   "scatter_rank")?;
        // the queue is in order: the scratch is free for the next expert's kernels
        a.rewind(mk);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn moe(&self, cd: &Card, l: usize, m: &Moe, x: &DevBuf, att: &DevBuf, nxt: Option<&Layer>, t: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.g;
        let (ne, used, c) = (g.n_expert as usize, g.n_expert_used as usize, g.n_embd as usize);
        let a = &cd.arena;
        let mut tm = Instant::now();
        let logits = a.f32(t * ne)?;
        self.matmul(cd, &m.router, x, t, &logits)?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        // a guess of the next layer's routing from this layer's residual stream (before this layer's feed-forward is added),
        // queued with this one so the read-back below carries both
        let guess = match nxt {
            Some(nl) => match &nl.ffn {
                Ffn::Moe(nm) => {
                    let pn = a.f32(c)?;
                    self.rms(cd, att, &nl.ffn_norm, &pn, 1, c)?;
                    let pl = a.f32(ne)?;
                    self.matmul(cd, &nm.router, &pn, 1, &pl)?;
                    Some((pl, nm))
                }
                Ffn::Dense(_) => None,
            },
            None => None,
        };
        let lg = logits.to_f32()?;
        let guess = match guess {
            Some((pl, nm)) => Some((pl.to_f32()?, nm)),
            None => None,
        };
        // routing on the host: sigmoid probabilities, the bias added to choose, the chosen probabilities renormalized and scaled
        let mut per: Vec<Vec<(i32, f32, i32)>> = vec![Vec::new(); ne];
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
            for (r, &i) in pick.iter().enumerate() {
                let w = probs[i] / sum * g.expert_scale;
                wts.push(w);
                per[i].push((tok as i32, w, r as i32));
            }
        }
        let wbuf = DevBuf::from_f32(&cd.gpu, &wts)?;
        tap(&format!("ffn_moe_weights_scaled-{l}"), &wbuf)?;
        self.lap(cd, l, 3, &mut tm);
        // one plane a rank: every (rank, token) row is written by exactly one expert, and summed in rank order below
        let plane = t * c;
        let ys = a.f32(used * plane)?;
        let li = l - cd.layers.start;
        let need: Vec<usize> = (0..ne).filter(|&i| !per[i].is_empty()).collect();
        let (mut hot, mut miss): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for &ex in &need {
            match &cd.res[li][ex] {
                Some(r) => {
                    hot.push(ex);
                    let ctr = if r.vram { &self.counts.vram } else { &self.counts.pinned };
                    ctr.fetch_add(1, Relaxed);
                }
                None => miss.push(ex),
            }
            if t <= 8 {
                self.usage[l * 128 + ex].fetch_add(1, Relaxed);
            }
        }
        self.counts.file.fetch_add(miss.len() as u64, Relaxed);
        // experts of this layer read ahead during the one before: taken from the table; the rest wait for their wave
        let mut pre: Vec<(usize, usize, Reads)> = Vec::new();
        {
            let mut pfs = cd.pfs.lock().unwrap();
            let wanted = miss.len();
            miss.retain(|&ex| match pfs.map.remove(&(l, ex)) {
                Some((slot_i, hs)) => {
                    pre.push((ex, slot_i, hs));
                    false
                }
                None => true,
            });
            self.counts.ahead_wanted.fetch_add(wanted as u64, Relaxed);
            self.counts.ahead_used.fetch_add(pre.len() as u64, Relaxed);
        }
        // one token: the activation quantized once for the shared expert and every pick, the picks' ids / weights / ranks in one upload
        let one = if t == 1 && [&m.gate, &m.up, &m.down].iter().all(|w| self.mmvq_ok(w.ty, w.cols)) && self.mmvq_ok(m.shared.gate.ty, m.shared.gate.cols)
            && self.mmvq_ok(m.shared.up.ty, m.shared.up.cols) {
            let xq = self.quantize(cd, x, 1, c)?;
            let meta = a.bytes(used * 12)?;
            let mut mb = Vec::with_capacity(used * 12);
            mb.extend(std::iter::repeat_n(0i32, used).flat_map(|v| v.to_le_bytes()));
            mb.extend(wts.iter().flat_map(|w| w.to_le_bytes()));
            mb.extend((0..used as i32).flat_map(|r| r.to_le_bytes()));
            meta.write_async(0, &mb)?;
            Some((xq, meta))
        } else {
            None
        };
        let run = |base: *const u8, ex: usize| -> Result<()> {
            match &one {
                Some((xq, meta)) => self.expert_compute1(cd, m, base, xq, &ys, plane, meta, used, per[ex][0].2 as usize),
                None => self.expert_compute(cd, m, base, x, &ys, plane, &per[ex]),
            }
        };
        let slot = cd.slot;
        let total = m.layout().2;
        let mut used_slots: Vec<usize> = Vec::new();
        // the staging ring: wave k reads into slots [(k % 2) * WAVE, +WAVE)
        let ring = cd.stage.device_view(0, cd.stage.len)?;
        let waves: Vec<&[usize]> = miss.chunks(WAVE).collect();
        let issue = |k: usize| -> Reads {
            let mut hs = Reads::new();
            for (j, &ex) in waves[k].iter().enumerate() {
                // SAFETY: the ring holds 2 * WAVE slots of `slot` >= an expert's bytes.
                let dst = unsafe { ring.ptr().cast::<u8>().add(((k % 2) * WAVE + j) * slot) };
                hs.extend(spawn_reads(&self.reader, m.src(ex), dst));
            }
            hs
        };
        let tr = Instant::now();
        let mut wait = 0.0;
        let mut inflight: Option<Reads> = (!waves.is_empty()).then(|| issue(0));
        // the shared expert and the experts already in VRAM run while the file is read and the pinned ones are copied
        let sh = match &one {
            Some((xq, _)) => self.dense_q(cd, &m.shared, x, xq)?,
            None => self.dense(cd, &m.shared, x, t)?,
        };
        // the experts that need a copy into VRAM (pinned, read ahead, read in a wave), in the order they run; a copy runs on the
        // copy lane beside the compute queue (tickets order the two on the device), into a ring of NX VRAM slots
        enum Job {
            Pinned(usize),
            Pre(usize, usize, Option<Reads>),
            Wave(usize, usize, usize),
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut vram_hot: Vec<usize> = Vec::new();
        for &ex in &hot {
            if cd.res[li][ex].as_ref().unwrap().vram {
                vram_hot.push(ex);
            } else {
                jobs.push(Job::Pinned(ex));
            }
        }
        for (ex, slot_i, hs) in pre {
            jobs.push(Job::Pre(ex, slot_i, Some(hs)));
        }
        for (k, w) in waves.iter().enumerate() {
            for (j, &ex) in w.iter().enumerate() {
                jobs.push(Job::Wave(ex, k, j));
            }
        }
        let njobs = jobs.len();
        let mut tickets: Vec<i64> = Vec::with_capacity(njobs); // a job's compute ticket, once queued
        let mut copy_t: Vec<i64> = Vec::with_capacity(njobs); // a job's copy ticket
        let mut joined = 0usize; // waves whose reads are joined
        let lane = copy_lane();
        let issue_copy = |jobs: &mut Vec<Job>, j: usize, tickets: &[i64], copy_t: &mut Vec<i64>, joined: &mut usize, inflight: &mut Option<Reads>, wait: &mut f64| -> Result<()> {
            let src: DevBuf = match &mut jobs[j] {
                Job::Pinned(ex) => {
                    let r = cd.res[li][*ex].as_ref().unwrap();
                    r.buf.view(0, total)?
                }
                Job::Pre(_, slot_i, hs) => {
                    let tw = Instant::now();
                    join_all(hs.take().unwrap())?;
                    *wait += tw.elapsed().as_secs_f64();
                    ring.view((2 * WAVE + *slot_i) * slot, total)?
                }
                Job::Wave(_, k, jj) => {
                    while *joined <= *k {
                        let tw = Instant::now();
                        join_all(inflight.take().unwrap())?;
                        *wait += tw.elapsed().as_secs_f64();
                        if *joined >= 1 {
                            // wave joined-1's products are done: its half of the ring is free for the reads after
                            cd.gpu.sync()?;
                        }
                        *joined += 1;
                        if *joined < waves.len() {
                            *inflight = Some(issue(*joined));
                        }
                    }
                    ring.view(((*k % 2) * WAVE + *jj) * slot, total)?
                }
            };
            let tgt = cd.xfer.view((j % NX) * slot, total)?;
            let after = if j >= NX { Some(tickets[j - NX]) } else { None };
            let ct = ffi::stream_copy(cd.gpu.raw(), &tgt, &src, total, after, lane)?;
            copy_t.push(ct);
            Ok(())
        };
        let mut issued = 0usize;
        // the first copies go out before the VRAM experts run, so the copy lane is busy during them
        while issued < njobs && issued < LA && !matches!(jobs[issued], Job::Wave(..)) {
            issue_copy(&mut jobs, issued, &tickets, &mut copy_t, &mut joined, &mut inflight, &mut wait)?;
            issued += 1;
        }
        for &ex in &vram_hot {
            let r = cd.res[li][ex].as_ref().unwrap();
            run(r.buf.ptr().cast::<u8>().cast_const(), ex)?;
        }
        // read ahead for the next layer now: its reads run during this layer's products
        if let (Some((pg, nm)), true) = (&guess, ahead() > 0) {
            let nli = l + 1 - cd.layers.start;
            let probs: Vec<f32> = pg.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&i, &j| (probs[j] + nm.bias[j]).total_cmp(&(probs[i] + nm.bias[i])).then(i.cmp(&j)));
            let mut pfs = cd.pfs.lock().unwrap();
            for &ex in order.iter().take(ahead()) {
                if cd.res[nli][ex].is_some() || pfs.map.contains_key(&(l + 1, ex)) {
                    continue;
                }
                let Some(slot_i) = pfs.free.pop() else { break };
                // SAFETY: slot_i of the read-ahead region of the stage, past the 2 * WAVE ring slots.
                let dst = unsafe { ring.ptr().cast::<u8>().add((2 * WAVE + slot_i) * slot) };
                pfs.map.insert((l + 1, ex), (slot_i, spawn_reads(&self.reader, nm.src(ex), dst)));
                self.counts.ahead.fetch_add(1, Relaxed);
            }
        }
        for i in 0..njobs {
            while issued <= i {
                issue_copy(&mut jobs, issued, &tickets, &mut copy_t, &mut joined, &mut inflight, &mut wait)?;
                issued += 1;
            }
            // the compute queue waits, on the device, for the copy
            ffi::await_ticket(cd.gpu.raw(), copy_t[i])?;
            let (ex, pre_slot) = match &jobs[i] {
                Job::Pinned(ex) => (*ex, None),
                Job::Pre(ex, s, _) => (*ex, Some(*s)),
                Job::Wave(ex, ..) => (*ex, None),
            };
            run(cd.xfer.view((i % NX) * slot, total)?.ptr().cast::<u8>().cast_const(), ex)?;
            tickets.push(ffi::mark(cd.gpu.raw())?);
            if let Some(s) = pre_slot {
                used_slots.push(s);
            }
            // the copies after this one, while it computes: the next may wait for its reads, the ones after only when no read is due
            while issued < njobs && issued <= i + LA {
                if issued > i + 1 && !matches!(jobs[issued], Job::Pinned(_)) {
                    break;
                }
                issue_copy(&mut jobs, issued, &tickets, &mut copy_t, &mut joined, &mut inflight, &mut wait)?;
                issued += 1;
            }
        }
        // the ring is read by queued kernels until they finish: the next layer's reads must not start before
        cd.gpu.sync()?;
        {
            // slots of this layer's read-ahead come back (guesses nobody used: their reads waited for first)
            let mut pfs = cd.pfs.lock().unwrap();
            pfs.free.extend(used_slots);
            let stale: Vec<(usize, usize)> = pfs.map.keys().filter(|k| k.0 <= l).copied().collect();
            for k in stale {
                if let Some((slot_i, hs)) = pfs.map.remove(&k) {
                    let _ = join_all(hs);
                    pfs.free.push(slot_i);
                }
            }
        }
        if self.pf {
            let w = self.cur_wide.load(Relaxed) as usize;
            let mut lay = self.lay.lock().unwrap();
            lay[w][l][4] += wait;
            lay[w][l][5] += tr.elapsed().as_secs_f64() - wait;
        }
        tm = Instant::now();
        let y = ys.view(0, plane * 4)?;
        for r in 1..used {
            self.add(cd, &y, &ys.view(r * plane * 4, plane * 4)?, plane)?;
        }
        tap(&format!("ffn_moe_out-{l}"), &y)?;
        tap(&format!("ffn_shexp-{l}"), &sh)?;
        self.add(cd, &y, &sh, t * c)?;
        self.lap(cd, l, 6, &mut tm);
        Ok(y)
    }

    /// Whether a matrix of this type and width has a decode-width kernel
    fn mmvq_ok(&self, ty: GType, cols: usize) -> bool {
        // SAFETY: plain query.
        cols.is_multiple_of(256) && ty != GType::F32 && unsafe { (self.k.mmvq_supported)(ty.code() as i32) } != 0
    }

    /// x [n, cols] as Q8_1 blocks (n at most 8 columns)
    fn quantize(&self, cd: &Card, x: &DevBuf, n: usize, cols: usize) -> Result<DevBuf> {
        // SAFETY: plain size query.
        let bytes = unsafe { (self.k.q8_1_bytes)(cols as i64, n as i64) };
        let q = cd.arena.bytes(bytes)?;
        // SAFETY: x holds n * cols floats (the callers'), q the blocks.
        ffi::check(unsafe { (self.k.quantize_q8_1)(cd.gpu.raw(), x.fp(), q.ptr(), cols as i64, n as i64) }, "quantize_q8_1")?;
        Ok(q)
    }

    /// y [n, rows] = the matrix at `w` (stored blocks of `ty`, `rows` rows of `cols`) . the n columns of `xq`
    #[allow(clippy::too_many_arguments)]
    fn mmvq_at(&self, cd: &Card, ty: GType, w: *const u8, xq: &DevBuf, cols: usize, rows: usize, n: usize, y: &DevBuf) -> Result<()> {
        y.bounds(0, n * rows * 4)?;
        // SAFETY: w is a device matrix of rows * cols values (the callers' slices), xq the n columns, y checked.
        let rc = unsafe { (self.k.mmvq)(cd.gpu.raw(), ty.code() as i32, w.cast(), xq.ptr().cast_const(), y.fp(), cols as i64, rows as i64, n as i64) };
        ffi::check(rc, "mmvq")
    }

    /// y [n, rows] = x16 [n, cols] . W^T for a matrix of stored blocks at `wp`: dequantized to fp16 on the GPU, multiplied by oneMKL
    fn half_product(&self, cd: &Card, w: &Experts, wp: *const u8, x16: &DevBuf, n: usize, y: &DevBuf) -> Result<()> {
        let m = cd.arena.mark();
        let nv = w.rows * w.cols;
        let w16 = cd.arena.bytes(nv * 2)?;
        // SAFETY: wp holds the matrix's rows * cols values in the type; w16 holds as many halves; x16 n * cols, y n * rows.
        unsafe {
            ffi::check((self.k.dequant_f16)(cd.gpu.raw(), w.ty.code() as i32, wp.cast(), nv as i64, w16.ptr().cast::<u16>()), "dequant_f16")?;
            ffi::check((self.k.gemm_f16)(cd.gpu.raw(), n as i64, w.rows as i64, w.cols as i64, x16.ptr().cast::<u16>().cast_const(), w.cols as i64,
                                         w16.ptr().cast::<u16>().cast_const(), y.fp(), w.rows as i64, 0), "gemm_f16")?;
        }
        cd.arena.rewind(m);
        Ok(())
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
        self.cur_wide.store(t > 1, Relaxed);
        let c = self.g.n_embd as usize;
        let p0 = s.pos;
        let pass_t0 = Instant::now();
        let mut x = self.embed(&self.cards[0], tokens)?;
        tap("inp_embd", &x)?;
        let mut cur = 0usize;
        for (l, ly) in self.layers.iter().enumerate() {
            let ci = self.owner[l];
            let cd = &self.cards[ci];
            if ci != cur {
                // the hidden state hops to the next card
                let nx = DevBuf::f32(&cd.gpu, t * c)?;
                let mut stage = vec![0u8; t * c * 4];
                nx.copy_from_peer(&x, &mut stage)?;
                x = nx;
                cur = ci;
            }
            let m0 = cd.arena.mark();
            let mut tm = Instant::now();
            let xn = cd.arena.f32(t * c)?;
            self.rms(cd, &x, &ly.attn_norm, &xn, t, c)?;
            tap(&format!("attn_norm-{l}"), &xn)?;
            self.lap(cd, l, 7, &mut tm);
            let att = self.attention(cd, l, ly, s, &xn, t, p0, &mut *tap)?;
            tm = Instant::now();
            self.add(cd, &att, &x, t * c)?;
            tap(&format!("ffn_inp-{l}"), &att)?;
            let hn = cd.arena.f32(t * c)?;
            self.rms(cd, &att, &ly.ffn_norm, &hn, t, c)?;
            tap(&format!("ffn_norm-{l}"), &hn)?;
            self.lap(cd, l, 7, &mut tm);
            let f = match &ly.ffn {
                Ffn::Dense(d) => {
                    let o = self.dense(cd, d, &hn, t)?;
                    self.lap(cd, l, 6, &mut tm);
                    o
                }
                Ffn::Moe(m) => {
                    let nxt = (t == 1 && l + 1 < self.layers.len() && self.owner[l + 1] == ci).then(|| &self.layers[l + 1]);
                    let o = self.moe(cd, l, m, &hn, &att, nxt, t, &mut *tap)?;
                    tm = Instant::now();
                    o
                }
            };
            tap(&format!("ffn_out-{l}"), &f)?;
            self.add(cd, &f, &att, t * c)?;
            tap(&format!("l_out-{l}"), &f)?;
            let nx = DevBuf::f32(&cd.gpu, t * c)?;
            nx.copy_within(0, &f, 0, t * c * 4)?;
            cd.gpu.sync()?;
            x = nx;
            cd.arena.rewind(m0);
            self.lap(cd, l, 7, &mut tm);
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
        let cd = self.cards.last().unwrap();
        let xr = x.view((t - n_out) * c * 4, n_out * c * 4)?;
        let xn = DevBuf::f32(&cd.gpu, n_out * c)?;
        self.rms(cd, &xr, &self.out_norm, &xn, n_out, c)?;
        tap("result_norm", &xn)?;
        let logits = DevBuf::f32(&cd.gpu, n_out * self.head.rows)?;
        self.matmul(cd, &self.head, &xn, n_out, &logits)?;
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
        for cd in &self.cards {
            let mine = &self.layers[cd.layers.clone()];
            let n_sparse = mine.iter().filter(|l| l.idx.is_some()).count();
            let need = (mine.len() * 2 * max_ctx * kv * hd * 2 + n_sparse * max_ctx * il * 2) as u64;
            let (total, free) = cd.gpu.memory()?;
            let free = free.unwrap_or(total);
            if need + (1536 << 20) > free {
                return Err(e(format!("{}: a session of {max_ctx} tokens needs {:.2} GiB; {:.2} GiB are free (1.5 GiB kept spare)", cd.gpu.name, need as f64 / GIB as f64,
                                     free as f64 / GIB as f64)));
            }
        }
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut ik = Vec::new();
        for (l, ly) in self.layers.iter().enumerate() {
            let gpu = &self.cards[self.owner[l]].gpu;
            k.push(DevBuf::new(gpu, max_ctx * kv * hd * 2)?);
            v.push(DevBuf::new(gpu, max_ctx * kv * hd * 2)?);
            ik.push(if ly.idx.is_some() { Some(DevBuf::new(gpu, max_ctx * il * 2)?) } else { None });
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
        for cd in &self.cards {
            cd.gpu.sync()?;
        }
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
