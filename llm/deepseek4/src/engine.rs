//! DeepSeek-V4-Flash on one or more GPUs: every step of `model.rs`'s outline in order, float32 activations. The layers
//! are split over the GPUs (a `Part` each: its layers' weights in their stored form, its own expert cache in VRAM and
//! its own share of the pinned host mirror); the token stream [T, 4, hidden] crosses to the next GPU through host
//! memory where the layers do. Up to 8 rows multiply from the stored blocks (decode), more are expanded in chunks.
//!
//! The framework is the glm5next engine's (its mHC, expert store, MoE paths, prompt pipeline and decode loop, read
//! the same way); the attention is this model's own: a ring of the last raw keys per layer, the compressed blocks'
//! caches (and the indexer's) built by the compressors as their blocks complete, ds4.h's attention over both.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};

use crate::model::{Model, Role};
use crate::ops::Ops;
use nextsycl_gguf::GType;

use nextsycl_llm::{GpuInfo, Sampler, Tap};

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;

/// Time the host spent waiting for the routers' logits (the GPU finishing the layer up to them), all layers
pub static ROUTER_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Time the host spent choosing each row's experts (all layers)
pub static ROUTE_HOST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Prompt chunks of this many tokens or more multiply their experts in fp16 on the XMX units (NS_PROMPT_F16_MIN,
/// default 1024; 0 = never)
/// Dense products of this many rows or more in fp16 on the XMX units, when the fp16 expander takes their type
/// (NS_DS4_DENSE_F16_MIN, default 64; 0 = never): the attention block's 512-row q_b and o_a take it (B70 + B65, a
/// 3.7K prompt: 447 -> 526 tok/s against the experts' 1024)
fn dense_f16_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS4_DENSE_F16_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64))
}

/// Query rows the attention block takes at a time in a prompt pass (NS_DS4_ATT_ROWS, default 512)
fn att_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS4_ATT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(ATT_ROWS).clamp(64, 8192))
}

fn f16_min() -> usize {
    static F16_MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F16_MIN.get_or_init(|| std::env::var("NS_PROMPT_F16_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1024))
}

/// The widest dense matrix input the fp16 path takes (o_b: 8 groups of 1024)
const X16_COLS: usize = 8192;

/// Experts swapped in ahead of their layer, at most, a layer (NS_PREFETCH, default 8; 0 = none)
fn prefetch_limit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFETCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8))
}

/// The prompt path's experts with at most this many tokens take the fused gate | up kernel (NS_FUSED_MAX, default 64)
fn fused_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_FUSED_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(64))
}

/// Whether a prompt of several chunks on two GPUs runs as a pipeline (NS_PIPELINE=0: in turn)
fn pipeline() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PIPELINE").map_or(true, |v| v != "0"))
}

/// NS_VRAM_GUARD_GIB: a prompt read stops with an error once a GPU has less VRAM free than this after a chunk
fn vram_guard() -> Option<u64> {
    static V: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_VRAM_GUARD_GIB").ok().and_then(|v| v.parse::<f64>().ok()).map(|g| (g * (1u64 << 30) as f64) as u64))
}

fn check_vram(gpu: &Gpu) -> Result<()> {
    if let Some(g) = vram_guard() {
        if let Some(free) = gpu.memory()?.1 {
            if free < g {
                return Err(Error(format!("{}: {:.2} GiB of VRAM free, under the guard (NS_VRAM_GUARD_GIB)", gpu.name, gib(free as usize))));
            }
        }
    }
    Ok(())
}

/// `tokens` in chunks of `c`, the last two split evenly when the last would be under half a chunk
fn chunked(tokens: &[u32], c: usize) -> Vec<&[u32]> {
    let mut v: Vec<&[u32]> = tokens.chunks(c).collect();
    if v.len() >= 2 && v[v.len() - 1].len() < c / 2 {
        let n = v.len();
        let start = (n - 2) * c;
        let both = &tokens[start..];
        let half = both.len().div_ceil(2);
        v.truncate(n - 2);
        v.push(&both[..half]);
        v.push(&both[half..]);
    }
    v
}

/// NS_ESIMD_MAX: a cap on the tokens of an expert the ESIMD gate | up takes (the GPU's own limit otherwise)
fn esimd_max() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_ESIMD_MAX").ok().and_then(|v| v.parse().ok()))
}

/// The expert profile (NS_EXPERT_PROFILE: "layer expert weight" lines), read once
fn expert_profile() -> Option<&'static HashMap<(u64, u64), f64>> {
    static V: std::sync::OnceLock<Option<HashMap<(u64, u64), f64>>> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("NS_EXPERT_PROFILE").ok().and_then(|p| read_profile(std::path::Path::new(&p)))).as_ref()
}

fn read_profile(path: &std::path::Path) -> Option<HashMap<(u64, u64), f64>> {
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

/// The rows the fused kernel computes for n tokens: to 32, above 64 to 64 (its wide form)
fn fused_rows(n: usize) -> usize {
    n.next_multiple_of(if n > 64 { 64 } else { 32 })
}

/// Whether an arriving expert's gate | up runs before its down is in (NS_SPLIT_COPY=0: no)
fn split_copy() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_SPLIT_COPY").map_or(true, |v| v != "0"))
}

/// Whether decode's swaps wait for the GPU's queued work first (NS_SWAP_WAIT=1; not needed - see `ensure`)
fn swap_wait() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_SWAP_WAIT").is_ok_and(|v| v == "1"))
}

/// The chance of use from which a guessed expert is fetched ahead (NS_PREFETCH_P, default 0.6)
fn prefetch_p() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFETCH_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6))
}

/// How often the next layer's router, run on this layer's input, has its k-th best expert among the ones the next
/// layer asks for (GLM-5.3's measurement, the same kind of router: a first guess here)
const GUESS_HIT: [f32; 16] = [0.96, 0.92, 0.86, 0.77, 0.66, 0.56, 0.45, 0.36, 0.29, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08];

/// The arena of decode and short prompt chunks; past it, a pass takes the big one back from the expert store
const SMALL_ARENA: usize = 384 << 20;

/// Whether the big arena's memory holds experts while no prompt chunk needs it (NS_LEND=0: it does not)
fn lend_arena() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_LEND").map_or(true, |v| v != "0"))
}

/// Prompt-lookup drafts (NS_NGRAM=K, 1..; 0 = off): when the last 3 tokens occurred earlier in the conversation, the
/// up to K tokens that followed them there are verified in one pass
pub fn ngram_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_NGRAM").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(MAX_VERIFY - 1))
}

/// Whether the dense matrices take the fp16 path too (NS_DENSE_F16=0: only the experts)
fn dense_f16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NS_DENSE_F16").map_or(true, |v| v != "0"))
}

/// The ggml types the fp16 expander (ns_dequant_f16) takes
const F16_TYPES: [u32; 16] = [6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 20, 21, 22, 23, 29, 42];
/// MXFP4: this engine's own kernels (ds4.h)
const MXFP4: u32 = 39;
/// the decode kernels take up to this many rows (tokens) at once
const MMVQ_COLS: usize = 8;
/// widest matrix input (the mHC mixes read 4 x 4096)
const MAX_COLS: usize = 16384;
/// expert slots and the mirror are allocated in chunks of this size (single allocations stay small)
const CHUNK: usize = 2 << 30;
/// Prompt tokens per forward pass (NS_PREFILL_CHUNK, default 4096, 64..8000)
pub fn prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096).clamp(64, 8000))
}

/// Bytes of each GPU's arena (NS_ARENA_MIB; default: 1 GiB, or ~0.66 MiB a token of the prompt chunk)
pub fn arena_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("NS_ARENA_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) => m << 20,
        None => (1usize << 30).max(prefill_chunk() * (680 << 10)),
    })
}
/// rows of a verify pass (the token and its drafts)
pub const MAX_VERIFY: usize = 8;
/// the queries' side of a prompt chunk's attention a block of rows at a time (q and its output are 128 KiB a row; B70 +
/// B65, a 3.7K prompt: 512 rows 550 tok/s, 2048 596 - q_b and o_a in bigger products, o_a expanded fewer times)
const ATT_ROWS: usize = 2048;
/// the indexer's scores per GEMM (floats): scored in chunks that fit (64 MiB)
const IDX_S_FLOATS: usize = 16 << 20;

/// The raw keys' ring: a chunk and the window before it, and a verify pass rolled back
pub fn raw_ring() -> usize {
    prefill_chunk() + 128 + MAX_VERIFY
}

/// The compressor's ring of the last tokens' projections: the blocks being built, and a verify pass rolled back
fn comp_ring(ratio: usize) -> usize {
    if ratio == 4 { 2 * ratio + MAX_VERIFY } else { ratio + MAX_VERIFY }
}

/// A matrix on the GPU: `rows` x `cols` (a tensor's outer dimensions folded into rows), in its stored quantized
/// form, or - for the 16- and 32-bit ones - as float32.
struct Mat {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
    /// expanded at load (BF16 / F16 / F32 matrices: small, multiplied as they are)
    f32: bool,
}

impl Mat {
    fn row_bytes(&self) -> usize {
        self.ty.bytes(self.cols as u64).unwrap_or(0) as usize
    }
    /// rows [r0, r0 + n) as a matrix of their own
    fn rows_view(&self, r0: usize, n: usize) -> Result<Mat> {
        let rb = if self.f32 { self.cols * 4 } else { self.row_bytes() };
        Ok(Mat { buf: self.buf.view(r0 * rb, n * rb)?, ty: self.ty, rows: n, cols: self.cols, f32: self.f32 })
    }
}

/// Where a layer's [gate | up | down] sit in a slot: (byte offset, bytes, type) each.
#[derive(Clone, Copy)]
struct ExpertParts {
    gate: (usize, usize, GType),
    up: (usize, usize, GType),
    down: (usize, usize, GType),
}

/// Decode's dense products of one input through the multi-matrix kernel, a launch a type (NS_DS4_MULTI_DENSE=0: the
/// shared kernels, a launch a matrix)
fn multi_dense() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS4_MULTI_DENSE").map_or(true, |v| v != "0"))
}

/// NS_PROFILE_SPLIT=1: the profile's decode passes apart from the prompt's
fn profile_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PROFILE_SPLIT").is_ok_and(|v| v == "1"))
}

/// "decode: NAME", made once a name
fn decode_name(name: &'static str) -> &'static str {
    static NAMES: Mutex<BTreeMap<&'static str, &'static str>> = Mutex::new(BTreeMap::new());
    let mut m = NAMES.lock().unwrap();
    m.entry(name).or_insert_with(|| Box::leak(format!("decode: {name}").into_boxed_str()))
}

/// A section's device stamps: (name, its GPU, start, end)
type Stamp = (&'static str, Arc<Gpu>, i64, i64);

/// A profiling mark: a host time (the GPU synced) or a device stamp's ticket
#[derive(Clone, Copy)]
enum Mk {
    Host(Instant),
    Gpu(i64),
}

/// Where an expert is.
#[derive(Clone, Copy)]
enum Loc {
    /// a VRAM slot
    V(usize),
    /// a pinned host memory slot
    R(usize),
}

/// The routed experts of one GPU's layers, in an exclusive two-level cache: VRAM slots and pinned host memory
/// slots hold different experts (both filled from the file at load, VRAM first); a VRAM miss swaps - the least
/// recently used VRAM expert goes down to a free host slot (D2H), the wanted one comes up (H2D) and frees its host
/// slot. What neither holds is read from the file on demand.
struct Store {
    vram: Vec<DevBuf>,
    host: Vec<HostBuf>,
    slot_bytes: usize,
    per_chunk: usize,
    loc: HashMap<(u64, u64), Loc>,
    vowner: Vec<Option<(u64, u64)>>,
    vused: Vec<u64>,
    /// host slots not holding an expert
    rfree: Vec<usize>,
    tick: u64,
    hits: u64,
    misses: u64,
    from_host: u64,
    /// host-slot experts a prompt pass reads from pinned memory
    direct: u64,
    /// per host slot, the copy down that last wrote it, and the last copy down of all
    rwrite: HashMap<usize, i64>,
    down_any: Option<i64>,
    /// per VRAM slot, the copy up that last filled it; the last copy up
    vfill: HashMap<usize, i64>,
    last_up: Option<i64>,
    /// experts swapped in ahead of their layer, not yet asked for: their copy up's tickets
    prefetched: HashMap<(u64, u64), (i64, i64)>,
    /// decode's requests per expert since load
    usage: HashMap<(u64, u64), u64>,
    /// the VRAM slots of its own (`base`), then `lend` more in the big arena's memory
    base: usize,
    lend: usize,
    lent: bool,
    pf_issued: u64,
    pf_used: u64,
}

impl Store {
    fn vat(&self, s: usize) -> (usize, usize) {
        if s >= self.base {
            return (self.vram.len() - 1, (s - self.base) * self.slot_bytes);
        }
        (s / self.per_chunk, (s % self.per_chunk) * self.slot_bytes)
    }
    fn usable(&self, s: usize) -> bool {
        s < self.base || self.lent
    }
    fn rat(&self, r: usize) -> (usize, usize) {
        (r / self.per_chunk, (r % self.per_chunk) * self.slot_bytes)
    }
}

/// One GPU's share of the model.
pub struct Part {
    ops: Ops,
    pub arena: Arena,
    big: DevBuf,
    small: DevBuf,
    /// the routers' biases, kept on the host once read
    biases: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
    pub layers: Range<u64>,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    /// o_a's groups as matrices of their own (rows [g * o_lora, (g + 1) * o_lora))
    groups: BTreeMap<u64, Vec<Mat>>,
    scratch: DevBuf,
    x16: Option<DevBuf>,
    /// Q8_1 of up to MMVQ_COLS rows of MAX_COLS
    q8: DevBuf,
    experts: Mutex<Store>,
    esimd_gu: usize,
    hold: std::sync::atomic::AtomicBool,
    pub expert_slots: usize,
    pub host_slots: usize,
    pub weight_bytes: u64,
    /// pinned rings for the passes' small tables up and the routers' logits down, copied by kernels on the compute
    /// queue (the runtime's memcpy goes to a copy engine and waits there behind the expert store's copies - the
    /// deepseek2 engine's lesson)
    up: Mutex<(HostBuf, usize)>,
    back: Mutex<HostBuf>,
    /// o_a's multi-matrix tables by (layer, rows): the groups' pointers for each row, then the columns
    oa_tabs: Mutex<HashMap<(u64, usize), Arc<DevBuf>>>,
    /// the dense matrices' multi-matrix tables (mm_many), by layer, rows and matrices
    multi_tabs: Mutex<HashMap<String, Arc<DevBuf>>>,
}

/// the pinned rings' sizes
const UP_BYTES: usize = 8 << 20;
const BACK_BYTES: usize = 4 << 20;

/// A layer's attention state: the raw keys' ring (fp16 [raw_ring()][512], slot pos % ring), and at a compressed
/// layer its compressor's
struct Att {
    raw: DevBuf,
    comp: Option<Comp>,
}

/// A compressed layer's blocks (fp16 [ctx / ratio + 1][512], normed and rotated) and the ring of its last tokens'
/// projections ([comp_ring][2W] floats); at ratio 4 the indexer's own compressor too
struct Comp {
    ratio: usize,
    cache: DevBuf,
    ring: DevBuf,
    idx: Option<IdxC>,
}

/// The indexer's compressed keys (float32 [ctx / 4 + 1][128]) and its ring ([comp_ring(4)][2 x 256])
struct IdxC {
    cache: DevBuf,
    ring: DevBuf,
}

/// A conversation's state: per layer its attention state (on its layer's GPU); `pos` tokens so far.
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<Att>,
    /// (first position, rows) of the last forward pass of 2..=MAX_VERIFY rows (a rollback's bounds)
    snapped: Option<(usize, usize)>,
}

/// A conversation's state at a position, in host memory (`Ds4::save` / `Ds4::restore`).
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    /// host bytes held
    pub bytes: usize,
}

impl Checkpoint {
    /// The checkpoint as a file: "NSD41", the position, the buffers
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let u = |w: &mut dyn std::io::Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
        w.write_all(b"NSD41")?;
        u(w, self.pos)?;
        u(w, self.bufs.len())?;
        for b in &self.bufs {
            u(w, b.len())?;
            w.write_all(b)?;
        }
        Ok(())
    }

    /// `write_to`'s file back
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Checkpoint> {
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "not a checkpoint file");
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)?;
        if &magic != b"NSD41" {
            return Err(bad());
        }
        let pos = u(r)?;
        let n = u(r)?;
        let mut bufs = Vec::with_capacity(n);
        let mut bytes = 0;
        for _ in 0..n {
            let len = u(r)?;
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// Generation from a fed prompt: `Ds4::step` commits the next token(s) each call. The last committed token is fed
/// at the start of the following call (a stop token is never fed).
pub struct Decoder {
    logits: Vec<f32>,
    next: Option<u32>,
    look: Option<Lookup>,
    pub ng_drafted: u64,
    pub ng_accepted: u64,
}

/// A conversation's tokens with a 3-gram index: `after[(a, b, c)]` is the position following that 3-gram's most recent
/// earlier occurrence
#[derive(Default)]
struct Lookup {
    hist: Vec<u32>,
    after: HashMap<(u32, u32, u32), usize>,
}

impl Lookup {
    fn push(&mut self, t: u32) {
        self.hist.push(t);
        let n = self.hist.len();
        if n >= 4 {
            self.after.insert((self.hist[n - 4], self.hist[n - 3], self.hist[n - 2]), n - 1);
        }
    }
    fn propose(&self, k: usize) -> Vec<u32> {
        let n = self.hist.len();
        if n < 3 || k == 0 {
            return Vec::new();
        }
        match self.after.get(&(self.hist[n - 3], self.hist[n - 2], self.hist[n - 1])) {
            Some(&p) if p < n => self.hist[p..(p + k).min(n)].to_vec(),
            _ => Vec::new(),
        }
    }
}

impl Decoder {
    pub fn set_context(&mut self, tokens: &[u32]) {
        if ngram_k() == 0 {
            return;
        }
        let mut l = Lookup { hist: Vec::with_capacity(tokens.len() + 4096), after: HashMap::with_capacity(tokens.len()) };
        for &t in tokens {
            l.push(t);
        }
        self.look = Some(l);
    }

    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
}

pub struct Ds4<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// the hash layers' tables: token -> its experts (n_expert_used each)
    hash: HashMap<u64, Vec<i32>>,
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
    gpu_prof: bool,
    stamps: Mutex<Vec<Stamp>>,
    /// the pass running is a decode pass (NS_PROFILE_SPLIT=1: its sections counted apart, as "decode: ...")
    decoding: std::sync::atomic::AtomicBool,
}

const VECTORS: [Role; 15] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale,
                             Role::HeadHcBase, Role::HeadHcScale, Role::QANorm, Role::KvNorm, Role::Sinks, Role::CompNorm, Role::IdxCompNorm,
                             Role::RouterBias];
/// the compressors' position terms: float32 [ratio][W], read as they are
const APES: [Role; 2] = [Role::CompApe, Role::IdxCompApe];

fn e(x: impl std::fmt::Display) -> Error {
    Error(x.to_string())
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn mem_available() -> usize {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// sqrt(softplus(z)): the router's score (ggml's softplus: z itself past 20)
fn sqrt_softplus(z: f32) -> f32 {
    (if z > 20.0 { z } else { z.exp().ln_1p() }).sqrt()
}

/// The layout of layer `l`'s expert slots.
fn expert_parts(m: &Model, l: u64) -> Result<ExpertParts> {
    let mut off = 0;
    let mut part = |r: Role| -> Result<(usize, usize, GType)> {
        let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
        let b = (t.bytes / m.g.n_expert) as usize;
        let p = (off, b, t.ty);
        off += b;
        Ok(p)
    };
    Ok(ExpertParts { gate: part(Role::ExpGate)?, up: part(Role::ExpUp)?, down: part(Role::ExpDown)? })
}

/// Expert `ex` of layer `l` from the file into `host` (a slot's layout).
fn read_expert(m: &Model, l: u64, ex: u64, p: &ExpertParts, host: &mut [u8]) -> Result<()> {
    for (r, (o, n, _)) in [(Role::ExpGate, p.gate), (Role::ExpUp, p.up), (Role::ExpDown, p.down)] {
        let t = m.tensor(l, r).ok_or("expert tensor missing")?;
        m.file.read_into(t, ex * n as u64, &mut host[o..o + n]).map_err(e)?;
    }
    Ok(())
}

impl Part {
    /// The part's layers onto its GPU (and the head when `last`), its expert cache and mirror.
    #[allow(clippy::too_many_arguments)]
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, last: bool, expert_bytes: Option<usize>, mirror_bytes: usize, kv_reserve: usize,
            log: &mut dyn FnMut(String)) -> Result<Part> {
        let g = &m.g;
        let ops = Ops::new(gpu.clone())?;
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let x16 = if dense_f16_min() > 0 && dense_f16() && prefill_chunk() >= dense_f16_min() {
            Some(DevBuf::new(gpu, prefill_chunk() * X16_COLS * 2)?)
        } else {
            None
        };
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            todo.push((0, vec![Role::OutputNorm, Role::Output, Role::HeadHcFn, Role::HeadHcBase, Role::HeadHcScale]));
        }
        todo.extend(layers.clone().map(|l| (l, m.roles(l))));
        for (l, roles) in todo {
            for r in roles {
                if matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown | Role::TokenEmbd | Role::HashTable) {
                    continue; // experts through the cache; the embedding's rows are read per token; the tables on the host
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let raw = DevBuf::new(gpu, t.bytes as usize)?;
                raw.write(0, &m.file.read(t).map_err(e)?)?;
                bytes += t.bytes;
                let n = t.elements() as usize;
                if VECTORS.contains(&r) || APES.contains(&r) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    vecs.insert((l, r), f);
                } else {
                    let cols = *t.shape.last().unwrap_or(&1) as usize;
                    if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
                        let f = DevBuf::f32(gpu, n)?;
                        ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                        mats.insert((l, r), Mat { buf: f, ty: GType::F32, rows: n / cols, cols, f32: true });
                    } else {
                        mats.insert((l, r), Mat { buf: raw, ty: t.ty, rows: n / cols, cols, f32: false });
                    }
                }
            }
        }
        gpu.sync()?;
        // o_a by group: its rows [g * o_lora, ..) read the group's 8 heads
        let mut groups = BTreeMap::new();
        for l in layers.clone() {
            let w = mats.get(&(l, Role::OutA)).ok_or_else(|| Error(format!("block {l}: o_a missing")))?;
            let gs = (0..g.o_groups as usize).map(|gi| w.rows_view(gi * g.o_lora as usize, g.o_lora as usize)).collect::<Result<Vec<_>>>()?;
            groups.insert(l, gs);
        }
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(MAX_COLS, MMVQ_COLS))?;
        // the store: slots of the largest layer's [gate | up | down], in chunks; VRAM, then pinned host memory
        let moe: Vec<u64> = layers.clone().collect();
        let n_exp = moe.len() * g.n_expert as usize;
        let slot_bytes = moe.iter().map(|l| m.expert_bytes(*l) as usize).max().unwrap_or(256).next_multiple_of(256);
        let budget = match expert_bytes {
            Some(b) => b,
            // what is free less the arena, the forward pass's own buffers, and the sessions' attention caches - sized
            // now so a long context never pushes VRAM into the driver's spill path
            None => gpu.memory()?.1.map_or(8usize << 30, |f| {
                let pass = (600 << 20) + prefill_chunk() * (352 << 10);
                (f as usize).saturating_sub(pass + arena_bytes().max(1 << 30) + SMALL_ARENA + kv_reserve)
            }),
        };
        let per_chunk = (CHUNK / slot_bytes).max(1);
        let nv = (budget / slot_bytes).min(n_exp);
        let nr = if n_exp > nv { (mirror_bytes / slot_bytes).min(n_exp - nv + 1) } else { 0 };
        let alloc_v = |n: usize| -> Result<Vec<DevBuf>> {
            let mut v = Vec::new();
            let mut have = 0;
            while have < n {
                let k = per_chunk.min(n - have);
                v.push(DevBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
            Ok(v)
        };
        let vram = alloc_v(nv)?;
        let mut host = Vec::new();
        {
            let mut have = 0;
            while have < nr {
                let k = per_chunk.min(nr - have);
                host.push(HostBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
        }
        // what goes where at load: VRAM first, then host (all but the spare) - the most used in the expert profile
        // first, else in layer order
        let mut all: Vec<(u64, u64)> = moe.iter().flat_map(|l| (0..g.n_expert).map(move |x| (*l, x))).collect();
        if let Some(prof) = expert_profile() {
            all.sort_by_key(|k| std::cmp::Reverse(prof.get(k).copied().unwrap_or(0.0) as u64));
        }
        let in_v: Vec<(u64, u64)> = all.iter().take(nv).copied().collect();
        let in_r: Vec<(u64, u64)> = all.iter().skip(nv).take(nr.saturating_sub(1)).copied().collect();
        let t0 = Instant::now();
        let lay: HashMap<u64, ExpertParts> = moe.iter().map(|l| Ok((*l, expert_parts(m, *l)?))).collect::<Result<_>>()?;
        let threads = 8;
        let failed: Mutex<Option<String>> = Mutex::new(None);
        {
            let views: Vec<Mutex<&mut [u8]>> = host.iter_mut().map(|c| Mutex::new(c.as_mut_slice())).collect();
            std::thread::scope(|sc| {
                for th in 0..threads {
                    let (in_v, in_r, views, lay, failed, vram) = (&in_v, &in_r, &views, &lay, &failed, &vram);
                    sc.spawn(move || {
                        let mut buf = vec![0u8; slot_bytes];
                        let mut job = |i: usize, key: (u64, u64), to_v: bool| -> Result<()> {
                            read_expert(m, key.0, key.1, &lay[&key.0], &mut buf)?;
                            let (ch, o) = (i / per_chunk, (i % per_chunk) * slot_bytes);
                            if to_v {
                                vram[ch].write(o, &buf)
                            } else {
                                views[ch].lock().unwrap()[o..o + slot_bytes].copy_from_slice(&buf);
                                Ok(())
                            }
                        };
                        for (i, key) in in_v.iter().enumerate().skip(th).step_by(threads) {
                            if let Err(x) = job(i, *key, true) {
                                *failed.lock().unwrap() = Some(x.0);
                                return;
                            }
                        }
                        for (i, key) in in_r.iter().enumerate().skip(th).step_by(threads) {
                            if let Err(x) = job(i, *key, false) {
                                *failed.lock().unwrap() = Some(x.0);
                                return;
                            }
                        }
                    });
                }
            });
        }
        if let Some(x) = failed.into_inner().unwrap() {
            return Err(Error(format!("loading the experts: {x}")));
        }
        let mut loc = HashMap::new();
        for (i, k) in in_v.iter().enumerate() {
            loc.insert(*k, Loc::V(i));
        }
        for (i, k) in in_r.iter().enumerate() {
            loc.insert(*k, Loc::R(i));
        }
        let rfree: Vec<usize> = (in_r.len()..nr).collect();
        let on_file = n_exp - in_v.len() - in_r.len();
        log(format!("{} (layers {}-{}): {:.2} GiB of weights; experts {} in VRAM ({:.1} GiB), {} in pinned host memory ({:.1} GiB), {} on the file only ({:.1} s)",
                    gpu.name, layers.start, layers.end - 1, gib(bytes as usize), in_v.len(), gib(nv * slot_bytes), in_r.len(), gib(nr * slot_bytes), on_file,
                    t0.elapsed().as_secs_f64()));
        let mut vowner = vec![None; nv];
        for (i, k) in in_v.iter().enumerate() {
            vowner[i] = Some(*k);
        }
        let mut store = Store { vram, host, slot_bytes, per_chunk, loc, vowner, vused: vec![0; nv], rfree, tick: 0, hits: 0, misses: 0, from_host: 0, direct: 0,
                                rwrite: HashMap::new(), down_any: None, vfill: HashMap::new(), last_up: None, prefetched: HashMap::new(), usage: HashMap::new(),
                                pf_issued: 0, pf_used: 0, base: nv, lend: 0, lent: false };
        let big = DevBuf::new(gpu, arena_bytes())?;
        let small = DevBuf::new(gpu, SMALL_ARENA)?;
        let lend = if lend_arena() && !moe.is_empty() { big.len / slot_bytes } else { 0 };
        if lend > 0 {
            store.vram.push(big.view(0, lend * slot_bytes)?);
            store.vowner.resize(nv + lend, None);
            store.vused.resize(nv + lend, 0);
        }
        store.base = nv;
        store.lend = lend;
        store.lent = lend > 0;
        let arena = Arena::on(if lend > 0 { small.view(0, small.len)? } else { big.view(0, big.len)? });
        let esimd_gu = ops.moe_esimd_gu_max();
        let up = Mutex::new((HostBuf::new(gpu, UP_BYTES)?, 0));
        let back = Mutex::new(HostBuf::new(gpu, BACK_BYTES)?);
        Ok(Part { ops, arena, big, small, biases: Mutex::new(HashMap::new()), layers, mats, vecs, groups, scratch, x16, q8, esimd_gu, experts: Mutex::new(store),
                  hold: std::sync::atomic::AtomicBool::new(false), expert_slots: nv + lend, host_slots: nr, weight_bytes: bytes, up, back, oa_tabs: Mutex::new(HashMap::new()), multi_tabs: Mutex::new(HashMap::new()) })
    }

    /// The multi-matrix table of layer l's matrices `ks` (of `roles`) for t rows (made once): entry (k, r) = matrix
    /// k against column r, its rows after the previous entry's
    fn multi_table(&self, l: u64, roles: &[Role], ks: &[usize], ws: &[&Mat], t: usize) -> Result<Arc<DevBuf>> {
        let key = format!("{l}:{t}:{:?}", ks.iter().map(|&k| roles[k]).collect::<Vec<_>>());
        let mut m = self.multi_tabs.lock().unwrap();
        if let Some(b) = m.get(&key) {
            return Ok(b.clone());
        }
        let (mut ptrs, mut cols, mut starts) = (Vec::new(), Vec::new(), vec![0i32]);
        for &k in ks {
            for r in 0..t {
                ptrs.push(ws[k].buf.ptr() as u64);
                cols.push(r as i32);
                starts.push(starts.last().unwrap() + ws[k].rows as i32);
            }
        }
        let mut tb: Vec<u8> = Vec::new();
        for v in &ptrs {
            tb.extend(v.to_le_bytes());
        }
        for v in cols.iter().chain(&starts) {
            tb.extend(v.to_le_bytes());
        }
        let b = Arc::new(DevBuf::new(&self.ops.gpu, tb.len())?);
        b.write(0, &tb)?;
        m.insert(key, b.clone());
        Ok(b)
    }

    /// Layer l's o_a table for `tr` rows (made once): entry r * groups + g = group g's rows against column of the same
    fn oa_table(&self, l: u64, tr: usize, gm: &[Mat]) -> Result<Arc<DevBuf>> {
        let mut m = self.oa_tabs.lock().unwrap();
        if let Some(b) = m.get(&(l, tr)) {
            return Ok(b.clone());
        }
        let n = tr * gm.len();
        let mut tb: Vec<u8> = Vec::with_capacity(n * 12);
        for _ in 0..tr {
            for w in gm {
                tb.extend((w.buf.ptr() as u64).to_le_bytes());
            }
        }
        for e in 0..n as i32 {
            tb.extend(e.to_le_bytes());
        }
        let b = Arc::new(DevBuf::new(&self.ops.gpu, tb.len())?);
        b.write(0, &tb)?;
        m.insert((l, tr), b.clone());
        Ok(b)
    }

    /// `data` into dst's front in queue order: through the pinned ring, copied by a kernel (big ones, or a length the
    /// kernel does not take, by write_async)
    fn upload(&self, dst: &DevBuf, data: &[u8]) -> Result<()> {
        let n = data.len();
        let mut r = self.up.lock().unwrap();
        if n == 0 || n % 4 != 0 || n > r.0.len / 4 {
            drop(r);
            return dst.write_async(0, data);
        }
        if r.1 + n > r.0.len {
            // what the kernels read from the ring so far is done once the queue is
            self.ops.gpu.sync()?;
            r.1 = 0;
        }
        let off = r.1;
        r.0.as_mut_slice()[off..off + n].copy_from_slice(data);
        r.1 = (off + n).next_multiple_of(64);
        let src = r.0.device_view(off, n)?;
        self.ops.kcopy(dst, 0, &src, 0, n)?;
        Ok(())
    }

    /// Several buffers' first floats on their way down (kernels copy them into pinned memory): the last copy's ticket
    /// and where each lands, for `read_finish` - the queue goes on meanwhile. None when they do not fit the buffer.
    fn read_start(&self, xs: &[(&DevBuf, usize)]) -> Result<Option<(i64, Vec<(usize, usize)>)>> {
        let back = self.back.lock().unwrap();
        let total: usize = xs.iter().map(|(_, n)| (n * 4).next_multiple_of(64)).sum();
        if total > back.len {
            return Ok(None);
        }
        let (mut at, mut spans, mut tk) = (0, Vec::new(), 0);
        for (x, n) in xs {
            tk = self.ops.kcopy(&back.device_view(at, n * 4)?, 0, x, 0, n * 4)?;
            spans.push((at, n * 4));
            at += (n * 4).next_multiple_of(64);
        }
        Ok(Some((tk, spans)))
    }
    fn read_finish(&self, tk: i64, spans: &[(usize, usize)]) -> Result<Vec<Vec<f32>>> {
        self.ops.ticket_wait(tk)?;
        let back = self.back.lock().unwrap();
        Ok(spans.iter().map(|(a, b)| back.as_slice()[*a..a + b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).collect())
    }

    /// The arena a pass of `t` tokens needs: the small one, or the big one - its memory then out of the expert store.
    fn arena_for(&self, t: usize) -> Result<()> {
        let mut c = self.experts.lock().unwrap();
        if c.lend == 0 {
            return Ok(());
        }
        let need = t * (arena_bytes() / prefill_chunk()).max(1) + (64 << 20);
        let c = &mut *c;
        if need > SMALL_ARENA && c.lent {
            let after = Some(self.ops.mark()?);
            let sb = c.slot_bytes;
            let (base, lend) = (c.base, c.lend);
            for s in base..base + lend {
                let Some(v) = c.vowner[s].take() else { continue };
                c.loc.remove(&v);
                c.prefetched.remove(&v);
                let (vch, vo) = c.vat(s);
                if let Some(f) = c.rfree.pop() {
                    let (rch, ro) = c.rat(f);
                    let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb, &[after, c.vfill.get(&s).copied(), c.rwrite.get(&f).copied()])?;
                    c.rwrite.insert(f, t);
                    c.down_any = Some(t);
                    c.loc.insert(v, Loc::R(f));
                }
                c.vfill.remove(&s);
            }
            for t in [c.down_any, c.last_up].into_iter().flatten() {
                self.ops.await_ticket(t)?;
            }
            self.ops.gpu.sync()?;
            c.lent = false;
            self.arena.set(self.big.view(0, self.big.len)?);
        } else if need <= SMALL_ARENA && !c.lent && !self.hold.load(std::sync::atomic::Ordering::Relaxed) {
            c.lent = true;
            self.arena.set(self.small.view(0, self.small.len)?);
        }
        Ok(())
    }

    /// Layer `l`'s router bias in host memory (read from the GPU once)
    fn router_bias(&self, l: u64) -> Result<Arc<Vec<f32>>> {
        if let Some(b) = self.biases.lock().unwrap().get(&l) {
            return Ok(b.clone());
        }
        let b = Arc::new(self.vec(l, Role::RouterBias)?.to_f32()?);
        self.biases.lock().unwrap().insert(l, b.clone());
        Ok(b)
    }

    fn mat(&self, l: u64, r: Role) -> Result<&Mat> {
        self.mats.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: matrix {r:?} not on this GPU")))
    }
    fn vec(&self, l: u64, r: Role) -> Result<&DevBuf> {
        self.vecs.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: vector {r:?} not on this GPU")))
    }

    /// y[t rows from y.1, y.2 apart] (+)= x . W^T. Float matrices multiply as they are; quantized ones from their
    /// blocks for up to MMVQ_COLS contiguous rows (decode), else expanded in row chunks.
    fn matmul(&self, w: &Mat, t: usize, x: (&DevBuf, usize, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        if w.f32 {
            // decode widths a row at a time (oneMKL picks its kernel by the row count: a verify pass's rows equal
            // one-token passes)
            if t <= MMVQ_COLS {
                for r in 0..t {
                    self.ops.gemm_at(1, w.rows, w.cols, (x.0, x.1 + r * x.2, x.2), (&w.buf, 0), (y.0, y.1 + r * y.2, y.2), acc)?;
                }
                return Ok(());
            }
            return self.ops.gemm_at(t, w.rows, w.cols, x, (&w.buf, 0), y, acc);
        }
        if !acc && t <= MMVQ_COLS && x.2 == w.cols && y.2 == w.rows && w.cols <= MAX_COLS && self.ops.mmvq_supported(w.ty.code()) {
            self.ops.quantize_q8_1((x.0, x.1), &self.q8, w.cols, t)?;
            return self.ops.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (y.0, y.1), w.cols, w.rows, t);
        }
        let rb = w.row_bytes();
        let f16 = dense_f16_min();
        let fits = |b: &DevBuf| t * w.cols * 2 <= b.len;
        if let Some(x16) = self.x16.as_ref().filter(|b| f16 > 0 && t >= f16 && t > MMVQ_COLS && fits(b))
            .filter(|_| x.2 == w.cols && w.cols.is_multiple_of(256) && F16_TYPES.contains(&w.ty.code())) {
            self.ops.to_f16(&x.0.view(x.1 * 4, t * w.cols * 4)?, x16, t * w.cols)?;
            let chunk = (SCRATCH * 2 / w.cols).max(1).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                self.ops.dequant_f16(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
                self.ops.gemm_f16(t, r, w.cols, (x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
                r0 += r;
            }
            return Ok(());
        }
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let mut r0 = 0;
        while r0 < w.rows {
            let r = chunk.min(w.rows - r0);
            self.ops.dequant(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
            self.ops.gemm_at(t, r, w.cols, x, (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
            r0 += r;
        }
        Ok(())
    }

    /// y [t, rows] = x [t, cols] . W^T (contiguous)
    fn mm(&self, l: u64, r: Role, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let w = self.mat(l, r)?;
        let y = self.arena.f32(t * w.rows)?;
        self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
        Ok(y)
    }

    /// Where every expert of `need` (layer `l`) is read from, in `need`'s order, as a view of its slot (glm5next's
    /// store: `promote` makes each VRAM-resident, else host-slot experts are read in place). The second value marks
    /// the experts whose copy is still on its way, the third the tickets of the last copy.
    #[allow(clippy::type_complexity)]
    fn ensure(&self, m: &Model, l: u64, need: &[u64], promote: bool) -> Result<(Vec<(DevBuf, bool)>, Vec<bool>, Option<(i64, i64)>)> {
        let mut c = self.experts.lock().unwrap();
        let c = &mut *c;
        if c.vowner.len() < need.len() {
            return Err(Error(format!("the expert store has {} VRAM slots; one layer needs {}", c.vowner.len(), need.len())));
        }
        c.tick += 1;
        let tick = c.tick;
        if promote {
            for &x in need {
                *c.usage.entry((l, x)).or_insert(0) += 1;
            }
        }
        let mut missing = Vec::new();
        let mut pending: Vec<(u64, u64)> = Vec::new();
        let mut last: Option<(i64, i64)> = None;
        let mut reading: HashMap<usize, i64> = HashMap::new();
        for &ex in need {
            match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                    if let Some(tk) = c.prefetched.remove(&(l, ex)) {
                        c.pf_used += 1;
                        pending.push((l, ex));
                        last = Some(last.map_or(tk, |(a, b)| (a.max(tk.0), b.max(tk.1))));
                    }
                }
                Some(Loc::R(_)) if !promote => {}
                _ => missing.push(ex),
            }
        }
        c.prefetched.retain(|k, _| k.0 != l);
        if !missing.is_empty() {
            let mut order: Vec<usize> = (0..c.vowner.len()).filter(|&i| c.vused[i] != tick && c.usable(i)).collect();
            order.sort_by_key(|&i| if c.vowner[i].is_none() { 0 } else { c.vused[i] + 1 });
            let mut buf: Option<Vec<u8>> = None;
            let after = if promote && !swap_wait() { None } else { Some(self.ops.mark()?) };
            for (ex, &s) in missing.iter().zip(&order) {
                c.misses += 1;
                if let Some(tk) = self.swap_in(m, c, l, *ex, s, after, &mut reading, &mut buf)? {
                    last = Some(last.map_or(tk, |(a, b)| (a.max(tk.0), b.max(tk.1))));
                    pending.push((l, *ex));
                }
                c.vused[s] = tick;
            }
        }
        let sb = c.slot_bytes;
        let slots = need.iter().map(|ex| match c.loc[&(l, *ex)] {
            Loc::V(s) => {
                let (ch, o) = c.vat(s);
                Ok((c.vram[ch].view(o, sb)?, false))
            }
            Loc::R(r) => {
                c.direct += 1;
                let (rch, ro) = c.rat(r);
                Ok((c.host[rch].device_view(ro, sb)?, true))
            }
        }).collect::<Result<Vec<_>>>()?;
        let arriving = need.iter().map(|ex| pending.contains(&(l, *ex))).collect();
        if !promote {
            if let Some(t) = c.down_any.take() {
                self.ops.await_ticket(t)?;
            }
        }
        Ok((slots, arriving, last))
    }

    /// Expert `(l, ex)` into VRAM slot `s` (glm5next's swap: the slot's expert down to a free host slot, the wanted
    /// one up from its host slot or the file)
    #[allow(clippy::too_many_arguments)]
    fn swap_in(&self, m: &Model, c: &mut Store, l: u64, ex: u64, s: usize, after: Option<i64>, reading: &mut HashMap<usize, i64>,
               buf: &mut Option<Vec<u8>>) -> Result<Option<(i64, i64)>> {
        let key = (l, ex);
        let (vch, vo) = c.vat(s);
        let sb = c.slot_bytes;
        let from = c.loc.get(&key).copied();
        let fill = c.vfill.get(&s).copied();
        let mut down: Option<i64> = None;
        if let Some(v) = c.vowner[s].take() {
            c.loc.remove(&v);
            c.prefetched.remove(&v);
            if let Some(f) = c.rfree.pop() {
                let (rch, ro) = c.rat(f);
                let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb, &[after, reading.get(&f).copied(), fill])?;
                down = Some(t);
                c.rwrite.insert(f, t);
                c.down_any = Some(t);
                c.loc.insert(v, Loc::R(f));
            }
        }
        let up = match from {
            Some(Loc::R(r)) => {
                c.from_host += 1;
                let (rch, ro) = c.rat(r);
                let split = expert_parts(m, l)?.down.0.min(sb);
                let deps = [down.or(after), c.rwrite.get(&r).copied()];
                let host = c.host[rch].device_view(ro, sb)?;
                let t1 = self.ops.stream_copy_on(1, &c.vram[vch], vo, &host, 0, split, &deps)?;
                let t2 = self.ops.stream_copy_on(1, &c.vram[vch], vo + split, &host, split, sb - split, &deps)?;
                reading.insert(r, t2);
                c.rfree.push(r);
                c.vfill.insert(s, t2);
                c.last_up = Some(t2);
                Some((t1, t2))
            }
            _ => {
                for t in [c.last_up, c.down_any].into_iter().flatten() {
                    self.ops.await_ticket(t)?;
                }
                self.ops.gpu.sync()?;
                let parts = expert_parts(m, l)?;
                let b = buf.get_or_insert_with(|| vec![0u8; sb]);
                read_expert(m, l, ex, &parts, b)?;
                c.vram[vch].write(vo, b)?;
                c.vfill.remove(&s);
                None
            }
        };
        c.vowner[s] = Some(key);
        c.loc.insert(key, Loc::V(s));
        Ok(up)
    }

    /// Decode: the experts layer `l` is expected to want, swapped in now so they arrive while the layer's attention
    /// runs. Only ones in host memory; at most `limit`.
    fn prefetch(&self, m: &Model, l: u64, want: &[u64], limit: usize) -> Result<()> {
        let mut c = self.experts.lock().unwrap();
        let c = &mut *c;
        let tick = c.tick;
        let cand: Vec<u64> = want.iter().copied().filter(|ex| matches!(c.loc.get(&(l, *ex)), Some(Loc::R(_)))).take(limit).collect();
        if cand.is_empty() {
            return Ok(());
        }
        let mut order: Vec<usize> = (0..c.vowner.len()).filter(|&i| c.vused[i] != tick && c.usable(i)).collect();
        order.sort_by_key(|&i| if c.vowner[i].is_none() { 0 } else { c.vused[i] + 1 });
        let after = if swap_wait() { Some(self.ops.mark()?) } else { None };
        let mut reading: HashMap<usize, i64> = HashMap::new();
        let mut buf = None;
        for (ex, &s) in cand.iter().zip(&order) {
            if let Some(tk) = self.swap_in(m, c, l, *ex, s, after, &mut reading, &mut buf)? {
                c.prefetched.insert((l, *ex), tk);
                c.pf_issued += 1;
            }
            c.vused[s] = tick;
        }
        Ok(())
    }

    /// Several matrices on the same input x [t, cols]: one Q8_1 quantization of x serves every decode product.
    fn mm_many(&self, l: u64, roles: &[Role], x: &DevBuf, t: usize) -> Result<Vec<DevBuf>> {
        let ws: Vec<&Mat> = roles.iter().map(|r| self.mat(l, *r)).collect::<Result<_>>()?;
        let quantized = ws.iter().filter(|w| !w.f32 && self.ops.mmvq_supported(w.ty.code())).count();
        let cols = ws[0].cols;
        // decode widths: the matrices of each type this engine's multi-matrix kernel takes in one launch (each
        // (matrix, row) an entry, the tables kept a layer) - the projections in 2 launches, the shared gate | up in 1
        let own = |w: &Mat| !w.f32 && w.cols == cols && self.ops.multi_supported(w.ty.code(), cols);
        if t <= MMVQ_COLS && multi_dense() && ws.iter().any(|w| own(w)) {
            let xs = self.arena.bytes(self.ops.q8_bytes(cols, t))?;
            self.ops.quantize_q8((x, 0), &xs, cols, t)?;
            let mut out: Vec<Option<DevBuf>> = (0..ws.len()).map(|_| None).collect();
            let mut types: Vec<GType> = ws.iter().filter(|w| own(w)).map(|w| w.ty).collect();
            types.dedup();
            types.sort_by_key(|ty| ty.code());
            types.dedup();
            for ty in types {
                let ks: Vec<usize> = (0..ws.len()).filter(|&k| own(ws[k]) && ws[k].ty == ty).collect();
                let total: usize = ks.iter().map(|&k| t * ws[k].rows).sum();
                let tab = self.multi_table(l, roles, &ks, &ws, t)?;
                let y = self.arena.f32(total)?;
                self.ops.mmvq_multi_var(ty.code(), &tab, ks.len() * t, &xs, t, (&y, 0), cols, total)?;
                let mut at = 0;
                for &k in &ks {
                    out[k] = Some(y.view(at * 4, t * ws[k].rows * 4)?);
                    at += t * ws[k].rows;
                }
            }
            return ws.iter().zip(out).map(|(w, o)| match o {
                Some(y) => Ok(y),
                None => {
                    let y = self.arena.f32(t * w.rows)?;
                    self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
                    Ok(y)
                }
            }).collect();
        }
        if t <= MMVQ_COLS && quantized > 1 && ws.iter().all(|w| w.cols == cols) && cols <= MAX_COLS {
            self.ops.quantize_q8_1((x, 0), &self.q8, cols, t)?;
            return ws.iter().map(|w| {
                let y = self.arena.f32(t * w.rows)?;
                if !w.f32 && self.ops.mmvq_supported(w.ty.code()) {
                    self.ops.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (&y, 0), w.cols, w.rows, t)?;
                } else {
                    self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
                }
                Ok(y)
            }).collect();
        }
        roles.iter().map(|r| self.mm(l, *r, x, t)).collect()
    }

    /// One part of a resident expert on x rows [n, cols] from float `x.1` into y rows [n, rows] from float `y.1`.
    /// `quantize`: x into Q8_1 first (false: the previous call's quantization of the same rows serves).
    #[allow(clippy::too_many_arguments)]
    fn expert_into(&self, buf: &DevBuf, at: usize, part: (usize, usize, GType), rows: usize, cols: usize, x: (&DevBuf, usize), y: (&DevBuf, usize),
                   n: usize, quantize: bool) -> Result<()> {
        let (off, bytes, ty) = part;
        if n <= MMVQ_COLS {
            if quantize {
                self.ops.quantize_q8_1(x, &self.q8, cols, n)?;
            }
            if ty.code() == MXFP4 {
                return self.ops.mxfp4_mmvq((buf, at + off), &self.q8, y, cols, rows, n);
            }
            return self.ops.mmvq(ty.code(), (buf, at + off), bytes, &self.q8, y, cols, rows, n);
        }
        // more rows: the expert expanded to fp16 (the float32 expander lacks IQ3_XXS), its rows through the half GEMM
        let xh = self.arena.bytes(n * cols * 2)?;
        self.ops.to_f16(&x.0.view(x.1 * 4, n * cols * 4)?, &xh, n * cols)?;
        self.expand_f16(ty, buf, at + off, bytes, rows * cols, &self.scratch)?;
        self.ops.gemm_f16(n, rows, cols, (&xh, 0, cols), (&self.scratch, 0), (y.0, y.1, rows), false)
    }

    /// n values of an expert part expanded to fp16 (MXFP4 by this engine's kernel)
    fn expand_f16(&self, ty: GType, buf: &DevBuf, at: usize, bytes: usize, n: usize, dst: &DevBuf) -> Result<()> {
        if ty.code() == MXFP4 {
            self.ops.mxfp4_dequant(buf, at, n, dst, true)
        } else {
            self.ops.dequant_f16(ty.code(), buf, at, bytes, n, dst)
        }
    }
}

impl<'g> Ds4<'g> {
    /// Loads the model over `gpus` (layers split by the cards' speed, or as NS_SPLIT says; the head on the last). Per GPU,
    /// `expert_bytes` of VRAM for routed experts (None: what is free less the passes' needs); `mirror_bytes` of pinned
    /// host memory in all for the experts that do not fit (None: what the host has available less 10 GiB), shared by
    /// the parts in proportion to their layers. `kv`: (context, sessions) - their attention caches are held back from
    /// the expert store.
    pub fn load(file: &'g nextsycl_gguf::Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, kv: (usize, usize),
                log: &mut dyn FnMut(String)) -> Result<Ds4<'g>> {
        let t0 = Instant::now();
        let m = Model::open(file).map_err(e)?;
        let errs = m.check();
        if !errs.is_empty() {
            return Err(Error(format!("the file's layout: {}", errs.join("; "))));
        }
        if gpus.is_empty() {
            return Err(Error("no GPU given".into()));
        }
        let g = &m.g;
        let n = g.n_layer;
        let k = gpus.len() as u64;
        let mirror = mirror_bytes.unwrap_or(mem_available().saturating_sub(10 << 30));
        let mut parts = Vec::new();
        let mut owner = vec![0usize; n as usize];
        // layers by each card's speed: the square root of its compute (units x clock), so a faster card takes more
        // without its share of the pinned memory running short (B70 + B65: 25 / 18 layers - by count 21 / 22 decoded
        // 20.4 tok/s, 24-26 21.1-21.6; by compute alone 28, its experts spilled to the file: 14.9)
        let speed: Vec<f64> = gpus.iter().map(|g| nextsycl_core::gpu_units(g.index).map_or(1.0, |(u, m)| ((u as f64) * (m as f64)).max(1.0).sqrt())).collect();
        let total: f64 = speed.iter().sum();
        let mut acc = 0.0;
        let mut bounds: Vec<u64> = vec![0];
        for s in &speed {
            acc += s;
            bounds.push(((acc / total) * n as f64).round() as u64);
        }
        bounds[k as usize] = n;
        if let Ok(v) = std::env::var("NS_SPLIT") {
            let counts: Vec<u64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if counts.len() + 1 == k as usize && counts.iter().sum::<u64>() < n {
                let mut at = 0;
                for (i, c) in counts.iter().enumerate() {
                    at += c;
                    bounds[i + 1] = at;
                }
            } else {
                return Err(Error(format!("NS_SPLIT={v}: {} layer counts below {n} for {k} GPUs", k - 1)));
            }
        }
        let mut spare = 0usize;
        for (i, gpu) in gpus.iter().enumerate() {
            let range = bounds[i]..bounds[i + 1];
            for l in range.clone() {
                owner[l as usize] = i;
            }
            let last = i + 1 == gpus.len();
            let share = mirror * (range.end - range.start) as usize / n as usize + spare;
            // per session and layer: the raw ring, and a compressed layer's blocks (fp16) and indexer keys (float32)
            let reserve: usize = range.clone().map(|l| {
                let r = g.ratio(l) as usize;
                let mut b = raw_ring() * 512 * 2;
                if r > 0 {
                    b += (kv.0 / r + 1) * 512 * 2;
                }
                if r == 4 {
                    b += (kv.0 / 4 + 1) * g.idx_dim as usize * 4;
                }
                b
            }).sum::<usize>() * kv.1;
            if reserve > 0 {
                log(format!("{}: {:.2} GiB held for the sessions' {} tokens", gpu.name, gib(reserve), kv.0 * kv.1));
            }
            let part = Part::load(&m, gpu, range, last, expert_bytes, share, reserve, log)?;
            spare = share.saturating_sub(part.host_slots * part.experts.lock().unwrap().slot_bytes);
            parts.push(part);
        }
        // the hash layers' tables, on the host
        let mut hash = HashMap::new();
        for l in 0..g.hash_layers {
            let t = m.tensor(l, Role::HashTable).ok_or_else(|| Error(format!("block {l}: the hash table is missing")))?;
            let raw = m.file.read(t).map_err(e)?;
            let v: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            if let Some(x) = v.iter().find(|x| **x < 0 || **x as u64 >= g.n_expert) {
                return Err(Error(format!("block {l}: the hash table names expert {x}")));
            }
            hash.insert(l, v);
        }
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let gpu_prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "gpu");
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1" || v == "gpu").then(|| Mutex::new(BTreeMap::new()));
        Ok(Ds4 { m, parts, owner, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, hash, prof, gpu_prof, stamps: Mutex::new(Vec::new()),
                 decoding: false.into() })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.parts.iter().map(|p| {
            let (total, free) = p.ops.gpu.memory().unwrap_or((0, None));
            GpuInfo { index: p.ops.gpu.index, name: p.ops.gpu.name.clone(), pci: p.ops.gpu.pci.clone(), total, free, layers: (p.layers.start, p.layers.end),
                      expert_slots: p.expert_slots, host_slots: p.host_slots }
        }).collect()
    }

    pub fn arena_peaks(&self) -> Vec<(usize, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        self.parts.iter().map(|p| (p.arena.peak.load(Relaxed), p.arena.spills.load(Relaxed))).collect()
    }

    /// (VRAM hits, swaps, of them from host memory, host-memory reads by prompt passes, prefetched, prefetches used)
    pub fn expert_stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        self.parts.iter().fold((0, 0, 0, 0, 0, 0), |a, p| {
            let c = p.experts.lock().unwrap();
            (a.0 + c.hits, a.1 + c.misses, a.2 + c.from_host, a.3 + c.direct, a.4 + c.pf_issued, a.5 + c.pf_used)
        })
    }

    fn timed<T>(&self, p: &Part, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.prof.is_none() {
            return f();
        }
        let m = self.mark(p);
        let r = f()?;
        self.lap(p, name, m);
        Ok(r)
    }

    fn mark(&self, p: &Part) -> Option<Mk> {
        self.prof.as_ref()?;
        if self.gpu_prof {
            return p.ops.stamp().ok().map(Mk::Gpu);
        }
        p.ops.gpu.sync().ok()?;
        Some(Mk::Host(Instant::now()))
    }

    fn lap(&self, p: &Part, name: &'static str, from: Option<Mk>) -> Option<Mk> {
        let (pr, t0) = (self.prof.as_ref()?, from?);
        let name = if self.decoding.load(std::sync::atomic::Ordering::Relaxed) { decode_name(name) } else { name };
        match t0 {
            Mk::Gpu(a) => {
                let b = p.ops.stamp().ok()?;
                let mut st = self.stamps.lock().unwrap();
                st.push((name, p.ops.gpu.clone(), a, b));
                if st.len() > 8192 {
                    let v = std::mem::take(&mut *st);
                    drop(st);
                    self.read_stamps(v);
                }
                Some(Mk::Gpu(b))
            }
            Mk::Host(t0) => {
                p.ops.gpu.sync().ok()?;
                let mut m = pr.lock().unwrap();
                let x = m.entry(name).or_insert((0.0, 0));
                x.0 += t0.elapsed().as_secs_f64();
                x.1 += 1;
                Some(Mk::Host(Instant::now()))
            }
        }
    }

    fn read_stamps(&self, v: Vec<Stamp>) {
        let Some(pr) = &self.prof else { return };
        let mut m = pr.lock().unwrap();
        for (name, gpu, a, b) in v {
            let dt = Ops::new(gpu).and_then(|o| o.elapsed(a, b)).unwrap_or(0.0);
            let x = m.entry(name).or_insert((0.0, 0));
            x.0 += dt;
            x.1 += 1;
        }
    }

    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let v = std::mem::take(&mut *self.stamps.lock().unwrap());
        self.read_stamps(v);
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// A new conversation, up to `max_ctx` tokens.
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            let gpu = &self.parts[self.owner[l as usize]].ops.gpu;
            let r = g.ratio(l) as usize;
            let raw = DevBuf::new(gpu, raw_ring() * 512 * 2)?;
            let comp = if r > 0 {
                let w = if r == 4 { 1024 } else { 512 };
                let ring = DevBuf::f32(gpu, comp_ring(r) * 2 * w)?;
                ring.fill(0)?;
                let idx = if r == 4 {
                    let iw = 2 * g.idx_dim as usize;
                    let iring = DevBuf::f32(gpu, comp_ring(4) * 2 * iw)?;
                    iring.fill(0)?;
                    Some(IdxC { cache: DevBuf::f32(gpu, (max_ctx / 4 + 1) * g.idx_dim as usize)?, ring: iring })
                } else {
                    None
                };
                Some(Comp { ratio: r, cache: DevBuf::new(gpu, (max_ctx / r + 1) * 512 * 2)?, ring, idx })
            } else {
                None
            };
            layers.push(Att { raw, comp });
        }
        Ok(Session { pos: 0, max_ctx, layers, snapped: None })
    }

    /// The raw ring's slots of the window before `pos` (at most 136 positions: (first byte, bytes) of up to two runs)
    fn window_runs(pos: usize) -> Vec<(usize, usize)> {
        let r = raw_ring();
        let n = pos.min(128 + MAX_VERIFY);
        if n == 0 {
            return Vec::new();
        }
        let first = (pos - n) % r;
        let row = 512 * 2;
        if first + n <= r {
            vec![(first * row, n * row)]
        } else {
            vec![(first * row, (r - first) * row), (0, (first + n - r) * row)]
        }
    }

    /// The conversation's state at `sess.pos`, copied to host memory: per layer the raw keys of the window, and its
    /// compressor's blocks so far and ring. `restore` puts it back into a session - the prompt cache's checkpoints.
    pub fn save(&self, sess: &Session) -> Result<Checkpoint> {
        let n = sess.pos;
        let idx_d = self.m.g.idx_dim as usize;
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut take = |b: &DevBuf, at: usize, len: usize| -> Result<()> {
            let mut v = vec![0u8; len];
            if len > 0 {
                b.read(at, &mut v)?;
            }
            bufs.push(v);
            Ok(())
        };
        for a in &sess.layers {
            let runs = Self::window_runs(n);
            for k in 0..2 {
                let (at, len) = runs.get(k).copied().unwrap_or((0, 0));
                take(&a.raw, at, len)?;
            }
            if let Some(c) = &a.comp {
                take(&c.cache, 0, n / c.ratio * 512 * 2)?;
                take(&c.ring, 0, c.ring.len)?;
                if let Some(ix) = &c.idx {
                    take(&ix.cache, 0, n / 4 * idx_d * 4)?;
                    take(&ix.ring, 0, ix.ring.len)?;
                }
            }
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: n, bufs, bytes })
    }

    /// `sess` becomes the conversation `ck` was saved from.
    pub fn restore(&self, sess: &mut Session, ck: &Checkpoint) -> Result<()> {
        if ck.pos + 1 > sess.max_ctx {
            return Err(Error(format!("restore: a checkpoint of {} tokens into a session of {}", ck.pos, sess.max_ctx)));
        }
        let mut it = ck.bufs.iter();
        let mut put = |b: &DevBuf, at: usize| -> Result<()> {
            let v = it.next().ok_or("restore: the checkpoint is short")?;
            if !v.is_empty() {
                b.write(at, v)?;
            }
            Ok(())
        };
        for a in &sess.layers {
            let runs = Self::window_runs(ck.pos);
            for k in 0..2 {
                put(&a.raw, runs.get(k).map_or(0, |r| r.0))?;
            }
            if let Some(c) = &a.comp {
                put(&c.cache, 0)?;
                put(&c.ring, 0)?;
                if let Some(ix) = &c.idx {
                    put(&ix.cache, 0)?;
                    put(&ix.ring, 0)?;
                }
            }
        }
        sess.pos = ck.pos;
        sess.snapped = None;
        Ok(())
    }

    /// Back to an empty conversation (the caches need nothing: `pos` bounds what is read)
    pub fn reset_session(&self, sess: &mut Session) -> Result<()> {
        sess.pos = 0;
        sess.snapped = None;
        Ok(())
    }

    /// `dst` becomes a copy of `src` (same context size)
    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if dst.max_ctx != src.max_ctx || dst.layers.len() != src.layers.len() {
            return Err(Error("copy_session: the sessions differ in shape".into()));
        }
        let idx_d = self.m.g.idx_dim as usize;
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            for (at, len) in Self::window_runs(src.pos) {
                d.raw.copy_within(at, &s.raw, at, len)?;
            }
            if let (Some(dc), Some(sc)) = (&d.comp, &s.comp) {
                let nb = src.pos / sc.ratio * 512 * 2;
                if nb > 0 {
                    dc.cache.copy_within(0, &sc.cache, 0, nb)?;
                }
                dc.ring.copy_within(0, &sc.ring, 0, sc.ring.len)?;
                if let (Some(di), Some(si)) = (&dc.idx, &sc.idx) {
                    let nb = src.pos / 4 * idx_d * 4;
                    if nb > 0 {
                        di.cache.copy_within(0, &si.cache, 0, nb)?;
                    }
                    di.ring.copy_within(0, &si.ring, 0, si.ring.len)?;
                }
            }
        }
        dst.pos = src.pos;
        dst.snapped = None;
        Ok(())
    }

    /// Token embeddings [t, n_embd] on `p`'s GPU: the rows read from the file, expanded.
    fn embed(&self, p: &Part, tokens: &[u32]) -> Result<DevBuf> {
        let g = &self.m.g;
        let emb = self.m.tensor(0, Role::TokenEmbd).ok_or("token_embd missing")?;
        let rb = emb.ty.bytes(g.n_embd).unwrap_or(0) as usize;
        let mut rows = vec![0u8; tokens.len() * rb];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as u64 >= g.n_vocab {
                return Err(Error(format!("token {tok} outside the vocabulary")));
            }
            self.m.file.read_into(emb, *tok as u64 * rb as u64, &mut rows[i * rb..(i + 1) * rb]).map_err(e)?;
        }
        let raw = DevBuf::new(&p.ops.gpu, rows.len())?;
        p.upload(&raw, &rows)?;
        let x = DevBuf::f32(&p.ops.gpu, tokens.len() * g.n_embd as usize)?;
        p.ops.dequant(emb.ty.code(), &raw, 0, rows.len(), tokens.len() * g.n_embd as usize, &x)?;
        Ok(x)
    }

    /// A compressor of layer `l` over this pass's projections kv, sc [t, w]: the blocks it completes pooled, normed
    /// (`norm`), their tail of `d` rotated at each block's start, into `cache` (fp16 when `f16`, else float32 rows)
    #[allow(clippy::too_many_arguments)]
    fn compress(&self, p: &Part, l: u64, kv: &DevBuf, sc: &DevBuf, ape: Role, norm: Role, ring: &DevBuf, cache: &DevBuf, f16: bool, d: usize, ratio: usize,
                pos0: usize, t: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (j0, j1) = (pos0 / ratio, (pos0 + t) / ratio);
        let nb = j1 - j0;
        let pooled = p.arena.f32(nb.max(1) * d)?;
        o.pool(ring, kv, sc, p.vec(l, ape)?, &pooled, pos0, t, d, ratio, ratio == 4, comp_ring(ratio))?;
        if nb == 0 {
            return Ok(());
        }
        o.rms_norm(&pooled, Some(p.vec(l, norm)?), &pooled, nb, d, g.rms_eps as f32)?;
        let rd = g.rope_dim as usize;
        o.rope(&pooled, nb, 1, d, d, d - rd, rd, j0 * ratio, ratio, self.m.rope(l), false)?;
        if f16 {
            o.rows_f16(&pooled, cache, nb, j0, d)
        } else {
            cache.copy_within(j0 * d * 4, &pooled, 0, nb * d * 4)
        }
    }

    /// The lightning indexer's picks for the rows [r0, r0 + tr) of a pass at pos0 (x their attention input, qr their
    /// q LoRA): each row's best `top_k` compressed blocks, when the last row sees more than that (else None: every
    /// row reads all it sees)
    #[allow(clippy::too_many_arguments)]
    fn select(&self, p: &Part, l: u64, x: &DevBuf, qr: &DevBuf, ix: &IdxC, pos0: usize, tr: usize, tap: Tap) -> Result<Option<DevBuf>> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, hh, kp) = (g.idx_dim as usize, g.idx_heads as usize, g.idx_top_k as usize);
        let n = (pos0 + tr) / 4; // the blocks the last row sees
        if n <= kp {
            return Ok(None);
        }
        let iq = p.mm(l, Role::IdxQB, qr, tr)?; // [tr, heads * d]
        let rd = g.rope_dim as usize;
        o.rope(&iq, tr, hh, hh * d, d, d - rd, rd, pos0, 1, self.m.rope(l), false)?;
        // the head weights; their 1/sqrt(d * heads) scale is positive, so it cannot change a top-k: left out
        let w = p.mm(l, Role::IdxProj, x, tr)?;
        let rows = if tr <= MMVQ_COLS { 1 } else { (IDX_S_FLOATS / n).clamp(1, tr) };
        let score = p.arena.f32(rows * n)?;
        let nc_max = (IDX_S_FLOATS / (rows * hh)).max(1).min(n);
        let sbuf = p.arena.f32(rows * hh * nc_max)?;
        let sel = p.arena.bytes(tr * kp * 4)?;
        for q0 in (0..tr).step_by(rows) {
            let qn = rows.min(tr - q0);
            let sc = score.view(0, qn * n * 4)?;
            let mut j0 = 0;
            while j0 < n {
                let nc = nc_max.min(n - j0);
                o.gemm_at(qn * hh, nc, d, (&iq, q0 * hh * d, d), (&ix.cache, j0 * d), (&sbuf, 0, nc), false)?;
                o.idx_score(&sbuf, &w.view(q0 * hh * 4, qn * hh * 4)?, &sc, qn, hh, j0, nc, n, pos0 + q0)?;
                j0 += nc;
            }
            if qn == tr {
                tap(&format!("indexer_score-{l}"), &sc)?;
            }
            o.topk(&sc, &sel.view(q0 * kp * 4, qn * kp * 4)?, qn, n, n, kp)?;
        }
        Ok(Some(sel))
    }

    /// The attention half of layer `l` on x [t, d] (normed) at positions pos0..: the keys into the ring, the blocks the
    /// pass completes into the compressors' caches, then per block of query rows the attention and the grouped output
    /// projection. The output [t, d].
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, p: &Part, l: u64, x: &DevBuf, st: &Att, pos0: usize, t: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let eps = g.rms_eps as f32;
        let (d, nh, hd, rd, ql) = (g.n_embd as usize, g.n_head as usize, g.head_dim as usize, g.rope_dim as usize, g.q_lora as usize);
        let (ng, ol) = (g.o_groups as usize, g.o_lora as usize);
        let gw = nh * hd / ng;
        let rope = self.m.rope(l);
        let ratio = g.ratio(l) as usize;
        let t_proj = self.mark(p);
        let mut roles = vec![Role::QA, Role::Kv];
        if ratio > 0 {
            roles.extend([Role::CompKv, Role::CompGate]);
        }
        if ratio == 4 {
            roles.extend([Role::IdxCompKv, Role::IdxCompGate]);
        }
        let pj = p.mm_many(l, &roles, x, t)?;
        let qr = p.arena.f32(t * ql)?;
        o.rms_norm(&pj[0], Some(p.vec(l, Role::QANorm)?), &qr, t, ql, eps)?;
        tap(&format!("qr_norm-{l}"), &qr)?;
        let kv = p.arena.f32(t * hd)?;
        o.rms_norm(&pj[1], Some(p.vec(l, Role::KvNorm)?), &kv, t, hd, eps)?;
        o.rope(&kv, t, 1, hd, hd, hd - rd, rd, pos0, 1, rope, false)?;
        tap(&format!("kv-{l}"), &kv)?;
        let rr = raw_ring();
        o.ring_put(&kv, &st.raw, t, pos0, rr, hd)?;
        let t_comp = self.lap(p, "attn: projections", t_proj);
        if let Some(c) = &st.comp {
            self.compress(p, l, &pj[2], &pj[3], Role::CompApe, Role::CompNorm, &c.ring, &c.cache, true, hd, ratio, pos0, t)?;
            if let Some(ix) = &c.idx {
                self.compress(p, l, &pj[4], &pj[5], Role::IdxCompApe, Role::IdxCompNorm, &ix.ring, &ix.cache, false, g.idx_dim as usize, 4, pos0, t)?;
            }
        }
        let t_q = self.lap(p, "attn: compressors", t_comp);
        // the queries' side, a block of rows at a time
        let rb = if t <= MMVQ_COLS { t } else { att_rows().min(t) };
        let oa = p.arena.f32(t * ng * ol)?;
        let q = p.arena.f32(rb * nh * hd)?;
        let att = p.arena.f32(rb * nh * hd)?;
        let og = p.arena.f32(rb * ng * ol)?;
        let sinks = p.vec(l, Role::Sinks)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let qb = p.mat(l, Role::QB)?;
        let mut t_s = t_q;
        for r0 in (0..t).step_by(rb) {
            let tr = rb.min(t - r0);
            p.matmul(qb, tr, (&qr, r0 * ql, ql), (&q, 0, nh * hd), false)?;
            o.rms_norm(&q, None, &q, tr * nh, hd, eps)?;
            o.rope(&q, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, false)?;
            if r0 == 0 && tr == t {
                tap(&format!("q-{l}"), &q)?;
            }
            t_s = self.lap(p, "attn: q_b", t_s);
            let sel = match st.comp.as_ref().and_then(|c| c.idx.as_ref()) {
                Some(ix) => {
                    let xv = x.view(r0 * d * 4, tr * d * 4)?;
                    let qv = qr.view(r0 * ql * 4, tr * ql * 4)?;
                    self.select(p, l, &xv, &qv, ix, pos0 + r0, tr, &mut *tap)?
                }
                None => None,
            };
            t_s = self.lap(p, "attn: indexer", t_s);
            let comp = st.comp.as_ref().map(|c| (&c.cache, c.ratio, c.cache.len / (512 * 2)));
            o.attend(&q, &st.raw, rr, comp, sel.as_ref().map(|s| (s, g.idx_top_k as usize)), sinks, &att, tr, nh, pos0 + r0, g.window as usize, scale)?;
            t_s = self.lap(p, "attn: attend", t_s);
            o.rope(&att, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, true)?;
            if r0 == 0 && tr == t {
                tap(&format!("attn_derope-{l}"), &att)?;
            }
            let gm = p.groups.get(&l).ok_or("o_a's groups")?;
            if tr <= MMVQ_COLS && !gm[0].f32 && o.multi_supported(gm[0].ty.code(), gw) {
                // decode: every (row, group) one entry of the multi-matrix product - att is [tr][groups][gw], so entry
                // e is column e of its quantized rows and its output row e of oa ([tr][groups][o_lora]); the pointer
                // table is the layer's own, kept (B70: 89 -> 63 us a layer against 8 launches and 2 permutes)
                let xq = p.arena.bytes(o.q8_bytes(gw, tr * ng))?;
                o.quantize_q8((&att, 0), &xq, gw, tr * ng)?;
                let tab = p.oa_table(l, tr, gm)?;
                o.mmvq_multi(gm[0].ty.code(), &tab, tr * ng, &xq, tr * ng, (&oa, r0 * ng * ol), gw, ol)?;
            } else {
                // the grouped output: [tr][groups][gw] -> [groups][tr][gw], each group through its rows of o_a, back
                let perm = q.view(0, tr * nh * hd * 4)?; // q is read: its buffer takes the permuted rows
                o.permute(&att, &perm, tr, ng, gw)?;
                for (gi, w) in gm.iter().enumerate() {
                    p.matmul(w, tr, (&perm, gi * tr * gw, gw), (&og, gi * tr * ol, ol), false)?;
                }
                o.permute(&og, &oa.view(r0 * ng * ol * 4, tr * ng * ol * 4)?, ng, tr, ol)?;
            }
            t_s = self.lap(p, "attn: o_a", t_s);
        }
        let t_out = t_s;
        let out = p.mm(l, Role::OutB, &oa, t)?;
        self.lap(p, "attn: o_b", t_out);
        tap(&format!("attn_out-{l}"), &out)?;
        Ok(out)
    }

    /// The next `tokens` of a conversation in chunks of at most `prefill_chunk()`: the logits of the last token.
    pub fn feed(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(sess, tokens, &|_| false, tap)?.1)
    }

    /// As `feed`, but `stop(read)` - asked before each chunk past the first - ends it early at a chunk's end
    pub fn feed_until(&self, sess: &mut Session, tokens: &[u32], stop: &(dyn Fn(usize) -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let chunks = chunked(tokens, prefill_chunk());
        let runs = self.runs();
        if chunks.len() >= 2 && runs.len() == 2 && pipeline() {
            return self.feed_pipelined(sess, &chunks, (runs[0], runs[1]), stop, tap);
        }
        let (mut fed, mut logits) = (0, Vec::new());
        for (i, c) in chunks.iter().enumerate() {
            if i > 0 && stop(fed) {
                break;
            }
            logits = self.forward(sess, c, &mut *tap)?;
            fed += c.len();
        }
        Ok((fed, logits))
    }

    /// `feed` on two GPUs with the chunks in a pipeline: a thread runs the first GPU's layers a chunk ahead while this
    /// one runs the second's and the head. The chunks' order on each GPU is kept: the same result.
    fn feed_pipelined(&self, sess: &mut Session, chunks: &[&[u32]], runs: ((usize, u64, u64), (usize, u64, u64)), stop: &(dyn Fn(usize) -> bool + Sync),
                      tap: Tap) -> Result<(usize, Vec<f32>)> {
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        if sess.pos + total > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, total)));
        }
        let rows = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
        for p in &self.parts {
            p.arena_for(rows)?;
        }
        let Session { layers, pos, snapped, .. } = sess;
        let layers: &[Att] = layers;
        let start = *pos;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<u8>>>(1);
        let logits = std::thread::scope(|sc| -> Result<(usize, Vec<f32>)> {
            let ahead = sc.spawn(move || {
                let mut none = |_: &str, _: &DevBuf| Ok(());
                let mut at = start;
                for (i, c) in chunks.iter().enumerate() {
                    if i > 0 && stop(chunks[..i].iter().map(|c| c.len()).sum()) {
                        return;
                    }
                    let r = self.streams(c, &mut none).and_then(|x| self.run_layers(layers, runs.0, x, at, c, &mut none)).and_then(|x| {
                        let mut v = vec![0u8; x.len];
                        x.read(0, &mut v)?;
                        Ok(v)
                    });
                    let r = r.and_then(|v| check_vram(&self.parts[runs.0.0].ops.gpu).map(|_| v));
                    let failed = r.is_err();
                    if tx.send(r).is_err() || failed {
                        return;
                    }
                    at += c.len();
                }
            });
            let mut out = Ok(Vec::new());
            let mut at = start;
            let p1 = &self.parts[runs.1.0];
            for c in chunks.iter() {
                let Ok(v) = rx.recv() else { break };
                let step = (|| -> Result<Vec<f32>> {
                    let v = v?;
                    let x = self.timed(p1, "GPU to GPU", || {
                        let x = DevBuf::new(&p1.ops.gpu, v.len())?;
                        x.write(0, &v)?;
                        Ok(x)
                    })?;
                    let x = self.run_layers(layers, runs.1, x, at, c, &mut *tap)?;
                    let l = self.head(&x, c.len(), 1, &mut *tap)?.pop().unwrap_or_default();
                    check_vram(&p1.ops.gpu)?;
                    Ok(l)
                })();
                match step {
                    Ok(l) => out = Ok(l),
                    Err(e) => {
                        out = Err(e);
                        break;
                    }
                }
                at += c.len();
            }
            drop(rx);
            ahead.join().map_err(|_| Error("the pipeline's first stage panicked".into()))?;
            out.map(|l| (at - start, l))
        })?;
        *pos += logits.0;
        *snapped = None;
        Ok(logits)
    }

    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.forward_rows(sess, tokens, 1, tap)?.pop().unwrap_or_default())
    }

    /// As `forward`, the logits of the last `n_out` tokens (1..=8).
    pub fn forward_rows(&self, sess: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        if t == 0 || t > prefill_chunk() || n_out == 0 || n_out > t.min(MMVQ_COLS) {
            return Err(Error(format!("forward: {t} tokens, {n_out} outputs")));
        }
        for p in &self.parts {
            p.arena_for(t)?;
        }
        let pos0 = sess.pos;
        if pos0 + t > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, t)));
        }
        let mut x = self.streams(tokens, &mut *tap)?;
        let mut staging = Vec::new();
        for (k, run) in self.runs().into_iter().enumerate() {
            if k > 0 {
                x = self.to_part(run.0, &x, &mut staging)?;
            }
            x = self.run_layers(&sess.layers, run, x, pos0, tokens, &mut *tap)?;
        }
        let out = self.head(&x, t, n_out, &mut *tap)?;
        sess.pos += t;
        sess.snapped = (2..=MAX_VERIFY).contains(&t).then_some((pos0, t));
        Ok(out)
    }

    /// The layers in order as runs on one part: (part, first layer, end)
    fn runs(&self) -> Vec<(usize, u64, u64)> {
        let mut v: Vec<(usize, u64, u64)> = Vec::new();
        for (l, &pi) in self.owner.iter().enumerate() {
            match v.last_mut() {
                Some(r) if r.0 == pi => r.2 = l as u64 + 1,
                _ => v.push((pi, l as u64, l as u64 + 1)),
            }
        }
        v
    }

    /// The embedding rows on the first part's GPU, as the 4 streams [t, 4, d]
    fn streams(&self, tokens: &[u32], tap: Tap) -> Result<DevBuf> {
        let d = self.m.g.n_embd as usize;
        let t = tokens.len();
        let p0 = &self.parts[0];
        let m_emb = self.mark(p0);
        let x0 = self.embed(p0, tokens)?;
        tap("inp_embd", &x0)?;
        let x_init = if t <= MMVQ_COLS {
            // each row 4 times on the GPU (kernel copies: no trip through the host behind the expert copies)
            let xi = DevBuf::f32(&p0.ops.gpu, t * 4 * d)?;
            for ti in 0..t {
                for s in 0..4 {
                    p0.ops.kcopy(&xi, (ti * 4 + s) * d * 4, &x0, ti * d * 4, d * 4)?;
                }
            }
            xi
        } else {
            let e0 = x0.to_f32()?;
            let mut xs = vec![0f32; t * 4 * d];
            for ti in 0..t {
                for s in 0..4 {
                    xs[(ti * 4 + s) * d..(ti * 4 + s + 1) * d].copy_from_slice(&e0[ti * d..(ti + 1) * d]);
                }
            }
            DevBuf::from_f32(&p0.ops.gpu, &xs)?
        };
        tap("hc_init", &x_init)?;
        self.lap(p0, "embedding + the 4 streams", m_emb);
        Ok(x_init)
    }

    fn to_part(&self, pi: usize, x: &DevBuf, staging: &mut Vec<u8>) -> Result<DevBuf> {
        let p = &self.parts[pi];
        let next = DevBuf::new(&p.ops.gpu, x.len)?;
        // decode widths: down by a kernel into the previous part's pinned buffer, up through this one's ring (the
        // runtime's copies wait behind the expert store's on the copy engines)
        if pi > 0 && x.len % 4 == 0 {
            let src = &self.parts[pi - 1];
            if let Some((tk, spans)) = src.read_start(&[(x, x.len / 4)])? {
                return self.timed(p, "GPU to GPU", || {
                    let v = src.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default();
                    p.upload(&next, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
                    Ok(next)
                });
            }
        }
        staging.resize(staging.len().max(x.len), 0);
        self.timed(p, "GPU to GPU", || next.copy_from_peer(x, staging))?;
        Ok(next)
    }

    /// mHC before a half of layer `l` on the streams x: h (the half's input), post, comb; then the half's norm into
    /// `normed`
    #[allow(clippy::too_many_arguments)]
    fn hc_pre(&self, p: &Part, l: u64, fn_: Role, base: Role, scale: Role, norm: Role, x: &DevBuf, ws: &[DevBuf; 6], t: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let [flat, h, normed, post, comb, pre] = ws;
        let w = p.mat(l, fn_)?;
        if t <= MMVQ_COLS && w.f32 && w.rows == 24 && std::env::var("NS_HC_FUSED").map_or(true, |v| v != "0") {
            return o.hc_pre_fused(x, &w.buf, p.vec(l, scale)?, p.vec(l, base)?, p.vec(l, norm)?, h, post, comb, pre, normed, &p.arena.f32(t * 32 * 25)?, t, d, eps,
                                  g.hc_eps as f32, g.hc_iters as u32);
        }
        let mixes = p.arena.f32(t * 24)?;
        if w.f32 && w.rows == 24 {
            o.hc_mix(x, &w.buf, &mixes, &p.arena.f32(t * 32 * 25)?, t, 4 * d, eps)?;
        } else {
            o.rms_norm(x, None, flat, t, 4 * d, eps)?;
            p.matmul(w, t, (flat, 0, w.cols), (&mixes, 0, w.rows), false)?;
        }
        o.hc_pre(&mixes, p.vec(l, scale)?, p.vec(l, base)?, x, h, post, comb, pre, t, d, g.hc_eps as f32, g.hc_iters as u32)?;
        o.rms_norm(h, Some(p.vec(l, norm)?), normed, t, d, eps)
    }

    /// One run of layers (`run`: its part, first layer, end) over the stream `x` of the pass's `tokens` at `pos0`
    fn run_layers(&self, layers: &[Att], run: (usize, u64, u64), x: DevBuf, pos0: usize, tokens: &[u32], tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let t = tokens.len();
        self.decoding.store(profile_split() && t <= MMVQ_COLS, std::sync::atomic::Ordering::Relaxed);
        let d = g.n_embd as usize;
        let p = &self.parts[run.0];
        let o = &p.ops;
        let gpu = &o.gpu;
        let mut ring = vec![x, DevBuf::f32(gpu, t * 4 * d)?, DevBuf::f32(gpu, t * 4 * d)?];
        let mut ix = 0usize;
        // flat, h, normed, post, comb, pre
        let ws = [DevBuf::f32(gpu, t * 4 * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * 4)?, DevBuf::f32(gpu, t * 16)?,
                  DevBuf::f32(gpu, t * 4)?];
        for l in run.1..run.2 {
            p.arena.reset();
            let x = &ring[ix];
            let (x1, x2) = (&ring[(ix + 1) % 3], &ring[(ix + 2) % 3]);
            let (normed, post, comb) = (&ws[2], &ws[3], &ws[4]);
            self.timed(p, "hc pre", || self.hc_pre(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, x, &ws, t))?;
            tap(&format!("attn_norm-{l}"), normed)?;
            let att = self.attention(p, l, normed, &layers[l as usize], pos0, t, &mut *tap)?;
            o.hc_post(&att, x, post, comb, x1, t, d)?;
            tap(&format!("hc_attn_post-{l}"), x1)?;
            self.timed(p, "hc pre", || self.hc_pre(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, x1, &ws, t))?;
            tap(&format!("ffn_norm-{l}"), normed)?;
            let ffn = self.moe(p, l, tokens, normed, &mut *tap)?;
            tap(&format!("ffn_out-{l}"), &ffn)?;
            o.hc_post(&ffn, x1, post, comb, x2, t, d)?;
            tap(&format!("l_last-{l}"), x2)?;
            ix = (ix + 2) % 3;
        }
        Ok(ring.swap_remove(ix))
    }

    /// The head after the last layer (`x` [t, 4, d], on the last part) for the last `n_out` rows: the streams by the
    /// head's mHC weights, the norm, the output matrix
    fn head(&self, x: &DevBuf, t: usize, n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.m.g;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        let m_head = self.mark(p);
        p.arena.reset();
        let xv = x.view((t - n_out) * 4 * d * 4, n_out * 4 * d * 4)?;
        let flat = p.arena.f32(n_out * 4 * d)?;
        o.rms_norm(&xv, None, &flat, n_out, 4 * d, eps)?;
        let w = p.mat(0, Role::HeadHcFn)?;
        let mixes = p.arena.f32(n_out * 4)?;
        p.matmul(w, n_out, (&flat, 0, w.cols), (&mixes, 0, w.rows), false)?;
        let h = p.arena.f32(n_out * d)?;
        o.hc_head(&mixes, p.vec(0, Role::HeadHcScale)?, p.vec(0, Role::HeadHcBase)?, &xv, &h, n_out, d, g.hc_eps as f32)?;
        tap("hc_head", &h)?;
        let out = p.arena.f32(n_out * d)?;
        o.rms_norm(&h, Some(p.vec(0, Role::OutputNorm)?), &out, n_out, d, eps)?;
        tap("result_norm", &out)?;
        let logits = p.mm(0, Role::Output, &out, n_out)?;
        tap("result_output", &logits)?;
        let n = n_out * g.n_vocab as usize;
        let v = match p.read_start(&[(&logits, n)])? {
            Some((tk, spans)) => p.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default(),
            None => logits.to_f32()?,
        };
        self.lap(p, "head", m_head);
        let vocab = g.n_vocab as usize;
        Ok(v.chunks(vocab).map(|c| c.to_vec()).collect())
    }

    /// The decode requests per expert since load blended into the profile at `path` (old weights halved)
    pub fn save_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let mut prof: HashMap<(u64, u64), f64> = read_profile(path).unwrap_or_default();
        for v in prof.values_mut() {
            *v *= 0.5;
        }
        let mut n = 0;
        for p in &self.parts {
            let c = p.experts.lock().unwrap();
            for (k, u) in &c.usage {
                *prof.entry(*k).or_insert(0.0) += *u as f64;
                n += 1;
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

    pub fn hold_arena(&self, on: bool) {
        for p in &self.parts {
            p.hold.store(on, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// One decode step of several conversations at once: `tokens[b]` the next token of `sess[b]`. What treats rows
    /// alike (the mHC, the experts, the head) runs once for all rows; the attention per session on its row.
    pub fn forward_batch(&self, sess: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.m.g;
        let t = tokens.len();
        if t == 0 || t != sess.len() || t > MMVQ_COLS {
            return Err(Error(format!("forward_batch: {t} tokens for {} sessions (1..={MMVQ_COLS})", sess.len())));
        }
        if let Some(s) = sess.iter().find(|s| s.pos + 1 > s.max_ctx) {
            return Err(Error(format!("the context is {} tokens; no more fit", s.max_ctx)));
        }
        for p in &self.parts {
            p.arena_for(t)?;
        }
        let d = g.n_embd as usize;
        let mut x = self.streams(tokens, &mut *tap)?;
        let mut staging = vec![0u8; t * 4 * d * 4];
        for (k, run) in self.runs().into_iter().enumerate() {
            let p = &self.parts[run.0];
            let o = &p.ops;
            let gpu = &o.gpu;
            if k > 0 {
                x = self.to_part(run.0, &x, &mut staging)?;
            }
            let mut ring = vec![x, DevBuf::f32(gpu, t * 4 * d)?, DevBuf::f32(gpu, t * 4 * d)?];
            let mut ix = 0usize;
            let ws = [DevBuf::f32(gpu, t * 4 * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * 4)?, DevBuf::f32(gpu, t * 16)?,
                      DevBuf::f32(gpu, t * 4)?];
            for l in run.1..run.2 {
                p.arena.reset();
                let xs = &ring[ix];
                let (x1, x2) = (&ring[(ix + 1) % 3], &ring[(ix + 2) % 3]);
                let (normed, post, comb) = (&ws[2], &ws[3], &ws[4]);
                self.timed(p, "hc pre", || self.hc_pre(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, xs, &ws, t))?;
                let att = p.arena.f32(t * d)?;
                for (b, s) in sess.iter().enumerate() {
                    let nb = normed.view(b * d * 4, d * 4)?;
                    let y = self.attention(p, l, &nb, &s.layers[l as usize], s.pos, 1, &mut *tap)?;
                    att.copy_within(b * d * 4, &y, 0, d * 4)?;
                }
                o.hc_post(&att, xs, post, comb, x1, t, d)?;
                self.timed(p, "hc pre", || self.hc_pre(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, x1, &ws, t))?;
                let ffn = self.moe(p, l, tokens, normed, &mut *tap)?;
                o.hc_post(&ffn, x1, post, comb, x2, t, d)?;
                ix = (ix + 2) % 3;
            }
            x = ring.swap_remove(ix);
        }
        let out = self.head(&x, t, t, &mut *tap)?;
        for s in sess.iter_mut() {
            s.pos += 1;
            s.snapped = None;
        }
        Ok(out)
    }

    /// Keeps only the first `keep` tokens of the last forward pass (a verify pass): the position back. The rings and
    /// caches need nothing: positions bound what is read, and the rows the next tokens write replace the rest.
    pub fn rollback(&self, sess: &mut Session, keep: usize) -> Result<()> {
        let (pos0, t) = sess.snapped.take().ok_or("rollback: the last pass was not a verify pass")?;
        if keep == 0 || keep >= t {
            return Err(Error(format!("rollback: keep {keep} of {t} rows")));
        }
        sess.pos = pos0 + keep;
        Ok(())
    }

    pub fn decoder(&self, logits: Vec<f32>) -> Decoder {
        Decoder { logits, next: None, look: None, ng_drafted: 0, ng_accepted: 0 }
    }

    pub fn decoder_after(&self, next: u32) -> Decoder {
        Decoder { logits: Vec::new(), next: Some(next), look: None, ng_drafted: 0, ng_accepted: 0 }
    }

    /// The next committed token(s): one, or more when prompt-lookup drafts (NS_NGRAM) are accepted - each exactly when
    /// the token drawn for its position is the draft itself.
    pub fn step(&self, sess: &mut Session, dec: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let ng: Vec<u32> = match (&dec.look, dec.next) {
            (Some(l), Some(_)) => {
                let mut v = l.propose(ngram_k());
                v.truncate(sess.max_ctx.saturating_sub(sess.pos + 1));
                v
            }
            _ => Vec::new(),
        };
        let (out, tail) = if let (Some(x), true) = (dec.next, ng.len() >= 2) {
            let mut toks = vec![x];
            toks.extend_from_slice(&ng);
            let rows = self.forward_rows(sess, &toks, toks.len(), &mut *tap)?;
            let mut out = Vec::with_capacity(toks.len());
            for (i, r) in rows.iter().enumerate() {
                let y = smp.sample(r);
                out.push(y);
                if i < ng.len() {
                    dec.ng_drafted += 1;
                    if y == ng[i] {
                        dec.ng_accepted += 1;
                        continue;
                    }
                }
                break;
            }
            if out.len() < toks.len() {
                self.rollback(sess, out.len())?;
            }
            let tail = *out.last().unwrap();
            (out, tail)
        } else {
            if let Some(x) = dec.next.take() {
                dec.logits = self.forward(sess, &[x], &mut *tap)?;
            }
            let y = smp.sample(&dec.logits);
            (vec![y], y)
        };
        if let Some(l) = &mut dec.look {
            for &t in &out {
                l.push(t);
            }
        }
        dec.next = Some(tail);
        Ok(out)
    }

    /// The MoE half of layer `l` on x [t, d] (the pass's `tokens`): the router (on the host) or the token's hash
    /// table, the shared expert, the routed experts grouped by expert (glm5next's paths).
    fn moe(&self, p: &Part, l: u64, tokens: &[u32], x: &DevBuf, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let t = tokens.len();
        let (d, ne, used) = (g.n_embd as usize, g.n_expert as usize, g.n_expert_used as usize);
        let lim = g.swiglu_limit as f32;
        let logits = self.timed(p, "router", || p.mm(l, Role::Router, x, t))?;
        let pf = prefetch_limit();
        let guess = if t <= MMVQ_COLS && pf > 0 && !g.is_hash(l + 1) && p.mat(l + 1, Role::Router).is_ok() {
            Some(p.mm(l + 1, Role::Router, x, t)?)
        } else {
            None
        };
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        // the logits on their way down (kernels into pinned memory), the shared expert queued behind them: it runs
        // while the host picks the routed experts
        let mut down: Vec<(&DevBuf, usize)> = vec![(&logits, t * ne)];
        if let Some(gl) = &guess {
            down.push((gl, t * ne));
        }
        let pending = p.read_start(&down)?;
        let y = self.timed(p, "shared expert", || {
            let mut pj = p.mm_many(l, &[Role::ShGate, Role::ShUp], x, t)?.into_iter();
            let (sg, su) = (pj.next().unwrap(), pj.next().unwrap());
            o.swiglu_clamp(&sg, &su, &sg, t * g.ffn_expert as usize * g.n_expert_shared as usize, lim)?;
            p.mm(l, Role::ShDown, &sg, t)
        })?;
        let tw = Instant::now();
        let (lv, guess) = match pending {
            Some((tk, spans)) => {
                let mut v = p.read_finish(tk, &spans)?.into_iter();
                (v.next().unwrap_or_default(), v.next())
            }
            None => (logits.to_f32()?, match &guess { Some(gl) => Some(gl.to_f32()?), None => None }),
        };
        ROUTER_WAIT_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let hash = self.hash.get(&l);
        let bias = if hash.is_none() { Some(p.router_bias(l)?) } else { None };
        let th = Instant::now();
        // each row's experts (by score + bias, or the table's for its token) and their weights: the scores
        // normalized over them, x the scale
        let pick = |ti: usize| -> Vec<(usize, f32)> {
            let pr: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| sqrt_softplus(*z)).collect();
            let sel: Vec<usize> = match (hash, &bias) {
                (Some(tab), _) => tab[tokens[ti] as usize * used..(tokens[ti] as usize + 1) * used].iter().map(|x| *x as usize).collect(),
                (None, Some(b)) => {
                    let key = |i: usize| pr[i] + b[i];
                    let mut order: Vec<usize> = (0..ne).collect();
                    order.select_nth_unstable_by(used - 1, |a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
                    let mut s = order[..used].to_vec();
                    s.sort_by(|a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
                    s
                }
                (None, None) => Vec::new(),
            };
            let sum: f32 = sel.iter().map(|i| pr[*i]).sum::<f32>().max(6.103_516e-5);
            sel.iter().map(|&i| (i, if g.expert_norm { pr[i] / sum } else { pr[i] } * g.expert_scale as f32)).collect()
        };
        let rows: Vec<Vec<(usize, f32)>> = if t >= 256 {
            let parts = 16.min(t);
            let per = t.div_ceil(parts);
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..parts).map(|k| { let pick = &pick; sc.spawn(move || (k * per..((k + 1) * per).min(t)).map(pick).collect::<Vec<_>>()) }).collect();
                hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            })
        } else {
            (0..t).map(pick).collect()
        };
        let mut by: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
        for (ti, sel) in rows.into_iter().enumerate() {
            for (i, w) in sel {
                by.entry(i).or_default().push((ti as i32, w));
            }
        }
        ROUTE_HOST_NS.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // the shared expert (queued above), then each routed expert added into it
        tap(&format!("ffn_shexp-{l}"), &y)?;
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        let promote = t <= MMVQ_COLS && !std::env::var("NS_DECODE_DIRECT").is_ok_and(|v| v == "1");
        let (slots, arriving, copies) = self.timed(p, "expert misses (swaps / file)", || p.ensure(&self.m, l, &need, promote))?;
        let parts = expert_parts(&self.m, l)?;
        let (f, cols) = (g.ffn_expert as usize, d);
        self.timed(p, "routed experts", || {
            let total: usize = by.values().map(|v| v.len()).sum();
            let mut tok = Vec::with_capacity(total);
            let mut per_token: Vec<Vec<(i32, f32)>> = vec![Vec::new(); t];
            for list in by.values() {
                for (ti, w) in list {
                    per_token[*ti as usize].push((tok.len() as i32, *w));
                    tok.push(*ti);
                }
            }
            let mut ints: Vec<i32> = tok.clone();
            let mut t_ptr = vec![0i32];
            let mut ent = Vec::with_capacity(total);
            let mut wts = Vec::with_capacity(total);
            for list in &per_token {
                for (row, w) in list {
                    ent.push(*row);
                    wts.push(*w);
                }
                t_ptr.push(ent.len() as i32);
            }
            let base = ints.len();
            ints.extend(&t_ptr);
            ints.extend(&ent);
            let wb = p.arena.f32(wts.len())?;
            p.upload(&wb, &wts.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            // big prompt chunks: each expert expanded to fp16 once a chunk, its tokens through oneMKL's half GEMM
            let f16_min = f16_min();
            if f16_min > 0 && t >= f16_min && t > MMVQ_COLS {
                if let Some((_, tk)) = copies {
                    o.await_ticket(tk)?;
                }
                let mut gtok: Vec<i32> = Vec::with_capacity(total);
                let mut gw: Vec<f32> = Vec::with_capacity(total);
                for list in by.values() {
                    for (ti, w) in list {
                        gtok.push(*ti);
                        gw.push(*w);
                    }
                }
                let tb = p.arena.bytes(total * 4)?;
                p.upload(&tb, &gtok.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let wtb = p.arena.f32(total)?;
                p.upload(&wtb, &gw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let esimd = esimd_max().map_or(p.esimd_gu, |m| m.min(p.esimd_gu));
                let iq2 = parts.gate.2.code() == 16 && parts.up.2.code() == 16 && parts.up.0 == parts.gate.0 + parts.gate.1;
                let esimd_rows = |n: usize| (n <= esimd && iq2).then(|| n.max(32).next_power_of_two());
                let nmax = by.values().map(|l| fused_rows(l.len()).max(esimd_rows(l.len()).unwrap_or(0))).max().unwrap_or(32);
                let xh = p.arena.bytes(nmax * d * 2)?;
                let w16 = p.arena.bytes(2 * f * d * 2)?;
                let wu = w16.view(f * d * 2, f * d * 2)?;
                let gu = p.arena.f32(nmax * 2 * f)?;
                let hh = p.arena.bytes(nmax * f * 2)?;
                let dn = p.arena.f32(nmax * d)?;
                let sb = parts.down.0 + parts.down.1;
                const RING: usize = 3;
                let ring = (0..RING).map(|_| p.arena.bytes(sb)).collect::<Result<Vec<_>>>()?;
                let hosts: Vec<usize> = slots.iter().enumerate().filter(|(_, s)| s.1).map(|(k, _)| k).collect();
                let mut copied: Vec<Option<(i64, usize)>> = vec![None; slots.len()];
                let mut freed: [Option<i64>; RING] = [None; RING];
                let mut next = 0;
                let issue = |next: &mut usize, freed: &[Option<i64>; RING], copied: &mut [Option<(i64, usize)>]| -> Result<()> {
                    if let Some(&k) = hosts.get(*next) {
                        let r = *next % RING;
                        copied[k] = Some((o.stream_copy(&ring[r], 0, &slots[k].0, 0, sb, freed[r])?, r));
                        *next += 1;
                    }
                    Ok(())
                };
                for _ in 0..RING {
                    issue(&mut next, &freed, &mut copied)?;
                }
                let mut row = 0;
                for (k, (list, (hbuf, host))) in by.values().zip(&slots).enumerate() {
                    let n = list.len();
                    let toks = tb.view(row * 4, n * 4)?;
                    let (buf, slot) = match copied[k] {
                        Some((ticket, r)) if *host => {
                            o.await_ticket(ticket)?;
                            (&ring[r], Some(r))
                        }
                        _ => (hbuf, None),
                    };
                    o.gather_f16(x, &toks, &xh, n, d)?;
                    if let Some(rows) = esimd_rows(n) {
                        o.moe_fused_gu_esimd(&xh, buf, parts.gate.0, &gu, rows, 2 * f, d)?;
                    } else if n <= fused_max() && iq2 {
                        o.moe_fused_gu(&xh, buf, parts.gate.0, &gu, fused_rows(n), 2 * f, d)?;
                    } else {
                        p.expand_f16(parts.gate.2, buf, parts.gate.0, parts.gate.1, f * d, &w16)?;
                        p.expand_f16(parts.up.2, buf, parts.up.0, parts.up.1, f * d, &wu)?;
                        o.gemm_f16(n, 2 * f, d, (&xh, 0, d), (&w16, 0), (&gu, 0, 2 * f), false)?;
                    }
                    o.swiglu_gu_f16(&gu, &hh, n, f, lim)?;
                    p.expand_f16(parts.down.2, buf, parts.down.0, parts.down.1, d * f, &w16)?;
                    if let Some(r) = slot {
                        freed[r] = Some(o.mark()?);
                        issue(&mut next, &freed, &mut copied)?;
                    }
                    o.gemm_f16(n, d, f, (&hh, 0, f), (&w16, 0), (&dn, 0, d), false)?;
                    o.scatter_add(&y, &dn, &toks, &wtb.view(row * 4, n * 4)?, n, d)?;
                    row += n;
                }
                return Ok(());
            }
            // decode widths with this engine's own multi-expert products: every expert's gate and up in one launch, the
            // clamped SwiGLU, every down in one - the resident experts first, then the ones being swapped in
            if t <= MMVQ_COLS && std::env::var("NS_DS4_MULTI").map_or(true, |v| v != "0") && parts.gate.2 == parts.up.2 && o.multi_supported(parts.gate.2.code(), d) && o.multi_supported(parts.down.2.code(), f) {
                let lists: Vec<&Vec<(i32, f32)>> = by.values().collect();
                let (now, later): (Vec<usize>, Vec<usize>) = (0..lists.len()).partition(|&k| !arriving[k]);
                let xq = p.arena.bytes(o.q8_bytes(d, t))?;
                o.quantize_q8((x, 0), &xq, d, t)?;
                let dn = p.arena.f32(total * d)?;
                let gu = p.arena.f32(2 * total * f)?;
                // the entries in launch order: (token, weight) per row of dn
                let mut rows: Vec<(i32, f32)> = Vec::with_capacity(total);
                for (set, wait) in [(&now, false), (&later, true)] {
                    let n: usize = set.iter().map(|&k| lists[k].len()).sum();
                    if n == 0 {
                        continue;
                    }
                    if wait {
                        if let Some((_, all)) = copies {
                            self.timed(p, "experts: waiting for swaps", || o.await_ticket(all))?;
                        }
                    }
                    let e0 = rows.len();
                    let (mut pg, mut pu, mut pd, mut tk) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
                    for &k in set.iter() {
                        let base = slots[k].0.ptr() as u64;
                        for &(ti, w) in lists[k] {
                            pg.push(base + parts.gate.0 as u64);
                            pu.push(base + parts.up.0 as u64);
                            pd.push(base + parts.down.0 as u64);
                            tk.push(ti);
                            rows.push((ti, w));
                        }
                    }
                    // gate | up: 2n entries (the gates, then the ups, each against its token); down: n against h's rows
                    let mut tb: Vec<u8> = Vec::with_capacity(2 * n * 12 + n * 12);
                    for v in pg.iter().chain(&pu) {
                        tb.extend(v.to_le_bytes());
                    }
                    for v in tk.iter().chain(&tk) {
                        tb.extend(v.to_le_bytes());
                    }
                    let gtab = p.arena.bytes(tb.len())?;
                    p.upload(&gtab, &tb)?;
                    let mut db: Vec<u8> = Vec::with_capacity(n * 12);
                    for v in &pd {
                        db.extend(v.to_le_bytes());
                    }
                    for i in 0..n as i32 {
                        db.extend(i.to_le_bytes());
                    }
                    let dtab = p.arena.bytes(db.len())?;
                    p.upload(&dtab, &db)?;
                    let g0 = 2 * e0 * f;
                    self.timed(p, "experts: gate | up", || o.mmvq_multi(parts.gate.2.code(), &gtab, 2 * n, &xq, t, (&gu, g0), d, f))?;
                    let (gv, uv) = (gu.view(g0 * 4, n * f * 4)?, gu.view((g0 + n * f) * 4, n * f * 4)?);
                    let hq = p.arena.bytes(o.q8_bytes(f, n))?;
                    self.timed(p, "experts: swiglu, quantize", || {
                        o.swiglu_clamp(&gv, &uv, &gv, n * f, lim)?;
                        o.quantize_q8((&gv, 0), &hq, f, n)
                    })?;
                    self.timed(p, "experts: down", || o.mmvq_multi(parts.down.2.code(), &dtab, n, &hq, n, (&dn, e0 * d), f, d))?;
                }
                // the combine: per token its rows of dn, with their weights
                let mut t_ptr = vec![0i32];
                let mut ent = Vec::with_capacity(total);
                let mut wv = Vec::with_capacity(total);
                for ti in 0..t as i32 {
                    for (r, &(rt, w)) in rows.iter().enumerate() {
                        if rt == ti {
                            ent.push(r as i32);
                            wv.push(w);
                        }
                    }
                    t_ptr.push(ent.len() as i32);
                }
                let cb = p.arena.bytes((t + 1 + total) * 4)?;
                p.upload(&cb, &t_ptr.iter().chain(&ent).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let wb2 = p.arena.f32(total)?;
                p.upload(&wb2, &wv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                return self.timed(p, "experts: combine", || o.moe_combine(&y, &dn, &cb, &wb2, t, total, d));
            }
            if parts.gate.2 == parts.up.2 && o.moe_grouped_supported(parts.gate.2.code(), parts.down.2.code(), d, f) {
                let lists: Vec<&Vec<(i32, f32)>> = by.values().collect();
                let mut first = Vec::with_capacity(lists.len());
                let mut e0 = 0i32;
                for list in &lists {
                    first.push(e0);
                    e0 += list.len() as i32;
                }
                let table = |gs: &[usize]| -> Result<(DevBuf, usize, usize)> {
                    let n: usize = gs.iter().map(|&k| lists[k].len()).sum();
                    let mut tb: Vec<u8> = Vec::with_capacity(gs.len() * 8 + (gs.len() + 2 + 2 * n) * 4);
                    for &k in gs {
                        tb.extend((slots[k].0.ptr() as u64).to_le_bytes());
                    }
                    let mut start = 0i32;
                    tb.extend(start.to_le_bytes());
                    for &k in gs {
                        start += lists[k].len() as i32;
                        tb.extend(start.to_le_bytes());
                    }
                    tb.extend((gs.len() as i32).to_le_bytes());
                    for &k in gs {
                        for j in 0..lists[k].len() as i32 {
                            tb.extend((first[k] + j).to_le_bytes());
                        }
                    }
                    for &k in gs {
                        for j in 0..lists[k].len() {
                            tb.extend(tok[first[k] as usize + j].to_le_bytes());
                        }
                    }
                    let b = p.arena.bytes(tb.len())?;
                    p.upload(&b, &tb)?;
                    Ok((b, gs.len(), n))
                };
                let xq = p.arena.bytes(o.q8_1_bytes(d, t))?;
                o.quantize_q8_1((x, 0), &xq, d, t)?;
                let scratch = p.arena.bytes(o.moe_scratch_bytes(total, f))?;
                let dn = p.arena.f32(total * d)?;
                static PROMPT_LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let pl = *PROMPT_LANES.get_or_init(|| std::env::var("NS_PROMPT_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(32));
                static DECODE_LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let dl = *DECODE_LANES.get_or_init(|| std::env::var("NS_DECODE_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(4));
                let lanes = if t > MMVQ_COLS { pl } else { dl };
                let (now, later): (Vec<usize>, Vec<usize>) = (0..lists.len()).partition(|&k| !arriving[k]);
                for (gs, wait) in [(now, false), (later, true)] {
                    if gs.is_empty() {
                        continue;
                    }
                    let (tb, groups, n) = table(&gs)?;
                    match copies.filter(|_| wait) {
                        Some((gu, all)) if split_copy() => {
                            o.await_ticket(gu)?;
                            o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, n, &xq, &scratch, &dn, lim, lanes, 1)?;
                            o.await_ticket(all)?;
                            o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, n, &xq, &scratch, &dn, lim, lanes, 2)?;
                        }
                        Some((_, all)) => {
                            o.await_ticket(all)?;
                            o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, n, &xq, &scratch, &dn, lim, lanes, 0)?;
                        }
                        None => o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, n, &xq, &scratch, &dn, lim, lanes, 0)?,
                    }
                }
                let cb = p.arena.bytes((t + 1 + total) * 4)?;
                p.upload(&cb, &t_ptr.iter().chain(&ent).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                return o.moe_combine(&y, &dn, &cb, &wb, t, total, d);
            }
            // the types the grouped kernels do not take (MXFP4's down): each expert's three products
            let t_one = self.mark(p);
            if let Some((_, tk)) = copies {
                o.await_ticket(tk)?;
            }
            let ib = p.arena.bytes(ints.len() * 4)?;
            p.upload(&ib, &ints.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            let xe = p.arena.f32(total * d)?;
            o.gather(x, &ib, &xe, total, d)?;
            let gt = p.arena.f32(total * f)?;
            let ut = p.arena.f32(total * f)?;
            let dn = p.arena.f32(total * d)?;
            let mut row = 0;
            for (list, (buf, _)) in by.values().zip(&slots) {
                let n = list.len();
                p.expert_into(buf, 0, parts.gate, f, cols, (&xe, row * cols), (&gt, row * f), n, true)?;
                p.expert_into(buf, 0, parts.up, f, cols, (&xe, row * cols), (&ut, row * f), n, false)?;
                row += n;
            }
            o.swiglu_clamp(&gt, &ut, &gt, total * f, lim)?;
            let mut row = 0;
            for (list, (buf, _)) in by.values().zip(&slots) {
                let n = list.len();
                p.expert_into(buf, 0, parts.down, d, f, (&gt, row * f), (&dn, row * d), n, true)?;
                row += n;
            }
            let ints_view = p.arena.bytes((t + 1 + total) * 4)?;
            ints_view.copy_within(0, &ib, base * 4, (t + 1 + total) * 4)?;
            o.moe_combine(&y, &dn, &ints_view, &wb, t, total, d)?;
            self.lap(p, "experts: one at a time (MXFP4 / IQ2_S layers)", t_one);
            Ok(())
        })?;
        if let Some(gv) = guess {
            let gb = p.router_bias(l + 1)?;
            let mut miss: BTreeMap<usize, f32> = BTreeMap::new();
            for ti in 0..t {
                let pr: Vec<f32> = gv[ti * ne..(ti + 1) * ne].iter().map(|z| sqrt_softplus(*z)).collect();
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|a, b| (pr[*b] + gb[*b]).total_cmp(&(pr[*a] + gb[*a])));
                for (rank, &e) in order.iter().take(GUESS_HIT.len()).enumerate() {
                    *miss.entry(e).or_insert(1.0) *= 1.0 - GUESS_HIT[rank];
                }
            }
            let th = prefetch_p();
            let mut want: Vec<(usize, f32)> = miss.into_iter().map(|(e, q)| (e, 1.0 - q)).filter(|x| x.1 >= th).collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1));
            let want: Vec<u64> = want.into_iter().map(|(e, _)| e as u64).collect();
            self.timed(p, "expert prefetch", || p.prefetch(&self.m, l + 1, &want, pf))?;
        }
        Ok(y)
    }
}
