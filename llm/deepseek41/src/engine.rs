//! DeepSeek-V4.1-Flash on one or more GPUs: `model.rs`'s outline in order, float32 activations. The layers are split
//! over the GPUs at a kv source (a source's compressed keys stay with the layers that read them; B70 + B65: 0-19 |
//! 20-39), each `Part` holding its layers' weights in their stored form (q4_K, fp8_32x32, float32 for the small and
//! BF16 ones), its expert stores (VRAM slots, pinned host slots, the rest read from the file: deepseek4's exclusive
//! two-level cache), and the token stream [T, 4, hidden] crossing to the next GPU through host memory - with the mix
//! the next half collapses it by ([T, 4] after the streams).
//!
//! The routed experts are VQ: a slot holds an expert's three DQV3 payloads as the file has them (gate | up | down); a
//! decode pass multiplies them straight from the indices (ds41.h: vq_mmv, every expert of a layer in one launch), a
//! prompt pass expands each to fp16 and runs oneMKL's half GEMM over its tokens.
//!
//! The draft (DSpark, three towers) runs on the last part: after each pass the committed rows' stream means at the
//! target layers go through main_proj into each tower's window ring; a draft block is [token, 4 noise] over the towers,
//! the head and the Markov bias, verified by the main model in one pass.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};

use crate::engram::{self, Engram};
use crate::model::{Model, Role};
use crate::ops::Ops;
use nextsycl_gguf::{GType, Tensor};

use nextsycl_llm::{GpuInfo, Sampler, Tap};

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;
/// the decode kernels take up to this many rows (tokens) at once
pub const MMVQ_COLS: usize = 8;
/// rows of a verify pass (the token and its drafts)
pub const MAX_VERIFY: usize = 8;
/// expert slots and the mirror are allocated in chunks of this size
const CHUNK: usize = 2 << 30;
/// the pinned rings' sizes
const UP_BYTES: usize = 8 << 20;
const BACK_BYTES: usize = 4 << 20;
/// the arena of decode and short prompt chunks; past it, a pass takes the big one back from the expert store
const SMALL_ARENA: usize = 512 << 20;
/// the indexer's scores per pass of rows (floats)
const IDX_S_FLOATS: usize = 32 << 20;
/// the draft towers' window (positions of main_x a tower's ring holds)
const MTP_WIN: usize = 128;
/// a pass's last rows whose stream means the towers keep
const MTP_KEEP: usize = MTP_WIN + MAX_VERIFY;

/// Time the host spent waiting for the routers' logits / choosing experts (all layers)
pub static ROUTER_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ROUTE_HOST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Decode: the host's time from a layer's router logits to its expert products queued (the GPU has nothing queued
/// meanwhile but the shared expert), and of it the file reads
pub static TURN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FILE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Prompt tokens per forward pass (NS_PREFILL_CHUNK, default 2048, 64..8192)
pub fn prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_PREFILL_CHUNK", 2048).clamp(64, 8192))
}

/// Bytes of each GPU's big arena (NS_ARENA_MIB; default 1 GiB or 0.5 MiB a prompt token)
pub fn arena_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("NS_ARENA_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) => m << 20,
        None => (1usize << 30).max(prefill_chunk() * (512 << 10)),
    })
}

/// A prompt pass's experts with at most this many tokens multiplied from their indices (the decode kernels), the rest
/// expanded to fp16 for a GEMM (NS_DS41_PROMPT_MMV, default 0: none - the XMX product from the indices, vq_gemm, is faster at every count)
fn prompt_mmv_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_MMV", 0))
}
/// The decode kernels' variant for a prompt's few-token experts (NS_DS41_PROMPT_MMV_V, default 3: a sub-group a row, the
/// fp16 codebook - its columns share each index's weights)
fn prompt_mmv_variant() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_MMV_V", 3) as i32)
}
/// A prompt pass's VRAM staging slots for the experts it reads from host memory or the file (NS_DS41_PROMPT_RING,
/// default 24; 0: read in place from host memory, the file's swapped in)
fn prompt_ring() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_RING", 24))
}
/// A prompt pass's experts with more tokens than NS_DS41_PROMPT_MMV and at most this many multiplied on XMX from their
/// indices (ns_ds41_vq_gemm), more expanded to fp16 for a GEMM (NS_DS41_VQ_GEMM_MAX, default 100000: all; 0: none)
fn vq_gemm_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_VQ_GEMM_MAX", 100_000))
}
/// A prompt pass's pinned slots for the experts only the file holds (NS_DS41_PROMPT_FILE_SLOTS, default 64; 0: read a
/// wave at a time on the host thread)
fn prompt_file_slots() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_PROMPT_FILE_SLOTS", 64))
}
/// NS_DS41_PIPE_LOG=1: each prompt chunk's time on each GPU of the pipeline; 2: and each layer's, its attention and
/// its experts apart (a GPU sync between them: no overlap across layers)
fn pipe_log() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_PIPE_LOG").is_ok_and(|v| v == "1" || v == "2"))
}
fn pipe_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_PIPE_LOG").is_ok_and(|v| v == "2"))
}
/// Token slots of one batch of a prompt's few-token experts (their gate | up and h in the arena)
const PROMPT_MMV_SLOTS: usize = 1024;

/// Dense products of this many rows or more in fp16 on the XMX units (NS_DS41_DENSE_F16_MIN, default 64)
fn dense_f16_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_DENSE_F16_MIN", 64))
}

/// Experts swapped in ahead of their layer at most (NS_PREFETCH, default 8; 0 none), from a chance of use of
/// NS_PREFETCH_P (default 0.6)
fn prefetch_limit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_PREFETCH", 8))
}
fn prefetch_p() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFETCH_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6))
}
/// How often the next layer's router run on this layer's input has its k-th best among the next layer's picks (the
/// glm5next / deepseek4 measurement: a first guess here)
const GUESS_HIT: [f32; 16] = [0.96, 0.92, 0.86, 0.77, 0.66, 0.56, 0.45, 0.36, 0.29, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08];

/// Decode's q4_K products by this engine's kernel on float32 activations, o_a's groups in one launch (NS_DS41_Q4K=0:
/// the shared Q8_1 quantization and product, a launch a group)
fn q4k_own() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_Q4K").map_or(true, |v| v != "0"))
}

/// Decode's mHC mixes through the fused kernels (NS_HC_FUSED=0: the mixes, then the Sinkhorn)
fn hc_fused() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_HC_FUSED").map_or(true, |v| v != "0"))
}

/// Whether the big arena lends its memory to the expert store while no prompt chunk needs it (NS_LEND=0: no)
fn lend_arena() -> bool {
    std::env::var("NS_LEND").map_or(true, |v| v != "0")
}

/// Whether a prompt of several chunks on two GPUs runs as a pipeline (NS_PIPELINE=0: in turn)
fn pipeline() -> bool {
    std::env::var("NS_PIPELINE").map_or(true, |v| v != "0")
}

/// Whether the DSpark towers load and draft (NS_DS41_MTP=1; default off)
pub fn mtp_on() -> bool {
    std::env::var("NS_DS41_MTP").is_ok_and(|v| v == "1")
}

/// Drafts a verify pass checks (NS_DS41_DRAFTS, 1..7, default 2: the third is rarely kept)
pub fn mtp_drafts() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS41_DRAFTS", 2).clamp(1, MAX_VERIFY - 1))
}

/// VRAM each GPU keeps free beyond the plan (NS_DS41_FREE_GIB, default 1.5: the xe driver has no out-of-memory path)
fn keep_free() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| (std::env::var("NS_DS41_FREE_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.5) * (1u64 << 30) as f64) as usize)
}

/// The pinned host memory for experts in all (NS_DS41_PIN_GIB, default 40, at most what is available less 12 GiB)
fn pin_cap() -> usize {
    (std::env::var("NS_DS41_PIN_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(40.0) * (1u64 << 30) as f64) as usize
}

/// The expert profile (NS_EXPERT_PROFILE: "layer expert weight" lines)
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

/// `tokens` in chunks of `c`, the last two split evenly when the last would be under half a chunk
fn chunked(tokens: &[u32], c: usize) -> Vec<&[u32]> {
    let mut v: Vec<&[u32]> = tokens.chunks(c).collect();
    if v.len() >= 2 && v[v.len() - 1].len() < c / 2 {
        let n = v.len();
        let both = &tokens[(n - 2) * c..];
        let half = both.len().div_ceil(2);
        v.truncate(n - 2);
        v.push(&both[..half]);
        v.push(&both[half..]);
    }
    v
}

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

/// The highest logit's token
pub fn argmax(l: &[f32]) -> u32 {
    l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

/// sqrt(softplus(z)): the router's score
fn sqrt_softplus(z: f32) -> f32 {
    (if z > 20.0 { z } else { z.exp().ln_1p() }).sqrt()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A matrix on the GPU: rows x cols in its stored form (q4_K, fp8_32x32) or as float32 (the 16- and 32-bit ones)
struct Mat {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn f32(&self) -> bool {
        self.ty == GType::F32
    }
    /// rows [r0, r0 + n) as a matrix of their own (q4_K / float32 only: an fp8_32x32 tile spans 32 rows)
    fn rows_view(&self, r0: usize, n: usize) -> Result<Mat> {
        let rb = if self.f32() { self.cols * 4 } else { self.ty.bytes(self.cols as u64).ok_or("rows of this type")? as usize };
        Ok(Mat { buf: self.buf.view(r0 * rb, n * rb)?, ty: self.ty, rows: n, cols: self.cols })
    }
}

/// A layer's VQ experts: where each expert's three payloads lie in the blob (its span: gate | up | down back to back,
/// the same layout for every expert), the matrices' places in a span, the index width, the codebook on the GPU
struct LayerVq {
    blob: Tensor,
    /// each expert's span's first byte in the blob
    starts: Vec<u64>,
    span: usize,
    gate: usize,
    up: usize,
    down: usize,
    nbit: usize,
    /// the layer's E4M3 codebook bytes (on the GPU of the layer's part)
    cb_bytes: Vec<u8>,
}

/// Reads the slot table and checks the payloads: every expert's three payloads back to back in one layout
fn layer_vq(m: &Model, l: u64) -> Result<LayerVq> {
    let g = &m.g;
    let t = m.tensor(l, Role::Experts).ok_or_else(|| Error(format!("block {l}: the expert blob is missing")))?.clone();
    let (ne, _) = g.experts(l);
    let mut head = vec![0u8; 16 + ne as usize * 24];
    m.file.read_into(&t, 0, &mut head).map_err(e)?;
    let u32_at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u64_at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    if u32_at(&head, 0) != 0x4C56_5144 || u32_at(&head, 4) != 3 || u32_at(&head, 12) as u64 != ne {
        return Err(Error(format!("block {l}: not a DQVL v3 blob of {ne} experts")));
    }
    let slot = |x: usize, w: usize| u64_at(&head, 16 + (x * 3 + w) * 8);
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let mut ph = [0u8; 32];
    m.file.read_into(&t, slot(0, 0), &mut ph).map_err(e)?;
    let nc = u16::from_le_bytes([ph[6], ph[7]]) as usize;
    let nc = if nc == 0 { 65536 } else { nc };
    let nbit = (usize::BITS - (nc - 1).leading_zeros()) as usize;
    let cb_off = u64_at(&ph, 24);
    let s0 = slot(0, 0);
    let (gate, up, down) = (0usize, (slot(0, 1) - s0) as usize, (slot(0, 2) - s0) as usize);
    let mrow = |cols: usize| cols / 8 * 12 / 8;
    let plen = |rows: usize, cols: usize| 32 + rows * 2 + rows * mrow(cols) + if nbit > 12 { rows * (cols / 64) } else { 0 } + 4;
    let span = down + plen(d, f);
    if up < plen(f, d) || down - up < plen(f, d) {
        return Err(Error(format!("block {l}: expert 0's payloads overlap")));
    }
    let mut starts = Vec::with_capacity(ne as usize);
    for x in 0..ne as usize {
        let st = slot(x, 0);
        if slot(x, 1) - st != up as u64 || slot(x, 2) - st != down as u64 || st + span as u64 > t.bytes {
            return Err(Error(format!("block {l}: expert {x}'s payloads are not laid out as expert 0's")));
        }
        starts.push(st);
    }
    // each payload's header: the matrix's shape and the layer's codebook
    for (o, rows, cols) in [(gate, f, d), (up, f, d), (down, d, f)] {
        let mut h = [0u8; 32];
        m.file.read_into(&t, s0 + o as u64, &mut h).map_err(e)?;
        if u32_at(&h, 0) != 0x3356_5144 || u16::from_le_bytes([h[4], h[5]]) != 8 || u32_at(&h, 8) as usize != rows || u32_at(&h, 12) as usize != cols
            || u64_at(&h, 24) != cb_off || (u32_at(&h, 16) & 2 != 0) != (nbit > 12) {
            return Err(Error(format!("block {l}: a payload is not DQV3 [{rows} x {cols}] on the layer's codebook")));
        }
    }
    let mut cb_bytes = vec![0u8; nc * 8];
    m.file.read_into(&t, cb_off, &mut cb_bytes).map_err(e)?;
    Ok(LayerVq { blob: t, starts, span, gate, up, down, nbit, cb_bytes })
}

/// Expert `ex` of `vq`'s layer from the file into `host` (its span)
fn read_expert(m: &Model, vq: &LayerVq, ex: u64, host: &mut [u8]) -> Result<()> {
    m.file.read_into(&vq.blob, vq.starts[ex as usize], &mut host[..vq.span]).map_err(e)
}

/// Where an expert is.
#[derive(Clone, Copy)]
enum Loc {
    V(usize),
    R(usize),
}

/// The routed experts of some of a GPU's layers (of one span size), in an exclusive two-level cache: VRAM slots and
/// pinned host slots hold different experts; a VRAM miss swaps (the least recently used VRAM expert down to a free
/// host slot, the wanted one up); what neither holds is read from the file (deepseek4's store).
struct Store {
    vram: Vec<DevBuf>,
    host: Vec<HostBuf>,
    slot_bytes: usize,
    per_chunk: usize,
    loc: HashMap<(u64, u64), Loc>,
    vowner: Vec<Option<(u64, u64)>>,
    vused: Vec<u64>,
    rfree: Vec<usize>,
    tick: u64,
    hits: u64,
    misses: u64,
    from_host: u64,
    from_file: u64,
    direct: u64,
    rwrite: HashMap<usize, i64>,
    down_any: Option<i64>,
    vfill: HashMap<usize, i64>,
    last_up: Option<i64>,
    prefetched: HashMap<(u64, u64), i64>,
    usage: HashMap<(u64, u64), u64>,
    base: usize,
    lend: usize,
    lent: bool,
    pf_issued: u64,
    pf_used: u64,
    /// pinned staging slots for experts read from the file (copied up from there on the copy lane, not through a
    /// queue drain), and the copy last reading each
    stage: HostBuf,
    stage_tk: Vec<Option<i64>>,
    stage_next: usize,
}

/// the file reads a layer takes at once (in parallel, into the staging slots)
const STAGE: usize = 16;

impl Store {
    fn victims(&self, tick: u64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.vowner.len()).filter(|&i| self.vused[i] != tick && self.usable(i)).collect();
        order.sort_by_key(|&i| if self.vowner[i].is_none() { 0 } else { self.vused[i] + 1 });
        order
    }
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
    pub layers: Range<u64>,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    /// o_a's groups as matrices of their own
    groups: BTreeMap<u64, Vec<Mat>>,
    scratch: DevBuf,
    /// Q8_1 of up to MMVQ_COLS rows
    q8: DevBuf,
    /// the routers' biases on the host
    biases: Mutex<HashMap<u64, Arc<Vec<f32>>>>,
    vq: HashMap<u64, LayerVq>,
    cbs: HashMap<u64, DevBuf>,
    /// the codebooks as float32 (the decode products' weights without a conversion each)
    cbf: HashMap<u64, DevBuf>,
    /// and as fp16 (exact), for the variants that gather 16 bytes an index
    cbh: HashMap<u64, DevBuf>,
    /// the decode products' kernel variant by index width: [gate | up, down] (measured at load: vq_pick)
    vq_pick: HashMap<usize, [i32; 2]>,
    /// the expert stores (one a span size) and the store of each layer
    stores: Vec<Mutex<Store>>,
    store_of: HashMap<u64, usize>,
    /// the towers' experts, all in VRAM (the last part): a buffer a tower
    draft: HashMap<u64, DevBuf>,
    hold: std::sync::atomic::AtomicBool,
    pub expert_slots: usize,
    pub host_slots: usize,
    pub weight_bytes: u64,
    up: Mutex<(HostBuf, usize)>,
    back: Mutex<HostBuf>,
    /// the index sources' picks [prefill_chunk][top_k] and the candidate source's blocks [prefill_chunk][cand_top],
    /// kept for the layers after them in a pass
    sel: DevBuf,
    cand: DevBuf,
    /// the compressor's position terms (zero: V4.1 has none) for the pooling kernel
    zeros: DevBuf,
    /// a prompt pass's staging slots in VRAM for the experts in host memory or the file (copied up on the copy lane while
    /// the others are multiplied; none when NS_DS41_PROMPT_RING=0): the buffer and its slots' size
    ring: Option<(DevBuf, usize)>,
    /// the ticket after the last pass's readers of the ring (its next copies wait for it)
    ring_free: Mutex<Option<i64>>,
    /// a prompt pass's pinned slots for the experts only the file holds: read on threads while the GPU multiplies the
    /// others, then copied up through the ring; the copy last reading each slot
    pstage: Option<PStage>,
}

/// A prompt pass's pinned slots for the file's experts: the buffer, a slot's bytes, the copy last reading each
type PStage = (Mutex<HostBuf>, usize, Mutex<Vec<Option<i64>>>);

/// Where a prompt pass reads an expert from (Part::locate)
enum Src {
    V(DevBuf),
    R(DevBuf),
    F,
}

/// The pass-wide state the attention layers hand down: whether the last index source picked (its top-k rows in
/// `Part::sel`) and whether the candidate source kept blocks (`Part::cand`)
#[derive(Clone, Copy, Default)]
struct Shared {
    sel: bool,
    cand: bool,
}

impl Part {
    fn mat(&self, l: u64, r: Role) -> Result<&Mat> {
        self.mats.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: matrix {r:?} not on this GPU")))
    }
    fn vec(&self, l: u64, r: Role) -> Result<&DevBuf> {
        self.vecs.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: vector {r:?} not on this GPU")))
    }

    /// `data` into dst in queue order: through the pinned ring, copied by a kernel (big ones by write_async)
    fn upload(&self, dst: &DevBuf, data: &[u8]) -> Result<()> {
        let n = data.len();
        let mut r = self.up.lock().unwrap();
        if n == 0 || !n.is_multiple_of(4) || n > r.0.len / 4 {
            drop(r);
            return dst.write_async(0, data);
        }
        if r.1 + n > r.0.len {
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

    /// Floats of device buffers read back through the pinned ring by kernels (None: they do not fit it)
    #[allow(clippy::type_complexity)]
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
    fn read_f32(&self, x: &DevBuf, n: usize) -> Result<Vec<f32>> {
        match self.read_start(&[(x, n)])? {
            Some((tk, spans)) => Ok(self.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default()),
            None => Ok(x.view(0, n * 4)?.to_f32()?),
        }
    }

    /// The arena a pass of `t` tokens needs: the small one, or the big one (its memory then out of the stores)
    fn arena_for(&self, t: usize) -> Result<()> {
        let need = t * (arena_bytes() / prefill_chunk()).max(1) + (64 << 20);
        let lent_any = self.stores.iter().any(|s| s.lock().unwrap().lend > 0);
        if !lent_any {
            return Ok(());
        }
        let hold = self.hold.load(std::sync::atomic::Ordering::Relaxed);
        let lent = self.stores.iter().any(|s| s.lock().unwrap().lent);
        if need > SMALL_ARENA && lent {
            let after = Some(self.ops.mark()?);
            for st in &self.stores {
                let mut c = st.lock().unwrap();
                let c = &mut *c;
                if c.lend == 0 {
                    continue;
                }
                let sb = c.slot_bytes;
                for s in c.base..c.base + c.lend {
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
                c.lent = false;
            }
            self.ops.gpu.sync()?;
            self.arena.set(self.big.view(0, self.big.len)?);
        } else if need <= SMALL_ARENA && !lent && !hold {
            for st in &self.stores {
                let mut c = st.lock().unwrap();
                if c.lend > 0 {
                    c.lent = true;
                }
            }
            self.arena.set(self.small.view(0, self.small.len)?);
        }
        Ok(())
    }

    /// Layer l's router bias on the host
    fn router_bias(&self, l: u64) -> Result<Arc<Vec<f32>>> {
        if let Some(b) = self.biases.lock().unwrap().get(&l) {
            return Ok(b.clone());
        }
        let b = Arc::new(self.vec(l, Role::RouterBias)?.to_f32()?);
        self.biases.lock().unwrap().insert(l, b.clone());
        Ok(b)
    }

    /// y[t rows from y.1, y.2 apart] (+)= x . W^T: float matrices by oneMKL; q4_K from its blocks for up to MMVQ_COLS
    /// rows, else expanded; fp8_32x32 from its bytes for up to MMVQ_COLS rows, else expanded to fp16
    fn matmul(&self, w: &Mat, t: usize, x: (&DevBuf, usize, usize), y: (&DevBuf, usize, usize), acc: bool) -> Result<()> {
        let o = &self.ops;
        if w.f32() {
            if t <= MMVQ_COLS {
                for r in 0..t {
                    o.gemm_at(1, w.rows, w.cols, (x.0, x.1 + r * x.2, x.2), (&w.buf, 0), (y.0, y.1 + r * y.2, y.2), acc)?;
                }
                return Ok(());
            }
            return o.gemm_at(t, w.rows, w.cols, x, (&w.buf, 0), y, acc);
        }
        let contiguous = x.2 == w.cols && y.2 == w.rows;
        if w.ty == GType::Fp8B32 {
            if !acc && t <= MMVQ_COLS && contiguous {
                return o.fp8_mmv(&w.buf, w.rows, w.cols, (x.0, x.1, w.cols), None, (y.0, y.1, w.rows), t);
            }
            let x16 = self.arena.bytes(t * w.cols * 2)?;
            if x.2 == w.cols {
                o.to_f16(&x.0.view(x.1 * 4, t * w.cols * 4)?, &x16, t * w.cols)?;
            } else {
                for r in 0..t {
                    o.to_f16(&x.0.view((x.1 + r * x.2) * 4, w.cols * 4)?, &x16.view(r * w.cols * 2, w.cols * 2)?, w.cols)?;
                }
            }
            // whole 32-row tiles a chunk
            let chunk = ((SCRATCH * 2 / w.cols) / 32 * 32).max(32).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                o.fp8_dequant(&w.buf, 0, w.rows, w.cols, r0, r, &self.scratch, true)?;
                o.gemm_f16(t, r, w.cols, (&x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
                r0 += r;
            }
            return Ok(());
        }
        if !acc && t <= MMVQ_COLS && w.ty == GType::Q4K && q4k_own() {
            return o.q4k_mv(&w.buf, w.rows, w.cols, x, None, y, t);
        }
        if !acc && t <= MMVQ_COLS && contiguous && o.mmvq_supported(w.ty.code()) {
            o.quantize_q8_1((x.0, x.1), &self.q8, w.cols, t)?;
            return o.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (y.0, y.1), w.cols, w.rows, t);
        }
        let rb = w.ty.bytes(w.cols as u64).ok_or("matmul: a row of whole blocks")? as usize;
        if t >= dense_f16_min() && x.2 == w.cols && w.cols.is_multiple_of(256) {
            let x16 = self.arena.bytes(t * w.cols * 2)?;
            o.to_f16(&x.0.view(x.1 * 4, t * w.cols * 4)?, &x16, t * w.cols)?;
            let chunk = (SCRATCH * 2 / w.cols).max(1).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                o.dequant_f16(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
                o.gemm_f16(t, r, w.cols, (&x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
                r0 += r;
            }
            return Ok(());
        }
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let mut r0 = 0;
        while r0 < w.rows {
            let r = chunk.min(w.rows - r0);
            o.dequant(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
            o.gemm_at(t, r, w.cols, x, (&self.scratch, 0), (y.0, y.1 + r0, y.2), acc)?;
            r0 += r;
        }
        Ok(())
    }

    /// y [t, rows] = x [t, cols] . W^T
    fn mm(&self, l: u64, r: Role, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let w = self.mat(l, r)?;
        let y = self.arena.f32(t * w.rows)?;
        self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows), false)?;
        Ok(y)
    }

    fn store(&self, l: u64) -> Result<&Mutex<Store>> {
        self.store_of.get(&l).map(|&i| &self.stores[i]).ok_or_else(|| Error(format!("block {l}: no expert store on this GPU")))
    }

    /// Where every expert of `need` (layer l) is read from, as a view of its slot: `promote` makes each VRAM-resident
    /// (decode), else host-slot experts are read in place. The second value marks the experts whose copy is on its way,
    /// the third the last copy's ticket.
    #[allow(clippy::type_complexity)]
    fn ensure(&self, m: &Model, l: u64, need: &[u64], promote: bool) -> Result<(Vec<DevBuf>, Vec<bool>, Option<i64>)> {
        if let Some(buf) = self.draft.get(&l) {
            let span = self.vq[&l].span.next_multiple_of(256);
            let v = need.iter().map(|x| buf.view(*x as usize * span, span)).collect::<Result<Vec<_>>>()?;
            return Ok((v, vec![false; need.len()], None));
        }
        let vq = &self.vq[&l];
        let mut c = self.store(l)?.lock().unwrap();
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
        let mut pending: Vec<u64> = Vec::new();
        let mut last: Option<i64> = None;
        let mut reading: HashMap<usize, i64> = HashMap::new();
        for &ex in need {
            match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                    if let Some(tk) = c.prefetched.remove(&(l, ex)) {
                        c.pf_used += 1;
                        pending.push(ex);
                        last = Some(last.map_or(tk, |a| a.max(tk)));
                    }
                }
                Some(Loc::R(_)) if !promote => {}
                _ => missing.push(ex),
            }
        }
        c.prefetched.retain(|k, _| k.0 != l);
        if !missing.is_empty() {
            let order = c.victims(tick);
            let staged = self.stage_reads(m, vq, c, l, &missing)?;
            for (ex, &s) in missing.iter().zip(&order) {
                c.misses += 1;
                if let Some(tk) = self.swap_in(m, vq, c, l, *ex, s, &mut reading, &staged)? {
                    last = Some(last.map_or(tk, |a| a.max(tk)));
                    pending.push(*ex);
                }
                c.vused[s] = tick;
            }
        }
        let sb = c.slot_bytes;
        let slots = need.iter().map(|ex| match c.loc[&(l, *ex)] {
            Loc::V(s) => {
                let (ch, o) = c.vat(s);
                c.vram[ch].view(o, sb)
            }
            Loc::R(r) => {
                c.direct += 1;
                let (rch, ro) = c.rat(r);
                c.host[rch].device_view(ro, sb)
            }
        }).collect::<Result<Vec<_>>>()?;
        let arriving = need.iter().map(|ex| pending.contains(ex)).collect();
        if !promote {
            if let Some(t) = c.down_any.take() {
                self.ops.await_ticket(t)?;
            }
        }
        Ok((slots, arriving, last))
    }

    /// A prompt pass's experts of `need` (layer l) where they are, nothing moved: a VRAM slot, a pinned host slot, or the
    /// file only
    fn locate(&self, l: u64, need: &[u64]) -> Result<Vec<Src>> {
        let mut c = self.store(l)?.lock().unwrap();
        let c = &mut *c;
        c.tick += 1;
        let tick = c.tick;
        let sb = c.slot_bytes;
        let mut out = Vec::with_capacity(need.len());
        for &ex in need {
            out.push(match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                    let (ch, o) = c.vat(s);
                    Src::V(c.vram[ch].view(o, sb)?)
                }
                Some(Loc::R(r)) => {
                    c.direct += 1;
                    let (rch, ro) = c.rat(r);
                    Src::R(c.host[rch].device_view(ro, sb)?)
                }
                None => {
                    c.from_file += 1;
                    Src::F
                }
            });
        }
        // a copy into a VRAM slot or down to a host slot still on its way (decode's swaps) lands before a product reads it
        for t in [c.last_up, c.down_any.take()].into_iter().flatten() {
            self.ops.await_ticket(t)?;
        }
        Ok(out)
    }

    /// Experts (k, ex) of layer l only the file holds read into the pinned prompt slots on threads (each slot's last
    /// copy up done first): k -> its slot. At most as many as there are slots; the rest are left to stage_up.
    fn read_files(&self, m: &Model, l: u64, files: &[(usize, u64)]) -> Result<HashMap<usize, usize>> {
        let Some((buf, sb, tks)) = &self.pstage else { return Ok(HashMap::new()) };
        let vq = &self.vq[&l];
        let mut buf = buf.lock().unwrap();
        let n = files.len().min(tks.lock().unwrap().len());
        {
            let mut t = tks.lock().unwrap();
            for tk in t.iter_mut().take(n) {
                if let Some(x) = tk.take() {
                    self.ops.ticket_wait(x)?;
                }
            }
        }
        let failed: Mutex<Option<String>> = Mutex::new(None);
        let t0 = Instant::now();
        {
            let mut chunks: Vec<&mut [u8]> = buf.as_mut_slice().chunks_mut(*sb).take(n).collect();
            // 8 readers, each a share of the experts
            let jobs: Vec<(u64, &mut [u8])> = files.iter().take(n).map(|f| f.1).zip(chunks.drain(..)).collect();
            let per = jobs.len().div_ceil(8).max(1);
            let mut jobs = jobs;
            std::thread::scope(|sc| {
                while !jobs.is_empty() {
                    let part: Vec<(u64, &mut [u8])> = jobs.drain(..per.min(jobs.len())).collect();
                    let failed = &failed;
                    sc.spawn(move || {
                        for (ex, b) in part {
                            if let Err(e) = read_expert(m, vq, ex, b) {
                                *failed.lock().unwrap() = Some(e.0);
                            }
                        }
                    });
                }
            });
        }
        FILE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Some(x) = failed.into_inner().unwrap() {
            return Err(Error(format!("reading experts: {x}")));
        }
        Ok(files.iter().take(n).enumerate().map(|(i, f)| (f.0, i)).collect())
    }

    /// Experts `ks` of layer l into the staging ring's slots `slots` (their sources `src`: host slots copied up, the
    /// file's read into pinned staging on threads first), each copy after `free[slot]`: their views, the last copy's ticket
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn stage_up(&self, m: &Model, l: u64, ks: &[usize], need: &[u64], src: &[Src], slots: &[usize], free: &[Option<i64>], read: &HashMap<usize, usize>)
                -> Result<(Vec<DevBuf>, Option<i64>)> {
        let (ring, rs) = self.ring.as_ref().ok_or("no staging ring")?;
        let vq = &self.vq[&l];
        let mut last = None;
        let mut views = Vec::with_capacity(ks.len());
        // the file's experts read ahead (read_files): copied up from their pinned slots
        let mut rest = Vec::new();
        for (&k, &s) in ks.iter().zip(slots) {
            if let (Src::F, Some(&i), Some((buf, sb, tks))) = (&src[k], read.get(&k), &self.pstage) {
                let v = buf.lock().unwrap().device_view(i * sb, vq.span)?;
                let t = self.ops.stream_copy_on(1, ring, s * rs, &v, 0, vq.span, &[free[s]])?;
                tks.lock().unwrap()[i] = Some(t);
                last = Some(t);
            } else {
                rest.push((k, s));
            }
        }
        let files: Vec<u64> = rest.iter().filter(|(k, _)| matches!(src[*k], Src::F)).map(|(k, _)| need[*k]).collect();
        let mut c = self.store(l)?.lock().unwrap();
        let c = &mut *c;
        let staged = if files.is_empty() { HashMap::new() } else { self.stage_reads(m, vq, c, l, &files)? };
        let sb = c.slot_bytes;
        for &(k, s) in &rest {
            let from = match &src[k] {
                Src::R(h) => h.view(0, vq.span)?,
                Src::F => {
                    let st = match staged.get(&need[k]) {
                        Some(&st) => st,
                        None => {
                            let st = self.stage_slot(c)?;
                            read_expert(m, vq, need[k], &mut c.stage.as_mut_slice()[st * sb..(st + 1) * sb])?;
                            st
                        }
                    };
                    let v = c.stage.device_view(st * sb, vq.span)?;
                    let t = self.ops.stream_copy_on(1, ring, s * rs, &v, 0, vq.span, &[free[s]])?;
                    c.stage_tk[st] = Some(t);
                    last = Some(t);
                    continue;
                }
                Src::V(_) => return Err(Error("stage_up: an expert in VRAM".into())),
            };
            let t = self.ops.stream_copy_on(1, ring, s * rs, &from, 0, vq.span, &[free[s]])?;
            last = Some(t);
        }
        for &s in slots {
            views.push(ring.view(s * rs, *rs)?);
        }
        Ok((views, last))
    }

    /// Expert (l, ex) into VRAM slot s: the slot's expert down to a free host slot, the wanted one up from its host
    /// slot or the file (read on this thread: the queue first drained of the copies that could touch the slot)
    #[allow(clippy::too_many_arguments)]
    fn swap_in(&self, m: &Model, vq: &LayerVq, c: &mut Store, l: u64, ex: u64, s: usize, reading: &mut HashMap<usize, i64>, staged: &HashMap<u64, usize>)
               -> Result<Option<i64>> {
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
                let t = self.ops.stream_copy_on(0, &c.host[rch].device_view(ro, sb)?, 0, &c.vram[vch], vo, sb, &[reading.get(&f).copied(), fill])?;
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
                let deps = [down, c.rwrite.get(&r).copied()];
                let host = c.host[rch].device_view(ro, sb)?;
                let t = self.ops.stream_copy_on(1, &c.vram[vch], vo, &host, 0, vq.span, &deps)?;
                reading.insert(r, t);
                c.rfree.push(r);
                c.vfill.insert(s, t);
                c.last_up = Some(t);
                Some(t)
            }
            _ => {
                c.from_file += 1;
                let k = match staged.get(&ex) {
                    Some(&k) => k,
                    None => {
                        let k = self.stage_slot(c)?;
                        read_expert(m, vq, ex, &mut c.stage.as_mut_slice()[k * sb..(k + 1) * sb])?;
                        k
                    }
                };
                let t = self.ops.stream_copy_on(1, &c.vram[vch], vo, &c.stage.device_view(k * sb, sb)?, 0, vq.span, &[down, fill])?;
                c.stage_tk[k] = Some(t);
                c.vfill.insert(s, t);
                c.last_up = Some(t);
                Some(t)
            }
        };
        c.vowner[s] = Some(key);
        c.loc.insert(key, Loc::V(s));
        Ok(up)
    }

    /// A free staging slot (the copy last reading it done)
    fn stage_slot(&self, c: &mut Store) -> Result<usize> {
        let k = c.stage_next % STAGE;
        c.stage_next += 1;
        if let Some(t) = c.stage_tk[k].take() {
            self.ops.ticket_wait(t)?;
        }
        Ok(k)
    }

    /// The experts of `missing` held by neither VRAM nor host memory read from the file into staging slots, up to
    /// STAGE of them at once on threads of their own: expert -> its slot
    fn stage_reads(&self, m: &Model, vq: &LayerVq, c: &mut Store, l: u64, missing: &[u64]) -> Result<HashMap<u64, usize>> {
        let file: Vec<u64> = missing.iter().copied().filter(|x| !c.loc.contains_key(&(l, *x))).take(STAGE).collect();
        let mut at = HashMap::new();
        if file.is_empty() {
            return Ok(at);
        }
        let t0 = Instant::now();
        let ks = file.iter().map(|_| self.stage_slot(c)).collect::<Result<Vec<_>>>()?;
        let sb = c.slot_bytes;
        let mut slices: Vec<Option<&mut [u8]>> = c.stage.as_mut_slice().chunks_mut(sb).map(Some).collect();
        let jobs: Vec<(u64, &mut [u8])> = file.iter().zip(&ks).map(|(x, k)| (*x, slices[*k].take().unwrap())).collect();
        let failed: Mutex<Option<String>> = Mutex::new(None);
        std::thread::scope(|sc| {
            for (x, buf) in jobs {
                let failed = &failed;
                sc.spawn(move || {
                    if let Err(e) = read_expert(m, vq, x, buf) {
                        *failed.lock().unwrap() = Some(e.0);
                    }
                });
            }
        });
        if let Some(x) = failed.into_inner().unwrap() {
            return Err(Error(format!("reading experts: {x}")));
        }
        for (x, k) in file.iter().zip(ks) {
            at.insert(*x, k);
        }
        FILE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(at)
    }

    /// Decode: the experts layer l is expected to want, swapped in now (only ones in host memory; at most `limit`)
    fn prefetch(&self, m: &Model, l: u64, want: &[u64], limit: usize) -> Result<()> {
        let Ok(st) = self.store(l) else { return Ok(()) };
        let vq = &self.vq[&l];
        let mut c = st.lock().unwrap();
        let c = &mut *c;
        let tick = c.tick;
        let cand: Vec<u64> = want.iter().copied().filter(|ex| matches!(c.loc.get(&(l, *ex)), Some(Loc::R(_)))).take(limit).collect();
        if cand.is_empty() {
            return Ok(());
        }
        let order = c.victims(tick);
        let mut reading: HashMap<usize, i64> = HashMap::new();
        let none = HashMap::new();
        for (ex, &s) in cand.iter().zip(&order) {
            if let Some(tk) = self.swap_in(m, vq, c, l, *ex, s, &mut reading, &none)? {
                c.prefetched.insert((l, *ex), tk);
                c.pf_issued += 1;
            }
            c.vused[s] = tick;
        }
        Ok(())
    }
}

/// A source layer's compressed keys: the cache (fp16 [ctx / ratio + 1][512], normed, rotated, fp4-simulated), at
/// ratio 2 the ring of its last tokens' projections, and the indexer's keys (fp16 [ctx / ratio + 1][128])
struct Comp {
    ratio: usize,
    cache: DevBuf,
    ring: Option<DevBuf>,
    idx: DevBuf,
}

/// A layer's attention state: the raw keys' ring (fp16 [raw_ring()][512]), and at a kv source its compressor's
struct Att {
    raw: DevBuf,
    comp: Option<Comp>,
}

/// the compressor's ring of a ratio-2 source: the block being built, and a verify pass rolled back
fn comp_ring(ratio: usize) -> usize {
    ratio + MAX_VERIFY
}

/// The raw keys' ring: a chunk and the window before it, and a verify pass rolled back
pub fn raw_ring() -> usize {
    prefill_chunk() + 128 + MAX_VERIFY
}

/// The towers' side of a conversation: each tower's window of main_x keys (fp16 [128][512], slot pos % 128), the last
/// pass's rows' stream means at the target layers ([3][MTP_KEEP][d]) and the ring slots the last pass overwrote
pub struct MtpS {
    rings: Vec<DevBuf>,
    mh: DevBuf,
    /// (first position, rows) of the last pass's rows in `mh`
    mh_rows: (usize, usize),
    /// the ring slots a verify pass's push overwrote: [tower][MAX_VERIFY][512] fp16, from position snap_pos
    snap: DevBuf,
    snap_at: Option<(usize, usize)>,
}

/// A conversation's state: per layer its attention state (on its layer's GPU); `pos` tokens so far, all of them in
/// `hist` (the Engram hashes the 3 before a token)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<Att>,
    hist: Vec<u32>,
    /// (first position, rows) of the last pass of 2..=MAX_VERIFY rows (a rollback's bounds)
    snapped: Option<(usize, usize)>,
    mtp: Option<MtpS>,
}

/// A conversation's state at a position, in host memory
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
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
            if len > 1 << 34 {
                return Err(bad());
            }
            let mut v = vec![0u8; len];
            r.read_exact(&mut v)?;
            bytes += len;
            bufs.push(v);
        }
        Ok(Checkpoint { pos, bufs, bytes })
    }
}

/// Generation from a fed prompt: `Ds41::step` commits the next token(s) each call. The last committed token is fed at
/// the start of the following call.
pub struct Decoder {
    logits: Vec<f32>,
    next: Option<u32>,
    /// the towers' drafts for the positions after `next`
    drafts: Vec<u32>,
    mtp: bool,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn pending(&mut self) -> Option<u32> {
        self.drafts.clear();
        self.next.take()
    }
}

pub struct Ds41<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    pub engram: Option<Engram>,
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
    /// the towers are loaded (on the last part)
    pub mtp: bool,
}

/// the decode products' kernel variants (ds41.h: ns_ds41_vq_mmv): 0 a sub-group a row (float32 codebook), 1 a lane a row
/// (the E4M3 bytes), 2 a lane a row (fp16 codebook), 3 a sub-group a row (fp16), 4 a lane a row (the bytes two at once),
/// 5 by lookup table (each group of 8 values of x against every codebook entry, in local memory)
const VQ_VARIANTS: usize = 6;

/// The widened codebook variant `var` reads: float32 (0), fp16 (2, 3), none (the bytes)
fn vq_wide<'a>(wide: (&'a HashMap<u64, DevBuf>, &'a HashMap<u64, DevBuf>), l: u64, var: i32) -> Option<&'a DevBuf> {
    match var {
        0 => wide.0.get(&l),
        2 | 3 => wide.1.get(&l),
        _ => None,
    }
}

/// The decode products' fastest kernel variant on this GPU for each index width of its layers, [gate | up, down]: both
/// timed on six of a layer's experts in VRAM, one token (NS_DS41_VQ_V=0..4 sets it instead). Which wins depends on the
/// GPU and the shape (B70 + B65: 3 or 4 for gate | up, 2 or 4 for down).
#[allow(clippy::too_many_arguments)]
fn vq_pick(o: &Ops, g: &crate::model::Geometry, vq: &HashMap<u64, LayerVq>, cbs: &HashMap<u64, DevBuf>, wide: (&HashMap<u64, DevBuf>, &HashMap<u64, DevBuf>),
           stores: &[Mutex<Store>],
           store_of: &HashMap<u64, usize>, draft: &HashMap<u64, DevBuf>, log: &mut dyn FnMut(String)) -> Result<HashMap<usize, [i32; 2]>> {
    let mut out = HashMap::new();
    if let Some(v) = std::env::var("NS_DS41_VQ_V").ok().and_then(|v| v.parse::<i32>().ok()) {
        for l in vq.values() {
            out.insert(l.nbit, [v, v]);
        }
        return Ok(out);
    }
    let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
    let mut layers: Vec<u64> = vq.keys().copied().collect();
    layers.sort();
    for l in layers {
        let v = &vq[&l];
        if out.contains_key(&v.nbit) {
            continue;
        }
        // six of the layer's experts in VRAM: the store's slots it owns, or the towers' buffer
        let bases: Vec<u64> = match (draft.get(&l), store_of.get(&l)) {
            (Some(buf), _) => (0..6).map(|x| buf.ptr() as u64 + (x * v.span.next_multiple_of(256)) as u64).collect(),
            (None, Some(&si)) => {
                let c = stores[si].lock().unwrap();
                (0..c.base).filter(|s| c.vowner[*s].is_some_and(|k| k.0 == l)).take(6).map(|s| {
                    let (ch, at) = c.vat(s);
                    c.vram[ch].ptr() as u64 + at as u64
                }).collect()
            }
            _ => Vec::new(),
        };
        if bases.len() < 6 {
            continue;
        }
        let n = bases.len();
        let mut tb: Vec<u8> = Vec::new();
        let mut meta: Vec<i32> = Vec::new();
        for w in 0..2 {
            for (x, b) in bases.iter().enumerate() {
                tb.extend((b + if w == 0 { v.gate } else { v.up } as u64).to_le_bytes());
                meta.extend([1, (w * n + x) as i32, 0, 0, 0, 0, 0, 0, 0, 0]);
            }
        }
        tb.extend(i32s(&meta));
        let (gl, gmax) = crate::ops::lut_lists(1, (0..2 * n).map(|e| (0, e, 0))).ok_or("vq_pick: the lists")?;
        tb.extend(gl);
        let mut db: Vec<u8> = bases.iter().flat_map(|b| (b + v.down as u64).to_le_bytes()).collect();
        db.extend(i32s(&(0..n).flat_map(|x| [1, x as i32, x as i32, 0, 0, 0, 0, 0, 0, 0]).collect::<Vec<_>>()));
        let (dl, dmax) = crate::ops::lut_lists(n, (0..n).map(|e| (e, e, 0))).ok_or("vq_pick: the lists")?;
        db.extend(dl);
        let (gt, dt) = (DevBuf::new(&o.gpu, tb.len())?, DevBuf::new(&o.gpu, db.len())?);
        gt.write(0, &tb)?;
        dt.write(0, &db)?;
        let x = DevBuf::f32(&o.gpu, n * f.max(d))?;
        x.fill(0)?;
        let y = DevBuf::f32(&o.gpu, 2 * n * f.max(d))?;
        let mut pick = [0i32; 2];
        for (k, (tab, e, rows, cols, lut)) in [(&gt, 2 * n, f, d, (1, gmax)), (&dt, n, d, f, (n, dmax))].into_iter().enumerate() {
            let run = |var: i32, reps: usize| -> Result<f64> {
                let t0 = Instant::now();
                for _ in 0..reps {
                    o.vq_mmv(tab, e, (&cbs[&l], vq_wide(wide, l, var)), &x, (&y, 0), e, rows, cols, v.nbit, var, Some(lut))?;
                }
                o.gpu.sync()?;
                Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
            };
            // the clocks up first, then each twice in turn
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < 0.15 {
                run(0, 10)?;
            }
            let mut best = [f64::MAX; VQ_VARIANTS];
            for _ in 0..2 {
                for (var, b) in best.iter_mut().enumerate() {
                    *b = b.min(run(var as i32, 20)?);
                }
            }
            pick[k] = (0..VQ_VARIANTS).min_by(|a, b| best[*a].total_cmp(&best[*b])).unwrap_or(0) as i32;
            log(format!("{}: {}-bit VQ {} products: {} us (variants 0-{}) - taking {}", o.gpu.name, v.nbit, ["gate | up", "down"][k],
                        best.iter().map(|b| format!("{b:.0}")).collect::<Vec<_>>().join(" / "), VQ_VARIANTS - 1, pick[k]));
        }
        out.insert(v.nbit, pick);
    }
    Ok(out)
}

/// the small tensors kept as float32 vectors
const VECTORS: [Role; 15] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale,
                             Role::QANorm, Role::KvNorm, Role::Sinks, Role::CompNorm, Role::IdxKNorm, Role::RouterBias, Role::MtpMainNorm, Role::MtpOutNorm];

impl Part {
    /// The part's layers onto its GPU (the head and the towers when `last`), its expert stores and pinned mirror.
    #[allow(clippy::too_many_arguments)]
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, last: bool, mtp: bool, expert_bytes: Option<usize>, mirror_bytes: usize, kv_reserve: usize,
            log: &mut dyn FnMut(String)) -> Result<Part> {
        let g = &m.g;
        let ops = Ops::new(gpu.clone())?;
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let towers: Vec<u64> = if last && mtp { (g.n_layer..g.n_layer + g.n_mtp).collect() } else { Vec::new() };
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            let mut v = vec![Role::OutputNorm, Role::Output];
            if mtp {
                v.extend([Role::MtpMainProj, Role::MtpMainNorm, Role::MtpMarkovEmbd, Role::MtpMarkovHead, Role::MtpConfidence, Role::MtpOutNorm]);
            }
            todo.push((0, v));
        }
        todo.extend(layers.clone().chain(towers.iter().copied()).map(|l| (l, m.roles(l))));
        for (l, roles) in todo {
            for r in roles {
                if matches!(r, Role::Experts | Role::TokenEmbd | Role::EngramQ | Role::EngramK) {
                    continue; // the experts through the stores; the embedding's rows per token; q x k below
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let n = t.elements() as usize;
                let raw = DevBuf::new(gpu, t.bytes as usize)?;
                raw.write(0, &m.file.read(t).map_err(e)?)?;
                bytes += t.bytes;
                let cols = *t.shape.last().unwrap_or(&1) as usize;
                if VECTORS.contains(&r) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    vecs.insert((l, r), f);
                } else if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    mats.insert((l, r), Mat { buf: f, ty: GType::F32, rows: n / cols, cols });
                } else if matches!(t.ty, GType::Q4K | GType::Fp8B32) {
                    mats.insert((l, r), Mat { buf: raw, ty: t.ty, rows: n / cols, cols });
                } else {
                    return Err(Error(format!("{}: type {} is not one this engine multiplies", t.name, t.ty.name())));
                }
            }
            // the Engram's q x k (only ever used as their product)
            if g.is_engram(l) && layers.contains(&l) {
                let rd = |r: Role| -> Result<Vec<f32>> {
                    let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                    Ok(m.file.read(t).map_err(e)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                };
                let (q, k) = (rd(Role::EngramQ)?, rd(Role::EngramK)?);
                vecs.insert((l, Role::EngramQ), DevBuf::from_f32(gpu, &q.iter().zip(&k).map(|(a, b)| a * b).collect::<Vec<f32>>())?);
            }
        }
        gpu.sync()?;
        let mut groups = BTreeMap::new();
        for l in layers.clone().chain(towers.iter().copied()) {
            let w = mats.get(&(l, Role::OutA)).ok_or_else(|| Error(format!("block {l}: o_a missing")))?;
            if w.ty != GType::Fp8B32 {
                let gs = (0..g.o_groups as usize).map(|gi| w.rows_view(gi * g.o_lora as usize, g.o_lora as usize)).collect::<Result<Vec<_>>>()?;
                groups.insert(l, gs);
            }
        }
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(4 * g.n_embd as usize, MMVQ_COLS))?;
        // the experts: each layer's layout and codebook
        let mut vq = HashMap::new();
        let mut cbs = HashMap::new();
        let mut cbf = HashMap::new();
        let mut cbh = HashMap::new();
        for l in layers.clone().chain(towers.iter().copied()) {
            let v = layer_vq(m, l)?;
            if !ops.vq_supported(g.ffn_expert as usize, g.n_embd as usize, v.nbit) {
                return Err(Error(format!("block {l}: {}-bit VQ experts are not taken by the kernels", v.nbit)));
            }
            let cb = DevBuf::new(gpu, v.cb_bytes.len())?;
            cb.write(0, &v.cb_bytes)?;
            let f = DevBuf::f32(gpu, v.cb_bytes.len())?;
            ops.vq_codebook_f32(&cb, v.cb_bytes.len(), &f)?;
            cbf.insert(l, f);
            let h = DevBuf::new(gpu, v.cb_bytes.len() * 2)?;
            ops.vq_codebook_f16(&cb, v.cb_bytes.len(), &h)?;
            cbh.insert(l, h);
            cbs.insert(l, cb);
            vq.insert(l, v);
        }
        // the towers' experts: all of them, in VRAM
        let mut draft = HashMap::new();
        let mut draft_bytes = 0;
        for &l in &towers {
            let v = &vq[&l];
            let span = v.span.next_multiple_of(256);
            let buf = DevBuf::new(gpu, span * v.starts.len())?;
            let mut host = vec![0u8; v.span];
            for x in 0..v.starts.len() as u64 {
                read_expert(m, v, x, &mut host)?;
                buf.write(x as usize * span, &host)?;
            }
            draft_bytes += buf.len;
            draft.insert(l, buf);
        }
        if !towers.is_empty() {
            log(format!("{}: the {} draft towers' experts in VRAM ({:.2} GiB)", gpu.name, towers.len(), gib(draft_bytes)));
        }
        // the stores: one a span size, VRAM and the mirror shared by their experts' bytes
        let mut classes: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        for l in layers.clone() {
            classes.entry(vq[&l].span.next_multiple_of(256)).or_default().push(l);
        }
        let total_bytes: usize = classes.iter().map(|(sb, ls)| sb * ls.len() * g.n_expert as usize).sum::<usize>().max(1);
        let ring_slot = classes.keys().copied().max().unwrap_or(0);
        let ring_bytes = prompt_ring() * ring_slot;
        let budget = match expert_bytes {
            Some(b) => b,
            None => gpu.memory()?.1.map_or(8usize << 30, |f| {
                let pass = (900 << 20) + prefill_chunk() * (400 << 10);
                (f as usize).saturating_sub(pass + arena_bytes() + SMALL_ARENA + kv_reserve + keep_free() + ring_bytes + (512 << 20))
            }),
        };
        let mut stores = Vec::new();
        let mut store_of = HashMap::new();
        let (mut nv_all, mut nr_all) = (0, 0);
        let t0 = Instant::now();
        for (slot_bytes, ls) in &classes {
            let slot_bytes = *slot_bytes;
            let n_exp = ls.len() * g.n_expert as usize;
            let share = |b: usize| (b as u128 * (slot_bytes * n_exp) as u128 / total_bytes as u128) as usize;
            let per_chunk = (CHUNK / slot_bytes).max(1);
            let nv = (share(budget) / slot_bytes).min(n_exp);
            let nr = if n_exp > nv { (share(mirror_bytes) / slot_bytes).min(n_exp - nv + 1) } else { 0 };
            let mut vram = Vec::new();
            let mut have = 0;
            while have < nv {
                let k = per_chunk.min(nv - have);
                vram.push(DevBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
            let mut host = Vec::new();
            let mut have = 0;
            while have < nr {
                let k = per_chunk.min(nr - have);
                host.push(HostBuf::new(gpu, k * slot_bytes)?);
                have += k;
            }
            // without a profile, every layer the same share of VRAM, host memory and the file (expert-major order): in
            // layer order the last layers were the file's alone - a prompt pass read all of their experts from the file
            let mut all: Vec<(u64, u64)> = (0..g.n_expert).flat_map(|x| ls.iter().map(move |l| (*l, x))).collect();
            if let Some(prof) = expert_profile() {
                all.sort_by_key(|k| std::cmp::Reverse(prof.get(k).copied().unwrap_or(0.0) as u64));
            }
            let in_v: Vec<(u64, u64)> = all.iter().take(nv).copied().collect();
            let in_r: Vec<(u64, u64)> = all.iter().skip(nv).take(nr.saturating_sub(1)).copied().collect();
            let threads = 8;
            let failed: Mutex<Option<String>> = Mutex::new(None);
            {
                let views: Vec<Mutex<&mut [u8]>> = host.iter_mut().map(|c| Mutex::new(c.as_mut_slice())).collect();
                std::thread::scope(|sc| {
                    for th in 0..threads {
                        let (in_v, in_r, views, vq, failed, vram) = (&in_v, &in_r, &views, &vq, &failed, &vram);
                        sc.spawn(move || {
                            let mut buf = vec![0u8; slot_bytes];
                            let mut job = |i: usize, key: (u64, u64), to_v: bool| -> Result<()> {
                                let v = &vq[&key.0];
                                read_expert(m, v, key.1, &mut buf)?;
                                let (ch, o) = (i / per_chunk, (i % per_chunk) * slot_bytes);
                                if to_v {
                                    vram[ch].write(o, &buf[..v.span])
                                } else {
                                    views[ch].lock().unwrap()[o..o + v.span].copy_from_slice(&buf[..v.span]);
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
            let mut vowner = vec![None; nv];
            for (i, k) in in_v.iter().enumerate() {
                loc.insert(*k, Loc::V(i));
                vowner[i] = Some(*k);
            }
            for (i, k) in in_r.iter().enumerate() {
                loc.insert(*k, Loc::R(i));
            }
            log(format!("{} (layers {:?}, {:.2} MiB experts): {} in VRAM ({:.1} GiB), {} in pinned host memory ({:.1} GiB), {} on the file only",
                        gpu.name, ls, slot_bytes as f64 / 1048576.0, in_v.len(), gib(nv * slot_bytes), in_r.len(), gib(nr * slot_bytes), n_exp - in_v.len() - in_r.len()));
            nv_all += nv;
            nr_all += nr;
            for l in ls {
                store_of.insert(*l, stores.len());
            }
            stores.push(Mutex::new(Store { vram, host, slot_bytes, per_chunk, loc, vowner, vused: vec![0; nv], rfree: (in_r.len()..nr).collect(), tick: 0, hits: 0,
                                           misses: 0, from_host: 0, from_file: 0, direct: 0, rwrite: HashMap::new(), down_any: None, vfill: HashMap::new(),
                                           last_up: None, prefetched: HashMap::new(), usage: HashMap::new(), base: nv, lend: 0, lent: false, pf_issued: 0,
                                           pf_used: 0, stage: HostBuf::new(gpu, STAGE * slot_bytes)?, stage_tk: vec![None; STAGE], stage_next: 0 }));
        }
        log(format!("{}: {:.2} GiB of weights, the experts loaded in {:.1} s", gpu.name, gib(bytes as usize), t0.elapsed().as_secs_f64()));
        let big = DevBuf::new(gpu, arena_bytes())?;
        let small = DevBuf::new(gpu, SMALL_ARENA)?;
        // the big arena lent to the largest store while no prompt chunk needs it
        let mut lent = false;
        if lend_arena() {
            if let Some(st) = stores.iter().max_by_key(|s| { let c = s.lock().unwrap(); c.slot_bytes * c.vowner.len() }) {
                let mut c = st.lock().unwrap();
                let n = big.len / c.slot_bytes;
                if n > 0 {
                    let v = big.view(0, n * c.slot_bytes)?;
                    c.vram.push(v);
                    let nv = c.vowner.len();
                    c.vowner.resize(nv + n, None);
                    c.vused.resize(nv + n, 0);
                    c.base = nv;
                    c.lend = n;
                    c.lent = true;
                    lent = true;
                    nv_all += n;
                }
            }
        }
        let arena = Arena::on(if lent { small.view(0, small.len)? } else { big.view(0, big.len)? });
        let k = g.idx_top_k as usize;
        let sel = DevBuf::new(gpu, prefill_chunk().max(MAX_VERIFY) * k * 4)?;
        let cand = DevBuf::new(gpu, if g.cand_top > 0 { prefill_chunk().max(MAX_VERIFY) * g.cand_top as usize * 4 } else { 4 })?;
        let zeros = DevBuf::f32(gpu, 4 * g.head_dim as usize)?;
        zeros.fill(0)?;
        let vq_pick = vq_pick(&ops, g, &vq, &cbs, (&cbf, &cbh), &stores, &store_of, &draft, log)?;
        let ring = if ring_bytes > 0 { Some((DevBuf::new(gpu, ring_bytes)?, ring_slot)) } else { None };
        let nfs = prompt_file_slots();
        let pstage = if ring_bytes > 0 && nfs > 0 { Some((Mutex::new(HostBuf::new(gpu, nfs * ring_slot)?), ring_slot, Mutex::new(vec![None; nfs]))) } else { None };
        Ok(Part { ops, arena, big, small, layers, mats, vecs, groups, scratch, q8, biases: Mutex::new(HashMap::new()), vq, cbs, cbf, cbh, vq_pick, stores, store_of, draft,
                  hold: std::sync::atomic::AtomicBool::new(false), expert_slots: nv_all, host_slots: nr_all, weight_bytes: bytes,
                  up: Mutex::new((HostBuf::new(gpu, UP_BYTES)?, 0)), back: Mutex::new(HostBuf::new(gpu, BACK_BYTES)?), sel, cand, zeros, ring, ring_free: Mutex::new(None), pstage })
    }
}

impl<'g> Ds41<'g> {
    /// Loads the model over `gpus`: the layers split at a kv source (by default the one nearest half the layers; NS_SPLIT
    /// moves it, to another source), the head and the towers on the last. Per GPU, `expert_bytes` of VRAM for routed
    /// experts (None: what is free less the passes' needs and NS_DS41_FREE_GIB); `mirror_bytes` of pinned host memory in
    /// all (None: NS_DS41_PIN_GIB, at most what is available less 12 GiB). `kv`: (context, sessions) held back.
    pub fn load(file: &'g nextsycl_gguf::Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, kv: (usize, usize), draft: bool,
                log: &mut dyn FnMut(String)) -> Result<Ds41<'g>> {
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
        let k = gpus.len();
        let mirror = mirror_bytes.unwrap_or_else(|| pin_cap().min(mem_available().saturating_sub(12 << 30)));
        // the boundaries: kv sources (a source's readers stay with it); even by count unless NS_SPLIT says
        let mut bounds: Vec<u64> = vec![0];
        let cuts: Vec<u64> = match std::env::var("NS_SPLIT") {
            Ok(v) => v.split(',').filter_map(|x| x.trim().parse::<u64>().ok()).scan(0, |a, c| { *a += c; Some(*a) }).collect(),
            Err(_) => (1..k).map(|i| {
                let want = n as f64 * i as f64 / k as f64;
                *g.kv_sources.iter().chain(std::iter::once(&0)).min_by(|a, b| (**a as f64 - want).abs().total_cmp(&(**b as f64 - want).abs())).unwrap_or(&0)
            }).collect(),
        };
        if cuts.len() + 1 != k {
            return Err(Error(format!("NS_SPLIT: {} layer counts for {k} GPUs", cuts.len())));
        }
        for c in cuts {
            if c <= *bounds.last().unwrap() || c >= n || (g.kv_source(c).is_some() && !g.kv_sources.contains(&c)) {
                return Err(Error(format!("a split at layer {c}: a part must begin at a kv source ({:?}) or before the compressed layers", g.kv_sources)));
            }
            bounds.push(c);
        }
        bounds.push(n);
        let mut parts = Vec::new();
        let mut owner = vec![0usize; n as usize];
        // the towers only when asked for (NS_DS41_MTP=1): on the B70 + B65 their drafts cost more than they save on
        // prose (a verify pass's rows cost nearly what one-token passes do: the decode products are ALU-bound), and
        // without them the last GPU holds ~2.9 GiB more experts
        let mtp = draft && g.n_mtp > 0 && mtp_on();
        for (i, gpu) in gpus.iter().enumerate() {
            let range = bounds[i]..bounds[i + 1];
            for l in range.clone() {
                owner[l as usize] = i;
            }
            let last = i + 1 == k;
            let share = mirror * (range.end - range.start) as usize / n as usize;
            // per session: the raw rings, the sources' compressed and index keys (fp16), the towers' rings
            let reserve: usize = range.clone().map(|l| {
                let mut b = raw_ring() * 512 * 2;
                if g.kv_sources.contains(&l) {
                    let r = g.ratio(l) as usize;
                    b += (kv.0 / r + 1) * (512 + g.idx_dim as usize) * 2;
                }
                b
            }).sum::<usize>() * kv.1 + if last && mtp { kv.1 * 3 * MTP_WIN * 1024 } else { 0 };
            if reserve > 0 {
                log(format!("{}: {:.2} GiB held for the sessions' {} tokens", gpu.name, gib(reserve), kv.0 * kv.1));
            }
            parts.push(Part::load(&m, gpu, range, last, mtp, expert_bytes, share, reserve, log)?);
        }
        let engram = if g.engram_layers.is_empty() {
            None
        } else {
            let dir = engram::engram_dir(&file.paths[0]);
            let en = Engram::open(file, &dir)?;
            log(format!("the Engram: layers {:?}, tables in {}", en.layers, dir.display()));
            Some(en)
        };
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(BTreeMap::new()));
        Ok(Ds41 { m, parts, owner, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, engram, prof, mtp })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.parts.iter().map(|p| {
            let (total, free) = p.ops.gpu.memory().unwrap_or((0, None));
            GpuInfo { index: p.ops.gpu.index, name: p.ops.gpu.name.clone(), pci: p.ops.gpu.pci.clone(), total, free, layers: (p.layers.start, p.layers.end),
                      expert_slots: p.expert_slots, host_slots: p.host_slots }
        }).collect()
    }

    /// (VRAM hits, swaps, of them from host memory, host-memory reads by prompt passes, prefetched, prefetches used)
    pub fn expert_stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        let mut a = (0, 0, 0, 0, 0, 0);
        for p in &self.parts {
            for s in &p.stores {
                let c = s.lock().unwrap();
                a = (a.0 + c.hits, a.1 + c.misses, a.2 + c.from_host, a.3 + c.direct, a.4 + c.pf_issued, a.5 + c.pf_used);
            }
        }
        a
    }

    /// experts read from the file on a miss
    pub fn file_reads(&self) -> u64 {
        self.parts.iter().flat_map(|p| p.stores.iter()).map(|s| s.lock().unwrap().from_file).sum()
    }

    pub fn arena_peaks(&self) -> Vec<(usize, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        self.parts.iter().map(|p| (p.arena.peak.load(Relaxed), p.arena.spills.load(Relaxed))).collect()
    }

    fn timed<T>(&self, p: &Part, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(pr) = &self.prof else { return f() };
        p.ops.gpu.sync()?;
        let t0 = Instant::now();
        let r = f()?;
        p.ops.gpu.sync()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Ok(r)
    }

    /// With NS_PROFILE=1: the time since `t0` (the GPU drained) added to `name`
    fn lap(&self, p: &Part, name: &'static str, t0: Instant) -> Result<()> {
        let Some(pr) = &self.prof else { return Ok(()) };
        p.ops.gpu.sync()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Ok(())
    }

    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    pub fn hold_arena(&self, on: bool) {
        for p in &self.parts {
            p.hold.store(on, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn part_of(&self, l: u64) -> &Part {
        &self.parts[self.owner.get(l as usize).copied().unwrap_or(self.parts.len() - 1)]
    }

    /// A new conversation, up to `max_ctx` tokens.
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            let gpu = &self.part_of(l).ops.gpu;
            let raw = DevBuf::new(gpu, raw_ring() * 512 * 2)?;
            let comp = if g.kv_sources.contains(&l) {
                let r = g.ratio(l) as usize;
                let ring = if r > 1 {
                    let b = DevBuf::f32(gpu, comp_ring(r) * 2 * 512)?;
                    b.fill(0)?;
                    Some(b)
                } else {
                    None
                };
                Some(Comp { ratio: r, cache: DevBuf::new(gpu, (max_ctx / r + 1) * 512 * 2)?, ring, idx: DevBuf::new(gpu, (max_ctx / r + 1) * g.idx_dim as usize * 2)? })
            } else {
                None
            };
            layers.push(Att { raw, comp });
        }
        let mtp = if self.mtp {
            let gpu = &self.parts.last().unwrap().ops.gpu;
            let rings = (0..g.n_mtp).map(|_| DevBuf::new(gpu, MTP_WIN * 512 * 2)).collect::<Result<Vec<_>>>()?;
            Some(MtpS { rings, mh: DevBuf::f32(gpu, 3 * MTP_KEEP * g.n_embd as usize)?, mh_rows: (0, 0), snap: DevBuf::new(gpu, g.n_mtp as usize * MAX_VERIFY * 1024)?,
                        snap_at: None })
        } else {
            None
        };
        Ok(Session { pos: 0, max_ctx, layers, hist: Vec::new(), snapped: None, mtp })
    }

    /// The raw ring's slots of the window before `pos` (at most 136 positions, in up to two runs)
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

    /// The conversation's state at `sess.pos` in host memory: per layer the window's raw keys, at a source its
    /// compressed keys so far, its ring and its index keys; the towers' rings; the tokens
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
                take(&c.idx, 0, n / c.ratio * idx_d * 2)?;
                if let Some(r) = &c.ring {
                    take(r, 0, r.len)?;
                }
            }
        }
        if let Some(ms) = &sess.mtp {
            for r in &ms.rings {
                take(r, 0, r.len)?;
            }
        }
        bufs.push(i32s(&sess.hist[..n].iter().map(|t| *t as i32).collect::<Vec<_>>()));
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: n, bufs, bytes })
    }

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
                put(&c.idx, 0)?;
                if let Some(r) = &c.ring {
                    put(r, 0)?;
                }
            }
        }
        if let Some(ms) = &sess.mtp {
            for r in &ms.rings {
                put(r, 0)?;
            }
        }
        let toks = it.next().ok_or("restore: the checkpoint has no tokens")?;
        sess.hist = toks.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
        sess.pos = ck.pos;
        sess.snapped = None;
        if let Some(ms) = sess.mtp.as_mut() {
            ms.mh_rows = (0, 0);
            ms.snap_at = None;
        }
        Ok(())
    }

    pub fn reset_session(&self, sess: &mut Session) -> Result<()> {
        sess.pos = 0;
        sess.hist.clear();
        sess.snapped = None;
        if let Some(ms) = sess.mtp.as_mut() {
            ms.mh_rows = (0, 0);
            ms.snap_at = None;
        }
        Ok(())
    }

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
                let nb = src.pos / sc.ratio;
                if nb > 0 {
                    dc.cache.copy_within(0, &sc.cache, 0, nb * 512 * 2)?;
                    dc.idx.copy_within(0, &sc.idx, 0, nb * idx_d * 2)?;
                }
                if let (Some(dr), Some(sr)) = (&dc.ring, &sc.ring) {
                    dr.copy_within(0, sr, 0, sr.len)?;
                }
            }
        }
        if let (Some(dm), Some(sm)) = (dst.mtp.as_mut(), src.mtp.as_ref()) {
            for (d, s) in dm.rings.iter().zip(&sm.rings) {
                d.copy_within(0, s, 0, s.len)?;
            }
            dm.mh_rows = (0, 0);
            dm.snap_at = None;
        }
        dst.hist = src.hist.clone();
        dst.pos = src.pos;
        dst.snapped = None;
        Ok(())
    }

    /// Rows of a [n_vocab, cols] matrix read from the file (the embedding: q4_K; the Markov embedding: BF16), as
    /// float32 on p's GPU
    fn rows_of(&self, p: &Part, r: Role, tokens: &[u32]) -> Result<DevBuf> {
        let t = self.m.tensor(0, r).ok_or_else(|| Error(format!("{r:?} missing")))?;
        let cols = *t.shape.last().unwrap_or(&1);
        let rb = t.ty.bytes(cols).unwrap_or(0) as usize;
        let mut rows = vec![0u8; tokens.len() * rb];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as u64 >= self.m.g.n_vocab {
                return Err(Error(format!("token {tok} outside the vocabulary")));
            }
            self.m.file.read_into(t, *tok as u64 * rb as u64, &mut rows[i * rb..(i + 1) * rb]).map_err(e)?;
        }
        let raw = DevBuf::new(&p.ops.gpu, rows.len())?;
        p.upload(&raw, &rows)?;
        let x = DevBuf::f32(&p.ops.gpu, tokens.len() * cols as usize)?;
        p.ops.dequant(t.ty.code(), &raw, 0, rows.len(), tokens.len() * cols as usize, &x)?;
        Ok(x)
    }

    /// The embedding rows on part `pi` as the 4 streams [t, 4, d], then the first mix [t, 4] (stream 0)
    fn streams(&self, pi: usize, tokens: &[u32], tap: Tap) -> Result<DevBuf> {
        let d = self.m.g.n_embd as usize;
        let t = tokens.len();
        let p = &self.parts[pi];
        let x0 = self.rows_of(p, Role::TokenEmbd, tokens)?;
        tap("inp_embd", &x0)?;
        let mut pre = vec![0f32; t * 4];
        for ti in 0..t {
            pre[ti * 4] = 1.0;
        }
        if t > MMVQ_COLS {
            let e0 = x0.to_f32()?;
            let mut xs = Vec::with_capacity(t * 4 * d + t * 4);
            for ti in 0..t {
                for _ in 0..4 {
                    xs.extend_from_slice(&e0[ti * d..(ti + 1) * d]);
                }
            }
            xs.extend_from_slice(&pre);
            return DevBuf::from_f32(&p.ops.gpu, &xs);
        }
        // each row 4 times on the GPU (kernel copies: no trip through the host behind the expert copies)
        let xi = DevBuf::f32(&p.ops.gpu, t * 4 * d + t * 4)?;
        for ti in 0..t {
            for s in 0..4 {
                p.ops.kcopy(&xi, (ti * 4 + s) * d * 4, &x0, ti * d * 4, d * 4)?;
            }
        }
        p.upload(&xi.view(t * 4 * d * 4, t * 16)?, &f32s(&pre))?;
        Ok(xi)
    }

    /// The carried state [t, 4, d] streams + [t, 4] mix onto part pi (through host memory)
    fn to_part(&self, pi: usize, x: &DevBuf, staging: &mut Vec<u8>) -> Result<DevBuf> {
        let p = &self.parts[pi];
        let next = DevBuf::new(&p.ops.gpu, x.len)?;
        if pi > 0 && x.len.is_multiple_of(4) {
            let src = &self.parts[pi - 1];
            if let Some((tk, spans)) = src.read_start(&[(x, x.len / 4)])? {
                let v = src.read_finish(tk, &spans)?.into_iter().next().unwrap_or_default();
                p.upload(&next, &f32s(&v))?;
                return Ok(next);
            }
        }
        staging.resize(staging.len().max(x.len), 0);
        next.copy_from_peer(x, staging)?;
        Ok(next)
    }

    /// Engram layer l on the streams x [t, 4, d] (rows: the layer's raw table rows [t * 24][264])
    fn engram_apply(&self, p: &Part, l: u64, x: &DevBuf, raw: &[u8], t: usize, tap: Tap) -> Result<()> {
        let g = &self.m.g;
        let en = self.engram.as_ref().ok_or("the Engram is not loaded")?;
        let (d, hd, nc) = (g.n_embd as usize, en.hd, en.cols());
        let rb = p.arena.bytes(raw.len().next_multiple_of(4))?;
        let mut padded = raw.to_vec();
        padded.resize(raw.len().next_multiple_of(4), 0);
        p.upload(&rb, &padded)?;
        let rows = p.arena.f32(t * nc * hd)?;
        p.ops.engram_rows(&rb, &rows, t * nc, hd)?;
        let kv = p.mm(l, Role::EngramWkv, &rows, t)?;
        p.ops.engram_gate(x, &kv, p.vec(l, Role::EngramQ)?, t, d, g.rms_eps as f32)?;
        tap(&format!("hc_engram-{l}"), &x.view(0, t * 4 * d * 4)?)?;
        Ok(())
    }

    /// The compressor of kv source l over this pass's x [t, d] (normed): the groups it completes - pooled at ratio 2,
    /// projected at 1 - normed; their index keys (wk, k_norm, rotated, fp4) and compressed keys (rotated, fp4 E4M3) into
    /// the caches
    #[allow(clippy::too_many_arguments)]
    fn compress(&self, p: &Part, l: u64, x: &DevBuf, c: &Comp, pos0: usize, t: usize, tap: Tap) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (hd, id, rd, eps) = (g.head_dim as usize, g.idx_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let r = c.ratio;
        let (j0, j1) = (pos0 / r, (pos0 + t) / r);
        let nb = j1 - j0;
        let kv = p.mm(l, Role::CompKv, x, t)?;
        let lat = if r == 1 {
            kv
        } else {
            let sc = p.mm(l, Role::CompGate, x, t)?;
            let pooled = p.arena.f32(nb.max(1) * hd)?;
            o.pool(c.ring.as_ref().ok_or("a ratio-2 source without its ring")?, &kv, &sc, &p.zeros, &pooled, pos0, t, hd, r, false, comp_ring(r))?;
            pooled
        };
        if nb == 0 {
            return Ok(());
        }
        o.rms_norm(&lat, Some(p.vec(l, Role::CompNorm)?), &lat, nb, hd, eps)?;
        let rope = self.m.rope(l);
        let ik = p.mm(l, Role::IdxWk, &lat, nb)?;
        o.rms_norm(&ik, Some(p.vec(l, Role::IdxKNorm)?), &ik, nb, id, eps)?;
        o.rope(&ik, nb, 1, id, id, id - rd, rd, j0 * r, r, rope, false)?;
        o.quant_sim(&ik, nb, id, 32, 1)?;
        o.rows_f16(&ik, &c.idx, nb, j0, id)?;
        o.rope(&lat, nb, 1, hd, hd, hd - rd, rd, j0 * r, r, rope, false)?;
        o.quant_sim(&lat, nb, hd, 16, 2)?;
        tap(&format!("comp-{l}"), &lat.view(0, nb * hd * 4)?)?;
        o.rows_f16(&lat, &c.cache, nb, j0, hd)
    }

    /// Index source l's picks for the rows of a pass at pos0 (x their attention input, qr their q LoRA) into p.sel:
    /// false when every row reads all it sees (no more than top_k). The candidate source keeps each row's best blocks
    /// (p.cand) for the index sources after it, which pick inside them.
    #[allow(clippy::too_many_arguments)]
    fn select(&self, p: &Part, l: u64, x: &DevBuf, qr: &DevBuf, c: &Comp, pos0: usize, t: usize, sh: &mut Shared, tap: Tap) -> Result<bool> {
        let g = &self.m.g;
        let o = &p.ops;
        let (id, ih, kp, rd) = (g.idx_dim as usize, g.idx_heads as usize, g.idx_top_k as usize, g.rope_dim as usize);
        let r = c.ratio;
        let n = (pos0 + t) / r;
        let is_cand = g.cand_source == Some(l);
        if is_cand {
            sh.cand = false;
        }
        if n <= kp {
            return Ok(false);
        }
        let (bs, ct) = (g.cand_block as usize, g.cand_top as usize);
        let nb = n.div_ceil(bs);
        let use_cand = !is_cand && sh.cand && g.cand_source.is_some_and(|s| l > s);
        let rope = self.m.rope(l);
        let rows = (IDX_S_FLOATS / n).clamp(1, t);
        for q0 in (0..t).step_by(rows) {
            let qn = rows.min(t - q0);
            let mk = p.arena.mark();
            let iq = p.mm(l, Role::IdxQB, &qr.view(q0 * g.q_lora as usize * 4, qn * g.q_lora as usize * 4)?, qn)?;
            o.rope(&iq, qn, ih, ih * id, id, id - rd, rd, pos0 + q0, 1, rope, false)?;
            o.quant_sim(&iq, qn * ih, id, 32, 1)?;
            // the head weights' positive scale ranks nothing differently: left out
            let w = p.mm(l, Role::IdxProj, &x.view(q0 * g.n_embd as usize * 4, qn * g.n_embd as usize * 4)?, qn)?;
            let score = p.arena.f32(qn * n)?;
            o.idx_score(&iq, &w, &c.idx, &score, qn, ih, id, n, n, pos0 + q0, r)?;
            if is_cand && nb > ct {
                let bmax = p.arena.f32(qn * nb)?;
                o.block_max(&score, &bmax, qn, n, n, bs, pos0 + q0, r)?;
                o.topk(&bmax, &p.cand.view(q0 * ct * 4, qn * ct * 4)?, qn, nb, nb, ct)?;
                sh.cand = true;
            } else if use_cand && nb > ct {
                let mark = p.arena.bytes(qn * nb)?;
                o.cand_mask(&score, &p.cand.view(q0 * ct * 4, qn * ct * 4)?, &mark, qn, n, n, ct, bs)?;
            }
            if qn == t {
                tap(&format!("indexer_score-{l}"), &score)?;
            }
            o.topk(&score, &p.sel.view(q0 * kp * 4, qn * kp * 4)?, qn, n, n, kp)?;
            p.arena.rewind(mk);
        }
        Ok(true)
    }

    /// The attention half of layer l on x [t, d] (normed) at pos0: the window's keys into the ring, a source's
    /// compressor, an index source's picks, then per block of query rows the attention and the grouped output
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, p: &Part, l: u64, x: &DevBuf, layers: &[Att], pos0: usize, t: usize, sh: &mut Shared, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let eps = g.rms_eps as f32;
        let (nh, hd, rd, ql) = (g.n_head as usize, g.head_dim as usize, g.rope_dim as usize, g.q_lora as usize);
        let (ng, ol) = (g.o_groups as usize, g.o_lora as usize);
        let gw = nh * hd / ng;
        let rope = self.m.rope(l);
        let st = &layers[l as usize];
        let qr = p.mm(l, Role::QA, x, t)?;
        o.rms_norm(&qr, Some(p.vec(l, Role::QANorm)?), &qr, t, ql, eps)?;
        let kv = p.mm(l, Role::Kv, x, t)?;
        o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, t, hd, eps)?;
        o.rope(&kv, t, 1, hd, hd, hd - rd, rd, pos0, 1, rope, false)?;
        o.quant_sim(&kv, t, hd, 32, 0)?;
        tap(&format!("kv-{l}"), &kv)?;
        let rr = raw_ring();
        o.ring_put(&kv, &st.raw, t, pos0, rr, hd)?;
        // the compressed keys this layer reads: its source's
        let src = g.kv_source(l);
        if let Some(c) = &st.comp {
            self.compress(p, l, x, c, pos0, t, &mut *tap)?;
        }
        let comp = src.and_then(|s| layers[s as usize].comp.as_ref());
        if g.index_sources.contains(&l) {
            let c = comp.ok_or("an index source without its kv source's keys")?;
            sh.sel = self.select(p, l, x, &qr, c, pos0, t, sh, &mut *tap)?;
        }
        let kp = g.idx_top_k as usize;
        let rb = if t <= MMVQ_COLS { t } else { 1024.min(t) };
        let oa = p.arena.f32(t * ng * ol)?;
        let q = p.arena.f32(rb * nh * hd)?;
        let att = p.arena.f32(rb * nh * hd)?;
        let og = p.arena.f32(rb * ng * ol)?;
        let sinks = p.vec(l, Role::Sinks)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let qb = p.mat(l, Role::QB)?;
        for r0 in (0..t).step_by(rb) {
            let tr = rb.min(t - r0);
            p.matmul(qb, tr, (&qr, r0 * ql, ql), (&q, 0, nh * hd), false)?;
            o.rope(&q, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, false)?;
            if r0 == 0 && tr == t {
                tap(&format!("q-{l}"), &q)?;
            }
            let compv = comp.map(|c| (&c.cache, c.ratio, c.cache.len / (512 * 2)));
            let sel = if comp.is_some() && sh.sel { Some((p.sel.view(r0 * kp * 4, tr * kp * 4)?, kp)) } else { None };
            o.attend(&q, &st.raw, rr, compv, sel.as_ref().map(|(s, k)| (s, *k)), sinks, &att, tr, nh, pos0 + r0, g.window as usize, scale)?;
            o.rope(&att, tr, nh, nh * hd, hd, hd - rd, rd, pos0 + r0, 1, rope, true)?;
            if r0 == 0 && tr == t {
                tap(&format!("attn_derope-{l}"), &att)?;
            }
            o.quant_sim(&att, tr * nh, hd, 32, 0)?;
            self.out_a(p, l, &att, &q, &og, &oa, r0, tr)?;
        }
        let out = p.mm(l, Role::OutB, &oa, t)?;
        let _ = gw;
        tap(&format!("attn_out-{l}"), &out)?;
        Ok(out)
    }

    /// The grouped o_a: att [tr][groups][gw] -> oa rows from r0 ([t][groups][o_lora]); `perm` / `og` scratch
    #[allow(clippy::too_many_arguments)]
    fn out_a(&self, p: &Part, l: u64, att: &DevBuf, perm: &DevBuf, og: &DevBuf, oa: &DevBuf, r0: usize, tr: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (ng, ol, gw) = (g.o_groups as usize, g.o_lora as usize, (g.n_head * g.head_dim / g.o_groups) as usize);
        match p.groups.get(&l) {
            Some(_) if tr <= MMVQ_COLS && q4k_own() => {
                // every group in one launch: rows [g o_lora, ..) read their heads [g gw, ..) of att's row
                let w = p.mat(l, Role::OutA)?;
                o.q4k_mv(&w.buf, w.rows, w.cols, (att, 0, ng * gw), Some((ol, gw)), (oa, r0 * ng * ol, ng * ol), tr)
            }
            Some(gm) => {
                let perm = perm.view(0, tr * ng * gw * 4)?;
                o.permute(att, &perm, tr, ng, gw)?;
                for (gi, w) in gm.iter().enumerate() {
                    p.matmul(w, tr, (&perm, gi * tr * gw, gw), (og, gi * tr * ol, ol), false)?;
                }
                o.permute(og, &oa.view(r0 * ng * ol * 4, tr * ng * ol * 4)?, ng, tr, ol)
            }
            None if tr <= MMVQ_COLS => {
                // fp8 o_a (the towers'), every group in one launch as the q4_K one
                let w = p.mat(l, Role::OutA)?;
                o.fp8_mmv(&w.buf, w.rows, w.cols, (att, 0, ng * gw), Some((ol, gw)), (oa, r0 * ng * ol, ng * ol), tr)
            }
            None => {
                // fp8 o_a (the towers'): each group's rows [g * o_lora, ..) expanded (whole 32-row tiles) against its heads
                let w = p.mat(l, Role::OutA)?;
                let w16 = p.arena.bytes(ol * gw * 2)?;
                let x16 = p.arena.bytes(tr * gw * 2)?;
                let perm = perm.view(0, tr * ng * gw * 4)?;
                o.permute(att, &perm, tr, ng, gw)?;
                for gi in 0..ng {
                    o.fp8_dequant(&w.buf, 0, w.rows, w.cols, gi * ol, ol, &w16, true)?;
                    o.to_f16(&perm.view(gi * tr * gw * 4, tr * gw * 4)?, &x16, tr * gw)?;
                    o.gemm_f16(tr, ol, gw, (&x16, 0, gw), (&w16, 0), (og, gi * tr * ol, ol), false)?;
                }
                o.permute(og, &oa.view(r0 * ng * ol * 4, tr * ng * ol * 4)?, ng, tr, ol)
            }
        }
    }

    /// The mixes of a half of block l on the streams x: post, comb and this half's pre (for the half after); the
    /// half's input collapsed by `pre_in` (the half before's), normed into `normed`
    #[allow(clippy::too_many_arguments)]
    fn hc_half(&self, p: &Part, l: u64, fn_: Role, base: Role, scale: Role, norm: Role, x: &DevBuf, pre_in: &DevBuf, ws: &[DevBuf; 6], t: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let [h, normed, post, comb, pre_out, mixes] = ws;
        let w = p.mat(l, fn_)?;
        if !w.f32() || w.rows != 24 {
            return Err(Error(format!("block {l}: the mHC mixes are float32 [24, 4 d]")));
        }
        if t <= MMVQ_COLS && hc_fused() {
            // decode widths: the mixes and the Sinkhorn in glm5next's two fused launches (its h and norm, by this half's
            // own mix, are replaced below)
            o.hc_pre_fused(x, &w.buf, p.vec(l, scale)?, p.vec(l, base)?, p.vec(l, norm)?, h, post, comb, pre_out, normed, &p.arena.f32(t * 32 * 25)?, t, d,
                           eps, g.hc_eps as f32, g.hc_iters as u32)?;
        } else {
            o.hc_mix(x, &w.buf, mixes, &p.arena.f32(t * 32 * 25)?, t, 4 * d, eps)?;
            o.hc_pre(mixes, p.vec(l, scale)?, p.vec(l, base)?, x, h, post, comb, pre_out, t, d, g.hc_eps as f32, g.hc_iters as u32)?;
        }
        o.hc_collapse(x, pre_in, h, t, d)?;
        o.rms_norm(h, Some(p.vec(l, norm)?), normed, t, d, eps)
    }

    /// One run of layers (`run`: its part, first, end) over the carried state x ([t, 4, d] streams, [t, 4] mix) of the
    /// pass's `tokens` at pos0; the Engram's rows taken at its layers; the target layers' stream means kept for the towers
    #[allow(clippy::too_many_arguments)]
    fn run_layers(&self, layers: &[Att], run: (usize, u64, u64), x: DevBuf, pos0: usize, tokens: &[u32], eng: &[(usize, Vec<u64>, Arc<engram::Fetch>)],
                  mut mh: Option<&mut MtpS>, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let t = tokens.len();
        let d = g.n_embd as usize;
        let p = &self.parts[run.0];
        let o = &p.ops;
        let gpu = &o.gpu;
        let sz = t * 4 * d + t * 4;
        let mut ring = vec![x, DevBuf::f32(gpu, sz)?, DevBuf::f32(gpu, sz)?];
        let mut ix = 0usize;
        // h, normed, post, comb, pre_out, mixes; and the attention's pre (the FFN's input mix)
        let ws = [DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * 4)?, DevBuf::f32(gpu, t * 16)?, DevBuf::f32(gpu, t * 4)?,
                  DevBuf::f32(gpu, t * 24)?];
        let attn_pre = DevBuf::f32(gpu, t * 4)?;
        let mut sh = Shared::default();
        let keep = t.min(MTP_KEEP);
        for l in run.1..run.2 {
            let lt0 = Instant::now();
            p.arena.reset();
            let xs = ring[ix].view(0, t * 4 * d * 4)?;
            let pre_in = ring[ix].view(t * 4 * d * 4, t * 16)?;
            let (x1, x2) = (&ring[(ix + 1) % 3], &ring[(ix + 2) % 3]);
            tap(&format!("l_in-{l}"), &xs)?;
            if l > 0 {
                tap(&format!("pre_mix-{l}"), &pre_in)?;
            }
            if let Some(k) = g.engram_layers.iter().position(|e| *e == l) {
                let en = self.engram.as_ref().ok_or("the Engram is not loaded")?;
                let (_, rows, f) = eng.iter().find(|(li, _, _)| *li == k).ok_or("the Engram's rows were not fetched")?;
                let raw = self.timed(p, "engram: waiting for rows", || en.take(k, rows, f))?;
                self.timed(p, "engram", || self.engram_apply(p, l, &xs, &raw, t, &mut *tap))?;
            }
            if let (Some(ms), Some(k)) = (mh.as_deref_mut(), g.mtp_targets.iter().position(|x| *x == l)) {
                let mean = p.arena.f32(t * d)?;
                o.hc_mean(&xs, &mean, t, d)?;
                o.kcopy(&ms.mh, (k * MTP_KEEP) * d * 4, &mean, (t - keep) * d * 4, keep * d * 4)?;
            }
            let (normed, post, comb, pre_out) = (&ws[1], &ws[2], &ws[3], &ws[4]);
            self.timed(p, "hc", || self.hc_half(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, &xs, &pre_in, &ws, t))?;
            o.kcopy(&attn_pre, 0, pre_out, 0, t * 16)?;
            tap(&format!("attn_norm-{l}"), normed)?;
            let att = self.timed(p, "attention", || self.attention(p, l, normed, layers, pos0, t, &mut sh, &mut *tap))?;
            o.hc_post(&att, &xs, post, comb, x1, t, d)?;
            tap(&format!("hc_attn_post-{l}"), &x1.view(0, t * 4 * d * 4)?)?;
            self.timed(p, "hc", || self.hc_half(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, x1, &attn_pre, &ws, t))?;
            tap(&format!("ffn_norm-{l}"), normed)?;
            let split = (t > MMVQ_COLS && pipe_split()).then(|| -> Result<Instant> {
                gpu.sync()?;
                Ok(Instant::now())
            }).transpose()?;
            let ffn = self.moe(p, l, t, normed, &mut *tap)?;
            if let Some(t1) = split {
                gpu.sync()?;
                let (a, m) = (t1.duration_since(lt0).as_secs_f64(), t1.elapsed().as_secs_f64());
                eprintln!("[layer {l} ({t} tokens): attention and the rest {:.1} ms, experts {:.1} ms]", a * 1e3, m * 1e3);
            }
            tap(&format!("ffn_out-{l}"), &ffn)?;
            o.hc_post(&ffn, x1, post, comb, x2, t, d)?;
            // the FFN's pre rides with the streams to the next layer (and part)
            o.kcopy(x2, t * 4 * d * 4, pre_out, 0, t * 16)?;
            tap(&format!("l_out-{l}"), &x2.view(0, t * 4 * d * 4)?)?;
            ix = (ix + 2) % 3;
        }
        Ok(ring.swap_remove(ix))
    }

    /// The head on the last part for the last `n_out` rows of x: the streams collapsed by the carried mix, the norm,
    /// the output matrix
    fn head(&self, x: &DevBuf, t: usize, n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.m.g;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        p.arena.reset();
        let xv = x.view((t - n_out) * 4 * d * 4, n_out * 4 * d * 4)?;
        let pre = x.view((t * 4 * d + (t - n_out) * 4) * 4, n_out * 16)?;
        let h = p.arena.f32(n_out * d)?;
        o.hc_collapse(&xv, &pre, &h, n_out, d)?;
        o.rms_norm(&h, Some(p.vec(0, Role::OutputNorm)?), &h, n_out, d, eps)?;
        tap("result_norm", &h)?;
        let logits = p.mm(0, Role::Output, &h, n_out)?;
        tap("result_output", &logits)?;
        let v = p.read_f32(&logits, n_out * g.n_vocab as usize)?;
        Ok(v.chunks(g.n_vocab as usize).map(|c| c.to_vec()).collect())
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

    /// The Engram's rows of the pass's tokens (positions pos0..), on their way: (engram layer, rows, the fetch)
    fn engram_start(&self, hist: &[u32], pos0: usize) -> Vec<(usize, Vec<u64>, Arc<engram::Fetch>)> {
        let Some(en) = &self.engram else { return Vec::new() };
        en.rows(hist, pos0).into_iter().enumerate().map(|(k, rows)| {
            let f = en.fetch(k, &rows);
            (k, rows, f)
        }).collect()
    }

    pub fn feed(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(sess, tokens, &|_| false, tap)?.1)
    }

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
    /// one runs the second's, the head and the towers' push. Each GPU keeps the chunks' order: the same result.
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
        let start = sess.pos;
        sess.hist.truncate(start);
        for c in chunks {
            sess.hist.extend_from_slice(c);
        }
        let Session { layers, hist, mtp, .. } = sess;
        let layers: &[Att] = layers;
        let hist: &[u32] = hist;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<u8>>>(1);
        let done = std::thread::scope(|sc| -> Result<(usize, Vec<f32>)> {
            let ahead = sc.spawn(move || {
                let mut none = |_: &str, _: &DevBuf| Ok(());
                let mut at = start;
                // each chunk's Engram rows asked for a chunk ahead (the reads of a chunk's rows outlast its first layer)
                let mut next = chunks.first().map(|c| self.engram_start(&hist[..at + c.len()], at));
                for (i, c) in chunks.iter().enumerate() {
                    if i > 0 && stop(chunks[..i].iter().map(|c| c.len()).sum()) {
                        return;
                    }
                    let t0 = Instant::now();
                    let eng = next.take().unwrap_or_default();
                    if let Some(n) = chunks.get(i + 1) {
                        next = Some(self.engram_start(&hist[..at + c.len() + n.len()], at + c.len()));
                    }
                    let r = self.streams(runs.0.0, c, &mut none).and_then(|x| self.run_layers(layers, runs.0, x, at, c, &eng, None, &mut none)).and_then(|x| {
                        let mut v = vec![0u8; x.len];
                        x.read(0, &mut v)?;
                        Ok(v)
                    });
                    if pipe_log() {
                        eprintln!("[pipeline: chunk {i} ({} tokens) on the first GPU in {:.3} s]", c.len(), t0.elapsed().as_secs_f64());
                    }
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
                let tw = Instant::now();
                let Ok(v) = rx.recv() else { break };
                let waited = tw.elapsed().as_secs_f64();
                let step = (|| -> Result<Vec<f32>> {
                    let v = v?;
                    let x = DevBuf::new(&p1.ops.gpu, v.len())?;
                    x.write(0, &v)?;
                    let eng = Vec::new();
                    let x = self.run_layers(layers, runs.1, x, at, c, &eng, mtp.as_mut(), &mut *tap)?;
                    let l = self.head(&x, c.len(), 1, &mut *tap)?.pop().unwrap_or_default();
                    if let Some(ms) = mtp.as_mut() {
                        self.mtp_push(ms, at, c.len(), false)?;
                    }
                    Ok(l)
                })();
                if pipe_log() {
                    eprintln!("[pipeline: chunk at {at} on the second GPU in {:.3} s after waiting {waited:.3} s]", tw.elapsed().as_secs_f64() - waited);
                }
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
        sess.pos += done.0;
        sess.hist.truncate(sess.pos);
        sess.snapped = None;
        Ok(done)
    }

    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.forward_rows(sess, tokens, 1, tap)?.pop().unwrap_or_default())
    }

    /// One pass of at most a chunk: the logits of the last `n_out` tokens (1..=8). The towers' rings take the pass's
    /// rows (a verify pass's overwritten slots kept for `rollback`).
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
        sess.hist.truncate(pos0);
        sess.hist.extend_from_slice(tokens);
        let eng = self.engram_start(&sess.hist, pos0);
        let mut x = self.streams(0, tokens, &mut *tap)?;
        let mut staging = Vec::new();
        for (k, run) in self.runs().into_iter().enumerate() {
            if k > 0 {
                x = self.to_part(run.0, &x, &mut staging)?;
            }
            let last = k + 1 == self.parts.len();
            x = self.run_layers(&sess.layers, run, x, pos0, tokens, &eng, if last { sess.mtp.as_mut() } else { None }, &mut *tap)?;
        }
        let out = self.head(&x, t, n_out, &mut *tap)?;
        if let Some(ms) = sess.mtp.as_mut() {
            self.mtp_push(ms, pos0, t, (2..=MAX_VERIFY).contains(&t))?;
        }
        sess.pos += t;
        sess.snapped = (2..=MAX_VERIFY).contains(&t).then_some((pos0, t));
        Ok(out)
    }

    /// Keeps only the first `keep` tokens of the last (verify) pass: the position back, the towers' ring slots the
    /// rest overwrote restored (the main caches need nothing: positions bound what is read)
    pub fn rollback(&self, sess: &mut Session, keep: usize) -> Result<()> {
        let (pos0, t) = sess.snapped.take().ok_or("rollback: the last pass was not a verify pass")?;
        if keep == 0 || keep >= t {
            return Err(Error(format!("rollback: keep {keep} of {t} rows")));
        }
        sess.pos = pos0 + keep;
        sess.hist.truncate(sess.pos);
        if let Some(ms) = sess.mtp.as_mut() {
            if let Some((p0, n)) = ms.snap_at.take() {
                let p = self.parts.last().unwrap();
                for (ti, ring) in ms.rings.iter().enumerate() {
                    for i in keep..n {
                        let slot = (p0 + i) % MTP_WIN;
                        p.ops.kcopy(ring, slot * 1024, &ms.snap, (ti * MAX_VERIFY + i) * 1024, 1024)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The pass's rows (their stream means at the target layers, kept in ms.mh) into each tower's window: main_x =
    /// main_norm(main_proj(the means)), each tower's key kv_norm(wkv(main_x)) rotated at the row's position, fp8;
    /// with `snap`, the slots overwritten kept first
    fn mtp_push(&self, ms: &mut MtpS, pos0: usize, t: usize, snap: bool) -> Result<()> {
        let g = &self.m.g;
        let (d, hd, rd, eps) = (g.n_embd as usize, g.head_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        let keep = t.min(MTP_KEEP);
        let first = pos0 + t - keep;
        ms.mh_rows = (first, keep);
        let mk = p.arena.mark();
        // [3][keep][d] -> [keep][3][d]
        let cat = p.arena.f32(keep * 3 * d)?;
        let src = if keep == MTP_KEEP {
            ms.mh.view(0, 3 * keep * d * 4)?
        } else {
            let s = p.arena.f32(3 * keep * d)?;
            for k in 0..3 {
                o.kcopy(&s, k * keep * d * 4, &ms.mh, k * MTP_KEEP * d * 4, keep * d * 4)?;
            }
            s
        };
        o.permute(&src, &cat, 3, keep, d)?;
        let mx = p.mm(0, Role::MtpMainProj, &cat, keep)?;
        o.rms_norm(&mx, Some(p.vec(0, Role::MtpMainNorm)?), &mx, keep, d, eps)?;
        // only the last 128 matter to a window
        let (n, skip) = (keep.min(MTP_WIN), keep.saturating_sub(MTP_WIN));
        let mxv = mx.view(skip * d * 4, n * d * 4)?;
        ms.snap_at = None;
        if snap && n <= MAX_VERIFY {
            for (ti, ring) in ms.rings.iter().enumerate() {
                for i in 0..n {
                    let slot = (first + skip + i) % MTP_WIN;
                    o.kcopy(&ms.snap, (ti * MAX_VERIFY + i) * 1024, ring, slot * 1024, 1024)?;
                }
            }
            ms.snap_at = Some((first + skip, n));
        }
        for (ti, ring) in ms.rings.iter().enumerate() {
            let l = g.n_layer + ti as u64;
            let kv = p.mm(l, Role::Kv, &mxv, n)?;
            o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, n, hd, eps)?;
            o.rope(&kv, n, 1, hd, hd, hd - rd, rd, first + skip, 1, self.m.rope(l), false)?;
            o.quant_sim(&kv, n, hd, 32, 0)?;
            o.ring_put(&kv, ring, n, first + skip, MTP_WIN, hd)?;
        }
        p.arena.rewind(mk);
        Ok(())
    }

    /// The towers' draft after the committed token `tail` (fed next, at position `pos`): the block [tail, noise x 4]
    /// through the three towers, the head, the Markov bias token by token; up to `k` tokens for the positions after
    /// `tail`'s
    fn mtp_draft(&self, ms: &MtpS, tail: u32, pos: usize, k: usize, tap: Tap) -> Result<Vec<u32>> {
        let g = &self.m.g;
        let (d, hd, rd, eps) = (g.n_embd as usize, g.head_dim as usize, g.rope_dim as usize, g.rms_eps as f32);
        let (nh, ql) = (g.n_head as usize, g.q_lora as usize);
        let b = g.mtp_block as usize;
        let pi = self.parts.len() - 1;
        let p = &self.parts[pi];
        let o = &p.ops;
        let t0 = Instant::now();
        let mut toks = vec![g.mtp_noise; b];
        toks[0] = tail;
        let mut x = self.streams(pi, &toks, &mut *tap)?;
        let sz = b * 4 * d + b * 4;
        let x2 = DevBuf::f32(&o.gpu, sz)?;
        let ws = [DevBuf::f32(&o.gpu, b * d)?, DevBuf::f32(&o.gpu, b * d)?, DevBuf::f32(&o.gpu, b * 4)?, DevBuf::f32(&o.gpu, b * 16)?, DevBuf::f32(&o.gpu, b * 4)?,
                  DevBuf::f32(&o.gpu, b * 24)?];
        let attn_pre = DevBuf::f32(&o.gpu, b * 4)?;
        let x1 = DevBuf::f32(&o.gpu, sz)?;
        let n_ring = pos.min(MTP_WIN);
        // every block row reads the whole window and the whole block: the keys as one list, picked all by each row
        let nk = n_ring + b;
        let mut sel = Vec::with_capacity(b * nk);
        for _ in 0..b {
            sel.extend((0..nk as i32).collect::<Vec<_>>());
        }
        let selb = DevBuf::new(&o.gpu, b * nk * 4)?;
        p.upload(&selb, &i32s(&sel))?;
        let dummy = DevBuf::new(&o.gpu, 1024)?;
        for (ti, ring) in ms.rings.iter().enumerate() {
            let l = g.n_layer + ti as u64;
            p.arena.reset();
            let xs = x.view(0, b * 4 * d * 4)?;
            let pre_in = x.view(b * 4 * d * 4, b * 16)?;
            let (normed, post, comb, pre_out) = (&ws[1], &ws[2], &ws[3], &ws[4]);
            self.timed(p, "draft: hc", || self.hc_half(p, l, Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, Role::AttnNorm, &xs, &pre_in, &ws, b))?;
            o.kcopy(&attn_pre, 0, pre_out, 0, b * 16)?;
            let ta = Instant::now();
            // the attention: q and the block's keys from the block, the window's from the ring
            let rope = self.m.rope(l);
            let qr = p.mm(l, Role::QA, normed, b)?;
            o.rms_norm(&qr, Some(p.vec(l, Role::QANorm)?), &qr, b, ql, eps)?;
            let q = p.mm(l, Role::QB, &qr, b)?;
            o.rope(&q, b, nh, nh * hd, hd, hd - rd, rd, pos, 1, rope, false)?;
            let kv = p.mm(l, Role::Kv, normed, b)?;
            o.rms_norm(&kv, Some(p.vec(l, Role::KvNorm)?), &kv, b, hd, eps)?;
            o.rope(&kv, b, 1, hd, hd, hd - rd, rd, pos, 1, rope, false)?;
            o.quant_sim(&kv, b, hd, 32, 0)?;
            // a spare row a block row past the list: the kernel's rows-seen check (each row sees nk + 1.. of them)
            let keys = p.arena.bytes((nk + b + 1) * hd * 2)?;
            if n_ring > 0 {
                o.kcopy(&keys, 0, ring, 0, n_ring * hd * 2)?;
            }
            o.rows_f16(&kv, &keys, b, n_ring, hd)?;
            let att = p.arena.f32(b * nh * hd)?;
            // as the attention's compressed keys at ratio 1, every row seeing more than nk of them and picking all
            o.attend(&q, &dummy, 1, Some((&keys, 1, nk + b + 1)), Some((&selb, nk)), p.vec(l, Role::Sinks)?, &att, b, nh, nk, 0, 1.0 / (hd as f32).sqrt())?;
            o.rope(&att, b, nh, nh * hd, hd, hd - rd, rd, pos, 1, rope, true)?;
            o.quant_sim(&att, b * nh, hd, 32, 0)?;
            let oa = p.arena.f32(b * g.o_groups as usize * g.o_lora as usize)?;
            let og = p.arena.f32(b * g.o_groups as usize * g.o_lora as usize)?;
            self.out_a(p, l, &att, &q, &og, &oa, 0, b)?;
            let a = p.mm(l, Role::OutB, &oa, b)?;
            self.lap(p, "draft: attention", ta)?;
            o.hc_post(&a, &xs, post, comb, &x1, b, d)?;
            self.hc_half(p, l, Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, Role::FfnNorm, &x1, &attn_pre, &ws, b)?;
            let ffn = self.timed(p, "draft: moe", || self.moe(p, l, b, normed, &mut *tap))?;
            o.hc_post(&ffn, &x1, post, comb, &x2, b, d)?;
            o.kcopy(&x2, b * 4 * d * 4, pre_out, 0, b * 16)?;
            o.kcopy(&x, 0, &x2, 0, sz)?;
        }
        // the head: the streams by the last mix, the towers' norm, the model's output matrix
        p.arena.reset();
        let h = p.arena.f32(b * d)?;
        o.hc_collapse(&x.view(0, b * 4 * d * 4)?, &x.view(b * 4 * d * 4, b * 16)?, &h, b, d)?;
        o.rms_norm(&h, Some(p.vec(0, Role::MtpOutNorm)?), &h, b, d, eps)?;
        // only the rows a draft is taken from (row i: the token after position pos + i)
        let k = k.min(b);
        let logits = self.timed(p, "draft: head", || p.mm(0, Role::Output, &h, k))?;
        tap("mtp_logits", &logits)?;
        let vocab = g.n_vocab as usize;
        // the Markov bias, on the GPU: the previous token's (tail, then each draft) rank-256 embedding through its head,
        // added to its row, the argmax the next previous token
        let toks = p.arena.bytes((k + 1) * 4)?;
        p.upload(&toks, &(tail as i32).to_le_bytes())?;
        let emb = p.mat(0, Role::MtpMarkovEmbd)?;
        let me = p.arena.f32(emb.cols)?;
        for i in 0..k {
            o.gather_rows(&emb.buf, emb.rows, &toks, i * 4, 1, emb.cols, &me)?;
            let bias = p.mm(0, Role::MtpMarkovHead, &me, 1)?;
            o.markov_step((&logits, i * vocab), &bias, vocab, &toks, (i + 1) * 4)?;
        }
        let tv = p.read_f32(&toks, k + 1)?;
        let _ = (t0, &mut x);
        Ok(tv[1..].iter().map(|f| f.to_bits()).collect())
    }

    pub fn decoder(&self, logits: Vec<f32>, draft: bool) -> Decoder {
        Decoder { logits, next: None, drafts: Vec::new(), mtp: draft && self.mtp, drafted: 0, accepted: 0 }
    }

    pub fn decoder_after(&self, next: u32, draft: bool) -> Decoder {
        Decoder { logits: Vec::new(), next: Some(next), drafts: Vec::new(), mtp: draft && self.mtp, drafted: 0, accepted: 0 }
    }

    /// The next committed token(s): the pending token with the towers' drafts in one pass, each draft kept exactly when
    /// the token drawn for its position is the draft itself (what is committed is distributed as plain sampling)
    pub fn step(&self, sess: &mut Session, dec: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        let room = sess.max_ctx.saturating_sub(sess.pos + 1);
        let (out, tail) = match dec.next {
            Some(x) if dec.mtp && !dec.drafts.is_empty() && room > 0 => {
                let mut dr = std::mem::take(&mut dec.drafts);
                dr.truncate(room.min(mtp_drafts()));
                let mut toks = vec![x];
                toks.extend_from_slice(&dr);
                let rows = self.forward_rows(sess, &toks, toks.len(), &mut *tap)?;
                let mut out = Vec::with_capacity(toks.len());
                for (i, r) in rows.iter().enumerate() {
                    let y = smp.sample(r);
                    out.push(y);
                    if i < dr.len() {
                        dec.drafted += 1;
                        if y == dr[i] {
                            dec.accepted += 1;
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
            }
            _ => {
                if let Some(x) = dec.next.take() {
                    dec.logits = self.forward(sess, &[x], &mut *tap)?;
                }
                let y = smp.sample(&dec.logits);
                (vec![y], y)
            }
        };
        dec.drafts = match (dec.mtp, sess.mtp.as_ref()) {
            (true, Some(ms)) if sess.pos + 2 <= sess.max_ctx => self.timed(self.parts.last().unwrap(), "draft", || self.mtp_draft(ms, tail, sess.pos, mtp_drafts(), &mut *tap))?,
            _ => Vec::new(),
        };
        dec.next = Some(tail);
        Ok(out)
    }

    /// One decode step of several conversations: each its own pass (the same result as alone)
    pub fn forward_batch(&self, sess: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != sess.len() {
            return Err(Error(format!("forward_batch: {} tokens for {} sessions", tokens.len(), sess.len())));
        }
        sess.iter_mut().zip(tokens).map(|(s, t)| self.forward(s, &[*t], &mut *tap)).collect()
    }

    /// The MoE half of block l on x [t, d]: the router (on the host), the shared expert, the routed experts grouped by
    /// expert - from their indices at decode widths, expanded to fp16 for a prompt
    fn moe(&self, p: &Part, l: u64, t: usize, x: &DevBuf, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let d = g.n_embd as usize;
        let f = g.ffn_expert as usize;
        let (ne, used) = { let (a, b) = g.experts(l); (a as usize, b as usize) };
        let lim = g.swiglu_limit as f32;
        let logits = p.mm(l, Role::Router, x, t)?;
        let pf = prefetch_limit();
        let next_main = l + 1 < g.n_layer && !g.is_mtp(l) && p.layers.contains(&(l + 1));
        let guess = if t <= MMVQ_COLS && pf > 0 && next_main { Some(p.mm(l + 1, Role::Router, x, t)?) } else { None };
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let mut down: Vec<(&DevBuf, usize)> = vec![(&logits, t * ne)];
        if let Some(gl) = &guess {
            down.push((gl, t * ne));
        }
        let pending = p.read_start(&down)?;
        // the shared expert, queued while the host picks the routed ones
        let y = {
            let sg = p.mm(l, Role::ShGate, x, t)?;
            let su = p.mm(l, Role::ShUp, x, t)?;
            o.swiglu_clamp(&sg, &su, &sg, t * f, lim)?;
            p.mm(l, Role::ShDown, &sg, t)?
        };
        let tw = Instant::now();
        let (lv, guess) = match pending {
            Some((tk, spans)) => {
                let mut v = p.read_finish(tk, &spans)?.into_iter();
                (v.next().unwrap_or_default(), v.next())
            }
            None => (logits.to_f32()?, match &guess { Some(gl) => Some(gl.to_f32()?), None => None }),
        };
        ROUTER_WAIT_NS.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let turn = Instant::now();
        let bias = p.router_bias(l)?;
        let th = Instant::now();
        let pick = |ti: usize| -> Vec<(usize, f32)> {
            let pr: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| sqrt_softplus(*z)).collect();
            let key = |i: usize| pr[i] + bias[i];
            let mut order: Vec<usize> = (0..ne).collect();
            order.select_nth_unstable_by(used - 1, |a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
            let mut s = order[..used].to_vec();
            s.sort_by(|a, c| key(*c).total_cmp(&key(*a)).then(a.cmp(c)));
            let sum: f32 = s.iter().map(|i| pr[*i]).sum::<f32>() + 1e-20;
            s.iter().map(|&i| (i, pr[i] / sum * g.expert_scale as f32)).collect()
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
        tap(&format!("ffn_shexp-{l}"), &y)?;
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        let promote = t <= MMVQ_COLS;
        let staging = !promote && p.ring.is_some() && !p.draft.contains_key(&l);
        let (slots_of, arriving, copies) =
            if staging { (Vec::new(), Vec::new(), None) } else { self.timed(p, "experts: misses", || p.ensure(&self.m, l, &need, promote))? };
        let vq = &p.vq[&l];
        let cb = &p.cbs[&l];
        let nbit = vq.nbit;
        let total: usize = by.values().map(|v| v.len()).sum();
        let lists: Vec<&Vec<(i32, f32)>> = by.values().collect();
        // the combine: per token its rows of dn, with their weights, in expert order
        let mut first = Vec::with_capacity(lists.len());
        let mut e0 = 0usize;
        for list in &lists {
            first.push(e0);
            e0 += list.len();
        }
        let dn = p.arena.f32(total * d)?;
        if t <= MMVQ_COLS {
            self.timed(p, "experts: products", || {
                let gu = p.arena.f32(2 * total * f)?;
                let (now, later): (Vec<usize>, Vec<usize>) = (0..lists.len()).partition(|&k| !arriving[k]);
                for (set, wait) in [(&now, false), (&later, true)] {
                    if set.is_empty() {
                        continue;
                    }
                    if wait {
                        if let Some(tk) = copies {
                            o.await_ticket(tk)?;
                        }
                    }
                    // gate and up entries (an expert's tokens as its columns), then down
                    let mut ptrs: Vec<u64> = Vec::new();
                    let mut meta: Vec<i32> = Vec::new();
                    let mut dptr: Vec<u64> = Vec::new();
                    let mut dmeta: Vec<i32> = Vec::new();
                    for which in 0..2 {
                        for &k in set.iter() {
                            let base = slots_of[k].ptr() as u64;
                            ptrs.push(base + if which == 0 { vq.gate } else { vq.up } as u64);
                            let mut m = vec![lists[k].len() as i32, (which * total + first[k]) as i32];
                            m.extend(lists[k].iter().map(|(ti, _)| *ti));
                            m.resize(10, 0);
                            meta.extend(m);
                        }
                    }
                    for &k in set.iter() {
                        dptr.push(slots_of[k].ptr() as u64 + vq.down as u64);
                        let mut m = vec![lists[k].len() as i32, first[k] as i32];
                        m.extend((0..lists[k].len()).map(|j| (first[k] + j) as i32));
                        m.resize(10, 0);
                        dmeta.extend(m);
                    }
                    let mut tb: Vec<u8> = ptrs.iter().flat_map(|v| v.to_le_bytes()).collect();
                    tb.extend(i32s(&meta));
                    let mut db: Vec<u8> = dptr.iter().flat_map(|v| v.to_le_bytes()).collect();
                    db.extend(i32s(&dmeta));
                    let mut pick = p.vq_pick.get(&nbit).copied().unwrap_or([0, 0]);
                    // the lookup-table variant's lists: gate | up entries by token (the column of x), downs by their row of h
                    let (mut glut, mut dlut) = (None, None);
                    if pick[0] == crate::ops::VQ_LUT {
                        let mut pairs = Vec::new();
                        for w in 0..2 {
                            for (i, &k) in set.iter().enumerate() {
                                for (q, (ti, _)) in lists[k].iter().enumerate() {
                                    pairs.push((*ti as usize, w * set.len() + i, q));
                                }
                            }
                        }
                        match crate::ops::lut_lists(t, pairs) {
                            Some((v, mx)) => {
                                tb.extend(v);
                                glut = Some((t, mx));
                            }
                            None => pick[0] = 0,
                        }
                    }
                    if pick[1] == crate::ops::VQ_LUT {
                        let mut pairs = Vec::new();
                        for (i, &k) in set.iter().enumerate() {
                            for q in 0..lists[k].len() {
                                pairs.push((first[k] + q, i, q));
                            }
                        }
                        match crate::ops::lut_lists(total, pairs) {
                            Some((v, mx)) => {
                                db.extend(v);
                                dlut = Some((total, mx));
                            }
                            None => pick[1] = 0,
                        }
                    }
                    let gtab = p.arena.bytes(tb.len())?;
                    p.upload(&gtab, &tb)?;
                    let dtab = p.arena.bytes(db.len())?;
                    p.upload(&dtab, &db)?;
                    o.vq_mmv(&gtab, ptrs.len(), (cb, vq_wide((&p.cbf, &p.cbh), l, pick[0])), x, (&gu, 0), 2 * total, f, d, nbit, pick[0], glut)?;
                    // h = silu(gate) up, in place of the gates (rows of the set only: the same op on the others is a no-op
                    // since they are computed once - the whole range is cheap and simplest)
                    let (gv, uv) = (gu.view(0, total * f * 4)?, gu.view(total * f * 4, total * f * 4)?);
                    let hset = p.arena.f32(total * f)?;
                    o.swiglu_clamp(&gv, &uv, &hset, total * f, lim)?;
                    o.vq_mmv(&dtab, dptr.len(), (cb, vq_wide((&p.cbf, &p.cbh), l, pick[1])), &hset, (&dn, 0), total, d, f, nbit, pick[1], dlut)?;
                    if !wait {
                        TURN_NS.fetch_add(turn.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Ok(())
            })?;
        } else if staging {
            self.timed(p, "experts: prompt products", || {
                // the experts in VRAM multiplied while those in host memory and the file are copied up into the staging
                // ring, half of it a wave: a wave's copies queued while the one before is multiplied
                let src = p.locate(l, &need)?;
                let mut views: Vec<Option<DevBuf>> = src.iter().map(|s| match s {
                    Src::V(v) => v.view(0, v.len).map(Some),
                    _ => Ok(None),
                }).collect::<Result<_>>()?;
                let resident: Vec<usize> = (0..need.len()).filter(|&k| views[k].is_some()).collect();
                let mut away: Vec<usize> = (0..need.len()).filter(|&k| matches!(src[k], Src::R(_))).collect();
                away.extend((0..need.len()).filter(|&k| matches!(src[k], Src::F)));
                let k_ring = p.ring.as_ref().map_or(0, |r| r.0.len / r.1);
                let w = (k_ring / 2).max(1);
                let waves: Vec<&[usize]> = away.chunks(w).collect();
                let mut free: Vec<Option<i64>> = vec![*p.ring_free.lock().unwrap(); k_ring];
                // the file's experts read on threads meanwhile (the reads end before the first wave holding one is copied)
                let files: Vec<(usize, u64)> = away.iter().filter(|&&k| matches!(src[k], Src::F)).map(|&k| (k, need[k])).collect();
                let has_file = |wave: &[usize]| wave.iter().any(|&k| matches!(src[k], Src::F));
                if pipe_split() {
                    eprintln!("[layer {l}: experts {} in VRAM, {} in host memory, {} from the file]", resident.len(), away.len() - files.len(), files.len());
                }
                std::thread::scope(|sc| -> Result<()> {
                    let mut reader = (!files.is_empty()).then(|| sc.spawn(|| {
                        let t0 = Instant::now();
                        let r = p.read_files(&self.m, l, &files);
                        if pipe_split() {
                            eprintln!("[layer {l}: {} file experts read in {:.1} ms]", files.len(), t0.elapsed().as_secs_f64() * 1e3);
                        }
                        r
                    }));
                    let mut read: HashMap<usize, usize> = HashMap::new();
                    let mut join = |read: &mut HashMap<usize, usize>| -> Result<()> {
                        if let Some(h) = reader.take() {
                            *read = h.join().map_err(|_| Error("an expert reader panicked".into()))??;
                        }
                        Ok(())
                    };
                    let mut pending: Option<Option<i64>> = None;
                    if let Some(w0) = waves.first() {
                        if has_file(w0) {
                            join(&mut read)?;
                        }
                        let slots: Vec<usize> = (0..w0.len()).collect();
                        let (v, tk) = p.stage_up(&self.m, l, w0, &need, &src, &slots, &free, &read)?;
                        for (&k, b) in w0.iter().zip(v) {
                            views[k] = Some(b);
                        }
                        pending = Some(tk);
                    }
                    self.prompt_set(p, l, x, &lists, &first, &views, &resident, &dn, total)?;
                    for (wi, wave) in waves.iter().enumerate() {
                        if let Some(Some(tk)) = pending.take() {
                            o.await_ticket(tk)?;
                        }
                        self.prompt_set(p, l, x, &lists, &first, &views, wave, &dn, total)?;
                        let done = Some(o.mark()?);
                        free[(wi % 2) * w..(wi % 2) * w + wave.len()].fill(done);
                        if let Some(next) = waves.get(wi + 1) {
                            if has_file(next) {
                                join(&mut read)?;
                            }
                            let h = (wi + 1) % 2;
                            let slots: Vec<usize> = (h * w..h * w + next.len()).collect();
                            let (v, tk) = p.stage_up(&self.m, l, next, &need, &src, &slots, &free, &read)?;
                            for (&k, b) in next.iter().zip(v) {
                                views[k] = Some(b);
                            }
                            pending = Some(tk);
                        }
                    }
                    join(&mut read)
                })?;
                // the ring's last readers: before the next layer's copies
                *p.ring_free.lock().unwrap() = Some(o.mark()?);
                Ok(())
            })?;
        } else {
            self.timed(p, "experts: prompt products", || {
                if let Some(tk) = copies {
                    o.await_ticket(tk)?;
                }
                let views: Vec<Option<DevBuf>> = slots_of.iter().map(|v| v.view(0, v.len).map(Some)).collect::<Result<_>>()?;
                let all: Vec<usize> = (0..lists.len()).collect();
                self.prompt_set(p, l, x, &lists, &first, &views, &all, &dn, total)
            })?;
        }
        let mut t_ptr = vec![0i32];
        let mut ent = Vec::with_capacity(total);
        let mut wv = Vec::with_capacity(total);
        let mut per_tok: Vec<Vec<(usize, f32)>> = vec![Vec::new(); t];
        for (k, list) in lists.iter().enumerate() {
            for (j, (ti, w)) in list.iter().enumerate() {
                per_tok[*ti as usize].push((first[k] + j, *w));
            }
        }
        for v in &per_tok {
            for (r, w) in v {
                ent.push(*r as i32);
                wv.push(*w);
            }
            t_ptr.push(ent.len() as i32);
        }
        let cbuf = p.arena.bytes((t + 1 + total) * 4)?;
        p.upload(&cbuf, &i32s(&t_ptr.iter().chain(&ent).copied().collect::<Vec<_>>()))?;
        let wb = p.arena.f32(total)?;
        p.upload(&wb, &f32s(&wv))?;
        o.moe_combine(&y, &dn, &cbuf, &wb, t, total, d)?;
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
            self.timed(p, "experts: prefetch", || p.prefetch(&self.m, l + 1, &want, pf))?;
        }
        Ok(y)
    }

    /// A prompt pass's products of experts `set` (their payloads at views[k]) for their tokens (lists, first: as moe):
    /// the few-token ones from their indices, the others expanded to fp16 for a GEMM; into dn's rows
    #[allow(clippy::too_many_arguments)]
    fn prompt_set(&self, p: &Part, l: u64, x: &DevBuf, lists: &[&Vec<(i32, f32)>], first: &[usize], views: &[Option<DevBuf>], set: &[usize], dn: &DevBuf,
                  total: usize) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
        let lim = g.swiglu_limit as f32;
        let vq = &p.vq[&l];
        let cb = &p.cbs[&l];
        let nbit = vq.nbit;
        if set.is_empty() {
            return Ok(());
        }
            // experts with few tokens straight from their indices (the decode kernels, up to 8 tokens an entry): an
            // expansion to fp16 writes and reads its whole matrix whatever its tokens
            let few = prompt_mmv_max();
            let small: Vec<usize> = set.iter().copied().filter(|&k| lists[k].len() <= few).collect();
            let mut at = 0;
            while at < small.len() {
                // a batch of experts, at most PROMPT_MMV_SLOTS of their tokens
                let mut end = at;
                let mut slots = 0;
                while end < small.len() && (slots + lists[small[end]].len() <= PROMPT_MMV_SLOTS || end == at) {
                    slots += lists[small[end]].len();
                    end += 1;
                }
                let batch = &small[at..end];
                at = end;
                let mk = p.arena.mark();
                let gu = p.arena.f32(2 * slots * f)?;
                let hh = p.arena.f32(slots * f)?;
                let (mut gp, mut gm, mut dp, mut dm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut slot = 0;
                for &k in batch {
                    let base = views[k].as_ref().ok_or("an expert not in place")?.ptr() as u64;
                    for (ci, ch) in lists[k].chunks(MMVQ_COLS).enumerate() {
                        let j0 = ci * MMVQ_COLS;
                        for (w, off) in [(0usize, vq.gate), (1, vq.up)] {
                            gp.push(base + off as u64);
                            let mut m = vec![ch.len() as i32, (w * slots + slot) as i32];
                            m.extend(ch.iter().map(|(ti, _)| *ti));
                            m.resize(10, 0);
                            gm.extend(m);
                        }
                        dp.push(base + vq.down as u64);
                        let mut m = vec![ch.len() as i32, (first[k] + j0) as i32];
                        m.extend((0..ch.len()).map(|c| (slot + c) as i32));
                        m.resize(10, 0);
                        dm.extend(m);
                        slot += ch.len();
                    }
                }
                let mut tb: Vec<u8> = gp.iter().flat_map(|v| v.to_le_bytes()).collect();
                tb.extend(i32s(&gm));
                let mut db: Vec<u8> = dp.iter().flat_map(|v| v.to_le_bytes()).collect();
                db.extend(i32s(&dm));
                let gtab = p.arena.bytes(tb.len())?;
                p.upload(&gtab, &tb)?;
                let dtab = p.arena.bytes(db.len())?;
                p.upload(&dtab, &db)?;
                let var = prompt_mmv_variant();
                o.vq_mmv(&gtab, gp.len(), (cb, vq_wide((&p.cbf, &p.cbh), l, var)), x, (&gu, 0), 2 * slots, f, d, nbit, var, None)?;
                o.swiglu_clamp(&gu.view(0, slots * f * 4)?, &gu.view(slots * f * 4, slots * f * 4)?, &hh, slots * f, lim)?;
                o.vq_mmv(&dtab, dp.len(), (cb, vq_wide((&p.cbf, &p.cbh), l, var)), &hh, (dn, 0), total, d, f, nbit, var, None)?;
                p.arena.rewind(mk);
            }
            let lists_big: Vec<usize> = set.iter().copied().filter(|&k| lists[k].len() > few).collect();
            if lists_big.is_empty() {
                return Ok(());
            }
            // fp16 x and h readable to whole tiles of 64 tokens (the XMX product's)
            let nmax = lists_big.iter().map(|&k| lists[k].len()).max().unwrap_or(1).next_multiple_of(64);
            let xh = p.arena.bytes(nmax * d * 2)?;
            let gu = p.arena.f32(nmax * 2 * f)?;
            let hh = p.arena.bytes(nmax * f * 2)?;
            let gmax = vq_gemm_max();
            let w16 = if lists_big.iter().any(|&k| lists[k].len() > gmax) { Some(p.arena.bytes(2 * f * d * 2)?) } else { None };
            let toks: Vec<i32> = lists.iter().flat_map(|l| l.iter().map(|(ti, _)| *ti)).collect();
            let tb = p.arena.bytes(total * 4)?;
            p.upload(&tb, &i32s(&toks))?;
            let cbh = &p.cbh[&l];
            for &k in &lists_big {
                let n = lists[k].len();
                let buf = views[k].as_ref().ok_or("an expert not in place")?;
                let tv = tb.view(first[k] * 4, n * 4)?;
                o.gather_f16(x, &tv, &xh, n, d)?;
                match &w16 {
                    Some(w16) if n > gmax => {
                        o.vq_dequant(buf, vq.gate, vq.up - vq.gate, cb, f, d, nbit, &w16.view(0, f * d * 2)?, true)?;
                        o.vq_dequant(buf, vq.up, vq.down - vq.up, cb, f, d, nbit, &w16.view(f * d * 2, f * d * 2)?, true)?;
                        o.gemm_f16(n, 2 * f, d, (&xh, 0, d), (w16, 0), (&gu, 0, 2 * f), false)?;
                        o.swiglu_gu_f16(&gu, &hh, n, f, lim)?;
                        o.vq_dequant(buf, vq.down, vq.span - vq.down, cb, d, f, nbit, &w16.view(0, d * f * 2)?, true)?;
                        o.gemm_f16(n, d, f, (&hh, 0, f), (w16, 0), (dn, first[k] * d, d), false)?;
                    }
                    _ => {
                        let (gv, uv, dv) = (buf.view(vq.gate, vq.up - vq.gate)?, buf.view(vq.up, vq.down - vq.up)?, buf.view(vq.down, vq.span - vq.down)?);
                        o.vq_gemm(&gv, Some(&uv), f, 2 * f, d, nbit, cbh, &xh, d, n, (&gu, 0), 2 * f)?;
                        o.swiglu_gu_f16(&gu, &hh, n, f, lim)?;
                        o.vq_gemm(&dv, None, d, d, f, nbit, cbh, &hh, f, n, (dn, first[k] * d), d)?;
                    }
                }
            }
        Ok(())
    }

    /// The decode requests per expert since load blended into the profile at `path` (old weights halved)
    pub fn save_expert_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let mut prof: HashMap<(u64, u64), f64> = read_profile(path).unwrap_or_default();
        for v in prof.values_mut() {
            *v *= 0.5;
        }
        let mut n = 0;
        for p in &self.parts {
            for s in &p.stores {
                let c = s.lock().unwrap();
                for (k, u) in &c.usage {
                    *prof.entry(*k).or_insert(0.0) += *u as f64;
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
}
