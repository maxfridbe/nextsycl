//! DeepSeek-V3 / R1 on one or more GPUs (`model.rs` has the architecture). The layers are split over the GPUs (a
//! `Part` each: its layers' weights in their stored form, its experts' three-tier store - `store.rs`); the token
//! stream [t, 7168] crosses to the next GPU through host memory where the layers do.
//!
//! MLA is absorbed: the cache holds per token and layer the normed latent and the rotated key part [512 | 64] in fp16;
//! the queries are [k_b^T q_nope | rope(q_pe)] (576), the scores one fp16 GEMM over the cache, the values the
//! probabilities against the cache's latents, then v_b per head (llama.cpp runs this file's whole kv_b unabsorbed - the
//! same arithmetic in another order). Every product with the cache is oneMKL's fp16 GEMM (the XMX units).
//!
//! The routed experts of decode widths run in two launches a layer from their VRAM slots (gate | up, then down, every
//! token's every expert an entry); prompt chunks expand each expert to fp16 once and multiply its tokens. Decode guesses
//! the next layer's experts by running its router on this layer's input, and starts their loads - from pinned memory
//! up to VRAM, from the file into pinned memory - while the GPU computes (`prefetch`).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{Arena, DevBuf, Error, Gpu, HostBuf, Result};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_llm::{GpuInfo, Sampler, Tap};

use crate::model::{Model, Role};
use crate::ops::{Ops, Rope};
use crate::store::{ExpertParts, Reader, Stats, Store};

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;
/// the decode kernels take up to this many rows (tokens) at once
const MMVQ_COLS: usize = 8;
/// widest matrix input (the dense FFN's down projection)
const MAX_COLS: usize = 18432;
/// expert slots and the host tier are allocated in chunks of this size
const CHUNK: usize = 2 << 30;
/// an attention block's scores at most (bytes): rows of a prompt chunk are taken in blocks that fit
const SCORES: usize = 128 << 20;
/// the arena of decode passes; prompt passes take the big one, whose memory holds experts otherwise
const SMALL_ARENA: usize = 384 << 20;
/// the pinned rings of small uploads and readbacks
const RING_BYTES: usize = 8 << 20;
const BACK_BYTES: usize = 4 << 20;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Prompt tokens a forward pass takes at most (NS_DS2_CHUNK, default 1024)
pub fn prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_CHUNK", 1024).clamp(1, 4096))
}

/// Bytes of each GPU's arena (NS_DS2_ARENA_MIB; default 256 MiB and ~1.4 MiB a token of the prompt chunk)
fn arena_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("NS_DS2_ARENA_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) => m << 20,
        None => (256 << 20) + prefill_chunk() * (1400 << 10),
    })
}

/// Passes of at most this many tokens run the experts from VRAM in the decode kernels (NS_DS2_MULTI_MAX, default
/// 32); longer ones expand each expert to fp16 and multiply its tokens
fn multi_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_MULTI_MAX", 32))
}

/// Prompt chunks of at least this many tokens multiply the dense matrices in fp16 (NS_DS2_F16_MIN, default 64)
fn f16_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_F16_MIN", 64))
}

/// Experts a layer starts loading ahead of their layer (NS_DS2_PREFETCH, default 0 = none: with the CPU taking the pinned experts, reading guesses ahead measured slower - 6.70 tok/s without, 5.64 with 4 at P 0.9)
fn prefetch_limit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_PREFETCH", 0))
}

/// The chance of use from which a guessed expert is loaded ahead (NS_DS2_PREFETCH_P, default 0.5)
fn prefetch_p() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS2_PREFETCH_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5))
}

/// Whether a matrix's decode products take this engine's own kernels (Q4_K, Q5_K: the qwen35 silo's wide kernels,
/// copied; NS_DS2_OWN_K=0: the shared ones)
fn own_k(o: &Ops, w: &Mat) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !w.f32 && matches!(w.ty.code(), 12 | 13) && *V.get_or_init(|| std::env::var("NS_DS2_OWN_K").map_or(true, |v| v != "0")) && o.own_mmvq_supported(w.ty.code(), w.cols)
}

/// Decode passes take the one-launch MLA up to this many cells (NS_DS2_MLA_FUSED_MAX, default 8192; 0 = never): every
/// head reads the cache, from the GPU's L2 while it fits
fn fused_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_MLA_FUSED_MAX", 8192))
}

/// Whether decode multiplies the pinned-memory experts on the CPU (NS_DS2_CPU=0: they go up to VRAM)
fn cpu_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS2_CPU").map_or(true, |v| v != "0"))
}

/// The CPU's workers (NS_DS2_CPU_THREADS, default 16: 6.47 tok/s against 24's 6.33)
fn cpu_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_CPU_THREADS", 16).clamp(1, 256))
}

/// Pinned experts the CPU multiplied that a layer copies up to VRAM behind it, at most (NS_DS2_PROMOTE, default 3)
fn promote_limit() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_PROMOTE", 3))
}

/// How often the next layer's router, run on this layer's input, has its k-th best expert among the ones the next
/// layer asks for (GLM-5.3's measurement, reference/expert-cache; a starting point for this model)
const GUESS_HIT: [f32; 16] = [0.96, 0.92, 0.86, 0.77, 0.66, 0.56, 0.45, 0.36, 0.29, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08];

/// The routing's groups (NS_DS2_GROUPS=G,K: the experts in G groups, the best K groups' only - DeepSeek-V3's
/// n_group 8, topk_group 4, the default; 0 = every expert, as llama.cpp runs a file without the group metadata)
fn groups() -> Option<(usize, usize)> {
    static V: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("NS_DS2_GROUPS").unwrap_or_else(|_| "8,4".into());
        let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        match p.as_slice() {
            [g, k] if *g > 1 && *k >= 1 && k < g => Some((*g, *k)),
            _ => None,
        }
    })
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

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

use crate::cpu::f16_bits;

/// A matrix on the GPU: `rows` x `cols`, in its stored form, or as float32 (the F32 / F16 / BF16 ones)
struct Mat {
    buf: DevBuf,
    ty: GType,
    rows: usize,
    cols: usize,
    f32: bool,
}

impl Mat {
    fn row_bytes(&self) -> usize {
        self.ty.bytes(self.cols as u64).unwrap_or(0) as usize
    }
}

/// One GPU's share of the model.
pub struct Part {
    ops: Ops,
    arena: Arena,
    /// the arenas' memory: `small` for decode, `big` for prompt passes (lent to the expert store otherwise)
    small: DevBuf,
    big: DevBuf,
    pub layers: Range<u64>,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    /// per layer: k_b [heads][kv_lora][nope] and v_b [heads][value][kv_lora], fp16
    kb: BTreeMap<u64, DevBuf>,
    vb: BTreeMap<u64, DevBuf>,
    /// the routers' biases, on the host
    biases: BTreeMap<u64, Vec<f32>>,
    scratch: DevBuf,
    /// Q8_1 of up to MMVQ_COLS rows of MAX_COLS (the generic decode products)
    q8: DevBuf,
    /// the same rows in this engine's split Q8 layout (its own decode products: Q4_K, Q5_K, the experts')
    xs: DevBuf,
    /// pinned rings for small uploads and readbacks, copied by kernels on the compute queue (the runtime's memcpy
    /// goes to a copy engine and waits behind the expert store's copies: 0.5 ms a table at decode)
    ring: Mutex<(HostBuf, usize)>,
    back: Mutex<HostBuf>,
    experts: Mutex<Store>,
    pub expert_slots: usize,
    pub host_slots: usize,
    pub weight_bytes: u64,
}

/// A conversation: per layer the cache [max_ctx, 576] fp16 (on its layer's GPU), `pos` tokens so far, and the last
/// pass (its first position and rows: `rollback`)
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    caches: Vec<DevBuf>,
    last: Option<(usize, usize)>,
}

/// A conversation's state at a position, in host memory (`Ds::save` / `Ds::restore`)
pub struct Checkpoint {
    pub pos: usize,
    bufs: Vec<Vec<u8>>,
    pub bytes: usize,
}

impl Checkpoint {
    /// "NSDS1", the position, the buffers
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(b"NSDS1")?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&(self.bufs.len() as u64).to_le_bytes())?;
        for b in &self.bufs {
            w.write_all(&(b.len() as u64).to_le_bytes())?;
            w.write_all(b)?;
        }
        Ok(())
    }
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Checkpoint> {
        let u = |r: &mut dyn std::io::Read| -> std::io::Result<usize> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b) as usize)
        };
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)?;
        if &magic != b"NSDS1" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a deepseek2 checkpoint"));
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

/// Generation from a fed prompt: `Ds::step` commits the next token(s) each call; the last committed one is fed at the
/// start of the following call
pub struct Decoder {
    logits: Vec<f32>,
    next: Option<u32>,
    /// the conversation's tokens and where each 3-gram last continued (prompt lookup, NS_DS2_NGRAM)
    look: Option<Lookup>,
    pub drafted: u64,
    pub accepted: u64,
}

impl Decoder {
    pub fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
    /// The conversation so far: drafts from its repeats (NS_DS2_NGRAM)
    pub fn set_context(&mut self, tokens: &[u32]) {
        if ngram_k() == 0 {
            return;
        }
        let mut l = Lookup::default();
        for &t in tokens {
            l.push(t);
        }
        self.look = Some(l);
    }
}

/// A conversation's tokens with a 3-gram index: `after[(a, b, c)]` is the position after that 3-gram's latest
/// earlier occurrence
#[derive(Default)]
struct Lookup {
    hist: Vec<u32>,
    after: std::collections::HashMap<(u32, u32, u32), usize>,
}

impl Lookup {
    fn push(&mut self, t: u32) {
        self.hist.push(t);
        let n = self.hist.len();
        if n >= 4 {
            self.after.insert((self.hist[n - 4], self.hist[n - 3], self.hist[n - 2]), n - 1);
        }
    }
    /// Up to k tokens that followed the last 3 where they occurred before
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

/// Prompt-lookup drafts (NS_DS2_NGRAM=K, 0..7; default 0): when the last 3 tokens occurred earlier in the conversation,
/// the up to K tokens that followed them there are verified with the next token in one pass (the model has no draft
/// block); the accepted prefix is kept, the cache rolled back past the rest
fn ngram_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_usize("NS_DS2_NGRAM", 0).min(MMVQ_COLS - 1))
}

pub struct Ds<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    rope: Rope,
    kq_scale: f32,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// NS_PROFILE=1: seconds and calls per section (the GPU synced at each boundary)
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
    /// host time waiting for the routers' logits, and choosing the experts (ns)
    pub router_wait: std::sync::atomic::AtomicU64,
    /// rows of the pass running now (NS_PROFILE=1 times one-row passes only: decode, not the prompt)
    rows_now: std::sync::atomic::AtomicUsize,
    /// the CPU's workers for the pinned experts' products (NS_DS2_CPU=0: none - those experts go up to VRAM)
    pool: Option<crate::cpu::Pool>,
    pub route_host: std::sync::atomic::AtomicU64,
}

/// Vectors (norm weights, biases): float32 on the GPU
const VECTORS: [Role; 6] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::QANorm, Role::KvANorm, Role::RouterBias];

impl Part {
    /// The part's layers onto its GPU (and the head when `last`), its expert store: `expert_bytes` of VRAM (None: what
    /// is free less the rest), `mirror` bytes of pinned host memory.
    #[allow(clippy::too_many_arguments)]
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, last: bool, expert_bytes: Option<usize>, mirror: usize, kv_reserve: usize, reader: &Arc<Reader>,
            log: &mut dyn FnMut(String)) -> Result<Part> {
        let g = &m.g;
        let ops = Ops::new(gpu.clone())?;
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let (mut kb, mut vb) = (BTreeMap::new(), BTreeMap::new());
        let mut biases = BTreeMap::new();
        let mut bytes = 0u64;
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            todo.push((0, vec![Role::OutputNorm, Role::Output]));
        }
        todo.extend(layers.clone().map(|l| (l, m.roles(l))));
        let (nh, nope, vl, lat) = (g.n_head as usize, g.nope() as usize, g.value_len as usize, g.kv_lora as usize);
        for (l, roles) in todo {
            for r in roles {
                if matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown | Role::TokenEmbd) {
                    continue; // experts through the store; the embedding's rows are read per token
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let raw = DevBuf::new(gpu, t.bytes as usize)?;
                raw.write(0, &m.file.read(t).map_err(e)?)?;
                let n = t.elements() as usize;
                let cols = *t.shape.last().unwrap_or(&1) as usize;
                if matches!(r, Role::KvB | Role::KB | Role::VB) {
                    // k_b and v_b as fp16, laid out for the batched products: k_b[h] [kv_lora][nope], v_b[h] [value][kv_lora]
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    let v = f.to_f32()?;
                    let mut k16 = Vec::new();
                    let mut v16 = Vec::new();
                    match r {
                        Role::KvB => {
                            // rows h * (nope + value) + i: the head's k part (i < nope), then its v part; lat columns
                            let per = nope + vl;
                            k16 = vec![0u16; nh * lat * nope];
                            v16 = vec![0u16; nh * vl * lat];
                            for h in 0..nh {
                                for i in 0..nope {
                                    let row = &v[(h * per + i) * lat..(h * per + i + 1) * lat];
                                    for (c, x) in row.iter().enumerate() {
                                        k16[(h * lat + c) * nope + i] = f16_bits(*x);
                                    }
                                }
                                for i in 0..vl {
                                    let row = &v[(h * per + nope + i) * lat..(h * per + nope + i + 1) * lat];
                                    for (c, x) in row.iter().enumerate() {
                                        v16[(h * vl + i) * lat + c] = f16_bits(*x);
                                    }
                                }
                            }
                        }
                        Role::KB => k16 = v.iter().map(|x| f16_bits(*x)).collect(),
                        _ => v16 = v.iter().map(|x| f16_bits(*x)).collect(),
                    }
                    let up = |h: &[u16]| -> Result<DevBuf> {
                        let b = DevBuf::new(gpu, h.len() * 2)?;
                        b.write(0, &h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                        Ok(b)
                    };
                    if !k16.is_empty() {
                        bytes += (k16.len() * 2) as u64;
                        kb.insert(l, up(&k16)?);
                    }
                    if !v16.is_empty() {
                        bytes += (v16.len() * 2) as u64;
                        vb.insert(l, up(&v16)?);
                    }
                    continue;
                }
                bytes += t.bytes;
                if VECTORS.contains(&r) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    if r == Role::RouterBias {
                        biases.insert(l, f.to_f32()?);
                    }
                    vecs.insert((l, r), f);
                } else if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    mats.insert((l, r), Mat { buf: f, ty: GType::F32, rows: n / cols, cols, f32: true });
                } else {
                    mats.insert((l, r), Mat { buf: raw, ty: t.ty, rows: n / cols, cols, f32: false });
                }
            }
        }
        gpu.sync()?;
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(MAX_COLS, MMVQ_COLS))?;
        let xs = DevBuf::new(gpu, ops.q8_bytes(MAX_COLS, MMVQ_COLS))?;
        let small = DevBuf::new(gpu, SMALL_ARENA)?;
        let big = DevBuf::new(gpu, arena_bytes())?;
        let arena = Arena::on(small.view(0, small.len)?);
        // the store: slots of the largest layer's [gate | up | down], in chunks; VRAM, then pinned host memory
        let moe: Vec<u64> = layers.clone().filter(|l| g.is_moe(*l)).collect();
        let n_exp = moe.len() * g.n_expert as usize;
        let parts: std::collections::HashMap<u64, ExpertParts> = moe.iter().map(|l| Ok((*l, ExpertParts::of(m, *l)?))).collect::<Result<_>>()?;
        // padded: the decode kernels' 16-byte loads may read a few bytes past a matrix's last block
        let slot = (parts.values().map(|p| p.bytes()).max().unwrap_or(256) + 256).next_multiple_of(4096);
        let margin = 1536usize << 20;
        let budget = match expert_bytes {
            Some(b) => b,
            None => gpu.memory()?.1.map_or(8usize << 30, |f| (f as usize).saturating_sub(kv_reserve + margin)),
        };
        let per_chunk = (CHUNK / slot).max(1);
        let nv = if n_exp == 0 { 0 } else { (budget / slot).min(n_exp) };
        // at least 64 host slots when anything stays on the file: the reads from it land in them
        let nr = if n_exp > nv { (mirror / slot).max(64).min(n_exp - nv) } else { 0 };
        let mut vram = Vec::new();
        let mut have = 0;
        while have < nv {
            let k = per_chunk.min(nv - have);
            vram.push(DevBuf::new(gpu, k * slot)?);
            have += k;
        }
        let mut host = Vec::new();
        let mut have = 0;
        while have < nr {
            let k = per_chunk.min(nr - have);
            host.push(HostBuf::new(gpu, k * slot)?);
            have += k;
        }
        let mut store = Store::new(vram, host, slot, per_chunk, nv, nr, parts.clone(), reader.clone());
        // NS_DS2_LEND=0: the prompt arena's memory is not lent to the experts during decode
        let lent = if n_exp > 0 && std::env::var("NS_DS2_LEND").map_or(true, |v| v != "0") { store.add_lend(big.view(0, big.len)?) } else { 0 };
        // what goes where at load: every layer's experts in turn (expert 0 of each layer, then 1, ...), VRAM first, then
        // pinned memory; the rest stays on the file. NS_EXPERT_PROFILE (earlier runs' requests) puts the most asked first.
        let mut all: Vec<(u64, u64)> = (0..g.n_expert).flat_map(|x| moe.iter().map(move |l| (*l, x))).collect();
        if let Some(prof) = std::env::var("NS_EXPERT_PROFILE").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
            let mut w: std::collections::HashMap<(u64, u64), f64> = std::collections::HashMap::new();
            for line in prof.lines().filter(|l| !l.starts_with('#')) {
                let f: Vec<&str> = line.split_whitespace().collect();
                if let [l, x, c, ..] = f.as_slice() {
                    if let (Ok(l), Ok(x), Ok(c)) = (l.parse(), x.parse(), c.parse()) {
                        w.insert((l, x), c);
                    }
                }
            }
            all.sort_by(|a, b| w.get(b).copied().unwrap_or(0.0).total_cmp(&w.get(a).copied().unwrap_or(0.0)));
        }
        let t0 = Instant::now();
        // host slots straight from the file; VRAM slots through a pinned staging area, a batch at a time
        for (r, key) in all.iter().skip(nv).take(nr).enumerate() {
            store.place_r(*key, r);
        }
        {
            let mut jobs = Vec::new();
            for r in 0..nr.min(all.len().saturating_sub(nv)) {
                let key = all[nv + r];
                let p = parts[&key.0];
                let dst = store.host_ptr(r);
                // SAFETY: the slot's slot_bytes at dst, written by the reads alone until they are waited for below
                let buf = unsafe { std::slice::from_raw_parts_mut(dst, p.bytes()) };
                jobs.push((p, key.1, buf));
            }
            for chunk in jobs.chunks_mut(256) {
                std::thread::scope(|sc| -> Result<()> {
                    let hs: Vec<_> = chunk.iter_mut().map(|(p, x, buf)| {
                        let (p, x) = (*p, *x);
                        let reader = reader.clone();
                        let buf: &mut [u8] = buf;
                        sc.spawn(move || reader.read_now(&p, x, buf))
                    }).collect();
                    for h in hs {
                        h.join().map_err(|_| Error("a load thread panicked".into()))??;
                    }
                    Ok(())
                })?;
            }
        }
        if nv > 0 {
            const BATCH: usize = 64;
            let mut staging = HostBuf::new(gpu, BATCH * slot)?;
            for (b, keys) in all[..nv].chunks(BATCH).enumerate() {
                {
                    let base = staging.as_mut_slice();
                    std::thread::scope(|sc| -> Result<()> {
                        let hs: Vec<_> = base.chunks_mut(slot).zip(keys).map(|(buf, key)| {
                            let p = parts[&key.0];
                            let reader = reader.clone();
                            let x = key.1;
                            sc.spawn(move || reader.read_now(&p, x, buf))
                        }).collect();
                        for h in hs {
                            h.join().map_err(|_| Error("a load thread panicked".into()))??;
                        }
                        Ok(())
                    })?;
                }
                let mut last = None;
                for (i, key) in keys.iter().enumerate() {
                    let s = b * BATCH + i;
                    let src = staging.device_view(i * slot, slot)?;
                    last = Some(ops.stream_copy_on(0, &store.vview(s)?, 0, &src, 0, slot, &[])?);
                    store.place_v(*key, s);
                }
                // the copy lane is in order: its last copy done, the staging area is free again
                if let Some(t) = last {
                    ops.wait_ticket(t)?;
                }
            }
        }
        let on_file = n_exp - nv.min(n_exp) - nr;
        log(format!("{} (layers {}-{}): {:.2} GiB of weights; experts {} in VRAM ({:.1} GiB, and {} slots in the prompt arena while decode runs), {} in pinned host memory ({:.1} GiB), {} on the file only ({:.1} s)",
                    gpu.name, layers.start, layers.end - 1, gib(bytes as usize), nv, gib(nv * slot), lent, nr, gib(nr * slot), on_file, t0.elapsed().as_secs_f64()));
        let ring = Mutex::new((HostBuf::new(gpu, RING_BYTES)?, 0));
        let back = Mutex::new(HostBuf::new(gpu, BACK_BYTES)?);
        Ok(Part { ops, arena, small, big, layers, mats, vecs, kb, vb, biases, scratch, q8, xs, ring, back, experts: Mutex::new(store), expert_slots: nv, host_slots: nr,
                  weight_bytes: bytes })
    }

    /// The arena a pass of `t` tokens needs: the small one for decode widths; past them the big one, its memory taken
    /// back from the expert store first (lent again once a pass fits the small one)
    fn arena_for(&self, t: usize) -> Result<()> {
        let mut st = self.experts.lock().unwrap();
        if t > MMVQ_COLS {
            if st.lent() {
                // what is queued may read the lent slots: done first
                self.ops.gpu.sync()?;
                st.reclaim(&self.ops)?;
            }
            if self.arena.len() != self.big.len {
                self.arena.set(self.big.view(0, self.big.len)?);
            }
        } else if self.arena.len() != self.small.len {
            // the big arena's work is queued before any copy into its memory: the copies wait for the queue (a mark)
            self.ops.gpu.sync()?;
            self.arena.set(self.small.view(0, self.small.len)?);
            st.relend();
        }
        Ok(())
    }

    /// `data` on the GPU (an arena buffer): through the pinned ring, copied by a kernel in queue order
    fn upload(&self, data: &[u8]) -> Result<DevBuf> {
        let n = data.len().next_multiple_of(16).max(16);
        let dst = self.arena.bytes(n)?;
        let mut r = self.ring.lock().unwrap();
        if n > r.0.len {
            drop(r);
            dst.write(0, data)?;
            return Ok(dst);
        }
        if r.1 + n > r.0.len {
            // what the kernels read from the ring so far is done once the queue is
            self.ops.gpu.sync()?;
            r.1 = 0;
        }
        let off = r.1;
        r.0.as_mut_slice()[off..off + data.len()].copy_from_slice(data);
        r.1 = (off + n).next_multiple_of(64);
        let src = r.0.device_view(off, n)?;
        self.ops.kcopy(&dst, 0, &src, 0, n, None)?;
        Ok(dst)
    }
    fn upload_i32(&self, v: &[i32]) -> Result<DevBuf> {
        self.upload(&i32_bytes(v))
    }
    fn upload_f32(&self, v: &[f32]) -> Result<DevBuf> {
        self.upload(&f32_bytes(v))
    }
    /// Several buffers' first floats read back with one wait (kernels copy them into pinned memory)
    fn read_many(&self, xs: &[(&DevBuf, usize)]) -> Result<Vec<Vec<f32>>> {
        let back = self.back.lock().unwrap();
        let total: usize = xs.iter().map(|(_, n)| (n * 4).next_multiple_of(64)).sum();
        if total > back.len {
            drop(back);
            return xs.iter().map(|(x, n)| self.read_f32(x, *n)).collect();
        }
        let mut at = 0;
        let mut spans = Vec::new();
        for (x, n) in xs {
            let b = (n * 4).next_multiple_of(16);
            self.ops.kcopy(&back.device_view(at, b)?, 0, x, 0, b.min(x.len), None)?;
            spans.push((at, n * 4));
            at += (n * 4).next_multiple_of(64);
        }
        self.ops.gpu.sync()?;
        Ok(spans.iter().map(|(a, b)| back.as_slice()[*a..a + b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).collect())
    }
    /// `data` into dst from byte `at` (through the pinned ring, copied by a kernel in queue order)
    fn upload_into(&self, dst: &DevBuf, at: usize, data: &[u8]) -> Result<()> {
        let src = self.upload(data)?;
        self.ops.kcopy(dst, at, &src, 0, data.len(), None)?;
        Ok(())
    }

    /// The first n floats of x read back (a kernel copies them into pinned memory; the queue waited for)
    fn read_f32(&self, x: &DevBuf, n: usize) -> Result<Vec<f32>> {
        let b = n * 4;
        let back = self.back.lock().unwrap();
        if b.next_multiple_of(16) > back.len {
            drop(back);
            return Ok(x.to_f32()?[..n].to_vec());
        }
        let dst = back.device_view(0, b.next_multiple_of(16))?;
        self.ops.kcopy(&dst, 0, x, 0, b.next_multiple_of(16).min(x.len), None)?;
        self.ops.gpu.sync()?;
        Ok(back.as_slice()[..b].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn mat(&self, l: u64, r: Role) -> Result<&Mat> {
        self.mats.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: matrix {r:?} not on this GPU")))
    }
    fn vec(&self, l: u64, r: Role) -> Result<&DevBuf> {
        self.vecs.get(&(l, r)).ok_or_else(|| Error(format!("block {l}: vector {r:?} not on this GPU")))
    }

    /// y [t rows from y.1, y.2 apart] = x . W^T. Float matrices multiply as they are; quantized ones from their
    /// blocks for up to MMVQ_COLS contiguous rows (decode), else expanded (fp16 from f16_min rows) in row chunks.
    fn matmul(&self, w: &Mat, t: usize, x: (&DevBuf, usize, usize), y: (&DevBuf, usize, usize)) -> Result<()> {
        let o = &self.ops;
        if w.f32 {
            return o.gemm_at(t, w.rows, w.cols, x, (&w.buf, 0), y, false);
        }
        if t <= MMVQ_COLS && x.2 == w.cols && y.2 == w.rows && w.cols <= MAX_COLS && own_k(o, w) {
            o.quantize_q8((x.0, x.1), &self.xs, w.cols, t)?;
            return o.own_mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.xs, (y.0, y.1), w.cols, w.rows, t);
        }
        if t <= MMVQ_COLS && x.2 == w.cols && y.2 == w.rows && w.cols <= MAX_COLS && o.mmvq_supported(w.ty.code()) {
            o.quantize_q8_1((x.0, x.1), &self.q8, w.cols, t)?;
            return o.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (y.0, y.1), w.cols, w.rows, t);
        }
        let rb = w.row_bytes();
        if t >= f16_min() && x.2 == w.cols && w.cols.is_multiple_of(256) && o.f16_type(w.ty.code()) {
            // the activations to fp16 once, the matrix expanded to fp16 a row block at a time, oneMKL's half GEMM
            let mark = self.arena.mark();
            let x16 = self.arena.bytes(t * w.cols * 2)?;
            o.to_f16_at(x.0, x.1, &x16, 0, t * w.cols)?;
            let chunk = (SCRATCH * 2 / w.cols).max(1).min(w.rows);
            let mut r0 = 0;
            while r0 < w.rows {
                let r = chunk.min(w.rows - r0);
                o.dequant_f16(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
                o.gemm_f16(t, r, w.cols, (&x16, 0, w.cols), (&self.scratch, 0), (y.0, y.1 + r0, y.2), false)?;
                r0 += r;
            }
            self.arena.rewind(mark);
            return Ok(());
        }
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let mut r0 = 0;
        while r0 < w.rows {
            let r = chunk.min(w.rows - r0);
            o.dequant(w.ty.code(), &w.buf, r0 * rb, r * rb, r * w.cols, &self.scratch)?;
            o.gemm_at(t, r, w.cols, x, (&self.scratch, 0), (y.0, y.1 + r0, y.2), false)?;
            r0 += r;
        }
        Ok(())
    }

    /// y [t, rows] = x [t, cols] . W^T (contiguous)
    fn mm(&self, l: u64, r: Role, x: &DevBuf, t: usize) -> Result<DevBuf> {
        let w = self.mat(l, r)?;
        let y = self.arena.f32(t * w.rows)?;
        self.matmul(w, t, (x, 0, w.cols), (&y, 0, w.rows))?;
        Ok(y)
    }

    /// Several matrices on the same input: one Q8_1 quantization of x serves every decode product
    fn mm_many(&self, l: u64, roles: &[Role], x: &DevBuf, t: usize) -> Result<Vec<DevBuf>> {
        let ws: Vec<&Mat> = roles.iter().map(|r| self.mat(l, *r)).collect::<Result<_>>()?;
        let o = &self.ops;
        let cols = ws[0].cols;
        if t <= MMVQ_COLS && ws.iter().all(|w| w.cols == cols && !w.f32 && (own_k(o, w) || o.mmvq_supported(w.ty.code()))) && cols <= MAX_COLS {
            // each layout's quantization once (this engine's kernels for Q4_K / Q5_K, the shared ones for the rest)
            if ws.iter().any(|w| own_k(o, w)) {
                o.quantize_q8((x, 0), &self.xs, cols, t)?;
            }
            if ws.iter().any(|w| !own_k(o, w)) {
                o.quantize_q8_1((x, 0), &self.q8, cols, t)?;
            }
            return ws.iter().map(|w| {
                let y = self.arena.f32(t * w.rows)?;
                if own_k(o, w) {
                    o.own_mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.xs, (&y, 0), w.cols, w.rows, t)?;
                } else {
                    o.mmvq(w.ty.code(), (&w.buf, 0), w.buf.len, &self.q8, (&y, 0), w.cols, w.rows, t)?;
                }
                Ok(y)
            }).collect();
        }
        roles.iter().map(|r| self.mm(l, *r, x, t)).collect()
    }
}

impl<'g> Ds<'g> {
    /// Loads the model over `gpus` (layers split evenly by count, or as NS_SPLIT says; the head on the last). Per GPU,
    /// `expert_bytes` of VRAM for routed experts (None: what is free less the rest); `mirror_bytes` of pinned host
    /// memory in all (None: what the host has available less 8 GiB: 50 GB pinned on a 61 GB host read 13% fewer experts from the file than 46), shared by the parts by their MoE layers. `kv`:
    /// (context, sessions) - their caches are held back from the expert store.
    pub fn load(file: &'g Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, kv: (usize, usize),
                log: &mut dyn FnMut(String)) -> Result<Ds<'g>> {
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
        let reader = Reader::new(file)?;
        let mirror = mirror_bytes.unwrap_or(mem_available().saturating_sub(8 << 30));
        let mut bounds: Vec<u64> = (0..=k).map(|i| i * n / k).collect();
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
        let moe_total = (g.n_dense..n).count().max(1);
        let mut owner = vec![0usize; n as usize];
        let mut parts = Vec::new();
        for (i, gpu) in gpus.iter().enumerate() {
            let range = bounds[i]..bounds[i + 1];
            for l in range.clone() {
                owner[l as usize] = i;
            }
            let moe = range.clone().filter(|l| g.is_moe(*l)).count();
            let share = mirror * moe / moe_total;
            let reserve = kv.1 * kv.0 * (range.end - range.start) as usize * g.cache_row() as usize * 2;
            if reserve > 0 {
                log(format!("{}: {:.2} GiB held for the sessions' {} tokens", gpu.name, gib(reserve), kv.0 * kv.1));
            }
            parts.push(Part::load(&m, gpu, range, i + 1 == gpus.len(), expert_bytes, share, reserve, &reader, log)?);
        }
        let (freq_scale, ext_factor, mscale, c0, c1, kq_scale) = m.yarn();
        let rope = Rope { base: g.rope_base as f32, freq_scale, ext_factor, mscale, corr: (c0, c1) };
        log(format!("YaRN: freq scale {freq_scale}, ext {ext_factor}, cos/sin x {mscale}, corr dims {c0}-{c1}; scores x {kq_scale}; routing groups {:?}", groups()));
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let pool = cpu_on().then(|| crate::cpu::Pool::new(cpu_threads()));
        if let Some(pl) = &pool {
            log(format!("CPU: the pinned experts multiplied by {} threads{}", pl.threads, if crate::cpu::avx512_on() { " (AVX-512 VNNI)" } else { "" }));
        }
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(BTreeMap::new()));
        Ok(Ds { m, parts, owner, rope, kq_scale, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, prof, router_wait: 0.into(), rows_now: 0.into(), route_host: 0.into(), pool })
    }

    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.parts.iter().map(|p| {
            let (total, free) = p.ops.gpu.memory().unwrap_or((0, None));
            GpuInfo { index: p.ops.gpu.index, name: p.ops.gpu.name.clone(), pci: p.ops.gpu.pci.clone(), total, free, layers: (p.layers.start, p.layers.end),
                      expert_slots: p.expert_slots, host_slots: p.host_slots }
        }).collect()
    }

    /// The parts' expert stores' counts, summed, and the bytes read from the file
    pub fn expert_stats(&self) -> (Stats, usize) {
        let mut s = Stats::default();
        let mut file = 0;
        for p in &self.parts {
            let c = p.experts.lock().unwrap();
            let x = c.stats;
            s.hits += x.hits;
            s.from_host += x.from_host;
            s.from_file += x.from_file;
            s.direct += x.direct;
            s.prefetched += x.prefetched;
            s.prefetch_used += x.prefetch_used;
            s.file_prefetched += x.file_prefetched;
            s.on_cpu += x.on_cpu;
            s.promoted += x.promoted;
            file = c.bytes_from_file();
        }
        (s, file)
    }

    /// Profiling inside a section: the GPU synced and the time since `t0` added to `name`; the next lap's start
    fn lap(&self, p: &Part, name: &'static str, t0: Option<Instant>) -> Option<Instant> {
        let pr = self.prof.as_ref().filter(|_| self.rows_now.load(std::sync::atomic::Ordering::Relaxed) == 1)?;
        let t0 = t0?;
        p.ops.gpu.sync().ok()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Some(Instant::now())
    }
    fn mark(&self, p: &Part) -> Option<Instant> {
        self.prof.as_ref().filter(|_| self.rows_now.load(std::sync::atomic::Ordering::Relaxed) == 1)?;
        p.ops.gpu.sync().ok()?;
        Some(Instant::now())
    }

    /// Runs `f`, adding its time to section `name` when profiling (`p`'s GPU synced around it)
    fn timed<T>(&self, p: &Part, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(pr) = self.prof.as_ref().filter(|_| self.rows_now.load(std::sync::atomic::Ordering::Relaxed) == 1) else { return f() };
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

    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// A new conversation of up to `max_ctx` tokens (its caches zeroed: the attention's padded cells read them)
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let row = self.m.g.cache_row() as usize * 2;
        let mut caches = Vec::new();
        for l in 0..self.m.g.n_layer {
            let gpu = &self.parts[self.owner[l as usize]].ops.gpu;
            let c = DevBuf::new(gpu, max_ctx.max(1) * row)?;
            c.fill(0)?;
            caches.push(c);
        }
        Ok(Session { pos: 0, max_ctx, caches, last: None })
    }

    pub fn reset_session(&self, s: &mut Session) -> Result<()> {
        s.pos = 0;
        s.last = None;
        Ok(())
    }

    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if dst.max_ctx != src.max_ctx || dst.caches.len() != src.caches.len() {
            return Err(Error("copy_session: the sessions differ in shape".into()));
        }
        let row = self.m.g.cache_row() as usize * 2;
        if src.pos > 0 {
            for (d, s) in dst.caches.iter().zip(&src.caches) {
                d.copy_within(0, s, 0, src.pos * row)?;
            }
        }
        dst.pos = src.pos;
        dst.last = None;
        Ok(())
    }

    pub fn save(&self, s: &Session) -> Result<Checkpoint> {
        let row = self.m.g.cache_row() as usize * 2;
        let mut bufs = Vec::new();
        for c in &s.caches {
            let mut v = vec![0u8; s.pos * row];
            if !v.is_empty() {
                c.read(0, &mut v)?;
            }
            bufs.push(v);
        }
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: s.pos, bufs, bytes })
    }

    pub fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        if ck.pos + 1 > s.max_ctx || ck.bufs.len() != s.caches.len() {
            return Err(Error(format!("restore: a checkpoint of {} tokens into a session of {}", ck.pos, s.max_ctx)));
        }
        for (c, b) in s.caches.iter().zip(&ck.bufs) {
            if !b.is_empty() {
                c.write(0, b)?;
            }
        }
        s.pos = ck.pos;
        s.last = None;
        Ok(())
    }

    /// Keeps the first `keep` rows of the last pass (the cache's later rows are written over by what comes next)
    pub fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let (p0, t) = s.last.ok_or_else(|| Error("rollback: no pass to roll back".into()))?;
        if keep > t {
            return Err(Error(format!("rollback: keep {keep} of a {t}-row pass")));
        }
        s.pos = p0 + keep;
        s.last = None;
        Ok(())
    }

    pub fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(s, tokens, &|| false, tap)?.1)
    }

    pub fn feed_until(&self, s: &mut Session, tokens: &[u32], stop: &(dyn Fn() -> bool + Sync), tap: Tap) -> Result<(usize, Vec<f32>)> {
        let (mut fed, mut logits) = (0, Vec::new());
        for (i, c) in tokens.chunks(prefill_chunk()).enumerate() {
            if i > 0 && stop() {
                break;
            }
            logits = self.forward(s, c, &mut *tap)?;
            fed += c.len();
        }
        Ok((fed, logits))
    }

    pub fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.forward_rows(s, tokens, 1, tap)?.pop().unwrap_or_default())
    }

    /// The next `tokens` (at most a chunk): the logits of the last `n_out`
    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let t = tokens.len();
        if t == 0 || t > prefill_chunk() || n_out == 0 || n_out > t.min(MMVQ_COLS) {
            return Err(Error(format!("forward: {t} tokens, {n_out} outputs")));
        }
        let pos0 = s.pos;
        if pos0 + t > s.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", s.max_ctx, t)));
        }
        let d = self.m.g.n_embd as usize;
        self.rows_now.store(t, std::sync::atomic::Ordering::Relaxed);
        for p in &self.parts {
            p.arena_for(t)?;
        }
        let mut x = self.embed(&self.parts[0], tokens)?;
        tap("inp_embd", &x)?;
        let mut staging = Vec::new();
        for (k, run) in self.runs().into_iter().enumerate() {
            if k > 0 {
                let p = &self.parts[run.0];
                let next = DevBuf::new(&p.ops.gpu, x.len)?;
                staging.resize(staging.len().max(x.len), 0);
                self.timed(p, "GPU to GPU", || next.copy_from_peer(&x, &mut staging))?;
                x = next;
            }
            self.run_layers(&s.caches, run, &x, pos0, t, &mut *tap)?;
        }
        let out = self.head(&x, t, n_out, d, &mut *tap)?;
        s.pos += t;
        s.last = Some((pos0, t));
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

    /// Token embeddings [t, n_embd] on `p`'s GPU: the rows read from the file, expanded (an owned buffer: the stream)
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
        p.arena.reset();
        let raw = p.upload(&rows)?;
        let x = DevBuf::f32(&p.ops.gpu, tokens.len() * g.n_embd as usize)?;
        p.ops.dequant(emb.ty.code(), &raw, 0, rows.len(), tokens.len() * g.n_embd as usize, &x)?;
        Ok(x)
    }

    /// One run of layers over the stream x [t, d] in place
    fn run_layers(&self, caches: &[DevBuf], run: (usize, u64, u64), x: &DevBuf, pos0: usize, t: usize, tap: Tap) -> Result<()> {
        let g = &self.m.g;
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let p = &self.parts[run.0];
        let o = &p.ops;
        for l in run.1..run.2 {
            p.arena.reset();
            let h = p.arena.f32(t * d)?;
            o.rms_norm(x, Some(p.vec(l, Role::AttnNorm)?), &h, t, d, eps)?;
            tap(&format!("attn_norm-{l}"), &h)?;
            let att = self.timed(p, "MLA", || self.mla(p, l, &h, &caches[l as usize], pos0, t, &mut *tap))?;
            o.add(x, &att, t * d)?;
            tap(&format!("ffn_inp-{l}"), x)?;
            o.rms_norm(x, Some(p.vec(l, Role::FfnNorm)?), &h, t, d, eps)?;
            tap(&format!("ffn_norm-{l}"), &h)?;
            let f = if g.is_moe(l) {
                self.moe(p, l, t, &h, &mut *tap)?
            } else {
                self.timed(p, "dense FFN", || {
                    let mut pj = p.mm_many(l, &[Role::FfnGate, Role::FfnUp], &h, t)?.into_iter();
                    let (gt, up) = (pj.next().unwrap(), pj.next().unwrap());
                    o.swiglu(&gt, &up, &gt, t * g.ffn_dense as usize)?;
                    p.mm(l, Role::FfnDown, &gt, t)
                })?
            };
            tap(&format!("ffn_out-{l}"), &f)?;
            o.add(x, &f, t * d)?;
            tap(&format!("l_out-{l}"), x)?;
        }
        Ok(())
    }

    /// MLA of layer `l` on h [t, d] (normed) at positions pos0..: this pass's cache rows written, every earlier one
    /// attended to; the output [t, d]
    #[allow(clippy::too_many_arguments)]
    fn mla(&self, p: &Part, l: u64, h: &DevBuf, cache: &DevBuf, pos0: usize, t: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let eps = g.rms_eps as f32;
        let (nh, kl, nope, rd, lat, vl) = (g.n_head as usize, g.key_len as usize, g.nope() as usize, g.rope_dim as usize, g.kv_lora as usize, g.value_len as usize);
        let cr = lat + rd;
        let m0 = self.mark(p);
        let mut pj = p.mm_many(l, &[Role::QA, Role::KvA], h, t)?.into_iter();
        let (qa, kv) = (pj.next().unwrap(), pj.next().unwrap());
        tap(&format!("kv_cmpr_pe-{l}"), &kv)?;
        let qr = p.arena.f32(t * g.q_lora as usize)?;
        o.rms_norm(&qa, Some(p.vec(l, Role::QANorm)?), &qr, t, g.q_lora as usize, eps)?;
        let q = p.mm(l, Role::QB, &qr, t)?; // [t, heads, nope + rope]
        tap(&format!("q-{l}"), &q)?;
        let m1 = self.lap(p, "MLA 1: projections", m0);
        o.kv_store(&kv, p.vec(l, Role::KvANorm)?, eps, cache, t, lat, rd, pos0, &self.rope)?;
        // decode widths over a context the GPU's cache holds well: one launch for the rest of the attention
        if t <= MMVQ_COLS && pos0 + t <= fused_max() && (nh, kl, nope, rd, lat, vl) == (nh, 192, 128, 64, 512, 128) {
            o.rope(&q, t, nh, nh * kl, kl, nope, rd, pos0, &self.rope, None)?;
            tap(&format!("Qcur-{l}"), &q)?;
            let oh = p.arena.f32(t * nh * vl)?;
            o.mla_decode(&q, p.kb.get(&l).ok_or("k_b missing")?, p.vb.get(&l).ok_or("v_b missing")?, cache, &oh, t, nh, pos0, self.kq_scale)?;
            tap(&format!("kqv_out-{l}"), &oh)?;
            let m1 = self.lap(p, "MLA 1: projections", m0);
            let out = p.mm(l, Role::Out, &oh, t)?;
            self.lap(p, "MLA 5: output projection", m1);
            tap(&format!("attn_out-{l}"), &out)?;
            return Ok(out);
        }
        // the queries [t, heads, lat + rope]: the rotated part straight from the rotation, the absorbed one by k_b
        let qf = p.arena.f32(t * nh * cr)?;
        o.rope(&q, t, nh, nh * kl, kl, nope, rd, pos0, &self.rope, Some((&qf.view(lat * 4, (t * nh * cr - lat) * 4)?, nh * cr, cr)))?;
        tap(&format!("Qcur-{l}"), &q)?;
        let q16 = p.arena.bytes(t * nh * kl * 2)?;
        o.to_f16(&q, &q16, t * nh * kl)?;
        let kb = p.kb.get(&l).ok_or("k_b missing")?;
        // per head: q~ = k_b[h] . q_nope. oneMKL wants a batch's outputs apart, so decode widths go a row at a time (the
        // heads one batch), prompt chunks a head at a time (its rows one product)
        if t <= MMVQ_COLS {
            for r in 0..t {
                o.gemm_batch_h(nh, true, 1, lat, nope, (&q16, r * nh * kl, kl, kl), (kb, 0, nope, lat * nope), (&qf, r * nh * cr, cr, cr))?;
            }
        } else {
            for hh in 0..nh {
                o.gemm_f16(t, lat, nope, (&q16, hh * kl, nh * kl), (kb, hh * lat * nope), (&qf, hh * cr, nh * cr), false)?;
            }
        }
        let qh = p.arena.bytes(t * nh * cr * 2)?;
        o.to_f16(&qf, &qh, t * nh * cr)?;
        let m2 = self.lap(p, "MLA 2: cache row, rope, absorbed queries", m1);
        // scores over every cell so far (padded to 64 within the cache: its unwritten rows are zeros, masked)
        let n = pos0 + t;
        let nc = n.next_multiple_of(64).min(cache.len / (cr * 2));
        let u = p.arena.f32(t * nh * lat)?;
        let rb = (SCORES / (nh * nc * 4)).clamp(1, t);
        let mark = p.arena.mark();
        for r0 in (0..t).step_by(rb) {
            p.arena.rewind(mark);
            let tr = rb.min(t - r0);
            let sc = p.arena.f32(tr * nh * nc)?;
            o.gemm_f16(tr * nh, nc, cr, (&qh, r0 * nh * cr, cr), (cache, 0), (&sc, 0, nc), false)?;
            let cnt: Vec<i32> = (0..tr).map(|i| (pos0 + r0 + i + 1) as i32).collect();
            let cb = p.upload_i32(&cnt)?;
            o.softmax_masked(&sc, tr, nh, nc, &cb, self.kq_scale)?;
            let ph = p.arena.bytes(tr * nh * nc * 2)?;
            o.to_f16(&sc, &ph, tr * nh * nc)?;
            // one product (oneMKL wants a batch's strides to cover its matrices even for one)
            o.gemm_batch_h(1, false, tr * nh, lat, nc, (&ph, 0, nc, tr * nh * nc), (cache, 0, cr, nc * cr), (&u, r0 * nh * lat, lat, tr * nh * lat))?;
        }
        p.arena.rewind(mark);
        let m3 = self.lap(p, "MLA 3: attention", m2);
        let uh = p.arena.bytes(t * nh * lat * 2)?;
        o.to_f16(&u, &uh, t * nh * lat)?;
        let vb = p.vb.get(&l).ok_or("v_b missing")?;
        let oh = p.arena.f32(t * nh * vl)?;
        if t <= MMVQ_COLS {
            for r in 0..t {
                o.gemm_batch_h(nh, true, 1, vl, lat, (&uh, r * nh * lat, lat, lat), (vb, 0, lat, vl * lat), (&oh, r * nh * vl, vl, vl))?;
            }
        } else {
            for hh in 0..nh {
                o.gemm_f16(t, vl, lat, (&uh, hh * lat, nh * lat), (vb, hh * vl * lat), (&oh, hh * vl, nh * vl), false)?;
            }
        }
        tap(&format!("kqv_out-{l}"), &oh)?;
        let m4 = self.lap(p, "MLA 4: v_b", m3);
        let out = p.mm(l, Role::Out, &oh, t)?;
        self.lap(p, "MLA 5: output projection", m4);
        tap(&format!("attn_out-{l}"), &out)?;
        Ok(out)
    }

    /// Each row's experts and weights from its router logits: the sigmoid, the bias for the choice (within the best
    /// groups), the best n_expert_used, their sigmoids normalized and scaled
    fn route(&self, lv: &[f32], bias: &[f32], t: usize) -> Vec<Vec<(usize, f32)>> {
        let g = &self.m.g;
        let (ne, used) = (g.n_expert as usize, g.n_expert_used as usize);
        let pick = |ti: usize| -> Vec<(usize, f32)> {
            let pr: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
            let key: Vec<f32> = (0..ne).map(|i| pr[i] + bias.get(i).copied().unwrap_or(0.0)).collect();
            let mut cand: Vec<usize> = (0..ne).collect();
            if let Some((ng, kg)) = groups().filter(|(ng, _)| ne % ng == 0) {
                // a group's score: its best two choices' sum; the best kg groups' experts stay candidates
                let per = ne / ng;
                let mut gs: Vec<(usize, f32)> = (0..ng).map(|gi| {
                    let mut v: Vec<f32> = key[gi * per..(gi + 1) * per].to_vec();
                    v.sort_by(|a, b| b.total_cmp(a));
                    (gi, v.iter().take(2).sum())
                }).collect();
                gs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let keep: Vec<usize> = gs.iter().take(kg).map(|x| x.0).collect();
                cand.retain(|i| keep.contains(&(i / per)));
            }
            cand.sort_by(|a, b| key[*b].total_cmp(&key[*a]).then(a.cmp(b)));
            cand.truncate(used);
            let sum: f32 = cand.iter().map(|i| pr[*i]).sum::<f32>().max(6.103_516e-5);
            cand.iter().map(|&i| (i, if g.expert_norm { pr[i] / sum } else { pr[i] } * g.expert_scale as f32)).collect()
        };
        (0..t).map(pick).collect()
    }

    /// The MoE half of layer `l` on x [t, d]: the router (on the host), the routed experts, the shared expert
    fn moe(&self, p: &Part, l: u64, t: usize, x: &DevBuf, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, f, ne) = (g.n_embd as usize, g.ffn_expert as usize, g.n_expert as usize);
        let logits = self.timed(p, "router", || p.mm(l, Role::Router, x, t))?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        // decode: the next layer's router on this layer's input - its guessed experts start loading while this
        // layer's experts compute
        let pf = prefetch_limit();
        let guess = if t <= MMVQ_COLS && pf > 0 && p.mats.contains_key(&(l + 1, Role::Router)) { Some(p.mm(l + 1, Role::Router, x, t)?) } else { None };
        let parts = p.experts.lock().unwrap().parts.get(&l).copied().ok_or("no experts for this layer here")?;
        // the CPU takes this layer's pinned experts at decode widths when all its expert matrices are IQ1_S
        let hybrid = self.pool.is_some() && t <= MMVQ_COLS && [parts.gate.2, parts.up.2, parts.down.2].iter().all(|ty| ty.code() == 19);
        let tw = Instant::now();
        let mut reads: Vec<(&DevBuf, usize)> = vec![(&logits, t * ne)];
        if hybrid {
            reads.push((x, t * d));
        }
        if let Some(gl) = guess.as_ref() {
            reads.push((gl, t * ne));
        }
        let mut back = p.read_many(&reads)?.into_iter();
        let lv = back.next().unwrap_or_default();
        let xh = if hybrid { back.next().unwrap_or_default() } else { Vec::new() };
        let guess = guess.map(|_| back.next().unwrap_or_default());
        self.router_wait.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let th = Instant::now();
        let no_bias = Vec::new();
        let bias = p.biases.get(&l).unwrap_or(&no_bias);
        let rows = self.route(&lv, bias, t);
        // NS_DS2_TRACE_ROUTE=L: layer L's choices, a line a row (a debugging aid)
        if std::env::var("NS_DS2_TRACE_ROUTE").ok().and_then(|v| v.parse::<u64>().ok()) == Some(l) {
            for (ti, sel) in rows.iter().enumerate() {
                eprintln!("route {l} {ti} {}", sel.iter().map(|(i, w)| format!("{i}:{w:.4}")).collect::<Vec<_>>().join(" "));
            }
        }
        // expert -> its (token, weight) entries
        let mut by: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
        for (ti, sel) in rows.iter().enumerate() {
            for &(i, w) in sel {
                by.entry(i).or_default().push((ti as i32, w));
            }
        }
        self.route_host.fetch_add(th.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        let y = p.arena.f32(t * d)?;
        o.fill(&y, 0.0, t * d)?;
        // the shared expert first: it runs while the routed experts' copies and reads are on their way
        let sh = self.timed(p, "shared expert", || {
            let mut pj = p.mm_many(l, &[Role::ShGate, Role::ShUp], x, t)?.into_iter();
            let (sg, su) = (pj.next().unwrap(), pj.next().unwrap());
            o.swiglu(&sg, &su, &sg, t * f * g.n_expert_shared as usize)?;
            p.mm(l, Role::ShDown, &sg, t)
        })?;
        if hybrid {
            self.moe_hybrid(p, l, t, x, &xh, &by, &parts, &y, &mut *tap)?;
        } else if t <= multi_max() {
            // every entry (token, expert) through the decode kernels from VRAM: gate | up for all, SwiGLU, down for all.
            // The experts already in VRAM or coming from pinned memory first; the ones read from the file while those
            // compute, then theirs
            let (first, fresh, arriving) = self.timed(p, "expert loads", || p.experts.lock().unwrap().ensure(o, l, &need))?;
            if parts.up.0 != parts.gate.0 + parts.gate.1 || parts.gate.2 != parts.up.2 {
                return Err(Error("expert slots: gate and up are not one matrix".into()));
            }
            let lists: Vec<(&usize, &Vec<(i32, f32)>)> = by.iter().collect();
            let n: usize = lists.iter().map(|(_, v)| v.len()).sum();
            let xs = p.arena.bytes(o.q8_bytes(d, t))?;
            o.quantize_q8((x, 0), &xs, d, t)?;
            let gu = p.arena.f32(n * 2 * f)?;
            let hh = p.arena.f32(n * f)?;
            let dn = p.arena.f32(n * d)?;
            // per token its (entry, weight) pairs: the combine reads the weights in this order
            let mut per_token: Vec<Vec<(i32, f32)>> = vec![Vec::new(); t];
            let mut e0 = 0usize;
            // (token, expert) -> its entry
            let mut entry_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
            // one group of experts (`ks`: indices into lists, `slots` theirs) through the kernels: entries e0..
            let group = |ks: &[usize], slots: &[usize], ticket: Option<i64>, e0: &mut usize, per_token: &mut Vec<Vec<(i32, f32)>>,
                         entry_of: &mut BTreeMap<(usize, usize), usize>| -> Result<()> {
                if ks.is_empty() {
                    return Ok(());
                }
                let g0 = self.mark(p);
                let mut ptrs: Vec<u64> = Vec::new();
                let mut cols: Vec<i32> = Vec::new();
                {
                    let store = p.experts.lock().unwrap();
                    for (k, s) in ks.iter().zip(slots) {
                        let base = store.vview(*s)?.ptr() as u64;
                        for (ti, w) in lists[*k].1 {
                            entry_of.insert((*ti as usize, *lists[*k].0), *e0 + ptrs.len());
                            per_token[*ti as usize].push(((*e0 + ptrs.len()) as i32, *w));
                            ptrs.push(base);
                            cols.push(*ti);
                        }
                    }
                }
                let m = ptrs.len();
                let mut tb: Vec<u8> = Vec::with_capacity(m * 12);
                tb.extend(ptrs.iter().flat_map(|v| (v + parts.gate.0 as u64).to_le_bytes()));
                tb.extend(i32_bytes(&cols));
                let mut db: Vec<u8> = Vec::with_capacity(m * 12);
                db.extend(ptrs.iter().flat_map(|v| (v + parts.down.0 as u64).to_le_bytes()));
                db.extend(i32_bytes(&(0..m as i32).collect::<Vec<_>>()));
                let gt = p.upload(&tb)?;
                let dt = p.upload(&db)?;
                let g1 = self.lap(p, "experts a: tables", g0);
                if let Some(tk) = ticket {
                    o.await_ticket(tk)?;
                }
                let g2 = self.lap(p, "experts b: copies awaited", g1);
                let (gu_g, hh_g, dn_g) = (gu.view(*e0 * 2 * f * 4, m * 2 * f * 4)?, hh.view(*e0 * f * 4, m * f * 4)?, dn.view(*e0 * d * 4, m * d * 4)?);
                o.mmvq_multi(parts.gate.2.code(), &gt, m, &xs, t, &gu_g, d, 2 * f)?;
                let g3 = self.lap(p, "experts c: gate | up", g2);
                o.swiglu_rows(&gu_g, &hh_g, m, f)?;
                let hs = p.arena.bytes(o.q8_bytes(f, m))?;
                o.quantize_q8((&hh_g, 0), &hs, f, m)?;
                let g4 = self.lap(p, "experts d: swiglu + quantize", g3);
                o.mmvq_multi(parts.down.2.code(), &dt, m, &hs, m, &dn_g, f, d)?;
                self.lap(p, "experts e: down", g4);
                *e0 += m;
                Ok(())
            };
            // the resident ones without waiting, then the ones arriving from pinned memory (their copies done)
            let (now_k, now_s): (Vec<usize>, Vec<usize>) = first.iter().enumerate().filter_map(|(k, s)| s.filter(|_| !fresh[k]).map(|s| (k, s))).unzip();
            self.timed(p, "routed experts (resident)", || group(&now_k, &now_s, None, &mut e0, &mut per_token, &mut entry_of))?;
            let (up_k, up_s): (Vec<usize>, Vec<usize>) = first.iter().enumerate().filter_map(|(k, s)| s.filter(|_| fresh[k]).map(|s| (k, s))).unzip();
            self.timed(p, "routed experts (from pinned memory)", || group(&up_k, &up_s, arriving, &mut e0, &mut per_token, &mut entry_of))?;
            let later: Vec<usize> = (0..first.len()).filter(|k| first[*k].is_none()).collect();
            if !later.is_empty() {
                let (rest, tb) = self.timed(p, "expert loads (file)", || p.experts.lock().unwrap().ensure_rest(o, l, &need, &first))?;
                self.timed(p, "routed experts (from the file)", || group(&later, &rest, tb, &mut e0, &mut per_token, &mut entry_of))?;
            }
            // NS_DS2_TAP_EXPERTS=1: each row's experts' gate, up and down outputs in its choice order (llama.cpp's
            // ffn_moe_gate / _up / _down: [t, used, width]), for the checks (a debugging aid: reads them back)
            if std::env::var("NS_DS2_TAP_EXPERTS").is_ok_and(|v| v == "1") {
                let (guv, dnv) = (gu.to_f32()?, dn.to_f32()?);
                let (mut ga, mut ua, mut da) = (Vec::new(), Vec::new(), Vec::new());
                for (ti, sel) in rows.iter().enumerate() {
                    for (x, _) in sel {
                        let Some(&e) = entry_of.get(&(ti, *x)) else { continue };
                        ga.extend_from_slice(&guv[e * 2 * f..e * 2 * f + f]);
                        ua.extend_from_slice(&guv[e * 2 * f + f..(e + 1) * 2 * f]);
                        da.extend_from_slice(&dnv[e * d..(e + 1) * d]);
                    }
                }
                for (name, v) in [("ffn_moe_gate", ga), ("ffn_moe_up", ua), ("ffn_moe_down", da)] {
                    tap(&format!("{name}-{l}"), &DevBuf::from_f32(&o.gpu, &v)?)?;
                }
            }
            self.timed(p, "routed experts", || {
                let mut ints: Vec<i32> = vec![0];
                let mut ent: Vec<i32> = Vec::with_capacity(n);
                let mut wts: Vec<f32> = Vec::with_capacity(n);
                for list in &per_token {
                    for (e, w) in list {
                        ent.push(*e);
                        wts.push(*w);
                    }
                    ints.push(ent.len() as i32);
                }
                ints.extend(&ent);
                let ib = p.upload_i32(&ints)?;
                let wb = p.upload_f32(&wts)?;
                o.moe_combine(&y, &dn, &ib, &wb, t, n, d)
            })?;
        } else {
            self.timed(p, "routed experts (prompt)", || self.moe_prompt(p, l, x, &by, &parts, &y))?;
        }
        tap(&format!("ffn_moe_out-{l}"), &y)?;
        tap(&format!("ffn_shexp-{l}"), &sh)?;
        o.add(&y, &sh, t * d)?;
        if let Some(gv) = guess.as_ref() {
            // NS_DS2_TRACE_GUESS=FILE: layer l + 1's router on layer l's input, its 32 best in order a line (with the
            // routing's groups as `route` picks), beside NS_DS2_TRACE_EXPERTS' actual requests
            if let Ok(fpath) = std::env::var("NS_DS2_TRACE_GUESS") {
                use std::io::Write;
                let gb = p.biases.get(&(l + 1)).unwrap_or(&no_bias);
                let mut lines = String::new();
                for ti in 0..t {
                    let pr: Vec<f32> = gv[ti * ne..(ti + 1) * ne].iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
                    let mut order: Vec<usize> = (0..ne).collect();
                    order.sort_by(|a, b| (pr[*b] + gb.get(*b).copied().unwrap_or(0.0)).total_cmp(&(pr[*a] + gb.get(*a).copied().unwrap_or(0.0))));
                    lines += &format!("G {} {}\n", l + 1, order[..32].iter().map(|e| e.to_string()).collect::<Vec<_>>().join(","));
                }
                if let Ok(mut w) = std::fs::OpenOptions::new().create(true).append(true).open(fpath) {
                    let _ = w.write_all(lines.as_bytes());
                }
            }
        }
        if let Some(gv) = guess {
            let gb = p.biases.get(&(l + 1)).unwrap_or(&no_bias);
            // each expert's chance of being asked for: per row, the hit rate of its rank in the guess, rows combined
            let mut miss: BTreeMap<usize, f32> = BTreeMap::new();
            for ti in 0..t {
                let pr: Vec<f32> = gv[ti * ne..(ti + 1) * ne].iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|a, b| (pr[*b] + gb.get(*b).copied().unwrap_or(0.0)).total_cmp(&(pr[*a] + gb.get(*a).copied().unwrap_or(0.0))));
                for (rank, &x) in order.iter().take(GUESS_HIT.len()).enumerate() {
                    *miss.entry(x).or_insert(1.0) *= 1.0 - GUESS_HIT[rank];
                }
            }
            let th = prefetch_p();
            let mut want: Vec<(usize, f32)> = miss.into_iter().map(|(x, q)| (x, 1.0 - q)).filter(|x| x.1 >= th).collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1));
            let want: Vec<u64> = want.into_iter().map(|(x, _)| x as u64).collect();
            self.timed(p, "expert prefetch", || p.experts.lock().unwrap().prefetch(o, l + 1, &want, pf))?;
        }
        Ok(y)
    }

    /// Decode's routed experts with the CPU's share (all IQ1_S): the VRAM-resident ones on the GPU in the decode kernels,
    /// the pinned ones on the CPU where they are (no copy over PCIe), the file-only ones on the CPU as their reads land;
    /// the CPU's output rows uploaded beside the GPU's, one combine. Then a few of the CPU's experts copied up to VRAM
    /// in the background, while the copy lane is idle (`Store::promote`).
    #[allow(clippy::too_many_arguments)]
    fn moe_hybrid(&self, p: &Part, l: u64, t: usize, x: &DevBuf, xh: &[f32], by: &BTreeMap<usize, Vec<(i32, f32)>>, parts: &ExpertParts, y: &DevBuf,
                  tap: Tap) -> Result<()> {
        use crate::store::Where;
        let g = &self.m.g;
        let o = &p.ops;
        let pool = self.pool.as_ref().ok_or("no CPU workers")?;
        let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        let lists: Vec<(&usize, &Vec<(i32, f32)>)> = by.iter().collect();
        let n: usize = lists.iter().map(|(_, v)| v.len()).sum();
        let (wh, arriving) = self.timed(p, "expert loads", || p.experts.lock().unwrap().place(o, l, &need))?;
        let dn = p.arena.f32(n * d)?;
        let mut per_token: Vec<Vec<(i32, f32)>> = vec![Vec::new(); t];
        let mut e0 = 0usize;
        // the GPU's: resident ones first, then the ones whose copy up is still on its way
        let xs = p.arena.bytes(o.q8_bytes(d, t))?;
        o.quantize_q8((x, 0), &xs, d, t)?;
        for (fresh_group, ticket) in [(false, None), (true, arriving)] {
            let ks: Vec<usize> = (0..wh.len()).filter(|k| matches!(wh[*k], Where::Gpu(_, fr) if fr == fresh_group)).collect();
            if ks.is_empty() {
                continue;
            }
            self.timed(p, "routed experts (GPU)", || {
                let mut ptrs: Vec<u64> = Vec::new();
                let mut cols: Vec<i32> = Vec::new();
                {
                    let store = p.experts.lock().unwrap();
                    for &k in &ks {
                        let Where::Gpu(s, _) = wh[k] else { continue };
                        let base = store.vview(s)?.ptr() as u64;
                        for (ti, w) in lists[k].1 {
                            per_token[*ti as usize].push(((e0 + ptrs.len()) as i32, *w));
                            ptrs.push(base);
                            cols.push(*ti);
                        }
                    }
                }
                let m = ptrs.len();
                let mut tb: Vec<u8> = Vec::with_capacity(m * 12);
                tb.extend(ptrs.iter().flat_map(|v| (v + parts.gate.0 as u64).to_le_bytes()));
                tb.extend(i32_bytes(&cols));
                let mut db: Vec<u8> = Vec::with_capacity(m * 12);
                db.extend(ptrs.iter().flat_map(|v| (v + parts.down.0 as u64).to_le_bytes()));
                db.extend(i32_bytes(&(0..m as i32).collect::<Vec<_>>()));
                let (gt, dt) = (p.upload(&tb)?, p.upload(&db)?);
                if let Some(tk) = ticket {
                    o.await_ticket(tk)?;
                }
                let gu = p.arena.f32(m * 2 * f)?;
                o.mmvq_multi(parts.gate.2.code(), &gt, m, &xs, t, &gu, d, 2 * f)?;
                let hh = p.arena.f32(m * f)?;
                o.swiglu_rows(&gu, &hh, m, f)?;
                let hs = p.arena.bytes(o.q8_bytes(f, m))?;
                o.quantize_q8((&hh, 0), &hs, f, m)?;
                o.mmvq_multi(parts.down.2.code(), &dt, m, &hs, m, &dn.view(e0 * d * 4, m * d * 4)?, f, d)?;
                e0 += m;
                Ok(())
            })?;
        }
        // the CPU's: the pinned ones first (the file's reads still landing), then the file's
        let xq: Vec<crate::cpu::Q8> = (0..t).map(|ti| crate::cpu::quantize(&xh[ti * d..(ti + 1) * d])).collect();
        let mut cpu_keys: Vec<u64> = Vec::new();
        for file in [false, true] {
            let ks: Vec<usize> = (0..wh.len()).filter(|k| matches!(wh[*k], Where::Cpu(_)) != file && matches!(wh[*k], Where::Cpu(_) | Where::File(_))).collect();
            if ks.is_empty() {
                continue;
            }
            let rows = self.timed(p, if file { "routed experts (CPU, from the file)" } else { "routed experts (CPU)" }, || {
                let rs: Vec<usize> = ks.iter().map(|k| match wh[*k] { Where::Cpu(r) | Where::File(r) => r, _ => 0 }).collect();
                let mut store = p.experts.lock().unwrap();
                if file {
                    store.finish_reads(&rs)?;
                }
                let mut jobs = Vec::new();
                for (k, r) in ks.iter().zip(&rs) {
                    for (ti, _) in lists[*k].1 {
                        jobs.push(crate::cpu::Job { slot: store.host_slice(*r), x: &xq[*ti as usize] });
                    }
                }
                Ok(crate::cpu::experts(pool, &jobs, d, f, parts.down.0))
            })?;
            let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * d);
            let mut j = 0;
            for &k in &ks {
                cpu_keys.push(*lists[k].0 as u64);
                for (ti, w) in lists[k].1 {
                    per_token[*ti as usize].push(((e0 + j) as i32, *w));
                    flat.extend_from_slice(&rows[j]);
                    j += 1;
                }
            }
            p.upload_into(&dn, e0 * d * 4, &f32_bytes(&flat))?;
            e0 += j;
        }
        let _ = &mut *tap;
        self.timed(p, "routed experts (combine)", || {
            let mut ints: Vec<i32> = vec![0];
            let mut ent: Vec<i32> = Vec::with_capacity(n);
            let mut wts: Vec<f32> = Vec::with_capacity(n);
            for list in &per_token {
                for (e, w) in list {
                    ent.push(*e);
                    wts.push(*w);
                }
                ints.push(ent.len() as i32);
            }
            ints.extend(&ent);
            let ib = p.upload_i32(&ints)?;
            let wb = p.upload_f32(&wts)?;
            o.moe_combine(y, &dn, &ib, &wb, t, n, d)
        })?;
        let lim = promote_limit();
        if lim > 0 && !cpu_keys.is_empty() {
            self.timed(p, "expert promotion", || p.experts.lock().unwrap().promote(o, l, &cpu_keys, lim))?;
        }
        Ok(())
    }

    /// A prompt chunk's routed experts: each expert expanded to fp16 once and its tokens multiplied (oneMKL's half
    /// GEMMs); host experts copied into a VRAM staging ring first, the copies running ahead of the compute
    #[allow(clippy::too_many_arguments)]
    fn moe_prompt(&self, p: &Part, l: u64, x: &DevBuf, by: &BTreeMap<usize, Vec<(i32, f32)>>, parts: &ExpertParts, y: &DevBuf) -> Result<()> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, f) = (g.n_embd as usize, g.ffn_expert as usize);
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        let slots = p.experts.lock().unwrap().locate(o, l, &need)?;
        let total: usize = by.values().map(|v| v.len()).sum();
        let mut gtok: Vec<i32> = Vec::with_capacity(total);
        let mut gw: Vec<f32> = Vec::with_capacity(total);
        for list in by.values() {
            for (ti, w) in list {
                gtok.push(*ti);
                gw.push(*w);
            }
        }
        let tb = p.upload_i32(&gtok)?;
        let wtb = p.upload_f32(&gw)?;
        let nmax = by.values().map(|v| v.len()).max().unwrap_or(1);
        let xh = p.arena.bytes(nmax * d * 2)?;
        let w16 = p.arena.bytes(2 * f * d * 2)?;
        let gu = p.arena.f32(nmax * 2 * f)?;
        let hh = p.arena.bytes(nmax * f * 2)?;
        let dn = p.arena.f32(nmax * d)?;
        const RING: usize = 3;
        let sb = parts.bytes();
        let ring = (0..RING).map(|_| p.arena.bytes(sb)).collect::<Result<Vec<_>>>()?;
        let hosts: Vec<usize> = slots.iter().enumerate().filter(|(_, s)| s.1).map(|(k, _)| k).collect();
        let mut copied: Vec<Option<(i64, usize)>> = vec![None; slots.len()];
        let mut freed: [Option<i64>; RING] = [None; RING];
        let mut next = 0;
        let issue = |next: &mut usize, freed: &[Option<i64>; RING], copied: &mut [Option<(i64, usize)>]| -> Result<()> {
            if let Some(&k) = hosts.get(*next) {
                let r = *next % RING;
                copied[k] = Some((o.stream_copy_on(0, &ring[r], 0, &slots[k].0, 0, sb, &[freed[r], slots[k].2])?, r));
                *next += 1;
            }
            Ok(())
        };
        for _ in 0..RING {
            issue(&mut next, &freed, &mut copied)?;
        }
        let mut row = 0;
        let mut last_copy = None;
        for (k, list) in by.values().enumerate() {
            let n = list.len();
            let toks = tb.view(row * 4, n * 4)?;
            let (buf, slot) = match copied[k] {
                Some((ticket, r)) => {
                    o.await_ticket(ticket)?;
                    last_copy = Some(ticket);
                    (&ring[r], Some(r))
                }
                None => (&slots[k].0, None),
            };
            o.gather_f16(x, &toks, &xh, n, d)?;
            o.dequant_f16(parts.gate.2.code(), buf, parts.gate.0, parts.gate.1, f * d, &w16)?;
            o.dequant_f16(parts.up.2.code(), buf, parts.up.0, parts.up.1, f * d, &w16.view(f * d * 2, f * d * 2)?)?;
            o.gemm_f16(n, 2 * f, d, (&xh, 0, d), (&w16, 0), (&gu, 0, 2 * f), false)?;
            o.swiglu_gu_f16(&gu, &hh, n, f)?;
            o.dequant_f16(parts.down.2.code(), buf, parts.down.0, parts.down.1, d * f, &w16)?;
            if let Some(r) = slot {
                freed[r] = Some(o.mark()?);
                issue(&mut next, &freed, &mut copied)?;
            }
            o.gemm_f16(n, d, f, (&hh, 0, f), (&w16, 0), (&dn, 0, d), false)?;
            o.scatter_add(y, &dn, &toks, &wtb.view(row * 4, n * 4)?, n, d)?;
            row += n;
        }
        // nothing may overwrite the host slots read here before their copies are done
        if let Some(t) = last_copy {
            let keys: Vec<(u64, u64)> = hosts.iter().map(|&k| (l, need[k])).collect();
            p.experts.lock().unwrap().host_read_by(&keys, t);
        }
        Ok(())
    }

    /// The head after the last layer (x on the last part), for the last `n_out` rows
    fn head(&self, x: &DevBuf, t: usize, n_out: usize, d: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        p.arena.reset();
        let last = x.view((t - n_out) * d * 4, n_out * d * 4)?;
        let out = p.arena.f32(n_out * d)?;
        o.rms_norm(&last, Some(p.vec(0, Role::OutputNorm)?), &out, n_out, d, self.m.g.rms_eps as f32)?;
        tap("result_norm", &out)?;
        let logits = self.timed(p, "head", || p.mm(0, Role::Output, &out, n_out))?;
        tap("result_output", &logits)?;
        let v = p.read_f32(&logits, n_out * self.m.g.n_vocab as usize)?;
        let vocab = self.m.g.n_vocab as usize;
        Ok(v.chunks(vocab).map(|c| c.to_vec()).collect())
    }

    pub fn forward_batch(&self, sessions: &mut [&mut Session], tokens: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        sessions.iter_mut().zip(tokens).map(|(s, t)| self.forward(s, &[*t], &mut *tap)).collect()
    }

    pub fn decoder(&self, logits: Vec<f32>) -> Decoder {
        Decoder { logits, next: None, look: None, drafted: 0, accepted: 0 }
    }
    pub fn decoder_after(&self, next: u32) -> Decoder {
        Decoder { logits: Vec::new(), next: Some(next), look: None, drafted: 0, accepted: 0 }
    }

    /// The next committed token(s): the last one fed first - with the context's repeats as drafts, verified in the
    /// same pass when there are some (each row sampled as a one-token pass would be; the first that differs from its
    /// draft ends the run, and the cache keeps only what is committed)
    pub fn step(&self, s: &mut Session, d: &mut Decoder, smp: &mut dyn Sampler, tap: Tap) -> Result<Vec<u32>> {
        if let Some(n) = d.next.take() {
            let drafts: Vec<u32> = match &d.look {
                Some(l) => {
                    let mut v = l.propose(ngram_k());
                    v.truncate(s.max_ctx.saturating_sub(s.pos + 1));
                    v
                }
                None => Vec::new(),
            };
            if drafts.is_empty() {
                d.logits = self.forward(s, &[n], tap)?;
            } else {
                let mut toks = vec![n];
                toks.extend_from_slice(&drafts);
                let rows = self.forward_rows(s, &toks, toks.len(), tap)?;
                let mut out = Vec::with_capacity(toks.len());
                for (i, r) in rows.iter().enumerate() {
                    let y = smp.sample(r);
                    smp.record(r, y);
                    out.push(y);
                    if i < drafts.len() {
                        d.drafted += 1;
                        if y == drafts[i] {
                            d.accepted += 1;
                            continue;
                        }
                    }
                    break;
                }
                if out.len() < toks.len() {
                    self.rollback(s, out.len())?;
                }
                if let Some(l) = d.look.as_mut() {
                    for &y in &out {
                        l.push(y);
                    }
                }
                d.next = out.last().copied();
                return Ok(out);
            }
        }
        if d.logits.is_empty() {
            return Err(Error("step: no logits to sample from".into()));
        }
        let tok = smp.sample(&d.logits);
        smp.record(&d.logits, tok);
        if let Some(l) = d.look.as_mut() {
            l.push(tok);
        }
        d.next = Some(tok);
        Ok(vec![tok])
    }
}
