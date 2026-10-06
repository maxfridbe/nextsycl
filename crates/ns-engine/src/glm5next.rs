//! GLM-5.3-Flash on one or more GPUs: every step of `docs/glm5next.md` in order, float32 activations, the kernels
//! of `kernels/ns`. The layers are split over the GPUs (a `Part` each: its layers' weights in their stored form,
//! its own expert cache in VRAM and its own share of the pinned host mirror - pinned memory belongs to one GPU's
//! context); the token stream [T, 4, hidden] crosses to the next GPU through host memory where the layers do. Up
//! to 8 rows multiply from the stored blocks (decode), more are expanded in chunks. MLA attends to every earlier
//! token (the indexer selects all of them up to ~2,048 tokens of context, docs/glm5next.md).
//!
//! MTP (the ds4 file's block 45, on the last GPU): the draft block reads each position's final hidden state (the
//! mean of the 4 streams) with the token that follows it and predicts the one after. It runs over the prompt too
//! (batched, behind each chunk), so its own latent cache covers the whole conversation. `step` drafts one token
//! and verifies it with the next in one 2-row pass; a rejected draft rolls the KDA states back to their snapshot
//! after the first row (no replay).

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_core::{Arena, DevBuf, Error, Gpu, HostBuf, Ops, Result};
use ns_gguf::{GType, Gguf};
use ns_model::glm5next::{Model, Role, Scheme};

use crate::Tap;

/// float32 values expanded per matrix chunk (128 MiB)
const SCRATCH: usize = 32 << 20;
/// the decode kernels take up to this many rows (tokens) at once
const MMVQ_COLS: usize = 8;
/// widest matrix input (MLA's output projection)
const MAX_COLS: usize = 16384;
/// expert slots and the mirror are allocated in chunks of this size (single allocations stay small)
const CHUNK: usize = 2 << 30;
/// Prompt tokens per forward pass (NS_PREFILL_CHUNK, default 4096, 64..8192): bigger chunks give each expert's
/// weights more tokens (Strata's lesson: experts are streamed once a chunk), and need a bigger arena.
pub fn prefill_chunk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096).clamp(64, 8192))
}

/// Bytes of each GPU's arena (NS_ARENA_MIB; default: 1 GiB, or ~0.55 MiB a token of the prompt chunk past 1,900 -
/// the temporaries measured 0.56 GiB at 512 and 2.0 GiB at 4,096 with the fp16 expert path)
pub fn arena_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("NS_ARENA_MIB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(m) => m << 20,
        // the fp16 path's peak: ~0.5 MB a token at 4,096 (measured 2.0 GiB)
        None => (1usize << 30).max(prefill_chunk() * (560 << 10)),
    })
}
/// rows of a verify pass (the token and its draft); the KDA states keep a snapshot after each row but the last
pub const MAX_VERIFY: usize = 2;
/// the indexer's pool scores per GEMM (floats): pools are scored in chunks that fit (64 MiB)
const IDX_S_FLOATS: usize = 16 << 20;

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
}

/// Where a layer's [gate | up | down] sit in a slot: (byte offset, bytes, type) each.
#[derive(Clone, Copy)]
struct ExpertParts {
    gate: (usize, usize, GType),
    up: (usize, usize, GType),
    down: (usize, usize, GType),
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
/// slot. When VRAM and host slots together hold every expert of the part (two 32 GB cards and ~45 GB of host memory
/// for the IQ2 file), nothing is read from the file after load; otherwise the rest comes from the file on demand.
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
    /// host-slot experts a prompt pass reads from pinned memory (in place, or copied to a staging slot first)
    direct: u64,
}

impl Store {
    fn vat(&self, s: usize) -> (usize, usize) {
        (s / self.per_chunk, (s % self.per_chunk) * self.slot_bytes)
    }
    fn rat(&self, r: usize) -> (usize, usize) {
        (r / self.per_chunk, (r % self.per_chunk) * self.slot_bytes)
    }
}

/// One GPU's share of the model.
pub struct Part {
    ops: Ops,
    /// the forward pass's temporaries, reset at each layer
    pub arena: Arena,
    pub layers: Range<u64>,
    mats: BTreeMap<(u64, Role), Mat>,
    vecs: BTreeMap<(u64, Role), DevBuf>,
    scratch: DevBuf,
    /// Q8_1 of up to MMVQ_COLS rows of MAX_COLS
    q8: DevBuf,
    experts: Mutex<Store>,
    /// the MTP block's eh_proj as its [embedding | hidden] halves, [n_embd, n_embd] each (on the part holding it)
    eh: Option<[Mat; 2]>,
    pub expert_slots: usize,
    pub host_slots: usize,
    pub weight_bytes: u64,
}

/// A conversation's state: per KDA layer the recurrent state and the convolutions' last inputs, per MLA layer the
/// latent cache (each on its layer's GPU); `pos` tokens so far.
pub struct Session {
    pub pos: usize,
    pub max_ctx: usize,
    layers: Vec<LayerState>,
    /// (first position, rows) of the last forward pass when it kept state snapshots (2..=MAX_VERIFY rows)
    snapped: Option<(usize, usize)>,
    mtp: Option<MtpState>,
}

enum LayerState {
    /// `snap`: the recurrent state and the convolutions' inputs after each row of a verify pass but the last
    Kda { s: DevBuf, conv: [DevBuf; 3], snap: Option<Box<(DevBuf, [DevBuf; 3])>> },
    Mla { c: DevBuf, idx: Idx },
}

/// An MLA layer's indexer state: the last tokens' key and pool gate (ring [4][2 * idx_dim], slot pos % 4) and the
/// pooled key of every completed pool [ctx / 4 + 1, idx_dim].
struct Idx {
    ring: DevBuf,
    pooled: DevBuf,
}

impl Idx {
    fn new(gpu: &Arc<Gpu>, max_ctx: usize, d: usize) -> Result<Idx> {
        let ring = DevBuf::f32(gpu, 8 * d)?;
        ring.fill(0)?;
        Ok(Idx { ring, pooled: DevBuf::f32(gpu, (max_ctx / 4 + 1) * d)? })
    }
    /// `self` = `src` for a conversation of `pos` tokens
    fn copy_from(&self, src: &Idx, pos: usize, d: usize) -> Result<()> {
        self.ring.copy_within(0, &src.ring, 0, src.ring.len)?;
        if pos >= 4 {
            self.pooled.copy_within(0, &src.pooled, 0, pos / 4 * d * 4)?;
        }
        Ok(())
    }
}

/// The draft block's side of a conversation (on the last GPU): its latent cache (slot p: the pair at position p),
/// and the rows of the last forward pass it has not read yet - their final hidden states and, for all but the last,
/// the token that follows each (the last one's comes with the next call).
struct MtpState {
    cache: DevBuf,
    idx: Idx,
    hid: DevBuf,
    rows: usize,
    slot0: usize,
    next: Vec<u32>,
}

/// One GPU's share of the model (`Glm::gpu_info`).
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub total: u64,
    pub free: Option<u64>,
    pub layers: (u64, u64),
    pub expert_slots: usize,
    pub host_slots: usize,
}

/// A conversation's state at a position, in host memory (`Glm::save` / `Glm::restore`).
pub struct Checkpoint {
    pub pos: usize,
    max_ctx: usize,
    bufs: Vec<Vec<u8>>,
    mtp: Option<(usize, usize, Vec<u32>)>,
    /// host bytes held
    pub bytes: usize,
}

/// Generation from a fed prompt: `Glm::step` commits the next token(s) each call. The last committed token is fed
/// at the start of the following call (a stop token is never fed).
pub struct Decoder {
    logits: Vec<f32>,
    next: Option<u32>,
    draft: Option<u32>,
    mtp: bool,
    /// drafts verified, and accepted
    pub drafted: u64,
    pub accepted: u64,
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

pub struct Glm<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// the MTP block's layer, when it is loaded (on the last part)
    pub mtp: Option<u64>,
    /// NS_PROFILE=1: seconds and calls per section (the GPU synced at each boundary)
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
}

const VECTORS: [Role; 21] = [Role::MtpENorm, Role::MtpHNorm, Role::MtpHeadNorm, Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale, Role::KdaQConv,
                             Role::KdaKConv, Role::KdaVConv, Role::KdaDtBias, Role::KdaA, Role::KdaONorm, Role::MlaQANorm, Role::MlaKvANorm,
                             Role::IdxKNorm, Role::IdxKNormBias, Role::RouterBias];

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
    /// The part's layers onto its GPU (and the embedding / head when `first` / `last`), its expert cache and
    /// mirror.
    #[allow(clippy::too_many_arguments)]
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, extra: Range<u64>, last: bool, expert_bytes: Option<usize>, mirror_bytes: usize,
            kv_reserve: usize, log: &mut dyn FnMut(String)) -> Result<Part> {
        let ops = Ops { gpu: gpu.clone() };
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            todo.push((0, vec![Role::OutputNorm, Role::Output]));
        }
        todo.extend(layers.clone().chain(extra.clone()).map(|l| (l, m.roles(l))));
        let mut eh = None;
        for (l, roles) in todo {
            for r in roles {
                if matches!(r, Role::ExpGate | Role::ExpUp | Role::ExpDown | Role::TokenEmbd) {
                    continue; // experts through the cache; the embedding's rows are read per token
                }
                let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
                let raw = DevBuf::new(gpu, t.bytes as usize)?;
                raw.write(0, &m.file.read(t).map_err(e)?)?;
                bytes += t.bytes;
                let n = t.elements() as usize;
                if VECTORS.contains(&r) || matches!(r, Role::MlaKB | Role::MlaVB) {
                    let f = DevBuf::f32(gpu, n)?;
                    ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                    if r == Role::KdaA && m.scheme == Scheme::Ds4 {
                        // ds4 stores A_log; the gate wants A = -exp(A_log) (llama.cpp's ssm_a holds that already)
                        let v: Vec<f32> = f.to_f32()?.iter().map(|a| -a.exp()).collect();
                        f.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                    }
                    vecs.insert((l, r), f);
                } else {
                    let cols = *t.shape.last().unwrap_or(&1) as usize;
                    if r == Role::MtpEhProj {
                        // [d, 2d] over concat(enorm(embedding), hnorm(hidden)): split into the two [d, d] halves
                        let f = DevBuf::f32(gpu, n)?;
                        ops.dequant(t.ty.code(), &raw, 0, t.bytes as usize, n, &f)?;
                        let (rows, half) = (n / cols, cols / 2);
                        let v = f.to_f32()?;
                        let mut halves = [Vec::with_capacity(rows * half), Vec::with_capacity(rows * half)];
                        for row in v.chunks(cols) {
                            halves[0].extend_from_slice(&row[..half]);
                            halves[1].extend_from_slice(&row[half..]);
                        }
                        let mk = |h: &[f32]| -> Result<Mat> { Ok(Mat { buf: DevBuf::from_f32(gpu, h)?, ty: GType::F32, rows, cols: half, f32: true }) };
                        eh = Some([mk(&halves[0])?, mk(&halves[1])?]);
                    } else if matches!(t.ty, GType::F32 | GType::F16 | GType::BF16) {
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
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(MAX_COLS, MMVQ_COLS))?;
        // the store: slots of the largest layer's [gate | up | down], in chunks; VRAM, then pinned host memory
        let moe: Vec<u64> = layers.clone().filter(|l| m.g.is_moe(*l)).chain(extra.clone()).collect();
        let n_exp = moe.len() * m.g.n_expert as usize;
        let slot_bytes = moe.iter().map(|l| m.expert_bytes(*l) as usize).max().unwrap_or(256).next_multiple_of(256);
        let budget = match expert_bytes {
            Some(b) => b,
            // what is free less 3 GiB (the arena, the forward pass, at least 1.5 GiB left over) and the sessions'
            // attention caches, sized now so a long context never pushes VRAM into the driver's spill path
            // (the arena past its 1 GiB comes out of the experts too)
            None => gpu.memory()?.1.map_or(8usize << 30, |f| (f as usize).saturating_sub((2 << 30) + arena_bytes().max(1 << 30) + kv_reserve)),
        };
        let per_chunk = (CHUNK / slot_bytes).max(1);
        let nv = (budget / slot_bytes).min(n_exp);
        // host slots: the rest of the part's experts, and one spare for the swaps
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
        // what goes where at load: the part's experts in layer order, VRAM first, then host (all but the spare)
        let all: Vec<(u64, u64)> = moe.iter().flat_map(|l| (0..m.g.n_expert).map(move |x| (*l, x))).collect();
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
        let store = Store { vram, host, slot_bytes, per_chunk, loc, vowner, vused: vec![0; nv], rfree, tick: 0, hits: 0, misses: 0, from_host: 0, direct: 0 };
        let arena = Arena::new(gpu, arena_bytes())?;
        Ok(Part { ops, arena, layers, mats, vecs, scratch, q8, experts: Mutex::new(store), eh, expert_slots: nv, host_slots: nr, weight_bytes: bytes })
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
            // decode widths a row at a time: oneMKL picks its kernel (and summation order) by the row count, so this
            // keeps a verify pass's rows equal to one-token passes
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
        let chunk = (SCRATCH / w.cols).max(1).min(w.rows);
        let rb = w.row_bytes();
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

    /// Where every expert of `need` (layer `l`) is read from, in `need`'s order, as a view of its slot. With
    /// `promote`, every one is made VRAM-resident, none evicting another: a miss swaps with the least recently used
    /// VRAM expert (it goes down to a free host slot, the wanted one comes up from its host slot or the file).
    /// Without, an expert in a pinned host slot is read there in place by the GPU over PCIe (once per pass - what a
    /// prompt chunk wants), and only the ones on the file alone come into VRAM.
    fn ensure(&self, m: &Model, l: u64, need: &[u64], promote: bool) -> Result<Vec<(DevBuf, bool)>> {
        let mut c = self.experts.lock().unwrap();
        let c = &mut *c;
        if c.vowner.len() < need.len() {
            return Err(Error(format!("the expert store has {} VRAM slots; one layer needs {}", c.vowner.len(), need.len())));
        }
        c.tick += 1;
        let tick = c.tick;
        let mut missing = Vec::new();
        for &ex in need {
            match c.loc.get(&(l, ex)).copied() {
                Some(Loc::V(s)) => {
                    c.hits += 1;
                    c.vused[s] = tick;
                }
                Some(Loc::R(_)) if !promote => {}
                _ => missing.push(ex),
            }
        }
        if !missing.is_empty() {
            let mut order: Vec<usize> = (0..c.vowner.len()).filter(|&i| c.vused[i] != tick).collect();
            order.sort_by_key(|&i| if c.vowner[i].is_none() { 0 } else { c.vused[i] + 1 });
            let parts = expert_parts(m, l)?;
            let mut buf: Option<Vec<u8>> = None;
            for (ex, &s) in missing.iter().zip(&order) {
                c.misses += 1;
                let key = (l, *ex);
                let (vch, vo) = c.vat(s);
                let from = c.loc.get(&key).copied();
                // 1. the victim down to a free host slot (when there is one; else it is only on the file again)
                if let Some(v) = c.vowner[s].take() {
                    c.loc.remove(&v);
                    // the wanted expert's own host slot frees below, so a full host side still has room after it
                    if let Some(f) = c.rfree.pop() {
                        let (rch, ro) = c.rat(f);
                        let sb = c.slot_bytes;
                        c.vram[vch].read(vo, &mut c.host[rch].as_mut_slice()[ro..ro + sb])?;
                        c.loc.insert(v, Loc::R(f));
                    }
                }
                // 2. the wanted expert up
                match from {
                    Some(Loc::R(r)) => {
                        c.from_host += 1;
                        let (rch, ro) = c.rat(r);
                        let sb = c.slot_bytes;
                        c.vram[vch].write(vo, &c.host[rch].as_slice()[ro..ro + sb])?;
                        c.rfree.push(r);
                    }
                    _ => {
                        let b = buf.get_or_insert_with(|| vec![0u8; c.slot_bytes]);
                        read_expert(m, l, *ex, &parts, b)?;
                        c.vram[vch].write(vo, b)?;
                    }
                }
                c.vowner[s] = Some(key);
                c.vused[s] = tick;
                c.loc.insert(key, Loc::V(s));
            }
        }
        let sb = c.slot_bytes;
        need.iter().map(|ex| match c.loc[&(l, *ex)] {
            Loc::V(s) => {
                let (ch, o) = c.vat(s);
                Ok((c.vram[ch].view(o, sb)?, false))
            }
            Loc::R(r) => {
                c.direct += 1;
                let (rch, ro) = c.rat(r);
                Ok((c.host[rch].device_view(ro, sb)?, true))
            }
        }).collect()
    }

    /// Several matrices on the same input x [t, cols]: one Q8_1 quantization of x serves every decode product.
    fn mm_many(&self, l: u64, roles: &[Role], x: &DevBuf, t: usize) -> Result<Vec<DevBuf>> {
        let ws: Vec<&Mat> = roles.iter().map(|r| self.mat(l, *r)).collect::<Result<_>>()?;
        let quantized = ws.iter().filter(|w| !w.f32 && self.ops.mmvq_supported(w.ty.code())).count();
        let cols = ws[0].cols;
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

    /// One part of a resident expert (its VRAM chunk `buf`, slot offset `at`) on x rows [n, cols] from float
    /// `x.1` into y rows [n, rows] from float `y.1`. `quantize`: x into Q8_1 first (false: the previous call's
    /// quantization of the same rows serves - gate and up share it).
    #[allow(clippy::too_many_arguments)]
    fn expert_into(&self, buf: &DevBuf, at: usize, part: (usize, usize, GType), rows: usize, cols: usize, x: (&DevBuf, usize), y: (&DevBuf, usize),
                   n: usize, quantize: bool) -> Result<()> {
        let (off, bytes, ty) = part;
        if n <= MMVQ_COLS && self.ops.mmvq_supported(ty.code()) {
            if quantize {
                self.ops.quantize_q8_1(x, &self.q8, cols, n)?;
            }
            self.ops.mmvq(ty.code(), (buf, at + off), bytes, &self.q8, y, cols, rows, n)
        } else {
            self.ops.dequant(ty.code(), buf, at + off, bytes, rows * cols, &self.scratch)?;
            self.ops.gemm_at(n, rows, cols, (x.0, x.1, cols), (&self.scratch, 0), (y.0, y.1, rows), false)
        }
    }

}

impl<'g> Glm<'g> {
    /// Loads the model over `gpus` (layers split evenly by count; the head on the last). Per GPU, `expert_bytes` of
    /// VRAM for routed experts (None: what is free less 3 GiB); `mirror_bytes` of pinned host memory in all for
    /// mirrored experts (None: what the host has available less 10 GiB), shared by the parts in proportion to their
    /// layers. `mtp`: the draft block too (when the file has one), on the last GPU. `kv`: (context, sessions) - the
    /// MLA latents and indexer caches of that many sessions of that context are held back from the expert store.
    #[allow(clippy::too_many_arguments)]
    pub fn load(file: &'g Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, mtp: bool, kv: (usize, usize),
                log: &mut dyn FnMut(String)) -> Result<Glm<'g>> {
        let t0 = Instant::now();
        let m = Model::open(file).map_err(e)?;
        if gpus.is_empty() {
            return Err(Error("no GPU given".into()));
        }
        let n = m.g.n_layer;
        let k = gpus.len() as u64;
        let mirror = mirror_bytes.unwrap_or(mem_available().saturating_sub(10 << 30));
        let mut parts = Vec::new();
        let mut owner = vec![0usize; n as usize];
        for (i, gpu) in gpus.iter().enumerate() {
            let range = (i as u64 * n / k)..((i as u64 + 1) * n / k);
            for l in range.clone() {
                owner[l as usize] = i;
            }
            let last = i + 1 == gpus.len();
            let extra = if last && mtp && m.g.n_mtp > 0 { n..n + 1 } else { n..n };
            let blocks = n + u64::from(mtp && m.g.n_mtp > 0);
            let share = mirror * (range.end - range.start + extra.end - extra.start) as usize / blocks as usize;
            // per session: each MLA layer's latents [ctx, kv_lora] and pooled keys [ctx / 4, idx_dim], float32
            let per_layer = kv.0 * m.g.kv_lora as usize * 4 + (kv.0 / 4 + 1) * m.g.idx_dim as usize * 4;
            let mla = range.clone().filter(|l| m.g.is_mla(*l)).count() + (extra.end - extra.start) as usize;
            let reserve = kv.1 * mla * per_layer;
            if reserve > 0 {
                log(format!("{}: {:.2} GiB held for {} session(s) of {} tokens ({} attention layers)", gpu.name, gib(reserve), kv.1, kv.0, mla));
            }
            parts.push(Part::load(&m, gpu, range, extra, last, expert_bytes, share, reserve, log)?);
        }
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(BTreeMap::new()));
        let mtp = (mtp && m.g.n_mtp > 0).then_some(n);
        Ok(Glm { m, parts, owner, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, mtp, prof })
    }

    /// Per GPU: its number and name, memory (total, free when the driver says), its layers, its expert slots in
    /// VRAM and in pinned host memory.
    pub fn gpu_info(&self) -> Vec<GpuInfo> {
        self.parts.iter().map(|p| {
            let (total, free) = p.ops.gpu.memory().unwrap_or((0, None));
            GpuInfo { index: p.ops.gpu.index, name: p.ops.gpu.name.clone(), total, free, layers: (p.layers.start, p.layers.end),
                      expert_slots: p.expert_slots, host_slots: p.host_slots }
        }).collect()
    }

    /// Per GPU: the arena's most used bytes between resets, and the requests that did not fit it (own allocations)
    pub fn arena_peaks(&self) -> Vec<(usize, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        self.parts.iter().map(|p| (p.arena.peak.load(Relaxed), p.arena.spills.load(Relaxed))).collect()
    }

    pub fn expert_slots(&self) -> usize {
        self.parts.iter().map(|p| p.expert_slots).sum()
    }

    /// (VRAM hits, misses (swaps), swaps served from pinned host memory, host-slot experts a prompt pass read from
    /// pinned memory) so far
    pub fn expert_stats(&self) -> (u64, u64, u64, u64) {
        self.parts.iter().fold((0, 0, 0, 0), |a, p| {
            let c = p.experts.lock().unwrap();
            (a.0 + c.hits, a.1 + c.misses, a.2 + c.from_host, a.3 + c.direct)
        })
    }

    /// Runs `f`, adding its time to section `name` when profiling (`p`'s GPU synced around it).
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

    /// Profiling inside a section: now (the GPU synced), when profiling.
    fn mark(&self, p: &Part) -> Option<Instant> {
        self.prof.as_ref()?;
        p.ops.gpu.sync().ok()?;
        Some(Instant::now())
    }

    /// Adds the time since `from` to `name` and starts the next lap.
    fn lap(&self, p: &Part, name: &'static str, from: Option<Instant>) -> Option<Instant> {
        let (pr, t0) = (self.prof.as_ref()?, from?);
        p.ops.gpu.sync().ok()?;
        let mut m = pr.lock().unwrap();
        let x = m.entry(name).or_insert((0.0, 0));
        x.0 += t0.elapsed().as_secs_f64();
        x.1 += 1;
        Some(Instant::now())
    }

    /// The profile so far: (section, seconds, calls), slowest first.
    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// A new conversation, up to `max_ctx` tokens (MLA attends to every earlier token: exact up to ~2,048). With the
    /// draft block loaded, the KDA states get room for their verify snapshots and the draft block its cache.
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let spec = self.mtp.is_some();
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            let gpu = &self.parts[self.owner[l as usize]].ops.gpu;
            layers.push(if g.is_mla(l) {
                LayerState::Mla { c: DevBuf::f32(gpu, max_ctx * g.kv_lora as usize)?, idx: Idx::new(gpu, max_ctx, g.idx_dim as usize)? }
            } else {
                let sz = (g.kda_heads * g.kda_dim * g.kda_dim) as usize;
                let s = DevBuf::f32(gpu, sz)?;
                s.fill(0)?;
                let cw = (g.kda_conv as usize - 1) * kw;
                let mk = || -> Result<DevBuf> {
                    let b = DevBuf::f32(gpu, cw)?;
                    b.fill(0)?;
                    Ok(b)
                };
                let snap = if spec {
                    let r = MAX_VERIFY - 1;
                    Some(Box::new((DevBuf::f32(gpu, r * sz)?, [DevBuf::f32(gpu, r * cw)?, DevBuf::f32(gpu, r * cw)?, DevBuf::f32(gpu, r * cw)?])))
                } else {
                    None
                };
                LayerState::Kda { s, conv: [mk()?, mk()?, mk()?], snap }
            });
        }
        let mtp = match self.mtp {
            Some(_) => {
                let gpu = &self.parts.last().unwrap().ops.gpu;
                Some(MtpState { cache: DevBuf::f32(gpu, max_ctx * g.kv_lora as usize)?, idx: Idx::new(gpu, max_ctx, g.idx_dim as usize)?, hid: DevBuf::f32(gpu, prefill_chunk() * g.n_embd as usize)?, rows: 0,
                                slot0: 0, next: Vec::new() })
            }
            None => None,
        };
        Ok(Session { pos: 0, max_ctx, layers, snapped: None, mtp })
    }

    /// The next `tokens` of a conversation in chunks of at most `prefill_chunk()` (the arenas bound a chunk): the
    /// logits of the last token.
    pub fn feed(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for c in tokens.chunks(prefill_chunk()) {
            logits = self.forward(sess, c, &mut *tap)?;
        }
        Ok(logits)
    }

    /// The conversation's whole state at `sess.pos`, copied to host memory: per KDA layer its recurrent state and
    /// convolution inputs, per MLA layer its latents and pooled keys up to the position and its indexer ring, and the
    /// draft block's side (its cache, indexer, pending rows). `restore` puts it back into a session of the same shape
    /// - the prompt cache's checkpoints. Pageable memory: a checkpoint spans both GPUs' contexts.
    pub fn save(&self, sess: &Session) -> Result<Checkpoint> {
        let g = &self.m.g;
        let (lat, d, n) = (g.kv_lora as usize * 4, g.idx_dim as usize * 4, sess.pos);
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut take = |b: &DevBuf, len: usize| -> Result<()> {
            let mut v = vec![0u8; len];
            if len > 0 {
                b.read(0, &mut v)?;
            }
            bufs.push(v);
            Ok(())
        };
        for l in &sess.layers {
            match l {
                LayerState::Kda { s, conv, .. } => {
                    take(s, s.len)?;
                    for c in conv {
                        take(c, c.len)?;
                    }
                }
                LayerState::Mla { c, idx } => {
                    take(c, n * lat)?;
                    take(&idx.ring, idx.ring.len)?;
                    take(&idx.pooled, n / 4 * d)?;
                }
            }
        }
        let mtp = match &sess.mtp {
            Some(ms) => {
                let upto = if ms.rows > 0 { ms.slot0 } else { n };
                take(&ms.cache, upto * lat)?;
                take(&ms.idx.ring, ms.idx.ring.len)?;
                take(&ms.idx.pooled, upto / 4 * d)?;
                take(&ms.hid, ms.rows * g.n_embd as usize * 4)?;
                Some((ms.rows, ms.slot0, ms.next.clone()))
            }
            None => None,
        };
        let bytes = bufs.iter().map(|b| b.len()).sum();
        Ok(Checkpoint { pos: n, max_ctx: sess.max_ctx, bufs, mtp, bytes })
    }

    /// `sess` becomes the conversation `ck` was saved from (same context size).
    pub fn restore(&self, sess: &mut Session, ck: &Checkpoint) -> Result<()> {
        if ck.max_ctx != sess.max_ctx {
            return Err(Error("restore: the checkpoint is of another context size".into()));
        }
        let mut it = ck.bufs.iter();
        let mut put = |b: &DevBuf| -> Result<()> {
            let v = it.next().ok_or("restore: the checkpoint is short")?;
            if !v.is_empty() {
                b.write(0, v)?;
            }
            Ok(())
        };
        for l in &sess.layers {
            match l {
                LayerState::Kda { s, conv, .. } => {
                    put(s)?;
                    for c in conv {
                        put(c)?;
                    }
                }
                LayerState::Mla { c, idx } => {
                    put(c)?;
                    put(&idx.ring)?;
                    put(&idx.pooled)?;
                }
            }
        }
        if let (Some(ms), Some((rows, slot0, next))) = (&mut sess.mtp, &ck.mtp) {
            put(&ms.cache)?;
            put(&ms.idx.ring)?;
            put(&ms.idx.pooled)?;
            put(&ms.hid)?;
            ms.rows = *rows;
            ms.slot0 = *slot0;
            ms.next = next.clone();
        }
        sess.pos = ck.pos;
        sess.snapped = None;
        Ok(())
    }

    /// Back to an empty conversation (the recurrent states zeroed; the MLA caches need nothing: `pos` bounds them).
    pub fn reset_session(&self, sess: &mut Session) -> Result<()> {
        for l in &sess.layers {
            if let LayerState::Kda { s, conv, .. } = l {
                s.fill(0)?;
                for c in conv {
                    c.fill(0)?;
                }
            }
        }
        sess.pos = 0;
        sess.snapped = None;
        if let Some(m) = &mut sess.mtp {
            m.rows = 0;
            m.next.clear();
        }
        Ok(())
    }

    /// `dst` becomes a copy of `src` (same model, same context size): the recurrent states and the MLA caches up to
    /// `src.pos`, the draft block's cache and pending rows. How a conversation is kept at the end of a prompt and
    /// resumed from there.
    pub fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        if dst.max_ctx != src.max_ctx || dst.layers.len() != src.layers.len() {
            return Err(Error("copy_session: the sessions differ in shape".into()));
        }
        let lat = self.m.g.kv_lora as usize * 4;
        for (d, s) in dst.layers.iter().zip(&src.layers) {
            match (d, s) {
                (LayerState::Kda { s: ds, conv: dc, .. }, LayerState::Kda { s: ss, conv: sc, .. }) => {
                    ds.copy_within(0, ss, 0, ss.len)?;
                    for (a, b) in dc.iter().zip(sc) {
                        a.copy_within(0, b, 0, b.len)?;
                    }
                }
                (LayerState::Mla { c: dcache, idx: di }, LayerState::Mla { c: scache, idx: si }) => {
                    if src.pos > 0 {
                        dcache.copy_within(0, scache, 0, src.pos * lat)?;
                    }
                    di.copy_from(si, src.pos, self.m.g.idx_dim as usize)?;
                }
                _ => return Err(Error("copy_session: layer kinds differ".into())),
            }
        }
        if let (Some(dm), Some(sm)) = (&mut dst.mtp, &src.mtp) {
            // the slots written: those below the pending rows (all of them when none are pending)
            let upto = if sm.rows > 0 { sm.slot0 } else { src.pos };
            if upto > 0 {
                dm.cache.copy_within(0, &sm.cache, 0, upto * lat)?;
            }
            dm.idx.copy_from(&sm.idx, upto, self.m.g.idx_dim as usize)?;
            if sm.rows > 0 {
                dm.hid.copy_within(0, &sm.hid, 0, sm.rows * self.m.g.n_embd as usize * 4)?;
            }
            dm.rows = sm.rows;
            dm.slot0 = sm.slot0;
            dm.next = sm.next.clone();
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
        raw.write(0, &rows)?;
        let x = DevBuf::f32(&p.ops.gpu, tokens.len() * g.n_embd as usize)?;
        p.ops.dequant(emb.ty.code(), &raw, 0, rows.len(), tokens.len() * g.n_embd as usize, &x)?;
        Ok(x)
    }

    /// MLA of layer `l` on x [t, d] (normed) at positions pos0.., its latents written to `cache` and attending to
    /// every earlier one: the output [t, d].
    #[allow(clippy::too_many_arguments)]
    fn mla(&self, p: &Part, l: u64, normed: &DevBuf, cache: &DevBuf, idx: &Idx, pos0: usize, t: usize, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let eps = g.rms_eps as f32;
        let (nh, hd, lat) = (g.n_head as usize, g.head_dim as usize, g.kv_lora as usize);
        let t_proj = self.mark(p);
        let mut pj = p.mm_many(l, &[Role::MlaQA, Role::MlaKvA], normed, t)?.into_iter();
        let (qa, kv) = (pj.next().unwrap(), pj.next().unwrap());
        let qr = p.arena.f32(t * g.q_lora as usize)?;
        o.rms_norm(&qa, Some(p.vec(l, Role::MlaQANorm)?), &qr, t, g.q_lora as usize, eps)?;
        tap(&format!("q_resid-{l}"), &qr)?;
        let q = p.mm(l, Role::MlaQB, &qr, t)?;
        let c = p.arena.f32(t * lat)?;
        o.rms_norm(&kv, Some(p.vec(l, Role::MlaKvANorm)?), &c, t, lat, eps)?;
        tap(&format!("kv_cmpr-{l}"), &c)?;
        cache.copy_within(pos0 * lat * 4, &c, 0, t * lat * 4)?;
        let t_abs = self.lap(p, "MLA: projections", t_proj);
        // the absorbed queries: per head, q~ = k_b[h] . q_h
        let kb = p.vec(l, Role::MlaKB)?;
        let qt = p.arena.f32(t * nh * lat)?;
        // per token, every head's slice is contiguous, so a token's 64 heads are one batched call (oneMKL wants the
        // batches' outputs apart: rows interleaved by head only work one row at a time). Decode widths go a row at a
        // time, so a verify pass computes each row exactly as a one-token pass does.
        if t <= MMVQ_COLS {
            for r in 0..t {
                o.gemm_batch(nh, 1, lat, hd, (&q, r * nh * hd, hd, hd), (kb, 0, lat * hd), (&qt, r * nh * lat, lat, lat), false)?;
            }
        } else {
            for hh in 0..nh {
                o.gemm_at(t, lat, hd, (&q, hh * hd, nh * hd), (kb, hh * lat * hd), (&qt, hh * lat, nh * lat), false)?;
            }
        }
        let t_idx = self.lap(p, "MLA: absorbed queries", t_abs);
        // the indexer: this pass's pooled keys; the rows that see more pools than it keeps attend to its selection
        let sel = self.select(p, l, normed, &qr, idx, pos0, t, &mut *tap)?;
        let t_att = self.lap(p, "MLA: indexer", t_idx);
        let u = p.arena.f32(t * nh * lat)?;
        let kp = (g.idx_top_k / g.idx_pool) as usize;
        let scale = 1.0 / (hd as f32).sqrt();
        if t <= MMVQ_COLS || std::env::var("NS_MLA_KERNEL").is_ok_and(|v| v == "1") {
            // decode widths: the per-row kernel (a verify pass's rows equal one-token passes)
            o.mla_attend_sel(&qt, cache, &u, t, nh, lat, pos0, scale, sel.as_ref().map(|(a, b)| (a, b)), kp)?;
        } else {
            // prompt chunks as GEMMs, a block of rows at a time: each row's cells gathered contiguous, then per row
            // scores [heads, cells] = q~ . G^T, a masked softmax, latents [heads, 512] = P . G
            let nc = if sel.is_some() { 4 * kp + 3 } else { pos0 + t };
            let idx = p.arena.bytes(t * nc * 4)?;
            let cnt = p.arena.bytes(t * 4)?;
            o.mla_cells(sel.as_ref().map(|(a, b)| (a, b)), t, kp, pos0, &idx, &cnt, nc)?;
            const RB: usize = 16;
            let gb = p.arena.f32(RB * nc * lat)?;
            let sb = p.arena.f32(RB * nh * nc)?;
            for r0 in (0..t).step_by(RB) {
                let tr = RB.min(t - r0);
                o.gather(cache, &idx.view(r0 * nc * 4, tr * nc * 4)?, &gb, tr * nc, lat)?;
                o.gemm_batch(tr, nh, nc, lat, (&qt, r0 * nh * lat, lat, nh * lat), (&gb, 0, nc * lat), (&sb, 0, nc, nh * nc), false)?;
                o.softmax_masked(&sb, tr, nh, nc, &cnt.view(r0 * 4, tr * 4)?, scale)?;
                o.gemm_batch_nn(tr, nh, lat, nc, (&sb, 0, nc, nh * nc), (&gb, 0, lat, nc * lat), (&u, r0 * nh * lat, lat, nh * lat))?;
            }
        }
        let t_vb = self.lap(p, "MLA: attention", t_att);
        let vb = p.vec(l, Role::MlaVB)?;
        let oh = p.arena.f32(t * nh * hd)?;
        if t <= MMVQ_COLS {
            for r in 0..t {
                o.gemm_batch(nh, 1, hd, lat, (&u, r * nh * lat, lat, lat), (vb, 0, hd * lat), (&oh, r * nh * hd, hd, hd), false)?;
            }
        } else {
            for hh in 0..nh {
                o.gemm_at(t, hd, lat, (&u, hh * lat, nh * lat), (vb, hh * hd * lat), (&oh, hh * hd, nh * hd), false)?;
            }
        }
        tap(&format!("kqv_out-{l}"), &oh)?;
        let t_out = self.lap(p, "MLA: values (v_b)", t_vb);
        let out = p.mm(l, Role::MlaOut, &oh, t)?;
        self.lap(p, "MLA: output projection", t_out);
        tap(&format!("attn_out-{l}"), &out)?;
        Ok(out)
    }

    /// The DSA lightning indexer of MLA layer `l` (docs/glm5next.md) on x [t, d] (the attention input) and qr [t,
    /// q_lora]: the pooled keys of the pools this pass completes; then, when the last row sees more pools than the
    /// indexer keeps (idx_top_k / idx_pool = 512, past ~2,048 tokens), each row's top pools by score (`sel` [t, 512])
    /// and its count (`cnt` [t]: 512, or -1 for a row that still sees fewer: every earlier token). None: all dense.
    #[allow(clippy::too_many_arguments)]
    fn select(&self, p: &Part, l: u64, x: &DevBuf, qr: &DevBuf, idx: &Idx, pos0: usize, t: usize, tap: Tap) -> Result<Option<(DevBuf, DevBuf)>> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, hh) = (g.idx_dim as usize, g.idx_heads as usize);
        let kp = (g.idx_top_k / g.idx_pool) as usize;
        let raw = p.mm(l, Role::IdxK, x, t)?;
        let ik = p.arena.f32(t * d)?;
        o.layer_norm(&raw, p.vec(l, Role::IdxKNorm)?, p.vec(l, Role::IdxKNormBias)?, &ik, t, d, g.ln_eps as f32)?;
        tap(&format!("indexer_k-{l}"), &ik)?;
        let ig = p.mm(l, Role::IdxPoolGate, x, t)?;
        o.idx_pool(&idx.ring, &ik, &ig, &p.mat(l, Role::IdxPoolApe)?.buf, &idx.pooled, pos0, t, d)?;
        let n = (pos0 + t) / 4; // the pools the last row sees
        if n <= kp {
            return Ok(None);
        }
        let iq = p.mm(l, Role::IdxQB, qr, t)?; // [t, heads * d]
        // the head weights; their 1/sqrt(d * heads) scale is positive, so it cannot change a top-k: left out
        let w = p.mm(l, Role::IdxProj, x, t)?;
        let score = p.arena.f32(t * n)?;
        // decode widths a row at a time (a verify pass's rows equal one-token passes); prompt chunks at once
        let rows = if t <= MMVQ_COLS { 1 } else { t };
        let nc_max = (IDX_S_FLOATS / (rows * hh)).max(1).min(n);
        let sbuf = p.arena.f32(rows * hh * nc_max)?;
        for r0 in (0..t).step_by(rows) {
            let tr = rows.min(t - r0);
            let mut j0 = 0;
            while j0 < n {
                let nc = nc_max.min(n - j0);
                o.gemm_at(tr * hh, nc, d, (&iq, r0 * hh * d, d), (&idx.pooled, j0 * d), (&sbuf, 0, nc), false)?;
                o.idx_score(&sbuf, &w.view(r0 * hh * 4, tr * hh * 4)?, &score.view(r0 * n * 4, tr * n * 4)?, tr, hh, j0, nc, n, pos0 + r0)?;
                j0 += nc;
            }
        }
        tap(&format!("indexer_score-{l}"), &score)?;
        let sel = p.arena.bytes(t * kp * 4)?;
        o.topk(&score, &sel, t, n, n, kp)?;
        let cnt: Vec<i32> = (0..t).map(|r| if (pos0 + r + 1) / 4 > kp { kp as i32 } else { -1 }).collect();
        let cb = p.arena.bytes(t * 4)?;
        cb.write(0, &cnt.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(Some((sel, cb)))
    }

    /// The next `tokens` of a conversation (at most a chunk; `feed` splits longer ones): the logits of the last
    /// one. `tap` sees each named step.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.forward_rows(sess, tokens, 1, tap)?.pop().unwrap_or_default())
    }

    /// As `forward`, the logits of the last `n_out` tokens (1..=8). A pass of 2..=MAX_VERIFY tokens keeps the KDA
    /// states after each but the last (`rollback`). With the draft block loaded, the rows the previous pass left it
    /// are read first (their following token is `tokens[0]`), and this pass's rows are left to the next one.
    pub fn forward_rows(&self, sess: &mut Session, tokens: &[u32], n_out: usize, tap: Tap) -> Result<Vec<Vec<f32>>> {
        let g = &self.m.g;
        let t = tokens.len();
        if t == 0 || t > prefill_chunk() || n_out == 0 || n_out > t.min(MMVQ_COLS) {
            return Err(Error(format!("forward: {t} tokens, {n_out} outputs")));
        }
        if sess.mtp.as_ref().is_some_and(|m| m.rows > 0) {
            self.mtp_run(sess, tokens[0], false, &mut *tap)?;
        }
        let pos0 = sess.pos;
        if pos0 + t > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, t)));
        }
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let (kh, kd) = (g.kda_heads as usize, g.kda_dim as usize);
        let snapping = (2..=MAX_VERIFY).contains(&t) && self.mtp.is_some();

        // the embedding rows, on the first part's GPU
        let p0 = &self.parts[0];
        let x0 = self.embed(p0, tokens)?;
        tap("inp_embd", &x0)?;
        // the 4 streams start as copies of the embedding
        let e0 = x0.to_f32()?;
        let mut xs = vec![0f32; t * 4 * d];
        for ti in 0..t {
            for s in 0..4 {
                xs[(ti * 4 + s) * d..(ti * 4 + s + 1) * d].copy_from_slice(&e0[ti * d..(ti + 1) * d]);
            }
        }
        let x_init = DevBuf::from_f32(&p0.ops.gpu, &xs)?;
        tap("hc_init", &x_init)?;
        let mut staging = vec![0u8; t * 4 * d * 4];

        // the stream rotates through three buffers per GPU (a layer's input, after its attention, after its FFN);
        // every other temporary of a layer comes from the part's arena, reset at the layer's start
        let mut cur = usize::MAX;
        let mut ring: Vec<DevBuf> = Vec::new();
        let mut ix = 0usize;
        let mut ws: Option<[DevBuf; 6]> = None; // flat, h, normed, post, comb, pre on the current part's GPU
        for l in 0..g.n_layer {
            let pi = self.owner[l as usize];
            let p = &self.parts[pi];
            let o = &p.ops;
            let gpu = &o.gpu;
            if pi != cur {
                let next: Vec<DevBuf> = (0..3).map(|_| DevBuf::f32(gpu, t * 4 * d)).collect::<Result<_>>()?;
                if cur == usize::MAX {
                    next[0].copy_within(0, &x_init, 0, t * 4 * d * 4)?;
                } else {
                    // the stream to this part's GPU
                    self.timed(p, "GPU to GPU", || next[0].copy_from_peer(&ring[ix], &mut staging))?;
                }
                ring = next;
                ix = 0;
                ws = Some([DevBuf::f32(gpu, t * 4 * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * d)?, DevBuf::f32(gpu, t * 4)?, DevBuf::f32(gpu, t * 16)?,
                           DevBuf::f32(gpu, t * 4)?]);
                cur = pi;
            }
            p.arena.reset();
            let x = &ring[ix];
            let (x1, x2) = (&ring[(ix + 1) % 3], &ring[(ix + 2) % 3]);
            let [flat, h, normed, post, comb, pre] = ws.as_ref().unwrap();
            // before a half: the mixes, h, post, comb; then the half's norm
            let hc_pre = |fn_: Role, base: Role, scale: Role, x: &DevBuf, norm: Role| -> Result<()> {
                let w = p.mat(l, fn_)?;
                let mixes = p.arena.f32(t * 24)?;
                if w.f32 && w.rows == 24 {
                    o.hc_mix(x, &w.buf, &mixes, &p.arena.f32(t * 32 * 25)?, t, 4 * d, eps)?;
                } else {
                    o.rms_norm(x, None, flat, t, 4 * d, eps)?;
                    p.matmul(w, t, (flat, 0, w.cols), (&mixes, 0, w.rows), false)?;
                }
                o.hc_pre(&mixes, p.vec(l, scale)?, p.vec(l, base)?, x, h, post, comb, pre, t, d, g.hc_eps as f32, g.hc_iters as u32)?;
                o.rms_norm(h, Some(p.vec(l, norm)?), normed, t, d, eps)
            };

            // ---- attention half
            self.timed(p, "hc pre", || hc_pre(Role::HcAttnFn, Role::HcAttnBase, Role::HcAttnScale, x, Role::AttnNorm))?;
            tap(&format!("attn_norm-{l}"), normed)?;
            let att = self.timed(p, if g.is_mla(l) { "MLA" } else { "KDA" }, || -> Result<DevBuf> {
                if let LayerState::Kda { s: kstate, conv: cstate, snap } = &sess.layers[l as usize] {
                    let snap = snap.as_ref().filter(|_| snapping);
                    // every product of the layer's input at once (one quantization of it)
                    let k0 = self.mark(p);
                    let mut pj = p.mm_many(l, &[Role::KdaQ, Role::KdaK, Role::KdaV, Role::KdaFA, Role::KdaGA, Role::KdaBeta], normed, t)?.into_iter();
                    let k1 = self.lap(p, "KDA: input projections", k0);
                    let (pq, pk, pv, fa, ga, beta) = (pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap());
                    let conv = |pr: &DevBuf, w: Role, state: &DevBuf, sn: Option<&DevBuf>| -> Result<DevBuf> {
                        let out = p.arena.f32(t * kw)?;
                        o.conv_silu(pr, state, p.vec(l, w)?, &out, t, kw, g.kda_conv as usize, sn)?;
                        Ok(out)
                    };
                    let q = conv(&pq, Role::KdaQConv, &cstate[0], snap.map(|s| &s.1[0]))?;
                    let k = conv(&pk, Role::KdaKConv, &cstate[1], snap.map(|s| &s.1[1]))?;
                    let v = conv(&pv, Role::KdaVConv, &cstate[2], snap.map(|s| &s.1[2]))?;
                    tap(&format!("kda_q_conv-{l}"), &q)?;
                    tap(&format!("kda_k_conv-{l}"), &k)?;
                    tap(&format!("kda_v_conv-{l}"), &v)?;
                    o.l2_norm(&q, t * kh, kd, 1e-6)?;
                    o.l2_norm(&k, t * kh, kd, 1e-6)?;
                    let gate = p.mm(l, Role::KdaFB, &fa, t)?;
                    o.kda_gate(&gate, p.vec(l, Role::KdaDtBias)?, p.vec(l, Role::KdaA)?, t, kh, kd, g.kda_gate_low as f32)?;
                    tap(&format!("kda_g1-{l}"), &gate)?;
                    o.sigmoid(&beta, t * kh)?;
                    tap(&format!("kda_beta-{l}"), &beta)?;
                    let scan = p.arena.f32(t * kw)?;
                    let k2 = self.lap(p, "KDA: conv, norms, gates", k1);
                    o.kda_scan(&q, &k, &v, &gate, &beta, kstate, &scan, t, kh, kd, snap.map(|s| &s.0))?;
                    let k3 = self.lap(p, "KDA: scan", k2);
                    tap(&format!("kda_scan_out-{l}"), &scan)?;
                    let g2 = p.mm(l, Role::KdaGB, &ga, t)?;
                    tap(&format!("kda_g2-{l}"), &g2)?;
                    let y = p.arena.f32(t * kw)?;
                    o.kda_out(&scan, &g2, p.vec(l, Role::KdaONorm)?, &y, t, kh, kd, eps)?;
                    let out = p.mm(l, Role::KdaOut, &y, t)?;
                    self.lap(p, "KDA: output gate + projection", k3);
                    tap(&format!("kda_out-{l}"), &out)?;
                    Ok(out)
                } else {
                    let LayerState::Mla { c: cache, idx } = &sess.layers[l as usize] else { return Err(Error("layer state".into())) };
                    self.mla(p, l, normed, cache, idx, pos0, t, &mut *tap)
                }
            })?;
            o.hc_post(&att, x, post, comb, x1, t, d)?;
            tap(&format!("hc_attn_post-{l}"), x1)?;

            // ---- feed-forward half
            self.timed(p, "hc pre", || hc_pre(Role::HcFfnFn, Role::HcFfnBase, Role::HcFfnScale, x1, Role::FfnNorm))?;
            tap(&format!("ffn_norm-{l}"), normed)?;
            let lim = g.swiglu_limit as f32;
            let ffn = if !g.is_moe(l) {
                self.timed(p, "dense FFN", || {
                    let mut pj = p.mm_many(l, &[Role::FfnGate, Role::FfnUp], normed, t)?.into_iter();
                    let (gt, up) = (pj.next().unwrap(), pj.next().unwrap());
                    o.swiglu_clamp(&gt, &up, &gt, t * g.ffn_dense as usize, lim)?;
                    p.mm(l, Role::FfnDown, &gt, t)
                })?
            } else {
                self.moe(p, l, t, normed, &mut *tap)?
            };
            tap(&format!("ffn_out-{l}"), &ffn)?;
            o.hc_post(&ffn, x1, post, comb, x2, t, d)?;
            tap(&format!("l_out-{l}"), x2)?;
            ix = (ix + 2) % 3;
        }

        // the head, for the last n_out tokens (on the last part)
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        p.arena.reset();
        let x = &ring[ix];
        let mean = p.arena.f32(t * d)?;
        o.hc_mean(x, &mean, t, d)?;
        if let Some(ms) = &mut sess.mtp {
            // the draft block reads these rows with the tokens that follow them
            ms.hid.copy_within(0, &mean, 0, t * d * 4)?;
            ms.rows = t;
            ms.slot0 = pos0;
            ms.next = tokens[1..].to_vec();
        }
        let last = mean.view((t - n_out) * d * 4, n_out * d * 4)?;
        let out = p.arena.f32(n_out * d)?;
        o.rms_norm(&last, Some(p.vec(0, Role::OutputNorm)?), &out, n_out, d, eps)?;
        tap("result_norm", &out)?;
        let logits = p.mm(0, Role::Output, &out, n_out)?;
        tap("result_output", &logits)?;
        sess.pos += t;
        sess.snapped = snapping.then_some((pos0, t));
        let v = logits.to_f32()?;
        let vocab = g.n_vocab as usize;
        Ok(v.chunks(vocab).map(|c| c.to_vec()).collect())
    }

    /// Keeps only the first `keep` tokens of the last forward pass (a verify pass of 2..=MAX_VERIFY): the KDA
    /// states back to their snapshots after row `keep - 1`, the position back, the draft block's rows cut. The MLA
    /// caches need nothing: the position bounds what is read, later rows overwrite the rest.
    pub fn rollback(&self, sess: &mut Session, keep: usize) -> Result<()> {
        let (pos0, t) = sess.snapped.take().ok_or("rollback: the last pass kept no snapshots")?;
        if keep == 0 || keep >= t {
            return Err(Error(format!("rollback: keep {keep} of {t} rows")));
        }
        let g = &self.m.g;
        let sz = (g.kda_heads * g.kda_dim * g.kda_dim) as usize * 4;
        let cw = (g.kda_conv as usize - 1) * (g.kda_heads * g.kda_dim) as usize * 4;
        for l in &sess.layers {
            if let LayerState::Kda { s, conv, snap: Some(sn) } = l {
                let (ss, sc) = &**sn;
                s.copy_within(0, ss, (keep - 1) * sz, sz)?;
                for (c, scn) in conv.iter().zip(sc) {
                    c.copy_within(0, scn, (keep - 1) * cw, cw)?;
                }
            }
        }
        sess.pos = pos0 + keep;
        if let Some(ms) = &mut sess.mtp {
            ms.rows = keep;
            ms.next.truncate(keep - 1);
        }
        Ok(())
    }

    /// The draft block over the rows the last forward pass left it, each with the token that follows it (`next`
    /// for the last row): their slots of its cache written; with `head`, the draft logits after the last row.
    fn mtp_run(&self, sess: &mut Session, next: u32, head: bool, tap: Tap) -> Result<Option<Vec<f32>>> {
        let (Some(ml), Some(ms)) = (self.mtp, sess.mtp.as_mut()) else { return Ok(None) };
        if ms.rows == 0 {
            return Ok(None);
        }
        let g = &self.m.g;
        let (n, d, eps) = (ms.rows, g.n_embd as usize, g.rms_eps as f32);
        let mut toks = std::mem::take(&mut ms.next);
        toks.push(next);
        let (slot0, rows) = (ms.slot0, n);
        ms.rows = 0;
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        let r = self.timed(p, "MTP draft", || -> Result<Option<Vec<f32>>> {
            p.arena.reset();
            let emb = self.embed(p, &toks)?;
            let en = p.arena.f32(n * d)?;
            o.rms_norm(&emb, Some(p.vec(ml, Role::MtpENorm)?), &en, n, d, eps)?;
            let hn = p.arena.f32(n * d)?;
            o.rms_norm(&ms.hid, Some(p.vec(ml, Role::MtpHNorm)?), &hn, n, d, eps)?;
            // eh_proj . concat(en, hn) = W_e . en + W_h . hn
            let [we, wh] = p.eh.as_ref().ok_or("the MTP block's eh_proj is not loaded")?;
            let cur = p.arena.f32(n * d)?;
            p.matmul(we, n, (&en, 0, d), (&cur, 0, d), false)?;
            p.matmul(wh, n, (&hn, 0, d), (&cur, 0, d), true)?;
            tap("mtp_eh", &cur)?;
            let an = p.arena.f32(n * d)?;
            o.rms_norm(&cur, Some(p.vec(ml, Role::AttnNorm)?), &an, n, d, eps)?;
            let att = self.mla(p, ml, &an, &ms.cache, &ms.idx, slot0, rows, &mut *tap)?;
            o.add(&cur, &att, n * d)?;
            let fnm = p.arena.f32(n * d)?;
            o.rms_norm(&cur, Some(p.vec(ml, Role::FfnNorm)?), &fnm, n, d, eps)?;
            let y = self.moe(p, ml, n, &fnm, &mut *tap)?;
            o.add(&y, &cur, n * d)?;
            tap("mtp_out", &y)?;
            if !head {
                return Ok(None);
            }
            let last = y.view((n - 1) * d * 4, d * 4)?;
            let hn2 = p.arena.f32(d)?;
            o.rms_norm(&last, Some(p.vec(ml, Role::MtpHeadNorm)?), &hn2, 1, d, eps)?;
            Ok(Some(p.mm(0, Role::Output, &hn2, 1)?.to_f32()?))
        })?;
        Ok(r)
    }

    /// Generation from a prompt `feed` returned `logits` for; `mtp`: draft with the MTP block (when loaded).
    pub fn decoder(&self, logits: Vec<f32>, mtp: bool) -> Decoder {
        Decoder { logits, next: None, draft: None, mtp: mtp && self.mtp.is_some(), drafted: 0, accepted: 0 }
    }

    /// The next committed token(s): one, or two when a draft is accepted. `sample` draws a token from logits. With
    /// MTP a draft is accepted exactly when the token drawn for its position is the draft itself, so what is
    /// committed is distributed as plain sampling would be.
    pub fn step(&self, sess: &mut Session, dec: &mut Decoder, sample: &mut dyn FnMut(&[f32]) -> u32, tap: Tap) -> Result<Vec<u32>> {
        let fits = sess.pos + MAX_VERIFY <= sess.max_ctx;
        match (dec.next, dec.draft) {
            (Some(x), Some(d)) if dec.mtp && fits => {
                let rows = self.forward_rows(sess, &[x, d], 2, &mut *tap)?;
                let y0 = sample(&rows[0]);
                dec.drafted += 1;
                let (out, tail) = if y0 == d {
                    dec.accepted += 1;
                    let y1 = sample(&rows[1]);
                    (vec![d, y1], y1)
                } else {
                    self.rollback(sess, 1)?;
                    (vec![y0], y0)
                };
                dec.draft = self.mtp_run(sess, tail, true, &mut *tap)?.map(|l| argmax(&l));
                dec.next = Some(tail);
                Ok(out)
            }
            _ => {
                if let Some(x) = dec.next.take() {
                    dec.logits = self.forward(sess, &[x], &mut *tap)?;
                }
                let y = sample(&dec.logits);
                dec.draft = if dec.mtp { self.mtp_run(sess, y, true, &mut *tap)?.map(|l| argmax(&l)) } else { None };
                dec.next = Some(y);
                Ok(vec![y])
            }
        }
    }

    /// The MoE half of layer `l` on x [t, d]: the router (on the host), the shared expert, the routed experts
    /// grouped by expert (made resident together first).
    fn moe(&self, p: &Part, l: u64, t: usize, x: &DevBuf, tap: Tap) -> Result<DevBuf> {
        let g = &self.m.g;
        let o = &p.ops;
        let (d, ne, used) = (g.n_embd as usize, g.n_expert as usize, g.n_expert_used as usize);
        let lim = g.swiglu_limit as f32;
        let logits = self.timed(p, "router", || p.mm(l, Role::Router, x, t))?;
        tap(&format!("ffn_moe_logits-{l}"), &logits)?;
        let lv = logits.to_f32()?;
        let bias = p.vec(l, Role::RouterBias)?.to_f32()?;
        // expert -> (token, weight)
        let mut by: BTreeMap<usize, Vec<(i32, f32)>> = BTreeMap::new();
        for ti in 0..t {
            let pr: Vec<f32> = lv[ti * ne..(ti + 1) * ne].iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|a, b| (pr[*b] + bias[*b]).total_cmp(&(pr[*a] + bias[*a])));
            let sel = &order[..used];
            let sum: f32 = sel.iter().map(|i| pr[*i]).sum::<f32>().max(6.103_516e-5); // the smallest normal half, as llama.cpp clamps
            for i in sel {
                let w = if g.expert_norm { pr[*i] / sum } else { pr[*i] } * g.expert_scale as f32;
                by.entry(*i).or_default().push((ti as i32, w));
            }
        }
        // the shared expert first, then each routed expert added into it
        let y = self.timed(p, "shared expert", || {
            let mut pj = p.mm_many(l, &[Role::ShGate, Role::ShUp], x, t)?.into_iter();
            let (sg, su) = (pj.next().unwrap(), pj.next().unwrap());
            o.swiglu_clamp(&sg, &su, &sg, t * g.ffn_expert as usize * g.n_expert_shared as usize, lim)?;
            p.mm(l, Role::ShDown, &sg, t)
        })?;
        let need: Vec<u64> = by.keys().map(|e| *e as u64).collect();
        // prompt chunks read host-slot experts in place; decode makes them resident (NS_DECODE_DIRECT=1: in place too)
        let promote = t <= MMVQ_COLS && !std::env::var("NS_DECODE_DIRECT").is_ok_and(|v| v == "1");
        let slots = self.timed(p, "expert misses (swaps / file)", || p.ensure(&self.m, l, &need, promote))?;
        let parts = expert_parts(&self.m, l)?;
        let (f, cols) = (g.ffn_expert as usize, d);
        self.timed(p, "routed experts", || {
            // the entries grouped by expert (rows of xe / gt / ut / dn), and per token its entries (the combine):
            // one upload of the layer's routing
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
            wb.write(0, &wts.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            // the grouped kernels (two launches for the layer), when they take this layer's types
            // Big prompt chunks: each expert expanded to fp16 once a chunk and its tokens multiplied by oneMKL's half
            // GEMM (the XMX units) - Strata's prompt path. The expansion is the cost and the chunk amortizes it, so
            // from NS_PROMPT_F16_MIN tokens (default 1024; 0 = never) - below, the grouped kernels win (measured
            // 2.0x slower at 512). Memory stays one expert's worth however big the chunk: its tokens gathered to
            // fp16, its outputs added into y.
            static F16_MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let f16_min = *F16_MIN.get_or_init(|| std::env::var("NS_PROMPT_F16_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1024));
            if f16_min > 0 && t >= f16_min && t > MMVQ_COLS {
                let mut gtok: Vec<i32> = Vec::with_capacity(total);
                let mut gw: Vec<f32> = Vec::with_capacity(total);
                for list in by.values() {
                    for (ti, w) in list {
                        gtok.push(*ti);
                        gw.push(*w);
                    }
                }
                let tb = p.arena.bytes(total * 4)?;
                tb.write(0, &gtok.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let wtb = p.arena.f32(total)?;
                wtb.write(0, &gw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let nmax = by.values().map(|l| l.len()).max().unwrap_or(1);
                let xh = p.arena.bytes(nmax * d * 2)?;
                let w16 = p.arena.bytes(2 * f * d * 2)?; // gate | up of one expert; down reuses the first half
                let wu = w16.view(f * d * 2, f * d * 2)?;
                let g32 = p.arena.f32(nmax * f)?;
                let u32b = p.arena.f32(nmax * f)?;
                let hh = p.arena.bytes(nmax * f * 2)?;
                let dn = p.arena.f32(nmax * d)?;
                // an expert in pinned host memory is copied into a VRAM staging slot first (the copy engine moves it
                // at full PCIe speed; the expansion kernel reading it in place moved a few bytes a transaction)
                let sb = parts.down.0 + parts.down.1;
                let ring = [p.arena.bytes(sb)?, p.arena.bytes(sb)?];
                let mut row = 0;
                for (k, (list, (hbuf, host))) in by.values().zip(&slots).enumerate() {
                    let n = list.len();
                    let toks = tb.view(row * 4, n * 4)?;
                    let buf = if *host {
                        ring[k % 2].copy_within(0, hbuf, 0, sb)?;
                        &ring[k % 2]
                    } else {
                        hbuf
                    };
                    let m0 = self.mark(p);
                    o.gather_f16(x, &toks, &xh, n, d)?;
                    let m1 = self.lap(p, "MoE f16: gather", m0);
                    o.dequant_f16(parts.gate.2.code(), buf, parts.gate.0, parts.gate.1, f * d, &w16)?;
                    o.dequant_f16(parts.up.2.code(), buf, parts.up.0, parts.up.1, f * d, &wu)?;
                    let m2 = self.lap(p, "MoE f16: expand gate/up", m1);
                    o.gemm_f16(n, f, d, (&xh, 0, d), (&w16, 0), (&g32, 0, f), false)?;
                    o.gemm_f16(n, f, d, (&xh, 0, d), (&wu, 0), (&u32b, 0, f), false)?;
                    let m3 = self.lap(p, "MoE f16: gemm gate/up", m2);
                    o.swiglu_clamp(&g32, &u32b, &g32, n * f, lim)?;
                    o.to_f16(&g32, &hh, n * f)?;
                    let m4 = self.lap(p, "MoE f16: swiglu", m3);
                    o.dequant_f16(parts.down.2.code(), buf, parts.down.0, parts.down.1, d * f, &w16)?;
                    let m5 = self.lap(p, "MoE f16: expand down", m4);
                    o.gemm_f16(n, d, f, (&hh, 0, f), (&w16, 0), (&dn, 0, d), false)?;
                    let m6 = self.lap(p, "MoE f16: gemm down", m5);
                    o.scatter_add(&y, &dn, &toks, &wtb.view(row * 4, n * 4)?, n, d)?;
                    self.lap(p, "MoE f16: scatter", m6);
                    row += n;
                }
                return Ok(());
            }
            // NS_PROMPT_DEQUANT=1: prompt chunks take the float32 expanded-GEMM path instead (a measurement switch)
            let dequant = t > MMVQ_COLS && std::env::var("NS_PROMPT_DEQUANT").is_ok_and(|v| v == "1");
            if !dequant && parts.gate.2 == parts.up.2 && o.moe_grouped_supported(parts.gate.2.code(), parts.down.2.code(), d, f) {
                let groups = by.len();
                let mut table: Vec<u8> = Vec::with_capacity(groups * 8 + (groups + 2 + 2 * total) * 4);
                for (slot, _) in &slots {
                    table.extend((slot.ptr() as u64).to_le_bytes());
                }
                let mut start = 0i32;
                table.extend(start.to_le_bytes());
                for list in by.values() {
                    start += list.len() as i32;
                    table.extend(start.to_le_bytes());
                }
                table.extend((groups as i32).to_le_bytes());
                for e in 0..total as i32 {
                    table.extend(e.to_le_bytes()); // ent_dst: an entry's own row
                }
                for ti in &tok {
                    table.extend(ti.to_le_bytes());
                }
                let tb = p.arena.bytes(table.len())?;
                tb.write(0, &table)?;
                let xq = p.arena.bytes(o.q8_1_bytes(d, t))?;
                o.quantize_q8_1((x, 0), &xq, d, t)?;
                let scratch = p.arena.bytes(o.moe_scratch_bytes(total, f))?;
                let dn = p.arena.f32(total * d)?;
                // prompt chunks a sub-group a row (NS_PROMPT_LANES), decode the kernels' default
                static PROMPT_LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let pl = *PROMPT_LANES.get_or_init(|| std::env::var("NS_PROMPT_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(32));
                let lanes = if t > MMVQ_COLS { pl } else { 0 };
                o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, total, &xq, &scratch, &dn, lim, lanes)?;
                let cb = p.arena.bytes((t + 1 + total) * 4)?;
                cb.write(0, &t_ptr.iter().chain(&ent).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                return o.moe_combine(&y, &dn, &cb, &wb, t, total, d);
            }
            let ib = p.arena.bytes(ints.len() * 4)?;
            ib.write(0, &ints.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
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
            // the combine reads t_ptr and ent behind the entries' token list in `ib`
            let ints_view = p.arena.bytes((t + 1 + total) * 4)?;
            ints_view.copy_within(0, &ib, base * 4, (t + 1 + total) * 4)?;
            o.moe_combine(&y, &dn, &ints_view, &wb, t, total, d)
        })?;
        Ok(y)
    }
}
