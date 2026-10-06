//! GLM-5.3-Flash on one or more GPUs: every step of `docs/glm5next.md` in order, float32 activations, the kernels
//! of `kernels/ns`. The layers are split over the GPUs (a `Part` each: its layers' weights in their stored form,
//! its own expert cache in VRAM and its own share of the pinned host mirror - pinned memory belongs to one GPU's
//! context); the token stream [T, 4, hidden] crosses to the next GPU through host memory where the layers do. Up
//! to 8 rows multiply from the stored blocks (decode), more are expanded in chunks. MLA attends to every earlier
//! token (the indexer selects all of them up to ~2,048 tokens of context, docs/glm5next.md).

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
}

enum LayerState {
    Kda { s: DevBuf, conv: [DevBuf; 3] },
    Mla { c: DevBuf },
}

pub struct Glm<'g> {
    pub m: Model<'g>,
    pub parts: Vec<Part>,
    /// layer -> part
    owner: Vec<usize>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// NS_PROFILE=1: seconds and calls per section (the GPU synced at each boundary)
    prof: Option<Mutex<BTreeMap<&'static str, (f64, u64)>>>,
}

const VECTORS: [Role; 18] = [Role::OutputNorm, Role::AttnNorm, Role::FfnNorm, Role::HcAttnBase, Role::HcAttnScale, Role::HcFfnBase, Role::HcFfnScale, Role::KdaQConv,
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
    fn load(m: &Model, gpu: &Arc<Gpu>, layers: Range<u64>, last: bool, expert_bytes: Option<usize>, mirror_bytes: usize,
            log: &mut dyn FnMut(String)) -> Result<Part> {
        let ops = Ops { gpu: gpu.clone() };
        let scratch = DevBuf::f32(gpu, SCRATCH)?;
        let mut mats = BTreeMap::new();
        let mut vecs = BTreeMap::new();
        let mut bytes = 0u64;
        let mut todo: Vec<(u64, Vec<Role>)> = Vec::new();
        if last {
            todo.push((0, vec![Role::OutputNorm, Role::Output]));
        }
        todo.extend(layers.clone().map(|l| (l, m.roles(l))));
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
        let q8 = DevBuf::new(gpu, ops.q8_1_bytes(MAX_COLS, MMVQ_COLS))?;
        // the store: slots of the largest layer's [gate | up | down], in chunks; VRAM, then pinned host memory
        let moe: Vec<u64> = layers.clone().filter(|l| m.g.is_moe(*l)).collect();
        let n_exp = moe.len() * m.g.n_expert as usize;
        let slot_bytes = moe.iter().map(|l| m.expert_bytes(*l) as usize).max().unwrap_or(256).next_multiple_of(256);
        let budget = match expert_bytes {
            Some(b) => b,
            None => gpu.memory()?.1.map_or(8usize << 30, |f| (f as usize).saturating_sub(3 << 30)),
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
        let store = Store { vram, host, slot_bytes, per_chunk, loc, vowner, vused: vec![0; nv], rfree, tick: 0, hits: 0, misses: 0, from_host: 0 };
        let arena = Arena::new(gpu, 1 << 30)?;
        Ok(Part { ops, arena, layers, mats, vecs, scratch, q8, experts: Mutex::new(store), expert_slots: nv, host_slots: nr, weight_bytes: bytes })
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

    /// Every expert of `need` (layer `l`) in VRAM at once, none evicting another: a miss swaps with the least
    /// recently used VRAM expert (it goes down to a free host slot, the wanted one comes up from its host slot or
    /// the file). Their VRAM slots, (chunk, byte offset), in `need`'s order.
    fn ensure(&self, m: &Model, l: u64, need: &[u64]) -> Result<Vec<(usize, usize)>> {
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
        Ok(need.iter().map(|ex| match c.loc[&(l, *ex)] {
            Loc::V(s) => c.vat(s),
            Loc::R(_) => (0, 0), // not reached: every needed expert is in VRAM now
        }).collect())
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
    /// layers.
    pub fn load(file: &'g Gguf, gpus: &[Arc<Gpu>], expert_bytes: Option<usize>, mirror_bytes: Option<usize>, log: &mut dyn FnMut(String)) -> Result<Glm<'g>> {
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
            let share = mirror * (range.end - range.start) as usize / n as usize;
            parts.push(Part::load(&m, gpu, range, i + 1 == gpus.len(), expert_bytes, share, log)?);
        }
        let load_bytes = parts.iter().map(|p| p.weight_bytes).sum();
        let prof = std::env::var("NS_PROFILE").is_ok_and(|v| v == "1").then(|| Mutex::new(BTreeMap::new()));
        Ok(Glm { m, parts, owner, load_seconds: t0.elapsed().as_secs_f64(), load_bytes, prof })
    }

    pub fn expert_slots(&self) -> usize {
        self.parts.iter().map(|p| p.expert_slots).sum()
    }

    /// (VRAM hits, misses, misses served from pinned host memory) of the expert stores so far
    pub fn expert_stats(&self) -> (u64, u64, u64) {
        self.parts.iter().fold((0, 0, 0), |a, p| {
            let c = p.experts.lock().unwrap();
            (a.0 + c.hits, a.1 + c.misses, a.2 + c.from_host)
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

    /// The profile so far: (section, seconds, calls), slowest first.
    pub fn profile(&self) -> Vec<(&'static str, f64, u64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let mut v: Vec<_> = p.lock().unwrap().iter().map(|(k, (s, n))| (*k, *s, *n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// A new conversation, up to `max_ctx` tokens (MLA attends to every earlier token: exact up to ~2,048).
    pub fn session(&self, max_ctx: usize) -> Result<Session> {
        let g = &self.m.g;
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let mut layers = Vec::new();
        for l in 0..g.n_layer {
            let gpu = &self.parts[self.owner[l as usize]].ops.gpu;
            layers.push(if g.is_mla(l) {
                LayerState::Mla { c: DevBuf::f32(gpu, max_ctx * g.kv_lora as usize)? }
            } else {
                let s = DevBuf::f32(gpu, (g.kda_heads * g.kda_dim * g.kda_dim) as usize)?;
                s.fill(0)?;
                let mk = || -> Result<DevBuf> {
                    let b = DevBuf::f32(gpu, (g.kda_conv as usize - 1) * kw)?;
                    b.fill(0)?;
                    Ok(b)
                };
                LayerState::Kda { s, conv: [mk()?, mk()?, mk()?] }
            });
        }
        Ok(Session { pos: 0, max_ctx, layers })
    }

    /// The next `tokens` of a conversation: the logits of the last one. `tap` sees each named step.
    pub fn forward(&self, sess: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        let g = &self.m.g;
        let t = tokens.len();
        let pos0 = sess.pos;
        if pos0 + t > sess.max_ctx {
            return Err(Error(format!("the context is {} tokens; {} more do not fit", sess.max_ctx, t)));
        }
        let (d, eps) = (g.n_embd as usize, g.rms_eps as f32);
        let kw = (g.kda_heads * g.kda_dim) as usize;
        let (kh, kd) = (g.kda_heads as usize, g.kda_dim as usize);

        // the embedding rows, read from the file, expanded on the first part's GPU
        let p0 = &self.parts[0];
        let emb = self.m.tensor(0, Role::TokenEmbd).ok_or("token_embd missing")?;
        let rb = emb.ty.bytes(g.n_embd).unwrap_or(0) as usize;
        let mut rows = vec![0u8; t * rb];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as u64 >= g.n_vocab {
                return Err(Error(format!("token {tok} outside the vocabulary")));
            }
            self.m.file.read_into(emb, *tok as u64 * rb as u64, &mut rows[i * rb..(i + 1) * rb]).map_err(e)?;
        }
        let raw = DevBuf::new(&p0.ops.gpu, rows.len())?;
        raw.write(0, &rows)?;
        let x0 = DevBuf::f32(&p0.ops.gpu, t * d)?;
        p0.ops.dequant(emb.ty.code(), &raw, 0, rows.len(), t * d, &x0)?;
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
                Ok(if let LayerState::Kda { s: kstate, conv: cstate } = &sess.layers[l as usize] {
                    // every product of the layer's input at once (one quantization of it)
                    let mut pj = p.mm_many(l, &[Role::KdaQ, Role::KdaK, Role::KdaV, Role::KdaFA, Role::KdaGA, Role::KdaBeta], normed, t)?.into_iter();
                    let (pq, pk, pv, fa, ga, beta) = (pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap(), pj.next().unwrap());
                    let conv = |pr: &DevBuf, w: Role, state: &DevBuf| -> Result<DevBuf> {
                        let out = p.arena.f32(t * kw)?;
                        o.conv_silu(pr, state, p.vec(l, w)?, &out, t, kw, g.kda_conv as usize)?;
                        Ok(out)
                    };
                    let q = conv(&pq, Role::KdaQConv, &cstate[0])?;
                    let k = conv(&pk, Role::KdaKConv, &cstate[1])?;
                    let v = conv(&pv, Role::KdaVConv, &cstate[2])?;
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
                    o.kda_scan(&q, &k, &v, &gate, &beta, kstate, &scan, t, kh, kd)?;
                    tap(&format!("kda_scan_out-{l}"), &scan)?;
                    let g2 = p.mm(l, Role::KdaGB, &ga, t)?;
                    tap(&format!("kda_g2-{l}"), &g2)?;
                    let y = p.arena.f32(t * kw)?;
                    o.kda_out(&scan, &g2, p.vec(l, Role::KdaONorm)?, &y, t, kh, kd, eps)?;
                    let out = p.mm(l, Role::KdaOut, &y, t)?;
                    tap(&format!("kda_out-{l}"), &out)?;
                    out
                } else {
                    let LayerState::Mla { c: cache } = &sess.layers[l as usize] else { return Err(Error("layer state".into())) };
                    let (nh, hd, lat) = (g.n_head as usize, g.head_dim as usize, g.kv_lora as usize);
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
                    // the absorbed queries: per head, q~ = k_b[h] . q_h
                    let kb = p.vec(l, Role::MlaKB)?;
                    let qt = p.arena.f32(t * nh * lat)?;
                    // one token: every head's slice is contiguous, so the 64 heads are one batched call (oneMKL wants
                    // the batches' outputs apart: rows interleaved by head only work at one row)
                    if t == 1 {
                        o.gemm_batch(nh, 1, lat, hd, (&q, 0, hd, hd), (kb, 0, lat * hd), (&qt, 0, lat, lat), false)?;
                    } else {
                        for hh in 0..nh {
                            o.gemm_at(t, lat, hd, (&q, hh * hd, nh * hd), (kb, hh * lat * hd), (&qt, hh * lat, nh * lat), false)?;
                        }
                    }
                    let u = p.arena.f32(t * nh * lat)?;
                    o.mla_attend(&qt, cache, &u, t, nh, lat, pos0, 1.0 / (hd as f32).sqrt())?;
                    let vb = p.vec(l, Role::MlaVB)?;
                    let oh = p.arena.f32(t * nh * hd)?;
                    if t == 1 {
                        o.gemm_batch(nh, 1, hd, lat, (&u, 0, lat, lat), (vb, 0, hd * lat), (&oh, 0, hd, hd), false)?;
                    } else {
                        for hh in 0..nh {
                            o.gemm_at(t, hd, lat, (&u, hh * lat, nh * lat), (vb, hh * hd * lat), (&oh, hh * hd, nh * hd), false)?;
                        }
                    }
                    tap(&format!("kqv_out-{l}"), &oh)?;
                    let out = p.mm(l, Role::MlaOut, &oh, t)?;
                    tap(&format!("attn_out-{l}"), &out)?;
                    out
                })
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

        // the head, for the last token (on the last part)
        let p = self.parts.last().unwrap();
        let o = &p.ops;
        p.arena.reset();
        let x = &ring[ix];
        let mean = p.arena.f32(t * d)?;
        o.hc_mean(x, &mean, t, d)?;
        let last = p.arena.f32(d)?;
        last.copy_within(0, &mean, (t - 1) * d * 4, d * 4)?;
        let out = p.arena.f32(d)?;
        o.rms_norm(&last, Some(p.vec(0, Role::OutputNorm)?), &out, 1, d, eps)?;
        tap("result_norm", &out)?;
        let logits = p.mm(0, Role::Output, &out, 1)?;
        tap("result_output", &logits)?;
        sess.pos += t;
        logits.to_f32()
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
        let slots = self.timed(p, "expert misses (swaps / file)", || p.ensure(&self.m, l, &need))?;
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
            if parts.gate.2 == parts.up.2 && o.moe_grouped_supported(parts.gate.2.code(), parts.down.2.code(), d, f) {
                let groups = by.len();
                let c = p.experts.lock().unwrap();
                let mut table: Vec<u8> = Vec::with_capacity(groups * 8 + (groups + 2 + 2 * total) * 4);
                for slot in &slots {
                    table.extend((c.vram[slot.0].ptr() as u64 + slot.1 as u64).to_le_bytes());
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
                o.moe_grouped(parts.gate.2.code(), parts.down.2.code(), d, f, &tb, groups, total, &xq, &scratch, &dn, lim)?;
                drop(c);
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
            let c = p.experts.lock().unwrap();
            let mut row = 0;
            for (list, slot) in by.values().zip(&slots) {
                let n = list.len();
                let buf = &c.vram[slot.0];
                p.expert_into(buf, slot.1, parts.gate, f, cols, (&xe, row * cols), (&gt, row * f), n, true)?;
                p.expert_into(buf, slot.1, parts.up, f, cols, (&xe, row * cols), (&ut, row * f), n, false)?;
                row += n;
            }
            o.swiglu_clamp(&gt, &ut, &gt, total * f, lim)?;
            let mut row = 0;
            for (list, slot) in by.values().zip(&slots) {
                let n = list.len();
                p.expert_into(&c.vram[slot.0], slot.1, parts.down, d, f, (&gt, row * f), (&dn, row * d), n, true)?;
                row += n;
            }
            drop(c);
            // the combine reads t_ptr and ent behind the entries' token list in `ib`
            let ints_view = p.arena.bytes((t + 1 + total) * 4)?;
            ints_view.copy_within(0, &ib, base * 4, (t + 1 + total) * 4)?;
            o.moe_combine(&y, &dn, &ints_view, &wb, t, total, d)
        })?;
        Ok(y)
    }
}
