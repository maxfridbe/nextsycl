//! The routed experts of one GPU's layers in three tiers: VRAM slots, pinned host memory slots, and the file. Each
//! expert is in at most one slot (exclusive). A decode request makes its experts VRAM-resident: a miss takes the least
//! recently used VRAM expert's slot - that expert goes down to a host slot - and brings the wanted one up from its
//! host slot, or from the file through one. A host slot for either comes from the free list or from the least
//! recently used host expert, which goes back to being on the file only. Prompt passes read host experts where they
//! are (the caller copies them through a staging ring) and bring file-only ones into host slots.
//!
//! The file tier is read by a pool of threads (`Reader`) with plain preads into the pinned slots, 1 MiB a job, so a
//! layer's misses - and the next layer's guessed experts, read ahead (`prefetch`) - are on the disk at once (AirLLM's
//! loading overlapped with compute); the pages are dropped from the page cache behind them (the pinned slots are the
//! cache). A pinned slot the copy engine may still be reading is waited for on the host before a read overwrites it.

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use nextsycl_core::{DevBuf, Error, HostBuf, Result};
use nextsycl_gguf::{GType, Gguf};

use crate::model::{Model, Role};
use crate::ops::Ops;

/// Where a layer's [gate | up | down] sit in a slot: (byte offset, bytes, type) each, and in the file: (shard,
/// absolute offset of expert 0, bytes an expert) each
#[derive(Clone, Copy, Debug)]
pub struct ExpertParts {
    pub gate: (usize, usize, GType),
    pub up: (usize, usize, GType),
    pub down: (usize, usize, GType),
    file: [(usize, u64, usize); 3],
}

impl ExpertParts {
    pub fn of(m: &Model, l: u64) -> Result<ExpertParts> {
        let mut off = 0;
        let mut file = [(0usize, 0u64, 0usize); 3];
        let mut part = |k: usize, r: Role| -> Result<(usize, usize, GType)> {
            let t = m.tensor(l, r).ok_or_else(|| Error(format!("block {l}: {r:?} missing")))?;
            let b = (t.bytes / m.g.n_expert) as usize;
            file[k] = (t.shard, t.offset, b);
            let p = (off, b, t.ty);
            off += b;
            Ok(p)
        };
        let (gate, up, down) = (part(0, Role::ExpGate)?, part(1, Role::ExpUp)?, part(2, Role::ExpDown)?);
        Ok(ExpertParts { gate, up, down, file })
    }
    pub fn bytes(&self) -> usize {
        self.down.0 + self.down.1
    }
}

/// A batch of reads: how many jobs are left, the first error
pub struct Done {
    left: AtomicUsize,
    err: Mutex<Option<String>>,
    m: Mutex<()>,
    cv: Condvar,
}

impl Done {
    fn new(n: usize) -> Arc<Done> {
        Arc::new(Done { left: AtomicUsize::new(n), err: Mutex::new(None), m: Mutex::new(()), cv: Condvar::new() })
    }
    fn finish(&self, e: Option<String>) {
        if let Some(e) = e {
            self.err.lock().unwrap().get_or_insert(e);
        }
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = self.m.lock().unwrap();
            self.cv.notify_all();
        }
    }
    pub fn wait(&self) -> Result<()> {
        let mut g = self.m.lock().unwrap();
        while self.left.load(Ordering::Acquire) > 0 {
            g = self.cv.wait(g).unwrap();
        }
        drop(g);
        match self.err.lock().unwrap().clone() {
            Some(e) => Err(Error(e)),
            None => Ok(()),
        }
    }
    pub fn ready(&self) -> bool {
        self.left.load(Ordering::Acquire) == 0
    }
}

struct Job {
    shard: usize,
    offset: u64,
    len: usize,
    dst: usize,
    done: Arc<Done>,
}

/// The readers' work: what a layer waits for now first, then what is read ahead
#[derive(Default)]
struct Queue {
    now: std::collections::VecDeque<Job>,
    ahead: std::collections::VecDeque<Job>,
    closed: bool,
}

/// The file tier's readers: NS_DS2_READERS threads (default 16) with their own handles on the model's files
pub struct Reader {
    q: Arc<(Mutex<Queue>, Condvar)>,
    pub bytes: AtomicUsize,
}

fn readers() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS2_READERS").ok().and_then(|v| v.parse().ok()).unwrap_or(16).clamp(1, 64))
}

extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
const POSIX_FADV_DONTNEED: i32 = 4;

impl Drop for Reader {
    fn drop(&mut self) {
        self.q.0.lock().unwrap().closed = true;
        self.q.1.notify_all();
    }
}

impl Reader {
    pub fn new(f: &Gguf) -> Result<Arc<Reader>> {
        let files: Vec<std::fs::File> = f.paths.iter().map(std::fs::File::open).collect::<std::io::Result<_>>().map_err(|e| Error(format!("opening the model: {e}")))?;
        let files = Arc::new(files);
        let q: Arc<(Mutex<Queue>, Condvar)> = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        // NS_DS2_KEEP_PAGES=1: the read pages stay in the page cache (by default dropped: the pinned slots are the cache)
        let drop_pages = !std::env::var("NS_DS2_KEEP_PAGES").is_ok_and(|v| v == "1");
        for _ in 0..readers() {
            let (q, files) = (q.clone(), files.clone());
            std::thread::spawn(move || loop {
                let job = {
                    let mut g = q.0.lock().unwrap();
                    loop {
                        if let Some(j) = g.now.pop_front().or_else(|| g.ahead.pop_front()) {
                            break j;
                        }
                        if g.closed {
                            return;
                        }
                        g = q.1.wait(g).unwrap();
                    }
                };
                // SAFETY: dst..dst + len is a pinned slot's range the store reserved for this read (nothing else
                // writes or reads it until `done` says so).
                let buf = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, job.len) };
                let f = &files[job.shard];
                let r = f.read_exact_at(buf, job.offset).map_err(|e| format!("reading an expert: {e}"));
                if drop_pages {
                    // SAFETY: a plain advisory call on an open descriptor.
                    unsafe { posix_fadvise(f.as_raw_fd(), job.offset as i64, job.len as i64, POSIX_FADV_DONTNEED) };
                }
                job.done.finish(r.err());
            });
        }
        Ok(Arc::new(Reader { q, bytes: AtomicUsize::new(0) }))
    }

    /// Expert `ex` of a layer laid out by `p` into host memory at `dst` (a slot): its batch. `now`: ahead of the
    /// reads queued for later layers
    fn read_expert(&self, p: &ExpertParts, ex: u64, dst: *mut u8, now: bool) -> Arc<Done> {
        const PIECE: usize = 1 << 20;
        let mut jobs = Vec::new();
        for (k, (shard, base, n)) in p.file.iter().enumerate() {
            let at = [p.gate.0, p.up.0, p.down.0][k];
            let mut o = 0;
            while o < *n {
                let len = PIECE.min(n - o);
                jobs.push((*shard, base + ex * *n as u64 + o as u64, len, dst as usize + at + o));
                o += len;
            }
        }
        let done = Done::new(jobs.len());
        self.bytes.fetch_add(p.bytes(), Ordering::Relaxed);
        let mut g = self.q.0.lock().unwrap();
        for (shard, offset, len, dst) in jobs {
            let j = Job { shard, offset, len, dst, done: done.clone() };
            if now { g.now.push_back(j) } else { g.ahead.push_back(j) }
        }
        drop(g);
        self.q.1.notify_all();
        done
    }

    /// A batch read ahead that a layer now waits for: its jobs still queued move to the front
    fn hurry(&self, d: &Arc<Done>) {
        let mut g = self.q.0.lock().unwrap();
        let (mut keep, mut moved) = (std::collections::VecDeque::new(), Vec::new());
        while let Some(j) = g.ahead.pop_front() {
            if Arc::ptr_eq(&j.done, d) { moved.push(j) } else { keep.push_back(j) }
        }
        g.ahead = keep;
        for j in moved.into_iter().rev() {
            g.now.push_front(j);
        }
    }

    /// The same, waited for
    pub fn read_now(&self, p: &ExpertParts, ex: u64, dst: &mut [u8]) -> Result<()> {
        if dst.len() < p.bytes() {
            return Err(Error("read_now: the buffer is short".into()));
        }
        self.read_expert(p, ex, dst.as_mut_ptr(), true).wait()
    }
}

/// Whether decode's look ahead also copies guessed pinned-memory experts up to VRAM (NS_DS2_PREFETCH_UP=1); by
/// default it only reads guessed file-only ones into pinned memory: copies up ahead took VRAM slots from experts still
/// in use and queued before the next layer's own copies (128 tokens, B70 + B65: 4.98 tok/s with them, 5.35 without
/// any look ahead)
fn prefetch_up() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS2_PREFETCH_UP").is_ok_and(|v| v == "1"))
}

/// Where an expert is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    V(usize),
    R(usize),
    /// on its way from the file into host slot r
    P(usize),
}

/// Where decode multiplies an expert (`Store::place`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Where {
    /// its VRAM slot; true: its copy up may still be on its way
    Gpu(usize, bool),
    /// its pinned slot, on the CPU
    Cpu(usize),
    /// its pinned slot once the read from the file into it is done, on the CPU
    File(usize),
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    /// experts the CPU multiplied from pinned memory (decode)
    pub on_cpu: u64,
    /// pinned experts copied up to VRAM behind the CPU's products
    pub promoted: u64,
    pub hits: u64,
    pub from_host: u64,
    pub from_file: u64,
    /// host experts a prompt pass read where they are
    pub direct: u64,
    pub prefetched: u64,
    pub prefetch_used: u64,
    pub file_prefetched: u64,
}

pub struct Store {
    pub vram: Vec<DevBuf>,
    pub host: Vec<HostBuf>,
    pub slot: usize,
    per_chunk: usize,
    pub loc: HashMap<(u64, u64), Loc>,
    vowner: Vec<Option<(u64, u64)>>,
    vused: Vec<u64>,
    rowner: Vec<Option<(u64, u64)>>,
    rused: Vec<u64>,
    rfree: Vec<usize>,
    /// per host slot, the last copy that read or wrote it (a read from the file into it waits for that on the host)
    rlast: Vec<Option<i64>>,
    /// per VRAM slot, the copy up that last filled it (a copy down out of it waits for that)
    vfill: Vec<Option<i64>>,
    /// reads from the file in flight, by host slot
    reads: HashMap<usize, Arc<Done>>,
    /// experts swapped in ahead of their layer, not yet asked for: their copy up's ticket
    prefetched: HashMap<(u64, u64), i64>,
    /// the last copy up a promotion queued (the next waits for the lane to be idle)
    last_up: Option<i64>,
    tick: u64,
    pub stats: Stats,
    pub parts: HashMap<u64, ExpertParts>,
    reader: Arc<Reader>,
}

impl Store {
    #[allow(clippy::too_many_arguments)]
    pub fn new(vram: Vec<DevBuf>, host: Vec<HostBuf>, slot: usize, per_chunk: usize, nv: usize, nr: usize, parts: HashMap<u64, ExpertParts>,
               reader: Arc<Reader>) -> Store {
        Store { vram, host, slot, per_chunk, loc: HashMap::new(), vowner: vec![None; nv], vused: vec![0; nv], rowner: vec![None; nr], rused: vec![0; nr],
                rfree: (0..nr).rev().collect(), rlast: vec![None; nr], vfill: vec![None; nv], reads: HashMap::new(), prefetched: HashMap::new(), last_up: None, tick: 0,
                stats: Stats::default(), parts, reader }
    }
    pub fn vram_slots(&self) -> usize {
        self.vowner.len()
    }
    pub fn host_slots(&self) -> usize {
        self.rowner.len()
    }
    fn vat(&self, s: usize) -> (usize, usize) {
        (s / self.per_chunk, (s % self.per_chunk) * self.slot)
    }
    fn rat(&self, r: usize) -> (usize, usize) {
        (r / self.per_chunk, (r % self.per_chunk) * self.slot)
    }
    /// VRAM slot s as a view
    pub fn vview(&self, s: usize) -> Result<DevBuf> {
        let (c, o) = self.vat(s);
        self.vram[c].view(o, self.slot)
    }
    /// host slot r as the GPU sees it
    pub fn rview(&self, r: usize) -> Result<DevBuf> {
        let (c, o) = self.rat(r);
        self.host[c].device_view(o, self.slot)
    }
    fn rptr(&mut self, r: usize) -> *mut u8 {
        let (c, o) = self.rat(r);
        // SAFETY: o + slot is inside chunk c (rat's layout).
        unsafe { self.host[c].as_mut_slice().as_mut_ptr().add(o) }
    }

    /// At load: expert `key` into VRAM slot s / host slot r (written by the caller)
    pub fn place_v(&mut self, key: (u64, u64), s: usize) {
        self.vowner[s] = Some(key);
        self.loc.insert(key, Loc::V(s));
    }
    pub fn place_r(&mut self, key: (u64, u64), r: usize) {
        self.rfree.retain(|x| *x != r);
        self.rowner[r] = Some(key);
        self.loc.insert(key, Loc::R(r));
    }
    pub fn host_ptr(&mut self, r: usize) -> *mut u8 {
        self.rptr(r)
    }

    /// Starts a new request (a layer's experts): their slots are not victims until the next. Reads ahead that are done
    /// make their experts host experts (whether or not a layer asks for them)
    pub fn begin(&mut self) -> u64 {
        let done: Vec<usize> = self.reads.iter().filter(|(_, d)| d.ready()).map(|(r, _)| *r).collect();
        for r in done {
            let ok = self.reads.remove(&r).is_some_and(|d| d.wait().is_ok());
            match self.rowner[r] {
                Some(k) if ok => {
                    self.loc.insert(k, Loc::R(r));
                }
                Some(k) => {
                    // a failed read: the expert is on the file only again
                    self.loc.remove(&k);
                    self.rowner[r] = None;
                    self.rfree.push(r);
                }
                None => {}
            }
        }
        self.tick += 1;
        self.tick
    }

    /// A host slot for something new: a free one, else the least recently used host expert's (not in use this
    /// request, not on its way) - which is then on the file only. With the last copy that touched it: the host waits
    /// for that before a read from the file writes it (`for_read`); a copy into it takes it as a dependency.
    fn host_slot(&mut self, o: &Ops, for_read: bool) -> Result<(usize, Option<i64>)> {
        let r = match self.rfree.pop() {
            Some(r) => r,
            None => {
                let tick = self.tick;
                let victim = (0..self.rowner.len()).filter(|&i| self.rowner[i].is_some() && self.rused[i] != tick && !self.reads.contains_key(&i))
                    .min_by_key(|&i| self.rused[i]).ok_or_else(|| Error("the expert store has no host slot to give".into()))?;
                if let Some(k) = self.rowner[victim].take() {
                    self.loc.remove(&k);
                }
                victim
            }
        };
        let dep = self.rlast[r].take();
        if for_read {
            if let Some(t) = dep {
                if !o.ticket_done(t) {
                    o.wait_ticket(t)?;
                }
            }
            return Ok((r, None));
        }
        Ok((r, dep))
    }

    /// Expert key from the file into a host slot, started (not waited for): the slot. `now`: a layer waits for it
    fn start_read(&mut self, o: &Ops, key: (u64, u64), now: bool) -> Result<usize> {
        let (r, _) = self.host_slot(o, true)?;
        let p = *self.parts.get(&key.0).ok_or("store: a layer without experts here")?;
        let dst = self.rptr(r);
        let done = self.reader.read_expert(&p, key.1, dst, now);
        self.reads.insert(r, done);
        self.rowner[r] = Some(key);
        self.rused[r] = self.tick;
        self.loc.insert(key, Loc::P(r));
        Ok(r)
    }

    /// Waits for the read into host slot r (moved ahead of the reads for later layers); the expert is a host expert then
    fn finish_read(&mut self, r: usize) -> Result<()> {
        if let Some(d) = self.reads.remove(&r) {
            if !d.ready() {
                self.reader.hurry(&d);
            }
            d.wait()?;
        }
        if let Some(k) = self.rowner[r] {
            self.loc.insert(k, Loc::R(r));
        }
        Ok(())
    }

    /// The least recently used VRAM slot not in use this request
    fn vram_victim(&self) -> Result<usize> {
        let tick = self.tick;
        (0..self.vowner.len()).filter(|&i| self.vused[i] != tick).min_by_key(|&i| if self.vowner[i].is_none() { 0 } else { self.vused[i] + 1 })
            .ok_or_else(|| Error(format!("the expert store has {} VRAM slots; a layer needs more", self.vowner.len())))
    }

    /// Host expert in slot r up into VRAM slot s: the slot's expert down to a host slot first. The copy up's ticket.
    fn up(&mut self, o: &Ops, r: usize, s: usize) -> Result<i64> {
        let key = self.rowner[r].ok_or("store: an empty host slot")?;
        let sb = self.slot;
        let mut down = None;
        if let Some(v) = self.vowner[s].take() {
            self.prefetched.remove(&v);
            self.loc.remove(&v);
            // the victim to a host slot (never r itself: it is in use this request)
            self.rused[r] = self.tick;
            let (r2, dep) = self.host_slot(o, false)?;
            let t = o.stream_copy_on(0, &self.rview(r2)?, 0, &self.vview(s)?, 0, sb, &[self.vfill[s], dep])?;
            self.rlast[r2] = Some(t);
            self.rowner[r2] = Some(v);
            self.rused[r2] = self.tick.saturating_sub(1);
            self.loc.insert(v, Loc::R(r2));
            down = Some(t);
        }
        let t = o.stream_copy_on(1, &self.vview(s)?, 0, &self.rview(r)?, 0, sb, &[down, self.rlast[r]])?;
        self.vfill[s] = Some(t);
        self.rlast[r] = Some(t);
        self.rowner[r] = None;
        self.rfree.push(r);
        self.vowner[s] = Some(key);
        self.vused[s] = self.tick;
        self.loc.insert(key, Loc::V(s));
        Ok(t)
    }

    /// Decode, first half: the experts of `need` (layer l) in VRAM, or on their way up from pinned memory (also the ones
    /// read ahead whose reads are done); the file-only ones' reads started. Their slots in `need`'s order (None: on the
    /// file - `ensure_rest`), which of them are still arriving, and the last copy up among them (the GPU's queue awaits
    /// it before reading those).
    #[allow(clippy::type_complexity)]
    pub fn ensure(&mut self, o: &Ops, l: u64, need: &[u64]) -> Result<(Vec<Option<usize>>, Vec<bool>, Option<i64>)> {
        let tick = self.begin();
        self.trace(o, l, need);
        let mut last: Option<i64> = None;
        // the ones whose copy up may still be on its way
        let mut fresh: Vec<(u64, u64)> = Vec::new();
        // mark the resident ones in use first (none of them is a victim for the others)
        for &x in need {
            match self.loc.get(&(l, x)).copied() {
                Some(Loc::V(s)) => {
                    self.vused[s] = tick;
                    self.stats.hits += 1;
                    if let Some(t) = self.prefetched.remove(&(l, x)) {
                        self.stats.prefetch_used += 1;
                        last = Some(last.map_or(t, |a| a.max(t)));
                        fresh.push((l, x));
                    }
                }
                Some(Loc::R(r)) | Some(Loc::P(r)) => self.rused[r] = tick,
                None => {}
            }
        }
        // the file-only ones' reads all started at once, ahead of everything read for later layers
        for &x in need {
            match self.loc.get(&(l, x)).copied() {
                None => {
                    self.start_read(o, (l, x), true)?;
                    self.stats.from_file += 1;
                }
                Some(Loc::P(r)) => {
                    if let Some(d) = self.reads.get(&r) {
                        self.reader.hurry(d);
                    }
                }
                _ => {}
            }
        }
        // the host ones up now
        for &x in need {
            if let Some(Loc::R(r)) = self.loc.get(&(l, x)).copied() {
                self.stats.from_host += 1;
                let s = self.vram_victim()?;
                self.vused[s] = tick;
                let t = self.up(o, r, s)?;
                last = Some(last.map_or(t, |a| a.max(t)));
                fresh.push((l, x));
            }
        }
        let slots = need.iter().map(|x| match self.loc.get(&(l, *x)) {
            Some(Loc::V(s)) => Some(*s),
            _ => None,
        }).collect();
        let arriving = need.iter().map(|x| fresh.contains(&(l, *x))).collect();
        Ok((slots, arriving, last))
    }

    /// Decode, second half: the ones `ensure` left on the file (None in `first`), waited for and brought up. Their
    /// slots (in `need`'s order, for those), and the last copy up among them.
    pub fn ensure_rest(&mut self, o: &Ops, l: u64, need: &[u64], first: &[Option<usize>]) -> Result<(Vec<usize>, Option<i64>)> {
        let tick = self.tick;
        let mut last: Option<i64> = None;
        let mut slots = Vec::new();
        for (&x, f) in need.iter().zip(first) {
            if f.is_some() {
                continue;
            }
            let r = match self.loc.get(&(l, x)).copied() {
                Some(Loc::P(r)) => {
                    self.finish_read(r)?;
                    r
                }
                Some(Loc::R(r)) => r,
                _ => return Err(Error("store: an expert went missing".into())),
            };
            let s = self.vram_victim()?;
            self.vused[s] = tick;
            let t = self.up(o, r, s)?;
            last = Some(last.map_or(t, |a| a.max(t)));
            slots.push(s);
        }
        Ok((slots, last))
    }

    /// Decode with the CPU's share: where each expert of `need` (layer l) is multiplied - on the GPU from its VRAM slot
    /// (`Gpu`; `fresh`: its copy up may still be on its way, `ticket` the last such copy), on the CPU from its pinned
    /// slot (`Cpu`), or on the CPU once its read from the file into a pinned slot is done (`File`, the read started)
    pub fn place(&mut self, o: &Ops, l: u64, need: &[u64]) -> Result<(Vec<Where>, Option<i64>)> {
        let tick = self.begin();
        self.trace(o, l, need);
        let mut last: Option<i64> = None;
        let mut out = Vec::with_capacity(need.len());
        for &x in need {
            let w = match self.loc.get(&(l, x)).copied() {
                Some(Loc::V(s)) => {
                    self.vused[s] = tick;
                    self.stats.hits += 1;
                    let fresh = match self.prefetched.remove(&(l, x)) {
                        Some(t) => {
                            self.stats.prefetch_used += 1;
                            last = Some(last.map_or(t, |a| a.max(t)));
                            true
                        }
                        None => false,
                    };
                    Where::Gpu(s, fresh)
                }
                Some(Loc::R(r)) => {
                    self.rused[r] = tick;
                    self.stats.on_cpu += 1;
                    Where::Cpu(r)
                }
                Some(Loc::P(r)) => {
                    self.rused[r] = tick;
                    if let Some(d) = self.reads.get(&r) {
                        self.reader.hurry(d);
                    }
                    self.stats.on_cpu += 1;
                    Where::File(r)
                }
                None => Where::File(usize::MAX),
            };
            out.push(w);
        }
        for (k, &x) in need.iter().enumerate() {
            if out[k] == Where::File(usize::MAX) {
                let r = self.start_read(o, (l, x), true)?;
                self.stats.from_file += 1;
                out[k] = Where::File(r);
            }
        }
        Ok((out, last))
    }

    /// Waits for the reads from the file into these pinned slots
    pub fn finish_reads(&mut self, rs: &[usize]) -> Result<()> {
        for &r in rs {
            if self.reads.contains_key(&r) {
                self.finish_read(r)?;
            }
        }
        Ok(())
    }

    /// Pinned slot r's bytes (for the CPU's products; its read from the file done - `finish_reads`)
    pub fn host_slice(&self, r: usize) -> &[u8] {
        let (c, o) = self.rat(r);
        &self.host[c].as_slice()[o..o + self.slot]
    }

    /// After a layer the CPU multiplied these pinned experts of: up to `limit` of them promoted to VRAM in the
    /// background (their copies queued, nothing waits) - only while the copy lane is idle, so the copies never pile up
    /// ahead of what a later layer needs
    pub fn promote(&mut self, o: &Ops, l: u64, xs: &[u64], limit: usize) -> Result<usize> {
        let mut n = 0;
        for &x in xs {
            if n >= limit || self.last_up.is_some_and(|t| !o.ticket_done(t)) {
                break;
            }
            let Some(Loc::R(r)) = self.loc.get(&(l, x)).copied() else { continue };
            let s = self.vram_victim()?;
            self.vused[s] = self.tick;
            let t = self.up(o, r, s)?;
            self.prefetched.insert((l, x), t);
            self.last_up = Some(t);
            self.stats.promoted += 1;
            n += 1;
        }
        Ok(n)
    }

    fn trace(&self, o: &Ops, l: u64, need: &[u64]) {
        // NS_DS2_TRACE_EXPERTS=FILE: every decode request (GPU, layer, experts) for offline policy studies
        if let Ok(f) = std::env::var("NS_DS2_TRACE_EXPERTS") {
            use std::io::Write;
            if let Ok(mut w) = std::fs::OpenOptions::new().create(true).append(true).open(f) {
                let _ = writeln!(w, "{} {} {}", o.gpu.index, l, need.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
        }
    }

    /// Decode's look ahead: the experts layer l is guessed to want - host ones swapped up now (at most `limit`), file
    /// ones read into host slots (not waited for)
    pub fn prefetch(&mut self, o: &Ops, l: u64, want: &[u64], limit: usize) -> Result<()> {
        let mut n = 0;
        for &x in want {
            if n >= limit {
                break;
            }
            match self.loc.get(&(l, x)).copied() {
                Some(Loc::R(r)) if prefetch_up() => {
                    let s = self.vram_victim()?;
                    self.vused[s] = self.tick;
                    let t = self.up(o, r, s)?;
                    self.prefetched.insert((l, x), t);
                    self.stats.prefetched += 1;
                    n += 1;
                }
                None => {
                    self.start_read(o, (l, x), false)?;
                    self.stats.file_prefetched += 1;
                    n += 1;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A prompt pass: where each expert of `need` is read - its VRAM slot (false), or its host slot (true) with the last
    /// copy that wrote it (the caller's copy out of it waits for that); the file-only ones read into host slots first.
    pub fn locate(&mut self, o: &Ops, l: u64, need: &[u64]) -> Result<Vec<(DevBuf, bool, Option<i64>)>> {
        let tick = self.begin();
        for &x in need {
            match self.loc.get(&(l, x)).copied() {
                Some(Loc::V(s)) => self.vused[s] = tick,
                Some(Loc::R(r)) | Some(Loc::P(r)) => self.rused[r] = tick,
                None => {}
            }
        }
        for &x in need {
            if !self.loc.contains_key(&(l, x)) {
                self.start_read(o, (l, x), true)?;
                self.stats.from_file += 1;
            }
        }
        need.iter().map(|x| match self.loc.get(&(l, *x)).copied() {
            Some(Loc::V(s)) => {
                self.stats.hits += 1;
                Ok((self.vview(s)?, false, None))
            }
            Some(Loc::P(r)) => {
                self.finish_read(r)?;
                self.stats.direct += 1;
                Ok((self.rview(r)?, true, None))
            }
            Some(Loc::R(r)) => {
                self.stats.direct += 1;
                Ok((self.rview(r)?, true, self.rlast[r]))
            }
            None => Err(Error("store: an expert went missing".into())),
        }).collect()
    }

    /// After a prompt pass read host slots in place (or through copies): those reads' ticket, so nothing overwrites
    /// them before
    pub fn host_read_by(&mut self, need: &[(u64, u64)], t: i64) {
        for k in need {
            if let Some(Loc::R(r)) = self.loc.get(k).copied() {
                self.rlast[r] = Some(t);
            }
        }
    }

    pub fn bytes_from_file(&self) -> usize {
        self.reader.bytes.load(Ordering::Relaxed)
    }
}
