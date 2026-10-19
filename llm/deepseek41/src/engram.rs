//! The Engram's host side: the n-gram hash of the tokens (DeepSeek's `NgramHashState`, the constants the GGUF carries)
//! and the table rows read from the two official shards on disk.
//!
//! A token's rows at an Engram layer: its compressed id and the 3 before it (the pad id before the start) multiplied by
//! the layer's 4 multipliers and XORed one lookback at a time - the 2-, 3- and 4-gram's hashes - each modulo 8 primes
//! (a head each) plus the head's offset: 24 rows of 256 E4M3 weights and 8 E8M0 scales, a 512-byte read (O_DIRECT)
//! for each, the weight and the scale apart. The rows are known from the tokens before any layer runs: `Engram::fetch`
//! hands them to a pool of reader threads (one read in flight each: the queue depth) and returns at once; `Fetch::wait`
//! takes them at the layer. A small cache keeps the rows read last (a verify pass re-reads its positions).

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use nextsycl_core::{Error, Result};
use nextsycl_gguf::{Gguf, Value};

/// bytes of a row as the GPU takes it: the weights, then the scales
pub fn row_bytes(hd: usize) -> usize {
    hd + hd / 32
}

/// NS_DS41_ENGRAM_DIR: where the two shards are (default: `deepseek-engram` beside the model)
pub fn engram_dir(model: &Path) -> PathBuf {
    std::env::var("NS_DS41_ENGRAM_DIR").map(PathBuf::from).unwrap_or_else(|_| model.parent().unwrap_or(Path::new(".")).join("deepseek-engram"))
}

/// NS_DS41_ENGRAM_READERS: reader threads (the reads in flight; default 48)
fn readers() -> usize {
    std::env::var("NS_DS41_ENGRAM_READERS").ok().and_then(|v| v.parse().ok()).unwrap_or(48).clamp(1, 512)
}

/// NS_DS41_ENGRAM_CACHE: rows kept in host memory (default 262144: 69 MB)
fn cache_rows() -> usize {
    std::env::var("NS_DS41_ENGRAM_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(262_144)
}

/// the rows read last by (table, row), and their order
type RowCache = (HashMap<(usize, u64), Vec<u8>>, VecDeque<(usize, u64)>);

struct Table {
    file: File,
    weight: u64,
    scale: u64,
}

/// One pass's rows on their way: rows k of layers li into `out` [n][row_bytes]
pub struct Fetch {
    out: Mutex<Vec<u8>>,
    left: AtomicUsize,
    done: Condvar,
    lock: Mutex<()>,
    err: Mutex<Option<String>>,
}

impl Fetch {
    /// The rows, once read: [n][row_bytes]
    pub fn wait(&self) -> Result<Vec<u8>> {
        let mut g = self.lock.lock().unwrap();
        while self.left.load(Ordering::Acquire) > 0 {
            g = self.done.wait(g).unwrap();
        }
        drop(g);
        if let Some(e) = self.err.lock().unwrap().take() {
            return Err(Error(e));
        }
        Ok(std::mem::take(&mut *self.out.lock().unwrap()))
    }
}

struct Job {
    fetch: Arc<Fetch>,
    /// (table, row, where in out, the row's weights (false) or its scales (true))
    items: Vec<(usize, u64, usize, bool)>,
}

pub struct Engram {
    pub layers: Vec<u64>,
    token_map: Vec<i64>,
    /// [layer][ngram]
    mult: Vec<i64>,
    /// [layer][ngram - 1][heads]
    primes: Vec<i64>,
    /// [layer][cols]
    offsets: Vec<i64>,
    pad: i64,
    ngram: usize,
    heads: usize,
    pub hd: usize,
    /// the shards (the readers hold them too)
    _tables: Arc<Vec<Table>>,
    jobs: Mutex<Option<std::sync::mpsc::Sender<Job>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    cache: Arc<Mutex<RowCache>>,
    /// rows read from disk, found in the cache; reads
    pub read_rows: AtomicU64,
    pub cached_rows: AtomicU64,
    pub pages: Arc<AtomicU64>,
    /// nanoseconds the passes waited for rows
    pub wait_ns: AtomicU64,
}

impl Drop for Engram {
    fn drop(&mut self) {
        self.jobs.lock().unwrap().take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn ints(f: &Gguf, name: &str) -> Result<Vec<i64>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{name} is missing")))?;
    let raw = f.read(t).map_err(|e| Error(e.0))?;
    Ok(match t.ty {
        nextsycl_gguf::GType::I32 => raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        nextsycl_gguf::GType::I64 => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        other => return Err(Error(format!("{name}: type {}, integers expected", other.name()))),
    })
}

/// a page-aligned bounce buffer for O_DIRECT reads
struct Bounce {
    ptr: *mut u8,
    len: usize,
}
impl Bounce {
    fn new(len: usize) -> Bounce {
        let layout = std::alloc::Layout::from_size_align(len, 4096).unwrap();
        // SAFETY: a non-zero size, freed in Drop with the same layout.
        Bounce { ptr: unsafe { std::alloc::alloc(layout) }, len }
    }
    fn buf(&mut self) -> &mut [u8] {
        // SAFETY: allocated with len bytes, owned by this value.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for Bounce {
    fn drop(&mut self) {
        // SAFETY: allocated in new with this layout.
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, 4096).unwrap()) }
    }
}

/// bytes read from the tables, and the reads' time (each read's own, summed) and the longest one, in nanoseconds
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
static READ_NS: AtomicU64 = AtomicU64::new(0);
static READ_MAX_NS: AtomicU64 = AtomicU64::new(0);
fn bytes_read() -> &'static AtomicU64 {
    &READ_BYTES
}

/// NS_DS41_ENGRAM_ALIGN: the reads' unit in bytes (default 512: the drives' logical block, what O_DIRECT needs; a row's
/// 256 weights are one block, its scales another - 4096 reads whole pages, 8 times the bytes)
fn read_align() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("NS_DS41_ENGRAM_ALIGN").ok().and_then(|v| v.parse::<u64>().ok()).filter(|v| v.is_power_of_two() && *v >= 512).unwrap_or(512))
}

/// bytes [off, off + dst.len()) of f through block-aligned reads (the bounce buffer is page-aligned)
fn read_at(f: &File, b: &mut Bounce, off: u64, dst: &mut [u8], pages: &AtomicU64) -> std::io::Result<()> {
    let al = read_align();
    let a = off & !(al - 1);
    let end = (off + dst.len() as u64).next_multiple_of(al);
    let n = (end - a) as usize;
    if n > b.len {
        *b = Bounce::new(n);
    }
    let buf = &mut b.buf()[..n];
    let t0 = std::time::Instant::now();
    let mut got = 0;
    while got < n {
        let k = f.read_at(&mut buf[got..], a + got as u64)?;
        if k == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "past the table's end"));
        }
        got += k;
    }
    let ns = t0.elapsed().as_nanos() as u64;
    READ_NS.fetch_add(ns, Ordering::Relaxed);
    READ_MAX_NS.fetch_max(ns, Ordering::Relaxed);
    pages.fetch_add(1, Ordering::Relaxed);
    bytes_read().fetch_add(n as u64, Ordering::Relaxed);
    let s = (off - a) as usize;
    dst.copy_from_slice(&buf[s..s + dst.len()]);
    Ok(())
}

impl Engram {
    /// The hash constants from the GGUF, the shards from `dir` (their names the metadata's table_path's), the readers.
    pub fn open(f: &Gguf, dir: &Path) -> Result<Engram> {
        let m = |k: &str| f.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| Error(format!("metadata {k} is missing")));
        let layers: Vec<u64> = f.arch_meta("engram.layer_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        let token_map = ints(f, "engram.token_map")?;
        let pad_id = m("engram.pad_id_compressed")? as usize;
        let pad = *token_map.get(pad_id).ok_or("engram: the pad id is past the token map")?;
        let (ngram, heads, hd) = (m("engram.max_ngram_size")? as usize, m("engram.head_count")? as usize, m("engram.head_dim")? as usize);
        let mut tables = Vec::new();
        for i in 0..layers.len() {
            let path = f.arch_meta(&format!("engram.{i}.table_path")).and_then(Value::as_str).ok_or("engram: a table_path is missing")?;
            let name = Path::new(path).file_name().ok_or("engram: a table_path without a file name")?;
            let p = dir.join(name);
            // O_DIRECT: the 203 GB of tables read in pages must not push the experts out of the page cache
            let file = std::fs::OpenOptions::new().read(true).custom_flags(libc_o_direct()).open(&p).or_else(|_| File::open(&p))
                .map_err(|e| Error(format!("the Engram table {}: {e} (NS_DS41_ENGRAM_DIR names their directory)", p.display())))?;
            tables.push(Table { file, weight: m(&format!("engram.{i}.weight_offset"))?, scale: m(&format!("engram.{i}.scale_offset"))? });
        }
        let tables = Arc::new(tables);
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let pages = Arc::new(AtomicU64::new(0));
        let rb = row_bytes(hd);
        let mut workers = Vec::new();
        for i in 0..readers() {
            let (rx, tables, pages) = (rx.clone(), tables.clone(), pages.clone());
            workers.push(std::thread::Builder::new().name(format!("ds41-engram-{i}")).spawn(move || {
                let mut b = Bounce::new(8192);
                loop {
                    let job = { rx.lock().unwrap().recv() };
                    let Ok(job) = job else { break };
                    let mut part = vec![0u8; rb];
                    for &(ti, r, at, scales) in &job.items {
                        let t = &tables[ti];
                        let (off, n, at) = if scales { (t.scale + r * (hd / 32) as u64, hd / 32, at + hd) } else { (t.weight + r * hd as u64, hd, at) };
                        match read_at(&t.file, &mut b, off, &mut part[..n], &pages) {
                            Ok(()) => job.fetch.out.lock().unwrap()[at..at + n].copy_from_slice(&part[..n]),
                            Err(e) => *job.fetch.err.lock().unwrap() = Some(format!("reading an Engram row: {e}")),
                        }
                    }
                    if job.fetch.left.fetch_sub(job.items.len(), Ordering::AcqRel) == job.items.len() {
                        let _g = job.fetch.lock.lock().unwrap();
                        job.fetch.done.notify_all();
                    }
                }
            }).map_err(|e| Error(format!("an Engram reader: {e}")))?);
        }
        Ok(Engram { layers, token_map, mult: ints(f, "engram.multipliers")?, primes: ints(f, "engram.primes")?, offsets: ints(f, "engram.offsets")?, pad, ngram,
                    heads, hd, _tables: tables, jobs: Mutex::new(Some(tx)), workers, cache: Arc::new(Mutex::new((HashMap::new(), VecDeque::new()))),
                    read_rows: AtomicU64::new(0), cached_rows: AtomicU64::new(0), pages, wait_ns: AtomicU64::new(0) })
    }

    /// rows a token reads at a layer
    pub fn cols(&self) -> usize {
        (self.ngram - 1) * self.heads
    }

    /// The rows of the tokens `hist[from..]` (each with the 3 before it from `hist`), for every Engram layer:
    /// [layer][token][col]
    pub fn rows(&self, hist: &[u32], from: usize) -> Vec<Vec<u64>> {
        let (ng, nc) = (self.ngram, self.cols());
        let comp = |p: usize| -> i64 { self.token_map.get(hist[p] as usize).copied().unwrap_or(self.pad) };
        let mut out = vec![Vec::with_capacity((hist.len() - from) * nc); self.layers.len()];
        for p in from..hist.len() {
            let toks: Vec<i64> = (0..ng).map(|s| if p >= s { comp(p - s) } else { self.pad }).collect();
            for (li, o) in out.iter_mut().enumerate() {
                let mut rolling = toks[0].wrapping_mul(self.mult[li * ng]);
                for (i, tk) in toks.iter().enumerate().skip(1) {
                    rolling ^= tk.wrapping_mul(self.mult[li * ng + i]);
                    for h in 0..self.heads {
                        let prime = self.primes[(li * (ng - 1) + i - 1) * self.heads + h];
                        let c = (i - 1) * self.heads + h;
                        o.push((rolling.rem_euclid(prime) + self.offsets[li * nc + c]) as u64);
                    }
                }
            }
        }
        out
    }

    /// Layer li's rows `rows` on their way into [n][row_bytes] (the cached ones copied now)
    pub fn fetch(&self, li: usize, rows: &[u64]) -> Arc<Fetch> {
        let rb = row_bytes(self.hd);
        let mut out = vec![0u8; rows.len() * rb];
        let mut todo = Vec::new();
        {
            let c = self.cache.lock().unwrap();
            for (k, &r) in rows.iter().enumerate() {
                match c.0.get(&(li, r)) {
                    Some(v) => out[k * rb..(k + 1) * rb].copy_from_slice(v),
                    None => {
                        todo.push((li, r, k * rb, false));
                        todo.push((li, r, k * rb, true));
                    }
                }
            }
        }
        self.cached_rows.fetch_add((rows.len() - todo.len() / 2) as u64, Ordering::Relaxed);
        self.read_rows.fetch_add((todo.len() / 2) as u64, Ordering::Relaxed);
        let fetch = Arc::new(Fetch { out: Mutex::new(out), left: AtomicUsize::new(todo.len()), done: Condvar::new(), lock: Mutex::new(()), err: Mutex::new(None) });
        if todo.is_empty() {
            return fetch;
        }
        // a job a read or a few (a row's weights and its scales apart, in flight at once): the drive is bound by reads a
        // second (a DRAM-less one ~67K at any size up to 4 KiB), so the first layer's reads go out first and alone
        // (fetch is called in layer order) and the later layers' queue behind them
        let per = todo.len().div_ceil(readers()).clamp(1, 16);
        let jobs = self.jobs.lock().unwrap();
        for items in todo.chunks(per) {
            if let Some(tx) = jobs.as_ref() {
                let _ = tx.send(Job { fetch: fetch.clone(), items: items.to_vec() });
            }
        }
        fetch
    }

    /// A fetch's rows, waited for, then kept in the cache
    pub fn take(&self, li: usize, rows: &[u64], f: &Fetch) -> Result<Vec<u8>> {
        let t0 = std::time::Instant::now();
        let out = f.wait()?;
        self.wait_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let rb = row_bytes(self.hd);
        let cap = cache_rows();
        if cap > 0 {
            let mut c = self.cache.lock().unwrap();
            let (map, order) = &mut *c;
            for (k, &r) in rows.iter().enumerate() {
                if map.insert((li, r), out[k * rb..(k + 1) * rb].to_vec()).is_none() {
                    order.push_back((li, r));
                }
            }
            while order.len() > cap {
                if let Some(old) = order.pop_front() {
                    map.remove(&old);
                }
            }
        }
        Ok(out)
    }

    /// (rows read from disk, rows from the cache, reads, bytes read, the reads' mean and longest time in microseconds,
    /// seconds the passes waited)
    pub fn stats(&self) -> (u64, u64, u64, u64, f64, f64, f64) {
        let reads = self.pages.load(Ordering::Relaxed);
        (self.read_rows.load(Ordering::Relaxed), self.cached_rows.load(Ordering::Relaxed), reads, READ_BYTES.load(Ordering::Relaxed),
         READ_NS.load(Ordering::Relaxed) as f64 * 1e-3 / reads.max(1) as f64, READ_MAX_NS.load(Ordering::Relaxed) as f64 * 1e-3,
         self.wait_ns.load(Ordering::Relaxed) as f64 * 1e-9)
    }
}

/// O_DIRECT's value on Linux x86-64 / aarch64 (no libc crate in the workspace for one constant)
fn libc_o_direct() -> i32 {
    if cfg!(target_arch = "aarch64") { 0o200000 } else { 0o40000 }
}

#[cfg(test)]
mod tests {
    /// The hash of DeepSeek's NgramHashState on a small table: the product XOR a lookback at a time, modulo the
    /// head's prime, plus its offset
    #[test]
    fn rolling_hash() {
        let (ng, heads) = (3usize, 2usize);
        let mult = [3i64, 5, 7];
        let primes = [11i64, 13, 17, 19];
        let offs = [0i64, 11, 24, 41];
        let toks = [4i64, 9, 2]; // the token, then the ones before it
        let p = [toks[0] * mult[0], toks[1] * mult[1], toks[2] * mult[2]];
        let mut want = Vec::new();
        let mut rolling = p[0];
        for i in 1..ng {
            rolling ^= p[i];
            for h in 0..heads {
                want.push(rolling % primes[(i - 1) * heads + h] + offs[(i - 1) * heads + h]);
            }
        }
        assert_eq!(want, vec![(12 ^ 45) % 11, (12 ^ 45) % 13 + 11, (12 ^ 45 ^ 14) % 17 + 24, (12 ^ 45 ^ 14) % 19 + 41]);
    }
}
