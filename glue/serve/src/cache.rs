//! The server's prompt cache: checkpoints of the conversation state (`Glm::save`) at token prefixes, in host memory,
//! least recently used out first once they pass the budget. A request mounts the longest cached prefix of its
//! tokens and reads only the rest.
//!
//! With a disk tier (`--cache-dir`), a checkpoint pushed out of memory is written there instead of dropped (a 256K
//! prompt's checkpoint is ~3.5 GiB: the memory budget holds one) and mounted from the file when it is the best
//! prefix - ~1-3 s for 3.5 GiB from the NVMe against minutes to read the prompt again. The files outlive the server
//! (a stop writes the ones still in memory too): they sit in a subdirectory per fingerprint - the model file, the
//! cache's form, the draft block - so a checkpoint is only mounted by a server whose sessions it fits (the context
//! sizes may differ: one only needs to be as long). A file unused for the TTL (24 h) is removed, at start and at
//! each write; past the size budget the least recently used go first.

use std::path::PathBuf;

use nextsycl_llm::Checkpoint;

struct Entry {
    tokens: Vec<u32>,
    ck: Checkpoint,
    used: u64,
}

struct DiskEntry {
    tokens: Vec<u32>,
    path: PathBuf,
    bytes: usize,
    used: u64,
    /// when it was last written or mounted (the file's modification time)
    touched: std::time::SystemTime,
}

struct Disk {
    dir: PathBuf,
    budget: usize,
    ttl: std::time::Duration,
    entries: Vec<DiskEntry>,
    next: u64,
}

/// 64-bit FNV-1a of a string, in hex (the fingerprint's directory name)
fn fnv(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn sidecar(path: &std::path::Path) -> PathBuf {
    path.with_extension("tok")
}

/// The tokens a checkpoint file is the state after ([count u64][u32 ...]), beside it
fn write_tokens(path: &std::path::Path, tokens: &[u32]) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(8 + tokens.len() * 4);
    b.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        b.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(sidecar(path), b)
}

fn read_tokens(path: &std::path::Path) -> Option<Vec<u32>> {
    let b = std::fs::read(sidecar(path)).ok()?;
    let n = u64::from_le_bytes(b.get(..8)?.try_into().ok()?) as usize;
    let body = b.get(8..8 + n * 4)?;
    Some(body.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn remove(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(sidecar(path));
}

/// Where a cached prefix is
#[derive(Clone, Copy)]
pub enum Hit {
    Ram(usize),
    Disk(usize),
}

pub struct PromptCache {
    entries: Vec<Entry>,
    budget: usize,
    clock: u64,
    pub evictions: u64,
    disk: Option<Disk>,
}

impl PromptCache {
    pub fn new(budget: usize) -> PromptCache {
        PromptCache { entries: Vec::new(), budget, clock: 0, evictions: 0, disk: None }
    }

    /// A disk tier of `budget` bytes under `root`, in the subdirectory of `fingerprint` (what a checkpoint must
    /// match to fit this server's sessions): the files there from earlier runs taken back, those unused for `ttl`
    /// (in every subdirectory) removed
    pub fn with_disk(mut self, root: PathBuf, budget: usize, fingerprint: &str, ttl: std::time::Duration) -> Result<PromptCache, String> {
        let now = std::time::SystemTime::now();
        let stale = |p: &std::path::Path| p.metadata().and_then(|m| m.modified()).map_or(true, |t| now.duration_since(t).unwrap_or_default() > ttl);
        std::fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
        // every fingerprint's stale files out (a model or a cache form no longer served leaves its files to this)
        for d in std::fs::read_dir(&root).map_err(|e| e.to_string())?.flatten().filter(|d| d.path().is_dir()) {
            for f in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
                let p = f.path();
                if p.extension().is_some_and(|x| x == "nsck") && stale(&p) {
                    remove(&p);
                }
            }
            let _ = std::fs::remove_dir(d.path()); // only when empty
        }
        let dir = root.join(fnv(fingerprint));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let _ = std::fs::write(dir.join("fingerprint.txt"), format!("{fingerprint}\n"));
        // the checkpoints of earlier runs with this fingerprint, oldest first (their use order)
        let mut found: Vec<(std::time::SystemTime, DiskEntry, u64)> = Vec::new();
        for f in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
            let p = f.path();
            if p.extension().is_none_or(|x| x != "nsck") {
                continue;
            }
            let (Some(tokens), Ok(meta)) = (read_tokens(&p), p.metadata()) else {
                remove(&p); // half written
                continue;
            };
            let n = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse().ok()).unwrap_or(0);
            let t = meta.modified().unwrap_or(now);
            found.push((t, DiskEntry { tokens, path: p, bytes: meta.len() as usize, used: 0, touched: t }, n));
        }
        found.sort_by_key(|f| f.0);
        let next = found.iter().map(|f| f.2).max().unwrap_or(0);
        let entries: Vec<DiskEntry> = found.into_iter().map(|(_, mut e, _)| {
            self.clock += 1;
            e.used = self.clock;
            e
        }).collect();
        self.disk = Some(Disk { dir, budget, ttl, entries, next });
        if let Some(d) = &mut self.disk {
            d.prune(0);
        }
        Ok(self)
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.ck.bytes()).sum()
    }

    /// (checkpoints, bytes, budget) on disk
    pub fn disk(&self) -> Option<(usize, usize, usize)> {
        self.disk.as_ref().map(|d| (d.entries.len(), d.entries.iter().map(|e| e.bytes).sum(), d.budget))
    }

    /// (tokens, bytes, last use - higher is more recent, on disk) per entry, the most recently used first
    pub fn list(&self) -> Vec<(usize, usize, u64, bool)> {
        let mut v: Vec<_> = self.entries.iter().map(|e| (e.tokens.len(), e.ck.bytes(), e.used, false)).collect();
        if let Some(d) = &self.disk {
            v.extend(d.entries.iter().map(|e| (e.tokens.len(), e.bytes, e.used, true)));
        }
        v.sort_by_key(|e| std::cmp::Reverse(e.2));
        v
    }

    pub fn clear(&mut self) -> usize {
        let mut n = self.entries.len();
        self.entries.clear();
        if let Some(d) = &mut self.disk {
            n += d.entries.len();
            for e in d.entries.drain(..) {
                remove(&e.path);
            }
        }
        n
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The longest entry, in memory or on disk, that is a proper prefix of `ids` (some token is left to read): where
    /// it is and its length. Memory wins a tie.
    pub fn best(&self, ids: &[u32]) -> Option<(Hit, usize)> {
        let fits = |t: &[u32]| t.len() < ids.len() && ids.starts_with(t);
        let ram = self.entries.iter().enumerate().filter(|(_, e)| fits(&e.tokens)).max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, e)| (Hit::Ram(i), e.tokens.len()));
        let disk = self.disk.as_ref().and_then(|d| {
            d.entries.iter().enumerate().filter(|(_, e)| fits(&e.tokens)).max_by_key(|(_, e)| e.tokens.len())
                .map(|(i, e)| (Hit::Disk(i), e.tokens.len()))
        });
        match (ram, disk) {
            (Some(r), Some(d)) => Some(if d.1 > r.1 { d } else { r }),
            (r, d) => r.or(d),
        }
    }

    /// `f` on the checkpoint at `hit` (from `best`, marked as used): in memory, or read from its file
    /// (`read`: the engine's reader of its checkpoint files)
    pub fn with<R>(&mut self, hit: Hit, read: &dyn Fn(&mut dyn std::io::Read) -> std::io::Result<Checkpoint>, f: impl FnOnce(&Checkpoint) -> R) -> Result<R, String> {
        self.clock += 1;
        match hit {
            Hit::Ram(i) => {
                self.entries[i].used = self.clock;
                Ok(f(&self.entries[i].ck))
            }
            Hit::Disk(i) => {
                let d = self.disk.as_mut().ok_or("no disk tier")?;
                d.entries[i].used = self.clock;
                d.entries[i].touched = std::time::SystemTime::now();
                if let Ok(f) = std::fs::File::options().write(true).open(&d.entries[i].path) {
                    let _ = f.set_modified(d.entries[i].touched);
                }
                let file = std::fs::File::open(&d.entries[i].path).map_err(|e| format!("{}: {e}", d.entries[i].path.display()))?;
                let ck = read(&mut std::io::BufReader::with_capacity(8 << 20, file)).map_err(|e| e.to_string())?;
                Ok(f(&ck))
            }
        }
    }

    /// Whether `tokens` are cached already, in memory or on disk (then marked as used).
    pub fn touch(&mut self, tokens: &[u32]) -> bool {
        self.clock += 1;
        let c = self.clock;
        if let Some(e) = self.entries.iter_mut().find(|e| e.tokens == tokens) {
            e.used = c;
            return true;
        }
        if let Some(e) = self.disk.as_mut().and_then(|d| d.entries.iter_mut().find(|e| e.tokens == tokens)) {
            e.used = c;
            return true;
        }
        false
    }

    /// Keeps `ck` for `tokens`, making room by moving the least recently used to disk (or dropping them without a
    /// disk tier). A checkpoint larger than the whole memory budget goes straight to disk (or is dropped).
    pub fn put(&mut self, tokens: Vec<u32>, ck: Checkpoint) -> bool {
        self.entries.retain(|e| e.tokens != tokens);
        if ck.bytes() > self.budget {
            let c = self.clock + 1;
            self.clock = c;
            return self.spill(Entry { tokens, ck, used: c });
        }
        while self.bytes() + ck.bytes() > self.budget {
            let Some((i, _)) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used) else { break };
            let e = self.entries.swap_remove(i);
            self.evictions += 1;
            self.spill(e);
        }
        self.clock += 1;
        self.entries.push(Entry { tokens, ck, used: self.clock });
        true
    }

    /// Every checkpoint still in memory onto the disk tier (a stop: the next server takes them back); how many
    pub fn persist_all(&mut self) -> usize {
        let v = std::mem::take(&mut self.entries);
        v.into_iter().map(|e| self.spill(e)).filter(|w| *w).count()
    }

    /// An entry out of memory onto the disk tier (stale files and then the least recently used removed to make room)
    fn spill(&mut self, e: Entry) -> bool {
        let Some(d) = &mut self.disk else { return false };
        if e.ck.bytes() > d.budget || d.entries.iter().any(|x| x.tokens == e.tokens) {
            return false;
        }
        d.prune(e.ck.bytes());
        d.next += 1;
        let path = d.dir.join(format!("{}.nsck", d.next));
        // the tokens' file last: a checkpoint without it is a half-written one
        let written = std::fs::File::create(&path).and_then(|f| {
            let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
            e.ck.write_to(&mut w)?;
            std::io::Write::flush(&mut w)
        }).and_then(|_| write_tokens(&path, &e.tokens));
        match written {
            Ok(()) => {
                d.entries.push(DiskEntry { tokens: e.tokens, path, bytes: e.ck.bytes(), used: e.used, touched: std::time::SystemTime::now() });
                true
            }
            Err(err) => {
                eprintln!("[prompt cache: writing {} failed: {err}]", path.display());
                remove(&path);
                false
            }
        }
    }
}

impl Disk {
    /// Files unused for the TTL out, then the least recently used until `room` more bytes fit the budget
    fn prune(&mut self, room: usize) {
        let now = std::time::SystemTime::now();
        let ttl = self.ttl;
        self.entries.retain(|e| {
            let keep = now.duration_since(e.touched).unwrap_or_default() <= ttl;
            if !keep {
                remove(&e.path);
            }
            keep
        });
        while self.entries.iter().map(|x| x.bytes).sum::<usize>() + room > self.budget {
            let Some((i, _)) = self.entries.iter().enumerate().min_by_key(|(_, x)| x.used) else { break };
            let x = self.entries.swap_remove(i);
            remove(&x.path);
        }
    }
}
