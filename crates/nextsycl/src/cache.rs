//! The server's prompt cache: checkpoints of the conversation state (`Glm::save`) at token prefixes, in host memory,
//! least recently used out first once they pass the budget. A request mounts the longest cached prefix of its
//! tokens and reads only the rest.
//!
//! With a disk tier (`--cache-dir`), a checkpoint pushed out of memory is written there instead of dropped (a 256K
//! prompt's checkpoint is ~3.5 GiB: the memory budget holds one) and mounted from the file when it is the best
//! prefix - ~1-3 s for 3.5 GiB from the NVMe against minutes to read the prompt again. The directory is emptied at
//! start: a checkpoint only fits sessions of the process that made it (the model, the split, the cache's form).

use std::path::PathBuf;

use ns_engine::glm5next::Checkpoint;

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
}

struct Disk {
    dir: PathBuf,
    budget: usize,
    entries: Vec<DiskEntry>,
    next: u64,
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

    /// A disk tier of `budget` bytes in `dir` (emptied of earlier checkpoint files)
    pub fn with_disk(mut self, dir: PathBuf, budget: usize) -> Result<PromptCache, String> {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for f in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
            if f.path().extension().is_some_and(|x| x == "nsck") {
                let _ = std::fs::remove_file(f.path());
            }
        }
        self.disk = Some(Disk { dir, budget, entries: Vec::new(), next: 0 });
        Ok(self)
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.ck.bytes).sum()
    }

    /// (checkpoints, bytes, budget) on disk
    pub fn disk(&self) -> Option<(usize, usize, usize)> {
        self.disk.as_ref().map(|d| (d.entries.len(), d.entries.iter().map(|e| e.bytes).sum(), d.budget))
    }

    /// (tokens, bytes, last use - higher is more recent, on disk) per entry, the most recently used first
    pub fn list(&self) -> Vec<(usize, usize, u64, bool)> {
        let mut v: Vec<_> = self.entries.iter().map(|e| (e.tokens.len(), e.ck.bytes, e.used, false)).collect();
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
                let _ = std::fs::remove_file(&e.path);
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
    pub fn with<R>(&mut self, hit: Hit, f: impl FnOnce(&Checkpoint) -> R) -> Result<R, String> {
        self.clock += 1;
        match hit {
            Hit::Ram(i) => {
                self.entries[i].used = self.clock;
                Ok(f(&self.entries[i].ck))
            }
            Hit::Disk(i) => {
                let d = self.disk.as_mut().ok_or("no disk tier")?;
                d.entries[i].used = self.clock;
                let file = std::fs::File::open(&d.entries[i].path).map_err(|e| format!("{}: {e}", d.entries[i].path.display()))?;
                let ck = Checkpoint::read_from(&mut std::io::BufReader::with_capacity(8 << 20, file)).map_err(|e| e.to_string())?;
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
        if ck.bytes > self.budget {
            let c = self.clock + 1;
            self.clock = c;
            return self.spill(Entry { tokens, ck, used: c });
        }
        while self.bytes() + ck.bytes > self.budget {
            let Some((i, _)) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used) else { break };
            let e = self.entries.swap_remove(i);
            self.evictions += 1;
            self.spill(e);
        }
        self.clock += 1;
        self.entries.push(Entry { tokens, ck, used: self.clock });
        true
    }

    /// An entry out of memory onto the disk tier (its own least recently used files removed to make room)
    fn spill(&mut self, e: Entry) -> bool {
        let Some(d) = &mut self.disk else { return false };
        if e.ck.bytes > d.budget || d.entries.iter().any(|x| x.tokens == e.tokens) {
            return false;
        }
        while d.entries.iter().map(|x| x.bytes).sum::<usize>() + e.ck.bytes > d.budget {
            let Some((i, _)) = d.entries.iter().enumerate().min_by_key(|(_, x)| x.used) else { break };
            let x = d.entries.swap_remove(i);
            let _ = std::fs::remove_file(&x.path);
        }
        d.next += 1;
        let path = d.dir.join(format!("{}.nsck", d.next));
        let written = std::fs::File::create(&path).and_then(|f| {
            let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
            e.ck.write_to(&mut w)?;
            std::io::Write::flush(&mut w)
        });
        match written {
            Ok(()) => {
                d.entries.push(DiskEntry { tokens: e.tokens, path, bytes: e.ck.bytes, used: e.used });
                true
            }
            Err(err) => {
                eprintln!("[prompt cache: writing {} failed: {err}]", path.display());
                let _ = std::fs::remove_file(&path);
                false
            }
        }
    }
}
