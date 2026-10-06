//! The server's prompt cache: checkpoints of the conversation state (`Glm::save`) at token prefixes, in host memory,
//! least recently used out first once they pass the budget. A request mounts the longest cached prefix of its
//! tokens and reads only the rest.

use ns_engine::glm5next::Checkpoint;

struct Entry {
    tokens: Vec<u32>,
    ck: Checkpoint,
    used: u64,
}

pub struct PromptCache {
    entries: Vec<Entry>,
    budget: usize,
    clock: u64,
    pub evictions: u64,
}

impl PromptCache {
    pub fn new(budget: usize) -> PromptCache {
        PromptCache { entries: Vec::new(), budget, clock: 0, evictions: 0 }
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

    /// The longest entry that is a proper prefix of `ids` (some token is left to read): its index and length.
    pub fn best(&self, ids: &[u32]) -> Option<(usize, usize)> {
        self.entries.iter().enumerate()
            .filter(|(_, e)| e.tokens.len() < ids.len() && ids.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, e)| (i, e.tokens.len()))
    }

    /// Entry `i` (from `best`), marked as used.
    pub fn get(&mut self, i: usize) -> &Checkpoint {
        self.clock += 1;
        self.entries[i].used = self.clock;
        &self.entries[i].ck
    }

    /// Whether `tokens` are cached already (then marked as used).
    pub fn touch(&mut self, tokens: &[u32]) -> bool {
        self.clock += 1;
        let c = self.clock;
        match self.entries.iter_mut().find(|e| e.tokens == tokens) {
            Some(e) => {
                e.used = c;
                true
            }
            None => false,
        }
    }

    /// Keeps `ck` for `tokens`, making room by evicting the least recently used. A checkpoint larger than the
    /// whole budget is dropped.
    pub fn put(&mut self, tokens: Vec<u32>, ck: Checkpoint) -> bool {
        if ck.bytes > self.budget {
            return false;
        }
        self.entries.retain(|e| e.tokens != tokens);
        while self.bytes() + ck.bytes > self.budget {
            let Some((i, _)) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used) else { break };
            self.entries.swap_remove(i);
            self.evictions += 1;
        }
        self.clock += 1;
        self.entries.push(Entry { tokens, ck, used: self.clock });
        true
    }
}
