//! The voices' cache: a call's prompt (the character card, then a saved voice's reference) is most of its start, and
//! the same voice and card come back call after call. So the prompt's result is kept on the host, per loaded model:
//! the language model's state after it (each DeltaNet layer's convolution rows and recurrence, each attention layer's
//! keys and values of the prompt's positions), its last hidden state and the random numbers' state after the
//! reference's noise - keyed by the recording (all its samples hashed), the card's text, speak-first, the latent noise,
//! the prompt's seed and the path that made it. A repeat call copies the state back (~75 MB for a 39 s voice, a few
//! ms) instead of encoding the reference and running ~560 positions. Restored, the state is the same bytes, so a hit
//! gives what the cold prompt gave.
//!
//! Its bound: NS_HARU_VOICE_CACHE MiB of host memory (default 1024; 0: no cache), the least recently used prompt
//! leaving first. A 39 s voice's prompt is ~75 MB (the DeltaNet states 50 MB whatever the voice, the keys and values
//! 32 KB a position), a 60 s one ~85 MB: about a dozen voices in the default. The reference's encoding (its latent
//! means, 256 bytes a frame) is kept apart, for the last 64 voices, so a new card on a known voice skips the encoder.
//! Nothing is on the card, and the cache goes with the model when it is unloaded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::lm::Snapshot;
use crate::sample::Rng;

/// The voices whose encodings are kept
const MEANS: usize = 64;

/// What makes a prompt: equal keys, equal prompts
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PromptKey {
    /// the reference's fingerprint (its length and every sample's bits)
    pub voice: Option<u64>,
    pub card: String,
    pub speak_first: bool,
    pub noise: bool,
    /// the reference's noise's seed
    pub seed: u64,
    /// (batch encode, wide passes): the paths' prompts differ in their last bits, so they are kept apart
    pub path: (bool, bool),
}

/// A prompt's result
pub struct Entry {
    pub snapshot: Snapshot,
    pub hidden: Vec<f32>,
    /// the random numbers after the reference's noise
    pub rng: Rng,
}

impl Entry {
    fn bytes(&self) -> usize {
        self.snapshot.bytes() + self.hidden.len() * 4
    }
}

#[derive(Default)]
struct Inner {
    prompts: HashMap<PromptKey, (Arc<Entry>, u64)>,
    means: HashMap<u64, (Arc<Vec<f32>>, u64)>,
    bytes: usize,
    tick: u64,
    hits: u64,
    misses: u64,
}

pub struct VoiceCache {
    /// the bound (bytes; 0: nothing kept)
    pub limit: usize,
    inner: Mutex<Inner>,
}

/// A recording's fingerprint: FNV-1a over its length and every sample's bits
pub fn fingerprint(x: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in (x.len() as u64).to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    for v in x {
        h = (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl VoiceCache {
    pub fn new(limit: usize) -> VoiceCache {
        VoiceCache { limit, inner: Mutex::new(Inner::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The prompt kept for `key` (and it is now the most recently used)
    pub fn get(&self, key: &PromptKey) -> Option<Arc<Entry>> {
        let mut c = self.lock();
        c.tick += 1;
        let t = c.tick;
        let e = c.prompts.get_mut(key).map(|(e, used)| {
            *used = t;
            e.clone()
        });
        if e.is_some() {
            c.hits += 1;
        } else {
            c.misses += 1;
        }
        e
    }

    /// Keep a prompt, the least recently used ones leaving for it (one larger than the bound is not kept)
    pub fn put(&self, key: PromptKey, e: Entry) {
        let n = e.bytes();
        if n > self.limit {
            return;
        }
        let mut c = self.lock();
        if let Some((old, _)) = c.prompts.remove(&key) {
            c.bytes -= old.bytes();
        }
        while c.bytes + n > self.limit {
            let Some(k) = c.prompts.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| k.clone()) else { break };
            if let Some((old, _)) = c.prompts.remove(&k) {
                c.bytes -= old.bytes();
            }
        }
        c.tick += 1;
        let t = c.tick;
        c.bytes += n;
        c.prompts.insert(key, (Arc::new(e), t));
    }

    /// A voice's encoding (its latent means), if kept
    pub fn means(&self, voice: u64) -> Option<Vec<f32>> {
        if self.limit == 0 {
            return None;
        }
        let mut c = self.lock();
        c.tick += 1;
        let t = c.tick;
        c.means.get_mut(&voice).map(|(m, used)| {
            *used = t;
            m.to_vec()
        })
    }

    pub fn keep_means(&self, voice: u64, m: &[f32]) {
        if self.limit == 0 {
            return;
        }
        let mut c = self.lock();
        while c.means.len() >= MEANS {
            let Some(k) = c.means.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k) else { break };
            c.means.remove(&k);
        }
        c.tick += 1;
        let t = c.tick;
        c.means.insert(voice, (Arc::new(m.to_vec()), t));
    }

    /// (prompts kept, their bytes, hits, misses)
    pub fn stats(&self) -> (usize, usize, u64, u64) {
        let c = self.lock();
        (c.prompts.len(), c.bytes, c.hits, c.misses)
    }

    /// Everything dropped
    pub fn clear(&self) {
        let mut c = self.lock();
        c.prompts.clear();
        c.means.clear();
        c.bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(card: &str) -> PromptKey {
        PromptKey { voice: Some(1), card: card.into(), speak_first: false, noise: true, seed: 7, path: (true, true) }
    }

    fn entry(floats: usize) -> Entry {
        Entry { snapshot: Snapshot::empty(), hidden: vec![0.0; floats], rng: Rng::new(1) }
    }

    #[test]
    fn the_least_recently_used_prompt_leaves_for_a_new_one_past_the_bound() {
        let c = VoiceCache::new(3 * 400);
        for k in ["a", "b", "c"] {
            c.put(key(k), entry(100));
        }
        assert!(c.get(&key("a")).is_some());
        c.put(key("d"), entry(100));
        assert!(c.get(&key("b")).is_none(), "b was the least recently used");
        assert!(c.get(&key("a")).is_some() && c.get(&key("c")).is_some() && c.get(&key("d")).is_some());
        assert_eq!(c.stats().1, 1200);
        // larger than the bound: not kept, nothing evicted for it
        c.put(key("e"), entry(1000));
        assert_eq!(c.stats().0, 3);
    }

    #[test]
    fn a_prompt_differs_by_voice_card_and_seed() {
        let c = VoiceCache::new(1 << 20);
        c.put(key("a"), entry(1));
        assert!(c.get(&PromptKey { seed: 8, ..key("a") }).is_none());
        assert!(c.get(&PromptKey { voice: Some(2), ..key("a") }).is_none());
        assert!(c.get(&PromptKey { path: (false, false), ..key("a") }).is_none());
        assert!(c.get(&key("a")).is_some());
    }

    #[test]
    fn a_fingerprint_reads_every_sample() {
        let mut x = vec![0.25f32; 10_000];
        let a = fingerprint(&x);
        x[5_003] = 0.2500001;
        assert_ne!(a, fingerprint(&x));
        assert_ne!(fingerprint(&[0.0; 3]), fingerprint(&[0.0; 4]));
    }

    #[test]
    fn no_cache_keeps_nothing() {
        let c = VoiceCache::new(0);
        c.put(key("a"), entry(1));
        c.keep_means(1, &[1.0]);
        assert!(c.get(&key("a")).is_none() && c.means(1).is_none());
    }
}
