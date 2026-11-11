//! What makes rapid, related requests cheap (a live editor's: the same prompt at another seed or size, the same
//! picture edited again, a preview then its polish), and the LoRAs on the blocks:
//!
//! - `Lru`: a few results kept by key, the least recently used dropped first; capacity 0 keeps nothing.
//! - the text cache (NS_H3I_TEXT_CACHE entries): the text encoder's rows for a prompt and its pictures, and the token
//!   refiner's output for them under the LoRAs then on the blocks (on the card: [T, 5376] bfloat16, 1 - 17 MB);
//! - the picture cache (NS_H3I_IMAGE_CACHE entries): a picture's VAE latents at a size, as an edit's reference or as an
//!   img2img start (host memory: 24 x H/16 x W/16 floats, 0.4 MB at 1 MP);
//! - `Loras`: the card's LoRA (role `lora`, at NS_H3I_LORA_STRENGTH) and a preset's, on the side path, with a request's
//!   own stacked on top for as long as requests keep asking for the same ones (they are read once, not per request).

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use nextsycl_h3::device::Device;
use nextsycl_h3::lora::{BlockLora, LoraSet};
use nextsycl_h3::safetensors::Checkpoint;
use nextsycl_image::{LoraUse, Picture};

/// The last `cap` values by key; a hit becomes the most recent
pub struct Lru<V> {
    cap: usize,
    items: Vec<(u64, V)>,
    pub hits: u64,
    pub misses: u64,
}

impl<V> Lru<V> {
    pub fn new(cap: usize) -> Lru<V> {
        Lru { cap, items: Vec::new(), hits: 0, misses: 0 }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The value under `key`, now the most recent (counted as a hit or a miss)
    pub fn get(&mut self, key: u64) -> Option<&mut V> {
        match self.items.iter().position(|(k, _)| *k == key) {
            Some(i) => {
                self.hits += 1;
                let it = self.items.remove(i);
                self.items.push(it);
                self.items.last_mut().map(|(_, v)| v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Keeps `value` under `key` (replacing one there), dropping the least recent beyond the capacity
    pub fn put(&mut self, key: u64, value: V) {
        if self.cap == 0 {
            return;
        }
        self.items.retain(|(k, _)| *k != key);
        self.items.push((key, value));
        while self.items.len() > self.cap {
            self.items.remove(0);
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

fn hasher() -> std::collections::hash_map::DefaultHasher {
    std::collections::hash_map::DefaultHasher::new()
}

/// A picture's identity: its size, channels and bytes
pub fn picture_key(p: &Picture) -> u64 {
    let mut h = hasher();
    (p.width, p.height, p.channels).hash(&mut h);
    p.data.hash(&mut h);
    h.finish()
}

/// The text encoder's input: the prompt and its pictures (each by `picture_key`)
pub fn text_key(prompt: &str, pictures: &[u64]) -> u64 {
    let mut h = hasher();
    "text".hash(&mut h);
    prompt.hash(&mut h);
    pictures.hash(&mut h);
    h.finish()
}

/// A picture's latents at a size, for a use ("ref": an edit's reference at its own VAE size; "init": an img2img start
/// at the canvas)
pub fn latent_key(picture: u64, w: u32, h: u32, purpose: &str) -> u64 {
    let mut hs = hasher();
    (picture, w, h, purpose).hash(&mut hs);
    hs.finish()
}

/// LoRAs as a key (0: none)
pub fn loras_key(loras: &[LoraUse]) -> u64 {
    if loras.is_empty() {
        return 0;
    }
    let mut h = hasher();
    for l in loras {
        (l.path.as_os_str(), l.scale.to_bits()).hash(&mut h);
    }
    h.finish().max(1)
}

/// The LoRAs on the blocks: those loaded with the engine (`base` counts per block and linear) and a request's on top
pub struct Loras {
    pub set: Option<LoraSet>,
    base: Vec<[usize; 4]>,
    base_layers: usize,
    /// the request's LoRAs stacked now
    pub extra: Vec<LoraUse>,
}

fn empty(blocks: usize, refiner: usize) -> LoraSet {
    let none = || -> BlockLora { std::array::from_fn(|_| Vec::new()) };
    LoraSet { blocks: (0..blocks).map(|_| none()).collect(), refiner: (0..refiner).map(|_| none()).collect(), max_rank: 0, layers: 0 }
}

/// Each block's (then each refiner block's) number of LoRAs per linear
fn counts(set: &LoraSet) -> Vec<[usize; 4]> {
    set.blocks.iter().chain(&set.refiner).map(|b| std::array::from_fn(|j| b[j].len())).collect()
}

impl Loras {
    pub fn new(set: Option<LoraSet>) -> Loras {
        let base = set.as_ref().map(counts).unwrap_or_default();
        let base_layers = set.as_ref().map_or(0, |s| s.layers);
        Loras { set, base, base_layers, extra: Vec::new() }
    }

    /// The key of what is on the blocks beyond the engine's own (0: nothing)
    pub fn extra_key(&self) -> u64 {
        loras_key(&self.extra)
    }

    /// Back to the engine's own LoRAs (a request's dropped, their card memory freed)
    pub fn trim(&mut self) {
        if let Some(s) = self.set.as_mut() {
            let base = &self.base;
            for (i, b) in s.blocks.iter_mut().chain(s.refiner.iter_mut()).enumerate() {
                for (j, v) in b.iter_mut().enumerate() {
                    v.truncate(base.get(i).map_or(0, |c| c[j]));
                }
            }
            s.layers = self.base_layers;
        }
        if self.base.is_empty() {
            self.set = None;
        }
        self.extra.clear();
    }

    /// A request's LoRAs on top of the engine's (kept while requests keep asking for the same ones); `log` says what
    /// was read
    pub fn want(&mut self, dev: &Arc<Device>, want: &[LoraUse], blocks: usize, refiner: usize, max_rank: usize, log: &mut dyn FnMut(String))
                -> nextsycl_h3::Result<()> {
        if self.extra == want {
            return Ok(());
        }
        self.trim();
        for l in want {
            let add = LoraSet::load(dev, &Checkpoint::open(&l.path)?, l.scale, blocks, refiner)?;
            if add.max_rank > max_rank {
                self.trim();
                return Err(nextsycl_h3::Error(format!("{}: rank {} is over {max_rank}", l.path.display(), add.max_rank)));
            }
            log(format!("request LoRA {} x {}: {} layers, rank up to {}", l.path.display(), l.scale, add.layers, add.max_rank));
            if self.set.is_none() {
                self.set = Some(empty(blocks, refiner));
                self.base = vec![[0; 4]; blocks + refiner];
            }
            if let Some(s) = self.set.as_mut() {
                s.stack(add);
            }
            self.extra.push(l.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_least_recent_goes_first() {
        let mut c: Lru<&str> = Lru::new(2);
        c.put(1, "a");
        c.put(2, "b");
        assert_eq!(c.get(1).copied(), Some("a"));
        c.put(3, "c");
        // 2 was the least recently used
        assert!(c.get(2).is_none());
        assert_eq!((c.get(1).copied(), c.get(3).copied()), (Some("a"), Some("c")));
        assert_eq!((c.hits, c.misses, c.len()), (3, 1, 2));
        c.put(3, "d");
        assert_eq!((c.get(3).copied(), c.len()), (Some("d"), 2));
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn capacity_0_keeps_nothing() {
        let mut c: Lru<u32> = Lru::new(0);
        c.put(1, 1);
        assert!(c.get(1).is_none() && c.is_empty() && c.cap() == 0);
    }

    #[test]
    fn keys_tell_inputs_apart() {
        let p = Picture { width: 2, height: 1, channels: 3, data: vec![1, 2, 3, 4, 5, 6] };
        let mut q = p.clone();
        q.data[5] = 7;
        assert_ne!(picture_key(&p), picture_key(&q));
        assert_eq!(picture_key(&p), picture_key(&p.clone()));
        let (a, b) = (picture_key(&p), picture_key(&q));
        assert_ne!(text_key("a cat", &[a]), text_key("a cat", &[b]));
        assert_ne!(text_key("a cat", &[]), text_key("a cat.", &[]));
        assert_ne!(latent_key(a, 1024, 1024, "init"), latent_key(a, 1024, 1024, "ref"));
        assert_ne!(latent_key(a, 1024, 1024, "init"), latent_key(a, 1024, 1056, "init"));
        let l = |s: f32| LoraUse { name: "x".into(), path: "/x.safetensors".into(), scale: s };
        assert_eq!(loras_key(&[]), 0);
        assert_ne!(loras_key(&[l(1.0)]), loras_key(&[l(0.5)]));
        assert_ne!(loras_key(&[l(1.0)]), 0);
    }

    #[test]
    fn trimming_keeps_the_engines_own_count() {
        // no device needed: an empty set's counts and a trim back to them
        let mut l = Loras::new(Some(empty(3, 2)));
        assert_eq!(l.base, vec![[0; 4]; 5]);
        l.extra.push(LoraUse { name: "x".into(), path: "/x".into(), scale: 1.0 });
        assert_ne!(l.extra_key(), 0);
        l.trim();
        assert!(l.extra.is_empty() && l.extra_key() == 0 && l.set.is_some());
        // without LoRAs of its own the set goes with the request's
        let mut none = Loras::new(None);
        none.set = Some(empty(3, 2));
        none.base = vec![[0; 4]; 5];
        none.trim();
        assert!(none.set.is_some(), "a set made for a request keeps its (empty) base until the engine has none");
        let mut fresh = Loras::new(None);
        fresh.trim();
        assert!(fresh.set.is_none());
    }
}
