//! The text stream's sampling (haru/sampling.py): temperature, top-k / top-p, the DRY repetition penalty over the
//! words it has said (PAD / EOP / EOS never count), and its incremental text (a run of silent frames is one space; a
//! character split over tokens waits for its last byte). And the session's random numbers: the noise the head starts
//! from and the codec's latent noise.

use std::collections::HashMap;

use nextsycl_tok::Tokenizer;

/// splitmix64 for the seed, xoshiro256** for the stream; normals by Box-Muller
pub struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng { s: [next(), next(), next(), next()], spare: None }
    }

    pub fn u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        self.spare = Some((r * s) as f32);
        (r * c) as f32
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

/// DRY: discourage a token that would extend a sequence already said (multiplier 0.8, base 1.75, allowed length 1,
/// lookback 1024, longest match 32, at most 32)
#[derive(Clone, Copy, Debug)]
pub struct Dry {
    pub multiplier: f32,
    pub base: f32,
    pub allowed: usize,
    pub lookback: usize,
    pub max_match: usize,
    pub max_penalty: f32,
}

impl Default for Dry {
    fn default() -> Dry {
        Dry { multiplier: 0.8, base: 1.75, allowed: 1, lookback: 1024, max_match: 32, max_penalty: 32.0 }
    }
}

/// {token: penalty} for the tokens that would extend an earlier repeat of the history's end
pub fn dry_penalties(history: &[u32], c: &Dry) -> HashMap<u32, f32> {
    let n = history.len();
    let mut longest: HashMap<u32, usize> = HashMap::new();
    if n < c.allowed.max(1) {
        return HashMap::new();
    }
    let last = history[n - 1];
    let lo = (n as isize - 1 - c.lookback as isize).max(0) as usize;
    for j in (lo..n.saturating_sub(1)).rev() {
        if history[j] != last {
            continue;
        }
        let mut len = 1;
        while len < c.max_match && j >= len && history[j - len] == history[n - 1 - len] {
            len += 1;
        }
        if len >= c.allowed {
            let cand = history[j + 1];
            if len > *longest.get(&cand).unwrap_or(&0) {
                longest.insert(cand, len);
            }
        }
    }
    longest.into_iter().map(|(t, len)| (t, c.max_penalty.min(c.multiplier * c.base.powi((len - c.allowed) as i32)))).collect()
}

/// The running DRY history of one text stream; `exempt` tokens never count
pub struct DryState {
    pub c: Dry,
    exempt: Vec<u32>,
    history: Vec<u32>,
    penalties: HashMap<u32, f32>,
}

impl DryState {
    pub fn new(c: Dry, exempt: Vec<u32>) -> DryState {
        DryState { c, exempt, history: Vec::new(), penalties: HashMap::new() }
    }

    pub fn update(&mut self, t: u32) {
        if self.exempt.contains(&t) {
            return;
        }
        self.history.push(t);
        let keep = self.c.lookback + self.c.max_match + 1;
        if self.history.len() > keep {
            let extra = self.history.len() - keep;
            self.history.drain(..extra);
        }
        self.penalties = dry_penalties(&self.history, &self.c).into_iter().filter(|(t, _)| !self.exempt.contains(t)).collect();
    }

    pub fn apply(&self, logits: &mut [f32]) {
        for (t, p) in &self.penalties {
            if let Some(l) = logits.get_mut(*t as usize) {
                *l -= p;
            }
        }
    }
}

/// A token from float32 logits: greedy at temperature <= 0, else top-k (0: all) then top-p (1: all) of the softmax
pub fn draw(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
    }
    let k = if top_k == 0 || top_k >= logits.len() { logits.len() } else { top_k };
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_by(cmp);
    let m = logits[idx[0] as usize] / temperature;
    let mut p: Vec<f64> = idx.iter().map(|i| ((logits[*i as usize] / temperature - m) as f64).exp()).collect();
    let sum: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= sum);
    if top_p > 0.0 && top_p < 1.0 {
        let mut acc = 0.0;
        for x in p.iter_mut() {
            let before = acc;
            acc += *x;
            if before >= top_p as f64 {
                *x = 0.0;
            }
        }
    }
    let total: f64 = p.iter().sum();
    let mut u = rng.uniform() * total;
    for (i, x) in p.iter().enumerate() {
        if u < *x {
            return idx[i];
        }
        u -= x;
    }
    idx[p.iter().rposition(|x| *x > 0.0).unwrap_or(0)]
}

/// The text stream's new text a token at a time: a run of silent frames (PAD / EOP) is one space; bytes of a
/// character split over tokens are held until it is whole
pub struct Detok {
    silent: Vec<u32>,
    pending: Vec<u32>,
    gap: bool,
}

impl Detok {
    pub fn new(silent: Vec<u32>) -> Detok {
        Detok { silent, pending: Vec::new(), gap: false }
    }

    pub fn push(&mut self, tok: &Tokenizer, t: u32) -> String {
        if self.silent.contains(&t) {
            if self.gap {
                return String::new();
            }
            self.gap = true;
            return " ".into();
        }
        self.gap = false;
        self.pending.push(t);
        let b = tok.decode_bytes(&self.pending);
        match std::str::from_utf8(&b) {
            Ok(s) => {
                self.pending.clear();
                s.to_string()
            }
            // a character not finished yet (at most 4 bytes, so a few tokens)
            Err(e) if e.error_len().is_none() && self.pending.len() < 8 => String::new(),
            Err(_) => {
                self.pending.clear();
                String::from_utf8_lossy(&b).into_owned()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_penalizes_what_would_repeat() {
        // ... 5 6 7 ... 5 6 -> 7 would repeat "5 6 7": length 2 (5 6) -> 0.8 * 1.75^(2 - 1)
        let p = dry_penalties(&[5, 6, 7, 1, 5, 6], &Dry::default());
        assert!((p[&7] - 1.4).abs() < 1e-5, "{p:?}");
        assert!(dry_penalties(&[1, 2, 3], &Dry::default()).is_empty());
    }

    #[test]
    fn draws_are_the_softmax_and_greedy_is_argmax() {
        let mut r = Rng::new(7);
        assert_eq!(draw(&[0.1, 3.0, -1.0], 0.0, 25, 1.0, &mut r), 1);
        let mut counts = [0usize; 3];
        for _ in 0..20000 {
            counts[draw(&[0.0, (2.0f32).ln(), -100.0], 1.0, 2, 1.0, &mut r) as usize] += 1;
        }
        // p = 1/3, 2/3, 0 (top-k 2 drops the last)
        assert!((counts[1] as f64 / 20000.0 - 2.0 / 3.0).abs() < 0.02 && counts[2] == 0, "{counts:?}");
        let n: Vec<f32> = r.normals(20000);
        let mean = n.iter().sum::<f32>() / n.len() as f32;
        let var = n.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n.len() as f32;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "{mean} {var}");
    }
}
