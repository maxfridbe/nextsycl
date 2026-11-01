//! The draws, as upstream's sampling.py `distribution` makes them, on the host over a phase's candidates (the score's:
//! the text tokens and ABC_END; the codes': the 32768 codes and MUSIC_END): the end forbidden before `min_tokens`, the
//! repetition penalty over the last `window` tokens drawn (a logit divided by penalty^count, a negative one
//! multiplied), the temperature, top-k (every candidate at or above the k-th), top-p over what is left in descending
//! order (the first always kept - the first three for cot off's historical arithmetic), then one draw from the softmax.
//! The generator is this engine's (mt19937 seeded by the request; the reference's torch.multinomial on its device draws
//! other numbers, so a song is the same song only within one engine).

use nextsycl_diffusion::torch_rng::Mt19937;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub penalty: f32,
    pub window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}

/// the score's (GenerationConfig.abc)
pub const ABC: Sampling = Sampling { temperature: 0.7, top_p: 0.9, top_k: 30, penalty: 1.005, window: 100, min_tokens: 32, max_tokens: 4096 };
/// the codes' (GenerationConfig.semantic): 25 a second, six minutes at most
pub const SEMANTIC: Sampling = Sampling { temperature: 1.0, top_p: 0.95, top_k: 100, penalty: 1.2, window: 50, min_tokens: 200, max_tokens: 9000 };

pub struct Rng(Mt19937);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(Mt19937::new(seed))
    }

    /// uniform in [0, 1), 53 bits
    pub fn uniform(&mut self) -> f64 {
        let a = (self.0.next_u32() >> 5) as u64;
        let b = (self.0.next_u32() >> 6) as u64;
        ((a << 26) | b) as f64 / (1u64 << 53) as f64
    }
}

/// The next candidate: `scores` the candidates' logits (changed in place), `end` the end's index among them,
/// `history` the candidates drawn so far (indexes), `step` how many, `keep` the candidates top-p always keeps (1; 3 for
/// cot off)
pub fn draw(scores: &mut [f32], end: usize, s: &Sampling, history: &[usize], step: usize, keep: usize, rng: &mut Rng) -> usize {
    if step < s.min_tokens {
        scores[end] = f32::NEG_INFINITY;
    }
    if s.penalty != 1.0 {
        let recent = &history[history.len().saturating_sub(s.window)..];
        let mut counts: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();
        for t in recent {
            *counts.entry(*t).or_default() += 1;
        }
        for (t, c) in counts {
            let a = s.penalty.powi(c);
            let v = scores[t];
            scores[t] = if v < 0.0 { v * a } else { v / a };
        }
    }
    if s.temperature == 0.0 {
        return argmax(scores);
    }
    if s.temperature != 1.0 {
        for v in scores.iter_mut() {
            *v /= s.temperature;
        }
    }
    // top-k: the k-th largest is the threshold; everything at or above it stays
    let k = s.top_k.min(scores.len()).max(1);
    let mut copy: Vec<f32> = scores.to_vec();
    let (_, kth, _) = copy.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let thr = *kth;
    let mut cand: Vec<(f32, usize)> = scores.iter().enumerate().filter(|(_, v)| **v >= thr && v.is_finite()).map(|(i, v)| (*v, i)).collect();
    if cand.is_empty() {
        return argmax(scores);
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mx = cand[0].0 as f64;
    let mut p: Vec<f64> = cand.iter().map(|(v, _)| (*v as f64 - mx).exp()).collect();
    let z: f64 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= z;
    }
    if s.top_p < 1.0 {
        // removed where the mass before a candidate is past top_p
        let mut before = 0f64;
        let mut n = p.len();
        for (i, v) in p.iter().enumerate() {
            if i >= keep && before > s.top_p as f64 {
                n = i;
                break;
            }
            before += v;
        }
        p.truncate(n);
        cand.truncate(n);
    }
    let z: f64 = p.iter().sum();
    let mut u = rng.uniform() * z;
    for (i, v) in p.iter().enumerate() {
        u -= v;
        if u < 0.0 {
            return cand[i].1;
        }
    }
    cand[cand.len() - 1].1
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_and_the_end_rule() {
        let mut rng = Rng::new(1);
        let s = Sampling { temperature: 1.0, top_p: 1.0, top_k: 2, penalty: 1.0, window: 10, min_tokens: 5, max_tokens: 10 };
        for _ in 0..50 {
            let mut v = vec![0.0, 5.0, 4.9, -3.0, 9.0];
            // the end (index 4) is forbidden before min_tokens: the two best left are 1 and 2
            let d = draw(&mut v, 4, &s, &[], 0, 1, &mut rng);
            assert!(d == 1 || d == 2, "{d}");
        }
        let mut v = vec![0.0, 5.0, 4.9, -3.0, 9.0];
        assert_eq!(draw(&mut v, 4, &Sampling { temperature: 0.0, ..s }, &[], 6, 1, &mut rng), 4);
    }

    #[test]
    fn the_penalty_and_top_p() {
        let mut rng = Rng::new(2);
        // a penalty of 100 on the best: the second wins every greedy draw
        let s = Sampling { temperature: 0.0, top_p: 1.0, top_k: 10, penalty: 100.0, window: 4, min_tokens: 0, max_tokens: 10 };
        let mut v = vec![1.0, 3.0, 2.0];
        assert_eq!(draw(&mut v, 0, &s, &[1], 1, 1, &mut rng), 2);
        // top-p 0.5 over (0.9, 0.05, 0.05): the first alone
        let s = Sampling { temperature: 1.0, top_p: 0.5, penalty: 1.0, ..s };
        for _ in 0..20 {
            let mut v = vec![(0.9f32).ln(), (0.05f32).ln(), (0.05f32).ln()];
            assert_eq!(draw(&mut v, 2, &s, &[], 0, 1, &mut rng), 0);
        }
    }
}
