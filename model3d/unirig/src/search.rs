//! The decoding rules, on the host (transformers 4.51's, which UniRig runs): the logits processors - the repetition
//! penalty, the grammar mask (`tokens::allowed`), temperature, top-k, top-p - and the two searches: greedy, and the
//! beam search with sampling UniRig's inference config asks for (15 beams; 2 x 15 continuations drawn without
//! replacement from the softmax of the beams' accumulated log-probabilities, the best 15 unfinished kept running, the
//! finished kept by their length-normalized score).

use crate::ops::Rng;
use crate::tokens::{allowed, EOS, VOCAB};

/// The sampling settings (UniRig's ar_inference_articulationxl.yaml)
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub beams: usize,
    pub sample: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub max_new: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings { beams: 15, sample: true, top_k: 5, top_p: 0.95, repetition_penalty: 3.0, temperature: 1.5, max_new: 2048 }
    }
}

/// The repetition penalty on one row of scores over the tokens `seen` (each once): a negative score times p, a
/// positive one over p
pub fn repetition(scores: &mut [f32], seen: &[u32], p: f32) {
    let mut done = [false; VOCAB];
    for &t in seen {
        let t = t as usize;
        if t < VOCAB && !done[t] {
            done[t] = true;
            scores[t] = if scores[t] < 0.0 { scores[t] * p } else { scores[t] / p };
        }
    }
}

/// The grammar: tokens it does not allow after `start ++ seq` to -inf
pub fn grammar(scores: &mut [f32], start: &[u32], seq: &[u32]) {
    let ids: Vec<u32> = start.iter().chain(seq).copied().collect();
    let ok = allowed(&ids);
    for (s, k) in scores.iter_mut().zip(ok) {
        if !k {
            *s += f32::NEG_INFINITY;
        }
    }
}

/// log_softmax of a row (float32, as torch: x - max - log(sum(exp(x - max))))
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = x.iter().map(|v| (v - m).exp()).sum();
    let l = s.ln();
    x.iter().map(|v| v - m - l).collect()
}

/// Top-k (keeping at least `keep`): scores below the k-th largest to -inf
pub fn top_k(scores: &mut [f32], k: usize, keep: usize) {
    let k = k.max(keep).min(scores.len());
    let mut s = scores.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    let kth = s[k - 1];
    for x in scores.iter_mut() {
        if *x < kth {
            *x = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (keeping at least `keep`): sorted ascending, the tokens whose cumulative probability stays within 1 - p go
pub fn top_p(scores: &mut [f32], p: f32, keep: usize) {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let sorted: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    let m = sorted.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = sorted.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    let mut cum = 0f32;
    let n = scores.len();
    for (r, &i) in idx.iter().enumerate() {
        cum += e[r] / z;
        if cum <= 1.0 - p && r < n.saturating_sub(keep) {
            scores[i] = f32::NEG_INFINITY;
        }
    }
}

/// The processors of a beam step on log-probabilities (rows of `beams`), in transformers' order
fn process(scores: &mut [f32], start: &[u32], seq: &[u32], s: &Settings, n_eos: usize) {
    repetition(scores, seq, s.repetition_penalty);
    grammar(scores, start, seq);
    if s.sample {
        let keep = if s.beams > 1 { n_eos + 1 } else { 1 };
        if s.temperature != 1.0 {
            scores.iter_mut().for_each(|x| *x /= s.temperature);
        }
        if s.top_k > 0 {
            top_k(scores, s.top_k, keep);
        }
        if s.top_p < 1.0 {
            top_p(scores, s.top_p, keep);
        }
    }
}

/// One greedy step: the processed logits' best token (the repetition penalty and the grammar on the raw logits)
pub fn greedy(logits: &[f32], start: &[u32], seq: &[u32], s: &Settings) -> u32 {
    let mut x = logits.to_vec();
    repetition(&mut x, seq, s.repetition_penalty);
    grammar(&mut x, start, seq);
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
}

/// `k` distinct indices drawn from the distribution `p` (unnormalized weights) without replacement, in the order of
/// their keys (p / Exp(1) largest first: torch.multinomial's)
fn draw(p: &[f32], k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut keys: Vec<(f64, usize)> = p.iter().enumerate().filter(|(_, w)| **w > 0.0).map(|(i, w)| {
        let e = -(rng.unit().max(1e-300)).ln();
        (*w as f64 / e, i)
    }).collect();
    keys.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    keys.into_iter().take(k).map(|x| x.1).collect()
}

/// A beam search in progress
pub struct Beams {
    pub s: Settings,
    start: Vec<u32>,
    /// the running beams' tokens (generated only) and scores
    pub running: Vec<Vec<u32>>,
    pub scores: Vec<f32>,
    /// the finished: (tokens with eos, length-normalized score)
    finished: Vec<(Vec<u32>, f32)>,
    pub done: bool,
}

/// What a step decided: for each new running beam, the old beam it continues and its new token
pub struct Step {
    pub parents: Vec<usize>,
    pub tokens: Vec<u32>,
}

impl Beams {
    pub fn new(s: Settings, start: &[u32]) -> Beams {
        let mut scores = vec![-1e9f32; s.beams];
        scores[0] = 0.0;
        Beams { s, start: start.to_vec(), running: vec![Vec::new(); s.beams], scores, finished: Vec::new(), done: false }
    }

    /// One step from the running beams' logits [beams, 267] (`cur` tokens generated so far): the beams to run next
    pub fn step(&mut self, logits: &[f32], rng: &mut Rng) -> Step {
        let nb = self.s.beams;
        let cur = self.running[0].len();
        let mut acc = Vec::with_capacity(nb * VOCAB);
        for b in 0..nb {
            let mut lp = log_softmax(&logits[b * VOCAB..(b + 1) * VOCAB]);
            process(&mut lp, &self.start, &self.running[b], &self.s, 1);
            acc.extend(lp.iter().map(|x| x + self.scores[b]));
        }
        let keep = 2 * nb;
        let picks: Vec<usize> = if self.s.sample {
            let m = acc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = acc.iter().map(|x| (x - m).exp()).collect();
            let mut d = draw(&p, keep, rng);
            // fewer possible continuations than wanted: the rest by score
            if d.len() < keep {
                let mut rest: Vec<usize> = (0..acc.len()).filter(|i| !d.contains(i)).collect();
                rest.sort_by(|&a, &b| acc[b].total_cmp(&acc[a]));
                d.extend(rest.into_iter().take(keep - d.len()));
            }
            d
        } else {
            let mut all: Vec<usize> = (0..acc.len()).collect();
            all.sort_by(|&a, &b| acc[b].total_cmp(&acc[a]).then(a.cmp(&b)));
            all.truncate(keep);
            all
        };
        // the candidates: (score, parent beam, token, ends)
        let max_len = self.s.max_new;
        let cands: Vec<(f32, usize, u32, bool)> = picks.iter().map(|&i| {
            let (b, t) = (i / VOCAB, (i % VOCAB) as u32);
            (acc[i], b, t, t == EOS || cur + 1 >= max_len)
        }).collect();
        // finished candidates among the first `nb`: kept by their score over their length
        for (k, c) in cands.iter().enumerate() {
            if k < nb && c.3 && c.0 > -1e8 {
                let mut seq = self.running[c.1].clone();
                seq.push(c.2);
                self.finished.push((seq, c.0 / (cur + 1) as f32));
            }
        }
        self.finished.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.finished.truncate(nb);
        // the best unfinished continue
        let mut live: Vec<(f32, usize, u32)> = cands.iter().map(|c| (if c.3 { c.0 - 1e9 } else { c.0 }, c.1, c.2)).collect();
        live.sort_by(|a, b| b.0.total_cmp(&a.0));
        live.truncate(nb);
        let old = std::mem::take(&mut self.running);
        self.running = live.iter().map(|c| {
            let mut s = old[c.1].clone();
            s.push(c.2);
            s
        }).collect();
        self.scores = live.iter().map(|c| c.0).collect();
        // stop when no running beam can beat the worst finished one (all slots full), or nothing can continue
        let len = (cur + 1) as f32;
        let best_running = self.scores[0] / len;
        let worst = if self.finished.len() == nb { self.finished.last().map_or(-1e9, |f| f.1) } else { -1e9 };
        let all_end = cands.iter().all(|c| c.3);
        self.done = best_running <= worst || best_running.is_nan() || all_end || cur + 1 >= max_len;
        Step { parents: live.iter().map(|c| c.1).collect(), tokens: live.iter().map(|c| c.2).collect() }
    }

    /// The best finished sequence (generated tokens, eos last), or the best running one cut where it is
    pub fn best(&self) -> Vec<u32> {
        self.finished.first().map(|f| f.0.clone()).unwrap_or_else(|| self.running[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_penalty_and_the_filters() {
        let mut s = vec![-1.0, 2.0, -3.0, 0.5];
        repetition(&mut s, &[0, 1, 1], 2.0);
        assert_eq!(s, vec![-2.0, 1.0, -3.0, 0.5]);
        let mut t = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        top_k(&mut t, 2, 1);
        assert_eq!(t.iter().filter(|x| x.is_finite()).count(), 2);
        // top-p: one token with nearly all the mass keeps it and the `keep` largest
        let mut u = vec![0.0, 10.0, -1.0, 1.0];
        top_p(&mut u, 0.9, 2);
        assert!(u[1].is_finite() && u[3].is_finite() && !u[0].is_finite() && !u[2].is_finite());
    }

    #[test]
    fn draws_are_distinct_and_weighted() {
        let mut r = Rng::new(3);
        let d = draw(&[0.0, 1.0, 1000.0, 1.0], 3, &mut r);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 2);
        assert!(!d.contains(&0));
    }

    #[test]
    fn a_beam_search_ends_on_eos() {
        let s = Settings { beams: 2, sample: false, max_new: 10, ..Default::default() };
        let start = [crate::tokens::BOS, crate::tokens::CLS_ARTICULATIONXL];
        let mut b = Beams::new(s, &start);
        let mut r = Rng::new(1);
        // logits that prefer eos wherever the grammar allows it, else coordinate 7
        for _ in 0..10 {
            let mut l = vec![0f32; 2 * VOCAB];
            for row in 0..2 {
                l[row * VOCAB + 7] = 5.0;
                l[row * VOCAB + EOS as usize] = 50.0;
            }
            b.step(&l, &mut r);
            if b.done {
                break;
            }
        }
        let best = b.best();
        assert_eq!(*best.last().unwrap(), EOS);
        // eos is allowed right after the class token: the best is the empty skeleton
        assert_eq!(best, vec![EOS]);
    }
}
