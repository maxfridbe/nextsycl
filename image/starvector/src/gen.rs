//! How StarVector's generate_im2svg chooses its tokens, as transformers 4.49 does (generation/utils.py `_sample` and
//! `_beam_search`, beam_search.py's BeamSearchScorer / BeamHypotheses, logits_process.py's warpers), on the host over
//! the logits the model gives. Pure: tested without a GPU.
//!
//! The model's own setting: 2 beams, sampled (`do_sample`), top_p 0.9 keeping at least 2 tokens, temperature 1, early
//! stopping, length penalty 1. A step: each beam's log-softmax, the warpers, its beam score added; 2 x beams (beam,
//! token) pairs drawn without replacement from the softmax over all of them, sorted by score; the eos ones among the
//! first `beams` become finished hypotheses (score / length), the rest the next beams. It stops when `beams`
//! hypotheses are finished, when the first beam ends with the stop sequence ("</svg>": StarVector's stopping
//! criterion looks at row 0 only), or at the length limit; the best hypothesis (finished or still running) wins.
//! One beam: plain sampling (or argmax) over the warped logits, stopping at eos, the stop sequence or the limit.

/// What a step needs: the logits after feeding each beam's last token (one row a beam, `vocab` each), and the beams'
/// caches made to follow a reordering
pub trait Lm {
    fn vocab(&self) -> usize;
    /// the logits of the prefix (the picture and the prompt): the first step's, for every beam
    fn first(&mut self) -> nextsycl_image::Result<Vec<f32>>;
    /// beams[i] takes over the cache of beams[from[i]] (from the step before), then each beam is fed tokens[i] at the
    /// next position: the logits [beams, vocab]
    fn step(&mut self, from: &[usize], tokens: &[u32]) -> nextsycl_image::Result<Vec<f32>>;
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub beams: usize,
    pub sample: bool,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    /// new tokens at most (the model's max_length less the prefix)
    pub max_new: usize,
    pub eos: u32,
    pub stop: Vec<u32>,
    pub seed: u64,
}

/// A small, fast generator (SplitMix64): the draws are this engine's own, not torch's
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// log_softmax of one row
pub fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = row.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let ls = s.ln() as f32 + m;
    row.iter().map(|&v| v - ls).collect()
}

/// RepetitionPenaltyLogitsProcessor: each token already in `seen` (negative scores times p, positive ones divided)
pub fn repetition(row: &mut [f32], seen: &[u32], p: f32) {
    if p == 1.0 {
        return;
    }
    let mut done = std::collections::HashSet::new();
    for &t in seen {
        if done.insert(t) {
            let v = &mut row[t as usize];
            *v = if *v < 0.0 { *v * p } else { *v / p };
        }
    }
}

/// TopPLogitsWarper (`filter_value` -inf): the tokens whose ascending cumulative probability is at most 1 - top_p
/// go, the `keep` likeliest always stay. In place; the kept tokens' indices come back (likeliest first)
pub fn top_p(row: &mut [f32], p: f32, keep: usize) -> Vec<usize> {
    if p >= 1.0 {
        return (0..row.len()).filter(|&i| row[i].is_finite()).collect();
    }
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f64> = row.iter().map(|&v| ((v - m) as f64).exp()).collect();
    let total: f64 = probs.iter().sum();
    // the likeliest first, enough of them to reach top_p (a peaked distribution: a handful)
    let mut idx: Vec<usize> = (0..row.len()).collect();
    let mut k = 64.min(idx.len());
    let kept = loop {
        if k < idx.len() {
            idx.select_nth_unstable_by(k, |a, b| probs[*b].total_cmp(&probs[*a]));
        }
        let mut top: Vec<usize> = idx[..k].to_vec();
        top.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]).then(a.cmp(b)));
        // kept while the mass above it (descending) is under top_p: ascending cumsum = 1 - that mass > 1 - top_p
        let mut above = 0.0;
        let mut n = 0;
        for &i in &top {
            if above / total >= p as f64 && n >= keep {
                break;
            }
            above += probs[i];
            n += 1;
        }
        if n < top.len() || k == idx.len() {
            top.truncate(n.max(keep.min(top.len())));
            break top;
        }
        k = (k * 4).min(idx.len());
    };
    let mut on = vec![false; row.len()];
    for &i in &kept {
        on[i] = true;
    }
    for (i, v) in row.iter_mut().enumerate() {
        if !on[i] {
            *v = f32::NEG_INFINITY;
        }
    }
    kept
}

/// The warpers in transformers' order: temperature, then top_p
fn warp(row: &mut [f32], s: &Settings, keep: usize) -> Vec<usize> {
    if s.temperature != 1.0 && s.temperature > 0.0 {
        for v in row.iter_mut() {
            *v /= s.temperature;
        }
    }
    top_p(row, s.top_p, keep)
}

fn ends_with(seq: &[u32], stop: &[u32]) -> bool {
    !stop.is_empty() && seq.len() >= stop.len() && seq[seq.len() - stop.len()..] == *stop
}

/// What decoding gave: the tokens after the prompt (no eos), whether the stop sequence ended them
pub struct Decoded {
    pub tokens: Vec<u32>,
    pub stopped: bool,
}

/// One beam: argmax, or a draw from the warped softmax (torch.multinomial, one sample)
fn single(lm: &mut dyn Lm, s: &Settings, rng: &mut Rng, on_token: &mut dyn FnMut(&[u32]) -> nextsycl_image::Result<()>)
          -> nextsycl_image::Result<Decoded> {
    let mut seq: Vec<u32> = Vec::new();
    let mut logits = lm.first()?;
    loop {
        let mut row = logits;
        repetition(&mut row, &seq, s.repetition_penalty);
        let t = if s.sample {
            let kept = warp(&mut row, s, 1);
            let m = kept.iter().map(|&i| row[i]).fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f64> = kept.iter().map(|&i| ((row[i] - m) as f64).exp()).collect();
            let mut u = rng.uniform() * w.iter().sum::<f64>();
            let mut pick = kept[kept.len() - 1];
            for (j, &i) in kept.iter().enumerate() {
                if u < w[j] {
                    pick = i;
                    break;
                }
                u -= w[j];
            }
            pick as u32
        } else {
            let mut best = 0;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best as u32
        };
        if t == s.eos {
            return Ok(Decoded { tokens: seq, stopped: false });
        }
        seq.push(t);
        on_token(&seq)?;
        if ends_with(&seq, &s.stop) {
            return Ok(Decoded { tokens: seq, stopped: true });
        }
        if seq.len() >= s.max_new {
            return Ok(Decoded { tokens: seq, stopped: false });
        }
        logits = lm.step(&[0], &[t])?;
    }
}

/// BeamHypotheses (n-best of finished sequences)
struct Hyps {
    beams: Vec<(f64, Vec<u32>)>,
    worst: f64,
    n: usize,
}

impl Hyps {
    fn add(&mut self, hyp: Vec<u32>, sum_logprobs: f64, generated_len: usize) {
        let score = sum_logprobs / generated_len as f64;
        if self.beams.len() < self.n || score > self.worst {
            self.beams.push((score, hyp));
            if self.beams.len() > self.n {
                let mut sorted: Vec<(f64, usize)> = self.beams.iter().enumerate().map(|(i, b)| (b.0, i)).collect();
                sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                self.beams.remove(sorted[0].1);
                self.worst = sorted[1].0;
            } else {
                self.worst = self.worst.min(score);
            }
        }
    }
}

/// Beam search, sampled or not (transformers' `_beam_search` with `BeamSearchScorer`, early stopping on)
fn beams(lm: &mut dyn Lm, s: &Settings, rng: &mut Rng, on_token: &mut dyn FnMut(&[u32]) -> nextsycl_image::Result<()>)
         -> nextsycl_image::Result<Decoded> {
    let nb = s.beams;
    let v = lm.vocab();
    let mut seqs: Vec<Vec<u32>> = vec![Vec::new(); nb];
    let mut scores: Vec<f64> = (0..nb).map(|i| if i == 0 { 0.0 } else { -1e9 }).collect();
    let mut hyps = Hyps { beams: Vec::new(), worst: 1e9, n: nb };
    let first = lm.first()?;
    // every beam starts from the same prefix: the same logits
    let mut logits: Vec<f32> = (0..nb).flat_map(|_| first.iter().cloned()).collect();
    let keep = 2;
    let n_pick = 2 * nb;
    let mut done = false;
    let mut stopped = false;
    let mut cur = 0;
    loop {
        // (score, beam, token) of every candidate the warpers kept
        let mut cand: Vec<(f64, usize, u32)> = Vec::new();
        for b in 0..nb {
            // (the -1e9 beams of the first step take part with a probability of exactly 0)
            let mut row = log_softmax(&logits[b * v..(b + 1) * v]);
            repetition(&mut row, &seqs[b], s.repetition_penalty);
            let kept: Vec<usize> = if s.sample { warp(&mut row, s, keep) } else { (0..v).collect() };
            for i in kept {
                cand.push((row[i] as f64 + scores[b], b, i as u32));
            }
        }
        let mut picked: Vec<(f64, usize, u32)> = if s.sample {
            // torch.multinomial without replacement over softmax(all the beams' scores)
            let m = cand.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
            let mut w: Vec<f64> = cand.iter().map(|c| (c.0 - m).exp()).collect();
            let mut used = vec![false; cand.len()];
            let mut out = Vec::new();
            for _ in 0..n_pick.min(cand.len()) {
                let total: f64 = w.iter().sum();
                let j = if total > 0.0 {
                    let mut u = rng.uniform() * total;
                    let mut j = w.iter().rposition(|&x| x > 0.0).unwrap_or(0);
                    for (k, &x) in w.iter().enumerate() {
                        if x <= 0.0 {
                            continue;
                        }
                        if u < x {
                            j = k;
                            break;
                        }
                        u -= x;
                    }
                    j
                } else {
                    // fewer candidates of non-zero probability than draws: torch's CUDA multinomial (without
                    // replacement) still returns them all, the zero ones last - here the best scored of those
                    (0..cand.len()).filter(|&k| !used[k]).max_by(|a, b| cand[*a].0.total_cmp(&cand[*b].0)).expect("a candidate left")
                };
                out.push(cand[j]);
                used[j] = true;
                w[j] = 0.0;
            }
            out
        } else {
            let k = n_pick.min(cand.len());
            cand.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            cand.truncate(n_pick);
            cand
        };
        picked.sort_by(|a, b| b.0.total_cmp(&a.0));
        // BeamSearchScorer.process
        let mut next: Vec<(f64, usize, u32)> = Vec::new();
        for (rank, c) in picked.iter().enumerate() {
            if c.2 == s.eos {
                if rank >= nb {
                    continue;
                }
                hyps.add(seqs[c.1].clone(), c.0, cur + 1);
            } else {
                next.push(*c);
            }
            if next.len() == nb {
                break;
            }
        }
        if next.len() < nb {
            return Err(nextsycl_image::Error(format!("beam search: {} candidates for {nb} beams", next.len())));
        }
        done = done || hyps.beams.len() >= nb;
        let from: Vec<usize> = next.iter().map(|c| c.1).collect();
        seqs = next.iter().map(|c| {
            let mut q = seqs[c.1].clone();
            q.push(c.2);
            q
        }).collect();
        scores = next.iter().map(|c| c.0).collect();
        cur += 1;
        on_token(&seqs[0])?;
        if ends_with(&seqs[0], &s.stop) {
            stopped = true;
        }
        if done || stopped || cur >= s.max_new {
            break;
        }
        let toks: Vec<u32> = next.iter().map(|c| c.2).collect();
        logits = lm.step(&from, &toks)?;
    }
    // finalize: the running beams join the finished ones unless already done
    if !done {
        for b in 0..nb {
            let l = seqs[b].len();
            hyps.add(seqs[b].clone(), scores[b], l);
        }
    }
    let mut best = 0;
    for (i, h) in hyps.beams.iter().enumerate() {
        if h.0 >= hyps.beams[best].0 {
            best = i;
        }
    }
    let tokens = hyps.beams.swap_remove(best).1;
    let stopped = ends_with(&tokens, &s.stop);
    Ok(Decoded { tokens, stopped })
}

/// Decode as generate_im2svg does with these settings; `on_token` sees the leading beam after each step
pub fn decode(lm: &mut dyn Lm, s: &Settings, on_token: &mut dyn FnMut(&[u32]) -> nextsycl_image::Result<()>) -> nextsycl_image::Result<Decoded> {
    let mut rng = Rng::new(s.seed);
    if s.beams <= 1 {
        single(lm, s, &mut rng, on_token)
    } else {
        beams(lm, s, &mut rng, on_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy model: token t's logits favour t + 1 (eos after 9), eos unlikely before
    struct Toy {
        v: usize,
        pos: Vec<usize>,
    }

    impl Lm for Toy {
        fn vocab(&self) -> usize {
            self.v
        }
        fn first(&mut self) -> nextsycl_image::Result<Vec<f32>> {
            self.pos = vec![0; 4];
            let mut r = vec![0f32; self.v];
            r[5] = 10.0;
            r[0] = -20.0; // eos: unlikely until the toy's end
            Ok(r)
        }
        fn step(&mut self, from: &[usize], tokens: &[u32]) -> nextsycl_image::Result<Vec<f32>> {
            let pos: Vec<usize> = from.iter().map(|&f| self.pos[f] + 1).collect();
            self.pos[..pos.len()].copy_from_slice(&pos);
            let mut out = vec![0f32; self.v * tokens.len()];
            for (b, &t) in tokens.iter().enumerate() {
                let next = if t >= 9 { 0 } else { t as usize + 1 };
                if next != 0 {
                    out[b * self.v] = -20.0;
                }
                out[b * self.v + next] = 10.0;
            }
            Ok(out)
        }
    }

    fn settings(beams: usize, sample: bool) -> Settings {
        Settings { beams, sample, top_p: 0.9, temperature: 1.0, repetition_penalty: 1.0, max_new: 50, eos: 0, stop: vec![8, 9], seed: 1 }
    }

    #[test]
    fn greedy_follows_the_logits_to_the_stop_sequence() {
        let d = decode(&mut Toy { v: 12, pos: vec![] }, &settings(1, false), &mut |_| Ok(())).unwrap();
        assert_eq!((d.tokens, d.stopped), (vec![5, 6, 7, 8, 9], true));
    }

    #[test]
    fn beams_sampled_from_a_peaked_model_find_the_same_path() {
        for seed in 0..5 {
            let mut s = settings(2, true);
            s.seed = seed;
            let d = decode(&mut Toy { v: 12, pos: vec![] }, &s, &mut |_| Ok(())).unwrap();
            assert_eq!(d.tokens, vec![5, 6, 7, 8, 9], "seed {seed}");
            assert!(d.stopped);
        }
        let d = decode(&mut Toy { v: 12, pos: vec![] }, &settings(2, false), &mut |_| Ok(())).unwrap();
        assert_eq!(d.tokens, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn the_length_limit_stops_it() {
        let mut s = settings(2, true);
        s.stop = vec![];
        s.max_new = 3;
        let d = decode(&mut Toy { v: 12, pos: vec![] }, &s, &mut |_| Ok(())).unwrap();
        assert_eq!(d.tokens, vec![5, 6, 7]);
        assert!(!d.stopped);
    }

    #[test]
    fn top_p_keeps_the_head_and_at_least_two() {
        let mut row = vec![0f32; 100];
        row[3] = 10.0;
        row[7] = 1.0;
        let kept = top_p(&mut row, 0.9, 2);
        assert_eq!(kept, vec![3, 7], "the likeliest holds over 0.9 alone; two are kept");
        assert!(row[50].is_infinite() && row[3] == 10.0);
        // a flat row: 90 of 100 tokens are needed to reach 0.9
        let mut flat = vec![0f32; 100];
        assert_eq!(top_p(&mut flat, 0.9, 1).len(), 90);
    }

    #[test]
    fn hypotheses_keep_the_best_n() {
        let mut h = Hyps { beams: vec![], worst: 1e9, n: 2 };
        h.add(vec![1], -3.0, 1);
        h.add(vec![2], -1.0, 1);
        h.add(vec![3], -2.0, 1);
        let mut s: Vec<f64> = h.beams.iter().map(|b| b.0).collect();
        s.sort_by(f64::total_cmp);
        assert_eq!(s, vec![-2.0, -1.0]);
        assert_eq!(h.worst, -2.0);
    }
}
