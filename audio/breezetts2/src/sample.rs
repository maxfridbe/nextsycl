//! The draws, as upstream's streaming runtime makes them (models/cudagraph/sampling.py, depth_decoder_graph.py): the
//! guided logits (uncond + cfg (cond - uncond)), the backbone's repetition penalty over the frames' first codes so far,
//! the codec's reserved ids masked, then temperature, top-k, top-p, softmax and one draw; or the best (greedy). On the
//! host, from a seeded generator of its own (a seed repeats here, not torch's draws).

/// splitmix64: a seeded stream of uniform doubles in [0, 1)
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x6272_6565_7a65_0002)
    }

    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How codes are drawn
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    /// false: the most likely code (greedy)
    pub sample: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Draw {
    /// upstream's (runtime.py update_generation_config_for_breeze): sampled, 0.9, top 50, top-p 1.0 - the backbone's
    /// and the depth decoder's alike
    pub const DEFAULT: Draw = Draw { sample: true, temperature: 0.9, top_k: 50, top_p: 1.0 };
    pub const GREEDY: Draw = Draw { sample: false, temperature: 1.0, top_k: 0, top_p: 1.0 };
}

/// The guided logits: uncond + cfg * (cond - uncond) (cond alone without a second branch)
pub fn guided(cond: &[f32], uncond: Option<&[f32]>, cfg: f32) -> Vec<f32> {
    match uncond {
        Some(u) => cond.iter().zip(u).map(|(c, u)| u + cfg * (c - u)).collect(),
        None => cond.to_vec(),
    }
}

/// HF's repetition penalty over the ids seen (once each): a positive logit divided by `p`, a negative one multiplied
pub fn repetition_penalty(logits: &mut [f32], seen: &[i32], p: f32) {
    if p == 1.0 {
        return;
    }
    let mut done = std::collections::BTreeSet::new();
    for &id in seen {
        if id >= 0 && (id as usize) < logits.len() && done.insert(id) {
            let l = &mut logits[id as usize];
            *l = if *l > 0.0 { *l / p } else { *l * p };
        }
    }
}

/// The ids [lo, hi) masked out (the codec's reserved codes)
pub fn suppress(logits: &mut [f32], lo: usize, hi: usize) {
    for v in logits.iter_mut().take(hi).skip(lo) {
        *v = f32::NEG_INFINITY;
    }
}

/// The code: the best, or a draw from the temperature-scaled logits' top `k` (ties at the threshold kept) within
/// top-p's mass
pub fn pick(logits: &[f32], d: Draw, rng: &mut Rng) -> usize {
    let best = || logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
    if !d.sample {
        return best();
    }
    let t = d.temperature.max(1e-5);
    let mut v: Vec<f32> = logits.iter().map(|x| x / t).collect();
    if d.top_k > 0 && d.top_k < v.len() {
        let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if s.len() > d.top_k {
            s.select_nth_unstable_by(d.top_k - 1, |a, b| b.total_cmp(a));
            let thr = s[d.top_k - 1];
            v.iter_mut().for_each(|x| if *x < thr { *x = f32::NEG_INFINITY });
        }
    }
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return best();
    }
    let mut p: Vec<f64> = v.iter().map(|x| if x.is_finite() { ((x - mx) as f64).exp() } else { 0.0 }).collect();
    let total: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= total);
    if d.top_p < 1.0 {
        // HF's: the smallest tail whose mass is at most 1 - top_p goes (the best stays)
        let mut order: Vec<usize> = (0..p.len()).filter(|i| p[*i] > 0.0).collect();
        order.sort_by(|a, b| p[*a].total_cmp(&p[*b]));
        let mut cum = 0.0;
        for (n, i) in order.iter().enumerate() {
            cum += p[*i];
            if cum > 1.0 - d.top_p as f64 || n + 1 == order.len() {
                break;
            }
            p[*i] = 0.0;
        }
        let t: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= t);
    }
    let mut u = rng.uniform();
    for (i, w) in p.iter().enumerate() {
        if *w > 0.0 {
            if u < *w {
                return i;
            }
            u -= w;
        }
    }
    p.iter().rposition(|w| *w > 0.0).unwrap_or_else(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guidance_penalty_and_draws() {
        assert_eq!(guided(&[2.0, 0.0], Some(&[1.0, 1.0]), 4.0), vec![5.0, -3.0]);
        assert_eq!(guided(&[2.0, 0.0], None, 4.0), vec![2.0, 0.0]);
        let mut l = vec![2.0, -2.0, 1.0, 9.0];
        repetition_penalty(&mut l, &[0, 1, 0], 2.0);
        suppress(&mut l, 3, 4);
        assert_eq!(l, vec![1.0, -4.0, 1.0, f32::NEG_INFINITY]);
        let mut r = Rng::new(3);
        let v = vec![0.1, 5.0, 0.3, 0.2];
        for _ in 0..20 {
            assert_eq!(pick(&v, Draw { sample: true, temperature: 0.9, top_k: 1, top_p: 1.0 }, &mut r), 1);
        }
        assert_eq!(pick(&v, Draw::GREEDY, &mut r), 1);
        for _ in 0..200 {
            assert!(pick(&[3.0, 2.9, -1.0, -2.0], Draw { sample: true, temperature: 1.0, top_k: 2, top_p: 1.0 }, &mut r) < 2);
        }
    }
}
