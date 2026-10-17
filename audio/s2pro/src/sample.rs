//! The draws, as fish-speech's decode_one_token_ar makes them (fish_speech/models/text2semantic/inference.py
//! logits_to_probs): top-k and top-p cut on the plain softmax of the logits sorted (the best always kept), then the
//! temperature, a softmax and one draw; the frame's first code repetition-aware (RAS: a draw already among the last 10
//! semantic ids is drawn again hotter, temperature 1.0 and top-p 0.9). Or the best (greedy). On the host, from a seeded
//! generator of its own (a seed repeats here, not torch's draws).

/// splitmix64: a seeded stream of uniform doubles in [0, 1)
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x7332_7072_6f00_0001)
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draw {
    /// false: the most likely id (greedy)
    pub sample: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Draw {
    /// upstream's defaults (fish_speech/models/text2semantic/sampling.py: the Fish Audio SDK's)
    pub const DEFAULT: Draw = Draw { sample: true, temperature: 0.7, top_p: 0.7, top_k: 30 };
    /// RAS's second draw (RAS_HIGH_TEMP, RAS_HIGH_TOP_P)
    pub const HOT: Draw = Draw { sample: true, temperature: 1.0, top_p: 0.9, top_k: 30 };
    pub const GREEDY: Draw = Draw { sample: false, temperature: 1.0, top_p: 1.0, top_k: 0 };
}

/// RAS's window: the last 10 first-code ids drawn
pub const WINDOW: usize = 10;

fn best(l: &[f32]) -> usize {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0
}

/// The id drawn from logits (-inf: never)
pub fn pick(l: &[f32], d: Draw, rng: &mut Rng) -> usize {
    if !d.sample {
        return best(l);
    }
    // sorted by logit, best first; the plain softmax's running sum decides top-p (the id that crosses it goes too)
    let mut order: Vec<usize> = (0..l.len()).filter(|i| l[*i].is_finite()).collect();
    if order.is_empty() {
        return best(l);
    }
    order.sort_by(|a, b| l[*b].total_cmp(&l[*a]));
    let mx = l[order[0]];
    let total: f64 = order.iter().map(|i| ((l[*i] - mx) as f64).exp()).sum();
    let mut cum = 0.0;
    let mut kept = Vec::new();
    for (n, i) in order.iter().enumerate() {
        cum += ((l[*i] - mx) as f64).exp() / total;
        if n > 0 && (n >= d.top_k.max(1) || cum > d.top_p as f64) {
            continue;
        }
        kept.push(*i);
    }
    // the kept ids at the temperature
    let t = d.temperature.max(1e-5) as f64;
    let p: Vec<f64> = kept.iter().map(|i| ((l[*i] - mx) as f64 / t).exp()).collect();
    let s: f64 = p.iter().sum();
    let mut u = rng.uniform() * s;
    for (i, w) in kept.iter().zip(&p) {
        if u < *w {
            return *i;
        }
        u -= w;
    }
    *kept.last().unwrap_or(&order[0])
}

/// The frame's first code: drawn, and drawn again hotter when it repeats one of the last ids (`window`) and is a
/// semantic id (`semantic(id)`; the end is never redrawn)
pub fn pick_ras(l: &[f32], d: Draw, window: &[usize], semantic: impl Fn(usize) -> bool, rng: &mut Rng) -> usize {
    let first = pick(l, d, rng);
    if !d.sample {
        return first;
    }
    // upstream draws both every time (one random number each), then takes the hot one on a repeat
    let hot = pick(l, Draw { temperature: Draw::HOT.temperature, top_p: Draw::HOT.top_p, ..d }, rng);
    if semantic(first) && window.contains(&first) { hot } else { first }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_p_is_cut_before_the_temperature() {
        let mut r = Rng::new(3);
        // softmax at 1: [0.665, 0.245, 0.090]; top-p 0.7: the second crosses 0.7 -> only the best is kept
        let l = [2.0f32, 1.0, 0.0];
        for _ in 0..50 {
            assert_eq!(pick(&l, Draw { sample: true, temperature: 5.0, top_p: 0.7, top_k: 30 }, &mut r), 0);
        }
        // top-p 0.95 keeps the first two (0.91 < 0.95, the third crosses)
        let mut seen = [0; 3];
        for _ in 0..400 {
            seen[pick(&l, Draw { sample: true, temperature: 1.0, top_p: 0.95, top_k: 30 }, &mut r)] += 1;
        }
        assert!(seen[0] > 0 && seen[1] > 0 && seen[2] == 0, "{seen:?}");
        // top-k 1, -inf never, greedy
        assert_eq!(pick(&[0.0, f32::NEG_INFINITY, 3.0], Draw { sample: true, temperature: 1.0, top_p: 1.0, top_k: 1 }, &mut r), 2);
        assert_eq!(pick(&[0.0, 9.0, 3.0], Draw::GREEDY, &mut r), 1);
    }

    #[test]
    fn a_repeat_is_drawn_again_hotter() {
        let mut r = Rng::new(7);
        let l = [5.0f32, 4.9, 4.0];
        // softmax [0.44, 0.40, 0.16]: top-p 0.7 keeps id 0 alone, a repeat of it is drawn hot (top-p 0.9: ids 0 and 1)
        let mut seen = [0; 3];
        for _ in 0..200 {
            seen[pick_ras(&l, Draw::DEFAULT, &[0, 1], |_| true, &mut r)] += 1;
        }
        assert!(seen[0] > 0 && seen[1] > 0 && seen[2] == 0, "{seen:?}");
        // not semantic: never redrawn; greedy: the best
        assert_eq!(pick_ras(&[1.0, 0.0], Draw::DEFAULT, &[0], |_| false, &mut r), 0);
        assert_eq!(pick_ras(&l, Draw::GREEDY, &[0], |_| true, &mut r), 0);
    }
}
