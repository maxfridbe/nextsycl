//! Drawing a row of ten codes from the backbone's logits, as the reference's `sample_tts` does: padding and every
//! special id but the stop id masked, the repetition penalty over each of the first codebooks' recent codes, the
//! temperature, top-k, top-p and min-p, one draw a codebook (an exponential race: argmax p / q, q ~ Exp(1), what
//! torch.multinomial does), and the delay pattern's leading triangle padded on a cold start.
//!
//! Origin: Zyphra's `zonos3.tts.sampler` and `zonos3.reference.synth.sample_row` (github.com/Zyphra/ZONOS3-Mini, MIT),
//! the defaults the generation registry's.

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub topk: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub repetition_codebooks: usize,
}

impl Default for Sampling {
    fn default() -> Sampling {
        Sampling { temperature: 0.75, topk: 80, top_p: 0.69, min_p: 0.0, repetition_penalty: 2.4, repetition_window: 96, repetition_codebooks: 6 }
    }
}

/// PCG32 (O'Neill): the request's seed to its draws
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
    seed: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng { state: 0, inc: (seed << 1) | 1, seed };
        r.next();
        r.state = r.state.wrapping_add(seed ^ 0x853c_49e6_748f_ea9b);
        r.next();
        r
    }

    /// the seed it started from
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xs.rotate_right(rot)
    }

    /// uniform in (0, 1]
    pub fn unit(&mut self) -> f64 {
        ((self.next() as u64) << 21 ^ (self.next() as u64)) as f64 / (1u64 << 53) as f64 + f64::EPSILON
    }

    pub fn exponential(&mut self) -> f64 {
        -self.unit().ln()
    }
}

/// One codebook's draw from its logits (`width` = the audio vocabulary): the id
#[allow(clippy::too_many_arguments)]
pub fn draw(logits: &[f32], s: &Sampling, codebook: usize, pad: usize, stop: usize, recent: &[u32], rng: &mut Rng) -> u32 {
    let v = logits.len();
    let mut x: Vec<f32> = logits.iter().enumerate()
        .map(|(i, l)| if i == pad || (i >= codebook && i != stop) { f32::NEG_INFINITY } else { *l }).collect();
    // the penalty once per distinct recent code
    let mut seen = vec![false; v];
    for c in recent {
        let c = *c as usize;
        if c < codebook && !seen[c] {
            seen[c] = true;
            x[c] = if x[c] > 0.0 { x[c] / s.repetition_penalty } else { x[c] * s.repetition_penalty };
        }
    }
    let t = s.temperature.max(1e-6);
    for l in x.iter_mut() {
        *l /= t;
    }
    if s.topk > 0 && s.topk < v {
        let mut sorted: Vec<f32> = x.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[s.topk - 1];
        for l in x.iter_mut() {
            if *l < kth {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return stop as u32;
    }
    let mut p: Vec<f64> = x.iter().map(|l| ((l - m) as f64).exp()).collect();
    let z: f64 = p.iter().sum();
    p.iter_mut().for_each(|q| *q /= z);
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|a, b| p[*b].total_cmp(&p[*a]).then(a.cmp(b)));
        let mut cum = 0.0;
        for &i in &order {
            let before = cum;
            cum += p[i];
            if before > s.top_p as f64 {
                p[i] = 0.0;
            }
        }
        let z: f64 = p.iter().sum::<f64>().max(1e-8);
        p.iter_mut().for_each(|q| *q /= z);
    }
    if s.min_p > 0.0 {
        let top = p.iter().copied().fold(0.0, f64::max);
        for q in p.iter_mut() {
            if *q < s.min_p as f64 * top {
                *q = 0.0;
            }
        }
        let z: f64 = p.iter().sum::<f64>().max(1e-8);
        p.iter_mut().for_each(|q| *q /= z);
    }
    // the exponential race (one draw per id of the row, as torch.multinomial's)
    let width = stop.max(pad) + 1;
    let mut best = (f64::NEG_INFINITY, stop);
    for (i, q) in p.iter().enumerate().take(width.min(v)) {
        let e = rng.exponential();
        if *q > 0.0 && q / e > best.0 {
            best = (q / e, i);
        }
    }
    best.1 as u32
}

/// A row of `ncb` codes from logits [ncb, vocab]: `history` the generated rows so far (the repetition penalty's
/// window), `total` how many; the cold start pads codebooks whose delayed stream has not begun (`stream_row`: the
/// row's place in the audio stream, the silence prefill counted)
#[allow(clippy::too_many_arguments)]
pub fn row(logits: &[f32], ncb: usize, s: &Sampling, codebook: usize, pad: u32, stop: u32, history: &[[u32; 10]], stream_row: usize, rng: &mut Rng)
           -> [u32; 10] {
    let v = logits.len() / ncb;
    let mut out = [pad; 10];
    let active = history.len().min(s.repetition_window);
    for (cb, o) in out.iter_mut().enumerate().take(ncb) {
        let recent: Vec<u32> = if cb < s.repetition_codebooks && s.repetition_penalty != 1.0 {
            history[history.len() - active..].iter().map(|r| r[cb]).collect()
        } else {
            Vec::new()
        };
        let id = draw(&logits[cb * v..(cb + 1) * v], s, codebook, pad as usize, stop as usize, &recent, rng);
        *o = if cb > stream_row { pad } else { id };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_and_specials_are_never_drawn() {
        let mut rng = Rng::new(7);
        let mut l = vec![0.0f32; 4098];
        l[4097] = 100.0;
        l[4096] = 1.0;
        l[5] = 2.0;
        let s = Sampling::default();
        for _ in 0..50 {
            let id = draw(&l, &s, 4096, 4097, 4096, &[], &mut rng);
            assert_ne!(id, 4097);
            assert!(id == 5 || id == 4096 || id < 4096);
        }
    }

    #[test]
    fn a_sharp_logit_wins_and_the_penalty_bites() {
        let mut rng = Rng::new(1);
        let mut l = vec![-10.0f32; 4098];
        l[17] = 10.0;
        let s = Sampling::default();
        assert_eq!(draw(&l, &s, 4096, 4097, 4096, &[], &mut rng), 17);
        // 10 / 2.4 is still far above the rest
        assert_eq!(draw(&l, &s, 4096, 4097, 4096, &[17], &mut rng), 17);
        let mut l2 = vec![-10.0f32; 4098];
        l2[17] = 1.0;
        l2[18] = 0.9;
        let s1 = Sampling { temperature: 0.01, ..Sampling::default() };
        assert_eq!(draw(&l2, &s1, 4096, 4097, 4096, &[17], &mut rng), 18);
    }

    #[test]
    fn the_cold_start_pads_the_triangle() {
        let mut rng = Rng::new(3);
        let l = vec![0.0f32; 10 * 4098];
        let r = row(&l, 10, &Sampling::default(), 4096, 4097, 4096, &[], 3, &mut rng);
        assert!(r[..4].iter().all(|v| *v < 4097));
        assert!(r[4..].iter().all(|v| *v == 4097));
    }

    #[test]
    fn the_rng_is_seeded() {
        let (mut a, mut b, mut c) = (Rng::new(5), Rng::new(5), Rng::new(6));
        let va: Vec<u32> = (0..4).map(|_| a.next()).collect();
        assert_eq!(va, (0..4).map(|_| b.next()).collect::<Vec<_>>());
        assert_ne!(va, (0..4).map(|_| c.next()).collect::<Vec<_>>());
        let u = a.unit();
        assert!(u > 0.0 && u <= 1.0 + f64::EPSILON);
    }
}
