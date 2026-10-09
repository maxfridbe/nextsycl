//! The autoregressive stage's sampling, as the reference does it (diffusers' `_sample_top_k` and the guided semantic
//! step): classifier-free guidance on the logits, the 50 best kept, softmax, one draw - on the host, from a seeded
//! generator of its own (so a seed repeats here, though not the reference's torch draws).

/// splitmix64: a seeded stream of uniform doubles in [0, 1)
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x6d6d_3300_5eed_0000)
    }

    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal draw (Box-Muller)
    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

pub const TOP_K: usize = 50;

/// The `k`-th largest of `v` (NaN as -1e9, infinities as -/+1e9: torch.nan_to_num first, as the reference)
fn kth(v: &[f32], k: usize) -> f32 {
    let mut s: Vec<f32> = v.iter().map(|x| clean(*x)).collect();
    let k = k.min(s.len());
    s.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    s[k - 1]
}

fn clean(x: f32) -> f32 {
    if x.is_nan() {
        -1e9
    } else {
        x.clamp(-1e9, 1e9)
    }
}

/// `_sample_top_k`: the `TOP_K` best of `logits` (ties at the threshold kept), softmax, one draw: the index
pub fn top_k(logits: &[f32], rng: &mut Rng) -> usize {
    let thr = kth(logits, TOP_K);
    let vals: Vec<f32> = logits.iter().map(|x| clean(*x)).collect();
    let mx = vals.iter().cloned().filter(|x| *x >= thr).fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f64> = vals.iter().map(|x| if *x >= thr { ((*x - mx) as f64).exp() } else { 0.0 }).collect();
    let total: f64 = p.iter().sum();
    let mut u = rng.uniform() * total;
    for (i, w) in p.iter().enumerate() {
        if *w > 0.0 {
            if u < *w {
                return i;
            }
            u -= w;
        }
    }
    p.iter().rposition(|w| *w > 0.0).unwrap_or(0)
}

/// Guidance: unconditional + (conditional - unconditional) * scale
pub fn guide(cond: &[f32], unc: &[f32], scale: f32) -> Vec<f32> {
    cond.iter().zip(unc).map(|(c, u)| u + (c - u) * scale).collect()
}

/// The semantic step: guided logits restricted to the conditional row's `TOP_K` best (`cond < its threshold` out),
/// then `top_k`
pub fn semantic(cond: &[f32], unc: &[f32], scale: f32, rng: &mut Rng) -> usize {
    let thr = kth(cond, TOP_K);
    let g: Vec<f32> = guide(cond, unc, scale).into_iter().zip(cond).map(|(g, c)| if clean(*c) < thr { f32::NEG_INFINITY } else { g }).collect();
    top_k(&g, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_draws_only_from_the_best() {
        let mut r = Rng::new(1);
        let mut v = vec![0f32; 200];
        v[7] = 10.0;
        for _ in 0..50 {
            assert_eq!(top_k(&v, &mut r), 7);
        }
        // 60 equal logits: every one is tied at the 50th, so all may be drawn - and nothing else
        let mut v = vec![-5f32; 200];
        for x in v.iter_mut().take(60) {
            *x = 1.0;
        }
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..2000 {
            let i = top_k(&v, &mut r);
            assert!(i < 60);
            seen.insert(i);
        }
        assert!(seen.len() > 50);
    }

    #[test]
    fn the_semantic_step_keeps_the_conditional_best() {
        let mut r = Rng::new(2);
        let mut c = vec![0f32; 100];
        let mut u = vec![0f32; 100];
        // the guided value is best at 99, but the conditional row ranks it below its top 50
        u[99] = -100.0;
        c[99] = -1.0;
        for (i, x) in c.iter_mut().enumerate().take(60) {
            *x = 1.0 + i as f32 * 0.01;
        }
        for _ in 0..200 {
            assert_ne!(semantic(&c, &u, 1.5, &mut r), 99);
        }
    }
}
