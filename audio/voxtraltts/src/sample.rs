//! The flow's starting noise: a seeded stream of standard normal values of the engine's own (splitmix64 and
//! Box-Muller) - a seed repeats here, not torch's draws. The semantic code is always its head's argmax (upstream's),
//! so the noise is a run's only randomness.

/// splitmix64: uniform doubles in [0, 1)
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x766f_7874_7261_6c00)
    }

    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// n standard normal values (Box-Muller, both of each pair used)
    pub fn normal(&mut self, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            v.push((r * a.cos()) as f32);
            v.push((r * a.sin()) as f32);
        }
        v.truncate(n);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_normal_values() {
        let a = Rng::new(7).normal(10_001);
        assert_eq!(a, Rng::new(7).normal(10_001));
        assert_ne!(a[..36], Rng::new(8).normal(36)[..]);
        let mean = a.iter().map(|x| *x as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "{mean} {var}");
    }
}
