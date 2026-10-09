//! Noise schedules for flow-matching models: `steps + 1` levels from 1 (pure noise) down to 0 (the image).

/// How the noise levels are spaced
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// even steps, then shifted toward the noisy end by `shift` (the flow models' default; a resolution-dependent
    /// shift is the "dynamic" variant: the engine picks `shift` from the image size)
    Shift,
    /// even steps
    Simple,
    /// Karras et al.: dense near the clean end (rho 7)
    Karras,
    /// the Beta(0.6, 0.6) quantiles: dense at both ends
    Beta,
}

impl Schedule {
    pub const ALL: [Schedule; 4] = [Schedule::Shift, Schedule::Simple, Schedule::Karras, Schedule::Beta];

    pub fn name(self) -> &'static str {
        match self {
            Schedule::Shift => "shift",
            Schedule::Simple => "simple",
            Schedule::Karras => "karras",
            Schedule::Beta => "beta",
        }
    }

    pub fn parse(s: &str) -> Option<Schedule> {
        Schedule::ALL.into_iter().find(|x| x.name() == s)
    }
}

/// The schedule's `steps + 1` noise levels, from 1 down to 0. `shift` applies to `Shift` (1 = no shift).
pub fn sigmas(schedule: Schedule, steps: u32, shift: f32) -> Vec<f32> {
    let n = steps.max(1) as usize;
    let even = |i: usize| 1.0 - i as f64 / n as f64;
    let shifted = |s: f64, k: f64| k * s / (1.0 + (k - 1.0) * s);
    let mut v: Vec<f64> = match schedule {
        Schedule::Simple => (0..=n).map(even).collect(),
        Schedule::Shift => (0..=n).map(|i| shifted(even(i), shift.max(1e-3) as f64)).collect(),
        Schedule::Karras => {
            // between sigma_max = 1 and a small sigma_min, rho 7, then 0
            let (lo, hi, rho) = (0.002f64, 1.0f64, 7.0f64);
            let (a, b) = (hi.powf(1.0 / rho), lo.powf(1.0 / rho));
            let mut k: Vec<f64> = (0..n).map(|i| (a + i as f64 / (n - 1).max(1) as f64 * (b - a)).powf(rho)).collect();
            k.push(0.0);
            k
        }
        Schedule::Beta => (0..=n).map(|i| 1.0 - beta_quantile(i as f64 / n as f64, 0.6, 0.6)).collect(),
    };
    v[0] = 1.0;
    v[n] = 0.0;
    v.into_iter().map(|x| x as f32).collect()
}

/// The `p` quantile of Beta(a, b), by bisection on its CDF (a numeric integral: precise enough for a schedule)
fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // the CDF by the midpoint rule on t = x^(1/a) substitutions would be overkill: a fine grid of the density,
    // integrated once, then interpolated
    const N: usize = 4096;
    let dens = |x: f64| x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0);
    let mut cdf = vec![0.0f64; N + 1];
    for i in 0..N {
        let x = (i as f64 + 0.5) / N as f64;
        cdf[i + 1] = cdf[i] + dens(x) / N as f64;
    }
    let total = cdf[N];
    let target = p * total;
    let j = cdf.partition_point(|c| *c < target).clamp(1, N);
    let (c0, c1) = (cdf[j - 1], cdf[j]);
    let f = if c1 > c0 { (target - c0) / (c1 - c0) } else { 0.0 };
    ((j - 1) as f64 + f) / N as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schedule_runs_from_noise_to_image() {
        for s in Schedule::ALL {
            let v = sigmas(s, 8, 3.0);
            assert_eq!(v.len(), 9, "{s:?}");
            assert_eq!((v[0], v[8]), (1.0, 0.0), "{s:?}");
            assert!(v.windows(2).all(|w| w[0] >= w[1]), "{s:?} not decreasing: {v:?}");
            assert_eq!(Schedule::parse(s.name()), Some(s));
        }
    }

    #[test]
    fn shift_moves_steps_toward_noise() {
        let plain = sigmas(Schedule::Simple, 4, 1.0);
        let shifted = sigmas(Schedule::Shift, 4, 3.0);
        assert!(shifted[2] > plain[2]);
        assert_eq!(sigmas(Schedule::Shift, 4, 1.0), plain);
    }
}
