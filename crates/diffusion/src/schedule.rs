//! Noise schedules for flow-matching models: levels from 1 (pure noise) down to 0 (the image).
//!
//! `Shift` is the flow models' own (even steps, shifted). The rest are ComfyUI's schedulers (comfy/samplers.py
//! `SCHEDULER_HANDLERS`) computed for a flow model's sampling: ComfyUI's `ModelSamplingDiscreteFlow` with the given
//! shift, 1000 timesteps, multiplier 1000 - its table of sigmas `time_snr_shift(shift, k / 1000)`, k = 1..=1000,
//! kept in float32 as ComfyUI keeps it, so that the schedulers that pick from the table pick the same entries.

/// How the noise levels are spaced
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// even steps, then shifted toward the noisy end by `shift` (the flow models' default; a resolution-dependent
    /// shift is the "dynamic" variant: the engine picks `shift` from the image size)
    Shift,
    /// ComfyUI's "simple": evenly spaced entries of the model's sigma table
    Simple,
    /// ComfyUI's "sgm_uniform": even timesteps from sigma_max to sigma_min, the last dropped, then 0
    SgmUniform,
    /// Karras et al. (rho 7) between the model's sigma_min and sigma_max: dense near the clean end
    Karras,
    /// even in log sigma between sigma_min and sigma_max
    Exponential,
    /// ComfyUI's "ddim_uniform": every (1000 / steps)-th table entry (can give a step more than asked for)
    DdimUniform,
    /// ComfyUI's "beta": the table entries at the Beta(0.6, 0.6) quantiles, dense at both ends (repeats dropped, so
    /// a step or more fewer at high step counts)
    Beta,
    /// ComfyUI's "normal": even timesteps from sigma_max to sigma_min, then 0
    Normal,
    /// Mochi's linear then quadratic schedule (ComfyUI's "linear_quadratic")
    LinearQuadratic,
    /// the KL-optimal schedule (even in arctan sigma)
    KlOptimal,
}

impl Schedule {
    pub const ALL: [Schedule; 10] = [
        Schedule::Shift, Schedule::Simple, Schedule::SgmUniform, Schedule::Karras, Schedule::Exponential,
        Schedule::DdimUniform, Schedule::Beta, Schedule::Normal, Schedule::LinearQuadratic, Schedule::KlOptimal,
    ];

    /// its name (ComfyUI's, but for `shift`)
    pub fn name(self) -> &'static str {
        match self {
            Schedule::Shift => "shift",
            Schedule::Simple => "simple",
            Schedule::SgmUniform => "sgm_uniform",
            Schedule::Karras => "karras",
            Schedule::Exponential => "exponential",
            Schedule::DdimUniform => "ddim_uniform",
            Schedule::Beta => "beta",
            Schedule::Normal => "normal",
            Schedule::LinearQuadratic => "linear_quadratic",
            Schedule::KlOptimal => "kl_optimal",
        }
    }

    /// the name, '-' taken for '_'
    pub fn parse(s: &str) -> Option<Schedule> {
        let s = s.replace('-', "_");
        Schedule::ALL.into_iter().find(|x| x.name() == s)
    }
}

/// The schedule's noise levels, from (about) 1 down to 0: `steps + 1` of them but for `DdimUniform` and `Beta`, which
/// can give ComfyUI's one more or fewer. `shift` is `Shift`'s shift and the flow model sampling's for the others
/// (1 = no shift; 0 or less is taken as 1 for those).
pub fn sigmas(schedule: Schedule, steps: u32, shift: f32) -> Vec<f32> {
    let n = steps.max(1) as usize;
    if schedule == Schedule::Shift {
        let even = |i: usize| 1.0 - i as f64 / n as f64;
        let shifted = |s: f64, k: f64| k * s / (1.0 + (k - 1.0) * s);
        let mut v: Vec<f64> = (0..=n).map(|i| shifted(even(i), shift.max(1e-3) as f64)).collect();
        v[0] = 1.0;
        v[n] = 0.0;
        return v.into_iter().map(|x| x as f32).collect();
    }
    let flow = Flow::new(if shift > 0.0 { shift } else { 1.0 });
    match schedule {
        Schedule::Shift => unreachable!(),
        Schedule::Simple => flow.simple(n),
        Schedule::SgmUniform => flow.normal(n, true),
        Schedule::Karras => karras(n, flow.sigma_min() as f64, flow.sigma_max() as f64, 7.0),
        Schedule::Exponential => exponential(n, flow.sigma_min() as f64, flow.sigma_max() as f64),
        Schedule::DdimUniform => flow.ddim_uniform(n),
        Schedule::Beta => flow.beta(n, 0.6, 0.6),
        Schedule::Normal => flow.normal(n, false),
        Schedule::LinearQuadratic => flow.linear_quadratic(n, 0.025),
        Schedule::KlOptimal => kl_optimal(n, flow.sigma_min() as f64, flow.sigma_max() as f64),
    }
}

/// ComfyUI's `ModelSamplingDiscreteFlow`: the shift and the 1000 sigmas it keeps (float32)
struct Flow {
    shift: f32,
    table: Vec<f32>,
}

impl Flow {
    fn new(shift: f32) -> Flow {
        let mut f = Flow { shift, table: Vec::new() };
        f.table = (1..=1000).map(|k| f.sigma((k as f32 / 1000.0) * 1000.0)).collect();
        f
    }

    /// `time_snr_shift(shift, timestep / 1000)` in float32, as ComfyUI computes it on a tensor
    fn sigma(&self, timestep: f32) -> f32 {
        let t = timestep / 1000.0;
        if self.shift == 1.0 {
            return t;
        }
        self.shift * t / (1.0 + (self.shift as f64 - 1.0) as f32 * t)
    }

    fn sigma_min(&self) -> f32 {
        self.table[0]
    }

    fn sigma_max(&self) -> f32 {
        self.table[999]
    }

    fn simple(&self, steps: usize) -> Vec<f32> {
        let ss = 1000.0 / steps as f64;
        let mut v: Vec<f32> = (0..steps).map(|x| self.table[999 - (x as f64 * ss) as usize]).collect();
        v.push(0.0);
        v
    }

    fn normal(&self, steps: usize, sgm: bool) -> Vec<f32> {
        let (start, end) = (self.sigma_max() * 1000.0, self.sigma_min() * 1000.0);
        let mut append_zero = true;
        let ts = if sgm {
            let mut t = linspace32(start, end, steps + 1);
            t.pop();
            t
        } else {
            let mut steps = steps;
            if self.sigma(end).abs() <= 1e-5 {
                steps += 1;
                append_zero = false;
            }
            linspace32(start, end, steps)
        };
        let mut v: Vec<f32> = ts.into_iter().map(|t| self.sigma(t)).collect();
        if append_zero {
            v.push(0.0);
        }
        v
    }

    fn ddim_uniform(&self, steps: usize) -> Vec<f32> {
        let mut steps = steps;
        let mut v = if self.table[1].abs() <= 1e-5 {
            steps += 1;
            vec![]
        } else {
            vec![0.0]
        };
        let ss = (1000 / steps).max(1);
        let mut x = 1;
        while x < 1000 {
            v.push(self.table[x]);
            x += ss;
        }
        v.reverse();
        v
    }

    fn beta(&self, steps: usize, a: f64, b: f64) -> Vec<f32> {
        let mut v = Vec::new();
        let mut last: Option<usize> = None;
        for i in 0..steps {
            let p = 1.0 - i as f64 / steps as f64;
            let t = round_half_even(beta_quantile(p, a, b) * 999.0) as usize;
            if last != Some(t) {
                v.push(self.table[t]);
            }
            last = Some(t);
        }
        v.push(0.0);
        v
    }

    fn linear_quadratic(&self, steps: usize, threshold: f64) -> Vec<f32> {
        if steps == 1 {
            return vec![self.sigma_max(), 0.0];
        }
        let linear_steps = steps / 2;
        let mut sched: Vec<f64> = (0..linear_steps).map(|i| i as f64 * threshold / linear_steps as f64).collect();
        let diff = linear_steps as f64 - threshold * steps as f64;
        let quadratic_steps = (steps - linear_steps) as f64;
        let quadratic_coef = diff / (linear_steps as f64 * quadratic_steps * quadratic_steps);
        let linear_coef = threshold / linear_steps as f64 - 2.0 * diff / (quadratic_steps * quadratic_steps);
        let c = quadratic_coef * (linear_steps * linear_steps) as f64;
        sched.extend((linear_steps..steps).map(|i| quadratic_coef * (i * i) as f64 + linear_coef * i as f64 + c));
        sched.push(1.0);
        sched.into_iter().map(|x| (1.0 - x) as f32 * self.sigma_max()).collect()
    }
}

/// torch.linspace in float32 (its two-sided form: from the start for the first half, from the end for the rest)
fn linspace32(start: f32, end: f32, n: usize) -> Vec<f32> {
    match n {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f32;
            let half = n / 2;
            (0..n).map(|i| if i < half { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
        }
    }
}

/// Karras et al.'s schedule, then 0
fn karras(n: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f32> {
    let (lo, hi) = (sigma_min.powf(1.0 / rho), sigma_max.powf(1.0 / rho));
    let ramp = |i: usize| if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
    let mut v: Vec<f32> = (0..n).map(|i| (hi + ramp(i) * (lo - hi)).powf(rho) as f32).collect();
    v.push(0.0);
    v
}

/// even in log sigma, then 0
fn exponential(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f32> {
    let (a, b) = (sigma_max.ln(), sigma_min.ln());
    let ramp = |i: usize| if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
    let mut v: Vec<f32> = (0..n).map(|i| (a + ramp(i) * (b - a)).exp() as f32).collect();
    v.push(0.0);
    v
}

/// even in arctan sigma, then 0 (one step: just sigma_max, where ComfyUI divides by zero)
fn kl_optimal(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| {
            let a = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
            (a * sigma_min.atan() + (1.0 - a) * sigma_max.atan()).tan() as f32
        })
        .collect();
    v.push(0.0);
    v
}

/// numpy.rint
fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}

/// The `p` quantile of Beta(a, b) (scipy's beta.ppf), by bisection on the regularized incomplete beta function
fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    if a == b && p == 0.5 {
        return 0.5;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if incomplete_beta(mid, a, b) < p { lo = mid } else { hi = mid }
        if hi - lo < 1e-15 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// I_x(a, b), the regularized incomplete beta function (the continued fraction of Numerical Recipes' betacf)
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(x, a, b) / a
    } else {
        1.0 - front * beta_cf(1.0 - x, b, a) / b
    }
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..1000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, n = 9: about 15 digits)
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8, 771.323_428_777_653_1,
        -176.615_029_162_140_6, 12.507_343_278_686_905, -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schedule_runs_from_noise_to_image() {
        for s in Schedule::ALL {
            let v = sigmas(s, 8, 3.0);
            if !matches!(s, Schedule::DdimUniform | Schedule::Beta) {
                assert_eq!(v.len(), 9, "{s:?}");
            }
            assert!(v[0] > 0.9 && v[0] <= 1.0, "{s:?}: {v:?}");
            assert_eq!(v[v.len() - 1], 0.0, "{s:?}");
            assert!(v.windows(2).all(|w| w[0] >= w[1]), "{s:?} not decreasing: {v:?}");
            assert_eq!(Schedule::parse(s.name()), Some(s));
        }
        assert_eq!(Schedule::parse("sgm-uniform"), Some(Schedule::SgmUniform));
    }

    #[test]
    fn shift_moves_steps_toward_noise() {
        let plain = sigmas(Schedule::Simple, 4, 1.0);
        let shifted = sigmas(Schedule::Shift, 4, 3.0);
        assert!(shifted[2] > plain[2]);
        assert_eq!(sigmas(Schedule::Shift, 4, 1.0), plain);
    }

    #[test]
    fn beta_quantiles_are_scipys() {
        // scipy.stats.beta.ppf(p, 0.6, 0.6)
        for (p, want) in [(0.1, 0.039_584_405_588_780_85), (0.25, 0.175_680_378_652_694_98), (0.9, 0.960_415_594_411_219_2)] {
            let got = beta_quantile(p, 0.6, 0.6);
            assert!((got - want).abs() < 1e-9, "{p}: {got} vs {want}");
        }
    }

    #[test]
    fn every_scheduler_matches_comfyui() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/samplers/expected.txt");
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let (mut checked, mut worst) = (0, 0.0f32);
        for line in text.lines().filter(|l| l.starts_with("schedule ")) {
            let w: Vec<&str> = line.split_whitespace().collect();
            let s = Schedule::parse(w[1]).unwrap();
            let (steps, shift): (u32, f32) = (w[2].parse().unwrap(), w[3].parse().unwrap());
            let want: Vec<f32> = w[4..].iter().map(|v| v.parse().unwrap()).collect();
            let got = sigmas(s, steps, shift);
            assert_eq!(got.len(), want.len(), "{line}\n got {got:?}");
            for (g, e) in got.iter().zip(&want) {
                assert!((g - e).abs() <= 1e-5 * e.abs().max(1e-3), "{} {steps} {shift}: {got:?}\nwant {want:?}", w[1]);
                worst = worst.max((g - e).abs() / e.abs().max(1e-3));
            }
            checked += 1;
        }
        assert!(checked >= 100, "{checked} schedules checked");
        println!("{checked} schedules, max relative error vs ComfyUI {worst:.2e}");
    }
}
