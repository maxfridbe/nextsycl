//! The diffusion head's sampler: diffusers' DPMSolverMultistepScheduler as the checkpoint's generation config sets it
//! (squaredcos_cap_v2 betas over 1,000 training steps, v-prediction, DPM-Solver++ of order 2 with the midpoint
//! update, linspace timesteps, the last sigma 0, first order at the start and the end), on the host: a frame's latent
//! is 64 values. Float32 as diffusers computes it (its sigmas are float32 tensors).

/// The schedule for `steps` steps and the multistep state of one sampling
pub struct Dpm {
    /// the model's timestep at each step (999 ... )
    pub timesteps: Vec<f32>,
    /// sigma at each step, then 0
    pub sigmas: Vec<f32>,
    /// the data predictions so far (the last two)
    prev: Option<Vec<f32>>,
    at: usize,
}

const TRAIN: usize = 1000;

/// alphas_cumprod of the cosine schedule (betas_for_alpha_bar: beta_i = min(1 - abar((i + 1) / N) / abar(i / N), 0.999))
fn alphas_cumprod() -> Vec<f32> {
    let abar = |t: f64| ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).cos().powi(2);
    let mut ac = Vec::with_capacity(TRAIN);
    // torch's cumprod of float32 on the CPU accumulates in double
    let mut p = 1f64;
    for i in 0..TRAIN {
        let beta = (1.0 - abar((i + 1) as f64 / TRAIN as f64) / abar(i as f64 / TRAIN as f64)).min(0.999) as f32;
        p *= (1.0f32 - beta) as f64;
        ac.push(p as f32);
    }
    ac
}

/// (alpha_t, sigma_t) of a sigma: 1 / sqrt(sigma^2 + 1), sigma alpha_t
fn alpha_sigma(s: f32) -> (f32, f32) {
    let a = 1.0 / (s * s + 1.0).sqrt();
    (a, s * a)
}

impl Dpm {
    pub fn new(steps: usize) -> Dpm {
        let ac = alphas_cumprod();
        let steps = steps.max(1);
        // linspace(0, 999, steps + 1).round(), reversed, the last dropped
        let timesteps: Vec<usize> = (0..=steps).map(|i| ((TRAIN - 1) as f64 * i as f64 / steps as f64).round() as usize).rev().take(steps).collect();
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| ((1.0 - ac[t]) / ac[t]).sqrt()).collect();
        sigmas.push(0.0);
        Dpm { timesteps: timesteps.iter().map(|&t| t as f32).collect(), sigmas, prev: None, at: 0 }
    }

    /// A fresh sampling with the same schedule
    pub fn reset(&mut self) {
        self.prev = None;
        self.at = 0;
    }

    /// Step i's coefficients: the data prediction x0 = a_s x - s_s v, then x = ratio x - ae x0, or with `second`
    /// (the midpoint update from the previous x0, m1) x = ratio x - ae x0 - 0.5 ae inv_r0 (x0 - m1)
    pub fn coef(&self, i: usize) -> Coef {
        let n = self.timesteps.len();
        let (a_s, s_s) = alpha_sigma(self.sigmas[i]);
        let (a_t, s_t) = alpha_sigma(self.sigmas[i + 1]);
        let lambda = |a: f32, s: f32| a.ln() - s.ln();
        let l_t = lambda(a_t, s_t);
        let l_s = lambda(a_s, s_s);
        let h = l_t - l_s;
        // exp(-h) - 1, with the last step's sigma 0 (h infinite): -1
        let em1 = (-h).exp() - 1.0;
        // first order at the start and the end; else the midpoint second order: D1 = (m0 - m1) / r0, r0 = h_0 / h
        let second = i > 0 && i < n - 1;
        let inv_r0 = if second {
            let (a_1, s_1) = alpha_sigma(self.sigmas[i - 1]);
            1.0 / ((l_s - lambda(a_1, s_1)) / h)
        } else {
            0.0
        };
        Coef { a_s, s_s, ratio: s_t / s_s, ae: a_t * em1, inv_r0, second }
    }

    /// One step: the model's v-prediction `v` at the current timestep moves `x` toward the next sigma
    pub fn step(&mut self, v: &[f32], x: &mut [f32]) {
        let c = self.coef(self.at);
        let x0: Vec<f32> = x.iter().zip(v).map(|(x, v)| c.a_s * x - c.s_s * v).collect();
        match (&self.prev, c.second) {
            (Some(m1), true) => {
                for ((x, d0), m1) in x.iter_mut().zip(&x0).zip(m1) {
                    let d1 = c.inv_r0 * (d0 - m1);
                    *x = c.ratio * *x - c.ae * d0 - 0.5 * c.ae * d1;
                }
            }
            _ => {
                for (x, d0) in x.iter_mut().zip(&x0) {
                    *x = c.ratio * *x - c.ae * d0;
                }
            }
        }
        self.prev = Some(x0);
        self.at += 1;
    }
}

/// One step's coefficients (Dpm::coef)
#[derive(Clone, Copy, Debug)]
pub struct Coef {
    pub a_s: f32,
    pub s_s: f32,
    pub ratio: f32,
    pub ae: f32,
    pub inv_r0: f32,
    pub second: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// diffusers' sigmas for 10 steps (reference/vibevoice/ref.py's sched.npy)
    const SIGMAS_10: [f32; 10] = [20291.303, 6.364592, 3.1037297, 1.9806975, 1.3907515, 1.0155509, 0.7402812, 0.5223599, 0.33746067,
                                  0.17104375];

    #[test]
    fn the_schedule_is_diffusers() {
        // diffusers 0.41's DPMSolverMultistepScheduler(beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction")
        // .set_timesteps(10): timesteps and sigmas (reference/vibevoice/ref.py's sched.npy)
        let d = Dpm::new(10);
        assert_eq!(d.timesteps, vec![999.0, 899.0, 799.0, 699.0, 599.0, 500.0, 400.0, 300.0, 200.0, 100.0]);
        assert_eq!(d.sigmas.len(), 11);
        assert_eq!(*d.sigmas.last().unwrap(), 0.0);
        for (got, want) in d.sigmas.iter().zip(SIGMAS_10) {
            assert!((got - want).abs() <= 1e-5 * want, "sigma {got} vs {want}");
        }
    }

    #[test]
    fn a_perfect_prediction_lands_on_its_data() {
        // with v = alpha eps - sigma x0 for a fixed x0 and noise, every step keeps x on that path and the last is x0
        let x0 = [0.3f32, -1.2, 0.0, 2.5];
        let eps = [1.0f32, -0.5, 0.7, 0.1];
        let mut d = Dpm::new(10);
        let s0 = d.sigmas[0];
        let (a, s) = alpha_sigma(s0);
        let mut x: Vec<f32> = x0.iter().zip(&eps).map(|(x, e)| a * x + s * e).collect();
        for i in 0..10 {
            let (a, s) = alpha_sigma(d.sigmas[i]);
            let v: Vec<f32> = x0.iter().zip(&eps).map(|(x, e)| a * e - s * x).collect();
            d.step(&v, &mut x);
        }
        for (g, w) in x.iter().zip(&x0) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }
}
