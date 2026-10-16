//! The paint model's sampler: diffusers' UniPCMultistepScheduler as Tencent configures it - 1,000 training steps,
//! scaled-linear betas (0.00085 to 0.012) rescaled to a zero terminal SNR (the last alpha-bar then 2^-24),
//! v-prediction, "trailing" timesteps, order 2 with the B(h) = e^h - 1 ("bh2") predictor and corrector, the data
//! prediction form, lower orders at the start and the last step, and a final sigma of 0.
//!
//! Host math on the latents (12 pictures of 64 x 64 x 4: a few hundred thousand values).

const TRAIN: usize = 1000;

pub struct UniPc {
    pub timesteps: Vec<usize>,
    /// sigma = sqrt((1 - abar) / abar) per step, then 0
    pub sigmas: Vec<f64>,
    outputs: Vec<Vec<f32>>,
    last_sample: Option<Vec<f32>>,
    lower_order_nums: usize,
    this_order: usize,
    step: usize,
}

/// alpha-bar per training step (zero terminal SNR)
pub fn alphas_cumprod() -> Vec<f64> {
    let (a, b) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let betas: Vec<f64> = (0..TRAIN).map(|i| { let s = a + (b - a) * i as f64 / (TRAIN - 1) as f64; s * s }).collect();
    let mut ac = Vec::with_capacity(TRAIN);
    let mut p = 1f64;
    for b in &betas {
        p *= 1.0 - b;
        ac.push(p);
    }
    // rescale_zero_terminal_snr: shift and scale sqrt(abar) so the last is 0 and the first is kept
    let sq: Vec<f64> = ac.iter().map(|x| x.sqrt()).collect();
    let (s0, st) = (sq[0], sq[TRAIN - 1]);
    let mut out: Vec<f64> = sq.iter().map(|x| ((x - st) * s0 / (s0 - st)).powi(2)).collect();
    out[TRAIN - 1] = 2f64.powi(-24);
    out
}

/// (alpha_t, sigma_t) of a sigma
fn alpha_sigma(s: f64) -> (f64, f64) {
    let a = 1.0 / (s * s + 1.0).sqrt();
    (a, s * a)
}

fn lambda(s: f64) -> f64 {
    let (a, sg) = alpha_sigma(s);
    a.ln() - sg.ln()
}

impl UniPc {
    pub fn new(steps: usize) -> UniPc {
        let ac = alphas_cumprod();
        let ratio = TRAIN as f64 / steps as f64;
        // trailing: round(arange(1000, 0, -ratio)) - 1
        let timesteps: Vec<usize> = (0..steps).map(|i| ((TRAIN as f64 - i as f64 * ratio).round() as usize).saturating_sub(1)).collect();
        let mut sigmas: Vec<f64> = timesteps.iter().map(|&t| ((1.0 - ac[t]) / ac[t]).sqrt() as f32 as f64).collect();
        sigmas.push(0.0);
        UniPc { timesteps, sigmas, outputs: Vec::new(), last_sample: None, lower_order_nums: 0, this_order: 1, step: 0 }
    }

    /// x0 from the model's v at this step
    fn x0(&self, v: &[f32], x: &[f32]) -> Vec<f32> {
        let (a, s) = alpha_sigma(self.sigmas[self.step]);
        x.iter().zip(v).map(|(x, v)| (a * *x as f64 - s * *v as f64) as f32).collect()
    }

    /// The (h_phi_1, B(h), rhos) of an update from sigma index `s0` to `t` with the earlier outputs' sigma indices
    fn coefficients(&self, t: usize, s0: usize, earlier: &[usize], corrector: bool) -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
        let (sig_t, sig_s0) = (self.sigmas[t], self.sigmas[s0]);
        let (alpha_t, sigma_t) = alpha_sigma(sig_t);
        let (_, sigma_s0) = alpha_sigma(sig_s0);
        let h = lambda(sig_t) - lambda(sig_s0);
        let rks: Vec<f64> = earlier.iter().map(|&si| (lambda(self.sigmas[si]) - lambda(sig_s0)) / h).collect();
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1();
        let order = earlier.len() + 1;
        let mut r_all: Vec<f64> = rks.clone();
        r_all.push(1.0);
        let mut b = Vec::new();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut fact = 1f64;
        for i in 1..=order {
            b.push(h_phi_k * fact / b_h);
            fact *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / fact;
        }
        let rhos = if corrector {
            if order == 1 { vec![0.5] } else { solve(&r_all, &b, order) }
        } else if order == 2 {
            vec![0.5]
        } else if order > 2 {
            solve(&r_all[..order - 1], &b[..order - 1], order - 1)
        } else {
            vec![]
        };
        (sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * b_h, rhos, rks)
    }

    /// One step: the model's v at the current timestep for latents x; the next latents
    pub fn step(&mut self, v: &[f32], x: &[f32]) -> Vec<f32> {
        let m = self.x0(v, x);
        let mut sample = x.to_vec();
        // the corrector, with this output, on the last step's prediction
        if self.step > 0 {
            if let Some(last) = &self.last_sample {
                let order = self.this_order;
                let s0 = self.step - 1;
                let earlier: Vec<usize> = (1..order).map(|i| self.step - (i + 1)).collect();
                let (ratio, c0, cb, rhos, rks) = self.coefficients(self.step, s0, &earlier, true);
                let m0 = self.outputs.last().expect("an output");
                let d1s: Vec<&Vec<f32>> = (1..order).map(|i| &self.outputs[self.outputs.len() - 1 - i]).collect();
                let rc = rhos[rhos.len() - 1];
                sample = (0..x.len()).map(|j| {
                    let mut corr = 0f64;
                    for (k, mi) in d1s.iter().enumerate() {
                        corr += rhos[k] * (mi[j] as f64 - m0[j] as f64) / rks[k];
                    }
                    let d1t = m[j] as f64 - m0[j] as f64;
                    (ratio * last[j] as f64 - c0 * m0[j] as f64 - cb * (corr + rc * d1t)) as f32
                }).collect();
            }
        }
        self.outputs.push(m);
        if self.outputs.len() > 2 {
            self.outputs.remove(0);
        }
        let this_order = 2usize.min(self.timesteps.len() - self.step);
        self.this_order = this_order.min(self.lower_order_nums + 1);
        self.last_sample = Some(sample.clone());
        // the predictor
        let order = self.this_order;
        let earlier: Vec<usize> = (1..order).map(|i| self.step - i).collect();
        let (ratio, c0, cb, rhos, rks) = self.coefficients(self.step + 1, self.step, &earlier, false);
        let m0 = self.outputs.last().expect("an output");
        let prev: Vec<f32> = (0..x.len()).map(|j| {
            let mut pred = 0f64;
            for k in 0..order - 1 {
                let mi = &self.outputs[self.outputs.len() - 2 - k];
                pred += rhos[k] * (mi[j] as f64 - m0[j] as f64) / rks[k];
            }
            let s = sample[j] as f64;
            // the last step (sigma 0): sigma_t / sigma_s0 = 0, and the data prediction is all there is
            let base = if ratio.is_finite() { ratio * s } else { 0.0 };
            (base - c0 * m0[j] as f64 - cb * pred) as f32
        }).collect();
        if self.lower_order_nums < 2 {
            self.lower_order_nums += 1;
        }
        self.step += 1;
        prev
    }
}

/// Solves R x = b for the first n of R's rows (R[i][j] = rks[j]^i), by Gaussian elimination
fn solve(rks: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut a: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| rks[j].powi(i as i32)).chain(std::iter::once(b[i])).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap_or(c);
        a.swap(c, p);
        for r in 0..n {
            if r != c && a[c][c] != 0.0 {
                let f = a[r][c] / a[c][c];
                let pivot = a[c].clone();
                for (x, p) in a[r].iter_mut().zip(&pivot).skip(c) {
                    *x -= f * p;
                }
            }
        }
    }
    (0..n).map(|i| a[i][n] / a[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_timesteps_and_a_zero_terminal_snr() {
        let s = UniPc::new(15);
        assert_eq!(s.timesteps, vec![999, 932, 866, 799, 732, 666, 599, 532, 466, 399, 332, 266, 199, 132, 66]);
        assert_eq!(s.sigmas.len(), 16);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        // abar at 999 is 2^-24: sigma ~ 4096
        assert!((s.sigmas[0] - 4096.0).abs() < 1.0, "{}", s.sigmas[0]);
        let ac = alphas_cumprod();
        assert!((ac[0] - (1.0 - 0.00085)).abs() < 1e-9);
        assert!(s.sigmas.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn a_perfect_model_lands_on_its_data() {
        // a model that knows x0 (v = alpha eps - sigma x0 for the sample at hand) ends at x0
        let x0 = [0.3f32, -0.7, 1.1];
        let eps = [0.9f32, -0.2, 0.5];
        let mut s = UniPc::new(15);
        let (a, sg) = alpha_sigma(s.sigmas[0]);
        let mut x: Vec<f32> = (0..3).map(|i| (a * x0[i] as f64 + sg * eps[i] as f64) as f32).collect();
        for k in 0..15 {
            let (a, sg) = alpha_sigma(s.sigmas[k]);
            // eps implied by x and x0
            let v: Vec<f32> = (0..3).map(|i| {
                let e = (x[i] as f64 - a * x0[i] as f64) / sg;
                (a * e - sg * x0[i] as f64) as f32
            }).collect();
            x = s.step(&v, &x);
        }
        for i in 0..3 {
            assert!((x[i] - x0[i]).abs() < 1e-3, "{x:?}");
        }
    }
}
