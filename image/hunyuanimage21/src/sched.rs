//! The sampling the official pipeline does: the static-shift flow schedule, the timesteps (MeanFlow's next one), and the
//! base model's guidance - plain CFG, or the pipeline's default "MIX_mode_0": APG (adaptive projected guidance) after
//! a few steps of plain CFG.

/// The pipeline's get_timesteps_sigmas: linspace(1, 0, steps + 1), shifted s' = shift s / (1 + (shift - 1) s)
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    nextsycl_diffusion::schedule::sigmas(nextsycl_diffusion::Schedule::Shift, steps.max(1) as u32, shift)
}

/// The (t, t_next) a step passes the DiT, as timesteps (sigma x 1000): t_next is MeanFlow's (the next step's t, 0 on
/// the last), None without it
pub fn timesteps(sigmas: &[f32], meanflow: bool) -> Vec<(f32, Option<f32>)> {
    let n = sigmas.len().saturating_sub(1);
    (0..n).map(|i| (sigmas[i] * 1000.0, meanflow.then(|| if i + 1 < n { sigmas[i + 1] * 1000.0 } else { 0.0 }))).collect()
}

/// The pipeline's AdaptiveProjectedGuidance (guidance 10, momentum -0.5, norm threshold 10, eta 0) as MIX_mode_0
/// runs it: plain CFG at the request's scale through step `start` (the guider still sees every step, for its momentum),
/// then APG
pub struct Apg {
    pub start: usize,
    running: Vec<f64>,
}

pub const APG_SCALE: f64 = 10.0;
pub const APG_MOMENTUM: f64 = -0.5;
pub const APG_THRESHOLD: f64 = 10.0;
/// the steps of plain CFG before APG: 5, or 38 when the prompt's glyph text is set (the pipeline's OCR guider)
pub const APG_START: usize = 5;
pub const APG_START_OCR: usize = 38;

impl Apg {
    pub fn new(start: usize) -> Apg {
        Apg { start, running: Vec::new() }
    }

    /// The guided velocity of step `i` from the prompt's (`cond`) and the negative's (`uncond`); `cfg` the plain
    /// guidance's scale
    pub fn guide(&mut self, i: usize, cond: &[f32], uncond: &[f32], cfg: f32) -> Vec<f32> {
        let n = cond.len();
        if i == 0 || self.running.len() != n {
            self.running = vec![0.0; n];
        }
        // the momentum buffer: running = diff + momentum running
        for ((r, c), u) in self.running.iter_mut().zip(cond).zip(uncond) {
            *r = (*c as f64 - *u as f64) + APG_MOMENTUM * *r;
        }
        if i <= self.start {
            return cond.iter().zip(uncond).map(|(c, u)| u + cfg * (c - u)).collect();
        }
        // the update clipped to the norm threshold, its part along the prompt's prediction removed (eta 0)
        let norm = self.running.iter().map(|d| d * d).sum::<f64>().sqrt();
        let scale = (APG_THRESHOLD / norm).min(1.0);
        let cn = cond.iter().map(|c| (*c as f64) * (*c as f64)).sum::<f64>().sqrt().max(1e-12);
        let dot: f64 = self.running.iter().zip(cond).map(|(d, c)| d * scale * (*c as f64 / cn)).sum();
        self.running.iter().zip(cond).zip(uncond).map(|((d, c), u)| {
            let par = dot * (*c as f64 / cn);
            let orth = d * scale - par;
            (*u as f64 + APG_SCALE * orth) as f32
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_schedule_is_the_pipelines_static_shift() {
        let s = super::sigmas(8, 4.0);
        assert_eq!(s.len(), 9);
        assert_eq!((s[0], s[8]), (1.0, 0.0));
        // linspace(1, 0, 9)[4] = 0.5 -> 4 * 0.5 / (1 + 3 * 0.5) = 0.8
        assert!((s[4] - 0.8).abs() < 1e-6);
        let t = super::timesteps(&s, true);
        assert_eq!(t.len(), 8);
        assert_eq!(t[0].0, 1000.0);
        assert!((t[0].1.unwrap() - s[1] * 1000.0).abs() < 1e-3);
        assert_eq!(t[7].1, Some(0.0));
        assert!(super::timesteps(&s, false).iter().all(|x| x.1.is_none()));
    }

    #[test]
    fn apg_is_cfg_at_first_then_projected() {
        let mut g = super::Apg::new(1);
        let (c, u) = (vec![1.0f32, 0.0], vec![0.0f32, 0.0]);
        assert_eq!(g.guide(0, &c, &u, 3.5), vec![3.5, 0.0]);
        // step 1 (<= start): still CFG; running = diff + -0.5 running = (0.5, 0)
        assert_eq!(g.guide(1, &c, &u, 3.5), vec![3.5, 0.0]);
        // step 2: running = (1 - 0.25, 0); along cond entirely: nothing orthogonal is left
        let v = g.guide(2, &c, &u, 3.5);
        assert!(v.iter().all(|x| x.abs() < 1e-6), "{v:?}");
        // an update across the prediction passes, clipped to the threshold
        let mut g = super::Apg::new(0);
        let v = g.guide(1, &[0.0, 100.0], &[0.0, 0.0], 3.5);
        assert_eq!(v, vec![0.0, 0.0]);
        let mut g = super::Apg::new(0);
        g.guide(0, &[1.0, 0.0], &[1.0, 0.0], 3.5);
        let v = g.guide(1, &[1.0, 0.0], &[1.0, -100.0], 3.5);
        // diff (0, 100) clipped to norm 10, orthogonal to (1, 0): uncond + 10 x (0, 10)
        assert!((v[0] - 1.0).abs() < 1e-5 && (v[1] - 0.0).abs() < 1e-4, "{v:?}");
    }
}
