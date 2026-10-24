//! The flow sampler's arithmetic on the latents (on the host: a song's latents are [frames, 64] floats, 25 frames a
//! second - 18 MB for ten minutes), as the turbo model's `generate_audio` (modeling_acestep_v15_turbo.py) does it:
//! the timestep schedule, the Euler step, DCW (the wavelet-domain correction, on by default for turbo: Haar over
//! pairs of frames), a repaint's injection of the noised source outside the repainted span and its boundary blend.

/// The latents' channels
pub const LATENT: usize = 64;

/// The schedule: `steps` timesteps from 1 (no final 0), shifted (t' = s t / (1 + (s - 1) t)) - generate_audio's
/// variable-step schedule, which the platform's 8 steps at shift 3 take (= its SHIFT_TIMESTEPS[3]); at most 20.
pub fn schedule(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps.clamp(1, 20);
    (0..n)
        .map(|i| {
            let t = 1.0 - i as f64 / n as f64;
            let s = shift as f64;
            (if s != 1.0 { s * t / (1.0 + (s - 1.0) * t) } else { t }) as f32
        })
        .collect()
}

/// x -= v dt (the Euler step from t to t - dt)
pub fn euler(x: &mut [f32], v: &[f32], dt: f32) {
    for (a, b) in x.iter_mut().zip(v) {
        *a -= b * dt;
    }
}

/// The clean estimate x - v t
pub fn denoised(x: &[f32], v: &[f32], t: f32) -> Vec<f32> {
    x.iter().zip(v).map(|(a, b)| a - b * t).collect()
}

/// DCW "double" (dcw_primitives.py dcw_double, haar): a one-level Haar transform along the frames of x (the step's
/// result) and y (the clean estimate), the low band pushed away from y's by `low`, the high band by `high`, and back.
/// Frames pair as (2k, 2k + 1); an odd last frame pairs with a zero (pytorch_wavelets' zero padding, trimmed after).
/// The reference's dumps match this to 5e-7 (the 10 s song's eight steps).
pub fn dcw_double(x: &mut [f32], y: &[f32], low: f32, high: f32) {
    if low == 0.0 && high == 0.0 {
        return;
    }
    let t = x.len() / LATENT;
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let mut k = 0;
    while k < t {
        let pair = k + 1 < t;
        for c in 0..LATENT {
            let (a, ya) = (x[k * LATENT + c], y[k * LATENT + c]);
            let (b, yb) = if pair { (x[(k + 1) * LATENT + c], y[(k + 1) * LATENT + c]) } else { (0.0, 0.0) };
            let (mut lo, mut hi) = ((a + b) * r, (a - b) * r);
            let (ylo, yhi) = ((ya + yb) * r, (ya - yb) * r);
            lo += low * (lo - ylo);
            hi += high * (hi - yhi);
            x[k * LATENT + c] = (lo + hi) * r;
            if pair {
                x[(k + 1) * LATENT + c] = (lo - hi) * r;
            }
        }
        k += 2;
    }
}

/// DCW's per-step strengths at t (DCWCorrector, double mode: the low band t * scaler, the high (1 - t) * high_scaler)
pub fn dcw_strengths(t: f32, scaler: f32, high_scaler: f32) -> (f32, f32) {
    (t * scaler, (1.0 - t) * high_scaler)
}

/// A repaint's injection after a step (`_repaint_step_injection`): outside the repainted frames (mask false) x takes
/// the source noised to t_next
pub fn repaint_inject(x: &mut [f32], clean: &[f32], mask: &[bool], t_next: f32, noise: &[f32]) {
    for (f, m) in mask.iter().enumerate() {
        if !*m {
            for c in 0..LATENT {
                let i = f * LATENT + c;
                x[i] = t_next * noise[i] + (1.0 - t_next) * clean[i];
            }
        }
    }
}

/// The repainted span's edges blended into the source over `cf` frames (`_repaint_boundary_blend`)
pub fn repaint_blend(x: &mut [f32], clean: &[f32], mask: &[bool], cf: usize) {
    let t = mask.len();
    let mut soft: Vec<f32> = mask.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect();
    if cf > 0 && mask.iter().any(|m| *m) && !mask.iter().all(|m| *m) {
        let left = mask.iter().position(|m| *m).unwrap_or(0);
        let right = mask.iter().rposition(|m| *m).map_or(t, |i| i + 1);
        // torch.linspace(0, 1, n + 2)[1:-1] before, linspace(1, 0, n + 2)[1:-1] after
        let fs = left.saturating_sub(cf);
        let n = left - fs;
        for (j, s) in soft[fs..left].iter_mut().enumerate() {
            *s = (j + 1) as f32 / (n + 1) as f32;
        }
        let fe = (right + cf).min(t);
        let n = fe - right;
        for (j, s) in soft[right..fe].iter_mut().enumerate() {
            *s = 1.0 - (j + 1) as f32 / (n + 1) as f32;
        }
    }
    for (f, s) in soft.iter().enumerate() {
        for c in 0..LATENT {
            let i = f * LATENT + c;
            x[i] = s * x[i] + (1.0 - s) * clean[i];
        }
    }
}

/// APG's momentum (MomentumBuffer, -0.75): the running average of the guidance differences
pub struct Momentum {
    avg: Option<Vec<f32>>,
}

impl Default for Momentum {
    fn default() -> Momentum {
        Momentum { avg: None }
    }
}

/// apg_forward (eta 0, norm threshold 2.5) over latents [frames, 64] with dims=[1] - the frames: each channel's
/// difference (its momentum's running average) clipped to norm 2.5 over the frames, its part along the conditional
/// prediction (that channel's, over the frames) taken out in double, scaled by g - 1 onto the conditional
pub fn apg(cond: &[f32], uncond: &[f32], g: f32, m: &mut Momentum) -> Vec<f32> {
    let mut diff: Vec<f32> = cond.iter().zip(uncond).map(|(c, u)| c - u).collect();
    if let Some(a) = &m.avg {
        for (d, r) in diff.iter_mut().zip(a) {
            *d += -0.75 * r;
        }
    }
    m.avg = Some(diff.clone());
    let frames = cond.len() / LATENT;
    let mut out = cond.to_vec();
    for c in 0..LATENT {
        let col = |v: &[f32], f: usize| v[f * LATENT + c] as f64;
        let dn = (0..frames).map(|f| col(&diff, f).powi(2)).sum::<f64>().sqrt() as f32;
        let sc = (2.5f32 / dn).min(1.0);
        let pn = (0..frames).map(|f| col(cond, f).powi(2)).sum::<f64>().sqrt().max(1e-12);
        // F.normalize's eps 1e-12; the clipped difference's projection on the unit conditional column
        let dot: f64 = (0..frames).map(|f| (diff[f * LATENT + c] * sc) as f64 * col(cond, f) / pn).sum();
        for f in 0..frames {
            let d = (diff[f * LATENT + c] * sc) as f64;
            let par = dot * col(cond, f) / pn;
            out[f * LATENT + c] = cond[f * LATENT + c] + (g - 1.0) * (d - par) as f32;
        }
    }
    out
}

/// The base model's schedule: linspace(1, 0, n + 1) shifted (its final 0 kept)
pub fn schedule_base(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps.max(1);
    (0..=n)
        .map(|i| {
            let t = (1.0 - i as f64 / n as f64) as f32;
            if shift != 1.0 { shift * t / (1.0 + (shift - 1.0) * t) } else { t }
        })
        .collect()
}

/// A seeded standard normal draw (xorshift64* and Box-Muller): the starting noise (the reference's torch generator is
/// not reproduced - a check feeds its noise instead)
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }

    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_steps_at_shift_three_are_the_turbo_table() {
        let s = schedule(8, 3.0);
        let want = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3];
        assert_eq!(s.len(), 8);
        for (a, b) in s.iter().zip(want) {
            assert!((*a as f64 - b).abs() < 1e-6, "{s:?}");
        }
        assert_eq!(schedule(30, 3.0).len(), 20, "at most 20");
        assert!((schedule(4, 1.0)[1] - 0.75).abs() < 1e-7);
    }

    #[test]
    fn dcw_moves_the_bands_and_leaves_a_clean_step_alone() {
        // two frames: the low band (the pair's mean) and the high (their difference) pushed independently
        let mut x = vec![0.0f32; 2 * LATENT];
        x[0] = 1.0;
        x[LATENT] = 3.0;
        let y = vec![0.0f32; 2 * LATENT];
        dcw_double(&mut x, &y, 0.5, 0.0);
        // lo = 4/sqrt2 -> 1.5x; hi unchanged: a = (6/sqrt2 + -2/sqrt2)/sqrt2 = 2, b = 4
        assert!((x[0] - 2.0).abs() < 1e-6 && (x[LATENT] - 4.0).abs() < 1e-6, "{} {}", x[0], x[LATENT]);
        let mut z = x.clone();
        dcw_double(&mut z, &x, 0.3, 0.7);
        assert_eq!(z, x, "x = y: no push");
        // an odd last frame is paired with zero
        let mut o = vec![2.0f32; 3 * LATENT];
        let y = vec![0.0f32; 3 * LATENT];
        dcw_double(&mut o, &y, 0.1, 0.1);
        assert!((o[2 * LATENT] - 2.2).abs() < 1e-6, "{}", o[2 * LATENT]);
    }

    #[test]
    fn a_repaint_keeps_the_source_outside_its_span() {
        let mask = vec![false, false, true, true, false];
        let clean = vec![1.0f32; 5 * LATENT];
        let noise = vec![0.0f32; 5 * LATENT];
        let mut x = vec![9.0f32; 5 * LATENT];
        repaint_inject(&mut x, &clean, &mask, 0.25, &noise);
        assert_eq!((x[0], x[2 * LATENT], x[4 * LATENT]), (0.75, 9.0, 0.75));
        let mut y = vec![5.0f32; 5 * LATENT];
        repaint_blend(&mut y, &clean, &mask, 1);
        // one frame of fade each side: linspace(0, 1, 3)[1:-1] = 0.5
        assert_eq!((y[0], y[LATENT], y[2 * LATENT], y[4 * LATENT]), (1.0, 3.0, 5.0, 3.0));
    }

    #[test]
    fn apg_takes_out_the_part_along_the_prediction() {
        // one channel's difference along the conditional prediction is dropped, its orthogonal part scaled by g - 1
        let frames = 2;
        let mut cond = vec![0f32; frames * LATENT];
        let mut unc = vec![0f32; frames * LATENT];
        cond[0] = 1.0;
        unc[0] = 0.5;
        unc[LATENT] = -0.5;
        let out = apg(&cond, &unc, 3.0, &mut Momentum::default());
        // diff (0.5, 0.5): parallel (0.5, 0), orthogonal (0, 0.5): out = cond + 2 (0, 0.5)
        assert!((out[0] - 1.0).abs() < 1e-6 && (out[LATENT] - 1.0).abs() < 1e-6, "{} {}", out[0], out[LATENT]);
        let s = schedule_base(4, 1.0);
        assert_eq!(s, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn the_noise_is_a_standard_normal() {
        let v = Rng::new(42).normals(200_000);
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.02, "{mean} {var}");
        assert_eq!(Rng::new(7).normals(4), Rng::new(7).normals(4));
    }
}
