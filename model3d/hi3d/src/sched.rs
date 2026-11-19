//! The sampler, as sparc3d.schedulers.schedule and flow_match_heun have it, in their float32 arithmetic: the token-count
//! warp of the trajectory, the two-level sigma inflation, the second-order Heun update (an Euler predictor, then the
//! slopes at both ends averaged), guidance against the zero condition only from time 0.7 up (4 at the first time,
//! 10 t^2 on [0.7, 1)).

/// The schedule for `steps` updates over `tokens` voxels: (sigmas [steps + 1], the inflated model times [steps])
pub fn schedule(steps: usize, tokens: usize, sigma_min: f32, time_rescale: f32) -> (Vec<f32>, Vec<f32>) {
    let n = tokens as f64;
    let warp = ((19.0 * n - 20000.0) / (3.0 * n + 60000.0)).clamp(1.0, 5.0);
    // torch.linspace(1, 0, steps + 1) in float32: from the start up to half way, from the end after
    let step = -1.0f32 / steps as f32;
    let half = steps.div_ceil(2); // (steps + 1) points / 2
    let base: Vec<f32> = (0..=steps).map(|i| if i < half { 1.0 + step * i as f32 } else { 0.0 - step * (steps - i) as f32 }).collect();
    let rescale = |t: f32, f: f64| -> f32 { (f as f32 * t) / (1.0f32 + (f - 1.0) as f32 * t) };
    let warped: Vec<f32> = base.iter().map(|t| rescale(rescale(*t, time_rescale as f64), warp)).collect();
    let span = (1.0 - sigma_min as f64) as f32;
    let sigmas: Vec<f32> = warped.iter().map(|w| sigma_min + span * w).collect();
    let hats: Vec<f32> = (0..steps).map(|i| {
        let gamma = if warped[i] > 0.985 { 0.04f32 } else { 0.002f32 };
        sigmas[i] * (1.0 + gamma)
    }).collect();
    (sigmas, hats)
}

/// Whether the zero condition is evaluated at model time t, and the guided velocity's scale there
pub fn guidance(t: f32) -> Option<f32> {
    if (t as f64) < 0.7 {
        None
    } else if t >= 1.0 {
        Some(4.0)
    } else {
        Some(10.0 * (t * t))
    }
}

/// neg + scale (pos - neg), in place in pos
pub fn guide(pos: &mut [f32], neg: &[f32], scale: f32) {
    for (p, n) in pos.iter_mut().zip(neg) {
        *p = n + scale * (*p - n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_ends_and_inflates() {
        let (s, h) = schedule(25, 12961, 0.002, 2.0);
        assert_eq!((s.len(), h.len()), (26, 25));
        assert!((s[0] - 1.0).abs() < 1e-6 && (s[25] - 0.002).abs() < 1e-7);
        // the first update is inflated by 4 % (above 0.985), late ones by 0.2 %
        assert!((h[0] - 1.04).abs() < 1e-6);
        assert!((h[24] / s[24] - 1.002).abs() < 1e-5);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(guidance(1.04), Some(4.0));
        assert_eq!(guidance(0.5), None);
        assert!((guidance(0.8).unwrap() - 6.4).abs() < 1e-5);
    }
}
