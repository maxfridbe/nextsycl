//! The noise levels Qwen-Image 2.1 samples on: diffusers' FlowMatchEulerDiscreteScheduler as its pipeline configures
//! it - `steps` levels evenly from 1 to 1/steps, shifted by the image's size (dynamic shifting, exponential: mu from
//! the token count between 256 and 8,192 tokens, 0.5 to 0.9), stretched so the last level is 0.02, then 0 appended.

/// mu for `tokens` image tokens (the pipeline's calculate_shift with the scheduler's configuration)
pub fn mu(tokens: usize) -> f64 {
    let (base_len, max_len, base_shift, max_shift) = (256.0, 8192.0, 0.5, 0.9);
    let m = (max_shift - base_shift) / (max_len - base_len);
    tokens as f64 * m + (base_shift - m * base_len)
}

/// The `steps + 1` sigmas, from about 1 down to 0
pub fn sigmas(steps: usize, tokens: usize) -> Vec<f32> {
    let n = steps.max(1);
    let mu = mu(tokens);
    let even: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 * (1.0 - 1.0 / n as f64) / (n - 1).max(1) as f64).collect();
    // exponential time shift: e^mu / (e^mu + (1 / t - 1))
    let shifted: Vec<f64> = even.iter().map(|t| mu.exp() / (mu.exp() + (1.0 / t - 1.0))).collect();
    // stretch to the terminal 0.02: 1 - (1 - t) / ((1 - t_last) / (1 - 0.02))
    let last = 1.0 - shifted[n - 1];
    let scale = last / (1.0 - 0.02);
    let mut out: Vec<f32> = shifted.iter().map(|t| (1.0 - (1.0 - t) / scale) as f32).collect();
    out.push(0.0);
    out
}

/// A few-step model's own nodes (`nodes`, from 1 down), then 0: with the image size's exponential shift
/// (`dynamic`, as diffusers' pipeline applies it to `sigmas=` - Viggle's turbo) or exactly as given (Pruna's)
pub fn preset(nodes: &[f64], tokens: usize, dynamic: bool) -> Vec<f32> {
    let e = mu(tokens).exp();
    let mut out: Vec<f32> = nodes.iter().map(|&t| if dynamic { e / (e + (1.0 / t - 1.0)) } else { t } as f32).collect();
    out.push(0.0);
    out
}

/// The last `steps` steps of `sigmas` (a run started partway: an edit's strength below 1, as ComfyUI's denoise)
pub fn tail(sigmas: &[f32], steps: usize) -> Vec<f32> {
    sigmas[sigmas.len().saturating_sub(steps + 1)..].to_vec()
}

/// A fixed schedule started partway: `strength` itself, then the levels of `sigmas` below it (a few-step model's
/// own levels can sit far apart: starting at the first one below would skip most of the change asked)
pub fn below(sigmas: &[f32], strength: f32) -> Vec<f32> {
    std::iter::once(strength).chain(sigmas.iter().copied().filter(|s| *s < strength)).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_schedule_ends_at_the_terminal_level() {
        let s = super::sigmas(20, 1024);
        assert_eq!(s.len(), 21);
        assert!((s[19] - 0.02).abs() < 1e-6);
        assert_eq!(s[20], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn a_preset_keeps_its_nodes_or_shifts_them() {
        let n = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25];
        assert_eq!(super::preset(&n, 4096, false), vec![1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0]);
        let s = super::preset(&n, 4096, true);
        assert_eq!(s[0], 1.0);
        assert!(s[5] > 0.25 && s[6] == 0.0, "a large image shifts toward noise: {s:?}");
    }

    #[test]
    fn a_partway_run_keeps_its_steps_and_starts_lower() {
        let full = super::sigmas(40, 4096);
        let t = super::tail(&full, 20);
        assert_eq!(t.len(), 21);
        assert!(t[0] < 0.9 && t[20] == 0.0);
        let b = super::below(&[1.0, 0.9, 0.5, 0.25, 0.0], 0.6);
        assert_eq!(b, vec![0.6, 0.5, 0.25, 0.0]);
        assert_eq!(super::below(&[1.0, 0.9, 0.0], 0.05), vec![0.05, 0.0]);
    }
}
