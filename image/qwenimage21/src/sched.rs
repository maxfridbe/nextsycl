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
}
