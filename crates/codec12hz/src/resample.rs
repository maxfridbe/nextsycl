//! A recording to the codec's rate (24 kHz) before it is encoded, as qwen-tts takes it.

use crate::RATE;

/// Mono samples at `rate` to 24 kHz (a windowed-sinc resampler: Hann, 32 zero crossings, cut at the lower Nyquist)
pub fn resample(x: &[f32], rate: u32) -> Vec<f32> {
    if rate == RATE || x.is_empty() {
        return x.to_vec();
    }
    let ratio = RATE as f64 / rate as f64;
    let cut = ratio.min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let center = i as f64 / ratio;
        let c0 = center.floor() as isize;
        let mut s = 0.0;
        for j in c0 - half..=c0 + half {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = (j as f64 - center) * cut;
            if d.abs() >= zc {
                continue;
            }
            let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
            let win = 0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos();
            s += x[j as usize] as f64 * sinc * win * cut;
        }
        s as f32
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resampled_tone_keeps_its_length_ratio() {
        let x: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin()).collect();
        assert_eq!(resample(&x, 16000).len(), 24000);
        assert_eq!(resample(&x, RATE), x);
    }
}
