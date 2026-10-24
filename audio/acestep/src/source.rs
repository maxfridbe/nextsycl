//! A song's source recording as the handler takes it (core/generation/handler/io_audio.py `process_src_audio`): mono
//! doubled to stereo, the first two channels kept, resampled to 48 kHz by torchaudio's resampler (its default:
//! windowed sinc, Hann, 6 zero crossings, rolloff 0.99), clamped to [-1, 1]; channel-major [2, n].

use nextsycl_audio::{Audio, Error, Result};

use crate::vae::RATE;

/// `a` as the handler's source: stereo [2, n] at 48 kHz, channel-major
pub fn stereo48(a: &Audio) -> Result<Vec<f32>> {
    let ch = a.channels.max(1) as usize;
    let n = a.samples.len() / ch;
    if n == 0 || a.rate == 0 {
        return Err(Error("audio: no samples".into()));
    }
    let chan = |c: usize| -> Vec<f32> { (0..n).map(|i| a.samples[i * ch + c.min(ch - 1)]).collect() };
    let (l, r) = (chan(0), chan(1));
    let (l, r) = if a.rate == RATE { (l, r) } else { (resample(&l, a.rate, RATE), resample(&r, a.rate, RATE)) };
    let mut out = Vec::with_capacity(l.len() * 2);
    out.extend(l.iter().map(|x| x.clamp(-1.0, 1.0)));
    out.extend(r.iter().map(|x| x.clamp(-1.0, 1.0)));
    Ok(out)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99): the kernel of each output
/// phase over the padded input, a stride of the reduced input rate, cut to ceil(new n / orig)
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from as u64, to as u64);
    let (orig, new) = ((from as u64 / g) as usize, (to as u64 / g) as usize);
    let (width_z, rolloff) = (6.0f64, 0.99f64);
    let base = orig.min(new) as f64 * rolloff;
    let width = (width_z * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    let mut kern = vec![0f64; new * klen];
    for j in 0..new {
        for (k, w) in kern[j * klen..(j + 1) * klen].iter_mut().enumerate() {
            let idx = (k as f64 - width as f64) / orig as f64;
            let mut t = (-(j as f64) / new as f64 + idx) * base;
            t = t.clamp(-width_z, width_z);
            let window = (t * std::f64::consts::PI / width_z / 2.0).cos().powi(2);
            let tp = t * std::f64::consts::PI;
            let sinc = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            *w = sinc * window * base / orig as f64;
        }
    }
    // padded: width before, width + orig after
    let n = x.len();
    let at = |i: isize| -> f64 { if i < 0 || i as usize >= n { 0.0 } else { x[i as usize] as f64 } };
    let frames = (n + 2 * width + orig - klen) / orig + 1;
    let target = (new * n).div_ceil(orig);
    let mut out = Vec::with_capacity(target);
    'all: for f in 0..frames {
        for j in 0..new {
            if out.len() == target {
                break 'all;
            }
            let k = &kern[j * klen..(j + 1) * klen];
            let s0 = (f * orig) as isize - width as isize;
            out.push(k.iter().enumerate().map(|(i, w)| w * at(s0 + i as isize)).sum::<f64>() as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_keeps_its_pitch_and_level_through_the_resampler() {
        let n = 44_100;
        let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5).collect();
        let y = resample(&x, 44_100, 48_000);
        assert_eq!(y.len(), 48_000);
        // away from the edges: the same tone at 48 kHz
        let err = (2000..46_000).map(|i| (y[i] - (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5).abs()).fold(0f32, f32::max);
        assert!(err < 2e-3, "{err}");
        let a = Audio { rate: 48_000, channels: 1, samples: vec![0.25, 2.0] };
        assert_eq!(stereo48(&a).unwrap(), vec![0.25, 1.0, 0.25, 1.0]);
    }
}
