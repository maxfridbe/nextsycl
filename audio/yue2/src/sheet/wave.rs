//! A recording to SheetSage2's input, 24 kHz mono, as upstream's `load_audio` makes it: `ffmpeg -i FILE -ac 1 -ar 24000
//! -f f32le` (its default preset). Two of FFmpeg's choices change what the model hears, so both are FFmpeg's here:
//!
//! - **The downmix.** libswresample mixes stereo to mono as sqrt(1/2) (L + R), not their mean: with a float output its
//!   matrix is not normalized (rematrix_maxval is INT_MAX for float formats), so the mono signal is 3 dB louder than
//!   the mean. The model's features are log-mel in dB against fixed statistics, so the level matters. (Measured on the
//!   test recording: the reference's waveform is 0.70711 (L + R) to 1.5e-7.)
//! - **The resampler.** libswresample's default: a Kaiser-windowed sinc (beta 9) of ceil(32 / factor) taps, factor =
//!   0.97 out / in (the cutoff), normalized by its sum; exact rational phases (one filter a phase of out / gcd, so
//!   48 kHz -> 24 kHz is one filter on every other sample); the first output centred on the first input; the start
//!   padded by reflecting the first samples (invert_initial_buffer: x[-n] = x[n]) and the end by reflecting the last
//!   (resample_flush: x[N + j] = x[N - 1 - j], (min(N, taps) + 1) / 2 of them). The taps are float32 as FFmpeg keeps
//!   them; the sum is in float64 here (FFmpeg's SIMD sums in float32 in its own order: the two agree to ~1e-7).
//!
//! MP3's encoder delay and padding are trimmed (gapless: the LAME / Xing header), as FFmpeg's demuxer does.

/// The model's sample rate
pub const RATE: u32 = 24_000;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The modified Bessel function I0 (its power series, to double precision)
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let (mut t, mut v) = (1.0f64, 1.0f64);
    for k in 1..500 {
        t *= q / (k as f64 * k as f64);
        v += t;
        if t < v * 1e-18 {
            break;
        }
    }
    v
}

/// FFmpeg's downmix of `channels` interleaved to mono (stereo: sqrt(1/2) (L + R) in float32; more channels: their mean)
pub fn downmix(samples: &[f32], channels: usize) -> Vec<f32> {
    match channels {
        0 | 1 => samples.to_vec(),
        2 => {
            let c = std::f32::consts::FRAC_1_SQRT_2;
            samples.chunks_exact(2).map(|f| c * f[0] + c * f[1]).collect()
        }
        n => samples.chunks_exact(n).map(|f| f.iter().sum::<f32>() / n as f32).collect(),
    }
}

/// libswresample's default resampler (see the module's comment): mono `x` at `from` to `to`
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from as u64, to as u64);
    let (p_exact, m) = ((to as u64 / g) as usize, (from as u64 / g) as usize);
    let factor = (to as f64 * 0.97 / from as f64).min(1.0);
    let mut taps = ((32.0 / factor).ceil() as usize).max(1);
    if taps > 1 && taps % 2 == 1 {
        taps += 1;
    }
    let center = (taps - 1) / 2;
    // exact rational phases when out / gcd fits FFmpeg's 1024; else 1024 phases (the nearest one: the rare odd rate)
    let phases = if p_exact <= 1024 { p_exact } else { 1024 };
    let build = |ph: usize| -> Vec<f64> {
        (0..taps).map(|i| {
            let x = std::f64::consts::PI * ((i as f64 - center as f64) - ph as f64 / phases as f64) * factor;
            let y = if x == 0.0 { 1.0 } else { x.sin() / x };
            let w = 2.0 * x / (factor * taps as f64 * std::f64::consts::PI);
            y * bessel_i0(9.0 * (1.0 - w * w).max(0.0).sqrt())
        }).collect()
    };
    // every phase normalized by phase 0's sum (build_filter's norm), the taps kept in float32
    let norm: f64 = build(0).iter().sum();
    let bank: Vec<Vec<f32>> = (0..phases).map(|ph| build(ph).iter().map(|v| (v / norm) as f32).collect()).collect();
    let n = x.len();
    let refl = n.min(taps).div_ceil(2);
    // the input with its reflections: x[-n] = x[n] ahead, x[N + j] = x[N - 1 - j] after
    let at = |i: isize| -> f32 {
        if i < 0 {
            x[((-i) as usize).min(n - 1)]
        } else if (i as usize) < n {
            x[i as usize]
        } else {
            let j = i as usize - n;
            if j < refl { x[n - 1 - j.min(n - 1)] } else { 0.0 }
        }
    };
    // output k: index (in 1/phases of an input sample) -phases * center + k * step; its taps from sample floor(index / phases)
    let step = if p_exact <= 1024 { m as f64 } else { from as f64 * phases as f64 / to as f64 };
    let limit = (n + refl) as isize;
    let mut out = Vec::new();
    let mut k = 0usize;
    loop {
        let idx = -((phases * center) as f64) + k as f64 * step;
        let sample = (idx / phases as f64).floor() as isize;
        if sample + taps as isize > limit {
            break;
        }
        let ph = ((idx - sample as f64 * phases as f64).round() as usize).min(phases - 1);
        let h = &bank[ph];
        let mut acc = 0f64;
        for (i, w) in h.iter().enumerate() {
            acc += at(sample + i as isize) as f64 * *w as f64;
        }
        out.push(acc as f32);
        k += 1;
    }
    out
}

/// A recording (any rate, its channels interleaved) as the model's input: FFmpeg's mono at 24 kHz
pub fn prepare(samples: &[f32], channels: usize, rate: u32) -> Vec<f32> {
    resample(&downmix(samples, channels), rate, RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_keeps_the_length_and_a_constant() {
        let x = vec![0.25f32; 4800];
        let y = resample(&x, 48_000, 24_000);
        assert_eq!(y.len(), 2400);
        assert!(y.iter().all(|v| (v - 0.25).abs() < 1e-5), "{:?}", &y[..4]);
        let z = resample(&vec![0.5f32; 4410], 44_100, 24_000);
        assert!((z.len() as i64 - 2400).abs() <= 1, "{}", z.len());
    }
}
