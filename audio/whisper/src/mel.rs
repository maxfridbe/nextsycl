//! The frontend, on the host: a recording at any rate to 16 kHz (a polyphase windowed-sinc resampler), then Whisper's
//! log-mel spectrogram as openai-whisper computes it (`whisper.audio.log_mel_spectrogram`): a 400-point periodic Hann
//! STFT every 160 samples (centred, reflect-padded), its power, the 128 (v3) or 80 Slaney mel filters librosa makes,
//! log10 clamped at 1e-10, floored 8 below its maximum, then (x + 4) / 4.
//!
//! The recording is padded with 30 s of silence ahead of the STFT (transcribe's `padding=N_SAMPLES`), so the last window
//! is whole; the frames that see nothing but that silence are not computed - their value is the floor's.

/// The model's sample rate, hop and window
pub const RATE: u32 = 16000;
pub const HOP: usize = 160;
pub const N_FFT: usize = 400;
/// a window: 30 s, 3000 frames (1500 encoder positions)
pub const CHUNK_SAMPLES: usize = 480_000;
pub const CHUNK_FRAMES: usize = 3000;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Mono samples at `rate` to `to`: a windowed sinc (Hann, 32 zero crossings, cut at the lower Nyquist), one table of
/// taps a phase when the ratio is a small fraction (every common rate to 16 kHz), else each sample's taps computed
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() || rate == 0 {
        return x.to_vec();
    }
    let g = gcd(rate as u64, to as u64);
    let (l, m) = ((to as u64 / g) as usize, (rate as u64 / g) as usize);
    let cut = (to as f64 / rate as f64).min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = ((x.len() as u128 * l as u128) / m as u128) as usize;
    let tap = |d: f64| -> f64 {
        // d: the distance from the output's instant, in input samples, scaled by the cut
        let d = d * cut;
        if d.abs() >= zc {
            return 0.0;
        }
        let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
        sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut
    };
    let width = (2 * half + 1) as usize;
    // phase p: the output instant base + p / l; tap k at input base + k - half
    let table: Option<Vec<f32>> = (l <= 4096).then(|| {
        let mut t = vec![0f32; l * width];
        for p in 0..l {
            for k in 0..width {
                t[p * width + k] = tap((k as isize - half) as f64 - p as f64 / l as f64) as f32;
            }
        }
        t
    });
    let mut out = vec![0f32; n];
    let threads = std::thread::available_parallelism().map_or(4, |c| c.get()).clamp(1, 16);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per).enumerate() {
            let table = table.as_deref();
            s.spawn(move || {
                for (j, o) in chunk.iter_mut().enumerate() {
                    let i = ci * per + j;
                    let num = i as u128 * m as u128;
                    let base = (num / l as u128) as isize;
                    let p = (num % l as u128) as usize;
                    let mut acc = 0f32;
                    for k in 0..width {
                        let at = base + k as isize - half;
                        if at < 0 || at as usize >= x.len() {
                            continue;
                        }
                        let w = match table {
                            Some(t) => t[p * width + k],
                            None => tap((k as isize - half) as f64 - p as f64 / l as f64) as f32,
                        };
                        acc += x[at as usize] * w;
                    }
                    *o = acc;
                }
            });
        }
    });
    out
}

fn hz_to_mel(f: f64) -> f64 {
    // librosa's Slaney scale: linear below 1 kHz, logarithmic above
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// librosa.filters.mel(sr=16000, n_fft=400, n_mels) - Slaney scale and area norm, 0 to 8 kHz: [n_mels, 201]
pub fn mel_filters(n_mels: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let fft: Vec<f64> = (0..bins).map(|i| i as f64 * (RATE as f64 / 2.0) / (bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(RATE as f64 / 2.0));
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut w = vec![0f32; n_mels * bins];
    for i in 0..n_mels {
        let (d0, d1) = (pts[i + 1] - pts[i], pts[i + 2] - pts[i + 1]);
        let enorm = 2.0 / (pts[i + 2] - pts[i]);
        for (j, f) in fft.iter().enumerate() {
            let lower = (f - pts[i]) / d0;
            let upper = (pts[i + 2] - f) / d1;
            w[i * bins + j] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    w
}

/// The log-mel spectrogram of `x` (16 kHz) padded with a window of silence: [n_mels, frames] row-major, frames
/// = (len + CHUNK_SAMPLES) / HOP; and the frames that hold the recording (len / HOP)
pub fn log_mel(x: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    let total = x.len() + CHUNK_SAMPLES;
    let frames = total / HOP;
    let content = x.len() / HOP;
    let bins = N_FFT / 2 + 1;
    let filters = mel_filters(n_mels);
    let window: Vec<f64> = (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()).collect();
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..N_FFT).map(|i| {
        let a = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
        (a.cos(), a.sin())
    }).unzip();
    // the padded signal, reflected at the start (torch.stft's center, pad_mode reflect); the silence after it reflects
    // to silence
    let pad = N_FFT / 2;
    let sample = |i: isize| -> f64 {
        let j = i - pad as isize;
        let j = if j < 0 { -j } else { j } as usize;
        if j < x.len() {
            x[j] as f64
        } else if j < total {
            0.0
        } else {
            // reflected at the end of the padded signal: back inside the silence
            0.0
        }
    };
    // frames whose window sees a sample of the recording
    let live = (x.len() + pad).div_ceil(HOP).min(frames);
    let mut mel = vec![f32::NAN; n_mels * frames];
    let threads = std::thread::available_parallelism().map_or(4, |c| c.get()).clamp(1, 16);
    let per = live.div_ceil(threads).max(1);
    let cols: Vec<Vec<f32>> = std::thread::scope(|s| {
        let jobs: Vec<_> = (0..live).step_by(per).map(|f0| {
            let (filters, window, cos, sin, sample) = (&filters, &window, &cos, &sin, &sample);
            s.spawn(move || {
                let f1 = (f0 + per).min(live);
                let mut out = vec![0f32; (f1 - f0) * n_mels];
                let mut frame = vec![0f64; N_FFT];
                let mut power = vec![0f64; bins];
                for f in f0..f1 {
                    for (i, v) in frame.iter_mut().enumerate() {
                        *v = sample((f * HOP + i) as isize) * window[i];
                    }
                    for (k, p) in power.iter_mut().enumerate() {
                        let (mut re, mut im) = (0f64, 0f64);
                        let mut idx = 0usize;
                        for v in frame.iter() {
                            re += v * cos[idx];
                            im -= v * sin[idx];
                            idx += k;
                            if idx >= N_FFT {
                                idx -= N_FFT;
                            }
                        }
                        *p = re * re + im * im;
                    }
                    for b in 0..n_mels {
                        let row = &filters[b * bins..(b + 1) * bins];
                        let e: f64 = row.iter().zip(power.iter()).map(|(w, p)| *w as f64 * p).sum();
                        out[(f - f0) * n_mels + b] = e.max(1e-10).log10() as f32;
                    }
                }
                out
            })
        }).collect();
        jobs.into_iter().map(|j| j.join().expect("a mel thread")).collect()
    });
    for (ci, c) in cols.iter().enumerate() {
        let f0 = ci * per;
        for (r, row) in c.chunks_exact(n_mels).enumerate() {
            for (b, v) in row.iter().enumerate() {
                mel[b * frames + f0 + r] = *v;
            }
        }
    }
    // the silent frames: log10(1e-10)
    for v in mel.iter_mut().filter(|v| v.is_nan()) {
        *v = -10.0;
    }
    let max = mel.iter().copied().fold(f32::MIN, f32::max);
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    (mel, content)
}

/// `count` frames from `at` of a spectrogram [n_mels, frames], zeros after them (transcribe's pad_or_trim of
/// `mel[:, seek:seek + segment_size]`: a window past the recording's end is padded with zeros, not with the silence's
/// spectrogram): [n_mels, 3000]
pub fn window(mel: &[f32], n_mels: usize, frames: usize, at: usize, count: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_mels * CHUNK_FRAMES];
    let n = frames.saturating_sub(at).min(CHUNK_FRAMES).min(count);
    for b in 0..n_mels {
        out[b * CHUNK_FRAMES..b * CHUNK_FRAMES + n].copy_from_slice(&mel[b * frames + at..b * frames + at + n]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mel_filters_are_librosas() {
        // librosa 0.11's filters.mel(sr=16000, n_fft=400, n_mels=128) (openai-whisper's mel_filters.npz)
        let w = mel_filters(128);
        let bins = 201;
        assert!((w[1] - 0.012_373_987).abs() < 1e-8, "{}", w[1]);
        assert_eq!(w[bins + 2], 0.0);
        let last = &w[127 * bins..];
        let (at, max) = last.iter().enumerate().fold((0, 0f32), |m, (i, v)| if *v > m.1 { (i, *v) } else { m });
        assert_eq!(at, 195);
        assert!((max - 0.005_041_601_6).abs() < 1e-9, "{max}");
        let sums: Vec<f32> = (0..3).map(|i| w[i * bins..(i + 1) * bins].iter().sum()).collect();
        for (g, want) in sums.iter().zip([0.012_373_99f32, 0.030_392_56, 0.024_747_97]) {
            assert!((g - want).abs() < 1e-7, "{g} {want}");
        }
    }

    #[test]
    fn the_log_mel_is_openai_whispers() {
        // whisper.audio.log_mel_spectrogram(x, n_mels=128, padding=480000) (openai-whisper 20250625, torch float32) of
        // three tones over 2 s: (band, frame, value)
        let x: Vec<f32> = (0..32000).map(|i| {
            let t = i as f64 / 16000.0;
            (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() + 0.2 * (2.0 * std::f64::consts::PI * 1234.5 * t + 0.3).sin()
                + 0.05 * (2.0 * std::f64::consts::PI * 5000.0 * t).sin()) as f32
        }).collect();
        let (mel, content) = log_mel(&x, 128);
        let frames = mel.len() / 128;
        assert_eq!((frames, content), (3200, 200));
        let want = [(0, 0, 0.841_423_15f32), (10, 5, -0.625_567_56), (20, 100, 1.176_473_9), (40, 150, -0.385_326_5), (60, 199, 0.052_731_63),
                    (100, 10, -0.625_567_56), (127, 50, -0.625_567_56), (5, 200, 0.712_192_9), (5, 201, 0.193_128_53), (30, 250, -0.625_567_56),
                    (64, 3000, -0.625_567_56)];
        for (b, f, v) in want {
            let g = mel[b * frames + f];
            assert!((g - v).abs() < 2e-4, "band {b} frame {f}: {g}, openai-whisper {v}");
        }
        let max = mel.iter().copied().fold(f32::MIN, f32::max);
        assert!((max - 1.374_432_4).abs() < 1e-4, "{max}");
    }

    /// The first window of real recordings against openai-whisper's (NS_TEST_WHISPER_REF: reference/whisper/ref.py's
    /// audio16k.npy and mel.npy)
    #[test]
    fn real_recordings_mel_is_openais() {
        let Ok(dir) = std::env::var("NS_TEST_WHISPER_REF") else { return };
        for clip in ["short", "long", "song"] {
            let d = std::path::Path::new(&dir).join(clip);
            let a: Vec<f32> = crate::check::read_npy(&d.join("audio16k.npy")).unwrap().1.iter().map(|x| *x as f32).collect();
            let want: Vec<f32> = crate::check::read_npy(&d.join("mel.npy")).unwrap().1.iter().map(|x| *x as f32).collect();
            let (spec, content) = log_mel(&a, 128);
            let w = window(&spec, 128, spec.len() / 128, 0, content.min(CHUNK_FRAMES));
            let (rel, mx, _) = crate::check::compare(&w, &want);
            assert!(rel < 1e-5 && mx < 1e-3, "{clip}: rel {rel:.2e} max {mx:.2e}");
        }
    }

    #[test]
    fn a_tone_lights_its_mel_band() {
        // 1 kHz for 1 s: the band holding 1 kHz is the brightest at every frame of the tone, and the values sit in
        // Whisper's range ((log10 + 4) / 4, floored 8 below the max)
        let x: Vec<f32> = (0..16000).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin() * 0.5).collect();
        let (mel, content) = log_mel(&x, 80);
        let frames = mel.len() / 80;
        assert_eq!((frames, content), (3100, 100));
        let peak = (0..80).max_by(|a, b| mel[a * frames + 50].partial_cmp(&mel[b * frames + 50]).unwrap()).unwrap();
        let want = (0..80).find(|b| {
            let f = mel_filters(80);
            let row = &f[b * 201..(b + 1) * 201];
            row.iter().enumerate().max_by(|a, c| a.1.partial_cmp(c.1).unwrap()).unwrap().0 == 25
        });
        assert!(want.is_some_and(|w| (peak as isize - w as isize).abs() <= 1), "peak band {peak}, 1 kHz's {want:?}");
        let max = mel.iter().copied().fold(f32::MIN, f32::max);
        let min = mel.iter().copied().fold(f32::MAX, f32::min);
        assert!((max - min - 2.0).abs() < 1e-5, "floored 8 below the max: {min} {max}");
        // the silence after it is the floor
        assert_eq!(mel[2 * frames + 2000], min);
    }

    #[test]
    fn resampling_keeps_a_tone() {
        let rate = 44100;
        let x: Vec<f32> = (0..rate).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin()).collect();
        let y = resample(&x, rate as u32, 16000);
        assert_eq!(y.len(), 16000);
        // against the tone at 16 kHz, away from the edges
        let err = (1000..15000).map(|i| (y[i] - (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()).abs()).fold(0f32, f32::max);
        assert!(err < 2e-3, "{err}");
        // 48 kHz: a phase table of one
        let x: Vec<f32> = (0..48000).map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 48000.0).sin()).collect();
        let y = resample(&x, 48000, 16000);
        let err = (1000..15000).map(|i| (y[i] - (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16000.0).sin()).abs()).fold(0f32, f32::max);
        assert!(err < 2e-3, "{err}");
    }
}
