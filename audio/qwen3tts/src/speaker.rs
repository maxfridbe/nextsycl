//! A recording's speaker embedding (the Base checkpoint's ECAPA-TDNN x-vector, qwen-tts' Qwen3TTSSpeakerEncoder): the
//! voice a clone speaks in, one row the talker's prompt takes in place of a built-in speaker.
//!
//! ```text
//!   24 kHz samples -> log mel (128 bands, 1,024-point STFT a 256-sample hop, Hann, slaney filters to 12 kHz)
//!   -> TDNN k5 to 512 -> 3 x SE-Res2Net (k3, dilations 2 / 3 / 4, scale 8) -> their outputs concatenated (1,536)
//!   -> TDNN k1 -> attentive statistics pooling (mean and deviation, 3,072) -> 1x1 to 2,048
//! ```
//!
//! A few million weights over a few hundred frames: on the host, in float32, its convolutions spread over the cores
//! ("same" reflect padding, as the reference's).

use nextsycl_audio::{Error, Result};

use crate::ops::Shards;

pub const RATE: u32 = 24000;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const MELS: usize = 128;
const FMAX: f64 = 12000.0;

/// A 1-D convolution's weights [co, ci, k] and bias
struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    co: usize,
    ci: usize,
    k: usize,
    dil: usize,
}

struct SeRes2 {
    tdnn1: Conv,
    parts: Vec<Conv>,
    tdnn2: Conv,
    se1: Conv,
    se2: Conv,
}

pub struct Speaker {
    first: Conv,
    blocks: Vec<SeRes2>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
    filters: Vec<f32>,
    pub dim: usize,
}

fn conv(f: &Shards, name: &str, dil: usize) -> Result<Conv> {
    let s = f.shape(&format!("{name}.weight"))?;
    Ok(Conv { w: f.f32(&format!("{name}.weight"))?, b: f.f32(&format!("{name}.bias"))?, co: s[0], ci: s[1], k: s[2], dil })
}

/// Reflect index into [0, n) (no edge repeat, as torch's "reflect")
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    if n == 1 {
        return 0;
    }
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * (n - 1) - i };
    }
    i as usize
}

impl Conv {
    /// x [ci, t] -> [co, t], "same" reflect padding, then ReLU when `relu`
    fn run(&self, x: &[f32], t: usize, relu: bool) -> Vec<f32> {
        let (ci, k, dil) = (self.ci, self.k, self.dil);
        let left = (dil * (k - 1)) / 2;
        // the input's taps gathered once: [ci * k, t]
        let mut cols = vec![0f32; ci * k * t];
        for c in 0..ci {
            for j in 0..k {
                let off = (j * dil) as isize - left as isize;
                let row = &mut cols[(c * k + j) * t..(c * k + j + 1) * t];
                for (i, v) in row.iter_mut().enumerate() {
                    *v = x[c * t + reflect(i as isize + off, t)];
                }
            }
        }
        let mut out = vec![0f32; self.co * t];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(self.co);
        let per = self.co.div_ceil(threads);
        std::thread::scope(|s| {
            for (ch, o) in out.chunks_mut(per * t).enumerate() {
                let cols = &cols;
                s.spawn(move || {
                    for (r, orow) in o.chunks_mut(t).enumerate() {
                        let co = ch * per + r;
                        orow.fill(self.b[co]);
                        let w = &self.w[co * ci * k..(co + 1) * ci * k];
                        for (q, wv) in w.iter().enumerate() {
                            let col = &cols[q * t..(q + 1) * t];
                            orow.iter_mut().zip(col).for_each(|(a, b)| *a += wv * b);
                        }
                        if relu {
                            orow.iter_mut().for_each(|v| *v = v.max(0.0));
                        }
                    }
                });
            }
        });
        out
    }
}

/// librosa.filters.mel (slaney scale and norm): [MELS, N_FFT / 2 + 1]
pub fn mel_filters(sr: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m };
    let bins = n_fft / 2 + 1;
    let fft: Vec<f64> = (0..bins).map(|i| i as f64 * sr / n_fft as f64).collect();
    let (m0, m1) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
    let mut w = vec![0f32; n_mels * bins];
    for i in 0..n_mels {
        let (d0, d1) = (mel_f[i + 1] - mel_f[i], mel_f[i + 2] - mel_f[i + 1]);
        let norm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (j, f) in fft.iter().enumerate() {
            let lower = (f - mel_f[i]) / d0;
            let upper = (mel_f[i + 2] - f) / d1;
            w[i * bins + j] = (lower.min(upper).max(0.0) * norm) as f32;
        }
    }
    w
}

/// In-place radix-2 FFT of `re`, `im` (a power of two long)
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (c, si) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (s + k, s + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * si, re[b] * si + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// The reference's mel_spectrogram: [MELS, frames] and the frame count
pub fn mel(y: &[f32], filters: &[f32]) -> (Vec<f32>, usize) {
    let pad = (N_FFT - HOP) / 2;
    let n = y.len();
    let padded: Vec<f64> = (0..n + 2 * pad).map(|i| y[reflect(i as isize - pad as isize, n)] as f64).collect();
    if padded.len() < N_FFT {
        return (Vec::new(), 0);
    }
    let frames = 1 + (padded.len() - N_FFT) / HOP;
    let bins = N_FFT / 2 + 1;
    let win: Vec<f64> = (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()).collect();
    let mut out = vec![0f32; MELS * frames];
    let (mut re, mut im) = (vec![0f64; N_FFT], vec![0f64; N_FFT]);
    let mut mag = vec![0f64; bins];
    for fr in 0..frames {
        for i in 0..N_FFT {
            re[i] = padded[fr * HOP + i] * win[i];
            im[i] = 0.0;
        }
        fft(&mut re, &mut im);
        for (k, m) in mag.iter_mut().enumerate() {
            *m = (re[k] * re[k] + im[k] * im[k] + 1e-9).sqrt();
        }
        for b in 0..MELS {
            let s: f64 = filters[b * bins..(b + 1) * bins].iter().zip(&mag).map(|(w, m)| *w as f64 * m).sum();
            out[b * frames + fr] = s.max(1e-5).ln() as f32;
        }
    }
    (out, frames)
}

impl Speaker {
    /// From a Base checkpoint's tensors (`speaker_encoder.`)
    pub fn load(f: &Shards) -> Result<Speaker> {
        let p = "speaker_encoder.";
        let first = conv(f, &format!("{p}blocks.0.conv"), 1)?;
        let mut blocks = Vec::new();
        for (i, dil) in [(1, 2), (2, 3), (3, 4)] {
            let b = format!("{p}blocks.{i}.");
            let parts = (0..)
                .map_while(|j| f.find(&format!("{b}res2net_block.blocks.{j}.conv.weight")).ok().map(|_| j))
                .map(|j| conv(f, &format!("{b}res2net_block.blocks.{j}.conv"), dil))
                .collect::<Result<Vec<_>>>()?;
            blocks.push(SeRes2 {
                tdnn1: conv(f, &format!("{b}tdnn1.conv"), 1)?,
                parts,
                tdnn2: conv(f, &format!("{b}tdnn2.conv"), 1)?,
                se1: conv(f, &format!("{b}se_block.conv1"), 1)?,
                se2: conv(f, &format!("{b}se_block.conv2"), 1)?,
            });
        }
        let fc = conv(f, &format!("{p}fc"), 1)?;
        Ok(Speaker {
            dim: fc.co,
            first,
            blocks,
            mfa: conv(f, &format!("{p}mfa.conv"), 1)?,
            asp_tdnn: conv(f, &format!("{p}asp.tdnn.conv"), 1)?,
            asp_conv: conv(f, &format!("{p}asp.conv"), 1)?,
            fc,
            filters: mel_filters(RATE as f64, N_FFT, MELS, 0.0, FMAX),
        })
    }

    /// 24 kHz mono samples to the speaker's embedding
    pub fn embed(&self, wav: &[f32]) -> Result<Vec<f32>> {
        let (m, t) = mel(wav, &self.filters);
        if t < 2 {
            return Err(Error("the reference recording is too short".into()));
        }
        let mut x = self.first.run(&m, t, true);
        let mut outs = Vec::new();
        for b in &self.blocks {
            let h = b.tdnn1.run(&x, t, true);
            let scale = b.parts.len() + 1;
            let w = h.len() / t / scale;
            let mut r2 = vec![0f32; h.len()];
            r2[..w * t].copy_from_slice(&h[..w * t]);
            let mut prev: Vec<f32> = Vec::new();
            for (i, c) in b.parts.iter().enumerate() {
                let mut part = h[(i + 1) * w * t..(i + 2) * w * t].to_vec();
                if i > 0 {
                    part.iter_mut().zip(&prev).for_each(|(a, b)| *a += b);
                }
                prev = c.run(&part, t, true);
                r2[(i + 1) * w * t..(i + 2) * w * t].copy_from_slice(&prev);
            }
            let mut h = b.tdnn2.run(&r2, t, true);
            // squeeze-excitation over the time mean
            let ch = h.len() / t;
            let mean: Vec<f32> = (0..ch).map(|c| h[c * t..(c + 1) * t].iter().sum::<f32>() / t as f32).collect();
            let s1 = b.se1.run(&mean, 1, true);
            let s2 = b.se2.run(&s1, 1, false);
            for c in 0..ch {
                let g = 1.0 / (1.0 + (-s2[c]).exp());
                h[c * t..(c + 1) * t].iter_mut().zip(&x[c * t..(c + 1) * t]).for_each(|(v, r)| *v = *v * g + r);
            }
            x = h;
            outs.push(x.clone());
        }
        let cat: Vec<f32> = outs.concat();
        let x = self.mfa.run(&cat, t, true);
        let ch = x.len() / t;
        // attentive statistics pooling
        let stats = |w: &dyn Fn(usize, usize) -> f32| -> (Vec<f32>, Vec<f32>) {
            let mut mean = vec![0f32; ch];
            let mut std = vec![0f32; ch];
            for c in 0..ch {
                let row = &x[c * t..(c + 1) * t];
                let m: f32 = row.iter().enumerate().map(|(i, v)| w(c, i) * v).sum();
                let v: f32 = row.iter().enumerate().map(|(i, v)| w(c, i) * (v - m) * (v - m)).sum();
                mean[c] = m;
                std[c] = v.max(1e-12).sqrt();
            }
            (mean, std)
        };
        let (mean, std) = stats(&|_, _| 1.0 / t as f32);
        let mut att_in = x.clone();
        for v in [&mean, &std] {
            for x in v {
                att_in.extend(std::iter::repeat_n(*x, t));
            }
        }
        let mut a = self.asp_tdnn.run(&att_in, t, true);
        a.iter_mut().for_each(|v| *v = v.tanh());
        let mut a = self.asp_conv.run(&a, t, false);
        for c in 0..ch {
            let row = &mut a[c * t..(c + 1) * t];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            row.iter_mut().for_each(|v| {
                *v = (*v - mx).exp();
                s += *v;
            });
            row.iter_mut().for_each(|v| *v /= s);
        }
        let (mean, std) = stats(&|c, i| a[c * t + i]);
        let pooled = [mean, std].concat();
        Ok(self.fc.run(&pooled, 1, false))
    }
}

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
    fn fft_matches_a_dft() {
        let x: Vec<f64> = (0..16).map(|i| ((i * 7 % 5) as f64) - 2.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; 16]);
        fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut r, mut i) = (0.0, 0.0);
            for (n, v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / 16.0;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9);
        }
    }

    #[test]
    fn reflect_and_filters() {
        assert_eq!((-2..7).map(|i| reflect(i, 5)).collect::<Vec<_>>(), vec![2, 1, 0, 1, 2, 3, 4, 3, 2]);
        let f = mel_filters(24000.0, 1024, 128, 0.0, 12000.0);
        assert_eq!(f.len(), 128 * 513);
        // every band has weight, none below zero
        for b in 0..128 {
            assert!(f[b * 513..(b + 1) * 513].iter().any(|v| *v > 0.0));
        }
        assert!(f.iter().all(|v| *v >= 0.0));
        // a resampled tone keeps its length ratio
        let x: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin()).collect();
        assert_eq!(resample(&x, 16000).len(), 24000);
    }
}
