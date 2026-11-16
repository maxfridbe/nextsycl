//! The brain's ear: speech at 16 kHz to the rows the language model reads in place of its `<|audio|>` tokens, as the
//! release computes them (sophon-realtime finetune/sophon_swift: processing.py, modeling_qwen3_asr_encoder.py, the
//! projector of modeling_duplexjev.py; the same maths as duplexjev_vllm).
//!
//! ```text
//!   samples, padded with silence to a whole 0.16 s (2,560 samples)
//!   -> Whisper's log-mel (WhisperFeatureExtractor: 400-point Hann STFT every 160, centred and reflected, 128 Slaney
//!      bands, log10 floored 8 below the clip's max, (x + 4) / 4): L = n / 160 frames
//!   -> Qwen3-ASR's encoder: 100-frame chunks (the last padded with zeros), three 3 x 3 stride-2 convolutions of 480
//!      (GELU), a linear to 896 plus sinusoids, the valid frames (13 a chunk) through 18 layers of 896 (14 heads,
//!      attention within windows of 104 frames, GELU MLP of 3,584), a layer norm
//!   -> the projector: pairs of frames stacked (1,792), RMS norm, a linear to 4,096, x[:2048] * silu(x[2048:]), RMS
//!      norm, a linear to the LM's 5,120: one row a 16 mel frames (6.25 a second)
//! ```
//!
//! On the brain's first card in float32 (oneDNN's products and attention, sp.h's patches).

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Ops, Shards};

pub const RATE: u32 = 16000;
const HOP: usize = 160;
const N_FFT: usize = 400;
const MELS: usize = 128;
/// samples a projected row: 8 (the encoder's downsampling) x 2 (the stack) mel frames
pub const SAMPLES_PER_ROW: usize = HOP * 16;
const CHUNK: usize = 100;
const ROWS_A_CHUNK: usize = 13;

/// The encoder's rows for `l` mel frames (Qwen3-ASR's _get_feat_extract_output_lengths, n_window 50)
pub fn out_len(l: usize) -> usize {
    let r = l % CHUNK;
    let f = if r == 0 { 0 } else { (r - 1) / 2 + 1 };
    let tail = if r == 0 { 0 } else { ((f - 1) / 2 + 1 - 1) / 2 + 1 };
    tail + (l / CHUNK) * ROWS_A_CHUNK
}

/// Valid output frames of a chunk with `m` valid mel frames (three halvings)
fn post_cnn(m: usize) -> usize {
    let mut l = m;
    for _ in 0..3 {
        l = if l > 0 { (l - 1) / 2 + 1 } else { 0 };
    }
    l
}

/// Samples padded to a whole number of rows (processing.py's _align)
pub fn align(x: &[f32]) -> Vec<f32> {
    let n = x.len().max(2 * HOP).div_ceil(SAMPLES_PER_ROW) * SAMPLES_PER_ROW;
    let mut v = x.to_vec();
    v.resize(n, 0.0);
    v
}

/// The rows (language-model tokens) a clip of `samples` makes
pub fn rows_for(samples: usize) -> usize {
    samples.max(2 * HOP).div_ceil(SAMPLES_PER_ROW)
}

fn hz_to_mel(f: f64) -> f64 {
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

/// The Slaney mel filters (norm slaney) 0 to 8 kHz: [128, 201]
fn mel_filters() -> Vec<f64> {
    let bins = N_FFT / 2 + 1;
    let fft: Vec<f64> = (0..bins).map(|i| i as f64 * (RATE as f64 / 2.0) / (bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(RATE as f64 / 2.0));
    let pts: Vec<f64> = (0..MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64)).collect();
    let mut w = vec![0f64; MELS * bins];
    for i in 0..MELS {
        let (d0, d1) = (pts[i + 1] - pts[i], pts[i + 2] - pts[i + 1]);
        let enorm = 2.0 / (pts[i + 2] - pts[i]);
        for (j, f) in fft.iter().enumerate() {
            let lower = (f - pts[i]) / d0;
            let upper = (pts[i + 2] - f) / d1;
            w[i * bins + j] = lower.min(upper).max(0.0) * enorm;
        }
    }
    w
}

/// The log-mel features of aligned samples (WhisperFeatureExtractor, padding "longest"): [128, n / 160] row-major
pub fn log_mel(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let frames = n / HOP;
    let bins = N_FFT / 2 + 1;
    let filters = mel_filters();
    let window: Vec<f64> = (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()).collect();
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..N_FFT).map(|i| {
        let a = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
        (a.cos(), a.sin())
    }).unzip();
    let pad = (N_FFT / 2) as isize;
    // centred: reflected at both ends (numpy's "reflect": no edge repeat)
    let sample = |i: isize| -> f64 {
        let mut j = i - pad;
        if j < 0 {
            j = -j;
        }
        if j >= n as isize {
            j = 2 * (n as isize - 1) - j;
        }
        x[j.clamp(0, n as isize - 1) as usize] as f64
    };
    let threads = std::thread::available_parallelism().map_or(4, |c| c.get()).clamp(1, 16);
    let per = frames.div_ceil(threads).max(1);
    let mut mel = vec![0f32; MELS * frames];
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let jobs: Vec<_> = (0..frames).step_by(per).map(|f0| {
            let (filters, window, cos, sin, sample) = (&filters, &window, &cos, &sin, &sample);
            s.spawn(move || {
                let f1 = (f0 + per).min(frames);
                let mut out = vec![0f32; (f1 - f0) * MELS];
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
                    for b in 0..MELS {
                        let row = &filters[b * bins..(b + 1) * bins];
                        let e: f64 = row.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
                        out[(f - f0) * MELS + b] = e.max(1e-10).log10() as f32;
                    }
                }
                out
            })
        }).collect();
        jobs.into_iter().map(|j| j.join().expect("a mel thread")).collect()
    });
    for (ci, c) in parts.iter().enumerate() {
        let f0 = ci * per;
        for (r, row) in c.chunks_exact(MELS).enumerate() {
            for (b, v) in row.iter().enumerate() {
                mel[b * frames + f0 + r] = *v;
            }
        }
    }
    let max = mel.iter().copied().fold(f32::MIN, f32::max);
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    mel
}

struct Layer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    out: DevBuf,
    out_b: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
}

pub struct Ear {
    pub ops: Ops,
    pub nsd: Nsd,
    conv: [(DevBuf, DevBuf); 3],
    conv_out: DevBuf,
    pos: DevBuf,
    layers: Vec<Layer>,
    ln_post: (DevBuf, DevBuf),
    ln_pre: DevBuf,
    lin1: DevBuf,
    ln_mid: DevBuf,
    lin2: DevBuf,
    pub d: usize,
    heads: usize,
    ffn: usize,
    ch: usize,
    proj_hidden: usize,
    pub out: usize,
}

const A: &str = "audio_tower.";

impl Ear {
    pub fn load(f: &Shards, gpu: &Arc<Gpu>) -> Result<Ear> {
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let up = |v: &[f32]| DevBuf::from_f32(gpu, v);
        let t = |n: &str| f.f32(&format!("{A}{n}"));
        let d = f.shape(&format!("{A}layers.0.self_attn.q_proj.weight"))?[0];
        let ffn = f.shape(&format!("{A}layers.0.fc1.weight"))?[0];
        let ch = f.shape(&format!("{A}conv2d1.weight"))?[0];
        let n_layer = (0..).take_while(|l| f.has(&format!("{A}layers.{l}.fc1.weight"))).count();
        if n_layer == 0 || d % 64 != 0 {
            return Err(Error("the thinker has no audio tower (audio_tower.*)".into()));
        }
        let conv = [
            (up(&t("conv2d1.weight")?)?, up(&t("conv2d1.bias")?)?),
            (up(&t("conv2d2.weight")?)?, up(&t("conv2d2.bias")?)?),
            (up(&t("conv2d3.weight")?)?, up(&t("conv2d3.bias")?)?),
        ];
        // conv_out reads [channel, band] (c * F + f) in the reference; the patches here are [band, channel]: its
        // columns reordered once
        let co = t("conv_out.weight")?;
        let fb = MELS.div_ceil(2).div_ceil(2).div_ceil(2);
        if co.len() != d * ch * fb {
            return Err(Error(format!("audio_tower.conv_out: {} values, {d} x {ch} x {fb} expected", co.len())));
        }
        let mut cw = vec![0f32; co.len()];
        for o in 0..d {
            for c in 0..ch {
                for b in 0..fb {
                    cw[o * ch * fb + b * ch + c] = co[o * ch * fb + c * fb + b];
                }
            }
        }
        // sinusoids: [sin(t * inv) | cos(t * inv)], inv = exp(-ln(10000) / (d/2 - 1) * i)
        let half = d / 2;
        let inc = (10000f64).ln() / (half - 1) as f64;
        let mut pos = vec![0f32; ROWS_A_CHUNK * d];
        for tpos in 0..ROWS_A_CHUNK {
            for i in 0..half {
                let a = tpos as f64 * (-inc * i as f64).exp();
                pos[tpos * d + i] = a.sin() as f32;
                pos[tpos * d + half + i] = a.cos() as f32;
            }
        }
        let mut layers = Vec::new();
        for l in 0..n_layer {
            let lp = |n: &str| format!("layers.{l}.{n}");
            let mut qkv = t(&lp("self_attn.q_proj.weight"))?;
            qkv.extend(t(&lp("self_attn.k_proj.weight"))?);
            qkv.extend(t(&lp("self_attn.v_proj.weight"))?);
            let mut qkv_b = t(&lp("self_attn.q_proj.bias"))?;
            // the key projection has no bias in Qwen3-ASR's encoder when the file lacks one
            qkv_b.extend(if f.has(&format!("{A}{}", lp("self_attn.k_proj.bias"))) { t(&lp("self_attn.k_proj.bias"))? } else { vec![0.0; d] });
            qkv_b.extend(t(&lp("self_attn.v_proj.bias"))?);
            layers.push(Layer {
                ln1_w: up(&t(&lp("self_attn_layer_norm.weight"))?)?,
                ln1_b: up(&t(&lp("self_attn_layer_norm.bias"))?)?,
                qkv: up(&qkv)?,
                qkv_b: up(&qkv_b)?,
                out: up(&t(&lp("self_attn.out_proj.weight"))?)?,
                out_b: up(&t(&lp("self_attn.out_proj.bias"))?)?,
                ln2_w: up(&t(&lp("final_layer_norm.weight"))?)?,
                ln2_b: up(&t(&lp("final_layer_norm.bias"))?)?,
                fc1: up(&t(&lp("fc1.weight"))?)?,
                fc1_b: up(&t(&lp("fc1.bias"))?)?,
                fc2: up(&t(&lp("fc2.weight"))?)?,
                fc2_b: up(&t(&lp("fc2.bias"))?)?,
            });
        }
        // the projector; linear_1's halves swapped so the shared SwiGLU (silu(first) * second) is x[:h] * silu(x[h:])
        let l1 = f.f32("multi_modal_projector.linear_1.weight")?;
        let s1 = f.shape("multi_modal_projector.linear_1.weight")?;
        let (ph, pin) = (s1[0], s1[1]);
        let mut l1s = l1[ph / 2 * pin..].to_vec();
        l1s.extend_from_slice(&l1[..ph / 2 * pin]);
        let s2 = f.shape("multi_modal_projector.linear_2.weight")?;
        Ok(Ear {
            conv,
            conv_out: up(&cw)?,
            pos: up(&pos)?,
            layers,
            ln_post: (up(&t("ln_post.weight")?)?, up(&t("ln_post.bias")?)?),
            ln_pre: up(&f.f32("multi_modal_projector.ln_pre.weight")?)?,
            lin1: up(&l1s)?,
            ln_mid: up(&f.f32("multi_modal_projector.ln_mid.weight")?)?,
            lin2: up(&f.f32("multi_modal_projector.linear_2.weight")?)?,
            d,
            heads: d / 64,
            ffn,
            ch,
            proj_hidden: ph,
            out: s2[0],
            ops,
            nsd,
        })
    }

    /// 16 kHz samples -> (the encoder's rows [n, d], the projected rows [tokens, out]) on the host
    pub fn hear(&self, samples: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let x = align(samples);
        let mel = log_mel(&x);
        self.encode(&mel, x.len() / HOP)
    }

    /// Mel features [128, l] -> the encoder's rows and the projected rows. Row counts are padded to powers of two (the
    /// chunks, the frames - at least 64 -, the projected rows - at least 32; the extra rows zeros, their results unused)
    /// so oneDNN makes a handful of primitives for every clip length, not one a length
    pub fn encode(&self, mel: &[f32], l: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (ops, nsd, g) = (&self.ops, &self.nsd, &self.ops.gpu);
        let nc = l.div_ceil(CHUNK);
        let ncb = nc.next_power_of_two();
        // [ncb, 128, 100, 1] channels-last; the last chunk zero-padded, the padding chunks zeros
        let mut chunks = vec![0f32; ncb * MELS * CHUNK];
        for c in 0..nc {
            for b in 0..MELS {
                for w in 0..CHUNK {
                    let t = c * CHUNK + w;
                    if t < l {
                        chunks[(c * MELS + b) * CHUNK + w] = mel[b * l + t];
                    }
                }
            }
        }
        let mut cur = DevBuf::from_f32(g, &chunks)?;
        let (mut h, mut w, mut ci) = (MELS, CHUNK, 1usize);
        for (wgt, bias) in &self.conv {
            let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
            let patches = DevBuf::f32(g, ncb * ho * wo * ci * 9)?;
            ops.im2col_s2(cur.fp(), ncb, h, w, ci, patches.fp())?;
            let out = DevBuf::f32(g, ncb * ho * wo * self.ch)?;
            nsd.linear(patches.ptr(), Dt::F32, ncb * ho * wo, ci * 9, wgt.ptr(), self.ch, bias.ptr(), out.ptr(), Dt::F32)?;
            nsd.gelu(out.ptr(), Dt::F32, ncb * ho * wo * self.ch, false)?;
            cur = out;
            (h, w, ci) = (ho, wo, self.ch);
        }
        // [ncb, F, T, C] -> [ncb, T, F, C] -> conv_out (+ sinusoids) [ncb * T, d]
        let (fb, tt) = (h, w);
        let sw = DevBuf::f32(g, ncb * fb * tt * self.ch)?;
        ops.swap12(cur.fp(), ncb, fb, tt, self.ch, sw.fp())?;
        let d = self.d;
        let all = DevBuf::f32(g, ncb * tt * d)?;
        nsd.linear(sw.ptr(), Dt::F32, ncb * tt, fb * self.ch, self.conv_out.ptr(), d, std::ptr::null(), all.ptr(), Dt::F32)?;
        ops.add_rows(all.fp(), self.pos.fp(), ncb, tt, d)?;
        // each chunk's valid frames
        let mut idx = Vec::new();
        for c in 0..nc {
            let m = (l - c * CHUNK).min(CHUNK);
            idx.extend((0..post_cnn(m)).map(|t| (c * tt + t) as i32));
        }
        let n = idx.len();
        if n != out_len(l) {
            return Err(Error(format!("the encoder kept {n} frames of {l}; {} expected", out_len(l))));
        }
        let nb = n.next_power_of_two().max(64);
        let idx_d = DevBuf::new(g, n * 4)?;
        idx_d.write(0, &idx.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
        let x = DevBuf::f32(g, nb * d)?;
        x.fill(0)?;
        ops.gather(all.fp(), idx_d.ptr().cast(), n, d, x.fp())?;
        drop((all, sw, cur));
        // the layers: attention within windows of 104 frames (the true frames only)
        let win = ROWS_A_CHUNK * 8;
        let hb = DevBuf::f32(g, nb * d)?;
        let qkv = DevBuf::f32(g, nb * 3 * d)?;
        let att = DevBuf::f32(g, nb * d)?;
        att.fill(0)?;
        let mid = DevBuf::f32(g, nb * self.ffn)?;
        for l in &self.layers {
            nsd.layer_norm(x.ptr(), Dt::F32, nb, d, l.ln1_w.ptr(), l.ln1_b.ptr(), 1e-5, hb.ptr(), Dt::F32)?;
            nsd.linear(hb.ptr(), Dt::F32, nb, d, l.qkv.ptr(), 3 * d, l.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            ops.attn_win(qkv.fp(), fp(&qkv, d), fp(&qkv, 2 * d), n, self.heads, 64, 3 * d, win, att.fp())?;
            nsd.linear(att.ptr(), Dt::F32, nb, d, l.out.ptr(), d, l.out_b.ptr(), hb.ptr(), Dt::F32)?;
            ops.add(x.fp(), hb.fp(), nb * d)?;
            nsd.layer_norm(x.ptr(), Dt::F32, nb, d, l.ln2_w.ptr(), l.ln2_b.ptr(), 1e-5, hb.ptr(), Dt::F32)?;
            nsd.linear(hb.ptr(), Dt::F32, nb, d, l.fc1.ptr(), self.ffn, l.fc1_b.ptr(), mid.ptr(), Dt::F32)?;
            nsd.gelu(mid.ptr(), Dt::F32, nb * self.ffn, false)?;
            nsd.linear(mid.ptr(), Dt::F32, nb, self.ffn, l.fc2.ptr(), d, l.fc2_b.ptr(), hb.ptr(), Dt::F32)?;
            ops.add(x.fp(), hb.fp(), nb * d)?;
        }
        // ln_post; an odd count padded with a zero row (the stack of two), the pairs padded to 32s
        let rows = n.div_ceil(2);
        let rb = rows.next_power_of_two().max(32);
        let enc = DevBuf::f32(g, rb * 2 * d)?;
        enc.fill(0)?;
        nsd.layer_norm(x.ptr(), Dt::F32, n, d, self.ln_post.0.ptr(), self.ln_post.1.ptr(), 1e-5, enc.ptr(), Dt::F32)?;
        let tower = enc.to_f32()?[..n * d].to_vec();
        // the projector over pairs of rows: [rb, 2d]
        let a = DevBuf::f32(g, rb * 2 * d)?;
        nsd.rms_norm_mod(enc.ptr(), Dt::F32, rb, 2 * d, self.ln_pre.ptr(), 1e-6, none(), none(), none(), a.ptr(), Dt::F32)?;
        let b = DevBuf::f32(g, rb * self.proj_hidden)?;
        nsd.linear(a.ptr(), Dt::F32, rb, 2 * d, self.lin1.ptr(), self.proj_hidden, std::ptr::null(), b.ptr(), Dt::F32)?;
        let c = DevBuf::f32(g, rb * self.proj_hidden / 2)?;
        nsd.swiglu(b.ptr(), Dt::F32, rb, self.proj_hidden / 2, c.ptr(), Dt::F32)?;
        nsd.rms_norm_mod(c.ptr(), Dt::F32, rb, self.proj_hidden / 2, self.ln_mid.ptr(), 1e-6, none(), none(), none(), c.ptr(), Dt::F32)?;
        let o = DevBuf::f32(g, rb * self.out)?;
        nsd.linear(c.ptr(), Dt::F32, rb, self.proj_hidden / 2, self.lin2.ptr(), self.out, std::ptr::null(), o.ptr(), Dt::F32)?;
        let tokens = l.div_ceil(16);
        let proj = o.to_f32()?[..tokens.min(rows) * self.out].to_vec();
        Ok((tower, proj))
    }

    /// Every padded shape a clip of up to `seconds` makes, made once (oneDNN's primitives), so no conversation waits
    /// for them
    pub fn warm(&self, seconds: usize) -> Result<()> {
        for s in 1..=seconds.max(1) {
            self.hear(&vec![0f32; s * RATE as usize])?;
        }
        self.ops.gpu.sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_the_references() {
        // _get_feat_extract_output_lengths(L, 50) and the tokens (ceil(L / 16)) for clips of whole rows
        assert_eq!(out_len(100), 13);
        assert_eq!(out_len(16), 2);
        assert_eq!(out_len(272), 13 * 2 + 9);
        assert_eq!(rows_for(16000), 7);
        assert_eq!(align(&[0.0; 100]).len(), 2560);
        // every whole-row clip's encoder rows pair into at least its tokens (the projector's rows past them are cut, as
        // the reference's audio_token_len mask does)
        for rows in 1..200 {
            let l = rows * 16;
            assert!(out_len(l).div_ceil(2) >= rows, "{l} frames");
        }
    }
}
