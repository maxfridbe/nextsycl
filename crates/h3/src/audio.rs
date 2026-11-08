//! The audio decoder: latents [32, 2, T] (40 frames a second) -> stereo sound at 32 kHz, 800 samples per frame.
//!
//! BigVGAN, the two stereo channels as a batch of two mono signals: a 1x1 projection to 2048 channels, a 7-tap
//! convolution to 1024, then seven upsampling stages (x5, x5, x2 ... x2; the channels halve each time), each a
//! transposed convolution followed by the average of three residual stacks (kernels 3, 7, 11, dilations 1, 3, 5)
//! whose activations are anti-aliased Snake functions; a last activation, a 7-tap convolution to one channel, and a
//! clamp to [-1, 1]. Float32 throughout, as the checkpoint is stored (ComfyUI `ldm/minimax/audio_vae.py`).

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::host;
use crate::dtype::DType;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

const UP_RATES: [usize; 7] = [5, 5, 2, 2, 2, 2, 2];
const UP_KERNELS: [usize; 7] = [9, 9, 4, 4, 4, 4, 4];
const RES_KERNELS: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];
pub const SAMPLE_RATE: u32 = 32_000;

fn upload(dev: &Arc<Device>, ck: &Checkpoint, name: &str) -> Result<Tensor> {
    let v = host(ck, name)?;
    let e = ck.get(name)?;
    Tensor::from_bytes(dev, DType::F32, &e.shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    /// [Co, Ci, K] for a convolution, [Ci, Co, K] for a transposed one
    co: usize,
    ci: usize,
    k: usize,
}

impl Conv {
    fn load(dev: &Arc<Device>, ck: &Checkpoint, p: &str, transposed: bool) -> Result<Conv> {
        let w = upload(dev, ck, &format!("{p}.weight"))?;
        let b = match ck.entries.contains_key(&format!("{p}.bias")) {
            true => Some(upload(dev, ck, &format!("{p}.bias"))?),
            false => None,
        };
        let (a, c, k) = (w.shape[0], w.shape[1], w.shape[2]);
        let (co, ci) = if transposed { (c, a) } else { (a, c) };
        Ok(Conv { w, b, co, ci, k })
    }
}

/// One anti-aliased Snake activation: its per-channel (log) alpha and beta, its two 12-tap filters.
struct Act {
    alpha: Tensor,
    beta: Tensor,
    up: Tensor,
    down: Tensor,
}

impl Act {
    fn load(dev: &Arc<Device>, ck: &Checkpoint, p: &str) -> Result<Act> {
        Ok(Act {
            alpha: upload(dev, ck, &format!("{p}.act.alpha"))?,
            beta: upload(dev, ck, &format!("{p}.act.beta"))?,
            up: upload(dev, ck, &format!("{p}.upsample.filter"))?,
            down: upload(dev, ck, &format!("{p}.downsample.lowpass.filter"))?,
        })
    }
}

struct ResStack {
    k: usize,
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    acts: Vec<Act>,
}

pub struct AudioDecoder {
    dev: Arc<Device>,
    in_proj: Conv,
    conv_pre: Conv,
    ups: Vec<Conv>,
    res: Vec<ResStack>,
    act_post: Act,
    conv_post: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// A signal [B, C, L] on the device, float32.
struct Sig {
    t: Tensor,
    b: usize,
    c: usize,
    l: usize,
}

impl AudioDecoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<AudioDecoder> {
        let mut res = Vec::new();
        for i in 0..UP_RATES.len() * RES_KERNELS.len() {
            let p = format!("decoder.resblocks.{i}");
            res.push(ResStack {
                k: RES_KERNELS[i % RES_KERNELS.len()],
                convs1: (0..3).map(|j| Conv::load(dev, ck, &format!("{p}.convs1.{j}"), false)).collect::<Result<_>>()?,
                convs2: (0..3).map(|j| Conv::load(dev, ck, &format!("{p}.convs2.{j}"), false)).collect::<Result<_>>()?,
                acts: (0..6).map(|j| Act::load(dev, ck, &format!("{p}.activations.{j}"))).collect::<Result<_>>()?,
            });
        }
        let ups: Vec<Conv> = (0..UP_RATES.len()).map(|i| Conv::load(dev, ck, &format!("decoder.ups.{i}.0"), true)).collect::<Result<_>>()?;
        if ups.iter().map(|u| u.k).ne(UP_KERNELS) {
            return Err(Error("the audio decoder's upsampling kernels are not BigVGAN's 32 kHz configuration".into()));
        }
        Ok(AudioDecoder {
            dev: dev.clone(),
            in_proj: Conv::load(dev, ck, "dec_in_proj", false)?,
            conv_pre: Conv::load(dev, ck, "decoder.conv_pre", false)?,
            ups,
            res,
            act_post: Act::load(dev, ck, "decoder.activation_post")?,
            conv_post: Conv::load(dev, ck, "decoder.conv_post", false)?,
            mean: host(ck, "latents_mean")?,
            std: host(ck, "latents_std")?,
        })
    }

    fn new_sig(&self, b: usize, c: usize, l: usize) -> Result<Sig> {
        Ok(Sig { t: Tensor::new(&self.dev, DType::F32, &[b, c, l])?, b, c, l })
    }

    fn conv(&self, x: &Sig, cv: &Conv, dil: usize, pad: usize) -> Result<Sig> {
        if x.c != cv.ci {
            return Err(Error(format!("audio conv: {} channels into a {}-channel convolution", x.c, cv.ci)));
        }
        let lo = x.l + 2 * pad - dil * (cv.k - 1);
        let y = self.new_sig(x.b, cv.co, lo)?;
        let d = &self.dev;
        // SAFETY: float32 device buffers of this device, sized as declared above.
        let rc = unsafe {
            (d.api.conv1d)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, cv.w.buf.ptr().cast(), cv.co as i64, cv.k as i64,
                           cv.b.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast_const().cast()), 1, dil as i64, pad as i64, y.t.buf.ptr().cast(), lo as i64)
        };
        d.check(rc)?;
        Ok(y)
    }

    fn conv_t(&self, x: &Sig, cv: &Conv, stride: usize) -> Result<Sig> {
        let pad = (cv.k - stride) / 2;
        let lo = (x.l - 1) * stride + cv.k - 2 * pad;
        let y = self.new_sig(x.b, cv.co, lo)?;
        let d = &self.dev;
        // SAFETY: as in `conv`.
        let rc = unsafe {
            (d.api.conv_transpose1d)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, cv.w.buf.ptr().cast(), cv.co as i64, cv.k as i64,
                                     cv.b.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast_const().cast()), stride as i64, pad as i64, y.t.buf.ptr().cast(), lo as i64)
        };
        d.check(rc)?;
        Ok(y)
    }

    fn act(&self, x: &Sig, a: &Act) -> Result<Sig> {
        let y = self.new_sig(x.b, x.c, x.l)?;
        let d = &self.dev;
        // SAFETY: as in `conv`; alpha and beta have x.c values, the filters 12.
        let rc = unsafe {
            (d.api.aa_snake)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, a.alpha.buf.ptr().cast(), a.beta.buf.ptr().cast(),
                             a.up.buf.ptr().cast(), a.down.buf.ptr().cast(), y.t.buf.ptr().cast())
        };
        d.check(rc)?;
        Ok(y)
    }

    /// One residual stack (BigVGAN's AMPBlock1) on x; x itself is left as it is.
    fn stack(&self, x: &Sig, r: &ResStack) -> Result<Sig> {
        let mut cur = self.new_sig(x.b, x.c, x.l)?;
        cur.t.copy_rows(0, &x.t, 0, x.b)?;
        for (j, d) in DILATIONS.iter().enumerate() {
            let t = self.act(&cur, &r.acts[2 * j])?;
            let t = self.conv(&t, &r.convs1[j], *d, (r.k * d - d) / 2)?;
            let t = self.act(&t, &r.acts[2 * j + 1])?;
            let t = self.conv(&t, &r.convs2[j], 1, (r.k - 1) / 2)?;
            crate::ops::add(&t.t, &cur.t)?;
            cur = t;
        }
        Ok(cur)
    }

    /// Normalized latents [32, 2, T] -> one buffer of samples per stereo channel (800 T each), in [-1, 1].
    pub fn decode(&self, z: &[f32], t: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<Vec<Vec<f32>>> {
        let c = self.mean.len();
        if z.len() != c * 2 * t {
            return Err(Error(format!("audio latents: {} values for {c} x 2 x {t}", z.len())));
        }
        // [32, 2, T] -> [2, 32, T], denormalized
        let mut zz = vec![0f32; z.len()];
        for s in 0..2 {
            for ci in 0..c {
                for ti in 0..t {
                    zz[(s * c + ci) * t + ti] = z[(ci * 2 + s) * t + ti] * self.std[ci] + self.mean[ci];
                }
            }
        }
        let x = Sig { t: Tensor::from_bytes(&self.dev, DType::F32, &[2, c, t], &zz.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?, b: 2, c, l: t };
        let x = self.conv(&x, &self.in_proj, 1, 0)?;
        let mut x = self.conv(&x, &self.conv_pre, 1, 3)?;
        let n = RES_KERNELS.len();
        for (i, up) in self.ups.iter().enumerate() {
            tick()?;
            x = self.conv_t(&x, up, UP_RATES[i])?;
            let xs = self.stack(&x, &self.res[i * n])?;
            for j in 1..n {
                let y = self.stack(&x, &self.res[i * n + j])?;
                crate::ops::add(&xs.t, &y.t)?;
            }
            // SAFETY: float32 device buffer of xs.t's size.
            let rc = unsafe { (self.dev.api.scale)(self.dev.ctx, xs.t.buf.ptr().cast(), xs.t.elements() as i64, 1.0 / n as f32) };
            self.dev.check(rc)?;
            x = xs;
        }
        let x = self.act(&x, &self.act_post)?;
        let y = self.conv(&x, &self.conv_post, 1, 3)?;
        let v = y.t.to_f32()?;
        let l = y.l;
        Ok((0..2).map(|s| v[s * l..(s + 1) * l].iter().map(|x| x.clamp(-1.0, 1.0)).collect()).collect())
    }
}

/// The audio encoder: stereo sound at 32 kHz -> normalized latents [32, 2, T] (800 samples a frame) - what turns a
/// previous clip's last second (an audio keyframe) or a voice sample (a reference) into latents.
///
/// DAC's encoder on each channel as a mono signal (a 7-tap convolution to 64 channels, five blocks of three dilated
/// residual units and a strided convolution - x2, x4, x4, x5, x5, the channels doubling to 2048 - Snake activations),
/// then a small attention head over the frames (causal, 8 heads, averaged and pooled down to 32 features), a
/// GeGLU MLP and a 1x1 projection, normalized by the latent statistics (ComfyUI `ldm/minimax/audio_vae.py`).
pub struct AudioEncoder {
    dev: Arc<Device>,
    conv_in: Conv,
    /// per block: three residual units (snake, 7-tap dilated conv, snake, 1x1 conv), a snake, the strided conv
    blocks: Vec<EncBlock>,
    snake_out: Tensor,
    conv_out: Conv,
    // the attention head (host side but for the wide projection)
    norm1: (Vec<f32>, Vec<f32>),
    norm3: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    qkv: crate::ops::Linear,
    qkv_bias: Vec<f32>,
    attn_proj: (Vec<f32>, Vec<f32>),
    proj: (Vec<f32>, Vec<f32>),
    mlp_norm: (Vec<f32>, Vec<f32>),
    w0: (Vec<f32>, Vec<f32>),
    w1: (Vec<f32>, Vec<f32>),
    w2: (Vec<f32>, Vec<f32>),
    mean_proj: (Vec<f32>, Vec<f32>),
    mean: Vec<f32>,
    std: Vec<f32>,
}

struct EncUnit {
    a1: Tensor,
    c1: Conv,
    a2: Tensor,
    c2: Conv,
    dilation: usize,
}

struct EncBlock {
    units: Vec<EncUnit>,
    snake: Tensor,
    down: Conv,
    stride: usize,
}

const ENC_STRIDES: [usize; 5] = [2, 4, 4, 5, 5];
const ENC_DILATIONS: [usize; 3] = [1, 3, 9];
const HOP: usize = 800;
const HEADS: usize = 8;

fn layer_norm_rows(x: &[f32], c: usize, (w, b): &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for r in x.chunks(c) {
        let mean = r.iter().sum::<f32>() / c as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        out.extend(r.iter().enumerate().map(|(i, v)| (v - mean) * inv * w[i] + b[i]));
    }
    out
}

/// rows of x [n, k] times W^T [o, k] plus b
fn linear_rows(x: &[f32], k: usize, (w, b): &(Vec<f32>, Vec<f32>)) -> Vec<f32> {
    let o = b.len();
    let mut out = Vec::with_capacity(x.len() / k * o);
    for r in x.chunks(k) {
        out.extend((0..o).map(|j| b[j] + w[j * k..(j + 1) * k].iter().zip(r).map(|(a, c)| a * c).sum::<f32>()));
    }
    out
}

impl AudioEncoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<AudioEncoder> {
        let hv = |n: &str| host(ck, n);
        let pair = |p: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((hv(&format!("{p}.weight"))?, hv(&format!("{p}.bias"))?)) };
        let mut blocks = Vec::new();
        for (bi, stride) in ENC_STRIDES.iter().enumerate() {
            let p = format!("encoder.block.{}", bi + 1);
            let mut units = Vec::new();
            for (ui, d) in ENC_DILATIONS.iter().enumerate() {
                let q = format!("{p}.block.{ui}.block");
                units.push(EncUnit {
                    a1: upload(dev, ck, &format!("{q}.0.alpha"))?,
                    c1: Conv::load(dev, ck, &format!("{q}.1"), false)?,
                    a2: upload(dev, ck, &format!("{q}.2.alpha"))?,
                    c2: Conv::load(dev, ck, &format!("{q}.3"), false)?,
                    dilation: *d,
                });
            }
            blocks.push(EncBlock { units, snake: upload(dev, ck, &format!("{p}.block.3.alpha"))?, down: Conv::load(dev, ck, &format!("{p}.block.4"), false)?, stride: *stride });
        }
        let qkv_w = ck.get("pre_block.attn.qkv.weight")?.shape.clone();
        let mut qkv_bias = hv("pre_block.attn.q_bias")?;
        qkv_bias.extend(hv("pre_block.attn.zero_k_bias")?);
        qkv_bias.extend(hv("pre_block.attn.v_bias")?);
        Ok(AudioEncoder {
            dev: dev.clone(),
            conv_in: Conv::load(dev, ck, "encoder.block.0", false)?,
            blocks,
            snake_out: upload(dev, ck, "encoder.block.6.alpha")?,
            conv_out: Conv::load(dev, ck, "encoder.block.7", false)?,
            norm1: pair("pre_block.norm1")?,
            norm3: pair("pre_block.norm3")?,
            norm2: pair("pre_block.norm2")?,
            qkv: crate::ops::Linear { weight: Tensor::from_bytes(dev, DType::F32, &qkv_w, &hv("pre_block.attn.qkv.weight")?.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?, bias: None },
            qkv_bias,
            attn_proj: pair("pre_block.attn.proj")?,
            proj: pair("pre_block.proj")?,
            mlp_norm: pair("pre_block.mlp.norm")?,
            w0: pair("pre_block.mlp.w0")?,
            w1: pair("pre_block.mlp.w1")?,
            w2: pair("pre_block.mlp.w2")?,
            mean_proj: pair("mean_proj")?,
            mean: hv("latents_mean")?,
            std: hv("latents_std")?,
        })
    }

    fn snake(&self, x: &Sig, alpha: &Tensor) -> Result<Sig> {
        let y = Sig { t: Tensor::new(&self.dev, DType::F32, &[x.b, x.c, x.l])?, b: x.b, c: x.c, l: x.l };
        let d = &self.dev;
        // SAFETY: float32 device buffers; alpha has x.c values.
        let rc = unsafe { (d.api.snake)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, alpha.buf.ptr().cast(), y.t.buf.ptr().cast()) };
        d.check(rc)?;
        Ok(y)
    }

    fn conv(&self, x: &Sig, cv: &Conv, stride: usize, dil: usize, pad: usize) -> Result<Sig> {
        let lo = (x.l + 2 * pad - dil * (cv.k - 1) - 1) / stride + 1;
        let y = Sig { t: Tensor::new(&self.dev, DType::F32, &[x.b, cv.co, lo])?, b: x.b, c: cv.co, l: lo };
        let d = &self.dev;
        // SAFETY: float32 device buffers of this device, sized as declared.
        let rc = unsafe {
            (d.api.conv1d)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, cv.w.buf.ptr().cast(), cv.co as i64, cv.k as i64,
                           cv.b.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast_const().cast()), stride as i64, dil as i64, pad as i64, y.t.buf.ptr().cast(), lo as i64)
        };
        d.check(rc)?;
        Ok(y)
    }

    /// One buffer of samples per stereo channel (32 kHz, in [-1, 1]) -> normalized latents [32, 2, T] and T.
    pub fn encode(&self, channels: &[Vec<f32>]) -> Result<(Vec<f32>, usize)> {
        if channels.len() != 2 {
            return Err(Error("the audio encoder takes stereo sound".into()));
        }
        let len = channels[0].len().div_ceil(HOP) * HOP;
        // right-padded with zeros to whole frames
        let mut wav = vec![0f32; 2 * len];
        for (s, ch) in channels.iter().enumerate() {
            wav[s * len..s * len + ch.len()].copy_from_slice(ch);
        }
        let mut x = Sig { t: Tensor::from_bytes(&self.dev, DType::F32, &[2, 1, len], &wav.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?, b: 2, c: 1, l: len };
        x = self.conv(&x, &self.conv_in, 1, 1, 3)?;
        for b in &self.blocks {
            for u in &b.units {
                let y = self.snake(&x, &u.a1)?;
                let y = self.conv(&y, &u.c1, 1, u.dilation, 3 * u.dilation)?;
                let y = self.snake(&y, &u.a2)?;
                let y = self.conv(&y, &u.c2, 1, 1, 0)?;
                crate::ops::add(&y.t, &x.t)?;
                x = y;
            }
            let y = self.snake(&x, &b.snake)?;
            x = self.conv(&y, &b.down, b.stride, 1, b.stride.div_ceil(2))?;
        }
        let y = self.snake(&x, &self.snake_out)?;
        let x = self.conv(&y, &self.conv_out, 1, 1, 1)?; // [2, 2048, T]
        let (c, t) = (x.c, x.l);
        let enc = x.t.to_f32()?;
        // the attention head, per stereo channel: rows [T, 2048]
        let mut z = vec![0f32; 32 * 2 * t];
        for s in 0..2 {
            let rows: Vec<f32> = (0..t).flat_map(|ti| (0..c).map(move |ci| (ti, ci))).map(|(ti, ci)| enc[(s * c + ci) * t + ti]).collect();
            let n1 = layer_norm_rows(&rows, c, &self.norm1, 1e-5);
            let n3 = layer_norm_rows(&rows, c, &self.norm3, 1e-5);
            // q, k, v on the card (the one wide product)
            let inp = Tensor::from_bytes(&self.dev, DType::F32, &[t, c], &n1.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
            let out = Tensor::new(&self.dev, DType::F32, &[t, 3 * c])?;
            self.qkv.forward(&inp, &out)?;
            let mut qkv = out.to_f32()?;
            for r in qkv.chunks_mut(3 * c) {
                r.iter_mut().zip(&self.qkv_bias).for_each(|(v, b)| *v += b);
            }
            let hd = c / HEADS;
            let scale = 1.0 / (hd as f32).sqrt();
            // causal attention per head, then the mean over heads [T, hd]
            let mut mean_h = vec![0f32; t * hd];
            for h in 0..HEADS {
                for i in 0..t {
                    let q = &qkv[i * 3 * c + h * hd..][..hd];
                    let scores: Vec<f32> = (0..=i).map(|j| q.iter().zip(&qkv[j * 3 * c + c + h * hd..][..hd]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                    let sum: f32 = e.iter().sum();
                    for (j, ej) in e.iter().enumerate() {
                        let v = &qkv[j * 3 * c + 2 * c + h * hd..][..hd];
                        for d in 0..hd {
                            mean_h[i * hd + d] += ej / sum * v[d] / HEADS as f32;
                        }
                    }
                }
            }
            // pooled down to 32 features (average of each run of hd / 32), the attention's projection
            let k = hd / 32;
            let pooled: Vec<f32> = (0..t).flat_map(|i| (0..32).map(move |j| (i, j))).map(|(i, j)| mean_h[i * hd + j * k..i * hd + (j + 1) * k].iter().sum::<f32>() / k as f32).collect();
            let a = linear_rows(&pooled, 32, &self.attn_proj);
            let p = linear_rows(&n3, c, &self.proj);
            let x2: Vec<f32> = p.iter().zip(&a).map(|(a, b)| a + b).collect();
            // the GeGLU MLP (its own norm after the head's)
            let m = layer_norm_rows(&layer_norm_rows(&x2, 32, &self.norm2, 1e-5), 32, &self.mlp_norm, 1e-5);
            let g0 = linear_rows(&m, 32, &self.w0);
            let g1 = linear_rows(&m, 32, &self.w1);
            let gelu = |v: f32| 0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh());
            let hmid: Vec<f32> = g0.iter().zip(&g1).map(|(a, b)| gelu(*a) * b).collect();
            let mo = linear_rows(&hmid, 64, &self.w2);
            let x3: Vec<f32> = x2.iter().zip(&mo).map(|(a, b)| a + b).collect();
            let zz = linear_rows(&x3, 32, &self.mean_proj);
            for ti in 0..t {
                for ci in 0..32 {
                    z[(ci * 2 + s) * t + ti] = (zz[ti * 32 + ci] - self.mean[ci]) / self.std[ci];
                }
            }
        }
        Ok((z, t))
    }
}
