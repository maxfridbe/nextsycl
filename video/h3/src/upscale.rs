//! The latent upscaler: video latents [24, T, H, W] -> [24, T, s H, s W] before the video decoder, so a clip
//! sampled small comes out at s times the resolution without paying the denoiser's cost at that size.
//!
//! A convolution network (the LBH-123-AI "3d_conv" pack, as reference/h3x.py runs it): a 3x3x3 convolution to 512
//! channels, 12 residual blocks (GroupNorm + SiLU + 3x3x3 convolution, twice, the second norm scaled and shifted
//! by an embedding of the scale factor) with a temporal depthwise convolution after every other one, a trilinear
//! resize to the target size, 12 more blocks at that size, a norm and a 3x3x3 convolution back to 24 channels.
//! Clips longer than 32 latent frames go through in overlapping 32-frame pieces, blended.
//!
//! Volumes are channels-last [T, H, W, C] in bfloat16 on the device; the 1x1x1 convolutions are plain linears.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{host, small};
use crate::dtype::DType;
use crate::ops::{self, Linear};
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

const GROUPS: i64 = 32;
const EPS: f32 = 1e-5;
const CHUNK: usize = 32;

/// The latent statistics the upscaler normalizes with (the video decoder's; reference h3_upscaler.py).
#[allow(clippy::excessive_precision)]
const MEAN: [f32; 24] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];
#[allow(clippy::excessive_precision)]
const STD: [f32; 24] = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
];

struct Norm {
    w: Tensor,
    b: Tensor,
}

struct Conv {
    /// bfloat16 [Co, Ci, k, k, k], as stored (the library reorders it once)
    w: Tensor,
    b: Tensor,
    co: usize,
    k: usize,
}

enum Block {
    Res { in_norm: Norm, conv1: Conv, emb_w: Vec<f32>, emb_b: Vec<f32>, out_norm: Norm, conv2: Conv },
    Temporal { norm: Norm, dw_w: Tensor, dw_b: Tensor, k: usize, pw: Linear },
}

pub struct Upscaler {
    dev: Arc<Device>,
    conv_in: Conv,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: Norm,
    conv_out: Conv,
    /// the scale embedding: Linear(1, 64) -> SiLU -> Linear(64, 64)
    e0_w: Vec<f32>,
    e0_b: Vec<f32>,
    e2_w: Vec<f32>,
    e2_b: Vec<f32>,
    channels: usize,
}

/// A volume [t, h, w, c] on the device.
struct Vol {
    x: Tensor,
    t: usize,
    h: usize,
    w: usize,
    c: usize,
}

fn f32_dev(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `y = W x + b`, W [n, k] row-major
fn matvec(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    b.iter().enumerate().map(|(o, bo)| bo + w[o * k..(o + 1) * k].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
}

impl Upscaler {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<Upscaler> {
        let norm = |p: &str| -> Result<Norm> { Ok(Norm { w: small(dev, ck, &format!("{p}.weight"))?, b: small(dev, ck, &format!("{p}.bias"))? }) };
        let conv = |p: &str| -> Result<Conv> {
            let e = ck.get(&format!("{p}.weight"))?;
            if e.dtype != DType::BF16 {
                return Err(Error(format!("{p}.weight: the upscaler is expected in bfloat16")));
            }
            Ok(Conv { w: Tensor::from_bytes(dev, DType::BF16, &e.shape, &ck.read(&format!("{p}.weight"))?)?, b: small(dev, ck, &format!("{p}.bias"))?, co: e.shape[0], k: e.shape[2] })
        };
        let blocks = |side: &str| -> Result<Vec<Block>> {
            let mut out = Vec::new();
            for i in 0.. {
                let p = format!("{side}.{i}");
                if ck.entries.contains_key(&format!("{p}.in_layers.0.weight")) {
                    out.push(Block::Res {
                        in_norm: norm(&format!("{p}.in_layers.0"))?,
                        conv1: conv(&format!("{p}.in_layers.2"))?,
                        emb_w: host(ck, &format!("{p}.emb_layers.1.weight"))?,
                        emb_b: host(ck, &format!("{p}.emb_layers.1.bias"))?,
                        out_norm: norm(&format!("{p}.out_norm"))?,
                        conv2: conv(&format!("{p}.out_layers.2"))?,
                    });
                } else if ck.entries.contains_key(&format!("{p}.dwconv.weight")) {
                    let e = ck.get(&format!("{p}.dwconv.weight"))?; // [C, 1, k, 1, 1]
                    let pw = ck.get(&format!("{p}.pwconv.weight"))?; // [C, C, 1, 1, 1]
                    out.push(Block::Temporal {
                        norm: norm(&format!("{p}.norm"))?,
                        dw_w: small(dev, ck, &format!("{p}.dwconv.weight"))?,
                        dw_b: small(dev, ck, &format!("{p}.dwconv.bias"))?,
                        k: e.shape[2],
                        pw: Linear {
                            weight: Tensor::from_bytes(dev, pw.dtype, &[pw.shape[0], pw.shape[1]], &ck.read(&format!("{p}.pwconv.weight"))?)?,
                            bias: Some(small(dev, ck, &format!("{p}.pwconv.bias"))?),
                        },
                    });
                } else {
                    break;
                }
            }
            Ok(out)
        };
        let conv_in = conv("conv_in")?;
        Ok(Upscaler {
            dev: dev.clone(),
            channels: conv_in.co,
            conv_in,
            in_blocks: blocks("in_blocks")?,
            out_blocks: blocks("out_blocks")?,
            norm_out: norm("norm_out")?,
            conv_out: conv("conv_out")?,
            e0_w: host(ck, "embed.0.weight")?,
            e0_b: host(ck, "embed.0.bias")?,
            e2_w: host(ck, "embed.2.weight")?,
            e2_b: host(ck, "embed.2.bias")?,
        })
    }

    fn vol(&self, t: usize, h: usize, w: usize, c: usize) -> Result<Vol> {
        Ok(Vol { x: Tensor::new(&self.dev, DType::BF16, &[t * h * w, c])?, t, h, w, c })
    }

    fn conv(&self, v: &Vol, cv: &Conv) -> Result<Vol> {
        let y = self.vol(v.t, v.h, v.w, cv.co)?;
        let d = &self.dev;
        // SAFETY: device buffers of this device; x [t h w, c], w [co, c, k, k, k], out [t h w, co].
        let rc = unsafe {
            (d.api.conv3d)(d.ctx, v.x.buf.ptr(), DType::BF16.kernel_code()?, v.t as i64, v.h as i64, v.w as i64, v.c as i64, cv.w.buf.ptr(),
                           cv.co as i64, cv.k as i64, cv.b.buf.ptr().cast(), y.x.buf.ptr())
        };
        d.check(rc)?;
        Ok(y)
    }

    fn norm_silu(&self, v: &Vol, n: &Norm, mod_: Option<(&Tensor, &Tensor)>) -> Result<Vol> {
        let y = self.vol(v.t, v.h, v.w, v.c)?;
        let d = &self.dev;
        let (sc, sh) = mod_.map_or((std::ptr::null(), std::ptr::null()), |(a, b)| (a.buf.ptr().cast_const().cast(), b.buf.ptr().cast_const().cast()));
        // SAFETY: as in `conv`; the norm and modulation vectors are float32 [c].
        let rc = unsafe {
            (d.api.group_norm_silu)(d.ctx, v.x.buf.ptr(), DType::BF16.kernel_code()?, 1, (v.t * v.h * v.w) as i64, v.c as i64, GROUPS, n.w.buf.ptr().cast(),
                                    n.b.buf.ptr().cast(), EPS, sc, sh, y.x.buf.ptr())
        };
        d.check(rc)?;
        Ok(y)
    }

    fn block(&self, v: Vol, b: &Block, emb: &[f32]) -> Result<Vol> {
        let d = &self.dev;
        let h = match b {
            Block::Res { in_norm, conv1, emb_w, emb_b, out_norm, conv2 } => {
                let h = self.norm_silu(&v, in_norm, None)?;
                let h = self.conv(&h, conv1)?;
                let e: Vec<f32> = emb.iter().map(|x| silu(*x)).collect();
                let ss = matvec(emb_w, emb_b, &e);
                let c = h.c;
                let (scale, shift) = (f32_dev(d, &[c], &ss[..c])?, f32_dev(d, &[c], &ss[c..2 * c])?);
                let h = self.norm_silu(&h, out_norm, Some((&scale, &shift)))?;
                self.conv(&h, conv2)?
            }
            Block::Temporal { norm, dw_w, dw_b, k, pw } => {
                let h = self.norm_silu(&v, norm, None)?;
                let t = self.vol(v.t, v.h, v.w, v.c)?;
                // SAFETY: as in `conv`; the depthwise weights float32 [c, k].
                let rc = unsafe {
                    (d.api.temporal_dwconv)(d.ctx, h.x.buf.ptr(), DType::BF16.kernel_code()?, v.t as i64, (v.h * v.w) as i64, v.c as i64,
                                            dw_w.buf.ptr().cast(), *k as i64, dw_b.buf.ptr().cast(), t.x.buf.ptr())
                };
                d.check(rc)?;
                let y = self.vol(v.t, v.h, v.w, v.c)?;
                pw.forward(&t.x, &y.x)?;
                y
            }
        };
        ops::add(&h.x, &v.x)?;
        Ok(h)
    }

    /// One piece: [T, H, W, 24] channels-last (normalized) -> [T, Ho, Wo, 24] on the host.
    fn segment(&self, x: &[f32], (t, h, w): (usize, usize, usize), (to, ho, wo): (usize, usize, usize), scale: f32, tick: &mut dyn FnMut() -> Result<()>) -> Result<Vec<f32>> {
        let e = matvec(&self.e0_w, &self.e0_b, &[scale - 1.0]);
        let e: Vec<f32> = e.iter().map(|v| silu(*v)).collect();
        let emb = matvec(&self.e2_w, &self.e2_b, &e);
        let mut v = Vol { x: Tensor::from_bytes(&self.dev, DType::BF16, &[t * h * w, 24], &crate::denoiser::bf16_bytes(x))?, t, h, w, c: 24 };
        v = self.conv(&v, &self.conv_in)?;
        for b in &self.in_blocks {
            tick()?;
            v = self.block(v, b, &emb)?;
        }
        let r = self.vol(to, ho, wo, self.channels)?;
        let d = &self.dev;
        // SAFETY: as in `conv`.
        let rc = unsafe {
            (d.api.trilinear)(d.ctx, v.x.buf.ptr(), DType::BF16.kernel_code()?, v.t as i64, v.h as i64, v.w as i64, v.c as i64, to as i64, ho as i64,
                              wo as i64, r.x.buf.ptr())
        };
        d.check(rc)?;
        v = r;
        for b in &self.out_blocks {
            tick()?;
            v = self.block(v, b, &emb)?;
        }
        let v = self.norm_silu(&v, &self.norm_out, None)?;
        let v = self.conv(&v, &self.conv_out)?;
        v.x.to_f32()
    }

    /// Normalized latents [24, T, H, W] -> [24, T, Ho, Wo], Ho = round(s H), Wo = round(s W).
    pub fn upscale(&self, z: &[f32], t: usize, h: usize, w: usize, scale: f32, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize, usize)> {
        let (ho, wo) = ((h as f32 * scale).round() as usize, (w as f32 * scale).round() as usize);
        let n = h * w;
        // channels-last and normalized once more by the latent statistics (as the reference does)
        let mut x = vec![0f32; z.len()];
        for c in 0..24 {
            for i in 0..t * n {
                x[i * 24 + c] = (z[c * t * n + i] - MEAN[c]) / STD[c];
            }
        }
        let frame = n * 24;
        let out_frame = ho * wo * 24;
        let out: Vec<f32> = if t <= CHUNK {
            self.segment(&x, (t, h, w), (t, ho, wo), scale, tick)?
        } else {
            // overlapping pieces of 32 frames: each piece reads `ov` frames more on each side (replicate-padded at
            // the clip's ends), keeps `ov` frames more than its own, and the overlaps are cross-faded
            let ov = self
                .in_blocks
                .iter()
                .find_map(|b| match b {
                    Block::Temporal { k, .. } => Some(*k),
                    _ => None,
                })
                .unwrap_or(0);
            let padded = |i: isize| -> usize { (i - ov as isize).clamp(0, t as isize - 1) as usize };
            let mut acc = vec![0f32; t * out_frame];
            let mut wsum = vec![0f32; t];
            let mut start = 0;
            while start < t {
                let seg_end = (start + CHUNK).min(t);
                let out_start = start.saturating_sub(ov);
                let out_end = (seg_end + ov).min(t);
                let lo = out_start.saturating_sub(ov); // in padded coordinates
                let hi = (t + 2 * ov).min(out_end + ov);
                let mut seg = Vec::with_capacity((hi - lo) * frame);
                for i in lo..hi {
                    let f = padded(i as isize);
                    seg.extend_from_slice(&x[f * frame..(f + 1) * frame]);
                }
                let y = self.segment(&seg, (hi - lo, h, w), (hi - lo, ho, wo), scale, tick)?;
                let s0 = out_start + ov - lo;
                let nv = out_end - out_start;
                for j in 0..nv {
                    let mut wt = 1.0f32;
                    if start > out_start && j < start - out_start {
                        let bl = start - out_start;
                        wt = (j + 1) as f32 / (bl + 1) as f32;
                    }
                    if out_end > seg_end && j >= nv - (out_end - seg_end) {
                        let bl = out_end - seg_end;
                        wt = (nv - j) as f32 / (bl + 1) as f32;
                    }
                    let src = &y[(s0 + j) * out_frame..(s0 + j + 1) * out_frame];
                    let dst = &mut acc[(out_start + j) * out_frame..(out_start + j + 1) * out_frame];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += s * wt;
                    }
                    wsum[out_start + j] += wt;
                }
                start += CHUNK;
            }
            for (f, ws) in wsum.iter().enumerate() {
                let inv = 1.0 / ws.max(1e-8);
                acc[f * out_frame..(f + 1) * out_frame].iter_mut().for_each(|v| *v *= inv);
            }
            acc
        };
        // back to [24, T, Ho, Wo], un-normalized
        let no = ho * wo;
        let mut z2 = vec![0f32; 24 * t * no];
        for c in 0..24 {
            for i in 0..t * no {
                z2[c * t * no + i] = out[i * 24 + c] * STD[c] + MEAN[c];
            }
        }
        Ok((z2, ho, wo))
    }
}
