//! The FLUX.2 VAE's encoder (diffusers' AutoencoderKLFlux2, black-forest-labs/FLUX.2-dev `vae/`) as Idol's ImageEncoder
//! (Budget3D's deterministic variant: the posterior's mean) runs it: the 1024 x 1024 picture in [-1, 1] to 32 latent
//! channels at 128 x 128, 2 x 2 patchified to 128 channels at 64 x 64, normalized by the VAE's batch-norm statistics,
//! flattened to 4096 tokens behind 5 zero rows (Idol's padding to DINOv3's 4101). Channels-last half on the GPU.
//!
//! ```text
//!   conv_in (3x3, 128)
//!   down: 4 blocks of 2 resnets (128, 256, 512, 512), a stride-2 3x3 conv (padded right and bottom) after the first 3
//!   mid:  resnet, attention (group norm, one head of 512 over every pixel), resnet
//!   group norm, SiLU, conv_out (3x3, 64), quant_conv (1x1): the first 32 channels are the mean
//! resnet: x + conv2(silu(norm2(conv1(silu(norm1(x)))))), a 1x1 shortcut conv when the width changes
//! ```
//! Group norms: 32 groups, eps 1e-6.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi::K;
use crate::w::{self, St};

const EPS: f32 = 1e-6;
const GROUPS: usize = 32;
const BN_EPS: f32 = 1e-4;
pub const Z: usize = 32;
/// the tokens: 5 zero rows then 64 x 64 patches of 128
pub const TOKENS: usize = 4101;
pub const TOK_C: usize = 128;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Norm {
    gamma: DevBuf,
    beta: DevBuf,
}

struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

pub struct VaeEncoder {
    gpu: Arc<Gpu>,
    conv_in: Conv,
    down: Vec<(Vec<Resnet>, Option<Conv>)>,
    mid: [Resnet; 2],
    att_norm: Norm,
    /// q, k, v, out: half [c, c]; float32 biases
    att_w: Vec<DevBuf>,
    att_b: Vec<DevBuf>,
    norm_out: Norm,
    conv_out: Conv,
    quant: Conv,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
}

impl VaeEncoder {
    pub fn load(f: &St, gpu: &Arc<Gpu>) -> Result<VaeEncoder> {
        if !f.has("encoder.conv_in.weight") || !f.has("bn.running_mean") || !f.has("quant_conv.weight") {
            return Err(Error(format!("{}: not the FLUX.2 VAE (diffusers AutoencoderKLFlux2: encoder, quant_conv, bn)", f.path())));
        }
        let conv = |p: &str| -> Result<Conv> {
            let s = f.shape(&format!("{p}.weight"))?;
            if s.len() != 4 || s[2] != s[3] {
                return Err(Error(format!("{p}: a square 2-D convolution expected, {s:?}")));
            }
            Ok(Conv { w: w::dev_half(gpu, &f.vec(&format!("{p}.weight"))?)?, b: DevBuf::from_f32(gpu, &f.vec_n(&format!("{p}.bias"), s[0])?)?,
                      ci: s[1], co: s[0], k: s[2] })
        };
        let norm = |p: &str| -> Result<Norm> {
            Ok(Norm { gamma: DevBuf::from_f32(gpu, &f.vec(&format!("{p}.weight"))?)?, beta: DevBuf::from_f32(gpu, &f.vec(&format!("{p}.bias"))?)? })
        };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet { norm1: norm(&format!("{p}.norm1"))?, conv1: conv(&format!("{p}.conv1"))?, norm2: norm(&format!("{p}.norm2"))?,
                        conv2: conv(&format!("{p}.conv2"))?,
                        shortcut: if f.has(&format!("{p}.conv_shortcut.weight")) { Some(conv(&format!("{p}.conv_shortcut"))?) } else { None } })
        };
        let mut down = Vec::new();
        for i in 0.. {
            let p = format!("encoder.down_blocks.{i}");
            if !f.has(&format!("{p}.resnets.0.conv1.weight")) {
                break;
            }
            let rs = (0..).map_while(|j| f.has(&format!("{p}.resnets.{j}.conv1.weight")).then(|| resnet(&format!("{p}.resnets.{j}"))))
                .collect::<Result<Vec<_>>>()?;
            let ds = if f.has(&format!("{p}.downsamplers.0.conv.weight")) { Some(conv(&format!("{p}.downsamplers.0.conv"))?) } else { None };
            down.push((rs, ds));
        }
        let a = "encoder.mid_block.attentions.0";
        let mut att_w = Vec::new();
        let mut att_b = Vec::new();
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            att_w.push(w::dev_half(gpu, &f.vec(&format!("{a}.{n}.weight"))?)?);
            att_b.push(DevBuf::from_f32(gpu, &f.vec(&format!("{a}.{n}.bias"))?)?);
        }
        let e = VaeEncoder {
            gpu: gpu.clone(),
            conv_in: conv("encoder.conv_in")?,
            down,
            mid: [resnet("encoder.mid_block.resnets.0")?, resnet("encoder.mid_block.resnets.1")?],
            att_norm: norm(&format!("{a}.group_norm"))?,
            att_w,
            att_b,
            norm_out: norm("encoder.conv_norm_out")?,
            conv_out: conv("encoder.conv_out")?,
            quant: conv("quant_conv")?,
            bn_mean: f.vec_n("bn.running_mean", 4 * Z)?,
            bn_var: f.vec_n("bn.running_var", 4 * Z)?,
        };
        if e.conv_out.co != 2 * Z || e.quant.co != 2 * Z {
            return Err(Error(format!("{}: {} latent channels out, the FLUX.2 VAE makes 2 x 32", f.path(), e.conv_out.co)));
        }
        Ok(e)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.fp().cast(), out.ptr())?;
        Ok(out)
    }

    fn norm(&self, k: &K, x: &DevBuf, px: usize, c: usize, n: &Norm, silu: bool) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        let scratch = DevBuf::f32(&self.gpu, GROUPS * 66)?;
        k.group_norm(x.ptr(), px, c, GROUPS, EPS, n.gamma.fp(), n.beta.fp(), silu, out.ptr(), scratch.fp())?;
        Ok(out)
    }

    fn resnet(&self, nsd: &Nsd, k: &K, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let ci = r.conv1.ci;
        let a = self.norm(k, &x, px, ci, &r.norm1, true)?;
        let a = self.conv(nsd, &r.conv1, &a, h, w)?;
        let b = self.norm(k, &a, px, r.conv1.co, &r.norm2, true)?;
        let b = self.conv(nsd, &r.conv2, &b, h, w)?;
        let skip = match &r.shortcut {
            Some(s) => self.conv(nsd, s, &x, h, w)?,
            None => x,
        };
        k.add(b.ptr(), Dt::F16, skip.ptr(), Dt::F16, px * r.conv2.co)?;
        Ok(b)
    }

    /// The middle's attention: x + out(attention(q, k, v of group_norm(x))), one head of every channel over every
    /// pixel, a block of query rows at a time
    fn attention(&self, nsd: &Nsd, k: &K, x: DevBuf, px: usize, c: usize) -> Result<DevBuf> {
        let g = &self.gpu;
        let n = self.norm(k, &x, px, c, &self.att_norm, false)?;
        let (q, kk, vt) = (DevBuf::new(g, px * c * 2)?, DevBuf::new(g, px * c * 2)?, DevBuf::new(g, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, self.att_w[0].ptr(), c, self.att_b[0].fp().cast(), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, self.att_w[1].ptr(), c, self.att_b[1].fp().cast(), kk.ptr(), Dt::F16)?;
        // v transposed [c, px]: the weights times the normed map (its bias added after the softmax: rows sum to 1)
        nsd.linear(self.att_w[2].ptr(), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
        let rows_max = ((256usize << 20) / (px * 4)).clamp(1, px);
        let scores = DevBuf::f32(g, rows_max * px)?;
        let probs = DevBuf::new(g, rows_max * px * 2)?;
        let o = DevBuf::new(g, px * c * 2)?;
        let scale = 1.0 / (c as f32).sqrt();
        let mut r0 = 0;
        while r0 < px {
            let rows = rows_max.min(px - r0);
            let qa: *const std::ffi::c_void = q.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(qa, Dt::F16, rows, c, kk.ptr(), px, none(), scores.ptr(), Dt::F32)?;
            k.softmax_rows(scores.fp(), rows, px, scale, probs.ptr())?;
            let oa: *mut std::ffi::c_void = o.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(probs.ptr(), Dt::F16, rows, px, vt.ptr(), c, self.att_b[2].fp().cast(), oa, Dt::F16)?;
            r0 += rows;
        }
        nsd.linear(o.ptr(), Dt::F16, px, c, self.att_w[3].ptr(), c, self.att_b[3].fp().cast(), n.ptr(), Dt::F16)?;
        k.add(x.ptr(), Dt::F16, n.ptr(), Dt::F16, px * c)?;
        Ok(x)
    }

    /// The tokens [4101, 128] of a picture's planes [3, size, size] in [0, 1]
    pub fn encode(&self, nsd: &Nsd, k: &K, x: &[f32], size: usize) -> Result<Vec<f32>> {
        let (mut h, mut w) = (size, size);
        if x.len() != 3 * size * size {
            return Err(Error(format!("the VAE: {} input values for {size} x {size}", x.len())));
        }
        let mut px = vec![0f32; size * size * 3];
        for i in 0..size * size {
            for c in 0..3 {
                px[i * 3 + c] = x[c * size * size + i] * 2.0 - 1.0;
            }
        }
        let mut cur = self.conv(nsd, &self.conv_in, &w::dev_half(&self.gpu, &px)?, h, w)?;
        for (rs, ds) in &self.down {
            for r in rs {
                cur = self.resnet(nsd, k, r, cur, h, w)?;
            }
            if let Some(d) = ds {
                let c = d.ci;
                let padded = DevBuf::new(&self.gpu, (h + 1) * (w + 1) * c * 2)?;
                k.pad_br(cur.ptr(), h, w, c, padded.ptr())?;
                let (oh, ow) = ((h + 1 - 3) / 2 + 1, (w + 1 - 3) / 2 + 1);
                let out = DevBuf::new(&self.gpu, oh * ow * d.co * 2)?;
                nsd.conv3d_ex(padded.ptr(), Dt::F16, (1, h + 1, w + 1), c, d.w.ptr(), d.co, (1, 3, 3), (1, 2, 2), d.b.fp().cast(), out.ptr())?;
                cur = out;
                (h, w) = (oh, ow);
            }
        }
        let c = self.mid[0].conv1.ci;
        cur = self.resnet(nsd, k, &self.mid[0], cur, h, w)?;
        cur = self.attention(nsd, k, cur, h * w, c)?;
        cur = self.resnet(nsd, k, &self.mid[1], cur, h, w)?;
        let n = self.norm(k, &cur, h * w, c, &self.norm_out, true)?;
        let o = self.conv(nsd, &self.conv_out, &n, h, w)?;
        let q = self.conv(nsd, &self.quant, &o, h, w)?;
        let qf = DevBuf::f32(&self.gpu, h * w * 2 * Z)?;
        k.convert(q.ptr(), Dt::F16, h * w * 2 * Z, qf.ptr(), Dt::F32)?;
        self.gpu.sync()?;
        let lat = qf.to_f32()?;
        Ok(self.tokens(&lat, h, w))
    }

    /// The posterior's mean (channels 0..32 of [h, w, 64] channels-last) patchified 2 x 2 (channel c 4 + dy 2 + dx),
    /// batch-normed, as [5 zero rows | (h / 2) (w / 2) rows of 128]
    pub fn tokens(&self, lat: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (hh, ww) = (h / 2, w / 2);
        let mut out = vec![0f32; (5 + hh * ww) * 4 * Z];
        for y in 0..hh {
            for x in 0..ww {
                let row = &mut out[(5 + y * ww + x) * 4 * Z..(6 + y * ww + x) * 4 * Z];
                for c in 0..Z {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let ch = c * 4 + dy * 2 + dx;
                            let v = lat[((2 * y + dy) * w + 2 * x + dx) * 2 * Z + c];
                            row[ch] = (v - self.bn_mean[ch]) / (self.bn_var[ch] + BN_EPS).sqrt();
                        }
                    }
                }
            }
        }
        out
    }
}
