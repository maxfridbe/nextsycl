//! The decoder: the FLUX.2 VAE (diffusers `AutoencoderKLFlux2`), from the DiT's latents to the picture. Channels-last
//! half on the GPU.
//!
//! ```text
//!   latents [128, h, w] -> z * sqrt(running_var + 1e-4) + running_mean -> unpatchify (2 x 2: 32 channels, 2h x 2w)
//!   post_quant_conv (1x1) -> conv_in (3x3, 384)
//!   mid:  resnet, attention (group norm, one head of 384 over every pixel), resnet
//!   up:   4 blocks of 3 resnets (384, 384, 192, 96), a nearest 2x + 3x3 conv after the first three
//!   group norm, SiLU, conv_out (3x3) -> RGB in [-1, 1]
//! resnet: x + conv2(silu(norm2(conv1(silu(norm1(x)))))), a 1x1 shortcut conv when the width changes
//! ```
//!
//! Group norms: 32 groups, eps 1e-6.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi;

const EPS: f32 = 1e-6;
const GROUPS: usize = 32;
/// the latent batch norm's epsilon (config.json batch_norm_eps)
const BN_EPS: f32 = 1e-4;
/// latent channels after unpatchify
pub const Z: usize = 32;

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

struct Att {
    norm: Norm,
    /// q, k, v, out: half [c, c] and float32 biases
    w: [DevBuf; 4],
    b: [DevBuf; 4],
}

struct UpBlock {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    /// the latents' batch norm: sqrt(var + eps) and mean, per channel of the packed 128
    scale: Vec<f32>,
    shift: Vec<f32>,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    ups: Vec<UpBlock>,
    out_norm: Norm,
    conv_out: Conv,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let k = ffi::api()?;
        let st = SafeTensors::open(file).map_err(ge)?;
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(name).map_err(ge)?) };
        // a weight as half on the card
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let v = st.f32(name).map_err(ge)?;
            let f = DevBuf::from_f32(&gpu, &v)?;
            let out = DevBuf::new(&gpu, v.len() * 2)?;
            // SAFETY: n floats in, n halves out.
            ffi::check(unsafe { (k.to_half)(gpu.raw(), f.fp(), out.ptr(), v.len() as i64) }, "to half")?;
            nsd.wait()?;
            Ok((out, st.need(name).map_err(ge)?.shape.clone()))
        };
        let conv = |p: &str| -> Result<Conv> {
            let (w, shape) = half(&format!("{p}.weight"))?;
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci: shape[1] as usize, co: shape[0] as usize, k: shape[2] as usize })
        };
        let norm = |p: &str| -> Result<Norm> { Ok(Norm { gamma: vec(&format!("{p}.weight"))?, beta: vec(&format!("{p}.bias"))? }) };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: norm(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"))?,
                norm2: norm(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                shortcut: if st.tensor(&format!("{p}.conv_shortcut.weight")).is_some() { Some(conv(&format!("{p}.conv_shortcut"))?) } else { None },
            })
        };
        let mut ups = Vec::new();
        for i in 0.. {
            let p = format!("decoder.up_blocks.{i}");
            if st.tensor(&format!("{p}.resnets.0.conv1.weight")).is_none() {
                break;
            }
            let n = (0..).take_while(|j| st.tensor(&format!("{p}.resnets.{j}.conv1.weight")).is_some()).count();
            ups.push(UpBlock {
                resnets: (0..n).map(|j| resnet(&format!("{p}.resnets.{j}"))).collect::<Result<_>>()?,
                up: if st.tensor(&format!("{p}.upsamplers.0.conv.weight")).is_some() { Some(conv(&format!("{p}.upsamplers.0.conv"))?) } else { None },
            });
        }
        if ups.is_empty() || st.tensor("bn.running_mean").is_none() {
            return Err(Error(format!("{}: not the FLUX.2 VAE (no decoder.up_blocks.0 / bn)", file.display())));
        }
        let a = "decoder.mid_block.attentions.0";
        let (mut aw, mut ab) = (Vec::new(), Vec::new());
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            aw.push(half(&format!("{a}.{n}.weight"))?.0);
            ab.push(vec(&format!("{a}.{n}.bias"))?);
        }
        let att = Att {
            norm: norm(&format!("{a}.group_norm"))?,
            w: aw.try_into().map_err(|_| Error("attention weights".into()))?,
            b: ab.try_into().map_err(|_| Error("attention biases".into()))?,
        };
        let var = st.f32("bn.running_var").map_err(ge)?;
        let vae = Vae {
            gpu: gpu.clone(),
            scale: var.iter().map(|v| (v + BN_EPS).sqrt()).collect(),
            shift: st.f32("bn.running_mean").map_err(ge)?,
            post_quant: conv("post_quant_conv")?,
            conv_in: conv("decoder.conv_in")?,
            mid: [resnet("decoder.mid_block.resnets.0")?, resnet("decoder.mid_block.resnets.1")?],
            att,
            ups,
            out_norm: norm("decoder.conv_norm_out")?,
            conv_out: conv("decoder.conv_out")?,
        };
        if vae.post_quant.ci != Z || vae.conv_out.co != 3 || vae.scale.len() != crate::dit::LATENT {
            return Err(Error(format!("{}: {} latent channels, {} out and {} batch-norm channels: the FLUX.2 VAE takes 32, makes RGB and packs 128", file.display(),
                                     vae.post_quant.ci, vae.conv_out.co, vae.scale.len())));
        }
        log(format!("moss-image vae: {} up blocks on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.fp().cast(), out.ptr())?;
        Ok(out)
    }

    /// group norm (and SiLU when asked) of x [px, c], a new buffer
    fn norm(&self, x: &DevBuf, px: usize, c: usize, n: &Norm, silu: bool) -> Result<DevBuf> {
        let k = ffi::api()?;
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        let scratch = DevBuf::f32(&self.gpu, (GROUPS * 66).max(1))?;
        // SAFETY: px x c halves in and out; the scratch is `ns_image_mo_group_norm_scratch(32)` = 32 x 66 floats.
        ffi::check(unsafe { (k.group_norm)(self.gpu.raw(), x.ptr(), px as i64, c as i64, GROUPS as i64, EPS, n.gamma.fp(), n.beta.fp(), silu as i32, out.ptr(), scratch.fp()) },
                   "group norm")?;
        self.gpu.sync()?;
        Ok(out)
    }

    /// Each intermediate is let go as soon as the next is made (after the work on it: a free does not wait for the queue)
    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a1 = self.norm(&x, px, r.conv1.ci, &r.norm1, true)?;
        let a2 = self.conv(nsd, &r.conv1, &a1, h, w)?;
        nsd.wait()?;
        drop(a1);
        let a3 = self.norm(&a2, px, r.conv1.co, &r.norm2, true)?;
        drop(a2);
        let out = self.conv(nsd, &r.conv2, &a3, h, w)?;
        nsd.wait()?;
        drop(a3);
        let short = match &r.shortcut {
            Some(c) => {
                let s = self.conv(nsd, c, &x, h, w)?;
                nsd.wait()?;
                drop(x);
                s
            }
            None => x,
        };
        nsd.gate_add(out.ptr(), Dt::F16, px, r.conv2.co, short.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(out)
    }

    /// The middle's attention: x + out(attention(q, k, v of group_norm(x))), one head of every channel over every pixel -
    /// two GEMMs and a softmax, a block of query rows at a time; v made transposed (its bias added after: each row of
    /// weights sums to 1)
    fn attention(&self, nsd: &Nsd, a: &Att, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let px = h * w;
        let n = self.norm(&x, px, c, &a.norm, false)?;
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, a.w[0].ptr(), c, a.b[0].fp().cast(), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, a.w[1].ptr(), c, a.b[1].fp().cast(), kk.ptr(), Dt::F16)?;
        nsd.linear(a.w[2].ptr(), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
        let rows_max = ((256usize << 20) / (px * 4)).clamp(1, px);
        let scores = DevBuf::f32(gpu, rows_max * px)?;
        let probs = DevBuf::new(gpu, rows_max * px * 2)?;
        let o = DevBuf::new(gpu, px * c * 2)?;
        let scale = 1.0 / (c as f32).sqrt();
        let mut r0 = 0;
        while r0 < px {
            let rows = rows_max.min(px - r0);
            let qa: *const std::ffi::c_void = q.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(qa, Dt::F16, rows, c, kk.ptr(), px, none(), scores.ptr(), Dt::F32)?;
            // SAFETY: scores holds rows x px floats, probs as many halves.
            ffi::check(unsafe { (k.softmax_rows)(gpu.raw(), scores.fp(), rows as i64, px as i64, scale, probs.ptr()) }, "softmax")?;
            let oa: *mut std::ffi::c_void = o.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(probs.ptr(), Dt::F16, rows, px, vt.ptr(), c, a.b[2].fp().cast(), oa, Dt::F16)?;
            r0 += rows;
        }
        nsd.linear(o.ptr(), Dt::F16, px, c, a.w[3].ptr(), c, a.b[3].fp().cast(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// The latent tokens (float32 [h * w, 128], row-major) de-batchnormed and unpatchified to channels-last [2h * 2w, 32]
    pub fn z_rows(&self, tokens: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0f32; 4 * h * w * Z];
        for y in 0..h {
            for x in 0..w {
                for c in 0..crate::dit::LATENT {
                    let v = tokens[(y * w + x) * crate::dit::LATENT + c] * self.scale[c] + self.shift[c];
                    // channel c = (c' * 2 + i) * 2 + j sits at pixel (2y + i, 2x + j) of channel c'
                    let (cp, i, j) = (c / 4, (c / 2) % 2, c % 2);
                    out[((2 * y + i) * 2 * w + 2 * x + j) * Z + cp] = v;
                }
            }
        }
        out
    }

    /// The picture from the latent tokens float32 [h * w, 128]: (RGB in [-1, 1] as half on the card [2h * 2w, 3] after
    /// the decoder, its width, height); the picture is 16 pixels a latent token
    pub fn decode_half(&self, nsd: &Nsd, tokens: &[f32], h: usize, w: usize) -> Result<(DevBuf, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let zl = self.z_rows(tokens, h, w);
        let (mut h, mut w) = (2 * h, 2 * w);
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: n floats in, n halves out.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, &self.att, x, h, w, self.conv_in.co)?;
        let mut x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        for b in &self.ups {
            for r in &b.resnets {
                x = self.resnet(nsd, r, x, h, w)?;
            }
            if let Some(c) = &b.up {
                let u = DevBuf::new(gpu, 4 * h * w * c.ci * 2)?;
                // SAFETY: x holds h x w x ci halves, u four times that.
                ffi::check(unsafe { (k.up2)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr()) }, "upsample")?;
                nsd.wait()?;
                drop(x);
                h *= 2;
                w *= 2;
                x = self.conv(nsd, c, &u, h, w)?;
                nsd.wait()?;
            }
        }
        let a = self.norm(&x, h * w, self.conv_out.ci, &self.out_norm, true)?;
        drop(x);
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        nsd.wait()?;
        Ok((img, w, h))
    }

    /// RGB [H, W, 3] bytes from the latent tokens float32 [h * w, 128]
    pub fn decode(&self, nsd: &Nsd, tokens: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let (img, pw, ph) = self.decode_half(nsd, tokens, h, w)?;
        let rgb = DevBuf::new(gpu, pw * ph * 3)?;
        // SAFETY: ph x pw x 3 halves in, as many bytes out.
        ffi::check(unsafe { (k.to_rgb8)(gpu.raw(), img.ptr(), ph as i64, pw as i64, rgb.ptr().cast()) }, "to rgb")?;
        nsd.wait()?;
        gpu.sync()?;
        let mut out = vec![0u8; pw * ph * 3];
        rgb.read(0, &mut out)?;
        Ok((out, pw, ph))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unpatchify_puts_each_channel_in_its_pixel() {
        // channel c = (c' * 2 + i) * 2 + j of token (y, x) is pixel (2y + i, 2x + j) of channel c'
        let (c, i, j) = (13usize, (13 / 2) % 2, 13 % 2);
        assert_eq!((c / 4, i, j), (3, 0, 1));
    }
}
