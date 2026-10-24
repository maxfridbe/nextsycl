//! The decoder: Ming-Image's VAE (diffusers' AutoencoderKLQwenImage with 4 input channels: Wan 2.1's decoder making
//! RGBA), from latents to the picture. Everything is channels-last half on the GPU; one picture is one frame, so the
//! causal 3-D convolutions are their last time slice (the frames before are zeros) and the time convolutions of the
//! upsamplers are skipped (the first chunk's), as the reference decodes a single frame.
//!
//! ```text
//!   z [h, w, 16] = latents / 8.0064 -> 1x1 conv -> 3x3 conv (384)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   up: 3 resnets (384), up 2x + conv (192); 3 resnets (384), up 2x + conv (192); 3 resnets (192), up 2x + conv (96);
//!       3 resnets (96)
//!   rms norm, silu, 3x3 conv -> 4 channels (RGBA) in [-1, 1]
//! resnet: x + conv(silu(norm(conv(silu(norm(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! The file names its tensors as Wan does (decoder.middle.0.residual.2, decoder.upsamples.3.resample.1, ...).

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi;

/// F.normalize's floor, as an epsilon on the mean square: never reached by real activations
const EPS: f32 = 1e-24;
/// diffusers' vae/config.json scaling_factor (shift_factor 0)
pub const SCALE: f32 = 8.0064;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Resnet {
    norm1: DevBuf,
    conv1: Conv,
    norm2: DevBuf,
    conv2: Conv,
    shortcut: Option<Conv>,
}

enum Up {
    Res(Resnet),
    /// nearest 2x, then the conv
    Resample(Conv),
}

struct Att {
    norm: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    ups: Vec<Up>,
    out_norm: DevBuf,
    conv_out: Conv,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        // a weight as half on the card; a 3-D kernel [co, ci, kt, kh, kw] as its last time slice [co, ci, kh, kw]
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            if t.dtype != "BF16" {
                return Err(Error(format!("{name}: {} weights are not read yet", t.dtype)));
            }
            let mut raw = st.read(t).map_err(ge)?;
            let mut shape = t.shape.clone();
            if shape.len() == 5 {
                let (co, ci, kt, kh, kw) = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize, shape[4] as usize);
                let plane = kh * kw * 2;
                let mut out = Vec::with_capacity(co * ci * plane);
                for i in 0..co * ci {
                    let at = (i * kt + kt - 1) * plane;
                    out.extend_from_slice(&raw[at..at + plane]);
                }
                raw = out;
                shape = vec![shape[0], shape[1], shape[3], shape[4]];
            }
            let n = raw.len() / 2;
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * 2)?;
            nsd.dequant(src.ptr(), 30, n, out.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok((out, shape))
        };
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(name).map_err(ge)?) };
        let conv = |p: &str| -> Result<Conv> {
            let (w, shape) = half(&format!("{p}.weight"))?;
            let (co, ci, k) = (shape[0] as usize, shape[1] as usize, *shape.last().unwrap_or(&1) as usize);
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci, co, k })
        };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: vec(&format!("{p}.residual.0.gamma"))?,
                conv1: conv(&format!("{p}.residual.2"))?,
                norm2: vec(&format!("{p}.residual.3.gamma"))?,
                conv2: conv(&format!("{p}.residual.6"))?,
                shortcut: if st.tensor(&format!("{p}.shortcut.weight")).is_some() { Some(conv(&format!("{p}.shortcut"))?) } else { None },
            })
        };
        let mut ups = Vec::new();
        for i in 0.. {
            let p = format!("decoder.upsamples.{i}");
            if st.tensor(&format!("{p}.residual.0.gamma")).is_some() {
                ups.push(Up::Res(resnet(&p)?));
            } else if st.tensor(&format!("{p}.resample.1.weight")).is_some() {
                ups.push(Up::Resample(conv(&format!("{p}.resample.1"))?));
            } else {
                break;
            }
        }
        if ups.is_empty() {
            return Err(Error(format!("{}: not a Wan 2.1 decoder (no decoder.upsamples.0)", file.display())));
        }
        let att = Att {
            norm: vec("decoder.middle.1.norm.gamma")?,
            qkv: half("decoder.middle.1.to_qkv.weight")?.0,
            qkv_b: vec("decoder.middle.1.to_qkv.bias")?,
            proj: half("decoder.middle.1.proj.weight")?.0,
            proj_b: vec("decoder.middle.1.proj.bias")?,
        };
        let vae = Vae {
            gpu: gpu.clone(),
            post_quant: conv("conv2")?,
            conv_in: conv("decoder.conv1")?,
            mid: [resnet("decoder.middle.0")?, resnet("decoder.middle.2")?],
            att,
            ups,
            out_norm: vec("decoder.head.0.gamma")?,
            conv_out: conv("decoder.head.2")?,
        };
        if vae.post_quant.ci != crate::dit::LATENT || vae.conv_out.co != 4 {
            return Err(Error(format!("{}: {} latent channels and {} out: Ming-Image's VAE takes 16 and makes RGBA", file.display(), vae.post_quant.ci,
                                     vae.conv_out.co)));
        }
        log(format!("ming-image vae: {} up stages on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.fp().cast(), out.ptr())?;
        Ok(out)
    }

    /// silu(rms norm(x) * gamma), a new buffer
    fn norm_silu(&self, nsd: &Nsd, x: &DevBuf, px: usize, c: usize, gamma: &DevBuf) -> Result<DevBuf> {
        let k = ffi::api()?;
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, gamma.fp().cast(), EPS, none(), none(), none(), out.ptr(), Dt::F16)?;
        // SAFETY: px x c halves.
        ffi::check(unsafe { (k.silu)(self.gpu.raw(), out.ptr(), (px * c) as i64) }, "silu")?;
        Ok(out)
    }

    /// Each intermediate is let go as soon as the next is made (after the work on it: a free does not wait for the
    /// queue) - at 2048 x 2048 a buffer of the last stage is 768 MiB, and the DiT's weights stay beside the VAE
    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a1 = self.norm_silu(nsd, &x, px, r.conv1.ci, &r.norm1)?;
        let a2 = self.conv(nsd, &r.conv1, &a1, h, w)?;
        nsd.wait()?;
        drop(a1);
        let a3 = self.norm_silu(nsd, &a2, px, r.conv1.co, &r.norm2)?;
        nsd.wait()?;
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

    /// The middle's attention: x + proj(attention(qkv(norm(x)))), one head of every channel over every pixel - two
    /// GEMMs and a softmax, a block of query rows at a time; v made transposed (its bias added after: each row of
    /// weights sums to 1)
    fn attention(&self, nsd: &Nsd, a: &Att, x: DevBuf, h: usize, w: usize, c: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, a.norm.fp().cast(), EPS, none(), none(), none(), n.ptr(), Dt::F16)?;
        let wpart = |i: usize| -> *const std::ffi::c_void { a.qkv.ptr().cast::<u16>().wrapping_add(i * c * c).cast() };
        let bpart = |i: usize| -> *const std::ffi::c_void { a.qkv_b.fp().wrapping_add(i * c).cast() };
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(0), c, bpart(0), q.ptr(), Dt::F16)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, wpart(1), c, bpart(1), kk.ptr(), Dt::F16)?;
        nsd.linear(wpart(2), Dt::F16, c, c, n.ptr(), px, none(), vt.ptr(), Dt::F16)?;
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
            nsd.linear(probs.ptr(), Dt::F16, rows, px, vt.ptr(), c, bpart(2), oa, Dt::F16)?;
            r0 += rows;
        }
        nsd.linear(o.ptr(), Dt::F16, px, c, a.proj.ptr(), c, a.proj_b.fp().cast(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// The picture from latents float32 [16, h, w] (as the denoiser leaves them): (its RGBA in [-1, 1] as half on the
    /// card, its width, height)
    pub fn decode_planes(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(DevBuf, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let z = crate::dit::LATENT;
        // channels last, divided by the scaling factor
        let mut zl = vec![0f32; h * w * z];
        for c in 0..z {
            for p in 0..h * w {
                zl[p * z + c] = lat[c * h * w + p] / SCALE;
            }
        }
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x 16 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, &self.att, x, h, w, self.conv_in.co)?;
        let mut x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            match up {
                Up::Res(r) => x = self.resnet(nsd, r, x, h, w)?,
                Up::Resample(c) => {
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
        }
        let a = self.norm_silu(nsd, &x, h * w, self.conv_out.ci, &self.out_norm)?;
        nsd.wait()?;
        drop(x);
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        nsd.wait()?;
        Ok((img, w, h))
    }

    /// RGBA [H, W, 4] bytes from latents float32 [16, h, w]
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let (img, pw, ph) = self.decode_planes(nsd, lat, h, w)?;
        let rgba = DevBuf::new(gpu, pw * ph * 4)?;
        // SAFETY: ph x pw x 4 halves in, as many bytes out.
        ffi::check(unsafe { (k.to_rgba8)(gpu.raw(), img.ptr(), ph as i64, pw as i64, rgba.ptr().cast()) }, "to rgba")?;
        nsd.wait()?;
        gpu.sync()?;
        let mut out = vec![0u8; pw * ph * 4];
        rgba.read(0, &mut out)?;
        Ok((out, pw, ph))
    }
}
