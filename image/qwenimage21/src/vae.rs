//! The decoder: Qwen-Image 2.1's VAE (diffusers' AutoencoderKLQwenImage21, a Wan 2.2-style decoder with 2-D kernels
//! and 64 latent channels, 16x down) from latents to RGBA. Everything is channels-last half on the GPU; one picture is
//! one frame, so the causal video parts reduce to their first chunk (the time convolutions are skipped, each
//! duplicate-upsample shortcut keeps its last time copy).
//!
//! ```text
//!   z [h, w, 64] (latents * std + mean) -> 1x1 conv -> 3x3 conv (1152)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   5 up blocks (1152, 1152, 576, 288, 144): 3 resnets, then (blocks 0-3) nearest 2x + 3x3 conv, plus the block
//!     input's duplicate-upsample shortcut
//!   rms norm, silu, 3x3 conv -> 4 channels (RGBA) in [-1, 1]
//! resnet: x + conv(silu(norm(conv(silu(norm(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! The file names its tensors as Wan / ComfyUI do (decoder.middle.0.residual.2, decoder.upsamples.3.upsamples.0, ...).

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi;

/// F.normalize's floor, as an epsilon on the mean square: never reached by real activations
const EPS: f32 = 1e-24;

/// A convolution: half weights [Co, Ci, k, k], float32 bias
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

struct Up {
    resnets: Vec<Resnet>,
    /// the 2x upsample's conv (blocks 0-3)
    resample: Option<Conv>,
    /// the shortcut's time copies (2: a block that also upsamples in time; 1: space only), when it upsamples
    ft: Option<i32>,
    ci: usize,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att_norm: DevBuf,
    att_qkv: DevBuf,
    att_qkv_b: DevBuf,
    att_proj: DevBuf,
    att_proj_b: DevBuf,
    ups: Vec<Up>,
    out_norm: DevBuf,
    conv_out: Conv,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub z: usize,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// Wan 2.2's latent statistics for the 64-channel VAE (diffusers' vae/config.json: latents_mean, latents_std), read
/// from the model's config when it is beside the file, else these
fn stats(file: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let dir = file.parent()?;
    for name in ["config.json", "vae_config.json"] {
        if let Ok(s) = std::fs::read_to_string(dir.join(name)) {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let f = |k: &str| v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect::<Vec<f32>>());
            if let (Some(m), Some(s)) = (f("latents_mean"), f("latents_std")) {
                return Some((m, s));
            }
        }
    }
    None
}

impl Vae {
    /// The decoder from its file; `stats`: the latents' per-channel (mean, std)
    pub fn load(file: &Path, nsd: &Nsd, latent_stats: (Vec<f32>, Vec<f32>), log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            let raw = st.read(t).map_err(ge)?;
            let n = t.elements() as usize;
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * 2)?;
            let code = match t.dtype.as_str() {
                "BF16" => 30,
                other => return Err(Error(format!("{name}: {other} weights are not read yet"))),
            };
            nsd.dequant(src.ptr(), code, n, out.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok((out, t.shape.clone()))
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
            let p = format!("decoder.upsamples.{i}.upsamples");
            if st.tensor(&format!("{p}.0.residual.0.gamma")).is_none() {
                break;
            }
            let resnets = (0..3).map(|j| resnet(&format!("{p}.{j}"))).collect::<Result<Vec<_>>>()?;
            let ci = resnets[0].conv1.ci;
            let up = if st.tensor(&format!("{p}.3.resample.1.weight")).is_some() {
                let time = st.tensor(&format!("{p}.3.time_conv.weight")).is_some();
                Up { resnets, resample: Some(conv(&format!("{p}.3.resample.1"))?), ft: Some(if time { 2 } else { 1 }), ci }
            } else {
                Up { resnets, resample: None, ft: None, ci }
            };
            ups.push(up);
        }
        let (att_qkv, _) = half("decoder.middle.1.to_qkv.weight")?;
        let (att_proj, _) = half("decoder.middle.1.proj.weight")?;
        let post_quant = conv("conv2")?;
        let z = post_quant.ci;
        let (mean, std) = stats(file).unwrap_or(latent_stats);
        if mean.len() != z || std.len() != z {
            return Err(Error(format!("the VAE takes {z} latent channels; the statistics have {} / {}", mean.len(), std.len())));
        }
        let vae = Vae {
            gpu: gpu.clone(),
            post_quant,
            conv_in: conv("decoder.conv1")?,
            mid: [resnet("decoder.middle.0")?, resnet("decoder.middle.2")?],
            att_norm: vec("decoder.middle.1.norm.gamma")?,
            att_qkv,
            att_qkv_b: vec("decoder.middle.1.to_qkv.bias")?,
            att_proj,
            att_proj_b: vec("decoder.middle.1.proj.bias")?,
            ups,
            out_norm: vec("decoder.head.0.gamma")?,
            conv_out: conv("decoder.head.2")?,
            mean,
            std,
            z,
        };
        log(format!("qwen-image 2.1 vae: {} up blocks on {} in {:.1} s", vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.ptr(), out.ptr())?;
        Ok(out)
    }

    /// silu(rms norm(x) * gamma), a new buffer
    fn norm_silu(&self, nsd: &Nsd, x: &DevBuf, px: usize, c: usize, gamma: &DevBuf) -> Result<DevBuf> {
        let k = ffi::api()?;
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, gamma.ptr(), EPS, none(), none(), none(), out.ptr(), Dt::F16)?;
        // SAFETY: px x c halves.
        ffi::check(unsafe { (k.silu)(self.gpu.raw(), out.ptr(), (px * c) as i64) }, "silu")?;
        Ok(out)
    }

    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a = self.norm_silu(nsd, &x, px, r.conv1.ci, &r.norm1)?;
        let a = self.conv(nsd, &r.conv1, &a, h, w)?;
        let a = self.norm_silu(nsd, &a, px, r.conv1.co, &r.norm2)?;
        let out = self.conv(nsd, &r.conv2, &a, h, w)?;
        let short = match &r.shortcut {
            Some(c) => self.conv(nsd, c, &x, h, w)?,
            None => x,
        };
        nsd.gate_add(out.ptr(), Dt::F16, px, r.conv2.co, short.ptr(), Dt::F16, none(), none())?;
        // the temporaries are freed on return: the work on them first (a free does not wait for the queue)
        nsd.wait()?;
        Ok(out)
    }

    /// RGBA [H, W, 4] bytes from latents float32 [h * w, z] (as the denoiser leaves them, before the statistics)
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let z = self.z;
        let zl: Vec<f32> = lat.iter().enumerate().map(|(i, v)| v * self.std[i % z] + self.mean[i % z]).collect();
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x z values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), zf.fp(), x.ptr(), zl.len() as i64) }, "to half")?;
        let x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        let x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        let x = self.resnet(nsd, &self.mid[0], x, h, w)?;
        let x = self.attention(nsd, x, h, w)?;
        let mut x = self.resnet(nsd, &self.mid[1], x, h, w)?;
        let (mut h, mut w) = (h, w);
        for up in &self.ups {
            // the block's input, for its shortcut
            let keep = match up.ft {
                Some(_) => {
                    let b = DevBuf::new(gpu, h * w * up.ci * 2)?;
                    b.copy_within(0, &x, 0, h * w * up.ci * 2)?;
                    Some(b)
                }
                None => None,
            };
            for r in &up.resnets {
                x = self.resnet(nsd, r, x, h, w)?;
            }
            if let (Some(c), Some(ft), Some(inp)) = (&up.resample, up.ft, keep) {
                let co = c.co;
                let u = DevBuf::new(gpu, 4 * h * w * c.ci * 2)?;
                // SAFETY: x holds h x w x ci halves, u four times that.
                ffi::check(unsafe { (k.up2)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr()) }, "upsample")?;
                x = self.conv(nsd, c, &u, 2 * h, 2 * w)?;
                // SAFETY: inp holds h x w x ci halves, x 2h x 2w x co.
                ffi::check(unsafe { (k.dupup_add)(gpu.raw(), inp.ptr(), h as i64, w as i64, up.ci as i64, co as i64, ft, x.ptr()) }, "shortcut")?;
                nsd.wait()?;
                h *= 2;
                w *= 2;
            }
        }
        let c = self.conv_out.ci;
        let a = self.norm_silu(nsd, &x, h * w, c, &self.out_norm)?;
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        let rgba = DevBuf::new(gpu, h * w * 4)?;
        // SAFETY: h x w x 4 halves in, as many bytes out.
        ffi::check(unsafe { (k.to_rgba8)(gpu.raw(), img.ptr(), h as i64, w as i64, rgba.ptr().cast()) }, "to rgba")?;
        nsd.wait()?;
        gpu.sync()?;
        let mut out = vec![0u8; h * w * 4];
        rgba.read(0, &mut out)?;
        Ok((out, w, h))
    }

    /// The middle's attention: x + proj(attention(qkv(norm(x)))), one head of every channel over every pixel
    fn attention(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let (px, c) = (h * w, self.conv_in.co);
        let n = DevBuf::new(gpu, px * c * 2)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F16, px, c, self.att_norm.ptr(), EPS, none(), none(), none(), n.ptr(), Dt::F16)?;
        let qkv = DevBuf::new(gpu, px * 3 * c * 2)?;
        nsd.linear(n.ptr(), Dt::F16, px, c, self.att_qkv.ptr(), 3 * c, self.att_qkv_b.ptr(), qkv.ptr(), Dt::F16)?;
        let at = |i: usize| -> *const std::ffi::c_void { qkv.ptr().cast::<u16>().wrapping_add(i * c).cast() };
        let o = DevBuf::new(gpu, px * c * 2)?;
        nsd.attention(at(0), at(1), at(2), Dt::F16, px, 1, c, 3 * c, o.ptr(), Dt::F16)?;
        nsd.linear(o.ptr(), Dt::F16, px, c, self.att_proj.ptr(), c, self.att_proj_b.ptr(), n.ptr(), Dt::F16)?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }
}
