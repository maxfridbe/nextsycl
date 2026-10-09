//! The pixel upscaler: an ESRGAN-type network (Real-ESRGAN's compact `SRVGGNetCompact` - `realesr-animevideov3`,
//! `realesr-general-x4v3`) on decoded frames, then an area resize to the target size. The other way to a larger clip
//! than the latent upscaler (upscale.rs): the clip is decoded at its sampled size and the frames are enlarged in
//! pixels, which skips the video decoder's cost at the larger size (2026-10-05 on the B65: 15.0 s to decode a 5 s
//! clip at 768x576 against 5.6 + 36.3 s through the latent upscaler to 1152x864). Off unless a job names it
//! (`pixel_upscaler`); the latent upscaler stays the default - it keeps more of the decoder's own fine texture.
//!
//! The network, per frame: a 3x3 convolution 3 -> 64, PReLU, then `n` times (3x3 convolution 64 -> 64, PReLU), a 3x3
//! convolution 64 -> 3 r^2, PixelShuffle(r), plus the input repeated r x r (nearest). The weights are Real-ESRGAN's
//! release files converted to safetensors with their own names (`body.<i>.weight`, `body.<i>.bias`) by
//! reference/esrgan_to_safetensors.py. Frames go through a few at a time, channels-last, in IEEE half.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dtype::{bytes_to_f32, f32_to_f16, DType};
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

/// frames per pass through the network (activations: 64 channels x the frame x 2 bytes each, twice)
const BATCH: usize = 4;

enum Layer {
    /// half [Co, Ci, k, k], float32 bias [Co]
    Conv { w: Tensor, b: Tensor, ci: usize, co: usize, k: usize },
    /// float32 [C]
    Prelu { alpha: Tensor },
}

pub struct PixelUpscaler {
    dev: Arc<Device>,
    layers: Vec<Layer>,
    /// the network's own factor (4 for the Real-ESRGAN compact models)
    pub scale: usize,
}

fn f32_tensor(dev: &Arc<Device>, v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

impl PixelUpscaler {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<PixelUpscaler> {
        let mut layers = Vec::new();
        for i in 0.. {
            let wn = format!("body.{i}.weight");
            let Ok(e) = ck.get(&wn) else { break };
            let v = bytes_to_f32(&ck.read(&wn)?, e.dtype)?;
            if e.shape.len() == 4 {
                let (co, ci, k) = (e.shape[0], e.shape[1], e.shape[2]);
                let half: Vec<u8> = v.iter().flat_map(|f| f32_to_f16(*f).to_le_bytes()).collect();
                let w = Tensor::from_bytes(dev, DType::F16, &e.shape, &half)?;
                let bn = format!("body.{i}.bias");
                let b = bytes_to_f32(&ck.read(&bn)?, ck.get(&bn)?.dtype)?;
                layers.push(Layer::Conv { w, b: f32_tensor(dev, &b)?, ci, co, k });
            } else {
                layers.push(Layer::Prelu { alpha: f32_tensor(dev, &v)? });
            }
        }
        let last_co = match layers.last() {
            Some(Layer::Conv { co, .. }) => *co,
            _ => return Err(Error("pixel upscaler: the checkpoint has no body.<i> layers ending in a convolution".into())),
        };
        let scale = ((last_co / 3) as f64).sqrt().round() as usize;
        if scale * scale * 3 != last_co {
            return Err(Error(format!("pixel upscaler: a last convolution of {last_co} channels is not 3 r^2")));
        }
        Ok(PixelUpscaler { dev: dev.clone(), layers, scale })
    }

    /// Frames planar [3, frames, h, w] in [0, 1] -> planar [3, frames, ho, wo]: the network (x `scale`), then the
    /// area resize to ho x wo. `tick` before every batch of frames (cancel checks).
    #[allow(clippy::too_many_arguments)]
    pub fn upscale(&self, px: &[f32], frames: usize, h: usize, w: usize, ho: usize, wo: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<Vec<f32>> {
        let d = &self.dev;
        let dt = DType::F16;
        let code = dt.kernel_code()?;
        let r = self.scale;
        let plane = h * w;
        if px.len() != 3 * frames * plane {
            return Err(Error(format!("pixel upscaler: {} values for 3 x {frames} x {h} x {w}", px.len())));
        }
        let feat = self.layers.iter().filter_map(|l| if let Layer::Conv { co, .. } = l { Some(*co) } else { None }).max().unwrap_or(64);
        let nb = BATCH.min(frames.max(1));
        let x = Tensor::new(d, dt, &[nb, h, w, 3])?;
        let a = Tensor::new(d, dt, &[nb * plane * feat])?;
        let b = Tensor::new(d, dt, &[nb * plane * feat])?;
        let big = Tensor::new(d, dt, &[nb * plane * r * r * 3])?;
        let small = Tensor::new(d, DType::F32, &[3 * nb * ho * wo])?;
        let mut out = vec![0f32; 3 * frames * ho * wo];
        let mut f0 = 0;
        while f0 < frames {
            tick()?;
            let n = nb.min(frames - f0);
            // this batch's frames, channels-last half
            let mut host = vec![0u8; n * plane * 3 * 2];
            for fi in 0..n {
                for i in 0..plane {
                    for c in 0..3 {
                        let v = f32_to_f16(px[(c * frames + f0 + fi) * plane + i]).to_le_bytes();
                        let o = ((fi * plane + i) * 3 + c) * 2;
                        host[o..o + 2].copy_from_slice(&v);
                    }
                }
            }
            x.buf.write(0, &host)?;
            // the body: ping-pong between a and b
            let (mut src, mut dst) = (&x, &a);
            let mut cur_c = 3;
            for (li, layer) in self.layers.iter().enumerate() {
                match layer {
                    Layer::Conv { w: wt, b: bias, ci, co, k } => {
                        if *ci != cur_c {
                            return Err(Error(format!("pixel upscaler: layer {li} takes {ci} channels, has {cur_c}")));
                        }
                        let target = if li + 1 == self.layers.len() { &big } else { dst };
                        // SAFETY: device buffers of this context, sized above for n frames of h x w and `co` channels.
                        let rc = unsafe {
                            (d.api.conv2d)(d.ctx, src.buf.ptr(), code, n as i64, h as i64, w as i64, *ci as i64, wt.buf.ptr(), *co as i64, *k as i64,
                                           bias.buf.ptr().cast(), target.buf.ptr())
                        };
                        d.check(rc)?;
                        cur_c = *co;
                        if li + 1 < self.layers.len() {
                            src = dst;
                            dst = if std::ptr::eq(dst, &a) { &b } else { &a };
                        }
                    }
                    Layer::Prelu { alpha } => {
                        // SAFETY: src holds n * plane rows of cur_c channels.
                        let rc = unsafe { (d.api.prelu)(d.ctx, src.buf.ptr(), code, (n * plane) as i64, cur_c as i64, alpha.buf.ptr().cast()) };
                        d.check(rc)?;
                    }
                }
            }
            // the last convolution wrote 3 r^2 channels into `big`'s front; shuffle (+ the input, nearest) into `a`
            let up = &a;
            // SAFETY: big [n, h, w, 3 r^2]; x [n, h, w, 3]; a has room for [n, h r, w r, 3] (64 >= 3 r^2 channels
            // per input pixel for r <= 4).
            if feat < 3 * r * r {
                return Err(Error("pixel upscaler: the work buffer is too small for the pixel shuffle".into()));
            }
            let rc = unsafe { (d.api.pixel_shuffle_add)(d.ctx, big.buf.ptr(), code, n as i64, h as i64, w as i64, 3, r as i64, x.buf.ptr(), up.buf.ptr()) };
            d.check(rc)?;
            // SAFETY: up [n, h r, w r, 3]; small has room for [3, n, ho, wo] float32.
            let rc = unsafe { (d.api.resize_area)(d.ctx, up.buf.ptr(), code, n as i64, (h * r) as i64, (w * r) as i64, 3, ho as i64, wo as i64, small.buf.ptr().cast()) };
            d.check(rc)?;
            let res = small.to_f32()?;
            for c in 0..3 {
                for fi in 0..n {
                    let s0 = (c * n + fi) * ho * wo;
                    let d0 = (c * frames + f0 + fi) * ho * wo;
                    out[d0..d0 + ho * wo].copy_from_slice(&res[s0..s0 + ho * wo]);
                }
            }
            f0 += n;
        }
        Ok(out)
    }
}
