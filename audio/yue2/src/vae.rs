//! YuE2-Vae's decoder (modeling_vae.py `OobleckDecoder`, stable-audio-tools' Oobleck with SnakeBeta): 64 latent channels
//! at 25 Hz to 48 kHz stereo. A 7-tap convolution to 2048 channels, six blocks (SnakeBeta, a transposed convolution of
//! twice the stride - strides 6, 5, 4, 4, 2, 2, channels halving to 64 -, three residual units at dilations 1, 3, 9),
//! SnakeBeta and a 7-tap convolution to two channels; no tanh. The weight norm is folded in the GGUF; float32
//! throughout (MiniMax Music 3's oneDNN convolutions, strict float32). A song is decoded as `decode_tiled` does:
//! cores of 512 frames with 16 more either side (the decoder's support is 12), each core's samples kept exactly, no
//! blending - the full decoder's waveform.

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::Gguf;

use crate::ffi::{self, Api};
use crate::lm::{host_vec, Scratch, LATENT, M, P};

/// samples a latent frame (6 x 5 x 4 x 4 x 2 x 2)
pub const HOP: usize = 1920;
pub const RATE: u32 = 48_000;
/// the tiles: cores of this many frames (the pipeline's for a 12 GB budget), HALO either side
pub const CORE: usize = 512;
pub const HALO: usize = 16;
const STRIDES: [usize; 6] = [6, 5, 4, 4, 2, 2];

struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    ci: usize,
    co: usize,
    k: usize,
}

struct Snake {
    alpha: DevBuf,
    inv_beta: DevBuf,
    c: usize,
}

struct Res {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
    dil: usize,
}

struct Block {
    s: Snake,
    up: Conv,
    stride: usize,
    res: Vec<Res>,
}

pub struct Vae {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static Api,
    conv_in: Conv,
    blocks: Vec<Block>,
    snake_out: Snake,
    conv_out: Conv,
    pub bytes: u64,
}

impl Vae {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>) -> Result<Vae> {
        if f.architecture() != "yue2-vae" {
            return Err(Error(format!("{}: a {} file; YuE2's VAE is arch yue2-vae", f.paths[0].display(), f.architecture())));
        }
        let strides: Vec<usize> = f.meta("yue2vae.strides").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect())
            .unwrap_or_default();
        if strides.iter().rev().copied().collect::<Vec<_>>() != STRIDES {
            return Err(Error(format!("{}: strides {strides:?} (YuE2-Vae's 2, 2, 4, 4, 5, 6 expected)", f.paths[0].display())));
        }
        if f.meta("yue2vae.weight_norm_folded").and_then(|v| v.as_u64()) == Some(0) {
            return Err(Error("the VAE's weight norm is not folded in this file".into()));
        }
        let bytes = std::cell::Cell::new(0u64);
        let up = |v: &[f32]| -> Result<DevBuf> {
            bytes.set(bytes.get() + v.len() as u64 * 4);
            DevBuf::from_f32(gpu, v)
        };
        let conv = |p: &str, transposed: bool| -> Result<Conv> {
            let t = f.tensor(&format!("{p}.weight")).ok_or_else(|| Error(format!("no {p}.weight")))?;
            let (d0, d1, k) = (t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize);
            let w = up(&host_vec(f, &format!("{p}.weight"))?)?;
            let b = if f.tensor(&format!("{p}.bias")).is_some() { Some(up(&host_vec(f, &format!("{p}.bias"))?)?) } else { None };
            // a convolution [co, ci, k]; a transposed one [ci, co, k]
            let (ci, co) = if transposed { (d0, d1) } else { (d1, d0) };
            Ok(Conv { w, b, ci, co, k })
        };
        let snake = |p: &str| -> Result<Snake> {
            let a = host_vec(f, &format!("{p}.alpha"))?;
            let b = host_vec(f, &format!("{p}.beta"))?;
            // SnakeBeta with alpha_logscale: x + 1 / (e^beta + 1e-9) sin^2(x e^alpha)
            let alpha: Vec<f32> = a.iter().map(|x| x.exp()).collect();
            let inv: Vec<f32> = b.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect();
            Ok(Snake { c: alpha.len(), alpha: up(&alpha)?, inv_beta: up(&inv)? })
        };
        let mut blocks = Vec::new();
        for (i, stride) in STRIDES.iter().enumerate() {
            let p = format!("dec.blk.{}", i + 1);
            let res = [1, 3, 9].iter().enumerate().map(|(r, d)| -> Result<Res> {
                let q = format!("{p}.res.{r}");
                Ok(Res { s1: snake(&format!("{q}.snake1"))?, c1: conv(&format!("{q}.conv1"), false)?, s2: snake(&format!("{q}.snake2"))?,
                         c2: conv(&format!("{q}.conv2"), false)?, dil: *d })
            }).collect::<Result<Vec<_>>>()?;
            blocks.push(Block { s: snake(&format!("{p}.snake_pre"))?, up: conv(&format!("{p}.upsample"), true)?, stride: *stride, res });
        }
        let conv_in = conv("dec.conv_in", false)?;
        if conv_in.ci != LATENT {
            return Err(Error(format!("dec.conv_in: {} inputs, 64 expected", conv_in.ci)));
        }
        Ok(Vae { conv_in, blocks, snake_out: snake("dec.snake_out")?, conv_out: conv("dec.conv_out", false)?, k: ffi::api()?, nsd: Nsd::new(gpu)?,
                 gpu: gpu.clone(), bytes: bytes.get() })
    }

    fn snake(&self, s: &Snake, x: P, l: usize, out: M) -> Result<()> {
        // SAFETY: x and out hold s.c x l floats.
        ffi::check(unsafe { (self.k.snake)(self.gpu.raw(), x.cast(), s.c as i64, l as i64, s.alpha.fp(), s.inv_beta.fp(), out.cast()) }, "snake")
    }

    fn conv(&self, c: &Conv, x: P, l: usize, dil: usize, pad: usize, out: M) -> Result<usize> {
        let lo = (l + 2 * pad - dil * (c.k - 1) - 1) + 1;
        let b = c.b.as_ref().map_or(std::ptr::null(), |b| b.fp().cast_const());
        // SAFETY: x [ci, l] floats, the weights [co, ci, k], out [co, lo].
        ffi::check(unsafe { (self.k.conv1d)(self.gpu.raw(), x.cast(), 1, c.ci as i64, l as i64, c.w.fp(), c.co as i64, c.k as i64, b, 1, dil as i64, pad as i64, out.cast(), lo as i64) }, "conv1d")?;
        Ok(lo)
    }

    /// The samples a window of `frames` latents decodes to (`_output_length`)
    pub fn out_len(frames: usize) -> usize {
        let mut l = frames;
        for s in STRIDES {
            let (k, p) = (2 * s, s.div_ceil(2));
            l = (l - 1) * s - 2 * p + k;
        }
        l
    }

    /// One window: latents [frames, 64] (frames-major) to stereo [2, out_len(frames)] (channel-major) on the host
    fn window(&self, s: &mut Scratch, z: &[f32]) -> Result<Vec<f32>> {
        let frames = z.len() / LATENT;
        let g = &self.gpu;
        // the biggest activation: 64 channels at the audio rate (2048 x frames at the start is smaller)
        let big = (frames * HOP * 64).max(frames * 2048) * 4 + 4096;
        let bufs = [s.get(g, "v0", big)?, s.get(g, "v1", big)?, s.get(g, "v2", big)?];
        let mut zt = vec![0f32; z.len()];
        for f in 0..frames {
            for c in 0..LATENT {
                zt[c * frames + f] = z[f * LATENT + c];
            }
        }
        let zin = s.get(g, "vz", zt.len() * 4)?;
        s.buf("vz").write(0, crate::f32_bytes(&zt))?;
        let (mut x, a, mut b) = (bufs[0], bufs[1], bufs[2]);
        let mut l = self.conv(&self.conv_in, zin, frames, 1, 3, x)?;
        for blk in &self.blocks {
            self.snake(&blk.s, x, l, a)?;
            let pad = blk.stride.div_ceil(2);
            let lo = (l - 1) * blk.stride - 2 * pad + blk.up.k;
            let bias = blk.up.b.as_ref().map_or(std::ptr::null(), |b| b.fp().cast_const());
            // SAFETY: a [ci, l] floats, the weights as stored, b [co, lo].
            ffi::check(unsafe { (self.k.conv_transpose1d)(self.gpu.raw(), a.cast(), 1, blk.up.ci as i64, l as i64, blk.up.w.fp(), blk.up.co as i64, blk.up.k as i64, bias,
                                                          blk.stride as i64, pad as i64, b.cast(), lo as i64) }, "conv_transpose1d")?;
            l = lo;
            std::mem::swap(&mut x, &mut b);
            for r in &blk.res {
                // x += conv2(snake2(conv1(snake1(x))))
                self.snake(&r.s1, x, l, a)?;
                self.conv(&r.c1, a, l, r.dil, 3 * r.dil, b)?;
                self.snake(&r.s2, b, l, b)?;
                self.conv(&r.c2, b, l, 1, 0, a)?;
                self.nsd.gate_add(x, Dt::F32, 1, r.c1.co * l, a, Dt::F32, none(), none())?;
            }
        }
        self.snake(&self.snake_out, x, l, a)?;
        self.conv(&self.conv_out, a, l, 1, 3, b)?;
        let name = if b == bufs[0] { "v0" } else if b == bufs[1] { "v1" } else { "v2" };
        let mut raw = vec![0u8; 2 * l * 4];
        s.buf(name).read(0, &mut raw)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The song's latents [frames, 64] to stereo [2, n] (channel-major), clamped to [-1, 1]: `decode_tiled`'s cores
    pub fn decode(&self, s: &mut Scratch, z: &[f32], progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let frames = z.len() / LATENT;
        if frames == 0 {
            return Err(Error("no latents to decode".into()));
        }
        let total = Self::out_len(frames);
        let mut out = vec![0f32; 2 * total];
        let tiles = frames.div_ceil(CORE);
        for (ti, start) in (0..frames).step_by(CORE).enumerate() {
            progress(ti, tiles)?;
            let end = (start + CORE).min(frames);
            let (left, right) = (start.saturating_sub(HALO), (end + HALO).min(frames));
            let w = self.window(s, &z[left * LATENT..right * LATENT])?;
            let wl = w.len() / 2;
            let (o0, o1) = (start * HOP, (end * HOP).min(total));
            let c0 = (start - left) * HOP;
            if c0 + (o1 - o0) > wl {
                return Err(Error("a VAE tile did not cover its core".into()));
            }
            for ch in 0..2 {
                for (i, v) in w[ch * wl + c0..ch * wl + c0 + (o1 - o0)].iter().enumerate() {
                    out[ch * total + o0 + i] = v.clamp(-1.0, 1.0);
                }
            }
        }
        progress(tiles, tiles)?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error("the VAE made non-finite audio".into()));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_decoder_length_is_upstreams() {
        // "Full waveform, FP32 [B, 2, 1920 T - 64]"
        assert_eq!(super::Vae::out_len(250), 1920 * 250 - 64);
        assert_eq!(super::Vae::out_len(1), 1920 - 64);
    }
}
