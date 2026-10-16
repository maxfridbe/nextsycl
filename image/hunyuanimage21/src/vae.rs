//! The decoder: HunyuanImage 2.1's 32x VAE (the official HunyuanVAE2D's Decoder, DC-AE style), latents [h, w, 64] to
//! RGB [32h, 32w, 3]. Channels-last half on the GPU, the weights F16 from the GGUF.
//!
//! ```text
//!   z = latents / 0.75289
//!   h = conv_in(z) (3x3, 64 -> 1024) + z repeated 16x per channel
//!   middle: resnet, attention (one head of 1024 over every pixel), resnet
//!   6 levels (1024, 1024, 512, 512, 256, 128): 3 resnets each; levels 0-4 then upsample:
//!     conv (3x3, C -> 4 C') rearranged "(r1 r2 c) h w -> c (h r1) (w r2)", plus the input repeated 4 C' / C times
//!     per channel, rearranged the same way
//!   GroupNorm(32), swish, conv_out (3x3, 128 -> 3)
//! resnet: x + conv2(swish(GN2(conv1(swish(GN1(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! The official decoder is run whole (its spatial tiling "leads to severe reconstruction artifacts with the pretrained
//! VAE checkpoint"); a 2048 x 2048 picture's largest activation is 128 x 2048 x 2048 halves (1 GiB).

use std::path::Path;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi::K;
use crate::prof;
use crate::w::{floats, ge, host, tensor};

pub const SCALING: f32 = 0.75289;
const Z: usize = 64;
const CHANNELS: [usize; 6] = [1024, 1024, 512, 512, 256, 128];
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;

/// A convolution: half weights [Co, Ci, k, k], float32 bias
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Norm {
    w: DevBuf,
    b: DevBuf,
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
    q: Conv,
    k: Conv,
    v: Conv,
    proj: Conv,
}

struct Level {
    resnets: Vec<Resnet>,
    up: Option<Conv>,
}

pub struct Vae {
    k: K,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    levels: Vec<Level>,
    norm_out: Norm,
    conv_out: Conv,
    scratch: DevBuf,
    pub bytes: usize,
}

impl Vae {
    pub fn load(file: &Path, nsd: &Nsd, k: &K, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = &nsd.gpu;
        let f = Gguf::open(file).map_err(ge)?;
        let mut bytes = 0usize;
        let mut conv = |name: &str| -> Result<Conv> {
            let t = tensor(&f, &format!("{name}.weight"))?;
            let (co, ci, kk) = match t.shape.as_slice() {
                [co, ci, kh, kw] if kh == kw => (*co as usize, *ci as usize, *kh as usize),
                s => return Err(Error(format!("{name}.weight: {s:?}, not a square 2-D convolution"))),
            };
            let raw = f.read(t).map_err(ge)?;
            let half: Vec<u8> = match t.ty {
                GType::F16 => raw,
                ty => floats(name, ty, &raw)?.iter().flat_map(|v| half_bits(*v).to_le_bytes()).collect(),
            };
            let w = DevBuf::new(gpu, half.len())?;
            w.write(0, &half)?;
            let b = DevBuf::from_f32(gpu, &host(&f, &format!("{name}.bias"))?)?;
            bytes += w.len + b.len;
            Ok(Conv { w, b, ci, co, k: kk })
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm { w: DevBuf::from_f32(gpu, &host(&f, &format!("{name}.weight"))?)?, b: DevBuf::from_f32(gpu, &host(&f, &format!("{name}.bias"))?)? })
        };
        let has = |name: &str| f.tensor(name).is_some();
        let resnet = |p: &str, conv: &mut dyn FnMut(&str) -> Result<Conv>| -> Result<Resnet> {
            Ok(Resnet {
                norm1: norm(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1"))?,
                norm2: norm(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2"))?,
                shortcut: if has(&format!("{p}.nin_shortcut.weight")) { Some(conv(&format!("{p}.nin_shortcut"))?) } else { None },
            })
        };
        let conv_in = conv("decoder.conv_in")?;
        if conv_in.ci != Z || conv_in.co != CHANNELS[0] {
            return Err(Error(format!("{}: decoder.conv_in {} -> {}: not HunyuanImage 2.1's 32x VAE", file.display(), conv_in.ci, conv_in.co)));
        }
        let mid = [resnet("decoder.mid.block_1", &mut conv)?, resnet("decoder.mid.block_2", &mut conv)?];
        let att = Att {
            norm: norm("decoder.mid.attn_1.norm")?,
            q: conv("decoder.mid.attn_1.q")?,
            k: conv("decoder.mid.attn_1.k")?,
            v: conv("decoder.mid.attn_1.v")?,
            proj: conv("decoder.mid.attn_1.proj_out")?,
        };
        let mut levels = Vec::new();
        for i in 0..CHANNELS.len() {
            let mut resnets = Vec::new();
            for j in 0.. {
                let p = format!("decoder.up.{i}.block.{j}");
                if !has(&format!("{p}.conv1.weight")) {
                    break;
                }
                resnets.push(resnet(&p, &mut conv)?);
            }
            let up = if has(&format!("decoder.up.{i}.upsample.conv.weight")) { Some(conv(&format!("decoder.up.{i}.upsample.conv"))?) } else { None };
            levels.push(Level { resnets, up });
        }
        let norm_out = norm("decoder.norm_out")?;
        let conv_out = conv("decoder.conv_out")?;
        let scratch = DevBuf::f32(gpu, k.gn_scratch(GROUPS))?;
        log(format!("vae: the 32x decoder ({:.2} GiB, half) in {:.1} s", bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
        Ok(Vae { k: k.clone(), conv_in, mid, att, levels, norm_out, conv_out, scratch, bytes })
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&nsd.gpu, h * w * c.co * 2)?;
        nsd.conv2d(x.ptr(), Dt::F16, 1, h, w, c.ci, c.w.ptr(), c.co, c.k, c.b.ptr(), out.ptr())?;
        prof::mark(nsd, if c.k == 1 { "vae conv 1x1" } else { "vae conv 3x3" })?;
        Ok(out)
    }

    fn gn(&self, nsd: &Nsd, n: &Norm, x: &DevBuf, px: usize, c: usize, silu: bool) -> Result<DevBuf> {
        let out = DevBuf::new(&nsd.gpu, px * c * 2)?;
        self.k.group_norm(x.ptr(), px, c, GROUPS, n.w.fp(), n.b.fp(), EPS, silu, self.scratch.fp(), out.ptr())?;
        prof::mark(nsd, "vae group norm")?;
        Ok(out)
    }

    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let a = self.gn(nsd, &r.norm1, &x, px, r.conv1.ci, true)?;
        let a = self.conv(nsd, &r.conv1, &a, h, w)?;
        let b = self.gn(nsd, &r.norm2, &a, px, r.conv2.ci, true)?;
        drop(a);
        let b = self.conv(nsd, &r.conv2, &b, h, w)?;
        let x = match &r.shortcut {
            Some(s) => self.conv(nsd, s, &x, h, w)?,
            None => x,
        };
        nsd.gate_add(x.ptr(), Dt::F16, px, r.conv2.co, b.ptr(), Dt::F16, none(), none())?;
        prof::mark(nsd, "vae add")?;
        Ok(x)
    }

    /// The middle's attention: x + proj(softmax(q k^T / sqrt(C)) v) over every pixel, one head
    fn attention(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let gpu = &nsd.gpu;
        let a = &self.att;
        let (s, c) = (h * w, a.q.co);
        let n = self.gn(nsd, &a.norm, &x, s, c, false)?;
        let lin = |cv: &Conv, x: &DevBuf| -> Result<DevBuf> {
            let o = DevBuf::new(gpu, s * cv.co * 2)?;
            nsd.linear(x.ptr(), Dt::F16, s, cv.ci, cv.w.ptr(), cv.co, cv.b.ptr(), o.ptr(), Dt::F16)?;
            Ok(o)
        };
        let (q, kk, v) = (lin(&a.q, &n)?, lin(&a.k, &n)?, lin(&a.v, &n)?);
        let scores = DevBuf::f32(gpu, s * s)?;
        nsd.linear(q.ptr(), Dt::F16, s, c, kk.ptr(), s, none(), scores.ptr(), Dt::F32)?;
        let p = DevBuf::new(gpu, s * s * 2)?;
        self.k.softmax_rows(scores.fp(), s, s, 1.0 / (c as f32).sqrt(), p.ptr())?;
        drop(scores);
        let vt = DevBuf::new(gpu, s * c * 2)?;
        self.k.transpose(v.ptr(), s, c, vt.ptr())?;
        let o = DevBuf::new(gpu, s * c * 2)?;
        nsd.linear(p.ptr(), Dt::F16, s, s, vt.ptr(), c, none(), o.ptr(), Dt::F16)?;
        let pr = lin(&a.proj, &o)?;
        nsd.gate_add(x.ptr(), Dt::F16, s, c, pr.ptr(), Dt::F16, none(), none())?;
        prof::mark(nsd, "vae attention")?;
        Ok(x)
    }

    /// RGB bytes [32h x 32w x 3] from latents float32 [h * w, 64] (the DiT's, before the scaling)
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let (x, hh, ww) = self.decode_half(nsd, lat, h, w)?;
        let rgb = DevBuf::new(&nsd.gpu, hh * ww * 3)?;
        self.k.to_rgb8(x.ptr(), hh * ww, rgb.ptr().cast())?;
        nsd.wait()?;
        let mut out = vec![0u8; hh * ww * 3];
        rgb.read(0, &mut out)?;
        Ok((out, ww, hh))
    }

    /// The decoder's output in [-1, 1]: float32 [32h, 32w, 3]
    pub fn decode_float(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let (x, hh, ww) = self.decode_half(nsd, lat, h, w)?;
        let f = DevBuf::f32(&nsd.gpu, hh * ww * 3)?;
        self.k.to_float(x.ptr(), f.fp(), hh * ww * 3)?;
        nsd.wait()?;
        f.to_f32()
    }

    fn decode_half(&self, nsd: &Nsd, lat: &[f32], h: usize, w: usize) -> Result<(DevBuf, usize, usize)> {
        let gpu = &nsd.gpu;
        if lat.len() != h * w * Z {
            return Err(Error(format!("{} latents for {h} x {w} x {Z}", lat.len())));
        }
        prof::reset_clock(nsd)?;
        let l = DevBuf::from_f32(gpu, lat)?;
        let z = DevBuf::new(gpu, h * w * Z * 2)?;
        self.k.to_half(l.fp(), z.ptr(), h * w * Z, 1.0 / SCALING)?;
        let mut x = self.conv(nsd, &self.conv_in, &z, h, w)?;
        self.k.rep_add(x.ptr(), z.ptr(), h * w, self.conv_in.co, Z)?;
        drop(z);
        let (mut hh, mut ww) = (h, w);
        x = self.resnet(nsd, &self.mid[0], x, hh, ww)?;
        x = self.attention(nsd, x, hh, ww)?;
        x = self.resnet(nsd, &self.mid[1], x, hh, ww)?;
        let mut c = self.conv_in.co;
        for lv in &self.levels {
            for r in &lv.resnets {
                x = self.resnet(nsd, r, x, hh, ww)?;
                c = r.conv2.co;
            }
            if let Some(up) = &lv.up {
                let cv = self.conv(nsd, up, &x, hh, ww)?;
                let co = up.co / 4;
                let o = DevBuf::new(gpu, 4 * hh * ww * co * 2)?;
                self.k.up_shuffle(cv.ptr(), x.ptr(), hh, ww, co, c, o.ptr())?;
                prof::mark(nsd, "vae upsample")?;
                drop(cv);
                x = o;
                hh *= 2;
                ww *= 2;
                c = co;
            }
        }
        let n = self.gn(nsd, &self.norm_out, &x, hh * ww, c, true)?;
        drop(x);
        let out = self.conv(nsd, &self.conv_out, &n, hh, ww)?;
        nsd.wait()?;
        prof::used(nsd, "at the VAE's last convolution");
        Ok((out, hh, ww))
    }
}

/// float32 -> IEEE half bits (round to nearest even)
fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_bits_round_to_nearest_even() {
        assert_eq!(super::half_bits(1.0), 0x3c00);
        assert_eq!(super::half_bits(-2.0), 0xc000);
        assert_eq!(super::half_bits(65504.0), 0x7bff);
        assert_eq!(super::half_bits(1e6), 0x7c00);
        assert_eq!(super::half_bits(1.0 + 1.0 / 2048.0), 0x3c00, "a tie to even");
        assert_eq!(super::half_bits(5.960464e-8), 0x0001, "the smallest subnormal");
    }
}
