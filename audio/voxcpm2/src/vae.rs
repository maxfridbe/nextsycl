//! AudioVAE V2 (voxcpm's modules/audiovae/audio_vae_v2.py): 64-wide latents at 25 a second, decoded to 48 kHz and
//! encoded from 16 kHz. Causal convolutions throughout, weight-normalized (the weights made at load: g * v / |v|).
//!
//! ```text
//!   decode: depthwise k7 (64), 1x1 to 2,048; 6 blocks (x8, x6, x5, x2, x2, x2), each after a sample-rate scale and
//!           bias (48 kHz's bucket): Snake, a transposed conv halving the channels, 3 residual units (Snake, depthwise
//!           k7 at dilation 1 / 3 / 9, Snake, 1x1); Snake, conv k7 to 1, tanh
//!   encode: conv k7 to 128; 4 blocks (x2, x5, x8, x8: 3 residual units, Snake, a strided conv doubling the
//!           channels); conv k3 to 64 (the mean)
//! ```
//!
//! Long latents are decoded in chunks, each behind some of the latents before it (the convolutions are causal: their
//! outputs are cut).

use std::collections::HashMap;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::Ops;
use crate::pth::Tensor;

/// A causal convolution: weight [co, ci, k] (depthwise: [c, 1, k]; transposed: [ci, co, k]), bias
struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    ci: usize,
    co: usize,
    k: usize,
    depthwise: bool,
}

struct Unit {
    a1: DevBuf,
    dw: Conv,
    a2: DevBuf,
    pw: Conv,
    dil: usize,
}

struct DecBlock {
    sr_scale: DevBuf,
    sr_bias: DevBuf,
    act: DevBuf,
    up: Conv,
    stride: usize,
    units: Vec<Unit>,
}

struct EncBlock {
    units: Vec<Unit>,
    act: DevBuf,
    down: Conv,
    stride: usize,
}

pub struct Vae {
    d_in: Conv,
    d_pw: Conv,
    blocks: Vec<DecBlock>,
    d_act: DevBuf,
    d_out: Conv,
    e_in: Conv,
    e_blocks: Vec<EncBlock>,
    e_mu: Conv,
    pub latent: usize,
    /// output samples a latent (48 kHz), input samples a latent (16 kHz)
    pub hop_out: usize,
    pub hop_in: usize,
}

/// The weight of a weight-normalized layer: g * v / |v| (the norm over all but the first dimension)
fn wn(t: &HashMap<String, Tensor>, p: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let miss = |n: &str| Error(format!("the AudioVAE has no {p}.{n}"));
    let v = t.get(&format!("{p}.weight_v")).ok_or_else(|| miss("weight_v"))?;
    let g = t.get(&format!("{p}.weight_g")).ok_or_else(|| miss("weight_g"))?;
    let per = v.data.len() / v.shape[0];
    let mut w = v.data.clone();
    for (o, row) in w.chunks_mut(per).enumerate() {
        let n = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        let s = g.data[o] / n.max(1e-12);
        row.iter_mut().for_each(|x| *x *= s);
    }
    Ok((w, v.shape.clone()))
}

fn dev(ops: &Ops, v: &[f32]) -> Result<DevBuf> {
    DevBuf::from_f32(&ops.gpu, v)
}

impl Vae {
    pub fn load(ops: &Ops, t: &HashMap<String, Tensor>, conf: &serde_json::Value) -> Result<Vae> {
        let get = |n: &str| t.get(n).ok_or_else(|| Error(format!("the AudioVAE has no {n}")));
        // depthwise: one input channel a group, as many groups as channels (the weight alone cannot say: [co, 1, k]
        // is also a convolution of a single channel)
        let conv_g = |p: &str, transposed: bool, depthwise: bool| -> Result<Conv> {
            let (w, s) = wn(t, p)?;
            let b = t.get(&format!("{p}.bias")).map(|b| dev(ops, &b.data)).transpose()?;
            let (ci, co) = if transposed { (s[0], s[1]) } else if depthwise { (s[0], s[0]) } else { (s[1], s[0]) };
            Ok(Conv { depthwise, w: dev(ops, &w)?, b, ci, co, k: s[2] })
        };
        let conv = |p: &str, transposed: bool| conv_g(p, transposed, false);
        let alpha = |p: &str| -> Result<DevBuf> { dev(ops, &get(&format!("{p}.alpha"))?.data) };
        let unit = |p: &str, dil: usize| -> Result<Unit> {
            Ok(Unit { a1: alpha(&format!("{p}.block.0"))?, dw: conv_g(&format!("{p}.block.1"), false, true)?, a2: alpha(&format!("{p}.block.2"))?,
                      pw: conv(&format!("{p}.block.3"), false)?, dil })
        };
        let vc = &conf["audio_vae_config"];
        let rates: Vec<usize> = vc["decoder_rates"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let erates: Vec<usize> = vc["encoder_rates"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        let out_rate = vc["out_sample_rate"].as_u64().unwrap_or(48000) as i64;
        let bounds: Vec<i64> = vc["sr_bin_boundaries"].as_array().into_iter().flatten().filter_map(|v| v.as_i64()).collect();
        // torch.bucketize: the first boundary at or above the rate
        let bucket = bounds.iter().position(|b| out_rate <= *b).unwrap_or(bounds.len());
        let mut blocks = Vec::new();
        for (i, &s) in rates.iter().enumerate() {
            let p = format!("decoder.model.{}", i + 2);
            let sc = get(&format!("decoder.sr_cond_model.{}.scale_embed.weight", i + 2))?;
            let bi = get(&format!("decoder.sr_cond_model.{}.bias_embed.weight", i + 2))?;
            let c = sc.shape[1];
            blocks.push(DecBlock {
                sr_scale: dev(ops, &sc.data[bucket * c..(bucket + 1) * c])?,
                sr_bias: dev(ops, &bi.data[bucket * c..(bucket + 1) * c])?,
                act: alpha(&format!("{p}.block.0"))?,
                up: conv(&format!("{p}.block.1"), true)?,
                stride: s,
                units: [1, 3, 9].iter().enumerate().map(|(j, d)| unit(&format!("{p}.block.{}", j + 2), *d)).collect::<Result<_>>()?,
            });
        }
        let n = rates.len() + 2;
        let mut e_blocks = Vec::new();
        for (i, &s) in erates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            e_blocks.push(EncBlock {
                units: [1, 3, 9].iter().enumerate().map(|(j, d)| unit(&format!("{p}.block.{j}"), *d)).collect::<Result<_>>()?,
                act: alpha(&format!("{p}.block.3"))?,
                down: conv(&format!("{p}.block.4"), false)?,
                stride: s,
            });
        }
        let v = Vae {
            d_in: conv_g("decoder.model.0", false, true)?,
            d_pw: conv("decoder.model.1", false)?,
            blocks,
            d_act: alpha(&format!("decoder.model.{n}"))?,
            d_out: conv(&format!("decoder.model.{}", n + 1), false)?,
            e_in: conv("encoder.block.0", false)?,
            e_blocks,
            e_mu: conv("encoder.fc_mu", false)?,
            latent: vc["latent_dim"].as_u64().unwrap_or(64) as usize,
            hop_out: rates.iter().product(),
            hop_in: erates.iter().product(),
        };
        ops.gpu.sync()?;
        Ok(v)
    }

    fn conv(&self, ops: &Ops, c: &Conv, x: *const f32, l: usize, dil: usize, out: *mut f32) -> Result<()> {
        let bias = c.b.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        if c.depthwise {
            ops.dwconv(x, c.ci, l, &c.w, bias, c.k, dil, out)
        } else {
            ops.conv_causal(x, 1, c.ci, l, &c.w, c.co, c.k, bias, dil, out)
        }
    }

    /// x [c, l] (in a) += the units, through b and c
    #[allow(clippy::too_many_arguments)]
    fn units(&self, ops: &Ops, units: &[Unit], ch: usize, l: usize, a: &DevBuf, b: &DevBuf, c: &DevBuf) -> Result<()> {
        for u in units {
            ops.snake(a.fp(), ch, l, &u.a1, b.fp())?;
            self.conv(ops, &u.dw, b.fp(), l, u.dil, c.fp())?;
            ops.snake(c.fp(), ch, l, &u.a2, b.fp())?;
            self.conv(ops, &u.pw, b.fp(), l, 1, c.fp())?;
            ops.add(a.fp(), c.fp(), ch * l)?;
        }
        Ok(())
    }

    /// Latents [64, t] (channels first) to samples (48 kHz)
    fn decode_once(&self, ops: &Ops, z: &[f32], t: usize) -> Result<Vec<f32>> {
        let g = &ops.gpu;
        let mut widest = self.d_pw.co * t;
        let mut l = t;
        for b in &self.blocks {
            l *= b.stride;
            widest = widest.max(b.up.co * l);
        }
        let (a, b, c) = (DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?);
        b.write(0, &crate::talker_bytes(z))?;
        self.conv(ops, &self.d_in, b.fp(), t, 1, c.fp())?;
        self.conv(ops, &self.d_pw, c.fp(), t, 1, a.fp())?;
        let (mut l, mut ch) = (t, self.d_pw.co);
        for blk in &self.blocks {
            ops.affine(a.fp(), ch, l, blk.sr_scale.fp(), blk.sr_bias.fp())?;
            ops.snake(a.fp(), ch, l, &blk.act, b.fp())?;
            ops.conv_up(b.fp(), 1, ch, l, &blk.up.w, blk.up.co, blk.up.k, blk.up.b.as_ref().map_or(std::ptr::null(), |x| x.fp()), blk.stride, a.fp())?;
            l *= blk.stride;
            ch = blk.up.co;
            self.units(ops, &blk.units, ch, l, &a, &b, &c)?;
        }
        ops.snake(a.fp(), ch, l, &self.d_act, b.fp())?;
        self.conv(ops, &self.d_out, b.fp(), l, 1, c.fp())?;
        ops.tanh(c.fp(), l)?;
        g.sync()?;
        Ok(c.to_f32()?[..l].to_vec())
    }

    /// Latents [64, t] to samples, in chunks of `chunk` latents behind `ctx` of the ones before
    pub fn decode(&self, ops: &Ops, z: &[f32], chunk: usize, ctx: usize) -> Result<Vec<f32>> {
        let d = self.latent;
        let t = z.len() / d;
        if t <= chunk + ctx {
            return self.decode_once(ops, z, t);
        }
        let mut out = Vec::with_capacity(t * self.hop_out);
        let mut s = 0;
        while s < t {
            let e = (s + chunk).min(t);
            let c0 = s.saturating_sub(ctx);
            let part: Vec<f32> = (0..d).flat_map(|ch| z[ch * t + c0..ch * t + e].iter().copied()).collect();
            let wav = self.decode_once(ops, &part, e - c0)?;
            out.extend_from_slice(&wav[(s - c0) * self.hop_out..]);
            s = e;
        }
        Ok(out)
    }

    /// 16 kHz samples (a multiple of the hop long) to latents [64, t] (channels first)
    pub fn encode(&self, ops: &Ops, wav: &[f32]) -> Result<Vec<f32>> {
        let g = &ops.gpu;
        let n = wav.len();
        if n == 0 || !n.is_multiple_of(self.hop_in) {
            return Err(Error(format!("{n} samples: a multiple of {}", self.hop_in)));
        }
        let widest = self.e_in.co * n * 2;
        let (a, b, c) = (DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?);
        b.write(0, &crate::talker_bytes(wav))?;
        self.conv(ops, &self.e_in, b.fp(), n, 1, a.fp())?;
        let (mut l, mut ch) = (n, self.e_in.co);
        for blk in &self.e_blocks {
            self.units(ops, &blk.units, ch, l, &a, &b, &c)?;
            ops.snake(a.fp(), ch, l, &blk.act, b.fp())?;
            l = ops.conv_strided(b.fp(), 1, ch, l, &blk.down.w, blk.down.co, blk.down.k,
                                 blk.down.b.as_ref().map_or(std::ptr::null(), |x| x.fp()), blk.stride, a.fp())?;
            ch = blk.down.co;
        }
        self.conv(ops, &self.e_mu, a.fp(), l, 1, b.fp())?;
        g.sync()?;
        Ok(b.to_f32()?[..self.latent * l].to_vec())
    }
}
