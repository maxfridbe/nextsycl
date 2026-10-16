//! Stable Diffusion's VAE (diffusers' AutoencoderKL: 128 / 256 / 512 / 512 channels, a single-head attention in the
//! middle, 4 latent channels at an eighth of the size, scaling factor 0.18215): the encoder takes the paint model's
//! condition pictures (the views' normal and position maps, the reference picture) to latents (the posterior's mean),
//! and the decoder takes the generated views back to pictures. One picture at a time: the decoder's maps at 512 px
//! are 64 MB for every 128 channels.

use nextsycl_core::Result;
use nextsycl_diffusion::kernels::Dt;

use super::nn::{floats, halves, Conv, Gn, Map, Mem, Resnet};
use crate::ops::{hp, read_f32, Lin, Ops, Weights};

pub const SCALE: f32 = 0.18215;
const EPS: f32 = 1e-6;

struct Attn {
    gn: Gn,
    qkv: Lin,
    out: Lin,
}

impl Attn {
    fn load(w: &Weights, p: &str) -> Result<Attn> {
        let q = w.host_mat(&format!("{p}.query.weight"))?.stack(&w.host_mat(&format!("{p}.key.weight"))?).stack(&w.host_mat(&format!("{p}.value.weight"))?);
        let mut b = w.host_f32(&format!("{p}.query.bias"))?;
        b.extend(w.host_f32(&format!("{p}.key.bias"))?);
        b.extend(w.host_f32(&format!("{p}.value.bias"))?);
        Ok(Attn { gn: Gn::load(w, &format!("{p}.group_norm"), EPS)?, qkv: w.upload(&q, Some(b))?, out: w.lin(&format!("{p}.proj_attn"))? })
    }

    /// x + out(attention(q, k, v of gn(x))): one head of all the channels over every pixel of each picture
    fn run(&self, ops: &Ops, mem: &Mem, x: &Map) -> Result<Map> {
        let c = x.c;
        let h = self.gn.run(ops, &mem.tmp, &mem.tmp, x, false)?;
        let qkv = halves(&mem.tmp, x.rows() * 3 * c)?;
        ops.lin(&self.qkv, h.p(), x.rows(), qkv.ptr(), Dt::F16)?;
        let att = halves(&mem.tmp, x.rows() * c)?;
        ops.nsd.attention_batch(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), Dt::F16, x.n, x.h * x.w, 1, c, 3 * c, att.ptr(), Dt::F16)?;
        let o = Map::new(&mem.keep, x.n, x.h, x.w, c)?;
        ops.lin(&self.out, att.ptr(), x.rows(), o.buf.ptr(), Dt::F16)?;
        ops.add_half(o.buf.ptr(), x.p(), o.len(), 1.0)?;
        Ok(o)
    }
}

struct Mid {
    r0: Resnet,
    attn: Attn,
    r1: Resnet,
}

impl Mid {
    fn load(w: &Weights, p: &str) -> Result<Mid> {
        Ok(Mid { r0: Resnet::load(w, &format!("{p}.resnets.0"), EPS)?, attn: Attn::load(w, &format!("{p}.attentions.0"))?, r1: Resnet::load(w, &format!("{p}.resnets.1"), EPS)? })
    }
    fn run(&self, ops: &Ops, mem: &Mem, x: &Map) -> Result<Map> {
        let h = self.r0.run(ops, mem, x, None)?;
        mem.end_block();
        let h = self.attn.run(ops, mem, &h)?;
        mem.end_block();
        let h = self.r1.run(ops, mem, &h, None)?;
        mem.end_block();
        Ok(h)
    }
}

type Stage = (Vec<Resnet>, Option<Conv>);

pub struct SdVae {
    e_in: Conv,
    e_down: Vec<Stage>,
    e_mid: Mid,
    e_norm: Gn,
    e_out: Conv,
    quant: Lin,
    post_quant: Lin,
    d_in: Conv,
    d_mid: Mid,
    d_up: Vec<Stage>,
    d_norm: Gn,
    d_out: Conv,
    pub bytes: usize,
}

impl SdVae {
    pub fn load(w: &Weights) -> Result<SdVae> {
        let mut e_down = Vec::new();
        for i in 0..4 {
            let p = format!("encoder.down_blocks.{i}");
            let rs = (0..2).map(|j| Resnet::load(w, &format!("{p}.resnets.{j}"), EPS)).collect::<Result<Vec<_>>>()?;
            let ds = if w.has(&format!("{p}.downsamplers.0.conv.weight")) { Some(Conv::load(w, &format!("{p}.downsamplers.0.conv"), Some((0, 1, 0, 1)))?) } else { None };
            e_down.push((rs, ds));
        }
        let mut d_up = Vec::new();
        for i in 0..4 {
            let p = format!("decoder.up_blocks.{i}");
            let rs = (0..3).map(|j| Resnet::load(w, &format!("{p}.resnets.{j}"), EPS)).collect::<Result<Vec<_>>>()?;
            let us = if w.has(&format!("{p}.upsamplers.0.conv.weight")) { Some(Conv::load(w, &format!("{p}.upsamplers.0.conv"), None)?) } else { None };
            d_up.push((rs, us));
        }
        Ok(SdVae {
            e_in: Conv::load(w, "encoder.conv_in", None)?,
            e_down,
            e_mid: Mid::load(w, "encoder.mid_block")?,
            e_norm: Gn::load(w, "encoder.conv_norm_out", EPS)?,
            e_out: Conv::load(w, "encoder.conv_out", None)?,
            quant: w.lin("quant_conv")?,
            post_quant: w.lin("post_quant_conv")?,
            d_in: Conv::load(w, "decoder.conv_in", None)?,
            d_mid: Mid::load(w, "decoder.mid_block")?,
            d_up,
            d_norm: Gn::load(w, "decoder.conv_norm_out", EPS)?,
            d_out: Conv::load(w, "decoder.conv_out", None)?,
            bytes: w.bytes.get(),
        })
    }

    /// The latents of one picture (host float32 [h, w, 3] in [0, 1]): the posterior's mean times the scaling factor,
    /// host float32 [h/8, w/8, 4]
    pub fn encode(&self, ops: &Ops, mem: &Mem, px: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let mark = mem.keep.mark();
        let x = Map { buf: super::nn::upload_half(&mem.keep, &px.iter().map(|v| v * 2.0 - 1.0).collect::<Vec<_>>())?, n: 1, h, w, c: 3 };
        let mut y = self.e_in.run(ops, &mem.tmp, &mem.keep, &x, None)?;
        mem.end_block();
        for (rs, ds) in &self.e_down {
            for r in rs {
                y = r.run(ops, mem, &y, None)?;
                mem.end_block();
            }
            if let Some(d) = ds {
                y = d.run(ops, &mem.tmp, &mem.keep, &y, None)?;
                mem.end_block();
            }
        }
        y = self.e_mid.run(ops, mem, &y)?;
        let yn = self.e_norm.run(ops, &mem.tmp, &mem.keep, &y, true)?;
        let m = self.e_out.run(ops, &mem.tmp, &mem.keep, &yn, None)?;
        let q = floats(&mem.keep, m.rows() * 8)?;
        ops.lin(&self.quant, m.p(), m.rows(), q.ptr(), Dt::F32)?;
        let qh = read_f32(&q, m.rows() * 8)?;
        mem.end_block();
        mem.keep.rewind(mark);
        Ok(qh.chunks_exact(8).flat_map(|r| [r[0] * SCALE, r[1] * SCALE, r[2] * SCALE, r[3] * SCALE]).collect())
    }

    /// One picture (host float32 [8h, 8w, 3], clamped to [0, 1]) of latents (host float32 [h, w, 4], scaled)
    pub fn decode(&self, ops: &Ops, mem: &Mem, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        let mark = mem.keep.mark();
        let zin = Map { buf: super::nn::upload_half(&mem.keep, &z.iter().map(|v| v / SCALE).collect::<Vec<_>>())?, n: 1, h, w, c: 4 };
        let pq = Map::new(&mem.keep, 1, h, w, 4)?;
        ops.lin(&self.post_quant, zin.p(), zin.rows(), pq.buf.ptr(), Dt::F16)?;
        let mut x = self.d_in.run(ops, &mem.tmp, &mem.keep, &pq, None)?;
        mem.end_block();
        x = self.d_mid.run(ops, mem, &x)?;
        for (rs, us) in &self.d_up {
            for r in rs {
                x = r.run(ops, mem, &x, None)?;
                mem.end_block();
            }
            if let Some(u) = us {
                let up = Map::new(&mem.tmp, x.n, 2 * x.h, 2 * x.w, x.c)?;
                ops.up2(x.p(), x.n, x.h, x.w, x.c, up.buf.ptr())?;
                x = u.run(ops, &mem.tmp, &mem.keep, &up, None)?;
                mem.end_block();
            }
        }
        let xn = self.d_norm.run(ops, &mem.tmp, &mem.keep, &x, true)?;
        let o = self.d_out.run(ops, &mem.tmp, &mem.keep, &xn, None)?;
        let of = floats(&mem.keep, o.len())?;
        ops.to_float(o.p(), of.ptr(), o.len())?;
        let px = read_f32(&of, o.len())?;
        mem.end_block();
        mem.keep.rewind(mark);
        Ok(px.into_iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect())
    }
}
