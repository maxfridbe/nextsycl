//! The sparse-structure decoder (TRELLIS's SparseStructureDecoder, ss_dec_conv3d_16l8): the 8-channel 16^3 latent
//! through dense 3-D convolutions - 512 channels at 16^3 (two middle and two more residual blocks), a pixel-shuffle
//! upsample to 128 at 32^3, another to 32 at 64^3 - to one occupancy logit a cell of 64^3. Channel layer norms
//! (eps 1e-5, affine), SiLU; the convolutions as the sparse ones over the full grid, in half.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::sconv::{grid, upload_nmap, SConv};
use crate::sparse::neighbours;
use crate::w::File;
use crate::Ctx;

struct Res {
    n1: (DevBuf, DevBuf),
    c1: SConv,
    n2: (DevBuf, DevBuf),
    c2: SConv,
}

enum Blk {
    Res(Box<Res>),
    Up(SConv),
}

pub struct SsDecoder {
    input: SConv,
    blocks: Vec<Blk>,
    out_norm: (DevBuf, DevBuf),
    out: SConv,
}

impl SsDecoder {
    pub fn load(f: &File, gpu: &Arc<Gpu>) -> Result<SsDecoder> {
        let ln = |p: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(gpu, &f.vec(&format!("{p}.weight"))?)?, DevBuf::from_f32(gpu, &f.vec(&format!("{p}.bias"))?)?))
        };
        let res = |p: &str| -> Result<Res> {
            Ok(Res { n1: ln(&format!("{p}.norm1"))?, c1: SConv::load_dense(f, gpu, &format!("{p}.conv1"))?, n2: ln(&format!("{p}.norm2"))?,
                     c2: SConv::load_dense(f, gpu, &format!("{p}.conv2"))? })
        };
        let mut blocks = vec![Blk::Res(Box::new(res("middle_block.0")?)), Blk::Res(Box::new(res("middle_block.1")?))];
        for i in 0.. {
            let p = format!("blocks.{i}");
            if f.has(&format!("{p}.conv1.weight")) {
                blocks.push(Blk::Res(Box::new(res(&p)?)));
            } else if f.has(&format!("{p}.conv.weight")) {
                blocks.push(Blk::Up(SConv::load_dense(f, gpu, &format!("{p}.conv"))?));
            } else {
                break;
            }
        }
        let input = SConv::load_dense(f, gpu, "input_layer")?;
        if input.ci != 8 {
            return Err(Error(format!("{}: a sparse-structure decoder of {} latent channels, not 8", f.path(), input.ci)));
        }
        Ok(SsDecoder { input, blocks, out_norm: ln("out_layer.0")?, out: SConv::load_dense(f, gpu, "out_layer.2")? })
    }

    /// The occupancy logits [d^3] (d = 16 x 4 = 64, row-major (x, y, z)) of the latent z [8, 16, 16, 16] (channels
    /// first, as the reference's tensor)
    pub fn decode(&self, cx: &Ctx, z: &[f32]) -> Result<(usize, Vec<f32>)> {
        let g = &cx.gpu;
        let mut d = 16;
        if z.len() != 8 * d * d * d {
            return Err(Error(format!("the structure latent: {} values, not 8 x 16^3", z.len())));
        }
        let n0 = d * d * d;
        let x: Vec<f32> = (0..n0 * 8).map(|i| z[(i % 8) * n0 + i / 8]).collect();
        let mut nm = upload_nmap(g, &neighbours(&grid(d)))?;
        let xh = crate::w::dev_half(g, &x)?;
        let mut c = self.input.co;
        let mut h = DevBuf::new(g, n0 * c * 2)?;
        self.input.apply(cx, &xh, n0, &nm, h.ptr(), Dt::F16)?;
        for b in &self.blocks {
            let n = d * d * d;
            match b {
                Blk::Res(r) => {
                    let t = DevBuf::new(g, n * c * 2)?;
                    let u = DevBuf::new(g, n * c * 2)?;
                    cx.k.ln_mod(h.ptr(), Dt::F16, n, c, 1e-5, r.n1.0.fp(), r.n1.1.fp(), std::ptr::null(), std::ptr::null(), t.ptr(), Dt::F16)?;
                    cx.k.silu(t.ptr(), Dt::F16, n * c)?;
                    r.c1.apply(cx, &t, n, &nm, u.ptr(), Dt::F16)?;
                    cx.k.ln_mod(u.ptr(), Dt::F16, n, c, 1e-5, r.n2.0.fp(), r.n2.1.fp(), std::ptr::null(), std::ptr::null(), t.ptr(), Dt::F16)?;
                    cx.k.silu(t.ptr(), Dt::F16, n * c)?;
                    r.c2.apply(cx, &t, n, &nm, u.ptr(), Dt::F16)?;
                    cx.k.add(h.ptr(), Dt::F16, u.ptr(), Dt::F16, n * c)?;
                }
                Blk::Up(cv) => {
                    let u = DevBuf::new(g, n * cv.co * 2)?;
                    cv.apply(cx, &h, n, &nm, u.ptr(), Dt::F16)?;
                    c = cv.co / 8;
                    h = DevBuf::new(g, n * 8 * c * 2)?;
                    cx.k.pixel_shuffle3d(u.ptr(), d, c, h.ptr())?;
                    d *= 2;
                    nm = upload_nmap(g, &neighbours(&grid(d)))?;
                }
            }
        }
        let n = d * d * d;
        let t = DevBuf::new(g, n * c * 2)?;
        cx.k.ln_mod(h.ptr(), Dt::F16, n, c, 1e-5, self.out_norm.0.fp(), self.out_norm.1.fp(), std::ptr::null(), std::ptr::null(), t.ptr(), Dt::F16)?;
        cx.k.silu(t.ptr(), Dt::F16, n * c)?;
        let o = DevBuf::f32(g, n)?;
        self.out.apply(cx, &t, n, &nm, o.ptr(), Dt::F32)?;
        Ok((d, o.to_f32()?))
    }
}

/// The occupied cells at `res` (max-pooled from the decoder's grid when coarser): the reference's argwhere order
pub fn occupied(d: usize, logits: &[f32], res: usize) -> Vec<[i32; 3]> {
    let r = d / res;
    let mut out = Vec::new();
    for x in 0..res {
        for y in 0..res {
            for z in 0..res {
                let any = (0..r * r * r).any(|o| {
                    let (a, b, c) = (x * r + o / (r * r), y * r + o / r % r, z * r + o % r);
                    logits[(a * d + b) * d + c] > 0.0
                });
                if any {
                    out.push([x as i32, y as i32, z as i32]);
                }
            }
        }
    }
    out
}
