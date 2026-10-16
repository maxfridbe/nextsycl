//! The Stable Diffusion 2 building blocks the paint model is made of, on half feature maps laid out channels last
//! ([images, height, width, channels]): convolutions (the shared kernels' oneDNN ones; a stride-2 one after explicit
//! zero padding), GroupNorm (+ SiLU), the residual blocks with their timestep bias.
//!
//! Memory: a pass takes its buffers from two arenas - `Mem::keep` for what outlives a block (its output, the UNet's
//! skips) and `Mem::tmp` for a block's temporaries, rewound after each block. Nothing is freed while kernels that read
//! it are queued (the queue is in order: memory handed out again is written only after the earlier readers ran).
#![allow(clippy::too_many_arguments)]

use nextsycl_core::{Arena, DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{Lin, Ops, Weights, P};

/// A pass's memory: what outlives a block, and a block's temporaries
pub struct Mem {
    pub keep: Arena,
    pub tmp: Arena,
}

impl Mem {
    pub fn new(gpu: &std::sync::Arc<Gpu>, keep: usize, tmp: usize) -> Result<Mem> {
        Ok(Mem { keep: Arena::new(gpu, keep)?, tmp: Arena::new(gpu, tmp)? })
    }
    /// The block is done: its temporaries can go
    pub fn end_block(&self) {
        self.tmp.reset();
    }
}

/// A temporary of `n` halves (or floats) from an arena
pub fn halves(a: &Arena, n: usize) -> Result<DevBuf> {
    a.bytes(n * 2)
}
pub fn floats(a: &Arena, n: usize) -> Result<DevBuf> {
    a.bytes(n * 4)
}

/// A feature map: n images of h x w x c, half, channels last
pub struct Map {
    pub buf: DevBuf,
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Map {
    pub fn new(a: &Arena, n: usize, h: usize, w: usize, c: usize) -> Result<Map> {
        Ok(Map { buf: halves(a, n * h * w * c)?, n, h, w, c })
    }
    pub fn len(&self) -> usize {
        self.n * self.h * self.w * self.c
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn rows(&self) -> usize {
        self.n * self.h * self.w
    }
    pub fn p(&self) -> P {
        self.buf.ptr()
    }
}

/// A k x k convolution with its float32 bias; stride 1 (same padding) or 2 (the padding given)
pub struct Conv {
    w: DevBuf,
    pub b: DevBuf,
    pub ci: usize,
    pub co: usize,
    pub k: usize,
    /// stride 2: (top, bottom, left, right) zero padding first
    pad2: Option<(usize, usize, usize, usize)>,
}

impl Conv {
    pub fn load(w: &Weights, name: &str, pad2: Option<(usize, usize, usize, usize)>) -> Result<Conv> {
        let s = w.shape(&format!("{name}.weight"))?;
        if s.len() != 4 || s[2] != s[3] || s[2] % 2 == 0 {
            return Err(Error(format!("{name}: a convolution of shape {s:?}")));
        }
        let m = w.host_mat(&format!("{name}.weight"))?;
        let lin = w.upload(&m, Some(w.host_f32(&format!("{name}.bias"))?))?;
        Ok(Conv { w: lin.w, b: lin.b.expect("a bias"), ci: s[1], co: s[0], k: s[2], pad2 })
    }

    /// The output's size for x
    pub fn out_size(&self, x: &Map) -> (usize, usize) {
        match self.pad2 {
            None => (x.h, x.w),
            Some(p) => ((x.h + p.0 + p.1 - self.k) / 2 + 1, (x.w + p.2 + p.3 - self.k) / 2 + 1),
        }
    }

    /// conv(x) into a new map of `keep` (`bias` in place of its own when given); temporaries from `tmp`
    pub fn run(&self, ops: &Ops, tmp: &Arena, keep: &Arena, x: &Map, bias: Option<P>) -> Result<Map> {
        if x.c != self.ci {
            return Err(Error(format!("a convolution of {} channels given {}", self.ci, x.c)));
        }
        let b = bias.unwrap_or(self.b.ptr());
        let (ho, wo) = self.out_size(x);
        let o = Map::new(keep, x.n, ho, wo, self.co)?;
        match self.pad2 {
            None => ops.nsd.conv2d(x.p(), Dt::F16, x.n, x.h, x.w, self.ci, self.w.ptr(), self.co, self.k, b, o.buf.ptr())?,
            Some(pad) => {
                let (hp, wp) = (x.h + pad.0 + pad.1, x.w + pad.2 + pad.3);
                let padded = halves(tmp, x.n * hp * wp * x.c)?;
                ops.pad2d(x.p(), x.n, x.h, x.w, x.c, pad, padded.ptr())?;
                ops.nsd.conv3d_ex(padded.ptr(), Dt::F16, (x.n, hp, wp), self.ci, self.w.ptr(), self.co, (1, self.k, self.k), (1, 2, 2), b, o.buf.ptr())?;
            }
        }
        Ok(o)
    }
}

/// A GroupNorm's affine weights (32 groups)
pub struct Gn {
    w: DevBuf,
    b: DevBuf,
    pub eps: f32,
}

pub const GROUPS: usize = 32;

impl Gn {
    pub fn load(w: &Weights, name: &str, eps: f32) -> Result<Gn> {
        Ok(Gn { w: w.vec(&format!("{name}.weight"))?, b: w.vec(&format!("{name}.bias"))?, eps })
    }
    /// GroupNorm(x) (+ SiLU) into a new map of `out`
    pub fn run(&self, ops: &Ops, tmp: &Arena, out: &Arena, x: &Map, silu: bool) -> Result<Map> {
        let o = Map::new(out, x.n, x.h, x.w, x.c)?;
        let stats = floats(tmp, ops.group_norm_scratch(x.n, x.h * x.w, x.c, GROUPS))?;
        ops.group_norm(x.p(), x.n, x.h * x.w, x.c, GROUPS, self.w.ptr(), self.b.ptr(), self.eps, silu, stats.ptr(), o.buf.ptr())?;
        Ok(o)
    }
}

/// A residual block (diffusers' ResnetBlock2D): GN + SiLU, conv (its bias plus the timestep's projection), GN + SiLU,
/// conv, plus the input (through a 1 x 1 convolution when the channels change)
pub struct Resnet {
    n1: Gn,
    c1: Conv,
    temb: Option<Lin>,
    n2: Gn,
    c2: Conv,
    short: Option<Lin>,
}

impl Resnet {
    pub fn load(w: &Weights, p: &str, eps: f32) -> Result<Resnet> {
        let short = if w.has(&format!("{p}.conv_shortcut.weight")) { Some(w.lin(&format!("{p}.conv_shortcut"))?) } else { None };
        Ok(Resnet {
            n1: Gn::load(w, &format!("{p}.norm1"), eps)?,
            c1: Conv::load(w, &format!("{p}.conv1"), None)?,
            temb: if w.has(&format!("{p}.time_emb_proj.weight")) { Some(w.lin(&format!("{p}.time_emb_proj"))?) } else { None },
            n2: Gn::load(w, &format!("{p}.norm2"), eps)?,
            c2: Conv::load(w, &format!("{p}.conv2"), None)?,
            short,
        })
    }

    /// The block on x into a new map of `mem.keep`; `temb`: silu(the timestep embedding), half [1280] (the UNet's),
    /// or none (the VAE's)
    pub fn run(&self, ops: &Ops, mem: &Mem, x: &Map, temb: Option<P>) -> Result<Map> {
        let (tmp, keep) = (&mem.tmp, &mem.keep);
        let h = self.n1.run(ops, tmp, tmp, x, true)?;
        crate::prof::part(&ops.nsd, "resnet norm")?;
        let bias = match (&self.temb, temb) {
            (Some(l), Some(t)) => {
                let b = floats(tmp, l.n)?;
                ops.lin(l, t, 1, b.ptr(), Dt::F32)?;
                ops.add(b.ptr(), self.c1.b.ptr(), false, 1, l.n, none())?;
                Some(b)
            }
            _ => None,
        };
        let h = self.c1.run(ops, tmp, tmp, &h, bias.as_ref().map(|b| b.ptr() as P))?;
        crate::prof::part(&ops.nsd, "resnet conv")?;
        let h2 = self.n2.run(ops, tmp, tmp, &h, true)?;
        crate::prof::part(&ops.nsd, "resnet norm")?;
        let out = self.c2.run(ops, tmp, keep, &h2, None)?;
        crate::prof::part(&ops.nsd, "resnet conv")?;
        match &self.short {
            Some(s) => {
                let sc = halves(tmp, x.rows() * s.n)?;
                ops.lin(s, x.p(), x.rows(), sc.ptr(), Dt::F16)?;
                ops.add_half(out.buf.ptr(), sc.ptr(), out.len(), 1.0)?;
            }
            None => ops.add_half(out.buf.ptr(), x.p(), out.len(), 1.0)?,
        }
        Ok(out)
    }
}

/// Stable Diffusion's timestep features: cos | sin of t e^(-ln(1e4) i / (dim / 2)) (diffusers' Timesteps with
/// flip_sin_to_cos=True, downscale_freq_shift=0)
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let a = t * (-(10000f32.ln()) * i as f32 / half as f32).exp();
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

/// Host floats as halves on the GPU, in an arena
pub fn upload_half(a: &Arena, v: &[f32]) -> Result<DevBuf> {
    let h: Vec<u16> = v.iter().map(|&x| crate::ops::f32_to_f16(x)).collect();
    let b = halves(a, h.len())?;
    // SAFETY: u16s viewed as bytes.
    b.write(0, unsafe { std::slice::from_raw_parts(h.as_ptr().cast::<u8>(), h.len() * 2) })?;
    Ok(b)
}
