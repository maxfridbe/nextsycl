//! The σ-VAE tokenizers, streaming: the acoustic encoder (a voice's recording to 64-value latents at 7.5 Hz), the
//! acoustic decoder (a latent to 3,200 samples at 24 kHz) and the semantic encoder (each frame's samples back to 128
//! values for the language model). Each is a stem convolution, then stages of ConvNeXt blocks (RMS norm over channels,
//! a depthwise causal convolution of 7, a layer scale; RMS norm, a GELU MLP of 4x, a layer scale) joined by strided
//! convolutions (the encoders: kernel 2r, stride r) or transposed ones (the decoder), then a head convolution. Every
//! convolution is causal, so a stream that carries each one's last inputs (its context: zeros at the start) gives
//! what the whole clip would - upstream decodes and re-encodes a frame at a time this way.
//!
//! The signal runs as time rows [L, C] (the norms and the MLP work on rows); a channel-mixing convolution transposes
//! to [C, L] around oneDNN's (mm3's conv1d), the depthwise one runs on rows (vv.cpp).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{Mat, Ops};
use crate::weights::Weights;

/// A channel-mixing causal convolution (or a transposed one): float32 weights [co, ci, k] ([ci, co, k] transposed)
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
    s: usize,
    tr: bool,
}

/// The input rows a convolution carries between calls: its causal padding (k - s), or for a transposed one the inputs
/// its next outputs still overlap (ceil(k / s) - 1)
fn ctx_rows(k: usize, s: usize, tr: bool) -> usize {
    if tr {
        k.div_ceil(s) - 1
    } else {
        k - s
    }
}

/// The rows a convolution makes of `l`
fn out_rows(s: usize, tr: bool, l: usize) -> usize {
    if tr {
        l * s
    } else {
        l / s
    }
}

impl Conv {
    fn ctx(&self) -> usize {
        ctx_rows(self.k, self.s, self.tr)
    }

    fn out_rows(&self, l: usize) -> usize {
        out_rows(self.s, self.tr, l)
    }
}

struct Block {
    c: usize,
    k: usize,
    norm: DevBuf,
    dw_w: DevBuf,
    dw_b: DevBuf,
    gamma: DevBuf,
    ffn_norm: DevBuf,
    l1: Mat,
    l2: Mat,
    ffn_gamma: DevBuf,
}

/// An encoder or the decoder
pub struct Coder {
    /// convs[i] then stages[i]'s blocks, for each stage; then the head
    convs: Vec<Conv>,
    stages: Vec<Vec<Block>>,
    head: Conv,
    eps: f32,
    pub dim_in: usize,
    pub dim_out: usize,
}

/// A stream's carried rows: each convolution's and each block's depthwise convolution's (zeros at the start)
pub struct Stream {
    ctx: Vec<DevBuf>,
}

/// The work buffers for calls of up to `rows` input rows
pub struct Work {
    pub rows: usize,
    ab: [DevBuf; 2],
    t: DevBuf,
    u: DevBuf,
    v: DevBuf,
    h: DevBuf,
    f: DevBuf,
    xh: DevBuf,
}

impl Coder {
    /// An encoder (`prefix` ...encoder.: downsample_layers.N, stages.N, head) or the decoder (...decoder.:
    /// upsample_layers.N with transposed convolutions from N = 1)
    pub fn load(ops: &Ops, f: &Weights, prefix: &str, decoder: bool, eps: f32, ratios: &[usize]) -> Result<Coder> {
        let layer = if decoder { "upsample_layers" } else { "downsample_layers" };
        let mut convs = Vec::new();
        let mut stages = Vec::new();
        for i in 0.. {
            let tr = decoder && i > 0;
            let base = if tr { format!("{prefix}{layer}.{i}.0.convtr.convtr") } else { format!("{prefix}{layer}.{i}.0.conv.conv") };
            if !f.has(&format!("{base}.weight")) {
                break;
            }
            let conv = conv(ops, f, &base, tr, i > 0)?;
            // the stride: the configuration's ratio of this stage (the decoder's in reverse)
            if i > 0 {
                let want = if decoder { ratios[ratios.len() - i] } else { ratios[i - 1] };
                if conv.s != want {
                    return Err(Error(format!("{base}: a kernel of {} (stride {}), the configuration's ratio is {want}", conv.k, conv.s)));
                }
            }
            let mut blocks = Vec::new();
            for j in 0.. {
                let n = |s: &str| format!("{prefix}stages.{i}.{j}.{s}");
                if !f.has(&n("norm.weight")) {
                    break;
                }
                let dw = n("mixer.conv.conv.conv.weight");
                let s = f.shape(&dw)?;
                if s.len() != 3 || s[0] != conv.co || s[1] != 1 {
                    return Err(Error(format!("{dw}: shape {s:?}, a depthwise kernel of {} channels expected", conv.co)));
                }
                blocks.push(Block {
                    c: conv.co,
                    k: s[2],
                    norm: f.vec(ops, &n("norm.weight"))?,
                    dw_w: f.vec(ops, &dw)?,
                    dw_b: f.vec(ops, &n("mixer.conv.conv.conv.bias"))?,
                    gamma: f.vec(ops, &n("gamma"))?,
                    ffn_norm: f.vec(ops, &n("ffn_norm.weight"))?,
                    l1: f.mat(ops, &[&n("ffn.linear1.weight")], Some(&[&n("ffn.linear1.bias")]), false)?,
                    l2: f.mat(ops, &[&n("ffn.linear2.weight")], Some(&[&n("ffn.linear2.bias")]), false)?,
                    ffn_gamma: f.vec(ops, &n("ffn_gamma"))?,
                });
            }
            convs.push(conv);
            stages.push(blocks);
        }
        if convs.len() != ratios.len() + 1 {
            return Err(Error(format!("{prefix}: {} convolutions, {} expected (a stem and one a ratio)", convs.len(), ratios.len() + 1)));
        }
        let head = conv(ops, f, &format!("{prefix}head.conv.conv"), false, false)?;
        Ok(Coder { dim_in: convs[0].ci, dim_out: head.co, convs, stages, head, eps })
    }

    /// Weights on the card (bytes)
    pub fn bytes(&self) -> usize {
        let conv = |c: &Conv| c.w.len + c.b.len;
        self.convs.iter().map(conv).sum::<usize>() + conv(&self.head)
            + self.stages.iter().flatten().map(|b| b.l1.w.len + b.l2.w.len + b.dw_w.len).sum::<usize>()
    }

    /// A fresh stream (its context zeros)
    pub fn stream(&self, ops: &Ops) -> Result<Stream> {
        let mut ctx = Vec::new();
        let zeros = |n: usize| -> Result<DevBuf> {
            let b = DevBuf::f32(&ops.gpu, n.max(1))?;
            b.fill(0)?;
            Ok(b)
        };
        for (c, blocks) in self.convs.iter().zip(&self.stages) {
            ctx.push(zeros(c.ctx() * c.ci)?);
            for b in blocks {
                ctx.push(zeros((b.k - 1) * b.c)?);
            }
        }
        ctx.push(zeros(self.head.ctx() * self.head.ci)?);
        Ok(Stream { ctx })
    }

    /// Back to the start (zeros), as upstream clears its caches at an audio end
    pub fn reset(&self, s: &Stream) -> Result<()> {
        s.ctx.iter().try_for_each(|c| c.fill(0))
    }

    /// Work buffers for calls of up to `rows` input rows (a multiple of every stride for an encoder)
    pub fn work(&self, ops: &Ops, rows: usize) -> Result<Work> {
        // the widest [rows, channels] on the way, with the carried rows, and the MLPs' 4x
        let (mut l, mut io, mut tu, mut hid) = (rows, rows * self.dim_in, 0usize, 0usize);
        for (c, blocks) in self.convs.iter().zip(&self.stages).map(|(c, b)| (c, Some(b))).chain(std::iter::once((&self.head, None))) {
            tu = tu.max((c.ctx() + l) * c.ci);
            l = c.out_rows(l);
            io = io.max(l * c.co);
            if let Some(b) = blocks.and_then(|b| b.first()) {
                hid = hid.max(l * b.l1.n);
            }
        }
        let g = &ops.gpu;
        Ok(Work {
            rows,
            ab: [DevBuf::f32(g, io)?, DevBuf::f32(g, io)?],
            t: DevBuf::f32(g, tu)?,
            u: DevBuf::f32(g, tu)?,
            v: DevBuf::f32(g, io)?,
            h: DevBuf::f32(g, io.max(1))?,
            f: DevBuf::f32(g, hid.max(io))?,
            xh: DevBuf::new(g, hid.max(io) * 2)?,
        })
    }

    /// One convolution of `l` rows of `x` with its carried rows into `out` (rows): the rows it made
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, ops: &Ops, c: &Conv, ctx: &DevBuf, x: &DevBuf, l: usize, w: &Work, out: &DevBuf) -> Result<usize> {
        let p = c.ctx();
        // [ctx; x] as rows, then the context moves on
        if p > 0 {
            w.t.copy_within(0, ctx, 0, p * c.ci * 4)?;
        }
        w.t.copy_within(p * c.ci * 4, x, 0, l * c.ci * 4)?;
        ops.ctx_push(ctx, p, x.fp(), l, c.ci)?;
        ops.transpose(w.t.fp(), p + l, c.ci, w.u.fp())?;
        let lo = c.out_rows(l);
        if c.tr {
            // the outputs of the carried inputs (cut from the start) and the overlap past the last input (cut from
            // the end) are not this call's
            ops.conv(true, w.u.fp(), c.ci, p + l, &c.w, c.co, c.k, &c.b, c.s, p * c.s, c.k - c.s, w.v.fp(), lo)?;
        } else {
            ops.conv(false, w.u.fp(), c.ci, p + l, &c.w, c.co, c.k, &c.b, c.s, 0, 0, w.v.fp(), lo)?;
        }
        ops.transpose(w.v.fp(), c.co, lo, out.fp())?;
        Ok(lo)
    }

    /// A ConvNeXt block on `x` [l, c] in place
    #[allow(clippy::too_many_arguments)]
    fn block(&self, ops: &Ops, nsd: &Nsd, b: &Block, ctx: &DevBuf, x: &DevBuf, l: usize, w: &Work) -> Result<()> {
        let c = b.c;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, c, b.norm.ptr(), self.eps, none(), none(), none(), w.h.ptr(), Dt::F32)?;
        ops.dwconv(w.h.fp(), ctx, l, c, &b.dw_w, &b.dw_b, b.k, w.f.fp())?;
        ops.ctx_push(ctx, b.k - 1, w.h.fp(), l, c)?;
        ops.scale_add(x.fp(), w.f.fp(), &b.gamma, l, c)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, c, b.ffn_norm.ptr(), self.eps, none(), none(), none(), w.h.ptr(), Dt::F32)?;
        b.l1.apply(ops, nsd, w.h.fp(), l, &w.xh, w.f.fp())?;
        nsd.gelu(w.f.ptr(), Dt::F32, l * b.l1.n, false)?;
        b.l2.apply(ops, nsd, w.f.fp(), l, &w.xh, w.h.fp())?;
        ops.scale_add(x.fp(), w.h.fp(), &b.ffn_gamma, l, c)
    }

    /// `l` new rows of `x` [l, dim_in] through the stream: its output rows [lo, dim_out], in one of the work's
    /// buffers (the returned one)
    pub fn push<'w>(&self, ops: &Ops, nsd: &Nsd, s: &Stream, x: &DevBuf, l: usize, w: &'w Work) -> Result<(&'w DevBuf, usize)> {
        if l > w.rows {
            return Err(Error(format!("{l} rows through work buffers for {}", w.rows)));
        }
        let mut ci = 0;
        let mut rows = l;
        let mut cur: Option<usize> = None;
        for (c, blocks) in self.convs.iter().zip(&self.stages) {
            if !c.tr && !rows.is_multiple_of(c.s) {
                return Err(Error(format!("{rows} rows into a convolution of stride {}", c.s)));
            }
            let dst = cur.map_or(0, |i| 1 - i);
            let src = cur.map_or(x, |i| &w.ab[i]);
            rows = self.conv(ops, c, &s.ctx[ci], src, rows, w, &w.ab[dst])?;
            ci += 1;
            cur = Some(dst);
            for b in blocks {
                self.block(ops, nsd, b, &s.ctx[ci], &w.ab[dst], rows, w)?;
                ci += 1;
            }
        }
        let i = cur.unwrap_or(0);
        let rows = self.conv(ops, &self.head, &s.ctx[ci], &w.ab[i], rows, w, &w.ab[1 - i])?;
        Ok((&w.ab[1 - i], rows))
    }
}

/// A convolution's weights (float32: oneDNN's convolutions take them so) and its kernel / stride from their shape:
/// a strided one has a kernel of twice its stride
fn conv(ops: &Ops, f: &Weights, base: &str, tr: bool, strided: bool) -> Result<Conv> {
    let wn = format!("{base}.weight");
    let s = f.shape(&wn)?;
    if s.len() != 3 {
        return Err(Error(format!("{wn}: shape {s:?}, a 1-D convolution's expected")));
    }
    let (ci, co) = if tr { (s[0], s[1]) } else { (s[1], s[0]) };
    let k = s[2];
    let stride = if strided { k / 2 } else { 1 };
    Ok(Conv { w: f.vec(ops, &wn)?, b: f.vec(ops, &format!("{base}.bias"))?, ci, co, k, s: stride, tr })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(k: usize, s: usize, tr: bool) -> (usize, usize) {
        (ctx_rows(k, s, tr), out_rows(s, tr, 8))
    }

    #[test]
    fn a_stream_carries_what_its_next_outputs_read() {
        // the stem and heads (kernel 7): six rows; the encoder's stride-2 convolution of 4: two rows of padding, 4 out of 8
        assert_eq!(c(7, 1, false), (6, 8));
        assert_eq!(c(4, 2, false), (2, 4));
        // the decoder's transposed convolution of 16, stride 8: the input before is all the next outputs overlap
        assert_eq!(c(16, 8, true), (1, 64));
        assert_eq!(c(10, 5, true), (1, 40));
    }
}
