//! The σ-VAE tokenizers, streaming: the acoustic encoder (a voice's recording to 64-value latents at 7.5 Hz), the
//! acoustic decoder (a latent to 3,200 samples at 24 kHz) and the semantic encoder (each frame's samples back to 128
//! values for the language model). Each is a stem convolution, then stages of ConvNeXt blocks (RMS norm over channels,
//! a depthwise causal convolution of 7, a layer scale; RMS norm, a GELU MLP of 4x, a layer scale) joined by strided
//! convolutions (the encoders: kernel 2r, stride r) or transposed ones (the decoder), then a head convolution. Every
//! convolution is causal, so a stream that carries each one's last inputs (its context: zeros at the start) gives
//! what the whole clip would - upstream decodes and re-encodes a frame at a time this way.
//!
//! The signal runs as time rows [L, C]. Each convolution reads a buffer of its carried rows followed by its new input
//! rows ([ctx; x]: the stage before writes its rows there), so a window of k input rows is k * C consecutive floats and
//! the convolution is a product over windows (`ops.prod`): a strided one's windows start s rows apart (weights
//! [co, k * ci]); a transposed one of kernel 2s makes s output rows from each pair of input rows (weights
//! [s * co, 2 * ci]). Its carried rows then move to the front (`shift`). A block's depthwise convolution carries its
//! last normed rows the same way, in two buffers that alternate (the one a call writes is not the one it reads). The
//! weights are the file's values exactly: bfloat16 (or float32), products in float32.

use std::cell::Cell;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Nsd;

use crate::ops::{fp, split_len, Epi, Mat, Ops, Split};
use crate::weights::Weights;

/// A channel-mixing causal convolution (or a transposed one) as a product over windows of its input rows
struct Conv {
    w: Mat,
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

    /// The windows over [ctx; l rows]: (how many, the floats between their starts)
    fn windows(&self, l: usize) -> (usize, usize) {
        if self.tr {
            (l, self.ci)
        } else {
            (l / self.s, self.s * self.ci)
        }
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

/// A stream: each convolution's input buffer [its carried rows; up to the call's rows] (a stage's rows live in the
/// next convolution's), each block's two alternating buffers of normed rows, the work buffers, and the output
pub struct Stream {
    pub rows: usize,
    cin: Vec<DevBuf>,
    hist: Vec<[DevBuf; 2]>,
    /// the rows each block's last call added (where its carried rows start in the buffer it wrote)
    prev: Vec<Cell<usize>>,
    parity: Cell<usize>,
    h: DevBuf,
    f: DevBuf,
    split: Split,
    out: DevBuf,
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
                if s.len() != 3 || s[0] != conv.co || s[1] != 1 || s[2] > 32 {
                    return Err(Error(format!("{dw}: shape {s:?}, a depthwise kernel of {} channels (at most 32 taps) expected", conv.co)));
                }
                blocks.push(Block {
                    c: conv.co,
                    k: s[2],
                    norm: f.vec(ops, &n("norm.weight"))?,
                    dw_w: f.vec(ops, &dw)?,
                    dw_b: f.vec(ops, &n("mixer.conv.conv.conv.bias"))?,
                    gamma: f.vec(ops, &n("gamma"))?,
                    ffn_norm: f.vec(ops, &n("ffn_norm.weight"))?,
                    l1: f.mat_exact(ops, &n("ffn.linear1.weight"), Some(&n("ffn.linear1.bias")))?,
                    l2: f.mat_exact(ops, &n("ffn.linear2.weight"), Some(&n("ffn.linear2.bias")))?,
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
        self.convs.iter().chain(std::iter::once(&self.head)).map(|c| c.w.bytes()).sum::<usize>()
            + self.stages.iter().flatten().map(|b| b.l1.bytes() + b.l2.bytes() + b.dw_w.len).sum::<usize>()
    }

    /// The convolutions in order, the head last
    fn all_convs(&self) -> impl Iterator<Item = &Conv> {
        self.convs.iter().chain(std::iter::once(&self.head))
    }

    /// A fresh stream (its carried rows zeros) for calls of up to `rows` input rows (a multiple of every stride for
    /// an encoder)
    pub fn stream(&self, ops: &Ops, rows: usize) -> Result<Stream> {
        let g = &ops.gpu;
        let zeros = |n: usize| -> Result<DevBuf> {
            let b = DevBuf::f32(g, n.max(4))?;
            b.fill(0)?;
            Ok(b)
        };
        let (mut cin, mut hist) = (Vec::new(), Vec::new());
        let (mut l, mut h, mut fw, mut sp, mut pt) = (rows, 0usize, 0usize, 0usize, 0usize);
        for (i, c) in self.all_convs().enumerate() {
            cin.push(zeros((c.ctx() + l) * c.ci)?);
            let (m, ldx) = c.windows(l);
            sp = sp.max(split_len(m, ldx, c.w.k));
            pt = pt.max(ops.gemm_scratch(m, c.w.n, c.w.k));
            l = c.out_rows(l);
            if let Some(blocks) = self.stages.get(i) {
                for b in blocks {
                    hist.push([zeros((b.k - 1 + l) * b.c)?, zeros((b.k - 1 + l) * b.c)?]);
                    h = h.max(l * b.c);
                    fw = fw.max(l * b.l1.n);
                    sp = sp.max(split_len(l, b.c, b.c)).max(split_len(l, b.l1.n, b.l1.n));
                    pt = pt.max(ops.gemm_scratch(l, b.l1.n, b.c)).max(ops.gemm_scratch(l, b.c, b.l1.n));
                }
            }
        }
        // the split inputs of the matrix engines' products (bfloat16 pairs), zeros past what a call writes
        let split = Split { hi: zeros(sp.div_ceil(2))?, lo: zeros(sp.div_ceil(2))?, part: DevBuf::f32(g, pt.max(4))? };
        Ok(Stream { rows, prev: (0..hist.len()).map(|_| Cell::new(0)).collect(), hist, cin, parity: Cell::new(0), h: DevBuf::f32(g, h.max(4))?,
                    f: DevBuf::f32(g, fw.max(4))?, out: DevBuf::f32(g, (l * self.head.co).max(4))?, split })
    }

    /// Back to the start (zeros), as upstream clears its caches at an audio end
    pub fn reset(&self, s: &Stream) -> Result<()> {
        s.cin.iter().chain(s.hist.iter().flatten()).try_for_each(|c| c.fill(0))?;
        s.prev.iter().for_each(|p| p.set(0));
        Ok(())
    }

    /// The buffer the stream's next input rows go into (after the stem's carried rows), and its first float
    pub fn input<'s>(&self, s: &'s Stream) -> (&'s DevBuf, usize) {
        (&s.cin[0], self.convs[0].ctx() * self.convs[0].ci)
    }

    /// `l` new rows of `x` [l, dim_in] through the stream (copied into its input first; None: already there, see
    /// `input`): its output rows [lo, dim_out]
    pub fn push<'s>(&self, ops: &Ops, _nsd: &Nsd, s: &'s Stream, x: Option<&DevBuf>, l: usize) -> Result<(&'s DevBuf, usize)> {
        if l > s.rows {
            return Err(Error(format!("{l} rows through a stream for {}", s.rows)));
        }
        if let Some(x) = x {
            let (b, at) = self.input(s);
            b.copy_within(at * 4, x, 0, l * self.dim_in * 4)?;
        }
        let par = s.parity.get();
        let mut rows = l;
        let mut bi = 0;
        let convs: Vec<&Conv> = self.all_convs().collect();
        for (i, c) in convs.iter().enumerate() {
            if !c.tr && !rows.is_multiple_of(c.s) {
                return Err(Error(format!("{rows} rows into a convolution of stride {}", c.s)));
            }
            // the output: the next convolution's input buffer after its carried rows (the head's: the stream's output)
            let (dst, ldo) = match convs.get(i + 1) {
                Some(n) => (fp(&s.cin[i + 1], n.ctx() * n.ci), c.w.n),
                None => (s.out.fp(), c.w.n),
            };
            let (m, ldx) = c.windows(rows);
            ops.mark(|| "(before)".into())?;
            ops.prod(s.cin[i].fp(), m, ldx, &c.w, dst, ldo, Epi::Store, None, Some(&s.split))?;
            ops.mark(|| format!("{i} conv [{m}, {}] x {}", c.w.k, c.w.n))?;
            if c.ctx() > 0 {
                ops.shift(s.cin[i].fp(), c.ctx(), rows, c.ci)?;
            }
            ops.mark(|| "shift".into())?;
            rows = c.out_rows(rows);
            for b in self.stages.get(i).into_iter().flatten() {
                let (prev, cur) = (&s.hist[bi][1 - par], &s.hist[bi][par]);
                let ch = b.c;
                ops.norm_hist(prev, s.prev[bi].get(), dst, rows, ch, b.k - 1, &b.norm, self.eps, cur)?;
                ops.block_mix(cur, dst, rows, ch, b.k, &b.dw_w, &b.dw_b, &b.gamma, &b.ffn_norm, self.eps, s.h.fp())?;
                ops.mark(|| format!("{i} norms + depthwise [{rows}, {ch}]"))?;
                ops.prod(s.h.fp(), rows, ch, &b.l1, s.f.fp(), b.l1.n, Epi::Gelu, None, Some(&s.split))?;
                ops.mark(|| format!("{i} mlp1 [{rows}, {ch}] x {}", b.l1.n))?;
                ops.prod(s.f.fp(), rows, b.l1.n, &b.l2, dst, ch, Epi::Res, Some((b.ffn_gamma.fp(), 0)), Some(&s.split))?;
                ops.mark(|| format!("{i} mlp2 [{rows}, {}] x {ch}", b.l1.n))?;
                s.prev[bi].set(rows);
                bi += 1;
            }
        }
        s.parity.set(1 - par);
        Ok((&s.out, rows))
    }
}

/// A convolution's weights as a product over windows of its input rows (the file's values: bfloat16, or float32) and
/// its kernel / stride from their shape: a strided one has a kernel of twice its stride. A strided or plain one
/// [co, ci, k] becomes [co, k * ci]; a transposed one [ci, co, 2s] becomes [s * co, 2 * ci]: output row r of a window
/// (input rows i - 1, i) takes tap r + s of row i - 1 and tap r of row i.
fn conv(ops: &Ops, f: &Weights, base: &str, tr: bool, strided: bool) -> Result<Conv> {
    let wn = format!("{base}.weight");
    let sh = f.shape(&wn)?;
    if sh.len() != 3 {
        return Err(Error(format!("{wn}: shape {sh:?}, a 1-D convolution's expected")));
    }
    let (ci, co) = if tr { (sh[0], sh[1]) } else { (sh[1], sh[0]) };
    let k = sh[2];
    let s = if strided { k / 2 } else { 1 };
    let v = f.all(&wn)?;
    let bias = f.all(&format!("{base}.bias"))?;
    let (w, n, kk, b) = if tr {
        if k != 2 * s {
            return Err(Error(format!("{wn}: a transposed kernel of {k}, stride {s} (2s expected)")));
        }
        let mut w = vec![0f32; s * co * 2 * ci];
        for r in 0..s {
            for c in 0..co {
                let row = &mut w[(r * co + c) * 2 * ci..(r * co + c + 1) * 2 * ci];
                for i in 0..ci {
                    row[i] = v[(i * co + c) * k + r + s];
                    row[ci + i] = v[(i * co + c) * k + r];
                }
            }
        }
        let b: Vec<f32> = (0..s).flat_map(|_| bias.iter().copied()).collect();
        (w, s * co, 2 * ci, b)
    } else {
        let mut w = vec![0f32; co * k * ci];
        for c in 0..co {
            for t in 0..k {
                for i in 0..ci {
                    w[c * k * ci + t * ci + i] = v[(c * ci + i) * k + t];
                }
            }
        }
        (w, co, k * ci, bias)
    };
    let w = f.mat_host(ops, &w, n, kk, f.is_bf16(&wn), Some(&b))?;
    Ok(Conv { w, ci, co, k, s, tr })
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
