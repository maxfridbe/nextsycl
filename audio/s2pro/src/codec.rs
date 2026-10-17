//! Fish's codec (fish_speech/models/dac/modded_dac.py and rvq.py, configs/modded_dac_vq.yaml): ten codes a frame, 2,048
//! samples of 44.1 kHz audio (21.5 frames a second). Its weights are in the model's GGUF (`c.`-prefixed, F16, weight
//! norm already folded), in float32 here as the reference runs it.
//!
//! ```text
//!   decode  codes [10, T] -> the semantic codebook (4,096 x 8) and nine residual ones (1,024 x 8), each through its
//!           1x1 projection to 1,024, summed -> post transformer (8 layers of 1,024, 16 heads of 64, a 128-frame causal
//!           window, layer-scaled residuals, SwiGLU 3,072) -> 2 x (transposed conv x2 + ConvNeXt) -> causal conv k7 to
//!           1,536 -> 4 blocks (Snake, transposed conv x8 / x8 / x4 / x2 halving the channels, residual units at
//!           dilations 1, 3, 9) -> Snake -> causal conv k7 to 1 -> tanh
//!   encode  44.1 kHz samples (padded to whole frames) -> causal conv k7 to 64 -> 4 blocks (residual units, Snake, a
//!           strided causal conv x2 / x4 / x8 / x8 doubling the channels; the last with a 4-layer transformer, a
//!           512-step window) -> Snake -> causal conv k3 -> 2 x (strided conv x2 + ConvNeXt) -> pre transformer ->
//!           the semantic quantizer, then nine residual ones (cosine nearest in 8 dimensions)
//! ```
//!
//! The quantizers run on the host (their projections are 8 wide). The decoder's convolution stack goes in chunks of
//! frames, each with frames of the one before as context (cut from its output): the stack is causal with a reach of a
//! few frames, so the chunks join as the whole clip would come out.

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, rope_table, Lin, Ops};
use crate::weights::Weights;

pub const RATE: u32 = 44100;
/// samples a frame
pub const HOP: usize = 2048;
const HEADS: usize = 16;
const HEAD: usize = 64;
const EPS: f32 = 1e-5;
/// frames a chunk of the decoder's convolutions, and the frames of context ahead of it
const CHUNK: usize = 200;
const CONTEXT: usize = 16;

/// A float32 convolution: w [co, ci, k] (transposed: [ci, co, k]), bias [co]
pub struct Conv {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub ci: usize,
    pub co: usize,
    pub k: usize,
}

impl Conv {
    fn load(ops: &Ops, f: &Weights, name: &str, transposed: bool) -> Result<Conv> {
        let mut s = f.shape(&format!("{name}.weight"))?;
        // one output channel: the converter's shape drops the leading 1 ([in, k])
        if s.len() == 2 && !transposed {
            s.insert(0, 1);
        }
        if s.len() != 3 {
            return Err(Error(format!("{name}.weight: shape {s:?}, a convolution's [out, in, k] expected")));
        }
        let (ci, co) = if transposed { (s[0], s[1]) } else { (s[1], s[0]) };
        let bias = format!("{name}.bias");
        Ok(Conv { w: f.dev(ops, &format!("{name}.weight"))?, b: if f.has(&bias) { Some(f.dev(ops, &bias)?) } else { None }, ci, co, k: s[2] })
    }

    fn bias(&self) -> *const f32 {
        self.b.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }
}

/// DAC's residual unit: x + conv k1 (Snake (conv k7 dilated (Snake x)))
struct Unit {
    a1: DevBuf,
    c1: Conv,
    a2: DevBuf,
    c2: Conv,
    dil: usize,
}

impl Unit {
    fn load(ops: &Ops, f: &Weights, p: &str, dil: usize) -> Result<Unit> {
        Ok(Unit { a1: f.dev(ops, &format!("{p}.block.0.alpha"))?, c1: Conv::load(ops, f, &format!("{p}.block.1.conv"), false)?,
                  a2: f.dev(ops, &format!("{p}.block.2.alpha"))?, c2: Conv::load(ops, f, &format!("{p}.block.3.conv"), false)?, dil })
    }

    /// x [c, l] in place; y, z work buffers of c * l
    fn run(&self, ops: &Ops, nsd: &Nsd, x: &DevBuf, y: &DevBuf, z: &DevBuf, l: usize) -> Result<()> {
        let c = self.c1.ci;
        nsd.snake(x.ptr(), 1, c, l, self.a1.ptr(), y.ptr())?;
        ops.conv_causal(y.fp(), c, l, &self.c1.w, c, self.c1.k, self.c1.bias(), self.dil, z.fp())?;
        nsd.snake(z.ptr(), 1, c, l, self.a2.ptr(), y.ptr())?;
        ops.conv_causal(y.fp(), c, l, &self.c2.w, c, self.c2.k, self.c2.bias(), 1, z.fp())?;
        ops.add(x.fp(), z.fp(), c * l)
    }
}

/// A transformer block of the codec (WindowLimitedTransformer: causal, a window of `window` steps)
struct TLayer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Lin,
    o: Lin,
    gu: Lin,
    down: Lin,
    attn_scale: DevBuf,
    mlp_scale: DevBuf,
}

struct Transformer {
    layers: Vec<TLayer>,
    norm: DevBuf,
    window: usize,
}

impl Transformer {
    fn load(ops: &Ops, f: &Weights, p: &str, window: usize) -> Result<Transformer> {
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("{p}.layers.{l}.{s}");
            if !f.has(&n("attention_norm.weight")) {
                break;
            }
            layers.push(TLayer {
                in_norm: f.dev(ops, &n("attention_norm.weight"))?,
                post_norm: f.dev(ops, &n("ffn_norm.weight"))?,
                qkv: f.lin(ops, &n("attention.wqkv"), false)?,
                o: f.lin(ops, &n("attention.wo"), false)?,
                gu: f.lin_stack(ops, &[n("feed_forward.w1.weight"), n("feed_forward.w3.weight")])?,
                down: f.lin(ops, &n("feed_forward.w2"), false)?,
                attn_scale: f.dev(ops, &n("attention_layer_scale.gamma"))?,
                mlp_scale: f.dev(ops, &n("ffn_layer_scale.gamma"))?,
            });
        }
        let first = layers.first().ok_or_else(|| Error(format!("no {p}.layers in the file")))?;
        if first.qkv.n != 3 * HEADS * HEAD || first.o.n != HEADS * HEAD {
            return Err(Error(format!("{p}: a transformer of {} wide, {} expected", first.o.n, HEADS * HEAD)));
        }
        Ok(Transformer { layers, norm: f.dev(ops, &format!("{p}.norm.weight"))?, window })
    }

    /// Rows x [l, 1,024] through the layers, in place; `cs` the rotary table of at least l positions
    fn run(&self, ops: &Ops, nsd: &Nsd, x: &DevBuf, l: usize, cs: &DevBuf) -> Result<()> {
        let g = &ops.gpu;
        let h = HEADS * HEAD;
        let ffn = self.layers[0].gu.n / 2;
        let hb = DevBuf::f32(g, l * h)?;
        let qkv = DevBuf::f32(g, l * 2 * ffn.max(3 * h / 2))?;
        let att = DevBuf::f32(g, l * ffn.max(h))?;
        for ly in &self.layers {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, h, ly.in_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
            ly.qkv.apply(nsd, hb.fp(), l, qkv.fp())?;
            ops.norm_rope(qkv.fp(), 3 * h, l, l, HEADS, HEAD, None, EPS, cs, 0)?;
            ops.norm_rope(fp(&qkv, h), 3 * h, l, l, HEADS, HEAD, None, EPS, cs, 0)?;
            ops.window_attn(qkv.fp(), fp(&qkv, h), fp(&qkv, 2 * h), l, HEADS, HEAD, 3 * h, 3 * h, self.window, att.fp())?;
            ly.o.apply(nsd, att.fp(), l, hb.fp())?;
            ops.scale_add(x.fp(), hb.fp(), &ly.attn_scale, l, h)?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, h, ly.post_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
            ly.gu.apply(nsd, hb.fp(), l, qkv.fp())?;
            nsd.swiglu(qkv.ptr(), Dt::F32, l, ffn, att.ptr(), Dt::F32)?;
            ly.down.apply(nsd, att.fp(), l, hb.fp())?;
            ops.scale_add(x.fp(), hb.fp(), &ly.mlp_scale, l, h)?;
        }
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, h, self.norm.ptr(), EPS, none(), none(), none(), x.ptr(), Dt::F32)
    }
}

/// A resampling stage of the quantizer: a (transposed or strided) conv by 2, then a ConvNeXt block (a depthwise causal
/// conv k7, layer norm, 4x MLP with exact GELU, gamma-scaled residual)
struct Next {
    conv: Conv,
    dw_w: DevBuf,
    dw_b: DevBuf,
    norm_w: DevBuf,
    norm_b: DevBuf,
    pw1: Lin,
    pw2: Lin,
    gamma: DevBuf,
}

impl Next {
    fn load(ops: &Ops, f: &Weights, p: &str, transposed: bool) -> Result<Next> {
        Ok(Next {
            conv: Conv::load(ops, f, &format!("{p}.0.conv"), transposed)?,
            dw_w: f.dev(ops, &format!("{p}.1.dwconv.conv.weight"))?,
            dw_b: f.dev(ops, &format!("{p}.1.dwconv.conv.bias"))?,
            norm_w: f.dev(ops, &format!("{p}.1.norm.weight"))?,
            norm_b: f.dev(ops, &format!("{p}.1.norm.bias"))?,
            pw1: f.lin(ops, &format!("{p}.1.pwconv1"), true)?,
            pw2: f.lin(ops, &format!("{p}.1.pwconv2"), true)?,
            gamma: f.dev(ops, &format!("{p}.1.gamma"))?,
        })
    }

    /// x [c, l] -> its conv (up: l * 2; down: l / 2) and the ConvNeXt block, in x; its length returned
    fn run(&self, ops: &Ops, nsd: &Nsd, x: &DevBuf, l: usize, up: bool) -> Result<usize> {
        let g = &ops.gpu;
        let c = self.conv.co;
        let lo = if up { l * self.conv.k } else { l.div_ceil(self.conv.k) };
        let a = DevBuf::f32(g, c * lo)?;
        let b = DevBuf::f32(g, c * lo * 4)?;
        let r = DevBuf::f32(g, c * lo)?;
        if up {
            ops.conv_up(x.fp(), self.conv.ci, l, &self.conv.w, c, self.conv.k, self.conv.bias(), self.conv.k, a.fp())?;
        } else {
            ops.conv_strided(x.fp(), self.conv.ci, l, &self.conv.w, c, self.conv.k, self.conv.bias(), self.conv.k, a.fp())?;
        }
        // a: the block's input [c, lo]; its depthwise conv, then rows [lo, c]
        ops.dwconv(a.fp(), c, lo, &self.dw_w, &self.dw_b, 7, x.fp())?;
        ops.transpose(x.fp(), c, lo, r.fp())?;
        nsd.layer_norm(r.ptr(), Dt::F32, lo, c, self.norm_w.ptr(), self.norm_b.ptr(), 1e-6, x.ptr(), Dt::F32)?;
        self.pw1.apply(nsd, x.fp(), lo, b.fp())?;
        nsd.gelu(b.ptr(), Dt::F32, lo * self.pw1.n, false)?;
        self.pw2.apply(nsd, b.fp(), lo, x.fp())?;
        // the input as rows, + gamma * the MLP's, back to [c, lo]
        ops.transpose(a.fp(), c, lo, r.fp())?;
        ops.scale_add(r.fp(), x.fp(), &self.gamma, lo, c)?;
        ops.transpose(r.fp(), lo, c, x.fp())?;
        Ok(lo)
    }
}

/// A codebook on the host: its rows normalized for the nearest search, the in projection [8, 1,024] + bias, and each
/// row's out projection (to 1,024, with the bias) precomputed
struct Book {
    norm: Vec<f32>,
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out: Vec<f32>,
    size: usize,
}

const CD: usize = 8;

impl Book {
    fn load(f: &Weights, p: &str, dim: usize) -> Result<Book> {
        let rows = f.f32(&format!("{p}.codebook.weight"))?;
        let size = rows.len() / CD;
        let in_w = f.f32(&format!("{p}.in_proj.weight"))?;
        let in_b = f.f32(&format!("{p}.in_proj.bias"))?;
        let out_w = f.f32(&format!("{p}.out_proj.weight"))?;
        let out_b = f.f32(&format!("{p}.out_proj.bias"))?;
        if in_w.len() != CD * dim || out_w.len() != dim * CD || rows.len() != size * CD {
            return Err(Error(format!("{p}: a quantizer of {} x {CD} from {dim} expected", size)));
        }
        // F.normalize: x / max(|x|, 1e-12)
        let norm = rows.chunks(CD).flat_map(|r| {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            r.iter().map(move |v| v / n)
        }).collect();
        let mut out = vec![0f32; size * dim];
        for (i, r) in rows.chunks(CD).enumerate() {
            for (o, (w, b)) in out[i * dim..(i + 1) * dim].iter_mut().zip(out_w.chunks(CD).zip(&out_b)) {
                *o = b + w.iter().zip(r).map(|(a, x)| a * x).sum::<f32>();
            }
        }
        Ok(Book { norm, in_w, in_b, out, size })
    }

    /// The nearest code of a 1,024-wide vector (DAC's: cosine, as the distance of the normalized vectors)
    fn nearest(&self, z: &[f32]) -> usize {
        let mut e = [0f32; CD];
        for (j, v) in e.iter_mut().enumerate() {
            *v = self.in_b[j] + self.in_w[j * z.len()..(j + 1) * z.len()].iter().zip(z).map(|(a, b)| a * b).sum::<f32>();
        }
        let n = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        e.iter_mut().for_each(|v| *v /= n);
        let ee: f32 = e.iter().map(|v| v * v).sum();
        let mut best = (0, f32::INFINITY);
        for (i, c) in self.norm.chunks(CD).enumerate() {
            let (dot, cc) = c.iter().zip(&e).fold((0f32, 0f32), |(d, s), (a, b)| (d + a * b, s + a * a));
            let dist = ee - 2.0 * dot + cc;
            if dist < best.1 {
                best = (i, dist);
            }
        }
        best.0
    }
}

/// A decoder block: Snake, a transposed conv by `stride` (halving the channels), three residual units
struct DecBlock {
    act: DevBuf,
    up: Conv,
    stride: usize,
    units: Vec<Unit>,
}

/// An encoder block: three residual units, Snake, a strided conv by `stride` (doubling the channels), a transformer
struct EncBlock {
    units: Vec<Unit>,
    act: DevBuf,
    down: Conv,
    stride: usize,
    tr: Option<Transformer>,
}

pub struct Codec {
    ops: Ops,
    dim: usize,
    semantic: Book,
    residual: Vec<Book>,
    post: Transformer,
    pre: Transformer,
    ups: Vec<Next>,
    downs: Vec<Next>,
    first: Conv,
    blocks: Vec<DecBlock>,
    last_act: DevBuf,
    last: Conv,
    enc_first: Conv,
    enc_blocks: Vec<EncBlock>,
    enc_act: DevBuf,
    enc_last: Conv,
    /// the rotary table (θ 1e4, heads of 64), positions as long as the longest sequence
    cs: std::sync::Mutex<(usize, DevBuf)>,
}

impl Codec {
    pub fn load(gpu: &Arc<Gpu>, f: &Weights) -> Result<Codec> {
        let ops = &Ops::new(gpu)?;
        let dim = f.num("codec.latent_dim")? as usize;
        let q = "c.quantizer";
        let semantic = Book::load(f, &format!("{q}.semantic_quantizer.quantizers.0"), dim)?;
        let residual = (0..f.num("codec.quantizer_residual_codebooks")? as usize)
            .map(|i| Book::load(f, &format!("{q}.quantizer.quantizers.{i}"), dim)).collect::<Result<Vec<_>>>()?;
        let window = f.num("codec.rvq_transformer.window_size")? as usize;
        let post = Transformer::load(ops, f, &format!("{q}.post_module"), window)?;
        let pre = Transformer::load(ops, f, &format!("{q}.pre_module"), window)?;
        let ups = (0..2).map(|i| Next::load(ops, f, &format!("{q}.upsample.{i}"), true)).collect::<Result<Vec<_>>>()?;
        let downs = (0..2).map(|i| Next::load(ops, f, &format!("{q}.downsample.{i}"), false)).collect::<Result<Vec<_>>>()?;
        let d = "c.decoder.model";
        let first = Conv::load(ops, f, &format!("{d}.0.conv"), false)?;
        let mut blocks = Vec::new();
        let mut i = 1;
        while f.has(&format!("{d}.{i}.block.1.conv.weight")) {
            let b = format!("{d}.{i}.block");
            let up = Conv::load(ops, f, &format!("{b}.1.conv"), true)?;
            let stride = up.k / 2;
            let units = [1, 3, 9].iter().enumerate().map(|(j, dil)| Unit::load(ops, f, &format!("{b}.{}", j + 2), *dil)).collect::<Result<Vec<_>>>()?;
            blocks.push(DecBlock { act: f.dev(ops, &format!("{b}.0.alpha"))?, up, stride, units });
            i += 1;
        }
        let last_act = f.dev(ops, &format!("{d}.{i}.alpha"))?;
        let last = Conv::load(ops, f, &format!("{d}.{}.conv", i + 1), false)?;
        let e = "c.encoder.block";
        let enc_first = Conv::load(ops, f, &format!("{e}.0.conv"), false)?;
        let mut enc_blocks = Vec::new();
        let mut i = 1;
        while f.has(&format!("{e}.{i}.block.4.conv.weight")) {
            let b = format!("{e}.{i}.block");
            let units = [1, 3, 9].iter().enumerate().map(|(j, dil)| Unit::load(ops, f, &format!("{b}.{j}"), *dil)).collect::<Result<Vec<_>>>()?;
            let down = Conv::load(ops, f, &format!("{b}.4.conv"), false)?;
            let stride = down.k / 2;
            // the encoder's transformers: a 512-step window (modded_dac's EncoderBlock default)
            let tr = if f.has(&format!("{b}.5.layers.0.attention_norm.weight")) { Some(Transformer::load(ops, f, &format!("{b}.5"), 512)?) } else { None };
            enc_blocks.push(EncBlock { units, act: f.dev(ops, &format!("{b}.3.alpha"))?, down, stride, tr });
            i += 1;
        }
        let enc_act = f.dev(ops, &format!("{e}.{i}.alpha"))?;
        let enc_last = Conv::load(ops, f, &format!("{e}.{}.conv", i + 1), false)?;
        let cs = DevBuf::from_f32(gpu, &rope_table(4096, HEAD, 10000.0))?;
        let c = Codec { ops: ops.clone(), dim, semantic, residual, post, pre, ups, downs, first, blocks, last_act, last, enc_first, enc_blocks, enc_act, enc_last,
                        cs: std::sync::Mutex::new((4096, cs)) };
        if c.hop() != HOP || c.enc_hop() * 4 != HOP {
            return Err(Error(format!("the codec makes {} samples a frame, {HOP} expected", c.hop())));
        }
        gpu.sync()?;
        Ok(c)
    }

    /// Samples a frame (the transposed convolutions' strides)
    pub fn hop(&self) -> usize {
        self.ups.iter().map(|u| u.conv.k).product::<usize>() * self.blocks.iter().map(|b| b.stride).product::<usize>()
    }

    fn enc_hop(&self) -> usize {
        self.enc_blocks.iter().map(|b| b.stride).product()
    }

    /// The rotary table of at least `l` positions
    fn table(&self, l: usize) -> Result<std::sync::MutexGuard<'_, (usize, DevBuf)>> {
        let mut t = self.cs.lock().map_err(|_| Error("the codec's rotary table".into()))?;
        if t.0 < l {
            let n = l.next_power_of_two();
            *t = (n, DevBuf::from_f32(&self.ops.gpu, &rope_table(n, HEAD, 10000.0))?);
        }
        Ok(t)
    }

    /// The quantizers' sum of frames [T][10] (clamped into their codebooks, as upstream's decode): [1,024, T] host
    pub fn dequantize(&self, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        let (d, t) = (self.dim, frames.len());
        let mut z = vec![0f32; d * t];
        for (fi, fr) in frames.iter().enumerate() {
            if fr.len() != 1 + self.residual.len() {
                return Err(Error(format!("a frame of {} codes; the codec takes {}", fr.len(), 1 + self.residual.len())));
            }
            for (q, &c) in fr.iter().enumerate() {
                let b = if q == 0 { &self.semantic } else { &self.residual[q - 1] };
                let c = c.clamp(0, b.size as i32 - 1) as usize;
                for (j, v) in b.out[c * d..(c + 1) * d].iter().enumerate() {
                    z[j * t + fi] += v;
                }
            }
        }
        Ok(z)
    }

    /// The quantizer's decode on the GPU: [1,024, T] -> the post transformer -> upsampled [1,024, 4T] (in a returned
    /// buffer, channels first)
    pub fn latents(&self, nsd: &Nsd, frames: &[Vec<i32>]) -> Result<(DevBuf, usize)> {
        let (ops, g, d) = (&self.ops, &self.ops.gpu, self.dim);
        let t = frames.len();
        let z = DevBuf::from_f32(g, &self.dequantize(frames)?)?;
        let rows = DevBuf::f32(g, d * t * 4)?;
        ops.transpose(z.fp(), d, t, rows.fp())?;
        self.post.run(ops, nsd, &rows, t, &self.table(t)?.1)?;
        let x = DevBuf::f32(g, d * t * 4)?;
        ops.transpose(rows.fp(), t, d, x.fp())?;
        let mut l = t;
        for u in &self.ups {
            l = u.run(ops, nsd, &x, l, true)?;
        }
        Ok((x, l))
    }

    /// Frames [T][10] to samples (44.1 kHz mono)
    pub fn decode(&self, nsd: &Nsd, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        let (ops, g) = (&self.ops, &self.ops.gpu);
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let (x, l) = self.latents(nsd, frames)?;
        let up = l / frames.len();
        let c0 = self.first.ci;
        // the latents as rows [l, c0]: a chunk is a contiguous slice
        let rows = DevBuf::f32(g, l * c0)?;
        ops.transpose(x.fp(), c0, l, rows.fp())?;
        drop(x);
        // the widest signal of a chunk: a block's output, or the first conv's
        let lc = (CHUNK + CONTEXT) * up;
        let mut widest = self.first.co * lc;
        let mut len = lc;
        for b in &self.blocks {
            len *= b.stride;
            widest = widest.max(b.up.co * len);
        }
        let (a, b, c) = (DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?);
        let mut out = Vec::with_capacity(frames.len() * HOP);
        let mut s = 0;
        while s < frames.len() {
            let e = (s + CHUNK).min(frames.len());
            let ctx = s.min(CONTEXT);
            let (p0, n) = ((s - ctx) * up, (e - s + ctx) * up);
            ops.transpose(fp(&rows, p0 * c0), n, c0, a.fp())?;
            let wav = self.stack(nsd, &a, &b, &c, n)?;
            out.extend_from_slice(&wav[ctx * HOP..]);
            s = e;
        }
        Ok(out)
    }

    /// The decoder's convolutions over x [c0, n] (in a): samples
    fn stack(&self, nsd: &Nsd, a: &DevBuf, b: &DevBuf, c: &DevBuf, n: usize) -> Result<Vec<f32>> {
        let ops = &self.ops;
        let f = &self.first;
        ops.conv_causal(a.fp(), f.ci, n, &f.w, f.co, f.k, f.bias(), 1, b.fp())?;
        let mut l = n;
        // b: the signal; a, c: work
        for blk in &self.blocks {
            let up = &blk.up;
            nsd.snake(b.ptr(), 1, up.ci, l, blk.act.ptr(), a.ptr())?;
            ops.conv_up(a.fp(), up.ci, l, &up.w, up.co, up.k, up.bias(), blk.stride, b.fp())?;
            l *= blk.stride;
            for u in &blk.units {
                u.run(ops, nsd, b, a, c, l)?;
            }
        }
        let k = &self.last;
        nsd.snake(b.ptr(), 1, k.ci, l, self.last_act.ptr(), a.ptr())?;
        ops.conv_causal(a.fp(), k.ci, l, &k.w, 1, k.k, k.bias(), 1, b.fp())?;
        ops.tanh(b.fp(), l)?;
        ops.gpu.sync()?;
        let mut v = vec![0u8; l * 4];
        b.read(0, &mut v)?;
        Ok(v.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect())
    }

    /// The pre-quantizer latents of a recording at 44.1 kHz (padded to whole frames): [1,024, F] host, F = frames
    pub fn encode_latents(&self, nsd: &Nsd, wav: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (ops, g) = (&self.ops, &self.ops.gpu);
        let frames = wav.len().div_ceil(HOP).max(1);
        let n = frames * HOP;
        let mut x = wav.to_vec();
        x.resize(n, 0.0);
        // the widest signal: the first block's units at the full rate, at half its width
        let widest = (self.enc_first.co * n).max(self.dim * n / self.enc_hop() * 4);
        let (a, b, c) = (DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?);
        a.write(0, &crate::bytes(&x))?;
        let f = &self.enc_first;
        ops.conv_causal(a.fp(), 1, n, &f.w, f.co, f.k, f.bias(), 1, b.fp())?;
        let mut l = n;
        // b: the signal
        for blk in &self.enc_blocks {
            for u in &blk.units {
                u.run(ops, nsd, &b, &a, &c, l)?;
            }
            let down = &blk.down;
            nsd.snake(b.ptr(), 1, down.ci, l, blk.act.ptr(), a.ptr())?;
            l = ops.conv_strided(a.fp(), down.ci, l, &down.w, down.co, down.k, down.bias(), blk.stride, b.fp())?;
            if let Some(tr) = &blk.tr {
                let w = down.co;
                ops.transpose(b.fp(), w, l, a.fp())?;
                tr.run(ops, nsd, &a, l, &self.table(l)?.1)?;
                ops.transpose(a.fp(), l, w, b.fp())?;
            }
        }
        let k = &self.enc_last;
        nsd.snake(b.ptr(), 1, k.ci, l, self.enc_act.ptr(), a.ptr())?;
        ops.conv_causal(a.fp(), k.ci, l, &k.w, k.co, k.k, k.bias(), 1, b.fp())?;
        let mut lo = l;
        for d in &self.downs {
            lo = d.run(ops, nsd, &b, lo, false)?;
        }
        let w = self.dim;
        ops.transpose(b.fp(), w, lo, a.fp())?;
        self.pre.run(ops, nsd, &a, lo, &self.table(lo)?.1)?;
        g.sync()?;
        let mut v = vec![0u8; lo * w * 4];
        a.read(0, &mut v)?;
        let rows: Vec<f32> = v.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
        if lo != frames {
            return Err(Error(format!("the encoder made {lo} frames of {n} samples, not {frames}")));
        }
        Ok((rows, lo))
    }

    /// A recording at 44.1 kHz to its frames' codes [F][10]: the semantic code, then nine residual ones
    pub fn encode(&self, nsd: &Nsd, wav: &[f32]) -> Result<Vec<Vec<i32>>> {
        let (rows, frames) = self.encode_latents(nsd, wav)?;
        let d = self.dim;
        Ok((0..frames).map(|t| {
            let mut r = rows[t * d..(t + 1) * d].to_vec();
            let mut codes = Vec::with_capacity(1 + self.residual.len());
            for b in std::iter::once(&self.semantic).chain(&self.residual) {
                let c = b.nearest(&r);
                r.iter_mut().zip(&b.out[c * d..(c + 1) * d]).for_each(|(x, q)| *x -= q);
                codes.push(c as i32);
            }
            codes
        }).collect())
    }

    /// The codebooks' sizes (the semantic one first)
    pub fn sizes(&self) -> Vec<usize> {
        std::iter::once(self.semantic.size).chain(self.residual.iter().map(|b| b.size)).collect()
    }
}
