//! The 12 Hz codec's decoder (qwen-tts' Qwen3TTSTokenizerV2Decoder): a frame's 16 codes to 1,920 samples at 24 kHz.
//!
//! ```text
//!   codes [16, T] -> split RVQ: codebook rows (embedding_sum / usage, 256 wide), the first and the other fifteen
//!                    summed apart, each through its own 1x1 projection to 512, added
//!   -> causal conv k3 to 1,024 -> transformer (8 layers of 512, 16 heads of 64, RoPE 10k, a 72-frame causal window,
//!      layer-scaled residuals, SwiGLU 1,024; projections in and out) -> 2 x (transposed conv x2 + ConvNeXt)
//!   -> causal conv k7 to 1,536 -> 4 blocks (SnakeBeta, transposed conv x8 / x5 / x4 / x3 halving the channels,
//!      residual units at dilations 1, 3, 9) -> SnakeBeta -> causal conv k7 to 1 -> clamp [-1, 1]
//! ```
//!
//! Long clips go in 300-frame chunks, each with up to 25 frames of the one before as context (cut from its output),
//! as the reference's chunked_decode does. Float32 throughout (the checkpoint's own); the convolutions on oneDNN.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Ops, Shards};

pub const RATE: u32 = 24000;
/// samples a frame
pub const HOP: usize = 1920;
const CHUNK: usize = 300;
const CONTEXT: usize = 25;
const WINDOW: usize = 72;
const HEADS: usize = 16;
const HEAD: usize = 64;

pub(crate) struct Conv {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub ci: usize,
    pub co: usize,
    pub k: usize,
}

struct Snake {
    a: DevBuf,
    b: DevBuf,
}

struct Unit {
    act1: Snake,
    conv1: Conv,
    act2: Snake,
    conv2: Conv,
    dil: usize,
}

struct Block {
    act: Snake,
    up: Conv,
    stride: usize,
    units: Vec<Unit>,
}

struct Next {
    up: Conv,
    dw_w: DevBuf,
    dw_b: DevBuf,
    norm_w: DevBuf,
    norm_b: DevBuf,
    pw1: Lin,
    pw2: Lin,
    gamma: DevBuf,
}

/// A float32 linear layer [n, k] (+ bias)
pub(crate) struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Lin,
    o: Lin,
    gu: Lin,
    down: Lin,
    attn_scale: DevBuf,
    mlp_scale: DevBuf,
}

pub struct Codec {
    /// host codebooks [16][2048 x 256] (already divided by their usage)
    books: Vec<Vec<f32>>,
    dim: usize,
    out_first: Conv,
    out_rest: Conv,
    pre: Conv,
    input: Lin,
    layers: Vec<Layer>,
    norm: DevBuf,
    output: Lin,
    ups: Vec<Next>,
    first: Conv,
    blocks: Vec<Block>,
    last_act: Snake,
    last: Conv,
    pub groups: usize,
}

pub(crate) fn lin(ops: &Ops, f: &Shards, name: &str, bias: bool) -> Result<Lin> {
    let s = f.shape(&format!("{name}.weight"))?;
    Ok(Lin {
        w: f.dev_f32(ops, &format!("{name}.weight"))?,
        b: if bias { Some(f.dev_f32(ops, &format!("{name}.bias"))?) } else { None },
        n: s[0],
        k: s[1],
    })
}

/// Linear layers stacked by rows (a fused q/k/v, gate/up)
pub(crate) fn stack(ops: &Ops, f: &Shards, names: &[String]) -> Result<Lin> {
    let mut v = Vec::new();
    let mut k = 0;
    for n in names {
        k = f.shape(n)?[1];
        v.extend(f.f32(n)?);
    }
    Ok(Lin { n: v.len() / k, k, w: DevBuf::from_f32(&ops.gpu, &v)?, b: None })
}

pub(crate) fn conv(ops: &Ops, f: &Shards, name: &str, transposed: bool, bias: bool) -> Result<Conv> {
    let s = f.shape(&format!("{name}.weight"))?;
    let (ci, co) = if transposed { (s[0], s[1]) } else { (s[1], s[0]) };
    Ok(Conv {
        w: f.dev_f32(ops, &format!("{name}.weight"))?,
        b: if bias { Some(f.dev_f32(ops, &format!("{name}.bias"))?) } else { None },
        ci,
        co,
        k: s[2],
    })
}

fn snake(ops: &Ops, f: &Shards, name: &str) -> Result<Snake> {
    Ok(Snake { a: f.dev_f32(ops, &format!("{name}.alpha"))?, b: f.dev_f32(ops, &format!("{name}.beta"))? })
}

impl Lin {
    pub(crate) fn apply(&self, nsd: &Nsd, x: *const f32, m: usize, out: *mut f32) -> Result<()> {
        nsd.linear(x.cast(), Dt::F32, m, self.k, self.w.ptr(), self.n, self.b.as_ref().map_or(none(), |b| b.ptr()), out.cast(), Dt::F32)
    }
}

impl Conv {
    pub(crate) fn bias(&self) -> *const f32 {
        self.b.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }
}

/// Work buffers: two signals of the widest [C, L] in the chain, a third for a residual
struct Work {
    a: DevBuf,
    b: DevBuf,
    c: DevBuf,
}

impl Codec {
    /// From the speech tokenizer's safetensors (its decoder's tensors, `decoder.` prefixed)
    pub fn load(ops: &Ops, f: &Shards) -> Result<Codec> {
        let p = "decoder.";
        let mut books = Vec::new();
        for (part, n) in [("rvq_first", 1), ("rvq_rest", usize::MAX)] {
            for i in 0..n {
                let base = format!("{p}quantizer.{part}.vq.layers.{i}._codebook.");
                if f.find(&format!("{base}embedding_sum")).is_err() {
                    break;
                }
                let sum = f.f32(&format!("{base}embedding_sum"))?;
                let usage = f.f32(&format!("{base}cluster_usage"))?;
                let d = sum.len() / usage.len();
                books.push(sum.chunks(d).zip(&usage).flat_map(|(r, u)| r.iter().map(move |x| x / u.max(1e-5))).collect::<Vec<f32>>());
            }
        }
        let groups = books.len();
        if groups < 2 {
            return Err(Error("the codec has no codebooks".into()));
        }
        let dim = f.shape(&format!("{p}quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"))?[1];
        let tr = format!("{p}pre_transformer.");
        let mut layers = Vec::new();
        for l in 0.. {
            let pre = format!("{tr}layers.{l}.");
            if f.find(&format!("{pre}input_layernorm.weight")).is_err() {
                break;
            }
            let w = |s: &str| format!("{pre}{s}.weight");
            layers.push(Layer {
                in_norm: f.dev_f32(ops, &w("input_layernorm"))?,
                post_norm: f.dev_f32(ops, &w("post_attention_layernorm"))?,
                qkv: stack(ops, f, &[w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj")])?,
                o: lin(ops, f, &format!("{pre}self_attn.o_proj"), false)?,
                gu: stack(ops, f, &[w("mlp.gate_proj"), w("mlp.up_proj")])?,
                down: lin(ops, f, &format!("{pre}mlp.down_proj"), false)?,
                attn_scale: f.dev_f32(ops, &format!("{pre}self_attn_layer_scale.scale"))?,
                mlp_scale: f.dev_f32(ops, &format!("{pre}mlp_layer_scale.scale"))?,
            });
        }
        let mut ups = Vec::new();
        for i in 0.. {
            let u = format!("{p}upsample.{i}.");
            if f.find(&format!("{u}0.conv.weight")).is_err() {
                break;
            }
            ups.push(Next {
                up: conv(ops, f, &format!("{u}0.conv"), true, true)?,
                dw_w: f.dev_f32(ops, &format!("{u}1.dwconv.conv.weight"))?,
                dw_b: f.dev_f32(ops, &format!("{u}1.dwconv.conv.bias"))?,
                norm_w: f.dev_f32(ops, &format!("{u}1.norm.weight"))?,
                norm_b: f.dev_f32(ops, &format!("{u}1.norm.bias"))?,
                pw1: lin(ops, f, &format!("{u}1.pwconv1"), true)?,
                pw2: lin(ops, f, &format!("{u}1.pwconv2"), true)?,
                gamma: f.dev_f32(ops, &format!("{u}1.gamma"))?,
            });
        }
        let mut blocks = Vec::new();
        let mut i = 1;
        while f.find(&format!("{p}decoder.{i}.block.1.conv.weight")).is_ok() {
            let b = format!("{p}decoder.{i}.block.");
            let up = conv(ops, f, &format!("{b}1.conv"), true, true)?;
            let stride = up.k / 2;
            let units = [1usize, 3, 9].iter().enumerate().map(|(j, d)| {
                let u = format!("{b}{}.", j + 2);
                Ok(Unit {
                    act1: snake(ops, f, &format!("{u}act1"))?,
                    conv1: conv(ops, f, &format!("{u}conv1.conv"), false, true)?,
                    act2: snake(ops, f, &format!("{u}act2"))?,
                    conv2: conv(ops, f, &format!("{u}conv2.conv"), false, true)?,
                    dil: *d,
                })
            }).collect::<Result<Vec<_>>>()?;
            blocks.push(Block { act: snake(ops, f, &format!("{b}0"))?, up, stride, units });
            i += 1;
        }
        let c = Codec {
            out_first: conv(ops, f, &format!("{p}quantizer.rvq_first.output_proj"), false, false)?,
            out_rest: conv(ops, f, &format!("{p}quantizer.rvq_rest.output_proj"), false, false)?,
            pre: conv(ops, f, &format!("{p}pre_conv.conv"), false, true)?,
            input: lin(ops, f, &format!("{tr}input_proj"), true)?,
            norm: f.dev_f32(ops, &format!("{tr}norm.weight"))?,
            output: lin(ops, f, &format!("{tr}output_proj"), true)?,
            first: conv(ops, f, &format!("{p}decoder.0.conv"), false, true)?,
            last_act: snake(ops, f, &format!("{p}decoder.{i}"))?,
            last: conv(ops, f, &format!("{p}decoder.{}.conv", i + 1), false, true)?,
            books,
            dim,
            layers,
            ups,
            blocks,
            groups,
        };
        ops.gpu.sync()?;
        Ok(c)
    }

    /// Samples a frame (the transposed convolutions' strides)
    pub fn hop(&self) -> usize {
        self.ups.iter().map(|u| u.up.k).product::<usize>() * self.blocks.iter().map(|b| b.stride).product::<usize>()
    }

    /// Frames [T][16] to samples (24 kHz mono), in chunks as the reference
    pub fn decode(&self, ops: &Ops, nsd: &Nsd, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        let hop = self.hop();
        let mut out = Vec::with_capacity(frames.len() * hop);
        let mut s = 0;
        while s < frames.len() {
            let e = (s + CHUNK).min(frames.len());
            let ctx = if s > CONTEXT { CONTEXT } else { s };
            let wav = self.run(ops, nsd, &frames[s - ctx..e])?;
            out.extend_from_slice(&wav[ctx * hop..]);
            s = e;
        }
        Ok(out)
    }

    /// One chunk through the whole decoder
    fn run(&self, ops: &Ops, nsd: &Nsd, frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        let g = &ops.gpu;
        let t = frames.len();
        let hop = self.hop();
        let up: usize = self.ups.iter().map(|u| u.up.k).product();
        // the widest signal: the first block's input, or a block's output, or the ConvNeXt's 4x MLP
        let mut widest = (self.first.co * t * up).max(4 * self.input.k * t * up);
        let mut l = t * up;
        for b in &self.blocks {
            l *= b.stride;
            widest = widest.max(b.up.co * l);
        }
        let w = Work { a: DevBuf::f32(g, widest)?, b: DevBuf::f32(g, widest)?, c: DevBuf::f32(g, widest)? };
        // the codebooks: the first and the rest summed apart, channel-major [256, T]
        let d = self.dim;
        let (mut e1, mut e2) = (vec![0f32; d * t], vec![0f32; d * t]);
        for (fi, fr) in frames.iter().enumerate() {
            if fr.len() != self.groups {
                return Err(Error(format!("a frame of {} codes; the codec takes {}", fr.len(), self.groups)));
            }
            for (q, &c) in fr.iter().enumerate() {
                let c = c.clamp(0, (self.books[q].len() / d) as i32 - 1) as usize;
                let row = &self.books[q][c * d..(c + 1) * d];
                let e = if q == 0 { &mut e1 } else { &mut e2 };
                for (j, v) in row.iter().enumerate() {
                    e[j * t + fi] += v;
                }
            }
        }
        w.c.write(0, &crate::talker::bytes(&e1))?;
        w.c.write(d * t * 4, &crate::talker::bytes(&e2))?;
        let cd = self.out_first.co;
        ops.conv_causal(w.c.fp(), 1, d, t, &self.out_first.w, cd, 1, std::ptr::null(), 1, w.a.fp())?;
        ops.conv_causal(fp(&w.c, d * t), 1, d, t, &self.out_rest.w, cd, 1, std::ptr::null(), 1, w.b.fp())?;
        ops.add(w.a.fp(), w.b.fp(), cd * t)?;
        let p = &self.pre;
        ops.conv_causal(w.a.fp(), 1, p.ci, t, &p.w, p.co, p.k, p.bias(), 1, w.b.fp())?;
        // the transformer, rows [T, C]
        let lat = p.co;
        ops.transpose(w.b.fp(), lat, t, w.a.fp())?;
        self.transformer(ops, nsd, &w, t)?;
        // w.b [T, latent] -> [latent, T] in a
        ops.transpose(w.b.fp(), t, lat, w.a.fp())?;
        let mut l = t;
        for u in &self.ups {
            l = self.next(ops, nsd, &w, u, l)?;
        }
        // a [latent, l]
        let c0 = &self.first;
        ops.conv_causal(w.a.fp(), 1, c0.ci, l, &c0.w, c0.co, c0.k, c0.bias(), 1, w.b.fp())?;
        let (x, y) = (&w.b, &w.a);
        for b in &self.blocks {
            ops.snake_beta(x.fp(), b.up.ci, l, &b.act.a, &b.act.b, y.fp())?;
            ops.conv_up(y.fp(), 1, b.up.ci, l, &b.up.w, b.up.co, b.up.k, b.up.bias(), b.stride, x.fp())?;
            l *= b.stride;
            let c = b.up.co;
            for u in &b.units {
                // x += conv2(snake(conv1(snake(x))))
                ops.snake_beta(x.fp(), c, l, &u.act1.a, &u.act1.b, y.fp())?;
                ops.conv_causal(y.fp(), 1, c, l, &u.conv1.w, c, u.conv1.k, u.conv1.bias(), u.dil, w.c.fp())?;
                ops.snake_beta(w.c.fp(), c, l, &u.act2.a, &u.act2.b, y.fp())?;
                ops.conv_causal(y.fp(), 1, c, l, &u.conv2.w, c, u.conv2.k, u.conv2.bias(), 1, w.c.fp())?;
                ops.add(x.fp(), w.c.fp(), c * l)?;
            }
        }
        let c = self.last.ci;
        ops.snake_beta(x.fp(), c, l, &self.last_act.a, &self.last_act.b, y.fp())?;
        ops.conv_causal(y.fp(), 1, c, l, &self.last.w, 1, self.last.k, self.last.bias(), 1, x.fp())?;
        ops.clamp(x.fp(), l, -1.0, 1.0)?;
        g.sync()?;
        if l != t * hop {
            return Err(Error(format!("the decoder made {l} samples of {t} frames, not {}", t * hop)));
        }
        Ok(x.to_f32()?[..l].to_vec())
    }

    /// The transformer over rows [T, latent] in w.a; its output [T, latent] in w.b
    fn transformer(&self, ops: &Ops, nsd: &Nsd, w: &Work, t: usize) -> Result<()> {
        let h = self.input.n;
        // x [T, h] in b; the layer's work after it in b and in c
        let x = w.b.fp();
        self.input.apply(nsd, w.a.fp(), t, x)?;
        let hb = fp(&w.b, t * h);
        let att = fp(&w.b, 2 * t * h);
        let (qkv, mid) = (w.c.fp(), fp(&w.c, 0));
        for l in &self.layers {
            let qw = l.qkv.n;
            let q = HEADS * HEAD;
            nsd.rms_norm_mod(x.cast(), Dt::F32, t, h, l.in_norm.ptr(), 1e-5, none(), none(), none(), hb.cast(), Dt::F32)?;
            l.qkv.apply(nsd, hb, t, qkv)?;
            ops.rope(qkv, qw, t, HEADS, HEAD, 10000.0, 0)?;
            ops.rope(fp(&w.c, q), qw, t, (qw - q) / 2 / HEAD, HEAD, 10000.0, 0)?;
            let kvh = (qw - q) / 2;
            ops.window_attn(qkv, fp(&w.c, q), fp(&w.c, q + kvh), t, HEADS, HEAD, qw, qw, WINDOW, att)?;
            l.o.apply(nsd, att, t, hb)?;
            ops.scale_add(x, hb, &l.attn_scale, t, h)?;
            nsd.rms_norm_mod(x.cast(), Dt::F32, t, h, l.post_norm.ptr(), 1e-5, none(), none(), none(), hb.cast(), Dt::F32)?;
            l.gu.apply(nsd, hb, t, mid)?;
            let ffn = l.gu.n / 2;
            nsd.swiglu(mid.cast(), Dt::F32, t, ffn, att.cast(), Dt::F32)?;
            l.down.apply(nsd, att, t, hb)?;
            ops.scale_add(x, hb, &l.mlp_scale, t, h)?;
        }
        nsd.rms_norm_mod(x.cast(), Dt::F32, t, h, self.norm.ptr(), 1e-5, none(), none(), none(), hb.cast(), Dt::F32)?;
        // into a, then back to b for the caller
        self.output.apply(nsd, hb, t, w.a.fp())?;
        w.b.copy_within(0, &w.a, 0, t * self.output.n * 4)
    }

    /// One upsampling stage on a [C, l]: transposed conv, then a ConvNeXt block; the result [C, l * k] in a
    fn next(&self, ops: &Ops, nsd: &Nsd, w: &Work, u: &Next, l: usize) -> Result<usize> {
        let c = u.up.co;
        ops.conv_up(w.a.fp(), 1, u.up.ci, l, &u.up.w, c, u.up.k, u.up.bias(), u.up.k, w.b.fp())?;
        let l = l * u.up.k;
        // b: the input x [C, l]; its depthwise conv into a, then rows [l, C] in c
        ops.dwconv(w.b.fp(), c, l, &u.dw_w, &u.dw_b, 7, w.a.fp())?;
        ops.transpose(w.a.fp(), c, l, w.c.fp())?;
        nsd.layer_norm(w.c.ptr(), Dt::F32, l, c, u.norm_w.ptr(), u.norm_b.ptr(), 1e-6, w.a.ptr(), Dt::F32)?;
        u.pw1.apply(nsd, w.a.fp(), l, w.c.fp())?;
        nsd.gelu(w.c.ptr(), Dt::F32, l * u.pw1.n, false)?;
        u.pw2.apply(nsd, w.c.fp(), l, w.a.fp())?;
        // x as rows, + gamma * the MLP's, back to [C, l]
        ops.transpose(w.b.fp(), c, l, w.c.fp())?;
        ops.scale_add(w.c.fp(), w.a.fp(), &u.gamma, l, c)?;
        ops.transpose(w.c.fp(), l, c, w.a.fp())?;
        Ok(l)
    }
}
