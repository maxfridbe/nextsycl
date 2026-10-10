//! The 12 Hz codec's encoder (the speech tokenizer's Mimi encoder, transformers' MimiModel.encode): a recording to
//! its frames' 16 codes - what the in-context clone shows the talker as the voice's example.
//!
//! ```text
//!   24 kHz samples -> SEANet: causal conv k7 to 64, then for strides 4, 5, 6, 8: a residual unit (ELU, k3 to half,
//!                     ELU, k1 back) and ELU + a strided causal conv doubling the channels; ELU, conv k3 to 512 (25 Hz)
//!   -> transformer (8 layers of 512, 8 heads of 64, LayerNorm, RoPE 10k, causal, GELU MLP 2,048, layer scale)
//!   -> downsample (k4, stride 2, replicate padding: 12.5 Hz)
//!   -> split RVQ: a 1x1 projection to 256 and the nearest code of 2,048 - the first codebook alone, the next fifteen
//!      residual after their own projection
//! ```
//!
//! The convolutions and the transformer on the GPU (float32, oneDNN), the downsample and the nearest-code search on
//! the host.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::codec::{conv, lin, stack, Conv, Lin};
use crate::ops::{fp, Ops, Shards};

const HEADS: usize = 8;
const HEAD: usize = 64;
/// the codes a frame the talker takes (of the encoder's 32)
const GROUPS: usize = 16;
/// samples a frame
const HOP: usize = 1920;

struct Res {
    c1: Conv,
    c2: Conv,
}

struct Layer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    qkv: Lin,
    o: Lin,
    fc1: Lin,
    fc2: Lin,
    attn_scale: DevBuf,
    mlp_scale: DevBuf,
}

/// A residual quantizer: its 1x1 input projection [256, 512] and codebooks [2048 x 256] (host)
struct Rvq {
    proj: Vec<f32>,
    books: Vec<Vec<f32>>,
    dim: usize,
}

pub struct Encoder {
    first: Conv,
    /// per stride: the residual unit and the strided conv
    stages: Vec<(Res, Conv, usize)>,
    last: Conv,
    layers: Vec<Layer>,
    /// the downsample [512, 512, 4] (host)
    down: Vec<f32>,
    down_k: usize,
    semantic: Rvq,
    acoustic: Rvq,
}

fn rvq(f: &Shards, p: &str, n: usize) -> Result<Rvq> {
    let s = f.shape(&format!("{p}input_proj.weight"))?;
    let mut books = Vec::new();
    for i in 0..n {
        let b = format!("{p}layers.{i}.codebook.");
        let sum = f.f32(&format!("{b}embed_sum"))?;
        let usage = f.f32(&format!("{b}cluster_usage"))?;
        let d = sum.len() / usage.len();
        books.push(sum.chunks(d).zip(&usage).flat_map(|(r, u)| r.iter().map(move |x| x / u.max(1e-5))).collect());
    }
    Ok(Rvq { proj: f.f32(&format!("{p}input_proj.weight"))?, books, dim: s[0] })
}

impl Rvq {
    /// Frames [T, 512] to each frame's codes, `n` codebooks residually (appended to `out[t]`)
    fn encode(&self, x: &[f32], t: usize, out: &mut [Vec<i32>]) {
        let (d, k) = (self.dim, x.len() / t);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(t.max(1));
        let per = t.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    for (j, codes) in chunk.iter_mut().enumerate() {
                        let fr = &x[(ci * per + j) * k..(ci * per + j + 1) * k];
                        let mut r: Vec<f32> = (0..d).map(|o| self.proj[o * k..(o + 1) * k].iter().zip(fr).map(|(a, b)| a * b).sum()).collect();
                        for book in &self.books {
                            let (mut best, mut bd) = (0, f32::INFINITY);
                            for (c, e) in book.chunks(d).enumerate() {
                                let dist: f32 = e.iter().zip(&r).map(|(a, b)| (a - b) * (a - b)).sum();
                                if dist < bd {
                                    bd = dist;
                                    best = c;
                                }
                            }
                            codes.push(best as i32);
                            r.iter_mut().zip(&book[best * d..(best + 1) * d]).for_each(|(a, b)| *a -= b);
                        }
                    }
                });
            }
        });
    }
}

impl Encoder {
    /// From the speech tokenizer's safetensors (`encoder.`)
    pub fn load(ops: &Ops, f: &Shards) -> Result<Encoder> {
        let p = "encoder.encoder.layers.";
        let first = conv(ops, f, &format!("{p}0.conv"), false, true)?;
        let mut stages = Vec::new();
        let mut i = 1;
        while f.find(&format!("{p}{i}.block.1.conv.weight")).is_ok() {
            let res = Res { c1: conv(ops, f, &format!("{p}{i}.block.1.conv"), false, true)?, c2: conv(ops, f, &format!("{p}{i}.block.3.conv"), false, true)? };
            let down = conv(ops, f, &format!("{p}{}.conv", i + 2), false, true)?;
            let stride = down.k / 2;
            stages.push((res, down, stride));
            i += 3;
        }
        let last = conv(ops, f, &format!("{p}{}.conv", i + 1), false, true)?;
        let t = "encoder.encoder_transformer.layers.";
        let mut layers = Vec::new();
        for l in 0.. {
            let pre = format!("{t}{l}.");
            if f.find(&format!("{pre}input_layernorm.weight")).is_err() {
                break;
            }
            let w = |s: &str| format!("{pre}{s}.weight");
            layers.push(Layer {
                ln1_w: f.dev_f32(ops, &w("input_layernorm"))?,
                ln1_b: f.dev_f32(ops, &format!("{pre}input_layernorm.bias"))?,
                ln2_w: f.dev_f32(ops, &w("post_attention_layernorm"))?,
                ln2_b: f.dev_f32(ops, &format!("{pre}post_attention_layernorm.bias"))?,
                qkv: stack(ops, f, &[w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj")])?,
                o: lin(ops, f, &format!("{pre}self_attn.o_proj"), false)?,
                fc1: lin(ops, f, &format!("{pre}mlp.fc1"), false)?,
                fc2: lin(ops, f, &format!("{pre}mlp.fc2"), false)?,
                attn_scale: f.dev_f32(ops, &format!("{pre}self_attn_layer_scale.scale"))?,
                mlp_scale: f.dev_f32(ops, &format!("{pre}mlp_layer_scale.scale"))?,
            });
        }
        if stages.is_empty() || layers.is_empty() {
            return Err(Error("the speech tokenizer has no encoder".into()));
        }
        let q = "encoder.quantizer.";
        Ok(Encoder {
            first,
            stages,
            last,
            layers,
            down: f.f32("encoder.downsample.conv.weight")?,
            down_k: f.shape("encoder.downsample.conv.weight")?[2],
            semantic: rvq(f, &format!("{q}semantic_residual_vector_quantizer."), 1)?,
            acoustic: rvq(f, &format!("{q}acoustic_residual_vector_quantizer."), GROUPS - 1)?,
        })
    }

    /// 24 kHz mono samples to their frames' codes [T][16] (T = ceil(samples / 1,920))
    pub fn encode(&self, ops: &Ops, nsd: &Nsd, wav: &[f32]) -> Result<Vec<Vec<i32>>> {
        let g = &ops.gpu;
        let n = wav.len();
        if n < HOP {
            return Err(Error("the reference recording is shorter than a frame".into()));
        }
        // the widest signal: the first conv's (64 channels at the input rate), or a residual unit's half of it
        let widest = self.first.co * (n + 8);
        let (a, b, c) = (DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?, DevBuf::f32(g, widest)?);
        a.write(0, &crate::talker::bytes(wav))?;
        let f0 = &self.first;
        ops.conv_causal(a.fp(), 1, 1, n, &f0.w, f0.co, f0.k, f0.bias(), 1, b.fp())?;
        // x in b
        let mut l = n;
        let mut ch = f0.co;
        for (res, down, stride) in &self.stages {
            // x += c2(elu(c1(elu(x))))
            ops.elu(b.fp(), ch * l, a.fp())?;
            ops.conv_causal(a.fp(), 1, ch, l, &res.c1.w, res.c1.co, res.c1.k, res.c1.bias(), 1, c.fp())?;
            ops.elu(c.fp(), res.c1.co * l, c.fp())?;
            ops.conv_causal(c.fp(), 1, res.c1.co, l, &res.c2.w, ch, res.c2.k, res.c2.bias(), 1, a.fp())?;
            ops.add(b.fp(), a.fp(), ch * l)?;
            ops.elu(b.fp(), ch * l, a.fp())?;
            l = ops.conv_strided(a.fp(), 1, ch, l, &down.w, down.co, down.k, down.bias(), *stride, b.fp())?;
            ch = down.co;
        }
        ops.elu(b.fp(), ch * l, a.fp())?;
        let la = &self.last;
        ops.conv_causal(a.fp(), 1, ch, l, &la.w, la.co, la.k, la.bias(), 1, b.fp())?;
        let h = la.co;
        // the transformer over rows [l, h] in a
        ops.transpose(b.fp(), h, l, a.fp())?;
        let x = a.fp();
        let (hb, att, mid) = (b.fp(), fp(&b, l * h), c.fp());
        for y in &self.layers {
            let qw = y.qkv.n;
            nsd.layer_norm(x.cast(), Dt::F32, l, h, y.ln1_w.ptr(), y.ln1_b.ptr(), 1e-5, hb.cast(), Dt::F32)?;
            y.qkv.apply(nsd, hb, l, mid)?;
            ops.rope(mid, qw, l, HEADS, HEAD, 10000.0, 0)?;
            ops.rope(fp(&c, HEADS * HEAD), qw, l, HEADS, HEAD, 10000.0, 0)?;
            ops.window_attn(mid, fp(&c, HEADS * HEAD), fp(&c, 2 * HEADS * HEAD), l, HEADS, HEAD, qw, qw, l, att)?;
            y.o.apply(nsd, att, l, hb)?;
            ops.scale_add(x, hb, &y.attn_scale, l, h)?;
            nsd.layer_norm(x.cast(), Dt::F32, l, h, y.ln2_w.ptr(), y.ln2_b.ptr(), 1e-5, hb.cast(), Dt::F32)?;
            y.fc1.apply(nsd, hb, l, mid)?;
            nsd.gelu(mid.cast(), Dt::F32, l * y.fc1.n, false)?;
            y.fc2.apply(nsd, mid, l, hb)?;
            ops.scale_add(x, hb, &y.mlp_scale, l, h)?;
        }
        g.sync()?;
        let rows = a.to_f32()?[..l * h].to_vec();
        // the downsample: causal, stride 2, the edges replicated
        let (k, s) = (self.down_k, 2);
        let pt = k - s;
        let frames = (l + pt).saturating_sub(k).div_ceil(s) + 1;
        let mut d = vec![0f32; frames * h];
        for t in 0..frames {
            for (o, out) in d[t * h..(t + 1) * h].iter_mut().enumerate() {
                let mut acc = 0f32;
                for j in 0..k {
                    let src = (t * s + j) as isize - pt as isize;
                    let row = &rows[src.clamp(0, l as isize - 1) as usize * h..][..h];
                    let w = &self.down[o * h * k..(o + 1) * h * k];
                    acc += row.iter().enumerate().map(|(i, v)| w[i * k + j] * v).sum::<f32>();
                }
                *out = acc;
            }
        }
        let mut codes = vec![Vec::with_capacity(GROUPS); frames];
        self.semantic.encode(&d, frames, &mut codes);
        self.acoustic.encode(&d, frames, &mut codes);
        codes.truncate(n.div_ceil(HOP));
        Ok(codes)
    }
}
