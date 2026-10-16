//! Hunyuan3D 2.1's denoiser: `HunYuanDiTPlain` - 21 blocks of 2,048 features (16 heads x 128) over the timestep
//! token and 4,096 latent tokens, each block self-attention, cross-attention to the DINOv2 tokens and an MLP; U-ViT
//! skips (blocks 11-20 take the outputs of blocks 9 down to 0), and in blocks 15-20 a mixture of 8 experts (the 2
//! most likely, their softmax probabilities as weights, not renormalized) beside a shared expert.
//!
//! ```text
//!   x = [t_embedder(t) | x_embedder(latents)]           t: sin | cos of t e^(-ln(1e4) i / 1024), an MLP (gelu)
//!   block:  x = skip_norm(skip_linear([skip | x]))       (blocks 11..20)
//!           x += out(attention(rms q, rms k, v of ln1(x)))
//!           x += out2(attention(rms q of ln2(x), rms k / v of the context))
//!           x += mlp(ln3(x))  |  moe(ln3(x)) + shared(ln3(x))
//!   v = linear(ln(x))[1:]
//! ```
//!
//! Tencent splits q, k, v per head out of the concatenation [q | k | v] (and the cross-attention's k, v out of
//! [k | v]) - head h's q is features [384 h, 384 h + 128) of it, its k the next 128, its v the next - so the stacked
//! matrices' rows are reordered at load into whole q, k, v. CFG runs the conditional and the unconditional (all-zero
//! context) passes one after the other; the context's keys and values are computed once a generation.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{fp, hp, read_f32, HostMat, Lin, Ops, Weights};

const EPS: f32 = 1e-6;
const HEAD: usize = 128;

/// Rows of a per-head split of a `parts`-way concatenation regrouped by part: new row (p, h, j) is old row
/// h * parts * HEAD + p * HEAD + j
fn by_part(parts: usize, heads: usize) -> Vec<usize> {
    (0..parts).flat_map(|p| (0..heads).flat_map(move |h| (0..HEAD).map(move |j| h * parts * HEAD + p * HEAD + j))).collect()
}

struct Ffn {
    up: Lin,
    down: Lin,
}

enum Mlp {
    Dense(Ffn),
    Moe { gate: Lin, experts: Vec<Ffn>, shared: Ffn, top: usize },
}

struct Block {
    skip: Option<(Lin, Lin, DevBuf, DevBuf)>,
    n1: (DevBuf, DevBuf),
    qkv: Lin,
    qn1: DevBuf,
    kn1: DevBuf,
    out1: Lin,
    n2: (DevBuf, DevBuf),
    q2: Lin,
    kv2: Lin,
    qn2: DevBuf,
    kn2: DevBuf,
    out2: Lin,
    n3: (DevBuf, DevBuf),
    mlp: Mlp,
}

pub struct Plain {
    pub width: usize,
    heads: usize,
    pub channels: usize,
    pub context: usize,
    x_in: Lin,
    t1: Lin,
    t2: Lin,
    blocks: Vec<Block>,
    nf: (DevBuf, DevBuf),
    out: Lin,
    pub bytes: usize,
}

/// The timestep's features: sin | cos of t * exp(-ln(10000) i / (dim / 2))
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let a = t * (-(10000f32.ln()) * i as f32 / half as f32).exp();
        out[i] = a.sin();
        out[half + i] = a.cos();
    }
    out
}

/// One context's keys and values in every block: [rows, 2 w] half (k | v), k normalized
struct Ctx {
    rows: usize,
    kv: Vec<DevBuf>,
}

pub struct Scratch {
    tokens: usize,
    ctx: [Ctx; 2],
    x: DevBuf,
    xh: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    tmpf: DevBuf,
    mid: DevBuf,
    skips: Vec<DevBuf>,
    logits: DevBuf,
    ridx: DevBuf,
    rw: DevBuf,
    lidx: DevBuf,
    lw: DevBuf,
    rows_h: DevBuf,
    th: DevBuf,
    xin: DevBuf,
    v: [DevBuf; 2],
}

fn norm(w: &Weights, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((w.vec(&format!("{p}.weight"))?, w.vec(&format!("{p}.bias"))?))
}

impl Plain {
    pub fn load(w: &Weights) -> Result<Plain> {
        let width = w.shape("x_embedder.weight")?[0];
        let channels = w.shape("x_embedder.weight")?[1];
        let heads = width / HEAD;
        let ffn = |p: &str| -> Result<Ffn> { Ok(Ffn { up: w.lin(&format!("{p}.0.proj"))?, down: w.lin(&format!("{p}.2"))? }) };
        let mut blocks = Vec::new();
        let mut context = 0;
        while w.has(&format!("blocks.{}.norm1.weight", blocks.len())) {
            let p = format!("blocks.{}.", blocks.len());
            let a1 = format!("{p}attn1.");
            let qkv = w.host_mat(&format!("{a1}to_q.weight"))?.stack(&w.host_mat(&format!("{a1}to_k.weight"))?).stack(&w.host_mat(&format!("{a1}to_v.weight"))?);
            let a2 = format!("{p}attn2.");
            let kv: HostMat = w.host_mat(&format!("{a2}to_k.weight"))?.stack(&w.host_mat(&format!("{a2}to_v.weight"))?);
            context = kv.k;
            let skip = if w.has(&format!("{p}skip_linear.weight")) {
                let s = w.host_mat(&format!("{p}skip_linear.weight"))?;
                let b = w.host_f32(&format!("{p}skip_linear.bias"))?;
                let (sn_w, sn_b) = norm(w, &format!("{p}skip_norm"))?;
                Some((w.upload(&s.cols(0, width), Some(b))?, w.upload(&s.cols(width, 2 * width), None)?, sn_w, sn_b))
            } else {
                None
            };
            let mlp = if w.has(&format!("{p}moe.gate.weight")) {
                let mut experts = Vec::new();
                while w.has(&format!("{p}moe.experts.{}.net.0.proj.weight", experts.len())) {
                    experts.push(ffn(&format!("{p}moe.experts.{}.net", experts.len()))?);
                }
                Mlp::Moe { gate: w.lin(&format!("{p}moe.gate"))?, experts, shared: ffn(&format!("{p}moe.shared_experts.net"))?, top: 2 }
            } else {
                Mlp::Dense(Ffn { up: w.lin(&format!("{p}mlp.fc1"))?, down: w.lin(&format!("{p}mlp.fc2"))? })
            };
            blocks.push(Block {
                skip,
                n1: norm(w, &format!("{p}norm1"))?,
                qkv: w.upload(&qkv.rows(by_part(3, heads)), None)?,
                qn1: w.vec(&format!("{a1}q_norm.weight"))?,
                kn1: w.vec(&format!("{a1}k_norm.weight"))?,
                out1: w.lin(&format!("{a1}out_proj"))?,
                n2: norm(w, &format!("{p}norm2"))?,
                q2: w.lin(&format!("{a2}to_q"))?,
                kv2: w.upload(&kv.rows(by_part(2, heads)), None)?,
                qn2: w.vec(&format!("{a2}q_norm.weight"))?,
                kn2: w.vec(&format!("{a2}k_norm.weight"))?,
                out2: w.lin(&format!("{a2}out_proj"))?,
                n3: norm(w, &format!("{p}norm3"))?,
                mlp,
            });
        }
        if blocks.is_empty() {
            return Err(Error(format!("{}: no HunYuanDiTPlain blocks under {}", w.st.path.display(), w.prefix)));
        }
        Ok(Plain {
            width,
            heads,
            channels,
            context,
            x_in: w.lin("x_embedder")?,
            t1: w.lin("t_embedder.mlp.0")?,
            t2: w.lin("t_embedder.mlp.2")?,
            blocks,
            nf: norm(w, "final_layer.norm_final")?,
            out: w.lin("final_layer.linear")?,
            bytes: w.bytes.get(),
        })
    }

    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// The keys and values of a context (float32 [rows, context features]; None: the unconditional zeros)
    fn context_kv(&self, ops: &Ops, cond: Option<&DevBuf>, rows: usize) -> Result<Ctx> {
        let gpu = &ops.gpu;
        let wd = self.width;
        let ch = DevBuf::new(gpu, rows * self.context * 2)?;
        match cond {
            Some(c) => ops.to_half(c.ptr(), ch.ptr(), rows * self.context)?,
            None => ch.fill(0)?,
        }
        let mut kv = Vec::new();
        for b in &self.blocks {
            let o = DevBuf::new(gpu, rows * 2 * wd * 2)?;
            ops.lin(&b.kv2, ch.ptr(), rows, o.ptr(), Dt::F16)?;
            ops.head_norm(o.ptr(), rows, self.heads, HEAD, 2 * wd, b.kn2.ptr(), none(), EPS, true)?;
            kv.push(o);
        }
        Ok(Ctx { rows, kv })
    }

    /// The working set of a generation: `tokens` latent rows, the context `cond` (float32 [rows, features])
    pub fn scratch(&self, ops: &Ops, cond: &DevBuf, rows: usize, tokens: usize) -> Result<Scratch> {
        let gpu = &ops.gpu;
        let (wd, s) = (self.width, tokens + 1);
        let fmax = self.blocks.iter().map(|b| match &b.mlp {
            Mlp::Dense(f) => f.up.n,
            Mlp::Moe { shared, .. } => shared.up.n,
        }).max().unwrap_or(4 * wd);
        let nskip = self.blocks.len() / 2;
        Ok(Scratch {
            tokens,
            ctx: [self.context_kv(ops, Some(cond), rows)?, self.context_kv(ops, None, rows)?],
            x: DevBuf::new(gpu, s * wd * 4)?,
            xh: DevBuf::new(gpu, s * wd * 2)?,
            qkv: DevBuf::new(gpu, s * 3 * wd * 2)?,
            att: DevBuf::new(gpu, s * wd * 2)?,
            tmp: DevBuf::new(gpu, s * wd * 2)?,
            tmpf: DevBuf::new(gpu, s * wd * 4)?,
            mid: DevBuf::new(gpu, s * fmax * 2)?,
            skips: (0..nskip).map(|_| DevBuf::new(gpu, s * wd * 2)).collect::<Result<_>>()?,
            logits: DevBuf::new(gpu, s * 64 * 4)?,
            ridx: DevBuf::new(gpu, s * 8 * 4)?,
            rw: DevBuf::new(gpu, s * 8 * 4)?,
            lidx: DevBuf::new(gpu, s * 8 * 4)?,
            lw: DevBuf::new(gpu, s * 8 * 4)?,
            rows_h: DevBuf::new(gpu, s * wd * 2)?,
            th: DevBuf::new(gpu, 4 * wd * 2)?,
            xin: DevBuf::new(gpu, tokens * self.channels * 2)?,
            v: [DevBuf::new(gpu, tokens * self.channels * 4)?, DevBuf::new(gpu, tokens * self.channels * 4)?],
        })
    }

    fn ffn(&self, ops: &Ops, sc: &Scratch, f: &Ffn, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void) -> Result<()> {
        ops.lin(&f.up, x, m, sc.mid.ptr(), Dt::F16)?;
        ops.gelu(sc.mid.ptr(), m * f.up.n, false)?;
        ops.lin(&f.down, sc.mid.ptr(), m, out, Dt::F16)
    }

    /// The velocity of latents `x` (float32 [tokens, channels]) at `t` with context `which` (0 the picture's, 1 the
    /// unconditional), into `out`; `stop_after`: stop after that block (a check: the stream stays in the scratch)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, ops: &Ops, sc: &Scratch, x: &DevBuf, t: f32, which: usize, out: &DevBuf, stop_after: Option<usize>) -> Result<()> {
        let (wd, h) = (self.width, self.heads);
        let n = sc.tokens;
        let s = n + 1;
        let ctx = &sc.ctx[which];
        // the time token, then the latents
        let tf: Vec<u16> = timestep_features(t, self.t1.k).iter().map(|&v| crate::ops::f32_to_f16(v)).collect();
        // SAFETY: u16s viewed as bytes.
        sc.th.write(0, unsafe { std::slice::from_raw_parts(tf.as_ptr().cast::<u8>(), tf.len() * 2) })?;
        ops.lin(&self.t1, sc.th.ptr(), 1, sc.mid.ptr(), Dt::F16)?;
        ops.gelu(sc.mid.ptr(), self.t1.n, false)?;
        ops.lin(&self.t2, sc.mid.ptr(), 1, sc.x.ptr(), Dt::F32)?;
        ops.to_half(x.ptr(), sc.xin.ptr(), n * self.channels)?;
        ops.lin(&self.x_in, sc.xin.ptr(), n, fp(&sc.x, wd).cast(), Dt::F32)?;
        let nskip = sc.skips.len();
        for (i, b) in self.blocks.iter().enumerate() {
            if let Some((wa, wb, nw, nb)) = &b.skip {
                let skip = &sc.skips[nskip - 1 - (i - nskip - 1)];
                ops.lin(wa, skip.ptr(), s, sc.tmpf.ptr(), Dt::F32)?;
                ops.to_half(sc.x.ptr(), sc.xh.ptr(), s * wd)?;
                ops.lin(wb, sc.xh.ptr(), s, sc.tmp.ptr(), Dt::F16)?;
                ops.add(sc.tmpf.ptr(), sc.tmp.ptr(), true, s, wd, none())?;
                ops.ln(sc.tmpf.ptr(), false, s, wd, EPS, nw.ptr(), nb.ptr(), none(), none(), sc.x.ptr(), false)?;
            }
            // self-attention
            ops.ln(sc.x.ptr(), false, s, wd, EPS, b.n1.0.ptr(), b.n1.1.ptr(), none(), none(), sc.xh.ptr(), true)?;
            ops.lin(&b.qkv, sc.xh.ptr(), s, sc.qkv.ptr(), Dt::F16)?;
            ops.head_norm(sc.qkv.ptr(), s, h, HEAD, 3 * wd, b.qn1.ptr(), none(), EPS, true)?;
            ops.head_norm(hp(&sc.qkv, wd), s, h, HEAD, 3 * wd, b.kn1.ptr(), none(), EPS, true)?;
            ops.attention(sc.qkv.ptr(), hp(&sc.qkv, wd), hp(&sc.qkv, 2 * wd), s, h, HEAD, 3 * wd, sc.att.ptr())?;
            ops.lin(&b.out1, sc.att.ptr(), s, sc.tmp.ptr(), Dt::F16)?;
            ops.add(sc.x.ptr(), sc.tmp.ptr(), true, s, wd, none())?;
            // cross-attention to the context
            ops.ln(sc.x.ptr(), false, s, wd, EPS, b.n2.0.ptr(), b.n2.1.ptr(), none(), none(), sc.xh.ptr(), true)?;
            ops.lin(&b.q2, sc.xh.ptr(), s, sc.qkv.ptr(), Dt::F16)?;
            ops.head_norm(sc.qkv.ptr(), s, h, HEAD, wd, b.qn2.ptr(), none(), EPS, true)?;
            let kv = &ctx.kv[i];
            ops.cross_attention(sc.qkv.ptr(), s, wd, kv.ptr(), hp(kv, wd), ctx.rows, 2 * wd, h, HEAD, sc.att.ptr())?;
            ops.lin(&b.out2, sc.att.ptr(), s, sc.tmp.ptr(), Dt::F16)?;
            ops.add(sc.x.ptr(), sc.tmp.ptr(), true, s, wd, none())?;
            // the MLP, or the experts
            ops.ln(sc.x.ptr(), false, s, wd, EPS, b.n3.0.ptr(), b.n3.1.ptr(), none(), none(), sc.xh.ptr(), true)?;
            match &b.mlp {
                Mlp::Dense(f) => {
                    self.ffn(ops, sc, f, sc.xh.ptr(), s, sc.tmp.ptr())?;
                    ops.add(sc.x.ptr(), sc.tmp.ptr(), true, s, wd, none())?;
                }
                Mlp::Moe { gate, experts, shared, top } => {
                    self.moe(ops, sc, gate, experts, *top, s)?;
                    self.ffn(ops, sc, shared, sc.xh.ptr(), s, sc.tmp.ptr())?;
                    ops.add(sc.x.ptr(), sc.tmp.ptr(), true, s, wd, none())?;
                }
            }
            if i < nskip {
                ops.to_half(sc.x.ptr(), sc.skips[i].ptr(), s * wd)?;
            }
            if stop_after == Some(i) {
                return Ok(());
            }
        }
        ops.ln(fp(&sc.x, wd).cast(), false, n, wd, EPS, self.nf.0.ptr(), self.nf.1.ptr(), none(), none(), sc.xh.ptr(), true)?;
        ops.lin(&self.out, sc.xh.ptr(), n, out.ptr(), Dt::F32)
    }

    /// The experts' share, added into the stream: each row's `top` experts (their probabilities as weights) run on
    /// the rows routed to them, gathered and scattered back
    fn moe(&self, ops: &Ops, sc: &Scratch, gate: &Lin, experts: &[Ffn], top: usize, s: usize) -> Result<()> {
        let wd = self.width;
        let e = experts.len();
        ops.lin(gate, sc.xh.ptr(), s, sc.logits.ptr(), Dt::F32)?;
        ops.route(sc.logits.ptr(), s, e, top, sc.ridx.ptr(), sc.rw.ptr())?;
        let mut ib = vec![0u8; s * top * 4];
        sc.ridx.read(0, &mut ib)?;
        let w = read_f32(&sc.rw, s * top)?;
        let idx: Vec<i32> = ib.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // rows by expert, and their weights
        let mut lists: Vec<(Vec<i32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); e];
        for r in 0..s {
            for j in 0..top {
                let ex = idx[r * top + j] as usize;
                if ex >= e {
                    return Err(Error(format!("the router chose expert {ex} of {e}")));
                }
                lists[ex].0.push(r as i32);
                lists[ex].1.push(w[r * top + j]);
            }
        }
        let (mut allr, mut allw) = (Vec::with_capacity(s * top), Vec::with_capacity(s * top));
        for (r, ww) in &lists {
            allr.extend_from_slice(r);
            allw.extend_from_slice(ww);
        }
        // SAFETY: i32s viewed as bytes.
        sc.lidx.write(0, unsafe { std::slice::from_raw_parts(allr.as_ptr().cast::<u8>(), allr.len() * 4) })?;
        crate::ops::write_f32(&sc.lw, 0, &allw)?;
        let mut at = 0usize;
        for (k, (r, _)) in lists.iter().enumerate() {
            let c = r.len();
            if c == 0 {
                continue;
            }
            let ip = sc.lidx.ptr().cast::<i32>().wrapping_add(at).cast();
            let wp = fp(&sc.lw, at).cast();
            ops.gather(sc.xh.ptr(), ip, c, wd, sc.rows_h.ptr())?;
            self.ffn(ops, sc, &experts[k], sc.rows_h.ptr(), c, sc.tmp.ptr())?;
            ops.scatter_add(sc.tmp.ptr(), ip, wp, c, wd, sc.x.ptr())?;
            at += c;
        }
        Ok(())
    }

    /// The guided velocity: the conditional and the unconditional passes, u + cfg (c - u), into `out`
    pub fn forward_cfg(&self, ops: &Ops, sc: &mut Scratch, x: &DevBuf, t: f32, cfg: f32, out: &DevBuf) -> Result<()> {
        let n = sc.tokens * self.channels;
        self.forward(ops, sc, x, t, 0, &sc.v[0], None)?;
        self.forward(ops, sc, x, t, 1, &sc.v[1], None)?;
        ops.cfg(out.ptr(), sc.v[0].ptr(), sc.v[1].ptr(), n, cfg)
    }

    /// The stream (float32 [tokens + 1, width]) after a forward that stopped early
    pub fn stream(&self, sc: &Scratch) -> Result<Vec<f32>> {
        read_f32(&sc.x, (sc.tokens + 1) * self.width)
    }
}
