//! Hunyuan3D-2mini's denoiser: `Hunyuan3DDiT`, a Flux-style MMDiT over the shape's latent tokens and the picture's
//! DINOv2 tokens - 1,024 features (16 heads x 64), 8 double-stream blocks (each stream its own weights, one joint
//! attention), 16 single-stream blocks (attention and MLP in parallel), adaLN modulation from the timestep and the
//! distilled guidance, no positional embedding.
//!
//! ```text
//!   vec = time_in(emb(t)) + guidance_in(emb(g))         emb: 256 features, t * 1000 at max period 1000 (sic)
//!   double: per stream  h = ln(x) (1 + scale1) + shift1;  q k v = h Wqkv; q, k: rms per head
//!           joint attention over [context | latents];    x += gate1 * proj(att)
//!           x += gate2 * fc2(gelu_tanh(fc1(ln(x) (1 + scale2) + shift2)))
//!   single: h = ln(x) (1 + scale) + shift;  [q k v | mlp] = h W1;  x += gate * W2 [att | gelu_tanh(mlp)]
//!   out = linear(ln(latents) (1 + scale) + shift)
//! ```
//!
//! The context and latent rows live in one float32 buffer [context | latents], so the single blocks run over it as
//! the reference's concatenation. The single blocks' fused W1 / W2 are split at load (qkv | mlp rows, att | mlp
//! columns): each product writes a buffer of its own, and the two halves of W2 are added into the stream separately.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{fp, hp, Lin, Ops, Weights, P};

const EPS: f32 = 1e-6;
const HEAD: usize = 64;

struct Stream {
    m: Lin,
    qkv: Lin,
    qn: DevBuf,
    kn: DevBuf,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

struct Double {
    img: Stream,
    txt: Stream,
}

struct Single {
    m: Lin,
    qkv: Lin,
    mlp: Lin,
    att_out: Lin,
    mlp_out: Lin,
    qn: DevBuf,
    kn: DevBuf,
}

pub struct Flux {
    pub width: usize,
    heads: usize,
    ffn: usize,
    pub channels: usize,
    pub context: usize,
    latent_in: Lin,
    time_in: (Lin, Lin),
    guidance_in: Option<(Lin, Lin)>,
    cond_in: Lin,
    doubles: Vec<Double>,
    singles: Vec<Single>,
    final_mod: Lin,
    final_lin: Lin,
    pub bytes: usize,
}

/// Flux's sinusoidal embedding of `t` (`dim` features: cos | sin), `t * time_factor` at `max_period`
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32, time_factor: f32) -> Vec<f32> {
    let half = dim / 2;
    let t = time_factor * t;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let f = (-(max_period.ln()) * i as f32 / half as f32).exp();
        let a = t * f;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

/// Per-generation buffers: the context's projection and the working set of one forward
pub struct Scratch {
    rows: usize,
    tokens: usize,
    /// cond_in(context), float32 [context rows, width]: the stream's start every step
    txt0: DevBuf,
    x: DevBuf,
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    mlp: DevBuf,
    tmp: DevBuf,
    tmp2: DevBuf,
    vec: DevBuf,
    vech: DevBuf,
    small: DevBuf,
    smallh: DevBuf,
    mods: DevBuf,
    xin: DevBuf,
}

impl Flux {
    pub fn load(w: &Weights) -> Result<Flux> {
        let width = w.shape("latent_in.weight")?[0];
        let channels = w.shape("latent_in.weight")?[1];
        let context = w.shape("cond_in.weight")?[1];
        let heads = width / HEAD;
        let pair = |p: &str| -> Result<(Lin, Lin)> { Ok((w.lin(&format!("{p}.in_layer"))?, w.lin(&format!("{p}.out_layer"))?)) };
        let stream = |p: &str| -> Result<Stream> {
            Ok(Stream {
                m: w.lin(&format!("{p}_mod.lin"))?,
                qkv: w.lin(&format!("{p}_attn.qkv"))?,
                qn: w.vec(&format!("{p}_attn.norm.query_norm.scale"))?,
                kn: w.vec(&format!("{p}_attn.norm.key_norm.scale"))?,
                proj: w.lin(&format!("{p}_attn.proj"))?,
                fc1: w.lin(&format!("{p}_mlp.0"))?,
                fc2: w.lin(&format!("{p}_mlp.2"))?,
            })
        };
        let mut doubles = Vec::new();
        while w.has(&format!("double_blocks.{}.img_mod.lin.weight", doubles.len())) {
            let p = format!("double_blocks.{}.", doubles.len());
            doubles.push(Double { img: stream(&format!("{p}img"))?, txt: stream(&format!("{p}txt"))? });
        }
        let mut singles = Vec::new();
        while w.has(&format!("single_blocks.{}.linear1.weight", singles.len())) {
            let p = format!("single_blocks.{}.", singles.len());
            let l1 = w.host_mat(&format!("{p}linear1.weight"))?;
            let b1 = w.host_f32(&format!("{p}linear1.bias"))?;
            let l2 = w.host_mat(&format!("{p}linear2.weight"))?;
            let b2 = w.host_f32(&format!("{p}linear2.bias"))?;
            if l1.n < 3 * width || l2.k != l1.n - 2 * width {
                return Err(Error(format!("{p}linear1 / linear2: {}x{} and {}x{}", l1.n, l1.k, l2.n, l2.k)));
            }
            singles.push(Single {
                m: w.lin(&format!("{p}modulation.lin"))?,
                qkv: w.upload(&l1.rows(0..3 * width), Some(b1[..3 * width].to_vec()))?,
                mlp: w.upload(&l1.rows(3 * width..l1.n), Some(b1[3 * width..].to_vec()))?,
                att_out: w.upload(&l2.cols(0, width), Some(b2))?,
                mlp_out: w.upload(&l2.cols(width, l2.k), None)?,
                qn: w.vec(&format!("{p}norm.query_norm.scale"))?,
                kn: w.vec(&format!("{p}norm.key_norm.scale"))?,
            });
        }
        if doubles.is_empty() || singles.is_empty() {
            return Err(Error(format!("{}: no Hunyuan3DDiT blocks under {}", w.st.path.display(), w.prefix)));
        }
        let ffn = doubles[0].img.fc1.n;
        let f = Flux {
            width,
            heads,
            ffn,
            channels,
            context,
            latent_in: w.lin("latent_in")?,
            time_in: pair("time_in")?,
            guidance_in: if w.has("guidance_in.in_layer.weight") { Some(pair("guidance_in")?) } else { None },
            cond_in: w.lin("cond_in")?,
            doubles,
            singles,
            final_mod: w.lin("final_layer.adaLN_modulation.1")?,
            final_lin: w.lin("final_layer.linear")?,
            bytes: w.bytes.get(),
        };
        Ok(f)
    }

    pub fn blocks(&self) -> (usize, usize) {
        (self.doubles.len(), self.singles.len())
    }

    /// The working set for `tokens` latent rows beside the context `cond` (float32 [rows, context features])
    pub fn scratch(&self, ops: &Ops, cond: &DevBuf, rows: usize, tokens: usize) -> Result<Scratch> {
        let gpu = &ops.gpu;
        let (wd, s) = (self.width, rows + tokens);
        let condh = DevBuf::new(gpu, rows * self.context * 2)?;
        ops.to_half(cond.ptr(), condh.ptr(), rows * self.context)?;
        let txt0 = DevBuf::new(gpu, rows * wd * 4)?;
        ops.lin(&self.cond_in, condh.ptr(), rows, txt0.ptr(), Dt::F32)?;
        let nmods = self.doubles.len() * 12 + self.singles.len() * 3 + 2;
        Ok(Scratch {
            rows,
            tokens,
            txt0,
            x: DevBuf::new(gpu, s * wd * 4)?,
            h: DevBuf::new(gpu, s * wd * 2)?,
            qkv: DevBuf::new(gpu, s * 3 * wd * 2)?,
            att: DevBuf::new(gpu, s * wd * 2)?,
            mlp: DevBuf::new(gpu, s * self.ffn * 2)?,
            tmp: DevBuf::new(gpu, s * wd * 2)?,
            tmp2: DevBuf::new(gpu, s * wd * 2)?,
            vec: DevBuf::new(gpu, wd * 4)?,
            vech: DevBuf::new(gpu, wd.max(256) * 2)?,
            small: DevBuf::new(gpu, wd * 4)?,
            smallh: DevBuf::new(gpu, wd.max(256) * 2)?,
            mods: DevBuf::new(gpu, nmods * wd * 4)?,
            xin: DevBuf::new(gpu, tokens * self.channels * 2)?,
        })
    }

    /// in -> silu -> out (MLPEmbedder) of a 256-feature embedding, added to `vec` (float32 [width]) or written
    fn embed(&self, ops: &Ops, sc: &Scratch, pair: &(Lin, Lin), e: &[f32], add: bool) -> Result<()> {
        let eh: Vec<u16> = e.iter().map(|&v| crate::ops::f32_to_f16(v)).collect();
        // SAFETY: u16s viewed as bytes.
        sc.smallh.write(0, unsafe { std::slice::from_raw_parts(eh.as_ptr().cast::<u8>(), eh.len() * 2) })?;
        ops.lin(&pair.0, sc.smallh.ptr(), 1, sc.small.ptr(), Dt::F32)?;
        ops.silu(sc.small.ptr(), sc.vech.ptr(), self.width)?;
        if add {
            ops.lin(&pair.1, sc.vech.ptr(), 1, sc.small.ptr(), Dt::F32)?;
            ops.add(sc.vec.ptr(), sc.small.ptr(), false, 1, self.width, none())
        } else {
            ops.lin(&pair.1, sc.vech.ptr(), 1, sc.vec.ptr(), Dt::F32)
        }
    }

    /// The velocity for latents `x` (float32 [tokens, channels]) at time `t` (reversed: 0 = noise) with guidance
    /// `g`, into `out` (float32 [tokens, channels]). `stop_after`: Some(0) stops after double block 0 and leaves the
    /// stream in the scratch (a check).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, ops: &Ops, sc: &Scratch, x: &DevBuf, t: f32, g: f32, out: &DevBuf, stop_after: Option<usize>) -> Result<()> {
        let (wd, h, d, f) = (self.width, self.heads, HEAD, self.ffn);
        let (rows, n) = (sc.rows, sc.tokens);
        let s = rows + n;
        // the conditioning vector, and every block's modulation from silu(vec)
        self.embed(ops, sc, &self.time_in, &timestep_embedding(t, 256, 1000.0, 1000.0), false)?;
        if let Some(gi) = &self.guidance_in {
            self.embed(ops, sc, gi, &timestep_embedding(g, 256, 1000.0, 1000.0), true)?;
        }
        ops.silu(sc.vec.ptr(), sc.vech.ptr(), wd)?;
        let mut at = 0usize;
        let mut modv = |l: &Lin| -> Result<usize> {
            let a = at;
            ops.lin(l, sc.vech.ptr(), 1, fp(&sc.mods, a).cast(), Dt::F32)?;
            at += l.n;
            Ok(a)
        };
        let mp = |a: usize, i: usize| -> P { fp(&sc.mods, a + i * wd) as P };
        // the stream: [context | latents]
        sc.x.copy_within(0, &sc.txt0, 0, rows * wd * 4)?;
        ops.to_half(x.ptr(), sc.xin.ptr(), n * self.channels)?;
        ops.lin(&self.latent_in, sc.xin.ptr(), n, fp(&sc.x, rows * wd).cast(), Dt::F32)?;
        for (bi, b) in self.doubles.iter().enumerate() {
            let mi = modv(&b.img.m)?;
            let mt = modv(&b.txt.m)?;
            // (stream, its rows' start, count, its modulation)
            let parts = [(&b.txt, 0usize, rows, mt), (&b.img, rows, n, mi)];
            for (st, r0, m, mo) in parts {
                ops.ln(fp(&sc.x, r0 * wd).cast(), false, m, wd, EPS, none(), none(), mp(mo, 1), mp(mo, 0), hp(&sc.h, r0 * wd), true)?;
                ops.lin(&st.qkv, hp(&sc.h, r0 * wd), m, hp(&sc.qkv, r0 * 3 * wd), Dt::F16)?;
                ops.head_norm(hp(&sc.qkv, r0 * 3 * wd), m, h, d, 3 * wd, st.qn.ptr(), none(), EPS, true)?;
                ops.head_norm(hp(&sc.qkv, r0 * 3 * wd + wd), m, h, d, 3 * wd, st.kn.ptr(), none(), EPS, true)?;
            }
            ops.attention(sc.qkv.ptr(), hp(&sc.qkv, wd), hp(&sc.qkv, 2 * wd), s, h, d, 3 * wd, sc.att.ptr())?;
            for (st, r0, m, mo) in parts {
                let xr = fp(&sc.x, r0 * wd).cast();
                ops.lin(&st.proj, hp(&sc.att, r0 * wd), m, sc.tmp.ptr(), Dt::F16)?;
                ops.add(xr, sc.tmp.ptr(), true, m, wd, mp(mo, 2))?;
                ops.ln(xr, false, m, wd, EPS, none(), none(), mp(mo, 4), mp(mo, 3), sc.h.ptr(), true)?;
                ops.lin(&st.fc1, sc.h.ptr(), m, sc.mlp.ptr(), Dt::F16)?;
                ops.gelu(sc.mlp.ptr(), m * f, true)?;
                ops.lin(&st.fc2, sc.mlp.ptr(), m, sc.tmp.ptr(), Dt::F16)?;
                ops.add(xr, sc.tmp.ptr(), true, m, wd, mp(mo, 5))?;
            }
            if stop_after == Some(bi) {
                return Ok(());
            }
        }
        for b in &self.singles {
            let mo = modv(&b.m)?;
            ops.ln(sc.x.ptr(), false, s, wd, EPS, none(), none(), mp(mo, 1), mp(mo, 0), sc.h.ptr(), true)?;
            ops.lin(&b.qkv, sc.h.ptr(), s, sc.qkv.ptr(), Dt::F16)?;
            ops.lin(&b.mlp, sc.h.ptr(), s, sc.mlp.ptr(), Dt::F16)?;
            ops.head_norm(sc.qkv.ptr(), s, h, d, 3 * wd, b.qn.ptr(), none(), EPS, true)?;
            ops.head_norm(hp(&sc.qkv, wd), s, h, d, 3 * wd, b.kn.ptr(), none(), EPS, true)?;
            ops.attention(sc.qkv.ptr(), hp(&sc.qkv, wd), hp(&sc.qkv, 2 * wd), s, h, d, 3 * wd, sc.att.ptr())?;
            ops.gelu(sc.mlp.ptr(), s * f, true)?;
            ops.lin(&b.att_out, sc.att.ptr(), s, sc.tmp.ptr(), Dt::F16)?;
            ops.lin(&b.mlp_out, sc.mlp.ptr(), s, sc.tmp2.ptr(), Dt::F16)?;
            ops.add(sc.x.ptr(), sc.tmp.ptr(), true, s, wd, mp(mo, 2))?;
            ops.add(sc.x.ptr(), sc.tmp2.ptr(), true, s, wd, mp(mo, 2))?;
        }
        // the final layer: shift | scale
        let mo = modv(&self.final_mod)?;
        ops.ln(fp(&sc.x, rows * wd).cast(), false, n, wd, EPS, none(), none(), mp(mo, 1), mp(mo, 0), sc.h.ptr(), true)?;
        ops.lin(&self.final_lin, sc.h.ptr(), n, out.ptr(), Dt::F32)
    }

    /// The stream after a forward stopped early: (context rows, latent rows) as float32 on the host
    pub fn stream(&self, sc: &Scratch) -> Result<(Vec<f32>, Vec<f32>)> {
        let all = crate::ops::read_f32(&sc.x, (sc.rows + sc.tokens) * self.width)?;
        let (a, b) = all.split_at(sc.rows * self.width);
        Ok((a.to_vec(), b.to_vec()))
    }
}
