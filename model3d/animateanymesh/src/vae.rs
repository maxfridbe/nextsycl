//! DyMeshVAE (`networks/dymeshvae.py`): the rest shape's vertices embedded, mixed with their mesh neighbours
//! (attention over the faces' edges - a CSR, not the reference's dense N x N mask), sampled by farthest points into
//! trajectory tokens and encoded by cross-attention over every vertex; decoded by self-attention over the tokens and
//! every vertex's query over them into 16 frames of offsets.
//!
//! Only what inference reads is computed: the encoder's rest-shape stream (its motion stream feeds the posterior the
//! sampler replaces with noise), and its rest latents `x0` (the flow is clamped to them).

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ops::{Lin, Ln, Ops, Weights};

pub const DIM: usize = 256;
pub const LATENT: usize = 32;
pub const FRAMES: usize = 16;
const EMBED: usize = 48;
const DEC_HEADS: usize = 8;

struct Ffn {
    norm: Ln,
    up: Lin,
    down: Lin,
}

impl Ffn {
    fn load(w: &Weights, block: &str, i: usize) -> Result<Ffn> {
        Ok(Ffn {
            norm: w.ln(&format!("{block}.1.norm{i}"), DIM)?,
            up: w.lin(&format!("{block}.1.ffn.ffn{i}.0"), 8 * DIM, DIM, true)?,
            down: w.lin(&format!("{block}.1.ffn.ffn{i}.2"), DIM, 4 * DIM, true)?,
        })
    }

    /// x [m, DIM] += ffn(norm(x)); t: m DIM floats, u: m 8 DIM, g: m 4 DIM
    fn add(&self, o: &Ops, x: &DevBuf, m: usize, t: &DevBuf, u: &DevBuf, g: &DevBuf) -> Result<()> {
        o.ln(x, m, DIM, 1e-5, self.norm.norm(), t)?;
        o.linear(t, m, &self.up, u)?;
        o.k.geglu(u, m, 4 * DIM, g)?;
        o.linear(g, m, &self.down, t)?;
        o.k.gated_add(x, t, m, DIM, 1, 1, None, None)
    }
}

struct EncBlock {
    norm_q: Ln,
    norm_k: Ln,
    norm_v1: Ln,
    to_q: Lin,
    to_k: Lin,
    to_v1: Lin,
    to_out1: Lin,
    ffn1: Ffn,
}

struct DecBlock {
    norm_q: Ln,
    norm_k: Ln,
    norm_v1: Ln,
    norm_v2: Ln,
    to_q: Lin,
    to_k: Lin,
    to_v1: Lin,
    to_v2: Lin,
    to_out1: Lin,
    to_out2: Lin,
    ffn1: Ffn,
    ffn2: Ffn,
}

pub struct Vae {
    basis: Vec<f32>,
    point_mlp: Lin,
    nb_norm: [Ln; 3],
    nb_qkv: [Lin; 3],
    nb_out: Lin,
    enc: Vec<EncBlock>,
    mean_x0: Lin,
    proj_x0: Lin,
    proj_xt: Lin,
    dec: Vec<DecBlock>,
    fc_query: Lin,
    fin_norm: [Ln; 3],
    fin_q: Lin,
    fin_k: Lin,
    fin_v: Lin,
    fin_out: Lin,
    to_outputs: Lin,
}

/// The mesh as the encoder sees it: L rows (the vertices, then zero padding), their neighbours
pub struct Shape {
    /// the vertices' rows (positions), N of them
    pub n: usize,
    /// rows with padding: max(4096, N)
    pub l: usize,
    /// every row's embedding after the neighbour attention [L, DIM] (the decoder's queries read it too)
    pub embed: DevBuf,
}

impl Vae {
    pub fn load(w: &Weights) -> Result<Vae> {
        let (bs, basis) = w.host("point_embed.basis")?;
        if bs != [3, EMBED / 2] {
            return Err(Error(format!("point_embed.basis is {bs:?}, not [3, 24]")));
        }
        let mut enc = Vec::new();
        for b in 0..8 {
            let p = format!("enc_blocks.{b}");
            enc.push(EncBlock {
                norm_q: w.ln(&format!("{p}.0.norm_q"), DIM)?,
                norm_k: w.ln(&format!("{p}.0.norm_k"), DIM)?,
                norm_v1: w.ln(&format!("{p}.0.norm_v1"), DIM)?,
                to_q: w.lin(&format!("{p}.0.attn.to_q"), DIM, DIM, false)?,
                to_k: w.lin(&format!("{p}.0.attn.to_k"), DIM, DIM, false)?,
                to_v1: w.lin(&format!("{p}.0.attn.to_v1"), DIM, DIM, false)?,
                to_out1: w.lin(&format!("{p}.0.attn.to_out1"), DIM, DIM, true)?,
                ffn1: Ffn::load(w, &p, 1)?,
            });
        }
        let mut dec = Vec::new();
        for b in 0..8 {
            let p = format!("dec_blocks.{b}");
            let inner = DEC_HEADS * 64;
            dec.push(DecBlock {
                norm_q: w.ln(&format!("{p}.0.norm_q"), DIM)?,
                norm_k: w.ln(&format!("{p}.0.norm_k"), DIM)?,
                norm_v1: w.ln(&format!("{p}.0.norm_v1"), DIM)?,
                norm_v2: w.ln(&format!("{p}.0.norm_v2"), DIM)?,
                to_q: w.lin(&format!("{p}.0.attn.to_q"), inner, DIM, false)?,
                to_k: w.lin(&format!("{p}.0.attn.to_k"), inner, DIM, false)?,
                to_v1: w.lin(&format!("{p}.0.attn.to_v1"), inner, DIM, false)?,
                to_v2: w.lin(&format!("{p}.0.attn.to_v2"), inner, DIM, false)?,
                to_out1: w.lin(&format!("{p}.0.attn.to_out1"), DIM, inner, true)?,
                to_out2: w.lin(&format!("{p}.0.attn.to_out2"), DIM, inner, true)?,
                ffn1: Ffn::load(w, &p, 1)?,
                ffn2: Ffn::load(w, &p, 2)?,
            });
        }
        let nb = "neighbor_layers.0";
        Ok(Vae {
            basis,
            point_mlp: w.lin("point_embed.mlp", DIM, EMBED + 3, true)?,
            nb_norm: [w.ln(&format!("{nb}.norm_q"), DIM)?, w.ln(&format!("{nb}.norm_k"), DIM)?, w.ln(&format!("{nb}.norm_v"), DIM)?],
            nb_qkv: [w.lin(&format!("{nb}.fn.to_q"), 2 * DIM, DIM, false)?, w.lin(&format!("{nb}.fn.to_k"), 2 * DIM, DIM, false)?,
                     w.lin(&format!("{nb}.fn.to_v"), 2 * DIM, DIM, false)?],
            nb_out: w.lin(&format!("{nb}.fn.to_out"), DIM, 2 * DIM, true)?,
            enc,
            mean_x0: w.lin("mean_fc_x0", LATENT, DIM, true)?,
            proj_x0: w.lin("proj_x0", DIM, LATENT, true)?,
            proj_xt: w.lin("proj_xt", DIM, LATENT, true)?,
            dec,
            fc_query: w.lin("fc_query", 2 * DIM, DIM, true)?,
            fin_norm: [w.ln("decoder_final_ca.norm_q", 2 * DIM)?, w.ln("decoder_final_ca.norm_k", DIM)?, w.ln("decoder_final_ca.norm_v", DIM)?],
            fin_q: w.lin("decoder_final_ca.fn.to_q", 2 * DIM, 2 * DIM, false)?,
            fin_k: w.lin("decoder_final_ca.fn.to_k", 2 * DIM, DIM, false)?,
            fin_v: w.lin("decoder_final_ca.fn.to_v", 2 * DIM, DIM, false)?,
            fin_out: w.lin("decoder_final_ca.fn.to_out", 2 * DIM, 2 * DIM, true)?,
            to_outputs: w.lin("to_outputs", 3 * FRAMES, 2 * DIM, true)?,
        })
    }

    /// The point embedding's input features of `pos` [L, 3]: sin and cos of the positions on the basis, then the
    /// positions (PointEmbed.embed), [L, 51]
    pub fn features(&self, pos: &[[f32; 3]]) -> Vec<f32> {
        let e = EMBED / 2;
        let mut out = Vec::with_capacity(pos.len() * (EMBED + 3));
        for p in pos {
            let proj: Vec<f32> = (0..e).map(|j| p[0] * self.basis[j] + p[1] * self.basis[e + j] + p[2] * self.basis[2 * e + j]).collect();
            out.extend(proj.iter().map(|x| x.sin()));
            out.extend(proj.iter().map(|x| x.cos()));
            out.extend_from_slice(p);
        }
        out
    }

    /// The point embedding of every row [L, DIM]
    pub fn embed0(&self, o: &Ops, pos: &[[f32; 3]]) -> Result<DevBuf> {
        let f = DevBuf::from_f32(&o.gpu, &self.features(pos))?;
        let out = o.buf(pos.len() * DIM)?;
        o.linear(&f, pos.len(), &self.point_mlp, &out)?;
        Ok(out)
    }

    /// x [L, DIM] += the neighbour attention's (rowptr / cols: the CSR of the faces' edges, int32 buffers)
    pub fn neighbours(&self, o: &Ops, x: &DevBuf, l: usize, rowptr: &DevBuf, cols: &DevBuf) -> Result<()> {
        let t = o.buf(l * DIM)?;
        let qkv: Vec<DevBuf> = (0..3).map(|_| o.buf(l * 2 * DIM)).collect::<Result<_>>()?;
        for ((n, lin), out) in self.nb_norm.iter().zip(&self.nb_qkv).zip(&qkv) {
            o.ln(x, l, DIM, 1e-5, n.norm(), &t)?;
            o.linear(&t, l, lin, out)?;
        }
        let a = o.buf(l * 2 * DIM)?;
        o.k.csr_attention(&qkv[0], &qkv[1], &qkv[2], l, 2 * DIM, rowptr, cols, 1.0 / ((2 * DIM) as f32).sqrt(), &a)?;
        o.linear(&a, l, &self.nb_out, &t)?;
        o.k.gated_add(x, &t, l, DIM, 1, 1, None, None)
    }

    /// The rest latents x0 [K, 32] of the trajectory tokens `idx` (int32 [K], rows of the shape)
    pub fn encode(&self, o: &Ops, sh: &Shape, idx: &DevBuf, k: usize) -> Result<DevBuf> {
        let l = sh.l;
        let q0 = o.buf(k * DIM)?;
        o.k.gather(&sh.embed, idx, k, DIM, &q0)?;
        let (tq, tk, tv) = (o.buf(k * DIM)?, o.buf(l * DIM)?, o.buf(l * DIM)?);
        let (qq, kk, vv, a) = (o.buf(k * DIM)?, o.buf(l * DIM)?, o.buf(l * DIM)?, o.buf(k * DIM)?);
        let (u, g) = (o.buf(k * 8 * DIM)?, o.buf(k * 4 * DIM)?);
        let scratch = o.buf((k * l).min(32 << 20).max(l))?;
        for b in &self.enc {
            o.ln(&q0, k, DIM, 1e-5, b.norm_q.norm(), &tq)?;
            o.ln(&sh.embed, l, DIM, 1e-5, b.norm_k.norm(), &tk)?;
            o.ln(&sh.embed, l, DIM, 1e-5, b.norm_v1.norm(), &tv)?;
            o.linear(&tq, k, &b.to_q, &qq)?;
            o.linear(&tk, l, &b.to_k, &kk)?;
            o.linear(&tv, l, &b.to_v1, &vv)?;
            o.k.attention((&qq, DIM, DIM), (&kk, DIM, DIM), (&vv, DIM, DIM), 1, k, l, 1, DIM, DIM, 1.0 / (DIM as f32).sqrt(), false, &scratch,
                          (&a, DIM))?;
            o.linear(&a, k, &b.to_out1, &tq)?;
            o.k.gated_add(&q0, &tq, k, DIM, 1, 1, None, None)?;
            b.ffn1.add(o, &q0, k, &tq, &u, &g)?;
        }
        let x0 = o.buf(k * LATENT)?;
        o.linear(&q0, k, &self.mean_x0, &x0)?;
        Ok(x0)
    }

    /// The 16 frames' offsets [n, 48] (frame t's at 3t) of the shape's first n rows from the latents x0, xt [K, 32]
    pub fn decode(&self, o: &Ops, sh: &Shape, x0l: &DevBuf, xtl: &DevBuf, k: usize) -> Result<DevBuf> {
        let n = sh.n;
        let inner = DEC_HEADS * 64;
        let (x0, xt) = (o.buf(k * DIM)?, o.buf(k * DIM)?);
        o.linear(x0l, k, &self.proj_x0, &x0)?;
        o.linear(xtl, k, &self.proj_xt, &xt)?;
        let t: Vec<DevBuf> = (0..4).map(|_| o.buf(k * DIM)).collect::<Result<_>>()?;
        let p: Vec<DevBuf> = (0..4).map(|_| o.buf(k * inner)).collect::<Result<_>>()?;
        let (a1, a2) = (o.buf(k * inner)?, o.buf(k * inner)?);
        let (u, g) = (o.buf(k * 8 * DIM)?, o.buf(k * 4 * DIM)?);
        let scratch = o.buf((DEC_HEADS * k * k).min(32 << 20).max(k))?;
        for b in &self.dec {
            o.ln(&x0, k, DIM, 1e-5, b.norm_q.norm(), &t[0])?;
            o.ln(&x0, k, DIM, 1e-5, b.norm_k.norm(), &t[1])?;
            o.ln(&x0, k, DIM, 1e-5, b.norm_v1.norm(), &t[2])?;
            o.ln(&xt, k, DIM, 1e-5, b.norm_v2.norm(), &t[3])?;
            o.linear(&t[0], k, &b.to_q, &p[0])?;
            o.linear(&t[1], k, &b.to_k, &p[1])?;
            o.linear(&t[2], k, &b.to_v1, &p[2])?;
            o.linear(&t[3], k, &b.to_v2, &p[3])?;
            let sc = 1.0 / 8.0;
            o.k.attention((&p[0], inner, 64), (&p[1], inner, 64), (&p[2], inner, 64), 1, k, k, DEC_HEADS, 64, 64, sc, false, &scratch, (&a1, inner))?;
            o.k.attention((&p[0], inner, 64), (&p[1], inner, 64), (&p[3], inner, 64), 1, k, k, DEC_HEADS, 64, 64, sc, false, &scratch, (&a2, inner))?;
            o.linear(&a1, k, &b.to_out1, &t[0])?;
            o.linear(&a2, k, &b.to_out2, &t[1])?;
            o.k.gated_add(&x0, &t[0], k, DIM, 1, 1, None, None)?;
            o.k.gated_add(&xt, &t[1], k, DIM, 1, 1, None, None)?;
            b.ffn1.add(o, &x0, k, &t[0], &u, &g)?;
            b.ffn2.add(o, &xt, k, &t[0], &u, &g)?;
        }
        // every vertex's query over the tokens
        let d2 = 2 * DIM;
        let (qin, qn, q, att) = (o.buf(n * d2)?, o.buf(n * d2)?, o.buf(n * d2)?, o.buf(n * d2)?);
        o.linear(&sh.embed, n, &self.fc_query, &qin)?;
        o.ln(&qin, n, d2, 1e-5, self.fin_norm[0].norm(), &qn)?;
        o.linear(&qn, n, &self.fin_q, &q)?;
        o.ln(&x0, k, DIM, 1e-5, self.fin_norm[1].norm(), &t[0])?;
        o.ln(&xt, k, DIM, 1e-5, self.fin_norm[2].norm(), &t[1])?;
        let (kk, vv) = (o.buf(k * d2)?, o.buf(k * d2)?);
        o.linear(&t[0], k, &self.fin_k, &kk)?;
        o.linear(&t[1], k, &self.fin_v, &vv)?;
        let scratch = o.buf((n * k).min(32 << 20).max(k))?;
        o.k.attention((&q, d2, d2), (&kk, d2, d2), (&vv, d2, d2), 1, n, k, 1, d2, d2, 1.0 / (d2 as f32).sqrt(), false, &scratch, (&att, d2))?;
        o.linear(&att, n, &self.fin_out, &qn)?;
        let out = o.buf(n * 3 * FRAMES)?;
        o.linear(&qn, n, &self.to_outputs, &out)?;
        Ok(out)
    }
}
