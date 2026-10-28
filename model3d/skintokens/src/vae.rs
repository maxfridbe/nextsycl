//! The skin FSQ-CVAE (SkinFSQCVAEModel, src/model/skin_vae), the part of it TokenRig runs: the condition encoder (the surface
//! samples, Fourier-embedded, to 384 tokens), the quantizer's codebook (a skin token is one of 32,768 five-level FSQ codes,
//! four a bone), and the decoder, which reads a bone's four codes and the condition tokens through ten self-attention blocks
//! and answers, for every surface sample, one cross-attention block away, how much that bone weighs there.
//!
//! The blocks are Tripo2's DiTBlocks (width 768, 12 heads of 64): a LayerNorm, attention, a LayerNorm, a GELU feed-forward.
//! Their attention concatenates the projections before it splits the heads, so a head's q, k and v are slices of one
//! 192-wide (self) or its k and v of one 128-wide (cross) piece of the concatenation - read here by strides.

use nextsycl_core::DevBuf;
use nextsycl_model3d::Result;

use crate::ffi::{GELU, SIGMOID};
use crate::ops::{Lin, Ln, Ops, Weights};

pub const WIDTH: usize = 768;
pub const LATENT: usize = 512;
const HEADS: usize = 12;
const HD: usize = WIDTH / HEADS;
const FREQS: usize = 8;
/// The positional embedding of a sample and its normal
const FEATS: usize = 3 + 2 * 3 * FREQS + 3;
/// The condition tokens, and the skin tokens a bone has
pub const COND_TOKENS: usize = 384;
pub const PER_BONE: usize = 4;
/// Condition encoder: the random subset the queries are picked from
pub const COND_SUBSET: usize = 4 * COND_TOKENS;
const SEQ: usize = PER_BONE + COND_TOKENS;
const LEVELS: [i32; 5] = [8; 5];
/// The surface samples' rows the decoder's cross-attention takes at a time
const CHUNK: usize = 9_000;
/// The bones the decoder's self-attention blocks take at a time
const BONES: usize = 16;

struct Ff {
    norm3: Ln,
    fc: Lin,
    proj: Lin,
}

struct SelfBlock {
    norm1: Ln,
    qkv: Lin,
    out: Lin,
    ff: Ff,
}

struct CrossBlock {
    norm2: Ln,
    q: Lin,
    norm_cross: Ln,
    kv: Lin,
    out: Lin,
    ff: Ff,
}

pub struct Vae {
    cond_in: Lin,
    cond_cross: CrossBlock,
    cond_self: Vec<SelfBlock>,
    cond_norm: Ln,
    cond_quant: Lin,
    post_quant: Lin,
    /// FSQ.project_out [512, 5] and its bias, on the host (a bone's four codes are 20 numbers)
    fsq_w: Vec<f32>,
    fsq_b: Vec<f32>,
    blocks: Vec<SelfBlock>,
    proj_query: Lin,
    cross: CrossBlock,
    norm_out: Ln,
    proj_out_w: DevBuf,
    proj_out_b: f32,
    phase: [f32; FREQS],
}

/// FrequencyPositionalEmbedding (8 frequencies 2^k pi, the PMPE phases) of the points, then their normals: [n, 54]
pub fn features(pts: &[[f32; 3]], nrm: &[[f32; 3]], phase: &[f32; FREQS]) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    let fr: Vec<f32> = (0..FREQS).map(|k| (1u32 << k) as f32 * pi).collect();
    let mut out = Vec::with_capacity(pts.len() * FEATS);
    for (p, n) in pts.iter().zip(nrm) {
        let e: Vec<f32> = (0..3).flat_map(|c| fr.iter().map(move |f| p[c] * f)).collect();
        let ph: Vec<f32> = (0..3).flat_map(|c| (0..FREQS).map(move |k| p[c] * pi * 0.5 + phase[k])).collect();
        out.extend_from_slice(p);
        out.extend(e.iter().zip(&ph).map(|(a, b)| a.sin() + b.sin()));
        out.extend(e.iter().zip(&ph).map(|(a, b)| a.cos() + b.cos()));
        out.extend_from_slice(n);
    }
    out
}

fn phases() -> [f32; FREQS] {
    let mut p = [0f32; FREQS];
    for (i, x) in p.iter_mut().enumerate() {
        let t = (i + 1) as f64 / FREQS as f64;
        *x = (((FREQS as f64).powf(1.0 - t) + t) * 2.0 * std::f64::consts::PI) as f32;
    }
    p
}

/// The level indices of a code: FSQ.indices_to_codes' digits, each (digit - 4) / 4
pub fn fsq_levels(code: u32) -> [f32; 5] {
    let mut out = [0f32; 5];
    let mut basis = 1i32;
    for (i, l) in LEVELS.iter().enumerate() {
        let d = (code as i32 / basis) % l;
        out[i] = (d - l / 2) as f32 / (l / 2) as f32;
        basis *= l;
    }
    out
}

impl Vae {
    pub fn load(w: &Weights) -> Result<Vae> {
        let ff = |p: &str| -> Result<Ff> {
            Ok(Ff { norm3: w.ln(&format!("{p}.norm3"), WIDTH)?, fc: w.lin(&format!("{p}.ff.net.0.proj"), 4 * WIDTH, WIDTH, true)?,
                    proj: w.lin(&format!("{p}.ff.net.2"), WIDTH, 4 * WIDTH, true)? })
        };
        let self_block = |p: &str| -> Result<SelfBlock> {
            let a = format!("{p}.attn1");
            Ok(SelfBlock {
                norm1: w.ln(&format!("{p}.norm1"), WIDTH)?,
                qkv: w.lin_cat(&[&format!("{a}.to_q"), &format!("{a}.to_k"), &format!("{a}.to_v")], &[WIDTH; 3], WIDTH)?,
                out: w.lin(&format!("{a}.to_out.0"), WIDTH, WIDTH, true)?,
                ff: ff(p)?,
            })
        };
        let cross_block = |p: &str| -> Result<CrossBlock> {
            let a = format!("{p}.attn2");
            Ok(CrossBlock {
                norm2: w.ln(&format!("{p}.norm2"), WIDTH)?,
                q: w.lin(&format!("{a}.to_q"), WIDTH, WIDTH, false)?,
                norm_cross: w.ln(&format!("{a}.norm_cross"), WIDTH)?,
                kv: w.lin_cat(&[&format!("{a}.to_k"), &format!("{a}.to_v")], &[WIDTH; 2], WIDTH)?,
                out: w.lin(&format!("{a}.to_out.0"), WIDTH, WIDTH, true)?,
                ff: ff(p)?,
            })
        };
        let c = "cond_encoder";
        let d = "decoder";
        let (_, fsq_w) = w.host("FSQ.project_out.weight")?;
        let (_, fsq_b) = w.host("FSQ.project_out.bias")?;
        let (_, pob) = w.host(&format!("{d}.proj_out.bias"))?;
        Ok(Vae {
            cond_in: w.lin(&format!("{c}.proj_in"), WIDTH, FEATS, true)?,
            cond_cross: cross_block(&format!("{c}.blocks.0"))?,
            cond_self: (1..3).map(|i| self_block(&format!("{c}.blocks.{i}"))).collect::<Result<_>>()?,
            cond_norm: w.ln(&format!("{c}.norm_out"), WIDTH)?,
            cond_quant: w.lin("cond_quant", LATENT, WIDTH, true)?,
            post_quant: w.lin("post_quant", WIDTH, LATENT, true)?,
            fsq_w,
            fsq_b,
            blocks: (0..10).map(|i| self_block(&format!("{d}.blocks.{i}"))).collect::<Result<_>>()?,
            proj_query: w.lin(&format!("{d}.proj_query"), WIDTH, FEATS, true)?,
            cross: cross_block(&format!("{d}.blocks.10"))?,
            norm_out: w.ln(&format!("{d}.norm_out"), WIDTH)?,
            proj_out_w: w.dev(&format!("{d}.proj_out.weight"), &[1, WIDTH])?,
            proj_out_b: pob[0],
            phase: phases(),
        })
    }

    fn ff(&self, o: &Ops, f: &Ff, x: &DevBuf, rows: usize, t: &DevBuf, h: &DevBuf) -> Result<()> {
        o.ln(x, rows, WIDTH, Some(&f.norm3), t)?;
        o.linear(t, rows, &f.fc, h)?;
        o.k.act(h, rows * 4 * WIDTH, GELU)?;
        o.linear(h, rows, &f.proj, t)?;
        o.k.add(x, t, rows * WIDTH)
    }

    /// x [b * n, 768] through a self-attention block, `b` sequences of `n` rows
    fn self_block(&self, o: &Ops, bl: &SelfBlock, x: &DevBuf, b: usize, n: usize, s: &Scratch) -> Result<()> {
        let rows = b * n;
        o.ln(x, rows, WIDTH, Some(&bl.norm1), &s.t)?;
        o.linear(&s.t, rows, &bl.qkv, &s.qkv)?;
        let k = s.qkv.view(HD * 4, s.qkv.len - HD * 4)?;
        let v = s.qkv.view(2 * HD * 4, s.qkv.len - 2 * HD * 4)?;
        o.k.attention((&s.qkv, 3 * WIDTH, 3 * HD), (&k, 3 * WIDTH, 3 * HD), (&v, 3 * WIDTH, 3 * HD), b, n, n, HEADS, HEADS, HD, HD,
                      1.0 / (HD as f32).sqrt(), false, &s.scratch, (&s.a, WIDTH))?;
        o.linear(&s.a, rows, &bl.out, &s.t)?;
        o.k.add(x, &s.t, rows * WIDTH)?;
        self.ff(o, &bl.ff, x, rows, &s.t, &s.h)
    }

    /// The keys and values of a cross-attention block's encoder states `enc` [ne, 768] (normed, projected): [ne, 1536]
    fn cross_kv(&self, o: &Ops, bl: &CrossBlock, enc: &DevBuf, ne: usize) -> Result<DevBuf> {
        let e = o.buf(ne * WIDTH)?;
        o.ln(enc, ne, WIDTH, Some(&bl.norm_cross), &e)?;
        let kv = o.buf(ne * 2 * WIDTH)?;
        o.linear(&e, ne, &bl.kv, &kv)?;
        Ok(kv)
    }

    /// The attention of queries q [rows, 768] over the keys / values kv [ne, 1536]: [rows, 768] into `a`
    #[allow(clippy::too_many_arguments)]
    fn cross_attend(&self, o: &Ops, q: &DevBuf, rows: usize, kv: &DevBuf, ne: usize, scratch: &DevBuf, a: &DevBuf) -> Result<()> {
        let v = kv.view(HD * 4, kv.len - HD * 4)?;
        o.k.attention((q, WIDTH, HD), (kv, 2 * WIDTH, 2 * HD), (&v, 2 * WIDTH, 2 * HD), 1, rows, ne, HEADS, HEADS, HD, HD, 1.0 / (HD as f32).sqrt(), false,
                      scratch, (a, WIDTH))
    }

    /// The condition tokens [384, 512] of the surface samples: `ind` the random subset's indices into them, `fps` its
    /// farthest points (indices into the subset) as the queries
    pub fn cond(&self, o: &Ops, pts: &[[f32; 3]], nrm: &[[f32; 3]], ind: &[usize], fps: &[usize]) -> Result<DevBuf> {
        let n = pts.len();
        let t = fps.len();
        let kvf = DevBuf::from_f32(&o.gpu, &features(pts, nrm, &self.phase))?;
        let e = o.buf(n * WIDTH)?;
        o.linear(&kvf, n, &self.cond_in, &e)?;
        drop(kvf);
        let qi: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        let qp: Vec<[f32; 3]> = qi.iter().map(|&i| pts[i]).collect();
        let qn: Vec<[f32; 3]> = qi.iter().map(|&i| nrm[i]).collect();
        let qf = DevBuf::from_f32(&o.gpu, &features(&qp, &qn, &self.phase))?;
        let x = o.buf(t * WIDTH)?;
        o.linear(&qf, t, &self.cond_in, &x)?;
        let s = Scratch::new(o, 1, t, (HEADS * t * n).min(64 << 20).max(n))?;
        // the cross block
        let bl = &self.cond_cross;
        o.ln(&x, t, WIDTH, Some(&bl.norm2), &s.t)?;
        let q = o.buf(t * WIDTH)?;
        o.linear(&s.t, t, &bl.q, &q)?;
        let kv = self.cross_kv(o, bl, &e, n)?;
        drop(e);
        self.cross_attend(o, &q, t, &kv, n, &s.scratch, &s.a)?;
        o.linear(&s.a, t, &bl.out, &s.t)?;
        o.k.add(&x, &s.t, t * WIDTH)?;
        self.ff(o, &bl.ff, &x, t, &s.t, &s.h)?;
        for b in &self.cond_self {
            self.self_block(o, b, &x, 1, t, &s)?;
        }
        o.ln(&x, t, WIDTH, Some(&self.cond_norm), &x)?;
        let out = o.buf(t * LATENT)?;
        o.linear(&x, t, &self.cond_quant, &out)?;
        Ok(out)
    }

    /// The codes' vectors [n * 512] (FSQ.indices_to_codes: the levels through project_out) of FSQ code indices
    pub fn codes(&self, idx: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(idx.len() * LATENT);
        for &c in idx {
            let l = fsq_levels(c);
            for r in 0..LATENT {
                let mut s = 0f32;
                for (i, x) in l.iter().enumerate() {
                    s += self.fsq_w[r * 5 + i] * x;
                }
                out.push(s + self.fsq_b[r]);
            }
        }
        out
    }

    /// The bones' weights over the surface samples: `z` the bones' codes' vectors [J * 4 * 512], `cond` the condition tokens;
    /// [N, J] (the bone's column its sigmoid output), and the first bone's decoder states (the cross-attention's keys and
    /// values) [388, 768]
    #[allow(clippy::too_many_arguments)]
    pub fn decode(&self, o: &Ops, z: &[f32], j: usize, cond: &DevBuf, pts: &[[f32; 3]], nrm: &[[f32; 3]], progress: &mut dyn FnMut(usize) -> Result<()>)
                  -> Result<(Vec<f32>, Vec<f32>)> {
        let n = pts.len();
        // what every bone shares: the queries' embeddings and the cross block's q
        let f = DevBuf::from_f32(&o.gpu, &features(pts, nrm, &self.phase))?;
        let x0 = o.buf(n * WIDTH)?;
        o.linear(&f, n, &self.proj_query, &x0)?;
        drop(f);
        let q = o.buf(n * WIDTH)?;
        {
            let t = o.buf(n * WIDTH)?;
            o.ln(&x0, n, WIDTH, Some(&self.cross.norm2), &t)?;
            o.linear(&t, n, &self.cross.q, &q)?;
        }
        let mut skin = vec![0f32; n * j];
        let mut kv0 = Vec::new();
        let cond_host = cond.to_f32()?;
        let ch = CHUNK.min(n);
        let cs = Scratch::new(o, 1, ch, (HEADS * ch * SEQ).clamp(SEQ, 64 << 20))?;
        let xr = o.buf(ch * WIDTH)?;
        let col = o.buf(ch)?;
        for b0 in (0..j).step_by(BONES) {
            let nb = BONES.min(j - b0);
            // [z (4 rows) | the condition tokens] a bone
            let mut inp = Vec::with_capacity(nb * SEQ * LATENT);
            for b in b0..b0 + nb {
                inp.extend_from_slice(&z[b * PER_BONE * LATENT..(b + 1) * PER_BONE * LATENT]);
                inp.extend_from_slice(&cond_host);
            }
            let inp = DevBuf::from_f32(&o.gpu, &inp)?;
            let x = o.buf(nb * SEQ * WIDTH)?;
            o.linear(&inp, nb * SEQ, &self.post_quant, &x)?;
            drop(inp);
            let s = Scratch::new(o, nb, SEQ, (HEADS * nb * SEQ * SEQ).clamp(SEQ, 64 << 20))?;
            for bl in &self.blocks {
                self.self_block(o, bl, &x, nb, SEQ, &s)?;
            }
            if b0 == 0 {
                kv0 = x.view(0, SEQ * WIDTH * 4)?.to_f32()?;
            }
            drop(s);
            for bi in 0..nb {
                let kvb = x.view(bi * SEQ * WIDTH * 4, SEQ * WIDTH * 4)?;
                let kv = self.cross_kv(o, &self.cross, &kvb, SEQ)?;
                for r0 in (0..n).step_by(ch) {
                    let rows = ch.min(n - r0);
                    let qc = q.view(r0 * WIDTH * 4, rows * WIDTH * 4)?;
                    self.cross_attend(o, &qc, rows, &kv, SEQ, &cs.scratch, &cs.a)?;
                    o.linear(&cs.a, rows, &self.cross.out, &cs.t)?;
                    xr.copy_within(0, &x0, r0 * WIDTH * 4, rows * WIDTH * 4)?;
                    o.k.add(&xr, &cs.t, rows * WIDTH)?;
                    self.ff(o, &self.cross.ff, &xr, rows, &cs.t, &cs.h)?;
                    o.ln(&xr, rows, WIDTH, Some(&self.norm_out), &cs.t)?;
                    o.k.rowdot(&cs.t, rows, WIDTH, &self.proj_out_w, self.proj_out_b, SIGMOID, &col)?;
                    let v = col.to_f32()?;
                    for (r, x) in v.iter().take(rows).enumerate() {
                        skin[(r0 + r) * j + b0 + bi] = *x;
                    }
                }
                progress(b0 + bi + 1)?;
            }
        }
        Ok((skin, kv0))
    }
}

/// The scratch of the blocks on `b` sequences of `n` rows
struct Scratch {
    t: DevBuf,
    qkv: DevBuf,
    a: DevBuf,
    h: DevBuf,
    scratch: DevBuf,
}

impl Scratch {
    fn new(o: &Ops, b: usize, n: usize, scores: usize) -> Result<Scratch> {
        let rows = b * n;
        Ok(Scratch { t: o.buf(rows * WIDTH)?, qkv: o.buf(rows * 3 * WIDTH)?, a: o.buf(rows * WIDTH)?, h: o.buf(rows * 4 * WIDTH)?, scratch: o.buf(scores)? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_is_five_levels() {
        // code 0: every level at the bottom; 4 + 4 * 8 + ...: the middle ones are 0
        assert_eq!(fsq_levels(0), [-1.0; 5]);
        let mid = 4 + 4 * 8 + 4 * 64 + 4 * 512 + 4 * 4096;
        assert_eq!(fsq_levels(mid), [0.0; 5]);
        assert_eq!(fsq_levels(32_767), [0.75; 5]);
        assert_eq!(fsq_levels(1)[0], -0.75);
    }

    #[test]
    fn features_are_54_a_point() {
        let f = features(&[[0.5, 0.0, -0.25]], &[[0.0, 0.0, 1.0]], &phases());
        assert_eq!(f.len(), FEATS);
        assert_eq!(&f[..3], &[0.5, 0.0, -0.25]);
        assert_eq!(f[51..], [0.0, 0.0, 1.0]);
        // x = 0 at frequency 0: sin(0) + sin(phase 0)
        let ph = phases();
        assert!((f[3] - ((0.5f32 * std::f32::consts::PI).sin() + (0.5 * std::f32::consts::PI * 0.5 + ph[0]).sin())).abs() < 1e-5);
    }
}
