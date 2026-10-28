//! The shape encoder (Michelangelo's ShapeAsLatentPerceiverEncoder, `sal_perceiver.py`) as TokenRig has it: 2,048 of the
//! points drawn at random, 512 of them by farthest point sampling as the queries; every point (with its normal)
//! Fourier-embedded as the keys; one cross-attention block, 8 self-attention blocks, a final norm: [512, 512].

use nextsycl_core::DevBuf;
use nextsycl_model3d::Result;

use crate::ffi::GELU;
use crate::ops::{Lin, Ln, Ops, Weights};

pub const WIDTH: usize = 512;
pub const TOKENS: usize = 512;
const HEADS: usize = 8;
const LAYERS: usize = 8;
const FREQS: usize = 8;
const FEATS: usize = 3 + 3 * 2 * FREQS + 3;

struct Mlp {
    fc: Lin,
    proj: Lin,
}

impl Mlp {
    fn load(w: &Weights, p: &str) -> Result<Mlp> {
        Ok(Mlp { fc: w.lin(&format!("{p}.c_fc"), 4 * WIDTH, WIDTH, true)?, proj: w.lin(&format!("{p}.c_proj"), WIDTH, 4 * WIDTH, true)? })
    }
}

struct SelfBlock {
    ln_1: Ln,
    c_qkv: Lin,
    c_proj: Lin,
    ln_2: Ln,
    mlp: Mlp,
}

pub struct Encoder {
    input_proj: Lin,
    ln_1: Ln,
    ln_2: Ln,
    ln_3: Ln,
    c_q: Lin,
    c_kv: Lin,
    c_proj: Lin,
    mlp: Mlp,
    blocks: Vec<SelfBlock>,
    ln_post: Ln,
}

/// The Fourier features of points and their normals: [x, sin(x 2^k), cos(x 2^k), n] per row (54)
pub fn features(pts: &[[f32; 3]], nrm: &[[f32; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pts.len() * FEATS);
    for (p, n) in pts.iter().zip(nrm) {
        let e: Vec<f32> = (0..3).flat_map(|c| (0..FREQS).map(move |k| p[c] * (1u32 << k) as f32)).collect();
        out.extend_from_slice(p);
        out.extend(e.iter().map(|x| x.sin()));
        out.extend(e.iter().map(|x| x.cos()));
        out.extend_from_slice(n);
    }
    out
}

/// torch_cluster's farthest point sampling of `pts` (the first point first; squared distances summed x, y, z): k
/// indices into `pts`
pub fn fps(pts: &[[f32; 3]], k: usize) -> Vec<usize> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let d2 = |a: [f32; 3], b: [f32; 3]| {
        let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        (x * x + y * y) + z * z
    };
    let mut dist: Vec<f32> = pts.iter().map(|p| d2(*p, pts[0])).collect();
    let mut out = vec![0usize];
    for _ in 1..k.min(n) {
        let mut j = 0;
        for i in 1..n {
            if dist[i] > dist[j] {
                j = i;
            }
        }
        out.push(j);
        for (i, d) in dist.iter_mut().enumerate() {
            *d = d.min(d2(pts[i], pts[j]));
        }
    }
    out
}

impl Encoder {
    pub fn load(w: &Weights) -> Result<Encoder> {
        let p = "mesh_encoder.encoder";
        let mut blocks = Vec::new();
        for b in 0..LAYERS {
            let q = format!("{p}.self_attn.resblocks.{b}");
            blocks.push(SelfBlock {
                ln_1: w.ln(&format!("{q}.ln_1"), WIDTH)?,
                c_qkv: w.lin(&format!("{q}.attn.c_qkv"), 3 * WIDTH, WIDTH, false)?,
                c_proj: w.lin(&format!("{q}.attn.c_proj"), WIDTH, WIDTH, true)?,
                ln_2: w.ln(&format!("{q}.ln_2"), WIDTH)?,
                mlp: Mlp::load(w, &format!("{q}.mlp"))?,
            });
        }
        let c = format!("{p}.cross_attn");
        Ok(Encoder {
            input_proj: w.lin(&format!("{p}.input_proj"), WIDTH, FEATS, true)?,
            ln_1: w.ln(&format!("{c}.ln_1"), WIDTH)?,
            ln_2: w.ln(&format!("{c}.ln_2"), WIDTH)?,
            ln_3: w.ln(&format!("{c}.ln_3"), WIDTH)?,
            c_q: w.lin(&format!("{c}.attn.c_q"), WIDTH, WIDTH, false)?,
            c_kv: w.lin(&format!("{c}.attn.c_kv"), 2 * WIDTH, WIDTH, false)?,
            c_proj: w.lin(&format!("{c}.attn.c_proj"), WIDTH, WIDTH, true)?,
            mlp: Mlp::load(w, &format!("{c}.mlp"))?,
            blocks,
            ln_post: w.ln(&format!("{p}.ln_post"), WIDTH)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp(&self, o: &Ops, m: &Mlp, x: &DevBuf, rows: usize, ln: &Ln, t: &DevBuf, h: &DevBuf) -> Result<()> {
        o.ln(x, rows, WIDTH, Some(ln), t)?;
        o.linear(t, rows, &m.fc, h)?;
        o.k.act(h, rows * 4 * WIDTH, GELU)?;
        o.linear(h, rows, &m.proj, t)?;
        o.k.add(x, t, rows * WIDTH)
    }

    /// The latents [512, 512] of the points (with their normals): `queries` the indices of the 512 query points
    /// (the random subset's farthest points)
    pub fn encode(&self, o: &Ops, pts: &[[f32; 3]], nrm: &[[f32; 3]], queries: &[usize]) -> Result<DevBuf> {
        let n = pts.len();
        let t = queries.len();
        // the keys: every point
        let f = DevBuf::from_f32(&o.gpu, &features(pts, nrm))?;
        let data = o.buf(n * WIDTH)?;
        o.linear(&f, n, &self.input_proj, &data)?;
        drop(f);
        o.ln(&data, n, WIDTH, Some(&self.ln_2), &data)?;
        let kv = o.buf(n * 2 * WIDTH)?;
        o.linear(&data, n, &self.c_kv, &kv)?;
        drop(data);
        // the queries
        let qp: Vec<[f32; 3]> = queries.iter().map(|&i| pts[i]).collect();
        let qn: Vec<[f32; 3]> = queries.iter().map(|&i| nrm[i]).collect();
        let fq = DevBuf::from_f32(&o.gpu, &features(&qp, &qn))?;
        let x = o.buf(t * WIDTH)?;
        o.linear(&fq, t, &self.input_proj, &x)?;
        let (tq, q, a, h) = (o.buf(t * WIDTH)?, o.buf(t * WIDTH)?, o.buf(t * WIDTH)?, o.buf(t * 4 * WIDTH)?);
        let hd = WIDTH / HEADS;
        let scratch = o.buf((HEADS * t * n).min(64 << 20).max(n))?;
        o.ln(&x, t, WIDTH, Some(&self.ln_1), &tq)?;
        o.linear(&tq, t, &self.c_q, &q)?;
        // per head: k then v, 64 each, in the kv row
        let kv_v = kv.view(hd * 4, kv.len - hd * 4)?;
        o.k.attention((&q, WIDTH, hd), (&kv, 2 * WIDTH, 2 * hd), (&kv_v, 2 * WIDTH, 2 * hd), 1, t, n, HEADS, HEADS, hd, hd, 1.0 / (hd as f32).sqrt(), false,
                      &scratch, (&a, WIDTH))?;
        drop(kv);
        o.linear(&a, t, &self.c_proj, &tq)?;
        o.k.add(&x, &tq, t * WIDTH)?;
        self.mlp(o, &self.mlp, &x, t, &self.ln_3, &tq, &h)?;
        let qkv = o.buf(t * 3 * WIDTH)?;
        for b in &self.blocks {
            o.ln(&x, t, WIDTH, Some(&b.ln_1), &tq)?;
            o.linear(&tq, t, &b.c_qkv, &qkv)?;
            let k = qkv.view(hd * 4, qkv.len - hd * 4)?;
            let v = qkv.view(2 * hd * 4, qkv.len - 2 * hd * 4)?;
            o.k.attention((&qkv, 3 * WIDTH, 3 * hd), (&k, 3 * WIDTH, 3 * hd), (&v, 3 * WIDTH, 3 * hd), 1, t, t, HEADS, HEADS, hd, hd, 1.0 / (hd as f32).sqrt(),
                          false, &scratch, (&a, WIDTH))?;
            o.linear(&a, t, &b.c_proj, &tq)?;
            o.k.add(&x, &tq, t * WIDTH)?;
            self.mlp(o, &b.mlp, &x, t, &b.ln_2, &tq, &h)?;
        }
        o.ln(&x, t, WIDTH, Some(&self.ln_post), &x)?;
        Ok(x)
    }
}

/// The encoder's query points: 2,048 of `n` drawn at random (without repeats when there are enough), their farthest
/// 512 - (subset, farthest points as indices into the subset)
pub fn queries(pts: &[[f32; 3]], rng: &mut crate::ops::Rng) -> (Vec<usize>, Vec<usize>) {
    subset_fps(pts, 4 * TOKENS, TOKENS, rng)
}

/// `want` of the points drawn at random (without repeats when there are enough), their `k` farthest points: (the subset's
/// indices into `pts`, the farthest points as indices into the subset)
pub fn subset_fps(pts: &[[f32; 3]], want: usize, k: usize, rng: &mut crate::ops::Rng) -> (Vec<usize>, Vec<usize>) {
    let n = pts.len();
    let ind: Vec<usize> = if n >= want {
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..want {
            let j = i + (rng.unit() * (n - i) as f64) as usize;
            all.swap(i, j.min(n - 1));
        }
        all.truncate(want);
        all
    } else {
        (0..want).map(|_| ((rng.unit() * n as f64) as usize).min(n - 1)).collect()
    };
    let sub: Vec<[f32; 3]> = ind.iter().map(|&i| pts[i]).collect();
    let f = fps(&sub, k);
    (ind, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn farthest_points_spread() {
        let pts: Vec<[f32; 3]> = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect();
        // from 0: the far end, then the middle
        assert_eq!(fps(&pts, 3), vec![0, 9, 4]);
    }

    #[test]
    fn features_are_54_a_point() {
        let f = features(&[[0.5, 0.0, -0.25]], &[[0.0, 0.0, 1.0]]);
        assert_eq!(f.len(), 54);
        assert_eq!(&f[..3], &[0.5, 0.0, -0.25]);
        // sin(0.5 * 2) at index 3 + 1
        assert!((f[4] - 1.0f32.sin()).abs() < 1e-7);
        assert_eq!(f[51..], [0.0, 0.0, 1.0]);
    }
}
