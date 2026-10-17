//! The backbone: Ministral 3B (Mistral's decoder: 26 layers of 3,072, 32 query / 8 key-value heads of 128, a 9,216
//! SwiGLU, RMS norm ε 1e-5, RoPE θ 1e6). The consolidated checkpoint rotates adjacent pairs; q and k are loaded with
//! their rows reordered (weights.rs pair_halves) so the rotate-half kernel does the same rotation. Weights in half; a
//! half cache [B, Hkv, T, 128].

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

pub const HEAD: usize = 128;

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Lm {
    layers: Vec<Layer>,
    norm: DevBuf,
    pub hidden: usize,
    heads: usize,
    kv: usize,
    ffn: usize,
    eps: f32,
    /// RoPE's frequencies [64]
    inv_freq: DevBuf,
}

/// One decoder's cache and work buffers: up to `rows` rows a pass, `b` sequences, `t` positions
pub struct Cache {
    t: usize,
    b: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    xh: DevBuf,
    part: DevBuf,
    pub out: DevBuf,
}

/// Plain RoPE frequencies for heads of 128, θ: 1 / θ^(2i / 128) in float32 (as vLLM's rotary cache)
pub fn rope_freqs(theta: f64) -> Vec<f32> {
    (0..HEAD / 2).map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / HEAD as f32)).collect()
}

impl Lm {
    /// The layers `<p>blk.N.*` of the GGUF
    pub fn load(ops: &Ops, f: &Weights, p: &str, eps: f32, inv_freq: &[f32]) -> Result<Lm> {
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("{p}blk.{l}.{s}.weight");
            if !f.has(&n("attn_norm")) {
                break;
            }
            layers.push(Layer {
                in_norm: f.vec(ops, &n("attn_norm"))?,
                post_norm: f.vec(ops, &n("ffn_norm"))?,
                qkv: f.mat_with(ops, &[&n("attn_q"), &n("attn_k"), &n("attn_v")], &[&n("attn_q"), &n("attn_k")], HEAD)?,
                o: f.mat(ops, &[&n("attn_output")])?,
                gu: f.mat(ops, &[&n("ffn_gate"), &n("ffn_up")])?,
                down: f.mat(ops, &[&n("ffn_down")])?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}blk.* layers in the file"))) };
        let hidden = first.o.n;
        let (heads, kv, ffn) = (first.o.k / HEAD, (first.qkv.n - first.o.k) / 2 / HEAD, first.gu.n / 2);
        if inv_freq.len() != HEAD / 2 {
            return Err(Error(format!("{p}: {} RoPE frequencies, {} expected", inv_freq.len(), HEAD / 2)));
        }
        Ok(Lm { norm: f.vec(ops, &format!("{p}output_norm.weight"))?, layers, hidden, heads, kv, ffn, eps, inv_freq: DevBuf::from_f32(&ops.gpu, inv_freq)? })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv) * HEAD
    }

    /// A cache for `b` sequences of up to `t` positions, passes of up to `rows` rows (all sequences)
    pub fn cache(&self, ops: &Ops, b: usize, rows: usize, t: usize) -> Result<Cache> {
        let g = &ops.gpu;
        let kv = || -> Result<Vec<DevBuf>> { (0..self.layers.len()).map(|_| DevBuf::new(g, b * self.kv * t * HEAD * 2)).collect() };
        let (r, w) = (rows.max(2), self.qkv_width());
        let widest = self.hidden.max(self.heads * HEAD).max(self.ffn);
        Ok(Cache {
            t,
            b,
            k: kv()?,
            v: kv()?,
            x: DevBuf::f32(g, r * self.hidden)?,
            hb: DevBuf::f32(g, r * self.hidden)?,
            qkv: DevBuf::f32(g, r * w)?,
            att: DevBuf::f32(g, r * self.heads * HEAD)?,
            gu: DevBuf::f32(g, r * 2 * self.ffn)?,
            act: DevBuf::f32(g, r * self.ffn)?,
            tmp: DevBuf::f32(g, r * self.hidden)?,
            xh: DevBuf::new(g, r * widest * 2)?,
            // the split-key scratch of a one-row decode at the cache's end (a wide pass takes none)
            part: DevBuf::f32(g, ops.attn_scratch(b, 1, self.heads, HEAD, t.saturating_sub(1)).max(ops.attn_scratch(b, r / b.max(1), self.heads, HEAD, 0)).max(1))?,
            out: DevBuf::f32(g, r * self.hidden)?,
        })
    }

    /// x [r, k] . W^T added into c.x (a few rows in one pass; more through `tmp`)
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize) -> Result<()> {
        if r <= 8 {
            return ops.gemv(x, r, m, c.x.fp(), m.n, true);
        }
        m.apply(ops, nsd, x, r, &c.xh, c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions of each of the cache's sequences (c.x: rows b * n + s, from position p0) through every layer;
    /// the normed output in c.out
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t {
            return Err(Error(format!("the cache holds {} positions, not {}", c.t, p0 + n)));
        }
        let (h, w, q, b) = (self.hidden, self.qkv_width(), self.heads * HEAD, c.b);
        let rows = b * n;
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, c.hb.fp(), rows, &c.xh, c.qkv.fp())?;
            ops.head_norm_rope(c.qkv.fp(), w, rows, n, self.heads, HEAD, None, self.eps, &self.inv_freq, p0)?;
            ops.head_norm_rope(fp(&c.qkv, q), w, rows, n, self.kv, HEAD, None, self.eps, &self.inv_freq, p0)?;
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, b, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], b, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), rows)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, c.hb.fp(), rows, &c.xh, c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, rows, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), rows)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_frequencies() {
        let f = rope_freqs(1e6);
        assert_eq!((f.len(), f[0]), (64, 1.0));
        assert!((f[63] - 1e6f32.powf(-126.0 / 128.0)).abs() < 1e-9);
    }
}
