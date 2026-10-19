//! The two causal decoders: the backbone (Qwen3: 28 layers of 2,048, 16 query / 8 key-value heads of 128, a 6,144
//! SwiGLU, q / k RMS norms, RoPE θ 1e6, ε 1e-6) and the depth decoder (Llama: 12 layers of 1,024, 8 / 2 heads of 128,
//! an 8,192 SwiGLU, no q / k norms, llama3-scaled RoPE θ 5e5 over 16 positions, ε 1e-5). Weights in half; a half
//! cache [B, Hkv, T, 128] for B sequences that move together (the depth decoder's guided pair; the backbone runs
//! one).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

pub const HEAD: usize = 128;

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    /// Qwen3's q / k norms (the backbone's); none in the depth decoder
    qk_norm: Option<(DevBuf, DevBuf)>,
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

/// llama3's RoPE scaling of the frequencies (transformers' _compute_llama3_parameters)
pub fn llama3_inv_freq(theta: f64, dim: usize, factor: f64, low: f64, high: f64, old: f64) -> Vec<f32> {
    let (low_wl, high_wl) = (old / low, old / high);
    (0..dim / 2).map(|i| {
        let f = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
        let wl = 2.0 * std::f64::consts::PI / f;
        let scaled = if wl > low_wl { f / factor } else { f };
        if wl >= high_wl && wl <= low_wl {
            let s = (old / wl - low) / (high - low);
            ((1.0 - s) * scaled / factor + s * scaled) as f32
        } else {
            scaled as f32
        }
    }).collect()
}

impl Lm {
    /// The layers `<p>blk.N.*` of the GGUF; `qk_norm`: Qwen3's q / k norms are read
    pub fn load(ops: &Ops, f: &Weights, p: &str, qk_norm: bool, eps: f32, inv_freq: &[f32]) -> Result<Lm> {
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("{p}blk.{l}.{s}.weight");
            if !f.has(&n("attn_norm")) {
                break;
            }
            layers.push(Layer {
                in_norm: f.vec(ops, &n("attn_norm"))?,
                post_norm: f.vec(ops, &n("ffn_norm"))?,
                qk_norm: if qk_norm { Some((f.vec(ops, &n("attn_q_norm"))?, f.vec(ops, &n("attn_k_norm"))?)) } else { None },
                qkv: f.mat(ops, &[&n("attn_q"), &n("attn_k"), &n("attn_v")])?,
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

    /// x [r, k] . W^T added into c.x (a few rows in one pass - up to 16 of several sequences in passes of 8; more
    /// through `tmp`)
    #[allow(clippy::too_many_arguments)]
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize, b: usize) -> Result<()> {
        if r <= 8 || (b > 1 && r <= 16) {
            for at in (0..r).step_by(8) {
                ops.gemv(x.wrapping_add(at * m.k), (r - at).min(8), m, fp(&c.x, at * m.n), m.n, true)?;
            }
            return Ok(());
        }
        m.apply(ops, nsd, x, r, &c.xh, c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions of each of the cache's sequences (c.x: rows b * n + s, from position p0) through every layer;
    /// the normed output in c.out
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        self.pass_b(ops, nsd, c, c.b, n, p0)
    }

    /// `pass` over the cache's first `b` sequences (a batch of fewer than it holds: its rows the first b * n)
    pub fn pass_b(&self, ops: &Ops, nsd: &Nsd, c: &Cache, b: usize, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t || b > c.b {
            return Err(Error(format!("the cache holds {} positions of {} sequences, not {} of {b}", c.t, c.b, p0 + n)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        let rows = b * n;
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            apply_rows(ops, nsd, &l.qkv, c.hb.fp(), rows, b, &c.xh, c.qkv.fp())?;
            match &l.qk_norm {
                Some((qn, kn)) => {
                    ops.qk_norm_rope(c.qkv.fp(), w, rows, n, self.heads, qn, self.eps, &self.inv_freq, p0)?;
                    ops.qk_norm_rope(fp(&c.qkv, q), w, rows, n, self.kv, kn, self.eps, &self.inv_freq, p0)?;
                }
                None => {
                    ops.head_norm_rope(c.qkv.fp(), w, rows, n, self.heads, HEAD, None, self.eps, &self.inv_freq, p0)?;
                    ops.head_norm_rope(fp(&c.qkv, q), w, rows, n, self.kv, HEAD, None, self.eps, &self.inv_freq, p0)?;
                }
            }
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, b, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], b, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), rows, b)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            apply_rows(ops, nsd, &l.gu, c.hb.fp(), rows, b, &c.xh, c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, rows, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), rows, b)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
    }
}

/// `Mat::apply`, with the rows of several sequences (a batch's depth decoder: two rows a request) up to 16 by the
/// engine's own product in passes of 8 - a row's result the same as alone, as oneDNN's GEMM would not give
#[allow(clippy::too_many_arguments)]
fn apply_rows(ops: &Ops, nsd: &Nsd, m: &Mat, x: *const f32, rows: usize, b: usize, xh: &DevBuf, out: *mut f32) -> Result<()> {
    if b > 1 && rows > 8 && rows <= 16 && m.k.is_multiple_of(256) {
        for at in (0..rows).step_by(8) {
            ops.gemv(x.wrapping_add(at * m.k), (rows - at).min(8), m, out.wrapping_add(at * m.n), m.n, false)?;
        }
        return Ok(());
    }
    m.apply(ops, nsd, x, rows, xh, out)
}

/// The work buffers of a batched decode step: one row a sequence, up to `rows`
pub struct Work {
    rows: usize,
    pub x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    pub out: DevBuf,
}

impl Lm {
    /// Work buffers for steps of up to `rows` sequences (8: the product's rows)
    pub fn work(&self, ops: &Ops, rows: usize) -> Result<Work> {
        let g = &ops.gpu;
        Ok(Work {
            rows,
            x: DevBuf::f32(g, rows * self.hidden)?,
            hb: DevBuf::f32(g, rows * self.hidden)?,
            qkv: DevBuf::f32(g, rows * self.qkv_width())?,
            att: DevBuf::f32(g, rows * self.heads * HEAD)?,
            gu: DevBuf::f32(g, rows * 2 * self.ffn)?,
            act: DevBuf::f32(g, rows * self.ffn)?,
            out: DevBuf::f32(g, rows * self.hidden)?,
        })
    }

    /// One new position of each of several sequences, each its own cache at its own position (`seqs`: the cache and
    /// the position written; w.x row i the i-th's input): the products over all the rows at once (the weights read
    /// once), the attention a sequence at a time; the normed output in w.out. A row comes out as `pass` makes it
    /// alone (the product's rows are independent)
    pub fn step(&self, ops: &Ops, nsd: &Nsd, w: &Work, seqs: &[(&Cache, usize)]) -> Result<()> {
        let rows = seqs.len();
        if rows == 0 || rows > w.rows || rows > 8 {
            return Err(Error(format!("a step of {rows} sequences ({} at most)", w.rows.min(8))));
        }
        for (c, p) in seqs {
            if p + 1 > c.t || c.b != 1 {
                return Err(Error(format!("a sequence's cache holds {} positions of {}, not position {p} of one", c.t, c.b)));
            }
        }
        let (h, qw, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, rows, h, l.in_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
            ops.gemv(w.hb.fp(), rows, &l.qkv, w.qkv.fp(), l.qkv.n, false)?;
            for (i, (c, p)) in seqs.iter().enumerate() {
                let row = fp(&w.qkv, i * qw);
                match &l.qk_norm {
                    Some((qn, kn)) => {
                        ops.qk_norm_rope(row, qw, 1, 1, self.heads, qn, self.eps, &self.inv_freq, *p)?;
                        ops.qk_norm_rope(row.wrapping_add(q), qw, 1, 1, self.kv, kn, self.eps, &self.inv_freq, *p)?;
                    }
                    None => {
                        ops.head_norm_rope(row, qw, 1, 1, self.heads, HEAD, None, self.eps, &self.inv_freq, *p)?;
                        ops.head_norm_rope(row.wrapping_add(q), qw, 1, 1, self.kv, HEAD, None, self.eps, &self.inv_freq, *p)?;
                    }
                }
                ops.kv_store(row.wrapping_add(q), row.wrapping_add(q + self.kv * HEAD), qw, 1, 1, self.kv, HEAD, c.t, *p, &c.k[li], &c.v[li])?;
                ops.attn(row, qw, &c.k[li], &c.v[li], 1, 1, self.heads, self.kv, HEAD, c.t, *p, fp(&w.att, i * q), c.part.fp())?;
            }
            ops.gemv(w.att.fp(), rows, &l.o, w.x.fp(), l.o.n, true)?;
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, rows, h, l.post_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
            ops.gemv(w.hb.fp(), rows, &l.gu, w.gu.fp(), l.gu.n, false)?;
            nsd.swiglu(w.gu.ptr(), Dt::F32, rows, self.ffn, w.act.ptr(), Dt::F32)?;
            ops.gemv(w.act.fp(), rows, &l.down, w.x.fp(), l.down.n, true)?;
        }
        nsd.rms_norm_mod(w.x.ptr(), Dt::F32, rows, h, self.norm.ptr(), self.eps, none(), none(), none(), w.out.ptr(), Dt::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_scaling_keeps_the_fast_and_divides_the_slow() {
        // the depth decoder's: θ 5e5, factor 32, low 1/512, high 1/128, 16 original positions
        let f = llama3_inv_freq(500_000.0, 128, 32.0, 0.001953125, 0.0078125, 16.0);
        assert_eq!(f.len(), 64);
        // the first pair's wavelength (2π) is below 2,048: unchanged
        assert_eq!(f[0], 1.0);
        // the last's (~ 2π 5e5^(126/128)) is above 8,192: divided by the factor
        let plain = 1.0 / 500_000f64.powf(126.0 / 128.0);
        assert!(((f[63] as f64) - plain / 32.0).abs() < 1e-12);
    }
}
