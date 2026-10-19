//! The Dual-AR's two causal transformers (fish-speech's llama.py TransformerBlock, both of 2,560 with 32 query and 8
//! key-value heads of 128, a 9,728 SwiGLU, ε 1e-6, RoPE base 1e6 on interleaved pairs): the slow model (36 layers,
//! Qwen3's q / k RMS norms) over the prompt and the frames, and the fast model (4 layers, no q / k norms) over a frame's
//! codebooks. Weights as the file stores them (Q8_0 or half); a half cache [B, 8, T, 128].

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

pub const HEAD: usize = 128;

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
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
    /// the widest matrix's float32 size (the staging a wide pass expands a matrix into)
    pub widest: usize,
}

/// One transformer's cache and work buffers: up to `rows` rows a pass, `b` sequences, `t` positions
pub struct Cache {
    pub t: usize,
    b: usize,
    rows: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    /// a matrix in float32 for a wide pass (none when passes are of a few rows)
    wf: Option<DevBuf>,
    part: DevBuf,
    pub out: DevBuf,
}

impl Lm {
    /// The layers `<p>N.*` of the GGUF (`layers.` or `fast_layers.`), the final norm `norm`; `qk_norm`: Qwen3's q / k
    /// norms are read
    pub fn load(ops: &Ops, f: &Weights, p: &str, norm: &str, qk_norm: bool, eps: f32) -> Result<Lm> {
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("{p}{l}.{s}.weight");
            if !f.has(&n("attention_norm")) {
                break;
            }
            layers.push(Layer {
                in_norm: f.dev(ops, &n("attention_norm"))?,
                post_norm: f.dev(ops, &n("ffn_norm"))?,
                qk_norm: if qk_norm { Some((f.dev(ops, &n("attention.q_norm"))?, f.dev(ops, &n("attention.k_norm"))?)) } else { None },
                qkv: f.mat(ops, &[&n("attention.wqkv")])?,
                o: f.mat(ops, &[&n("attention.wo")])?,
                gu: f.mat(ops, &[&n("feed_forward.w1"), &n("feed_forward.w3")])?,
                down: f.mat(ops, &[&n("feed_forward.w2")])?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}N layers in the file"))) };
        let hidden = first.o.n;
        let (heads, kv, ffn) = (first.o.k / HEAD, (first.qkv.n - first.o.k) / 2 / HEAD, first.gu.n / 2);
        if heads == 0 || kv == 0 || heads % kv != 0 || first.qkv.k != hidden {
            return Err(Error(format!("{p}: {heads} query and {kv} key-value heads of {HEAD} over {hidden}: not a shape this engine runs")));
        }
        let widest = layers.iter().flat_map(|l| [l.qkv.f32_bytes(), l.o.f32_bytes(), l.gu.f32_bytes(), l.down.f32_bytes()]).max().unwrap_or(0);
        Ok(Lm { norm: f.dev(ops, &format!("{norm}.weight"))?, layers, hidden, heads, kv, ffn, eps, widest })
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
        Ok(Cache {
            t,
            b,
            rows: r,
            k: kv()?,
            v: kv()?,
            x: DevBuf::f32(g, r * self.hidden)?,
            hb: DevBuf::f32(g, r * self.hidden)?,
            qkv: DevBuf::f32(g, r * w)?,
            att: DevBuf::f32(g, r * self.heads * HEAD)?,
            gu: DevBuf::f32(g, r * 2 * self.ffn)?,
            act: DevBuf::f32(g, r * self.ffn)?,
            tmp: DevBuf::f32(g, r * self.hidden)?,
            wf: if r > 8 { Some(DevBuf::new(g, self.widest)?) } else { None },
            // the split-key scratch of a one-row decode at the cache's end, or of a wide pass
            part: DevBuf::f32(g, ops.attn_scratch(b, 1, self.heads, HEAD, t.saturating_sub(1)).max(ops.attn_scratch(b, r / b.max(1), self.heads, HEAD, 0)).max(1))?,
            out: DevBuf::f32(g, r * self.hidden)?,
        })
    }

    /// x [r, k] . W^T added into c.x (a few rows in one pass - up to 16 of several sequences in passes of 8; more
    /// through `tmp`)
    #[allow(clippy::too_many_arguments)]
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize, b: usize) -> Result<()> {
        if r <= 8 || (b > 1 && r <= 16 && m.k.is_multiple_of(256)) {
            for at in (0..r).step_by(8) {
                m.gemv(ops, x.wrapping_add(at * m.k), (r - at).min(8), fp(&c.x, at * m.n), m.n, true)?;
            }
            return Ok(());
        }
        m.apply(ops, nsd, x, r, c.wf.as_ref(), c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions of each of the cache's sequences (c.x: rows b * n + s, from position p0) through every layer;
    /// the normed output in c.out. `cs`: the rotary table (cos, sin) of positions [0, p0 + n)
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize, cs: &DevBuf) -> Result<()> {
        self.pass_b(ops, nsd, c, c.b, n, p0, cs)
    }

    /// `pass` over the cache's first `b` sequences (a batch of fewer than it holds: its rows the first b * n)
    #[allow(clippy::too_many_arguments)]
    pub fn pass_b(&self, ops: &Ops, nsd: &Nsd, c: &Cache, b: usize, n: usize, p0: usize, cs: &DevBuf) -> Result<()> {
        if p0 + n > c.t || b > c.b {
            return Err(Error(format!("the cache holds {} positions of {} sequences, not {} of {b}", c.t, c.b, p0 + n)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        let rows = b * n;
        if rows > c.rows {
            return Err(Error(format!("a pass of {rows} rows; the cache's buffers take {}", c.rows)));
        }
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            apply_rows(ops, nsd, &l.qkv, c.hb.fp(), rows, b, c.wf.as_ref(), c.qkv.fp())?;
            let (qn, kn) = match &l.qk_norm {
                Some((qn, kn)) => (Some(qn), Some(kn)),
                None => (None, None),
            };
            ops.norm_rope(c.qkv.fp(), w, rows, n, self.heads, HEAD, qn, self.eps, cs, p0)?;
            ops.norm_rope(fp(&c.qkv, q), w, rows, n, self.kv, HEAD, kn, self.eps, cs, p0)?;
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, b, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], b, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), rows, b)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            apply_rows(ops, nsd, &l.gu, c.hb.fp(), rows, b, c.wf.as_ref(), c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, rows, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), rows, b)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, rows, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
    }
}

/// `Mat::apply`, with the rows of several sequences (a batch's fast model: two rows a request) up to 16 by the
/// decode's product in passes of 8 - a row's result the same as alone
#[allow(clippy::too_many_arguments)]
fn apply_rows(ops: &Ops, nsd: &Nsd, m: &Mat, x: *const f32, rows: usize, b: usize, wf: Option<&DevBuf>, out: *mut f32) -> Result<()> {
    if b > 1 && rows > 8 && rows <= 16 && m.k.is_multiple_of(256) {
        for at in (0..rows).step_by(8) {
            m.gemv(ops, x.wrapping_add(at * m.k), (rows - at).min(8), out.wrapping_add(at * m.n), m.n, false)?;
        }
        return Ok(());
    }
    m.apply(ops, nsd, x, rows, wf, out)
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

    /// One new position of each of several sequences, each its own cache (of one sequence) at its own position
    /// (`seqs`: the cache and the position written; w.x row i the i-th's input): the products over all the rows at
    /// once, the attention a sequence at a time; the normed output in w.out - each row as `pass` makes it alone
    pub fn step(&self, ops: &Ops, nsd: &Nsd, w: &Work, seqs: &[(&Cache, usize)], cs: &DevBuf) -> Result<()> {
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
            l.qkv.gemv(ops, w.hb.fp(), rows, w.qkv.fp(), l.qkv.n, false)?;
            let (qn, kn) = match &l.qk_norm {
                Some((qn, kn)) => (Some(qn), Some(kn)),
                None => (None, None),
            };
            for (i, (c, p)) in seqs.iter().enumerate() {
                let row = fp(&w.qkv, i * qw);
                ops.norm_rope(row, qw, 1, 1, self.heads, HEAD, qn, self.eps, cs, *p)?;
                ops.norm_rope(row.wrapping_add(q), qw, 1, 1, self.kv, HEAD, kn, self.eps, cs, *p)?;
                ops.kv_store(row.wrapping_add(q), row.wrapping_add(q + self.kv * HEAD), qw, 1, 1, self.kv, HEAD, c.t, *p, &c.k[li], &c.v[li])?;
                ops.attn(row, qw, &c.k[li], &c.v[li], 1, 1, self.heads, self.kv, HEAD, c.t, *p, fp(&w.att, i * q), c.part.fp())?;
            }
            l.o.gemv(ops, w.att.fp(), rows, w.x.fp(), l.o.n, true)?;
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, rows, h, l.post_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
            l.gu.gemv(ops, w.hb.fp(), rows, w.gu.fp(), l.gu.n, false)?;
            nsd.swiglu(w.gu.ptr(), Dt::F32, rows, self.ffn, w.act.ptr(), Dt::F32)?;
            l.down.gemv(ops, w.act.fp(), rows, w.x.fp(), l.down.n, true)?;
        }
        nsd.rms_norm_mod(w.x.ptr(), Dt::F32, rows, h, self.norm.ptr(), self.eps, none(), none(), none(), w.out.ptr(), Dt::F32)
    }
}
