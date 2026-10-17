//! The language model: Qwen2.5-1.5B (28 layers of 1,536; 12 query / 2 key-value heads of 128 with q / k / v biases; an
//! 8,960 SwiGLU; RoPE θ 1e6; ε 1e-6). Its matrices as the file keeps them - Q8_0 codes and block scales (the q8_0
//! file; read at half the bytes of half) or half (the f16 file) - the gate / up rows interleaved for the product's
//! SwiGLU; a half cache [Hkv, T, 128] a sequence - the script's (the positive branch) and the guidance's (the negative
//! one, from an audio start alone). Its output head is the token embedding (tied), of which only the four tokens it may
//! say are read (lib.rs).
//!
//! A decode step (`decode`: one row of one sequence, or the paired step's row of each) runs on the engine's own
//! kernels, its positions read from device memory: per layer a norm, the q / k / v product, RoPE with the key / value
//! store, the attention, the output product added in, a norm, the gate / up product with SwiGLU, the down product
//! added in. A prompt (`prefill`) runs in passes of up to 256 rows on oneDNN's half GEMM.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Epi, Mat, Ops};
use crate::weights::Weights;

pub const HEAD: usize = 128;

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    /// gate and up, their rows interleaved (gate i, up i)
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
    /// a Q8_0 matrix expanded for a prompt's wide products (half, the largest's size)
    deq: Option<DevBuf>,
}

/// One sequence's cache and work buffers: up to `rows` rows a pass, `t` positions
pub struct Cache {
    pub t: usize,
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
    xh: DevBuf,
    part: DevBuf,
    /// a decode step's attention: the keys in this many pieces
    pieces: usize,
    /// a decode step's positions (device int32 [2]; `posv` the view the host writes through without waiting)
    _pos: DevBuf,
    posv: DevBuf,
    pub out: DevBuf,
}

/// The pieces a decode step's attention splits its keys into, for a cache of t positions: about 64 keys a piece when
/// full, at least 4 and at most 128 (the GPU's ~2,000 sub-groups kept busy, the merge short)
fn pieces(t: usize) -> usize {
    t.div_ceil(64).clamp(4, 128)
}

impl Lm {
    pub fn load(ops: &Ops, f: &Weights, heads: usize, kv: usize, eps: f32, theta: f64) -> Result<Lm> {
        let p = "model.language_model.";
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("{p}layers.{l}.{s}");
            if !f.has(&n("input_layernorm.weight")) {
                break;
            }
            let (q, k, v) = (n("self_attn.q_proj.weight"), n("self_attn.k_proj.weight"), n("self_attn.v_proj.weight"));
            let (qb, kb, vb) = (n("self_attn.q_proj.bias"), n("self_attn.k_proj.bias"), n("self_attn.v_proj.bias"));
            layers.push(Layer {
                in_norm: f.vec(ops, &n("input_layernorm.weight"))?,
                post_norm: f.vec(ops, &n("post_attention_layernorm.weight"))?,
                qkv: f.mat_lm(ops, &[&q, &k, &v], Some(&[&qb, &kb, &vb]), false)?,
                o: f.mat_lm(ops, &[&n("self_attn.o_proj.weight")], None, false)?,
                gu: f.mat_lm(ops, &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")], None, true)?,
                down: f.mat_lm(ops, &[&n("mlp.down_proj.weight")], None, false)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers.* in the file"))) };
        let hidden = first.o.n;
        let ffn = first.gu.n / 2;
        if first.o.k != heads * HEAD || first.qkv.n != (heads + 2 * kv) * HEAD {
            return Err(Error(format!("the attention's matrices do not fit {heads} query and {kv} key-value heads of {HEAD}")));
        }
        let widest = layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).filter(|m| m.wt != 0).map(|m| m.n * m.k * 2).max();
        let deq = widest.map(|b| DevBuf::new(&ops.gpu, b)).transpose()?;
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / HEAD as f64)) as f32).collect();
        Ok(Lm { norm: f.vec(ops, &format!("{p}norm.weight"))?, layers, hidden, heads, kv, ffn, eps, inv_freq: DevBuf::from_f32(&ops.gpu, &inv)?, deq })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Bytes a decode step reads (every layer's matrices)
    pub fn step_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.qkv.bytes() + l.o.bytes() + l.gu.bytes() + l.down.bytes()).sum()
    }

    /// Layer 0's matrices (a bench's): q / k / v, o, gate / up, down
    pub fn mats(&self) -> [&Mat; 4] {
        let l = &self.layers[0];
        [&l.qkv, &l.o, &l.gu, &l.down]
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv) * HEAD
    }

    /// The cache's bytes a position (every layer's keys and values, half)
    pub fn position_bytes(&self) -> usize {
        self.layers.len() * 2 * self.kv * HEAD * 2
    }

    /// A cache of up to `t` positions, passes of up to `rows` rows
    pub fn cache(&self, ops: &Ops, rows: usize, t: usize) -> Result<Cache> {
        let g = &ops.gpu;
        let kv = || -> Result<Vec<DevBuf>> { (0..self.layers.len()).map(|_| DevBuf::new(g, self.kv * t * HEAD * 2)).collect() };
        let (r, w) = (rows.max(2), self.qkv_width());
        let widest = self.hidden.max(self.heads * HEAD).max(self.ffn);
        let pieces = pieces(t);
        // the split-key scratch: a wide pass's (the shared attention's) or a two-row decode step's
        let part = ops.attn_scratch(1, r, self.heads, HEAD, 0).max(2 * self.heads * pieces * (HEAD + 2)).max(1);
        let pos = DevBuf::new(g, 64)?;
        Ok(Cache {
            t,
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
            xh: DevBuf::new(g, r * widest * 2)?,
            part: DevBuf::f32(g, part)?,
            pieces,
            posv: pos.view(0, 64)?,
            _pos: pos,
            out: DevBuf::f32(g, r * self.hidden)?,
        })
    }

    /// x [r, k] . W^T added into c.x (a few rows in one pass; more through `tmp`)
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize) -> Result<()> {
        if r <= 8 {
            return ops.gemv(x, r, m, c.x.fp(), m.n, true);
        }
        m.apply_with(ops, nsd, x, r, &c.xh, c.tmp.fp(), self.deq.as_ref())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// `n` new positions (c.x rows, from position p0) through every layer; the normed output in c.out (a prompt's
    /// pass, on the shared kernels and oneDNN's GEMM)
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t || n > c.rows {
            return Err(Error(format!("a pass of {n} rows at {p0}: the cache holds {} positions, {} rows a pass", c.t, c.rows)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        let deq = self.deq.as_ref();
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply_with(ops, nsd, c.hb.fp(), n, &c.xh, c.qkv.fp(), deq)?;
            ops.rope(c.qkv.fp(), w, n, n, self.heads, HEAD, &self.inv_freq, p0)?;
            ops.rope(fp(&c.qkv, q), w, n, n, self.kv, HEAD, &self.inv_freq, p0)?;
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, 1, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], 1, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), n)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply_with(ops, nsd, c.hb.fp(), n, &c.xh, c.gu.fp(), deq)?;
            ops.swiglu_pairs(c.gu.fp(), n, self.ffn, c.act.fp())?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), n)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
    }

    /// A decode step: row r of `w.x` at position `at[r].1` of sequence `at[r].0` (one or two rows: the paired step's
    /// branches, each its own cache), in w's work buffers; the rows' normed outputs in `last` [rows, hidden]
    fn decode(&self, ops: &Ops, w: &Cache, at: &[(&Cache, usize)], last: &DevBuf) -> Result<()> {
        let r = at.len();
        if r == 0 || r > 2 || w.rows < r || at.iter().any(|(c, p)| *p >= c.t) {
            return Err(Error(format!("a decode step at {:?}: the caches hold {:?} positions", at.iter().map(|a| a.1).collect::<Vec<_>>(),
                                     at.iter().map(|a| a.0.t).collect::<Vec<_>>())));
        }
        let (h, wd) = (self.hidden, self.qkv_width());
        let pos: Vec<u8> = at.iter().flat_map(|(_, p)| (*p as i32).to_le_bytes()).collect();
        w.posv.write_async(0, &pos)?;
        for (li, l) in self.layers.iter().enumerate() {
            let seqs: Vec<(&DevBuf, &DevBuf, usize)> = at.iter().map(|(c, _)| (&c.k[li], &c.v[li], c.t)).collect();
            ops.rms_norm(w.x.fp(), r, h, Some(&l.in_norm), self.eps, None, w.hb.fp())?;
            ops.gemv(w.hb.fp(), r, &l.qkv, w.qkv.fp(), wd, false)?;
            ops.rope_kv(w.qkv.fp(), wd, &w.posv, self.heads, self.kv, HEAD, &self.inv_freq, &seqs)?;
            ops.attn_dec(w.qkv.fp(), wd, &w.posv, &seqs, self.heads, self.kv, w.pieces, w.att.fp(), &w.part)?;
            ops.gemv(w.att.fp(), r, &l.o, w.x.fp(), h, true)?;
            ops.rms_norm(w.x.fp(), r, h, Some(&l.post_norm), self.eps, None, w.hb.fp())?;
            ops.gemv_epi(w.hb.fp(), r, h, &l.gu, w.act.fp(), self.ffn, Epi::Swiglu, None)?;
            ops.gemv(w.act.fp(), r, &l.down, w.x.fp(), h, true)?;
        }
        ops.rms_norm(w.x.fp(), r, h, Some(&self.norm), self.eps, None, last.fp())
    }

    /// One row through two sequences at once - `row` at position `pa` of cache `a` and at `pb` of cache `b` - so the
    /// weights are read once for both (the guidance's two branches read the same input). Their normed outputs in
    /// `last` [2, hidden]: a's, then b's. Runs in a's work buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn pair(&self, ops: &Ops, _nsd: &Nsd, a: &Cache, pa: usize, b: &Cache, pb: usize, row: &DevBuf, last: &DevBuf) -> Result<()> {
        let h = self.hidden;
        a.x.copy_within(0, row, 0, h * 4)?;
        a.x.copy_within(h * 4, row, 0, h * 4)?;
        self.decode(ops, a, &[(a, pa), (b, pb)], last)
    }

    /// Rows [L, hidden] (host) from position p0 in passes of the cache's width; the last row's normed output in `last`
    pub fn prefill(&self, ops: &Ops, nsd: &Nsd, c: &Cache, rows: &[f32], p0: usize, last: &DevBuf) -> Result<()> {
        let h = self.hidden;
        let l = rows.len() / h;
        let mut at = 0;
        while at < l {
            let n = (l - at).min(c.rows);
            c.x.write(0, &crate::bytes(&rows[at * h..(at + n) * h]))?;
            self.pass(ops, nsd, c, n, p0 + at)?;
            at += n;
            if at == l {
                last.copy_within(0, &c.out, (n - 1) * h * 4, h * 4)?;
            }
        }
        Ok(())
    }

    /// One row already on the GPU (`row`, hidden floats) at position p; its normed output in `last`
    pub fn step(&self, ops: &Ops, _nsd: &Nsd, c: &Cache, row: &DevBuf, p: usize, last: &DevBuf) -> Result<()> {
        let h = self.hidden;
        c.x.copy_within(0, row, 0, h * 4)?;
        self.decode(ops, c, &[(c, p)], &c.out)?;
        last.copy_within(0, &c.out, 0, h * 4)
    }
}
