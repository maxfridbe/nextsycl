//! The language model: Qwen2.5-1.5B (28 layers of 1,536; 12 query / 2 key-value heads of 128 with q / k / v biases; an
//! 8,960 SwiGLU; RoPE θ 1e6; ε 1e-6). Weights in half (the q8_0 file's blocks expanded); a half cache [Hkv, T, 128]
//! a sequence - the script's (the positive branch) and the guidance's (the negative one, from an audio start alone).
//! Its output head is the token embedding (tied), of which only the four tokens it may say are read (lib.rs).

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
    pub out: DevBuf,
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
                qkv: f.mat(ops, &[&q, &k, &v], Some(&[&qb, &kb, &vb]), false)?,
                o: f.mat(ops, &[&n("self_attn.o_proj.weight")], None, false)?,
                gu: f.mat(ops, &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")], None, false)?,
                down: f.mat(ops, &[&n("mlp.down_proj.weight")], None, false)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers.* in the file"))) };
        let hidden = first.o.n;
        let ffn = first.gu.n / 2;
        if first.o.k != heads * HEAD || first.qkv.n != (heads + 2 * kv) * HEAD {
            return Err(Error(format!("the attention's matrices do not fit {heads} query and {kv} key-value heads of {HEAD}")));
        }
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / HEAD as f64)) as f32).collect();
        Ok(Lm { norm: f.vec(ops, &format!("{p}norm.weight"))?, layers, hidden, heads, kv, ffn, eps, inv_freq: DevBuf::from_f32(&ops.gpu, &inv)? })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
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
            // the split-key scratch of a one-row decode at the cache's end (a wide pass takes none)
            part: DevBuf::f32(g, ops.attn_scratch(1, 1, self.heads, HEAD, t.saturating_sub(1)).max(ops.attn_scratch(1, r, self.heads, HEAD, 0)).max(1))?,
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

    /// `n` new positions (c.x rows, from position p0) through every layer; the normed output in c.out
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        if p0 + n > c.t || n > c.rows {
            return Err(Error(format!("a pass of {n} rows at {p0}: the cache holds {} positions, {} rows a pass", c.t, c.rows)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, c.hb.fp(), n, &c.xh, c.qkv.fp())?;
            ops.rope(c.qkv.fp(), w, n, n, self.heads, HEAD, &self.inv_freq, p0)?;
            ops.rope(fp(&c.qkv, q), w, n, n, self.kv, HEAD, &self.inv_freq, p0)?;
            ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, 1, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
            ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], 1, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            self.residual(ops, nsd, c, &l.o, c.att.fp(), n)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, c.hb.fp(), n, &c.xh, c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, n, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), n)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
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
    pub fn step(&self, ops: &Ops, nsd: &Nsd, c: &Cache, row: &DevBuf, p: usize, last: &DevBuf) -> Result<()> {
        let h = self.hidden;
        c.x.copy_within(0, row, 0, h * 4)?;
        self.pass(ops, nsd, c, 1, p)?;
        last.copy_within(0, &c.out, 0, h * 4)
    }
}
