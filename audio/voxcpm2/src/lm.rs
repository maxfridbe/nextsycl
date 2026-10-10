//! MiniCPM4's decoder (voxcpm's modules/minicpm4): RMS norms, grouped attention (head 128, 8 query heads a key head
//! here), SwiGLU, rotary positions from the long-RoPE frequency table (or none: the residual language model). Two uses:
//!
//! - the language models (the base 28 layers of 2,048, the residual 8): causal over a prompt, then a row a patch
//!   against their caches (`cache`, `pass`);
//! - the local transformers (the feature encoder's and the DiT's 12 layers of 1,024): every row sees every row of its
//!   group - B groups of S rows a call (`local`).
//!
//! use_mup is off in VoxCPM2's config: no embedding scale, plain residuals.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, nul, Mat, Ops, Shards};

const HEAD: usize = 128;

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
    pub heads: usize,
    pub kv: usize,
    ffn: usize,
    eps: f32,
    /// the rotary frequencies (inv_freq / the long-RoPE factor), none: no positions
    rope: Option<DevBuf>,
}

/// Work buffers for up to `rows` rows a pass; a language model's caches for `t` positions
pub struct Cache {
    pub t: usize,
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
    wh: Option<DevBuf>,
    part: DevBuf,
    pub out: DevBuf,
    rows: usize,
}

/// MiniCPM4's long RoPE (max positions = the original's: the short factors, no attention scale): inv_freq[i] /
/// factor[i]
pub fn long_rope(theta: f32, d: usize, factors: &[f32]) -> Vec<f32> {
    (0..d / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / d as f32) / factors.get(i).copied().unwrap_or(1.0)).collect()
}

impl Lm {
    /// The decoder under `p` (`<p>layers.N.`, `<p>norm.weight`); `rope`: its frequency table
    pub fn load(ops: &Ops, f: &Shards, p: &str, eps: f32, rope: Option<&[f32]>, int8: bool) -> Result<Lm> {
        let mut layers = Vec::new();
        for l in 0.. {
            let pre = format!("{p}layers.{l}.");
            if f.find(&format!("{pre}input_layernorm.weight")).is_err() {
                break;
            }
            let n = |s: &str| format!("{pre}{s}.weight");
            let (q, k) = (f.shape(&n("self_attn.q_proj"))?[0], f.shape(&n("self_attn.k_proj"))?[0]);
            let ffn = f.shape(&n("mlp.gate_proj"))?[0];
            layers.push(Layer {
                in_norm: f.dev_f32(ops, &n("input_layernorm"))?,
                post_norm: f.dev_f32(ops, &n("post_attention_layernorm"))?,
                qkv: f.mat(ops, &[(&n("self_attn.q_proj"), 0, q), (&n("self_attn.k_proj"), 0, k), (&n("self_attn.v_proj"), 0, k)], int8)?,
                o: f.mat1(ops, &n("self_attn.o_proj"), int8)?,
                gu: f.mat(ops, &[(&n("mlp.gate_proj"), 0, ffn), (&n("mlp.up_proj"), 0, ffn)], int8)?,
                down: f.mat1(ops, &n("mlp.down_proj"), int8)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers in the checkpoint"))) };
        let (hidden, q) = (first.o.n, first.o.k);
        let (heads, kv, ffn) = (q / HEAD, (first.qkv.n - q) / 2 / HEAD, first.gu.n / 2);
        let rope = rope.map(|r| DevBuf::from_f32(&ops.gpu, r)).transpose()?;
        Ok(Lm { norm: f.dev_f32(ops, &format!("{p}norm.weight"))?, layers, hidden, heads, kv, ffn, eps, rope })
    }

    fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv) * HEAD
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Buffers for `rows` rows a pass, and caches of `t` positions (0: none - a local transformer)
    pub fn cache(&self, ops: &Ops, rows: usize, t: usize) -> Result<Cache> {
        let g = &ops.gpu;
        let kv = || -> Result<Vec<DevBuf>> {
            if t == 0 {
                return Ok(Vec::new());
            }
            (0..self.layers.len()).map(|_| DevBuf::new(g, self.kv * t * HEAD * 2)).collect()
        };
        let (r, w) = (rows.max(2), self.qkv_width());
        let widest = self.hidden.max(self.heads * HEAD).max(2 * self.ffn);
        let int8 = self.layers.first().is_some_and(|l| l.qkv.scale.is_some());
        Ok(Cache {
            t,
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
            wh: match int8 {
                true => Some(DevBuf::new(g, self.layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.n * m.k).max().unwrap_or(0) * 2)?),
                false => None,
            },
            part: DevBuf::f32(g, if t > 0 { ops.attn_scratch(1, r, self.heads, HEAD, t).max(1) } else { 1 })?,
            out: DevBuf::f32(g, r * self.hidden)?,
            rows: r,
        })
    }

    /// x [r, k] . W^T added into c.x
    fn residual(&self, ops: &Ops, nsd: &Nsd, c: &Cache, m: &Mat, x: *const f32, r: usize) -> Result<()> {
        if r <= 8 {
            return ops.gemv(x, r, m.k, m, nul(), c.x.fp(), m.n, true);
        }
        m.apply(ops, nsd, x, r, Some(&c.xh), c.wh.as_ref(), nul(), c.tmp.fp())?;
        ops.add(c.x.fp(), c.tmp.fp(), r * m.n)
    }

    /// The layers over c.x: causal against the caches from position p0 (`group` 0), or within groups of `group` rows;
    /// the normed output in c.out
    fn run(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize, group: usize) -> Result<()> {
        if n > c.rows {
            return Err(Error(format!("{n} rows; the buffers take {}", c.rows)));
        }
        if group == 0 && p0 + n > c.t {
            return Err(Error(format!("the cache holds {} positions, not {}", c.t, p0 + n)));
        }
        let (h, w, q) = (self.hidden, self.qkv_width(), self.heads * HEAD);
        let s = if group == 0 { n } else { group };
        for (li, l) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.in_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.qkv.apply(ops, nsd, c.hb.fp(), n, Some(&c.xh), c.wh.as_ref(), nul(), c.qkv.fp())?;
            if let Some(r) = &self.rope {
                ops.rope(c.qkv.fp(), w, n, s, self.heads, HEAD, r, if group == 0 { p0 } else { 0 })?;
                ops.rope(fp(&c.qkv, q), w, n, s, self.kv, HEAD, r, if group == 0 { p0 } else { 0 })?;
            }
            if group == 0 {
                ops.kv_store(fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), w, 1, n, self.kv, HEAD, c.t, p0, &c.k[li], &c.v[li])?;
                ops.attn(c.qkv.fp(), w, &c.k[li], &c.v[li], 1, n, self.heads, self.kv, HEAD, c.t, p0, c.att.fp(), c.part.fp())?;
            } else {
                ops.local_attn(c.qkv.fp(), fp(&c.qkv, q), fp(&c.qkv, q + self.kv * HEAD), n.checked_div(group).unwrap_or(0), group, self.heads, self.kv, HEAD, w, w, c.att.fp())?;
            }
            self.residual(ops, nsd, c, &l.o, c.att.fp(), n)?;
            nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, l.post_norm.ptr(), self.eps, none(), none(), none(), c.hb.ptr(), Dt::F32)?;
            l.gu.apply(ops, nsd, c.hb.fp(), n, Some(&c.xh), c.wh.as_ref(), nul(), c.gu.fp())?;
            nsd.swiglu(c.gu.ptr(), Dt::F32, n, self.ffn, c.act.ptr(), Dt::F32)?;
            self.residual(ops, nsd, c, &l.down, c.act.fp(), n)?;
        }
        nsd.rms_norm_mod(c.x.ptr(), Dt::F32, n, h, self.norm.ptr(), self.eps, none(), none(), none(), c.out.ptr(), Dt::F32)
    }

    /// `n` new positions (c.x) causally from position p0, cached
    pub fn pass(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, p0: usize) -> Result<()> {
        self.run(ops, nsd, c, n, p0, 0)
    }

    /// `n` rows in groups of `group`, each group on its own
    pub fn local(&self, ops: &Ops, nsd: &Nsd, c: &Cache, n: usize, group: usize) -> Result<()> {
        if group == 0 || !n.is_multiple_of(group) {
            return Err(Error(format!("{n} rows in groups of {group}")));
        }
        self.run(ops, nsd, c, n, 0, group)
    }
}
