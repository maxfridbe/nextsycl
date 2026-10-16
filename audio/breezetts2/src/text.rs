//! The text encoder (the checkpoint's own models/t5gemma2_compat.py: T5Gemma2's encoder, 26 layers of 1,152) and its
//! projection to the backbone's width. Each text segment of a prompt is its own sequence.
//!
//! ```text
//!   ids -> embedding x sqrt(1,152) (the end-of-image id: its own row)
//!   -> 26 x [ RMSNorm -> q (4 heads of 256), k, v (1 head); q / k RMSNorm per head; RoPE -> bidirectional attention
//!             (sliding layers: keys within (i - 256, i + 257), θ 1e4; every sixth layer the whole text, θ 1e6 / 8)
//!             -> o -> RMSNorm -> + ;  RMSNorm -> gelu_tanh(gate) * up -> down -> RMSNorm -> + ]
//!   -> RMSNorm -> projection 1,152 -> 2,048
//! ```
//!
//! The norms are Gemma's (x / rms * (1 + w)); the file holds them with the one already added (its
//! `breeze.te.norm_weights_pre_offset`), so here they are plain RMS norms. Scores are scaled by
//! query_pre_attn_scalar^-1/2 (1/16).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

struct Layer {
    pre_attn: DevBuf,
    post_attn: DevBuf,
    pre_ffn: DevBuf,
    post_ffn: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
    /// a full-attention layer (else a sliding one)
    full: bool,
}

pub struct TextEncoder {
    layers: Vec<Layer>,
    norm: DevBuf,
    proj: Mat,
    pub hidden: usize,
    heads: usize,
    kv: usize,
    head: usize,
    ffn: usize,
    eps: f32,
    embed_scale: f32,
    scale: f32,
    /// the symmetric window: keys within (i - left, i + right)
    left: usize,
    right: usize,
    inv_sliding: DevBuf,
    inv_full: DevBuf,
    vocab: usize,
    eoi: Option<(u32, Vec<f32>)>,
}

/// Default RoPE frequencies (θ, head size), divided by a linear scaling's factor
fn inv_freq(theta: f64, d: usize, factor: f64) -> Vec<f32> {
    (0..d / 2).map(|i| (1.0 / theta.powf((2 * i) as f64 / d as f64) / factor) as f32).collect()
}

impl TextEncoder {
    pub fn load(ops: &Ops, f: &Weights) -> Result<TextEncoder> {
        if !f.flag("te.norm_weights_pre_offset") {
            return Err(Error("the text encoder's norms are not stored with Gemma's one added (breeze.te.norm_weights_pre_offset): a conversion this engine does not read".into()));
        }
        let types: Vec<bool> = f.g.meta("breeze.te.layer_types").and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u64() == Some(1)).collect()).ok_or_else(|| Error("no breeze.te.layer_types".into()))?;
        let mut layers = Vec::new();
        for (l, full) in types.iter().enumerate() {
            let n = |s: &str| format!("te.blk.{l}.{s}.weight");
            layers.push(Layer {
                pre_attn: f.vec(ops, &n("attn_pre_norm"))?,
                post_attn: f.vec(ops, &n("attn_post_norm"))?,
                pre_ffn: f.vec(ops, &n("ffn_pre_norm"))?,
                post_ffn: f.vec(ops, &n("ffn_post_norm"))?,
                q_norm: f.vec(ops, &n("attn_q_norm"))?,
                k_norm: f.vec(ops, &n("attn_k_norm"))?,
                qkv: f.mat(ops, &[&n("attn_q"), &n("attn_k"), &n("attn_v")])?,
                o: f.mat(ops, &[&n("attn_output")])?,
                gu: f.mat(ops, &[&n("ffn_gate"), &n("ffn_up")])?,
                down: f.mat(ops, &[&n("ffn_down")])?,
                full: *full,
            });
        }
        let num = |k: &str| f.num(k);
        let (heads, kv, head) = (num("te.n_heads")? as usize, num("te.n_kv_heads")? as usize, num("te.head_dim")? as usize);
        let first = layers.first().ok_or_else(|| Error("no text encoder layers".into()))?;
        if first.qkv.n != (heads + 2 * kv) * head {
            return Err(Error(format!("the text encoder's q / k / v: {} rows, {} expected", first.qkv.n, (heads + 2 * kv) * head)));
        }
        let hidden = first.o.n;
        let ffn = first.gu.n / 2;
        let rope = |kind: &str| -> Result<Vec<f32>> {
            let theta = num(&format!("te.rope_theta_{kind}"))?;
            let factor = match f.g.meta(&format!("breeze.te.rope_type_{kind}")).and_then(|v| v.as_str()) {
                Some("linear") => num(&format!("te.rope_factor_{kind}"))?,
                Some("default") | None => 1.0,
                Some(t) => return Err(Error(format!("the text encoder's {kind} RoPE: {t}, not default or linear"))),
            };
            Ok(inv_freq(theta, head, factor))
        };
        let eoi = match (f.g.meta("breeze.te.eoi_token_index").and_then(|v| v.as_u64()), f.has("te.eoi_embd")) {
            (Some(i), true) => Some((i as u32, f.rows("te.eoi_embd", 0, hidden)?)),
            _ => None,
        };
        let tok = f.tensor("te.token_embd.weight")?;
        Ok(TextEncoder {
            norm: f.vec(ops, "te.output_norm.weight")?,
            proj: f.mat(ops, &["te_proj.weight"])?,
            hidden,
            heads,
            kv,
            head,
            ffn,
            eps: num("te.rms_norm_eps")? as f32,
            embed_scale: num("te.embed_scale")? as f32,
            scale: num("te.attn_scale")? as f32,
            left: num("te.sliding_window_left")? as usize,
            right: num("te.sliding_window_right")? as usize,
            inv_sliding: DevBuf::from_f32(&ops.gpu, &rope("sliding")?)?,
            inv_full: DevBuf::from_f32(&ops.gpu, &rope("full")?)?,
            vocab: tok.shape[0] as usize,
            eoi,
            layers,
        })
    }

    pub fn out_width(&self) -> usize {
        self.proj.n
    }

    /// The embedding's rows of `ids` (read from the file), scaled; [n, hidden] on the host
    pub fn embed(&self, f: &Weights, ids: &[u32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            if id as usize >= self.vocab {
                return Err(Error(format!("token {id} is past the text encoder's {} rows", self.vocab)));
            }
            match &self.eoi {
                Some((e, row)) if *e == id => out.extend_from_slice(row),
                _ => out.extend(f.rows("te.token_embd.weight", id as usize, id as usize + 1)?.into_iter().map(|v| v * self.embed_scale)),
            }
        }
        Ok(out)
    }

    /// One segment's ids to its rows for the backbone [n, 2,048] (host); `keep`: the encoder's own output [n, 1,152]
    /// too (a check's)
    pub fn encode(&self, ops: &Ops, nsd: &Nsd, f: &Weights, ids: &[u32], keep: Option<&mut Vec<f32>>) -> Result<Vec<f32>> {
        let g = &ops.gpu;
        let n = ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let (h, hd) = (self.hidden, self.head);
        let qw = (self.heads + 2 * self.kv) * hd;
        let widest = qw.max(2 * self.ffn).max(self.proj.n);
        let x = DevBuf::from_f32(g, &self.embed(f, ids)?)?;
        let hb = DevBuf::f32(g, n * h)?;
        let t1 = DevBuf::f32(g, n * widest)?;
        let t2 = DevBuf::f32(g, n * widest)?;
        let xh = DevBuf::new(g, n * widest * 2)?;
        let norm = |src: *const f32, w: &DevBuf, dst: *mut f32| nsd.rms_norm_mod(src.cast(), Dt::F32, n, h, w.ptr(), self.eps, none(), none(), none(), dst.cast(), Dt::F32);
        for l in &self.layers {
            norm(x.fp(), &l.pre_attn, hb.fp())?;
            l.qkv.apply(ops, nsd, hb.fp(), n, &xh, t1.fp())?;
            let inv = if l.full { &self.inv_full } else { &self.inv_sliding };
            ops.head_norm_rope(t1.fp(), qw, n, n, self.heads, hd, Some(&l.q_norm), self.eps, inv, 0)?;
            ops.head_norm_rope(fp(&t1, self.heads * hd), qw, n, n, self.kv, hd, Some(&l.k_norm), self.eps, inv, 0)?;
            let (left, right) = if l.full { (n, n) } else { (self.left, self.right) };
            ops.attn_window(t1.fp(), fp(&t1, self.heads * hd), fp(&t1, (self.heads + self.kv) * hd), n, self.heads, self.kv, hd, qw, qw, left, right, self.scale,
                            t2.fp())?;
            l.o.apply(ops, nsd, t2.fp(), n, &xh, t1.fp())?;
            norm(t1.fp(), &l.post_attn, hb.fp())?;
            ops.add(x.fp(), hb.fp(), n * h)?;
            norm(x.fp(), &l.pre_ffn, hb.fp())?;
            l.gu.apply(ops, nsd, hb.fp(), n, &xh, t1.fp())?;
            ops.gelu_mul(t1.fp(), n, self.ffn, t2.fp())?;
            l.down.apply(ops, nsd, t2.fp(), n, &xh, t1.fp())?;
            norm(t1.fp(), &l.post_ffn, hb.fp())?;
            ops.add(x.fp(), hb.fp(), n * h)?;
        }
        norm(x.fp(), &self.norm, hb.fp())?;
        if let Some(k) = keep {
            g.sync()?;
            *k = hb.to_f32()?[..n * h].to_vec();
        }
        self.proj.apply(ops, nsd, hb.fp(), n, &xh, t1.fp())?;
        g.sync()?;
        Ok(t1.to_f32()?[..n * self.proj.n].to_vec())
    }
}
