//! The backbone: Qwen3-0.6B (28 layers of 1,024, 16 query heads over 8 key / value heads of 128, an MLP of 3,072) run
//! as a bidirectional encoder - every row sees every row of its own sequence, no causal mask, no cache: a step reads
//! the whole prompt again, since the codes unmasked at a step change what every row attends to. A step is one pass over
//! two sequences side by side: the conditional one (style and text tokens, a recording's codes, the target) and the
//! unconditional one (the target alone), each at positions from 0; the products take all the rows together, the
//! attention each sequence on its own.
//!
//! The matrices are half on the card (expanded once from the file's type), the residual stream float32, the
//! activations into each product half (upstream runs the model in half on CUDA).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{at, fp, Ops};
use crate::weights::Weights;

pub const HEAD: usize = 128;

struct Layer {
    qkv: DevBuf,
    o: DevBuf,
    gu: DevBuf,
    down: DevBuf,
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

pub struct Lm {
    layers: Vec<Layer>,
    norm: DevBuf,
    /// the audio heads, half [codebooks * audio vocabulary, hidden]
    heads: DevBuf,
    /// the audio embeddings, float32 [codebooks * audio vocabulary, hidden]
    pub audio_emb: DevBuf,
    pub hidden: usize,
    pub hq: usize,
    pub hkv: usize,
    pub inter: usize,
    pub eps: f32,
    pub theta: f64,
    pub heads_n: usize,
}

/// A pass's work buffers for R rows
pub struct Work {
    pub rows: usize,
    pub x: DevBuf,
    xn: DevBuf,
    qkv: DevBuf,
    qkv3: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    pub cs: DevBuf,
    pub logits: DevBuf,
}

/// What a check keeps of a pass: the rows after the first layer and after the final norm
#[derive(Default)]
pub struct Taps {
    pub layer0: Vec<f32>,
    pub hidden: Vec<f32>,
}

impl Lm {
    pub fn load(ops: &Ops, w: &Weights) -> Result<Lm> {
        let a = "omnivoice-lm";
        let n = w.num(&format!("{a}.block_count"))? as usize;
        let hidden = w.num(&format!("{a}.embedding_length"))? as usize;
        let hq = w.num(&format!("{a}.attention.head_count"))? as usize;
        let hkv = w.num(&format!("{a}.attention.head_count_kv"))? as usize;
        let inter = w.num(&format!("{a}.feed_forward_length"))? as usize;
        let d = w.num(&format!("{a}.attention.key_length"))? as usize;
        if d != HEAD || !hq.is_multiple_of(hkv) || hq * HEAD != w.shape("llm.layers.0.self_attn.q_proj.weight")?[0] {
            return Err(Error(format!("heads of {d} ({hq} over {hkv}): not the Qwen3 backbone this engine runs")));
        }
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("llm.layers.{i}.");
            let s = |x: &str| format!("{p}{x}");
            let (qkv, _, k) = w.half(ops, &[s("self_attn.q_proj.weight"), s("self_attn.k_proj.weight"), s("self_attn.v_proj.weight")])?;
            if k != hidden {
                return Err(Error(format!("{p}: projections of {k}, the model is {hidden} wide")));
            }
            let (o, _, _) = w.half(ops, &[s("self_attn.o_proj.weight")])?;
            let (gu, _, _) = w.half(ops, &[s("mlp.gate_proj.weight"), s("mlp.up_proj.weight")])?;
            let (down, _, _) = w.half(ops, &[s("mlp.down_proj.weight")])?;
            layers.push(Layer {
                qkv,
                o,
                gu,
                down,
                in_norm: w.dev(ops, &s("input_layernorm.weight"))?,
                post_norm: w.dev(ops, &s("post_attention_layernorm.weight"))?,
                q_norm: w.dev(ops, &s("self_attn.q_norm.weight"))?,
                k_norm: w.dev(ops, &s("self_attn.k_norm.weight"))?,
            });
        }
        let (heads, heads_n, hk) = w.half(ops, &["audio_heads.weight".to_string()])?;
        let audio_emb = w.dev(ops, "audio_embeddings.weight")?;
        if hk != hidden || w.shape("audio_embeddings.weight")? != vec![heads_n, hidden] {
            return Err(Error("the audio heads or embeddings do not fit the backbone".into()));
        }
        Ok(Lm {
            layers,
            norm: w.dev(ops, "llm.norm.weight")?,
            heads,
            audio_emb,
            hidden,
            hq,
            hkv,
            inter,
            eps: w.num(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f32,
            theta: w.num(&format!("{a}.rope.freq_base"))?,
            heads_n,
        })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Work buffers for `rows` rows and `targets` target rows a sequence
    pub fn work(&self, ops: &Ops, rows: usize, targets: usize) -> Result<Work> {
        let (h, g) = (self.hidden, &ops.gpu);
        let qd = self.hq * HEAD;
        Ok(Work {
            rows,
            x: DevBuf::f32(g, rows * h)?,
            xn: DevBuf::new(g, rows * h * 2)?,
            qkv: DevBuf::new(g, rows * (qd + 2 * self.hkv * HEAD) * 2)?,
            qkv3: DevBuf::new(g, rows * 3 * qd * 2)?,
            att: DevBuf::new(g, rows * qd * 2)?,
            tmp: DevBuf::f32(g, rows * h)?,
            gu: DevBuf::new(g, rows * 2 * self.inter * 2)?,
            act: DevBuf::new(g, rows * self.inter * 2)?,
            cs: DevBuf::f32(g, rows * HEAD)?,
            logits: DevBuf::f32(g, 2 * targets * self.heads_n)?,
        })
    }

    /// The rotary table of the rows (cos, sin per pair of HEAD / 2): positions `pos`, Qwen3's in float32
    /// (inv_freq = 1 / theta^(2i / d), the angle position * inv_freq)
    pub fn rope_rows(&self, pos: &[usize]) -> Vec<f32> {
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| 1.0 / (self.theta as f32).powf((2 * i) as f32 / HEAD as f32)).collect();
        let mut t = Vec::with_capacity(pos.len() * HEAD);
        for p in pos {
            for f in &inv {
                let a = *p as f32 * f;
                t.push(a.cos());
                t.push(a.sin());
            }
        }
        t
    }

    /// The pass over `w.x` (the input embeddings): the conditional sequence's `l` rows, then the unconditional
    /// sequence's `t`; the logits of the last `t` rows of each (the targets) into `w.logits` [2 t, heads]
    pub fn pass(&self, ops: &Ops, w: &Work, l: usize, t: usize, mut taps: Option<&mut Taps>) -> Result<()> {
        let nsd = &ops.nsd;
        let r = l + t;
        if r > w.rows || t > l {
            return Err(Error(format!("a pass of {l} + {t} rows over work for {}", w.rows)));
        }
        let (h, qd, kd) = (self.hidden, self.hq * HEAD, self.hkv * HEAD);
        let wq = qd + 2 * kd;
        for (i, ly) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, r, h, ly.in_norm.ptr(), self.eps, none(), none(), none(), w.xn.ptr(), Dt::F16)?;
            ops.linear_h(w.xn.ptr(), r, h, &ly.qkv, wq, w.qkv.ptr(), Dt::F16)?;
            nsd.rms_rope(w.qkv.ptr(), Dt::F16, r, self.hq, HEAD, wq, ly.q_norm.ptr(), self.eps, w.cs.ptr(), HEAD)?;
            nsd.rms_rope(at(&w.qkv, qd * 2), Dt::F16, r, self.hkv, HEAD, wq, ly.k_norm.ptr(), self.eps, w.cs.ptr(), HEAD)?;
            ops.expand_qkv(w.qkv.ptr(), r, self.hq, self.hkv, HEAD, w.qkv3.ptr())?;
            let s3 = 3 * qd;
            for (r0, n) in [(0, l), (l, t)] {
                let q = at(&w.qkv3, r0 * s3 * 2);
                nsd.attention(q, q.wrapping_byte_add(qd * 2), q.wrapping_byte_add(2 * qd * 2), Dt::F16, n, self.hq, HEAD, s3, at(&w.att, r0 * qd * 2), Dt::F16)?;
            }
            ops.linear_h(w.att.ptr(), r, qd, &ly.o, h, w.tmp.ptr(), Dt::F32)?;
            ops.add(w.x.fp(), w.tmp.fp(), r * h)?;
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, r, h, ly.post_norm.ptr(), self.eps, none(), none(), none(), w.xn.ptr(), Dt::F16)?;
            ops.linear_h(w.xn.ptr(), r, h, &ly.gu, 2 * self.inter, w.gu.ptr(), Dt::F16)?;
            nsd.swiglu(w.gu.ptr(), Dt::F16, r, self.inter, w.act.ptr(), Dt::F16)?;
            ops.linear_h(w.act.ptr(), r, self.inter, &ly.down, h, w.tmp.ptr(), Dt::F32)?;
            ops.add(w.x.fp(), w.tmp.fp(), r * h)?;
            if i == 0 {
                if let Some(k) = taps.as_deref_mut() {
                    k.layer0 = crate::ops::read_f32(&w.x, 0, r * h)?;
                }
            }
        }
        if let Some(k) = taps {
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, r, h, self.norm.ptr(), self.eps, none(), none(), none(), w.tmp.ptr(), Dt::F32)?;
            k.hidden = crate::ops::read_f32(&w.tmp, 0, r * h)?;
        }
        // the final norm and the heads over the 2 t rows [l - t, l + t): the conditional targets, then the unconditional
        let r0 = l - t;
        nsd.rms_norm_mod(fp(&w.x, r0 * h).cast(), Dt::F32, 2 * t, h, self.norm.ptr(), self.eps, none(), none(), none(), w.xn.ptr(), Dt::F16)?;
        ops.linear_h(w.xn.ptr(), 2 * t, h, &self.heads, self.heads_n, w.logits.ptr(), Dt::F32)
    }
}
