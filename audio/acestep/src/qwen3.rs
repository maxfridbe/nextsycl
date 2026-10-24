//! Qwen3 (transformers' Qwen3Model) on the shared layer (nn.rs): ACE-Step's text encoder (Qwen3-Embedding-0.6B: its
//! last hidden state is the caption's encoding, its token embeddings the lyrics') and the planner's backbone (the
//! 5Hz LM, 1.7B: lm.rs adds its cache and its head). Causal attention over the prompt; the embedding table is read on
//! the host (a row a token).

use std::path::Path;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Dt;
use nextsycl_gguf::safetensors::SafeTensors;
use serde_json::Value;

use crate::nn::{bf, rope_table, Layer, Loader, Mask, Ops, Scratch, HD};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub inter: usize,
    pub vocab: usize,
    pub theta: f64,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let v: Value = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| Error(format!("{}: not JSON", path.display())))?;
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{}: no {k}", path.display())));
        if v["head_dim"].as_u64().is_some_and(|d| d as usize != HD) {
            return Err(Error(format!("{}: head_dim {} (128 expected)", path.display(), v["head_dim"])));
        }
        Ok(Config { hidden: n("hidden_size")?, layers: n("num_hidden_layers")?, heads: n("num_attention_heads")?, kv_heads: n("num_key_value_heads")?,
                    inter: n("intermediate_size")?, vocab: n("vocab_size")?, theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0) })
    }
}

pub struct Qwen3 {
    pub cfg: Config,
    /// the token embeddings, bfloat16 bits [vocab, hidden] (on the host)
    pub emb: Vec<u16>,
    pub layers: Vec<Layer>,
    pub norm: DevBuf,
    pub bytes: u64,
}

impl Qwen3 {
    /// `prefix`: the tensors' prefix ("" for Qwen3-Embedding's bare names, "model." for a causal LM's)
    pub fn load(weights: &Path, cfg: Config, prefix: &str, gpu: &Arc<Gpu>) -> Result<Qwen3> {
        let st = SafeTensors::open(weights).map_err(|e| Error(e.0))?;
        let l = Loader::new(&st, gpu);
        let (raw, shape) = l.bf_raw(&format!("{prefix}embed_tokens.weight"))?;
        if shape != [cfg.vocab as u64, cfg.hidden as u64] {
            return Err(Error(format!("{}: embed_tokens {shape:?}, not [{}, {}]", weights.display(), cfg.vocab, cfg.hidden)));
        }
        let emb = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut layers = Vec::new();
        for i in 0..cfg.layers {
            layers.push(Layer::load(&l, &format!("{prefix}layers.{i}"), cfg.heads, cfg.kv_heads, Mask::Causal)?);
        }
        let norm = l.float(&format!("{prefix}norm.weight"))?;
        Ok(Qwen3 { bytes: l.bytes.get(), cfg, emb, layers, norm })
    }

    /// the embeddings of `ids` as floats [n, hidden]
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let mut x = Vec::with_capacity(ids.len() * h);
        for &t in ids {
            let t = t as usize;
            if t >= self.cfg.vocab {
                return Err(Error(format!("token {t} past the vocabulary ({})", self.cfg.vocab)));
            }
            x.extend(self.emb[t * h..(t + 1) * h].iter().map(|b| bf(*b)));
        }
        Ok(x)
    }

    /// The last hidden state of `ids` (causal, positions from 0), float32 [n, hidden] on the host
    pub fn encode(&self, ops: &Ops, s: &mut Scratch, ids: &[u32]) -> Result<Vec<f32>> {
        let (m, c) = (ids.len(), self.cfg.hidden);
        if m == 0 {
            return Err(Error("nothing to encode".into()));
        }
        let x = ops.put_f32(s, "qx", &self.embed(ids)?)?;
        let cs = ops.put_f32(s, "qcs", &rope_table(m, self.cfg.theta, 0))?;
        for l in &self.layers {
            ops.layer(s, l, x, m, c, cs)?;
        }
        let out = s.get(&ops.gpu, "qout", m * c * 4)?;
        ops.nsd.rms_norm_mod(x, Dt::F32, m, c, self.norm.ptr(), crate::nn::EPS, nextsycl_diffusion::kernels::none(), nextsycl_diffusion::kernels::none(),
                             nextsycl_diffusion::kernels::none(), out, Dt::F32)?;
        crate::nn::read_f32(s.buf("qout"), m * c)
    }
}
