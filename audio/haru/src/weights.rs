//! The checkpoint (higurashi-ai/Haru-4B: one model.safetensors in bfloat16 - the language model `lm.*`, the two audio
//! projectors, the MeanFlow head `head.*`, the codec `codec.*` and the latents' scale and bias - beside its
//! config.json and tokenizer.json). Tensors are found by name and put on the GPU as the file holds them: the matrices
//! in bfloat16 (the products read the checkpoint's own values), the rest in float32.

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_gguf::safetensors::{f16_to_f32, SafeTensors, StTensor};
use serde_json::Value;

use crate::ops::{Mat, Ops};

pub struct Weights {
    pub st: SafeTensors,
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Float32 values (that came from bfloat16: exact) as bfloat16 bytes
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect()
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let st = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let w = Weights { st };
        for name in ["lm.embed_tokens.weight", "head.cond_proj.weight", "codec.decoder.head.weight", "codec.encoder.stem.weight", "model_audio_proj.fc1.weight",
                     "user_audio_proj.fc1.weight", "latent_scale"] {
            if !w.has(name) {
                return Err(Error(format!("{}: no {name} - not a Haru checkpoint (higurashi-ai/Haru-4B's model.safetensors)", path.display())));
            }
        }
        Ok(w)
    }

    pub fn tensor(&self, name: &str) -> Result<&StTensor> {
        self.st.need(name).map_err(|e| Error(e.0))
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_some()
    }

    /// A tensor's shape (outermost first)
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// A whole tensor as float32 on the host
    pub fn all(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32(name).map_err(|e| Error(e.0))
    }

    /// A one-value tensor (the latents' scale and bias)
    pub fn scalar(&self, name: &str) -> Result<f32> {
        self.all(name)?.first().copied().ok_or_else(|| Error(format!("{name}: empty")))
    }

    /// A tensor as float32 on the GPU (norms, biases, layer scales, convolutions); `plus`: added to every value (a
    /// zero-centred norm's 1 + w)
    pub fn vec(&self, ops: &Ops, name: &str, plus: f32) -> Result<DevBuf> {
        let v: Vec<f32> = self.all(name)?.into_iter().map(|x| x + plus).collect();
        DevBuf::from_f32(&ops.gpu, &v)
    }

    pub fn is_bf16(&self, name: &str) -> bool {
        self.tensor(name).is_ok_and(|t| t.dtype == "BF16")
    }

    /// Rows [r0, r1) of a matrix tensor as bfloat16 bytes (the file's own, or converted from float32 / half)
    fn rows_bf16(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<u8>> {
        let t = self.tensor(name)?;
        let k: usize = t.shape.iter().skip(1).product::<u64>() as usize;
        match t.dtype.as_str() {
            "BF16" => {
                let mut b = vec![0u8; (r1 - r0) * k * 2];
                self.st.read_into(t, (r0 * k * 2) as u64, &mut b).map_err(|e| Error(e.0))?;
                Ok(b)
            }
            "F32" | "F16" => {
                let es = if t.dtype == "F32" { 4 } else { 2 };
                let mut b = vec![0u8; (r1 - r0) * k * es];
                self.st.read_into(t, (r0 * k * es) as u64, &mut b).map_err(|e| Error(e.0))?;
                let v: Vec<f32> = if es == 4 {
                    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
                } else {
                    b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
                };
                // rounded to nearest even
                Ok(v.iter().flat_map(|x| {
                    let u = x.to_bits();
                    (((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
                }).collect())
            }
            d => Err(Error(format!("{name}: {d}, not a float type"))),
        }
    }

    /// Matrices [n_i, k] stacked by rows (a fused projection), bfloat16 on the GPU; `interleave`: two of as many rows
    /// alternated (gate i, up i: Epi::Swiglu's order); `bias`: their biases, stacked the same way
    pub fn mat(&self, ops: &Ops, names: &[&str], bias: Option<&[&str]>, interleave: bool) -> Result<Mat> {
        let mut k = 0;
        let mut rows = Vec::new();
        for name in names {
            let s = self.shape(name)?;
            if s.len() != 2 || (k != 0 && s[1] != k) {
                return Err(Error(format!("{name}: shape {s:?}, a matrix {k} wide expected")));
            }
            k = s[1];
            rows.push(s[0]);
        }
        if interleave && (names.len() != 2 || rows[0] != rows[1]) {
            return Err(Error(format!("{names:?}: two matrices of as many rows to interleave")));
        }
        let n: usize = rows.iter().sum();
        let w = DevBuf::new(&ops.gpu, n * k * 2)?;
        // in pieces of about 32 MiB
        let step = ((32usize << 20) / (k * 2)).max(1);
        if interleave {
            let mut r0 = 0;
            while r0 < rows[0] {
                let r1 = (r0 + step / 2).max(r0 + 1).min(rows[0]);
                let (a, b) = (self.rows_bf16(names[0], r0, r1)?, self.rows_bf16(names[1], r0, r1)?);
                let mut out = Vec::with_capacity(a.len() * 2);
                for (ra, rb) in a.chunks_exact(k * 2).zip(b.chunks_exact(k * 2)) {
                    out.extend_from_slice(ra);
                    out.extend_from_slice(rb);
                }
                w.write(2 * r0 * k * 2, &out)?;
                r0 = r1;
            }
        } else {
            let mut at = 0;
            for (name, &nr) in names.iter().zip(&rows) {
                let mut r0 = 0;
                while r0 < nr {
                    let r1 = (r0 + step).min(nr);
                    w.write((at + r0) * k * 2, &self.rows_bf16(name, r0, r1)?)?;
                    r0 = r1;
                }
                at += nr;
            }
        }
        let bias = match bias {
            Some(bs) => {
                let mut v = Vec::with_capacity(n);
                for bn in bs {
                    v.extend(self.all(bn)?);
                }
                if v.len() != n {
                    return Err(Error(format!("{bs:?}: {} biases for {n} rows", v.len())));
                }
                Some(DevBuf::from_f32(&ops.gpu, &v)?)
            }
            None => None,
        };
        Ok(Mat { w, wt: 1, bias, n, k })
    }

    /// A matrix [n, k] from host values: bfloat16 on the GPU when `bf16` (values that came from bfloat16, so exact),
    /// else float32
    pub fn mat_host(&self, ops: &Ops, v: &[f32], n: usize, k: usize, bf16: bool, bias: Option<&[f32]>) -> Result<Mat> {
        if v.len() != n * k {
            return Err(Error(format!("{} values for a [{n}, {k}] matrix", v.len())));
        }
        let gpu = &ops.gpu;
        let w = if bf16 {
            let b = bf16_bytes(v);
            let w = DevBuf::new(gpu, b.len())?;
            w.write(0, &b)?;
            w
        } else {
            DevBuf::from_f32(gpu, v)?
        };
        let bias = bias.map(|b| DevBuf::from_f32(gpu, b)).transpose()?;
        Ok(Mat { w, wt: if bf16 { 1 } else { 3 }, bias, n, k })
    }
}

/// The checkpoint's config.json (`model_type: haru`): the frame, the special ids, the language model's, the codec's
/// and the head's shapes
#[derive(Clone, Debug)]
pub struct Config {
    pub rate: usize,
    pub frame: usize,
    pub latent: usize,
    pub pad: u32,
    pub eop: u32,
    pub eos: Option<u32>,
    pub hidden: usize,
    pub ffn: usize,
    pub layers: usize,
    /// "linear_attention" / "full_attention" per layer
    pub full: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot: usize,
    pub theta: f64,
    pub eps: f32,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_k: usize,
    pub vocab: usize,
    pub ratios: Vec<usize>,
    pub codec_eps: f32,
    pub latent_std: f32,
    pub head_layers: usize,
    pub head_hidden: usize,
    pub head_eps: f32,
    pub interval: f32,
    pub head_steps: usize,
    pub guidance: f32,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let c: Value = serde_json::from_slice(&b).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Config::parse(&c).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
    }

    pub fn parse(c: &Value) -> Result<Config> {
        if c["model_type"].as_str().unwrap_or("haru") != "haru" {
            return Err(Error(format!("model_type {}: not a Haru checkpoint", c["model_type"])));
        }
        let num = |v: &Value, k: &str| v[k].as_f64().ok_or_else(|| Error(format!("no {k}: not Haru's config.json")));
        let us = |v: &Value, k: &str| num(v, k).map(|x| x as usize);
        let list = |v: &Value, k: &str| -> Result<Vec<usize>> {
            v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).ok_or_else(|| Error(format!("no {k}")))
        };
        let lm = &c["lm"];
        let codec = &c["codec"];
        let head = &c["head"];
        let layers = us(lm, "num_hidden_layers")?;
        let full: Vec<bool> = match lm["layer_types"].as_array() {
            Some(a) => a.iter().map(|x| x.as_str() == Some("full_attention")).collect(),
            None => {
                let every = us(lm, "full_attention_interval")?;
                (0..layers).map(|l| (l + 1) % every == 0).collect()
            }
        };
        if full.len() != layers {
            return Err(Error(format!("{} layer types for {layers} layers", full.len())));
        }
        let rp = &lm["rope_parameters"];
        let head_dim = us(lm, "head_dim")?;
        let partial = rp["partial_rotary_factor"].as_f64().or_else(|| lm["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let theta = rp["rope_theta"].as_f64().or_else(|| lm["rope_theta"].as_f64()).ok_or_else(|| Error("no rope_theta".into()))?;
        Ok(Config {
            rate: us(c, "sample_rate")?,
            frame: us(c, "frame_samples")?,
            latent: us(c, "latent_dim")?,
            pad: us(c, "pad_token_id")? as u32,
            eop: us(c, "eop_token_id")? as u32,
            eos: lm["eos_token_id"].as_u64().map(|x| x as u32),
            hidden: us(lm, "hidden_size")?,
            ffn: us(lm, "intermediate_size")?,
            layers,
            full,
            heads: us(lm, "num_attention_heads")?,
            kv_heads: us(lm, "num_key_value_heads")?,
            head_dim,
            rot: (head_dim as f64 * partial) as usize,
            theta,
            eps: num(lm, "rms_norm_eps")? as f32,
            lin_k_heads: us(lm, "linear_num_key_heads")?,
            lin_v_heads: us(lm, "linear_num_value_heads")?,
            lin_k_dim: us(lm, "linear_key_head_dim")?,
            lin_v_dim: us(lm, "linear_value_head_dim")?,
            conv_k: us(lm, "linear_conv_kernel_dim")?,
            vocab: us(lm, "vocab_size")?,
            ratios: list(codec, "ratios")?,
            codec_eps: num(codec, "norm_eps").unwrap_or(1e-5) as f32,
            latent_std: num(codec, "latent_std").unwrap_or(0.5) as f32,
            head_layers: us(head, "num_layers")?,
            head_hidden: us(head, "hidden_size")?,
            head_eps: num(head, "norm_eps").unwrap_or(1e-5) as f32,
            interval: num(head, "interval_embed").unwrap_or(7.0) as f32,
            head_steps: us(head, "num_steps").unwrap_or(1).max(1),
            guidance: num(head, "guidance_scale").unwrap_or(1.5) as f32,
        })
    }

    /// The Gated DeltaNet's convolved channels: queries, keys and values
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_release_config_is_read() {
        let c = serde_json::json!({
            "model_type": "haru", "sample_rate": 24000, "frame_samples": 3200, "latent_dim": 64, "pad_token_id": 248076, "eop_token_id": 248071,
            "lm": {"vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216, "num_hidden_layers": 4, "num_attention_heads": 16,
                   "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6, "full_attention_interval": 4, "linear_conv_kernel_dim": 4,
                   "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                   "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000}, "eos_token_id": 248044},
            "codec": {"ratios": [8, 5, 5, 4, 2, 2], "norm_eps": 1e-5, "latent_std": 0.5},
            "head": {"hidden_size": 2560, "num_layers": 4, "norm_eps": 1e-5, "interval_embed": 7.0, "num_steps": 1, "guidance_scale": 1.5}
        });
        let c = Config::parse(&c).unwrap();
        assert_eq!((c.rot, c.conv_dim(), c.full.clone(), c.eos), (64, 8192, vec![false, false, false, true], Some(248044)));
        assert!(Config::parse(&serde_json::json!({"model_type": "other"})).is_err());
    }

    #[test]
    fn floats_round_to_bfloat16() {
        assert_eq!(bf16_bytes(&[1.5, -1.0]), vec![0xC0, 0x3F, 0x80, 0xBF]);
    }
}
