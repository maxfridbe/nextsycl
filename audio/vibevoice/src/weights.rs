//! The model's GGUF (gguf-org's conversion of microsoft/VibeVoice-1.5B: the checkpoint's own tensor names, no
//! metadata) and the checkpoint's config.json (vibevoice/VibeVoice-1.5B-hf's: the shapes and constants the file does
//! not state). Tensors are found by name and expanded from their blocks on the host - F32, F16, BF16, Q8_0 (the q8_0
//! file quantizes the language model's matrices and keeps the rest in BF16) - and put on the GPU in half (matrices),
//! or float32 (norms, biases, convolutions). The token embedding stays in the file: a row read a token.

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_gguf::safetensors::f16_to_f32;
use nextsycl_gguf::{GType, Gguf, Tensor};
use serde_json::Value;

use crate::ops::{Mat, Ops};

pub struct Weights {
    pub g: Gguf,
}

/// One block type's values (host): `b` holds whole blocks of `t`
pub fn dequant(t: GType, b: &[u8], out: &mut [f32]) -> Result<()> {
    match t {
        GType::F32 => out.iter_mut().zip(b.chunks_exact(4)).for_each(|(o, c)| *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        GType::F16 => out.iter_mut().zip(b.chunks_exact(2)).for_each(|(o, c)| *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        GType::BF16 => out.iter_mut().zip(b.chunks_exact(2)).for_each(|(o, c)| *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)),
        GType::Q8_0 => {
            for (blk, o) in out.chunks_exact_mut(32).enumerate() {
                let at = blk * 34;
                let d = f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]));
                for (j, v) in o.iter_mut().enumerate() {
                    *v = d * b[at + 2 + j] as i8 as f32;
                }
            }
        }
        t => return Err(Error(format!("tensor type {} is not one this engine reads (F32, F16, BF16, Q8_0)", t.name()))),
    }
    Ok(())
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        let g = Gguf::open(path).map_err(|e| Error(e.0))?;
        let w = Weights { g };
        for name in ["model.language_model.embed_tokens.weight", "model.prediction_head.cond_proj.weight", "model.acoustic_tokenizer.decoder.head.conv.conv.weight",
                     "model.semantic_tokenizer.encoder.head.conv.conv.weight", "model.acoustic_connector.fc1.weight"] {
            if !w.has(name) {
                return Err(Error(format!("{}: no {name} - not a VibeVoice GGUF (gguf-org/vibevoice-gguf's tensor names)", path.display())));
            }
        }
        Ok(w)
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.g.paths[0].display())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    /// A tensor's shape (outermost first)
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensor(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// Rows [r0, r1) of a tensor (its first dimension; the rest flattened), as float32 on the host
    pub fn rows(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let k = if t.shape.len() > 1 { t.shape[1..].iter().product::<u64>() as usize } else { 1 };
        let (per, bytes) = t.ty.block();
        if k % per != 0 || r1 > t.shape.first().copied().unwrap_or(1) as usize {
            return Err(Error(format!("{name}: rows {r0}..{r1} of {k} values ({} blocks of {per})", t.ty.name())));
        }
        let row_bytes = k / per * bytes;
        let mut b = vec![0u8; (r1 - r0) * row_bytes];
        self.g.read_into(t, (r0 * row_bytes) as u64, &mut b).map_err(|e| Error(e.0))?;
        let mut out = vec![0f32; (r1 - r0) * k];
        // the rows split over threads (a matrix of tens of millions of values)
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
        let per_t = (r1 - r0).div_ceil(threads).max(1);
        std::thread::scope(|s| -> Result<()> {
            let jobs: Vec<_> = out.chunks_mut(per_t * k).zip(b.chunks(per_t * row_bytes)).map(|(o, bb)| s.spawn(move || dequant(t.ty, bb, o))).collect();
            for j in jobs {
                j.join().map_err(|_| Error("a dequantizing thread failed".into()))??;
            }
            Ok(())
        })?;
        Ok(out)
    }

    /// A whole tensor as float32 on the host
    pub fn all(&self, name: &str) -> Result<Vec<f32>> {
        let rows = self.tensor(name)?.shape.first().copied().unwrap_or(1) as usize;
        self.rows(name, 0, rows)
    }

    /// A one-value tensor (the latents' scale and bias)
    pub fn scalar(&self, name: &str) -> Result<f32> {
        self.all(name)?.first().copied().ok_or_else(|| Error(format!("{name}: empty")))
    }

    /// A tensor as float32 on the GPU (norms, biases, layer scales, convolutions)
    pub fn vec(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(&ops.gpu, &self.all(name)?)
    }

    /// Matrices [n_i, k] stacked by rows (a fused q / k / v, gate / up), in half on the GPU; `bias`: their biases
    /// (stacked the same way); `pad`: k padded with zero columns to a multiple of 256
    pub fn mat(&self, ops: &Ops, names: &[&str], bias: Option<&[&str]>, pad: bool) -> Result<Mat> {
        let mut k_in = 0;
        let mut parts = Vec::new();
        for name in names {
            let s = self.shape(name)?;
            if s.len() != 2 || (k_in != 0 && s[1] != k_in) {
                return Err(Error(format!("{name}: shape {s:?}, a matrix {k_in} wide expected")));
            }
            k_in = s[1];
            parts.push((*name, s[0]));
        }
        self.mat_rows(ops, &parts.iter().map(|(n, r)| (*n, 0, *r)).collect::<Vec<_>>(), k_in, bias, pad)
    }

    /// Row ranges [r0, r1) of matrices k_in wide, stacked, in half (a slice of a fused matrix: one modulation's
    /// chunk); `bias` the matching ranges' biases
    pub fn mat_rows(&self, ops: &Ops, parts: &[(&str, usize, usize)], k_in: usize, bias: Option<&[&str]>, pad: bool) -> Result<Mat> {
        let k = if pad { k_in.next_multiple_of(256) } else { k_in };
        let n: usize = parts.iter().map(|(_, a, b)| b - a).sum();
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * 2)?;
        if k != k_in {
            w.fill(0)?;
        }
        // in pieces of up to 16M values through a float32 staging buffer
        let step = ((16usize << 20) / k).max(1);
        let stage = DevBuf::f32(gpu, step.min(n.max(1)) * k)?;
        let mut row = 0;
        for (name, a, b) in parts {
            let mut r0 = *a;
            while r0 < *b {
                let r1 = (r0 + step).min(*b);
                let v = self.rows(name, r0, r1)?;
                let v = if k == k_in { v } else { v.chunks(k_in).flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(0.0, k - k_in))).collect() };
                stage.write(0, &crate::bytes(&v))?;
                ops.to_half(stage.fp(), w.ptr().wrapping_byte_add((row + r0 - a) * k * 2), v.len())?;
                gpu.sync()?;
                r0 = r1;
            }
            row += b - a;
        }
        let bias = match bias {
            Some(bs) => {
                let mut v = Vec::with_capacity(n);
                for ((_, a, b), bn) in parts.iter().zip(bs) {
                    v.extend_from_slice(&self.all(bn)?[*a..*b]);
                }
                Some(DevBuf::from_f32(gpu, &v)?)
            }
            None => None,
        };
        Ok(Mat { w, wt: 0, scales: None, bias, n, k, k_in })
    }

    /// The language model's matrices, stacked by rows as `mat` does: kept as the file holds them when that is Q8_0
    /// (the codes and the block scales apart: half the bytes a step reads), else in half. `interleave`: the two
    /// matrices' rows alternated - gate i, up i - as Epi::Swiglu takes them.
    pub fn mat_lm(&self, ops: &Ops, names: &[&str], bias: Option<&[&str]>, interleave: bool) -> Result<Mat> {
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
        if interleave && (names.len() != 2 || rows[0] != rows[1] || bias.is_some()) {
            return Err(Error(format!("{names:?}: two matrices of as many rows, without biases, to interleave")));
        }
        let q8 = names.iter().all(|n| self.tensor(n).is_ok_and(|t| t.ty == GType::Q8_0)) && k.is_multiple_of(32);
        if !q8 && !interleave {
            return self.mat(ops, names, bias, false);
        }
        let n: usize = rows.iter().sum();
        let gpu = &ops.gpu;
        let blocks = k / 32;
        let (w, scales) = if q8 { (DevBuf::new(gpu, n * k)?, Some(DevBuf::new(gpu, n * blocks * 2)?)) } else { (DevBuf::new(gpu, n * k * 2)?, None) };
        // rows [r0, r1) of one matrix on the host: Q8_0 codes and scales (the f16 bits), or float32 values
        let host = |name: &str, r0: usize, r1: usize| -> Result<(Vec<u8>, Vec<u8>, Vec<f32>)> {
            if !q8 {
                return Ok((Vec::new(), Vec::new(), self.rows(name, r0, r1)?));
            }
            let t = self.tensor(name)?;
            let rb = blocks * 34;
            let mut raw = vec![0u8; (r1 - r0) * rb];
            self.g.read_into(t, (r0 * rb) as u64, &mut raw).map_err(|e| Error(e.0))?;
            let (mut codes, mut d) = (Vec::with_capacity((r1 - r0) * k), Vec::with_capacity((r1 - r0) * blocks * 2));
            for b in raw.chunks_exact(34) {
                d.extend_from_slice(&b[..2]);
                codes.extend_from_slice(&b[2..]);
            }
            Ok((codes, d, Vec::new()))
        };
        // output rows from `at` on, from the host
        let stage = if q8 { None } else { Some(DevBuf::f32(gpu, ((8usize << 20) / k).max(2) * k)?) };
        let upload = |at: usize, (codes, d, v): (Vec<u8>, Vec<u8>, Vec<f32>)| -> Result<()> {
            if let Some(sc) = &scales {
                w.write(at * k, &codes)?;
                return sc.write(at * blocks * 2, &d);
            }
            let stage = stage.as_ref().ok_or_else(|| Error("no staging buffer".into()))?;
            stage.write(0, &crate::bytes(&v))?;
            ops.to_half(stage.fp(), w.ptr().wrapping_byte_add(at * k * 2), v.len())?;
            gpu.sync()
        };
        // a staging buffer's worth of output rows a piece (half of it from each matrix when interleaving)
        let step = (((8usize << 20) / k).max(2) / 2).max(1);
        if interleave {
            let mut r0 = 0;
            while r0 < rows[0] {
                let r1 = (r0 + step).min(rows[0]);
                let (a, b) = (host(names[0], r0, r1)?, host(names[1], r0, r1)?);
                let mut out = (Vec::new(), Vec::new(), Vec::new());
                for i in 0..r1 - r0 {
                    for p in [&a, &b] {
                        if q8 {
                            out.0.extend_from_slice(&p.0[i * k..(i + 1) * k]);
                            out.1.extend_from_slice(&p.1[i * blocks * 2..(i + 1) * blocks * 2]);
                        } else {
                            out.2.extend_from_slice(&p.2[i * k..(i + 1) * k]);
                        }
                    }
                }
                upload(2 * r0, out)?;
                r0 = r1;
            }
        } else {
            let mut at = 0;
            for (name, &nr) in names.iter().zip(&rows) {
                let mut r0 = 0;
                while r0 < nr {
                    let r1 = (r0 + 2 * step).min(nr);
                    upload(at + r0, host(name, r0, r1)?)?;
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
                Some(DevBuf::from_f32(gpu, &v)?)
            }
            None => None,
        };
        Ok(Mat { w, wt: if q8 { 2 } else { 0 }, scales, bias, n, k, k_in: k })
    }
}

/// What the checkpoint's config.json (the -hf conversion's) says that the GGUF does not
#[derive(Clone, Debug)]
pub struct Config {
    pub rope_theta: f64,
    pub lm_eps: f32,
    pub heads: usize,
    pub kv_heads: usize,
    /// the acoustic and semantic tokenizers' σ-VAE: per-stage ratios (encoder order), ConvNeXt depths, kernel, norm
    pub ratios: Vec<usize>,
    pub depths: Vec<usize>,
    pub kernel: usize,
    pub vae_eps: f32,
    /// the sampling noise's scale on a voice's latents
    pub vae_std: f32,
    pub head_eps: f32,
    pub freq_dim: usize,
    pub max_period: f64,
    pub audio_bos: u32,
    pub audio_eos: u32,
    pub audio: u32,
    pub eos: u32,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let c: Value = serde_json::from_slice(&b).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Config::parse(&c).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
    }

    pub fn parse(c: &Value) -> Result<Config> {
        let t = &c["text_config"];
        let a = &c["audio_config"];
        let h = &c["diffusion_head_config"];
        let num = |v: &Value, k: &str| v[k].as_f64().ok_or_else(|| Error(format!("no {k}: not VibeVoice's (the -hf checkpoint's) config.json")));
        let list = |v: &Value, k: &str| -> Result<Vec<usize>> {
            v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).ok_or_else(|| Error(format!("no audio_config.{k}")))
        };
        let theta = t["rope_parameters"]["rope_theta"].as_f64().or_else(|| t["rope_theta"].as_f64()).ok_or_else(|| Error("no text_config rope_theta".into()))?;
        let id = |k: &str| num(c, k).map(|v| v as u32);
        Ok(Config {
            rope_theta: theta,
            lm_eps: num(t, "rms_norm_eps")? as f32,
            heads: num(t, "num_attention_heads")? as usize,
            kv_heads: num(t, "num_key_value_heads")? as usize,
            ratios: list(a, "downsampling_ratios")?,
            depths: list(a, "depths")?,
            kernel: num(a, "kernel_size")? as usize,
            vae_eps: num(a, "rms_norm_eps")? as f32,
            vae_std: num(a, "vae_std")? as f32,
            head_eps: num(h, "rms_norm_eps")? as f32,
            freq_dim: num(h, "frequency_embedding_size")? as usize,
            max_period: num(h, "diffusion_max_period")?,
            audio_bos: id("audio_bos_token_id")?,
            audio_eos: id("audio_eos_token_id")?,
            audio: id("audio_token_id")?,
            eos: id("eos_token_id")?,
        })
    }

    /// Samples a frame: the product of the ratios (3,200 at 24 kHz: 7.5 frames a second)
    pub fn hop(&self) -> usize {
        self.ratios.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_expand_as_llama_cpp_does() {
        // Q8_0: d = 0.5 (half 0x3800), codes -2, 3, then zeros
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        b[2] = (-2i8) as u8;
        b[3] = 3;
        let mut o = vec![0f32; 32];
        dequant(GType::Q8_0, &b, &mut o).unwrap();
        assert_eq!(&o[..3], &[-1.0, 1.5, 0.0]);
        // BF16: the float's top half (1.5 = 0x3FC0_0000)
        let mut o = vec![0f32; 2];
        dequant(GType::BF16, &[0xC0, 0x3F, 0x80, 0xBF], &mut o).unwrap();
        assert_eq!(o, vec![1.5, -1.0]);
    }

    #[test]
    fn the_hf_config_is_read() {
        let c = serde_json::json!({
            "audio_bos_token_id": 151652, "audio_eos_token_id": 151653, "audio_token_id": 151654, "eos_token_id": 151643,
            "audio_config": {"depths": [3, 3, 3, 3, 3, 3, 8], "downsampling_ratios": [2, 2, 4, 5, 5, 8], "kernel_size": 7, "rms_norm_eps": 1e-5,
                             "vae_std": 0.625},
            "diffusion_head_config": {"rms_norm_eps": 1e-5, "frequency_embedding_size": 256, "diffusion_max_period": 10000},
            "text_config": {"num_attention_heads": 12, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e6}}
        });
        let c = Config::parse(&c).unwrap();
        assert_eq!((c.hop(), c.heads, c.kv_heads, c.audio), (3200, 12, 2, 151654));
        assert_eq!(c.rope_theta, 1e6);
        assert!(Config::parse(&serde_json::json!({})).is_err());
    }
}
