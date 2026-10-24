//! The DiT's audio codes (modeling_acestep_v15_turbo.py AceStepAudioTokenizer, AudioTokenDetokenizer): 25 Hz latents
//! to one code a 5-frame patch and back. A cover reads its source through both (the "LM hints": what the source's codes
//! say, not its sound); the planner LM writes codes that the detokenizer alone turns into hints.
//!
//! ```text
//!   tokenize    latents [T, 64] (padded to whole patches with the silence latent) -> audio_acoustic_proj -> per patch
//!               [special token, its 5 frames] through the attention pooler (2 layers, each patch its own sequence),
//!               row 0 -> ResidualFSQ (one quantizer, levels 8 8 8 5 5 5: project_in to 6, a soft clamp, rounding,
//!               project_out) -> quantized [T / 5, 2048] and the codes (0 .. 64000)
//!   detokenize  quantized [N, 2048] -> embed_tokens, repeated 5 times plus the 5 special tokens -> 2 layers (each
//!               patch its own sequence) -> norm -> proj_out -> hints [5 N, 64]
//! ```
//!
//! The FSQ's arithmetic (6 values a patch) runs on the host in float32, as vector_quantize_pytorch forces it.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt};

use crate::dit::Config;
use crate::nn::{read_f32, rope_table, Layer, Lin, Loader, Mask, Ops, Scratch, EPS};
use crate::sample::LATENT;

pub struct Codes {
    acoustic_proj: Lin,
    pool_embed: Lin,
    pool_special: Vec<f32>,
    pool_layers: Vec<Layer>,
    pool_norm: DevBuf,
    /// ResidualFSQ's project_in [6, hidden] and project_out [hidden, 6] with their biases, float32 on the host
    q_in: (Vec<f32>, Vec<f32>),
    q_out: (Vec<f32>, Vec<f32>),
    det_embed: Lin,
    det_special: Vec<f32>,
    det_layers: Vec<Layer>,
    det_norm: DevBuf,
    det_out: Lin,
    levels: Vec<u32>,
    pool: usize,
    hidden: usize,
    theta: f64,
}

/// FSQ (preserve_symmetry, a hard clamp after ResidualFSQ's soft one): one patch's 6 values to its level indices
pub fn quantize(z: &[f32], levels: &[u32]) -> Vec<u32> {
    z.iter().zip(levels).map(|(x, l)| {
        let lm1 = (*l - 1) as f32;
        // ResidualFSQ's soft clamp to 1 + 1 / (L - 1), then FSQ's hard one to [-1, 1]
        let c = 1.0 + 1.0 / lm1;
        let x = ((x / c).tanh() * c).clamp(-1.0, 1.0);
        ((lm1 * (x + 1.0) / 2.0) + 0.5).floor().clamp(0.0, lm1) as u32
    }).collect()
}

/// A patch's level indices to its code (FSQ codes_to_indices: the levels' mixed radix, the first fastest)
pub fn index(lv: &[u32], levels: &[u32]) -> u32 {
    let mut basis = 1u32;
    let mut i = 0;
    for (v, l) in lv.iter().zip(levels) {
        i += v * basis;
        basis *= l;
    }
    i
}

/// A code's values in [-1, 1] (indices_to_level_indices, then 2 / (L - 1) level - 1)
pub fn values(code: u32, levels: &[u32]) -> Vec<f32> {
    let mut basis = 1u32;
    levels.iter().map(|l| {
        let lv = (code / basis) % l;
        basis *= l;
        lv as f32 * (2.0 / (*l - 1) as f32) - 1.0
    }).collect()
}

impl Codes {
    pub fn load(l: &Loader, cfg: &Config) -> Result<Codes> {
        // each patch its own sequence: the pooler's of the special token and the patch's frames, the detokenizer's of
        // the patch's frames
        let layers = |p: &str, seg: usize| -> Result<Vec<Layer>> {
            (0..cfg.pooler_layers).map(|i| Layer::load(l, &format!("{p}.layers.{i}"), cfg.heads, cfg.kv_heads, Mask::Segments(seg))).collect()
        };
        let pair = |p: &str| -> Result<(Vec<f32>, Vec<f32>)> { Ok((l.host(&format!("{p}.weight"))?, l.host(&format!("{p}.bias"))?)) };
        let levels = cfg.fsq_levels.clone();
        if levels.is_empty() {
            return Err(Error("the DiT's config names no fsq_input_levels".into()));
        }
        Ok(Codes {
            acoustic_proj: l.lin("tokenizer.audio_acoustic_proj", true)?,
            pool_embed: l.lin("tokenizer.attention_pooler.embed_tokens", true)?,
            pool_special: l.host("tokenizer.attention_pooler.special_token")?,
            pool_layers: layers("tokenizer.attention_pooler", cfg.pool + 1)?,
            pool_norm: l.float("tokenizer.attention_pooler.norm.weight")?,
            q_in: pair("tokenizer.quantizer.project_in")?,
            q_out: pair("tokenizer.quantizer.project_out")?,
            det_embed: l.lin("detokenizer.embed_tokens", true)?,
            det_special: l.host("detokenizer.special_tokens")?,
            det_layers: layers("detokenizer", cfg.pool)?,
            det_norm: l.float("detokenizer.norm.weight")?,
            det_out: l.lin("detokenizer.proj_out", true)?,
            levels,
            pool: cfg.pool,
            hidden: cfg.hidden,
            theta: cfg.theta,
        })
    }

    /// Sequences of `seg` rows (float32 [n seg, hidden] on the host) through `layers` and the norm, each its own
    /// sequence (positions 0 .. seg): float32 [n seg, hidden] on the host
    fn segments(&self, ops: &Ops, s: &mut Scratch, rows: &[f32], seg: usize, layers: &[Layer], norm: &DevBuf) -> Result<Vec<f32>> {
        let c = self.hidden;
        let m = rows.len() / c;
        let x = ops.put_f32(s, "cx", rows)?;
        let one = rope_table(seg, self.theta, 0);
        let mut cs = Vec::with_capacity(m * one.len() / seg);
        for _ in 0..m / seg {
            cs.extend_from_slice(&one);
        }
        let cs = ops.put_f32(s, "ccs", &cs)?;
        for l in layers {
            if l.mask != Mask::Segments(seg) {
                return Err(Error(format!("a layer of {:?} over segments of {seg}", l.mask)));
            }
            ops.layer(s, l, x, m, c, cs)?;
        }
        let out = s.get(&ops.gpu, "cout", m * c * 4)?;
        ops.nsd.rms_norm_mod(x, Dt::F32, m, c, norm.ptr(), EPS, none(), none(), none(), out, Dt::F32)?;
        read_f32(s.buf("cout"), m * c)
    }

    /// float32 rows through a linear on the card: float32 on the host
    fn lin(&self, ops: &Ops, s: &mut Scratch, x: &[f32], w: &Lin) -> Result<Vec<f32>> {
        let m = x.len() / w.k;
        let xi = ops.put_bf(s, "cl_in", x)?;
        let o = s.get(&ops.gpu, "cl_out", m * w.n * 4)?;
        ops.linear(xi, m, w, o, Dt::F32)?;
        read_f32(s.buf("cl_out"), m * w.n)
    }

    /// Latents [T, 64] to (codes [ceil(T / 5)], quantized [ceil(T / 5), hidden]); `silence` pads the last patch
    pub fn tokenize(&self, ops: &Ops, s: &mut Scratch, latents: &[f32], silence: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let (c, p) = (self.hidden, self.pool);
        let t = latents.len() / LATENT;
        let n = t.div_ceil(p);
        let mut x = latents.to_vec();
        x.extend_from_slice(&silence[..(n * p - t) * LATENT]);
        let h = self.lin(ops, s, &x, &self.acoustic_proj)?;
        let h = self.lin(ops, s, &h, &self.pool_embed)?;
        // each patch: [special, its frames]
        let seg = p + 1;
        let mut rows = Vec::with_capacity(n * seg * c);
        for i in 0..n {
            rows.extend_from_slice(&self.pool_special);
            rows.extend_from_slice(&h[i * p * c..(i + 1) * p * c]);
        }
        let out = self.segments(ops, s, &rows, seg, &self.pool_layers, &self.pool_norm)?;
        let (wi, bi) = &self.q_in;
        let d = self.levels.len();
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let r = &out[i * seg * c..i * seg * c + c];
            let z: Vec<f32> = (0..d).map(|j| bi[j] + wi[j * c..(j + 1) * c].iter().zip(r).map(|(a, b)| a * b).sum::<f32>()).collect();
            codes.push(index(&quantize(&z, &self.levels), &self.levels));
        }
        let q = self.quantized(&codes);
        Ok((codes, q))
    }

    /// Codes to their quantized vectors [n, hidden] (ResidualFSQ.get_output_from_indices: the code's values through
    /// project_out)
    pub fn quantized(&self, codes: &[u32]) -> Vec<f32> {
        let c = self.hidden;
        let (wo, bo) = &self.q_out;
        let d = self.levels.len();
        let mut q = Vec::with_capacity(codes.len() * c);
        for code in codes {
            let v = values(*code, &self.levels);
            q.extend((0..c).map(|o| bo[o] + (0..d).map(|j| wo[o * d + j] * v[j]).sum::<f32>()));
        }
        q
    }

    /// Quantized [n, hidden] to the 25 Hz hints [5 n, 64]
    pub fn detokenize(&self, ops: &Ops, s: &mut Scratch, quantized: &[f32]) -> Result<Vec<f32>> {
        let (c, p) = (self.hidden, self.pool);
        let n = quantized.len() / c;
        let e = self.lin(ops, s, quantized, &self.det_embed)?;
        let mut rows = Vec::with_capacity(n * p * c);
        for i in 0..n {
            for k in 0..p {
                rows.extend(e[i * c..(i + 1) * c].iter().zip(&self.det_special[k * c..(k + 1) * c]).map(|(a, b)| a + b));
            }
        }
        let out = self.segments(ops, s, &rows, p, &self.det_layers, &self.det_norm)?;
        self.lin(ops, s, &out, &self.det_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_is_vector_quantize_pytorchs() {
        let levels = [8, 8, 8, 5, 5, 5];
        // 0 sits between levels 3 and 4 of 8 (floor(3.5 + 0.5) = 4), at level 2 of 5
        assert_eq!(quantize(&[0.0; 6], &levels), vec![4, 4, 4, 2, 2, 2]);
        // far out: clamped to the ends
        assert_eq!(quantize(&[9.0, -9.0, 9.0, -9.0, 9.0, -9.0], &levels), vec![7, 0, 7, 0, 4, 0]);
        let lv = [1, 2, 3, 4, 0, 1];
        let code = index(&lv, &levels);
        assert_eq!(code, 1 + 2 * 8 + 3 * 64 + 4 * 512 + 12800);
        let v = values(code, &levels);
        assert!((v[0] - (2.0 / 7.0 - 1.0)).abs() < 1e-6 && (v[3] - 1.0).abs() < 1e-6 && (v[4] + 1.0).abs() < 1e-6);
        assert_eq!(index(&[7, 7, 7, 4, 4, 4], &levels), 63_999);
    }
}
