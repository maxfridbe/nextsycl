//! nextsycl-qwen3vl: Qwen3-VL's language model as a text encoder - what Qwen-Image 2.1 (and MiniMax H3) condition on.
//! It reads ComfyUI's int8 ConvRot file (`qwen3vl_8b_int8_convrot.safetensors`: every matrix int8 with a scale per
//! output row, its input features rotated by a normalized regular Hadamard matrix in groups of 256) and its tokenizer
//! (`tokenizer.json`), and returns the last layer's hidden state for every token - before the final norm, as the
//! image pipelines take it.
//!
//! One layer, as the reference computes it (activations float32; each linear quantizes its rotated input per row):
//!
//! ```text
//!   h = rmsnorm(x)                  q k v = h [Wq|Wk|Wv]   (32 query heads, 8 key/value heads, 128 features each)
//!   q, k: per-head rmsnorm + rotary positions (theta 5e6, pairs (i, 64 + i))
//!   x += attention(q, k, v) Wo      causal; 4 query heads share each key/value head
//!   h = rmsnorm(x)
//!   x += (silu(h Wgate) * (h Wup)) Wdown
//! ```
//!
//! Text-only prompts: the three M-RoPE axes carry the same position, so the rotation is plain RoPE. Images in the
//! prompt (an edit's reference pictures through the vision tower) are not read yet.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_tok::Tokenizer;

const EPS: f32 = 1e-6;
const HEAD: usize = 128;
const GROUP: usize = 256;

/// One layer's weights on the GPU
struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    /// q | k | v rows, int8 [qkv_rows, hidden], and their scales
    qkv: DevBuf,
    qkv_s: DevBuf,
    o: DevBuf,
    o_s: DevBuf,
    /// gate | up rows
    gu: DevBuf,
    gu_s: DevBuf,
    down: DevBuf,
    down_s: DevBuf,
}

pub struct TextEncoder {
    file: SafeTensors,
    pub tok: Tokenizer,
    gpu: Arc<Gpu>,
    layers: Vec<Layer>,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    theta: f64,
    /// the normalized 256 x 256 regular Hadamard matrix (the embedding rows are un-rotated on the host)
    hadamard: Vec<f32>,
}

/// The normalized regular Hadamard matrix of size g: H4 (x) H4 (x) ... / sqrt(g) (comfy-kitchen's ConvRot)
pub fn hadamard(g: usize) -> Vec<f32> {
    const H4: [[i32; 4]; 4] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];
    let norm = 1.0 / (g as f32).sqrt();
    let mut h = vec![0f32; g * g];
    for i in 0..g {
        for j in 0..g {
            let (mut a, mut b, mut s, mut sign) = (i, j, 1, 1);
            while s < g {
                sign *= H4[a % 4][b % 4];
                a /= 4;
                b /= 4;
                s *= 4;
            }
            h[i * g + j] = sign as f32 * norm;
        }
    }
    h
}

/// A file-format error as this crate's
fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

fn upload_f32(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    DevBuf::from_f32(gpu, v)
}

impl TextEncoder {
    /// The encoder from its int8 ConvRot file and tokenizer, its layers on `gpu`
    pub fn load(file: &Path, tokenizer: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let st = SafeTensors::open(file).map_err(ge)?;
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        let emb = st.need("model.embed_tokens.weight").map_err(ge)?;
        let hidden = *emb.shape.get(1).ok_or_else(|| Error("the token embedding is not 2-D".into()))? as usize;
        let n_layer = (0..).take_while(|l| st.tensor(&format!("model.layers.{l}.input_layernorm.weight")).is_some()).count();
        let q = st.need("model.layers.0.self_attn.q_proj.weight").map_err(ge)?;
        let k = st.need("model.layers.0.self_attn.k_proj.weight").map_err(ge)?;
        let g = st.need("model.layers.0.mlp.gate_proj.weight").map_err(ge)?;
        let heads = q.shape[0] as usize / HEAD;
        let kv_heads = k.shape[0] as usize / HEAD;
        let ffn = g.shape[0] as usize;
        for name in ["model.layers.0.self_attn.q_proj.comfy_quant", "model.embed_tokens.comfy_quant"] {
            let raw = st.read(st.need(name).map_err(ge)?).map_err(ge)?;
            let meta: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| Error(format!("{name}: {e}")))?;
            if meta["format"] != "int8_tensorwise" || meta["convrot"] != true || meta["convrot_groupsize"] != GROUP {
                return Err(Error(format!("{}: {name} is {meta}, not int8 ConvRot in groups of {GROUP}", file.display())));
            }
        }
        // the matrices of a layer, concatenated by rows where one linear reads them together
        let int8 = |names: &[&str]| -> Result<(DevBuf, DevBuf)> {
            let mut w = Vec::new();
            let mut s = Vec::new();
            for n in names {
                let t = st.need(&format!("{n}.weight")).map_err(ge)?;
                if t.dtype != "I8" {
                    return Err(Error(format!("{n}: {}, not I8", t.dtype)));
                }
                w.extend_from_slice(&st.read(t).map_err(ge)?);
                s.extend(st.f32(&format!("{n}.weight_scale")).map_err(ge)?);
            }
            let wb = DevBuf::new(gpu, w.len())?;
            wb.write(0, &w)?;
            Ok((wb, upload_f32(gpu, &s)?))
        };
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = format!("model.layers.{l}.");
            let (qkv, qkv_s) = int8(&[&format!("{p}self_attn.q_proj"), &format!("{p}self_attn.k_proj"), &format!("{p}self_attn.v_proj")])?;
            let (o, o_s) = int8(&[&format!("{p}self_attn.o_proj")])?;
            let (gu, gu_s) = int8(&[&format!("{p}mlp.gate_proj"), &format!("{p}mlp.up_proj")])?;
            let (down, down_s) = int8(&[&format!("{p}mlp.down_proj")])?;
            layers.push(Layer {
                in_norm: upload_f32(gpu, &st.f32(&format!("{p}input_layernorm.weight")).map_err(ge)?)?,
                post_norm: upload_f32(gpu, &st.f32(&format!("{p}post_attention_layernorm.weight")).map_err(ge)?)?,
                q_norm: upload_f32(gpu, &st.f32(&format!("{p}self_attn.q_norm.weight")).map_err(ge)?)?,
                k_norm: upload_f32(gpu, &st.f32(&format!("{p}self_attn.k_norm.weight")).map_err(ge)?)?,
                qkv, qkv_s, o, o_s, gu, gu_s, down, down_s,
            });
        }
        log(format!("qwen3vl: {n_layer} layers ({hidden} wide, {heads} / {kv_heads} heads, {ffn} ffn) on {} in {:.1} s", gpu.name,
                    t0.elapsed().as_secs_f64()));
        Ok(TextEncoder { file: st, tok, gpu: gpu.clone(), layers, hidden, heads, kv_heads, ffn, theta: 5_000_000.0, hadamard: hadamard(GROUP) })
    }

    /// The tokens' embedding rows, dequantized and un-rotated, float32 [S, hidden] on the host
    fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.file.need("model.embed_tokens.weight").map_err(ge)?;
        let s = self.file.need("model.embed_tokens.weight_scale").map_err(ge)?;
        let d = self.hidden;
        let mut out = vec![0f32; ids.len() * d];
        let mut row = vec![0u8; d];
        let mut sc = [0u8; 4];
        for (i, id) in ids.iter().enumerate() {
            self.file.read_into(t, *id as u64 * d as u64, &mut row).map_err(ge)?;
            self.file.read_into(s, *id as u64 * 4, &mut sc).map_err(ge)?;
            let scale = f32::from_le_bytes(sc);
            let o = &mut out[i * d..(i + 1) * d];
            // x = (q * scale) . H per group of 256 (comfy-kitchen's dequantize_int8_embedding)
            for g in 0..d / GROUP {
                let src: Vec<f32> = (0..GROUP).map(|j| row[g * GROUP + j] as i8 as f32 * scale).collect();
                for j in 0..GROUP {
                    let mut acc = 0f32;
                    for (k, x) in src.iter().enumerate() {
                        acc += x * self.hadamard[j * GROUP + k];
                    }
                    o[g * GROUP + j] = acc;
                }
            }
        }
        Ok(out)
    }

    /// The hidden state after the last layer for every token of `ids`: float32 [S, hidden] on the GPU
    pub fn encode_ids(&self, nsd: &Nsd, ids: &[u32]) -> Result<DevBuf> {
        self.encode_layers(nsd, ids, self.layers.len())
    }

    /// The hidden state after the first `n` layers (a check of the layers one by one)
    pub fn encode_layers(&self, nsd: &Nsd, ids: &[u32], n: usize) -> Result<DevBuf> {
        let s = ids.len();
        let (d, h, kv, f) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let qkv_w = (h + 2 * kv) * HEAD;
        let gpu = &self.gpu;
        let x = upload_f32(gpu, &self.embed(ids)?)?;
        let hb = DevBuf::f32(gpu, s * d)?;
        let qkv = DevBuf::f32(gpu, s * qkv_w)?;
        let att = DevBuf::f32(gpu, s * h * HEAD)?;
        let o = DevBuf::f32(gpu, s * d)?;
        let gu = DevBuf::f32(gpu, s * 2 * f)?;
        let act = DevBuf::f32(gpu, s * f)?;
        // (cos, sin) per token and pair (i, 64 + i)
        let mut cs = vec![0f32; s * HEAD];
        for p in 0..s {
            for i in 0..HEAD / 2 {
                let a = p as f64 / self.theta.powf(2.0 * i as f64 / HEAD as f64);
                cs[p * HEAD + 2 * i] = a.cos() as f32;
                cs[p * HEAD + 2 * i + 1] = a.sin() as f32;
            }
        }
        let cs = upload_f32(gpu, &cs)?;
        let f32p = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.fp().wrapping_add(at).cast() };
        for l in self.layers.iter().take(n) {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, l.in_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
            nsd.int8_linear(hb.ptr(), Dt::F32, s, d, l.qkv.ptr(), qkv_w, l.qkv_s.ptr(), qkv_w, none(), qkv.ptr(), Dt::F32, GROUP)?;
            nsd.rms_rope(qkv.ptr(), Dt::F32, s, h, HEAD, qkv_w, l.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(f32p(&qkv, h * HEAD), Dt::F32, s, kv, HEAD, qkv_w, l.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.attention_causal(qkv.ptr(), f32p(&qkv, h * HEAD), f32p(&qkv, (h + kv) * HEAD), Dt::F32, s, h, kv, HEAD, qkv_w, qkv_w, att.ptr())?;
            nsd.int8_linear(att.ptr(), Dt::F32, s, h * HEAD, l.o.ptr(), d, l.o_s.ptr(), d, none(), o.ptr(), Dt::F32, GROUP)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, s, d, l.post_norm.ptr(), EPS, none(), none(), none(), hb.ptr(), Dt::F32)?;
            nsd.int8_linear(hb.ptr(), Dt::F32, s, d, l.gu.ptr(), 2 * f, l.gu_s.ptr(), 2 * f, none(), gu.ptr(), Dt::F32, GROUP)?;
            nsd.swiglu(gu.ptr(), Dt::F32, s, f, act.ptr(), Dt::F32)?;
            nsd.int8_linear(act.ptr(), Dt::F32, s, f, l.down.ptr(), d, l.down_s.ptr(), d, none(), o.ptr(), Dt::F32, GROUP)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
        }
        nsd.wait()?;
        Ok(x)
    }

    /// `text` tokenized as it is (the caller builds the chat template)
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }
}

#[cfg(test)]
mod tests {
    use super::hadamard;

    #[test]
    fn the_hadamard_matrix_is_orthonormal_and_symmetric() {
        let g = 16;
        let h = hadamard(g);
        for i in 0..g {
            for j in 0..g {
                assert_eq!(h[i * g + j], h[j * g + i]);
                let dot: f32 = (0..g).map(|k| h[i * g + k] * h[j * g + k]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5);
            }
        }
    }
}
