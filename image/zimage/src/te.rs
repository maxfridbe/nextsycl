//! The text encoder: Qwen3-4B (transformers' `Qwen3Model`, the class the pipeline loads) from ComfyUI's single
//! bfloat16 file (`qwen_3_4b.safetensors`, the official shards' tensors under the same names). The pipeline takes
//! `hidden_states[-2]` - the residual stream after layer 35 of 36, before any norm - of the prompt's tokens, so the last
//! layer is neither loaded nor run.
//!
//! ```text
//!   prompt -> "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" (Qwen3's template, thinking on: nothing
//!             after the assistant's header), at most 512 tokens
//!   per layer:  h = rmsnorm(x);  q k v = h [Wq|Wk|Wv] (32 query heads, 8 key/value heads, 128 features each)
//!               q, k: per-head rmsnorm + RoPE (theta 1e6, pairs (i, 64 + i));  x += causal attention . Wo
//!               x += (silu(h' Wgate) * (h' Wup)) Wdown,  h' = rmsnorm(x)
//! ```
//!
//! The residual stream and the attention float32; the products take half inputs (the matrices made half at load: the
//! bfloat16 values fit), the gated product at 1/256 as the DiT's.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_tok::Tokenizer;

use crate::dit::{Part, Weights};
use crate::ffi;

pub const HIDDEN: usize = 2560;
const HEAD: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;
/// the pipeline's max_sequence_length: longer prompts are cut there (the template's end with them)
pub const MAX_TOKENS: usize = 512;
/// the layers run: hidden_states[-2] of 36
pub const LAYERS: usize = 35;

/// The prompt in Qwen3's chat template as the pipeline applies it (one user turn, the assistant's header, thinking
/// enabled)
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    qkv: Part,
    o: Part,
    gu: Part,
    down: Part,
}

pub struct TextEncoder {
    file: SafeTensors,
    pub tok: Tokenizer,
    pub gpu: Arc<Gpu>,
    nsd: Nsd,
    layers: Vec<Layer>,
    heads: usize,
    kv_heads: usize,
    ffn: usize,
    pub bytes: u64,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// Bytes the encoder takes on a card (its layers' matrices as half)
pub fn plan_bytes(file: &Path) -> Result<u64> {
    let st = SafeTensors::open(file).map_err(ge)?;
    let per: u64 = st.tensors.iter().filter(|t| t.name.starts_with("model.layers.0.")).map(|t| t.elements() * 2).sum();
    Ok(per * LAYERS as u64)
}

impl TextEncoder {
    pub fn load(file: &Path, tokenizer: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let st = SafeTensors::open(file).map_err(ge)?;
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        let nsd = Nsd::new(gpu)?;
        let emb = st.need("model.embed_tokens.weight").map_err(ge)?;
        let n_layer = (0..).take_while(|l| st.tensor(&format!("model.layers.{l}.input_layernorm.weight")).is_some()).count();
        let q = st.need("model.layers.0.self_attn.q_proj.weight").map_err(ge)?;
        let kp = st.need("model.layers.0.self_attn.k_proj.weight").map_err(ge)?;
        let g = st.need("model.layers.0.mlp.gate_proj.weight").map_err(ge)?;
        if emb.shape.get(1) != Some(&(HIDDEN as u64)) || n_layer < LAYERS {
            return Err(Error(format!("{}: {:?} embeddings and {n_layer} layers; Qwen3-4B's are {HIDDEN} wide and 36 deep", file.display(), emb.shape)));
        }
        let (heads, kv_heads, ffn) = (q.shape[0] as usize / HEAD, kp.shape[0] as usize / HEAD, g.shape[0] as usize);
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &st.f32(name).map_err(ge)?) };
        // the matrices of a linear, stacked by rows, made half on the card
        let half = |names: &[String]| -> Result<Part> {
            let mut w = Vec::new();
            let (mut rows, mut kk) = (0usize, 0usize);
            for n in names {
                let t = st.need(n).map_err(ge)?;
                if t.dtype != "BF16" {
                    return Err(Error(format!("{n}: {}, not BF16", t.dtype)));
                }
                rows += t.shape[0] as usize;
                kk = t.shape[1] as usize;
                w.extend_from_slice(&st.read(t).map_err(ge)?);
            }
            let src = DevBuf::new(gpu, w.len())?;
            src.write(0, &w)?;
            let h = DevBuf::new(gpu, rows * kk * 2)?;
            nsd.dequant(src.ptr(), 30, rows * kk, h.ptr(), Dt::F16)?;
            nsd.wait()?;
            Ok(Part { w: Weights::Half(h), rows, k: kk })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        let mut bytes = 0u64;
        for l in 0..LAYERS {
            let p = format!("model.layers.{l}.");
            let layer = Layer {
                in_norm: vec(&format!("{p}input_layernorm.weight"))?,
                post_norm: vec(&format!("{p}post_attention_layernorm.weight"))?,
                q_norm: vec(&format!("{p}self_attn.q_norm.weight"))?,
                k_norm: vec(&format!("{p}self_attn.k_norm.weight"))?,
                qkv: half(&[format!("{p}self_attn.q_proj.weight"), format!("{p}self_attn.k_proj.weight"), format!("{p}self_attn.v_proj.weight")])?,
                o: half(&[format!("{p}self_attn.o_proj.weight")])?,
                gu: half(&[format!("{p}mlp.gate_proj.weight"), format!("{p}mlp.up_proj.weight")])?,
                down: half(&[format!("{p}mlp.down_proj.weight")])?,
            };
            bytes += layer.qkv.bytes() + layer.o.bytes() + layer.gu.bytes() + layer.down.bytes();
            layers.push(layer);
        }
        log(format!("qwen3-4b: {LAYERS} of {n_layer} layers ({heads} / {kv_heads} heads, {ffn} ffn), {:.1} GiB on {} in {:.1} s", bytes as f64 / (1u64 << 30) as f64,
                    gpu.name, t0.elapsed().as_secs_f64()));
        Ok(TextEncoder { file: st, tok, gpu: gpu.clone(), nsd, layers, heads, kv_heads, ffn, bytes })
    }

    /// The prompt's tokens: the template, cut at 512 as the pipeline's tokenizer call does
    pub fn ids(&self, prompt: &str) -> Vec<u32> {
        let mut ids = self.tok.encode(&template(prompt));
        ids.truncate(MAX_TOKENS);
        ids
    }

    /// The tokens' embedding rows, float32 [S, 2560] on the host
    fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.file.need("model.embed_tokens.weight").map_err(ge)?;
        let mut out = Vec::with_capacity(ids.len() * HIDDEN);
        let mut row = vec![0u8; HIDDEN * 2];
        for id in ids {
            self.file.read_into(t, *id as u64 * (HIDDEN * 2) as u64, &mut row).map_err(ge)?;
            out.extend(row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)));
        }
        Ok(out)
    }

    /// The prompt's condition: the hidden state after 35 layers of every token, float32 [S, 2560] on the host
    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        self.encode_ids(&self.ids(prompt), LAYERS, None)
    }

    /// The hidden state after the first `n` layers for `ids` (`taps`: also after these layer counts, for a check)
    pub fn encode_ids(&self, ids: &[u32], n: usize, mut taps: Option<&mut Vec<(usize, Vec<f32>)>>) -> Result<Vec<f32>> {
        let nsd = &self.nsd;
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let s = ids.len();
        let (d, h, kv, f) = (HIDDEN, self.heads, self.kv_heads, self.ffn);
        let qkv_w = (h + 2 * kv) * HEAD;
        let x = DevBuf::from_f32(gpu, &self.embed(ids)?)?;
        let hb = DevBuf::new(gpu, s * d * 2)?;
        let qkv = DevBuf::f32(gpu, s * qkv_w)?;
        let att = DevBuf::f32(gpu, s * h * HEAD)?;
        let atth = DevBuf::new(gpu, s * h * HEAD * 2)?;
        let o = DevBuf::f32(gpu, s * d)?;
        let gu = DevBuf::f32(gpu, s * 2 * f)?;
        let act = DevBuf::new(gpu, s * f * 2)?;
        // (cos, sin) per token and pair (i, 64 + i), float32 as transformers' rotary embedding makes them
        let mut cs = vec![0f32; s * HEAD];
        for t in 0..s {
            for i in 0..HEAD / 2 {
                let inv = 1.0 / THETA.powf((2 * i) as f32 / HEAD as f32);
                let a = t as f32 * inv;
                cs[t * HEAD + 2 * i] = a.cos();
                cs[t * HEAD + 2 * i + 1] = a.sin();
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let f32p = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.fp().wrapping_add(at).cast() };
        for (li, l) in self.layers.iter().take(n).enumerate() {
            // SAFETY: x holds s x d floats, hb s x d halves; the norm weights d floats.
            ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), s as i64, d as i64, l.in_norm.fp(), EPS, std::ptr::null(), hb.ptr()) }, "norm")?;
            l.qkv.apply(nsd, hb.ptr(), s, None, qkv.ptr(), Dt::F32)?;
            nsd.rms_rope(qkv.ptr(), Dt::F32, s, h, HEAD, qkv_w, l.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(f32p(&qkv, h * HEAD), Dt::F32, s, kv, HEAD, qkv_w, l.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.attention_causal(qkv.ptr(), f32p(&qkv, h * HEAD), f32p(&qkv, (h + kv) * HEAD), Dt::F32, s, h, kv, HEAD, qkv_w, qkv_w, att.ptr())?;
            // SAFETY: s x h x 128 floats in, as many halves out.
            ffi::check(unsafe { (k.to_half)(gpu.raw(), att.fp(), atth.ptr(), (s * h * HEAD) as i64) }, "to half")?;
            l.o.apply(nsd, atth.ptr(), s, None, o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            // SAFETY: as above.
            ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), s as i64, d as i64, l.post_norm.fp(), EPS, std::ptr::null(), hb.ptr()) }, "norm")?;
            l.gu.apply(nsd, hb.ptr(), s, None, gu.ptr(), Dt::F32)?;
            // SAFETY: gu holds s x 2f floats, act s x f halves.
            ffi::check(unsafe { (k.swiglu_f)(gpu.raw(), gu.fp(), s as i64, f as i64, ffi::ACT_SCALE, act.ptr()) }, "swiglu")?;
            l.down.apply(nsd, act.ptr(), s, None, o.ptr(), Dt::F32)?;
            nsd.scale(o.ptr(), s * d, 1.0 / ffi::ACT_SCALE)?;
            nsd.gate_add(x.ptr(), Dt::F32, s, d, o.ptr(), Dt::F32, none(), none())?;
            if let Some(t) = taps.as_deref_mut() {
                if t.iter().any(|(at, _)| *at == li + 1) {
                    nsd.wait()?;
                    let v = x.to_f32()?;
                    if let Some(e) = t.iter_mut().find(|(at, _)| *at == li + 1) {
                        e.1 = v;
                    }
                }
            }
        }
        nsd.wait()?;
        x.to_f32()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_template_is_the_pipelines() {
        // tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=True)
        assert_eq!(super::template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n");
    }
}
