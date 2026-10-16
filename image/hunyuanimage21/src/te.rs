//! The text encoder: Qwen2.5-VL-7B's language model (transformers' Qwen2_5_VLTextModel, text only - the three M-RoPE
//! axes then turn alike: plain NEOX RoPE, base 1e6) from ComfyUI's GGUF (`dog`, Q4_0; the vision tower in the file is
//! not read). The pipeline takes hidden_states[-3] - layer 25's output, no final norm - so the last two layers are not
//! loaded. 28 layers of 3584: RMS norm (eps 1e-6) -> q (28 heads of 128), k, v (4 heads), with biases -> RoPE ->
//! causal attention -> out -> residual; RMS norm -> SwiGLU (18944) -> residual.
//!
//! The matrices stay Q4_0 on the card (expanded to half just before each product); the residual stream, attention and
//! the products' outputs are float32. The token embedding stays on the host (a row expanded a token).
//!
//! The prompt goes through the official template (PROMPT_TEMPLATE_ENCODE_V2) and the first 34 tokens - the template's
//! system turn - are dropped from the output (crop_start), at most 1,000 tokens kept after them.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_tok::Tokenizer;

use crate::ffi::K;
use crate::w::{ge, host, q4_0_row, tensor, Keep, Lin};

pub const TEMPLATE_HEAD: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
pub const TEMPLATE_TAIL: &str = "<|im_end|>";
/// the template's tokens dropped from the output
pub const CROP: usize = 34;
/// the most tokens the pipeline encodes (its text_len 1000 after the crop)
pub const MAX_TOKENS: usize = 1000 + CROP;
/// hidden_states[-3]: the output of this many layers
pub const LAYERS_USED: usize = 26;

const DIM: usize = 3584;
const HEADS: usize = 28;
const KV_HEADS: usize = 4;
const HEAD: usize = 128;
const FF: usize = 18944;
const EPS: f32 = 1e-6;
const THETA: f64 = 1e6;

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    qkv: Lin,
    o: Lin,
    gu: Lin,
    down: Lin,
}

pub struct TextEncoder {
    k: K,
    pub tok: Tokenizer,
    layers: Vec<Layer>,
    /// the token embedding's Q4_0 rows, on the host
    embd: Vec<u8>,
    vocab: usize,
    scratch: DevBuf,
    pub bytes: usize,
}

impl TextEncoder {
    pub fn load(file: &Path, tokenizer: &Path, nsd: &Nsd, k: &K, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let gpu: &Arc<Gpu> = &nsd.gpu;
        let f = Gguf::open(file).map_err(ge)?;
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        let et = tensor(&f, "token_embd.weight")?;
        if et.ty != GType::Q4_0 || et.shape.get(1) != Some(&(DIM as u64)) {
            return Err(Error(format!("{}: token_embd.weight is {} {:?}; Qwen2.5-VL-7B Q4_0 [vocab, 3584] here", file.display(), et.ty.name(), et.shape)));
        }
        let vocab = et.shape[0] as usize;
        let embd = f.read(et).map_err(ge)?;
        let mut layers = Vec::with_capacity(LAYERS_USED);
        for l in 0..LAYERS_USED {
            let n = |s: &str| format!("blk.{l}.{s}");
            let lin = |names: &[String], biases: &[String]| -> Result<Lin> {
                let names: Vec<&str> = names.iter().map(String::as_str).collect();
                let biases: Vec<&str> = biases.iter().map(String::as_str).collect();
                Lin::load(&f, nsd, k, &names, &biases, &[], Keep::Stored)
            };
            layers.push(Layer {
                attn_norm: DevBuf::from_f32(gpu, &host(&f, &n("attn_norm.weight"))?)?,
                ffn_norm: DevBuf::from_f32(gpu, &host(&f, &n("ffn_norm.weight"))?)?,
                qkv: lin(&[n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight")], &[n("attn_q.bias"), n("attn_k.bias"), n("attn_v.bias")])?,
                o: lin(&[n("attn_output.weight")], &[])?,
                gu: lin(&[n("ffn_gate.weight"), n("ffn_up.weight")], &[])?,
                down: lin(&[n("ffn_down.weight")], &[])?,
            });
        }
        if layers[0].qkv.n != (HEADS + 2 * KV_HEADS) * HEAD || layers[0].gu.n != 2 * FF {
            return Err(Error(format!("{}: not Qwen2.5-VL-7B's shapes", file.display())));
        }
        let biggest = layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.n * m.k).max().unwrap_or(0);
        let bytes = layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.bytes()).sum::<usize>() + biggest * 2;
        log(format!("text encoder: Qwen2.5-VL-7B, {LAYERS_USED} of its layers (Q4_0, {:.1} GiB) on {} in {:.1} s", bytes as f64 / (1u64 << 30) as f64,
                    gpu.name, t0.elapsed().as_secs_f64()));
        Ok(TextEncoder { k: k.clone(), tok, layers, embd, vocab, scratch: DevBuf::new(gpu, biggest * 2)?, bytes })
    }

    /// The template's ids for `prompt`, truncated as the pipeline's tokenizer (max_length 1034)
    pub fn ids(&self, prompt: &str) -> Vec<u32> {
        let mut ids = self.tok.encode(&format!("{TEMPLATE_HEAD}{prompt}{TEMPLATE_TAIL}"));
        ids.truncate(MAX_TOKENS);
        ids
    }

    /// The prompt's embedding for the DiT: float32 [T - 34, 3584] (hidden_states[-3], the template's tokens dropped)
    pub fn encode(&self, nsd: &Nsd, prompt: &str) -> Result<Vec<f32>> {
        let ids = self.ids(prompt);
        let x = self.run(nsd, &ids, LAYERS_USED)?.to_f32()?;
        Ok(x[CROP.min(ids.len()) * DIM..].to_vec())
    }

    /// The hidden state after `layers` layers for every token: float32 [T, 3584] (a check reads layer 1's)
    pub fn run(&self, nsd: &Nsd, ids: &[u32], layers: usize) -> Result<DevBuf> {
        let gpu = &nsd.gpu;
        let t = ids.len();
        let k = &self.k;
        let mut e = Vec::with_capacity(t * DIM);
        let rb = DIM / 32 * 18;
        for id in ids {
            let i = *id as usize;
            if i >= self.vocab {
                return Err(Error(format!("token {i} is past the vocabulary")));
            }
            e.extend(q4_0_row(&self.embd[i * rb..(i + 1) * rb]));
        }
        let x = DevBuf::from_f32(gpu, &e)?;
        let (qw, kvw) = (HEADS * HEAD, KV_HEADS * HEAD);
        let qs = qw + 2 * kvw;
        let h = DevBuf::f32(gpu, t * DIM)?;
        let hh = DevBuf::new(gpu, t * FF * 2)?;
        let qkv = DevBuf::f32(gpu, t * qs)?;
        let att = DevBuf::f32(gpu, t * qw)?;
        let o = DevBuf::f32(gpu, t * DIM)?;
        let wide = DevBuf::f32(gpu, t * 2 * FF)?;
        let act = DevBuf::f32(gpu, t * FF)?;
        // NEOX RoPE: pair (i, 64 + i) turns by position x theta^(-2i / 128)
        let mut cs = vec![0f32; t * HEAD];
        for p in 0..t {
            for i in 0..HEAD / 2 {
                let inv = (1.0 / THETA.powf(2.0 * i as f64 / HEAD as f64)) as f32;
                let a = p as f32 * inv;
                cs[p * HEAD + 2 * i] = a.cos();
                cs[p * HEAD + 2 * i + 1] = a.sin();
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let mul = |m: &Lin, x: &DevBuf, n: usize, out: &DevBuf| -> Result<()> {
            k.to_half(x.fp(), hh.ptr(), t * n, 1.0)?;
            m.apply(nsd, k, &self.scratch, hh.ptr(), t, out.ptr(), Dt::F32)
        };
        for ly in self.layers.iter().take(layers) {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, DIM, ly.attn_norm.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F32)?;
            mul(&ly.qkv, &h, DIM, &qkv)?;
            nsd.rope(qkv.ptr(), Dt::F32, t, HEADS, HEAD, qs, cs.ptr(), HEAD)?;
            nsd.rope(qkv.fp().wrapping_add(qw).cast(), Dt::F32, t, KV_HEADS, HEAD, qs, cs.ptr(), HEAD)?;
            k.attn_small(qkv.fp(), qs, qkv.fp().wrapping_add(qw), qkv.fp().wrapping_add(qw + kvw), qs, t, HEADS, KV_HEADS, HEAD, 1.0 / (HEAD as f32).sqrt(),
                         std::ptr::null(), true, att.fp())?;
            mul(&ly.o, &att, qw, &o)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, DIM, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, DIM, ly.ffn_norm.ptr(), EPS, none(), none(), none(), h.ptr(), Dt::F32)?;
            mul(&ly.gu, &h, DIM, &wide)?;
            nsd.swiglu(wide.ptr(), Dt::F32, t, FF, act.ptr(), Dt::F32)?;
            mul(&ly.down, &act, FF, &o)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, DIM, o.ptr(), Dt::F32, none(), none())?;
        }
        nsd.wait()?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_template_crop_is_its_system_turn() {
        // the template's head is 34 tokens of Qwen's vocabulary up to and including "user\n" (checked against the
        // pipeline's tokenizer in the reference dumps); the prompt starts right after
        assert!(super::TEMPLATE_HEAD.ends_with("<|im_start|>user\n"));
        assert_eq!(super::CROP, 34);
        assert_eq!(super::MAX_TOKENS, 1034);
    }
}
