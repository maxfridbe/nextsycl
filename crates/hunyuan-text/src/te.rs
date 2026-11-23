//! The text encoder: Qwen2.5-VL-7B's language model (transformers' Qwen2_5_VLTextModel, text only - the three M-RoPE
//! axes then turn alike: plain NEOX RoPE, base 1e6) from ComfyUI's GGUF (`dog`, Q4_0; the vision tower in the file is
//! not read). Both Hunyuan pipelines take hidden_states[-3] - layer 25's output, no final norm - so the last two layers
//! are not loaded. 28 layers of 3584: RMS norm (eps 1e-6) -> q (28 heads of 128), k, v (4 heads), with biases -> RoPE
//! -> causal attention -> out -> residual; RMS norm -> SwiGLU (18944) -> residual.
//!
//! The matrices stay Q4_0 on the card (expanded to half just before each product); the residual stream, attention and
//! the products' outputs are float32. The token embedding stays on the host (a row expanded a token).
//!
//! The prompt goes through its pipeline's template (`Template`) and the template's first tokens are dropped from the
//! output (crop_start), at most 1,000 tokens kept after them.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_tok::Tokenizer;

use crate::{ge, host, q4_0_row, stacked, tensor, Kern};

/// hidden_states[-3]: the output of this many layers
pub const LAYERS_USED: usize = 26;
pub const DIM: usize = 3584;
const HEADS: usize = 28;
const KV_HEADS: usize = 4;
const HEAD: usize = 128;
const FF: usize = 18944;
const EPS: f32 = 1e-6;
const THETA: f64 = 1e6;

/// How a pipeline wraps its prompt: head + prompt + tail, the first `crop` tokens dropped from the output, at most
/// `max_tokens` in all
#[derive(Clone, Debug)]
pub struct Template {
    pub head: String,
    pub tail: String,
    pub crop: usize,
    pub max_tokens: usize,
    /// an empty prompt read as one space (HunyuanVideo's apply_text_to_template)
    pub empty_as_space: bool,
}

/// A Q4_0 matrix as stored, expanded to half before each product (float32 out), and its bias
struct QLin {
    w: DevBuf,
    bias: Option<DevBuf>,
    n: usize,
    k: usize,
}

impl QLin {
    fn load(f: &Gguf, gpu: &Arc<Gpu>, names: &[String], biases: &[String]) -> Result<QLin> {
        let (raw, n, k, ty) = stacked(f, names)?;
        if ty != GType::Q4_0 {
            return Err(Error(format!("{}: {} where Q4_0 was expected", names[0], ty.name())));
        }
        let w = DevBuf::new(gpu, raw.len())?;
        w.write(0, &raw)?;
        let bias = if biases.is_empty() {
            None
        } else {
            let mut v = Vec::with_capacity(n);
            for b in biases {
                v.extend(host(f, b)?);
            }
            if v.len() != n {
                return Err(Error(format!("{}: a bias of {} for {n} rows", biases[0], v.len())));
            }
            Some(DevBuf::from_f32(gpu, &v)?)
        };
        Ok(QLin { w, bias, n, k })
    }

    /// out [m, n] float32 = x [m, k] half . W^T + bias, W expanded into `scratch`
    fn apply(&self, k: &dyn Kern, scratch: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut f32) -> Result<()> {
        k.q4_0_half(self.w.ptr(), self.n * self.k, scratch.ptr())?;
        let bias = self.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        k.linear_f32(x, m, self.k, scratch.ptr(), self.n, bias, out)
    }
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    qkv: QLin,
    o: QLin,
    gu: QLin,
    down: QLin,
}

pub struct TextEncoder {
    k: Arc<dyn Kern>,
    pub tok: Tokenizer,
    pub template: Template,
    layers: Vec<Layer>,
    /// the token embedding's Q4_0 rows, on the host
    embd: Vec<u8>,
    vocab: usize,
    scratch: DevBuf,
    pub bytes: usize,
}

impl TextEncoder {
    /// `template.crop` 0: the head's own token count (HunyuanVideo's calculate_crop_start: the tokens up to and
    /// including "<|im_start|>user\n", which ends its head)
    pub fn load(file: &Path, tokenizer: &Path, nsd: &Nsd, k: Arc<dyn Kern>, mut template: Template, log: &mut dyn FnMut(String))
                -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let gpu: &Arc<Gpu> = &nsd.gpu;
        let f = Gguf::open(file).map_err(ge)?;
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        if template.crop == 0 {
            template.crop = tok.encode(&template.head).len();
        }
        let et = tensor(&f, "token_embd.weight")?;
        if et.ty != GType::Q4_0 || et.shape.get(1) != Some(&(DIM as u64)) {
            return Err(Error(format!("{}: token_embd.weight is {} {:?}; Qwen2.5-VL-7B Q4_0 [vocab, 3584] here", file.display(), et.ty.name(), et.shape)));
        }
        let vocab = et.shape[0] as usize;
        let embd = f.read(et).map_err(ge)?;
        let mut layers = Vec::with_capacity(LAYERS_USED);
        for l in 0..LAYERS_USED {
            let n = |s: &str| format!("blk.{l}.{s}");
            layers.push(Layer {
                attn_norm: DevBuf::from_f32(gpu, &host(&f, &n("attn_norm.weight"))?)?,
                ffn_norm: DevBuf::from_f32(gpu, &host(&f, &n("ffn_norm.weight"))?)?,
                qkv: QLin::load(&f, gpu, &[n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight")], &[n("attn_q.bias"), n("attn_k.bias"), n("attn_v.bias")])?,
                o: QLin::load(&f, gpu, &[n("attn_output.weight")], &[])?,
                gu: QLin::load(&f, gpu, &[n("ffn_gate.weight"), n("ffn_up.weight")], &[])?,
                down: QLin::load(&f, gpu, &[n("ffn_down.weight")], &[])?,
            });
        }
        if layers[0].qkv.n != (HEADS + 2 * KV_HEADS) * HEAD || layers[0].gu.n != 2 * FF {
            return Err(Error(format!("{}: not Qwen2.5-VL-7B's shapes", file.display())));
        }
        let all = || layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]);
        let biggest = all().map(|m| m.n * m.k).max().unwrap_or(0);
        let bytes = all().map(|m| m.w.len + m.bias.as_ref().map_or(0, |b| b.len)).sum::<usize>() + biggest * 2;
        log(format!("text encoder: Qwen2.5-VL-7B, {LAYERS_USED} of its layers (Q4_0, {:.1} GiB) on {} in {:.1} s (crop {})", bytes as f64 / (1u64 << 30) as f64,
                    gpu.name, t0.elapsed().as_secs_f64(), template.crop));
        Ok(TextEncoder { k, tok, template, layers, embd, vocab, scratch: DevBuf::new(gpu, biggest * 2)?, bytes })
    }

    /// The template's ids for `prompt`, truncated as the pipeline's tokenizer
    pub fn ids(&self, prompt: &str) -> Vec<u32> {
        let t = &self.template;
        let p = if prompt.is_empty() && t.empty_as_space { " " } else { prompt };
        let mut ids = self.tok.encode(&format!("{}{p}{}", t.head, t.tail));
        ids.truncate(t.max_tokens);
        ids
    }

    /// The prompt's embedding for the DiT: float32 [T - crop, 3584] (hidden_states[-3], the template's head dropped)
    pub fn encode(&self, nsd: &Nsd, prompt: &str) -> Result<Vec<f32>> {
        let ids = self.ids(prompt);
        let x = self.run(nsd, &ids, LAYERS_USED)?.to_f32()?;
        Ok(x[self.template.crop.min(ids.len()) * DIM..].to_vec())
    }

    /// The hidden state after `layers` layers for every token: float32 [T, 3584] (a check reads layer 1's)
    pub fn run(&self, nsd: &Nsd, ids: &[u32], layers: usize) -> Result<DevBuf> {
        let gpu = &nsd.gpu;
        let t = ids.len();
        let k = &*self.k;
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
        let mul = |m: &QLin, x: &DevBuf, n: usize, out: &DevBuf| -> Result<()> {
            k.to_half(x.fp(), hh.ptr(), t * n, 1.0)?;
            m.apply(k, &self.scratch, hh.ptr(), t, out.fp())
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
