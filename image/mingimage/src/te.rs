//! The text side: Ming-Image's "mllm" without its vision tower - Ling-Mini-2.0 (BailingMoeV2: 20 layers of 2,048, 16
//! query / 4 key-value heads of 128, a dense first layer and 256 experts after it, 8 a token from 4 of 8 groups, one
//! shared expert), then the connector (a Qwen2 1.5B, bidirectional over the 256 query tokens) and the projections.
//! All of it stays on its card in the GGUF's types (Q4_K matrices, half / bfloat16 vectors): 9.3 GB, the second card's.
//!
//! ```text
//!   ids: the chat template, <image>, 256 <imagePatch>, </image>; the patches' embeddings are the learned query tokens
//!   layer: x += attention(rmsnorm(x)) (q, k per-head rms norms, video rope on the first 64 of 128 features, causal)
//!          x += mlp(rmsnorm(x))   (layer 0 dense; else the routed experts' sum + the shared expert; the query tokens
//!                                  routed by the image gate, the text by the text gate)
//!   cap    = proj_out(connector(proj_in(final[query tokens])))                [256, 2560]
//!   direct = linear(rmsnorm([x before layer 5 | x before layer 12 | final], 1e-5))[template tokens]   [L, 3840]
//! ```
//!
//! The experts are routed on the host (the gates' logits come back each layer, a few hundred KB): the token rows of
//! each expert gathered, its two matrices run over them, the results summed back by token in a fixed order.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_tok::Tokenizer;

use crate::ffi;

pub const HIDDEN: usize = 2048;
const HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD: usize = 128;
const ROPE: usize = 64;
const THETA: f64 = 600_000.0;
const EPS: f32 = 1e-6;
const LAYERS: usize = 20;
const DENSE_FFN: usize = 5120;
const EXPERTS: usize = 256;
const EXPERT_FFN: usize = 512;
const TOP_K: usize = 8;
const GROUPS: usize = 8;
const TOP_GROUPS: usize = 4;
const ROUTED_SCALE: f32 = 2.5;
/// the hidden states the direct condition reads: before layers 5 and 12, and the last (normed)
const CAPTURE: [usize; 2] = [5, 12];

pub const QUERY: usize = 256;
pub const CAP_DIM: usize = 2560;
pub const DIRECT_DIM: usize = 3840;
const IMAGE_START: u32 = 157158;
const IMAGE_PATCH: u32 = 157157;
const IMAGE_END: u32 = 157159;
/// processing_bailingmm2's system turn (no thinking)
const SYSTEM: &str = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off";

// the connector (Qwen2 1.5B)
const C_HIDDEN: usize = 1536;
const C_HEADS: usize = 12;
const C_KV: usize = 2;
const C_FFN: usize = 8960;
const C_LAYERS: usize = 28;
const C_THETA: f64 = 1_000_000.0;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A matrix on the card as stored, [rows, k]
struct Mat {
    w: DevBuf,
    qtype: u32,
    rows: usize,
    k: usize,
}

impl Mat {
    #[allow(clippy::too_many_arguments)]
    /// out [m, rows] (`ldo` apart) = x [m, k] half . W^T (+ bias)
    fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt, ldo: usize, bias: *const f32) -> Result<()> {
        ffi::linear_q(nsd, x, Dt::F16, m, self.k, self.w.ptr(), self.qtype, self.rows, bias.cast(), out, out_dt, ldo)
    }
}

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    qkv: Mat,
    dense: Mat,
    q_norm: DevBuf,
    k_norm: DevBuf,
    mlp: Mlp,
}

enum Mlp {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe {
        /// the gates float32 [256, 2048] and their routing biases
        gate: DevBuf,
        gate_bias: Vec<f32>,
        image_gate: DevBuf,
        image_bias: Vec<f32>,
        /// [256 experts, 1024 rows (gate | up), 2048] and [256, 2048, 512], each expert's rows contiguous
        gate_up: DevBuf,
        down: DevBuf,
        qtype_gu: u32,
        qtype_down: u32,
        expert_bytes_gu: usize,
        expert_bytes_down: usize,
        shared_gate: Mat,
        shared_up: Mat,
        shared_down: Mat,
    },
}

struct CLayer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    q: Mat,
    k: Mat,
    v: Mat,
    qb: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
}

pub struct TextEncoder {
    pub gpu: Arc<Gpu>,
    nsd: Nsd,
    pub tok: Tokenizer,
    embed: Mat,
    query: DevBuf,
    layers: Vec<Layer>,
    norm: DevBuf,
    proj_in: Mat,
    proj_in_b: DevBuf,
    clayers: Vec<CLayer>,
    cnorm: DevBuf,
    proj_out: Mat,
    proj_out_b: DevBuf,
    direct_norm: DevBuf,
    direct: Mat,
    direct_b: DevBuf,
    pub bytes: u64,
}

/// The thinker's states for a check: each layer's output, each MoE layer's experts per token (sorted), the final
#[derive(Default)]
pub struct Trace {
    pub layers: Vec<Vec<f32>>,
    pub routes: Vec<Vec<u32>>,
    pub fin: Vec<f32>,
}

/// What the DiT is given for a prompt (float32 on the host): the query tokens' features and the direct condition
pub struct Condition {
    pub cap: Vec<f32>,
    pub direct: Vec<f32>,
    pub direct_rows: usize,
}

/// A float vector (norm weights, biases, the gates) as float32, whatever its stored type
fn host_vec(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{name}: a {} vector", other.name()))),
    })
}

impl TextEncoder {
    pub fn load(path: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(path).map_err(ge)?;
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", path.display())));
        // the tokenizer: the original tokenizer.json, its bytes as float32 values in a tensor
        let tj = {
            let t = tensor("tokenizer_json")?;
            let b = f.read(t).map_err(ge)?;
            let bytes: Vec<u8> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u8).collect();
            String::from_utf8(bytes).map_err(|e| Error(format!("tokenizer_json: {e}")))?
        };
        let tok = Tokenizer::from_hf_json_text(&tj).map_err(Error)?;
        let bytes: u64 = f.tensors.iter().filter(|t| t.name != "tokenizer_json").map(|t| t.bytes).sum();
        let (_, free) = gpu.memory()?;
        if let Some(fr) = free {
            if bytes + (1u64 << 30) > fr {
                return Err(Error(format!("the text encoder needs about {:.1} GiB on {}; {:.1} GiB are free", (bytes + (1 << 30)) as f64 / (1u64 << 30) as f64,
                                          gpu.name, fr as f64 / (1u64 << 30) as f64)));
            }
        }
        let nsd = Nsd::new(gpu)?;
        let up = |b: &[u8]| -> Result<DevBuf> {
            let d = DevBuf::new(gpu, b.len())?;
            d.write(0, b)?;
            Ok(d)
        };
        let mat = |name: &str| -> Result<Mat> {
            let t = tensor(name)?;
            if t.shape.len() != 2 {
                return Err(Error(format!("{name}: {:?}, a matrix", t.shape)));
            }
            Ok(Mat { w: up(&f.read(t).map_err(ge)?)?, qtype: t.ty.code(), rows: t.shape[0] as usize, k: t.shape[1] as usize })
        };
        let vecd = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &host_vec(&f, name)?) };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("thinker.layers.{i}.");
            let mlp = if f.tensor(&format!("{p}mlp.gate_proj.weight")).is_some() {
                Mlp::Dense { gate: mat(&format!("{p}mlp.gate_proj.weight"))?, up: mat(&format!("{p}mlp.up_proj.weight"))?, down: mat(&format!("{p}mlp.down_proj.weight"))? }
            } else {
                let gu = tensor(&format!("{p}mlp.experts.gate_up_proj.weight"))?;
                let dn = tensor(&format!("{p}mlp.experts.down_proj.weight"))?;
                if gu.shape != [EXPERTS as u64, 2 * EXPERT_FFN as u64, HIDDEN as u64] || dn.shape != [EXPERTS as u64, HIDDEN as u64, EXPERT_FFN as u64] {
                    return Err(Error(format!("{p}experts: {:?} / {:?}", gu.shape, dn.shape)));
                }
                Mlp::Moe {
                    gate: vecd(&format!("{p}mlp.gate.proj.weight"))?,
                    gate_bias: host_vec(&f, &format!("{p}mlp.gate.expert_bias"))?,
                    image_gate: vecd(&format!("{p}mlp.image_gate.proj.weight"))?,
                    image_bias: host_vec(&f, &format!("{p}mlp.image_gate.expert_bias"))?,
                    gate_up: up(&f.read(gu).map_err(ge)?)?,
                    down: up(&f.read(dn).map_err(ge)?)?,
                    qtype_gu: gu.ty.code(),
                    qtype_down: dn.ty.code(),
                    expert_bytes_gu: (gu.bytes / EXPERTS as u64) as usize,
                    expert_bytes_down: (dn.bytes / EXPERTS as u64) as usize,
                    shared_gate: mat(&format!("{p}mlp.shared_experts.gate_proj.weight"))?,
                    shared_up: mat(&format!("{p}mlp.shared_experts.up_proj.weight"))?,
                    shared_down: mat(&format!("{p}mlp.shared_experts.down_proj.weight"))?,
                }
            };
            layers.push(Layer {
                in_norm: vecd(&format!("{p}input_layernorm.weight"))?,
                post_norm: vecd(&format!("{p}post_attention_layernorm.weight"))?,
                qkv: mat(&format!("{p}attention.query_key_value.weight"))?,
                dense: mat(&format!("{p}attention.dense.weight"))?,
                q_norm: vecd(&format!("{p}attention.q_norm.weight"))?,
                k_norm: vecd(&format!("{p}attention.k_norm.weight"))?,
                mlp,
            });
        }
        let mut clayers = Vec::with_capacity(C_LAYERS);
        for i in 0..C_LAYERS {
            let p = format!("connector.layers.{i}.");
            clayers.push(CLayer {
                in_norm: vecd(&format!("{p}input_layernorm.weight"))?,
                post_norm: vecd(&format!("{p}post_attention_layernorm.weight"))?,
                q: mat(&format!("{p}self_attn.q_proj.weight"))?,
                k: mat(&format!("{p}self_attn.k_proj.weight"))?,
                v: mat(&format!("{p}self_attn.v_proj.weight"))?,
                qb: vecd(&format!("{p}self_attn.q_proj.bias"))?,
                kb: vecd(&format!("{p}self_attn.k_proj.bias"))?,
                vb: vecd(&format!("{p}self_attn.v_proj.bias"))?,
                o: mat(&format!("{p}self_attn.o_proj.weight"))?,
                gate: mat(&format!("{p}mlp.gate_proj.weight"))?,
                up: mat(&format!("{p}mlp.up_proj.weight"))?,
                down: mat(&format!("{p}mlp.down_proj.weight"))?,
            });
        }
        let te = TextEncoder {
            gpu: gpu.clone(),
            embed: mat("thinker.embed_tokens.weight")?,
            query: DevBuf::from_f32(gpu, &host_vec(&f, "query_tokens")?)?,
            layers,
            norm: vecd("thinker.norm.weight")?,
            proj_in: mat("proj_in.weight")?,
            proj_in_b: vecd("proj_in.bias")?,
            clayers,
            cnorm: vecd("connector.norm.weight")?,
            proj_out: mat("proj_out.weight")?,
            proj_out_b: vecd("proj_out.bias")?,
            direct_norm: vecd("proj_directvlm.0.weight")?,
            direct: mat("proj_directvlm.1.weight")?,
            direct_b: vecd("proj_directvlm.1.bias")?,
            nsd,
            tok,
            bytes,
        };
        te.nsd.wait()?;
        log(format!("ming-image text encoder: {LAYERS} layers ({EXPERTS} experts), the {C_LAYERS}-layer connector, {:.1} GiB on {} in {:.1} s",
                    bytes as f64 / (1u64 << 30) as f64, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(te)
    }

    /// The template's tokens for `prompt`, and the whole sequence (the generated image's block after them)
    pub fn ids(&self, prompt: &str) -> (Vec<u32>, Vec<u32>) {
        let text = format!("{SYSTEM}<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>");
        let ids = self.tok.encode(&text);
        let mut all = ids.clone();
        all.push(IMAGE_START);
        all.extend(std::iter::repeat_n(IMAGE_PATCH, QUERY));
        all.push(IMAGE_END);
        (ids, all)
    }

    /// (cos, sin) per token and rotated pair, for the thinker's video rope: the template at its index on every axis,
    /// the 256 query tokens at t = h = L + 1 and w = L + 1 + (0..255) - 127, the end token after
    fn rope_table(l: usize, t: usize) -> Vec<f32> {
        let s = (l + 1) as f64;
        let inv: Vec<f64> = (0..ROPE / 2).map(|k| 1.0 / THETA.powf(2.0 * k as f64 / ROPE as f64)).collect();
        let mut cs = vec![0f32; t * ROPE];
        for i in 0..t {
            let (pt, ph, pw) = if i <= l {
                (i as f64, i as f64, i as f64)
            } else if i < l + 1 + QUERY {
                let j = (i - l - 1) as f64;
                (s, s, s + j - ((QUERY - 1) / 2) as f64)
            } else {
                let p = s + 1.0 + (i - l - 1 - QUERY) as f64;
                (p, p, p)
            };
            for (k, f) in inv.iter().enumerate() {
                let pos = if k < 24 { if k % 2 == 0 { ph } else { pw } } else { pt };
                let a = pos * f;
                cs[i * ROPE + 2 * k] = a.cos() as f32;
                cs[i * ROPE + 2 * k + 1] = a.sin() as f32;
            }
        }
        cs
    }

    /// rms norm of x float32 [m, c] (weight `w`) into half
    fn rms(&self, x: &DevBuf, m: usize, c: usize, w: &DevBuf, eps: f32, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        // SAFETY: x holds m x c floats, out as many halves, w c floats.
        ffi::check(unsafe { (k.rms_mod)(self.gpu.raw(), x.fp(), m as i64, c as i64, w.fp(), eps, std::ptr::null(), out.ptr()) }, "rms norm")
    }

    /// x float32 [m, c] += o float32 [m, c] * s (s: an MLP's ACT_SCALE undone)
    fn add(&self, x: &DevBuf, m: usize, c: usize, o: &DevBuf, s: f32) -> Result<()> {
        if s != 1.0 {
            self.nsd.scale(o.ptr(), m * c, s)?;
        }
        self.nsd.gate_add(x.ptr(), Dt::F32, m, c, o.ptr(), Dt::F32, none(), none())
    }

    /// The top 8 experts of a token from its gate scores (sigmoid): routed on scores + bias, the best 4 of 8 groups by
    /// their top two, weighted by the scores themselves, normalized and scaled by 2.5 (BailingMoeV2Gate)
    fn route(logits: &[f32], bias: &[f32]) -> ([usize; TOP_K], [f32; TOP_K]) {
        let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let r: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
        let per = EXPERTS / GROUPS;
        let mut gs: Vec<(f32, usize)> = (0..GROUPS).map(|g| {
            let mut v: Vec<f32> = r[g * per..(g + 1) * per].to_vec();
            v.sort_by(|a, b| b.total_cmp(a));
            (v[0] + v[1], g)
        }).collect();
        gs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let keep: Vec<usize> = gs[..TOP_GROUPS].iter().map(|g| g.1).collect();
        let mut cand: Vec<(f32, usize)> = (0..EXPERTS).filter(|e| keep.contains(&(e / per))).map(|e| (r[e], e)).collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut idx = [0usize; TOP_K];
        let mut w = [0f32; TOP_K];
        for j in 0..TOP_K {
            idx[j] = cand[j].1;
            w[j] = scores[cand[j].1];
        }
        let s: f32 = w.iter().sum::<f32>() + 1e-20;
        for v in &mut w {
            *v = *v / s * ROUTED_SCALE;
        }
        (idx, w)
    }

    /// The prompt's condition for the DiT
    pub fn encode(&self, prompt: &str) -> Result<Condition> {
        let (ids, all) = self.ids(prompt);
        self.encode_ids(&ids, &all, None)
    }

    /// ... from its token ids; `dumps`: the thinker's final hidden state kept (a check)
    pub fn encode_ids(&self, ids: &[u32], all: &[u32], mut trace: Option<&mut Trace>) -> Result<Condition> {
        let gpu = &self.gpu;
        let nsd = &self.nsd;
        let k = ffi::api()?;
        let l = ids.len();
        let t = all.len();
        let qs = l + 1;
        // the embeddings: each token's row dequantized; the query tokens' rows the learned ones
        let x = DevBuf::f32(gpu, t * HIDDEN)?;
        let row_bytes = self.embed.w.len / self.embed.rows;
        for (i, id) in all.iter().enumerate() {
            if (qs..qs + QUERY).contains(&i) {
                continue;
            }
            let id = (*id as usize).min(self.embed.rows - 1);
            let src: *const std::ffi::c_void = self.embed.w.ptr().cast::<u8>().wrapping_add(id * row_bytes).cast();
            nsd.dequant(src, self.embed.qtype, HIDDEN, x.fp().wrapping_add(i * HIDDEN).cast(), Dt::F32)?;
        }
        x.copy_within(qs * HIDDEN * 4, &self.query, 0, QUERY * HIDDEN * 4)?;
        let cs = DevBuf::from_f32(gpu, &Self::rope_table(l, t))?;
        let h = DevBuf::new(gpu, t * HIDDEN * 2)?;
        let hf = DevBuf::f32(gpu, t * HIDDEN)?;
        let qkv = DevBuf::new(gpu, t * (HEADS + 2 * KV_HEADS) * HEAD * 2)?;
        let att = DevBuf::new(gpu, t * HIDDEN * 2)?;
        let o = DevBuf::f32(gpu, t * HIDDEN)?;
        let gu = DevBuf::new(gpu, t * 2 * DENSE_FFN * 2)?;
        let act = DevBuf::new(gpu, t * DENSE_FFN * 2)?;
        let qkv_w = (HEADS + 2 * KV_HEADS) * HEAD;
        let half_at = |b: &DevBuf, n: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(n).cast() };
        let mut captured: Vec<DevBuf> = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            if CAPTURE.contains(&li) {
                let c = DevBuf::f32(gpu, l * HIDDEN)?;
                c.copy_within(0, &x, 0, l * HIDDEN * 4)?;
                captured.push(c);
            }
            self.rms(&x, t, HIDDEN, &layer.in_norm, EPS, &h)?;
            layer.qkv.apply(nsd, h.ptr(), t, qkv.ptr(), Dt::F16, 0, std::ptr::null())?;
            nsd.rms_rope(qkv.ptr(), Dt::F16, t, HEADS, HEAD, qkv_w, layer.q_norm.ptr(), EPS, cs.ptr(), ROPE)?;
            nsd.rms_rope(half_at(&qkv, HEADS * HEAD), Dt::F16, t, KV_HEADS, HEAD, qkv_w, layer.k_norm.ptr(), EPS, cs.ptr(), ROPE)?;
            nsd.attention_causal(qkv.ptr(), half_at(&qkv, HEADS * HEAD), half_at(&qkv, (HEADS + KV_HEADS) * HEAD), Dt::F16, t, HEADS, KV_HEADS, HEAD, qkv_w,
                                 qkv_w, att.ptr())?;
            layer.dense.apply(nsd, att.ptr(), t, o.ptr(), Dt::F32, 0, std::ptr::null())?;
            self.add(&x, t, HIDDEN, &o, 1.0)?;
            self.rms(&x, t, HIDDEN, &layer.post_norm, EPS, &h)?;
            match &layer.mlp {
                Mlp::Dense { gate, up, down } => {
                    gate.apply(nsd, h.ptr(), t, gu.ptr(), Dt::F16, 2 * DENSE_FFN, std::ptr::null())?;
                    up.apply(nsd, h.ptr(), t, half_at(&gu, DENSE_FFN), Dt::F16, 2 * DENSE_FFN, std::ptr::null())?;
                    ffi::swiglu(gpu, gu.ptr(), t, DENSE_FFN, act.ptr())?;
                    down.apply(nsd, act.ptr(), t, o.ptr(), Dt::F32, 0, std::ptr::null())?;
                    self.add(&x, t, HIDDEN, &o, 1.0 / ffi::ACT_SCALE)?;
                }
                Mlp::Moe { gate, gate_bias, image_gate, image_bias, gate_up, down, qtype_gu, qtype_down, expert_bytes_gu, expert_bytes_down,
                           shared_gate, shared_up, shared_down } => {
                    // the gates in float32 on the norm in float32 (as BailingMoeV2Gate casts it)
                    nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, layer.post_norm.fp().cast(), EPS, none(), none(), none(), hf.ptr(), Dt::F32)?;
                    let lg = DevBuf::f32(gpu, 2 * t * EXPERTS)?;
                    nsd.linear(hf.ptr(), Dt::F32, t, HIDDEN, gate.ptr(), EXPERTS, none(), lg.ptr(), Dt::F32)?;
                    nsd.linear(hf.ptr(), Dt::F32, t, HIDDEN, image_gate.ptr(), EXPERTS, none(), lg.fp().wrapping_add(t * EXPERTS).cast(), Dt::F32)?;
                    let lv = lg.to_f32()?;
                    // each token's experts; then the (token, slot) pairs sorted by expert
                    let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(t * TOP_K); // (expert, token, weight)
                    let mut routes = Vec::with_capacity(t * TOP_K);
                    for tok in 0..t {
                        let image = (qs..qs + QUERY).contains(&tok);
                        let (lgs, b) = if image { (&lv[t * EXPERTS + tok * EXPERTS..t * EXPERTS + (tok + 1) * EXPERTS], image_bias) } else { (&lv[tok * EXPERTS..(tok + 1) * EXPERTS], gate_bias) };
                        let (idx, w) = Self::route(lgs, b);
                        let mut sorted = idx;
                        sorted.sort();
                        routes.extend(sorted.iter().map(|e| *e as u32));
                        for j in 0..TOP_K {
                            pairs.push((idx[j], tok, w[j]));
                        }
                    }
                    pairs.sort_by_key(|p| (p.0, p.1));
                    let n = pairs.len();
                    let rows: Vec<i32> = pairs.iter().map(|p| p.1 as i32).collect();
                    let mut slot = vec![-1i32; t * TOP_K];
                    let mut wts = vec![0f32; t * TOP_K];
                    let mut fill = vec![0usize; t];
                    for (s, p) in pairs.iter().enumerate() {
                        let j = fill[p.1];
                        slot[p.1 * TOP_K + j] = s as i32;
                        // the experts' products are made at ACT_SCALE: their weights undo it
                        wts[p.1 * TOP_K + j] = p.2 / ffi::ACT_SCALE;
                        fill[p.1] += 1;
                    }
                    let i32b = |v: &[i32]| -> Result<DevBuf> {
                        let b = DevBuf::new(gpu, v.len() * 4)?;
                        b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                        Ok(b)
                    };
                    let (rows_d, slot_d, w_d) = (i32b(&rows)?, i32b(&slot)?, DevBuf::from_f32(gpu, &wts)?);
                    let hs = DevBuf::new(gpu, n * HIDDEN * 2)?;
                    // SAFETY: h holds t x 2048 halves; rows n indices below t; hs n x 2048 halves.
                    ffi::check(unsafe { (k.gather_rows)(gpu.raw(), h.ptr(), HIDDEN as i64, rows_d.ptr().cast(), n as i64, hs.ptr()) }, "gather")?;
                    let eg = DevBuf::new(gpu, n * 2 * EXPERT_FFN * 2)?;
                    let ea = DevBuf::new(gpu, n * EXPERT_FFN * 2)?;
                    let ys = DevBuf::f32(gpu, n * HIDDEN)?;
                    let mut s0 = 0;
                    while s0 < n {
                        let e = pairs[s0].0;
                        let mut s1 = s0;
                        while s1 < n && pairs[s1].0 == e {
                            s1 += 1;
                        }
                        let m = s1 - s0;
                        let wgu: *const std::ffi::c_void = gate_up.ptr().cast::<u8>().wrapping_add(e * expert_bytes_gu).cast();
                        let wdn: *const std::ffi::c_void = down.ptr().cast::<u8>().wrapping_add(e * expert_bytes_down).cast();
                        ffi::linear_q(nsd, half_at(&hs, s0 * HIDDEN), Dt::F16, m, HIDDEN, wgu, *qtype_gu, 2 * EXPERT_FFN, none(), half_at(&eg, s0 * 2 * EXPERT_FFN), Dt::F16, 0)?;
                        ffi::swiglu(gpu, half_at(&eg, s0 * 2 * EXPERT_FFN), m, EXPERT_FFN, half_at(&ea, s0 * EXPERT_FFN))?;
                        ffi::linear_q(nsd, half_at(&ea, s0 * EXPERT_FFN), Dt::F16, m, EXPERT_FFN, wdn, *qtype_down, HIDDEN, none(), ys.fp().wrapping_add(s0 * HIDDEN).cast(), Dt::F32, 0)?;
                        s0 = s1;
                    }
                    // SAFETY: x t x 2048 floats; ys n x 2048 floats; slot / w t x 8.
                    ffi::check(unsafe { (k.combine)(gpu.raw(), x.fp(), t as i64, HIDDEN as i64, ys.ptr(), slot_d.ptr().cast(), w_d.fp(), TOP_K as i64) }, "combine")?;
                    // the shared expert
                    shared_gate.apply(nsd, h.ptr(), t, gu.ptr(), Dt::F16, 2 * EXPERT_FFN, std::ptr::null())?;
                    shared_up.apply(nsd, h.ptr(), t, half_at(&gu, EXPERT_FFN), Dt::F16, 2 * EXPERT_FFN, std::ptr::null())?;
                    ffi::swiglu(gpu, gu.ptr(), t, EXPERT_FFN, act.ptr())?;
                    shared_down.apply(nsd, act.ptr(), t, o.ptr(), Dt::F32, 0, std::ptr::null())?;
                    self.add(&x, t, HIDDEN, &o, 1.0 / ffi::ACT_SCALE)?;
                    // the routing's buffers are freed on return: the work on them first
                    nsd.wait()?;
                    if let Some(tr) = trace.as_deref_mut() {
                        tr.routes.push(routes);
                    }
                }
            }
            if let Some(tr) = trace.as_deref_mut() {
                nsd.wait()?;
                tr.layers.push(x.to_f32()?);
            }
        }
        // the final norm, float32
        self.rms(&x, t, HIDDEN, &self.norm, EPS, &h)?;
        let fin = DevBuf::f32(gpu, t * HIDDEN)?;
        // SAFETY: t x 2048 values each way.
        ffi::check(unsafe { (k.to_float)(gpu.raw(), h.ptr(), fin.fp(), (t * HIDDEN) as i64) }, "to float")?;
        if let Some(tr) = trace {
            tr.fin = fin.to_f32()?;
        }
        // the direct condition: [before 5 | before 12 | final] over the template, rms norm (1e-5), linear
        let dw = HIDDEN * (CAPTURE.len() + 1);
        let cat = DevBuf::f32(gpu, l * dw)?;
        for (j, c) in captured.iter().chain(std::iter::once(&fin)).enumerate() {
            ffi::copy_2d(gpu, cat.ptr().cast::<u8>().wrapping_add(j * HIDDEN * 4).cast(), dw * 4, c.ptr(), HIDDEN * 4, HIDDEN * 4, l)?;
        }
        let dh = DevBuf::new(gpu, l * dw * 2)?;
        self.rms(&cat, l, dw, &self.direct_norm, 1e-5, &dh)?;
        let direct = DevBuf::f32(gpu, l * DIRECT_DIM)?;
        self.direct.apply(nsd, dh.ptr(), l, direct.ptr(), Dt::F32, 0, self.direct_b.fp())?;
        // the query tokens through proj_in, the connector, proj_out
        let qh = DevBuf::new(gpu, QUERY * HIDDEN * 2)?;
        // SAFETY: the query rows of fin, QUERY x 2048 values.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), fin.fp().wrapping_add(qs * HIDDEN), qh.ptr(), (QUERY * HIDDEN) as i64) }, "to half")?;
        let cx = DevBuf::f32(gpu, QUERY * C_HIDDEN)?;
        self.proj_in.apply(nsd, qh.ptr(), QUERY, cx.ptr(), Dt::F32, 0, self.proj_in_b.fp())?;
        let cx = self.connector(cx)?;
        let ch = DevBuf::new(gpu, QUERY * C_HIDDEN * 2)?;
        self.rms(&cx, QUERY, C_HIDDEN, &self.cnorm, EPS, &ch)?;
        let cap = DevBuf::f32(gpu, QUERY * CAP_DIM)?;
        self.proj_out.apply(nsd, ch.ptr(), QUERY, cap.ptr(), Dt::F32, 0, self.proj_out_b.fp())?;
        nsd.wait()?;
        Ok(Condition { cap: cap.to_f32()?, direct: direct.to_f32()?, direct_rows: l })
    }

    /// The connector's 28 Qwen2 layers over x float32 [256, 1536], bidirectional (the pipeline's all-ones mask)
    fn connector(&self, x: DevBuf) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let nsd = &self.nsd;
        let k = ffi::api()?;
        let n = QUERY;
        let hd = HEAD;
        let qw = (C_HEADS + 2 * C_KV) * hd;
        // rope over all 128 features, pairs (i, 64 + i), positions 0..255
        let mut cs = vec![0f32; n * hd];
        for p in 0..n {
            for i in 0..hd / 2 {
                let a = p as f64 / C_THETA.powf(2.0 * i as f64 / hd as f64);
                cs[p * hd + 2 * i] = a.cos() as f32;
                cs[p * hd + 2 * i + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let h = DevBuf::new(gpu, n * C_HIDDEN * 2)?;
        let qkv = DevBuf::new(gpu, n * qw * 2)?;
        let kf = DevBuf::new(gpu, n * C_HEADS * hd * 2)?;
        let vf = DevBuf::new(gpu, n * C_HEADS * hd * 2)?;
        let att = DevBuf::new(gpu, n * C_HIDDEN * 2)?;
        let o = DevBuf::f32(gpu, n * C_HIDDEN)?;
        let gu = DevBuf::new(gpu, n * 2 * C_FFN * 2)?;
        let act = DevBuf::new(gpu, n * C_FFN * 2)?;
        let half_at = |b: &DevBuf, at: usize| -> *mut std::ffi::c_void { b.ptr().cast::<u16>().wrapping_add(at).cast() };
        for c in &self.clayers {
            self.rms(&x, n, C_HIDDEN, &c.in_norm, EPS, &h)?;
            c.q.apply(nsd, h.ptr(), n, qkv.ptr(), Dt::F16, qw, c.qb.fp())?;
            c.k.apply(nsd, h.ptr(), n, half_at(&qkv, C_HEADS * hd), Dt::F16, qw, c.kb.fp())?;
            c.v.apply(nsd, h.ptr(), n, half_at(&qkv, (C_HEADS + C_KV) * hd), Dt::F16, qw, c.vb.fp())?;
            nsd.rope(qkv.ptr(), Dt::F16, n, C_HEADS, hd, qw, cs.ptr(), hd)?;
            nsd.rope(half_at(&qkv, C_HEADS * hd), Dt::F16, n, C_KV, hd, qw, cs.ptr(), hd)?;
            // SAFETY: qkv holds n rows of qw halves; kf / vf n x 12 x 128.
            ffi::check(unsafe { (k.repeat_heads)(gpu.raw(), qkv.ptr(), n as i64, qw as i64, (C_HEADS * hd) as i64, C_KV as i64, C_HEADS as i64, hd as i64, kf.ptr()) }, "heads")?;
            ffi::check(unsafe { (k.repeat_heads)(gpu.raw(), qkv.ptr(), n as i64, qw as i64, ((C_HEADS + C_KV) * hd) as i64, C_KV as i64, C_HEADS as i64, hd as i64, vf.ptr()) }, "heads")?;
            nsd.attention_qk(qkv.ptr(), n, qw, kf.ptr(), vf.ptr(), n, C_HEADS * hd, Dt::F16, C_HEADS, hd, att.ptr(), Dt::F16)?;
            c.o.apply(nsd, att.ptr(), n, o.ptr(), Dt::F32, 0, std::ptr::null())?;
            self.add(&x, n, C_HIDDEN, &o, 1.0)?;
            self.rms(&x, n, C_HIDDEN, &c.post_norm, EPS, &h)?;
            c.gate.apply(nsd, h.ptr(), n, gu.ptr(), Dt::F16, 2 * C_FFN, std::ptr::null())?;
            c.up.apply(nsd, h.ptr(), n, half_at(&gu, C_FFN), Dt::F16, 2 * C_FFN, std::ptr::null())?;
            ffi::swiglu(gpu, gu.ptr(), n, C_FFN, act.ptr())?;
            c.down.apply(nsd, act.ptr(), n, o.ptr(), Dt::F32, 0, std::ptr::null())?;
            self.add(&x, n, C_HIDDEN, &o, 1.0 / ffi::ACT_SCALE)?;
        }
        nsd.wait()?;
        Ok(x)
    }
}
