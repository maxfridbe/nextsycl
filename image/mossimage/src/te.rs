//! The text encoder: Qwen3.5-9B's text model (`Qwen3_5Model`, the transformers class the pipeline reads) as llama.cpp's
//! `qwen35` GGUF holds it: 32 layers of 4,096 features, three gated-DeltaNet layers then one gated full-attention layer,
//! each followed by a SwiGLU FFN. The pipeline takes the final norm's output of every token (no LM head).
//!
//! ```text
//!   per layer:  h = attn_norm(x);  x += linear(h) or full(h);  x += ffn(post_attention_norm(x))
//!   out = output_norm(x)                                               (the norms' weights carry their +1)
//!   full:    q | gate = attn_q(h) [16 heads x (256 + 256)], k, v (4 heads x 256); per-head RMSNorm + RoPE (the first 64
//!            features, theta 1e7, rotate-half) on q and k; causal attention; x sigmoid(gate); attn_output
//!   linear:  qkv = attn_qkv(h) -> conv (4 taps) + SiLU -> q, k (16 heads x 128), v (32 x 128); q, k L2-normed;
//!            the delta rule over the rows; RMSNorm(o) * w * silu(attn_gate(h)); ssm_out
//! ```
//!
//! Text positions are one number for all of M-RoPE's three axes, so the interleaved sections turn every pair by the
//! row's position and plain RoPE is the same. Activations on the residual stream and in the recurrences are float32;
//! the products take half rows (the stored Q8_0 blocks read by the shared quantized linear).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};
use nextsycl_tok::Tokenizer;

use crate::ffi;
use crate::part::{ge, vecd, Part};

pub const HIDDEN: usize = 4096;

/// The layer geometry (the GGUF's `qwen35.*` keys)
struct Geometry {
    layers: usize,
    ffn: usize,
    heads: usize,
    heads_kv: usize,
    head: usize,
    n_rot: usize,
    theta: f32,
    eps: f32,
    conv: usize,
    state: usize,
    k_heads: usize,
    v_heads: usize,
    inner: usize,
    interval: usize,
}

impl Geometry {
    fn read(f: &Gguf) -> Result<Geometry> {
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("no {}.{k}", f.architecture())));
        let fl = |k: &str| f.arch_meta(k).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| Error(format!("no {}.{k}", f.architecture())));
        let g = Geometry {
            layers: u("block_count")?,
            ffn: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            heads_kv: u("attention.head_count_kv")?,
            head: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            theta: fl("rope.freq_base")?,
            eps: fl("attention.layer_norm_rms_epsilon")?,
            conv: u("ssm.conv_kernel")?,
            state: u("ssm.state_size")?,
            k_heads: u("ssm.group_count")?,
            v_heads: u("ssm.time_step_rank")?,
            inner: u("ssm.inner_size")?,
            interval: u("full_attention_interval").unwrap_or(4),
        };
        if u("embedding_length")? != HIDDEN || g.state != 128 || g.inner != g.v_heads * g.state || !g.v_heads.is_multiple_of(g.k_heads) || g.conv > 8 {
            return Err(Error(format!("{}: not the Qwen3.5-9B text model this engine reads (hidden {}, DeltaNet head {}, {} key / {} value heads)", f.paths[0].display(),
                                     u("embedding_length")?, g.state, g.k_heads, g.v_heads)));
        }
        if !g.head.is_multiple_of(16) || !g.heads.is_multiple_of(g.heads_kv) || g.n_rot > g.head {
            return Err(Error(format!("{}: attention heads of {} ({} / {}, {} rotary)", f.paths[0].display(), g.head, g.heads, g.heads_kv, g.n_rot)));
        }
        Ok(g)
    }

    fn full(&self, l: usize) -> bool {
        (l + 1).is_multiple_of(self.interval)
    }

    /// conv channels: q, k (key heads) and v (value heads)
    fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.state + self.inner
    }
}

struct Full {
    q: Part,
    k: Part,
    v: Part,
    o: Part,
    q_norm: DevBuf,
    k_norm: DevBuf,
}

struct Linear {
    qkv: Part,
    z: Part,
    beta: Part,
    alpha: Part,
    conv: DevBuf,
    dt: DevBuf,
    a: DevBuf,
    norm: DevBuf,
    out: Part,
}

struct Layer {
    attn_norm: DevBuf,
    post_norm: DevBuf,
    full: Option<Full>,
    linear: Option<Linear>,
    gate: Part,
    up: Part,
    down: Part,
}

pub struct TextEncoder {
    pub gpu: Arc<Gpu>,
    nsd: Nsd,
    g: Geometry,
    file: Gguf,
    tok: Tokenizer,
    layers: Vec<Layer>,
    output_norm: DevBuf,
    pub bytes: u64,
}

/// Intermediates kept for a check: the residual stream after each layer
#[derive(Default)]
pub struct Trace {
    pub layers: Vec<Vec<f32>>,
}

/// Q8_0 row bytes as float32 (blocks of an f16 scale and 32 signed bytes)
fn q8_0_row(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(34).flat_map(|blk| {
        let d = nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        blk[2..].iter().map(move |q| *q as i8 as f32 * d)
    }).collect()
}

impl TextEncoder {
    pub fn load(path: &std::path::Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        let t0 = std::time::Instant::now();
        let f = Gguf::open(path).map_err(ge)?;
        if f.architecture() != "qwen35" {
            return Err(Error(format!("{}: a {} model, not the qwen35 text encoder", path.display(), f.architecture())));
        }
        let g = Geometry::read(&f)?;
        let tok = Tokenizer::from_gguf(&f).map_err(Error)?;
        let part = |name: &str| Part::load(&f, gpu, name);
        let vd = |name: &str| vecd(&f, gpu, name);
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            let full = if g.full(l) {
                Some(Full { q: part(&n("attn_q"))?, k: part(&n("attn_k"))?, v: part(&n("attn_v"))?, o: part(&n("attn_output"))?, q_norm: vd(&n("attn_q_norm"))?,
                            k_norm: vd(&n("attn_k_norm"))? })
            } else {
                None
            };
            let linear = if g.full(l) {
                None
            } else {
                Some(Linear {
                    qkv: part(&n("attn_qkv"))?,
                    z: part(&n("attn_gate"))?,
                    beta: part(&n("ssm_beta"))?,
                    alpha: part(&n("ssm_alpha"))?,
                    conv: vd(&n("ssm_conv1d"))?,
                    dt: vd(&format!("blk.{l}.ssm_dt.bias"))?,
                    a: vd(&format!("blk.{l}.ssm_a"))?,
                    norm: vd(&n("ssm_norm"))?,
                    out: part(&n("ssm_out"))?,
                })
            };
            layers.push(Layer { attn_norm: vd(&n("attn_norm"))?, post_norm: vd(&n("post_attention_norm"))?, full, linear, gate: part(&n("ffn_gate"))?, up: part(&n("ffn_up"))?,
                                down: part(&n("ffn_down"))? });
            if l % 8 == 7 {
                log(format!("moss-image text encoder: layer {} of {} ({:.0} s)", l + 1, g.layers, t0.elapsed().as_secs_f64()));
            }
        }
        let output_norm = vd("output_norm.weight")?;
        let nsd = Nsd::new(gpu)?;
        let bytes = f.tensors.iter().filter(|t| t.name != "output.weight").map(|t| t.bytes).sum();
        log(format!("moss-image text encoder: {} layers on {} in {:.1} s", g.layers, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(TextEncoder { gpu: gpu.clone(), nsd, g, file: f, tok, layers, output_norm, bytes })
    }

    /// The prompt as the pipeline feeds it: the chat template (one user turn, the generation prompt, thinking on)
    pub fn template(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    pub fn ids(&self, prompt: &str) -> Vec<u32> {
        self.tokenize_templated(&Self::template(prompt))
    }

    /// An already templated prompt (the reference's prompt.txt) as token ids
    pub fn tokenize_templated(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// The token rows of the embedding table as float32 [ids, 4096]
    fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = self.file.tensor("token_embd.weight").ok_or_else(|| Error("no token_embd.weight".into()))?;
        let vocab = t.shape[0] as usize;
        let row = t.ty.bytes(HIDDEN as u64).ok_or_else(|| Error("token_embd.weight: a row is not whole blocks".into()))? as usize;
        let mut out = Vec::with_capacity(ids.len() * HIDDEN);
        let mut buf = vec![0u8; row];
        for id in ids {
            if *id as usize >= vocab {
                return Err(Error(format!("token {id} is past the {vocab} the table has")));
            }
            self.file.read_into(t, *id as u64 * row as u64, &mut buf).map_err(ge)?;
            out.extend(match t.ty {
                GType::Q8_0 => q8_0_row(&buf),
                GType::F32 => buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                GType::BF16 => buf.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                GType::F16 => buf.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                other => return Err(Error(format!("token_embd.weight: {} rows are not read yet", other.name()))),
            });
        }
        Ok(out)
    }

    /// The final norm's output of every token: float32 [ids, 4096]
    pub fn encode_ids(&self, ids: &[u32], mut trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        let (gpu, nsd, g) = (&self.gpu, &self.nsd, &self.g);
        let t = ids.len();
        if t == 0 || t > 4096 {
            return Err(Error(format!("a prompt of {t} tokens: from 1 to 4096")));
        }
        let x = DevBuf::from_f32(gpu, &self.embed(ids)?)?;
        let s = Scratch::new(gpu, g, t)?;
        // NS_MOSS_PROFILE: a GPU wait after each part of a layer, the totals (full attention, DeltaNet, the FFN) on stderr
        let profile = std::env::var_os("NS_MOSS_PROFILE").is_some();
        let mut spent = [0f64; 3];
        let mut lap = std::time::Instant::now();
        let mut mark = |nsd: &Nsd, part: usize| -> Result<()> {
            if profile {
                nsd.wait()?;
                spent[part] += lap.elapsed().as_secs_f64();
                lap = std::time::Instant::now();
            }
            Ok(())
        };
        for (l, ly) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, ly.attn_norm.ptr(), g.eps, none(), none(), none(), s.h.ptr(), Dt::F16)?;
            mark(nsd, 2)?;
            match (&ly.full, &ly.linear) {
                (Some(a), _) => {
                    self.full(a, &s, t)?;
                    mark(nsd, 0)?;
                }
                (_, Some(a)) => {
                    self.linear(a, &s, t)?;
                    mark(nsd, 1)?;
                }
                _ => return Err(Error(format!("layer {l}: neither attention nor DeltaNet"))),
            }
            nsd.gate_add(x.ptr(), Dt::F32, t, HIDDEN, s.of.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, ly.post_norm.ptr(), g.eps, none(), none(), none(), s.h.ptr(), Dt::F16)?;
            ly.gate.apply(nsd, s.h.ptr(), t, None, s.gate.ptr(), Dt::F32, 0)?;
            ly.up.apply(nsd, s.h.ptr(), t, None, s.up.ptr(), Dt::F32, 0)?;
            let k = ffi::api()?;
            // SAFETY: t x ffn floats each, the activation as many halves.
            ffi::check(unsafe { (k.silu_mul)(gpu.raw(), s.gate.fp(), s.up.fp(), s.act.ptr(), (t * g.ffn) as i64) }, "silu mul")?;
            ly.down.apply(nsd, s.act.ptr(), t, None, s.of.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, HIDDEN, s.of.ptr(), Dt::F32, none(), none())?;
            mark(nsd, 2)?;
            if let Some(tr) = trace.as_deref_mut() {
                nsd.wait()?;
                tr.layers.push(x.to_f32()?);
            }
        }
        if profile {
            eprintln!("moss-image text encoder, {t} tokens: full attention {:.3} s, DeltaNet {:.3} s, FFN and norms {:.3} s", spent[0], spent[1], spent[2]);
        }
        let out = DevBuf::f32(gpu, t * HIDDEN)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, t, HIDDEN, self.output_norm.ptr(), g.eps, none(), none(), none(), out.ptr(), Dt::F32)?;
        nsd.wait()?;
        out.to_f32()
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        self.encode_ids(&self.ids(prompt), None)
    }

    /// A gated full-attention layer on `s.h` (half [t, 4096]): its output in `s.of` (float32 [t, 4096])
    fn full(&self, a: &Full, s: &Scratch, t: usize) -> Result<()> {
        let (gpu, nsd, g) = (&self.gpu, &self.nsd, &self.g);
        let k = ffi::api()?;
        let (hq, hk, d) = (g.heads, g.heads_kv, g.head);
        a.q.apply(nsd, s.h.ptr(), t, None, s.qf.ptr(), Dt::F32, 0)?;
        a.k.apply(nsd, s.h.ptr(), t, None, s.kr.ptr(), Dt::F32, 0)?;
        a.v.apply(nsd, s.h.ptr(), t, None, s.vr.ptr(), Dt::F32, 0)?;
        // SAFETY: q's rows are [hq x (d + d gate)] floats, k's [hk x d]; the outputs [t, heads x d].
        unsafe {
            ffi::check((k.qk_norm_rope)(gpu.raw(), s.qf.fp(), (hq * 2 * d) as i64, (2 * d) as i64, s.q.fp(), t as i64, hq as i64, d as i64, a.q_norm.fp(), g.eps,
                                        g.n_rot as i64, g.theta), "q norm rope")?;
            ffi::check((k.qk_norm_rope)(gpu.raw(), s.kr.fp(), (hk * d) as i64, d as i64, s.kn.fp(), t as i64, hk as i64, d as i64, a.k_norm.fp(), g.eps, g.n_rot as i64,
                                        g.theta), "k norm rope")?;
        }
        match ffi::silo()?.filter(|sl| unsafe { (sl.attn_supported)(gpu.raw(), t as i64, hq as i64, hk as i64, d as i64) } == 1) {
            // SAFETY: q [t, hq x d], k and v [t, hk x d] floats, the output [t, hq x d].
            Some(sl) => ffi::check(unsafe { (sl.attn_causal)(gpu.raw(), s.q.ptr(), s.kn.ptr(), s.vr.ptr(), t as i64, hq as i64, hk as i64, d as i64, (hq * d) as i64, (hk * d) as i64, s.att.fp()) }, "causal attention")?,
            None => nsd.attention_causal(s.q.ptr(), s.kn.ptr(), s.vr.ptr(), Dt::F32, t, hq, hk, d, hq * d, hk * d, s.att.ptr())?,
        }
        // SAFETY: att [t, hq x d] gated by qf's gate columns.
        ffi::check(unsafe { (k.gate_mul)(gpu.raw(), s.att.fp(), s.qf.fp(), (hq * 2 * d) as i64, t as i64, hq as i64, d as i64) }, "output gate")?;
        // SAFETY: t x 4096 floats in, as many halves out.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), s.att.fp(), s.h.ptr(), (t * hq * d) as i64) }, "to half")?;
        a.o.apply(nsd, s.h.ptr(), t, None, s.of.ptr(), Dt::F32, 0)
    }

    /// A gated-DeltaNet layer on `s.h`: its output in `s.of`
    fn linear(&self, a: &Linear, s: &Scratch, t: usize) -> Result<()> {
        let (gpu, nsd, g) = (&self.gpu, &self.nsd, &self.g);
        let k = ffi::api()?;
        let (hv, hk, d, cd) = (g.v_heads, g.k_heads, g.state, g.conv_dim());
        let qd = hk * d;
        a.qkv.apply(nsd, s.h.ptr(), t, None, s.qkv.ptr(), Dt::F32, 0)?;
        a.z.apply(nsd, s.h.ptr(), t, None, s.z.ptr(), Dt::F32, 0)?;
        a.beta.apply(nsd, s.h.ptr(), t, None, s.beta.ptr(), Dt::F32, 0)?;
        a.alpha.apply(nsd, s.h.ptr(), t, None, s.alpha.ptr(), Dt::F32, 0)?;
        // the layer's state starts empty: the recurrence's [hv][d][d] and the conv's last inputs
        let state = DevBuf::from_f32(gpu, &vec![0f32; hv * d * d])?;
        let conv_state = DevBuf::from_f32(gpu, &vec![0f32; (g.conv - 1) * cd])?;
        // SAFETY: the buffers as named in mo.h.
        unsafe {
            ffi::check((k.gdn_gates)(gpu.raw(), s.alpha.fp(), s.beta.fp(), a.dt.fp(), a.a.fp(), s.eg.fp(), t as i64, hv as i64, d as i64), "gates")?;
            ffi::check((k.conv_silu)(gpu.raw(), s.qkv.fp(), conv_state.fp(), a.conv.fp(), s.co.fp(), t as i64, cd as i64, g.conv as i32), "conv silu")?;
            ffi::check((k.expand)(gpu.raw(), s.co.fp(), cd as i64, s.lq.fp(), t as i64, hk as i64, hv as i64, d as i64), "expand q")?;
            ffi::check((k.expand)(gpu.raw(), s.co.fp().add(qd), cd as i64, s.lk.fp(), t as i64, hk as i64, hv as i64, d as i64), "expand k")?;
            ffi::check((k.expand)(gpu.raw(), s.co.fp().add(2 * qd), cd as i64, s.lv.fp(), t as i64, hv as i64, hv as i64, d as i64), "expand v")?;
            ffi::check((k.l2_norm)(gpu.raw(), s.lq.fp(), (t * hv) as i64, d as i64, g.eps), "l2 q")?;
            ffi::check((k.l2_norm)(gpu.raw(), s.lk.fp(), (t * hv) as i64, d as i64, g.eps), "l2 k")?;
            ffi::check((k.kda_scan)(gpu.raw(), s.lq.fp(), s.lk.fp(), s.lv.fp(), s.eg.fp(), s.beta.fp(), state.fp(), s.lo.fp(), t as i64, hv as i64), "delta rule")?;
            ffi::check((k.gdn_out)(gpu.raw(), s.lo.fp(), s.z.fp(), g.inner as i64, a.norm.fp(), s.att.fp(), t as i64, hv as i64, d as i64, g.eps), "gated norm")?;
            ffi::check((k.to_half)(gpu.raw(), s.att.fp(), s.h.ptr(), (t * g.inner) as i64), "to half")?;
        }
        a.out.apply(nsd, s.h.ptr(), t, None, s.of.ptr(), Dt::F32, 0)?;
        // the recurrence and conv state are read by queued kernels: kept until they have run
        nsd.wait()
    }
}

/// A pass's buffers, for `t` tokens
struct Scratch {
    h: DevBuf,
    of: DevBuf,
    gate: DevBuf,
    up: DevBuf,
    act: DevBuf,
    // full attention
    qf: DevBuf,
    kr: DevBuf,
    vr: DevBuf,
    q: DevBuf,
    kn: DevBuf,
    att: DevBuf,
    // DeltaNet
    qkv: DevBuf,
    z: DevBuf,
    beta: DevBuf,
    alpha: DevBuf,
    eg: DevBuf,
    co: DevBuf,
    lq: DevBuf,
    lk: DevBuf,
    lv: DevBuf,
    lo: DevBuf,
}

impl Scratch {
    fn new(gpu: &Arc<Gpu>, g: &Geometry, t: usize) -> Result<Scratch> {
        let (hq, hk, d, hv, cd) = (g.heads, g.heads_kv, g.head, g.v_heads, g.conv_dim());
        let f = |n: usize| DevBuf::f32(gpu, n);
        Ok(Scratch {
            h: DevBuf::new(gpu, t * HIDDEN.max(g.inner).max(hq * d) * 2)?,
            of: f(t * HIDDEN)?,
            gate: f(t * g.ffn)?,
            up: f(t * g.ffn)?,
            act: DevBuf::new(gpu, t * g.ffn * 2)?,
            qf: f(t * hq * 2 * d)?,
            kr: f(t * hk * d)?,
            vr: f(t * hk * d)?,
            q: f(t * hq * d)?,
            kn: f(t * hk * d)?,
            att: f(t * (hq * d).max(g.inner))?,
            qkv: f(t * cd)?,
            z: f(t * g.inner)?,
            beta: f(t * hv)?,
            alpha: f(t * hv)?,
            eg: f(t * hv * g.state)?,
            co: f(t * cd)?,
            lq: f(t * hv * g.state)?,
            lk: f(t * hv * g.state)?,
            lv: f(t * hv * g.state)?,
            lo: f(t * hv * g.state)?,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_chat_template_is_the_pipelines() {
        assert_eq!(super::TextEncoder::template("a fox"), "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn q8_0_rows_dequantize() {
        // one block: scale 0.5 (f16 0x3800), values -2 .. 29
        let mut b = vec![0x00, 0x38];
        b.extend((0..32).map(|i| (i as i8 - 2) as u8));
        let v = super::q8_0_row(&b);
        assert_eq!(v.len(), 32);
        assert_eq!((v[0], v[2], v[31]), (-1.0, 0.0, 14.5));
    }
}
