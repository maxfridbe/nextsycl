//! The decoder (modeling_paddleocr_vl.py: Ernie4_5Model), from the language GGUF (`paddleocr`): ERNIE-4.5 0.3B -
//! 18 layers of 1024: RMS norm (eps 1e-5) -> q (16 heads of 128), k and v (2 heads each) -> multimodal rotary
//! positions (M-RoPE, Qwen2-VL's: of a head's 64 frequencies the first 16 turn with the token's time position, the
//! next 24 its row, the last 24 its column; base 500000, NEOX halves) -> attention -> out -> residual; RMS norm ->
//! SwiGLU of 3072 -> residual. Then the final norm and an untied head over 103424 tokens.
//!
//! Weights are BF16 in the file, expanded to half once at load (oneDNN's half linears); the residual stream, the
//! caches and the attention are float32. The cache holds every token (prompt and written).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::Gguf;

use crate::ffi;
use crate::vision::{stack, vec_buf};

pub struct Geometry {
    pub layers: usize,
    pub n_embd: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f64,
    /// the frequencies of each position axis (time, row, column)
    pub sections: [usize; 3],
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    /// q | k | v stacked [q + 2 kv][n_embd], o [n_embd][q], gate | up [2 n_ff][n_embd], down [n_embd][n_ff] (half)
    qkv: DevBuf,
    o: DevBuf,
    gu: DevBuf,
    down: DevBuf,
}

pub struct Decoder {
    gpu: Arc<Gpu>,
    pub g: Geometry,
    /// the token embedding's BF16 rows (expanded as they are needed)
    embd: DevBuf,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    head: DevBuf,
}

/// A reading's caches: per layer K and V [cap][kv_heads * head] float32; and the buffers of a one-token pass, kept
/// for the reading (an allocation a pass costs more than the pass)
pub struct Cache {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
    pub len: usize,
    pub cap: usize,
    one: Option<Work>,
}

/// A pass's buffers for t rows
struct Work {
    cs: DevBuf,
    h: DevBuf,
    hh: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    o: DevBuf,
    wide: DevBuf,
    act: DevBuf,
    scratch: DevBuf,
    logits: DevBuf,
    tok: DevBuf,
}

impl Decoder {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Decoder> {
        let t0 = std::time::Instant::now();
        let u = |k: &str| f.arch_meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("the model has no {k}")));
        let secs: Vec<usize> = f.arch_meta("rope.dimension_sections").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect()).unwrap_or_else(|| vec![16, 24, 24]);
        let n_embd = u("embedding_length")?;
        let heads = u("attention.head_count")?;
        let et = f.tensor("token_embd.weight").ok_or_else(|| Error("the model has no token_embd.weight".into()))?;
        let g = Geometry {
            layers: u("block_count")?,
            n_embd,
            heads,
            kv_heads: u("attention.head_count_kv")?,
            head: u("attention.key_length").unwrap_or(n_embd / heads),
            n_ff: u("feed_forward_length")?,
            vocab: et.shape[0] as usize,
            eps: f.arch_meta("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            theta: f.arch_meta("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(500000.0),
            sections: [secs.first().copied().unwrap_or(16), secs.get(1).copied().unwrap_or(24), secs.get(2).copied().unwrap_or(24)],
        };
        if g.sections.iter().sum::<usize>() != g.head / 2 {
            return Err(Error(format!("rope sections {:?} do not cover a head of {}", g.sections, g.head)));
        }
        let m = |names: &[&str]| stack(f, gpu, nsd, names, Dt::F16);
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}.weight");
            layers.push(Layer {
                attn_norm: vec_buf(gpu, f, &[&n("attn_norm")])?,
                ffn_norm: vec_buf(gpu, f, &[&n("ffn_norm")])?,
                qkv: m(&[&n("attn_q"), &n("attn_k"), &n("attn_v")])?,
                o: m(&[&n("attn_output")])?,
                gu: m(&[&n("ffn_gate"), &n("ffn_up")])?,
                down: m(&[&n("ffn_down")])?,
            });
        }
        let hn = if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let head = m(&[hn])?;
        if et.ty != nextsycl_gguf::GType::BF16 {
            return Err(Error(format!("token_embd.weight: {} (BF16 here)", et.ty.name())));
        }
        let raw = f.read(et).map_err(|e| Error(e.0))?;
        let embd = DevBuf::new(gpu, raw.len())?;
        embd.write(0, &raw)?;
        let d = Decoder { gpu: gpu.clone(), embd, layers, out_norm: vec_buf(gpu, f, &["output_norm.weight"])?, head, g };
        log(format!("paddleocrvl decoder: {} layers of {} ({} heads, {} kv), expanded to half on {} in {:.1} s", d.g.layers, d.g.n_embd, d.g.heads,
                    d.g.kv_heads, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(d)
    }

    pub fn cache(&self, cap: usize) -> Result<Cache> {
        let kv = self.g.kv_heads * self.g.head;
        let mut k = Vec::with_capacity(self.g.layers);
        let mut v = Vec::with_capacity(self.g.layers);
        for _ in 0..self.g.layers {
            k.push(DevBuf::f32(&self.gpu, cap * kv)?);
            v.push(DevBuf::f32(&self.gpu, cap * kv)?);
        }
        Ok(Cache { k, v, len: 0, cap, one: Some(self.work(1, cap)?) })
    }

    fn work(&self, t: usize, keys: usize) -> Result<Work> {
        let g = &self.g;
        let gpu = &self.gpu;
        let (e, qw) = (g.n_embd, g.heads * g.head);
        let qs = qw + 2 * g.kv_heads * g.head;
        // SAFETY: a size query.
        let sc = unsafe { (ffi::api()?.attn1_scratch)(keys as i64, g.heads as i64, g.head as i64) } as usize;
        Ok(Work {
            cs: DevBuf::f32(gpu, t * g.head)?,
            h: DevBuf::f32(gpu, t * e)?,
            hh: DevBuf::new(gpu, t * (2 * g.n_ff).max(qw).max(e) * 2)?,
            qkv: DevBuf::f32(gpu, t * qs)?,
            att: DevBuf::f32(gpu, t * qw)?,
            o: DevBuf::f32(gpu, t * e)?,
            wide: DevBuf::f32(gpu, t * 2 * g.n_ff)?,
            act: DevBuf::f32(gpu, t * g.n_ff)?,
            scratch: DevBuf::f32(gpu, if t == 1 { sc } else { 1 })?,
            logits: DevBuf::f32(gpu, g.vocab)?,
            tok: DevBuf::f32(gpu, t * e)?,
        })
    }

    /// One written token's pass: its embedding, at position `pos`, appended to the cache; its logits
    pub fn step(&self, nsd: &Nsd, c: &mut Cache, token: u32, pos: u32) -> Result<Vec<f32>> {
        if token as usize >= self.g.vocab {
            return Err(Error(format!("token {token} is past the vocabulary")));
        }
        let e = self.g.n_embd;
        let w = c.one.take().ok_or_else(|| Error("the cache's buffers are in use".into()))?;
        nsd.dequant(self.embd.view(token as usize * e * 2, e * 2)?.ptr(), nextsycl_gguf::GType::BF16.code(), e, w.tok.ptr(), Dt::F32)?;
        let r = self.run(nsd, c, &w, &w.tok, &[[pos; 3]], &mut |_, _| Ok(()));
        c.one = Some(w);
        r
    }

    /// The tokens' embeddings [n][n_embd]
    pub fn embed(&self, nsd: &Nsd, tokens: &[u32]) -> Result<DevBuf> {
        let e = self.g.n_embd;
        let x = DevBuf::f32(&self.gpu, tokens.len() * e)?;
        for (i, t) in tokens.iter().enumerate() {
            if *t as usize >= self.g.vocab {
                return Err(Error(format!("token {t} is past the vocabulary")));
            }
            nsd.dequant(self.embd.view(*t as usize * e * 2, e * 2)?.ptr(), nextsycl_gguf::GType::BF16.code(), e, x.view(i * e * 4, e * 4)?.ptr(), Dt::F32)?;
        }
        Ok(x)
    }

    fn to_half(&self, x: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        // SAFETY: n floats in, n halfs out.
        ffi::check(unsafe { (k.to_half)(self.gpu.raw(), x.fp(), out.ptr(), n as i64) }, "to half")
    }

    /// The rotary table of rows at positions `pos` (time, row, column each): (cos, sin) a row and pair
    fn rope_table(&self, pos: &[[u32; 3]]) -> Vec<f32> {
        let g = &self.g;
        let half = g.head / 2;
        let mut cs = vec![0f32; pos.len() * half * 2];
        for (r, p) in pos.iter().enumerate() {
            for i in 0..half {
                let axis = if i < g.sections[0] { 0 } else if i < g.sections[0] + g.sections[1] { 1 } else { 2 };
                let a = p[axis] as f64 / g.theta.powf(2.0 * i as f64 / g.head as f64);
                cs[(r * half + i) * 2] = a.cos() as f32;
                cs[(r * half + i) * 2 + 1] = a.sin() as f32;
            }
        }
        cs
    }

    /// One pass over rows `x` [t][n_embd] at positions `pos`, appended to the cache: the prompt (cache empty, causal)
    /// or one written token (attending to everything cached). The last row's logits.
    pub fn pass(&self, nsd: &Nsd, c: &mut Cache, x: &DevBuf, pos: &[[u32; 3]], tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        if pos.len() == 1 && c.len > 0 {
            let w = c.one.take().ok_or_else(|| Error("the cache's buffers are in use".into()))?;
            let r = self.run(nsd, c, &w, x, pos, tap);
            c.one = Some(w);
            return r;
        }
        let w = self.work(pos.len(), 1)?;
        self.run(nsd, c, &w, x, pos, tap)
    }

    fn run(&self, nsd: &Nsd, c: &mut Cache, w: &Work, x: &DevBuf, pos: &[[u32; 3]], tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let g = &self.g;
        let t = pos.len();
        let (e, hd) = (g.n_embd, g.head);
        let (qw, kvw) = (g.heads * hd, g.kv_heads * hd);
        let qs = qw + 2 * kvw;
        if c.len + t > c.cap {
            return Err(Error(format!("the cache holds {} tokens", c.cap)));
        }
        if c.len > 0 && t != 1 {
            return Err(Error("a pass is the prompt or one token".into()));
        }
        let Work { cs, h, hh, qkv, att, o, wide, act, scratch, logits, .. } = w;
        cs.write(0, &self.rope_table(pos).iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        for (l, ly) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.attn_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            self.to_half(h, t * e, hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, e, ly.qkv.ptr(), qs, none(), qkv.ptr(), Dt::F32)?;
            nsd.rope(qkv.ptr(), Dt::F32, t, g.heads, hd, qs, cs.ptr(), hd)?;
            nsd.rope(qkv.fp().wrapping_add(qw).cast(), Dt::F32, t, g.kv_heads, hd, qs, cs.ptr(), hd)?;
            // SAFETY: the rows' k and v into their cache rows (stride kv_heads * head).
            unsafe {
                ffi::check((k.copy_rows)(gpu.raw(), qkv.fp().add(qw), qs as i64, c.k[l].fp().add(c.len * kvw), kvw as i64, t as i64, kvw as i64), "k cache")?;
                ffi::check((k.copy_rows)(gpu.raw(), qkv.fp().add(qw + kvw), qs as i64, c.v[l].fp().add(c.len * kvw), kvw as i64, t as i64, kvw as i64),
                           "v cache")?;
            }
            if c.len == 0 {
                nsd.attention_causal(qkv.ptr(), c.k[l].ptr(), c.v[l].ptr(), Dt::F32, t, g.heads, g.kv_heads, hd, qs, kvw, att.ptr())?;
            } else {
                // SAFETY: one query row over the cache's len + 1 rows; scratch [heads][len + 1].
                ffi::check(unsafe {
                    (k.attn1)(gpu.raw(), qkv.fp(), c.k[l].fp(), c.v[l].fp(), (c.len + 1) as i64, kvw as i64, g.heads as i64, g.kv_heads as i64, hd as i64,
                              scratch.fp(), att.fp())
                }, "attention")?;
            }
            self.to_half(att, t * qw, hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, qw, ly.o.ptr(), e, none(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.ffn_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            self.to_half(h, t * e, hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, e, ly.gu.ptr(), 2 * g.n_ff, none(), wide.ptr(), Dt::F32)?;
            nsd.swiglu(wide.ptr(), Dt::F32, t, g.n_ff, act.ptr(), Dt::F32)?;
            self.to_half(act, t * g.n_ff, hh)?;
            nsd.linear(hh.ptr(), Dt::F16, t, g.n_ff, ly.down.ptr(), e, none(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
            tap(&format!("l_out-{l}"), x)?;
        }
        c.len += t;
        // the last row's logits
        let last = x.view((t - 1) * e * 4, e * 4)?;
        nsd.rms_norm_mod(last.ptr(), Dt::F32, 1, e, self.out_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
        self.to_half(h, e, hh)?;
        nsd.linear(hh.ptr(), Dt::F16, 1, e, self.head.ptr(), g.vocab, none(), logits.ptr(), Dt::F32)?;
        nsd.wait()?;
        logits.to_f32()
    }
}
