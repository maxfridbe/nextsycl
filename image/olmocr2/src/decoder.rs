//! The decoder (transformers' Qwen2_5_VLTextModel: Qwen2-7B), from the language GGUF (`qwen2vl`, Q8_0): 28 layers
//! of 3584: RMS norm (eps 1e-6) -> q (28 heads of 128), k and v (4 heads each), all with bias -> multimodal rotary
//! positions (M-RoPE: of a head's 64 frequencies the first 16 turn with the token's time position, the next 24 its row,
//! the last 24 its column; base 1e6, NEOX halves) -> attention -> out -> residual; RMS norm -> SwiGLU of 18944 ->
//! residual. Then the final norm and an untied head over 152064 tokens.
//!
//! Weights stay Q8_0 on the card (8 GiB), split at load into 16-byte aligned int8 rows and the blocks' scales: a
//! prompt's pass expands each matrix to half for oneDNN's product over its rows, a written token's pass reads the
//! values directly (one row: q8r_gemv, 16 bytes a lane). The residual stream, the caches and the
//! attention are float32.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;
use crate::vision::f32s;

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
    pub sections: [usize; 3],
}

/// A Q8_0 matrix on the card, repacked: its int8 values [rows][cols] and its blocks' scales (half) [rows][cols / 32];
/// and its bias
struct Mat {
    qs: DevBuf,
    d: DevBuf,
    rows: usize,
    cols: usize,
    bias: Option<DevBuf>,
}

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Decoder {
    gpu: Arc<Gpu>,
    pub g: Geometry,
    embd: DevBuf,
    layers: Vec<Layer>,
    out_norm: DevBuf,
    head: Mat,
    pub weight_bytes: u64,
}

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
    /// a matrix expanded to half (the prompt's pass)
    wh: DevBuf,
}

fn q8_rows(f: &Gguf, gpu: &Arc<Gpu>, names: &[&str], biases: &[&str]) -> Result<Mat> {
    let mut raw = Vec::new();
    let mut rows = 0usize;
    let mut cols = 0usize;
    for n in names {
        let t = f.tensor(n).ok_or_else(|| Error(format!("the model has no {n}")))?;
        if t.ty != GType::Q8_0 {
            return Err(Error(format!("{n}: {} (Q8_0 here)", t.ty.name())));
        }
        cols = t.shape[1] as usize;
        rows += t.shape[0] as usize;
        raw.extend(f.read(t).map_err(|e| Error(e.0))?);
    }
    // the blocks (half scale, 32 int8) split into aligned values and scales
    let nb = raw.len() / 34;
    let mut q = Vec::with_capacity(nb * 32);
    let mut d = Vec::with_capacity(nb * 2);
    for b in raw.chunks_exact(34) {
        d.extend_from_slice(&b[..2]);
        q.extend_from_slice(&b[2..]);
    }
    let qs = DevBuf::new(gpu, q.len())?;
    qs.write(0, &q)?;
    let db = DevBuf::new(gpu, d.len())?;
    db.write(0, &d)?;
    let bias = if biases.is_empty() {
        None
    } else {
        let mut v = Vec::new();
        for b in biases {
            v.extend(f32s(f, b)?);
        }
        Some(DevBuf::from_f32(gpu, &v)?)
    };
    Ok(Mat { qs, d: db, rows, cols, bias })
}

impl Decoder {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Decoder> {
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
            head: n_embd / heads,
            n_ff: u("feed_forward_length")?,
            vocab: et.shape[0] as usize,
            eps: f.arch_meta("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
            theta: f.arch_meta("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(1e6),
            sections: [secs.first().copied().unwrap_or(16), secs.get(1).copied().unwrap_or(24), secs.get(2).copied().unwrap_or(24)],
        };
        if g.sections.iter().sum::<usize>() != g.head / 2 {
            return Err(Error(format!("rope sections {:?} do not cover a head of {}", g.sections, g.head)));
        }
        let mut bytes = 0u64;
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            let n = |s: &str| format!("blk.{l}.{s}");
            let ly = Layer {
                attn_norm: DevBuf::from_f32(gpu, &f32s(f, &n("attn_norm.weight"))?)?,
                ffn_norm: DevBuf::from_f32(gpu, &f32s(f, &n("ffn_norm.weight"))?)?,
                qkv: q8_rows(f, gpu, &[&n("attn_q.weight"), &n("attn_k.weight"), &n("attn_v.weight")],
                             &[&n("attn_q.bias"), &n("attn_k.bias"), &n("attn_v.bias")])?,
                o: q8_rows(f, gpu, &[&n("attn_output.weight")], &[])?,
                gu: q8_rows(f, gpu, &[&n("ffn_gate.weight"), &n("ffn_up.weight")], &[])?,
                down: q8_rows(f, gpu, &[&n("ffn_down.weight")], &[])?,
            };
            bytes += [&ly.qkv, &ly.o, &ly.gu, &ly.down].iter().map(|m| (m.qs.len + m.d.len) as u64).sum::<u64>();
            layers.push(ly);
        }
        let head = q8_rows(f, gpu, &[if f.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" }], &[])?;
        let raw = f.read(et).map_err(|e| Error(e.0))?;
        let embd = DevBuf::new(gpu, raw.len())?;
        embd.write(0, &raw)?;
        bytes += (head.qs.len + head.d.len + embd.len) as u64;
        let d = Decoder { gpu: gpu.clone(), embd, layers, out_norm: DevBuf::from_f32(gpu, &f32s(f, "output_norm.weight")?)?, head, g, weight_bytes: bytes };
        log(format!("olmocr2 decoder: {} layers of {} ({} heads, {} kv), {:.1} GiB of Q8_0 on {} in {:.1} s", d.g.layers, d.g.n_embd, d.g.heads,
                    d.g.kv_heads, bytes as f64 / (1u64 << 30) as f64, gpu.name, t0.elapsed().as_secs_f64()));
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
        let biggest = self.layers.iter().flat_map(|l| [&l.qkv, &l.o, &l.gu, &l.down]).map(|m| m.rows * m.cols).max().unwrap_or(0);
        Ok(Work {
            cs: DevBuf::f32(gpu, t * g.head)?,
            h: DevBuf::f32(gpu, t * e)?,
            hh: DevBuf::new(gpu, t * g.n_ff.max(qw).max(e) * 2)?,
            qkv: DevBuf::f32(gpu, t * qs)?,
            att: DevBuf::f32(gpu, t * qw)?,
            o: DevBuf::f32(gpu, t * e)?,
            wide: DevBuf::f32(gpu, t * 2 * g.n_ff)?,
            act: DevBuf::f32(gpu, t * g.n_ff)?,
            scratch: DevBuf::f32(gpu, if t == 1 { sc } else { 1 })?,
            logits: DevBuf::f32(gpu, g.vocab)?,
            tok: DevBuf::f32(gpu, t * e)?,
            wh: DevBuf::new(gpu, if t == 1 { 2 } else { biggest * 2 })?,
        })
    }

    /// The tokens' embeddings [n][n_embd]
    pub fn embed(&self, nsd: &Nsd, tokens: &[u32]) -> Result<DevBuf> {
        let e = self.g.n_embd;
        let x = DevBuf::f32(&self.gpu, tokens.len() * e)?;
        for (i, t) in tokens.iter().enumerate() {
            self.embed_into(nsd, *t, &x.view(i * e * 4, e * 4)?)?;
        }
        Ok(x)
    }

    fn embed_into(&self, nsd: &Nsd, t: u32, out: &DevBuf) -> Result<()> {
        if t as usize >= self.g.vocab {
            return Err(Error(format!("token {t} is past the vocabulary")));
        }
        let rb = self.g.n_embd / 32 * 34;
        nsd.dequant(self.embd.view(t as usize * rb, rb)?.ptr(), GType::Q8_0.code(), self.g.n_embd, out.ptr(), Dt::F32)
    }

    fn to_half(&self, x: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        // SAFETY: n floats in, n halfs out.
        ffi::check(unsafe { (k.to_half)(self.gpu.raw(), x.fp(), out.ptr(), n as i64) }, "to half")
    }

    /// out [t][m.rows] = x [t][m.cols] m^T (+ bias): one row straight from the blocks, more through half
    fn mul(&self, nsd: &Nsd, w: &Work, m: &Mat, x: &DevBuf, t: usize, out: &DevBuf) -> Result<()> {
        let bias = m.bias.as_ref().map_or(none(), |b| b.ptr() as *const _);
        if t == 1 {
            let k = ffi::api()?;
            // SAFETY: rows x cols int8 and their scales, x [cols], out [rows], bias [rows] or null.
            return ffi::check(unsafe {
                (k.q8r_gemv)(self.gpu.raw(), m.qs.ptr().cast(), m.d.ptr(), x.fp(), m.rows as i64, m.cols as i64, bias.cast(), out.fp())
            }, "q8 product");
        }
        let k = ffi::api()?;
        // SAFETY: rows x cols values expanded into wh (sized for the largest matrix).
        ffi::check(unsafe { (k.q8r_half)(self.gpu.raw(), m.qs.ptr().cast(), m.d.ptr(), m.rows as i64, m.cols as i64, w.wh.ptr()) }, "expand")?;
        self.to_half(x, t * m.cols, &w.hh)?;
        nsd.linear(w.hh.ptr(), Dt::F16, t, m.cols, w.wh.ptr(), m.rows, bias, out.ptr(), Dt::F32)
    }

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

    /// The prompt's pass over rows `x` [t][n_embd] at positions `pos` (cache empty, causal); the last row's logits
    pub fn pass(&self, nsd: &Nsd, c: &mut Cache, x: &DevBuf, pos: &[[u32; 3]], tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        let w = self.work(pos.len(), 1)?;
        self.run(nsd, c, &w, x, pos, tap)
    }

    /// One written token's pass at position `pos`; its logits
    pub fn step(&self, nsd: &Nsd, c: &mut Cache, token: u32, pos: u32) -> Result<Vec<f32>> {
        let w = c.one.take().ok_or_else(|| Error("the cache's buffers are in use".into()))?;
        let r = self.embed_into(nsd, token, &w.tok).and_then(|_| self.run(nsd, c, &w, &w.tok, &[[pos; 3]], &mut |_, _| Ok(())));
        c.one = Some(w);
        r
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
        let Work { cs, h, qkv, att, o, wide, act, scratch, logits, .. } = w;
        cs.write(0, &self.rope_table(pos).iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        for (l, ly) in self.layers.iter().enumerate() {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.attn_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            self.mul(nsd, w, &ly.qkv, h, t, qkv)?;
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
                // SAFETY: one query row over the cache's len + 1 rows; scratch sized for the cache.
                ffi::check(unsafe {
                    (k.attn1)(gpu.raw(), qkv.fp(), c.k[l].fp(), c.v[l].fp(), (c.len + 1) as i64, kvw as i64, g.heads as i64, g.kv_heads as i64, hd as i64,
                              scratch.fp(), att.fp())
                }, "attention")?;
            }
            self.mul(nsd, w, &ly.o, att, t, o)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, t, e, ly.ffn_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            self.mul(nsd, w, &ly.gu, h, t, wide)?;
            nsd.swiglu(wide.ptr(), Dt::F32, t, g.n_ff, act.ptr(), Dt::F32)?;
            self.mul(nsd, w, &ly.down, act, t, o)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, e, o.ptr(), Dt::F32, none(), none())?;
            tap(&format!("l_out-{l}"), x)?;
        }
        c.len += t;
        let last = x.view((t - 1) * e * 4, e * 4)?;
        nsd.rms_norm_mod(last.ptr(), Dt::F32, 1, e, self.out_norm.ptr(), g.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
        tap("result_norm", &h.view(0, e * 4)?)?;
        let k1 = ffi::api()?;
        // SAFETY: the head's vocab rows against the one normed row.
        ffi::check(unsafe {
            (k1.q8r_gemv)(gpu.raw(), self.head.qs.ptr().cast(), self.head.d.ptr(), h.fp(), self.head.rows as i64, e as i64, std::ptr::null(), logits.fp())
        }, "head")?;
        nsd.wait()?;
        logits.to_f32()
    }
}
