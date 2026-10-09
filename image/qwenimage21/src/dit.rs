//! The denoiser: Qwen-Image 2.1's single-stream transformer (diffusers' QwenImage21Transformer2DModel), 32 blocks of
//! 4,096 features (32 heads x 128), its weights from the GGUF (Q8_0 and BF16) dequantized to half on the GPU.
//!
//! The text and the image share one sequence, but attention is block-causal: the text is causal and never sees the
//! image, and text tokens are modulated from t = 0 (`causal_condition`) - so the text's keys and values in every
//! layer depend on the prompt alone. They are computed once (`prefix`) and every step runs only the image's tokens,
//! each attending to [the text's keys | the image's own]. One block, for the rows it runs:
//!
//! ```text
//!   h = layernorm(x) * (1 + scale1)          q k v = h [Wq|Wk|Wv]; q, k: per-head rmsnorm + 3-axis RoPE
//!   x += tanh(gate1) * attention(q, k, v) Wo
//!   h = layernorm(x) * (1 + scale2)
//!   x += tanh(gate2) * (silu(h Wgate) * (h Wproj)) Wout
//! ```
//!
//! scale / gate: one modulation shared by every block, from the timestep (the image's rows) or t = 0 (the text's).
//! The RoPE pairs features (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so q's and k's output rows (and their
//! norms) are permuted at load - attention scores do not change, v is untouched.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD: usize = 128;
pub const FFN: usize = 12288;
pub const LATENT: usize = 64;
const EPS: f32 = 1e-6;
/// RoPE: the frame, height and width axes' features (pairs: 8, 28, 28)
const AXES: [usize; 3] = [16, 56, 56];

/// One block's weights (half)
struct Block {
    qkv: DevBuf,
    out: DevBuf,
    /// gate | proj rows (SwiGLU reads them as one product)
    gp: DevBuf,
    down: DevBuf,
    norm_q: DevBuf,
    norm_k: DevBuf,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    blocks: Vec<Block>,
    // the small linears, float32
    img_in: DevBuf,
    txt_norm: DevBuf,
    txt_in1: DevBuf,
    txt_in2: DevBuf,
    t_lin1: DevBuf,
    t_lin2: DevBuf,
    modulation: DevBuf,
    norm_out: DevBuf,
    proj_out: DevBuf,
}

/// The text's state every step reads: each layer's q | k | v rows of the text (k with its RoPE), half
pub struct Prefix {
    pub tokens: usize,
    kv: Vec<DevBuf>,
}

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

impl Dit {
    pub fn load(f: &Gguf, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        // a matrix's rows as stored, optionally reordered (`perm`: new row r reads old row perm(r) within each
        // 128-row head), dequantized to `dt` on the GPU
        let load_rows = |names: &[&str], perm: &[bool], dt: Dt| -> Result<DevBuf> {
            let mut raw = Vec::new();
            let mut n = 0usize;
            let mut ty = None;
            for (name, p) in names.iter().zip(perm) {
                let t = tensor(name)?;
                if ty.is_some_and(|x| x != t.ty) {
                    return Err(Error(format!("{name}: {} beside {:?}", t.ty.name(), ty)));
                }
                ty = Some(t.ty);
                let bytes = f.read(t).map_err(ge)?;
                if *p {
                    let rows = t.shape[0] as usize;
                    let rb = bytes.len() / rows;
                    for r in 0..rows {
                        let (h, j) = (r / HEAD, r % HEAD);
                        let src = h * HEAD + rope_perm(j);
                        raw.extend_from_slice(&bytes[src * rb..(src + 1) * rb]);
                    }
                } else {
                    raw.extend_from_slice(&bytes);
                }
                n += t.elements() as usize;
            }
            let ty = ty.ok_or_else(|| Error("no tensors".into()))?;
            // Q8_0, Q4_K, Q6_K, BF16: what the dequant kernel reads
            let code = ty.code();
            if ![8, 12, 14, 30].contains(&code) {
                return Err(Error(format!("{}: {} weights are not read yet", names[0], ty.name())));
            }
            let src = DevBuf::new(&gpu, raw.len())?;
            src.write(0, &raw)?;
            let out = DevBuf::new(&gpu, n * if dt == Dt::F32 { 4 } else { 2 })?;
            nsd.dequant(src.ptr(), code, n, out.ptr(), dt)?;
            nsd.wait()?;
            Ok(out)
        };
        let half = |name: &str| load_rows(&[name], &[false], Dt::F16);
        let single = |name: &str| load_rows(&[name], &[false], Dt::F32);
        // a vector (norm weights) as float32, permuted like q / k when asked
        let vector = |name: &str, permute: bool| -> Result<DevBuf> {
            let t = tensor(name)?;
            let b = f.read(t).map_err(ge)?;
            let v: Vec<f32> = match t.ty {
                GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
                GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                other => return Err(Error(format!("{name}: a {} vector", other.name()))),
            };
            let v = if permute { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(&gpu, &v)
        };
        let n_blocks = (0..).take_while(|i| f.tensor(&format!("transformer_blocks.{i}.attn.to_q.weight")).is_some()).count();
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = format!("transformer_blocks.{i}.");
            blocks.push(Block {
                qkv: load_rows(&[&format!("{p}attn.to_q.weight"), &format!("{p}attn.to_k.weight"), &format!("{p}attn.to_v.weight")],
                               &[true, true, false], Dt::F16)?,
                out: half(&format!("{p}attn.to_out.0.weight"))?,
                gp: load_rows(&[&format!("{p}img_mlp.gate_layer.weight"), &format!("{p}img_mlp.proj.weight")], &[false, false], Dt::F16)?,
                down: half(&format!("{p}img_mlp.out.weight"))?,
                norm_q: vector(&format!("{p}attn.norm_q.weight"), true)?,
                norm_k: vector(&format!("{p}attn.norm_k.weight"), true)?,
            });
        }
        // the text norm stores (scale - 1): the kernel multiplies by its weight
        let tn = {
            let t = tensor("txt_in.text_norm.weight")?;
            let b = f.read(t).map_err(ge)?;
            let v: Vec<f32> = b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) + 1.0).collect();
            DevBuf::from_f32(&gpu, &v)?
        };
        let dit = Dit {
            gpu: gpu.clone(),
            blocks,
            img_in: single("img_in.weight")?,
            txt_norm: tn,
            txt_in1: single("txt_in.in_layer.weight")?,
            txt_in2: single("txt_in.out_layer.weight")?,
            t_lin1: single("time_text_embed.timestep_embedder.linear_1.weight")?,
            t_lin2: single("time_text_embed.timestep_embedder.linear_2.weight")?,
            modulation: single("modulation.1.weight")?,
            norm_out: single("norm_out.linear.weight")?,
            proj_out: single("proj_out.weight")?,
        };
        log(format!("qwen-image 2.1: {n_blocks} blocks on {} (half) in {:.1} s", gpu.name, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The modulation for timestep `t` (a sigma in [0, 1]) and for t = 0: per row (0: t, 1: zero) scale1, tanh(gate1),
    /// scale2, tanh(gate2), the output norm's scale - each [2, 4096] float32 on the GPU
    fn modulation(&self, nsd: &Nsd, t: f32) -> Result<[DevBuf; 5]> {
        let gpu = &self.gpu;
        // the sinusoid: cos then sin of 1000 t * 10000^(-i / 128)
        let mut emb = vec![0f32; 2 * 256];
        for (row, tt) in [t, 0.0].iter().enumerate() {
            for i in 0..128 {
                let a = 1000.0 * *tt as f64 * (-(10000f64).ln() * i as f64 / 128.0).exp();
                emb[row * 256 + i] = a.cos() as f32;
                emb[row * 256 + 128 + i] = a.sin() as f32;
            }
        }
        let silu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
        let e = DevBuf::from_f32(gpu, &emb)?;
        let h = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(e.ptr(), Dt::F32, 2, 256, self.t_lin1.ptr(), DIM, none(), h.ptr(), Dt::F32)?;
        let mut hv = h.to_f32()?;
        silu(&mut hv);
        let h = DevBuf::from_f32(gpu, &hv)?;
        let temb = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(h.ptr(), Dt::F32, 2, DIM, self.t_lin2.ptr(), DIM, none(), temb.ptr(), Dt::F32)?;
        let mut tv = temb.to_f32()?;
        silu(&mut tv);
        let st = DevBuf::from_f32(gpu, &tv)?;
        let m = DevBuf::f32(gpu, 2 * 4 * DIM)?;
        nsd.linear(st.ptr(), Dt::F32, 2, DIM, self.modulation.ptr(), 4 * DIM, none(), m.ptr(), Dt::F32)?;
        let o = DevBuf::f32(gpu, 2 * DIM)?;
        nsd.linear(st.ptr(), Dt::F32, 2, DIM, self.norm_out.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
        let mv = m.to_f32()?;
        // [scale1 | gate1 | scale2 | gate2] per row
        let part = |k: usize, tanh: bool| -> Result<DevBuf> {
            let mut v = vec![0f32; 2 * DIM];
            for row in 0..2 {
                for c in 0..DIM {
                    let x = mv[row * 4 * DIM + k * DIM + c];
                    v[row * DIM + c] = if tanh { x.tanh() } else { x };
                }
            }
            DevBuf::from_f32(gpu, &v)
        };
        Ok([part(0, false)?, part(1, true)?, part(2, false)?, part(3, true)?, DevBuf::from_f32(gpu, &o.to_f32()?)?])
    }

    /// (cos, sin) per token and pair for positions `idx` (frame, height, width per token), in the permuted order
    fn rope_table(&self, idx: &[[i64; 3]]) -> Result<DevBuf> {
        let mut cs = vec![0f32; idx.len() * HEAD];
        for (t, p) in idx.iter().enumerate() {
            let mut pair = 0;
            for (axis, dim) in AXES.iter().enumerate() {
                for i in 0..dim / 2 {
                    let inv = 1.0 / 10000f64.powf(2.0 * i as f64 / *dim as f64);
                    let a = p[axis] as f64 * inv;
                    cs[t * HEAD + 2 * pair] = a.cos() as f32;
                    cs[t * HEAD + 2 * pair + 1] = a.sin() as f32;
                    pair += 1;
                }
            }
        }
        DevBuf::from_f32(&self.gpu, &cs)
    }

    /// One block's rows `x` (float32 [m, 4096], in place): modulation row `row` of `md`; attention through `att`
    #[allow(clippy::too_many_arguments)]
    fn block_ffn(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, md: &[DevBuf; 5], row: usize, zeros: &DevBuf, h: &DevBuf, gp: &DevBuf,
                 act: &DevBuf, o: &DevBuf) -> Result<()> {
        let k = ffi::api()?;
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(row * DIM) };
        nsd.layer_norm(x.ptr(), Dt::F32, m, DIM, none(), none(), EPS, h.ptr(), Dt::F16)?;
        // SAFETY: h holds m x 4096 halves; the scale row 4096 floats.
        ffi::check(unsafe { (k.modulate)(self.gpu.raw(), h.ptr(), m as i64, DIM as i64, off(&md[2])) }, "modulate")?;
        nsd.linear(h.ptr(), Dt::F16, m, DIM, b.gp.ptr(), 2 * FFN, none(), gp.ptr(), Dt::F16)?;
        nsd.swiglu(gp.ptr(), Dt::F16, m, FFN, act.ptr(), Dt::F16)?;
        nsd.linear(act.ptr(), Dt::F16, m, FFN, b.down.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, o.ptr(), Dt::F32, zeros.ptr(), off(&md[3]).cast())
    }

    /// The text's pass through every block (once a prompt): `embeds` float32 [T, 3584 or 4096] -> each layer's q|k|v
    pub fn prefix(&self, nsd: &Nsd, embeds: &DevBuf, tokens: usize, context_dim: usize) -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let t = tokens;
        let md = self.modulation(nsd, 0.0)?;
        let zeros = DevBuf::new(gpu, t.max(1) * 4)?;
        zeros.fill(0)?;
        // txt_in: zero-centred RMS norm, linear, gelu (tanh), linear
        let hn = DevBuf::f32(gpu, t * context_dim)?;
        nsd.rms_norm_mod(embeds.ptr(), Dt::F32, t, context_dim, self.txt_norm.ptr(), EPS, none(), none(), none(), hn.ptr(), Dt::F32)?;
        let h1 = DevBuf::f32(gpu, t * DIM)?;
        nsd.linear(hn.ptr(), Dt::F32, t, context_dim, self.txt_in1.ptr(), DIM, none(), h1.ptr(), Dt::F32)?;
        let h1h = DevBuf::new(gpu, t * DIM * 2)?;
        // SAFETY: t x 4096 values each way.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), h1.fp(), h1h.ptr(), (t * DIM) as i64) }, "to half")?;
        ffi::check(unsafe { (k.gelu_tanh)(gpu.raw(), h1h.ptr(), (t * DIM) as i64) }, "gelu")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), h1h.ptr(), h1.fp(), (t * DIM) as i64) }, "to float")?;
        let x = DevBuf::f32(gpu, t * DIM)?;
        nsd.linear(h1.ptr(), Dt::F32, t, DIM, self.txt_in2.ptr(), DIM, none(), x.ptr(), Dt::F32)?;
        // positions: the text's index on all three axes
        let idx: Vec<[i64; 3]> = (0..t as i64).map(|p| [p, p, p]).collect();
        let cs = self.rope_table(&idx)?;
        let (h, qkv, att, o) = (DevBuf::new(gpu, t * DIM * 2)?, DevBuf::new(gpu, t * 3 * DIM * 2)?, DevBuf::new(gpu, t * DIM * 2)?, DevBuf::f32(gpu, t * DIM)?);
        let (gp, act) = (DevBuf::new(gpu, t * 2 * FFN * 2)?, DevBuf::new(gpu, t * FFN * 2)?);
        let mut kv = Vec::with_capacity(self.blocks.len());
        let off = |buf: &DevBuf| -> *const f32 { buf.fp().wrapping_add(DIM) }; // the t = 0 row
        for b in &self.blocks {
            nsd.layer_norm(x.ptr(), Dt::F32, t, DIM, none(), none(), EPS, h.ptr(), Dt::F16)?;
            // SAFETY: h holds t x 4096 halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.modulate)(gpu.raw(), h.ptr(), t as i64, DIM as i64, off(&md[0])) }, "modulate")?;
            nsd.linear(h.ptr(), Dt::F16, t, DIM, b.qkv.ptr(), 3 * DIM, none(), qkv.ptr(), Dt::F16)?;
            let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
            nsd.rms_rope(qkv.ptr(), Dt::F16, t, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&qkv, DIM), Dt::F16, t, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.attention_causal(qkv.ptr(), half_at(&qkv, DIM), half_at(&qkv, 2 * DIM), Dt::F16, t, HEADS, HEADS, HEAD, 3 * DIM, 3 * DIM, att.ptr())?;
            // the layer's text q|k|v, kept for every step
            let keep = DevBuf::new(gpu, t * 3 * DIM * 2)?;
            keep.copy_within(0, &qkv, 0, t * 3 * DIM * 2)?;
            kv.push(keep);
            nsd.linear(att.ptr(), Dt::F16, t, DIM, b.out.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, t, DIM, o.ptr(), Dt::F32, zeros.ptr(), off(&md[1]).cast())?;
            self.block_ffn(nsd, b, &x, t, &md, 1, &zeros, &h, &gp, &act, &o)?;
        }
        nsd.wait()?;
        Ok(Prefix { tokens: t, kv })
    }

    /// The velocity for latents `lat` (float32 [hw.0 * hw.1, 64]) at sigma `t`, after the text `pre`: float32 [N, 64].
    /// `block0`: block 0's output rows kept there (a check), when given.
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), t: f32, block0: Option<&DevBuf>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let tt = pre.tokens;
        let md = self.modulation(nsd, t)?;
        let zeros = DevBuf::new(gpu, n * 4)?;
        zeros.fill(0)?;
        let x = DevBuf::f32(gpu, n * DIM)?;
        nsd.linear(lat.ptr(), Dt::F32, n, LATENT, self.img_in.ptr(), DIM, none(), x.ptr(), Dt::F32)?;
        // positions: the frame after the text; height and width centred on zero, row-major
        let (hh, ww) = (hw.0 as i64, hw.1 as i64);
        let idx: Vec<[i64; 3]> = (0..n as i64).map(|i| [tt as i64, i / ww - (hh - hh / 2), i % ww - (ww - ww / 2)]).collect();
        let cs = self.rope_table(&idx)?;
        let h = DevBuf::new(gpu, n * DIM * 2)?;
        // [the text's q|k|v | the image's], one layer at a time
        let all = DevBuf::new(gpu, (tt + n) * 3 * DIM * 2)?;
        let att = DevBuf::new(gpu, n * DIM * 2)?;
        let o = DevBuf::f32(gpu, n * DIM)?;
        let (gp, act) = (DevBuf::new(gpu, n * 2 * FFN * 2)?, DevBuf::new(gpu, n * FFN * 2)?);
        let half_at = |buf: &DevBuf, at: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(at).cast() };
        let img = tt * 3 * DIM; // the image's first row in `all`, in halves
        for (bi, b) in self.blocks.iter().enumerate() {
            all.copy_within(0, &pre.kv[bi], 0, tt * 3 * DIM * 2)?;
            nsd.layer_norm(x.ptr(), Dt::F32, n, DIM, none(), none(), EPS, h.ptr(), Dt::F16)?;
            // SAFETY: h holds n x 4096 halves; the scale row 4096 floats.
            ffi::check(unsafe { (k.modulate)(gpu.raw(), h.ptr(), n as i64, DIM as i64, md[0].fp()) }, "modulate")?;
            nsd.linear(h.ptr(), Dt::F16, n, DIM, b.qkv.ptr(), 3 * DIM, none(), half_at(&all, img), Dt::F16)?;
            nsd.rms_rope(half_at(&all, img), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_q.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&all, img + DIM), Dt::F16, n, HEADS, HEAD, 3 * DIM, b.norm_k.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.attention_qk(half_at(&all, img), n, 3 * DIM, half_at(&all, DIM), half_at(&all, 2 * DIM), tt + n, 3 * DIM, Dt::F16, HEADS, HEAD,
                             att.ptr(), Dt::F16)?;
            nsd.linear(att.ptr(), Dt::F16, n, DIM, b.out.ptr(), DIM, none(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, DIM, o.ptr(), Dt::F32, zeros.ptr(), md[1].ptr())?;
            self.block_ffn(nsd, b, &x, n, &md, 0, &zeros, &h, &gp, &act, &o)?;
            if bi == 0 {
                if let Some(keep) = block0 {
                    keep.copy_within(0, &x, 0, n * DIM * 4)?;
                }
            }
        }
        // the output norm: layernorm * (1 + scale), then the projection to 64 latent channels
        let xn = DevBuf::f32(gpu, n * DIM)?;
        nsd.layer_norm(x.ptr(), Dt::F32, n, DIM, none(), none(), EPS, xn.ptr(), Dt::F32)?;
        let xh = DevBuf::new(gpu, n * DIM * 2)?;
        // SAFETY: n x 4096 values each way; the scale row 4096 floats.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), xn.fp(), xh.ptr(), (n * DIM) as i64) }, "to half")?;
        ffi::check(unsafe { (k.modulate)(gpu.raw(), xh.ptr(), n as i64, DIM as i64, md[4].fp()) }, "modulate")?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), xh.ptr(), xn.fp(), (n * DIM) as i64) }, "to float")?;
        let v = DevBuf::f32(gpu, n * LATENT)?;
        nsd.linear(xn.ptr(), Dt::F32, n, DIM, self.proj_out.ptr(), LATENT, none(), v.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(v)
    }
}
