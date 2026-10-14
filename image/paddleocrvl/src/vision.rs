//! The vision tower and the projector (modeling_paddleocr_vl.py: PaddleOCRVisionModel, PaddleOCRProjector), from the
//! mmproj GGUF (`clip.projector_type paddleocr`): a SigLIP NaViT that sees the page at its own size -
//!
//! ```text
//!   patches (14 x 14, a linear with bias) + the learned 27 x 27 position grid resized bilinear to the page's grid
//!     (align_corners, as transformers' implementation)
//!   27 blocks of 1152: LayerNorm -> q, k, v (bias) -> 2D rotary positions (each head of 72: 18 frequencies of the
//!     patch's row, 18 of its column, NEOX halves; base 10000) -> attention over the whole page (16 heads) -> out ->
//!     residual; LayerNorm -> fc1 -> GELU (tanh) -> fc2 -> residual           (LayerNorm eps 1e-6)
//!   post LayerNorm
//!   projector: LayerNorm (eps 1e-5) -> 2 x 2 patches into one row of 4608 -> linear -> GELU (erf) -> linear to 1024
//! ```
//!
//! Weights are BF16 in the file, expanded to float32 at load (the tower's activations stay float32).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

pub const OUT: usize = 1024;
const THETA: f64 = 10000.0;

pub struct Block {
    ln1: (DevBuf, DevBuf),
    ln2: (DevBuf, DevBuf),
    /// q | k | v stacked [3 C][C] and their biases
    qkv: DevBuf,
    qkv_b: DevBuf,
    o: DevBuf,
    o_b: DevBuf,
    up: DevBuf,
    up_b: DevBuf,
    down: DevBuf,
    down_b: DevBuf,
}

pub struct Vision {
    gpu: Arc<Gpu>,
    pub c: usize,
    pub heads: usize,
    pub ff: usize,
    eps: f32,
    patch: DevBuf,
    patch_b: DevBuf,
    /// the learned grid [side][side][C] (host)
    pos: Vec<f32>,
    side: usize,
    blocks: Vec<Block>,
    post: (DevBuf, DevBuf),
    pre_norm: (DevBuf, DevBuf),
    mm1: DevBuf,
    mm1_b: DevBuf,
    mm2: DevBuf,
    mm2_b: DevBuf,
}

pub fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| Error(format!("the model has no {name}")))?;
    let raw = g.read(t).map_err(|e| Error(e.0))?;
    match t.ty {
        GType::F32 => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        GType::BF16 => Ok(raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
        other => Err(Error(format!("{name}: {} (F32 or BF16 here)", other.name()))),
    }
}

/// Tensors stacked by rows into one matrix of type `dt` (F32 or F16) on the GPU: BF16 / F16 / Q8_0 / K-quants
/// expanded by the shared dequantizer, F32 copied (F32 targets only)
pub fn stack(g: &Gguf, gpu: &Arc<Gpu>, nsd: &Nsd, names: &[&str], dt: Dt) -> Result<DevBuf> {
    let es = if dt == Dt::F32 { 4 } else { 2 };
    let mut n = 0usize;
    for name in names {
        let t = g.tensor(name).ok_or_else(|| Error(format!("the model has no {name}")))?;
        n += t.elements() as usize;
    }
    let out = DevBuf::new(gpu, n * es)?;
    let mut at = 0usize;
    for name in names {
        let t = g.tensor(name).unwrap();
        let m = t.elements() as usize;
        let raw = g.read(t).map_err(|e| Error(e.0))?;
        let view = out.view(at * es, m * es)?;
        if t.ty == GType::F32 {
            if dt != Dt::F32 {
                return Err(Error(format!("{name}: float32 into a 16-bit matrix")));
            }
            view.write(0, &raw)?;
        } else {
            let src = DevBuf::new(gpu, raw.len())?;
            src.write(0, &raw)?;
            nsd.dequant(src.ptr(), t.ty.code(), m, view.ptr(), dt)?;
            // src goes at the end of this iteration: its work first
            nsd.wait()?;
        }
        at += m;
    }
    Ok(out)
}

pub fn vec_buf(gpu: &Arc<Gpu>, g: &Gguf, names: &[&str]) -> Result<DevBuf> {
    let mut v = Vec::new();
    for n in names {
        v.extend(f32s(g, n)?);
    }
    DevBuf::from_f32(gpu, &v)
}

/// The learned grid [side][side][C] resized bilinear to [gh][gw][C], align_corners (the corners' values kept)
pub fn resize_pos(pos: &[f32], side: usize, c: usize, gh: usize, gw: usize) -> Vec<f32> {
    let src = |o: usize, n: usize| -> (usize, usize, f32) {
        let s = if n > 1 { o as f64 * (side - 1) as f64 / (n - 1) as f64 } else { 0.0 };
        let i0 = (s.floor() as usize).min(side - 1);
        let i1 = (i0 + 1).min(side - 1);
        (i0, i1, (s - i0 as f64) as f32)
    };
    let mut out = vec![0f32; gh * gw * c];
    for y in 0..gh {
        let (y0, y1, fy) = src(y, gh);
        for x in 0..gw {
            let (x0, x1, fx) = src(x, gw);
            let o = &mut out[(y * gw + x) * c..(y * gw + x + 1) * c];
            let p = |yy: usize, xx: usize| &pos[(yy * side + xx) * c..(yy * side + xx + 1) * c];
            let (a, b, cc, d) = (p(y0, x0), p(y0, x1), p(y1, x0), p(y1, x1));
            for i in 0..c {
                o[i] = (1.0 - fy) * ((1.0 - fx) * a[i] + fx * b[i]) + fy * ((1.0 - fx) * cc[i] + fx * d[i]);
            }
        }
    }
    out
}

impl Vision {
    pub fn load(g: &Gguf, gpu: &Arc<Gpu>, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        if g.meta("clip.projector_type").and_then(|v| v.as_str()) != Some("paddleocr") {
            return Err(Error("the projector file is not PaddleOCR-VL's (clip.projector_type paddleocr)".into()));
        }
        let u = |k: &str| g.meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("the projector has no {k}")));
        let c = u("clip.vision.embedding_length")?;
        let heads = u("clip.vision.attention.head_count")?;
        let ff = u("clip.vision.feed_forward_length")?;
        let n_blocks = u("clip.vision.block_count")?;
        let eps = g.meta("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let m = |names: &[&str]| stack(g, gpu, nsd, names, Dt::F32);
        let b = |names: &[&str]| vec_buf(gpu, g, names);
        let mut blocks = Vec::with_capacity(n_blocks);
        for l in 0..n_blocks {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            let (q, k, v) = (n("attn_q.weight"), n("attn_k.weight"), n("attn_v.weight"));
            let (qb, kb, vb) = (n("attn_q.bias"), n("attn_k.bias"), n("attn_v.bias"));
            blocks.push(Block {
                ln1: (b(&[&n("ln1.weight")])?, b(&[&n("ln1.bias")])?),
                ln2: (b(&[&n("ln2.weight")])?, b(&[&n("ln2.bias")])?),
                qkv: m(&[&q, &k, &v])?,
                qkv_b: b(&[&qb, &kb, &vb])?,
                o: m(&[&n("attn_out.weight")])?,
                o_b: b(&[&n("attn_out.bias")])?,
                up: m(&[&n("ffn_up.weight")])?,
                up_b: b(&[&n("ffn_up.bias")])?,
                down: m(&[&n("ffn_down.weight")])?,
                down_b: b(&[&n("ffn_down.bias")])?,
            });
        }
        let pos = f32s(g, "v.position_embd.weight")?;
        let side = ((pos.len() / c) as f64).sqrt() as usize;
        let v = Vision {
            gpu: gpu.clone(),
            c,
            heads,
            ff,
            eps,
            patch: m(&["v.patch_embd.weight"])?,
            patch_b: b(&["v.patch_embd.bias"])?,
            pos,
            side,
            blocks,
            post: (b(&["v.post_ln.weight"])?, b(&["v.post_ln.bias"])?),
            pre_norm: (b(&["mm.input_norm.weight"])?, b(&["mm.input_norm.bias"])?),
            mm1: m(&["mm.1.weight"])?,
            mm1_b: b(&["mm.1.bias"])?,
            mm2: m(&["mm.2.weight"])?,
            mm2_b: b(&["mm.2.bias"])?,
        };
        log(format!("paddleocrvl vision: {n_blocks} blocks of {c} ({heads} heads), a {side} x {side} position grid, on {} in {:.1} s", gpu.name,
                    t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    /// A page's patches [gh * gw][3 * 14 * 14] to its image tokens [gh/2 * gw/2][1024]; `tap` sees each stage
    pub fn see(&self, nsd: &Nsd, patches: &[f32], gh: usize, gw: usize, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let (c, n) = (self.c, gh * gw);
        let hd = c / self.heads;
        let kp = patches.len() / n;
        let px = DevBuf::from_f32(gpu, patches)?;
        let x = DevBuf::f32(gpu, n * c)?;
        nsd.linear(px.ptr(), Dt::F32, n, kp, self.patch.ptr(), c, self.patch_b.ptr(), x.ptr(), Dt::F32)?;
        let pos = DevBuf::from_f32(gpu, &resize_pos(&self.pos, self.side, c, gh, gw))?;
        nsd.gate_add(x.ptr(), Dt::F32, n, c, pos.ptr(), Dt::F32, none(), none())?;
        tap("v_embd", &x)?;
        // the rotary table: (cos, sin) a patch and pair - pairs 0..18 its row, 18..36 its column
        let half = hd / 2;
        let q4 = half / 2;
        let mut cs = vec![0f32; n * half * 2];
        for r in 0..n {
            let (y, xx) = ((r / gw) as f64, (r % gw) as f64);
            for i in 0..half {
                let (p, j) = if i < q4 { (y, i) } else { (xx, i - q4) };
                let a = p / THETA.powf(2.0 * j as f64 / half as f64);
                cs[(r * half + i) * 2] = a.cos() as f32;
                cs[(r * half + i) * 2 + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let h = DevBuf::f32(gpu, n * c)?;
        let qkv = DevBuf::f32(gpu, n * 3 * c)?;
        let att = DevBuf::f32(gpu, n * c)?;
        let wide = DevBuf::f32(gpu, n * self.ff)?;
        for (l, b) in self.blocks.iter().enumerate() {
            nsd.layer_norm(x.ptr(), Dt::F32, n, c, b.ln1.0.ptr(), b.ln1.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, n, c, b.qkv.ptr(), 3 * c, b.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            nsd.rope(qkv.ptr(), Dt::F32, n, self.heads, hd, 3 * c, cs.ptr(), hd)?;
            nsd.rope(qkv.fp().wrapping_add(c).cast(), Dt::F32, n, self.heads, hd, 3 * c, cs.ptr(), hd)?;
            nsd.attention(qkv.ptr(), qkv.fp().wrapping_add(c).cast(), qkv.fp().wrapping_add(2 * c).cast(), Dt::F32, n, self.heads, hd, 3 * c, att.ptr(),
                          Dt::F32)?;
            nsd.linear(att.ptr(), Dt::F32, n, c, b.o.ptr(), c, b.o_b.ptr(), h.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, c, h.ptr(), Dt::F32, none(), none())?;
            nsd.layer_norm(x.ptr(), Dt::F32, n, c, b.ln2.0.ptr(), b.ln2.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, n, c, b.up.ptr(), self.ff, b.up_b.ptr(), wide.ptr(), Dt::F32)?;
            nsd.gelu(wide.ptr(), Dt::F32, n * self.ff, true)?;
            nsd.linear(wide.ptr(), Dt::F32, n, self.ff, b.down.ptr(), c, b.down_b.ptr(), h.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, c, h.ptr(), Dt::F32, none(), none())?;
            tap(&format!("v_layer-{l}"), &x)?;
        }
        nsd.layer_norm(x.ptr(), Dt::F32, n, c, self.post.0.ptr(), self.post.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
        tap("v_out", &h)?;
        // the projector
        nsd.layer_norm(h.ptr(), Dt::F32, n, c, self.pre_norm.0.ptr(), self.pre_norm.1.ptr(), 1e-5, x.ptr(), Dt::F32)?;
        // SAFETY: x [gh][gw][C] into h [gh/2][gw/2][4 C] (as many values).
        ffi::check(unsafe { (k.merge2)(gpu.raw(), x.fp(), gh as i64, gw as i64, c as i64, h.fp()) }, "merge")?;
        let m = n / 4;
        let mid = DevBuf::f32(gpu, m * 4 * c)?;
        nsd.linear(h.ptr(), Dt::F32, m, 4 * c, self.mm1.ptr(), 4 * c, self.mm1_b.ptr(), mid.ptr(), Dt::F32)?;
        nsd.gelu(mid.ptr(), Dt::F32, m * 4 * c, false)?;
        let out = DevBuf::f32(gpu, m * OUT)?;
        nsd.linear(mid.ptr(), Dt::F32, m, 4 * c, self.mm2.ptr(), OUT, self.mm2_b.ptr(), out.ptr(), Dt::F32)?;
        tap("projected", &out)?;
        // the temporaries go when this returns: their work first (sycl::free does not wait for queued work)
        nsd.wait()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_kept() {
        // a 2 x 2 grid of one channel to 3 x 3: corners as they were, the middle their mean
        let p = resize_pos(&[0.0, 1.0, 2.0, 3.0], 2, 1, 3, 3);
        assert_eq!(p, vec![0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0]);
    }
}
