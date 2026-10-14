//! The vision tower and the merger (transformers' Qwen2_5_VisionTransformerPretrainedModel), from the mmproj GGUF
//! (`clip.projector_type qwen2.5vl_merger`, F16):
//!
//! ```text
//!   patches (14 x 14, two frames of the same picture: the two temporal kernels summed into one linear, no bias)
//!   32 blocks of 1280: RMS norm -> q, k, v (bias) -> 2D rotary positions (each head of 80: 20 frequencies of the
//!     patch's row, 20 of its column, NEOX halves; base 10000) -> attention within the patch's window (blocks 7, 15,
//!     23, 31: over the whole page; 16 heads) -> out (bias) -> residual; RMS norm -> SwiGLU of 3420 (biases) ->
//!     residual                                                              (RMS eps 1e-6)
//!   merger: RMS norm -> a unit's 4 patches into one row of 5120 -> linear -> GELU (erf) -> linear to 3584
//! ```
//!
//! Patches are computed in window order (prep.rs) and the merger's rows put back in row order. Weights float32 here.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::{ffi, prep};

const THETA: f64 = 10000.0;

pub struct Block {
    ln1: DevBuf,
    ln2: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    o: DevBuf,
    o_b: DevBuf,
    gu: DevBuf,
    gu_b: DevBuf,
    down: DevBuf,
    down_b: DevBuf,
}

pub struct Vision {
    gpu: Arc<Gpu>,
    pub c: usize,
    pub heads: usize,
    pub ff: usize,
    pub out: usize,
    eps: f32,
    /// every `full`-th block attends over the whole page
    full: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    patch: DevBuf,
    blocks: Vec<Block>,
    post: DevBuf,
    mm0: DevBuf,
    mm0_b: DevBuf,
    mm2: DevBuf,
    mm2_b: DevBuf,
}

fn half_to_f32(b: u16) -> f32 {
    let (s, e, m) = ((b >> 15) as u32, ((b >> 10) & 31) as u32, (b & 1023) as u32);
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            let mut m = m;
            let mut e = 113u32;
            while m & 1024 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 1023) << 13)
        }
        31 => (s << 31) | (255 << 23) | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| Error(format!("the model has no {name}")))?;
    let raw = g.read(t).map_err(|e| Error(e.0))?;
    match t.ty {
        GType::F32 => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        GType::F16 => Ok(raw.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
        GType::BF16 => Ok(raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
        other => Err(Error(format!("{name}: {} (F32, F16 or BF16 here)", other.name()))),
    }
}

/// Tensors stacked by rows, as float32 on the GPU
fn stack(g: &Gguf, gpu: &Arc<Gpu>, names: &[&str]) -> Result<DevBuf> {
    let mut v = Vec::new();
    for n in names {
        v.extend(f32s(g, n)?);
    }
    DevBuf::from_f32(gpu, &v)
}

impl Vision {
    pub fn load(g: &Gguf, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        if g.meta("clip.projector_type").and_then(|v| v.as_str()) != Some("qwen2.5vl_merger") {
            return Err(Error("the projector file is not Qwen2.5-VL's (clip.projector_type qwen2.5vl_merger)".into()));
        }
        let u = |k: &str| g.meta(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| Error(format!("the projector has no {k}")));
        let c = u("clip.vision.embedding_length")?;
        let heads = u("clip.vision.attention.head_count")?;
        let ff = u("clip.vision.feed_forward_length")?;
        let n_blocks = u("clip.vision.block_count")?;
        let out = u("clip.vision.projection_dim")?;
        let full = u("clip.vision.n_wa_pattern").unwrap_or(8);
        let eps = g.meta("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
        let three = |k: &str, d: [f32; 3]| -> [f32; 3] {
            g.meta(k).and_then(|v| v.as_array()).map(|a| {
                let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                if v.len() == 3 { [v[0], v[1], v[2]] } else { d }
            }).unwrap_or(d)
        };
        let mean = three("clip.vision.image_mean", [0.48145466, 0.4578275, 0.40821073]);
        let std = three("clip.vision.image_std", [0.26862954, 0.2613026, 0.2757771]);
        let mut blocks = Vec::with_capacity(n_blocks);
        for l in 0..n_blocks {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            blocks.push(Block {
                ln1: stack(g, gpu, &[&n("ln1.weight")])?,
                ln2: stack(g, gpu, &[&n("ln2.weight")])?,
                qkv: stack(g, gpu, &[&n("attn_q.weight"), &n("attn_k.weight"), &n("attn_v.weight")])?,
                qkv_b: stack(g, gpu, &[&n("attn_q.bias"), &n("attn_k.bias"), &n("attn_v.bias")])?,
                o: stack(g, gpu, &[&n("attn_out.weight")])?,
                o_b: stack(g, gpu, &[&n("attn_out.bias")])?,
                gu: stack(g, gpu, &[&n("ffn_gate.weight"), &n("ffn_up.weight")])?,
                gu_b: stack(g, gpu, &[&n("ffn_gate.bias"), &n("ffn_up.bias")])?,
                down: stack(g, gpu, &[&n("ffn_down.weight")])?,
                down_b: stack(g, gpu, &[&n("ffn_down.bias")])?,
            });
        }
        // the two frames' kernels summed: the same picture in both
        let mut pw = f32s(g, "v.patch_embd.weight")?;
        if g.tensor("v.patch_embd.weight.1").is_some() {
            for (a, b) in pw.iter_mut().zip(f32s(g, "v.patch_embd.weight.1")?) {
                *a += b;
            }
        }
        let v = Vision {
            gpu: gpu.clone(),
            c,
            heads,
            ff,
            out,
            eps,
            full,
            mean,
            std,
            patch: DevBuf::from_f32(gpu, &pw)?,
            blocks,
            post: stack(g, gpu, &["v.post_ln.weight"])?,
            mm0: stack(g, gpu, &["mm.0.weight"])?,
            mm0_b: stack(g, gpu, &["mm.0.bias"])?,
            mm2: stack(g, gpu, &["mm.2.weight"])?,
            mm2_b: stack(g, gpu, &["mm.2.bias"])?,
        };
        log(format!("olmocr2 vision: {n_blocks} blocks of {c} ({heads} heads, whole-page attention every {full}), on {} in {:.1} s", gpu.name,
                    t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    /// A page's patches (prep.rs, window order) to its image tokens [units][out] in row order; `tap` sees each stage
    /// (the blocks' outputs in window order, as llama.cpp computes them)
    pub fn see(&self, nsd: &Nsd, s: &prep::Seen, tap: &mut dyn FnMut(&str, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let (c, n) = (self.c, s.pos.len());
        let hd = c / self.heads;
        let kp = s.patches.len() / n;
        let px = DevBuf::from_f32(gpu, &s.patches)?;
        let x = DevBuf::f32(gpu, n * c)?;
        nsd.linear(px.ptr(), Dt::F32, n, kp, self.patch.ptr(), c, none(), x.ptr(), Dt::F32)?;
        // the rotary table: (cos, sin) a patch and pair - pairs 0..20 its row, 20..40 its column
        let half = hd / 2;
        let q4 = half / 2;
        let mut cs = vec![0f32; n * half * 2];
        for (r, &(y, xx)) in s.pos.iter().enumerate() {
            for i in 0..half {
                let (p, j) = if i < q4 { (y as f64, i) } else { (xx as f64, i - q4) };
                let a = p / THETA.powf(2.0 * j as f64 / half as f64);
                cs[(r * half + i) * 2] = a.cos() as f32;
                cs[(r * half + i) * 2 + 1] = a.sin() as f32;
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let i32buf = |v: &[i32]| -> Result<DevBuf> {
            let b = DevBuf::new(gpu, v.len() * 4)?;
            b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            Ok(b)
        };
        let (lo, hi) = (i32buf(&s.lo)?, i32buf(&s.hi)?);
        let h = DevBuf::f32(gpu, n * c)?;
        let qkv = DevBuf::f32(gpu, n * 3 * c)?;
        let att = DevBuf::f32(gpu, n * c)?;
        let wide = DevBuf::f32(gpu, n * 2 * self.ff)?;
        let act = DevBuf::f32(gpu, n * self.ff)?;
        for (l, b) in self.blocks.iter().enumerate() {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, n, c, b.ln1.ptr(), self.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, n, c, b.qkv.ptr(), 3 * c, b.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            nsd.rope(qkv.ptr(), Dt::F32, n, self.heads, hd, 3 * c, cs.ptr(), hd)?;
            nsd.rope(qkv.fp().wrapping_add(c).cast(), Dt::F32, n, self.heads, hd, 3 * c, cs.ptr(), hd)?;
            let (qp, kp_, vp) = (qkv.fp(), qkv.fp().wrapping_add(c), qkv.fp().wrapping_add(2 * c));
            if (l + 1) % self.full == 0 {
                nsd.attention(qp.cast(), kp_.cast(), vp.cast(), Dt::F32, n, self.heads, hd, 3 * c, att.ptr(), Dt::F32)?;
            } else {
                // SAFETY: n rows of q / k / v (stride 3 C), each attending within [lo, hi); out [n][C].
                ffi::check(unsafe {
                    (k.seg_attn)(gpu.raw(), qp, kp_, vp, (3 * c) as i64, n as i64, self.heads as i64, hd as i64, lo.ptr().cast(), hi.ptr().cast(), att.fp())
                }, "window attention")?;
            }
            nsd.linear(att.ptr(), Dt::F32, n, c, b.o.ptr(), c, b.o_b.ptr(), h.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, c, h.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, n, c, b.ln2.ptr(), self.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
            nsd.linear(h.ptr(), Dt::F32, n, c, b.gu.ptr(), 2 * self.ff, b.gu_b.ptr(), wide.ptr(), Dt::F32)?;
            nsd.swiglu(wide.ptr(), Dt::F32, n, self.ff, act.ptr(), Dt::F32)?;
            nsd.linear(act.ptr(), Dt::F32, n, self.ff, b.down.ptr(), c, b.down_b.ptr(), h.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, c, h.ptr(), Dt::F32, none(), none())?;
            tap(&format!("layer_out-{l}"), &x)?;
        }
        // the merger: a unit's 4 patches are 4 rows in a row already
        nsd.rms_norm_mod(x.ptr(), Dt::F32, n, c, self.post.ptr(), self.eps, none(), none(), none(), h.ptr(), Dt::F32)?;
        let m = n / 4;
        let mid = DevBuf::f32(gpu, m * 4 * c)?;
        nsd.linear(h.ptr(), Dt::F32, m, 4 * c, self.mm0.ptr(), 4 * c, self.mm0_b.ptr(), mid.ptr(), Dt::F32)?;
        nsd.gelu(mid.ptr(), Dt::F32, m * 4 * c, false)?;
        let win = DevBuf::f32(gpu, m * self.out)?;
        nsd.linear(mid.ptr(), Dt::F32, m, 4 * c, self.mm2.ptr(), self.out, self.mm2_b.ptr(), win.ptr(), Dt::F32)?;
        let back = i32buf(&s.back)?;
        let out = DevBuf::f32(gpu, m * self.out)?;
        // SAFETY: m rows of `out` values gathered back to row order.
        ffi::check(unsafe { (k.gather)(gpu.raw(), win.fp(), self.out as i64, back.ptr().cast(), m as i64, out.fp()) }, "unwindow")?;
        tap("image_tokens", &out)?;
        // the temporaries go when this returns: their work first (sycl::free does not wait for queued work)
        nsd.wait()?;
        Ok(out)
    }
}
