//! I2V's image encoder: SigLIP so400m-patch14-384's vision tower (transformers' SiglipVisionModel; the pipeline loads
//! FLUX.1-Redux-dev's `image_encoder`, an so400m-patch14-384 tower) - 27 layers of 1152, 16 heads of 72, GELU-tanh MLP
//! 4304, layer norms eps 1e-6, a 14 x 14 patch embedding of a 384 x 384 picture (729 tokens) plus learned positions,
//! the post layer norm: last_hidden_state [729, 1152], what the DiT's vision_in reads. float32 throughout.
//!
//! The picture: resize_and_center_crop to the clip's canvas (Lanczos), then SiglipImageProcessor: 384 x 384 (bicubic),
//! / 255, (x - 0.5) / 0.5.

use std::path::Path;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::ffi::{Act, K, T};
use crate::w::St;

pub const SIDE: usize = 384;
const PATCH: usize = 14;
const GRID: usize = SIDE / PATCH;
pub const TOKENS: usize = GRID * GRID;
pub const DIM: usize = 1152;
const HEADS: usize = 16;
const HEAD: usize = 72;
const EPS: f32 = 1e-6;

struct Lin {
    w: DevBuf,
    b: DevBuf,
    n: usize,
    k: usize,
}

struct Layer {
    ln1: (DevBuf, DevBuf),
    qkv: Lin,
    o: Lin,
    ln2: (DevBuf, DevBuf),
    fc1: Lin,
    fc2: Lin,
}

pub struct Vision {
    k: K,
    patch: Lin,
    pos: DevBuf,
    layers: Vec<Layer>,
    post: (DevBuf, DevBuf),
    pub bytes: usize,
}

impl Vision {
    /// The vision tower of a SigLIP safetensors file (`vision_model.*`: google's whole model or Redux's image_encoder)
    pub fn load(file: &Path, k: &K, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        let gpu = &k.gpu;
        let st = St::open(file)?;
        let mut bytes = 0usize;
        let mut lin = |names: &[String]| -> Result<Lin> {
            let (mut w, mut b, mut n, mut kk) = (Vec::new(), Vec::new(), 0usize, 0usize);
            for name in names {
                let (x, shape) = st.get(&format!("{name}.weight"))?;
                n += shape[0];
                kk = shape.iter().skip(1).product();
                w.extend(x);
                b.extend(st.vec(&format!("{name}.bias"))?);
            }
            bytes += (w.len() + b.len()) * 4;
            Ok(Lin { w: DevBuf::from_f32(gpu, &w)?, b: DevBuf::from_f32(gpu, &b)?, n, k: kk })
        };
        let p = "vision_model";
        let patch = lin(&[format!("{p}.embeddings.patch_embedding")])?;
        if patch.n != DIM || patch.k != 3 * PATCH * PATCH {
            return Err(Error(format!("{}: a patch embedding of {} x {}, not SigLIP so400m-patch14's", file.display(), patch.n, patch.k)));
        }
        let pos = DevBuf::from_f32(gpu, &st.vec(&format!("{p}.embeddings.position_embedding.weight"))?)?;
        let ln = |name: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(gpu, &st.vec(&format!("{name}.weight"))?)?, DevBuf::from_f32(gpu, &st.vec(&format!("{name}.bias"))?)?))
        };
        let mut layers = Vec::new();
        for l in 0.. {
            let q = format!("{p}.encoder.layers.{l}");
            if !st.has(&format!("{q}.layer_norm1.weight")) {
                break;
            }
            layers.push(Layer {
                ln1: ln(&format!("{q}.layer_norm1"))?,
                qkv: lin(&["q_proj", "k_proj", "v_proj"].map(|x| format!("{q}.self_attn.{x}")))?,
                o: lin(&[format!("{q}.self_attn.out_proj")])?,
                ln2: ln(&format!("{q}.layer_norm2"))?,
                fc1: lin(&[format!("{q}.mlp.fc1")])?,
                fc2: lin(&[format!("{q}.mlp.fc2")])?,
            });
        }
        let post = ln(&format!("{p}.post_layernorm"))?;
        log(format!("image encoder: SigLIP so400m-patch14-384, {} layers (float32, {:.2} GiB) in {:.1} s", layers.len(), bytes as f64 / (1u64 << 30) as f64,
                    t0.elapsed().as_secs_f64()));
        Ok(Vision { k: k.clone(), patch, pos, layers, post, bytes })
    }

    /// The tokens of a picture as SigLIP sees it (float32 [384, 384, 3], already normalized to [-1, 1]): float32
    /// [729, 1152]
    pub fn encode(&self, nsd: &Nsd, px: &[f32]) -> Result<Vec<f32>> {
        if px.len() != SIDE * SIDE * 3 {
            return Err(Error(format!("{} values: a 384 x 384 RGB picture here", px.len())));
        }
        let gpu = &nsd.gpu;
        let k = &self.k;
        // the patches in the convolution's order (channel, row, column within the patch)
        let mut patches = vec![0f32; TOKENS * 3 * PATCH * PATCH];
        for gy in 0..GRID {
            for gx in 0..GRID {
                let t = gy * GRID + gx;
                for c in 0..3 {
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            patches[t * 3 * PATCH * PATCH + (c * PATCH + y) * PATCH + x] = px[((gy * PATCH + y) * SIDE + gx * PATCH + x) * 3 + c];
                        }
                    }
                }
            }
        }
        let pd = DevBuf::from_f32(gpu, &patches)?;
        let x = DevBuf::f32(gpu, TOKENS * DIM)?;
        let lin = |l: &Lin, inp: &DevBuf, out: &DevBuf| nsd.linear(inp.ptr(), Dt::F32, TOKENS, l.k, l.w.ptr(), l.n, l.b.ptr(), out.ptr(), Dt::F32);
        lin(&self.patch, &pd, &x)?;
        k.axpy(x.fp(), self.pos.fp(), TOKENS * DIM, 1.0)?;
        let h = DevBuf::f32(gpu, TOKENS * DIM)?;
        let qkv = DevBuf::f32(gpu, TOKENS * 3 * DIM)?;
        let att = DevBuf::f32(gpu, TOKENS * DIM)?;
        let o = DevBuf::f32(gpu, TOKENS * DIM)?;
        let ff = self.layers.first().map_or(0, |l| l.fc1.n);
        let mid = DevBuf::f32(gpu, TOKENS * ff)?;
        for l in &self.layers {
            k.ln_mod(x.fp(), TOKENS, DIM, EPS, l.ln1.0.fp(), l.ln1.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), T::F32)?;
            lin(&l.qkv, &h, &qkv)?;
            k.attn_small(qkv.fp(), 3 * DIM, qkv.fp().wrapping_add(DIM), qkv.fp().wrapping_add(2 * DIM), 3 * DIM, TOKENS, HEADS, HEADS, HEAD,
                         1.0 / (HEAD as f32).sqrt(), std::ptr::null(), false, att.fp())?;
            lin(&l.o, &att, &o)?;
            k.axpy(x.fp(), o.fp(), TOKENS * DIM, 1.0)?;
            k.ln_mod(x.fp(), TOKENS, DIM, EPS, l.ln2.0.fp(), l.ln2.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), T::F32)?;
            lin(&l.fc1, &h, &mid)?;
            k.act(mid.ptr(), T::F32, TOKENS * ff, Act::GeluTanh)?;
            nsd.linear(mid.ptr(), Dt::F32, TOKENS, ff, l.fc2.w.ptr(), l.fc2.n, l.fc2.b.ptr(), o.ptr(), Dt::F32)?;
            k.axpy(x.fp(), o.fp(), TOKENS * DIM, 1.0)?;
        }
        k.ln_mod(x.fp(), TOKENS, DIM, EPS, self.post.0.fp(), self.post.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), T::F32)?;
        nsd.wait()?;
        h.to_f32()
    }
}

/// A picture (8-bit RGB) resized to cover w x h and cut in the middle (the pipeline's resize_and_center_crop and its
/// condition picture: Lanczos), float32 [h, w, 3] in [-1, 1]
pub fn cover(rgb: &[u8], iw: usize, ih: usize, w: usize, h: usize) -> Result<Vec<f32>> {
    let img = image::RgbImage::from_raw(iw as u32, ih as u32, rgb.to_vec()).ok_or_else(|| Error("a picture of the wrong size".into()))?;
    let s = (w as f64 / iw as f64).max(h as f64 / ih as f64);
    let (rw, rh) = (((iw as f64 * s).round() as u32).max(w as u32), ((ih as f64 * s).round() as u32).max(h as u32));
    let r = image::imageops::resize(&img, rw, rh, image::imageops::FilterType::Lanczos3);
    let (x0, y0) = ((rw as usize - w) / 2, (rh as usize - h) / 2);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let p = r.get_pixel((x0 + x) as u32, (y0 + y) as u32);
            out.extend(p.0.iter().map(|v| *v as f32 / 127.5 - 1.0));
        }
    }
    Ok(out)
}

/// SiglipImageProcessor on a picture (float32 [h, w, 3] in [-1, 1], from `cover`): 384 x 384, bicubic, as [-1, 1]
pub fn siglip_pixels(px: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
    let bytes: Vec<u8> = px.iter().map(|v| ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8).collect();
    let img = image::RgbImage::from_raw(w as u32, h as u32, bytes).ok_or_else(|| Error("a picture of the wrong size".into()))?;
    let r = image::imageops::resize(&img, SIDE as u32, SIDE as u32, image::imageops::FilterType::CatmullRom);
    Ok(r.into_raw().iter().map(|v| *v as f32 / 127.5 - 1.0).collect())
}
