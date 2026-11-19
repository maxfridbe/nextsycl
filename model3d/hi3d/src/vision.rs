//! The two frozen picture encoders of sparc3d's ImageConditioner (float32 in the reference), as transformers runs them:
//!
//! - DINOv2-large (Dinov2Model, its last hidden state): 756 px, 54 x 54 patches of 14 and the class token = 2,917
//!   tokens of 1,024; the 37 x 37 position table resized bicubically (align_corners false, A = -0.75: transformers
//!   4.46's interpolate_pos_encoding); 24 layers of pre-norm attention and a GELU MLP, each scaled (layer scale), the
//!   final layer norm. eps 1e-6.
//! - CLIP ViT-L/14 (CLIPVisionModel, its pooler output): 224 px, 16 x 16 patches (no bias) and the class embedding,
//!   the 257 positions, pre_layrnorm, 24 layers with quick_gelu, post_layernorm of the class token. eps 1e-5.
//!
//! The patches are gathered on the host into rows of (channel, ky, kx) and projected on the GPU; half activations into
//! the products, a float32 residual stream.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::w::{half_bytes, File, Lin};
use crate::Ctx;

const PATCH: usize = 14;

struct Layer {
    n1: (DevBuf, DevBuf),
    qkv: Lin,
    o: Lin,
    ls1: Option<DevBuf>,
    n2: (DevBuf, DevBuf),
    fc1: Lin,
    fc2: Lin,
    ls2: Option<DevBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Dino,
    Clip,
}

pub struct Vit {
    kind: Kind,
    layers: Vec<Layer>,
    patch: Lin,
    /// [the class token + its position | the patches' positions], float32 [tokens, width]
    pos: DevBuf,
    pre: Option<(DevBuf, DevBuf)>,
    post: (DevBuf, DevBuf),
    pub width: usize,
    heads: usize,
    eps: f32,
    pub size: usize,
    pub grid: usize,
}

/// PyTorch's bicubic resize of a [c, n, n] table to [c, m, m] (align_corners false, A = -0.75, edges clamped)
pub fn bicubic(src: &[f32], c: usize, n: usize, m: usize) -> Vec<f32> {
    let w = |t: f32| -> [f32; 4] {
        let a = -0.75f32;
        let c1 = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        let c2 = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
    };
    let scale = n as f32 / m as f32;
    let taps: Vec<(i64, [f32; 4])> = (0..m).map(|o| {
        let s = (o as f32 + 0.5) * scale - 0.5;
        let f = s.floor();
        (f as i64, w(s - f))
    }).collect();
    let at = |i: i64| i.clamp(0, n as i64 - 1) as usize;
    let mut out = vec![0f32; c * m * m];
    for ch in 0..c {
        let plane = &src[ch * n * n..(ch + 1) * n * n];
        for (oy, (fy, wy)) in taps.iter().enumerate() {
            for (ox, (fx, wx)) in taps.iter().enumerate() {
                let mut acc = 0f32;
                for (j, wyj) in wy.iter().enumerate() {
                    let row = &plane[at(fy - 1 + j as i64) * n..];
                    let mut r = 0f32;
                    for (i, wxi) in wx.iter().enumerate() {
                        r += wxi * row[at(fx - 1 + i as i64)];
                    }
                    acc += wyj * r;
                }
                out[ch * m * m + oy * m + ox] = acc;
            }
        }
    }
    out
}

/// Three matrices [n, k] stacked into one [3 n, k] layer with their biases
fn stack(gpu: &Arc<Gpu>, f: &File, names: [&str; 3]) -> Result<Lin> {
    let mut w = Vec::new();
    let mut b = Vec::new();
    let mut n = 0;
    let mut k = 0;
    for p in names {
        let (sh, v) = f.host(&format!("{p}.weight"))?;
        (n, k) = (n + sh[0], sh[1]);
        w.extend(v);
        b.extend(f.vec(&format!("{p}.bias"))?);
    }
    Lin::from_host(gpu, &w, Some(&b), n, k, false)
}

impl Vit {
    /// DINOv2 (`f`'s names as Dinov2Model's) for pictures of `size` pixels
    pub fn dino(f: &File, gpu: &Arc<Gpu>, size: usize) -> Result<Vit> {
        let mut layers = Vec::new();
        while f.has(&format!("encoder.layer.{}.norm1.weight", layers.len())) {
            let p = format!("encoder.layer.{}.", layers.len());
            let a = format!("{p}attention.attention.");
            layers.push(Layer {
                n1: (f.dev(gpu, &format!("{p}norm1.weight"))?, f.dev(gpu, &format!("{p}norm1.bias"))?),
                qkv: stack(gpu, f, [&format!("{a}query"), &format!("{a}key"), &format!("{a}value")])?,
                o: Lin::load(f, gpu, &format!("{p}attention.output.dense"), false)?,
                ls1: Some(f.dev(gpu, &format!("{p}layer_scale1.lambda1"))?),
                n2: (f.dev(gpu, &format!("{p}norm2.weight"))?, f.dev(gpu, &format!("{p}norm2.bias"))?),
                fc1: Lin::load(f, gpu, &format!("{p}mlp.fc1"), false)?,
                fc2: Lin::load(f, gpu, &format!("{p}mlp.fc2"), false)?,
                ls2: Some(f.dev(gpu, &format!("{p}layer_scale2.lambda1"))?),
            });
        }
        if layers.is_empty() {
            return Err(Error(format!("{}: no DINOv2 layers", f.path())));
        }
        let patch = Lin::load(f, gpu, "embeddings.patch_embeddings.projection", false)?;
        let width = patch.n;
        let grid = size / PATCH;
        let table = f.vec("embeddings.position_embeddings")?;
        let cls = f.vec("embeddings.cls_token")?;
        let n = ((table.len() / width - 1) as f64).sqrt().round() as usize;
        if n * n + 1 != table.len() / width {
            return Err(Error(format!("a position table of {} rows", table.len() / width)));
        }
        let mut pos: Vec<f32> = (0..width).map(|c| cls[c] + table[c]).collect();
        if grid == n {
            pos.extend_from_slice(&table[width..]);
        } else {
            let planes: Vec<f32> = (0..width).flat_map(|c| (0..n * n).map(move |i| (c, i))).map(|(c, i)| table[(1 + i) * width + c]).collect();
            let r = bicubic(&planes, width, n, grid);
            for i in 0..grid * grid {
                pos.extend((0..width).map(|c| r[c * grid * grid + i]));
            }
        }
        Ok(Vit {
            kind: Kind::Dino,
            layers,
            patch,
            pos: DevBuf::from_f32(gpu, &pos)?,
            pre: None,
            post: (f.dev(gpu, "layernorm.weight")?, f.dev(gpu, "layernorm.bias")?),
            width,
            heads: width / 64,
            eps: 1e-6,
            size,
            grid,
        })
    }

    /// CLIP's vision tower (`f`'s names as CLIPModel's, prefix "vision_model.")
    pub fn clip(f: &File, gpu: &Arc<Gpu>) -> Result<Vit> {
        let mut layers = Vec::new();
        while f.has(&format!("encoder.layers.{}.layer_norm1.weight", layers.len())) {
            let p = format!("encoder.layers.{}.", layers.len());
            let a = format!("{p}self_attn.");
            layers.push(Layer {
                n1: (f.dev(gpu, &format!("{p}layer_norm1.weight"))?, f.dev(gpu, &format!("{p}layer_norm1.bias"))?),
                qkv: stack(gpu, f, [&format!("{a}q_proj"), &format!("{a}k_proj"), &format!("{a}v_proj")])?,
                o: Lin::load(f, gpu, &format!("{a}out_proj"), false)?,
                ls1: None,
                n2: (f.dev(gpu, &format!("{p}layer_norm2.weight"))?, f.dev(gpu, &format!("{p}layer_norm2.bias"))?),
                fc1: Lin::load(f, gpu, &format!("{p}mlp.fc1"), false)?,
                fc2: Lin::load(f, gpu, &format!("{p}mlp.fc2"), false)?,
                ls2: None,
            });
        }
        if layers.is_empty() {
            return Err(Error(format!("{}: no CLIP vision layers", f.path())));
        }
        let patch = Lin::load(f, gpu, "embeddings.patch_embedding", false)?;
        let width = patch.n;
        let table = f.vec("embeddings.position_embedding.weight")?;
        let cls = f.vec("embeddings.class_embedding")?;
        let tokens = table.len() / width;
        let grid = ((tokens - 1) as f64).sqrt().round() as usize;
        let mut pos = table.clone();
        for c in 0..width {
            pos[c] += cls[c];
        }
        Ok(Vit {
            kind: Kind::Clip,
            layers,
            patch,
            pos: DevBuf::from_f32(gpu, &pos)?,
            pre: Some((f.dev(gpu, "pre_layrnorm.weight")?, f.dev(gpu, "pre_layrnorm.bias")?)),
            post: (f.dev(gpu, "post_layernorm.weight")?, f.dev(gpu, "post_layernorm.bias")?),
            width,
            heads: width / 64,
            eps: 1e-5,
            size: grid * PATCH,
            grid,
        })
    }

    pub fn tokens(&self) -> usize {
        self.grid * self.grid + 1
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.qkv.bytes() + l.o.bytes() + l.fc1.bytes() + l.fc2.bytes()).sum()
    }

    /// DINOv2: the last hidden state [tokens, width]; CLIP: the pooled class token [width] - of `pixels` (float32
    /// [3, size, size], normalized)
    pub fn forward(&self, cx: &Ctx, pixels: &[f32]) -> Result<Vec<f32>> {
        let (g, c, s) = (self.grid, self.width, self.tokens());
        if pixels.len() != 3 * self.size * self.size {
            return Err(Error(format!("{:?}: {} values for a {}-pixel picture", self.kind, pixels.len(), self.size)));
        }
        let (k, gpu) = (&cx.k, &cx.gpu);
        let kk = 3 * PATCH * PATCH;
        let mut cols = vec![0f32; g * g * kk];
        for py in 0..g {
            for px in 0..g {
                let row = &mut cols[(py * g + px) * kk..(py * g + px + 1) * kk];
                for ch in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            row[ch * PATCH * PATCH + ky * PATCH + kx] = pixels[ch * self.size * self.size + (py * PATCH + ky) * self.size + px * PATCH + kx];
                        }
                    }
                }
            }
        }
        let colsd = DevBuf::new(gpu, cols.len() * 2)?;
        colsd.write(0, &half_bytes(&cols))?;
        let x = DevBuf::f32(gpu, s * c)?;
        x.fill(0)?;
        // SAFETY: rows 1.. of the [s, c] stream.
        self.patch.apply(cx, colsd.ptr(), g * g, unsafe { x.fp().add(c) }.cast(), Dt::F32)?;
        k.add(x.ptr(), Dt::F32, self.pos.ptr(), Dt::F32, s * c)?;
        if let Some((w, b)) = &self.pre {
            k.ln_mod(x.ptr(), Dt::F32, s, c, self.eps, w.fp(), b.fp(), std::ptr::null(), std::ptr::null(), x.ptr(), Dt::F32)?;
        }
        let ffn = self.layers[0].fc1.n;
        let h = DevBuf::new(gpu, s * c * 2)?;
        let qkv = DevBuf::new(gpu, s * 3 * c * 2)?;
        let att = DevBuf::new(gpu, s * c * 2)?;
        let tmp = DevBuf::f32(gpu, s * c)?;
        let mid = DevBuf::new(gpu, s * ffn * 2)?;
        let hp = |b: &DevBuf, off: usize| -> *mut std::ffi::c_void {
            // SAFETY: offsets inside the half buffer.
            unsafe { b.ptr().cast::<u16>().add(off).cast() }
        };
        let scaled_add = |x: &DevBuf, t: &DevBuf, ls: &Option<DevBuf>| -> Result<()> {
            k.gate_add(x.fp(), s, c, t.ptr(), Dt::F32, ls.as_ref().map_or(std::ptr::null(), |l| l.fp()))
        };
        for l in &self.layers {
            k.ln_mod(x.ptr(), Dt::F32, s, c, self.eps, l.n1.0.fp(), l.n1.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), Dt::F16)?;
            l.qkv.apply(cx, h.ptr(), s, qkv.ptr(), Dt::F16)?;
            cx.nsd.attention(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), Dt::F16, s, self.heads, c / self.heads, 3 * c, att.ptr(), Dt::F16)?;
            l.o.apply(cx, att.ptr(), s, tmp.ptr(), Dt::F32)?;
            scaled_add(&x, &tmp, &l.ls1)?;
            k.ln_mod(x.ptr(), Dt::F32, s, c, self.eps, l.n2.0.fp(), l.n2.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), Dt::F16)?;
            l.fc1.apply(cx, h.ptr(), s, mid.ptr(), Dt::F16)?;
            match self.kind {
                Kind::Dino => cx.nsd.gelu(mid.ptr(), Dt::F16, s * ffn, false)?,
                Kind::Clip => k.act(mid.ptr(), Dt::F16, s * ffn, 1)?,
            }
            l.fc2.apply(cx, mid.ptr(), s, tmp.ptr(), Dt::F32)?;
            scaled_add(&x, &tmp, &l.ls2)?;
        }
        let rows = if self.kind == Kind::Clip { 1 } else { s };
        let out = DevBuf::f32(gpu, rows * c)?;
        k.ln_mod(x.ptr(), Dt::F32, rows, c, self.eps, self.post.0.fp(), self.post.1.fp(), std::ptr::null(), std::ptr::null(), out.ptr(), Dt::F32)?;
        out.to_f32()
    }
}

/// Rows scaled to unit length (F.normalize, eps 1e-12)
pub fn l2_rows(v: &mut [f32], width: usize) {
    for r in v.chunks_mut(width) {
        let n = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        r.iter_mut().for_each(|x| *x /= n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_keeps_a_constant_and_a_size() {
        let v = vec![0.5f32; 2 * 37 * 37];
        let r = bicubic(&v, 2, 37, 54);
        assert_eq!(r.len(), 2 * 54 * 54);
        assert!(r.iter().all(|x| (x - 0.5).abs() < 1e-5));
        let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(bicubic(&ramp, 1, 4, 4), ramp);
    }
}
