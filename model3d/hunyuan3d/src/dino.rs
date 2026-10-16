//! The image conditioner: DINOv2 (transformers' Dinov2Model, its last hidden state) - Large (1,024 features, 24
//! layers, a GELU MLP) for Hunyuan3D 2.1 at 518 px, giant (1,536, 40 layers, SwiGLU) for 2mini at 1,022 px, where
//! the 37 x 37 position table is resized (bicubic, as transformers does) to the 73 x 73 patches.
//!
//! ```text
//!   x = [cls | patches 14 x 14 . W + b] + positions
//!   per layer:  x += ls1 * attention(ln1(x)) Wo        (q, k, v with biases; 64-feature heads)
//!               x += ls2 * mlp(ln2(x))                  (fc2(gelu(fc1)) | down(silu(gate) * up))
//!   out = ln(x)
//! ```

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ops::{fp, hp, Lin, Ops, Weights, P};

const EPS: f32 = 1e-6;
const PATCH: usize = 14;
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Layer {
    n1w: DevBuf,
    n1b: DevBuf,
    /// q | k | v rows
    qkv: Lin,
    o: Lin,
    ls1: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    /// fc1, or gate | up (SwiGLU)
    up: Lin,
    down: Lin,
    ls2: DevBuf,
}

pub struct Dino {
    layers: Vec<Layer>,
    patch: Lin,
    /// [cls + position 0 | the patches' positions], float32 [tokens, width]
    pos: DevBuf,
    lnw: DevBuf,
    lnb: DevBuf,
    pub width: usize,
    heads: usize,
    swiglu: bool,
    /// the input's side in pixels, and patches a side
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

impl Dino {
    /// The encoder under `prefix` (".../model."), for pictures of `size` pixels a side
    pub fn load(w: &Weights, size: usize) -> Result<Dino> {
        let shape = w.shape("embeddings.patch_embeddings.projection.weight")?;
        let width = shape[0];
        let heads = width / 64;
        let mut layers = Vec::new();
        let swiglu = w.has("encoder.layer.0.mlp.weights_in.weight");
        while w.has(&format!("encoder.layer.{}.norm1.weight", layers.len())) {
            let p = format!("encoder.layer.{}.", layers.len());
            let a = format!("{p}attention.attention.");
            let qkv = w.host_mat(&format!("{a}query.weight"))?.stack(&w.host_mat(&format!("{a}key.weight"))?).stack(&w.host_mat(&format!("{a}value.weight"))?);
            let mut qb = w.host_f32(&format!("{a}query.bias"))?;
            qb.extend(w.host_f32(&format!("{a}key.bias"))?);
            qb.extend(w.host_f32(&format!("{a}value.bias"))?);
            let (up, down) = if swiglu { (w.lin(&format!("{p}mlp.weights_in"))?, w.lin(&format!("{p}mlp.weights_out"))?) } else {
                (w.lin(&format!("{p}mlp.fc1"))?, w.lin(&format!("{p}mlp.fc2"))?)
            };
            layers.push(Layer {
                n1w: w.vec(&format!("{p}norm1.weight"))?,
                n1b: w.vec(&format!("{p}norm1.bias"))?,
                qkv: w.upload(&qkv, Some(qb))?,
                o: w.lin(&format!("{p}attention.output.dense"))?,
                ls1: w.vec(&format!("{p}layer_scale1.lambda1"))?,
                n2w: w.vec(&format!("{p}norm2.weight"))?,
                n2b: w.vec(&format!("{p}norm2.bias"))?,
                up,
                down,
                ls2: w.vec(&format!("{p}layer_scale2.lambda1"))?,
            });
        }
        if layers.is_empty() {
            return Err(Error(format!("no DINOv2 layers under {}", w.prefix)));
        }
        let patch = w.lin("embeddings.patch_embeddings.projection")?;
        // the position table at this size: the stored 37 x 37 resized to the patches, the class token's own in front
        let grid = size / PATCH;
        let table = w.host_f32("embeddings.position_embeddings")?;
        let cls = w.host_f32("embeddings.cls_token")?;
        let n = ((table.len() / width - 1) as f64).sqrt().round() as usize;
        if n * n + 1 != table.len() / width {
            return Err(Error(format!("a position table of {} rows", table.len() / width)));
        }
        let mut pos: Vec<f32> = (0..width).map(|c| cls[c] + table[c]).collect();
        if grid == n {
            pos.extend_from_slice(&table[width..]);
        } else {
            // [n * n, c] -> [c, n, n] -> bicubic -> [m * m, c]
            let planes: Vec<f32> = (0..width).flat_map(|c| (0..n * n).map(move |i| (c, i))).map(|(c, i)| table[(1 + i) * width + c]).collect();
            let r = bicubic(&planes, width, n, grid);
            for i in 0..grid * grid {
                pos.extend((0..width).map(|c| r[c * grid * grid + i]));
            }
        }
        let posd = DevBuf::from_f32(&w.gpu, &pos)?;
        Ok(Dino {
            layers,
            patch,
            pos: posd,
            lnw: w.vec("layernorm.weight")?,
            lnb: w.vec("layernorm.bias")?,
            width,
            heads,
            swiglu,
            size,
            grid,
        })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Tokens out: the class token and the patches
    pub fn tokens(&self) -> usize {
        self.grid * self.grid + 1
    }

    /// The last hidden state of `pixels` (float32 [3, size, size], normalized) as float32 [tokens, width]
    pub fn forward(&self, ops: &Ops, pixels: &[f32]) -> Result<DevBuf> {
        let (g, c, s) = (self.grid, self.width, self.tokens());
        if pixels.len() != 3 * self.size * self.size {
            return Err(Error(format!("DINOv2: {} values for a {}-pixel picture", pixels.len(), self.size)));
        }
        let gpu = &ops.gpu;
        // the patches as rows of (channel, ky, kx), the convolution's weight order
        let k = 3 * PATCH * PATCH;
        let mut cols = vec![0u16; g * g * k];
        for py in 0..g {
            for px in 0..g {
                let row = &mut cols[(py * g + px) * k..(py * g + px + 1) * k];
                for ch in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            let v = pixels[ch * self.size * self.size + (py * PATCH + ky) * self.size + px * PATCH + kx];
                            row[ch * PATCH * PATCH + ky * PATCH + kx] = crate::ops::f32_to_f16(v);
                        }
                    }
                }
            }
        }
        let colsd = DevBuf::new(gpu, cols.len() * 2)?;
        // SAFETY: u16s viewed as bytes.
        colsd.write(0, unsafe { std::slice::from_raw_parts(cols.as_ptr().cast::<u8>(), cols.len() * 2) })?;
        let x = DevBuf::new(gpu, s * c * 4)?;
        x.fill(0)?;
        ops.lin(&self.patch, colsd.ptr(), g * g, fp(&x, c).cast(), Dt::F32)?;
        ops.add(x.ptr(), self.pos.ptr(), false, s, c, none())?;
        let ffn = self.layers[0].up.n;
        let h = DevBuf::new(gpu, s * c * 2)?;
        let qkv = DevBuf::new(gpu, s * 3 * c * 2)?;
        let att = DevBuf::new(gpu, s * c * 2)?;
        let tmp = DevBuf::new(gpu, s * c * 2)?;
        let mid = DevBuf::new(gpu, s * ffn * 2)?;
        let act = if self.swiglu { Some(DevBuf::new(gpu, s * ffn / 2 * 2)?) } else { None };
        for l in &self.layers {
            ops.ln(x.ptr(), false, s, c, EPS, l.n1w.ptr(), l.n1b.ptr(), none(), none(), h.ptr(), true)?;
            ops.lin(&l.qkv, h.ptr(), s, qkv.ptr(), Dt::F16)?;
            ops.attention(qkv.ptr(), hp(&qkv, c), hp(&qkv, 2 * c), s, self.heads, c / self.heads, 3 * c, att.ptr())?;
            ops.lin(&l.o, att.ptr(), s, tmp.ptr(), Dt::F16)?;
            ops.add(x.ptr(), tmp.ptr(), true, s, c, l.ls1.ptr())?;
            ops.ln(x.ptr(), false, s, c, EPS, l.n2w.ptr(), l.n2b.ptr(), none(), none(), h.ptr(), true)?;
            ops.lin(&l.up, h.ptr(), s, mid.ptr(), Dt::F16)?;
            let a = match &act {
                Some(a) => {
                    ops.nsd.swiglu(mid.ptr(), Dt::F16, s, ffn / 2, a.ptr(), Dt::F16)?;
                    a.ptr() as P
                }
                None => {
                    ops.gelu(mid.ptr(), s * ffn, false)?;
                    mid.ptr() as P
                }
            };
            ops.lin(&l.down, a, s, tmp.ptr(), Dt::F16)?;
            ops.add(x.ptr(), tmp.ptr(), true, s, c, l.ls2.ptr())?;
        }
        let out = DevBuf::new(gpu, s * c * 4)?;
        ops.ln(x.ptr(), false, s, c, EPS, self.lnw.ptr(), self.lnb.ptr(), none(), none(), out.ptr(), false)?;
        Ok(out)
    }
}

/// Pixels [3, n, n] in [0, 1] normalized as DINOv2 takes them (ImageNet mean and deviation), in place
pub fn normalize(px: &mut [f32], n: usize) {
    for ch in 0..3 {
        for v in &mut px[ch * n * n..(ch + 1) * n * n] {
            *v = (*v - MEAN[ch]) / STD[ch];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_keeps_a_constant_and_a_size() {
        let v = vec![0.5f32; 2 * 37 * 37];
        let r = bicubic(&v, 2, 37, 73);
        assert_eq!(r.len(), 2 * 73 * 73);
        assert!(r.iter().all(|x| (x - 0.5).abs() < 1e-5));
        // the same size is the identity
        let ramp: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(bicubic(&ramp, 1, 4, 4), ramp);
    }
}
