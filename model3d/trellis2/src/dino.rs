//! DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m, transformers' DINOv3ViTModel) as TRELLIS.2's
//! DinoV3FeatureExtractor runs it: the picture resized to the working size (512 or 1024, PIL's Lanczos), ImageNet
//! normalisation, the CLS token, 4 registers and the 16 x 16 patches, 24 layers (rotary positions on the patches:
//! their centres in [-1, 1], 16 frequencies of rope_theta 100 per axis, half-split pairs; layer scales), then a
//! parameter-free layer norm (not the model's own). The linears in float32 (as the reference), attention in half.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt};
use nextsycl_diffusion::Picture;

use crate::w::{File, Keep, Lin};
use crate::Ctx;

pub const DIM: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const PATCH: usize = 16;
const PREFIX: usize = 5;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Layer {
    n1: (DevBuf, DevBuf),
    qkv: Lin,
    o: Lin,
    ls1: DevBuf,
    n2: (DevBuf, DevBuf),
    up: Lin,
    down: Lin,
    ls2: DevBuf,
}

pub struct Dino {
    embed: Lin,
    prefix: Vec<f32>,
    layers: Vec<Layer>,
}

/// The normalized picture the model sees at `size`: [3, size, size] planes (the reference's tensor)
pub fn input(pic: &Picture, size: usize) -> Vec<f32> {
    let r = pic.rgb().resize_lanczos(size as u32, size as u32);
    let mut out = vec![0f32; 3 * size * size];
    for (i, px) in r.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * size * size + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

impl Dino {
    pub fn load(f: &File, cx: &Ctx) -> Result<Dino> {
        let g = &cx.gpu;
        let ln = |p: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(g, &f.vec(&format!("{p}.weight"))?)?, DevBuf::from_f32(g, &f.vec(&format!("{p}.bias"))?)?))
        };
        let mut layers = Vec::new();
        for i in 0..24 {
            let p = format!("layer.{i}");
            let (q, k, v) = (f.vec(&format!("{p}.attention.q_proj.weight"))?, f.vec(&format!("{p}.attention.k_proj.weight"))?,
                             f.vec(&format!("{p}.attention.v_proj.weight"))?);
            let mut w = q;
            w.extend(k);
            w.extend(v);
            let mut b = f.vec(&format!("{p}.attention.q_proj.bias"))?;
            b.extend(std::iter::repeat_n(0.0, DIM));       // no key bias
            b.extend(f.vec(&format!("{p}.attention.v_proj.bias"))?);
            layers.push(Layer {
                n1: ln(&format!("{p}.norm1"))?,
                qkv: Lin::from_host(g, &w, Some(&b), 3 * DIM, DIM, Keep::F32)?,
                o: Lin::load(f, g, &format!("{p}.attention.o_proj"), Keep::F32)?,
                ls1: DevBuf::from_f32(g, &f.vec(&format!("{p}.layer_scale1.lambda1"))?)?,
                n2: ln(&format!("{p}.norm2"))?,
                up: Lin::load(f, g, &format!("{p}.mlp.up_proj"), Keep::F32)?,
                down: Lin::load(f, g, &format!("{p}.mlp.down_proj"), Keep::F32)?,
                ls2: DevBuf::from_f32(g, &f.vec(&format!("{p}.layer_scale2.lambda1"))?)?,
            });
        }
        let mut prefix = f.vec("embeddings.cls_token")?;
        prefix.extend(f.vec("embeddings.register_tokens")?);
        if prefix.len() != PREFIX * DIM {
            return Err(Error(format!("{}: {} prefix values, not CLS + 4 registers", f.path(), prefix.len())));
        }
        Ok(Dino { embed: Lin::load(f, g, "embeddings.patch_embeddings", Keep::F32)?, prefix, layers })
    }

    /// The features [5 + (size / 16)^2, 1024] float32 (on the GPU) of a normalized picture [3, size, size]
    pub fn encode(&self, cx: &Ctx, x: &[f32], size: usize) -> Result<DevBuf> {
        let g = &cx.gpu;
        let np = size / PATCH;
        let p = np * np;
        let t = PREFIX + p;
        if x.len() != 3 * size * size {
            return Err(Error(format!("DINOv3: {} input values for {size} x {size}", x.len())));
        }
        // the patches as rows in the convolution weight's (c, ky, kx) order
        let mut patches = vec![0f32; p * 3 * PATCH * PATCH];
        for py in 0..np {
            for px in 0..np {
                let row = &mut patches[(py * np + px) * 768..(py * np + px + 1) * 768];
                for c in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            row[(c * PATCH + ky) * PATCH + kx] = x[(c * size + py * PATCH + ky) * size + px * PATCH + kx];
                        }
                    }
                }
            }
        }
        let pd = DevBuf::from_f32(g, &patches)?;
        let h = DevBuf::f32(g, t * DIM)?;
        h.write(0, &crate::w::f32_bytes(&self.prefix))?;
        // SAFETY: rows 5 .. t of h.
        self.embed.apply(&cx.nsd, &cx.k, &h, pd.ptr(), p, unsafe { h.fp().add(PREFIX * DIM) }.cast(), Dt::F32)?;
        // rotary angles: 2 pi * centre * 100^(-j/16), y then x, as (cos, sin) for pairs (i, i + 32)
        let mut cs = vec![0f32; p * 32 * 2];
        for py in 0..np {
            for px in 0..np {
                let cy = ((py as f32 + 0.5) / np as f32) * 2.0 - 1.0;
                let cxv = ((px as f32 + 0.5) / np as f32) * 2.0 - 1.0;
                for i in 0..32 {
                    let inv = 1.0 / 100f32.powf((i % 16) as f32 * 4.0 / HD as f32);
                    let a = 2.0 * std::f32::consts::PI * if i < 16 { cy } else { cxv } * inv;
                    let at = ((py * np + px) * 32 + i) * 2;
                    cs[at] = a.cos();
                    cs[at + 1] = a.sin();
                }
            }
        }
        let cs = DevBuf::from_f32(g, &cs)?;
        let hn = DevBuf::f32(g, t * DIM)?;
        let qkv = DevBuf::f32(g, t * 3 * DIM)?;
        let qkvh = DevBuf::new(g, t * 3 * DIM * 2)?;
        let att = DevBuf::new(g, t * DIM * 2)?;
        let attf = DevBuf::f32(g, t * DIM)?;
        let mlp = DevBuf::f32(g, t * 4 * DIM)?;
        let (nsd, k) = (&cx.nsd, &cx.k);
        for l in &self.layers {
            nsd.layer_norm(h.ptr(), Dt::F32, t, DIM, l.n1.0.ptr(), l.n1.1.ptr(), 1e-5, hn.ptr(), Dt::F32)?;
            l.qkv.apply(nsd, k, &qkv, hn.ptr(), t, qkv.ptr(), Dt::F32)?;
            // SAFETY: q and k of the patch rows (from row 5) inside qkv.
            unsafe {
                nsd.rope(qkv.fp().add(PREFIX * 3 * DIM).cast(), Dt::F32, p, HEADS, HD, 3 * DIM, cs.ptr(), HD)?;
                nsd.rope(qkv.fp().add(PREFIX * 3 * DIM + DIM).cast(), Dt::F32, p, HEADS, HD, 3 * DIM, cs.ptr(), HD)?;
            }
            k.convert(qkv.ptr(), Dt::F32, t * 3 * DIM, qkvh.ptr(), Dt::F16)?;
            // SAFETY: k and v columns inside qkvh's rows.
            let (kp, vp) = unsafe { (qkvh.ptr().cast::<u16>().add(DIM).cast(), qkvh.ptr().cast::<u16>().add(2 * DIM).cast()) };
            nsd.attention_qk(qkvh.ptr(), t, 3 * DIM, kp, vp, t, 3 * DIM, Dt::F16, HEADS, HD, att.ptr(), Dt::F16)?;
            k.convert(att.ptr(), Dt::F16, t * DIM, attf.ptr(), Dt::F32)?;
            l.o.apply(nsd, k, &hn, attf.ptr(), t, hn.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), t, DIM, hn.ptr(), Dt::F32, l.ls1.fp())?;
            nsd.layer_norm(h.ptr(), Dt::F32, t, DIM, l.n2.0.ptr(), l.n2.1.ptr(), 1e-5, hn.ptr(), Dt::F32)?;
            l.up.apply(nsd, k, &mlp, hn.ptr(), t, mlp.ptr(), Dt::F32)?;
            nsd.gelu(mlp.ptr(), Dt::F32, t * 4 * DIM, false)?;
            l.down.apply(nsd, k, &hn, mlp.ptr(), t, hn.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), t, DIM, hn.ptr(), Dt::F32, l.ls2.fp())?;
        }
        let out = DevBuf::f32(g, t * DIM)?;
        k.ln_mod(h.ptr(), Dt::F32, t, DIM, 1e-5, none().cast(), none().cast(), none().cast(), none().cast(), out.ptr(), Dt::F32)?;
        Ok(out)
    }
}
