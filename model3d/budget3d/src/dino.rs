//! DINOv3 ViT-H+/16 (facebook/dinov3-vith16plus-pretrain-lvd1689m, transformers' DINOv3ViTModel) as Idol's ImageEncoder
//! runs it: the 1024 x 1024 picture ImageNet-normalized, the CLS token, 4 registers and 64 x 64 patches, 32 layers of
//! 1280 (20 heads of 64; rotary positions on the patches: their centres in [-1, 1], 16 frequencies of base 100 per
//! axis, half-split pairs; layer scales; a SwiGLU MLP of 5120), the model's own final norm (last_hidden_state), then
//! Idol's parameter-free layer norm. float32 matrices (Idol runs it in bfloat16), attention in half.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::ffi::Rows;
use crate::w::{self, Lin, St};
use crate::Ctx;

pub const DIM: usize = 1280;
const HEADS: usize = 20;
const HD: usize = 64;
const FF: usize = 5120;
const LAYERS: usize = 32;
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
    /// gate_proj and up_proj side by side [2 x 5120, 1280]
    gu: Lin,
    down: Lin,
    ls2: DevBuf,
}

pub struct Dino {
    embed: Lin,
    prefix: Vec<f32>,
    layers: Vec<Layer>,
    norm: (DevBuf, DevBuf),
}

impl Dino {
    pub fn load(f: &St, cx: &Ctx) -> Result<Dino> {
        let g = &cx.gpu;
        // the transformers checkpoint: each part with or without a "model." prefix (transformers 5 names the layers
        // model.layer.N, Meta's converted files layer.N)
        let pre_of = |name: &str| -> Option<&'static str> {
            if f.has(name) { Some("") } else if f.has(&format!("model.{name}")) { Some("model.") } else { None }
        };
        let not = || Error(format!("{}: not a DINOv3 ViT (transformers' DINOv3ViTModel)", f.path()));
        let pre = pre_of("embeddings.cls_token").ok_or_else(not)?;
        let lpre = pre_of("layer.0.norm1.weight").ok_or_else(not)?;
        let npre = pre_of("norm.weight").ok_or_else(not)?;
        let hidden = f.shape(&format!("{pre}embeddings.cls_token"))?.last().copied().unwrap_or(0);
        if hidden != DIM || !f.has(&format!("{lpre}layer.{}.mlp.gate_proj.weight", LAYERS - 1)) {
            return Err(Error(format!("{}: not DINOv3 ViT-H+/16 (width {hidden}; Idol reads the 1280-wide, 32-layer SwiGLU model)", f.path())));
        }
        let ln = |p: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(g, &f.vec_n(&format!("{p}.weight"), DIM)?)?, DevBuf::from_f32(g, &f.vec_n(&format!("{p}.bias"), DIM)?)?))
        };
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = format!("{lpre}layer.{i}");
            let a = format!("{p}.attention");
            let mut wq = f.vec_n(&format!("{a}.q_proj.weight"), DIM * DIM)?;
            wq.extend(f.vec_n(&format!("{a}.k_proj.weight"), DIM * DIM)?);
            wq.extend(f.vec_n(&format!("{a}.v_proj.weight"), DIM * DIM)?);
            let bias = |n: &str| -> Result<Vec<f32>> {
                let k = format!("{a}.{n}_proj.bias");
                if f.has(&k) { f.vec_n(&k, DIM) } else { Ok(vec![0.0; DIM]) }
            };
            let b = [bias("q")?, bias("k")?, bias("v")?].concat();
            let mut gu = f.vec_n(&format!("{p}.mlp.gate_proj.weight"), FF * DIM)?;
            gu.extend(f.vec_n(&format!("{p}.mlp.up_proj.weight"), FF * DIM)?);
            let mbias = |n: &str| -> Result<Vec<f32>> {
                let k = format!("{p}.mlp.{n}.bias");
                if f.has(&k) { f.vec_n(&k, FF) } else { Ok(vec![0.0; FF]) }
            };
            let gub = [mbias("gate_proj")?, mbias("up_proj")?].concat();
            layers.push(Layer {
                n1: ln(&format!("{p}.norm1"))?,
                qkv: Lin::from_host(g, &wq, Some(&b), 3 * DIM, DIM, false)?,
                o: Lin::load(f, g, &format!("{a}.o_proj"), false)?,
                ls1: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.layer_scale1.lambda1"), DIM)?)?,
                n2: ln(&format!("{p}.norm2"))?,
                gu: Lin::from_host(g, &gu, Some(&gub), 2 * FF, DIM, false)?,
                down: Lin::load(f, g, &format!("{p}.mlp.down_proj"), false)?,
                ls2: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.layer_scale2.lambda1"), DIM)?)?,
            });
        }
        let mut prefix = f.vec_n(&format!("{pre}embeddings.cls_token"), DIM)?;
        prefix.extend(f.vec_n(&format!("{pre}embeddings.register_tokens"), 4 * DIM)?);
        Ok(Dino { embed: Lin::load(f, g, &format!("{pre}embeddings.patch_embeddings"), false)?, prefix, layers, norm: ln(&format!("{npre}norm"))? })
    }

    /// The tokens [5 + (size / 16)^2, 1280] of a picture's planes [3, size, size] in [0, 1]
    pub fn encode(&self, cx: &Ctx, x: &[f32], size: usize) -> Result<Vec<f32>> {
        let g = &cx.gpu;
        let np = size / PATCH;
        let p = np * np;
        let t = PREFIX + p;
        if x.len() != 3 * size * size || !size.is_multiple_of(PATCH) {
            return Err(Error(format!("DINOv3: {} input values for {size} x {size}", x.len())));
        }
        // the normalized patches as rows in the convolution weight's (c, ky, kx) order
        let mut patches = vec![0f32; p * 3 * PATCH * PATCH];
        for py in 0..np {
            for px in 0..np {
                let row = &mut patches[(py * np + px) * 768..(py * np + px + 1) * 768];
                for c in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            let v = x[(c * size + py * PATCH + ky) * size + px * PATCH + kx];
                            row[(c * PATCH + ky) * PATCH + kx] = (v - MEAN[c]) / STD[c];
                        }
                    }
                }
            }
        }
        let pd = DevBuf::from_f32(g, &patches)?;
        let h = DevBuf::f32(g, t * DIM)?;
        h.write(0, &w::f32_bytes(&self.prefix))?;
        let (nsd, k) = (&cx.nsd, &cx.k);
        // SAFETY: rows 5 .. t of h.
        self.embed.apply(nsd, k, pd.ptr(), p, unsafe { h.fp().add(PREFIX * DIM) }.cast(), Dt::F32)?;
        // rotary angles: 2 pi * centre * 100^(-4 j / 64), y then x, as (cos, sin) for pairs (i, i + 32)
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
        let gu = DevBuf::f32(g, t * 2 * FF)?;
        let mlp = DevBuf::f32(g, t * FF)?;
        for l in &self.layers {
            nsd.layer_norm(h.ptr(), Dt::F32, t, DIM, l.n1.0.ptr(), l.n1.1.ptr(), 1e-5, hn.ptr(), Dt::F32)?;
            l.qkv.apply(nsd, k, hn.ptr(), t, qkv.ptr(), Dt::F32)?;
            // SAFETY: q and k of the patch rows (from row 5) inside qkv.
            unsafe {
                nsd.rope(qkv.fp().add(PREFIX * 3 * DIM).cast(), Dt::F32, p, HEADS, HD, 3 * DIM, cs.ptr(), HD)?;
                nsd.rope(qkv.fp().add(PREFIX * 3 * DIM + DIM).cast(), Dt::F32, p, HEADS, HD, 3 * DIM, cs.ptr(), HD)?;
            }
            k.convert(qkv.ptr(), Dt::F32, t * 3 * DIM, qkvh.ptr(), Dt::F16)?;
            let kp = qkvh.ptr().cast::<u16>().wrapping_add(DIM).cast();
            let vp = qkvh.ptr().cast::<u16>().wrapping_add(2 * DIM).cast();
            nsd.attention_qk(qkvh.ptr(), t, 3 * DIM, kp, vp, t, 3 * DIM, Dt::F16, HEADS, HD, att.ptr(), Dt::F16)?;
            k.convert(att.ptr(), Dt::F16, t * DIM, attf.ptr(), Dt::F32)?;
            l.o.apply(nsd, k, attf.ptr(), t, hn.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), t, DIM, hn.ptr(), Dt::F32, l.ls1.fp(), 0, 0)?;
            nsd.layer_norm(h.ptr(), Dt::F32, t, DIM, l.n2.0.ptr(), l.n2.1.ptr(), 1e-5, hn.ptr(), Dt::F32)?;
            l.gu.apply(nsd, k, hn.ptr(), t, gu.ptr(), Dt::F32)?;
            nsd.swiglu(gu.ptr(), Dt::F32, t, FF, mlp.ptr(), Dt::F32)?;
            l.down.apply(nsd, k, mlp.ptr(), t, hn.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), t, DIM, hn.ptr(), Dt::F32, l.ls2.fp(), 0, 0)?;
        }
        // the model's norm, then Idol's parameter-free one
        nsd.layer_norm(h.ptr(), Dt::F32, t, DIM, self.norm.0.ptr(), self.norm.1.ptr(), 1e-5, hn.ptr(), Dt::F32)?;
        k.ln_mod(hn.ptr(), Dt::F32, t, DIM, 1e-5, std::ptr::null(), std::ptr::null(), Rows::none(), h.ptr(), Dt::F32)?;
        g.sync()?;
        h.to_f32()
    }
}
