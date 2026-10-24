//! The denoiser: Ming-Image's DiT, Z-Image's single-stream transformer (inclusionAI's diffusion/transformer.py
//! `DiffusionTransformer`): 30 blocks of 3,840 features (30 heads x 128), two noise-refiner blocks over the image's
//! tokens and two context-refiner blocks over the caption's, the matrices kept on the card in the GGUF's types (Q8_0
//! for the blocks, bfloat16 for the refiners: 7 GB).
//!
//! ```text
//!   image tokens: the latents' 2 x 2 patches [N, 64] -> x_embedder -> noise refiner (modulated)
//!   caption: rmsnorm + linear of the 256 query features [256, 2560], then the direct condition [L, 3840] as it is ->
//!            context refiner (not modulated; computed once a prompt)
//!   [image | caption] -> 30 blocks -> the image's rows: layernorm * (1 + scale), linear -> [N, 64]
//!   block: x += tanh(g1) * rmsnorm(attention(rmsnorm(x) * (1 + s1)))
//!          x += tanh(g2) * rmsnorm(w2(silu(w1 h) * w3 h)),  h = rmsnorm(x) * (1 + s2)
//! ```
//!
//! s1 g1 s2 g2: the block's adaLN linear of the timestep's embedding (no SiLU before it, Z-Image's), the same for every
//! token. Positions: caption token j at (1 + j, 0, 0); the image's at (the caption's length rounded up to 32, + 1,
//! row, column) - the padding the reference masks out is left out. The RoPE pairs features (2i, 2i + 1); the shared
//! kernel rotates (i, 64 + i), so q's and k's rows (and their norms) are permuted at load.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

pub const DIM: usize = 3840;
pub const HEADS: usize = 30;
pub const HEAD: usize = 128;
pub const FFN: usize = 10240;
/// a patch's features: 2 x 2 x 16 latent channels
pub const PATCH_IN: usize = 64;
pub const LATENT: usize = 16;
const EPS: f32 = 1e-5;
const ADALN: usize = 256;
const AXES: [usize; 3] = [32, 48, 48];
const THETA: f64 = 256.0;
const SEQ_MULTI_OF: usize = 32;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A matrix as stored, on the card
struct Part {
    w: DevBuf,
    qtype: u32,
    rows: usize,
    k: usize,
}

impl Part {
    fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt, ldo: usize) -> Result<()> {
        nsd.linear_q(x, Dt::F16, m, self.k, self.w.ptr(), self.qtype, self.rows, none(), out, out_dt, ldo)
    }
}

struct Block {
    q: Part,
    k: Part,
    v: Part,
    o: Part,
    w1: Part,
    w3: Part,
    w2: Part,
    norm_q: DevBuf,
    norm_k: DevBuf,
    n1: DevBuf,
    n2: DevBuf,
    f1: DevBuf,
    f2: DevBuf,
    /// the adaLN linear float32 [4 * 3840, 256] and its bias (the modulated blocks)
    ada: Option<(DevBuf, DevBuf)>,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    x_w: DevBuf,
    x_b: DevBuf,
    cap_norm: DevBuf,
    cap_w: Part,
    cap_b: DevBuf,
    t1: (Vec<f32>, Vec<f32>),
    t2: (Vec<f32>, Vec<f32>),
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    layers: Vec<Block>,
    /// the final layer: its adaLN (after a SiLU) on the host, its linear float32 on the card
    final_ada: (Vec<f32>, Vec<f32>),
    final_w: DevBuf,
    final_b: DevBuf,
    pub bytes: u64,
}

/// One block's modulation for a step: scale1, tanh(gate1), scale2, tanh(gate2), float32 [3840] each
pub struct Mod {
    s1: DevBuf,
    g1: DevBuf,
    s2: DevBuf,
    g2: DevBuf,
}

/// A step's modulations: the noise refiner's, the layers', the final layer's scale
pub struct StepMods {
    nr: Vec<Mod>,
    layers: Vec<Mod>,
    final_scale: DevBuf,
}

/// The caption's rows after the context refiner (float32 [C, 3840]) and their RoPE table, once a prompt
pub struct Prefix {
    pub rows: usize,
    pub x: DevBuf,
    cs: DevBuf,
}

/// Intermediates kept for a check
#[derive(Default)]
pub struct Grabs {
    pub nr_out: Vec<f32>,
    pub cr_out: Vec<f32>,
    pub l0_out: Vec<f32>,
}

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

fn host_vec(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{name}: a {} tensor where float was expected", other.name()))),
    })
}

/// (cos, sin) per token and pair for positions (frame, height, width), in the permuted order
fn rope_table(idx: &[[i64; 3]]) -> Vec<f32> {
    let mut cs = vec![0f32; idx.len() * HEAD];
    for (t, p) in idx.iter().enumerate() {
        let mut pair = 0;
        for (axis, dim) in AXES.iter().enumerate() {
            for i in 0..dim / 2 {
                let inv = 1.0 / THETA.powf(2.0 * i as f64 / *dim as f64);
                let a = p[axis] as f64 * inv;
                cs[t * HEAD + 2 * pair] = a.cos() as f32;
                cs[t * HEAD + 2 * pair + 1] = a.sin() as f32;
                pair += 1;
            }
        }
    }
    cs
}

/// y = W x + b on the host (W [n, k] row-major)
fn host_linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * (*c as f64)).sum::<f64>() as f32 + bi).collect()
}

fn silu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
}

impl Dit {
    pub fn load(f: &Gguf, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        let part = |name: &str, perm: bool| -> Result<Part> {
            let t = tensor(name)?;
            let code = t.ty.code();
            if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
                return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
            }
            let (rows, k) = (t.shape[0] as usize, t.shape[1] as usize);
            let mut bytes = f.read(t).map_err(ge)?;
            if perm {
                let rb = bytes.len() / rows;
                let mut out = Vec::with_capacity(bytes.len());
                for r in 0..rows {
                    let (h, j) = (r / HEAD, r % HEAD);
                    let src = h * HEAD + rope_perm(j);
                    out.extend_from_slice(&bytes[src * rb..(src + 1) * rb]);
                }
                bytes = out;
            }
            let w = DevBuf::new(&gpu, bytes.len())?;
            w.write(0, &bytes)?;
            Ok(Part { w, qtype: code, rows, k })
        };
        let vecd = |name: &str, permute: bool| -> Result<DevBuf> {
            let v = host_vec(f, name)?;
            let v = if permute { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(&gpu, &v)
        };
        let block = |p: &str, modulated: bool| -> Result<Block> {
            Ok(Block {
                q: part(&format!("{p}attention.to_q.weight"), true)?,
                k: part(&format!("{p}attention.to_k.weight"), true)?,
                v: part(&format!("{p}attention.to_v.weight"), false)?,
                o: part(&format!("{p}attention.to_out.0.weight"), false)?,
                w1: part(&format!("{p}feed_forward.w1.weight"), false)?,
                w3: part(&format!("{p}feed_forward.w3.weight"), false)?,
                w2: part(&format!("{p}feed_forward.w2.weight"), false)?,
                norm_q: vecd(&format!("{p}attention.norm_q.weight"), true)?,
                norm_k: vecd(&format!("{p}attention.norm_k.weight"), true)?,
                n1: vecd(&format!("{p}attention_norm1.weight"), false)?,
                n2: vecd(&format!("{p}attention_norm2.weight"), false)?,
                f1: vecd(&format!("{p}ffn_norm1.weight"), false)?,
                f2: vecd(&format!("{p}ffn_norm2.weight"), false)?,
                ada: if modulated {
                    Some((vecd(&format!("{p}adaLN_modulation.0.weight"), false)?, vecd(&format!("{p}adaLN_modulation.0.bias"), false)?))
                } else {
                    None
                },
            })
        };
        let count = |pre: &str| (0..).take_while(|i| f.tensor(&format!("{pre}{i}.attention.to_q.weight")).is_some()).count();
        let (nl, nr, nc) = (count("layers."), count("noise_refiner."), count("context_refiner."));
        if nl == 0 || f.tensor("all_x_embedder.2-1.weight").is_none() {
            return Err(Error(format!("{}: not Ming-Image's DiT (no layers / all_x_embedder.2-1)", f.paths[0].display())));
        }
        let dit = Dit {
            gpu: gpu.clone(),
            x_w: vecd("all_x_embedder.2-1.weight", false)?,
            x_b: vecd("all_x_embedder.2-1.bias", false)?,
            cap_norm: vecd("cap_embedder.0.weight", false)?,
            cap_w: part("cap_embedder.1.weight", false)?,
            cap_b: vecd("cap_embedder.1.bias", false)?,
            t1: (host_vec(f, "t_embedder.mlp.0.weight")?, host_vec(f, "t_embedder.mlp.0.bias")?),
            t2: (host_vec(f, "t_embedder.mlp.2.weight")?, host_vec(f, "t_embedder.mlp.2.bias")?),
            noise_refiner: (0..nr).map(|i| block(&format!("noise_refiner.{i}."), true)).collect::<Result<_>>()?,
            context_refiner: (0..nc).map(|i| block(&format!("context_refiner.{i}."), false)).collect::<Result<_>>()?,
            layers: (0..nl).map(|i| block(&format!("layers.{i}."), true)).collect::<Result<_>>()?,
            final_ada: (host_vec(f, "all_final_layer.2-1.adaLN_modulation.1.weight")?, host_vec(f, "all_final_layer.2-1.adaLN_modulation.1.bias")?),
            final_w: vecd("all_final_layer.2-1.linear.weight", false)?,
            final_b: vecd("all_final_layer.2-1.linear.bias", false)?,
            bytes: f.tensors.iter().map(|t| t.bytes).sum(),
        };
        log(format!("ming-image dit: {nl} blocks, {nr} + {nc} refiners on {} in {:.1} s", gpu.name, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The timestep's embedding for sigma `s` (the pipeline's 1 - s, times t_scale 1000): sinusoid [cos | sin] of 256,
    /// linear, SiLU, linear
    fn t_emb(&self, s: f32) -> Vec<f32> {
        let t = (1.0 - s as f64) * 1000.0;
        let mut e = vec![0f32; ADALN];
        for i in 0..ADALN / 2 {
            let a = t * (-(10000f64).ln() * i as f64 / (ADALN / 2) as f64).exp();
            e[i] = a.cos() as f32;
            e[ADALN / 2 + i] = a.sin() as f32;
        }
        let mut h = host_linear(&self.t1.0, &self.t1.1, &e);
        silu(&mut h);
        host_linear(&self.t2.0, &self.t2.1, &h)
    }

    /// Each step's modulations for the levels `sigmas` (the steps run: all but the last level)
    pub fn mods(&self, nsd: &Nsd, sigmas: &[f32]) -> Result<Vec<StepMods>> {
        let gpu = &self.gpu;
        let steps = sigmas.len().saturating_sub(1);
        let embs: Vec<Vec<f32>> = sigmas[..steps].iter().map(|s| self.t_emb(*s)).collect();
        let e = DevBuf::from_f32(gpu, &embs.concat())?;
        let one = |b: &Block| -> Result<Vec<Mod>> {
            let (w, bias) = b.ada.as_ref().ok_or_else(|| Error("an unmodulated block".into()))?;
            let out = DevBuf::f32(gpu, steps * 4 * DIM)?;
            nsd.linear(e.ptr(), Dt::F32, steps, ADALN, w.ptr(), 4 * DIM, bias.fp().cast(), out.ptr(), Dt::F32)?;
            let v = out.to_f32()?;
            (0..steps).map(|i| {
                let r = &v[i * 4 * DIM..(i + 1) * 4 * DIM];
                let tanh = |x: &[f32]| x.iter().map(|v| v.tanh()).collect::<Vec<f32>>();
                Ok(Mod {
                    s1: DevBuf::from_f32(gpu, &r[..DIM])?,
                    g1: DevBuf::from_f32(gpu, &tanh(&r[DIM..2 * DIM]))?,
                    s2: DevBuf::from_f32(gpu, &r[2 * DIM..3 * DIM])?,
                    g2: DevBuf::from_f32(gpu, &tanh(&r[3 * DIM..]))?,
                })
            }).collect()
        };
        let per_nr: Vec<Vec<Mod>> = self.noise_refiner.iter().map(one).collect::<Result<_>>()?;
        let per_l: Vec<Vec<Mod>> = self.layers.iter().map(one).collect::<Result<_>>()?;
        let mut nr_it: Vec<std::vec::IntoIter<Mod>> = per_nr.into_iter().map(|v| v.into_iter()).collect();
        let mut l_it: Vec<std::vec::IntoIter<Mod>> = per_l.into_iter().map(|v| v.into_iter()).collect();
        let mut out = Vec::with_capacity(steps);
        for emb in &embs {
            let mut se = emb.clone();
            silu(&mut se);
            let fs = host_linear(&self.final_ada.0, &self.final_ada.1, &se);
            out.push(StepMods {
                nr: nr_it.iter_mut().map(|i| i.next().expect("a step's modulation")).collect(),
                layers: l_it.iter_mut().map(|i| i.next().expect("a step's modulation")).collect(),
                final_scale: DevBuf::from_f32(gpu, &fs)?,
            });
        }
        Ok(out)
    }

    /// One block over x float32 [m, 3840] in place; `md`: its modulation (None: an unmodulated refiner)
    #[allow(clippy::too_many_arguments)]
    fn block(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, cs: &DevBuf, md: Option<&Mod>, t: &Tmp) -> Result<()> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let qw = 3 * DIM;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        let nul = std::ptr::null::<f32>();
        let (s1, g1, s2, g2) = match md {
            Some(v) => (v.s1.fp().cast_const(), v.g1.fp().cast_const(), v.s2.fp().cast_const(), v.g2.fp().cast_const()),
            None => (nul, nul, nul, nul),
        };
        // SAFETY: x holds m x 3840 floats, t.h as many halves; the weights and scales 3840 floats.
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), m as i64, DIM as i64, b.n1.fp(), EPS, s1, t.h.ptr()) }, "norm")?;
        // q, k and v float32 one at a time, put in half scaled by ACT_SCALE: v reaches past half's range in the last
        // blocks (a prompt made infinities in block 28); q and k are rms-normed after (their scale goes), v's comes
        // back through the attention and the output linear (multiplied back below)
        for (i, p) in [&b.q, &b.k, &b.v].into_iter().enumerate() {
            p.apply(nsd, t.h.ptr(), m, t.of.ptr(), Dt::F32, 0)?;
            // SAFETY: of holds m x 3840 floats; qkv m rows of qw halves.
            ffi::check(unsafe { (k.to_half_rows)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, ffi::ACT_SCALE, half_at(&t.qkv, i * DIM), qw as i64) }, "q k v")?;
        }
        // (the norm of q s with eps s^2 is the norm of q with eps: the scale is exact)
        let eps = EPS * ffi::ACT_SCALE * ffi::ACT_SCALE;
        nsd.rms_rope(t.qkv.ptr(), Dt::F16, m, HEADS, HEAD, qw, b.norm_q.ptr(), eps, cs.ptr(), HEAD)?;
        nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, m, HEADS, HEAD, qw, b.norm_k.ptr(), eps, cs.ptr(), HEAD)?;
        nsd.attention_qk(t.qkv.ptr(), m, qw, half_at(&t.qkv, DIM), half_at(&t.qkv, 2 * DIM), m, qw, Dt::F16, HEADS, HEAD, t.att.ptr(), Dt::F16)?;
        b.o.apply(nsd, t.att.ptr(), m, t.of.ptr(), Dt::F32, 0)?;
        nsd.scale(t.of.ptr(), m * DIM, 1.0 / ffi::ACT_SCALE)?;
        // SAFETY: as above.
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, b.n2.fp(), EPS, std::ptr::null(), t.h.ptr()) }, "norm")?;
        ffi::check(unsafe { (k.gate_add)(gpu.raw(), x.fp(), m as i64, DIM as i64, t.h.ptr(), g1) }, "gate")?;
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), m as i64, DIM as i64, b.f1.fp(), EPS, s2, t.h.ptr()) }, "norm")?;
        b.w1.apply(nsd, t.h.ptr(), m, t.gp.ptr(), Dt::F16, 2 * FFN)?;
        b.w3.apply(nsd, t.h.ptr(), m, half_at(&t.gp, FFN), Dt::F16, 2 * FFN)?;
        // the gated product past half's range here (a layer makes infinities): scaled down, multiplied back after w2
        ffi::swiglu(gpu, t.gp.ptr(), m, FFN, t.act.ptr())?;
        b.w2.apply(nsd, t.act.ptr(), m, t.of.ptr(), Dt::F32, 0)?;
        nsd.scale(t.of.ptr(), m * DIM, 1.0 / ffi::ACT_SCALE)?;
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, b.f2.fp(), EPS, std::ptr::null(), t.h.ptr()) }, "norm")?;
        ffi::check(unsafe { (k.gate_add)(gpu.raw(), x.fp(), m as i64, DIM as i64, t.h.ptr(), g2) }, "gate")
    }

    /// The caption's rows: the query features (float32 [256, 2560]) embedded, the direct condition ([L, 3840]) after
    /// them, through the context refiner
    pub fn prefix(&self, nsd: &Nsd, cap: &[f32], direct: &[f32], direct_rows: usize, grabs: Option<&mut Grabs>) -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let nq = cap.len() / self.cap_w.k;
        let c = nq + direct_rows;
        let cf = DevBuf::from_f32(gpu, cap)?;
        let ch = DevBuf::new(gpu, cap.len() * 2)?;
        // SAFETY: nq x 2560 values; the weight 2560 floats.
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), cf.fp(), nq as i64, self.cap_w.k as i64, self.cap_norm.fp(), EPS, std::ptr::null(), ch.ptr()) }, "norm")?;
        let x = DevBuf::f32(gpu, c * DIM)?;
        nsd.linear_q(ch.ptr(), Dt::F16, nq, self.cap_w.k, self.cap_w.w.ptr(), self.cap_w.qtype, DIM, self.cap_b.fp().cast(), x.ptr(), Dt::F32, 0)?;
        let d = DevBuf::from_f32(gpu, direct)?;
        x.copy_within(nq * DIM * 4, &d, 0, direct_rows * DIM * 4)?;
        let idx: Vec<[i64; 3]> = (0..c).map(|j| [1 + j as i64, 0, 0]).collect();
        let cs = DevBuf::from_f32(gpu, &rope_table(&idx))?;
        let t = Tmp::new(gpu, c)?;
        for b in &self.context_refiner {
            self.block(nsd, b, &x, c, &cs, None, &t)?;
        }
        nsd.wait()?;
        if let Some(g) = grabs {
            g.cr_out = x.to_f32()?;
        }
        Ok(Prefix { rows: c, x, cs })
    }

    /// The model's output for the image's patches `lat` (float32 [hh * ww, 64], row-major patches of (2, 2, 16)) at a
    /// step with modulations `md`: float32 [hh * ww, 64] (the reference's, before the pipeline negates it)
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), md: &StepMods, mut grabs: Option<&mut Grabs>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let c = pre.rows;
        let s = n + c;
        let x = DevBuf::f32(gpu, s * DIM)?;
        nsd.linear(lat.ptr(), Dt::F32, n, PATCH_IN, self.x_w.ptr(), DIM, self.x_b.fp().cast(), x.ptr(), Dt::F32)?;
        // positions: the image at the frame after the caption (its length rounded up to 32), rows and columns from 0
        let frame = (c.div_ceil(SEQ_MULTI_OF) * SEQ_MULTI_OF + 1) as i64;
        let idx: Vec<[i64; 3]> = (0..n).map(|i| [frame, (i / hw.1) as i64, (i % hw.1) as i64]).collect();
        let mut cs_all = rope_table(&idx);
        let cs_img = DevBuf::from_f32(gpu, &cs_all)?;
        let t = Tmp::new(gpu, s)?;
        for (b, m) in self.noise_refiner.iter().zip(&md.nr) {
            self.block(nsd, b, &x, n, &cs_img, Some(m), &t)?;
        }
        if let Some(g) = grabs.as_deref_mut() {
            nsd.wait()?;
            g.nr_out = x.to_f32()?[..n * DIM].to_vec();
        }
        // [image | caption]
        x.copy_within(n * DIM * 4, &pre.x, 0, c * DIM * 4)?;
        let cap_cs = pre.cs.to_f32()?;
        cs_all.extend_from_slice(&cap_cs);
        let cs = DevBuf::from_f32(gpu, &cs_all)?;
        for (i, (b, m)) in self.layers.iter().zip(&md.layers).enumerate() {
            self.block(nsd, b, &x, s, &cs, Some(m), &t)?;
            if i == 0 {
                if let Some(g) = grabs.as_deref_mut() {
                    nsd.wait()?;
                    g.l0_out = x.to_f32()?;
                }
            }
        }
        // the final layer on the image's rows
        // SAFETY: n x 3840 values each way; the scale 3840 floats.
        ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, 1e-6, md.final_scale.fp(), t.h.ptr()) }, "final norm")?;
        let hf = DevBuf::f32(gpu, n * DIM)?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), t.h.ptr(), hf.fp(), (n * DIM) as i64) }, "to float")?;
        let v = DevBuf::f32(gpu, n * PATCH_IN)?;
        nsd.linear(hf.ptr(), Dt::F32, n, DIM, self.final_w.ptr(), PATCH_IN, self.final_b.fp().cast(), v.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(v)
    }
}

/// A pass's buffers, for up to `rows` tokens
struct Tmp {
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    of: DevBuf,
    gp: DevBuf,
    act: DevBuf,
}

impl Tmp {
    fn new(gpu: &Arc<Gpu>, rows: usize) -> Result<Tmp> {
        Ok(Tmp {
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            of: DevBuf::f32(gpu, rows * DIM)?,
            gp: DevBuf::new(gpu, rows * 2 * FFN * 2)?,
            act: DevBuf::new(gpu, rows * FFN * 2)?,
        })
    }
}

/// The latents [16, H, W] (H, W even) as the DiT's patches [H/2 * W/2, 64]: feature (ph * 2 + pw) * 16 + c
pub fn patchify(lat: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let mut out = vec![0f32; hh * ww * PATCH_IN];
    for y in 0..hh {
        for x in 0..ww {
            for ph in 0..2 {
                for pw in 0..2 {
                    for c in 0..LATENT {
                        out[(y * ww + x) * PATCH_IN + (ph * 2 + pw) * LATENT + c] = lat[c * h * w + (2 * y + ph) * w + 2 * x + pw];
                    }
                }
            }
        }
    }
    out
}

/// The inverse: patches [H/2 * W/2, 64] to latents [16, H, W]
pub fn unpatchify(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let mut out = vec![0f32; LATENT * h * w];
    for y in 0..hh {
        for x in 0..ww {
            for ph in 0..2 {
                for pw in 0..2 {
                    for c in 0..LATENT {
                        out[c * h * w + (2 * y + ph) * w + 2 * x + pw] = p[(y * ww + x) * PATCH_IN + (ph * 2 + pw) * LATENT + c];
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn patches_go_and_come_back() {
        let (h, w) = (6, 4);
        let lat: Vec<f32> = (0..16 * h * w).map(|i| i as f32).collect();
        let p = super::patchify(&lat, h, w);
        assert_eq!(p.len(), 3 * 2 * 64);
        // patch (0, 0): (ph 0, pw 1, c 0) is pixel (0, 1) of channel 0
        assert_eq!(p[16], 1.0);
        // (ph 1, pw 0, c 2): pixel (1, 0) of channel 2
        assert_eq!(p[2 * 16 + 2], (2 * h * w + w) as f32);
        assert_eq!(super::unpatchify(&p, h, w), lat);
    }

    #[test]
    fn rope_pairs_follow_the_axes() {
        let cs = super::rope_table(&[[3, 0, 0], [0, 5, 0]]);
        // token 0: the frame axis's first pair turns by 3, the height pairs not at all
        assert!((cs[0] - 3f32.cos()).abs() < 1e-6 && (cs[1] - 3f32.sin()).abs() < 1e-6);
        assert!((cs[2 * 16] - 1.0).abs() < 1e-6);
        // token 1: the first height pair (pair 16) turns by 5
        assert!((cs[128 + 2 * 16] - 5f32.cos()).abs() < 1e-6);
    }
}
