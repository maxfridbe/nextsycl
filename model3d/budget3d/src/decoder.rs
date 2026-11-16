//! Idol's decoders, unchanged by Budget3D: the latent [8192, 16] to Gaussians.
//!
//! ```text
//!   OctreeProbabilityFixedlenDecoder: from the root cell, 8 levels: each cell's 8 children logits (4 cross-only
//!     blocks over the latent, modulated by the level) -> softmax -> the cell's points split among its children by
//!     systematic resampling; the leaves' points land at random inside their 1/256 cells: the anchors
//!   GaussianFixedlenDecoder: 16 blocks (self-attention over the anchors, cross-attention to the latent, MLP) ->
//!     480 numbers an anchor = 32 Gaussians (offset, colour, scale, rotation, opacity, offset scale)
//! ```
//! Half blocks (Idol keeps them half), float32 residual stream, the input and output layers in float32.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;
use nextsycl_model3d::Splats;

use crate::ffi::Rows;
use crate::w::{self, sinusoid, HostLin, Lin, St};
use crate::Ctx;

const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const MLP: usize = 4096;
/// the latent: 8192 tokens of 16
const LAT: usize = 8192;
const LC: usize = 16;
/// Gaussians an anchor
pub const PER: usize = 32;
/// the octree's depth (config.json max_voxel_level)
pub const LEVELS: u32 = 8;

/// Idol's PcdAbsolutePositionEmbedderv2 of points in [0, 1]^3: [sin, cos](pi x 2^(10 i / 169)) an axis, padded to 1024
fn pos_v2(x: &[[f32; 3]]) -> Vec<f32> {
    let fd = D / 3 / 2;
    let freqs: Vec<f32> = (0..fd).map(|i| 2f32.powf(10.0 * i as f32 / (fd - 1) as f32)).collect();
    let mut out = vec![0f32; x.len() * D];
    for (p, row) in x.iter().zip(out.chunks_exact_mut(D)) {
        for a in 0..3 {
            for (i, f) in freqs.iter().enumerate() {
                let v = p[a] * f * std::f32::consts::PI;
                row[a * 2 * fd + i] = v.sin();
                row[a * 2 * fd + fd + i] = v.cos();
            }
        }
    }
    out
}

/// A cross-attention's matrices (the decoders' MultiHeadAttention, cross)
struct Cross {
    q: Lin,
    kv: Lin,
    q_gamma: DevBuf,
    k_gamma: DevBuf,
    out: Lin,
}

impl Cross {
    fn load(f: &St, g: &Arc<Gpu>, p: &str) -> Result<Cross> {
        Ok(Cross {
            q: Lin::load(f, g, &format!("{p}.to_q"), true)?,
            kv: Lin::load(f, g, &format!("{p}.to_kv"), true)?,
            q_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.q_rms_norm.gamma"), D)?)?,
            k_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.k_rms_norm.gamma"), D)?)?,
            out: Lin::load(f, g, &format!("{p}.to_out"), true)?,
        })
    }

    /// Its keys (RMS-normed) and values for the latent (half [8192, 16]): half [8192, 2048]
    fn kv_of(&self, cx: &Ctx, lat: &DevBuf) -> Result<DevBuf> {
        let kv = DevBuf::new(&cx.gpu, LAT * 2 * D * 2)?;
        self.kv.apply(&cx.nsd, &cx.k, lat.ptr(), LAT, kv.ptr(), Dt::F16)?;
        cx.k.rope_norm(kv.ptr(), LAT, HEADS, HD, 2 * D, std::ptr::null(), std::ptr::null(), 0, 0, self.k_gamma.fp())?;
        Ok(kv)
    }

    /// x [m, 1024] (half, in hn) attending to kv: the output projection in float32 into tmp
    fn apply(&self, cx: &Ctx, b: &Buf, m: usize, kv: &DevBuf) -> Result<()> {
        let (nsd, k) = (&cx.nsd, &cx.k);
        self.q.apply(nsd, k, b.hn.ptr(), m, b.q.ptr(), Dt::F16)?;
        k.rope_norm(b.q.ptr(), m, HEADS, HD, D, std::ptr::null(), std::ptr::null(), 0, 0, self.q_gamma.fp())?;
        let vp: *const std::ffi::c_void = kv.ptr().cast::<u16>().wrapping_add(D).cast();
        nsd.attention_qk(b.q.ptr(), m, D, kv.ptr(), vp, LAT, 2 * D, Dt::F16, HEADS, HD, b.att.ptr(), Dt::F16)?;
        self.out.apply(nsd, k, b.att.ptr(), m, b.tmp.ptr(), Dt::F32)
    }
}

/// A call's buffers for m query rows
struct Buf {
    h: DevBuf,
    hn: DevBuf,
    q: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    mlp: DevBuf,
}

impl Buf {
    fn new(g: &Arc<Gpu>, m: usize, selfattn: bool) -> Result<Buf> {
        Ok(Buf {
            h: DevBuf::f32(g, m * D)?,
            hn: DevBuf::new(g, m * D * 2)?,
            q: DevBuf::new(g, m * D * 2)?,
            qkv: DevBuf::new(g, if selfattn { m * 3 * D * 2 } else { 2 })?,
            att: DevBuf::new(g, m * D * 2)?,
            tmp: DevBuf::f32(g, m * D)?,
            mlp: DevBuf::new(g, m * MLP * 2)?,
        })
    }
}

struct Mlp {
    a: Lin,
    b: Lin,
}

impl Mlp {
    fn load(f: &St, g: &Arc<Gpu>, p: &str) -> Result<Mlp> {
        Ok(Mlp { a: Lin::load(f, g, &format!("{p}.mlp.0"), true)?, b: Lin::load(f, g, &format!("{p}.mlp.2"), true)? })
    }
    /// hn (half) -> tmp (float32)
    fn apply(&self, cx: &Ctx, b: &Buf, m: usize) -> Result<()> {
        self.a.apply(&cx.nsd, &cx.k, b.hn.ptr(), m, b.mlp.ptr(), Dt::F16)?;
        cx.nsd.gelu(b.mlp.ptr(), Dt::F16, m * MLP, true)?;
        self.b.apply(&cx.nsd, &cx.k, b.mlp.ptr(), m, b.tmp.ptr(), Dt::F32)
    }
}

/// The input: in_proj(x) + the sinusoids, then input_layer; on the card [m, 1024] float32
struct Input {
    proj: HostLin,
    layer: Lin,
}

impl Input {
    fn load(f: &St, g: &Arc<Gpu>) -> Result<Input> {
        Ok(Input { proj: HostLin::load(f, "in_proj")?, layer: Lin::load(f, g, "input_layer", false)? })
    }
    fn apply(&self, cx: &Ctx, x: &[[f32; 3]], out: &DevBuf) -> Result<()> {
        let mut h = pos_v2(x);
        for (p, row) in x.iter().zip(h.chunks_exact_mut(D)) {
            for (o, v) in row.iter_mut().zip(self.proj.apply(p)) {
                *o += v;
            }
        }
        let hd = DevBuf::from_f32(&cx.gpu, &h)?;
        self.layer.apply(&cx.nsd, &cx.k, hd.ptr(), x.len(), out.ptr(), Dt::F32)
    }
}

pub struct OctreeDecoder {
    input: Input,
    l0: HostLin,
    l2: HostLin,
    ada: HostLin,
    blocks: Vec<(Cross, Mlp)>,
    out: Lin,
}

pub struct GaussianDecoder {
    input: Input,
    blocks: Vec<GsBlock>,
    out: Lin,
    perturb: Vec<f32>,
    base_offset_scale: f32,
}

struct GsBlock {
    qkv: Lin,
    q_gamma: DevBuf,
    k_gamma: DevBuf,
    out: Lin,
    norm2: (DevBuf, DevBuf),
    cross: Cross,
    mlp: Mlp,
}

/// The latent as the decoders read it: half [8192, 16]
pub fn latent(g: &Arc<Gpu>, z: &[f32]) -> Result<DevBuf> {
    if z.len() != LAT * LC {
        return Err(Error(format!("the latent: {} values, not 8192 x 16", z.len())));
    }
    w::dev_half(g, z)
}

impl OctreeDecoder {
    pub fn load(f: &St, g: &Arc<Gpu>) -> Result<OctreeDecoder> {
        if !f.has("l_embedder.mlp.0.weight") || !f.has("blocks.0.cross_attn.to_kv.weight") {
            return Err(Error(format!("{}: not Idol's octree decoder", f.path())));
        }
        let blocks = (0..4).map(|i| -> Result<(Cross, Mlp)> {
            let p = format!("blocks.{i}");
            Ok((Cross::load(f, g, &format!("{p}.cross_attn"))?, Mlp::load(f, g, &format!("{p}.mlp"))?))
        }).collect::<Result<Vec<_>>>()?;
        Ok(OctreeDecoder { input: Input::load(f, g)?, l0: HostLin::load(f, "l_embedder.mlp.0")?, l2: HostLin::load(f, "l_embedder.mlp.2")?,
                           ada: HostLin::load(f, "adaLN_modulation.1")?, blocks, out: Lin::load(f, g, "out_proj", false)? })
    }

    /// The 8 children logits [m, 8] of cells centred at x (in [0, 1]^3) at resolution `res`, the blocks' keys and
    /// values `kvs` made from the latent
    pub fn logits(&self, cx: &Ctx, x: &[[f32; 3]], res: u32, kvs: &[DevBuf]) -> Result<Vec<f32>> {
        let g = &cx.gpu;
        let m = x.len();
        let b = Buf::new(g, m, false)?;
        self.input.apply(cx, x, &b.h)?;
        let le = self.l2.apply(&w::silu(&self.l0.apply(&sinusoid(res as f32, 256, 1024.0, true))));
        let md = DevBuf::from_f32(g, &self.ada.apply(&w::silu(&le)))?;
        // SAFETY: parts of the 6 x 1024 row.
        let p = |j: usize| unsafe { md.fp().add(j * D) as *const f32 };
        let k = &cx.k;
        for ((cr, mlp), kv) in self.blocks.iter().zip(kvs) {
            k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-6, std::ptr::null(), std::ptr::null(), Rows::one(p(1), p(0)), b.hn.ptr(), Dt::F16)?;
            cr.apply(cx, &b, m, kv)?;
            k.gate_add(b.h.fp(), m, D, b.tmp.ptr(), Dt::F32, p(2), 0, 0)?;
            k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-6, std::ptr::null(), std::ptr::null(), Rows::one(p(4), p(3)), b.hn.ptr(), Dt::F16)?;
            mlp.apply(cx, &b, m)?;
            k.gate_add(b.h.fp(), m, D, b.tmp.ptr(), Dt::F32, p(5), 0, 0)?;
        }
        k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-5, std::ptr::null(), std::ptr::null(), Rows::none(), b.tmp.ptr(), Dt::F32)?;
        let o = DevBuf::f32(g, m * 8)?;
        self.out.apply(&cx.nsd, k, b.tmp.ptr(), m, o.ptr(), Dt::F32)?;
        g.sync()?;
        o.to_f32()
    }

    pub fn kvs(&self, cx: &Ctx, lat: &DevBuf) -> Result<Vec<DevBuf>> {
        self.blocks.iter().map(|(c, _)| c.kv_of(cx, lat)).collect()
    }

    /// `points` anchors in [0, 1]^3, placed by descending the octree (Idol's sample): each level's cells split their
    /// points among their children by systematic resampling of the children's probabilities, the leaves' points at
    /// random inside their cells. `rng` is the uniform draw [0, 1)
    pub fn anchors(&self, cx: &Ctx, lat: &DevBuf, points: usize, rng: &mut dyn FnMut() -> f32, mut on_level: impl FnMut(u32, &[[f32; 3]], &[f32]))
                   -> Result<Vec<[f32; 3]>> {
        let kvs = self.kvs(cx, lat)?;
        let mut cells: Vec<[u32; 3]> = vec![[0, 0, 0]];
        let mut counts: Vec<usize> = vec![points];
        for l in 1..=LEVELS {
            let rp = (1u32 << (l - 1)) as f32;
            let x: Vec<[f32; 3]> = cells.iter().map(|c| [(c[0] as f32 + 0.5) / rp, (c[1] as f32 + 0.5) / rp, (c[2] as f32 + 0.5) / rp]).collect();
            let lg = self.logits(cx, &x, 1 << l, &kvs)?;
            on_level(l, &x, &lg);
            let mut next_cells = Vec::new();
            let mut next_counts = Vec::new();
            for (i, c) in cells.iter().enumerate() {
                let row = &lg[i * 8..i * 8 + 8];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
                let s: f32 = e.iter().sum();
                let probs: Vec<f32> = e.iter().map(|v| v / s).collect();
                let split = systematic(&probs, counts[i], rng);
                for (ch, n) in split.iter().enumerate() {
                    if *n > 0 {
                        next_cells.push([2 * c[0] + (ch % 2) as u32, 2 * c[1] + ((ch / 2) % 2) as u32, 2 * c[2] + (ch / 4) as u32]);
                        next_counts.push(*n);
                    }
                }
            }
            cells = next_cells;
            counts = next_counts;
        }
        let side = (1u32 << LEVELS) as f32;
        let mut out = Vec::with_capacity(points);
        for (c, n) in cells.iter().zip(&counts) {
            for _ in 0..*n {
                out.push([(c[0] as f32 + rng()) / side, (c[1] as f32 + rng()) / side, (c[2] as f32 + rng()) / side]);
            }
        }
        Ok(out)
    }
}

/// `n` items split among the bins of `probs` by systematic resampling (Idol's systematic_sample): a random offset in
/// [0, 1/n) and n evenly spaced points looked up in the cumulative sum
pub fn systematic(probs: &[f32], n: usize, rng: &mut dyn FnMut() -> f32) -> Vec<usize> {
    let p = probs.len();
    let mut out = vec![0usize; p];
    if n == 0 {
        return out;
    }
    let sum: f32 = probs.iter().map(|v| v.max(0.0)).sum();
    let mut cdf = Vec::with_capacity(p);
    let mut acc = 0f32;
    for v in probs {
        acc += if sum > 0.0 { v.max(0.0) / sum } else { 1.0 / p as f32 };
        cdf.push(acc.min(1.0 - 1e-7));
    }
    let u0 = rng() / n as f32;
    for i in 0..n {
        let u = (u0 + i as f32 / n as f32).min(1.0 - 1e-7);
        let idx = cdf.iter().position(|c| *c >= u).unwrap_or(p - 1);
        out[idx] += 1;
    }
    out
}

impl GaussianDecoder {
    pub fn load(f: &St, g: &Arc<Gpu>) -> Result<GaussianDecoder> {
        if !f.has("points_offset_perturbation") || !f.has("blocks.0.self_attn.to_qkv.weight") {
            return Err(Error(format!("{}: not Idol's Gaussian decoder", f.path())));
        }
        let blocks = (0..16).map(|i| -> Result<GsBlock> {
            let p = format!("blocks.{i}");
            Ok(GsBlock {
                qkv: Lin::load(f, g, &format!("{p}.self_attn.to_qkv"), true)?,
                q_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.self_attn.q_rms_norm.gamma"), D)?)?,
                k_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.self_attn.k_rms_norm.gamma"), D)?)?,
                out: Lin::load(f, g, &format!("{p}.self_attn.to_out"), true)?,
                norm2: (DevBuf::from_f32(g, &f.vec_n(&format!("{p}.norm2.weight"), D)?)?, DevBuf::from_f32(g, &f.vec_n(&format!("{p}.norm2.bias"), D)?)?),
                cross: Cross::load(f, g, &format!("{p}.cross_attn"))?,
                mlp: Mlp::load(f, g, &format!("{p}.mlp"))?,
            })
        }).collect::<Result<Vec<_>>>()?;
        Ok(GaussianDecoder {
            input: Input::load(f, g)?,
            blocks,
            out: Lin::load(f, g, "out_proj", false)?,
            perturb: f.vec_n("points_offset_perturbation", PER * 3)?,
            base_offset_scale: f.vec_n("base_offset_scale", 1)?[0],
        })
    }

    /// The anchors' outputs [k, 480] from the latent
    pub fn features(&self, cx: &Ctx, lat: &DevBuf, anchors: &[[f32; 3]]) -> Result<Vec<f32>> {
        let (g, nsd, k) = (&cx.gpu, &cx.nsd, &cx.k);
        let m = anchors.len();
        let b = Buf::new(g, m, true)?;
        self.input.apply(cx, anchors, &b.h)?;
        let n = std::ptr::null();
        for blk in &self.blocks {
            k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-6, n, n, Rows::none(), b.hn.ptr(), Dt::F16)?;
            blk.qkv.apply(nsd, k, b.hn.ptr(), m, b.qkv.ptr(), Dt::F16)?;
            let hp = |off: usize| -> *mut std::ffi::c_void { b.qkv.ptr().cast::<u16>().wrapping_add(off).cast() };
            k.rope_norm(hp(0), m, HEADS, HD, 3 * D, n, n, 0, 0, blk.q_gamma.fp())?;
            k.rope_norm(hp(D), m, HEADS, HD, 3 * D, n, n, 0, 0, blk.k_gamma.fp())?;
            nsd.attention_qk(hp(0), m, 3 * D, hp(D), hp(2 * D), m, 3 * D, Dt::F16, HEADS, HD, b.att.ptr(), Dt::F16)?;
            blk.out.apply(nsd, k, b.att.ptr(), m, b.tmp.ptr(), Dt::F32)?;
            k.gate_add(b.h.fp(), m, D, b.tmp.ptr(), Dt::F32, n, 0, 0)?;
            k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-6, blk.norm2.0.fp(), blk.norm2.1.fp(), Rows::none(), b.hn.ptr(), Dt::F16)?;
            let kv = blk.cross.kv_of(cx, lat)?;
            blk.cross.apply(cx, &b, m, &kv)?;
            k.gate_add(b.h.fp(), m, D, b.tmp.ptr(), Dt::F32, n, 0, 0)?;
            k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-6, n, n, Rows::none(), b.hn.ptr(), Dt::F16)?;
            blk.mlp.apply(cx, &b, m)?;
            k.gate_add(b.h.fp(), m, D, b.tmp.ptr(), Dt::F32, n, 0, 0)?;
        }
        k.ln_mod(b.h.ptr(), Dt::F32, m, D, 1e-5, n, n, Rows::none(), b.tmp.ptr(), Dt::F32)?;
        let o = DevBuf::f32(g, m * 480)?;
        self.out.apply(nsd, k, b.tmp.ptr(), m, o.ptr(), Dt::F32)?;
        g.sync()?;
        o.to_f32()
    }

    /// The Gaussians of the anchors' outputs (to_representation and Idol's Gaussian: activated, z-up, in [-0.5, 0.5]^3)
    pub fn gaussians(&self, anchors: &[[f32; 3]], feats: &[f32]) -> Splats {
        let softplus = |x: f32| if x > 20.0 { x } else { x.exp().ln_1p() };
        let inv_softplus = |x: f32| x + (-(-x).exp_m1()).ln();
        let scaling_off = inv_softplus(0.004);
        let opacity_off = (0.1f32 / 0.9).ln();
        let k = anchors.len();
        let mut s = Splats::with_capacity(k * PER);
        for (a, f) in anchors.iter().zip(feats.chunks_exact(480)) {
            for j in 0..PER {
                let os = softplus(f[448 + j] + self.base_offset_scale);
                let mut pos = [0f32; 3];
                for c in 0..3 {
                    let o = (f[j * 3 + c] + self.perturb[j * 3 + c]).tanh() * 0.5 * 1.5 * os;
                    pos[c] = o + a[c] - 0.5;
                }
                let dc = [f[96 + j * 3], f[96 + j * 3 + 1], f[96 + j * 3 + 2]];
                let sc: [f32; 3] = std::array::from_fn(|c| {
                    let v = softplus(f[192 + j * 3 + c] + scaling_off);
                    (v * v + 0.0009 * 0.0009).sqrt()
                });
                let mut q = [f[288 + j * 4] * 0.1 + 1.0, f[288 + j * 4 + 1] * 0.1, f[288 + j * 4 + 2] * 0.1, f[288 + j * 4 + 3] * 0.1];
                let nq = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                q.iter_mut().for_each(|v| *v /= nq);
                let op = 1.0 / (1.0 + (-(f[416 + j] + opacity_off)).exp());
                s.push(pos, dc, op, sc, q);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systematic_resampling_keeps_the_count() {
        let mut x = 0.37f32;
        let mut rng = || {
            x = (x * 9301.0 + 0.49297).fract();
            x
        };
        let c = systematic(&[0.5, 0.25, 0.25, 0.0], 8, &mut rng);
        assert_eq!(c.iter().sum::<usize>(), 8);
        assert_eq!(c[3], 0);
        assert!(c[0] == 4 && c[1] == 2 && c[2] == 2, "{c:?}");
        assert_eq!(systematic(&[1.0, 0.0], 0, &mut rng), vec![0, 0]);
    }

    #[test]
    fn decoder_sinusoids_span_ten_octaves() {
        let e = pos_v2(&[[1.0, 0.0, 0.0]]);
        // x = 1, f = 1: sin(pi) ~ 0, cos(pi) = -1
        assert!(e[0].abs() < 1e-6 && (e[170] + 1.0).abs() < 1e-6);
        // the last frequency is 2^10: cos(1024 pi) = 1
        assert!((e[339] - 1.0).abs() < 1e-3);
    }
}
