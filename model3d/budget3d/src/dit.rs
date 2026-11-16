//! The denoiser: Idol's LatentSeqMMFlowModel (a rectified-flow transformer over the object latent's 8192 Sobol tokens,
//! the picture's 4101 tokens and the camera's one) with Budget3D's adapters - the latent seen through a token layout
//! (merged cells and cubes: ScaleMixer3D embeddings, integrated positional encodings, scale embeddings) and refined
//! back to every point after the trunk (RefineStage, the Wiener high-frequency skip). Idol-turbo's phase LoRAs are
//! merged into their matrices at load, one copy for each phase (slow t >= 0.5, fast below).
//!
//! ```text
//!   z [8192, 16] -> input_layer + the Sobol positions' sinusoids          (dense: these are the tokens)
//!                -> octree order -> mixer(2 points) = cells -> mixer(8 cells) = cubes; tokens [cubes | cells | points]
//!   noise refiner: 2 blocks over the tokens        context refiner: 2 blocks over the picture's tokens (once an image)
//!   joint: 24 blocks over [tokens | picture | camera] (guided and unguided side by side)
//!   head: layer norm, the time's shift / scale -> out_layer: a velocity a token -> refiners: a velocity a point
//! block: x + gate * attn(ln(x) * (1 + scale) + shift), the attention's q and k rotated by RePo's learned positions
//!        (each token's features -> a 3-D offset a head) then RMS-normed; x + gate * mlp(...) (GELU-tanh, 4096)
//! ```
//! float32 residual stream, half activations and matrices in the blocks; RePo, the refiners and the heads in float32
//! (as Budget3D keeps them).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt};

use crate::ffi::Rows;
use crate::layout::{self, Layout, Octree, N_CELL, N_CUBE, N_SOBOL};
use crate::w::{self, gelu_tanh, sinusoid, HostLin, Lin, St};
use crate::Ctx;

pub const D: usize = 1024;
pub const HEADS: usize = 16;
pub const HD: usize = 64;
const MLP: usize = 4096;
pub const IN: usize = 16;
pub const CAM: usize = 5;
pub const COND1: usize = 1280;
pub const COND2: usize = 128;
/// the picture's tokens: CLS, 4 registers and 64 x 64 patches
pub const L_COND: usize = 4101;
const REPO: usize = 128;
/// RePo's rotary pairs a head per axis (x, y, z): 10, 10, 12
const AXES: (usize, usize) = (10, 10);
const RW: usize = 128;
const RHEADS: usize = 4;

/// A block's four matrices (one set a phase for Idol-turbo)
struct Lins {
    qkv: Lin,
    out: Lin,
    mlp0: Lin,
    mlp2: Lin,
}

/// Idol's learned rotary positions: layer norm, a gated MLP to a 3-D offset a head
struct Repo {
    norm: (DevBuf, DevBuf),
    /// gate_map and content_map side by side: [256, 1024]
    gc: Lin,
    fin: Lin,
    freqs: DevBuf,
}

struct Block {
    repo: Repo,
    /// the time-modulated blocks' own shift table (the context refiner has affine norms instead)
    shift_table: Option<Vec<f32>>,
    norms: Option<[(DevBuf, DevBuf); 2]>,
    q_gamma: DevBuf,
    k_gamma: DevBuf,
    lins: Vec<Lins>,
}

/// ScaleMixer3D: n child token features -> one parent's
struct Mixer {
    n: usize,
    pos: DevBuf,
    down: Lin,
    sw: DevBuf,
    sb: DevBuf,
    up: Lin,
}

struct RBlock {
    ada: Lin,
    qkv: Lin,
    proj: Lin,
    mlp0: Lin,
    mlp2: Lin,
}

/// RefineStage: a parent's velocity -> each of its n children's
struct Stage {
    n: usize,
    embed: Lin,
    embed_zt: Lin,
    /// slots + pos_proj(fourier(child offset)) [n, 128]
    slot: DevBuf,
    blocks: Vec<RBlock>,
    out_norm: DevBuf,
    out: Lin,
}

pub struct Dit {
    pub turbo: bool,
    pub octree: Octree,
    t0: HostLin,
    t2: HostLin,
    ada: HostLin,
    input: Lin,
    /// the Sobol positions' sinusoids [8192, 1024], Idol's order
    pos: DevBuf,
    cond1: Lin,
    cond2: Lin,
    noise_ref: Vec<Block>,
    ctx_ref: Vec<Block>,
    blocks: Vec<Block>,
    cam0: HostLin,
    cam2: HostLin,
    head_table: Vec<f32>,
    out: Lin,
    cam_out: HostLin,
    mix2: Mixer,
    mix4: Mixer,
    /// per cube / cell: the integrated positional encoding plus the scale embedding; the points' scale embedding
    add4: DevBuf,
    add2: DevBuf,
    add1: DevBuf,
    ref2: Stage,
    ref4: Stage,
    pub bytes: usize,
    /// NS_M3D_B3D_PROFILE=2: seconds in a block's parts over the calls [RePo, norm + qkv + rotation, attention,
    /// out + gated add, MLP] (each part synced: slower)
    pub prof: std::sync::Mutex<Option<[f64; 5]>>,
}

/// The phase whose matrices a call uses (Idol-turbo: slow at t >= 0.5; the base model has one)
pub fn phase(turbo: bool, t: f32) -> usize {
    if turbo && t < 0.5 { 1 } else { 0 }
}

/// Idol's PcdAbsolutePositionEmbedder: [sin(2 pi x f), cos(2 pi x f)] a coordinate, f = 2^(0..15) then 154 more up to
/// 2^16, zero-padded to 1024
fn pos_embed(pos: &[f32]) -> Vec<f32> {
    let fd = D / 3 / 2;
    let res_dim = fd - 16;
    let freqs: Vec<f32> = (0..16).map(|i| i as f32).chain((0..res_dim).map(|i| i as f32 / res_dim as f32 * 16.0)).map(|e| 2f32.powf(e)).collect();
    let mut out = vec![0f32; pos.len() / 3 * D];
    for (p, row) in pos.chunks_exact(3).zip(out.chunks_exact_mut(D)) {
        for a in 0..3 {
            for (i, f) in freqs.iter().enumerate() {
                let v = p[a] * f * 2.0 * std::f32::consts::PI;
                row[a * 2 * fd + i] = v.sin();
                row[a * 2 * fd + fd + i] = v.cos();
            }
        }
    }
    out
}

/// Budget3D's fourier3d: [sin, cos](x 2^k pi), k < 4, a coordinate
fn fourier3d(p: &[f32; 3]) -> Vec<f32> {
    let mut o = Vec::with_capacity(24);
    for v in p {
        let ang: Vec<f32> = (0..4).map(|k| v * 2f32.powi(k) * std::f32::consts::PI).collect();
        o.extend(ang.iter().map(|a| a.sin()));
        o.extend(ang.iter().map(|a| a.cos()));
    }
    o
}

/// MipNeRFIPE3D's features of a cube (centre in [0, 1]^3, scale 2 or 4): 8 octaves damped by its variance, and by
/// the excess over an s = 2 cell's
fn ipe_features(c: &[f32; 3], scale: f32) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    let base_var = (2.0 * pi / 32.0 * 2.0).powi(2) / 12.0;
    let var_s = (2.0 * pi * scale / 32.0).powi(2) / 12.0;
    let mut o = Vec::with_capacity(96);
    for v in c {
        let mu = (2.0 * v - 1.0) * pi;
        let mut parts = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for k in 0..8 {
            let w = 2f32.powi(k);
            let ang = mu * w;
            let damp = (-0.5 * w * w * var_s).exp();
            let excess = (-0.5 * w * w * (var_s - base_var).max(0.0)).exp_m1();
            parts[0].push(ang.sin() * damp);
            parts[1].push(ang.cos() * damp);
            parts[2].push(ang.sin() * excess);
            parts[3].push(ang.cos() * excess);
        }
        for p in parts {
            o.extend(p);
        }
    }
    o
}

/// The buffers of a call over `seqs` sequences of `l` rows (n latent tokens each)
pub struct Work {
    pub n: usize,
    pub l: usize,
    pub seqs: usize,
    h: DevBuf,
    hn: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    mlp: DevBuf,
    rp_ln: DevBuf,
    rp_gc: DevBuf,
    rp_sw: DevBuf,
    delta: DevBuf,
    mods: DevBuf,
    /// the tokens [n, 1024] float32 (the noise refiner works on them in place)
    x: DevBuf,
    big: DevBuf,
    z: DevBuf,
    htok: DevBuf,
    vtok: DevBuf,
}

impl Work {
    fn rows(&self) -> usize {
        self.seqs * self.l
    }
    /// The tokens `Dit::tokens` made [n, 1024]
    pub fn tokens(&self) -> Result<Vec<f32>> {
        self.x.gpu().sync()?;
        let mut v = self.x.to_f32()?;
        v.truncate(self.n * D);
        Ok(v)
    }
}

/// What a call leaves for the check: the noise refiner's tokens, the trunk's token rows, the head's, out_layer's
#[derive(Default)]
pub struct Probe {
    pub nref: Vec<f32>,
    pub joint: Vec<f32>,
    pub head: Vec<f32>,
    pub vtok: Vec<f32>,
    pub vpts: Vec<f32>,
}

impl Dit {
    /// `f`: Budget3D's (or Budget3D-Idol-Turbo's) denoiser; `idol`: Idol's own, for the Sobol positions (pos_pe)
    pub fn load(f: &St, idol: &St, cx: &Ctx, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let g = &cx.gpu;
        if !f.has("bp_refiners.2.out.weight") {
            return Err(Error(format!("{}: not a Budget3D denoiser (no bp_refiners)", f.path())));
        }
        let turbo = f.0.tensor("blocks.0.attn.qkv.slow_A").is_some();
        let pos_pe = idol.vec_n("pos_pe", N_SOBOL * 3)?;
        let octree = Octree::new(&pos_pe)?;
        let mut bytes = 0usize;
        let mut lin = |p: &str, half: bool| -> Result<Lin> {
            let l = Lin::load(f, g, p, half)?;
            bytes += l.bytes();
            Ok(l)
        };
        let input = lin("input_layer", false)?;
        let cond1 = lin("cond_embedder", false)?;
        let cond2 = lin("cond_embedder2", false)?;
        let out = lin("out_layer", false)?;
        let blocks_of = |prefix: &str, repo: &str, count: usize, modulated: bool, log: &mut dyn FnMut(String)| -> Result<Vec<Block>> {
            (0..count).map(|i| {
                let p = format!("{prefix}.{i}");
                let r = format!("{repo}.{i}");
                let mut gc = f.vec_n(&format!("{r}.gate_map.weight"), REPO * D)?;
                gc.extend(f.vec_n(&format!("{r}.content_map.weight"), REPO * D)?);
                let mut freqs = f.vec_n(&format!("{r}.freqs_0"), AXES.0)?;
                freqs.extend(f.vec_n(&format!("{r}.freqs_1"), AXES.1)?);
                freqs.extend(f.vec_n(&format!("{r}.freqs_2"), HD / 2 - AXES.0 - AXES.1)?);
                let repo = Repo {
                    norm: (DevBuf::from_f32(g, &f.vec_n(&format!("{r}.norm.weight"), D)?)?, DevBuf::from_f32(g, &f.vec_n(&format!("{r}.norm.bias"), D)?)?),
                    gc: Lin::from_host(g, &gc, None, 2 * REPO, D, false)?,
                    fin: Lin::from_host(g, &f.vec_n(&format!("{r}.final_map.weight"), 3 * HEADS * REPO)?, None, 3 * HEADS, REPO, false)?,
                    freqs: DevBuf::from_f32(g, &freqs)?,
                };
                let lins = Self::lins(f, cx, &p, turbo)?;
                if i == 0 && turbo {
                    log(format!("  {prefix}: Idol-turbo's phase LoRAs merged (slow, fast)"));
                }
                let norm = |n: &str| -> Result<(DevBuf, DevBuf)> {
                    Ok((DevBuf::from_f32(g, &f.vec_n(&format!("{p}.{n}.weight"), D)?)?, DevBuf::from_f32(g, &f.vec_n(&format!("{p}.{n}.bias"), D)?)?))
                };
                Ok(Block {
                    repo,
                    shift_table: if modulated { Some(f.vec_n(&format!("{p}.shift_table"), 6 * D)?) } else { None },
                    norms: if modulated { None } else { Some([norm("norm1")?, norm("norm2")?]) },
                    q_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.attn.q_norm.gamma"), D)?)?,
                    k_gamma: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.attn.k_norm.gamma"), D)?)?,
                    lins,
                })
            }).collect()
        };
        let noise_ref = blocks_of("noise_refiner", "noise_repo_layers", 2, true, log)?;
        let ctx_ref = blocks_of("context_refiner", "context_repo_layers", 2, false, log)?;
        let blocks = blocks_of("blocks", "repo_layers", 24, true, log)?;
        let per_set = (3 + 1 + 4 + 4) * D * D * 2;
        bytes += (noise_ref.len() + ctx_ref.len() + blocks.len()) * per_set * if turbo { 2 } else { 1 };

        let mixer = |k: &str, n: usize| -> Result<Mixer> {
            Ok(Mixer {
                n,
                pos: DevBuf::from_f32(g, &f.vec_n(&format!("bp_mixers.{k}.pos"), n * D)?)?,
                down: Lin::load(f, g, &format!("bp_mixers.{k}.down"), false)?,
                sw: DevBuf::from_f32(g, &f.vec_n(&format!("bp_mixers.{k}.spatial_w"), 256 * n * n)?)?,
                sb: DevBuf::from_f32(g, &f.vec_n(&format!("bp_mixers.{k}.spatial_b"), n * 256)?)?,
                up: Lin::load(f, g, &format!("bp_mixers.{k}.up"), false)?,
            })
        };
        let stage = |k: &str, n: usize, rel: &[[f32; 3]]| -> Result<Stage> {
            let p = format!("bp_refiners.{k}");
            let pp = HostLin::load(f, &format!("{p}.pos_proj"))?;
            let slots = f.vec_n(&format!("{p}.slots"), n * RW)?;
            let mut slot = Vec::with_capacity(n * RW);
            for s in 0..n {
                let e = pp.apply(&fourier3d(&rel[s]));
                slot.extend(e.iter().zip(&slots[s * RW..(s + 1) * RW]).map(|(a, b)| a + b));
            }
            let blocks = (0..2).map(|i| -> Result<RBlock> {
                let b = format!("{p}.blocks.{i}");
                Ok(RBlock { ada: Lin::load(f, g, &format!("{b}.ada"), false)?, qkv: Lin::load(f, g, &format!("{b}.qkv"), false)?,
                            proj: Lin::load(f, g, &format!("{b}.proj"), false)?, mlp0: Lin::load(f, g, &format!("{b}.mlp.0"), false)?,
                            mlp2: Lin::load(f, g, &format!("{b}.mlp.2"), false)? })
            }).collect::<Result<Vec<_>>>()?;
            Ok(Stage { n, embed: Lin::load(f, g, &format!("{p}.embed"), false)?, embed_zt: Lin::load(f, g, &format!("{p}.embed_zt"), false)?,
                       slot: DevBuf::from_f32(g, &slot)?, blocks, out_norm: DevBuf::from_f32(g, &f.vec_n(&format!("{p}.out_norm.weight"), RW)?)?,
                       out: Lin::load(f, g, &format!("{p}.out"), false)? })
        };
        // the cubes' and cells' additive terms: IPE(centre, scale) + the scale's embedding
        let se = f.vec_n("bp_scale_emb", 4 * D)?;
        let ipe0 = HostLin::load(f, "bp_ipe.mlp.0")?;
        let ipe2 = HostLin::load(f, "bp_ipe.mlp.2")?;
        let ipe = |cs: &[[f32; 3]], scale: f32, row: usize| -> Vec<f32> {
            cs.iter().flat_map(|c| {
                let hid = w::silu(&ipe0.apply(&ipe_features(c, scale)));
                ipe2.apply(&hid).into_iter().enumerate().map(|(i, v)| v + se[row * D + i]).collect::<Vec<f32>>()
            }).collect()
        };
        let add4 = DevBuf::from_f32(g, &ipe(&layout::cube_centres(), 4.0, 2))?;
        let add2 = DevBuf::from_f32(g, &ipe(&layout::cell_centres(), 2.0, 1))?;
        let add1 = DevBuf::from_f32(g, &se[..D])?;
        let d = Dit {
            turbo,
            t0: HostLin::load(f, "t_embedder.mlp.0")?,
            t2: HostLin::load(f, "t_embedder.mlp.2")?,
            ada: HostLin::load(f, "adaLN_modulation.1")?,
            input,
            pos: DevBuf::from_f32(g, &pos_embed(&pos_pe))?,
            cond1,
            cond2,
            noise_ref,
            ctx_ref,
            blocks,
            cam0: HostLin::load(f, "cam_refiner.mlp.0")?,
            cam2: HostLin::load(f, "cam_refiner.mlp.2")?,
            head_table: f.vec_n("shift_table", 2 * D)?,
            out,
            cam_out: HostLin::load(f, "cam_out_layer")?,
            mix2: mixer("2", 2)?,
            mix4: mixer("4", 8)?,
            add4,
            add2,
            add1,
            ref2: stage("2", 2, &layout::cell_child_offsets())?,
            ref4: stage("4", 8, &layout::cube_child_offsets())?,
            octree,
            bytes,
            prof: std::sync::Mutex::new(None),
        };
        g.sync()?;
        Ok(d)
    }

    /// A block's matrices: as stored (half), or with Idol-turbo's two experts merged in (W + 4 B A, slow then fast)
    fn lins(f: &St, cx: &Ctx, p: &str, turbo: bool) -> Result<Vec<Lins>> {
        let g = &cx.gpu;
        let names = ["attn.qkv", "attn.out", "mlp.mlp.0", "mlp.mlp.2"];
        let mut sets = Vec::new();
        for ph in if turbo { &["slow", "fast"][..] } else { &[""][..] } {
            let mut ls = Vec::new();
            for n in names {
                let q = format!("{p}.{n}");
                if ph.is_empty() {
                    ls.push(Lin::load(f, g, &q, true)?);
                    continue;
                }
                let base = Lin::load(f, g, &q, false)?;
                let a = f.vec(&format!("{q}.{ph}_A"))?;
                let b = f.vec(&format!("{q}.{ph}_B"))?;
                let r = a.len() / base.k;
                if r * base.k != a.len() || b.len() != base.n * r {
                    return Err(Error(format!("{q}: LoRA {ph} of {} / {} values for a {} x {} matrix", a.len(), b.len(), base.n, base.k)));
                }
                // W [n, k] += (4 B) [n, r] . A [r, k]: the product x = 4 B, weights A^T [k, r]
                let bs: Vec<f32> = b.iter().map(|v| v * 4.0).collect();
                let mut at = vec![0f32; base.k * r];
                for i in 0..r {
                    for j in 0..base.k {
                        at[j * r + i] = a[i * base.k + j];
                    }
                }
                let (bd, ad) = (DevBuf::from_f32(g, &bs)?, DevBuf::from_f32(g, &at)?);
                cx.nsd.linear_acc(bd.ptr(), Dt::F32, base.n, r, ad.ptr(), base.k, base.w.ptr())?;
                let wh = DevBuf::new(g, base.n * base.k * 2)?;
                cx.k.convert(base.w.ptr(), Dt::F32, base.n * base.k, wh.ptr(), Dt::F16)?;
                cx.nsd.wait()?;
                ls.push(Lin { w: wh, half: true, bias: base.bias, n: base.n, k: base.k });
            }
            let mut it = ls.into_iter();
            sets.push(Lins { qkv: it.next().unwrap(), out: it.next().unwrap(), mlp0: it.next().unwrap(), mlp2: it.next().unwrap() });
        }
        Ok(sets)
    }

    /// The buffers for `seqs` sequences of n tokens (and the picture's and the camera's)
    pub fn work(&self, gpu: &Arc<Gpu>, n: usize, seqs: usize) -> Result<Work> {
        let l = n + L_COND + 1;
        let rows = seqs * l.max(L_COND);
        Ok(Work {
            n,
            l,
            seqs,
            h: DevBuf::f32(gpu, rows * D)?,
            hn: DevBuf::new(gpu, rows * D * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * D * 2)?,
            att: DevBuf::new(gpu, rows * D * 2)?,
            tmp: DevBuf::f32(gpu, rows * D)?,
            mlp: DevBuf::new(gpu, rows * MLP * 2)?,
            rp_ln: DevBuf::f32(gpu, rows * D)?,
            rp_gc: DevBuf::f32(gpu, rows * 2 * REPO)?,
            rp_sw: DevBuf::f32(gpu, rows * REPO)?,
            delta: DevBuf::f32(gpu, rows * 3 * HEADS)?,
            mods: DevBuf::f32(gpu, 26 * 6 * D)?,
            x: DevBuf::f32(gpu, N_SOBOL * D)?,
            big: DevBuf::f32(gpu, N_SOBOL * (2 * D + D + 512))?,
            z: DevBuf::f32(gpu, N_SOBOL * IN)?,
            htok: DevBuf::f32(gpu, seqs * n * D)?,
            vtok: DevBuf::f32(gpu, seqs * n * IN)?,
        })
    }

    /// The time embedding [1024] of t in [0, 1] (Idol's t argument is 1000 t)
    fn t_emb(&self, t: f32) -> Vec<f32> {
        self.t2.apply(&w::silu(&self.t0.apply(&sinusoid(1000.0 * t, 256, 10000.0, false))))
    }

    /// One block over `rows` rows of h (sequences of `seq` rows), modulated by `md` (the device row of 6 x 1024:
    /// shift, scale, gate of the attention, then of the MLP) or by its affine norms
    #[allow(clippy::too_many_arguments)]
    fn block(&self, cx: &Ctx, w: &Work, h: &DevBuf, rows: usize, seq: usize, b: &Block, ph: usize, md: Option<*const f32>) -> Result<()> {
        let (nsd, k) = (&cx.nsd, &cx.k);
        let ls = &b.lins[ph.min(b.lins.len() - 1)];
        // SAFETY: the 6 x 1024 modulation row's parts.
        let m = |j: usize| md.map_or(std::ptr::null(), |p| unsafe { p.add(j * D) });
        let hp = |buf: &DevBuf, off: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(off).cast() };
        let mut prof = *self.prof.lock().unwrap();
        let mut t0 = std::time::Instant::now();
        let mut tick = |i: usize| -> Result<()> {
            if let Some(p) = prof.as_mut() {
                cx.gpu.sync()?;
                p[i] += t0.elapsed().as_secs_f64();
                t0 = std::time::Instant::now();
            }
            Ok(())
        };
        // RePo: the block's input -> a 3-D offset a head
        nsd.layer_norm(h.ptr(), Dt::F32, rows, D, b.repo.norm.0.ptr(), b.repo.norm.1.ptr(), 1e-5, w.rp_ln.ptr(), Dt::F32)?;
        b.repo.gc.apply(nsd, k, w.rp_ln.ptr(), rows, w.rp_gc.ptr(), Dt::F32)?;
        nsd.swiglu(w.rp_gc.ptr(), Dt::F32, rows, REPO, w.rp_sw.ptr(), Dt::F32)?;
        b.repo.fin.apply(nsd, k, w.rp_sw.ptr(), rows, w.delta.ptr(), Dt::F32)?;
        tick(0)?;
        // the attention
        let (nw, nb, md1) = match &b.norms {
            Some(n) => (n[0].0.fp() as *const f32, n[0].1.fp() as *const f32, Rows::none()),
            None => (std::ptr::null(), std::ptr::null(), Rows::one(m(1), m(0))),
        };
        k.ln_mod(h.ptr(), Dt::F32, rows, D, 1e-6, nw, nb, md1, w.hn.ptr(), Dt::F16)?;
        ls.qkv.apply(nsd, k, w.hn.ptr(), rows, w.qkv.ptr(), Dt::F16)?;
        k.rope_norm(w.qkv.ptr(), rows, HEADS, HD, 3 * D, w.delta.fp(), b.repo.freqs.fp(), AXES.0, AXES.1, b.q_gamma.fp())?;
        k.rope_norm(hp(&w.qkv, D), rows, HEADS, HD, 3 * D, w.delta.fp(), b.repo.freqs.fp(), AXES.0, AXES.1, b.k_gamma.fp())?;
        tick(1)?;
        for s in 0..rows / seq {
            let r0 = s * seq;
            nsd.attention_qk(hp(&w.qkv, r0 * 3 * D), seq, 3 * D, hp(&w.qkv, r0 * 3 * D + D), hp(&w.qkv, r0 * 3 * D + 2 * D), seq, 3 * D, Dt::F16, HEADS, HD,
                             hp(&w.att, r0 * D), Dt::F16)?;
        }
        tick(2)?;
        ls.out.apply(nsd, k, w.att.ptr(), rows, w.tmp.ptr(), Dt::F32)?;
        k.gate_add(h.fp(), rows, D, w.tmp.ptr(), Dt::F32, m(2), 0, 0)?;
        tick(3)?;
        // the MLP
        let (nw, nb, md2) = match &b.norms {
            Some(n) => (n[1].0.fp() as *const f32, n[1].1.fp() as *const f32, Rows::none()),
            None => (std::ptr::null(), std::ptr::null(), Rows::one(m(4), m(3))),
        };
        k.ln_mod(h.ptr(), Dt::F32, rows, D, 1e-6, nw, nb, md2, w.hn.ptr(), Dt::F16)?;
        ls.mlp0.apply(nsd, k, w.hn.ptr(), rows, w.mlp.ptr(), Dt::F16)?;
        nsd.gelu(w.mlp.ptr(), Dt::F16, rows * MLP, true)?;
        ls.mlp2.apply(nsd, k, w.mlp.ptr(), rows, w.tmp.ptr(), Dt::F32)?;
        k.gate_add(h.fp(), rows, D, w.tmp.ptr(), Dt::F32, m(5), 0, 0)?;
        tick(4)?;
        if prof.is_some() {
            *self.prof.lock().unwrap() = prof;
        }
        Ok(())
    }

    /// The picture's tokens refined (feature1 [4101, 1280] DINOv3, feature2 [4101, 128] FLUX.2 VAE): [4101, 1024]
    /// float32, made once an image (an Idol-turbo phase each)
    pub fn context(&self, cx: &Ctx, w: &Work, f1: &[f32], f2: &[f32], ph: usize) -> Result<DevBuf> {
        if f1.len() != L_COND * COND1 || f2.len() != L_COND * COND2 {
            return Err(Error(format!("the condition: {} and {} values, not 4101 x 1280 and 4101 x 128", f1.len(), f2.len())));
        }
        let g = &cx.gpu;
        let (a, b) = (DevBuf::from_f32(g, f1)?, DevBuf::from_f32(g, f2)?);
        let h = DevBuf::f32(g, L_COND * D)?;
        self.cond1.apply(&cx.nsd, &cx.k, a.ptr(), L_COND, h.ptr(), Dt::F32)?;
        self.cond2.apply(&cx.nsd, &cx.k, b.ptr(), L_COND, w.tmp.ptr(), Dt::F32)?;
        cx.k.add(h.ptr(), Dt::F32, w.tmp.ptr(), Dt::F32, L_COND * D)?;
        for blk in &self.ctx_ref {
            self.block(cx, w, &h, L_COND, L_COND, blk, ph, None)?;
        }
        g.sync()?;
        Ok(h)
    }

    /// The layout's tokens of z (Idol's order, [8192, 16]) into w.x: [cubes | cells | points] (dense: z's own order)
    pub fn tokens(&self, cx: &Ctx, w: &Work, z: &[f32], lay: &Layout) -> Result<()> {
        let (nsd, k, g) = (&cx.nsd, &cx.k, &cx.gpu);
        w.z.write(0, &w::f32_bytes(z))?;
        let target = if lay.is_dense() { &w.x } else { &w.big };
        self.input.apply(nsd, k, w.z.ptr(), N_SOBOL, target.ptr(), Dt::F32)?;
        k.add(target.ptr(), Dt::F32, self.pos.ptr(), Dt::F32, N_SOBOL * D)?;
        if lay.is_dense() {
            return Ok(());
        }
        // big = [h (Idol order) | h in octree order | the mixers' scratch]; x gets the tokens
        let h = w.big.fp();
        // SAFETY: parts of `big` (2 x 8192 x 1024 + 8192 x 1536 floats).
        let (hoct, scr) = unsafe { (h.add(N_SOBOL * D), h.add(2 * N_SOBOL * D)) };
        let perm: Vec<i32> = self.octree.perm.iter().map(|v| *v as i32).collect();
        let pd = w::dev_i32(g, &perm)?;
        k.gather_add(h, D, pd.ptr().cast(), N_SOBOL, std::ptr::null(), std::ptr::null(), hoct)?;
        // the cells into h's first 4096 rows (free now), their cubes' children gathered into the next 4096, the cubes
        let h2 = h;
        self.mix(cx, &self.mix2, hoct, N_CELL, scr, h2)?;
        let cc: Vec<i32> = self.octree.cube_cells.iter().flatten().map(|v| *v as i32).collect();
        let ccd = w::dev_i32(g, &cc)?;
        // SAFETY: rows 4096 .. 8192 of the first part.
        let sub4 = unsafe { h2.add(N_CELL * D) };
        k.gather_add(h2, D, ccd.ptr().cast(), N_CUBE * 8, std::ptr::null(), std::ptr::null(), sub4)?;
        let cubes = DevBuf::f32(g, N_CUBE * D)?;
        self.mix(cx, &self.mix4, sub4, N_CUBE, scr, cubes.fp())?;
        let (n4, n2) = (lay.n4(), lay.n2());
        let i32s = |v: &[usize]| v.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let x = w.x.fp();
        if n4 > 0 {
            let i4 = w::dev_i32(g, &i32s(&lay.idx4))?;
            k.gather_add(cubes.fp(), D, i4.ptr().cast(), n4, self.add4.fp(), i4.ptr().cast(), x)?;
        }
        if n2 > 0 {
            let i2 = w::dev_i32(g, &i32s(&lay.idx2))?;
            // SAFETY: rows n4 .. n4 + n2 of x.
            k.gather_add(h2, D, i2.ptr().cast(), n2, self.add2.fp(), i2.ptr().cast(), unsafe { x.add(n4 * D) })?;
        }
        if !lay.idx1.is_empty() {
            let pts: Vec<i32> = lay.idx1.iter().flat_map(|c| [2 * *c as i32, 2 * *c as i32 + 1]).collect();
            let i1 = w::dev_i32(g, &pts)?;
            // SAFETY: rows n4 + n2 .. of x.
            k.gather_add(hoct, D, i1.ptr().cast(), pts.len(), self.add1.fp(), std::ptr::null(), unsafe { x.add((n4 + n2) * D) })?;
        }
        g.sync()
    }

    /// ScaleMixer3D over r parents of m.n children (`sub` [r, n, 1024]) into `out` [r, 1024]; `scr` room for
    /// r n 1024 + r n 256 x 2 floats
    fn mix(&self, cx: &Ctx, m: &Mixer, sub: *const f32, r: usize, scr: *mut f32, out: *mut f32) -> Result<()> {
        let (nsd, k) = (&cx.nsd, &cx.k);
        let rn = r * m.n;
        // SAFETY: parts of the scratch, sized by the caller.
        let (a, dn, mx) = unsafe { (scr, scr.add(rn * D), scr.add(rn * D + rn * 256)) };
        k.slot_add(sub, r, m.n, D, m.pos.fp(), a)?;
        m.down.apply(nsd, k, a.cast(), rn, dn.cast(), Dt::F32)?;
        nsd.gelu(dn.cast(), Dt::F32, rn * 256, false)?;
        k.mix(dn, r, m.n, 256, m.sw.fp(), m.sb.fp(), mx)?;
        m.up.apply(nsd, k, mx.cast(), r, out.cast(), Dt::F32)?;
        // SAFETY: the mean goes where the slot sums were (no longer read).
        k.group_mean(sub, r, m.n, D, a)?;
        k.add(out.cast(), Dt::F32, a.cast(), Dt::F32, r * D)
    }

    /// The velocity (Idol's order [8192, 16]) and the camera's [5] of z, cam at t (in [0, 1]) for the layout, guided
    /// by `cfg` against the second context when two are given (w.seqs 2), else the first's alone
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(&self, cx: &Ctx, w: &Work, z: &[f32], cam: &[f32], t: f32, lay: &Layout, ctx: &[&DevBuf], cfg: f32,
                    mut probe: Option<&mut Probe>) -> Result<(Vec<f32>, Vec<f32>)> {
        let (nsd, k, g) = (&cx.nsd, &cx.k, &cx.gpu);
        let n = lay.tokens();
        if n != w.n || ctx.len() != w.seqs {
            return Err(Error(format!("a call of {n} tokens and {} contexts on buffers for {} and {}", ctx.len(), w.n, w.seqs)));
        }
        let ph = phase(self.turbo, t);
        let te = self.t_emb(t);
        let tmod = self.ada.apply(&w::silu(&te));
        let mut table = Vec::with_capacity(26 * 6 * D);
        for b in self.noise_ref.iter().chain(&self.blocks) {
            let st = b.shift_table.as_ref().expect("time-modulated blocks");
            table.extend(tmod.iter().zip(st).map(|(a, b)| a + b));
        }
        w.mods.write(0, &w::f32_bytes(&table))?;
        // SAFETY: rows of the [26, 6144] table.
        let row = |i: usize| unsafe { w.mods.fp().add(i * 6 * D) as *const f32 };
        self.tokens(cx, w, z, lay)?;
        for (i, b) in self.noise_ref.iter().enumerate() {
            self.block(cx, w, &w.x, n, n, b, ph, Some(row(i)))?;
        }
        if let Some(p) = probe.as_deref_mut() {
            g.sync()?;
            p.nref = w.x.to_f32()?[..n * D].to_vec();
        }
        // the camera's token, on the host
        let camh: Vec<f32> = self.cam2.apply(&self.cam0.apply(cam).into_iter().map(gelu_tanh).collect::<Vec<f32>>());
        let l = w.l;
        for (s, c) in ctx.iter().enumerate() {
            w.h.copy_within(s * l * D * 4, &w.x, 0, n * D * 4)?;
            w.h.copy_within((s * l + n) * D * 4, c, 0, L_COND * D * 4)?;
            w.h.write(((s + 1) * l - 1) * D * 4, &w::f32_bytes(&camh))?;
        }
        let rows = w.rows();
        for (i, b) in self.blocks.iter().enumerate() {
            self.block(cx, w, &w.h, rows, l, b, ph, Some(row(2 + i)))?;
        }
        // the head: layer norm, then the time's scale and shift
        let shift: Vec<f32> = self.head_table[..D].iter().zip(&te).map(|(a, b)| a + b).collect();
        let scale: Vec<f32> = self.head_table[D..].iter().zip(&te).map(|(a, b)| a + b).collect();
        let ss = DevBuf::from_f32(g, &[scale.clone(), shift.clone()].concat())?;
        // SAFETY: the shift after the 1024 scales.
        let md = Rows::one(ss.fp(), unsafe { ss.fp().add(D) });
        let mut vcams = Vec::new();
        for s in 0..w.seqs {
            // SAFETY: sequence s's token rows of h, its rows of htok.
            let (src, dst) = unsafe { (w.h.fp().add(s * l * D), w.htok.fp().add(s * n * D)) };
            k.ln_mod(src.cast(), Dt::F32, n, D, 1e-5, std::ptr::null(), std::ptr::null(), md, dst.cast(), Dt::F32)?;
            let mut c = vec![0u8; D * 4];
            w.h.read(((s + 1) * l - 1) * D * 4, &mut c)?;
            let c: Vec<f32> = c.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let mean = c.iter().sum::<f32>() / D as f32;
            let var = c.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
            let hc: Vec<f32> = c.iter().enumerate().map(|(i, v)| (v - mean) / (var + 1e-5).sqrt() * (1.0 + scale[i]) + shift[i]).collect();
            vcams.push(self.cam_out.apply(&hc));
        }
        self.out.apply(nsd, k, w.htok.ptr(), w.seqs * n, w.vtok.ptr(), Dt::F32)?;
        g.sync()?;
        if let Some(p) = probe.as_deref_mut() {
            let hh = w.h.to_f32()?;
            p.joint = (0..w.seqs).flat_map(|s| hh[s * l * D..(s * l + n) * D].to_vec()).collect();
            p.head = w.htok.to_f32()?[..w.seqs * n * D].to_vec();
            p.vtok = w.vtok.to_f32()?[..w.seqs * n * IN].to_vec();
        }
        let vtok = w.vtok.to_f32()?;
        let mut vs = Vec::new();
        for s in 0..w.seqs {
            let vt = &vtok[s * n * IN..(s + 1) * n * IN];
            if lay.is_dense() {
                vs.push(vt.to_vec());
            } else {
                // SAFETY: sequence s's rows.
                let (vtp, htp) = unsafe { (w.vtok.fp().add(s * n * IN) as *const f32, w.htok.fp().add(s * n * D) as *const f32) };
                let vp = self.expand(cx, vt, vtp, htp, z, t, lay)?;
                if let Some(p) = probe.as_deref_mut() {
                    p.vpts.extend_from_slice(&vp);
                }
                vs.push(self.octree.from_octree(&vp, IN));
            }
        }
        if vs.len() == 1 {
            return Ok((vs.remove(0), vcams.remove(0)));
        }
        let guide = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(p, q)| (1.0 + cfg) * p - cfg * q).collect() };
        Ok((guide(&vs[0], &vs[1]), guide(&vcams[0], &vcams[1])))
    }

    /// The merged tokens' velocities back to one a point (Budget3D's _expand): octree order [4096 cells, 2, 16].
    /// `vt` the token velocities (host), `vtp` / `htp` them and the head's features on the card
    #[allow(clippy::too_many_arguments)]
    fn expand(&self, cx: &Ctx, vt: &[f32], vtp: *const f32, htp: *const f32, z: &[f32], t: f32, lay: &Layout) -> Result<Vec<f32>> {
        let g = &cx.gpu;
        let (n4, n2) = (lay.n4(), lay.n2());
        let zo = self.octree.to_octree(z, IN);
        let mut v = vec![0f32; N_SOBOL * IN];
        // the Wiener high-frequency gain on the residual of z_t inside the merged region
        let gain = t / (t * t + 0.0625 * (1.0 - t) * (1.0 - t)).max(1e-6);
        for (i, c) in lay.idx1.iter().enumerate() {
            let src = (n4 + n2 + 2 * i) * IN;
            v[2 * c * IN..2 * c * IN + 2 * IN].copy_from_slice(&vt[src..src + 2 * IN]);
        }
        let cell_pts = |c: usize| &zo[2 * c * IN..2 * c * IN + 2 * IN];
        if n2 > 0 {
            let zs: Vec<f32> = lay.idx2.iter().flat_map(|c| cell_pts(*c).to_vec()).collect();
            // SAFETY: the cells' rows after the cubes'.
            let (out, _) = self.refine(cx, &self.ref2, n2, unsafe { vtp.add(n4 * IN) }, unsafe { htp.add(n4 * D) }, &zs, None)?;
            let o = out.to_f32()?;
            for (i, c) in lay.idx2.iter().enumerate() {
                let zc = cell_pts(*c);
                for ch in 0..IN {
                    let m = 0.5 * (zc[ch] + zc[IN + ch]);
                    for s in 0..2 {
                        v[(2 * c + s) * IN + ch] = o[(2 * i + s) * IN + ch] + gain * (zc[s * IN + ch] - m);
                    }
                }
            }
        }
        if n4 > 0 {
            // the cubes' 8 cells: their two points' mean for the first stage, the points for the second
            let cells: Vec<usize> = lay.idx4.iter().flat_map(|j| self.octree.cube_cells[*j]).collect();
            let zmean: Vec<f32> = cells.iter().flat_map(|c| {
                let p = cell_pts(*c);
                (0..IN).map(|ch| 0.5 * (p[ch] + p[IN + ch])).collect::<Vec<f32>>()
            }).collect();
            let (cell_v, cell_h) = self.refine(cx, &self.ref4, n4, vtp, htp, &zmean, None)?;
            let m8 = n4 * 8;
            let rep: Vec<i32> = (0..m8).map(|r| (r / 8) as i32).collect();
            let repd = w::dev_i32(g, &rep)?;
            let cond8 = DevBuf::f32(g, m8 * D)?;
            cx.k.gather_add(htp, D, repd.ptr().cast(), m8, std::ptr::null(), std::ptr::null(), cond8.fp())?;
            let zs: Vec<f32> = cells.iter().flat_map(|c| cell_pts(*c).to_vec()).collect();
            let (out, _) = self.refine(cx, &self.ref2, m8, cell_v.fp(), cond8.fp(), &zs, Some(cell_h.fp()))?;
            let o = out.to_f32()?;
            for (ci, cube) in lay.idx4.iter().enumerate() {
                let cc = &self.octree.cube_cells[*cube];
                let mut mean = [0f32; IN];
                for c in cc {
                    let p = cell_pts(*c);
                    for ch in 0..IN {
                        mean[ch] += (p[ch] + p[IN + ch]) / 16.0;
                    }
                }
                for (k8, c) in cc.iter().enumerate() {
                    let zc = cell_pts(*c);
                    let r = ci * 8 + k8;
                    for s in 0..2 {
                        for ch in 0..IN {
                            v[(2 * c + s) * IN + ch] = o[(2 * r + s) * IN + ch] + gain * (zc[s * IN + ch] - mean[ch]);
                        }
                    }
                }
            }
        }
        Ok(v)
    }

    /// RefineStage over m parents: lat [m, 16] and cond [m, 1024] on the card, z_stage [m, n, 16] (host), carry
    /// [m, 128] -> (the children's latents [m n, 16], the hidden state [m n, 128])
    #[allow(clippy::too_many_arguments)]
    fn refine(&self, cx: &Ctx, st: &Stage, m: usize, lat: *const f32, cond: *const f32, z_stage: &[f32], carry: Option<*const f32>)
              -> Result<(DevBuf, DevBuf)> {
        let (nsd, k, g) = (&cx.nsd, &cx.k, &cx.gpu);
        let s = st.n;
        let ms = m * s;
        let e = DevBuf::f32(g, m * RW)?;
        st.embed.apply(nsd, k, lat.cast(), m, e.ptr(), Dt::F32)?;
        let zd = DevBuf::from_f32(g, z_stage)?;
        let zt = DevBuf::f32(g, ms * RW)?;
        st.embed_zt.apply(nsd, k, zd.ptr(), ms, zt.ptr(), Dt::F32)?;
        let h = DevBuf::f32(g, ms * RW)?;
        k.refine_init(e.fp(), m, s, RW, st.slot.fp(), zt.fp(), carry.unwrap_or(std::ptr::null()), h.fp())?;
        // silu(cond), kept for both blocks' modulations
        let sc = DevBuf::f32(g, m * D)?;
        k.gather_add(cond, D, Self::iota(g, m)?.ptr().cast(), m, std::ptr::null(), std::ptr::null(), sc.fp())?;
        k.silu(sc.ptr(), Dt::F32, m * D)?;
        let ada = DevBuf::f32(g, m * 6 * RW)?;
        let hn = DevBuf::f32(g, ms * RW)?;
        let qkv = DevBuf::f32(g, ms * 3 * RW)?;
        let att = DevBuf::f32(g, ms * RW)?;
        let mlp = DevBuf::f32(g, ms * 4 * RW)?;
        let a = ada.fp();
        // SAFETY: the [m, 768] modulation's parts.
        let part = |j: usize| unsafe { a.add(j * RW) as *const f32 };
        for b in &st.blocks {
            b.ada.apply(nsd, k, sc.ptr(), m, ada.ptr(), Dt::F32)?;
            let rows = |sc_: usize, sh: usize| Rows { scale: part(sc_), shift: part(sh), stride: 6 * RW, group: s };
            k.ln_mod(h.ptr(), Dt::F32, ms, RW, 1e-6, std::ptr::null(), std::ptr::null(), rows(1, 0), hn.ptr(), Dt::F32)?;
            b.qkv.apply(nsd, k, hn.ptr(), ms, qkv.ptr(), Dt::F32)?;
            k.small_attn(qkv.fp(), m, s, RHEADS, RW / RHEADS, att.fp())?;
            b.proj.apply(nsd, k, att.ptr(), ms, hn.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), ms, RW, hn.ptr(), Dt::F32, part(2), 6 * RW, s)?;
            k.ln_mod(h.ptr(), Dt::F32, ms, RW, 1e-6, std::ptr::null(), std::ptr::null(), rows(4, 3), hn.ptr(), Dt::F32)?;
            b.mlp0.apply(nsd, k, hn.ptr(), ms, mlp.ptr(), Dt::F32)?;
            nsd.gelu(mlp.ptr(), Dt::F32, ms * 4 * RW, true)?;
            b.mlp2.apply(nsd, k, mlp.ptr(), ms, att.ptr(), Dt::F32)?;
            k.gate_add(h.fp(), ms, RW, att.ptr(), Dt::F32, part(5), 6 * RW, s)?;
        }
        nsd.rms_norm_mod(h.ptr(), Dt::F32, ms, RW, st.out_norm.ptr(), 1e-6, none(), none(), none(), hn.ptr(), Dt::F32)?;
        let o = DevBuf::f32(g, ms * IN)?;
        st.out.apply(nsd, k, hn.ptr(), ms, o.ptr(), Dt::F32)?;
        // + the parent's latent, repeated over its children
        let rep: Vec<i32> = (0..ms).map(|r| (r / s) as i32).collect();
        let repd = w::dev_i32(g, &rep)?;
        let out = DevBuf::f32(g, ms * IN)?;
        k.gather_add(lat, IN, repd.ptr().cast(), ms, o.fp(), Self::iota(g, ms)?.ptr().cast(), out.fp())?;
        g.sync()?;
        Ok((out, h))
    }

    /// 0 .. n as an int32 buffer
    fn iota(g: &Arc<Gpu>, n: usize) -> Result<DevBuf> {
        w::dev_i32(g, &(0..n as i32).collect::<Vec<i32>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_sinusoids_are_sin_then_cos_an_axis() {
        let e = pos_embed(&[0.0, 0.25, 0.5]);
        assert_eq!(e.len(), D);
        // x = 0: sines 0, cosines 1
        assert_eq!(e[0], 0.0);
        assert_eq!(e[170], 1.0);
        // y = 0.25, f = 1: sin(pi / 2)
        assert!((e[340] - 1.0).abs() < 1e-6);
        // the padding
        assert_eq!(&e[1020..], &[0.0; 4]);
    }

    #[test]
    fn an_s2_cell_has_no_excess_term() {
        let f = ipe_features(&[0.5, 0.5, 0.5], 2.0);
        assert_eq!(f.len(), 96);
        assert!(f[16..32].iter().all(|v| v.abs() < 1e-7));
        let f4 = ipe_features(&[0.5, 0.5, 0.5], 4.0);
        assert!(f4[16..32].iter().any(|v| v.abs() > 1e-3));
    }

    #[test]
    fn turbo_phases_switch_at_half() {
        assert_eq!(phase(true, 0.5), 0);
        assert_eq!(phase(true, 0.49), 1);
        assert_eq!(phase(false, 0.1), 0);
    }
}
