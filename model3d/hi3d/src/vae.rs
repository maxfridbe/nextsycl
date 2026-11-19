//! The self-pruning sparse decoder (sparc3d's SparconvVAE / PTv3MeshDecoder): the 32-channel latents on the structure's
//! voxels to per-corner signed distances and offsets on a grid 8 times finer.
//!
//! ```text
//!   z_to_f (conv 32 -> 512) -> mid: ResBlock(512), 2 serialized-attention blocks (windows of 48 along z-order and
//!   transposed z-order; a conv position encoding), ResBlock(512)
//!   -> dec3: 2 ResBlocks(512), unpool -> dec2: ResBlocks(512 -> 256, 256), unpool -> dec1: ResBlocks(256 -> 128, 128),
//!      unpool -> dec0: 2 ResBlocks(128) -> GroupNorm, Swish, conv -> 8 sign logits + 32 attributes (tanh)
//!   unpool: proj (C -> 8 C: each child's features), the occupancy head (GroupNorm, Swish, conv C -> 8) - the children
//!           whose logit is > 0 are kept, at 2 x + their offset
//!   ResBlock: x (or its linear) + conv(swish(gn(conv(swish(gn(x))))))   (passed through above 12M voxels)
//! ```
//!
//! PointGroupNorm is one mean and variance a group of channels over every voxel (float32, E[x^2] - mean^2); its sums are
//! taken a block of rows at a time on the GPU and added on the host in double. The children's neighbour maps come
//! from their parents' (no hashing after the first level). float32 features, half into the products.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::ffi::Epi;
use crate::sconv::SConv;
use crate::w::{dev_i32, File, Lin};
use crate::{sparse, Ctx};

const GROUPS: usize = 32;
const PATCH: usize = 48;
/// SparseResnetBlock's skip_points: above this many voxels a residual block passes its input through (the reference's
/// guard, reproduced)
pub const SKIP_POINTS: usize = 12_000_000;
/// rows a block of the group norm's sums
const GN_ROWS: usize = 2048;

struct Gn {
    w: DevBuf,
    b: DevBuf,
    c: usize,
}

impl Gn {
    fn load(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<Gn> {
        let w = f.vec(&format!("{p}.weights"))?;
        Ok(Gn { c: w.len(), w: DevBuf::from_f32(gpu, &w)?, b: f.dev(gpu, &format!("{p}.bias"))? })
    }

    /// swish(groupnorm(x)) into out (out_dt)
    fn apply(&self, cx: &Ctx, x: &DevBuf, n: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let groups = if 4 * GROUPS > self.c { self.c / 4 } else { GROUPS };
        let blocks = n.div_ceil(GN_ROWS);
        let part = DevBuf::f32(&cx.gpu, blocks * groups * 2)?;
        cx.k.gn_sums(x.fp(), n, self.c, groups, GN_ROWS, part.fp())?;
        let p = part.to_f32()?;
        let cp = self.c / groups;
        let count = (n as f32) * cp as f32;
        let eps = 1e-6f32;
        let mut mean = vec![0f32; groups];
        let mut inv = vec![0f32; groups];
        for g in 0..groups {
            let (mut s, mut q) = (0f64, 0f64);
            for b in 0..blocks {
                s += p[(b * groups + g) * 2] as f64;
                q += p[(b * groups + g) * 2 + 1] as f64;
            }
            let m = s as f32 / (count + eps);
            let var = q as f32 / (count + eps) - m * m;
            mean[g] = m;
            inv[g] = 1.0 / (var + eps).sqrt();
        }
        let md = DevBuf::from_f32(&cx.gpu, &mean)?;
        let id = DevBuf::from_f32(&cx.gpu, &inv)?;
        cx.k.gn_swish(x.fp(), n, self.c, groups, md.fp(), id.fp(), self.w.fp(), self.b.fp(), out, out_dt)
    }
}

struct Res {
    gn1: Gn,
    conv1: SConv,
    gn2: Gn,
    conv2: SConv,
    res: Option<Lin>,
}

impl Res {
    fn load(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<Res> {
        Ok(Res {
            gn1: Gn::load(f, gpu, &format!("{p}.block.0"))?,
            conv1: SConv::load(f, gpu, &format!("{p}.block.2"))?,
            gn2: Gn::load(f, gpu, &format!("{p}.block.3"))?,
            conv2: SConv::load(f, gpu, &format!("{p}.block.5"))?,
            res: if f.has(&format!("{p}.res_conv.weight")) { Some(Lin::load(f, gpu, &format!("{p}.res_conv"), false)?) } else { None },
        })
    }

    fn out(&self) -> usize {
        self.conv2.co
    }
}

struct Attn {
    cpe_conv: SConv,
    cpe_lin: Lin,
    cpe_ln: (DevBuf, DevBuf),
    n1: (DevBuf, DevBuf),
    qkv: Lin,
    proj: Lin,
    n2: (DevBuf, DevBuf),
    fc1: Lin,
    fc2: Lin,
    heads: usize,
    trans: bool,
}

struct Up {
    gn: Gn,
    cls: SConv,
    proj: Lin,
}

pub struct Vae {
    z2f: SConv,
    mid_in: Res,
    attn: Vec<Attn>,
    mid_out: Res,
    /// dec3, dec2, dec1, dec0: their blocks and the unpooling after them
    stages: Vec<(Vec<Res>, Option<Up>)>,
    out_gn: Gn,
    out_conv: SConv,
    pub bytes: usize,
}

/// Voxels at one level and their features (float32 [n, c])
pub struct Level {
    pub coords: Vec<[i32; 3]>,
    pub nmap: DevBuf,
    pub feat: DevBuf,
    pub c: usize,
}

impl Level {
    pub fn n(&self) -> usize {
        self.coords.len()
    }
}

/// The decoder's output: the finest voxels, their 8 sign logits and 32 attributes (tanh: 8 distances, 8 x 3 offsets)
pub struct Decoded {
    pub coords: Vec<[i32; 3]>,
    pub res: usize,
    pub sign: Vec<f32>,
    pub feat: Vec<f32>,
}

/// An unpooling's parents and their 8 occupancy logits each
pub type Pruning = (Vec<[i32; 3]>, Vec<f32>);

/// What `decode` shows the check: z_to_f's and mid's features, each unpooling's parents and logits
#[derive(Default)]
pub struct Probe {
    pub z2f: Vec<f32>,
    pub mid: Vec<f32>,
    pub ups: Vec<Pruning>,
    /// decode at these parents and logits instead of the decoder's own (the reference's pruning)
    pub force: Option<Vec<Pruning>>,
}

fn ln_pair(f: &File, gpu: &Arc<Gpu>, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((f.dev(gpu, &format!("{p}.weight"))?, f.dev(gpu, &format!("{p}.bias"))?))
}

impl Vae {
    pub fn load(f: &File, gpu: &Arc<Gpu>) -> Result<Vae> {
        let d = "decoder";
        let mut attn = Vec::new();
        while f.has(&format!("{d}.mid.block{}.attn.qkv.weight", attn.len())) {
            let p = format!("{d}.mid.block{}", attn.len());
            let qkv = Lin::load(f, gpu, &format!("{p}.attn.qkv"), false)?;
            attn.push(Attn {
                cpe_conv: SConv::load(f, gpu, &format!("{p}.cpe.0"))?,
                cpe_lin: Lin::load(f, gpu, &format!("{p}.cpe.1"), false)?,
                cpe_ln: ln_pair(f, gpu, &format!("{p}.cpe.2"))?,
                n1: ln_pair(f, gpu, &format!("{p}.norm1.0"))?,
                heads: qkv.k / 16,
                qkv,
                proj: Lin::load(f, gpu, &format!("{p}.attn.proj"), false)?,
                n2: ln_pair(f, gpu, &format!("{p}.norm2.0"))?,
                fc1: Lin::load(f, gpu, &format!("{p}.mlp.0.fc1"), false)?,
                fc2: Lin::load(f, gpu, &format!("{p}.mlp.0.fc2"), false)?,
                trans: attn.len() % 2 == 1,
            });
        }
        if attn.len() != 2 || attn[0].qkv.k != 512 {
            return Err(Error(format!("{}: not PTv3MeshDecoder's mid stack (2 attention blocks of 512)", f.path())));
        }
        let mut stages = Vec::new();
        for s in (0..4).rev() {
            let p = format!("{d}.dec.dec{s}");
            let blocks = vec![Res::load(f, gpu, &format!("{p}.block0"))?, Res::load(f, gpu, &format!("{p}.block1"))?];
            let up = if s > 0 {
                Some(Up {
                    gn: Gn::load(f, gpu, &format!("{p}.up.upsample.cls_head.fc.0"))?,
                    cls: SConv::load(f, gpu, &format!("{p}.up.upsample.cls_head.fc.2"))?,
                    proj: Lin::load(f, gpu, &format!("{p}.up.upsample.proj"), false)?,
                })
            } else {
                None
            };
            stages.push((blocks, up));
        }
        let vae = Vae {
            z2f: SConv::load(f, gpu, &format!("{d}.z_to_f.conv.0"))?,
            mid_in: Res::load(f, gpu, &format!("{d}.mid.block_in"))?,
            attn,
            mid_out: Res::load(f, gpu, &format!("{d}.mid.block_out"))?,
            stages,
            out_gn: Gn::load(f, gpu, &format!("{d}.conv_out.conv.0"))?,
            out_conv: SConv::load(f, gpu, &format!("{d}.conv_out.conv.2"))?,
            bytes: f.st.tensors.iter().map(|t| t.bytes as usize).sum::<usize>() / 2,
        };
        if vae.out_conv.co != 40 || vae.z2f.ci != 32 {
            return Err(Error(format!("{}: not the released decoder (32 latent channels in, 8 + 32 out)", f.path())));
        }
        Ok(vae)
    }

    fn res(&self, cx: &Ctx, r: &Res, lv: &mut Level) -> Result<()> {
        let n = lv.n();
        if n > SKIP_POINTS {
            // the reference's guard: the block is a no-op
            return Ok(());
        }
        let (k, gpu) = (&cx.k, &cx.gpu);
        let co = r.out();
        let out = DevBuf::f32(gpu, n * co)?;
        match &r.res {
            Some(l) => {
                let xh = DevBuf::new(gpu, n * lv.c * 2)?;
                k.convert(lv.feat.ptr(), Dt::F32, n * lv.c, xh.ptr(), Dt::F16)?;
                l.apply(cx, xh.ptr(), n, out.ptr(), Dt::F32)?;
            }
            None => out.copy_within(0, &lv.feat, 0, n * co * 4)?,
        }
        let h = DevBuf::new(gpu, n * lv.c.max(co) * 2)?;
        r.gn1.apply(cx, &lv.feat, n, h.ptr(), Dt::F16)?;
        let t = DevBuf::f32(gpu, n * r.conv1.co)?;
        r.conv1.apply(cx, &h, n, &lv.nmap, t.ptr(), Dt::F32)?;
        r.gn2.apply(cx, &t, n, h.ptr(), Dt::F16)?;
        drop(t);
        let t2 = DevBuf::f32(gpu, n * co)?;
        r.conv2.apply(cx, &h, n, &lv.nmap, t2.ptr(), Dt::F32)?;
        k.add(out.ptr(), Dt::F32, t2.ptr(), Dt::F32, n * co)?;
        lv.feat = out;
        lv.c = co;
        Ok(())
    }

    fn attention(&self, cx: &Ctx, a: &Attn, lv: &mut Level) -> Result<()> {
        let (k, gpu, nsd) = (&cx.k, &cx.gpu, &cx.nsd);
        let (n, c) = (lv.n(), lv.c);
        // the conv position encoding: conv -> linear -> layer norm, added
        let h = DevBuf::new(gpu, n * c * 2)?;
        k.convert(lv.feat.ptr(), Dt::F32, n * c, h.ptr(), Dt::F16)?;
        let t = DevBuf::f32(gpu, n * c)?;
        a.cpe_conv.apply(cx, &h, n, &lv.nmap, t.ptr(), Dt::F32)?;
        k.convert(t.ptr(), Dt::F32, n * c, h.ptr(), Dt::F16)?;
        a.cpe_lin.apply(cx, h.ptr(), n, t.ptr(), Dt::F32)?;
        k.ln_mod(t.ptr(), Dt::F32, n, c, 1e-5, a.cpe_ln.0.fp(), a.cpe_ln.1.fp(), std::ptr::null(), std::ptr::null(), t.ptr(), Dt::F32)?;
        k.add(lv.feat.ptr(), Dt::F32, t.ptr(), Dt::F32, n * c)?;
        // windows of 48 along the serialized order
        k.ln_mod(lv.feat.ptr(), Dt::F32, n, c, 1e-5, a.n1.0.fp(), a.n1.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), Dt::F16)?;
        let qkv = DevBuf::new(gpu, n * 3 * c * 2)?;
        a.qkv.apply(cx, h.ptr(), n, qkv.ptr(), Dt::F16)?;
        let (order, inverse) = sparse::serialize(&lv.coords, a.trans);
        let (windows, s, pad) = sparse::padding(n, PATCH);
        let rows: Vec<i32> = pad.iter().map(|p| order[*p]).collect();
        let rows = dev_i32(gpu, &rows)?;
        let qp = DevBuf::new(gpu, windows * s * 3 * c * 2)?;
        k.gather_rows(qkv.ptr(), Dt::F16, 3 * c, rows.ptr().cast(), windows * s, qp.ptr())?;
        drop(qkv);
        let ap = DevBuf::new(gpu, windows * s * c * 2)?;
        k.win_attn(qp.ptr(), windows, s, a.heads, c / a.heads, 1.0 / ((c / a.heads) as f32).sqrt(), ap.ptr())?;
        drop(qp);
        // each voxel's row is its place in the order (the padded copies are dropped)
        let inv = dev_i32(gpu, &inverse)?;
        k.gather_rows(ap.ptr(), Dt::F16, c, inv.ptr().cast(), n, h.ptr())?;
        drop(ap);
        a.proj.apply_epi(nsd, k, h.ptr(), n, lv.feat.ptr(), Dt::F32, Epi::Resid(std::ptr::null()), Some(&t))?;
        // the MLP
        k.ln_mod(lv.feat.ptr(), Dt::F32, n, c, 1e-5, a.n2.0.fp(), a.n2.1.fp(), std::ptr::null(), std::ptr::null(), h.ptr(), Dt::F16)?;
        let mid = DevBuf::new(gpu, n * a.fc1.n * 2)?;
        a.fc1.apply_epi(nsd, k, h.ptr(), n, mid.ptr(), Dt::F16, Epi::GeluErf, None)?;
        a.fc2.apply_epi(nsd, k, mid.ptr(), n, lv.feat.ptr(), Dt::F32, Epi::Resid(std::ptr::null()), Some(&t))?;
        Ok(())
    }

    /// The occupancy logits [n, 8] of a level (the unpooling's head)
    fn logits(&self, cx: &Ctx, up: &Up, lv: &Level) -> Result<Vec<f32>> {
        let n = lv.n();
        let h = DevBuf::new(&cx.gpu, n * lv.c * 2)?;
        up.gn.apply(cx, &lv.feat, n, h.ptr(), Dt::F16)?;
        let o = DevBuf::f32(&cx.gpu, n * 8)?;
        up.cls.apply(cx, &h, n, &lv.nmap, o.ptr(), Dt::F32)?;
        o.to_f32()
    }

    /// The unpooling: the children whose logit is > 0, their features from proj, their neighbour maps from the parents'
    fn unpool(&self, cx: &Ctx, up: &Up, lv: Level, logits: &[f32]) -> Result<Level> {
        let (k, gpu) = (&cx.k, &cx.gpu);
        let (n, c) = (lv.n(), lv.c);
        let mut kids = vec![-1i32; n * 8];
        let mut coords = Vec::new();
        let mut parent = Vec::new();
        let mut sub = Vec::new();
        let mut rows = Vec::new();
        for (p, pc) in lv.coords.iter().enumerate() {
            for s in 0..8 {
                if logits[p * 8 + s] > 0.0 {
                    kids[p * 8 + s] = coords.len() as i32;
                    coords.push([2 * pc[0] + (s >> 2) as i32, 2 * pc[1] + ((s >> 1) & 1) as i32, 2 * pc[2] + (s & 1) as i32]);
                    parent.push(p as i32);
                    sub.push(s as i32);
                    rows.push((p * 8 + s) as i32);
                }
            }
        }
        let m = coords.len();
        if m == 0 {
            return Err(Error("the decoder kept no voxel (every occupancy logit <= 0)".into()));
        }
        let xh = DevBuf::new(gpu, n * c * 2)?;
        k.convert(lv.feat.ptr(), Dt::F32, n * c, xh.ptr(), Dt::F16)?;
        let pf = DevBuf::new(gpu, n * 8 * c * 2)?;
        up.proj.apply(cx, xh.ptr(), n, pf.ptr(), Dt::F16)?;
        drop(xh);
        let rows = dev_i32(gpu, &rows)?;
        let ch = DevBuf::new(gpu, m * c * 2)?;
        k.gather_rows(pf.ptr(), Dt::F16, c, rows.ptr().cast(), m, ch.ptr())?;
        drop(pf);
        let feat = DevBuf::f32(gpu, m * c)?;
        k.convert(ch.ptr(), Dt::F16, m * c, feat.ptr(), Dt::F32)?;
        let (kd, pd, sd) = (dev_i32(gpu, &kids)?, dev_i32(gpu, &parent)?, dev_i32(gpu, &sub)?);
        let nmap = DevBuf::new(gpu, m * 27 * 4)?;
        k.child_nmap(lv.nmap.ptr().cast(), kd.ptr().cast(), pd.ptr().cast(), sd.ptr().cast(), m, nmap.ptr().cast())?;
        cx.gpu.sync()?;
        Ok(Level { coords, nmap, feat, c })
    }

    /// The latents (float32 [n, 32], denormalized) on `coords` at structure resolution `res`
    pub fn decode(&self, cx: &Ctx, coords: &[[i32; 3]], lat: &[f32], res: usize, mut probe: Option<&mut Probe>) -> Result<Decoded> {
        let (k, gpu) = (&cx.k, &cx.gpu);
        let n = coords.len();
        if lat.len() != n * 32 {
            return Err(Error(format!("{} latent values for {n} voxels", lat.len())));
        }
        let nmap = dev_i32(gpu, &sparse::nmap(coords))?;
        let zh = DevBuf::new(gpu, n * 32 * 2)?;
        zh.write(0, &crate::w::half_bytes(lat))?;
        let feat = DevBuf::f32(gpu, n * 512)?;
        self.z2f.apply(cx, &zh, n, &nmap, feat.ptr(), Dt::F32)?;
        let mut lv = Level { coords: coords.to_vec(), nmap, feat, c: 512 };
        if let Some(p) = probe.as_deref_mut() {
            p.z2f = lv.feat.to_f32()?;
        }
        self.res(cx, &self.mid_in, &mut lv)?;
        for a in &self.attn {
            self.attention(cx, a, &mut lv)?;
        }
        self.res(cx, &self.mid_out, &mut lv)?;
        if let Some(p) = probe.as_deref_mut() {
            p.mid = lv.feat.to_f32()?;
        }
        let mut res = res;
        for (ui, (blocks, up)) in self.stages.iter().enumerate() {
            for b in blocks {
                self.res(cx, b, &mut lv)?;
            }
            if let Some(up) = up {
                let mut logits = self.logits(cx, up, &lv)?;
                if let Some(p) = probe.as_deref_mut() {
                    p.ups.push((lv.coords.clone(), logits.clone()));
                    if let Some(f) = p.force.as_ref().and_then(|f| f.get(ui)) {
                        // the reference's parents must be these voxels (same set); its logits put in our order
                        let at: std::collections::HashMap<[i32; 3], usize> = f.0.iter().enumerate().map(|(i, c)| (*c, i)).collect();
                        if at.len() != lv.n() || !lv.coords.iter().all(|c| at.contains_key(c)) {
                            return Err(Error(format!("unpooling {ui}: the reference's parents are not these voxels")));
                        }
                        logits = lv.coords.iter().flat_map(|c| f.1[at[c] * 8..at[c] * 8 + 8].to_vec()).collect();
                    }
                }
                lv = self.unpool(cx, up, lv, &logits)?;
                res *= 2;
            }
        }
        let n = lv.n();
        let h = DevBuf::new(gpu, n * lv.c * 2)?;
        self.out_gn.apply(cx, &lv.feat, n, h.ptr(), Dt::F16)?;
        let o = DevBuf::f32(gpu, n * 40)?;
        self.out_conv.apply(cx, &h, n, &lv.nmap, o.ptr(), Dt::F32)?;
        drop(h);
        let o = o.to_f32()?;
        let _ = k;
        let mut sign = Vec::with_capacity(n * 8);
        let mut feat = Vec::with_capacity(n * 32);
        for r in o.chunks_exact(40) {
            sign.extend_from_slice(&r[..8]);
            feat.extend(r[8..].iter().map(|v| v.tanh()));
        }
        Ok(Decoded { coords: lv.coords, res, sign, feat })
    }
}
