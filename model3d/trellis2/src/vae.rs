//! The sparse UNet decoders (TRELLIS.2's SparseUnetVaeDecoder; FlexiDualGridVaeDecoder is the shape one): the 32-wide
//! latent to 1024 channels, then five stages of [1024, 512, 256, 128, 64] channels with [4, 16, 8, 4, 0] ConvNeXt
//! blocks (a full 3x3x3 submanifold convolution, a layer norm, a 4x SiLU MLP, the residual), the first four ending in a
//! channel-to-space block that subdivides each voxel into the children its `to_subdiv` logits keep (the shape
//! decoder) or the shape decoder's own choices (the texture decoder: `Tree`), and a last layer norm and linear.
//! The residual stream in float32, the convolutions' and MLPs' inputs in half.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::Dt;

use crate::sconv::{upload_nmap, SConv};
use crate::sparse::{child_neighbours, neighbours, subdivide, Tree};
use crate::w::{File, Keep, Lin};
use crate::Ctx;

struct ConvNeXt {
    conv: SConv,
    norm: (DevBuf, DevBuf),
    mlp0: Lin,
    mlp2: Lin,
}

struct Up {
    norm1: (DevBuf, DevBuf),
    conv1: SConv,
    conv2: SConv,
    to_subdiv: Option<Lin>,
    cin: usize,
    cout: usize,
}

pub struct Decoder {
    from_latent: Lin,
    stages: Vec<(Vec<ConvNeXt>, Option<Up>)>,
    out: Lin,
    scratch: DevBuf,
    pub out_ch: usize,
}

/// What a decode leaves: the last level's voxels and their features, every level's coordinates entering its up block,
/// the subdivision logits (the shape decoder), the tree, the last level's neighbour map (the mesh is cut from it)
#[derive(Default)]
pub struct Decoded {
    pub coords: Vec<[i32; 3]>,
    pub feats: Vec<f32>,
    pub level_coords: Vec<Vec<[i32; 3]>>,
    pub subs: Vec<Vec<f32>>,
    pub tree: Tree,
    pub nmap: Vec<i32>,
    /// the features after each up block (for the check), when asked
    pub ups: Vec<Vec<f32>>,
}

/// What `run` does
pub struct Plan<'a> {
    /// the texture decoder: the shape decoder's subdivisions
    pub guide: Option<&'a Tree>,
    /// stop after this many up blocks and return their coordinates (the cascade's `upsample`)
    pub upsample: Option<usize>,
    /// keep each up block's output features (the check)
    pub keep_ups: bool,
}

impl Decoder {
    pub fn load(f: &File, gpu: &Arc<Gpu>) -> Result<Decoder> {
        let ln = |p: &str| -> Result<(DevBuf, DevBuf)> {
            Ok((DevBuf::from_f32(gpu, &f.vec(&format!("{p}.weight"))?)?, DevBuf::from_f32(gpu, &f.vec(&format!("{p}.bias"))?)?))
        };
        let mut stages = Vec::new();
        for i in 0.. {
            if !f.has(&format!("blocks.{i}.0.conv.weight")) && !f.has(&format!("blocks.{i}.0.conv1.weight")) {
                break;
            }
            let mut blocks = Vec::new();
            let mut up = None;
            for j in 0.. {
                let p = format!("blocks.{i}.{j}");
                if f.has(&format!("{p}.conv.weight")) {
                    blocks.push(ConvNeXt { conv: SConv::load(f, gpu, &format!("{p}.conv"))?, norm: ln(&format!("{p}.norm"))?,
                                           mlp0: Lin::load(f, gpu, &format!("{p}.mlp.0"), Keep::Half)?,
                                           mlp2: Lin::load(f, gpu, &format!("{p}.mlp.2"), Keep::Half)? });
                } else if f.has(&format!("{p}.conv1.weight")) {
                    let conv1 = SConv::load(f, gpu, &format!("{p}.conv1"))?;
                    let conv2 = SConv::load(f, gpu, &format!("{p}.conv2"))?;
                    up = Some(Up { norm1: ln(&format!("{p}.norm1"))?, cin: conv1.ci, cout: conv2.co, conv1, conv2,
                                   to_subdiv: if f.has(&format!("{p}.to_subdiv.weight")) {
                                       Some(Lin::load(f, gpu, &format!("{p}.to_subdiv"), Keep::Half)?)
                                   } else {
                                       None
                                   } });
                    break;
                } else {
                    break;
                }
            }
            stages.push((blocks, up));
        }
        if stages.is_empty() {
            return Err(Error(format!("{}: not a sparse UNet decoder (blocks.N.M.conv)", f.path())));
        }
        let from_latent = Lin::load(f, gpu, "from_latent", Keep::Half)?;
        let out = Lin::load(f, gpu, "output_layer", Keep::Half)?;
        let widest = stages.iter().flat_map(|(b, _)| b.iter().map(|c| c.mlp0.n * c.mlp0.k)).max().unwrap_or(0);
        Ok(Decoder { out_ch: out.n, from_latent, stages, out, scratch: DevBuf::new(gpu, widest.max(16) * 2)? })
    }

    pub fn predicts_subdivision(&self) -> bool {
        self.stages.iter().any(|(_, u)| u.as_ref().is_some_and(|u| u.to_subdiv.is_some()))
    }

    /// The decoder over latents [n, 32] at `coords`
    pub fn run(&self, cx: &Ctx, coords: &[[i32; 3]], latent: &[f32], plan: &Plan, mut progress: impl FnMut(u32, u32) -> Result<()>) -> Result<Decoded> {
        let g = &cx.gpu;
        let (nsd, k) = (&cx.nsd, &cx.k);
        let mut n = coords.len();
        if latent.len() != n * self.from_latent.k {
            return Err(Error(format!("the decoder's latent: {} values for {n} voxels of {}", latent.len(), self.from_latent.k)));
        }
        let mut out = Decoded::default();
        let mut coords = coords.to_vec();
        let mut nmap_h = neighbours(&coords);
        let mut nmap = upload_nmap(g, &nmap_h)?;
        let lat = crate::w::dev_half(g, latent)?;
        let mut c = self.from_latent.n;
        let mut h = DevBuf::f32(g, n * c)?;
        self.from_latent.apply(nsd, k, &self.scratch, lat.ptr(), n, h.ptr(), Dt::F32)?;
        drop(lat);
        let total: usize = self.stages.iter().map(|(b, u)| b.len() + u.is_some() as usize).sum();
        let mut done = 0u32;
        for (si, (blocks, up)) in self.stages.iter().enumerate() {
            if plan.upsample == Some(si) {
                out.coords = coords;
                return Ok(out);
            }
            for b in blocks {
                let hh = DevBuf::new(g, n * c * 2)?;
                k.convert(h.ptr(), Dt::F32, n * c, hh.ptr(), Dt::F16)?;
                let cv = DevBuf::f32(g, n * c)?;
                b.conv.apply(cx, &hh, n, &nmap, cv.ptr(), Dt::F32)?;
                k.ln_mod(cv.ptr(), Dt::F32, n, c, 1e-6, b.norm.0.fp(), b.norm.1.fp(), std::ptr::null(), std::ptr::null(), hh.ptr(), Dt::F16)?;
                drop(cv);
                // the MLP a row block at a time (its 4C-wide middle is the largest buffer here)
                let rows = ((256usize << 20) / (b.mlp0.n * 2)).clamp(1, n);
                let mid = DevBuf::new(g, rows * b.mlp0.n * 2)?;
                for r0 in (0..n).step_by(rows) {
                    let m = rows.min(n - r0);
                    // SAFETY: row blocks inside hh (half [n, c]) and h (float32 [n, c]).
                    let (xi, xo) = unsafe { (hh.ptr().cast::<u16>().add(r0 * c).cast(), h.fp().add(r0 * c)) };
                    b.mlp0.apply(nsd, k, &self.scratch, xi, m, mid.ptr(), Dt::F16)?;
                    k.silu(mid.ptr(), Dt::F16, m * b.mlp0.n)?;
                    let o = DevBuf::f32(g, m * c)?;
                    b.mlp2.apply(nsd, k, &self.scratch, mid.ptr(), m, o.ptr(), Dt::F32)?;
                    k.add(xo.cast(), Dt::F32, o.ptr(), Dt::F32, m * c)?;
                }
                done += 1;
                progress(done, total as u32)?;
            }
            let Some(u) = up else { continue };
            if u.cin != c {
                return Err(Error(format!("stage {si}'s up block takes {} channels, the stage has {c}", u.cin)));
            }
            out.level_coords.push(coords.clone());
            // which children: the logits (shape) or the shape decoder's tree (texture)
            let sub = if let Some(t) = plan.guide {
                t.levels.get(si).cloned().ok_or_else(|| Error(format!("the guide has no level {si}")))?
            } else {
                let ts = u.to_subdiv.as_ref().ok_or_else(|| Error("a decoder without to_subdiv needs the shape decoder's subdivisions".into()))?;
                let hh = DevBuf::new(g, n * c * 2)?;
                k.convert(h.ptr(), Dt::F32, n * c, hh.ptr(), Dt::F16)?;
                let lg = DevBuf::f32(g, n * 8)?;
                ts.apply(nsd, k, &self.scratch, hh.ptr(), n, lg.ptr(), Dt::F32)?;
                let lg = lg.to_f32()?;
                let s = subdivide(&coords, |p, o| lg[p * 8 + o] > 0.0);
                out.subs.push(lg);
                s
            };
            if sub.parent.is_empty() {
                return Err(Error(format!("the decoder kept no voxels at level {}", si + 1)));
            }
            // conv1 over the parents (norm1, SiLU), its 8 Cout channels split into the kept children
            let hh = DevBuf::new(g, n * c * 2)?;
            k.ln_mod(h.ptr(), Dt::F32, n, c, 1e-6, u.norm1.0.fp(), u.norm1.1.fp(), std::ptr::null(), std::ptr::null(), hh.ptr(), Dt::F16)?;
            k.silu(hh.ptr(), Dt::F16, n * c)?;
            let c1 = DevBuf::new(g, n * 8 * u.cout * 2)?;
            u.conv1.apply(cx, &hh, n, &nmap, c1.ptr(), Dt::F16)?;
            drop(hh);
            let nc = sub.parent.len();
            let idx: Vec<i32> = sub.parent.iter().zip(&sub.sub).map(|(p, s)| p * 8 + s).collect();
            let up_i = |v: &[i32]| -> Result<DevBuf> { upload_nmap(g, v) };
            let (idx_d, par_d, sub_d) = (up_i(&idx)?, up_i(&sub.parent)?, up_i(&sub.sub)?);
            let hc = DevBuf::new(g, nc * u.cout * 2)?;
            k.gather_rows(c1.ptr(), Dt::F16, u.cout, idx_d.ptr().cast(), nc, hc.ptr())?;
            drop(c1);
            let xs = DevBuf::f32(g, nc * u.cout)?;
            k.c2s_skip(h.fp(), c, par_d.ptr().cast(), sub_d.ptr().cast(), nc, u.cout, u.cout / (c / 8), xs.fp())?;
            drop(h);
            k.ln_mod(hc.ptr(), Dt::F16, nc, u.cout, 1e-6, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), hc.ptr(), Dt::F16)?;
            k.silu(hc.ptr(), Dt::F16, nc * u.cout)?;
            nmap_h = child_neighbours(&nmap_h, &sub);
            drop(nmap);
            nmap = upload_nmap(g, &nmap_h)?;
            let c2 = DevBuf::f32(g, nc * u.cout)?;
            u.conv2.apply(cx, &hc, nc, &nmap, c2.ptr(), Dt::F32)?;
            k.add(xs.ptr(), Dt::F32, c2.ptr(), Dt::F32, nc * u.cout)?;
            h = xs;
            n = nc;
            c = u.cout;
            coords.clone_from(&sub.coords);
            if plan.keep_ups {
                out.ups.push(h.to_f32()?);
            }
            out.tree.levels.push(sub);
            done += 1;
            progress(done, total as u32)?;
        }
        if plan.upsample.is_some() {
            out.coords = coords;
            return Ok(out);
        }
        let hh = DevBuf::new(g, n * c * 2)?;
        k.ln_mod(h.ptr(), Dt::F32, n, c, 1e-5, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), hh.ptr(), Dt::F16)?;
        let o = DevBuf::f32(g, n * self.out_ch)?;
        self.out.apply(nsd, k, &self.scratch, hh.ptr(), n, o.ptr(), Dt::F32)?;
        out.feats = o.to_f32()?;
        out.coords = coords;
        out.nmap = nmap_h;
        Ok(out)
    }
}
