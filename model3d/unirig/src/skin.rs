//! The skin model (`unirig_skin.py`, UniRigSkin): a global shape encoding (Michelangelo again), the bones encoded and
//! refined by 8 cross-attention blocks over themselves and it, every sample's PTv3 feature refined by 16 cross-attention
//! blocks over the bones and the shape, then the head: per head a softmax over the bones of the sample's query
//! against each bone's key, beside the voxel prior, through a small MLP to one logit a (sample, bone); a softmax over
//! the bones, times the prior's mask cubed, renormalized.

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::encoder::{self, Encoder};
use crate::ffi::GELU;
use crate::ops::{Lin, Ln, Ops, Weights};
use crate::ptv3::{Layout, Ptv3};

pub const FEAT: usize = 768;
const HEADS: usize = 16;
const BONE_EMBED: usize = 1024;
const MLP: usize = 512;
/// samples the head takes at once
const CHUNK: usize = 2048;

/// Linear, layer norm, GELU
struct Lng {
    l: Lin,
    n: Ln,
}

impl Lng {
    fn load(w: &Weights, lin: &str, ln: &str, n: usize, k: usize) -> Result<Lng> {
        Ok(Lng { l: w.lin(lin, n, k, true)?, n: w.ln(ln, n)? })
    }
    fn run(&self, o: &Ops, x: &DevBuf, rows: usize, out: &DevBuf) -> Result<()> {
        o.linear(x, rows, &self.l, out)?;
        o.ln(out, rows, self.l.n, Some(&self.n), out)?;
        o.k.act(out, rows * self.l.n, GELU)
    }
}

/// flash_attn's MHA (cross) in a post-norm residual block with an FFN
struct CrossBlock {
    wq: Lin,
    wkv: Lin,
    out: Lin,
    norm1: Ln,
    norm2: Ln,
    ffn0: Lin,
    ffn2: Lin,
}

impl CrossBlock {
    fn load(w: &Weights, p: &str) -> Result<CrossBlock> {
        Ok(CrossBlock {
            wq: w.lin(&format!("{p}.attention.Wq"), FEAT, FEAT, true)?,
            wkv: w.lin(&format!("{p}.attention.Wkv"), 2 * FEAT, FEAT, true)?,
            out: w.lin(&format!("{p}.attention.out_proj"), FEAT, FEAT, true)?,
            norm1: w.ln(&format!("{p}.norm1"), FEAT)?,
            norm2: w.ln(&format!("{p}.norm2"), FEAT)?,
            ffn0: w.lin(&format!("{p}.ffn.0"), 4 * FEAT, FEAT, true)?,
            ffn2: w.lin(&format!("{p}.ffn.2"), FEAT, 4 * FEAT, true)?,
        })
    }

    /// x [n, 768] = norm2(y + ffn(y)), y = norm1(x + attn(x, kv)); kvp: Wkv of the keys' rows [m, 1536] (shared)
    #[allow(clippy::too_many_arguments)]
    fn run(&self, o: &Ops, x: &DevBuf, n: usize, kvp: &DevBuf, m: usize, t: (&DevBuf, &DevBuf, &DevBuf), scratch: &DevBuf) -> Result<()> {
        let (q, a, h) = t;
        let d = FEAT / HEADS;
        o.linear(x, n, &self.wq, q)?;
        let v = kvp.view(FEAT * 4, kvp.len - FEAT * 4)?;
        o.k.attention((q, FEAT, d), (kvp, 2 * FEAT, d), (&v, 2 * FEAT, d), 1, n, m, HEADS, d, d, 1.0 / (d as f32).sqrt(), false, scratch, (a, FEAT))?;
        o.linear(a, n, &self.out, q)?;
        o.k.add(x, q, n * FEAT)?;
        o.ln(x, n, FEAT, Some(&self.norm1), x)?;
        o.linear(x, n, &self.ffn0, h)?;
        o.k.act(h, n * 4 * FEAT, GELU)?;
        o.linear(h, n, &self.ffn2, q)?;
        o.k.add(x, q, n * FEAT)?;
        o.ln(x, n, FEAT, Some(&self.norm2), x)
    }
}

pub struct SkinModel {
    pub global: Encoder,
    out_proj: Lng,
    pub ptv3: Ptv3,
    feat_map: Lng,
    bone: [Lng; 3],
    bone_attn: Vec<CrossBlock>,
    mesh_attn: Vec<CrossBlock>,
    qmesh: Lin,
    kmesh: Lin,
    attn_norm: Ln,
    vox_w: DevBuf,
    vox_b: DevBuf,
    vox_norm: Ln,
    down: Lng,
    pred: Vec<Lng>,
    last: Lin,
}

/// The skin stage's intermediates, for the checks
#[derive(Default)]
pub struct Trace {
    pub glob: Vec<f32>,
    pub bone: Vec<f32>,
    pub ptv3: Vec<f32>,
    pub mesh: Vec<f32>,
}

/// The bones' frequency embedding (6 frequencies of pi 2^k, the input first): 39 a joint
pub fn bone_features(joints: &[[f32; 3]], min: [f32; 3]) -> Vec<f32> {
    let f: Vec<f32> = (0..6).map(|k| (1u32 << k) as f32 * std::f32::consts::PI).collect();
    let mut out = Vec::with_capacity(joints.len() * 39);
    for j in joints {
        let x = [j[0] - min[0], j[1] - min[1], j[2] - min[2]];
        let e: Vec<f32> = (0..3).flat_map(|c| f.iter().map(move |fk| x[c] * fk)).collect();
        out.extend_from_slice(&x);
        out.extend(e.iter().map(|v| v.sin()));
        out.extend(e.iter().map(|v| v.cos()));
    }
    out
}

impl SkinModel {
    pub fn load(w: &Weights) -> Result<SkinModel> {
        let sub = |p: &str| Weights { pth: w.pth, prefix: format!("{}{p}", w.prefix), gpu: w.gpu };
        let mut pred = vec![Lng::load(w, "skinweight_pred.net.0", "skinweight_pred.net.1", MLP, HEADS)?];
        for i in 1..4 {
            pred.push(Lng::load(w, &format!("skinweight_pred.net.{}", 3 * i), &format!("skinweight_pred.net.{}", 3 * i + 1), MLP, MLP)?);
        }
        let (s, vw) = w.host("voxel_skin_embed.weight")?;
        if s != [HEADS, 1] {
            return Err(Error(format!("voxel_skin_embed.weight is {s:?}")));
        }
        Ok(SkinModel {
            global: Encoder::load(&sub("global_encoder."))?,
            out_proj: Lng::load(w, "out_proj.0", "out_proj.1", FEAT, encoder::WIDTH)?,
            ptv3: Ptv3::load(&sub("mesh_encoder."))?,
            feat_map: Lng::load(w, "feat_map.0", "feat_map.1", FEAT, crate::ptv3::OUT)?,
            bone: [
                Lng::load(w, "bone_encoder.bone_encoder.1", "bone_encoder.bone_encoder.2", BONE_EMBED, 39)?,
                Lng::load(w, "bone_encoder.bone_encoder.4", "bone_encoder.bone_encoder.5", 4 * BONE_EMBED, BONE_EMBED)?,
                Lng::load(w, "bone_encoder.bone_encoder.7", "bone_encoder.bone_encoder.8", FEAT, 4 * BONE_EMBED)?,
            ],
            bone_attn: (0..8).map(|i| CrossBlock::load(w, &format!("bone_encoder.attn.{i}"))).collect::<Result<_>>()?,
            mesh_attn: (0..16).map(|i| CrossBlock::load(w, &format!("mesh_bone_attn.{i}"))).collect::<Result<_>>()?,
            qmesh: w.lin("qmesh", HEADS * FEAT, FEAT, true)?,
            kmesh: w.lin("kmesh", HEADS * FEAT, FEAT, true)?,
            attn_norm: w.ln("attn_skin_norm", HEADS)?,
            vox_w: DevBuf::from_f32(w.gpu, &vw)?,
            vox_b: w.dev("voxel_skin_embed.bias", &[HEADS])?,
            vox_norm: w.ln("voxel_skin_norm", HEADS)?,
            down: Lng::load(w, "downscale.0", "downscale.1", HEADS, 2 * HEADS)?,
            pred,
            last: w.lin("skinweight_pred.net.12", 1, MLP, true)?,
        })
    }

    /// The samples' weights [n, J] (softmax over the bones, before the prior's mask): the samples `pts` / `nrm`, the
    /// encoder's queries, PTv3's layout, the joints (the skin stage's frame) and the prior sampled [n, J]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&self, o: &Ops, pts: &[[f32; 3]], nrm: &[[f32; 3]], queries: &[usize], lay: &Layout, joints: &[[f32; 3]], prior: &[f32],
                   mut trace: Option<&mut Trace>) -> Result<Vec<f32>> {
        let n = pts.len();
        let j = joints.len();
        crate::prof::reset_clock(&o.nsd)?;
        // the shape
        let lat = self.global.encode(o, pts, nrm, queries)?;
        let t = encoder::TOKENS;
        let glob = o.buf(t * FEAT)?;
        self.out_proj.run(o, &lat, t, &glob)?;
        drop(lat);
        crate::prof::mark(&o.nsd, "skin shape encoder")?;
        // the bones
        let mut min = [f32::INFINITY; 3];
        for p in pts {
            for c in 0..3 {
                min[c] = min[c].min(p[c]);
            }
        }
        let bf = DevBuf::from_f32(&o.gpu, &bone_features(joints, min))?;
        let (b1, b2) = (o.buf(j * BONE_EMBED)?, o.buf(j * 4 * BONE_EMBED)?);
        self.bone[0].run(o, &bf, j, &b1)?;
        self.bone[1].run(o, &b1, j, &b2)?;
        let lat2 = o.buf((j + t) * FEAT)?;
        let bx = lat2.view(0, j * FEAT * 4)?;
        self.bone[2].run(o, &b2, j, &bx)?;
        lat2.copy_within(j * FEAT * 4, &glob, 0, t * FEAT * 4)?;
        let bone = o.buf(j * FEAT)?;
        bone.copy_within(0, &bx, 0, j * FEAT * 4)?;
        let kvp = o.buf((j + t) * 2 * FEAT)?;
        let scratch = o.buf((64usize << 20).max(HEADS * (j + t)))?;
        {
            let (q, a, h) = (o.buf(j * FEAT)?, o.buf(j * FEAT)?, o.buf(j * 4 * FEAT)?);
            for b in &self.bone_attn {
                // the keys: the bones as encoded (not as refined) and the shape
                o.linear(&lat2, j + t, &b.wkv, &kvp)?;
                b.run(o, &bone, j, &kvp, j + t, (&q, &a, &h), &scratch)?;
            }
        }
        crate::prof::mark(&o.nsd, "skin bones")?;
        // the samples
        let feat: Vec<f32> = pts.iter().zip(nrm).flat_map(|(p, q)| [p[0], p[1], p[2], q[0], q[1], q[2], 0.0, 0.0, 0.0]).collect();
        let pf = self.ptv3.forward(o, &feat, lay)?;
        let mesh = o.buf(n * FEAT)?;
        self.feat_map.run(o, &pf, n, &mesh)?;
        drop(pf);
        crate::prof::mark(&o.nsd, "skin ptv3")?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.glob = glob.to_f32()?;
            tr.bone = bone.to_f32()?;
            tr.ptv3 = mesh.to_f32()?;
        }
        lat2.copy_within(0, &bone, 0, j * FEAT * 4)?;
        {
            let (q, a, h) = (o.buf(n * FEAT)?, o.buf(n * FEAT)?, o.buf(n * 4 * FEAT)?);
            // the scores of many sample rows at once (fewer passes of the attention)
            let big = o.buf((n * HEADS * (j + t)).min(192 << 20).max(HEADS * (j + t)))?;
            for b in &self.mesh_attn {
                o.linear(&lat2, j + t, &b.wkv, &kvp)?;
                b.run(o, &mesh, n, &kvp, j + t, (&q, &a, &h), &big)?;
            }
        }
        if let Some(tr) = trace {
            tr.mesh = mesh.to_f32()?;
        }
        crate::prof::mark(&o.nsd, "skin mesh-bone attention")?;
        // the head, a chunk of samples at a time
        let kb = o.buf(j * HEADS * FEAT)?;
        o.linear(&bone, j, &self.kmesh, &kb)?;
        let c = CHUNK.min(n);
        let (qm, sc, xin, x16, x512a, x512b, lg) = (o.buf(c * HEADS * FEAT)?, o.buf(c * j * HEADS)?, o.buf(c * j * 2 * HEADS)?, o.buf(c * j * HEADS)?,
                                                     o.buf(c * j * MLP)?, o.buf(c * j * MLP)?, o.buf(c * j)?);
        let pv = DevBuf::from_f32(&o.gpu, prior)?;
        let mut logits = vec![0f32; n * j];
        for r0 in (0..n).step_by(c) {
            let r = c.min(n - r0);
            let rj = r * j;
            o.linear(&mesh.view(r0 * FEAT * 4, r * FEAT * 4)?, r, &self.qmesh, &qm)?;
            o.k.bone_attn(&qm, &kb, r, j, HEADS, FEAT, 1.0 / (FEAT as f32).sqrt(), &sc)?;
            o.k.skin_in(&sc, &pv.view(r0 * j * 4, rj * 4)?, rj, HEADS, &self.attn_norm.w, &self.attn_norm.b, &self.vox_w, &self.vox_b, &self.vox_norm.w,
                        &self.vox_norm.b, &xin)?;
            self.down.run(o, &xin, rj, &x16)?;
            self.pred[0].run(o, &x16, rj, &x512a)?;
            self.pred[1].run(o, &x512a, rj, &x512b)?;
            self.pred[2].run(o, &x512b, rj, &x512a)?;
            self.pred[3].run(o, &x512a, rj, &x512b)?;
            o.linear(&x512b, rj, &self.last, &lg)?;
            let v = lg.to_f32()?;
            logits[r0 * j..r0 * j + rj].copy_from_slice(&v[..rj]);
        }
        crate::prof::mark(&o.nsd, "skin head")?;
        crate::prof::report(&mut |s| eprintln!("{s}"));
        // softmax over the bones
        for row in logits.chunks_exact_mut(j) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        Ok(logits)
    }
}

/// The prior's mask: each bone's prior plus its children's (in bone order, as the reference adds them), cubed, times
/// the softmax; renormalized
pub fn masked(soft: &[f32], prior: &[f32], parents: &[Option<usize>], power: f32) -> Vec<f32> {
    let j = parents.len();
    let mut out = soft.to_vec();
    for (row, pr) in out.chunks_exact_mut(j).zip(prior.chunks_exact(j)) {
        let mut mask = pr.to_vec();
        for (i, p) in parents.iter().enumerate() {
            if let Some(p) = p {
                mask[*p] += mask[i];
            }
        }
        for (x, m) in row.iter_mut().zip(&mask) {
            *x *= m.powf(power);
        }
        let s: f32 = row.iter().sum();
        if s > 0.0 {
            row.iter_mut().for_each(|x| *x /= s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_adds_children_into_parents() {
        // bones 0 <- 1 <- 2; a sample whose prior is all on bone 2
        let w = masked(&[0.2, 0.3, 0.5], &[0.0, 0.0, 1.0], &[None, Some(0), Some(1)], 3.0);
        // bone 1 gets bone 2's prior (1), bone 0 gets bone 1's prior as it was when added (0, before bone 2 came)
        assert_eq!(w[0], 0.0);
        assert!((w[1] - 0.3 / 0.8).abs() < 1e-6 && (w[2] - 0.5 / 0.8).abs() < 1e-6);
        let f = bone_features(&[[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0]);
        assert_eq!(f.len(), 39);
        assert_eq!(&f[..3], &[0.0, 1.0, 2.0]);
    }
}
