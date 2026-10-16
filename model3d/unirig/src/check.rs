//! Stage by stage against the reference (reference/unirig/ref.py: UniRig's own modules in float32 on the CPU). Each
//! stage starts from the reference's own input - its samples, its random subsets, its skeleton - so a difference
//! shows where it is born:
//!
//! ```text
//!   encoder   skel_pts + skel_nrm + enc_ind + enc_fps  -> enc_lat.npy, cond.npy
//!   fps       skel_pts + enc_ind                       -> enc_fps.npy (the farthest points: how many agree)
//!   logits    cond + greedy.npy                        -> greedy_logits.npy (teacher-forced)
//!   greedy    cond                                     -> greedy.npy (the tokens, greedy with the grammar)
//!   detok     beam.npy                                 -> joints.npy, parents.npy
//!   order     skin_pts                                 -> ptv3_order.npy (the curves' orders, given ptv3_perm)
//!   skin      skin_pts + skin_nrm + skin_vox + skin_joints + skin_enc_ind / fps + ptv3_order
//!                                                      -> skin_glob, skin_bone, skin_ptv3, skin_mesh, skin .npy
//!   voxel     mesh_v + mesh_f + skin_norm + skin_joints -> voxel_skin.npy (the prior; voxelization and shortest paths)
//!   reskin    skin_pts + skin + mesh_v + mesh_f        -> skin_v.npy (the weights on the mesh's vertices)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_model3d::{Error, Result};

use crate::{encoder, search, tokens, UniRig, START};

/// A little-endian .npy of float32 or int64: (shape, values as f64)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head} (float32 or int64 expected)", path.display())));
    };
    Ok((shape, v))
}

pub fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

pub fn points(v: &[f64]) -> Vec<[f32; 3]> {
    v.chunks_exact(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
}

/// How far `got` is from `want`: (relative RMS error, max abs error, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (g2.sqrt() * w2.sqrt()).max(1e-30))
}

pub fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    if got.len() != want.len() {
        log(format!("{what:<12} {} values here, {} in the reference", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<12} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.7}  ({} values)", want.len()));
    rel
}

/// Every stage of `stages` against the dumps in `dir`; the worst relative error
pub fn run(m: &UniRig, dir: &Path, stages: &[&str], _mesh: Option<&Path>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut note = |r: f64| worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
    let rd = |n: &str| read_npy(&dir.join(format!("{n}.npy")));
    let need = |s: &str| stages.contains(&s);
    if need("fps") {
        let pts = points(&rd("skel_pts")?.1);
        let ind: Vec<usize> = rd("enc_ind")?.1.iter().map(|x| *x as usize).collect();
        let sub: Vec<[f32; 3]> = ind.iter().map(|&i| pts[i]).collect();
        let want: Vec<usize> = rd("enc_fps")?.1.iter().map(|x| *x as usize).collect();
        let got = encoder::fps(&sub, want.len());
        let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        log(format!("fps          the first {same} of {} indices agree", want.len()));
        if same < want.len() {
            note(1.0);
        }
    }
    let mut prefix = None;
    if need("encoder") || need("logits") || need("greedy") {
        let pts = points(&rd("skel_pts")?.1);
        let nrm = points(&rd("skel_nrm")?.1);
        let ind: Vec<usize> = rd("enc_ind")?.1.iter().map(|x| *x as usize).collect();
        let fps: Vec<usize> = rd("enc_fps")?.1.iter().map(|x| *x as usize).collect();
        let q: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        let t0 = Instant::now();
        let (lat, x) = m.prefix(&pts, &nrm, &q)?;
        m.ops.gpu.sync()?;
        log(format!("encoder      {:.2} s", t0.elapsed().as_secs_f64()));
        if need("encoder") {
            note(line("enc latents", &lat.to_f32()?, &f32s(&rd("enc_lat")?.1), log));
            let c = x.to_f32()?;
            note(line("cond", &c[..encoder::TOKENS * crate::opt::WIDTH], &f32s(&rd("cond")?.1), log));
        }
        prefix = Some(x);
    }
    if need("logits") {
        let x = prefix.as_ref().ok_or_else(|| Error("no prefix".into()))?;
        let toks: Vec<u32> = rd("greedy")?.1.iter().map(|x| *x as u32).collect();
        let t0 = Instant::now();
        let got = m.logits(x, &toks)?;
        log(format!("logits       {} positions in {:.2} s", toks.len() + 1, t0.elapsed().as_secs_f64()));
        let want = f32s(&rd("greedy_logits")?.1);
        note(line("logits", &got, &want, log));
        let am = |v: &[f32]| v.chunks_exact(tokens::VOCAB).map(|r| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap_or(0)).collect::<Vec<_>>();
        let same = am(&got).iter().zip(am(&want)).filter(|(a, b)| **a == *b).count();
        log(format!("logits       the argmax agrees at {same} of {} positions", toks.len() + 1));
    }
    if need("greedy") {
        let x = prefix.as_ref().ok_or_else(|| Error("no prefix".into()))?;
        let want: Vec<u32> = rd("greedy")?.1.iter().map(|x| *x as u32).collect();
        let t0 = Instant::now();
        let s = search::Settings { beams: 1, sample: false, ..Default::default() };
        let got = m.generate(x, s, &mut crate::ops::Rng::new(0), &mut |_| Ok(()))?;
        let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        log(format!("greedy       {} tokens here, {} there, the first {same} the same ({:.2} s)", got.len(), want.len(), t0.elapsed().as_secs_f64()));
        if got != want {
            note(1.0);
        }
    }
    if need("detok") {
        let beam: Vec<u32> = rd("beam")?.1.iter().map(|x| *x as u32).collect();
        let mut ids = vec![START[0]];
        ids.extend(&beam);
        ids.push(tokens::EOS);
        let sk = tokens::detokenize(&ids).map_err(Error)?;
        let got: Vec<f32> = sk.joints.iter().flatten().copied().collect();
        note(line("joints", &got, &f32s(&rd("joints")?.1), log));
        let want: Vec<i64> = rd("parents")?.1.iter().map(|x| *x as i64).collect();
        let gp: Vec<i64> = sk.parents.iter().map(|p| p.map_or(-1, |x| x as i64)).collect();
        log(format!("parents      {}", if gp == want { "the same".to_string() } else { format!("DIFFERENT: {gp:?} here, {want:?} there") }));
        if gp != want {
            note(1.0);
        }
    }
    if need("order") {
        let pts = points(&rd("skin_pts")?.1);
        let perm: Vec<usize> = rd("ptv3_perm")?.1.iter().map(|x| *x as usize).collect();
        let (_, want) = rd("ptv3_order")?;
        let gc = crate::ptv3::grid_coords(&pts);
        let got = crate::ptv3::orders(&gc, [perm[0], perm[1], perm[2], perm[3]]);
        let codes = crate::ptv3::codes(&gc);
        for (k, o) in got.iter().enumerate() {
            let w = &want[k * pts.len()..(k + 1) * pts.len()];
            let same = o.iter().zip(w).filter(|(a, b)| **a as f64 == **b).count();
            // equal codes (points sharing a voxel) may come in either order
            let c = &codes[perm[k]];
            let tied = o.iter().zip(w).filter(|(a, b)| **a as f64 != **b && c[**a] == c[**b as usize]).count();
            log(format!("order {k}      {same} of {} the same, {tied} more at equal codes", pts.len()));
            if same + tied != pts.len() {
                note(1.0);
            }
        }
    }
    if need("skin") {
        let pts = points(&rd("skin_pts")?.1);
        let nrm = points(&rd("skin_nrm")?.1);
        let joints = points(&rd("skin_joints")?.1);
        let prior = f32s(&rd("skin_vox")?.1);
        let ind: Vec<usize> = rd("skin_enc_ind")?.1.iter().map(|x| *x as usize).collect();
        let fps: Vec<usize> = rd("skin_enc_fps")?.1.iter().map(|x| *x as usize).collect();
        let q: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        let (_, ord) = rd("ptv3_order")?;
        let n = pts.len();
        let lay = crate::ptv3::Layout { gc: crate::ptv3::grid_coords(&pts), orders: (0..4).map(|k| ord[k * n..(k + 1) * n].iter().map(|x| *x as usize).collect()).collect() };
        let parents: Vec<Option<usize>> = rd("parents")?.1.iter().map(|x| if *x < 0.0 { None } else { Some(*x as usize) }).collect();
        let mut tr = crate::skin::Trace::default();
        let t0 = Instant::now();
        let soft = m.skin.predict(&m.ops, &pts, &nrm, &q, &lay, &joints, &prior, Some(&mut tr))?;
        log(format!("skin         {:.2} s", t0.elapsed().as_secs_f64()));
        note(line("skin glob", &tr.glob, &f32s(&rd("skin_glob")?.1), log));
        note(line("skin bone", &tr.bone, &f32s(&rd("skin_bone")?.1), log));
        note(line("skin ptv3", &tr.ptv3, &f32s(&rd("skin_ptv3")?.1), log));
        note(line("skin mesh", &tr.mesh, &f32s(&rd("skin_mesh")?.1), log));
        let w = crate::skin::masked(&soft, &prior, &parents, 3.0);
        note(line("skin", &w, &f32s(&rd("skin")?.1), log));
    }
    if need("voxel") || need("reskin") {
        let mv = points(&rd("mesh_v")?.1);
        let mf: Vec<[u32; 3]> = rd("mesh_f")?.1.chunks_exact(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect();
        let nn = rd("skin_norm")?.1;
        let (c, s) = ([nn[0] as f32, nn[1] as f32, nn[2] as f32], nn[3] as f32);
        let nv: Vec<[f32; 3]> = mv.iter().map(|p| crate::prep::apply(*p, c, s)).collect();
        let parents: Vec<Option<usize>> = rd("parents")?.1.iter().map(|x| if *x < 0.0 { None } else { Some(*x as usize) }).collect();
        if need("voxel") {
            let joints = points(&rd("skin_joints")?.1);
            let t0 = Instant::now();
            let got = crate::voxel::voxel_skin(&nv, &mf, &joints, 196);
            log(format!("voxel        {:.2} s", t0.elapsed().as_secs_f64()));
            let want = f32s(&rd("voxel_skin")?.1);
            note(line("voxel skin", &got, &want, log));
            let j = joints.len();
            let am = |v: &[f32]| v.chunks_exact(j).map(|r| (0..j).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap_or(0)).collect::<Vec<_>>();
            let same = am(&got).iter().zip(am(&want)).filter(|(a, b)| **a == *b).count();
            log(format!("voxel skin   the largest joint agrees at {same} of {} vertices", nv.len()));
        }
        if need("reskin") {
            let pts = points(&rd("skin_pts")?.1);
            let sw = f32s(&rd("skin")?.1);
            let t0 = Instant::now();
            let got = crate::reskin::reskin(&pts, &sw, &nv, &mf, &parents);
            log(format!("reskin       {:.2} s", t0.elapsed().as_secs_f64()));
            note(line("skin on mesh", &got, &f32s(&rd("skin_v")?.1), log));
        }
    }
    Ok(worst)
}
