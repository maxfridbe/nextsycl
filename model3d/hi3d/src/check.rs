//! Stage by stage against the reference (reference/hi3d/ref.py: sparc3d's own modules on the same files, float32 on
//! the CPU): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   pre      the picture (--picture)                 -> dino_in, clip_in (the framed square's encoder inputs)
//!   cond     dino_in, clip_in                         -> dino_raw, clip_raw, cond
//!   vox      the coarse mesh (--mesh; 128: mesh_64)   -> coords_S (the same voxels, the same order)
//!   dit      coords_S, noise_S, cond                  -> dit_S_<tap> (the first call: input, in0, in1, blk0, blk23, out0,
//!                                                        out1, vel) and dit_S_neg (the zero condition)
//!   steps    coords_S, noise_S, cond                  -> x_S_<k> after each update the reference ran, latents_S
//!   vae      coords_S, latents_S                      -> vae_S_z2f, vae_S_mid, each unpooling's logits (at the
//!                                                        reference's pruning), the voxels it keeps on its own, the
//!                                                        final sign logits and attributes
//!   mesh     vae_S_coords / sign / feat               -> mesh_S_raw, mesh_S (counts, Chamfer distance)
//!   post     mesh_S                                   -> final_S (counts, Chamfer distance)
//!   full     --picture, --mesh, the seed              -> final_S (the whole refinement; Chamfer distance)
//! ```
//! (S: --structure, 64 or 128.) Chamfer distances are in the reference mesh's units: the final meshes have radius 1.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use nextsycl_model3d::{Error, Result};

use crate::{dit, extract, post, Hi3d, Picture};

/// A little-endian float32 or int32 .npy: (shape, values as f32 - ints exactly up to 2^24)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: Fortran order", path.display())));
    }
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let body = &b[at + hl..];
    let v = if head.contains("'<f4'") {
        body.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if head.contains("'<i4'") {
        body.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32).collect()
    } else if head.contains("'<i8'") {
        body.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32).collect()
    } else {
        return Err(Error(format!("{}: {head} (float32 or int32 expected)", path.display())));
    };
    Ok((shape, v))
}

fn coords3(v: &[f32]) -> Vec<[i32; 3]> {
    v.chunks_exact(3).map(|c| [c[0] as i32, c[1] as i32, c[2] as i32]).collect()
}

/// (relative RMS error, max abs error, cosine)
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

/// The mean of each point's distance to the nearest of the other set, both ways, halved (a sample of up to 200,000
/// points a side; a uniform grid searched in growing shells)
pub fn chamfer(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let one = |from: &[[f32; 3]], to: &[[f32; 3]]| -> f64 {
        if from.is_empty() || to.is_empty() {
            return f64::INFINITY;
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in to {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0f32, f32::max).max(1e-9);
        let cells = (to.len() as f64).cbrt().clamp(1.0, 512.0) as i64;
        let cs = ext / cells as f32;
        let cell = |p: &[f32; 3]| -> [i64; 3] { [0, 1, 2].map(|k| ((p[k] - lo[k]) / cs).floor() as i64) };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in to.iter().enumerate() {
            grid.entry(cell(p)).or_default().push(i as u32);
        }
        let step = (from.len() / 200_000).max(1);
        let mut sum = 0f64;
        let mut n = 0usize;
        for p in from.iter().step_by(step) {
            let c = cell(p);
            let mut best = f32::INFINITY;
            for r in 0..=(2 * cells + 2) {
                // every cell at Chebyshev distance r; done once a found point is nearer than the shell's inner edge
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                continue;
                            }
                            if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                                for &j in v {
                                    let q = to[j as usize];
                                    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                                    best = best.min(d);
                                }
                            }
                        }
                    }
                }
                if best <= r as f32 * cs {
                    break;
                }
            }
            sum += best as f64;
            n += 1;
        }
        sum / n as f64
    };
    0.5 * (one(a, b) + one(b, a))
}

fn tri_of(dir: &Path, stem: &str) -> Result<extract::Tri> {
    let (_, v) = read_npy(&dir.join(format!("{stem}_v.npy")))?;
    let (_, f) = read_npy(&dir.join(format!("{stem}_f.npy")))?;
    Ok(extract::Tri { v: v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), f: f.chunks_exact(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect() })
}

/// Runs the stages; the worst relative error of the float stages
#[allow(clippy::too_many_arguments)]
pub fn run(e: &Hi3d, dir: &Path, stages: &[&str], structure: usize, picture: Option<&Picture>, mesh: Option<&nextsycl_model3d::Mesh>, seed: u64,
           log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let s = structure;
    let mut cmp = |name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)| -> Result<()> {
        if got.len() != want.len() {
            return Err(Error(format!("{name}: {} values against the reference's {}", got.len(), want.len())));
        }
        let (rel, mx, cos) = compare(got, want);
        log(format!("  {name:<22} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
        worst = worst.max(rel);
        Ok(())
    };
    let npy = |n: &str| read_npy(&dir.join(n)).map(|x| x.1);
    let want = |st: &str| stages.contains(&st) || stages.contains(&"all");
    if want("pre") || want("cond") {
        let pic = picture.ok_or_else(|| Error("pre / cond: --picture (the reference's picture)".into()))?;
        let framed = e.frame(pic, false)?;
        let (din, cin) = crate::prep::inputs(&framed);
        log("pre:".into());
        cmp("dino_in", &din, &npy("dino_in.npy")?, log)?;
        cmp("clip_in", &cin, &npy("clip_in.npy")?, log)?;
        if want("cond") {
            log("cond (from the reference's inputs):".into());
            let d = e.dino()?.forward(&e.cx, &npy("dino_in.npy")?)?;
            let c = e.clip()?.forward(&e.cx, &npy("clip_in.npy")?)?;
            cmp("dino_raw", &d, &npy("dino_raw.npy")?, log)?;
            cmp("clip_raw", &c, &npy("clip_raw.npy")?, log)?;
            let (_, _, cond) = e.condition(&framed)?;
            cmp("cond (own picture)", &cond, &npy("cond.npy")?, log)?;
        }
    }
    if want("vox") {
        let (pos, tris): (Vec<[f32; 3]>, Vec<[u32; 3]>) = if s == 64 {
            let m = mesh.ok_or_else(|| Error("vox 64: --mesh (the coarse mesh)".into()))?;
            (m.positions.clone(), m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect())
        } else {
            let t = tri_of(dir, &format!("mesh_{}", crate::LADDER[crate::LADDER.iter().position(|l| l.0 == s).unwrap_or(1) - 1].0))?;
            (t.v, t.f)
        };
        let budget = crate::LADDER.iter().find(|l| l.0 == s).map_or(80_000, |l| l.1);
        let (got, grid) = crate::vox::adaptive(&pos, &tris, s, budget).map_err(Error)?;
        let want_c = coords3(&npy(&format!("coords_{s}.npy"))?);
        let a: HashSet<[i32; 3]> = got.iter().copied().collect();
        let b: HashSet<[i32; 3]> = want_c.iter().copied().collect();
        log(format!("vox {s}: {} voxels at {grid} (reference {}), {} only here, {} only there, same order: {}", got.len(), want_c.len(),
                    a.difference(&b).count(), b.difference(&a).count(), got == want_c));
    }
    let coords = || -> Result<Vec<[i32; 3]>> { Ok(coords3(&npy(&format!("coords_{s}.npy"))?)) };
    if want("dit") {
        let c = coords()?;
        let m = e.transformer(s)?;
        let geo = m.geo(&e.cx.gpu, &c)?;
        let cond = npy("cond.npy")?;
        let ch = nextsycl_core::DevBuf::new(&e.cx.gpu, cond.len() * 2)?;
        ch.write(0, &crate::w::half_bytes(&cond))?;
        let kvs = m.cache_cross(&e.cx, &ch, crate::COND_TOKENS)?;
        let neg = m.const_cross(&e.cx)?;
        let hats = npy(&format!("sigma_hats_{s}.npy"))?;
        let noise = npy(&format!("noise_{s}.npy"))?;
        let (own_s, own_h) = crate::sched::schedule(hats.len(), c.len(), e.sched.sigma_min, e.sched.time_rescale);
        log(format!("dit {s}: {} voxels, {} tokens; the schedule: sigmas {:.2e}, sigma_hats {:.2e} (rel)", c.len(), geo.p,
                    compare(&own_s, &npy(&format!("sigmas_{s}.npy"))?).0, compare(&own_h, &hats).0));
        let mut probe = dit::Probe::default();
        let v = m.forward(&e.cx, &geo, &noise, 1000.0 * hats[0], dit::Cross::Cached(&kvs, crate::COND_TOKENS), Some(&mut probe))?;
        for (name, t) in &probe.taps {
            cmp(name, t, &npy(&format!("dit_{s}_{name}.npy"))?, log)?;
        }
        cmp("vel", &v, &npy(&format!("dit_{s}_vel.npy"))?, log)?;
        let u = m.forward(&e.cx, &geo, &noise, 1000.0 * hats[0], dit::Cross::Const(&neg), None)?;
        cmp("neg (zero condition)", &u, &npy(&format!("dit_{s}_neg.npy"))?, log)?;
    }
    if want("steps") {
        let c = coords()?;
        let m = e.transformer(s)?;
        let k = (0..).take_while(|i| dir.join(format!("x_{s}_{i}.npy")).exists()).count();
        if k == 0 {
            return Err(Error(format!("steps: no x_{s}_0.npy in {}", dir.display())));
        }
        let full = dir.join(format!("latents_{s}.npy")).exists() && k == e.sched.steps as usize;
        let mut ups = Vec::new();
        let steps = e.sched.steps as usize;
        // a partial reference run: our sampler over the same schedule, stopped after its updates
        let r = e.sample(&m, &c, &npy("cond.npy")?, npy(&format!("noise_{s}.npy"))?, steps, e.sched.heun, Some(&mut ups), |at, _| {
            if at as usize >= k && !full { Err(Error("enough".into())) } else { Ok(()) }
        });
        log(format!("steps {s}: {k} updates of the reference's {}", if full { "whole run" } else { "partial run" }));
        for (i, x) in ups.iter().enumerate().take(k) {
            if i < 3 || i + 1 == k || i % 5 == 4 {
                cmp(&format!("x after update {}", i + 1), x, &npy(&format!("x_{s}_{i}.npy"))?, log)?;
            }
        }
        if full {
            cmp("latents", &r?, &npy(&format!("latents_{s}.npy"))?, log)?;
        }
    }
    if want("vae") {
        let c = coords()?;
        let lat = npy(&format!("latents_{s}.npy"))?;
        let vae = e.vae()?;
        let ups_ref: Vec<(Vec<[i32; 3]>, Vec<f32>)> = (0..).map_while(|i| {
            let cc = read_npy(&dir.join(format!("vae_{s}_up{i}_coords.npy"))).ok()?;
            let l = read_npy(&dir.join(format!("vae_{s}_up{i}_logits.npy"))).ok()?;
            Some((coords3(&cc.1), l.1))
        }).collect();
        // the reference's pruning, to compare every stage on the same voxels
        let mut probe = crate::vae::Probe { force: Some(ups_ref.clone()), ..Default::default() };
        let d = vae.decode(&e.cx, &c, &lat, s, Some(&mut probe))?;
        log(format!("vae {s} (at the reference's pruning):"));
        cmp("z_to_f", &probe.z2f, &npy(&format!("vae_{s}_z2f.npy"))?, log)?;
        cmp("mid", &probe.mid, &npy(&format!("vae_{s}_mid.npy"))?, log)?;
        for (i, ((pc, pl), (rc, rl))) in probe.ups.iter().zip(&ups_ref).enumerate() {
            let at: HashMap<[i32; 3], usize> = rc.iter().enumerate().map(|(j, c)| (*c, j)).collect();
            let theirs: Vec<f32> = pc.iter().flat_map(|c| rl[at[c] * 8..at[c] * 8 + 8].to_vec()).collect();
            let flips = pl.iter().zip(&theirs).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
            cmp(&format!("unpool {i} logits"), pl, &theirs, log)?;
            log(format!("    {} of {} children decided otherwise", flips, pl.len()));
        }
        let rc = coords3(&npy(&format!("vae_{s}_coords.npy"))?);
        let at: HashMap<[i32; 3], usize> = rc.iter().enumerate().map(|(j, c)| (*c, j)).collect();
        if d.coords.len() == rc.len() && d.coords.iter().all(|c| at.contains_key(c)) {
            let (rs, rf) = (npy(&format!("vae_{s}_sign.npy"))?, npy(&format!("vae_{s}_feat.npy"))?);
            let ts: Vec<f32> = d.coords.iter().flat_map(|c| rs[at[c] * 8..at[c] * 8 + 8].to_vec()).collect();
            let tf: Vec<f32> = d.coords.iter().flat_map(|c| rf[at[c] * 32..at[c] * 32 + 32].to_vec()).collect();
            cmp("sign logits", &d.sign, &ts, log)?;
            cmp("attributes", &d.feat, &tf, log)?;
        }
        let own = vae.decode(&e.cx, &c, &lat, s, None)?;
        let a: HashSet<[i32; 3]> = own.coords.iter().copied().collect();
        log(format!("vae {s} on its own: {} voxels kept (reference {}), {} only here, {} only there", own.coords.len(), rc.len(),
                    own.coords.iter().filter(|c| !at.contains_key(*c)).count(), rc.iter().filter(|c| !a.contains(*c)).count()));
    }
    if want("mesh") {
        let c = coords()?;
        let rc = coords3(&npy(&format!("vae_{s}_coords.npy"))?);
        let res = s * 8;
        let t0 = std::time::Instant::now();
        let raw = extract::to_mesh(&rc, res, &npy(&format!("vae_{s}_sign.npy"))?, &npy(&format!("vae_{s}_feat.npy"))?, extract::matrix(&c));
        let t1 = t0.elapsed().as_secs_f64();
        let filled = extract::fill_holes(&raw);
        let t2 = t0.elapsed().as_secs_f64() - t1;
        let (rr, rf) = (tri_of(dir, &format!("mesh_{s}_raw"))?, tri_of(dir, &format!("mesh_{s}"))?);
        log(format!("mesh {s} (from the reference's decoded voxels; FlexiCubes {t1:.1} s, holes {t2:.1} s):"));
        log(format!("  FlexiCubes: {} vertices, {} faces (reference {}, {}); Chamfer {:.3e}", raw.v.len(), raw.f.len(), rr.v.len(), rr.f.len(),
                    chamfer(&raw.v, &rr.v)));
        log(format!("  holes filled: {} faces (reference {})", filled.f.len(), rf.f.len()));
        let mat = npy(&format!("vae_{s}_matrix.npy"))?;
        let (sc, ctr) = extract::matrix(&c);
        log(format!("  the latent matrix: scale 1/{sc:.6} (reference 1/{:.6}), centre {:?}", 1.0 / mat[0], ctr));
    }
    if want("post") {
        let t = tri_of(dir, &format!("mesh_{s}"))?;
        let target = crate::LADDER.iter().find(|l| l.0 == s).map_or(1_000_000, |l| l.3);
        let t0 = std::time::Instant::now();
        let m = post::postprocess(&t, target);
        let f = tri_of(dir, &format!("final_{s}"))?;
        log(format!("post {s} (the reference's mesh; {:.1} s): {} vertices, {} faces (reference {}, {}); Chamfer {:.3e}", t0.elapsed().as_secs_f64(),
                    m.positions.len(), m.triangles(), f.v.len(), f.f.len(), chamfer(&m.positions, &f.v)));
    }
    if want("full") {
        let pic = picture.ok_or_else(|| Error("full: --picture".into()))?;
        let mesh = mesh.ok_or_else(|| Error("full: --mesh".into()))?;
        let res = crate::LADDER.iter().find(|l| l.0 == s).map_or(512, |l| l.2);
        let framed = e.frame(pic, false)?;
        let tris: Vec<[u32; 3]> = mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut quiet = |_: nextsycl_model3d::Step| Ok(());
        let (m, tr) = e.refine(&framed, &mesh.positions, &tris, res, e.sched.steps as usize, e.sched.heun, seed, None, &mut quiet)?;
        let f = tri_of(dir, &format!("final_{s}"))?;
        log(format!("full {res}: {} vertices, {} faces (reference {}, {}); Chamfer {:.3e}; passes {:?}", m.positions.len(), m.triangles(), f.v.len(),
                    f.f.len(), chamfer(&m.positions, &f.v), tr.passes));
        let out = dir.join(format!("ours_{s}.glb"));
        std::fs::write(&out, nextsycl_model3d::mesh::glb::write(&m).map_err(Error)?).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chamfer_of_a_shift() {
        let a: Vec<[f32; 3]> = (0..1000).map(|i| [(i % 10) as f32, (i / 10 % 10) as f32, (i / 100) as f32]).collect();
        let b: Vec<[f32; 3]> = a.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        assert!((chamfer(&a, &b) - 0.1).abs() < 1e-5);
        assert_eq!(chamfer(&a, &a), 0.0);
    }
}
