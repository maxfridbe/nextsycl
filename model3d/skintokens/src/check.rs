//! Stage by stage against the reference (reference/skintokens/ref.py: TokenRig's own modules, transcribed, in float32 on the
//! CPU). Each stage starts from the reference's own input - its samples, its random subsets, its tokens - so a difference
//! shows where it is born:
//!
//! ```text
//!   fps       samp_pts + enc_ind / vae_ind               -> enc_fps.npy, vae_fps.npy (the farthest points: how many agree)
//!   encoder   samp_pts + samp_nrm + enc_ind + enc_fps    -> enc_lat.npy, cond.npy (output_proj)
//!   vae       samp_pts + samp_nrm + vae_ind + vae_fps    -> cond_lat.npy (the skin VAE's condition tokens)
//!   logits    cond + greedy.npy                          -> greedy_logits.npy (teacher-forced, every position)
//!   greedy    cond                                       -> greedy.npy (the tokens, greedy with the grammar)
//!   detok     greedy.npy                                 -> joints.npy, parents.npy
//!   fsq       greedy.npy (its skin tokens)               -> fsq_z.npy (the codes' vectors)
//!   decoder   fsq_z + cond_lat + samp_pts + samp_nrm     -> dec_kv0.npy (the first joint's decoder states), skin.npy
//!   weights   skin + samp_pts + mesh_vnorm               -> skin_v.npy (on the vertices), weights.npy (four a vertex)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_core::DevBuf;
use nextsycl_model3d::mesh::tools;
use nextsycl_model3d::{Error, Result};

use crate::{encoder, qwen, search, tokens, SkinTokens};

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

fn indices(v: &[f64]) -> Vec<usize> {
    v.iter().map(|x| *x as usize).collect()
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
pub fn run(m: &SkinTokens, dir: &Path, stages: &[&str], _mesh: Option<&Path>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut note = |r: f64| worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
    let rd = |n: &str| read_npy(&dir.join(format!("{n}.npy")));
    let need = |s: &str| stages.contains(&s);
    if need("fps") {
        let pts = points(&rd("samp_pts")?.1);
        for (ind, fps) in [("enc_ind", "enc_fps"), ("vae_ind", "vae_fps")] {
            let sub: Vec<[f32; 3]> = indices(&rd(ind)?.1).iter().map(|&i| pts[i]).collect();
            let want = indices(&rd(fps)?.1);
            let got = encoder::fps(&sub, want.len());
            let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            log(format!("fps          {fps}: the first {same} of {} indices agree", want.len()));
            if same < want.len() {
                note(1.0);
            }
        }
    }
    let mut prefix = None;
    if need("encoder") || need("logits") || need("greedy") {
        let pts = points(&rd("samp_pts")?.1);
        let nrm = points(&rd("samp_nrm")?.1);
        let ind = indices(&rd("enc_ind")?.1);
        let fps = indices(&rd("enc_fps")?.1);
        let q: Vec<usize> = fps.iter().map(|&i| ind[i]).collect();
        let t0 = Instant::now();
        let (lat, x) = m.prefix(&pts, &nrm, &q)?;
        m.ops.gpu.sync()?;
        log(format!("encoder      {:.2} s", t0.elapsed().as_secs_f64()));
        if need("encoder") {
            note(line("enc latents", &lat.to_f32()?, &f32s(&rd("enc_lat")?.1), log));
            let c = x.to_f32()?;
            note(line("cond", &c[..encoder::TOKENS * qwen::WIDTH], &f32s(&rd("cond")?.1), log));
        }
        prefix = Some(x);
    }
    let greedy: Vec<u32> = if need("logits") || need("greedy") || need("detok") || need("fsq") { rd("greedy")?.1.iter().map(|x| *x as u32).collect() } else { Vec::new() };
    if need("vae") || need("decoder") {
        let pts = points(&rd("samp_pts")?.1);
        let nrm = points(&rd("samp_nrm")?.1);
        let t0 = Instant::now();
        let cond = m.vae.cond(&m.ops, &pts, &nrm, &indices(&rd("vae_ind")?.1), &indices(&rd("vae_fps")?.1))?;
        m.ops.gpu.sync()?;
        log(format!("vae cond     {:.2} s", t0.elapsed().as_secs_f64()));
        note(line("cond tokens", &cond.to_f32()?, &f32s(&rd("cond_lat")?.1), log));
    }
    if need("logits") {
        let x = prefix.as_ref().ok_or_else(|| Error("no prefix".into()))?;
        let t0 = Instant::now();
        let got = m.logits(x, &greedy)?;
        log(format!("logits       {} positions in {:.2} s", greedy.len() + 1, t0.elapsed().as_secs_f64()));
        let want = f32s(&rd("greedy_logits")?.1);
        note(line("logits", &got, &want, log));
        // per token: the worst row
        let mut worst_row = (0usize, 0f64);
        for (i, (g, w)) in got.chunks_exact(tokens::TOTAL).zip(want.chunks_exact(tokens::TOTAL)).enumerate() {
            let (rel, _, _) = compare(g, w);
            if rel > worst_row.1 {
                worst_row = (i, rel);
            }
        }
        log(format!("logits       the worst position is {} (rel rms {:.2e})", worst_row.0, worst_row.1));
        let am = |v: &[f32]| v.chunks_exact(tokens::TOTAL).map(|r| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap_or(0)).collect::<Vec<_>>();
        let same = am(&got).iter().zip(am(&want)).filter(|(a, b)| **a == *b).count();
        log(format!("logits       the argmax agrees at {same} of {} positions", greedy.len() + 1));
    }
    if need("greedy") {
        let x = prefix.as_ref().ok_or_else(|| Error("no prefix".into()))?;
        let t0 = Instant::now();
        let s = search::Settings { beams: 1, sample: false, ..Default::default() };
        let got = m.generate(x, s, &mut crate::ops::Rng::new(0), &mut |_| Ok(()))?;
        let same = got.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
        log(format!("greedy       {} tokens here, {} there, the first {same} the same ({:.2} s)", got.len(), greedy.len(), t0.elapsed().as_secs_f64()));
        if got != greedy {
            note(1.0);
        }
    }
    let mut read = None;
    if need("detok") || need("fsq") || need("decoder") || need("weights") {
        let (sk, codes) = m.read(&greedy)?;
        if need("detok") {
            let got: Vec<f32> = sk.joints.iter().flatten().copied().collect();
            note(line("joints", &got, &f32s(&rd("joints")?.1), log));
            let want: Vec<i64> = rd("parents")?.1.iter().map(|x| *x as i64).collect();
            let gp: Vec<i64> = sk.parents.iter().map(|p| p.map_or(-1, |x| x as i64)).collect();
            log(format!("parents      {}", if gp == want { "the same".to_string() } else { format!("DIFFERENT: {gp:?} here, {want:?} there") }));
            if gp != want {
                note(1.0);
            }
        }
        read = Some((sk.joints.len(), codes));
    }
    if need("fsq") || need("decoder") || need("weights") {
        let (j, codes) = read.as_ref().ok_or_else(|| Error("no tokens".into()))?;
        let z = m.vae.codes(codes);
        if need("fsq") {
            note(line("fsq z", &z, &f32s(&rd("fsq_z")?.1), log));
        }
        if need("decoder") || need("weights") {
            let pts = points(&rd("samp_pts")?.1);
            let nrm = points(&rd("samp_nrm")?.1);
            let cond = DevBuf::from_f32(&m.ops.gpu, &f32s(&rd("cond_lat")?.1))?;
            let want_z = f32s(&rd("fsq_z")?.1);
            let t0 = Instant::now();
            let (skin, kv0) = m.vae.decode(&m.ops, &want_z, *j, &cond, &pts, &nrm, &mut |_| Ok(()))?;
            log(format!("decoder      {j} joints over {} samples in {:.2} s", pts.len(), t0.elapsed().as_secs_f64()));
            note(line("decoder kv0", &kv0, &f32s(&rd("dec_kv0")?.1), log));
            let want = f32s(&rd("skin")?.1);
            note(line("skin", &skin, &want, log));
            let am = |v: &[f32]| v.chunks_exact(*j).map(|r| (0..*j).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap_or(0)).collect::<Vec<_>>();
            let same = am(&skin).iter().zip(am(&want)).filter(|(a, b)| **a == *b).count();
            log(format!("skin         the largest joint agrees at {same} of {} samples", pts.len()));
            if need("weights") {
                let vn = points(&rd("mesh_vnorm")?.1);
                let got = crate::knn::carry(&pts, &want, *j, &vn, crate::NEIGHBOURS);
                note(line("skin on mesh", &got, &f32s(&rd("skin_v")?.1), log));
                let mut top = vec![0f32; got.len()];
                for (v, row) in tools::skin::top4(&got, *j).iter().enumerate() {
                    for (jt, w) in row {
                        top[v * *j + *jt as usize] += w;
                    }
                }
                note(line("weights", &top,&f32s(&rd("weights")?.1), log));
            }
        }
    }
    Ok(worst)
}
