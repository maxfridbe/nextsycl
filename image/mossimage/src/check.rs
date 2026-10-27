//! The engine against reference/mossimage/ref.py's dumps (OpenMOSS's DiT classes on the GGUF dequantised, the text
//! encoder and the VAE from their original weights, on the CPU in float32): each stage fed the reference's own input,
//! so a stage's error is its own.
//!
//! ```text
//!   sigmas  the step count                    -> sigmas.npy
//!   dit     te_final / te_neg (the reference's) -> txt, emb and blocks 0 1 21 22 41 at step 0, then the velocities
//!                                                (cond, and uncond skipping block 22) at steps 0, n/2, n-1
//!   steps   lat_0 through every step          -> lat_final
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Result};

use crate::{dit, sigmas, te, MossImage, UNCOND_SKIP};

pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
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
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

/// (relative L2 error, max |difference|, cosine)
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
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}

fn note(name: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    let (rel, mx, cos) = compare(got, want);
    let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt();
    let worst_at = got.iter().zip(want).enumerate().max_by(|a, b| (a.1.0 - a.1.1).abs().total_cmp(&(b.1.0 - b.1.1).abs())).map(|(i, (g, w))| (i, *g, *w));
    if rel > 1e-2 {
        log(format!("    rms ours {:.4e} ref {:.4e}; worst element {:?}; non-finite ours {}", rms(got), rms(want), worst_at, got.iter().filter(|x| !x.is_finite()).count()));
    }
    let n = if got.len() == want.len() { String::new() } else { format!(", LENGTHS {} vs {}", got.len(), want.len()) };
    log(format!("{name:<24} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values{n})", want.len().min(got.len())));
    if got.len() == want.len() { rel } else { 1.0 }
}

pub fn run(m: &MossImage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, f32s(v))) };
    let mut worst = 0f64;
    let sig = get("sigmas")?.1;
    let (lshape, lat0) = get("lat_0")?;
    let hw = (lshape[1], lshape[2]);
    let n = hw.0 * hw.1;
    let gpu = m.nsd.gpu.clone();
    if stages.contains(&"sigmas") {
        let ours = sigmas(sig.len() - 1);
        log(format!("sigmas: ours {} levels, the reference's {}", ours.len(), sig.len()));
        let (rel, mx, _) = compare(&ours, &sig);
        log(format!("sigmas                   rel {rel:.2e}  max {mx:.3e}"));
        if ours.len() != sig.len() {
            worst = worst.max(1.0);
        } else {
            worst = worst.max(rel);
        }
    }
    let ids: Vec<u32> = read_npy(&dir.join("ids.npy"))?.1.iter().map(|x| *x as u32).collect();
    if stages.contains(&"tokens") {
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
        let ours = m.te.tokenize_templated(&prompt);
        let same = ours == ids;
        log(format!("tokens                   {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), ids.len(),
                    ours.iter().zip(&ids).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
        if !same {
            worst = worst.max(1.0);
        }
    }
    if stages.contains(&"te") {
        let mut tr = te::Trace::default();
        let ours = m.te.encode_ids(&ids, Some(&mut tr))?;
        for (i, x) in tr.layers.iter().enumerate() {
            let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
            log(format!("te layer {i:<2}  x rms {rms:.3}  max {:.1}", x.iter().fold(0f32, |a, v| a.max(v.abs()))));
        }
        worst = worst.max(note("te final hidden", &ours, &get("te_final")?.1, log));
        let nids: Vec<u32> = read_npy(&dir.join("ids_neg.npy"))?.1.iter().map(|x| *x as u32).collect();
        worst = worst.max(note("te final hidden (empty)", &m.te.encode_ids(&nids, None)?, &get("te_neg")?.1, log));
    }
    let te = get("te_final")?.1;
    let neg = get("te_neg")?.1;
    if stages.contains(&"dit") {
        let mut g = dit::Grabs::default();
        let pre = m.dit.prefix(&m.nsd, &te, Some(&mut g))?;
        worst = worst.max(note("text embedder", &g.txt, &get("txt")?.1, log));
        let npre = m.dit.prefix(&m.nsd, &neg, None)?;
        let mods = m.dit.mods(&m.nsd, &sig)?;
        let steps = sig.len() - 1;
        for i in [0usize, steps / 2, steps - 1] {
            let (_, li) = get(&format!("lat_{i}"))?;
            let lat = DevBuf::from_f32(&gpu, &dit::tokens(&li, n))?;
            let mut gi = dit::Grabs::default();
            let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[i], &[], if i == 0 { Some(&mut gi) } else { None })?.to_f32()?;
            if i == 0 {
                worst = worst.max(note("video embedder", &gi.emb, &get("emb")?.1, log));
                for (name, ours) in &gi.probe {
                    let (rf, rows) = match *name {
                        "mod" => ("p_mod", 0), "head" => ("p_head", 0), "rope" => ("p_rope", 0),
                        "x0" => ("p_x0", 0), "y1" => ("p_y1", 0), "v" => ("p_v", 0), "att" => ("p_att", 0), "x1" => ("p_x1", 0), "act" => ("p_act", 0), "x2" => ("p_x2", 0),
                        _ => continue,
                    };
                    let _ = rows;
                    let want = get(rf)?.1;
                    // the reference's rope is [sin | cos] x 64; ours (cos, sin) pairs: compared as ours' cos plane
                    if *name == "rope" {
                        let cos_ours: Vec<f32> = ours.chunks_exact(dit::HEAD).flat_map(|r| r.chunks_exact(2).map(|p| p[0]).collect::<Vec<_>>()).collect();
                        let cos_ref: Vec<f32> = want.chunks_exact(dit::HEAD).flat_map(|r| r[dit::HEAD / 2..].to_vec()).collect();
                        worst = worst.max(note("probe rope (cos)", &cos_ours, &cos_ref, log));
                        continue;
                    }
                    worst = worst.max(note(&format!("probe {name}"), ours, &want, log));
                }
                for (b, x) in &gi.blocks {
                    let want = get(&format!("blk_{b}"))?.1;
                    let (img, txt) = (n * dit::DIM, x.len());
                    let r = note(&format!("block {b} (image rows)"), &x[..img], &want[..img], log);
                    note(&format!("block {b} (text rows)"), &x[img..txt], &want[img..], log);
                    worst = worst.max(r);
                }
            }
            worst = worst.max(note(&format!("velocity (cond), step {i}"), &dit::planes(&v, n), &get(&format!("vel_c_{i}"))?.1, log));
            let vu = m.dit.velocity(&m.nsd, &npre, &lat, hw, &mods[i], UNCOND_SKIP, None)?.to_f32()?;
            worst = worst.max(note(&format!("velocity (uncond), step {i}"), &dit::planes(&vu, n), &get(&format!("vel_u_{i}"))?.1, log));
        }
    }
    if stages.contains(&"steps") {
        let pre = m.dit.prefix(&m.nsd, &te, None)?;
        let npre = m.dit.prefix(&m.nsd, &neg, None)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, Some(&npre), 3.0, &dit::tokens(&lat0, n), hw, &sig, &mut |_| Ok(()))?;
        log(format!("{} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents after the steps", &dit::planes(&lat, n), &get("lat_final")?.1, log));
    }
    if stages.contains(&"vae") {
        let (_, lf) = get("lat_final")?;
        let (img, pw, ph) = m.vae.decode_half(&m.nsd, &dit::tokens(&lf, n), hw.0, hw.1)?;
        let f = DevBuf::f32(&gpu, pw * ph * 3)?;
        let k = crate::ffi::api()?;
        // SAFETY: ph x pw x 3 halves in, as many floats out.
        crate::ffi::check(unsafe { (k.to_float)(gpu.raw(), img.ptr(), f.fp(), (pw * ph * 3) as i64) }, "to float")?;
        let v = f.to_f32()?;
        // ours [H, W, 3] (unclamped) -> [3, H, W], clamped as the reference's
        let planes: Vec<f32> = (0..3).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| v[p * 3 + c].clamp(-1.0, 1.0)).collect();
        worst = worst.max(note("vae (rgb)", &planes, &get("rgb")?.1, log));
    }
    if stages.contains(&"full") {
        // all ours - the encoder's rows too - from the reference's starting noise: a picture to put beside ref.png
        let text = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
        let pre = m.condition(text.replace("<|im_start|>user\n", "").split("<|im_end|>").next().unwrap_or(""))?;
        let npre = m.condition("")?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, Some(&npre), 3.0, &dit::tokens(&lat0, n), hw, &sig, &mut |_| Ok(()))?;
        log(format!("full: {} steps in {:.1} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents (all ours)", &dit::planes(&lat, n), &get("lat_final")?.1, log));
        let (rgb, pw, ph) = m.vae.decode(&m.nsd, &lat, hw.0, hw.1)?;
        let pic = nextsycl_image::Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb };
        pic.write_png(&dir.join("ours_full.png"), &[]).map_err(Error)?;
        let want = get("rgb")?.1;
        let ours: Vec<f32> = (0..3).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| pic.data[p * 3 + c] as f32 / 127.5 - 1.0).collect();
        note("picture (all ours)", &ours, &want, log);
    }
    Ok(worst)
}
