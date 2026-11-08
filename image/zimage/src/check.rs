//! The engine against reference/z-image-turbo/ref.py's dumps (diffusers' classes on the CPU in float32, on the same
//! files): each stage fed the reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  prompt.txt through the template      -> ids.npy (exactly)
//!   te      ids.npy                              -> te_h1, te_h18, te_h35 (the caption) [L, 2560]
//!   dit     te_h35 (the reference's)             -> cap_emb, cr_out; at every step lat_i -> vel_i, at step 0 also
//!                                                   t_emb, x_emb, nr_out, l0_out, l29_out
//!   steps   lat_0 through every step             -> lat_final
//!   vae     lat_final                            -> rgb [3, H, W]; our picture of our latents written as ours.png
//!   full    prompt.txt, all ours from lat_0      -> lat_final, ours_full.png
//! ```

use std::path::Path;

use nextsycl_image::{Error, Picture, Result};

use crate::{dit, te, ZImage};

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
    let n = want.len().min(got.len());
    log(format!("{name:<26} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({n} values{})", if got.len() != want.len() { format!(", {} vs {}", got.len(), want.len()) } else { String::new() }));
    if got.len() != want.len() {
        return 1.0;
    }
    rel
}

pub fn run(m: &ZImage, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, f32s(v))) };
    let mut worst = 0f64;
    let ids: Vec<u32> = read_npy(&dir.join("ids.npy"))?.1.iter().map(|x| *x as u32).collect();
    let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map_err(|e| Error(format!("prompt.txt: {e}")))?;
    if stages.contains(&"tokens") {
        let ours = m.te.ids(&prompt);
        let same = ours == ids;
        log(format!("tokens                     {} ({} vs {} ids{})", if same { "identical" } else { "DIFFER" }, ours.len(), ids.len(),
                    ours.iter().zip(&ids).position(|(a, b)| a != b).map_or(String::new(), |p| format!(", first at {p}"))));
        if !same {
            worst = worst.max(1.0);
        }
    }
    if stages.contains(&"te") {
        let mut taps = vec![(1usize, Vec::new()), (18, Vec::new())];
        let t0 = std::time::Instant::now();
        let fin = m.te.encode_ids(&ids, te::LAYERS, Some(&mut taps))?;
        log(format!("te: {} tokens in {:.3} s", ids.len(), t0.elapsed().as_secs_f64()));
        for (at, v) in &taps {
            worst = worst.max(note(&format!("te after layer {at}"), v, &get(&format!("te_h{at}"))?.1, log));
        }
        worst = worst.max(note("te after layer 35 (cap)", &fin, &get("te_h35")?.1, log));
    }
    let cap = get("te_h35")?.1;
    let sig = get("sigmas")?.1;
    let (lshape, lat0) = get("lat_0")?;
    let (lh, lw) = (lshape[1], lshape[2]);
    let hw = (lh / 2, lw / 2);
    let n = hw.0 * hw.1;
    let np = dit::padded(n);
    let ours_sig = crate::sigmas(sig.len() - 1, crate::SHIFT);
    note("sigmas", &ours_sig, &sig, log);
    // the starting noise: ours from the seed against torch.randn's (seed.txt, when the dump has it)
    if let Some(seed) = std::fs::read_to_string(dir.join("seed.txt")).ok().and_then(|s| s.trim().parse::<u64>().ok()) {
        worst = worst.max(note(&format!("noise from seed {seed}"), &crate::noise(seed, lat0.len()), &lat0, log));
    }
    if stages.contains(&"dit") {
        let mut g = dit::Grabs::default();
        let pre = m.dit.prefix(&m.nsd, &cap, Some(&mut g))?;
        let c = pre.rows;
        worst = worst.max(note("caption embedded (+pad)", &g.cap_emb, &get("cap_emb")?.1, log));
        worst = worst.max(note("context refiner", &g.cr_out, &get("cr_out")?.1, log));
        worst = worst.max(note("t_emb (step 0)", &m.dit.t_emb(sig[0]), &get("t_emb")?.1, log));
        let mods = m.dit.mods(&m.nsd, &sig)?;
        for (i, md) in mods.iter().enumerate() {
            let (_, li) = get(&format!("lat_{i}"))?;
            let lat = nextsycl_core::DevBuf::from_f32(&m.nsd.gpu, &dit::patchify(&li, lh, lw))?;
            let mut gi = dit::Grabs::default();
            let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, md, if i == 0 { Some(&mut gi) } else { None })?.to_f32()?;
            if i == 0 {
                worst = worst.max(note("image embedded (+pad)", &gi.x_emb, &get("x_emb")?.1, log));
                worst = worst.max(note("noise refiner", &gi.nr_out, &get("nr_out")?.1, log));
                for (name, ours, want) in [("layer 0", &gi.l0_out, get("l0_out")?.1), ("layer 29", &gi.l29_out, get("l29_out")?.1)] {
                    worst = worst.max(note(&format!("{name} (image rows)"), &ours[..np * dit::DIM], &want[..np * dit::DIM], log));
                    worst = worst.max(note(&format!("{name} (caption rows)"), &ours[np * dit::DIM..(np + c) * dit::DIM], &want[np * dit::DIM..], log));
                }
            }
            let want = dit::patchify(&get(&format!("vel_{i}"))?.1, lh, lw);
            worst = worst.max(note(&format!("velocity, step {i}"), &v, &want, log));
        }
    }
    if stages.contains(&"bench") {
        // each stage warm, then timed, at the dump's size; one forward profiled part by part; the products alone
        fn time(log: &mut dyn FnMut(String), what: &str, f: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
            f()?;
            let t0 = std::time::Instant::now();
            f()?;
            let s = t0.elapsed().as_secs_f64();
            log(format!("bench {what:<26} {s:.3} s"));
            Ok(s)
        }
        log(format!("bench prompt of {} tokens, latents {lh} x {lw} ({n} image tokens)", ids.len()));
        time(log, "text encoder", &mut || m.te.encode_ids(&ids, te::LAYERS, None).map(|_| ()))?;
        time(log, "caption (context refiner)", &mut || m.dit.prefix(&m.nsd, &cap, None).map(|_| ()))?;
        let pre = m.dit.prefix(&m.nsd, &cap, None)?;
        let mods = m.dit.mods(&m.nsd, &sig)?;
        let lat = nextsycl_core::DevBuf::from_f32(&m.nsd.gpu, &dit::patchify(&lat0, lh, lw))?;
        time(log, "DiT forward", &mut || m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[0], None).map(|_| ()))?;
        let mut g = dit::Grabs { profile: true, ..Default::default() };
        m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[0], Some(&mut g))?;
        let total: f64 = g.prof.iter().map(|(_, t)| t).sum();
        for (name, t) in &g.prof {
            log(format!("bench   DiT part {name:<18} {t:.3} s  {:4.1} %", 100.0 * t / total));
        }
        for (what, t, fl) in m.dit.gemm_probe(&m.nsd, np + pre.rows)? {
            log(format!("bench product {what:<12} {:.2} ms  ({:.0} TFLOPS)", t * 1e3, fl / t / 1e12));
        }
        time(log, "VAE decode", &mut || m.vae.decode(&m.nsd, &lat0, lh, lw).map(|_| ()))?;
    }
    if stages.contains(&"sizes") {
        // the whole request (text encoder, 8 steps, VAE) at the model's own sizes of the 1024 budget, the larger squares
        // and the largest sides: each made twice, the second timed
        use nextsycl_image::ImageEngine;
        let mut sizes: Vec<(u32, u32)> = crate::SIZES[..11].to_vec();
        sizes.extend([(1280, 1280), (1536, 1024), (1536, 1536), (2048, 1152), (2048, 2048)]);
        for (w, h) in sizes {
            let req = nextsycl_image::ImageRequest { prompt: prompt.clone(), width: Some(w), height: Some(h), seed: 42, n: 1, ..Default::default() };
            m.generate(&req, &mut |_| Ok(()))?;
            let t0 = std::time::Instant::now();
            m.generate(&req, &mut |_| Ok(()))?;
            log(format!("size {w}x{h}: {:.2} s a picture", t0.elapsed().as_secs_f64()));
        }
    }
    let mut ours_lat = None;
    if stages.contains(&"steps") {
        let pre = m.dit.prefix(&m.nsd, &cap, None)?;
        let t0 = std::time::Instant::now();
        let lat = m.sample(&pre, &lat0, lh, lw, &sig, &mut |_| Ok(()))?;
        log(format!("{} steps in {:.2} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents after the steps", &lat, &get("lat_final")?.1, log));
        ours_lat = Some(lat);
    }
    if stages.contains(&"vae") {
        let (_, lf) = get("lat_final")?;
        let t0 = std::time::Instant::now();
        let k = crate::ffi::api()?;
        let gpu = m.nsd.gpu.clone();
        // each stage's largest magnitude and values past half's range (infinities, NaNs)
        let mut trace = |name: &str, x: &nextsycl_core::DevBuf, n: usize| -> Result<()> {
            let f = nextsycl_core::DevBuf::f32(&gpu, n)?;
            // SAFETY: n halves in, n floats out.
            crate::ffi::check(unsafe { (k.to_float)(gpu.raw(), x.ptr(), f.fp(), n as i64) }, "to float")?;
            let v = f.to_f32()?;
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            let mx = v.iter().filter(|x| x.is_finite()).fold(0f32, |a, x| a.max(x.abs()));
            log(format!("vae {name:<20} max |x| {mx:.1}{}", if bad > 0 { format!(", {bad} not finite") } else { String::new() }));
            Ok(())
        };
        let (img, pw, ph) = m.vae.decode_traced(&m.nsd, &lf, lh, lw, &mut trace)?;
        let f = nextsycl_core::DevBuf::f32(&m.nsd.gpu, pw * ph * 3)?;
        // SAFETY: ph x pw x 3 halves in, as many floats out.
        crate::ffi::check(unsafe { (k.to_float)(m.nsd.gpu.raw(), img.ptr(), f.fp(), (pw * ph * 3) as i64) }, "to float")?;
        let v = f.to_f32()?;
        log(format!("vae: {pw}x{ph} in {:.2} s", t0.elapsed().as_secs_f64()));
        // ours [H, W, 3] -> [3, H, W]
        let planes: Vec<f32> = (0..3).flat_map(|c| (0..pw * ph).map(move |p| (c, p))).map(|(c, p)| v[p * 3 + c]).collect();
        worst = worst.max(note("vae (rgb)", &planes, &get("rgb")?.1, log));
        let lat = ours_lat.unwrap_or(lf);
        let (rgb, pw, ph) = m.vae.decode(&m.nsd, &lat, lh, lw)?;
        Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb }.write_png(&dir.join("ours.png"), &[]).map_err(Error)?;
        log(format!("ours.png: {pw}x{ph}"));
    }
    if stages.contains(&"full") {
        // the whole pipeline on our side - our text encoder's caption too - from the reference's starting noise: the
        // picture to put beside ref.png
        let t0 = std::time::Instant::now();
        let pre = m.dit.prefix(&m.nsd, &m.te.encode(&prompt)?, None)?;
        let lat = m.sample(&pre, &lat0, lh, lw, &sig, &mut |_| Ok(()))?;
        log(format!("full: {} steps in {:.2} s", sig.len() - 1, t0.elapsed().as_secs_f64()));
        worst = worst.max(note("latents (all ours)", &lat, &get("lat_final")?.1, log));
        let (rgb, pw, ph) = m.vae.decode(&m.nsd, &lat, lh, lw)?;
        let want = get("rgb")?.1;
        let ours: Vec<f32> = (0..3).flat_map(|ch| (0..pw * ph).map(move |p| (ch, p))).map(|(ch, p)| rgb[p * 3 + ch] as f32 / 127.5 - 1.0).collect();
        let clamped: Vec<f32> = want.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        note("picture (all ours)", &ours, &clamped, log);
        Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb }.write_png(&dir.join("ours_full.png"), &[]).map_err(Error)?;
    }
    Ok(worst)
}
