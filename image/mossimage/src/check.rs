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
    if stages.contains(&"repeat") {
        // the same forward twice on the reference's latents: any difference is a race, not rounding
        let pre = m.dit.prefix(&m.nsd, &te, None)?;
        let npre = m.dit.prefix(&m.nsd, &neg, None)?;
        let mods = m.dit.mods(&m.nsd, &sig)?;
        // the prompt side too: three prefixes of one prompt, and a checksum to compare runs by
        let sum = |p: &dit::Prefix| -> Result<(f64, Vec<f32>)> {
            let v = p.x.to_f32()?;
            Ok((v.iter().map(|x| *x as f64).sum(), v))
        };
        let (s0, v0) = sum(&pre)?;
        for r in 0..2 {
            let (s1, v1) = sum(&m.dit.prefix(&m.nsd, &te, None)?)?;
            let d = v0.iter().zip(&v1).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            log(format!("repeat prefix {r}: sum {s0:.6} vs {s1:.6}, differs by {d:.3e}"));
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..sig.len() - 1 {
            let Ok((_, li)) = get(&format!("lat_{i}")) else { continue };
            let lat = DevBuf::from_f32(&gpu, &dit::tokens(&li, n))?;
            let a = m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[i], &[], None)?.to_f32()?;
            let b = m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[i], &[], None)?.to_f32()?;
            let ua = m.dit.velocity(&m.nsd, &npre, &lat, hw, &mods[i], UNCOND_SKIP, None)?.to_f32()?;
            let ub = m.dit.velocity(&m.nsd, &npre, &lat, hw, &mods[i], UNCOND_SKIP, None)?.to_f32()?;
            let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            log(format!("repeat step {i}: cond differs by {:.3e}, uncond by {:.3e}", d(&a, &b), d(&ua, &ub)));
        }
    }
    if stages.contains(&"repeat") {
        // the whole sampler twice from the same noise: the steps' latents are compared as they go
        let pre = m.dit.prefix(&m.nsd, &te, None)?;
        let npre = m.dit.prefix(&m.nsd, &neg, None)?;
        let start = dit::tokens(&lat0, n);
        // the latents after 4 steps, then step 4's forwards repeated on them
        let l4 = m.sample(&pre, Some(&npre), 3.0, &start, hw, &sig[..=4], &mut |_| Ok(()))?;
        let lat = DevBuf::from_f32(&gpu, &l4)?;
        let mods = m.dit.mods(&m.nsd, &sig)?;
        let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        let c: Vec<Vec<f32>> = (0..3).map(|_| m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[4], &[], None).and_then(|v| v.to_f32())).collect::<Result<_>>()?;
        let u: Vec<Vec<f32>> = (0..3).map(|_| m.dit.velocity(&m.nsd, &npre, &lat, hw, &mods[4], UNCOND_SKIP, None).and_then(|v| v.to_f32())).collect::<Result<_>>()?;
        log(format!("repeat step 4 on its latents: cond {:.3e} {:.3e}, uncond {:.3e} {:.3e}", d(&c[0], &c[1]), d(&c[1], &c[2]), d(&u[0], &u[1]), d(&u[1], &u[2])));
        // the same forward with different memory around it: buffers of other sizes held while it runs
        let mut keep: Vec<DevBuf> = Vec::new();
        let mut outs: Vec<Vec<f32>> = Vec::new();
        for j in 0..5usize {
            keep.push(DevBuf::new(&gpu, (j * 37 + 11) * 1_000_003 + 4096 * j)?);
            outs.push(m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[4], &[], None)?.to_f32()?);
        }
        log(format!("repeat step 4 with other buffers held: {:?}", outs.iter().map(|o| format!("{:.2e}", d(o, &outs[0]))).collect::<Vec<_>>()));
        let m2 = m.dit.mods(&m.nsd, &sig)?;
        let m3 = m.dit.mods(&m.nsd, &sig)?;
        let md = |a: &[dit::StepMod], b: &[dit::StepMod]| -> Result<f32> {
            let mut w = 0f32;
            for (x, y) in a.iter().zip(b) {
                w = w.max(d(&x.blocks.to_f32()?, &y.blocks.to_f32()?)).max(d(&x.head.to_f32()?, &y.head.to_f32()?));
            }
            Ok(w)
        };
        log(format!("repeat mods: two computations differ by {:.3e}, against the first {:.3e}", md(&m2, &m3)?, md(&mods, &m2)?));
        // sample()'s loop by hand, cond only, with the latents either updated in place on the card or through the host
        for through_host in [false, true] {
            let mut runs: Vec<Vec<f32>> = Vec::new();
            for _ in 0..3 {
                let ka = crate::ffi::api()?;
                let mut lat = DevBuf::from_f32(&gpu, &start)?;
                for i in 0..6 {
                    let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, &mods[i], &[], None)?;
                    // SAFETY: n floats each.
                    crate::ffi::check(unsafe { (ka.axpy)(gpu.raw(), lat.fp(), v.fp(), start.len() as i64, sig[i + 1] - sig[i]) }, "euler step")?;
                    gpu.sync()?;
                    if through_host {
                        lat = DevBuf::from_f32(&gpu, &lat.to_f32()?)?;
                    }
                }
                runs.push(lat.to_f32()?);
            }
            log(format!("repeat by hand, 6 steps, cond only, through host {through_host}: runs differ by {:.3e} {:.3e}", d(&runs[0], &runs[1]), d(&runs[1], &runs[2])));
        }
        for (what, cfg_on) in [("cond only", false), ("cond + uncond", true)] {
            let mut runs: Vec<Vec<f32>> = Vec::new();
            for _ in 0..3 {
                runs.push(m.sample(&pre, if cfg_on { Some(&npre) } else { None }, 3.0, &start, hw, &sig[..=6], &mut |_| Ok(()))?);
            }
            log(format!("repeat sampler, 6 steps, {what}: runs differ by {:.3e} {:.3e}", d(&runs[0], &runs[1]), d(&runs[1], &runs[2])));
        }
    }
    if stages.contains(&"bench") {
        // the size the picture is made at (NS_MOSS_BENCH_SIZE, default 1024): each stage warm, then timed
        let size: usize = std::env::var("NS_MOSS_BENCH_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        let bh = (size / 16, size / 16);
        let bn = bh.0 * bh.1;
        // NS_MOSS_BENCH_REPEAT: the prompt said that many times over (a long prompt through the text encoder)
        let reps: usize = std::env::var("NS_MOSS_BENCH_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let prompt = "A small red fox in fresh snow, soft morning light, a wooden sign beside it reading \"WELCOME\". ".repeat(reps);
        let prompt = prompt.trim_end();
        fn time(log: &mut dyn FnMut(String), what: &str, f: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
            f()?;
            let t0 = std::time::Instant::now();
            f()?;
            let s = t0.elapsed().as_secs_f64();
            log(format!("bench {what:<28} {s:.3} s"));
            Ok(s)
        }
        let ids = m.te.ids(prompt);
        log(format!("bench prompt of {} tokens, picture {size} x {size} ({bn} tokens)", ids.len()));
        time(log, "text encoder", &mut || m.te.encode_ids(&ids, None).map(|_| ()))?;
        let rows = m.te.encode_ids(&ids, None)?;
        time(log, "text embedder (prefix)", &mut || m.dit.prefix(&m.nsd, &rows, None).map(|_| ()))?;
        let pre = m.dit.prefix(&m.nsd, &rows, None)?;
        let mods = m.dit.mods(&m.nsd, &sig)?;
        let lat = DevBuf::from_f32(&gpu, &crate::noise(7, bn * dit::LATENT))?;
        time(log, "DiT forward", &mut || m.dit.velocity(&m.nsd, &pre, &lat, bh, &mods[0], &[], None).map(|_| ()))?;
        let t0 = std::time::Instant::now();
        m.dit.velocity(&m.nsd, &pre, &lat, bh, &mods[0], &[], None)?;
        log(format!("bench DiT forward: queued by the host in {:.3} s, done after {:.3} s", m.dit.queued(), t0.elapsed().as_secs_f64()));
        let mut g = dit::Grabs { prof: Some(dit::Prof::new()), ..Default::default() };
        m.dit.velocity(&m.nsd, &pre, &lat, bh, &mods[0], &[], Some(&mut g))?;
        if let Some(p) = g.prof {
            let total: f64 = p.acc.iter().map(|(_, t)| t).sum();
            for (name, t) in &p.acc {
                log(format!("bench   DiT op {name:<16} {t:.3} s  {:4.1} %", 100.0 * t / total));
            }
        }
        let flops = 2.0 * (bn + 32) as f64 * 3072.0 * 16384.0;
        for (what, t) in m.dit.gemm_probe(&m.nsd, bn + 32)? {
            let fl = if what.starts_with("down") { 2.0 * (bn + 32) as f64 * 8192.0 * 3072.0 } else { flops };
            log(format!("bench product, {what:<20} {:.2} ms  ({:.0} TFLOPS)", t * 1e3, fl / t / 1e12));
        }
        for (what, t, bytes) in m.dit.silo_probe(&m.nsd, bn + 32)? {
            log(format!("bench silo {what:<26} {:.3} ms  ({:.0} GB/s)", t * 1e3, bytes / t / 1e9));
        }
        let toks = crate::noise(1, bn * dit::LATENT);
        time(log, "VAE decode", &mut || m.vae.decode(&m.nsd, &toks, bh.0, bh.1).map(|_| ()))?;
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
