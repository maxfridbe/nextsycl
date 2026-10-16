//! Stage by stage against Tencent's own code (reference/hunyuan3d/ref_mini.py, ref_21.py: float32 on the CPU, the
//! same files). Each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   pre    the picture                -> pixels.npy      the cut-out, recentred, resized, normalized picture
//!   cond   pixels.npy                 -> cond.npy        DINOv2's last hidden state
//!   dit    x_<i> + cond.npy at sigma  -> v_<i>.npy       the (guided) velocity of every step; 2.1 also each pass
//!                                     -> d0_*.npy, b0.npy   the stream after the first block (step 0)
//!   steps  noise.npy + cond.npy       -> latents.npy     the whole sampler
//!   vae    latents.npy                -> vae_tokens.npy  the decoder's transformer
//!          + queries.npy              -> logits.npy      the geometry decoder at fixed points
//!   grid   latents.npy                -> grid.npy        the dense grid at the dump's resolution, then its mesh
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result, Texture};

use crate::ops::read_f32;
use crate::{Dit, Hunyuan3d, ARCH_MINI};

/// A little-endian float32 .npy: (shape, values)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if !head.contains("'<f4'") || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let v = b[at + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((shape, v))
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

fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> Result<f64> {
    if got.len() != want.len() {
        return Err(Error(format!("{what}: {} values, the reference {}", got.len(), want.len())));
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<16} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
    Ok(rel)
}

/// Every stage in `stages` against the dumps in `dir` (`picture`: the reference's input, for "pre"); the worst
/// relative error
pub fn run(e: &Hunyuan3d, dir: &Path, stages: &[String], picture: Option<&Texture>, log: &mut dyn FnMut(String)) -> Result<f64> {
    let npy = |name: &str| read_npy(&dir.join(name)).map(|x| x.1);
    let meta: serde_json::Value = std::fs::read(dir.join("meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let has = |s: &str| stages.iter().any(|x| x == s || x == "all");
    let gpu = e.ops.gpu.clone();
    let mut worst = 0f64;
    let mut keep = |r: f64| worst = worst.max(r);
    if has("pre") {
        match picture {
            Some(t) => {
                let (px, note) = crate::picture::prepare(t, true, e.arch.proc_size, e.arch.enc_size, crate::BORDER)?;
                log(format!("pre: {note}"));
                keep(line("pre pixels", &px, &npy("pixels.npy")?, log)?);
            }
            None => log("pre: no picture given (--picture): skipped".into()),
        }
    }
    let cond_ref = npy("cond.npy")?;
    if has("cond") {
        let t = std::time::Instant::now();
        let c = e.condition(&npy("pixels.npy")?)?;
        let got = read_f32(&c, cond_ref.len())?;
        log(format!("cond: DINOv2 in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("cond", &got, &cond_ref, log)?);
    }
    let cond = DevBuf::from_f32(&gpu, &cond_ref)?;
    let noise = npy("noise.npy")?;
    let tokens = noise.len() / e.vae.channels;
    let sigmas = npy("sigmas.npy")?;
    let cfg = meta["guidance"].as_f64().unwrap_or(e.arch.cfg as f64) as f32;
    if has("dit") || has("steps") {
        let mut st = e.run_state(&cond, tokens)?;
        if has("dit") {
            let v = DevBuf::new(&gpu, noise.len() * 4)?;
            let n = noise.len();
            for i in 0..sigmas.len() - 1 {
                let Ok(xi) = npy(&format!("x_{i}.npy")) else { break };
                let x = DevBuf::from_f32(&gpu, &xi)?;
                if i == 0 {
                    // the stream after the first block
                    match (&e.dit, &mut st.dit) {
                        (Dit::Flux(f), crate::RunDit::Flux(sc)) => {
                            f.forward(&e.ops, sc, &x, sigmas[0], cfg, &v, Some(0))?;
                            let (txt, img) = f.stream(sc)?;
                            if let (Ok(wt), Ok(wi)) = (npy("d0_txt.npy"), npy("d0_img.npy")) {
                                keep(line("dit block 0 txt", &txt, &wt, log)?);
                                keep(line("dit block 0 img", &img, &wi, log)?);
                            }
                        }
                        (Dit::Plain(p), crate::RunDit::Plain(sc)) => {
                            p.forward(&e.ops, sc, &x, sigmas[0], 0, &v, Some(0))?;
                            if let Ok(w) = npy("b0.npy") {
                                keep(line("dit block 0", &p.stream(sc)?, &w, log)?);
                            }
                            for (which, name) in [(0, "vc_0.npy"), (1, "vu_0.npy")] {
                                if let Ok(w) = npy(name) {
                                    p.forward(&e.ops, sc, &x, sigmas[0], which, &v, None)?;
                                    keep(line(if which == 0 { "dit cond pass" } else { "dit uncond pass" }, &read_f32(&v, n)?, &w, log)?);
                                }
                            }
                        }
                        _ => return Err(Error("the DiT and its scratch disagree".into())),
                    }
                }
                let t = std::time::Instant::now();
                e.velocity(&mut st, &x, sigmas[i], cfg, &v)?;
                let got = read_f32(&v, n)?;
                let secs = t.elapsed().as_secs_f64();
                keep(line(&format!("dit v_{i} ({secs:.2} s)"), &got, &npy(&format!("v_{i}.npy"))?, log)?);
            }
        }
        if has("steps") {
            let t = std::time::Instant::now();
            let got = e.sample(&mut st, &noise, &sigmas, cfg, &mut |_, _| Ok(()))?;
            log(format!("steps: {} in {:.2} s", sigmas.len() - 1, t.elapsed().as_secs_f64()));
            keep(line("latents", &got, &npy("latents.npy")?, log)?);
        }
    }
    let latents = npy("latents.npy")?;
    if has("vae") {
        let t = std::time::Instant::now();
        let toks = e.vae.tokens(&e.ops, &latents)?;
        let got = read_f32(&toks, tokens * e.vae.width)?;
        log(format!("vae: transformer in {:.2} s", t.elapsed().as_secs_f64()));
        keep(line("vae tokens", &got, &npy("vae_tokens.npy")?, log)?);
        let data = e.vae.data(&e.ops, &toks, tokens)?;
        let sc = e.vae.query_scratch(&e.ops, 8192)?;
        let got = e.vae.query_points(&e.ops, &data, &sc, &npy("queries.npy")?)?;
        let want = npy("logits.npy")?;
        keep(line("vae logits", &got, &want, log)?);
        let agree = got.iter().zip(&want).filter(|(a, b)| (**a > 0.0) == (**b > 0.0)).count();
        log(format!("vae inside / outside agree at {agree} of {} points", want.len()));
    }
    if has("grid") {
        let res = meta["res"].as_u64().unwrap_or(128) as usize;
        let t = std::time::Instant::now();
        let got = e.decode_dense(&latents, res, e.vae.chunk(), &mut |_, _| Ok(()))?;
        log(format!("grid {res}: {:.2} s", t.elapsed().as_secs_f64()));
        let want = npy("grid.npy")?;
        keep(line("grid logits", &got, &want, log)?);
        let flips = got.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        log(format!("grid: {flips} of {} points on the other side of the surface", want.len()));
        // coarse to fine: the same surface from a share of the points
        let t = std::time::Instant::now();
        let (fast, queried) = e.decode_fast(&latents, res, e.vae.chunk(), &mut |_, _| Ok(()))?;
        let flips = fast.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        log(format!("grid {res} coarse to fine: {:.2} s, {queried} points queried ({:.0} %), {flips} on the other side of the surface", t.elapsed().as_secs_f64(),
                    100.0 * queried as f64 / want.len() as f64));
        let t = std::time::Instant::now();
        let m = Hunyuan3d::mesh(&got, res);
        log(format!("mesh: {} vertices, {} triangles in {:.2} s (the reference's skimage: {} vertices, {} faces)", m.positions.len(), m.triangles(),
                    t.elapsed().as_secs_f64(), meta["vertices"], meta["faces"]));
    }
    if e.arch.name == ARCH_MINI && meta["tokens"].as_u64().is_some_and(|t| t as usize != e.arch.tokens) {
        log(format!("note: the reference sampled {} tokens, the engine's default is {}", meta["tokens"], e.arch.tokens));
    }
    Ok(worst)
}
