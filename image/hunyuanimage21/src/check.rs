//! Stage by stage against the reference (reference/hunyuanimage21/ref.py: the official modules on the same GGUF files,
//! float32 on the CPU): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   te     tokens.json                -> te_layer0.npy, embeds.npy (neg_*: the negative prompt's)
//!   byt5   byt5.json                  -> byt5.npy
//!   dit    embeds + byt5 + noise      -> vec0, txt0 (the token refiner), dbl0_img / dbl0_txt (double block 0),
//!                                        dbl_img (after the double blocks), v0 (the first velocity)
//!   steps                             -> latents_N.npy every step (Euler over sigmas.npy; the base model with CFG)
//!   vae    latents_last               -> image.npy (RGB in [-1, 1])
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Result};

use crate::{byt5, dit, sched, HunyuanImage21};

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

fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    if got.len() != want.len() {
        log(format!("{what:<14} {} values here, {} in the reference", got.len(), want.len()));
        return f64::INFINITY;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<14} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
    rel
}

fn json(path: &Path) -> Result<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?)
        .map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
}

/// Every stage in `stages` ("te", "byt5", "dit", "steps", "vae") against the dumps in `dir`; the worst relative error
pub fn run(m: &HunyuanImage21, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let gpu = &m.nsd.gpu;
    let mut worst = 0f64;
    if stages.contains(&"te") {
        let toks = json(&dir.join("tokens.json"))?;
        for (p, key) in [("", "ids"), ("neg_", "neg_ids")] {
            let want = ids(&toks[key]);
            if want.is_empty() {
                continue;
            }
            let prompt = toks[if p.is_empty() { "prompt" } else { "negative" }].as_str().unwrap_or("");
            let mine = m.te.ids(prompt);
            log(format!("{p}tokens: {} here, {} in the reference{}", mine.len(), want.len(), if mine == want { " - the same" } else { " - DIFFERENT" }));
            if mine != want {
                worst = f64::INFINITY;
            }
            let t0 = std::time::Instant::now();
            let l0 = m.te.run(&m.nsd, &want, 1)?.to_f32()?;
            let (_, r0) = read_npy(&dir.join(format!("{p}te_layer0.npy")))?;
            worst = worst.max(line(&format!("{p}te layer 0"), &l0, &r0, log));
            let all = m.te.run(&m.nsd, &want, crate::te::LAYERS_USED)?.to_f32()?;
            log(format!("te: {} tokens in {:.0} ms", want.len(), t0.elapsed().as_secs_f64() * 1e3));
            let (_, emb) = read_npy(&dir.join(format!("{p}embeds.npy")))?;
            worst = worst.max(line(&format!("{p}te embeds"), &all[crate::te::CROP * dit::DIM..], &emb, log));
        }
    }
    if stages.contains(&"byt5") {
        let b = json(&dir.join("byt5.json"))?;
        let text = b["text"].as_str().map(str::to_string);
        let prompt = json(&dir.join("tokens.json")).ok().and_then(|t| t["prompt"].as_str().map(str::to_string));
        if let Some(p) = prompt {
            let mine = byt5::quoted(&p);
            log(format!("glyph text: {:?} here, {:?} in the reference{}", mine, text, if mine == text { " - the same" } else { " - DIFFERENT" }));
            if mine != text {
                worst = f64::INFINITY;
            }
        }
        if let Some(t) = text {
            let want = ids(&b["ids"]);
            let mine = byt5::ids(&t);
            if mine != want {
                log(format!("byt5 ids differ: {mine:?} vs {want:?}"));
                worst = f64::INFINITY;
            }
            let t0 = std::time::Instant::now();
            let got = m.byt5.encode(&m.nsd, &want)?;
            log(format!("byt5: {} tokens in {:.0} ms", want.len(), t0.elapsed().as_secs_f64() * 1e3));
            let (_, r) = read_npy(&dir.join("byt5.npy"))?;
            worst = worst.max(line("byt5", &got, &r, log));
        }
    }
    let meta = json(&dir.join("dit.json")).unwrap_or_default();
    let size = meta["size"].as_u64().unwrap_or(512) as usize;
    let steps = meta["steps"].as_u64().unwrap_or(8) as usize;
    let hw = (size / 32, size / 32);
    if stages.contains(&"dit") || stages.contains(&"steps") {
        let (_, sig_ref) = read_npy(&dir.join("sigmas.npy"))?;
        let shift = meta["shift"].as_f64().unwrap_or(4.0) as f32;
        worst = worst.max(line("sigmas", &sched::sigmas(steps, shift), &sig_ref, log));
        let guidance = m.dit.guided.then(|| meta["guidance"].as_f64().unwrap_or(3.25) as f32 * 1000.0);
        let enc = |p: &str| -> Result<crate::Encoded> {
            let (_, e) = read_npy(&dir.join(format!("{p}embeds.npy")))?;
            let b = read_npy(&dir.join(format!("{p}byt5.npy"))).ok().map(|x| x.1);
            Ok(crate::Encoded { text: m.dit.text(&m.nsd, &e, b.as_deref())?, glyphs: b.is_some() })
        };
        let pos = enc("")?;
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        let lat = DevBuf::from_f32(gpu, &noise)?;
        let ts = sched::timesteps(&sig_ref, m.dit.meanflow);
        let t0 = std::time::Instant::now();
        let st = m.dit.steps(&m.nsd, &ts, guidance)?;
        log(format!("dit: the {} steps' modulations in {:.0} ms", ts.len(), t0.elapsed().as_secs_f64() * 1e3));
        let (_, vec0) = read_npy(&dir.join("vec0.npy"))?;
        worst = worst.max(line("vec 0", &st.vec.to_f32()?[..dit::DIM], &vec0, log));
        let br = m.dit.batch(&m.nsd, &[&pos.text], hw)?;
        let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
        let t0 = std::time::Instant::now();
        let v = m.dit.forward(&m.nsd, &st, 0, &br, &[&lat], &mut |name, buf, r0, rows| {
            let all = buf.to_f32()?;
            taps.push((name.to_string(), all[r0 * dit::DIM..(r0 + rows) * dit::DIM].to_vec()));
            Ok(())
        })?;
        log(format!("dit: one pass over {} image + {} text tokens in {:.0} ms (with the checks' copies)", hw.0 * hw.1, pos.text.tokens(),
                    t0.elapsed().as_secs_f64() * 1e3));
        for (name, got) in &taps {
            if let Ok((_, want)) = read_npy(&dir.join(format!("{name}.npy"))) {
                worst = worst.max(line(name, got, &want, log));
            }
        }
        let cfg = meta["cfg"].as_f64().map(|c| c as f32);
        let neg = match cfg {
            Some(c) => Some((enc("neg_")?, c)),
            None => None,
        };
        let v0 = match &neg {
            Some((ne, c)) => {
                let bn = m.dit.batch(&m.nsd, &[&ne.text], hw)?;
                let vn = m.dit.forward(&m.nsd, &st, 0, &bn, &[&lat], &mut |_, _, _, _| Ok(()))?;
                let (a, b) = (v[0].to_f32()?, vn[0].to_f32()?);
                a.iter().zip(&b).map(|(p, n)| n + c * (p - n)).collect()
            }
            None => v[0].to_f32()?,
        };
        let (_, want) = read_npy(&dir.join("v0.npy"))?;
        worst = worst.max(line("velocity 0", &v0, &want, log));
        if stages.contains(&"steps") {
            let t0 = std::time::Instant::now();
            let negs = neg.as_ref().map(|(e, c)| (e, *c));
            m.denoise(&pos, negs, guidance.map(|g| g / 1000.0), &noise, hw, &sig_ref, crate::Guidance::Cfg, &mut |i, l| {
                if let Ok((_, want)) = read_npy(&dir.join(format!("latents_{i}.npy"))) {
                    worst = worst.max(line(&format!("latents {i}"), &l.to_f32()?, &want, log));
                }
                Ok(())
            })?;
            log(format!("dit: {} steps in {:.1} s", sig_ref.len() - 1, t0.elapsed().as_secs_f64()));
        }
    }
    if stages.contains(&"vae") {
        let lat_file = std::env::var("NS_CHECK_LATENTS").map(std::path::PathBuf::from).unwrap_or_else(|_| dir.join(format!("latents_{}.npy", steps - 1)));
        let (_, lat) = read_npy(&lat_file)?;
        let t0 = std::time::Instant::now();
        let got = m.vae.decode_float(&m.nsd, &lat, hw.0, hw.1)?;
        log(format!("vae: {}x{} in {:.0} ms", hw.1 * 32, hw.0 * 32, t0.elapsed().as_secs_f64() * 1e3));
        let (_, img) = read_npy(&dir.join("image.npy"))?;
        worst = worst.max(line("image", &got, &img, log));
        let (rgb, w, h) = m.vae.decode(&m.nsd, &lat, hw.0, hw.1)?;
        let pic = nextsycl_image::Picture { width: w as u32, height: h as u32, channels: 3, data: rgb };
        let out = dir.join("nextsycl.png");
        pic.write_png(&out, &[]).map_err(Error)?;
        log(format!("vae: wrote {}", out.display()));
    }
    Ok(worst)
}
