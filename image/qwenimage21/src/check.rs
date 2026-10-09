//! Stage by stage against the reference (reference/qwenimage21/ref.py, run on the same quantized files): each stage
//! starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   te     tokens.json      -> embeds.npy        the text encoder, the system turn dropped
//!   dit    embeds + noise   -> block0.npy        block 0's image rows at the first step
//!                           -> v0.npy            the first velocity
//!                           -> latents_N.npy     every step (Euler over sigmas.npy)
//!   vae    latents_last     -> image.npy         RGBA in [-1, 1]
//! ```

use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_image::{Error, Result};

use crate::{dit, sched, QwenImage21};

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
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<14} rel rms {rel:.2e}  max abs {mx:.3e}  cosine {cos:.6}  ({} values)", want.len()));
    rel
}

/// Every stage in `stages` ("te", "dit", "steps", "vae") against the dumps in `dir`; the worst relative error
pub fn run(m: &QwenImage21, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let gpu = &m.nsd.gpu;
    let mut worst = 0f64;
    let toks: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokens.json")).map_err(|e| Error(format!("tokens.json: {e}")))?)
        .map_err(|e| Error(format!("tokens.json: {e}")))?;
    let ids: Vec<u32> = toks["ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default();
    let drop = toks["drop"].as_u64().unwrap_or(0) as usize;
    let (_, embeds) = read_npy(&dir.join("embeds.npy"))?;
    let tokens = ids.len() - drop;
    if stages.contains(&"te") {
        let t0 = std::time::Instant::now();
        if let Ok((_, l0)) = read_npy(&dir.join("te_layer0.npy")) {
            worst = worst.max(line("te layer 0", &m.te.encode_layers(&m.nsd, &ids, 1)?.to_f32()?, &l0, log));
        }
        let all = m.te.encode_ids(&m.nsd, &ids)?.to_f32()?;
        let d = m.te.hidden;
        log(format!("te: {} tokens in {:.0} ms", ids.len(), t0.elapsed().as_secs_f64() * 1e3));
        worst = worst.max(line("te embeds", &all[drop * d..], &embeds, log));
    }
    if stages.contains(&"edit-te") {
        let vis = m.te.vision.as_ref().ok_or_else(|| Error("the text encoder's file has no vision tower".into()))?;
        // the processor's patches, from the reference's resized picture (opaque here: over white is itself)
        let (shape, rgba) = read_npy(&dir.join("cond_rgba.npy"))?;
        let (hh, ww) = (shape[0], shape[1]);
        let rgb: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| {
            let a = p[3] / 255.0;
            [p[0] * a + 255.0 * (1.0 - a), p[1] * a + 255.0 * (1.0 - a), p[2] * a + 255.0 * (1.0 - a)]
        }).collect();
        let (px, grid) = nextsycl_qwen3vl::vision::patches(&rgb, hh, ww)?;
        let (_, want_px) = read_npy(&dir.join("pixel_values.npy"))?;
        worst = worst.max(line("patches", &px, &want_px, log));
        let t0 = std::time::Instant::now();
        let seen = vis.see_patches(&m.nsd, &want_px, grid)?;
        log(format!("vision: {} patches in {:.0} ms", grid.0 * grid.1, t0.elapsed().as_secs_f64() * 1e3));
        let (_, merged) = read_npy(&dir.join("vis_merged.npy"))?;
        worst = worst.max(line("vision tokens", &seen.tokens.to_f32()?, &merged, log));
        let (_, deep) = read_npy(&dir.join("vis_deep.npy"))?;
        let per = merged.len();
        for (i, ds) in seen.deep.iter().enumerate() {
            worst = worst.max(line(&format!("deepstack {i}"), &ds.to_f32()?, &deep[i * per..(i + 1) * per], log));
        }
        let pad = m.te.tok.id("<|image_pad|>").unwrap_or(0);
        if let Ok(pr) = std::env::var("NS_CHECK_PROMPT") {
            let mine = m.edit_ids(&pr, &[seen.n])?;
            log(format!("edit ids: {} here, {} in the reference{}", mine.len(), ids.len(), if mine == ids { " - the same" } else { " - DIFFERENT" }));
            if mine != ids {
                let at = mine.iter().zip(&ids).position(|(a, b)| a != b).unwrap_or(0);
                log(format!("  first difference at {at}: {:?} vs {:?}", &mine[at..(at + 8).min(mine.len())], &ids[at..(at + 8).min(ids.len())]));
                worst = f64::INFINITY;
            }
        }
        let pos: Vec<f32> = nextsycl_qwen3vl::TextEncoder::positions(&ids, pad, &[grid])?.iter().flat_map(|p| p.iter().map(|x| *x as f32)).collect();
        let (_, want_pos) = read_npy(&dir.join("positions.npy"))?;
        worst = worst.max(line("positions", &pos, &want_pos, log));
        if let Ok((_, l0)) = read_npy(&dir.join("te_layer0.npy")) {
            worst = worst.max(line("te layer 0", &m.te.encode_seen(&m.nsd, &ids, &[&seen], 1)?.to_f32()?, &l0, log));
        }
        let all = m.te.encode_seen(&m.nsd, &ids, &[&seen], usize::MAX)?.to_f32()?;
        let d = m.te.hidden;
        let got = &all[drop * d..];
        line("te embeds", got, &embeds, log);
        // the denoiser reads the text's rows only (the picture's are replaced by its latents)
        let text: Vec<usize> = ids[drop..].iter().enumerate().filter(|(_, t)| **t != pad).map(|(i, _)| i).collect();
        let pick = |v: &[f32]| -> Vec<f32> { text.iter().flat_map(|i| v[i * d..(i + 1) * d].iter().copied()).collect() };
        worst = worst.max(line("te text rows", &pick(got), &pick(&embeds), log));
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("dit.json")).unwrap_or_else(|_| "{}".into())).unwrap_or_default();
    let size = meta["size"].as_u64().unwrap_or(512) as usize;
    let steps = meta["steps"].as_u64().unwrap_or(20) as usize;
    let hw = (size / 16, size / 16);
    let n = hw.0 * hw.1;
    if stages.contains(&"dit") || stages.contains(&"steps") {
        let (_, sig_ref) = read_npy(&dir.join("sigmas.npy"))?;
        let sig = match &m.preset {
            Some(p) => sched::preset(&p.nodes, n, p.dynamic),
            None => sched::sigmas(steps, n),
        };
        worst = worst.max(line("sigmas", &sig, &sig_ref, log));
        let e = DevBuf::from_f32(gpu, &embeds)?;
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        let lat = DevBuf::from_f32(gpu, &noise)?;
        let t0 = std::time::Instant::now();
        let pre = m.dit.prefix(&m.nsd, &e, tokens, embeds.len() / tokens)?;
        log(format!("dit: the text's {tokens} tokens in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3));
        let b0 = DevBuf::f32(gpu, n * dit::DIM)?;
        let t0 = std::time::Instant::now();
        let v = m.dit.velocity(&m.nsd, &pre, &lat, hw, sig_ref[0], Some(&b0))?;
        log(format!("dit: one step over {n} image tokens in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3));
        let (_, block0) = read_npy(&dir.join("block0.npy"))?;
        worst = worst.max(line("block 0", &b0.to_f32()?, &block0[tokens * dit::DIM..], log));
        let (_, v0) = read_npy(&dir.join("v0.npy"))?;
        worst = worst.max(line("velocity 0", &v.to_f32()?, &v0, log));
        if stages.contains(&"steps") {
            let lat = DevBuf::from_f32(gpu, &noise)?;
            let t0 = std::time::Instant::now();
            m.denoise(&e, tokens, &lat, hw, &sig_ref, &mut |i, l| {
                let (_, want) = read_npy(&dir.join(format!("latents_{i}.npy")))?;
                let rel = line(&format!("latents {i}"), &l.to_f32()?, &want, log);
                worst = worst.max(rel);
                Ok(())
            })?;
            log(format!("dit: {steps} steps in {:.1} s", t0.elapsed().as_secs_f64()));
        }
    }
    if stages.contains(&"edit-dit") {
        // the denoiser with the condition picture in its slots: the reference's embeddings and latents in
        let pads: Vec<bool> = toks["image_pad"].as_array().map(|a| a.iter().map(|x| x.as_u64() == Some(1)).collect()).unwrap_or_default();
        let (h, w) = (toks["h"].as_u64().unwrap_or(512) as usize / 16, toks["w"].as_u64().unwrap_or(512) as usize / 16);
        let raw = |f: String| -> Result<Vec<f32>> {
            Ok(std::fs::read(&f).map_err(|e| Error(format!("{f}: {e}")))?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let (_, cl) = read_npy(&dir.join("cond_latents.npy"))?;
        let cl = match std::env::var("NS_CHECK_COND") { Ok(f) => raw(f)?, Err(_) => cl };
        let cond = DevBuf::from_f32(gpu, &cl)?;
        // NS_CHECK_EMBEDS: raw float32 embeddings to use instead of the reference's (the engine's own, to see their effect)
        let embeds = match std::env::var("NS_CHECK_EMBEDS") {
            Ok(f) => std::fs::read(&f).map_err(|e| Error(format!("{f}: {e}")))?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Err(_) => embeds.clone(),
        };
        let e = DevBuf::from_f32(gpu, &embeds)?;
        let (_, noise) = read_npy(&dir.join("noise.npy"))?;
        let noise = match std::env::var("NS_CHECK_NOISE") { Ok(f) => raw(f)?, Err(_) => noise };
        let (_, sig_ref) = read_npy(&dir.join("sigmas.npy"))?;
        let mine = sched::sigmas(sig_ref.len() - 1, h * w);
        line("edit sigmas", &mine, &sig_ref, log);
        log(format!("  here {:?}\n  ref  {:?}", &mine[..4.min(mine.len())], &sig_ref[..4.min(sig_ref.len())]));
        let lat = DevBuf::from_f32(gpu, &noise)?;
        let t0 = std::time::Instant::now();
        let pre = m.dit.prefix_with(&m.nsd, &e, tokens, embeds.len() / tokens, &pads, &[(&cond, (h, w))])?;
        log(format!("dit: the prefix (text + the picture's {} latents: {} rows) in {:.0} ms", h * w, pre.tokens, t0.elapsed().as_secs_f64() * 1e3));
        let v = m.dit.velocity(&m.nsd, &pre, &lat, (h, w), sig_ref[0], None)?;
        let (_, v0) = read_npy(&dir.join("v0.npy"))?;
        worst = worst.max(line("edit velocity 0", &v.to_f32()?, &v0, log));
        let lat = DevBuf::from_f32(gpu, &noise)?;
        let t0 = std::time::Instant::now();
        m.denoise_cfg(&e, tokens, &pads, None, &[(&cond, (h, w))], &lat, (h, w), &sig_ref, &mut |i, l| {
            let (_, want) = read_npy(&dir.join(format!("latents_{i}.npy")))?;
            worst = worst.max(line(&format!("edit latents {i}"), &l.to_f32()?, &want, log));
            Ok(())
        })?;
        log(format!("dit: {} steps in {:.1} s", sig_ref.len() - 1, t0.elapsed().as_secs_f64()));
        let (rgba, pw, ph) = m.vae.decode(&m.nsd, &lat.to_f32()?, h, w)?;
        let (_, img) = read_npy(&dir.join("image.npy"))?;
        let want: Vec<f32> = img.iter().map(|x| ((x / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round()).collect();
        worst = worst.max(line("edit image", &rgba.iter().map(|x| *x as f32).collect::<Vec<_>>(), &want, log));
        let pic = nextsycl_image::Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgba };
        pic.write_png(&dir.join("nextsycl.png"), &[]).map_err(Error)?;
    }
    if stages.contains(&"edit-pre") {
        // the pipeline's preprocessing: the picture resized (PIL's Lanczos) to the reference's size
        let (shape, want) = read_npy(&dir.join("cond_rgba.npy"))?;
        let src = std::env::var("NS_CHECK_PICTURE").map_err(|_| Error("edit-pre: NS_CHECK_PICTURE=<the original picture>".into()))?;
        let p = nextsycl_image::Picture::read_png(Path::new(&src)).map_err(Error)?.rgba().resize_lanczos(shape[1] as u32, shape[0] as u32);
        worst = worst.max(line("resized", &p.data.iter().map(|x| *x as f32).collect::<Vec<_>>(), &want, log));
    }
    if stages.contains(&"edit-vae") {
        // an edit's condition picture through the VAE's encoder
        let (shape, rgba) = read_npy(&dir.join("vae_in.npy"))?;
        let (hh, ww) = (shape[0], shape[1]);
        let t0 = std::time::Instant::now();
        let got = m.vae.encode(&m.nsd, &rgba, hh, ww)?;
        log(format!("vae encoder: {ww}x{hh} in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3));
        let (_, want) = read_npy(&dir.join("cond_latents.npy"))?;
        worst = worst.max(line("cond latents", &got, &want, log));
    }
    if stages.contains(&"vae") {
        let (_, lat) = read_npy(&dir.join(format!("latents_{}.npy", steps - 1)))?;
        let t0 = std::time::Instant::now();
        let (rgba, w, h) = m.vae.decode(&m.nsd, &lat, hw.0, hw.1)?;
        log(format!("vae: {w}x{h} in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3));
        let (_, img) = read_npy(&dir.join("image.npy"))?;
        let want: Vec<f32> = img.iter().map(|x| ((x / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round()).collect();
        let got: Vec<f32> = rgba.iter().map(|x| *x as f32).collect();
        worst = worst.max(line("image (0-255)", &got, &want, log));
        let pic = nextsycl_image::Picture { width: w as u32, height: h as u32, channels: 4, data: rgba };
        let out = dir.join("nextsycl.png");
        pic.write_png(&out, &[]).map_err(Error)?;
        log(format!("vae: wrote {}", out.display()));
    }
    Ok(worst)
}
