//! Stage by stage against the reference (reference/voxcpm2/ref.py: voxcpm's own code in float32 on the same files),
//! each from the reference's own input.
//!
//! ```text
//!   prompt   request.json (+ ref16k / prompt16k.npy) -> tokens.npy, audio_mask.npy, audio_feat.npy (the VAE's encoder)
//!   prefill  the reference's prompt                   -> lm0.npy, res0.npy
//!   patches  noise.npy forced                         -> feats.npy (each patch from the reference's noise)
//!   vae      feats.npy                                -> speech.npy (48 kHz)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::model::{FEAT, PATCH};
use crate::{Prompt, Spec, VoxCpm2};

/// A little-endian .npy of float32, int32 or int64 (C order): (shape, values as f64)
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
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

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

fn report(log: &mut dyn FnMut(String), worst: &mut f64, what: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        log(format!("{what:<24} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<24} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

pub fn run(e: &VoxCpm2, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(f32s(read_npy(&dir.join(n))?.1)) };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("request.json")).map_err(|e| Error(e.to_string()))?)
        .map_err(|e| Error(e.to_string()))?;
    let s = |k: &str| r[k].as_str().unwrap_or("").to_string();
    // the reference's prompt, as it gave it
    let toks: Vec<u32> = read_npy(&dir.join("tokens.npy"))?.1.iter().map(|x| *x as u32).collect();
    let audio: Vec<bool> = read_npy(&dir.join("audio_mask.npy"))?.1.iter().map(|x| *x != 0.0).collect();
    let feats = want("audio_feat.npy")?;
    let rp = Prompt {
        patches: feats.chunks(PATCH * FEAT).map(|c| c.to_vec()).collect(),
        target: e.tokens(&s("text")).len(),
        tokens: toks.clone(),
        audio: audio.clone(),
    };
    if stages.contains(&"prompt") {
        let spec = Spec {
            text: s("text"),
            reference: read_npy(&dir.join("ref16k.npy")).ok().map(|(_, v)| f32s(v)),
            prompt: read_npy(&dir.join("prompt16k.npy")).ok().map(|(_, v)| (f32s(v), s("prompt_text"))),
        };
        let p = e.prompt(&spec)?;
        let same = p.tokens.len() == toks.len() && p.tokens.iter().zip(&toks).all(|(a, b)| a == b) && p.audio == audio;
        log(format!("{:<24} {} rows here, {} in the reference: {}", "prompt tokens", p.tokens.len(), toks.len(), if same { "identical" } else { "DIFFERENT" }));
        if !same {
            log(format!("  here {:?}\n  ref  {:?}", &p.tokens[..p.tokens.len().min(40)], &toks[..toks.len().min(40)]));
            worst = f64::INFINITY;
        }
        if audio.iter().any(|a| *a) && p.patches.len() == rp.patches.len() {
            let (g, w): (Vec<f32>, Vec<f32>) = p.patches.iter().zip(&rp.patches).zip(&audio).filter(|(_, a)| **a)
                .flat_map(|((g, w), _)| g.iter().copied().zip(w.iter().copied())).unzip();
            report(log, &mut worst, "recording's latents", &g, &w);
        }
    }
    if stages.contains(&"prefill") || stages.contains(&"patches") {
        let noise = want("noise.npy")?;
        let wfeats = want("feats.npy")?;
        let n = noise.len() / (FEAT * PATCH);
        let steps = r["steps"].as_u64().unwrap_or(10) as usize;
        let cfg = r["cfg"].as_f64().unwrap_or(2.0) as f32;
        let mut keep = Vec::new();
        let forced = stages.contains(&"patches");
        let t0 = Instant::now();
        // each patch from the reference's noise (draws past its last: zeros); a prefill check makes one
        let (got, context) = e.generate_patches(&rp, if forced { n } else { 1 }, steps, cfg, &mut |i| {
            noise.get(i * FEAT * PATCH..(i + 1) * FEAT * PATCH).map(|x| x.to_vec()).unwrap_or_else(|| vec![0.0; FEAT * PATCH])
        }, t0, &mut |_| Ok(()), Some(&mut keep), None)?;
        report(log, &mut worst, "base LM (last row)", &keep[0], &want("lm0.npy")?);
        report(log, &mut worst, "residual LM (last row)", &keep[1], &want("res0.npy")?);
        if forced {
            let got: Vec<f32> = got.concat();
            let k = got.len().min(wfeats.len());
            log(format!("{:<24} {} patches here ({} context), {} in the reference; {:.1} s", "patches (forced noise)", got.len() / (FEAT * PATCH), context,
                        wfeats.len() / (FEAT * PATCH), t0.elapsed().as_secs_f64()));
            let rels: Vec<String> = got[..k].chunks(FEAT * PATCH).zip(wfeats[..k].chunks(FEAT * PATCH)).map(|(g, w)| format!("{:.0e}", compare(g, w).0)).collect();
            log(format!("{:<24} {}", "each patch (rel)", rels.join(" ")));
            report(log, &mut worst, "all patches", &got[..k], &wfeats[..k]);
            // the reference's patches fed back: each patch from the same history as the reference's
            let teach: Vec<Vec<f32>> = wfeats.chunks(FEAT * PATCH).map(|c| c.to_vec()).collect();
            let (tg, _) = e.generate_patches(&rp, n, steps, cfg, &mut |i| {
                noise.get(i * FEAT * PATCH..(i + 1) * FEAT * PATCH).map(|x| x.to_vec()).unwrap_or_else(|| vec![0.0; FEAT * PATCH])
            }, t0, &mut |_| Ok(()), None, Some(&teach[context..]))?;
            let tg: Vec<f32> = tg.concat();
            let k = tg.len().min(wfeats.len());
            let rels: Vec<String> = tg[..k].chunks(FEAT * PATCH).zip(wfeats[..k].chunks(FEAT * PATCH)).map(|(g, w)| format!("{:.0e}", compare(g, w).0)).collect();
            log(format!("{:<24} {}", "teacher-forced (rel)", rels.join(" ")));
            report(log, &mut worst, "teacher-forced, all", &tg[..k], &wfeats[..k]);
        }
    }
    if stages.contains(&"vae") {
        let wfeats = want("feats.npy")?;
        let patches: Vec<Vec<f32>> = wfeats.chunks(FEAT * PATCH).map(|c| c.to_vec()).collect();
        let context = if audio.last() == Some(&true) { crate::CONTEXT.min(audio.iter().filter(|a| **a).count()) } else { 0 };
        let t = Instant::now();
        let wav = e.decode(&patches, context)?;
        log(format!("vae: {} patches to {:.2} s in {:.2} s", patches.len(), wav.len() as f64 / 48000.0, t.elapsed().as_secs_f64()));
        report(log, &mut worst, "speech", &wav, &want("speech.npy")?);
    }
    Ok(worst)
}
