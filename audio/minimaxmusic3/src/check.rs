//! Stage by stage against the reference (reference/minimaxmusic3/ref.py: diffusers' own MiniMaxMusic3 code, float32,
//! on the same files): each stage starts from the reference's own input, so a difference shows where it is born.
//!
//! ```text
//!   ar     tokens.json               the prompt's ids
//!          -> prefill.npy            the prompt's last hidden state, both rows
//!          codes.npy (forced)        -> c0_logits.npy (frame 0's guided logits), lm_hidden.npy and depth.npy (every
//!                                       frame's language-model and depth hidden states), frame_hiddens.npy (mixed)
//!   dit    frame_hiddens + noise     -> condition.npy, v0_cond / v0_uncond.npy, latents_N.npy every step
//!   voc    latents.npy               -> audio.npy
//!   chunks frame_hiddens + noise_K   -> latents_K.npy (each window, its overlap), audio.npy (stitched)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ar::{Probe, CODEBOOKS, HIDDEN};
use crate::flow::{Flow, LATENT};
use crate::sample::Rng;
use crate::MiniMaxMusic3;

/// A little-endian .npy of float32 or int32: (shape, values as f32)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let int = head.contains("'<i4'");
    if !(head.contains("'<f4'") || int) || head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: {head} (float32 or int32, C order expected)", path.display())));
    }
    let shape: Vec<usize> = head
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let v = b[at + hl..].chunks_exact(4).map(|c| {
        let w = [c[0], c[1], c[2], c[3]];
        if int { i32::from_le_bytes(w) as f32 } else { f32::from_le_bytes(w) }
    }).collect();
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
    ((e2 / w2.max(1e-30)).sqrt(), mx, dot / (w2.sqrt() * g2.sqrt()).max(1e-30))
}

/// [C, L] channels first -> [L, C] rows
fn rows(v: &[f32], c: usize) -> Vec<f32> {
    let l = v.len() / c;
    (0..l * c).map(|i| v[(i % c) * l + i / c]).collect()
}

fn report(log: &mut dyn FnMut(String), worst: &mut f64, what: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        log(format!("{what:<28} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<28} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

/// The reference's frame_hiddens [F, 8 * HIDDEN] mixed as the engine mixes: [F, HIDDEN]
fn mixed(e: &MiniMaxMusic3, fh: &[f32]) -> Vec<f32> {
    let f = fh.len() / (CODEBOOKS * HIDDEN);
    let mut out = vec![0f32; f * HIDDEN];
    for i in 0..f {
        for j in 0..HIDDEN {
            let s: f32 = (0..CODEBOOKS).map(|l| e.flow.mix[l] * fh[(i * CODEBOOKS + l) * HIDDEN + j]).sum();
            out[i * HIDDEN + j] = e.flow.mix_scale * s;
        }
    }
    out
}

/// Every stage in `stages` ("ar", "dit", "voc", "chunks") against the dumps in `dir`: the worst relative error
pub fn run(e: &MiniMaxMusic3, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let (ops, nsd) = (&e.ops, &e.nsd);
    let mut worst = 0f64;
    let npy = |n: &str| read_npy(&dir.join(n));
    if stages.contains(&"ar") {
        let tj: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("tokens.json")).map_err(|x| Error(format!("tokens.json: {x}")))?)
            .map_err(|x| Error(format!("tokens.json: {x}")))?;
        let want: Vec<u32> = tj["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
        let text = tj["text"].as_str().unwrap_or("");
        let mine = e.tok.encode(text);
        log(format!("tokens: {} here, {} in the reference{}", mine.len(), want.len(), if mine == want { " - the same" } else { " - DIFFERENT" }));
        if mine != want {
            worst = f64::INFINITY;
        }
        let unc = crate::prompt::unconditional(&want);
        let (_, codes) = npy("codes.npy")?;
        let nf = codes.len() / CODEBOOKS;
        let s = e.ar.session(ops, want.len(), nf + 1)?;
        let t = Instant::now();
        e.ar.prefill(ops, nsd, &s, &want, &unc)?;
        log(format!("prefill: {} tokens in {:.2} s", want.len(), t.elapsed().as_secs_f64()));
        let mut p = Probe::default();
        e.ar.probe_prefill(&s, &mut p)?;
        report(log, &mut worst, "prefill (both rows)", &p.prefill, &npy("prefill.npy")?.1);
        let mut rng = Rng::new(0);
        let t = Instant::now();
        let mut mixed_here = vec![0f32; nf.saturating_sub(1) * HIDDEN];
        let fr = DevBuf::f32(&ops.gpu, HIDDEN)?;
        for f in 0..nf {
            let c: Vec<i32> = codes[f * CODEBOOKS..(f + 1) * CODEBOOKS].iter().map(|x| *x as i32).collect();
            crate::ar::Ar::check_codes(&c)?;
            e.ar.frame(ops, nsd, &s, want.len() + f, &mut rng, Some(&c), Some(&mut p))?;
            if f > 0 {
                ops.mix(s.stage.fp(), 8, HIDDEN, &e.flow.mix, e.flow.mix_scale, fr.fp())?;
                mixed_here[(f - 1) * HIDDEN..f * HIDDEN].copy_from_slice(&fr.to_f32()?);
            }
        }
        log(format!("{nf} frames (forced, read back) in {:.2} s", t.elapsed().as_secs_f64()));
        report(log, &mut worst, "frame 0 guided logits", &p.c0, &npy("c0_logits.npy")?.1);
        let (_, lm) = npy("lm_hidden.npy")?;
        let (_, dp) = npy("depth.npy")?;
        for f in [0, 1, nf / 2, nf - 1] {
            report(log, &mut worst, &format!("frame {f}: language model"), &p.lm[f], &lm[f * 2 * HIDDEN..(f + 1) * 2 * HIDDEN]);
            report(log, &mut worst, &format!("frame {f}: depth (c1..c7)"), &p.depth[f], &dp[f * 7 * HIDDEN..(f + 1) * 7 * HIDDEN]);
        }
        let all_lm: Vec<f32> = p.lm.concat();
        report(log, &mut worst, "every frame: language model", &all_lm, &lm[..all_lm.len()]);
        report(log, &mut worst, "every frame: depth", &p.depth.concat(), &dp);
        report(log, &mut worst, "frames, mixed", &mixed_here, &mixed(e, &npy("frame_hiddens.npy")?.1));
    }
    if stages.contains(&"dit") {
        let fh = mixed(e, &npy("frame_hiddens.npy")?.1);
        let n = fh.len() / HIDDEN;
        let frames = DevBuf::from_f32(&ops.gpu, &fh)?;
        let cond = e.flow.condition(ops, nsd, &frames, 0, n)?;
        report(log, &mut worst, "condition", &cond.to_f32()?, &npy("condition.npy")?.1);
        let (_, noise) = npy("noise.npy")?;
        let l = noise.len() / LATENT;
        let lat = DevBuf::from_f32(&ops.gpu, &rows(&noise, LATENT))?;
        let w = e.flow.work(ops, l)?;
        let (_, sig) = npy("sigmas.npy")?;
        let times = Flow::times(sig.len() - 1);
        report(log, &mut worst, "flow times", &times, &sig);
        let wc = DevBuf::from_f32(&ops.gpu, &npy("condition.npy")?.1)?;
        e.flow.velocity(ops, nsd, &w, &lat, &wc, times[0])?;
        let v = w.v.to_f32()?;
        report(log, &mut worst, "step 0 velocity (cond)", &v[..l * LATENT], &rows(&npy("v0_cond.npy")?.1, LATENT));
        report(log, &mut worst, "step 0 velocity (uncond)", &v[l * LATENT..], &rows(&npy("v0_uncond.npy")?.1, LATENT));
        let t = Instant::now();
        let steps = times.len() - 1;
        let mut lats = Vec::new();
        e.flow.denoise(ops, nsd, &w, &lat, &wc, &times, crate::flow::GUIDANCE, None, &mut |i| {
            if i == 0 || i + 1 == steps || i == steps / 2 {
                lats.push((i, lat.to_f32()?));
            }
            Ok(())
        })?;
        log(format!("{steps} steps of {l} latents in {:.2} s", t.elapsed().as_secs_f64()));
        for (i, got) in lats {
            report(log, &mut worst, &format!("latents after step {i}"), &got, &rows(&npy(&format!("latents_{i}.npy"))?.1, LATENT));
        }
    }
    if stages.contains(&"voc") {
        let (_, want_lat) = npy("latents.npy")?;
        let l = want_lat.len() / LATENT;
        let lat = DevBuf::from_f32(&ops.gpu, &rows(&want_lat, LATENT))?;
        let t = Instant::now();
        let wav = e.voc.decode(ops, nsd, &lat, l)?;
        log(format!("decoded {l} latents in {:.2} s", t.elapsed().as_secs_f64()));
        let wav: Vec<f32> = wav.iter().map(|x| x.clamp(-1.0, 1.0)).collect();
        report(log, &mut worst, "audio", &wav, &npy("audio.npy")?.1);
    }
    if stages.contains(&"chunks") {
        let fh = mixed(e, &npy("frame_hiddens.npy")?.1);
        let n = fh.len() / HIDDEN;
        let frames = DevBuf::from_f32(&ops.gpu, &fh)?;
        let windows = Flow::starts(n).len();
        let mut kept = Vec::new();
        let steps = std::env::var("NS_MM3_CHECK_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let out = e.sound(&frames, n, steps, crate::flow::GUIDANCE, &mut |k, _| Ok(rows(&npy(&format!("noise_{k}.npy"))?.1, LATENT)), Instant::now(),
                          &mut |_| Ok(()), Some(&mut kept))?;
        for (k, got) in kept.iter().enumerate().take(windows) {
            report(log, &mut worst, &format!("window {k} latents"), got, &rows(&npy(&format!("latents_{k}.npy"))?.1, LATENT));
        }
        let mut wav = out[0].clone();
        wav.extend_from_slice(&out[1]);
        report(log, &mut worst, "stitched audio", &wav, &npy("audio.npy")?.1);
    }
    Ok(worst)
}
