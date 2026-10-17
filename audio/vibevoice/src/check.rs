//! Stage by stage against the reference (reference/vibevoice/ref.py: the model's transformers implementation in
//! float32 on the CPU), each stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   prompt  request.json (+ voice{i}.wav) -> the token ids (ids.npy: exact), each voice's encoder latents
//!           (voice{i}_mean.npy), the acoustic connector on the reference's noised latents (voice_feats.npy ->
//!           voice_embeds.npy), the prompt's rows (prefill.npy)
//!   lm      prefill.npy, then the reference's tokens and next inputs (tokens.npy, embeds.npy: teacher forcing) -> each
//!           step's four logits (logits.npy) and each frame's positive and negative hidden states (cond.npy)
//!   head    cond.npy and noise.npy -> the head's first output (eps0.npy) and the sampled latents (latents.npy)
//!   codec   latents.npy -> the acoustic decoder's speech (speech.npy); speech.npy -> the semantic encoder's features
//!           (semantic.npy); both connectors -> the next inputs (embeds.npy)
//!   run     prefill.npy and noise.npy, free-running -> the frames, latents and speech against the reference's; the
//!           speech written as run.wav beside the dump
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Audio, Error, Reference, Result};
use nextsycl_core::DevBuf;

use crate::{bytes, Keep, Rng, Settings, VibeVoice};

/// A little-endian .npy of float32, int32 or int64: (shape, values as f64)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: Fortran order (ref.py writes C order)", path.display())));
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
        return Err(Error(format!("{}: {head} (float32, int32 or int64 expected)", path.display())));
    };
    Ok((shape, v))
}

/// (relative RMS error, max abs error, cosine)
pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        if !g.is_finite() || !w.is_finite() {
            continue;
        }
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

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

pub fn run(e: &VibeVoice, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(f32s(read_npy(&dir.join(n))?.1)) };
    let p = dir.join("request.json");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
        .map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let script = r["script"].as_str().unwrap_or_default().to_string();
    let n_voices = r["voices"].as_u64().unwrap_or(0) as usize;
    let cfg = r["cfg"].as_f64().unwrap_or(crate::CFG as f64) as f32;
    let steps = r["steps"].as_u64().unwrap_or(crate::STEPS as u64) as usize;
    log(format!("{n_voices} voices, cfg {cfg}, {steps} steps: {script:?}"));
    let h = e.lm.hidden;
    let lat = e.head.latent;
    if stages.contains(&"prompt") {
        let lines = crate::prompt::parse(&script);
        let mut voices = Vec::new();
        let mut frames = Vec::new();
        for i in 0..n_voices {
            let b = std::fs::read(dir.join(format!("voice{i}.wav"))).map_err(|x| Error(format!("voice{i}.wav: {x}")))?;
            let (samples, rate) = nextsycl_audio::decode_audio(&b).map_err(Error)?;
            let x = e.voice_samples(&Reference { samples, rate, text: None })?;
            frames.push(Some(x.len() / e.cfg.hop()));
            let t = Instant::now();
            let got = e.encode(&x)?;
            log(format!("{:<24} {} frames in {:.2} s", format!("voice {i} encoder"), x.len() / e.cfg.hop(), t.elapsed().as_secs_f64()));
            report(log, &mut worst, &format!("voice {i} latents"), &got, &want(&format!("voice{i}_mean.npy"))?);
            voices.push(got);
        }
        let (ids, placed) = crate::prompt::ids(&e.tok, e.specials(), &lines, &frames)?;
        let (_, w) = read_npy(&dir.join("ids.npy"))?;
        let w: Vec<u32> = w.into_iter().map(|x| x as u32).collect();
        let first = ids.iter().zip(&w).position(|(a, b)| a != b);
        log(format!("{:<24} {} here, {} in the reference; {}", "token ids", ids.len(), w.len(),
                    if ids == w { "identical".to_string() } else { format!("first difference at {first:?}") }));
        if ids != w {
            worst = f64::INFINITY;
        }
        let mut vrows = Vec::new();
        if n_voices > 0 {
            let feats = want("voice_feats.npy")?;
            let got = e.connect(&e.acoustic, &feats, lat)?;
            report(log, &mut worst, "voice rows (connector)", &got, &want("voice_embeds.npy")?);
            // the reference's rows, split by voice in the prompt's order
            let mut at = 0;
            for (_, _, f) in &placed {
                vrows.push(got[at * h..(at + f) * h].to_vec());
                at += f;
            }
        }
        let rows = e.prompt_rows(&ids, &placed, &vrows)?;
        report(log, &mut worst, "prompt rows", &rows, &want("prefill.npy")?);
    }
    let prefill = want("prefill.npy")?;
    let ops = &e.ops;
    let g = &ops.gpu;
    if stages.contains(&"lm") {
        let toks: Vec<u32> = read_npy(&dir.join("tokens.npy"))?.1.into_iter().map(|x| x as u32).collect();
        let embeds = want("embeds.npy")?;
        let l = prefill.len() / h;
        let pos = e.lm.cache(ops, crate::CHUNK, l + toks.len() + 1)?;
        let neg = e.lm.cache(ops, 2, toks.len() + 2)?;
        let (cond, tmp, next, prev) = (DevBuf::f32(g, h)?, DevBuf::f32(g, h)?, DevBuf::f32(g, h)?, DevBuf::f32(g, h)?);
        let bos = DevBuf::from_f32(g, &e.embedding(e.cfg.audio_bos)?)?;
        let logits = DevBuf::f32(g, 4)?;
        let t = Instant::now();
        e.lm.prefill(ops, &e.nsd, &pos, &prefill, 0, &cond)?;
        let (mut got_l, mut got_c, mut agree) = (Vec::new(), Vec::new(), 0);
        let (mut frame, mut np, mut have_prev) = (0usize, 0usize, false);
        let tokens = [e.cfg.audio_bos, e.cfg.audio_eos, e.cfg.audio, e.cfg.eos];
        for (step, &tok) in toks.iter().enumerate() {
            ops.gemv(cond.fp(), 1, &e.logit_rows, logits.fp(), 4, false)?;
            let l4 = logits.to_f32()?;
            let best = (0..4).fold(0, |b, i| if l4[i] > l4[b] { i } else { b });
            if tokens[best] == tok {
                agree += 1;
            }
            got_l.extend_from_slice(&l4);
            if tok == e.cfg.audio_bos {
                e.lm.step(ops, &e.nsd, &neg, &bos, 0, &tmp)?;
                np = 1;
            }
            if tok == e.cfg.audio {
                e.lm.step(ops, &e.nsd, &neg, if have_prev { &prev } else { &bos }, np, &tmp)?;
                np += 1;
                got_c.extend(cond.to_f32()?);
                got_c.extend(tmp.to_f32()?);
                next.write(0, &bytes(&embeds[frame * h..(frame + 1) * h]))?;
                frame += 1;
            } else {
                next.write(0, &bytes(&e.embedding(tok)?))?;
            }
            if tok == e.cfg.eos || step + 1 == toks.len() {
                break;
            }
            e.lm.step(ops, &e.nsd, &pos, &next, l + step, &cond)?;
            prev.copy_within(0, &next, 0, h * 4)?;
            have_prev = true;
        }
        log(format!("{:<24} {} steps ({frame} frames) in {:.1} s; the best of four agrees at {agree} of {}", "lm (forced)", toks.len(),
                    t.elapsed().as_secs_f64(), toks.len()));
        report(log, &mut worst, "logits", &got_l, &want("logits.npy")?);
        let wc = want("cond.npy")?;
        let pick = |v: &[f32], b: usize| -> Vec<f32> { v.chunks(2 * h).flat_map(|c| c[b * h..(b + 1) * h].to_vec()).collect() };
        report(log, &mut worst, "hidden (positive)", &pick(&got_c, 0), &pick(&wc, 0));
        report(log, &mut worst, "hidden (negative)", &pick(&got_c, 1), &pick(&wc, 1));
    }
    if stages.contains(&"head") {
        let (wc, noise) = (want("cond.npy")?, want("noise.npy")?);
        let mut hw = e.head.work(ops, steps)?;
        let cond = DevBuf::f32(g, 2 * h)?;
        let (mut eps0, mut lats) = (Vec::new(), Vec::new());
        let t = Instant::now();
        let n = noise.len() / lat;
        for f in 0..n {
            cond.write(0, &bytes(&wc[f * 2 * h..(f + 1) * 2 * h]))?;
            let mut first = Vec::new();
            lats.extend(e.head.sample(ops, &e.nsd, &mut hw, &cond, &noise[f * lat..(f + 1) * lat], cfg, Some(&mut first))?);
            eps0.extend(first);
        }
        log(format!("{:<24} {n} frames in {:.2} s", "head", t.elapsed().as_secs_f64()));
        report(log, &mut worst, "head first output", &eps0, &want("eps0.npy")?);
        report(log, &mut worst, "latents", &lats, &want("latents.npy")?);
    }
    if stages.contains(&"codec") {
        let (lats, speech, sem) = (want("latents.npy")?, want("speech.npy")?, want("semantic.npy")?);
        let n = lats.len() / lat;
        let hop = e.cfg.hop();
        let (dw, sw) = (e.dec.work(ops, 1)?, e.sem.work(ops, hop)?);
        let (ds, ss) = (e.dec.stream(ops)?, e.sem.stream(ops)?);
        let (lin, chunk) = (DevBuf::f32(g, lat)?, DevBuf::f32(g, hop)?);
        let (mut got_s, mut got_m) = (Vec::new(), Vec::new());
        let t = Instant::now();
        for f in 0..n {
            let un: Vec<f32> = lats[f * lat..(f + 1) * lat].iter().map(|v| v / e.scale - e.bias).collect();
            lin.write(0, &bytes(&un))?;
            let (y, rows) = e.dec.push(ops, &e.nsd, &ds, &lin, 1, &dw)?;
            let mut v = vec![0f32; rows];
            y.read(0, crate::as_bytes_mut(&mut v))?;
            got_s.extend(v);
            // the semantic encoder on the reference's own samples
            chunk.write(0, &bytes(&speech[f * hop..(f + 1) * hop]))?;
            let (y, _) = e.sem.push(ops, &e.nsd, &ss, &chunk, hop, &sw)?;
            let mut m = vec![0f32; e.sem.dim_out];
            y.read(0, crate::as_bytes_mut(&mut m))?;
            got_m.extend(m);
        }
        log(format!("{:<24} {n} frames in {:.2} s", "decoder + semantic", t.elapsed().as_secs_f64()));
        report(log, &mut worst, "speech (decoder)", &got_s, &speech);
        report(log, &mut worst, "semantic features", &got_m, &sem);
        let mut a = e.connect(&e.acoustic, &lats, lat)?;
        let b = e.connect(&e.semantic, &sem, e.sem.dim_out)?;
        a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
        report(log, &mut worst, "next inputs (connectors)", &a, &want("embeds.npy")?);
    }
    if stages.contains(&"run") {
        let noise = want("noise.npy")?;
        let wl = want("latents.npy")?;
        let n = noise.len() / lat;
        let per: Vec<Vec<f32>> = noise.chunks(lat).map(<[f32]>::to_vec).collect();
        let mut k = Keep::default();
        let t = Instant::now();
        let s = Settings { cfg, steps, frames: n + 8, seed: 0 };
        // the reference's noise for its frames; past them (a longer run here) the seed's
        let extra: Vec<Vec<f32>> = {
            let mut r = Rng::new(0);
            (0..8).map(|_| (0..lat).map(|_| r.normal()).collect()).collect()
        };
        let all: Vec<Vec<f32>> = per.iter().cloned().chain(extra).collect();
        let speech = e.speak(&prefill, s, t, &mut |_| Ok(()), Some(&all), Some(&mut k))?;
        let secs = t.elapsed().as_secs_f64();
        log(format!("{:<24} {} frames here, {n} in the reference, in {secs:.1} s ({:.2} s of speech: real time x{:.2})", "run (free)", k.latents.len(),
                    speech.len() as f64 / crate::RATE as f64, speech.len() as f64 / crate::RATE as f64 / secs));
        let m = k.latents.len().min(n);
        // an autoregressive run departs from the reference once a small difference changes a frame: where it does
        let rel = |f: usize| compare(&k.latents[f], &wl[f * lat..(f + 1) * lat]).0;
        let depart = |t: f64| (0..m).position(|f| rel(f) > t).map_or(format!("none of {m}"), |f| f.to_string());
        log(format!("{:<24} the first frame 1e-2 off: {}; 1e-1 off: {}", "run departs", depart(1e-2), depart(1e-1)));
        let got: Vec<f32> = k.latents[..m].concat();
        report(log, &mut worst, "run latents", &got, &wl[..m * lat]);
        let ws = want("speech.npy")?;
        let m = speech.len().min(ws.len());
        report(log, &mut worst, "run speech", &speech[..m], &ws[..m]);
        let a = Audio { rate: crate::RATE, channels: 1, samples: speech };
        a.write_wav(&dir.join("run.wav"), &[("INAM", &script)]).map_err(Error)?;
    }
    Ok(worst)
}
