//! The engine against reference/yue2/ref.py's dumps (upstream's yue2_infer wheel as it is, on the CPU in float32 or a
//! card in bf16, the language model the same Q8_0 file dequantized or the bf16 checkpoint): each stage fed the
//! reference's own input, so a stage's error is its own.
//!
//! ```text
//!   tokens  the first prompt, the semantic one (with the dumped score), the twin's  -> prefix, sem_prefix, neg_prefix (exactly)
//!   ar      the whole sequence teacher-forced (prompt + score + codes)                -> ar_logits at ar_rows: cosine, argmax
//!   neg     the twin's prompt                                                         -> neg_logits
//!   noise   torch.randn((frames, 64)) from the seed                                    -> noise (to 1e-6)
//!   vel     the chunk's AR rows, the reference's noise at t = 1                       -> vel0
//!   steps   each midpoint step from the reference's state                             -> mids[i], states[i + 1]
//!   solve   every step from the noise                                                 -> latents
//!   vae     the reference's latents                                                   -> audio [samples, 2]; ours.wav written
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::torch_rng::Mt19937;

use crate::lm::{Scratch, LATENT, VOCAB};
use crate::prompt::{self, Cot, CODEC_OFFSET, MUSIC_END};
use crate::Yue2;

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
    let mismatch = if got.len() != want.len() { format!("  LENGTHS {} vs {}", got.len(), want.len()) } else { String::new() };
    log(format!("{name:<28} rel {rel:.2e}  max {mx:.3e}  cos {cos:.6}  ({} values){mismatch}", want.len().min(got.len())));
    if got.len() != want.len() { 0.0 } else { cos }
}

/// Runs the stages; the lowest cosine seen
pub fn run(e: &Yue2, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> { read_npy(&dir.join(format!("{name}.npy"))).map(|(s, v)| (s, v.into_iter().map(|x| x as f32).collect())) };
    let ids = |name: &str| -> Result<Vec<u32>> { Ok(read_npy(&dir.join(format!("{name}.npy")))?.1.iter().map(|x| *x as u32).collect()) };
    let req: serde_json::Value = std::fs::read(dir.join("request.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| Error(format!("{}: no request.json", dir.display())))?;
    let cot = Cot::parse(req["cot"].as_str().unwrap_or("full")).ok_or_else(|| Error("request.json: cot".into()))?;
    let seed = req["seed"].as_u64().unwrap_or(0);
    let steps = req["steps"].as_u64().unwrap_or(32) as usize;
    let cfg = req["cfg"].as_f64().unwrap_or(1.0);
    let (style, lyrics) = (req["style"].as_str().unwrap_or(""), req["lyrics"].as_str().unwrap_or(""));
    let mut worst = 1f64;
    let mut s = Scratch::default();
    let abc = ids("abc").unwrap_or_default();
    let sem_prefix = ids("sem_prefix")?;
    let codes = ids("codes")?;
    if stages.contains(&"tokens") {
        let ours = prompt::prefix(&e.tok, cot, style, lyrics);
        let want = ids("prefix")?;
        let same = |a: &[u32], b: &[u32]| a == b;
        log(format!("tokens: prefix {} ({} tokens)", if same(&ours, &want) { "identical" } else { "DIFFER" }, want.len()));
        if !same(&ours, &want) {
            let at = ours.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(ours.len().min(want.len()));
            log(format!("  first difference at {at}: ours {:?} theirs {:?}", &ours[at..(at + 8).min(ours.len())], &want[at..(at + 8).min(want.len())]));
            worst = 0.0;
        }
        let sp = prompt::semantic(&want, cot, &abc);
        log(format!("tokens: semantic prompt {} ({} tokens)", if sp == sem_prefix { "identical" } else { "DIFFER" }, sem_prefix.len()));
        if sp != sem_prefix {
            worst = 0.0;
        }
        if let Ok(np) = ids("neg_prefix") {
            let ours = prompt::negative(&e.tok, cot, &abc);
            log(format!("tokens: the twin's prompt {} ({} tokens)", if ours == np { "identical" } else { "DIFFER" }, np.len()));
            if ours != np {
                worst = 0.0;
            }
        }
    }
    if stages.contains(&"ar") {
        let t0 = Instant::now();
        let mut seq = sem_prefix.clone();
        seq.extend(codes.iter().map(|c| c + CODEC_OFFSET));
        let rows: Vec<usize> = ids("ar_rows")?.into_iter().map(|r| r as usize).collect();
        let (_, want) = get("ar_logits")?;
        let mut c = e.lm.cache(seq.len() + 1)?;
        let x = e.lm.prefill(&mut s, &mut c, &seq, &rows)?;
        let got = e.lm.logits_of(&mut s, &x, 0, VOCAB)?;
        let mut agree = 0;
        for (i, r) in rows.iter().enumerate() {
            let (g, w) = (&got[i * VOCAB..(i + 1) * VOCAB], &want[i * VOCAB..(i + 1) * VOCAB]);
            // the argmax over the phase's candidates: a score row's text tokens and ABC_END, a code row's codes and MUSIC_END
            let code_row = *r >= sem_prefix.len() - 1;
            let range = |v: &[f32]| -> usize {
                let (lo, hi) = if code_row { (MUSIC_END as usize, VOCAB - 83) } else { (0, prompt::ABC_END as usize + 1) };
                lo + crate::sample::argmax(&v[lo..hi])
            };
            let same = range(g) == range(w);
            agree += same as usize;
            worst = worst.min(note(&format!("ar row {r}{}", if code_row { " (codes)" } else { " (score)" }), g, w, log));
            if !same {
                log(format!("  argmax differs: ours {} theirs {}", range(g), range(w)));
            }
        }
        log(format!("ar: argmax agrees on {agree} of {} rows ({} tokens teacher-forced in {:.1} s)", rows.len(), seq.len(), t0.elapsed().as_secs_f64()));
    }
    if stages.contains(&"neg") {
        if let Ok(np) = ids("neg_prefix") {
            let (_, want) = get("neg_logits")?;
            let mut c = e.lm.cache(np.len())?;
            let x = e.lm.prefill(&mut s, &mut c, &np, &[np.len() - 1])?;
            let got = e.lm.logits_of(&mut s, &x, 0, VOCAB)?;
            worst = worst.min(note("neg prefill logits", &got, &want, log));
        } else {
            log(format!("neg: no twin (cfg {cfg})"));
        }
    }
    let (_, noise) = get("noise")?;
    if stages.contains(&"noise") {
        let ours = Mt19937::new(seed).randn(codes.len() * LATENT);
        worst = worst.min(note("noise (torch.randn)", &ours, &noise, log));
    }
    let flow = ["vel", "steps", "solve"].iter().any(|st| stages.contains(st));
    if flow {
        let mut ids = sem_prefix.clone();
        ids.extend(codes.iter().map(|c| c + CODEC_OFFSET));
        ids.push(MUSIC_END);
        let frames = codes.len();
        let mut c = e.lm.cache(ids.len() + frames + 2)?;
        e.lm.prefill(&mut s, &mut c, &ids, &[])?;
        let g = &e.lm.gpu;
        let pos = e.lm.positions(frames + 2)?;
        let v = DevBuf::f32(g, frames * LATENT)?;
        let temb = |raw: f64| DevBuf::from_f32(g, &e.lm.time_embedding(raw));
        let logit = |t: f64| (t / (1.0 - t)).ln().clamp(-20.0, 20.0);
        if stages.contains(&"vel") {
            let st = DevBuf::from_f32(g, &noise)?;
            e.lm.velocity(&mut s, &c, &st, frames, &temb(20.0)?, &pos, v.ptr())?;
            worst = worst.min(note("velocity at t = 1", &v.to_f32()?, &get("vel0")?.1, log));
        }
        if stages.contains(&"steps") {
            let (_, states) = get("states")?;
            let (_, mids) = get("mids")?;
            let n = frames * LATENT;
            let dt = 1.0 / steps as f64;
            for i in 0..steps {
                let t = 1.0 - i as f64 * dt;
                let st = DevBuf::from_f32(g, &states[i * n..(i + 1) * n])?;
                e.lm.velocity(&mut s, &c, &st, frames, &temb(logit(t))?, &pos, v.ptr())?;
                let mid = DevBuf::f32(g, n)?;
                e.lm.axpy(mid.ptr(), st.ptr(), v.ptr(), n, -(dt / 2.0) as f32)?;
                worst = worst.min(note(&format!("step {i}: midpoint"), &mid.to_f32()?, &mids[i * n..(i + 1) * n], log));
                let m = DevBuf::from_f32(g, &mids[i * n..(i + 1) * n])?;
                e.lm.velocity(&mut s, &c, &m, frames, &temb(logit(t - dt / 2.0))?, &pos, v.ptr())?;
                e.lm.axpy(st.ptr(), st.ptr(), v.ptr(), n, -dt as f32)?;
                worst = worst.min(note(&format!("step {i}: state"), &st.to_f32()?, &states[(i + 1) * n..(i + 2) * n], log));
            }
        }
        if stages.contains(&"solve") {
            let t0 = Instant::now();
            let lat = e.solve(&mut s, &c, &noise, steps, |_, _, _| Ok(()))?;
            worst = worst.min(note(&format!("solve ({steps} steps, {:.1} s)", t0.elapsed().as_secs_f64()), &lat, &get("latents")?.1, log));
        }
    }
    drop(s);
    if stages.contains(&"vae") {
        let (_, lat) = get("latents")?;
        let (_, want) = get("audio")?;
        let t0 = Instant::now();
        let wav = e.sound(&lat, t0, &mut |_| Ok(()))?;
        let n = wav.len() / 2;
        let got: Vec<f32> = (0..n).flat_map(|i| [wav[i], wav[n + i]]).collect();
        worst = worst.min(note(&format!("vae ({:.2} s of audio, {:.1} s)", n as f64 / 48000.0, t0.elapsed().as_secs_f64()), &got, &want, log));
        let a = nextsycl_audio::Audio { rate: crate::RATE, channels: 2, samples: got };
        a.write_wav(&dir.join("ours.wav"), &[]).map_err(Error)?;
    }
    Ok(worst)
}
