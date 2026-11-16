//! Stage by stage against the reference (reference/sophon/ref.py: the release's own audio code and the language
//! model's maths in float32 on the same files), each stage from the reference's own input so a difference shows where
//! it is born.
//!
//! ```text
//!   mel     audio16k.npy                 -> mel.npy [128, L]
//!   ear     mel.npy                      -> tower.npy [n, 896] (the encoder), audio_emb.npy [tokens, 5120] (the projector)
//!   lm      ids.npy + audio_emb.npy      -> h{l}.npy (the residual after layer l), logits.npy (the last row), argmax.npy
//!                                           (every row's: a teacher-forced comparison of a --follow dump)
//!   check   (a turn-question dump)       -> letters.json's P(finished)
//!   write   greedy from the prompt       -> follow.npy, follow.txt here (for ref.py --follow)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::brain::{argmax, Rows};
use crate::{ear, Sophon};

/// A little-endian .npy of float32, float64, int32 or int64: (shape, values as f64)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error(format!("{}: not a .npy", path.display())));
    }
    let (hl, at) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = String::from_utf8_lossy(&b[at..at + hl]).to_string();
    let shape: Vec<usize> = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next())
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    let d = &b[at + hl..];
    let v: Vec<f64> = if head.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<f8'") {
        d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head} (float or int expected)", path.display())));
    };
    if head.contains("'fortran_order': True") {
        return Err(Error(format!("{}: a Fortran-order array", path.display())));
    }
    Ok((shape, v))
}

fn f32s(path: &Path) -> Result<Vec<f32>> {
    Ok(read_npy(path)?.1.into_iter().map(|x| x as f32).collect())
}

/// (relative RMS error, max abs error, cosine)
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
        log(format!("{what:<26} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<26} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

/// The stages asked for (none: all) against the dump in `dir`; the worst relative error
pub fn run(e: &Sophon, dir: &Path, stages: &[String], log: &mut dyn FnMut(String)) -> Result<f64> {
    let want = |s: &str| stages.is_empty() || stages.iter().any(|x| x == s);
    let mut worst = 0f64;
    let mut m = e.inner.mind.lock().unwrap_or_else(|x| x.into_inner());
    let req: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("request.json")).map_err(|x| Error(x.to_string()))?)
        .map_err(|x| Error(format!("request.json: {x}")))?;
    if want("bench") && !stages.is_empty() {
        let mind = &mut *m;
        mind.brain.bench(&mind.check, log)?;
    }
    let mel_ref = f32s(&dir.join("mel.npy"))?;
    let l = mel_ref.len() / 128;
    if want("mel") {
        let a = f32s(&dir.join("audio16k.npy"))?;
        let x = ear::align(&a);
        let mel = ear::log_mel(&x);
        report(log, &mut worst, "mel (Whisper features)", &mel, &mel_ref);
    }
    let emb_ref = f32s(&dir.join("audio_emb.npy"))?;
    if want("ear") {
        let t = Instant::now();
        let (tower, proj) = m.ear.encode(&mel_ref, l)?;
        log(format!("ear: {l} mel frames in {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
        report(log, &mut worst, "ear: encoder", &tower, &f32s(&dir.join("tower.npy"))?);
        report(log, &mut worst, "ear: projected rows", &proj, &emb_ref);
    }
    if want("lm") || want("check") || want("write") {
        let ids: Vec<u32> = read_npy(&dir.join("ids.npy"))?.1.iter().map(|x| *x as u32).collect();
        let n_prompt = req["n_prompt"].as_u64().unwrap_or(ids.len() as u64) as usize;
        let at = ids.iter().position(|t| *t == 151669).ok_or_else(|| Error("ids.npy has no audio".into()))?;
        let keys: Vec<u64> = ids.iter().enumerate().map(|(i, t)| if *t == 151669 { (1u64 << 62) | (1 << 40) | i as u64 } else { *t as u64 }).collect();
        let spans = [(at, emb_ref.as_slice())];
        let layers: Vec<usize> = (0..64).filter(|li| dir.join(format!("h{li}.npy")).exists()).collect();
        let mind = &mut *m;
        mind.reply.keys.clear();
        mind.brain.taps.lock().unwrap_or_else(|x| x.into_inner()).0 = layers.clone();
        let t = Instant::now();
        let (logits, am, _) = mind.brain.prompt(&mut mind.reply, &keys, &Rows { ids: &ids, spans: &spans }, true)?;
        let secs = t.elapsed().as_secs_f64();
        log(format!("lm: {} positions in {secs:.2} s (and every row's logits, for the check)", ids.len()));
        let taps = std::mem::take(&mut mind.brain.taps.lock().unwrap_or_else(|x| x.into_inner()).1);
        mind.brain.taps.lock().unwrap_or_else(|x| x.into_inner()).0.clear();
        if want("lm") {
            for (li, x) in &taps {
                report(log, &mut worst, &format!("lm: after layer {li}"), x, &f32s(&dir.join(format!("h{li}.npy")))?);
            }
            // the last prompt row's logits, and the argmax rows
            let am_ref: Vec<u32> = read_npy(&dir.join("argmax.npy"))?.1.iter().map(|x| *x as u32).collect();
            let last_ref = f32s(&dir.join("logits.npy"))?;
            // the prompt's last row: the logits computed with every row's (the last of all rows when no --follow)
            let lp = if n_prompt == ids.len() { logits.clone() } else { Vec::new() };
            if !lp.is_empty() {
                report(log, &mut worst, "lm: logits (last row)", &lp, &last_ref);
                log(format!("lm: next token here {} (reference {})", argmax(&lp), argmax(&last_ref)));
            }
            let agree = am.iter().zip(&am_ref).filter(|(a, b)| a == b).count();
            log(format!("lm: every row's argmax agrees on {agree} of {}", am_ref.len()));
            if n_prompt < ids.len() {
                let f = &am[n_prompt - 1..ids.len() - 1];
                let fr = &ids[n_prompt..];
                let same = f.iter().zip(fr).filter(|(a, b)| a == b).count();
                log(format!("lm: teacher-forced: {same} of the {} followed tokens are this engine's argmax too", fr.len()));
            }
        }
        if want("check") && dir.join("letters.json").exists() {
            let lj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("letters.json")).map_err(|x| Error(x.to_string()))?)
                .map_err(|x| Error(x.to_string()))?;
            let p = crate::duplex::letter_probs(&e.inner, &logits);
            let lang = req["lang"].as_str().unwrap_or("en");
            let (_, fin, back) = crate::text::turn_question(lang, "");
            log(format!("check: P(finished) {:.4} (reference {:.4}), P(backchannel) {:.4} (reference {:.4})", p[fin], lj["p_finished"].as_f64().unwrap_or(-1.0),
                        p[back], lj["p_backchannel"].as_f64().unwrap_or(-1.0)));
        }
        if want("write") && n_prompt == ids.len() {
            // greedy from the prompt: what this engine writes (follow.npy for ref.py --follow)
            let mut out = Vec::new();
            let mut lg = logits;
            let t = Instant::now();
            for _ in 0..48 {
                let tk = argmax(&lg) as u32;
                if tk == 151645 || tk == 151643 {
                    break;
                }
                out.push(tk);
                lg = mind.brain.step(&mut mind.reply, tk as u64, tk)?;
            }
            let secs = t.elapsed().as_secs_f64();
            let text = e.inner.tok.decode(&out);
            log(format!("write: {} tokens in {secs:.2} s (tg {:.1} tok/s): {text:?}", out.len(), out.len() as f64 / secs.max(1e-6)));
            let mut npy = b"\x93NUMPY\x01\x00".to_vec();
            let mut head = format!("{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}", out.len());
            while (head.len() + 11) % 64 != 0 {
                head.push(' ');
            }
            head.push('\n');
            npy.extend_from_slice(&(head.len() as u16).to_le_bytes());
            npy.extend_from_slice(head.as_bytes());
            for t in &out {
                npy.extend_from_slice(&(*t as i32).to_le_bytes());
            }
            std::fs::write(dir.join("follow.npy"), npy).map_err(|x| Error(x.to_string()))?;
            std::fs::write(dir.join("follow.txt"), &text).map_err(|x| Error(x.to_string()))?;
        }
    }
    Ok(worst)
}
