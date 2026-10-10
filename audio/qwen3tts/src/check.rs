//! Stage by stage against the reference (reference/qwen3tts/ref.py: qwen-tts' own code in float32 on the same files),
//! each stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   prompt  the request (request.json)  -> prefill.npy (the talker's input rows), logits0.npy (the first frame's)
//!   frames  greedy, from the prompt     -> codes.npy [T, 16] (a --greedy dump): how many frames agree
//!   codec   codes.npy                   -> speech.npy / speech.wav (24 kHz)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::prompt::{Spec, Voice};
use crate::sample::Draw;
use crate::Qwen3Tts;

/// A little-endian .npy of float32, int32 or int64: (shape, values as f64)
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
        return Err(Error(format!("{}: {head} (float32, int32 or int64 expected)", path.display())));
    };
    Ok((shape, v))
}

/// A 16-bit or float WAV's mono samples (the first channel)
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    nextsycl_audio::decode_wav(&b).map(|(s, _)| s).map_err(|e| Error(format!("{}: {e}", path.display())))
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
        log(format!("{what:<24} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<24} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

/// The dump's request (request.json: mode, text, speaker, language, instruct), as ref.py writes it
pub fn spec_of(dir: &Path) -> Result<Spec> {
    let p = dir.join("request.json");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
        .map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let s = |k: &str| r[k].as_str().filter(|v| !v.is_empty()).map(str::to_string);
    let voice = match r["mode"].as_str() {
        Some("custom") => Voice::Speaker(s("speaker").ok_or_else(|| Error("request.json: no speaker".into()))?),
        Some("design") => Voice::Described,
        Some("clone") => {
            let (_, v) = read_npy(&dir.join("spk.npy"))?;
            Voice::XVector(v.iter().map(|x| *x as f32).collect())
        }
        m => return Err(Error(format!("request.json: mode {m:?}"))),
    };
    let streaming = r["non_streaming_mode"].as_bool().map(|n| !n).unwrap_or(matches!(voice, Voice::XVector(_)));
    Ok(Spec { text: s("text").unwrap_or_default(), language: s("language"), instruct: s("instruct"), voice, streaming })
}

pub fn run(e: &Qwen3Tts, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(read_npy(&dir.join(n))?.1.into_iter().map(|x| x as f32).collect()) };
    let t0 = Instant::now();
    let codes_ref: Option<Vec<Vec<i32>>> = match read_npy(&dir.join("codes.npy")) {
        Ok((shape, v)) => Some(v.chunks(shape.get(1).copied().unwrap_or(16)).map(|r| r.iter().map(|x| *x as i32).collect()).collect()),
        Err(_) => None,
    };
    if stages.contains(&"prompt") || stages.contains(&"frames") {
        let spec = spec_of(dir)?;
        if let Voice::XVector(x) = &spec.voice {
            // the recording's embedding here too, against the reference's
            if let (Some(sp), Ok(wav)) = (&e.speaker, read_wav(&dir.join("ref.wav"))) {
                let got = sp.embed(&crate::speaker::resample(&wav, crate::speaker::RATE))?;
                report(log, &mut worst, "speaker embedding", &got, x);
            }
        }
        let mut d = e.drawing(&Default::default())?;
        d.talker = Draw::GREEDY;
        d.predictor = Draw::GREEDY;
        d.repetition_penalty = 1.0;
        let max = codes_ref.as_ref().map_or(200, |c| c.len() + 8);
        let mut keep = Vec::new();
        let greedy = stages.contains(&"frames");
        let got = e.frames(&spec, &d, if greedy { max } else { 1 }, 0, t0, &mut |_| Ok(()), None, Some(&mut keep))?;
        report(log, &mut worst, "prompt rows", &keep[0], &want("prefill.npy")?);
        report(log, &mut worst, "first logits", &keep[1], &want("logits0.npy")?);
        if greedy {
            if let Some(r) = &codes_ref {
                let first = got.iter().zip(r).position(|(a, b)| a != b);
                let same = got.iter().zip(r).filter(|(a, b)| a == b).count();
                let firsts = got.iter().zip(r).filter(|(a, b)| a[0] == b[0]).count();
                log(format!("{:<24} {} here, {} in the reference; {same} identical, {firsts} first codes agree; first difference at {}", "frames (greedy)",
                            got.len(), r.len(), first.map_or("none".into(), |f| f.to_string())));
            }
        }
    }
    if stages.contains(&"codec") {
        let codes = codes_ref.ok_or_else(|| Error(format!("{}: no codes.npy", dir.display())))?;
        let t = Instant::now();
        let got = e.codec.decode(&e.ops, &e.nsd, &codes)?;
        log(format!("codec: {} frames to {:.2} s in {:.2} s", codes.len(), got.len() as f64 / crate::codec::RATE as f64, t.elapsed().as_secs_f64()));
        let wav = match read_npy(&dir.join("speech.npy")) {
            Ok((_, v)) => v.into_iter().map(|x| x as f32).collect(),
            Err(_) => read_wav(&dir.join("speech.wav"))?,
        };
        report(log, &mut worst, "speech", &got, &wav);
    }
    Ok(worst)
}
