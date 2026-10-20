//! Stage by stage against the reference (reference/whisper/ref.py: transformers' WhisperForConditionalGeneration in
//! float32 on the CPU and openai-whisper's frontend and transcribe, on the same files), each from the reference's own
//! input.
//!
//! ```text
//!   mel         audio16k.npy                         -> mel.npy [n_mels, 3000] (the first window, as transcribe pads it)
//!   encoder     mel.npy                              -> enc.npy [1500, d]
//!   logits      tokens.npy forced (teacher forcing)  -> logits.npy [T, vocab] (and the greedy choice at each row)
//!   transcribe  audio16k.npy                         -> result.json (openai-whisper's transcribe, word_timestamps,
//!                                                       greedy, condition_on_previous_text off): text, segments, words
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result, TranscribeRequest};
use serde_json::Value;

use crate::mel;
use crate::Whisper;

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

/// Word error rate of `got` against `want` (words lowercased, punctuation dropped): edits / reference words
pub fn wer(got: &str, want: &str) -> f64 {
    let norm = |s: &str| -> Vec<String> {
        s.split_whitespace().map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>().to_lowercase()).filter(|w| !w.is_empty()).collect()
    };
    let (g, w) = (norm(got), norm(want));
    if w.is_empty() {
        return if g.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=g.len()).collect();
    for (i, wi) in w.iter().enumerate() {
        let mut cur = vec![i + 1; g.len() + 1];
        for (j, gj) in g.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(wi != gj)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[g.len()] as f64 / w.len() as f64
}

pub fn run(e: &Whisper, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(f32s(read_npy(&dir.join(n))?.1)) };
    let c = &e.model.cfg;
    let audio = want("audio16k.npy")?;
    if stages.contains(&"mel") {
        let t = Instant::now();
        let (spec, content) = mel::log_mel(&audio, c.n_mels);
        let frames = spec.len() / c.n_mels;
        let w0 = mel::window(&spec, c.n_mels, frames, 0, content.min(mel::CHUNK_FRAMES));
        log(format!("mel: {:.2} s of recording in {:.2} s", audio.len() as f64 / 16000.0, t.elapsed().as_secs_f64()));
        report(log, &mut worst, "mel (first window)", &w0, &want("mel.npy")?);
    }
    let rmel = want("mel.npy")?;
    if stages.contains(&"encoder") || stages.contains(&"logits") {
        let t = Instant::now();
        e.model.encode(&rmel)?;
        e.model.nsd.wait()?;
        log(format!("encoder: one window in {:.3} s", t.elapsed().as_secs_f64()));
        if stages.contains(&"encoder") {
            report(log, &mut worst, "encoder output", &e.model.encoder_output()?, &want("enc.npy")?);
        }
    }
    if stages.contains(&"logits") {
        let toks: Vec<u32> = read_npy(&dir.join("tokens.npy"))?.1.iter().map(|x| *x as u32).collect();
        let wl = want("logits.npy")?;
        let vs = c.vocab;
        let rows = wl.len() / vs;
        let t = Instant::now();
        let got = e.model.decode(&toks[..rows], 0, 0, None)?;
        log(format!("decoder: {rows} rows teacher-forced in {:.3} s", t.elapsed().as_secs_f64()));
        report(log, &mut worst, "logits (all rows)", &got, &wl);
        let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::MIN), |m, (i, x)| if *x > m.1 { (i, *x) } else { m }).0;
        let same = got.chunks(vs).zip(wl.chunks(vs)).filter(|(a, b)| am(a) == am(b)).count();
        log(format!("{:<24} {same} of {rows} rows", "same greedy token"));
        // the step-by-step path: the cache, one token at a time, against the same rows
        let p = 3.min(rows);
        let mut steps = e.model.decode(&toks[..p], 0, 0, None)?;
        for (i, t) in toks[p..rows].iter().enumerate() {
            steps.extend(e.model.decode(&[*t], p + i, 0, None)?);
        }
        report(log, &mut worst, "logits (a token a step)", &steps, &wl);
    }
    if stages.contains(&"transcribe") {
        let r: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("result.json")).map_err(|e| Error(e.to_string()))?)
            .map_err(|e| Error(e.to_string()))?;
        // temperature 0 only, as the reference ran
        let extra = [("fallback".to_string(), "0".to_string())].into_iter().collect();
        let req = TranscribeRequest { samples: audio.clone(), rate: 16000, language: r["asked_language"].as_str().map(str::to_string), words: true, extra,
                                      ..Default::default() };
        let t = Instant::now();
        let got = e.run(&req, &mut |_| Ok(()))?;
        let secs = t.elapsed().as_secs_f64();
        log(format!("transcribe: {:.1} s of recording in {secs:.2} s ({:.0}x), language {} (reference {})", got.duration, got.duration / secs.max(1e-9),
                    got.language, r["language"].as_str().unwrap_or("?")));
        let rt = r["text"].as_str().unwrap_or("");
        let w = wer(&got.text, rt);
        log(format!("{:<24} {:.1}% ({} words in the reference)", "word error rate", 100.0 * w, rt.split_whitespace().count()));
        if w > 0.0 {
            log(format!("  here {}\n  ref  {}", got.text, rt.trim()));
        }
        let rsegs = r["segments"].as_array().cloned().unwrap_or_default();
        log(format!("{:<24} {} here, {} in the reference", "segments", got.segments.len(), rsegs.len()));
        for (g, w) in got.segments.iter().zip(&rsegs).take(6) {
            log(format!("  {:7.2}-{:7.2} {:?}\n  {:7.2}-{:7.2} {:?}", g.start, g.end, g.text, w["start"].as_f64().unwrap_or(0.0), w["end"].as_f64().unwrap_or(0.0),
                        w["text"].as_str().unwrap_or("").trim()));
        }
        // the words' times, where the words are the same
        let rw: Vec<&Value> = rsegs.iter().flat_map(|s| s["words"].as_array().into_iter().flatten()).collect();
        let pairs: Vec<(f64, f64)> = got.words.iter().zip(&rw).filter(|(g, w)| g.text == w["word"].as_str().unwrap_or("").trim())
            .flat_map(|(g, w)| [(g.start, w["start"].as_f64().unwrap_or(0.0)), (g.end, w["end"].as_f64().unwrap_or(0.0))]).collect();
        if !pairs.is_empty() {
            let mut d: Vec<f64> = pairs.iter().map(|(a, b)| (a - b).abs()).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            log(format!("{:<24} {} of {} words in step; their times off by median {:.3} s, 90% {:.3} s, max {:.3} s", "words", pairs.len() / 2, rw.len(),
                        d[d.len() / 2], d[(d.len() * 9 / 10).min(d.len() - 1)], d[d.len() - 1]));
        } else {
            log(format!("{:<24} {} here, {} in the reference, none in step", "words", got.words.len(), rw.len()));
        }
        worst = worst.max(w);
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    #[test]
    fn word_error_rate() {
        assert_eq!(super::wer("Hello, world!", "hello world"), 0.0);
        assert!((super::wer("hello there world", "hello world") - 0.5).abs() < 1e-12);
        assert!((super::wer("hello", "hello big world") - 2.0 / 3.0).abs() < 1e-12);
    }
}
