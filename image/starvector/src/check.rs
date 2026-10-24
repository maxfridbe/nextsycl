//! Stage by stage against the reference (reference/starvector/ref.py: the starvector package's own code in float32 on
//! the CPU, on the same checkpoint), each stage from the reference's own input.
//!
//! ```text
//!   pixels    the picture                    -> pixels.npy [3, 224, 224] (PIL's flatten, pad, bicubic; normalised)
//!   encoder   pixels.npy                     -> enc.npy [257, 1024], adapter.npy [257, 2048]
//!   prefill   adapter.npy + the prompt       -> logits0.npy [vocab] (and prompt.npy, stop.npy: the tokenizer's)
//!   logits    tokens.npy forced              -> logits.npy [rows, vocab] (and the greedy choice at each row)
//!   greedy    the picture, beams 1, argmax   -> tokens.npy / greedy.txt: tokens in step, the text
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_image::{Error, Picture, Result, VectorizeRequest};

use crate::{model, pre, StarVector};

/// A little-endian .npy of float32 or int64 (C order): (shape, values as f64)
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

/// (relative error, worst absolute difference, cosine)
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
        log(format!("{what:<14} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<14} relative error {rel:.2e}, worst {mx:.3e}, cos {cos:.6}"));
    *worst = worst.max(rel);
}

fn argmax(r: &[f32]) -> usize {
    let mut b = 0;
    for (i, &v) in r.iter().enumerate() {
        if v > r[b] {
            b = i;
        }
    }
    b
}

/// Every stage named in `stages` against the dumps in `dir` (`picture`: the one the reference read); the worst
/// relative error of the tensor stages
pub fn run(e: &StarVector, dir: &Path, picture: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |name: &str| -> Result<Vec<f32>> { Ok(f32s(read_npy(&dir.join(name))?.1)) };
    let pic = Picture::read_png(picture).map_err(Error)?;
    if stages.contains(&"pixels") {
        let got = pre::pixels(&pic);
        report(log, &mut worst, "pixels", &got, &want("pixels.npy")?);
    }
    let mut m = e.model.lock().unwrap_or_else(|p| p.into_inner());
    let (want_prompt, want_stop) = (read_npy(&dir.join("prompt.npy"))?.1, read_npy(&dir.join("stop.npy"))?.1);
    let same = |a: &[u32], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| *x as f64 == *y);
    log(format!("tokenizer      prompt {:?} {}, stop {:?} {}", e.vocab.prompt, if same(&e.vocab.prompt, &want_prompt) { "as the reference" } else { "DIFFERS" },
                e.vocab.stop, if same(&e.vocab.stop, &want_stop) { "as the reference" } else { "DIFFERS" }));
    if !same(&e.vocab.prompt, &want_prompt) || !same(&e.vocab.stop, &want_stop) {
        worst = f64::INFINITY;
    }
    let adapter_ref = want("adapter.npy")?;
    if stages.contains(&"encoder") {
        let t = Instant::now();
        let (enc, adapter) = m.encode(&want("pixels.npy")?)?;
        log(format!("encoder        {:.2} s", t.elapsed().as_secs_f64()));
        report(log, &mut worst, "encoder", &enc, &want("enc.npy")?);
        report(log, &mut worst, "adapter", &adapter, &adapter_ref);
    }
    let tokens: Vec<u32> = read_npy(&dir.join("tokens.npy"))?.1.into_iter().map(|t| t as u32).collect();
    if stages.contains(&"prefill") || stages.contains(&"logits") {
        let mut x = adapter_ref.clone();
        for &t in &e.vocab.prompt {
            x.extend(m.token_row(t));
        }
        let len = x.len() / model::D;
        let t = Instant::now();
        let l0 = m.prefill(&x, 1)?;
        log(format!("prefill        {len} rows in {:.3} s", t.elapsed().as_secs_f64()));
        let w0 = want("logits0.npy")?;
        report(log, &mut worst, "prefill", &l0, &w0);
        log(format!("               argmax {} here, {} in the reference", argmax(&l0), argmax(&w0)));
        if stages.contains(&"logits") {
            let (shape, wl) = read_npy(&dir.join("logits.npy"))?;
            let (rows, v) = (shape[0], shape[1]);
            let wl = f32s(wl);
            let mut got = l0;
            let mut all = Vec::new();
            let mut agree = 0;
            let t = Instant::now();
            for r in 0..rows {
                if argmax(&got) == tokens[r] as usize {
                    agree += 1;
                }
                all.extend_from_slice(&got);
                if r + 1 < rows {
                    got = m.step(&[0], &[tokens[r]], len + r)?;
                }
            }
            log(format!("logits         {rows} rows teacher-forced in {:.2} s: the argmax is the reference's token at {agree} of {rows}", t.elapsed().as_secs_f64()));
            report(log, &mut worst, "logits", &all, &wl[..rows * v]);
        }
    }
    if stages.contains(&"force") {
        // every reference token forced: where our argmax differs, how near our top two were (a near-tie in half
        // against float32, or a real difference)
        let mut x = adapter_ref.clone();
        for &t in &e.vocab.prompt {
            x.extend(m.token_row(t));
        }
        let len = x.len() / model::D;
        let mut got = m.prefill(&x, 1)?;
        let mut differ = Vec::new();
        for (r, &want_t) in tokens.iter().enumerate() {
            let a = argmax(&got);
            if a != want_t as usize {
                differ.push((r, got[a] - got[want_t as usize]));
            }
            if r + 1 < tokens.len() {
                got = m.step(&[0], &[want_t], len + r)?;
            }
        }
        let worst_gap = differ.iter().map(|d| d.1).fold(0f32, f32::max);
        log(format!("force          {} tokens forced: our argmax differs at {} row(s){}{}", tokens.len(), differ.len(),
                    if differ.is_empty() { String::new() } else { format!(", by {worst_gap:.4} at most (our top logit over the reference's token)") },
                    differ.iter().take(8).map(|(r, g)| format!(" [row {r}: {g:.4}]")).collect::<String>()));
    }
    drop(m);
    if stages.contains(&"greedy") {
        let want_text = std::fs::read_to_string(dir.join("greedy.txt")).unwrap_or_default();
        let req = VectorizeRequest { picture: pic, max_length: (model::QUERY + e.vocab.prompt.len() + tokens.len()) as u32, beams: Some(1), sample: Some(false),
                                     ..Default::default() };
        let t = Instant::now();
        let out = nextsycl_image::ImageEngine::vectorize(e, &req, &mut |_| Ok(()))?;
        let secs = t.elapsed().as_secs_f64();
        let common = out.svg.chars().zip(want_text.chars()).take_while(|(a, b)| a == b).count();
        log(format!("greedy         {} tokens in {secs:.2} s ({:.1} tokens/s): the text {} ({} of {} characters in step{})", out.tokens,
                    out.tokens as f64 / out.seconds.max(1e-9), if out.svg == want_text { "is the reference's" } else { "DIFFERS" }, common,
                    want_text.chars().count(), if out.closed { "; ended by </svg>" } else { "" }));
    }
    Ok(worst)
}
