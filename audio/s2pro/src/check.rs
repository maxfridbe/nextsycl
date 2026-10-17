//! Stage by stage against the reference (reference/s2pro/ref.py: fish-speech's own code in float32 on the checkpoint),
//! each stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   prompt  request.json (+ ref.wav, ref_codes.npy) -> the prompt's rows (ids.npy [11, L]: exact), the recording's codes
//!           (the codec's encoder here, against ref_codes.npy), prefill.npy (the slow model's input rows)
//!   frames  codes.npy forced (teacher forcing)       -> each frame's slow logits (logits.npy [T, 4097]) and fast logits
//!           (fast.npy [T, 9, 4096]), and how many codes the model here would have chosen otherwise; then a greedy run
//!           from the prompt alone: how many frames agree
//!   codec   codes.npy                                -> codec_z.npy (the quantizer's decode, upsampled), speech.npy;
//!           decoded_half.npy when there (the reference's codec with the GGUF's half weights)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::prompt::Spec;
use crate::sample::Draw;
use crate::{Keep, S2Pro};

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

fn frames_of(shape: &[usize], v: &[f64]) -> Vec<Vec<i32>> {
    v.chunks(shape.get(1).copied().unwrap_or(10).max(1)).map(|f| f.iter().map(|x| *x as i32).collect()).collect()
}

/// The dump's request (request.json, ref.py's) as a spec, the recording's frames the reference's own
pub fn spec_of(dir: &Path) -> Result<Spec> {
    let p = dir.join("request.json");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
        .map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let s = |k: &str| r[k].as_str().filter(|v| !v.is_empty()).map(str::to_string);
    let reference = match s("ref_text") {
        Some(t) => {
            let (shape, v) = read_npy(&dir.join("ref_codes.npy"))?;
            Some((t, frames_of(&shape, &v)))
        }
        None => None,
    };
    Ok(Spec { text: s("text").unwrap_or_default(), instruction: None, reference })
}

/// The index of the largest value
fn best(l: &[f32]) -> usize {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0
}

pub fn run(e: &S2Pro, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let want = |n: &str| -> Result<Vec<f32>> { Ok(read_npy(&dir.join(n))?.1.into_iter().map(|x| x as f32).collect()) };
    let spec = spec_of(dir)?;
    log(format!("{}: {:?}{}", if spec.reference.is_some() { "clone" } else { "plain" }, spec.text,
                spec.reference.as_ref().map_or(String::new(), |(t, f)| format!(", a recording of {} frames saying {t:?}", f.len()))));
    let codes_ref: Option<Vec<Vec<i32>>> = read_npy(&dir.join("codes.npy")).ok().map(|(s, v)| frames_of(&s, &v));
    let t0 = Instant::now();
    if stages.contains(&"prompt") {
        let mut rows = crate::prompt::system(&e.tok, &e.ids, spec.reference.as_ref());
        for u in spec.batches() {
            rows.extend(crate::prompt::user(&e.tok, &e.ids, &u));
        }
        let (shape, w) = read_npy(&dir.join("ids.npy"))?;
        let l = shape.get(1).copied().unwrap_or(0);
        let mut got = vec![0i64; (e.groups + 1) * rows.len()];
        for (j, r) in rows.iter().enumerate() {
            got[j] = r.token as i64;
            for (i, c) in r.codes.iter().flatten().enumerate() {
                got[(i + 1) * rows.len() + j] = *c as i64;
            }
        }
        let w: Vec<i64> = w.into_iter().map(|x| x as i64).collect();
        let first = (0..rows.len().min(l)).find(|j| (0..=e.groups).any(|i| got[i * rows.len() + j] != w[i * l + j]));
        log(format!("{:<24} {} here, {l} in the reference; {}", "prompt rows (ids, codes)", rows.len(),
                    if got == w { "identical".to_string() } else { format!("first difference at {first:?}") }));
        if got != w {
            worst = f64::INFINITY;
        }
        if let (Some((_, rc)), Ok(wav)) = (&spec.reference, std::fs::read(dir.join("ref.wav"))) {
            // the recording (ref.py wrote it at 44.1 kHz) through the codec's encoder here, against the reference's codes
            let (samples, rate) = nextsycl_audio::decode_audio(&wav).map_err(Error)?;
            let t = Instant::now();
            let got = e.codec.encode(&e.nsd, &crate::resample(&samples, rate, crate::codec::RATE))?;
            let same: usize = got.iter().zip(rc).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x == y).count()).sum();
            let first: usize = got.iter().zip(rc).filter(|(a, b)| a[0] == b[0]).count();
            log(format!("{:<24} {} frames here, {} in the reference; {same} of {} codes agree ({first} of the semantic ones), {:.2} s", "recording's codes",
                        got.len(), rc.len(), rc.len() * e.groups, t.elapsed().as_secs_f64()));
        }
        let x = e.embed_rows(&rows)?;
        report(log, &mut worst, "prompt input rows", &x, &want("prefill.npy")?);
    }
    let greedy = Draw::GREEDY;
    if stages.contains(&"frames") {
        let r = codes_ref.as_ref().ok_or_else(|| Error(format!("{}: no codes.npy", dir.display())))?;
        let mut k = Keep::default();
        let t = Instant::now();
        e.frames(&spec, greedy, r.len() + 1, 0, t0, &mut |_| Ok(()), Some(r), Some(&mut k))?;
        log(format!("{:<24} {} frames in {:.1} s", "frames (forced)", r.len(), t.elapsed().as_secs_f64()));
        let wl = want("logits.npy")?;
        let wf = want("fast.npy")?;
        let w0 = e.codes + 1;
        let n = k.logits.len().min(wl.len() / w0);
        let got: Vec<f32> = k.logits[..n].concat();
        report(log, &mut worst, "slow logits", &got, &wl[..got.len()]);
        let nf = k.fast.len().min(wf.len() / ((e.groups - 1) * e.codes));
        let gf: Vec<f32> = k.fast.iter().take(nf).flatten().flatten().copied().collect();
        report(log, &mut worst, "fast logits", &gf, &wf[..gf.len()]);
        // where the most likely code here differs from the reference's choice (a greedy dump: its own argmax)
        let (mut m0, mut mr) = (0, 0);
        for (f, fr) in r.iter().enumerate().take(n.min(nf)) {
            if best(&k.logits[f]) as i32 != fr[0] {
                m0 += 1;
            }
            for (l, c) in k.fast[f].iter().zip(&fr[1..]) {
                if best(l) as i32 != *c {
                    mr += 1;
                }
            }
        }
        log(format!("{:<24} first codes {m0} of {} frames differ, the other codes {mr} of {}", "argmax vs reference", n.min(nf), n.min(nf) * (e.groups - 1)));
        let got = e.frames(&spec, greedy, r.len(), 0, t0, &mut |_| Ok(()), None, None)?;
        let first = got.iter().zip(r).position(|(a, b)| a != b);
        let same = got.iter().zip(r).filter(|(a, b)| a == b).count();
        log(format!("{:<24} {} here, {} in the reference; {same} identical; first difference at {}", "frames (greedy)", got.len(), r.len(),
                    first.map_or("none".into(), |f| f.to_string())));
    }
    if stages.contains(&"codec") {
        let codes = codes_ref.ok_or_else(|| Error(format!("{}: no codes.npy", dir.display())))?;
        let t = Instant::now();
        let (z, l) = e.codec.latents(&e.nsd, &codes)?;
        e.ops.gpu.sync()?;
        let z = z.to_f32()?;
        report(log, &mut worst, "codec latents", &z[..1024 * l], &want("codec_z.npy")?);
        let got = e.codec.decode(&e.nsd, &codes)?;
        log(format!("codec: {} frames to {:.2} s in {:.2} s", codes.len(), got.len() as f64 / crate::codec::RATE as f64, t.elapsed().as_secs_f64()));
        report(log, &mut worst, "speech", &got, &want("speech.npy")?);
        // the reference's codec with its weights rounded to half, as the GGUF holds them (ref.py --decode --codec-half):
        // what is left is this port's
        if let Ok(h) = want("decoded_half.npy") {
            report(log, &mut worst, "speech (half weights)", &got, &h);
        }
    }
    Ok(worst)
}
