//! Stage by stage against the reference (reference/voxtraltts/ref.py: upstream's own code in float32 on the
//! checkpoint), each stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   prompt  request.json                       -> the token ids (ids.npy: exact) and the backbone's input rows
//!                                                 (prefill.npy)
//!   frames  codes.npy + noise.npy forced        -> each frame's hidden state (hidden.npy), semantic logits (sem.npy)
//!           (teacher forcing)                     and flow result (lat.npy); how many codes would differ; then a run
//!                                                 with the reference's noise alone: how many frames agree
//!   flow    hidden.npy + noise.npy              -> the first frames' Euler velocities (vel.npy) and results (lat.npy)
//!                                                 from the reference's own hidden states: the acoustic transformer alone
//!   codec   codes.npy                           -> speech.npy (24 kHz)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::flow::{self, ACOUSTIC};
use crate::{Force, Keep, Spec, VoxtralTts};

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

fn rows_of(shape: &[usize], v: &[f64]) -> Vec<Vec<f32>> {
    let w = shape[1..].iter().product::<usize>().max(1);
    v.chunks(w).map(|r| r.iter().map(|x| *x as f32).collect()).collect()
}

/// The dump's request (request.json, ref.py's) as a spec
pub fn spec_of(dir: &Path) -> Result<Spec> {
    let p = dir.join("request.json");
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
        .map_err(|e| Error(format!("{}: {e}", p.display())))?;
    Ok(Spec { text: r["text"].as_str().unwrap_or_default().to_string(), voice: r["voice"].as_str().unwrap_or(crate::DEFAULT_VOICE).to_string(),
              cfg: r["cfg"].as_f64().unwrap_or(crate::CFG as f64) as f32, steps: crate::STEPS })
}

pub fn run(e: &VoxtralTts, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let npy = |n: &str| -> Result<(Vec<usize>, Vec<f64>)> { read_npy(&dir.join(n)) };
    let flat = |n: &str| -> Result<Vec<f32>> { Ok(npy(n)?.1.into_iter().map(|x| x as f32).collect()) };
    let spec = spec_of(dir)?;
    log(format!("{:?} in {}, cfg {}", spec.text, spec.voice, spec.cfg));
    let codes: Vec<Vec<i32>> = match npy("codes.npy") {
        Ok((s, v)) => rows_of(&s, &v).into_iter().map(|r| r.into_iter().map(|x| x as i32).collect()).collect(),
        Err(_) => Vec::new(),
    };
    let noise = npy("noise.npy").map(|(s, v)| rows_of(&s, &v)).unwrap_or_default();
    // the codes ref.py keeps end with the [END_AUDIO] frame: the speech is the frames before it
    let spoken: Vec<Vec<i32>> = codes.iter().take_while(|f| f[0] != flow::END as i32).cloned().collect();
    let t0 = Instant::now();
    if stages.contains(&"prompt") {
        let (ids, rows) = e.prompt(&spec)?;
        let want: Vec<u32> = npy("ids.npy")?.1.into_iter().map(|x| x as u32).collect();
        let first = ids.iter().zip(&want).position(|(a, b)| a != b);
        log(format!("{:<24} {} here, {} in the reference; {}", "token ids", ids.len(), want.len(),
                    if ids == want { "identical".to_string() } else { format!("first difference at {first:?}") }));
        if ids != want {
            worst = f64::INFINITY;
        }
        report(log, &mut worst, "prompt rows", &rows, &flat("prefill.npy")?);
    }
    if stages.contains(&"frames") {
        if codes.is_empty() {
            return Err(Error(format!("{}: no codes.npy", dir.display())));
        }
        let mut k = Keep::default();
        let t = Instant::now();
        e.frames(&spec, codes.len() + 1, 0, t0, &mut |_| Ok(()), &Force { codes: Some(&spoken), noise: Some(&noise) }, Some(&mut k))?;
        log(format!("{:<24} {} frames in {:.1} s", "frames (forced)", spoken.len(), t.elapsed().as_secs_f64()));
        let n = k.hidden.len().min(codes.len());
        let wh = flat("hidden.npy")?;
        let ws = flat("sem.npy")?;
        let wl = flat("lat.npy")?;
        let d = e.bb.hidden;
        report(log, &mut worst, "hidden states", &k.hidden[..n].concat(), &wh[..n * d]);
        let sw = ws.len() / codes.len();
        report(log, &mut worst, "semantic logits", &k.sem[..n].concat(), &ws[..n * sw]);
        let m = k.lat.len().min(spoken.len());
        report(log, &mut worst, "flow results", &k.lat[..m].concat(), &wl[..m * ACOUSTIC]);
        let s0 = k.sem[..n].iter().zip(&codes).filter(|(l, c)| flow::semantic(l, e.flow.codes) as i32 != c[0]).count();
        let a0: usize = k.lat[..m].iter().zip(&spoken).map(|(l, c)| flow::quantize(l).iter().zip(&c[1..]).filter(|(a, b)| a != b).count()).sum();
        log(format!("{:<24} semantic {s0} of {n} frames differ, acoustic levels {a0} of {}", "codes vs reference", m * ACOUSTIC));
        let got = e.frames(&spec, codes.len() + 8, 0, t0, &mut |_| Ok(()), &Force { codes: None, noise: Some(&noise) }, None)?;
        let first = got.iter().zip(&spoken).position(|(a, b)| a != b);
        let same = got.iter().zip(&spoken).filter(|(a, b)| a == b).count();
        log(format!("{:<24} {} here, {} in the reference; {same} identical; first difference at {}", "frames (ref's noise)", got.len(), spoken.len(),
                    first.map_or("none".into(), |f| f.to_string())));
    }
    if stages.contains(&"flow") {
        let (_, hv) = npy("hidden.npy")?;
        let d = e.bb.hidden;
        let hs: Vec<f32> = hv.iter().map(|x| *x as f32).collect();
        let wv = flat("vel.npy")?;
        let wl = flat("lat.npy")?;
        let n = spoken.len().min(16);
        let work = e.flow.work(&e.ops, crate::STEPS)?;
        let hb = DevBuf::f32(&e.ops.gpu, d)?;
        let (mut gv, mut gl) = (Vec::new(), Vec::new());
        let mut flips = 0;
        for f in 0..n {
            hb.write(0, &crate::bytes(&hs[f * d..(f + 1) * d]))?;
            let mut vel = Vec::new();
            let lat = e.flow.acoustic(&e.ops, &e.nsd, &work, hb.fp(), &noise[f], spec.cfg, Some(&mut vel))?;
            flips += flow::quantize(&lat).iter().zip(&spoken[f][1..]).filter(|(a, b)| a != b).count();
            gv.extend(vel.concat());
            gl.extend(lat);
        }
        let per = crate::STEPS * 2 * ACOUSTIC;
        report(log, &mut worst, &format!("velocities ({n} frames)"), &gv, &wv[..n * per]);
        report(log, &mut worst, "flow results", &gl, &wl[..n * ACOUSTIC]);
        log(format!("{:<24} {flips} of {} levels differ", "flow levels", n * ACOUSTIC));
    }
    if stages.contains(&"codec") {
        if spoken.is_empty() {
            return Err(Error(format!("{}: no codes.npy", dir.display())));
        }
        let t = Instant::now();
        let got = e.codec.decode(&e.ops, &e.nsd, &spoken, &mut |_, _| Ok(()))?;
        log(format!("codec: {} frames to {:.2} s in {:.2} s", spoken.len(), got.len() as f64 / crate::RATE as f64, t.elapsed().as_secs_f64()));
        report(log, &mut worst, "speech", &got, &flat("speech.npy")?);
    }
    Ok(worst)
}
