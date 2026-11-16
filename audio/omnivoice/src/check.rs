//! Stage by stage against the reference (reference/omnivoice/ref.py: upstream's own code in float32 on the CPU), each
//! stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   prompt  request.json             -> the prompt's tokens (ids.npy's first row up to the audio: exact), T and L
//!   encode  ref24.npy (a clone)      -> sem16, semantic (HuBERT's mean), esem, eac, fc (the quantizer's input) and the
//!                                       codes (ref_codes.npy: how many agree); with ref_audio in request.json also the
//!                                       recording's preprocessing (ref24.npy against ours)
//!   pass    ids.npy, ids_mid.npy     -> the first step's emb0, layer0, hidden0 and logits0 [2, 8, T, 1025]; the middle
//!                                       step's logits_mid; how many positions' best codes agree
//!   run     request.json             -> a deterministic run (position temperature 0) from our own prompt: how many of
//!                                       tokens.npy's codes agree; its sound (ours.wav) for listening
//!   codec   tokens.npy               -> raw.npy (the decoder)
//!   post    raw.npy                  -> out.npy (silences, level, fades)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Audio, Error, Result};

use crate::codec::EncTaps;
use crate::{Gen, Keep, OmniVoice, Rng, Voice};

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
    } else if head.contains("'<f8'") {
        d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
    } else if head.contains("'<i4'") {
        d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else if head.contains("'<i8'") {
        d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64).collect()
    } else {
        return Err(Error(format!("{}: {head} (float32/64, int32 or int64 expected)", path.display())));
    };
    Ok((shape, v))
}

/// A float32 .npy (C order) of `shape`
pub fn write_npy(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}{}), }}", dims.join(", "), if shape.len() == 1 { "," } else { "" });
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(head.len() as u16).to_le_bytes());
    b.extend_from_slice(head.as_bytes());
    b.extend(v.iter().flat_map(|f| f.to_le_bytes()));
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
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

struct Req {
    text: String,
    language: Option<String>,
    instruct: Option<String>,
    ref_text: Option<String>,
    ref_audio: Option<String>,
    steps: usize,
    duration: Option<f32>,
    t: usize,
    l: usize,
    audio_start: usize,
    ref_rms: Option<f32>,
}

fn request(dir: &Path) -> Result<Req> {
    let p = dir.join("request.json");
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
        .map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let s = |k: &str| j[k].as_str().map(str::to_string);
    let n = |k: &str| j[k].as_u64().unwrap_or(0) as usize;
    Ok(Req {
        text: s("text").ok_or_else(|| Error("request.json: no text".into()))?,
        language: s("language"),
        instruct: s("instruct"),
        ref_text: s("ref_text"),
        ref_audio: s("ref_audio"),
        steps: j["steps"].as_u64().unwrap_or(32) as usize,
        duration: j["duration"].as_f64().map(|d| d as f32),
        t: n("T"),
        l: n("L"),
        audio_start: n("audio_start"),
        ref_rms: j["ref_rms"].as_f64().map(|v| v as f32),
    })
}

/// The reference's input ids [8, L] as the step's audio rows (rows audio_start..L, then the target again)
fn audio_rows(ids: &[f64], l: usize, a: usize, t: usize, c: usize) -> Vec<i32> {
    let mut v = Vec::new();
    for r in a..l {
        v.extend((0..c).map(|k| ids[k * l + r] as i32));
    }
    for r in l - t..l {
        v.extend((0..c).map(|k| ids[k * l + r] as i32));
    }
    v
}

/// [2 t, c v] (ours) -> [2, c, t, v] (the reference's)
fn logits_ref_order(x: &[f32], t: usize, c: usize, v: usize) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for b in 0..2 {
        for k in 0..c {
            for f in 0..t {
                let src = (b * t + f) * c * v + k * v;
                let dst = ((b * c + k) * t + f) * v;
                o[dst..dst + v].copy_from_slice(&x[src..src + v]);
            }
        }
    }
    o
}

/// How many positions' best codes (after guidance, the mask excluded) agree between two [2, c, t, v] logit sets
fn argmax_agree(a: &[f32], b: &[f32], c: usize, t: usize, v: usize, g: f32, mask: usize) -> (usize, usize) {
    let lsm = |x: &[f32]| -> Vec<f64> {
        let m = x.iter().fold(f32::NEG_INFINITY, |m, y| m.max(*y)) as f64;
        let s: f64 = x.iter().map(|y| (*y as f64 - m).exp()).sum();
        x.iter().map(|y| *y as f64 - m - s.ln()).collect()
    };
    let best = |x: &[f32], k: usize, f: usize| -> usize {
        let lc = lsm(&x[(k * t + f) * v..(k * t + f + 1) * v]);
        let lu = lsm(&x[((c + k) * t + f) * v..((c + k) * t + f + 1) * v]);
        (0..v).filter(|i| *i != mask).max_by(|i, j| {
            let zi = lc[*i] + g as f64 * (lc[*i] - lu[*i]);
            let zj = lc[*j] + g as f64 * (lc[*j] - lu[*j]);
            zi.total_cmp(&zj).then(j.cmp(i))
        }).unwrap_or(0)
    };
    let mut same = 0;
    for k in 0..c {
        for f in 0..t {
            if best(a, k, f) == best(b, k, f) {
                same += 1;
            }
        }
    }
    (same, c * t)
}

pub fn run(e: &OmniVoice, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let rq = request(dir)?;
    let mut worst = 0f64;
    let c = e.codebooks;
    let v = e.vocab;
    let lang = e.langs.resolve(rq.language.as_deref());
    let instruct = match rq.instruct.as_deref() {
        Some(i) => crate::text::resolve_instruct(i, &rq.text)?,
        None => None,
    };
    let g = Gen { steps: rq.steps, position_temperature: 0.0, duration: rq.duration, ..Gen::default() };
    let (ids_shape, ids) = read_npy(&dir.join("ids.npy"))?;
    if ids_shape != vec![c, rq.l] {
        return Err(Error(format!("ids.npy: shape {ids_shape:?}, [{c}, {}] expected", rq.l)));
    }
    let ref_frames = rq.l - rq.audio_start - rq.t;
    // the recording as the reference encoded it, its codes ours or the reference's
    let mut voice: Option<Voice> = None;
    if let Some(rt) = rq.ref_text.as_deref() {
        let codes: Vec<i32> = (0..c).flat_map(|k| (rq.audio_start..rq.audio_start + ref_frames).map(move |r| (k, r))).map(|(k, r)| ids[k * rq.l + r] as i32).collect();
        voice = Some(Voice { codes, frames: ref_frames, text: crate::text::add_punctuation(rt), rms: rq.ref_rms });
    }
    if stages.contains(&"prompt") {
        let p = e.prompt(&rq.text, voice.as_ref(), lang.as_deref(), instruct.as_deref(), g.denoise, rq.t);
        let want: Vec<u32> = (0..rq.audio_start).map(|r| ids[r] as u32).collect();
        let t_est = e.frames_for(&rq.text, voice.as_ref(), &g, None);
        log(format!("prompt: {} tokens here, {} in the reference: {}; T {t_est} here, {} there; L {} / {}", p.tokens.len(), want.len(),
                    if p.tokens == want { "the same" } else { "DIFFERENT" }, rq.t, p.len(), rq.l));
        if p.tokens != want {
            log(format!("  here:  {:?}\n  there: {want:?}", p.tokens));
            worst = f64::INFINITY;
        }
        if t_est != rq.t {
            worst = f64::INFINITY;
        }
    }
    if stages.contains(&"encode") && dir.join("ref24.npy").exists() {
        let (_, w) = read_npy(&dir.join("ref24.npy"))?;
        let w = f32s(&w);
        if let (Some(src), Some(rt)) = (rq.ref_audio.as_deref(), rq.ref_text.as_deref()) {
            let b = std::fs::read(src).map_err(|e| Error(format!("{src}: {e}")))?;
            let (s, rate) = nextsycl_audio::decode_audio(&b).map_err(Error)?;
            let (_, ours) = e.voice(&s, rate, rt, None)?;
            log(format!("preprocessing: {} samples here, {} in the reference", ours.len(), w.len()));
            if ours.len() == w.len() {
                report(log, &mut worst, "ref24", &ours, &w);
            }
        }
        let mut taps = EncTaps::default();
        let t0 = Instant::now();
        let (codes, frames) = e.codec.encode(&e.ops, &w, Some(&mut taps))?;
        log(format!("encode: {frames} frames in {:.2} s", t0.elapsed().as_secs_f64()));
        for (name, got) in [("sem16", &taps.sem16), ("semantic", &taps.semantic), ("esem", &taps.esem), ("eac", &taps.eac), ("fc", &taps.fc)] {
            let (_, want) = read_npy(&dir.join(format!("{name}.npy")))?;
            report(log, &mut worst, name, got, &f32s(&want));
        }
        // the semantic encoder from the reference's own HuBERT features
        let (ss, sw) = read_npy(&dir.join("semantic.npy"))?;
        let got = e.codec.semantic_encoder(&e.ops, &f32s(&sw), ss[0], None)?;
        let (_, want) = read_npy(&dir.join("esem.npy"))?;
        report(log, &mut worst, "esem (from theirs)", &got, &f32s(&want));
        let (_, rc) = read_npy(&dir.join("ref_codes.npy"))?;
        let same = codes.iter().zip(&rc).filter(|(a, b)| **a == **b as i32).count();
        let per: Vec<String> = (0..c).map(|k| {
            let s = (0..frames).filter(|f| codes[k * frames + f] == rc[k * frames + f] as i32).count();
            format!("{:.0}%", 100.0 * s as f64 / frames as f64)
        }).collect();
        log(format!("codes: {same} of {} agree ({:.1}%; by codebook {})", rc.len(), 100.0 * same as f64 / rc.len().max(1) as f64, per.join(" ")));
    }
    if stages.contains(&"pass") {
        let p = e.prompt(&rq.text, voice.as_ref(), lang.as_deref(), instruct.as_deref(), g.denoise, rq.t);
        let s = e.seq(&p)?;
        let mut keep = Keep::default();
        let mut lg = Vec::new();
        let t0 = Instant::now();
        e.step(&s, &audio_rows(&ids, rq.l, rq.audio_start, rq.t, c), g.guidance, Some(&mut keep), Some(&mut lg))?;
        log(format!("pass: {} + {} rows in {:.3} s (with the taps' reads)", rq.l, rq.t, t0.elapsed().as_secs_f64()));
        for (name, got) in [("emb0", &keep.emb0), ("layer0", &keep.taps.layer0), ("hidden0", &keep.taps.hidden)] {
            let (_, want) = read_npy(&dir.join(format!("{name}.npy")))?;
            report(log, &mut worst, name, got, &f32s(&want));
        }
        let (_, want) = read_npy(&dir.join("logits0.npy"))?;
        let want = f32s(&want);
        let got = logits_ref_order(&lg, rq.t, c, v);
        report(log, &mut worst, "logits0", &got, &want);
        let (same, all) = argmax_agree(&got, &want, c, rq.t, v, g.guidance, e.mask);
        log(format!("  best codes (guided): {same} of {all} agree ({:.1}%)", 100.0 * same as f64 / all as f64));
        if dir.join("ids_mid.npy").exists() {
            let (_, idm) = read_npy(&dir.join("ids_mid.npy"))?;
            let mut lg = Vec::new();
            e.step(&s, &audio_rows(&idm, rq.l, rq.audio_start, rq.t, c), g.guidance, None, Some(&mut lg))?;
            let (_, want) = read_npy(&dir.join("logits_mid.npy"))?;
            let want = f32s(&want);
            let got = logits_ref_order(&lg, rq.t, c, v);
            report(log, &mut worst, "logits_mid", &got, &want);
            let (same, all) = argmax_agree(&got, &want, c, rq.t, v, g.guidance, e.mask);
            log(format!("  best codes (guided): {same} of {all} agree ({:.1}%)", 100.0 * same as f64 / all as f64));
        }
    }
    if stages.contains(&"run") {
        let p = e.prompt(&rq.text, voice.as_ref(), lang.as_deref(), instruct.as_deref(), g.denoise, rq.t);
        let t0 = Instant::now();
        let codes = e.codes(&p, &g, &mut Rng::new(0), None, &mut |_| Ok(()))?;
        let took = t0.elapsed().as_secs_f64();
        let (_, want) = read_npy(&dir.join("tokens.npy"))?;
        let same = codes.iter().zip(&want).filter(|(a, b)| **a == **b as i32).count();
        log(format!("run: {} steps in {took:.2} s; {same} of {} codes agree with the reference's ({:.1}%)", g.steps, want.len(),
                    100.0 * same as f64 / want.len().max(1) as f64));
        let raw = e.codec.decode(&e.ops, &codes, rq.t)?;
        let out = e.finish(&raw, rq.ref_rms, &g);
        let a = Audio { rate: crate::codec::RATE, channels: 1, samples: out };
        a.write_wav(&dir.join("ours.wav"), &[("INAM", &rq.text)]).map_err(Error)?;
        let (total, free) = e.ops.gpu.memory()?;
        log(format!("  ours.wav: {:.2} s; the card: {:.2} GiB free of {:.2} after it", a.seconds(), free.unwrap_or(0) as f64 / (1u64 << 30) as f64,
                    total as f64 / (1u64 << 30) as f64));
    }
    if stages.contains(&"codec") {
        let (shape, tk) = read_npy(&dir.join("tokens.npy"))?;
        let t = shape.get(1).copied().unwrap_or(0);
        let codes: Vec<i32> = tk.iter().map(|x| *x as i32).collect();
        let t0 = Instant::now();
        let raw = e.codec.decode(&e.ops, &codes, t)?;
        log(format!("decode: {t} frames in {:.3} s", t0.elapsed().as_secs_f64()));
        let (_, want) = read_npy(&dir.join("raw.npy"))?;
        report(log, &mut worst, "raw", &raw, &f32s(&want));
    }
    if stages.contains(&"post") {
        let (_, raw) = read_npy(&dir.join("raw.npy"))?;
        let out = e.finish(&f32s(&raw), rq.ref_rms, &g);
        let (_, want) = read_npy(&dir.join("out.npy"))?;
        log(format!("post: {} samples here, {} in the reference", out.len(), want.len()));
        report(log, &mut worst, "out", &out, &f32s(&want));
    }
    Ok(worst)
}
