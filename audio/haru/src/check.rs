//! Stage by stage against the reference (reference/haru/ref.py: the release's own PyTorch code in float32 on the same
//! files), each stage from the reference's own input so a difference shows where it is born.
//!
//! ```text
//!   codec    enc_in.npy (a user clip)            -> enc_mean.npy (the encoder's latent means, a frame at a time here)
//!            dec_in.npy (latents)                -> dec_out.npy (the decoder's audio)
//!   prompt   card.txt                            -> card_ids.npy (the tokenizer, the ChatML turn)
//!            p_tokens / p_model / p_user / masks -> p_emb.npy (the three streams summed), p_h{l}.npy (the last row's
//!                                                   residual after each layer; 32: the final norm), p_logits.npy
//!   frames   f_tok / f_model / f_user (teacher-forced) -> f_hidden.npy, f_logits.npy (argmax agreement), and the
//!            head from f_noise.npy -> f_latent.npy
//!   speed    frames of a live session (the prompt: the card), timed: ms a frame against real time (133)
//!   start    voice*.wav (saved voices' recordings), card.txt, user.wav -> a call's start on each voice: the old path
//!            (a frame at a time through the encoder, 16 positions a pass) and the fast ones (StartPath), timed; then
//!            the old path's frames against the new one's, cold and from the voices' cache (session.rs)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::session::{Prompt, Settings, StartPath};
use crate::{Haru, Model, RATE};

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
    } else if head.contains("'|b1'") {
        d.iter().map(|x| *x as f64).collect()
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

fn ints(path: &Path) -> Result<Vec<u32>> {
    Ok(read_npy(path)?.1.into_iter().map(|x| x as u32).collect())
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
        log(format!("{what:<28} sizes differ: {} here, {} in the reference", got.len(), want.len()));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!("{what:<28} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"));
    *worst = worst.max(rel);
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

/// The rows of `v` ([n, d]) where `mask` is set, as (first row, values): the masks are one run each
fn run_of(v: &[f32], mask: &[u32], d: usize) -> (usize, Vec<f32>) {
    let first = mask.iter().position(|m| *m != 0).unwrap_or(mask.len());
    let n = mask[first..].iter().take_while(|m| **m != 0).count();
    (first, v[first * d..(first + n) * d].to_vec())
}

/// The stages asked for (none: all) against the dump in `dir`; the worst relative error
pub fn run(e: &Haru, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let want = |s: &str| stages.is_empty() || stages.contains(&s);
    let mut worst = 0f64;
    let mut m = e.inner.model.lock().map_err(|_| Error("the model's lock".into()))?;
    let c = m.cfg.clone();
    let d = c.latent;
    if want("codec") {
        let x = f32s(&dir.join("enc_in.npy"))?;
        let s = m.enc.stream(&m.ops, c.frame)?;
        let t = Instant::now();
        let mean = m.encode_frames(&s, &x)?;
        log(format!("codec: {} frames encoded in {:.0} ms", x.len() / c.frame, t.elapsed().as_secs_f64() * 1e3));
        report(log, &mut worst, "codec: encoder means", &mean, &f32s(&dir.join("enc_mean.npy"))?);
        let lat = f32s(&dir.join("dec_in.npy"))?;
        let s = m.dec.stream(&m.ops, 1)?;
        let t = Instant::now();
        let mut audio = Vec::new();
        for l in lat.chunks_exact(d) {
            audio.extend(m.decode_frame(&s, l)?);
        }
        log(format!("codec: {} frames decoded in {:.0} ms", lat.len() / d, t.elapsed().as_secs_f64() * 1e3));
        // the reference's whole decode is not clamped; the frames here are (a frame's audio as the session gives it)
        let want_audio: Vec<f32> = f32s(&dir.join("dec_out.npy"))?.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        report(log, &mut worst, "codec: decoder audio", &audio, &want_audio);
    }
    let mut st = None;
    if want("prompt") || want("frames") {
        let card = std::fs::read_to_string(dir.join("card.txt")).map_err(|x| Error(x.to_string()))?;
        let ids = m.tok.encode(&format!("<|im_start|>system\n{card}<|im_end|>\n"));
        let ids_ref = ints(&dir.join("card_ids.npy"))?;
        let same = ids == ids_ref;
        log(format!("prompt: the card's {} ids {}", ids_ref.len(), if same { "match".to_string() } else { format!("differ: {} here", ids.len()) }));
        if !same {
            worst = f64::INFINITY;
        }
        let tokens = ints(&dir.join("p_tokens.npy"))?;
        let p = Prompt { model: run_of(&f32s(&dir.join("p_model.npy"))?, &ints(&dir.join("p_mmask.npy"))?, d),
                         user: run_of(&f32s(&dir.join("p_user.npy"))?, &ints(&dir.join("p_umask.npy"))?, d), tokens };
        // room for the dump's frames (a long teacher-forced run: ref.py --frames 300)
        let nf = if want("frames") { ints(&dir.join("f_tok.npy"))?.len() } else { 0 };
        let mut s = m.lm.state(&m.ops, p.tokens.len() + 64 + nf)?;
        let t = Instant::now();
        let (emb, taps) = prefill_tapped(&m, &mut s, &p)?;
        log(format!("prompt: {} positions in {:.0} ms", p.tokens.len(), t.elapsed().as_secs_f64() * 1e3));
        if want("prompt") {
            report(log, &mut worst, "prompt: embeddings", &emb, &f32s(&dir.join("p_emb.npy"))?);
            for (l, x) in taps.iter().enumerate() {
                if l % 4 == 3 || l < 3 {
                    report(log, &mut worst, &format!("prompt: after layer {l}"), x, &f32s(&dir.join(format!("p_h{l}.npy")))?);
                }
            }
            report(log, &mut worst, "prompt: hidden (final norm)", &m.lm.hidden.to_f32()?[..c.hidden], &f32s(&dir.join(format!("p_h{}.npy", c.layers)))?);
            let lg = m.lm.logits(&m.ops, 0)?;
            let lr = f32s(&dir.join("p_logits.npy"))?;
            report(log, &mut worst, "prompt: logits", &lg, &lr);
            log(format!("prompt: argmax {} here, {} in the reference", argmax(&lg), argmax(&lr)));
        }
        st = Some(s);
    }
    if want("frames") {
        let mut s = st.take().ok_or_else(|| Error("frames follow the prompt".into()))?;
        let tok = ints(&dir.join("f_tok.npy"))?;
        let (fm, fu) = (f32s(&dir.join("f_model.npy"))?, f32s(&dir.join("f_user.npy"))?);
        let (fh, fl) = (f32s(&dir.join("f_hidden.npy"))?, f32s(&dir.join("f_logits.npy"))?);
        let (fnz, flat) = (f32s(&dir.join("f_noise.npy"))?, f32s(&dir.join("f_latent.npy"))?);
        let v = c.vocab;
        let (mut agree, mut hw, mut lw, mut zw, mut z1w) = (0, 0f64, 0f64, 0f64, 0f64);
        // the silent-or-speak decision as the session samples it (temperature 0.7): P(PAD or EOP) here and in the
        // reference, frame by frame
        let silent = |l: &[f32]| {
            let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e = |x: f32| ((x as f64 - mx) / 0.7).exp();
            let z: f64 = l.iter().map(|x| e(*x)).sum();
            (e(l[c.pad as usize]) + e(l[c.eop as usize])) / z
        };
        let (mut ps_here, mut ps_ref, mut ps_worst) = (0f64, 0f64, 0f64);
        let mut sess = m.session(Settings { latent_noise: false, ..Default::default() }, 16)?;
        std::mem::swap(&mut sess.state, &mut s);
        for k in 0..=tok.len() {
            let h = m.lm.hidden.to_f32()?[..c.hidden].to_vec();
            let lg = m.lm.logits(&m.ops, 0)?;
            let z = m.head.sample(&m.ops, m.lm.hidden.fp(), &fnz[k * d..(k + 1) * d], 1.5)?;
            let raw: Vec<f32> = z.iter().map(|x| x / m.latent_scale - m.latent_bias).collect();
            let (rh, rl, rz) = (compare(&h, &fh[k * c.hidden..(k + 1) * c.hidden]).0, compare(&lg, &fl[k * v..(k + 1) * v]), compare(&raw, &flat[k * d..(k + 1) * d]).0);
            if argmax(&lg) == argmax(&fl[k * v..(k + 1) * v]) {
                agree += 1;
            }
            let (sh, sr) = (silent(&lg), silent(&fl[k * v..(k + 1) * v]));
            (ps_here, ps_ref, ps_worst) = (ps_here + sh, ps_ref + sr, ps_worst.max((sh - sr).abs()));
            // the head alone: from the reference's own hidden state (the language model's difference left out)
            m.lm.hidden.write(0, &crate::weights::bytes(&fh[k * c.hidden..(k + 1) * c.hidden]))?;
            let z1 = m.head.sample(&m.ops, m.lm.hidden.fp(), &fnz[k * d..(k + 1) * d], 1.5)?;
            let raw1: Vec<f32> = z1.iter().map(|x| x / m.latent_scale - m.latent_bias).collect();
            let rz1 = compare(&raw1, &flat[k * d..(k + 1) * d]).0;
            log(format!("frame {k:>2}: hidden rel {rh:.2e}, logits rel {:.2e} cos {:.6}, argmax {} / {}, head latent rel {rz:.2e} (from the reference's hidden \
                         {rz1:.2e})", rl.0, rl.2, argmax(&lg), argmax(&fl[k * v..(k + 1) * v])));
            (hw, lw, zw, z1w) = (hw.max(rh), lw.max(rl.0), zw.max(rz), z1w.max(rz1));
            if k == tok.len() {
                break;
            }
            m.advance(&mut sess, tok[k], &fm[k * d..(k + 1) * d], &fu[k * d..(k + 1) * d])?;
        }
        log(format!("frames: {} teacher-forced; argmax agrees on {agree} of {}; worst rel: hidden {hw:.2e}, logits {lw:.2e}, head latent {zw:.2e} \
                     (from the reference's hidden {z1w:.2e})", tok.len(), tok.len() + 1));
        let n = (tok.len() + 1) as f64;
        log(format!("frames: P(silent: PAD or EOP) at temperature 0.7, mean {:.4} here / {:.4} in the reference, worst frame differs by {ps_worst:.2e}",
                    ps_here / n, ps_ref / n));
        worst = worst.max(hw).max(lw).max(z1w);
    }
    if want("speed") && !stages.is_empty() {
        speed(&m, e.inner.ctx, dir, log)?;
    }
    if want("start") && !stages.is_empty() {
        let r = start(&mut m, e.inner.ctx, dir, log);
        m.path = StartPath::default();
        worst = worst.max(r?);
    }
    Ok(worst)
}

/// The prompt's rows through the model as Model::prefill does, keeping the input rows' embeddings and the last row's
/// residual after each layer
fn prefill_tapped(m: &Model, s: &mut crate::lm::State, p: &Prompt) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let c = &m.cfg;
    let n = p.tokens.len();
    let (mut emb, mut taps) = (Vec::new(), vec![Vec::new(); c.layers]);
    let mut at = 0;
    while at < n {
        let e = (at + crate::lm::ROWS).min(n);
        let span = |(r0, v): &(usize, Vec<f32>)| -> Option<(usize, Vec<f32>)> {
            let rows = v.len() / c.latent;
            let (a, b) = ((*r0).max(at), (r0 + rows).min(e));
            (a < b).then(|| (a - at, m.scale(&v[(a - r0) * c.latent..(b - r0) * c.latent])))
        };
        let (mm, uu) = (span(&p.model), span(&p.user));
        m.lm.embed_rows(&m.ops, &p.tokens[at..e], mm.as_ref().map(|(a, v)| (*a, v.as_slice())), uu.as_ref().map(|(a, v)| (*a, v.as_slice())))?;
        for r in 0..e - at {
            emb.extend(m.lm.residual(r)?);
        }
        let last = e == n;
        let r = e - at;
        m.lm.forward_tap(&m.ops, s, r, false, &mut |l, x| {
            if last {
                let v = x.to_f32()?;
                taps[l] = v[(r - 1) * c.hidden..r * c.hidden].to_vec();
            }
            Ok(())
        })?;
        at = e;
    }
    Ok((emb, taps))
}

/// A live session's frames timed: the card (card.txt) as the prompt, the user's clip (enc_in.npy) then silence as the
/// input, 120 frames
fn speed(m: &Model, ctx: usize, dir: &Path, log: &mut dyn FnMut(String)) -> Result<()> {
    let card = std::fs::read_to_string(dir.join("card.txt")).unwrap_or_default();
    let user = f32s(&dir.join("enc_in.npy")).unwrap_or_default();
    let free0 = m.ops.gpu.memory()?.1;
    let mut s = m.session(Settings::default(), ctx)?;
    let t = Instant::now();
    m.start(&mut s, &card, None)?;
    if let (Some(a), (total, Some(b))) = (free0, m.ops.gpu.memory()?) {
        log(format!("speed: a conversation of {ctx} positions holds {:.2} GiB; the card has {:.2} of {:.2} GiB in use", (a.saturating_sub(b)) as f64 / (1u64 << 30) as f64,
                    total.saturating_sub(b) as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64));
    }
    log(format!("speed: the prompt ({} positions) in {:.0} ms", s.prompt_len, t.elapsed().as_secs_f64() * 1e3));
    let fs = m.cfg.frame;
    let frames = 120;
    let mut times = Vec::new();
    let mut text = String::new();
    for k in 0..frames {
        let pcm: Vec<f32> = (0..fs).map(|i| user.get(k * fs + i).copied().unwrap_or(0.0)).collect();
        let f = m.step(&mut s, &pcm)?;
        times.push(f.seconds);
        text.push_str(&f.text);
    }
    m.ops.report(frames);
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / frames as f64;
    log(format!("speed: {frames} frames: {:.1} ms a frame (median {:.1}, p95 {:.1}, worst {:.1}; real time 133.3 ms) - {:.2}x real time on {}", mean * 1e3,
                times[frames / 2] * 1e3, times[frames * 95 / 100] * 1e3, times[frames - 1] * 1e3, 0.13333 / mean, m.ops.gpu.name));
    log(format!("speed: it said {:?}", text.split_whitespace().collect::<Vec<_>>().join(" ")));
    Ok(())
}

/// A WAV file's samples, mono at 24 kHz
fn wav24(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let (x, rate) = nextsycl_audio::decode_wav(&b).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    Ok(crate::resample::resample(&x, rate, RATE))
}

/// The start's paths compared: the old one, each fast one alone, both, both with the voices' cache
const PATHS: [(&str, StartPath); 5] = [
    ("old: 1 frame / 16 positions a pass", StartPath { batch_encode: false, wide: false, cache: false }),
    ("encoder 8 frames a pass", StartPath { batch_encode: true, wide: false, cache: false }),
    ("positions 128 a pass", StartPath { batch_encode: false, wide: true, cache: false }),
    ("both", StartPath { batch_encode: true, wide: true, cache: false }),
    ("both + the voices' cache", StartPath { batch_encode: true, wide: true, cache: true }),
];

/// Frames compared after a start: the old path's, the new path's cold and from the cache
const FRAMES: usize = 40;

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

/// A call's start on each voice (voice*.wav in `dir`) with the card (card.txt): timed on each path, then the old
/// path's first frames against the new one's (teacher-forced: the new sessions take the old one's token and latent
/// each frame, so a difference does not carry), cold and from the cache. The user's side: user.wav, then silence.
fn start(m: &mut Model, ctx: usize, dir: &Path, log: &mut dyn FnMut(String)) -> Result<f64> {
    let card = std::fs::read_to_string(dir.join("card.txt")).map_err(|e| Error(format!("card.txt: {e}")))?;
    let user = wav24(&dir.join("user.wav"))?;
    let mut voices: Vec<std::path::PathBuf> = std::fs::read_dir(dir).map_err(|e| Error(e.to_string()))?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("voice") && n.ends_with(".wav"))).collect();
    voices.sort();
    if voices.is_empty() {
        return Err(Error(format!("{}: no voice*.wav", dir.display())));
    }
    let fs = m.cfg.frame;
    let pad = m.cfg.pad;
    let mut worst = 0f64;
    log(format!("start: the silo (libnextsycl-haru.so) {}", if m.ops.has_silo() { "open: its kernels in the wide passes" } else { "not open: the engine's own kernels" }));
    for vp in &voices {
        let name = vp.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let r = crate::session::prepare_reference(&wav24(vp)?, 180.0);
        let frames = r.len().div_ceil(fs);
        log(format!("start: {name}: {:.1} s of voice (trimmed), {frames} frames to encode", r.len() as f64 / RATE as f64));
        let t = Instant::now();
        let mut s = m.session(Settings { seed: 41, ..Default::default() }, ctx)?;
        m.ops.gpu.sync()?;
        log(format!("start: {name}: a conversation's state ({ctx} positions) made in {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
        for (what, path) in PATHS {
            m.path = path;
            m.cache.clear();
            for run in 0..3 {
                let (_, t) = m.start_timed(&mut s, &card, Some(&r))?;
                log(format!("start: {name}: {what:<36} run {run}: frame 0 after {:>5.0} ms ({} positions; encode {:.0} ms, prefill {:.0} ms, restore {:.0} ms, \
                             store {:.0} ms, frame 0 {:.0} ms){}", 1e3 * t.total, t.positions, 1e3 * t.encode, 1e3 * t.prefill, 1e3 * t.restore,
                            1e3 * t.store, 1e3 * t.frame0, if t.hit { " - from the cache" } else { "" }));
            }
            if m.ops.profiling() {
                log(format!("start: {name}: {what}: the operations of a start (NS_HARU_PROFILE: each waited for), per start:"));
                m.ops.report(3);
            }
        }
        log(format!("start: {name}: the cache holds {:.1} MB for this voice and card", m.cache.stats().1 as f64 / 1e6));
        drop(s);
        for speak_first in [false, true] {
            let set = Settings { seed: 41, speak_first, ..Default::default() };
            let cctx = frames + 1024 + FRAMES;
            m.path = PATHS[0].1;
            m.cache.clear();
            let (mut a, mut b, mut h) = (m.session(set.clone(), cctx)?, m.session(set.clone(), cctx)?, m.session(set.clone(), cctx)?);
            let mut fa = m.start(&mut a, &card, Some(&r))?;
            m.path = StartPath::default();
            let mut fb = m.start(&mut b, &card, Some(&r))?;
            let mut fh = m.start(&mut h, &card, Some(&r))?;
            let (mut arg, mut cls, mut tok, mut aud, mut hit_tok, mut hit_max) = (0, 0, 0, 0, 0, 0f32);
            let mut rels = Vec::new();
            let mut said = (String::new(), String::new());
            for k in 0..=FRAMES {
                arg += (fa.argmax == fb.argmax) as usize;
                cls += ((fa.argmax == pad) == (fb.argmax == pad)) as usize;
                tok += (fa.token == fb.token) as usize;
                aud += ((rms(&fa.audio) > 1e-3) == (rms(&fb.audio) > 1e-3)) as usize;
                rels.push(compare(&fb.audio, &fa.audio).0);
                hit_tok += (fh.token == fb.token && fh.argmax == fb.argmax) as usize;
                hit_max = fh.audio.iter().zip(&fb.audio).map(|(x, y)| (x - y).abs()).fold(hit_max, f32::max);
                said.0.push_str(&fa.text);
                said.1.push_str(&fb.text);
                if k == FRAMES {
                    break;
                }
                let la = a.latent().to_vec();
                m.force(&mut b, fa.token, &la);
                m.force(&mut h, fa.token, &la);
                let pcm: Vec<f32> = (0..fs).map(|i| user.get(k * fs + i).copied().unwrap_or(0.0)).collect();
                fa = m.step(&mut a, &pcm)?;
                fb = m.step(&mut b, &pcm)?;
                fh = m.step(&mut h, &pcm)?;
            }
            let n = FRAMES + 1;
            let mut sorted = rels.clone();
            sorted.sort_by(f64::total_cmp);
            log(format!("start: {name}, speak first {speak_first}: the new path against the old over {n} frames (teacher-forced): argmax {arg} of {n}, \
                         silent-vs-speak {cls} of {n}, sampled tokens {tok} of {n}, audible-vs-quiet audio {aud} of {n}; audio rel median {:.2e}, worst {:.2e}",
                        sorted[n / 2], sorted[n - 1]));
            log(format!("start: {name}, speak first {speak_first}: old said {:?}; new said {:?}", said.0.split_whitespace().collect::<Vec<_>>().join(" "),
                        said.1.split_whitespace().collect::<Vec<_>>().join(" ")));
            log(format!("start: {name}, speak first {speak_first}: a cache hit against the cold start: tokens and argmax {hit_tok} of {n} the same, audio max \
                         difference {hit_max:.1e}"));
            if hit_tok != n || hit_max != 0.0 {
                worst = f64::INFINITY;
            }
            if arg != n || cls != n {
                worst = worst.max(1.0);
            }
        }
    }
    m.cache.clear();
    Ok(worst)
}
