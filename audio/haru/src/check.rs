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
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{Error, Result};

use crate::session::{Prompt, Settings};
use crate::{Haru, Model};

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
    let m = e.inner.model.lock().map_err(|_| Error("the model's lock".into()))?;
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
        let mut s = m.lm.state(&m.ops, p.tokens.len() + 64)?;
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
        worst = worst.max(hw).max(lw).max(z1w);
    }
    if want("speed") && !stages.is_empty() {
        speed(&m, e.inner.ctx, dir, log)?;
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
