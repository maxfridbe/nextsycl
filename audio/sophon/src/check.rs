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
//!
//! The talker alone (`talker`, its own card: no brain loaded), against reference/sophon/talker_ref.py's dump (the
//! release's proc.py tag_clean and thinker tokenizer, then the streaming x-vector prompt talker.py builds, through the
//! qwen-tts package on the CPU, greedy):
//!
//! ```text
//!   talker  request.json's text          -> the speakable ids against ids.npy, the prompt's rows against prefill.npy,
//!                                           the first logits against logits0.npy, greedy codes against codes.npy;
//!           (its "reference": a recording) -> the clone's x-vector against xvec.npy (talker_ref.py --ref-wav), ns_xvec.npy
//!           (and "ref_text": in context)  -> the transcript's ids against ref_ids.npy, the recording's frames against
//!                                           ref_codes.npy (talker_ref.py --ref-text: qwen-tts's ICL prompt)
//!   speak   lines.json (lines x seeds)   -> <name>-s<seed>.wav here, sampled as served (NS_SOPHON_TAGS applies);
//!           (its "voices": [{name, reference?, text?, words?}]) -> <voice>-<name>-s<seed>.wav and <voice>.xvec.npy,
//!           each voice a release voice by name or a recording's clone; "temperature" draws the talker and its
//!           predictor at it; "clone": "icl" clones a voice with a text in context ("layout": prompt | stream,
//!           "icl_max_s": its longest recording, cut by "words")
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{shards, Audio, Error, ModelFiles, Result};
use nextsycl_core::Gpu;
use nextsycl_tok::Tokenizer;

use crate::brain::{argmax, Rows};
use crate::sample::Draw;
use crate::speak::{Feed, IclLayout, Probe, Talk, Voice};
use crate::{ear, icl_ids, icl_part, Sophon};

/// The stages the talker alone answers (`talker`)
pub const TALKER_STAGES: &[&str] = &["talker", "speak"];

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

/// A float32 .npy of one row
fn write_npy(path: &Path, v: &[f32]) -> Result<()> {
    let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", v.len());
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(head.len() as u16).to_le_bytes());
    b.extend_from_slice(head.as_bytes());
    b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
}

/// A voice for a check: a recording's clone (`reference`: a WAV's path; with `icl` and its `ref_text` - or `text` -
/// in context, `words` its word times) or a release voice by name
fn check_voice(talk: &Talk, tok: &Tokenizer, v: &serde_json::Value, lang: &str, icl: bool, max_s: f64, log: &mut dyn FnMut(String)) -> Result<Voice> {
    match v["reference"].as_str() {
        Some(p) => {
            let b = std::fs::read(p).map_err(|e| Error(format!("{p}: {e}")))?;
            let (samples, rate) = nextsycl_audio::decode_wav(&b).map_err(|e| Error(format!("{p}: {e}")))?;
            let text = v["ref_text"].as_str().or_else(|| v["text"].as_str()).map(str::to_string);
            let words = v["words"].as_array().into_iter().flatten().filter_map(|w| {
                Some(nextsycl_audio::Timed { start: w["start"].as_f64()?, end: w["end"].as_f64()?, text: w["text"].as_str()?.to_string(), probability: None })
            }).collect();
            let r = nextsycl_audio::Reference { samples, rate, text, words };
            if icl {
                match icl_part(&r, max_s) {
                    Some((n, t)) => {
                        let i = talk.icl(&r.samples[..n], rate, &t, &icl_ids(tok, &t))?;
                        log(format!("voice {p}: in context, {:.2} s kept ({} frames, {} transcript tokens, made in {:.2} s): {t:?}", i.seconds, i.codes.len(),
                                    i.ids.len(), i.made_s));
                        return Ok(Voice::InContext(i));
                    }
                    None => log(format!("voice {p}: no transcript to clone in context by (or too long without word times): its x-vector")),
                }
            }
            Ok(Voice::XVector(talk.clone_voice(&r.samples, r.rate)?))
        }
        None => Ok(Voice::XVector(talk.voice(v["voice"].as_str().or_else(|| v["name"].as_str()).unwrap_or("chelsie"), lang)?)),
    }
}

/// A .npy of int32 rows ([n] or [n, w])
fn write_npy_i32(path: &Path, v: &[i32], w: usize) -> Result<()> {
    let shape = if w > 1 { format!("({}, {w})", v.len() / w) } else { format!("({},)", v.len()) };
    let mut head = format!("{{'descr': '<i4', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + head.len() + 1) % 64 != 0 {
        head.push(' ');
    }
    head.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(head.len() as u16).to_le_bytes());
    b.extend_from_slice(head.as_bytes());
    b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    std::fs::write(path, b).map_err(|e| Error(format!("{}: {e}", path.display())))
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

fn json(path: &Path) -> Result<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(|x| Error(format!("{}: {x}", path.display())))?)
        .map_err(|x| Error(format!("{}: {x}", path.display())))
}

/// The talker alone on `gpu` (TALKER_STAGES; none: both) against the dump in `dir`; the worst relative error
pub fn talker(files: &ModelFiles, gpu: &Arc<Gpu>, dir: &Path, stages: &[String], log: &mut dyn FnMut(String)) -> Result<f64> {
    let want = |s: &str| stages.is_empty() || stages.iter().any(|x| x == s);
    let role = |r: &str| files.get(r).cloned().ok_or_else(|| Error(format!("no {r} file")));
    let tok = Tokenizer::from_hf_json(&role("tokenizer")?).map_err(Error)?;
    let voices: Vec<PathBuf> = shards(files, "voice");
    let voices_dir = voices.first().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
    let talk = Talk::load(&role("talker")?, &role("talker-config")?, &role("codec")?, &voices_dir, gpu, false, log)?;
    let mut worst = 0f64;
    let mut sink = |_: &[f32]| Ok(());
    if want("talker") {
        let req = json(&dir.join("request.json"))?;
        let text = req["text"].as_str().unwrap_or_default();
        let lang = req["lang"].as_str().unwrap_or("en");
        // in context when the dump is of qwen-tts's ICL prompt (talker_ref.py --ref-text), the recording kept whole
        let icl = req["ref_text"].is_string();
        let voice = check_voice(&talk, &tok, &req, lang, icl, f64::MAX, log)?;
        if req["reference"].is_string() {
            write_npy(&dir.join("ns_xvec.npy"), voice.xvec())?;
            if dir.join("xvec.npy").exists() {
                report(log, &mut worst, "talker: the clone's x-vector", voice.xvec(), &f32s(&dir.join("xvec.npy"))?);
            }
        }
        if let Voice::InContext(i) = &voice {
            let ids_ref: Vec<u32> = read_npy(&dir.join("ref_ids.npy"))?.1.iter().map(|x| *x as u32).collect();
            log(format!("talker: the transcript's ids {} (reference: {}): {:?}", if i.ids == ids_ref { "the same" } else { "DIFFER" }, ids_ref.len(), i.ids));
            if i.ids != ids_ref {
                log(format!("talker: the reference's {ids_ref:?}"));
                worst = f64::INFINITY;
            }
            write_npy_i32(&dir.join("ns_ref_codes.npy"), &i.codes.concat(), 16)?;
            let (shape, c) = read_npy(&dir.join("ref_codes.npy"))?;
            let rc: Vec<Vec<i32>> = c.chunks(shape.get(1).copied().unwrap_or(16)).map(|r| r.iter().map(|x| *x as i32).collect()).collect();
            let same: usize = i.codes.iter().zip(&rc).map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x == y).count()).sum();
            let first: usize = i.codes.iter().zip(&rc).filter(|(a, b)| a[0] == b[0]).count();
            log(format!("talker: the recording's frames {} here, {} in the reference; codes agree on {same} of {} (first codebook {first} of {})", i.codes.len(),
                        rc.len(), i.codes.len().min(rc.len()) * 16, i.codes.len().min(rc.len())));
        }
        // the tags handed over whatever NS_SOPHON_TAGS says: the reference is upstream's path (a dump made with tags)
        let ids = crate::speakable_ids(&tok, text, true, true);
        let ids_ref: Vec<u32> = read_npy(&dir.join("ids.npy"))?.1.iter().map(|x| *x as u32).collect();
        log(format!("talker: the speakable ids {} (reference: {}): {:?}", if ids == ids_ref { "the same" } else { "DIFFER" }, ids_ref.len(), ids));
        if ids != ids_ref {
            log(format!("talker: the reference's {ids_ref:?}"));
            worst = f64::INFINITY;
        }
        // in context: the prompt made from the reference's own frames too (the encoder's few differing codes apart)
        let mut runs = vec![("", voice.clone())];
        if let Voice::InContext(i) = &voice {
            let (shape, c) = read_npy(&dir.join("ref_codes.npy"))?;
            let rc: Vec<Vec<i32>> = c.chunks(shape.get(1).copied().unwrap_or(16)).map(|r| r.iter().map(|x| *x as i32).collect()).collect();
            let p = req["reference"].as_str().unwrap_or_default();
            let (samples, rate) = nextsycl_audio::decode_wav(&std::fs::read(p).map_err(|e| Error(format!("{p}: {e}")))?).map_err(Error)?;
            runs.push((" (the reference's frames)", Voice::InContext(talk.icl_with(&samples, rate, &i.text, &i.ids, Some(rc))?)));
        }
        for (what, voice) in runs {
            let mut probe = Probe::default();
            let mut next = |_: usize| Feed { ids: ids.clone(), done: true };
            talk.speak_probed(&voice, IclLayout::Prompt, lang, Draw::GREEDY, 0, &mut next, &mut sink, &|| false, Some(&mut probe))?;
            log(format!("talker: the run{what}"));
            report(log, &mut worst, "talker: prompt rows", &probe.prefill, &f32s(&dir.join("prefill.npy"))?);
            let l0 = f32s(&dir.join("logits0.npy"))?;
            report(log, &mut worst, "talker: first logits", &probe.logits0, &l0);
            log(format!("talker: first code here {} (reference {})", argmax(&probe.logits0), argmax(&l0)));
            let (shape, c) = read_npy(&dir.join("codes.npy"))?;
            let g = shape.get(1).copied().unwrap_or(16);
            let codes_ref: Vec<Vec<i32>> = c.chunks(g).map(|r| r.iter().map(|x| *x as i32).collect()).collect();
            let n = probe.codes.len().min(codes_ref.len());
            let same_frames = probe.codes.iter().zip(&codes_ref).take_while(|(a, b)| a == b).count();
            let c0 = probe.codes.iter().zip(&codes_ref).filter(|(a, b)| a[0] == b[0]).count();
            let all: usize = probe.codes.iter().zip(&codes_ref).map(|(a, b)| a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()).sum();
            log(format!("talker: greedy frames {} here, {} in the reference; the first {same_frames} identical; of the {n} in both, first codes agree on {c0}, \
                         all codes on {all} of {}", probe.codes.len(), codes_ref.len(), n * g));
            log(format!("talker: the first frames here {:?}", &probe.codes[..probe.codes.len().min(3)]));
            log(format!("talker: the reference's       {:?}", &codes_ref[..codes_ref.len().min(3)]));
            // the reference's codes forced frame by frame: where greedy here would have taken another code
            let mut forced = Probe { force: Some(codes_ref.clone()), ..Probe::default() };
            let mut next = |_: usize| Feed { ids: ids.clone(), done: true };
            talk.speak_probed(&voice, IclLayout::Prompt, lang, Draw::GREEDY, 0, &mut next, &mut sink, &|| false, Some(&mut forced))?;
            log(format!("talker: forced through the reference's {} frames: first code misses {}, other codes' misses {} of {}", forced.codes.len(),
                        forced.misses[0], forced.misses[1], forced.codes.len() * (g - 1)));
        }
    }
    if want("speak") {
        let lj = json(&dir.join("lines.json"))?;
        let lang = lj["lang"].as_str().unwrap_or("en");
        // one voice (lines.json's "voice": the files <line>-s<seed>.wav), or several (its "voices": <voice>-<line>-s<seed>.wav)
        let icl = lj["clone"].as_str() == Some("icl");
        let layout = IclLayout::parse(lj["layout"].as_str().unwrap_or("prompt"))?;
        let max_s = lj["icl_max_s"].as_f64().unwrap_or(crate::ICL_MAX_S);
        let mut voices: Vec<(String, Voice)> = Vec::new();
        match lj["voices"].as_array() {
            Some(vs) => {
                for v in vs {
                    let name = v["name"].as_str().unwrap_or("voice").to_string();
                    let x = check_voice(&talk, &tok, v, lang, icl, max_s, log)?;
                    write_npy(&dir.join(format!("{name}.xvec.npy")), x.xvec())?;
                    voices.push((format!("{name}-"), x));
                }
            }
            None => voices.push((String::new(), check_voice(&talk, &tok, &lj, lang, icl, max_s, log)?)),
        }
        log(format!("speak: clones {} ({layout:?} layout)", if icl { "in context where a transcript is given" } else { "by their x-vectors" }));
        let seeds: Vec<u64> = lj["seeds"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).collect();
        let draw = match lj["temperature"].as_f64() {
            Some(t) => Draw { temperature: t as f32, ..Draw::DEFAULT },
            None => Draw::DEFAULT,
        };
        log(format!("speak: the sound tags {} the talker (NS_SOPHON_TAGS); temperature {}", if crate::tags_spoken() { "handed to" } else { "kept from" },
                    draw.temperature));
        for (voice, xvec) in &voices {
            for line in lj["lines"].as_array().into_iter().flatten() {
                let (name, text) = (line["name"].as_str().unwrap_or("line"), line["text"].as_str().unwrap_or_default());
                let ids = crate::speakable_ids(&tok, text, true, crate::tags_spoken());
                for &seed in &seeds {
                    let mut wav = Vec::new();
                    let mut next = |_: usize| Feed { ids: ids.clone(), done: true };
                    let mut keep = |x: &[f32]| {
                        wav.extend_from_slice(x);
                        Ok(())
                    };
                    let t = Instant::now();
                    let sp = talk.speak(xvec, layout, lang, draw, seed, &mut next, &mut keep, &|| false)?;
                    let path = dir.join(format!("{voice}{name}-s{seed}.wav"));
                    Audio { rate: crate::speak::RATE, channels: 1, samples: wav }.write_wav(&path, &[("ICMT", text)]).map_err(Error)?;
                    log(format!("speak: {} ({} frames) in {:.1} s, first audio {:.2} s: {text:?}", path.display(), sp.frames, t.elapsed().as_secs_f64(),
                                sp.first_audio));
                }
            }
        }
    }
    Ok(worst)
}
