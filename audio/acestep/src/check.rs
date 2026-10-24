//! Stage by stage against the reference (reference/acestep/ref.py: ACE-Step's own pipeline, float32 on the CPU, as the
//! platform's music worker calls it), each from the reference's own input.
//!
//! ```text
//!   text      request.json                         -> text.json's ids (the caption prompt's, the lyrics')
//!   te        text.json's caption ids              -> te_out.npy [L, 1024]; lyric_emb.npy (the lyrics' embeddings)
//!   enc       te_out, lyric_emb, refer.npy         -> enc.npy [lyrics + 1 + caption, 2048] (the packed conditions)
//!   velocity  step_<i>_x, ctx.npy, steps.json, enc -> step_<i>_v.npy (every step, each from the reference's x)
//!   encode    vae_enc_in.npy (a task's source)      -> vae_enc_out.npy (the first window's posterior mean)
//!   hints     src.npy (a cover's source latents)    -> ctx.npy's first 64 channels (the codes and back)
//!   sample    noise.npy, enc.npy                   -> latents.npy (the whole flow from the reference's noise; a cover's
//!             or repaint's plan from its ctx / enc_1 / ctx_1 / repaint_mask / clean_src)
//!   vae       vae_in.npy                           -> wav_raw.npy [2, S] (the first window)
//!   vaefull   latents.npy                          -> audio.npy (every window, the peak)
//!   song      request.json, noise.npy              -> audio.npy (all of it: the texts, the flow, the decode, the peak;
//!             with the planner, its metadata and codes the reference's)
//!   lm        lm.json's prompts and tokens          -> the prompts and ids, the logits of each phase's first steps (the
//!             reference's tokens fed), the constrained decoder replayed over its tokens, the codes' hints (hints.npy)
//! ```

use std::path::Path;
use std::time::Instant;

use nextsycl_audio::{AudioRequest, Error, Result};
use serde_json::Value;

use crate::nn::Scratch;
use crate::sample::LATENT;
use crate::{AceStep, Plan, Task, Trace};

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
    } else if head.contains("'|b1'") {
        d.iter().map(|x| *x as f64).collect()
    } else {
        return Err(Error(format!("{}: {head}", path.display())));
    };
    Ok((shape, v))
}

fn f32s(dir: &Path, name: &str) -> Result<Vec<f32>> {
    Ok(read_npy(&dir.join(name))?.1.into_iter().map(|x| x as f32).collect())
}

/// (relative error |got - want| / |want|, cosine, max |got - want|)
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
    ((e2 / w2.max(1e-30)).sqrt(), dot / (w2.sqrt() * g2.sqrt()).max(1e-30), mx)
}

fn json(dir: &Path, name: &str) -> Result<Value> {
    std::fs::read(dir.join(name)).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| Error(format!("{}: no {name}", dir.display())))
}

/// The reference's request as this engine's (its source read from the file it names)
pub fn request(dir: &Path) -> Result<AudioRequest> {
    let r = json(dir, "request.json")?;
    let a = &r["args"];
    let lyrics = match a["lyrics_file"].as_str().filter(|f| !f.is_empty()) {
        Some(_) => r["params"]["lyrics"].as_str().map(str::to_string),
        None => a["lyrics"].as_str().filter(|l| !l.is_empty()).map(str::to_string),
    };
    let mut extra = nextsycl_audio::Given::new();
    if let Some(b) = a["bpm"].as_u64().filter(|b| *b > 0) {
        extra.insert("bpm".into(), b.to_string());
    }
    for (k, n) in [("key", "key"), ("timesig", "time-signature"), ("language", "language")] {
        if let Some(v) = a[k].as_str().filter(|v| !v.is_empty() && *v != "unknown") {
            extra.insert(n.into(), v.into());
        }
    }
    extra.insert("shift".into(), a["shift"].as_f64().unwrap_or(3.0).to_string());
    let task = a["task"].as_str().unwrap_or("text2music");
    let mut source = None;
    if task != "text2music" {
        extra.insert("task".into(), task.into());
        if task == "lego" {
            extra.insert("track".into(), "vocals".into());
        }
        extra.insert("cover-strength".into(), a["cover_strength"].as_f64().unwrap_or(0.5).to_string());
        extra.insert("repaint-start".into(), a["repaint_start"].as_f64().unwrap_or(0.0).to_string());
        extra.insert("repaint-end".into(), a["repaint_end"].as_f64().unwrap_or(-1.0).to_string());
        if task == "edit" {
            let f = a["source_lyrics_file"].as_str().unwrap_or("");
            extra.insert("source-lyrics".into(), std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?.trim().to_string());
        }
        let f = a["src"].as_str().unwrap_or("");
        let b = std::fs::read(f).map_err(|e| Error(format!("the reference's source {f}: {e}")))?;
        source = Some(nextsycl_audio::read_audio(&b).map_err(|e| Error(format!("{f}: {e}")))?);
    }
    extra.insert("planner".into(), (a["thinking"].as_bool() == Some(true)).to_string());
    if a["vae_mean"].as_bool() == Some(true) {
        extra.insert("vae-sample".into(), "0".into());
    }
    Ok(AudioRequest { prompt: a["caption"].as_str().unwrap_or("").into(), lyrics, seconds: a["duration"].as_f64().map(|d| d as f32),
                      steps: a["steps"].as_u64().map(|s| s as u32), seed: a["seed"].as_u64().unwrap_or(0), extra, source, ..Default::default() })
}

/// [C, N] channel-major to [N, C] frames-major
fn frames_major(shape: &[usize], v: &[f64]) -> Vec<f32> {
    let (c, n) = (shape[0], shape[1]);
    let mut out = vec![0f32; c * n];
    for ch in 0..c {
        for f in 0..n {
            out[f * c + ch] = v[ch * n + f] as f32;
        }
    }
    out
}

/// The reference's plan for its song: its context(s), its conditions' switch, its repaint mask
fn plan_of(e: &AceStep, dir: &Path, song: &crate::Song) -> Result<(Plan, Vec<f32>)> {
    let ctx = f32s(dir, "ctx.npy")?;
    let frames = ctx.len() / (2 * LATENT);
    // prepare_condition runs for the service, then in generate_audio (the cover's), then the non-cover branch's: the
    // last of them is the switch's when the strength is under 1
    let last = (1..8).take_while(|i| dir.join(format!("ctx_{i}.npy")).is_file()).last();
    let nc = match last {
        Some(i) if song.cover_strength < 1.0 && i >= 2 => Some((f32s(dir, &format!("enc_{i}.npy"))?, f32s(dir, &format!("ctx_{i}.npy"))?)),
        _ => None,
    };
    let repaint = match (&song.repaint, dir.join("repaint_mask.npy").is_file()) {
        (Some(r), true) => {
            let mask = f32s(dir, "repaint_mask.npy")?.iter().map(|x| *x > 0.5).collect();
            let (ratio, cf, _) = r.config();
            Some((mask, f32s(dir, "clean_src.npy")?, ratio, cf))
        }
        _ => None,
    };
    let start = if song.cover_noise > 0.0 { Some((f32s(dir, "src.npy")?, song.cover_noise)) } else { None };
    let _ = e;
    Ok((Plan { frames, out_frames: frames, ctx, nc, strength: song.cover_strength, start, repaint }, f32s(dir, "enc.npy")?))
}

fn line(log: &mut dyn FnMut(String), what: &str, got: &[f32], want: &[f32], worst: &mut f64) {
    let (rel, cos, mx) = compare(got, want);
    *worst = worst.max(rel);
    log(format!("{what:<24} relative error {rel:.2e}  cos {cos:.6}  max |d| {mx:.3e}  ({} values{})", want.len(),
                if got.len() != want.len() { format!(", got {}", got.len()) } else { String::new() }));
}

/// Runs the stages named against `dir`'s dumps; the worst relative error
pub fn run(e: &AceStep, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    let mut worst = 0f64;
    let mut s = Scratch::default();
    let req = request(dir)?;
    let mut song = e.song(&req)?;
    // the reference ran on the CPU: its decode windows are 256 frames
    song.decode_chunk = Some(crate::vae::CHUNK);
    let mut planned = None;
    if song.planner {
        // the reference's plan (its draws are torch's): its metadata and codes
        let r = json(dir, "lm.json")?;
        let m: std::collections::HashMap<String, String> = r["result"]["metadata"].as_object().into_iter().flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
        planned = Some(e.apply_plan(&mut song, &m, ref_codes(&r))?);
    }
    let texts = json(dir, "text.json")?;
    let ids = |i: usize| -> Vec<u32> { texts[i]["ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default() };
    let want_cap = ids(0);
    let want_lyr = ids(1);
    let all = stages.contains(&"all");
    let on = |n: &str| all || stages.contains(&n);
    if on("text") {
        let mut t = Trace::default();
        let _ = e.conditions(&mut s, &song, &mut t)?;
        let same = |a: &[u32], b: &[u32]| a == b;
        log(format!("text: caption ids {} ({} tokens{}), lyric ids {} ({} tokens{})", if same(&t.caption_ids, &want_cap) { "identical" } else { "DIFFER" },
                    want_cap.len(), if same(&t.caption_ids, &want_cap) { String::new() } else { format!(", got {}", t.caption_ids.len()) },
                    if same(&t.lyric_ids, &want_lyr) { "identical" } else { "DIFFER" }, want_lyr.len(),
                    if same(&t.lyric_ids, &want_lyr) { String::new() } else { format!(", got {}", t.lyric_ids.len()) }));
        if t.caption_ids != want_cap {
            let n = t.caption_ids.iter().zip(&want_cap).take_while(|(a, b)| a == b).count();
            log(format!("  caption ids part at {n}: ours {:?} theirs {:?}", &t.caption_ids[n..(n + 8).min(t.caption_ids.len())],
                        &want_cap[n..(n + 8).min(want_cap.len())]));
        }
        if t.lyric_ids != want_lyr {
            let n = t.lyric_ids.iter().zip(&want_lyr).take_while(|(a, b)| a == b).count();
            log(format!("  lyric ids part at {n}: ours {:?} theirs {:?}", &t.lyric_ids[n..(n + 8).min(t.lyric_ids.len())], &want_lyr[n..(n + 8).min(want_lyr.len())]));
        }
    }
    if on("te") {
        let t0 = Instant::now();
        let got = e.te.encode(&e.ops, &mut s, &want_cap)?;
        line(log, &format!("te ({:.2} s)", t0.elapsed().as_secs_f64()), &got, &f32s(dir, "te_out.npy")?, &mut worst);
        line(log, "lyric embeddings", &e.te.embed(&want_lyr)?, &f32s(dir, "lyric_emb.npy")?, &mut worst);
    }
    let enc_ref = f32s(dir, "enc.npy")?;
    if on("enc") {
        let got = e.dit.conditions(&e.ops, &mut s, &f32s(dir, "te_out.npy")?, &f32s(dir, "lyric_emb.npy")?, &f32s(dir, "refer.npy")?)?;
        line(log, "enc (conditions)", &got, &enc_ref, &mut worst);
    }
    let steps = json(dir, "steps.json")?;
    if on("lm") && dir.join("lm.json").is_file() {
        lm_stage(e, dir, log, &mut worst)?;
    }
    if on("encode") && dir.join("vae_enc_in.npy").is_file() {
        let (shape, w) = read_npy(&dir.join("vae_enc_in.npy"))?;
        let wav: Vec<f32> = w.iter().map(|x| *x as f32).collect();
        let t0 = Instant::now();
        let got = e.vae.encode_mean(&e.ops, &mut s, &wav)?;
        let (ws, want) = read_npy(&dir.join("vae_enc_out.npy"))?;
        line(log, &format!("encode ({} s of source, {:.2} s)", shape[1] / 48_000, t0.elapsed().as_secs_f64()), &got, &frames_major(&ws, &want), &mut worst);
    }
    if on("hints") && song.task == Task::Cover {
        let src = f32s(dir, "src.npy")?;
        let t = src.len() / LATENT;
        let t0 = Instant::now();
        let (codes, q) = e.dit.codes.tokenize(&e.ops, &mut s, &src, &e.dit.silence)?;
        let mut hints = e.dit.codes.detokenize(&e.ops, &mut s, &q)?;
        hints.truncate(t * LATENT);
        let ctx = f32s(dir, "ctx.npy")?;
        let want: Vec<f32> = (0..t).flat_map(|f| ctx[f * 2 * LATENT..f * 2 * LATENT + LATENT].to_vec()).collect();
        line(log, &format!("hints ({} codes, {:.2} s)", codes.len(), t0.elapsed().as_secs_f64()), &hints, &want, &mut worst);
    }
    if on("velocity") {
        let (plan, _) = plan_of(e, dir, &song)?;
        let n = steps.as_array().map_or(0, |a| a.len());
        let switch = (n as f64 * plan.strength) as usize;
        let cond = e.dit.cond(&e.ops, &mut s, &enc_ref)?;
        let cond_nc = match &plan.nc {
            Some((enc, _)) => Some(e.dit.cond(&e.ops, &mut s, enc)?),
            None => None,
        };
        let edit_tar = if song.task == Task::Edit { Some(e.dit.cond(&e.ops, &mut s, &f32s(dir, "enc_1.npy")?)?) } else { None };
        for (i, st) in steps.as_array().into_iter().flatten().enumerate() {
            let x = dir.join(format!("step_{i}_x.npy"));
            if !x.is_file() {
                break;
            }
            let (c, ctx) = match (&cond_nc, &plan.nc) {
                (Some(c), Some((_, ctx))) if i >= switch => (c, ctx),
                // an edit's calls: the source's (old words), then the target's, each step
                _ if song.task == Task::Edit && i % 2 == 1 => (edit_tar.as_ref().unwrap_or(&cond), &plan.ctx),
                _ => (&cond, &plan.ctx),
            };
            let t0 = Instant::now();
            let got = e.dit.velocity(&e.ops, &mut s, &f32s(dir, &format!("step_{i}_x.npy"))?, ctx, st["t"].as_f64().unwrap_or(0.0) as f32, c)?;
            line(log, &format!("velocity {i} (t {:.3}, {:.2} s)", st["t"].as_f64().unwrap_or(0.0), t0.elapsed().as_secs_f64()), &got,
                 &f32s(dir, &format!("step_{i}_v.npy"))?, &mut worst);
        }
    }
    if on("guided") && !e.dit.cfg.turbo {
        // the base model's guided step from each of the reference's states: APG's momentum carried as in its loop
        let (plan, enc) = plan_of(e, dir, &song)?;
        let c = e.dit.cfg.hidden;
        let null: Vec<f32> = (0..enc.len() / c).flat_map(|_| e.dit.null_cond.iter().copied()).collect();
        let (cc, cn) = (e.dit.cond(&e.ops, &mut s, &enc)?, e.dit.cond(&e.ops, &mut s, &null)?);
        let ts = crate::sample::schedule_base(song.steps, song.shift);
        let mut m = crate::sample::Momentum::default();
        for i in 0..ts.len() - 1 {
            let (a, b) = (dir.join(format!("step_{i}_x.npy")), dir.join(format!("step_{}_x.npy", i + 1)));
            if !a.is_file() || !b.is_file() {
                break;
            }
            let x = f32s(dir, &format!("step_{i}_x.npy"))?;
            let vc = e.dit.velocity(&e.ops, &mut s, &x, &plan.ctx, ts[i], &cc)?;
            let vu = e.dit.velocity(&e.ops, &mut s, &x, &plan.ctx, ts[i], &cn)?;
            let v = crate::sample::apg(&vc, &vu, song.guidance, &mut m);
            let next: Vec<f32> = x.iter().zip(&v).map(|(a, b)| a - b * (ts[i] - ts[i + 1])).collect();
            let want = f32s(dir, &format!("step_{}_x.npy", i + 1))?;
            let step: Vec<f32> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
            let wstep: Vec<f32> = want.iter().zip(&x).map(|(a, b)| a - b).collect();
            line(log, &format!("guided step {i} (the step's change)"), &step, &wstep, &mut worst);
        }
    }
    if on("sample") {
        let t0 = Instant::now();
        let mut t = Trace::default();
        let (plan, enc) = plan_of(e, dir, &song)?;
        let lat = if song.task == Task::Edit {
            let mut k = 0;
            e.flow_edit(&mut s, &song, &plan, &enc, &f32s(dir, "enc_1.npy")?, &f32s(dir, "src.npy")?, &mut |_| {
                k += 1;
                f32s(dir, &format!("edit_noise_{}.npy", k - 1))
            }, &mut |_, _| Ok(()))?
        } else {
            e.sample(&mut s, &song, &plan, &enc, f32s(dir, "noise.npy")?, &mut t, &mut |_, _| Ok(()))?
        };
        line(log, &format!("sample ({:.2} s)", t0.elapsed().as_secs_f64()), &lat, &f32s(dir, "latents.npy")?, &mut worst);
    }
    if on("vae") {
        let (shape, z) = read_npy(&dir.join("vae_in.npy"))?;
        let (c, n) = (shape[0], shape[1]);
        let mut zf = vec![0f32; c * n];
        for ch in 0..c {
            for f in 0..n {
                zf[f * LATENT + ch] = z[ch * n + f] as f32;
            }
        }
        let t0 = Instant::now();
        let wav = e.vae.decode(&e.ops, &mut s, &zf, crate::vae::CHUNK, &mut |_, _| Ok(()))?;
        line(log, &format!("vae ({:.2} s)", t0.elapsed().as_secs_f64()), &wav, &f32s(dir, "wav_raw.npy")?, &mut worst);
    }
    if on("vaefull") {
        // the reference's latents through the tiled decode and the peak: its audio
        let t0 = Instant::now();
        let lat = f32s(dir, "latents.npy")?;
        let a = e.sound(&mut s, &song, &lat, lat.len() / LATENT, true, &mut |_, _| Ok(()))?;
        let n = a.samples.len() / 2;
        let mut cm = vec![0f32; a.samples.len()];
        for i in 0..n {
            cm[i] = a.samples[2 * i];
            cm[n + i] = a.samples[2 * i + 1];
        }
        line(log, &format!("vae, tiled ({:.2} s)", t0.elapsed().as_secs_f64()), &cm, &f32s(dir, "audio.npy")?, &mut worst);
    }
    if on("song") {
        let t0 = Instant::now();
        let mut t = Trace::default();
        let target = match &song.source {
            Some(w) => Some(e.target(&mut s, &song, w, &mut |_, _| Ok(()))?),
            None => planned.clone(),
        };
        let enc = e.conditions(&mut s, &song, &mut t)?;
        let nc = if song.cover_strength < 1.0 {
            let d = crate::text::instruction("text2music", None);
            Some(if d == song.instruction { enc.clone() } else { e.conditions_as(&mut s, &song, &d, &mut Trace::default())? })
        } else {
            None
        };
        let plan = e.plan(&mut s, &song, target, nc, &mut t)?;
        if let Ok(want) = f32s(dir, "src.npy") {
            if song.task != Task::Text2Music || song.codes.is_some() {
                let src: Vec<f32> = if song.task == Task::Edit { t.target.clone() } else {
                    (0..plan.frames).flat_map(|f| plan.ctx[f * 2 * LATENT..f * 2 * LATENT + LATENT].to_vec()).collect()
                };
                let what = if song.task == Task::Cover { "song: the cover's hints" } else { "song: the source latents" };
                let want = if song.task == Task::Cover { f32s(dir, "ctx.npy")?.chunks(2 * LATENT).flat_map(|c| c[..LATENT].to_vec()).collect() } else { want };
                line(log, what, &src, &want, &mut worst);
            }
        }
        let lat = if song.task == Task::Edit {
            let old = crate::Song { lyrics: song.source_lyrics.clone().unwrap_or_default(), ..song.clone() };
            let se = e.conditions_as(&mut s, &old, &song.instruction, &mut Trace::default())?;
            let mut k = 0;
            e.flow_edit(&mut s, &song, &plan, &se, &enc, &t.target.clone(), &mut |_| {
                k += 1;
                f32s(dir, &format!("edit_noise_{}.npy", k - 1))
            }, &mut |_, _| Ok(()))?
        } else {
            e.sample(&mut s, &song, &plan, &enc, f32s(dir, "noise.npy")?, &mut t, &mut |_, _| Ok(()))?
        };
        line(log, "song: the latents", &lat, &f32s(dir, "latents.npy")?, &mut worst);
        let a = e.sound(&mut s, &song, &lat, plan.out_frames, true, &mut |_, _| Ok(()))?;
        let n = a.samples.len() / 2;
        let mut cm = vec![0f32; a.samples.len()];
        for i in 0..n {
            cm[i] = a.samples[2 * i];
            cm[n + i] = a.samples[2 * i + 1];
        }
        line(log, &format!("song ({:.2} s)", t0.elapsed().as_secs_f64()), &cm, &f32s(dir, "audio.npy")?, &mut worst);
    }
    Ok(worst)
}

/// The reference planner's codes (lm.json's result)
fn ref_codes(r: &Value) -> Vec<u32> {
    let s = r["result"]["audio_codes"].as_str().unwrap_or("");
    s.split("<|audio_code_").skip(1).filter_map(|x| x.split("|>").next()?.parse::<u32>().ok()).map(|c| c.min(crate::fsm::MAX_AUDIO_CODE)).collect()
}

/// The planner against the reference's run (lm.json): each phase's prompt and ids, its first logits with the reference's
/// tokens fed, the constrained decoder over the reference's tokens; the codes' hints
fn lm_stage(e: &AceStep, dir: &Path, log: &mut dyn FnMut(String), worst: &mut f64) -> Result<()> {
    use crate::fsm::{Fsm, Phase};
    use crate::planner::{codes_prompt, cot_prompt, cot_text, parse};
    let p = e.planner.as_ref().ok_or_else(|| Error("lm: the planner is not loaded (NS_ACE_PLANNER)".into()))?;
    let r = json(dir, "lm.json")?;
    let req = json(dir, "request.json")?;
    let caption = req["params"]["caption"].as_str().unwrap_or("");
    let lyrics = req["params"]["lyrics"].as_str().unwrap_or("");
    let mut s = Scratch::default();
    let mut cot_out = String::new();
    for ph in r["phases"].as_array().into_iter().flatten() {
        let phase = ph["phase"].as_str().unwrap_or("");
        let theirs = ph["prompt"].as_str().unwrap_or("");
        let ours = match phase {
            "cot" => cot_prompt(caption, lyrics),
            _ => codes_prompt(caption, lyrics, &cot_text(&parse(&cot_out)), false, "NO USER INPUT"),
        };
        let same = ours == theirs;
        let at = ours.chars().zip(theirs.chars()).take_while(|(a, b)| a == b).count();
        log(format!("lm {phase}: the prompt {}", if same { "identical".to_string() } else {
            format!("DIFFERS at char {at}: ours {:?} theirs {:?}", ours.chars().skip(at).take(40).collect::<String>(), theirs.chars().skip(at).take(40).collect::<String>())
        }));
        let ids_of = |k: &str| -> Vec<u32> { ph[k].as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|v| v as u32)).collect() };
        let (want, toks) = (ids_of("ids"), ids_of("tokens"));
        let got = p.tok.encode(theirs);
        log(format!("lm {phase}: ids {} ({} tokens)", if got == want { "identical" } else { "DIFFER" }, want.len()));
        // the logits of the first steps, the reference's tokens fed
        let uncond = ids_of("uncond_ids");
        let mut sq = p.lm.seq(&p.ops.gpu, want.len() + 8)?;
        let mut su = p.lm.seq(&p.ops.gpu, uncond.len().max(1) + 8)?;
        let t0 = Instant::now();
        let mut lc = p.lm.prefill(&p.ops, &mut s, &mut sq, &want)?;
        let mut lu = if uncond.is_empty() { Vec::new() } else { p.lm.prefill(&p.ops, &mut s, &mut su, &uncond)? };
        let pre = t0.elapsed().as_secs_f64();
        for i in 0..3 {
            let f = dir.join(format!("lm_{phase}_logits_{i}.npy"));
            if !f.is_file() {
                break;
            }
            let ref_l = f32s(dir, &format!("lm_{phase}_logits_{i}.npy"))?;
            let v = p.lm.vocab;
            let top = |l: &[f32]| l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (j, x)| if *x > b.1 { (j, *x) } else { b }).0;
            line(log, &format!("lm {phase} logits {i} (cond{}, top {} vs {})", if i == 0 { format!(", prefill {pre:.2} s") } else { String::new() }, top(&lc),
                               top(&ref_l[..v])), &lc, &ref_l[..v], worst);
            if !lu.is_empty() && ref_l.len() >= 2 * v {
                line(log, &format!("lm {phase} logits {i} (uncond)"), &lu, &ref_l[v..2 * v], worst);
            }
            let Some(t) = toks.get(i).copied() else { break };
            if lu.is_empty() {
                let mut seqs = [&mut sq];
                lc = p.lm.step(&p.ops, &mut s, &mut seqs, &[t])?;
            } else {
                let mut seqs = [&mut sq, &mut su];
                let both = p.lm.step(&p.ops, &mut s, &mut seqs, &[t, t])?;
                lc = both[..v].to_vec();
                lu = both[v..].to_vec();
            }
        }
        // the constrained decoder over the reference's tokens: each must be one it allows
        let user: std::collections::HashMap<&'static str, String> = ph["user_metadata"].as_object().into_iter().flatten()
            .filter_map(|(k, v)| {
                let k: &'static str = match k.as_str() { "bpm" => "bpm", "duration" => "duration", "keyscale" => "keyscale", "timesignature" => "timesignature", _ => return None };
                Some((k, v.as_str().map_or_else(|| v.to_string(), str::to_string)))
            }).collect();
        let mut fsm = match phase {
            "cot" => Fsm::new(&p.tables, &p.tok, Phase::Cot, None, user, true, false, false),
            _ => Fsm::new(&p.tables, &p.tok, Phase::Codes, ph["target_duration"].as_f64(), Default::default(), false, true, true),
        };
        let mut all = want.clone();
        let mut bad = Vec::new();
        let mut forced = 0;
        for (i, t) in toks.iter().enumerate() {
            // the reference's token the most likely (the decoder's top-1 choices - a language's first token, a caption's
            // end - then follow it)
            let mut sc = vec![0f32; p.lm.vocab];
            sc[*t as usize] = 1.0;
            fsm.process(&all, &mut sc);
            let n = sc.iter().filter(|x| x.is_finite()).count();
            forced += (n == 1) as usize;
            if !sc[*t as usize].is_finite() {
                let allowed: Vec<String> = sc.iter().enumerate().filter(|(_, x)| x.is_finite()).take(6).map(|(j, _)| format!("{j} {:?}", p.tok.decode(&[j as u32]))).collect();
                log(format!("  token {i} {t} {:?} refused in {:?}; allowed ({n}): {}", p.tok.decode(&[*t]), fsm.state, allowed.join(", ")));
                bad.push(i);
            }
            fsm.update(*t);
            all.push(*t);
        }
        log(format!("lm {phase}: the constrained decoder allows {} of the reference's {} tokens ({forced} forced){}", toks.len() - bad.len(), toks.len(),
                    if bad.is_empty() { String::new() } else { format!(" - refused at {:?}", &bad[..bad.len().min(8)]) }));
        if phase == "cot" {
            cot_out = ph["output"].as_str().unwrap_or("").to_string();
            let m = parse(&cot_out);
            log(format!("lm cot: parsed {:?}", m));
        }
    }
    if dir.join("hints.npy").is_file() {
        let codes = ref_codes(&r);
        let q = e.dit.codes.quantized(&codes);
        let mut s = Scratch::default();
        let h = e.dit.codes.detokenize(&e.ops, &mut s, &q)?;
        let want = f32s(dir, "hints.npy")?;
        line(log, &format!("lm hints ({} codes)", codes.len()), &h[..want.len().min(h.len())], &want[..want.len().min(h.len())], worst);
    }
    Ok(())
}
