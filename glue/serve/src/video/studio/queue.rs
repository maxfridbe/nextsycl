//! The clip queue: what a request becomes (a queue item), holds (all, per project, per batch), projects, and the
//! files they persist to. The item shape is the legacy server's, so scene files and tools written for it work
//! unchanged (docs/LEGACY-API.md).

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::{RpcError, Studio};

pub const MAGIC: [&str; 4] = ["prev", "prev_cam", "first", "first_cam"];
pub const CHAIN_MODES: [(&str, &str); 4] = [
    ("video", "the previous clip's mp4: its last frame decoded and encoded again"),
    ("png", "the previous clip's lossless last frame (.last.png)"),
    ("latent", "the previous clip's last latent frame (no decode, no codec)"),
    ("none", "no chaining"),
];

/// The holds and the queue, persisted.
#[derive(Default)]
pub struct Queue {
    pub items: Vec<Value>,
    pub paused: bool,
    pub paused_projects: BTreeSet<String>,
    pub paused_batches: BTreeSet<String>,
    /// Holds by name (the front door's "maintenance", "gpu run"): each lifts only its own, so one cannot take another's
    /// pause - or a person's - for its own and leave it behind
    pub held_by: BTreeSet<String>,
}

pub fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// "Name 03/12: ..." -> "Name": the series a clip belongs to (the text before the clip number).
pub fn label_prefix(label: &str) -> String {
    let bytes = label.as_bytes();
    for (i, c) in label.char_indices() {
        if c.is_ascii_digit() {
            // digits, then '/'
            let j = label[i..].find(|ch: char| !ch.is_ascii_digit()).map_or(label.len(), |k| i + k);
            if j < bytes.len() && bytes[j] == b'/' && (i == 0 || !label[..i].ends_with(|ch: char| ch.is_ascii_digit())) {
                return label[..i].trim().to_string();
            }
        }
    }
    String::new()
}

/// "Name 03/12: ..." -> 3: the edit order.
pub fn label_number(label: &str) -> Option<u64> {
    for (i, c) in label.char_indices() {
        if c.is_whitespace() {
            let rest = &label[i + c.len_utf8()..];
            let digits: String = rest.chars().take_while(|d| d.is_ascii_digit()).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with('/') && rest[digits.len() + 1..].starts_with(|d: char| d.is_ascii_digit()) {
                return digits.parse().ok();
            }
        }
    }
    None
}

/// A clip's place in the edit: the `clip` its request stated; for clips made before that key, the label's "NN/MM"
/// number, else the pass number the creative platform put after the `#` of its `reentrant_id`
/// ("<run>:shots:0:0#17" is the 17th; its labels, "<run>:shots:0:017", carry no "NN/MM"), else 0.
pub fn clip_number(item: &Value) -> u64 {
    item["clip"].as_u64().filter(|c| *c > 0).or_else(|| label_number(&s(item, "label"))).or_else(|| {
        let id = s(item, "reentrant_id");
        let tail = id.rsplit_once('#')?.1;
        tail.chars().all(|c| c.is_ascii_digit()).then(|| tail.parse().ok()).flatten()
    }).unwrap_or(0)
}

/// The project an item belongs to: its own, or its label's series.
pub fn project_of(item: &Value) -> String {
    let p = s(item, "project");
    if !p.is_empty() {
        return p;
    }
    let label = s(item, "label");
    // the reference requires "<prefix> NN/MM"
    match label_number(&label) {
        Some(_) => label_prefix(&label),
        None => String::new(),
    }
}

/// The spoken lines of a prompt ("<d>[English] ...</d>"), joined.
pub fn dialogue_of(prompt: &str) -> String {
    let mut out = Vec::new();
    let mut rest = prompt;
    while let Some(a) = rest.find("<d>[") {
        let after = &rest[a + 4..];
        let Some(close) = after.find(']') else { break };
        let body = after[close + 1..].trim_start();
        let Some(end) = body.find("</d>") else { break };
        out.push(body[..end].split_whitespace().collect::<Vec<_>>().join(" "));
        rest = &body[end + 4..];
    }
    out.join(" / ")
}

/// A job's name for people: a request's `title`, on one line, 120 characters at most (none: null, the label shows).
/// Never matched on, so nothing else is asked of it.
pub fn title_of(p: &Value) -> Value {
    let t: String = p["title"].as_str().unwrap_or("").chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    match t.trim().chars().take(120).collect::<String>() {
        t if t.is_empty() => Value::Null,
        t => json!(t),
    }
}

/// The legacy label filter: word characters and ` -.,:'()!?/`, 80 at most.
fn clean_label(l: &str) -> String {
    let kept: String = l.chars().filter(|c| c.is_alphanumeric() || *c == '_' || " -.,:'()!?/".contains(*c)).collect();
    kept.chars().take(80).collect::<String>().trim().to_string()
}

pub fn check_canvas(w: i64, h: i64) -> Result<(), RpcError> {
    if w % 32 != 0 || h % 32 != 0 {
        return Err(RpcError::param("width and height must be multiples of 32"));
    }
    if !(256..=1344).contains(&w) || !(256..=1344).contains(&h) {
        return Err(RpcError::param("width and height must be between 256 and 1344"));
    }
    if w * h > 768 * 1344 {
        return Err(RpcError::param(&format!("area {} exceeds the model's 1032192 pixel cap", w * h)));
    }
    Ok(())
}

/// Whether the daemon runs the model of registry id `id` (the start's models, then those it listed since)
pub fn engine_id_known(id: &str) -> bool {
    engine_pairs().iter().any(|(_, i)| i == id)
}

/// The engine a request asks for, as the front end names it: its `model` - a registry id the daemon runs (`known`),
/// refused otherwise rather than made by another model - else its `engine` (INT8 when it names none). A model and an
/// engine that disagree are refused.
pub fn requested_engine(p: &Value, known: &dyn Fn(&str) -> bool) -> Result<String, RpcError> {
    let by_engine = p.get("engine").and_then(|v| v.as_str()).map(str::trim).filter(|e| !e.is_empty());
    let Some(id) = p.get("model").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(engine_name(by_engine));
    };
    if !known(id) {
        return Err(RpcError::param(&format!("model {id}: not a video model the engine runs (video.engines.list; a registered video model is added by enabling it)")));
    }
    let name = engine_quant(id);
    if let Some(e) = by_engine.map(|e| engine_name(Some(e))).filter(|e| *e != name) {
        return Err(RpcError::param(&format!("model {id} is engine {name}, not {e}: name one of them")));
    }
    Ok(name)
}

/// HunyuanVideo-1.5's engines (video.generate engine HV15, HV15-CFG, HV15-I2V): their default steps and the most
/// they take, or None for H3's
pub fn hv15_rules(engine: &str) -> Option<(f64, f64)> {
    match engine {
        "HV15-I2V" => Some((8.0, 50.0)),
        // the cfg-distilled T2V card's fast default (the user's pick, 2026-11-23): 30 steps with DeepCache and Sage
        // attention from its card - 3.7x the 50-step clip, another take of the scene as sharp
        "HV15" => Some((30.0, 100.0)),
        e if e.starts_with("HV15") => Some((50.0, 100.0)),
        _ => None,
    }
}

/// HunyuanVideo-1.5's frame counts: 4k + 1, 5 s is 121 (what it was trained on), at most 10 s
pub const HV15_FRAMES: (u64, u64) = (5, 241);

/// The request rules of the engines on HunyuanVideo-1.5's code (HV15..., MovieRoll's MR...): default and most steps,
/// the frames (least, most) on a grid of `grid` frames + 1, the default canvas
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rules {
    pub steps: (f64, f64),
    pub frames: (u64, u64),
    pub grid: u64,
    pub canvas: (f64, f64),
}

pub fn engine_rules(engine: &str) -> Option<Rules> {
    if let Some(steps) = hv15_rules(engine) {
        return Some(Rules { steps, frames: HV15_FRAMES, grid: 4, canvas: (848.0, 480.0) });
    }
    // MovieRoll (video/hunyuanvideo15 movieroll.rs): chunks of 16 frames, 1 s to 60 s, 864 x 480 or 480 x 864; the
    // distilled cards' fixed 5-step ladder, the base card's 50 (a multiple of the window's 5)
    let mr = |steps| Some(Rules { steps, frames: (17, 1441), grid: 16, canvas: (864.0, 480.0) });
    match engine {
        "MR" | "MR-I2V" => mr((5.0, 5.0)),
        "MR-BASE" => mr((50.0, 100.0)),
        _ => None,
    }
}

/// video.roll's events: [{prompt, seconds}] - 1 to 64, each prompt 1 to 4,000 characters, 0.67 to 60 s (whole chunks of
/// 16 frames in the engine). The limits and why (a film's length itself has none: the engine's memory is bounded
/// whatever it is - movieroll.rs Room, kept_chunks, the decoder's window):
/// - 64 events a call: each is encoded before the film starts (or before its chunk, appended), the card waiting - 64
///   keeps that wait under a minute; a live film takes more with video.rollAppend, 64 a call again;
/// - 60 s an event: an event's chunks stay in host memory from its middle on until it ends (a recompute takes its
///   middle), 45 chunks at most then; longer, the same scene goes on as consecutive events;
/// - 4,000 characters a prompt: the text encoder reads 1,000 tokens (MovieRoll's text_len) - more is cut anyway.
pub fn roll_events(v: Option<&Value>) -> Result<Vec<Value>, RpcError> {
    let list = v.and_then(|v| v.as_array()).ok_or_else(|| RpcError::param("events: a list of {prompt, seconds}"))?;
    if list.is_empty() || list.len() > 64 {
        return Err(RpcError::param("events: 1 to 64 of them"));
    }
    list.iter()
        .enumerate()
        .map(|(i, e)| {
            let prompt = e["prompt"].as_str().map(str::trim).unwrap_or("");
            if prompt.is_empty() || prompt.chars().count() > 4000 {
                return Err(RpcError::param(&format!("events[{i}].prompt: 1 to 4,000 characters")));
            }
            let sec = e["seconds"].as_f64().unwrap_or(10.0);
            if !(0.6..=60.0).contains(&sec) {
                return Err(RpcError::param(&format!("events[{i}].seconds: 0.67 to 60 (chunks of 16 frames)")));
            }
            Ok(json!({"prompt": prompt, "seconds": sec}))
        })
        .collect()
}

/// The frames for `seconds` on a grid of `grid` + 1 (snapped up), at most `max`
pub fn grid_frames(seconds: f64, grid: u64, max: u64) -> u64 {
    let f = ((seconds * 24.0).round() as u64).max(1);
    ((f - 1).div_ceil(grid) * grid + 1).min(max)
}

/// HunyuanVideo-1.5's frames for `seconds` (the engine's frames_for: 24 fps, up to the next 4k + 1)
pub fn hv15_frames(seconds: f64) -> u64 {
    let f = ((seconds * 24.0).round() as u64).max(1);
    (f - 1).div_ceil(4) * 4 + 1
}

/// HunyuanVideo-1.5's canvas: sides multiples of 16 (the engine takes the 480p bucket nearest in shape)
pub fn check_canvas_16(w: i64, h: i64) -> Result<(), RpcError> {
    if w % 16 != 0 || h % 16 != 0 {
        return Err(RpcError::param("width and height must be multiples of 16"));
    }
    if !(256..=1344).contains(&w) || !(256..=1344).contains(&h) {
        return Err(RpcError::param("width and height must be between 256 and 1344"));
    }
    Ok(())
}

/// Frame counts the model makes: 17k + 5, from 22 (0.92 s) to 362 (15.08 s) at 24 fps
pub const FRAMES: (u64, u64) = (22, 362);

/// `frames`: one the model makes, else the error naming the nearest ones
pub fn check_frames(f: u64) -> Result<u64, RpcError> {
    if f % 17 == 5 && (FRAMES.0..=FRAMES.1).contains(&f) {
        return Ok(f);
    }
    let down = (FRAMES.0..=f.min(FRAMES.1)).rev().find(|n| n % 17 == 5);
    let up = (f.max(FRAMES.0)..=FRAMES.1).find(|n| n % 17 == 5);
    let near: Vec<String> = down.into_iter().chain(up).collect::<BTreeSet<_>>().into_iter().map(|n| n.to_string()).collect();
    Err(RpcError::param(&format!("frames {f}: the model makes 17k + 5 frames ({}-{}); nearest: {}", FRAMES.0, FRAMES.1, near.join(" or "))))
}

/// The keys a chain object stands for. `chain: {from, context_frames, sound, still, mode, color_reanchor}`:
/// - context_frames N (17k + 5): the last N frames of `from` as one moving keyframe at frame 0 (guide_clip) - the
///   clip replays them, so a join trims N / 2 each side; with `sound` (default true) the sound's last N / 24 s too
///   (first_audio, first_audio_s)
/// - still (default: when context_frames is 0): `from`'s last frame as the first frame (first_frame, chain_mode `mode`)
/// - color_reanchor: an anchor or picture whose brightness the carried pictures are scaled to (exposure_ref)
///
/// Keys given flat as well win. pin_renorm and keys the box does not have are listed back as ignored.
pub fn unchain(p: &Value) -> Result<(Value, Vec<String>), RpcError> {
    let mut out = p.clone();
    let mut ignored = Vec::new();
    let Some(c) = p.get("chain").filter(|c| !c.is_null()) else { return Ok((out, ignored)) };
    let c = c.as_object().ok_or_else(|| RpcError::param("chain: an object {from, context_frames, sound, still, mode, color_reanchor}"))?;
    let from = c.get("from").and_then(|v| v.as_str()).unwrap_or("prev").to_string();
    let n = c.get("context_frames").map_or(Ok(0), |v| v.as_u64().ok_or_else(|| RpcError::param("chain.context_frames: a whole number")))?;
    if n > 0 && (n % 17 != 5 || n > FRAMES.1) {
        let near = check_frames(n).err().map(|e| e.message).unwrap_or_default();
        return Err(RpcError::param(&format!("chain.context_frames {n}: 5, 22, 39 ... (17k + 5) - {}", near.rsplit("; ").next().unwrap_or(""))));
    }
    let set = |out: &mut Value, k: &str, v: Value| {
        if out.get(k).is_none_or(|x| x.is_null()) {
            out[k] = v;
        }
    };
    // a clip made to a soundtrack has its sound already: the chain carries the picture only
    let soundtrack = p.get("soundtrack").is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty()));
    if n > 0 {
        set(&mut out, "guide_clip", json!(format!("{from}:{n}:0")));
        if !soundtrack && c.get("sound").and_then(|v| v.as_bool()) != Some(false) {
            set(&mut out, "first_audio", json!(from));
            set(&mut out, "first_audio_s", json!(n as f64 / 24.0));
        }
    }
    if c.get("still").and_then(|v| v.as_bool()).unwrap_or(n == 0) {
        set(&mut out, "first_frame", json!(from));
        if let Some(m) = c.get("mode") {
            set(&mut out, "chain_mode", m.clone());
        }
    }
    if let Some(r) = c.get("color_reanchor").filter(|v| !v.is_null()) {
        let r = match r {
            Value::Bool(true) => json!(from),
            Value::Bool(false) => Value::Null,
            v => v.clone(),
        };
        set(&mut out, "exposure_ref", r);
    }
    for k in c.keys().filter(|k| !["from", "context_frames", "sound", "still", "mode", "color_reanchor"].contains(&k.as_str())) {
        ignored.push(format!("chain.{k}"));
    }
    if let Some(o) = out.as_object_mut() {
        o.remove("chain");
    }
    Ok((out, ignored))
}

/// Request keys the box takes and does nothing with: H3 runs at cfg 1 (one pass, no unconditional branch), so
/// there is nothing for a negative prompt to steer
pub const IGNORED: [&str; 2] = ["negative_prompt", "cfg"];

/// What each denoiser takes (engines.list): the text/image-to-video ones and Ref2VA
/// Keys accepted and listed back in `ignored`: nothing on this box reads them (cfg is 1, a single pass a step; no
/// pin renormalisation)
fn ignored_keys() -> Vec<String> {
    IGNORED.iter().map(|k| k.to_string()).chain(["chain.pin_renorm".to_string()]).collect()
}

/// The common keys the MPEG engine refuses (crates/h3/src/jobs.rs `generate`: what its two-track sampler does not run);
/// references and the ControlNet besides
pub const MPEG_NOT: [&str; 8] = ["ref_audios", "sampler", "schedule", "knots", "speed", "source", "regen", "regen_box"];

/// A queue item for engine MPEG: what its two-track sampler does not run, refused here (a 400) rather than by the
/// engine when the clip starts. Keyframes (first_frame, last_frame, guide_clip) and sound (first_audio, soundtrack)
/// it takes.
fn mpeg_check(it: &Map<String, Value>) -> Result<(), RpcError> {
    let set = |k: &str| !it.get(k).is_none_or(|v| v.is_null() || v.as_str() == Some("") || v.as_array().is_some_and(|a| a.is_empty()));
    if set("ref_images") || set("ref_videos") {
        return Err(RpcError::param("REF2VA is not supported by MPEG yet: ref_images and ref_videos need engine REF2VA"));
    }
    let w = it.get("width").and_then(Value::as_i64).unwrap_or(0);
    let h = it.get("height").and_then(Value::as_i64).unwrap_or(0);
    if w % 64 != 0 || h % 64 != 0 {
        return Err(RpcError::param(&format!("engine MPEG makes the clip at half resolution first: width and height must be multiples of 64 ({w}x{h}; e.g. 1344x768, 768x576, 896x512)")));
    }
    for (k, what) in [("ref_audios", "voice references"), ("control_video", "a ControlNet"), ("control_mask", "a ControlNet mask"), ("source", "a masked run (source)"),
                      ("regen", "a masked run (regen)"), ("knots", "knots")] {
        if set(k) {
            return Err(RpcError::param(&format!("engine MPEG takes no {k} ({what}) yet: INT8 does")));
        }
    }
    let euler = it.get("sampler").and_then(Value::as_str).is_none_or(|s| nextsycl_diffusion::Sampler::parse(s) == Some(nextsycl_diffusion::Sampler::Euler));
    let shift = it.get("schedule").and_then(Value::as_str).is_none_or(|s| nextsycl_diffusion::Schedule::parse(s) == Some(nextsycl_diffusion::Schedule::Shift));
    if !euler || !shift {
        return Err(RpcError::param("engine MPEG runs Euler on H3's own schedule: no other sampler, schedule or speed"));
    }
    Ok(())
}

pub fn inputs_of(quant: &str) -> Value {
    let common = ["prompt", "title", "seconds", "frames", "steps", "seed", "width", "height", "te", "first_frame", "last_frame", "exposure_ref", "guide_clip",
                  "first_audio", "first_audio_s", "soundtrack", "soundtrack_start", "exposure_fix", "exposure_offset", "exposure_ease", "exposure_target", "ref_audios", "loras", "upscale", "upscaler", "shift_video", "shift_audio", "sampler", "schedule", "knots", "speed", "chain", "source", "regen", "regen_box"];
    if quant == "REGEN2K" {
        json!({"takes": ["prompt", "title", "seed", "regen2k", "regen_sampler", "regen_resolution", "sharp"],
               "not": ["width", "height", "seconds", "frames", "steps", "first_frame", "last_frame", "guide_clip", "ref_images", "ref_videos", "ref_audios", "loras",
                       "control_video", "control_mask", "source", "upscale"],
               "chosen": "by itself when regen2k (a draft clip) is given", "ignored": ignored_keys()})
    } else if quant == "MPEG" {
        // MPEG-Diffusion (minimax-h3-mpeg): the INT8 denoiser on two tracks; keyframes, guides and sound condition both
        json!({"takes": common.iter().filter(|k| !MPEG_NOT.contains(k)).collect::<Vec<_>>(),
               "not": MPEG_NOT.iter().chain(["ref_images", "ref_videos", "control_video", "control_mask", "control_source", "control_strength", "control_start",
                                             "control_end"].iter()).collect::<Vec<_>>(),
               "chosen": "by engine: a fast draft (half resolution, then full resolution on the subject the model picks); canvas sides multiples of 64",
               "ignored": ignored_keys()})
    } else if quant == "REF2VA" {
        json!({"takes": common.iter().chain(["ref_images", "ref_videos", "ref_image_size", "ref_video_sound", "te_pictures"].iter()).collect::<Vec<_>>(),
               "not": ["control_video", "control_mask", "control_source", "control_strength", "control_start", "control_end"],
               "chosen": "by itself when ref_images or ref_videos are given", "ignored": ignored_keys()})
    } else {
        json!({"takes": common.iter().chain(["control_video", "control_mask", "control_source", "control_strength", "control_start", "control_end"].iter()).collect::<Vec<_>>(),
               "not": ["ref_images", "ref_videos"], "chosen": if quant == "INT8" { "by default" } else { "by engine" }, "ignored": ignored_keys()})
    }
}

/// A number parameter: the default when absent, rejected (not clamped) outside [lo, hi].
pub fn num(p: &Value, key: &str, default: f64, lo: f64, hi: f64, int: bool) -> Result<Value, RpcError> {
    let v = match p.get(key) {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::Bool(b)) => *b as i64 as f64,
        Some(Value::String(t)) => t.trim().parse::<f64>().map_err(|_| RpcError::param(&format!("{key} must be a number")))?,
        _ => return Err(RpcError::param(&format!("{key} must be a number"))),
    };
    let v = if int { v.trunc() } else { v };
    if !(lo..=hi).contains(&v) {
        return Err(RpcError::param(&format!("{key} must be between {lo} and {hi}")));
    }
    Ok(if int { json!(v as i64) } else { json!(v) })
}

/// A request's denoiser -> its name in the front end's terms: INT8 (the default; the legacy Q8_0 was its counterpart),
/// or a GGUF form the engine was given (Q6_K, Q4_K_M).
pub fn engine_name(e: Option<&str>) -> String {
    match e.map(|e| e.trim().to_uppercase()).filter(|e| !e.is_empty()).as_deref() {
        None | Some("Q8_0") | Some("INT8") => "INT8".into(),
        Some(e) => e.to_string(),
    }
}

/// nextsycl: the front end's denoiser names (INT8, Q6_K, Q4_K_M) and the registry ids of the models the daemon has
/// (`--model`, `--engine`), set once by the studio's start
pub static ENGINES: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();

/// Models the daemon lists that the start did not name (named by their suffix: `quant_of`)
static LEARNED: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

/// The daemon's model ids, as it lists them: any the start did not name join under their suffix's name
pub fn learn_engines(ids: &[String]) {
    let mut l = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    for id in ids {
        let named = ENGINES.get().is_some_and(|m| m.iter().any(|(_, i)| i == id)) || l.iter().any(|(_, i)| i == id);
        let q = quant_of(id).to_string();
        if !named && !ENGINES.get().is_some_and(|m| m.iter().any(|(n, _)| *n == q)) {
            l.push((q, id.clone()));
        }
    }
}

/// Every name -> id pair: the start's, then the learned ones
fn engine_pairs() -> Vec<(String, String)> {
    let mut v = ENGINES.get().cloned().unwrap_or_default();
    v.extend(LEARNED.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned());
    v
}

/// The front end's name for a model id: by its suffix (minimax-h3 -> INT8, -q6k -> Q6_K, -q4km -> Q4_K_M, -q8 -> Q8_0,
/// -ref2va -> REF2VA)
pub fn quant_of(id: &str) -> &'static str {
    // HunyuanVideo-1.5's models (video/hunyuanvideo15): by the whole id, their suffix (q8) is not a denoiser of H3's
    if let Some(rest) = id.strip_prefix("hunyuanvideo-1.5-") {
        return match rest {
            "480p-t2v-q8" => "HV15-CFG",
            "480p-i2v-step-distilled-q8" => "HV15-I2V",
            _ => "HV15",
        };
    }
    // MovieRoll's (films up to 60 s on HunyuanVideo-1.5's code)
    if let Some(rest) = id.strip_prefix("movieroll-") {
        return match rest {
            "distilled-i2v-q8" => "MR-I2V",
            "base-t2v-q8" => "MR-BASE",
            _ => "MR",
        };
    }
    match id.rsplit('-').next().unwrap_or("") {
        "q6k" => "Q6_K",
        "q4km" => "Q4_K_M",
        "q8" => "Q8_0",
        "ref2va" => "REF2VA",
        // minimax-h3-regen2k-ic: Open-H3-Regenerate-2K-IC (the regen2k job)
        "ic" => "REGEN2K",
        // minimax-h3-mpeg: MPEG-Diffusion's two-track sampler (video.generate engine: "MPEG")
        "mpeg" => "MPEG",
        _ => "INT8",
    }
}

/// The front-end names of the models the daemon has (INT8 always)
pub fn engine_names() -> Vec<String> {
    let mut v = vec!["INT8".to_string()];
    for (q, _) in engine_pairs() {
        if !v.contains(&q) {
            v.push(q);
        }
    }
    v
}

/// Whether the daemon has a model of that front-end name
pub fn engine_known(name: &str) -> bool {
    engine_pairs().iter().any(|(q, _)| q == name)
}

/// A front-end name -> the daemon's model id (an unknown name passes as it is)
pub fn engine_id(name: &str) -> String {
    engine_pairs().into_iter().find(|(q, _)| q == name).map_or_else(|| name.to_string(), |(_, id)| id)
}

/// A daemon's model id -> the front end's name
pub fn engine_quant(id: &str) -> String {
    engine_pairs().into_iter().find(|(_, i)| i == id).map_or_else(|| id.to_string(), |(q, _)| q)
}

impl Studio {
    /// A file beside the clips, or of a clip being made (the request then waits for that clip: slots.rs)
    fn here_or_coming(&self, f: &str) -> bool {
        self.out.join(f).exists() || self.running_names.lock().unwrap().iter().any(|n| f.split('.').next() == Some(n.as_str()))
    }

    /// A request -> a queue item, checked (the legacy `build_item`, plus the engine's new options).
    pub fn build_item(&self, p: &Value) -> Result<Value, RpcError> {
        let prompt = p.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return Err(RpcError::param("prompt is required"));
        }
        let te = p.get("te").and_then(|v| v.as_str()).unwrap_or("teacher");
        if te != "teacher" && te != "student" {
            return Err(RpcError::param("te must be teacher or student"));
        }
        // the engine: the model the request names (a registry id the daemon runs), else `engine` (2026-11-23: a
        // video.generate naming hunyuanvideo-1.5-480p-t2v-distilled-q8 ran as H3 - `model` was not read)
        let requested = requested_engine(p, &|id| {
            if engine_id_known(id) {
                return true;
            }
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            engine_id_known(id)
        })?;
        // HunyuanVideo-1.5 (engine HV15...) and MovieRoll (MR...): their own canvas, length and step rules
        let hv = engine_rules(&requested);
        let (dw, dh) = hv.map_or((768.0, 576.0), |r| r.canvas);
        let width = num(p, "width", dw, 256.0, 1344.0, true)?;
        let height = num(p, "height", dh, 256.0, 1344.0, true)?;
        match hv {
            Some(_) => check_canvas_16(width.as_i64().unwrap_or(0), height.as_i64().unwrap_or(0))?,
            None => check_canvas(width.as_i64().unwrap_or(0), height.as_i64().unwrap_or(0))?,
        }
        let chain_mode = p.get("chain_mode").and_then(|v| v.as_str()).unwrap_or("video");
        if !CHAIN_MODES.iter().any(|(m, _)| *m == chain_mode) {
            return Err(RpcError::param("chain_mode must be one of video, png, latent, none"));
        }
        // the sampler and schedule by ComfyUI's names (none: the engine's own Euler on its shifted schedule)
        let named = |k: &str, ok: &dyn Fn(&str) -> bool, names: &str| -> Result<Value, RpcError> {
            match p.get(k).and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
                None => Ok(Value::Null),
                Some(v) if ok(v) => Ok(json!(v)),
                Some(v) => Err(RpcError::param(&format!("{k} {v}? ({names})"))),
            }
        };
        let sampler = named("sampler", &|v| nextsycl_diffusion::Sampler::parse(v).is_some(), &nextsycl_diffusion::samplers::NAMES.join(", "))?;
        let schedule = named("schedule", &|v| nextsycl_diffusion::Schedule::parse(v).is_some(),
                             &nextsycl_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>().join(", "))?;
        let mut it = Map::new();
        it.insert("prompt".into(), json!(prompt));
        it.insert("sampler".into(), sampler);
        it.insert("schedule".into(), schedule);
        // a distilled model's own noise levels: timesteps of 1000, falling, from at most 1000 to 0 (Pika H3 Flex: 1000,750,500,250,100,40,0)
        let knots = match p.get("knots").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
            None => Value::Null,
            Some(v) => {
                let k: Vec<f64> = v.split(',').map(|x| x.trim().parse::<f64>().map_err(|_| RpcError::param("knots: numbers, comma separated"))).collect::<Result<_, _>>()?;
                if !(2..=33).contains(&k.len()) || k.iter().any(|x| !(0.0..=1000.0).contains(x)) || k.windows(2).any(|w| w[1] >= w[0]) || k[k.len() - 1] != 0.0 {
                    return Err(RpcError::param("knots: 2 to 33 falling timesteps of 1000, the last 0 (1000,750,500,250,100,40,0)"));
                }
                json!(v)
            }
        };
        it.insert("knots".into(), knots);
        // frames, when given, set the length exactly (seconds = frames / 24); else seconds snapped up to the grid
        match p.get("frames").filter(|v| !v.is_null()) {
            Some(f) if hv.is_some() => {
                let r = hv.unwrap_or(Rules { steps: (0.0, 0.0), frames: HV15_FRAMES, grid: 4, canvas: (0.0, 0.0) });
                let f = f.as_u64().filter(|f| f % r.grid == 1 && (r.frames.0..=r.frames.1).contains(f))
                    .ok_or_else(|| RpcError::param(&format!("frames: engine {requested} makes {}k + 1 frames ({}-{})", r.grid, r.frames.0, r.frames.1)))?;
                it.insert("seconds".into(), json!(f as f64 / 24.0));
                it.insert("frames".into(), json!(f));
            }
            None if hv.is_some() => {
                let r = hv.unwrap_or(Rules { steps: (0.0, 0.0), frames: HV15_FRAMES, grid: 4, canvas: (0.0, 0.0) });
                let sec = num(p, "seconds", if r.grid == 16 { 10.0 } else { 5.0 }, 0.2, r.frames.1 as f64 / 24.0, false)?;
                let f = grid_frames(sec.as_f64().unwrap_or(5.0), r.grid, r.frames.1);
                it.insert("frames".into(), json!(f));
                it.insert("seconds".into(), json!(f as f64 / 24.0));
            }
            Some(f) => {
                let f = check_frames(f.as_u64().ok_or_else(|| RpcError::param("frames: a whole number"))?)?;
                it.insert("seconds".into(), json!(f as f64 / 24.0));
                it.insert("frames".into(), json!(f));
            }
            None => {
                let sec = num(p, "seconds", 10.0, 1.0, 15.1, false)?;
                it.insert("frames".into(), json!(super::runner::frames_of(sec.as_f64().unwrap_or(10.0))));
                it.insert("seconds".into(), sec);
            }
        }
        // MovieRoll's video.roll: the film's events (each its seconds), live (it waits for more), its HLS playlist; the
        // length is the events' (whole chunks)
        if p.get("events").is_some_and(|v| !v.is_null()) {
            if !requested.starts_with("MR") {
                return Err(RpcError::param("events: a MovieRoll film's (video.roll, a movieroll model)"));
            }
            let ev = roll_events(p.get("events"))?;
            // (no limit on their sum: 64 events of 60 s at most, and the engine's memory does not grow with the film)
            let chunks: u64 = ev.iter().map(|e| ((e["seconds"].as_f64().unwrap_or(10.0) * 24.0 / 16.0).round() as u64).max(1)).sum();
            let f = 16 * chunks + 1;
            it.insert("frames".into(), json!(f));
            it.insert("seconds".into(), json!(f as f64 / 24.0));
            it.insert("events".into(), json!(ev));
            it.insert("live".into(), json!(p["live"] == true));
            // an hour at most: a live film waiting for its next event holds its card (nothing else runs there meanwhile)
            it.insert("max_idle_s".into(), num(p, "max_idle_s", 300.0, 0.0, 3600.0, false)?);
            it.insert("hls".into(), json!(true));
        }
        it.insert("reentrant_id".into(), reentrant(p).map_or(Value::Null, |r| json!(r)));
        // the clip's place in its project's edit (1 and up): the timeline's n, stated rather than read from the label
        match p.get("clip").filter(|v| !v.is_null()) {
            Some(c) => match c.as_u64().filter(|c| (1..=100_000).contains(c)) {
                Some(c) => it.insert("clip".into(), json!(c)),
                None => return Err(RpcError::param("clip: a whole number from 1 (the clip's place in its project's timeline)")),
            },
            None => it.insert("clip".into(), Value::Null),
        };
        let (dsteps, max_steps) = hv.map_or((10.0, 40.0), |r| r.steps);
        it.insert("steps".into(), num(p, "steps", dsteps, 1.0, max_steps, true)?);
        it.insert("seed".into(), num(p, "seed", 0.0, -2147483648.0, 2147483648.0, true)?);
        it.insert("te".into(), json!(te));
        it.insert("title".into(), title_of(p));
        it.insert("label".into(), json!(clean_label(&p.get("label").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default())));
        let engine = requested;
        if engine != "INT8" && !engine_known(&engine) {
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            if !engine_known(&engine) {
                return Err(RpcError::param(&format!("engine {engine}? (video.engines.list: {})", engine_names().join(", "))));
            }
        }
        it.insert("engine".into(), json!(engine));
        it.insert("priority".into(), num(p, "priority", 0.0, -9.0, 9.0, true)?);
        it.insert("width".into(), width);
        it.insert("height".into(), height);
        it.insert("chain_mode".into(), json!(chain_mode));
        for k in ["cond_noise_aug", "camera", "exposure_ref", "last_frame", "first_audio", "first_audio_s", "upscale", "upscaler", "guide_clip", "shift_video",
                  "shift_audio", "source", "regen", "regen_box"] {
            it.insert(k.into(), p.get(k).cloned().unwrap_or(Value::Null));
        }
        for k in ["project", "batch"] {
            let v = p.get(k).and_then(|v| v.as_str()).unwrap_or("");
            it.insert(k.into(), json!(v.chars().take(60).collect::<String>().trim()));
        }
        let loras: Vec<String> = p.get("loras").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()).chars().take(200).collect()).collect()).unwrap_or_default();
        it.insert("loras".into(), json!(loras));
        // speed: the model's own six-step mode (Pika H3 Flex): its adapter, the lcm sampler on its knots and six steps, whatever
        // sampler, knots, steps or other LoRAs were also named (those stay beside it)
        match p.get("speed").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
            None | Some("standard") => {}
            Some("pika-flex") => {
                let paths: Vec<String> = self.engines().iter().filter_map(|e| e["path"].as_str().map(str::to_string)).collect();
                let adapter = pika_flex_lora(&paths).ok_or_else(|| RpcError::param("speed pika-flex: the Pika H3 Flex adapter is not on this box (nextsycl models pull pika-h3-flex-fl2va)"))?;
                it.insert("sampler".into(), json!("lcm"));
                it.insert("knots".into(), json!(PIKA_FLEX_KNOTS));
                it.insert("steps".into(), json!(6));
                let mut ls = vec![format!("{adapter}:1.0")];
                ls.extend(loras);
                it.insert("loras".into(), json!(ls));
            }
            Some(v) => return Err(RpcError::param(&format!("speed {v}? (standard, pika-flex)"))),
        }
        // a first frame that is not an anchor word is a file in out/
        let ff = p.get("first_frame").and_then(|v| v.as_str()).unwrap_or("");
        let ff = if ff.is_empty() {
            Value::Null
        } else if MAGIC.contains(&ff) {
            json!(ff)
        } else {
            let b = basename(ff);
            if !self.here_or_coming(&b) {
                return Err(RpcError::not_found(&format!("first_frame {b} not found")));
            }
            json!(b)
        };
        it.insert("first_frame".into(), ff);
        for (k, max, what) in [("ref_images", 9, "ref_image"), ("ref_videos", 3, "ref_video"), ("ref_audios", 3, "ref_audio")] {
            let list: Vec<String> = p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).map(basename).collect()).unwrap_or_default();
            if list.len() > max {
                return Err(RpcError::param(&format!("{k}: at most {max} ({} given)", list.len())));
            }
            for f in &list {
                if !self.here_or_coming(f) {
                    return Err(RpcError::not_found(&format!("{what} {f} not found in out/")));
                }
            }
            it.insert(k.into(), if list.is_empty() { Value::Null } else { json!(list) });
        }
        // pictures and clips as references: the Ref2VA denoiser, the one trained for them (not when MPEG is asked for: its
        // two-track sampler runs the FL2VA denoiser)
        if it["ref_images"].is_array() || it["ref_videos"].is_array() {
            if it["engine"] == "MPEG" {
                return Err(RpcError::param("REF2VA is not supported by MPEG yet: ref_images and ref_videos need engine REF2VA (or INT8 to have it chosen); drop them for MPEG"));
            }
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            if !engine_known("REF2VA") {
                return Err(RpcError::param("reference pictures and clips need the Ref2VA denoiser (nextsycl models pull minimax-h3-ref2va, then add it to NS_VIDEO_ENGINES)"));
            }
            it.insert("engine".into(), json!("REF2VA"));
        }
        for k in ["ref_video_sound", "te_pictures"] {
            it.insert(k.into(), p.get(k).filter(|v| v.is_boolean()).cloned().unwrap_or(Value::Null));
        }
        // Open-H3-Regenerate-2K-IC: a draft clip made again at 2560x1440 (or 1080p) - the Regenerate-2K IC model, the
        // prompt as the draft's caption; the draft sets the length, the resolution the canvas
        match p.get("regen2k").and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
            Some(d) => {
                let _ = self.engines();
                if !engine_known("REGEN2K") {
                    return Err(RpcError::param("regen2k needs the Regenerate-2K IC model (nextsycl models pull minimax-h3-regen2k-ic, then add it to NS_VIDEO_ENGINES)"));
                }
                if it["ref_images"].is_array() || it["ref_videos"].is_array() || p.get("source").is_some_and(|v| !v.is_null()) {
                    return Err(RpcError::param("regen2k takes a draft and its caption: no references or source beside it"));
                }
                // the draft: its length sets the clip's (all of it is made again, in windows past 124 frames), its shape must
                // be landscape 16:9 (it is cover-fitted to 1344x768); a draft being made or an anchor is checked when it runs
                let mut real = REGEN_WINDOW;
                let d = if MAGIC.contains(&d) {
                    d.to_string()
                } else {
                    // a finished clip by its bare name too (clips are .mp4): h3_20261113_150714 is h3_20261113_150714.mp4
                    let mut b = basename(d);
                    if Path::new(&b).extension().is_none() {
                        b.push_str(".mp4");
                    }
                    if !self.here_or_coming(&b) {
                        return Err(RpcError::not_found(&format!("regen2k draft {b} not found in out/")));
                    }
                    if self.out.join(&b).exists() {
                        let (w, h, secs) = probe(&self.out.join(&b)).ok_or_else(|| RpcError::param(&format!("regen2k draft {b}: not a video ffprobe can read")))?;
                        real = check_draft(&b, w, h, secs)?;
                    }
                    b
                };
                let pick = |k: &str, ok: &[&str], dflt: &str| -> Result<String, RpcError> {
                    match p.get(k).and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
                        None => Ok(dflt.to_string()),
                        Some(v) if ok.contains(&v) => Ok(v.to_string()),
                        Some(v) => Err(RpcError::param(&format!("{k} {v}? ({})", ok.join(", ")))),
                    }
                };
                let sampler = pick("regen_sampler", &["up3", "up4", "hf8"], "up3")?;
                let res = pick("regen_resolution", &["2k", "1080p"], "2k")?;
                let (w, h) = if res == "1080p" { (1920, 1080) } else { (2560, 1440) };
                it.insert("regen2k".into(), json!(d));
                it.insert("regen_sampler".into(), json!(sampler));
                it.insert("regen_resolution".into(), json!(res));
                it.insert("sharp".into(), num(p, "sharp", 1.0, 0.0, 2.0, false)?);
                it.insert("engine".into(), json!("REGEN2K"));
                it.insert("width".into(), json!(w));
                it.insert("height".into(), json!(h));
                // steps: every window's (the status counts them: "sampling 4/6" over two windows)
                let per = match sampler.as_str() { "up4" => 4, "hf8" => 8, _ => 3 };
                it.insert("frames".into(), json!(real));
                it.insert("seconds".into(), json!(real as f64 / 24.0));
                it.insert("regen_windows".into(), json!(regen_windows(real)));
                it.insert("steps".into(), json!(per * regen_windows(real)));
            }
            None => {
                for k in ["regen2k", "regen_sampler", "regen_resolution", "sharp"] {
                    it.insert(k.into(), Value::Null);
                }
            }
        }
        // the ControlNet: a control video, a mask over a source; files beside the clips
        for k in ["control_video", "control_mask", "control_source"] {
            let v = p.get(k).and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty());
            // a clip named without its extension (video.control's `file`, a video.generate clip's name) is its .mp4
            let v = v.map(|f| with_mp4(&f, |n| self.here_or_coming(n)));
            if let Some(f) = &v {
                if !self.here_or_coming(f) {
                    return Err(RpcError::not_found(&format!("{k} {f} not found in out/")));
                }
            }
            it.insert(k.into(), v.map_or(Value::Null, |f| json!(f)));
        }
        // a driving soundtrack: a song beside the clips (video.upload)
        match p.get("soundtrack").and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty()) {
            Some(f) => {
                if !self.out.join(&f).exists() {
                    return Err(RpcError::not_found(&format!("soundtrack {f} not found in out/ (video.upload it)")));
                }
                if p.get("first_audio").is_some_and(|v| !v.is_null()) {
                    return Err(RpcError::param("soundtrack is the clip's whole sound: not with first_audio"));
                }
                it.insert("soundtrack".into(), json!(f));
                it.insert("soundtrack_start".into(), num(p, "soundtrack_start", 0.0, 0.0, 86400.0, false)?);
            }
            None => {
                it.insert("soundtrack".into(), Value::Null);
                it.insert("soundtrack_start".into(), Value::Null);
            }
        }
        // the exposure fix after the decode: off unless asked for; its settings only with it
        if p.get("exposure_fix").and_then(|v| v.as_bool()) == Some(true) {
            it.insert("exposure_fix".into(), json!(true));
            it.insert("exposure_offset".into(), num(p, "exposure_offset", 2.3, 0.0, 20.0, false)?);
            it.insert("exposure_ease".into(), num(p, "exposure_ease", 24.0, 0.0, 240.0, true)?);
            it.insert("exposure_target".into(), if p.get("exposure_target").is_some_and(|v| !v.is_null()) { num(p, "exposure_target", 0.0, 0.0, 255.0, false)? } else { Value::Null });
        } else {
            for k in ["exposure_fix", "exposure_offset", "exposure_ease", "exposure_target"] {
                it.insert(k.into(), Value::Null);
            }
        }
        if !it["control_video"].is_null() || !it["control_mask"].is_null() {
            if it["engine"] == "REF2VA" {
                return Err(RpcError::param("the ControlNet steers the text/image-to-video denoiser, not Ref2VA: references and control do not mix"));
            }
            it.insert("control_strength".into(), num(p, "control_strength", 1.0, 0.0, 10.0, false)?);
            it.insert("control_start".into(), num(p, "control_start", 0.0, 0.0, 1.0, false)?);
            it.insert("control_end".into(), num(p, "control_end", 1.0, 0.0, 1.0, false)?);
        }
        let ris = p.get("ref_image_size").and_then(|v| v.as_str()).filter(|v| *v == "match" || *v == "max");
        it.insert("ref_image_size".into(), ris.map_or(Value::Null, |v| json!(v)));
        if it["engine"] == "MPEG" {
            mpeg_check(&it)?;
        }
        Ok(Value::Object(it))
    }
}

/// Open-H3-Regenerate-2K-IC's limits (crates/h3/src/regen.rs: FRAMES, MIN_REAL, MAX_FRAMES, ASPECT_SLACK): its window is
/// the model's 124-frame clip; a draft needs 113 frames at least, takes 362 at most, and is landscape 16:9 within 10%
pub const REGEN_WINDOW: usize = 124;
const REGEN_MIN: usize = 113;
const REGEN_MAX: usize = 362;

/// A clip's width, height and seconds (ffprobe; the daemon's container's when the host's cannot read H.264)
fn probe(p: &Path) -> Option<(f64, f64, f64)> {
    let o = crate::video::media::tool("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of",
                                                       "default=noprint_wrappers=1"]).arg(p).output().ok()?;
    let text = String::from_utf8_lossy(&o.stdout);
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).and_then(|v| v.trim().parse::<f64>().ok());
    Some((get("width")?, get("height")?, get("duration")?))
}

/// A draft's frames at 24 fps, or why it cannot be regenerated (a 400 naming the limit)
fn check_draft(name: &str, w: f64, h: f64, secs: f64) -> Result<usize, RpcError> {
    let aspect = (w / h) / (16.0 / 9.0);
    if (aspect - 1.0).abs() > 0.10 {
        return Err(RpcError::param(&format!("regen2k draft {name} is {w}x{h}: Open-H3-Regenerate-2K takes landscape 16:9 drafts (cover-fitted to 1344x768; this one would lose too much of its picture)")));
    }
    let frames = (secs * 24.0).round() as usize;
    if frames < REGEN_MIN {
        return Err(RpcError::param(&format!("regen2k draft {name} is {secs:.2} s ({frames} frames at 24 fps): the model needs {REGEN_MIN} at least (4.7 s)")));
    }
    if frames > REGEN_MAX {
        return Err(RpcError::param(&format!("regen2k draft {name} is {secs:.2} s ({frames} frames at 24 fps): at most {REGEN_MAX} ({:.1} s) - cut it first", REGEN_MAX as f64 / 24.0)));
    }
    Ok(frames)
}

/// The windows a draft of `frames` frames is made in: one per 124-frame clip of the model, overlapping by 2 latent frames
/// at least (crates/h3/src/regen.rs `grid`, `windows`: the same count)
pub fn regen_windows(frames: usize) -> usize {
    let mut n = frames.max(REGEN_WINDOW);
    while n % 17 != 5 {
        n += 1;
    }
    let lt = (n - 5) / 17 * 5 + 2;
    if lt <= 37 { 1 } else { (lt - 2).div_ceil(35) }
}

/// A request's reentrant_id (reentrantid is read too)
pub fn reentrant(p: &Value) -> Option<String> {
    ["reentrant_id", "reentrantid"].iter().find_map(|k| p.get(*k).and_then(|v| v.as_str())).map(str::trim).filter(|r| !r.is_empty()).map(|r| r.chars().take(200).collect())
}

/// `f`, or `f.mp4` when `f` has no extension, is not a file and the .mp4 is (`exists` asks)
pub fn with_mp4(f: &str, exists: impl Fn(&str) -> bool) -> String {
    if !f.contains('.') && !exists(f) && exists(&format!("{f}.mp4")) {
        format!("{f}.mp4")
    } else {
        f.to_string()
    }
}

pub fn basename(p: &str) -> String {
    Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
}

impl Queue {
    /// Where an item of this priority goes: after every item of the same or a higher one
    pub fn slot(&self, priority: i64) -> usize {
        self.items.iter().position(|i| i["priority"].as_i64().unwrap_or(0) < priority).unwrap_or(self.items.len())
    }

    pub fn held(&self, item: &Value) -> bool {
        let p = project_of(item);
        let b = s(item, "batch");
        self.paused || !self.held_by.is_empty() || (!p.is_empty() && self.paused_projects.contains(&p)) || (!b.is_empty() && self.paused_batches.contains(&b))
    }

    /// A pause or a resume by scope: a named hold (`by`: only that one), a batch, a project, or the whole queue (a
    /// whole-queue resume lifts everything, the named holds too - a person's word wins); the scope's name
    pub fn hold(&mut self, on: bool, project: &str, batch: &str, by: &str) -> String {
        if !by.is_empty() {
            if on { self.held_by.insert(by.into()) } else { self.held_by.remove(by) };
            format!("hold {by}")
        } else if !batch.is_empty() {
            if on { self.paused_batches.insert(batch.into()) } else { self.paused_batches.remove(batch) };
            format!("batch {batch}")
        } else if !project.is_empty() {
            if on { self.paused_projects.insert(project.into()) } else { self.paused_projects.remove(project) };
            format!("project {project}")
        } else {
            self.paused = on;
            if !on {
                self.paused_projects.clear();
                self.paused_batches.clear();
                self.held_by.clear();
            }
            "queue".into()
        }
    }

    pub fn load(dir: &Path) -> Queue {
        let items = std::fs::read(dir.join("queue.json")).ok().and_then(|b| serde_json::from_slice::<Vec<Value>>(&b).ok()).unwrap_or_default();
        let p: Value = std::fs::read(dir.join("paused.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let set = |k: &str| -> BTreeSet<String> { p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        Queue { items, paused: p.get("all").and_then(|v| v.as_bool()).unwrap_or(false), paused_projects: set("projects"), paused_batches: set("batches"), held_by: set("by") }
    }

    pub fn save(&self, dir: &Path) {
        let _ = super::write_atomic(&dir.join("queue.json"), &serde_json::to_vec(&self.items).unwrap_or_default());
    }

    pub fn save_paused(&self, dir: &Path) {
        let v = json!({"all": self.paused, "projects": self.paused_projects, "batches": self.paused_batches, "by": self.held_by});
        let _ = super::write_atomic(&dir.join("paused.json"), &serde_json::to_vec(&v).unwrap_or_default());
    }

    /// The queued projects in queue order, with their batches.
    pub fn projects(&self) -> Vec<Value> {
        let mut order: Vec<String> = Vec::new();
        for it in &self.items {
            let p = project_of(it);
            if !order.contains(&p) {
                order.push(p);
            }
        }
        order
            .iter()
            .map(|p| {
                let mine: Vec<&Value> = self.items.iter().filter(|i| project_of(i) == *p).collect();
                let first = self.items.iter().position(|i| project_of(i) == *p).unwrap_or(0);
                let mut batches: Vec<String> = mine.iter().map(|i| s(i, "batch")).collect();
                batches.sort();
                batches.dedup();
                let secs = |v: &[&Value]| (v.iter().map(|i| i["seconds"].as_f64().unwrap_or(0.0)).sum::<f64>() * 100.0).round() / 100.0;
                json!({"project": p, "clips": mine.len(), "video_s": secs(&mine), "paused": self.paused_projects.contains(p), "first_index": first,
                       "batches": batches.iter().map(|b| {
                           let bi: Vec<&Value> = mine.iter().copied().filter(|i| s(i, "batch") == *b).collect();
                           json!({"batch": b, "clips": bi.len(), "video_s": secs(&bi), "paused": self.paused_batches.contains(b)})
                       }).collect::<Vec<_>>()})
            })
            .collect()
    }

    pub fn payload(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("queue".into(), json!(self.items.iter().map(|i| { let l = s(i, "label"); if l.is_empty() { s(i, "prompt").chars().take(40).collect() } else { l } }).collect::<Vec<_>>()));
        m.insert("queue_items".into(), json!(self.items.iter().map(|i| json!({
            "label": s(i, "label"), "title": i["title"], "seconds": i["seconds"], "steps": i["steps"], "engine": i["engine"], "chain": !i["first_frame"].is_null(),
            "size": format!("{}x{}", i["width"].as_i64().unwrap_or(640), i["height"].as_i64().unwrap_or(480)),
            "project": project_of(i), "batch": s(i, "batch"), "held": self.held(i)})).collect::<Vec<_>>()));
        m.insert("paused".into(), json!(self.paused));
        m.insert("paused_projects".into(), json!(self.paused_projects));
        m.insert("paused_batches".into(), json!(self.paused_batches));
        m.insert("held_by".into(), json!(self.held_by));
        m.insert("projects".into(), json!(self.projects()));
        m
    }
}

/// The six-step mode of MiniMax H3 (Pika H3 Flex): the adapter (model-cards/lora/pika-h3-flex-fl2va.yml) and the noise
/// levels it was post-trained on
const PIKA_FLEX_LORA_FILE: &str = "pika_h3_flex_fl2va_comfy_bf16.safetensors";
const PIKA_FLEX_KNOTS: &str = "1000,750,500,250,100,40,0";

/// Where the adapter is on this box: in the `loras` folder next to the denoisers' `diffusion_models`
fn pika_flex_lora(denoisers: &[String]) -> Option<String> {
    denoisers.iter().find_map(|p| {
        let f = std::path::Path::new(p).parent()?.parent()?.join("loras").join(PIKA_FLEX_LORA_FILE);
        f.is_file().then(|| f.display().to_string())
    })
}

#[cfg(test)]
mod tests {
    /// A video.generate naming a HunyuanVideo-1.5 model reaches its engine and that engine's rules (1 s: 25 frames, not
    /// H3's 39); an unknown model is refused, not made by H3; a model and an engine that disagree are refused
    #[test]
    fn a_clip_naming_a_model_takes_its_engine_and_rules() {
        use super::*;
        let ids = ["hunyuanvideo-1.5-480p-t2v-distilled-q8", "hunyuanvideo-1.5-480p-t2v-q8", "hunyuanvideo-1.5-480p-i2v-step-distilled-q8"];
        learn_engines(&ids.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let known = |id: &str| engine_id_known(id);
        for (id, want) in ids.iter().zip(["HV15", "HV15-CFG", "HV15-I2V"]) {
            let p = json!({"model": id, "prompt": "a fox", "seconds": 1});
            let e = requested_engine(&p, &known).unwrap();
            assert_eq!(e, want, "{id}");
            assert!(hv15_rules(&e).is_some());
            assert_eq!(hv15_frames(1.0), 25);
            assert!(engine_known(&e));
            assert_eq!(engine_id(&e), *id);
        }
        // no model: engine as before, H3's INT8 by default
        assert_eq!(requested_engine(&json!({"prompt": "x"}), &known).unwrap(), "INT8");
        assert_eq!(requested_engine(&json!({"engine": "HV15"}), &known).unwrap(), "HV15");
        // an unknown model: refused
        let e = requested_engine(&json!({"model": "no-such-video-model"}), &known).unwrap_err();
        assert!(format!("{e:?}").contains("no-such-video-model"), "{e:?}");
        // disagreeing
        assert!(requested_engine(&json!({"model": ids[0], "engine": "INT8"}), &known).is_err());
        assert!(requested_engine(&json!({"model": ids[0], "engine": "HV15"}), &known).is_ok());
    }

    /// HunyuanVideo-1.5's models by id, and its rules: 4k + 1 frames, its steps, sides of 16
    #[test]
    fn hunyuanvideo_engines_have_their_own_rules() {
        use super::*;
        assert_eq!(quant_of("hunyuanvideo-1.5-480p-t2v-distilled-q8"), "HV15");
        assert_eq!(quant_of("hunyuanvideo-1.5-480p-t2v-q8"), "HV15-CFG");
        assert_eq!(quant_of("hunyuanvideo-1.5-480p-i2v-step-distilled-q8"), "HV15-I2V");
        assert_eq!(quant_of("minimax-h3-q6k"), "Q6_K");
        assert_eq!(hv15_rules("HV15"), Some((30.0, 100.0)));
        assert_eq!(hv15_rules("HV15-CFG"), Some((50.0, 100.0)));
        assert_eq!(hv15_rules("HV15-I2V"), Some((8.0, 50.0)));
        assert_eq!(hv15_rules("INT8"), None);
        assert_eq!(hv15_frames(5.0), 121);
        assert_eq!(hv15_frames(1.0), 25);
        assert!(check_canvas_16(848, 480).is_ok() && check_canvas_16(850, 480).is_err());
        assert_eq!(engine_rules("HV15").map(|r| r.grid), Some(4));
    }

    /// MovieRoll's models by id, and its rules: 16k + 1 frames up to 60 s, its fixed 5 steps (base 50), 864 x 480
    #[test]
    fn roll_events_are_checked() {
        use super::*;
        let ok = roll_events(Some(&json!([{"prompt": "a fox", "seconds": 10}, {"prompt": " sits ", "seconds": 5}]))).unwrap();
        assert_eq!(ok, vec![json!({"prompt": "a fox", "seconds": 10.0}), json!({"prompt": "sits", "seconds": 5.0})]);
        assert!(roll_events(Some(&json!([]))).is_err());
        assert!(roll_events(Some(&json!([{"prompt": "", "seconds": 5}]))).is_err());
        assert!(roll_events(Some(&json!([{"prompt": "x", "seconds": 61}]))).is_err());
        assert!(roll_events(None).is_err());
    }

    #[test]
    fn movieroll_engines_have_their_own_rules() {
        use super::*;
        assert_eq!(quant_of("movieroll-distilled-t2v-q8"), "MR");
        assert_eq!(quant_of("movieroll-distilled-i2v-q8"), "MR-I2V");
        assert_eq!(quant_of("movieroll-base-t2v-q8"), "MR-BASE");
        let r = engine_rules("MR").unwrap();
        assert_eq!((r.steps, r.frames, r.grid, r.canvas), ((5.0, 5.0), (17, 1441), 16, (864.0, 480.0)));
        assert_eq!(engine_rules("MR-BASE").unwrap().steps, (50.0, 100.0));
        assert_eq!(grid_frames(10.0, 16, 1441), 241);
        assert_eq!(grid_frames(60.0, 16, 1441), 1441);
        assert_eq!(grid_frames(9.0, 16, 1441), 225);
        assert_eq!(grid_frames(5.0, 4, 241), 121);
        assert_eq!(grid_frames(90.0, 16, 1441), 1441);
        assert!(engine_rules("INT8").is_none());
    }

    /// A regen2k draft: the platform's 864x480 10.125 s clip is two windows, all of it; a portrait one, a 3 s one and a
    /// 20 s one are refused, saying why
    /// Engine MPEG takes the singer film's keyframes and sound, and refuses references with a message naming REF2VA
    #[test]
    fn mpeg_takes_keyframes_and_sound_but_not_references() {
        let it = |extra: Value| -> Map<String, Value> {
            let mut m = json!({"engine": "MPEG", "width": 1344, "height": 768, "guide_clip": "h3_a.mp4:22:0,anchor.png", "first_frame": "a.png",
                               "last_frame": "b.png", "soundtrack": "song.flac", "first_audio": null, "sampler": null, "schedule": null, "knots": null,
                               "ref_images": null, "ref_audios": null}).as_object().unwrap().clone();
            m.extend(extra.as_object().unwrap().clone());
            m
        };
        assert!(super::mpeg_check(&it(json!({}))).is_ok());
        assert!(super::mpeg_check(&it(json!({"sampler": "euler", "schedule": "shift"}))).is_ok());
        let e = super::mpeg_check(&it(json!({"ref_images": ["p.png"]}))).unwrap_err().message;
        assert!(e.contains("REF2VA is not supported by MPEG yet"), "{e}");
        assert!(super::mpeg_check(&it(json!({"width": 864, "height": 480}))).unwrap_err().message.contains("multiples of 64"));
        assert!(super::mpeg_check(&it(json!({"sampler": "lcm"}))).is_err());
        assert!(super::mpeg_check(&it(json!({"control_video": "c.mp4"}))).is_err());
        let i = inputs_of("MPEG");
        let has = |k: &str, l: &str| i[l].as_array().unwrap().iter().any(|v| v == k);
        for k in ["first_frame", "last_frame", "guide_clip", "first_audio", "first_audio_s", "soundtrack", "soundtrack_start"] {
            assert!(has(k, "takes") && !has(k, "not"), "{k}");
        }
        for k in ["ref_images", "ref_videos", "control_video", "ref_audios"] {
            assert!(has(k, "not") && !has(k, "takes"), "{k}");
        }
    }

    #[test]
    fn a_regen_draft_is_taken_whole_or_refused() {
        assert_eq!(super::check_draft("a", 864.0, 480.0, 10.125).ok(), Some(243));
        assert_eq!(super::regen_windows(243), 2);
        assert_eq!(super::regen_windows(124), 1);
        assert_eq!(super::regen_windows(113), 1);
        assert_eq!(super::regen_windows(362), 3);
        assert_eq!(super::check_draft("a", 1344.0, 768.0, 5.17).ok(), Some(124));
        assert!(super::check_draft("a", 480.0, 864.0, 10.0).is_err());
        assert!(super::check_draft("a", 1344.0, 768.0, 3.0).is_err());
        assert!(super::check_draft("a", 1344.0, 768.0, 20.0).is_err());
    }

    #[test]
    fn the_pika_flex_adapter_is_found_beside_the_denoisers() {
        let root = std::env::temp_dir().join(format!("ns-pika-flex-{}", std::process::id()));
        std::fs::create_dir_all(root.join("diffusion_models")).unwrap();
        std::fs::create_dir_all(root.join("loras")).unwrap();
        let denoiser = root.join("diffusion_models/h3.gguf").display().to_string();
        assert_eq!(super::pika_flex_lora(std::slice::from_ref(&denoiser)), None);
        let file = root.join("loras").join(super::PIKA_FLEX_LORA_FILE);
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(super::pika_flex_lora(&["/nowhere/a/b.gguf".into(), denoiser]), Some(file.display().to_string()));
        std::fs::remove_dir_all(&root).unwrap();
    }

    use super::*;

    #[test]
    fn a_control_clip_may_be_named_without_its_extension() {
        let have = |n: &str| ["ctl_1_depth.mp4", "plain", "clip.webm"].contains(&n);
        assert_eq!(with_mp4("ctl_1_depth", have), "ctl_1_depth.mp4");
        assert_eq!(with_mp4("ctl_1_depth.mp4", have), "ctl_1_depth.mp4");
        // a file that is there as named, an extension of another kind, or nothing there: as it came
        assert_eq!(with_mp4("plain", have), "plain");
        assert_eq!(with_mp4("clip.webm", have), "clip.webm");
        assert_eq!(with_mp4("gone", have), "gone");
    }

    #[test]
    fn a_title_is_one_line_of_120_at_most_and_none_is_null() {
        assert_eq!(title_of(&json!({"title": "  Blackbird (Mach 3.2) - shots - pass 12 of 24 "})), json!("Blackbird (Mach 3.2) - shots - pass 12 of 24"));
        assert_eq!(title_of(&json!({"title": "a\nb"})), json!("a b"));
        assert_eq!(title_of(&json!({"title": "é".repeat(200)})).as_str().unwrap().chars().count(), 120);
        for none in [json!({}), json!({"title": "  "}), json!({"title": 7})] {
            assert_eq!(title_of(&none), Value::Null);
        }
    }

    #[test]
    fn a_named_hold_lifts_only_itself_and_never_sticks() {
        let it = json!({"label": "x", "project": "P"});
        let mut q = Queue::default();
        // 2026-11-03: maintenance on/off around each test run, a GPU run's hold between - the queue left held for a day
        for _ in 0..3 {
            q.hold(true, "", "", "gpu run");
            q.hold(true, "", "", "maintenance");
            q.hold(false, "", "", "gpu run");
            assert!(q.held(&it), "maintenance still holds it");
            q.hold(false, "", "", "maintenance");
            assert!(!q.held(&it), "every hold lifted: the queue goes on");
        }
        // a person's pause stands through a hold's on and off
        q.hold(true, "", "", "");
        q.hold(true, "", "", "maintenance");
        q.hold(false, "", "", "maintenance");
        assert!(q.held(&it) && q.paused);
        // and a person's resume of the whole queue lifts every hold
        q.hold(true, "", "", "gpu run");
        q.hold(true, "P", "", "");
        q.hold(false, "", "", "");
        assert!(!q.held(&it) && q.held_by.is_empty() && q.paused_projects.is_empty());
        // kept across a restart
        let dir = std::env::temp_dir().join(format!("ns-holds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        q.hold(true, "", "", "maintenance");
        q.save_paused(&dir);
        assert!(Queue::load(&dir).held(&it));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn labels() {
        assert_eq!(label_prefix("Harbour Night 03/12: the lamp"), "Harbour Night");
        assert_eq!(label_number("Harbour Night 03/12: the lamp"), Some(3));
        assert_eq!(label_number("no number"), None);
        assert_eq!(project_of(&json!({"label": "Speech 07/20: words"})), "Speech");
        assert_eq!(project_of(&json!({"label": "x", "project": "P"})), "P");
        assert_eq!(dialogue_of("says: <d>[English] First   batch.</d> then <d>[English] More.</d>"), "First batch. / More.");
        assert_eq!(clean_label("a<b>c 01/02: ok"), "abc 01/02: ok");
    }

    #[test]
    fn the_creative_platforms_clips_are_numbered_by_their_pass() {
        let c = |label: &str, id: Value| json!({"label": label, "reentrant_id": id});
        // labels without "NN/MM": the pass after the # (all were 0, so a timeline kept one clip of the project)
        assert_eq!(clip_number(&c("a2c6221e-bde2:shots:0:016", json!("a2c6221e-bde2:shots:0:0#16"))), 16);
        assert_eq!(clip_number(&c("a2c6221e-bde2:shots:0:017", json!("a2c6221e-bde2:shots:0:0#17"))), 17);
        assert_eq!(clip_number(&c("Name 03/12: x", json!("r#9"))), 3, "a label's number comes first");
        assert_eq!(clip_number(&c("cp-smoke:shots:0:0", Value::Null)), 0);
        assert_eq!(clip_number(&c("x", json!("r#x1"))), 0);
        // stated: it wins over both
        assert_eq!(clip_number(&json!({"clip": 5, "label": "Name 03/12: x", "reentrant_id": "r#9"})), 5);
        assert_eq!(clip_number(&json!({"clip": null, "label": "Name 03/12: x"})), 3);
    }

    #[test]
    fn frames_and_chains() {
        assert_eq!(check_frames(107).unwrap(), 107);
        let e = check_frames(97).unwrap_err().message;
        assert!(e.contains("90 or 107"), "{e}");
        assert!(check_frames(400).unwrap_err().message.contains("nearest: 362"));
        assert!(check_frames(10).unwrap_err().message.contains("nearest: 22"));
        let (p, ig) = unchain(&json!({"prompt": "x", "chain": {"from": "prev", "context_frames": 22, "color_reanchor": "first", "pin_renorm": true}})).unwrap();
        assert_eq!(p["guide_clip"], "prev:22:0");
        assert_eq!(p["first_audio"], "prev");
        assert!((p["first_audio_s"].as_f64().unwrap() - 22.0 / 24.0).abs() < 1e-9);
        assert!(p["first_frame"].is_null() && p.get("chain").is_none());
        assert_eq!(p["exposure_ref"], "first");
        let (p, _) = unchain(&json!({"soundtrack": "song.flac", "soundtrack_start": 9.21, "chain": {"from": "prev", "context_frames": 22}})).unwrap();
        assert_eq!(p["guide_clip"], "prev:22:0");
        assert!(p.get("first_audio").is_none() && p.get("first_audio_s").is_none());
        let (p, _) = unchain(&json!({"chain": {"from": "prev", "context_frames": 22}})).unwrap();
        assert_eq!(p["first_audio"], "prev");
        assert_eq!(ig, vec!["chain.pin_renorm".to_string()]);
        let (p, _) = unchain(&json!({"chain": {"mode": "png"}, "first_frame": "a.png"})).unwrap();
        assert_eq!(p["first_frame"], "a.png"); // the flat key wins
        assert_eq!(p["chain_mode"], "png");
        assert!(unchain(&json!({"chain": {"context_frames": 30}})).is_err());
    }

    #[test]
    fn priority_places() {
        let q = Queue { items: vec![json!({"priority": 2}), json!({"priority": 0}), json!({}), json!({"priority": -1})], ..Queue::default() };
        assert_eq!(q.slot(0), 3); // after its equals, ahead of lower ones
        assert_eq!(q.slot(1), 1);
        assert_eq!(q.slot(5), 0);
        assert_eq!(q.slot(-5), 4);
    }
}
