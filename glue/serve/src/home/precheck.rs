//! A job's request checked when it is submitted (wait: false answers at once): the values its server would refuse -
//! steps a few-step model does not run, a size, a sampler, a picture that does not decode, a file to carry on from -
//! are a 400 now, not a failed job later. The servers still check everything; this is what can be known without the
//! model loaded (the registry: its defaults and its own sigmas).

use serde_json::{json, Value};

use super::life::s;
use super::Home;

type Fail = (u16, String);

fn bad(m: String) -> Fail {
    (400, m)
}

/// A number field: none, or a number within [lo, hi]
fn number(b: &Value, k: &str, lo: f64, hi: f64) -> Result<Option<f64>, Fail> {
    match &b[k] {
        Value::Null => Ok(None),
        v => {
            let x = v.as_f64().ok_or_else(|| bad(format!("{k}: a number")))?;
            if !(lo..=hi).contains(&x) {
                return Err(bad(format!("{k} {x}: from {lo} to {hi}")));
            }
            Ok(Some(x))
        }
    }
}

/// A whole-number field within [lo, hi]
fn whole(b: &Value, k: &str, lo: u64, hi: u64) -> Result<Option<u64>, Fail> {
    match &b[k] {
        Value::Null => Ok(None),
        v => {
            let x = v.as_u64().ok_or_else(|| bad(format!("{k}: a whole number")))?;
            if !(lo..=hi).contains(&x) {
                return Err(bad(format!("{k} {x}: from {lo} to {hi}")));
            }
            Ok(Some(x))
        }
    }
}

/// The steps an image model takes (image.info's steps): its default, and the only count a few-step model runs
pub fn steps_of(m: &Value) -> Value {
    let own = m["env"]["NS_QI_SIGMAS"].as_str().map(|v| v.split(',').filter(|x| !x.trim().is_empty()).count());
    let default = m["defaults"]["steps"].as_u64().or(own.map(|n| n as u64));
    match own {
        Some(n) => json!({"default": default, "locked": true, "allowed": [n], "min": n, "max": n,
                          "note": format!("a few-step model: its own {n} levels; another count is refused (other schedules than shift take any)")}),
        None => json!({"default": default, "locked": false, "allowed": Value::Null, "min": 1, "max": 100}),
    }
}

impl Home {
    /// The checks for `name` (a job method) before it is queued
    pub(super) fn precheck(&self, name: &str, b: &Value, all: &[Value]) -> Result<(), Fail> {
        match name {
            "image.generate" | "image.edit" => self.precheck_image(name == "image.edit", b, all),
            "audio.speak" => self.precheck_speak(b, all),
            "audio.generate" => {
                number(b, "seconds", 1.0, 600.0)?;
                whole(b, "steps", 1, 200)?;
                number(b, "cfg", 0.0, 30.0)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn precheck_image(&self, edit: bool, b: &Value, all: &[Value]) -> Result<(), Fail> {
        let id = self.pick("image", b, all)?;
        let m = all.iter().find(|x| x["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
        whole(b, "n", 1, 8)?;
        if let Some(f) = b["response_format"].as_str() {
            if !["url", "b64_json"].contains(&f) {
                return Err(bad(format!("response_format {f}: url or b64_json")));
            }
        }
        match b["size"].as_str() {
            None | Some("auto") => {}
            Some(sz) => {
                let wh = sz.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
                match wh {
                    Some((w, h)) if w >= 64 && h >= 64 && w % 16 == 0 && h % 16 == 0 => {}
                    Some(_) => return Err(bad(format!("size {sz}: both sides multiples of 16, at least 64"))),
                    None => return Err(bad(format!("size {sz}: WIDTHxHEIGHT"))),
                }
            }
        }
        if !b["size"].is_null() && !b["size"].is_string() {
            return Err(bad("size: \"WIDTHxHEIGHT\" or \"auto\"".into()));
        }
        let name = |k: &str| b[k].as_str().filter(|v| !v.is_empty() && *v != "default").map(str::to_string);
        if let Some(v) = name("sampler") {
            nextsycl_image::Sampler::parse(&v).ok_or_else(|| bad(format!("sampler {v}?")))?;
        }
        let schedule = name("schedule");
        if let Some(v) = &schedule {
            nextsycl_image::Schedule::parse(v).ok_or_else(|| bad(format!("schedule {v}?")))?;
        }
        let steps = whole(b, "steps", 1, 100)?;
        // a few-step model runs its own levels (unless another schedule, or the request's own sigmas)
        let own = m["env"]["NS_QI_SIGMAS"].as_str().map(|v| v.split(',').filter(|x| !x.trim().is_empty()).count());
        let shift = schedule.as_deref().is_none_or(|v| v.eq_ignore_ascii_case("shift"));
        if let (Some(n), Some(st), true, true) = (own, steps, shift, b["options"]["sigmas"].is_null()) {
            if st as usize != n {
                return Err(bad(format!("{id} runs its own {n} sigmas (a few-step distill): steps {st} is not one of its schedules - send {n} or leave steps out")));
            }
        }
        let cfg = number(b, "cfg", 0.0, 30.0)?.unwrap_or(1.0);
        if cfg > 1.0 && s(b, "negative_prompt").trim().is_empty() {
            return Err(bad(format!("cfg {cfg}: guidance above 1 needs a negative_prompt (the model's true_cfg_scale)")));
        }
        let strength = number(b, "strength", 0.05, 1.0)?.unwrap_or(1.0);
        let pics: Vec<(String, &str)> = if edit {
            let list: Vec<&Value> = match (&b["images"], &b["image"]) {
                (Value::Array(a), _) => a.iter().collect(),
                (_, v @ Value::String(_)) => vec![v],
                _ => Vec::new(),
            };
            if list.is_empty() {
                return Err(bad("images: one to 8 pictures (base64 or data: URLs)".into()));
            }
            if list.len() > 8 {
                return Err(bad(format!("images: {} pictures, at most 8", list.len())));
            }
            list.iter().enumerate().map(|(i, v)| v.as_str().map(|x| (format!("images[{i}]"), x)).ok_or_else(|| bad(format!("images[{i}]: a base64 string or data: URL"))))
                .collect::<Result<_, _>>()?
        } else {
            match ["image", "init_image"].iter().find_map(|k| b[*k].as_str().filter(|v| !v.is_empty())) {
                Some(x) => {
                    if strength >= 1.0 {
                        return Err(bad("image (an init picture) needs strength below 1: at 1 the picture would be ignored".into()));
                    }
                    vec![("image".to_string(), x)]
                }
                None if strength < 1.0 => return Err(bad("strength: needs a picture to start from (image, an init picture; or image.edit)".into())),
                None => Vec::new(),
            }
        };
        if edit && b["reference"] == false && strength >= 1.0 {
            return Err(bad("reference false (a plain img2img) needs strength below 1".into()));
        }
        for (what, data) in pics {
            let bytes = crate::http::unbase64(data).map_err(|e| bad(format!("{what}: {e}")))?;
            crate::image::picture(&bytes, &what).map_err(bad)?;
        }
        Ok(())
    }

    fn precheck_speak(&self, b: &Value, all: &[Value]) -> Result<(), Fail> {
        if s(b, "input").trim().is_empty() && s(b, "text").trim().is_empty() {
            return Err(bad("input (the text to say): required".into()));
        }
        let id = self.speech_model(b, all)?;
        let m = all.iter().find(|x| x["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
        let can = |c: &str| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == c));
        number(b, "seconds", 0.5, 600.0)?;
        let saved = |v: &str| self.voices().get(v).is_some();
        let carried = s(b, "continue_from");
        if !carried.is_empty() {
            if !can("clone") {
                return Err(bad(format!("continue_from: {id} does not clone a voice - use a model that does (qwen3-tts-base, voxcpm2) or leave model out")));
            }
            let name = carried.rsplit('/').next().unwrap_or(&carried).to_string();
            if !name.ends_with(".wav") || !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
                return Err(bad(format!("continue_from {name}: a speech file's name (<name>.wav, as audio.speak answered)")));
            }
            let out = self.o.cfg.get("NS_AUDIO_OUT").unwrap_or_else(|| format!("{}/.local/share/nextsycl/audio", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
            let path = std::path::Path::new(&out).join(&name);
            let said = nextsycl_audio::wav_info(&path).map_err(|_| (404, format!("continue_from {name}: no speech file of that name here")))?.0;
            if !said.iter().any(|(k, v)| k == "INAM" && !v.trim().is_empty()) {
                return Err(bad(format!("continue_from {name}: the file does not say what it says (not speech made here)")));
            }
        }
        let voice = s(b, "voice");
        if !voice.is_empty() && carried.is_empty() && !can("voices") && !saved(&voice) {
            return Err(bad(format!("voice {voice}: not a saved voice (audio.voices.list), and {id} has no built-in voices")));
        }
        if !voice.is_empty() && saved(&voice) && !can("clone") {
            return Err(bad(format!("voice {voice} is a saved voice: speak it with a model that clones (qwen3-tts-base, voxcpm2)")));
        }
        if let Some(a) = b["ref_audio"].as_str().filter(|a| !a.is_empty()) {
            if !can("clone") {
                return Err(bad(format!("ref_audio: {id} does not clone a voice")));
            }
            let wav = crate::http::unbase64(a).map_err(|e| bad(format!("ref_audio: {e}")))?;
            nextsycl_audio::decode_wav(&wav).map_err(|e| bad(format!("ref_audio: {e} (a WAV)")))?;
        }
        Ok(())
    }
}
