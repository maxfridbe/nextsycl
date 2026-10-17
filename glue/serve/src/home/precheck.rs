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

/// A job's queue keys: priority -9..9, held and queue true or false, project and batch text
fn queue_fields(b: &Value) -> Result<(), Fail> {
    if !b["priority"].is_null() && b["priority"].as_i64().is_none_or(|p| !(-9..=9).contains(&p)) {
        return Err(bad(format!("priority {}: a whole number from -9 to 9", b["priority"])));
    }
    for k in ["held", "queue"] {
        if !b[k].is_null() && !b[k].is_boolean() {
            return Err(bad(format!("{k} {}: true or false", b[k])));
        }
    }
    for k in ["project", "batch"] {
        if !b[k].is_null() && !b[k].is_string() {
            return Err(bad(format!("{k} {}: a name (text)", b[k])));
        }
    }
    Ok(())
}

impl Home {
    /// The checks for `name` (a job method) before it is queued
    pub(super) fn precheck(&self, name: &str, b: &Value, all: &[Value]) -> Result<(), Fail> {
        queue_fields(b)?;
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
                return Err(bad(format!("steps {st}: {id} runs its own {n} sigmas (a few-step distill) - send {n} or leave steps out")));
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
        number(b, "cfg", 0.0, 30.0)?;
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
        // the model's own voices and languages, from its config (Qwen3-TTS: talker_config); a model without lists
        // (VoxCPM2) takes any language
        let (own_voices, own_langs) = speech_lists(&m);
        let lang = s(b, "language").to_lowercase();
        if !lang.is_empty() && lang != "auto" && !own_langs.is_empty() && !own_langs.contains(&lang) {
            let shown: Vec<&str> = own_langs.iter().map(String::as_str).filter(|l| !l.ends_with("_dialect")).collect();
            return Err(bad(format!("language {lang}: not one {id} speaks ({}, or auto)", shown.join(", "))));
        }
        if !voice.is_empty() && carried.is_empty() && can("voices") && !saved(&voice) && !own_voices.is_empty() && !own_voices.contains(&voice.to_lowercase()) {
            return Err(bad(format!("voice {voice}: not a built-in voice of {id} ({}) nor a saved one (audio.voices.list)", own_voices.join(", "))));
        }
        if !voice.is_empty() && carried.is_empty() && !can("voices") && !saved(&voice) {
            return Err(bad(format!("voice {voice}: not a saved voice (audio.voices.list), and {id} has no built-in voices")));
        }
        // a saved voice: a model that clones it, or one it is enrolled for (no recording or carry-on beside it)
        let enrolled = carried.is_empty() && s(b, "ref_audio").is_empty() && super::api::enrolled_for(&self.voices(), &voice, &m);
        if !voice.is_empty() && saved(&voice) && !can("clone") && !enrolled {
            return Err(bad(match can("enrolled") {
                true => format!("voice {voice} is a saved voice not enrolled for {id}: enroll it (nextsycl audio voice enroll {voice}, experimental) or \
                                 use a model that clones (qwen3-tts-base, voxcpm2)"),
                false => format!("voice {voice} is a saved voice: speak it with a model that clones (qwen3-tts-base, voxcpm2)"),
            }));
        }
        // a model that clones only with the transcript: a saved voice must have one, a recording its ref_text
        if can("transcript") {
            if !voice.is_empty() && carried.is_empty() && self.voices().get(&voice).is_some_and(|v| v["text"].as_str().is_none_or(|t| t.trim().is_empty())) {
                return Err(bad(format!("voice {voice}: {id} clones only with what the sample says, and this voice has no transcript - add it again with text, or leave model out")));
            }
            if b["ref_audio"].as_str().is_some_and(|a| !a.is_empty()) && s(b, "ref_text").trim().is_empty() {
                return Err(bad(format!("ref_text: {id} clones from a recording and its exact transcript")));
            }
        }
        // a multi-speaker script's voices: saved ones (voices) or recordings (ref_audios), a speaker each
        let names: Vec<&Value> = b["voices"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        let recs: Vec<&Value> = b["ref_audios"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        if !names.is_empty() || !recs.is_empty() {
            if !can("speakers") {
                return Err(bad(format!("voices / ref_audios: {id} speaks one voice - use a model of several speakers (vibevoice-1.5b) or leave model out")));
            }
            if !names.is_empty() && !recs.is_empty() {
                return Err(bad("voices or ref_audios: one list of the speakers' voices, not both".into()));
            }
            if !voice.is_empty() || b["ref_audio"].as_str().is_some_and(|a| !a.is_empty()) || !carried.is_empty() {
                return Err(bad("voices / ref_audios name every speaker's voice: not with voice, ref_audio or continue_from".into()));
            }
            if names.len().max(recs.len()) > 4 {
                return Err(bad(format!("voices: {} speakers, at most 4", names.len().max(recs.len()))));
            }
            for v in names {
                let n = v.as_str().unwrap_or_default();
                if !saved(n) {
                    return Err(bad(format!("voices: {n:?} is not a saved voice (audio.voices.list)")));
                }
            }
            for (i, a) in recs.iter().enumerate() {
                let wav = crate::http::unbase64(a.as_str().unwrap_or_default()).map_err(|e| bad(format!("ref_audios[{i}]: {e}")))?;
                nextsycl_audio::decode_wav(&wav).map_err(|e| bad(format!("ref_audios[{i}]: {e} (a WAV)")))?;
            }
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

/// A speech model's built-in voices and languages (lower case), read from its config file, else from its entry's own
/// `voices` / `languages` (a card's, for a model whose file holds them: Voxtral TTS's GGUF); empty: no list (any)
pub fn speech_lists(m: &Value) -> (Vec<String>, Vec<String>) {
    let Some(c) = m["files"]["config"].as_str().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else {
        let list = |k: &str| -> Vec<String> { m[k].as_array().into_iter().flatten().filter_map(|x| x.as_str()).map(str::to_lowercase).collect() };
        return (list("voices"), list("languages"));
    };
    let keys = |k: &str| -> Vec<String> { c["talker_config"][k].as_object().into_iter().flatten().map(|(n, _)| n.to_lowercase()).collect() };
    (keys("spk_id"), keys("codec_language_id"))
}


#[cfg(test)]
mod tests {
    use super::{number, speech_lists, steps_of, whole};
    use serde_json::json;

    #[test]
    fn a_few_step_model_locks_its_steps() {
        let turbo = json!({"defaults": {"steps": 6}, "env": {"NS_QI_SIGMAS": "1,0.9375,0.875,0.75,0.5,0.25"}});
        let s = steps_of(&turbo);
        assert_eq!((s["locked"].as_bool(), s["allowed"].clone(), s["default"].as_u64()), (Some(true), json!([6]), Some(6)));
        let full = steps_of(&json!({"defaults": {"steps": 40}}));
        assert_eq!((full["locked"].as_bool(), full["max"].as_u64()), (Some(false), Some(100)));
        assert!(full["allowed"].is_null());
    }

    #[test]
    fn queue_keys_are_checked() {
        assert!(super::queue_fields(&json!({"priority": 3, "held": true, "queue": false, "project": "film"})).is_ok());
        for b in [json!({"priority": 10}), json!({"priority": 1.5}), json!({"held": "yes"}), json!({"project": 4})] {
            assert_eq!(super::queue_fields(&b).unwrap_err().0, 400, "{b}");
        }
    }

    #[test]
    fn numbers_are_checked_against_their_range() {
        let b = json!({"steps": 8, "cfg": "high", "strength": 0.01, "n": 2.5});
        assert_eq!(whole(&b, "steps", 1, 100).unwrap(), Some(8));
        assert!(whole(&b, "steps", 1, 6).unwrap_err().1.starts_with("steps 8:"));
        assert!(number(&b, "cfg", 0.0, 30.0).is_err());
        assert!(number(&b, "strength", 0.05, 1.0).unwrap_err().1.starts_with("strength"));
        assert!(whole(&b, "n", 1, 8).is_err(), "not a whole number");
        assert_eq!(whole(&b, "seed", 0, 9).unwrap(), None);
    }

    #[test]
    fn a_speech_model_lists_its_voices_from_its_config() {
        let dir = std::env::temp_dir().join(format!("ns-precheck-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.json");
        std::fs::write(&p, r#"{"talker_config": {"spk_id": {"Ryan": 1, "serena": 2}, "codec_language_id": {"english": 1, "beijing_dialect": 2}}}"#).unwrap();
        let (v, l) = speech_lists(&json!({"files": {"config": p.to_str().unwrap()}}));
        assert!(v.contains(&"ryan".to_string()) && v.contains(&"serena".to_string()));
        assert!(l.contains(&"english".to_string()));
        // no config (VoxCPM2's has no lists): any
        assert_eq!(speech_lists(&json!({})), (Vec::new(), Vec::new()));
        // no config but the entry's own lists (Voxtral TTS's card)
        let (v, l) = speech_lists(&json!({"voices": ["Neutral_Female", "fr_male"], "languages": ["french"]}));
        assert_eq!((v, l), (vec!["neutral_female".to_string(), "fr_male".to_string()], vec!["french".to_string()]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
