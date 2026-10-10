//! The API on :8000.
//!
//! - `POST /rpc/<kind>.<method>`: H3's rules - POST only, one fixed route a method, every variable in the JSON body (no
//!   path variables, no query strings); the answer `{"ok": true, "result": ...}` or `{"ok": false, "error": {"code",
//!   "message"}}` with the HTTP status. Kinds: `system` (the box: models, enabling, loading), `llm`, `image`, `audio`,
//!   `video` (the studio's methods, `video.generate` ...). Every kind but system has `describePromptGuideMD {model}`.
//! - The OpenAI-compatible routes (`/v1/models`, `/v1/chat/completions`, `/v1/images/generations`, `/v1/images/edits`,
//!   `/v1/audio/speech` and the files they link) - passed to the switcher or the kind's server, which is started for
//!   the model the request names first.
//! - Both want `Authorization: Bearer <token>`. `GET /openapi.yml` (open) describes the methods of the kinds that have
//!   an enabled model.

use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::life::s;
use super::{now, Home};
use crate::http::{self, call_for, Conn, Request};
use super::registry;

/// One method: its kind and name, what it does, its body's fields (name, JSON type, description, required) and an
/// example body
pub struct Method {
    pub kind: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub fields: &'static [(&'static str, &'static str, &'static str, bool)],
    pub example: &'static str,
}

const MODEL: (&str, &str, &str, bool) = ("model", "string", "the model's registry id (system.models)", true);
const MODEL_OPT: (&str, &str, &str, bool) = ("model", "string", "the model's id; none: the loaded one, else the first enabled", false);

pub const METHODS: &[Method] = &[
    Method { kind: "system", name: "state", summary: "The box: the GPUs (memory by process and vouched for), every model's state, the front ends", fields: &[], example: "{}" },
    Method { kind: "system", name: "models", summary: "Every registered model: kind, enabled, its GPUs, its state (disabled, unloaded, loading, loaded, busy, ready)", fields: &[], example: "{}" },
    Method { kind: "system", name: "enable", summary: "Offer a model on GPU(s): it loads on its first request and unloads when idle; chat models join Open WebUI's list",
             fields: &[MODEL, ("gpus", "integer[]", "the cards (default: as registered); several for a model that spans cards", false)],
             example: r#"{"model": "qwen-image-2.1-q8", "gpus": [0]}"# },
    Method { kind: "system", name: "gpus", summary: "Change the cards a model may use (enabled or not); more than fit is allowed (the card shows overbooked). A loaded model moved off its card unloads: its next request loads it on the new cards",
             fields: &[MODEL, ("gpus", "integer[]", "the cards: one for image and audio models, one or more for chat and video", true)],
             example: r#"{"model": "qwen3.8-27b-q4", "gpus": [0, 1]}"# },
    Method { kind: "system", name: "disable", summary: "Withdraw a model: unloaded (after the requests it is answering), off Open WebUI's list and this spec",
             fields: &[MODEL, ("force", "boolean", "a video model while a clip renders: stop the clip at its next step", false)], example: r#"{"model": "qwen-image-2.1-q8"}"# },
    Method { kind: "system", name: "load", summary: "Load an enabled model now (otherwise its first request does)",
             fields: &[MODEL, ("wait", "boolean", "answer when it has loaded (default: at once, loading in the background)", false)],
             example: r#"{"model": "minimax-music3", "wait": true}"# },
    Method { kind: "system", name: "unload", summary: "Unload a model now (it stays enabled)", fields: &[MODEL], example: r#"{"model": "minimax-music3"}"# },
    Method { kind: "system", name: "config", summary: "The settings file (nextsycl.conf): its text, every setting the program reads (value, default, what reads it), the file's others",
             fields: &[], example: "{}" },
    Method { kind: "system", name: "configSet", summary: "Change settings (the file copied first): set {NAME: value, NAME: null to remove} or text (the whole file); answers what changed and what must restart",
             fields: &[("set", "object", "{NAME: value | null}", false), ("text", "string", "the whole file instead", false)],
             example: r#"{"set": {"NS_IDLE_MINUTES": "15"}}"# },
    Method { kind: "system", name: "restart", summary: "Restart what reads the settings: serve, switch, studio, video, image, audio or llm",
             fields: &[("service", "string", "serve | switch | studio | video | image | audio | llm", true)], example: r#"{"service": "switch"}"# },

    Method { kind: "llm", name: "chat", summary: "A chat completion (OpenAI's body; stream is ignored - use /v1/chat/completions to stream)",
             fields: &[("model", "string", "a chat model's id", true), ("messages", "object[]", "[{role, content}]", true),
                       ("max_tokens", "integer", "", false), ("temperature", "number", "", false), ("tools", "object[]", "OpenAI tool definitions", false)],
             example: r#"{"model": "qwen3.8-flash-next-iq2_xs", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 200}"# },
    Method { kind: "llm", name: "models", summary: "The enabled chat models (what Open WebUI lists) and which is loaded", fields: &[], example: "{}" },
    Method { kind: "llm", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "qwen3.8-flash-next-iq2_xs"}"# },

    Method { kind: "image", name: "generate", summary: "Pictures from a prompt (the model loads first if it must)",
             fields: &[MODEL_OPT, ("prompt", "string", "", true), ("size", "string", "WxH, e.g. 1024x1024", false), ("n", "integer", "pictures", false),
                       ("steps", "integer", "", false), ("seed", "integer", "", false), ("sampler", "string", "ComfyUI's names (euler, euler_ancestral, dpmpp_2m ...)", false),
                       ("schedule", "string", "", false), ("cfg", "number", "true guidance (needs negative_prompt)", false), ("negative_prompt", "string", "", false),
                       ("loras", "string[]", "name:scale", false), ("response_format", "string", "url (default) or b64_json", false)],
             example: r#"{"model": "qwen-image-2.1-q8", "prompt": "a lighthouse at dusk, oil painting", "size": "1024x768"}"# },
    Method { kind: "image", name: "edit", summary: "Edit or compose pictures: the first is changed, the others are references the prompt names",
             fields: &[MODEL_OPT, ("prompt", "string", "the instructions", true), ("images", "string[]", "up to 8, base64 or data: URLs", true),
                       ("size", "string", "default: the last picture's aspect", false), ("steps", "integer", "", false), ("seed", "integer", "", false),
                       ("response_format", "string", "url (default) or b64_json", false)],
             example: r#"{"prompt": "make it night, keep the lighthouse", "images": ["data:image/png;base64,iVBORw0..."]}"# },
    Method { kind: "image", name: "info", summary: "The model's defaults, samplers, schedules, options and LoRAs (loaded: from the server)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "image", name: "progress", summary: "The request running: step of steps, how many wait", fields: &[], example: "{}" },
    Method { kind: "image", name: "history", summary: "The pictures made, newest first, with their settings", fields: &[], example: "{}" },
    Method { kind: "image", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "qwen-image-2.1-q8"}"# },

    Method { kind: "audio", name: "generate", summary: "A song from a description and lyrics (the model loads first if it must); the answer links the WAV",
             fields: &[MODEL_OPT, ("instructions", "string", "the music: genre, mood, tempo, instruments, voice", true),
                       ("input", "string", "the lyrics, [verse] / [chorus] tags on lines of their own (empty: instrumental)", false),
                       ("seconds", "number", "", false), ("seed", "integer", "", false), ("steps", "integer", "", false), ("cfg", "number", "", false)],
             example: r#"{"model": "minimax-music3", "instructions": "warm acoustic folk, 90 BPM, a soft male voice", "input": "[verse]\nRiver runs slow tonight\n[chorus]\nCarry me home", "seconds": 60}"# },
    Method { kind: "audio", name: "speak", summary: "Speech from text (a speech model: Qwen3-TTS, VoxCPM2); the answer links the WAV",
             fields: &[MODEL_OPT, ("input", "string", "the text to say", true),
                       ("voice", "string", "a built-in voice (audio.info lists them) or a saved one (audio.voices.list: spoken by a model that clones)", false),
                       ("language", "string", "english, chinese, japanese ... or auto (default)", false),
                       ("instructions", "string", "how to say it (CustomVoice), or the voice to make up (VoiceDesign)", false),
                       ("ref_audio", "string", "a recording of the voice to clone, WAV as base64 or a data: URL (the Base model)", false),
                       ("ref_text", "string", "what the recording says: the closer, in-context clone", false),
                       ("seconds", "number", "the most to make", false), ("seed", "integer", "", false),
                       ("options", "object", "the engine's own: greedy, temperature, top-k, streaming", false)],
             example: r#"{"model": "qwen3-tts-custom", "input": "Good morning. The bread is still warm.", "voice": "ryan", "language": "english", "instructions": "cheerful, a little hurried"}"# },
    Method { kind: "audio", name: "voices.list", summary: "The saved voices (cloned or designed): name, what the sample says, its description, length; sample: its WAV",
             fields: &[], example: "{}" },
    Method { kind: "audio", name: "voices.add", summary: "Save a voice cloned from a recording (WAV, MP3, FLAC, Ogg, M4A; 3-15 s of one speaker is best)",
             fields: &[("name", "string", "lowercase letters, digits, - and _", true), ("sample", "string", "the recording, base64 or a data: URL", true),
                       ("text", "string", "what the recording says (a closer clone)", false), ("description", "string", "", false),
                       ("language", "string", "", false), ("replace", "boolean", "over a voice of that name", false)],
             example: r#"{"name": "grandpa", "sample": "data:audio/mpeg;base64,SUQzBAAAAA...", "text": "Well, back in my day we walked to school."}"# },
    Method { kind: "audio", name: "voices.design", summary: "Make up a voice from a description (the voice-design model reads a sample) and save it",
             fields: &[("name", "string", "", true), ("instructions", "string", "the voice: age, gender, timbre, pace, emotion, accent", true),
                       ("text", "string", "what the sample says (default: a ten-second line in the language)", false), ("language", "string", "", false),
                       ("seed", "integer", "another seed, another speaker of that kind", false), ("model", "string", "a voice-design model", false),
                       ("replace", "boolean", "", false)],
             example: r#"{"name": "narrator", "instructions": "A deep, warm male narrator in his fifties, unhurried, a slight rasp.", "language": "english"}"# },
    Method { kind: "audio", name: "voices.remove", summary: "Delete a saved voice", fields: &[("name", "string", "", true)], example: r#"{"name": "narrator"}"# },
    Method { kind: "audio", name: "info", summary: "The model's defaults, limits and options (a speech model: its voices and languages)", fields: &[MODEL_OPT], example: "{}" },
    Method { kind: "audio", name: "progress", summary: "The request running: its phase and how far", fields: &[], example: "{}" },
    Method { kind: "audio", name: "history", summary: "The songs and speech made, newest first", fields: &[], example: "{}" },
    Method { kind: "audio", name: "cancel", summary: "Stop the sound being made (between steps or frames)", fields: &[], example: "{}" },
    Method { kind: "audio", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "minimax-music3"}"# },

    Method { kind: "video", name: "generate", summary: "Queue a clip (the studio's generate: every key of the studio's form)",
             fields: &[("prompt", "string", "the structured prompt (video.describePromptGuideMD)", true), ("seconds", "number", "4-15", false),
                       ("steps", "integer", "", false), ("seed", "integer", "", false), ("width", "integer", "", false), ("height", "integer", "", false),
                       ("engine", "string", "INT8, Q6_K, Q4_K_M, REF2VA", false), ("label", "string", "", false), ("project", "string", "", false),
                       ("first_frame", "string", "a picture or clip in the studio's output, or prev", false), ("last_frame", "string", "", false),
                       ("ref_images", "string[]", "Ref2VA references (files in the output)", false), ("ref_videos", "string[]", "", false), ("ref_audios", "string[]", "", false),
                       ("control_video", "string", "", false), ("control_mask", "string", "", false), ("control_source", "string", "", false),
                       ("sampler", "string", "", false), ("schedule", "string", "", false), ("queue", "boolean", "queue when busy (default: 409 busy)", false)],
             example: r#"{"prompt": "integrated_multimodal_description: [Shot 1] Live-action, a red fox trots through deep snow at dawn.\n\noverall_soundscape: soft wind, crunching snow.\n\nnon_diegetic_music: None.", "seconds": 5, "steps": 8, "queue": true}"# },
    Method { kind: "video", name: "status", summary: "The clip running or last: stage, percent, ETA, timings, the queue, version", fields: &[], example: "{}" },
    Method { kind: "video", name: "wait", summary: "Long poll: answers when the status's version changes, or at the timeout",
             fields: &[("version", "string", "the last version seen", false), ("timeout_s", "number", "", false)], example: r#"{"version": "", "timeout_s": 30}"# },
    Method { kind: "video", name: "cancel", summary: "Stop the running clip at its next step", fields: &[], example: "{}" },
    Method { kind: "video", name: "queue.list", summary: "The queue", fields: &[], example: "{}" },
    Method { kind: "video", name: "queue.pause", summary: "Hold a project or batch (none: everything)", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: r#"{"project": "Harbour"}"# },
    Method { kind: "video", name: "queue.resume", summary: "Release a project or batch (none: everything)", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "video", name: "queue.clear", summary: "Drop the queue, or a project's / batch's part of it", fields: &[("project", "string", "", false), ("batch", "string", "", false)], example: "{}" },
    Method { kind: "video", name: "jobs.list", summary: "Finished clips", fields: &[("limit", "integer", "", false)], example: r#"{"limit": 20}"# },
    Method { kind: "video", name: "jobs.get", summary: "A clip's record: settings, per-stage seconds, energy", fields: &[("name", "string", "h3_YYYYMMDD_HHMMSS", true)], example: r#"{"name": "h3_20261009_113734"}"# },
    Method { kind: "video", name: "projects.list", summary: "The projects and batches", fields: &[], example: "{}" },
    Method { kind: "video", name: "films.list", summary: "Joined films", fields: &[], example: "{}" },
    Method { kind: "video", name: "templates.list", summary: "Prompt templates", fields: &[], example: "{}" },
    Method { kind: "video", name: "engines.list", summary: "The denoisers the daemon has", fields: &[], example: "{}" },
    Method { kind: "video", name: "canvases.list", summary: "The canvases and their measured cost", fields: &[], example: "{}" },
    Method { kind: "video", name: "summary", summary: "Measured GPU time per second of video, energy, queue ETA", fields: &[], example: "{}" },
    Method { kind: "video", name: "upload", summary: "Put a reference, control video, mask or keyframe beside the clips (its name is what the other keys take)",
             fields: &[("name", "string", "the file's name (picture, clip or sound)", true), ("data", "string", "the file, base64", true)],
             example: r#"{"name": "fox.png", "data": "iVBORw0KGgo..."}"# },
    Method { kind: "video", name: "describePromptGuideMD", summary: "How to prompt the model, as Markdown", fields: &[MODEL], example: r#"{"model": "minimax-h3"}"# },
];

/// The prompt guide of a model's architecture
fn guide(kind: &str, arch: &str) -> &'static str {
    match (kind, arch) {
        ("video", _) => include_str!("guides/minimax-h3.md"),
        ("image", _) => include_str!("guides/qwen-image-2.1.md"),
        ("audio", "qwen3-tts") => include_str!("guides/qwen3-tts.md"),
        ("audio", "voxcpm2") => include_str!("guides/voxcpm2.md"),
        ("audio", _) => include_str!("guides/minimax-music3.md"),
        _ => include_str!("guides/chat.md"),
    }
}

fn code_of(status: u16) -> &'static str {
    match status {
        400 => "invalid_params",
        401 => "unauthorized",
        404 => "not_found",
        409 => "conflict",
        503 => "unavailable",
        504 => "timeout",
        _ => "internal",
    }
}

type Fail = (u16, String);

impl Home {
    fn authorized(&self, req: &Request) -> bool {
        http::header(&req.head, "authorization").is_some_and(|h| h.trim().strip_prefix("Bearer ").is_some_and(|t| t.trim() == self.token))
    }

    pub(crate) fn api(self: &Arc<Self>, mut s: Conn, req: Request) {
        let rpc = req.path.starts_with("/rpc/");
        let fail = |s: &mut Conn, st: u16, m: String| {
            if rpc {
                http::respond(s, st, &json!({"ok": false, "error": {"code": code_of(st), "message": m}}));
            } else {
                http::respond(s, st, &json!({"error": {"message": m, "type": if st == 401 { "invalid_api_key" } else { "invalid_request_error" }}}));
            }
        };
        if !self.authorized(&req) {
            return fail(&mut s, 401, "Authorization: Bearer <token> (the token is on the page's API tab)".into());
        }
        if rpc {
            if req.method != "POST" || req.target.contains('?') {
                return fail(&mut s, 400, "POST only, every variable in the JSON body (no query strings)".into());
            }
            let body: Value = if req.body.is_empty() { json!({}) } else {
                match serde_json::from_slice(&req.body) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => return fail(&mut s, 400, "the body must be a JSON object".into()),
                }
            };
            match self.rpc(&req.path["/rpc/".len()..], &body) {
                Ok(v) => {
                    // the files an answer links, as the caller reaches them (through here: /v1/.../files/...)
                    let host = req.host.clone().unwrap_or_else(|| "localhost:8000".into());
                    let v: Value = serde_json::from_str(&v.to_string().replace("http://nextsycl/", &format!("http://{host}/"))).unwrap_or(v);
                    http::respond(&mut s, 200, &json!({"ok": true, "result": v}))
                }
                Err((st, m)) => fail(&mut s, st, m),
            }
            return;
        }
        if let Err((st, m)) = self.openai(&mut s, &req) {
            fail(&mut s, st, m);
        }
    }

    /// The kinds with an enabled model (system always)
    fn kinds_on(&self, all: &[Value]) -> Vec<&'static str> {
        let mut v = vec!["system"];
        for k in ["llm", "image", "audio", "video"] {
            if all.iter().any(|m| registry::kind_of(m) == k && m["enabled"] != false) {
                v.push(k);
            }
        }
        v
    }

    /// A kind's model for a request: the one it names (enabled, of the kind), else the loaded one, else the first
    /// enabled
    fn pick(&self, kind: &str, body: &Value, all: &[Value]) -> Result<String, Fail> {
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false).collect();
        if let Some(id) = body["model"].as_str().filter(|x| !x.is_empty()) {
            return match all.iter().find(|m| m["id"] == id) {
                Some(m) if registry::kind_of(m) != kind => Err((400, format!("{id} is a {} model, not {kind}", registry::kind_of(m)))),
                Some(m) if m["enabled"] == false => Err((409, format!("{id} is disabled (system.enable)"))),
                Some(_) => Ok(id.to_string()),
                None => Err((404, format!("no model {id} (system.models)"))),
            };
        }
        let loaded = call_for(&self.target(kind), "GET", "/api/info", None, 2).ok().map(|i| s(&i, "model"));
        loaded.filter(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, format!("no {kind} model is enabled (system.enable)")))
    }

    /// The saved voices (beside the audio server's output: NS_AUDIO_OUT)
    pub(crate) fn voices(&self) -> crate::voices::Library {
        let out = self.o.cfg.get("NS_AUDIO_OUT").unwrap_or_else(|| format!("{}/.local/share/nextsycl/audio", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
        crate::voices::Library::of(std::path::Path::new(&out))
    }

    /// The speech model for a request that names none (or names one not registered): one that clones for a saved
    /// voice or a recording, the built-in voices' for a voice, the voice-design one for instructions alone; the
    /// loaded one first
    fn speech_model(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        if b["model"].as_str().is_some_and(|m| all.iter().any(|x| x["id"] == m)) {
            return self.pick("audio", b, all);
        }
        let voice = s(b, "voice");
        let need = if !s(b, "ref_audio").is_empty() || (!voice.is_empty() && self.voices().get(&voice).is_some()) {
            "clone"
        } else if !voice.is_empty() {
            "voices"
        } else if !s(b, "instructions").is_empty() {
            "design"
        } else {
            ""
        };
        let can: Vec<String> = all.iter()
            .filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
            .filter(|m| m["speech"].as_array().is_some_and(|a| need.is_empty() || a.iter().any(|x| x == need)))
            .map(|m| s(m, "id")).collect();
        let loaded = call_for(&self.target("audio"), "GET", "/api/info", None, 2).ok().map(|i| s(&i, "model"));
        loaded.filter(|l| can.contains(l)).or_else(|| can.first().cloned()).ok_or_else(|| {
            (409, match need {
                "clone" => "no enabled speech model clones a voice (system.enable qwen3-tts-base)".to_string(),
                "design" => "no enabled speech model designs a voice (system.enable qwen3-tts-design)".to_string(),
                "voices" => "no enabled speech model has built-in voices (system.enable qwen3-tts-custom)".to_string(),
                _ => "no speech model is enabled (system.enable qwen3-tts-custom ...)".to_string(),
            })
        })
    }

    /// audio.voices.design: the voice-design model reads a sample in the voice described, which is saved
    fn design_voice(self: &Arc<Self>, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        let name = s(b, "name");
        crate::voices::check_name(&name).map_err(|e| (400, e))?;
        let lib = self.voices();
        if lib.get(&name).is_some() && b["replace"] != true {
            return Err((409, format!("a voice {name:?} exists already (replace: true)")));
        }
        let lang = b["language"].as_str().filter(|l| !l.is_empty());
        let text = b["text"].as_str().filter(|t| !t.trim().is_empty()).map(str::to_string)
            .or_else(|| crate::voices::design_text(lang).map(str::to_string))
            .ok_or_else(|| (400, format!("text: what the sample says (no default line in {})", lang.unwrap_or("that language"))))?;
        let id = self.speech_model(&json!({"model": b["model"], "instructions": s(b, "instructions")}), all)?;
        let t = self.ensure(&id)?;
        let mut q = json!({"model": id, "input": text, "instructions": s(b, "instructions"), "response_format": "url"});
        for k in ["language", "seed", "options"] {
            if b.get(k).is_some_and(|v| !v.is_null()) {
                q[k] = b[k].clone();
            }
        }
        let r = call_for(&t, "POST", "/v1/audio/speech", Some(&q), 600).map_err(|e| (502, e))?;
        let url = r["data"][0]["url"].as_str().ok_or_else(|| (502, format!("the speech server answered {r}")))?;
        let file = lib.dir.parent().map(|d| d.join(url.rsplit('/').next().unwrap_or(""))).unwrap_or_default();
        let bytes = std::fs::read(&file).map_err(|e| (500, format!("{}: {e}", file.display())))?;
        let seed = r["nextsycl"]["seed"].clone();
        let v = lib.add(&name, &bytes, Some(&text), Some(&s(b, "instructions")), lang, "design", &format!("designed by {id}, seed {seed}"), b["replace"] == true)
            .map_err(|e| (400, e))?;
        Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{name}.wav"), "seed": seed}))
    }

    fn rpc(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let Some(m) = METHODS.iter().find(|m| format!("{}.{}", m.kind, m.name) == name) else {
            return Err((404, format!("no method {name} (GET /openapi.yml)")));
        };
        for (f, _, _, req) in m.fields {
            if *req && b.get(*f).is_none_or(Value::is_null) {
                return Err((400, format!("{name} needs \"{f}\"")));
            }
        }
        let all = self.models();
        if !self.kinds_on(&all).contains(&m.kind) {
            return Err((409, format!("no {} model is enabled (system.enable)", m.kind)));
        }
        let model = || s(b, "model");
        let t = self.target(m.kind);
        match (m.kind, m.name) {
            ("system", "state") => Ok(self.state()),
            ("system", "models") => {
                let cards = self.named_cards(&all);
                Ok(json!({"models": self.model_states(&all, &cards)}))
            }
            ("system", "enable") => self.enable(&model(), b.get("gpus")).map(|t| json!({"model": model(), "title": t, "enabled": true})),
            ("system", "gpus") => self.set_gpus(&model(), b.get("gpus")),
            ("system", "disable") => self.disable(&model(), b["force"] == true).map(|_| json!({"model": model(), "enabled": false})),
            ("system", "load") if b["wait"] == true => self.load(&model()).map(|_| json!({"model": model(), "loaded": true})),
            ("system", "load") => self.load_async(&model()).map(|_| json!({"model": model(), "loading": true})),
            ("system", "unload") => self.unload(&model()).map(|_| json!({"model": model(), "loaded": false})),
            ("system", "config") => Ok(self.config()),
            ("system", "configSet") => self.config_set(b),
            ("system", "restart") => self.restart(&s(b, "service")),
            (k, "describePromptGuideMD") => {
                let e = all.iter().find(|x| x["id"] == model().as_str()).ok_or_else(|| (404, format!("no model {}", model())))?;
                if registry::kind_of(e) != k {
                    return Err((400, format!("{} is a {} model", model(), registry::kind_of(e))));
                }
                Ok(json!({"model": model(), "arch": e["arch"], "kind": k, "markdown": guide(k, &s(e, "arch"))}))
            }
            ("llm", "chat") => {
                let mut body = b.clone();
                body["stream"] = json!(false);
                call_for(&t, "POST", "/v1/chat/completions", Some(&body), 1800).map_err(|e| (502, e))
            }
            ("llm", "models") => {
                let loaded = self.chat_loaded(&all);
                let list: Vec<Value> = all.iter().filter(|m| registry::kind_of(m) == "llm" && m["enabled"] != false)
                    .map(|m| json!({"id": m["id"], "title": m["title"], "gpus": m["gpus"], "loaded": loaded.as_deref() == m["id"].as_str()})).collect();
                Ok(json!({"models": list}))
            }
            ("image", "generate") | ("image", "edit") => {
                let id = self.pick("image", b, &all)?;
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                if body.get("response_format").is_none() {
                    body["response_format"] = json!("url");
                }
                let path = if m.name == "edit" { "/v1/images/edits" } else { "/v1/images/generations" };
                call_for(&t, "POST", path, Some(&body), 3600).map_err(|e| (502, e))
            }
            ("audio", "generate" | "speak") => {
                let mut b = b.clone();
                if m.name == "speak" {
                    b["model"] = json!(self.speech_model(&b, &all)?);
                }
                let b = &b;
                let id = self.pick("audio", b, &all)?;
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                call_for(&t, "POST", "/v1/audio/speech", Some(&body), 3600).map_err(|e| (502, e))
            }
            ("image" | "audio", "info") => {
                let id = self.pick(m.kind, b, &all)?;
                match call_for(&t, "GET", "/api/info", None, 5) {
                    Ok(i) if s(&i, "model") == id => Ok(json!({"loaded": true, "info": i})),
                    _ => {
                        let e = all.iter().find(|x| x["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
                        Ok(json!({"loaded": false, "model": id, "title": e["title"], "defaults": e["defaults"],
                                  "note": "not loaded: the server's own details come once it is (system.load)"}))
                    }
                }
            }
            ("image" | "audio", "progress") => Ok(call_for(&t, "GET", "/api/progress", None, 5).unwrap_or(json!({"busy": false, "loaded": false}))),
            ("image" | "audio", "history") => Ok(call_for(&t, "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}))),
            ("audio", "cancel") => call_for(&t, "POST", "/api/cancel", Some(&json!({})), 10).map_err(|e| (502, e)),
            ("audio", "voices.list") => {
                let list: Vec<Value> = self.voices().list().into_iter().map(|mut v| {
                    v["sample"] = json!(format!("http://nextsycl/v1/audio/voices/{}.wav", s(&v, "name")));
                    v
                }).collect();
                Ok(json!({"voices": list}))
            }
            ("audio", "voices.remove") => self.voices().remove(&s(b, "name")).map(|_| json!({"removed": s(b, "name")})).map_err(|e| (404, e)),
            ("audio", "voices.add") => {
                let bytes = http::unbase64(&s(b, "sample")).map_err(|e| (400, format!("sample: {e}")))?;
                let o = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty());
                let v = self.voices().add(&s(b, "name"), &bytes, o("text"), o("description"), o("language"), "clone", "a recording", b["replace"] == true)
                    .map_err(|e| (400, e))?;
                Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{}.wav", s(b, "name"))}))
            }
            ("audio", "voices.design") => self.design_voice(b, &all),
            ("video", "upload") => {
                let data = http::unbase64(&s(b, "data")).map_err(|e| (400, format!("data: {e}")))?;
                let r = self.studio_upload(&s(b, "name"), &data)?;
                Ok(r)
            }
            ("video", method) => {
                let v = call_for(&t, "POST", &format!("/rpc/{method}"), Some(b), 120).map_err(|e| (502, e))?;
                Ok(v.get("result").cloned().unwrap_or(v))
            }
            _ => Err((404, format!("no method {name}"))),
        }
    }

    /// A file into the studio's output (its upload route takes the raw bytes)
    fn studio_upload(&self, name: &str, data: &[u8]) -> Result<Value, Fail> {
        use std::io::{Read, Write};
        let mut c = Conn::connect(&self.target("video")).map_err(|e| (503, format!("the studio does not answer: {e}")))?;
        let q: String = name.chars().map(|ch| if ch.is_ascii_alphanumeric() || "._-".contains(ch) { ch.to_string() } else { format!("%{:02X}", ch as u32 & 0xff) }).collect();
        write!(c, "POST /api/upload?name={q} HTTP/1.1\r\nHost: studio\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", data.len())
            .and_then(|_| c.write_all(data)).map_err(|e| (502, e.to_string()))?;
        let mut out = Vec::new();
        c.read_to_end(&mut out).map_err(|e| (502, e.to_string()))?;
        let text = String::from_utf8_lossy(&out);
        let v: Value = text.split_once("\r\n\r\n").and_then(|(_, b)| serde_json::from_str(b).ok()).unwrap_or(Value::Null);
        if v["ok"] == true { Ok(json!({"name": v["name"]})) } else { Err((400, v["error"].as_str().unwrap_or("the upload failed").to_string())) }
    }

    /// The OpenAI-compatible routes: to the switcher, or to the kind's server for the model the request names
    fn openai(self: &Arc<Self>, s: &mut Conn, req: &Request) -> Result<(), Fail> {
        let all = self.models();
        let on = self.kinds_on(&all);
        let need = |k: &str| if on.contains(&k) { Ok(()) } else { Err((409u16, format!("no {k} model is enabled"))) };
        let pass = |s: &mut Conn, t: &crate::http::Target| http::forward(s, t, req).map_err(|e| (502, e));
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => {
                let mut data: Vec<Value> = Vec::new();
                for m in all.iter().filter(|m| m["enabled"] != false && matches!(registry::kind_of(m), "llm" | "image" | "audio")) {
                    data.push(json!({"id": m["id"], "object": "model", "owned_by": "nextsycl", "created": 0, "kind": registry::kind_of(m), "name": m["title"]}));
                }
                http::respond(s, 200, &json!({"object": "list", "data": data}));
                Ok(())
            }
            ("POST", "/v1/chat/completions") => {
                need("llm")?;
                pass(s, &self.target("llm"))
            }
            ("POST", p @ ("/v1/images/generations" | "/v1/images/edits" | "/v1/audio/speech")) => {
                let kind = if p.starts_with("/v1/images") { "image" } else { "audio" };
                need(kind)?;
                // the model the body names (a JSON field, or a multipart part)
                let ctype = http::header(&req.head, "content-type").unwrap_or_default();
                let named = if let Some(b) = ctype.split("boundary=").nth(1) {
                    http::multipart(&req.body, b.trim_matches('"')).ok()
                        .and_then(|ps| ps.into_iter().find(|p| p.name == "model").map(|p| String::from_utf8_lossy(&p.data).trim().to_string()))
                } else {
                    serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string))
                };
                // OpenAI's own names (dall-e-3, tts-1 ...) mean "the kind's model"
                let named = named.filter(|n| all.iter().any(|m| m["id"] == n.as_str()));
                // speech with a voice or a recording and no model of ours named: the speech model for it
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                let has = |k: &str| body[k].as_str().is_some_and(|v| !v.is_empty());
                let id = if kind == "audio" && named.is_none() && (has("voice") || has("ref_audio")) {
                    self.speech_model(&body, &all)?
                } else {
                    self.pick(kind, &json!({"model": named}), &all)?
                };
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("GET", p) if p.starts_with("/v1/images/files/") => pass(s, &self.target("image")),
            ("GET", p) if p.starts_with("/v1/audio/files/") => pass(s, &self.target("audio")),
            ("GET", p) if p.starts_with("/v1/audio/voices/") => {
                let name = p["/v1/audio/voices/".len()..].trim_end_matches(".wav");
                match self.voices().sample_path(name).and_then(|f| std::fs::read(f).ok()) {
                    Some(b) => {
                        http::respond_bytes(s, 200, "audio/wav", &b, "Cache-Control: no-cache\r\n");
                        Ok(())
                    }
                    None => Err((404, format!("no saved voice {name}"))),
                }
            }
            (m, p) => Err((404, format!("no route {m} {p} (GET /openapi.yml)"))),
        }
    }

    /// The OpenAPI document of what the enabled kinds offer, with the token for the examples
    pub(crate) fn openapi(&self, req: &Request) -> Value {
        let all = self.models();
        let on = self.kinds_on(&all);
        let host = req.host.clone().unwrap_or_else(|| "localhost:8000".into());
        let ids = |k: &str| -> Vec<Value> { all.iter().filter(|m| registry::kind_of(m) == k && m["enabled"] != false).map(|m| m["id"].clone()).collect() };
        let envelope = json!({"type": "object", "properties": {
            "ok": {"type": "boolean"}, "result": {"description": "the method's answer"},
            "error": {"type": "object", "properties": {"code": {"type": "string"}, "message": {"type": "string"}}}}});
        let mut paths = Map::new();
        for m in METHODS.iter().filter(|m| on.contains(&m.kind)) {
            let mut props = Map::new();
            let mut required = Vec::new();
            for (f, ty, d, req) in m.fields {
                let mut p = match *ty {
                    t if t.ends_with("[]") => json!({"type": "array", "items": {"type": &t[..t.len() - 2]}}),
                    t => json!({"type": t}),
                };
                if !d.is_empty() {
                    p["description"] = json!(d);
                }
                if *f == "model" && m.kind != "system" {
                    p["enum"] = Value::Array(ids(m.kind));
                }
                props.insert(f.to_string(), p);
                if *req {
                    required.push(json!(f));
                }
            }
            let mut schema = json!({"type": "object", "properties": props});
            if !required.is_empty() {
                schema["required"] = Value::Array(required);
            }
            let example: Value = serde_json::from_str(m.example).unwrap_or(json!({}));
            let route = format!("/rpc/{}.{}", m.kind, m.name);
            let curl = format!("curl -s http://{host}{route} -H 'Authorization: Bearer {}' -H 'Content-Type: application/json' -d '{}'",
                               self.token, example.to_string().replace('\'', "'\\''"));
            paths.insert(route, json!({"post": {
                "tags": [m.kind], "operationId": format!("{}.{}", m.kind, m.name), "summary": m.summary,
                "requestBody": {"required": true, "content": {"application/json": {"schema": schema, "example": example}}},
                "responses": {"200": {"description": "{\"ok\": true, \"result\": ...}", "content": {"application/json": {"schema": envelope.clone()}}},
                              "4XX": {"description": "{\"ok\": false, \"error\": {\"code\", \"message\"}}"}},
                "x-curl": curl,
            }}));
        }
        let oa = |summary: &str, body: Value| json!({"tags": ["openai"], "summary": summary,
            "requestBody": {"required": true, "content": {"application/json": {"example": body}}}, "responses": {"200": {"description": "as OpenAI's"}}});
        paths.insert("/v1/models".into(), json!({"get": {"tags": ["openai"], "summary": "The enabled chat, image and audio models", "responses": {"200": {"description": "OpenAI's list"}}}}));
        if on.contains(&"llm") {
            paths.insert("/v1/chat/completions".into(), json!({"post": oa("Chat (streams with stream: true; tools)",
                json!({"model": ids("llm").first(), "messages": [{"role": "user", "content": "Hello"}], "stream": false}))}));
        }
        if on.contains(&"image") {
            paths.insert("/v1/images/generations".into(), json!({"post": oa("Pictures (and ours: steps, seed, sampler, schedule, cfg, negative_prompt, loras)",
                json!({"model": ids("image").first(), "prompt": "a lighthouse at dusk", "size": "1024x1024"}))}));
            paths.insert("/v1/images/edits".into(), json!({"post": oa("Edits: multipart (image, image[] ...) or JSON images",
                json!({"prompt": "make it night", "images": ["data:image/png;base64,..."]}))}));
            paths.insert("/v1/images/files/{name}".into(), json!({"get": {"tags": ["openai"], "summary": "A picture an answer links (OpenAI's url form)",
                "parameters": [{"name": "name", "in": "path", "required": true, "schema": {"type": "string"}}], "responses": {"200": {"description": "PNG"}}}}));
        }
        if on.contains(&"audio") {
            paths.insert("/v1/audio/speech".into(), json!({"post": oa("A song (input: the lyrics, instructions: the music) or speech (input: the text; voice, \
                language, instructions, ref_audio, ref_text); response_format wav or url",
                json!({"model": ids("audio").first(), "instructions": "warm folk", "input": "[verse]\nRiver runs slow", "response_format": "url"}))}));
            paths.insert("/v1/audio/files/{name}".into(), json!({"get": {"tags": ["openai"], "summary": "A song an answer links",
                "parameters": [{"name": "name", "in": "path", "required": true, "schema": {"type": "string"}}], "responses": {"200": {"description": "WAV"}}}}));
        }
        json!({
            "openapi": "3.0.3",
            "info": {"title": "nextsycl", "version": crate::VERSION, "description": format!(
                "The box's API (nextsycl serve). Rules: POST /rpc/<kind>.<method>, every variable in the JSON body - no path variables, no query \
                 strings; answers {{\"ok\": true, \"result\": ...}} or {{\"ok\": false, \"error\": {{\"code\", \"message\"}}}}. The OpenAI-compatible \
                 /v1 routes are the exception. Send Authorization: Bearer {} with every call. Kinds here: {}. Generated {:.0}.",
                self.token, on.join(", "), now())},
            "servers": [{"url": format!("http://{host}")}],
            "security": [{"bearer": []}],
            "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": format!("the token: {}", self.token)}}},
            "x-token": self.token,
            "paths": paths,
        })
    }
}
