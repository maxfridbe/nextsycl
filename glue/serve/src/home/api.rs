//! The API on :8000.
//!
//! - `POST /rpc/<kind>.<method>`: H3's rules - POST only, one fixed route a method, every variable in the JSON body (no
//!   path variables, no query strings); the answer `{"ok": true, "result": ...}` or `{"ok": false, "error": {"code",
//!   "message", "retryable", "status", "method", "field"}}` - the body says it all: every /rpc answer is HTTP 200,
//!   errors too (error.status carries the number by convention), so another transport can carry the same envelope;
//!   only the file routes (/v1/.../files/...) use HTTP statuses. Kinds: `system` (the box: models, enabling, loading), `llm`, `image`, `audio`,
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
use super::spec;

// The methods are declared in methods.rs: each its #[rpc] handler with its typed request (rpcgen); the table,
// the document's operations and the wfe client are made from them.
use super::methods::{self as table, Method};

/// system.version's api: raised when an answer changes in a way an old client misreads
const API_VERSION: u32 = 3;

/// What this build does, by names a client can test for (system.version); names are added, never reused
const FEATURES: &[&str] = &[
    "system.busy", "system.busy.cards", "system.busy.slots", "system.ping", "system.version", "system.files.stat", "system.files.delete",
    "jobs.image", "jobs.audio", "jobs.cancel.label", "reentrant_id",
    "video.frames", "video.chain", "video.priority", "video.cancel.label", "video.queue.move", "video.queue.remove", "video.engines.inputs",
    "video.negative_prompt.ignored", "image.canvases", "image.cancel", "spec.open",
    "image.edit.strength", "audio.speak.continue_from", "video.preview", "system.busy.parallel",
    "errors.retryable", "errors.no_room", "jobs.precheck", "image.info.steps", "image.generate.init_image", "image.edit.reference",
    "video.failure", "video.gpu", "video.engines.ignored", "audio.speak.precheck.voice", "jobs.cancel.reentrant_id", "rpc.errors.body", "audio.transcribe", "image.upscale", "video.upscale", "image.vectorize",
    "rpc.http200", "result.run", "models.capabilities", "system.cards", "jobs.list", "jobs.status.one", "jobs.queue", "jobs.priority", "jobs.held",
    "jobs.stage", "uploads", "voices.owner", "voices.ready", "room.free_idle", "room.wait", "llm.no_room", "llm.keep_up_s",
    "system.maintenance", "errors.maintenance", "jobs.model3d", "model3d.generate", "model3d.rig", "model3d.upload.mesh",
    "system.drives", "system.move", "errors.cards_taken", "audio.voices.enroll", "drives.sidecars", "image.loras.request", "video.soundtrack", "video.exposure_fix",
    "backplane", "system.pin", "system.eta", "rsvp_s", "max_willing_to_wait_s", "errors.wait_too_long", "jobs.eta", "video.on_demand",
];

/// Qwen-Image's sizes: its training buckets (multiples of 16) and the guide's defaults
const IMAGE_CANVASES: &[(u32, u32, &str, &str)] = &[
    (1024, 1024, "1:1", "the default"), (1328, 1328, "1:1", "the model's own square, the largest trained"),
    (1344, 768, "16:9", ""), (768, 1344, "9:16", ""), (1664, 928, "16:9", "the model's own wide"), (928, 1664, "9:16", "the model's own tall"),
    (1152, 864, "4:3", ""), (864, 1152, "3:4", ""), (1472, 1104, "4:3", "the model's own 4:3 (1472 x 1140 there, 1104 here: multiples of 16)"),
    (1104, 1472, "3:4", ""), (1248, 832, "3:2", ""), (832, 1248, "2:3", ""), (1584, 1056, "3:2", "the model's own 3:2"), (1056, 1584, "2:3", ""),
    (512, 512, "1:1", "a quick draft"),
];

pub(crate) fn code_of(status: u16) -> &'static str {
    match status {
        400 => "invalid_params",
        401 => "unauthorized",
        404 => "not_found",
        499 => "cancelled",
        409 => "conflict",
        502 => "unavailable",
        503 => "no_room",
        504 => "timeout",
        _ => "internal",
    }
}

/// Whether the same call may succeed later unchanged: the box was busy or a server did not answer (not a bad value,
/// a missing or disabled model, a cancel)
pub(crate) fn retryable(status: u16, code: &str) -> bool {
    matches!(status, 502..=504) || code == "busy" || code == "maintenance" || code == "cards_taken"
}

/// An OpenAI route's error: {"error": {message, type}}; cards taken by a GPU run (crate::gpulock) as maintenance's,
/// with its code and Retry-After; a model not downloaded says so (code model_not_downloaded, its size, the call)
fn openai_error(s: &mut Conn, st: u16, m: &str) {
    if let Some((code, fields, words)) = super::backplane::untag(m) {
        let mut e = json!({"message": words, "type": code, "code": code});
        if let (Some(o), Some(f)) = (e.as_object_mut(), fields.as_object()) {
            o.extend(f.clone());
        }
        return http::respond(s, st, &json!({"error": e}));
    }
    if let Some((f, words)) = super::pulls::untag(m) {
        let mut e = json!({"message": words, "type": "model_not_downloaded", "code": "model_not_downloaded"});
        if let (Some(o), Some(f)) = (e.as_object_mut(), f.as_object()) {
            o.extend(f.clone());
        }
        return http::respond(s, st, &json!({"error": e}));
    }
    if m.starts_with("cards taken:") {
        let body = json!({"error": {"message": m, "type": "cards_taken", "code": "cards_taken"}});
        let extra = format!("Retry-After: {}\r\n", crate::gpulock::RETRY_AFTER_S);
        return http::respond_bytes(s, st, "application/json", body.to_string().as_bytes(), &extra);
    }
    http::respond(s, st, &json!({"error": {"message": m, "type": if st == 401 { "invalid_api_key" } else { "invalid_request_error" }}}));
}

/// An error's body: code, message, whether to retry, and the status as a number (the same as the HTTP one, so a
/// client - or another transport - needs only the body)
pub(crate) fn error_of(status: u16, m: &str) -> Value {
    let code = code_of(status);
    json!({"code": code, "message": m, "retryable": retryable(status, code), "status": status})
}

/// An /rpc error in full: error_of with the method, the request key at fault when the message starts with one
/// ("steps 8: ...", "images[0]: ...", "voice bogus: ..."), and a code of its own where the status's is too broad
pub(crate) fn rpc_error(method: &str, status: u16, code: Option<&str>, m: &str) -> Value {
    // an error of the backplane's own (wait_too_long): its code and fields
    if let Some((c, fields, words)) = super::backplane::untag(m) {
        let mut e = rpc_error(method, status, Some(&c), &words);
        if let (Some(o), Some(f)) = (e.as_object_mut(), fields.as_object()) {
            o.extend(f.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())));
        }
        return e;
    }
    // a model a card offers that is not downloaded (pulls.rs): its own code and what fetches it
    if let Some((fields, words)) = super::pulls::untag(m) {
        let mut e = rpc_error(method, status, Some("model_not_downloaded"), &words);
        if let (Some(o), Some(f)) = (e.as_object_mut(), fields.as_object()) {
            o.extend(f.clone());
        }
        return e;
    }
    let mut e = error_of(status, m);
    // a job refused with queue: false says "busy: ..." (as video's busy)
    let code = code.or_else(|| (status == 409 && m.starts_with("busy:")).then_some("busy"));
    // a load refused while a GPU run holds the cards (crate::gpulock) says "cards taken: ..."
    let code = code.or_else(|| m.starts_with("cards taken:").then_some("cards_taken"));
    if let Some(c) = code {
        e["code"] = json!(c);
        e["retryable"] = json!(retryable(status, c));
    }
    if e["code"] == "cards_taken" {
        e["retry_after_s"] = json!(crate::gpulock::RETRY_AFTER_S);
    }
    e["method"] = if method.is_empty() { Value::Null } else { json!(method) };
    e["field"] = field_of(method, m).map_or(Value::Null, |f| json!(f));
    e
}

/// The request key a message names first, when it is one of the method's fields
fn field_of(method: &str, m: &str) -> Option<String> {
    let (kind, name) = method.split_once('.')?;
    let me = table::methods().iter().find(|x| x.kind == kind && x.name == name)?;
    let first: String = m.trim_start_matches('"').chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    let names = || me.fields.iter().map(|f| f.name.as_str()).chain(["strength", "image", "reference", "continue_from"]);
    // a field the message says is needed wins over its first word ("image.generate needs \"prompt\"" starts with
    // image, a field too): the fields' order is no longer what decided it
    names().find(|f| m.contains(&format!("needs \"{f}\""))).or_else(|| names().find(|f| first == *f)).map(str::to_string)
}

type Fail = (u16, String);

/// A call to a kind's server whose failure keeps that server's status (a 400 stays a 400); 502 when it does not answer
fn relay(t: &crate::http::Target, method: &str, path: &str, body: Option<&Value>, secs: u64) -> Result<Value, Fail> {
    match http::call_status(t, method, path, body, secs) {
        Ok((200, v)) => Ok(v),
        Ok((st, v)) => Err((st, v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("request failed").to_string())),
        Err(e) => Err((502, e)),
    }
}

impl Home {
    fn authorized(&self, req: &Request) -> bool {
        http::header(&req.head, "authorization").is_some_and(|h| h.trim().strip_prefix("Bearer ").is_some_and(|t| t.trim() == self.token))
    }

    pub(crate) fn api(self: &Arc<Self>, mut s: Conn, req: Request) {
        let rpc = req.path.starts_with("/rpc/");
        let method = req.path.strip_prefix("/rpc/").unwrap_or("").to_string();
        let fail_as = |s: &mut Conn, st: u16, code: Option<&str>, m: String| {
            if rpc {
                // HTTP 200: the envelope is the answer, error.status keeps the number
                http::respond(s, 200, &json!({"ok": false, "error": rpc_error(&method, st, code, &m)}));
            } else {
                openai_error(s, st, &m);
            }
        };
        let fail = |s: &mut Conn, st: u16, m: String| {
            if rpc {
                fail_as(s, st, None, m);
            } else {
                openai_error(s, st, &m);
            }
        };
        if !self.authorized(&req) {
            return fail(&mut s, 401, "Authorization: Bearer <token> (the token is on the page's API tab)".into());
        }
        // maintenance: what would start work refused with the notice (maint.rs)
        if self.maintenance_refused(&mut s, &req) {
            return;
        }
        if rpc {
            if req.method != "POST" || req.target.contains('?') {
                return fail_as(&mut s, 400, Some("invalid_request"), "POST only, every variable in the JSON body (no query strings)".into());
            }
            let body: Value = if req.body.is_empty() { json!({}) } else {
                match serde_json::from_slice(&req.body) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => return fail_as(&mut s, 400, Some("invalid_request"), "the body must be a JSON object".into()),
                }
            };
            // a method's name in any case: the one the table declares (image.generate for IMAGE.Generate)
            let asked = &req.path["/rpc/".len()..];
            let name = table::find(asked).map(|m| m.full()).unwrap_or(asked);
            if let Some((_, v)) = self.studio_rpc(name, &body, req.host.as_deref().unwrap_or("localhost:8000")) {
                return http::respond(&mut s, 200, &v);
            }
            match self.rpc(name, &body) {
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
    pub(super) fn kinds_on(&self, all: &[Value]) -> Vec<&'static str> {
        let mut v = vec!["system"];
        for k in ["llm", "image", "audio", "video", "model3d"] {
            if all.iter().any(|m| registry::kind_of(m) == k && m["enabled"] != false) {
                v.push(k);
            }
        }
        v
    }

    /// A kind's model for a request: the one it names (enabled, of the kind), else the loaded one, else the first
    /// enabled
    pub(super) fn pick(&self, kind: &str, body: &Value, all: &[Value]) -> Result<String, Fail> {
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false).collect();
        if let Some(id) = body["model"].as_str().filter(|x| !x.is_empty()) {
            return match all.iter().find(|m| m["id"] == id) {
                Some(m) if registry::kind_of(m) != kind => Err((400, format!("{id} is a {} model, not {kind}", registry::kind_of(m)))),
                Some(m) if m["enabled"] == false => Err((409, format!("{id} is disabled (system.enable)"))),
                Some(_) => Ok(id.to_string()),
                // a card's model not here yet: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first -
                // a job's request id lets it say so while it waits
                None if self.not_downloaded(id).is_some() => self.need_downloaded(id, body["request_id"].as_str(), true).map(|_| id.to_string()),
                None => Err((404, format!("no model {id} (system.models)"))),
            };
        }
        let loaded = self.loaded_of(kind);
        loaded.into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, format!("no {kind} model is enabled (system.enable)")))
    }

    /// The saved voices (beside the audio server's output: NS_AUDIO_OUT)
    pub(crate) fn voices(&self) -> crate::voices::Library {
        let out = self.o.cfg.get("NS_AUDIO_OUT").unwrap_or_else(|| format!("{}/.local/share/nextsycl/audio", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
        crate::voices::Library::of(std::path::Path::new(&out))
    }

    /// The upscaler for image.upscale / video.upscale: the one named (it must upscale), else - by the variant asked
    /// (waifu2x's are models-*) - the loaded or first enabled one of that family, else any
    pub(super) fn upscaler_for(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
            let id = self.pick("image", b, all)?;
            if all.iter().any(|m| m["id"] == id.as_str() && !upscales(m)) {
                return Err((400, format!("{id} draws pictures; upscaling needs an upscaler (real-esrgan, waifu2x)")));
            }
            return Ok(id);
        }
        let want = match b["variant"].as_str() {
            Some(v) if v.starts_with("models-") => Some("waifu2x"),
            Some(v) if !v.is_empty() => Some("realesrgan"),
            _ => None,
        };
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && upscales(m)
            && want.is_none_or(|a| s(m, "arch") == a)).collect();
        self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, format!("no upscaler{} is enabled (system.enable real-esrgan / waifu2x)", want.map(|a| format!(" ({a})")).unwrap_or_default())))
    }

    /// The recogniser for audio.transcribe: the one named (it must listen), else the loaded or first enabled one
    pub(super) fn recogniser(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        let listens = |m: &Value| s(m, "arch") == "whisper";
        if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
            let id = self.pick("audio", b, all)?;
            if all.iter().any(|m| m["id"] == id.as_str() && !listens(m)) {
                return Err((400, format!("{id} makes sound; audio.transcribe needs a recogniser (whisper-large-v3-turbo)")));
            }
            return Ok(id);
        }
        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && listens(m)).collect();
        self.loaded_of("audio").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
            .ok_or_else(|| (409, "no recogniser is enabled (system.enable whisper-large-v3-turbo)".to_string()))
    }

    /// The speech model for a request that names none (or names one not registered): one that clones for a saved
    /// voice or a recording, the built-in voices' for a voice, the voice-design one for instructions alone; the
    /// loaded one first
    pub(super) fn speech_model(&self, b: &Value, all: &[Value]) -> Result<String, Fail> {
        // one named that is here, or a card's not downloaded yet (pick refuses or fetches it)
        if b["model"].as_str().is_some_and(|m| all.iter().any(|x| x["id"] == m) || self.not_downloaded(m).is_some()) {
            return self.pick("audio", b, all);
        }
        let voice = s(b, "voice");
        let cloned = !s(b, "ref_audio").is_empty() || !s(b, "continue_from").is_empty() || (!voice.is_empty() && self.voices().get(&voice).is_some());
        // a saved voice alone (no recording, no carry-on): a model it is enrolled for speaks it too
        let saved_only = !voice.is_empty() && s(b, "ref_audio").is_empty() && s(b, "continue_from").is_empty() && self.voices().get(&voice).is_some();
        let lib = self.voices();
        let speakers = ["voices", "ref_audios"].iter().any(|k| b[*k].as_array().is_some_and(|a| !a.is_empty()));
        // a clone with an instruction: a model that directs a cloned voice (Breeze TTS 2)
        let need = if speakers {
            "speakers"
        } else if cloned && !s(b, "instructions").is_empty() {
            "direction"
        } else if cloned {
            "clone"
        } else if !voice.is_empty() {
            "voices"
        } else if !s(b, "instructions").is_empty() {
            "design"
        } else {
            ""
        };
        // a saved voice without its transcript: not a model that clones only with one
        let untold = !voice.is_empty() && self.voices().get(&voice).is_some_and(|v| v["text"].as_str().is_none_or(|t| t.trim().is_empty()));
        let can: Vec<String> = all.iter()
            .filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
            .filter(|m| m["speech"].as_array().is_some_and(|a| need.is_empty() || a.iter().any(|x| x == need)))
            .filter(|m| !(untold && needs_text(all, &m["id"])))
            // a built-in voice: a model that has it (or lists none)
            .filter(|m| need != "voices" || speaks(m, &voice))
            .map(|m| s(m, "id")).collect();
        // a saved voice no enabled model clones: a model it is enrolled for (experimental; models that clone come first)
        let can = if can.is_empty() && need == "clone" && saved_only {
            all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && enrolled_for(&lib, &voice, m)).map(|m| s(m, "id")).collect()
        } else {
            can
        };
        let loaded = self.loaded_of("audio");
        loaded.into_iter().find(|l| can.contains(l)).or_else(|| can.first().cloned()).ok_or_else(|| {
            (409, match need {
                "clone" => "no enabled speech model clones a voice (system.enable qwen3-tts-base)".to_string(),
                "speakers" => "no enabled speech model speaks a script of several voices (system.enable vibevoice-1.5b)".to_string(),
                "direction" => "no enabled speech model directs a cloned voice by instructions (system.enable breeze-tts-2)".to_string(),
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
        self.owned(&name, b)?;
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
        let r = relay(&t, "POST", "/v1/audio/speech", Some(&q), 600)?;
        let url = r["data"][0]["url"].as_str().ok_or_else(|| (502, format!("the speech server answered {r}")))?;
        let file = lib.dir.parent().map(|d| d.join(url.rsplit('/').next().unwrap_or(""))).unwrap_or_default();
        let bytes = std::fs::read(&file).map_err(|e| (500, format!("{}: {e}", file.display())))?;
        let seed = r["nextsycl"]["seed"].clone();
        let mut v = lib.add(&name, &bytes, Some(&text), Some(&s(b, "instructions")), lang, "design", &format!("designed by {id}, seed {seed}"), b["replace"] == true)
            .map_err(|e| (400, e))?;
        if let Some(owner) = b["owner"].as_str().filter(|o| !o.trim().is_empty()) {
            v = lib.tag(&name, "owner", json!(owner.trim())).map_err(|e| (500, e))?;
        }
        Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{name}.wav"), "seed": seed}))
    }

    /// The method a name calls, its body's required fields there, its kind enabled
    fn checked(&self, name: &str, b: &Value, all: &[Value]) -> Result<&'static Method, Fail> {
        let Some(m) = table::find(name) else {
            return Err((404, format!("no method {name} (GET /openapi.yml)")));
        };
        for f in m.fields.iter().filter(|f| !f.optional) {
            if b.get(&f.name).is_none_or(Value::is_null) {
                return Err((400, format!("{name} needs \"{}\"", f.name)));
            }
        }
        // the known fields' types (a string where a number goes ...): the request struct's own check
        (m.check)(b).map_err(|e| (400, format!("{name}: {}", e.0)))?;
        // a model named that a card offers and is not downloaded: refused now (409 model_not_downloaded), or with
        // NS_AUTO_PULL its download started and the request let through (its kind is on once it lands; pick waits)
        if m.kind != "system" {
            if let Some(id) = b["model"].as_str().filter(|id| !all.iter().any(|x| x["id"] == *id) && self.not_downloaded(id).is_some()) {
                self.need_downloaded(id, None, false)?;
                return Ok(m);
            }
        }
        if !self.kinds_on(all).contains(&m.kind) {
            return Err((409, format!("no {} model is enabled (system.enable)", m.kind)));
        }
        Ok(m)
    }

    /// A studio method (video.*, but upload and the guide): the studio's answer as it gave it - its status and error
    /// code (busy, not_running ...) - with the files it names as URLs through here
    fn studio_rpc(&self, name: &str, b: &Value, host: &str) -> Option<(u16, Value)> {
        // video.upscale is the image servers' (an upscaler's job), not the studio's
        let method = name.strip_prefix("video.").filter(|m| !matches!(*m, "upload" | "describePromptGuideMD" | "upscale"))?;
        let all = self.models();
        let envelope = |st: u16, m: String| (st, json!({"ok": false, "error": rpc_error(name, st, None, &m)}));
        if let Err((st, m)) = self.checked(name, b, &all) {
            return Some(envelope(st, m));
        }
        // any video call brings a stopped daemon up
        if let Err((st, m)) = self.video_up() {
            return Some(envelope(st, m));
        }
        // a clip withdrawn at its limit says so when asked for
        if method == "status" {
            if let Some(e) = b["label"].as_str().and_then(|l| self.video_withdrawn(l)) {
                return Some((200, json!({"ok": false, "error": e})));
            }
        }
        // a clip's limit: declined here when its wait is known and too long; what the door keeps is not the studio's
        let admitted = if method == "generate" {
            match self.video_admit(b) {
                Ok(x) => Some(x),
                Err((st, m)) => return Some(envelope(st, m)),
            }
        } else {
            None
        };
        let body = admitted.as_ref().unwrap_or(b);
        let (st, mut v) = match http::call_status(&self.target("video"), "POST", &format!("/rpc/{method}"), Some(body), 120) {
            Ok(r) => r,
            Err(e) => return Some(envelope(502, format!("the video studio does not answer: {e}"))),
        };
        if v["ok"] == false {
            // one spelling for both servers
            if v["error"]["code"] == "invalid_param" {
                v["error"]["code"] = json!("invalid_params");
            }
            // the studio's own code kept; the rest as every /rpc error has it
            let code = v["error"]["code"].as_str().unwrap_or("").to_string();
            let m = v["error"]["message"].as_str().unwrap_or("").to_string();
            let mut e = rpc_error(name, st, Some(&code).filter(|c| !c.is_empty()).map(String::as_str), &m);
            if let Some(o) = v["error"].as_object() {
                for (k, x) in o.iter().filter(|(k, _)| !["code", "message"].contains(&k.as_str())) {
                    e[k.as_str()] = x.clone();
                }
            }
            v["error"] = e;
            return Some((st, v));
        }
        if let Some(sent) = &admitted {
            self.video_taken(b, sent, &mut v["result"]);
        }
        self.video_annotate(method, &mut v);
        let url = |kind: &str, n: &str| format!("http://{host}/v1/video/{kind}/{n}");
        let r = &mut v["result"];
        for k in ["file", "preview"] {
            if let Some(f) = r[k].as_str().and_then(|f| f.strip_prefix("/out/")).map(str::to_string) {
                r[k] = json!(url("files", &f));
            }
        }
        for c in r.get_mut("clips").and_then(Value::as_array_mut).into_iter().flatten() {
            if c["state"] == "done" {
                if let Some(n) = c["name"].as_str().map(str::to_string) {
                    c["file"] = json!(url("files", &format!("{n}.mp4")));
                    c["thumb"] = json!(url("thumbs", &format!("{n}.webp")));
                }
            }
        }
        Some((st, v))
    }

    /// Whether anything is working: models loading or answering, chat requests, image and audio jobs, the clip being
    /// made and the queue; where work runs (slots) and each card's share of it
    pub(super) fn busy(&self, all: &[Value]) -> Value {
        let cards = self.named_cards(all);
        let states = self.model_states(all, &cards);
        let with = |st: &str| states.iter().filter(|m| m["state"] == st).collect::<Vec<_>>();
        let loading: Vec<Value> = with("loading").iter().map(|m| m["id"].clone()).collect();
        let working: Vec<Value> = with("busy").iter().map(|m| json!({"id": m["id"], "kind": m["kind"]})).collect();
        let mut reasons: Vec<String> = loading.iter().map(|m| format!("{} loading", m.as_str().unwrap_or(""))).collect();
        reasons.extend(working.iter().map(|m| format!("{} answering", m["id"].as_str().unwrap_or(""))));
        let active = if self.kinds_on(all).contains(&"llm") {
            let dir = self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextsycl")
            });
            call_for(&crate::http::Target::Unix(dir.join("nextsycl.sock")), "GET", "/server/requests", None, 3).ok()
                .and_then(|v| v["active"].as_array().map(|a| a.len())).unwrap_or(0)
        } else {
            0
        };
        if active > 0 {
            reasons.push(format!("{active} chat request(s)"));
        }
        let jobs = self.jobs_open();
        // held jobs wait for a release, not for the box: they are listed, not reasons
        let live = jobs.iter().filter(|j| j["held"] != true).count();
        if live > 0 {
            reasons.push(format!("{live} image/audio/3D job(s)"));
        }
        let mut slots = self.work_slots(all, &cards);
        let video = if self.kinds_on(all).contains(&"video") {
            let st = call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 10).ok().map(|v| v["result"].clone()).unwrap_or(Value::Null);
            let items = st["queue_items"].as_array().cloned().unwrap_or_default();
            let held = items.iter().filter(|i| i["held"] == true).count();
            let running = !st.is_null() && st["idle"] != true;
            let runnable = items.len() - held;
            if running {
                reasons.push(format!("clip {} {}", st["job"]["label"].as_str().unwrap_or(""), st["stage"].as_str().unwrap_or("")));
            }
            if runnable > 0 {
                reasons.push(format!("{runnable} clip(s) queued"));
            }
            // the studio's job as the engine reports it, else the card the daemon shows busy
            let gpu = Some(st["gpu"].clone()).filter(|g| running && !g.is_null())
                .or_else(|| slots.iter().find(|x| x["kind"] == "video" && x["busy"] == true).map(|x| x["gpu"].clone())).unwrap_or(Value::Null);
            let on = |k: &str| if running { st[k].clone() } else { Value::Null };
            json!({"running": running, "label": if running { st["job"]["label"].clone() } else { Value::Null }, "name": if running { st["job"]["name"].clone() } else { Value::Null },
                   "stage": on("stage"), "pct": on("pct"), "eta_s": on("eta"), "elapsed_s": on("elapsed"), "gpu": gpu,
                   "energy_wh": if running { st["job"]["energy_wh"].clone() } else { Value::Null },
                   "queued": items.len(), "runnable": runnable, "held": held,
                   "accepts": !st.is_null() && !running && runnable == 0, "reachable": !st.is_null(),
                   "slots": 1, "cards": slots.iter().filter(|x| x["kind"] == "video").count()})
        } else {
            Value::Null
        };
        // a chat server answering marks its slot busy
        for x in slots.iter_mut().filter(|x| x["kind"] == "llm") {
            x["busy"] = json!(active > 0);
        }
        let per_card: Vec<Value> = cards.as_array().into_iter().flatten().map(|c| {
            let g = c["index"].clone();
            let here: Vec<&Value> = slots.iter().filter(|x| x["gpu"] == g || x["gpus"].as_array().is_some_and(|a| a.contains(&g))).collect();
            let models: Vec<Value> = c["procs"].as_array().into_iter().flatten().filter_map(|p| p["model_id"].as_str().map(|m| json!(m))).collect();
            json!({"gpu": g, "name": c["name"], "vram_total_mb": c["vram_total_mb"], "vram_used_mb": c["vram_used_mb"], "busy_pct": c["busy_pct"],
                   "power_w": c["power_w"], "models": models, "busy": here.iter().any(|x| x["busy"] == true),
                   "work": here.iter().filter(|x| x["busy"] == true).map(|x| json!({"kind": x["kind"], "model": x["model"]})).collect::<Vec<_>>()})
        }).collect();
        let maintenance = self.maintenance().map(|n| super::maint::public(&n));
        json!({"busy": !reasons.is_empty(), "reasons": reasons, "loading": loading, "working": working, "llm": {"active": active},
               "maintenance": maintenance, "cards_taken": self.cards_taken(),
               "jobs": jobs, "video": video, "cards": per_card, "slots": slots, "parallel": self.parallel(all)})
    }

    /// How many requests of each image and audio model run at once: a copy a card it is enabled on, as many as the
    /// kind has servers (a second copy starts when the first has work and its card has the room then)
    fn parallel(&self, all: &[Value]) -> Value {
        let mut out = serde_json::Map::new();
        for kind in super::jobs::KINDS {
            let servers = self.slots(kind).len();
            let models: serde_json::Map<String, Value> = all.iter().filter(|m| registry::kind_of(m) == kind && m["enabled"] != false)
                .map(|m| (s(m, "id"), json!(self.gpus_of(m).len().min(servers)))).collect();
            out.insert(kind.into(), json!({"servers": servers, "models": models}));
        }
        Value::Object(out)
    }

    /// A method by its name (any case): its #[rpc] handler (methods.rs), which decodes the request into its type
    fn rpc(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let Some(m) = table::find(name) else {
            return Err((404, format!("no method {name} (GET /openapi.yml)")));
        };
        // the required fields first, in the words callers know ("image.generate needs \"prompt\"")
        self.checked(m.full(), b, &self.models())?;
        (m.handler)(self, b.clone()).map_err(|e| e.0)
    }

    /// A method's work, its request checked (what the handlers in methods.rs call)
    pub(super) fn rpc_inner(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        // pictures and recordings sent before (upload:<name>) read in
        let resolved = self.uploads_in(name, b)?;
        let b = resolved.as_ref().unwrap_or(b);
        // pictures, songs and speech run as jobs (jobs.rs), which call work
        if super::backplane::runs_model(name) {
            super::backplane::check_fields(b)?;
        }
        if super::jobs::METHODS.contains(&name) {
            let all = self.models();
            self.checked(name, b, &all)?;
            // what the server would refuse, refused now (wait: false would answer OK, then fail)
            self.precheck(name, b, &all)?;
            return self.submit(name, b);
        }
        // a method that runs a model and is not a job (chat, OCR): the same promises, on this thread
        if super::backplane::runs_model(name) {
            self.admit(name, b)?;
            super::backplane::begin(name, b, now());
            let body = super::backplane::stripped(b);
            let r = self.work(name, &body);
            let outcome = match &r {
                Ok(_) => "ok",
                Err((499, _)) => "cancelled",
                Err(_) => "failed",
            };
            let granted = self.finish_request(name, &body, outcome);
            return r.map(|mut v| {
                if let (Some(g), true) = (granted, v.is_object()) {
                    v["rsvp_granted_s"] = json!(g);
                }
                v
            });
        }
        self.work(name, b)
    }

    /// A method's work (a job's too)
    pub(super) fn work(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let all = self.models();
        let m = self.checked(name, b, &all)?;
        let model = || s(b, "model");
        let t = self.target(m.kind);
        match (m.kind, m.name) {
            ("system", "state") => Ok(self.state()),
            ("system", "busy") => Ok(self.busy(&all)),
            ("system", "ping") => {
                let n = self.maintenance().map(|n| super::maint::public(&n));
                Ok(json!({"available": true, "maintenance": n.is_some(),
                          "reason": n.as_ref().and_then(|n| n["message"].as_str()).filter(|m| !m.trim().is_empty()).map(str::to_string)
                                      .or_else(|| n.as_ref().map(|_| "maintenance".to_string())),
                          "retry_after_s": n.as_ref().and_then(|n| n["retry_after_s"].as_u64()),
                          "version": crate::VERSION, "api": API_VERSION, "at": now()}))
            }
            ("system", "version") => {
                let on = self.kinds_on(&all);
                let methods: Vec<String> = table::methods().iter().filter(|m| on.contains(&m.kind)).map(|m| m.full().to_string()).collect();
                Ok(json!({"version": crate::VERSION, "api": API_VERSION, "kinds": on, "features": FEATURES, "methods": methods,
                          "maintenance": self.maintenance().map(|n| super::maint::public(&n))}))
            }
            ("system", "files.stat" | "files.delete") => self.file_op(m.name, b),
            ("system", "models") => {
                let cards = self.named_cards(&all);
                // the registered ones (downloaded), then the cards' that are not downloaded (added: no field renamed)
                let mut models = self.model_states(&all, &cards);
                for m in models.iter_mut() {
                    m["downloaded"] = json!(true);
                }
                for mut nd in self.not_downloaded_all() {
                    nd["state"] = json!("not_downloaded");
                    nd["enabled"] = json!(false);
                    models.push(nd);
                }
                Ok(json!({"models": models}))
            }
            ("system", "models.pull") => self.pull_start(&model()).map(|p| json!({"pull": p.view()})),
            ("system", "models.pulls") => Ok(json!({"pulls": self.pulls_view(), "auto_pull": self.auto_pull()})),
            ("system", "models.pull.cancel") => self.pull_cancel(&model()),
            ("system", "cards") => Ok(self.cards_view(&all)),
            ("system", "enable") => self.enable(&model(), b.get("gpus")).map(|t| json!({"model": model(), "title": t, "enabled": true})),
            ("system", "gpus") => self.set_gpus(&model(), b.get("gpus")),
            ("system", "disable") => self.disable(&model(), b["force"] == true).map(|_| json!({"model": model(), "enabled": false})),
            ("system", "load") if b["wait"] == true => self.load(&model()).map(|_| json!({"model": model(), "loaded": true})),
            ("system", "load") => self.load_async(&model()).map(|_| json!({"model": model(), "loading": true})),
            ("system", "unload") => self.unload(&model()).map(|_| json!({"model": model(), "loaded": false})),
            ("system", "pin") => self.pin(&model()).map(|_| json!({"model": model(), "pinned": true, "loading": true})),
            ("system", "unpin") => self.unpin(&model()).map(|_| json!({"model": model(), "pinned": false})),
            ("system", "eta") => self.eta_answer(b),
            ("system", "config") => Ok(self.config()),
            ("system", "configSet") => self.config_set(b),
            ("system", "restart") => self.restart(&s(b, "service")),
            ("system", "maintenance") => self.maintenance_set(b),
            ("system", "drives") => {
                let cards = self.named_cards(&all);
                Ok(self.drives_view(&all, &self.model_states(&all, &cards)))
            }
            ("system", "drive.measure") => self.drive_measure(&s(b, "drive")),
            ("system", "move") => self.model_move(&model(), &s(b, "drive")),
            ("system", "move.cancel") => self.move_cancel(),
            (k, "describePromptGuideMD") => {
                let e = all.iter().find(|x| x["id"] == model().as_str()).ok_or_else(|| (404, format!("no model {}", model())))?;
                if registry::kind_of(e) != k {
                    return Err((400, format!("{} is a {} model", model(), registry::kind_of(e))));
                }
                // the model card's guide (model-cards/), else its architecture's or its kind's
                let md = self.o.cfg.guide(e).ok_or_else(|| (404, format!("no prompt guide for {} (its model card's `guide`)", model())))?;
                Ok(json!({"model": model(), "arch": e["arch"], "kind": k, "markdown": md}))
            }
            ("llm", "chat") => {
                let t = self.chat_target(b["model"].as_str())?;
                if let Some(id) = b["model"].as_str().map(str::to_string).or_else(|| self.chat_loaded(&all)) {
                    self.request_on(&id);
                }
                let mut body = b.clone();
                body["stream"] = json!(false);
                let keep = body.as_object_mut().and_then(|o| o.remove("keep_up_s"));
                let r = relay(&t, "POST", "/v1/chat/completions", Some(&body), 1800);
                // kept from the end of the answer (the model that answered)
                if let Some(k) = keep.filter(|k| !k.is_null()) {
                    let secs = k.as_f64().filter(|x| *x >= 0.0).map_or(crate::switch::KEEP_UP_S, |x| x as u64);
                    if let Some(id) = self.chat_loaded(&self.models()) {
                        self.keep_chat(&id, secs);
                    }
                }
                r
            }
            ("llm", "requests") => {
                // the chat server's control socket (its /server routes are not on its port)
                let dir = self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
                    std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextsycl")
                });
                let v = call_for(&crate::http::Target::Unix(dir.join("nextsycl.sock")), "GET", "/server/requests", None, 5)
                    .map_err(|e| (502, format!("the chat server does not answer: {e}")))?;
                let keep = ["id", "model", "state", "via", "started", "ended", "finish", "prompt_tokens", "reused", "source", "read", "read_tok_s", "read_seconds",
                            "generated", "max_tokens", "tok_s", "generate_seconds", "drafts", "energy_j", "avg_watts"];
                let slim = |r: &Value| Value::from(keep.iter().filter_map(|k| r.get(*k).map(|x| (k.to_string(), x.clone()))).collect::<serde_json::Map<_, _>>());
                let list = |k: &str| v[k].as_array().into_iter().flatten().map(slim).collect::<Vec<_>>();
                Ok(json!({"active": list("active"), "done": list("done").into_iter().take(10).collect::<Vec<_>>()}))
            }
            ("llm", "models") => {
                let loaded = self.chat_loaded(&all);
                let list: Vec<Value> = all.iter().filter(|m| registry::kind_of(m) == "llm" && m["enabled"] != false)
                    .map(|m| json!({"id": m["id"], "title": m["title"], "gpus": m["gpus"], "loaded": loaded.as_deref() == m["id"].as_str()})).collect();
                Ok(json!({"models": list}))
            }
            ("image", "ocr") => {
                // a reading model: the one named, else the loaded or first enabled one whose architecture reads
                let reads = |m: &Value| s(m, "arch").contains("ocr");
                let id = match b["model"].as_str().filter(|x| !x.is_empty()) {
                    Some(_) => self.pick("image", b, &all)?,
                    None => {
                        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && reads(m)).collect();
                        self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                            .ok_or_else(|| (409, "no OCR model is enabled (system.enable unlimited-ocr-q8)".to_string()))?
                    }
                };
                if !all.iter().any(|m| m["id"] == id.as_str() && reads(m)) {
                    return Err((400, format!("{id} makes pictures; image.ocr needs a model that reads them")));
                }
                let t = self.ensure(&id)?;
                let mut body = b.clone();
                body["model"] = json!(id);
                relay(&t, "POST", "/v1/ocr", Some(&body), 3600)
            }
            ("image", "generate") | ("image", "edit") => {
                let id = if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
                    self.pick("image", b, &all)?
                } else {
                    // no model named: the loaded or first enabled one that draws (not a reader, not an upscaler)
                    let draws = |m: &Value| !s(m, "arch").contains("ocr") && !upscales(m) && !vectorizes(m);
                    let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && draws(m)).collect();
                    self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                        .ok_or_else(|| (409, "no image model that draws is enabled (system.enable qwen-image-2.1-q8)".to_string()))?
                };
                if all.iter().any(|m| m["id"] == id.as_str() && upscales(m)) {
                    return Err((400, format!("{id} upscales pictures; image.{} needs a model that draws (image.upscale for {id})", m.name)));
                }
                if all.iter().any(|m| m["id"] == id.as_str() && vectorizes(m)) {
                    return Err((400, format!("{id} writes pictures as SVG; image.{} needs a model that draws (image.vectorize for {id})", m.name)));
                }
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                if body.get("response_format").is_none() {
                    body["response_format"] = json!("url");
                }
                let path = if m.name == "edit" { "/v1/images/edits" } else { "/v1/images/generations" };
                relay(&t, "POST", path, Some(&body), 3600)
            }
            ("image", "vectorize") => {
                let id = if b["model"].as_str().is_some_and(|x| !x.is_empty()) {
                    let id = self.pick("image", b, &all)?;
                    if all.iter().any(|m| m["id"] == id.as_str() && !vectorizes(m)) {
                        return Err((400, format!("{id} does not write SVG; image.vectorize needs a vectorizer (starvector-1b-im2svg)")));
                    }
                    id
                } else {
                    let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "image" && m["enabled"] != false && vectorizes(m)).collect();
                    self.loaded_of("image").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                        .ok_or_else(|| (409, "no vectorizer is enabled (system.enable starvector-1b-im2svg)".to_string()))?
                };
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                relay(&t, "POST", "/v1/images/vectorize", Some(&body), 3600)
            }
            ("image", "upscale") | ("video", "upscale") => {
                let id = self.upscaler_for(b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                if m.kind == "image" {
                    if body.get("response_format").is_none() {
                        body["response_format"] = json!("url");
                    }
                    return relay(&t, "POST", "/v1/images/upscale", Some(&body), 3600);
                }
                // a clip made here: its path (the same machine), not its bytes
                if let Some(f) = b["file"].as_str().filter(|f| !f.is_empty()) {
                    let plain = crate::files::plain(f.rsplit('/').next().unwrap_or(f)).ok_or_else(|| (400, format!("file {f}: a clip's name")))?;
                    let name = if plain.ends_with(".mp4") { plain.to_string() } else { format!("{plain}.mp4") };
                    let dir = self.o.cfg.get("NS_VIDEO_OUT").filter(|d| !d.is_empty())
                        .unwrap_or_else(|| format!("{}/.local/share/nextsycl/video", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
                    let path = std::path::Path::new(&dir).join(&name);
                    if !path.is_file() {
                        return Err((404, format!("file {f}: no clip {name} here (video.list)")));
                    }
                    body["path"] = json!(path.to_string_lossy());
                }
                relay(&t, "POST", "/v1/videos/upscale", Some(&body), 86400)
            }
            ("audio", "transcribe") => {
                let id = self.recogniser(b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                relay(&t, "POST", "/v1/audio/transcriptions", Some(&body), 7200)
            }
            ("audio", "generate" | "speak") => {
                let mut b = b.clone();
                if m.name == "speak" {
                    b["model"] = json!(self.speech_model(&b, &all)?);
                } else if b["model"].as_str().is_none_or(|x| x.is_empty()) {
                    // a song: the loaded or first enabled audio model that makes songs (not a speech model, not a recogniser)
                    let songs: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && !m["speech"].is_array()
                        && s(m, "arch") != "whisper").collect();
                    if let Some(id) = self.loaded_of("audio").into_iter().find(|l| songs.iter().any(|m| m["id"] == l.as_str())).or_else(|| songs.first().map(|m| s(m, "id"))) {
                        b["model"] = json!(id);
                    }
                }
                let b = &b;
                let id = self.pick("audio", b, &all)?;
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                relay(&t, "POST", "/v1/audio/speech", Some(&body), 3600)
            }
            ("model3d", "tool") => self.mesh_tool(b),
            ("model3d", "generate" | "rig") => {
                // a model that does it: the one named, else the loaded or first enabled one whose card says so (rig: a
                // rigger or an animator)
                // (a mesh given to model3d.generate is painted: a model whose card says paint - one that textures what it
                // makes need not paint a mesh it is given: TRELLIS.2 would make a new one)
                let painting = m.name == "generate" && ["mesh", "file"].iter().any(|k| b[*k].as_str().is_some_and(|v| !v.is_empty()));
                let want = if m.name == "rig" { "rig" } else if painting { "paint" } else { "generate" };
                let able = |m: &Value| m["does"].as_array().is_none_or(|a| a.iter().any(|x| x == want || (want == "rig" && x == "animate")));
                let id = match b["model"].as_str().filter(|x| !x.is_empty()) {
                    Some(_) => self.pick("model3d", b, &all)?,
                    None => {
                        let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "model3d" && m["enabled"] != false && able(m)).collect();
                        self.loaded_of("model3d").into_iter().find(|l| on.iter().any(|m| m["id"] == l.as_str())).or_else(|| on.first().map(|m| s(m, "id")))
                            .ok_or_else(|| (409, format!("no enabled 3D model does {want} (system.models: capabilities)")))?
                    }
                };
                if !all.iter().any(|m| m["id"] == id.as_str() && able(m)) {
                    return Err((400, format!("{id} does not {want} (model3d.{})", if want == "rig" { "generate" } else { "rig" })));
                }
                let job = b["request_id"].as_str();
                let t = self.ensure_for(&id, job)?;
                self.sent(job);
                let mut body = b.clone();
                body["model"] = json!(id);
                body["response_format"] = json!("url");
                relay(&t, "POST", &format!("/v1/model3d/{}", m.name), Some(&body), 7200)
            }
            ("image" | "audio" | "model3d", "info") => {
                let id = self.pick(m.kind, b, &all)?;
                let e = all.iter().find(|x| x["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
                // what a request may send, known without the model loaded
                let limits = if m.kind == "image" {
                    json!({"steps": super::precheck::steps_of(&e), "max_pictures": 8, "max_picture_pixels": crate::image::MAX_PICTURE_PIXELS,
                           "max_request_bytes": 256u64 << 20, "size_multiple": super::precheck::size_multiple(&e)})
                } else {
                    json!({"max_request_bytes": 256u64 << 20})
                };
                let mut v = match call_for(&t, "GET", "/api/info", None, 5) {
                    Ok(i) if s(&i, "model") == id => json!({"loaded": true, "model": id, "info": i}),
                    _ => json!({"loaded": false, "model": id, "title": e["title"], "defaults": e["defaults"], "voices": e["voices"], "languages": e["languages"],
                                "note": "not loaded: the server's own details (info) come once it is (system.load); steps and the limits are known now"}),
                };
                if let (Some(o), Some(l)) = (v.as_object_mut(), limits.as_object()) {
                    o.extend(l.clone());
                }
                Ok(v)
            }
            ("image" | "audio" | "model3d", "progress") => Ok(call_for(&t, "GET", "/api/progress", None, 5).unwrap_or(json!({"busy": false, "loaded": false}))),
            ("image" | "audio" | "model3d", "history") => Ok(call_for(&t, "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}))),
            ("image" | "audio" | "model3d", "status") => self.job_status(m.kind, b),
            ("image" | "audio" | "model3d", "jobs") => self.job_list(m.kind, b),
            ("image" | "audio" | "model3d", "queue.list") => Ok(self.queue_list(m.kind)),
            ("image" | "audio" | "model3d", "queue.move") => self.queue_move(m.kind, b),
            ("image" | "audio" | "model3d", "queue.pause") => self.queue_hold(m.kind, b, true),
            ("image" | "audio" | "model3d", "queue.resume") => self.queue_hold(m.kind, b, false),
            ("image" | "audio" | "model3d", "upload") => self.upload(m.kind, b),
            ("image", "canvases.list") => self.image_canvases(b, &all),
            ("image" | "audio" | "model3d", "cancel") => self.job_cancel(m.kind, b),
            ("audio", "voices.list") => {
                // the enabled models that speak a saved voice: the ones that clone
                let speakers: Vec<Value> = all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false)
                    .filter(|m| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "clone"))).map(|m| m["id"].clone()).collect();
                let lib = self.voices();
                let owner = s(b, "owner");
                let list: Vec<Value> = lib.list().into_iter().filter(|v| owner.is_empty() || v["owner"] == owner.as_str()).map(|mut v| {
                    let name = s(&v, "name");
                    v["sample"] = json!(format!("http://nextsycl/v1/audio/voices/{name}.wav"));
                    // ready: its sample is there and reads, and an enabled model can speak it now
                    let reads = lib.sample_path(&name).is_some_and(|f| nextsycl_audio::wav_info(&f).is_ok_and(|(_, secs)| secs > 0.0));
                    // a voice that does not say what its sample says: not the models that clone only with the transcript
                    let said = v["text"].as_str().is_some_and(|t| !t.trim().is_empty());
                    let mut speakers: Vec<Value> = speakers.iter().filter(|id| said || !needs_text(&all, id)).cloned().collect();
                    // and the models without an encoder it is enrolled for (experimental: Voxtral TTS)
                    speakers.extend(all.iter().filter(|m| registry::kind_of(m) == "audio" && m["enabled"] != false && enrolled_for(&lib, &name, m))
                        .map(|m| m["id"].clone()));
                    v["ready"] = json!(reads && !speakers.is_empty());
                    v["models"] = json!(speakers);
                    if !reads {
                        v["problem"] = json!("its sample is missing or does not read: add it again");
                    } else if speakers.is_empty() {
                        v["problem"] = json!("no enabled speech model clones a voice (system.enable qwen3-tts-base or voxcpm2)");
                    }
                    v
                }).collect();
                Ok(json!({"voices": list}))
            }
            ("audio", "voices.remove") => {
                self.owned(&s(b, "name"), b)?;
                self.voices().remove(&s(b, "name")).map(|_| json!({"removed": s(b, "name")})).map_err(|e| (404, e))
            }
            ("audio", "voices.add") => {
                let bytes = http::unbase64(&s(b, "sample")).map_err(|e| (400, format!("sample: {e}")))?;
                let o = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty());
                if b["replace"] == true {
                    self.owned(&s(b, "name"), b)?;
                }
                let mut v = self.voices().add(&s(b, "name"), &bytes, o("text"), o("description"), o("language"), "clone", "a recording", b["replace"] == true)
                    .map_err(|e| (400, e))?;
                if let Some(owner) = o("owner") {
                    v = self.voices().tag(&s(b, "name"), "owner", json!(owner)).map_err(|e| (500, e))?;
                }
                Ok(json!({"voice": v, "sample": format!("http://nextsycl/v1/audio/voices/{}.wav", s(b, "name"))}))
            }
            ("audio", "voices.design") => self.design_voice(b, &all),
            ("audio", "voices.enroll") => self.enroll_voice(b, &all),
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

    /// `model3d.tool`: a CPU mesh tool run here, its result saved beside the meshes the 3D servers make
    fn mesh_tool(&self, b: &Value) -> Result<Value, Fail> {
        use nextsycl_model3d::mesh::tools;
        let dir = self.out_dir("model3d");
        let bytes = match (b["mesh"].as_str().filter(|x| !x.is_empty()), b["file"].as_str().filter(|x| !x.is_empty())) {
            (Some(m), _) => http::unbase64(m).map_err(|e| (400, format!("mesh: {e}")))?,
            (None, Some(f)) => {
                let f = f.rsplit('/').next().unwrap_or(f);
                let plain = crate::files::plain(f).filter(|x| x.ends_with(".glb")).ok_or_else(|| (400, format!("file {f}: a GLB's name")))?;
                std::fs::read(dir.join(plain)).map_err(|_| (404, format!("file {f}: no GLB of that name here")))?
            }
            _ => return Err((400, "mesh (GLB or OBJ, base64 / a data: URL / upload:<name>) or file (a GLB made here): required".into())),
        };
        let m = nextsycl_model3d::read_mesh(&bytes).map_err(|e| (400, format!("mesh: {e}")))?;
        let op = s(b, "op");
        let before = m.triangles();
        let t0 = std::time::Instant::now();
        let num = |k: &str| b[k].as_f64();
        let out = match op.as_str() {
            "decimate" => {
                let faces = num("faces").ok_or((400, "faces: required by decimate".to_string()))? as usize;
                let mat = m.material.clone();
                let mut d = tools::decimate(&m, faces.max(4));
                d.material = mat;
                d.compute_normals();
                d
            }
            "remesh" => {
                let mut r = tools::remesh(&m, num("edge").map(|x| x as f32), num("passes").map_or(5, |x| x.clamp(1.0, 20.0) as usize));
                r.compute_normals();
                r
            }
            "uv" => {
                let size = num("size").map_or(2048, |x| x.clamp(64.0, 8192.0) as u32);
                let mut u = tools::unwrap(&m, size, 4);
                if m.material.as_ref().is_some_and(|x| x.base_color.is_some()) || !m.colors.is_empty() {
                    u.material = Some(tools::bake(&u, &m, size).map_err(|e| (400, e))?);
                }
                if u.normals.is_empty() {
                    u.compute_normals();
                }
                u
            }
            "convert" => m,
            other => return Err((400, format!("op {other:?}: decimate, remesh, uv or convert"))),
        };
        let glb = nextsycl_model3d::mesh::glb::write(&out).map_err(|e| (500, e))?;
        std::fs::create_dir_all(&dir).map_err(|e| (500, format!("{}: {e}", dir.display())))?;
        let created = now() as u64;
        let name = format!("{created}-tool-{op}-{}.glb", self.next.lock().map(|mut n| { *n += 1; *n }).unwrap_or(0));
        let path = dir.join(&name);
        std::fs::write(&path, &glb).map_err(|e| (500, format!("{name}: {e}")))?;
        let took = t0.elapsed().as_secs_f64();
        let meta = json!({"model": "tool", "arch": format!("tool:{op}"), "prompt": op, "triangles": out.triangles(), "before": before, "took": took, "created": created,
                          "textured": out.material.as_ref().is_some_and(|x| x.base_color.is_some()), "rigged": out.skin.as_ref().map(|s| s.joints.len()),
                          "animated": out.animation.as_ref().map(|a| a.frames.len())});
        let _ = std::fs::write(path.with_extension("json"), serde_json::to_vec_pretty(&meta).unwrap_or_default());
        let mut e = crate::model3d::entry(&path).unwrap_or(meta);
        e["url"] = json!(format!("http://nextsycl/v1/model3d/files/{name}"));
        Ok(json!({"created": created, "data": [e]}))
    }

    /// A kind's output directory (the servers' NS_IMAGE_OUT / NS_AUDIO_OUT / NS_MODEL3D_OUT)
    fn out_dir(&self, kind: &str) -> std::path::PathBuf {
        let (key, def) = match kind {
            "image" => ("NS_IMAGE_OUT", "images"),
            "model3d" => ("NS_MODEL3D_OUT", "model3d"),
            _ => ("NS_AUDIO_OUT", "audio"),
        };
        std::path::PathBuf::from(self.o.cfg.get(key).unwrap_or_else(|| format!("{}/.local/share/nextsycl/{def}", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))))
    }

    /// `<kind>.upload {name, data}`: a picture (checked, kept as sent) or a recording (any format voices take, kept as
    /// WAV) under <output>/uploads, named up_<unix>_<name>
    fn upload(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        let data = http::unbase64(&s(b, "data")).map_err(|e| (400, format!("data: {e}")))?;
        if data.is_empty() {
            return Err((400, "data: empty".into()));
        }
        let raw = s(b, "name");
        let clean: String = raw.rsplit('/').next().unwrap_or("").chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(60).collect();
        let ext = clean.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default();
        let (stem, bytes) = if kind == "image" {
            if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") {
                return Err((400, format!("name {raw:?}: a picture's name (png, jpg, jpeg, webp)")));
            }
            crate::image::picture(&data, "data").map_err(|e| (400, e))?;
            (clean.clone(), data)
        } else if kind == "model3d" {
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "webp" => {
                    nextsycl_model3d::read_picture(&data).map_err(|e| (400, format!("data: {e}")))?;
                }
                "glb" | "obj" => {
                    nextsycl_model3d::read_mesh(&data).map_err(|e| (400, format!("data: {e}")))?;
                }
                _ => return Err((400, format!("name {raw:?}: a picture's or a mesh's name (png, jpg, jpeg, webp, glb, obj)"))),
            }
            (clean.clone(), data)
        } else {
            if !matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg" | "m4a") {
                return Err((400, format!("name {raw:?}: a recording's name (wav, mp3, flac, ogg, m4a)")));
            }
            // kept as WAV, its channels kept (a song's source is stereo; a voice or a transcript mixes them when read)
            let a = nextsycl_audio::read_audio(&data).map_err(|e| (400, format!("data: {e}")))?;
            (format!("{}.wav", clean.rsplit_once('.').map_or(clean.as_str(), |x| x.0)), a.wav(&[]))
        };
        let dir = self.out_dir(kind).join("uploads");
        std::fs::create_dir_all(&dir).map_err(|e| (500, format!("{}: {e}", dir.display())))?;
        let name = format!("up_{}_{stem}", now() as u64);
        std::fs::write(dir.join(&name), &bytes).map_err(|e| (500, format!("{name}: {e}")))?;
        let mut r = crate::files::stat(&dir.join(&name)).map_err(|e| (500, e))?;
        r["name"] = json!(name);
        r["ref"] = json!(format!("upload:{name}"));
        Ok(r)
    }

    /// The fields of a request that may name an upload, by method: (field, is a list)
    fn upload_fields(name: &str) -> (&'static str, &'static [(&'static str, bool)]) {
        match name {
            "image.generate" => ("image", &[("image", false), ("init_image", false)]),
            "image.edit" => ("image", &[("images", true), ("image", false)]),
            "image.ocr" => ("image", &[("images", true)]),
            "audio.speak" => ("audio", &[("ref_audio", false), ("ref_audios", true)]),
            "audio.transcribe" => ("audio", &[("audio", false)]),
            "audio.generate" => ("audio", &[("audio", false)]),
            "audio.voices.add" => ("audio", &[("sample", false)]),
            "model3d.generate" => ("model3d", &[("image", false), ("images", true)]),
            "image.upscale" | "image.vectorize" => ("image", &[("image", false)]),
            "model3d.rig" | "model3d.tool" => ("model3d", &[("mesh", false)]),
            _ => ("", &[]),
        }
    }

    /// The request with every "upload:<name>" read in as base64 (none named: None); an upload not here is 404
    fn uploads_in(&self, method: &str, b: &Value) -> Result<Option<Value>, Fail> {
        let (kind, fields) = Self::upload_fields(method);
        // a picture of a list may be {image, side} (model3d.generate's views): its image named
        let named = |v: &Value| v.as_str().or_else(|| v["image"].as_str()).and_then(|x| x.strip_prefix("upload:")).map(str::to_string);
        if !fields.iter().any(|(f, list)| if *list { b[*f].as_array().is_some_and(|a| a.iter().any(|v| named(v).is_some())) } else { named(&b[*f]).is_some() }) {
            return Ok(None);
        }
        let dir = self.out_dir(kind).join("uploads");
        let read = |what: String, n: &str| -> Result<Value, Fail> {
            let plain = crate::files::plain(n).filter(|x| x.starts_with("up_")).ok_or_else(|| (400, format!("{what}: upload:{n} is not an upload's name ({kind}.upload answers it)")))?;
            let bytes = std::fs::read(dir.join(plain)).map_err(|_| (404, format!("{what}: no upload {plain} here ({kind}.upload it, or it was deleted)")))?;
            Ok(json!(http::base64(&bytes)))
        };
        let mut out = b.clone();
        for (f, list) in fields {
            if *list {
                if let Some(a) = out[*f].as_array_mut() {
                    for (i, v) in a.iter_mut().enumerate() {
                        if let Some(n) = named(v) {
                            if v.is_object() {
                                v["image"] = read(format!("{f}[{i}].image"), &n)?;
                            } else {
                                *v = read(format!("{f}[{i}]"), &n)?;
                            }
                        }
                    }
                }
            } else if let Some(n) = named(&out[*f]) {
                out[*f] = read(f.to_string(), &n)?;
            }
        }
        Ok(Some(out))
    }

    /// A saved voice that has an owner may be replaced or removed only by a request naming that owner
    fn owned(&self, name: &str, b: &Value) -> Result<(), Fail> {
        let Some(v) = self.voices().get(name) else { return Ok(()) };
        match v["owner"].as_str() {
            Some(o) if o != s(b, "owner").trim() => Err((409, format!("owner: voice {name} belongs to {o:?} - name that owner to replace or remove it"))),
            _ => Ok(()),
        }
    }

    /// system.cards: each GPU in one neutral shape - what any engine on any box can report
    fn cards_view(&self, all: &[Value]) -> Value {
        let mut cards = self.named_cards(all);
        let models = self.model_states(all, &cards);
        self.vouch(&mut cards, &models);
        let list: Vec<Value> = cards.as_array().into_iter().flatten().map(|c| {
            let total = c["vram_total_mb"].as_u64().unwrap_or(0);
            let used = c["vram_used_mb"].as_u64().unwrap_or(0);
            let procs: Vec<Value> = c["procs"].as_array().into_iter().flatten().map(|p| json!({"pid": p["pid"], "program": p["program"], "kind": p["kind"],
                "model": p["model_id"].as_str().or(p["model"].as_str()), "vram_mb": p["vram_mb"]})).collect();
            let models: Vec<Value> = c["procs"].as_array().into_iter().flatten().filter_map(|p| p["model_id"].as_str().map(|m| json!(m))).collect();
            json!({"gpu": c["index"], "name": c["name"], "bus": c["pci"], "vram_total_mb": total, "vram_used_mb": used, "vram_free_mb": total.saturating_sub(used),
                   "busy_pct": c["busy_pct"], "power_w": c["power_w"], "power_cap_w": c["power_cap_w"], "temp_c": c["temp_pkg"], "temp_vram_c": c["temp_vram"],
                   "models": models, "processes": procs, "vouched_mb": c["vouched_mb"], "overbooked": c["overbooked"],
                   "enabled_models": c["vouched"].as_array().into_iter().flatten().map(|v| v["model"].clone()).collect::<Vec<_>>()})
        }).collect();
        json!({"cards": list})
    }

    /// system.files.stat / files.delete: a picture or sound in its kind's output directory, or through the studio
    /// a clip or upload
    fn file_op(&self, op: &str, b: &Value) -> Result<Value, Fail> {
        let kind = s(b, "kind");
        let name = crate::files::plain(&s(b, "name")).map(str::to_string).ok_or_else(|| (400, "name: a plain file name (the last part of its URL)".to_string()))?;
        if kind == "video" {
            let (st, v) = http::call_status(&self.target("video"), "POST", &format!("/rpc/{op}"), Some(&json!({"name": name})), 120)
                .map_err(|e| (502, format!("the video studio does not answer: {e}")))?;
            if v["ok"] != true {
                return Err((st, s(&v["error"], "message")));
            }
            let mut r = v["result"].clone();
            if op == "files.stat" {
                r["url"] = json!(format!("http://nextsycl/v1/video/files/{name}"));
            }
            return Ok(r);
        }
        let sub = match kind.as_str() {
            "image" => "images",
            "audio" => "audio",
            "model3d" => "model3d",
            _ => return Err((400, "kind: image, audio, video or model3d".into())),
        };
        // an upload (up_...) is kept under uploads/
        let upload = name.starts_with("up_");
        let dir = if upload { self.out_dir(&kind).join("uploads") } else { self.out_dir(&kind) };
        let path = dir.join(&name);
        // what the servers make, not the saved voices or anything else kept there
        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        let ok = match kind.as_str() {
            "image" => matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp"),
            // a mesh and the picture and settings beside it; an upload (a picture or a mesh)
            "model3d" => matches!(ext.as_str(), "glb" | "obj" | "png" | "jpg" | "jpeg" | "webp" | "json"),
            _ => ext == "wav",
        };
        if !ok || !path.is_file() {
            return Err((404, format!("no {kind} file {name}")));
        }
        if op == "files.stat" {
            let mut r = crate::files::stat(&path).map_err(|e| (404, e))?;
            r["url"] = if upload { Value::Null } else { json!(format!("http://nextsycl/v1/{sub}/files/{name}")) };
            if upload {
                r["ref"] = json!(format!("upload:{name}"));
            }
            return Ok(r);
        }
        std::fs::remove_file(&path).map_err(|e| (500, format!("{name}: {e}")))?;
        Ok(json!({"deleted": [name]}))
    }

    /// image.canvases.list: the model's tested sizes, with the time measured on the pictures kept
    fn image_canvases(&self, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        let id = self.pick("image", b, all)?;
        let hist = call_for(&self.file_target("image"), "GET", "/api/history", None, 10).unwrap_or(json!({"data": []}));
        let def = all.iter().find(|m| m["id"] == id.as_str()).and_then(|m| m["defaults"]["size"].as_str().map(str::to_string)).unwrap_or_else(|| "1024x1024".into());
        let model = all.iter().find(|m| m["id"] == id.as_str()).cloned().unwrap_or(Value::Null);
        let grid = super::precheck::size_multiple(&model);
        let list: Vec<Value> = IMAGE_CANVASES.iter().filter(|(w, h, _, _)| w % grid == 0 && h % grid == 0).map(|(w, h, a, note)| {
            // seconds per step of the pictures of this size (several to a request: its time over them)
            let made: Vec<f64> = hist["data"].as_array().into_iter().flatten()
                .filter(|e| e["width"] == *w && e["height"] == *h && e["kind"] != "ocr")
                .filter_map(|e| Some(e["seconds"].as_f64()? / e["steps"].as_f64().filter(|s| *s > 0.0)?)).collect();
            let size = format!("{w}x{h}");
            json!({"size": size, "width": w, "height": h, "aspect": a, "note": note, "mpx": ((w * h) as f64 / 1e4).round() / 100.0, "default": size == def,
                   "seconds": if made.is_empty() { Value::Null } else { json!((made.iter().sum::<f64>() / made.len() as f64 * 100.0).round() / 100.0) },
                   "made": made.len()})
        }).collect();
        let rule = match model["arch"].as_str() {
            Some("moss-image") => "any width and height that are multiples of 16, 64 to 4096; 1024 x 1024 is the default and 2048 x 2048 works (no tiling), several times the time".to_string(),
            Some("hunyuanimage-2.1") => "any width and height that are multiples of 32; 2048 x 2048 is native".to_string(),
            _ => "any width and height that are multiples of 16; the model was trained up to about 1.8 megapixels (1328 x 1328) - larger works, slower and less sure".to_string(),
        };
        Ok(json!({"model": id, "rule": rule, "canvases": list}))
    }

    /// A file into the studio's output (its upload route takes the raw bytes)
    fn studio_upload(&self, name: &str, data: &[u8]) -> Result<Value, Fail> {
        use std::io::{Read, Write};
        let mut c = Conn::connect(&self.target("video")).map_err(|e| (502, format!("the studio does not answer: {e}")))?;
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
                let model = serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string));
                // a card's model not downloaded is said as such (chat_target refuses or fetches it), not "none enabled"
                if !model.as_deref().is_some_and(|m| self.not_downloaded(m).is_some()) {
                    need("llm")?;
                }
                let t = self.chat_target(model.as_deref())?;
                let r = pass(s, &t);
                if let Some(secs) = crate::switch::keep_up_s(&req.head) {
                    if let Some(id) = self.chat_loaded(&self.models()) {
                        self.keep_chat(&id, secs);
                    }
                }
                r
            }
            ("POST", p @ ("/v1/images/generations" | "/v1/images/edits" | "/v1/audio/speech")) => {
                let kind = if p.starts_with("/v1/images") { "image" } else { "audio" };
                // the model the body names (a JSON field, or a multipart part)
                let ctype = http::header(&req.head, "content-type").unwrap_or_default();
                let named = if let Some(b) = ctype.split("boundary=").nth(1) {
                    http::multipart(&req.body, b.trim_matches('"')).ok()
                        .and_then(|ps| ps.into_iter().find(|p| p.name == "model").map(|p| String::from_utf8_lossy(&p.data).trim().to_string()))
                } else {
                    serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["model"].as_str().map(str::to_string))
                };
                // a card's model not downloaded: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first
                let fetched = named.as_deref().filter(|n| self.not_downloaded(n).is_some()).map(str::to_string);
                if let Some(n) = &fetched {
                    self.need_downloaded(n, None, true)?;
                }
                let all = if fetched.is_some() { self.models() } else { all.clone() };
                if fetched.is_none() {
                    need(kind)?;
                }
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
            ("POST", "/v1/audio/transcriptions") => {
                need("audio")?;
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                // OpenAI's whisper-1 (or no name): the recogniser here
                let named = body["model"].as_str().filter(|n| all.iter().any(|m| m["id"] == *n));
                let id = self.recogniser(&json!({"model": named}), &all)?;
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("GET", p) if p.starts_with("/v1/images/files/") => pass(s, &self.file_target("image")),
            ("GET", p) if p.starts_with("/v1/audio/files/") => pass(s, &self.file_target("audio")),
            ("GET", p) if p.starts_with("/v1/model3d/files/") => {
                // from the directory (a tool's result needs no 3D server running)
                let name = p["/v1/model3d/files/".len()..].split('?').next().unwrap_or("");
                let kind = match name.rsplit('.').next() {
                    Some("glb") => "model/gltf-binary",
                    Some("png") => "image/png",
                    Some("json") => "application/json",
                    _ => return Err((404, format!("no file {name}"))),
                };
                let plain = crate::files::plain(name).ok_or_else(|| (404, format!("no file {name}")))?;
                let b = std::fs::read(self.out_dir("model3d").join(plain)).map_err(|_| (404, format!("no file {name}")))?;
                http::respond_bytes(s, 200, kind, &b, "Cache-Control: max-age=31536000, immutable\r\n");
                Ok(())
            }
            ("POST", p @ ("/v1/model3d/generate" | "/v1/model3d/rig")) => {
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                // a card's 3D model not downloaded: refused, or with NS_AUTO_PULL fetched first
                let fetched = body["model"].as_str().filter(|n| self.not_downloaded(n).is_some()).map(str::to_string);
                if let Some(n) = &fetched {
                    self.need_downloaded(n, None, true)?;
                } else {
                    need("model3d")?;
                }
                let all = if fetched.is_some() { self.models() } else { all.clone() };
                let named = body["model"].as_str().filter(|n| all.iter().any(|m| m["id"] == *n)).map(str::to_string);
                let painting = !p.ends_with("/rig") && ["mesh", "file"].iter().any(|k| body[*k].as_str().is_some_and(|v| !v.is_empty()));
                let want = if p.ends_with("/rig") { "rig" } else if painting { "paint" } else { "generate" };
                let id = match named {
                    Some(n) => self.pick("model3d", &json!({"model": n}), &all)?,
                    None => all.iter().filter(|m| registry::kind_of(m) == "model3d" && m["enabled"] != false)
                        .find(|m| m["does"].as_array().is_none_or(|a| a.iter().any(|x| x == want || (want == "rig" && x == "animate"))))
                        .map(|m| m["id"].as_str().unwrap_or("").to_string()).ok_or_else(|| (409u16, format!("no enabled 3D model does {want}")))?,
                };
                let t = self.ensure(&id)?;
                pass(s, &t)
            }
            ("GET", p) if p.starts_with("/v1/video/files/") => {
                // a clip in the studio's output (its /out/ route; a Range header passed on for seeking)
                need("video")?;
                let name = p["/v1/video/files/".len()..].split('?').next().unwrap_or("");
                let still = name.ends_with(".preview.png") || name.ends_with(".last.png");
                if name.is_empty() || name.contains('/') || !(name.ends_with(".mp4") || still) {
                    return Err((404, format!("no clip {name}")));
                }
                let head = String::from_utf8_lossy(&req.head).into_owned();
                let range = head.split("\r\n").find(|l| l.split(':').next().is_some_and(|k| k.trim().eq_ignore_ascii_case("range")));
                let up = format!("GET /out/{name} HTTP/1.1\r\nHost: studio\r\n{}Connection: close\r\n\r\n", range.map(|r| format!("{r}\r\n")).unwrap_or_default());
                let req = Request { method: "GET".into(), path: format!("/out/{name}"), target: format!("/out/{name}"), head: up.into_bytes(),
                                    origin: None, host: None, body: Vec::new() };
                http::forward(s, &self.target("video"), &req).map_err(|e| (502, e))
            }
            ("GET", p) if p.starts_with("/v1/video/thumbs/") => {
                need("video")?;
                let name = &p["/v1/video/thumbs/".len()..];
                if name.is_empty() || name.contains('/') || !name.ends_with(".webp") {
                    return Err((404, format!("no still {name}")));
                }
                let req = Request { method: "GET".into(), path: format!("/thumb/{name}"), target: format!("/thumb/{name}"),
                                    head: format!("GET /thumb/{name} HTTP/1.1\r\nHost: studio\r\nConnection: close\r\n\r\n").into_bytes(),
                                    origin: None, host: None, body: Vec::new() };
                http::forward(s, &self.target("video"), &req).map_err(|e| (502, e))
            }
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

    /// The OpenAPI document of what the enabled kinds offer. Open (GET /openapi.yml needs no token), so it carries
    /// none: `<token>` in the examples; the page's own copy (`token`) has it filled in
    pub(crate) fn openapi(&self, req: &Request, token: bool) -> Value {
        let tok = if token { self.token.as_str() } else { "<token>" };
        let all = self.models();
        let on = self.kinds_on(&all);
        let host = req.host.clone().unwrap_or_else(|| "localhost:8000".into());
        let ids = |k: &str| -> Vec<Value> { all.iter().filter(|m| registry::kind_of(m) == k && m["enabled"] != false).map(|m| m["id"].clone()).collect() };
        let envelope = json!({"type": "object", "properties": {
            "ok": {"type": "boolean"}, "result": {"description": "the method's answer"},
            "error": {"type": "object", "required": ["code", "message", "retryable", "status"], "description": "ok false: everything about the failure (the HTTP status is 200 all the same)", "properties": {
                "code": {"type": "string", "description": "the convention to branch on (the list in the description above): invalid_request, invalid_params, unauthorized, not_found, conflict, busy, not_running, model_not_downloaded, no_room, cancelled, unavailable, timeout, request_too_large, internal"},
                "message": {"type": "string", "description": "what was wrong, in words: the value, the limit or the nearest legal one, what to do"},
                "retryable": {"type": "boolean", "description": "true: the same call may work later unchanged (no_room, unavailable, timeout, busy); false: change the request"},
                "status": {"type": "integer", "description": "the code's number by convention (400, 404, 409, 499, 502, 503 ...): in the body, not the HTTP status, so another transport carries it"},
                "method": {"type": "string", "nullable": true, "description": "the method called"},
                "field": {"type": "string", "nullable": true, "description": "the request key at fault, when the message names one (steps, voice, images, size ...)"}}}}});
        let engines: Vec<String> = if on.contains(&"video") {
            call_for(&self.target("video"), "POST", "/rpc/engines.list", Some(&json!({})), 3).ok()
                .and_then(|v| v["result"]["engines"].as_array().map(|a| a.iter().filter_map(|e| e["quant"].as_str().map(str::to_string)).collect()))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut paths = Map::new();
        for m in table::methods().iter().filter(|m| on.contains(&m.kind)) {
            let mut props = Map::new();
            let mut required = Vec::new();
            for fd in &m.fields {
                // the field as its request struct declares it (type, #[schema] limits, its doc comment)
                let f = &fd.name.as_str();
                let req = &!fd.optional;
                // an optional field may be left out: its schema is the value's, not "or null"
                let mut bare = fd.clone();
                if let rpcgen::Shape::Nullable(inner) = &fd.shape {
                    bare.shape = (**inner).clone();
                }
                let mut p = rpcgen::field_schema(&bare);
                if !fd.doc.is_empty() {
                    p["description"] = json!(fd.doc);
                }
                if *f == "model" && m.kind != "system" {
                    p["enum"] = Value::Array(ids(m.kind));
                }
                // the names the studio takes now: the daemon's denoisers, ComfyUI's samplers and schedules
                if m.kind == "video" && m.name == "generate" {
                    match *f {
                        "engine" if !engines.is_empty() => p["enum"] = json!(engines),
                        "sampler" => p["enum"] = json!(nextsycl_diffusion::samplers::NAMES),
                        "schedule" => p["enum"] = json!(nextsycl_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>()),
                        _ => {}
                    }
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
            let example: Value = m.example.clone();
            let route = format!("/rpc/{}.{}", m.kind, m.name);
            let op = format!("{}.{}", m.kind, m.name);
            let mut ok = envelope.clone();
            if let Some(r) = spec::result(&op) {
                ok["properties"]["result"] = serde_json::from_str(&r).unwrap_or_else(|e| panic!("{op} result: {e}"));
            }
            // every answer is HTTP 200 with the envelope; the errors it can carry are listed (x-errors, and in words)
            let mut errors = vec![json!({"status": 400, "code": "invalid_request", "when": "not POST, a query string, the body not a JSON object"}),
                                  json!({"status": 400, "code": "invalid_params", "when": "a value refused (error.message and error.field say which)"}),
                                  json!({"status": 401, "code": "unauthorized", "when": "no or a wrong token"})];
            let mut add = |st: u16, code: &str, when: String| match errors.iter_mut().find(|e| e["code"] == code && e["status"] == st) {
                // one entry a code: the method's own case first
                Some(e) => e["when"] = json!(format!("{when}; {}", e["when"].as_str().unwrap_or(""))),
                None => errors.push(json!({"status": st, "code": code, "when": when})),
            };
            for (st, code, when) in spec::errors(&op) {
                add(*st, code, when.to_string());
            }
            if m.kind != "system" {
                add(409, "conflict", format!("no {} model is enabled", m.kind));
            }
            if super::maint::starts_work(&op) {
                add(spec::MAINTENANCE.0, spec::MAINTENANCE.1, spec::MAINTENANCE.2.to_string());
            }
            let words: Vec<String> = errors.iter().map(|e| format!("{} {}: {}", e["code"].as_str().unwrap_or(""), e["status"], e["when"].as_str().unwrap_or(""))).collect();
            let responses = json!({"200": {"description": format!("{{\"ok\": true, \"result\": ...}}, or {{\"ok\": false, \"error\": ...}} - errors are HTTP 200 too. Its errors: {}", words.join("; ")),
                                           "content": {"application/json": {"schema": ok}}}});
            let curl = format!("curl -s http://{host}{route} -H 'Authorization: Bearer {}' -H 'Content-Type: application/json' -d '{}'",
                               tok, example.to_string().replace('\'', "'\\''"));
            paths.insert(route, json!({"post": {
                "tags": [m.kind], "operationId": format!("{}.{}", m.kind, m.name), "summary": m.summary,
                "requestBody": {"required": true, "content": {"application/json": {"schema": schema, "example": example}}},
                "responses": responses,
                "x-errors": errors,
                "x-curl": curl,
            }}));
        }
        let oa = |summary: &str, body: Value, answer: &str| {
            let schema: Value = spec::result(answer).and_then(|r| serde_json::from_str(&r).ok()).unwrap_or(json!({"type": "object"}));
            json!({"tags": ["openai"], "summary": summary, "requestBody": {"required": true, "content": {"application/json": {"example": body}}},
                   "responses": {"200": {"description": format!("OpenAI's shape, no envelope (as {answer}'s result)"), "content": {"application/json": {"schema": schema}}},
                                 "4XX": {"description": "OpenAI's error: {\"error\": {\"message\", \"type\"}}"}}})
        };
        // a file an answer links: the bearer token like every route here; byte ranges where the kind's server serves them
        let file = |summary: &str, name: &str, mime: &str, ranges: bool| {
            let mut params = vec![json!({"name": "name", "in": "path", "required": true, "schema": {"type": "string"}, "description": name})];
            let mut responses = json!({"200": {"description": format!("the file ({mime})"), "content": {mime: {"schema": {"type": "string", "format": "binary"}}}},
                                       "401": {"description": "unauthorized: the bearer token is needed here too"},
                                       "404": {"description": "no such file"}});
            if ranges {
                params.push(json!({"name": "Range", "in": "header", "required": false, "schema": {"type": "string"},
                                   "description": "bytes=a-b, bytes=a- or bytes=-n (one span): answered 206 with Content-Range; Accept-Ranges: bytes"}));
                responses["206"] = json!({"description": "the span asked for", "content": {mime: {"schema": {"type": "string", "format": "binary"}}}});
            }
            json!({"get": {"tags": ["files"], "summary": format!("{summary}{}", if ranges { "; byte ranges (Range) supported" } else { "; no byte ranges (always the whole file)" }),
                           "security": [{"bearer": []}], "parameters": params, "responses": responses}})
        };
        paths.insert("/v1/models".into(), json!({"get": {"tags": ["openai"], "summary": "The enabled chat, image and audio models", "responses": {"200": {"description": "OpenAI's list"}}}}));
        if on.contains(&"llm") {
            paths.insert("/v1/chat/completions".into(), json!({"post": oa("Chat (streams with stream: true; tools)",
                json!({"model": ids("llm").first(), "messages": [{"role": "user", "content": "Hello"}], "stream": false}), "llm.chat")}));
        }
        if on.contains(&"image") {
            paths.insert("/v1/images/generations".into(), json!({"post": oa("Pictures (and ours: steps, seed, sampler, schedule, cfg, negative_prompt, loras)",
                json!({"model": ids("image").first(), "prompt": "a lighthouse at dusk", "size": "1024x1024"}), "image.generate")}));
            paths.insert("/v1/images/edits".into(), json!({"post": oa("Edits: multipart (image, image[] ...) or JSON images",
                json!({"prompt": "make it night", "images": ["data:image/png;base64,..."]}), "image.edit")}));
            paths.insert("/v1/images/files/{name}".into(), file("A picture an answer links (data[].url of image.generate / image.edit)", "<file>.png", "image/png", false));
        }
        if on.contains(&"audio") {
            paths.insert("/v1/audio/speech".into(), json!({"post": oa("A song (input: the lyrics, instructions: the music) or speech (input: the text; voice, \
                language, instructions, ref_audio, ref_text); response_format wav, pcm (16-bit mono 24 kHz), mp3, opus, aac, flac or url; \
                pcm and the encoded formats are sent as the sound is made (Transfer-Encoding: chunked, as OpenAI's), wav with stream: true; \
                speed 0.25-4; input may be a list of texts, said in turn as one stream (gap seconds between); stream_format sse sends \
                speech.audio.delta events",
                json!({"model": ids("audio").first(), "instructions": "warm folk", "input": "[verse]\nRiver runs slow", "response_format": "url"}), "audio.generate")}));
            paths.insert("/v1/audio/files/{name}".into(), file("A song or speech an answer links (data[].url of audio.generate / audio.speak)", "<file>.wav", "audio/wav", true));
            paths.insert("/v1/audio/voices/{name}".into(), file("A saved voice's sample (audio.voices.list's sample)", "<voice>.wav", "audio/wav", false));
        }
        if on.contains(&"model3d") {
            paths.insert("/v1/model3d/generate".into(), json!({"post": oa("A mesh from a picture (image) or several views (images: [{image, side}]); texture, seed, steps, faces; response_format glb or url",
                json!({"model": ids("model3d").first(), "image": "data:image/png;base64,iVBORw0...", "texture": true, "response_format": "url"}), "model3d.generate")}));
            paths.insert("/v1/model3d/rig".into(), json!({"post": oa("A mesh rigged (skeleton and skin weights) or animated (prompt: the motion); mesh as base64 GLB / OBJ, or file: a GLB made here",
                json!({"model": ids("model3d").first(), "file": "1792132541-trellis2-42.glb", "response_format": "url"}), "model3d.rig")}));
            paths.insert("/v1/model3d/files/{name}".into(), file("A mesh an answer links (data[].url of model3d.generate / model3d.rig), or its picture (.png)", "<file>.glb", "model/gltf-binary", true));
        }
        if on.contains(&"video") {
            paths.insert("/v1/video/files/{name}".into(), file("A finished clip: <clip>.mp4 (video.jobs.list's file, video.status's and video.timeline's file URLs); also <clip>.preview.png (video.status's preview while it samples) and <clip>.last.png (a clip's last frame), image/png", "<clip>.mp4, e.g. h3_20261011_220723.mp4", "video/mp4", true));
            paths.insert("/v1/video/thumbs/{name}".into(), file("A finished clip's 240-pixel still, 5 s in (video.timeline's thumb)", "<clip>.webp", "image/webp", false));
        }
        json!({
            "openapi": "3.0.3",
            "info": {"title": "nextsycl", "version": crate::VERSION, "description": format!(
                "The box's API (nextsycl serve). Rules: POST /rpc/<kind>.<method>, every variable in the JSON body - no path variables, no query \
                 strings; answers {{\"ok\": true, \"result\": ...}} or {{\"ok\": false, \"error\": {{\"code\", \"message\", ...}}}}, always HTTP 200. The \
                 OpenAI-compatible /v1 routes and the files are the exception (HTTP statuses). Send Authorization: Bearer <token> with every /rpc and /v1 call (the token is on the page's API tab; \
                 this document needs none). Kinds here: {}. Generated {:.0}.\n\n{}\n\n{}",
                on.join(", "), now(), spec::CODES, spec::LABELS)},
            "servers": [{"url": format!("http://{host}")}],
            "security": [{"bearer": []}],
            // the named types the methods' requests refer to (their enums: ChainMode, FileKind ...), from methods.rs
            "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": "the token on the page's API tab (NS_API_TOKEN, or serve.json)"}},
                           "schemas": rpcgen::components(&table::infos()).unwrap_or_default()},
            "paths": paths,
        })
    }
}

/// Whether a speech model clones only with what the recording says (its card's `speech` lists `transcript`)
pub(super) fn needs_text(all: &[Value], id: &Value) -> bool {
    all.iter().find(|m| &m["id"] == id).is_some_and(|m| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "transcript")))
}

/// Whether a model speaks the saved voice `voice` by its enrolment (its card's `speech` lists `enrolled`, and the voice
/// has a fit for its architecture)
pub(super) fn enrolled_for(lib: &crate::voices::Library, voice: &str, m: &Value) -> bool {
    m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "enrolled")) && m["arch"].as_str().is_some_and(|arch| lib.enrolled_path(voice, arch).is_some())
}

/// Whether a speech model speaks the built-in voice `voice`: it lists it (any case), or lists none (its config or card
/// names no voices: any name is its engine's to check)
pub(super) fn speaks(m: &Value, voice: &str) -> bool {
    let (own, _) = super::precheck::speech_lists(m);
    own.is_empty() || own.contains(&voice.to_lowercase())
}

/// A registry entry that upscales (an upscaler's architecture) rather than draws or reads
pub(super) fn upscales(m: &Value) -> bool {
    matches!(s(m, "arch").as_str(), "realesrgan" | "waifu2x")
}

/// A registry entry that writes pictures as SVG (StarVector) rather than draws them
pub(super) fn vectorizes(m: &Value) -> bool {
    s(m, "arch") == "starvector"
}

#[cfg(test)]
mod tests {
    use super::{code_of, enrolled_for, rpc_error, speaks, table};

    #[test]
    fn nextsycls_methods_are_all_declared() {
        let has = |n: &str| table::methods().iter().any(|m| m.full() == n);
        for n in ["system.ping", "model3d.generate", "model3d.rig", "model3d.tool", "model3d.describePromptGuideMD", "system.drives", "system.move",
                  "audio.speak", "audio.voices.enroll", "image.edit", "video.generate", "video.describePromptGuideMD"] {
            assert!(has(n), "{n} is declared");
        }
        let m = |k: &str, n: &str| table::methods().iter().find(|m| m.kind == k && m.name == n).unwrap();
        // nextsycl's own keys: a multi-speaker script's voices, the LoRAs of an edit
        assert!(m("audio", "speak").fields.iter().any(|f| f.name == "voices") && m("image", "edit").fields.iter().any(|f| f.name == "loras"));
    }

    #[test]
    fn an_enrolled_voice_goes_to_its_model() {
        let dir = std::env::temp_dir().join(format!("ns-enrolled-{}", std::process::id()));
        let lib = crate::voices::Library::of(&dir);
        let vox = serde_json::json!({"id": "voxtral-4b-tts", "arch": "voxtral-tts", "speech": ["voices", "enrolled"]});
        assert!(!enrolled_for(&lib, "anna", &vox));
        let f = lib.enrolled_file("anna", "voxtral-tts").unwrap();
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(&f, b"x").unwrap();
        assert!(enrolled_for(&lib, "anna", &vox));
        // a model whose card does not speak enrolled voices, or of another architecture
        assert!(!enrolled_for(&lib, "anna", &serde_json::json!({"arch": "voxtral-tts", "speech": ["voices"]})));
        assert!(!enrolled_for(&lib, "anna", &serde_json::json!({"arch": "qwen3-tts", "speech": ["enrolled"]})));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_built_in_voice_goes_to_a_model_that_has_it() {
        let vox = serde_json::json!({"id": "voxtral-4b-tts", "voices": ["neutral_female", "fr_male"]});
        assert!(speaks(&vox, "Neutral_Female"));
        assert!(!speaks(&vox, "serena"));
        // no list: the engine decides
        assert!(speaks(&serde_json::json!({"id": "x"}), "serena"));
    }

    #[test]
    fn an_rpc_error_says_everything_in_its_body() {
        let e = rpc_error("image.generate", 400, None, "steps 8: qwen-image-2.1-turbo-q8 runs its own 6 sigmas");
        assert_eq!((e["code"].as_str(), e["status"].as_u64(), e["retryable"].as_bool()), (Some("invalid_params"), Some(400), Some(false)));
        assert_eq!((e["method"].as_str(), e["field"].as_str()), (Some("image.generate"), Some("steps")));
        // a missing field, named in quotes
        assert_eq!(rpc_error("image.generate", 400, None, "image.generate needs \"prompt\"")["field"], "prompt");
        // a key that is not the method's is no field
        assert!(rpc_error("image.generate", 400, None, "qwen-image is busy")["field"].is_null());
        // no method (an unreadable request): null, not ""
        assert!(rpc_error("", 400, Some("request_too_large"), "over 256 MiB")["method"].is_null());
    }

    #[test]
    fn retry_follows_the_code_not_the_status_alone() {
        assert_eq!(rpc_error("system.load", 503, None, "its cards are busy")["code"], "no_room");
        assert_eq!(rpc_error("system.load", 503, None, "x")["retryable"], true);
        assert_eq!(rpc_error("video.generate", 409, Some("busy"), "a clip is being made")["retryable"], true);
        assert_eq!(rpc_error("video.cancel", 409, Some("not_running"), "no job is running")["retryable"], false);
        assert_eq!(rpc_error("image.enable", 409, None, "disabled")["retryable"], false);
        assert_eq!(rpc_error("audio.speak", 499, None, "cancelled")["code"], "cancelled");
        let e = rpc_error("image.generate", 503, None, "cards taken: a GPU run holds /tmp/ds2run.lock (pid 7)");
        assert_eq!((e["code"].as_str(), e["retryable"].as_bool(), e["retry_after_s"].as_u64()), (Some("cards_taken"), Some(true), Some(120)));
        for (st, c) in [(400, "invalid_params"), (401, "unauthorized"), (404, "not_found"), (502, "unavailable"), (504, "timeout"), (500, "internal")] {
            assert_eq!(code_of(st), c);
        }
    }
}
