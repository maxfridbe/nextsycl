//! The audio server: the reference server's speech endpoint (SGLang-Omni's, which MiniMax Music 3 ships with) over one
//! audio engine (`nextsycl_audio::AudioEngine`), and with `wfe` a web page that drives it with every option it has.
//!
//! ```text
//!   POST /v1/audio/speech         a song: {input: the lyrics, instructions: the description, seed, max_new_tokens
//!                                 (frames, 25 a second) | seconds, response_format: wav (the file) | url (JSON with its
//!                                 link)} and ours: steps, cfg, options {NAME: value} (the engine's own: /api/info);
//!                                 speech: {input: the text, voice, language, instructions (its style; a designed
//!                                 voice's description), ref_audio (base64 WAV / data URL: a voice to clone),
//!                                 ref_text, continue_from (a speech file made here: its last sentences are the
//!                                 reference, after the voice's - the voice carries on), seconds (the most), seed,
//!                                 response_format (wav | pcm | mp3 | opus | aac | flac | url), speed (0.25 - 4),
//!                                 sample_rate, options}; OpenAI's own shape - pcm (16-bit mono, 24 kHz) and the
//!                                 encoded formats are always sent as the sound is made, `Transfer-Encoding:
//!                                 chunked` (curl ... --output - | ffplay -f s16le -ar 24k -ac 1 -nodisp -); wav
//!                                 with stream: true / stream_format: audio (unknown lengths); the rate is in
//!                                 X-Sample-Rate. Another format, speed or rate goes through ffmpeg. `input` as a
//!                                 list of texts: said in turn as ONE stream (the same voice and seed, `gap`
//!                                 seconds between, 0.3), stream_format: sse - speech.audio.delta events (base64
//!                                 16-bit PCM), then speech.audio.done with the file's link (an engine that does not
//!                                 stream sends its sound whole at the end)
//!   POST /v1/audio/transcriptions a recording written down (a recogniser: whisper): {audio (base64 of WAV, MP3, FLAC, Ogg,
//!                                 M4A, or a data URL), language (a code or a name; none: detected), words (bool, true:
//!                                 each word timed), translate (into English), prompt (text leading the first window),
//!                                 options} -> {text, language, duration, segments [{start, end, text}], words [{start,
//!                                 end, text, probability}], nextsycl: {took, wh}} (JSON only, no multipart form)
//!   GET  /v1/audio/files/<f>      a song made here (saved in the output directory; byte ranges for seeking)
//!   POST /api/cancel              stops the request running (between frames or steps)
//!   GET  /v1/models, /health
//!   GET  /api/info                the model, its defaults and limits, its options
//!   GET  /api/progress            the request running: its phase (prompt, tokens, flow, decode), how far, seconds
//!   GET  /api/history             the songs in the output directory, newest first (their settings from the WAVs)
//!   GET  /api/gpu                 the card: VRAM, power, temperatures
//!   GET  /, /ui/..., /static/...  the web front end (--wfe: wfe/audio, built into dist/wfe; H3's scheme)
//! ```
//!
//! One request runs at a time (one engine on its GPU), or as many as the engine makes together (`AudioEngine::batch`:
//! a speech engine that advances several requests a frame at a time); the others wait their turn.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nextsycl_audio::{AudioEngine, AudioRequest};
use nextsycl_core::Gpu;
use serde_json::{json, Value};

use crate::http::{self, Conn};
use crate::speechfmt::{self, Fmt, PCM_RATE};
use crate::serve::cors_headers;
use crate::telemetry::Telemetry;

pub struct AudioServer {
    /// the model id clients name
    pub model: String,
    /// where songs are saved (and served from); none: only returned
    pub out_dir: Option<PathBuf>,
    /// the built front ends (dist/wfe); none: no page
    pub wfe: Option<PathBuf>,
    pub cors: Vec<String>,
    gpu: Arc<Gpu>,
    tele: Arc<Telemetry>,
    /// the engine: shared by the requests it makes together (`AudioEngine::batch`), held whole to end the process
    engine: RwLock<Box<dyn AudioEngine>>,
    /// requests on the engine now (at most its batch; the others wait here)
    slots: (Mutex<usize>, Condvar),
    progress: Mutex<Value>,
    waiting: AtomicUsize,
    cancel: crate::cancel::Cancel,
    history: Mutex<VecDeque<Value>>,
    /// ends the process after an idle stretch (`--idle-exit`; none: never)
    pub idle: Option<&'static crate::idle::Idle>,
    /// this process's card memory once the model was loaded (MiB), and whether to end after the request in hand: a
    /// GPU error, or memory grown well past the loaded model's (caches that keep a piece of every request's shape) -
    /// the front door starts a fresh server for the next request, so a leak never fills the card
    vram_loaded: AtomicU64,
    retire: AtomicBool,
}

/// A float32 setting as JSON without its binary tail (1.7, not 1.7000000476837158)
fn f(x: f32) -> Value {
    json!(((x as f64) * 1e4).round() / 1e4)
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn err(m: impl Into<String>) -> Value {
    json!({"error": {"message": m.into(), "type": "invalid_request_error"}})
}

/// A song's history entry from its file: the description (INAM), lyrics (ILYR), settings (ICMT, JSON)
fn entry(p: &std::path::Path, created: u64) -> Option<Value> {
    let (info, seconds) = nextsycl_audio::wav_info(p).ok()?;
    let get = |k: &str| info.iter().find(|(i, _)| i == k).map(|(_, v)| v.clone());
    let mut e: Value = get("ICMT").and_then(|c| serde_json::from_str(&c).ok()).unwrap_or_else(|| json!({}));
    e["url"] = json!(format!("/v1/audio/files/{}", p.file_name()?.to_string_lossy()));
    e["prompt"] = json!(get("INAM").unwrap_or_default());
    e["lyrics"] = json!(get("ILYR").unwrap_or_default());
    e["seconds"] = json!(seconds);
    if e.get("created").is_none() {
        e["created"] = json!(created);
    }
    Some(e)
}

/// The songs already in `dir`, newest first
fn earlier(dir: &std::path::Path) -> VecDeque<Value> {
    let mut files: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files.truncate(200);
    files.into_iter().filter_map(|(t, p)| entry(&p, t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0))).collect()
}

/// A Range header's first span ("bytes=a-b", "bytes=a-", "bytes=-n") within `len` bytes
fn range(head: &[u8], len: usize) -> Option<(usize, usize)> {
    let h = String::from_utf8_lossy(head);
    let v = h.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case("range").then(|| v.trim().to_string())
    })?;
    let spec = v.strip_prefix("bytes=")?.split(',').next()?.trim().to_string();
    let (a, b) = spec.split_once('-')?;
    let (start, end) = match (a.trim().parse::<usize>().ok(), b.trim().parse::<usize>().ok()) {
        (Some(a), Some(b)) => (a, b.min(len.saturating_sub(1))),
        (Some(a), None) => (a, len.saturating_sub(1)),
        (None, Some(n)) => (len.saturating_sub(n), len.saturating_sub(1)),
        (None, None) => return None,
    };
    (start <= end && end < len).then_some((start, end))
}

impl AudioServer {
    pub fn new(model: String, gpu: Arc<Gpu>, engine: Box<dyn AudioEngine>, out_dir: Option<PathBuf>, wfe: Option<PathBuf>, cors: Vec<String>)
               -> Result<AudioServer, String> {
        if let Some(d) = &out_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        if let Some(w) = &wfe {
            if !w.join("audio/index.html").is_file() {
                return Err(format!("{}: no web front end built there (./build.sh wfe)", w.display()));
            }
        }
        let tele = Telemetry::start(std::slice::from_ref(&gpu.pci));
        let history = Mutex::new(out_dir.as_deref().map(earlier).unwrap_or_default());
        Ok(AudioServer { model, out_dir, wfe, cors, gpu, tele, engine: RwLock::new(engine), slots: Default::default(), progress: Mutex::new(json!({"busy": false})),
                         idle: None, waiting: AtomicUsize::new(0), cancel: Default::default(), history,
                         vram_loaded: AtomicU64::new(own_vram_mb()), retire: AtomicBool::new(false) })
    }

    /// Answers on `addr` until the process ends
    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        if let Some(i) = self.idle {
            let me = self.clone();
            i.watch("nextsycl audio", Box::new(move || {
                me.waiting.load(Ordering::Relaxed) > 0 || me.progress.lock().map(|p| p["busy"] == true).unwrap_or(true)
            }));
        }
        eprintln!("nextsycl audio: {} on http://{addr}/v1/audio/speech{}", self.model,
                  if self.wfe.is_some() { format!(" and http://{addr}/ (the web front end)") } else { String::new() });
        for c in l.incoming().flatten() {
            let me = self.clone();
            std::thread::spawn(move || me.handle(Conn::Tcp(c)));
        }
        Ok(())
    }

    fn handle(&self, mut s: Conn) {
        let req = match http::read_request(&mut s) {
            Ok(r) if r.method == "POST" || r.path == "/" => {
                // work (a request, a page opened) keeps the server; the page's polling does not
                if let Some(i) = self.idle {
                    i.touch();
                }
                r
            }
            Ok(r) => r,
            Err(e) => return http::respond(&mut s, 400, &err(e)),
        };
        let cors = cors_headers(req.origin.as_deref(), &self.cors);
        match (req.method.as_str(), req.path.as_str()) {
            ("OPTIONS", _) => {
                let h = format!("{cors}Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization, Range\r\n");
                http::respond_bytes(&mut s, 204, "text/plain", b"", &h)
            }
            ("GET", "/") | ("GET", "/index.html") if self.wfe.is_some() => self.page(&mut s, &cors),
            ("GET", p) if self.wfe.is_some() && (p.starts_with("/ui/") || p.starts_with("/static/")) => self.asset(&mut s, p, &cors),
            ("GET", "/api/gpu") => http::respond_with(&mut s, 200, &self.gpu_json(), &cors),
            ("GET", "/health") => http::respond_with(&mut s, 200, &json!({"status": "ok"}), &cors),
            ("GET", "/v1/models") | ("GET", "/models") => http::respond_with(&mut s, 200, &json!({"object": "list", "data": [
                {"id": self.model, "object": "model", "owned_by": "nextsycl", "created": 0}]}), &cors),
            ("GET", "/api/info") => http::respond_with(&mut s, 200, &self.info(), &cors),
            ("GET", "/api/progress") => {
                let mut p = self.progress.lock().unwrap().clone();
                p["waiting"] = json!(self.waiting.load(Ordering::Relaxed));
                p["request_id"] = json!(self.cancel.current());
                http::respond_with(&mut s, 200, &p, &cors)
            }
            ("GET", "/api/history") => {
                // files deleted since (system.files.delete) drop out
                let mut h = self.history.lock().unwrap();
                if let Some(d) = &self.out_dir {
                    h.retain(|e| e["url"].as_str().and_then(|u| u.rsplit('/').next()).is_none_or(|f| d.join(f).exists()));
                }
                http::respond_with(&mut s, 200, &json!({"data": h.iter().collect::<Vec<_>>()}), &cors)
            }
            ("POST", "/api/cancel") => {
                // {"id"}: that request only (the front door's job), running or waiting; none: the one running
                let busy = self.progress.lock().unwrap()["busy"] == true;
                let id = serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["id"].as_str().map(str::to_string));
                let running = self.cancel.request(id.as_deref(), busy);
                http::respond_with(&mut s, 200, &json!({"cancelled": running, "id": id}), &cors)
            }
            ("GET", p) if p.starts_with("/v1/audio/files/") => self.file(&mut s, &p["/v1/audio/files/".len()..], &req.head, &cors),
            ("GET", "/api/voices") => http::respond_with(&mut s, 200, &json!({"data": self.voices().map(|l| l.list()).unwrap_or_default()}), &cors),
            ("GET", p) if p.starts_with("/v1/audio/voices/") => {
                let name = p["/v1/audio/voices/".len()..].trim_end_matches(".wav");
                match self.voices().and_then(|l| l.sample_path(name)).and_then(|p| std::fs::read(p).ok()) {
                    Some(b) => http::respond_bytes(&mut s, 200, "audio/wav", &b, &format!("{cors}Cache-Control: no-cache\r\n")),
                    None => http::respond_with(&mut s, 404, &err(format!("no saved voice {name}")), &cors),
                }
            }
            // an enrolment is a front-door job (it asks for a card like a load): the page reaches it through /open/<id>/
            (_, "/api/voices/enroll") => http::respond_with(&mut s, 404,
                &err("enrolling a voice is the front door's: open this page through it (http://<box>:8000/open/<model>/) or run nextsycl audio voice enroll"), &cors),
            ("POST", "/api/voices/add") | ("POST", "/api/voices/remove") => {
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                match self.voice_op(&req.path, &body) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err(m) => http::respond_with(&mut s, 400, &err(m), &cors),
                }
            }
            ("POST", "/v1/audio/transcriptions") | ("POST", "/audio/transcriptions") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON (a multipart form is not read here): {e}")), &cors),
                };
                match self.transcribe(&body) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
                self.maybe_retire();
            }
            ("POST", "/v1/audio/speech") | ("POST", "/audio/speech") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                let plan = match streamed(&body) {
                    Ok(p) => p,
                    Err(m) => return http::respond_with(&mut s, 400, &err(m), &cors),
                };
                match plan {
                    Some(plan) => {
                        let mut out = Out::new(&mut s, plan.how, &cors, plan.speed, plan.rate);
                        // one voice over the parts: a seed they share
                        let mut part = body.clone();
                        if !part["seed"].is_u64() {
                            part["seed"] = json!(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 % 1_000_000).unwrap_or(7));
                        }
                        let mut r = Ok((None, Vec::new()));
                        for (i, text) in plan.parts.iter().enumerate() {
                            // the parts as one stream: a pause between them
                            if i > 0 && out.silence(plan.gap).is_err() {
                                return;
                            }
                            part["input"] = json!(text);
                            r = self.generate(&part, &base, Some(&mut out));
                            if r.is_err() {
                                break;
                            }
                        }
                        out.finish(r);
                    }
                    None => match self.generate(&body, &base, None) {
                        Ok((Some(v), _)) => http::respond_with(&mut s, 200, &v, &cors),
                        Ok((None, wav)) => http::respond_bytes(&mut s, 200, "audio/wav", &wav, &cors),
                        Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                    },
                }
                self.maybe_retire();
            }
            _ => http::respond_with(&mut s, 404, &err(format!("no route {} {}", req.method, req.path)), &cors),
        }
    }

    /// Ends the process after an answer when it should (a GPU error; card memory grown past the loaded model's by
    /// more than RETIRE_GROWTH_MB or half of it) and nothing waits; the front door starts a fresh server
    fn maybe_retire(&self) {
        let (was, now) = (self.vram_loaded.load(Ordering::Relaxed), own_vram_mb());
        if was > 0 && now > was + RETIRE_GROWTH_MB.max(was / 2) {
            eprintln!("nextsycl audio: card memory {now} MiB, {was} MiB once loaded - ending so a fresh server takes the next request");
            self.retire.store(true, Ordering::Relaxed);
        }
        if !self.retire.load(Ordering::Relaxed) {
            return;
        }
        // the engine held while ending: a request that got it after this one's answer is on the card (it retires
        // the server after its own answer), and none starts while the runtime is torn down
        let Ok(_engine) = self.engine.try_write() else { return };
        if self.waiting.load(Ordering::Relaxed) == 0 {
            std::process::exit(0);
        }
    }

    /// The page: index.html with the stylesheet inlined (H3's server does the same)
    fn page(&self, s: &mut Conn, cors: &str) {
        let dir = self.wfe.as_ref().expect("checked by the route").join("audio");
        match (std::fs::read_to_string(dir.join("index.html")), std::fs::read_to_string(dir.join("style.css"))) {
            (Ok(h), Ok(c)) => http::respond_bytes(s, 200, "text/html; charset=utf-8", h.replace("__CSS__", &c).as_bytes(),
                                                  &format!("{cors}Cache-Control: no-cache\r\n")),
            _ => http::respond_with(s, 500, &err(format!("{}: the page is not built (./build.sh wfe)", dir.display())), cors),
        }
    }

    /// A built module or a static file: /ui/<path> from dist/wfe, /static/<f> from dist/wfe/static
    fn asset(&self, s: &mut Conn, path: &str, cors: &str) {
        let root = self.wfe.as_ref().expect("checked by the route");
        let rel = path.strip_prefix("/ui/").unwrap_or(path.trim_start_matches('/'));
        if rel.split('/').any(|c| c == ".." || c.is_empty()) {
            return http::respond_with(s, 404, &err("no such file"), cors);
        }
        match std::fs::read(root.join(rel)) {
            Ok(b) => http::respond_bytes(s, 200, http::content_type(rel), &b, &format!("{cors}Cache-Control: no-cache\r\n")),
            Err(_) => http::respond_with(s, 404, &err(format!("no file {rel}")), cors),
        }
    }

    fn gpu_json(&self) -> Value {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        let used = free.map(|f| total.saturating_sub(f)).unwrap_or(0);
        let r = self.tele.readings().first().copied().unwrap_or_default();
        json!({"name": self.gpu.name, "vram_used_mb": used >> 20, "vram_total_mb": total >> 20,
               "power_w": r.watts, "temp_pkg": r.temp_c, "temp_vram": r.vram_c})
    }

    fn info(&self) -> Value {
        let Ok(e) = self.engine.try_read() else {
            // busy: what does not change
            return json!({"model": self.model, "busy": true});
        };
        let d = e.defaults();
        json!({
            "model": self.model, "arch": e.arch(), "lyrics": e.lyrics(), "wfe": self.wfe.is_some(),
            "recogniser": e.recogniser().map(|r| json!({"languages": r.languages, "words": r.words, "translate": r.translate})),
            "speech": e.speech().map(|sp| json!({"voices": sp.voices, "languages": sp.languages, "instructions": sp.instructions, "design": sp.design,
                                                 "clone": sp.clone, "clone_needs_text": sp.clone_needs_text, "clone_takes_text": sp.clone_takes_text,
                                                 "direction": sp.direction, "events": sp.events, "requires": sp.requires,
                                                 "speakers": sp.speakers.max(1), "enrolled": sp.enrolled})),
            "defaults": {"seconds": f(d.seconds), "max_seconds": f(d.max_seconds), "steps": d.steps, "cfg": f(d.cfg), "rate": d.rate},
            "options": e.options().iter().filter(|o| o.at != nextsycl_core::At::Load)
                .map(|o| json!({"name": o.name, "value": o.value, "help": o.help})).collect::<Vec<_>>(),
            "report": e.report(),
            "saves": self.out_dir.is_some(),
            "streams": e.streams(),
        })
    }

    /// A song made here, whole or the byte range asked for
    fn file(&self, s: &mut Conn, name: &str, head: &[u8], cors: &str) {
        let ok = !name.is_empty() && name.ends_with(".wav") && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && !name.contains("..");
        let b = match (&self.out_dir, ok) {
            (Some(d), true) => std::fs::read(d.join(name)).ok(),
            _ => None,
        };
        let Some(b) = b else { return http::respond_with(s, 404, &err(format!("no song {name}")), cors) };
        let cache = "Cache-Control: max-age=31536000, immutable\r\nAccept-Ranges: bytes\r\n";
        match range(head, b.len()) {
            Some((a, z)) => http::respond_bytes(s, 206, "audio/wav", &b[a..=z], &format!("{cors}{cache}Content-Range: bytes {a}-{z}/{}\r\n", b.len())),
            None => http::respond_bytes(s, 200, "audio/wav", &b, &format!("{cors}{cache}")),
        }
    }

    /// The saved voices (beside the output directory; none without one)
    fn voices(&self) -> Option<crate::voices::Library> {
        self.out_dir.as_deref().map(crate::voices::Library::of)
    }

    /// Save a voice - a recording sent (base64 / a data: URL, any format) or a file made here - or remove one
    fn voice_op(&self, path: &str, b: &Value) -> Result<Value, String> {
        let lib = self.voices().ok_or("this server keeps no files (start it with an output directory)")?;
        let name = b["name"].as_str().unwrap_or("");
        if path.ends_with("/remove") {
            lib.remove(name)?;
            return Ok(json!({"removed": name}));
        }
        let s = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty()).map(str::to_string);
        let (bytes, from, made) = match (s("sample"), s("file")) {
            (Some(a), _) => (http::unbase64(&a).map_err(|e| format!("sample: {e}"))?, "a recording", Value::Null),
            (None, Some(f)) => {
                // a file made here: what it says, the voice's description (a designed one's instructions), its language
                let f = f.rsplit('/').next().unwrap_or(&f).to_string();
                let dir = self.out_dir.as_ref().expect("checked above");
                if f.contains("..") || !f.ends_with(".wav") {
                    return Err(format!("file {f}: a WAV made here"));
                }
                let p = dir.join(&f);
                (std::fs::read(&p).map_err(|e| format!("{f}: {e}"))?, "made here", entry(&p, 0).unwrap_or_default())
            }
            _ => return Err("sample (a recording, base64 or a data: URL) or file (a WAV made here)".into()),
        };
        let m = |k: &str| made[k].as_str().filter(|x| !x.trim().is_empty()).map(str::to_string);
        let text = s("text").or_else(|| m("prompt"));
        let description = s("description").or_else(|| m("instructions").filter(|_| made["cloned"] == false && made["voice"].is_null()));
        let language = s("language").or_else(|| m("language"));
        let kind = if description.is_some() && from == "made here" { "design" } else { "clone" };
        let from = match m("model") {
            Some(model) => format!("{from} ({model})"),
            None => from.to_string(),
        };
        lib.add(name, &bytes, text.as_deref(), description.as_deref(), language.as_deref(), kind, &from, b["replace"] == true)
    }

    /// A recording written down: the engine's transcript as JSON
    fn transcribe(&self, b: &Value) -> Result<Value, (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model && !m.starts_with("openai/whisper") && *m != "whisper-1") {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let a = b["audio"].as_str().or_else(|| b["file"].as_str()).filter(|x| !x.trim().is_empty())
            .ok_or_else(|| bad("audio (the recording: base64 or a data URL): required".into()))?;
        let bytes = http::unbase64(a).map_err(|e| bad(format!("audio: {e}")))?;
        let (samples, rate) = nextsycl_audio::decode_audio(&bytes).map_err(|e| bad(format!("audio: {e}")))?;
        let text = |k: &str| b[k].as_str().filter(|x| !x.trim().is_empty()).map(str::to_string);
        // OpenAI's timestamp_granularities ["word"] asks for the words too
        let words = b["words"].as_bool().unwrap_or(true)
            || b["timestamp_granularities"].as_array().is_some_and(|g| g.iter().any(|x| x == "word"));
        let given: nextsycl_core::options::Given = b["options"].as_object().into_iter().flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let engine = self.engine.read().unwrap_or_else(|p| p.into_inner());
        // one recording at a time, as the engine's batch says (a recogniser's is 1)
        let _slot = Slot::take(&self.slots, engine.batch());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        if engine.recogniser().is_none() {
            return Err(bad(format!("{} makes sound; it does not transcribe (audio.transcribe needs a recogniser: whisper-large-v3-turbo)", self.model)));
        }
        let mut req = nextsycl_audio::TranscribeRequest { samples, rate, language: text("language"), translate: b["translate"] == true || b["task"] == "translate",
                                                          words, prompt: text("prompt"), extra: Default::default() };
        if !given.is_empty() {
            nextsycl_core::options::resolve(&given, engine.options(), nextsycl_core::At::Request, &self.model).map_err(bad)?;
            req.extra = nextsycl_core::options::by_name(&given, engine.options());
        }
        let secs = req.samples.len() as f64 / req.rate.max(1) as f64;
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        let p = &self.progress;
        let what = format!("{secs:.1} s of recording");
        *p.lock().unwrap() = json!({"busy": true, "prompt": what, "phase": "listen", "at": 0, "of": 1, "seconds": 0});
        let out = engine.transcribe(&req, &mut |st| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": what, "phase": st.phase, "at": st.at, "of": st.of, "seconds": st.seconds});
            if self.cancel.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        });
        *p.lock().unwrap() = json!({"busy": false});
        drop(engine);
        if let Err(e) = &out {
            if gpu_failed(&e.0) {
                eprintln!("nextsycl audio: a GPU error ({}) - ending after this answer", e.0);
                self.retire.store(true, Ordering::Relaxed);
            }
        }
        let t = out.map_err(|e| (if e.0 == "cancelled" { 499 } else { 400 }, e.0))?;
        let took = t0.elapsed().as_secs_f64();
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let r = |x: f64| (x * 100.0).round() / 100.0;
        let timed = |v: &[nextsycl_audio::Timed]| -> Vec<Value> {
            v.iter().map(|x| {
                let mut o = json!({"start": r(x.start), "end": r(x.end), "text": x.text});
                if let Some(pr) = x.probability {
                    o["probability"] = json!((pr as f64 * 1000.0).round() / 1000.0);
                }
                o
            }).collect()
        };
        Ok(json!({"model": self.model, "text": t.text, "language": t.language, "duration": r(t.duration), "segments": timed(&t.segments), "words": timed(&t.words),
                  "created": now(), "nextsycl": {"took": took, "wh": wh, "seconds": r(t.duration)}}))
    }

    /// Runs a request: (JSON for response_format url or a streamed answer's end, else none) and the WAV's bytes;
    /// `stream`: the sound sent there as it is made
    fn generate(&self, b: &Value, base: &str, stream: Option<&mut Out>) -> Result<(Option<Value>, Vec<u8>), (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model && *m != "MiniMaxAI/MiniMax-Music3" && !m.starts_with("Qwen/Qwen3-TTS")) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let text = |k: &[&str]| k.iter().find_map(|k| b[*k].as_str()).filter(|p| !p.trim().is_empty()).map(str::to_string);
        let format = b["response_format"].as_str().unwrap_or("wav");
        if format != "url" && crate::speechfmt::Fmt::parse(format).is_none() {
            return Err(bad(format!("response_format {format}: wav, pcm, mp3, opus, aac, flac or url")));
        }
        if format == "url" && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        // a streamed answer's last event links the file (sse)
        let link = format == "url" || stream.as_ref().is_some_and(|o| o.how == Streamed::Sse);
        let num = |k: &[&str]| k.iter().find_map(|k| b[*k].as_f64());
        let seed = match &b["seed"] {
            Value::Number(x) => x.as_u64().ok_or_else(|| bad("seed: a whole number".into()))?,
            _ => SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 % 1_000_000).unwrap_or(7),
        };
        let given: nextsycl_core::options::Given = b["options"].as_object().into_iter().flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
        // a recording to clone: base64 WAV or a data URL
        let reference = match text(&["ref_audio", "reference_audio"]) {
            Some(a) => {
                let wav = http::unbase64(&a).map_err(|e| bad(format!("ref_audio: {e}")))?;
                let (samples, rate) = nextsycl_audio::decode_wav(&wav).map_err(|e| bad(format!("ref_audio: {e}")))?;
                Some(nextsycl_audio::Reference { samples, rate, text: text(&["ref_text", "reference_text"]) })
            }
            None => None,
        };
        // a song made from a recording (a cover, a repaint, an extension): base64 (WAV, MP3, FLAC, Ogg, M4A) or a data
        // URL, its channels kept
        let source = match text(&["audio", "src_audio", "source_audio"]) {
            Some(a) => {
                let bytes = http::unbase64(&a).map_err(|e| bad(format!("audio: {e}")))?;
                Some(nextsycl_audio::read_audio(&bytes).map_err(|e| bad(format!("audio: {e}")))?)
            }
            None => None,
        };
        // one request at a time, or as many as the engine makes together
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let engine = self.engine.read().unwrap_or_else(|p| p.into_inner());
        let _slot = Slot::take(&self.slots, engine.batch());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let Some(running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        if source.is_some() && !engine.sources() {
            return Err(bad(format!("audio: {} makes no song from a recording (acestep-1.5 does: task cover, repaint, edit)", self.model)));
        }
        // a song's request option may come as a field of its own (bpm, key, time_signature, task, repaint_start ...):
        // the engine's option of that name, underscores for its dashes (`options` says it first)
        let mut given = given;
        if engine.speech().is_none() {
            for o in engine.options().iter().filter(|o| o.at != nextsycl_core::At::Load) {
                let v = &b[o.name.replace('-', "_").as_str()];
                if !v.is_null() && !given.contains_key(o.name) {
                    given.insert(o.name.to_string(), v.as_str().map_or_else(|| v.to_string(), str::to_string));
                }
            }
        }
        let speech = engine.speech().is_some();
        // speech: input is the text to say, instructions its style or voice; a song: input the lyrics, instructions
        // the music
        let (prompt, lyrics, instructions) = if speech {
            (text(&["input", "text"]).ok_or_else(|| bad("input (the text to say): required".into()))?, None, text(&["instructions"]))
        } else {
            // from a recording the description may be left out (the engine keeps the recording's sound)
            let p = match text(&["instructions", "prompt", "description"]) {
                Some(p) => p,
                None if source.is_some() => String::new(),
                None => return Err(bad("instructions (the music's description): required".into())),
            };
            (p, text(&["input", "lyrics"]), None)
        };
        // speech: the text as it is said (symbols as words, no markdown) unless normalize: false / NS_SPEECH_NORMALIZE=0
        let prompt = if speech && b["normalize"] != false && !std::env::var("NS_SPEECH_NORMALIZE").is_ok_and(|v| v == "0") { speakable(&prompt) } else { prompt };
        let frames_per_s = if speech { 12.5 } else { 25.0 };
        let seconds = num(&["seconds", "duration", "audio_duration"]).or_else(|| num(&["max_new_tokens"]).map(|f| f / frames_per_s)).map(|v| v as f32);
        let mut saved: Option<String> = None;
        // a saved voice's enrolment (its fitted codes' file), for an engine that speaks enrolled voices
        let mut enrolled: Option<String> = None;
        let mut req = AudioRequest {
            prompt: prompt.clone(),
            lyrics: lyrics.clone(),
            seconds,
            steps: num(&["steps", "num_inference_steps"]).map(|v| v as u32),
            cfg: num(&["cfg", "guidance_scale"]).map(|v| v as f32),
            seed,
            extra: Default::default(),
            voice: text(&["voice", "speaker"]),
            language: text(&["language"]),
            instructions: instructions.clone(),
            reference,
            references: Vec::new(),
            source,
        };
        let carried = text(&["continue_from"]);
        // carrying on: a voice that is not a saved one (a built-in voice of the chunk before) is in that chunk already
        if carried.is_some() && req.voice.as_ref().is_some_and(|v| !self.voices().is_some_and(|l| l.get(v).is_some())) {
            req.voice = None;
        }
        // a saved voice: its recording and transcript as the reference, for a model that clones
        if let (Some(sp), Some(v)) = (engine.speech(), req.voice.clone()) {
            if !sp.voices.iter().any(|x| x.eq_ignore_ascii_case(&v)) {
                let lib = self.voices().ok_or_else(|| bad(format!("voice {v}: not a built-in one, and this server keeps no saved voices")))?;
                if lib.get(&v).is_none() {
                    let mut known = sp.voices.clone();
                    known.extend(lib.list().iter().filter_map(|x| x["name"].as_str().map(str::to_string)));
                    return Err(bad(format!("voice {v}: none such ({})", known.join(", "))));
                }
                // a model without an encoder speaks a saved voice it was enrolled for: its fitted codes, not the recording
                let enrolled_file = if sp.clone { None } else { lib.enrolled_path(&v, engine.arch()) };
                if !sp.clone && !(sp.enrolled && enrolled_file.is_some()) {
                    return Err(bad(match sp.enrolled {
                        true => format!("voice {v} is a saved voice not enrolled for {}: nextsycl audio voice enroll {v} (experimental), or a model that \
                                         clones (qwen3-tts-base)", engine.arch()),
                        false => format!("voice {v} is a saved voice: speak it with a model that clones (qwen3-tts-base)"),
                    }));
                }
                if let Some(f) = enrolled_file {
                    enrolled = Some(f.to_string_lossy().into_owned());
                } else if req.reference.is_none() {
                    req.reference = Some(lib.reference(&v).map_err(bad)?);
                }
                req.voice = None;
                saved = Some(v);
            }
        }
        // a multi-speaker script's voices, a speaker each in order: "voices" (saved voices' names) or "ref_audios"
        // (recordings, base64 WAV); the first is the reference, the others `references`
        let names: Vec<String> = b["voices"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect();
        let recs: Vec<String> = b["ref_audios"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect();
        if !names.is_empty() || !recs.is_empty() {
            let sp = engine.speech().ok_or_else(|| bad("voices: speech's (this is a song model)".into()))?;
            if !names.is_empty() && !recs.is_empty() {
                return Err(bad("voices or ref_audios: one list of the speakers' voices, not both".into()));
            }
            if req.reference.is_some() || saved.is_some() || carried.is_some() {
                return Err(bad("voices / ref_audios name every speaker's voice: not with voice, ref_audio or continue_from".into()));
            }
            let n = names.len().max(recs.len());
            if n > sp.speakers.max(1) {
                return Err(bad(format!("{n} speakers' voices: {} speaks {} at most", self.model, sp.speakers.max(1))));
            }
            let mut refs = Vec::new();
            for v in &names {
                let lib = self.voices().ok_or_else(|| bad(format!("voices: {v} - this server keeps no saved voices")))?;
                if lib.get(v).is_none() {
                    return Err(bad(format!("voices: {v} - no saved voice of that name")));
                }
                refs.push(lib.reference(v).map_err(bad)?);
            }
            for (i, a) in recs.iter().enumerate() {
                let wav = http::unbase64(a).map_err(|e| bad(format!("ref_audios[{i}]: {e}")))?;
                let (samples, rate) = nextsycl_audio::decode_wav(&wav).map_err(|e| bad(format!("ref_audios[{i}]: {e}")))?;
                refs.push(nextsycl_audio::Reference { samples, rate, text: None });
            }
            req.reference = Some(refs.remove(0));
            req.references = refs;
            req.voice = None;
            saved = names.first().cloned();
        }
        // carry on from speech made here: the end of that file (its last sentences) after the voice's reference
        if let Some(name) = &carried {
            let sp = engine.speech().ok_or_else(|| bad("continue_from: speech's (this is a song model)".into()))?;
            if !sp.clone {
                return Err(bad("continue_from: speak it with a model that clones (qwen3-tts-base, voxcpm2)".into()));
            }
            let name = name.rsplit('/').next().unwrap_or(name);
            let ok = name.ends_with(".wav") && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
            let path = self.out_dir.as_ref().filter(|_| ok).map(|d| d.join(name)).filter(|p| p.is_file())
                .ok_or_else(|| (404u16, format!("continue_from {name}: no speech file of that name here")))?;
            let (info, _) = nextsycl_audio::wav_info(&path).map_err(bad)?;
            let said = info.iter().find(|(k, _)| k == "INAM").map(|(_, v)| v.clone()).filter(|t| !t.trim().is_empty())
                .ok_or_else(|| bad(format!("continue_from {name}: the file does not say what it says (not speech made here)")))?;
            let bytes = std::fs::read(&path).map_err(|e| (500u16, format!("{name}: {e}")))?;
            let (samples, rate) = nextsycl_audio::decode_wav(&bytes).map_err(bad)?;
            let (end, end_text) = crate::carry::tail(&samples, rate, &said, crate::carry::KEEP_S);
            let before = nextsycl_audio::Reference { samples: end, rate, text: Some(end_text) };
            req.reference = Some(match req.reference.take() {
                Some(voice) if voice.text.is_some() => crate::carry::join(voice, before),
                _ => before,
            });
            req.voice = None;
        }
        if !given.is_empty() {
            // a file path the server reads: only a saved voice's enrolment names it (a request names the voice)
            if given.keys().any(|k| k == "enrolled" || k.ends_with("_ENROLLED")) {
                return Err(bad("options.enrolled: not a request's (name the enrolled saved voice in voice)".into()));
            }
            nextsycl_core::options::resolve(&given, engine.options(), nextsycl_core::At::Request, &self.model).map_err(bad)?;
            req.extra = nextsycl_core::options::by_name(&given, engine.options());
        }
        if let Some(f) = enrolled {
            req.extra.insert("enrolled".into(), f);
        }
        let d = engine.defaults();
        let (steps, cfg) = (req.steps.unwrap_or(d.steps), req.cfg.unwrap_or(d.cfg));
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        let p = &self.progress;
        let want = req.seconds.unwrap_or(d.seconds);
        *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "phase": "prompt", "at": 0, "of": 1, "seconds": 0, "want": want});
        let mut report = |st: nextsycl_audio::Step| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "phase": st.phase, "at": st.at, "of": st.of, "seconds": st.seconds, "want": want,
                                        "running": self.cancel.running()});
            if running.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        };
        let out = match stream {
            Some(o) => {
                o.rate = d.rate;
                engine.generate_streamed(&req, &mut report, &mut |x| o.send(x).map_err(|e| nextsycl_core::Error(format!("the client is gone: {e}"))))
            }
            None => engine.generate(&req, &mut report),
        };
        // the last of those running: idle
        if self.cancel.running() <= 1 {
            *p.lock().unwrap() = json!({"busy": false});
        }
        drop(running);
        drop(_slot);
        drop(engine);
        if let Err(e) = &out {
            // the device failed (out of resources, lost): this process's GPU state is not to be trusted again
            if gpu_failed(&e.0) {
                eprintln!("nextsycl audio: a GPU error ({}) - ending after this answer", e.0);
                self.retire.store(true, Ordering::Relaxed);
            }
        }
        let audio = out.map_err(|e| (if e.0 == "cancelled" { 499 } else { 400 }, e.0))?;
        let took = t0.elapsed().as_secs_f64();
        // the card's energy over the request (its counter's rise), watt-hours
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let created = now();
        let meta = json!({"model": self.model, "seed": seed, "steps": steps, "cfg": f(cfg), "took": took, "wh": wh, "created": created}).to_string();
        let mut meta: Value = serde_json::from_str(&meta).unwrap_or_default();
        if speech {
            meta["voice"] = json!(req.voice.clone().or(saved));
            meta["language"] = json!(req.language);
            meta["instructions"] = json!(instructions);
            // a saved voice spoken by its enrolment (no recording in the prompt: its fitted codes) is a clone too
            meta["cloned"] = json!(req.reference.is_some() || req.extra.contains_key("enrolled"));
            if req.extra.contains_key("enrolled") {
                meta["enrolled"] = json!(true);
            }
            meta["continued"] = json!(carried);
        }
        let meta = meta.to_string();
        let wav = audio.wav(&[("INAM", &prompt), ("ILYR", lyrics.as_deref().unwrap_or("")), ("ICMT", &meta)]);
        let mut answer = None;
        if let Some(dir) = &self.out_dir {
            let name = format!("{created}-{}-{seed}.wav", self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"));
            let path = dir.join(&name);
            std::fs::write(&path, &wav).map_err(|e| (500u16, format!("{}: {e}", dir.display())))?;
            if let Some(e) = entry(&path, created) {
                let mut h = self.history.lock().unwrap();
                h.push_front(e.clone());
                h.truncate(200);
                if link {
                    let mut v = e;
                    v["url"] = json!(format!("{base}/v1/audio/files/{name}"));
                    answer = Some(json!({"created": created, "data": [v], "nextsycl": {"took": took, "seconds": audio.seconds(), "seed": seed, "steps": steps,
                                                                                     "cfg": f(cfg), "wh": wh}}));
                }
            }
        }
        Ok((answer, wav))
    }
}

/// A place on the engine: taken when fewer than its batch run (else waited for), given back when dropped
struct Slot<'a>(&'a (Mutex<usize>, Condvar));

impl<'a> Slot<'a> {
    fn take(slots: &'a (Mutex<usize>, Condvar), batch: usize) -> Slot<'a> {
        let mut n = slots.0.lock().unwrap_or_else(|p| p.into_inner());
        while *n >= batch.max(1) {
            n = slots.1.wait(n).unwrap_or_else(|p| p.into_inner());
        }
        *n += 1;
        Slot(slots)
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0 .0.lock().unwrap_or_else(|p| p.into_inner()) -= 1;
        self.0 .1.notify_one();
    }
}

/// How a streamed answer is sent
#[derive(Clone, Copy, Debug, PartialEq)]
enum Streamed {
    /// the sound as it is made, in a format: wav (its lengths unknown), pcm (16-bit mono, 24 kHz as OpenAI's), or
    /// mp3, opus, aac, flac (ffmpeg encodes as it comes); sent chunked (`Transfer-Encoding: chunked`)
    Audio(Fmt),
    /// server-sent events (OpenAI's stream_format sse): speech.audio.delta {audio: base64 16-bit PCM} as the sound
    /// comes, then speech.audio.done (with the file's link and our numbers); an error ends it as an error event
    Sse,
}

/// A streamed request: how, the texts to say in turn (one stream: `input` as a list), the pause between them, the
/// speed, the rate asked for
struct Plan {
    how: Streamed,
    parts: Vec<String>,
    gap: f32,
    speed: f64,
    rate: Option<u32>,
}

/// Whether the request wants its sound streamed, and how. OpenAI answers a speech request as a chunked stream of the
/// format's bytes; here that is every format but wav (whole unless `stream: true` / `stream_format: "audio"`) and
/// url. `stream_format: "sse"` sends events. `input` may be a list of texts: they are said in turn as one stream
/// (`gap` seconds between; a list asks for a stream). `speed` (0.25 - 4) and `sample_rate` need ffmpeg, as mp3, opus,
/// aac and flac do
fn streamed(b: &Value) -> Result<Option<Plan>, String> {
    let format = b["response_format"].as_str().unwrap_or("wav");
    let sf = b["stream_format"].as_str();
    let speed = crate::speechfmt::speed_of(b)?;
    let parts: Vec<String> = match &b["input"] {
        Value::Array(a) => {
            let v: Vec<String> = a.iter().filter_map(|x| x.as_str()).map(str::to_string).filter(|x| !x.trim().is_empty()).collect();
            if v.is_empty() || v.len() != a.len() {
                return Err("input: a text, or a list of texts (each not empty)".into());
            }
            v
        }
        _ => Vec::new(),
    };
    let gap = match &b["gap"] {
        Value::Null => 0.3,
        v => v.as_f64().filter(|g| (0.0..=5.0).contains(g)).ok_or("gap: seconds from 0 to 5")? as f32,
    };
    let rate = match &b["sample_rate"] {
        Value::Null => None,
        v => Some(v.as_u64().filter(|r| (8000..=96000).contains(r)).ok_or("sample_rate: Hz from 8000 to 96000")? as u32),
    };
    let fmt = Fmt::parse(format);
    if !matches!(sf, None | Some("audio") | Some("sse")) {
        return Err(format!("stream_format {}: audio or sse", sf.unwrap_or("")));
    }
    let list = !parts.is_empty();
    let always = matches!(fmt, Some(Fmt::Pcm | Fmt::Mp3 | Fmt::Opus | Fmt::Aac | Fmt::Flac));
    let asked = b["stream"] == true || sf.is_some();
    if !(asked || always || (fmt == Some(Fmt::Wav) && (speed != 1.0 || list || rate.is_some()))) {
        if format == "url" && (speed != 1.0 || list) {
            return Err("speed and a list of texts: a stream (wav, pcm, mp3, opus, aac, flac), not response_format url".into());
        }
        return Ok(None);
    }
    let parts = if list { parts } else { vec![b["input"].as_str().unwrap_or("").to_string()] };
    let how = if sf == Some("sse") {
        if speed != 1.0 || rate.is_some() {
            return Err("speed and sample_rate: a stream of audio, not events".into());
        }
        Streamed::Sse
    } else {
        match fmt {
            Some(f) => Streamed::Audio(f),
            None => return Err("response_format url: a streamed answer is audio (wav, pcm, mp3, opus, aac, flac) - stream_format sse's last event links the file".into()),
        }
    };
    if let Streamed::Audio(f) = how {
        if (!matches!(f, Fmt::Wav | Fmt::Pcm) || speed != 1.0) && !crate::speechfmt::have_ffmpeg() {
            return Err(format!("response_format {format}{}: ffmpeg is not on this server's path", if speed != 1.0 { " with speed" } else { "" }));
        }
    }
    Ok(Some(Plan { how, parts, gap, speed, rate }))
}

/// A streamed answer on its connection: the head sent with the first sound (an error before it is answered as
/// usual), each piece of sound written as it comes as a chunk
struct Out<'a> {
    s: &'a mut Conn,
    how: Streamed,
    cors: &'a str,
    started: bool,
    /// the engine's rate
    rate: u32,
    speed: f64,
    /// the rate asked for (none: pcm's 24 kHz, else the engine's)
    want: Option<u32>,
    /// ffmpeg between the engine and the client (another format, speed or rate)
    pipe: Option<crate::speechfmt::Pipe>,
}

impl<'a> Out<'a> {
    fn new(s: &'a mut Conn, how: Streamed, cors: &'a str, speed: f64, want: Option<u32>) -> Out<'a> {
        Out { s, how, cors, started: false, rate: 0, speed, want, pipe: None }
    }

    /// The rate of the sound that reaches the client
    fn out_rate(&self) -> u32 {
        match self.how {
            Streamed::Audio(Fmt::Pcm) => self.want.unwrap_or(PCM_RATE),
            _ => self.want.unwrap_or(self.rate),
        }
    }

    fn start(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        if self.started {
            return Ok(());
        }
        self.started = true;
        let kind = match self.how {
            Streamed::Audio(f) => f.content_type(),
            Streamed::Sse => "text/event-stream",
        };
        if let Streamed::Audio(f) = self.how {
            let to = Some(self.out_rate()).filter(|r| *r != self.rate);
            if !matches!(f, Fmt::Wav | Fmt::Pcm) || self.speed != 1.0 || to.is_some() {
                match crate::speechfmt::Pipe::start(self.rate, f, to, self.speed) {
                    Ok(p) => self.pipe = Some(p),
                    // no ffmpeg: wav and pcm go out at the engine's rate (X-Sample-Rate says which)
                    Err(e) if matches!(f, Fmt::Wav | Fmt::Pcm) && self.speed == 1.0 => {
                        eprintln!("nextsycl audio: {e} - sending the engine's rate");
                        self.want = None;
                    }
                    Err(e) => return Err(std::io::Error::other(e)),
                }
            }
        }
        write!(self.s, "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nX-Sample-Rate: {}\r\n{}Connection: close\r\n\r\n",
               self.out_rate(), self.cors)?;
        if self.how == Streamed::Audio(Fmt::Wav) && self.pipe.is_none() {
            speechfmt::chunk(&mut *self.s, &nextsycl_audio::wav_stream_head(self.rate, 1))?;
        }
        self.s.flush()
    }

    fn send(&mut self, samples: &[f32]) -> std::io::Result<()> {
        use std::io::Write;
        self.start()?;
        let pcm = nextsycl_audio::pcm16(samples);
        match (self.how, self.pipe.as_mut()) {
            (Streamed::Sse, _) => speechfmt::chunk(&mut *self.s, format!("data: {}\n\n", json!({"type": "speech.audio.delta", "audio": http::base64(&pcm)})).as_bytes())?,
            (_, Some(p)) => {
                p.feed(&pcm)?;
                for c in p.ready() {
                    speechfmt::chunk(&mut *self.s, &c)?;
                }
            }
            _ => speechfmt::chunk(&mut *self.s, &pcm)?,
        }
        self.s.flush()
    }

    /// A pause between the parts of one stream
    fn silence(&mut self, seconds: f32) -> std::io::Result<()> {
        let n = (seconds * self.rate as f32) as usize;
        if n == 0 || !self.started {
            return Ok(());
        }
        self.send(&vec![0.0; n])
    }

    /// The end: an error not answered yet as usual, the events' last one (done or the error). An error after the
    /// head leaves the chunks unterminated, so the client knows the sound is cut short
    fn finish(mut self, r: Result<(Option<Value>, Vec<u8>), (u16, String)>) {
        use std::io::Write;
        match (r, self.started) {
            (Err((code, m)), false) => http::respond_with(&mut *self.s, code, &err(m), self.cors),
            (Err((_, m)), true) => {
                if self.how == Streamed::Sse {
                    let _ = speechfmt::chunk(&mut *self.s, format!("data: {}\n\n", json!({"type": "error", "error": {"message": m}})).as_bytes());
                    let _ = speechfmt::chunk_end(&mut *self.s);
                }
                let _ = self.s.flush();
            }
            (Ok((answer, _)), _) => {
                // nothing handed out (an empty sound): the head, then the end
                if self.start().is_err() {
                    return;
                }
                if let Some(p) = self.pipe.take() {
                    for c in p.finish() {
                        if speechfmt::chunk(&mut *self.s, &c).is_err() {
                            return;
                        }
                    }
                }
                if self.how == Streamed::Sse {
                    let mut done = json!({"type": "speech.audio.done", "sample_rate": self.rate});
                    if let Some(a) = answer {
                        done["data"] = a["data"].clone();
                        done["nextsycl"] = a["nextsycl"].clone();
                    }
                    let _ = speechfmt::chunk(&mut *self.s, format!("data: {done}\n\n").as_bytes());
                }
                let _ = speechfmt::chunk_end(&mut *self.s);
                let _ = self.s.flush();
            }
        }
    }
}

/// How far a server's card memory may grow past the loaded model's before it ends (MiB; or half the model's)
const RETIRE_GROWTH_MB: u64 = 4096;

/// Whether an engine error is the device failing (not a bad request)
fn gpu_failed(m: &str) -> bool {
    ["UR_RESULT_ERROR", "level_zero backend failed", "DEVICE_LOST", "out of device memory"].iter().any(|k| m.contains(k))
}

/// This process's memory on the cards (MiB): its DRM clients' resident VRAM (/proc/self/fdinfo); 0 when unreadable
pub fn own_vram_mb() -> u64 {
    let Ok(fds) = std::fs::read_dir("/proc/self/fd") else { return 0 };
    let mut seen = std::collections::HashSet::new();
    let mut kb = 0;
    for fd in fds.flatten() {
        if !std::fs::read_link(fd.path()).is_ok_and(|t| t.starts_with("/dev/dri/")) {
            continue;
        }
        let Ok(info) = std::fs::read_to_string(std::path::Path::new("/proc/self/fdinfo").join(fd.file_name())) else { continue };
        let field = |k: &str| info.lines().find_map(|l| l.strip_prefix(k)).map(|v| v.trim().to_string());
        // one client can be open on several descriptors: count it once
        if field("drm-client-id:").is_some_and(|id| seen.insert(id)) {
            kb += field("drm-resident-vram0:").and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok()).unwrap_or(0);
        }
    }
    kb / 1024
}

/// A text as it is to be said: math and code symbols as words (a model reads "Q·K^T / √d_k" as noise - a voice
/// looped on one such line), markdown's marks out, dashes and brackets as pauses
pub fn speakable(text: &str) -> String {
    // a script of "Speaker N:" lines (VibeVoice): each line its own, the breaks kept - joined, the speakers'
    // lines read as one (the model said "Speaker" aloud)
    let speaker_line = |l: &str| {
        let l = l.trim_start().to_ascii_lowercase();
        l.strip_prefix("speaker").is_some_and(|r| r.starts_with(char::is_whitespace) && r.trim_start().starts_with(|c: char| c.is_ascii_digit()))
    };
    if text.lines().any(speaker_line) {
        return text.lines().map(speakable_line).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("\n");
    }
    speakable_line(text)
}

fn speakable_line(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let word = |ch: char| ch.is_alphanumeric();
    let mut out = String::with_capacity(text.len() + 16);
    for (i, &ch) in c.iter().enumerate() {
        let prev = if i > 0 { c[i - 1] } else { ' ' };
        let next = c.get(i + 1).copied().unwrap_or(' ');
        let say: Option<&str> = match ch {
            '*' | '#' | '`' => Some(" "),
            '√' => Some(" square root of "),
            '·' | '⋅' | '∙' => Some(" dot "),
            '×' => Some(" times "),
            '÷' => Some(" divided by "),
            '=' => Some(" equals "),
            '≈' => Some(" approximately "),
            '≠' => Some(" not equal to "),
            '≤' => Some(" less than or equal to "),
            '≥' => Some(" greater than or equal to "),
            '<' if prev == ' ' || next == ' ' => Some(" less than "),
            '>' if prev == ' ' || next == ' ' => Some(" greater than "),
            '+' => Some(" plus "),
            '±' => Some(" plus or minus "),
            '∑' => Some(" the sum of "),
            '∏' => Some(" the product of "),
            '∞' => Some(" infinity "),
            '→' | '⟶' => Some(" to "),
            '⇒' => Some(" implies "),
            '∈' => Some(" in "),
            '∀' => Some(" for all "),
            '∂' => Some(" partial "),
            '%' => Some(" percent "),
            '&' => Some(" and "),
            '^' => Some(" to the "),
            '°' => Some(" degrees "),
            'π' => Some(" pi "),
            'θ' => Some(" theta "),
            'α' => Some(" alpha "),
            'β' => Some(" beta "),
            'γ' => Some(" gamma "),
            'Δ' | 'δ' => Some(" delta "),
            'ε' => Some(" epsilon "),
            'λ' => Some(" lambda "),
            'μ' => Some(" mu "),
            'σ' => Some(" sigma "),
            'ω' => Some(" omega "),
            // d_k: "d sub k"; snake_case and the rest: a space
            '_' if word(prev) && word(next) => Some(" sub "),
            '_' | '~' | '|' => Some(" "),
            // a / b spaced: divided; a/b between short terms: over; TCP/IP: slash
            '/' if prev == ' ' && next == ' ' => Some(" divided by "),
            '/' => Some(" slash "),
            '(' | ')' | '[' | ']' | '{' | '}' => Some(", "),
            '—' | '–' => Some(", "),
            '-' if prev == ' ' && next == ' ' => Some(", "),
            _ => None,
        };
        match say {
            Some(s) => out.push_str(s),
            None => out.push(ch),
        }
    }
    // spaces and pauses tidied: one space, no space before a comma or a stop, no pause twice
    let mut t = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for (a, b) in [(" ,", ","), (" .", "."), (",,", ","), (", ,", ","), (". ,", "."), (",.", "."), (" :", ":"), (" ;", ";"), (" ?", "?"), (" !", "!")] {
        while t.contains(a) {
            t = t.replace(a, b);
        }
    }
    t.trim_matches(|ch: char| ch == ',' || ch == ' ').to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn speakable_says_symbols() {
        let s = super::speakable;
        assert_eq!(s("Computes attention: Q·K^T, scaled by √d_k."), "Computes attention: Q dot K to the T, scaled by square root of d sub k.");
        assert_eq!(s("Attention(Q, K, V) = softmax(QK^T / √d_k) V"), "Attention, Q, K, V, equals softmax, QK to the T divided by square root of d sub k, V");
        assert_eq!(s("**Bold** and `code` - done"), "Bold and code, done");
        assert_eq!(s("TCP/IP, 3-4 heads, snake_case"), "TCP slash IP, 3-4 heads, snake sub case");
        assert_eq!(s("Plain words stay plain."), "Plain words stay plain.");
        // a speakers' script keeps its lines
        assert_eq!(s("Speaker 1: a = b\n\nSpeaker 2: *yes*"), "Speaker 1: a equals b\nSpeaker 2: yes");
        assert_eq!(s("one\ntwo"), "one two", "plain text is one line, as before");
    }

    use super::{gpu_failed, own_vram_mb, range};

    #[test]
    fn a_speech_request_streams_as_openai_does() {
        use super::{streamed, Fmt, Streamed};
        use serde_json::json;
        let how = |b: serde_json::Value| streamed(&b).map(|p| p.map(|p| (p.how, p.parts.len())));
        // wav and url stay whole unless asked
        assert_eq!(how(json!({"input": "a"})), Ok(None));
        assert_eq!(how(json!({"input": "a", "response_format": "url"})), Ok(None));
        assert_eq!(how(json!({"input": "a", "stream": true})), Ok(Some((Streamed::Audio(Fmt::Wav), 1))));
        // pcm and the encoded formats are a chunked stream by default
        assert_eq!(how(json!({"input": "a", "response_format": "pcm"})), Ok(Some((Streamed::Audio(Fmt::Pcm), 1))));
        assert_eq!(how(json!({"input": "a", "response_format": "pcm", "stream": false})), Ok(Some((Streamed::Audio(Fmt::Pcm), 1))));
        assert_eq!(how(json!({"input": "a", "response_format": "flac"})), Ok(Some((Streamed::Audio(Fmt::Flac), 1))));
        assert_eq!(how(json!({"input": "a", "stream_format": "sse"})), Ok(Some((Streamed::Sse, 1))));
        // a list of texts is one stream, even as wav
        assert_eq!(how(json!({"input": ["a", "b", "c"]})), Ok(Some((Streamed::Audio(Fmt::Wav), 3))));
        assert!(how(json!({"input": ["a", ""]})).is_err());
        assert!(how(json!({"input": [], "stream": true})).is_err());
        assert!(how(json!({"input": ["a", "b"], "response_format": "url"})).is_err());
        // speed and the rate: OpenAI's range; not with events
        assert!(how(json!({"input": "a", "speed": 0.1})).is_err());
        assert!(how(json!({"input": "a", "speed": 2.0, "stream_format": "sse"})).is_err());
        assert!(how(json!({"input": "a", "gap": 9})).is_err());
        assert!(how(json!({"input": "a", "stream_format": "ogg"})).is_err());
        // an unknown format is generate's to refuse (400), before any sound
        assert_eq!(how(json!({"input": "a", "response_format": "ogg"})), Ok(None));
    }

    /// Reads a chunked answer off a socket: (head, body)
    fn dechunk(raw: &[u8]) -> (String, Vec<u8>) {
        let at = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let head = String::from_utf8_lossy(&raw[..at]).to_string();
        let (mut body, mut i) = (Vec::new(), at);
        loop {
            let eol = raw[i..].windows(2).position(|w| w == b"\r\n").unwrap() + i;
            let n = usize::from_str_radix(std::str::from_utf8(&raw[i..eol]).unwrap(), 16).unwrap();
            if n == 0 {
                assert_eq!(&raw[eol..], b"\r\n\r\n", "the last chunk ends the answer");
                return (head, body);
            }
            body.extend_from_slice(&raw[eol + 2..eol + 2 + n]);
            i = eol + 2 + n + 2;
        }
    }

    /// Two parts of one stream through an Out on a real socket: pcm at 24 kHz from a 48 kHz engine, a pause between
    #[test]
    fn parts_are_one_chunked_stream() {
        use super::{Fmt, Out, Streamed};
        use std::io::Read;
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut c = std::net::TcpStream::connect(addr).unwrap();
            let mut raw = Vec::new();
            c.read_to_end(&mut raw).unwrap();
            raw
        });
        let mut conn = crate::http::Conn::Tcp(l.accept().unwrap().0);
        let mut out = Out::new(&mut conn, Streamed::Audio(Fmt::Pcm), "", 1.0, None);
        out.rate = 48_000;
        out.send(&vec![0.1; 48_000]).unwrap();
        out.silence(0.5).unwrap();
        out.send(&vec![0.1; 48_000]).unwrap();
        out.finish(Ok((None, Vec::new())));
        drop(conn);
        let (head, body) = dechunk(&client.join().unwrap());
        assert!(head.contains("Transfer-Encoding: chunked") && head.contains("Content-Type: audio/pcm") && head.contains("X-Sample-Rate: 24000"), "{head}");
        // 2.5 s at 24 kHz, 16-bit (ffmpeg's resampler trims a few samples)
        if crate::speechfmt::have_ffmpeg() {
            assert!((body.len() as i64 - 120_000).abs() < 1500, "{}", body.len());
        }
    }

    #[test]
    fn a_device_failure_retires_the_server_a_bad_request_does_not() {
        assert!(gpu_failed("conv1d: sycl: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)"));
        assert!(!gpu_failed("ref_text: the recording's exact transcript"));
        assert!(!gpu_failed("cancelled"));
        // no card open here: nothing to count
        assert_eq!(own_vram_mb(), 0);
    }

    #[test]
    fn byte_ranges() {
        let h = b"GET /x HTTP/1.1\r\nHost: a\r\nRange: bytes=10-19\r\n\r\n";
        assert_eq!(range(h, 100), Some((10, 19)));
        assert_eq!(range(b"Range: bytes=90-\r\n", 100), Some((90, 99)));
        assert_eq!(range(b"Range: bytes=-5\r\n", 100), Some((95, 99)));
        assert_eq!(range(b"Range: bytes=200-\r\n", 100), None);
        assert_eq!(range(b"Host: a\r\n", 100), None);
    }
}

