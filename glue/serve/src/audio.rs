//! The audio server: the reference server's speech endpoint (SGLang-Omni's, which MiniMax Music 3 ships with) over one
//! audio engine (`nextsycl_audio::AudioEngine`), and with `wfe` a web page that drives it with every option it has.
//!
//! ```text
//!   POST /v1/audio/speech         a song: {input: the lyrics, instructions: the description, seed, max_new_tokens
//!                                 (frames, 25 a second) | seconds, response_format: wav (the file) | url (JSON with its
//!                                 link)} and ours: steps, cfg, options {NAME: value} (the engine's own: /api/info);
//!                                 speech: {input: the text, voice, language, instructions (its style; a designed
//!                                 voice's description), ref_audio (base64 WAV / data URL: a voice to clone),
//!                                 ref_text, seconds (the most), seed, response_format, options}
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
//! One request runs at a time (one engine on its GPU); the others wait their turn.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nextsycl_audio::{AudioEngine, AudioRequest};
use nextsycl_core::Gpu;
use serde_json::{json, Value};

use crate::http::{self, Conn};
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
    engine: Mutex<Box<dyn AudioEngine>>,
    progress: Mutex<Value>,
    waiting: AtomicUsize,
    cancel: AtomicBool,
    history: Mutex<VecDeque<Value>>,
    /// ends the process after an idle stretch (`--idle-exit`; none: never)
    pub idle: Option<&'static crate::idle::Idle>,
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
        Ok(AudioServer { model, out_dir, wfe, cors, gpu, tele, engine: Mutex::new(engine), progress: Mutex::new(json!({"busy": false})),
                         idle: None, waiting: AtomicUsize::new(0), cancel: AtomicBool::new(false), history })
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
                http::respond_with(&mut s, 200, &p, &cors)
            }
            ("GET", "/api/history") => http::respond_with(&mut s, 200, &json!({"data": self.history.lock().unwrap().iter().collect::<Vec<_>>()}), &cors),
            ("POST", "/api/cancel") => {
                let busy = self.progress.lock().unwrap()["busy"] == true;
                if busy {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                http::respond_with(&mut s, 200, &json!({"cancelled": busy}), &cors)
            }
            ("GET", p) if p.starts_with("/v1/audio/files/") => self.file(&mut s, &p["/v1/audio/files/".len()..], &req.head, &cors),
            ("POST", "/v1/audio/speech") | ("POST", "/audio/speech") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                match self.generate(&body, &base) {
                    Ok((Some(v), _)) => http::respond_with(&mut s, 200, &v, &cors),
                    Ok((None, wav)) => http::respond_bytes(&mut s, 200, "audio/wav", &wav, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            _ => http::respond_with(&mut s, 404, &err(format!("no route {} {}", req.method, req.path)), &cors),
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
        let Ok(e) = self.engine.try_lock() else {
            // busy: what does not change
            return json!({"model": self.model, "busy": true});
        };
        let d = e.defaults();
        json!({
            "model": self.model, "arch": e.arch(), "lyrics": e.lyrics(), "wfe": self.wfe.is_some(),
            "speech": e.speech().map(|sp| json!({"voices": sp.voices, "languages": sp.languages, "instructions": sp.instructions, "design": sp.design,
                                                 "clone": sp.clone, "clone_needs_text": sp.clone_needs_text})),
            "defaults": {"seconds": f(d.seconds), "max_seconds": f(d.max_seconds), "steps": d.steps, "cfg": f(d.cfg), "rate": d.rate},
            "options": e.options().iter().filter(|o| o.at != nextsycl_core::At::Load)
                .map(|o| json!({"name": o.name, "value": o.value, "help": o.help})).collect::<Vec<_>>(),
            "report": e.report(),
            "saves": self.out_dir.is_some(),
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

    /// Runs a request: (JSON for response_format url, else none) and the WAV's bytes
    fn generate(&self, b: &Value, base: &str) -> Result<(Option<Value>, Vec<u8>), (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model && *m != "MiniMaxAI/MiniMax-Music3" && !m.starts_with("Qwen/Qwen3-TTS")) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let text = |k: &[&str]| k.iter().find_map(|k| b[*k].as_str()).filter(|p| !p.trim().is_empty()).map(str::to_string);
        let format = b["response_format"].as_str().unwrap_or("wav");
        if !["wav", "url"].contains(&format) {
            return Err(bad(format!("response_format {format}: wav or url")));
        }
        if format == "url" && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        if b.get("stream").and_then(Value::as_bool) == Some(true) {
            return Err(bad("stream: not served (the sound is made whole)".into()));
        }
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
        // one request at a time
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let engine = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let speech = engine.speech().is_some();
        // speech: input is the text to say, instructions its style or voice; a song: input the lyrics, instructions
        // the music
        let (prompt, lyrics, instructions) = if speech {
            (text(&["input", "text"]).ok_or_else(|| bad("input (the text to say): required".into()))?, None, text(&["instructions"]))
        } else {
            let p = text(&["instructions", "prompt", "description"]).ok_or_else(|| bad("instructions (the music's description): required".into()))?;
            (p, text(&["input", "lyrics"]), None)
        };
        let frames_per_s = if speech { 12.5 } else { 25.0 };
        let seconds = num(&["seconds", "duration", "audio_duration"]).or_else(|| num(&["max_new_tokens"]).map(|f| f / frames_per_s)).map(|v| v as f32);
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
        };
        if !given.is_empty() {
            nextsycl_core::options::resolve(&given, engine.options(), nextsycl_core::At::Request, &self.model).map_err(bad)?;
            req.extra = nextsycl_core::options::by_name(&given, engine.options());
        }
        let d = engine.defaults();
        let (steps, cfg) = (req.steps.unwrap_or(d.steps), req.cfg.unwrap_or(d.cfg));
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        self.cancel.store(false, Ordering::Relaxed);
        let p = &self.progress;
        let want = req.seconds.unwrap_or(d.seconds);
        *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "phase": "prompt", "at": 0, "of": 1, "seconds": 0, "want": want});
        let out = engine.generate(&req, &mut |st| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "phase": st.phase, "at": st.at, "of": st.of, "seconds": st.seconds, "want": want});
            if self.cancel.load(Ordering::Relaxed) {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        });
        *p.lock().unwrap() = json!({"busy": false});
        drop(engine);
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
            meta["voice"] = json!(req.voice);
            meta["language"] = json!(req.language);
            meta["instructions"] = json!(instructions);
            meta["cloned"] = json!(req.reference.is_some());
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
                if format == "url" {
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

#[cfg(test)]
mod tests {
    use super::range;

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
