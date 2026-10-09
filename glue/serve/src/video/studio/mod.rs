//! The studio: the web front end's own API - the clip queue, projects and scenes, the finished clips and films,
//! the summary, and the language-model switch - on top of the engine daemon (`sycl-h3 serve` runs it on the host;
//! it talks to the daemon over its Unix socket). It replaces the legacy Python server the front end was written for
//! and answers its routes the same way (docs/LEGACY-API.md), so the front end, scene files and queue tools work as
//! they did; the clips themselves are made by the engine (`generate` jobs) instead of a PyTorch container.
//!
//! State: the queue and holds (`queue.json`, `paused.json`) and the running job (`job.json`) in the studio's own
//! directory; each finished clip's record (`<clip>.json`) beside its mp4 in the clips directory.

pub mod http;
pub mod library;
pub mod llm;
pub mod queue;
pub mod runner;
pub mod speech;

use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use self::http::Target;
use serde_json::{json, Value};

use queue::{num, project_of, s, Queue};

pub const TYPES_TS: &str = include_str!("types.ts");

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// Python's round(): halves to the even neighbour (the legacy numbers depend on it).
pub fn pyround(x: f64) -> f64 {
    let f = x.floor();
    if (x - f - 0.5).abs() < 1e-12 {
        if f % 2.0 == 0.0 { f } else { f + 1.0 }
    } else {
        x.round()
    }
}

/// A clip name from the local time: h3_YYYYmmdd_HHMMSS (unique: waits out a second already taken).
pub fn stamp() -> String {
    loop {
        let n = std::process::Command::new("date").arg("+h3_%Y%m%d_%H%M%S").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        let mut last = LAST_STAMP.lock().unwrap();
        if !n.is_empty() && *last != n {
            *last = n.clone();
            return n;
        }
        drop(last);
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}
static LAST_STAMP: Mutex<String> = Mutex::new(String::new());

pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, data)?;
    std::fs::rename(tmp, path)
}

/// An RPC error: a code, a message, an HTTP status.
#[derive(Debug)]
pub struct RpcError {
    pub code: &'static str,
    pub message: String,
    pub http: u16,
}

impl RpcError {
    pub fn param(m: &str) -> RpcError {
        RpcError { code: "invalid_param", message: m.into(), http: 400 }
    }
    pub fn not_found(m: &str) -> RpcError {
        RpcError { code: "not_found", message: m.into(), http: 404 }
    }
    fn new(code: &'static str, m: &str, http: u16) -> RpcError {
        RpcError { code, message: m.into(), http }
    }
}

pub struct Job {
    /// the job record (the item, the resolved anchors, the name, the start time ...)
    pub rec: Value,
    pub engine_id: u64,
    pub energy0: Option<f64>,
    pub saved: bool,
    /// (log lines seen, when the count last changed): the stall watchdog
    pub stall: (usize, f64),
    pub finished: bool,
    /// the engine's last answer about the job
    pub last: Value,
}

#[derive(Default)]
pub struct State {
    pub q: Queue,
    pub job: Option<Job>,
    pub last_file: Option<String>,
    pub current_item: Option<Value>,
    pub fails: u32,
    pub torn: bool,
}

#[derive(Default)]
pub struct Cache {
    sidecars: Option<(f64, Vec<(String, Value)>)>,
    gpu: Option<(f64, Value)>,
}

pub struct Studio {
    /// the clips directory on this machine, and what the engine calls it
    pub out: PathBuf,
    pub out_in: String,
    /// the studio's state files
    pub dir: PathBuf,
    pub socket: PathBuf,
    pub ui: PathBuf,
    pub templates: Option<PathBuf>,
    pub gpustat: PathBuf,
    pub stall_s: u64,
    pub st: Mutex<State>,
    pub cache: Mutex<Cache>,
    pub llm: &'static llm::Llm,
}

pub struct Options {
    pub listen: String,
    pub socket: PathBuf,
    pub ui: PathBuf,
    pub out: PathBuf,
    pub out_in: String,
    pub dir: PathBuf,
    pub templates: Option<PathBuf>,
    pub gpustat: PathBuf,
    pub llm_modes: Option<PathBuf>,
    pub logs: PathBuf,
    /// nextsycl: the front end's denoiser names and the daemon's model ids (INT8 -> minimax-h3, ...)
    pub engines: Vec<(String, String)>,
}

impl Studio {
    /// The engine has a model loaded (or is loading one) on some GPU: no language model may start.
    /// Whether the engine holds the card the language models use: a GPU marked shared (`--shared-gpu`) has an engine
    /// loaded or a job running. With no GPU marked shared, any GPU counts (one card for both). An engine on a GPU of
    /// its own does not keep the model off its card (2026-10-05: a job on the second GPU held the restore back).
    pub fn engine_holds_card(&self) -> bool {
        match http::call(&Target::Unix(self.socket.clone()), "GET", "/engine/status", None) {
            Ok(v) => v["gpus"].as_array().is_some_and(|g| {
                let any_shared = g.iter().any(|x| x["shared"] == true);
                g.iter().filter(|x| !any_shared || x["shared"] == true).any(|x| !x["worker"].is_null() || !x["running"].is_null())
            }),
            Err(_) => false,
        }
    }

    pub fn has_runnable(&self) -> bool {
        let st = self.st.lock().unwrap();
        self.running(&st) || st.q.items.iter().any(|i| !st.q.held(i))
    }

    fn versioned_status(&self) -> Value {
        let mut st = self.st.lock().unwrap();
        let mut v = self.status(&mut st);
        let sig = json!([v["job"]["name"], v["stage"], v["pct"], v["steps_done"], v["done"], v["error"], v["exited"], st.q.items.len(), st.last_file]);
        let mut h: u64 = 1469598103934665603;
        for b in sig.to_string().bytes() {
            h = (h ^ b as u64).wrapping_mul(1099511628211);
        }
        v["version"] = json!(format!("{:012x}", h & 0xffff_ffff_ffff));
        v["queue_len"] = json!(st.q.items.len());
        v["stale"] = json!(false);
        v
    }

    fn hold(&self, on: bool, project: &str, batch: &str) -> Value {
        let mut st = self.st.lock().unwrap();
        let scope = if !batch.is_empty() {
            if on { st.q.paused_batches.insert(batch.into()) } else { st.q.paused_batches.remove(batch) };
            format!("batch {batch}")
        } else if !project.is_empty() {
            if on { st.q.paused_projects.insert(project.into()) } else { st.q.paused_projects.remove(project) };
            format!("project {project}")
        } else {
            st.q.paused = on;
            if !on {
                st.q.paused_projects.clear();
                st.q.paused_batches.clear();
            }
            "queue".into()
        };
        if !on {
            st.fails = 0;
        }
        st.q.save_paused(&self.dir);
        let held = st.q.items.iter().filter(|i| st.q.held(i)).count();
        json!({if on { "paused" } else { "resumed" }: scope, "queued": st.q.items.len(), "held": held, "paused_all": st.q.paused,
               "paused_projects": st.q.paused_projects, "paused_batches": st.q.paused_batches})
    }

    fn generate(&self, p: &Value) -> Result<Value, RpcError> {
        let item = self.build_item(p)?;
        let mut st = self.st.lock().unwrap();
        let _ = self.status(&mut st); // registers a just-finished clip before anything replaces it
        // busy: a clip running, or a runnable item waiting (an all-held queue does not block a direct request)
        let busy = self.running(&st) || st.q.items.iter().any(|i| !st.q.held(i));
        if busy {
            if p["queue"] != true {
                return Err(RpcError::new("busy", "a job is running; pass queue:true to queue it", 409));
            }
            let held = st.q.held(&item);
            let (label, project, batch) = (s(&item, "label"), project_of(&item), s(&item, "batch"));
            st.q.items.push(item);
            st.q.save(&self.dir);
            if st.q.paused {
                st.q.paused = false;
                st.q.save_paused(&self.dir);
            }
            return Ok(json!({"queued": true, "position": st.q.items.len(), "label": label, "project": project, "batch": batch, "held": held}));
        }
        self.launch(&mut st, item).map_err(|e| RpcError::new("internal", &e, 503))?;
        Ok(json!({"queued": false, "job": st.job.as_ref().map(|j| j.rec.clone())}))
    }

    fn rpc(&self, method: &str, p: &Value) -> Result<Value, RpcError> {
        let ps = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        Ok(match method {
            "status" => self.versioned_status(),
            "gpu" => self.gpu(),
            "summary" => self.summary(&mut self.st.lock().unwrap()),
            "engines.list" => json!({"engines": self.engines()}),
            "canvases.list" => json!({"canvases": self.canvases()}),
            "plan" => self.plan(),
            "templates.list" => json!({"templates": self.templates()}),
            "wait" => {
                let timeout = num(p, "timeout_s", 30.0, 0.0, 60.0, false)?.as_f64().unwrap_or(30.0);
                let want = ps("version");
                let deadline = now() + timeout;
                loop {
                    let mut v = self.versioned_status();
                    if v["version"].as_str() != Some(want.as_str()) || now() >= deadline {
                        v["changed"] = json!(v["version"].as_str() != Some(want.as_str()));
                        break v;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
            "generate" => self.generate(p)?,
            "queue.list" => json!({"queue": self.st.lock().unwrap().q.items}),
            "queue.pause" => self.hold(true, &ps("project"), &ps("batch")),
            "queue.resume" => self.hold(false, &ps("project"), &ps("batch")),
            "queue.clear" => {
                let mut st = self.st.lock().unwrap();
                let before = st.q.items.len();
                let (pr, b) = (ps("project"), ps("batch"));
                if !b.is_empty() {
                    st.q.items.retain(|i| s(i, "batch") != b);
                } else if !pr.is_empty() {
                    st.q.items.retain(|i| project_of(i) != pr);
                } else {
                    st.q.items.clear();
                }
                st.q.save(&self.dir);
                json!({"cleared": before - st.q.items.len()})
            }
            "cancel" => {
                let mut st = self.st.lock().unwrap();
                self.cancel(&mut st).map_err(|e| RpcError::new("not_running", &e, 409))?;
                json!({"stopped": true})
            }
            "llm.mode" => self.llm.set(self, p.get("mode").and_then(|m| m.as_str()))?,
            "scene.export" => self.scene_export(&self.st.lock().unwrap(), &ps("project")),
            "scene.import" => {
                if !p["scene"].is_object() {
                    return Err(RpcError::param("scene must be the document object"));
                }
                let pr = ps("project");
                self.scene_import(&mut self.st.lock().unwrap(), &p["scene"], (!pr.is_empty()).then_some(pr.as_str()), p["paused"] == true)?
            }
            "films.list" => json!({"films": self.films()}),
            "timeline" => json!({"clips": self.timeline(&self.st.lock().unwrap(), &ps("project"))}),
            "projects.list" => {
                let st = self.st.lock().unwrap();
                json!({"projects": st.q.projects(), "paused_all": st.q.paused})
            }
            "jobs.list" => {
                let limit = num(p, "limit", 50.0, 1.0, 500.0, true)?.as_u64().unwrap_or(50) as usize;
                json!({"jobs": self.listing().into_iter().take(limit).collect::<Vec<_>>()})
            }
            "jobs.get" => {
                let n = queue::basename(&ps("name"));
                let n = n.trim_end_matches(".mp4");
                self.record(n).ok_or_else(|| RpcError::not_found(&format!("no record for {n}")))?
            }
            other => return Err(RpcError::new("unknown_method", &format!("unknown method {other}"), 404)),
        })
    }

    fn record(&self, name: &str) -> Option<Value> {
        if !library::is_clip_name(name) {
            return None;
        }
        std::fs::read(self.out.join(format!("{name}.json"))).ok().and_then(|b| serde_json::from_slice(&b).ok())
    }

    fn route(&self, stream: &TcpStream, req: http::Request) -> http::Result<()> {
        let path = req.path.clone();
        let query = req.target.split_once('?').map(|q| q.1.to_string()).unwrap_or_default();
        let qv = |k: &str| -> String {
            query.split('&').filter_map(|kv| kv.split_once('=')).find(|(a, _)| *a == k).map(|(_, v)| unescape(v)).unwrap_or_default()
        };
        if path.starts_with("/engine/") {
            if let Err(e) = http::forward(stream, &Target::Unix(self.socket.clone()), &req) {
                let _ = http::respond(stream, 503, &json!({"error": format!("the engine is not running (nextsycl video start): {}", e.0)}));
            }
            return Ok(());
        }
        if req.method == "POST" {
            if let Some(m) = path.strip_prefix("/rpc/") {
                if req.target.contains('?') {
                    return http::respond(stream, 400, &json!({"ok": false, "error": {"code": "invalid_request", "message": "query strings are not accepted"}}));
                }
                let body = if req.body.is_empty() { Ok(json!({})) } else { serde_json::from_slice::<Value>(&req.body) };
                let r = match body {
                    Ok(b) if b.is_object() => self.rpc(m, &b),
                    Ok(_) => Err(RpcError::new("invalid_request", "body must be a JSON object", 400)),
                    Err(_) => Err(RpcError::new("invalid_request", "body must be valid JSON", 400)),
                };
                return match r {
                    Ok(v) => http::respond(stream, 200, &json!({"ok": true, "result": v})),
                    Err(e) => http::respond(stream, e.http, &json!({"ok": false, "error": {"code": e.code, "message": e.message}})),
                };
            }
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
            return match path.as_str() {
                "/api/queue/pause" => {
                    let on = body["on"] == true;
                    self.hold(on, body["project"].as_str().unwrap_or("").trim(), body["batch"].as_str().unwrap_or("").trim());
                    http::respond(stream, 200, &json!({"ok": true}))
                }
                "/api/queue/clear" => {
                    let mut st = self.st.lock().unwrap();
                    let n = st.q.items.len();
                    st.q.items.clear();
                    st.q.save(&self.dir);
                    http::respond(stream, 200, &json!({"cleared": n}))
                }
                "/api/scene/import" => {
                    let doc = if body["scene"].is_object() { &body["scene"] } else { &body };
                    match self.scene_import(&mut self.st.lock().unwrap(), doc, body["project"].as_str(), body["paused"] == true) {
                        Ok(v) => http::respond(stream, 200, &v),
                        Err(e) => http::respond(stream, e.http, &json!({"error": e.message})),
                    }
                }
                "/api/gen" => {
                    // the oldest route: clamps where the RPC rejects
                    let mut b = body.clone();
                    let f = |v: &Value, d: f64| v.as_f64().or_else(|| v.as_str().and_then(|x| x.parse().ok())).unwrap_or(d);
                    b["seconds"] = json!(f(&body["seconds"], 10.0).clamp(1.0, 15.1));
                    b["steps"] = json!((f(&body["steps"], 10.0) as i64).clamp(1, 40));
                    if !matches!(body["te"].as_str(), Some("teacher" | "student")) {
                        b["te"] = json!("teacher");
                    }
                    match self.generate(&b) {
                        Ok(v) if v["queued"] == true => http::respond(stream, 200, &json!({"queued": v["position"], "label": v["label"]})),
                        Ok(v) => http::respond(stream, 200, &v["job"]),
                        Err(e) => http::respond(stream, e.http, &json!({"error": e.message})),
                    }
                }
                _ => http::respond(stream, 404, &json!({"error": format!("no route POST {path}")})),
            };
        }
        let respond_json = |v: &Value| http::respond(stream, 200, v);
        match path.as_str() {
            "/" | "/index.html" => {
                // nextsycl: the built front ends' root (dist/wfe), this one's page in video/
                let html = std::fs::read_to_string(self.ui.join("video/index.html"))?;
                let css = std::fs::read_to_string(self.ui.join("video/style.css")).unwrap_or_default();
                http::respond_bytes(stream, 200, "text/html; charset=utf-8", html.replace("__CSS__", &css).as_bytes())
            }
            "/api/gpu" => respond_json(&self.gpu()),
            "/api/summary" => respond_json(&self.summary(&mut self.st.lock().unwrap())),
            "/api/engines" => respond_json(&json!({"engines": self.engines()})),
            "/api/canvases" => respond_json(&json!({"canvases": self.canvases()})),
            "/api/plan" => respond_json(&self.plan()),
            "/api/status" => respond_json(&self.versioned_status()),
            "/api/list" => respond_json(&json!(self.listing())),
            "/api/templates" => respond_json(&json!(self.templates())),
            "/api/films" => respond_json(&json!(self.films())),
            "/api/timeline" => respond_json(&json!({"clips": self.timeline(&self.st.lock().unwrap(), &qv("project"))})),
            "/api/types.ts" => http::respond_bytes(stream, 200, "application/typescript; charset=utf-8", TYPES_TS.as_bytes()),
            "/api/scene" => {
                let p = qv("project");
                let doc = self.scene_export(&self.st.lock().unwrap(), &p);
                http::respond_bytes(stream, 200, "application/json; charset=utf-8", serde_json::to_string_pretty(&doc).unwrap_or_default().as_bytes())
            }
            "/api/clip" => {
                let label = qv("label");
                let st = self.st.lock().unwrap();
                let running = st.job.as_ref().filter(|j| !j.finished).map(|j| (j.rec.clone(), "running"));
                let found = running.into_iter().chain(st.q.items.iter().map(|i| (i.clone(), if st.q.held(i) { "held" } else { "queued" }))).find(|(i, _)| s(i, "label") == label);
                match found {
                    Some((mut i, state)) => {
                        i["dialogue"] = json!(queue::dialogue_of(&s(&i, "prompt")));
                        i["state"] = json!(state);
                        respond_json(&i)
                    }
                    None => http::respond(stream, 404, &json!({"error": "no queued clip with that label"})),
                }
            }
            _ => {
                if let Some(name) = path.strip_prefix("/api/job/") {
                    let n = queue::basename(name);
                    return match self.record(&n) {
                        Some(v) => respond_json(&v),
                        None => http::respond(stream, 404, &json!({"error": format!("no record for {n}")})),
                    };
                }
                if let Some(t) = path.strip_prefix("/thumb/") {
                    let n = queue::basename(t);
                    let n = n.strip_suffix(".webp").unwrap_or(&n);
                    return match library::is_clip_name(n).then(|| self.thumb(n)).flatten() {
                        Some(b) => http::respond_bytes(stream, 200, "image/webp", &b),
                        None => http::respond(stream, 404, &json!({"error": "no thumbnail"})),
                    };
                }
                if let Some(f) = path.strip_prefix("/out/") {
                    return self.file(stream, &req, &queue::basename(f));
                }
                if let Some(rel) = path.strip_prefix("/ui/") {
                    if !rel.split('/').any(|c| c == ".." || c.is_empty()) {
                        return match std::fs::read(self.ui.join(rel)) {
                            Ok(b) => http::respond_bytes(stream, 200, http::content_type(rel), &b),
                            Err(_) => http::respond(stream, 404, &json!({"error": format!("no front-end file {rel}")})),
                        };
                    }
                }
                if let Some(rel) = path.strip_prefix("/static/") {
                    if let Ok(b) = std::fs::read(self.ui.join("static").join(queue::basename(rel))) {
                        return http::respond_bytes(stream, 200, http::content_type(rel), &b);
                    }
                }
                http::respond(stream, 404, &json!({"error": format!("no route {} {path}", req.method)}))
            }
        }
    }

    /// An mp4 of out/, with byte ranges (the video player seeks).
    fn file(&self, mut stream: &TcpStream, req: &http::Request, name: &str) -> http::Result<()> {
        let p = self.out.join(name);
        if !name.ends_with(".mp4") || !p.is_file() {
            return http::respond(stream, 404, &json!({"error": "no such clip"}));
        }
        let mut f = std::fs::File::open(&p)?;
        let size = f.metadata()?.len();
        let head = String::from_utf8_lossy(&req.head).to_string();
        let range = head.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case("range").then(|| v.trim().to_string())
        });
        let (start, end, partial) = match range.as_deref().and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.split_once('-')) {
            Some((a, b)) => {
                let (a, b) = (a.trim().parse::<u64>().ok(), b.trim().parse::<u64>().ok());
                let start = a.unwrap_or_else(|| size.saturating_sub(b.unwrap_or(0)));
                let end = if a.is_some() { b.unwrap_or(size - 1).min(size - 1) } else { size - 1 };
                (start.min(size.saturating_sub(1)), end, true)
            }
            None => (0, size.saturating_sub(1), false),
        };
        let len = end + 1 - start;
        write!(stream, "HTTP/1.1 {}\r\nContent-Type: video/mp4\r\nAccept-Ranges: bytes\r\nContent-Length: {len}\r\n{}Connection: close\r\n\r\n",
               if partial { "206 Partial Content" } else { "200 OK" },
               if partial { format!("Content-Range: bytes {start}-{end}/{size}\r\n") } else { String::new() })?;
        f.seek(SeekFrom::Start(start))?;
        let mut left = len;
        let mut buf = vec![0u8; 1 << 20];
        while left > 0 {
            let n = f.read(&mut buf[..(left as usize).min(1 << 20)])?;
            if n == 0 {
                break;
            }
            stream.write_all(&buf[..n])?;
            left -= n as u64;
        }
        Ok(())
    }
}

fn unescape(s: &str) -> String {
    let b = s.replace('+', " ").into_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&String::from_utf8_lossy(&b[i + 1..i + 3]), 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn run(o: Options) -> http::Result<()> {
    if !o.ui.join("video/index.html").exists() {
        return Err(http::Error(format!("{}: no built front end there (./build.sh wfe)", o.ui.display())));
    }
    std::fs::create_dir_all(&o.dir)?;
    std::fs::create_dir_all(&o.logs)?;
    let llm: &'static llm::Llm = Box::leak(Box::new(llm::Llm::load(o.llm_modes.clone(), &o.dir, o.logs.clone())));
    let mut st = State { q: Queue::load(&o.dir), ..State::default() };
    // a job that was running when the studio stopped: follow it again if the engine still knows it
    if let Some(rec) = runner::recover(&o.dir) {
        if let Some(id) = rec["id"].as_str().and_then(|i| i.parse::<u64>().ok()) {
            // a clip already saved is nothing to follow: restored as the current job, with no engine state to show,
            // it read as "waiting for the engine" until the next clip (2026-10-09)
            let saved = o.out.join(format!("{}.json", s(&rec, "name"))).exists();
            if !saved {
                st.job = Some(Job { energy0: rec["energy0"].as_f64(), rec, engine_id: id, saved, stall: (0, now()), finished: false, last: Value::Null });
            }
        }
    }
    let studio = Arc::new(Studio {
        out: o.out,
        out_in: o.out_in,
        dir: o.dir,
        socket: o.socket,
        ui: o.ui,
        templates: o.templates,
        gpustat: o.gpustat,
        stall_s: std::env::var("H3_STALL_S").ok().and_then(|v| v.parse().ok()).unwrap_or(1200),
        st: Mutex::new(st),
        cache: Mutex::new(Cache::default()),
        llm,
    });
    let listener = TcpListener::bind(&o.listen).map_err(|e| http::Error(format!("cannot listen on {}: {e}", o.listen)))?;
    let _ = queue::ENGINES.set(o.engines.clone());
    eprintln!("nextsycl video studio {} on http://{}/ - clips in {}, engine via {}", crate::VERSION, o.listen, studio.out.display(), studio.socket.display());
    let sched = studio.clone();
    std::thread::spawn(move || sched.scheduler());
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let studio = studio.clone();
        std::thread::spawn(move || match http::read_request(&stream) {
            Ok(req) => {
                if let Err(e) = studio.route(&stream, req) {
                    let _ = http::respond(&stream, 500, &json!({"error": e.0}));
                }
            }
            Err(e) => {
                let _ = http::respond(&stream, 400, &json!({"error": e.0}));
            }
        });
    }
    Ok(())
}
