//! The 3D server: one 3D engine (`nextsycl_model3d::Model3dEngine`) over HTTP, and with `wfe` a web page that drives
//! it (a viewer of what it made).
//!
//! ```text
//!   POST /v1/model3d/generate     a mesh from pictures: {image (PNG / JPEG / WebP as base64 or a data: URL) | images
//!                                 [{image, side}] (several views of the object), texture (bool), seed, steps, cfg,
//!                                 resolution, faces, texture_size, remove_background (bool), options {NAME: value},
//!                                 response_format: glb (the file) | url (JSON with its link)}
//!   POST /v1/model3d/rig          a mesh rigged (a skeleton and skin weights) or animated (prompt: the motion):
//!                                 {mesh (GLB or OBJ, base64 / a data: URL) | file (a GLB made here), prompt, frames,
//!                                 fps, seed, steps, cfg, options, response_format}
//!   GET  /v1/model3d/files/<f>    a GLB made here (and its picture, <f>.png; its settings, <f>.json)
//!   POST /api/cancel              stops the request running (between steps)
//!   GET  /v1/models, /health
//!   GET  /api/info                the model, what it does, its defaults, its options
//!   GET  /api/progress            the request running: its phase, how far, seconds
//!   GET  /api/history             the meshes in the output directory, newest first (read from it each time)
//!   GET  /api/gpu                 the card: VRAM, power, temperatures
//!   GET  /, /ui/..., /static/...  the web front end (--wfe: wfe/model3d, built into dist/wfe)
//! ```
//!
//! One request runs at a time (one engine on its GPU); the others wait their turn.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nextsycl_core::Gpu;
use nextsycl_model3d::{MeshRequest, Model3dEngine, RigRequest, View};
use serde_json::{json, Value};

use crate::http::{self, Conn};
use crate::serve::cors_headers;
use crate::telemetry::Telemetry;

/// Sets the progress back to not busy when dropped
struct Idle<'a>(&'a Mutex<Value>);

impl Drop for Idle<'_> {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = json!({"busy": false});
    }
}

pub struct Model3dServer {
    /// the model id clients name
    pub model: String,
    /// where meshes are saved (and served from); none: only returned
    pub out_dir: Option<PathBuf>,
    /// the built front ends (dist/wfe); none: no page
    pub wfe: Option<PathBuf>,
    pub cors: Vec<String>,
    gpu: Arc<Gpu>,
    tele: Arc<Telemetry>,
    engine: Mutex<Box<dyn Model3dEngine>>,
    progress: Mutex<Value>,
    waiting: AtomicUsize,
    cancel: crate::cancel::Cancel,
    history: Mutex<VecDeque<Value>>,
    /// ends the process after an idle stretch (`--idle-exit`; none: never)
    pub idle: Option<&'static crate::idle::Idle>,
}

fn f(x: f32) -> Value {
    json!(((x as f64) * 1e4).round() / 1e4)
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn err(m: impl Into<String>) -> Value {
    json!({"error": {"message": m.into(), "type": "invalid_request_error"}})
}

/// A file name made here: letters, digits, - _ . and no way out of the directory
fn safe(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && !name.contains("..")
}

/// A mesh's history entry: its settings (the sidecar <name>.json), its links
pub fn entry(glb: &Path) -> Option<Value> {
    let name = glb.file_name()?.to_string_lossy().into_owned();
    let mut e: Value = std::fs::read(glb.with_extension("json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| json!({}));
    e["url"] = json!(format!("/v1/model3d/files/{name}"));
    if glb.with_extension("png").is_file() {
        e["picture"] = json!(format!("/v1/model3d/files/{}", name.replace(".glb", ".png")));
    }
    e["bytes"] = json!(std::fs::metadata(glb).ok()?.len());
    if e.get("created").is_none() {
        e["created"] = json!(std::fs::metadata(glb).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs());
    }
    Some(e)
}

/// The meshes already in `dir`, newest first
pub fn earlier(dir: &Path) -> VecDeque<Value> {
    let mut files: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "glb"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files.truncate(200);
    files.into_iter().filter_map(|(_, p)| entry(&p)).collect()
}

impl Model3dServer {
    pub fn new(model: String, gpu: Arc<Gpu>, engine: Box<dyn Model3dEngine>, out_dir: Option<PathBuf>, wfe: Option<PathBuf>, cors: Vec<String>)
               -> Result<Model3dServer, String> {
        if let Some(d) = &out_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        if let Some(w) = &wfe {
            if !w.join("model3d/index.html").is_file() {
                return Err(format!("{}: no web front end built there (./build.sh wfe)", w.display()));
            }
        }
        let tele = Telemetry::start(std::slice::from_ref(&gpu.pci));
        let history = Mutex::new(out_dir.as_deref().map(earlier).unwrap_or_default());
        Ok(Model3dServer { model, out_dir, wfe, cors, gpu, tele, engine: Mutex::new(engine), progress: Mutex::new(json!({"busy": false})),
                           idle: None, waiting: AtomicUsize::new(0), cancel: Default::default(), history })
    }

    /// Answers on `addr` until the process ends
    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        if let Some(i) = self.idle {
            let me = self.clone();
            i.watch("nextsycl model3d", Box::new(move || {
                me.waiting.load(Ordering::Relaxed) > 0 || me.progress.lock().map(|p| p["busy"] == true).unwrap_or(true)
            }));
        }
        eprintln!("nextsycl model3d: {} on http://{addr}/v1/model3d/{{generate,rig}}{}", self.model,
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
                let h = format!("{cors}Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n");
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
                // read again: the front door's model3d.tool writes meshes here too, and deleted ones drop out
                let mut h = self.history.lock().unwrap();
                if let Some(d) = &self.out_dir {
                    *h = earlier(d);
                }
                http::respond_with(&mut s, 200, &json!({"data": h.iter().collect::<Vec<_>>()}), &cors)
            }
            ("POST", "/api/cancel") => {
                let busy = self.progress.lock().unwrap()["busy"] == true;
                let id = serde_json::from_slice::<Value>(&req.body).ok().and_then(|v| v["id"].as_str().map(str::to_string));
                let running = self.cancel.request(id.as_deref(), busy);
                http::respond_with(&mut s, 200, &json!({"cancelled": running, "id": id}), &cors)
            }
            ("GET", p) if p.starts_with("/v1/model3d/files/") => self.file(&mut s, &p["/v1/model3d/files/".len()..], &cors),
            ("POST", p @ ("/v1/model3d/generate" | "/v1/model3d/rig")) => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                match self.run_request(p.ends_with("/rig"), &body, &base) {
                    Ok((Some(v), _)) => http::respond_with(&mut s, 200, &v, &cors),
                    Ok((None, glb)) => http::respond_bytes(&mut s, 200, "model/gltf-binary", &glb, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            _ => http::respond_with(&mut s, 404, &err(format!("no route {} {}", req.method, req.path)), &cors),
        }
    }

    fn page(&self, s: &mut Conn, cors: &str) {
        let dir = self.wfe.as_ref().expect("checked by the route").join("model3d");
        match (std::fs::read_to_string(dir.join("index.html")), std::fs::read_to_string(dir.join("style.css"))) {
            (Ok(h), Ok(c)) => http::respond_bytes(s, 200, "text/html; charset=utf-8", h.replace("__CSS__", &c).as_bytes(),
                                                  &format!("{cors}Cache-Control: no-cache\r\n")),
            _ => http::respond_with(s, 500, &err(format!("{}: the page is not built (./build.sh wfe)", dir.display())), cors),
        }
    }

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
            return json!({"model": self.model, "busy": true});
        };
        let (a, d) = (e.abilities(), e.defaults());
        json!({
            "model": self.model, "arch": e.arch(), "wfe": self.wfe.is_some(),
            "abilities": {"generate": a.generate, "texture": a.texture, "views": a.views, "rig": a.rig, "animate": a.animate},
            "defaults": {"steps": d.steps, "cfg": f(d.cfg), "resolution": d.resolution, "faces": d.faces, "texture_size": d.texture_size,
                         "frames": d.frames, "fps": f(d.fps)},
            "options": e.options().iter().filter(|o| o.at != nextsycl_core::At::Load)
                .map(|o| json!({"name": o.name, "value": o.value, "help": o.help})).collect::<Vec<_>>(),
            "report": e.report(),
            "saves": self.out_dir.is_some(),
        })
    }

    /// A file made here: the GLB, its picture (.png) or its settings (.json)
    fn file(&self, s: &mut Conn, name: &str, cors: &str) {
        let kind = match name.rsplit('.').next() {
            Some("glb") => "model/gltf-binary",
            Some("png") => "image/png",
            Some("json") => "application/json",
            _ => "",
        };
        let b = match (&self.out_dir, safe(name) && !kind.is_empty()) {
            (Some(d), true) => std::fs::read(d.join(name)).ok(),
            _ => None,
        };
        match b {
            Some(b) => http::respond_bytes(s, 200, kind, &b, &format!("{cors}Cache-Control: max-age=31536000, immutable\r\n")),
            None => http::respond_with(s, 404, &err(format!("no file {name}")), cors),
        }
    }

    /// The request's pictures: `image`, or `images` (each a picture or {image, side})
    fn views(b: &Value) -> Result<Vec<(View, Vec<u8>)>, String> {
        let one = |v: &Value, at: &str| -> Result<(View, Vec<u8>), String> {
            let (pic, side) = match v {
                Value::String(s) => (s.clone(), None),
                Value::Object(_) => (v["image"].as_str().ok_or_else(|| format!("{at}: {{image, side}}"))?.to_string(), v["side"].as_str().map(str::to_string)),
                _ => return Err(format!("{at}: a picture (base64 / a data: URL) or {{image, side}}")),
            };
            let bytes = http::unbase64(&pic).map_err(|e| format!("{at}: {e}"))?;
            let image = nextsycl_model3d::read_picture(&bytes).map_err(|e| format!("{at}: {e}"))?;
            Ok((View { image, side }, bytes))
        };
        let mut out = Vec::new();
        if !b["image"].is_null() {
            out.push(one(&b["image"], "image")?);
        }
        for (i, v) in b["images"].as_array().into_iter().flatten().enumerate() {
            out.push(one(v, &format!("images[{i}]"))?);
        }
        Ok(out)
    }

    /// Runs a request: (JSON for response_format url, else none) and the GLB's bytes
    fn run_request(&self, rig: bool, b: &Value, base: &str) -> Result<(Option<Value>, Vec<u8>), (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let format = b["response_format"].as_str().unwrap_or("glb");
        if !["glb", "url"].contains(&format) {
            return Err(bad(format!("response_format {format}: glb or url")));
        }
        if format == "url" && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        let num = |k: &str| b[k].as_f64();
        let seed = match &b["seed"] {
            Value::Number(x) => x.as_u64().ok_or_else(|| bad("seed: a whole number".into()))?,
            _ => SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 % 1_000_000).unwrap_or(7),
        };
        let given: nextsycl_core::options::Given = b["options"].as_object().into_iter().flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
        // what comes in, read before the engine is taken (a bad picture fails at once)
        let views = if rig { Vec::new() } else { Self::views(b).map_err(bad)? };
        // a mesh given to model3d.generate is painted (the pictures say how), not made
        let paint_given = !rig && (b["mesh"].as_str().is_some_and(|m| !m.is_empty()) || b["file"].as_str().is_some_and(|m| !m.is_empty()));
        let mesh_in = if rig || paint_given {
            let bytes = match (b["mesh"].as_str().filter(|m| !m.is_empty()), b["file"].as_str().filter(|m| !m.is_empty())) {
                (Some(m), _) => http::unbase64(m).map_err(|e| bad(format!("mesh: {e}")))?,
                (None, Some(f)) => {
                    let f = f.rsplit('/').next().unwrap_or(f);
                    let path = self.out_dir.as_ref().filter(|_| safe(f) && f.ends_with(".glb")).map(|d| d.join(f)).filter(|p| p.is_file())
                        .ok_or_else(|| (404u16, format!("file {f}: no GLB of that name here")))?;
                    std::fs::read(&path).map_err(|e| (500u16, format!("{f}: {e}")))?
                }
                _ => return Err(bad("mesh (a GLB or OBJ, base64 or a data: URL) or file (a GLB made here): required".into())),
            };
            Some(nextsycl_model3d::read_mesh(&bytes).map_err(|e| bad(format!("mesh: {e}")))?)
        } else {
            None
        };
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let engine = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        let a = engine.abilities();
        let extra = if given.is_empty() {
            Default::default()
        } else {
            nextsycl_core::options::resolve(&given, engine.options(), nextsycl_core::At::Request, &self.model).map_err(bad)?;
            nextsycl_core::options::by_name(&given, engine.options())
        };
        let d = engine.defaults();
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        let what = if rig { b["prompt"].as_str().unwrap_or("rig").to_string() } else { format!("{} picture(s)", views.len()) };
        let p = &self.progress;
        *p.lock().unwrap() = json!({"busy": true, "prompt": what, "phase": "start", "at": 0, "of": 1, "seconds": 0});
        // not busy again however this ends: a request refused below (texture on a shape-only model) left the server
        // busy for good and the front door counted it working (2026-10-19)
        let _idle = Idle(p);
        let mut report = |st: nextsycl_model3d::Step| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": what, "phase": st.phase, "at": st.at, "of": st.of, "seconds": st.seconds});
            if self.cancel.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        };
        let mut meta = json!({"model": self.model, "arch": engine.arch(), "seed": seed});
        let out = if rig {
            if !a.rig && !a.animate {
                return Err(bad(format!("{} makes meshes; it does not rig them (model3d.generate)", self.model)));
            }
            let prompt = b["prompt"].as_str().map(str::to_string).filter(|p| !p.trim().is_empty());
            if a.animate && !a.rig && prompt.is_none() {
                return Err(bad("prompt (the motion, in words): required by an animating model".into()));
            }
            let req = RigRequest { mesh: mesh_in.expect("read above"), seed, prompt: prompt.clone(), frames: num("frames").map(|v| v as u32),
                                   fps: num("fps").map(|v| v as f32), steps: num("steps").map(|v| v as u32), cfg: num("cfg").map(|v| v as f32), extra };
            meta["prompt"] = json!(prompt);
            meta["frames"] = json!(req.frames.unwrap_or(d.frames));
            engine.rig(&req, &mut report)
        } else {
            if !a.generate && !(paint_given && a.texture) {
                return Err(bad(format!("{} rigs meshes; it does not make them (model3d.rig)", self.model)));
            }
            if paint_given && !a.texture {
                return Err(bad(format!("{} has no paint model: it cannot paint a mesh (texture)", self.model)));
            }
            if views.is_empty() {
                return Err(bad("image (a picture of the object) or images (several views): required".into()));
            }
            if views.len() > a.views.max(1) as usize {
                return Err(bad(format!("{} pictures: {} takes {} at most", views.len(), self.model, a.views.max(1))));
            }
            let texture = b["texture"].as_bool().unwrap_or(a.texture);
            if paint_given && !texture {
                return Err(bad("texture: false with a mesh given leaves nothing to do".into()));
            }
            if texture && !a.texture {
                return Err(bad(format!("texture: {} makes the shape alone (texture: false)", self.model)));
            }
            let req = MeshRequest { views: views.iter().map(|v| v.0.clone()).collect(), mesh: mesh_in, texture, seed, steps: num("steps").map(|v| v as u32),
                                    cfg: num("cfg").map(|v| v as f32), resolution: num("resolution").map(|v| v as u32), faces: num("faces").map(|v| v as u32),
                                    texture_size: num("texture_size").map(|v| v as u32), remove_background: b["remove_background"].as_bool(), extra };
            meta["texture"] = json!(texture);
            meta["steps"] = json!(req.steps.unwrap_or(d.steps));
            meta["cfg"] = f(req.cfg.unwrap_or(d.cfg));
            meta["resolution"] = json!(req.resolution.unwrap_or(d.resolution));
            meta["faces"] = json!(req.faces.unwrap_or(d.faces));
            meta["views"] = json!(views.iter().map(|v| v.0.side.clone()).collect::<Vec<_>>());
            engine.generate(&req, &mut report)
        };
        *p.lock().unwrap() = json!({"busy": false});
        drop(engine);
        let mesh = out.map_err(|e| (if e.0 == "cancelled" { 499 } else { 400 }, e.0))?;
        let glb = nextsycl_model3d::mesh::glb::write(&mesh).map_err(|e| (500u16, format!("writing the GLB: {e}")))?;
        let took = t0.elapsed().as_secs_f64();
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let created = now();
        meta["took"] = json!(took);
        meta["wh"] = json!(wh);
        meta["created"] = json!(created);
        meta["triangles"] = json!(mesh.triangles());
        meta["textured"] = json!(mesh.material.as_ref().is_some_and(|m| m.base_color.is_some()));
        meta["rigged"] = json!(mesh.skin.as_ref().map(|s| s.joints.len()));
        meta["animated"] = json!(mesh.animation.as_ref().map(|a| a.frames.len()));
        let mut answer = None;
        if let Some(dir) = &self.out_dir {
            let name = format!("{created}-{}-{seed}.glb", self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"));
            let path = dir.join(&name);
            std::fs::write(&path, &glb).map_err(|e| (500u16, format!("{}: {e}", dir.display())))?;
            // the first picture beside it (the gallery shows it), and the settings
            if let Some((_, bytes)) = views.first() {
                if let Ok(t) = nextsycl_model3d::read_picture(bytes).and_then(|t| t.png()) {
                    let _ = std::fs::write(path.with_extension("png"), t);
                }
            }
            let _ = std::fs::write(path.with_extension("json"), serde_json::to_vec_pretty(&meta).unwrap_or_default());
            if let Some(e) = entry(&path) {
                let mut h = self.history.lock().unwrap();
                h.push_front(e.clone());
                h.truncate(200);
                if format == "url" {
                    let mut v = e;
                    v["url"] = json!(format!("{base}/v1/model3d/files/{name}"));
                    if let Some(pic) = v["picture"].as_str().map(str::to_string) {
                        v["picture"] = json!(format!("{base}{pic}"));
                    }
                    answer = Some(json!({"created": created, "data": [v], "nextsycl": {"took": took, "seed": seed, "wh": wh,
                                                                                     "triangles": mesh.triangles()}}));
                }
            }
        }
        Ok((answer, glb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_stay_inside() {
        assert!(safe("1792-trellis2-7.glb"));
        assert!(!safe("../x.glb"));
        assert!(!safe("a/b.glb"));
        assert!(!safe(""));
    }

    #[test]
    fn pictures_are_read_from_either_field() {
        let mut t = nextsycl_model3d::Texture::new(2, 2);
        t.rgba.fill(200);
        let png = t.png().unwrap();
        let b64 = |b: &[u8]| {
            const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut s = String::new();
            for c in b.chunks(3) {
                let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
                for k in 0..4 {
                    s.push(if k <= c.len() { A[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
                }
            }
            s
        };
        let p = format!("data:image/png;base64,{}", b64(&png));
        let v = Model3dServer::views(&json!({"image": p, "images": [{"image": p, "side": "back"}]})).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].0.side.as_deref(), Some("back"));
        assert_eq!(v[0].0.image.width, 2);
        assert!(Model3dServer::views(&json!({"image": 5})).is_err());
    }
}
