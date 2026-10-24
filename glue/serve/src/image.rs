//! The image server: OpenAI's images API over one image engine (`nextsycl_image::ImageEngine`), and with `wfe` a web
//! page that drives it with every option the engine has.
//!
//! ```text
//!   POST /v1/images/generations   OpenAI's body - prompt, n, size, response_format (b64_json | url), background
//!                                 (transparent: RGBA) - and ours: steps, seed, sampler, schedule, shift, cfg,
//!                                 negative_prompt, loras ["name:scale" | {name, scale}], options {NAME: value} (the
//!                                 engine's own: /api/info lists them)
//!   POST /v1/images/edits         OpenAI's form (multipart: image / image[] files - the picture to change, then
//!                                 references -, prompt, the fields above) or JSON with `images`: [base64 or data
//!                                 URLs]; the instructions name the pictures ("the hat from picture 2"); the size
//!                                 follows the last picture's aspect unless one is given; strength (0.05 to 1,
//!                                 default 1): below 1 the picture starts from the first one, noised that far;
//!                                 reference (default true; false: a plain img2img, the pictures do not condition)
//!                                 generations take image (an init picture: a plain img2img, strength below 1)
//!   POST /v1/ocr                  an OCR model's reading: images [base64 or data URLs: pictures or PDFs, split into
//!                                 their pages here] (pages, in order), each_page (several pages each read on its
//!                                 own; default true - false: the model's one-pass multi-page reading), prompt,
//!                                 tiles (a single page's tiles too; default true), figures (read the text inside
//!                                 the pictures it marks; default true), max_tokens, temperature, ngram (0: no
//!                                 repetition rule) -> text (the model's markup kept), tokens, seconds; the text
//!                                 so far is in /api/progress while it is written
//!   GET  /v1/images/files/<f>     a picture made here (saved in the output directory)
//!   GET  /v1/models, /health
//!   GET  /api/info                the model, its defaults, samplers, schedules, the LoRAs it can take
//!   GET  /api/progress            the request running: its picture, step, seconds; requests waiting
//!   GET  /api/history             the pictures made since the server started, newest first
//!   GET  /api/gpu                 the card: VRAM, power, temperatures
//!   GET  /, /ui/..., /static/...  the web front end (--wfe: wfe/image, or wfe/ocr for a model that reads, built into
//!                                 dist/wfe - TSX on snabbdom, H3's scheme; the page's CSS inlined at __CSS__ as H3's
//!                                 server does)
//! ```
//!
//! One request runs at a time (one engine on its GPU); the others wait their turn. A request asking for LoRAs other
//! than those merged in: an engine that takes them on a side path (`side_loras`: Qwen-Image 2.1) runs them beside the
//! weights for that request only - each LoRA file read once and kept, put on the GPU for the request and dropped
//! after, its few-step settings (sigmas) applied to the request; any other engine (or a request that drops a LoRA
//! merged at load) reloads with them merged (the old one is dropped first: its VRAM is the new one's).

use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nextsycl_core::Gpu;
use nextsycl_image::{ImageEngine, ImageRequest, LoraUse, Sampler, Schedule};
use serde_json::{json, Value};

use crate::http::{self, Conn};
use crate::serve::cors_headers;
use crate::telemetry::Telemetry;

/// Loads the engine with these LoRAs merged
pub type Loader = Box<dyn Fn(&[LoraUse], &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>, String> + Send + Sync>;

/// The engine and the LoRAs merged into it
type Loaded = (Box<dyn ImageEngine>, Vec<LoraUse>);

/// A LoRA the server can merge: its id, file, title
#[derive(Clone, Debug)]
pub struct KnownLora {
    pub id: String,
    pub path: PathBuf,
    pub title: String,
    /// its registry entry's settings (a few-step LoRA's sigma preset): on the side path, those an engine option takes
    /// per request apply to the request (a merge applies them at load)
    pub env: Vec<(String, String)>,
}

pub struct ImageServer {
    /// the model id clients name
    pub model: String,
    pub loras_known: Vec<KnownLora>,
    /// where pictures are saved (and served from); none: only returned
    pub out_dir: Option<PathBuf>,
    /// the built front ends (dist/wfe); none: no page
    pub wfe: Option<PathBuf>,
    pub cors: Vec<String>,
    gpu: Arc<Gpu>,
    tele: Arc<Telemetry>,
    load: Loader,
    engine: Mutex<Option<Loaded>>,
    progress: Mutex<Value>,
    waiting: AtomicUsize,
    history: Mutex<VecDeque<Value>>,
    cancel: crate::cancel::Cancel,
    /// ends the process after an idle stretch (`--idle-exit`; none: never)
    pub idle: Option<&'static crate::idle::Idle>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn err(m: impl Into<String>) -> Value {
    json!({"error": {"message": m.into(), "type": "invalid_request_error"}})
}

/// The pictures already in `dir` (an earlier session's), newest first, from the settings in their PNGs
fn earlier(dir: &std::path::Path) -> VecDeque<Value> {
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files.truncate(200);
    files.into_iter().filter_map(|(t, p)| {
        let text: std::collections::BTreeMap<String, String> = nextsycl_image::Picture::png_text(&p).ok()?.into_iter().collect();
        let (w, h) = text.get("size")?.split_once('x')?;
        let num = |k: &str| text.get(k).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        Some(json!({"url": format!("/v1/images/files/{}", p.file_name()?.to_string_lossy()), "prompt": text.get("prompt")?,
                    "seed": num("seed"), "steps": num("steps"), "width": w.parse::<u32>().ok()?, "height": h.parse::<u32>().ok()?,
                    "loras": text.get("loras").map(|l| l.split(',').filter(|x| !x.is_empty()).collect::<Vec<_>>()).unwrap_or_default(),
                    "seconds": text.get("seconds").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0),
                    "pictures": num("pictures"),
                    "wh": text.get("wh").and_then(|v| v.parse::<f64>().ok()),
                    "created": t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)}))
    }).collect()
}

/// The most pictures an edit takes (each is a block of the denoiser's prefix: memory and time grow with them)
const MAX_PICTURES: usize = 8;
/// pages one OCR request reads at most (each a global view of 273 tokens)
const MAX_PAGES: usize = 24;
/// The most pixels a picture sent may have (it is resized to the model's area, about a megapixel, anyway)
pub const MAX_PICTURE_PIXELS: u64 = 40_000_000;

/// A picture sent, checked against the pixel limit
pub(crate) fn picture(bytes: &[u8], what: &str) -> Result<nextsycl_image::Picture, String> {
    let p = nextsycl_image::Picture::decode(bytes).map_err(|e| format!("{what}: {e}"))?;
    let px = p.width as u64 * p.height as u64;
    if px > MAX_PICTURE_PIXELS {
        return Err(format!("{what}: {}x{} is {:.1} MP; at most {} MP (it is resized to about 1 MP anyway)", p.width, p.height, px as f64 / 1e6,
                           MAX_PICTURE_PIXELS / 1_000_000));
    }
    Ok(p)
}

/// image.generate's init picture (base64 or a data URL)
fn init_picture(s: &str) -> Result<nextsycl_image::Picture, String> {
    picture(&http::unbase64(s).map_err(|e| format!("image: {e}"))?, "image")
}

/// a PDF's pages are rendered at this (a letter page: 1275 x 1650)
const PDF_DPI: f32 = 150.0;

/// An edit request's fields (as the JSON body a generation takes) and its pictures: OpenAI's multipart form (image /
/// image[] files, the other fields as text; `options` and `loras` as JSON text), or JSON with `images` (or `image`):
/// base64 / data URLs
fn edit_body(req: &http::Request) -> Result<(Value, Vec<nextsycl_image::Picture>), String> {
    let ct = http::header(&req.head, "content-type").unwrap_or_default();
    let mut pics = Vec::new();
    let body = if ct.to_ascii_lowercase().starts_with("multipart/form-data") {
        let boundary = ct.split(';').map(str::trim).find_map(|a| a.strip_prefix("boundary=")).ok_or("multipart without a boundary")?.trim_matches('"');
        let mut b = serde_json::Map::new();
        for p in http::multipart(&req.body, boundary)? {
            let name = p.name.trim_end_matches("[]");
            if name == "mask" {
                if !p.data.is_empty() {
                    return Err("masks are not part of this model's edits (describe the change instead)".into());
                }
                continue;
            }
            if name == "image" || name.starts_with("image_") {
                pics.push(picture(&p.data, p.filename.as_deref().unwrap_or(name))?);
                continue;
            }
            let text = String::from_utf8_lossy(&p.data).to_string();
            let v = match name {
                "options" | "loras" => serde_json::from_str(&text).map_err(|e| format!("{name}: {e}"))?,
                "prompt" | "negative_prompt" => Value::String(text),
                _ => match (text.trim().parse::<u64>(), text.trim().parse::<f64>()) {
                    (Ok(i), _) => json!(i),
                    (_, Ok(x)) => json!(x),
                    _ => Value::String(text),
                },
            };
            b.insert(name.to_string(), v);
        }
        Value::Object(b)
    } else {
        let v: Value = serde_json::from_slice(&req.body).map_err(|e| format!("the body is not JSON or a multipart form: {e}"))?;
        let list: Vec<&str> = match (&v["images"], &v["image"]) {
            (Value::Array(a), _) => a.iter().filter_map(Value::as_str).collect(),
            (_, Value::String(s)) => vec![s.as_str()],
            (_, Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        for (i, s) in list.iter().enumerate() {
            let bytes = http::unbase64(s).map_err(|e| format!("images[{i}]: {e}"))?;
            pics.push(picture(&bytes, &format!("images[{i}]"))?);
        }
        v
    };
    if pics.is_empty() {
        return Err("an edit needs a picture (image, image[] or images)".into());
    }
    if pics.len() > MAX_PICTURES {
        return Err(format!("{} pictures: at most {MAX_PICTURES}", pics.len()));
    }
    Ok((body, pics))
}

impl ImageServer {
    /// The server, its engine loaded first with `loras` merged
    #[allow(clippy::too_many_arguments)]
    pub fn new(model: String, gpu: Arc<Gpu>, load: Loader, loras: Vec<LoraUse>, loras_known: Vec<KnownLora>, out_dir: Option<PathBuf>,
               wfe: Option<PathBuf>, cors: Vec<String>) -> Result<ImageServer, String> {
        let e = load(&loras, &mut |l| eprintln!("{l}"))?;
        if let Some(d) = &out_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        if let Some(w) = &wfe {
            if !w.join("image/index.html").is_file() {
                return Err(format!("{}: no web front end built there (./build.sh wfe)", w.display()));
            }
        }
        let tele = Telemetry::start(std::slice::from_ref(&gpu.pci));
        let history = Mutex::new(out_dir.as_deref().map(earlier).unwrap_or_default());
        Ok(ImageServer { model, loras_known, out_dir, wfe, cors, gpu, tele, load, engine: Mutex::new(Some((e, loras))), progress: Mutex::new(json!({"busy": false})),
                         idle: None, waiting: AtomicUsize::new(0), history, cancel: Default::default() })
    }

    /// Answers on `addr` until the process ends
    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        if let Some(i) = self.idle {
            let me = self.clone();
            i.watch("nextsycl image", Box::new(move || {
                me.waiting.load(Ordering::Relaxed) > 0 || me.progress.lock().map(|p| p["busy"] == true).unwrap_or(true)
            }));
        }
        eprintln!("nextsycl image: {} on http://{addr}/v1/images/generations{}", self.model,
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
                let h = format!("{cors}Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n");
                http::respond_bytes(&mut s, 204, "text/plain", b"", &h)
            }
            ("GET", "/") | ("GET", "/index.html") if self.wfe.is_some() => self.page(&mut s, &cors),
            ("GET", p) if self.wfe.is_some() && (p.starts_with("/ui/") || p.starts_with("/static/")) => self.asset(&mut s, p, &cors),
            ("GET", "/api/gpu") => http::respond_with(&mut s, 200, &self.gpu_json(), &cors),
            ("GET", "/health") => {
                let loaded = self.engine.try_lock().map(|e| e.is_some()).unwrap_or(true);
                http::respond_with(&mut s, if loaded { 200 } else { 503 }, &json!({"status": if loaded { "ok" } else { "loading" }}), &cors)
            }
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
            ("GET", p) if p.starts_with("/v1/images/files/") => self.file(&mut s, &p["/v1/images/files/".len()..], &cors),
            ("POST", "/v1/images/generations") | ("POST", "/images/generations") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                // an init picture: a plain img2img (the denoise starts from it; it is no edit's reference)
                let mut body = body;
                let mut pics = Vec::new();
                if let Some(i) = ["image", "init_image"].iter().find_map(|k| body[*k].as_str().filter(|v| !v.is_empty())).map(str::to_string) {
                    match init_picture(&i) {
                        Ok(p) => pics.push(p),
                        Err(m) => return http::respond_with(&mut s, 400, &err(m), &cors),
                    }
                    body["reference"] = json!(false);
                }
                match self.generate(&body, &base, pics) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            ("POST", "/v1/images/vectorize") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                match self.vectorize(&body) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            ("POST", "/v1/images/upscale") | ("POST", "/v1/videos/upscale") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                let r = if req.path == "/v1/images/upscale" { self.upscale(&body, &base) } else { self.upscale_video(&body, &base) };
                match r {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            ("POST", "/v1/ocr") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                match self.ocr(&body) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            ("POST", "/v1/images/edits") | ("POST", "/images/edits") => {
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                let r = edit_body(&req).map_err(|m| (400u16, m)).and_then(|(body, pics)| self.generate(&body, &base, pics));
                match r {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            _ => http::respond_with(&mut s, 404, &err(format!("no route {} {}", req.method, req.path)), &cors),
        }
    }

    /// The page: index.html with the stylesheet inlined (H3's server does the same)
    fn page(&self, s: &mut Conn, cors: &str) {
        // a model that reads pictures gets the OCR page
        let reads = self.engine.lock().map(|g| g.as_ref().is_some_and(|(e, _)| e.reads())).unwrap_or(false);
        let dir = self.wfe.as_ref().expect("checked by the route").join(if reads { "ocr" } else { "image" });
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
        let kind = match rel.rsplit('.').next() {
            Some("js") => "text/javascript; charset=utf-8",
            Some("css") => "text/css; charset=utf-8",
            Some("woff2") => "font/woff2",
            Some("svg") => "image/svg+xml",
            Some("png") => "image/png",
            Some("html") => "text/html; charset=utf-8",
            _ => "application/octet-stream",
        };
        match std::fs::read(root.join(rel)) {
            Ok(b) => http::respond_bytes(s, 200, kind, &b, &format!("{cors}Cache-Control: no-cache\r\n")),
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
        let g = self.engine.lock();
        let (arch, d, edits, loaded, samplers, schedules, guidance, opts, reads, prompts) = match g.as_ref().ok().and_then(|g| {
            g.as_ref().map(|(e, l)| (e.arch(), e.defaults(), e.edits(), l.clone(), e.samplers(), e.schedules(), e.guidance(), e.options(), e.reads(),
                                     e.ocr_prompts()))
        }) {
            Some(x) => x,
            None => return json!({"model": self.model, "loading": true}),
        };
        drop(g);
        json!({
            "model": self.model, "arch": arch, "edits": edits, "reads": reads, "guidance": guidance, "max_pictures": MAX_PICTURES,
            "max_pages": MAX_PAGES,
            "prompts": prompts.iter().map(|p| json!({"id": p.id, "prompt": p.prompt, "title": p.title})).collect::<Vec<_>>(),
            "defaults": {"width": d.width, "height": d.height, "steps": d.steps, "cfg": d.cfg, "sampler": d.sampler.name(),
                         "schedule": d.schedule.name(), "shift": d.shift},
            "samplers": samplers.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "schedules": schedules.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "options": opts.iter().filter(|o| o.at != nextsycl_core::At::Load)
                .map(|o| json!({"name": o.name, "value": o.value, "help": o.help})).collect::<Vec<_>>(),
            "loras": self.loras_known.iter().map(|k| json!({"id": k.id, "title": k.title,
                "loaded": loaded.iter().find(|l| l.name == k.id).map(|l| l.scale)})).collect::<Vec<_>>(),
            "saves": self.out_dir.is_some(), "wfe": self.wfe.is_some(),
            "vectorize": self.engine.lock().ok().and_then(|g| g.as_ref().map(|(e, _)| e.vectorizes())).unwrap_or(false),
            "upscale": self.engine.lock().ok().and_then(|g| g.as_ref().map(|(e, _)| e.upscale_variants())).unwrap_or_default().iter()
                .map(|v| json!({"variant": v.id, "about": v.about, "scales": v.scales, "noise": v.noise})).collect::<Vec<_>>(),
        })
    }

    fn file(&self, s: &mut Conn, name: &str, cors: &str) {
        let ok = !name.is_empty() && (name.ends_with(".png") || name.ends_with(".mp4")) && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            && !name.contains("..");
        let kind = if name.ends_with(".mp4") { "video/mp4" } else { "image/png" };
        match (&self.out_dir, ok) {
            (Some(d), true) => match std::fs::read(d.join(name)) {
                Ok(b) => http::respond_bytes(s, 200, kind, &b, &format!("{cors}Cache-Control: max-age=31536000, immutable\r\n")),
                Err(_) => http::respond_with(s, 404, &err(format!("no picture {name}")), cors),
            },
            _ => http::respond_with(s, 404, &err(format!("no picture {name}")), cors),
        }
    }

    /// The request's LoRAs: names (registered ids or files) and scales
    fn loras_of(&self, body: &Value) -> Result<Option<Vec<LoraUse>>, String> {
        let Some(list) = body.get("loras") else { return Ok(None) };
        let find = |n: &str| -> Option<PathBuf> {
            self.loras_known.iter().find(|k| k.id == n).map(|k| k.path.clone()).or_else(|| std::path::Path::new(n).is_file().then(|| PathBuf::from(n)))
        };
        let mut out = Vec::new();
        for l in list.as_array().ok_or("loras: a list of \"name:scale\" or {name, scale}")? {
            let u = match l {
                Value::String(t) => LoraUse::parse(t, &find)?,
                Value::Object(_) => {
                    let n = l["name"].as_str().ok_or("loras: {name, scale}")?;
                    let mut u = LoraUse::parse(n, &find)?;
                    u.scale = l["scale"].as_f64().unwrap_or(1.0) as f32;
                    u
                }
                _ => return Err("loras: a list of \"name:scale\" or {name, scale}".into()),
            };
            if u.scale != 0.0 {
                out.push(u);
            }
        }
        Ok(Some(out))
    }

    /// A reading (an OCR model): the pages decoded, read one request at a time, the text so far in the progress
    fn ocr(&self, b: &Value) -> Result<Value, (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let list: Vec<&str> = match (&b["images"], &b["image"]) {
            (Value::Array(a), _) => a.iter().filter_map(Value::as_str).collect(),
            (_, Value::String(s)) => vec![s.as_str()],
            _ => Vec::new(),
        };
        if list.is_empty() {
            return Err(bad("images: one or more pages (base64 or data: URLs)".into()));
        }
        if list.len() > MAX_PAGES {
            return Err(bad(format!("{} pages: at most {MAX_PAGES} a request", list.len())));
        }
        let mut pages = Vec::with_capacity(list.len());
        for (i, s) in list.iter().enumerate() {
            let bytes = http::unbase64(s).map_err(|e| bad(format!("images[{i}]: {e}")))?;
            if nextsycl_image::Picture::is_pdf(&bytes) {
                // a PDF: its pages, rendered here (150 dpi), as many as a request takes
                let room = MAX_PAGES.saturating_sub(pages.len());
                let (got, total) = nextsycl_image::Picture::from_pdf(&bytes, PDF_DPI, room).map_err(|e| bad(format!("images[{i}]: {e}")))?;
                if total > got.len() {
                    return Err(bad(format!("images[{i}]: a PDF of {total} pages - at most {MAX_PAGES} pages a request")));
                }
                pages.extend(got);
                continue;
            }
            pages.push(nextsycl_image::Picture::decode(&bytes).map_err(|e| bad(format!("images[{i}]: {e}")))?);
        }
        if pages.len() > MAX_PAGES {
            return Err(bad(format!("{} pages: at most {MAX_PAGES} a request", pages.len())));
        }
        let prompt = b["prompt"].as_str().unwrap_or("").to_string();
        let req = nextsycl_image::OcrRequest {
            pages,
            prompt: prompt.clone(),
            tiles: b["tiles"].as_bool(),
            max_tokens: b["max_tokens"].as_u64().unwrap_or(0) as u32,
            ngram: b["ngram"].as_u64().map(|v| v as u32),
            ngram_window: b["ngram_window"].as_u64().map(|v| v as u32),
            temperature: b["temperature"].as_f64().unwrap_or(0.0) as f32,
            seed: b["seed"].as_u64().unwrap_or(0),
            figures: b["figures"].as_bool(),
            each_page: b["each_page"].as_bool(),
            columns: b["columns"].as_bool(),
            extra: Default::default(),
        };
        let n_pages = req.pages.len();
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let (engine, _) = g.as_ref().ok_or((503, "the engine is not loaded".to_string()))?;
        if !engine.reads() {
            return Err(bad(format!("{} makes pictures; it does not read them (an OCR model does)", self.model)));
        }
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        let p = &self.progress;
        *p.lock().unwrap() = json!({"busy": true, "reading": true, "pages": n_pages, "prompt": prompt, "text": "", "tokens": 0, "seconds": 0});
        let mut text = String::new();
        let mut pieces = 0u64;
        let out = engine.ocr(&req, &mut |t| {
            text.push_str(t);
            pieces += 1;
            // the page polls this: the text so far, about a token a piece
            if pieces.is_multiple_of(4) {
                *p.lock().unwrap() = json!({"busy": true, "reading": true, "pages": n_pages, "prompt": prompt, "text": text, "tokens": pieces,
                                            "seconds": t0.elapsed().as_secs_f64()});
            }
        });
        *p.lock().unwrap() = json!({"busy": false});
        let out = out.map_err(|e| (400u16, e.0))?;
        let seconds = t0.elapsed().as_secs_f64();
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let tok_s = out.tokens as f64 / out.seconds.max(1e-9);
        let read_tok_s = out.prompt_tokens as f64 / out.prompt_seconds.max(1e-9);
        let mut h = self.history.lock().unwrap();
        h.push_front(json!({"kind": "ocr", "prompt": out.prompt, "pages": n_pages, "text": out.text, "tokens": out.tokens, "prompt_tokens": out.prompt_tokens,
                            "seconds": seconds, "tok_s": tok_s, "wh": wh, "created": now()}));
        h.truncate(50);
        Ok(json!({"created": now(), "model": self.model, "text": out.text, "prompt": out.prompt, "pages": n_pages, "tokens": out.tokens, "prompt_tokens": out.prompt_tokens,
                  "prompt_seconds": out.prompt_seconds, "read_tok_s": read_tok_s, "write_seconds": out.seconds, "tok_s": tok_s, "seconds": seconds, "wh": wh}))
    }

    /// POST /v1/images/vectorize: {image (base64 / data URL), max_length, beams, sample, top_p, temperature,
    /// repetition_penalty, seed} - the picture written as SVG by a vectorizer (StarVector)
    fn vectorize(&self, b: &Value) -> Result<Value, (u16, String)> {
        let bad = |m: String| (400u16, m);
        let raw = b["image"].as_str().filter(|v| !v.is_empty()).ok_or_else(|| bad("image: the picture (base64 or a data: URL) is required".into()))?;
        let pic = init_picture(raw).map_err(bad)?;
        let int = |k: &str| -> Result<Option<i64>, (u16, String)> {
            match &b[k] {
                Value::Null => Ok(None),
                v => v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).map(Some).ok_or_else(|| bad(format!("{k}: a whole number"))),
            }
        };
        let num = |k: &str| -> Result<Option<f32>, (u16, String)> {
            match &b[k] {
                Value::Null => Ok(None),
                v => v.as_f64().map(|x| Some(x as f32)).ok_or_else(|| bad(format!("{k}: a number"))),
            }
        };
        let req = nextsycl_image::VectorizeRequest {
            picture: pic,
            max_length: int("max_length")?.map(|v| v.clamp(1, 8192) as u32).unwrap_or(0),
            beams: int("beams")?.map(|v| v.clamp(1, 64) as u32),
            sample: b["sample"].as_bool(),
            top_p: num("top_p")?,
            temperature: num("temperature")?,
            repetition_penalty: num("repetition_penalty")?,
            seed: int("seed")?.map(|v| v as u64).unwrap_or_else(|| now() ^ 0x5f37),
        };
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let (engine, _) = match g.as_ref() {
            None => return Err((503, "the engine is not loaded".into())),
            Some((e, _)) if !e.vectorizes() => return Err((400, format!("{} does not write pictures as SVG (starvector-1b-im2svg does)", self.model))),
            Some(x) => x,
        };
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        let j0 = self.tele.joules();
        let t0 = Instant::now();
        let p = &self.progress;
        let what = format!("vectorize {}x{}", req.picture.width, req.picture.height);
        let mut n = 0u32;
        let out = engine.vectorize(&req, &mut |_piece| {
            n += 1;
            *p.lock().unwrap() = json!({"busy": true, "prompt": what, "at": n, "of": req.max_length.max(n), "phase": "tokens", "seconds": t0.elapsed().as_secs_f64()});
            if self.cancel.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        });
        *p.lock().unwrap() = json!({"busy": false});
        let out = out.map_err(|e| (if e.0 == "cancelled" { 499u16 } else { 400 }, e.0))?;
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let tok_s = out.tokens as f64 / out.seconds.max(1e-9);
        {
            let mut h = self.history.lock().unwrap();
            h.push_front(json!({"kind": "vectorize", "prompt": format!("{} tokens{}", out.tokens, if out.closed { "" } else { ", cut at max_length" }),
                                "seconds": out.prompt_seconds + out.seconds, "wh": wh, "created": now()}));
            h.truncate(200);
        }
        Ok(json!({"created": now(), "data": [{"svg": out.svg, "closed": out.closed, "tokens": out.tokens}],
                  "nextsycl": {"seconds": out.prompt_seconds + out.seconds, "prompt_seconds": out.prompt_seconds, "tokens": out.tokens, "tok_s": tok_s, "wh": wh}}))
    }

    /// An upscale request's settings (image.upscale and video.upscale share them)
    fn upscale_req(b: &Value, picture: nextsycl_image::Picture) -> Result<nextsycl_image::UpscaleRequest, (u16, String)> {
        let bad = |m: String| (400u16, m);
        let int = |k: &str| -> Result<Option<i64>, (u16, String)> {
            match &b[k] {
                Value::Null => Ok(None),
                v => v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).map(Some).ok_or_else(|| bad(format!("{k}: a whole number"))),
            }
        };
        Ok(nextsycl_image::UpscaleRequest {
            picture,
            variant: b["variant"].as_str().filter(|v| !v.is_empty()).map(str::to_string),
            scale: int("scale")?.map(|v| v.clamp(1, 32) as u32),
            noise: int("noise")?.map(|v| v as i32),
            tile: int("tile")?.map(|v| v.clamp(32, 4096) as u32),
        })
    }

    /// The engine for an upscale, one request at a time (as a generation)
    fn upscaler(&self, b: &Value) -> Result<std::sync::MutexGuard<'_, Option<(Box<dyn ImageEngine>, Vec<LoraUse>)>>, (u16, String)> {
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model) {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        match g.as_ref() {
            None => Err((503, "the engine is not loaded".into())),
            Some((e, _)) if e.upscale_variants().is_empty() => Err((400, format!("{} makes pictures; it does not upscale them (real-esrgan, waifu2x do)", self.model))),
            _ => Ok(g),
        }
    }

    /// POST /v1/images/upscale: {image (base64 / data URL), variant, scale, noise, tile, response_format} (POST /v1/images/vectorize: see `vectorize`)
    fn upscale(&self, b: &Value, base: &str) -> Result<Value, (u16, String)> {
        let bad = |m: String| (400u16, m);
        let raw = b["image"].as_str().filter(|v| !v.is_empty()).ok_or_else(|| bad("image: the picture (base64 or a data: URL) is required".into()))?;
        let pic = init_picture(raw).map_err(bad)?;
        let req = Self::upscale_req(b, pic)?;
        let url = b["response_format"].as_str() == Some("url");
        if url && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        let g = self.upscaler(b)?;
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        let (engine, _) = g.as_ref().expect("checked");
        let j0 = self.tele.joules();
        let p = &self.progress;
        let what = format!("upscale {}x{}", req.picture.width, req.picture.height);
        let out = engine.upscale(&req, &mut |st| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": what, "at": st.at, "of": st.of, "phase": "tiles", "seconds": st.seconds});
            if self.cancel.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        });
        *p.lock().unwrap() = json!({"busy": false});
        let (pic, info) = out.map_err(|e| (if e.0 == "cancelled" { 499u16 } else { 400 }, e.0))?;
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let meta = [("model", self.model.clone()), ("variant", info.variant.clone()), ("scale", info.scale.to_string()),
                    ("noise", info.noise.map(|n| n.to_string()).unwrap_or_default()), ("size", format!("{}x{}", pic.width, pic.height)),
                    ("seconds", format!("{:.2}", info.seconds))];
        let png = pic.png_bytes(&meta).map_err(|e| (500u16, e))?;
        let mut item = json!({"width": pic.width, "height": pic.height, "variant": info.variant, "scale": info.scale, "noise": info.noise});
        if let Some(dir) = &self.out_dir {
            let name = format!("{}-{}-x{}.png", now(), self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"), info.scale);
            std::fs::write(dir.join(&name), &png).map_err(|e| (500u16, format!("{}: {e}", dir.display())))?;
            let mut h = self.history.lock().unwrap();
            h.push_front(json!({"kind": "upscale", "url": format!("/v1/images/files/{name}"), "prompt": format!("{} {}x", info.variant, info.scale),
                                "width": pic.width, "height": pic.height, "seconds": info.seconds, "wh": wh, "created": now()}));
            h.truncate(200);
            if url {
                item["url"] = json!(format!("{base}/v1/images/files/{name}"));
            }
        }
        if !url {
            item["b64_json"] = json!(http::base64(&png));
        }
        Ok(json!({"created": now(), "data": [item], "nextsycl": {"seconds": info.seconds, "tiles": info.tiles, "tile": info.tile, "wh": wh}}))
    }

    /// POST /v1/videos/upscale: {video (base64 / data URL) | path (a clip on this machine: the front door's `file`),
    /// variant, scale, noise, tile, crf} - every frame through the upscaler (ffmpeg decodes and encodes, the audio
    /// copied as it was), an MP4 back as a link (the server keeps it) or base64
    fn upscale_video(&self, b: &Value, base: &str) -> Result<Value, (u16, String)> {
        use std::io::{Read, Write};
        use std::process::{Command, Stdio};
        let bad = |m: String| (400u16, m);
        let dir = self.out_dir.clone().ok_or_else(|| bad("video.upscale needs a server that keeps files (start it with an output directory)".into()))?;
        let stamp = now();
        let src = match (b["video"].as_str().filter(|v| !v.is_empty()), b["path"].as_str().filter(|v| !v.is_empty())) {
            (Some(v), _) => {
                let bytes = http::unbase64(v).map_err(|e| bad(format!("video: {e}")))?;
                let f = std::env::temp_dir().join(format!("nc-upscale-{stamp}-{}.in", std::process::id()));
                std::fs::write(&f, bytes).map_err(|e| (500u16, format!("{}: {e}", f.display())))?;
                (f, true)
            }
            (None, Some(p)) if std::path::Path::new(p).is_file() => (PathBuf::from(p), false),
            (None, Some(p)) => return Err((404, format!("{p}: no such clip"))),
            _ => return Err(bad("video: the clip (base64 or a data: URL), or file (a clip made here), is required".into())),
        };
        let cleanup = |f: &PathBuf, tmp: bool| {
            if tmp {
                let _ = std::fs::remove_file(f);
            }
        };
        let probe = Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                                 "stream=width,height,avg_frame_rate,nb_read_packets", "-of", "json"]).arg(&src.0).output();
        let info: Value = probe.ok().and_then(|o| serde_json::from_slice(&o.stdout).ok()).unwrap_or(Value::Null);
        let st = &info["streams"][0];
        let (w, h) = (st["width"].as_u64().unwrap_or(0) as usize, st["height"].as_u64().unwrap_or(0) as usize);
        if w == 0 || h == 0 {
            cleanup(&src.0, src.1);
            return Err(bad("video: not a clip ffprobe reads (no video stream)".into()));
        }
        let fps = st["avg_frame_rate"].as_str().unwrap_or("24/1").to_string();
        let frames: u32 = st["nb_read_packets"].as_str().and_then(|v| v.parse().ok()).unwrap_or(0);
        let probe_pic = nextsycl_image::Picture { width: w as u32, height: h as u32, channels: 3, data: Vec::new() };
        let req0 = Self::upscale_req(b, probe_pic)?;
        let g = self.upscaler(b)?;
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            cleanup(&src.0, src.1);
            return Err((499, "cancelled".into()));
        };
        let (engine, _) = g.as_ref().expect("checked");
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        let name = format!("{stamp}-{}-upscaled.mp4", self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"));
        let out_path = dir.join(&name);
        let crf = b["crf"].as_u64().unwrap_or(16).min(40).to_string();
        let mut dec = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&src.0).args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped()).stderr(Stdio::null()).spawn().map_err(|e| (500u16, format!("ffmpeg: {e}")))?;
        let mut enc: Option<std::process::Child> = None;
        let mut out_size = (0usize, 0usize);
        let mut info_last = None;
        let mut n = 0u32;
        let mut buf = vec![0u8; w * h * 3];
        let p = &self.progress;
        let r: Result<(), (u16, String)> = (|| {
            let mut stdout = dec.stdout.take().ok_or((500u16, "ffmpeg's output".to_string()))?;
            loop {
                if stdout.read_exact(&mut buf).is_err() {
                    break;
                }
                let mut req = req0.clone();
                req.picture = nextsycl_image::Picture { width: w as u32, height: h as u32, channels: 3, data: buf.clone() };
                let (pic, info) = engine.upscale(&req, &mut |_| if self.cancel.stopping() { Err(nextsycl_core::Error("cancelled".into())) } else { Ok(()) })
                    .map_err(|e| (if e.0 == "cancelled" { 499u16 } else { 400 }, e.0))?;
                if enc.is_none() {
                    out_size = (pic.width as usize, pic.height as usize);
                    // the source's audio, as it was; H.264 4:2:0 for any player
                    enc = Some(Command::new("ffmpeg").args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{}x{}", pic.width, pic.height),
                                                       "-r", &fps, "-i", "-", "-i"]).arg(&src.0)
                        .args(["-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", &crf, "-pix_fmt", "yuv420p", "-c:a", "copy",
                               "-movflags", "+faststart"]).arg(&out_path)
                        .stdin(Stdio::piped()).stderr(Stdio::null()).spawn().map_err(|e| (500u16, format!("ffmpeg: {e}")))?);
                }
                enc.as_mut().and_then(|e| e.stdin.as_mut()).ok_or((500u16, "ffmpeg's input".to_string()))?
                    .write_all(&pic.data).map_err(|e| (500u16, format!("the encoder: {e}")))?;
                n += 1;
                info_last = Some(info);
                *p.lock().unwrap() = json!({"busy": true, "prompt": format!("upscale a clip {w}x{h}"), "at": n, "of": frames.max(n), "phase": "frames",
                                            "seconds": t0.elapsed().as_secs_f64()});
            }
            Ok(())
        })();
        *p.lock().unwrap() = json!({"busy": false});
        let _ = dec.kill();
        let _ = dec.wait();
        let enc_ok = match enc {
            Some(mut e) => {
                drop(e.stdin.take());
                e.wait().map(|s| s.success()).unwrap_or(false)
            }
            None => false,
        };
        cleanup(&src.0, src.1);
        if let Err(e) = r {
            let _ = std::fs::remove_file(&out_path);
            return Err(e);
        }
        if n == 0 || !enc_ok {
            let _ = std::fs::remove_file(&out_path);
            return Err((500, if n == 0 { "the clip had no frames ffmpeg could decode".into() } else { "the encoder failed (ffmpeg)".into() }));
        }
        let seconds = t0.elapsed().as_secs_f64();
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let info = info_last.unwrap_or_default();
        let mut item = json!({"width": out_size.0, "height": out_size.1, "frames": n, "fps": fps, "variant": info.variant, "scale": info.scale, "noise": info.noise,
                              "name": name, "url": format!("{base}/v1/images/files/{name}")});
        if b["response_format"].as_str() == Some("b64_json") {
            let bytes = std::fs::read(&out_path).map_err(|e| (500u16, format!("{}: {e}", out_path.display())))?;
            item["b64_json"] = json!(http::base64(&bytes));
        }
        let mut hist = self.history.lock().unwrap();
        hist.push_front(json!({"kind": "upscale-video", "url": format!("/v1/images/files/{name}"), "prompt": format!("{} {}x, {n} frames", info.variant, info.scale),
                               "width": out_size.0, "height": out_size.1, "seconds": seconds, "wh": wh, "created": now()}));
        hist.truncate(200);
        Ok(json!({"created": now(), "data": [item], "nextsycl": {"seconds": seconds, "frames": n, "fps_done": n as f64 / seconds.max(1e-9), "wh": wh}}))
    }

    fn generate(&self, b: &Value, base: &str, pics: Vec<nextsycl_image::Picture>) -> Result<Value, (u16, String)> {
        let bad = |m: String| (400u16, m);
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty() && *m != self.model && *m != "dall-e-3" && *m != "dall-e-2" && *m != "gpt-image-1") {
            return Err((404, format!("model {m} is not served here (this server: {})", self.model)));
        }
        let prompt = b["prompt"].as_str().filter(|p| !p.trim().is_empty()).ok_or_else(|| bad("prompt: required".into()))?.to_string();
        let (width, height) = match b["size"].as_str() {
            None | Some("auto") => (None, None),
            Some(sz) => {
                let (w, h) = sz.split_once('x').ok_or_else(|| bad(format!("size {sz}: WIDTHxHEIGHT")))?;
                (Some(w.parse::<u32>().map_err(|_| bad(format!("size {sz}")))?), Some(h.parse::<u32>().map_err(|_| bad(format!("size {sz}")))?))
            }
        };
        let num = |k: &[&str]| k.iter().find_map(|k| b[*k].as_f64());
        let n = num(&["n"]).unwrap_or(1.0) as u32;
        if !(1..=8).contains(&n) {
            return Err(bad("n: 1 to 8".into()));
        }
        let seed = match &b["seed"] {
            Value::Number(x) => x.as_u64().ok_or_else(|| bad("seed: a whole number".into()))?,
            _ => SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 % 1_000_000).unwrap_or(7),
        };
        let parse_name = |k: &str| b[k].as_str().filter(|v| !v.is_empty() && *v != "default");
        let sampler = parse_name("sampler").map(|v| Sampler::parse(v).ok_or_else(|| bad(format!("sampler {v}?")))).transpose()?;
        let schedule = parse_name("schedule").map(|v| Schedule::parse(v).ok_or_else(|| bad(format!("schedule {v}?")))).transpose()?;
        let want_loras = self.loras_of(b).map_err(bad)?;
        // an edit's strength: how far from the first picture (1: as far as a new one; lower starts from it)
        let strength = match &b["strength"] {
            Value::Null => 1.0,
            v => v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(|| bad("strength: a number from 0.05 to 1".into()))?,
        };
        if !(0.05..=1.0).contains(&strength) {
            return Err(bad(format!("strength {strength}: from 0.05 to 1")));
        }
        // the pictures are the edit's references too, unless reference is false (a plain img2img)
        let condition = b["reference"] != false;
        if strength < 1.0 && pics.is_empty() {
            return Err(bad("strength: needs a picture to start from (image.generate's image, or image.edit's first picture)".into()));
        }
        if !condition && strength >= 1.0 {
            return Err(bad("an init picture without its use as a reference needs strength below 1 (at 1 the picture would be ignored)".into()));
        }
        let req = ImageRequest {
            prompt: prompt.clone(),
            negative: b["negative_prompt"].as_str().filter(|v| !v.trim().is_empty()).map(str::to_string),
            width,
            height,
            steps: num(&["steps", "num_inference_steps"]).map(|v| v as u32),
            cfg: num(&["cfg", "guidance_scale", "true_cfg_scale"]).map(|v| v as f32),
            seed,
            n,
            sampler,
            schedule,
            shift: num(&["shift"]).map(|v| v as f32),
            loras: Vec::new(),
            edit: (!pics.is_empty()).then(|| nextsycl_image::Edit { image: pics[0].clone(), refs: pics[1..].to_vec(), mask: None, strength: strength as f32, condition }),
            rgba: b["background"].as_str() == Some("transparent") || b["rgba"].as_bool() == Some(true),
            extra: Default::default(),
        };
        // the engine's own options for this request ("options": {NAME: value}), checked against what it declares
        let given: nextsycl_core::options::Given = b["options"].as_object().into_iter().flatten()
            .map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
        let url = b["response_format"].as_str() == Some("url");
        if url && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        if req.edit.is_some() {
            let edits = self.engine.lock().map(|g| g.as_ref().is_some_and(|(e, _)| e.edits())).unwrap_or(false);
            if !edits {
                return Err(bad(format!("{} does not take pictures (edits)", self.model)));
            }
        }
        let n_pics = pics.len();
        // one request at a time
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let mut g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let Some(_running) = self.cancel.begin(b["request_id"].as_str()) else {
            return Err((499, "cancelled".into()));
        };
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        // other LoRAs: beside the weights for this request when the engine takes them so (those merged at load kept, the
        // rest on its side path - no reload), else the engine reloaded with them merged
        let mut side: Vec<LoraUse> = Vec::new();
        if let Some(want) = &want_loras {
            let have = g.as_ref().map(|(_, h)| h.clone()).unwrap_or_default();
            let takes = g.as_ref().is_some_and(|(e, _)| e.side_loras());
            let kept = have.iter().all(|h| want.contains(h));
            if have != *want && takes && kept {
                side = want.iter().filter(|w| !have.contains(w)).cloned().collect();
            } else if have != *want {
                *self.progress.lock().unwrap() = json!({"busy": true, "loading": true, "prompt": prompt, "seconds": 0});
                *g = None; // its VRAM first
                match (self.load)(want, &mut |l| eprintln!("{l}")) {
                    Ok(e) => *g = Some((e, want.clone())),
                    Err(e) => {
                        // back to the plain model, so the server keeps serving
                        *g = (self.load)(&[], &mut |l| eprintln!("{l}")).ok().map(|e| (e, Vec::new()));
                        *self.progress.lock().unwrap() = json!({"busy": false});
                        return Err((500, format!("loading with those LoRAs: {e}")));
                    }
                }
            }
        }
        let (engine, loras) = g.as_ref().ok_or((503, "the engine is not loaded (a reload failed)".to_string()))?;
        let mut req = req;
        req.loras = side.clone();
        if !given.is_empty() {
            nextsycl_core::options::resolve(&given, engine.options(), nextsycl_core::At::Request, &self.model).map_err(bad)?;
            req.extra = nextsycl_core::options::by_name(&given, engine.options());
        }
        // a side LoRA's own settings an option takes per request (its sigma preset), unless the request gives them
        for l in &side {
            for (var, v) in self.loras_known.iter().filter(|k| k.id == l.name).flat_map(|k| k.env.iter()) {
                if let Some(o) = engine.options().iter().find(|o| o.env == var && o.at != nextsycl_core::At::Load) {
                    req.extra.entry(o.name.to_string()).or_insert_with(|| v.clone());
                }
            }
        }
        let steps = engine.steps_for(&req);
        let p = &self.progress;
        let pics = engine.generate(&req, &mut |st| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "picture": st.picture + 1, "pictures": n, "at": st.at, "of": st.of,
                                        "seconds": st.seconds});
            if self.cancel.stopping() {
                return Err(nextsycl_core::Error("cancelled".into()));
            }
            Ok(())
        });
        *p.lock().unwrap() = json!({"busy": false});
        let pics = pics.map_err(|e| (if e.0 == "cancelled" { 499u16 } else { 400 }, e.0))?;
        let seconds = t0.elapsed().as_secs_f64();
        // the card's energy over the request (its counter's rise), watt-hours
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let lora_txt: Vec<String> = loras.iter().chain(&side).map(|l| format!("{}:{}", l.name, l.scale)).collect();
        let mut data = Vec::new();
        for (i, pic) in pics.iter().enumerate() {
            let s = seed + i as u64;
            let meta = [("prompt", prompt.clone()), ("model", self.model.clone()), ("seed", s.to_string()),
                        ("size", format!("{}x{}", pic.width, pic.height)), ("steps", steps.to_string()), ("loras", lora_txt.join(",")),
                        ("seconds", format!("{seconds:.2}")), ("wh", wh.map(|w| format!("{:.3}", w / pics.len() as f64)).unwrap_or_default()),
                        ("pictures", if n_pics > 0 { n_pics.to_string() } else { String::new() })];
            let png = pic.png_bytes(&meta).map_err(|e| (500u16, e))?;
            let mut item = json!({"revised_prompt": prompt, "seed": s, "width": pic.width, "height": pic.height});
            if let Some(dir) = &self.out_dir {
                let name = format!("{}-{}-{s}.png", now(), self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"));
                std::fs::write(dir.join(&name), &png).map_err(|e| (500u16, format!("{}: {e}", dir.display())))?;
                let link = format!("{base}/v1/images/files/{name}");
                let mut h = self.history.lock().unwrap();
                h.push_front(json!({"url": format!("/v1/images/files/{name}"), "prompt": prompt, "seed": s, "steps": steps,
                                    "width": pic.width, "height": pic.height, "loras": lora_txt, "seconds": seconds,
                                    "wh": wh.map(|w| w / pics.len() as f64), "pictures": n_pics, "created": now()}));
                h.truncate(200);
                if url {
                    item["url"] = json!(link);
                }
            }
            if !url {
                item["b64_json"] = json!(http::base64(&png));
            }
            data.push(item);
        }
        Ok(json!({"created": now(), "data": data, "nextsycl": {"seconds": seconds, "steps": steps, "seed": seed, "loras": lora_txt, "wh": wh}}))
    }
}
