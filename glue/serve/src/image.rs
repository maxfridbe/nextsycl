//! The image server: OpenAI's images API over one image engine (`nextsycl_image::ImageEngine`), and with `wfe` a web
//! page that drives it with every option the engine has.
//!
//! ```text
//!   POST /v1/images/generations   OpenAI's body - prompt, n, size, response_format (b64_json | url), background
//!                                 (transparent: RGBA) - and ours: steps, seed, sampler, schedule, shift, cfg,
//!                                 negative_prompt, loras ["name:scale" | {name, scale}]
//!   GET  /v1/images/files/<f>     a picture made here (saved in the output directory)
//!   GET  /v1/models, /health
//!   GET  /api/info                the model, its defaults, samplers, schedules, the LoRAs it can take
//!   GET  /api/progress            the request running: its picture, step, seconds; requests waiting
//!   GET  /api/history             the pictures made since the server started, newest first
//!   GET  /api/gpu                 the card: VRAM, power, temperatures
//!   GET  /, /ui/..., /static/...  the web front end (--wfe: wfe/image, built into dist/wfe - TSX on snabbdom, H3's
//!                                 scheme; the page's CSS inlined at __CSS__ as H3's server does)
//! ```
//!
//! One request runs at a time (one engine on its GPU); the others wait their turn. A request asking for LoRAs other
//! than those merged in reloads the engine with them (the old one is dropped first: its VRAM is the new one's) - until
//! the engines take a LoRA per request on a side path.

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
                    "wh": text.get("wh").and_then(|v| v.parse::<f64>().ok()),
                    "created": t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)}))
    }).collect()
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
                         waiting: AtomicUsize::new(0), history })
    }

    /// Answers on `addr` until the process ends
    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
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
                http::respond_with(&mut s, 200, &p, &cors)
            }
            ("GET", "/api/history") => http::respond_with(&mut s, 200, &json!({"data": self.history.lock().unwrap().iter().collect::<Vec<_>>()}), &cors),
            ("GET", p) if p.starts_with("/v1/images/files/") => self.file(&mut s, &p["/v1/images/files/".len()..], &cors),
            ("POST", "/v1/images/generations") | ("POST", "/images/generations") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return http::respond_with(&mut s, 400, &err(format!("the body is not JSON: {e}")), &cors),
                };
                let base = req.host.as_deref().map(|h| format!("http://{h}")).unwrap_or_default();
                match self.generate(&body, &base) {
                    Ok(v) => http::respond_with(&mut s, 200, &v, &cors),
                    Err((code, m)) => http::respond_with(&mut s, code, &err(m), &cors),
                }
            }
            ("POST", "/v1/images/edits") => http::respond_with(&mut s, 400, &err("edits are not served yet"), &cors),
            _ => http::respond_with(&mut s, 404, &err(format!("no route {} {}", req.method, req.path)), &cors),
        }
    }

    /// The page: index.html with the stylesheet inlined (H3's server does the same)
    fn page(&self, s: &mut Conn, cors: &str) {
        let dir = self.wfe.as_ref().expect("checked by the route").join("image");
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
        let (arch, d, edits, loaded, samplers, schedules, guidance) = match g.as_ref().ok().and_then(|g| {
            g.as_ref().map(|(e, l)| (e.arch(), e.defaults(), e.edits(), l.clone(), e.samplers(), e.schedules(), e.guidance()))
        }) {
            Some(x) => x,
            None => return json!({"model": self.model, "loading": true}),
        };
        drop(g);
        json!({
            "model": self.model, "arch": arch, "edits": edits, "guidance": guidance,
            "defaults": {"width": d.width, "height": d.height, "steps": d.steps, "cfg": d.cfg, "sampler": d.sampler.name(),
                         "schedule": d.schedule.name(), "shift": d.shift},
            "samplers": samplers.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "schedules": schedules.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "loras": self.loras_known.iter().map(|k| json!({"id": k.id, "title": k.title,
                "loaded": loaded.iter().find(|l| l.name == k.id).map(|l| l.scale)})).collect::<Vec<_>>(),
            "saves": self.out_dir.is_some(),
        })
    }

    fn file(&self, s: &mut Conn, name: &str, cors: &str) {
        let ok = !name.is_empty() && name.ends_with(".png") && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && !name.contains("..");
        match (&self.out_dir, ok) {
            (Some(d), true) => match std::fs::read(d.join(name)) {
                Ok(b) => http::respond_bytes(s, 200, "image/png", &b, &format!("{cors}Cache-Control: max-age=31536000, immutable\r\n")),
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

    fn generate(&self, b: &Value, base: &str) -> Result<Value, (u16, String)> {
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
            edit: None,
            rgba: b["background"].as_str() == Some("transparent") || b["rgba"].as_bool() == Some(true),
        };
        let url = b["response_format"].as_str() == Some("url");
        if url && self.out_dir.is_none() {
            return Err(bad("response_format url: this server keeps no files (start it with an output directory)".into()));
        }
        // one request at a time
        self.waiting.fetch_add(1, Ordering::Relaxed);
        let mut g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let j0 = self.tele.joules();
        // other LoRAs: the engine reloaded with them merged
        if let Some(want) = &want_loras {
            let same = g.as_ref().is_some_and(|(_, have)| have == want);
            if !same {
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
        let d = engine.defaults();
        let steps = req.steps.unwrap_or(d.steps);
        let p = &self.progress;
        let pics = engine.generate(&req, &mut |st| {
            *p.lock().unwrap() = json!({"busy": true, "prompt": prompt, "picture": st.picture + 1, "pictures": n, "at": st.at, "of": st.of,
                                        "seconds": st.seconds});
        });
        *p.lock().unwrap() = json!({"busy": false});
        let pics = pics.map_err(|e| (400u16, e.0))?;
        let seconds = t0.elapsed().as_secs_f64();
        // the card's energy over the request (its counter's rise), watt-hours
        let wh = match (j0, self.tele.joules()) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / 3600.0),
            _ => None,
        };
        let lora_txt: Vec<String> = loras.iter().map(|l| format!("{}:{}", l.name, l.scale)).collect();
        let mut data = Vec::new();
        for (i, pic) in pics.iter().enumerate() {
            let s = seed + i as u64;
            let meta = [("prompt", prompt.clone()), ("model", self.model.clone()), ("seed", s.to_string()),
                        ("size", format!("{}x{}", pic.width, pic.height)), ("steps", steps.to_string()), ("loras", lora_txt.join(",")),
                        ("seconds", format!("{seconds:.2}")), ("wh", wh.map(|w| format!("{:.3}", w / pics.len() as f64)).unwrap_or_default())];
            let png = pic.png_bytes(&meta).map_err(|e| (500u16, e))?;
            let mut item = json!({"revised_prompt": prompt, "seed": s});
            if let Some(dir) = &self.out_dir {
                let name = format!("{}-{}-{s}.png", now(), self.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_"));
                std::fs::write(dir.join(&name), &png).map_err(|e| (500u16, format!("{}: {e}", dir.display())))?;
                let link = format!("{base}/v1/images/files/{name}");
                let mut h = self.history.lock().unwrap();
                h.push_front(json!({"url": format!("/v1/images/files/{name}"), "prompt": prompt, "seed": s, "steps": steps,
                                    "width": pic.width, "height": pic.height, "loras": lora_txt, "seconds": seconds,
                                    "wh": wh.map(|w| w / pics.len() as f64), "created": now()}));
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
