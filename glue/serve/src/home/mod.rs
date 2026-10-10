//! The front door (`nextsycl serve`, :8000): the box at a glance and one API for everything on it.
//!
//! - **The page** (`/`, wfe/home): every GPU - its memory by process, and the memory the enabled models are vouched
//!   for (hatched; several may vouch for one card: whichever loads first holds it) - and every registered model:
//!   enable it on GPU(s), start (load now: the button spins until it answers) and stop (unload), its page, its state;
//!   an API tab with the bearer token and a curl for every method.
//! - **Enabled models** load on their first request (an API call here, its page, Open WebUI through the switcher) and
//!   unload after `idle_minutes` without one (the image and audio servers exit, the switcher sets the chat mode to
//!   none, the video daemon's workers let their card go); they stay enabled. Disabled: unloaded, gone from Open WebUI's
//!   list (the switcher lists enabled chat models) and from the API spec. The registry's `enabled` and `gpus` are the
//!   state, so `nextsycl models enable|disable` and this page agree. Video: enabled models = the daemon's engines.
//! - **The API** (`api.rs`): `POST /rpc/<kind>.<method>` (H3's rules: POST only, fixed routes, the JSON body carries
//!   every variable, no path variables, no query strings; `{"ok": true, "result": ...}` or `{"ok": false, "error":
//!   {"code", "message"}}`) and the OpenAI-compatible `/v1/...`, both behind `Authorization: Bearer <token>`;
//!   `GET /openapi.yml` describes what the enabled kinds offer. `/api/...` is this page's own.
//!
//! A host process (libc only): the cards from sysfs and the DRM clients' fdinfo (gpustat's sampler, one per card), the
//! services over HTTP, the actions by running this same program (`nextsycl image start ...`) or the video studio's
//! llm.mode, which owns the chat card.

mod api;
mod config;
mod life;
pub mod registry;
mod yaml;

use std::collections::BTreeMap;
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::gpustat::{find_cards, Card};
use crate::http::{self, call_for, Conn, Target};

/// Where the services listen (this box's settings)
#[derive(Clone, Debug)]
pub struct Ports {
    pub llm: u16,
    pub switch: u16,
    pub image: u16,
    pub audio: u16,
    pub video: u16,
    /// the chat front end (Open WebUI), a link only
    pub chat_ui: u16,
}

pub struct Options {
    /// dist/wfe
    pub wfe: PathBuf,
    /// this program, for the actions
    pub exe: PathBuf,
    pub ports: Ports,
    /// the address servers started from here listen on (their pages are linked from other machines)
    pub serve_host: String,
    /// the registry and the settings (the program's: home::registry::Store)
    pub cfg: Box<dyn registry::Store>,
    /// unload an enabled model after this many minutes without a request (0: never)
    pub idle_minutes: u64,
}

struct Run {
    id: u64,
    label: String,
    started: f64,
    log: PathBuf,
    /// none while it runs; the exit code (or -1: did not start / an RPC error)
    rc: Option<i32>,
}

pub struct Home {
    o: Options,
    /// the API's bearer token (NS_API_TOKEN, or made once and kept in the state file)
    token: String,
    /// serve.json beside the registry: the token, the memory each model was seen to take per card
    state_file: PathBuf,
    cards: Mutex<Value>,
    runs: Mutex<Vec<Run>>,
    next: Mutex<u64>,
    /// models loading now (from here), and since when
    loading: Mutex<BTreeMap<String, f64>>,
    /// one load or swap at a time per kind
    kind_lock: BTreeMap<&'static str, Mutex<()>>,
    /// model id -> card -> MiB, the most seen while it ran
    seen: Mutex<BTreeMap<String, BTreeMap<usize, u64>>>,
}

pub(crate) fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn err(m: impl Into<String>) -> Value {
    json!({"error": m.into()})
}

/// What a process on a GPU is: a nextsycl command (kind, command, model, port, GPUs from its arguments), or the
/// program's name
fn process(pid: u32) -> Value {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let args: Vec<String> = raw.split(|b| *b == 0).filter(|a| !a.is_empty()).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
    let base = |a: &str| a.rsplit('/').next().unwrap_or(a).to_string();
    let Some(i) = args.iter().position(|a| base(a) == "nextsycl") else {
        let prog = args.first().map(|a| base(a)).unwrap_or_else(|| "?".into());
        // python and friends: the script is the name
        let what = if matches!(prog.as_str(), "python" | "python3" | "node") { args.get(1).map(|a| base(a)).unwrap_or(prog) } else { prog };
        return json!({"pid": pid, "program": what, "args": args.join(" ").chars().take(240).collect::<String>()});
    };
    let rest = &args[i + 1..];
    let get = |k: &str| rest.iter().position(|a| a == k).and_then(|j| rest.get(j + 1)).cloned();
    let gpus: Vec<String> = rest.iter().enumerate().filter(|(_, a)| *a == "--gpu").filter_map(|(j, _)| rest.get(j + 1).cloned()).collect();
    let kinds = ["llm", "image", "video", "audio"];
    let (kind, cmd, model) = match rest.first().map(String::as_str) {
        Some(k) if kinds.contains(&k) => {
            (k.to_string(), rest.get(1).cloned().unwrap_or_default(), rest.get(2).filter(|a| !a.starts_with("--")).map(|a| base(a)))
        }
        // the llm commands' old names (nextsycl serve FILE)
        Some(c) => ("llm".to_string(), c.to_string(), rest.get(1).filter(|a| !a.starts_with("--")).map(|a| base(a))),
        None => (String::new(), String::new(), None),
    };
    json!({"pid": pid, "program": "nextsycl", "kind": kind, "command": cmd, "model": get("--model").map(|m| base(&m)).or(model),
           "port": get("--port"), "gpus": gpus})
}

/// Whether something accepts connections on the port (a page that is not JSON)
fn listening(port: u16) -> bool {
    ("127.0.0.1", port).to_socket_addrs().ok().and_then(|mut a| a.next())
        .is_some_and(|a| TcpStream::connect_timeout(&a, Duration::from_millis(400)).is_ok())
}

fn meminfo() -> Value {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kb = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0);
    let load = std::fs::read_to_string("/proc/loadavg").ok().and_then(|l| l.split_whitespace().next()?.parse::<f64>().ok());
    json!({"mem_total_mb": kb("MemTotal:") / 1024, "mem_avail_mb": kb("MemAvailable:") / 1024, "load1": load,
           "cpus": std::thread::available_parallelism().map_or(1, |n| n.get())})
}

/// 32 random bytes as hex (the kernel's generator)
fn new_token() -> String {
    let mut b = [0u8; 24];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = std::io::Read::read_exact(&mut f, &mut b);
    }
    format!("ns-{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>())
}

impl Home {
    pub fn new(o: Options) -> Result<Arc<Home>, String> {
        let state_file = registry::registry(&o.cfg).with_file_name("serve.json");
        let mut st: Value = std::fs::read(&state_file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
        let token = match o.cfg.get("NS_API_TOKEN").or_else(|| st["token"].as_str().map(str::to_string)) {
            Some(t) => t,
            None => {
                let t = new_token();
                st["token"] = json!(t);
                std::fs::write(&state_file, serde_json::to_string_pretty(&st).unwrap_or_default())
                    .map_err(|e| format!("{}: {e} (the API token is kept there)", state_file.display()))?;
                t
            }
        };
        let seen = st["seen"].as_object().map(|m| m.iter().map(|(id, g)| {
            (id.clone(), g.as_object().map(|g| g.iter().filter_map(|(k, v)| Some((k.parse().ok()?, v.as_u64()?))).collect()).unwrap_or_default())
        }).collect()).unwrap_or_default();
        let h = Arc::new(Home {
            o, token, state_file,
            cards: Mutex::new(json!([])), runs: Mutex::new(Vec::new()), next: Mutex::new(1), loading: Mutex::new(BTreeMap::new()),
            kind_lock: ["llm", "image", "audio", "video"].into_iter().map(|k| (k, Mutex::new(()))).collect(),
            seen: Mutex::new(seen),
        });
        let me = h.clone();
        std::thread::spawn(move || me.sample_cards());
        // the video daemon as the enabled video models say (it is this page's to start and stop)
        let me = h.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(3));
            if let Err(e) = me.video_apply(false) {
                eprintln!("video: {e}");
            }
        });
        Ok(h)
    }

    /// Every card every 2 s (busy % and power are deltas); what each model was seen to take, kept
    fn sample_cards(&self) {
        let mut cards: Vec<Card> = find_cards().iter().map(|p| Card::open(p, None)).collect();
        let mut last_save = 0.0;
        loop {
            let v: Vec<Value> = cards.iter_mut().enumerate().map(|(i, c)| {
                let (mut r, procs) = c.sample(Duration::from_secs(10));
                r["index"] = json!(i);
                // (a process that opened every card holds a few MiB of context on the ones it does not use)
                r["procs"] = procs.iter().filter(|(_, mb)| *mb >= 256).map(|(pid, mb)| {
                    let mut p = process(*pid);
                    p["vram_mb"] = json!(mb);
                    p
                }).collect();
                r
            }).collect();
            *self.cards.lock().unwrap() = Value::from(v);
            // the most each model took on each card, kept for the vouching estimates
            let named = self.named_cards(&self.models());
            let mut changed = false;
            {
                let mut seen = self.seen.lock().unwrap();
                for c in named.as_array().into_iter().flatten() {
                    let gi = c["index"].as_u64().unwrap_or(0) as usize;
                    for p in c["procs"].as_array().into_iter().flatten() {
                        if let (Some(id), Some(mb)) = (p["model_id"].as_str(), p["vram_mb"].as_u64()) {
                            let e = seen.entry(id.to_string()).or_default().entry(gi).or_insert(0);
                            if mb > *e {
                                *e = mb;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if changed && now() - last_save > 30.0 {
                last_save = now();
                self.save_state();
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    fn save_state(&self) {
        let st = json!({"token": self.token, "seen": *self.seen.lock().unwrap()});
        let tmp = self.state_file.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_string_pretty(&st).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.state_file);
        }
    }

    /// The registry (read on every call: `nextsycl models` changes it too)
    fn models(&self) -> Vec<Value> {
        registry::all(&self.o.cfg).unwrap_or_default().into_iter().map(|mut m| {
            m["kind"] = json!(registry::kind_of(&m));
            m
        }).collect()
    }

    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        eprintln!("nextsycl serve on http://{addr}/ ({} GPUs)", find_cards().len());
        for c in l.incoming().flatten() {
            let me = self.clone();
            std::thread::spawn(move || me.handle(Conn::Tcp(c)));
        }
        Ok(())
    }

    fn handle(self: &Arc<Self>, mut s: Conn) {
        let req = match http::read_request(&mut s) {
            Ok(r) => r,
            Err(e) => return http::respond(&mut s, 400, &err(e)),
        };
        let p = req.path.as_str();
        if p.starts_with("/rpc/") || p.starts_with("/v1/") {
            return self.api(s, req);
        }
        match (req.method.as_str(), p) {
            ("GET", "/") | ("GET", "/index.html") => self.page(&mut s),
            ("GET", p) if p.starts_with("/ui/") || p.starts_with("/static/") => self.asset(&mut s, p),
            ("GET", "/health") => http::respond(&mut s, 200, &json!({"status": "ok"})),
            ("GET", "/openapi.yml") | ("GET", "/openapi.yaml") => {
                let spec = self.openapi(&req);
                http::respond_bytes(&mut s, 200, "application/yaml; charset=utf-8", yaml::to_yaml(&spec).as_bytes(), "Cache-Control: no-cache\r\n")
            }
            ("GET", "/openapi.json") => http::respond(&mut s, 200, &self.openapi(&req)),
            ("GET", "/api/state") => http::respond(&mut s, 200, &self.state()),
            ("GET", "/api/config") => http::respond(&mut s, 200, &self.config()),
            ("GET", "/api/spec") => http::respond(&mut s, 200, &json!({"token": self.token, "spec": self.openapi(&req)})),
            ("GET", p) if p.starts_with("/api/runs/") => {
                let id: u64 = p["/api/runs/".len()..].parse().unwrap_or(0);
                match self.run_json(id, 64 * 1024) {
                    Some(v) => http::respond(&mut s, 200, &v),
                    None => http::respond(&mut s, 404, &err(format!("no run {id}"))),
                }
            }
            ("POST", "/api/action") => {
                let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                match self.action(&body) {
                    Ok(v) => http::respond(&mut s, 200, &v),
                    Err((c, m)) => http::respond(&mut s, c, &err(m)),
                }
            }
            _ => http::respond(&mut s, 404, &err(format!("no route {} {}", req.method, req.path))),
        }
    }

    fn page(&self, s: &mut Conn) {
        let dir = self.o.wfe.join("home");
        match (std::fs::read_to_string(dir.join("index.html")), std::fs::read_to_string(dir.join("style.css"))) {
            (Ok(h), Ok(c)) => http::respond_bytes(s, 200, "text/html; charset=utf-8", h.replace("__CSS__", &c).as_bytes(), "Cache-Control: no-cache\r\n"),
            _ => http::respond(s, 500, &err(format!("{}: the page is not built (./build.sh wfe)", dir.display()))),
        }
    }

    fn asset(&self, s: &mut Conn, path: &str) {
        let rel = path.strip_prefix("/ui/").unwrap_or(path.trim_start_matches('/'));
        if rel.split('/').any(|c| c == ".." || c.is_empty()) {
            return http::respond(s, 404, &err("no such file"));
        }
        let kind = match rel.rsplit('.').next() {
            Some("js") => "text/javascript; charset=utf-8",
            Some("css") => "text/css; charset=utf-8",
            Some("woff2") => "font/woff2",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        };
        match std::fs::read(self.o.wfe.join(rel)) {
            Ok(b) => http::respond_bytes(s, 200, kind, &b, "Cache-Control: no-cache\r\n"),
            Err(_) => http::respond(s, 404, &err(format!("no file {rel}"))),
        }
    }

    /// The front ends and helpers beside the models: up or not, a link
    fn services(&self) -> Vec<Value> {
        let p = self.o.ports.clone();
        let probes: Vec<Box<dyn FnOnce() -> Value + Send>> = vec![
            Box::new(move || json!({"id": "chat-ui", "title": "Chat", "what": "the chat front end (Open WebUI)", "port": p.chat_ui,
                                    "up": listening(p.chat_ui), "link": "/"})),
            Box::new(move || {
                let m = call_for(&Target::Tcp(format!("127.0.0.1:{}", p.switch)), "GET", "/v1/models", None, 3).ok();
                let n = m.as_ref().and_then(|m| m["data"].as_array().map(|a| a.iter().filter(|x| x["kind"].is_null()).count()));
                json!({"id": "switch", "title": "Model switch", "what": "one OpenAI endpoint for Open WebUI over the enabled chat models",
                       "port": p.switch, "up": m.is_some(), "detail": n.map(|n| format!("{n} chat models listed")), "api": "/v1"})
            }),
            Box::new(move || {
                let up = call_for(&Target::Tcp(format!("127.0.0.1:{}", p.video)), "POST", "/rpc/status", Some(&json!({})), 3).is_ok();
                json!({"id": "studio", "title": "Video studio", "what": "nextsycl video studio (MiniMax H3): clips, queue, projects", "port": p.video,
                       "up": up, "link": "/"})
            }),
        ];
        let hs: Vec<_> = probes.into_iter().map(std::thread::spawn).collect();
        hs.into_iter().map(|h| h.join().unwrap_or(Value::Null)).collect()
    }

    fn state(&self) -> Value {
        let all = self.models();
        let cards = self.named_cards(&all);
        let models = self.model_states(&all, &cards);
        let runs: Vec<Value> = {
            let ids: Vec<u64> = self.runs.lock().unwrap().iter().rev().take(8).map(|r| r.id).collect();
            ids.into_iter().filter_map(|id| self.run_json(id, 2048)).collect()
        };
        let mut cards = cards;
        self.vouch(&mut cards, &models);
        json!({"ts": now(), "host": meminfo(), "cards": cards, "models": models, "services": self.services(), "runs": runs,
               "idle_minutes": self.o.idle_minutes, "ports": {"video": self.o.ports.video, "chat_ui": self.o.ports.chat_ui}})
    }

    /// The cards, a process's model by its registry id where its file is a registered one (`model_id`)
    fn named_cards(&self, all: &[Value]) -> Value {
        let base = |f: &str| f.rsplit('/').next().unwrap_or(f).to_string();
        let mut cards = self.cards.lock().unwrap().clone();
        // a chat process by what the chat server answers as (several entries can share one file)
        let chat = if cards.to_string().contains("\"kind\":\"llm\"") { self.chat_loaded(all) } else { None };
        for p in cards.as_array_mut().into_iter().flatten().flat_map(|c| c["procs"].as_array_mut().into_iter().flatten()) {
            if let (Some(id), "llm") = (&chat, p["kind"].as_str().unwrap_or("")) {
                p["model"] = json!(id);
            }
            if let Some(f) = p["model"].as_str().map(str::to_string) {
                if let Some(m) = all.iter().find(|m| m["file"].as_str().is_some_and(|x| base(x) == f) || m["id"] == f.as_str()) {
                    p["model"] = m["id"].clone();
                    p["model_id"] = m["id"].clone();
                    p["title"] = m["title"].clone();
                }
            }
        }
        cards
    }

    fn run_json(&self, id: u64, tail: usize) -> Option<Value> {
        let runs = self.runs.lock().unwrap();
        let r = runs.iter().find(|r| r.id == id)?;
        let out = std::fs::read(&r.log).unwrap_or_default();
        let out = String::from_utf8_lossy(&out[out.len().saturating_sub(tail)..]).into_owned();
        Some(json!({"id": r.id, "label": r.label, "started": r.started, "rc": r.rc, "out": out}))
    }

    fn new_run(&self, label: String) -> (u64, PathBuf) {
        let id = {
            let mut n = self.next.lock().unwrap();
            *n += 1;
            *n - 1
        };
        let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join("nextsycl-serve");
        let _ = std::fs::create_dir_all(&dir);
        let log = dir.join(format!("run-{}-{id}.log", std::process::id()));
        let mut runs = self.runs.lock().unwrap();
        runs.push(Run { id, label, started: now(), log: log.clone(), rc: None });
        if runs.len() > 50 {
            let old = runs.remove(0);
            let _ = std::fs::remove_file(old.log);
        }
        (id, log)
    }

    fn finish(&self, id: u64, rc: i32) {
        if let Some(r) = self.runs.lock().unwrap().iter_mut().find(|r| r.id == id) {
            r.rc = Some(rc);
        }
    }

    /// This program with these arguments, to the end; its output to a run's log (the page shows it). Ok: the run.
    fn exec(&self, label: String, args: &[String]) -> Result<u64, String> {
        let (id, log) = self.new_run(label.clone());
        let f = std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let f2 = f.try_clone().map_err(|e| e.to_string())?;
        let st = std::process::Command::new(&self.o.exe).args(args).stdin(std::process::Stdio::null()).stdout(f).stderr(f2).status();
        let rc = st.ok().and_then(|s| s.code()).unwrap_or(-1);
        self.finish(id, rc);
        if rc == 0 {
            Ok(id)
        } else {
            let out = std::fs::read_to_string(&log).unwrap_or_default();
            Err(format!("{label}: {}", out.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("failed")))
        }
    }

    /// The page's buttons: {"do": enable | disable | load | unload, "model", "gpus": [..], "force"}; a load runs on (the
    /// state shows it loading), the others answer when done
    fn action(self: &Arc<Self>, b: &Value) -> Result<Value, (u16, String)> {
        let model = b["model"].as_str().unwrap_or("").to_string();
        match b["do"].as_str().unwrap_or("") {
            "enable" => self.enable(&model, b.get("gpus")).map(|m| json!({"model": m})),
            "disable" => self.disable(&model, b["force"] == true).map(|_| json!({"model": model, "enabled": false})),
            "gpus" => self.set_gpus(&model, b.get("gpus")),
            "load" => {
                self.load_async(&model)?;
                Ok(json!({"model": model, "loading": true}))
            }
            "unload" => self.unload(&model).map(|_| json!({"model": model, "loaded": false})),
            "config" => self.config_set(b),
            "restart" => self.restart(b["service"].as_str().unwrap_or("")),
            other => Err((400, format!("no action {other:?} (enable, disable, gpus, load, unload, config, restart)"))),
        }
    }
}
