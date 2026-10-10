//! The box at a glance (`nextsycl serve`): every GPU - what it holds, by process, and how busy, hot and hungry it
//! is - and every nextsycl service - up or not, its model, what it is doing, a link to its page - with the controls
//! to start and stop the kinds' servers and to pick the chat model. A host process (libc only): the cards are read
//! from sysfs and the DRM clients' fdinfo (gpustat's sampler, one per card), the services over HTTP, and the actions
//! run this same program (`nextsycl image start ...`) or the video studio's llm.mode, which owns the chat card.
//!
//! Routes: `/` the page (wfe/home), `/ui/...` its modules, `GET /api/state`, `POST /api/action`
//! ({"do": "start" | "stop" | "mode", "kind", "model", "gpu", "force"}), `GET /api/runs/<id>` an action's output.

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
    /// the registry's entries (read on every request)
    pub models: Box<dyn Fn() -> Vec<Value> + Send + Sync>,
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
    cards: Mutex<Value>,
    runs: Mutex<Vec<Run>>,
    next: Mutex<u64>,
}

fn now() -> f64 {
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

impl Home {
    pub fn new(o: Options) -> Arc<Home> {
        let h = Arc::new(Home { o, cards: Mutex::new(json!([])), runs: Mutex::new(Vec::new()), next: Mutex::new(1) });
        let me = h.clone();
        std::thread::spawn(move || me.sample_cards());
        h
    }

    /// Every card every 2 s (busy % and power are deltas)
    fn sample_cards(&self) {
        let mut cards: Vec<Card> = find_cards().iter().map(|p| Card::open(p, None)).collect();
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
            std::thread::sleep(Duration::from_secs(2));
        }
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
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") | ("GET", "/index.html") => self.page(&mut s),
            ("GET", p) if p.starts_with("/ui/") || p.starts_with("/static/") => self.asset(&mut s, p),
            ("GET", "/health") => http::respond(&mut s, 200, &json!({"status": "ok"})),
            ("GET", "/api/state") => http::respond(&mut s, 200, &self.state()),
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

    /// The services, probed side by side (each answers in milliseconds, or times out alone)
    fn services(&self) -> Vec<Value> {
        let p = self.o.ports.clone();
        let t = |port: u16| Target::Tcp(format!("127.0.0.1:{port}"));
        let get = move |port: u16, path: &str, secs: u64| call_for(&t(port), "GET", path, None, secs).ok();
        let post = move |port: u16, path: &str| call_for(&Target::Tcp(format!("127.0.0.1:{port}")), "POST", path, Some(&json!({})), 3).ok();
        let probes: Vec<Box<dyn FnOnce() -> Value + Send>> = vec![
            Box::new(move || {
                let up = listening(p.chat_ui);
                json!({"id": "chat-ui", "title": "Chat", "what": "the chat front end (Open WebUI)", "port": p.chat_ui, "up": up, "link": "/"})
            }),
            Box::new(move || {
                let m = get(p.switch, "/v1/models", 3);
                let loaded = m.as_ref().and_then(|m| m["data"].as_array()?.iter().find(|x| x["loaded"] == true && x["kind"].is_null()).cloned());
                json!({"id": "switch", "title": "Model switch", "what": "one OpenAI endpoint for every chat model (nextsycl switch)",
                       "port": p.switch, "up": m.is_some(), "model": loaded.as_ref().map(|l| l["id"].clone()),
                       "detail": loaded.map(|l| l["name"].clone()), "api": "/v1"})
            }),
            Box::new(move || {
                let m = get(p.llm, "/v1/models", 3);
                let st = get(p.llm, "/status", 3);
                json!({"id": "llm", "kind": "llm", "title": "Chat model server", "what": "nextsycl llm serve", "port": p.llm, "up": m.is_some(),
                       "model": m.as_ref().and_then(|m| m["data"].get(0).map(|x| x["id"].clone())), "busy": st.as_ref().map(|s| s["busy"].clone()),
                       "api": "/v1"})
            }),
            Box::new(move || {
                let info = get(p.image, "/api/info", 3);
                let pr = get(p.image, "/api/progress", 3);
                json!({"id": "image", "kind": "image", "title": "Images", "what": "nextsycl image serve (Qwen-Image)", "port": p.image,
                       "up": info.is_some(), "model": info.as_ref().map(|i| i["model"].clone()), "busy": pr.as_ref().map(|p| p["busy"].clone()),
                       "waiting": pr.as_ref().map(|p| p["waiting"].clone()), "page": info.as_ref().map(|i| i["wfe"] != false), "link": "/",
                       "api": "/v1/images", "startable": true})
            }),
            Box::new(move || {
                let info = get(p.audio, "/api/info", 3);
                let pr = get(p.audio, "/api/progress", 3);
                json!({"id": "audio", "kind": "audio", "title": "Music", "what": "nextsycl audio serve (MiniMax Music)", "port": p.audio,
                       "up": info.is_some(), "model": info.as_ref().map(|i| i["model"].clone()), "busy": pr.as_ref().map(|p| p["busy"].clone()),
                       "waiting": pr.as_ref().map(|p| p["waiting"].clone()), "page": info.as_ref().map(|i| i["wfe"] != false), "link": "/",
                       "startable": true})
            }),
            Box::new(move || {
                let st = post(p.video, "/rpc/status").map(|v| v.get("result").cloned().unwrap_or(v));
                let mode = post(p.video, "/rpc/llm.mode").map(|v| v.get("result").cloned().unwrap_or(v));
                let detail = st.as_ref().map(|s| {
                    let label = s["job"]["label"].as_str().filter(|l| !l.is_empty()).or(s["job"]["name"].as_str()).unwrap_or("");
                    if s["idle"] == true || label.is_empty() {
                        format!("idle{}, {} queued", if s["paused"] == true { " (paused)" } else { "" }, s["queue_len"])
                    } else {
                        format!("{label}: {} {}%", s["stage"].as_str().unwrap_or(""), s["pct"].as_f64().map_or(0, |p| p.round() as i64))
                    }
                });
                json!({"id": "video", "kind": "video", "title": "Video studio", "what": "nextsycl video studio (MiniMax H3)", "port": p.video,
                       "up": st.is_some(), "detail": detail, "link": "/",
                       "chat": mode.map(|m| json!({"mode": m["mode"], "choices": m["choices"], "starting": m["starting"], "up": m["up"]}))})
            }),
        ];
        let hs: Vec<_> = probes.into_iter().map(std::thread::spawn).collect();
        hs.into_iter().map(|h| h.join().unwrap_or(Value::Null)).collect()
    }

    fn state(&self) -> Value {
        let all = (self.o.models)();
        let models: Vec<Value> = all.iter().filter(|m| m["enabled"] != false)
            .map(|m| json!({"id": m["id"], "title": m["title"], "kind": m["kind"], "gpus": m["gpus"]})).collect();
        let cards = self.named_cards(&all);
        let runs: Vec<Value> = {
            let ids: Vec<u64> = self.runs.lock().unwrap().iter().rev().take(8).map(|r| r.id).collect();
            ids.into_iter().filter_map(|id| self.run_json(id, 2048)).collect()
        };
        let gpustat = std::fs::read_to_string("/run/gpustat.json").ok().and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .map(|v| json!({"pci": v["pci"], "age": v["ts"].as_f64().map(|t| (now() - t).max(0.0))}));
        json!({"ts": now(), "host": meminfo(), "cards": cards, "services": self.services(), "models": models,
               "runs": runs, "gpustat": gpustat})
    }

    /// The cards, a process's model by its registry id where its file is a registered one
    fn named_cards(&self, all: &[Value]) -> Value {
        let base = |f: &str| f.rsplit('/').next().unwrap_or(f).to_string();
        let mut cards = self.cards.lock().unwrap().clone();
        for p in cards.as_array_mut().into_iter().flatten().flat_map(|c| c["procs"].as_array_mut().into_iter().flatten()) {
            if let Some(f) = p["model"].as_str().map(str::to_string) {
                if let Some(m) = all.iter().find(|m| m["file"].as_str().is_some_and(|x| base(x) == f) || m["id"] == f.as_str()) {
                    p["model"] = m["id"].clone();
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

    /// This program with these arguments, in the background; its output to the run's log
    fn spawn(self: &Arc<Self>, label: String, args: Vec<String>) -> Result<Value, (u16, String)> {
        let (id, log) = self.new_run(label);
        let f = std::fs::File::create(&log).map_err(|e| (500, format!("{}: {e}", log.display())))?;
        let f2 = f.try_clone().map_err(|e| (500, e.to_string()))?;
        let child = std::process::Command::new(&self.o.exe).args(&args).stdin(std::process::Stdio::null()).stdout(f).stderr(f2).spawn();
        match child {
            Ok(mut c) => {
                let me = self.clone();
                std::thread::spawn(move || {
                    let rc = c.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
                    me.finish(id, rc);
                });
                Ok(json!({"run": id}))
            }
            Err(e) => {
                self.finish(id, -1);
                Err((500, format!("{}: {e}", self.o.exe.display())))
            }
        }
    }

    fn action(self: &Arc<Self>, b: &Value) -> Result<Value, (u16, String)> {
        let s = |k: &str| b[k].as_str().unwrap_or("").to_string();
        let (what, kind, model) = (s("do"), s("kind"), s("model"));
        let safe = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.:".contains(c));
        match (what.as_str(), kind.as_str()) {
            ("start", "image" | "audio") => {
                if !safe(&model) {
                    return Err((400, "start needs a model".into()));
                }
                let gpu = b["gpu"].as_u64().ok_or((400, "start needs a gpu".to_string()))?;
                // a card another engine holds: refused unless forced (two engines on one card spill VRAM and can hang the box)
                let cards = self.named_cards(&(self.o.models)());
                let card = cards.get(gpu as usize).cloned().ok_or((400, format!("no GPU {gpu}")))?;
                let holders: Vec<String> = card["procs"].as_array().into_iter().flatten().filter(|p| p["vram_mb"].as_u64().unwrap_or(0) > 1024)
                    .map(|p| format!("{} ({} GiB)", p["model"].as_str().or(p["program"].as_str()).unwrap_or("?"), p["vram_mb"].as_u64().unwrap_or(0) / 1024)).collect();
                if !holders.is_empty() && b["force"] != true {
                    return Err((409, format!("GPU {gpu} holds {}; free it first, or start anyway", holders.join(", "))));
                }
                let port = if kind == "image" { self.o.ports.image } else { self.o.ports.audio };
                let args = vec![kind.clone(), "start".into(), model.clone(), "--gpu".into(), gpu.to_string(), "--port".into(), port.to_string(),
                                "--host".into(), self.o.serve_host.clone(), "--wfe".into()];
                self.spawn(format!("{kind} start {model} on GPU {gpu}"), args)
            }
            ("stop", "image" | "audio") => self.spawn(format!("{kind} stop"), vec![kind.clone(), "stop".into()]),
            ("mode", _) => {
                // the chat model through the video studio, which knows when a render holds the card ("none": no chat)
                if !safe(&model) {
                    return Err((400, "mode needs a model (or none)".into()));
                }
                let (id, log) = self.new_run(format!("chat model {model}"));
                let port = self.o.ports.video;
                let me = self.clone();
                std::thread::spawn(move || {
                    let r = call_for(&Target::Tcp(format!("127.0.0.1:{port}")), "POST", "/rpc/llm.mode", Some(&json!({"mode": model})), 600);
                    let (rc, text) = match r {
                        Ok(v) if v["ok"] != false => (0, serde_json::to_string_pretty(&v).unwrap_or_default()),
                        Ok(v) => (1, serde_json::to_string_pretty(&v).unwrap_or_default()),
                        Err(e) => (1, e),
                    };
                    let _ = std::fs::write(&log, text);
                    me.finish(id, rc);
                });
                Ok(json!({"run": id}))
            }
            _ => Err((400, format!("no action {what} {kind} (start | stop image / audio, mode)"))),
        }
    }
}

/// The registry's entries for the page (`kind` filled in)
pub fn with_kinds(all: Vec<Value>, kind_of: impl Fn(&Value) -> String) -> Vec<Value> {
    all.into_iter().map(|mut m| {
        m["kind"] = json!(kind_of(&m));
        m
    }).collect()
}
