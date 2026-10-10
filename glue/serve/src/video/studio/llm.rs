//! The card's other tenant: the language models the box serves when it is not rendering, and the switch between
//! them (`llm.mode`). The modes come from a JSON file (`H3_LLM_MODES`), since what they start and stop is the
//! machine's own business:
//!
//! ```json
//! {"default": "vllm",
//!  "modes": [{"name": "vllm", "title": "vLLM ...", "url": "http://127.0.0.1:8000", "health": "/health", "up_codes": [200, 503],
//!             "start": "bash /path/to/run.sh", "stop": "docker stop -t 120 vllm; docker rm -f vllm"},
//!            {"name": "coder", "title": "...", "url": "http://127.0.0.1:8085", "health": "/v1/models", "model": "coder-iq1_m\"",
//!             "start": "/path/to/run-coder.sh", "stop": "..."}]}
//! ```
//!
//! The engine daemon asks for mode `none` before it loads (the selected model stops) and for the previous mode after
//! it has unloaded (it starts again); this file only keeps the selection, runs the commands, and never starts a
//! model while the engine holds the card. After 30 min with the GPU idle and the selected model down, it starts it.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use super::{now, RpcError, Studio};

#[derive(Clone)]
pub struct Mode {
    pub name: String,
    pub title: String,
    pub url: String,
    pub health: String,
    pub model: Option<String>,
    pub up_codes: Vec<u16>,
    pub start: String,
    pub stop: String,
}

pub struct Llm {
    /// the modes file (read on every use: `nextsycl models` and `nextsycl serve` rewrite it as models are enabled)
    config: Option<PathBuf>,
    default: String,
    file: PathBuf,
    logs: PathBuf,
    starting: Mutex<Option<String>>,
    idle: Mutex<(Option<f64>, f64)>,
    cache: Mutex<Option<(f64, Value)>>,
}

/// GET url -> (status code, body), 4 s at most.
fn get(url: &str) -> Option<(u16, String)> {
    let (host, path) = super::http::split_url(url).ok()?;
    let addr = std::net::ToSocketAddrs::to_socket_addrs(&host).ok()?.next()?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(4)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(4))).ok()?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").ok()?;
    let mut buf = Vec::new();
    let _ = s.take(1 << 20).read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf).into_owned();
    let code = text.split_whitespace().nth(1)?.parse().ok()?;
    Some((code, text.split_once("\r\n\r\n").map(|x| x.1.to_string()).unwrap_or_default()))
}

impl Llm {
    fn read(config: Option<&PathBuf>) -> Value {
        config.and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}))
    }

    /// The modes as the file says now
    fn modes(&self) -> Vec<Mode> {
        Self::read(self.config.as_ref())["modes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| {
                        Some(Mode {
                            name: m["name"].as_str()?.to_string(),
                            title: m["title"].as_str().unwrap_or("").to_string(),
                            url: m["url"].as_str()?.to_string(),
                            health: m["health"].as_str().unwrap_or("/health").to_string(),
                            model: m["model"].as_str().map(str::to_string),
                            up_codes: m["up_codes"].as_array().map(|c| c.iter().filter_map(|x| x.as_u64().map(|x| x as u16)).collect()).unwrap_or_else(|| vec![200]),
                            start: m["start"].as_str().unwrap_or("").to_string(),
                            stop: m["stop"].as_str().unwrap_or("").to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn load(config: Option<PathBuf>, dir: &std::path::Path, logs: PathBuf) -> Llm {
        let v = Self::read(config.as_ref());
        Llm {
            config,
            default: v["default"].as_str().unwrap_or("none").to_string(),
            file: dir.join("llm-mode"),
            logs,
            starting: Mutex::new(None),
            idle: Mutex::new((None, 0.0)),
            cache: Mutex::new(None),
        }
    }

    fn mode(&self, name: &str) -> Option<Mode> {
        self.modes().into_iter().find(|m| m.name == name)
    }

    /// The selected mode (`none` when nothing is configured).
    pub fn selected(&self) -> String {
        let m = std::fs::read_to_string(&self.file).map(|s| s.trim().to_string()).unwrap_or_else(|_| self.default.clone());
        if m == "none" || self.mode(&m).is_some() { m } else if self.mode(&self.default).is_some() { self.default.clone() } else { "none".into() }
    }

    fn up(&self, m: &Mode) -> bool {
        match get(&format!("{}{}", m.url.trim_end_matches('/'), m.health)) {
            Some((code, body)) => match &m.model {
                Some(id) => body.contains(id.as_str()) && body.contains("\"loaded\""),
                None => m.up_codes.contains(&code),
            },
            None => false,
        }
    }

    pub fn status(&self) -> Value {
        if let Some((t, v)) = self.cache.lock().unwrap().as_ref() {
            if now() - t < 5.0 {
                return v.clone();
            }
        }
        let mode = self.selected();
        let modes = self.modes();
        let running: Vec<String> = modes.iter().filter(|m| self.up(m)).map(|m| m.name.clone()).collect();
        let v = json!({"mode": mode, "up": running.contains(&mode), "running": running, "starting": *self.starting.lock().unwrap(),
                       "choices": modes.iter().map(|m| (m.name.clone(), json!(m.title))).collect::<serde_json::Map<String, Value>>(),
                       "urls": modes.iter().map(|m| (m.name.clone(), json!(m.url))).collect::<serde_json::Map<String, Value>>(),
                       "url": self.mode(&mode).map(|m| m.url.clone())});
        *self.cache.lock().unwrap() = Some((now(), v.clone()));
        v
    }

    fn sh(cmd: &str) {
        if !cmd.trim().is_empty() {
            let _ = Command::new("sh").arg("-c").arg(format!("{cmd} >/dev/null 2>&1")).status();
        }
    }

    fn start(&'static self, name: String, why: &str) {
        let Some(m) = self.mode(&name) else { return };
        {
            let mut st = self.starting.lock().unwrap();
            if st.is_some() || self.up(&m) {
                return;
            }
            *st = Some(name.clone());
        }
        eprintln!("llm: starting {name} ({why})");
        let log = self.logs.join(format!("llm-{name}.log"));
        std::thread::spawn(move || {
            let out = std::fs::OpenOptions::new().create(true).append(true).open(&log).ok();
            let mut c = Command::new("setsid");
            c.arg("sh").arg("-c").arg(&m.start).stdin(Stdio::null());
            match out.and_then(|f| f.try_clone().ok().map(|g| (f, g))) {
                Some((a, b)) => {
                    c.stdout(a).stderr(b);
                }
                None => {
                    c.stdout(Stdio::null()).stderr(Stdio::null());
                }
            }
            let _ = c.spawn();
            for _ in 0..120 {
                std::thread::sleep(Duration::from_secs(10));
                if self.selected() != m.name {
                    Self::sh(&m.stop); // the selection moved on while it was starting
                    break;
                }
                if self.up(&m) {
                    break;
                }
            }
            *self.starting.lock().unwrap() = None;
            *self.cache.lock().unwrap() = None;
        });
    }

    /// `llm.mode {mode?}`: the status, after switching when a mode is given.
    pub fn set(&'static self, studio: &Studio, want: Option<&str>) -> Result<Value, RpcError> {
        let Some(want) = want else { return Ok(self.status()) };
        if want != "none" && self.mode(want).is_none() {
            let names: Vec<String> = self.modes().into_iter().map(|m| m.name).collect();
            return Err(RpcError::param(&format!("mode must be one of {names:?} or none")));
        }
        let cur = self.selected();
        let _ = std::fs::write(&self.file, want);
        *self.cache.lock().unwrap() = None;
        self.idle.lock().unwrap().0 = None;
        if self.starting.lock().unwrap().is_some() {
            return Ok(self.status()); // the starter sees the new selection and hands over
        }
        // stop everything but the wanted model; start it only when the engine does not hold the card
        for m in &self.modes() {
            if m.name != want && (want == "none" || want != cur || self.up(m)) {
                Self::sh(&m.stop);
            }
        }
        if want != "none" && !studio.engine_holds_card() {
            self.start(want.to_string(), "mode changed");
        }
        Ok(self.status())
    }

    /// Each scheduler tick: after 30 min with the GPU idle and the selected model down, start it (at most every 5 min).
    pub fn idle_watchdog(&'static self, studio: &Studio) {
        let busy = studio.engine_holds_card() || studio.has_runnable();
        let mut g = self.idle.lock().unwrap();
        if busy {
            g.0 = None;
            return;
        }
        let since = *g.0.get_or_insert(now());
        // the first tick after the studio starts (a boot, a restart) brings the selected model back at once: waiting
        // the half hour left the box without its chat model after every boot
        let first = g.1 == 0.0;
        if !first && (now() - since < 1800.0 || now() - g.1 < 300.0) {
            return;
        }
        g.1 = now();
        drop(g);
        let mode = self.selected();
        if mode != "none" && self.mode(&mode).is_some_and(|m| !self.up(&m)) {
            let why = if first { format!("the studio started and {mode} is down") } else { format!("GPU idle {} min and {mode} is down", ((now() - since) / 60.0) as u64) };
            self.start(mode.clone(), &why);
        }
    }
}
