//! The backplane's behaviour, with no GPU and no model: the real `Home` over a temporary registry, the program it runs a
//! shell script that notes what it was told, model servers that are threads answering as a server does, and the cards
//! given from here (Home::fix_cards) from what those "servers" hold. Called as a client calls it.
//!
//! What only the real thing can show is not here: real memory letting go, the switcher in front of a real chat server,
//! a daemon's real worker kept by /engine/keep.
//!
//! `cargo test -p nextsycl-serve --test backplane`

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nextsycl_serve::home::registry::Store;
use nextsycl_serve::home::{Home, Options, Ports};
use serde_json::{json, Value};

const TOKEN: &str = "ns-test-token";

struct Mem {
    dir: PathBuf,
    models: Arc<Mutex<Vec<Value>>>,
    max_rsvp: Option<String>,
}

impl Store for Mem {
    fn models(&self) -> Result<Vec<Value>, String> {
        Ok(self.models.lock().unwrap().clone())
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
        let mut all = self.models.lock().unwrap();
        let m = all.iter_mut().find(|m| m["id"] == id).ok_or_else(|| format!("no model {id}"))?;
        f(m);
        Ok(m.clone())
    }
    fn registry_file(&self) -> PathBuf {
        self.dir.join("models.json")
    }
    fn get(&self, name: &str) -> Option<String> {
        match name {
            "NS_API_TOKEN" => Some(TOKEN.to_string()),
            // no lock file of the box's own gets in the test's way
            "NS_GPU_LOCK" => Some("off".to_string()),
            "NS_SOCKET_DIR" => Some(self.dir.to_string_lossy().into_owned()),
            "NS_MAX_RSVP_S" => self.max_rsvp.clone(),
            _ => None,
        }
    }
    fn in_env(&self, _name: &str) -> bool {
        false
    }
    fn file(&self) -> PathBuf {
        self.dir.join("nextsycl.conf")
    }
    fn guide(&self, _entry: &Value) -> Option<String> {
        None
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn read_request(s: &mut TcpStream) -> Option<(String, String, Value)> {
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut raw = Vec::new();
    let mut b = [0u8; 4096];
    loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..i]).into_owned();
            let len: usize = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
            while raw.len() < i + 4 + len {
                let n = s.read(&mut b).ok()?;
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&b[..n]);
            }
            let mut first = head.lines().next()?.split_whitespace();
            let (m, p) = (first.next()?.to_string(), first.next()?.to_string());
            let body = serde_json::from_slice(&raw[i + 4..]).unwrap_or(Value::Null);
            return Some((m, p, body));
        }
        let n = s.read(&mut b).ok()?;
        if n == 0 {
            return None;
        }
        raw.extend_from_slice(&b[..n]);
    }
}

fn answer(s: &mut TcpStream, status: u16, v: &Value) {
    let b = v.to_string();
    let _ = write!(s, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{b}", b.len());
}

/// A box of fake servers around a real front door
struct Box_ {
    dir: PathBuf,
    addr: String,
    home: Arc<Home>,
    models: Arc<Mutex<Vec<Value>>>,
    video: u16,
    image: u16,
    stop: Arc<AtomicBool>,
}

impl Drop for Box_ {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(150));
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn note(dir: &Path, what: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("seen.log")) {
        let _ = writeln!(f, "{what}");
    }
}

/// A server's answers: an image, audio or chat server, or the studio with its daemon
fn serve_one(dir: &Path, port: u16, is_video: bool, id: &str, mut s: TcpStream) {
    let Some((method, path, body)) = read_request(&mut s) else { return };
    note(dir, &format!("{port} {method} {path} {body}"));
    let file = |n: &str| dir.join(format!("{n}-{port}"));
    if is_video {
        let status = std::fs::read(dir.join("studio.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .unwrap_or(json!({"idle": true, "queue_items": []}));
        let v = match path.as_str() {
            "/engine/status" => {
                let names: Vec<Value> = std::fs::read_to_string(dir.join("engines")).unwrap_or_default().split_whitespace().map(|n| json!({"name": n})).collect();
                let w = std::fs::read_to_string(dir.join("worker")).unwrap_or_default();
                let gpus = if w.trim().is_empty() { json!([{"gpu": 0, "engine": "unloaded"}]) } else { json!([{"gpu": 0, "engine": "loaded", "denoiser": w.trim()}]) };
                json!({"engines": names, "gpus": gpus})
            }
            "/engine/jobs" => {
                let _ = std::fs::write(dir.join("worker"), body["engine"].as_str().unwrap_or(""));
                json!({"id": 1})
            }
            "/engine/keep" => json!({"kept": [0]}),
            "/rpc/llm.mode" => json!({"ok": true, "result": {}}),
            "/rpc/jobs.list" => json!({"ok": true, "result": {"jobs": std::fs::read(dir.join("clips.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).unwrap_or(json!([]))}}),
            "/rpc/generate" => json!({"ok": true, "result": {"queued": false, "label": body["label"]}}),
            _ => json!({"ok": true, "result": status}),
        };
        return answer(&mut s, 200, &v);
    }
    let v = match (method.as_str(), path.as_str()) {
        ("GET", "/api/info") => json!({"model": id, "loading": false}),
        ("GET", "/api/progress") => json!({"busy": file("busy").exists(), "waiting": 0}),
        ("GET", "/v1/models") => json!({"data": [{"id": id, "status": {"value": "loaded"}}]}),
        ("POST", "/api/keepalive") => return answer(&mut s, 404, &json!({"error": "no route"})),
        ("POST", p) if p.starts_with("/v1/") => {
            if let Ok(t) = std::fs::read_to_string(file("slow")) {
                std::thread::sleep(Duration::from_secs_f64(t.trim().parse().unwrap_or(0.0)));
            }
            json!({"data": [{"url": "http://nextsycl/x.png"}], "nextsycl": {"seed": 1}, "choices": []})
        }
        _ => json!({}),
    };
    answer(&mut s, 200, &v);
}

/// A port that answers while its file (up-<port>: "<id> <gpu> <kind>") says a server is on it
fn fake_port(dir: PathBuf, port: u16, is_video: bool, stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let mut l: Option<TcpListener> = None;
        while !stop.load(Ordering::Relaxed) {
            let up = std::fs::read_to_string(dir.join(format!("up-{port}"))).ok();
            match (&up, &l) {
                (Some(_), None) => {
                    if let Ok(x) = TcpListener::bind(("127.0.0.1", port)) {
                        x.set_nonblocking(true).unwrap();
                        l = Some(x);
                    }
                }
                (None, Some(_)) => l = None,
                _ => {}
            }
            if let (Some(x), Some(u)) = (&l, &up) {
                if let Ok((s, _)) = x.accept() {
                    s.set_nonblocking(false).unwrap();
                    let (d, id) = (dir.clone(), u.split_whitespace().next().unwrap_or("").to_string());
                    std::thread::spawn(move || serve_one(&d, port, is_video, &id, s));
                    continue;
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
}

/// The cards as the fake servers hold them: `n` of 32 GiB
fn fake_cards(dir: PathBuf, home: Arc<Home>, models: Arc<Mutex<Vec<Value>>>, ports: Vec<u16>, n: usize, stop: Arc<AtomicBool>) {
    let set = move || {
        let mut cards: Vec<Value> = (0..n).map(|i| json!({"index": i, "vram_total_mb": 32768, "vram_used_mb": 0, "procs": []})).collect();
        for p in &ports {
            let Ok(u) = std::fs::read_to_string(dir.join(format!("up-{p}"))) else { continue };
            let f: Vec<&str> = u.split_whitespace().collect();
            let (id, gpu, kind) = (f[0], f[1].parse::<usize>().unwrap_or(0), f[2]);
            if kind == "video" {
                continue;
            }
            let mb = models.lock().unwrap().iter().find(|m| m["id"] == id).and_then(|m| m["vram_gib"].as_f64()).unwrap_or(1.0) * 1024.0;
            let c = &mut cards[gpu];
            c["vram_used_mb"] = json!(c["vram_used_mb"].as_u64().unwrap() + mb as u64);
            c["procs"].as_array_mut().unwrap().push(json!({"pid": 1, "program": "nextsycl", "kind": kind, "model": id, "port": p.to_string(), "vram_mb": mb as u64}));
        }
        home.fix_cards(Value::from(cards));
    };
    set();
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            set();
            std::thread::sleep(Duration::from_millis(100));
        }
    });
}

fn image(id: &str, gib: f64) -> Value {
    json!({"id": id, "kind": "image", "enabled": true, "arch": "qwenimage", "file": format!("/nonexistent/{id}"), "gpus": "0", "title": id, "vram_gib": gib})
}

fn chat(id: &str, gib: f64) -> Value {
    json!({"id": id, "kind": "llm", "enabled": true, "file": format!("/nonexistent/{id}"), "gpus": "0", "title": id, "vram_gib": gib})
}

fn clip_model(id: &str) -> Value {
    json!({"id": id, "kind": "video", "enabled": true, "file": format!("/nonexistent/{id}"), "gpus": "0", "title": id})
}

/// `tag`: the box's name; its registry; `dir`: a directory to come back to (a restart), else its own
fn start(tag: &str, entries: Vec<Value>, keep_dir: Option<(PathBuf, Arc<Mutex<Vec<Value>>>)>, max_rsvp: Option<&str>) -> Box_ {
    let (dir, models) = keep_dir.unwrap_or_else(|| {
        let d = std::env::temp_dir().join(format!("nextsycl-backplane-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        (d, Arc::new(Mutex::new(entries)))
    });
    let (img, img2, aud, m3d, llm, video, door) = (free_port(), free_port(), free_port(), free_port(), free_port(), free_port(), free_port());
    let stop = Arc::new(AtomicBool::new(false));
    // (a script of this box's own: a box "restarted" in a test takes the old one's away, its door lives on in the process)
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        if e.file_name().to_string_lossy().starts_with("fake-") {
            let _ = std::fs::remove_file(e.path());
        }
    }
    let script = dir.join(format!("fake-{door}.sh"));
    std::fs::write(&script, format!(r#"#!/bin/sh
D="{d}"
echo "$*" >> "$D/exec.log"
kind=$1; cmd=$2; id=$3; port=; slot=0; gpu=0
while [ $# -gt 0 ]; do case "$1" in --port) port=$2;; --slot) slot=$2;; --gpu) gpu=$2;; esac; shift; done
case "$kind $cmd" in
  "video start") echo "video 0 video" > "$D/up-{video}";;
  "video stop") rm -f "$D/up-{video}";;
  *" start") echo "$id $gpu $kind" > "$D/up-$port"; echo "$port" > "$D/slot-$kind-$slot"; sleep 0.2;;
  *" stop") p=$(cat "$D/slot-$kind-$slot" 2>/dev/null); rm -f "$D/up-$p";;
esac
"#, d = dir.display())).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    // what a restart finds is what the last box's ports held: nothing on these
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        if e.file_name().to_string_lossy().starts_with("up-") {
            let _ = std::fs::remove_file(e.path());
        }
    }
    let names: Vec<String> = models.lock().unwrap().iter().filter(|m| m["kind"] == "video").filter_map(|m| m["id"].as_str().map(str::to_string)).collect();
    std::fs::write(dir.join("engines"), names.join(" ")).unwrap();
    for (p, v) in [(img, false), (img2, false), (aud, false), (m3d, false), (llm, false), (video, true)] {
        fake_port(dir.clone(), p, v, stop.clone());
    }
    // the chat server's control socket: answering nothing
    let sock = dir.join("nextsycl.sock");
    let _ = std::fs::remove_file(&sock);
    let ul = std::os::unix::net::UnixListener::bind(&sock).unwrap();
    std::thread::spawn(move || {
        for mut c in ul.incoming().flatten() {
            let mut b = [0u8; 2048];
            let _ = c.read(&mut b);
            let body = json!({"active": []}).to_string();
            let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        }
    });
    let ports = Ports { llm, switch: 0, image: img, audio: aud, model3d: m3d, image2: img2, audio2: 0, model3d2: 0, video, chat_ui: 0 };
    let cfg = Box::new(Mem { dir: dir.clone(), models: models.clone(), max_rsvp: max_rsvp.map(str::to_string) });
    let addr = format!("127.0.0.1:{door}");
    let home = Home::new(Options { wfe: dir.join("wfe"), exe: script, ports, serve_host: "127.0.0.1".into(), door: addr.clone(), cfg, idle_minutes: 0 }).expect("the front door starts");
    fake_cards(dir.clone(), home.clone(), models.clone(), vec![img, img2, aud, m3d, llm, video], 1, stop.clone());
    let (h, a) = (home.clone(), addr.clone());
    std::thread::spawn(move || {
        let _ = h.run(&a);
    });
    for _ in 0..100 {
        if TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Box_ { dir, addr, home, models, video, image: img, stop }
}

impl Box_ {
    fn raw(&self, method: &str, path: &str, headers: &str, body: &Value) -> (u16, Value) {
        let mut s = TcpStream::connect(&self.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        let b = body.to_string();
        write!(s, "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nAuthorization: Bearer {TOKEN}\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", self.addr, b.len()).unwrap();
        let mut raw = Vec::new();
        let _ = s.read_to_end(&mut raw);
        let text = String::from_utf8_lossy(&raw).into_owned();
        let st = text.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        let body = text.split_once("\r\n\r\n").map_or("", |x| x.1);
        (st, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    fn rpc(&self, method: &str, body: Value) -> Value {
        let (st, v) = self.raw("POST", &format!("/rpc/{method}"), "", &body);
        assert_eq!(st, 200, "{method}: /rpc is HTTP 200 whatever the outcome: {v}");
        v
    }

    fn ok(&self, method: &str, body: Value) -> Value {
        let v = self.rpc(method, body);
        assert_eq!(v["ok"], true, "{method} failed: {v}");
        v["result"].clone()
    }

    fn err(&self, method: &str, body: Value) -> Value {
        let v = self.rpc(method, body);
        assert_eq!(v["ok"], false, "{method} should have failed: {v}");
        v["error"].clone()
    }

    fn exec_log(&self) -> String {
        std::fs::read_to_string(self.dir.join("exec.log")).unwrap_or_default()
    }

    fn seen(&self) -> String {
        std::fs::read_to_string(self.dir.join("seen.log")).unwrap_or_default()
    }

    fn up(&self, id: &str) -> bool {
        std::fs::read_dir(&self.dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("up-"))
            .any(|e| std::fs::read_to_string(e.path()).unwrap_or_default().split_whitespace().next() == Some(id))
    }

    fn until(&self, what: &str, secs: u64, f: impl Fn(&Box_) -> bool) {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(secs) {
            if f(self) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("{what}: not within {secs} s; exec log:\n{}", self.exec_log());
    }

    fn model_state(&self, id: &str) -> Value {
        self.ok("system.models", json!({}))["models"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone()
    }

    fn history(&self, id: &str, rows: &[Value]) {
        let d = self.dir.join("history");
        std::fs::create_dir_all(&d).unwrap();
        let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
        std::fs::write(d.join(format!("{id}.jsonl")), text).unwrap();
    }

    fn make(&self, id: &str, extra: Value) -> Value {
        let mut b = json!({"model": id, "prompt": "a lighthouse", "size": "1024x1024", "steps": 4, "label": format!("{id}-{}", self.exec_log().len())});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        self.rpc("image.generate", b)
    }
}

#[test]
fn only_idle_holders_neither_pinned_nor_held_go_for_room() {
    let b = start("evict", vec![image("a", 20.0), image("b", 20.0), image("c", 20.0)], None, None);
    // on demand: a loads for its first request; b's request stops idle a
    assert_eq!(b.make("a", json!({}))["ok"], true);
    assert!(b.up("a"));
    assert_eq!(b.make("b", json!({}))["ok"], true);
    assert!(b.up("b") && !b.up("a"), "the idle holder went: {}", b.exec_log());

    // a pin holds: no room for a, and the job says who and why
    assert_eq!(b.ok("system.pin", json!({"model": "b"})), json!({"model": "b", "pinned": true, "loaded": true}));
    assert_eq!(b.model_state("b")["pinned"], true);
    let e = b.err("system.unload", json!({"model": "b"}));
    assert_eq!((e["code"].clone(), e["status"].clone()), (json!("conflict"), json!(409)), "{e}");
    assert!(b.up("b"));
    let j = b.ok("image.generate", json!({"model": "a", "prompt": "x", "label": "after-pin", "wait": false}));
    assert_eq!(j["state"], "queued");
    b.until("the job says the pin", 20, |b| {
        let s = b.ok("image.status", json!({"label": "after-pin"}));
        s["waiting_for"].as_str().is_some_and(|w| w.contains("GPU 0") && w.contains("pinned: b (unpin to free the room)"))
    });
    assert!(b.up("b") && !b.up("a"), "a pinned model is never stopped");
    // unpinned: the waiting job goes on, on the room b leaves
    assert_eq!(b.ok("system.unpin", json!({"model": "b"})), json!({"model": "b", "pinned": false}));
    b.until("the job finishes once unpinned", 20, |b| b.ok("image.status", json!({"label": "after-pin"}))["state"] == "done");
    assert!(b.up("a") && !b.up("b"));

    // rsvp_s: granted, named in the refusal, a smaller value does not shorten it, 0 releases
    let r = b.make("c", json!({"rsvp_s": 500}));
    assert_eq!(r["result"]["rsvp_granted_s"], 500, "{r}");
    let r = b.make("c", json!({"rsvp_s": 10}));
    assert!(r["result"]["rsvp_granted_s"].as_u64().unwrap() > 400, "{r}");
    assert!(b.model_state("c")["held_s"].as_u64().unwrap() > 400);
    b.ok("image.generate", json!({"model": "a", "prompt": "x", "label": "after-rsvp", "wait": false}));
    b.until("the job says the hold", 20, |b| {
        b.ok("image.status", json!({"label": "after-rsvp"}))["waiting_for"].as_str().is_some_and(|w| w.contains("held by rsvp_s: c (") && w.contains(" s more)"))
    });
    assert!(b.up("c"));
    let e = b.ok("image.status", json!({"label": "after-rsvp"}));
    assert_ne!(e["state"], "done");
    // (the job above holds the kind's only server's turn: c is released by the tool's unload, which ends its hold)
    b.ok("system.unload", json!({"model": "c"}));
    assert_eq!(b.model_state("c")["held_s"], 0);
    b.until("the job finishes once released", 20, |b| b.ok("image.status", json!({"label": "after-rsvp"}))["state"] == "done");
    // rsvp_s 0 on a request releases what an earlier one held
    let r = b.make("a", json!({"rsvp_s": 300}));
    assert_eq!(r["result"]["rsvp_granted_s"], 300);
    let r = b.make("a", json!({"rsvp_s": 0}));
    assert_eq!(r["result"]["rsvp_granted_s"], 0);
    assert_eq!(b.model_state("a")["held_s"], 0);
}

#[test]
fn a_busy_server_is_never_stopped_and_the_memory_rule_stands() {
    let b = start("busy", vec![image("big", 28.0), image("small", 3.0)], None, None);
    assert_eq!(b.make("big", json!({}))["ok"], true);
    std::fs::write(b.dir.join(format!("busy-{}", b.image)), "").unwrap();
    b.ok("image.generate", json!({"model": "small", "prompt": "x", "label": "beside", "wait": false}));
    // 4 GiB free, small needs 3 + 1.5: no room, and the busy holder stays
    b.until("the job waits", 20, |b| b.ok("image.status", json!({"label": "beside"}))["waiting_for"].as_str().is_some_and(|w| !w.is_empty() && w.contains("big")));
    std::thread::sleep(Duration::from_secs(4));
    assert!(b.up("big") && !b.up("small"), "running work is not stopped: {}", b.exec_log());
    std::fs::remove_file(b.dir.join(format!("busy-{}", b.image))).unwrap();
    b.until("the job goes when the holder idles", 30, |b| b.ok("image.status", json!({"label": "beside"}))["state"] == "done");
}

#[test]
fn a_pin_comes_back_after_a_restart_and_a_hold_survives_one() {
    let dir;
    let models;
    {
        let b = start("restart", vec![image("p", 8.0), image("h", 8.0)], None, None);
        b.ok("system.pin", json!({"model": "p"}));
        assert_eq!(b.make("h", json!({"rsvp_s": 900}))["ok"], true);
        assert_eq!(b.models.lock().unwrap().iter().find(|m| m["id"] == "p").unwrap()["pinned"], true, "in the registry");
        dir = b.dir.clone();
        models = b.models.clone();
        b.stop.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(200));
        std::mem::forget(b);
    }
    let b = start("restart", vec![], Some((dir, models)), None);
    assert!(!b.up("p"), "a fresh box holds nothing");
    b.until("the pinned model loads again by itself", 20, |b| b.up("p"));
    assert_eq!(b.model_state("p")["pinned"], true);
    assert!(b.model_state("h")["held_s"].as_u64().unwrap() > 800, "the hold is in its file");
}

#[test]
fn a_known_wait_over_the_limit_is_declined_and_nothing_happens() {
    let b = start("declined", vec![image("cold", 8.0), image("hot", 8.0)], None, None);
    b.history("cold", &[json!({"shape": "image.generate/1024x1024/s4", "run_s": 30.0}), json!({"shape": "load", "load_s": 500.0})]);
    // read-only: what it would be, and nothing moved
    let v = b.ok("system.eta", json!({"method": "image.generate", "params": {"model": "cold", "size": "1024x1024", "steps": 4}, "max_willing_to_wait_s": 60}));
    assert_eq!((v["eta"]["wait_s"].clone(), v["eta"]["run_s"].clone(), v["eta"]["total_s"].clone()), (json!(500.0), json!(30.0), json!(530.0)), "{v}");
    assert_eq!((v["eta"]["basis"].clone(), v["eta"]["samples"].clone(), v["would_accept"].clone(), v["hot"].clone()), (json!("shape"), json!(1), json!(false), json!(false)), "{v}");
    assert!(!b.exec_log().contains("start cold"));
    assert_eq!(b.ok("system.eta", json!({"method": "image.generate", "params": {"model": "cold"}}))["eta"]["basis"], "model", "another shape: the model's runs");
    assert_eq!(b.err("system.eta", json!({"method": "system.busy"}))["code"], "invalid_params");

    let e = b.make("cold", json!({"max_willing_to_wait_s": 60}));
    assert_eq!(e["ok"], false, "{e}");
    let e = &e["error"];
    assert_eq!((e["code"].clone(), e["retryable"].clone(), e["status"].clone(), e["max_willing_to_wait_s"].clone()), (json!("wait_too_long"), json!(true), json!(503), json!(60.0)), "{e}");
    assert_eq!(e["eta"]["wait_s"], 500.0);
    assert!(e.get("waited_s").is_none());
    assert!(!b.exec_log().contains("start cold"), "nothing started");
    assert_eq!(b.ok("image.jobs", json!({}))["queued"], json!([]), "nothing queued");

    // a hot model is never declined, whatever its history says
    assert_eq!(b.make("hot", json!({}))["ok"], true);
    b.history("hot", &[json!({"shape": "load", "load_s": 900.0, "run_s": 900.0})]);
    let v = b.make("hot", json!({"max_willing_to_wait_s": 0}));
    assert_eq!(v["ok"], true, "{v}");
    assert_eq!(v["result"]["job"]["eta"]["basis"], "model");
}

#[test]
fn an_unknown_wait_goes_ahead_and_is_withdrawn_at_the_limit() {
    let b = start("withdrawn", vec![image("holder", 28.0), image("fresh", 28.0), image("fresh2", 28.0)], None, None);
    b.ok("system.pin", json!({"model": "holder"}));
    let v = b.ok("system.eta", json!({"method": "image.generate", "params": {"model": "fresh"}, "max_willing_to_wait_s": 2}));
    assert_eq!((v["eta"]["wait_s"].clone(), v["eta"]["basis"].clone(), v["would_accept"].clone()), (Value::Null, json!("none"), json!(true)), "no history: unknown, accepted: {v}");
    // the first runs and waits for room; it is withdrawn from there
    let e = b.err("image.generate", json!({"model": "fresh", "prompt": "x", "label": "w1", "max_willing_to_wait_s": 2}));
    assert_eq!(e["code"], "wait_too_long", "{e}");
    assert!(e["waited_s"].as_f64().unwrap() >= 2.0, "{e}");
    assert_eq!(e["max_willing_to_wait_s"], 2.0);
    assert!(!b.up("fresh") && b.up("holder"));
    assert_eq!(b.ok("image.status", json!({"label": "w1"}))["error"]["code"], "wait_too_long", "the job's own record says so too");
    // one waiting behind another in the queue is withdrawn there
    b.ok("image.generate", json!({"model": "fresh", "prompt": "x", "label": "w2", "wait": false}));
    b.ok("image.generate", json!({"model": "fresh2", "prompt": "x", "label": "w3", "wait": false, "max_willing_to_wait_s": 1}));
    b.until("the queued job is withdrawn", 15, |b| b.ok("image.status", json!({"label": "w3"}))["error"]["code"] == "wait_too_long");
    assert_eq!(b.ok("image.status", json!({"label": "w3"}))["state"], "failed");
    // queued and running jobs carry an eta
    let s = b.ok("image.status", json!({"label": "w2"}));
    assert_eq!(s["eta"]["basis"], "none", "{s}");
    assert!(s.get("eta_remaining_s").is_some(), "{s}");
    b.ok("image.cancel", json!({"label": "w2"}));
}

#[test]
fn history_answers_after_a_run_and_a_bad_value_is_refused() {
    let b = start("history", vec![image("m", 8.0)], None, None);
    assert_eq!(b.make("m", json!({}))["ok"], true);
    assert_eq!(b.make("m", json!({}))["ok"], true);
    let text = std::fs::read_to_string(b.dir.join("history/m.jsonl")).unwrap();
    assert_eq!(text.lines().filter(|l| l.contains("\"run_s\"")).count(), 2, "{text}");
    assert_eq!(text.lines().filter(|l| l.contains("\"load_s\"")).count(), 1, "{text}");
    let v = b.ok("system.eta", json!({"method": "image.generate", "params": {"model": "m", "size": "1024x1024", "steps": 4}}));
    assert_eq!((v["eta"]["samples"].clone(), v["eta"]["basis"].clone(), v["hot"].clone(), v["eta"]["wait_s"].clone()), (json!(2), json!("shape"), json!(true), json!(0.0)), "{v}");
    assert_eq!(b.err("image.generate", json!({"model": "m", "prompt": "x", "rsvp_s": -5}))["code"], "invalid_params");
    assert_eq!(b.err("image.generate", json!({"model": "m", "prompt": "x", "max_willing_to_wait_s": "soon"}))["status"], 400);
    let f = b.ok("system.version", json!({}));
    for name in ["system.pin", "system.eta", "rsvp_s", "max_willing_to_wait_s", "errors.wait_too_long"] {
        assert!(f["features"].as_array().unwrap().iter().any(|x| x == name), "{name}");
    }
    assert!(f["methods"].as_array().unwrap().iter().any(|x| x == "system.unpin"));
}

#[test]
fn rsvp_is_clamped_by_the_setting() {
    let b = start("clamp", vec![image("m", 8.0)], None, Some("60"));
    let r = b.make("m", json!({"rsvp_s": 86400}));
    assert_eq!(r["result"]["rsvp_granted_s"], 60, "{r}");
}

#[test]
fn a_video_call_brings_the_daemon_up_and_a_clip_has_its_eta_hot_or_not() {
    let b = start("video", vec![clip_model("clips")], None, None);
    let up = b.dir.join(format!("up-{}", b.video));
    // the door starts the daemon by itself; stopped later, any video call brings it back
    b.until("the daemon up", 15, |_| up.exists());
    std::fs::remove_file(&up).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(b.rpc("video.status", json!({}))["ok"], true);
    assert!(up.exists(), "the call started it: {}", b.exec_log());

    // the studio's records are the history: no worker up, yet the wait is known
    std::fs::write(b.dir.join("clips.json"), json!([{"total": 100.0, "file": "a.mp4", "seconds": 5, "steps": 8, "canvas": "768x576", "mtime": 1, "engine": "INT8"}]).to_string()).unwrap();
    b.until("the clips read in", 25, |b| b.dir.join("history/clips.jsonl").exists());
    let q = json!({"model": "clips", "width": 768, "height": 576, "seconds": 5, "steps": 8});
    let v = b.ok("system.eta", json!({"method": "video.generate", "params": q}));
    assert_eq!((v["hot"].clone(), v["eta"]["wait_s"].clone(), v["eta"]["run_s"].clone(), v["eta"]["basis"].clone()), (json!(false), json!(0.0), json!(100.0), json!("shape")), "{v}");

    // a clip ahead: known, and over a limit it is declined without reaching the studio
    std::fs::write(b.dir.join("studio.json"), json!({"idle": false, "eta": 1000, "job": {"label": "other"}, "queue_items": [], "waiting_for": null}).to_string()).unwrap();
    let e = b.err("video.generate", json!({"prompt": "x", "max_willing_to_wait_s": 10}));
    assert_eq!((e["code"].clone(), e["eta"]["wait_s"].clone()), (json!("wait_too_long"), json!(1000.0)), "{e}");
    assert!(!b.seen().contains("/rpc/generate"));
    let s = b.ok("video.status", json!({}));
    assert_eq!((s["eta"].clone(), s["eta_remaining_s"].clone()), (json!(1000), json!(1000)), "its number stays, what every job has is added: {s}");

    // a clip whose wait is not known (no history yet) that queued past its limit is taken out and says so
    std::fs::write(b.dir.join("clips.json"), "[]").unwrap();
    std::fs::remove_file(b.dir.join("history/clips.jsonl")).unwrap();
    std::fs::write(b.dir.join("studio.json"), json!({"idle": true, "queue_items": [{"label": "late", "held": false}]}).to_string()).unwrap();
    let r = b.ok("video.generate", json!({"prompt": "x", "label": "late", "queue": true, "max_willing_to_wait_s": 1}));
    assert!(r["eta"].is_object(), "{r}");
    b.until("the clip is withdrawn", 15, |b| b.seen().contains("/rpc/queue.remove"));
    let e = b.err("video.status", json!({"label": "late"}));
    assert_eq!(e["code"], "wait_too_long", "{e}");
    assert!(e["waited_s"].as_f64().unwrap() >= 1.0);

    // a clip's reservation: said on the answer, held when the clip is over
    std::fs::write(b.dir.join("studio.json"), json!({"idle": true, "queue_items": []}).to_string()).unwrap();
    let r = b.ok("video.generate", json!({"prompt": "x", "rsvp_s": 300}));
    assert_eq!(r["rsvp_granted_s"], 300, "{r}");
    assert!(b.seen().lines().filter(|l| l.contains("/rpc/generate")).all(|l| !l.contains("rsvp_s") && !l.contains("max_willing")), "the door's fields stay here");
    b.until("the hold starts as the clip ends", 10, |b| b.model_state("clips")["held_s"].as_u64().unwrap_or(0) > 0);

    // a pin loads the engine and keeps its worker from its idle unload
    assert_eq!(b.ok("system.pin", json!({"model": "clips"}))["loaded"], true);
    assert!(b.seen().contains("\"kind\":\"load\""), "{}", b.seen());
    b.until("the worker told to stay", 40, |b| b.seen().contains("/engine/keep"));
}

#[test]
fn chat_holds_are_one_hold_a_keep_yields_to_chat_and_a_reservation_does_not() {
    let b = start("chat", vec![chat("a", 10.0), chat("b", 10.0), image("pic", 28.0)], None, None);
    let msg = json!([{"role": "user", "content": "hi"}]);
    // X-Keep-Up-S through /v1: the one hold, seen in /chat/held and system.models
    let (st, _) = b.raw("POST", "/v1/chat/completions", "X-Keep-Up-S: 300\r\n", &json!({"model": "a", "messages": msg}));
    assert_eq!(st, 200);
    assert!(b.up("a"));
    assert!(b.model_state("a")["held_s"].as_u64().unwrap() > 250);
    let (st, v) = b.raw("GET", "/chat/held", "", &json!({}));
    assert_eq!((st, v), (200, json!({"held": true, "model": "a"})));
    // a kept chat model is not stopped for a picture ...
    b.ok("image.generate", json!({"model": "pic", "prompt": "x", "label": "pic1", "wait": false}));
    b.until("the job names the hold", 20, |b| b.ok("image.status", json!({"label": "pic1"}))["waiting_for"].as_str().is_some_and(|w| w.contains("held by rsvp_s: a (")));
    assert!(b.up("a"));
    b.ok("image.cancel", json!({"label": "pic1"}));
    // ... yet another chat model may take its place, as Open WebUI's picker always could
    assert_eq!(b.ok("llm.chat", json!({"model": "b", "messages": msg, "rsvp_s": 400}))["rsvp_granted_s"], 400);
    assert!(b.up("b") && !b.up("a"), "{}", b.exec_log());
    // a reservation does not give way
    let e = b.err("llm.chat", json!({"model": "a", "messages": msg}));
    assert_eq!((e["code"].clone(), e["retryable"].clone(), e["status"].clone()), (json!("no_room"), json!(true), json!(503)), "{e}");
    assert!(e["message"].as_str().unwrap().contains("held by rsvp_s: b ("), "{e}");
    // /v1's refusal is the HTTP status
    let (st, v) = b.raw("POST", "/v1/chat/completions", "", &json!({"model": "a", "messages": msg}));
    assert_eq!(st, 503, "{v}");
    // a pin neither
    b.ok("system.unload", json!({"model": "b"}));
    b.ok("system.pin", json!({"model": "b"}));
    let e = b.err("llm.chat", json!({"model": "a", "messages": msg}));
    assert!(e["message"].as_str().unwrap().contains("pinned: b (unpin to free the room)"), "{e}");
    let (_, v) = b.raw("GET", "/chat/held", "", &json!({}));
    assert_eq!(v, json!({"held": true, "model": "b"}), "the video daemon's --door check");
}

#[test]
fn a_pinned_or_held_server_is_told_to_stay_and_a_plain_one_is_not() {
    let b = start("stay", vec![image("p", 8.0), image("h", 8.0)], None, None);
    b.ok("system.pin", json!({"model": "p"}));
    assert_eq!(b.make("h", json!({"rsvp_s": 600}))["ok"], true);
    // any POST counts as work to a server's --idle-exit: both of the box's image servers hear it
    b.until("the pinned model's server told to stay", 40, |b| b.seen().lines().filter(|l| l.contains("/api/keepalive")).count() >= 2);
    b.ok("system.unpin", json!({"model": "p"}));
    b.ok("system.unload", json!({"model": "h"}));
    let before = b.seen().lines().filter(|l| l.contains("/api/keepalive")).count();
    std::thread::sleep(Duration::from_secs(34));
    assert_eq!(b.seen().lines().filter(|l| l.contains("/api/keepalive")).count(), before, "an unpinned model idles out as any other");
}

#[test]
fn maintenance_refuses_a_pin_and_reloads_none() {
    let b = start("maint", vec![image("p", 8.0)], None, None);
    b.ok("system.pin", json!({"model": "p"}));
    b.ok("system.maintenance", json!({"on": true, "message": "kernel work", "unload": true}));
    b.until("maintenance's unload takes a pinned model too", 10, |b| !b.up("p"));
    assert_eq!(b.err("system.pin", json!({"model": "p"}))["code"], "maintenance");
    assert_eq!(b.err("image.generate", json!({"model": "p", "prompt": "x"}))["code"], "maintenance");
    assert!(b.rpc("system.eta", json!({"method": "image.generate", "params": {"model": "p"}}))["ok"] == true, "a question is no work");
    std::thread::sleep(Duration::from_secs(8));
    assert!(!b.up("p"), "not reloaded while the box is set aside");
    b.ok("system.maintenance", json!({"on": false}));
    b.until("reloaded when the mode ends", 70, |b| b.up("p"));
    let _ = &b.home;
}
