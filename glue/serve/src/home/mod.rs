//! The front door (`nextsycl serve`, :8000): the box at a glance and one API for everything on it.
//!
//! - **The page** (`/`, wfe/home): every GPU - its memory by process, and the memory the enabled models are vouched
//!   for (hatched; several may vouch for one card: whichever loads first holds it) - and every registered model:
//!   enable it on GPU(s), pin it, its page, its state (no start or stop: requests load, idle or a need for room unloads;
//!   system.load and system.unload remain for tools);
//!   an API tab with the bearer token and a curl for every method; maintenance mode on and off (maint.rs).
//! - **The drives** (disks.rs, crate::drives): each drive the models are on - device, link, free space, a read speed
//!   measured on demand - and moving a model's files to another one (copied, checked, the registry repointed, the old
//!   files removed), one move at a time, never while the model is loaded.
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
mod backplane;
mod config;
mod disks;
mod enroll;
mod history;
mod jobs;
mod life;
mod maint;
mod methods;
mod precheck;
mod pulls;
mod spec;
pub mod registry;

/// What the #[rpc] methods make (methods.rs, rpcgen), as (path under the repo, content): the typed client of the
/// front door's methods for the wfe pages. `nextsycl gen-api` writes them; a test fails when the committed ones
/// are stale.
pub fn api_files() -> Vec<(&'static str, String)> {
    let ts = rpcgen::typescript("nextsycl's front door (POST /rpc/<kind>.<method>; the answer in {ok, result | error})", &methods::infos())
        .expect("the methods describe consistent types");
    vec![("wfe/home/src/gen/nextsycl.ts", ts)]
}

#[cfg(test)]
mod api_files_tests {
    use sha2::{Digest, Sha256};
    use std::path::Path;

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The committed client is what the methods make: a method changed without `nextsycl gen-api` fails here.
    #[test]
    fn the_committed_client_is_up_to_date() {
        for (name, text) in super::api_files() {
            let have = std::fs::read_to_string(root().join(name)).unwrap_or_default();
            assert!(have == text, "{name} is stale: run `cargo run -p nextsycl -- gen-api`");
        }
    }

    /// The document and the client say the same things: every field, limit, default, enum and example is in both.
    #[test]
    fn the_document_and_the_client_agree() {
        if let Err(e) = rpcgen::parity(&super::methods::infos()) {
            panic!("{e}");
        }
    }

    /// vendor/rpcgen is the platform's rpcgen as tooling/sync_rpcgen.sh copied it: the tree hash it recorded is the
    /// copy's (sha256 of the "sha256  path" lines, sorted by path). Edit rpcgen in vibe-creative-platform, then sync.
    #[test]
    fn the_vendored_rpcgen_is_unchanged() {
        let dir = root().join("vendor/rpcgen");
        let mut files = Vec::new();
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") || p.file_name().is_some_and(|n| n == "Cargo.toml") {
                    files.push(p.strip_prefix(&dir).unwrap().to_string_lossy().into_owned());
                }
            }
        }
        files.sort();
        let mut listing = String::new();
        for f in &files {
            let h = Sha256::digest(std::fs::read(dir.join(f)).unwrap());
            listing.push_str(&format!("{}  {f}\n", h.iter().map(|b| format!("{b:02x}")).collect::<String>()));
        }
        let tree: String = Sha256::digest(listing.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        let recorded = std::fs::read_to_string(dir.join("VENDORED")).unwrap_or_default();
        let want = recorded.lines().find_map(|l| l.strip_prefix("tree: ")).unwrap_or_default().trim().to_string();
        assert_eq!(tree, want, "vendor/rpcgen was changed here: change it in vibe-creative-platform/rpcgen, then tooling/sync_rpcgen.sh");
    }
}
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
    pub model3d: u16,
    /// the second image, audio and 3D servers' (a model enabled on two cards runs on both when the work asks; 0: none)
    pub image2: u16,
    pub audio2: u16,
    pub model3d2: u16,
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
    /// where this page listens, as this box reaches it (host:port): the video daemon asks it before stopping chat
    pub door: String,
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
    /// the requests sent on to each image and audio server and not answered yet, by its port
    pub(crate) inflight: Mutex<BTreeMap<u16, usize>>,
    /// what each image, audio and 3D server last said of itself (model, loading, busy), by its port
    answers: Mutex<BTreeMap<u16, (String, bool, bool)>>,
    /// image and audio requests as jobs (jobs.json beside serve.json)
    jobs: jobs::Jobs,
    /// a job's stage while its server is chosen, and what it waits for: job id -> (stage, waiting_for)
    stages: Mutex<BTreeMap<String, (String, String)>>,
    /// the holds (crate::hold: rsvp_s, X-Keep-Up-S, keep_up_s) in holds.json beside the registry, this door their one writer
    holds_file: PathBuf,
    holds_mx: Mutex<()>,
    /// history/*.jsonl beside serve.json: what work took (history.rs)
    history: history::History,
    /// the clips watched for a limit or a reservation, and the ones withdrawn at their limit, by label (backplane.rs)
    video_watch: Mutex<Vec<backplane::Watch>>,
    withdrawn: Mutex<BTreeMap<String, Value>>,
    /// the last chat request here
    chat_last: Mutex<f64>,
    /// the cards given from outside (fix_cards), not sampled
    cards_fixed: std::sync::atomic::AtomicBool,
    /// loads started here and not yet answering (or answering a moment ago): card, server port, MiB placed for, when
    /// it answered - what they have yet to take counts as taken (life.rs free_mb)
    pending: Mutex<Vec<Pending>>,
    /// the video queue paused while a GPU run held the cards (crate::gpulock): what to put back when it ends
    lock_paused: Mutex<Option<Value>>,
    /// the progress of running jobs no server runs (an enrolment: enroll.rs), by job id
    progress: Mutex<BTreeMap<String, Value>>,
    /// maintenance mode's notice (crate::maintenance: beside the registry, every service reads it) and the key that
    /// lets a request through it (kept in the state file; empty: none)
    maint_file: PathBuf,
    maint_bypass: Mutex<String>,
    /// the drives: their measured speeds (kept in the state file), a measurement and a move under way (disks.rs)
    disks: disks::Disks,
    /// models being downloaded, and the last ones (pulls.rs)
    pulls: pulls::Pulls,
}

/// The cards as this page sampled them, with what the root sampler's file (`nextsycl gpustat`: one card at its top
/// level, every card in `cards`) saw used where that is more - processes of other users this page cannot read; the
/// difference is `vram_unseen_mb`. A reading older than 15 s is left out.
fn merge_gpustat(cards: &mut [Value], st: &Value, now: f64) {
    if st["ts"].as_f64().is_none_or(|ts| (now - ts).abs() > 15.0) {
        return;
    }
    let mut seen: Vec<&Value> = vec![st];
    seen.extend(st["cards"].as_array().into_iter().flatten());
    for s in seen {
        let (Some(pci), Some(used)) = (s["pci"].as_str(), s["vram_used_mb"].as_u64()) else { continue };
        for c in cards.iter_mut().filter(|c| c["pci"] == pci) {
            let mine = c["vram_used_mb"].as_u64().unwrap_or(0);
            if used > mine {
                c["vram_used_mb"] = json!(used);
                c["vram_unseen_mb"] = json!(used - mine);
            }
        }
    }
}

/// A load under way: card, server port, MiB placed for, when it answered
type Pending = (usize, u16, u64, Option<std::time::Instant>);

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
    let kinds = ["llm", "image", "video", "audio", "model3d"];
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
        let jobs = jobs::Jobs::load(state_file.with_file_name("jobs.json"));
        let maint_file = crate::maintenance::file_beside(&registry::registry(&o.cfg));
        let maint_bypass = Mutex::new(st["maintenance_bypass"].as_str().unwrap_or("").to_string());
        let holds_file = crate::hold::file_beside(&registry::registry(&o.cfg));
        let history = history::History::open(state_file.with_file_name("history"));
        let h = Arc::new(Home {
            o, token, state_file, jobs,
            cards: Mutex::new(json!([])), runs: Mutex::new(Vec::new()), next: Mutex::new(1), loading: Mutex::new(BTreeMap::new()),
            kind_lock: ["llm", "image", "audio", "model3d", "video"].into_iter().map(|k| (k, Mutex::new(()))).collect(),
            seen: Mutex::new(seen), inflight: Mutex::new(BTreeMap::new()), answers: Mutex::new(BTreeMap::new()), stages: Mutex::new(BTreeMap::new()),
            holds_file, holds_mx: Mutex::new(()), history, video_watch: Mutex::new(Vec::new()), withdrawn: Mutex::new(BTreeMap::new()),
            chat_last: Mutex::new(now()), cards_fixed: std::sync::atomic::AtomicBool::new(false),
            pending: Mutex::new(Vec::new()), lock_paused: Mutex::new(None), progress: Mutex::new(BTreeMap::new()),
            maint_file, maint_bypass, disks: disks::Disks::new(&st), pulls: pulls::Pulls::default(),
        });
        let me = h.clone();
        std::thread::spawn(move || me.sample_cards());
        let me = h.clone();
        std::thread::spawn(move || me.video_room_loop());
        let me = h.clone();
        std::thread::spawn(move || me.backplane_loop());
        h.resume_jobs();
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

    /// The cards as given, from now on not sampled: a test's box with no GPU in it
    pub fn fix_cards(&self, cards: Value) {
        self.cards_fixed.store(true, std::sync::atomic::Ordering::Relaxed);
        *self.cards.lock().unwrap() = cards;
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
            let mut v = v;
            // the root sampler's reading (gpustat.json) sees every process's memory; this page only its user's
            let stat = self.o.cfg.get("NS_VIDEO_GPUSTAT").unwrap_or_else(|| "/run/gpustat.json".into());
            if let Some(st) = std::fs::read(&stat).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                merge_gpustat(&mut v, &st, now());
            }
            if !self.cards_fixed.load(std::sync::atomic::Ordering::Relaxed) {
                *self.cards.lock().unwrap() = Value::from(v);
            }
            // a GPU run holding the cards pauses the video queue until it ends (crate::gpulock)
            self.video_hold(self.cards_taken().is_some());
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

    /// The video queue while a GPU run holds the cards: paused when the run starts, put back as it was when it ends
    /// (maintenance on meanwhile: that mode's pause stands, the queue stays as it is until the mode ends)
    fn video_hold(&self, taken: bool) {
        let mut paused = self.lock_paused.lock().unwrap();
        if self.maintenance().is_some() {
            return;
        }
        if taken && paused.is_none() {
            eprintln!("{}: the video queue paused", self.cards_taken().unwrap_or_default());
            *paused = Some(self.video_pause());
        } else if !taken {
            if let Some(r) = paused.take() {
                eprintln!("the GPU run ended: the video queue as it was");
                self.video_restore(&r);
            }
        }
    }

    fn save_state(&self) {
        let st = json!({"token": self.token, "seen": *self.seen.lock().unwrap(), "maintenance_bypass": *self.maint_bypass.lock().unwrap(),
                        "drives": *self.disks.measured.lock().unwrap()});
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
        let req = match http::read_request_at(&mut s) {
            Ok(r) => r,
            // unreadable (too large, cut short): the /rpc envelope (HTTP 200 on /rpc, as every /rpc error; 400 elsewhere),
            // which an OpenAI client reads as error.message too
            Err((path, e)) => {
                let code = if e.contains("over 256 MiB") { "request_too_large" } else { "invalid_request" };
                let method = path.strip_prefix("/rpc/").unwrap_or("");
                let st = if path.starts_with("/rpc/") { 200 } else { 400 };
                return http::respond(&mut s, st, &json!({"ok": false, "error": api::rpc_error(method, 400, Some(code), &e)}));
            }
        };
        let p = req.path.as_str();
        if p.starts_with("/rpc/") || p.starts_with("/v1/") {
            return self.api(s, req);
        }
        if let Some(rest) = p.strip_prefix("/open/") {
            return self.open(&mut s, &req, rest);
        }
        match (req.method.as_str(), p) {
            ("GET", "/") | ("GET", "/index.html") => self.page(&mut s),
            ("GET", p) if p.starts_with("/ui/") || p.starts_with("/static/") => self.asset(&mut s, p),
            ("GET", "/health") => http::respond(&mut s, 200, &json!({"status": "ok"})),
            // for the video daemon, which has no token: read-only and from this box only
            ("GET", "/chat/held") => {
                if !s.same_box() {
                    return http::respond(&mut s, 404, &err("not found".to_string()));
                }
                http::respond(&mut s, 200, &self.chat_held_answer())
            }
            ("GET", "/openapi.yml") | ("GET", "/openapi.yaml") => {
                let spec = self.openapi(&req, false);
                http::respond_bytes(&mut s, 200, "application/yaml; charset=utf-8", yaml::to_yaml(&spec).as_bytes(), "Cache-Control: no-cache\r\n")
            }
            ("GET", "/openapi.json") => http::respond(&mut s, 200, &self.openapi(&req, false)),
            ("GET", "/api/state") => http::respond(&mut s, 200, &self.state()),
            ("GET", "/api/config") => http::respond(&mut s, 200, &self.config()),
            ("GET", "/api/spec") => http::respond(&mut s, 200, &json!({"token": self.token, "spec": self.openapi(&req, true)})),
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

    /// A model's own page (an image, audio or 3D model's web front end, which only its running server has), served
    /// through this door: `/open/<id>/<sub>` starts the model when it is not up - the same as its first request - and
    /// passes the request on to its server as `/<sub>` (any method: the page's own API calls go the same way).
    /// `/open/<id>` alone is sent to `/open/<id>/`, so the page's relative links resolve under it. Those servers check
    /// no token and listen on --serve-host (127.0.0.1): this is how another machine reaches them. A disabled model has
    /// none to open. The video studio is passed on the same way (it is up while a video model is enabled); the chat page is
    /// on the network: the browser is sent there.
    fn open(self: &Arc<Self>, s: &mut Conn, req: &http::Request, rest: &str) {
        let (id, sub) = rest.split_once('/').map_or((rest, None), |(i, r)| (i, Some(r)));
        let m = match registry::find(&self.o.cfg, id) {
            Ok(Some(m)) => m,
            _ => return http::respond(s, 404, &err(format!("no model {id}"))),
        };
        if m["enabled"] == false {
            return http::respond(s, 409, &err(format!("{id} is disabled: enable it to open its page")));
        }
        let query = req.target.find('?').map_or("", |i| &req.target[i..]);
        let port = match registry::kind_of(&m) {
            "image" | "audio" | "model3d" => {
                let Some(sub) = sub else {
                    let to = format!("/open/{id}/{query}");
                    return http::respond_bytes(s, 302, "text/plain; charset=utf-8", b"", &format!("Location: {to}\r\nCache-Control: no-cache\r\n"));
                };
                // a saved voice enrolled for Voxtral: this door's job (enroll.rs), not the page's server
                if sub == "api/voices/enroll" {
                    return self.open_enroll(s, req);
                }
                let lease = match self.ensure(id) {
                    Ok(l) => l,
                    Err((c, e)) => return http::respond(s, c, &err(e)),
                };
                if let Err(e) = http::forward_to(&mut *s, &lease, req, &format!("/{sub}{query}")) {
                    http::respond(s, 502, &err(e));
                }
                return;
            }
            // the studio: up while a video model is enabled (with the daemon); on 127.0.0.1 like the model servers
            "video" => {
                let Some(sub) = sub else {
                    let to = format!("/open/{id}/{query}");
                    return http::respond_bytes(s, 302, "text/plain; charset=utf-8", b"", &format!("Location: {to}\r\nCache-Control: no-cache\r\n"));
                };
                if let Err((c, e)) = self.video_up() {
                    return http::respond(s, c, &err(e));
                }
                let studio = http::Target::Tcp(format!("127.0.0.1:{}", self.o.ports.video));
                if let Err(e) = http::forward_to(&mut *s, &studio, req, &format!("/{sub}{query}")) {
                    http::respond(s, 502, &err(format!("the video studio does not answer: {e}")));
                }
                return;
            }
            _ => self.o.ports.chat_ui,
        };
        let host = req.host.as_deref().unwrap_or("localhost").rsplit_once(':').map_or_else(|| req.host.clone().unwrap_or_default(), |(h, _)| h.to_string());
        let host = if host.is_empty() { "localhost".to_string() } else { host };
        http::respond_bytes(s, 302, "text/plain; charset=utf-8", b"", &format!("Location: http://{host}:{port}/\r\nCache-Control: no-cache\r\n"));
    }

    /// An audio page's enrolments (`api/voices/enroll` under /open/<id>/): GET - the models that speak enrolled voices
    /// and the enrolment jobs; POST {name, model, gpu} - one queued (audio.voices.enroll, as the API's)
    fn open_enroll(self: &Arc<Self>, s: &mut Conn, req: &http::Request) {
        if req.method != "POST" {
            return http::respond(s, 200, &self.enroll_state());
        }
        if let Some(n) = self.maintenance() {
            return http::respond(s, 503, &err(crate::maintenance::message(&n)));
        }
        let b: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
        let name = b["name"].as_str().unwrap_or("").to_string();
        let mut body = json!({"name": name, "wait": false, "label": format!("enroll-{name}-{}", now() as u64)});
        for k in ["model", "gpu", "steps", "replace"] {
            if !b[k].is_null() {
                body[k] = b[k].clone();
            }
        }
        let all = self.models();
        match self.precheck("audio.voices.enroll", &body, &all).and_then(|_| self.submit("audio.voices.enroll", &body)) {
            Ok(v) => http::respond(s, 200, &v),
            Err((c, m)) => http::respond(s, c, &err(m)),
        }
    }

    fn page(&self, s: &mut Conn) {
        let dir = self.o.wfe.join("home");
        match (crate::wfe::text(&self.o.wfe, "home/index.html"), crate::wfe::text(&self.o.wfe, "home/style.css")) {
            (Some(h), Some(c)) => http::respond_bytes(s, 200, "text/html; charset=utf-8", h.replace("__CSS__", &c).as_bytes(), "Cache-Control: no-cache\r\n"),
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
        match crate::wfe::read(&self.o.wfe, rel).ok_or(()) {
            Ok(b) => http::respond_bytes(s, 200, kind, &b, "Cache-Control: no-cache\r\n"),
            Err(_) => http::respond(s, 404, &err(format!("no file {rel}"))),
        }
    }

    /// The front ends and helpers beside the models: up or not, a link
    fn services(&self) -> Vec<Value> {
        let p = self.o.ports.clone();
        // the studio's page through this door (its own port is on 127.0.0.1): under an enabled video model's /open/
        let studio = self.models().into_iter().find(|m| m["kind"] == "video" && m["enabled"] != false)
            .and_then(|m| m["id"].as_str().map(|i| format!("/open/{i}/")));
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
                       "up": up, "link": "/", "href": studio})
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
        let maintenance = self.maintenance().map(|n| {
            let mut v = maint::public(&n);
            v["bypass"] = json!(*self.maint_bypass.lock().unwrap());
            v
        });
        json!({"ts": now(), "host": meminfo(), "cards": cards, "models": models, "services": self.services(), "runs": runs,
               "idle_minutes": self.o.idle_minutes, "ports": {"video": self.o.ports.video, "chat_ui": self.o.ports.chat_ui},
               "maintenance": maintenance, "cards_taken": self.cards_taken(), "drives": self.drives_view(&all, &models),
               // the models a card offers that are not here yet, their downloads, and whether a request fetches one itself
               "available": self.not_downloaded_all(), "pulls": self.pulls_view(), "auto_pull": self.auto_pull()})
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

    /// The page's buttons: {"do": enable | disable | pin | unpin | load | unload | measure | move | move_cancel, "model", "gpus": [..],
    /// "force", "drive"}; a load, a measurement and a move run on (the state shows them), the others answer when done
    fn action(self: &Arc<Self>, b: &Value) -> Result<Value, (u16, String)> {
        let model = b["model"].as_str().unwrap_or("").to_string();
        match b["do"].as_str().unwrap_or("") {
            "enable" => self.enable(&model, b.get("gpus")).map(|m| json!({"model": m})),
            "disable" => self.disable(&model, b["force"] == true).map(|_| json!({"model": model, "enabled": false})),
            "gpus" => self.set_gpus(&model, b.get("gpus")),
            "pin" => self.pin(&model).map(|_| json!({"model": model, "pinned": true, "loading": true})),
            "unpin" => self.unpin(&model).map(|_| json!({"model": model, "pinned": false})),
            "load" => {
                self.load_async(&model)?;
                Ok(json!({"model": model, "loading": true}))
            }
            "unload" => self.unload(&model).map(|_| json!({"model": model, "loaded": false})),
            "config" => self.config_set(b),
            "restart" => self.restart(b["service"].as_str().unwrap_or("")),
            "maintenance" => self.maintenance_set(b),
            "measure" => self.drive_measure(b["drive"].as_str().unwrap_or("")),
            "move" => self.model_move(&model, b["drive"].as_str().unwrap_or("")),
            "move_cancel" => self.move_cancel(),
            "pull" => self.pull_start(&model).map(|p| json!({"pull": p.view()})),
            "pull_cancel" => self.pull_cancel(&model),
            // the page's checkbox: NS_AUTO_PULL in the settings file (read live: no restart)
            "auto_pull" => self.config_set(&json!({"set": {"NS_AUTO_PULL": if b["on"] == true { json!("1") } else { Value::Null }}}))
                .map(|_| json!({"auto_pull": self.auto_pull()})),
            other => Err((400, format!("no action {other:?} (enable, disable, gpus, pin, unpin, load, unload, config, restart, maintenance, measure, move, move_cancel, pull, pull_cancel, auto_pull)"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_samplers_reading_counts_where_it_sees_more() {
        let mut cards = vec![json!({"pci": "0000:03:00.0", "vram_used_mb": 1000, "vram_total_mb": 32768}),
                             json!({"pci": "0000:08:00.0", "vram_used_mb": 9000, "vram_total_mb": 32768})];
        let st = json!({"pci": "0000:03:00.0", "vram_used_mb": 31000, "ts": 100.0,
                        "cards": [{"pci": "0000:03:00.0", "vram_used_mb": 31000}, {"pci": "0000:08:00.0", "vram_used_mb": 4000}]});
        let mut stale = cards.clone();
        merge_gpustat(&mut stale, &st, 200.0);
        assert_eq!(stale, cards, "an old reading is left out");
        merge_gpustat(&mut cards, &st, 104.0);
        assert_eq!((cards[0]["vram_used_mb"].as_u64(), cards[0]["vram_unseen_mb"].as_u64()), (Some(31000), Some(30000)));
        // this page saw more on the second card: its own reading stands
        assert_eq!((cards[1]["vram_used_mb"].as_u64(), cards[1].get("vram_unseen_mb")), (Some(9000), None));
        // an old file with no `cards` (one card at the top level)
        let mut one = vec![json!({"pci": "0000:08:00.0", "vram_used_mb": 0})];
        merge_gpustat(&mut one, &json!({"pci": "0000:08:00.0", "vram_used_mb": 2048, "ts": 5.0}), 5.0);
        assert_eq!(one[0]["vram_used_mb"], 2048);
    }
}
