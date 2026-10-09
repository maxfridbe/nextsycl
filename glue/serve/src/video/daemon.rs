//! `nextsycl video daemon`: the video engine service, in its container - H3's `h3d daemon` as it was, engine-agnostic
//! (it knows engines only through their worker processes, `nextsycl video worker`, and their models by registry id). It keeps the job queue and answers the engine's API on a Unix
//! socket; it never opens a GPU. Each GPU it was given (`--all`, the default, or `--gpu N` for each) is a slot with its
//! own engine process, `nextsycl video worker --gpu N --model ID`, which the daemon starts when a job needs that GPU: it loads
//! the model and keeps it for the next jobs. The daemon ends it when told to (`unload`, `shutdown`) or after a while
//! without jobs, so a GPU shared with other programs is not held for nothing - and the memory comes back with the
//! process, whatever state the driver was in. A crash in an engine ends that worker, not the daemon: the job is marked
//! failed and the next one starts a fresh worker.
//!
//! Three kinds of process, two kinds of IPC:
//!
//! ```text
//!   sycl-h3 (the command line, on the host) --.
//!                                             +-- HTTP/JSON over the Unix socket --> h3d daemon --pipes--> h3d worker (GPU N)
//!   sycl-h3 web service (its own container) --'                                               (JSON lines)
//! ```
//!
//! The model: `--model ID` (a registry entry) unless a job names another one given as `--engine ID` (`"engine":
//! "minimax-h3-q4km"`: a GGUF form that holds less of the card). A slot whose worker holds another model than the
//! job's ends that worker and starts one with the job's; a shared GPU keeps its hooks meanwhile.
//!
//! Jobs wait in one queue and run on whichever GPU slot is free, in the order they came - an unshared GPU before a
//! shared one (below); a job that names a GPU (`"gpu": 1`) waits for that one. A job can be cancelled; it stops at the next block boundary, never inside a kernel
//! (a GPU process stopped mid-kernel can leave the xe driver stuck).
//!
//! The API (JSON over HTTP on the socket):
//!
//! ```text
//!   GET  /engine/status              every GPU slot (engine state, process, memory, busy, its job), the queue
//!   GET  /engine/gpus                the GPUs there are (served or not)
//!   GET  /engine/jobs                every job, newest first, without logs
//!   GET  /engine/jobs/<id>           one job with its log and result
//!   POST /engine/jobs                {"kind": "bench-blocks", ..., "gpu": 1 (optional)}  ->  {"id": 3}
//!   POST /engine/jobs/<id>/cancel
//!   POST /engine/jobs/<id>/remove    forget a job that is not running
//!   POST /engine/unload              {"gpu": N} or every GPU: give it back now (after the running job)
//!   POST /engine/shutdown            cancel the queue, finish or cancel the running jobs, unload, exit
//! ```
//!
//! On a GPU that another program normally holds (`--shared-gpu N`; by default every GPU counts as shared when the
//! hooks are given), two hooks make the daemon share it politely before loading: a lock file (wait until it is free,
//! then hold it while an engine on a shared GPU is loaded) and a front end's model switcher (ask the model it serves to
//! stop, and put it back after the last such engine ended). Either way the engine process then waits until its card
//! really has the memory free - it never loads into a card that is still occupied.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nextsycl_core::{Error, Result};
use serde_json::{json, Value};

use crate::http;
use crate::video::signals;

pub struct Options {
    /// The Unix socket the API answers on.
    pub socket: PathBuf,
    /// the default model: its registry id (the name jobs use, `"engine"`), and the file it loads (status shows it)
    pub model: PathBuf,
    pub model_name: String,
    /// the other models by id, with their files
    pub engines: Vec<(String, PathBuf)>,
    /// The GPUs to serve (indices of `h3 gpus`); `None`: every one there is.
    pub gpus: Option<Vec<usize>>,
    /// The GPUs the hooks below are about; `None`: every served GPU.
    pub shared_gpus: Option<Vec<usize>>,
    /// Unload after this long without a job; `None`: stay loaded until told.
    pub idle: Option<Duration>,
    /// A lock file shared with the GPU's other users: wait while someone else's name is in it.
    pub gpu_lock: Option<PathBuf>,
    /// A front end's model switcher (`POST {"mode": ...}`, answers `{"result": {"mode": ...}}`): its model is asked
    /// to stop before loading and restored after unloading.
    pub llm_switcher: Option<String>,
    pub threads: usize,
}

/// An engine process and its event stream.
struct Proc {
    child: Child,
    stdin: ChildStdin,
    events: mpsc::Receiver<Value>,
    /// it holds a share of the hooks (lock file, the front end's model)
    shared: bool,
    /// the denoiser it loaded (an engine name)
    engine: String,
}

struct JobRec {
    id: u64,
    spec: Value,
    state: &'static str, // queued, running, done, failed, cancelled
    log: Vec<String>,
    result: Option<Value>,
    error: Option<String>,
    cancel: Arc<AtomicBool>,
    created: f64,
    started: Option<f64>,
    finished: Option<f64>,
    /// (done, total) as the job last reported it
    progress: Option<(u64, u64)>,
    /// the GPU it runs or ran on
    gpu: Option<usize>,
}

impl JobRec {
    fn summary(&self) -> Value {
        json!({"id": self.id, "kind": self.spec.get("kind"), "state": self.state, "created": self.created,
               "started": self.started, "finished": self.finished, "error": self.error,
               "progress": self.progress.map(|(d, t)| json!({"done": d, "total": t})),
               "gpu": self.gpu.or_else(|| self.wants_gpu()), "last": self.log.last()})
    }
    fn full(&self) -> Value {
        let mut v = self.summary();
        v["spec"] = self.spec.clone();
        v["log"] = json!(self.log);
        v["result"] = self.result.clone().unwrap_or(Value::Null);
        v
    }
    fn wants_gpu(&self) -> Option<usize> {
        self.spec.get("gpu").and_then(|g| g.as_u64()).map(|g| g as usize)
    }
}

impl Options {
    /// The denoiser checkpoint for an engine name (`None`: the default).
    fn engine_path(&self, name: Option<&str>) -> Option<(String, PathBuf)> {
        match name {
            None => Some((self.model_name.clone(), self.model.clone())),
            Some(n) if n == self.model_name => Some((self.model_name.clone(), self.model.clone())),
            Some(n) => self.engines.iter().find(|(e, _)| e == n).cloned(),
        }
    }
    /// The name of the engine a worker reported loading (its `model` path).
    fn engine_of_path(&self, model: Option<&Value>) -> Option<String> {
        let m = PathBuf::from(model?.as_str()?);
        self.engine_names().into_iter().find(|n| self.engine_path(Some(n)).is_some_and(|(_, p)| p == m))
    }
    fn engine_names(&self) -> Vec<String> {
        std::iter::once(self.model_name.clone()).chain(self.engines.iter().map(|(n, _)| n.clone())).collect()
    }
}

fn job_engine(spec: &Value) -> Option<&str> {
    spec.get("engine").and_then(|e| e.as_str())
}

/// One GPU and the engine process on it.
struct Slot {
    gpu: usize,
    name: String,
    pci: String,
    mem_gib: f64,
    shared: bool,
    engine: String, // unloaded, starting..., waiting for the GPU..., loading, loaded, unloading
    running: Option<u64>,
    last_active: Instant,
    unload_requested: bool,
    info: Value,
    worker_pid: Option<u32>,
    /// the engine's last report: its GPU memory, the card's free memory
    stats: Value,
    /// the card's idle counter at the last status request: (when, idle ms)
    busy_sample: Option<(Instant, u64)>,
}

struct Shared {
    jobs: BTreeMap<u64, JobRec>,
    queue: VecDeque<u64>,
    next: u64,
    shutdown: bool,
    slots: Vec<Slot>,
}

/// The hooks are taken by the first engine on a shared GPU and given back after the last one.
#[derive(Default)]
struct Hooks {
    users: usize,
    saved_mode: Option<String>,
}

struct Daemon {
    opts: Options,
    /// the GPUs `h3d gpus` found at start
    found: Vec<Value>,
    s: Mutex<Shared>,
    cv: Condvar,
    hooks: Mutex<Hooks>,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl Daemon {
    fn log_job(&self, id: u64, line: String) {
        eprintln!("[job {id}] {line}");
        if let Some(j) = self.s.lock().unwrap().jobs.get_mut(&id) {
            j.log.push(line);
        }
    }

    fn set_engine(&self, k: usize, state: impl Into<String>) {
        let st = state.into();
        let mut s = self.s.lock().unwrap();
        eprintln!("[gpu {}] {st}", s.slots[k].gpu);
        s.slots[k].engine = st;
    }

    fn stopping(&self, cancel: &AtomicBool) -> bool {
        cancel.load(Ordering::Relaxed) || self.s.lock().unwrap().shutdown
    }

    /// The lock file: wait until it is absent or ours, then write our name into it.
    fn take_lock(&self, k: usize, cancel: &AtomicBool) -> Result<()> {
        let Some(path) = &self.opts.gpu_lock else { return Ok(()) };
        loop {
            match std::fs::read_to_string(path).ok() {
                Some(h) if !h.starts_with("sycl-h3 ") => {
                    self.set_engine(k, format!("waiting for the GPU lock (held: {})", h.trim()));
                    if self.stopping(cancel) {
                        return Err(Error("cancelled while waiting for the GPU lock".into()));
                    }
                    std::thread::sleep(Duration::from_secs(5));
                }
                _ => {
                    std::fs::write(path, format!("sycl-h3 {} engine loaded\n", now() as u64)).map_err(|e| Error(e.to_string()))?;
                    return Ok(());
                }
            }
        }
    }

    fn release_lock(&self) {
        if let Some(path) = &self.opts.gpu_lock {
            if std::fs::read_to_string(path).is_ok_and(|h| h.starts_with("sycl-h3 ")) {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// Asks the front end's model to stop; returns the mode to restore later.
    fn switch_llm_off(&self) -> Result<Option<String>> {
        let Some(url) = &self.opts.llm_switcher else { return Ok(None) };
        let (host, path) = http::split_url(url).map_err(Error)?;
        let cur = http::call(&http::Target::Tcp(host.clone()), "POST", &path, Some(&json!({}))).map_err(Error)?;
        let mode = cur.pointer("/result/mode").and_then(|m| m.as_str()).unwrap_or("none").to_string();
        if mode != "none" {
            http::call(&http::Target::Tcp(host), "POST", &path, Some(&json!({"mode": "none"}))).map_err(Error)?;
        }
        Ok(Some(mode))
    }

    fn restore_llm(&self, mode: &Option<String>) {
        let (Some(url), Some(mode)) = (&self.opts.llm_switcher, mode) else { return };
        if mode == "none" {
            return;
        }
        match http::split_url(url).and_then(|(h, p)| http::call(&http::Target::Tcp(h), "POST", &p, Some(&json!({"mode": mode})))) {
            Ok(_) => eprintln!("[daemon] the front end's model {mode:?} restored"),
            Err(e) => eprintln!("[daemon] could not restore the front end's model {mode:?}: {e}"),
        }
    }

    /// The first engine on a shared GPU takes the lock and stops the front end's model.
    fn acquire_hooks(&self, k: usize, cancel: &AtomicBool) -> Result<()> {
        let mut h = self.hooks.lock().unwrap();
        if h.users == 0 {
            self.take_lock(k, cancel)?;
            match self.switch_llm_off() {
                Ok(m) => h.saved_mode = m,
                Err(e) => {
                    self.release_lock();
                    return Err(e);
                }
            }
        }
        h.users += 1;
        Ok(())
    }

    /// The last engine on a shared GPU gives them back.
    fn release_hooks(&self) {
        let mut h = self.hooks.lock().unwrap();
        h.users = h.users.saturating_sub(1);
        if h.users == 0 {
            let mode = h.saved_mode.take();
            self.restore_llm(&mode);
            self.release_lock();
        }
    }

    /// Hooks (cheap, no GPU) when the slot's GPU is shared, then the engine process on that GPU, which waits for the
    /// card's memory and loads.
    /// `held`: the slot's share of the hooks is still taken (a worker with another denoiser just ended).
    fn start_worker(&self, k: usize, cancel: &AtomicBool, engine: &str, held: bool) -> Result<Proc> {
        let (gpu, shared) = {
            let s = self.s.lock().unwrap();
            (s.slots[k].gpu, s.slots[k].shared)
        };
        let (engine, model) = self.opts.engine_path(Some(engine)).ok_or_else(|| Error(format!("no engine {engine:?}")))?;
        if shared && !held {
            self.acquire_hooks(k, cancel)?;
        }
        let _ = model; // the worker finds the file by the id
        let spawned = Command::new(program()?)
            .args(["video", "worker", "--gpu", &gpu.to_string(), "--model", &engine])
            .args(["--threads", &self.opts.threads.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                if shared {
                    self.release_hooks();
                }
                return Err(Error(format!("cannot start the engine process: {e}")));
            }
        };
        let (tx, events) = mpsc::channel();
        let out = child.stdout.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let stdin = child.stdin.take().unwrap();
        self.set_engine(k, "starting the engine process");
        self.s.lock().unwrap().slots[k].worker_pid = Some(child.id());
        Ok(Proc { child, stdin, events, shared, engine })
    }

    /// What a worker said that is not about a job's result.
    fn engine_event(&self, k: usize, ev: &Value) {
        match ev["event"].as_str() {
            Some("engine") => {
                if let Some(st) = ev["state"].as_str() {
                    self.set_engine(k, st);
                }
            }
            Some("ready") => {
                self.s.lock().unwrap().slots[k].info = ev["info"].clone();
                self.set_engine(k, "loaded");
            }
            Some("stats") => self.s.lock().unwrap().slots[k].stats = ev.clone(),
            Some("progress") => {
                if let (Some(id), Some(d), Some(t)) = (ev["job"].as_u64(), ev["done"].as_u64(), ev["total"].as_u64()) {
                    if let Some(j) = self.s.lock().unwrap().jobs.get_mut(&id) {
                        j.progress = Some((d, t));
                    }
                }
            }
            _ => {}
        }
    }

    /// After a worker ended (asked or not): the slot forgets it, the hooks get their share back (unless `keep`: another
    /// worker follows at once).
    fn worker_gone(&self, k: usize, p: Proc, keep: bool) {
        {
            let mut s = self.s.lock().unwrap();
            let sl = &mut s.slots[k];
            sl.info = Value::Null;
            sl.stats = Value::Null;
            sl.worker_pid = None;
        }
        self.set_engine(k, "unloaded");
        // the hooks last: the model switcher starts its model only once our status shows the card free (2026-10-05:
        // released first, the studio still saw this worker and left the chat model off until its idle watchdog)
        if p.shared && !keep {
            self.release_hooks();
        }
    }

    /// Ends a worker: asks it to exit (it finishes the kernel it is in, unloads, ends) and waits for it. Never a kill:
    /// a GPU process stopped mid-kernel can leave the driver stuck.
    fn stop_worker(&self, k: usize, proc: &mut Option<Proc>) {
        self.end_worker(k, proc, false);
    }

    fn end_worker(&self, k: usize, proc: &mut Option<Proc>, keep_hooks: bool) {
        let Some(mut p) = proc.take() else { return };
        self.set_engine(k, "unloading");
        let _ = writeln!(p.stdin, "{}", json!({"exit": true}));
        let _ = p.stdin.flush();
        while let Ok(ev) = p.events.recv() {
            self.engine_event(k, &ev);
        }
        let status = p.child.wait();
        eprintln!("[gpu {k}] the engine process ended ({})", status.map_or("unknown".into(), |s| s.to_string()));
        self.worker_gone(k, p, keep_hooks);
    }

    /// A worker that ended on its own (a crash): clean up after it.
    fn reap_if_dead(&self, k: usize, proc: &mut Option<Proc>) -> Option<String> {
        let status = proc.as_mut()?.child.try_wait().ok()??;
        let p = proc.take().unwrap();
        self.worker_gone(k, p, false);
        Some(format!("the engine process ended unexpectedly ({status})"))
    }

    /// Hands a job to the slot's worker and relays its events until the result.
    fn run_job(&self, k: usize, proc: &mut Option<Proc>, id: u64, spec: &Value, cancel: &AtomicBool) -> std::result::Result<Value, String> {
        let want = self.opts.engine_path(job_engine(spec)).map(|(n, _)| n).ok_or("no such engine")?;
        let mut held = false;
        if let Some(p) = proc.as_ref().filter(|p| p.engine != want) {
            self.log_job(id, format!("engine : the GPU holds {}; loading {want} instead", p.engine));
            held = p.shared;
            self.end_worker(k, proc, true);
        }
        if proc.is_none() {
            match self.start_worker(k, cancel, &want, held) {
                Ok(p) => *proc = Some(p),
                Err(e) => {
                    if held {
                        self.release_hooks();
                    }
                    return Err(e.0);
                }
            }
        }
        let p = proc.as_mut().unwrap();
        writeln!(p.stdin, "{}", json!({"job": id, "spec": spec})).map_err(|e| format!("the engine process does not listen: {e}"))?;
        let _ = p.stdin.flush();
        let mut cancel_sent = false;
        loop {
            if cancel.load(Ordering::Relaxed) && !cancel_sent {
                let p = proc.as_mut().unwrap();
                let _ = writeln!(p.stdin, "{}", json!({"cancel": id}));
                let _ = p.stdin.flush();
                cancel_sent = true;
            }
            let ev = proc.as_ref().unwrap().events.recv_timeout(Duration::from_millis(200));
            match ev {
                Ok(ev) => match ev["event"].as_str() {
                    Some("log") => self.log_job(id, ev["line"].as_str().unwrap_or("").to_string()),
                    Some("result") if ev["job"].as_u64() == Some(id) => {
                        return if ev["ok"].as_bool() == Some(true) {
                            Ok(ev["value"].clone())
                        } else {
                            Err(ev["error"].as_str().unwrap_or("failed").to_string())
                        };
                    }
                    _ => {
                        if let Some(l) = ev["line"].as_str() {
                            self.log_job(id, l.to_string());
                        }
                        self.engine_event(k, &ev);
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // its output closed: it has ended or is ending
                    for _ in 0..600 {
                        if let Some(why) = self.reap_if_dead(k, proc) {
                            return Err(why);
                        }
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    return Err("the engine process closed its output but did not end".into());
                }
            }
        }
    }

    /// The thread that keeps slot `k`: takes the jobs it may run, starts and ends its engine process.
    fn slot_thread(self: Arc<Self>, k: usize) {
        enum Next {
            Run(u64, Value, Arc<AtomicBool>),
            Unload,
            Tick,
            Exit,
        }
        let mut proc: Option<Proc> = None;
        loop {
            if let Some(why) = self.reap_if_dead(k, &mut proc) {
                eprintln!("[gpu {k}] {why}");
            }
            if let Some(p) = &proc {
                while let Ok(ev) = p.events.try_recv() {
                    self.engine_event(k, &ev);
                }
            }
            let next = {
                let mut s = self.s.lock().unwrap();
                loop {
                    if s.shutdown {
                        break Next::Exit;
                    }
                    if s.slots[k].unload_requested {
                        s.slots[k].unload_requested = false;
                        break Next::Unload;
                    }
                    let gpu = s.slots[k].gpu;
                    // a shared GPU takes a job that names no GPU only while every unshared one is busy: loading there
                    // stops the other program's model, which a free unshared GPU would not
                    let unshared_free = s.slots[k].shared && s.slots.iter().any(|o| !o.shared && o.running.is_none());
                    let mine = s.queue.iter().position(|id| match s.jobs[id].wants_gpu() {
                        Some(g) => g == gpu,
                        None => !unshared_free,
                    });
                    if let Some(pos) = mine {
                        let id = s.queue.remove(pos).unwrap();
                        let j = s.jobs.get_mut(&id).unwrap();
                        j.state = "running";
                        j.started = Some(now());
                        j.gpu = Some(gpu);
                        let (spec, cancel) = (j.spec.clone(), j.cancel.clone());
                        s.slots[k].running = Some(id);
                        break Next::Run(id, spec, cancel);
                    }
                    let idle = s.slots[k].last_active.elapsed();
                    if let Some(limit) = self.opts.idle {
                        if proc.is_some() && idle >= limit {
                            break Next::Unload;
                        }
                    }
                    let (guard, timeout) = self.cv.wait_timeout(s, Duration::from_secs(1)).unwrap();
                    s = guard;
                    if timeout.timed_out() && proc.is_some() {
                        break Next::Tick; // look at the process and drain its events
                    }
                }
            };
            match next {
                Next::Exit => {
                    self.stop_worker(k, &mut proc);
                    return;
                }
                Next::Unload => self.stop_worker(k, &mut proc),
                Next::Tick => {}
                Next::Run(id, spec, cancel) => {
                    let outcome = self.run_job(k, &mut proc, id, &spec, &cancel);
                    let mut s = self.s.lock().unwrap();
                    let j = s.jobs.get_mut(&id).unwrap();
                    j.finished = Some(now());
                    match outcome {
                        Ok(v) => {
                            j.state = "done";
                            if let Some((_, t)) = j.progress {
                                j.progress = Some((t, t)); // the last report is the start of the last step
                            }
                            j.result = Some(v);
                        }
                        Err(e) => {
                            j.state = if cancel.load(Ordering::Relaxed) { "cancelled" } else { "failed" };
                            j.error = Some(e);
                        }
                    }
                    eprintln!("[job {id}] {}", j.state);
                    s.slots[k].running = None;
                    s.slots[k].last_active = Instant::now();
                    self.cv.notify_all();
                }
            }
        }
    }

    /// One request on the socket.
    fn route(&self, stream: &std::os::unix::net::UnixStream, req: http::Request) -> Result<()> {
        let (status, body) = match (req.path.strip_prefix("/engine/"), req.json()) {
            (Some(rest), Ok(v)) => self.handle(&req.method, rest, v),
            (None, _) => (404, json!({"error": format!("no route {} {}", req.method, req.path)})),
            (_, Err(e)) => (400, json!({"error": e})),
        };
        http::respond(stream, status, &body);
        Ok(())
    }

    fn handle(&self, method: &str, path: &str, body: Value) -> (u16, Value) {
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let mut s = self.s.lock().unwrap();
        match (method, parts.as_slice()) {
            ("GET", ["status"]) => {
                let idle_limit = self.opts.idle;
                let mut gpus = Vec::new();
                for k in 0..s.slots.len() {
                    let running = s.slots[k].running.and_then(|r| s.jobs.get(&r)).map(|j| {
                        let mut v = j.summary();
                        v["elapsed_seconds"] = json!(j.started.map(|t| now() - t));
                        v
                    });
                    let sl = &mut s.slots[k];
                    let idle = sl.last_active.elapsed().as_secs();
                    let busy = gpu_busy(&sl.pci, &mut sl.busy_sample);
                    let unload_in = match (idle_limit, sl.worker_pid, sl.running) {
                        (Some(limit), Some(_), None) => Some(limit.as_secs().saturating_sub(idle)),
                        _ => None,
                    };
                    gpus.push(json!({
                        "gpu": sl.gpu, "name": sl.name, "pci": sl.pci, "mem_gib": sl.mem_gib, "shared": sl.shared,
                        "engine": sl.engine, "info": sl.info, "denoiser": self.opts.engine_of_path(sl.info.get("model")),
                        "worker": sl.worker_pid.map(|pid| json!({"pid": pid, "rss_gib": rss_gib(pid)})),
                        "busy_pct": busy, "engine_gib": sl.stats["gib_in_use"], "cap_gib": sl.stats["gib_cap"],
                        "card_free_gib": sl.stats["gib_free_card"], "running": running,
                        "idle_seconds": if sl.running.is_some() { 0 } else { idle }, "unload_in_seconds": unload_in}));
                }
                (200, json!({"version": crate::VERSION, "model": self.opts.model, "engines": self.opts.engine_names().iter().filter_map(|n| self.opts.engine_path(Some(n))).map(|(n, p)| json!({
                                 "name": n, "path": p, "gib": std::fs::metadata(&p).map_or(0.0, |m| m.len() as f64 / (1u64 << 30) as f64),
                                 "default": n == self.opts.model_name})).collect::<Vec<_>>(), "gpus": gpus,
                             "queued": s.queue.iter().collect::<Vec<_>>(),
                             "unload_after_idle_seconds": self.opts.idle.map(|d| d.as_secs())}))
            }
            ("GET", ["gpus"]) => {
                let served: Vec<usize> = s.slots.iter().map(|sl| sl.gpu).collect();
                let list: Vec<Value> = self
                    .found
                    .iter()
                    .map(|g| {
                        let mut g = g.clone();
                        g["served"] = json!(g["index"].as_u64().is_some_and(|i| served.contains(&(i as usize))));
                        g
                    })
                    .collect();
                (200, json!(list))
            }
            ("GET", ["jobs"]) => (200, json!(s.jobs.values().rev().map(JobRec::summary).collect::<Vec<_>>())),
            ("GET", ["jobs", id]) => match id.parse::<u64>().ok().and_then(|i| s.jobs.get(&i)) {
                Some(j) => (200, j.full()),
                None => (404, json!({"error": format!("no job {id}")})),
            },
            ("POST", ["jobs"]) => {
                if s.shutdown {
                    return (409, json!({"error": "the daemon is shutting down"}));
                }
                if body.get("kind").and_then(|k| k.as_str()).is_none() {
                    return (400, json!({"error": "a job needs a \"kind\" (bench-blocks, check-block, denoise, decode, encode, generate)"}));
                }
                if let Some(e) = body.get("engine") {
                    if !e.as_str().is_some_and(|e| self.opts.engine_path(Some(e)).is_some()) {
                        return (400, json!({"error": format!("\"engine\": {e} is not one this daemon has ({:?})", self.opts.engine_names())}));
                    }
                }
                if let Some(g) = body.get("gpu") {
                    let served: Vec<usize> = s.slots.iter().map(|sl| sl.gpu).collect();
                    if !g.as_u64().is_some_and(|g| served.contains(&(g as usize))) {
                        return (400, json!({"error": format!("\"gpu\": {g} is not one this daemon serves ({served:?})")}));
                    }
                }
                let id = s.next;
                s.next += 1;
                s.jobs.insert(id, JobRec { id, spec: body, state: "queued", log: Vec::new(), result: None, error: None,
                                           cancel: Arc::new(AtomicBool::new(false)), created: now(), started: None,
                                           finished: None, progress: None, gpu: None });
                s.queue.push_back(id);
                self.cv.notify_all();
                (200, json!({"id": id}))
            }
            ("POST", ["jobs", id, "cancel"]) => {
                let Some(i) = id.parse::<u64>().ok() else { return (404, json!({"error": format!("no job {id}")})) };
                s.queue.retain(|q| *q != i);
                match s.jobs.get_mut(&i) {
                    Some(j) if j.state == "queued" => {
                        j.state = "cancelled";
                        j.finished = Some(now());
                        (200, json!({"id": i, "state": "cancelled"}))
                    }
                    Some(j) if j.state == "running" => {
                        j.cancel.store(true, Ordering::Relaxed);
                        (200, json!({"id": i, "state": "cancelling: it stops at the next block boundary"}))
                    }
                    Some(j) => (200, json!({"id": i, "state": j.state})),
                    None => (404, json!({"error": format!("no job {i}")})),
                }
            }
            ("POST", ["jobs", id, "remove"]) => {
                let Some(i) = id.parse::<u64>().ok() else { return (404, json!({"error": format!("no job {id}")})) };
                match s.jobs.get(&i).map(|j| j.state) {
                    None => (404, json!({"error": format!("no job {i}")})),
                    Some("running") => (409, json!({"error": format!("job {i} is running: stop it first")})),
                    Some(_) => {
                        s.queue.retain(|q| *q != i);
                        s.jobs.remove(&i);
                        (200, json!({"id": i, "removed": true}))
                    }
                }
            }
            ("POST", ["unload"]) => {
                let want = body.get("gpu").and_then(|g| g.as_u64()).map(|g| g as usize);
                let mut hit = Vec::new();
                for sl in s.slots.iter_mut().filter(|sl| want.is_none_or(|g| g == sl.gpu)) {
                    sl.unload_requested = true;
                    hit.push(json!({"gpu": sl.gpu, "unload": if sl.running.is_some() { "after the running job" } else { "now" }}));
                }
                self.cv.notify_all();
                if hit.is_empty() {
                    (400, json!({"error": "no such GPU served"}))
                } else {
                    (200, json!({"unload": hit}))
                }
            }
            ("POST", ["shutdown"]) => {
                s.shutdown = true;
                for id in std::mem::take(&mut s.queue) {
                    if let Some(j) = s.jobs.get_mut(&id) {
                        j.state = "cancelled";
                    }
                }
                let running: Vec<u64> = s.slots.iter().filter_map(|sl| sl.running).collect();
                for r in running {
                    s.jobs[&r].cancel.store(true, Ordering::Relaxed);
                }
                self.cv.notify_all();
                (200, json!({"shutdown": "running jobs stop at their next block boundary, then the engines unload"}))
            }
            _ => (404, json!({"error": format!("no route {method} /engine/{path}")})),
        }
    }
}

/// This program, to start workers with: the path it was started by (a rebuild renames a new binary over it, which the
/// next worker then runs; `current_exe` would name the replaced file, deleted - "No such file or directory")
fn program() -> Result<PathBuf> {
    match std::env::args_os().next().map(PathBuf::from) {
        Some(p) if p.is_absolute() && p.exists() => Ok(p),
        _ => std::env::current_exe().map_err(|e| Error(e.to_string())),
    }
}

/// The GPUs there are, asked of a short-lived `nextsycl video gpus --json` (the daemon itself does not start the GPU
/// runtime)
fn list_gpus() -> Result<Vec<Value>> {
    let out = Command::new(program()?).args(["video", "gpus", "--json"]).stderr(Stdio::inherit()).output().map_err(|e| Error(e.to_string()))?;
    if !out.status.success() {
        return Err(Error("cannot list the GPUs (nextsycl video gpus failed)".into()));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| Error(format!("nextsycl video gpus: {e}")))?;
    Ok(v.as_array().cloned().unwrap_or_default())
}

pub fn serve(opts: Options) -> Result<()> {
    for (name, path) in std::iter::once((&opts.model_name, &opts.model)).chain(opts.engines.iter().map(|(n, p)| (n, p))) {
        if !path.exists() {
            return Err(Error(format!("{}: no such file (model {name})", path.display())));
        }
    }
    let found = list_gpus().unwrap_or_else(|e| {
        eprintln!("[daemon] {e}: serving GPU 0 blind");
        vec![json!({"index": 0, "name": "?", "mem_gib": 0.0, "pci": ""})]
    });
    let indices: Vec<usize> = match &opts.gpus {
        Some(g) => g.clone(),
        None => found.iter().filter_map(|g| g["index"].as_u64()).map(|i| i as usize).collect(),
    };
    if indices.is_empty() {
        return Err(Error("no GPU to serve".into()));
    }
    let hooks = opts.gpu_lock.is_some() || opts.llm_switcher.is_some();
    let mut slots = Vec::new();
    for g in &indices {
        let Some(info) = found.iter().find(|f| f["index"].as_u64() == Some(*g as u64)) else {
            return Err(Error(format!("--gpu {g}: there is no GPU {g} ({} found; see `nextsycl video gpus`)", found.len())));
        };
        let shared = hooks && opts.shared_gpus.as_ref().is_none_or(|s| s.contains(g));
        slots.push(Slot {
            gpu: *g,
            name: info["name"].as_str().unwrap_or("?").to_string(),
            pci: info["pci"].as_str().unwrap_or("").to_string(),
            mem_gib: info["mem_gib"].as_f64().unwrap_or(0.0),
            shared,
            engine: "unloaded".into(),
            running: None,
            last_active: Instant::now(),
            unload_requested: false,
            info: Value::Null,
            worker_pid: None,
            stats: Value::Null,
            busy_sample: None,
        });
    }
    // a socket left by a daemon that did not end cleanly would make bind fail; one that answers means another daemon
    if opts.socket.exists() {
        if std::os::unix::net::UnixStream::connect(&opts.socket).is_ok() {
            return Err(Error(format!("{}: another daemon answers there", opts.socket.display())));
        }
        std::fs::remove_file(&opts.socket).map_err(|e| Error(e.to_string()))?;
    }
    let listener = UnixListener::bind(&opts.socket).map_err(|e| Error(format!("cannot listen on {}: {e}", opts.socket.display())))?;
    eprintln!("nextsycl video daemon {} on {} - model {} (loaded on the first job){}", crate::VERSION, opts.socket.display(), opts.model_name,
              opts.idle.map_or(String::new(), |d| format!(", unloaded after {} s idle", d.as_secs())));
    for (name, path) in &opts.engines {
        eprintln!("  model {name}: {}", path.display());
    }
    for sl in &slots {
        eprintln!("  GPU {}: {} ({:.0} GiB{}){}", sl.gpu, sl.name, sl.mem_gib,
                  if sl.pci.is_empty() { String::new() } else { format!(", {}", sl.pci) },
                  if sl.shared { " - shared: lock file / model switch before loading" } else { "" });
    }
    let n = slots.len();
    let socket = opts.socket.clone();
    let d = Arc::new(Daemon {
        opts,
        found,
        s: Mutex::new(Shared { jobs: BTreeMap::new(), queue: VecDeque::new(), next: 1, shutdown: false, slots }),
        cv: Condvar::new(),
        hooks: Mutex::new(Hooks::default()),
    });
    let threads: Vec<_> = (0..n)
        .map(|k| {
            let d = d.clone();
            std::thread::spawn(move || d.slot_thread(k))
        })
        .collect();
    // SIGTERM (podman stop) -> the same as POST /shutdown
    signals::install();
    listener.set_nonblocking(true).map_err(|e| Error(e.to_string()))?;
    loop {
        if signals::terminated() {
            let mut s = d.s.lock().unwrap();
            if !s.shutdown {
                eprintln!("[daemon] terminate signal: shutting down");
                s.shutdown = true;
                let running: Vec<u64> = s.slots.iter().filter_map(|sl| sl.running).collect();
                for r in running {
                    s.jobs[&r].cancel.store(true, Ordering::Relaxed);
                }
                d.cv.notify_all();
            }
        }
        if threads.iter().all(|t| t.is_finished()) {
            break;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let d = d.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_nonblocking(false);
                    match http::read_request(&stream) {
                        Ok(req) => {
                            if let Err(e) = d.route(&stream, req) {
                                http::respond(&stream, 500, &json!({"error": e.0}));
                            }
                        }
                        Err(e) => http::respond(&stream, 400, &json!({"error": e})),
                    }
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => eprintln!("[daemon] accept: {e}"),
        }
    }
    for t in threads {
        let _ = t.join();
    }
    let _ = std::fs::remove_file(&socket);
    eprintln!("[daemon] stopped");
    Ok(())
}

/// A card's compute-engine busy share since the last call, from the xe driver's idle counter (the whole card, every
/// process): 100 - (idle ms gained) / (wall ms passed). The card is found by its PCI address; the counter is the one
/// the driver names "gt<N>-rc" (render/compute; the media engine's "-mc" is left out). None when there is no such
/// counter, and on the first call.
fn gpu_busy(pci: &str, last: &mut Option<(Instant, u64)>) -> Option<f64> {
    let tile = if pci.is_empty() {
        // no address: the first card (right on a one-GPU machine)
        std::fs::read_dir("/sys/class/drm")
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("card") && !n.to_string_lossy().contains('-')))
            .map(|p| p.join("device/tile0"))
            .find(|t| t.exists())?
    } else {
        PathBuf::from(format!("/sys/bus/pci/devices/{pci}/tile0"))
    };
    let idle: u64 = std::fs::read_dir(&tile)
        .ok()?
        .flatten()
        .map(|gt| gt.path().join("gtidle"))
        .filter(|g| std::fs::read_to_string(g.join("name")).is_ok_and(|n| n.trim().ends_with("-rc")))
        .filter_map(|g| std::fs::read_to_string(g.join("idle_residency_ms")).ok())
        .filter_map(|v| v.trim().parse::<u64>().ok())
        .next()?;
    let now = Instant::now();
    let out = last.map(|(t, i)| {
        let wall = now.duration_since(t).as_secs_f64() * 1e3;
        (100.0 - idle.saturating_sub(i) as f64 / wall.max(1.0) * 100.0).clamp(0.0, 100.0)
    });
    *last = Some((now, idle));
    out
}

/// A process's resident memory, from /proc.
fn rss_gib(pid: u32) -> Option<f64> {
    let st = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kb: f64 = st.lines().find(|l| l.starts_with("VmRSS:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / (1024.0 * 1024.0))
}
