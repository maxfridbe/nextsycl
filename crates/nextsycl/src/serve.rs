//! `nextsycl serve`: an OpenAI-compatible server for one loaded model (what the box's model switch and Open WebUI
//! talk to).
//!
//!     GET  /health                 {"status": "ok"} once the model is loaded
//!     GET  /v1/models              the one model
//!     GET  /status                 {"busy": bool, "prompt_cache": ...}
//!     POST /v1/chat/completions    messages, max_tokens (none: --max-tokens, else the rest of the context),
//!                                  temperature, top_p, stream, reasoning_effort (or
//!                                  chat_template_kwargs.reasoning_effort): low | high | max; logprobs, top_logprobs
//!                                  (the answer's tokens, as OpenAI's choices[0].logprobs.content); usage.energy_wh;
//!                                  logprob_chain: true - each entry also "chain": its logprob chained through the
//!                                  attention to this turn's own earlier tokens (see chain_entry), and each
//!                                  alternative a "chained_logprob" (decoded one token a pass, no draft block)
//!     POST /api/chat               the same for a web page, simpler: {"messages": [{"role", "content", "thinking"?}],
//!                                  "effort"?, "max_tokens"?, "temperature"?, "top_p"?, "stream"? (default true)};
//!                                  streamed as JSON lines - {"thinking": text} and {"content": text} as they come,
//!                                  then {"done": true, "finish", "prompt_tokens", "reused", "generated",
//!                                  "read_seconds", "generate_seconds", "tok_s", "energy_wh"}; not streamed, one object
//!                                  with "content" and "thinking" beside those; "logprobs": true (+ "top_logprobs")
//!                                  adds the answer tokens' entries ("logprobs" on the lines, or all at the end)
//!
//! A browser may call these from a page on a loopback origin (http://localhost:*, http://127.0.0.1:*) or one
//! `--cors` names (NS_CORS; * = any): the answers carry the CORS headers, OPTIONS answers the preflight.
//!
//! The same routes and the control ones answer on a Unix socket too (`--socket`; `nextsycl start` puts it in
//! $XDG_RUNTIME_DIR/nextsycl), which the host command line talks to (client.rs), as sycl-h3 talks to h3d:
//!
//!     GET  /server/status          the model, its GPUs (memory, layers, expert slots), the request running, the cache
//!     GET  /server/requests        the requests: the running one and the last ones that ended (--keep-requests,
//!                                  100 by default)
//!     GET  /server/requests/<id>   one of them in full: its settings, messages, timings, previews of its prompt and
//!                                  answer
//!     GET  /server/cache           the prompt cache's checkpoints;  POST /server/cache/clear  drops them
//!     POST /server/shutdown        stop once no request runs (as SIGTERM)
//!
//! Up to `--parallel` requests (NS_PARALLEL, default 2) decode together: an engine thread owns that many sessions,
//! reads a new request's prompt into a free one, and steps every active session at once (`Engine::forward_batch`: the
//! weights read once for all of them; each row as its own pass would be) - one active request decodes alone, with
//! the draft block. The rest wait in order. The prompt cache (cache.rs, --prompt-cache-mib) keeps the
//! conversation state at three points of every prompt - the end of its first turn (a system prompt other
//! conversations share), the start of its last user turn (an edited or regenerated message), its end (the next turn
//! of the conversation) - and a request mounts the longest cached prefix of its tokens, or continues the live
//! session when that is longer, and reads only the rest. GLM's thinking comes back as `reasoning_content`, the
//! answer as `content`.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::mpsc;
use std::net::TcpListener;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_runtime::{Decoder, Engine, Session};

use crate::cache::PromptCache;
use crate::telemetry::Telemetry;
use crate::http::{self, Conn};
use ns_tok::{Effort, Message, Tokenizer};
use serde_json::{json, Value};

use crate::{sample, Rng};

static STOP: AtomicBool = AtomicBool::new(false);

pub struct Server {
    /// the model's engine (engines/<arch>, chosen by the file's architecture)
    pub engine: Box<dyn Engine>,
    pub tok: Tokenizer,
    pub name: String,
    /// the largest session's context (a request must fit one)
    pub max_ctx: usize,
    /// each session's context
    slot_ctx: Vec<usize>,
    pub default_effort: Effort,
    /// the requests waiting for a session, and the engine's wake-up
    queue: Mutex<VecDeque<Job>>,
    wake: std::sync::Condvar,
    /// requests taken and not yet ended (the stop waits for none)
    inflight: AtomicU64,
    /// the sessions decoding together at most
    parallel: usize,
    /// the tokens a request without max_tokens may make (None: to the end of the context)
    default_max: Option<usize>,
    /// the prompt cache (the engine thread's while it reads a prompt; status calls only try)
    cache: Mutex<PromptCache>,
    /// each session's tokens held, for status
    live_lens: Mutex<Vec<usize>>,
    started: Instant,
    /// the requests running (their JSON rows, updated as they go) and the last `keep` that ended
    running: Mutex<std::collections::BTreeMap<u64, Value>>,
    done: Mutex<VecDeque<Value>>,
    next_id: AtomicU64,
    /// how many ended requests `done` keeps
    keep: usize,
    /// the GPUs' power and temperature
    tele: Arc<Telemetry>,
    /// origins beyond the loopback ones a browser may call from
    cors: Vec<String>,
}

/// A chat request, from either API.
struct Ask {
    messages: Vec<(String, String, Option<String>)>,
    effort: Effort,
    max: Option<usize>,
    temp: f32,
    top_p: f32,
    stream: bool,
    /// logprobs asked for: how many alternatives with each (top_logprobs, at most 20)
    logprobs: Option<usize>,
    /// LogProbChain: the logprobs also chained through the attention to this turn's own earlier tokens
    chain: bool,
}

/// LogProbChain on one generated token's entry: for its token and each alternative, the logprob plus, over this
/// turn's earlier tokens equal to it, the attention the token's position gave each times that token's own logprob -
/// p' = p * prod_j p_j^a_j: an echo of what the turn already wrote counts only as much as that earlier token was
/// sure. `att`: per position (sums to 1); `turn`: the turn's tokens so far with their own logprobs, from `base`.
fn chain_entry(e: &mut Value, att: &[f32], turn: &[(u32, f64)], base: usize, tok: &Tokenizer) {
    let echo = |id: u32| -> (f64, Vec<(usize, f64, f64)>) {
        let mut sum = 0.0;
        let mut parts = Vec::new();
        for (k, (t, lp)) in turn.iter().enumerate() {
            if *t == id {
                let a = att.get(base + k).copied().unwrap_or(0.0) as f64;
                if a > 0.0 {
                    sum += a * lp;
                    parts.push((k, a, *lp));
                }
            }
        }
        (sum, parts)
    };
    let id = e["id"].as_u64().unwrap_or(0) as u32;
    let (sum, mut parts) = echo(id);
    let lp = e["logprob"].as_f64().unwrap_or(0.0);
    parts.sort_by(|a, b| (b.1 * -b.2).total_cmp(&(a.1 * -a.2)).then(b.1.total_cmp(&a.1)));
    let turn_att: f64 = att.iter().skip(base).map(|x| *x as f64).sum();
    e["chain"] = json!({
        "logprob": lp + sum,
        "attention_to_turn": turn_att,
        "echo": parts.iter().take(3).map(|(k, a, l)| json!({"turn_position": k, "token": String::from_utf8_lossy(&tok.decode_bytes(&[turn[*k].0])),
                                                          "attention": a, "logprob": l})).collect::<Vec<_>>(),
    });
    if let Some(top) = e["top_logprobs"].as_array_mut() {
        for t in top {
            let tid = t["id"].as_u64().unwrap_or(0) as u32;
            let (s2, _) = echo(tid);
            t["chained_logprob"] = json!(t["logprob"].as_f64().unwrap_or(0.0) + s2);
        }
    }
}

/// One generated token's logprob entry (OpenAI's shape): its log-probability under the model (the raw logits'
/// softmax, before temperature and top-p), its text and bytes, and the `k` likeliest tokens at that position
fn logprob_entry(tok: &Tokenizer, l: &[f32], y: u32, k: usize) -> Value {
    let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
    let one = |id: u32| {
        let b = tok.decode_bytes(&[id]);
        json!({"token": String::from_utf8_lossy(&b), "logprob": l[id as usize] as f64 - lse, "bytes": b, "id": id})
    };
    let mut e = one(y);
    let mut top: Vec<u32> = Vec::new();
    if k > 0 {
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, |a, b| l[*b as usize].total_cmp(&l[*a as usize]));
        idx.truncate(k);
        idx.sort_by(|a, b| l[*b as usize].total_cmp(&l[*a as usize]));
        top = idx;
    }
    e["top_logprobs"] = json!(top.into_iter().map(one).collect::<Vec<_>>());
    e
}

/// How a chat's answer is written: OpenAI's chunks (server-sent events), or /api/chat's JSON lines.
#[derive(Clone, Copy, PartialEq)]
enum Api {
    OpenAi,
    Lines,
}

/// The CORS header lines for a request from `origin`, when that origin may call
fn cors_headers(origin: Option<&str>, allowed: &[String]) -> String {
    let Some(o) = origin else { return String::new() };
    let loopback = ["http://localhost", "http://127.0.0.1", "http://[::1]"].iter()
        .any(|h| o == *h || o.strip_prefix(h).is_some_and(|rest| rest.starts_with(':')));
    if loopback || allowed.iter().any(|a| a == "*" || a.trim_end_matches('/') == o) {
        format!("Access-Control-Allow-Origin: {o}\r\nVary: Origin\r\n")
    } else {
        String::new()
    }
}

/// A request for the engine: its prompt's tokens, how to sample, where its tokens go
struct Job {
    #[allow(dead_code)]
    id: u64,
    ids: Vec<u32>,
    max: usize,
    temp: f32,
    top_p: f32,
    logprobs: Option<usize>,
    /// LogProbChain: decoded one token a pass (no draft block), the attention captured
    chain: bool,
    tx: mpsc::Sender<Ev>,
    /// the client went away: end it
    cancel: Arc<AtomicBool>,
}

/// What the engine tells a request's thread
enum Ev {
    /// its prompt is read: tokens reused, from where, seconds reading, checkpoints saved
    Read { from: usize, source: &'static str, seconds: f64, saved: usize },
    /// a token, and its logprob entry when asked for
    Tok(u32, Option<Value>),
    /// the end: why, and the drafts accepted / proposed
    End { finish: &'static str, accepted: u64, drafted: u64 },
    Fail(String),
}

/// A session of the engine's, and the tokens it holds
struct Slot {
    sess: Session,
    live: Vec<u32>,
}

/// A request decoding: its session, the state between steps, what it has made
struct Active {
    job: Job,
    slot: usize,
    /// decoding alone (with the draft block), or between batch steps: the logits to sample from, or the token
    /// committed and not fed yet
    dec: Option<Decoder>,
    logits: Option<Vec<f32>>,
    pending: Option<u32>,
    n: usize,
    committed: Vec<u32>,
    accepted: u64,
    drafted: u64,
    finish: Option<&'static str>,
    /// LogProbChain: this turn's tokens so far and their own logprobs
    turn: Vec<(u32, f64)>,
    /// its prompt still being read (a group of chunks a round while others decode)
    reading: Option<Reading>,
}

/// A prompt being read into its session: up to `at`, from `from` (`source`), the stops still ahead
struct Reading {
    at: usize,
    from: usize,
    source: &'static str,
    stops: Vec<usize>,
    t0: Instant,
    saved: usize,
}

/// Prompt tokens read a round while other requests decode: three chunks (18K; the GPUs' pipeline keeps ~75% of
/// its speed over three; a 256K prompt read at once held every other request for 4.5 minutes). Read alone, a prompt goes
/// to its end in one pipelined read, stopped at a chunk's end when a request arrives (groups of 8 alone had cost
/// 8%: the pipeline drained at each)
fn read_group(chunk: usize) -> usize {
    3 * chunk
}

/// The server's sampler for `Engine::step`: the request's temperature and top-p (speculative sampling of the drafts), each
/// committed token's logprob entry noted in order when asked for
struct ServeSampler<'a> {
    temp: f32,
    top_p: f32,
    rng: &'a mut Rng,
    tok: &'a Tokenizer,
    k: Option<usize>,
    lps: &'a mut VecDeque<Value>,
}

impl ns_runtime::Sampler for ServeSampler<'_> {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let y = sample(logits, self.temp, self.top_p, self.rng);
        self.record(logits, y);
        y
    }
    fn dist(&mut self, logits: &[f32]) -> Option<Vec<(u32, f32)>> {
        crate::dist(logits, self.temp, self.top_p)
    }
    fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }
    fn record(&mut self, logits: &[f32], token: u32) {
        if let Some(k) = self.k {
            self.lps.push_back(logprob_entry(self.tok, logits, token, k));
        }
    }
}

/// Prefixes shorter than this are read again rather than cached
const MIN_CHECKPOINT: usize = 64;

fn respond(s: &mut Conn, code: u16, body: &Value) {
    http::respond(s, code, body)
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A float32 setting as the client wrote it (0.95, not 0.949999988)
fn round4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

/// The start of a text for a request's record (at most PREVIEW characters)
fn preview(s: &str) -> String {
    const PREVIEW: usize = 300;
    match s.char_indices().nth(PREVIEW) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s.to_string(),
    }
}

/// A message's text: a string, or the text parts of a list.
fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str()).or_else(|| p.as_str())).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

impl Server {
    #[allow(clippy::too_many_arguments)]
    pub fn new(engine: Box<dyn Engine>, tok: Tokenizer, name: String, slot_ctx: Vec<usize>, default_effort: Effort, cache: PromptCache,
               cors: Vec<String>, keep: usize, default_max: Option<usize>) -> Result<Server, String> {
        let tele = Telemetry::start(&engine.gpu_info().iter().map(|g| g.pci.clone()).collect::<Vec<_>>());
        let parallel = slot_ctx.len().max(1);
        let max_ctx = slot_ctx.iter().copied().max().unwrap_or(8192);
        Ok(Server { engine, tok, name, max_ctx, slot_ctx, default_effort, queue: Mutex::new(VecDeque::new()), wake: std::sync::Condvar::new(),
                    inflight: AtomicU64::new(0), parallel, default_max, cache: Mutex::new(cache),
                    live_lens: Mutex::new(vec![0; parallel]), started: Instant::now(), running: Mutex::new(Default::default()),
                    done: Mutex::new(VecDeque::new()), next_id: AtomicU64::new(1), keep, tele, cors })
    }

    /// The checkpoint positions of a prompt: the end of its first turn and the start of its last user turn (each
    /// where a `<|user|>` token begins), and its end.
    fn stops(&self, ids: &[u32]) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(u) = self.tok.id("<|user|>") {
            let at: Vec<usize> = ids.iter().enumerate().filter(|(_, t)| **t == u).map(|(i, _)| i).collect();
            v.extend(at.first().copied());
            v.extend(at.last().copied());
        }
        v.push(ids.len());
        v.retain(|p| *p >= MIN_CHECKPOINT);
        v.sort();
        v.dedup();
        v
    }

    /// Serves until SIGTERM / SIGINT, then ends once no request runs: a GPU process stopped inside a kernel can
    /// leave the driver stuck, so a stop never cuts a generation off.
    pub fn run(self: Arc<Self>, addr: &str, socket: Option<PathBuf>) -> Result<(), String> {
        extern "C" fn on_signal(_: i32) {
            STOP.store(true, Ordering::SeqCst);
        }
        extern "C" {
            fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
        }
        // SAFETY: installing a handler that only stores to an atomic.
        unsafe {
            signal(15, on_signal);
            signal(2, on_signal);
        }
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        l.set_nonblocking(true).map_err(|e| e.to_string())?;
        let u = match &socket {
            Some(p) => {
                let _ = std::fs::remove_file(p); // one from a server before
                let u = UnixListener::bind(p).map_err(|e| format!("cannot listen on {}: {e}", p.display()))?;
                u.set_nonblocking(true).map_err(|e| e.to_string())?;
                Some(u)
            }
            None => None,
        };
        let me = self.clone();
        std::thread::spawn(move || me.engine());
        eprintln!("[serving {} on http://{addr}{}, {} request(s) at once]", self.name, socket.as_ref().map_or(String::new(), |p| format!(" and {}", p.display())),
                  self.parallel);
        loop {
            if STOP.load(Ordering::SeqCst) && self.inflight.load(Ordering::SeqCst) == 0 {
                eprintln!("[stopping: no request running]");
                // the experts decode asked for, for the next load's VRAM fill (NS_EXPERT_PROFILE)
                if let Ok(p) = std::env::var("NS_EXPERT_PROFILE") {
                    match self.engine.save_profile(std::path::Path::new(&p)) {
                        Ok(n) if n > 0 => eprintln!("[expert profile: {n} experts written to {p}]"),
                        Ok(_) => {}
                        Err(e) => eprintln!("[expert profile: writing {p} failed: {e}]"),
                    }
                }
                // the checkpoints in memory onto the disk tier, for the next server
                if let Ok(mut c) = self.cache.lock() {
                    let n = c.persist_all();
                    if n > 0 {
                        eprintln!("[prompt cache: {n} checkpoint(s) written to disk]");
                    }
                }
                if let Some(p) = &socket {
                    let _ = std::fs::remove_file(p);
                }
                return Ok(());
            }
            let mut idle = true;
            match l.accept() {
                Ok((c, _)) => {
                    let _ = c.set_nonblocking(false);
                    let me = self.clone();
                    std::thread::spawn(move || me.handle(Conn::Tcp(c)));
                    idle = false;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => {}
            }
            if let Some(u) = &u {
                if let Ok((c, _)) = u.accept() {
                    let _ = c.set_nonblocking(false);
                    let me = self.clone();
                    std::thread::spawn(move || me.handle(Conn::Unix(c)));
                    idle = false;
                }
            }
            if idle {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }

    /// The model, its GPUs, the requests running, the cache (the cache's figures only when the engine is not reading
    /// a prompt with it: a status call never waits).
    fn status_json(&self) -> Value {
        let rd = self.tele.readings();
        let gpus: Vec<Value> = self.engine.gpu_info().iter().enumerate().map(|(i, g)| {
            let r = rd.get(i).copied().unwrap_or_default();
            json!({"index": g.index, "name": g.name, "pci": g.pci, "total": g.total, "free": g.free, "layers": [g.layers.0, g.layers.1],
                   "expert_slots": g.expert_slots, "host_slots": g.host_slots, "watts": r.watts, "temp_c": r.temp_c, "vram_temp_c": r.vram_c})
        }).collect();
        let lives = self.live_lens.lock().unwrap().clone();
        let cache = self.cache.try_lock().ok().map(|c| json!({"entries": c.len(), "bytes": c.bytes(), "budget": c.budget(),
                                                             "evictions": c.evictions, "live_tokens": lives.iter().sum::<usize>(), "sessions": lives,
                                                             "disk": c.disk().map(|(n, b, g)| json!({"entries": n, "bytes": b, "budget": g}))}));
        let running: Vec<Value> = self.running.lock().unwrap().values().cloned().collect();
        json!({"model": self.name, "version": crate::VERSION, "uptime_seconds": self.started.elapsed().as_secs_f64(),
               "context": self.max_ctx, "contexts": self.slot_ctx, "mtp": self.engine.has_draft(), "parallel": self.parallel, "gpus": gpus,
               "busy": !running.is_empty(), "running": running.first().cloned(), "active": running,
               "waiting": self.queue.lock().unwrap().len(), "served": self.next_id.load(Ordering::Relaxed) - 1,
               "prompt_cache": cache, "stopping": STOP.load(Ordering::SeqCst)})
    }

    fn handle(&self, mut s: Conn) {
        let Ok(req) = http::read_request(match s.try_clone() {
            Ok(c) => c,
            Err(_) => return,
        }) else { return };
        let (method, path) = (req.method.as_str(), req.path.as_str());
        // the control routes answer on the socket only: the API port may be open to the network
        if path.starts_with("/server/") && s.kind() != "socket" {
            return respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path} (the control routes are on the server's socket)")}}));
        }
        let cors = cors_headers(req.origin.as_deref(), &self.cors);
        match (method, path) {
            ("OPTIONS", _) => {
                let _ = write!(s, "HTTP/1.1 204 No Content\r\n{cors}Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                                   Access-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 600\r\n\
                                   Content-Length: 0\r\nConnection: close\r\n\r\n");
            }
            ("GET", "/health") => respond(&mut s, 200, &json!({"status": "ok"})),
            ("GET", "/status") => {
                let st = self.status_json();
                respond(&mut s, 200, &json!({"busy": st["busy"], "prompt_cache": st["prompt_cache"]}))
            }
            ("GET", "/server/status") => respond(&mut s, 200, &self.status_json()),
            ("GET", "/server/requests") => {
                let done: Vec<Value> = self.done.lock().unwrap().iter().rev().cloned().collect();
                let running: Vec<Value> = self.running.lock().unwrap().values().cloned().collect();
                respond(&mut s, 200, &json!({"running": running.first().cloned(), "active": running, "done": done, "keep": self.keep}))
            }
            ("GET", p) if p.starts_with("/server/requests/") => {
                let id: Option<u64> = p["/server/requests/".len()..].trim_start_matches('#').parse().ok();
                let running = id.and_then(|i| self.running.lock().unwrap().get(&i).cloned());
                let found = running.or_else(|| self.done.lock().unwrap().iter().find(|r| r["id"].as_u64() == id).cloned());
                match found {
                    Some(r) => respond(&mut s, 200, &r),
                    None => respond(&mut s, 404, &json!({"error": {"message": format!("no request {} (the server keeps the last {})",
                                                                                    &p["/server/requests/".len()..], self.keep)}})),
                }
            }
            ("GET", "/server/cache") => match self.cache.try_lock() {
                Ok(c) => {
                    let list: Vec<Value> = c.list().iter().map(|(t, b, _, disk)| json!({"tokens": t, "bytes": b, "disk": disk})).collect();
                    respond(&mut s, 200, &json!({"entries": list, "bytes": c.bytes(), "budget": c.budget(), "evictions": c.evictions,
                                                 "disk": c.disk().map(|(n, b, g)| json!({"entries": n, "bytes": b, "budget": g}))}))
                }
                Err(_) => respond(&mut s, 409, &json!({"error": {"message": "a prompt is being read; ask again when it is done"}})),
            },
            ("POST", "/server/cache/clear") => match self.cache.try_lock() {
                Ok(mut c) => {
                    let n = c.clear();
                    respond(&mut s, 200, &json!({"dropped": n}))
                }
                Err(_) => respond(&mut s, 409, &json!({"error": {"message": "a prompt is being read; ask again when it is done"}})),
            },
            ("POST", "/server/shutdown") => {
                STOP.store(true, Ordering::SeqCst);
                respond(&mut s, 200, &json!({"stopping": true, "running": self.inflight.load(Ordering::SeqCst) > 0}))
            }
            ("GET", "/v1/models") | ("GET", "/models") => http::respond_with(&mut s, 200, &json!({"object": "list", "data": [
                {"id": self.name, "object": "model", "owned_by": "nextsycl", "created": now(), "status": {"value": "loaded"},
                 "context": self.max_ctx}]}), &cors),
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") | ("POST", "/api/chat") => {
                let api = if path == "/api/chat" { Api::Lines } else { Api::OpenAi };
                let ask = match serde_json::from_slice::<Value>(&req.body).map_err(|e| format!("the body is not JSON: {e}"))
                    .and_then(|b| self.ask(&b, api)) {
                    Ok(a) => a,
                    Err(e) => return http::respond_with(&mut s, 400, &json!({"error": {"message": e}}), &cors),
                };
                let via = if api == Api::Lines { "web" } else { s.kind() };
                let mut rid = 0;
                let r = self.chat(&mut s, &ask, api, &cors, via, &mut rid);
                // a request that ended on an error is still recorded
                if let Some(mut row) = self.running.lock().unwrap().remove(&rid) {
                    row["state"] = json!("failed");
                    row["error"] = json!(r.as_ref().err());
                    self.remember(row);
                }
                if let Err(e) = r {
                    http::respond_with(&mut s, 500, &json!({"error": {"message": e}}), &cors);
                }
            }
            _ => respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}})),
        }
    }

    fn remember(&self, row: Value) {
        let mut d = self.done.lock().unwrap();
        d.push_back(row);
        while d.len() > self.keep {
            d.pop_front();
        }
    }

    /// Updates a running request's row.
    fn live(&self, rid: u64, f: impl FnOnce(&mut Value)) {
        if let Some(row) = self.running.lock().unwrap().get_mut(&rid) {
            f(row);
        }
    }

    /// A request's body as an `Ask` (OpenAI's fields, or /api/chat's).
    fn ask(&self, req: &Value, api: Api) -> Result<Ask, String> {
        let thinking_key = if api == Api::Lines { "thinking" } else { "reasoning_content" };
        let messages = req["messages"].as_array().ok_or("messages are required")?.iter().map(|m| {
            (m["role"].as_str().unwrap_or("user").to_string(), text_of(&m["content"]), m[thinking_key].as_str().map(str::to_string))
        }).collect();
        let effort = match api {
            Api::Lines => req["effort"].as_str(),
            Api::OpenAi => req["reasoning_effort"].as_str().or_else(|| req["chat_template_kwargs"]["reasoning_effort"].as_str()),
        };
        let effort = match effort {
            Some(e) => Effort::parse(e).ok_or(format!("effort {e:?}: low, high or max"))?,
            None => self.default_effort,
        };
        let chain = req["logprob_chain"].as_bool().or_else(|| req["LogProbChain"].as_bool()).unwrap_or(false);
        Ok(Ask { messages, effort,
                 max: req["max_tokens"].as_u64().or_else(|| req["max_completion_tokens"].as_u64()).map(|n| n as usize),
                 temp: req["temperature"].as_f64().unwrap_or(1.0) as f32,
                 top_p: req["top_p"].as_f64().unwrap_or(0.95) as f32,
                 stream: req["stream"].as_bool().unwrap_or(api == Api::Lines),
                 logprobs: match &req["logprobs"] {
                     Value::Bool(true) => Some(req["top_logprobs"].as_u64().unwrap_or(0).min(20) as usize),
                     Value::Number(n) => Some(n.as_u64().unwrap_or(0).min(20) as usize), // the completions API's form
                     _ if chain => Some(req["top_logprobs"].as_u64().unwrap_or(5).min(20) as usize),
                     _ => None,
                 },
                 chain })
    }

    // ------------------------------------------------------------------ the engine thread

    /// Owns the sessions and the prompt cache: takes waiting requests into free sessions (reading their prompts),
    /// and steps the active ones - alone with the draft block, or together in one batch pass.
    fn engine(&self) {
        let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
        let mut slots: Vec<Slot> = match self.slot_ctx.iter().map(|&c| self.engine.session(c).map(|sess| Slot { sess, live: Vec::new() }))
            .collect::<ns_core::Result<Vec<_>>>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[the engine cannot make its sessions: {}]", e.0);
                return;
            }
        };
        let mut rng = Rng(0x5DEECE66D);
        let mut active: Vec<Active> = Vec::new();
        // the next prompt group waits until then while others decode; the last group's time (a request that starts
        // decoding gets that long before the next group)
        let mut read_after = Instant::now();
        let mut last_group = std::time::Duration::from_secs(15); // ~a group on a B65 + B70 until one is timed
        loop {
            // take waiting requests while sessions are free
            while active.len() < self.parallel {
                let Some(mut job) = self.queue.lock().unwrap().pop_front() else { break };
                let used: Vec<usize> = active.iter().map(|a| a.slot).collect();
                let fits = |i: usize| slots[i].sess.max_ctx() >= job.ids.len() + 16;
                // among the free sessions it fits: the one holding the longest prefix of this prompt, then the
                // smallest (a long session kept for the long prompts)
                let slot = (0..slots.len()).filter(|i| !used.contains(i) && fits(*i))
                    .max_by_key(|&i| {
                        let l = &slots[i].live;
                        (if l.len() < job.ids.len() && job.ids.starts_with(l) { l.len() + 1 } else { 0 }, std::cmp::Reverse(slots[i].sess.max_ctx()))
                    });
                let Some(slot) = slot else {
                    // only busy sessions fit it: it waits at the front
                    self.queue.lock().unwrap().push_front(job);
                    break;
                };
                // its tokens end at its session's context
                job.max = job.max.min(slots[slot].sess.max_ctx() - job.ids.len() - 1);
                match self.begin_read(&mut slots[slot], &job.ids) {
                    Ok(rd) => {
                        active.push(Active { job, slot, dec: None, logits: None, pending: None, n: 0, committed: Vec::new(), accepted: 0,
                                             drafted: 0, finish: None, turn: Vec::new(), reading: Some(rd) });
                    }
                    Err(e) => {
                        let _ = job.tx.send(Ev::Fail(e));
                        self.inflight.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                self.lives(&slots);
            }
            // the prompts being read: a group each round (new requests are taken between groups). While others decode
            // the GPUs are shared by time: after a group of t seconds they decode for t seconds (one decode step a
            // group gave a chat 0.23 tok/s beside a 256K prompt)
            let others = active.len() > 1 || !self.queue.lock().unwrap().is_empty();
            let mut decoding = active.iter().any(|a| a.reading.is_none());
            // the shortest left to read first (a chat behind a 256K prompt then starts at once)
            active.sort_by_key(|a| a.reading.as_ref().map_or(0, |r| a.job.ids.len() - r.at));
            let mut i = 0;
            while i < active.len() {
                // the decode share holds back long reads only: a prompt of a chunk or less reads at once (a second
                // short request waited out the first one's window - 11 s to its first token)
                let left = active[i].reading.as_ref().map_or(0, |r| active[i].job.ids.len() - r.at);
                if active[i].reading.is_none() || (decoding && left > self.engine.prefill_chunk() && Instant::now() < read_after) {
                    i += 1;
                    continue;
                }
                let g0 = Instant::now();
                let a = &mut active[i];
                let limit = if others { read_group(self.engine.prefill_chunk()) } else { usize::MAX };
                let r = if a.job.cancel.load(Ordering::Relaxed) {
                    Err("client gone".to_string())
                } else {
                    self.read_some(&mut slots[a.slot], &a.job.ids, a.reading.as_mut().unwrap(), limit, &mut none)
                };
                // the round's reading in all (a short prompt read after a long group must not shorten the share)
                read_after = read_after.max(Instant::now()) + g0.elapsed();
                if others && limit == read_group(self.engine.prefill_chunk()) && g0.elapsed() > last_group / 2 {
                    last_group = g0.elapsed();
                }
                match r {
                    Ok(Some(logits)) => {
                        // it decodes from now: a window before the next group
                        decoding = true;
                        read_after = read_after.max(Instant::now() + last_group);
                        let rd = a.reading.take().unwrap();
                        slots[a.slot].live = a.job.ids.clone();
                        let _ = a.job.tx.send(Ev::Read { from: rd.from, source: rd.source, seconds: rd.t0.elapsed().as_secs_f64(), saved: rd.saved });
                        a.logits = Some(logits);
                        i += 1;
                    }
                    Ok(None) => i += 1,
                    Err(e) => {
                        let a = active.remove(i);
                        let _ = a.job.tx.send(Ev::Fail(e));
                        self.inflight.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                self.lives(&slots);
            }
            if active.is_empty() {
                let q = self.queue.lock().unwrap();
                if q.is_empty() {
                    let _ = self.wake.wait_timeout(q, std::time::Duration::from_millis(200));
                }
                continue;
            }
            // the big arena stays with the prompts while one is being read
            self.engine.hold_arena(active.iter().any(|a| a.reading.is_some()));
            // the ones still reading wait out this round's steps
            let mut readers: Vec<Active> = Vec::new();
            let mut i = 0;
            while i < active.len() {
                if active[i].reading.is_some() {
                    readers.push(active.remove(i));
                } else {
                    i += 1;
                }
            }
            if active.is_empty() {
                active = readers;
                continue;
            }
            active.sort_by_key(|a| a.slot);
            for a in active.iter_mut() {
                if a.job.cancel.load(Ordering::Relaxed) {
                    a.finish = Some("client gone");
                }
            }
            // LogProbChain requests step one at a time, their attention captured; the others alone or together
            let mut r = Ok(());
            for a in active.iter_mut() {
                if a.job.chain && a.finish.is_none() && r.is_ok() {
                    r = self.step_chain(&mut slots, a, &mut rng, &mut none);
                }
            }
            let mut rest: Vec<Active> = Vec::new();
            let mut i = 0;
            while i < active.len() {
                if active[i].job.chain {
                    i += 1;
                } else {
                    rest.push(active.remove(i));
                }
            }
            if r.is_ok() && rest.iter().any(|a| a.finish.is_none()) {
                r = if rest.iter().filter(|a| a.finish.is_none()).count() == 1 {
                    self.step_alone(&mut slots, &mut rest, &mut rng, &mut none)
                } else {
                    self.step_together(&mut slots, &mut rest, &mut rng, &mut none)
                };
            }
            active.extend(rest);
            active.sort_by_key(|a| a.slot);
            if let Err(e) = r {
                for a in active.drain(..) {
                    let _ = a.job.tx.send(Ev::Fail(e.clone()));
                    self.inflight.fetch_sub(1, Ordering::SeqCst);
                }
                active.append(&mut readers);
                self.lives(&slots);
                continue;
            }
            // the ended ones: what their sessions hold now, their end
            let mut i = 0;
            while i < active.len() {
                if let Some(finish) = active[i].finish {
                    let mut a = active.remove(i);
                    if let Some(d) = a.dec.take() {
                        let (dr, ac) = d.drafts();
                        a.accepted += ac;
                        a.drafted += dr;
                    }
                    let sl = &mut slots[a.slot];
                    let fed = sl.sess.pos().saturating_sub(a.job.ids.len()).min(a.committed.len());
                    sl.live = a.job.ids.clone();
                    sl.live.extend_from_slice(&a.committed[..fed]);
                    let _ = a.job.tx.send(Ev::End { finish, accepted: a.accepted, drafted: a.drafted });
                    self.inflight.fetch_sub(1, Ordering::SeqCst);
                } else {
                    i += 1;
                }
            }
            active.append(&mut readers);
            self.lives(&slots);
        }
    }

    fn lives(&self, slots: &[Slot]) {
        *self.live_lens.lock().unwrap() = slots.iter().map(|s| s.live.len()).collect();
    }

    /// A prompt's read begun in a session: from the live state when it holds a prefix, a cached checkpoint when that
    /// is longer, else from the start (the cache only gives a prefix shorter than the prompt: a token is always fed)
    fn begin_read(&self, sl: &mut Slot, ids: &[u32]) -> Result<Reading, String> {
        let t0 = Instant::now();
        let mut cache = self.cache.lock().unwrap();
        let live_len = if sl.live.len() < ids.len() && ids.starts_with(&sl.live) { sl.live.len() } else { 0 };
        let cached = if cache.enabled() { cache.best(ids) } else { None };
        let (from, source) = match cached {
            Some((hit, len)) if len > live_len => {
                cache.with(hit, &|r| self.engine.read_checkpoint(r), |ck| self.engine.restore(&mut sl.sess, ck))?.map_err(|e| e.0)?;
                (len, if matches!(hit, crate::cache::Hit::Disk(_)) { "disk" } else { "cache" })
            }
            _ if live_len > 0 => (live_len, "live"),
            _ => {
                self.engine.reset_session(&mut sl.sess).map_err(|e| e.0)?;
                (0, "none")
            }
        };
        sl.live.clear(); // until this prompt is read, the session is in between
        let stops = self.stops(ids).into_iter().filter(|p| *p > from).collect();
        Ok(Reading { at: from, from, source, stops, t0, saved: 0 })
    }

    /// Up to `limit` more of a prompt's tokens (a checkpoint saved at each stop passed): the last token's logits once
    /// it is read to its end
    fn read_some(&self, sl: &mut Slot, ids: &[u32], rd: &mut Reading, limit: usize, none: ns_runtime::Tap) -> Result<Option<Vec<f32>>, String> {
        let mut budget = limit;
        while rd.at < ids.len() && budget > 0 {
            let stop = rd.stops.first().copied().unwrap_or(ids.len());
            let end = stop.min(rd.at.saturating_add(budget)).min(ids.len());
            // read alone (no limit), it stops at a chunk's end once another request waits
            let arrived = || limit == usize::MAX && !self.queue.lock().unwrap().is_empty();
            let (fed, logits) = self.engine.feed_until(&mut sl.sess, &ids[rd.at..end], &arrived, &mut *none).map_err(|e| e.0)?;
            budget = budget.saturating_sub(fed);
            rd.at += fed;
            if rd.at < end {
                return Ok(None); // stopped for a newcomer
            }
            if !rd.stops.is_empty() && end == stop {
                rd.stops.remove(0);
                let mut cache = self.cache.lock().unwrap();
                if cache.enabled() && !cache.touch(&ids[..stop]) {
                    let ck = self.engine.save(&sl.sess).map_err(|e| e.0)?;
                    if cache.put(ids[..stop].to_vec(), ck) {
                        rd.saved += 1;
                    }
                }
            }
            if rd.at == ids.len() {
                return Ok(Some(logits));
            }
        }
        Ok(None)
    }

    /// A committed token of `a`'s: to its request (or the end, at a stop token or its limit)
    fn commit(&self, a: &mut Active, y: u32, lp: Option<Value>) {
        a.committed.push(y);
        if a.finish.is_some() {
            return;
        }
        if self.tok.stop.contains(&y) {
            a.finish = Some("stop");
            return;
        }
        a.n += 1;
        if a.job.tx.send(Ev::Tok(y, lp)).is_err() {
            a.finish = Some("client gone");
        } else if a.n >= a.job.max {
            a.finish = Some("length");
        }
    }

    /// One request active: a step of its own, with the draft block (one token, or two with a draft accepted)
    fn step_alone(&self, slots: &mut [Slot], active: &mut [Active], rng: &mut Rng, none: ns_runtime::Tap) -> Result<(), String> {
        let Some(a) = active.iter_mut().find(|a| a.finish.is_none()) else { return Ok(()) };
        if a.dec.is_none() {
            let mut dec = match (a.logits.take(), a.pending.take()) {
                (Some(l), _) => self.engine.decoder(l, true),
                (None, Some(y)) => self.engine.decoder_after(y, true),
                (None, None) => return Err("a request without logits or a token to feed".into()),
            };
            // the conversation so far, for prompt-lookup drafts (NS_NGRAM)
            let mut ctx = a.job.ids.clone();
            ctx.extend_from_slice(&a.committed);
            dec.set_context(&ctx);
            a.dec = Some(dec);
        }
        let (temp, top_p, k) = (a.job.temp, a.job.top_p, a.job.logprobs);
        let mut lps: VecDeque<Value> = VecDeque::new();
        let mut draw = ServeSampler { temp, top_p, rng, tok: &self.tok, k, lps: &mut lps };
        let toks = self.engine.step(&mut slots[a.slot].sess, a.dec.as_mut().unwrap(), &mut draw, &mut *none).map_err(|e| e.0)?;
        for y in toks {
            let lp = lps.pop_front();
            self.commit(a, y, lp);
        }
        Ok(())
    }

    /// A LogProbChain request's step: one token (no draft block), the attention of the pass that made its logits
    /// captured, the token's entry chained through it
    fn step_chain(&self, slots: &mut [Slot], a: &mut Active, rng: &mut Rng, none: ns_runtime::Tap) -> Result<(), String> {
        if a.dec.is_none() {
            a.dec = Some(match (a.logits.take(), a.pending.take()) {
                (Some(l), _) => self.engine.decoder(l, false),
                (None, Some(y)) => self.engine.decoder_after(y, false),
                (None, None) => return Err("a request without logits or a token to feed".into()),
            });
        }
        let (temp, top_p, k) = (a.job.temp, a.job.top_p, a.job.logprobs.unwrap_or(5));
        let mut entry: Option<Value> = None;
        let tok = &self.tok;
        let mut draw = |l: &[f32]| {
            let y = sample(l, temp, top_p, rng);
            entry = Some(logprob_entry(tok, l, y, k));
            y
        };
        self.engine.capture_attention(true);
        let toks = self.engine.step(&mut slots[a.slot].sess, a.dec.as_mut().unwrap(), &mut draw, &mut *none);
        let att = self.engine.take_attention();
        self.engine.capture_attention(false);
        let toks = toks.map_err(|e| e.0)?;
        let mut e = entry.unwrap_or(Value::Null);
        if let Some(att) = &att {
            chain_entry(&mut e, att, &a.turn, a.job.ids.len(), &self.tok);
        }
        for y in toks {
            a.turn.push((y, e["logprob"].as_f64().unwrap_or(0.0)));
            self.commit(a, y, Some(e.clone()));
        }
        Ok(())
    }

    /// Several active: a token each, one batch pass for all of them
    fn step_together(&self, slots: &mut [Slot], active: &mut [Active], rng: &mut Rng, none: ns_runtime::Tap) -> Result<(), String> {
        // each one's next token: drawn from its logits, or the one its own steps committed and did not feed
        let mut feed: Vec<(usize, u32)> = Vec::new(); // (index in active, token)
        for (i, a) in active.iter_mut().enumerate() {
            if a.finish.is_some() {
                continue;
            }
            if let Some(mut d) = a.dec.take() {
                let (dr, ac) = d.drafts();
                a.accepted += ac;
                a.drafted += dr;
                a.pending = d.pending();
            }
            let y = match (a.logits.take(), a.pending.take()) {
                (Some(l), _) => {
                    let y = sample(&l, a.job.temp, a.job.top_p, rng);
                    let lp = a.job.logprobs.map(|k| logprob_entry(&self.tok, &l, y, k));
                    self.commit(a, y, lp);
                    y
                }
                (None, Some(y)) => y,
                (None, None) => return Err("a request without logits or a token to feed".into()),
            };
            if a.finish.is_none() {
                feed.push((i, y));
            }
        }
        if feed.is_empty() {
            return Ok(());
        }
        // the sessions of `feed`, in its order (active is sorted by slot, and slots differ)
        let want: Vec<usize> = feed.iter().map(|(i, _)| active[*i].slot).collect();
        let mut sess: Vec<&mut Session> = slots.iter_mut().enumerate().filter(|(si, _)| want.contains(si)).map(|(_, s)| &mut s.sess).collect();
        let toks: Vec<u32> = feed.iter().map(|(_, y)| *y).collect();
        let logits = self.engine.forward_batch(&mut sess, &toks, &mut *none).map_err(|e| e.0)?;
        for ((i, _), l) in feed.iter().zip(logits) {
            active[*i].logits = Some(l);
        }
        Ok(())
    }

    // ------------------------------------------------------------------ a request's thread

    fn chat(&self, s: &mut Conn, ask: &Ask, api: Api, cors: &str, via: &str, rid_out: &mut u64) -> Result<(), String> {
        let messages: Vec<Message> = ask.messages.iter().map(|(r, c, rc)| Message { role: r, content: c, reasoning: rc.as_deref() }).collect();
        let ids = self.tok.encode(&ns_tok::glm_chat(&messages, ask.effort));
        if ids.len() + 16 > self.max_ctx {
            return Err(format!("the prompt is {} tokens; the context is {}", ids.len(), self.max_ctx));
        }
        // a request without max_tokens: the server's cap (NS_MAX_TOKENS), else to the end of the context - a long
        // think is never cut off before its answer unless asked
        let max = ask.max.or(self.default_max).unwrap_or(usize::MAX).min(self.max_ctx - ids.len() - 1);
        let (temp, top_p, stream) = (ask.temp, ask.top_p, ask.stream);
        let id = format!("chatcmpl-{}", now());
        let rid = self.next_id.fetch_add(1, Ordering::Relaxed);
        *rid_out = rid;
        let effort = match ask.effort {
            Effort::Low => "low",
            Effort::High => "high",
            Effort::Max => "max",
        };
        let last_user = ask.messages.iter().rev().find(|m| m.0 == "user").map_or("", |m| m.1.as_str());
        self.running.lock().unwrap().insert(rid, json!({"id": rid, "via": via, "state": "waiting", "started": now(), "prompt_tokens": ids.len(),
                                                    "max_tokens": max, "generated": 0, "model": self.name,
                                                    "api": if api == Api::Lines { "/api/chat" } else { "/v1/chat/completions" },
                                                    "settings": {"effort": effort, "temperature": round4(temp), "top_p": round4(top_p), "stream": stream,
                                                                 "max_tokens": max},
                                                    "messages": ask.messages.len(),
                                                    "prompt_chars": ask.messages.iter().map(|m| m.1.len()).sum::<usize>(),
                                                    "last_user": preview(last_user)}));
        // the cards' energy counters at the start: the request's energy is their rise (other requests running at the
        // same time draw on the same counters)
        let j0 = self.tele.joules();
        let energy = || -> Option<f64> { Some(self.tele.joules()? - j0?) };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.inflight.fetch_add(1, Ordering::SeqCst);
        self.queue.lock().unwrap().push_back(Job { id: rid, ids: ids.clone(), max, temp, top_p, logprobs: ask.logprobs, chain: ask.chain, tx,
                                                   cancel: cancel.clone() });
        self.wake.notify_one();
        // its prompt read
        let (from, source, prefill, saved) = match rx.recv() {
            Ok(Ev::Read { from, source, seconds, saved }) => (from, source, seconds, saved),
            Ok(Ev::Fail(e)) => return Err(e),
            _ => return Err("the engine ended the request".into()),
        };
        self.live(rid, |r| {
            r["state"] = json!("generating");
            r["reused"] = json!(from);
            r["source"] = json!(source);
            r["read_seconds"] = json!(prefill);
        });

        if stream {
            let kind = if api == Api::Lines { "application/x-ndjson" } else { "text/event-stream" };
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nCache-Control: no-cache\r\n{cors}Connection: close\r\n\r\n");
        }
        let send = |s: &mut Conn, delta: Value, finish: Option<&str>| -> bool {
            let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": self.name,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            write!(s, "data: {chunk}\n\n").and_then(|_| s.flush()).is_ok()
        };
        // one piece of the answer as it comes, with the logprobs of the answer tokens it completes (when asked)
        let lp_on = ask.logprobs.is_some();
        let emit = |s: &mut Conn, thinking: bool, text: String, lps: Vec<Value>| -> bool {
            match api {
                Api::OpenAi => {
                    let delta = if thinking { json!({"reasoning_content": text}) } else { json!({"content": text}) };
                    let mut chunk = json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": self.name,
                                           "choices": [{"index": 0, "delta": delta, "finish_reason": null}]});
                    if lp_on {
                        chunk["choices"][0]["logprobs"] = if lps.is_empty() { Value::Null } else { json!({"content": lps}) };
                    }
                    write!(s, "data: {chunk}\n\n").and_then(|_| s.flush()).is_ok()
                }
                Api::Lines => {
                    let mut line = if thinking { json!({"thinking": text}) } else { json!({"content": text}) };
                    if !lps.is_empty() {
                        line["logprobs"] = json!(lps);
                    }
                    writeln!(s, "{line}").and_then(|_| s.flush()).is_ok()
                }
            }
        };
        if stream && api == Api::OpenAi {
            send(s, json!({"role": "assistant"}), None);
        }
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut thinking = true; // the prompt ends in <think>
        let mut pending: Vec<u8> = Vec::new();
        let mut n = 0;
        let t1 = Instant::now();
        let (mut lp_all, mut lp_pending): (Vec<Value>, Vec<Value>) = (Vec::new(), Vec::new());
        let mut gone = false;
        let (finish, accepted, drafted) = loop {
            let (next, lp) = match rx.recv() {
                Ok(Ev::Tok(y, lp)) => (y, lp),
                Ok(Ev::End { finish, accepted, drafted }) => break (finish, accepted, drafted),
                Ok(Ev::Fail(e)) => return Err(e),
                Ok(Ev::Read { .. }) => continue,
                Err(_) => return Err("the engine ended the request".into()),
            };
            if gone {
                continue; // drain until the engine ends it
            }
            // the thinking's tokens carry none (as OpenAI's reasoning models)
            if let Some(e) = lp.filter(|_| !thinking) {
                lp_all.push(e.clone());
                lp_pending.push(e);
            }
            n += 1;
            let el = t1.elapsed().as_secs_f64();
            self.live(rid, |r| {
                r["generated"] = json!(n);
                r["tok_s"] = json!(n as f64 / el.max(1e-9));
                r["energy_j"] = json!(energy());
            });
            pending.extend(self.tok.decode_bytes(&[next]));
            let valid = match std::str::from_utf8(&pending) {
                Ok(t) => t.len(),
                Err(e) => e.valid_up_to(),
            };
            let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            let mut parts: Vec<(bool, String)> = Vec::new();
            if thinking {
                // the thinking ends at </think> (one token or several pieces)
                let joined = format!("{reasoning}{piece}");
                if let Some(i) = joined.find("</think>") {
                    let before = joined[reasoning.len().min(i)..i].to_string();
                    let after = joined[i + "</think>".len()..].trim_start().to_string();
                    reasoning = joined[..i].to_string();
                    thinking = false;
                    parts.push((true, before));
                    content.push_str(&after);
                    parts.push((false, after));
                } else {
                    reasoning.push_str(&piece);
                    parts.push((true, piece));
                }
            } else {
                content.push_str(&piece);
                parts.push((false, piece));
            }
            if stream {
                for (is_r, p) in parts {
                    let l = if is_r { Vec::new() } else { std::mem::take(&mut lp_pending) };
                    if !p.is_empty() && !emit(s, is_r, p, l) {
                        gone = true;
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
            }
        };
        let dt = t1.elapsed().as_secs_f64();
        eprintln!("[request #{rid}: {} prompt tokens ({} reused from {source}, {} fed in {prefill:.1} s, {saved} checkpoint(s) saved), {n} generated in {dt:.1} s ({:.2} tok/s), drafts {}/{} accepted, {finish}]",
                  ids.len(), from, ids.len() - from, n as f64 / dt.max(1e-9), accepted, drafted);
        // the energy both cards drew while it ran (idle power included): joules, and watt-hours in the answer
        let e = energy();
        let wh = e.map(|j| (j / 3600.0 * 1e4).round() / 1e4);
        let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": n, "total_tokens": ids.len() + n, "energy_wh": wh});
        if let Some(mut row) = self.running.lock().unwrap().remove(&rid) {
            row["state"] = json!("done");
            row["finish"] = json!(finish);
            row["generated"] = json!(n);
            row["tok_s"] = json!(n as f64 / dt.max(1e-9));
            row["generate_seconds"] = json!(dt);
            row["drafts"] = json!([accepted, drafted]);
            row["checkpoints_saved"] = json!(saved);
            row["ended"] = json!(now());
            let r1 = |x: f64| (x * 10.0).round() / 10.0;
            row["energy_j"] = json!(e.map(r1));
            row["avg_watts"] = json!(e.map(|e| r1(e / (prefill + dt).max(1e-9))));
            row["answer"] = json!({"chars": content.trim().len(), "preview": preview(content.trim())});
            row["thinking"] = json!({"chars": reasoning.trim().len(), "preview": preview(reasoning.trim())});
            self.remember(row);
        }
        let finish = if finish == "client gone" { "stop" } else { finish };
        if api == Api::Lines {
            let mut end = json!({"done": true, "finish": finish, "model": self.name, "prompt_tokens": ids.len(), "reused": from, "generated": n,
                                 "read_seconds": prefill, "generate_seconds": dt, "tok_s": n as f64 / dt.max(1e-9), "energy_wh": wh});
            if stream && !lp_pending.is_empty() {
                end["logprobs"] = json!(lp_pending); // the answer's last tokens, when no text followed them
            }
            if stream {
                let _ = writeln!(s, "{end}");
                let _ = s.flush();
            } else {
                end["content"] = json!(content.trim());
                end["thinking"] = json!(reasoning.trim());
                if lp_on {
                    end["logprobs"] = json!(lp_all);
                }
                http::respond_with(s, 200, &end, cors);
            }
        } else if stream {
            if !lp_pending.is_empty() {
                emit(s, false, String::new(), std::mem::take(&mut lp_pending));
            }
            send(s, json!({}), Some(finish));
            let _ = write!(s, "data: {}\n\ndata: [DONE]\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": self.name,
                                                                      "choices": [], "usage": usage}));
        } else {
            http::respond_with(s, 200, &json!({"id": id, "object": "chat.completion", "created": now(), "model": self.name,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content.trim(),
                                                 "reasoning_content": reasoning.trim()}, "finish_reason": finish,
                                                 "logprobs": if lp_on { json!({"content": lp_all}) } else { Value::Null }}],
                                    "usage": usage,
                                    "timings": {"prompt_n": ids.len() - from, "prompt_ms": prefill * 1000.0, "predicted_n": n,
                                                "predicted_per_second": n as f64 / dt.max(1e-9), "energy_wh": wh}}), cors);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::cors_headers;

    #[test]
    fn cors_lets_loopback_and_named_origins_in() {
        let none: Vec<String> = Vec::new();
        assert!(cors_headers(Some("http://localhost:8095"), &none).contains("http://localhost:8095"));
        assert!(cors_headers(Some("http://127.0.0.1:5173"), &none).contains("Access-Control-Allow-Origin"));
        assert_eq!(cors_headers(Some("http://localhost.evil.com"), &none), "");
        assert_eq!(cors_headers(Some("https://example.com"), &none), "");
        assert_eq!(cors_headers(None, &none), "");
        let named = vec!["http://studio:8095/".to_string()];
        assert!(cors_headers(Some("http://studio:8095"), &named).contains("http://studio:8095"));
        assert!(cors_headers(Some("https://example.com"), &["*".to_string()]).contains("https://example.com"));
    }
}
