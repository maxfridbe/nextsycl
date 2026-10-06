//! `nextsycl serve`: an OpenAI-compatible server for one loaded model (what the box's model switch and Open WebUI
//! talk to).
//!
//!     GET  /health                 {"status": "ok"} once the model is loaded
//!     GET  /v1/models              the one model
//!     GET  /status                 {"busy": bool, "prompt_cache": ...}
//!     POST /v1/chat/completions    messages, max_tokens, temperature, top_p, stream, reasoning_effort (or
//!                                  chat_template_kwargs.reasoning_effort): low | high | max
//!
//! The same routes and the control ones answer on a Unix socket too (`--socket`; `nextsycl start` puts it in
//! $XDG_RUNTIME_DIR/nextsycl), which the host command line talks to (client.rs), as sycl-h3 talks to h3d:
//!
//!     GET  /server/status          the model, its GPUs (memory, layers, expert slots), the request running, the cache
//!     GET  /server/requests        the requests: the running one and the last 100
//!     GET  /server/cache           the prompt cache's checkpoints;  POST /server/cache/clear  drops them
//!     POST /server/shutdown        stop once no request runs (as SIGTERM)
//!
//! One request runs at a time (the others wait). The prompt cache (cache.rs, --prompt-cache-mib) keeps the
//! conversation state at three points of every prompt - the end of its first turn (a system prompt other
//! conversations share), the start of its last user turn (an edited or regenerated message), its end (the next turn
//! of the conversation) - and a request mounts the longest cached prefix of its tokens, or continues the live
//! session when that is longer, and reads only the rest. GLM's thinking comes back as `reasoning_content`, the
//! answer as `content`.

use std::collections::VecDeque;
use std::io::Write;
use std::net::TcpListener;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_engine::glm5next::{Glm, Session};

use crate::cache::PromptCache;
use crate::http::{self, Conn};
use ns_tok::{Effort, Message, Tokenizer};
use serde_json::{json, Value};

use crate::{sample, Rng};

static STOP: AtomicBool = AtomicBool::new(false);

pub struct Server {
    pub glm: Glm<'static>,
    pub tok: Tokenizer,
    pub name: String,
    pub max_ctx: usize,
    pub default_effort: Effort,
    state: Mutex<Conv>,
    busy: AtomicBool,
    started: Instant,
    /// the request running (its JSON row, updated as it goes) and the last 100 that ended
    current: Mutex<Option<Value>>,
    done: Mutex<VecDeque<Value>>,
    next_id: AtomicU64,
}

/// The working session with the tokens it has consumed, and the prompt cache.
struct Conv {
    work: Session,
    live: Vec<u32>,
    cache: PromptCache,
    rng: Rng,
}

/// Prefixes shorter than this are read again rather than cached
const MIN_CHECKPOINT: usize = 64;

fn respond(s: &mut Conn, code: u16, body: &Value) {
    http::respond(s, code, body)
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
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
    pub fn new(glm: Glm<'static>, tok: Tokenizer, name: String, max_ctx: usize, default_effort: Effort, cache_bytes: usize) -> Result<Server, String> {
        let work = glm.session(max_ctx).map_err(|e| e.0)?;
        Ok(Server { glm, tok, name, max_ctx, default_effort,
                    state: Mutex::new(Conv { work, live: Vec::new(), cache: PromptCache::new(cache_bytes), rng: Rng(0x5DEECE66D) }),
                    busy: AtomicBool::new(false), started: Instant::now(), current: Mutex::new(None), done: Mutex::new(VecDeque::new()),
                    next_id: AtomicU64::new(1) })
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
        eprintln!("[serving {} on http://{addr}{}]", self.name, socket.as_ref().map_or(String::new(), |p| format!(" and {}", p.display())));
        loop {
            if STOP.load(Ordering::SeqCst) {
                // finish what runs (the request's thread holds the conversation's lock)
                let _wait = self.state.lock().unwrap();
                eprintln!("[stopping: no request running]");
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

    /// The model, its GPUs, the request running, the cache (the cache's figures only when no request holds the
    /// conversation: a status call never waits).
    fn status_json(&self) -> Value {
        let gpus: Vec<Value> = self.glm.gpu_info().iter().map(|g| json!({
            "index": g.index, "name": g.name, "total": g.total, "free": g.free, "layers": [g.layers.0, g.layers.1],
            "expert_slots": g.expert_slots, "host_slots": g.host_slots})).collect();
        let cache = self.state.try_lock().ok().map(|st| json!({"entries": st.cache.len(), "bytes": st.cache.bytes(), "budget": st.cache.budget(),
                                                              "evictions": st.cache.evictions, "live_tokens": st.live.len()}));
        json!({"model": self.name, "version": crate::VERSION, "uptime_seconds": self.started.elapsed().as_secs_f64(),
               "context": self.max_ctx, "mtp": self.glm.mtp.is_some(), "gpus": gpus, "busy": self.busy.load(Ordering::Relaxed),
               "running": self.current.lock().unwrap().clone(), "served": self.next_id.load(Ordering::Relaxed) - 1,
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
        match (method, path) {
            ("GET", "/health") => respond(&mut s, 200, &json!({"status": "ok"})),
            ("GET", "/status") => {
                let st = self.status_json();
                respond(&mut s, 200, &json!({"busy": st["busy"], "prompt_cache": st["prompt_cache"]}))
            }
            ("GET", "/server/status") => respond(&mut s, 200, &self.status_json()),
            ("GET", "/server/requests") => {
                let done: Vec<Value> = self.done.lock().unwrap().iter().rev().cloned().collect();
                respond(&mut s, 200, &json!({"running": self.current.lock().unwrap().clone(), "done": done}))
            }
            ("GET", "/server/cache") => match self.state.try_lock() {
                Ok(st) => {
                    let list: Vec<Value> = st.cache.list().iter().map(|(t, b, _)| json!({"tokens": t, "bytes": b})).collect();
                    respond(&mut s, 200, &json!({"entries": list, "bytes": st.cache.bytes(), "budget": st.cache.budget(), "evictions": st.cache.evictions}))
                }
                Err(_) => respond(&mut s, 409, &json!({"error": {"message": "a request is running; ask again when it is done"}})),
            },
            ("POST", "/server/cache/clear") => match self.state.try_lock() {
                Ok(mut st) => {
                    let n = st.cache.clear();
                    respond(&mut s, 200, &json!({"dropped": n}))
                }
                Err(_) => respond(&mut s, 409, &json!({"error": {"message": "a request is running; ask again when it is done"}})),
            },
            ("POST", "/server/shutdown") => {
                STOP.store(true, Ordering::SeqCst);
                respond(&mut s, 200, &json!({"stopping": true, "running": self.busy.load(Ordering::Relaxed)}))
            }
            ("GET", "/v1/models") | ("GET", "/models") => respond(&mut s, 200, &json!({"object": "list", "data": [
                {"id": self.name, "object": "model", "owned_by": "nextsycl", "created": now(), "status": {"value": "loaded"}}]})),
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
                let body: Value = match serde_json::from_slice(&req.body) {
                    Ok(v) => v,
                    Err(e) => return respond(&mut s, 400, &json!({"error": {"message": format!("the body is not JSON: {e}")}})),
                };
                let via = s.kind();
                let r = self.chat(&mut s, &body, via);
                // a request that ended on an error is still recorded
                if let Some(mut row) = self.current.lock().unwrap().take() {
                    row["state"] = json!("failed");
                    row["error"] = json!(r.as_ref().err());
                    self.remember(row);
                }
                if let Err(e) = r {
                    respond(&mut s, 500, &json!({"error": {"message": e}}));
                }
            }
            _ => respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}})),
        }
    }

    fn remember(&self, row: Value) {
        let mut d = self.done.lock().unwrap();
        d.push_back(row);
        while d.len() > 100 {
            d.pop_front();
        }
    }

    /// Updates the running request's row.
    fn live(&self, f: impl FnOnce(&mut Value)) {
        if let Some(row) = self.current.lock().unwrap().as_mut() {
            f(row);
        }
    }

    fn chat(&self, s: &mut Conn, req: &Value, via: &str) -> Result<(), String> {
        let msgs: Vec<(String, String, Option<String>)> = req["messages"].as_array().ok_or("messages are required")?.iter().map(|m| {
            (m["role"].as_str().unwrap_or("user").to_string(), text_of(&m["content"]), m["reasoning_content"].as_str().map(str::to_string))
        }).collect();
        let effort = req["reasoning_effort"].as_str().or_else(|| req["chat_template_kwargs"]["reasoning_effort"].as_str())
            .and_then(Effort::parse).unwrap_or(self.default_effort);
        let messages: Vec<Message> = msgs.iter().map(|(r, c, rc)| Message { role: r, content: c, reasoning: rc.as_deref() }).collect();
        let ids = self.tok.encode(&ns_tok::glm_chat(&messages, effort));
        if ids.len() + 16 > self.max_ctx {
            return Err(format!("the prompt is {} tokens; the context is {}", ids.len(), self.max_ctx));
        }
        let max = req["max_tokens"].as_u64().or_else(|| req["max_completion_tokens"].as_u64()).unwrap_or(4096) as usize;
        let max = max.min(self.max_ctx - ids.len() - 1);
        let temp = req["temperature"].as_f64().unwrap_or(1.0) as f32;
        let top_p = req["top_p"].as_f64().unwrap_or(0.95) as f32;
        let stream = req["stream"].as_bool().unwrap_or(false);
        let id = format!("chatcmpl-{}", now());

        let mut st = self.state.lock().unwrap();
        self.busy.store(true, Ordering::Relaxed);
        let rid = self.next_id.fetch_add(1, Ordering::Relaxed);
        *self.current.lock().unwrap() = Some(json!({"id": rid, "via": via, "state": "reading", "started": now(), "prompt_tokens": ids.len(),
                                                    "max_tokens": max, "generated": 0}));
        let _idle = Guard(&self.busy);
        let st = &mut *st;
        let mut none = |_: &str, _: &ns_core::DevBuf| -> ns_core::Result<()> { Ok(()) };
        let t0 = Instant::now();
        // where to start: the live session when it holds a prefix of this prompt, a cached checkpoint when that is
        // longer, else the beginning
        let live_len = if st.live.len() < ids.len() && ids.starts_with(&st.live) { st.live.len() } else { 0 };
        let cached = if st.cache.enabled() { st.cache.best(&ids) } else { None };
        let (from, source) = match cached {
            Some((i, len)) if len > live_len => {
                let ck = st.cache.get(i);
                self.glm.restore(&mut st.work, ck).map_err(|e| e.0)?;
                (len, "cache")
            }
            _ if live_len > 0 => (live_len, "live"),
            _ => {
                self.glm.reset_session(&mut st.work).map_err(|e| e.0)?;
                (0, "none")
            }
        };
        st.live.clear(); // until this prompt is read, the session is in between
        // read the rest, stopping at the checkpoint positions past `from` to save the state there
        let mut logits = Vec::new();
        let mut at = from;
        let mut saved = 0;
        for stop in self.stops(&ids).into_iter().filter(|p| *p > from) {
            logits = self.glm.feed(&mut st.work, &ids[at..stop], &mut none).map_err(|e| e.0)?;
            at = stop;
            if st.cache.enabled() && !st.cache.touch(&ids[..stop]) {
                let ck = self.glm.save(&st.work).map_err(|e| e.0)?;
                if st.cache.put(ids[..stop].to_vec(), ck) {
                    saved += 1;
                }
            }
        }
        if at < ids.len() {
            logits = self.glm.feed(&mut st.work, &ids[at..], &mut none).map_err(|e| e.0)?;
        }
        st.live = ids.clone();
        let prefill = t0.elapsed().as_secs_f64();
        self.live(|r| {
            r["state"] = json!("generating");
            r["reused"] = json!(from);
            r["source"] = json!(source);
            r["read_seconds"] = json!(prefill);
        });

        if stream {
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
        }
        let send = |s: &mut Conn, delta: Value, finish: Option<&str>| -> bool {
            let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": self.name,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            write!(s, "data: {chunk}\n\n").and_then(|_| s.flush()).is_ok()
        };
        if stream {
            send(s, json!({"role": "assistant"}), None);
        }
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut thinking = true; // the prompt ends in <think>
        let mut pending: Vec<u8> = Vec::new();
        let mut n = 0;
        let mut finish = "length";
        let t1 = Instant::now();
        let mut dec = self.glm.decoder(logits, true);
        let mut out: std::collections::VecDeque<u32> = Default::default();
        let mut committed: Vec<u32> = Vec::new();
        while n < max {
            if out.is_empty() {
                let rng = &mut st.rng;
                let mut draw = |l: &[f32]| sample(l, temp, top_p, rng);
                let toks = self.glm.step(&mut st.work, &mut dec, &mut draw, &mut none).map_err(|e| e.0)?;
                committed.extend(&toks);
                out.extend(toks);
            }
            let next = out.pop_front().unwrap_or_default();
            if self.tok.stop.contains(&next) {
                finish = "stop";
                break;
            }
            n += 1;
            let el = t1.elapsed().as_secs_f64();
            self.live(|r| {
                r["generated"] = json!(n);
                r["tok_s"] = json!(n as f64 / el.max(1e-9));
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
                    if !p.is_empty() && !send(s, if is_r { json!({"reasoning_content": p}) } else { json!({"content": p}) }, None) {
                        finish = "client gone";
                    }
                }
                if finish == "client gone" {
                    break;
                }
            }
        }
        let dt = t1.elapsed().as_secs_f64();
        // what the session holds now: the prompt and the committed tokens it has fed (the last ones may be pending)
        let fed = st.work.pos.saturating_sub(ids.len()).min(committed.len());
        st.live.extend_from_slice(&committed[..fed]);
        eprintln!("[request: {} prompt tokens ({} reused from {source}, {} fed in {prefill:.1} s, {saved} checkpoint(s) saved; cache {} entries, {:.2} GiB), {n} generated in {dt:.1} s ({:.2} tok/s), drafts {}/{} accepted, {finish}]",
                  ids.len(), from, ids.len() - from, st.cache.len(), st.cache.bytes() as f64 / (1u64 << 30) as f64, n as f64 / dt.max(1e-9),
                  dec.accepted, dec.drafted);
        let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": n, "total_tokens": ids.len() + n});
        if let Some(mut row) = self.current.lock().unwrap().take() {
            row["state"] = json!("done");
            row["finish"] = json!(finish);
            row["generated"] = json!(n);
            row["tok_s"] = json!(n as f64 / dt.max(1e-9));
            row["generate_seconds"] = json!(dt);
            row["drafts"] = json!([dec.accepted, dec.drafted]);
            row["checkpoints_saved"] = json!(saved);
            self.remember(row);
        }
        let finish = if finish == "client gone" { "stop" } else { finish };
        if stream {
            send(s, json!({}), Some(finish));
            let _ = write!(s, "data: {}\n\ndata: [DONE]\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": self.name,
                                                                      "choices": [], "usage": usage}));
        } else {
            respond(s, 200, &json!({"id": id, "object": "chat.completion", "created": now(), "model": self.name,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content.trim(),
                                                 "reasoning_content": reasoning.trim()}, "finish_reason": finish}],
                                    "usage": usage,
                                    "timings": {"prompt_n": ids.len() - from, "prompt_ms": prefill * 1000.0, "predicted_n": n,
                                                "predicted_per_second": n as f64 / dt.max(1e-9)}}));
        }
        Ok(())
    }
}

/// Clears the busy flag when a request ends (also on an error).
struct Guard<'a>(&'a AtomicBool);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}
