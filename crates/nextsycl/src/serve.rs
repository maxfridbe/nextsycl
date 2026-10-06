//! `nextsycl serve`: an OpenAI-compatible server for one loaded model (what the box's model switch and Open WebUI
//! talk to).
//!
//!     GET  /health                 {"status": "ok"} once the model is loaded
//!     GET  /v1/models              the one model
//!     GET  /status                 {"busy": bool}
//!     POST /v1/chat/completions    messages, max_tokens, temperature, top_p, stream, reasoning_effort (or
//!                                  chat_template_kwargs.reasoning_effort): low | high | max
//!
//! One request runs at a time (the others wait). The prompt cache (cache.rs, --prompt-cache-mib) keeps the
//! conversation state at three points of every prompt - the end of its first turn (a system prompt other
//! conversations share), the start of its last user turn (an edited or regenerated message), its end (the next turn
//! of the conversation) - and a request mounts the longest cached prefix of its tokens, or continues the live
//! session when that is longer, and reads only the rest. GLM's thinking comes back as `reasoning_content`, the
//! answer as `content`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_engine::glm5next::{Glm, Session};

use crate::cache::PromptCache;
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

fn respond(s: &mut TcpStream, code: u16, body: &Value) {
    let b = body.to_string();
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
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
                    busy: AtomicBool::new(false) })
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
    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
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
        eprintln!("[serving {} on http://{addr}]", self.name);
        loop {
            if STOP.load(Ordering::SeqCst) {
                // finish what runs (the request's thread holds the conversation's lock)
                let _wait = self.state.lock().unwrap();
                eprintln!("[stopping: no request running]");
                return Ok(());
            }
            match l.accept() {
                Ok((c, _)) => {
                    let _ = c.set_nonblocking(false);
                    let me = self.clone();
                    std::thread::spawn(move || me.handle(c));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(std::time::Duration::from_millis(50)),
                Err(_) => {}
            }
        }
    }

    fn handle(&self, mut s: TcpStream) {
        let mut r = BufReader::new(match s.try_clone() {
            Ok(c) => c,
            Err(_) => return,
        });
        let mut line = String::new();
        if r.read_line(&mut line).is_err() {
            return;
        }
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
        let mut len = 0usize;
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).is_err() || h.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
        }
        let mut body = vec![0u8; len.min(64 << 20)];
        if r.read_exact(&mut body).is_err() {
            return;
        }
        let path = path.split('?').next().unwrap_or("").to_string();
        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => respond(&mut s, 200, &json!({"status": "ok"})),
            ("GET", "/status") => {
                let busy = self.busy.load(Ordering::Relaxed);
                // the cache's figures when no request holds the state (a status call never waits)
                let cache = self.state.try_lock().ok().map(|st| json!({"entries": st.cache.len(), "bytes": st.cache.bytes(),
                                                                      "evictions": st.cache.evictions, "live_tokens": st.live.len()}));
                respond(&mut s, 200, &json!({"busy": busy, "prompt_cache": cache}))
            }
            ("GET", "/v1/models") | ("GET", "/models") => respond(&mut s, 200, &json!({"object": "list", "data": [
                {"id": self.name, "object": "model", "owned_by": "nextsycl", "created": now(), "status": {"value": "loaded"}}]})),
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
                let req: Value = match serde_json::from_slice(&body) {
                    Ok(v) => v,
                    Err(e) => return respond(&mut s, 400, &json!({"error": {"message": format!("the body is not JSON: {e}")}})),
                };
                if let Err(e) = self.chat(&mut s, &req) {
                    respond(&mut s, 500, &json!({"error": {"message": e}}));
                }
            }
            _ => respond(&mut s, 404, &json!({"error": {"message": format!("no route {method} {path}")}})),
        }
    }

    fn chat(&self, s: &mut TcpStream, req: &Value) -> Result<(), String> {
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

        if stream {
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
        }
        let send = |s: &mut TcpStream, delta: Value, finish: Option<&str>| -> bool {
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
