//! `nextsycl serve`: an OpenAI-compatible server for one loaded model (what the box's model switch and Open WebUI
//! talk to).
//!
//!     GET  /health                 {"status": "ok"} once the model is loaded
//!     GET  /v1/models              the one model
//!     GET  /status                 {"busy": bool}
//!     POST /v1/chat/completions    messages, max_tokens, temperature, top_p, stream, reasoning_effort (or
//!                                  chat_template_kwargs.reasoning_effort): low | high | max
//!
//! One request runs at a time (the others wait). The conversation is kept at the end of each prompt (the recurrent
//! states and the MLA caches): a request whose prompt begins with the previous prompt feeds only the rest. GLM's
//! thinking comes back as `reasoning_content`, the answer as `content`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ns_engine::glm5next::{Glm, Session};
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

/// The working session, and a copy of it at the end of the last prompt with that prompt's tokens.
struct Conv {
    work: Session,
    snap: Session,
    snap_tokens: Vec<u32>,
    rng: Rng,
}

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
    pub fn new(glm: Glm<'static>, tok: Tokenizer, name: String, max_ctx: usize, default_effort: Effort) -> Result<Server, String> {
        let work = glm.session(max_ctx).map_err(|e| e.0)?;
        let snap = glm.session(max_ctx).map_err(|e| e.0)?;
        Ok(Server { glm, tok, name, max_ctx, default_effort, state: Mutex::new(Conv { work, snap, snap_tokens: Vec::new(), rng: Rng(0x5DEECE66D) }),
                    busy: AtomicBool::new(false) })
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
            ("GET", "/status") => respond(&mut s, 200, &json!({"busy": self.busy.load(Ordering::Relaxed)})),
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
        // resume from the previous prompt when this one continues it
        let reuse = !st.snap_tokens.is_empty() && ids.len() > st.snap_tokens.len() && ids.starts_with(&st.snap_tokens);
        let from = if reuse {
            self.glm.copy_session(&mut st.work, &st.snap).map_err(|e| e.0)?;
            st.snap_tokens.len()
        } else {
            self.glm.reset_session(&mut st.work).map_err(|e| e.0)?;
            0
        };
        let mut logits = self.glm.feed(&mut st.work, &ids[from..], &mut none).map_err(|e| e.0)?;
        self.glm.copy_session(&mut st.snap, &st.work).map_err(|e| e.0)?;
        st.snap_tokens = ids.clone();
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
        for _ in 0..max {
            let next = sample(&logits, temp, top_p, &mut st.rng);
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
            logits = self.glm.forward(&mut st.work, &[next], &mut none).map_err(|e| e.0)?;
        }
        let dt = t1.elapsed().as_secs_f64();
        eprintln!("[request: {} prompt tokens ({} reused, {} fed in {prefill:.1} s), {n} generated in {dt:.1} s ({:.2} tok/s), {finish}]",
                  ids.len(), from, ids.len() - from, n as f64 / dt.max(1e-9));
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
