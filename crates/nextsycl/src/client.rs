//! The host command line's side of the server's control socket (sycl-h3's client.rs, for a language model):
//!
//!     nextsycl status [--no-stream]     the server, live (like `docker stats`): the model, its GPUs, the request
//!                                       running, the prompt cache; --no-stream: once
//!     nextsycl ps [-a]                  the request running and the last ones (-a: all the server remembers)
//!     nextsycl inspect <id>             one request in full, as JSON
//!     nextsycl cache [ls | clear]       the prompt cache's checkpoints, or drop them
//!     nextsycl chat <text> [--effort low|high|max] [--max N] [--temp T]
//!                                       one request, streamed: the thinking dimmed, then the answer
//!
//! The server answers on a Unix socket (config.rs); `main` sets where.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::http::{self, Target};

/// The server's socket; set once by `main`.
pub static SERVER: OnceLock<Target> = OnceLock::new();

fn target() -> &'static Target {
    SERVER.get().expect("the server's socket is set before any call")
}

fn not_running(e: String) -> String {
    if e.starts_with("nothing answers") {
        "the server is not running: nextsycl start".into()
    } else {
        e
    }
}

pub fn get(path: &str) -> Result<Value, String> {
    http::call(target(), "GET", path, None).map_err(not_running)
}

pub fn post(path: &str, body: Option<&Value>) -> Result<Value, String> {
    http::call(target(), "POST", path, body).map_err(not_running)
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// 75 -> "1m15s"
fn dur(s: f64) -> String {
    let s = s.max(0.0) as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

/// a JSON value as column text (`{:>7}` pads strings; a Value ignores the width)
fn t(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        Value::Null => "-".into(),
        o => o.to_string(),
    }
}

fn gib(b: f64) -> String {
    format!("{:.1}GiB", b / (1u64 << 30) as f64)
}

fn card(name: &str) -> String {
    name.replace("Intel(R) ", "").replace("(TM)", "").replace(" Graphics", "").trim().to_string()
}

fn status_table(st: &Value) -> String {
    let mut out = format!(
        "{} - up {}, context {}, MTP {}, {} request(s) served{}\n\n",
        st["model"].as_str().unwrap_or("?"),
        dur(st["uptime_seconds"].as_f64().unwrap_or(0.0)),
        st["context"],
        if st["mtp"].as_bool() == Some(true) { "on" } else { "off" },
        st["served"],
        if st["stopping"].as_bool() == Some(true) { " - stopping" } else { "" }
    );
    out += &format!("{:<4} {:<18} {:>16} {:>9} {:>8} {:>16}\n", "GPU", "CARD", "VRAM USED", "FREE", "LAYERS", "EXPERTS VRAM/HOST");
    for g in st["gpus"].as_array().cloned().unwrap_or_default() {
        let total = g["total"].as_f64().unwrap_or(0.0);
        let free = g["free"].as_f64();
        let used = free.map_or("-".into(), |f| format!("{} / {}", gib(total - f), gib(total)));
        let layers = g["layers"].as_array().map_or("-".into(), |l| format!("{}-{}", l[0], l[1].as_u64().unwrap_or(1).saturating_sub(1)));
        out += &format!("{:<4} {:<18} {:>16} {:>9} {:>8} {:>16}\n", t(&g["index"]), card(g["name"].as_str().unwrap_or("?")), used,
                        free.map_or("-".into(), gib), layers, format!("{} / {}", g["expert_slots"], g["host_slots"]));
    }
    out += "\n";
    match st["running"].as_object() {
        Some(_) => {
            let r = &st["running"];
            let reused = match r["source"].as_str() {
                Some(src) => format!("{} reused ({src})", r["reused"]),
                None => "reading".into(),
            };
            out += &format!("request #{} ({}): {}, prompt {} tokens, {}, {} / {} generated{}, {}\n", r["id"], r["via"].as_str().unwrap_or("?"),
                            r["state"].as_str().unwrap_or("?"), r["prompt_tokens"], reused, r["generated"], r["max_tokens"],
                            r["tok_s"].as_f64().map_or(String::new(), |t| format!(" at {t:.1} tok/s")),
                            dur(now() - r["started"].as_f64().unwrap_or(now())));
        }
        None => out += "idle\n",
    }
    match st["prompt_cache"].as_object() {
        Some(_) => {
            let c = &st["prompt_cache"];
            out += &format!("prompt cache: {} checkpoint(s), {} of {}, {} evicted; the live session holds {} tokens\n", c["entries"],
                            gib(c["bytes"].as_f64().unwrap_or(0.0)), gib(c["budget"].as_f64().unwrap_or(0.0)), c["evictions"], c["live_tokens"]);
        }
        None => out += "prompt cache: (busy)\n",
    }
    out
}

/// `nextsycl status [--no-stream]`
pub fn status(raw: &[String]) -> Result<(), String> {
    let once = raw.iter().any(|a| a == "--no-stream");
    let tty = std::io::stdout().is_terminal();
    loop {
        let table = match get("/server/status") {
            Ok(st) => status_table(&st),
            Err(e) => format!("server: {e}\n"),
        };
        let mut out = std::io::stdout().lock();
        if once {
            let _ = write!(out, "{table}");
            return Ok(());
        }
        if tty {
            // redraw in place: home, the table, clear what is left of the screen
            let _ = write!(out, "\x1b[H{}\x1b[J", table.replace('\n', "\x1b[K\n"));
        } else {
            let _ = write!(out, "{table}");
        }
        let _ = out.flush();
        drop(out);
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// `nextsycl ps [-a]`
pub fn ps(raw: &[String]) -> Result<(), String> {
    let all = raw.iter().any(|a| a == "-a" || a == "--all");
    let v = get("/server/requests")?;
    println!("{:<6} {:<7} {:<11} {:>7} {:>16} {:>7} {:>10} {:>7} {:<7} {:>8}", "ID", "VIA", "STATE", "PROMPT", "REUSED", "READ", "GENERATED", "TOK/S",
             "FINISH", "AGO");
    let mut rows: Vec<Value> = v["running"].as_object().map(|_| vec![v["running"].clone()]).unwrap_or_default();
    rows.extend(v["done"].as_array().cloned().unwrap_or_default().into_iter().take(if all { usize::MAX } else { 10 }));
    for r in rows {
        let reused = r["source"].as_str().map_or("-".into(), |s| format!("{} {s}", r["reused"]));
        println!("{:<6} {:<7} {:<11} {:>7} {:>16} {:>7} {:>10} {:>7} {:<7} {:>8}", format!("#{}", r["id"]), r["via"].as_str().unwrap_or("?"),
                 r["state"].as_str().unwrap_or("?"), t(&r["prompt_tokens"]), reused,
                 r["read_seconds"].as_f64().map_or("-".into(), |s| format!("{s:.1}s")), t(&r["generated"]),
                 r["tok_s"].as_f64().map_or("-".into(), |t| format!("{t:.1}")), r["finish"].as_str().unwrap_or("-"),
                 dur(now() - r["started"].as_f64().unwrap_or(now())));
    }
    Ok(())
}

/// `nextsycl inspect <id>`: the server's record of one request (an ID from `ps`, with or without the #)
pub fn inspect(raw: &[String]) -> Result<(), String> {
    let id = raw.first().ok_or("nextsycl inspect <id>")?.trim_start_matches('#');
    if id.parse::<u64>().is_err() {
        return Err(format!("nextsycl inspect: {id:?} is not a request ID (nextsycl ps lists them)"));
    }
    let v = get(&format!("/server/requests/{id}"))?;
    println!("{}", serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?);
    Ok(())
}

/// `nextsycl cache [ls | clear]`
pub fn cache(raw: &[String]) -> Result<(), String> {
    match raw.first().map(String::as_str) {
        None | Some("ls") => {
            let v = get("/server/cache")?;
            println!("{:>4} {:>8} {:>10}", "#", "TOKENS", "SIZE");
            for (i, e) in v["entries"].as_array().cloned().unwrap_or_default().iter().enumerate() {
                println!("{:>4} {:>8} {:>10}", i + 1, t(&e["tokens"]), gib(e["bytes"].as_f64().unwrap_or(0.0)));
            }
            println!("{} of {}, {} evicted (most recently used first)", gib(v["bytes"].as_f64().unwrap_or(0.0)), gib(v["budget"].as_f64().unwrap_or(0.0)),
                     v["evictions"]);
            Ok(())
        }
        Some("clear") => {
            let v = post("/server/cache/clear", None)?;
            println!("dropped {} checkpoint(s)", v["dropped"]);
            Ok(())
        }
        Some(o) => Err(format!("nextsycl cache: ls or clear, not {o:?}")),
    }
}

/// `nextsycl chat <text> [--effort E] [--max N] [--temp T]`: one request through the socket, streamed.
pub fn chat(raw: &[String]) -> Result<(), String> {
    let mut text = Vec::new();
    let mut body = json!({"stream": true});
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--effort" => body["reasoning_effort"] = json!(it.next().ok_or("--effort low|high|max")?),
            "--max" => body["max_tokens"] = json!(it.next().and_then(|v| v.parse::<u64>().ok()).ok_or("--max N")?),
            "--temp" => body["temperature"] = json!(it.next().and_then(|v| v.parse::<f64>().ok()).ok_or("--temp T")?),
            _ => text.push(a.clone()),
        }
    }
    if text.is_empty() {
        return Err("nextsycl chat <text>".into());
    }
    body["messages"] = json!([{"role": "user", "content": text.join(" ")}]);
    let (status, r) = http::send(target(), "POST", "/v1/chat/completions", Some(&body)).map_err(not_running)?;
    if status != 200 {
        return Err(format!("the server answered {status}"));
    }
    let tty = std::io::stdout().is_terminal();
    let (dim, plain) = if tty { ("\x1b[2m", "\x1b[0m") } else { ("", "") };
    let mut out = std::io::stdout();
    let mut in_thinking = false;
    for line in r.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        let d = &v["choices"][0]["delta"];
        if let Some(t) = d["reasoning_content"].as_str() {
            if !in_thinking {
                let _ = write!(out, "{dim}");
                in_thinking = true;
            }
            let _ = write!(out, "{t}");
        }
        if let Some(t) = d["content"].as_str() {
            if in_thinking {
                let _ = write!(out, "{plain}\n\n");
                in_thinking = false;
            }
            let _ = write!(out, "{t}");
        }
        if let Some(u) = v["usage"].as_object() {
            let _ = write!(out, "{plain}\n\n{dim}[{} prompt tokens, {} generated]{plain}", u["prompt_tokens"], u["completion_tokens"]);
        }
        let _ = out.flush();
    }
    println!();
    Ok(())
}
