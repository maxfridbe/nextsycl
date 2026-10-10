//! The model switcher (`nextsycl switch`): one OpenAI endpoint for a chat front end (Open WebUI) over the one model
//! server that runs at a time. It lists every registered chat model; a request naming one that is not loaded asks
//! the owner of the cards (the video studio's `llm.mode`, which also knows when a render holds a GPU) to swap, waits
//! until the server answers as that model, then passes the request through - streamed answers included. A request
//! for the loaded model goes straight through.
//!
//! Two clients wanting different models must not take the card from each other mid-conversation: a swap waits for
//! one already in progress, and the loaded model is not swapped out while it is answering or within `in_use` of its
//! last request through here - that request gets a 409 naming the loaded model. Per-model switches from the
//! registry: `tools: false` drops a request's tool definitions, `tasks: false` declines Open WebUI's background
//! tasks ("### Task:" prompts). The log carries sizes only, never content.
//!
//! With `idle`, the loaded model is unloaded (the studio's llm.mode "none") once nothing has asked for it that long
//! and it is not answering; the next request loads it again.
//!
//! With an image server (`images`), the images API (`/v1/images/...`: generations, edits, files) goes to it and its
//! model is listed beside the chat ones while it answers - one base URL for a front end's chat and pictures.

use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::http::{self, Conn, Target};

/// A chat model the registry offers: its id (the front end's, the studio's mode and the name the server answers
/// under), title, whether it takes tools and background tasks
#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub title: String,
    pub tools: bool,
    pub tasks: bool,
}

/// The registry's enabled chat models, read on every request (an added or disabled model shows at once)
pub type Models = Box<dyn Fn() -> Vec<Model> + Send + Sync>;

pub struct Switch {
    pub models: Models,
    /// old ids -> registered ones (chats that named a model by an earlier id keep working; not listed)
    pub aliases: HashMap<String, String>,
    /// the model server
    pub upstream: Target,
    /// the studio's llm.mode route (host:port and path)
    pub studio: (Target, String),
    /// how long a swap may take (a render can hold the card longer than a load)
    pub wait: Duration,
    pub in_use: Duration,
    /// the image server (`nextsycl image start`), when there is one
    pub images: Option<Target>,
    /// unload the chat model after this long without a request
    pub idle: Option<Duration>,
    swap: Mutex<()>,
    last_use: Mutex<HashMap<String, Instant>>,
}

/// Python's truth for a JSON value the studio sends (null, false, "" and 0 are false)
fn truthy(v: &Value) -> bool {
    !(v.is_null() || v == &json!(false) || v == &json!("") || v == &json!(0))
}

fn err(code: u16, m: impl Into<String>) -> (u16, Value) {
    (code, json!({"error": {"message": m.into(), "type": "model_unavailable"}}))
}

impl Switch {
    pub fn new(models: Models, aliases: HashMap<String, String>, upstream: Target, studio_url: &str, wait: Duration, in_use: Duration)
               -> Result<Switch, String> {
        let (host, path) = http::split_url(studio_url)?;
        Ok(Switch { models, aliases, upstream, studio: (Target::Tcp(host), path), wait, in_use, images: None, idle: None, swap: Mutex::new(()), last_use: Mutex::new(HashMap::new()) })
    }

    pub fn run(self: Arc<Self>, addr: &str) -> Result<(), String> {
        let l = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        if let Some(idle) = self.idle {
            let me = self.clone();
            std::thread::spawn(move || me.unload_idle(idle));
        }
        eprintln!("model switch on {addr} -> {}, swaps via http://{}{}", self.upstream, self.studio.0, self.studio.1);
        for c in l.incoming().flatten() {
            let me = self.clone();
            std::thread::spawn(move || me.handle(Conn::Tcp(c)));
        }
        Ok(())
    }

    /// Every 30 s: the loaded model unloaded when nobody has asked for it for `idle` (a model loaded from elsewhere
    /// counts from when it was first seen here), unless it is answering or a swap runs
    fn unload_idle(&self, idle: Duration) {
        loop {
            std::thread::sleep(Duration::from_secs(30));
            let models = (self.models)();
            let Some(on) = self.loaded().and_then(|c| self.served(&c, &models)) else { continue };
            let since = *self.last_use.lock().unwrap().entry(on.clone()).or_insert_with(Instant::now);
            if since.elapsed() < idle || self.busy() {
                continue;
            }
            let Ok(_g) = self.swap.try_lock() else { continue };
            eprintln!("{on}: idle for {} min - unloading (the next request loads it again)", idle.as_secs() / 60);
            match self.studio(Some("none")) {
                Ok(_) => {
                    self.last_use.lock().unwrap().remove(&on);
                }
                Err(e) => eprintln!("  the studio's llm.mode: {e}"),
            }
        }
    }

    /// The studio's llm.mode: its state, or (with a mode) a swap to it
    fn studio(&self, mode: Option<&str>) -> Result<Value, String> {
        let body = mode.map_or_else(|| json!({}), |m| json!({"mode": m}));
        let v = http::call_for(&self.studio.0, "POST", &self.studio.1, Some(&body), if mode.is_some() { 300 } else { 15 })?;
        Ok(v.get("result").cloned().unwrap_or(v))
    }

    /// The id the model server answers as now, or none while nothing is up
    fn loaded(&self) -> Option<String> {
        let v = http::call_for(&self.upstream, "GET", "/v1/models", None, 4).ok()?;
        let m = v["data"].get(0)?;
        let up = m["status"]["value"].as_str().unwrap_or("loaded") == "loaded";
        up.then(|| m["id"].as_str().map(str::to_string)).flatten()
    }

    /// The registered id for what the server answers as: the longest id it ends with (ids extend one another:
    /// a model and its long-context variants)
    fn served(&self, cur: &str, models: &[Model]) -> Option<String> {
        models.iter().filter(|m| cur.ends_with(&m.id)).max_by_key(|m| m.id.len()).map(|m| m.id.clone())
    }

    /// Whether the model server is answering right now (any client)
    fn busy(&self) -> bool {
        http::call_for(&self.upstream, "GET", "/status", None, 4).ok().and_then(|v| v["busy"].as_bool()).unwrap_or(false)
    }

    /// `want` loaded, swapping if another model holds the server: an error to answer with, or none
    fn ensure(&self, want: &Model) -> Option<(u16, Value)> {
        let _g = self.swap.lock().unwrap_or_else(|p| p.into_inner());
        // a swap in progress (from anywhere) finishes first
        let t0 = Instant::now();
        while t0.elapsed() < self.wait {
            match self.studio(None) {
                Ok(s) if truthy(&s["starting"]) => std::thread::sleep(Duration::from_secs(3)),
                _ => break,
            }
        }
        let models = (self.models)();
        let cur = self.loaded();
        let other = cur.as_deref().and_then(|c| self.served(c, &models));
        if other.as_deref() == Some(want.id.as_str()) {
            return None;
        }
        if let Some(c) = &cur {
            let recent = other.as_ref().is_some_and(|o| self.last_use.lock().unwrap().get(o).is_some_and(|t| t.elapsed() < self.in_use));
            if recent || self.busy() {
                let oname = other.as_ref().and_then(|o| models.iter().find(|m| &m.id == o)).map_or(c.clone(), |m| m.title.clone());
                return Some(err(409, format!("{oname} is loaded and in use; switching to {} would cut that conversation off. Try again in a minute, or pick {oname}.",
                                             want.title)));
            }
        }
        if let Err(e) = self.studio(Some(&want.id)) {
            return Some(err(503, format!("the model switch (the studio's llm.mode) did not answer: {e}")));
        }
        let t0 = Instant::now();
        while t0.elapsed() < self.wait {
            if self.loaded().as_deref().and_then(|c| self.served(c, &models)).as_deref() == Some(want.id.as_str()) {
                return None;
            }
            std::thread::sleep(Duration::from_secs(3));
        }
        let why = match self.studio(None) {
            Ok(s) if s["mode"] == want.id.as_str() && !truthy(&s["starting"]) => " (a video render holds the GPU)",
            _ => "",
        };
        Some(err(503, format!("{} did not come up in {} minutes{why}", want.title, self.wait.as_secs() / 60)))
    }

    fn handle(&self, mut s: Conn) {
        let req = match http::read_request(&mut s) {
            Ok(r) => r,
            Err(e) => return http::respond(&mut s, 400, &json!({"error": {"message": e}})),
        };
        if let Some(img) = self.images.as_ref().filter(|_| req.path.starts_with("/v1/images/") || req.path.starts_with("/images/")) {
            eprintln!("images: {} {} ({} bytes)", req.method, req.path.split('?').next().unwrap_or(""), req.body.len());
            if let Err(e) = http::forward(&mut s, img, &req) {
                eprintln!("  passing it on: {e}");
                if e.contains("does not answer") {
                    http::respond(&mut s, 502, &json!({"error": {"message": "the image server is not running (nextsycl image start)"}}));
                }
            }
            return;
        }
        if req.method == "GET" && req.path.trim_end_matches('/').ends_with("/models") {
            let models = (self.models)();
            let cur = self.loaded();
            let on = cur.as_deref().and_then(|c| self.served(c, &models));
            let mut data: Vec<Value> = models.iter().map(|m| json!({"id": m.id, "object": "model", "owned_by": "nextsycl", "name": m.title,
                                                                     "loaded": on.as_deref() == Some(m.id.as_str())})).collect();
            // the image server's model, while it answers
            if let Some(v) = self.images.as_ref().and_then(|t| http::call_for(t, "GET", "/v1/models", None, 2).ok()) {
                data.extend(v["data"].as_array().into_iter().flatten().map(|m| json!({"id": m["id"], "object": "model", "owned_by": "nextsycl",
                                                                                      "name": m["id"], "loaded": true, "kind": "image"})));
            }
            return http::respond(&mut s, 200, &json!({"object": "list", "data": data}));
        }
        if req.method != "POST" {
            if let Err(e) = http::forward(&mut s, &self.upstream, &req) {
                http::respond(&mut s, 502, &json!({"error": {"message": format!("the model server is not answering: {e}")}}));
            }
            return;
        }
        let mut body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let mut changed = false;
        let mut want = body["model"].as_str().map(str::to_string);
        if let Some(new) = want.as_ref().and_then(|w| self.aliases.get(w)).cloned() {
            body["model"] = json!(new);
            want = Some(new);
            changed = true;
        }
        let model = want.as_ref().and_then(|w| (self.models)().into_iter().find(|m| &m.id == w));
        if let Some(m) = &model {
            let msgs = body["messages"].as_array().cloned().unwrap_or_default();
            let mut by_role: std::collections::BTreeMap<String, usize> = Default::default();
            for x in &msgs {
                let n = x["content"].as_str().map_or_else(|| x["content"].to_string().len(), str::len);
                *by_role.entry(x["role"].as_str().unwrap_or("?").to_string()).or_default() += n;
            }
            let tools = body["tools"].as_array().map_or(0, Vec::len);
            let tool_chars = if body["tools"].is_null() { 2 } else { body["tools"].to_string().len() };
            let others: Vec<&String> = body.as_object().map(|o| o.keys().filter(|k| !["messages", "tools", "model"].contains(&k.as_str())).collect()).unwrap_or_default();
            eprintln!("request for {}: {} messages, chars by role {by_role:?}, {tools} tools ({} chars), max_tokens {}, other keys {others:?}",
                      m.id, msgs.len(), tool_chars, body["max_tokens"]);
            if !m.tasks {
                let last = msgs.iter().rev().find(|x| x["role"] == "user").and_then(|x| x["content"].as_str()).unwrap_or("");
                if last.trim_start().starts_with("### Task:") {
                    eprintln!("  a background task: declined for this model");
                    let (c, v) = err(503, format!("{} does not take background tasks", m.title));
                    return http::respond(&mut s, c, &v);
                }
            }
            if !m.tools {
                let a = body.as_object_mut().map(|o| o.remove("tools").is_some() | o.remove("tool_choice").is_some()).unwrap_or(false);
                if a {
                    eprintln!("  tools dropped for this model");
                    changed = true;
                }
            }
            if let Some((c, v)) = self.ensure(m) {
                eprintln!("  {}", v["error"]["message"].as_str().unwrap_or(""));
                return http::respond(&mut s, c, &v);
            }
            self.last_use.lock().unwrap().insert(m.id.clone(), Instant::now());
        }
        let bytes = if changed { serde_json::to_vec(&body).unwrap_or_default() } else { req.body.clone() };
        if let Err(e) = http::forward_body(&mut s, &self.upstream, &req, &bytes) {
            eprintln!("  passing it on: {e}");
            // nothing was sent yet when the server did not answer at all (else the client went away mid-answer)
            if e.contains("does not answer") {
                http::respond(&mut s, 502, &json!({"error": {"message": format!("the model server is not answering: {e}")}}));
            }
        }
        if let Some(m) = &model {
            self.last_use.lock().unwrap().insert(m.id.clone(), Instant::now());
        }
    }
}
