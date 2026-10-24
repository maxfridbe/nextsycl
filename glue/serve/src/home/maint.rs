//! Maintenance mode in the front door (crate::maintenance has the notice and its file): turning it on and off, what
//! it refuses, what it pauses. On: the video studio's queue paused (what was paused before is kept and put back when
//! the mode ends), image, audio and 3D jobs held in their queues, idle models unloaded when asked (`unload`); work
//! running finishes. Off: the queues go on.

use std::sync::Arc;

use serde_json::{json, Value};

use super::api::rpc_error;
use super::{now, Home};
use crate::http::{self, call_for, Conn, Request};
use crate::maintenance;

type Fail = (u16, String);

/// The methods that start work or load a model (refused in maintenance); the rest - state, listings, status,
/// cancels, guides, uploads, unloading - answer as always
pub(super) fn starts_work(method: &str) -> bool {
    matches!(method, "system.enable" | "system.load" | "llm.chat" | "image.generate" | "image.edit" | "image.ocr" | "audio.generate" | "audio.speak"
                     | "audio.transcribe" | "audio.voices.design" | "audio.voices.enroll" | "video.generate" | "model3d.generate" | "model3d.rig"
                     | "image.upscale" | "video.upscale" | "image.vectorize")
}

/// The OpenAI routes that start work
fn openai_starts_work(method: &str, path: &str) -> bool {
    method == "POST" && matches!(path, "/v1/chat/completions" | "/v1/completions" | "/v1/images/generations" | "/v1/images/edits" | "/v1/audio/speech" | "/v1/audio/transcriptions" | "/v1/model3d/generate" | "/v1/model3d/rig"
                                       | "/v1/images/upscale" | "/v1/videos/upscale" | "/v1/images/vectorize")
}

impl Home {
    /// The notice while the mode is on
    pub(crate) fn maintenance(&self) -> Option<Value> {
        maintenance::read(&self.maint_file)
    }

    /// Whether a request may pass the mode: its header carries the bypass key
    fn bypasses(&self, req: &Request) -> bool {
        let key = self.maint_bypass.lock().unwrap().clone();
        !key.is_empty() && http::header(&req.head, "x-maintenance-bypass").is_some_and(|h| h.trim() == key)
    }

    /// A request the mode refuses, answered (true): /rpc in its envelope (HTTP 200, error.status 503), /v1 as an
    /// OpenAI error (503); both with Retry-After and the notice
    pub(super) fn maintenance_refused(&self, s: &mut Conn, req: &Request) -> bool {
        let Some(n) = self.maintenance() else { return false };
        let rpc = req.path.strip_prefix("/rpc/");
        let refuse = match rpc {
            Some(m) => starts_work(m),
            None => openai_starts_work(&req.method, &req.path),
        };
        if !refuse || self.bypasses(req) {
            return false;
        }
        let m = maintenance::message(&n);
        let retry = format!("Retry-After: {}\r\nCache-Control: no-cache\r\n", n["retry_after_s"].as_u64().unwrap_or(900));
        let public = public(&n);
        let (st, body) = match rpc {
            Some(method) => {
                let mut e = rpc_error(method, 503, Some("maintenance"), &m);
                e["maintenance"] = public.clone();
                e["retry_after_s"] = n["retry_after_s"].clone();
                (200, json!({"ok": false, "error": e}))
            }
            None => (503, json!({"error": {"message": m, "type": "maintenance", "code": "maintenance", "maintenance": public}})),
        };
        http::respond_bytes(s, st, "application/json", body.to_string().as_bytes(), &retry);
        true
    }

    /// system.maintenance {on, message, eta_s, unload}: on - the notice written, the video queue paused, the bypass key
    /// answered (the page and token holders only); off - the notice cleared, the queues going on. No `on`: the state
    pub(crate) fn maintenance_set(self: &Arc<Self>, b: &Value) -> Result<Value, Fail> {
        let Some(on) = b["on"].as_bool() else {
            return Ok(json!({"maintenance": self.maintenance().map(|n| public(&n)), "on": self.maintenance().is_some()}));
        };
        let cur = self.maintenance();
        if on {
            let eta = b["eta_s"].as_f64().filter(|x| *x > 0.0);
            let mut n = json!({"on": true, "message": b["message"].as_str().unwrap_or("").trim(),
                               "since": cur.as_ref().and_then(|c| c["since"].as_f64()).unwrap_or_else(now),
                               "until": eta.map_or(Value::Null, |e| json!(now() + e)),
                               "video_restore": cur.as_ref().map_or(Value::Null, |c| c["video_restore"].clone())});
            if cur.is_none() {
                n["video_restore"] = self.video_pause();
            }
            maintenance::write(&self.maint_file, Some(&n)).map_err(|e| (500, e))?;
            let key = {
                let mut k = self.maint_bypass.lock().unwrap();
                if k.is_empty() {
                    *k = super::new_token().replacen("ns-", "mb-", 1);
                }
                k.clone()
            };
            self.save_state();
            let mut unloading = Vec::new();
            if b["unload"] == true {
                unloading = self.unload_idle_now();
            }
            let n = self.maintenance().unwrap_or(n);
            eprintln!("maintenance on: {}", maintenance::message(&n));
            Ok(json!({"on": true, "maintenance": public(&n), "bypass": key, "bypass_header": "X-Maintenance-Bypass", "unloading": unloading,
                      "draining": self.busy(&self.models())["reasons"].clone()}))
        } else {
            if let Some(c) = &cur {
                self.video_restore(&c["video_restore"]);
            }
            maintenance::write(&self.maint_file, None).map_err(|e| (500, e))?;
            self.maint_bypass.lock().unwrap().clear();
            self.save_state();
            eprintln!("maintenance off");
            self.resume_jobs();
            Ok(json!({"on": false, "maintenance": Value::Null}))
        }
    }

    /// The video studio's queue paused; what was paused before (to put back): none when no studio answers
    pub(super) fn video_pause(&self) -> Value {
        if !self.kinds_on(&self.models()).contains(&"video") {
            return Value::Null;
        }
        let t = self.target("video");
        let was_all = call_for(&t, "POST", "/rpc/projects.list", Some(&json!({})), 30).ok().is_some_and(|v| v["result"]["paused_all"] == true);
        match call_for(&t, "POST", "/rpc/queue.pause", Some(&json!({})), 30) {
            Ok(v) if v["ok"] != false => {
                let r = &v["result"];
                json!({"projects": r["paused_projects"], "batches": r["paused_batches"], "was_all": was_all})
            }
            Ok(v) => {
                eprintln!("maintenance: the video queue not paused: {}", v["error"]["message"]);
                Value::Null
            }
            Err(e) => {
                eprintln!("maintenance: the video studio does not answer: {e}");
                Value::Null
            }
        }
    }

    /// The video queue as it was: resumed, then the projects and batches paused before paused again
    pub(super) fn video_restore(&self, r: &Value) {
        if r.is_null() || r["was_all"] == true {
            return;
        }
        let t = self.target("video");
        let _ = call_for(&t, "POST", "/rpc/queue.resume", Some(&json!({})), 30);
        for p in r["projects"].as_array().into_iter().flatten() {
            let _ = call_for(&t, "POST", "/rpc/queue.pause", Some(&json!({"project": p})), 30);
        }
        for b in r["batches"].as_array().into_iter().flatten() {
            let _ = call_for(&t, "POST", "/rpc/queue.pause", Some(&json!({"batch": b})), 30);
        }
    }

    /// The loaded models not answering, unloaded (in the background: a chat model after its answer); their ids
    fn unload_idle_now(self: &Arc<Self>) -> Vec<String> {
        let all = self.models();
        let cards = self.named_cards(&all);
        let ids: Vec<String> = self.model_states(&all, &cards).iter().filter(|m| matches!(m["state"].as_str(), Some("loaded" | "ready")))
            .filter_map(|m| m["id"].as_str().map(str::to_string)).collect();
        for id in ids.clone() {
            let me = self.clone();
            std::thread::spawn(move || {
                if let Err((_, e)) = me.unload(&id) {
                    eprintln!("maintenance: {id} not unloaded: {e}");
                }
            });
        }
        ids
    }
}

/// The notice as callers see it
pub(crate) fn public(n: &Value) -> Value {
    json!({"on": true, "message": n["message"], "since": n["since"], "until": n["until"], "retry_after_s": n["retry_after_s"],
           "notice": maintenance::message(n)})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_it_refuses() {
        assert!(starts_work("llm.chat") && starts_work("image.generate") && starts_work("video.generate") && starts_work("system.load"));
        assert!(!starts_work("system.busy") && !starts_work("image.status") && !starts_work("video.cancel") && !starts_work("system.unload"));
        assert!(openai_starts_work("POST", "/v1/chat/completions") && !openai_starts_work("GET", "/v1/models"));
    }
}
