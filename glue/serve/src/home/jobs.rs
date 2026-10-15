//! Image and audio requests as jobs, the way the video studio keeps clips: `image.generate`, `image.edit`,
//! `audio.generate` and `audio.speak` each become a job with a label, run here in the background (the HTTP call that
//! asked for it can drop: the job goes on), and are read with `<kind>.status {label}` and stopped with
//! `<kind>.cancel {label}`. With `wait` (the default) the call answers when the job ends, as before; `wait: false`
//! answers at once with the job. A `reentrant_id` makes a submit idempotent: one that names the id of a job queued,
//! running or done gets that job back instead of a second one (a failed or cancelled one is made again).
//!
//! The table (jobs.json beside serve.json) survives a restart: jobs that had not ended run again from the start.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::life::s;
use super::{now, Home};
use crate::http::{call_for, Target};

/// The methods that run as jobs
pub const METHODS: [&str; 4] = ["image.generate", "image.edit", "audio.generate", "audio.speak"];

/// Finished jobs kept
const KEEP: usize = 200;

pub struct Jobs {
    list: Mutex<Vec<Value>>,
    ended: Condvar,
    file: PathBuf,
}

type Fail = (u16, String);

fn ended(j: &Value) -> bool {
    matches!(j["state"].as_str(), Some("done" | "failed" | "cancelled"))
}

impl Jobs {
    pub fn load(file: PathBuf) -> Jobs {
        let mut list: Vec<Value> = std::fs::read(&file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        // the ones the last run left unfinished go again
        for j in list.iter_mut().filter(|j| !ended(j)) {
            j["state"] = json!("queued");
            j["restarted"] = json!(true);
            j["started"] = Value::Null;
        }
        Jobs { list: Mutex::new(list), ended: Condvar::new(), file }
    }

    fn save(&self, list: &[Value]) {
        let tmp = self.file.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec(list).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.file);
        }
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Value)) -> Option<Value> {
        let mut l = self.list.lock().unwrap();
        let j = l.iter_mut().find(|j| j["id"] == id)?;
        f(j);
        let out = j.clone();
        // the finished ones beyond KEEP go, oldest first
        let done = l.iter().filter(|j| ended(j)).count();
        if done > KEEP {
            let mut drop = done - KEEP;
            l.retain(|j| {
                if drop > 0 && ended(j) {
                    drop -= 1;
                    false
                } else {
                    true
                }
            });
        }
        self.save(&l);
        self.ended.notify_all();
        Some(out)
    }

    fn get(&self, id: &str) -> Option<Value> {
        self.list.lock().unwrap().iter().find(|j| j["id"] == id).cloned()
    }

    /// The jobs a request names: by label, else by reentrant_id; newest first
    fn named(&self, kind: &str, b: &Value) -> Vec<Value> {
        let (label, rid) = (s(b, "label"), reentrant(b));
        self.list.lock().unwrap().iter().rev()
            .filter(|j| j["kind"] == kind && (if !label.is_empty() { j["label"] == label.as_str() } else { rid.as_deref().is_some_and(|r| j["reentrant_id"] == r) }))
            .cloned().collect()
    }
}

/// A request's reentrant_id (reentrantid is read too)
pub fn reentrant(b: &Value) -> Option<String> {
    ["reentrant_id", "reentrantid"].iter().find_map(|k| b[*k].as_str()).map(str::trim).filter(|r| !r.is_empty()).map(|r| r.chars().take(200).collect())
}

impl Home {
    /// Jobs the last run left unfinished, started again
    pub(crate) fn resume_jobs(self: &Arc<Self>) {
        let ids: Vec<String> = self.jobs.list.lock().unwrap().iter().filter(|j| !ended(j)).filter_map(|j| j["id"].as_str().map(str::to_string)).collect();
        for id in ids {
            eprintln!("job {id}: unfinished at the last stop, run again");
            let me = self.clone();
            std::thread::spawn(move || me.run_job(&id));
        }
    }

    /// A request that runs as a job: its job made (or the one its reentrant_id names), run, and with wait its answer
    pub(crate) fn submit(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let (kind, _) = name.split_once('.').unwrap_or((name, ""));
        let rid = reentrant(b);
        let wait = b["wait"] != false;
        let again = rid.as_deref().and_then(|r| {
            self.jobs.list.lock().unwrap().iter().rev()
                .find(|j| j["kind"] == kind && j["reentrant_id"] == r && !matches!(j["state"].as_str(), Some("failed" | "cancelled")))
                .cloned()
        });
        let reentered = again.is_some();
        let job = match again {
            Some(j) => j,
            None => {
                let n = {
                    let mut n = self.next.lock().unwrap();
                    *n += 1;
                    *n - 1
                };
                let id = format!("{kind}-{}-{n}", now() as u64);
                let label = Some(s(b, "label")).filter(|l| !l.trim().is_empty()).map(|l| l.chars().take(120).collect()).unwrap_or_else(|| id.clone());
                let mut body = b.clone();
                if let Some(o) = body.as_object_mut() {
                    for k in ["wait", "label", "reentrant_id", "reentrantid"] {
                        o.remove(k);
                    }
                }
                let j = json!({"id": id, "kind": kind, "method": name, "label": label, "reentrant_id": rid, "state": "queued",
                               "created": now(), "started": null, "ended": null, "body": body});
                {
                    let mut l = self.jobs.list.lock().unwrap();
                    l.push(j.clone());
                    self.jobs.save(&l);
                }
                let me = self.clone();
                let jid = id.clone();
                std::thread::spawn(move || me.run_job(&jid));
                j
            }
        };
        let id = s(&job, "id");
        if !wait {
            let mut v = self.job_view(&self.jobs.get(&id).unwrap_or(job), None);
            v["reentered"] = json!(reentered);
            return Ok(v);
        }
        // the answer when it ends (an hour at most; the job goes on after that)
        let mut l = self.jobs.list.lock().unwrap();
        let deadline = now() + 3600.0;
        loop {
            match l.iter().find(|j| j["id"] == id.as_str()) {
                Some(j) if ended(j) => {
                    let j = j.clone();
                    drop(l);
                    return match s(&j, "state").as_str() {
                        "done" => {
                            let mut r = j["result"].clone();
                            let mut v = self.job_view(&j, None);
                            if let Some(o) = v.as_object_mut() {
                                o.remove("result");
                            }
                            v["reentered"] = json!(reentered);
                            r["job"] = v;
                            Ok(r)
                        }
                        _ => Err((j["error"]["status"].as_u64().unwrap_or(500) as u16, s(&j["error"], "message"))),
                    };
                }
                Some(_) => {}
                None => return Err((404, format!("job {id} is gone"))),
            }
            if now() > deadline {
                return Err((504, format!("still running after an hour: {kind}.status {{\"label\": {:?}}}", s(&job, "label"))));
            }
            l = self.jobs.ended.wait_timeout(l, Duration::from_secs(5)).unwrap().0;
        }
    }

    /// One job, to its end: the request run as before (the model loaded first if it must), its id passed on so a
    /// cancel stops this one only
    fn run_job(self: &Arc<Self>, id: &str) {
        let Some(j) = self.jobs.update(id, |j| {
            j["state"] = json!("running");
            j["started"] = json!(now());
        }) else { return };
        if j["cancel"] == true {
            self.jobs.update(id, |j| {
                j["state"] = json!("cancelled");
                j["ended"] = json!(now());
                j["error"] = super::api::rpc_error(&s(j, "method"), 499, None, "cancelled before it started");
                j["body"] = Value::Null;
            });
            return;
        }
        let mut body = j["body"].clone();
        body["request_id"] = json!(id);
        let method = s(&j, "method");
        let r = self.work(&method, &body);
        let cancel = self.jobs.get(id).is_some_and(|j| j["cancel"] == true);
        self.jobs.update(id, |j| {
            j["ended"] = json!(now());
            j["body"] = Value::Null; // the pictures and recordings it carried: not kept
            match r {
                Ok(v) => {
                    j["state"] = json!("done");
                    j["result"] = v;
                }
                Err((st, m)) => {
                    let st = if cancel && st != 200 { 499 } else { st };
                    j["state"] = json!(if st == 499 { "cancelled" } else { "failed" });
                    let e = super::api::rpc_error(&method, st, None, &m);
                    j["error"] = e;
                }
            }
        });
    }

    /// The servers' requests running now: request id -> (port, progress)
    fn running_ids(&self, kind: &str) -> Vec<(String, Value)> {
        self.slots(kind).into_iter().filter_map(|p| {
            let v = call_for(&Target::Tcp(format!("127.0.0.1:{p}")), "GET", "/api/progress", None, 2).ok()?;
            Some((v["request_id"].as_str()?.to_string(), v))
        }).collect()
    }

    /// A job as the API shows it: queued until its server is on it, its progress while it runs
    fn job_view(&self, j: &Value, running: Option<&[(String, Value)]>) -> Value {
        let mut v = json!({});
        for k in ["id", "kind", "method", "label", "reentrant_id", "created", "started", "ended", "restarted"] {
            if !j[k].is_null() {
                v[k] = j[k].clone();
            }
        }
        let mut state = s(j, "state");
        if state == "running" {
            let live;
            let r = match running {
                Some(r) => r,
                None => {
                    live = self.running_ids(&s(j, "kind"));
                    &live
                }
            };
            match r.iter().find(|(id, _)| j["id"] == id.as_str()) {
                Some((_, p)) => {
                    let mut p = p.clone();
                    if let Some(o) = p.as_object_mut() {
                        o.remove("request_id");
                        o.remove("prompt");
                    }
                    v["progress"] = p;
                }
                // sent on and waiting for its server, or its model loading
                None => state = "queued".into(),
            }
        }
        v["state"] = json!(state);
        if let (Some(a), Some(b)) = (j["started"].as_f64(), j["ended"].as_f64()) {
            v["seconds"] = json!(((b - a) * 10.0).round() / 10.0);
        }
        if state == "done" {
            v["result"] = j["result"].clone();
        }
        if !j["error"].is_null() {
            v["error"] = j["error"].clone();
        }
        v
    }

    /// `<kind>.status`: one job (by label or reentrant_id; the newest of that label), else the running, the queued
    /// and the last ended
    pub(crate) fn job_status(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        let running = self.running_ids(kind);
        if !s(b, "label").is_empty() || reentrant(b).is_some() {
            let j = self.jobs.named(kind, b).into_iter().next().ok_or_else(|| (404, format!("no {kind} job by that label or reentrant_id")))?;
            return Ok(self.job_view(&j, Some(&running)));
        }
        let all: Vec<Value> = self.jobs.list.lock().unwrap().iter().filter(|j| j["kind"] == kind).cloned().collect();
        let views: Vec<Value> = all.iter().rev().map(|j| self.job_view(j, Some(&running))).collect();
        let with = |st: &str| views.iter().filter(|v| v["state"] == st).cloned().collect::<Vec<_>>();
        let recent: Vec<Value> = views.iter().filter(|v| matches!(v["state"].as_str(), Some("done" | "failed" | "cancelled"))).take(20).cloned().collect();
        Ok(json!({"running": with("running"), "queued": with("queued"), "recent": recent}))
    }

    /// `<kind>.cancel {label}`: its jobs not ended stopped (a running one between steps, a waiting one before it
    /// starts); without a label whatever the kind's servers are making
    pub(crate) fn job_cancel(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        let label = s(b, "label");
        if label.is_empty() && reentrant(b).is_none() {
            let mut stopped = 0;
            for p in self.slots(kind) {
                if call_for(&Target::Tcp(format!("127.0.0.1:{p}")), "POST", "/api/cancel", Some(&json!({})), 10).is_ok_and(|v| v["cancelled"] == true) {
                    stopped += 1;
                }
            }
            return Ok(json!({"cancelled": stopped > 0, "stopped": stopped}));
        }
        let named = self.jobs.named(kind, b);
        let open: Vec<Value> = named.iter().filter(|j| !ended(j)).cloned().collect();
        let which = if label.is_empty() { format!("reentrant_id {:?}", reentrant(b).unwrap_or_default()) } else { format!("label {label:?}") };
        if open.is_empty() {
            // ended already: nothing to stop, which is what a cancel wants (its state says how it ended)
            if let Some(j) = named.first() {
                return Ok(json!({"cancelled": 0, "stopped": 0, "already_ended": true, "state": j["state"], "label": j["label"],
                                 "reentrant_id": j["reentrant_id"], "jobs": [j["id"]]}));
            }
            return Err((404, format!("no {kind} job with {which}")));
        }
        let mut stopped = 0;
        for j in &open {
            let id = s(j, "id");
            self.jobs.update(&id, |j| j["cancel"] = json!(true));
            // every server of the kind hears it: the one running it stops, the others drop it should it come
            for p in self.slots(kind) {
                if call_for(&Target::Tcp(format!("127.0.0.1:{p}")), "POST", "/api/cancel", Some(&json!({"id": id})), 10).is_ok_and(|v| v["cancelled"] == true) {
                    stopped += 1;
                }
            }
        }
        Ok(json!({"cancelled": open.len(), "stopped": stopped, "already_ended": false, "label": open[0]["label"], "reentrant_id": open[0]["reentrant_id"],
                  "jobs": open.iter().map(|j| j["id"].clone()).collect::<Vec<_>>()}))
    }

    /// Whether a kind has jobs not ended (system.busy)
    pub(crate) fn jobs_open(&self) -> Vec<Value> {
        self.jobs.list.lock().unwrap().iter().filter(|j| !ended(j))
            .map(|j| json!({"id": j["id"], "kind": j["kind"], "label": j["label"], "state": j["state"]})).collect()
    }
}
