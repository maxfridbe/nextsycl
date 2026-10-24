//! Image, audio and 3D requests as jobs, the way the video studio keeps clips: `image.generate`, `image.edit`,
//! `audio.generate`, `audio.speak`, `audio.transcribe`, `model3d.generate` and `model3d.rig` each become a job with a label, run here in the background (the HTTP call that
//! asked for it can drop: the job goes on), and are read with `<kind>.status {label}` (one) or `<kind>.jobs` (the
//! list), and stopped with `<kind>.cancel {label}`. With `wait` (the default) the call answers when the job ends;
//! `wait: false` answers at once with the job. A `reentrant_id` makes a submit idempotent: one that names the id of a
//! job queued, running or done gets that job back instead of a second one (a failed or cancelled one is made again).
//!
//! The queue, as video's: a kind runs as many jobs at once as it has servers (two: one a card); the rest wait in
//! order. `priority` (-9..9) places a new job after every waiting job of its priority or higher; `<kind>.queue.move`
//! puts one at a place. A job submitted `held`, or of a project or batch paused (`<kind>.queue.pause`), waits until
//! released. `queue: false` refuses (409 busy, retryable) a job that would not start at once.
//!
//! What a job waits for is said: `stage` (queued, held, waiting, loading, running, then how it ended) and
//! `waiting_for` in words - the jobs ahead, the model loading, the card's room and who holds it.
//!
//! The table (jobs.json beside serve.json, the pauses in jobs.paused.json) survives a restart: jobs that had not ended
//! run again from the start.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::life::s;
use super::{now, Home};
use crate::http::{call_for, Target};

/// The methods that run as jobs
pub const METHODS: [&str; 11] = ["image.generate", "image.edit", "image.upscale", "image.vectorize", "video.upscale", "audio.generate", "audio.speak", "audio.transcribe",
    "audio.voices.enroll", "model3d.generate", "model3d.rig"];

/// The queue a method's jobs wait in: its prefix, but video.upscale runs on the image servers (an upscaler there) and
/// so waits with the image jobs (image.status / image.cancel read it)
pub fn kind_of_method(name: &str) -> &str {
    if name == "video.upscale" {
        return "image";
    }
    name.split_once('.').map(|(k, _)| k).unwrap_or(name)
}

/// The kinds whose requests run as jobs here
pub const KINDS: [&str; 3] = ["image", "audio", "model3d"];

/// Finished jobs kept
const KEEP: usize = 200;

pub struct Jobs {
    list: Mutex<Vec<Value>>,
    ended: Condvar,
    file: PathBuf,
    /// by kind: {all, projects: [], batches: []} - what waits until resumed
    paused: Mutex<Value>,
}

type Fail = (u16, String);

fn ended(j: &Value) -> bool {
    matches!(j["state"].as_str(), Some("done" | "failed" | "cancelled"))
}

/// A job's place in its queue (a smaller one runs first)
fn order(j: &Value) -> f64 {
    j["order"].as_f64().or_else(|| j["created"].as_f64()).unwrap_or(0.0)
}

/// Whether a waiting job is held: its own hold, or its kind's queue, project or batch paused
fn held(j: &Value, paused: &Value) -> bool {
    let named = |k: &str, list: &str| !s(j, k).is_empty() && paused[list].as_array().is_some_and(|a| a.iter().any(|x| x == j[k].as_str().unwrap_or("")));
    j["held"] == true || paused["all"] == true || named("project", "projects") || named("batch", "batches")
}

/// A result's run details under a neutral name: `run` (the servers' own answer calls it `nextsycl`)
pub fn neutral(mut v: Value) -> Value {
    if let Some(o) = v.as_object_mut() {
        if let Some(r) = o.remove("nextsycl") {
            o.insert("run".into(), r);
        }
    }
    v
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
        let paused = std::fs::read(file.with_extension("paused.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(json!({}));
        // every kind's pauses (a file from before a kind was added lacks it)
        let mut paused = paused;
        for k in KINDS {
            if paused[k].is_null() {
                paused[k] = json!({"all": false, "projects": [], "batches": []});
            }
        }
        Jobs { list: Mutex::new(list), ended: Condvar::new(), file, paused: Mutex::new(paused) }
    }

    fn save(&self, list: &[Value]) {
        let tmp = self.file.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec(list).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.file);
        }
    }

    fn save_paused(&self, p: &Value) {
        let _ = std::fs::write(self.file.with_extension("paused.json"), serde_json::to_vec_pretty(p).unwrap_or_default());
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

    /// The kind's waiting jobs in run order (held ones too)
    fn waiting(l: &[Value], kind: &str) -> Vec<Value> {
        let mut w: Vec<Value> = l.iter().filter(|j| j["kind"] == kind && j["state"] == "queued").cloned().collect();
        w.sort_by(|a, b| order(a).total_cmp(&order(b)));
        w
    }
}

/// A request's reentrant_id (reentrantid is read too)
pub fn reentrant(b: &Value) -> Option<String> {
    ["reentrant_id", "reentrantid"].iter().find_map(|k| b[*k].as_str()).map(str::trim).filter(|r| !r.is_empty()).map(|r| r.chars().take(200).collect())
}

/// Where a new job of `priority` goes among the waiting (in run order): after the last of its priority or higher
fn place_for(waiting: &[Value], priority: i64) -> f64 {
    let p = |j: &Value| j["priority"].as_i64().unwrap_or(0);
    match waiting.iter().rposition(|j| p(j) >= priority) {
        None => waiting.first().map_or(now(), |j| order(j) - 1.0),
        Some(i) if i + 1 == waiting.len() => order(&waiting[i]).max(now() - 1.0) + 1.0,
        Some(i) => (order(&waiting[i]) + order(&waiting[i + 1])) / 2.0,
    }
}

impl Home {
    /// Jobs the last run left unfinished, started again
    pub(crate) fn resume_jobs(self: &Arc<Self>) {
        for kind in KINDS {
            self.dispatch(kind);
        }
    }

    /// Start the kind's next waiting jobs while it has fewer running than servers (the first not held, in order)
    fn dispatch(self: &Arc<Self>, kind: &str) {
        // maintenance: the queue waits (system.maintenance off dispatches again)
        if self.maintenance().is_some() {
            return;
        }
        let cap = self.slots(kind).len().max(1);
        let start: Vec<String> = {
            let mut l = self.jobs.list.lock().unwrap();
            let paused = self.jobs.paused.lock().unwrap()[kind].clone();
            let running = l.iter().filter(|j| j["kind"] == kind && j["state"] == "running").count();
            let next: Vec<String> = Jobs::waiting(&l, kind).iter().filter(|j| !held(j, &paused)).take(cap.saturating_sub(running))
                .map(|j| s(j, "id")).collect();
            for j in l.iter_mut().filter(|j| next.iter().any(|n| j["id"] == n.as_str())) {
                j["state"] = json!("running");
                j["started"] = json!(now());
            }
            if !next.is_empty() {
                self.jobs.save(&l);
            }
            next
        };
        for id in start {
            if let Some(j) = self.jobs.get(&id) {
                if j["restarted"] == true {
                    eprintln!("job {id}: unfinished at the last stop, run again");
                }
            }
            let me = self.clone();
            std::thread::spawn(move || me.run_job(&id));
        }
    }

    /// A job's stage while its server is chosen (waiting, loading) and what for; none: forgotten
    pub(crate) fn note(&self, job: Option<&str>, stage: &str, why: &str) {
        if let Some(j) = job {
            self.stages.lock().unwrap().insert(j.to_string(), (stage.to_string(), why.to_string()));
        }
    }

    /// The job is with its server now: its stage is the server's (running, or waiting behind another request)
    pub(crate) fn sent(&self, job: Option<&str>) {
        if let Some(j) = job {
            self.stages.lock().unwrap().remove(j);
        }
    }

    /// Whether a cancel was asked for the job
    pub(crate) fn job_cancelled(&self, id: &str) -> bool {
        self.jobs.get(id).is_some_and(|j| j["cancel"] == true)
    }

    /// A request that runs as a job: its job made (or the one its reentrant_id names), queued, and with wait its answer
    pub(crate) fn submit(self: &Arc<Self>, name: &str, b: &Value) -> Result<Value, Fail> {
        let kind = kind_of_method(name);
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
                let priority = b["priority"].as_i64().unwrap_or(0).clamp(-9, 9);
                let text = |k: &str| Some(s(b, k).trim().chars().take(60).collect::<String>()).filter(|x| !x.is_empty());
                let mut body = b.clone();
                if let Some(o) = body.as_object_mut() {
                    for k in ["wait", "label", "reentrant_id", "reentrantid", "priority", "held", "queue", "project", "batch"] {
                        o.remove(k);
                    }
                }
                let mut j = json!({"id": id, "kind": kind, "method": name, "label": label, "reentrant_id": rid, "state": "queued",
                                   "created": now(), "started": null, "ended": null, "priority": priority, "held": b["held"] == true,
                                   "project": text("project"), "batch": text("batch"), "body": body});
                // an enrolment says whose voice (its body is dropped when it ends)
                if name == "audio.voices.enroll" {
                    j["voice"] = b["name"].clone();
                }
                {
                    let mut l = self.jobs.list.lock().unwrap();
                    let waiting = Jobs::waiting(&l, kind);
                    if b["queue"] == false {
                        // only when it would start now: a server free and nothing runnable waiting before it
                        let paused = self.jobs.paused.lock().unwrap()[kind].clone();
                        let running = l.iter().filter(|x| x["kind"] == kind && x["state"] == "running").count();
                        let ahead = waiting.iter().filter(|x| !held(x, &paused) && x["priority"].as_i64().unwrap_or(0) >= priority).count();
                        if running >= self.slots(kind).len().max(1) || ahead > 0 || held(&j, &paused) {
                            return Err((409, format!("busy: {running} {kind} job(s) running and {ahead} waiting before this one (queue: true to wait in line)")));
                        }
                    }
                    j["order"] = json!(place_for(&waiting, priority));
                    l.push(j.clone());
                    self.jobs.save(&l);
                }
                self.dispatch(kind);
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
                            let mut r = neutral(j["result"].clone());
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

    /// One job (dispatch made it running), to its end: the request run as before (the model loaded first if it must),
    /// its id passed on so a cancel stops this one only; then the next waiting job starts
    fn run_job(self: &Arc<Self>, id: &str) {
        let Some(j) = self.jobs.get(id) else { return };
        let kind = s(&j, "kind");
        let method = s(&j, "method");
        if j["cancel"] == true {
            self.jobs.update(id, |j| {
                j["state"] = json!("cancelled");
                j["ended"] = json!(now());
                j["error"] = super::api::rpc_error(&method, 499, None, "cancelled before it started");
                j["body"] = Value::Null;
            });
            self.dispatch(&kind);
            return;
        }
        let mut body = j["body"].clone();
        body["request_id"] = json!(id);
        let r = self.work(&method, &body);
        let cancel = self.job_cancelled(id);
        self.stages.lock().unwrap().remove(id);
        self.jobs.update(id, |j| {
            j["ended"] = json!(now());
            j["body"] = Value::Null; // the pictures and recordings it carried: not kept
            match r {
                Ok(v) => {
                    j["state"] = json!("done");
                    j["result"] = neutral(v);
                }
                Err((st, m)) => {
                    let st = if cancel && st != 200 { 499 } else { st };
                    j["state"] = json!(if st == 499 { "cancelled" } else { "failed" });
                    j["error"] = super::api::rpc_error(&method, st, None, &m);
                }
            }
        });
        self.dispatch(&kind);
    }

    /// The servers' requests running now: request id -> (port, progress)
    fn running_ids(&self, kind: &str) -> Vec<(String, Value)> {
        self.slots(kind).into_iter().filter_map(|p| {
            let v = call_for(&Target::Tcp(format!("127.0.0.1:{p}")), "GET", "/api/progress", None, 2).ok()?;
            Some((v["request_id"].as_str()?.to_string(), v))
        }).collect()
    }

    /// A job as the API shows it: state (queued until its server is on it), stage and what it waits for, its progress
    /// while it runs, its place while it waits
    fn job_view(&self, j: &Value, running: Option<&[(String, Value)]>) -> Value {
        let mut v = json!({});
        for k in ["id", "kind", "method", "label", "reentrant_id", "created", "started", "ended", "restarted", "priority", "project", "batch", "voice"] {
            if !j[k].is_null() {
                v[k] = j[k].clone();
            }
        }
        let kind = s(j, "kind");
        let mut state = s(j, "state");
        let mut stage = state.clone();
        let mut waiting_for = Value::Null;
        match state.as_str() {
            "queued" => {
                let l = self.jobs.list.lock().unwrap();
                let paused = self.jobs.paused.lock().unwrap()[kind.as_str()].clone();
                let waiting = Jobs::waiting(&l, &kind);
                let me = waiting.iter().position(|x| x["id"] == j["id"]).unwrap_or(0);
                let ahead = waiting[..me].iter().filter(|x| !held(x, &paused)).count();
                let busy = l.iter().filter(|x| x["kind"] == kind.as_str() && x["state"] == "running").count();
                v["position"] = json!(me + 1);
                v["held"] = json!(held(j, &paused));
                if let Some(n) = self.maintenance() {
                    stage = "maintenance".into();
                    waiting_for = json!(format!("maintenance to end ({})", crate::maintenance::message(&n)));
                } else if held(j, &paused) {
                    stage = "held".into();
                    waiting_for = json!(if j["held"] == true { format!("released ({kind}.queue.resume with its label)") } else {
                        format!("its {} resumed ({kind}.queue.resume)", if paused["all"] == true { "queue" } else if s(j, "batch").is_empty() { "project" } else { "project or batch" })
                    });
                } else {
                    waiting_for = json!(format!("{ahead} {kind} job(s) ahead and {busy} running ({} at once)", self.slots(&kind).len().max(1)));
                }
            }
            "running" => {
                let live;
                let r = match running {
                    Some(r) => r,
                    None => {
                        live = self.running_ids(&kind);
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
                        stage = "running".into();
                    }
                    // started here, not yet on its server: the model loading, the room awaited, or another request first
                    None => {
                        state = "queued".into();
                        let noted = self.stages.lock().unwrap().get(&s(j, "id")).cloned();
                        let (st, why) = noted.unwrap_or_else(|| {
                            let other = r.iter().filter_map(|(rid, _)| self.jobs.get(rid)).map(|o| format!("{} job {:?}", s(&o, "kind"), s(&o, "label"))).next();
                            match other {
                                Some(o) => ("waiting".into(), format!("its server finishing {o}")),
                                None => ("starting".into(), "its server to take it".into()),
                            }
                        });
                        // a job run here, not by a server (an enrolment): running, with its progress
                        if st == "running" {
                            state = "running".into();
                            if let Some(p) = self.progress.lock().unwrap().get(&s(j, "id")) {
                                v["progress"] = p.clone();
                            }
                        } else {
                            waiting_for = json!(why);
                        }
                        stage = st;
                    }
                }
            }
            _ => {}
        }
        v["state"] = json!(state);
        v["stage"] = json!(stage);
        v["waiting_for"] = waiting_for;
        if let (Some(a), Some(b)) = (j["started"].as_f64(), j["ended"].as_f64()) {
            v["seconds"] = json!(((b - a) * 10.0).round() / 10.0);
        }
        if state == "done" {
            v["result"] = neutral(j["result"].clone());
        }
        if !j["error"].is_null() {
            v["error"] = j["error"].clone();
        }
        v
    }

    /// `<kind>.status`: one job, by label or reentrant_id (the newest of that label)
    pub(crate) fn job_status(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        if s(b, "label").is_empty() && reentrant(b).is_none() {
            return Err((400, format!("label: or reentrant_id - the job to answer for ({kind}.jobs lists them)")));
        }
        let running = self.running_ids(kind);
        let j = self.jobs.named(kind, b).into_iter().next().ok_or_else(|| (404, format!("no {kind} job by that label or reentrant_id")))?;
        Ok(self.job_view(&j, Some(&running)))
    }

    /// `<kind>.jobs`: the running, the waiting (in run order) and the last ended (limit, 20 by default), of a project
    /// when one is named
    pub(crate) fn job_list(&self, kind: &str, b: &Value) -> Result<Value, Fail> {
        let limit = b["limit"].as_u64().unwrap_or(20).min(KEEP as u64) as usize;
        let project = s(b, "project");
        let running = self.running_ids(kind);
        let mine = |j: &&Value| j["kind"] == kind && (project.is_empty() || j["project"] == project.as_str());
        let (live, waiting, done): (Vec<Value>, Vec<Value>, Vec<Value>) = {
            let l = self.jobs.list.lock().unwrap();
            (l.iter().filter(mine).filter(|j| j["state"] == "running").cloned().collect(),
             Jobs::waiting(&l, kind).iter().filter(mine).cloned().collect(),
             l.iter().rev().filter(mine).filter(|j| ended(j)).take(limit).cloned().collect())
        };
        let view = |l: &[Value]| l.iter().map(|j| self.job_view(j, Some(&running))).collect::<Vec<_>>();
        Ok(json!({"running": view(&live), "queued": view(&waiting), "recent": view(&done), "at_once": self.slots(kind).len().max(1),
                  "paused": self.jobs.paused.lock().unwrap()[kind].clone()}))
    }

    /// `<kind>.queue.list`: the waiting jobs in run order
    pub(crate) fn queue_list(&self, kind: &str) -> Value {
        let waiting = Jobs::waiting(&self.jobs.list.lock().unwrap(), kind);
        json!({"queue": waiting.iter().map(|j| self.job_view(j, Some(&[]))).collect::<Vec<_>>(), "paused": self.jobs.paused.lock().unwrap()[kind].clone()})
    }

    /// `<kind>.queue.move {label | reentrant_id, position, priority}`: a waiting job put at a place (1: next), its
    /// priority changed too when given
    pub(crate) fn queue_move(self: &Arc<Self>, kind: &str, b: &Value) -> Result<Value, Fail> {
        let id = self.jobs.named(kind, b).into_iter().find(|j| j["state"] == "queued").map(|j| s(&j, "id"))
            .ok_or_else(|| (404, format!("no waiting {kind} job by that label or reentrant_id")))?;
        let position = b["position"].as_u64().unwrap_or(1).max(1) as usize;
        let labels = {
            let mut l = self.jobs.list.lock().unwrap();
            let others: Vec<Value> = Jobs::waiting(&l, kind).into_iter().filter(|j| j["id"] != id.as_str()).collect();
            let at = (position - 1).min(others.len());
            let o = match (at.checked_sub(1).map(|i| order(&others[i])), others.get(at).map(order)) {
                (None, None) => now(),
                (None, Some(b)) => b - 1.0,
                (Some(a), None) => a + 1.0,
                (Some(a), Some(b)) => (a + b) / 2.0,
            };
            if let Some(j) = l.iter_mut().find(|j| j["id"] == id.as_str()) {
                j["order"] = json!(o);
                if let Some(p) = b["priority"].as_i64() {
                    j["priority"] = json!(p.clamp(-9, 9));
                }
            }
            self.jobs.save(&l);
            Jobs::waiting(&l, kind).iter().map(|j| j["label"].clone()).collect::<Vec<_>>()
        };
        self.dispatch(kind);
        Ok(json!({"label": self.jobs.get(&id).map(|j| j["label"].clone()), "position": position.min(labels.len()), "queue": labels}))
    }

    /// `<kind>.queue.pause` / `queue.resume`: a job (label or reentrant_id), a project, a batch, or (none) the kind's
    /// whole queue held or released; running jobs go on
    pub(crate) fn queue_hold(self: &Arc<Self>, kind: &str, b: &Value, hold: bool) -> Result<Value, Fail> {
        let what = if !s(b, "label").is_empty() || reentrant(b).is_some() {
            let open: Vec<String> = self.jobs.named(kind, b).iter().filter(|j| j["state"] == "queued").map(|j| s(j, "id")).collect();
            if open.is_empty() {
                return Err((404, format!("no waiting {kind} job by that label or reentrant_id")));
            }
            for id in &open {
                self.jobs.update(id, |j| j["held"] = json!(hold));
            }
            format!("{} job(s)", open.len())
        } else {
            let mut p = self.jobs.paused.lock().unwrap();
            let k = &mut p[kind];
            let (list, name) = if !s(b, "batch").is_empty() { ("batches", s(b, "batch")) } else if !s(b, "project").is_empty() { ("projects", s(b, "project")) } else { ("", String::new()) };
            if list.is_empty() {
                k["all"] = json!(hold);
                // releasing the queue releases its projects and batches too, as video's resume does
                if !hold {
                    k["projects"] = json!([]);
                    k["batches"] = json!([]);
                }
            } else {
                let mut a: Vec<Value> = k[list].as_array().cloned().unwrap_or_default();
                a.retain(|x| x != name.as_str());
                if hold {
                    a.push(json!(name));
                }
                k[list] = json!(a);
            }
            let snapshot = p.clone();
            self.jobs.save_paused(&snapshot);
            if list.is_empty() { "queue".to_string() } else { format!("{} {name}", &list[..list.len() - if list == "batches" { 2 } else { 1 }]) }
        };
        self.dispatch(kind);
        let paused = self.jobs.paused.lock().unwrap()[kind].clone();
        let waiting = Jobs::waiting(&self.jobs.list.lock().unwrap(), kind);
        let n_held = waiting.iter().filter(|j| held(j, &paused)).count();
        let mut v = json!({"queued": waiting.len(), "held": n_held, "paused_state": paused});
        v[if hold { "paused" } else { "resumed" }] = json!(what);
        Ok(v)
    }

    /// `<kind>.cancel {label}`: its jobs not ended stopped (a running one between steps, a waiting one at once); without
    /// a label whatever the kind's servers are making
    pub(crate) fn job_cancel(self: &Arc<Self>, kind: &str, b: &Value) -> Result<Value, Fail> {
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
            let method = s(j, "method");
            // waiting: ended now; running: asked to stop
            let was = self.jobs.update(&id, |j| {
                if j["state"] == "queued" {
                    j["state"] = json!("cancelled");
                    j["ended"] = json!(now());
                    j["error"] = super::api::rpc_error(&method, 499, None, "cancelled before it started");
                    j["body"] = Value::Null;
                } else {
                    j["cancel"] = json!(true);
                }
            });
            if was.is_some_and(|j| j["state"] == "cancelled") {
                continue;
            }
            // every server of the kind hears it: the one running it stops, the others drop it should it come
            for p in self.slots(kind) {
                if call_for(&Target::Tcp(format!("127.0.0.1:{p}")), "POST", "/api/cancel", Some(&json!({"id": id})), 10).is_ok_and(|v| v["cancelled"] == true) {
                    stopped += 1;
                }
            }
        }
        self.dispatch(kind);
        Ok(json!({"cancelled": open.len(), "stopped": stopped, "already_ended": false, "label": open[0]["label"], "reentrant_id": open[0]["reentrant_id"],
                  "jobs": open.iter().map(|j| j["id"].clone()).collect::<Vec<_>>()}))
    }

    /// The jobs not ended (system.busy)
    pub(crate) fn jobs_open(&self) -> Vec<Value> {
        let paused = self.jobs.paused.lock().unwrap().clone();
        self.jobs.list.lock().unwrap().iter().filter(|j| !ended(j))
            .map(|j| json!({"id": j["id"], "kind": j["kind"], "label": j["label"], "state": j["state"],
                            "held": j["state"] == "queued" && held(j, &paused[j["kind"].as_str().unwrap_or("")])})).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, order: f64, priority: i64) -> Value {
        json!({"id": id, "kind": "image", "state": "queued", "order": order, "priority": priority})
    }

    #[test]
    fn a_new_job_goes_after_its_priority() {
        let w = vec![job("a", 10.0, 5), job("b", 20.0, 0), job("c", 30.0, 0)];
        // the same priority as the last: at the end
        assert!(place_for(&w, 0) > 30.0);
        // above everyone but a: between a and b
        let o = place_for(&w, 3);
        assert!(o > 10.0 && o < 20.0, "{o}");
        // above all: first
        assert!(place_for(&w, 9) < 10.0);
        // below all: last
        assert!(place_for(&w, -9) > 30.0);
    }

    #[test]
    fn held_by_its_own_hold_or_a_pause() {
        let p = json!({"all": false, "projects": ["film"], "batches": []});
        assert!(!held(&json!({"project": "other"}), &p));
        assert!(held(&json!({"project": "film"}), &p));
        assert!(held(&json!({"held": true}), &p));
        assert!(held(&json!({}), &json!({"all": true})));
    }

    #[test]
    fn the_run_details_have_a_neutral_name() {
        let r = neutral(json!({"data": [], "nextsycl": {"seed": 4}}));
        assert_eq!(r["run"]["seed"], 4);
        assert!(r.get("nextsycl").is_none());
    }
}
