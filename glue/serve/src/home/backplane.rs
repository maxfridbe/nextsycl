//! The backplane: what a caller may promise and be promised about the cards, the same on every box (nextcuda's
//! docs/control-plane.md is the description both follow).
//!
//! - **Pin** (`system.pin` / `system.unpin`): loaded now in the background, never stopped for room or for idleness,
//!   `pinned` in the registry, loaded again after a restart. A load that fails leaves the pin set and is tried again.
//! - **`rsvp_s`** (optional on a method that runs a model): the model is held that long after the request ends, answered,
//!   failed or cancelled - clamped to NS_MAX_RSVP_S, never shortened by a later, shorter one; 0 releases (it goes as soon
//!   as it is idle, unless an earlier hold still stands); the result says `rsvp_granted_s`. crate::hold has the arithmetic.
//! - **ETA** (`system.eta`, and `eta` / `eta_remaining_s` on jobs): history.rs from what work took, here the work ahead.
//! - **`max_willing_to_wait_s`**: the limit on the time to START. A known wait over it is declined at once
//!   (`wait_too_long`, nothing queued, nothing stopped); an unknown one goes ahead and is withdrawn at the limit; a loaded
//!   model with a free server always starts. Given, it replaces the default waits (90 s a request, 30 min a job).
//! - **On demand**: any `video.*` call brings a stopped daemon up (VIDEO_UP_S at most), and a watcher brings back one that
//!   crashed. Models unload when idle or when room is needed; only idle holders neither pinned nor held go (life.rs idle_on).
//!
//! A request's limit, reservation and timings travel with its thread (`REQ`), so the room rule in life.rs can honour
//! them without every method's arm passing them along.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::history::{shape_of, Record};
use super::life::s;
use super::{now, registry, Home};
use crate::hold;
use crate::http::{call_for, Target};

type Fail = (u16, String);

/// How long a video call waits for a stopped daemon
pub(super) const VIDEO_UP_S: u64 = 120;

/// The methods that run a model: what takes `rsvp_s` and `max_willing_to_wait_s` and what `system.eta` answers for
pub(super) const RUNS_MODEL: &[&str] = &["llm.chat", "image.generate", "image.edit", "image.upscale", "image.vectorize", "image.ocr", "video.upscale", "video.generate", "video.engines.list",
    "audio.generate", "audio.speak", "audio.transcribe", "audio.voices.design", "audio.voices.enroll", "model3d.generate", "model3d.rig"];

/// The fields of the backplane that are the door's, not a server's
pub(super) const OWN_FIELDS: [&str; 2] = ["rsvp_s", "max_willing_to_wait_s"];

pub(super) fn runs_model(name: &str) -> bool {
    RUNS_MODEL.contains(&name)
}

const TAG: &str = "backplane:";

/// An error of the backplane's own as a Fail's message: its code and the fields it carries beside its words
pub(super) fn tag(code: &str, fields: &Value, words: &str) -> String {
    format!("{TAG}{code}\u{1f}{fields}\u{1f}{words}")
}

/// A message's code, fields and words (none: an ordinary message)
pub(super) fn untag(m: &str) -> Option<(String, Value, String)> {
    let mut it = m.strip_prefix(TAG)?.splitn(3, '\u{1f}');
    let (code, fields, words) = (it.next()?, it.next()?, it.next()?);
    Some((code.to_string(), serde_json::from_str(fields).ok()?, words.to_string()))
}

/// wait_too_long, declined before anything started
fn declined(eta: &Value, wait: f64, limit: f64) -> Fail {
    (503, tag("wait_too_long", &json!({"eta": eta, "max_willing_to_wait_s": limit}),
              &format!("it would start in about {wait:.0} s, past the {limit:.0} s you will wait; nothing was started for it")))
}

/// wait_too_long, withdrawn after waiting
pub(super) fn withdrawn(eta: &Value, limit: f64, waited: f64) -> Fail {
    (503, tag("wait_too_long", &json!({"eta": eta, "max_willing_to_wait_s": limit, "waited_s": waited.round()}),
              &format!("it had not started after {waited:.0} s, past the {limit:.0} s you will wait; withdrawn")))
}

/// A request's limit: a number, 0 or more (anything else: none)
pub(super) fn limit_of(b: &Value) -> Option<f64> {
    b["max_willing_to_wait_s"].as_f64().filter(|x| *x >= 0.0)
}

/// What a request on this thread promised and what became of it
#[derive(Clone, Debug, Default)]
struct Req {
    method: String,
    params: Value,
    rsvp_s: Option<f64>,
    limit: Option<f64>,
    /// when its wait began
    t0: f64,
    /// the model it went to, and when it was sent there
    model: Option<String>,
    sent: Option<f64>,
}

thread_local! {
    static REQ: RefCell<Option<Req>> = const { RefCell::new(None) };
}

/// This thread's request, from its body: what it promised, `t0` when it came
pub(super) fn begin(method: &str, b: &Value, t0: f64) {
    let r = Req { method: method.to_string(), params: stripped(b), rsvp_s: b["rsvp_s"].as_f64().filter(|x| *x >= 0.0), limit: limit_of(b), t0, model: None, sent: None };
    REQ.with(|x| *x.borrow_mut() = Some(r));
}

/// The request's model known and on its way (a chat model through the switcher, a lease from `ensure`)
pub(super) fn sent_to(id: &str) {
    REQ.with(|x| {
        if let Some(r) = x.borrow_mut().as_mut() {
            r.model = Some(id.to_string());
            r.sent = Some(now());
        }
    });
}

/// Whether this thread's request gave a limit: it replaces the default waits
pub(super) fn has_limit() -> bool {
    REQ.with(|x| x.borrow().as_ref().is_some_and(|r| r.limit.is_some()))
}

/// `rsvp_s` as it must be (the fields' types are the request struct's check)
pub(super) fn check_fields(b: &Value) -> Result<(), Fail> {
    match b.get("rsvp_s") {
        Some(v) if !v.is_null() && !v.as_f64().is_some_and(|x| x >= 0.0) => Err((400, "rsvp_s: seconds, 0 or more".into())),
        _ => Ok(()),
    }
}

/// The request's body as a server should see it
pub(super) fn stripped(b: &Value) -> Value {
    let mut v = b.clone();
    if let Some(o) = v.as_object_mut() {
        for k in OWN_FIELDS {
            o.remove(k);
        }
    }
    v
}

/// Who a clip belongs to while the door watches it: its reservation to grant when it ends, its limit until it starts
pub(super) struct Watch {
    label: String,
    model: String,
    params: Value,
    rsvp_s: Option<f64>,
    limit: Option<f64>,
    t0: f64,
}

/// A video engine named by a clip's row: a Ref2VA clip's, else the one the daemon defaults to
fn clip_model(row: &Value, all: &[Value]) -> Option<String> {
    let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).collect();
    let refs = row["engine"] == "REF2VA";
    on.iter().find(|m| s(m, "id").contains("ref2va") == refs).or(on.first()).map(|m| s(m, "id"))
}

/// Whether the daemon lacks an enabled model's engine (`running`: its engines; none: it is down)
fn video_missing(enabled: &[String], running: Option<&[String]>) -> bool {
    !enabled.is_empty() && running.is_none_or(|r| enabled.iter().any(|e| !r.contains(e)))
}

fn video_enabled(all: &[Value]) -> Vec<String> {
    all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).map(|m| s(m, "id")).collect()
}

impl Home {
    fn max_rsvp(&self) -> f64 {
        hold::max_rsvp_s(self.o.cfg.get("NS_MAX_RSVP_S"))
    }

    pub(super) fn holds(&self) -> hold::Holds {
        hold::read(&self.holds_file)
    }

    fn change_holds<T>(&self, f: impl FnOnce(&mut hold::Holds) -> T) -> T {
        let _g = self.holds_mx.lock().unwrap();
        let mut h = self.holds();
        let r = f(&mut h);
        if let Err(e) = hold::write(&self.holds_file, &h) {
            eprintln!("holds: {e}");
        }
        r
    }

    /// `rsvp_s` for `id` now: the seconds granted
    pub(super) fn grant_rsvp(&self, id: &str, secs: f64) -> f64 {
        let max = self.max_rsvp();
        self.change_holds(|h| hold::grant_rsvp(h, id, secs, max, now()))
    }

    /// A chat request's keep (X-Keep-Up-S / keep_up_s)
    pub(crate) fn keep_chat(&self, id: &str, secs: u64) {
        let max = self.max_rsvp();
        self.change_holds(|h| hold::grant_keep(h, id, secs, max, now()));
    }

    pub(super) fn pinned(&self, id: &str) -> bool {
        registry::find(&self.o.cfg, id).ok().flatten().is_some_and(|m| m["pinned"] == true)
    }

    /// Seconds rsvp_s holds `id` for yet
    pub(super) fn held_s(&self, id: &str) -> f64 {
        self.holds().get(id).map_or(0.0, |h| h.reserved(now()))
    }

    /// Whether `id` stays when room is needed: pinned or held by rsvp_s. (A chat model's keep stays too, in chat_hold.)
    pub(super) fn kept(&self, id: &str) -> bool {
        self.pinned(id) || self.held_s(id) > 0.0
    }

    /// Why the chat model loaded gives no way to another chat model: pinned or held by rsvp_s. A keep yields, as in nextcuda.
    pub(super) fn swap_blocked(&self, id: &str) -> Option<String> {
        if self.pinned(id) {
            return Some(format!("pinned: {id} (unpin to free the room)"));
        }
        let left = self.held_s(id);
        (left > 0.0).then(|| format!("held by rsvp_s: {id} ({left:.0} s more)"))
    }

    /// What no_room adds about the models on these cards that will not go
    pub(super) fn kept_words(&self, cards: &Value, gpus: &[usize]) -> String {
        let mut on: Vec<String> = gpus.iter().filter_map(|g| cards.get(*g)).flat_map(|c| c["procs"].as_array().into_iter().flatten())
            .filter_map(|p| p["model_id"].as_str().or(p["model"].as_str())).map(str::to_string).collect();
        on.sort();
        on.dedup();
        let pinned: Vec<&String> = on.iter().filter(|m| self.pinned(m)).collect();
        let held: Vec<&String> = on.iter().filter(|m| !self.pinned(m) && self.held_s(m) > 0.0).collect();
        let name = |v: &[&String]| v.iter().map(|x| x.as_str()).collect::<Vec<_>>().join(", ");
        let mut w = String::new();
        if !pinned.is_empty() {
            w += &format!("; pinned: {} (unpin to free the room)", name(&pinned));
        }
        if !held.is_empty() {
            w += &format!("; held by rsvp_s: {} (until its hold ends: {})", name(&held), held.iter().map(|h| format!("{h} {:.0} s", self.held_s(h))).collect::<Vec<_>>().join(", "));
        }
        w
    }

    /// `system.pin`: the flag beside `enabled`, then loaded in the background. A load that fails leaves the pin set: it is a run on the page, and tried again.
    pub(crate) fn pin(self: &Arc<Self>, id: &str) -> Result<(), Fail> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        registry::update(&self.o.cfg, id, |e| e["pinned"] = json!(true)).map_err(|e| (500, e))?;
        self.load_async(id)
    }

    /// `system.unpin`: the flag only. The model stays loaded until the idle rules take it.
    pub(crate) fn unpin(&self, id: &str) -> Result<(), Fail> {
        self.entry(id)?;
        registry::update(&self.o.cfg, id, |e| e["pinned"] = json!(false)).map_err(|e| (500, e))?;
        Ok(())
    }

    /// The daemon up with the enabled models' engines, for a video call: started when it is not, waited for (VIDEO_UP_S)
    pub(super) fn video_up(&self) -> Result<(), Fail> {
        let enabled = video_enabled(&self.models());
        let running = self.video_daemon().map(|d| d.0);
        if !video_missing(&enabled, running.as_deref()) {
            return Ok(());
        }
        if let Err(e) = self.video_apply(false) {
            // a daemon that runs and cannot change now (a clip renders) answers with the engines it has
            return if running.is_some() { Ok(()) } else { Err((502, format!("the video engine is not running and did not start: {e}"))) };
        }
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(VIDEO_UP_S) {
            if !video_missing(&enabled, self.video_daemon().map(|d| d.0).as_deref()) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err((504, format!("the video engine did not come up in {VIDEO_UP_S} s (nextsycl video start; the runs on this page have its log)")))
    }

    /// The model a request would run on, without loading or fetching anything: the one it names, else the loaded or
    /// first enabled one that does the method ("" when none does)
    pub(super) fn request_model(&self, method: &str, b: &Value, all: &[Value]) -> String {
        if let Some(id) = b["model"].as_str().filter(|x| !x.is_empty()) {
            return id.to_string();
        }
        let can: Vec<String> = all.iter().filter(|m| m["enabled"] != false && super::life::capabilities(m).1.contains(&method)).map(|m| s(m, "id")).collect();
        let loaded = |id: &String| match all.iter().find(|m| m["id"] == id.as_str()).map_or("", registry::kind_of) {
            "llm" => self.chat_loaded(all).as_deref() == Some(id.as_str()),
            "video" => self.video_daemon().is_some_and(|d| d.1.iter().any(|(e, _)| e == id)),
            k => self.servers(k).iter().any(|x| x.model.as_deref() == Some(id.as_str())),
        };
        can.iter().find(|id| loaded(id)).or(can.first()).cloned().unwrap_or_default()
    }

    /// The seconds a job has left or will take, from history (None: no history)
    fn job_run_est(&self, j: &Value, all: &[Value]) -> Option<f64> {
        let method = s(j, "method");
        let model = self.request_model(&method, &j["body"], all);
        self.history.eta(&method, &model, &j["body"], true, Some(0.0))["run_s"].as_f64()
    }

    /// The work ahead of a clip in the studio's queue, from what it says of the running clip and history for the queued (None: some of it has none)
    fn clips_ahead_s(&self, model: &str) -> Option<f64> {
        if !super::listening(self.o.ports.video) {
            return Some(0.0);
        }
        let v = call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok()?;
        let st = v.get("result").cloned().unwrap_or(v);
        let mut sum = if st["idle"] == true { 0.0 } else { st["eta"].as_f64()?.max(0.0) };
        for it in st["queue_items"].as_array().into_iter().flatten().filter(|i| i["held"] != true) {
            sum += self.history.eta("video.generate", model, it, true, Some(0.0))["run_s"].as_f64()?;
        }
        Some(sum)
    }

    /// The ETA of a request of `method` with body `b`, now: `exclude` is its own job when it is queued (only what is
    /// ahead counts). Nothing is started or changed.
    pub(crate) fn eta_for(&self, method: &str, b: &Value, exclude: Option<&str>) -> Value {
        let all = self.models();
        let model = self.request_model(method, b, &all);
        let kind = super::jobs::kind_of_method(method);
        let (hot, ahead) = match kind {
            "image" | "audio" | "model3d" => {
                let sv = self.servers(kind);
                let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
                let hot = sv.iter().any(|x| x.model.as_deref() == Some(model.as_str()) && !x.loading && !x.is_busy() && inflight(x.port) == 0);
                (hot, self.jobs_ahead_s(kind, exclude, &|j| self.job_run_est(j, &all)))
            }
            "llm" => {
                let hot = self.chat_loaded(&all).as_deref() == Some(model.as_str());
                (hot, if hot { Some(0.0) } else { None })
            }
            _ => {
                let hot = self.video_daemon().is_some_and(|d| d.0.contains(&model)) && self.video_idle();
                (hot, if hot { Some(0.0) } else { self.clips_ahead_s(&model) })
            }
        };
        self.history.eta(method, &model, b, hot, ahead)
    }

    /// `system.eta {method, params, model, max_willing_to_wait_s}`: read-only
    pub(crate) fn eta_answer(&self, b: &Value) -> Result<Value, Fail> {
        let name = s(b, "method");
        let method = super::methods::find(&name).map(|m| m.full()).filter(|m| runs_model(m))
            .ok_or_else(|| (400, format!("method {name}: a method that runs a model (image.generate, llm.chat ...)")))?;
        let mut params = if b["params"].is_object() { b["params"].clone() } else { json!({}) };
        if let Some(m) = b["model"].as_str().filter(|m| !m.is_empty()) {
            params["model"] = json!(m);
        }
        let eta = self.eta_for(method, &params, None);
        let max = b["max_willing_to_wait_s"].as_f64();
        let would = max.is_none_or(|m| eta["wait_s"].as_f64().is_none_or(|w| w <= m));
        Ok(json!({"method": method, "eta": eta, "max_willing_to_wait_s": max, "would_accept": would}))
    }

    /// Admission of a request with its limit: wait_too_long when its wait is known and longer, nothing queued or stopped
    pub(super) fn admit(&self, method: &str, b: &Value) -> Result<(), Fail> {
        let Some(limit) = limit_of(b) else { return Ok(()) };
        let eta = self.eta_for(method, b, None);
        match eta["wait_s"].as_f64() {
            Some(w) if w > limit => Err(declined(&eta, w, limit)),
            _ => Ok(()),
        }
    }

    /// The request's own limit, passed before it was sent: the failure to answer with
    pub(super) fn past_limit(&self) -> Option<Fail> {
        let (method, params, limit, waited) = REQ.with(|x| {
            let b = x.borrow();
            let r = b.as_ref()?;
            let limit = r.limit?;
            let waited = now() - r.t0;
            (r.sent.is_none() && waited > limit).then(|| (r.method.clone(), r.params.clone(), limit, waited))
        })?;
        Some(withdrawn(&self.eta_for(&method, &params, None), limit, waited))
    }

    /// The request on this thread ended (`outcome`: ok, failed, cancelled): its model held for what it reserved, its
    /// wait and run in the history. The seconds granted, when it reserved.
    pub(super) fn finish_request(&self, method: &str, b: &Value, outcome: &str) -> Option<f64> {
        let r = REQ.with(|x| x.borrow_mut().take())?;
        let id = r.model.or_else(|| b["model"].as_str().filter(|m| !m.is_empty()).map(str::to_string));
        let end = now();
        if let Some(id) = &id {
            let sent = r.sent.unwrap_or(end);
            self.history.record(Record { method: method.into(), model: id.clone(), shape: shape_of(method, id, b), wait_s: (sent - r.t0).max(0.0), run_s: (end - sent).max(0.0), outcome: outcome.into() });
        }
        let secs = r.rsvp_s?;
        Some(match &id {
            Some(id) => self.grant_rsvp(id, secs),
            None => hold::granted(secs, self.max_rsvp()),
        })
    }

    /// A lease or chat target for this thread's request: the model held from now (what it reserved starts over when the
    /// request ends), so nothing can take it between its answer and that
    pub(super) fn request_on(&self, id: &str) {
        sent_to(id);
        let secs = REQ.with(|x| x.borrow().as_ref().and_then(|r| r.rsvp_s)).filter(|s| *s > 0.0);
        if let Some(secs) = secs {
            self.grant_rsvp(id, secs);
        }
    }

    /// video.generate through the door: admitted against its limit; `b`'s label made when it had none (the door names
    /// the clip it watches)
    pub(super) fn video_admit(&self, b: &Value) -> Result<Value, Fail> {
        check_fields(b)?;
        self.admit("video.generate", b)?;
        let mut q = stripped(b);
        if OWN_FIELDS.iter().any(|k| !b[*k].is_null()) && s(&q, "label").is_empty() {
            q["label"] = json!(format!("clip-{}", now() as u64));
        }
        Ok(q)
    }

    /// A clip the studio took: its eta on the answer, the door watching it when it promised anything
    pub(super) fn video_taken(&self, b: &Value, sent: &Value, r: &mut Value) {
        r["eta"] = self.eta_for("video.generate", sent, None);
        let label = Some(s(sent, "label")).filter(|l| !l.is_empty()).or_else(|| r["label"].as_str().map(str::to_string)).unwrap_or_default();
        let (rsvp_s, limit) = (b["rsvp_s"].as_f64().filter(|x| *x >= 0.0), limit_of(b));
        if let Some(secs) = rsvp_s {
            r["rsvp_granted_s"] = json!(hold::granted(secs, self.max_rsvp()));
        }
        if (rsvp_s.is_some() || limit.is_some()) && !label.is_empty() {
            let all = self.models();
            let model = self.request_model("video.generate", b, &all);
            self.video_watch.lock().unwrap().push(Watch { label, model, params: sent.clone(), rsvp_s, limit, t0: now() });
        }
    }

    /// A clip withdrawn at its limit, for video.status with its label
    pub(super) fn video_withdrawn(&self, label: &str) -> Option<Value> {
        self.withdrawn.lock().unwrap().get(label).cloned()
    }

    /// video.status and the queue with what every box's jobs carry: `eta_remaining_s`, and on a queued clip its `eta`
    /// (video.status keeps its number `eta`, as clients read it now; `eta_estimate` is the object)
    pub(super) fn video_annotate(&self, method: &str, v: &mut Value) {
        let all = self.models();
        let run_of = |it: &Value| -> Value {
            let id = clip_model(it, &all).unwrap_or_default();
            self.history.eta("video.generate", &id, it, true, Some(0.0))
        };
        let r = &mut v["result"];
        match method {
            "status" if r["idle"] != true && !r.is_null() => {
                r["eta_remaining_s"] = r["eta"].clone();
                let e = run_of(&r["job"]);
                r["eta_estimate"] = e;
            }
            "queue.list" => {
                for it in r["queue"].as_array_mut().into_iter().flatten() {
                    let e = run_of(it);
                    it["eta_remaining_s"] = e["total_s"].clone();
                    it["eta"] = e;
                }
            }
            _ => {}
        }
    }

    /// The clips the door watches: a limit passed before the clip started takes it out (queued, or waiting for its card); a clip that ended gets its reservation
    fn watch_videos(&self) {
        if self.video_watch.lock().unwrap().is_empty() {
            return;
        }
        let Ok(v) = call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4) else { return };
        let st = v.get("result").cloned().unwrap_or(v);
        let queued = |l: &str| st["queue_items"].as_array().into_iter().flatten().any(|i| i["label"] == l);
        let running = |l: &str| st["idle"] != true && st["job"]["label"] == l;
        let mut list = std::mem::take(&mut *self.video_watch.lock().unwrap());
        list.retain_mut(|w| {
            let (q, r) = (queued(&w.label), running(&w.label));
            if let Some(limit) = w.limit {
                let waiting = q || (r && !st["waiting_for"].is_null());
                let waited = now() - w.t0;
                if waiting && waited > limit {
                    let call = if q { "/rpc/queue.remove" } else { "/rpc/cancel" };
                    let _ = call_for(&self.target("video"), "POST", call, Some(&json!({"label": w.label})), 15);
                    let f = withdrawn(&self.eta_for("video.generate", &w.params, None), limit, waited);
                    self.withdrawn.lock().unwrap().insert(w.label.clone(), super::api::rpc_error("video.generate", f.0, None, &f.1));
                    return false;
                }
                if r && !waiting {
                    w.limit = None;
                }
            }
            if !q && !r {
                if let Some(secs) = w.rsvp_s {
                    self.grant_rsvp(&w.model, secs);
                }
                return false;
            }
            true
        });
        self.video_watch.lock().unwrap().extend(list);
    }

    /// Clips made (the studio's records) into the history, so a clip's wait is known hot or not, from the first call
    /// after a restart too. `clips.seen` beside them: the ones already read.
    fn ingest_clips(&self) {
        if !super::listening(self.o.ports.video) {
            return;
        }
        let Ok(v) = call_for(&self.target("video"), "POST", "/rpc/jobs.list", Some(&json!({"limit": 100})), 15) else { return };
        let all = self.models();
        let file = self.history.dir().join("clips.seen");
        let mut seen: std::collections::BTreeSet<String> = std::fs::read_to_string(&file).unwrap_or_default().lines().map(str::to_string).collect();
        let mut rows: Vec<Value> = v["result"]["jobs"].as_array().cloned().unwrap_or_default();
        rows.reverse();
        let mut changed = false;
        for row in rows {
            let (Some(total), Some(name), Some(id)) = (row["total"].as_f64(), row["file"].as_str(), clip_model(&row, &all)) else { continue };
            if !seen.insert(name.to_string()) {
                continue;
            }
            changed = true;
            let (w, h) = s(&row, "canvas").split_once('x').map_or((0, 0), |(a, b)| (a.parse().unwrap_or(0), b.parse().unwrap_or(0)));
            let shape = shape_of("video.generate", &id, &json!({"width": w, "height": h, "seconds": row["seconds"], "steps": row["steps"]}));
            self.history.record(Record { method: "video.generate".into(), model: id, shape, wait_s: 0.0, run_s: total, outcome: "ok".into() });
        }
        if changed {
            let _ = std::fs::write(&file, seen.into_iter().collect::<Vec<_>>().join("\n"));
        }
    }

    /// Every 10 s: what is kept told to stay (a server's --idle-exit clock is its own, and GET / is a page load, work to
    /// it; a video worker by /engine/keep, 60 s from this tick), and a model whose last rsvp_s was 0 gone once idle
    fn keep_tick(&self) {
        let all = self.models();
        let holds = self.holds();
        let at = now();
        let kept = |id: &str| all.iter().any(|m| m["id"] == id && m["pinned"] == true) || holds.get(id).is_some_and(|h| h.reserved(at) > 0.0);
        let releasing: Vec<String> = holds.iter().filter(|(id, h)| h.release && !kept(id)).map(|(id, _)| id.clone()).collect();
        for m in all.iter().filter(|m| matches!(registry::kind_of(m), "image" | "audio" | "model3d")) {
            let (id, kind) = (s(m, "id"), registry::kind_of(m));
            let Some(x) = self.servers(kind).into_iter().find(|x| x.model.as_deref() == Some(id.as_str())) else { continue };
            if kept(&id) {
                let _ = call_for(&Target::Tcp(format!("127.0.0.1:{}", x.port)), "GET", "/", None, 3);
            } else if releasing.contains(&id) && !x.is_busy() && !x.loading && self.inflight.lock().unwrap().get(&x.port).copied().unwrap_or(0) == 0 {
                self.change_holds(|h| h.remove(&id));
                let _ = self.unload(&id);
            }
        }
        if all.iter().any(|m| registry::kind_of(m) == "video" && m["enabled"] != false && kept(&s(m, "id"))) {
            let _ = call_for(&self.target("video"), "POST", "/engine/keep", Some(&json!({"seconds": 60})), 3);
        }
        for id in releasing {
            if self.chat_idle().as_deref() == Some(id.as_str()) {
                self.change_holds(|h| h.remove(&id));
                let _ = self.unload(&id);
            }
        }
    }

    /// The pinned, enabled models not up, loaded in the background (their daemon is video_apply's: not a video model).
    /// Each tried every 5 minutes at most; a failed load is a run. Not while the box is set aside.
    fn reload_pinned(self: &Arc<Self>, tried: &mut BTreeMap<String, Instant>) {
        if self.maintenance().is_some() || self.cards_taken().is_some() {
            return;
        }
        for m in self.models().iter().filter(|m| m["pinned"] == true && m["enabled"] != false && registry::kind_of(m) != "video") {
            let id = s(m, "id");
            let state = self.model_states(std::slice::from_ref(m), &json!([])).first().map(|x| s(x, "state")).unwrap_or_default();
            if state == "unloaded" && tried.get(&id).is_none_or(|t| t.elapsed() > Duration::from_secs(300)) {
                tried.insert(id.clone(), Instant::now());
                let _ = self.load_async(&id);
            }
        }
    }

    /// A chat server with no switcher to unload it: idle out as the others do
    fn chat_idle_out(&self) {
        if self.o.idle_minutes == 0 || super::listening(self.o.ports.switch) {
            return;
        }
        let last = *self.chat_last.lock().unwrap();
        if now() - last < (self.o.idle_minutes * 60) as f64 {
            return;
        }
        if let Some(id) = self.chat_idle() {
            if let Err((_, e)) = self.unload(&id) {
                eprintln!("{id}: idle, not unloaded: {e}");
            }
        }
    }

    /// Running forever beside the card sampler: limits and watches every 2 s, keeps every 10, the daemon and the pins every 30
    pub(super) fn backplane_loop(self: Arc<Self>) {
        let mut tick = 0u64;
        let mut gone = 0;
        let mut tried = BTreeMap::new();
        loop {
            std::thread::sleep(Duration::from_secs(2));
            tick += 1;
            self.withdraw_queued();
            self.watch_videos();
            if tick.is_multiple_of(5) {
                self.keep_tick();
                self.chat_idle_out();
            }
            if tick % 15 == 7 {
                self.ingest_clips();
            }
            // first soon after a start (the sampler has seen the cards, the daemon has had its first apply)
            if tick % 15 == 2 {
                self.reload_pinned(&mut tried);
                // the daemon back when it is gone, twice in a row (a deploy stops and starts it within seconds)
                let enabled = video_enabled(&self.models());
                gone = if video_missing(&enabled, self.video_daemon().map(|d| d.0).as_deref()) { gone + 1 } else { 0 };
                if gone >= 2 {
                    if let Err(e) = self.video_apply(false) {
                        eprintln!("video: {e}");
                    }
                    gone = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_of_ours_keeps_its_fields_through_a_message() {
        let f = declined(&json!({"wait_s": 50.0}), 50.0, 10.0);
        assert_eq!(f.0, 503);
        let (code, fields, words) = untag(&f.1).unwrap();
        assert_eq!(code, "wait_too_long");
        assert_eq!(fields["max_willing_to_wait_s"], 10.0);
        assert_eq!(fields["eta"]["wait_s"], 50.0);
        assert!(fields.get("waited_s").is_none() && words.contains("nothing was started"));
        let (_, fields, _) = untag(&withdrawn(&json!({}), 10.0, 12.4).1).unwrap();
        assert_eq!(fields["waited_s"], 12.0, "withdrawn: how long it waited");
        assert!(untag("an ordinary message").is_none());
    }

    #[test]
    fn a_limit_replaces_the_default_waits_and_a_negative_is_none() {
        begin("image.generate", &json!({"max_willing_to_wait_s": 5}), now());
        assert!(has_limit());
        begin("image.generate", &json!({"max_willing_to_wait_s": -5}), now());
        assert!(!has_limit());
        REQ.with(|x| *x.borrow_mut() = None);
    }

    #[test]
    fn the_fields_are_the_doors_own() {
        assert!(check_fields(&json!({"rsvp_s": 0, "max_willing_to_wait_s": 3.5})).is_ok());
        assert_eq!(check_fields(&json!({"rsvp_s": -1})).unwrap_err().1, "rsvp_s: seconds, 0 or more");
        assert_eq!(stripped(&json!({"rsvp_s": 5, "max_willing_to_wait_s": 5, "prompt": "x"})), json!({"prompt": "x"}));
    }

    #[test]
    fn a_daemon_lacking_an_engine_counts_as_missing() {
        let e = vec!["a".to_string(), "b".to_string()];
        assert!(video_missing(&e, None));
        assert!(video_missing(&e, Some(&["a".to_string()])));
        assert!(!video_missing(&e, Some(&["b".to_string(), "a".to_string()])));
        assert!(!video_missing(&[], None), "no video model enabled: nothing is missing");
    }

    #[test]
    fn a_clip_row_names_its_engine() {
        let all = vec![json!({"id": "minimax-h3", "kind": "video"}), json!({"id": "minimax-h3-ref2va", "kind": "video"})];
        assert_eq!(clip_model(&json!({"engine": "INT8"}), &all).as_deref(), Some("minimax-h3"));
        assert_eq!(clip_model(&json!({"engine": "REF2VA"}), &all).as_deref(), Some("minimax-h3-ref2va"));
        assert_eq!(clip_model(&json!({}), &[]), None);
    }
}
