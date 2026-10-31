//! The backplane: what a caller may promise and be promised about the cards, the same on every box.
//!
//! - **Pin** (`system.pin` / `system.unpin`): loaded now, never stopped for room, `pinned` in the registry, loaded
//!   again after a restart. A pinned model's server is kept from its own idle exit.
//! - **`rsvp_s`** (optional on a method that runs a model): the model is held that long after the request ends - clamped
//!   to NS_MAX_RSVP_S, never shortened by a smaller value, 0 releases; the result says `rsvp_granted_s`. X-Keep-Up-S and
//!   keep_up_s are the same hold (crate::hold).
//! - **ETA** (`system.eta`, and `eta` / `eta_remaining_s` on jobs): from each model's history (eta.rs), the work ahead in
//!   the queues, and its load time when it is not up.
//! - **`max_willing_to_wait_s`**: the limit on the time to START. A known wait over it is declined at once
//!   (`wait_too_long`, nothing queued, nothing stopped); an unknown one goes ahead and is withdrawn at the limit; a hot
//!   model (loaded) is never declined.
//! - **On demand**: any `video.*` call brings a stopped daemon up (VIDEO_UP_S at most). Models unload when idle or when
//!   room is needed; only idle holders neither pinned nor held go (life.rs idle_on).
//!
//! A request's limit, reservation and timings travel with its thread (`REQ`), so the room rule in life.rs can honour
//! them without every method's arm passing them along.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::life::s;
use super::{eta, now, registry, Home};
use crate::hold;
use crate::http::{call_for, call_status, Target};

type Fail = (u16, String);

/// How long a video call waits for a stopped daemon
pub(super) const VIDEO_UP_S: u64 = 120;
/// How long a pinned clip engine may take to load
const VIDEO_LOAD_S: u64 = 900;

/// The methods that run a model: what takes `rsvp_s` and `max_willing_to_wait_s` and what `system.eta` answers for
pub(super) const RUNS_MODEL: &[&str] = &["llm.chat", "image.generate", "image.edit", "image.upscale", "image.vectorize", "image.ocr", "video.upscale", "video.generate",
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

/// wait_too_long: the wait known or waited, the limit, nothing started
pub(super) fn wait_too_long(eta: &Value, limit: f64, waited: Option<f64>) -> Fail {
    let mut f = json!({"eta": eta, "max_willing_to_wait_s": limit});
    let words = match waited {
        Some(w) => {
            f["waited_s"] = json!((w * 10.0).round() / 10.0);
            format!("withdrawn: not started after {w:.0} s, the limit was {limit} s")
        }
        None => format!("the wait is ~{:.0} s, the limit is {limit} s; nothing was queued", eta["wait_s"].as_f64().unwrap_or(0.0)),
    };
    (503, tag("wait_too_long", &f, &words))
}

/// What a request on this thread promised and what became of it
#[derive(Clone, Debug, Default)]
struct Req {
    rsvp_s: Option<f64>,
    limit: Option<f64>,
    /// when its wait began
    t0: f64,
    eta: Value,
    /// the model it went to, and when it was sent there
    model: Option<String>,
    sent: Option<f64>,
}

thread_local! {
    static REQ: RefCell<Option<Req>> = const { RefCell::new(None) };
}

/// This thread's request, from its body: what it promised, with `eta` as admission saw it and `t0` when it came
pub(super) fn begin(b: &Value, eta: Value, t0: f64) {
    let r = Req { rsvp_s: b["rsvp_s"].as_f64(), limit: b["max_willing_to_wait_s"].as_f64(), t0, eta, model: None, sent: None };
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

/// The request's own limit, passed: the failure to answer with (it has not started)
pub(super) fn past_limit() -> Option<Fail> {
    REQ.with(|x| {
        let b = x.borrow();
        let r = b.as_ref()?;
        let limit = r.limit?;
        let waited = now() - r.t0;
        (r.sent.is_none() && waited > limit).then(|| wait_too_long(&r.eta, limit, Some(waited)))
    })
}

/// What a request's `rsvp_s` and `max_willing_to_wait_s` must be
pub(super) fn check_fields(b: &Value) -> Result<(), Fail> {
    for k in OWN_FIELDS {
        match &b[k] {
            Value::Null => {}
            v if v.as_f64().is_some_and(|x| x >= 0.0) => {}
            v => return Err((400, format!("{k} {v}: seconds, 0 or more"))),
        }
    }
    Ok(())
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
    rsvp_s: Option<f64>,
    limit: Option<f64>,
    t0: f64,
    eta: Value,
}

/// A job the limit passed while it stood in its queue
pub(super) fn past_in_queue(j: &Value, at: f64) -> Option<f64> {
    let limit = j["max_willing_to_wait_s"].as_f64()?;
    let waited = at - j["created"].as_f64()?;
    (waited > limit).then_some(waited)
}

/// A video engine named by a clip's row: a Ref2VA clip's, else the one the daemon defaults to
fn clip_model(row: &Value, all: &[Value]) -> Option<String> {
    let on: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).collect();
    let refs = row["engine"] == "REF2VA";
    on.iter().find(|m| s(m, "id").contains("ref2va") == refs).or(on.first()).map(|m| s(m, "id"))
}

impl Home {
    fn max_rsvp(&self) -> u64 {
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

    /// `rsvp_s` for `id` now: the seconds held
    pub(super) fn grant_rsvp(&self, id: &str, secs: f64) -> u64 {
        let max = self.max_rsvp();
        self.change_holds(|h| hold::grant_rsvp(h, id, secs, max, now()))
    }

    /// A chat request's keep (X-Keep-Up-S / keep_up_s): the same hold
    pub(crate) fn keep_chat(&self, id: &str, secs: u64) {
        let max = self.max_rsvp();
        self.change_holds(|h| hold::grant_keep(h, id, secs, max, now()));
    }

    /// Hold `id` no more (it went away on purpose)
    pub(super) fn release_hold(&self, id: &str) {
        self.change_holds(|h| h.remove(id));
    }

    pub(super) fn pinned(&self, id: &str) -> bool {
        registry::find(&self.o.cfg, id).ok().flatten().is_some_and(|m| m["pinned"] == true)
    }

    /// Seconds the switcher keeps a chat model for (its X-Keep-Up-S)
    fn switcher_keep(&self, id: &str) -> f64 {
        if self.o.ports.switch == 0 {
            return 0.0;
        }
        call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.switch)), "GET", "/keep", None, 3).ok()
            .and_then(|k| k["kept"][id].as_f64()).unwrap_or(0.0)
    }

    /// Why `id` may not be stopped for room, in the words no_room uses: pinned, or held; none: it may
    pub(super) fn protected_why(&self, id: &str) -> Option<String> {
        if self.pinned(id) {
            return Some(format!("pinned: {id} (unpin to free the room)"));
        }
        let left = self.holds().get(id).map_or(0.0, |h| h.left(now())).max(self.switcher_keep(id));
        (left > 0.0).then(|| format!("held by rsvp_s: {id} ({} s more)", left.ceil() as u64))
    }

    /// Why the chat model loaded may not give way to another chat model: pinned or reserved. A keep yields, as it always has.
    pub(super) fn swap_blocked(&self, id: &str) -> Option<String> {
        if self.pinned(id) {
            return Some(format!("pinned: {id} (unpin to free the room)"));
        }
        let left = self.holds().get(id).map_or(0.0, |h| h.reserved(now()));
        (left > 0.0).then(|| format!("held by rsvp_s: {id} ({} s more)", left.ceil() as u64))
    }

    /// `system.pin {model, wait}`: pinned and loaded. With `wait` (the default) it answers when loaded; a model that
    /// cannot be is not left pinned, and the failure (no_room ...) is the answer. `wait: false` answers at once; the
    /// pin stands and the load is tried until it holds.
    pub(crate) fn pin(self: &Arc<Self>, id: &str, wait: bool) -> Result<Value, Fail> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        let was = m["pinned"] == true;
        registry::update(&self.o.cfg, id, |e| e["pinned"] = json!(true)).map_err(|e| (500, e))?;
        if !wait {
            let me = self.clone();
            let i = id.to_string();
            std::thread::spawn(move || me.pin_load_logged(&i));
            return Ok(json!({"model": id, "pinned": true, "loading": true}));
        }
        match self.pin_load(id) {
            Ok(()) => Ok(json!({"model": id, "pinned": true, "loaded": true})),
            Err(e) => {
                if !was {
                    let _ = registry::update(&self.o.cfg, id, |e| {
                        if let Some(o) = e.as_object_mut() {
                            o.remove("pinned");
                        }
                    });
                }
                Err(e)
            }
        }
    }

    /// `system.unpin {model}`: it may be stopped for room again and unloads when idle
    pub(crate) fn unpin(&self, id: &str) -> Result<Value, Fail> {
        self.entry(id)?;
        registry::update(&self.o.cfg, id, |e| {
            if let Some(o) = e.as_object_mut() {
                o.remove("pinned");
            }
        }).map_err(|e| (500, e))?;
        Ok(json!({"model": id, "pinned": false}))
    }

    /// A pinned model on its cards
    fn pin_load(&self, id: &str) -> Result<(), Fail> {
        if let Some(why) = self.cards_taken() {
            if !self.is_hot(id) {
                return Err((503, why));
            }
        }
        let m = self.entry(id)?;
        match registry::kind_of(&m) {
            "video" => {
                self.video_up()?;
                self.video_load(id)
            }
            _ => self.load(id),
        }
    }

    fn pin_load_logged(&self, id: &str) {
        if self.loading.lock().unwrap().insert(id.to_string(), now()).is_some() {
            return;
        }
        let r = self.pin_load(id);
        self.loading.lock().unwrap().remove(id);
        if let Err((_, e)) = r {
            let (rid, log) = self.new_run(format!("pinned: load {id}"));
            let _ = std::fs::write(log, &e);
            self.finish(rid, 1);
        }
    }

    /// A clip engine loaded with no clip (a job the daemon ends once its worker is up), for a pin
    fn video_load(&self, id: &str) -> Result<(), Fail> {
        let up = |h: &Home| h.video_daemon().is_some_and(|(_, loaded)| loaded.iter().any(|(e, _)| e == id));
        if up(self) {
            return Ok(());
        }
        call_for(&self.target("video"), "POST", "/engine/jobs", Some(&json!({"kind": "load", "engine": id})), 30)
            .map_err(|e| (502, format!("the video daemon: {e}")))?;
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(VIDEO_LOAD_S) {
            if up(self) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        Err((503, format!("{id} did not load in {} minutes (its card may be held)", VIDEO_LOAD_S / 60)))
    }

    /// The daemon up, for a video call: started when it is stopped, waited for (VIDEO_UP_S)
    pub(super) fn video_up(&self) -> Result<(), Fail> {
        if super::listening(self.o.ports.video) {
            return Ok(());
        }
        self.video_apply(false).map_err(|e| (502, format!("the video daemon did not start: {e}")))?;
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(VIDEO_UP_S) {
            if super::listening(self.o.ports.video) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err((502, format!("the video daemon did not come up in {VIDEO_UP_S} s")))
    }

    /// Whether the model is on its cards now
    pub(super) fn is_hot(&self, id: &str) -> bool {
        let all = self.models();
        let Some(m) = all.iter().find(|m| m["id"] == id) else { return false };
        match registry::kind_of(m) {
            "llm" => self.chat_loaded(&all).as_deref() == Some(id),
            "video" => self.video_daemon().is_some_and(|(_, l)| l.iter().any(|(e, _)| e == id)),
            k => self.servers(k).iter().any(|x| x.model.as_deref() == Some(id) && !x.loading),
        }
    }

    /// The model a request would run on, without loading or fetching anything: the one it names, else the loaded or first enabled one that does the method
    fn request_model(&self, method: &str, b: &Value, all: &[Value]) -> Option<String> {
        if let Some(id) = b["model"].as_str().filter(|x| !x.is_empty()) {
            return all.iter().find(|m| m["id"] == id).map(|m| s(m, "id"));
        }
        let can: Vec<String> = all.iter().filter(|m| m["enabled"] != false && super::life::capabilities(m).1.contains(&method)).map(|m| s(m, "id")).collect();
        can.iter().find(|id| self.is_hot(id)).or(can.first()).cloned()
    }

    /// Whether a load of `m` has room now (with the idle, unprotected holders gone)
    fn room_now(&self, m: &Value) -> bool {
        if self.cards_taken().is_some() {
            return false;
        }
        let all = self.models();
        let cards = self.named_cards(&all);
        let kind = registry::kind_of(m);
        let gpus = self.gpus_of(m);
        let card_mb = cards.get(gpus[0]).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
        let (need, _) = self.vouched_mb(m, card_mb);
        let want = |g: usize| need.get(&g).copied().unwrap_or(0) + super::life::MARGIN_MB;
        if kind == "llm" {
            if self.chat_loaded(&all).is_some_and(|c| self.swap_blocked(&c).is_some()) {
                return false;
            }
            let chat_on = |g: usize| cards.get(g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                .filter(|p| p["kind"] == "llm").map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
            return gpus.iter().all(|g| self.room_plan(&cards, *g, want(*g).saturating_sub(chat_on(*g)), "llm", false).is_some());
        }
        gpus.iter().any(|g| self.room_plan(&cards, *g, want(*g), kind, false).is_some())
    }

    /// Seconds of work ahead of a new request on a chat model or clip engine (None: unknown)
    fn ahead_s(&self, kind: &str, id: &str, rows: &[Value]) -> Option<f64> {
        match kind {
            "llm" => {
                let active = call_for(&Target::Unix(self.chat_socket()), "GET", "/server/requests", None, 3).ok()?["active"].as_array()?.len();
                if active == 0 {
                    return Some(0.0);
                }
                eta::estimate(rows, "").run_s.map(|r| r * active as f64)
            }
            "video" => {
                if !super::listening(self.o.ports.video) {
                    return Some(0.0);
                }
                let v = call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok()?;
                let st = v.get("result").cloned().unwrap_or(v);
                let running = if st["idle"] == true { 0.0 } else { st["eta"].as_f64()?.max(0.0) };
                let mut sum = running;
                for it in st["queue_items"].as_array().into_iter().flatten().filter(|i| i["held"] != true) {
                    sum += eta::estimate(rows, &eta::shape_of("video.generate", it)).run_s?;
                }
                let _ = id;
                Some(sum)
            }
            k => self.jobs_ahead_s(k),
        }
    }

    /// What a request would wait for and how long it would run: (eta, hot, the model). A clip's history includes its
    /// engine's load, so a cold clip engine's wait is what is ahead of it, known from its history whether or not it is hot.
    pub(super) fn eta_for(&self, method: &str, b: &Value) -> (Value, bool, Option<String>) {
        let all = self.models();
        let shape = eta::shape_of(method, b);
        let Some(id) = self.request_model(method, b, &all) else {
            return (eta::eta_json(&eta::estimate(&[], &shape), None, &shape), false, None);
        };
        let rows = eta::rows(&self.history, &id);
        let est = eta::estimate(&rows, &shape);
        let hot = self.is_hot(&id);
        let m = all.iter().find(|m| m["id"] == id.as_str());
        let kind = m.map_or("llm", registry::kind_of);
        let mut wait = self.ahead_s(kind, &id, &rows);
        if !hot {
            // no room even with the idle holders gone: how long that lasts is not known here
            if m.is_none_or(|m| !self.room_now(m)) {
                wait = None;
            } else if kind != "video" {
                wait = wait.zip(eta::load_s(&rows)).map(|(w, l)| w + l);
            }
        }
        (eta::eta_json(&est, wait, &shape), hot, Some(id))
    }

    /// `system.eta {method, params, max_willing_to_wait_s}`: read-only
    pub(crate) fn eta_answer(&self, b: &Value) -> Result<Value, Fail> {
        let name = s(b, "method");
        let method = super::methods::find(&name).map(|m| m.full()).filter(|m| runs_model(m))
            .ok_or_else(|| (400, format!("method {name}: one that runs a model ({})", RUNS_MODEL.join(", "))))?;
        check_fields(b)?;
        let params = if b["params"].is_object() { b["params"].clone() } else { json!({}) };
        let limit = b["max_willing_to_wait_s"].as_f64().or(params["max_willing_to_wait_s"].as_f64());
        let (e, hot, model) = self.eta_for(method, &params);
        Ok(json!({"method": method, "model": model, "hot": hot, "eta": e, "would_accept": eta::would_accept(hot, e["wait_s"].as_f64(), limit),
                  "max_willing_to_wait_s": limit}))
    }

    /// Admission of a request with its limit: its eta, or wait_too_long with nothing queued or stopped
    pub(super) fn admit(&self, method: &str, b: &Value) -> Result<Value, Fail> {
        let (e, hot, _) = self.eta_for(method, b);
        let limit = b["max_willing_to_wait_s"].as_f64();
        if !eta::would_accept(hot, e["wait_s"].as_f64(), limit) {
            return Err(wait_too_long(&e, limit.unwrap_or(0.0), None));
        }
        Ok(e)
    }

    /// The request on this thread ended: its model held for what it reserved, its run added to the history. The seconds granted, when it reserved.
    pub(super) fn finish_request(&self, method: &str, b: &Value, ok: bool) -> Option<u64> {
        let r = REQ.with(|x| x.borrow_mut().take())?;
        let id = r.model?;
        let end = now();
        if ok {
            let sent = r.sent.unwrap_or(end);
            eta::append(&self.history, &id, json!({"at": end, "method": method, "shape": eta::shape_of(method, b),
                                                   "wait_s": (sent - r.t0).max(0.0), "run_s": (end - sent).max(0.0)}));
        }
        r.rsvp_s.map(|secs| self.grant_rsvp(&id, secs))
    }

    /// A lease or chat target for this thread's request: the model held from now (what it reserved starts over when the request ends), so nothing can take it in between
    pub(super) fn request_on(&self, id: &str) {
        sent_to(id);
        let secs = REQ.with(|x| x.borrow().as_ref().and_then(|r| r.rsvp_s)).filter(|s| *s > 0.0);
        if let Some(secs) = secs {
            self.grant_rsvp(id, secs);
        }
    }

    /// A load's time, for the model's history
    pub(super) fn loaded_in(&self, id: &str, secs: f64) {
        eta::append(&self.history, id, json!({"at": now(), "shape": "load", "load_s": secs}));
    }

    /// video.generate through the door: admitted against its limit, watched for its limit and reservation; `b`'s label
    /// made when it had none (the door names the clip it watches)
    pub(super) fn video_admit(&self, b: &Value) -> Result<(Value, Value), Fail> {
        check_fields(b)?;
        let e = self.admit("video.generate", b)?;
        let mut q = stripped(b);
        let wants = OWN_FIELDS.iter().any(|k| !b[*k].is_null());
        if wants && s(&q, "label").is_empty() {
            q["label"] = json!(format!("clip-{}", now() as u64));
        }
        Ok((q, e))
    }

    /// A clip the studio took: its eta on the answer, the door watching it when it promised anything
    pub(super) fn video_taken(&self, b: &Value, sent: &Value, e: &Value, r: &mut Value) {
        r["eta"] = e.clone();
        let label = Some(s(sent, "label")).filter(|l| !l.is_empty()).or_else(|| r["label"].as_str().map(str::to_string)).unwrap_or_default();
        let (rsvp_s, limit) = (b["rsvp_s"].as_f64(), b["max_willing_to_wait_s"].as_f64());
        if let Some(secs) = rsvp_s {
            let max = self.max_rsvp();
            r["rsvp_granted_s"] = json!(secs.min(max as f64).round() as u64);
        }
        if (rsvp_s.is_some() || limit.is_some()) && !label.is_empty() {
            let all = self.models();
            let model = self.request_model("video.generate", b, &all).unwrap_or_default();
            self.video_watch.lock().unwrap().push(Watch { label, model, rsvp_s, limit, t0: now(), eta: e.clone() });
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
            let shape = eta::shape_of("video.generate", it);
            let id = clip_model(it, &all).unwrap_or_default();
            eta::eta_json(&eta::estimate(&eta::rows(&self.history, &id), &shape), None, &shape)
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
                    it["eta_remaining_s"] = e["run_s"].clone();
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
                    let f = wait_too_long(&w.eta, limit, Some(waited));
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

    /// Clips made (the studio's records) into the history, so a clip's wait is known hot or not, from the first call after a restart too
    fn ingest_clips(&self) {
        if !super::listening(self.o.ports.video) {
            return;
        }
        let Ok(v) = call_for(&self.target("video"), "POST", "/rpc/jobs.list", Some(&json!({"limit": 100})), 15) else { return };
        let all = self.models();
        let mut rows: Vec<Value> = v["result"]["jobs"].as_array().cloned().unwrap_or_default();
        rows.reverse();
        for row in rows {
            let (Some(total), Some(file), Some(id)) = (row["total"].as_f64(), row["file"].as_str(), clip_model(&row, &all)) else { continue };
            if eta::rows(&self.history, &id).iter().any(|r| r["file"] == file) {
                continue;
            }
            let (w, h) = s(&row, "canvas").split_once('x').map_or((0, 0), |(a, b)| (a.parse().unwrap_or(0), b.parse().unwrap_or(0)));
            let shape = eta::shape_of("video.generate", &json!({"width": w, "height": h, "seconds": row["seconds"], "steps": row["steps"]}));
            eta::append(&self.history, &id, json!({"at": row["mtime"], "method": "video.generate", "shape": shape, "file": file, "run_s": total}));
        }
    }

    /// A chat model coming up, whoever asked (Open WebUI through the switcher too): how long it took
    fn watch_chat_load(&self) {
        if !super::listening(self.o.ports.video) {
            return;
        }
        let starting = self.chat_mode().1;
        let mut seen = self.chat_starting.lock().unwrap();
        match (&*seen, starting) {
            (None, Some(id)) => *seen = Some((id, now())),
            (Some((id, t)), None) => {
                let (id, t) = (id.clone(), *t);
                *seen = None;
                drop(seen);
                if self.chat_loaded(&self.models()).as_deref() == Some(id.as_str()) {
                    self.loaded_in(&id, now() - t);
                }
            }
            _ => {}
        }
    }

    /// Pinned and held models told to stay: a server's own idle exit counts any POST as work, a clip worker's idle
    /// unload is held off by /engine/keep. A door that dies keeps nothing up for long.
    fn keep_up(&self) {
        let holds = self.holds();
        let all = self.models();
        let at = now();
        let kept = |m: &Value| m["pinned"] == true || holds.get(m["id"].as_str().unwrap_or("")).is_some_and(|h| h.left(at) > 0.0);
        for kind in ["image", "audio", "model3d"] {
            if !all.iter().any(|m| registry::kind_of(m) == kind && kept(m)) {
                continue;
            }
            for x in self.servers(kind) {
                let id = x.model.clone().unwrap_or_default();
                if all.iter().any(|m| m["id"] == id.as_str() && kept(m)) {
                    let _ = call_status(&Target::Tcp(format!("127.0.0.1:{}", x.port)), "POST", "/api/keepalive", Some(&json!({})), 3);
                }
            }
        }
        if all.iter().any(|m| registry::kind_of(m) == "video" && kept(m)) {
            for (id, g) in self.video_daemon().map_or(Vec::new(), |d| d.1) {
                if all.iter().any(|m| m["id"] == id.as_str() && kept(m)) {
                    let _ = call_for(&self.target("video"), "POST", "/engine/keep", Some(&json!({"gpu": g, "seconds": 120})), 5);
                }
            }
        }
    }

    /// The pinned models not up, loaded again (after a restart, or a server that died); not while the box is set aside
    fn reload_pinned(self: &Arc<Self>) {
        if self.maintenance().is_some() || self.cards_taken().is_some() {
            return;
        }
        for m in self.models().iter().filter(|m| m["pinned"] == true && m["enabled"] != false) {
            let id = s(m, "id");
            if !self.is_hot(&id) && !self.loading.lock().unwrap().contains_key(&id) && self.moving_why(&id).is_none() {
                let me = self.clone();
                std::thread::spawn(move || me.pin_load_logged(&id));
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
            if let Err((_, e)) = self.unload_now(&id) {
                eprintln!("{id}: idle, not unloaded: {e}");
            }
        }
    }

    /// Running forever beside the card sampler: limits, watches, keeps, pins
    pub(super) fn backplane_loop(self: Arc<Self>) {
        let mut tick = 0u64;
        loop {
            std::thread::sleep(Duration::from_secs(2));
            tick += 1;
            self.withdraw_queued();
            self.watch_videos();
            self.watch_chat_load();
            if tick % 15 == 0 {
                self.keep_up();
                self.chat_idle_out();
            }
            if tick % 15 == 7 {
                self.ingest_clips();
            }
            // first soon after a start (the sampler has seen the cards by then)
            if tick % 30 == 3 {
                self.reload_pinned();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_of_ours_keeps_its_fields_through_a_message() {
        let f = wait_too_long(&json!({"wait_s": 50.0}), 10.0, None);
        assert_eq!(f.0, 503);
        let (code, fields, words) = untag(&f.1).unwrap();
        assert_eq!(code, "wait_too_long");
        assert_eq!(fields["max_willing_to_wait_s"], 10.0);
        assert_eq!(fields["eta"]["wait_s"], 50.0);
        assert!(fields.get("waited_s").is_none() && words.contains("nothing was queued"));
        let (_, fields, _) = untag(&wait_too_long(&json!({}), 10.0, Some(12.0)).1).unwrap();
        assert_eq!(fields["waited_s"], 12.0, "withdrawn: how long it waited");
        assert!(untag("an ordinary message").is_none());
    }

    #[test]
    fn a_request_past_its_limit_before_it_was_sent_is_withdrawn() {
        begin(&json!({"max_willing_to_wait_s": 5}), json!({}), now() - 10.0);
        assert!(past_limit().is_some());
        sent_to("m");
        assert!(past_limit().is_none(), "started: the limit is on the time to start");
        begin(&json!({}), json!({}), now() - 10.0);
        assert!(past_limit().is_none(), "no limit");
        REQ.with(|x| *x.borrow_mut() = None);
    }

    #[test]
    fn the_fields_are_the_doors_own_and_seconds() {
        assert!(check_fields(&json!({"rsvp_s": 0, "max_willing_to_wait_s": 3.5})).is_ok());
        assert!(check_fields(&json!({"rsvp_s": -1})).is_err());
        assert!(check_fields(&json!({"max_willing_to_wait_s": "soon"})).is_err());
        let b = stripped(&json!({"rsvp_s": 5, "max_willing_to_wait_s": 5, "prompt": "x"}));
        assert_eq!(b, json!({"prompt": "x"}));
    }

    #[test]
    fn a_job_in_its_queue_is_past_its_limit_by_when_it_was_made() {
        let j = json!({"created": 100.0, "max_willing_to_wait_s": 10});
        assert_eq!(past_in_queue(&j, 105.0), None);
        assert_eq!(past_in_queue(&j, 115.0), Some(15.0));
        assert_eq!(past_in_queue(&json!({"created": 100.0}), 1e9), None);
    }

    #[test]
    fn a_clip_row_names_its_engine() {
        let all = vec![json!({"id": "minimax-h3", "kind": "video"}), json!({"id": "minimax-h3-ref2va", "kind": "video"})];
        assert_eq!(clip_model(&json!({"engine": "INT8"}), &all).as_deref(), Some("minimax-h3"));
        assert_eq!(clip_model(&json!({"engine": "REF2VA"}), &all).as_deref(), Some("minimax-h3-ref2va"));
        assert_eq!(clip_model(&json!({}), &[]), None);
    }
}
