//! "Loaded X ago, last request Y ago": the timing fields on every model row (system.models, system.state), on a
//! card's vouched rows and on system.busy's slots - nextcuda's shapes (its docs/control-plane.md).
//!
//! Times are UTC Unix seconds (float); ages are whole seconds as of the answer; every field may be null.
//! - `loaded_at`: when the copy finished loading (a model: its longest-loaded copy - the earliest).
//! - `last_request_at`: the last moment a request was on the copy - now while one runs, else when the last one ended;
//!   null when none came since the load (a model: the latest of its copies).
//! - `loaded_s_ago` = now - loaded_at; `idle_s` = now - last_request_at (0 while a request runs, null with none yet;
//!   a model: the smallest).
//! - `unload_in_s`: the seconds until the copy idle-unloads - null when pinned, held, busy or idle unload is off (a
//!   model: the soonest). The countdown runs from the last request, or from the load when there has been none.
//!
//! What counts as a request is anything that uses the copy (an RPC, an OpenAI call, its page through /open/<id>, a
//! live /v1/realtime session for as long as it lasts, chat through the switcher, clips on the video daemon); the
//! door's keep-alive touches and health probes do not.

use serde_json::{json, Map, Value};

/// One loaded copy of a model, as its server, the daemon or this door knows it
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Copy {
    pub gpu: Option<usize>,
    /// a copy spanning cards (Sophon-Realtime): every card it holds
    pub gpus: Vec<usize>,
    pub port: Option<u16>,
    /// a request on it now
    pub busy: bool,
    pub loaded_at: Option<f64>,
    /// the last request's end (or start, while unknown), not counting one running now
    pub last_request_at: Option<f64>,
    /// when it idle-unloads, as a time (null: never - idle unload off for it)
    pub unload_at: Option<f64>,
}

fn age(now: f64, t: Option<f64>) -> Value {
    t.map_or(Value::Null, |t| json!((now - t).max(0.0).floor() as u64))
}

impl Copy {
    /// The last request as the answer says it: now while one runs
    fn last(&self, now: f64) -> Option<f64> {
        if self.busy { Some(now) } else { self.last_request_at }
    }

    fn idle_s(&self, now: f64) -> Option<u64> {
        self.last(now).map(|t| (now - t).max(0.0).floor() as u64)
    }

    /// Seconds to its idle unload: none while busy, kept (pinned or held) or without one
    fn unload_in_s(&self, now: f64, kept: bool) -> Option<u64> {
        if kept || self.busy {
            return None;
        }
        self.unload_at.map(|t| (t - now).max(0.0).ceil() as u64)
    }

    /// The four time fields (loaded_at, last_request_at, loaded_s_ago, idle_s)
    pub fn times(&self, now: f64) -> Map<String, Value> {
        let mut o = Map::new();
        o.insert("loaded_at".into(), json!(self.loaded_at));
        o.insert("last_request_at".into(), json!(self.last(now)));
        o.insert("loaded_s_ago".into(), age(now, self.loaded_at));
        o.insert("idle_s".into(), json!(self.idle_s(now)));
        o
    }

    /// Its `instances[]` entry
    pub fn instance(&self, now: f64, kept: bool) -> Value {
        let mut o = Map::new();
        o.insert("gpu".into(), json!(self.gpu));
        if self.gpus.len() > 1 {
            o.insert("gpus".into(), json!(self.gpus));
        }
        o.insert("port".into(), json!(self.port));
        o.insert("busy".into(), json!(self.busy));
        o.extend(self.times(now));
        o.insert("unload_in_s".into(), json!(self.unload_in_s(now, kept)));
        Value::Object(o)
    }
}

/// The empty times (a model, card row or slot holding nothing)
pub(crate) fn none() -> Map<String, Value> {
    Copy::default().times(0.0).into_iter().map(|(k, _)| (k, Value::Null)).collect()
}

/// A model row's fields from its copies: loaded_at the earliest, last_request_at the latest, idle_s the smallest,
/// unload_in_s the soonest (none when `kept`: pinned or held), and `instances` (empty when not loaded)
pub(crate) fn model_fields(copies: &[Copy], now: f64, kept: bool) -> Map<String, Value> {
    let loaded_at = copies.iter().filter_map(|c| c.loaded_at).reduce(f64::min);
    let last = copies.iter().filter_map(|c| c.last(now)).reduce(f64::max);
    let idle = copies.iter().filter_map(|c| c.idle_s(now)).min();
    let unload = copies.iter().filter_map(|c| c.unload_in_s(now, kept)).min();
    let mut o = Map::new();
    o.insert("loaded_at".into(), json!(loaded_at));
    o.insert("last_request_at".into(), json!(last));
    o.insert("loaded_s_ago".into(), age(now, loaded_at));
    o.insert("idle_s".into(), json!(idle));
    o.insert("unload_in_s".into(), json!(unload));
    o.insert("instances".into(), json!(copies.iter().map(|c| c.instance(now, kept)).collect::<Vec<_>>()));
    o
}

/// The four time fields of a model row's copy on card `gpu` (a card's vouched row, a slot): null when it has none there
pub(crate) fn on_card(row: &Value, gpu: &Value) -> Map<String, Value> {
    let here = row["instances"].as_array().into_iter().flatten()
        .find(|i| i["gpu"] == *gpu || i["gpus"].as_array().is_some_and(|a| a.contains(gpu)));
    times_of(here)
}

/// The four time fields of a model row's copy on `port` (an image, audio or 3D server's slot)
pub(crate) fn on_port(row: &Value, port: &Value) -> Map<String, Value> {
    times_of(row["instances"].as_array().into_iter().flatten().find(|i| !port.is_null() && i["port"] == *port))
}

fn times_of(inst: Option<&Value>) -> Map<String, Value> {
    let mut o = none();
    if let Some(i) = inst {
        for k in ["loaded_at", "last_request_at", "loaded_s_ago", "idle_s"] {
            o.insert(k.into(), i[k].clone());
        }
    }
    o
}

/// What this door saw of an image, audio or 3D server (by its port): the model it holds and since when it was seen
/// (its load when the server does not say), and the requests sent on to it - running now, and when the last ended.
/// Requests only: the page's polling and system.load are not counted (life.rs Lease).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PortUse {
    pub model: String,
    pub seen_at: f64,
    pub last_request_at: Option<f64>,
    pub working: usize,
}

impl PortUse {
    /// The server on this port holds `model`: a different one than before is a new load (nothing asked of it yet)
    pub fn holds(&mut self, model: &str, now: f64) {
        if self.model != model {
            *self = PortUse { model: model.to_string(), seen_at: now, last_request_at: None, working: 0 };
        }
    }

    /// A request for `model` went to it
    pub fn start(&mut self, model: &str, now: f64) {
        self.holds(model, now);
        self.working += 1;
        self.last_request_at = Some(now);
    }

    /// A request for `model` ended
    pub fn end(&mut self, model: &str, now: f64) {
        if self.model == model {
            self.working = self.working.saturating_sub(1);
            self.last_request_at = Some(now);
        }
    }
}

/// When a copy idle-unloads, from its idle time `after` (none: never) and its clock: the last request, else the load
pub(crate) fn unload_at(after_s: Option<f64>, last_request_at: Option<f64>, loaded_at: Option<f64>) -> Option<f64> {
    let after = after_s.filter(|a| *a > 0.0)?;
    // (a request seen before this load - the one that asked for it - counts from the load)
    let from = match (last_request_at, loaded_at) {
        (Some(r), Some(l)) => r.max(l),
        (r, l) => r.or(l)?,
    };
    Some(from + after)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(gpu: usize, loaded: f64, last: Option<f64>, unload: Option<f64>) -> Copy {
        Copy { gpu: Some(gpu), port: Some(8086 + gpu as u16), loaded_at: Some(loaded), last_request_at: last, unload_at: unload, ..Default::default() }
    }

    #[test]
    fn a_model_takes_the_earliest_load_the_latest_request_the_smallest_idle_and_the_soonest_unload() {
        let now = 10_000.0;
        let a = copy(0, 1_000.0, Some(9_000.0), Some(10_600.0));
        let b = copy(1, 2_000.0, Some(9_500.0), Some(10_300.0));
        let f = model_fields(&[a, b], now, false);
        assert_eq!(f["loaded_at"], json!(1_000.0), "the longest-loaded copy");
        assert_eq!(f["loaded_s_ago"], json!(9_000));
        assert_eq!(f["last_request_at"], json!(9_500.0), "the latest request");
        assert_eq!(f["idle_s"], json!(500), "the smallest idle");
        assert_eq!(f["unload_in_s"], json!(300), "the soonest unload");
        assert_eq!(f["instances"].as_array().unwrap().len(), 2);
        assert_eq!(f["instances"][0]["idle_s"], json!(1_000));
        assert_eq!(f["instances"][1]["unload_in_s"], json!(300));
    }

    #[test]
    fn a_busy_copy_is_idle_zero_its_last_request_now_and_no_unload() {
        let now = 10_000.0;
        let mut a = copy(0, 1_000.0, Some(5_000.0), Some(10_100.0));
        a.busy = true;
        let b = copy(1, 1_500.0, Some(9_000.0), Some(10_200.0));
        let f = model_fields(&[a.clone(), b], now, false);
        assert_eq!(f["idle_s"], json!(0));
        assert_eq!(f["last_request_at"], json!(now));
        assert_eq!(f["instances"][0]["unload_in_s"], Value::Null, "a busy copy does not unload");
        assert_eq!(f["unload_in_s"], json!(200), "the other copy's");
        let f = model_fields(&[a], now, false);
        assert_eq!(f["unload_in_s"], Value::Null);
        assert_eq!(f["instances"][0]["busy"], json!(true));
    }

    #[test]
    fn no_request_since_the_load_is_null_and_the_countdown_runs_from_the_load() {
        let now = 1_100.0;
        let unload = unload_at(Some(600.0), None, Some(1_000.0));
        assert_eq!(unload, Some(1_600.0));
        let f = model_fields(&[copy(0, 1_000.0, None, unload)], now, false);
        assert_eq!(f["last_request_at"], Value::Null);
        assert_eq!(f["idle_s"], Value::Null);
        assert_eq!(f["loaded_s_ago"], json!(100));
        assert_eq!(f["unload_in_s"], json!(500));
        assert_eq!(unload_at(Some(600.0), Some(1_050.0), Some(1_000.0)), Some(1_650.0), "from the last request when there is one");
        assert_eq!(unload_at(Some(600.0), Some(900.0), Some(1_000.0)), Some(1_600.0), "a request before the load counts from the load");
        assert_eq!(unload_at(None, Some(1_050.0), Some(1_000.0)), None, "idle unload off");
        assert_eq!(unload_at(Some(0.0), Some(1_050.0), Some(1_000.0)), None, "idle unload off");
    }

    #[test]
    fn a_pinned_or_held_model_has_no_unload() {
        let f = model_fields(&[copy(0, 1_000.0, Some(1_500.0), Some(2_100.0))], 2_000.0, true);
        assert_eq!(f["unload_in_s"], Value::Null);
        assert_eq!(f["instances"][0]["unload_in_s"], Value::Null);
        assert_eq!(f["idle_s"], json!(500), "its other times stay");
    }

    #[test]
    fn a_model_not_loaded_has_null_times_and_no_instances() {
        let f = model_fields(&[], 5.0, false);
        for k in ["loaded_at", "last_request_at", "loaded_s_ago", "idle_s", "unload_in_s"] {
            assert_eq!(f[k], Value::Null, "{k}");
        }
        assert_eq!(f["instances"], json!([]));
    }

    #[test]
    fn a_ports_requests_are_counted_and_a_new_model_there_starts_over() {
        let mut u = PortUse::default();
        u.holds("a", 100.0);
        assert_eq!((u.seen_at, u.last_request_at, u.working), (100.0, None, 0), "loaded, nothing asked");
        u.start("a", 110.0);
        u.start("a", 115.0);
        u.end("a", 120.0);
        assert_eq!((u.working, u.last_request_at), (1, Some(120.0)));
        u.end("a", 130.0);
        assert_eq!((u.working, u.last_request_at), (0, Some(130.0)));
        u.holds("a", 200.0);
        assert_eq!(u.seen_at, 100.0, "the same model stays");
        u.end("b", 210.0);
        assert_eq!(u.last_request_at, Some(130.0), "another model's end is not this one's");
        u.start("b", 300.0);
        assert_eq!((u.model.as_str(), u.seen_at, u.working, u.last_request_at), ("b", 300.0, 1, Some(300.0)), "a swap is a new load");
    }

    #[test]
    fn an_overdue_unload_is_zero_and_a_spanning_copy_lists_its_cards() {
        let mut c = copy(0, 1_000.0, Some(1_000.0), Some(1_500.0));
        c.gpus = vec![0, 1];
        let f = model_fields(&[c], 2_000.0, false);
        assert_eq!(f["unload_in_s"], json!(0));
        assert_eq!(f["instances"][0]["gpus"], json!([0, 1]));
        let row = Value::Object(f);
        assert_eq!(on_card(&row, &json!(1))["idle_s"], json!(1_000), "found on either card");
        assert_eq!(on_card(&row, &json!(2))["idle_s"], Value::Null);
        assert_eq!(on_port(&row, &json!(8086))["loaded_at"], json!(1_000.0));
        assert_eq!(on_port(&row, &Value::Null)["loaded_at"], Value::Null);
    }
}
