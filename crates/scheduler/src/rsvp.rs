//! `rsvp_s`: how long, after this request, the model that serves it stays loaded - a keep-alive (Ollama's keep_alive) -
//! and `max_willing_to_wait_s`, the longest a caller will wait for its request to start.
//!
//! - the caller sends `rsvp_s` (seconds, optional) on any method that runs a model; the server caps it with
//!   `NS_MAX_RSVP_S` (unset or 0: [`DEFAULT_MAX_RSVP_S`], 24 h). Larger is not an error: it is clamped ([`granted`]) and the
//!   answer says what was granted (`rsvp_granted_s`);
//! - when the request ends the model is *held* until now + the granted seconds ([`extend`]: never shortened by a later,
//!   shorter rsvp). A held model is as good as in use: not idle, not evicted. A pin is a hold with no end ([`kept_from`]);
//! - no `rsvp_s`: no hold, the model's normal idle time. `rsvp_s: 0`: release - the model goes as soon as it is idle.
//!
//! The request fields themselves (`rsvp_s`, `max_willing_to_wait_s` in a method's request type) are the host's to declare in
//! its method table; the host reads them from the body with [`scope`] for the length of the call.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde_json::Value;

/// The longest keep-alive when NS_MAX_RSVP_S says none: a day
pub const DEFAULT_MAX_RSVP_S: f64 = 86400.0;

thread_local! {
    static ASKED: Cell<Option<f64>> = const { Cell::new(None) };
    static MAX_WAIT: Cell<Option<f64>> = const { Cell::new(None) };
    static STARTED: std::cell::RefCell<Option<(String, f64)>> = const { std::cell::RefCell::new(None) };
}

/// The caller's `rsvp_s` and `max_willing_to_wait_s` while a call runs on this thread (restored when dropped)
pub struct Scope(Option<f64>, Option<f64>);

pub fn scope(body: &Value) -> Scope {
    let was = ASKED.with(|a| a.replace(body["rsvp_s"].as_f64().filter(|x| *x >= 0.0)));
    let wait = MAX_WAIT.with(|a| a.replace(body["max_willing_to_wait_s"].as_f64().filter(|x| *x >= 0.0)));
    Scope(was, wait)
}

impl Drop for Scope {
    fn drop(&mut self) {
        ASKED.with(|a| a.set(self.0));
        MAX_WAIT.with(|a| a.set(self.1));
    }
}

/// The caller's `rsvp_s` for the call running on this thread
pub fn asked() -> Option<f64> {
    ASKED.with(Cell::get)
}

/// The longest the caller will wait for its request to start, for the call on this thread
pub fn max_wait() -> Option<f64> {
    MAX_WAIT.with(Cell::get)
}

/// The model that took the call on this thread and the seconds it waited for it (room, spin-up): noted when the request
/// starts, read when it ends (the history's wait_s)
pub fn note_start(model: &str, waited_s: f64) {
    STARTED.with(|s| *s.borrow_mut() = Some((model.to_string(), waited_s)));
}

pub fn take_start() -> Option<(String, f64)> {
    STARTED.with(|s| s.borrow_mut().take())
}

/// The server's longest keep-alive from its setting: unset, 0 or not a number is the default
pub fn cap_of(raw: Option<String>) -> f64 {
    raw.and_then(|r| r.trim().parse::<f64>().ok()).filter(|c| *c > 0.0).unwrap_or(DEFAULT_MAX_RSVP_S)
}

/// The keep-alive granted: the caller's against the cap, 0 or more; none when the caller asked for none
pub fn granted(asked: Option<f64>, cap: f64) -> Option<f64> {
    asked.map(|a| a.clamp(0.0, cap))
}

/// A model's hold: kept until `until` (seconds on the host's clock); `release`: the last rsvp was 0 (go when idle)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub until: f64,
    pub release: bool,
}

/// The hold after a request that ended at `now` with `granted` seconds: the latest of the old and the new, never
/// shortened; 0 releases (unless something still holds it)
pub fn extend(cur: Option<Hold>, now: f64, granted: f64) -> Hold {
    let old = cur.unwrap_or(Hold { until: 0.0, release: false });
    if granted > 0.0 {
        Hold { until: old.until.max(now + granted), release: false }
    } else {
        Hold { until: old.until, release: old.until <= now }
    }
}

/// Seconds a hold has left at `now` (0: none)
pub fn left(h: Option<&Hold>, now: f64) -> f64 {
    h.map_or(0.0, |h| (h.until - now).max(0.0))
}

/// The models kept: pinned, or held (a hold that has not run out at `now`)
pub fn kept_from(pinned: &BTreeSet<String>, holds: &BTreeMap<String, Hold>, now: f64) -> BTreeSet<String> {
    let mut v = pinned.clone();
    v.extend(holds.iter().filter(|(_, h)| h.until > now).map(|(id, _)| id.clone()));
    v
}

/// How long a request waits to start: the caller's `max_willing_to_wait_s` when it gave one (it replaces the default,
/// for a job's 30 minutes too, and enforces itself: wait_too_long), else the default
pub fn wait_limit(default: Duration, max_wait: Option<f64>) -> Duration {
    max_wait.map_or(default, |_| Duration::MAX)
}

const WAIT_TAG: &str = "wait_too_long:";

/// A `wait_too_long` refusal as a Fail's message: the error's fields as JSON (eta, max_willing_to_wait_s, waited_s)
pub fn wait_tag(fields: Value, words: &str) -> String {
    format!("{WAIT_TAG}{fields}\n{words}")
}

/// The fields and the words of a [`wait_tag`] message; the host puts them in its error body with code `wait_too_long`
pub fn wait_untag(m: &str) -> Option<(Value, String)> {
    let rest = m.strip_prefix(WAIT_TAG)?;
    let (f, words) = rest.split_once('\n')?;
    Some((serde_json::from_str(f).ok()?, words.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_server_cap_clamps_a_larger_ask() {
        assert_eq!(granted(Some(900.0), 600.0), Some(600.0));
        assert_eq!(granted(Some(30.0), 600.0), Some(30.0));
        assert_eq!(granted(Some(-5.0), 600.0), Some(0.0));
        assert_eq!(granted(None, 600.0), None);
    }

    #[test]
    fn the_cap_setting_defaults_to_a_day_when_unset_zero_or_junk() {
        assert_eq!(cap_of(None), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some("0".into())), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some("soon".into())), DEFAULT_MAX_RSVP_S);
        assert_eq!(cap_of(Some(" 120 ".into())), 120.0);
    }

    #[test]
    fn a_hold_extends_and_a_shorter_later_one_does_not_shorten_it() {
        let h = extend(None, 1000.0, 600.0);
        assert_eq!(h, Hold { until: 1600.0, release: false });
        let h2 = extend(Some(h), 1100.0, 60.0);
        assert_eq!(h2.until, 1600.0);
        assert_eq!(extend(Some(h2), 1200.0, 1000.0).until, 2200.0);
        assert_eq!(left(Some(&h2), 1100.0), 500.0);
        assert_eq!(left(Some(&h2), 1700.0), 0.0);
        assert_eq!(left(None, 1.0), 0.0);
    }

    #[test]
    fn zero_releases_unless_something_still_holds() {
        assert!(extend(None, 1000.0, 0.0).release);
        let held = extend(None, 1000.0, 600.0);
        assert!(!extend(Some(held), 1100.0, 0.0).release);
        assert!(extend(Some(held), 1700.0, 0.0).release);
        assert!(!extend(Some(Hold { until: 0.0, release: true }), 2000.0, 30.0).release);
    }

    #[test]
    fn a_pin_and_a_live_hold_are_kept_an_expired_one_is_not() {
        let pinned: BTreeSet<String> = ["p".to_string()].into();
        let holds = BTreeMap::from([("h".to_string(), Hold { until: 100.0, release: false }), ("old".to_string(), Hold { until: 5.0, release: false })]);
        let k = kept_from(&pinned, &holds, 50.0);
        assert!(k.contains("p") && k.contains("h") && !k.contains("old"));
        assert!(!kept_from(&pinned, &holds, 150.0).contains("h"));
    }

    #[test]
    fn the_scope_reads_the_callers_ask_and_restores() {
        assert_eq!(asked(), None);
        {
            let _s = scope(&json!({"rsvp_s": 12.5, "max_willing_to_wait_s": 3.0}));
            assert_eq!((asked(), max_wait()), (Some(12.5), Some(3.0)));
            {
                let _t = scope(&json!({}));
                assert_eq!((asked(), max_wait()), (None, None));
            }
            assert_eq!(asked(), Some(12.5));
        }
        assert_eq!((asked(), max_wait()), (None, None));
    }

    #[test]
    fn a_given_wait_limit_replaces_the_default() {
        let d = Duration::from_secs(90);
        assert_eq!(wait_limit(d, None), d);
        assert_eq!(wait_limit(d, Some(5.0)), Duration::MAX);
        assert_eq!(wait_limit(Duration::from_secs(1800), Some(7200.0)), Duration::MAX);
    }

    #[test]
    fn the_wait_tag_round_trips() {
        let m = wait_tag(json!({"max_willing_to_wait_s": 10.0}), "later");
        let (f, w) = wait_untag(&m).unwrap();
        assert_eq!((f["max_willing_to_wait_s"].as_f64(), w.as_str()), (Some(10.0), "later"));
        assert!(wait_untag("no").is_none());
    }
}
