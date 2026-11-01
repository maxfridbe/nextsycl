//! ETAs for requests and the decisions made from them: `max_willing_to_wait_s` (decline, withdraw), `system.eta`'s
//! `would_accept`, a running job's time left. The history ([`crate::history`]) holds the numbers; the host supplies the
//! jobs that are ahead.

use serde_json::{json, Value};

use crate::history::History;
use crate::rsvp::wait_tag;
use crate::Fail;

/// A job queued or running, as the ETA needs it
#[derive(Clone, Debug)]
pub struct JobView {
    pub id: String,
    pub kind: String,
    pub method: String,
    pub model: String,
    pub body: Value,
    /// "queued" | "running"
    pub state: String,
    pub started: Option<f64>,
    /// its place in the queue (a smaller one runs first)
    pub order: f64,
}

/// A job's expected run in seconds: its shape's median, else its model's (None: no history)
pub fn job_run_est(h: &History, j: &JobView) -> Option<f64> {
    h.eta(&j.method, &j.model, &j.body, true, Some(0.0))["run_s"].as_f64()
}

/// Seconds of work ahead of a new request (or of queued job `exclude`, when it is already queued: only the jobs ahead of
/// it count) on the kind's `servers` servers: running jobs' remaining time and the queued jobs ahead, each from history,
/// over the servers; None when any part has no history
pub fn ahead_s(h: &History, now: f64, jobs: &[JobView], kind: &str, exclude: Option<&str>, servers: usize) -> Option<f64> {
    let mine_order = exclude.and_then(|id| jobs.iter().find(|j| j.id == id)).map(|j| j.order);
    let mut total = Some(0.0);
    for j in jobs.iter().filter(|j| j.kind == kind && Some(j.id.as_str()) != exclude) {
        let ahead = j.state == "running" || (j.state == "queued" && mine_order.is_none_or(|o| j.order < o));
        if !ahead {
            continue;
        }
        let est = job_run_est(h, j);
        let left = est.map(|e| if j.state == "running" { (e - (now - j.started.unwrap_or(now))).max(0.0) } else { e });
        total = total.zip(left).map(|(a, b)| a + b);
    }
    total.map(|t| t / servers.max(1) as f64)
}

/// `max_willing_to_wait_s` against the ETA, before anything is queued, loaded or evicted: the refusal (503 with the
/// wait_too_long fields in its tagged message) when the ETA's wait is known and longer
pub fn check_wait(eta: &Value, max: f64) -> Result<(), Fail> {
    match eta["wait_s"].as_f64() {
        Some(w) if w > max => Err((503, wait_tag(json!({"eta": eta, "max_willing_to_wait_s": max}),
                                                 &format!("it would start in about {w:.0} s, past the {max:.0} s you will wait; nothing was started for it")))),
        _ => Ok(()),
    }
}

/// The error for a request withdrawn after `waited` s without starting (the ETA was unknown)
pub fn wait_expired(eta: &Value, max: f64, waited: f64) -> Fail {
    (503, wait_tag(json!({"eta": eta, "max_willing_to_wait_s": max, "waited_s": waited.round()}),
                   &format!("it had not started after {waited:.0} s, past the {max:.0} s you will wait; withdrawn")))
}

/// `system.eta`: whether a request with this ETA would be accepted under `max` (unknown waits are accepted: the clock enforces)
pub fn would_accept(eta: &Value, max: Option<f64>) -> bool {
    max.is_none_or(|m| eta["wait_s"].as_f64().is_none_or(|w| w <= m))
}

/// Whether a queued job that has waited `waited` s will not start in time: it has waited past its limit, or its updated
/// ETA says waited + wait_s is past it
pub fn late(eta: &Value, max: f64, waited: f64) -> bool {
    waited > max || eta["wait_s"].as_f64().is_some_and(|w| waited + w > max)
}

/// The words and fields of a withdrawn queued job's error
pub fn withdrawn(eta: &Value, max: f64, waited: f64) -> Fail {
    (503, wait_tag(json!({"eta": eta, "max_willing_to_wait_s": max, "waited_s": waited.round()}),
                   &format!("it would not start within the {max:.0} s you will wait ({waited:.0} s gone); withdrawn")))
}

/// A running job's seconds left: from its progress fraction when it has one, else its history's run time less what it
/// has run
pub fn running_left(eta: &Value, ran: f64, fraction: Option<f64>) -> Option<f64> {
    fraction.filter(|f| *f > 0.02 && *f < 1.0).map(|f| ran * (1.0 - f) / f).or_else(|| eta["run_s"].as_f64().map(|r| (r - ran).max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{MemStorage, Record};
    use crate::rsvp::wait_untag;

    fn hist() -> History {
        History::open(Box::new(MemStorage::default()))
    }

    fn rec(h: &History, model: &str, wait_s: f64, run_s: f64) {
        let shape = h.shapes().shape("audio.speak", model, &json!({"model": model}));
        h.record(Record { method: "audio.speak".into(), model: model.into(), shape, wait_s, run_s, outcome: "ok".into() });
    }

    fn job(id: &str, state: &str, order: f64, started: Option<f64>) -> JobView {
        JobView { id: id.into(), kind: "audio".into(), method: "audio.speak".into(), model: "a".into(), body: json!({"model": "a"}), state: state.into(), started, order }
    }

    #[test]
    fn waiting_on_a_busy_job_adds_its_remaining_time() {
        let h = hist();
        rec(&h, "a", 0.0, 30.0);
        // one running job that started 10 s ago (20 s left) and two queued: 20 + 30 + 30 over one server
        let jobs = vec![job("r", "running", 1.0, Some(90.0)), job("q1", "queued", 2.0, None), job("q2", "queued", 3.0, None)];
        assert_eq!(ahead_s(&h, 100.0, &jobs, "audio", None, 1), Some(80.0));
        // over two servers
        assert_eq!(ahead_s(&h, 100.0, &jobs, "audio", None, 2), Some(40.0));
        // for the second queued job: the running one and the first queued are ahead of it
        assert_eq!(ahead_s(&h, 100.0, &jobs, "audio", Some("q2"), 1), Some(50.0));
        // a job with no history makes the wait unknown
        let mut unknown = job("u", "queued", 0.5, None);
        unknown.model = "never-ran".into();
        unknown.body = json!({"model": "never-ran"});
        assert_eq!(ahead_s(&h, 100.0, &[unknown], "audio", None, 1), None);
    }

    #[test]
    fn a_wait_past_the_limit_is_declined_with_the_eta_and_a_known_one_below_it_is_accepted() {
        let eta = json!({"wait_s": 60.0, "basis": "shape"});
        let (f, words) = wait_untag(&check_wait(&eta, 10.0).unwrap_err().1).unwrap();
        assert_eq!(f["eta"]["wait_s"], 60.0);
        assert_eq!(f["max_willing_to_wait_s"], 10.0);
        assert!(words.contains("nothing was started"));
        assert!(check_wait(&eta, 120.0).is_ok());
        // an unknown wait is not declined: the clock enforces it
        assert!(check_wait(&json!({"wait_s": null}), 1.0).is_ok());
        assert!(!would_accept(&eta, Some(10.0)));
        assert!(would_accept(&eta, Some(60.0)));
        assert!(would_accept(&json!({"wait_s": null}), Some(1.0)));
        assert!(would_accept(&eta, None));
    }

    #[test]
    fn an_expired_wait_says_how_long_it_waited() {
        let (f, _) = wait_untag(&wait_expired(&json!({"wait_s": null}), 5.0, 6.2).1).unwrap();
        assert_eq!((f["waited_s"].as_f64(), f["max_willing_to_wait_s"].as_f64()), (Some(6.0), Some(5.0)));
    }

    #[test]
    fn a_queued_job_is_late_by_the_clock_or_by_its_updated_eta() {
        assert!(late(&json!({"wait_s": null}), 2.0, 30.0));
        assert!(!late(&json!({"wait_s": null}), 600.0, 30.0));
        assert!(late(&json!({"wait_s": 100.0}), 60.0, 10.0));
        assert!(!late(&json!({"wait_s": 20.0}), 60.0, 10.0));
        let (f, _) = wait_untag(&withdrawn(&json!({"wait_s": 100.0}), 60.0, 10.0).1).unwrap();
        assert_eq!(f["waited_s"], 10.0);
    }

    #[test]
    fn a_running_jobs_time_left_counts_down_from_its_progress_or_its_history() {
        assert_eq!(running_left(&json!({"run_s": 60.0}), 20.0, None), Some(40.0));
        assert_eq!(running_left(&json!({"run_s": 60.0}), 90.0, None), Some(0.0));
        assert_eq!(running_left(&json!({"run_s": null}), 20.0, None), None);
        // 25% done after 20 s: 60 s left, whatever the history says
        assert_eq!(running_left(&json!({"run_s": 10.0}), 20.0, Some(0.25)), Some(60.0));
    }
}
