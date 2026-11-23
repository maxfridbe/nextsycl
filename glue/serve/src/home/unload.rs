//! A force unload of an idle model (`system.unload` with `force_when_idle`, the page's ×): refused while the model is
//! answering a request, running a job or loading (409 busy, retryable) - nothing is ever stopped mid-work; idle, it
//! unloads now, over its pin (unpinned, and the answer says so) and its hold (rsvp_s or keep). Busy is the front
//! door's own notion, read from the model's row (system.state's models: its state and each copy's `busy`, which are
//! the servers' answers, the requests sent on from here and the chat server's active requests) and the open jobs
//! (system.busy's `jobs`) - never a second one.

use serde_json::{json, Value};

use super::Home;

type Fail = (u16, String);

/// What a force unload of an idle model does
#[derive(Debug, PartialEq)]
pub(crate) enum Verdict {
    /// not loaded (on that card): nothing to do
    NotLoaded,
    /// answering, running a job or loading: refused, nothing stopped (the words start "busy:": error.code busy)
    Busy(String),
    /// idle: unloaded now - unpinned first when pinned, its hold ended
    Unload { unpin: bool, hold: bool },
}

/// Whether the copy is on card `g`
fn on_card(i: &Value, g: u64) -> bool {
    i["gpu"].as_u64() == Some(g) || i["gpus"].as_array().is_some_and(|a| a.iter().any(|x| x.as_u64() == Some(g)))
}

/// The decision, from the model's row (model_states: id, state, instances[] {gpu, gpus, port, busy}, pinned, held_s),
/// the card asked about (None: every copy) and the open jobs (jobs_open: {model, held}; a held job waits for a release,
/// not for this model)
pub(crate) fn decide(row: &Value, gpu: Option<u64>, jobs: &[Value]) -> Verdict {
    let id = row["id"].as_str().unwrap_or("");
    let state = row["state"].as_str().unwrap_or("");
    let busy = |why: &str| Verdict::Busy(format!("busy: {id} {why} - try when idle (nothing is stopped mid-work)"));
    if state == "loading" {
        return busy("is loading");
    }
    if !matches!(state, "loaded" | "busy") {
        return Verdict::NotLoaded;
    }
    let copies: Vec<&Value> = row["instances"].as_array().into_iter().flatten().filter(|i| gpu.is_none_or(|g| on_card(i, g))).collect();
    if gpu.is_some() && copies.is_empty() {
        return Verdict::NotLoaded;
    }
    // any copy at work refuses, whichever card was asked about (the conservative reading: a model is busy or not)
    if state == "busy" || row["instances"].as_array().into_iter().flatten().any(|i| i["busy"] == true) {
        return busy("is answering a request");
    }
    let open = jobs.iter().filter(|j| j["model"] == id && j["held"] != true).count();
    if open > 0 {
        return busy(&format!("has {open} job(s) open"));
    }
    Verdict::Unload { unpin: row["pinned"] == true, hold: row["held_s"].as_f64().is_some_and(|s| s > 0.0) }
}

impl Home {
    /// `system.unload {model, force_when_idle: true, gpu}`: `decide`, then the unload of that copy (or every copy)
    pub(crate) fn unload_when_idle(&self, id: &str, gpu: Option<u64>) -> Result<Value, Fail> {
        self.entry(id)?;
        let all = self.models();
        let cards = self.named_cards(&all);
        let row = self.model_states(&all, &cards).into_iter().find(|m| m["id"] == id).ok_or_else(|| (404, format!("no model {id}")))?;
        match decide(&row, gpu, &self.jobs_open()) {
            Verdict::NotLoaded => Ok(json!({"model": id, "loaded": false, "was_loaded": false, "unpinned": false, "hold_ended": false})),
            Verdict::Busy(why) => Err((409, why)),
            Verdict::Unload { unpin, hold } => {
                if unpin {
                    self.unpin(id)?;
                    eprintln!("unload {id}: unpinned (a force unload of an idle model)");
                }
                let ports: Vec<u16> = row["instances"].as_array().into_iter().flatten()
                    .filter(|i| gpu.is_some_and(|g| on_card(i, g))).filter_map(|i| i["port"].as_u64().map(|p| p as u16)).collect();
                let only = gpu.map(|g| (g as usize, ports.as_slice()));
                self.unload_on(id, only)?;
                Ok(json!({"model": id, "loaded": false, "was_loaded": true, "gpu": gpu, "unpinned": unpin, "hold_ended": hold}))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(state: &str, instances: Value) -> Value {
        json!({"id": "m", "state": state, "instances": instances, "pinned": false, "held_s": 0.0})
    }

    #[test]
    fn busy_is_refused() {
        let r = row("loaded", json!([{"gpu": 1, "port": 8085, "busy": true}]));
        let Verdict::Busy(why) = decide(&r, None, &[]) else { panic!("a copy answering must refuse") };
        assert!(why.starts_with("busy:") && why.contains("try when idle"), "{why}");
        assert!(matches!(decide(&r, Some(1), &[]), Verdict::Busy(_)));
        // an image model whose server says busy (its row's state)
        assert!(matches!(decide(&row("busy", json!([{"gpu": 0, "port": 8086, "busy": false}])), None, &[]), Verdict::Busy(_)));
        // loading is refused too
        assert!(matches!(decide(&row("loading", json!([])), None, &[]), Verdict::Busy(_)));
        // a copy at work on the other card refuses an unload of this card's copy as well
        let two = row("loaded", json!([{"gpu": 0, "port": 8086, "busy": false}, {"gpu": 1, "port": 8087, "busy": true}]));
        assert!(matches!(decide(&two, Some(0), &[]), Verdict::Busy(_)));
    }

    #[test]
    fn an_open_job_is_refused_a_held_one_is_not() {
        let r = row("loaded", json!([{"gpu": 0, "port": 8086, "busy": false}]));
        let Verdict::Busy(why) = decide(&r, None, &[json!({"model": "m", "held": false})]) else { panic!("an open job must refuse") };
        assert!(why.contains("1 job(s)"), "{why}");
        assert!(matches!(decide(&r, None, &[json!({"model": "m", "held": true}), json!({"model": "other", "held": false})]), Verdict::Unload { .. }));
    }

    #[test]
    fn idle_unloads_over_a_pin_and_a_hold() {
        let mut r = row("loaded", json!([{"gpu": 1, "port": 8085, "busy": false}]));
        assert_eq!(decide(&r, None, &[]), Verdict::Unload { unpin: false, hold: false });
        r["pinned"] = json!(true);
        r["held_s"] = json!(300.0);
        assert_eq!(decide(&r, Some(1), &[]), Verdict::Unload { unpin: true, hold: true });
    }

    #[test]
    fn not_loaded_is_nothing_to_do() {
        for st in ["unloaded", "disabled", "ready"] {
            assert_eq!(decide(&row(st, json!([])), None, &[]), Verdict::NotLoaded, "{st}");
        }
        // loaded, but not on the card asked about (a GPU run there is no copy of the model)
        let r = row("loaded", json!([{"gpu": 1, "port": 8085, "busy": false}]));
        assert_eq!(decide(&r, Some(0), &[]), Verdict::NotLoaded);
        // a copy spanning both cards is on each
        let span = row("loaded", json!([{"gpu": 0, "gpus": [0, 1], "port": 8085, "busy": false}]));
        assert!(matches!(decide(&span, Some(1), &[]), Verdict::Unload { .. }));
    }
}
