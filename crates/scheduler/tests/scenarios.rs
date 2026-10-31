//! The scheduler's scenario suite: scripted timelines of requests, pins, holds, evictions, copies and exclusivity against
//! the fake host on its fake clock. Every scenario is a plain-language name. Run it after porting the scheduler to
//! another box with its own `FakeHost`-like harness (README.md: the checklist).

use std::collections::BTreeMap;
use std::time::Duration;

use backplane_scheduler::eta::{self, JobView};
use backplane_scheduler::fake::FakeHost;
use backplane_scheduler::history::{History, MemStorage};
use backplane_scheduler::host::Host;
use backplane_scheduler::place::{ensure, place, PlaceReq};
use backplane_scheduler::profile::Profile;
use backplane_scheduler::room::ModelInfo;
use backplane_scheduler::rsvp::{self, wait_untag};
use serde_json::{json, Value};

const CARD: u64 = 12288;

fn profile(v: Value) -> Profile {
    Profile::from_card(&v).unwrap()
}

fn plain() -> Profile {
    Profile::default()
}

fn exclusive() -> Profile {
    profile(json!({"coexistence": {"mode": "exclusive", "exclusive_reason": "fills the card"}}))
}

/// One card, `slots` audio servers
fn box1(slots: usize) -> FakeHost {
    FakeHost::new(&[CARD], &[("audio", slots)])
}

/// The request for model `m` that needs `mb` on each of its cards
fn req<'a>(m: &'a ModelInfo, needs: &'a BTreeMap<usize, u64>) -> PlaceReq<'a> {
    PlaceReq { model: m, job: None, need_mb: needs, asked_mb: 0 }
}

fn needs(m: &ModelInfo, mb: u64) -> BTreeMap<usize, u64> {
    m.gpus.iter().map(|g| (*g, mb)).collect()
}

fn ask(host: &FakeHost, m: &ModelInfo, mb: u64, wait_s: u64) -> Result<(usize, u16), (u16, String)> {
    let n = needs(m, mb);
    ensure(host, &req(m, &n), Duration::from_secs(wait_s), None, &json!({}))
}

#[test]
fn s01_a_cold_request_starts_the_model_on_its_card() {
    let h = box1(1);
    let a = h.add("a", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    assert_eq!(h.log(), vec!["start audio a slot 0 gpu 0"]);
    assert_eq!(h.loaded("audio"), vec![Some("a".to_string())]);
}

#[test]
fn s02_a_loaded_idle_model_is_used_without_starting_anything() {
    let h = box1(1);
    let a = h.add("a", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    ask(&h, &a, 7000, 90).unwrap();
    assert_eq!(h.count("start"), 1);
}

#[test]
fn s03_an_idle_unpinned_holder_is_stopped_for_the_room() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    ask(&h, &b, 7000, 90).unwrap();
    assert_eq!(h.count("stop audio slot 0"), 1);
    assert_eq!(h.loaded("audio"), vec![None, Some("b".to_string())]);
}

#[test]
fn s04_a_busy_holder_is_waited_for_on_the_clock_and_then_stopped() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.busy("a", 20.0, 0);
    let t0 = h.now();
    ask(&h, &b, 7000, 300).unwrap();
    assert!(h.now() - t0 >= 20.0, "it waited for a's 20 s of work: {}", h.now() - t0);
    assert_eq!(h.loaded("audio").into_iter().flatten().collect::<Vec<_>>(), vec!["b".to_string()]);
}

#[test]
fn s05_a_pinned_model_is_never_evicted_and_the_no_room_answer_names_the_pin() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.pin("a", true);
    let (st, why) = ask(&h, &b, 7000, 12).unwrap_err();
    assert_eq!(st, 503);
    assert!(why.contains("pinned: a") && why.contains("needs ~"), "{why}");
    assert_eq!(h.count("stop"), 0);
}

#[test]
fn s06_unpinning_makes_it_evictable_but_stops_nothing_by_itself() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.pin("a", true);
    assert!(ask(&h, &b, 7000, 6).is_err());
    h.pin("a", false);
    assert_eq!(h.count("stop"), 0, "unpin unloads nothing");
    ask(&h, &b, 7000, 90).unwrap();
    assert_eq!(h.count("stop"), 1);
}

#[test]
fn s07_a_hold_blocks_eviction_until_it_runs_out_on_the_clock() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.hold_after("a", 60.0);
    let (_, why) = ask(&h, &b, 7000, 20).unwrap_err();
    assert!(why.contains("held by rsvp_s: a"), "{why}");
    assert_eq!(h.count("stop"), 0);
    h.advance(61.0);
    ask(&h, &b, 7000, 90).unwrap();
    assert_eq!(h.count("stop"), 1);
}

#[test]
fn s08_a_hold_is_clamped_by_the_cap_never_shortened_and_zero_releases() {
    let h = box1(1);
    let a = h.add("a", "audio", 2000, &[0], plain());
    ask(&h, &a, 2000, 90).unwrap();
    // the caller asks a week; the server's cap is an hour
    let g = rsvp::granted(Some(604800.0), rsvp::cap_of(Some("3600".into()))).unwrap();
    assert_eq!(g, 3600.0);
    h.hold_after("a", g);
    h.hold_after("a", 10.0);
    assert!((h.held_s("a") - 3600.0).abs() < 1.0, "a later, shorter ask does not shorten it");
    // zero while held: the hold stands; once it has run out, zero releases
    h.hold_after("a", 0.0);
    assert!(h.held_s("a") > 3500.0);
    h.advance(3601.0);
    assert_eq!(h.held_s("a"), 0.0);
}

#[test]
fn s09_extra_copies_start_only_up_to_the_recommended_count_when_requests_wait() {
    let h = box1(3);
    let s = h.add("s", "audio", 2000, &[0], profile(json!({"concurrency": {"max_copies_per_card": 3, "recommended_copies": 2}})));
    ask(&h, &s, 2000, 90).unwrap();
    assert_eq!(h.count("start"), 1);
    // the first copy has work: a second starts on the same card
    h.busy("s", 600.0, 1);
    ask(&h, &s, 2000, 90).unwrap();
    assert_eq!(h.count("start"), 2);
    // both have work: the recommended 2 are there, a third is not started (it queues on one of them)
    h.busy("s", 600.0, 1);
    ask(&h, &s, 2000, 90).unwrap();
    assert_eq!(h.count("start"), 2, "{:?}", h.log());
}

#[test]
fn s10_by_default_there_is_one_copy_a_card_and_the_next_goes_to_the_other_card() {
    let h = FakeHost::new(&[CARD, CARD], &[("audio", 3)]);
    let s = h.add("s", "audio", 2000, &[0, 1], plain());
    ask(&h, &s, 2000, 90).unwrap();
    h.busy("s", 600.0, 1);
    ask(&h, &s, 2000, 90).unwrap();
    assert_eq!(h.log().last().unwrap(), "start audio s slot 1 gpu 1");
    h.busy("s", 600.0, 1);
    ask(&h, &s, 2000, 90).unwrap();
    assert_eq!(h.count("start"), 2, "one a card: no third copy");
}

#[test]
fn s11_an_exclusive_model_gets_its_card_to_itself_even_where_the_memory_would_fit() {
    let h = box1(2);
    let a = h.add("a", "audio", 2000, &[0], plain());
    let x = h.add("x", "audio", 2000, &[0], exclusive());
    ask(&h, &a, 2000, 90).unwrap();
    ask(&h, &x, 2000, 90).unwrap();
    assert_eq!(h.count("stop audio slot 0"), 1);
    assert_eq!(h.loaded("audio"), vec![None, Some("x".to_string())]);
}

#[test]
fn s12_nothing_else_starts_beside_a_loaded_exclusive_model_and_the_answer_names_it() {
    let h = box1(2);
    let a = h.add("a", "audio", 2000, &[0], plain());
    let x = h.add("x", "audio", 2000, &[0], exclusive());
    ask(&h, &x, 2000, 90).unwrap();
    h.pin("x", true);
    let (_, why) = ask(&h, &a, 2000, 10).unwrap_err();
    assert!(why.contains("exclusive: x") && why.contains("pinned: x"), "{why}");
    assert_eq!(h.count("start"), 1);
    // unpinned and idle, x is an idle holder like any: it goes for a
    h.pin("x", false);
    ask(&h, &a, 2000, 90).unwrap();
    assert_eq!(h.loaded("audio").iter().flatten().collect::<Vec<_>>(), vec!["a"]);
}

#[test]
fn s13_shared_with_limits_who_may_be_a_resident() {
    let h = box1(2);
    let a = h.add("a", "audio", 2000, &[0], profile(json!({"coexistence": {"shared_with": ["kind:image"]}})));
    let b = h.add("b", "audio", 2000, &[0], plain());
    ask(&h, &a, 2000, 90).unwrap();
    h.pin("a", true);
    assert!(ask(&h, &b, 2000, 10).is_err(), "b is not on a's list and a is pinned");
    h.pin("a", false);
    ask(&h, &b, 2000, 90).unwrap();
    assert_eq!(h.count("stop"), 1);
}

#[test]
fn s14_a_disabled_model_is_never_placed() {
    let h = box1(1);
    let a = h.add("a", "audio", 2000, &[0], plain());
    h.disable("a");
    let a = ModelInfo { enabled: false, ..a };
    let n = needs(&a, 2000);
    assert_eq!(place(&h, &req(&a, &n)).unwrap_err().0, 409);
    assert_eq!(h.log().len(), 0);
}

#[test]
fn s15_a_cancel_ends_a_jobs_wait() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.pin("a", true);
    h.cancel("job-1");
    let n = needs(&b, 7000);
    let r = PlaceReq { model: &b, job: Some("job-1"), need_mb: &n, asked_mb: 0 };
    let e = ensure(&h, &r, Duration::from_secs(1800), None, &json!({})).unwrap_err();
    assert_eq!(e.0, 499);
}

#[test]
fn s16_an_idle_chat_model_is_unloaded_for_room_and_a_pinned_one_is_not() {
    let h = FakeHost::new(&[CARD], &[("audio", 1)]);
    let a = h.add("a", "audio", 7000, &[0], plain());
    h.chat("qwen", 6000, 0, true);
    // 12288 - 300 - 6000 = 5988 free < 8536: the idle chat model goes
    ask(&h, &a, 7000, 90).unwrap();
    assert_eq!(h.count("unload chat qwen"), 1);
    let h = FakeHost::new(&[CARD], &[("audio", 1)]);
    let a = h.add("a", "audio", 7000, &[0], plain());
    h.chat("qwen", 6000, 0, true);
    h.pin("qwen", true);
    assert!(ask(&h, &a, 7000, 10).is_err());
    assert_eq!(h.count("unload chat"), 0);
}

#[test]
fn s17_a_request_declined_by_its_wait_leaves_no_side_effects() {
    let h = box1(1);
    let a = h.add("a", "audio", 2000, &[0], plain());
    let hist = History::open(Box::new(MemStorage::default()));
    hist.record_work("audio.speak", &json!({"model": "a"}), "a", 60.0, 5.0, "ok");
    let e = hist.eta("audio.speak", "a", &json!({"model": "a"}), false, Some(0.0));
    assert_eq!(e["wait_s"], 60.0);
    let (_, msg) = eta::check_wait(&e, 10.0).unwrap_err();
    let (fields, _) = wait_untag(&msg).unwrap();
    assert_eq!(fields["eta"]["basis"], "shape");
    assert!(h.log().is_empty(), "nothing started, stopped or queued");
    assert!(eta::check_wait(&e, 120.0).is_ok());
    assert!(!eta::would_accept(&e, Some(10.0)) && eta::would_accept(&e, Some(120.0)));
    let _ = a;
}

#[test]
fn s18_an_unknown_eta_is_enforced_by_the_clock_and_starts_nothing() {
    let h = box1(2);
    let a = h.add("a", "audio", 7000, &[0], plain());
    let b = h.add("b", "audio", 7000, &[0], plain());
    ask(&h, &a, 7000, 90).unwrap();
    h.pin("a", true);
    let n = needs(&b, 7000);
    let e = ensure(&h, &req(&b, &n), Duration::from_secs(1800), Some(8.0), &json!({"wait_s": null})).unwrap_err();
    let (f, _) = wait_untag(&e.1).expect("wait_too_long");
    assert!(f["waited_s"].as_f64().unwrap() >= 8.0);
    assert_eq!(h.count("start"), 1);
    assert_eq!(h.count("stop"), 0);
}

#[test]
fn s19_a_hot_model_always_starts_at_once_whatever_the_wait_limit() {
    let h = box1(1);
    let a = h.add("a", "audio", 2000, &[0], plain());
    ask(&h, &a, 2000, 90).unwrap();
    let n = needs(&a, 2000);
    let t0 = h.now();
    ensure(&h, &req(&a, &n), Duration::from_secs(90), Some(0.0), &json!({"wait_s": 0.0})).unwrap();
    assert_eq!(h.now(), t0);
    assert_eq!(h.count("start"), 1);
}

#[test]
fn s20_a_queued_job_whose_eta_grows_past_its_limit_is_withdrawn_and_one_within_it_stays() {
    let hist = History::open(Box::new(MemStorage::default()));
    hist.record_work("audio.speak", &json!({"model": "a"}), "a", 0.0, 30.0, "ok");
    let job = |id: &str, state: &str, order: f64, started: Option<f64>| JobView {
        id: id.into(), kind: "audio".into(), method: "audio.speak".into(), model: "a".into(), body: json!({"model": "a"}), state: state.into(), started, order };
    // the queue moved: now three jobs ahead of the last, 30 s each, on one server
    let jobs = vec![job("r", "running", 1.0, Some(100.0)), job("q1", "queued", 2.0, None), job("q2", "queued", 3.0, None), job("q3", "queued", 4.0, None)];
    let ahead = eta::ahead_s(&hist, 100.0, &jobs, "audio", Some("q3"), 1);
    assert_eq!(ahead, Some(90.0), "the running one's full 30 plus two queued");
    let e = hist.eta("audio.speak", "a", &json!({"model": "a"}), false, ahead);
    // (a never started cold: no spin-up samples; the wait is the work ahead alone only when the model needs no start)
    let hot_e = hist.eta("audio.speak", "a", &json!({"model": "a"}), true, ahead);
    assert_eq!(hot_e["wait_s"], 0.0);
    assert!(e["wait_s"].is_null(), "no spin-up history: unknown, never invented");
    let mut known = e.clone();
    known["wait_s"] = json!(120.0);
    assert!(eta::late(&known, 100.0, 10.0), "10 s gone + 120 s ahead is past 100");
    assert!(!eta::late(&known, 600.0, 10.0));
}

#[test]
fn s21_the_history_keeps_the_last_hundred_and_survives_a_restart() {
    let st = MemStorage::default();
    let hist = History::open(Box::new(st.clone()));
    for i in 0..130 {
        hist.record_work("audio.speak", &json!({"model": "a"}), "a", 0.0, i as f64, "ok");
    }
    let again = History::open(Box::new(st));
    let e = again.eta("audio.speak", "a", &json!({"model": "a"}), true, Some(0.0));
    assert_eq!(e["samples"], 100);
    assert_eq!(e["run_s"], 80.0, "the median of runs 30..129");
}

#[test]
fn s22_the_eta_falls_back_from_the_shape_to_the_model_to_none_and_failures_feed_no_run_time() {
    let hist = History::open(Box::new(MemStorage::default()));
    let p = json!({"model": "m", "input": "x".repeat(300)});
    assert_eq!(hist.eta("audio.speak", "m", &p, true, Some(0.0))["basis"], "none");
    hist.record_work("audio.speak", &json!({"model": "m", "input": "x".repeat(900)}), "m", 0.0, 9.0, "ok");
    let e = hist.eta("audio.speak", "m", &p, true, Some(0.0));
    assert_eq!((e["basis"].as_str(), e["run_s"].as_f64()), (Some("model"), Some(9.0)));
    hist.record_work("audio.speak", &p, "m", 0.0, 500.0, "failed");
    assert_eq!(hist.eta("audio.speak", "m", &p, true, Some(0.0))["basis"], "model", "a failed run is no run time");
    hist.record_work("audio.speak", &p, "m", 0.0, 4.0, "ok");
    let e = hist.eta("audio.speak", "m", &p, true, Some(0.0));
    assert_eq!((e["basis"].as_str(), e["run_s"].as_f64()), (Some("shape"), Some(4.0)));
}
