//! nextsycl's hooks on `Host` (CHANGELOG 0.1.0+nextsycl.1), on the fake host: a host that overrides a few and hands the rest to `FakeHost`.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::time::Duration;

use backplane_scheduler::fake::FakeHost;
use backplane_scheduler::host::{Guard, Host};
use backplane_scheduler::place::{ensure, PlaceReq};
use backplane_scheduler::profile::Profile;
use backplane_scheduler::room::{self, Card, Holder, Idleness, ModelInfo, Proc, Server, Snapshot};
use backplane_scheduler::Fail;
use serde_json::json;

#[derive(Default)]
struct Hooks {
    barred: RefCell<Option<String>>,
    not_ready: RefCell<Option<String>>,
    give_up_after: Cell<Option<u32>>,
    tries: Cell<u32>,
    ram: RefCell<Option<String>>,
    reserve_refusals: Cell<u32>,
    reserved: RefCell<Vec<(usize, u64)>>,
    placed: RefCell<Vec<u16>>,
    locks: Cell<u32>,
    first_listed: Cell<bool>,
    barred_on: RefCell<BTreeMap<usize, String>>,
    /// another model of the kind loading: it holds the kind's lock, so waiting for it waits for that load
    kind_lock_held: Cell<bool>,
    hot_locks: Cell<u32>,
}

struct H {
    f: FakeHost,
    k: Hooks,
}

impl Idleness for H {
    fn video_idle(&self) -> bool {
        self.f.video_idle()
    }
    fn chat_idle(&self) -> Option<String> {
        self.f.chat_idle()
    }
}

impl Host for H {
    fn now(&self) -> f64 {
        self.f.now()
    }
    fn sleep(&self, s: f64) {
        self.f.sleep(s)
    }
    fn snapshot(&self) -> Snapshot {
        self.f.snapshot()
    }
    fn server(&self, k: &str, p: u16) -> Option<Server> {
        self.f.server(k, p)
    }
    fn start(&self, k: &str, id: &str, s: usize, p: u16, g: usize) -> Result<(), Fail> {
        self.f.start(k, id, s, p, g)
    }
    fn stop(&self, k: &str, s: usize, p: u16, l: &str) -> Result<(), Fail> {
        self.f.stop(k, s, p, l)
    }
    fn free_video(&self, g: usize, i: &str) -> Result<(), Fail> {
        self.f.free_video(g, i)
    }
    fn free_chat(&self, c: &str) -> Result<(), Fail> {
        self.f.free_chat(c)
    }
    fn try_lock_kind<'a>(&'a self, k: &str) -> Option<Box<dyn Guard + 'a>> {
        self.f.try_lock_kind(k)
    }
    fn wait_loaded(&self, p: u16, i: &str) -> Result<(), Fail> {
        self.f.wait_loaded(p, i)
    }
    fn note(&self, j: Option<&str>, s: &str, w: &str) {
        self.f.note(j, s, w)
    }
    fn cancelled(&self, j: &str) -> bool {
        self.f.cancelled(j)
    }
    fn learn_ask(&self, i: &str, e: &str, g: usize, w: u64) -> bool {
        self.f.learn_ask(i, e, g, w)
    }
    fn wait_lock_kind<'a>(&'a self, _k: &str) -> Box<dyn Guard + 'a> {
        assert!(!self.k.kind_lock_held.get(), "waited for the kind's lock while another model loads");
        self.k.locks.set(self.k.locks.get() + 1);
        Box::new(())
    }
    fn placed(&self, p: u16) {
        self.k.placed.borrow_mut().push(p);
    }
    fn barred(&self) -> Option<String> {
        self.k.barred.borrow().clone()
    }
    fn barred_on(&self, g: usize) -> Option<String> {
        self.k.barred_on.borrow().get(&g).cloned()
    }
    fn lock_hot<'a>(&'a self, _k: &str) -> Box<dyn Guard + 'a> {
        self.k.hot_locks.set(self.k.hot_locks.get() + 1);
        Box::new(())
    }
    fn not_ready(&self, _i: &str) -> Option<String> {
        self.k.not_ready.borrow().clone()
    }
    fn give_up(&self) -> Option<Fail> {
        let n = self.k.tries.get() + 1;
        self.k.tries.set(n);
        self.k.give_up_after.get().filter(|a| n > *a).map(|_| (503, "past the limit".to_string()))
    }
    fn ram_plan(&self, _s: &Snapshot, _m: &ModelInfo) -> Result<Vec<Holder>, String> {
        match &*self.k.ram.borrow() {
            Some(why) => Err(why.clone()),
            None => Ok(Vec::new()),
        }
    }
    fn reserve(&self, _m: &ModelInfo, _s: usize, _p: u16, g: usize, w: u64) -> Result<(), String> {
        if self.k.reserve_refusals.get() > 0 {
            self.k.reserve_refusals.set(self.k.reserve_refusals.get() - 1);
            return Err("another load took the room".into());
        }
        self.k.reserved.borrow_mut().push((g, w));
        Ok(())
    }
    fn first_listed_card(&self) -> bool {
        self.k.first_listed.get()
    }
}

const CARD: u64 = 32768;

fn host(cards: &[u64]) -> H {
    H { f: FakeHost::new(cards, &[("image", 2)]), k: Hooks::default() }
}

fn ask(h: &H, m: &ModelInfo, job: Option<&str>, wait_s: u64) -> Result<(usize, u16), Fail> {
    let need = m.gpus.iter().map(|g| (*g, 4096)).collect::<BTreeMap<_, _>>();
    let req = PlaceReq { model: m, job, need_mb: &need, asked_mb: 0 };
    ensure(h, &req, Duration::from_secs(wait_s), None, &json!({}))
}

#[test]
fn the_driver_takes_the_lock_and_counts_the_request_before_it_goes() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    let (_, port) = ask(&h, &m, None, 90).unwrap();
    assert_eq!(*h.k.placed.borrow(), vec![port]);
    assert_eq!(h.k.locks.get(), 1);
    assert_eq!(*h.k.reserved.borrow(), vec![(0, 4096 + room::MARGIN_MB)]);
}

#[test]
fn barred_refuses_a_request_at_once_but_a_hot_model_still_answers() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    *h.k.barred.borrow_mut() = Some("cards taken: a run".into());
    let (c, e) = ask(&h, &m, None, 90).unwrap_err();
    assert_eq!((c, e.as_str()), (503, "cards taken: a run"));
    assert!(!h.f.log().iter().any(|l| l.starts_with("start")), "nothing started");
    *h.k.barred.borrow_mut() = None;
    ask(&h, &m, None, 90).unwrap();
    *h.k.barred.borrow_mut() = Some("cards taken: a run".into());
    assert!(ask(&h, &m, None, 90).is_ok(), "loaded: it answers");
}

#[test]
fn a_barred_job_waits_past_its_room_wait_and_starts_when_it_ends() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    *h.k.barred.borrow_mut() = Some("cards taken".into());
    h.k.give_up_after.set(Some(40));
    // a job is not refused; the fake's clock runs 3 s a try, past its 30 s room wait, and only the limit ends it here
    assert_eq!(ask(&h, &m, Some("j"), 30).unwrap_err().1, "past the limit");
    assert!(h.f.now() > 1100.0);
}

#[test]
fn files_moving_are_waited_for_and_refused_at_the_wait() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    *h.k.not_ready.borrow_mut() = Some("a is moving".into());
    assert_eq!(ask(&h, &m, None, 9).unwrap_err(), (409, "a is moving".to_string()));
    assert!(!h.f.log().iter().any(|l| l.starts_with("start")));
}

#[test]
fn a_limit_ends_the_drain_of_a_busy_server() {
    let h = H { f: FakeHost::new(&[CARD], &[("image", 1)]), k: Hooks::default() };
    let big = h.f.add("big", "image", 30000, &[0], Profile::default());
    let other = h.f.add("other", "image", 30000, &[0], Profile::default());
    ask(&h, &big, None, 90).unwrap();
    h.f.busy("big", 100000.0, 0);
    h.k.give_up_after.set(Some(3));
    let e = ask(&h, &other, None, 90).unwrap_err();
    assert_eq!(e.1, "past the limit");
    assert!(h.f.now() < 2000.0, "not the drain's 30 minutes: {}", h.f.now());
}

#[test]
fn no_host_memory_stops_a_load_before_anything_is_stopped() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    *h.k.ram.borrow_mut() = Some("host memory is held".into());
    let e = ask(&h, &m, None, 5).unwrap_err();
    assert!(e.1.starts_with("host memory is held"), "{e:?}");
    assert!(h.f.log().is_empty(), "{:?}", h.f.log());
}

#[test]
fn a_reservation_refused_plans_again_rather_than_failing() {
    let h = host(&[CARD]);
    let m = h.f.add("a", "image", 4096, &[0], Profile::default());
    h.k.reserve_refusals.set(2);
    assert!(ask(&h, &m, None, 90).is_ok());
    assert_eq!(h.f.log().iter().filter(|l| l.starts_with("start")).count(), 1);
}

#[test]
fn the_first_listed_card_when_asked_stopping_idle_work_there_before_going_on() {
    let (mut with, mut without) = (host(&[CARD, CARD + 1000]), host(&[CARD, CARD + 1000]));
    with.k.first_listed.set(true);
    for (h, card) in [(&mut with, 0), (&mut without, 1)] {
        let m = h.f.add("a", "image", 4096, &[0, 1], Profile::default());
        ask(h, &m, None, 90).unwrap();
        assert!(h.f.log().iter().any(|l| l.contains(&format!("gpu {card}"))), "{:?}", h.f.log());
    }
    let plan = |p: Option<Vec<Holder>>| p;
    let with_stop = room::pick_first([(0, 100, plan(Some(vec![Holder { what: room::Free::Video(0), mb: 1, model: "x".into(), order: 1 }]))), (1, 9000, plan(Some(vec![])))].into_iter());
    assert_eq!(with_stop.unwrap().0, 0, "stopping idle work on the first card beats room on a later one");
    let blocked = room::pick_first([(0, 100, plan(None)), (1, 9000, plan(Some(vec![])))].into_iter());
    assert_eq!(blocked.unwrap().0, 1, "a card that cannot be cleared is passed over");
}

fn info(id: &str, kind: &str, p: Profile) -> ModelInfo {
    ModelInfo { id: id.into(), kind: kind.into(), enabled: true, gpus: vec![0], profile: p, spans: false }
}

#[test]
fn the_last_check_sees_a_load_under_way_beside_an_exclusive_model() {
    let ex = Profile::from_card(&json!({"coexistence": {"mode": "exclusive", "exclusive_reason": "fills the card"}})).unwrap();
    let models: BTreeMap<String, ModelInfo> = [("ex".to_string(), info("ex", "audio", ex.clone())), ("a".to_string(), info("a", "image", Profile::default()))].into();
    // ex is loading: a process with no memory yet, listed by the host
    let card = |procs| vec![Card { index: 0, name: "c".into(), total_mb: CARD, used_mb: 1000, procs }];
    let loading = Proc { kind: Some("audio".into()), model: "ex".into(), port: Some(9), mb: 0 };
    let e = room::may_start(&card(vec![loading.clone()]), &models, &models["a"], 0, 4096).unwrap_err();
    assert!(e.contains("exclusive: ex"), "{e}");
    assert!(room::may_start(&card(vec![]), &models, &models["a"], 0, 4096).is_ok());
    // an exclusive model does not start beside what is loading either; a foreign process does not count
    let e = room::may_start(&card(vec![Proc { model: "a".into(), kind: Some("image".into()), port: Some(8), mb: 0 }]), &models, &models["ex"], 0, 4096).unwrap_err();
    assert!(e.contains("exclusive"), "{e}");
    assert!(room::may_start(&card(vec![Proc { model: "Xorg".into(), kind: None, port: None, mb: 500 }]), &models, &models["ex"], 0, 4096).is_ok());
    // the memory, and the copies a card may hold
    assert!(room::may_start(&card(vec![]), &models, &models["a"], 0, CARD).is_err());
    let e = room::may_start(&card(vec![Proc { model: "a".into(), kind: Some("image".into()), port: Some(8), mb: 0 }]), &models, &models["a"], 0, 4096).unwrap_err();
    assert!(e.contains("max_copies_per_card"), "{e}");
}

/// Two cards, two audio servers
fn span_host() -> H {
    H { f: FakeHost::new(&[CARD, CARD], &[("audio", 2)]), k: Hooks::default() }
}

/// The asking for `m` with its need on each card
fn ask_span(h: &H, m: &ModelInfo, per: &[(usize, u64)], wait_s: u64) -> Result<(usize, u16), Fail> {
    let need: BTreeMap<usize, u64> = per.iter().copied().collect();
    let req = PlaceReq { model: m, job: None, need_mb: &need, asked_mb: 0 };
    ensure(h, &req, Duration::from_secs(wait_s), None, &json!({}))
}

/// Sophon-Realtime's shares: the brain's card, the voice's
const SOPHON: [(usize, u64); 2] = [(0, 23552), (1, 5632)];

#[test]
fn a_spanning_model_frees_idle_holders_on_each_of_its_cards_and_starts_once_on_both() {
    let h = span_host();
    let b = h.f.add("b", "audio", 20000, &[0], Profile::default());
    let c = h.f.add("c", "audio", 28000, &[1], Profile::default());
    ask_span(&h, &b, &[(0, 20000)], 90).unwrap();
    ask_span(&h, &c, &[(1, 28000)], 90).unwrap();
    h.f.chat("chat", 4000, 0, true);
    let s = h.f.add_span("sophon", "audio", &SOPHON, Profile::default());
    ask_span(&h, &s, &SOPHON, 90).unwrap();
    let log = h.f.log();
    // b held card 0's room and c card 1's: both idle, both go; one start on both cards
    assert!(log.iter().any(|l| l.starts_with("stop audio slot 0")) && log.iter().any(|l| l.starts_with("stop audio slot 1")), "{log:?}");
    assert_eq!(log.iter().filter(|l| l.starts_with("start audio sophon")).collect::<Vec<_>>(), vec!["start audio sophon slot 0 gpus 0,1"]);
    // the room on each card taken before the start, at its own figure with the margin
    let r = h.k.reserved.borrow().clone();
    assert!(r.contains(&(0, 23552 + room::MARGIN_MB)) && r.contains(&(1, 5632 + room::MARGIN_MB)), "{r:?}");
    // its memory shows on both cards; the idle chat model was not needed (card 0 had the room once b went)
    let snap = h.snapshot();
    for (g, mb) in SOPHON {
        assert!(snap.cards[g].procs.iter().any(|p| p.model == "sophon" && p.mb == mb), "card {g}: {:?}", snap.cards[g].procs);
    }
    assert!(!log.iter().any(|l| l.contains("unload chat")), "{log:?}");
}

#[test]
fn a_spanning_model_waits_when_one_card_is_busy_and_stops_nothing_meanwhile() {
    let h = span_host();
    let b = h.f.add("b", "audio", 20000, &[0], Profile::default());
    ask_span(&h, &b, &[(0, 20000)], 90).unwrap();
    // card 1 held by a chat model answering a request: not idle
    h.f.chat("chat", 28000, 1, false);
    let s = h.f.add_span("sophon", "audio", &SOPHON, Profile::default());
    let (code, why) = ask_span(&h, &s, &SOPHON, 90).unwrap_err();
    assert_eq!(code, 503, "{why}");
    assert!(why.contains("GPU 1") && why.contains("chat"), "{why}");
    // the idle model on card 0 stays until both cards can be had
    assert_eq!(h.f.loaded("audio"), vec![Some("b".to_string()), None]);
    assert!(!h.f.log().iter().any(|l| l.contains("sophon")), "{:?}", h.f.log());
    // the chat model idles: it goes, b goes, sophon starts
    h.f.chat("chat", 28000, 1, true);
    ask_span(&h, &s, &SOPHON, 90).unwrap();
    assert!(h.f.log().iter().any(|l| l == "unload chat chat"), "{:?}", h.f.log());
    assert_eq!(h.f.count("start audio sophon slot"), 1);
}

#[test]
fn a_spanning_model_never_runs_a_second_copy() {
    let h = span_host();
    let two = Profile::from_card(&json!({"concurrency": {"max_copies_per_card": 2, "recommended_copies": 2}})).unwrap();
    let s = h.f.add_span("sophon", "audio", &SOPHON, two);
    let first = ask_span(&h, &s, &SOPHON, 90).unwrap();
    h.f.busy("sophon", 600.0, 1);
    // its server busy, a free slot and the room for another on both cards: the request shares the copy it has
    let again = ask_span(&h, &s, &SOPHON, 90).unwrap();
    assert_eq!(first, again);
    assert_eq!(h.f.count("start audio sophon"), 1, "{:?}", h.f.log());
}

#[test]
fn a_model_up_and_idle_answers_at_once_while_another_model_of_its_kind_loads() {
    // zonos3 idle on its audio server while Sophon loads on the other (that load holds the kind's lock for minutes): a
    // zonos3 request takes its idle server without waiting for the lock (2026-11-16: two jobs waited a minute)
    let h = span_host();
    let z = h.f.add("zonos3", "audio", 6000, &[0, 1], Profile::default());
    let first = ask_span(&h, &z, &[(0, 6000), (1, 6000)], 90).unwrap();
    assert_eq!(h.k.locks.get(), 1, "its first load took the kind's lock");
    h.k.kind_lock_held.set(true);
    let again = ask_span(&h, &z, &[(0, 6000), (1, 6000)], 90).unwrap();
    assert_eq!(first, again);
    assert_eq!(*h.k.placed.borrow(), vec![first.1, first.1], "counted on its server each time");
    assert!(h.k.hot_locks.get() >= 2, "counted under the short lock");
    assert_eq!(h.f.count("start audio zonos3"), 1);
}

#[test]
fn a_busy_model_still_goes_through_the_kinds_lock() {
    let h = span_host();
    let z = h.f.add("zonos3", "audio", 6000, &[0], Profile::default());
    ask_span(&h, &z, &[(0, 6000)], 90).unwrap();
    h.f.busy("zonos3", 600.0, 1);
    ask_span(&h, &z, &[(0, 6000)], 90).unwrap();
    assert_eq!(h.k.locks.get(), 2, "not idle: placed under the kind's lock, as before");
}

#[test]
fn a_gpu_run_on_one_card_leaves_the_other_serving() {
    let h = span_host();
    h.k.barred_on.borrow_mut().insert(1, "cards taken: a run on GPU 1".into());
    // a model that may go on either card loads on the free one
    let a = h.f.add("a", "audio", 6000, &[1, 0], Profile::default());
    ask_span(&h, &a, &[(0, 6000), (1, 6000)], 90).unwrap();
    assert!(h.f.log().iter().any(|l| l.starts_with("start audio a") && l.ends_with("gpu 0")), "{:?}", h.f.log());
    // one only on the taken card: refused (a job would wait)
    let b = h.f.add("b", "audio", 6000, &[1], Profile::default());
    assert_eq!(ask_span(&h, &b, &[(1, 6000)], 90).unwrap_err(), (503, "cards taken: a run on GPU 1".to_string()));
    // one whose one copy spans both cards: refused, nothing stopped or started for it
    let s = h.f.add_span("sophon", "audio", &SOPHON, Profile::default());
    assert_eq!(ask_span(&h, &s, &SOPHON, 90).unwrap_err().0, 503);
    assert!(!h.f.log().iter().any(|l| l.contains("sophon") || l.starts_with("stop")), "{:?}", h.f.log());
    // no second copy of `a` on the taken card while its first is busy: the requests share the first
    h.f.busy("a", 600.0, 1);
    ask_span(&h, &a, &[(0, 6000), (1, 6000)], 90).unwrap();
    assert_eq!(h.f.count("start audio a"), 1, "{:?}", h.f.log());
    // the run ends: the spanning model loads (a, idle again, goes for it)
    h.k.barred_on.borrow_mut().clear();
    h.f.busy("a", 0.0, 0);
    ask_span(&h, &s, &SOPHON, 90).unwrap();
    assert_eq!(h.f.count("start audio sophon"), 1);
}
