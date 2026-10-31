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
        self.k.locks.set(self.k.locks.get() + 1);
        Box::new(())
    }
    fn placed(&self, p: u16) {
        self.k.placed.borrow_mut().push(p);
    }
    fn barred(&self) -> Option<String> {
        self.k.barred.borrow().clone()
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
fn the_first_listed_card_with_room_now_when_asked_and_no_stopping_for_a_listed_one() {
    let (mut with, mut without) = (host(&[CARD, CARD + 1000]), host(&[CARD, CARD + 1000]));
    with.k.first_listed.set(true);
    for (h, card) in [(&mut with, 0), (&mut without, 1)] {
        let m = h.f.add("a", "image", 4096, &[0, 1], Profile::default());
        ask(h, &m, None, 90).unwrap();
        assert!(h.f.log().iter().any(|l| l.contains(&format!("gpu {card}"))), "{:?}", h.f.log());
    }
    let plan = |p: Option<Vec<Holder>>| p;
    let with_stop = room::pick_first([(0, 100, plan(Some(vec![Holder { what: room::Free::Video(0), mb: 1, model: "x".into(), order: 1 }]))), (1, 9000, plan(Some(vec![])))].into_iter());
    assert_eq!(with_stop.unwrap().0, 1, "room now beats stopping something on an earlier card");
}

fn info(id: &str, kind: &str, p: Profile) -> ModelInfo {
    ModelInfo { id: id.into(), kind: kind.into(), enabled: true, gpus: vec![0], profile: p }
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
