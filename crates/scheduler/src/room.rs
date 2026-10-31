//! The snapshot types and room planning: what a card holds, which holders can go for a model that needs the room, who may
//! share a card with whom. Pure functions over a [`Snapshot`] the host builds from its sampler and its servers.

use std::collections::{BTreeMap, BTreeSet};

use crate::profile::{Mode, Profile};

/// Memory to keep free on a card beyond a model's (a driver with no out-of-memory spills past the card)
pub const MARGIN_MB: u64 = 1536;

/// How long a request waits for room before 503 no_room; a job waits longer
pub const ROOM_WAIT_S: u64 = 90;
pub const JOB_ROOM_WAIT_S: u64 = 1800;

/// A process holding memory on a card. `kind` is None for a foreign process (a desktop, a notebook): never a resident
/// this scheduler plans for or against.
#[derive(Clone, Debug, PartialEq)]
pub struct Proc {
    pub kind: Option<String>,
    /// the model's id (or the program's name for a foreign process)
    pub model: String,
    /// the port of its server (picture, sound and 3D servers)
    pub port: Option<u16>,
    pub mb: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub index: usize,
    pub name: String,
    pub total_mb: u64,
    pub used_mb: u64,
    pub procs: Vec<Proc>,
}

impl Card {
    pub fn free(&self) -> u64 {
        self.total_mb.saturating_sub(self.used_mb)
    }
}

/// One of a kind's servers (a slot, a port): what it holds and whether it has work
#[derive(Clone, Debug, PartialEq)]
pub struct Server {
    pub slot: usize,
    pub port: u16,
    pub model: Option<String>,
    pub loading: bool,
    /// answering a request of its own, or one sent from here (inflight)
    pub busy: bool,
    /// requests sent on from here and not answered
    pub inflight: usize,
}

impl Server {
    pub fn idle(&self) -> bool {
        !self.busy && !self.loading && self.inflight == 0
    }
}

/// A registered model as the scheduler needs it
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub kind: String,
    pub enabled: bool,
    /// the cards it may use
    pub gpus: Vec<usize>,
    pub profile: Profile,
}

/// One consistent view of the box
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub cards: Vec<Card>,
    /// by kind
    pub servers: BTreeMap<String, Vec<Server>>,
    pub models: BTreeMap<String, ModelInfo>,
    pub pinned: BTreeSet<String>,
    /// pinned or held (an rsvp_s hold not run out)
    pub kept: BTreeSet<String>,
    /// seconds each held model has left
    pub held_left: BTreeMap<String, f64>,
}

impl Snapshot {
    pub fn servers_of(&self, kind: &str) -> &[Server] {
        self.servers.get(kind).map_or(&[], Vec::as_slice)
    }
}

/// Idle work that holds memory on a card and can go for a model that needs the room
#[derive(Clone, Debug)]
pub struct Holder {
    pub what: Free,
    /// its memory there
    pub mb: u64,
    pub model: String,
    /// which go first: picture and sound servers (0), the video worker (1), a chat model (2)
    pub order: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Free {
    /// an image, audio or 3D server: (kind, slot)
    Server(String, usize),
    /// the video daemon's worker on this card
    Video(usize),
    /// the chat model
    Chat(String),
}

/// What the host knows that a snapshot does not carry cheaply
pub trait Idleness {
    /// Whether the video studio has nothing to make
    fn video_idle(&self) -> bool;
    /// The chat model loaded, when it may be preempted
    fn chat_idle(&self) -> Option<String>;
}

pub fn card_free(cards: &[Card], g: usize) -> u64 {
    cards.iter().find(|c| c.index == g).map_or(0, Card::free)
}

/// The card the server listening on `port` is on
pub fn card_of_port(cards: &[Card], port: u16) -> Option<usize> {
    cards.iter().find(|c| c.procs.iter().any(|p| p.port == Some(port))).map(|c| c.index)
}

/// The holders a room plan may stop: not a kept model's
pub fn evictable(holders: Vec<Holder>, kept: &BTreeSet<String>) -> Vec<Holder> {
    holders.into_iter().filter(|h| !kept.contains(&h.model)).collect()
}

/// Whether card `g` may take another copy of a model that has copies on the cards `on` (one entry a copy), up to `per_card` there
pub fn extra_copy_allowed(on: &[usize], g: usize, per_card: usize) -> bool {
    on.iter().filter(|c| **c == g).count() < per_card
}

/// What of a card's idle holders must go so `want` MiB is free (it has `free`): none (Some(empty)), the lightest to
/// bring back first - picture and sound servers, then the video worker, then a chat model - or no way (None)
pub fn plan_room(free: u64, want: u64, mut idle: Vec<Holder>) -> Option<Vec<Holder>> {
    idle.sort_by_key(|h| (h.order, std::cmp::Reverse(h.mb)));
    let mut got = free;
    let mut plan = Vec::new();
    for h in idle {
        if got >= want {
            break;
        }
        got += h.mb;
        plan.push(h);
    }
    (got >= want).then_some(plan)
}

/// `plan_room` with the residents that may not stay: each of `residents` that `blocks` (all of them when `exclusive`)
/// goes with the idle holders, and when one is not among the idle holders (busy, pinned, held) there is no plan
pub fn plan_with_blockers(free: u64, want: u64, idle: Vec<Holder>, residents: &[String], exclusive: bool, blocks: &dyn Fn(&str) -> bool) -> Option<Vec<Holder>> {
    let must = |r: &str| exclusive || blocks(r);
    let blockers: Vec<&String> = residents.iter().filter(|r| must(r)).collect();
    if blockers.is_empty() {
        return plan_room(free, want, idle);
    }
    let (mut forced, idle): (Vec<Holder>, Vec<Holder>) = idle.into_iter().partition(|h| must(&h.model));
    // every blocker is a holder that can go
    let mut gone: Vec<&str> = forced.iter().map(|h| h.model.as_str()).collect();
    for b in &blockers {
        let i = gone.iter().position(|g| g == &b.as_str())?;
        gone.remove(i);
    }
    let freed: u64 = forced.iter().map(|h| h.mb).sum();
    let mut plan = plan_room(free + freed, want, idle)?;
    forced.append(&mut plan);
    Some(forced)
}

/// The card to load on, of (card, its free MiB, its plan) in the model's order: one with the room now - the one with the
/// most free, so an estimate on the low side does not land the model beside another on a full card - before the first
/// whose room needs idle work stopped
pub fn pick_card(plans: impl Iterator<Item = (usize, u64, Option<Vec<Holder>>)>) -> Option<(usize, Vec<Holder>)> {
    let mut later = None;
    let mut now: Option<(usize, u64, Vec<Holder>)> = None;
    for (g, free, plan) in plans {
        match plan {
            Some(p) if p.is_empty() => {
                if now.as_ref().is_none_or(|n| free > n.1) {
                    now = Some((g, free, p));
                }
            }
            Some(p) => {
                later.get_or_insert((g, p));
            }
            None => {}
        }
    }
    now.map(|(g, _, p)| (g, p)).or(later)
}

/// `pick_card`, first listed: the first card with the room now, before the first whose room needs idle work stopped
pub fn pick_first(plans: impl Iterator<Item = (usize, u64, Option<Vec<Holder>>)>) -> Option<(usize, Vec<Holder>)> {
    let mut later = None;
    for (g, _, plan) in plans {
        match plan {
            Some(p) if p.is_empty() => return Some((g, p)),
            Some(p) => {
                later.get_or_insert((g, p));
            }
            None => {}
        }
    }
    later
}

/// A test for another model's id: may it NOT share a card with the asking model
pub type Blocks<'a> = Box<dyn Fn(&str) -> bool + 'a>;

/// Whether model `m` is exclusive, and a test for another model's id: may it NOT share a card with `m`
pub fn sharing<'a>(snap: &'a Snapshot, m: &ModelInfo) -> (bool, Blocks<'a>) {
    let (id, kind, me) = (m.id.clone(), m.kind.clone(), m.profile.clone());
    let exclusive = me.coexistence.mode == Mode::Exclusive;
    (exclusive, Box::new(move |other: &str| other != id && snap.models.get(other).is_some_and(|o| !me.may_share(&id, &kind, &o.profile, other, &o.kind))))
}

/// What could go from card `g` so `want` MiB is free there: none needed (Some(empty)), the idle holders to stop (Some),
/// or no way now (None). Idle: a picture, sound or 3D server without work, the video worker while the studio has nothing
/// to make, a chat model answering nothing and not kept. `light`: servers only (for a second copy, which is no reason to
/// stop a video or chat model). The kind asking is never asked to free its own kind's chat or video. With a model `m`
/// asking, residents that may not share a card with it go too (an exclusive model's, its own exclusivity, `shared_with`)
/// and make the card no choice when they cannot go (busy, pinned, held).
pub fn room_plan(snap: &Snapshot, idle_of: &dyn Idleness, g: usize, want: u64, asking: &str, light: bool, m: Option<&ModelInfo>) -> Option<Vec<Holder>> {
    let c = snap.cards.iter().find(|c| c.index == g)?;
    let free = c.free();
    let (exclusive, blocks) = match m {
        Some(m) => sharing(snap, m),
        None => (false, Box::new(|_: &str| false) as Blocks<'_>),
    };
    // the nextcuda models on the card (a foreign process is not one)
    let residents: Vec<String> = c.procs.iter().filter(|p| p.kind.is_some()).map(|p| p.model.clone()).collect();
    if free >= want && !residents.iter().any(|r| exclusive || blocks(r)) {
        return Some(Vec::new());
    }
    let idle = idle_on(snap, idle_of, g, asking, light);
    plan_with_blockers(free, want, evictable(idle, &snap.kept), &residents, exclusive, &*blocks)
}

/// The idle holders on card `g`, pinned and held ones among them (see `evictable`)
pub fn idle_on(snap: &Snapshot, idle_of: &dyn Idleness, g: usize, asking: &str, light: bool) -> Vec<Holder> {
    let Some(c) = snap.cards.iter().find(|c| c.index == g) else { return Vec::new() };
    let mut idle: Vec<Holder> = Vec::new();
    let mut video_checked = None;
    for p in &c.procs {
        let (mb, model) = (p.mb, p.model.clone());
        match p.kind.as_deref() {
            Some(k @ ("image" | "audio" | "model3d")) => {
                let Some(port) = p.port else { continue };
                let Some(x) = snap.servers_of(k).iter().find(|x| x.port == port) else { continue };
                if x.idle() {
                    idle.push(Holder { what: Free::Server(k.to_string(), x.slot), mb, model, order: 0 });
                }
            }
            Some("video") if !light && asking != "video" => {
                if *video_checked.get_or_insert_with(|| idle_of.video_idle()) {
                    idle.push(Holder { what: Free::Video(g), mb, model, order: 1 });
                }
            }
            Some("llm") if !light && asking != "llm" => {
                if let Some(id) = idle_of.chat_idle() {
                    idle.push(Holder { what: Free::Chat(id), mb, model, order: 2 });
                }
            }
            _ => {}
        }
    }
    idle
}

/// The last check before a start, on what the cards hold now (loads under way among them): the memory is there, nothing
/// there may not share with the model, it has no more copies there than it may. Err says why.
pub fn may_start(cards: &[Card], models: &BTreeMap<String, ModelInfo>, m: &ModelInfo, g: usize, want: u64) -> Result<(), String> {
    let c = cards.iter().find(|c| c.index == g).ok_or_else(|| format!("no GPU {g}"))?;
    if c.free() < want {
        return Err(format!("GPU {g} has {:.1} GiB free, {} needs ~{:.1} GiB", c.free() as f64 / 1024.0, m.id, want as f64 / 1024.0));
    }
    let copies = c.procs.iter().filter(|p| p.kind.is_some() && p.model == m.id).count();
    if copies >= m.profile.concurrency.max_copies_per_card as usize {
        return Err(format!("{} already has {copies} on GPU {g}: its max_copies_per_card is {}", m.id, m.profile.concurrency.max_copies_per_card));
    }
    for p in c.procs.iter().filter(|p| p.kind.is_some() && p.model != m.id) {
        if m.profile.coexistence.mode == Mode::Exclusive {
            return Err(format!("{} is exclusive and {} is on GPU {g}", m.id, p.model));
        }
        if let Some(o) = models.get(&p.model) {
            if o.profile.coexistence.mode == Mode::Exclusive {
                return Err(format!("exclusive: {} (runs alone on its card)", o.id));
            }
            if !m.profile.may_share(&m.id, &m.kind, &o.profile, &o.id, &o.kind) {
                return Err(format!("{} may not share GPU {g} with {} (shared_with)", m.id, o.id));
            }
        }
    }
    Ok(())
}

/// The nextcuda models holding memory on the named cards that are in `of`
pub fn holders_in(cards: &[Card], gpus: &[usize], of: &BTreeSet<String>) -> Vec<String> {
    let mut v: Vec<String> = gpus.iter().filter_map(|g| cards.iter().find(|c| c.index == *g)).flat_map(|c| c.procs.iter())
        .filter(|p| of.contains(&p.model)).map(|p| p.model.clone()).collect();
    v.sort();
    v.dedup();
    v
}

/// What a no-room message adds about the models on these cards that will not go, and an exclusive model asking
pub fn kept_words(snap: &Snapshot, gpus: &[usize], m: Option<&ModelInfo>) -> String {
    let pinned = holders_in(&snap.cards, gpus, &snap.pinned);
    let held = holders_in(&snap.cards, gpus, &snap.kept.difference(&snap.pinned).cloned().collect());
    let exclusive: BTreeSet<String> = snap.models.values().filter(|o| o.profile.coexistence.mode == Mode::Exclusive).map(|o| o.id.clone()).collect();
    let excl = holders_in(&snap.cards, gpus, &exclusive);
    let mut pin = if pinned.is_empty() { String::new() } else { format!("; pinned: {} (unpin to free the room)", pinned.join(", ")) };
    if !excl.is_empty() {
        pin += &format!("; exclusive: {} (runs alone on its card)", excl.join(", "));
    }
    if let Some(m) = m.filter(|m| m.profile.coexistence.mode == Mode::Exclusive) {
        pin += &format!("; {} is exclusive: it needs its card to itself", m.id);
    }
    if !held.is_empty() {
        pin += &format!("; held by rsvp_s: {} (until its hold ends: {})", held.join(", "),
                        held.iter().map(|h| format!("{h} {:.0} s", snap.held_left.get(h).copied().unwrap_or(0.0))).collect::<Vec<_>>().join(", "));
    }
    pin
}

/// The message for no room: each card, what it has free and who holds the rest (busy or idle), and which of the holders
/// is a pin, a hold or exclusive, or whether the asking model is exclusive itself
pub fn no_room(snap: &Snapshot, gpus: &[usize], m: &ModelInfo, want: u64) -> String {
    let free = |g: usize| snap.cards.iter().find(|c| c.index == g).map_or(0, Card::free);
    let busy: Vec<String> = gpus.iter().map(|g| {
        let holders: Vec<String> = snap.cards.iter().find(|c| c.index == *g).into_iter().flat_map(|c| c.procs.iter())
            .map(|p| format!("{} {:.1} GiB", p.model, p.mb as f64 / 1024.0)).collect();
        format!("GPU {g}: {:.1} GiB free ({})", free(*g) as f64 / 1024.0, holders.join(", "))
    }).collect();
    let pin = kept_words(snap, gpus, Some(m));
    format!("{} held by work that is not idle - {}; {} needs ~{:.1} GiB{pin}", if gpus.len() > 1 { "its cards are" } else { "its card is" },
            busy.join("; "), m.id, want as f64 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(model: &str, mb: u64, order: u8) -> Holder {
        Holder { what: Free::Server("audio".into(), 0), mb, model: model.into(), order }
    }

    fn cards() -> Vec<Card> {
        vec![Card { index: 0, name: "c0".into(), total_mb: 12227, used_mb: 6134, procs: vec![] }, Card { index: 1, name: "c1".into(), total_mb: 12288, used_mb: 157, procs: vec![] }]
    }

    fn plans(cards: &[Card], gpus: &[usize], want: u64) -> Option<(usize, Vec<Holder>)> {
        pick_card(gpus.iter().map(|&g| (g, card_free(cards, g), plan_room(card_free(cards, g), want, if g == 0 { vec![h("breeze-tts-2", 6040, 0)] } else { vec![] }))))
    }

    #[test]
    fn a_model_goes_to_the_card_with_room_not_beside_another() {
        let c = cards();
        let (g, plan) = plans(&c, &[0, 1], 3914 + MARGIN_MB).unwrap();
        assert_eq!((g, plan.len()), (1, 0));
        let (g, plan) = plans(&c, &[0, 1], 7000).unwrap();
        assert_eq!((g, plan.len()), (1, 0));
        // only card 0: the idle server goes
        let (g, plan) = plans(&c, &[0], 7000).unwrap();
        assert_eq!((g, plan.len()), (0, 1));
    }

    #[test]
    fn a_pinned_or_held_model_is_not_an_evictable_holder() {
        let kept: BTreeSet<String> = ["breeze-tts-2".to_string()].into();
        let left = evictable(vec![h("breeze-tts-2", 6040, 0), h("other", 100, 0)], &kept);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].model, "other");
        assert!(plan_room(1500, 7000, evictable(vec![h("breeze-tts-2", 6040, 0)], &kept)).is_none());
        assert_eq!(plan_room(1500, 7000, evictable(vec![h("breeze-tts-2", 6040, 0)], &BTreeSet::new())).map(|p| p.len()), Some(1));
    }

    #[test]
    fn the_lightest_to_bring_back_go_first_and_only_as_many_as_needed() {
        let plan = plan_room(1000, 5000, vec![h("chat", 9000, 2), h("a", 3000, 0), h("b", 2500, 0), h("video", 4000, 1)]).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert!(plan_room(100, 20000, vec![h("a", 3000, 0)]).is_none());
        assert_eq!(plan_room(9000, 5000, vec![h("a", 3000, 0)]).unwrap().len(), 0);
    }

    #[test]
    fn an_exclusive_model_clears_its_card_of_idle_residents_and_waits_for_the_rest() {
        let res = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let none = |_: &str| false;
        let plan = plan_with_blockers(9000, 2000, vec![h("a", 3000, 0)], &res(&["a"]), true, &none).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
        assert!(plan_with_blockers(9000, 2000, vec![h("a", 3000, 0)], &res(&["a", "busy-one"]), true, &none).is_none());
        assert_eq!(plan_with_blockers(9000, 2000, vec![h("a", 3000, 0)], &res(&["a"]), false, &none).unwrap().len(), 0);
        let blocks = |m: &str| m == "a";
        let plan = plan_with_blockers(9000, 2000, vec![h("a", 3000, 0), h("b", 3000, 0)], &res(&["a", "b"]), false, &blocks).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
        let plan = plan_with_blockers(1000, 5000, vec![h("a", 4500, 0), h("b", 3000, 0)], &res(&["a", "b"]), false, &blocks).unwrap();
        assert_eq!(plan.iter().map(|x| x.model.as_str()).collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn extra_copies_stop_at_the_profiles_count_a_card() {
        assert!(!extra_copy_allowed(&[0], 0, 1));
        assert!(extra_copy_allowed(&[0], 1, 1));
        assert!(extra_copy_allowed(&[0], 0, 2));
        assert!(!extra_copy_allowed(&[0, 0], 0, 2));
        assert!(extra_copy_allowed(&[0, 0], 1, 2));
    }
}
