//! `place`: one try at a server for a model, over a [`Host`]. The caller holds the kind's lock while it lasts and loops
//! (sleeping, noting the wait) until it has a server or its wait runs out.

use std::collections::BTreeMap;

use crate::host::Host;
use crate::room::{card_free, card_of_port, extra_copy_allowed, no_room, pick_card, pick_first, room_plan, Card, Free, Holder, ModelInfo, Server, Snapshot,
                  JOB_ROOM_WAIT_S, MARGIN_MB};
use crate::Fail;

/// What to place
pub struct PlaceReq<'a> {
    pub model: &'a ModelInfo,
    /// the job asking (its stage is noted, a cancel ends a wait)
    pub job: Option<&'a str>,
    /// the memory the model is vouched for on each card, MiB
    pub need_mb: &'a BTreeMap<usize, u64>,
    /// MiB a server once refused for (learned): at least this is planned for
    pub asked_mb: u64,
    /// MiB the server of a model whose one copy spans its cards once refused for on one of its cards past the first
    /// (learned, `Host::learn_ask` with `rest`): at least this is planned for on each of them
    pub asked_rest_mb: u64,
}

/// The answer of one try
#[derive(Debug, PartialEq)]
pub enum Outcome {
    /// a server of the model to send the request to (the host leases its port)
    Placed { slot: usize, port: u16 },
    /// not yet: why (no card of its has the room, even with the idle models there stopped; a server stopped for it ...)
    NotYet(String),
    /// not loaded, and nothing may load now ([`Host::barred`])
    Barred(String),
}

/// Wait (30 s at most) until card `g` has `want` MiB free: a stopped process lets go of its memory in a moment. Whether
/// it came.
pub fn wait_room(host: &dyn Host, g: usize, want: u64) -> bool {
    let t0 = host.now();
    while host.now() - t0 < 30.0 {
        if host.free_mb(g) >= want {
            return true;
        }
        host.sleep(1.0);
    }
    false
}

/// Stop the idle holders of a plan (for `id`; the kind `kind` asking holds its own lock; another kind's is tried, and a
/// server whose lock is taken is left: the next try plans again)
pub fn free_plan(host: &dyn Host, plan: &[Holder], kind: &str, id: &str) -> Result<(), Fail> {
    for h in plan {
        match &h.what {
            Free::Server(k, slot) => {
                let guard = if k == kind { None } else {
                    match host.try_lock_kind(k) {
                        Some(g) => Some(g),
                        None => return Err((503, format!("{} is changing its {k} server; {id} tries again", h.model))),
                    }
                };
                // still idle (it may have taken a request since)
                let snap = host.snapshot();
                let x = snap.servers_of(k).iter().find(|x| x.slot == *slot).cloned();
                if let Some(x) = x.filter(|x| !x.busy && !x.loading && x.model.is_some()) {
                    host.stop(k, *slot, x.port, &format!("{k} stop ({}, idle, for {id})", h.model))?;
                }
                drop(guard);
            }
            Free::Video(g) => {
                if host.video_idle() {
                    host.free_video(*g, id)?;
                }
            }
            Free::Chat(chat) => {
                if host.chat_idle().as_deref() == Some(chat.as_str()) {
                    host.free_chat(chat)?;
                }
            }
        }
    }
    Ok(())
}

/// Stop the server `x` when it holds a model (one of the kind, its lock held), for `id`
fn stop_slot(host: &dyn Host, kind: &str, x: &Server, id: &str) -> Result<(), Fail> {
    if let Some(other) = host.server(kind, x.port).and_then(|s| s.model) {
        host.stop(kind, x.slot, x.port, &format!("{kind} stop ({other}, for {id})"))?;
    }
    Ok(())
}

/// Server `x` (another model's) after its requests: waited for, with the kind's lock held, so no new request is sent to
/// it meanwhile; then stopped
fn drain(host: &dyn Host, x: &Server, kind: &str, id: &str, job: Option<&str>) -> Result<(), Fail> {
    let Some(other) = &x.model else { return Ok(()) };
    let t0 = host.now();
    while host.server(kind, x.port).is_some_and(|s| s.busy || s.inflight > 0) {
        if job.is_some_and(|j| host.cancelled(j)) {
            return Err((499, format!("cancelled while it waited for {other}'s requests")));
        }
        if let Some(f) = host.give_up() {
            return Err(f);
        }
        if host.now() - t0 > JOB_ROOM_WAIT_S as f64 {
            return Err((503, format!("{other} has been busy for 30 minutes; {id} waits for it")));
        }
        host.note(job, "waiting", &format!("server {} finishing {other}'s requests, then {id} loads there", x.slot + 1));
        host.sleep(2.0);
    }
    stop_slot(host, kind, x, id)
}

/// Whether card `g`'s compute is taken by other work: a busy server of another model there (any kind's: a FlashVSR
/// upscale, a picture), or the video worker making something while it holds memory there. A server of the model
/// asking (`port`) does not count. Memory says nothing of this: an idle copy on a saturated card answers slowly (a
/// song beside a film upscale made no progress for 15 minutes on hardac1, 2026-11-13, while the other card sat idle)
pub fn card_contended(snap: &Snapshot, host: &dyn Host, g: usize, port: u16) -> bool {
    let Some(c) = snap.cards.iter().find(|c| c.index == g) else { return false };
    c.procs.iter().any(|p| match (p.kind.as_deref(), p.port) {
        (Some("video"), _) => !host.video_idle(),
        (Some(k), Some(pp)) if pp != port => snap.servers_of(k).iter().any(|x| x.port == pp && (x.busy || x.inflight > 0)),
        _ => false,
    })
}

/// Another copy of the model on one of `cards` (of its own, where it may have one more) that has the room now, or once
/// idle picture and sound servers there stop; in a slot no server holds, or one an idle model holds. Placed when it
/// started; None when there is no such card or slot, or the start failed (the caller uses the copy it has)
fn start_copy(host: &dyn Host, req: &PlaceReq, snap: &Snapshot, mine: &[&Server], cards: &dyn Fn(usize) -> bool) -> Option<Outcome> {
    let m = req.model;
    // one copy over its cards: never a second (its requests share it)
    if m.spans {
        return None;
    }
    let (id, kind, job) = (m.id.as_str(), m.kind.as_str(), req.job);
    let sv = snap.servers_of(kind);
    let kept = |x: &Server| x.model.as_ref().is_some_and(|mm| snap.kept.contains(mm));
    let idle_other = |x: &&Server| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x) && x.idle();
    let want = |g: usize| (req.need_mb.get(&g).copied().unwrap_or(0) + MARGIN_MB).max(req.asked_mb);
    let on: Vec<usize> = mine.iter().filter_map(|x| card_of_port(&snap.cards, x.port)).collect();
    // copies per card: extra ones up to the profile's recommended count (the default 1: one a card), never past its max
    let per_card = m.profile.concurrency.recommended_copies.min(m.profile.concurrency.max_copies_per_card) as usize;
    let spare = sv.iter().find(|x| x.model.is_none()).or_else(|| sv.iter().find(idle_other))?;
    // (no second copy while nothing may load, on a card taken, or when its host memory is not there: the requests share the first)
    let copy_plans = m.gpus.iter().copied().filter(|g| cards(*g) && extra_copy_allowed(&on, *g, per_card)).filter(|_| host.barred().is_none())
        .filter(|g| host.barred_on(*g).is_none())
        .map(|g| (g, card_free(&snap.cards, g), room_plan(snap, host, g, want(g), kind, true, Some(m))));
    let (gpu, plan) = if host.first_listed_card() { pick_first(copy_plans) } else { pick_card(copy_plans) }?;
    let ram = host.ram_plan(snap, m).ok()?;
    let x = spare;
    // (a plan another request's lock stands in the way of: no copy now, the requests share the first)
    let started = free_plan(host, &plan, kind, id).and_then(|_| free_plan(host, &ram, kind, id)).and_then(|_| {
        stop_slot(host, kind, x, id)?;
        if !host.wait_ram(id) {
            return Err((503, format!("host memory did not come free for {id}")));
        }
        if !wait_room(host, gpu, want(gpu)) {
            return Err((503, format!("GPU {gpu} did not free the room")));
        }
        host.reserve(m, x.slot, x.port, gpu, want(gpu)).map_err(|e| (503, e))?;
        host.note(job, "loading", &format!("a second copy of {id} loading on GPU {gpu} (server {})", x.slot + 1));
        host.start(kind, id, x.slot, x.port, gpu).inspect_err(|(_, e)| {
            short_of_room(host, &snap.cards, m, gpu, e, &[(gpu, want(gpu))]);
        })
    });
    match started {
        Ok(()) => Some(Outcome::Placed { slot: x.slot, port: x.port }),
        Err((_, e)) => {
            eprintln!("{id}: a second copy did not start ({e}); the requests share the first");
            None
        }
    }
}

/// One try at a server for the model (the kind's lock held while it lasts): a server, or why there is none yet (no card
/// of its has the room, even with the idle models there stopped)
pub fn place(host: &dyn Host, req: &PlaceReq) -> Result<Outcome, Fail> {
    let m = req.model;
    let (id, kind, job) = (m.id.as_str(), m.kind.as_str(), req.job);
    if !m.enabled {
        return Err((409, format!("{id} is disabled (enable it)")));
    }
    let gpus = &m.gpus;
    let snap = host.snapshot();
    let sv: Vec<Server> = snap.servers_of(kind).to_vec();
    let kept = |x: &Server| x.model.as_ref().is_some_and(|mm| snap.kept.contains(mm));
    let mine: Vec<&Server> = sv.iter().filter(|x| x.model.as_deref() == Some(id)).collect();
    // a server of it with nothing to do - one on a card no other work saturates first; when every such server's card is
    // taken by other work (a busy server of another model there, the video worker), a copy on a card of its that is not,
    // where one has the room now or once idle picture and sound servers there stop: two heavy jobs do not share one
    // card's compute while another card of the model sits idle
    let contended = |x: &Server| !x.loading && card_of_port(&snap.cards, x.port).is_some_and(|g| card_contended(&snap, host, g, x.port));
    let free_now: Vec<&&Server> = mine.iter().filter(|x| !x.busy && x.inflight == 0).collect();
    if let Some(x) = free_now.iter().find(|x| !contended(x)).or_else(|| free_now.first()).copied() {
        if contended(x) {
            let taken: Vec<usize> = mine.iter().filter_map(|x| card_of_port(&snap.cards, x.port)).collect();
            let quiet = |g: usize| !taken.contains(&g) && !card_contended(&snap, host, g, 0);
            if let Some(o) = start_copy(host, req, &snap, &mine, &quiet) {
                return Ok(o);
            }
        }
        if x.loading {
            host.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
            host.wait_loaded(x.port, id)?;
        }
        return Ok(Outcome::Placed { slot: x.slot, port: x.port });
    }
    let want = |g: usize| (req.need_mb.get(&g).copied().unwrap_or(0) + MARGIN_MB).max(req.asked_mb);
    let cards = &snap.cards;
    // a server that holds another model and could go now
    let idle_other = |x: &&Server| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x) && x.idle();
    if !mine.is_empty() {
        // its servers all have work: another copy on a card of its that has the room now (or once idle picture and sound
        // servers there stop) - else the copy with the fewest requests (its server queues it)
        if let Some(o) = start_copy(host, req, &snap, &mine, &|_| true) {
            return Ok(o);
        }
        let x = mine.iter().min_by_key(|x| x.inflight).unwrap();
        if x.loading {
            host.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
            host.wait_loaded(x.port, id)?;
        }
        return Ok(Outcome::Placed { slot: x.slot, port: x.port });
    }
    // not loaded: a card of its with the room now (the one with the most free), else the first that has it once idle
    // models there go (picture and sound servers, an idle video worker, a chat model not kept); a slot no server holds,
    // else one an idle model holds, else one a busy model of the kind holds, after its requests
    if let Some(why) = host.barred() {
        return Ok(Outcome::Barred(why));
    }
    // cards a GPU run holds: a model whose one copy spans one of them waits, any other goes on a free card of its
    let taken: Vec<(usize, String)> = gpus.iter().filter_map(|g| host.barred_on(*g).map(|w| (*g, w))).collect();
    let free_cards: Vec<usize> = gpus.iter().copied().filter(|g| !taken.iter().any(|(t, _)| t == g)).collect();
    if let Some((_, why)) = taken.first().filter(|_| m.spans || free_cards.is_empty()) {
        return Ok(Outcome::Barred(why.clone()));
    }
    if m.spans {
        return place_span(host, req, &snap);
    }
    let gpus = &free_cards;
    let plans = gpus.iter().map(|&g| (g, card_free(cards, g), room_plan(&snap, host, g, want(g), kind, false, Some(m))));
    let plan = if host.first_listed_card() { pick_first(plans) } else { pick_card(plans) };
    let empty = sv.iter().find(|x| x.model.is_none());
    let idle = sv.iter().filter(idle_other).min_by_key(|x| card_of_port(cards, x.port).is_none_or(|g| Some(g) != plan.as_ref().map(|p| p.0)));
    let on_mine = |x: &&Server| x.model.is_some() && !kept(x) && card_of_port(cards, x.port).is_some_and(|g| gpus.contains(&g));
    let Some((gpu, plan)) = plan else {
        // no room even then: a busy server of another model of the kind on a card of its goes after its requests (it
        // holds the room), then the next try loads it
        let held = |x: &Server| card_of_port(cards, x.port).map(|g| {
            let mb = cards.iter().find(|c| c.index == g).into_iter().flat_map(|c| c.procs.iter()).filter(|p| p.port == Some(x.port)).map(|p| p.mb).sum::<u64>();
            (g, mb)
        });
        let frees = |x: &&Server| held(x).is_some_and(|(g, mb)| room_plan(&snap, host, g, want(g).saturating_sub(mb), kind, false, Some(m)).is_some());
        if let Some(x) = sv.iter().filter(on_mine).filter(frees).min_by_key(|x| x.inflight) {
            drain(host, x, kind, id, job)?;
            return Ok(Outcome::NotYet(format!("server {} stopped for {id}", x.slot + 1)));
        }
        return Ok(Outcome::NotYet(no_room(&snap, gpus, m, want(gpus[0]))));
    };
    let ram = match host.ram_plan(&snap, m) {
        Ok(p) => p,
        Err(why) => return Ok(Outcome::NotYet(why)),
    };
    let x = match (empty, idle) {
        (Some(e), _) => e,
        (None, Some(i)) => i,
        // every server busy with another model: the one with the fewest requests, after them
        _ => {
            // (a kept model's server is never the one: every server kept, no room)
            let Some(x) = sv.iter().filter(|x| x.model.is_some() && !kept(x)).min_by_key(|x| (!on_mine(x) as u8, x.inflight)) else {
                return Ok(Outcome::NotYet(no_room(&snap, gpus, m, want(gpus[0]))));
            };
            drain(host, x, kind, id, job)?;
            x
        }
    };
    stop_slot(host, kind, x, id)?;
    // an idle model whose kind is busy changing its own server stays: the next try plans again
    if let Err((_, e)) = free_plan(host, &plan, kind, id).and_then(|_| free_plan(host, &ram, kind, id)) {
        return Ok(Outcome::NotYet(e));
    }
    if !host.wait_ram(id) {
        return Ok(Outcome::NotYet(format!("host memory did not come free for {id}")));
    }
    // what was stopped has let go of the room before the load (a holder that took work meanwhile is not stopped: the next
    // try plans again)
    if !wait_room(host, gpu, want(gpu)) {
        return Ok(Outcome::NotYet(format!("GPU {gpu} has not freed {:.1} GiB for {id} yet", want(gpu) as f64 / 1024.0)));
    }
    // (the room taken just before the start: another kind may have planned for this card meanwhile)
    if let Err(why) = host.reserve(m, x.slot, x.port, gpu, want(gpu)) {
        return Ok(Outcome::NotYet(why));
    }
    host.note(job, "loading", &format!("{id} loading on GPU {gpu} (server {})", x.slot + 1));
    if let Err((code, e)) = host.start(kind, id, x.slot, x.port, gpu) {
        // a refusal for memory is a wait, not a failure: what it asked is learned and the next try plans for it
        if let Some(why) = short_of_room(host, &snap.cards, m, gpu, &e, &[(gpu, want(gpu))]) {
            return Ok(Outcome::NotYet(why));
        }
        return Err((code, e));
    }
    Ok(Outcome::Placed { slot: x.slot, port: x.port })
}

/// What a model whose one copy spans its cards plans for on card `g`, MiB: its share there and the margin, at least a
/// refusal learned for that card - its first card's `asked_mb`, the others' `asked_rest_mb` (learned for the card the
/// refusal named: [`short_of_room`])
pub fn span_want(req: &PlaceReq, g: usize) -> u64 {
    let w = req.need_mb.get(&g).copied().unwrap_or(0) + MARGIN_MB;
    if req.model.gpus.first() == Some(&g) { w.max(req.asked_mb) } else { w.max(req.asked_rest_mb) }
}

/// The card a refusal for memory names ("needs about X GiB on <card name>; ..." or "... on <card name> (and 1.5 GiB
/// kept free); ..."), of the cards `among` (the ones the start took): by its name, the one with the least free when
/// two share it. None: no card named, or none of them.
pub fn refused_card(cards: &[Card], e: &str, among: &[usize]) -> Option<usize> {
    let name = e.split(" GiB on ").nth(1)?.split(';').next()?.split(" (").next()?.trim();
    cards.iter().filter(|c| among.contains(&c.index) && c.name == name).min_by_key(|c| c.free()).map(|c| c.index)
}

/// A start of `m` on `gpu` that failed (`e`), when it was a refusal for memory on one of the cards it took (`wants`: each
/// with what was planned there): the ask learned for the card the refusal names (the host's `learn_ask`; `rest`: not
/// the first card of a model whose one copy spans its cards) and why the request waits - the next try plans for the
/// ask, so it waits for busy work there or stops idle work, and the wait ends in 503 no_room. None: another failure,
/// or a need past that card's whole memory (it would never load there).
/// (nextcuda cf2debd: MiniMax Music 3 int8's starts were refused 36 times on its flow stage's card, the refusals learned
/// for the other card and sent back as failures)
pub fn short_of_room(host: &dyn Host, cards: &[Card], m: &ModelInfo, gpu: usize, e: &str, wants: &[(usize, u64)]) -> Option<String> {
    let (need, _) = asked_mb(e)?;
    let taken: Vec<usize> = wants.iter().map(|w| w.0).collect();
    let card = refused_card(cards, e, &taken).unwrap_or(gpu);
    let planned = wants.iter().find(|w| w.0 == card).map_or(0, |w| w.1);
    let rest = m.spans && m.gpus.first() != Some(&card);
    let more = host.learn_ask(&m.id, e, card, planned, rest);
    let total = cards.iter().find(|c| c.index == card).map_or(u64::MAX, |c| c.total_mb);
    if need > total {
        return None;
    }
    Some(if more {
        format!("{} needed more room on GPU {card} than planned ({e}); planned again", m.id)
    } else {
        format!("GPU {card} was short of the room {} asked ({e}); it waits for the room there", m.id)
    })
}

/// [`place`] for a model not loaded whose one copy spans its cards ([`ModelInfo::spans`]): the room on EACH of its cards
/// now or once idle holders there go (a card that cannot be cleared is no choice: the model waits, or a busy server of
/// its kind holding that card's room goes after its requests), a server slot as for any model, the room on every card
/// taken together, one start on all of them
fn place_span(host: &dyn Host, req: &PlaceReq, snap: &Snapshot) -> Result<Outcome, Fail> {
    let m = req.model;
    let (id, kind, job) = (m.id.as_str(), m.kind.as_str(), req.job);
    let gpus = &m.gpus;
    let want = |g: usize| span_want(req, g);
    let cards = &snap.cards;
    let sv: Vec<Server> = snap.servers_of(kind).to_vec();
    let kept = |x: &Server| x.model.as_ref().is_some_and(|mm| snap.kept.contains(mm));
    let plans: Vec<(usize, Option<Vec<Holder>>)> = gpus.iter().map(|&g| (g, room_plan(snap, host, g, want(g), kind, false, Some(m)))).collect();
    let short: Vec<usize> = plans.iter().filter(|(_, p)| p.is_none()).map(|(g, _)| *g).collect();
    if !short.is_empty() {
        // a busy server of another model of the kind holding a short card's room goes after its requests
        let on_short = |x: &&Server| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x)
            && card_of_port(cards, x.port).is_some_and(|g| short.contains(&g));
        let frees = |x: &&Server| short.iter().all(|&g| {
            let mb = cards.iter().find(|c| c.index == g).into_iter().flat_map(|c| c.procs.iter()).filter(|p| p.port == Some(x.port)).map(|p| p.mb).sum::<u64>();
            room_plan(snap, host, g, want(g).saturating_sub(mb), kind, false, Some(m)).is_some()
        });
        if let Some(x) = sv.iter().filter(on_short).filter(frees).min_by_key(|x| x.inflight) {
            drain(host, x, kind, id, job)?;
            return Ok(Outcome::NotYet(format!("server {} stopped for {id}", x.slot + 1)));
        }
        let lack = short.iter().map(|g| no_room(snap, &[*g], m, want(*g))).collect::<Vec<_>>().join(" / ");
        return Ok(Outcome::NotYet(lack));
    }
    let ram = match host.ram_plan(snap, m) {
        Ok(p) => p,
        Err(why) => return Ok(Outcome::NotYet(why)),
    };
    // a slot no server holds, else one an idle model holds (one on its cards first), else the one with the fewest requests after them
    let idle = sv.iter().filter(|x| x.model.is_some() && x.model.as_deref() != Some(id) && !kept(x) && x.idle())
        .min_by_key(|x| card_of_port(cards, x.port).is_none_or(|g| !gpus.contains(&g)));
    let x = match (sv.iter().find(|x| x.model.is_none()), idle) {
        (Some(e), _) => e,
        (None, Some(i)) => i,
        _ => {
            let Some(x) = sv.iter().filter(|x| x.model.is_some() && !kept(x)).min_by_key(|x| x.inflight) else {
                return Ok(Outcome::NotYet(no_room(snap, gpus, m, want(gpus[0]))));
            };
            drain(host, x, kind, id, job)?;
            x
        }
    };
    stop_slot(host, kind, x, id)?;
    for (_, plan) in &plans {
        if let Err((_, e)) = free_plan(host, plan.as_deref().unwrap_or(&[]), kind, id) {
            return Ok(Outcome::NotYet(e));
        }
    }
    if let Err((_, e)) = free_plan(host, &ram, kind, id) {
        return Ok(Outcome::NotYet(e));
    }
    if !host.wait_ram(id) {
        return Ok(Outcome::NotYet(format!("host memory did not come free for {id}")));
    }
    for &g in gpus {
        if !wait_room(host, g, want(g)) {
            return Ok(Outcome::NotYet(format!("GPU {g} has not freed {:.1} GiB for {id} yet", want(g) as f64 / 1024.0)));
        }
    }
    let wants: Vec<(usize, u64)> = gpus.iter().map(|&g| (g, want(g))).collect();
    if let Err(why) = host.reserve_span(m, x.slot, x.port, &wants) {
        return Ok(Outcome::NotYet(why));
    }
    let on = gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(" + ");
    host.note(job, "loading", &format!("{id} loading on GPUs {on} (server {})", x.slot + 1));
    if let Err((code, e)) = host.start(kind, id, x.slot, x.port, gpus[0]) {
        // a refusal for memory on any of its cards is a wait: learned for the card it names, planned for next try
        if let Some(why) = short_of_room(host, &snap.cards, m, gpus[0], &e, &wants) {
            return Ok(Outcome::NotYet(why));
        }
        return Err((code, e));
    }
    Ok(Outcome::Placed { slot: x.slot, port: x.port })
}

/// A server of the model up (not loading) with nothing to do, on a card no other work saturates: the request counted on
/// it under the kind's short lock ([`Host::lock_hot`]) and its (slot, port), without waiting for the kind's lock - which a
/// load of another model holds while it lasts. None: [`place`] decides, under the kind's lock (a copy to start, a load,
/// a server busy or contended)
pub fn hot(host: &dyn Host, req: &PlaceReq) -> Option<(usize, u16)> {
    let m = req.model;
    if !m.enabled {
        return None;
    }
    let _h = host.lock_hot(&m.kind);
    let snap = host.snapshot();
    let x = snap.servers_of(&m.kind).iter().find(|x| {
        x.model.as_deref() == Some(m.id.as_str()) && x.idle() && !card_of_port(&snap.cards, x.port).is_some_and(|g| card_contended(&snap, host, g, x.port))
    })?;
    host.placed(x.port);
    Some((x.slot, x.port))
}

/// The loop around [`place`]: tries again every 3 s until a server is found, a cancel (499), the wait runs out (503 no_room:
/// `default_wait`, or - when the caller gave `max_wait` - that, answered `wait_too_long` with the `eta` it was given) or
/// a failure. Returns the (slot, port) to send the request to.
pub fn ensure(host: &dyn Host, req: &PlaceReq, default_wait: std::time::Duration, max_wait: Option<f64>, eta: &serde_json::Value) -> Result<(usize, u16), Fail> {
    let limit = crate::rsvp::wait_limit(default_wait, max_wait);
    let mut t0 = host.now();
    let id = req.model.id.as_str();
    loop {
        if req.job.is_some_and(|j| host.cancelled(j)) {
            return Err((499, format!("cancelled while it waited for {id}")));
        }
        if let Some(why) = host.not_ready(id) {
            host.note(req.job, "waiting", &why);
            if let Some(f) = host.give_up() {
                return Err(f);
            }
            if host.now() - t0 > limit.as_secs_f64() {
                return Err((409, why));
            }
            host.sleep(3.0);
            continue;
        }
        // a server of it up and idle: taken at once, whatever another model of the kind is doing (a load holds the
        // kind's lock for minutes: two zonos3 jobs waited a minute for a Sophon load beside their idle server, 2026-11-16)
        if let Some(placed) = hot(host, req) {
            return Ok(placed);
        }
        // else one try with the kind's lock held; the request counted on its server before the lock goes
        let tried = {
            let _g = host.wait_lock_kind(&req.model.kind);
            let o = place(host, req)?;
            if let Outcome::Placed { port, .. } = o {
                let _h = host.lock_hot(&req.model.kind);
                host.placed(port);
            }
            o
        };
        if let Outcome::Placed { slot, port } = tried {
            return Ok((slot, port));
        }
        if let Some(f) = host.give_up() {
            return Err(f);
        }
        let why = match tried {
            Outcome::Barred(why) => {
                if req.job.is_none() {
                    return Err((503, why));
                }
                host.note(req.job, "cards_taken", &why);
                host.sleep(3.0);
                t0 = host.now();
                continue;
            }
            Outcome::NotYet(why) => why,
            Outcome::Placed { .. } => unreachable!(),
        };
        host.note(req.job, "waiting", &why);
        let waited = host.now() - t0;
        if let Some(mw) = max_wait.filter(|mw| waited > *mw) {
            return Err(crate::eta::wait_expired(eta, mw, waited));
        }
        if waited > limit.as_secs_f64() {
            return Err((503, format!("{why}; {id} loads once one of them idles out or is stopped")));
        }
        host.sleep(3.0);
    }
}

/// A server's "needs about X GiB on <card>; Y GiB are free" (its refusal to load), as (X, Y) in MiB
pub fn asked_mb(e: &str) -> Option<(u64, u64)> {
    let gib = |t: &str| t.trim().parse::<f64>().ok().map(|g| (g * 1024.0) as u64);
    let need = e.split("needs about ").nth(1)?.split(" GiB").next()?;
    let free = e.split(" GiB are free").next()?.rsplit(' ').next()?;
    Some((gib(need)?, gib(free)?))
}

/// What a card must have free, as the front door counts it (`front_free`: total less used), for a server that asked
/// `need` MiB and found `had` free: the difference is its context and what the driver keeps (half a GiB at least: the
/// sample may still count the server that ended), with a quarter GiB for the asking's rounding
pub fn ask_mb(need: u64, had: u64, front_free: u64) -> u64 {
    need + front_free.saturating_sub(had).max(512) + 256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_names_its_card() {
        let card = |index: usize, name: &str, used_mb: u64| Card { index, name: name.into(), total_mb: 32768, used_mb, procs: Vec::new() };
        let cards = [card(0, "Intel(R) Graphics [0xe223]", 1000), card(1, "Intel(R) Graphics [0xe222]", 9000)];
        // Sophon's own words (its talker's card named, with the kept margin in brackets)
        let e = "audio start sophon on GPUs 0 + 1: sophon needs about 6.2 GiB on Intel(R) Graphics [0xe222] (and 1.5 GiB kept free); 4.1 GiB are free";
        assert_eq!(refused_card(&cards, e, &[0, 1]), Some(1));
        assert_eq!(refused_card(&cards, "x needs about 2.0 GiB on Intel(R) Graphics [0xe223]; 1.0 GiB are free", &[0, 1]), Some(0));
        // not among the cards the start took, or no card named
        assert_eq!(refused_card(&cards, e, &[0]), None);
        assert_eq!(refused_card(&cards, "device out of memory", &[0, 1]), None);
        // two cards of one name: the one with less free
        let same = [card(0, "B", 1000), card(1, "B", 9000)];
        assert_eq!(refused_card(&same, "x needs about 2.0 GiB on B; 1.0 GiB are free", &[0, 1]), Some(1));
    }

    #[test]
    fn a_refusal_for_memory_is_read_and_planned_for() {
        let e = "audio start qwen3-tts-custom on GPU 0 (server 2): nextcuda: nextcuda-audio-1 ended while loading (nextcuda: qwen3-tts needs \
                 about 5.6 GiB on NVIDIA GeForce RTX 5070; 5.3 GiB are free): nextcuda audio logs";
        let (need, had) = asked_mb(e).unwrap();
        assert_eq!((need, had), (5734, 5427));
        let ask = ask_mb(need, had, 6093);
        assert!(ask > 6093, "{ask}");
        assert_eq!(asked_mb("qwen3.8 needs about 30.1 GiB on X (28.0 GiB of weights); 22.5 GiB are free").map(|x| x.0), Some(30822));
        assert_eq!(asked_mb("nextcuda-audio-1 ended while loading"), None);
    }
}
