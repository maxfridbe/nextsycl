//! `place`: one try at a server for a model, over a [`Host`]. The caller holds the kind's lock while it lasts and loops
//! (sleeping, noting the wait) until it has a server or its wait runs out.

use std::collections::BTreeMap;

use crate::host::Host;
use crate::room::{card_free, card_of_port, extra_copy_allowed, no_room, pick_card, pick_first, room_plan, Free, Holder, ModelInfo, Server, JOB_ROOM_WAIT_S, MARGIN_MB};
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
    // a server of it with nothing to do
    if let Some(x) = mine.iter().find(|x| !x.busy && x.inflight == 0) {
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
        // its servers all have work: another copy on a card of its without one that has the room now (or once idle
        // picture and sound servers there stop), in a slot no server holds or one an idle model holds - else the copy
        // with the fewest requests (its server queues it)
        let on: Vec<usize> = mine.iter().filter_map(|x| card_of_port(cards, x.port)).collect();
        // copies per card: extra ones up to the profile's recommended count (the default 1: one a card), never past its max
        let per_card = m.profile.concurrency.recommended_copies.min(m.profile.concurrency.max_copies_per_card) as usize;
        let spare = sv.iter().find(|x| x.model.is_none()).or_else(|| sv.iter().find(idle_other));
        // (no second copy while nothing may load, or when its host memory is not there: the requests share the first)
        let copy_plans = gpus.iter().copied().filter(|g| extra_copy_allowed(&on, *g, per_card)).filter(|_| host.barred().is_none())
            .map(|g| (g, card_free(cards, g), room_plan(&snap, host, g, want(g), kind, true, Some(m))));
        let plan = if host.first_listed_card() { pick_first(copy_plans) } else { pick_card(copy_plans) };
        let ram = plan.as_ref().and_then(|_| host.ram_plan(&snap, m).ok());
        if let (Some(x), Some((gpu, plan)), Some(ram)) = (spare, plan, ram) {
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
                    host.learn_ask(id, e, gpu, want(gpu));
                })
            });
            match started {
                Ok(()) => return Ok(Outcome::Placed { slot: x.slot, port: x.port }),
                Err((_, e)) => eprintln!("{id}: a second copy did not start ({e}); the requests share the first"),
            }
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
        // the server asked more than was planned for: the next try plans for that (stopping idle work for it)
        if host.learn_ask(id, &e, gpu, want(gpu)) {
            return Ok(Outcome::NotYet(format!("{id} needed more room on GPU {gpu} than planned ({e}); planned again")));
        }
        return Err((code, e));
    }
    Ok(Outcome::Placed { slot: x.slot, port: x.port })
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
        // one try with the kind's lock held; the request counted on its server before the lock goes
        let tried = {
            let _g = host.wait_lock_kind(&req.model.kind);
            let o = place(host, req)?;
            if let Outcome::Placed { port, .. } = o {
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
