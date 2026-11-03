//! How many clips at once: one per usable card. Pure: what the daemon and the host say goes in, the decision comes out.
//!
//! - **A card is usable** when the daemon serves it, nothing runs or waits for it, and the front door does not say a pin or
//!   hold has it (`blocked`, a chat model kept there included). With nothing running one clip is always let through, on
//!   whatever card: it waits there for room as it always did.
//! - **Host memory**: a worker's MemAvailable drop is about `WORKER_RAM_MB`; a further clip must leave `RAM_FLOOR_MB`. A worker
//!   still loading has not shown its drop yet, so it counts as well. The first clip has no such check.
//! - **A clip that needs another's output runs after it.** A clip naming an anchor (prev, first ...) resolves it when it
//!   starts, against the clips its series has finished: it waits for its own series' running clips, and what of that series
//!   is queued behind it waits too. A clip naming a running clip's files (h3_...[.mp4, .last.png, .lastlat.safetensors ...]) waits
//!   for that clip. A `prev` with nothing of its own, running or finished, means "the newest clip made": it waits for every
//!   running clip. Other clips pass whatever waits.
//! - **A failed or cancelled predecessor** gives its dependents an answer (`failed_dependency`), never `prev` falling on an
//!   older clip.

use serde_json::Value;

use super::queue::{label_prefix, project_of, s, Queue, MAGIC};

/// A video worker's drop in MemAvailable (one REF2VA worker on hardac1: about 15 GiB)
pub const WORKER_RAM_MB: u64 = 15 * 1024;
/// What a further clip must leave of MemAvailable
pub const RAM_FLOOR_MB: u64 = 8 * 1024;

/// Clips possible now
#[derive(Debug, PartialEq, Eq)]
pub struct Room {
    /// the cards the daemon serves
    pub total: usize,
    /// clips that can start now
    pub free: usize,
    /// why one fewer than the free cards would say
    pub why: Option<String>,
}

/// MemAvailable, MiB, from a /proc/meminfo's text
pub fn mem_available_mb(meminfo: &str) -> Option<u64> {
    meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

/// The room now. `engine`: the daemon's /engine/status (None: it does not answer); `running`: the studio's clips running,
/// `loading`: those whose worker has not loaded yet; `host_mb`: host memory free (the door's figure through the daemon,
/// else MemAvailable; None: not known, no check)
pub fn room(engine: Option<&Value>, running: usize, loading: usize, host_mb: Option<u64>) -> Room {
    let Some(gpus) = engine.and_then(|e| e["gpus"].as_array()).filter(|g| !g.is_empty()) else {
        return Room { total: 1, free: usize::from(running == 0), why: None };
    };
    let e = engine.unwrap();
    let blocked = |g: &Value| e["blocked"].as_array().is_some_and(|b| b.contains(&g["gpu"]));
    let open: Vec<&Value> = gpus.iter().filter(|g| g["running"].is_null() && !blocked(g)).collect();
    let waiting = e["queued"].as_array().map_or(0, Vec::len);
    let mut free = open.len().saturating_sub(waiting);
    let mut why = None;
    if running == 0 {
        // the first clip: as before, whatever the cards say; it waits on one
        return Room { total: gpus.len(), free: free.max(1), why };
    }
    // a worker already up on the card a clip would take has shown its drop
    let new_workers = open.iter().take(free).filter(|g| g["worker"].is_null()).count();
    if let Some(mb) = host_mb.filter(|_| new_workers > 0) {
        let spare = mb.saturating_sub(loading as u64 * WORKER_RAM_MB).saturating_sub(RAM_FLOOR_MB);
        let fit = (spare / WORKER_RAM_MB) as usize;
        if fit < new_workers {
            free -= new_workers - fit;
            why = Some(format!("host memory: {} GiB available, a further worker takes about {} GiB and must leave {} GiB{}",
                               mb / 1024, WORKER_RAM_MB / 1024, RAM_FLOOR_MB / 1024,
                               if loading > 0 { format!(" ({loading} still loading counted)") } else { String::new() }));
        }
    }
    Room { total: gpus.len(), free, why }
}

/// The anchors an item resolves when it starts, as written
fn anchor_words(item: &Value) -> Vec<String> {
    let mut v: Vec<String> = ["first_frame", "last_frame", "first_audio", "exposure_ref", "source"].iter().map(|k| s(item, k)).collect();
    v.extend(s(item, "guide_clip").split(',').map(|g| g.trim().split(':').next().unwrap_or("").to_string()));
    v
}

/// The clips an item names by file, by their stems (h3_x.last.png -> h3_x)
fn named_clips(item: &Value) -> Vec<String> {
    let mut v = anchor_words(item);
    for k in ["ref_images", "ref_videos", "ref_audios"] {
        v.extend(item[k].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string));
    }
    v.extend(["control_video", "control_mask", "control_source"].iter().map(|k| s(item, k)));
    v.iter().filter(|w| !w.is_empty() && !MAGIC.contains(&w.as_str())).map(|w| w.split('.').next().unwrap_or("").to_string()).collect()
}

/// The clip that keeps `item` from running for good, among those that ended (oldest first; `failed`: this one failed or was
/// cancelled and will not go round again, a dependent dropped included): one it names by file, or for an anchor its series'
/// newest that ended, if that failed. Its own earlier tries (the same label or reentrant_id) do not count, so a clip can be
/// made again. None: nothing stands in its way.
pub fn failed_dependency(item: &Value, ended: &[(&Value, bool)]) -> Option<String> {
    let named = named_clips(item);
    if let Some((r, _)) = ended.iter().find(|(r, f)| *f && !s(r, "name").is_empty() && named.contains(&s(r, "name"))) {
        return Some(name_of(r));
    }
    if !anchored(item) {
        return None;
    }
    let (label, rid) = (s(item, "label"), s(item, "reentrant_id"));
    for (r, failed) in ended.iter().rev().filter(|(r, _)| same_series(item, r)) {
        if (!label.is_empty() && s(r, "label") == label) || (!rid.is_empty() && s(r, "reentrant_id") == rid) {
            continue;
        }
        return failed.then(|| name_of(r));
    }
    None
}

/// Whether an item names an anchor (its picture, sound or motion comes from a clip not chosen yet)
pub fn anchored(item: &Value) -> bool {
    anchor_words(item).iter().any(|w| MAGIC.contains(&w.as_str()))
}

fn names_prev(item: &Value) -> bool {
    anchor_words(item).iter().any(|w| w == "prev")
}

/// One project, or one label series
pub fn same_series(a: &Value, b: &Value) -> bool {
    let (pa, pb) = (project_of(a), project_of(b));
    if !pa.is_empty() && pa == pb {
        return true;
    }
    let (la, lb) = (label_prefix(&s(a, "label")), label_prefix(&s(b, "label")));
    !la.is_empty() && la == lb
}

/// Whether a clip may start beside what runs
#[derive(Debug, PartialEq, Eq)]
pub enum Gate {
    Free,
    /// it needs these clips' output: it waits for them
    Series(Vec<String>),
    /// its `prev` means the newest clip anywhere: it waits for everything running
    All(Vec<String>),
}

fn name_of(v: &Value) -> String {
    let l = s(v, "label");
    if l.is_empty() { s(v, "name") } else { l }
}

/// `running`: the clips running (their records or items). `has_own`: its series has a finished clip for `prev` to fall on.
pub fn gate(item: &Value, running: &[Value], has_own: bool) -> Gate {
    let named = named_clips(item);
    let by_file: Vec<String> = running.iter().filter(|r| !s(r, "name").is_empty() && named.contains(&s(r, "name"))).map(name_of).collect();
    if !by_file.is_empty() {
        return Gate::Series(by_file);
    }
    if !anchored(item) {
        return Gate::Free;
    }
    let mine: Vec<String> = running.iter().filter(|r| same_series(item, r)).map(name_of).collect();
    if !mine.is_empty() {
        return Gate::Series(mine);
    }
    if names_prev(item) && !has_own && !running.is_empty() {
        return Gate::All(running.iter().map(name_of).collect());
    }
    Gate::Free
}

/// What the scheduler starts now
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// queue positions, in order
    pub starts: Vec<usize>,
}

/// The queue in its order (priority is its order): the first items that can start, up to `free`. An item waiting for another
/// clip does not hold back other series' clips; the rest of its own series stays behind it.
pub fn plan(q: &Queue, running: &[Value], free: usize, has_own: &dyn Fn(&Value) -> bool) -> Plan {
    let mut out = Plan::default();
    let mut beside: Vec<Value> = running.to_vec();
    let mut stuck: Vec<&Value> = Vec::new();
    for (i, it) in q.items.iter().enumerate() {
        if out.starts.len() >= free {
            break;
        }
        if q.held(it) || stuck.iter().any(|o| same_series(it, o)) {
            if !q.held(it) {
                stuck.push(it);
            }
            continue;
        }
        match gate(it, &beside, has_own(it)) {
            Gate::Free => {
                out.starts.push(i);
                beside.push(it.clone());
            }
            Gate::Series(_) | Gate::All(_) => stuck.push(it),
        }
    }
    out
}

/// Whether a request may start at once given the plan for the queue: a card left over, its own series not queued ahead of
/// it, and what it needs made. The reason why not, in words, for `queue: false`.
pub fn direct(item: &Value, q: &Queue, running: &[Value], free: usize, has_own: &dyn Fn(&Value) -> bool) -> Result<(), String> {
    let p = plan(q, running, free, has_own);
    if free <= p.starts.len() {
        return Err(if running.is_empty() { "the queued clips take the cards".to_string() } else { "a job is running on every usable card".to_string() });
    }
    if anchored(item) && q.items.iter().any(|i| !q.held(i) && same_series(item, i)) {
        return Err("clips of its series are queued ahead of it".to_string());
    }
    let mut beside = running.to_vec();
    beside.extend(p.starts.iter().map(|i| q.items[*i].clone()));
    match gate(item, &beside, has_own(item)) {
        Gate::Free => Ok(()),
        Gate::Series(on) => Err(format!("it needs the output of {}, still running", on.join(", "))),
        Gate::All(on) => Err(format!("it carries on from the newest clip and {} is still running", on.join(", "))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn engine(gpus: Value) -> Value {
        json!({"gpus": gpus, "queued": []})
    }

    #[test]
    fn one_clip_per_usable_card() {
        let both = engine(json!([{"gpu": 0, "running": null}, {"gpu": 1, "running": null}]));
        assert_eq!(room(Some(&both), 0, 0, Some(40_000)), Room { total: 2, free: 2, why: None });
        let one = engine(json!([{"gpu": 0, "running": {"id": 1}}, {"gpu": 1, "running": null}]));
        assert_eq!(room(Some(&one), 1, 0, Some(40_000)).free, 1);
        let none = engine(json!([{"gpu": 0, "running": {"id": 1}}, {"gpu": 1, "running": {"id": 2}}]));
        assert_eq!(room(Some(&none), 2, 0, Some(40_000)).free, 0);
        // a job waiting in the daemon has the card it is about to get
        let mut q = one.clone();
        q["queued"] = json!([9]);
        assert_eq!(room(Some(&q), 2, 0, Some(40_000)).free, 0);
    }

    #[test]
    fn a_card_a_pin_or_hold_has_is_no_slot_but_the_first_clip_still_goes() {
        let mut e = engine(json!([{"gpu": 0, "running": {"id": 1}}, {"gpu": 1, "running": null}]));
        e["blocked"] = json!([1]);
        assert_eq!(room(Some(&e), 1, 0, Some(40_000)).free, 0);
        let mut idle = engine(json!([{"gpu": 0, "running": null}, {"gpu": 1, "running": null}]));
        idle["blocked"] = json!([0, 1]);
        assert_eq!(room(Some(&idle), 0, 0, Some(40_000)).free, 1, "it waits on one, as it did");
    }

    #[test]
    fn a_daemon_that_does_not_answer_is_one_clip() {
        assert_eq!(room(None, 0, 0, None), Room { total: 1, free: 1, why: None });
        assert_eq!(room(None, 1, 0, None).free, 0);
    }

    #[test]
    fn host_memory_keeps_a_further_worker_out_and_not_the_first() {
        let e = engine(json!([{"gpu": 0, "running": {"id": 1}}, {"gpu": 1, "running": null}]));
        // the live figure: 43 GiB with one REF2VA worker
        assert_eq!(room(Some(&e), 1, 0, Some(43 * 1024)).free, 1);
        assert_eq!(room(Some(&e), 1, 0, Some(23 * 1024)).free, 1, "exactly 8 left");
        let r = room(Some(&e), 1, 0, Some(23 * 1024 - 1));
        assert_eq!(r.free, 0);
        assert!(r.why.unwrap().contains("host memory"));
        assert_eq!(room(Some(&e), 1, 1, Some(30 * 1024)).free, 0, "the first still loading: its drop is still to come");
        assert_eq!(room(Some(&e), 1, 0, None).free, 1, "not known: no check");
        let idle = engine(json!([{"gpu": 0, "running": null}, {"gpu": 1, "running": null}]));
        assert_eq!(room(Some(&idle), 0, 0, Some(1024)).free, 2, "no clip running: the rule is about a further one");
        let up = engine(json!([{"gpu": 0, "running": {"id": 1}}, {"gpu": 1, "running": null, "worker": {"pid": 4}}]));
        assert_eq!(room(Some(&up), 1, 0, Some(10 * 1024)).free, 1, "its worker is up: its drop is in the figure");
    }

    #[test]
    fn meminfo_is_read() {
        assert_eq!(mem_available_mb("MemTotal: 10 kB\nMemAvailable:   2097152 kB\n"), Some(2048));
        assert_eq!(mem_available_mb("MemTotal: 10 kB\n"), None);
    }

    fn q(items: Vec<Value>) -> Queue {
        Queue { items, ..Queue::default() }
    }

    #[test]
    fn a_chain_runs_in_order_and_other_chains_go_past() {
        let a2 = json!({"label": "A 02/09: x", "guide_clip": "prev:22:0"});
        let a3 = json!({"label": "A 03/09: x", "guide_clip": "prev:22:0"});
        let b2 = json!({"label": "B 02/09: x", "first_frame": "prev"});
        let run = [json!({"label": "A 01/09: x"})];
        let own = |_: &Value| true;
        let p = plan(&q(vec![a2.clone(), a3.clone(), b2.clone()]), &run, 1, &own);
        assert_eq!(p, Plan { starts: vec![2] }, "A waits for its own clip, B takes the card");
        assert_eq!(plan(&q(vec![a2.clone(), a3.clone()]), &[], 2, &own).starts, vec![0], "A's second waits for its first, even with a card free");
        // a clip with no anchors of the same series stays behind one held back, or its number would become `prev`
        let plain = json!({"label": "A 03/09: x"});
        assert_eq!(plan(&q(vec![a2, plain]), &run, 1, &own).starts, Vec::<usize>::new());
    }

    #[test]
    fn a_prev_with_nothing_of_its_own_waits_for_everything_and_others_pass_it() {
        let loose = json!({"label": "loose", "first_frame": "prev"});
        let next = json!({"label": "other"});
        let run = [json!({"label": "z"})];
        let none = |_: &Value| false;
        assert_eq!(plan(&q(vec![loose.clone(), next.clone()]), &run, 1, &none), Plan { starts: vec![1] }, "it waits, the other goes");
        assert_eq!(plan(&q(vec![loose.clone(), next.clone()]), &[], 2, &none).starts, vec![0, 1], "nothing running: it starts first, its anchor fixed as it does");
        assert!(direct(&loose, &q(vec![]), &run, 1, &none).is_err());
        assert!(direct(&next, &q(vec![loose]), &run, 1, &none).is_ok(), "nor does it keep a direct clip out");
    }

    #[test]
    fn a_file_of_a_finished_clip_needs_no_order_and_of_a_running_one_waits() {
        let a = json!({"label": "A 02/09: x", "guide_clip": "h3_20261031_120000:22:0", "first_audio": "h3_20261031_120000", "project": "song"});
        let run = [json!({"label": "A 01/09: x", "project": "song", "name": "h3_other"})];
        assert_eq!(gate(&a, &run, false), Gate::Free);
        let run = [json!({"label": "A 01/09: x", "name": "h3_20261031_120000"})];
        assert_eq!(gate(&a, &run, false), Gate::Series(vec!["A 01/09: x".to_string()]));
        for item in [json!({"first_frame": "h3_20261031_120000.last.png"}), json!({"source": "h3_20261031_120000.latents.safetensors"}), json!({"ref_videos": ["h3_20261031_120000.mp4"]})] {
            assert!(matches!(gate(&item, &run, false), Gate::Series(_)), "{item}");
        }
    }

    #[test]
    fn a_failed_predecessor_is_named_and_a_later_success_or_a_retake_clears_it() {
        let d = json!({"label": "A 03/09: x", "first_frame": "prev"});
        let bad = json!({"label": "A 02/09: x", "name": "h3_b"});
        let good = json!({"label": "A 01/09: x", "name": "h3_a"});
        assert_eq!(failed_dependency(&d, &[(&good, false), (&bad, true)]).as_deref(), Some("A 02/09: x"));
        assert_eq!(failed_dependency(&d, &[(&bad, true), (&good, false)]), None, "made since");
        assert_eq!(failed_dependency(&d, &[(&good, false)]), None);
        // a clip made again: its own failure is not in its way, an earlier clip's still is
        let again = json!({"label": "A 02/09: x", "first_frame": "prev"});
        assert_eq!(failed_dependency(&again, &[(&good, false), (&bad, true)]), None);
        // another series, or no anchor: not its business
        assert_eq!(failed_dependency(&json!({"label": "B 02/09: x", "first_frame": "prev"}), &[(&bad, true)]), None);
        assert_eq!(failed_dependency(&json!({"label": "A 03/09: x"}), &[(&bad, true)]), None);
        // by file
        assert_eq!(failed_dependency(&json!({"guide_clip": "h3_b:22:0"}), &[(&bad, true)]).as_deref(), Some("A 02/09: x"));
        // every guide of several counts
        assert_eq!(failed_dependency(&json!({"guide_clip": "anchor.png:1:120, h3_b:22:0"}), &[(&bad, true)]).as_deref(), Some("A 02/09: x"));
    }

    #[test]
    fn priority_and_holds_are_the_queues() {
        let mut queue = q(vec![json!({"label": "a", "batch": "h"}), json!({"label": "b"}), json!({"label": "c"})]);
        queue.paused_batches.insert("h".into());
        let own = |_: &Value| true;
        assert_eq!(plan(&queue, &[], 1, &own).starts, vec![1], "the held one is passed; the next in order goes");
        assert_eq!(plan(&queue, &[], 2, &own).starts, vec![1, 2]);
        queue.paused = true;
        assert!(plan(&queue, &[], 2, &own).starts.is_empty(), "a paused queue starts nothing");
    }

    #[test]
    fn a_direct_clip_goes_only_with_a_card_left_over() {
        let own = |_: &Value| true;
        let x = json!({"label": "x"});
        assert!(direct(&x, &q(vec![]), &[json!({"label": "y"})], 1, &own).is_ok());
        assert!(direct(&x, &q(vec![json!({"label": "z"})]), &[json!({"label": "y"})], 1, &own).is_err(), "the queued one has it");
        assert!(direct(&x, &q(vec![]), &[json!({"label": "y"})], 0, &own).is_err());
        assert!(direct(&x, &q(vec![json!({"label": "z"})]), &[], 2, &own).is_ok(), "two free, one queued");
    }
}
