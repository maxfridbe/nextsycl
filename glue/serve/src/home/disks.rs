//! The drives on the page and the API (crate::drives): the list in the state (each drive, and each model's place on
//! them with what stops a move), a drive's read speed measured in the background and kept in serve.json, a model's
//! files moved to another drive as a job - one at a time, never while the model is loaded, and nothing loads it
//! meanwhile (`moving_why`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use super::registry;
use super::{now, Home};
use crate::drives::{self, Drive};

type Fail = (u16, String);

/// The drives' part of the front door's state
#[derive(Default)]
pub(crate) struct Disks {
    /// drive id -> its last measurement {mb_s, bytes, seconds, at, file}: kept in serve.json
    pub measured: Mutex<Map<String, Value>>,
    /// the drive being measured now, and its cancel
    measuring: Mutex<Option<String>>,
    stop_measure: AtomicBool,
    /// the move under way, or the last one (its outcome)
    mv: Mutex<Option<Arc<drives::Move>>>,
}

impl Disks {
    pub(crate) fn new(st: &Value) -> Disks {
        Disks { measured: Mutex::new(st["drives"].as_object().cloned().unwrap_or_default()), ..Default::default() }
    }

    /// The move under way (not the last one, ended)
    fn running(&self) -> Option<Arc<drives::Move>> {
        self.mv.lock().unwrap().clone().filter(|m| !m.ended())
    }
}

/// A model's state that holds its files (a server or the video daemon has them open)
fn holds(state: &str) -> Option<&'static str> {
    match state {
        "loaded" | "busy" => Some("loaded: stop it first"),
        "loading" => Some("loading: wait, then stop it"),
        "ready" => Some("the video daemon has it: disable it first"),
        _ => None,
    }
}

impl Home {
    /// The models folders set (NS_MODELS, NS_MODEL_DIRS)
    fn folders(&self) -> Vec<PathBuf> {
        drives::folders(self.o.cfg.get("NS_MODELS"), self.o.cfg.get("NS_MODEL_DIRS"))
    }

    fn measured_mb_s(&self, id: &str) -> Option<f64> {
        self.disks.measured.lock().unwrap().get(id).and_then(|m| m["mb_s"].as_f64())
    }

    /// Why `id` cannot load now: its files are moving (enable, load and a job's start wait or refuse)
    pub(crate) fn moving_why(&self, id: &str) -> Option<String> {
        if let Some(m) = self.disks.running().filter(|m| m.plan.model == id) {
            return Some(format!("{id} is moving to {} (its files are being copied): it loads once the move ends", m.plan.to));
        }
        drives::moving_why(&registry::registry(&self.o.cfg), id)
    }

    /// system.drives: every drive (its device, link, size, free space, models, measured speed), every model's place on
    /// them and what stops a move to each other drive, the move under way (or the last)
    pub(crate) fn drives_view(&self, all: &[Value], states: &[Value]) -> Value {
        let cards = self.o.cfg.cards();
        // (a model's sidecars count with its files)
        let all = &drives::with_sidecars(all, &cards);
        let set = self.folders();
        let list = drives::drives(all, &set);
        let roots = drives::roots(&set, &list);
        let ms = drives::mounts();
        let measuring = self.disks.measuring.lock().unwrap().clone();
        let running = self.disks.running();
        let state_of = |id: &str| states.iter().find(|m| m["id"] == id).map(|m| m["state"].as_str().unwrap_or("").to_string()).unwrap_or_default();
        let loading = self.loading.lock().unwrap().clone();
        let mut models = Map::new();
        for m in all {
            let id = m["id"].as_str().unwrap_or("").to_string();
            let files = drives::files_of(m);
            let mut on: Map<String, Value> = Map::new();
            let mut bytes = 0u64;
            let mut missing = 0;
            for f in &files {
                match std::fs::metadata(f) {
                    Ok(md) => {
                        bytes += md.len();
                        let d = list.iter().find(|d| d.holds(f, &ms)).map_or("?".to_string(), |d| d.id.clone());
                        let e = on.entry(d).or_insert(json!(0));
                        *e = json!(e.as_u64().unwrap_or(0) + md.len());
                    }
                    Err(_) => missing += 1,
                }
            }
            let state = state_of(&id);
            let busy = running.as_ref().map(|r| format!("{} is moving: one move at a time", r.plan.model))
                .or_else(|| holds(&state).map(str::to_string))
                .or_else(|| loading.contains_key(&id).then(|| "loading: wait, then stop it".to_string()))
                .or_else(|| (missing > 0).then(|| format!("{missing} of its files are missing")));
            let mut moves = Map::new();
            for d in &list {
                // a drive it is all on already offers nothing
                if on.len() == 1 && on.contains_key(&d.id) {
                    continue;
                }
                let why = busy.clone().map_or_else(|| match drives::plan(all, &id, d, &list, &roots, &ms, &cards) {
                    Ok(p) => drives::room(p.need, d.free, d.size).err().map(|e| format!("{}: {e}", d.id)).map_or(json!({"need": p.need}), |e| json!({"need": p.need, "why": e})),
                    Err(e) => json!({"why": e}),
                }, |b| json!({"why": b}));
                moves.insert(d.id.clone(), why);
            }
            models.insert(id.clone(), json!({"bytes": bytes, "files": files.len(), "missing": missing, "drives": on,
                                             "streams": drives::streams(m, &cards), "moving": running.as_ref().is_some_and(|r| r.plan.model == id),
                                             "move": moves}));
        }
        let drives: Vec<Value> = list.iter().map(|d| {
            let mut v = d.json();
            v["measured"] = self.disks.measured.lock().unwrap().get(&d.id).cloned().unwrap_or(Value::Null);
            v["measuring"] = json!(measuring.as_deref() == Some(d.id.as_str()));
            v["rate_gb_s"] = json!(drives::rate(d, self.measured_mb_s(&d.id)).map(|r| (r * 100.0).round() / 100.0));
            v
        }).collect();
        let mv = self.disks.mv.lock().unwrap().as_ref().map(|m| m.view());
        json!({"drives": drives, "models": models, "move": mv, "measuring": measuring, "folders": set})
    }

    /// system.drive.measure: a drive's read speed, measured in the background (up to 4 GiB of its largest model files
    /// read past the page cache) and kept; refused while a load or a move reads or writes there
    pub(crate) fn drive_measure(self: &Arc<Self>, name: &str) -> Result<Value, Fail> {
        let all = self.models();
        let list = drives::drives(&all, &self.folders());
        let d = drives::find(&list, name).ok_or_else(|| (404, format!("no drive {name:?} (system.drives: {})", list.iter().map(|d| d.id.as_str()).collect::<Vec<_>>().join(", "))))?.clone();
        if let Some(m) = self.disks.measuring.lock().unwrap().as_ref() {
            return Err((409, format!("busy: {m} is being measured")));
        }
        let ms = drives::mounts();
        if let Some(r) = self.disks.running().filter(|r| r.plan.from.contains(&d.id) || r.plan.to == d.id) {
            return Err((409, format!("busy: {} is moving {} it", r.plan.model, if r.plan.to == d.id { "to" } else { "from" })));
        }
        // a model loading from it (its files read now) would share the drive with the measurement
        let states = self.model_states(&all, &self.named_cards(&all));
        let reading: Vec<String> = all.iter().filter(|m| {
            let id = m["id"].as_str().unwrap_or("");
            let st = states.iter().find(|s| s["id"] == id).map(|s| s["state"].as_str().unwrap_or("")).unwrap_or("");
            (st == "loading" || self.loading.lock().unwrap().contains_key(id)) && drives::files_of(m).iter().any(|f| d.holds(f, &ms))
        }).map(|m| m["id"].as_str().unwrap_or("").to_string()).collect();
        if !reading.is_empty() {
            return Err((409, format!("busy: {} loading from {} (measure once it has loaded)", reading.join(", "), d.id)));
        }
        let files = drives::largest_on(&all, &d, &ms);
        if files.is_empty() {
            return Err((409, format!("no model file on {} to read", d.id)));
        }
        *self.disks.measuring.lock().unwrap() = Some(d.id.clone());
        self.disks.stop_measure.store(false, Ordering::Relaxed);
        let (me, answer) = (self.clone(), json!({"drive": d.id, "measuring": true, "reads_up_to": drives::MEASURE_BYTES}));
        std::thread::spawn(move || {
            let r = drives::measure(&files, drives::MEASURE_BYTES, &me.disks.stop_measure);
            match r {
                Ok((bytes, secs)) if secs > 0.0 => {
                    let mb_s = (bytes as f64 / secs / 1e6).round();
                    me.disks.measured.lock().unwrap().insert(d.id.clone(), json!({"mb_s": mb_s, "bytes": bytes, "seconds": (secs * 100.0).round() / 100.0,
                                                                                   "at": now().round(), "file": files[0], "device": d.model, "link": d.link.as_ref().map(|l| &l.0)}));
                    me.save_state();
                }
                Ok(_) => {}
                Err(e) => {
                    let (rid, log) = me.new_run(format!("measure {}", d.id));
                    let _ = std::fs::write(log, e);
                    me.finish(rid, 1);
                }
            }
            *me.disks.measuring.lock().unwrap() = None;
        });
        Ok(answer)
    }

    /// system.move: a model's files to another drive, as a job (system.drives' move shows it); refused while the model
    /// is loaded, starting or answering, another move runs, a process has its files open, or the drive lacks the room
    pub(crate) fn model_move(self: &Arc<Self>, id: &str, name: &str) -> Result<Value, Fail> {
        let all = self.models();
        let m = all.iter().find(|m| m["id"] == id).ok_or_else(|| (404, format!("no model {id} (system.models)")))?.clone();
        if let Some(r) = self.disks.running() {
            return Err((409, format!("busy: {} is moving to {} (one move at a time)", r.plan.model, r.plan.to)));
        }
        let set = self.folders();
        let list = drives::drives(&all, &set);
        let to: Drive = drives::find(&list, name).cloned()
            .ok_or_else(|| (404, format!("no drive {name:?} (system.drives: {})", list.iter().map(|d| d.id.as_str()).collect::<Vec<_>>().join(", "))))?;
        let state = self.model_states(std::slice::from_ref(&m), &self.named_cards(&all)).first().map(|x| x["state"].as_str().unwrap_or("").to_string()).unwrap_or_default();
        if let Some(why) = holds(&state).or_else(|| self.loading.lock().unwrap().contains_key(id).then_some("loading: wait, then stop it")) {
            return Err((409, format!("{id} is {why}")));
        }
        if self.disks.measuring.lock().unwrap().is_some() {
            return Err((409, "busy: a drive is being measured".into()));
        }
        let ms = drives::mounts();
        let cards = self.o.cfg.cards();
        let plan = drives::plan(&all, id, &to, &list, &drives::roots(&set, &list), &ms, &cards).map_err(|e| (400, e))?;
        let (size, free) = drives::space(&to.mount).unwrap_or((to.size, to.free));
        drives::room(plan.need, free, size).map_err(|e| (409, format!("{}: {e}", to.id)))?;
        let open = drives::open_by(&plan.steps.iter().map(|s| s.from.clone()).collect::<Vec<_>>());
        if !open.is_empty() {
            return Err((409, format!("{id}'s files are open in {} (a server still has the model)", open.join(", "))));
        }
        let lock = drives::lock(&registry::registry(&self.o.cfg), id, &to.id).map_err(|e| (409, format!("busy: {e}")))?;
        // a model that reads its weights while it serves, to a slower drive: said (the page asked first)
        let warning = drives::streams(&m, &cards).then(|| {
            let slowest_from = plan.from.iter().filter_map(|f| list.iter().find(|d| d.id == *f)).filter_map(|d| drives::rate(d, self.measured_mb_s(&d.id))).fold(f64::INFINITY, f64::min);
            let here = drives::rate(&to, self.measured_mb_s(&to.id)).unwrap_or(0.0);
            (here < slowest_from).then(|| format!("{id} reads its weights from the disk while it serves: on {} ({here:.1} GB/s, from {slowest_from:.1}) it serves slower", to.id))
        }).flatten();
        let mv = Arc::new(drives::Move::new(plan, warning));
        *self.disks.mv.lock().unwrap() = Some(mv.clone());
        let me = self.clone();
        let job = mv.clone();
        std::thread::spawn(move || {
            let r = job.run(&*me.o.cfg);
            drop(lock);
            let (rid, log) = me.new_run(format!("move {} to {}", job.plan.model, job.plan.to));
            let _ = std::fs::write(log, match &r {
                Ok(v) => format!("moved: {} files, {:.1} GiB, {} s; removed {}, kept {}\n", v["files"], job.plan.bytes as f64 / (1u64 << 30) as f64, v["seconds"],
                                 v["removed"].as_array().map_or(0, Vec::len), v["kept"]),
                Err(e) => format!("{e}\n"),
            });
            me.finish(rid, if r.is_ok() { 0 } else { 1 });
        });
        Ok(json!({"move": mv.view()}))
    }

    /// system.move.cancel: the move under way stopped (what it copied removed, the registry as it was)
    pub(crate) fn move_cancel(&self) -> Result<Value, Fail> {
        let r = self.disks.running().ok_or_else(|| (409, "no move is under way".to_string()))?;
        r.cancel.store(true, Ordering::Relaxed);
        Ok(json!({"model": r.plan.model, "cancelling": true}))
    }
}
