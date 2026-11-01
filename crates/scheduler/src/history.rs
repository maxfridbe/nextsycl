//! What finished work took, and the ETA made from it.
//!
//! A rolling history, the last [`HISTORY_KEEP`] (100) work requests per key. Each request is stored under two keys: its
//! *shape* (`shape|<method|model|size-driving parameters>`, see [`crate::shape`]) and its *model* (`model|<id>`). The
//! host decides where the lines go with a [`Storage`]: [`FileStorage`] (a JSON line per request in `<key>.jsonl` under a
//! directory, appended by a writer thread - never on the request path - and cut back to the last 100 when a file has
//! grown to twice that) or [`MemStorage`] (tests, or a box that keeps nothing).
//!
//! An ETA is `{wait_s, run_s, total_s, samples, basis, spread, shape}`: run_s the median of the shape's ok runs (else the
//! model's, else none: no figure is invented), wait_s until the request would start (0 when its model is hot and has a
//! slot free; else the work ahead plus the model's spin-up from history; unknown when any part is).

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::shape::Shapes;

/// Requests kept per key
pub const HISTORY_KEEP: usize = 100;

/// One finished work request
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub method: String,
    pub model: String,
    pub shape: String,
    /// seconds from the request to its start (queue, room, spin-up)
    pub wait_s: f64,
    /// seconds it ran (only an ok run feeds run statistics)
    pub run_s: f64,
    /// ok | failed | cancelled
    pub outcome: String,
}

impl Record {
    pub fn to_line(&self) -> String {
        json!({"method": self.method, "model": self.model, "shape": self.shape, "wait_s": self.wait_s, "run_s": self.run_s, "outcome": self.outcome}).to_string()
    }
    pub fn from_line(l: &str) -> Option<Record> {
        let v: Value = serde_json::from_str(l).ok()?;
        let st = |k: &str| v[k].as_str().map(str::to_string);
        Some(Record { method: st("method")?, model: st("model")?, shape: st("shape")?, wait_s: v["wait_s"].as_f64()?, run_s: v["run_s"].as_f64()?, outcome: st("outcome")? })
    }
}

/// Where the history's lines are kept (the host's: a directory, a database, nowhere)
pub trait Storage: Send + Sync {
    /// Every key and its lines, oldest first (read once, at open)
    fn load(&self) -> Vec<(String, Vec<String>)>;
    /// One more line under a key (off the request path: the history calls it from its writer)
    fn append(&self, key: &str, line: &str);
    /// Wait until what was appended is kept
    fn sync(&self) {}
}

/// Nothing kept past the process, but shareable between two openings (a restart in a test)
#[derive(Clone, Default)]
pub struct MemStorage(pub Arc<Mutex<BTreeMap<String, Vec<String>>>>);

impl Storage for MemStorage {
    fn load(&self) -> Vec<(String, Vec<String>)> {
        self.0.lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
    fn append(&self, key: &str, line: &str) {
        self.0.lock().unwrap().entry(key.to_string()).or_default().push(line.to_string());
    }
}

enum Msg {
    Line(String, String),
    Sync(Sender<()>),
}

/// A directory of `<key>.jsonl` files: the first line of a file is `#key <key>`; a writer thread appends and cuts a file
/// back to the last HISTORY_KEEP lines when it has twice that
pub struct FileStorage {
    dir: PathBuf,
    tx: Mutex<Sender<Msg>>,
}

fn file_name(key: &str) -> String {
    let safe: String = key.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.=x".contains(c) { c } else { '_' }).collect();
    // (two keys that differ only in characters made `_` must not share a file: a short hash of the key)
    let h = key.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    format!("{}-{h:016x}.jsonl", safe.chars().take(80).collect::<String>())
}

fn append_file(dir: &Path, key: &str, line: &str) {
    use std::io::Write;
    let path = dir.join(file_name(key));
    let fresh = !path.exists();
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) else { return };
    if fresh {
        let _ = writeln!(f, "#key {key}");
    }
    let _ = writeln!(f, "{line}");
    drop(f);
    let n = std::fs::read_to_string(&path).map_or(0, |t| t.lines().count().saturating_sub(1));
    if n >= HISTORY_KEEP * 2 {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let lines: Vec<&str> = text.lines().skip(1).collect();
        let keep = &lines[lines.len().saturating_sub(HISTORY_KEEP)..];
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, format!("#key {key}\n{}\n", keep.join("\n"))).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

impl FileStorage {
    pub fn new(dir: PathBuf) -> FileStorage {
        let _ = std::fs::create_dir_all(&dir);
        let (tx, rx) = channel::<Msg>();
        let d = dir.clone();
        std::thread::spawn(move || {
            for m in rx {
                match m {
                    Msg::Sync(ack) => {
                        let _ = ack.send(());
                    }
                    Msg::Line(key, line) => append_file(&d, &key, &line),
                }
            }
        });
        FileStorage { dir, tx: Mutex::new(tx) }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Storage for FileStorage {
    fn load(&self) -> Vec<(String, Vec<String>)> {
        let mut out = Vec::new();
        for f in std::fs::read_dir(&self.dir).into_iter().flatten().flatten() {
            let text = std::fs::read_to_string(f.path()).unwrap_or_default();
            let mut lines = text.lines();
            if let Some(key) = lines.next().and_then(|l| l.strip_prefix("#key ")) {
                out.push((key.to_string(), lines.map(str::to_string).collect()));
            }
        }
        out
    }
    fn append(&self, key: &str, line: &str) {
        let _ = self.tx.lock().unwrap().send(Msg::Line(key.to_string(), line.to_string()));
    }
    fn sync(&self) {
        let (a, b) = channel();
        if self.tx.lock().unwrap().send(Msg::Sync(a)).is_ok() {
            let _ = b.recv();
        }
    }
}

pub struct History {
    by_key: Mutex<BTreeMap<String, VecDeque<Record>>>,
    storage: Box<dyn Storage>,
    shapes: Shapes,
    /// the methods whose record is the whole time, loading included: with history, nothing is added for a spin-up
    whole_runs: Vec<String>,
}

impl History {
    /// The history in `storage`, read
    pub fn open(storage: Box<dyn Storage>) -> History {
        History::open_with(storage, Shapes::standard())
    }

    /// With a host's own shape rules
    pub fn open_with(storage: Box<dyn Storage>, shapes: Shapes) -> History {
        let mut by_key: BTreeMap<String, VecDeque<Record>> = BTreeMap::new();
        for (key, lines) in storage.load() {
            let q = by_key.entry(key).or_default();
            for r in lines.iter().filter_map(|l| Record::from_line(l)) {
                q.push_back(r);
                if q.len() > HISTORY_KEEP {
                    q.pop_front();
                }
            }
        }
        History { by_key: Mutex::new(by_key), storage, shapes, whole_runs: Vec::new() }
    }

    /// A method whose records hold its whole time, loading included (a clip's, read from the studio's records)
    pub fn whole_run(mut self, method: &str) -> History {
        self.whole_runs.push(method.to_string());
        self
    }

    pub fn shapes(&self) -> &Shapes {
        &self.shapes
    }

    /// A finished request, under its shape and its model: in memory now, in the storage by its writer
    pub fn record(&self, r: Record) {
        let keys = [format!("shape|{}", r.shape), format!("model|{}", r.model)];
        let mut m = self.by_key.lock().unwrap();
        for k in keys {
            let q = m.entry(k.clone()).or_default();
            q.push_back(r.clone());
            if q.len() > HISTORY_KEEP {
                q.pop_front();
            }
            self.storage.append(&k, &r.to_line());
        }
    }

    /// Record a finished request from its parts (the shape is made here)
    pub fn record_work(&self, method: &str, body: &Value, model: &str, wait_s: f64, run_s: f64, outcome: &str) {
        if model.is_empty() {
            return;
        }
        self.record(Record { method: method.into(), model: model.into(), shape: self.shapes.shape(method, model, body), wait_s, run_s, outcome: outcome.into() });
    }

    /// Wait until what was recorded is in the storage
    pub fn sync(&self) {
        self.storage.sync();
    }

    /// The records under a key, oldest first
    pub fn of(&self, key: &str) -> Vec<Record> {
        self.by_key.lock().unwrap().get(key).map(|q| q.iter().cloned().collect()).unwrap_or_default()
    }

    /// The ETA for a request of this shape and model; `hot`: the model is loaded with a slot free; `ahead_s`: the
    /// work ahead of it on its server (None: some of it has no estimate)
    pub fn eta(&self, method: &str, model: &str, params: &Value, hot: bool, ahead_s: Option<f64>) -> Value {
        let shape = self.shapes.shape(method, model, params);
        let runs = |recs: Vec<Record>| -> Vec<f64> { recs.into_iter().filter(|r| r.outcome == "ok").map(|r| r.run_s).collect() };
        let by_shape = runs(self.of(&format!("shape|{shape}")));
        let by_model = runs(self.of(&format!("model|{model}")));
        let (basis, samples) = if !by_shape.is_empty() { ("shape", by_shape) } else if !by_model.is_empty() { ("model", by_model) } else { ("none", Vec::new()) };
        let run_s = median(&samples);
        let spread = if samples.len() >= 4 { Some((percentile(&samples, 0.25), percentile(&samples, 0.75))) } else {
            samples.iter().copied().reduce(f64::min).zip(samples.iter().copied().reduce(f64::max))
        };
        // wait: nothing when hot; else the work ahead and the model's spin-up (waits of its requests that had to start
        // it: all of them, failed and cancelled included, over 2 s)
        let spin: Vec<f64> = self.of(&format!("model|{model}")).into_iter().map(|r| r.wait_s).filter(|w| *w > 2.0).collect();
        let wait_s = if hot { Some(0.0) } else if !samples.is_empty() && self.whole_runs.iter().any(|m| m == method) { ahead_s } else { ahead_s.and_then(|a| if spin.is_empty() { None } else { Some(a + median(&spin).unwrap_or(0.0)) }) };
        let total_s = wait_s.zip(run_s).map(|(w, r)| w + r);
        json!({"wait_s": wait_s, "run_s": run_s, "total_s": total_s, "samples": samples.len(), "basis": basis,
               "spread": spread.map(|(a, b)| json!([a, b])), "shape": shape})
    }
}

pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    Some(percentile(v, 0.5))
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let i = ((s.len() - 1) as f64 * p).round() as usize;
    s[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("scheduler-hist-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn rec(shape: &str, run_s: f64, outcome: &str, wait_s: f64) -> Record {
        Record { method: "image.generate".into(), model: "m".into(), shape: shape.into(), wait_s, run_s, outcome: outcome.into() }
    }

    fn mem() -> History {
        History::open(Box::new(MemStorage::default()))
    }

    #[test]
    fn only_the_last_hundred_per_key_are_kept_and_they_survive_a_restart_in_files() {
        let d = tmp("keep");
        let h = History::open(Box::new(FileStorage::new(d.clone())));
        for i in 0..250 {
            h.record(rec("s", i as f64, "ok", 0.0));
        }
        let r = h.of("shape|s");
        assert_eq!(r.len(), HISTORY_KEEP);
        assert_eq!(r[0].run_s, 150.0);
        h.sync();
        let again = History::open(Box::new(FileStorage::new(d.clone())));
        let r = again.of("shape|s");
        assert_eq!(r.len(), HISTORY_KEEP);
        assert_eq!(r.last().unwrap().run_s, 249.0);
        assert_eq!(again.of("model|m").len(), HISTORY_KEEP);
        let lines = std::fs::read_dir(&d).unwrap().flatten().map(|f| std::fs::read_to_string(f.path()).unwrap().lines().count()).max().unwrap();
        assert!(lines <= HISTORY_KEEP * 2 + 1, "{lines}");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_memory_storage_survives_a_reopen_too() {
        let st = MemStorage::default();
        let h = History::open(Box::new(st.clone()));
        for i in 0..130 {
            h.record(rec("s", i as f64, "ok", 0.0));
        }
        let again = History::open(Box::new(st));
        assert_eq!(again.of("shape|s").len(), HISTORY_KEEP);
        assert_eq!(again.of("shape|s").last().unwrap().run_s, 129.0);
    }

    #[test]
    fn the_eta_comes_from_the_shape_then_the_model_then_none() {
        let h = mem();
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].is_null(), eta["total_s"].is_null()), (Some("none"), true, true), "{eta}");
        h.record(rec("image.generate|m|w=512,h=512,steps=28", 10.0, "ok", 0.0));
        h.record(rec("image.generate|m|w=512,h=512,steps=28", 20.0, "ok", 0.0));
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64(), eta["samples"].as_u64()), (Some("model"), Some(20.0), Some(2)));
        for r in [40.0, 50.0, 60.0] {
            h.record(rec("image.generate|m|w=1024,h=1024,steps=28", r, "ok", 0.0));
        }
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64(), eta["wait_s"].as_f64(), eta["total_s"].as_f64()), (Some("shape"), Some(50.0), Some(0.0), Some(50.0)));
        assert_eq!(eta["spread"], json!([40.0, 60.0]));
    }

    #[test]
    fn failed_and_cancelled_runs_do_not_feed_run_time_but_their_waits_count() {
        let h = mem();
        let s = "image.generate|m|w=1024,h=1024,steps=28";
        h.record(rec(s, 1.0, "failed", 30.0));
        h.record(rec(s, 2.0, "cancelled", 50.0));
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        let eta = h.eta("image.generate", "m", &p, false, Some(0.0));
        assert_eq!(eta["basis"], "none", "{eta}");
        assert!(eta["wait_s"].as_f64().unwrap() >= 30.0, "{eta}");
        h.record(rec(s, 40.0, "ok", 0.0));
        assert_eq!(h.eta("image.generate", "m", &p, true, Some(0.0))["run_s"], 40.0);
    }

    #[test]
    fn a_hot_model_waits_nothing_and_work_ahead_adds_its_time() {
        let h = mem();
        h.record(rec("image.generate|m|w=1024,h=1024,steps=28", 30.0, "ok", 20.0));
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        assert_eq!(h.eta("image.generate", "m", &p, true, Some(99.0))["wait_s"], 0.0);
        let busy = h.eta("image.generate", "m", &p, false, Some(45.0));
        assert_eq!(busy["wait_s"].as_f64(), Some(65.0), "ahead 45 + the model's spin-up 20: {busy}");
        assert!(h.eta("image.generate", "m", &p, false, None)["wait_s"].is_null());
    }
}
