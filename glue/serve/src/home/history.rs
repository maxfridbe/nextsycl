//! What finished work took, and the ETA made from it.
//!
//! A rolling history, the last HISTORY_KEEP (100) work requests per key, kept by the front door beside its registry
//! (`history/` next to serve.json): a JSON line per request in `<key>.jsonl`, appended by a writer thread (never on
//! the request path) and cut back to the last 100 when a file has grown to twice that. Each request is written under
//! two keys, its *shape* (method + model + the parameters that drive its size, bucketed: `shape_of`) and its *model*
//! (`model|<id>`). Read back at start.
//!
//! An ETA is `{wait_s, run_s, total_s, samples, basis, spread}`: run_s the median of the shape's ok runs (else the
//! model's, else none: no number is invented), wait_s until the request would start (0 when its model is hot and has a
//! slot free; else the work ahead on its server plus the model's spin-up from history; unknown when any part is).

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;

use serde_json::{json, Value};

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
    fn to_line(&self) -> String {
        json!({"method": self.method, "model": self.model, "shape": self.shape, "wait_s": self.wait_s, "run_s": self.run_s, "outcome": self.outcome}).to_string()
    }
    fn from_line(l: &str) -> Option<Record> {
        let v: Value = serde_json::from_str(l).ok()?;
        let st = |k: &str| v[k].as_str().map(str::to_string);
        Some(Record { method: st("method")?, model: st("model")?, shape: st("shape")?, wait_s: v["wait_s"].as_f64()?, run_s: v["run_s"].as_f64()?, outcome: st("outcome")? })
    }
}

enum Msg {
    Line(String, String),
    #[cfg(test)]
    Sync(Sender<()>),
}

pub struct History {
    dir: PathBuf,
    by_key: Mutex<BTreeMap<String, VecDeque<Record>>>,
    tx: Mutex<Sender<Msg>>,
}

fn file_name(key: &str) -> String {
    let safe: String = key.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.=x".contains(c) { c } else { '_' }).collect();
    // (two keys that differ only in characters made `_` must not share a file: a short hash of the key)
    let h = key.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    format!("{}-{h:016x}.jsonl", safe.chars().take(80).collect::<String>())
}

impl History {
    /// The history in `dir`, read; its writer thread started
    pub fn open(dir: PathBuf) -> History {
        let _ = std::fs::create_dir_all(&dir);
        let mut by_key: BTreeMap<String, VecDeque<Record>> = BTreeMap::new();
        for f in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let text = std::fs::read_to_string(f.path()).unwrap_or_default();
            let mut lines = text.lines();
            // the first line of a file is its key
            let Some(key) = lines.next().and_then(|l| l.strip_prefix("#key ")).map(str::to_string) else { continue };
            let q = by_key.entry(key).or_default();
            for r in lines.filter_map(Record::from_line) {
                q.push_back(r);
                if q.len() > HISTORY_KEEP {
                    q.pop_front();
                }
            }
        }
        let (tx, rx) = channel::<Msg>();
        let d = dir.clone();
        std::thread::spawn(move || {
            for m in rx {
                match m {
                    #[cfg(test)]
                    Msg::Sync(ack) => {
                        let _ = ack.send(());
                    }
                    Msg::Line(key, line) => append(&d, &key, &line),
                }
            }
        });
        History { dir, by_key: Mutex::new(by_key), tx: Mutex::new(tx) }
    }

    /// A finished request, under its shape and its model: in memory now, in the files by the writer
    pub fn record(&self, r: Record) {
        let keys = [format!("shape|{}", r.shape), format!("model|{}", r.model)];
        let mut m = self.by_key.lock().unwrap();
        for k in keys {
            let q = m.entry(k.clone()).or_default();
            q.push_back(r.clone());
            if q.len() > HISTORY_KEEP {
                q.pop_front();
            }
            let _ = self.tx.lock().unwrap().send(Msg::Line(k, r.to_line()));
        }
    }

    /// Wait until what was recorded is in the files
    #[cfg(test)]
    pub fn sync(&self) {
        let (a, b) = channel();
        if self.tx.lock().unwrap().send(Msg::Sync(a)).is_ok() {
            let _ = b.recv();
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The records under a key, oldest first
    pub fn of(&self, key: &str) -> Vec<Record> {
        self.by_key.lock().unwrap().get(key).map(|q| q.iter().cloned().collect()).unwrap_or_default()
    }

    /// The ETA for a request of this shape and model; `hot`: the model is loaded with a slot free; `ahead_s`: the
    /// work ahead of it on its server (None: some of it has no estimate)
    pub fn eta(&self, method: &str, model: &str, params: &Value, hot: bool, ahead_s: Option<f64>) -> Value {
        let shape = shape_of(method, model, params);
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
        // (a clip's record is its whole time, its engine's load in it: nothing to add to what is ahead)
        let wait_s = if hot { Some(0.0) } else if method == "video.generate" && !samples.is_empty() { ahead_s } else { ahead_s.and_then(|a| if spin.is_empty() { None } else { Some(a + median(&spin).unwrap_or(0.0)) }) };
        // (a model that has always been hot has no spin-up samples: its wait is the work ahead alone when it is loaded)
        let total_s = wait_s.zip(run_s).map(|(w, r)| w + r);
        json!({"wait_s": wait_s, "run_s": run_s, "total_s": total_s, "samples": samples.len(), "basis": basis,
               "spread": spread.map(|(a, b)| json!([a, b])), "shape": shape})
    }
}

fn append(dir: &Path, key: &str, line: &str) {
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
    // cut back to the last HISTORY_KEEP when it has grown to twice that
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

// ---- shapes: one place per kind ----

/// `n` rounded to a multiple of `step` (at least `step`)
fn near(n: f64, step: f64) -> i64 {
    ((n / step).round().max(1.0) * step) as i64
}

/// A size in powers of two: 0, then the smallest power of two at or above n
fn pow2(n: f64) -> i64 {
    if n <= 0.0 {
        0
    } else {
        (n.ceil() as u64).next_power_of_two() as i64
    }
}

fn num(p: &Value, k: &str) -> Option<f64> {
    p[k].as_f64().or_else(|| p[k].as_str().and_then(|s| s.parse().ok()))
}

fn text_len(p: &Value, k: &str) -> f64 {
    p[k].as_str().map_or(0.0, |s| s.chars().count() as f64)
}

/// The shape of a request: `method|model|size-driving parameters, bucketed`. A method with no rule, or a request
/// without the parameters, is `method|model` alone.
pub fn shape_of(method: &str, model: &str, p: &Value) -> String {
    let mut k: Vec<String> = Vec::new();
    let mut put = |name: &str, v: Option<i64>| {
        if let Some(v) = v {
            k.push(format!("{name}={v}"));
        }
    };
    match method {
        // pictures: size to 64, steps to 4
        "image.generate" | "image.edit" => {
            // (a request's `size`, "1024x768", is its width and height)
            let size = p["size"].as_str().and_then(|x| x.split_once('x')).and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)));
            let (w, h) = (num(p, "width").or(size.map(|x| x.0)), num(p, "height").or(size.map(|x| x.1)));
            put("w", w.map(|x| near(x, 64.0)));
            put("h", h.map(|x| near(x, 64.0)));
            put("steps", num(p, "steps").map(|x| near(x, 4.0)));
        }
        "image.upscale" | "video.upscale" => put("scale", num(p, "scale").map(|x| x as i64)),
        // clips: size to 64, frames to 8 (seconds at 24 fps when frames are not given), steps to 2, the engine
        "video.generate" => {
            put("w", num(p, "width").map(|x| near(x, 64.0)));
            put("h", num(p, "height").map(|x| near(x, 64.0)));
            put("frames", num(p, "frames").or_else(|| num(p, "seconds").map(|s| s * 24.0)).map(|x| near(x, 8.0)));
            put("steps", num(p, "steps").map(|x| near(x, 2.0)));
            if let Some(e) = p["engine"].as_str() {
                k.push(format!("engine={e}"));
            }
        }
        // speech: the text's length in powers of two (the voice does not change the time)
        "audio.speak" => put("text", Some(pow2(text_len(p, "text").max(text_len(p, "input"))))),
        // songs: the length asked for to 10 s, the lyrics' length in powers of two
        "audio.generate" => {
            put("secs", num(p, "duration").or_else(|| num(p, "seconds")).map(|x| near(x, 10.0)));
            put("lyrics", Some(pow2(text_len(p, "lyrics"))));
        }
        "audio.transcribe" => put("secs", num(p, "seconds").map(|x| near(x, 30.0))),
        // chat: the prompt (its messages' characters, ~4 per token) and max_tokens in powers of two
        "llm.chat" => {
            let chars: usize = p["messages"].as_array().into_iter().flatten().map(|m| m["content"].as_str().map_or(0, |c| c.chars().count())).sum();
            put("prompt", Some(pow2(chars as f64 / 4.0)));
            put("max", num(p, "max_tokens").map(|x| pow2(x)));
        }
        // 3D: its main knobs
        "model3d.generate" | "model3d.rig" => {
            put("steps", num(p, "steps").map(|x| near(x, 4.0)));
            put("res", num(p, "resolution").map(|x| x as i64));
            put("faces", num(p, "faces").map(|x| near(x, 10000.0)));
        }
        _ => {}
    }
        if k.is_empty() { format!("{method}|{model}") } else { format!("{method}|{model}|{}", k.join(",")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nextsycl-hist-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn rec(shape: &str, run_s: f64, outcome: &str, wait_s: f64) -> Record {
        Record { method: "image.generate".into(), model: "m".into(), shape: shape.into(), wait_s, run_s, outcome: outcome.into() }
    }

    #[test]
    fn only_the_last_hundred_per_key_are_kept_and_they_survive_a_restart() {
        let d = tmp("keep");
        let h = History::open(d.clone());
        for i in 0..250 {
            h.record(rec("s", i as f64, "ok", 0.0));
        }
        let r = h.of("shape|s");
        assert_eq!(r.len(), HISTORY_KEEP);
        assert_eq!(r[0].run_s, 150.0);
        h.sync();
        let again = History::open(d.clone());
        let r = again.of("shape|s");
        assert_eq!(r.len(), HISTORY_KEEP);
        assert_eq!(r.last().unwrap().run_s, 249.0);
        assert_eq!(again.of("model|m").len(), HISTORY_KEEP);
        // the files were cut back, not grown without end
        let lines = std::fs::read_dir(&d).unwrap().flatten().map(|f| std::fs::read_to_string(f.path()).unwrap().lines().count()).max().unwrap();
        assert!(lines <= HISTORY_KEEP * 2 + 1, "{lines}");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_shape_buckets_the_size_driving_parameters() {
        let a = shape_of("image.generate", "q", &json!({"width": 1020, "height": 1024, "steps": 31, "prompt": "a fox"}));
        let b = shape_of("image.generate", "q", &json!({"width": 1024, "height": 1000, "steps": 30, "prompt": "another"}));
        assert_eq!(a, "image.generate|q|w=1024,h=1024,steps=32");
        assert_eq!(a, b, "the same bucket whatever the prompt");
        assert_eq!(shape_of("image.generate", "q", &json!({"width": 512, "height": 512, "steps": 30})), "image.generate|q|w=512,h=512,steps=32");
        assert_eq!(shape_of("video.generate", "ltx2", &json!({"width": 768, "height": 512, "seconds": 5, "steps": 8, "engine": "ltx2"})),
                   "video.generate|ltx2|w=768,h=512,frames=120,steps=8,engine=ltx2");
        // speech: the text's length in powers of two, whatever the voice
        assert_eq!(shape_of("audio.speak", "t", &json!({"text": "x".repeat(300), "voice": "a"})), shape_of("audio.speak", "t", &json!({"text": "y".repeat(400), "voice": "b"})));
        assert_ne!(shape_of("audio.speak", "t", &json!({"text": "x".repeat(300)})), shape_of("audio.speak", "t", &json!({"text": "x".repeat(900)})));
        // chat: prompt and max_tokens
        assert_eq!(shape_of("llm.chat", "c", &json!({"messages": [{"role": "user", "content": "x".repeat(4000)}], "max_tokens": 700})), "llm.chat|c|prompt=1024,max=1024");
        // 3D knobs
        assert_eq!(shape_of("model3d.generate", "t", &json!({"steps": 30, "resolution": 512})), "model3d.generate|t|steps=32,res=512");
        // unknown params and unknown methods: method + model only
        assert_eq!(shape_of("image.generate", "q", &json!({"prompt": "x"})), "image.generate|q");
        assert_eq!(shape_of("audio.voices.design", "v", &json!({"width": 5})), "audio.voices.design|v");
    }

    #[test]
    fn a_pictures_size_is_its_width_and_height_and_a_clips_wait_needs_no_spin_up() {
        assert_eq!(shape_of("image.generate", "q", &json!({"size": "1020x768", "steps": 8})), "image.generate|q|w=1024,h=768,steps=8");
        let h = History::open(tmp("clip"));
        h.record(Record { method: "video.generate".into(), model: "v".into(), shape: "video.generate|v".into(), wait_s: 0.0, run_s: 100.0, outcome: "ok".into() });
        assert_eq!(h.eta("video.generate", "v", &json!({}), false, Some(30.0))["wait_s"], 30.0, "known with its engine cold");
        assert!(h.eta("video.generate", "other", &json!({}), false, Some(30.0))["wait_s"].is_null(), "no history: unknown");
    }

    #[test]
    fn the_eta_comes_from_the_shape_then_the_model_then_none() {
        let h = History::open(tmp("eta"));
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].is_null(), eta["total_s"].is_null()), (Some("none"), true, true), "{eta}");
        // a run of another shape: the model's
        h.record(rec("image.generate|m|w=512,h=512,steps=28", 10.0, "ok", 0.0));
        h.record(rec("image.generate|m|w=512,h=512,steps=28", 20.0, "ok", 0.0));
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64(), eta["samples"].as_u64()), (Some("model"), Some(20.0), Some(2)));
        // this shape's own runs win
        for r in [40.0, 50.0, 60.0] {
            h.record(rec("image.generate|m|w=1024,h=1024,steps=28", r, "ok", 0.0));
        }
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64(), eta["wait_s"].as_f64(), eta["total_s"].as_f64()), (Some("shape"), Some(50.0), Some(0.0), Some(50.0)));
        assert_eq!(eta["spread"], json!([40.0, 60.0]));
    }

    #[test]
    fn failed_and_cancelled_runs_do_not_feed_run_time_but_their_waits_count() {
        let h = History::open(tmp("failed"));
        let s = "image.generate|m|w=1024,h=1024,steps=28";
        h.record(rec(s, 1.0, "failed", 30.0));
        h.record(rec(s, 2.0, "cancelled", 50.0));
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        let eta = h.eta("image.generate", "m", &p, false, Some(0.0));
        assert_eq!(eta["basis"], "none", "{eta}");
        // the model is cold: its spin-up estimate is from the waits (30, 50 -> 50 at the upper median)
        assert!(eta["wait_s"].as_f64().unwrap() >= 30.0, "{eta}");
        h.record(rec(s, 40.0, "ok", 0.0));
        assert_eq!(h.eta("image.generate", "m", &p, true, Some(0.0))["run_s"], 40.0);
    }

    #[test]
    fn a_hot_model_waits_nothing_and_work_ahead_adds_its_time() {
        let h = History::open(tmp("wait"));
        let s = "image.generate|m|w=1024,h=1024,steps=28";
        h.record(rec(s, 30.0, "ok", 20.0));
        let p = json!({"width": 1024, "height": 1024, "steps": 28});
        // hot and free: 0, whatever is said to be ahead
        assert_eq!(h.eta("image.generate", "m", &p, true, Some(99.0))["wait_s"], 0.0);
        // loaded and busy (not "hot with a slot"): the work ahead plus no spin-up when the model needs none
        let busy = h.eta("image.generate", "m", &p, false, Some(45.0));
        assert_eq!(busy["wait_s"].as_f64(), Some(65.0), "ahead 45 + the model's spin-up 20: {busy}");
        // work ahead with no estimate: the wait is unknown, never invented
        assert!(h.eta("image.generate", "m", &p, false, None)["wait_s"].is_null());
    }
}
