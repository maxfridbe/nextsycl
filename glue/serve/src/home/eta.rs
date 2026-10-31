//! How long work takes, from what it took: the last 100 runs per shape and per model, one file a model in
//! `history/*.jsonl` beside serve.json. A shape is what makes runs of one model comparable (a picture's size and steps,
//! a clip's canvas, length and steps, how much text). Pure here; `backplane.rs` adds what is ahead in the queues.

use std::path::Path;

use serde_json::{json, Value};

/// Runs kept per shape and per model
pub(super) const KEEP: usize = 100;

/// A power of two at or above `n` (runs near each other share a shape)
fn bucket(n: u64) -> u64 {
    n.max(1).next_power_of_two()
}

/// The shape of a request: what its run time follows
pub(super) fn shape_of(method: &str, b: &Value) -> String {
    let text = |k: &str| b[k].as_str().map_or(0, |x| x.chars().count() as u64);
    let num = |k: &str| b[k].as_f64().map_or("-".to_string(), |x| x.to_string());
    let raw = |k: &str| b[k].as_str().unwrap_or("-").to_string();
    match method {
        "image.generate" | "image.edit" => format!("{method}/{}/s{}", raw("size"), num("steps")),
        "image.upscale" | "video.upscale" => format!("{method}/x{}", num("scale")),
        "audio.speak" => format!("{method}/c{}", bucket(text("input"))),
        "audio.generate" => format!("{method}/c{}/d{}", bucket(text("lyrics") + text("input")), num("duration")),
        "llm.chat" => format!("{method}/p{}/t{}", bucket(b["messages"].to_string().len() as u64), bucket(b["max_tokens"].as_u64().unwrap_or(0))),
        "video.generate" => format!("{method}/{}x{}/{}s/s{}", num("width"), num("height"), num("seconds"), num("steps")),
        _ => method.to_string(),
    }
}

/// A model's history file (a registry id with anything odd in it made plain)
fn file_of(dir: &Path, model: &str) -> std::path::PathBuf {
    let safe: String = model.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    dir.join(format!("{safe}.jsonl"))
}

/// A model's rows, oldest first
pub(super) fn rows(dir: &Path, model: &str) -> Vec<Value> {
    std::fs::read_to_string(file_of(dir, model)).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// Of the rows, those to keep: the last KEEP of each shape and the last KEEP of the model
pub(super) fn trimmed(all: Vec<Value>) -> Vec<Value> {
    let n = all.len();
    let mut per: std::collections::BTreeMap<String, usize> = Default::default();
    let mut keep = vec![false; n];
    for (i, r) in all.iter().enumerate().rev() {
        let c = per.entry(r["shape"].as_str().unwrap_or("").to_string()).or_insert(0);
        *c += 1;
        keep[i] = *c <= KEEP || n - i <= KEEP;
    }
    all.into_iter().zip(keep).filter(|(_, k)| *k).map(|(r, _)| r).collect()
}

/// A row added to a model's file (a failure to write loses a sample, nothing else)
pub(super) fn append(dir: &Path, model: &str, row: Value) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut all = rows(dir, model);
    all.push(row);
    let text: String = trimmed(all).iter().map(|r| format!("{r}\n")).collect();
    let f = file_of(dir, model);
    let tmp = f.with_extension("jsonl.tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, &f);
    }
}

fn median(v: &mut [f64]) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    (!v.is_empty()).then(|| v[v.len() / 2])
}

/// What the rows say of a run of `shape`: its time (their median), how many runs say so, whether they are of the shape
/// or only of the model, and how far they scatter (the middle 80 %'s width)
#[derive(Debug, PartialEq)]
pub(super) struct Estimate {
    pub run_s: Option<f64>,
    pub samples: usize,
    pub basis: &'static str,
    pub spread: Option<f64>,
}

pub(super) fn estimate(all: &[Value], shape: &str) -> Estimate {
    let runs = |only: bool| -> Vec<f64> { all.iter().filter(|r| !only || r["shape"] == shape).filter_map(|r| r["run_s"].as_f64()).collect() };
    let (mut v, basis) = match runs(true) {
        v if !v.is_empty() => (v, "shape"),
        _ => match runs(false) {
            v if !v.is_empty() => (v, "model"),
            _ => return Estimate { run_s: None, samples: 0, basis: "none", spread: None },
        },
    };
    let samples = v.len();
    let run_s = median(&mut v);
    let spread = v[samples * 9 / 10] - v[samples / 10];
    Estimate { run_s, samples, basis, spread: Some(spread) }
}

/// How long the model takes to load, from the rows that say (their median)
pub(super) fn load_s(all: &[Value]) -> Option<f64> {
    let mut v: Vec<f64> = all.iter().filter_map(|r| r["load_s"].as_f64()).collect();
    median(&mut v)
}

fn tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// The answer's `eta`: wait_s null when it is not known
pub(super) fn eta_json(e: &Estimate, wait_s: Option<f64>, shape: &str) -> Value {
    let wait = wait_s.filter(|_| e.samples > 0);
    json!({"wait_s": wait.map(tenth), "run_s": e.run_s.map(tenth), "total_s": wait.zip(e.run_s).map(|(w, r)| tenth(w + r)),
           "samples": e.samples, "basis": e.basis, "spread": e.spread.map(tenth), "shape": shape})
}

/// Whether a request with this limit on its time to start goes on: a hot model is never declined, an unknown wait
/// proceeds (the job is withdrawn at the limit)
pub(super) fn would_accept(hot: bool, wait_s: Option<f64>, limit: Option<f64>) -> bool {
    hot || limit.is_none_or(|l| wait_s.is_none_or(|w| w <= l))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(shape: &str, run: f64) -> Value {
        json!({"shape": shape, "run_s": run})
    }

    #[test]
    fn a_shape_answers_first_then_the_model_then_nothing() {
        let all = vec![row("a", 10.0), row("a", 20.0), row("a", 30.0), row("b", 100.0)];
        let e = estimate(&all, "a");
        assert_eq!((e.run_s, e.samples, e.basis), (Some(20.0), 3, "shape"));
        let e = estimate(&all, "c");
        assert_eq!((e.samples, e.basis), (4, "model"));
        assert_eq!(estimate(&[], "a"), Estimate { run_s: None, samples: 0, basis: "none", spread: None });
    }

    #[test]
    fn the_last_hundred_of_a_shape_and_of_the_model_stay() {
        let mut all: Vec<Value> = (0..150).map(|i| row("rare", i as f64)).collect();
        all.extend((0..150).map(|i| row("often", i as f64)));
        let t = trimmed(all);
        assert_eq!(t.iter().filter(|r| r["shape"] == "rare").count(), KEEP, "an older shape keeps its own hundred");
        assert_eq!(t.iter().filter(|r| r["shape"] == "often").count(), KEEP);
        assert_eq!(t.last().unwrap()["run_s"], 149.0, "the newest stay");
    }

    #[test]
    fn history_goes_to_a_file_a_model_and_comes_back() {
        let dir = std::env::temp_dir().join(format!("nextsycl-eta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        append(&dir, "m/odd", row("a", 1.5));
        append(&dir, "m/odd", json!({"shape": "load", "load_s": 40.0}));
        assert!(dir.join("m_odd.jsonl").is_file());
        let back = rows(&dir, "m/odd");
        assert_eq!(back.len(), 2);
        assert_eq!(load_s(&back), Some(40.0));
        assert_eq!(estimate(&back, "a").run_s, Some(1.5), "a load row is no run");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_history_means_the_wait_is_unknown() {
        let none = estimate(&[], "a");
        let v = eta_json(&none, Some(0.0), "a");
        assert_eq!((v["wait_s"].clone(), v["total_s"].clone(), v["basis"].clone()), (Value::Null, Value::Null, json!("none")));
        let some = estimate(&[row("a", 10.0)], "a");
        let v = eta_json(&some, Some(5.0), "a");
        assert_eq!((v["wait_s"].clone(), v["run_s"].clone(), v["total_s"].clone(), v["samples"].clone()), (json!(5.0), json!(10.0), json!(15.0), json!(1)));
    }

    #[test]
    fn who_is_declined() {
        assert!(!would_accept(false, Some(100.0), Some(10.0)), "a known wait over the limit");
        assert!(would_accept(false, Some(5.0), Some(10.0)));
        assert!(would_accept(false, None, Some(10.0)), "unknown: it proceeds, withdrawn at the limit");
        assert!(would_accept(true, Some(100.0), Some(10.0)), "a hot model never");
        assert!(would_accept(false, Some(100.0), None), "no limit");
    }

    #[test]
    fn shapes_tell_runs_apart() {
        let a = shape_of("image.generate", &json!({"size": "1024x768", "steps": 8}));
        assert_eq!(a, "image.generate/1024x768/s8");
        assert_ne!(a, shape_of("image.generate", &json!({"size": "1024x768", "steps": 20})));
        assert_eq!(shape_of("audio.speak", &json!({"input": "abc"})), shape_of("audio.speak", &json!({"input": "abcd"})), "near lengths share one");
        assert_eq!(shape_of("video.generate", &json!({"width": 768, "height": 576, "seconds": 5, "steps": 8})), "video.generate/768x576/5s/s8");
    }
}
