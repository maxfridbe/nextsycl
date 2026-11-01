//! What finished work took: crates/scheduler's rolling history and ETA (`history`, `shape`), in `history/` beside
//! serve.json, with this box's two differences: a picture's `size` ("1024x768") is its width and height, and a clip's
//! record is its whole time, its engine's load in it. The files are the ones the door has always written
//! (`<key>.jsonl`, `#key <key>` first, a JSON line per request).

use std::path::PathBuf;

use backplane_scheduler::history::FileStorage;
pub use backplane_scheduler::history::{History, Record};
use backplane_scheduler::shape::Shapes;
use serde_json::Value;

/// A picture: size to 64 (a request's `size` when it has no width and height), steps to 4
fn picture(p: &Value) -> Vec<(&'static str, i64)> {
    let num = |k: &str| p[k].as_f64().or_else(|| p[k].as_str().and_then(|s| s.parse().ok()));
    let near = |n: f64, step: f64| ((n / step).round().max(1.0) * step) as i64;
    let size = p["size"].as_str().and_then(|x| x.split_once('x')).and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)));
    let mut v = Vec::new();
    v.extend(num("width").or(size.map(|x| x.0)).map(|x| ("w", near(x, 64.0))));
    v.extend(num("height").or(size.map(|x| x.1)).map(|x| ("h", near(x, 64.0))));
    v.extend(num("steps").map(|x| ("steps", near(x, 4.0))));
    v
}

fn shapes() -> Shapes {
    Shapes::standard().with("image.generate", picture).with("image.edit", picture)
}

/// The history in `dir`, read; its writer thread started
pub fn open(dir: PathBuf) -> History {
    History::open_with(Box::new(FileStorage::new(dir)), shapes()).whole_run("video.generate")
}

/// The shape of a request
pub fn shape_of(method: &str, model: &str, p: &Value) -> String {
    shapes().shape(method, model, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nextsycl-hist-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_pictures_size_is_its_width_and_height_and_a_clips_wait_needs_no_spin_up() {
        assert_eq!(shape_of("image.generate", "q", &json!({"size": "1020x768", "steps": 8})), "image.generate|q|w=1024,h=768,steps=8");
        assert_eq!(shape_of("image.generate", "q", &json!({"width": 1020, "height": 1024, "steps": 31})), "image.generate|q|w=1024,h=1024,steps=32");
        assert_eq!(shape_of("image.generate", "q", &json!({"prompt": "x"})), "image.generate|q");
        let d = tmp("clip");
        let h = open(d.clone());
        h.record(Record { method: "video.generate".into(), model: "v".into(), shape: "video.generate|v".into(), wait_s: 0.0, run_s: 100.0, outcome: "ok".into() });
        assert_eq!(h.eta("video.generate", "v", &json!({}), false, Some(30.0))["wait_s"], 30.0, "known with its engine cold");
        assert!(h.eta("video.generate", "other", &json!({}), false, Some(30.0))["wait_s"].is_null(), "no history: unknown");
        let _ = std::fs::remove_dir_all(d);
    }

    /// A file as the door before the scheduler crate wrote it, name and all, with a clips.seen beside it: read, and appended to in place
    #[test]
    fn the_files_already_on_the_box_keep_working() {
        let d = tmp("old");
        std::fs::create_dir_all(&d).unwrap();
        let key = "shape|image.generate|m|w=1024,h=1024,steps=8";
        let line = json!({"method": "image.generate", "model": "m", "shape": "image.generate|m|w=1024,h=1024,steps=8", "wait_s": 40.0, "run_s": 12.5, "outcome": "ok"});
        // (the old file_name: the key's safe characters, then its FNV hash)
        let name = "shape_image.generate_m_w=1024_h=1024_steps=8-".to_string() + &format!("{:016x}", key.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))) + ".jsonl";
        std::fs::write(d.join(&name), format!("#key {key}\n{line}\n")).unwrap();
        std::fs::write(d.join("clips.seen"), "a.mp4\nb.mp4").unwrap();
        let h = open(d.clone());
        let p = json!({"size": "1024x1024", "steps": 8});
        let eta = h.eta("image.generate", "m", &p, true, Some(0.0));
        assert_eq!((eta["basis"].as_str(), eta["run_s"].as_f64()), (Some("shape"), Some(12.5)), "{eta}");
        h.record(Record { method: "image.generate".into(), model: "m".into(), shape: shape_of("image.generate", "m", &p), wait_s: 0.0, run_s: 20.0, outcome: "ok".into() });
        h.sync();
        assert_eq!(std::fs::read_to_string(d.join(&name)).unwrap().lines().count(), 3, "the same file, one line more");
        let _ = std::fs::remove_dir_all(d);
    }
}
