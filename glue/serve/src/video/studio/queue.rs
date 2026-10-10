//! The clip queue: what a request becomes (a queue item), holds (all, per project, per batch), projects, and the
//! files they persist to. The item shape is the legacy server's, so scene files and tools written for it work
//! unchanged (docs/LEGACY-API.md).

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::{RpcError, Studio};

pub const MAGIC: [&str; 4] = ["prev", "prev_cam", "first", "first_cam"];
pub const CHAIN_MODES: [(&str, &str); 4] = [
    ("video", "the previous clip's mp4: its last frame decoded and encoded again"),
    ("png", "the previous clip's lossless last frame (.last.png)"),
    ("latent", "the previous clip's last latent frame (no decode, no codec)"),
    ("none", "no chaining"),
];

/// The holds and the queue, persisted.
#[derive(Default)]
pub struct Queue {
    pub items: Vec<Value>,
    pub paused: bool,
    pub paused_projects: BTreeSet<String>,
    pub paused_batches: BTreeSet<String>,
}

pub fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

/// "Name 03/12: ..." -> "Name": the series a clip belongs to (the text before the clip number).
pub fn label_prefix(label: &str) -> String {
    let bytes = label.as_bytes();
    for (i, c) in label.char_indices() {
        if c.is_ascii_digit() {
            // digits, then '/'
            let j = label[i..].find(|ch: char| !ch.is_ascii_digit()).map_or(label.len(), |k| i + k);
            if j < bytes.len() && bytes[j] == b'/' && (i == 0 || !label[..i].ends_with(|ch: char| ch.is_ascii_digit())) {
                return label[..i].trim().to_string();
            }
        }
    }
    String::new()
}

/// "Name 03/12: ..." -> 3: the edit order.
pub fn label_number(label: &str) -> Option<u64> {
    for (i, c) in label.char_indices() {
        if c.is_whitespace() {
            let rest = &label[i + c.len_utf8()..];
            let digits: String = rest.chars().take_while(|d| d.is_ascii_digit()).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with('/') && rest[digits.len() + 1..].starts_with(|d: char| d.is_ascii_digit()) {
                return digits.parse().ok();
            }
        }
    }
    None
}

/// The project an item belongs to: its own, or its label's series.
pub fn project_of(item: &Value) -> String {
    let p = s(item, "project");
    if !p.is_empty() {
        return p;
    }
    let label = s(item, "label");
    // the reference requires "<prefix> NN/MM"
    match label_number(&label) {
        Some(_) => label_prefix(&label),
        None => String::new(),
    }
}

/// The spoken lines of a prompt ("<d>[English] ...</d>"), joined.
pub fn dialogue_of(prompt: &str) -> String {
    let mut out = Vec::new();
    let mut rest = prompt;
    while let Some(a) = rest.find("<d>[") {
        let after = &rest[a + 4..];
        let Some(close) = after.find(']') else { break };
        let body = after[close + 1..].trim_start();
        let Some(end) = body.find("</d>") else { break };
        out.push(body[..end].split_whitespace().collect::<Vec<_>>().join(" "));
        rest = &body[end + 4..];
    }
    out.join(" / ")
}

/// The legacy label filter: word characters and ` -.,:'()!?/`, 80 at most.
fn clean_label(l: &str) -> String {
    let kept: String = l.chars().filter(|c| c.is_alphanumeric() || *c == '_' || " -.,:'()!?/".contains(*c)).collect();
    kept.chars().take(80).collect::<String>().trim().to_string()
}

pub fn check_canvas(w: i64, h: i64) -> Result<(), RpcError> {
    if w % 32 != 0 || h % 32 != 0 {
        return Err(RpcError::param("width and height must be multiples of 32"));
    }
    if !(256..=1344).contains(&w) || !(256..=1344).contains(&h) {
        return Err(RpcError::param("width and height must be between 256 and 1344"));
    }
    if w * h > 768 * 1344 {
        return Err(RpcError::param(&format!("area {} exceeds the model's 1032192 pixel cap", w * h)));
    }
    Ok(())
}

/// A number parameter: the default when absent, rejected (not clamped) outside [lo, hi].
pub fn num(p: &Value, key: &str, default: f64, lo: f64, hi: f64, int: bool) -> Result<Value, RpcError> {
    let v = match p.get(key) {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::Bool(b)) => *b as i64 as f64,
        Some(Value::String(t)) => t.trim().parse::<f64>().map_err(|_| RpcError::param(&format!("{key} must be a number")))?,
        _ => return Err(RpcError::param(&format!("{key} must be a number"))),
    };
    let v = if int { v.trunc() } else { v };
    if !(lo..=hi).contains(&v) {
        return Err(RpcError::param(&format!("{key} must be between {lo} and {hi}")));
    }
    Ok(if int { json!(v as i64) } else { json!(v) })
}

/// A request's denoiser -> its name in the front end's terms: INT8 (the default; the legacy Q8_0 was its counterpart),
/// or a GGUF form the engine was given (Q6_K, Q4_K_M).
pub fn engine_name(e: Option<&str>) -> String {
    match e.map(|e| e.trim().to_uppercase()).filter(|e| !e.is_empty()).as_deref() {
        None | Some("Q8_0") | Some("INT8") => "INT8".into(),
        Some(e) => e.to_string(),
    }
}

/// nextsycl: the front end's denoiser names (INT8, Q6_K, Q4_K_M) and the registry ids of the models the daemon has
/// (`--model`, `--engine`), set once by the studio's start
pub static ENGINES: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();

/// Models the daemon lists that the start did not name (named by their suffix: `quant_of`)
static LEARNED: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

/// The daemon's model ids, as it lists them: any the start did not name join under their suffix's name
pub fn learn_engines(ids: &[String]) {
    let mut l = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    for id in ids {
        let named = ENGINES.get().is_some_and(|m| m.iter().any(|(_, i)| i == id)) || l.iter().any(|(_, i)| i == id);
        let q = quant_of(id).to_string();
        if !named && !ENGINES.get().is_some_and(|m| m.iter().any(|(n, _)| *n == q)) {
            l.push((q, id.clone()));
        }
    }
}

/// Every name -> id pair: the start's, then the learned ones
fn engine_pairs() -> Vec<(String, String)> {
    let mut v = ENGINES.get().cloned().unwrap_or_default();
    v.extend(LEARNED.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned());
    v
}

/// The front end's name for a model id: by its suffix (minimax-h3 -> INT8, -q6k -> Q6_K, -q4km -> Q4_K_M, -q8 -> Q8_0,
/// -ref2va -> REF2VA)
pub fn quant_of(id: &str) -> &'static str {
    match id.rsplit('-').next().unwrap_or("") {
        "q6k" => "Q6_K",
        "q4km" => "Q4_K_M",
        "q8" => "Q8_0",
        "ref2va" => "REF2VA",
        _ => "INT8",
    }
}

/// Whether the daemon has a model of that front-end name
pub fn engine_known(name: &str) -> bool {
    engine_pairs().iter().any(|(q, _)| q == name)
}

/// A front-end name -> the daemon's model id (an unknown name passes as it is)
pub fn engine_id(name: &str) -> String {
    engine_pairs().into_iter().find(|(q, _)| q == name).map_or_else(|| name.to_string(), |(_, id)| id)
}

/// A daemon's model id -> the front end's name
pub fn engine_quant(id: &str) -> String {
    engine_pairs().into_iter().find(|(_, i)| i == id).map_or_else(|| id.to_string(), |(q, _)| q)
}

impl Studio {
    /// A request -> a queue item, checked (the legacy `build_item`, plus the engine's new options).
    pub fn build_item(&self, p: &Value) -> Result<Value, RpcError> {
        let prompt = p.get("prompt").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return Err(RpcError::param("prompt is required"));
        }
        let te = p.get("te").and_then(|v| v.as_str()).unwrap_or("teacher");
        if te != "teacher" && te != "student" {
            return Err(RpcError::param("te must be teacher or student"));
        }
        let width = num(p, "width", 768.0, 256.0, 1344.0, true)?;
        let height = num(p, "height", 576.0, 256.0, 1344.0, true)?;
        check_canvas(width.as_i64().unwrap_or(0), height.as_i64().unwrap_or(0))?;
        let chain_mode = p.get("chain_mode").and_then(|v| v.as_str()).unwrap_or("video");
        if !CHAIN_MODES.iter().any(|(m, _)| *m == chain_mode) {
            return Err(RpcError::param("chain_mode must be one of video, png, latent, none"));
        }
        // the sampler and schedule by ComfyUI's names (none: the engine's own Euler on its shifted schedule)
        let named = |k: &str, ok: &dyn Fn(&str) -> bool, names: &str| -> Result<Value, RpcError> {
            match p.get(k).and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()) {
                None => Ok(Value::Null),
                Some(v) if ok(v) => Ok(json!(v)),
                Some(v) => Err(RpcError::param(&format!("{k} {v}? ({names})"))),
            }
        };
        let sampler = named("sampler", &|v| nextsycl_diffusion::Sampler::parse(v).is_some(), &nextsycl_diffusion::samplers::NAMES.join(", "))?;
        let schedule = named("schedule", &|v| nextsycl_diffusion::Schedule::parse(v).is_some(),
                             &nextsycl_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>().join(", "))?;
        let mut it = Map::new();
        it.insert("prompt".into(), json!(prompt));
        it.insert("sampler".into(), sampler);
        it.insert("schedule".into(), schedule);
        it.insert("seconds".into(), num(p, "seconds", 10.0, 1.0, 15.1, false)?);
        it.insert("steps".into(), num(p, "steps", 10.0, 1.0, 40.0, true)?);
        it.insert("seed".into(), num(p, "seed", 0.0, -2147483648.0, 2147483648.0, true)?);
        it.insert("te".into(), json!(te));
        it.insert("label".into(), json!(clean_label(&p.get("label").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default())));
        it.insert("engine".into(), json!(engine_name(p.get("engine").and_then(|v| v.as_str()))));
        it.insert("width".into(), width);
        it.insert("height".into(), height);
        it.insert("chain_mode".into(), json!(chain_mode));
        for k in ["cond_noise_aug", "camera", "exposure_ref", "last_frame", "first_audio", "first_audio_s", "upscale", "upscaler", "guide_clip", "shift_video",
                  "shift_audio", "source", "regen", "regen_box"] {
            it.insert(k.into(), p.get(k).cloned().unwrap_or(Value::Null));
        }
        for k in ["project", "batch"] {
            let v = p.get(k).and_then(|v| v.as_str()).unwrap_or("");
            it.insert(k.into(), json!(v.chars().take(60).collect::<String>().trim()));
        }
        let loras: Vec<String> = p.get("loras").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()).chars().take(200).collect()).collect()).unwrap_or_default();
        it.insert("loras".into(), json!(loras));
        // a first frame that is not an anchor word is a file in out/
        let ff = p.get("first_frame").and_then(|v| v.as_str()).unwrap_or("");
        let ff = if ff.is_empty() {
            Value::Null
        } else if MAGIC.contains(&ff) {
            json!(ff)
        } else {
            let b = basename(ff);
            if !self.out.join(&b).exists() {
                return Err(RpcError::not_found(&format!("first_frame {b} not found")));
            }
            json!(b)
        };
        it.insert("first_frame".into(), ff);
        for (k, max, what) in [("ref_images", 9, "ref_image"), ("ref_videos", 3, "ref_video"), ("ref_audios", 3, "ref_audio")] {
            let list: Vec<String> = p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().take(max).filter_map(|x| x.as_str()).map(basename).collect()).unwrap_or_default();
            for f in &list {
                if !self.out.join(f).exists() {
                    return Err(RpcError::not_found(&format!("{what} {f} not found in out/")));
                }
            }
            it.insert(k.into(), if list.is_empty() { Value::Null } else { json!(list) });
        }
        // pictures and clips as references: the Ref2VA denoiser, the one trained for them
        if it["ref_images"].is_array() || it["ref_videos"].is_array() {
            let _ = self.engines(); // the daemon's list (a model added since the studio started)
            if !engine_known("REF2VA") {
                return Err(RpcError::param("reference pictures and clips need the Ref2VA denoiser (nextsycl models pull minimax-h3-ref2va, then add it to NS_VIDEO_ENGINES)"));
            }
            it.insert("engine".into(), json!("REF2VA"));
        }
        for k in ["ref_video_sound", "te_pictures"] {
            it.insert(k.into(), p.get(k).filter(|v| v.is_boolean()).cloned().unwrap_or(Value::Null));
        }
        // the ControlNet: a control video, a mask over a source; files beside the clips
        for k in ["control_video", "control_mask", "control_source"] {
            let v = p.get(k).and_then(|v| v.as_str()).map(basename).filter(|v| !v.is_empty());
            if let Some(f) = &v {
                if !self.out.join(f).exists() {
                    return Err(RpcError::not_found(&format!("{k} {f} not found in out/")));
                }
            }
            it.insert(k.into(), v.map_or(Value::Null, |f| json!(f)));
        }
        if !it["control_video"].is_null() || !it["control_mask"].is_null() {
            if it["engine"] == "REF2VA" {
                return Err(RpcError::param("the ControlNet steers the text/image-to-video denoiser, not Ref2VA: references and control do not mix"));
            }
            it.insert("control_strength".into(), num(p, "control_strength", 1.0, 0.0, 10.0, false)?);
            it.insert("control_start".into(), num(p, "control_start", 0.0, 0.0, 1.0, false)?);
            it.insert("control_end".into(), num(p, "control_end", 1.0, 0.0, 1.0, false)?);
        }
        let ris = p.get("ref_image_size").and_then(|v| v.as_str()).filter(|v| *v == "match" || *v == "max");
        it.insert("ref_image_size".into(), ris.map_or(Value::Null, |v| json!(v)));
        Ok(Value::Object(it))
    }
}

pub fn basename(p: &str) -> String {
    Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
}

impl Queue {
    pub fn held(&self, item: &Value) -> bool {
        let p = project_of(item);
        let b = s(item, "batch");
        self.paused || (!p.is_empty() && self.paused_projects.contains(&p)) || (!b.is_empty() && self.paused_batches.contains(&b))
    }

    pub fn load(dir: &Path) -> Queue {
        let items = std::fs::read(dir.join("queue.json")).ok().and_then(|b| serde_json::from_slice::<Vec<Value>>(&b).ok()).unwrap_or_default();
        let p: Value = std::fs::read(dir.join("paused.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let set = |k: &str| -> BTreeSet<String> { p.get(k).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        Queue { items, paused: p.get("all").and_then(|v| v.as_bool()).unwrap_or(false), paused_projects: set("projects"), paused_batches: set("batches") }
    }

    pub fn save(&self, dir: &Path) {
        let _ = super::write_atomic(&dir.join("queue.json"), &serde_json::to_vec(&self.items).unwrap_or_default());
    }

    pub fn save_paused(&self, dir: &Path) {
        let v = json!({"all": self.paused, "projects": self.paused_projects, "batches": self.paused_batches});
        let _ = super::write_atomic(&dir.join("paused.json"), &serde_json::to_vec(&v).unwrap_or_default());
    }

    /// The queued projects in queue order, with their batches.
    pub fn projects(&self) -> Vec<Value> {
        let mut order: Vec<String> = Vec::new();
        for it in &self.items {
            let p = project_of(it);
            if !order.contains(&p) {
                order.push(p);
            }
        }
        order
            .iter()
            .map(|p| {
                let mine: Vec<&Value> = self.items.iter().filter(|i| project_of(i) == *p).collect();
                let first = self.items.iter().position(|i| project_of(i) == *p).unwrap_or(0);
                let mut batches: Vec<String> = mine.iter().map(|i| s(i, "batch")).collect();
                batches.sort();
                batches.dedup();
                let secs = |v: &[&Value]| (v.iter().map(|i| i["seconds"].as_f64().unwrap_or(0.0)).sum::<f64>() * 100.0).round() / 100.0;
                json!({"project": p, "clips": mine.len(), "video_s": secs(&mine), "paused": self.paused_projects.contains(p), "first_index": first,
                       "batches": batches.iter().map(|b| {
                           let bi: Vec<&Value> = mine.iter().copied().filter(|i| s(i, "batch") == *b).collect();
                           json!({"batch": b, "clips": bi.len(), "video_s": secs(&bi), "paused": self.paused_batches.contains(b)})
                       }).collect::<Vec<_>>()})
            })
            .collect()
    }

    pub fn payload(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("queue".into(), json!(self.items.iter().map(|i| { let l = s(i, "label"); if l.is_empty() { s(i, "prompt").chars().take(40).collect() } else { l } }).collect::<Vec<_>>()));
        m.insert("queue_items".into(), json!(self.items.iter().map(|i| json!({
            "label": s(i, "label"), "seconds": i["seconds"], "steps": i["steps"], "engine": i["engine"], "chain": !i["first_frame"].is_null(),
            "size": format!("{}x{}", i["width"].as_i64().unwrap_or(640), i["height"].as_i64().unwrap_or(480)),
            "project": project_of(i), "batch": s(i, "batch"), "held": self.held(i)})).collect::<Vec<_>>()));
        m.insert("paused".into(), json!(self.paused));
        m.insert("paused_projects".into(), json!(self.paused_projects));
        m.insert("paused_batches".into(), json!(self.paused_batches));
        m.insert("projects".into(), json!(self.projects()));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(label_prefix("Harbour Night 03/12: the lamp"), "Harbour Night");
        assert_eq!(label_number("Harbour Night 03/12: the lamp"), Some(3));
        assert_eq!(label_number("no number"), None);
        assert_eq!(project_of(&json!({"label": "Speech 07/20: words"})), "Speech");
        assert_eq!(project_of(&json!({"label": "x", "project": "P"})), "P");
        assert_eq!(dialogue_of("says: <d>[English] First   batch.</d> then <d>[English] More.</d>"), "First batch. / More.");
        assert_eq!(clean_label("a<b>c 01/02: ok"), "abc 01/02: ok");
    }
}
