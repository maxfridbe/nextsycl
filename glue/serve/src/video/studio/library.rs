//! What is on disk: the clips and their records, the joined films, thumbnails, the per-project timeline, scene
//! files (export / import), the summary and plan the front end shows, and the GPU's telemetry.

use std::path::Path;
use std::process::Command;

use serde_json::{json, Map, Value};

use super::queue::{dialogue_of, label_number, label_prefix, project_of, s, Queue};
use super::runner::snap_seconds;
use super::{now, pyround, RpcError, Studio};

/// The keys a scene file may lift into its defaults.
const SCENE_DEFAULTABLE: [&str; 28] = [
    "steps", "seed", "te", "engine", "width", "height", "chain_mode", "upscale", "first_frame", "last_frame", "exposure_ref", "first_audio",
    "first_audio_s", "cond_noise_aug", "loras", "ref_images", "ref_audios", "ref_image_size", "guide_clip", "shift_video", "shift_audio", "source",
    "regen", "sampler", "schedule", "ref_videos", "ref_video_sound", "te_pictures",
];

pub const CANVASES: [(i64, i64, &str, &str); 10] = [
    (640, 480, "4:3", ""),
    (768, 576, "4:3", ""),
    (896, 672, "4:3", ""),
    (1024, 768, "4:3", "native"),
    (1024, 576, "16:9", ""),
    (1344, 768, "16:9", "native, area cap"),
    (768, 768, "1:1", "native"),
    (576, 1024, "9:16", ""),
    (768, 1024, "3:4", "native"),
    (768, 1344, "9:16", "native, area cap"),
];

fn mtime(p: &Path) -> f64 {
    std::fs::metadata(p).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0.0, |d| d.as_secs_f64())
}

pub fn is_clip_name(n: &str) -> bool {
    let b = n.as_bytes();
    n.len() == 18 && n.starts_with("h3_") && b[11] == b'_' && n[3..11].bytes().all(|c| c.is_ascii_digit()) && n[12..].bytes().all(|c| c.is_ascii_digit())
}

impl Studio {
    /// Every finished clip's record: (file stem, record), oldest first.
    pub fn sidecars(&self) -> Vec<(String, Value)> {
        let mut c = self.cache.lock().unwrap();
        if let Some((t, v)) = &c.sidecars {
            if now() - t < 10.0 {
                return v.clone();
            }
        }
        let mut v: Vec<(String, f64, Value)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.out) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if let Some(stem) = n.strip_suffix(".json") {
                    if stem.starts_with("h3_") {
                        if let Some(j) = std::fs::read(e.path()).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
                            v.push((stem.to_string(), mtime(&e.path()), j));
                        }
                    }
                }
            }
        }
        v.sort_by(|a, b| a.0.cmp(&b.0));
        let out: Vec<(String, Value)> = v.into_iter().map(|(a, _, c)| (a, c)).collect();
        c.sidecars = Some((now(), out.clone()));
        out
    }

    pub fn invalidate(&self) {
        self.cache.lock().unwrap().sidecars = None;
    }

    /// Every mp4 in out/, newest first, with its record's summary when it has one.
    pub fn listing(&self) -> Vec<Value> {
        let mut rows: Vec<(f64, Value)> = Vec::new();
        let sides: std::collections::HashMap<String, Value> = self.sidecars().into_iter().collect();
        if let Ok(rd) = std::fs::read_dir(&self.out) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with('.') || !n.ends_with(".mp4") {
                    continue;
                }
                let size = e.metadata().map_or(0, |m| m.len());
                let mt = mtime(&e.path());
                let mut row = json!({"file": n, "size": size, "mtime": mt as i64});
                if let Some(side) = sides.get(n.trim_end_matches(".mp4")) {
                    let j = &side["job"];
                    for (k, v) in [("total", side["times"]["TOTAL"].clone()), ("steps", j["steps"].clone()), ("te", j["te"].clone()), ("label", j["label"].clone()),
                                   ("seconds", j["seconds"].clone()), ("energy_wh", side["energy_wh"].clone()), ("engine", j["engine"].clone()),
                                   ("speech_pct", j["speech_pct"].clone()), ("camera", j["camera"].clone())] {
                        row[k] = v;
                    }
                    row["size2"] = json!(size);
                    row["size"] = json!(format!("{}x{}", j["width"].as_i64().unwrap_or(640), j["height"].as_i64().unwrap_or(480)));
                    row["project"] = json!(project_of(j));
                }
                rows.push((mt, row));
            }
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        rows.into_iter().map(|r| r.1).collect()
    }

    /// Joined films: the mp4s that are not single clips.
    pub fn films(&self) -> Vec<Value> {
        let mut v: Vec<(f64, Value)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.out) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if !n.ends_with(".mp4") || n.starts_with("h3_") || n.ends_with(".part.mp4") || n.starts_with('.') {
                    continue;
                }
                let size = e.metadata().map_or(0, |m| m.len()) as f64;
                let secs = Command::new("ffprobe")
                    .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
                    .arg(e.path())
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok());
                let mt = mtime(&e.path());
                v.push((mt, json!({"file": n, "mb": (size / 1048576.0 * 10.0).round() / 10.0, "seconds": secs, "mtime": mt})));
            }
        }
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v.into_iter().map(|r| r.1).collect()
    }

    /// A 240-pixel still of a clip, 5 s in (or its first frame), cached.
    pub fn thumb(&self, name: &str) -> Option<Vec<u8>> {
        let dir = self.out.join(".thumbs");
        let dst = dir.join(format!("{name}.webp"));
        if let Ok(b) = std::fs::read(&dst) {
            return Some(b);
        }
        let src = self.out.join(format!("{name}.mp4"));
        if !src.exists() {
            return None;
        }
        let _ = std::fs::create_dir_all(&dir);
        for seek in [true, false] {
            let mut c = Command::new("ffmpeg");
            c.args(["-v", "error", "-y"]);
            if seek {
                c.args(["-ss", "5"]);
            }
            let _ = c.arg("-i").arg(&src).args(["-frames:v", "1", "-vf", "scale=240:-2", "-qscale:v", "60"]).arg(&dst).status();
            if let Ok(b) = std::fs::read(&dst) {
                if !b.is_empty() {
                    return Some(b);
                }
            }
        }
        None
    }

    /// Projects of the queue, then the finished clips' projects, newest first.
    pub fn all_projects(&self, q: &Queue) -> Vec<String> {
        let mut v: Vec<String> = q.projects().iter().map(|p| s(p, "project")).collect();
        for (_, side) in self.sidecars().iter().rev() {
            let p = project_of(&side["job"]);
            if !p.is_empty() && !v.contains(&p) {
                v.push(p);
            }
        }
        v
    }

    /// A project's clips in edit order: finished (the newest retry of each number), running, held, queued.
    pub fn timeline(&self, st: &super::State, project: &str) -> Vec<Value> {
        let prefixes: std::collections::BTreeSet<String> = st.q.items.iter().filter(|q| project_of(q) == project).map(|q| label_prefix(&s(q, "label"))).filter(|p| !p.is_empty()).collect();
        let belongs = |j: &Value| project_of(j) == project || prefixes.contains(&label_prefix(&s(j, "label")));
        let mut seen = std::collections::BTreeSet::new();
        let mut out: Vec<Value> = Vec::new();
        for (stem, side) in self.sidecars().iter().rev() {
            let j = &side["job"];
            if !belongs(j) || !self.out.join(format!("{stem}.mp4")).exists() {
                continue;
            }
            let n = label_number(&s(j, "label")).unwrap_or(0);
            if !seen.insert(n) {
                continue;
            }
            out.push(json!({"n": n, "name": stem, "state": "done", "seconds": j["seconds"], "camera": j["camera"], "label": j["label"],
                            "dialogue": dialogue_of(&s(j, "prompt")), "thumb": format!("/thumb/{stem}.webp")}));
        }
        let running = st.job.as_ref().filter(|j| !j.finished).map(|j| j.rec.clone());
        for (it, state) in running.iter().map(|r| (r, "running")).chain(st.q.items.iter().map(|i| (i, if st.q.held(i) { "held" } else { "queued" }))) {
            if project_of(it) != project {
                continue;
            }
            let n = label_number(&s(it, "label")).unwrap_or(0);
            if !seen.insert(n) {
                continue;
            }
            out.push(json!({"n": n, "name": it.get("name").cloned().unwrap_or(Value::Null), "state": state, "seconds": it["seconds"], "camera": it["camera"],
                            "label": it["label"], "dialogue": dialogue_of(&s(it, "prompt")), "thumb": Value::Null}));
        }
        out.sort_by_key(|c| c["n"].as_u64().unwrap_or(0));
        out
    }

    /// A portable scene file of a project: the clips' prompts and settings, the settings all clips share lifted into
    /// the defaults.
    pub fn scene_export(&self, st: &super::State, project: &str) -> Value {
        let sides: std::collections::HashMap<String, Value> = self.sidecars().into_iter().collect();
        let mut clips: Vec<Map<String, Value>> = Vec::new();
        for c in self.timeline(st, project) {
            let src = match c["name"].as_str().and_then(|n| sides.get(n)) {
                Some(side) => side["job"].clone(),
                None => st.job.iter().map(|j| &j.rec).chain(st.q.items.iter()).find(|i| i["label"] == c["label"]).cloned().unwrap_or(Value::Null),
            };
            if s(&src, "prompt").is_empty() {
                continue;
            }
            let mut m = Map::new();
            for (k, v) in [("n", c["n"].clone()), ("camera", src["camera"].clone()), ("seconds", src["seconds"].clone()), ("label", src["label"].clone()), ("prompt", src["prompt"].clone())] {
                m.insert(k.into(), v);
            }
            for k in SCENE_DEFAULTABLE {
                if !src[k].is_null() {
                    m.insert(k.into(), src[k].clone());
                }
            }
            clips.push(m);
        }
        let mut defaults = Map::new();
        if let Some(first) = clips.first().cloned() {
            for k in SCENE_DEFAULTABLE {
                if let Some(v) = first.get(k) {
                    if clips.iter().all(|c| c.get(k) == Some(v)) {
                        defaults.insert(k.into(), v.clone());
                    }
                }
            }
        }
        for c in clips.iter_mut() {
            for k in defaults.keys() {
                c.remove(k);
            }
        }
        json!({"version": 1, "title": project, "project": project, "batch": format!("{project}-import"), "exported": now().round() as i64,
               "notes": "Prompts are final: a runner only needs to merge defaults and POST each clip to /rpc/generate. Clip numbers are the edit order.",
               "defaults": defaults, "clips": clips})
    }

    /// A scene file -> queue items (all or nothing).
    pub fn scene_import(&self, st: &mut super::State, doc: &Value, project: Option<&str>, paused: bool) -> Result<Value, RpcError> {
        if doc["version"].as_i64() != Some(1) {
            return Err(RpcError::param(&format!("unsupported scene version {}", doc["version"])));
        }
        let name: String = project.map(str::to_string).or_else(|| doc["project"].as_str().map(str::to_string)).unwrap_or_else(|| "Imported".into()).trim().chars().take(60).collect();
        let mut clips: Vec<Value> = doc["clips"].as_array().cloned().unwrap_or_default();
        clips.sort_by_key(|c| c["n"].as_i64().unwrap_or(0));
        let mut items = Vec::new();
        for c in &clips {
            let mut p = doc["defaults"].as_object().cloned().unwrap_or_default();
            for (k, v) in c.as_object().into_iter().flatten() {
                if k != "n" {
                    p.insert(k.clone(), v.clone());
                }
            }
            p.insert("project".into(), json!(name));
            p.entry("batch").or_insert_with(|| json!(doc["batch"].as_str().unwrap_or(&name)));
            items.push(self.build_item(&Value::Object(p))?);
        }
        if items.is_empty() {
            return Err(RpcError::param("scene has no clips"));
        }
        let n = items.len();
        st.q.items.extend(items);
        st.q.save(&self.dir);
        if paused {
            st.q.paused_projects.insert(name.clone());
        } else {
            st.q.paused = false;
        }
        st.q.save_paused(&self.dir);
        let held = st.q.items.iter().filter(|i| st.q.held(i)).count();
        eprintln!("scene -> {n} clips into {name}");
        Ok(json!({"imported": n, "project": name, "queued": st.q.items.len(), "held": held}))
    }

    pub fn gpu(&self) -> Value {
        let mut c = self.cache.lock().unwrap();
        if let Some((t, v)) = &c.gpu {
            if now() - t < 2.0 {
                return v.clone();
            }
        }
        let v = match std::fs::read(&self.gpustat).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
            Some(mut v) => {
                if let Some(ts) = v["ts"].as_f64() {
                    v["age"] = json!(((now() - ts) * 10.0).round() / 10.0);
                }
                v
            }
            None => json!({"error": format!("no telemetry at {}; is a gpustat service running?", self.gpustat.display())}),
        };
        c.gpu = Some((now(), v.clone()));
        v
    }

    pub fn canvases(&self) -> Vec<Value> {
        let mut v: Vec<Value> = CANVASES
            .iter()
            .map(|(w, h, a, n)| {
                let r = (*w * *h) as f64 / (640.0 * 480.0);
                json!({"width": w, "height": h, "aspect": a, "note": n, "mpx": ((w * h) as f64 / 1e6 * 100.0).round() / 100.0,
                       "cost": ((0.44 * r * r + 0.56 * r) * 100.0).round() / 100.0, "default": (*w, *h) == (640, 480)})
            })
            .collect();
        v.sort_by_key(|c| (c["width"].as_i64().unwrap() * c["height"].as_i64().unwrap(), c["width"].as_i64().unwrap()));
        v
    }

    /// The denoisers the engine daemon has (its `--model`, INT8, and the GGUF forms of its `--engine`s), in the legacy
    /// shape, by the front end's names; INT8 alone when the daemon does not answer.
    pub fn engines(&self) -> Vec<Value> {
        let raw: Vec<Value> = super::http::call(&super::http::Target::Unix(self.socket.clone()), "GET", "/engine/status", None)
            .ok()
            .and_then(|v| v["engines"].as_array().cloned())
            .unwrap_or_default();
        super::queue::learn_engines(&raw.iter().filter_map(|e| e["name"].as_str().map(str::to_string)).collect::<Vec<_>>());
        let listed: Vec<Value> = raw
            .iter()
            .filter_map(|e| {
                let path = e["path"].as_str()?;
                Some(json!({"quant": super::queue::engine_quant(e["name"].as_str().unwrap_or("")), "file": path.rsplit('/').next().unwrap_or(path), "path": path,
                            "gib": (e["gib"].as_f64().unwrap_or(0.0) * 100.0).round() / 100.0, "ready": true, "default": e["default"]}))
            })
            .collect();
        if listed.is_empty() {
            return vec![json!({"quant": "INT8", "file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "path": "", "gib": 19.53, "ready": true, "default": true})];
        }
        let mut v = listed;
        v.sort_by(|a, b| a["gib"].as_f64().unwrap_or(0.0).total_cmp(&b["gib"].as_f64().unwrap_or(0.0)));
        v
    }

    /// The canvas table's model. `gpus` is what `nextsycl video plan measure` found on each of this box's GPUs (plan.json in
    /// the studio's directory, read on every call); without it, the fitted curve from before (one GPU, older build).
    pub fn plan(&self) -> Value {
        let gpus: Vec<Value> = std::fs::read(self.dir.join("plan.json")).ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v["gpus"].as_array().cloned())
            .unwrap_or_default();
        // the single curve older front ends read: the fastest GPU's, with the default denoiser
        let fastest = gpus.iter().filter(|g| g["engine"].as_str().unwrap_or("INT8") == "INT8").min_by(|a, b| a["step_a"].as_f64().unwrap_or(1.0).total_cmp(&b["step_a"].as_f64().unwrap_or(1.0)));
        let (a, b) = fastest.map(|g| (g["step_a"].as_f64().unwrap_or(1.829e-8), g["step_b"].as_f64().unwrap_or(3.4634e-4))).unwrap_or((1.829e-8, 3.4634e-4));
        json!({"engines": self.engines(), "canvases": self.canvases(),
               "defaults": {"seconds": 10, "steps": 10, "width": 768, "height": 576, "engine": "INT8"},
               "cap_gib": 30.3, "gib_per_token": 1.64e-4, "margin_gib": 0.8, "step_a": a, "step_b": b, "gpus": gpus,
               "chain_modes": super::queue::CHAIN_MODES.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<Map<String, Value>>(),
               "measured": {},
               "samplers": nextsycl_diffusion::samplers::NAMES,
               "schedules": nextsycl_diffusion::Schedule::ALL.iter().map(|s| s.name()).collect::<Vec<_>>()})
    }

    pub fn templates(&self) -> Vec<Value> {
        let Some(dir) = &self.templates else { return Vec::new() };
        let mut v: Vec<(String, Value)> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                let stem = n.strip_suffix(".txt")?.to_string();
                let text = std::fs::read_to_string(e.path()).ok()?;
                let name = stem.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start_matches('-').replace('-', " ");
                Some((n, json!({"name": name, "text": text.trim()})))
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v.into_iter().map(|x| x.1).collect()
    }

    fn disk(&self) -> Value {
        // `df` of the clips directory (statvfs without a libc dependency)
        let o = Command::new("df").args(["-B1", "--output=size,used,avail"]).arg(&self.out).output().ok();
        let nums: Vec<f64> = o.map(|o| String::from_utf8_lossy(&o.stdout).lines().nth(1).unwrap_or("").split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        if nums.len() < 3 {
            return Value::Null;
        }
        let g = |b: f64| (b / 1073741824.0 * 10.0).round() / 10.0;
        json!({"path": self.out, "total_gb": g(nums[0]), "used_gb": g(nums[1]), "free_gb": g(nums[2]), "pct": pyround(nums[1] / nums[0] * 100.0)})
    }

    pub fn summary(&self, st: &mut super::State) -> Value {
        let mut done: Vec<(String, f64, f64, Option<f64>)> = Vec::new();
        for r in self.listing() {
            if let (Some(t), Some(sec)) = (r["total"].as_f64(), r["seconds"].as_f64()) {
                done.push((s(&r, "label"), snap_seconds(sec), t, r["energy_wh"].as_f64()));
            }
        }
        let recent: Vec<_> = done.iter().take(10).collect();
        let vid: f64 = recent.iter().map(|d| d.1).sum::<f64>().max(1.0);
        let wall_per_vs = recent.iter().map(|d| d.2).sum::<f64>() / vid;
        let with_wh: Vec<_> = recent.iter().filter(|d| d.3.is_some()).collect();
        let wh_per_vs = if with_wh.is_empty() { 0.0 } else { with_wh.iter().map(|d| d.3.unwrap()).sum::<f64>() / with_wh.iter().map(|d| d.1).sum::<f64>().max(1e-9) };
        let stv = self.status(st);
        let running = stv["idle"] != true;
        let cur_left = if running { stv["eta"].as_f64().unwrap_or(0.0).max(0.0) } else { 0.0 };
        let cur_vid = if running { snap_seconds(stv["job"]["seconds"].as_f64().unwrap_or(0.0)) } else { 0.0 };
        let q_vid: f64 = st.q.items.iter().map(|i| snap_seconds(i["seconds"].as_f64().unwrap_or(0.0))).sum();
        let remaining = cur_left + q_vid * wall_per_vs;
        let project = stv["job"].get("project").and_then(|p| p.as_str()).unwrap_or("").to_string();
        json!({"rate_wall_per_video_s": (wall_per_vs * 10.0).round() / 10.0, "rate_wh_per_video_s": (wh_per_vs * 100.0).round() / 100.0,
               "sample_clips": recent.len(), "running": running, "current_eta_s": cur_left, "current_video_s": cur_vid,
               "queued": st.q.items.len(), "queued_video_s": q_vid, "remaining_wall_s": remaining,
               "eta_ts": if running || !st.q.items.is_empty() { json!(now() + remaining) } else { Value::Null },
               "remaining_kwh": (((cur_left / wall_per_vs.max(1e-9)) + q_vid) * wh_per_vs / 1000.0 * 100.0).round() / 100.0,
               "disk": self.disk(), "project_stats": self.project_stats(&project),
               "done_total": done.len(), "done_video_s": done.iter().map(|d| d.1).sum::<f64>(), "done_wall_s": done.iter().map(|d| d.2).sum::<f64>(),
               "done_wh": done.iter().filter_map(|d| d.3).sum::<f64>()})
    }

    fn project_stats(&self, name: &str) -> Value {
        if name.is_empty() {
            return Value::Null;
        }
        let (mut clips, mut vid, mut gpu, mut wh) = (0, 0.0, 0.0, 0.0);
        for (_, side) in self.sidecars() {
            let j = &side["job"];
            if project_of(j) != name && label_prefix(&s(j, "label")) != name {
                continue;
            }
            if let (Some(t), Some(sec)) = (side["times"]["TOTAL"].as_f64(), j["seconds"].as_f64()) {
                clips += 1;
                vid += sec;
                gpu += t;
                wh += side["energy_wh"].as_f64().unwrap_or(0.0);
            }
        }
        json!({"project": name, "clips": clips, "video_s": (vid * 10.0).round() / 10.0, "gpu_s": gpu as i64,
               "ratio": if vid > 0.0 { (gpu / vid * 10.0).round() / 10.0 } else { 0.0 }, "wh": wh as i64})
    }
}
