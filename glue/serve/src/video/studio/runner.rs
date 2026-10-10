//! Turning queue items into engine jobs: anchors resolved against the series' finished clips, the `generate` job
//! the daemon runs, its progress and result read back into the status the front end shows, the per-clip record
//! (sidecar) written when it is done, and the scheduler that starts the next item.
//!
//! Where the legacy server had bugs this does what it meant (docs/LEGACY-API.md): an audio anchor uses the clip's
//! `.lastaud.safetensors`, an end keyframe is always a picture, a finished clip is never queued again because its
//! container went away, and no command line is built from request strings.

use std::path::Path;
use std::time::Duration;

use super::http::Target;
use serde_json::{json, Value};

use super::queue::{label_number, label_prefix, project_of, s};
use super::{now, write_atomic, Job, State, Studio};

/// A finished clip of the series: (number, file stem, camera).
type SeriesClip = (u64, String, String);

impl Studio {
    fn engine(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
        super::http::call(&Target::Unix(self.socket.clone()), method, path, body).map_err(|e| e.0)
    }

    /// The series' finished clips, in edit order (the label's "NN/").
    fn series(&self, label: &str) -> Vec<SeriesClip> {
        let pre = label_prefix(label);
        if pre.is_empty() && label_number(label).is_none() {
            return Vec::new();
        }
        let mut v: Vec<SeriesClip> = Vec::new();
        for (stem, side) in self.sidecars() {
            let jl = s(&side["job"], "label");
            if label_prefix(&jl) == pre {
                if let Some(n) = label_number(&jl).or_else(|| jl[pre.len()..].trim_start().split('/').next().and_then(|d| d.parse().ok())) {
                    v.push((n, stem, s(&side["job"], "camera")));
                }
            }
        }
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        v
    }

    /// The newest finished clip anywhere.
    fn last_clip(&self, st: &State) -> Option<String> {
        st.last_file.clone().or_else(|| self.listing().first().map(|r| s(r, "file").trim_end_matches(".mp4").to_string()))
    }

    /// An anchor word (or a file) -> a clip stem / file name in out/.
    fn resolve(&self, st: &State, item: &Value, which: &str, v: &str) -> Option<String> {
        let series = self.series(&s(item, "label"));
        let cam = s(item, "camera");
        let same_cam = |first: bool| -> Option<String> {
            let mut it = series.iter().filter(|c| !cam.is_empty() && c.2 == cam);
            if first { it.next() } else { it.next_back() }.map(|c| c.1.clone())
        };
        match v {
            "prev" if which == "last_frame" => self.last_clip(st),
            "prev" => series.last().map(|c| c.1.clone()).or_else(|| self.last_clip(st)),
            "prev_cam" => same_cam(false).or_else(|| series.last().map(|c| c.1.clone())),
            "first" => series.first().map(|c| c.1.clone()),
            "first_cam" => same_cam(true),
            "" => None,
            f => Some(f.to_string()),
        }
    }

    /// The engine path of a clip's file, preferring the given extensions in order; else the name as given.
    fn clip_file(&self, stem_or_file: &str, prefer: &[&str]) -> String {
        let base = stem_or_file.strip_suffix(".mp4").unwrap_or(stem_or_file);
        for ext in prefer {
            if self.out.join(format!("{base}{ext}")).exists() {
                return format!("{}/{base}{ext}", self.out_in);
            }
        }
        format!("{}/{}", self.out_in, super::queue::basename(stem_or_file))
    }

    /// An item -> the engine's `generate` job (and the resolved anchors, recorded on the job).
    fn spec(&self, st: &State, item: &Value, name: &str) -> (Value, Value) {
        let mut spec = json!({
            "kind": "generate", "prompt": item["prompt"], "width": item["width"], "height": item["height"], "seconds": item["seconds"],
            "steps": item["steps"], "seed": item["seed"], "out": format!("{}/{name}.mp4", self.out_in),
            "engine": super::queue::engine_id(&super::queue::engine_name(item["engine"].as_str())),
        });
        let mut resolved = json!({});
        let mode = s(item, "chain_mode");
        let image = |f: &str| [".png", ".jpg", ".jpeg", ".webp"].iter().any(|e| f.to_lowercase().ends_with(e));
        if mode != "none" {
            if let Some(src) = self.resolve(st, item, "first_frame", &s(item, "first_frame")) {
                resolved["first_frame"] = json!(if image(&src) { src.clone() } else { format!("{}.mp4", src.trim_end_matches(".mp4")) });
                if image(&src) {
                    spec["first_frame"] = json!(format!("{}/{src}", self.out_in));
                } else if mode == "latent" && self.out.join(format!("{}.lastlat.safetensors", src.trim_end_matches(".mp4"))).exists() {
                    spec["first_latent"] = json!(self.clip_file(&src, &[".lastlat.safetensors"]));
                } else if mode == "png" || mode == "latent" {
                    spec["first_frame"] = json!(self.clip_file(&src, &[".last.png", ".mp4"]));
                } else {
                    spec["first_frame"] = json!(self.clip_file(&src, &[".mp4"]));
                }
            }
            // an end keyframe is a picture in every mode (the legacy server passed a latent here in latent mode)
            if let Some(src) = self.resolve(st, item, "last_frame", &s(item, "last_frame")) {
                resolved["last_frame"] = json!(src);
                spec["last_frame"] = json!(if image(&src) { format!("{}/{src}", self.out_in) } else { self.clip_file(&src, &[".last.png", ".mp4"]) });
            }
        }
        if let Some(src) = self.resolve(st, item, "first_audio", &s(item, "first_audio")) {
            // the clip's own sound tail at the model's level first (the legacy server always used the mp4)
            resolved["first_audio"] = json!(src);
            spec["first_audio"] = json!(self.clip_file(&src, &[".lastaud.safetensors", ""]));
            if let Some(sec) = item["first_audio_s"].as_f64() {
                spec["first_audio_s"] = json!(sec);
            }
        }
        if let Some(src) = self.resolve(st, item, "exposure_ref", &s(item, "exposure_ref")) {
            spec["first_frame_ref"] = json!(if image(&src) { format!("{}/{src}", self.out_in) } else { self.clip_file(&src, &[".last.png", ".mp4"]) });
        }
        // a motion guide: "<anchor or file>[:frames[:at]]"
        let g = s(item, "guide_clip");
        if !g.is_empty() {
            let mut parts = g.splitn(3, ':');
            let src = parts.next().unwrap_or("");
            if let Some(src) = self.resolve(st, item, "guide_clip", src) {
                let rest: Vec<&str> = parts.collect();
                resolved["guide_clip"] = json!(src);
                spec["guide_clip"] = json!(format!("{}{}{}", self.clip_file(&src, &[".mp4"]), if rest.is_empty() { "" } else { ":" }, rest.join(":")));
            }
        }
        let src = s(item, "source");
        if !src.is_empty() {
            if let Some(src) = self.resolve(st, item, "source", &src) {
                resolved["source"] = json!(src);
                spec["source"] = json!(self.clip_file(&src, &[".latents.safetensors"]));
            }
        }
        for k in ["cond_noise_aug", "shift_video", "shift_audio"] {
            if let Some(v) = item[k].as_f64() {
                spec[k] = json!(v);
            }
        }
        for k in ["regen", "regen_box", "sampler", "schedule"] {
            if item[k].is_string() {
                spec[k] = item[k].clone();
            }
        }
        if let Some(u) = item["upscale"].as_f64().or_else(|| item["upscale"].as_str().and_then(|v| v.parse().ok())).filter(|u| *u > 1.0) {
            spec["upscale"] = json!(u);
            // "upscaler": an ESRGAN-type network on the decoded frames - general-x4v3 by default (the user's pick,
            // 2026-10-05: sharp and natural on detailed shots, and about half the decode stage's time) - or "latent",
            // the latent upscaler. h3-core esrgan.rs; the weights from reference/esrgan_to_safetensors.py in
            // models/esrgan (a box without them falls back to the latent upscaler, see jobs.rs).
            let pixel = match item["upscaler"].as_str().unwrap_or("esrgan-general") {
                "esrgan-anime" => Some("/models/esrgan/realesr-animevideov3.safetensors"),
                "esrgan-general" => Some("/models/esrgan/realesr-general-x4v3.safetensors"),
                _ => None,
            };
            if let Some(p) = pixel {
                spec["pixel_upscaler"] = json!(p);
            }
        }
        if let Some(ls) = item["loras"].as_array().filter(|a| !a.is_empty()) {
            spec["lora"] = json!(ls.iter().filter_map(|l| l.as_str()).collect::<Vec<_>>().join(","));
        }
        for (k, key) in [("ref_audios", "ref_audio"), ("ref_images", "ref_images"), ("ref_videos", "ref_videos")] {
            if let Some(r) = item[k].as_array() {
                spec[key] = json!(r.iter().filter_map(|x| x.as_str()).map(|f| format!("{}/{f}", self.out_in)).collect::<Vec<_>>().join(","));
            }
        }
        for k in ["ref_image_size", "ref_video_sound", "te_pictures"] {
            if !item[k].is_null() {
                spec[k] = item[k].clone();
            }
        }
        (spec, resolved)
    }

    /// Start an item on the engine.
    pub(super) fn launch(&self, st: &mut State, item: Value) -> Result<(), String> {
        let name = super::stamp();
        let (spec, resolved) = self.spec(st, &item, &name);
        let r = self.engine("POST", "/engine/jobs", Some(&spec))?;
        let id = r["id"].as_u64().ok_or("the engine gave no job id")?;
        let mut rec = item.clone();
        for (k, v) in resolved.as_object().into_iter().flatten() {
            rec[k] = v.clone();
        }
        rec["id"] = json!(id.to_string());
        rec["name"] = json!(name);
        rec["started"] = json!(now());
        rec["project"] = json!(project_of(&item));
        let energy0 = self.gpu()["energy_j"].as_f64();
        rec["energy0"] = json!(energy0);
        let _ = write_atomic(&self.dir.join("job.json"), &serde_json::to_vec(&rec).unwrap_or_default());
        eprintln!("queue -> {name} (engine job {id}): {}", s(&item, "label"));
        st.current_item = Some(item);
        st.job = Some(Job { rec, engine_id: id, energy0, saved: false, stall: (0, now()), finished: false, last: Value::Null });
        st.torn = false;
        Ok(())
    }

    /// The current job as the front end sees it (the legacy `status()`): progress, stage, times, error; writes the
    /// clip's record once it is done.
    pub(super) fn status(&self, st: &mut State) -> Value {
        let mut out = serde_json::Map::new();
        let Some(job) = st.job.as_mut() else {
            out.insert("idle".into(), json!(true));
            out.extend(st.q.payload());
            out.insert("all_projects".into(), json!(self.all_projects(&st.q)));
            out.insert("llm".into(), self.llm.status());
            return Value::Object(out);
        };
        if !job.finished {
            match self.engine("GET", &format!("/engine/jobs/{}", job.engine_id), None) {
                Ok(j) => job.last = j,
                Err(e) => {
                    if job.last.is_null() || !matches!(job.last["state"].as_str(), Some("done" | "failed" | "cancelled")) {
                        job.last = json!({"state": "failed", "error": format!("the engine does not answer: {e}"), "log": job.last.get("log").cloned().unwrap_or(json!([]))});
                    }
                }
            }
        }
        let j = &job.last;
        let state = j["state"].as_str().unwrap_or("queued");
        let log: Vec<String> = j["log"].as_array().map(|a| a.iter().filter_map(|l| l.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let has = |p: &str| log.iter().any(|l| l.starts_with(p));
        let secs_after = |p: &str| -> Option<f64> {
            log.iter().rev().filter(|l| l.starts_with(p)).find(|l| l.contains(" in ")).and_then(|l| {
                let i = l.rfind(" in ")?;
                l[i + 4..].split_whitespace().next()?.parse().ok()
            })
        };
        let steps_total = job.rec["steps"].as_u64().unwrap_or(8);
        let (mut pct, mut stage) = (2.0, "starting".to_string());
        if state == "queued" {
            stage = "waiting for the engine (loading or another job)".into();
        }
        if has("prompt :") || has("te     :") {
            pct = 6.0;
            stage = "text encoder".into();
        }
        if has("encoded:") {
            pct = 10.0;
            stage = "conditioning done".into();
        }
        let mut steps_done = 0;
        if has("tokens :") {
            pct = 12.0;
            stage = "sampling (step 1 running)".into();
            if let Some((d, t)) = j["progress"]["done"].as_u64().zip(j["progress"]["total"].as_u64()).filter(|(_, t)| *t > 0) {
                let per = (t / steps_total.max(1)).max(1);
                steps_done = (d / per).min(steps_total);
                pct = 12.0 + (68 * d / t) as f64;
                stage = format!("sampling {}/{steps_total}", steps_done + 1);
            }
        }
        if has("sampled:") {
            pct = 80.0;
            steps_done = steps_total;
            stage = "sampling done, decoding".into();
        }
        if has("decoded:") {
            pct = 90.0;
            stage = "video decoded".into();
        }
        if has("audio  :") {
            pct = 95.0;
            stage = "audio decoded, muxing".into();
        }
        let done = state == "done";
        let failed = matches!(state, "failed" | "cancelled");
        let name = s(&job.rec, "name");
        if done {
            pct = 100.0;
            stage = "done".into();
            st.last_file = Some(name.clone());
        }
        // times, in the legacy record's labels
        let mut times = serde_json::Map::new();
        if let Some(t) = secs_after("encoded:") {
            times.insert("teacher TE (streamed)".into(), json!(t));
        }
        if let Some(t) = secs_after("text   :").or_else(|| log.iter().find(|l| l.starts_with("text   :")).and_then(|l| l.split(" in ").nth(1)?.split_whitespace().next()?.parse().ok())) {
            times.insert("text refiner".into(), json!(t));
        }
        if let Some(l) = log.iter().find(|l| l.starts_with("sampled:")) {
            if let Some(t) = l.split(" in ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse::<f64>().ok()) {
                times.insert(format!("sampled {steps_total} steps"), json!(t));
            }
            if let (Some(a), Some(b)) = (l.find('('), l.rfind(')')) {
                let mut cum = 0.0;
                for (i, v) in l[a + 1..b].split_whitespace().filter_map(|v| v.parse::<f64>().ok()).enumerate() {
                    cum += v;
                    times.insert(format!("step {}/{steps_total}", i + 1), json!((cum * 10.0f64).round() / 10.0));
                }
            }
        }
        for (p, k) in [("upscale:", "latent upscaled"), ("decoded:", "video decoded"), ("audio  :", "AUDIO decoded")] {
            if let Some(t) = secs_after(p) {
                times.insert(k.into(), json!(t));
            }
        }
        if let Some(t) = secs_after("clip   :") {
            times.insert("TOTAL".into(), json!(t));
        }
        // energy
        let mut energy = Value::Null;
        if let (Some(e0), Some(e1)) = (job.energy0, self.gpu()["energy_j"].as_f64()) {
            if e1 >= e0 {
                energy = json!(((e1 - e0) / 3600.0 * 10.0).round() / 10.0);
                if !job.finished {
                    job.rec["energy_wh"] = energy.clone();
                }
            }
        }
        let error = if failed {
            Some(format!("{}{}\n{}", if state == "cancelled" { "cancelled" } else { "failed" }, j["error"].as_str().map(|e| format!(": {e}")).unwrap_or_default(),
                         log.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n")))
        } else {
            None
        };
        if failed {
            stage = "failed".into();
        }
        let elapsed = now() - job.rec["started"].as_f64().unwrap_or(now());
        let frames = frames_of(job.rec["seconds"].as_f64().unwrap_or(5.0));
        let eta = if done || failed {
            0.0
        } else if steps_done > 0 && steps_done < steps_total {
            let per = (elapsed - 30.0).max(1.0) / steps_done as f64;
            per * (steps_total - steps_done) as f64 + 0.15 * frames as f64
        } else if has("sampled:") {
            0.15 * frames as f64
        } else {
            (40.0 - elapsed).max(5.0) + steps_total as f64 * 0.075 * frames as f64 + 0.15 * frames as f64
        };
        // the clip's record, once
        if done && !job.saved {
            job.finished = true;
            let mut rec = job.rec.clone();
            rec["energy_wh"] = energy.clone();
            rec["energy_wh_final"] = energy.clone();
            if let Some(t) = times.get(&format!("sampled {steps_total} steps")) {
                rec["sampled_s"] = t.clone();
            }
            if let Some(sp) = super::speech::analyse(&self.out.join(format!("{name}.mp4"))) {
                rec["speech_pct"] = json!(sp.speech_pct);
                rec["speech_s"] = json!(sp.speech_s);
                let words = super::speech::spoken_words(&s(&rec, "prompt"));
                if words > 0 && sp.speech_s > 0.0 {
                    rec["words"] = json!(words);
                    rec["words_per_s"] = json!((words as f64 / sp.speech_s * 100.0).round() / 100.0);
                }
            }
            let side = json!({"job": rec, "times": times, "rms": Value::Null, "frames": frames, "wall": (elapsed * 10.0).round() / 10.0, "energy_wh": energy});
            let _ = write_atomic(&self.out.join(format!("{name}.json")), &serde_json::to_vec_pretty(&side).unwrap_or_default());
            job.saved = true;
            self.invalidate();
        }
        if failed {
            job.finished = true;
        }
        let rec = job.rec.clone();
        let idle = done || failed;
        out.insert("idle".into(), json!(idle));
        out.insert("job".into(), rec);
        out.insert("pct".into(), json!(pct.round() as i64));
        out.insert("stage".into(), json!(stage));
        out.insert("done".into(), json!(done));
        out.insert("error".into(), json!(error));
        out.insert("exited".into(), json!(idle));
        out.insert("rc".into(), if idle { json!(if done { 0 } else { 1 }) } else { Value::Null });
        out.insert("elapsed".into(), json!(elapsed as i64));
        out.insert("eta".into(), json!(eta as i64));
        out.insert("rms".into(), Value::Null);
        out.insert("times".into(), Value::Object(times));
        out.insert("frames".into(), json!(frames));
        out.insert("steps_done".into(), json!(steps_done));
        out.insert("steps_total".into(), json!(steps_total));
        out.insert("per_step".into(), Value::Null);
        out.insert("metrics".into(), json!([]));
        out.insert("file".into(), if done { json!(format!("/out/{name}.mp4")) } else { Value::Null });
        out.insert("engine_log".into(), json!(log.iter().rev().take(12).rev().collect::<Vec<_>>()));
        out.extend(st.q.payload());
        out.insert("all_projects".into(), json!(self.all_projects(&st.q)));
        out.insert("llm".into(), self.llm.status());
        Value::Object(out)
    }

    pub(super) fn running(&self, st: &State) -> bool {
        st.job.as_ref().is_some_and(|j| !j.finished)
    }

    /// Stop the running clip (at the engine's next block boundary).
    pub(super) fn cancel(&self, st: &mut State) -> Result<(), String> {
        let id = st.job.as_ref().filter(|j| !j.finished).map(|j| j.engine_id).ok_or("no job is running")?;
        st.current_item = None; // no retry
        self.engine("POST", &format!("/engine/jobs/{id}/cancel"), None).map(|_| ())
    }

    /// One scheduler tick (every 5 s).
    pub(super) fn tick(&self) {
        self.llm.idle_watchdog(self);
        let mut st = self.st.lock().unwrap();
        if self.running(&st) {
            let _ = self.status(&mut st);
            // the watchdog: no new log line for the stall time -> stop it (becomes a failure -> the retry path)
            if let Some(j) = st.job.as_mut() {
                let n = j.last["log"].as_array().map_or(0, |a| a.len());
                if n != j.stall.0 {
                    j.stall = (n, now());
                } else if now() - j.stall.1 > self.stall_s as f64 && j.last["state"] == "running" {
                    eprintln!("watchdog: {} wrote nothing for {} min, stopping it", s(&j.rec, "name"), self.stall_s / 60);
                    let id = j.engine_id;
                    let _ = self.engine("POST", &format!("/engine/jobs/{id}/cancel"), None);
                    j.stall.1 = now();
                }
            }
            if self.running(&st) {
                return;
            }
        }
        if st.q.items.is_empty() || st.q.items.iter().all(|i| st.q.held(i)) {
            return;
        }
        let stv = self.status(&mut st);
        if stv["error"].is_string() && st.current_item.is_some() && st.job.as_ref().is_some_and(|j| !j.saved && j.rec["retried"].is_null()) {
            st.fails += 1;
            if st.fails <= 1 {
                let mut again = st.current_item.take().unwrap();
                again["retried"] = json!(true);
                st.q.items.insert(0, again);
                st.q.save(&self.dir);
            } else {
                st.q.paused = true;
                st.fails = 0;
                st.q.save_paused(&self.dir);
                st.current_item = None;
                eprintln!("queue: two failures in a row, holding the queue");
                return;
            }
        } else if stv["done"] == true {
            st.fails = 0;
        }
        if let Some(j) = st.job.as_mut() {
            j.rec["retried"] = json!(true); // looked at once: never re-queued again
        }
        let Some(idx) = st.q.items.iter().position(|i| !st.q.held(i)) else { return };
        let mut item = st.q.items.remove(idx);
        st.q.save(&self.dir);
        if let Some(o) = item.as_object_mut() {
            o.remove("retried");
        }
        if let Err(e) = self.launch(&mut st, item.clone()) {
            eprintln!("queue: launching failed ({e}); back at the front, queue held");
            st.q.items.insert(0, item);
            st.q.paused = true;
            st.q.save(&self.dir);
            st.q.save_paused(&self.dir);
        }
    }

    pub fn scheduler(self: std::sync::Arc<Self>) {
        loop {
            std::thread::sleep(Duration::from_secs(5));
            self.tick();
        }
    }
}

/// Seconds -> frames on the model's 17k + 5 grid (Python's round, half to even).
pub fn frames_of(seconds: f64) -> u64 {
    let mut n = (super::pyround(seconds * 24.0) as i64).max(5) as u64;
    while n % 17 != 5 {
        n += 1;
    }
    n
}

pub fn snap_seconds(s: f64) -> f64 {
    frames_of(s) as f64 / 24.0
}

/// The record of a job as it was when the studio restarted, if it is still worth following.
pub fn recover(dir: &Path) -> Option<Value> {
    std::fs::read(dir.join("job.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
}
