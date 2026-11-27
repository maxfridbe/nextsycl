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
use super::{library, now, write_atomic, Job, State, Studio, MAX_ENDED};

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

    /// The newest finished clip of the item's project (two chains run at once: not the other's); with none, anywhere
    /// (such a clip starts alone: slots.rs)
    fn last_clip(&self, st: &State, item: &Value) -> Option<String> {
        let p = project_of(item);
        let own = (!p.is_empty()).then(|| self.sidecars().into_iter().filter(|(_, side)| project_of(&side["job"]) == p).map(|(stem, _)| stem).max()).flatten();
        own.or_else(|| st.last_file.clone()).or_else(|| self.listing().first().map(|r| s(r, "file").trim_end_matches(".mp4").to_string()))
    }

    /// Whether `prev` has a clip of the item's own to fall on
    pub(super) fn has_own(&self, item: &Value) -> bool {
        let p = project_of(item);
        !self.series(&s(item, "label")).is_empty() || (!p.is_empty() && self.sidecars().iter().any(|(_, side)| project_of(&side["job"]) == p))
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
            "prev" if which == "last_frame" => self.last_clip(st, item),
            "prev" => series.last().map(|c| c.1.clone()).or_else(|| self.last_clip(st, item)),
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
        // Open-H3-Regenerate-2K-IC: the draft (an anchor or a clip in out/) made again at 2K - the regen2k job, the prompt
        // as its caption
        let draft = s(item, "regen2k");
        if !draft.is_empty() {
            if let Some(src) = self.resolve(st, item, "regen2k", &draft) {
                resolved["regen2k"] = json!(src);
                let mut r = json!({
                    "kind": "regen2k", "draft": self.clip_file(&src, &[".mp4"]), "caption": item["prompt"], "seed": item["seed"],
                    "sampler": item["regen_sampler"].as_str().unwrap_or("up3"), "resolution": item["regen_resolution"].as_str().unwrap_or("2k"),
                    "out": spec["out"], "engine": spec["engine"],
                });
                if let Some(x) = item["sharp"].as_f64() {
                    r["sharp"] = json!(x);
                }
                return (r, resolved);
            }
        }
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
        let track = s(item, "soundtrack");
        if !track.is_empty() {
            spec["soundtrack"] = json!(format!("{}/{track}", self.out_in));
            spec["soundtrack_start"] = json!(item["soundtrack_start"].as_f64().unwrap_or(0.0));
        }
        if let Some(src) = self.resolve(st, item, "exposure_ref", &s(item, "exposure_ref")) {
            spec["first_frame_ref"] = json!(if image(&src) { format!("{}/{src}", self.out_in) } else { self.clip_file(&src, &[".last.png", ".mp4"]) });
        }
        // motion guides: "<anchor or file>[:frames[:at]]", several comma-separated (a picture: a one-frame keyframe)
        let g = s(item, "guide_clip");
        let (mut srcs, mut files) = (Vec::new(), Vec::new());
        for one in g.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let mut parts = one.splitn(3, ':');
            let src = parts.next().unwrap_or("");
            if let Some(src) = self.resolve(st, item, "guide_clip", src) {
                let rest: Vec<&str> = parts.collect();
                let file = if image(&src) { format!("{}/{src}", self.out_in) } else { self.clip_file(&src, &[".mp4"]) };
                files.push(format!("{file}{}{}", if rest.is_empty() { "" } else { ":" }, rest.join(":")));
                srcs.push(src);
            }
        }
        if !files.is_empty() {
            resolved["guide_clip"] = json!(srcs.join(","));
            spec["guide_clip"] = json!(files.join(","));
        }
        if item["exposure_fix"].as_bool() == Some(true) {
            spec["exposure_fix"] = json!(true);
            for k in ["exposure_offset", "exposure_ease", "exposure_target"] {
                if let Some(v) = item[k].as_f64() {
                    spec[k] = json!(v);
                }
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
        for k in ["regen", "regen_box", "sampler", "schedule", "knots"] {
            if item[k].is_string() {
                spec[k] = item[k].clone();
            }
        }
        // a MovieRoll film (video.roll): its events, live, the HLS playlist beside it, the control file video.rollAppend
        // and video.rollEnd write
        if item["events"].is_array() {
            spec["events"] = item["events"].clone();
            spec["live"] = json!(item["live"] == true);
            spec["max_idle_s"] = item["max_idle_s"].clone();
            spec["hls"] = json!(item["hls"] == true);
            spec["control"] = json!(format!("{}/{name}.ctl.json", self.out_in));
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
        for k in ["control_video", "control_mask", "control_source"] {
            if let Some(f) = item[k].as_str() {
                spec[k] = json!(format!("{}/{f}", self.out_in));
            }
        }
        for k in ["ref_image_size", "ref_video_sound", "te_pictures", "control_strength", "control_start", "control_end"] {
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
        // every card's counter: which card this clip gets, the daemon says later
        let tele = self.gpu();
        let energy0 = tele["energy_j"].as_f64();
        rec["energy0"] = json!(energy0);
        rec["energy0_cards"] = card_energies(&tele);
        let _ = write_atomic(&self.dir.join("job.json"), &serde_json::to_vec(&rec).unwrap_or_default());
        eprintln!("queue -> {name} (engine job {id}): {}", s(&item, "label"));
        st.running.push(Job { rec, engine_id: id, energy0, saved: false, stall: (0, now()), finished: false, last: Value::Null, item: Some(item), settled: false, view: Value::Null, retrying: false });
        self.save_running(st);
        Ok(())
    }

    /// The clips being made, for a restart of the studio
    fn save_running(&self, st: &State) {
        *self.running_names.lock().unwrap() = st.running.iter().map(|j| s(&j.rec, "name")).collect();
        let recs: Vec<&Value> = st.running.iter().map(|j| &j.rec).collect();
        let _ = write_atomic(&self.dir.join("jobs.json"), &serde_json::to_vec(&recs).unwrap_or_default());
    }

    /// The daemon's state, once a pass, with the door's word on the cards and the host beside it
    pub(super) fn engine_state(&self) -> Option<Value> {
        let mut e = self.engine("GET", "/engine/status", None).ok()?;
        let word = self.door_word.lock().unwrap().clone();
        e["blocked"] = word["blocked"].clone();
        e["host_free_mb"] = word["host_free_mb"].clone();
        Some(e)
    }

    /// Clips possible now, with the host's memory by the door's figure, else this machine's
    pub(super) fn room(&self, st: &State, es: Option<&Value>) -> super::slots::Room {
        let loading = st.running.iter().filter(|j| !worker_loaded(&j.last)).count();
        let host = es.and_then(|e| e["host_free_mb"].as_u64()).or_else(|| std::fs::read_to_string(&self.meminfo).ok().and_then(|m| super::slots::mem_available_mb(&m)));
        super::slots::room(es, st.running.len(), loading, host)
    }

    /// Every running clip asked about; those that ended go to `ended`. What `status` answers from.
    pub(super) fn refresh(&self, st: &mut State, es: Option<&Value>) {
        let mut running = std::mem::take(&mut st.running);
        for job in &mut running {
            job.view = self.view(job, es, &mut st.last_file);
        }
        let before = running.len();
        for job in running {
            if job.finished {
                st.ended.push(job);
            } else {
                st.running.push(job);
            }
        }
        if st.running.len() != before {
            self.save_running(st);
        }
        let over = st.ended.len().saturating_sub(MAX_ENDED);
        // an unsettled failure is the scheduler's still: settled ones go first
        for _ in 0..over {
            let at = st.ended.iter().position(|j| j.settled).unwrap_or(0);
            st.ended.remove(at);
        }
    }

    /// The front end's status: the queue and holds; the top-level clip fields describe the first running clip, else the
    /// last that ended (as when there was one clip), `running_clips` is every clip running, `ended_clips` the latest ended.
    /// Writes a clip's record once it is done.
    pub(super) fn status(&self, st: &mut State) -> Value {
        let es = self.engine_state();
        self.refresh(st, es.as_ref());
        let mut out = serde_json::Map::new();
        let top = st.running.first().or(st.ended.last()).map(|j| j.view.clone());
        match top.as_ref().and_then(Value::as_object) {
            Some(v) => out.extend(v.clone()),
            None => {
                out.insert("idle".into(), json!(true));
                out.insert("job".into(), Value::Null); // nothing made since the studio started
            }
        }
        let room = self.room(st, es.as_ref());
        let recs: Vec<Value> = st.running.iter().map(|j| j.rec.clone()).collect();
        let plan = super::slots::plan(&st.q, &recs, room.free, &|i| self.has_own(i));
        let accepts = room.free > plan.starts.len();
        out.insert("running_clips".into(), json!(st.running.iter().map(|j| row(&j.view)).collect::<Vec<_>>()));
        out.insert("ended_clips".into(), json!(st.ended.iter().rev().take(MAX_ENDED).map(|j| row(&j.view)).collect::<Vec<_>>()));
        out.insert("slots_total".into(), json!(room.total));
        out.insert("slots_free".into(), json!(room.free));
        out.insert("slots_why".into(), json!(room.why));
        out.insert("accepts".into(), json!(accepts));
        out.extend(st.q.payload());
        out.insert("all_projects".into(), json!(self.all_projects(&st.q)));
        out.insert("llm".into(), self.llm.status());
        Value::Object(out)
    }

    /// One clip as the front end sees it (the legacy `status()`'s clip part): progress, stage, times, error; writes its record
    /// once it is done.
    fn view(&self, job: &mut Job, es: Option<&Value>, last_file: &mut Option<String>) -> Value {
        let mut out = serde_json::Map::new();
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
        let mut steps_total = job.rec["steps"].as_u64().unwrap_or(8);
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
        // a MovieRoll film (video.roll): its chunks made of those it has so far (a steered film's grow), and whether it waits
        // for its next event - its own lines, not H3's (2026-11-24: its status said "starting, 0 / 5" all the way)
        let film = film_progress(&job.rec, &j["progress"], &log);
        if let Some(f) = &film {
            (pct, steps_done, steps_total, stage) = (f.pct, f.done, f.total, f.stage.clone());
        }
        let done = state == "done";
        // the studio's watchdog stopped it (the engine hung): a failure, not a cancel - the daemon says cancelled because it
        // was sent a cancel
        let hung = job.rec["watchdog_stopped"].as_u64();
        let state = if hung.is_some() && state == "cancelled" { "failed" } else { state };
        let failed = matches!(state, "failed" | "cancelled");
        let name = s(&job.rec, "name");
        if done {
            pct = 100.0;
            stage = "done".into();
            *last_file = Some(name.clone());
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
        // this clip's card only: the other card's clip is not this one's energy. Not known while it has no card
        let tele = self.gpu();
        let pci = es.and_then(|e| e["gpus"].as_array()).and_then(|g| g.iter().find(|x| x["gpu"] == j["gpu"])).map(|g| s(g, "pci")).unwrap_or_default();
        let card = j["gpu"].as_u64().map(|g| g as usize);
        let e0 = card.and_then(|g| job.rec["energy0_cards"][g.to_string()].as_f64().or(if g == 0 && job.rec["energy0_cards"].is_null() { job.energy0 } else { None }));
        if let (Some(e0), Some(e1)) = (e0, card.and_then(|g| card_energy(&tele, g, &pci))) {
            if e1 >= e0 {
                energy = json!(((e1 - e0) / 3600.0 * 10.0).round() / 10.0);
                if !job.finished {
                    job.rec["energy_wh"] = energy.clone();
                }
            }
        }
        let error = if failed {
            let said = match hung {
                Some(min) => format!(": the engine hung (no output for {min} min); the studio stopped it and the worker was restarted"),
                None => j["error"].as_str().map(|e| format!(": {e}")).unwrap_or_default(),
            };
            Some(format!("{}{said}\n{}", if state == "cancelled" { "cancelled" } else { "failed" },
                         log.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n")))
        } else {
            None
        };
        if failed {
            stage = "failed".into();
        }
        let elapsed = now() - job.rec["started"].as_f64().unwrap_or(now());
        // a film's pace counts from its first chunk (the load and the first window before it)
        let mut film = film;
        if let Some(f) = film.as_mut() {
            if f.done >= 1 && job.rec["first_chunk_s"].is_null() {
                job.rec["first_chunk_s"] = json!(elapsed);
            }
            f.first_s = job.rec["first_chunk_s"].as_f64();
        }
        let frames = frames_of(job.rec["seconds"].as_f64().unwrap_or(5.0));
        let regen = !s(&job.rec, "regen2k").is_empty();
        let eta = if done || failed {
            0.0
        } else if let Some(f) = &film {
            f.eta(elapsed)
        } else if regen {
            // its steps as they ended ("step   : 1/3 in 581.5 s (window 1/2)"): the pace of the rest
            let step_s: Vec<f64> = log.iter().filter(|l| l.starts_with("step   :")).filter_map(|l| l.split(" in ").nth(1)?.split_whitespace().next()?.parse().ok()).collect();
            let frac = j["progress"]["done"].as_f64().zip(j["progress"]["total"].as_f64()).filter(|(_, t)| *t > 0.0).map(|(d, t)| (d / t).clamp(0.0, 1.0));
            regen_eta(&job.rec, elapsed, has("tokens :"), frac, &step_s, has("sampled:"))
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
        let kind = if failed { failure(state, j["error"].as_str().unwrap_or(""), &log, hung) } else { Value::Null };
        if failed {
            job.finished = true;
            // a failed or cancelled clip's record, once, kept out of the clips directory (its listing is the clips made):
            // video.jobs.get answers with it after the studio's ended list forgot it, or after a restart
            if !job.saved && library::is_clip_name(&name) {
                job.saved = true;
                let side = json!({"state": if state == "cancelled" { "cancelled" } else { "failed" }, "name": name, "job": job.rec, "error": error, "failure": kind,
                                  "frames": frames, "wall": (elapsed * 10.0).round() / 10.0, "energy_wh": energy,
                                  "engine_log": log.iter().rev().take(12).rev().collect::<Vec<_>>()});
                let dir = self.dir.join(super::FAILED_DIR);
                let _ = std::fs::create_dir_all(&dir);
                let _ = write_atomic(&dir.join(format!("{name}.json")), &serde_json::to_vec_pretty(&side).unwrap_or_default());
            }
        }
        let rec = job.rec.clone();
        let idle = done || failed;
        out.insert("idle".into(), json!(idle));
        out.insert("job".into(), rec);
        out.insert("pct".into(), json!(pct.round() as i64));
        out.insert("stage".into(), json!(stage));
        out.insert("done".into(), json!(done));
        out.insert("error".into(), json!(error));
        // the same as data: what kind of failure, and whether running it again as is can help
        out.insert("failure".into(), kind);
        // the card the engine runs it on (while it waits for one: the one it asked for, if any)
        out.insert("gpu".into(), if idle { Value::Null } else { j["gpu"].clone() });
        out.insert("waiting_for".into(), if idle { Value::Null } else { es.map_or(Value::Null, |e| super::waiting_on(e, &j["gpu"])) });
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
        // a MovieRoll film (video.roll): its live HLS playlist, growing while it is made (fragmented-MP4 segments of 2 s),
        // from its first segment on
        out.insert("live".into(), live_url(&self.out, &name, job.rec["hls"] == true));
        // while it samples: the engine's picture of the shot so far (its latents' middle frame, each step)
        let pv = self.out.join(format!("{name}.preview.png"));
        let pv_at = (!idle).then(|| std::fs::metadata(&pv).and_then(|m| m.modified()).ok()).flatten()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs());
        out.insert("preview".into(), pv_at.map_or(Value::Null, |t| json!(format!("/out/{name}.preview.png?t={t}"))));
        out.insert("engine_log".into(), json!(log.iter().rev().take(12).rev().collect::<Vec<_>>()));
        Value::Object(out)
    }

    pub(super) fn running(&self, st: &State) -> bool {
        !st.running.is_empty()
    }

    /// Stop running clips (at the engine's next block boundary): those `pick` chooses
    pub(super) fn cancel(&self, st: &mut State, pick: &dyn Fn(&Job) -> bool) -> Result<usize, String> {
        let mut n = 0;
        for j in st.running.iter_mut().filter(|j| pick(j)) {
            j.item = None; // no retry
            self.engine("POST", &format!("/engine/jobs/{}/cancel", j.engine_id), None)?;
            n += 1;
        }
        Ok(n)
    }

    /// What ended since the last look: one failure or more in a pass count once toward the hold (one engine crash can end two
    /// clips), each goes round once more, ahead of the queue; a second failure of the same clip is not run again.
    fn settle(&self, st: &mut State) {
        let mut again = Vec::new();
        let mut failed = false;
        for j in st.ended.iter_mut().filter(|j| !j.settled) {
            j.settled = true;
            if j.view["done"] == true {
                st.fails = 0;
            } else if j.view["error"].is_string() {
                if let Some(mut it) = j.item.take() {
                    failed = true;
                    if !j.rec["retry"].as_bool().unwrap_or(false) {
                        it["retried"] = json!(true);
                        again.push(it);
                        j.retrying = true;
                    }
                }
            }
        }
        self.drop_dependents(st);
        if !failed {
            return;
        }
        st.fails += 1;
        let hold = st.fails > 1;
        for it in again.into_iter().rev() {
            st.q.items.insert(0, it);
        }
        st.q.save(&self.dir);
        if hold {
            st.q.paused = true;
            st.fails = 0;
            st.q.save_paused(&self.dir);
            eprintln!("queue: two failures in a row, holding the queue");
        }
    }

    /// Queued clips whose predecessor failed for good or was cancelled leave the queue with that as their answer (status by
    /// label or reentrant_id, ended_clips); nothing makes them from an older clip. Those behind them in the series go too.
    fn drop_dependents(&self, st: &mut State) {
        let mut at = 0;
        while at < st.q.items.len() {
            let Some(on) = self.failed_dep(st, &st.q.items[at]) else {
                at += 1;
                continue;
            };
            let item = st.q.items.remove(at);
            let message = format!("{on} failed or was cancelled and this clip needs its output: nothing was started");
            eprintln!("queue: {} dropped: {message}", s(&item, "label"));
            let view = json!({"idle": true, "job": item, "stage": "failed", "pct": 0, "done": false, "error": format!("failed: {message}"), "exited": true, "rc": 1, "eta": 0, "elapsed": 0,
                              "gpu": null, "waiting_for": null, "preview": null, "file": null,
                              "failure": {"code": "dependency_failed", "retryable": false, "message": message, "hint": "make that clip again, then this one"}});
            st.ended.push(Job { rec: item, engine_id: 0, energy0: None, saved: false, stall: (0, now()), finished: true, last: Value::Null, item: None, settled: true, view, retrying: false });
            st.q.save(&self.dir);
        }
    }

    /// One scheduler tick (every second).
    pub(super) fn tick(&self) {
        self.llm.idle_watchdog(self);
        let mut st = self.st.lock().unwrap();
        let es = self.engine_state();
        self.refresh(&mut st, es.as_ref());
        // the watchdog: no new log line and no progress for the stall time -> stop it (becomes a failure -> the retry path).
        // Progress counts: a 2K regeneration's step is ~10 min of blocks with no line between (its engine reports each block)
        for j in &mut st.running {
            let n = j.last["log"].as_array().map_or(0, |a| a.len()) + j.last["progress"]["done"].as_u64().unwrap_or(0) as usize * 100_000;
            if n != j.stall.0 {
                j.stall = (n, now());
            } else if now() - j.stall.1 > self.stall_s as f64 && j.last["state"] == "running" {
                eprintln!("watchdog: {} wrote nothing for {} min, stopping it", s(&j.rec, "name"), self.stall_s / 60);
                // not a user's cancel: the clip ends as a retryable failure, engine_hung (`failure`), never as cancelled
                j.rec["watchdog_stopped"] = json!(self.stall_s / 60);
                let id = j.engine_id;
                let _ = self.engine("POST", &format!("/engine/jobs/{id}/cancel"), None);
                j.stall.1 = now();
            }
        }
        if st.q.items.is_empty() || st.q.items.iter().all(|i| st.q.held(i)) {
            return;
        }
        self.settle(&mut st);
        let room = self.room(&st, es.as_ref());
        let recs: Vec<Value> = st.running.iter().map(|j| j.rec.clone()).collect();
        let plan = super::slots::plan(&st.q, &recs, room.free, &|i| self.has_own(i));
        if plan.starts.is_empty() {
            return;
        }
        let mut items: Vec<Value> = Vec::new();
        for i in plan.starts.iter().rev() {
            items.insert(0, st.q.items.remove(*i));
        }
        st.q.save(&self.dir);
        let mut left = items.into_iter();
        while let Some(mut item) = left.next() {
            let retry = item.as_object_mut().and_then(|o| o.remove("retried")).is_some();
            if retry {
                item["retry"] = json!(true);
            }
            if let Err(e) = self.launch(&mut st, item.clone()) {
                eprintln!("queue: launching failed ({e}); back at the front, queue held");
                if let Some(o) = item.as_object_mut() {
                    o.remove("retry");
                }
                let back: Vec<Value> = std::iter::once(item).chain(left).collect();
                for it in back.into_iter().rev() {
                    st.q.items.insert(0, it);
                }
                st.q.paused = true;
                st.q.save(&self.dir);
                st.q.save_paused(&self.dir);
                return;
            }
        }
    }

    pub fn scheduler(self: std::sync::Arc<Self>) {
        // every second, not every 5: a tick is a status call over the engine's socket, and at 5 s a cancel, a finished
        // clip or the next queued one showed on the page up to 5 s late
        loop {
            std::thread::sleep(Duration::from_secs(1));
            self.tick();
        }
    }
}

/// A 2K regeneration's whole time, before it has a history: measured on the B65 2026-11-17 with the fast decoder (the
/// card's default; docs/ports/open-h3-regenerate-2k.md, 5.17 s and 15.08 s drafts) - per window ~180 s (text encoder,
/// vision tower, latent upscaler) and ~625 s a step of 177k tokens, per draft frame ~1.05 s (the VAE encode ~0.75, the
/// fast 2K decode with its load ~0.3), plus ~40 s of LoRAs and time tables. (The H3 video decoder adds ~0.8 s a frame.)
pub fn regen_estimate_s(frames: f64, windows: f64, steps_per_window: f64) -> f64 {
    40.0 + windows * (180.0 + 625.0 * steps_per_window) + 1.05 * frames
}

/// A running regeneration's time left, the whole job's (never one stage's): before a step has ended, the first estimate
/// less the time gone; once one has, the steps' own pace for the steps left (`frac`: the share of all windows' blocks
/// done), the text encoder of each window still to start, and the decode
fn regen_eta(rec: &Value, elapsed: f64, sampling: bool, frac: Option<f64>, step_s: &[f64], sampled: bool) -> f64 {
    let frames = rec["frames"].as_f64().unwrap_or(124.0);
    let windows = rec["regen_windows"].as_f64().unwrap_or(1.0).max(1.0);
    let steps = rec["steps"].as_f64().unwrap_or(3.0).max(1.0);
    // the fast decoder with its load and the mp4 (~0.3 s a frame at 2K)
    let decode = (0.3 * frames).max(15.0);
    if sampled {
        return decode;
    }
    let first = (regen_estimate_s(frames, windows, steps / windows) - elapsed).max(60.0);
    match (sampling, frac) {
        (true, Some(f)) if !step_s.is_empty() => {
            let per = step_s.iter().sum::<f64>() / step_s.len() as f64;
            let done = (f * steps).max(step_s.len() as f64).min(steps);
            let windows_started = ((done / (steps / windows)).floor() + 1.0).min(windows);
            per * (steps - done) + 180.0 * (windows - windows_started) + decode
        }
        _ => first,
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

/// The records of the clips being made when the studio stopped (a studio from before two clips at once left `job.json`).
pub fn recover(dir: &Path) -> Vec<Value> {
    if let Some(v) = std::fs::read(dir.join("jobs.json")).ok().and_then(|b| serde_json::from_slice::<Vec<Value>>(&b).ok()) {
        return v;
    }
    std::fs::read(dir.join("job.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).into_iter().collect()
}

/// Whether a clip's worker has loaded: its drop in host memory has happened (the text encoder's line comes after the load)
fn worker_loaded(last: &Value) -> bool {
    last["log"].as_array().is_some_and(|l| l.iter().any(|x| x.as_str().is_some_and(|x| x.starts_with("tokens :") || x.starts_with("encoded:"))))
}

/// The telemetry's energy counter of each card, by index (a gpustat from before per-card counters: only the first's)
fn card_energies(tele: &Value) -> Value {
    let mut m = serde_json::Map::new();
    for (i, c) in tele["cards"].as_array().into_iter().flatten().enumerate() {
        if let Some(e) = c["energy_j"].as_f64() {
            m.insert(i.to_string(), json!(e));
        }
    }
    if m.is_empty() {
        if let Some(e) = tele["energy_j"].as_f64() {
            m.insert("0".into(), json!(e));
        }
    }
    Value::Object(m)
}

/// One card's counter now: by PCI address, else by index; the telemetry's top line is its first card
fn card_energy(tele: &Value, gpu: usize, pci: &str) -> Option<f64> {
    let cards = tele["cards"].as_array();
    let row = cards.and_then(|c| c.iter().find(|x| !pci.is_empty() && x["pci"] == pci).or_else(|| c.get(gpu)));
    row.and_then(|r| r["energy_j"].as_f64()).or(if gpu == 0 || (!pci.is_empty() && tele["pci"] == pci) { tele["energy_j"].as_f64() } else { None })
}

/// A MovieRoll film's progress from its engine job: chunks made of those it has (progress done / total), its stage
pub struct FilmProgress {
    pub done: u64,
    pub total: u64,
    pub pct: f64,
    pub stage: String,
    pub waiting: bool,
    /// seconds from the job's start to its first chunk's (the load, the text, the first window) - unknown before it
    pub first_s: Option<f64>,
}

impl FilmProgress {
    /// The seconds left: at its pace a chunk so far (after the first's), the load and the first window's estimate before
    /// (~60 s, 17 s a chunk on the B65); a live film waiting for an event: what is left of its chunks (the wait is not
    /// known)
    pub fn eta(&self, elapsed: f64) -> f64 {
        let left = self.total.saturating_sub(self.done) as f64;
        match (self.done, self.first_s) {
            (d, Some(f)) if d > 1 && elapsed > f => (elapsed - f) / (d - 1) as f64 * left,
            _ => (60.0 - elapsed).max(0.0) + 17.0 * left,
        }
    }
}

pub fn film_progress(rec: &Value, progress: &Value, log: &[String]) -> Option<FilmProgress> {
    if !rec["events"].is_array() {
        return None;
    }
    let (done, total) = (progress["done"].as_u64().unwrap_or(0), progress["total"].as_u64().unwrap_or(0));
    let waiting = log.iter().rev().find(|l| l.contains("movieroll:")).is_some_and(|l| l.contains("live - "));
    let stage = if waiting {
        format!("live: {done} of {total} chunks made, waiting for the next event (video.rollAppend)")
    } else if total == 0 {
        "starting".into()
    } else {
        format!("rolling: {done} of {total} chunks made")
    };
    let pct = if total == 0 { 2.0 } else { 3.0 + 96.0 * done as f64 / total as f64 };
    let first_s = rec["first_chunk_s"].as_f64();
    Some(FilmProgress { done, total, pct, stage, waiting, first_s })
}

/// A film's live HLS playlist (`/out/<name>.live.m3u8`) once it lists a segment, else null: a player given it sooner
/// gets a 404 or an empty list. 2026-11-24: status said nothing of it (running_clips' rows left it out)
pub fn live_url(out: &std::path::Path, name: &str, hls: bool) -> Value {
    let m3u8 = format!("{name}.live.m3u8");
    let listed = hls && std::fs::read_to_string(out.join(&m3u8)).is_ok_and(|t| t.lines().any(|l| l.starts_with("#EXTINF")));
    if listed { json!(format!("/out/{m3u8}")) } else { Value::Null }
}

/// A clip's line in `running_clips` and `ended_clips`
fn row(v: &Value) -> Value {
    let j = &v["job"];
    json!({"label": j["label"], "title": j["title"], "name": j["name"], "project": j["project"], "reentrant_id": j["reentrant_id"], "engine": j["engine"], "seconds": j["seconds"], "steps": j["steps"],
           "width": j["width"], "height": j["height"], "energy_wh": j["energy_wh"], "job": j,
           "stage": v["stage"], "pct": v["pct"], "eta": v["eta"], "elapsed": v["elapsed"], "gpu": v["gpu"], "waiting_for": v["waiting_for"], "preview": v["preview"],
           "steps_done": v["steps_done"], "steps_total": v["steps_total"], "frames": v["frames"], "idle": v["idle"], "done": v["done"], "error": v["error"], "failure": v["failure"], "file": v["file"],
           "live": v["live"]})
}

/// A failed clip's kind: {code, retryable, message, hint}. Read from the engine's error and its last lines. `hung`: the
/// studio's watchdog stopped it after that many minutes without output - an engine that hung, whatever the daemon calls it
pub fn failure(state: &str, error: &str, log: &[String], hung: Option<u64>) -> Value {
    if let Some(min) = hung {
        // 2026-11-24 job 134: a clip silent for 20 min was stopped by the watchdog and its worker killed; the platform read
        // "stopped by video.cancel" and ended the take for good
        let message = format!("the engine hung (no output for {min} min); the studio stopped it and the worker was restarted{}",
                              if error.is_empty() { String::new() } else { format!(" ({error})") });
        return json!({"code": "engine_hung", "retryable": true, "message": message, "hint": "the same request may work on the fresh worker"});
    }
    let text = format!("{error}\n{}", log.iter().rev().take(8).cloned().collect::<Vec<_>>().join("\n")).to_lowercase();
    let has = |ps: &[&str]| ps.iter().any(|p| text.contains(p));
    let (code, retryable, hint) = if state == "cancelled" || error.starts_with("cancelled") {
        ("cancelled", false, "stopped by video.cancel")
    } else if error.starts_with("no_room") {
        // the card shared with other work right now (the clip fits it empty), or no room came within the worker's wait
        ("no_room", true, "the card is shared with other work right now; the same request fits once it is gone - another card or a later try")
    } else if error.contains("too long for") {
        ("too_long", false, "the clip does not fit on the card: fewer frames (the message says the most that fit), a smaller canvas or a smaller engine")
    } else if crate::video::daemon::allocation_failure(error) {
        // the daemon ends a worker the card refused memory and the next job gets a fresh one (whatever the old one held
        // outside its book goes with it): the same clip may well fit there
        ("out_of_memory", true, "the card refused memory; the worker was restarted and the same request may work on the fresh one")
    } else if has(&["memory cap", "out of memory", "out_of_device_memory", "out of device memory", "refused a scratch buffer", "cannot allocate"]) {
        ("out_of_memory", false, "the same clip will not fit again: fewer frames, a smaller canvas or a smaller engine (q6k, q4km)")
    } else if has(&["does not answer", "ended unexpectedly", "does not listen", "closed its output", "broken pipe", "connection reset", "killed"]) {
        ("engine_lost", true, "the engine process ended or stopped answering mid-clip; the same request may work")
    } else if has(&["no such file", "not found", "must be", "a multiple of", "cannot read", "cannot decode", "invalid"]) {
        ("invalid_input", false, "a file or value the engine could not use: change the request")
    } else {
        ("engine_error", false, "the engine's own error: see message")
    };
    json!({"code": code, "retryable": retryable, "message": error, "hint": hint})
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_films_status_says_its_chunks_its_wait_and_its_pace() {
        use super::film_progress;
        use serde_json::json;
        let rec = json!({"events": [{"prompt": "a fox", "seconds": 3}]});
        let log = |l: &str| vec!["movieroll: 864x480".to_string(), l.to_string()];
        // not a film: none
        assert!(film_progress(&json!({}), &json!({"done": 1, "total": 2}), &[]).is_none());
        let f = film_progress(&rec, &json!({"done": 3, "total": 10}), &log("movieroll: KV recompute at 3 chunks")).unwrap();
        assert_eq!((f.done, f.total, f.waiting), (3, 10, false));
        assert!(f.stage.contains("3 of 10") && (f.pct - (3.0 + 96.0 * 0.3)).abs() < 1e-9, "{}", f.stage);
        // live, out of events: waiting, said so
        let f = film_progress(&rec, &json!({"done": 5, "total": 5}), &log("movieroll: live - waiting for the next event (60 s)")).unwrap();
        assert!(f.waiting && f.stage.contains("waiting for the next event"), "{}", f.stage);
        // the pace: from the first chunk on (here 20 s a chunk after a first at 100 s)
        let mut f = film_progress(&rec, &json!({"done": 3, "total": 10}), &[]).unwrap();
        f.first_s = Some(100.0);
        assert!((f.eta(140.0) - 140.0).abs() < 1e-9);
    }

    #[test]
    fn a_films_live_playlist_shows_from_its_first_segment_in_status_and_its_row() {
        use super::{live_url, row};
        use serde_json::json;
        let d = std::env::temp_dir().join(format!("ns-live-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let name = "h3_20261124_120000";
        // not yet written, or written without a segment: nothing to play
        assert!(live_url(&d, name, true).is_null());
        std::fs::write(d.join(format!("{name}.live.m3u8")), "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-MAP:URI=\"x.init.mp4\"\n").unwrap();
        assert!(live_url(&d, name, true).is_null());
        // the first segment listed: the playlist
        std::fs::write(d.join(format!("{name}.live.m3u8")), "#EXTM3U\n#EXTINF:2.0,\nx.live.0000.m4s\n").unwrap();
        assert_eq!(live_url(&d, name, true), json!(format!("/out/{name}.live.m3u8")));
        // a clip without HLS has none
        assert!(live_url(&d, name, false).is_null());
        // and running_clips' rows carry it (the watcher matched rows by label and never saw it)
        let v = json!({"job": {"label": "fox film", "name": name}, "live": live_url(&d, name, true)});
        assert_eq!(row(&v)["live"], json!(format!("/out/{name}.live.m3u8")));
        let _ = std::fs::remove_dir_all(&d);
    }

    use super::failure;
    use super::regen_eta;
    use serde_json::json;

    /// A regeneration's eta is the whole job's time left and only falls: the platform saw 2,200 s at the text encoder,
    /// then 210 s at sampling 1/3, then 785 (one stage's pace taken for the job's)
    #[test]
    fn a_regeneration_eta_is_the_whole_jobs_and_falls() {
        let rec = json!({"frames": 124, "regen_windows": 1, "steps": 3});
        let te = regen_eta(&rec, 300.0, false, None, &[], false);
        let early = regen_eta(&rec, 420.0, true, Some(0.05), &[], false);
        let one = regen_eta(&rec, 1000.0, true, Some(0.34), &[580.0], false);
        let two = regen_eta(&rec, 1580.0, true, Some(0.67), &[580.0, 580.0], false);
        let decoding = regen_eta(&rec, 2160.0, true, Some(1.0), &[580.0, 580.0, 580.0], true);
        assert!(te > early && early > one && one > two && two > decoding, "{te} {early} {one} {two} {decoding}");
        assert!(early > 1500.0, "before a step has ended: the estimate, not a fraction's pace ({early})");
        assert!((1170.0..1230.0).contains(&one), "two steps of 580 s and the fast decode ({one})");
        let rec2 = json!({"frames": 185, "regen_windows": 2, "steps": 6});
        let w1 = regen_eta(&rec2, 1500.0, true, Some(0.30), &[600.0], false);
        assert!(w1 > 4.0 * 600.0 + 180.0, "the second window's steps and text encoder still ahead ({w1})");
    }

    #[test]
    fn failures_say_whether_to_retry() {
        let log = |l: &str| vec![l.to_string()];
        // a refusal recycles the worker (the daemon): the clip may fit on the fresh one - retryable, never a cancel
        let f = failure("failed", "nsd_alloc: 900 MiB would pass the device memory cap (30 GiB)", &[], None);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("out_of_memory"), Some(true)));
        let f = failure("failed", "nsd_alloc: the device refused 28 MiB", &[], None);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("out_of_memory"), Some(true)));
        // too long for the card: the same request never fits
        let f = failure("failed", "736 frames at 832x480 (87084 tokens) are too long for this card: ...; at most 600 frames (25.00 s) fit", &[], None);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("too_long"), Some(false)));
        // fits the card empty, not now (the card shared): retryable, never too_long
        let f = failure("failed", "no_room: the card is shared with other work right now (7.0 GiB of it held outside this engine): ...", &[], None);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("no_room"), Some(true)));
        let f = failure("failed", "the engine process ended unexpectedly (signal: 9)", &[], None);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("engine_lost"), Some(true)));
        let f = failure("cancelled", "", &log("sampling"), None);
        assert_eq!(f["code"], "cancelled");
        let f = failure("failed", "first_frame: no such file", &[], None);
        assert_eq!(f["code"], "invalid_input");
        assert_eq!(failure("failed", "something odd", &[], None)["code"], "engine_error");
    }

    #[test]
    fn a_clip_the_watchdog_stopped_is_a_retryable_hang_not_a_cancel() {
        // job 134 as the daemon ended it: cancelled (it was sent a cancel), the worker killed 180 s later
        let f = failure("cancelled", "the engine process did not stop 180 s after the cancel and was killed", &["tokens : 32870 in the denoiser".into()], Some(20));
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("engine_hung"), Some(true)));
        let m = f["message"].as_str().unwrap();
        assert!(m.starts_with("the engine hung") && m.contains("the worker was restarted") && !m.contains("video.cancel"), "{m}");
        // one that stopped at the next block after the watchdog's cancel: the same
        assert_eq!(failure("cancelled", "cancelled", &[], Some(20))["code"], "engine_hung");
        // a user's cancel stays a cancel
        assert_eq!(failure("cancelled", "cancelled", &[], None)["code"], "cancelled");
    }
}
