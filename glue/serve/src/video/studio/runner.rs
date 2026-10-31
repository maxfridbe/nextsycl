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
use super::{now, write_atomic, Job, State, Studio, MAX_ENDED};

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
        // the same as data: what kind of failure, and whether running it again as is can help
        out.insert("failure".into(), if failed { failure(state, j["error"].as_str().unwrap_or(""), &log) } else { Value::Null });
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
        // the watchdog: no new log line for the stall time -> stop it (becomes a failure -> the retry path)
        for j in &mut st.running {
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

/// A clip's line in `running_clips` and `ended_clips`
fn row(v: &Value) -> Value {
    let j = &v["job"];
    json!({"label": j["label"], "name": j["name"], "project": j["project"], "reentrant_id": j["reentrant_id"], "engine": j["engine"], "seconds": j["seconds"], "steps": j["steps"],
           "width": j["width"], "height": j["height"], "energy_wh": j["energy_wh"], "job": j,
           "stage": v["stage"], "pct": v["pct"], "eta": v["eta"], "elapsed": v["elapsed"], "gpu": v["gpu"], "waiting_for": v["waiting_for"], "preview": v["preview"],
           "steps_done": v["steps_done"], "steps_total": v["steps_total"], "frames": v["frames"], "idle": v["idle"], "done": v["done"], "error": v["error"], "failure": v["failure"], "file": v["file"]})
}

/// A failed clip's kind: {code, retryable, message, hint}. Read from the engine's error and its last lines.
pub fn failure(state: &str, error: &str, log: &[String]) -> Value {
    let text = format!("{error}\n{}", log.iter().rev().take(8).cloned().collect::<Vec<_>>().join("\n")).to_lowercase();
    let has = |ps: &[&str]| ps.iter().any(|p| text.contains(p));
    let (code, retryable, hint) = if state == "cancelled" || error.starts_with("cancelled") {
        ("cancelled", false, "stopped by video.cancel")
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
    use super::failure;

    #[test]
    fn failures_say_whether_to_retry() {
        let log = |l: &str| vec![l.to_string()];
        let f = failure("failed", "nsd_alloc: 900 MiB would pass the device memory cap (30 GiB)", &[]);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("out_of_memory"), Some(false)));
        let f = failure("failed", "the engine process ended unexpectedly (signal: 9)", &[]);
        assert_eq!((f["code"].as_str(), f["retryable"].as_bool()), (Some("engine_lost"), Some(true)));
        let f = failure("cancelled", "", &log("sampling"));
        assert_eq!(f["code"], "cancelled");
        let f = failure("failed", "first_frame: no such file", &[]);
        assert_eq!(f["code"], "invalid_input");
        assert_eq!(failure("failed", "something odd", &[])["code"], "engine_error");
    }
}
