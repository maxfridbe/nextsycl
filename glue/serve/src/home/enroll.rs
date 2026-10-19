//! A saved voice enrolled for a speech model without an encoder (Voxtral TTS: its open weights have none, so a voice
//! is fitted through its decoder - `nextsycl audio voice enroll`, experimental) as a front-door job:
//! `audio.voices.enroll {name}` (the audio page's "enroll for Voxtral" too). The job asks for a card like any load:
//! one of the model's cards (or `gpu`) with the fit's memory and 1.5 GiB more free - idle work there stops for it -,
//! never one a load is under way on, nothing while a GPU run holds the cards (crate::gpulock); while it runs its
//! memory counts as taken there (life.rs free_mb). Its progress is the fit's step; it ends with the voice enrolled
//! (`<voice>/enrolled/<arch>.npy`) and offered to the model (audio.voices.list: models).

use std::io::{BufRead, Seek, SeekFrom};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::life::s;
use super::{now, Home};

type Fail = (u16, String);

/// What a fit takes on its card, MiB (a 15 s clip through the decoder on a B65: ~6 GiB; the command's ENROLL_GIB)
pub(super) const ENROLL_MB: u64 = 8192;

/// The fit's steps by default and at most
const STEPS: u64 = 2000;
const MAX_STEPS: u64 = 20000;

/// The step a line of the fit's output reports (enroll.py prints {"step": N, "loss": ...} every 50)
pub(super) fn step_of(line: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    v["step"].as_u64()
}

/// The model an enrolment is for: the one named, else the first registered speech model that speaks enrolled voices
pub(super) fn enroll_model<'a>(b: &Value, all: &'a [Value]) -> Result<&'a Value, Fail> {
    let speaks = |m: &&Value| super::registry::kind_of(m) == "audio" && m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "enrolled"));
    match b["model"].as_str().filter(|x| !x.is_empty()) {
        Some(id) => {
            let m = all.iter().find(|m| m["id"] == id).ok_or_else(|| (404, format!("model {id}: no such model (system.models)")))?;
            if !speaks(&m) {
                return Err((400, format!("model {id}: it does not speak enrolled voices (Voxtral TTS does)")));
            }
            Ok(m)
        }
        None => all.iter().find(speaks).ok_or_else(|| (409, "no registered speech model speaks enrolled voices (register voxtral-4b-tts)".to_string())),
    }
}

impl Home {
    /// audio.voices.enroll checked at submit: the voice, the model, the card, the steps
    pub(super) fn precheck_enroll(&self, b: &Value, all: &[Value]) -> Result<(), Fail> {
        let name = s(b, "name");
        if self.voices().sample_path(&name).is_none() {
            return Err((404, format!("name {name}: no saved voice of that name (audio.voices.list)")));
        }
        let m = enroll_model(b, all)?;
        if s(m, "arch") != "voxtral-tts" {
            return Err((400, format!("model {}: only Voxtral TTS (voxtral-tts) enrolls voices", s(m, "id"))));
        }
        if let Some(g) = b.get("gpu").filter(|g| !g.is_null()) {
            if g.as_u64().is_none_or(|g| g as usize >= self.ncards()) {
                return Err((400, format!("gpu {g}: one of 0..{}", self.ncards().saturating_sub(1))));
            }
        }
        if let Some(n) = b.get("steps").filter(|g| !g.is_null()) {
            if n.as_u64().is_none_or(|n| !(50..=MAX_STEPS).contains(&n)) {
                return Err((400, format!("steps {n}: 50 to {MAX_STEPS}")));
            }
        }
        Ok(())
    }

    /// The job's work: a card, then the fit (`nextsycl audio voice enroll NAME --direct --gpu N`) to its end
    pub(super) fn enroll_voice(self: &Arc<Self>, b: &Value, all: &[Value]) -> Result<Value, Fail> {
        self.precheck_enroll(b, all)?;
        let name = s(b, "name");
        let m = enroll_model(b, all)?;
        let (id, arch) = (s(m, "id"), s(m, "arch"));
        let job = b["request_id"].as_str().map(str::to_string);
        let job = job.as_deref();
        let lib = self.voices();
        if lib.enrolled_path(&name, &arch).is_some() && b["replace"] != true {
            return Ok(json!({"voice": name, "model": id, "enrolled": lib.enrolled(&name), "already": true}));
        }
        let gpus: Vec<usize> = match b["gpu"].as_u64() {
            Some(g) => vec![g as usize],
            None => {
                let g = self.gpus_of(m);
                if g.is_empty() { (0..self.ncards()).collect() } else { g }
            }
        };
        let gpu = self.enroll_card(&name, &gpus, job)?;
        let steps = b["steps"].as_u64().unwrap_or(STEPS);
        let r = self.enroll_run(&name, gpu, steps, job);
        self.pending.lock().unwrap().retain(|p| !(p.0 == gpu && p.1 == 0));
        let secs = r?;
        let report: Value = lib.enrolled_path(&name, &arch).and_then(|p| std::fs::read(p.with_extension("json")).ok())
            .and_then(|t| serde_json::from_slice(&t).ok()).unwrap_or(Value::Null);
        let mut report = report;
        if let Some(o) = report.as_object_mut() {
            o.remove("history");
        }
        Ok(json!({"voice": name, "model": id, "arch": arch, "gpu": gpu, "steps": steps, "seconds": secs.round(),
                  "enrolled": lib.enrolled(&name), "report": report}))
    }

    /// A card for the fit: the first of `gpus` with its memory and the spare free (idle work there stopped for it),
    /// no load under way on it, no GPU run holding the cards; reserved (life.rs free_mb) until the fit ends. Waits
    /// as a job waits for room (cancel ends it)
    fn enroll_card(&self, name: &str, gpus: &[usize], job: Option<&str>) -> Result<usize, Fail> {
        let want = ENROLL_MB + super::life::MARGIN_MB;
        let what = format!("enrolling {name}");
        let mut t0 = Instant::now();
        loop {
            if job.is_some_and(|j| self.job_cancelled(j)) {
                return Err((499, format!("cancelled while {what} waited for a card")));
            }
            if let Some(why) = self.cards_taken() {
                self.note(job, "cards_taken", &why);
                std::thread::sleep(Duration::from_secs(3));
                t0 = Instant::now();
                continue;
            }
            let cards = self.named_cards(&self.models());
            let loading = |g: usize| self.pending.lock().unwrap().iter().any(|p| p.0 == g && p.3.is_none());
            let plan = gpus.iter().copied().filter(|g| !loading(*g)).find_map(|g| self.room_plan(&cards, g, want, "enroll", false).map(|p| (g, p)));
            if let Some((g, plan)) = plan {
                if self.free(&plan, "enroll", &what).is_ok() && self.wait_room(g, want) {
                    let mut p = self.pending.lock().unwrap();
                    // (a load that started meanwhile on that card: the next try)
                    if !p.iter().any(|x| x.0 == g && x.3.is_none()) {
                        p.push((g, 0, ENROLL_MB, None));
                        return Ok(g);
                    }
                }
            }
            let busy: Vec<usize> = gpus.iter().copied().filter(|g| loading(*g)).collect();
            let why = if busy.len() == gpus.len() {
                format!("a model loading on GPU {}; {what} waits for it", busy.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", "))
            } else {
                self.no_room(&cards, gpus, name, want)
            };
            self.note(job, "waiting", &why);
            if t0.elapsed() > Duration::from_secs(super::life::JOB_ROOM_WAIT_S) {
                return Err((503, format!("{why}; try again when one of them idles out or is stopped")));
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    }

    /// The fit on card `gpu`, its progress noted from its output (a run on the page), a cancel stopping its
    /// container; its seconds
    fn enroll_run(&self, name: &str, gpu: usize, steps: u64, job: Option<&str>) -> Result<f64, Fail> {
        let (rid, log) = self.new_run(format!("audio voice enroll {name} on GPU {gpu}"));
        let f = std::fs::File::create(&log).map_err(|e| (500, format!("{}: {e}", log.display())))?;
        let f2 = f.try_clone().map_err(|e| (500, e.to_string()))?;
        let mut child = std::process::Command::new(&self.o.exe)
            .args(["audio", "voice", "enroll", name, "--direct", "--gpu", &gpu.to_string(), "--steps", &steps.to_string()])
            .stdin(std::process::Stdio::null()).stdout(f).stderr(f2).spawn()
            .map_err(|e| {
                self.finish(rid, -1);
                (500, format!("audio voice enroll: {e}"))
            })?;
        let t0 = Instant::now();
        self.note(job, "running", &format!("enrolling {name} on GPU {gpu}: starting"));
        let mut read = std::fs::File::open(&log).ok();
        let mut at = 0u64;
        let mut stopped = false;
        let st = loop {
            if let Ok(Some(st)) = child.try_wait() {
                break st;
            }
            if let Some(r) = read.as_mut() {
                let _ = r.seek(SeekFrom::Start(at));
                let mut lines = std::io::BufReader::new(&mut *r);
                let mut line = String::new();
                while lines.read_line(&mut line).is_ok_and(|n| n > 0) {
                    if !line.ends_with('\n') {
                        break;
                    }
                    at += line.len() as u64;
                    if let Some(k) = step_of(&line) {
                        self.set_progress(job, json!({"phase": "fit", "at": k + 1, "of": steps, "seconds": t0.elapsed().as_secs()}));
                        self.note(job, "running", &format!("enrolling {name} on GPU {gpu}: step {} of {steps}", k + 1));
                    }
                    line.clear();
                }
            }
            if !stopped && job.is_some_and(|j| self.job_cancelled(j)) {
                // its container (the command names it nextsycl-enroll-<gpu>) stopped; the command then ends
                let ce = self.o.cfg.get("NS_CONTAINER_ENGINE").unwrap_or_else(|| "podman".into());
                let _ = std::process::Command::new(ce).args(["stop", "-t", "5", &format!("nextsycl-enroll-{gpu}")])
                    .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status();
                stopped = true;
            }
            std::thread::sleep(Duration::from_secs(1));
        };
        self.set_progress(job, Value::Null);
        let rc = st.code().unwrap_or(-1);
        self.finish(rid, rc);
        if stopped {
            return Err((499, format!("enrolling {name}: cancelled")));
        }
        if rc != 0 {
            let out = std::fs::read_to_string(&log).unwrap_or_default();
            let last = out.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("failed").to_string();
            return Err((500, format!("enrolling {name}: {last}")));
        }
        Ok(t0.elapsed().as_secs_f64())
    }

    /// A running job's progress (a job not on a server: an enrolment); null: gone
    pub(crate) fn set_progress(&self, job: Option<&str>, p: Value) {
        if let Some(j) = job {
            let mut l = self.progress.lock().unwrap();
            if p.is_null() {
                l.remove(j);
            } else {
                l.insert(j.to_string(), p);
            }
        }
    }

    /// The enrolments: the models that speak enrolled voices, the enrolment jobs not ended and the last ended (the
    /// audio page's "enroll for Voxtral")
    pub(crate) fn enroll_state(&self) -> Value {
        let all = self.models();
        let models: Vec<Value> = all.iter().filter(|m| super::registry::kind_of(m) == "audio" && m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == "enrolled")))
            .map(|m| json!({"id": m["id"], "arch": m["arch"], "title": m["title"], "enabled": m["enabled"] != false})).collect();
        let l = self.job_list("audio", &json!({"limit": 50})).unwrap_or_default();
        let jobs: Vec<Value> = ["running", "queued", "recent"].iter().flat_map(|k| l[*k].as_array().cloned().unwrap_or_default())
            .filter(|j| j["method"] == "audio.voices.enroll").collect();
        json!({"models": models, "jobs": jobs, "ts": now()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fits_steps_are_read() {
        assert_eq!(step_of(r#"{"step": 350, "loss": 0.21, "relaxed": true, "s": 12.5}"#), Some(350));
        assert_eq!(step_of("enrolling anna for voxtral-tts on GPU 1 ..."), None);
        assert_eq!(step_of(r#"{"measure": 1, "frames": 180, "nll": 3.2}"#), None);
    }

    #[test]
    fn the_model_is_the_named_one_or_the_first_that_speaks_enrolled_voices() {
        let all = vec![json!({"id": "qwen3-tts-base", "kind": "audio", "speech": ["clone"]}),
                       json!({"id": "voxtral-4b-tts", "kind": "audio", "arch": "voxtral-tts", "speech": ["voices", "enrolled"]})];
        assert_eq!(enroll_model(&json!({}), &all).unwrap()["id"], "voxtral-4b-tts");
        assert_eq!(enroll_model(&json!({"model": "qwen3-tts-base"}), &all).unwrap_err().0, 400);
        assert_eq!(enroll_model(&json!({"model": "nope"}), &all).unwrap_err().0, 404);
        assert_eq!(enroll_model(&json!({}), &all[..1]).unwrap_err().0, 409);
    }
}
