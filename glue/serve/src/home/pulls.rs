//! Models downloaded on demand: a model with a card (model-cards/) that is not in the registry is "not downloaded".
//! A request naming one answers 409 `model_not_downloaded` (its size, whether it fits, the call that fetches it), or
//! with NS_AUTO_PULL on starts its download and waits for it (a job says "downloading <model>: N%" meanwhile).
//! `system.models.pull` downloads one in the background: this same program's `nextsycl models pull <id> --dir <folder>`
//! (the catalog's own fetch: files linked when the machine has them, curl resuming a partial `.part`, the SHA-256
//! checked before the model is registered), in a process group of its own so a cancel stops curl too. How far it is
//! comes from the files on the disk: each target there whole, or its `.part` so far. One download at a time per
//! models folder; the folder is the first configured one (NS_MODELS, then NS_MODEL_DIRS, then each drive's) with the
//! room for it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use super::{now, registry, Home};
use crate::drives;

type Fail = (u16, String);

/// Finished downloads kept in the list
const KEEP: usize = 20;

/// The tag a not-downloaded refusal's message starts with: `not_downloaded:<id>:<bytes>:<fits 0|1>: <words>`. The
/// error builders (api.rs rpc_error, the OpenAI routes) turn it into code `model_not_downloaded` with model, bytes,
/// fits and pull, and drop it from the message
const TAG: &str = "not_downloaded:";

/// A not-downloaded refusal as a Fail's message
pub(crate) fn tag(id: &str, bytes: u64, fits: bool, words: &str) -> String {
    format!("{TAG}{id}:{bytes}:{}: {words}", fits as u8)
}

/// A message's not-downloaded tag: the error's fields (model, bytes, fits, pull) and the message without it
pub(crate) fn untag(m: &str) -> Option<(Value, String)> {
    let rest = m.strip_prefix(TAG)?;
    let mut it = rest.splitn(4, ':');
    let (id, bytes, fits, words) = (it.next()?, it.next()?.parse::<u64>().ok()?, it.next()?, it.next()?);
    Some((json!({"model": id, "bytes": bytes, "fits": fits == "1", "pull": "system.models.pull"}), words.trim_start().to_string()))
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// A size in words: GiB, or MB under one
fn size(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.1} GiB", gib(b)) } else { format!("{:.0} MB", b as f64 / 1e6) }
}

/// One download
pub(crate) struct Pull {
    pub model: String,
    pub kind: String,
    pub title: String,
    /// the models folder it goes into, and its drive's id
    pub root: PathBuf,
    pub drive: String,
    /// (where the file lands, its bytes)
    files: Vec<(PathBuf, u64)>,
    pub bytes_total: u64,
    /// the run its output goes to (the page's runs), the log file
    pub run: u64,
    log: PathBuf,
    st: Mutex<St>,
    cancel: AtomicBool,
}

#[derive(Default)]
struct St {
    state: String,
    error: Option<String>,
    started: f64,
    ended: Option<f64>,
    pid: Option<u32>,
    /// the last progress sample (when, bytes) and the speed it gave (bytes a second)
    sample: Option<(f64, u64)>,
    speed: f64,
    /// the run closed (Home::finish) once ended
    reported: bool,
}

impl Pull {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(model: &str, kind: &str, title: &str, root: &Path, drive: &str, files: &[(String, u64)], run: u64, log: PathBuf) -> Pull {
        let files: Vec<(PathBuf, u64)> = files.iter().map(|(p, n)| (root.join(p), *n)).collect();
        Pull {
            model: model.into(), kind: kind.into(), title: title.into(), root: root.into(), drive: drive.into(),
            bytes_total: files.iter().map(|f| f.1).sum(), files, run, log,
            st: Mutex::new(St { state: "queued".into(), started: now(), ..Default::default() }), cancel: AtomicBool::new(false),
        }
    }

    pub(crate) fn state(&self) -> String {
        self.st.lock().unwrap().state.clone()
    }

    pub(crate) fn error(&self) -> Option<String> {
        self.st.lock().unwrap().error.clone()
    }

    pub(crate) fn ended(&self) -> bool {
        matches!(self.state().as_str(), "done" | "failed" | "cancelled")
    }

    /// Bytes on the disk so far: each file whole where it is, else its `.part`
    pub(crate) fn bytes_done(&self) -> u64 {
        self.files.iter().map(|(p, n)| {
            if std::fs::metadata(p).is_ok_and(|m| m.len() == *n) {
                return *n;
            }
            let part = p.with_file_name(format!("{}.part", p.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default()));
            std::fs::metadata(part).map_or(0, |m| m.len().min(*n))
        }).sum()
    }

    /// Every file whole on the disk, as a target or a finished `.part` (curl is done: its SHA-256 is being checked)
    #[cfg(test)]
    fn all_fetched(&self) -> bool {
        self.bytes_done() >= self.bytes_total
    }

    /// How it is, as the API shows it: state (queued, downloading, verifying, done, failed, cancelled), bytes, speed,
    /// ETA
    pub(crate) fn view(&self) -> Value {
        let done = self.bytes_done();
        let t = now();
        let mut st = self.st.lock().unwrap();
        if st.state == "downloading" || st.state == "verifying" {
            if let Some((t0, b0)) = st.sample {
                if t - t0 >= 1.0 {
                    let v = done.saturating_sub(b0) as f64 / (t - t0);
                    st.speed = if st.speed > 0.0 { 0.5 * st.speed + 0.5 * v } else { v };
                    st.sample = Some((t, done));
                }
            } else {
                st.sample = Some((t, done));
            }
            // curl done, the hash being checked (or the next file linked)
            st.state = if done >= self.bytes_total { "verifying".into() } else { "downloading".into() };
        }
        let left = self.bytes_total.saturating_sub(done);
        let eta = (st.speed > 0.0 && left > 0 && st.state == "downloading").then(|| (left as f64 / st.speed).round());
        json!({"model": self.model, "kind": self.kind, "title": self.title, "drive": self.drive, "folder": self.root,
               "state": st.state, "error": st.error, "bytes_total": self.bytes_total, "bytes_done": done,
               "pct": if self.bytes_total > 0 { ((done as f64 / self.bytes_total as f64) * 1000.0).round() / 10.0 } else { 0.0 },
               "speed": st.speed.round(), "eta_s": eta, "started": st.started, "ended": st.ended, "run": self.run})
    }

    /// Run `cmd` (the pull) to its end in a process group of its own, its output in the log; `others` is asked first
    /// whether another download into the same folder is under way (it waits for it)
    pub(crate) fn run(&self, mut cmd: Command, others: &dyn Fn() -> bool) {
        while others() {
            if self.cancel.load(Ordering::Relaxed) {
                return self.end("cancelled", Some("cancelled before it started".into()));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if self.cancel.load(Ordering::Relaxed) {
            return self.end("cancelled", Some("cancelled before it started".into()));
        }
        let out = match std::fs::File::create(&self.log) {
            Ok(f) => f,
            Err(e) => return self.end("failed", Some(format!("{}: {e}", self.log.display()))),
        };
        let err = out.try_clone().ok();
        use std::os::unix::process::CommandExt;
        cmd.process_group(0).stdin(std::process::Stdio::null()).stdout(out);
        if let Some(e) = err {
            cmd.stderr(e);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return self.end("failed", Some(format!("the download did not start: {e}"))),
        };
        {
            let mut st = self.st.lock().unwrap();
            st.pid = Some(child.id());
            st.state = "downloading".into();
            st.sample = Some((now(), self.bytes_done()));
        }
        let rc = child.wait().map(|s| s.success()).unwrap_or(false);
        self.st.lock().unwrap().pid = None;
        if self.cancel.load(Ordering::Relaxed) {
            return self.end("cancelled", Some("cancelled (what was fetched is kept: a pull again resumes it)".into()));
        }
        if rc {
            self.end("done", None);
        } else {
            let text = std::fs::read_to_string(&self.log).unwrap_or_default();
            let last = text.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("the download failed").trim().to_string();
            self.end("failed", Some(last));
        }
    }

    fn end(&self, state: &str, error: Option<String>) {
        let mut st = self.st.lock().unwrap();
        st.state = state.into();
        st.error = error;
        st.ended = Some(now());
    }

    /// Stop it: the process group (curl with it) ended, or a queued one never started
    pub(crate) fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(pid) = self.st.lock().unwrap().pid {
            let _ = Command::new("kill").args(["-TERM", "--", &format!("-{pid}")]).status();
        }
    }

    /// The run once (Home::finish): the exit code, the first time it is seen ended
    fn report(&self) -> Option<i32> {
        let mut st = self.st.lock().unwrap();
        if st.reported || st.ended.is_none() {
            return None;
        }
        st.reported = true;
        Some(if st.state == "done" { 0 } else { 1 })
    }

    /// What a job waiting for it says
    fn words(&self) -> String {
        let v = self.view();
        match v["state"].as_str().unwrap_or("") {
            "queued" => format!("downloading {}: queued behind another download to {}", self.model, self.drive),
            "verifying" => format!("downloading {}: checking its SHA-256", self.model),
            _ => format!("downloading {}: {}% of {}", self.model, v["pct"], size(self.bytes_total)),
        }
    }
}

/// Wait for a download to end: Ok once done, its error otherwise. `tick` is told what it is doing every `every`, and
/// answers false to stop waiting (a job cancelled: the download goes on)
pub(crate) fn wait(p: &Pull, every: Duration, tick: &mut dyn FnMut(&str) -> bool) -> Result<(), Fail> {
    loop {
        match p.state().as_str() {
            "done" => return Ok(()),
            "failed" => return Err((502, format!("{}: the download failed: {}", p.model, p.error().unwrap_or_default()))),
            "cancelled" => return Err((499, format!("{}: the download was cancelled", p.model))),
            _ => {}
        }
        if !tick(&p.words()) {
            return Err((499, format!("cancelled while {} downloads (it goes on: system.models.pulls)", p.model)));
        }
        std::thread::sleep(every);
    }
}

/// The downloads, newest last
#[derive(Default)]
pub(crate) struct Pulls {
    list: Mutex<Vec<Arc<Pull>>>,
}

impl Pulls {
    /// The one under way (queued or downloading) for a model
    fn active(&self, id: &str) -> Option<Arc<Pull>> {
        self.list.lock().unwrap().iter().rev().find(|p| p.model == id && !p.ended()).cloned()
    }
}

impl Home {
    /// Whether a request for a model that is not downloaded starts its download and waits (NS_AUTO_PULL): the
    /// environment, else the settings file as it is now (the page's checkbox takes effect without a restart)
    pub(crate) fn auto_pull(&self) -> bool {
        let on = |v: &str| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on");
        if let Ok(v) = std::env::var("NS_AUTO_PULL") {
            return on(&v);
        }
        let text = std::fs::read_to_string(self.conf_path()).unwrap_or_default();
        super::config::entries(&text).iter().rev().find(|(k, _)| k == "NS_AUTO_PULL").is_some_and(|(_, v)| on(v))
    }

    /// The models folders a download may go into, in order: those set (NS_MODELS first), then each drive's
    fn pull_roots(&self, all: &[Value]) -> Vec<(PathBuf, String)> {
        let set = self.folders();
        let list = drives::drives(all, &set);
        let ms = drives::mounts();
        drives::roots(&set, &list).into_iter().map(|r| {
            let d = list.iter().find(|d| d.holds(&r, &ms)).map_or_else(|| r.display().to_string(), |d| d.id.clone());
            (r, d)
        }).collect()
    }

    /// Where `bytes` would go: the first folder with the room (else the first, which lacks it), its drive, whether it
    /// fits, its free bytes
    fn pull_place(&self, all: &[Value], bytes: u64) -> Option<(PathBuf, String, bool, u64)> {
        let roots = self.pull_roots(all);
        // a folder not made yet: the space of where it would be
        let space = |p: &Path| p.ancestors().find(|a| a.exists()).and_then(drives::space);
        let fits = |p: &Path| space(p).is_some_and(|(size, free)| drives::room(bytes, free, size).is_ok());
        let (r, d) = roots.iter().find(|(r, _)| fits(r)).or_else(|| roots.first())?.clone();
        let free = space(&r).map_or(0, |s| s.1);
        let ok = fits(&r);
        Some((r, d, ok, free))
    }

    /// A model with a card that is not in the registry: what it is, how big, where it would go - else None (it is
    /// registered, or no card names it)
    pub(crate) fn not_downloaded(&self, id: &str) -> Option<Value> {
        if id.is_empty() {
            return None;
        }
        let all = self.models();
        if all.iter().any(|m| m["id"] == id) {
            return None;
        }
        let card = self.o.cfg.cards().into_iter().find(|c| c["id"] == id && c["kind"] != "lora")?;
        let files = self.o.cfg.card_files(id);
        let bytes: u64 = files.iter().map(|f| f.1).sum();
        let (folder, drive, fits, free) = self.pull_place(&all, bytes).unwrap_or_default();
        let pull = self.pulls.active(id).map(|p| p.view());
        // a video model loads stage by stage (its text encoder, then the denoiser...): its largest file is what a card holds
        let held = if card["kind"] == "video" { files.iter().map(|f| f.1).max().unwrap_or(bytes) } else { bytes };
        let (runs_here, why_not) = self.runs_here(&card, held);
        Some(json!({"id": id, "kind": card["kind"], "title": card["title"], "about": card["about"], "gpus": card["gpus"], "bytes": bytes,
                    "downloaded": false, "drive": drive, "folder": folder, "fits": fits, "free": free, "pull": pull,
                    "runs_here": runs_here, "why_not": why_not}))
    }

    /// Whether a card's model can run on this box's cards: what it needs on each card (its `vram_gib`, else its files:
    /// whole on one card, or a split's share on each) against the largest of its cards less a GiB - and why not.
    /// (Its download stays possible through the API: a bigger card may come; the page's button is off.)
    fn runs_here(&self, card: &Value, bytes: u64) -> (bool, Option<String>) {
        let cards = self.cards.lock().unwrap().clone();
        let total = |g: usize| cards.get(g).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(0);
        let gpus: Vec<usize> = card["gpus"].as_str().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .or_else(|| card["gpus"].as_u64().map(|g| vec![g as usize])).unwrap_or_else(|| vec![0]);
        let present: Vec<usize> = gpus.iter().copied().filter(|g| total(*g) > 0).collect();
        if present.is_empty() {
            return (true, None); // the cards not known yet: no verdict
        }
        let split = card["split"] == true && present.len() > 1;
        let gib = |mb: u64| mb as f64 / 1024.0;
        let need_mb = match card["vram_gib"].as_f64() {
            Some(g) => (g * 1024.0) as u64,
            None => (bytes >> 20) * 11 / 10 / if split { present.len() as u64 } else { 1 },
        };
        let best = present.iter().map(|g| total(*g)).max().unwrap_or(0);
        let room = if split { present.iter().map(|g| total(*g)).min().unwrap_or(0) } else { best }.saturating_sub(1024);
        if need_mb <= room {
            return (true, None);
        }
        let name = |g: usize| cards.get(g).and_then(|c| c["name"].as_str()).unwrap_or("its card").replace("NVIDIA GeForce ", "");
        let on = present.iter().map(|g| name(*g)).collect::<Vec<_>>().join(" + ");
        (false, Some(format!("needs ~{:.1} GiB {}; {on} {} {:.1} GiB", gib(need_mb), if split { "on each card" } else { "on one card" },
                             if split { "hold" } else { "holds at most" }, gib(room + 1024))))
    }

    /// Every card model not downloaded (system.models, the page's "Not downloaded")
    pub(crate) fn not_downloaded_all(&self) -> Vec<Value> {
        let all = self.models();
        self.o.cfg.cards().iter().filter(|c| c["kind"] != "lora").filter_map(|c| c["id"].as_str())
            .filter(|id| !all.iter().any(|m| m["id"] == *id)).filter_map(|id| self.not_downloaded(id)).collect()
    }

    /// system.models.pull: a model's download started (or the one under way); refused for one registered, without a
    /// card, or without the room
    pub(crate) fn pull_start(&self, id: &str) -> Result<Arc<Pull>, Fail> {
        if let Some(p) = self.pulls.active(id) {
            return Ok(p);
        }
        let all = self.models();
        if all.iter().any(|m| m["id"] == id) {
            return Err((409, format!("{id} is downloaded already (system.models)")));
        }
        let nd = self.not_downloaded(id).ok_or_else(|| (404, format!("no model card {id} (system.models lists the ones that can be downloaded)")))?;
        if nd["fits"] != true {
            return Err((503, format!("no room for {id} ({}) on any models folder: {} has {} free (NS_MODEL_DIRS adds folders)",
                                     size(nd["bytes"].as_u64().unwrap_or(0)), nd["drive"].as_str().unwrap_or("?"), size(nd["free"].as_u64().unwrap_or(0)))));
        }
        let root = PathBuf::from(nd["folder"].as_str().unwrap_or(""));
        let (run, log) = self.new_run(format!("download {id} to {}", root.display()));
        let p = Arc::new(Pull::new(id, nd["kind"].as_str().unwrap_or(""), nd["title"].as_str().unwrap_or(id), &root, nd["drive"].as_str().unwrap_or(""),
                                   &self.o.cfg.card_files(id), run, log));
        {
            let mut l = self.pulls.list.lock().unwrap();
            l.push(p.clone());
            // the ended ones beyond KEEP go, oldest first
            let ended = l.iter().filter(|x| x.ended()).count();
            let mut drop = ended.saturating_sub(KEEP);
            l.retain(|x| if drop > 0 && x.ended() { drop -= 1; false } else { true });
        }
        let mut cmd = Command::new(&self.o.exe);
        cmd.arg("models").arg("pull").arg(id).arg("--dir").arg(&root);
        let job = p.clone();
        // another download into the same folder first (one at a time a folder)
        let ahead: Vec<Arc<Pull>> = self.pulls.list.lock().unwrap().iter().filter(|x| !Arc::ptr_eq(x, &p) && x.root == root && !x.ended()).cloned().collect();
        std::thread::spawn(move || job.run(cmd, &|| ahead.iter().any(|x| !x.ended())));
        Ok(p)
    }

    /// system.models.pull.cancel
    pub(crate) fn pull_cancel(&self, id: &str) -> Result<Value, Fail> {
        let p = self.pulls.active(id).ok_or_else(|| (409, format!("no download of {id} is under way (system.models.pulls)")))?;
        p.stop();
        Ok(json!({"model": id, "cancelling": true}))
    }

    /// system.models.pulls: every download kept, newest first; each ended one's run closed
    pub(crate) fn pulls_view(&self) -> Vec<Value> {
        let l: Vec<Arc<Pull>> = self.pulls.list.lock().unwrap().clone();
        for p in &l {
            if let Some(rc) = p.report() {
                self.finish(p.run, rc);
            }
        }
        l.iter().rev().map(|p| p.view()).collect()
    }

    /// A request naming `id`: Ok when it is registered (or no card names it - the caller says "no model"). Not
    /// downloaded: the 409 refusal, or with NS_AUTO_PULL its download started and, when `block`, waited for - a job
    /// (`job`, its request id) saying "downloading <model>: N%" meanwhile and stopping the wait when cancelled
    pub(crate) fn need_downloaded(&self, id: &str, job: Option<&str>, block: bool) -> Result<(), Fail> {
        let Some(nd) = self.not_downloaded(id) else { return Ok(()) };
        let bytes = nd["bytes"].as_u64().unwrap_or(0);
        let fits = nd["fits"] == true;
        if !self.auto_pull() {
            let room = if fits { String::new() } else { format!("; {} lacks the room", nd["drive"].as_str().unwrap_or("its drive")) };
            return Err((409, tag(id, bytes, fits, &format!("{id} is not downloaded ({}{room}): system.models.pull {{\"model\": \"{id}\"}} fetches it, or turn on NS_AUTO_PULL",
                                                         size(bytes)))));
        }
        let p = self.pull_start(id)?;
        if !block {
            return Ok(());
        }
        let r = wait(&p, Duration::from_secs(2), &mut |words| {
            self.note(job, "downloading", words);
            !job.is_some_and(|j| self.job_cancelled(j))
        });
        let _ = self.pulls_view();
        r?;
        // registered now: its kind's kinds_on and pick see it
        if registry::find(&self.o.cfg, id).ok().flatten().is_none() {
            return Err((500, format!("{id}: downloaded, but the registry has no entry for it")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let d = std::env::temp_dir().join(format!("nextsycl-pulls-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Tmp(d)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sh(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.arg("-c").arg(script);
        c
    }

    fn pull(t: &Tmp, files: &[(&str, u64)]) -> Pull {
        let files: Vec<(String, u64)> = files.iter().map(|(p, n)| (p.to_string(), *n)).collect();
        Pull::new("m", "image", "M", &t.0, "ssd", &files, 1, t.0.join("pull.log"))
    }

    #[test]
    fn a_refusal_carries_the_model_and_its_size() {
        let m = tag("qwen-image-2.1-q8", 21_000_000_000, true, "qwen-image-2.1-q8 is not downloaded (19.6 GiB): ...");
        let (f, words) = untag(&m).unwrap();
        assert_eq!(f, json!({"model": "qwen-image-2.1-q8", "bytes": 21_000_000_000u64, "fits": true, "pull": "system.models.pull"}));
        assert!(words.starts_with("qwen-image-2.1-q8 is not downloaded"));
        assert!(untag("no model x (system.models)").is_none());
        // the /rpc error object: its own code, not retryable, the fields beside the message
        let e = super::super::api::rpc_error("image.generate", 409, None, &m);
        assert_eq!(e["code"], "model_not_downloaded");
        assert_eq!(e["retryable"], false);
        assert_eq!(e["status"], 409);
        assert_eq!(e["model"], "qwen-image-2.1-q8");
        assert_eq!(e["bytes"], 21_000_000_000u64);
        assert_eq!(e["pull"], "system.models.pull");
        assert!(!e["message"].as_str().unwrap().starts_with(TAG));
    }

    #[test]
    fn progress_counts_whole_files_and_parts() {
        let t = Tmp::new("progress");
        let p = pull(&t, &[("a/x.gguf", 100), ("b/y.bin", 50)]);
        assert_eq!(p.bytes_done(), 0);
        std::fs::create_dir_all(t.0.join("a")).unwrap();
        std::fs::write(t.0.join("a/x.gguf.part"), vec![0u8; 40]).unwrap();
        assert_eq!(p.bytes_done(), 40);
        std::fs::rename(t.0.join("a/x.gguf.part"), t.0.join("a/x.gguf")).unwrap();
        std::fs::write(t.0.join("a/x.gguf"), vec![0u8; 100]).unwrap();
        assert_eq!(p.bytes_done(), 100);
        assert!(!p.all_fetched());
        std::fs::create_dir_all(t.0.join("b")).unwrap();
        std::fs::write(t.0.join("b/y.bin"), vec![0u8; 50]).unwrap();
        assert!(p.all_fetched());
        assert_eq!(p.view()["pct"], 100.0);
    }

    #[test]
    fn a_pull_runs_to_done_and_its_run_is_reported_once() {
        let t = Tmp::new("done");
        let p = pull(&t, &[("x.gguf", 10)]);
        assert_eq!(p.state(), "queued");
        p.run(sh(&format!("printf 0123456789 > {}/x.gguf.part && mv {0}/x.gguf.part {0}/x.gguf", t.0.display())), &|| false);
        assert_eq!(p.state(), "done");
        assert_eq!(p.view()["bytes_done"], 10);
        assert_eq!(p.report(), Some(0));
        assert_eq!(p.report(), None);
    }

    #[test]
    fn a_failed_pull_says_its_last_line() {
        let t = Tmp::new("fail");
        let p = pull(&t, &[("x.gguf", 10)]);
        p.run(sh("echo 'checking its SHA-256 ...'; echo 'x.gguf: SHA-256 abc, the catalog says def' >&2; exit 1"), &|| false);
        assert_eq!(p.state(), "failed");
        assert!(p.error().unwrap().contains("SHA-256 abc"));
        assert_eq!(p.report(), Some(1));
    }

    #[test]
    fn a_cancel_stops_the_process_group() {
        let t = Tmp::new("cancel");
        let p = Arc::new(pull(&t, &[("x.gguf", 10)]));
        let q = p.clone();
        // a child of the shell (as curl is of the pull) ends with the group
        let h = std::thread::spawn(move || q.run(sh("sleep 30 & wait"), &|| false));
        for _ in 0..100 {
            if p.state() == "downloading" {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let t0 = std::time::Instant::now();
        p.stop();
        h.join().unwrap();
        assert_eq!(p.state(), "cancelled");
        assert!(t0.elapsed() < Duration::from_secs(5), "the group was not stopped");
    }

    #[test]
    fn a_queued_pull_waits_for_the_one_ahead_and_can_be_cancelled_there() {
        let t = Tmp::new("queued");
        let p = Arc::new(pull(&t, &[("x.gguf", 10)]));
        let q = p.clone();
        let h = std::thread::spawn(move || q.run(sh("exit 0"), &|| true));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(p.state(), "queued");
        assert!(p.words().contains("queued behind another download"));
        p.stop();
        h.join().unwrap();
        assert_eq!(p.state(), "cancelled");
    }

    #[test]
    fn a_waiting_request_runs_once_the_download_is_done() {
        let t = Tmp::new("wait");
        let p = Arc::new(pull(&t, &[("x.gguf", 10)]));
        let q = p.clone();
        let dir = t.0.display().to_string();
        std::thread::spawn(move || q.run(sh(&format!("sleep 0.3; printf 0123456789 > {dir}/x.gguf")), &|| false));
        let mut said = Vec::new();
        let r = wait(&p, Duration::from_millis(50), &mut |w| {
            said.push(w.to_string());
            true
        });
        assert_eq!(r, Ok(()));
        assert!(said.iter().any(|w| w.starts_with("downloading m:")), "{said:?}");
    }

    #[test]
    fn a_waiting_request_fails_with_the_downloads_error_or_stops_when_its_job_is_cancelled() {
        let t = Tmp::new("waitfail");
        let p = Arc::new(pull(&t, &[("x.gguf", 10)]));
        let q = p.clone();
        std::thread::spawn(move || q.run(sh("sleep 0.2; echo 'the download failed (run pull again: it resumes)'; exit 1"), &|| false));
        let r = wait(&p, Duration::from_millis(50), &mut |_| true);
        let (st, m) = r.unwrap_err();
        assert_eq!(st, 502);
        assert!(m.contains("the download failed (run pull again"), "{m}");
        // a job cancelled: the wait ends, the download is left to go on
        let p2 = Arc::new(pull(&t, &[("y.gguf", 10)]));
        let q2 = p2.clone();
        std::thread::spawn(move || q2.run(sh("sleep 2"), &|| false));
        let (st, _) = wait(&p2, Duration::from_millis(20), &mut |_| false).unwrap_err();
        assert_eq!(st, 499);
        p2.stop();
    }
}
