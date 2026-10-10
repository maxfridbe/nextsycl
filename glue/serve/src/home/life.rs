//! What the enabled models do: their state, the memory they are vouched for, enabling and disabling, loading on
//! demand and unloading.
//!
//! - chat (llm): the video studio's llm.mode loads one at a time (it knows when a render holds the card); the switcher
//!   loads on Open WebUI's request and unloads after the idle time; enabled = listed by the switcher.
//! - image, audio: one server each (`nextsycl image|audio start ID --gpu N --idle-exit S`), started on the first
//!   request for an enabled model, swapped when a request names another (after the one running), ending itself when
//!   idle. A start waits for the card to have the model's memory free (1.5 GiB to spare: past that the xe driver
//!   spills into host memory and the box can hang), and answers 409 when it does not come free.
//! - video: the daemon (`nextsycl video start`) with the enabled video models as its engines, on their GPUs; its
//!   workers load per clip and let the card go when idle. No video model enabled: the daemon is stopped.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{now, Home};
use crate::http::{call_for, Target};
use super::registry;

/// Memory to keep free on a card beyond a model's (the xe driver has no out-of-memory: past the card it spills)
const MARGIN_MB: u64 = 1536;

/// The video daemon's engines (model ids), and (engine, GPU) per loaded worker
type VideoDaemon = (Vec<String>, Vec<(String, usize)>);

pub(crate) fn s(m: &Value, k: &str) -> String {
    m.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

impl Home {
    pub(crate) fn ncards(&self) -> usize {
        self.cards.lock().unwrap().as_array().map_or(0, Vec::len).max(1)
    }

    /// An entry's cards: its `gpus` ("all", "1", "0,1")
    pub(crate) fn gpus_of(&self, m: &Value) -> Vec<usize> {
        let g = s(m, "gpus");
        if g.is_empty() || g == "all" {
            if registry::kind_of(m) == "llm" && g.is_empty() { return vec![0] }
            return (0..self.ncards()).collect();
        }
        let v: Vec<usize> = g.split([',', ' ']).filter_map(|x| x.trim().parse().ok()).collect();
        if v.is_empty() { vec![0] } else { v }
    }

    fn port_of(&self, kind: &str) -> u16 {
        match kind {
            "image" => self.o.ports.image,
            "audio" => self.o.ports.audio,
            "video" => self.o.ports.video,
            _ => self.o.ports.switch,
        }
    }

    pub(crate) fn target(&self, kind: &str) -> Target {
        Target::Tcp(format!("127.0.0.1:{}", self.port_of(kind)))
    }

    /// The memory a model takes on each of its cards, MiB: as seen while it ran, else from its files' sizes
    fn vouched_mb(&self, m: &Value, card_mb: u64) -> (BTreeMap<usize, u64>, bool) {
        let id = s(m, "id");
        let gpus = self.gpus_of(m);
        if let Some(seen) = self.seen.lock().unwrap().get(&id).filter(|g| gpus.iter().all(|i| g.contains_key(i))) {
            return (gpus.iter().map(|g| (*g, seen[g])).collect(), true);
        }
        let files: u64 = registry::paths(m).iter().filter(|p| p.is_file()).filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len() >> 20).sum();
        let per = match registry::kind_of(m) {
            "video" => card_mb.saturating_sub(4096),
            "image" => files * 105 / 100 / gpus.len() as u64 + 4096,
            "audio" => files * 110 / 100 / gpus.len() as u64 + 2048,
            _ => files / gpus.len() as u64 + 2048,
        };
        (gpus.iter().map(|g| (*g, per.min(card_mb.saturating_sub(1024)))).collect(), false)
    }

    /// What the studio says of the chat card: (selected mode, the mode starting)
    fn chat_mode(&self) -> (Option<String>, Option<String>) {
        let v = call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({})), 4).ok().map(|v| v.get("result").cloned().unwrap_or(v));
        let g = |k: &str| v.as_ref().and_then(|v| v[k].as_str()).map(str::to_string);
        (g("mode"), g("starting"))
    }

    /// The registered chat model the chat server answers as now
    pub(crate) fn chat_loaded(&self, all: &[Value]) -> Option<String> {
        let v = call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.llm)), "GET", "/v1/models", None, 3).ok()?;
        let m = v["data"].get(0)?;
        if m["status"]["value"].as_str().unwrap_or("loaded") != "loaded" {
            return None;
        }
        let cur = m["id"].as_str()?;
        all.iter().filter(|x| registry::kind_of(x) == "llm" && cur.ends_with(&s(x, "id"))).max_by_key(|x| s(x, "id").len()).map(|x| s(x, "id"))
    }

    /// The image or audio server's model and whether it is busy, when one answers
    fn server(&self, kind: &str) -> Option<(String, bool, bool)> {
        let t = self.target(kind);
        let info = call_for(&t, "GET", "/api/info", None, 3).ok()?;
        let p = call_for(&t, "GET", "/api/progress", None, 3).unwrap_or(json!({}));
        Some((s(&info, "model"), info["loading"] == true, p["busy"] == true || p["waiting"].as_u64().unwrap_or(0) > 0))
    }

    /// The video daemon's engines (their model ids) and the ones loaded on a GPU, when it runs
    fn video_daemon(&self) -> Option<VideoDaemon> {
        let v = call_for(&self.target("video"), "GET", "/engine/status", None, 4).ok()?;
        let names = v["engines"].as_array()?.iter().filter_map(|e| e["name"].as_str().map(str::to_string)).collect();
        // a slot's engine is "unloaded" or holds a denoiser (by its model id)
        let loaded = v["gpus"].as_array().into_iter().flatten()
            .filter(|g| g["engine"].as_str().is_some_and(|e| e != "unloaded"))
            .filter_map(|g| Some((g["denoiser"].as_str()?.to_string(), g["gpu"].as_u64()? as usize)))
            .collect();
        Some((names, loaded))
    }

    /// Every registered model (LoRAs aside): kind, enabled, cards, state, memory vouched for
    pub(crate) fn model_states(&self, all: &[Value], cards: &Value) -> Vec<Value> {
        let card_mb = |g: usize| cards.get(g).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
        let loading = self.loading.lock().unwrap().clone();
        let need = |k: &str| all.iter().any(|m| registry::kind_of(m) == k);
        let chat = if need("llm") { (self.chat_loaded(all), self.chat_mode()) } else { (None, (None, None)) };
        let image = if need("image") { self.server("image") } else { None };
        let audio = if need("audio") { self.server("audio") } else { None };
        let video = if need("video") { self.video_daemon() } else { None };
        all.iter().filter(|m| registry::kind_of(m) != "lora").map(|m| {
            let (id, kind) = (s(m, "id"), registry::kind_of(m).to_string());
            let enabled = m["enabled"] != false;
            let gpus = self.gpus_of(m);
            let (vouch, measured) = self.vouched_mb(m, gpus.first().map_or(32768, |g| card_mb(*g)));
            let mut state = match kind.as_str() {
                "llm" if chat.0.as_deref() == Some(id.as_str()) => "loaded",
                "llm" if chat.1 .1.as_deref() == Some(id.as_str()) => "loading",
                "image" | "audio" => match if kind == "image" { &image } else { &audio } {
                    Some((mid, l, _)) if *mid == id && *l => "loading",
                    Some((mid, _, true)) if *mid == id => "busy",
                    Some((mid, _, _)) if *mid == id => "loaded",
                    _ => "unloaded",
                },
                "video" => match &video {
                    Some((_, loaded)) if loaded.iter().any(|(e, _)| *e == id) => "loaded",
                    Some((names, _)) if names.contains(&id) => "ready",
                    _ => "unloaded",
                },
                _ => "unloaded",
            };
            if loading.contains_key(&id) && state == "unloaded" {
                state = "loading";
            }
            if !enabled && state == "unloaded" {
                state = "disabled";
            }
            json!({"id": id, "kind": kind, "title": m["title"], "arch": m["arch"], "enabled": enabled, "gpus": gpus, "state": state,
                   "vouch_mb": vouch.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "measured": measured,
                   "page": match kind.as_str() { "image" => json!(self.o.ports.image), "audio" => json!(self.o.ports.audio),
                                                  "video" => json!(self.o.ports.video), _ => json!(self.o.ports.chat_ui) }})
        }).collect()
    }

    /// Each card's vouched memory: the enabled models on it (chat models count once - the largest, or the loaded one:
    /// one runs at a time), and whether that passes the card
    pub(crate) fn vouch(&self, cards: &mut Value, models: &[Value]) {
        for c in cards.as_array_mut().into_iter().flatten() {
            let gi = c["index"].as_u64().unwrap_or(0).to_string();
            let total = c["vram_total_mb"].as_u64().unwrap_or(0);
            let on: Vec<&Value> = models.iter().filter(|m| m["enabled"] == true && m["vouch_mb"].get(&gi).is_some()).collect();
            let mb = |m: &Value| m["vouch_mb"][&gi].as_u64().unwrap_or(0);
            // chat models and video engines count once a card: one of each runs there at a time (the loaded one, else
            // the largest)
            let mut list: Vec<Value> = on.iter().filter(|m| !matches!(m["kind"].as_str(), Some("llm" | "video")))
                .map(|m| json!({"model": m["id"], "kind": m["kind"], "mb": mb(m), "measured": m["measured"], "state": m["state"]})).collect();
            for kind in ["llm", "video"] {
                let group: Vec<&&Value> = on.iter().filter(|m| m["kind"] == kind).collect();
                if let Some(c) = group.iter().find(|m| m["state"] == "loaded").or_else(|| group.iter().max_by_key(|m| mb(m))) {
                    list.push(json!({"model": c["id"], "kind": kind, "mb": mb(c), "measured": c["measured"], "state": c["state"], "of": group.len()}));
                }
            }
            let sum: u64 = list.iter().map(|v| v["mb"].as_u64().unwrap_or(0)).sum();
            c["vouched"] = json!(list);
            c["vouched_mb"] = json!(sum);
            c["overbooked"] = json!(sum > total);
        }
    }

    fn entry(&self, id: &str) -> Result<Value, (u16, String)> {
        let m = registry::find(&self.o.cfg, id).map_err(|e| (500, e))?.ok_or_else(|| (404, format!("no model {id} (nextsycl models list)")))?;
        if registry::kind_of(&m) == "lora" {
            return Err((400, format!("{id} is a LoRA: it is enabled with the model that uses it")));
        }
        Ok(m)
    }

    /// Enable `id` on cards `gpus` (a list; none: as it was)
    pub(crate) fn enable(self: &Arc<Self>, id: &str, gpus: Option<&Value>) -> Result<String, (u16, String)> {
        let m = self.entry(id)?;
        let n = self.ncards();
        let gpus: Option<Vec<usize>> = gpus.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|g| g as usize).collect());
        if let Some(g) = &gpus {
            if g.is_empty() || g.iter().any(|x| *x >= n) {
                return Err((400, format!("gpus {g:?}: one or more of 0..{}", n - 1)));
            }
            if matches!(registry::kind_of(&m), "image" | "audio") && g.len() > 1 {
                return Err((400, format!("{id} runs on one card")));
            }
        }
        registry::update(&self.o.cfg, id, |e| {
            e["enabled"] = json!(true);
            if let Some(g) = &gpus {
                e["gpus"] = json!(g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
        }).map_err(|e| (500, e))?;
        if registry::kind_of(&m) == "video" {
            let me = self.clone();
            std::thread::spawn(move || {
                if let Err(e) = me.video_apply(false) {
                    let (rid, log) = me.new_run("video: apply the enabled models".into());
                    let _ = std::fs::write(log, e);
                    me.finish(rid, 1);
                }
            });
        }
        Ok(s(&m, "title"))
    }

    /// Disable `id`: unloaded first (after the requests it is answering); a video model while a clip renders only
    /// with `force` (the clip stops at its next step)
    pub(crate) fn disable(self: &Arc<Self>, id: &str, force: bool) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if registry::kind_of(&m) == "video" && !force && self.rendering() {
            return Err((409, "a clip is rendering: pause the studio's queue and wait, or force (the clip stops at its next step)".into()));
        }
        self.unload(id)?;
        registry::update(&self.o.cfg, id, |e| e["enabled"] = json!(false)).map_err(|e| (500, e))?;
        if registry::kind_of(&m) == "video" {
            self.video_apply(force).map_err(|e| (500, e))?;
        }
        Ok(())
    }

    fn rendering(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok()
            .map(|v| v.get("result").cloned().unwrap_or(v)).is_some_and(|r| r["idle"] == false)
    }

    /// Load `id` now, in the background (the page's start button: it spins while the state says loading)
    pub(crate) fn load_async(self: &Arc<Self>, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        if self.loading.lock().unwrap().insert(id.to_string(), now()).is_some() {
            return Ok(());
        }
        let (me, id) = (self.clone(), id.to_string());
        std::thread::spawn(move || {
            if let Err((_, e)) = me.load(&id) {
                let (rid, log) = me.new_run(format!("load {id}"));
                let _ = std::fs::write(log, &e);
                me.finish(rid, 1);
            }
            me.loading.lock().unwrap().remove(&id);
        });
        Ok(())
    }

    /// Load `id` and return when it answers
    pub(crate) fn load(&self, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        match registry::kind_of(&m) {
            "image" | "audio" => self.ensure(id).map(|_| ()),
            "video" => self.video_apply(false).map_err(|e| (500, e)),
            _ => {
                let all = self.models();
                if self.chat_loaded(&all).as_deref() == Some(id) {
                    return Ok(());
                }
                call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": id})), 600).map_err(|e| (503, format!("the studio's llm.mode: {e}")))?;
                let t0 = Instant::now();
                while t0.elapsed() < Duration::from_secs(900) {
                    if self.chat_loaded(&all).as_deref() == Some(id) {
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_secs(3));
                }
                Err((504, format!("{id} did not come up in 15 minutes (a render may hold the card)")))
            }
        }
    }

    /// Unload `id` if it is loaded (after the requests it is answering)
    pub(crate) fn unload(&self, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        match kind {
            "image" | "audio" => {
                let _g = self.kind_lock[kind].lock().unwrap();
                if self.server(kind).is_some_and(|(mid, _, _)| mid == id) {
                    self.exec(format!("{kind} stop ({id})"), &[kind.into(), "stop".into()]).map_err(|e| (500, e))?;
                }
                Ok(())
            }
            "video" => self.exec("video: unload the engines".into(), &["video".into(), "unload".into()]).map(|_| ()).map_err(|e| (500, e)),
            _ => {
                if self.chat_loaded(&self.models()).as_deref() == Some(id) {
                    call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": "none"})), 120)
                        .map_err(|e| (503, format!("the studio's llm.mode: {e}")))?;
                }
                Ok(())
            }
        }
    }

    /// The image or audio server holding `id`, started (or swapped to it) when it does not: where to send its requests
    pub(crate) fn ensure(&self, id: &str) -> Result<Target, (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        if !matches!(kind, "image" | "audio") {
            return Err((400, format!("{id} is a {kind} model")));
        }
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled (enable it on :8000 or nextsycl models enable {id})")));
        }
        let t = self.target(kind);
        let _g = self.kind_lock[kind].lock().unwrap();
        match self.server(kind) {
            Some((mid, false, _)) if mid == id => return Ok(t),
            Some((mid, _, _)) if mid == id => {
                self.wait_up(kind, id)?;
                return Ok(t);
            }
            Some((other, _, _)) => {
                // another model of the kind holds the server: its requests finish first
                let t0 = Instant::now();
                while self.server(kind).is_some_and(|(_, _, busy)| busy) {
                    if t0.elapsed() > Duration::from_secs(1800) {
                        return Err((409, format!("{other} has been busy for 30 minutes; {id} waits for it")));
                    }
                    std::thread::sleep(Duration::from_secs(2));
                }
                self.exec(format!("{kind} stop ({other}, for {id})"), &[kind.into(), "stop".into()]).map_err(|e| (500, e))?;
            }
            None => {}
        }
        // the card: the model's memory free, with room to spare
        let gpu = self.gpus_of(&m)[0];
        let card_mb = |c: &Value| c["vram_total_mb"].as_u64().unwrap_or(0);
        let (need, _) = self.vouched_mb(&m, self.cards.lock().unwrap().get(gpu).map_or(32768, card_mb));
        let need = need.get(&gpu).copied().unwrap_or(0);
        let t0 = Instant::now();
        loop {
            let cards = self.named_cards(&self.models());
            let c = cards.get(gpu).cloned().unwrap_or(Value::Null);
            let free = card_mb(&c).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0));
            if free >= need + MARGIN_MB {
                break;
            }
            if t0.elapsed() > Duration::from_secs(90) {
                let holders: Vec<String> = c["procs"].as_array().into_iter().flatten()
                    .map(|p| format!("{} {:.1} GiB", p["model"].as_str().or(p["program"].as_str()).unwrap_or("?"), p["vram_mb"].as_u64().unwrap_or(0) as f64 / 1024.0))
                    .collect();
                return Err((409, format!("GPU {gpu} is busy: {:.1} GiB free ({}), {id} needs ~{:.1} GiB; it loads once that idles out or is stopped",
                                         free as f64 / 1024.0, holders.join(", "), (need + MARGIN_MB) as f64 / 1024.0)));
            }
            std::thread::sleep(Duration::from_secs(3));
        }
        let secs = (self.o.idle_minutes * 60).to_string();
        let mut args = vec![kind.to_string(), "start".into(), id.into(), "--gpu".into(), gpu.to_string(), "--port".into(), self.port_of(kind).to_string(),
                            "--host".into(), self.o.serve_host.clone(), "--wfe".into()];
        if self.o.idle_minutes > 0 {
            args.extend(["--idle-exit".into(), secs]);
        }
        self.loading.lock().unwrap().insert(id.to_string(), now());
        let r = self.exec(format!("{kind} start {id} on GPU {gpu}"), &args);
        self.loading.lock().unwrap().remove(id);
        r.map_err(|e| (503, e))?;
        self.wait_up(kind, id)?;
        Ok(t)
    }

    fn wait_up(&self, kind: &str, id: &str) -> Result<(), (u16, String)> {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(600) {
            if self.server(kind).is_some_and(|(m, l, _)| m == id && !l) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        Err((504, format!("{id} did not answer in 10 minutes")))
    }

    /// The video daemon as the enabled video models say: their engines on their cards, or stopped when none is
    /// enabled; restarted only when that changes (and not while a clip renders, unless `force`)
    pub(crate) fn video_apply(&self, force: bool) -> Result<(), String> {
        let all = self.models();
        let enabled: Vec<&Value> = all.iter().filter(|m| registry::kind_of(m) == "video" && m["enabled"] != false).collect();
        if all.iter().all(|m| registry::kind_of(m) != "video") {
            return Ok(());
        }
        let _g = self.kind_lock["video"].lock().unwrap();
        let running = self.video_daemon();
        let marker = std::path::PathBuf::from(self.o.cfg.get("NS_SOCKET_DIR").unwrap_or_else(|| "/tmp".into())).join("video.disabled");
        if enabled.is_empty() {
            let _ = std::fs::write(&marker, "no video model is enabled (nextsycl serve)\n");
            if running.is_some() {
                if !force && self.rendering() {
                    return Err("a clip is rendering; the daemon stops when the queue is idle (or force)".into());
                }
                self.exec("video stop (no video model enabled)".into(), &["video".into(), "stop".into()])?;
            }
            return Ok(());
        }
        let _ = std::fs::remove_file(&marker);
        let first = enabled.iter().find(|m| s(m, "id") == "minimax-h3").or(enabled.first()).map(|m| s(m, "id")).unwrap_or_default();
        let mut want: Vec<String> = vec![first.clone()];
        want.extend(enabled.iter().map(|m| s(m, "id")).filter(|i| *i != first));
        let mut gpus: Vec<usize> = enabled.iter().flat_map(|m| self.gpus_of(m)).collect();
        gpus.sort();
        gpus.dedup();
        if let Some((names, _)) = &running {
            let (mut a, mut b) = (names.clone(), want.clone());
            a.sort();
            b.sort();
            if a == b {
                return Ok(());
            }
            if !force && self.rendering() {
                return Err("a clip is rendering; the engines change when the queue is idle (or force)".into());
            }
            self.exec("video stop (the enabled models changed)".into(), &["video".into(), "stop".into()])?;
        }
        let mut args = vec!["video".to_string(), "start".into(), "--model".into(), first];
        for o in &want[1..] {
            args.extend(["--engine".into(), o.clone()]);
        }
        for g in gpus {
            args.extend(["--gpu".into(), g.to_string()]);
        }
        self.exec(format!("video start ({})", want.join(", ")), &args).map(|_| ())
    }
}
