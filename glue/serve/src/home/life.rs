//! What the enabled models do: their state, the memory they are vouched for, enabling and disabling, loading on
//! demand and unloading. The decisions - which card, which idle holder goes, who may share a card, when to wait - are
//! crates/scheduler's (backplane-scheduler; nextcuda's, copied), run over `Sched` at the end of this file, the box's side
//! of its Host traits. Only an idle holder neither pinned nor held goes for room (backplane.rs has pins, holds, ETAs and
//! limits); nothing running is ever stopped for it.
//!
//! - chat (llm): the video studio's llm.mode loads one at a time (it knows when a render holds the card); the switcher
//!   loads on Open WebUI's request and unloads after the idle time; enabled = listed by the switcher. A box with
//!   neither (the front door alone): this page starts the chat server itself (`nextsycl llm start ID`) for the model
//!   a request names, stopping the one before it.
//! - image, audio, model3d: a server each (`nextsycl image|audio|model3d start ID --gpu N --idle-exit S`), started on the first
//!   request for an enabled model on the first of its cards with room, swapped when a request names another (after the
//!   one running), ending itself when idle. A second (`--slot 1`, NS_IMAGE_PORT2 / NS_AUDIO_PORT2) holds another model,
//!   or a second copy of a model enabled on two cards: started when a request finds its copy with work and its other
//!   card with room; requests go to the copy without work, else the one with fewer. A start waits for the card to have the model's memory free (1.5 GiB to spare: past that the xe driver
//!   spills into host memory and the box can hang), and answers 503 no_room when it does not come free. Free is what
//!   every process leaves (the root sampler's gpustat file counts other users'), less what loads started here have
//!   yet to take.
//!   A model whose entry or card says `ram_gib` passes the same rule for host memory: MemAvailable less the reclaimable cache the loaded models
//!   declare (`ram_cache_gib`) must leave its `ram_gib` and RAM_MARGIN_MB; idle holders on any card go for it, else it waits or answers no_room. A model without it: VRAM only.
//! - GPU work this page does not manage (a test, a benchmark) holds the run lock (crate::gpulock, NS_GPU_LOCK): while
//!   it is held no model loads - requests are refused with `cards_taken` (503, retryable), jobs wait, the video queue
//!   pauses (mod.rs video_hold); what is loaded keeps answering.
//! - video: the daemon (`nextsycl video start`) with the enabled video models as its engines, on their GPUs; its
//!   workers load per clip and let the card go when idle. No video model enabled: the daemon is stopped. A worker waiting for its card's memory (any card, shared or
//!   not) has the same room rule applied for it (video_room_loop): idle holders there go, a chat model kept or answering stays, and the worker gives up after VIDEO_ROOM_WAIT_S.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use backplane_scheduler::host::{Guard, Host};
use backplane_scheduler::place::{self, PlaceReq};
use backplane_scheduler::profile::Profile;
use backplane_scheduler::room::{self, Card, Holder, Idleness, ModelInfo, Proc, Snapshot};
pub(super) use backplane_scheduler::room::{JOB_ROOM_WAIT_S, MARGIN_MB, ROOM_WAIT_S};
use serde_json::{json, Value};

use super::{now, Home};
use crate::http::{call_for, Target};
use super::registry;

/// Host memory to leave beyond a model's `ram_gib`
pub(super) const RAM_MARGIN_MB: u64 = 4096;

/// Host memory free for a load, MiB: MemAvailable less the cache the loaded models keep warm (reclaimable, so MemAvailable counts it, but taking it costs them their speed)
pub(super) fn host_free_mb(available_mb: u64, warm_mb: u64) -> u64 {
    available_mb.saturating_sub(warm_mb)
}

/// /proc/meminfo's MemAvailable, MiB
fn mem_available_mb() -> Option<u64> {
    let m = std::fs::read_to_string("/proc/meminfo").ok()?;
    m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb / 1024)
}

/// Whether the chat model loaded may go for room: answering nothing, not changing, pinned and held by nobody (running work never goes; a hold keeps it until its time is out)
/// The chat model as the room rule sees it
#[derive(Debug, PartialEq)]
pub(super) enum ChatHold {
    NoModel,
    Idle(String),
    /// answering, changing or kept, or something that would say did not answer (`None`: not up yet)
    Held(Option<String>),
}

/// `None` for answering or kept: who would say did not answer
pub(super) fn chat_hold_of(id: String, answering: Option<bool>, changing: bool, kept_here: bool, kept_by_switcher: Option<bool>) -> ChatHold {
    match (answering, kept_by_switcher) {
        (Some(a), Some(k)) if chat_may_go(a, changing, kept_here, k) => ChatHold::Idle(id),
        _ => ChatHold::Held(Some(id)),
    }
}

pub(super) fn chat_may_go(answering: bool, changing: bool, kept_here: bool, kept_by_switcher: bool) -> bool {
    !answering && !changing && !kept_here && !kept_by_switcher
}

/// The daemon's workers waiting for a card's memory, from its /engine/status: (GPU, MiB they need) - shared or not, the state reads the same
/// ("waiting for the GPU: 2.4 GiB free, 27.5 GiB needed")
pub(super) fn waiting_workers(status: &Value) -> Vec<(usize, u64)> {
    status["gpus"].as_array().into_iter().flatten().filter_map(|g| {
        let st = g["engine"].as_str()?.strip_prefix("waiting for the GPU: ")?;
        let need: f64 = st.split(", ").nth(1)?.split_whitespace().next()?.parse().ok()?;
        Some((g["gpu"].as_u64()? as usize, (need * 1024.0) as u64))
    }).collect()
}


/// The video daemon's engines (model ids), and (engine, GPU) per loaded worker
type VideoDaemon = (Vec<String>, Vec<(String, usize)>);

/// Whether something listens on 127.0.0.1:`port`
fn listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok()
}

/// A card's free memory, MiB: total less used, less what each load under way (its server's port, the MiB it was
/// placed for) has not taken yet - two loads planned at once on one card must not both count the same free memory
fn free_on(c: &Value, pending: &[(u16, u64)]) -> u64 {
    let free = c["vram_total_mb"].as_u64().unwrap_or(0).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0));
    let to_come: u64 = pending.iter().map(|(port, mb)| {
        let has = c["procs"].as_array().into_iter().flatten().filter(|p| p["port"].as_str() == Some(port.to_string().as_str()))
            .map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
        mb.saturating_sub(has)
    }).sum();
    free.saturating_sub(to_come)
}

/// An image, audio or 3D model: it loads whole on one card (the first of its cards with room)
fn one_card(m: &Value) -> bool {
    matches!(registry::kind_of(m), "image" | "audio" | "model3d")
}

/// A chat model whose model card spreads one copy over every card (`gpus: all`; its title says "both GPUs"): it
/// does not fit on fewer, so its cards are not a choice (an image or audio model's `all` is where its copies may go)
pub(crate) fn spans_all(m: &Value, model_cards: &[Value]) -> bool {
    registry::kind_of(m) == "llm"
        && model_cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|id| !id.is_empty() && c["id"] == id.as_str()))
            .is_some_and(|c| s(c, "gpus") == "all")
}

/// The cards processes of model `id` hold memory on, in the named cards
fn held_on(cards: &Value, id: &str) -> Vec<usize> {
    cards.as_array().into_iter().flatten().filter(|c| c["procs"].as_array().into_iter().flatten().any(|p| p["model_id"] == id))
        .filter_map(|c| c["index"].as_u64()).map(|g| g as usize).collect()
}

/// The card the server listening on `port` is on (its process's --port), in the named cards
fn card_of_port(cards: &Value, port: u16) -> Option<usize> {
    let port = port.to_string();
    cards.as_array()?.iter().find(|c| c["procs"].as_array().into_iter().flatten().any(|p| p["port"].as_str() == Some(port.as_str())))
        .and_then(|c| c["index"].as_u64()).map(|g| g as usize)
}

/// One of the kind's servers (the scheduler's type): its slot, port, model, and whether it loads or has work
pub(crate) type Slot = room::Server;

/// A request's server, chosen by `ensure`: counted as having work until dropped (after the answer)
pub(crate) struct Lease<'a> {
    target: Target,
    pub(crate) port: u16,
    home: &'a Home,
}

impl std::ops::Deref for Lease<'_> {
    type Target = Target;
    fn deref(&self) -> &Target {
        &self.target
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(n) = self.home.inflight.lock().unwrap().get_mut(&self.port) {
            *n = n.saturating_sub(1);
        }
    }
}

/// What a model can do, in neutral words (a client picks a model by them, on any box): its tags, and the /rpc
/// methods that take it
pub(crate) fn capabilities(m: &Value) -> (Vec<&'static str>, Vec<&'static str>) {
    let arch = s(m, "arch");
    let speech = |c: &str| m["speech"].as_array().is_some_and(|a| a.iter().any(|x| x == c));
    match registry::kind_of(m) {
        "llm" => {
            let mut c = vec!["chat", "stream"];
            if m["tools"] == true {
                c.push("tools");
            }
            (c, vec!["llm.chat"])
        }
        "image" if arch.contains("ocr") => (vec!["ocr", "read_pages", "pdf"], vec!["image.ocr"]),
        "image" if arch == "realesrgan" || arch == "waifu2x" => (vec!["upscale", "upscale_video", "tiles"], vec!["image.upscale", "video.upscale"]),
        "image" if arch == "starvector" => (vec!["vectorize", "svg"], vec!["image.vectorize"]),
        "image" => {
            let mut c = vec!["generate", "edit", "compose", "img2img", "loras"];
            if m["env"]["NS_QI_SIGMAS"].is_string() {
                c.push("few_step");
            }
            (c, vec!["image.generate", "image.edit"])
        }
        "audio" if arch == "whisper" => (vec!["transcribe", "speech_to_text", "word_timestamps", "language_detection", "translate_to_english"], vec!["audio.transcribe"]),
        "audio" if m["speech"].is_array() => {
            let mut c = vec!["speech"];
            if speech("voices") {
                c.push("builtin_voices");
            }
            if speech("clone") {
                c.extend(["clone", "saved_voices", "continue_from"]);
            }
            // saved voices fitted for it (a model without an encoder: experimental)
            if speech("enrolled") {
                c.push("enrolled_voices");
            }
            if speech("design") {
                c.push("voice_design");
            }
            // a cloned voice steered by an instruction (a recording and instructions together)
            if speech("direction") {
                c.push("voice_direction");
            }
            // vocal events inline in the text ("(laugh)", "[笑]")
            if speech("events") {
                c.push("vocal_events");
            }
            // a script of several speakers, each its own voice (voices / ref_audios)
            if speech("speakers") {
                c.push("multi_speaker");
            }
            let mut methods = vec!["audio.speak"];
            if speech("design") {
                methods.push("audio.voices.design");
            }
            (c, methods)
        }
        "audio" => (vec!["song", "lyrics", "instrumental"], vec!["audio.generate"]),
        "model3d" => {
            // what its card says it does (`does`): generate (meshes from pictures), texture, views (several), rig, animate
            let does = |c: &str| m["does"].as_array().is_some_and(|a| a.iter().any(|x| x == c));
            let mut c = Vec::new();
            let mut methods = Vec::new();
            if does("generate") {
                c.extend(["mesh", "image_to_3d", "glb"]);
                if does("texture") {
                    c.push("texture");
                }
                if does("views") {
                    c.push("multi_view");
                }
                if does("paint") {
                    c.push("paint");
                }
                methods.push("model3d.generate");
            } else if does("paint") {
                // a paint-only model: model3d.generate with a mesh (or file) paints it
                c.extend(["texture", "paint"]);
                methods.push("model3d.generate");
            }
            if does("rig") {
                c.extend(["rig", "skeleton", "skin_weights"]);
            }
            if does("animate") {
                c.extend(["animate", "text_to_motion"]);
            }
            if does("rig") || does("animate") {
                methods.push("model3d.rig");
            }
            (c, methods)
        }
        "video" => {
            let mut c = vec!["text_to_video", "image_to_video", "sound", "chain", "first_last_frame", "voice_reference"];
            if s(m, "id").contains("ref2va") {
                c.extend(["reference_images", "reference_videos"]);
            } else {
                c.extend(["controlnet", "inpaint"]);
            }
            (c, vec!["video.generate"])
        }
        _ => (Vec::new(), Vec::new()),
    }
}

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
            "model3d" => self.o.ports.model3d,
            "video" => self.o.ports.video,
            _ => self.o.ports.switch,
        }
    }

    pub(crate) fn target(&self, kind: &str) -> Target {
        Target::Tcp(format!("127.0.0.1:{}", self.port_of(kind)))
    }

    /// Where a chat request goes: the switcher when one runs (it loads the model the request names), else the chat
    /// server, `model` (or the one loaded, or the first enabled) loaded first
    pub(crate) fn chat_target(&self, model: Option<&str>) -> Result<Target, (u16, String)> {
        *self.chat_last.lock().unwrap() = now();
        // a card's chat model not downloaded: refused (409 model_not_downloaded), or with NS_AUTO_PULL fetched first
        if let Some(id) = model {
            self.need_downloaded(id, None, true)?;
        }
        if listening(self.o.ports.switch) {
            // the switcher loads the model named: the room for it first, as any load
            if let Some(id) = model {
                self.chat_room(id)?;
            }
            return Ok(self.target("llm"));
        }
        let all = self.models();
        let chat = |m: &&Value| registry::kind_of(m) == "llm" && m["enabled"] != false;
        let id = model.filter(|id| all.iter().filter(chat).any(|m| m["id"] == *id)).map(str::to_string)
            .or_else(|| self.chat_loaded(&all))
            .or_else(|| all.iter().find(chat).map(|m| s(m, "id")))
            .ok_or((409, "no chat model is enabled (system.enable)".to_string()))?;
        self.load(&id)?;
        Ok(Target::Tcp(format!("127.0.0.1:{}", self.o.ports.llm)))
    }

    /// The chat server for `id` started by this page (no studio to ask): the one running stopped first
    fn chat_start(&self, id: &str) -> Result<(), (u16, String)> {
        let _g = self.kind_lock["llm"].lock().unwrap();
        if self.chat_loaded(&self.models()).as_deref() == Some(id) {
            return Ok(());
        }
        if listening(self.o.ports.llm) {
            self.exec("llm stop (another model)".into(), &["llm".into(), "stop".into()]).map_err(|e| (500, e))?;
        }
        let port = self.o.ports.llm.to_string();
        self.exec(format!("llm start {id}"), &["llm".into(), "start".into(), id.into(), "--port".into(), port]).map_err(|e| (500, e))?;
        Ok(())
    }

    /// The memory a model takes on each of its cards, MiB: as seen while it ran, else its model card's `vram_gib` (as
    /// measured where it runs), else from its files' sizes - never more than a card holds beside the spare. An image or
    /// audio model loads whole on one of its cards: what it takes there, for each.
    pub(super) fn vouched_mb(&self, m: &Value, card_mb: u64) -> (BTreeMap<usize, u64>, bool) {
        let id = s(m, "id");
        let gpus = self.gpus_of(m);
        let single = one_card(m);
        // a model that ran fits the card: a peak seen above what a card may be asked for (an edit's, with its vision tower,
        // touching the card's whole memory) would otherwise refuse the model from then on, its margin never free
        let cap = card_mb.saturating_sub(MARGIN_MB + 512);
        if let Some(seen) = self.seen.lock().unwrap().get(&id) {
            if single && !seen.is_empty() {
                let mb = seen.values().copied().max().unwrap_or(0).min(cap);
                return (gpus.iter().map(|g| (*g, mb)).collect(), true);
            }
            if !single && gpus.iter().all(|i| seen.contains_key(i)) {
                return (gpus.iter().map(|g| (*g, seen[g].min(cap))).collect(), true);
            }
        }
        if let Some(g) = m["vram_gib"].as_f64().filter(|g| *g > 0.0) {
            return (gpus.iter().map(|i| (*i, ((g * 1024.0) as u64).min(cap))).collect(), false);
        }
        let files: u64 = registry::paths(m).iter().filter(|p| p.is_file()).filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len() >> 20).sum();
        let per = match registry::kind_of(m) {
            "video" => card_mb.saturating_sub(4096),
            "image" => files * 105 / 100 + 4096,
            "audio" => files * 110 / 100 + 2048,
            "model3d" => files * 110 / 100 + 6144,
            _ => files / gpus.len() as u64 + 2048,
        };
        (gpus.iter().map(|g| (*g, per.min(cap))).collect(), false)
    }

    /// Of what a model is vouched for on each card, MiB: the part that is not its weights - the working memory it
    /// needs to run without running out (KV for its context, arenas, an image's or a song's activations). Its entry's
    /// or its model card's `headroom_gib` (as measured: steady vs peak), else what is vouched less its files' share of
    /// the card (an llm's files split over its cards; an image, audio or video model's whole on each). None known (0):
    /// files at least what is vouched - weights partly off the card (host-mirrored or streamed experts).
    /// Whether it was given (else derived from the files) comes with it.
    fn headroom_mb(m: &Value, cards: &[Value], vouch: &BTreeMap<usize, u64>) -> (BTreeMap<usize, u64>, bool) {
        let card = cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|id| !id.is_empty() && c["id"] == id.as_str()));
        let given = m["headroom_gib"].as_f64().or_else(|| card.and_then(|c| c["headroom_gib"].as_f64())).filter(|g| *g >= 0.0);
        if let Some(g) = given {
            return (vouch.iter().map(|(i, mb)| (*i, ((g * 1024.0) as u64).min(*mb))).collect(), true);
        }
        let paths: std::collections::BTreeSet<_> = registry::paths(m).into_iter().filter(|p| p.is_file()).collect();
        let files: u64 = paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|x| x.len() >> 20).sum();
        let weights = match registry::kind_of(m) {
            "llm" => files / vouch.len().max(1) as u64,
            _ => files,
        };
        (vouch.iter().map(|(i, mb)| (*i, if files > 0 { mb.saturating_sub(weights) } else { 0 })).collect(), false)
    }

    /// What the studio says of the chat card: (selected mode, the mode starting)
    pub(super) fn chat_mode(&self) -> (Option<String>, Option<String>) {
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

    /// The kind's servers' ports, the first's first (image, audio and model3d have a second when its port is set)
    pub(crate) fn slots(&self, kind: &str) -> Vec<u16> {
        let p = &self.o.ports;
        match kind {
            "image" => [p.image, p.image2].into_iter().filter(|x| *x != 0).collect(),
            "audio" => [p.audio, p.audio2].into_iter().filter(|x| *x != 0).collect(),
            "model3d" => [p.model3d, p.model3d2].into_iter().filter(|x| *x != 0).collect(),
            _ => vec![self.port_of(kind)],
        }
    }

    /// Each of the kind's servers: what it holds (the ones that answer), and whether it has work - its own or requests
    /// sent on from here
    pub(super) fn servers(&self, kind: &str) -> Vec<Slot> {
        self.slots(kind).into_iter().enumerate().map(|(slot, port)| {
            let sent = self.inflight.lock().unwrap().get(&port).copied().unwrap_or(0);
            // a server with a request from here is working on it and may not answer until it is done: an image
            // server mid-picture held system.state and system.busy for 6 s a server, past the platform's 8 s, and
            // the box went "offline" for the start of every picture (2026-10-22) - what it said last, busy
            let last = (sent > 0).then(|| self.answers.lock().unwrap().get(&port).cloned()).flatten();
            let st = last.map(|(m, l, _)| (m, l, true)).or_else(|| self.server_at(port));
            // a load started here that does not answer yet: its model, loading (so no other kind plans as if the card held nothing)
            let starting = if st.is_none() { self.pending.lock().unwrap().iter().find(|p| p.port == port && p.answered.is_none()).map(|p| p.model.clone()) } else { None };
            Slot { slot, port, model: st.as_ref().map(|x| x.0.clone()).or(starting.clone()), loading: st.as_ref().is_some_and(|x| x.1) || starting.is_some(),
                   busy: st.as_ref().is_some_and(|x| x.2) || sent > 0, inflight: sent }
        }).collect()
    }

    /// Every place work runs, and on which card: the image and audio servers (two each when the second port is set),
    /// the video daemon's GPUs, the chat server's cards
    pub(crate) fn work_slots(&self, all: &[Value], cards: &Value) -> Vec<Value> {
        let mut v = Vec::new();
        for kind in ["image", "audio", "model3d"] {
            if !all.iter().any(|m| registry::kind_of(m) == kind && m["enabled"] != false) {
                continue;
            }
            for x in self.servers(kind) {
                v.push(json!({"kind": kind, "slot": x.slot, "port": x.port, "gpu": card_of_port(cards, x.port),
                              "model": x.model, "loading": x.loading, "busy": x.busy && !x.loading, "up": x.model.is_some()}));
            }
        }
        if all.iter().any(|m| registry::kind_of(m) == "video" && m["enabled"] != false) {
            if let Ok(st) = call_for(&self.target("video"), "GET", "/engine/status", None, 4) {
                for g in st["gpus"].as_array().into_iter().flatten() {
                    v.push(json!({"kind": "video", "slot": g["gpu"], "gpu": g["gpu"], "model": g["denoiser"],
                                  "loading": false, "busy": !g["running"].is_null(), "up": g["engine"].as_str().is_some_and(|e| e != "unloaded")}));
                }
            }
        }
        if let Some(id) = all.iter().any(|m| registry::kind_of(m) == "llm").then(|| self.chat_loaded(all)).flatten() {
            let gpus = all.iter().find(|m| m["id"] == id.as_str()).map(|m| self.gpus_of(m)).unwrap_or_default();
            v.push(json!({"kind": "llm", "slot": 0, "gpu": gpus.first(), "gpus": gpus, "model": id, "loading": false, "busy": false, "up": true}));
        }
        v
    }

    /// The models the kind's servers hold
    pub(crate) fn loaded_of(&self, kind: &str) -> Vec<String> {
        self.slots(kind).into_iter().filter_map(|p| self.server_at(p).map(|x| x.0)).collect()
    }

    /// Where to fetch a file the kind's servers made (they share the output directory): a server that answers
    pub(crate) fn file_target(&self, kind: &str) -> Target {
        let p = self.slots(kind).into_iter().find(|p| listening(*p)).unwrap_or(self.port_of(kind));
        Target::Tcp(format!("127.0.0.1:{p}"))
    }

    /// The image or audio server on `port`: its model, whether it is loading, whether it is busy - when one answers
    fn server_at(&self, port: u16) -> Option<(String, bool, bool)> {
        let t = Target::Tcp(format!("127.0.0.1:{port}"));
        let info = call_for(&t, "GET", "/api/info", None, 3).ok()?;
        let p = call_for(&t, "GET", "/api/progress", None, 3).unwrap_or(json!({}));
        let st = (s(&info, "model"), info["loading"] == true, p["busy"] == true || p["waiting"].as_u64().unwrap_or(0) > 0);
        self.answers.lock().unwrap().insert(port, st.clone());
        Some(st)
    }

    /// The video daemon's engines (their model ids) and the ones loaded on a GPU, when it runs
    pub(super) fn video_daemon(&self) -> Option<VideoDaemon> {
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
        let image = if need("image") { self.servers("image") } else { Vec::new() };
        let audio = if need("audio") { self.servers("audio") } else { Vec::new() };
        let m3d = if need("model3d") { self.servers("model3d") } else { Vec::new() };
        let slots_of = |k: &str| -> &Vec<Slot> { match k { "image" => &image, "audio" => &audio, _ => &m3d } };
        let video = if need("video") { self.video_daemon() } else { None };
        // the model cards, read once: for the headroom of entries registered before cards carried it
        // and whether a chat model spans every card
        let mc = if all.iter().all(|m| m.get("headroom_gib").is_some()) && !need("llm") { Vec::new() } else { self.o.cfg.cards() };
        let holds = self.holds();
        all.iter().filter(|m| registry::kind_of(m) != "lora").map(|m| {
            let (id, kind) = (s(m, "id"), registry::kind_of(m).to_string());
            let enabled = m["enabled"] != false;
            let gpus = self.gpus_of(m);
            let (mut vouch, measured) = self.vouched_mb(m, gpus.first().map_or(32768, |g| card_mb(*g)));
            // an image or audio model allowed several cards counts on the ones its copies are on, else the first it tries
            if one_card(m) && vouch.len() > 1 {
                let on: Vec<usize> = held_on(cards, &id).into_iter().filter(|g| vouch.contains_key(g)).collect();
                let on = if on.is_empty() { vec![gpus[0]] } else { on };
                vouch.retain(|g, _| on.contains(g));
            }
            let (headroom, headroom_given) = Self::headroom_mb(m, &mc, &vouch);
            let mut state = match kind.as_str() {
                "llm" if chat.0.as_deref() == Some(id.as_str()) => "loaded",
                "llm" if chat.1 .1.as_deref() == Some(id.as_str()) => "loading",
                "image" | "audio" | "model3d" => {
                    let mine: Vec<&Slot> = slots_of(&kind).iter().filter(|x| x.model.as_deref() == Some(id.as_str())).collect();
                    if mine.iter().any(|x| x.busy && !x.loading) {
                        "busy"
                    } else if mine.iter().any(|x| !x.loading) {
                        "loaded"
                    } else if !mine.is_empty() {
                        "loading"
                    } else {
                        "unloaded"
                    }
                }
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
            let (caps, methods) = capabilities(m);
            // the profile its model card says ("unmeasured" until a bench does) and the copies running now
            let prof = self.profile_of(m);
            let (concurrency, coexistence) = prof.to_json();
            let copies_now = match kind.as_str() {
                "llm" => usize::from(chat.0.as_deref() == Some(id.as_str())),
                "image" | "audio" | "model3d" => slots_of(&kind).iter().filter(|x| x.model.as_deref() == Some(id.as_str()) && !x.loading).count(),
                "video" => video.as_ref().map_or(0, |(_, loaded)| loaded.iter().filter(|(e, _)| *e == id).count()),
                _ => 0,
            };
            json!({"concurrency": concurrency, "coexistence": coexistence, "copies_now": copies_now,
                   "recommended_copies": prof.concurrency.recommended_copies, "max_copies_per_card": prof.concurrency.max_copies_per_card,
                   "id": id, "kind": kind, "title": m["title"], "arch": m["arch"], "enabled": enabled, "gpus": gpus, "state": state,
                   "spans": spans_all(m, &mc),
                   "pinned": enabled && m["pinned"] == true, "held_s": holds.get(s(m, "id").as_str()).map_or(0.0, |h| h.reserved(now()).round()),
                   "capabilities": caps, "methods": methods,
                   "vouch_mb": vouch.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "measured": measured,
                   "headroom_mb": headroom.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "headroom_given": headroom_given,
                   "page": match kind.as_str() { "image" | "audio" | "model3d" => json!(slots_of(&kind).iter()
                                                      .find(|x| x.model.as_deref() == Some(id.as_str())).map_or(self.port_of(&kind), |x| x.port)),
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
            let hr = |m: &Value| m["headroom_mb"][&gi].as_u64().unwrap_or(0);
            // chat models and video engines count once a card: one of each runs there at a time (the loaded one, else
            // the largest)
            let mut list: Vec<Value> = on.iter().filter(|m| !matches!(m["kind"].as_str(), Some("llm" | "video")))
                .map(|m| json!({"model": m["id"], "kind": m["kind"], "mb": mb(m), "headroom_mb": hr(m), "headroom_given": m["headroom_given"],
                                "measured": m["measured"], "state": m["state"]})).collect();
            for kind in ["llm", "video"] {
                let group: Vec<&&Value> = on.iter().filter(|m| m["kind"] == kind).collect();
                if let Some(c) = group.iter().find(|m| m["state"] == "loaded").or_else(|| group.iter().max_by_key(|m| mb(m))) {
                    list.push(json!({"model": c["id"], "kind": kind, "mb": mb(c), "headroom_mb": hr(c), "headroom_given": c["headroom_given"],
                                     "measured": c["measured"], "state": c["state"], "of": group.len()}));
                }
            }
            let sum: u64 = list.iter().map(|v| v["mb"].as_u64().unwrap_or(0)).sum();
            c["vouched"] = json!(list);
            c["vouched_mb"] = json!(sum);
            // of it, the models' headroom (their working memory beyond the weights)
            c["vouched_headroom_mb"] = json!(list.iter().map(|v| v["headroom_mb"].as_u64().unwrap_or(0)).sum::<u64>());
            c["overbooked"] = json!(sum > total);
        }
    }

    pub(super) fn entry(&self, id: &str) -> Result<Value, (u16, String)> {
        let m = registry::find(&self.o.cfg, id).map_err(|e| (500, e))?.ok_or_else(|| (404, format!("no model {id} (nextsycl models list)")))?;
        if registry::kind_of(&m) == "lora" {
            return Err((400, format!("{id} is a LoRA: it is enabled with the model that uses it")));
        }
        Ok(m)
    }

    /// Enable `id` on cards `gpus` (a list; none: as it was)
    pub(crate) fn enable(self: &Arc<Self>, id: &str, gpus: Option<&Value>) -> Result<String, (u16, String)> {
        let m = self.entry(id)?;
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        let n = self.ncards();
        let gpus: Option<Vec<usize>> = gpus.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|g| g as usize).collect());
        if let Some(g) = &gpus {
            if g.is_empty() || g.iter().any(|x| *x >= n) {
                return Err((400, format!("gpus {g:?}: one or more of 0..{}", n - 1)));
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

    /// The cards an enabled (or disabled) model may use, changed in place - more than fit is allowed (the card is
    /// shown overbooked: the first model to load holds it). A loaded model whose card is taken away unloads (after
    /// the requests it is answering): its next request loads it on the new cards.
    pub(crate) fn set_gpus(self: &Arc<Self>, id: &str, gpus: Option<&Value>) -> Result<Value, (u16, String)> {
        let m = self.entry(id)?;
        let n = self.ncards();
        let g: Vec<usize> = gpus.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|g| g as usize).collect()).unwrap_or_default();
        if g.is_empty() || g.iter().any(|x| *x >= n) {
            return Err((400, format!("gpus {g:?}: one or more of 0..{}", n - 1)));
        }
        let kind = registry::kind_of(&m).to_string();
        let mut g = g;
        g.sort();
        g.dedup();
        if g.len() < n && spans_all(&m, &self.o.cfg.cards()) {
            return Err((400, format!("{id} is spread over every card (its model card: gpus all): it does not run on fewer")));
        }
        let old = self.gpus_of(&m);
        if old == g {
            return Ok(json!({"model": id, "gpus": g, "changed": false}));
        }
        // a loaded model leaves the cards it was on (it reloads on the new ones at its next request)
        let state = self.model_states(std::slice::from_ref(&m), &json!([])).first().map(|x| s(x, "state")).unwrap_or_default();
        // (an image or audio model only when its card is the one taken away)
        let on = held_on(&self.named_cards(&self.models()), id);
        let kept = one_card(&m) && !on.is_empty() && on.iter().all(|c| g.contains(c));
        if m["enabled"] != false && matches!(state.as_str(), "loaded" | "busy" | "loading") && kind != "video" && !kept {
            self.unload(id)?;
        }
        registry::update(&self.o.cfg, id, |e| e["gpus"] = json!(g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))).map_err(|e| (500, e))?;
        if kind == "video" && m["enabled"] != false {
            let me = self.clone();
            std::thread::spawn(move || {
                if let Err(e) = me.video_apply(false) {
                    let (rid, log) = me.new_run("video: apply the enabled models' cards".into());
                    let _ = std::fs::write(log, e);
                    me.finish(rid, 1);
                }
            });
        }
        Ok(json!({"model": id, "gpus": g, "changed": true, "was": old}))
    }

    /// Disable `id`: unloaded first (after the requests it is answering); a video model while a clip renders only
    /// with `force` (the clip stops at its next step)
    pub(crate) fn disable(self: &Arc<Self>, id: &str, force: bool) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if registry::kind_of(&m) == "video" && !force && self.rendering() {
            return Err((409, "a clip is rendering: pause the studio's queue and wait, or force (the clip stops at its next step)".into()));
        }
        self.unload(id)?;
        registry::update(&self.o.cfg, id, |e| {
            e["enabled"] = json!(false);
            e["pinned"] = json!(false);
        }).map_err(|e| (500, e))?;
        if registry::kind_of(&m) == "video" {
            self.video_apply(force).map_err(|e| (500, e))?;
        }
        Ok(())
    }

    fn rendering(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok()
            .map(|v| v.get("result").cloned().unwrap_or(v)).is_some_and(|r| r["idle"] == false)
    }

    /// Load `id` now, in the background (system.load, for tools)
    pub(crate) fn load_async(self: &Arc<Self>, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled: enable it first")));
        }
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        // a GPU run holds the cards: refused now (the load would only wait in the background)
        if let Some(why) = self.cards_taken() {
            let loaded = match registry::kind_of(&m) {
                "llm" => self.chat_loaded(&self.models()).as_deref() == Some(id),
                "video" => true,
                k => self.loaded_of(k).iter().any(|x| x == id),
            };
            if !loaded {
                return Err((503, why));
            }
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
        if let Some(why) = self.moving_why(id) {
            return Err((409, why));
        }
        match registry::kind_of(&m) {
            "image" | "audio" | "model3d" => self.ensure(id).map(|_| ()),
            "video" => self.video_apply(false).map_err(|e| (500, e)),
            _ => {
                let all = self.models();
                if self.chat_loaded(&all).as_deref() == Some(id) {
                    return Ok(());
                }
                self.chat_room(id)?;
                if !listening(self.o.ports.video) {
                    return self.chat_start(id);
                }
                call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": id})), 600).map_err(|e| (502, format!("the studio's llm.mode: {e}")))?;
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

    /// Unload `id` if it is loaded (after the requests it is answering): a tool's say (a pin brings it back)
    pub(crate) fn unload(&self, id: &str) -> Result<(), (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        match kind {
            "image" | "audio" | "model3d" => {
                let _g = self.kind_lock[kind].lock().unwrap();
                for x in self.servers(kind).iter().filter(|x| x.model.as_deref() == Some(id)) {
                    self.exec(format!("{kind} stop ({id})"), &[kind.into(), "stop".into(), "--slot".into(), x.slot.to_string()]).map_err(|e| (500, e))?;
                }
                Ok(())
            }
            "video" => self.exec("video: unload the engines".into(), &["video".into(), "unload".into()]).map(|_| ()).map_err(|e| (500, e)),
            _ => {
                if self.chat_loaded(&self.models()).as_deref() == Some(id) && !listening(self.o.ports.video) {
                    return self.exec(format!("llm stop ({id})"), &["llm".into(), "stop".into()]).map(|_| ()).map_err(|e| (500, e));
                }
                if self.chat_loaded(&self.models()).as_deref() == Some(id) {
                    call_for(&self.target("video"), "POST", "/rpc/llm.mode", Some(&json!({"mode": "none"})), 120)
                        .map_err(|e| (502, format!("the studio's llm.mode: {e}")))?;
                    // started some other way than the studio's mode: still up, stopped as such
                    if self.chat_loaded(&self.models()).as_deref() == Some(id) {
                        return self.exec(format!("llm stop ({id})"), &["llm".into(), "stop".into()]).map(|_| ()).map_err(|e| (500, e));
                    }
                }
                Ok(())
            }
        }
    }

    /// The image, audio or 3D server holding `id`, started (or swapped to it) when it does not: where to send its requests
    pub(crate) fn ensure(&self, id: &str) -> Result<Lease<'_>, (u16, String)> {
        self.ensure_for(id, None)
    }

    /// `ensure` for a job (`job`: its id): what it waits for is noted for <kind>.status (stage, waiting_for), it waits
    /// for room up to JOB_ROOM_WAIT_S (a request without a job ROOM_WAIT_S), and a cancel ends the wait
    pub(crate) fn ensure_for(&self, id: &str, job: Option<&str>) -> Result<Lease<'_>, (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        if !matches!(kind, "image" | "audio" | "model3d") {
            return Err((400, format!("{id} is a {kind} model")));
        }
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled (enable it on :8000 or nextsycl models enable {id})")));
        }
        // (the caller's max_willing_to_wait_s replaces these and enforces itself: Host::give_up)
        let default = if super::backplane::has_limit() { Duration::MAX } else { Duration::from_secs(if job.is_some() { JOB_ROOM_WAIT_S } else { ROOM_WAIT_S }) };
        let info = self.model_info(&m);
        let card_mb = self.cards.lock().unwrap().get(info.gpus[0]).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
        let (need_mb, _) = self.vouched_mb(&m, card_mb);
        let asked_mb = self.asked.lock().unwrap().get(id).copied().unwrap_or(0);
        let req = PlaceReq { model: &info, job, need_mb: &need_mb, asked_mb };
        // the scheduler's driver is the only wait: cancel, files moving, cards taken, the limit, room
        let (_, port) = place::ensure(&Sched(self), &req, default, None, &Value::Null)?;
        self.request_on(id);
        Ok(Lease { target: Target::Tcp(format!("127.0.0.1:{port}")), port, home: self })
    }

    /// One consistent view of the box for the scheduler: the cards (their free memory as `free_mb` counts it, loads under
    /// way among their holders), the servers, the models with their profiles, pins and holds
    pub(super) fn snapshot(&self) -> Snapshot {
        self.snapshot_of(&self.models(), true)
    }

    /// `snapshot`; without `servers` it asks no server, for what needs the cards alone (the last check before a start)
    fn snapshot_of(&self, all: &[Value], servers: bool) -> Snapshot {
        let named = self.named_cards(all);
        let daemon = if named.as_array().into_iter().flatten().any(|c| c["procs"].as_array().into_iter().flatten().any(|p| p["kind"] == "video")) { self.video_daemon() } else { None };
        let pending: Vec<super::Pending> = self.pending.lock().unwrap().iter().filter(|p| p.port != 0).cloned().collect();
        let cards = named.as_array().into_iter().flatten().enumerate().map(|(i, c)| {
            let index = c["index"].as_u64().map_or(i, |g| g as usize);
            let total_mb = c["vram_total_mb"].as_u64().unwrap_or(0);
            let mut procs: Vec<Proc> = c["procs"].as_array().into_iter().flatten().map(|p| {
                let kind = p["kind"].as_str().filter(|k| matches!(*k, "llm" | "image" | "audio" | "video" | "model3d")).map(str::to_string);
                let mut model = p["model_id"].as_str().or(p["model"].as_str()).or(p["program"].as_str()).unwrap_or("?").to_string();
                // a video worker by what its slot says it holds (its process's arguments may not name a registered model)
                if kind.as_deref() == Some("video") {
                    if let Some((e, _)) = daemon.as_ref().and_then(|d| d.1.iter().find(|(_, on)| *on == index)) {
                        model = e.clone();
                    }
                }
                Proc { kind, model, port: p["port"].as_str().and_then(|x| x.parse().ok()), mb: p["vram_mb"].as_u64().unwrap_or(0) }
            }).collect();
            // what only the root sampler sees (processes of other users): no resident, but the message names it
            if let Some(mb) = c["vram_unseen_mb"].as_u64().filter(|mb| *mb >= 256) {
                procs.push(Proc { kind: None, model: "processes of other users".into(), port: None, mb });
            }
            // a load under way is a resident before its memory shows
            for p in pending.iter().filter(|p| p.gpu == index && p.answered.is_none()) {
                if !procs.iter().any(|x| x.port == Some(p.port)) {
                    procs.push(Proc { kind: Some(p.kind.clone()), model: p.model.clone(), port: Some(p.port), mb: 0 });
                }
            }
            Card { index, name: s(c, "name"), total_mb, used_mb: total_mb.saturating_sub(self.free_mb(&named, i)), procs }
        }).collect();
        let servers = if servers { ["image", "audio", "model3d"].into_iter().map(|k| (k.to_string(), self.servers(k))).collect() } else { BTreeMap::new() };
        let models: BTreeMap<String, ModelInfo> = all.iter().filter(|m| registry::kind_of(m) != "lora").map(|m| (s(m, "id"), self.model_info(m))).collect();
        let pinned: std::collections::BTreeSet<String> = all.iter().filter(|m| m["pinned"] == true).map(|m| s(m, "id")).collect();
        let at = now();
        let held_left: BTreeMap<String, f64> = self.holds().iter().map(|(id, h)| (id.clone(), h.reserved(at))).filter(|(_, l)| *l > 0.0).collect();
        // (a chat model's keep is not here: it is chat_idle's, so it yields to another chat model)
        let mut kept = pinned.clone();
        kept.extend(held_left.keys().cloned());
        Snapshot { cards, servers, models, pinned, kept, held_left }
    }

    /// A registered model as the scheduler needs it
    pub(super) fn model_info(&self, m: &Value) -> ModelInfo {
        ModelInfo { id: s(m, "id"), kind: registry::kind_of(m).to_string(), enabled: m["enabled"] != false, gpus: self.gpus_of(m), profile: self.profile_of(m) }
    }

    /// A model's profile: its entry's, else its model card's (validated when the cards load), else one copy, shared
    pub(crate) fn profile_of(&self, m: &Value) -> Profile {
        if m.get("concurrency").is_some() || m.get("coexistence").is_some() {
            return Profile::of_entry(m);
        }
        self.o.cfg.cards().iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|id| !id.is_empty() && c["id"] == id.as_str()))
            .map_or_else(Profile::default, Profile::of_entry)
    }

    /// `system.busy`'s `engines`: what a dispatcher needs of each enabled model
    pub(super) fn engines(states: &[Value]) -> Vec<Value> {
        states.iter().filter(|m| m["enabled"] == true).map(|m| json!({"id": m["id"], "kind": m["kind"], "copies_now": m["copies_now"],
            "recommended_copies": m["recommended_copies"], "max_copies_per_card": m["max_copies_per_card"],
            "mode": m["coexistence"]["mode"], "status": m["concurrency"]["status"]})).collect()
    }

    /// The message for no room
    pub(super) fn no_room(&self, snap: &Snapshot, gpus: &[usize], m: &ModelInfo, want: u64) -> String {
        room::no_room(snap, gpus, m, want)
    }

    /// The scheduler's room plan for card `g`, with this box's idleness
    pub(super) fn room_plan(&self, snap: &Snapshot, g: usize, want: u64, asking: &str, light: bool, m: Option<&ModelInfo>) -> Option<Vec<Holder>> {
        room::room_plan(snap, &Sched(self), g, want, asking, light, m)
    }

    pub(super) fn free(&self, plan: &[Holder], kind: &str, id: &str) -> Result<(), (u16, String)> {
        place::free_plan(&Sched(self), plan, kind, id)
    }

    pub(super) fn wait_room(&self, g: usize, want: u64) -> bool {
        place::wait_room(&Sched(self), g, want)
    }

    /// A model's own host memory, MiB, and the cache it keeps warm (its entry's, else its model card's `ram_gib`, `ram_cache_gib`)
    fn ram_of(&self, id: &str) -> (Option<u64>, u64) {
        let all = self.models();
        let Some(m) = all.iter().find(|m| m["id"] == id) else { return (None, 0) };
        let cards = self.o.cfg.cards();
        let card = cards.iter().find(|c| [s(m, "catalog"), s(m, "id")].iter().any(|i| !i.is_empty() && c["id"] == i.as_str()));
        let f = |k: &str| m[k].as_f64().or_else(|| card.and_then(|c| c[k].as_f64())).filter(|g| *g > 0.0).map(|g| (g * 1024.0) as u64);
        (f("ram_gib"), f("ram_cache_gib").unwrap_or(0))
    }

    /// Host memory free for a load now, MiB (None: not known here)
    pub(super) fn host_free_now(&self, cards: &[Card]) -> Option<u64> {
        let seen: std::collections::BTreeSet<&str> = cards.iter().flat_map(|c| c.procs.iter()).filter(|p| p.kind.is_some()).map(|p| p.model.as_str()).collect();
        let warm: u64 = seen.iter().map(|i| self.ram_of(i).1).sum();
        Some(host_free_mb(mem_available_mb()?, warm))
    }

    /// The host memory rule for `id` (a model with `ram_gib`; none: no rule): what must go (idle holders on any card, their `ram_gib` and warm cache counted), or why it cannot
    /// load now. `leaving`: the chat model that goes anyway when this one comes.
    fn ram_room(&self, snap: &Snapshot, id: &str, asking: &str, leaving: Option<&str>) -> Result<Vec<Holder>, String> {
        let (Some(need), _) = self.ram_of(id) else { return Ok(Vec::new()) };
        let Some(mut free) = self.host_free_now(&snap.cards) else { return Ok(Vec::new()) };
        if let Some(l) = leaving {
            let (r, c) = self.ram_of(l);
            free += r.unwrap_or(0) + c;
        }
        let want = need + RAM_MARGIN_MB;
        if free >= want {
            return Ok(Vec::new());
        }
        let mut names = std::collections::BTreeSet::new();
        let mut idle: Vec<Holder> = Vec::new();
        for c in &snap.cards {
            for mut h in room::evictable(room::idle_on(snap, &Sched(self), c.index, asking, false), &snap.kept) {
                if !names.insert(h.model.clone()) {
                    continue;
                }
                let (r, w) = self.ram_of(&h.model);
                h.mb = r.unwrap_or(0) + w;
                idle.push(h);
            }
        }
        room::plan_room(free, want, idle).ok_or_else(|| {
            let all: Vec<usize> = snap.cards.iter().map(|c| c.index).collect();
            format!("host memory is held by work that is not idle - {:.1} GiB free; {id} needs ~{:.1} GiB{}", free as f64 / 1024.0, want as f64 / 1024.0, room::kept_words(snap, &all, None))
        })
    }

    /// Wait (30 s at most) until the host has `id`'s memory free after holders went
    fn wait_ram(&self, id: &str) -> bool {
        let (Some(need), _) = self.ram_of(id) else { return true };
        let cards = self.snapshot_of(&self.models(), false).cards;
        let t0 = Instant::now();
        loop {
            if self.host_free_now(&cards).is_none_or(|f| f >= need + RAM_MARGIN_MB) {
                return true;
            }
            if t0.elapsed() >= Duration::from_secs(30) {
                return false;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    /// The same room rule for the video daemon's workers waiting for a card's memory: on any card, however the model there was started. Running forever beside the card sampler.
    pub(super) fn video_room_loop(&self) {
        loop {
            std::thread::sleep(Duration::from_secs(5));
            if self.cards_taken().is_some() || !listening(self.o.ports.video) {
                continue;
            }
            let Ok(st) = call_for(&self.target("video"), "GET", "/engine/status", None, 4) else { continue };
            for (g, need) in waiting_workers(&st) {
                let snap = self.snapshot();
                if let Some(plan) = self.room_plan(&snap, g, need, "video", false, None) {
                    if let Err((_, e)) = self.free(&plan, "video", "a video job") {
                        eprintln!("video: GPU {g}: {e}");
                    }
                }
            }
        }
    }

    /// What card `g` has free now, MiB: its memory less what is in use - by every process, ours or not (the page's own
    /// sampler, or the root sampler's gpustat file when that sees more: mod.rs sample_cards) - and less what the
    /// loads started here have yet to take
    pub(crate) fn free_mb(&self, cards: &Value, g: usize) -> u64 {
        let Some(c) = cards.get(g) else { return 0 };
        let pending: Vec<(u16, u64)> = {
            let mut l = self.pending.lock().unwrap();
            l.retain(|p| p.answered.is_none_or(|t| t.elapsed() < Duration::from_secs(6)));
            l.iter().filter(|p| p.gpu == g).map(|p| (p.port, p.mb)).collect()
        };
        free_on(c, &pending)
    }

    /// Whether the studio has nothing to make: no clip running, none runnable in the queue
    pub(super) fn video_idle(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok().map(|v| v.get("result").cloned().unwrap_or(v))
            .is_some_and(|st| st["idle"] == true && st["queue_items"].as_array().is_none_or(|a| a.iter().all(|i| i["held"] == true)))
    }

    /// The chat model loaded, when it may be preempted: answering nothing, not changing, and not kept - a request
    /// that sent X-Keep-Up-S (Open WebUI's connection does: 300 s) keeps it that long after it ended, here or through
    /// the switcher (its GET /keep); a request without the header keeps nothing
    pub(super) fn chat_idle(&self) -> Option<String> {
        match self.chat_hold() {
            ChatHold::Idle(id) => Some(id),
            ChatHold::NoModel | ChatHold::Held(_) => None,
        }
    }

    /// `chat_idle`, telling "no model" from "held": a server or switcher that does not answer counts as held
    pub(super) fn chat_hold(&self) -> ChatHold {
        let Some(id) = self.chat_loaded(&self.models()) else {
            // one on its way up is somebody's request
            return if self.chat_mode().1.is_some() { ChatHold::Held(None) } else { ChatHold::NoModel };
        };
        let answering = call_for(&Target::Unix(self.chat_socket()), "GET", "/server/requests", None, 3).ok()
            .map(|v| v["active"].as_array().is_some_and(|a| !a.is_empty()));
        // a pin, rsvp_s or this door's keeps
        let kept_here = self.pinned(&id) || self.holds().get(&id).is_some_and(|h| h.left(now()) > 0.0);
        let kept_by_switcher = if listening(self.o.ports.switch) {
            call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.switch)), "GET", "/keep", None, 3).ok()
                .map(|k| k["kept"][id.as_str()].as_f64().is_some_and(|s| s > 0.0))
        } else {
            Some(false)
        };
        chat_hold_of(id, answering, self.chat_mode().1.is_some(), kept_here, kept_by_switcher)
    }

    /// `GET /chat/held`, what the video daemon asks before it stops the chat model
    pub(super) fn chat_held_answer(&self) -> Value {
        match self.chat_hold() {
            ChatHold::NoModel => json!({"held": false, "model": null}),
            ChatHold::Idle(id) => json!({"held": false, "model": id}),
            ChatHold::Held(id) => json!({"held": true, "model": id}),
        }
    }

    /// The chat server's control socket (its /server routes are not on its port)
    pub(crate) fn chat_socket(&self) -> std::path::PathBuf {
        self.o.cfg.get("NS_SOCKET_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into())).join("nextsycl")
        }).join("nextsycl.sock")
    }

    /// Room for chat model `id` before it loads (it is not loaded): the chat model loaded now counts as free (it goes
    /// when this one comes); idle picture, sound and video work on its cards stops for it; else, after ROOM_WAIT_S, 503
    /// no_room - the answer the other kinds give
    pub(crate) fn chat_room(&self, id: &str) -> Result<(), (u16, String)> {
        let all = self.models();
        let Some(m) = all.iter().find(|m| m["id"] == id && registry::kind_of(m) == "llm") else { return Ok(()) };
        if self.chat_loaded(&all).as_deref() == Some(id) {
            return Ok(());
        }
        // one chat server: a pinned or reserved model there does not give way to this one
        if let Some(why) = self.chat_loaded(&all).and_then(|c| self.swap_blocked(&c)) {
            return Err((503, format!("the chat server holds one model at a time and {why}; {id} loads once it is released")));
        }
        // a GPU run holds the cards (crate::gpulock): no chat model loads beside it
        if let Some(why) = self.cards_taken() {
            return Err((503, why));
        }
        let info = self.model_info(m);
        let gpus = info.gpus.clone();
        let t0 = Instant::now();
        let room_wait = if super::backplane::has_limit() { Duration::MAX } else { Duration::from_secs(ROOM_WAIT_S) };
        loop {
            if let Some(f) = self.past_limit() {
                return Err(f);
            }
            let full = self.snapshot_of(&all, true);
            let card_mb = |g: usize| full.cards.iter().find(|c| c.index == g).map_or(32768, |c| c.total_mb);
            let (need, _) = self.vouched_mb(m, card_mb(gpus[0]));
            // the chat model loaded now leaves when this one comes: what it holds counts as free and it is no resident
            let chat_on = |g: usize| full.cards.iter().find(|c| c.index == g).into_iter().flat_map(|c| c.procs.iter()).filter(|p| p.kind.as_deref() == Some("llm")).map(|p| p.mb).sum::<u64>();
            let mut snap = full.clone();
            for c in &mut snap.cards {
                c.used_mb = c.used_mb.saturating_sub(chat_on(c.index));
                c.procs.retain(|p| p.kind.as_deref() != Some("llm"));
            }
            let want = |g: usize| need.get(&g).copied().unwrap_or(0) + MARGIN_MB;
            let plans: Vec<(usize, Option<Vec<Holder>>)> = gpus.iter().map(|g| (*g, self.room_plan(&snap, *g, want(*g), "llm", false, Some(&info)))).collect();
            let short: Vec<usize> = plans.iter().filter(|(_, p)| !p.as_ref().is_some_and(Vec::is_empty)).map(|(g, _)| *g).collect();
            let loaded = self.chat_loaded(&all);
            let ram = self.ram_room(&full, id, "llm", loaded.as_deref());
            if short.is_empty() {
                match ram {
                    Ok(p) if p.is_empty() => return Ok(()),
                    Ok(p) => {
                        if self.free(&p, "llm", id).is_ok() && self.wait_ram(id) {
                            return Ok(());
                        }
                    }
                    Err(why) => {
                        if t0.elapsed() > room_wait {
                            return Err((503, format!("{why}; {id} loads once one of them idles out or is stopped")));
                        }
                    }
                }
                std::thread::sleep(Duration::from_secs(3));
                continue;
            }
            let mut freed = false;
            for (g, plan) in plans.iter().filter(|(g, _)| short.contains(g)) {
                if let Some(plan) = plan {
                    if self.free(plan, "llm", id).is_ok() {
                        // (the real card still holds the chat model that leaves)
                        self.wait_room(*g, want(*g).saturating_sub(chat_on(*g)));
                        freed = true;
                    }
                }
            }
            if t0.elapsed() > room_wait {
                let lack = short.iter().map(|g| self.no_room(&full, &[*g], &info, want(*g))).collect::<Vec<_>>().join(" / ");
                return Err((503, format!("{lack}; {id} loads once one of them idles out or is stopped")));
            }
            if !freed {
                std::thread::sleep(Duration::from_secs(3));
            }
        }
    }

    /// A GPU run holding the cards (crate::gpulock, NS_GPU_LOCK): the sentence a refused load or a waiting job reads
    pub(crate) fn cards_taken(&self) -> Option<String> {
        let f = crate::gpulock::path(self.o.cfg.get("NS_GPU_LOCK"))?;
        crate::gpulock::held(&f).map(|h| crate::gpulock::message(&f, &h))
    }

    /// `id` started in server `slot` on card `gpu`, answering; until a moment after it answers, the memory it has yet to
    /// take counts as taken there (free_mb; `Sched::reserve` took it)
    fn start_at(&self, kind: &str, id: &str, slot: usize, port: u16, gpu: usize) -> Result<(), (u16, String)> {
        let r = self.start_at_(kind, id, slot, port, gpu);
        // kept a few seconds more: the cards are sampled every 2 s, the server's memory shows on the next
        for p in self.pending.lock().unwrap().iter_mut().filter(|p| p.port == port && p.answered.is_none()) {
            p.answered = Some(Instant::now());
        }
        r
    }

    fn start_at_(&self, kind: &str, id: &str, slot: usize, port: u16, gpu: usize) -> Result<(), (u16, String)> {
        let secs = (self.o.idle_minutes * 60).to_string();
        // (a pinned model's server does not end itself when idle)
        let mut args = vec![kind.to_string(), "start".into(), id.into(), "--gpu".into(), gpu.to_string(), "--port".into(), port.to_string(),
                            "--host".into(), self.o.serve_host.clone(), "--wfe".into(), "--slot".into(), slot.to_string()];
        if self.o.idle_minutes > 0 && !self.pinned(id) {
            args.extend(["--idle-exit".into(), secs]);
        }
        self.loading.lock().unwrap().insert(id.to_string(), now());
        let r = self.exec(format!("{kind} start {id} on GPU {gpu}{}", if slot > 0 { format!(" (server {})", slot + 1) } else { String::new() }), &args);
        self.loading.lock().unwrap().remove(id);
        r.map_err(|e| (502, e))?;
        self.wait_up(port, id)
    }

    fn wait_up(&self, port: u16, id: &str) -> Result<(), (u16, String)> {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(600) {
            if self.server_at(port).is_some_and(|(m, l, _)| m == id && !l) {
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
        args.extend(["--door".into(), self.o.door.clone()]);
        self.exec(format!("video start ({})", want.join(", ")), &args).map(|_| ())
    }
}

/// The box as the scheduler sees it: crates/scheduler decides, this does (its README's table of what each method is)
pub(super) struct Sched<'h>(pub(super) &'h Home);

impl Idleness for Sched<'_> {
    fn video_idle(&self) -> bool {
        self.0.video_idle()
    }
    fn chat_idle(&self) -> Option<String> {
        self.0.chat_idle()
    }
}

impl Host for Sched<'_> {
    fn now(&self) -> f64 {
        now()
    }
    fn sleep(&self, secs: f64) {
        std::thread::sleep(Duration::from_secs_f64(secs));
    }
    fn snapshot(&self) -> Snapshot {
        self.0.snapshot()
    }
    fn server(&self, kind: &str, port: u16) -> Option<room::Server> {
        self.0.servers(kind).into_iter().find(|x| x.port == port)
    }
    fn start(&self, kind: &str, id: &str, slot: usize, port: u16, gpu: usize) -> Result<(), (u16, String)> {
        self.0.start_at(kind, id, slot, port, gpu)
    }
    fn stop(&self, kind: &str, slot: usize, _port: u16, label: &str) -> Result<(), (u16, String)> {
        self.0.exec(label.to_string(), &[kind.into(), "stop".into(), "--slot".into(), slot.to_string()]).map(|_| ()).map_err(|e| (500, e))
    }
    fn free_video(&self, gpu: usize, for_id: &str) -> Result<(), (u16, String)> {
        call_for(&self.0.target("video"), "POST", "/engine/unload", Some(&json!({"gpu": gpu})), 30).map_err(|e| (502, format!("the video daemon: {e}")))?;
        let (rid, log) = self.0.new_run(format!("video: GPU {gpu}'s idle worker let go, for {for_id}"));
        let _ = std::fs::write(log, "");
        self.0.finish(rid, 0);
        Ok(())
    }
    fn free_chat(&self, chat: &str) -> Result<(), (u16, String)> {
        self.0.unload(chat)
    }
    fn try_lock_kind<'a>(&'a self, kind: &str) -> Option<Box<dyn Guard + 'a>> {
        self.0.kind_lock.get(kind)?.try_lock().ok().map(|g| Box::new(g) as Box<dyn Guard + 'a>)
    }
    fn wait_lock_kind<'a>(&'a self, kind: &str) -> Box<dyn Guard + 'a> {
        Box::new(self.0.kind_lock[kind].lock().unwrap())
    }
    fn placed(&self, port: u16) {
        *self.0.inflight.lock().unwrap().entry(port).or_insert(0) += 1;
    }
    fn wait_loaded(&self, port: u16, id: &str) -> Result<(), (u16, String)> {
        self.0.wait_up(port, id)
    }
    fn note(&self, job: Option<&str>, stage: &str, why: &str) {
        self.0.note(job, stage, why);
    }
    fn cancelled(&self, job: &str) -> bool {
        self.0.job_cancelled(job)
    }
    fn learn_ask(&self, id: &str, err: &str, gpu: usize, want: u64) -> bool {
        let Some((need, had)) = place::asked_mb(err) else { return false };
        let ask = place::ask_mb(need, had, self.free_mb(gpu));
        eprintln!("{id}: its server asked {need} MiB with {had} MiB free on GPU {gpu}; planned {want} MiB, from now {ask}");
        let mut a = self.0.asked.lock().unwrap();
        let more = ask > want && a.get(id).is_none_or(|x| ask > *x);
        if more {
            a.insert(id.to_string(), ask);
        }
        more
    }
    fn barred(&self) -> Option<String> {
        self.0.cards_taken()
    }
    fn not_ready(&self, id: &str) -> Option<String> {
        self.0.moving_why(id)
    }
    fn give_up(&self) -> Option<(u16, String)> {
        self.0.past_limit()
    }
    fn ram_plan(&self, snap: &Snapshot, m: &ModelInfo) -> Result<Vec<Holder>, String> {
        self.0.ram_room(snap, &m.id, &m.kind, None)
    }
    fn wait_ram(&self, id: &str) -> bool {
        self.0.wait_ram(id)
    }
    fn free_mb(&self, g: usize) -> u64 {
        // the sampler's cards, no server asked: this is asked every second while memory lets go
        let cards = self.0.cards.lock().unwrap().clone();
        self.0.free_mb(&cards, g)
    }
    fn reserve(&self, m: &ModelInfo, _slot: usize, port: u16, gpu: usize, want: u64) -> Result<(), String> {
        // one at a time over every kind: the check and the taking are one step
        let _a = self.0.admit.lock().unwrap();
        let snap = self.0.snapshot_of(&self.0.models(), false);
        room::may_start(&snap.cards, &snap.models, m, gpu, want)?;
        self.0.pending.lock().unwrap().push(super::Pending { gpu, port, mb: want.saturating_sub(MARGIN_MB), answered: None, kind: m.kind.clone(), model: m.id.clone() });
        Ok(())
    }
    fn first_listed_card(&self) -> bool {
        // the order a model's cards are ticked in is the order to try them (the B70 before the B65 on hardac1)
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backplane_scheduler::profile::Mode;
    use backplane_scheduler::room::Free;

    #[test]
    fn exclusive_mode_reads_from_an_entry_and_the_default_is_shared() {
        assert_eq!(Profile::of_entry(&json!({"id": "x"})).coexistence.mode, Mode::Shared);
        let m = json!({"coexistence": {"mode": "exclusive", "exclusive_reason": "x"}});
        assert_eq!(Profile::of_entry(&m).coexistence.mode, Mode::Exclusive);
    }

    fn chat_holder(mb: u64) -> Holder {
        Holder { what: Free::Chat("m".into()), mb, model: "m".into(), order: 2 }
    }

    #[test]
    fn chat_hold_tells_idle_from_held_and_a_silent_source_is_held() {
        let h = |a, c, k, s| chat_hold_of("m".into(), a, c, k, s);
        assert_eq!(h(Some(false), false, false, Some(false)), ChatHold::Idle("m".into()));
        for held in [h(Some(true), false, false, Some(false)), h(Some(false), true, false, Some(false)), h(Some(false), false, true, Some(false)),
                     h(Some(false), false, false, Some(true)), h(None, false, false, Some(false)), h(Some(false), false, false, None)] {
            assert_eq!(held, ChatHold::Held(Some("m".into())));
        }
    }

    #[test]
    fn an_idle_chat_model_goes_for_a_worker_and_a_kept_or_answering_one_stays() {
        assert!(chat_may_go(false, false, false, false));
        assert!(!chat_may_go(true, false, false, false), "running work never goes");
        assert!(!chat_may_go(false, true, false, false), "changing");
        assert!(!chat_may_go(false, false, true, false), "X-Keep-Up-S through the front door");
        assert!(!chat_may_go(false, false, false, true), "X-Keep-Up-S through the switcher");
    }

    #[test]
    fn a_worker_waiting_on_any_card_is_seen_shared_or_not() {
        let st = json!({"gpus": [
            {"gpu": 0, "shared": false, "engine": "waiting for the GPU: 1.6 GiB free, 27.5 GiB needed"},
            {"gpu": 1, "shared": true, "engine": "waiting for the GPU: 2.0 GiB free, 8.0 GiB needed"},
        ]});
        assert_eq!(waiting_workers(&st), vec![(0, 28160), (1, 8192)]);
        let st = json!({"gpus": [{"gpu": 0, "engine": "loaded"}, {"gpu": 1, "engine": "waiting for the GPU lock (held: x)"}, {"gpu": 2, "engine": "unloaded"}]});
        assert!(waiting_workers(&st).is_empty(), "the lock's wait and a loaded worker are not this");
    }

    #[test]
    fn room_comes_from_idle_holders_or_there_is_none() {
        assert_eq!(room::plan_room(30000, 28000, vec![chat_holder(30000)]).map(|p| p.len()), Some(0), "room without stopping anyone");
        assert_eq!(room::plan_room(1600, 28000, vec![chat_holder(30000)]).map(|p| p.len()), Some(1), "the idle chat model goes");
        assert!(room::plan_room(1600, 28000, vec![]).is_none(), "kept or answering: not in the list, so no room");
        assert!(room::plan_room(1600, 28000, vec![chat_holder(100)]).is_none(), "not enough even with it");
    }

    #[test]
    fn host_memory_counts_the_warm_cache_as_taken() {
        // MemAvailable counts a loaded model's warm cache; a new load must not rely on taking it
        assert_eq!(host_free_mb(16400, 12700), 3700);
        assert_eq!(host_free_mb(1000, 12700), 0);
        // a 42.5 GiB model with the margin does not fit beside it; with the holder's memory back it does
        let want = (42.5 * 1024.0) as u64 + RAM_MARGIN_MB;
        assert!(host_free_mb(16400, 12700) < want);
        assert_eq!(room::plan_room(3700, want, vec![chat_holder(43520 + 12700)]).map(|p| p.len()), Some(1));
    }

    #[test]
    fn a_recogniser_transcribes_and_speaks_nothing() {
        let (caps, methods) = capabilities(&json!({"id": "whisper-large-v3-turbo", "kind": "audio", "arch": "whisper"}));
        assert_eq!(methods, vec!["audio.transcribe"]);
        assert!(caps.contains(&"word_timestamps") && !caps.contains(&"speech"));
        let (_, methods) = capabilities(&json!({"id": "voxcpm2", "kind": "audio", "arch": "voxcpm2", "speech": ["clone"]}));
        assert!(methods.contains(&"audio.speak") && !methods.contains(&"audio.transcribe"));
    }

    #[test]
    fn a_load_under_way_counts_what_it_has_yet_to_take() {
        let c = json!({"vram_total_mb": 32768, "vram_used_mb": 8192,
                       "procs": [{"port": "8086", "vram_mb": 6000}, {"port": "8087", "vram_mb": 2192}]});
        assert_eq!(free_on(&c, &[]), 24576);
        // an audio server loading on :8097 for 10 GiB, nothing of it on the card yet
        assert_eq!(free_on(&c, &[(8097, 10240)]), 24576 - 10240);
        // the image server on :8086 placed for 20000, 6000 of it taken so far
        assert_eq!(free_on(&c, &[(8086, 20000)]), 24576 - 14000);
        // one that took more than placed for owes nothing
        assert_eq!(free_on(&c, &[(8087, 1000)]), 24576);
        // never below nothing
        assert_eq!(free_on(&c, &[(8097, 40000)]), 0);
    }

    #[test]
    fn a_chat_model_split_over_every_card_has_no_choice_of_cards() {
        let cards = vec![
            json!({"id": "glm-5.3-flash-uncensored", "kind": "llm", "gpus": "all"}),
            json!({"id": "qwen3.8-27b-q8", "kind": "llm", "gpus": "0"}),
            json!({"id": "breeze-tts-2", "kind": "audio", "gpus": "all"}),
        ];
        // the registry entry may say one card (set before): the model card decides
        assert!(spans_all(&json!({"id": "glm-5.3-flash-uncensored", "kind": "llm", "gpus": "0"}), &cards));
        // a variant registered under its own id, its card named by `catalog`
        assert!(spans_all(&json!({"id": "glm-128k", "catalog": "glm-5.3-flash-uncensored", "kind": "llm"}), &cards));
        assert!(!spans_all(&json!({"id": "qwen3.8-27b-q8", "kind": "llm"}), &cards), "a one-card chat model");
        assert!(!spans_all(&json!({"id": "breeze-tts-2", "kind": "audio"}), &cards), "an audio model's all is where its copies may go");
        assert!(!spans_all(&json!({"id": "my-own", "kind": "llm", "gpus": "all"}), &cards), "no model card: a choice");
    }
}
