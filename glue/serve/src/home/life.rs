//! What the enabled models do: their state, the memory they are vouched for, enabling and disabling, loading on
//! demand and unloading.
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
//! - GPU work this page does not manage (a test, a benchmark) holds the run lock (crate::gpulock, NS_GPU_LOCK): while
//!   it is held no model loads - requests are refused with `cards_taken` (503, retryable), jobs wait, the video queue
//!   pauses (mod.rs video_hold); what is loaded keeps answering.
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
pub(super) const MARGIN_MB: u64 = 1536;

/// How long a request waits for a card to have room before 503 no_room (a job waits longer: JOB_ROOM_WAIT_S, its
/// status saying what for)
const ROOM_WAIT_S: u64 = 90;
pub(super) const JOB_ROOM_WAIT_S: u64 = 1800;


/// Idle work that holds memory on a card and can go for a model that needs the room
pub(super) struct Holder {
    what: Free,
    /// its memory there
    mb: u64,
    model: String,
    /// which go first: picture and sound servers (0), the video worker (1), a chat model (2)
    order: u8,
}

pub(super) enum Free {
    /// an image or audio server: (kind, slot)
    Server(String, usize),
    /// the video daemon's worker on this card
    Video(usize),
    /// the chat model
    Chat(String),
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

/// One of the kind's servers: its slot (`--slot`), port, model (when one answers), and whether it loads or has work
pub(crate) struct Slot {
    slot: usize,
    port: u16,
    model: Option<String>,
    loading: bool,
    busy: bool,
}

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
    fn vouched_mb(&self, m: &Value, card_mb: u64) -> (BTreeMap<usize, u64>, bool) {
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
    fn servers(&self, kind: &str) -> Vec<Slot> {
        self.slots(kind).into_iter().enumerate().map(|(slot, port)| {
            let st = self.server_at(port);
            let sent = self.inflight.lock().unwrap().get(&port).copied().unwrap_or(0);
            Slot { slot, port, model: st.as_ref().map(|x| x.0.clone()), loading: st.as_ref().is_some_and(|x| x.1),
                   busy: st.as_ref().is_some_and(|x| x.2) || sent > 0 }
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
        let image = if need("image") { self.servers("image") } else { Vec::new() };
        let audio = if need("audio") { self.servers("audio") } else { Vec::new() };
        let m3d = if need("model3d") { self.servers("model3d") } else { Vec::new() };
        let slots_of = |k: &str| -> &Vec<Slot> { match k { "image" => &image, "audio" => &audio, _ => &m3d } };
        let video = if need("video") { self.video_daemon() } else { None };
        // the model cards, read once: for the headroom of entries registered before cards carried it
        // and whether a chat model spans every card
        let mc = if all.iter().all(|m| m.get("headroom_gib").is_some()) && !need("llm") { Vec::new() } else { self.o.cfg.cards() };
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
            json!({"id": id, "kind": kind, "title": m["title"], "arch": m["arch"], "enabled": enabled, "gpus": gpus, "state": state,
                   "spans": spans_all(m, &mc),
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

    /// Unload `id` if it is loaded (after the requests it is answering)
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
        let limit = Duration::from_secs(if job.is_some() { JOB_ROOM_WAIT_S } else { ROOM_WAIT_S });
        let mut t0 = Instant::now();
        loop {
            if job.is_some_and(|j| self.job_cancelled(j)) {
                return Err((499, format!("cancelled while it waited for {id}")));
            }
            // its files moving to another drive: it waits for the move like for room
            if let Some(why) = self.moving_why(id) {
                self.note(job, "waiting", &why);
                if t0.elapsed() > limit {
                    return Err((409, why));
                }
                std::thread::sleep(Duration::from_secs(3));
                continue;
            }
            let why = match self.place(&m, id, kind, job)? {
                Ok(lease) => return Ok(lease),
                Err(why) => why,
            };
            // a GPU run holds the cards: a request is refused at once (cards_taken, retryable); a job waits in its queue
            // as long as the run lasts (as in maintenance), its room wait starting when the run ends
            if why.starts_with("cards taken:") {
                if job.is_none() {
                    return Err((503, why));
                }
                self.note(job, "cards_taken", &why);
                std::thread::sleep(Duration::from_secs(3));
                t0 = Instant::now();
                continue;
            }
            self.note(job, "waiting", &why);
            if t0.elapsed() > limit {
                return Err((503, format!("{why}; {id} loads once one of them idles out or is stopped")));
            }
            std::thread::sleep(Duration::from_secs(3));
        }
    }

    /// One try at a server for `id` (the kind's lock held while it lasts): a lease, or why there is none yet (no card
    /// of its has the room, even with the idle models there stopped)
    fn place(&self, m: &Value, id: &str, kind: &str, job: Option<&str>) -> Result<Result<Lease<'_>, String>, (u16, String)> {
        let gpus = self.gpus_of(m);
        let _g = self.kind_lock[kind].lock().unwrap();
        let sv = self.servers(kind);
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let mine: Vec<&Slot> = sv.iter().filter(|x| x.model.as_deref() == Some(id)).collect();
        // a server of it with nothing to do
        if let Some(x) = mine.iter().find(|x| !x.busy && inflight(x.port) == 0) {
            if x.loading {
                self.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
                self.wait_up(x.port, id)?;
            }
            return Ok(Ok(self.lease(x.port)));
        }
        let card_mb = |c: &Value| c["vram_total_mb"].as_u64().unwrap_or(0);
        let (need, _) = self.vouched_mb(m, self.cards.lock().unwrap().get(gpus[0]).map_or(32768, card_mb));
        let want = |g: usize| need.get(&g).copied().unwrap_or(0) + MARGIN_MB;
        let cards = self.named_cards(&self.models());
        // a server that holds another model and could go now
        let idle_other = |x: &&Slot| x.model.is_some() && x.model.as_deref() != Some(id) && !x.busy && !x.loading && inflight(x.port) == 0;
        if !mine.is_empty() {
            // its servers all have work: another copy on a card of its without one that has the room now (or once idle
            // picture and sound servers there stop), in a slot no server holds or one an idle model holds - else the copy
            // with the fewest requests (its server queues it)
            let on: Vec<usize> = mine.iter().filter_map(|x| card_of_port(&cards, x.port)).collect();
            let spare = sv.iter().find(|x| x.model.is_none()).or_else(|| sv.iter().find(idle_other));
            // (no second copy while a GPU run holds the cards)
            let plan = gpus.iter().copied().filter(|g| !on.contains(g)).filter(|_| self.cards_taken().is_none())
                .find_map(|g| self.room_plan(&cards, g, want(g), kind, true).map(|p| (g, p)));
            if let (Some(x), Some((gpu, plan))) = (spare, plan) {
                // (a plan another request's lock stands in the way of: no copy now, the requests share the first)
                let started = self.free(&plan, kind, id).and_then(|_| {
                    self.stop_slot(kind, x, id)?;
                    if !self.wait_room(gpu, want(gpu)) {
                        return Err((503, format!("GPU {gpu} did not come free")));
                    }
                    self.note(job, "loading", &format!("a second copy of {id} loading on GPU {gpu} (server {})", x.slot + 1));
                    self.start_at(kind, id, x, gpu, want(gpu))
                });
                match started {
                    Ok(()) => return Ok(Ok(self.lease(x.port))),
                    Err((_, e)) => eprintln!("{id}: a second copy did not start ({e}); the requests share the first"),
                }
            }
            let x = mine.iter().min_by_key(|x| inflight(x.port)).unwrap();
            if x.loading {
                self.note(job, "loading", &format!("{id} loading (server {})", x.slot + 1));
                self.wait_up(x.port, id)?;
            }
            return Ok(Ok(self.lease(x.port)));
        }
        // not loaded, and a GPU run holds the cards (crate::gpulock): nothing loads beside it
        if let Some(why) = self.cards_taken() {
            return Ok(Err(why));
        }
        // not loaded: the first card of its with the room now, or once idle models there go (picture and sound
        // servers, an idle video worker, a chat model not kept by X-Keep-Up-S); a slot no server holds, else one an
        // idle model holds, else one a busy model of the kind holds, after its requests
        let plan = gpus.iter().copied().find_map(|g| self.room_plan(&cards, g, want(g), kind, false).map(|p| (g, p)));
        let empty = sv.iter().find(|x| x.model.is_none());
        let idle = sv.iter().filter(idle_other).min_by_key(|x| card_of_port(&cards, x.port).is_none_or(|g| Some(g) != plan.as_ref().map(|p| p.0)));
        let on_mine = |x: &&Slot| x.model.is_some() && card_of_port(&cards, x.port).is_some_and(|g| gpus.contains(&g));
        let Some((gpu, plan)) = plan else {
            // no room even then: a busy server of another model of the kind on a card of its goes after its requests
            // (it holds the room), then the next try loads it
            let held = |x: &Slot| card_of_port(&cards, x.port).map(|g| {
                let mb = cards[g]["procs"].as_array().into_iter().flatten().filter(|p| p["port"].as_str() == Some(x.port.to_string().as_str()))
                    .map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
                (g, mb)
            });
            let frees = |x: &&Slot| held(x).is_some_and(|(g, mb)| self.room_plan(&cards, g, want(g).saturating_sub(mb), kind, false).is_some());
            if let Some(x) = sv.iter().filter(on_mine).filter(frees).min_by_key(|x| inflight(x.port)) {
                self.drain(x, kind, id, job)?;
                return Ok(Err(format!("server {} stopped for {id}", x.slot + 1)));
            }
            return Ok(Err(self.no_room(&cards, &gpus, id, want(gpus[0]))));
        };
        let x = match (empty, idle) {
            (Some(e), _) => e,
            (None, Some(i)) => i,
            // every server busy with another model: the one with the fewest requests, after them
            _ => {
                let x = sv.iter().filter(|x| x.model.is_some()).min_by_key(|x| (!on_mine(x) as u8, inflight(x.port))).unwrap_or(&sv[0]);
                self.drain(x, kind, id, job)?;
                x
            }
        };
        self.stop_slot(kind, x, id)?;
        // an idle model whose kind is busy changing its own server stays: the next try plans again
        if let Err((_, e)) = self.free(&plan, kind, id) {
            return Ok(Err(e));
        }
        // the memory let go in a moment; when it does not (or something else took it), no load: the next try plans again
        if !self.wait_room(gpu, want(gpu)) {
            return Ok(Err(self.no_room(&self.named_cards(&self.models()), &[gpu], id, want(gpu))));
        }
        self.note(job, "loading", &format!("{id} loading on GPU {gpu} (server {})", x.slot + 1));
        self.start_at(kind, id, x, gpu, want(gpu))?;
        Ok(Ok(self.lease(x.port)))
    }

    /// Server `x` (another model's) after its requests: waited for, with the kind's lock held, so no new request is
    /// sent to it meanwhile; then stopped
    fn drain(&self, x: &Slot, kind: &str, id: &str, job: Option<&str>) -> Result<(), (u16, String)> {
        let Some(other) = &x.model else { return Ok(()) };
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let t0 = Instant::now();
        while self.server_at(x.port).is_some_and(|(_, _, busy)| busy) || inflight(x.port) > 0 {
            if job.is_some_and(|j| self.job_cancelled(j)) {
                return Err((499, format!("cancelled while it waited for {other}'s requests")));
            }
            if t0.elapsed() > Duration::from_secs(JOB_ROOM_WAIT_S) {
                return Err((503, format!("{other} has been busy for 30 minutes; {id} waits for it")));
            }
            self.note(job, "waiting", &format!("server {} finishing {other}'s requests, then {id} loads there", x.slot + 1));
            std::thread::sleep(Duration::from_secs(2));
        }
        self.stop_slot(kind, x, id)
    }

    /// Stop the server in slot `x` when it holds a model (one of the kind, its lock held), for `id`
    fn stop_slot(&self, kind: &str, x: &Slot, id: &str) -> Result<(), (u16, String)> {
        if let Some(other) = self.server_at(x.port).map(|s| s.0) {
            self.exec(format!("{kind} stop ({other}, for {id})"), &[kind.into(), "stop".into(), "--slot".into(), x.slot.to_string()]).map_err(|e| (500, e))?;
        }
        Ok(())
    }

    /// The message for no room: each card, what it has free and who holds the rest (busy or idle)
    pub(super) fn no_room(&self, cards: &Value, gpus: &[usize], id: &str, want: u64) -> String {
        let free = |g: usize| self.free_mb(cards, g);
        let busy: Vec<String> = gpus.iter().map(|g| {
            let mut holders: Vec<String> = cards.get(*g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                .map(|p| format!("{} {:.1} GiB", p["model"].as_str().or(p["program"].as_str()).unwrap_or("?"), p["vram_mb"].as_u64().unwrap_or(0) as f64 / 1024.0))
                .collect();
            // what only the root sampler sees (processes of other users)
            if let Some(mb) = cards.get(*g).and_then(|c| c["vram_unseen_mb"].as_u64()).filter(|mb| *mb >= 256) {
                holders.push(format!("processes of other users {:.1} GiB", mb as f64 / 1024.0));
            }
            format!("GPU {g}: {:.1} GiB free ({})", free(*g) as f64 / 1024.0, holders.join(", "))
        }).collect();
        format!("{} held by work that is not idle - {}; {id} needs ~{:.1} GiB", if gpus.len() > 1 { "its cards are" } else { "its card is" },
                busy.join("; "), want as f64 / 1024.0)
    }

    /// What could go from card `g` so `want` MiB is free there: none needed (Some(empty)), the idle holders to stop
    /// (Some), or no way now (None). Idle: a picture or sound server without work, the video daemon's worker while the
    /// studio has nothing to make, a chat model answering nothing and not kept (chat_idle). `light`: picture and sound servers
    /// only (for a second copy, which is no reason to stop a video or chat model). The kind asking is never asked to
    /// free its own kind's chat or video.
    pub(super) fn room_plan(&self, cards: &Value, g: usize, want: u64, asking: &str, light: bool) -> Option<Vec<Holder>> {
        let c = cards.get(g)?;
        let free = self.free_mb(cards, g);
        if free >= want {
            return Some(Vec::new());
        }
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let mut idle: Vec<Holder> = Vec::new();
        let mut video_checked = None;
        for p in c["procs"].as_array().into_iter().flatten() {
            let mb = p["vram_mb"].as_u64().unwrap_or(0);
            let model = p["model_id"].as_str().or(p["model"].as_str()).unwrap_or("?").to_string();
            match p["kind"].as_str() {
                Some(k @ ("image" | "audio" | "model3d")) => {
                    let Some(port) = p["port"].as_str().and_then(|x| x.parse::<u16>().ok()) else { continue };
                    let Some(x) = self.servers(k).into_iter().find(|x| x.port == port) else { continue };
                    if !x.busy && !x.loading && inflight(port) == 0 {
                        idle.push(Holder { what: Free::Server(k.to_string(), x.slot), mb, model, order: 0 });
                    }
                }
                Some("video") if !light && asking != "video" => {
                    let ok = *video_checked.get_or_insert_with(|| self.video_idle());
                    if ok {
                        idle.push(Holder { what: Free::Video(g), mb, model, order: 1 });
                    }
                }
                Some("llm") if !light && asking != "llm" => {
                    if let Some(id) = self.chat_idle() {
                        idle.push(Holder { what: Free::Chat(id), mb, model, order: 2 });
                    }
                }
                _ => {}
            }
        }
        // the lightest to bring back first: picture and sound servers, then the video worker, then a chat model
        idle.sort_by_key(|h| (h.order, std::cmp::Reverse(h.mb)));
        let mut got = free;
        let mut plan = Vec::new();
        for h in idle {
            if got >= want {
                break;
            }
            got += h.mb;
            plan.push(h);
        }
        (got >= want).then_some(plan)
    }

    /// Stop the idle holders of a plan (for `id`; the kind `kind` asking holds its own lock; another kind's is tried, and
    /// a server whose lock is taken is left: the next try plans again)
    pub(super) fn free(&self, plan: &[Holder], kind: &str, id: &str) -> Result<(), (u16, String)> {
        for h in plan {
            match &h.what {
                Free::Server(k, slot) => {
                    let guard = if k == kind { None } else {
                        match self.kind_lock[k.as_str()].try_lock() {
                            Ok(g) => Some(g),
                            Err(_) => return Err((503, format!("{} is changing its {k} server; {id} tries again", h.model))),
                        }
                    };
                    // still idle (it may have taken a request since)
                    let x = self.servers(k).into_iter().find(|x| x.slot == *slot);
                    if x.as_ref().is_some_and(|x| !x.busy && !x.loading && x.model.is_some()) {
                        self.exec(format!("{k} stop ({}, idle, for {id})", h.model), &[k.clone(), "stop".into(), "--slot".into(), slot.to_string()]).map_err(|e| (500, e))?;
                    }
                    drop(guard);
                }
                Free::Video(g) => {
                    if self.video_idle() {
                        call_for(&self.target("video"), "POST", "/engine/unload", Some(&json!({"gpu": g})), 30).map_err(|e| (502, format!("the video daemon: {e}")))?;
                        let (rid, log) = self.new_run(format!("video: GPU {g}'s idle worker let go, for {id}"));
                        let _ = std::fs::write(log, "");
                        self.finish(rid, 0);
                    }
                }
                Free::Chat(chat) => {
                    if self.chat_idle().as_deref() == Some(chat.as_str()) {
                        self.unload(chat)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Wait (30 s at most) until card `g` has `want` MiB free: a stopped process lets go of its memory in a moment.
    /// False: it did not come free (the load must not start - past the card the xe driver spills into host memory)
    pub(super) fn wait_room(&self, g: usize, want: u64) -> bool {
        let t0 = Instant::now();
        loop {
            let cards = self.cards.lock().unwrap().clone();
            if self.free_mb(&cards, g) >= want {
                return true;
            }
            if t0.elapsed() >= Duration::from_secs(30) {
                return false;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    /// What card `g` has free now, MiB: its memory less what is in use - by every process, ours or not (the page's own
    /// sampler, or the root sampler's gpustat file when that sees more: mod.rs sample_cards) - and less what the
    /// loads started here have yet to take
    pub(crate) fn free_mb(&self, cards: &Value, g: usize) -> u64 {
        let Some(c) = cards.get(g) else { return 0 };
        let pending: Vec<(u16, u64)> = {
            let mut l = self.pending.lock().unwrap();
            l.retain(|p| p.3.is_none_or(|t| t.elapsed() < Duration::from_secs(6)));
            l.iter().filter(|p| p.0 == g).map(|p| (p.1, p.2)).collect()
        };
        free_on(c, &pending)
    }

    /// Whether the studio has nothing to make: no clip running, none runnable in the queue
    fn video_idle(&self) -> bool {
        call_for(&self.target("video"), "POST", "/rpc/status", Some(&json!({})), 4).ok().map(|v| v.get("result").cloned().unwrap_or(v))
            .is_some_and(|st| st["idle"] == true && st["queue_items"].as_array().is_none_or(|a| a.iter().all(|i| i["held"] == true)))
    }

    /// The chat model loaded, when it may be preempted: answering nothing, not changing, and not kept - a request
    /// that sent X-Keep-Up-S (Open WebUI's connection does: 300 s) keeps it that long after it ended, here or through
    /// the switcher (its GET /keep); a request without the header keeps nothing
    fn chat_idle(&self) -> Option<String> {
        let id = self.chat_loaded(&self.models())?;
        let v = call_for(&Target::Unix(self.chat_socket()), "GET", "/server/requests", None, 3).ok()?;
        if v["active"].as_array().is_some_and(|a| !a.is_empty()) || self.chat_mode().1.is_some() {
            return None;
        }
        if self.chat_keep.lock().unwrap().get(&id).is_some_and(|until| *until > now()) {
            return None;
        }
        if listening(self.o.ports.switch) {
            // the switcher not answering: kept, to be safe
            let k = call_for(&Target::Tcp(format!("127.0.0.1:{}", self.o.ports.switch)), "GET", "/keep", None, 3).ok()?;
            if k["kept"][id.as_str()].as_f64().is_some_and(|s| s > 0.0) {
                return None;
            }
        }
        Some(id)
    }

    /// A chat request's keep (X-Keep-Up-S / keep_up_s): `id` kept that long from now
    pub(crate) fn keep_chat(&self, id: &str, secs: u64) {
        self.chat_keep.lock().unwrap().insert(id.to_string(), now() + secs as f64);
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
        // a GPU run holds the cards (crate::gpulock): no chat model loads beside it
        if let Some(why) = self.cards_taken() {
            return Err((503, why));
        }
        let gpus = self.gpus_of(m);
        let t0 = Instant::now();
        loop {
            let cards = self.named_cards(&all);
            let card_mb = |g: usize| cards.get(g).and_then(|c| c["vram_total_mb"].as_u64()).unwrap_or(32768);
            let (need, _) = self.vouched_mb(m, card_mb(gpus[0]));
            // what the chat model loaded now holds on a card (it leaves)
            let chat_on = |g: usize| cards.get(g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                .filter(|p| p["kind"] == "llm").map(|p| p["vram_mb"].as_u64().unwrap_or(0)).sum::<u64>();
            let want = |g: usize| (need.get(&g).copied().unwrap_or(0) + MARGIN_MB).saturating_sub(chat_on(g));
            let free = |g: usize| self.free_mb(&cards, g);
            let short: Vec<usize> = gpus.iter().copied().filter(|g| free(*g) < want(*g)).collect();
            if short.is_empty() {
                return Ok(());
            }
            let mut freed = false;
            for g in &short {
                if let Some(plan) = self.room_plan(&cards, *g, want(*g), "llm", false) {
                    if self.free(&plan, "llm", id).is_ok() {
                        self.wait_room(*g, want(*g));
                        freed = true;
                    }
                }
            }
            if t0.elapsed() > Duration::from_secs(ROOM_WAIT_S) {
                let lack = short.iter().map(|g| self.no_room(&cards, &[*g], id, want(*g))).collect::<Vec<_>>().join(" / ");
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

    /// `id` started in slot `x` on card `gpu` (placed for `want` MiB, the spare included), answering; until it answers,
    /// the memory it has yet to take counts as taken there (free_mb)
    fn start_at(&self, kind: &str, id: &str, x: &Slot, gpu: usize, want: u64) -> Result<(), (u16, String)> {
        self.pending.lock().unwrap().push((gpu, x.port, want.saturating_sub(MARGIN_MB), None));
        let r = self.start_at_(kind, id, x, gpu);
        // kept a few seconds more: the cards are sampled every 2 s, the server's memory shows on the next
        for p in self.pending.lock().unwrap().iter_mut().filter(|p| p.1 == x.port && p.3.is_none()) {
            p.3 = Some(Instant::now());
        }
        r
    }

    fn start_at_(&self, kind: &str, id: &str, x: &Slot, gpu: usize) -> Result<(), (u16, String)> {
        let secs = (self.o.idle_minutes * 60).to_string();
        let mut args = vec![kind.to_string(), "start".into(), id.into(), "--gpu".into(), gpu.to_string(), "--port".into(), x.port.to_string(),
                            "--host".into(), self.o.serve_host.clone(), "--wfe".into(), "--slot".into(), x.slot.to_string()];
        if self.o.idle_minutes > 0 {
            args.extend(["--idle-exit".into(), secs]);
        }
        self.loading.lock().unwrap().insert(id.to_string(), now());
        let r = self.exec(format!("{kind} start {id} on GPU {gpu}{}", if x.slot > 0 { format!(" (server {})", x.slot + 1) } else { String::new() }), &args);
        self.loading.lock().unwrap().remove(id);
        r.map_err(|e| (502, e))?;
        self.wait_up(x.port, id)
    }

    /// A request on its way to the server on `port` (counted until the lease is dropped)
    fn lease(&self, port: u16) -> Lease<'_> {
        *self.inflight.lock().unwrap().entry(port).or_insert(0) += 1;
        Lease { target: Target::Tcp(format!("127.0.0.1:{port}")), port, home: self }
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
        self.exec(format!("video start ({})", want.join(", ")), &args).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
