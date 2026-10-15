//! What the enabled models do: their state, the memory they are vouched for, enabling and disabling, loading on
//! demand and unloading.
//!
//! - chat (llm): the video studio's llm.mode loads one at a time (it knows when a render holds the card); the switcher
//!   loads on Open WebUI's request and unloads after the idle time; enabled = listed by the switcher. A box with
//!   neither (the front door alone): this page starts the chat server itself (`nextsycl llm start ID`) for the model
//!   a request names, stopping the one before it.
//! - image, audio: a server each (`nextsycl image|audio start ID --gpu N --idle-exit S`), started on the first
//!   request for an enabled model on the first of its cards with room, swapped when a request names another (after the
//!   one running), ending itself when idle. A second (`--slot 1`, NS_IMAGE_PORT2 / NS_AUDIO_PORT2) holds another model,
//!   or a second copy of a model enabled on two cards: started when a request finds its copy with work and its other
//!   card with room; requests go to the copy without work, else the one with fewer. A start waits for the card to have the model's memory free (1.5 GiB to spare: past that the xe driver
//!   spills into host memory and the box can hang), and answers 503 no_room when it does not come free.
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

/// Whether something listens on 127.0.0.1:`port`
fn listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok()
}

/// An image or audio model: it loads whole on one card (the first of its cards with room)
fn one_card(m: &Value) -> bool {
    matches!(registry::kind_of(m), "image" | "audio")
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

    /// Where a chat request goes: the switcher when one runs (it loads the model the request names), else the chat
    /// server, `model` (or the one loaded, or the first enabled) loaded first
    pub(crate) fn chat_target(&self, model: Option<&str>) -> Result<Target, (u16, String)> {
        if listening(self.o.ports.switch) {
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

    /// The kind's servers' ports, the first's first (image and audio have a second when its port is set)
    pub(crate) fn slots(&self, kind: &str) -> Vec<u16> {
        let p = &self.o.ports;
        match kind {
            "image" => [p.image, p.image2].into_iter().filter(|x| *x != 0).collect(),
            "audio" => [p.audio, p.audio2].into_iter().filter(|x| *x != 0).collect(),
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
        for kind in ["image", "audio"] {
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
        let video = if need("video") { self.video_daemon() } else { None };
        // the model cards, read once: for the headroom of entries registered before cards carried it
        let mc = if all.iter().all(|m| m.get("headroom_gib").is_some()) { Vec::new() } else { self.o.cfg.cards() };
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
                "image" | "audio" => {
                    let mine: Vec<&Slot> = if kind == "image" { &image } else { &audio }.iter().filter(|x| x.model.as_deref() == Some(id.as_str())).collect();
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
            json!({"id": id, "kind": kind, "title": m["title"], "arch": m["arch"], "enabled": enabled, "gpus": gpus, "state": state,
                   "vouch_mb": vouch.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "measured": measured,
                   "headroom_mb": headroom.iter().map(|(g, mb)| (g.to_string(), json!(mb))).collect::<serde_json::Map<String, Value>>(),
                   "headroom_given": headroom_given,
                   "page": match kind.as_str() { "image" | "audio" => json!(if kind == "image" { &image } else { &audio }.iter()
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
        let old = self.gpus_of(&m);
        let mut g = g;
        g.sort();
        g.dedup();
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
            "image" | "audio" => {
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

    /// The image or audio server holding `id`, started (or swapped to it) when it does not: where to send its requests
    pub(crate) fn ensure(&self, id: &str) -> Result<Lease<'_>, (u16, String)> {
        let m = self.entry(id)?;
        let kind = registry::kind_of(&m);
        if !matches!(kind, "image" | "audio") {
            return Err((400, format!("{id} is a {kind} model")));
        }
        if m["enabled"] == false {
            return Err((409, format!("{id} is disabled (enable it on :8000 or nextsycl models enable {id})")));
        }
        let gpus = self.gpus_of(&m);
        let _g = self.kind_lock[kind].lock().unwrap();
        let sv = self.servers(kind);
        let inflight = |p: u16| self.inflight.lock().unwrap().get(&p).copied().unwrap_or(0);
        let mine: Vec<&Slot> = sv.iter().filter(|x| x.model.as_deref() == Some(id)).collect();
        // a server of it with nothing to do
        if let Some(x) = mine.iter().find(|x| !x.busy && inflight(x.port) == 0) {
            if x.loading {
                self.wait_up(x.port, id)?;
            }
            return Ok(self.lease(x.port));
        }
        let card_mb = |c: &Value| c["vram_total_mb"].as_u64().unwrap_or(0);
        let (need, _) = self.vouched_mb(&m, self.cards.lock().unwrap().get(gpus[0]).map_or(32768, card_mb));
        let want = |g: usize| need.get(&g).copied().unwrap_or(0) + MARGIN_MB;
        let free = |cards: &Value, g: usize| cards.get(g).map_or(0, |c| card_mb(c).saturating_sub(c["vram_used_mb"].as_u64().unwrap_or(0)));
        let cards = self.named_cards(&self.models());
        if !mine.is_empty() {
            // its servers all have work: another copy on a card of its that has none of it and the room now, in a slot no
            // server holds (nothing is waited for or stopped for it) - else the copy with the fewest requests
            let on: Vec<usize> = mine.iter().filter_map(|x| card_of_port(&cards, x.port)).collect();
            let gpu = gpus.iter().copied().find(|g| !on.contains(g) && free(&cards, *g) >= want(*g));
            if let (Some(gpu), Some(x)) = (gpu, sv.iter().find(|x| x.model.is_none())) {
                match self.start_at(kind, id, x, gpu) {
                    Ok(()) => return Ok(self.lease(x.port)),
                    Err((_, e)) => eprintln!("{id}: a second copy did not start ({e}); the requests share the first"),
                }
            }
            let x = mine.iter().min_by_key(|x| inflight(x.port)).unwrap();
            if x.loading {
                self.wait_up(x.port, id)?;
            }
            return Ok(self.lease(x.port));
        }
        // not loaded: a slot no server holds when one of its cards has the room now; else a server of another model on
        // one of its cards, stopped after its requests (idle ones first); else a free slot, waiting for the room; else the
        // first slot's server, stopped after its requests
        let room = gpus.iter().any(|g| free(&cards, *g) >= want(*g));
        let empty = sv.iter().find(|x| x.model.is_none());
        let on_mine = |x: &&Slot| x.model.is_some() && card_of_port(&cards, x.port).is_some_and(|g| gpus.contains(&g));
        let swap = sv.iter().filter(on_mine).min_by_key(|x| (x.busy || inflight(x.port) > 0) as u8);
        let x = match (room, empty, swap) {
            (true, Some(e), _) => e,
            (_, _, Some(s)) => s,
            (_, Some(e), None) => e,
            _ => &sv[0],
        };
        if let Some(other) = &x.model {
            // another model holds the server: its requests finish first
            let t0 = Instant::now();
            while self.server_at(x.port).is_some_and(|(_, _, busy)| busy) || inflight(x.port) > 0 {
                if t0.elapsed() > Duration::from_secs(1800) {
                    return Err((503, format!("{other} has been busy for 30 minutes; {id} waits for it")));
                }
                std::thread::sleep(Duration::from_secs(2));
            }
            self.exec(format!("{kind} stop ({other}, for {id})"), &[kind.into(), "stop".into(), "--slot".into(), x.slot.to_string()]).map_err(|e| (500, e))?;
        }
        // the card: the first of its cards with the model's memory free, with room to spare
        let t0 = Instant::now();
        let gpu = loop {
            let cards = self.named_cards(&self.models());
            if let Some(g) = gpus.iter().copied().find(|g| free(&cards, *g) >= want(*g)) {
                break g;
            }
            if t0.elapsed() > Duration::from_secs(90) {
                let busy: Vec<String> = gpus.iter().map(|g| {
                    let holders: Vec<String> = cards.get(*g).and_then(|c| c["procs"].as_array()).into_iter().flatten()
                        .map(|p| format!("{} {:.1} GiB", p["model"].as_str().or(p["program"].as_str()).unwrap_or("?"), p["vram_mb"].as_u64().unwrap_or(0) as f64 / 1024.0))
                        .collect();
                    format!("GPU {g}: {:.1} GiB free ({})", free(&cards, *g) as f64 / 1024.0, holders.join(", "))
                }).collect();
                return Err((503, format!("{} busy - {}; {id} needs ~{:.1} GiB; it loads once one idles out or is stopped",
                                         if gpus.len() > 1 { "its cards are" } else { "its card is" }, busy.join("; "),
                                         want(gpus[0]) as f64 / 1024.0)));
            }
            std::thread::sleep(Duration::from_secs(3));
        };
        self.start_at(kind, id, x, gpu)?;
        Ok(self.lease(x.port))
    }

    /// `id` started in slot `x` on card `gpu`, answering
    fn start_at(&self, kind: &str, id: &str, x: &Slot, gpu: usize) -> Result<(), (u16, String)> {
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
