//! Sophon-Realtime (Flash) on Intel Arc (SYCL): a full-duplex voice conversation (github.com/adventists-ai/sophon-realtime,
//! Apache-2.0 code and weights). The brain listens to the user's speech itself, decides when they have finished and
//! writes a reply with sound tags; the voice speaks it while it is written.
//!
//! ```text
//!   the user's audio -> server VAD (20 ms energy frames, duplex.rs)
//!   a pause -> the turn question: DuplexJev's one-letter answer over the speech (P(finished) against a threshold
//!              that falls as the pause grows; a reply at 2 s anyway)
//!   finished -> the ear (ear.rs: Whisper features, Qwen3-ASR encoder, projector) -> the brain (brain.rs: Qwen3-32B,
//!              q4 on one card) over system + history (earlier turns as audio) + this turn: the reply, token by token
//!            -> the talker (speak.rs: Qwen3-TTS 1.7B fine-tuned, on the other card) reads it as it comes ->
//!              the 12 Hz codec -> 24 kHz audio, out as it is made
//!   the user speaks over a reply (barge-in) -> it stops; the history keeps what was heard
//! ```
//!
//! The conversation runs through the audio contract's `Conversation` (the front door's /v1/realtime WebSocket, OpenAI
//! Realtime's events); `generate` speaks a text in a Sophon voice (the talker alone).
//!
//! A cloned voice (a saved voice's recording, a request's reference) is its x-vector (upstream's talker.py way), or
//! with NS_SOPHON_CLONE=icl - when its transcript is known - Qwen3-TTS's in-context clone (speak.rs): the recording
//! (its first NS_SOPHON_ICL_MAX_S, cut between two words by the saved voice's word times) and what it says in the
//! talker's prompt. The plain entry's Base talker clones closest that way; the -tags entry's fine-tuned talker ends
//! its speech early with it, so it keeps the x-vector. /v1/audio/speech takes upstream's ICL layout; a live reply
//! NS_SOPHON_ICL_LIVE's (stream: the reply fed a token a frame as it is written, the recording's prompt ahead).

pub mod brain;
pub mod check;
pub mod duplex;
pub mod ear;
mod ffi;
pub mod ops;
pub mod resample;
pub mod sample;
pub mod speak;
pub mod speaker;
pub mod talker;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{shards, At, Audio, AudioEngine, AudioKind, AudioRequest, Conversation, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Realtime,
                     RealtimeSession, Result, Speech, Step};
use nextsycl_core::Gpu;
use nextsycl_tok::Tokenizer;

use crate::brain::{Brain, Layout, Slot};
use crate::ear::Ear;
use crate::ops::Shards;
use crate::sample::Draw;
use crate::speak::{IclLayout, Talk, Voice};

pub const ARCH: &str = "sophon-realtime";
/// thinker-1..N: the thinker's shards; voice-1..N: the release's voices/<name>.pt
pub const ROLES: &[&str] = &["thinker*", "thinker-config", "tokenizer", "talker", "talker-config", "codec", "voice*"];
/// the context a conversation keeps (positions); the turn question's own
pub const CTX: usize = 16384;
const CHECK_CTX: usize = 1024;
/// the brain's layers kept in q8 under q4 by default (NS_SOPHON_Q8_LAYERS)
pub const Q8_LAYERS: &[usize] = &[62, 63];
const GIB: f64 = (1u64 << 30) as f64;
/// what every card keeps free (the xe driver spills to host memory past it)
const KEEP_FREE: u64 = 3 << 29;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "bits", env: "NS_SOPHON_BITS", value: "4|8",
                   help: "the brain's matrices: 4 (q4: ~19.5 GiB, one card) or 8 (q8: ~33 GiB, split over two cards - the close one, for checks)", at: At::Load },
    EngineOption { name: "split", env: "NS_SOPHON_SPLIT", value: "N",
                   help: "with two cards and bits 8: the brain's layers from N on the second card (default 30; the talker then shares the second)", at: At::Load },
    EngineOption { name: "ctx", env: "NS_SOPHON_CTX", value: "16384", help: "positions a conversation's context holds", at: At::Load },
    EngineOption { name: "q8-layers", env: "NS_SOPHON_Q8_LAYERS", value: "62,63|none",
                   help: "with bits 4: the brain's layers kept in q8 (where 4-bit codes cost the most; ~0.2 GiB each)", at: At::Load },
    EngineOption { name: "talker-int8", env: "NS_SOPHON_TALKER_INT8", value: "0|1", help: "the talker's matrices in int8 (less memory)", at: At::Load },
    EngineOption { name: "greedy", env: "NS_SOPHON_GREEDY", value: "0|1", help: "the most likely token and code every time (a deterministic run, for checks)",
                   at: At::Both },
    EngineOption { name: "turn-checks", env: "NS_SOPHON_TURN_CHECKS", value: "0.2,0.35,0.5,0.8,1.3,2.0",
                   help: "the pauses (s) at which the brain is asked whether the user has finished", at: At::Both },
    EngineOption { name: "turn-thresholds", env: "NS_SOPHON_TURN_THRESHOLDS", value: "0.95,0.85,0.7,0.5,0.35,0.25",
                   help: "P(finished) that answers at each of those pauses", at: At::Both },
    EngineOption { name: "clone", env: "NS_SOPHON_CLONE", value: "xvector|icl",
                   help: "a cloned voice: its x-vector, or in context (its recording's frames and transcript in the prompt; x-vector when no transcript is known)",
                   at: At::Both },
    EngineOption { name: "icl-max-s", env: "NS_SOPHON_ICL_MAX_S", value: "16",
                   help: "an in-context clone's recording at most this long (s): a longer one cut between two words (its word times), else its x-vector", at: At::Both },
    EngineOption { name: "icl-live", env: "NS_SOPHON_ICL_LIVE", value: "prompt|stream",
                   help: "a live reply's in-context layout: prompt (upstream's: the talker waits until the reply covers the recording or is whole) or stream (the reply fed as written: experimental, ends early or runs on)",
                   at: At::Both },
];

/// An in-context clone's longest recording (s) by default (Qwen3-TTS: a few seconds to about 15)
pub const ICL_MAX_S: f64 = 16.0;
/// and its shortest after a cut
const ICL_MIN_S: f64 = 3.0;

/// The part of a reference an in-context clone takes, and what it says. With the recogniser's word times (a saved
/// voice Whisper has heard): cut between two words - after the last sentence that ends in the second half of the first
/// `max_s` (of the recording), else after the last whole word there - the words before the cut its transcript, so a
/// recording clipped mid-word (a 15 s or 60 s cut of a longer clip) or a transcript that does not match it exactly
/// is not handed over as it is. Without them: the whole recording and its transcript, when it is at most `max_s`.
/// None: no transcript, too long with no word times to cut by, or under 3 s of words
pub fn icl_part(r: &nextsycl_audio::Reference, max_s: f64) -> Option<(usize, String)> {
    let text = r.text.as_deref().map(str::trim).filter(|t| !t.is_empty())?;
    let rate = r.rate.max(1) as f64;
    let secs = r.samples.len() as f64 / rate;
    let w: Vec<&nextsycl_audio::Timed> = r.words.iter().filter(|w| !w.text.trim().is_empty() && w.end > w.start).collect();
    if w.is_empty() {
        return (secs <= max_s).then(|| (r.samples.len(), text.to_string()));
    }
    let limit = max_s.min(secs);
    let fits: Vec<usize> = (0..w.len()).filter(|&k| w[k].end <= limit).collect();
    let last = *fits.last()?;
    let sentence_end = |t: &str| {
        let t = t.trim_end().trim_end_matches(['"', '\'', '”', '’', ')']);
        !t.ends_with("...") && !t.ends_with('…') && t.ends_with(['.', '!', '?', '。', '！', '？'])
    };
    let sentence = fits.iter().rev().copied().find(|&k| w[k].end >= (limit * 0.5).max(ICL_MIN_S) && sentence_end(&w[k].text));
    let k = sentence.unwrap_or(last);
    // halfway into the gap after the word (at most 0.25 s past it), never past the limit
    let gap_end = w.get(k + 1).map_or(w[k].end + 0.25, |n| n.start);
    let cut = (w[k].end + ((gap_end - w[k].end) / 2.0).clamp(0.0, 0.25)).min(limit);
    if cut < ICL_MIN_S {
        return None;
    }
    // whisper's words carry their leading space, or come bare (joined by one, but before a "-ho" or "'s")
    let spaced = w[1..=k].iter().any(|x| x.text.starts_with(char::is_whitespace));
    let said = if spaced {
        w[..=k].iter().map(|x| x.text.as_str()).collect::<String>()
    } else {
        let mut s = String::new();
        for (i, x) in w[..=k].iter().enumerate() {
            let t = x.text.trim();
            if i > 0 && !t.starts_with(['-', '\'', '’', ',', '.', '!', '?']) {
                s.push(' ');
            }
            s.push_str(t);
        }
        s
    };
    Some((((cut * rate) as usize).min(r.samples.len()), said.trim().to_string()))
}

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Sophon-Realtime Flash (full-duplex voice conversation: a 32B brain that listens and a Qwen3-TTS voice; /v1/realtime)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    Ok(Box::new(Sophon::load(files, gpus, o, log)?))
}

/// The brain's card side: the ear, the brain and its cache slots (one lock: they share the first card's queue)
pub struct Mind {
    pub ear: Ear,
    pub brain: Brain,
    /// the conversation's context, and the turn question's
    pub reply: Slot,
    pub check: Slot,
}

pub struct Inner {
    pub mind: Mutex<Mind>,
    /// the voice (None: loaded for a check of the brain alone)
    pub talk: Mutex<Option<Talk>>,
    pub tok: Tokenizer,
    pub greedy: bool,
    pub turn_checks: Vec<f32>,
    pub turn_thresholds: Vec<f32>,
    /// voices: name -> its x-vector file
    pub voices: Vec<(String, PathBuf)>,
    /// a cloned voice in context (NS_SOPHON_CLONE=icl), its recording at most this long, a live reply's layout
    pub clone_icl: bool,
    pub icl_max_s: f64,
    pub icl_live: IclLayout,
    /// the conversation's context (positions: NS_SOPHON_CTX)
    pub ctx: usize,
}

pub struct Sophon {
    pub inner: Arc<Inner>,
    load_s: f64,
    loaded: Instant,
    report: Vec<String>,
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// NS_SOPHON_CLONE / the clone option: whether a cloned voice is in context (default: its x-vector)
fn clone_mode(v: Option<&str>) -> Result<bool> {
    match v.map(str::trim) {
        None | Some("" | "xvector") => Ok(false),
        Some("icl") => Ok(true),
        Some(x) => Err(Error(format!("clone {x:?}: xvector or icl"))),
    }
}

/// NS_SOPHON_ICL_MAX_S / the icl-max-s option
fn icl_max(v: Option<&str>) -> Result<f64> {
    match v.map(str::trim).filter(|x| !x.is_empty()) {
        None => Ok(ICL_MAX_S),
        Some(x) => x.parse::<f64>().ok().filter(|s| *s >= ICL_MIN_S && *s <= 60.0).ok_or_else(|| Error(format!("icl-max-s {x:?}: seconds, 3 to 60"))),
    }
}

fn floats(s: &str) -> Result<Vec<f32>> {
    s.split(',').map(|x| x.trim().parse::<f32>().map_err(|_| Error(format!("{s:?}: numbers, comma-separated")))).collect()
}

/// A request's own option (`--opt-NAME`, an API request's `options`), else the load's setting
fn given<'a>(extra: &'a nextsycl_audio::Given, name: &str) -> Option<&'a str> {
    extra.get(name).map(String::as_str)
}

impl Sophon {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Sophon> {
        Self::load_with(files, gpus, o, true, log)
    }

    /// `load`; `talker` false: the brain and its ear only (a check of them: the talker's card and memory left alone)
    pub fn load_with(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, talker: bool, log: &mut dyn FnMut(String)) -> Result<Sophon> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let first = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let layout = Layout::parse(&o.setting("NS_SOPHON_BITS").unwrap_or_else(|| "4".into()))?;
        let ctx: usize = o.setting("NS_SOPHON_CTX").map(|v| v.parse().map_err(|_| Error(format!("ctx {v:?}: a number")))).transpose()?.unwrap_or(CTX);
        let talker_int8 = o.setting("NS_SOPHON_TALKER_INT8").is_some_and(|v| v == "1");
        let thinker = shards(files, "thinker");
        if thinker.is_empty() {
            return Err(Error(format!("{ARCH}: no thinker shards (thinker-1, thinker-2 ...)")));
        }
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(role(files, "thinker-config")?).map_err(|e| Error(e.to_string()))?)
            .map_err(|e| Error(format!("the thinker's config.json: {e}")))?;
        let tc = &cfg["text_config"];
        let theta = tc["rope_parameters"]["rope_theta"].as_f64().or_else(|| tc["rope_theta"].as_f64()).unwrap_or(5e6);
        let sh = Arc::new(Shards::open(&thinker)?);
        // where things go: the brain on the first card (q8: its layers split over two), the talker on the last
        let two = gpus.len() >= 2;
        let split: usize = o.setting("NS_SOPHON_SPLIT").and_then(|v| v.parse().ok()).unwrap_or(30);
        // layers kept in q8 under q4 (the first and last: where 4-bit codes cost the most)
        let q8_layers: Vec<usize> = match o.setting("NS_SOPHON_Q8_LAYERS") {
            Some(v) if !v.trim().is_empty() && v != "none" => v.split(',').map(|x| x.trim().parse::<usize>().map_err(|_| Error(format!("q8-layers {v:?}: layer numbers"))))
                .collect::<Result<_>>()?,
            Some(_) => Vec::new(),
            None => Q8_LAYERS.to_vec(),
        };
        let (n_layer, plan) = Brain::plan(&sh, layout, if layout == Layout::Q8 && two { split } else { usize::MAX }, &q8_layers)?;
        if layout == Layout::Q8 && !two {
            return Err(Error(format!("bits 8 needs two cards (~{:.0} GiB of matrices); load it with gpus '0,1', or bits 4", plan[0] as f64 / GIB)));
        }
        let talker_gpu = gpus.last().cloned().unwrap_or_else(|| first.clone());
        let talker_file = role(files, "talker")?;
        let talker_bytes = std::fs::metadata(&talker_file).map(|m| m.len()).unwrap_or(0) / if talker_int8 { 2 } else { 1 }
            + std::fs::metadata(role(files, "codec")?).map(|m| m.len()).unwrap_or(0) + (1u64 << 30);
        let ear_bytes = 900u64 << 20;
        let kv_per_pos = (n_layer * 2 * 8 * brain::HEAD * 2) as u64;
        // a card's work buffers: a prompt chunk's activations (~0.6 GiB at 512 rows) and the largest matrix expanded
        // to half (0.5 GiB)
        let work = 1600u64 << 20;
        // memory on each card, checked before anything loads
        let mut need: Vec<(Arc<Gpu>, u64)> = Vec::new();
        let mut add = |g: &Arc<Gpu>, b: u64| match need.iter_mut().find(|(x, _)| Arc::ptr_eq(x, g)) {
            Some(e) => e.1 += b,
            None => need.push((g.clone(), b)),
        };
        let kv = kv_per_pos * (ctx + CHECK_CTX) as u64;
        if plan.len() == 1 {
            add(&first, plan[0] + ear_bytes + kv + work);
        } else {
            let frac = split as u64;
            add(&first, plan[0] + ear_bytes + kv * frac / n_layer as u64 + work);
            add(&gpus[1], plan[1] + kv * (n_layer as u64 - frac) / n_layer as u64 + work);
        }
        if talker {
            add(&talker_gpu, talker_bytes);
        }
        for (g, b) in &need {
            let (total, free) = g.memory()?;
            let free = free.unwrap_or(total);
            if b + KEEP_FREE > free {
                return Err(Error::no_room(format!("{ARCH} needs about {:.1} GiB on {} (and 1.5 GiB kept free); {:.1} GiB are free", *b as f64 / GIB, g.name,
                                                   free as f64 / GIB)));
            }
            log(format!("{ARCH}: about {:.1} GiB on {} ({:.1} GiB free)", *b as f64 / GIB, g.name, free as f64 / GIB));
        }
        let tok = Tokenizer::from_hf_json(&role(files, "tokenizer")?).map_err(Error)?;
        if tok.id("<|audio|>") != Some(151669) || tok.id("<|im_end|>") != Some(151645) {
            return Err(Error("the tokenizer is not the thinker's (<|audio|> at 151669, <|im_end|> at 151645)".into()));
        }
        let ear = Ear::load(&sh, &first)?;
        log(format!("{ARCH}: the ear (Qwen3-ASR encoder, {} layers of {}) on {}", 18, ear.d, first.name));
        let brain = Brain::load(sh.clone(), if plan.len() == 1 { &gpus[..1] } else { &gpus[..2] }, layout, split, &q8_layers, theta, log)?;
        let reply = brain.slot(ctx)?;
        let check = brain.slot(CHECK_CTX)?;
        // the prompt products' and the ear's shapes made now, not in a conversation's first turns
        let tw = Instant::now();
        brain.warm(&check)?;
        ear.warm(20)?;
        log(format!("{ARCH}: the prompt shapes made ready in {:.1} s", tw.elapsed().as_secs_f64()));
        let mut voices: Vec<(String, PathBuf)> = shards(files, "voice").into_iter()
            .filter_map(|p| Some((p.file_stem()?.to_str()?.to_string(), p.clone()))).collect();
        voices.sort();
        let voices_dir = voices.first().and_then(|(_, p)| p.parent().map(Path::to_path_buf)).unwrap_or_default();
        let talk = if talker {
            Some(Talk::load(&talker_file, &role(files, "talker-config")?, &role(files, "codec")?, &voices_dir, &talker_gpu, talker_int8, log)?)
        } else {
            None
        };
        let greedy = o.setting("NS_SOPHON_GREEDY").is_some_and(|v| v == "1");
        let turn_checks = floats(&o.setting("NS_SOPHON_TURN_CHECKS").unwrap_or_else(|| "0.2,0.35,0.5,0.8,1.3,2.0".into()))?;
        let turn_thresholds = floats(&o.setting("NS_SOPHON_TURN_THRESHOLDS").unwrap_or_else(|| "0.95,0.85,0.7,0.5,0.35,0.25".into()))?;
        let clone_icl = clone_mode(o.setting("NS_SOPHON_CLONE").as_deref())?;
        let icl_max_s = icl_max(o.setting("NS_SOPHON_ICL_MAX_S").as_deref())?;
        let icl_live = IclLayout::parse(&o.setting("NS_SOPHON_ICL_LIVE").unwrap_or_else(|| "prompt".into()))?;
        // the codec's encoder now (an in-context clone's frames), not at the first cloned voice
        if let (true, Some(t)) = (clone_icl, &talk) {
            t.encoder()?;
        }
        let mut report = vec![format!("{ARCH} brain {} on {}, talker on {}", layout.name(),
                                      brain.devs.iter().map(|d| d.ops.gpu.name.clone()).collect::<Vec<_>>().join(" + "), talker_gpu.name),
                              format!("  a cloned voice {}", if clone_icl {
                                  format!("in context (its recording up to {icl_max_s} s and transcript; live replies {icl_live:?})")
                              } else {
                                  "by its x-vector".to_string()
                              })];
        for (g, b) in &need {
            report.push(format!("  planned {:.1} GiB on {}", *b as f64 / GIB, g.name));
        }
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: ready in {load_s:.0} s"));
        Ok(Sophon {
            inner: Arc::new(Inner { mind: Mutex::new(Mind { ear, brain, reply, check }), talk: Mutex::new(talk), tok, greedy, turn_checks, turn_thresholds, voices,
                                    clone_icl, icl_max_s, icl_live, ctx }),
            load_s,
            loaded: Instant::now(),
            report,
        })
    }
}

impl Inner {
    /// A voice's x-vector: a release voice by name (`<name>_<lang>`, then `<name>`), else the default (chelsie)
    pub fn voice(&self, name: Option<&str>, lang: &str) -> Result<Vec<f32>> {
        let want = name.unwrap_or("chelsie").to_lowercase();
        let find = |n: &str| self.voices.iter().find(|(k, _)| k == n).map(|(_, p)| p.clone());
        // its own language's take, else its plain one, else any language's (a voice speaks across languages)
        let hit = find(&format!("{want}_{lang}")).or_else(|| find(&want))
            .or_else(|| self.voices.iter().find(|(k, _)| k.strip_prefix(want.as_str()).is_some_and(|r| r.starts_with('_'))).map(|(_, p)| p.clone()));
        match hit {
            Some(p) => speak::read_pt_xvector(&p),
            None => Err(Error(format!("voice {want:?}: one of {} (or a saved voice's recording)", self.voice_names().join(", ")))),
        }
    }

    /// The voice a request or session speaks in: a release voice by name (its x-vector); a reference recording in
    /// context when that is asked for (`extra`'s clone, else NS_SOPHON_CLONE) and its transcript is known, else its
    /// x-vector. The talker's lock held by the caller
    pub fn speaker(&self, talk: &Talk, reference: Option<&nextsycl_audio::Reference>, name: Option<&str>, lang: &str, extra: &nextsycl_audio::Given)
                   -> Result<Voice> {
        let Some(r) = reference else {
            return Ok(Voice::XVector(self.voice(name, lang)?));
        };
        let icl = match given(extra, "clone") {
            Some(v) => clone_mode(Some(v))?,
            None => self.clone_icl,
        };
        let max_s = match given(extra, "icl-max-s") {
            Some(v) => icl_max(Some(v))?,
            None => self.icl_max_s,
        };
        if icl {
            match icl_part(r, max_s) {
                Some((n, text)) => {
                    let ids = icl_ids(&self.tok, &text);
                    let t = Instant::now();
                    let i = talk.icl(&r.samples[..n], r.rate, &text, &ids)?;
                    eprintln!("{ARCH}: an in-context clone of {:.1} s ({} frames, {} transcript tokens{}): {:?}", i.seconds, i.codes.len(), i.ids.len(),
                              if t.elapsed().as_secs_f64() < i.made_s / 2.0 { ", kept from before".to_string() } else { format!(", made in {:.2} s", i.made_s) },
                              text.chars().take(60).collect::<String>());
                    return Ok(Voice::InContext(i));
                }
                None => eprintln!("{ARCH}: a clone by its x-vector ({})", if r.text.as_deref().is_none_or(|t| t.trim().is_empty()) {
                    "no transcript known".to_string()
                } else {
                    format!("{:.1} s, over {max_s} s, and no word times to cut it by", r.samples.len() as f64 / r.rate.max(1) as f64)
                }),
            }
        }
        Ok(Voice::XVector(talk.clone_voice(&r.samples, r.rate)?))
    }

    /// A live reply's in-context layout (`extra`'s icl-live, else NS_SOPHON_ICL_LIVE)
    pub fn live_layout(&self, extra: &nextsycl_audio::Given) -> Result<IclLayout> {
        given(extra, "icl-live").map_or(Ok(self.icl_live), IclLayout::parse)
    }

    pub fn voice_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.voices.iter().map(|(n, _)| n.trim_end_matches("_en").trim_end_matches("_zh").to_string()).collect();
        v.dedup();
        // the release's default first (a session that names none speaks it)
        v.sort_by_key(|n| (n != "chelsie", n.clone()));
        v
    }

    /// How the brain draws its tokens (generation_config.json: 0.7, top-k 20, top-p 0.8; greedy for checks)
    pub fn brain_draw(&self, extra: &nextsycl_audio::Given, temperature: Option<f32>) -> Draw {
        let greedy = given(extra, "greedy").map_or(self.greedy, |v| v == "1");
        if greedy {
            return Draw::GREEDY;
        }
        Draw { sample: true, temperature: temperature.unwrap_or(0.7), top_k: 20, top_p: 0.8 }
    }

    pub fn talker_draw(&self, extra: &nextsycl_audio::Given) -> Draw {
        if given(extra, "greedy").map_or(self.greedy, |v| v == "1") {
            Draw::GREEDY
        } else {
            Draw::DEFAULT
        }
    }

    /// Speakable text's ids: the reply as the talker reads it, tokenized (the thinker's tokenizer)
    pub fn speakable_ids(&self, text: &str, last: bool) -> Vec<u32> {
        speakable_ids(&self.tok, text, last, tags_spoken())
    }
}

/// Whether the talker is handed the sound tags, as upstream hands them (NS_SOPHON_TAGS; 0: taken out, see
/// `speakable_ids`). Flash's own talker performs them; Qwen3-TTS-12Hz-1.7B-Base - the checkpoint pull once linked in
/// its place, the same name and size - reads them aloud
pub fn tags_spoken() -> bool {
    std::env::var("NS_SOPHON_TAGS").map_or(true, |v| v.trim() != "0")
}

/// An in-context clone's transcript as the talker reads it: qwen-tts's ref_ids - the reference template
/// "<|im_start|>assistant\n{text}<|im_end|>\n" (NFC) tokenized, the role's three ids and the last two off
pub fn icl_ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    use unicode_normalization::UnicodeNormalization;
    let t = format!("<|im_start|>assistant\n{}<|im_end|>\n", text.trim()).nfc().collect::<String>();
    let ids = tok.encode(&t);
    if ids.len() < 6 {
        return Vec::new();
    }
    ids[3..ids.len() - 2].to_vec()
}

/// Speakable text's ids (proc.py's thinker2talker_async_chunk): the text before a tool call, tag_clean (the sound
/// tags kept), the thinker's tokenizer without special tokens; `tags` false: the tags taken out too (text::without_tags)
pub fn speakable_ids(tok: &Tokenizer, text: &str, last: bool, tags: bool) -> Vec<u32> {
    let s = text::tag_clean(&text::cut_tool(text, last), last);
    let s = if tags { s } else { text::without_tags(&s) };
    let s = s.trim_start();
    if s.is_empty() {
        return Vec::new();
    }
    tok.encode(s)
}

impl AudioEngine for Sophon {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: 190.0, steps: 0, cfg: 1.0, rate: speak::RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: self.inner.voice_names(),
            languages: ["en", "zh", "de", "fr", "es", "it", "pt", "ja", "ru"].iter().map(|s| s.to_string()).collect(),
            instructions: false,
            design: false,
            clone: true,
            clone_needs_text: false,
            // in context (NS_SOPHON_CLONE=icl) a transcript makes the closer clone
            clone_takes_text: self.inner.clone_icl,
            direction: false,
            // none when the talker is not handed them (NS_SOPHON_TAGS=0: the plain entry's Base talker, which reads them)
            events: if tags_spoken() {
                ["[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]", "<laughter>...</laughter>", "<strong>...</strong>"]
                    .iter().map(|s| s.to_string()).collect()
            } else {
                Vec::new()
            },
            requires: None,
            enrolled: false,
            speakers: 1,
        })
    }
    fn markup(&self) -> Vec<String> {
        text::KEEP.iter().map(|k| k.to_string()).collect()
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    /// The talker alone: the request's text spoken in a Sophon voice (its sound tags performed)
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let mut samples = Vec::new();
        self.speak_text(req, progress, &mut |s| {
            samples.extend_from_slice(s);
            Ok(())
        })?;
        Ok(Audio { rate: speak::RATE, channels: 1, samples })
    }
    fn streams(&self) -> bool {
        true
    }
    fn generate_streamed(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, chunk: &mut dyn FnMut(&[f32]) -> Result<()>)
                         -> Result<Audio> {
        let mut samples = Vec::new();
        self.speak_text(req, progress, &mut |s| {
            samples.extend_from_slice(s);
            chunk(s)
        })?;
        Ok(Audio { rate: speak::RATE, channels: 1, samples })
    }
    fn realtime(&self) -> Option<Realtime> {
        Some(Realtime {
            input_rate: ear::RATE,
            output_rate: speak::RATE,
            frame_ms: 0,
            voices: self.inner.voice_names(),
            clone: true,
            languages: ["en", "zh", "de", "fr", "es", "it", "pt", "ja", "ru"].iter().map(|s| s.to_string()).collect(),
            tools: true,
        })
    }
    fn converse(&self, s: &RealtimeSession) -> Result<Box<dyn Conversation>> {
        Ok(Box::new(duplex::Duplex::start(self.inner.clone(), s)?))
    }
    fn report(&self) -> Vec<String> {
        let mut r = self.report.clone();
        r.push(format!("  loaded {:.0} s ago", self.loaded.elapsed().as_secs_f64()));
        r
    }
}

impl Sophon {
    fn speak_text(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, out: &mut dyn FnMut(&[f32]) -> Result<()>) -> Result<()> {
        let t0 = Instant::now();
        let lang = req.language.as_deref().filter(|l| !l.is_empty() && *l != "auto").unwrap_or("en").to_lowercase();
        let lang = &lang[..2.min(lang.len())];
        let guard = self.inner.talk.lock().unwrap_or_else(|e| e.into_inner());
        let talk = guard.as_ref().ok_or_else(|| Error("loaded without its talker".into()))?;
        let voice = self.inner.speaker(talk, req.reference.as_ref(), req.voice.as_deref(), lang, &req.extra)?;
        let ids = self.inner.speakable_ids(&req.prompt, true);
        if ids.is_empty() {
            return Err(Error("the text has nothing to say".into()));
        }
        progress(Step { phase: "tokens", at: 0, of: 1, seconds: 0.0 })?;
        let draw = self.inner.talker_draw(&req.extra);
        let mut next = |_: usize| speak::Feed { ids: ids.clone(), done: true };
        // the whole text in hand: upstream's in-context layout
        let sp = talk.speak(&voice, IclLayout::Prompt, lang, draw, req.seed, &mut next, out, &|| false)?;
        progress(Step { phase: "tokens", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        eprintln!("{ARCH}: {} frames ({:.1} s of speech) in {:.2} s, first audio {:.2} s{}", sp.frames, sp.samples as f64 / speak::RATE as f64, sp.seconds,
                  sp.first_audio, if voice.in_context() { " (in context)" } else { "" });
        Ok(())
    }
}

/// Floats as their bytes
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A release's directory as downloaded (Sophon-Realtime-32B-Flash: thinker/, talker/, voices/): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("thinker/model.safetensors.index.json"))
        .map_err(|e| Error(format!("{}: no thinker/model.safetensors.index.json ({e})", dir.display())))?)
        .map_err(|e| Error(format!("the thinker's index: {e}")))?;
    // the language model's and audio tower's shards (the vision tower's own file, visual.safetensors, is not read)
    let mut shards: Vec<String> = idx["weight_map"].as_object().into_iter().flatten().filter_map(|(_, v)| v.as_str().map(str::to_string))
        .filter(|f| f.starts_with("model")).collect();
    shards.sort();
    shards.dedup();
    for (i, s) in shards.iter().enumerate() {
        f.insert(format!("thinker-{}", i + 1), dir.join("thinker").join(s));
    }
    for (role, name) in [("thinker-config", "thinker/config.json"), ("tokenizer", "thinker/tokenizer.json"), ("talker", "talker/model.safetensors"),
                         ("talker-config", "talker/config.json"), ("codec", "talker/speech_tokenizer/model.safetensors")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    let mut voices: Vec<PathBuf> = std::fs::read_dir(dir.join("voices")).into_iter().flatten().flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "pt")).collect();
    voices.sort();
    for (i, p) in voices.into_iter().enumerate() {
        f.insert(format!("voice-{}", i + 1), p);
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextsycl_audio::{Reference, Timed};

    fn w(start: f64, end: f64, text: &str) -> Timed {
        Timed { start, end, text: text.into(), probability: None }
    }

    #[test]
    fn an_in_context_clone_takes_a_cut_between_words() {
        let rate = 1000;
        let r = |secs: f64, text: Option<&str>, words: Vec<Timed>| Reference { samples: vec![0.0; (secs * rate as f64) as usize], rate,
                                                                              text: text.map(str::to_string), words };
        // short enough and no word times: all of it, its text
        assert_eq!(icl_part(&r(10.0, Some(" Hello there. "), vec![]), 16.0), Some((10_000, "Hello there.".into())));
        // short enough with word times: the clipped last word left off, an ellipsis is no sentence's end, the words' text
        let clipped = vec![w(0.5, 2.0, "Hidey"), w(2.0, 2.3, "-ho"), w(2.3, 4.0, "there."), w(4.2, 6.0, "As"), w(6.1, 7.0, "I..."), w(7.2, 9.0, "it."),
                           w(9.5, 9.95, "Since...")];
        assert_eq!(icl_part(&r(10.0, Some("Heidi, hold there."), clipped.clone()), 16.0), Some((9_250, "Hidey-ho there. As I... it.".into())));
        assert_eq!(icl_part(&r(10.0, Some("x"), clipped), 6.5), Some((4_100, "Hidey-ho there.".into())));
        // no transcript: none (its x-vector)
        assert_eq!(icl_part(&r(10.0, None, vec![]), 16.0), None);
        assert_eq!(icl_part(&r(10.0, Some("  "), vec![]), 16.0), None);
        // too long and no word times: none
        assert_eq!(icl_part(&r(40.0, Some("Long."), vec![]), 16.0), None);
        // too long: cut after the last sentence that ends in the second half, halfway into the gap after it
        let words = vec![w(0.5, 2.0, " One"), w(2.1, 9.0, " two."), w(9.4, 12.0, " Three"), w(12.1, 15.8, " four"), w(15.9, 17.0, " five.")];
        assert_eq!(icl_part(&r(40.0, Some("One two. Three four five. Six."), words.clone()), 16.0), Some((9_200, "One two.".into())));
        // no sentence end late enough: after the last word that fits
        assert_eq!(icl_part(&r(40.0, Some("x"), words), 10.0), Some((9_200, "One two.".into())));
        let plain = vec![w(0.5, 4.0, "alpha"), w(4.2, 8.0, "beta"), w(8.2, 12.0, "gamma")];
        assert_eq!(icl_part(&r(40.0, Some("x"), plain), 10.0), Some((8_100, "alpha beta".into())));
        // under 3 s of words before the limit: none
        assert_eq!(icl_part(&r(40.0, Some("x"), vec![w(0.2, 2.0, " Hi."), w(2.5, 30.0, " long")]), 16.0), None);
    }

    #[test]
    fn clone_settings() {
        assert!(!clone_mode(None).unwrap() && !clone_mode(Some("xvector")).unwrap() && clone_mode(Some("icl")).unwrap());
        assert!(clone_mode(Some("both")).is_err());
        assert_eq!(icl_max(None).unwrap(), ICL_MAX_S);
        assert_eq!(icl_max(Some("12")).unwrap(), 12.0);
        assert!(icl_max(Some("1")).is_err() && icl_max(Some("x")).is_err());
        assert_eq!(IclLayout::parse("stream").unwrap(), IclLayout::Stream);
        assert!(IclLayout::parse("x").is_err());
    }
}
