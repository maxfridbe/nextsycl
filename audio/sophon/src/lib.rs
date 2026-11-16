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
use crate::speak::Talk;

pub const ARCH: &str = "sophon-realtime";
/// thinker-1..N: the thinker's shards; voice-1..N: the release's voices/<name>.pt
pub const ROLES: &[&str] = &["thinker*", "thinker-config", "tokenizer", "talker", "talker-config", "codec", "voice*"];
/// the context a conversation keeps (positions); the turn question's own
pub const CTX: usize = 6144;
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
    EngineOption { name: "ctx", env: "NS_SOPHON_CTX", value: "6144", help: "positions a conversation's context holds", at: At::Load },
    EngineOption { name: "q8-layers", env: "NS_SOPHON_Q8_LAYERS", value: "62,63|none",
                   help: "with bits 4: the brain's layers kept in q8 (where 4-bit codes cost the most; ~0.2 GiB each)", at: At::Load },
    EngineOption { name: "talker-int8", env: "NS_SOPHON_TALKER_INT8", value: "0|1", help: "the talker's matrices in int8 (less memory)", at: At::Load },
    EngineOption { name: "greedy", env: "NS_SOPHON_GREEDY", value: "0|1", help: "the most likely token and code every time (a deterministic run, for checks)",
                   at: At::Both },
    EngineOption { name: "turn-checks", env: "NS_SOPHON_TURN_CHECKS", value: "0.2,0.35,0.5,0.8,1.3,2.0",
                   help: "the pauses (s) at which the brain is asked whether the user has finished", at: At::Both },
    EngineOption { name: "turn-thresholds", env: "NS_SOPHON_TURN_THRESHOLDS", value: "0.95,0.85,0.7,0.5,0.35,0.25",
                   help: "P(finished) that answers at each of those pauses", at: At::Both },
];

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
        let mut report = vec![format!("{ARCH} brain {} on {}, talker on {}", layout.name(),
                                      brain.devs.iter().map(|d| d.ops.gpu.name.clone()).collect::<Vec<_>>().join(" + "), talker_gpu.name)];
        for (g, b) in &need {
            report.push(format!("  planned {:.1} GiB on {}", *b as f64 / GIB, g.name));
        }
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: ready in {load_s:.0} s"));
        Ok(Sophon {
            inner: Arc::new(Inner { mind: Mutex::new(Mind { ear, brain, reply, check }), talk: Mutex::new(talk), tok, greedy, turn_checks, turn_thresholds, voices }),
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
        let s = text::tag_clean(&text::cut_tool(text, last), last);
        let s = s.trim_start();
        if s.is_empty() {
            return Vec::new();
        }
        self.tok.encode(s)
    }
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
            clone_takes_text: false,
            direction: false,
            events: ["[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]", "<laughter>...</laughter>", "<strong>...</strong>"]
                .iter().map(|s| s.to_string()).collect(),
            requires: None,
            enrolled: false,
            speakers: 1,
        })
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
        let xvec = match &req.reference {
            Some(r) => talk.clone_voice(&r.samples, r.rate)?,
            None => self.inner.voice(req.voice.as_deref(), lang)?,
        };
        let ids = self.inner.speakable_ids(&req.prompt, true);
        if ids.is_empty() {
            return Err(Error("the text has nothing to say".into()));
        }
        progress(Step { phase: "tokens", at: 0, of: 1, seconds: 0.0 })?;
        let draw = self.inner.talker_draw(&req.extra);
        let mut next = |_: usize| speak::Feed { ids: ids.clone(), done: true };
        let sp = talk.speak(&xvec, lang, draw, req.seed, &mut next, out, &|| false)?;
        progress(Step { phase: "tokens", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        eprintln!("{ARCH}: {} frames ({:.1} s of speech) in {:.2} s, first audio {:.2} s", sp.frames, sp.samples as f64 / speak::RATE as f64, sp.seconds,
                  sp.first_audio);
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
