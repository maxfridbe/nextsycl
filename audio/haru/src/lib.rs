//! Haru-4B on Intel Arc (SYCL): a full-duplex spoken dialogue model (github.com/higurashi-ai/Haru, Apache-2.0 code and
//! weights). It listens and speaks at the same time, in lockstep at 7.5 Hz: every 133 ms one frame of the user's
//! audio in, one frame of its own audio and text out - silence included, so back-channels, interruptions and
//! talking over each other are the model's own doing, not a turn-taking rule's.
//!
//! ```text
//!   the user's frame (3,200 samples, 24 kHz) -> the codec's encoder -> a 64-value latent (drawn around its mean)
//!   position t = embed(the model's text token t) + model_proj(its latent t) + user_proj(the user's latent t)
//!     -> Qwen3.5-4B (24 Gated DeltaNet + 8 attention layers, lm.rs) -> the hidden state
//!     -> the tied output embedding: text token t + 1 (sampled: temperature 0.7, top-k 25, DRY)
//!     -> the MeanFlow head (one step, head.rs): latent t + 1 -> the codec's decoder -> the model's next frame
//!   the prompt: the character card (a ChatML system turn of fixed layout, card.rs), then the voice reference's
//!   latents on the model's stream (3 s - 3 min of one speaker)
//! ```
//!
//! The conversation runs through the audio contract's `Conversation` (the front door's /v1/realtime WebSocket, OpenAI
//! Realtime's events with the lockstep extension: duplex.rs); `generate` answers a recording offline (the release's
//! haru-inference: the model's track as long as the user's).

pub mod card;
pub mod check;
pub mod codec;
pub mod duplex;
mod ffi;
pub mod head;
pub mod lm;
pub mod ops;
pub mod resample;
pub mod sample;
pub mod session;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Conversation, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Realtime,
                     RealtimeSession, Result, Step};
use nextsycl_core::Gpu;
use nextsycl_tok::Tokenizer;

use crate::codec::Coder;
use crate::head::Head;
use crate::lm::Lm;
use crate::ops::Ops;
use crate::session::Settings;
use crate::weights::{Config, Weights};

pub const ARCH: &str = "haru";
/// model: the checkpoint's model.safetensors; config and tokenizer: its config.json and tokenizer.json
pub const ROLES: &[&str] = &["model", "config", "tokenizer"];
/// positions a conversation holds by default: an hour at 7.5 Hz and a prompt with three minutes of voice
pub const CTX: usize = 28672;
const GIB: f64 = (1u64 << 30) as f64;
/// what every card keeps free (the xe driver spills to host memory past it)
const KEEP_FREE: u64 = 3 << 29;
pub const RATE: u32 = 24000;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "ctx", env: "NS_HARU_CTX", value: "28672", help: "positions a conversation holds (7.5 a second, plus the prompt: 28672 is about an hour)",
                   at: At::Load },
    EngineOption { name: "guidance", env: "NS_HARU_GUIDANCE", value: "1.5", help: "the audio head's guidance (1 - 2): higher keeps the voice closer to the reference",
                   at: At::Both },
    EngineOption { name: "top-k", env: "NS_HARU_TOP_K", value: "25", help: "the text stream's top-k (0: all)", at: At::Both },
    EngineOption { name: "eop-bias", env: "NS_HARU_EOP_BIAS", value: "0", help: "added to the start-speaking logit: above 0 it starts speaking sooner",
                   at: At::Both },
    EngineOption { name: "speak-first", env: "NS_HARU_SPEAK_FIRST", value: "0|1", help: "the model opens the conversation", at: At::Both },
    EngineOption { name: "dry", env: "NS_HARU_DRY", value: "1|0", help: "the DRY repetition penalty on its words", at: At::Both },
    EngineOption { name: "stereo", env: "NS_HARU_STEREO", value: "0|1", help: "offline (a source recording): the user on the left channel, the model on the right",
                   at: At::Request },
    EngineOption { name: "profile", env: "NS_HARU_PROFILE", value: "0|1", help: "each frame's operations timed one by one (slower), the table on stderr", at: At::Load },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Haru-4B (full-duplex spoken dialogue in lockstep at 7.5 Hz: a frame of your audio in, a frame of its own out; /v1/realtime)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    Ok(Box::new(Haru::load(files, gpus, o, log)?))
}

/// Everything on the card: the language model, the head, the codec, and what a frame needs besides
pub struct Model {
    pub cfg: Config,
    pub ops: Ops,
    pub lm: Lm,
    pub head: Head,
    pub enc: Coder,
    pub dec: Coder,
    pub tok: Tokenizer,
    pub latent_scale: f32,
    pub latent_bias: f32,
    /// the encoded latent of digital silence (an audio stream before it has sound)
    pub silence: Vec<f32>,
}

/// The engine's shared part: the model (one frame at a time on the card), the settings
pub struct Inner {
    pub model: Mutex<Model>,
    pub ctx: usize,
    pub defaults: Settings,
    pub gpu: Arc<Gpu>,
}

pub struct Haru {
    pub inner: Arc<Inner>,
    load_s: f64,
    loaded: Instant,
    report: Vec<String>,
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// A request's own option, else the load's setting, else the default
fn given(extra: &nextsycl_audio::Given, o: Option<&LoadOptions>, name: &str, env: &str) -> Option<String> {
    extra.get(name).cloned().or_else(|| o.and_then(|o| o.setting(env)))
}

fn num<T: std::str::FromStr>(v: Option<String>, name: &str, d: T) -> Result<T> {
    match v {
        Some(s) => s.trim().parse().map_err(|_| Error(format!("{name} {s:?}: a number"))),
        None => Ok(d),
    }
}

/// A session's settings from the load's and a request's options
pub fn settings(extra: &nextsycl_audio::Given, o: Option<&LoadOptions>, base: &Settings) -> Result<Settings> {
    let g = |n: &str, e: &str| given(extra, o, n, e);
    let mut s = base.clone();
    s.guidance = num(g("guidance", "NS_HARU_GUIDANCE"), "guidance", s.guidance)?.clamp(1.0, 2.0);
    s.top_k = num(g("top-k", "NS_HARU_TOP_K"), "top-k", s.top_k)?;
    s.eop_bias = num(g("eop-bias", "NS_HARU_EOP_BIAS"), "eop-bias", s.eop_bias)?;
    if let Some(v) = g("speak-first", "NS_HARU_SPEAK_FIRST") {
        s.speak_first = v == "1" || v == "true";
    }
    if let Some(v) = g("dry", "NS_HARU_DRY") {
        s.dry = v != "0" && v != "false";
    }
    Ok(s)
}

impl Haru {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Haru> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let ctx: usize = num(o.setting("NS_HARU_CTX"), "ctx", CTX)?.max(512);
        let profile = o.setting("NS_HARU_PROFILE").is_some_and(|v| v == "1");
        let cfg = Config::read(&role(files, "config")?)?;
        let model_path = role(files, "model")?;
        let f = Weights::open(&model_path)?;
        let file_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        // the weights, one conversation's state and the work buffers (the logits, the codec streams, attention's pieces)
        let need = file_bytes + Lm::state_bytes(&cfg, ctx) as u64 + (1u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need + KEEP_FREE > free {
            return Err(Error::no_room(format!("{ARCH} needs about {:.1} GiB on {} (and 1.5 GiB kept free); {:.1} GiB are free", need as f64 / GIB, gpu.name,
                                               free as f64 / GIB)));
        }
        log(format!("{ARCH}: about {:.1} GiB on {} ({:.1} GiB free)", need as f64 / GIB, gpu.name, free as f64 / GIB));
        let tok = Tokenizer::from_hf_json(&role(files, "tokenizer")?).map_err(Error)?;
        if tok.id("<|im_start|>").is_none() || tok.tokens.len() > cfg.vocab {
            return Err(Error("the tokenizer is not the checkpoint's (Qwen3.5's: <|im_start|>, at most the model's vocabulary)".into()));
        }
        let ops = Ops::new(&gpu, profile)?;
        let lm = Lm::load(&ops, &f, &cfg, log)?;
        let head = Head::load(&ops, &f, &cfg)?;
        let enc = Coder::load(&ops, &f, false, cfg.codec_eps, &cfg.ratios)?;
        let dec = Coder::load(&ops, &f, true, cfg.codec_eps, &cfg.ratios)?;
        if enc.dim_in != 1 || enc.dim_out != cfg.latent || dec.dim_in != cfg.latent || dec.dim_out != 1 {
            return Err(Error(format!("the codec: {} -> {} and {} -> {} channels, 1 -> {} -> 1 expected", enc.dim_in, enc.dim_out, dec.dim_in, dec.dim_out,
                                     cfg.latent)));
        }
        let latent_scale = f.scalar("latent_scale")?;
        let latent_bias = f.scalar("latent_bias")?;
        gpu.sync()?;
        log(format!("{ARCH}: language model {:.2} GiB, head {:.2} GiB, codec {:.2} + {:.2} GiB on {}", lm.bytes() as f64 / GIB, head.bytes() as f64 / GIB,
                    enc.bytes() as f64 / GIB, dec.bytes() as f64 / GIB, gpu.name));
        let mut model = Model { cfg: cfg.clone(), ops, lm, head, enc, dec, tok, latent_scale, latent_bias, silence: Vec::new() };
        // digital silence's latent: the last of 16 frames of zeros
        let s = model.enc.stream(&model.ops, cfg.frame)?;
        let means = model.encode_frames(&s, &vec![0.0; 16 * cfg.frame])?;
        model.silence = means[15 * cfg.latent..].to_vec();
        let defaults = settings(&Default::default(), Some(o), &Settings { guidance: cfg.guidance, ..Default::default() })?;
        let used = gpu.memory().ok().and_then(|(_, f)| f).map(|f| free.saturating_sub(f) as f64 / GIB).unwrap_or(0.0);
        log(format!("{ARCH}: {used:.2} GiB of the card in use after the load (a conversation adds {:.2} GiB)", Lm::state_bytes(&cfg, ctx) as f64 / GIB));
        let report = vec![format!("{ARCH} on {}: bfloat16 weights {:.1} GiB, {used:.1} GiB in use after the load, {ctx} positions a conversation \
                                   ({:.2} GiB)", gpu.name, file_bytes as f64 / GIB, Lm::state_bytes(&cfg, ctx) as f64 / GIB)];
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: ready in {load_s:.0} s"));
        Ok(Haru { inner: Arc::new(Inner { model: Mutex::new(model), ctx, defaults, gpu }), load_s, loaded: Instant::now(), report })
    }

    /// The model's track answering `user` (mono 24 kHz), frame by frame as in a live conversation (haru-inference):
    /// as long as the user's, plus `tail` frames of silence after it
    pub fn answer(&self, card: &str, voice: Option<&[f32]>, user: &[f32], tail: usize, s: Settings, progress: &mut dyn FnMut(Step) -> Result<()>)
                  -> Result<(Vec<f32>, String)> {
        let t0 = Instant::now();
        let m = self.inner.model.lock().map_err(|_| Error("the model's lock".into()))?;
        let fs = m.cfg.frame;
        let frames = user.len().div_ceil(fs) + tail;
        let mut ses = m.session(s, self.inner.ctx.min(frames + 4096))?;
        let mut f = m.start(&mut ses, card, voice)?;
        let mut audio = f.audio.clone();
        let mut text = f.text.clone();
        let mut worst = f.seconds;
        let mut pcm = user.to_vec();
        pcm.resize(frames * fs, 0.0);
        let mut steps = 0f64;
        for (i, chunk) in pcm.chunks_exact(fs).enumerate() {
            if i + 1 >= frames {
                break;
            }
            f = m.step(&mut ses, chunk)?;
            steps += f.seconds;
            worst = worst.max(f.seconds);
            audio.extend(&f.audio);
            text.push_str(&f.text);
            progress(Step { phase: "frames", at: (i + 1) as u32, of: frames as u32, seconds: t0.elapsed().as_secs_f64() })?;
        }
        m.ops.report(frames);
        eprintln!("{ARCH}: {frames} frames in {:.1} s, {:.1} ms a frame (worst {:.1} ms; real time is {:.1} ms)", t0.elapsed().as_secs_f64(),
                  1e3 * steps / (frames - 1).max(1) as f64, 1e3 * worst, 1e3 * fs as f64 / RATE as f64);
        Ok((audio, text.split_whitespace().collect::<Vec<_>>().join(" ")))
    }
}

impl AudioEngine for Haru {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 20.0, max_seconds: 600.0, steps: 0, cfg: 1.5, rate: RATE }
    }
    fn sources(&self) -> bool {
        true
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    /// Offline: the request's `source` is the user's side of a conversation (any rate; none: `seconds` of silence,
    /// and the model opens), `prompt` the character (a card's text, or a persona for the default card), `reference`
    /// the voice; the answer is the model's track as long as the user's (stereo with option stereo=1: the user left)
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let lang = card::language(req.language.as_deref().filter(|l| *l != "auto"));
        let card = card::system_prompt(None, Some(&req.prompt), &lang)?;
        let voice = req.reference.as_ref().map(|r| session::prepare_reference(&resample::resample(&r.samples, r.rate, RATE), 180.0))
            .filter(|v| v.len() >= RATE as usize)
            .ok_or_else(|| Error("haru speaks in a voice it is given: a recording of 3 s to 3 minutes of one speaker (--ref-audio FILE, or a saved voice)".into()))?;
        let mut s = settings(&req.extra, None, &self.inner.defaults)?;
        s.seed = req.seed;
        if let Some(c) = req.cfg.filter(|c| *c > 0.0) {
            s.guidance = c.clamp(1.0, 2.0);
        }
        let user = match &req.source {
            Some(a) => resample::resample(&a.mono(), a.rate, RATE),
            None => {
                s.speak_first = true;
                vec![0.0; (req.seconds.unwrap_or(20.0).clamp(1.0, 600.0) * RATE as f32) as usize]
            }
        };
        let (audio, text) = self.answer(&card, Some(&voice), &user, 0, s, progress)?;
        eprintln!("{ARCH}: said: {text}");
        if req.extra.get("stereo").is_some_and(|v| v == "1") {
            let mut st = Vec::with_capacity(2 * audio.len());
            for (i, a) in audio.iter().enumerate() {
                st.push(user.get(i).copied().unwrap_or(0.0));
                st.push(*a);
            }
            return Ok(Audio { rate: RATE, channels: 2, samples: st });
        }
        Ok(Audio { rate: RATE, channels: 1, samples: audio })
    }
    fn realtime(&self) -> Option<Realtime> {
        Some(Realtime {
            input_rate: RATE,
            output_rate: RATE,
            frame_ms: 133,
            voices: Vec::new(),
            clone: true,
            languages: card::languages(),
            tools: false,
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

/// The checkpoint's directory as downloaded (higurashi-ai/Haru-4B): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (role, name) in [("model", "model.safetensors"), ("config", "config.json"), ("tokenizer", "tokenizer.json")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}
