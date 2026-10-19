//! Qwen3-TTS (12 Hz, 1.7B) on Intel Arc (SYCL): speech from text in ten languages - in one of nine built-in voices
//! styled by an instruction (CustomVoice), in a voice made up from a description (VoiceDesign), or in the voice of a
//! recording (Base: its x-vector, and with its transcript its codec frames as an example to continue).
//!
//! ```text
//!   text (+ instruction, voice, language) -> the talker's prompt (prompt.rs): projected text tokens + codec tokens
//!   -> the talker (talker.rs, a 28-layer Qwen3 of 2,048) draws a frame's first code, 12.5 frames a second; its code
//!      predictor (5 layers of 1,024) the other fifteen; the frame fed back with the next text token
//!   -> the 12 Hz codec's decoder (nextsycl-codec12hz): 16 codes a frame -> 1,920 samples at 24 kHz
//! ```
//!
//! The weights are the checkpoint's own files (`nextsycl models pull qwen3-tts-custom` ...): the model's bf16
//! safetensors (half on the GPU, the language models' matrices int8 with `--opt-int8`), the speech tokenizer's
//! (float32), the tokenizer's vocab.json / merges.txt. `check` compares the stages with the reference's dumps
//! (reference/qwen3tts/ref.py: qwen-tts' own code on the same files).

pub mod check;
pub mod device;
mod ffi;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod speaker;
pub mod talker;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;

pub use nextsycl_codec12hz as codec;
use nextsycl_codec12hz::{Codec, Encoder};

use crate::ops::{Ops, Shards};
use crate::prompt::{Config, Spec, Voice};
use crate::sample::{Draw, Rng};
use crate::speaker::Speaker;
use crate::talker::{bytes, Drawing, Talker};

pub const ARCH: &str = "qwen3-tts";
pub const ROLES: &[&str] = &["model", "config", "vocab", "merges", "tokenizer-config", "codec"];
/// frames a second, and the longest speech made (the checkpoint's max_new_tokens)
pub const FPS: f32 = 12.5;
const MAX_FRAMES: usize = 8192;
/// a streamed answer's first sound after this many frames (0.64 s of speech; NS_Q3T_STREAM_FIRST)
const STREAM_FIRST: usize = 8;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_Q3T_INT8", value: "0|1", help: "the talker's and code predictor's matrices in int8 (less VRAM, faster frames)", at: At::Load },
    EngineOption { name: "greedy", env: "NS_Q3T_GREEDY", value: "0|1", help: "always the most likely code, no repetition penalty (a deterministic read; flatter)",
                   at: At::Request },
    EngineOption { name: "temperature", env: "NS_Q3T_TEMPERATURE", value: "0.9", help: "the talker's sampling temperature", at: At::Request },
    EngineOption { name: "top-k", env: "NS_Q3T_TOP_K", value: "50", help: "the talker's top-k", at: At::Request },
    EngineOption { name: "streaming", env: "NS_Q3T_STREAMING", value: "0|1",
                   help: "the text fed a token a frame (a clone's default) rather than all ahead (the built-in and designed voices' default)", at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Qwen3-TTS 12 Hz 1.7B (speech: built-in voices, voices designed from a description, voices cloned from a recording)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Qwen3Tts::load(files, gpu, o, log)?))
}

/// Shown the frames' codes so far after each frame (a streamed answer's codec)
pub type CodesSink<'a> = Option<&'a mut dyn FnMut(&[Vec<i32>]) -> Result<()>>;
/// Handed the samples as they are made (a streamed answer)
type Samples<'a> = Option<&'a mut dyn FnMut(&[f32]) -> Result<()>>;

pub struct Qwen3Tts {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub conf: Config,
    pub talker: Talker,
    pub codec: Codec,
    pub speaker: Option<Speaker>,
    /// the codec's encoder (the Base checkpoint: a recording's codes for the in-context clone)
    pub encoder: Option<Encoder>,
    /// the model's tensors (its text embedding is read from here, a row a token)
    pub weights: Shards,
    int8: bool,
    loaded: Instant,
    load_s: f64,
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// A request's own option (`--opt-NAME`, an API request's `options`)
fn given<'a>(req: &'a AudioRequest, name: &str) -> Option<&'a str> {
    req.extra.get(name).map(String::as_str)
}

impl Qwen3Tts {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Qwen3Tts> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let int8 = o.setting("NS_Q3T_INT8").is_some_and(|v| v == "1");
        let (model, codec) = (role(files, "model")?, role(files, "codec")?);
        let vocab = role(files, "vocab")?;
        let dir = vocab.parent().unwrap_or(Path::new("."));
        // the tokenizer's three files sit together (as the checkpoint has them)
        for r in ["merges", "tokenizer-config"] {
            let p = role(files, r)?;
            if p.parent() != Some(dir) {
                return Err(Error(format!("{}: the tokenizer's files are expected beside {}", p.display(), vocab.display())));
            }
        }
        let conf = Config::read(&role(files, "config")?)?;
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        // the bf16 file less its text embedding (left on disk), half or int8 on the GPU; the codec in float32; 1.5 GiB free
        let lm = size(&model).saturating_sub(151_936 * 2048 * 2);
        let need = lm / if int8 { 2 } else { 1 } + size(&codec) + (2u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_vocab_merges(dir).map_err(Error)?;
        if tok.id("<|im_start|>") != Some(151644) {
            return Err(Error(format!("{}: not Qwen3-TTS's tokenizer (no <|im_start|> at 151644)", dir.display())));
        }
        let weights = Shards::open(&[model])?;
        let mut talker = Talker::load(&ops, &weights, int8, log)?;
        let silo = device::silo(&talker)?;
        if silo.is_some() {
            talker.ptab = Some(talker.project_tables(&ops, &nsd)?);
        }
        log(format!("{ARCH}: the frames {}", if silo.is_some() { "on the device (the silo's kernels, a graph a frame)" } else { "on the shared kernels, the draws on the host" }));
        let speaker = if weights.find("speaker_encoder.fc.weight").is_ok() { Some(Speaker::load(&weights)?) } else { None };
        let codec_f = codec::Weights::open(&codec)?;
        let codec = Codec::load(gpu, &codec_f)?;
        let encoder = if speaker.is_some() { Some(Encoder::load(gpu, &codec_f)?) } else { None };
        // the decoder's two shortest chunk lengths made once now (their primitives), so a short answer (up to 8 s)
        // does not wait for them
        let tw = Instant::now();
        codec.decode_bucketed(&nsd, &[vec![0; 16]])?;
        codec.decode_bucketed(&nsd, &vec![vec![0; 16]; 60])?;
        log(format!("{ARCH}: the codec's short chunks made ready in {:.1} s", tw.elapsed().as_secs_f64()));
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH} ({}) on {} in {load_s:.0} s", conf.kind, gpu.name));
        Ok(Qwen3Tts { ops, nsd, tok, conf, talker, codec, speaker, encoder, weights, int8, loaded: Instant::now(), load_s })
    }

    /// What the request asks for, checked against what this checkpoint does
    pub fn spec(&self, req: &AudioRequest) -> Result<Spec> {
        let instruct = req.instructions.clone().filter(|i| !i.trim().is_empty());
        let streaming = given(req, "streaming").map(|v| v == "1");
        let voice = match (self.conf.kind.as_str(), &req.reference, &req.voice) {
            ("custom_voice", _, v) => {
                let v = v.clone().filter(|v| !v.is_empty()).ok_or_else(|| {
                    Error(format!("voice: one of {}", self.conf.speakers.keys().cloned().collect::<Vec<_>>().join(", ")))
                })?;
                Voice::Speaker(v)
            }
            ("voice_design", _, _) => {
                if instruct.is_none() {
                    return Err(Error("instructions: the voice to make (\"a warm, low male voice, unhurried\")".into()));
                }
                Voice::Described
            }
            (_, Some(r), _) => {
                let sp = self.speaker.as_ref().ok_or_else(|| Error("this checkpoint has no speaker encoder".into()))?;
                let wav = speaker::resample(&r.samples, r.rate);
                let x = sp.embed(&wav)?;
                match r.text.as_deref().filter(|t| !t.trim().is_empty()) {
                    // with its transcript: the recording's codes as the example the talker continues
                    Some(text) => {
                        let enc = self.encoder.as_ref().ok_or_else(|| Error("this checkpoint has no codec encoder".into()))?;
                        let codes = enc.encode(&self.nsd, &wav)?;
                        Voice::InContext { spk: x, text: text.to_string(), codes }
                    }
                    None => Voice::XVector(x),
                }
            }
            _ => return Err(Error("reference: a recording of the voice to clone (this is the Base model)".into())),
        };
        let clone = matches!(voice, Voice::XVector(_) | Voice::InContext { .. });
        if clone && instruct.is_some() {
            return Err(Error("instructions: not taken by a cloned voice".into()));
        }
        Ok(Spec { text: req.prompt.clone(), language: req.language.clone(), instruct, voice, streaming: streaming.unwrap_or(clone) })
    }

    /// The frames' codes [T][16]: the talker and its predictor from the prompt. `force`: a check's codes to follow
    /// (each frame's taken from it, not drawn); `sink`: shown the codes so far after each frame (a streamed answer)
    #[allow(clippy::too_many_arguments)]
    pub fn frames(&self, s: &Spec, d: &Drawing, max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>,
                  force: Option<&[Vec<i32>]>, keep: Option<&mut Vec<Vec<f32>>>, mut sink: CodesSink)
                  -> Result<Vec<Vec<i32>>> {
        let (ops, nsd, t) = (&self.ops, &self.nsd, &self.talker);
        let bound = prompt::bound(&self.tok, s);
        let tq = Instant::now();
        let ss = t.session(ops, bound, max)?;
        let p = prompt::build(t, ops, nsd, &ss, &self.weights, &self.tok, &self.conf, s)?;
        let h = t.lm.hidden;
        let l = p.rows.len() / h;
        let tb = tq.elapsed().as_secs_f64();
        t.prefill(ops, nsd, &ss, &p.rows)?;
        let tpf = tq.elapsed().as_secs_f64() - tb;
        let tf = Instant::now();
        let mut keep = keep;
        if let Some(k) = keep.as_deref_mut() {
            k.push(p.rows.clone());
            k.push(t.logits0(ops, &ss)?);
        }
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        if let Some(k) = device::silo(t)?.filter(|_| t.ptab.is_some()) {
            let feed = [p.trailing.as_slice(), p.pad.as_slice()].concat();
            let mut codes: Vec<Vec<i32>> = Vec::new();
            let m = t.frames_device(k, ops, &ss, l, &feed, d, max, seed, force, &mut |c| {
                codes.push(c);
                if let Some(f) = sink.as_deref_mut() {
                    f(&codes)?;
                }
                progress(Step { phase: "tokens", at: codes.len() as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })
            })?;
            if let Some(k) = keep {
                k.push(vec![m[0] as f32, m[1] as f32]);
            }
            if codes.is_empty() {
                return Err(Error("the model ended the speech before its first frame".into()));
            }
            return Ok(codes);
        }
        // the rows fed a frame each, then tts_pad's
        let n_tr = p.trailing.len() / h;
        let feed = DevBuf::from_f32(&ops.gpu, &[p.trailing.as_slice(), p.pad.as_slice()].concat())?;
        let mut rng = Rng::new(seed);
        let mut codes: Vec<Vec<i32>> = Vec::new();
        let mut seen = Vec::new();
        for f in 0..max {
            let text = ops::fp(&feed, f.min(n_tr) * h);
            // a forced run ends where its codes do
            let forced = force.map(|fc| fc.get(f).cloned().unwrap_or_else(|| vec![d.eos; t.groups]));
            match t.frame(ops, nsd, &ss, l + f, text, d, &seen, &mut rng, forced.as_deref())? {
                None => break,
                Some(c) => {
                    seen.push(c[0]);
                    codes.push(c);
                    if let Some(f) = sink.as_deref_mut() {
                        f(&codes)?;
                    }
                }
            }
            progress(Step { phase: "tokens", at: codes.len() as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
        }
        ops.gpu.sync()?;
        if talker::profiling() {
            let pr = ss.prof.get();
            let n = codes.len().max(1) as f64;
            eprintln!("[q3t profile] prompt {l} rows: build {:.1} ms, prefill {:.1} ms; {} frames in {:.1} ms ({:.2} ms a frame): c0 logits {:.3}, c0 draw {:.3}, \
                       predictor {:.3}, talker {:.3} ms a frame", tb * 1e3, tpf * 1e3, codes.len(), tf.elapsed().as_secs_f64() * 1e3,
                      tf.elapsed().as_secs_f64() * 1e3 / n, pr[0] * 1e3 / n, pr[1] * 1e3 / n, pr[2] * 1e3 / n, pr[3] * 1e3 / n);
        }
        if let Some(k) = keep {
            // a forced run's disagreements, for a check
            let m = ss.misses.get();
            k.push(vec![m[0] as f32, m[1] as f32]);
        }
        if codes.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        Ok(codes)
    }

    /// How the request's codes are drawn
    pub fn drawing(&self, req: &AudioRequest) -> Result<Drawing> {
        let greedy = given(req, "greedy").is_some_and(|v| v == "1");
        let mut talker = if greedy { Draw::GREEDY } else { Draw::DEFAULT };
        if let Some(v) = given(req, "temperature") {
            talker.temperature = v.parse().map_err(|_| Error(format!("temperature: {v}?")))?;
        }
        if let Some(v) = given(req, "top-k") {
            talker.top_k = v.parse().map_err(|_| Error(format!("top-k: {v}?")))?;
        }
        Ok(Drawing {
            talker,
            predictor: if greedy { Draw::GREEDY } else { Draw::DEFAULT },
            repetition_penalty: if greedy { 1.0 } else { 1.05 },
            eos: self.conf.eos,
            suppress_from: self.conf.vocab - 1024,
            min_frames: 2,
        })
    }
    /// A request's speech; with `chunk`, handed out as it is made: the codec decodes the frames so far at a few
    /// points (`Codec::stream_due`: after the first frames, then where a chunk reaches one of its lengths) and the
    /// rest at the end, its chunks starting where the whole clip's do - the same samples as the whole clip's, but for
    /// the codec's own rounding (1 in 32,768 on ~0.07% of the samples, as two whole runs of a clip differ)
    fn make(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, mut chunk: Samples)
            -> Result<Audio> {
        let t0 = Instant::now();
        let s = self.spec(req)?;
        if talker::profiling() {
            eprintln!("[q3t profile] request (a clone's speaker embedding and codes) {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let d = self.drawing(req)?;
        let seconds = req.seconds.unwrap_or(self.defaults().seconds);
        if seconds <= 0.0 {
            return Err(Error(format!("{seconds} s: a length above zero")));
        }
        let max = ((seconds * FPS) as usize).clamp(1, MAX_FRAMES);
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        // an in-context clone: the recording's codes go ahead of the new ones through the codec (its context, as
        // qwen-tts decodes them), their samples left out
        let rc: &[Vec<i32>] = match &s.voice {
            Voice::InContext { codes, .. } => codes,
            _ => &[],
        };
        let first = std::env::var("NS_Q3T_STREAM_FIRST").ok().and_then(|v| v.parse().ok()).unwrap_or(STREAM_FIRST);
        let mut samples = Vec::new();
        // frames handed out so far (of the new ones)
        let mut sent = 0;
        let mut td = 0.0;
        let codes = match chunk.as_deref_mut() {
            None => self.frames(&s, &d, max, req.seed, t0, progress, None, None, None)?,
            Some(out) => {
                let mut sink = |codes: &[Vec<i32>]| -> Result<()> {
                    if !Codec::stream_due(codes.len(), sent, first) {
                        return Ok(());
                    }
                    let t = Instant::now();
                    let all: Vec<Vec<i32>> = rc.iter().chain(codes).cloned().collect();
                    let wav = self.codec.decode_from(&self.nsd, &all, rc.len() + sent)?;
                    td += t.elapsed().as_secs_f64();
                    sent = codes.len();
                    out(&wav)?;
                    samples.extend_from_slice(&wav);
                    Ok(())
                };
                self.frames(&s, &d, max, req.seed, t0, progress, None, None, Some(&mut sink))?
            }
        };
        progress(Step { phase: "decode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let t = Instant::now();
        let all: Vec<Vec<i32>> = rc.iter().chain(&codes).cloned().collect();
        let rest = self.codec.decode_from(&self.nsd, &all, rc.len() + sent)?;
        td += t.elapsed().as_secs_f64();
        if let Some(out) = chunk {
            if !rest.is_empty() {
                out(&rest)?;
            }
        }
        samples.extend_from_slice(&rest);
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        if talker::profiling() {
            eprintln!("[q3t profile] codec {:.1} ms", td * 1e3);
        }
        Ok(Audio { rate: codec::RATE, channels: 1, samples })
    }
}

impl AudioEngine for Qwen3Tts {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: 0, cfg: 1.0, rate: codec::RATE }
    }
    fn speech(&self) -> Option<Speech> {
        let mut languages: Vec<String> = self.conf.languages.keys().filter(|l| !l.ends_with("_dialect")).cloned().collect();
        languages.sort();
        let kind = self.conf.kind.as_str();
        Some(Speech {
            voices: self.conf.speakers.keys().cloned().collect(),
            languages,
            instructions: kind != "base",
            design: kind == "voice_design",
            clone: kind == "base" && self.speaker.is_some(),
            clone_needs_text: false,
            clone_takes_text: kind == "base" && self.encoder.is_some(),
            direction: false,
            events: Vec::new(),
            requires: Some(match kind {
                "custom_voice" => "voice",
                "voice_design" => "instructions",
                _ => "reference",
            }),
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
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        self.make(req, progress, None)
    }
    fn streams(&self) -> bool {
        true
    }
    fn generate_streamed(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>, chunk: &mut dyn FnMut(&[f32]) -> Result<()>)
                         -> Result<Audio> {
        self.make(req, progress, Some(chunk))
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} {} on {} ({}, loaded {:.0} s ago)", self.conf.kind, self.ops.gpu.name, if self.int8 { "int8" } else { "half" },
                     self.loaded.elapsed().as_secs_f64())]
    }
}

/// A checkpoint's directory as downloaded: its files by role (the speech tokenizer in `speech_tokenizer/`)
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (role, name) in [("model", "model.safetensors"), ("config", "config.json"), ("vocab", "vocab.json"), ("merges", "merges.txt"),
                         ("tokenizer-config", "tokenizer_config.json"), ("codec", "speech_tokenizer/model.safetensors")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}

/// Floats to their bytes (for a dump)
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytes(v)
}
