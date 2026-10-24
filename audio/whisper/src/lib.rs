//! Whisper (OpenAI's speech recogniser: large-v3-turbo's 32-layer encoder and 4-layer decoder) on SYCL: a recording
//! in any language to its text, its segments timed by the model's timestamp tokens, and its words timed by the
//! decoder's cross-attention - openai-whisper's `transcribe` with `word_timestamps`, greedy.
//!
//! ```text
//!   samples (any rate) -> mel.rs: 16 kHz, the log-mel spectrogram (128 mels for v3), 30 s of silence after it
//!   -> windows of 30 s from `seek`: model.rs's encoder, the decoder from <|startoftranscript|> <|lang|> <|transcribe|>
//!      (the language detected on the first window when not given), greedy under text.rs's timestamp rules
//!   -> text.rs: the window cut into segments by its timestamp pairs; seek moves to the last one (or the window's end)
//!   -> align.rs (asked for words): the alignment heads' attention over the window, DTW, the words placed
//! ```
//!
//! The weights are the checkpoint's own files (`nextsycl models pull whisper-large-v3-turbo`): model.safetensors
//! (half on the GPU, ~1.6 GB), config.json, tokenizer.json, generation_config.json (the alignment heads, the
//! suppressed tokens). `check` compares the stages with the reference's dumps (reference/whisper/ref.py).

pub mod align;
pub mod check;
mod ffi;
pub mod mel;
pub mod model;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Recogniser, Result, Step, Timed,
                     TranscribeRequest, Transcript};
use nextsycl_core::Gpu;

use crate::align::{find_alignment, place_words, Placed};
use crate::mel::{CHUNK_FRAMES, RATE};
use crate::model::{Config, Model, Taps};
use crate::text::{apply_rules, cut, pick, prob, Vocab};

pub const ARCH: &str = "whisper";
pub const ROLES: &[&str] = &["model", "config", "tokenizer", "generation"];
/// frames a second of the spectrogram
const FRAMES_PER_SECOND: f64 = 100.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "condition", env: "NS_WH_CONDITION", value: "0|1",
                   help: "each window prompted with the text before it (openai's condition_on_previous_text: steadier style, but a loop carries on; off by default, as transformers' pipeline)",
                   at: At::Both },
    EngineOption { name: "no-speech", env: "NS_WH_NO_SPEECH", value: "0.6", help: "a window is silence when <|nospeech|> is likelier than this and its text unsure (avg log-prob under -1)", at: At::Both },
    EngineOption { name: "fallback", env: "NS_WH_FALLBACK", value: "0|1",
                   help: "a window that runs on without ending, or decodes unsure (avg log-prob under -1), decoded again at rising temperatures (openai's fallback; on by default)", at: At::Both },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "Whisper (speech to text in 99 languages: segments and words timed; large-v3-turbo)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Whisper::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// A checkpoint's directory as downloaded (transformers' layout): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (r, name) in [("model", "model.safetensors"), ("config", "config.json"), ("tokenizer", "tokenizer.json"), ("generation", "generation_config.json")] {
        let p = dir.join(name);
        if !p.is_file() {
            return Err(Error(format!("{}: no {name} (a Whisper checkpoint's directory)", dir.display())));
        }
        f.insert(r.into(), p);
    }
    Ok(f)
}

pub struct Whisper {
    pub model: Model,
    pub vocab: Vocab,
    settings: LoadOptions,
    load_s: f64,
}

/// What one window's decoding gave
pub struct Decoded {
    /// the tokens sampled, <|endoftext|> left out
    pub tokens: Vec<u32>,
    pub avg_logprob: f64,
    pub no_speech: f64,
    /// it ended (<|endoftext|>) within the window's budget
    pub ended: bool,
    pub temperature: f32,
}

/// A small deterministic generator for the fallback's sampling
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Whisper {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Whisper> {
        let t0 = Instant::now();
        let cfg = Config::read(&role(files, "config")?)?;
        let vocab = Vocab::open(&role(files, "tokenizer")?, &role(files, "generation")?, cfg.dec_layers, cfg.dec_heads, cfg.vocab)?;
        if let Some((l, h)) = vocab.alignment_heads.iter().find(|(l, h)| *l >= cfg.dec_layers || *h >= cfg.dec_heads) {
            return Err(Error(format!("alignment head ({l}, {h}) is not in a {}-layer, {}-head decoder", cfg.dec_layers, cfg.dec_heads)));
        }
        let model = Model::load(&role(files, "model")?, cfg, gpu, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("whisper: loaded in {load_s:.1} s"));
        Ok(Whisper { model, vocab, settings: o.clone(), load_s })
    }

    /// An option: the request's, else the load's (its settings, the environment)
    fn opt(&self, req: &TranscribeRequest, name: &str, env: &str) -> Option<String> {
        req.extra.get(name).cloned().or_else(|| self.settings.setting(env))
    }

    /// The language of the window just encoded: the likeliest language token after <|startoftranscript|>
    pub fn detect_language(&self) -> Result<String> {
        let l = self.model.decode(&[self.vocab.sot], 0, 0, None)?;
        let best = self.vocab.languages.iter().max_by(|a, b| l[a.1 as usize].partial_cmp(&l[b.1 as usize]).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error("no language tokens".into()))?;
        Ok(best.0.clone())
    }

    /// Greedy (temperature 0) or sampled decoding of the window just encoded after `ctx` (a previous text's prompt and
    /// the start of transcript sequence; `sot_index`: where <|startoftranscript|> is in it)
    pub fn decode_window(&self, ctx: &[u32], sot_index: usize, temperature: f32, seed: u64, step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Decoded> {
        let v = &self.vocab;
        let vs = self.model.cfg.vocab;
        let budget = self.model.cfg.text_ctx / 2;
        let logits = self.model.decode(ctx, 0, sot_index, None)?;
        let no_speech = prob(&logits[..vs], v.no_speech);
        let rows = ctx.len() - sot_index;
        let mut last = logits[(rows - 1) * vs..rows * vs].to_vec();
        let mut sampled: Vec<u32> = Vec::new();
        let mut sum = 0f64;
        let mut ended = false;
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        for i in 0..budget {
            apply_rules(&mut last, &sampled, v, Some(v.max_initial_timestamp));
            let (tok, lp) = if temperature > 0.0 {
                // softmax(logits / T) sampled; its log-probability under the filtered logits at T
                let m = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f64> = last.iter().map(|x| (((x - m) / temperature) as f64).exp()).collect();
                let z: f64 = w.iter().sum();
                let mut r = rng.next() * z;
                let mut t = 0usize;
                for (j, x) in w.iter().enumerate() {
                    if *x == 0.0 {
                        continue;
                    }
                    t = j;
                    r -= x;
                    if r <= 0.0 {
                        break;
                    }
                }
                (t as u32, (w[t] / z).ln())
            } else {
                pick(&last)
            };
            sum += lp;
            if tok == v.eot {
                ended = true;
                break;
            }
            sampled.push(tok);
            step(i)?;
            if ctx.len() + sampled.len() >= self.model.cfg.text_ctx {
                break;
            }
            last = self.model.decode(&[tok], ctx.len() + sampled.len() - 1, 0, None)?;
        }
        let avg_logprob = sum / (sampled.len() + 1) as f64;
        Ok(Decoded { tokens: sampled, avg_logprob, no_speech, ended, temperature })
    }

    /// The words of a window's segments (its text tokens, the window encoded): timed from the window's start
    pub fn window_words(&self, text: &[u32], language: &str, translate: bool, num_frames: usize) -> Result<Vec<align::Word>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let v = &self.vocab;
        let c = &self.model.cfg;
        let sot = v.sot_sequence(language, translate);
        let mut fed = sot.clone();
        fed.push(v.no_timestamps);
        fed.extend_from_slice(text);
        fed.push(v.eot);
        if fed.len() > c.text_ctx {
            return Ok(Vec::new());
        }
        let mut layers: Vec<usize> = v.alignment_heads.iter().map(|(l, _)| *l).collect();
        layers.sort();
        layers.dedup();
        let mut taps = Taps { layers: layers.clone(), q: Vec::new() };
        let logits = self.model.decode(&fed, 0, sot.len(), Some(&mut taps))?;
        // the text tokens' probabilities (a softmax over the text vocabulary, as openai's)
        let vs = c.vocab;
        let e = v.eot as usize;
        let probs: Vec<f32> = text.iter().enumerate().map(|(i, t)| {
            let row = &logits[i * vs..i * vs + e];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = row.iter().map(|x| ((x - m) as f64).exp()).sum();
            (((row[*t as usize] - m) as f64).exp() / z) as f32
        }).collect();
        let m = (num_frames / 2).clamp(1, c.audio_ctx);
        let (d, hd) = (c.d, c.d / c.dec_heads);
        let rows = fed.len();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qk = Vec::new();
        let mut keys: Vec<(usize, Vec<f32>)> = Vec::new();
        for l in &layers {
            keys.push((*l, self.model.cross_keys(*l)?));
        }
        for (l, h) in &v.alignment_heads {
            let q = &taps.q[layers.iter().position(|x| x == l).unwrap_or(0)];
            let k = &keys.iter().find(|(x, _)| x == l).unwrap().1;
            let mut s = vec![0f32; rows * m];
            for r in 0..rows {
                let qr = &q[r * d + h * hd..r * d + (h + 1) * hd];
                for f in 0..m {
                    let kr = &k[f * d + h * hd..f * d + (h + 1) * hd];
                    s[r * m + f] = qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
            }
            qk.push(s);
        }
        Ok(find_alignment(&qk, rows, m, num_frames, sot.len(), text, &probs, v, language, 7))
    }

    pub fn run(&self, req: &TranscribeRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Transcript> {
        let t0 = Instant::now();
        let v = &self.vocab;
        let c = &self.model.cfg;
        let flag = |name: &str, env: &str, dflt: bool| self.opt(req, name, env).map_or(dflt, |x| x == "1" || x == "true");
        let condition = flag("condition", "NS_WH_CONDITION", false);
        let fallback = flag("fallback", "NS_WH_FALLBACK", true);
        let no_speech_t: f64 = self.opt(req, "no-speech", "NS_WH_NO_SPEECH").map(|x| x.parse().map_err(|_| Error(format!("no-speech: {x}?")))).transpose()?.unwrap_or(0.6);
        let x = mel::resample(&req.samples, req.rate, RATE);
        let duration = x.len() as f64 / RATE as f64;
        let (spec, content) = mel::log_mel(&x, c.n_mels);
        let frames = spec.len() / c.n_mels;
        let mut language = v.language(req.language.as_deref())?;
        let mut all_tokens: Vec<u32> = match req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
            Some(p) => v.tok.encode(&format!(" {p}")),
            None => Vec::new(),
        };
        let mut reset = 0usize;
        let mut segments: Vec<Placed> = Vec::new();
        let mut last_speech = 0f64;
        let mut seek = 0usize;
        let secs = |t: Instant| t.elapsed().as_secs_f64();
        while seek < content {
            let seg_frames = CHUNK_FRAMES.min(content - seek);
            let time_offset = seek as f64 / FRAMES_PER_SECOND;
            progress(Step { phase: "listen", at: seek as u32, of: content as u32, seconds: secs(t0) })?;
            self.model.encode(&mel::window(&spec, c.n_mels, frames, seek, seg_frames))?;
            let lang = match &language {
                Some(l) => l.clone(),
                None => {
                    let l = self.detect_language()?;
                    language = Some(l.clone());
                    l
                }
            };
            // the prompt: the text before (its last half-context of tokens), then the start of transcript
            let prev = &all_tokens[reset..];
            let mut ctx = Vec::new();
            if !prev.is_empty() {
                ctx.push(v.sot_prev);
                let keep = c.text_ctx / 2 - 1;
                ctx.extend_from_slice(&prev[prev.len().saturating_sub(keep)..]);
            }
            let sot_index = ctx.len();
            ctx.extend(v.sot_sequence(&lang, req.translate));
            let mut step = |i: usize| progress(Step { phase: "tokens", at: seek as u32 + (i as u32).min(seg_frames as u32 - 1), of: content as u32, seconds: secs(t0) });
            let mut d = self.decode_window(&ctx, sot_index, 0.0, seek as u64, &mut step)?;
            if fallback {
                for (k, t) in [0.2f32, 0.4, 0.6, 0.8, 1.0].iter().enumerate() {
                    let unsure = d.avg_logprob < -1.0 && !(d.no_speech > no_speech_t);
                    if d.ended && !unsure {
                        break;
                    }
                    d = self.decode_window(&ctx, sot_index, *t, seek as u64 * 7 + k as u64 + 1, &mut step)?;
                }
            }
            // silence: the window skipped
            if d.no_speech > no_speech_t && d.avg_logprob <= -1.0 {
                seek += seg_frames;
                continue;
            }
            let (cuts, advance) = cut(&d.tokens, v, seg_frames);
            let single_ending = d.tokens.len() >= 2 && d.tokens[d.tokens.len() - 2] < v.timestamp_begin && d.tokens[d.tokens.len() - 1] >= v.timestamp_begin;
            let mut placed: Vec<Placed> = cuts.into_iter().map(|c| {
                let text: Vec<u32> = c.tokens.iter().copied().filter(|t| *t < v.eot).collect();
                let empty = c.start == c.end || v.text(&text).trim().is_empty();
                Placed { start: time_offset + c.start, end: time_offset + c.end, text_tokens: if empty { Vec::new() } else { text }, words: Vec::new() }
            }).collect();
            let mut next = seek + advance;
            if req.words {
                progress(Step { phase: "words", at: seek as u32, of: content as u32, seconds: secs(t0) })?;
                let text: Vec<u32> = placed.iter().flat_map(|p| p.text_tokens.iter().copied()).collect();
                let words = self.window_words(&text, &lang, req.translate, seg_frames)?;
                last_speech = place_words(&mut placed, words, time_offset, last_speech);
                let end = placed.iter().rev().find_map(|p| p.words.last().map(|w| w.end)).or_else(|| placed.last().map(|p| p.end));
                if !single_ending {
                    if let Some(e) = end.filter(|e| *e > time_offset) {
                        next = (e * FRAMES_PER_SECOND).round() as usize;
                    }
                }
                if let Some(e) = end {
                    last_speech = e;
                }
            }
            // never back, never in place
            if next <= seek {
                next = seek + seg_frames;
            }
            for p in &placed {
                all_tokens.extend_from_slice(&p.text_tokens);
            }
            if !condition || d.temperature > 0.5 {
                reset = all_tokens.len();
            }
            segments.extend(placed.into_iter().filter(|p| !p.text_tokens.is_empty()));
            seek = next;
        }
        let out_segments: Vec<Timed> = segments.iter().map(|p| Timed { start: round2(p.start), end: round2(p.end), text: v.text(&p.text_tokens).trim().to_string(),
                                                                         probability: None }).collect();
        let words: Vec<Timed> = segments.iter().flat_map(|p| p.words.iter()).map(|w| Timed { start: w.start, end: w.end, text: w.word.trim().to_string(),
                                                                                                probability: Some(w.probability) }).collect();
        let text = v.text(&segments.iter().flat_map(|p| p.text_tokens.iter().copied()).collect::<Vec<_>>()).trim().to_string();
        progress(Step { phase: "done", at: content as u32, of: content as u32, seconds: secs(t0) })?;
        Ok(Transcript { text, language: language.unwrap_or_default(), segments: out_segments, words, duration })
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

impl AudioEngine for Whisper {
    fn arch(&self) -> &'static str {
        ARCH
    }

    fn defaults(&self) -> Defaults {
        Defaults { seconds: 0.0, max_seconds: 4.0 * 3600.0, steps: 0, cfg: 0.0, rate: RATE }
    }

    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }

    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, _req: &AudioRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        Err(Error("whisper writes speech down; it makes no sound (audio.transcribe)".into()))
    }

    fn recogniser(&self) -> Option<Recogniser> {
        Some(Recogniser { languages: self.vocab.languages.iter().map(|(c, _)| c.clone()).collect(), words: true, translate: true })
    }

    fn transcribe(&self, req: &TranscribeRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Transcript> {
        if req.samples.is_empty() {
            return Err(Error("the recording is empty".into()));
        }
        self.run(req, progress)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("whisper: {} encoder / {} decoder layers on {}", self.model.cfg.enc_layers, self.model.cfg.dec_layers, self.model.gpu.name)]
    }
}
