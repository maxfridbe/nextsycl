//! VibeVoice-1.5B (Microsoft) on Intel Arc (SYCL): long-form, multi-speaker speech at 24 kHz - a script of up to four
//! speakers ("Speaker 1: ...\nSpeaker 2: ..."), each speaker's voice cloned from a recording (no transcript needed),
//! or a voice of the model's own without one.
//!
//! ```text
//!   the prompt (prompt.rs): the script's text, each voice's recording as acoustic latents (vae.rs: the acoustic
//!   encoder, 7.5 a second) through the acoustic connector
//!   -> the language model (lm.rs, Qwen2.5-1.5B): at each step one of four tokens - audio start / end, audio, the end
//!   -> an audio token: the diffusion head (head.rs) draws the frame's 64-value latent from noise, guided (cfg) by a
//!      second, negative run of the language model from an audio start alone
//!   -> the acoustic decoder (vae.rs, streaming) turns the latent into 3,200 samples; the semantic encoder reads them
//!      back; the next input is the acoustic connector's latent plus the semantic connector's features
//! ```
//!
//! The weights are gguf-org's GGUF of microsoft/VibeVoice-1.5B (`nextsycl models pull vibevoice-1.5b`: q8_0 - the
//! language model's matrices; the tokenizers, connectors and head in BF16 - expanded to half at load); the tokenizer
//! and config.json are the -hf checkpoint's (vibevoice/VibeVoice-1.5B-hf). `check` compares the stages with the
//! reference's dumps (reference/vibevoice/ref.py: the model's transformers implementation, float32 on the CPU).
//!
//! License: MIT (the weights and upstream's code). Microsoft asks that it not be used to impersonate a voice without
//! its owner's consent, nor to pass synthetic speech off as real (its model card's responsible-use section).

pub mod check;
mod ffi;
pub mod head;
pub mod lm;
pub mod ops;
pub mod prompt;
pub mod sched;
pub mod vae;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Reference, Result, Speech, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_tok::Tokenizer;

use crate::head::Head;
use crate::lm::{Cache, Lm};
use crate::ops::{Mat, Ops};
use crate::prompt::{Line, Specials};
use crate::vae::Coder;
use crate::weights::{Config, Weights};

pub const ARCH: &str = "vibevoice";
pub const ROLES: &[&str] = &["model", "tokenizer", "config"];
pub const RATE: u32 = 24000;
/// upstream's demo settings: guidance 1.3, ten DPM-Solver steps
pub const CFG: f32 = 1.3;
pub const STEPS: u32 = 10;
/// the speakers a script may have (upstream's)
pub const SPEAKERS: usize = 4;
/// the language model's positions (Qwen2.5-1.5B's 65,536), and the longest a request runs (90 minutes, upstream's)
const MAX_POS: usize = 65536;
const MAX_SECONDS: f32 = 5400.0;
/// a run's steps at most, a prompt row each: a bound on a run that does not end. Upstream's is 2, which cuts a long
/// script read without recordings short (speech takes about 2.4 frames a text token); the request's seconds bound it too
const STEPS_A_ROW: usize = 4;
/// prompt rows a pass
pub(crate) const CHUNK: usize = 256;
/// a voice's recording through the encoder a piece at a time (frames a piece)
const ENC_FRAMES: usize = 8;

pub const OPTIONS: &[EngineOption] = &[];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "VibeVoice (long-form speech of up to four speakers, each cloned from a recording; 7.5 Hz latents drawn by a diffusion head)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(VibeVoice::load(files, gpu, o, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// splitmix64 with Box-Muller: a seeded stream of standard normal values (the diffusion's noise, a voice's VAE noise)
pub struct Rng(u64, Option<f64>);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x7669_6265_766f_6963, None)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.1.take() {
            return v as f32;
        }
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        let r = (-2.0 * u.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * v;
        self.1 = Some(r * a.sin());
        (r * a.cos()) as f32
    }
}

/// A connector: Linear (+ bias) -> RMS norm (ε 1e-6) -> Linear (+ bias), the acoustic latent's or the semantic
/// features' way into the language model
pub struct Connector {
    fc1: Mat,
    norm: DevBuf,
    fc2: Mat,
}

impl Connector {
    fn load(ops: &Ops, f: &Weights, p: &str) -> Result<Connector> {
        Ok(Connector {
            fc1: f.mat(ops, &[&format!("{p}.fc1.weight")], Some(&[&format!("{p}.fc1.bias")]), true)?,
            norm: f.vec(ops, &format!("{p}.norm.weight"))?,
            fc2: f.mat(ops, &[&format!("{p}.fc2.weight")], Some(&[&format!("{p}.fc2.bias")]), false)?,
        })
    }

    /// The input rows' width (padded with zeros to a multiple of 256)
    pub fn width(&self) -> usize {
        self.fc1.k
    }

    /// x [m, width] -> out [m, hidden]; `tmp`, `xh`: work of m * hidden floats (half: m * width)
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, ops: &Ops, nsd: &Nsd, x: &DevBuf, m: usize, tmp: &DevBuf, xh: &DevBuf, out: &DevBuf) -> Result<()> {
        let h = self.fc1.n;
        self.fc1.apply(ops, nsd, x.fp(), m, xh, out.fp())?;
        nsd.rms_norm_mod(out.ptr(), Dt::F32, m, h, self.norm.ptr(), 1e-6, none(), none(), none(), tmp.ptr(), Dt::F32)?;
        self.fc2.apply(ops, nsd, tmp.fp(), m, xh, out.fp())
    }
}

pub struct VibeVoice {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub w: Weights,
    pub cfg: Config,
    pub lm: Lm,
    pub head: Head,
    /// the acoustic encoder (voices), the acoustic decoder, the semantic encoder
    pub enc: Coder,
    pub dec: Coder,
    pub sem: Coder,
    pub acoustic: Connector,
    pub semantic: Connector,
    /// the four tokens' rows of the (tied) output head: audio start, audio end, audio, the end of text
    pub(crate) logit_rows: Mat,
    /// the latents' scale and bias (speech_scaling_factor, speech_bias_factor)
    pub scale: f32,
    pub bias: f32,
    loaded: Instant,
    load_s: f64,
}

/// What a check keeps of a run: each step's token and its four logits, each frame's conditions [2, hidden], the
/// head's first output [2, 64], latent, semantic features and next input row
#[derive(Default)]
pub struct Keep {
    pub tokens: Vec<u32>,
    pub logits: Vec<[f32; 4]>,
    pub cond: Vec<Vec<f32>>,
    pub eps0: Vec<Vec<f32>>,
    pub latents: Vec<Vec<f32>>,
    pub semantic: Vec<Vec<f32>>,
    pub embeds: Vec<Vec<f32>>,
}

/// A run's settings
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub cfg: f32,
    pub steps: usize,
    /// at most this many frames (the request's seconds)
    pub frames: usize,
    pub seed: u64,
}

impl VibeVoice {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<VibeVoice> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let (model, tokf, conf) = (role(files, "model")?, role(files, "tokenizer")?, role(files, "config")?);
        let cfg = Config::read(&conf)?;
        let w = Weights::open(&model)?;
        // the file's matrices in half (Q8_0 doubles, BF16 / F16 the same), the convolutions and vectors in float32;
        // the token embedding stays in the file; work room 1.5 GiB
        let embd = w.tensor("model.language_model.embed_tokens.weight")?.bytes;
        let expanded: u64 = w.g.tensors.iter().filter(|t| !t.name.ends_with("embed_tokens.weight"))
            .map(|t| t.elements() * if t.shape.len() == 2 { 2 } else { 4 }).sum();
        let need = expanded + (3u64 << 29);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(&tokf).map_err(Error)?;
        for (t, id) in [("<|vision_start|>", cfg.audio_bos), ("<|vision_end|>", cfg.audio_eos), ("<|vision_pad|>", cfg.audio), ("<|endoftext|>", cfg.eos)] {
            if tok.id(t) != Some(id) {
                return Err(Error(format!("{}: {t} is not {id} (the config's) - not VibeVoice's (Qwen2.5's) tokenizer", tokf.display())));
            }
        }
        let lm = Lm::load(&ops, &w, cfg.heads, cfg.kv_heads, cfg.lm_eps, cfg.rope_theta)?;
        log(format!("language model: {} layers of {} in {:.1} s (its {:.0} MB embedding read from the file)", lm.layers(), lm.hidden,
                    t0.elapsed().as_secs_f64(), embd as f64 / 1e6));
        let head = Head::load(&ops, &w, cfg.head_eps, cfg.freq_dim, cfg.max_period)?;
        let enc = Coder::load(&ops, &w, "model.acoustic_tokenizer.encoder.", false, cfg.vae_eps, &cfg.ratios)?;
        let dec = Coder::load(&ops, &w, "model.acoustic_tokenizer.decoder.", true, cfg.vae_eps, &cfg.ratios)?;
        let sem = Coder::load(&ops, &w, "model.semantic_tokenizer.encoder.", false, cfg.vae_eps, &cfg.ratios)?;
        let acoustic = Connector::load(&ops, &w, "model.acoustic_connector")?;
        let semantic = Connector::load(&ops, &w, "model.semantic_connector")?;
        if enc.dim_out != head.latent || dec.dim_in != head.latent || acoustic.fc1.k_in != head.latent || semantic.fc1.k_in != sem.dim_out
           || head.hidden != lm.hidden || dec.dim_out != 1 || enc.dim_in != 1 || sem.dim_in != 1 {
            return Err(Error("the tokenizers, connectors and diffusion head do not fit each other".into()));
        }
        let embed = "model.language_model.embed_tokens.weight";
        let ids = [cfg.audio_bos, cfg.audio_eos, cfg.audio, cfg.eos].map(|i| i as usize);
        let logit_rows = w.mat_rows(&ops, &ids.map(|i| (embed, i, i + 1)), lm.hidden, None, false)?;
        let scale = w.scalar("model.speech_scaling_factor")?;
        let bias = w.scalar("model.speech_bias_factor")?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = gpu.memory()?.1.unwrap_or(free);
        log(format!("{ARCH} on {} in {load_s:.0} s ({:.1} GiB of weights on the card; tokenizers {:.0} MB)", gpu.name,
                    free.saturating_sub(after) as f64 / (1u64 << 30) as f64, (enc.bytes() + dec.bytes() + sem.bytes()) as f64 / 1e6));
        Ok(VibeVoice { ops, nsd, tok, w, cfg, lm, head, enc, dec, sem, acoustic, semantic, logit_rows, scale, bias, loaded: Instant::now(), load_s })
    }

    pub fn specials(&self) -> Specials {
        Specials { audio_bos: self.cfg.audio_bos, audio_eos: self.cfg.audio_eos, audio: self.cfg.audio }
    }

    /// A token's embedding row (host)
    pub fn embedding(&self, id: u32) -> Result<Vec<f32>> {
        self.w.rows("model.language_model.embed_tokens.weight", id as usize, id as usize + 1)
    }

    /// A recording as the acoustic encoder takes it: 24 kHz, its level brought to -25 dBFS (and below full scale),
    /// zeros to a whole frame (the feature extractor's)
    pub fn voice_samples(&self, r: &Reference) -> Result<Vec<f32>> {
        let mut x = nextsycl_codec12hz::resample(&r.samples, r.rate);
        if x.is_empty() {
            return Err(Error("a voice's recording is empty".into()));
        }
        let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32;
        let gain = 10f32.powf(-25.0 / 20.0) / (rms + 1e-6);
        x.iter_mut().for_each(|v| *v *= gain);
        let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        if peak > 1.0 {
            x.iter_mut().for_each(|v| *v /= peak + 1e-6);
        }
        let hop = self.cfg.hop();
        x.resize(x.len().div_ceil(hop) * hop, 0.0);
        Ok(x)
    }

    /// The acoustic encoder's latents of prepared samples (voice_samples) [F, 64], a few frames at a time
    pub fn encode(&self, x: &[f32]) -> Result<Vec<f32>> {
        let hop = self.cfg.hop();
        let piece = ENC_FRAMES * hop;
        let w = self.enc.work(&self.ops, piece)?;
        let s = self.enc.stream(&self.ops)?;
        let input = DevBuf::f32(&self.ops.gpu, piece)?;
        let mut out = Vec::with_capacity(x.len() / hop * self.enc.dim_out);
        for p in x.chunks(piece) {
            input.write(0, &bytes(p))?;
            let (y, rows) = self.enc.push(&self.ops, &self.nsd, &s, &input, p.len(), &w)?;
            self.ops.gpu.sync()?;
            let mut v = vec![0f32; rows * self.enc.dim_out];
            y.read(0, as_bytes_mut(&mut v))?;
            out.extend(v);
        }
        Ok(out)
    }

    /// Latents [F, 64] (host) through a connector: rows [F, hidden]
    pub fn connect(&self, c: &Connector, x: &[f32], k_in: usize) -> Result<Vec<f32>> {
        let (k, h) = (c.width(), self.lm.hidden);
        let m = x.len() / k_in;
        let padded: Vec<f32> = x.chunks(k_in).flat_map(|r| r.iter().copied().chain(std::iter::repeat_n(0.0, k - k_in))).collect();
        let g = &self.ops.gpu;
        let xb = DevBuf::from_f32(g, &padded)?;
        let (tmp, out) = (DevBuf::f32(g, m * h)?, DevBuf::f32(g, m * h)?);
        let xh = DevBuf::new(g, m * k.max(h) * 2)?;
        c.apply(&self.ops, &self.nsd, &xb, m, &tmp, &xh, &out)?;
        out.to_f32()
    }

    /// A voice's prompt rows: its latents with the VAE's sampling noise (std = vae_std x a normal draw, then a normal
    /// draw a value: upstream's encode(sample=True)), shifted and scaled, through the acoustic connector
    pub fn voice_rows(&self, latents: &[f32], rng: &mut Rng) -> Result<(Vec<f32>, Vec<f32>)> {
        let std = self.cfg.vae_std * rng.normal();
        let feats: Vec<f32> = latents.iter().map(|v| (v + std * rng.normal() + self.bias) * self.scale).collect();
        let rows = self.connect(&self.acoustic, &feats, self.head.latent)?;
        Ok((feats, rows))
    }

    /// The prompt's rows [L, hidden] (host): the ids' embeddings, a voice's rows where its frames go
    pub fn prompt_rows(&self, ids: &[u32], placed: &[(usize, usize, usize)], voices: &[Vec<f32>]) -> Result<Vec<f32>> {
        let h = self.lm.hidden;
        let mut rows = Vec::with_capacity(ids.len() * h);
        for id in ids {
            rows.extend(self.embedding(*id)?);
        }
        for ((_, at, frames), v) in placed.iter().zip(voices) {
            if v.len() != frames * h {
                return Err(Error(format!("a voice of {} rows where {frames} go", v.len() / h)));
            }
            rows[at * h..(at + frames) * h].copy_from_slice(v);
        }
        Ok(rows)
    }

    /// The speech: from the prompt's rows, step by step until the model ends it (or `s.frames` frames, or STEPS_A_ROW
    /// times the prompt's length in steps, or the model's positions). `noise`: a check's noise a frame (else drawn from
    /// the seed); `keep`: what a check compares
    #[allow(clippy::too_many_arguments)]
    pub fn speak(&self, rows: &[f32], s: Settings, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>, noise: Option<&[Vec<f32>]>,
                 mut keep: Option<&mut Keep>) -> Result<Vec<f32>> {
        let (ops, nsd, h) = (&self.ops, &self.nsd, self.lm.hidden);
        let l = rows.len() / h;
        if l + 2 > MAX_POS {
            return Err(Error(format!("the prompt is {l} positions; the model reads {MAX_POS} with the speech")));
        }
        let max_steps = (STEPS_A_ROW * l).min(s.frames + 2).min(MAX_POS - l - 1);
        let g = &ops.gpu;
        let pos: Cache = self.lm.cache(ops, CHUNK, l + max_steps + 1)?;
        let neg: Cache = self.lm.cache(ops, 2, max_steps + 2)?;
        let cond = DevBuf::f32(g, 2 * h)?;
        self.lm.prefill(ops, nsd, &pos, rows, 0, &cond)?;
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let mut hw = self.head.work(ops, s.steps)?;
        let (dw, sw) = (self.dec.work(ops, 1)?, self.sem.work(ops, self.cfg.hop())?);
        let (ds, ss) = (self.dec.stream(ops)?, self.sem.stream(ops)?);
        let logits = DevBuf::f32(g, 4)?;
        let lat_in = DevBuf::f32(g, self.dec.dim_in)?;
        let (sem_in, lat_row) = (DevBuf::f32(g, self.semantic.width())?, DevBuf::f32(g, self.acoustic.width())?);
        sem_in.fill(0)?;
        lat_row.fill(0)?;
        let (tmp, ctmp, xh) = (DevBuf::f32(g, h)?, DevBuf::f32(g, h)?, DevBuf::new(g, 2 * h * 2)?);
        let next = DevBuf::f32(g, h)?;
        let bos = DevBuf::from_f32(g, &self.embedding(self.cfg.audio_bos)?)?;
        let tokens = [self.cfg.audio_bos, self.cfg.audio_eos, self.cfg.audio, self.cfg.eos];
        let mut rng = Rng::new(s.seed);
        let (mut p, mut np, mut frames) = (l, 0usize, 0usize);
        // whether cond's second row holds the negative branch's next hidden state already (the paired step made it)
        let mut paired = false;
        let mut samples = Vec::new();
        for _ in 0..max_steps {
            ops.gemv(cond.fp(), 1, &self.logit_rows, logits.fp(), 4, false)?;
            let l4 = logits.to_f32()?;
            let best = (0..4).fold(0, |b, i| if l4[i] > l4[b] { i } else { b });
            let tok = tokens[best];
            if let Some(k) = keep.as_deref_mut() {
                k.tokens.push(tok);
                k.logits.push([l4[0], l4[1], l4[2], l4[3]]);
            }
            if tok == self.cfg.audio_eos {
                // upstream clears the tokenizers' streams at an audio end
                self.dec.reset(&ds)?;
                self.sem.reset(&ss)?;
            }
            if tok == self.cfg.audio_bos {
                // the negative branch starts over: its audio start at position 0, the next input at 1
                self.lm.step(ops, nsd, &neg, &bos, 0, &tmp)?;
                np = 1;
                paired = false;
            }
            if tok == self.cfg.audio {
                // the negative branch reads what the positive one just read (an audio start, before any): the paired
                // step read it already
                if !paired {
                    self.lm.step(ops, nsd, &neg, &bos, np, &tmp)?;
                    cond.copy_within(h * 4, &tmp, 0, h * 4)?;
                }
                np += 1;
                let z: Vec<f32> = match noise {
                    Some(n) => n.get(frames).cloned().ok_or_else(|| Error(format!("no noise for frame {frames}")))?,
                    None => (0..self.head.latent).map(|_| rng.normal()).collect(),
                };
                let mut first = keep.as_ref().map(|_| Vec::new());
                let latent = self.head.sample(ops, nsd, &mut hw, &cond, &z, s.cfg, first.as_mut())?;
                // the acoustic decoder on the unscaled latent, then the semantic encoder on its samples
                let un: Vec<f32> = latent.iter().map(|v| v / self.scale - self.bias).collect();
                lat_in.write_async(0, &bytes(&un))?;
                let (y, n) = self.dec.push(ops, nsd, &ds, &lat_in, 1, &dw)?;
                let (sv, _) = self.sem.push(ops, nsd, &ss, y, n, &sw)?;
                sem_in.copy_within(0, sv, 0, self.sem.dim_out * 4)?;
                let mut chunk = vec![0f32; n];
                y.read(0, as_bytes_mut(&mut chunk))?;
                samples.extend_from_slice(&chunk);
                // the next input: the acoustic connector's latent plus the semantic connector's features
                lat_row.write_async(0, &bytes(&latent))?;
                self.acoustic.apply(ops, nsd, &lat_row, 1, &ctmp, &xh, &next)?;
                self.semantic.apply(ops, nsd, &sem_in, 1, &ctmp, &xh, &tmp)?;
                ops.add(next.fp(), tmp.fp(), h)?;
                frames += 1;
                if let Some(k) = keep.as_deref_mut() {
                    k.cond.push(cond.to_f32()?);
                    k.eps0.push(first.unwrap_or_default());
                    k.latents.push(latent);
                    k.semantic.push(sem_in.to_f32()?[..self.sem.dim_out].to_vec());
                    k.embeds.push(next.to_f32()?);
                }
                progress(Step { phase: "tokens", at: frames as u32, of: s.frames.min(max_steps) as u32, seconds: t0.elapsed().as_secs_f64() })?;
            } else {
                next.write(0, &bytes(&self.embedding(tok)?))?;
            }
            if tok == self.cfg.eos || frames >= s.frames {
                break;
            }
            // both branches read the next input in one pass (the weights read once): the positive always, the negative
            // for the next step if it is an audio frame (else its row goes unused and its position is written again)
            self.lm.pair(ops, nsd, &pos, p, &neg, np, &next, &cond)?;
            paired = true;
            p += 1;
        }
        g.sync()?;
        if samples.is_empty() {
            return Err(Error("the model ended the speech before its first frame".into()));
        }
        Ok(samples)
    }

    /// The request's script, voices and settings to a prompt's rows
    pub fn prepare(&self, req: &AudioRequest, rng: &mut Rng) -> Result<(Vec<u32>, Vec<f32>, Vec<Line>)> {
        if req.prompt.trim().is_empty() {
            return Err(Error("the text to say is empty".into()));
        }
        if let Some(v) = req.voice.as_deref().filter(|v| !v.is_empty()) {
            return Err(Error(format!("voice {v}: {ARCH} has no built-in voices (clone a recording, or a saved voice)")));
        }
        if req.instructions.as_deref().is_some_and(|i| !i.trim().is_empty()) {
            return Err(Error(format!("instructions: {ARCH} is not steered by instructions (its voices come from recordings)")));
        }
        let lines = prompt::parse(&req.prompt);
        let refs: Vec<&Reference> = req.reference.iter().chain(&req.references).collect();
        if refs.len() > SPEAKERS {
            return Err(Error(format!("{} voices: {ARCH} speaks {SPEAKERS} at most", refs.len())));
        }
        let speakers = lines.iter().map(|l| l.speaker + 1).max().unwrap_or(0);
        if speakers > SPEAKERS {
            return Err(Error(format!("Speaker {speakers}: {ARCH} speaks {SPEAKERS} at most")));
        }
        let mut voices: Vec<Option<usize>> = Vec::new();
        let mut lat = Vec::new();
        for r in &refs {
            let x = self.voice_samples(r)?;
            voices.push(Some(x.len() / self.cfg.hop()));
            lat.push(self.encode(&x)?);
        }
        let (ids, placed) = prompt::ids(&self.tok, self.specials(), &lines, &voices)?;
        let mut rows = Vec::new();
        for (s, _, _) in &placed {
            rows.push(self.voice_rows(&lat[*s], rng)?.1);
        }
        let rows = self.prompt_rows(&ids, &placed, &rows)?;
        Ok((ids, rows, lines))
    }
}

impl AudioEngine for VibeVoice {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 600.0, max_seconds: MAX_SECONDS, steps: STEPS, cfg: CFG, rate: RATE }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech {
            voices: Vec::new(),
            languages: vec!["chinese".into(), "english".into()],
            instructions: false,
            design: false,
            clone: true,
            clone_needs_text: false,
            clone_takes_text: false,
            direction: false,
            events: Vec::new(),
            requires: None,
            speakers: SPEAKERS,
            enrolled: false,
        })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let mut rng = Rng::new(req.seed ^ 0x5eed);
        let (_, rows, _) = self.prepare(req, &mut rng)?;
        let seconds = req.seconds.unwrap_or(self.defaults().seconds);
        if seconds <= 0.0 {
            return Err(Error(format!("{seconds} s: a length above zero")));
        }
        let fps = RATE as f32 / self.cfg.hop() as f32;
        let frames = ((seconds.min(MAX_SECONDS) * fps).ceil() as usize).max(1);
        let cfg = req.cfg.unwrap_or(CFG);
        if !(cfg.is_finite() && cfg > 0.0) {
            return Err(Error(format!("cfg {cfg}: above zero")));
        }
        let steps = req.steps.unwrap_or(STEPS).clamp(1, 100) as usize;
        let samples = self.speak(&rows, Settings { cfg, steps, frames, seed: req.seed }, t0, progress, None, None)?;
        Ok(Audio { rate: RATE, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (half, loaded {:.0} s ago)", self.ops.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A directory with the GGUF (q8_0 preferred), tokenizer.json and config.json (as `models pull` lays them out, or by
/// hand): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let gguf = std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("vibevoice")))
        // the q8_0 file when there are several (the card's), else the first by name
        .min_by_key(|p| (!p.to_string_lossy().contains("q8_0"), p.clone())).ok_or_else(|| Error(format!("{}: no vibevoice*.gguf", dir.display())))?;
    f.insert("model".into(), gguf);
    for (role, name) in [("tokenizer", "tokenizer.json"), ("config", "config.json")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Floats' storage as bytes (a read's destination)
pub fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: plain floats viewed as their bytes, the same length in bytes.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_noise_is_standard_normal_and_seeded() {
        let mut r = Rng::new(7);
        let v: Vec<f32> = (0..20000).map(|_| r.normal()).collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
        let mut a = Rng::new(7);
        assert_eq!(a.normal(), v[0]);
    }
}
