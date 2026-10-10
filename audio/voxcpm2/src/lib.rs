//! VoxCPM2 on Intel Arc (SYCL): tokenizer-free speech in 30 languages at 48 kHz. One model does it all: plain text
//! (the model picks a voice), a voice designed by a description in parentheses ahead of the text, a voice cloned
//! from a recording (its timbre; with the recording's transcript also its delivery, continued).
//!
//! ```text
//!   text (+ a recording) -> the prompt (lib.rs): token rows, the recording's AudioVAE latents as patches of 4
//!   -> model.rs: the base and residual MiniCPM4 language models, a patch at a time from the local DiT by flow
//!      matching (10 steps, guidance 2), the stop head
//!   -> vae.rs: AudioVAE V2's decoder, 25 latents a second to 48 kHz
//! ```
//!
//! The weights are the checkpoint's own files (`nextsycl models pull voxcpm2`): model.safetensors (bf16: half on the
//! GPU, the language models int8 with `--opt-int8`), audiovae.pth (float32, read by pth.rs), tokenizer.json. `check`
//! compares the stages with the reference's dumps (reference/voxcpm2/ref.py: voxcpm's own code on the same files).

pub mod check;
mod ffi;
pub mod lm;
pub mod model;
pub mod ops;
pub mod pth;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Speech, Step};
use nextsycl_core::Gpu;
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;

use crate::model::{Model, FEAT, PATCH};
use crate::ops::{Ops, Shards};
use crate::vae::Vae;

pub const ARCH: &str = "voxcpm2";
pub const ROLES: &[&str] = &["model", "config", "tokenizer", "vae"];
/// patches a second (25 latents a second, 4 a patch)
pub const PPS: f32 = 6.25;
const MAX_PATCHES: usize = 2000;
const AUDIO_START: u32 = 101;
const REF_START: u32 = 103;
const REF_END: u32 = 104;
/// continuation: the prompt's last patches decoded ahead of the new ones (the reference's streaming prefix less one)
const CONTEXT: usize = 3;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_VCP_INT8", value: "0|1", help: "the language models' matrices in int8 (less VRAM, faster patches)", at: At::Load },
    EngineOption { name: "max-ratio", env: "NS_VCP_MAX_RATIO", value: "6", help: "the most patches a text token (a runaway's bound, as the reference's)",
                   at: At::Request },
];

pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "VoxCPM2 2B (speech in 30 languages at 48 kHz: voices designed from a description, cloned from a recording)", roles: ROLES,
                load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(VoxCpm2::load(files, gpu, o, log)?))
}

/// Floats as their bytes
pub fn talker_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub struct VoxCpm2 {
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    /// multi-character Chinese tokens -> their characters' ids (as the reference's tokenizer wrapper)
    split: std::collections::HashMap<u32, Vec<u32>>,
    pub model: Model,
    pub vae: Vae,
    pub weights: Shards,
    int8: bool,
    loaded: Instant,
    load_s: f64,
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// What to say and in whose voice
#[derive(Clone, Debug)]
pub struct Spec {
    /// the text (a design or a style in parentheses ahead of it)
    pub text: String,
    /// a recording to take the timbre from (16 kHz)
    pub reference: Option<Vec<f32>>,
    /// a recording to continue (16 kHz) and its transcript
    pub prompt: Option<(Vec<f32>, String)>,
}

/// The prompt as the model takes it
pub struct Prompt {
    /// a row's token (0 at audio rows) and whether it is audio
    pub tokens: Vec<u32>,
    pub audio: Vec<bool>,
    /// each row's patch [4 x 64] (zeros at text rows)
    pub patches: Vec<Vec<f32>>,
    /// the target text's token count (the length bound)
    pub target: usize,
}

/// Mono samples at `rate` to `to` (a windowed-sinc resampler: Hann, 32 zero crossings, cut at the lower Nyquist)
pub fn resample(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / rate as f64;
    let cut = ratio.min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let center = i as f64 / ratio;
        let c0 = center.floor() as isize;
        let mut s = 0.0;
        for j in c0 - half..=c0 + half {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = (j as f64 - center) * cut;
            if d.abs() >= zc {
                continue;
            }
            let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
            s += x[j as usize] as f64 * sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut;
        }
        s as f32
    }).collect()
}

/// splitmix64 and Box-Muller: the sampler's noise from a seed
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x766f_7863_706d_3200)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
}

impl VoxCpm2 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<VoxCpm2> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let int8 = o.setting("NS_VCP_INT8").is_some_and(|v| v == "1");
        let (mp, cp, tp, vp) = (role(files, "model")?, role(files, "config")?, role(files, "tokenizer")?, role(files, "vae")?);
        let conf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp).map_err(|e| Error(format!("{}: {e}", cp.display())))?)
            .map_err(|e| Error(format!("{}: {e}", cp.display())))?;
        if conf["architecture"] != "voxcpm2" {
            return Err(Error(format!("{}: architecture {}, not voxcpm2", cp.display(), conf["architecture"])));
        }
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        // the bf16 file less its token embedding (left on disk); the VAE in float32 and its work; 1.5 GiB free
        let lm = size(&mp).saturating_sub(73_448 * 2048 * 2);
        let need = lm / if int8 { 2 } else { 1 } + size(&vp) + (3u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                     free as f64 / (1u64 << 30) as f64)));
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(&tp).map_err(Error)?;
        if tok.id("<0x41>").is_none() {
            return Err(Error(format!("{}: not VoxCPM2's tokenizer (no byte tokens)", tp.display())));
        }
        let is_cjk = |c: char| ('\u{4e00}'..='\u{9fff}').contains(&c);
        // as the reference: a token whose text without "▁" is one of the vocabulary's all-Chinese entries of two or
        // more characters becomes its characters (the "▁" dropped)
        let split = tok.tokens.iter().enumerate().filter_map(|(id, t)| {
            let clean: String = t.replace('\u{2581}', "");
            (clean.chars().count() >= 2 && clean.chars().all(is_cjk) && tok.id(&clean).is_some())
                .then(|| (id as u32, clean.chars().map(|c| tok.id(&c.to_string()).unwrap_or(0)).collect()))
        }).collect();
        let weights = Shards::open(&[mp])?;
        let model = Model::load(&ops, &nsd, &weights, &conf, int8, log)?;
        let vae = Vae::load(&ops, &pth::load(&vp)?, &conf)?;
        gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH} on {} in {load_s:.0} s", gpu.name));
        Ok(VoxCpm2 { ops, nsd, tok, split, model, vae, weights, int8, loaded: Instant::now(), load_s })
    }

    /// The text's token ids (no BOS), multi-character Chinese tokens split into their characters
    pub fn tokens(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text).into_iter().flat_map(|id| self.split.get(&id).cloned().unwrap_or_else(|| vec![id])).collect()
    }

    /// 16 kHz samples padded to whole patches (`left`: ahead) to patches [n][4 x 64]
    pub fn patches(&self, wav: &[f32], left: bool) -> Result<Vec<Vec<f32>>> {
        let pl = PATCH * self.vae.hop_in;
        let pad = (pl - wav.len() % pl) % pl;
        let mut w = Vec::with_capacity(wav.len() + pad);
        if left {
            w.extend(std::iter::repeat_n(0.0, pad));
        }
        w.extend_from_slice(wav);
        if !left {
            w.extend(std::iter::repeat_n(0.0, pad));
        }
        let z = self.vae.encode(&self.ops, &w)?;
        let t = z.len() / FEAT;
        let z = &z;
        Ok((0..t / PATCH).map(|i| (0..PATCH).flat_map(|p| (0..FEAT).map(move |c| z[c * t + i * PATCH + p])).collect()).collect())
    }

    /// The prompt for a spec, as the reference's _generate builds it
    pub fn prompt(&self, s: &Spec) -> Result<Prompt> {
        let mut p = Prompt { tokens: Vec::new(), audio: Vec::new(), patches: Vec::new(), target: self.tokens(&s.text).len() };
        let zero = vec![0f32; PATCH * FEAT];
        let push = |p: &mut Prompt, tok: u32, patch: Option<Vec<f32>>| {
            p.audio.push(patch.is_some());
            p.tokens.push(if patch.is_some() { 0 } else { tok });
            p.patches.push(patch.unwrap_or_else(|| zero.clone()));
        };
        if let Some(r) = &s.reference {
            push(&mut p, REF_START, None);
            for pa in self.patches(r, false)? {
                push(&mut p, 0, Some(pa));
            }
            push(&mut p, REF_END, None);
        }
        let text = match &s.prompt {
            Some((_, t)) => format!("{t}{}", s.text),
            None => s.text.clone(),
        };
        for t in self.tokens(&text) {
            push(&mut p, t, None);
        }
        push(&mut p, AUDIO_START, None);
        if let Some((w, _)) = &s.prompt {
            for pa in self.patches(w, true)? {
                push(&mut p, 0, Some(pa));
            }
        }
        Ok(p)
    }

    /// The patches [n][4 x 64]: the prompt's continuation context first (`context` of them), then the made ones.
    /// `noise(i)`: patch i's starting noise [64, 4] (a check's from the reference; else seeded draws). `teacher`: a
    /// check's patches fed back in place of the made ones (each made patch then from the reference's history)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_patches(&self, p: &Prompt, max: usize, steps: usize, cfg: f32, noise: &mut dyn FnMut(usize) -> Vec<f32>, t0: Instant,
                            progress: &mut dyn FnMut(Step) -> Result<()>, keep: Option<&mut Vec<Vec<f32>>>, teacher: Option<&[Vec<f32>]>)
                            -> Result<(Vec<Vec<f32>>, usize)> {
        let (ops, nsd, m) = (&self.ops, &self.nsd, &self.model);
        let h = m.base.hidden;
        let l = p.tokens.len();
        let s = m.session(ops, l, max)?;
        // the text rows' embeddings (read from the checkpoint, a row a token; zeros at audio rows)
        let mut rows = vec![0f32; l * h];
        for (i, t) in p.tokens.iter().enumerate() {
            if !p.audio[i] {
                rows[i * h..(i + 1) * h].copy_from_slice(&self.weights.rows_f32("base_lm.embed_tokens.weight", *t as usize, *t as usize + 1)?);
            }
        }
        m.prefill(ops, nsd, &s, &rows, &p.audio, &p.patches)?;
        if let Some(k) = keep {
            k.push(s.lm.to_f32()?);
            k.push(s.res_h.to_f32()?);
        }
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        // a continuation's last patches go ahead of the new ones (the decoder's context)
        let context = if p.audio.last() == Some(&true) { CONTEXT.min(p.audio.iter().filter(|a| **a).count()) } else { 0 };
        let mut out: Vec<Vec<f32>> = p.patches[l - context..].to_vec();
        let mut prev = p.patches[l - 1].clone();
        for i in 0..max {
            m.mu(ops, nsd, &s)?;
            let patch = m.sample(ops, nsd, &s, &noise(i), &prev, steps, cfg)?;
            out.push(patch.clone());
            progress(Step { phase: "tokens", at: (i + 1) as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
            let fed = match teacher {
                Some(t) => match t.get(i) {
                    Some(x) => x.clone(),
                    None => break,
                },
                None => {
                    if i > 2 && m.stop(ops, nsd, &s)? {
                        break;
                    }
                    patch
                }
            };
            m.advance(ops, nsd, &s, &fed, l + i)?;
            prev = fed;
        }
        ops.gpu.sync()?;
        Ok((out, context))
    }

    /// Patches [n][4 x 64] to 48 kHz samples, the first `context` patches' samples cut
    pub fn decode(&self, patches: &[Vec<f32>], context: usize) -> Result<Vec<f32>> {
        let t = patches.len() * PATCH;
        let z: Vec<f32> = (0..FEAT).flat_map(|c| patches.iter().flat_map(move |p| (0..PATCH).map(move |k| p[k * FEAT + c]))).collect();
        let wav = self.vae.decode(&self.ops, &z, 400, 32)?;
        let cut = (context * PATCH * self.vae.hop_out).min(wav.len());
        let _ = t;
        Ok(wav[cut..].to_vec())
    }

    /// What a request asks for
    pub fn spec(&self, req: &AudioRequest) -> Result<Spec> {
        if req.prompt.trim().is_empty() {
            return Err(Error("the text is empty".into()));
        }
        let text = match req.instructions.as_deref().map(str::trim).filter(|i| !i.is_empty()) {
            // a design, or a clone's style: in parentheses ahead of the text
            Some(i) => format!("({}){}", i.trim_start_matches('(').trim_end_matches(')'), req.prompt.trim()),
            None => req.prompt.trim().to_string(),
        };
        let (reference, prompt) = match &req.reference {
            None => (None, None),
            Some(r) => {
                let w = resample(&r.samples, r.rate, 16000);
                match r.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
                    // the ultimate clone: the recording both as the reference and as the prompt it continues
                    Some(t) => (Some(w.clone()), Some((w, t.to_string()))),
                    None => (Some(w), None),
                }
            }
        };
        Ok(Spec { text, reference, prompt })
    }
}

impl AudioEngine for VoxCpm2 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_PATCHES as f32 / PPS, steps: 10, cfg: 2.0, rate: 48000 }
    }
    fn speech(&self) -> Option<Speech> {
        Some(Speech { voices: Vec::new(), languages: Vec::new(), instructions: true, design: true, clone: true, clone_needs_text: false,
                      clone_takes_text: true, requires: None })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let t0 = Instant::now();
        let d = self.defaults();
        if req.voice.as_deref().is_some_and(|v| !v.is_empty()) {
            return Err(Error("voice: VoxCPM2 has no built-in voices (describe one in instructions, or send a recording)".into()));
        }
        let s = self.spec(req)?;
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let p = self.prompt(&s)?;
        let ratio: f32 = req.extra.get("max-ratio").map(|v| v.parse().map_err(|_| Error(format!("max-ratio: {v}?")))).transpose()?.unwrap_or(6.0);
        let seconds = req.seconds.unwrap_or(d.seconds);
        let max = ((p.target as f32 * ratio + 10.0) as usize).min((seconds * PPS).ceil() as usize).clamp(4, MAX_PATCHES);
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.unwrap_or(d.cfg);
        let mut rng = Rng::new(req.seed);
        let (patches, context) = self.generate_patches(&p, max, steps, cfg, &mut |_| (0..FEAT * PATCH).map(|_| rng.normal()).collect(), t0, progress, None, None)?;
        progress(Step { phase: "decode", at: 0, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let samples = self.decode(&patches, context)?;
        progress(Step { phase: "decode", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        Ok(Audio { rate: 48000, channels: 1, samples })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} ({}, loaded {:.0} s ago)", self.ops.gpu.name, if self.int8 { "int8" } else { "half" }, self.loaded.elapsed().as_secs_f64())]
    }
}

/// A checkpoint's directory as downloaded: its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (role, name) in [("model", "model.safetensors"), ("config", "config.json"), ("tokenizer", "tokenizer.json"), ("vae", "audiovae.pth")] {
        let p = dir.join(name);
        if !p.exists() {
            return Err(Error(format!("{}: no {name}", dir.display())));
        }
        f.insert(role.into(), p);
    }
    Ok(f)
}
