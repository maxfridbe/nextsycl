//! HunyuanVideo-1.5 on Intel Arc (SYCL): text to video at 480p, the whole pipeline on one GPU - the 480p T2V DiT
//! (cfg-distilled: one pass a step; or the base model with CFG over a negative prompt), the step-distilled I2V DiT
//! later.
//!
//! ```text
//!   prompt -> the video template (PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, chat-formatted) -> Qwen2.5-VL-7B
//!             (crates/hunyuan-text: hidden_states[-3], the template's head dropped)
//!          -> its quoted text "..." / “...” -> byT5-small (the glyph tokens)
//!   noise [F, h, w, 32] (torch.randn's draws from the seed, bfloat16 as the pipeline makes them)
//!     -> Euler over the shifted schedule (shift 5), the DiT's velocity each step (dit.rs: 54 double blocks; CFG on the
//!        base model)
//!   latents -> the 3-D causal VAE decoder (vae.rs: 16x in space, 4x in time; tiles of 256 pixels) -> frames -> mp4
//! ```
//!
//! The files are the official ones: the DiT's float32 safetensors (quantized to Q8_0 as it loads), the VAE's; the
//! text encoders are HunyuanImage 2.1's GGUFs (the same Qwen2.5-VL-7B and Glyph-SDXL-v2 byT5: model-cards/video).
//! `check` compares each stage with reference/hunyuanvideo15/ref.py's dumps.

pub mod check;
pub mod dit;
mod ffi;
pub mod movieroll;
pub mod prof;
pub mod roll;
pub mod vae;
pub mod vision;
mod w;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::Picture;
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_hunyuan_text::{byt5, te};
use nextsycl_video::{At, Clip, Defaults, EngineOption, Error, JobCtl, LoadOptions, ModelFiles, Progress, Result, Sampler, Schedule, VideoEngine,
                     VideoKind, VideoRequest};
use serde_json::{json, Value};

pub const ARCH: &str = "hunyuanvideo-1.5";
/// the denoiser (the transformer's safetensors), the text encoders, the VAE (its safetensors); an I2V card adds
/// `vision` (SigLIP so400m-patch14-384's safetensors: the image encoder)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "byt5", "vae"];

pub(crate) const GIB: f64 = (1u64 << 30) as f64;

/// The options it takes (`--opt-NAME`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "cfg", env: "NS_HV15_CFG", value: "X", help: "the classifier-free guidance scale (the base model's 6; 1: one pass a step, the cfg-distilled card)", at: At::Both },
    EngineOption { name: "steps", env: "NS_HV15_STEPS", value: "N", help: "the denoising steps when a request does not say (default 50: what the cfg-distilled card needs)", at: At::Load },
    EngineOption { name: "shift", env: "NS_HV15_SHIFT", value: "X", help: "the schedule's shift (default 5, the 480p cards')", at: At::Both },
    EngineOption { name: "negative", env: "", value: "TEXT", help: "the negative prompt of a guided run (default empty, as the pipeline)", at: At::Request },
    EngineOption { name: "attention", env: "NS_HV15_ATTN", value: "KERNEL",
                   help: "the DiT's attention: half (the silo's, the default), half:N (its tile N), shared (the shared kernels' flash), sage (int8 q and k: approximate)",
                   at: At::Both },
    EngineOption { name: "cache", env: "NS_HV15_CACHE", value: "SPEC",
                   help: "step caching: off (default), deepcache (upstream's: steps 11-44 of 50 reuse all but the last block every 3 in 4; scaled to the steps), \
                          deepcache:START,END,INTERVAL[,BLOCKS]",
                   at: At::Both },
];

/// DeepCache's schedule (upstream's --cache_start_step / --cache_end_step / --cache_step_interval / --no_cache_block_id
/// as the last `blocks` double blocks)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cache {
    pub start: usize,
    pub end: usize,
    pub interval: usize,
    pub blocks: usize,
}

impl Cache {
    /// off | deepcache | deepcache:START,END,INTERVAL[,BLOCKS]; plain deepcache is upstream's 11, 45, 4 (block 53) for
    /// 50 steps, its start and end scaled to other step counts
    pub fn parse(v: &str, steps: usize) -> std::result::Result<Option<Cache>, String> {
        let v = v.trim();
        if v.is_empty() || v == "off" || v == "0" {
            return Ok(None);
        }
        if v == "deepcache" || v == "1" || v == "on" {
            let at = |s: usize| (s * steps + 25) / 50;
            return Ok(Some(Cache { start: at(11), end: at(45), interval: 4, blocks: 1 }));
        }
        let bad = || format!("cache={v}: off, deepcache, or deepcache:START,END,INTERVAL[,BLOCKS]");
        let nums: Vec<usize> = v.strip_prefix("deepcache:").ok_or_else(bad)?.split(',').map(|x| x.trim().parse().map_err(|_| bad())).collect::<std::result::Result<_, _>>()?;
        match nums[..] {
            [start, end, interval] => Ok(Some(Cache { start, end, interval: interval.max(1), blocks: 1 })),
            [start, end, interval, blocks] if (1..=dit::DOUBLES).contains(&blocks) => Ok(Some(Cache { start, end, interval: interval.max(1), blocks })),
            _ => Err(bad()),
        }
    }
}

/// This engine's registry entry
pub fn kind() -> VideoKind {
    VideoKind { archs: &[ARCH], name: "HunyuanVideo-1.5 (8.3B DiT with Q8_0 products, Qwen2.5-VL + byT5 text, 3-D causal VAE; 480p text to video)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn VideoEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(HunyuanVideo15::load(files, gpu, o, log)?))
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// The video template (PROMPT_TEMPLATE_ENCODE_VIDEO_JSON through Qwen2.5-VL's chat template, the generation prompt
/// added; the system text keeps the source's line continuations as runs of spaces)
pub fn template() -> te::Template {
    let system = "You are a helpful assistant. Describe the video by detailing the following aspects:         1. The main content and theme of the video.         2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         4. background environment, light, style and atmosphere.         5. camera angles, movements, and transitions used in the video.";
    te::Template { head: format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"), tail: "<|im_end|>\n<|im_start|>assistant\n".into(), crop: 0,
                   max_tokens: 0, empty_as_space: true }
}

/// The 480p buckets (generate_crop_size_list(640, 16)): (height, width)
pub fn buckets() -> Vec<(usize, usize)> {
    let n = 1600usize;
    let (mut wp, mut hp) = (n, 1usize);
    let mut v = Vec::new();
    while wp > 0 {
        if wp.max(hp) as f64 / wp.min(hp) as f64 <= 4.0 {
            v.push((wp * 16, hp * 16));
        }
        if (hp + 1) * wp <= n {
            hp += 1;
        } else {
            wp -= 1;
        }
    }
    v
}

/// The bucket nearest in aspect to width x height (get_closest_ratio: the first of equals): (width, height)
pub fn canvas(width: usize, height: usize) -> (usize, usize) {
    let ar = height as f64 / width.max(1) as f64;
    let b = buckets();
    let mut best = 0;
    let mut bd = f64::INFINITY;
    for (i, (h, w)) in b.iter().enumerate() {
        let r = ((*h as f64 / *w as f64) * 1e5).round() / 1e5;
        let d = (r - ar).abs();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    (b[best].1, b[best].0)
}

/// Frames for `seconds` at 24 fps: 4k + 1, rounded up (121 for 5 s)
pub fn frames_for(seconds: f32) -> usize {
    let f = ((seconds.max(0.04) * 24.0).round() as usize).max(1);
    (f - 1).div_ceil(4) * 4 + 1
}

/// The pipeline's starting noise: torch.randn((1, 32, F, h, w), generator seeded, bfloat16) - the draws in that order,
/// rounded to bfloat16 - as tokens [F h w, 32]
pub fn noise(seed: u64, f: usize, h: usize, w: usize) -> Vec<f32> {
    let n = f * h * w;
    let v = nextsycl_diffusion::torch_rng::Mt19937::new(seed).randn(n * dit::LATENT);
    let bf = |x: f32| -> f32 {
        let b = x.to_bits();
        let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
        f32::from_bits(r & 0xffff_0000)
    };
    let mut out = vec![0f32; n * dit::LATENT];
    for c in 0..dit::LATENT {
        for p in 0..n {
            out[p * dit::LATENT + c] = bf(v[c * n + p]);
        }
    }
    out
}

/// The shifted schedule: linspace(1, 0, steps + 1), sigma' = shift sigma / (1 + (shift - 1) sigma), in float32
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let s = 1.0 - i as f32 / steps as f32;
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect()
}

/// A clip made: its frames planar [3, F, H, W] in [0, 1], the last latents, its timings and the card's peak use
pub struct Made {
    pub px: Vec<f32>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub latents: Vec<f32>,
    pub tokens: usize,
    pub steps: usize,
    pub denoise_s: f64,
    pub decode_s: f64,
    pub peak: u64,
}

/// A prompt encoded for the DiT
pub struct Encoded {
    pub text: dit::Text,
}

pub struct HunyuanVideo15 {
    pub nsd: Nsd,
    k: ffi::K,
    /// the text encoders (Qwen2.5-VL, byT5): loaded with the engine; MovieRoll drops them for a film's run (4.2 GiB)
    /// and they load again for the next prompt
    pub text: std::sync::Mutex<Option<(te::TextEncoder, byt5::Byt5)>>,
    text_files: (PathBuf, PathBuf, PathBuf),
    pub dit: dit::Dit,
    pub vae: vae::Vae,
    /// I2V's image encoder (a card with a vision file)
    pub vision: Option<vision::Vision>,
    cfg: f32,
    steps: usize,
    shift: f32,
    /// NS_HV15_CACHE (a request's "cache" over it)
    cache: String,
    /// NS_HV15_ATTN (a request's "attention" over it)
    attention: String,
    loaded: Instant,
    load_s: f64,
    /// the most of the card in use seen during the last clip (total less the driver's free): what a card must hold
    peak: std::sync::atomic::AtomicU64,
}

impl HunyuanVideo15 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<HunyuanVideo15> {
        Self::load_with(files, gpu, o, false, log)
    }

    /// `load`, with the VAE's encoder loaded even without a vision file (MovieRoll's image to video: the picture is
    /// the first chunk, no SigLIP)
    pub fn load_with(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, encoder: bool, log: &mut dyn FnMut(String)) -> Result<HunyuanVideo15> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("video");
        let size = |r: &str| -> Result<u64> { std::fs::metadata(role(files, r)?).map(|m| m.len()).map_err(|e| Error(format!("{r}: {e}"))) };
        // the plan before anything is allocated (the driver has no out-of-memory error): the DiT's float32 file as Q8_0
        // (a quarter, plus scales), 26 of the text encoder's 28 layers, byT5 in float32, the VAE's decoder in half (an
        // eighth of the float32 file: half of it the decoder) and its reordered copy, I2V's SigLIP tower (half of its
        // file) and encoder, and the work of a 480p 5 s clip: 3.6 GiB measured over the 15.1 GiB loaded on the B70 (a
        // CFG pass twice that), 5 GiB here (docs/ports/hunyuanvideo15.md)
        let vision = if files.contains_key("vision") { size("vision")? / 2 + size("vae")? / 16 } else { 0 };
        let need = size("denoiser")? * 34 / 128 + size("text-encoder")? * 26 / 28 + size("byt5")? * 2 + size("vae")? / 4 + vision + (5u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / GIB, gpu.name, free as f64 / GIB)));
        }
        let nsd = Nsd::new(gpu)?;
        let k = ffi::K::new(gpu)?;
        let setting = |name: &str| o.setting(name).filter(|v| !v.is_empty());
        let num = |name: &str, d: f32| -> Result<f32> { setting(name).map_or(Ok(d), |v| v.parse().map_err(|_| Error(format!("{name}={v}: a number")))) };
        let cfg = num("NS_HV15_CFG", 1.0)?;
        let steps = num("NS_HV15_STEPS", 50.0)? as usize;
        let shift = num("NS_HV15_SHIFT", 5.0)?;
        let cache = setting("NS_HV15_CACHE").unwrap_or_default().to_string();
        Cache::parse(&cache, steps).map_err(Error)?;
        let attention = setting("NS_HV15_ATTN").unwrap_or_default().to_string();
        let st = w::St::open(role(files, "denoiser")?)?;
        let dit = dit::Dit::load(&st, &k, log)?;
        drop(st);
        dit::Attn::parse(Some(&attention), dit.tiles).map_err(Error)?;
        let text_files = (role(files, "text-encoder")?.to_path_buf(), role(files, "tokenizer")?.to_path_buf(), role(files, "byt5")?.to_path_buf());
        let (te, byt5) = Self::load_text(&nsd, &k, &text_files, log)?;
        let i2v = files.contains_key("vision");
        if i2v && !dit.has_vision() {
            return Err(Error(format!("{ARCH}: a vision file with a DiT that has no vision projection")));
        }
        let vae = vae::Vae::load(role(files, "vae")?, &k, i2v || encoder, log)?;
        let vision = if i2v { Some(vision::Vision::load(role(files, "vision")?, &k, log)?) } else { None };
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: loaded in {load_s:.1} s (cfg {cfg}, {steps} steps, shift {shift}, attention {:?}{})",
                    dit::Attn::parse(Some(&attention), dit.tiles).unwrap_or(dit.attn), if cache.is_empty() { String::new() } else { format!(", cache {cache}") }));
        Ok(HunyuanVideo15 { nsd, k, text: std::sync::Mutex::new(Some((te, byt5))), text_files, dit, vae, vision, cfg, steps, shift, cache, attention,
                            loaded: Instant::now(), load_s, peak: Default::default() })
    }

    fn load_text(nsd: &Nsd, k: &ffi::K, f: &(PathBuf, PathBuf, PathBuf), log: &mut dyn FnMut(String)) -> Result<(te::TextEncoder, byt5::Byt5)> {
        let kern: Arc<dyn nextsycl_hunyuan_text::Kern> = Arc::new(k.clone());
        let mut te = te::TextEncoder::load(&f.0, &f.1, nsd, kern.clone(), template(), log)?;
        // the pipeline's text_len: 1000 tokens after the crop
        te.template.max_tokens = te.template.crop + 1000;
        let byt5 = byt5::Byt5::load(&f.2, nsd, kern, log)?;
        Ok((te, byt5))
    }

    /// The text encoders, loaded again when they were dropped
    pub fn with_text<R>(&self, f: impl FnOnce(&te::TextEncoder, &byt5::Byt5) -> Result<R>) -> Result<R> {
        let mut g = self.text.lock().map_err(|_| Error("the text encoders' lock".into()))?;
        if g.is_none() {
            *g = Some(Self::load_text(&self.nsd, &self.k, &self.text_files, &mut |l| eprintln!("{l}"))?);
        }
        let (te, b) = g.as_ref().ok_or_else(|| Error("no text encoders".into()))?;
        f(te, b)
    }

    /// Their memory back to the card (the next `encode` loads them again)
    pub fn unload_text(&self) -> Result<()> {
        self.nsd.wait()?;
        *self.text.lock().map_err(|_| Error("the text encoders' lock".into()))? = None;
        Ok(())
    }

    /// The card's memory in use now, kept as the clip's peak when it is the most seen
    pub(crate) fn note(&self) -> u64 {
        let used = self.nsd.gpu.memory().map_or(0, |(t, f)| t - f.unwrap_or(t));
        self.peak.fetch_max(used, std::sync::atomic::Ordering::Relaxed);
        used
    }

    /// The prompt through both text encoders and the DiT's prompt-only layers
    pub fn encode(&self, prompt: &str, vision: Option<&[f32]>) -> Result<Encoded> {
        let (q, b) = self.with_text(|te, b5| {
            let q = te.encode(&self.nsd, prompt)?;
            let b = match byt5::quoted(prompt, byt5::Quotes::InOrder) {
                Some(t) => Some(b5.encode(&self.nsd, &byt5::ids(&t, 256))?),
                None => None,
            };
            Ok((q, b))
        })?;
        Ok(Encoded { text: self.dit.text(&self.nsd, &q, b.as_deref(), vision)? })
    }

    /// Denoise `x` (float32 [F h w, 32]) over `sigmas` with Euler on the GPU; `neg` the guided run's negative and its
    /// scale; `cond` float32 [n, 33] (the condition latents and the mask; zeros for T2V). Returns the latents.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(&self, pos: &Encoded, neg: Option<(&Encoded, f32)>, x: &[f32], grid: (usize, usize, usize), cond: &[f32], sigmas: &[f32],
                   cache: Option<Cache>, attn: dit::Attn, each: &mut dyn FnMut(usize, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let nsd = &self.nsd;
        let steps = sigmas.len() - 1;
        let ts: Vec<(f32, Option<f32>)> = (0..steps)
            .map(|i| (sigmas[i] * 1000.0, self.dit.meanflow.then(|| if i + 1 < steps { sigmas[i + 1] * 1000.0 } else { 0.0 })))
            .collect();
        let st = self.dit.steps(nsd, &ts)?;
        let mut texts = vec![&pos.text];
        if let Some((e, _)) = neg {
            texts.push(&e.text);
        }
        let batch = self.dit.batch(nsd, &texts, grid)?;
        let lat = DevBuf::from_f32(&nsd.gpu, x)?;
        let cond = DevBuf::from_f32(&nsd.gpu, cond)?;
        let n = x.len();
        let dc = match cache {
            Some(c) => Some((self.dit.deep_cache(nsd, &batch, c.blocks)?, dit::deep_cache_skips(steps, c.start, c.end, c.interval))),
            None => None,
        };
        let all = [(0, grid.0 * grid.1 * grid.2, 0)];
        for i in 0..steps {
            let lats: Vec<&DevBuf> = texts.iter().map(|_| &lat).collect();
            let spans = [(all[0].0, all[0].1, i)];
            let v = self.dit.forward_cached(nsd, &st, i, &spans, &batch, &lats, &cond, dc.as_ref().map(|(c, skip)| (c, skip[i])), attn,
                                              &mut |_, _, _, _| Ok(()))?;
            let v = match neg {
                Some((_, scale)) => {
                    // v_neg + scale (v - v_neg)
                    self.k.axpy(v[0].fp(), v[1].fp(), n, -1.0)?;
                    self.k.axpy(v[1].fp(), v[0].fp(), n, scale)?;
                    v.into_iter().nth(1).ok_or_else(|| Error("no velocity".into()))?
                }
                None => v.into_iter().next().ok_or_else(|| Error("no velocity".into()))?,
            };
            self.k.axpy(lat.fp(), v.fp(), n, sigmas[i + 1] - sigmas[i])?;
            nsd.gpu.sync()?;
            each(i, &lat)?;
        }
        Ok(lat)
    }

    /// A clip from a prompt
    #[allow(clippy::too_many_arguments)]
    pub fn make(&self, prompt: &str, negative: &str, image: Option<&Picture>, width: usize, height: usize, frames: usize, steps: usize, cfg: f32, shift: f32,
                seed: u64, cache: Option<Cache>, attn: dit::Attn, progress: &mut dyn FnMut(&'static str, usize, usize) -> Result<()>, log: &mut dyn FnMut(String)) -> Result<Made> {
        // I2V: the canvas the picture's shape picks (get_closest_resolution_given_reference_image), else the request's
        let (w, h) = match image {
            Some(p) => canvas(p.width as usize, p.height as usize),
            None => canvas(width, height),
        };
        let f = (frames - 1) / 4 + 1;
        let grid = (f, h / 16, w / 16);
        let t0 = Instant::now();
        self.peak.store(0, std::sync::atomic::Ordering::Relaxed);
        let at_start = self.note();
        progress("encode", 0, 0)?;
        let n = grid.0 * grid.1 * grid.2;
        let mut cond = vec![0f32; n * (dit::IN - dit::LATENT)];
        let tokens = match image {
            Some(p) => {
                let v = self.vision.as_ref().ok_or_else(|| Error(format!("{ARCH}: this card makes video from text only (an I2V card takes a picture)")))?;
                let rgb = p.rgb();
                let px = vision::cover(&rgb.data, rgb.width as usize, rgb.height as usize, w, h)?;
                let lat = self.vae.encode(&self.nsd, &px, h, w)?;
                // the first frame's latent and the mask (1) on frame 0's rows; zeros after
                for (r, row) in lat.chunks_exact(dit::LATENT).enumerate() {
                    cond[r * (dit::IN - dit::LATENT)..r * (dit::IN - dit::LATENT) + dit::LATENT].copy_from_slice(row);
                    cond[r * (dit::IN - dit::LATENT) + dit::LATENT] = 1.0;
                }
                Some(v.encode(&self.nsd, &vision::siglip_pixels(&px, w, h)?)?)
            }
            None => None,
        };
        let pos = self.encode(prompt, tokens.as_deref())?;
        let neg = if cfg > 1.0 { Some(self.encode(negative, tokens.as_deref())?) } else { None };
        log(format!("{ARCH}: {w}x{h}, {frames} frames ({} x {} x {} latents, {} tokens), {steps} steps, cfg {cfg}, shift {shift}, seed {seed}; text in {:.1} s",
                    grid.0, grid.1, grid.2, grid.0 * grid.1 * grid.2, t0.elapsed().as_secs_f64()));
        let x = noise(seed, grid.0, grid.1, grid.2);
        let sig = sigmas(steps, shift);
        let t1 = Instant::now();
        if let Some(c) = cache {
            log(format!("{ARCH}: deepcache - steps {} to {} (all but every {}th) reuse the rows block {} got on the last full step; {} of {steps} steps cached",
                        c.start, c.end.saturating_sub(1), c.interval,
                        dit::DOUBLES - c.blocks, dit::deep_cache_skips(steps, c.start, c.end, c.interval).iter().filter(|x| **x).count()));
        }
        log(format!("{ARCH}: attention {attn:?}"));
        let lat = self.denoise(&pos, neg.as_ref().map(|e| (e, cfg)), &x, grid, &cond, &sig, cache, attn, &mut |i, _| {
            self.note();
            progress("denoise", i + 1, steps)
        })?;
        let lat = lat.to_f32()?;
        let ds = t1.elapsed().as_secs_f64();
        log(format!("{ARCH}: {steps} steps in {ds:.1} s ({:.2} s a step)", ds / steps as f64));
        for l in self.dit.prof.report(steps) {
            log(l);
        }
        progress("decode", 0, 0)?;
        let t2 = Instant::now();
        let (v, to, oh, ow) = self.vae.decode(&self.nsd, &lat, grid.0, grid.1, grid.2, &mut |s| {
            let used = self.note();
            log(format!("decode: {s} ({:.1} GiB in use)", used as f64 / GIB))
        })?;
        log(format!("{ARCH}: decoded {to} frames {ow}x{oh} in {:.1} s; the card's peak {:.1} GiB in use ({:.1} GiB loaded at the start)", t2.elapsed().as_secs_f64(),
                    self.peak.load(std::sync::atomic::Ordering::Relaxed) as f64 / GIB, at_start as f64 / GIB));
        let plane = oh * ow;
        let mut px = vec![0f32; 3 * to * plane];
        for t in 0..to {
            for p in 0..plane {
                for c in 0..3 {
                    px[(c * to + t) * plane + p] = (v[(t * plane + p) * 3 + c] * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
        }
        Ok(Made { px, frames: to.min(frames), height: oh, width: ow, latents: lat, tokens: n, steps, denoise_s: ds, decode_s: t2.elapsed().as_secs_f64(),
                  peak: self.peak.load(std::sync::atomic::Ordering::Relaxed) })
    }

    fn generate_job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        let s = |k: &str| spec.get(k).and_then(Value::as_str);
        let n = |k: &str| spec.get(k).and_then(Value::as_f64);
        let prompt = s("prompt").ok_or_else(|| Error("generate: no prompt".into()))?;
        let out = PathBuf::from(s("out").ok_or_else(|| Error("generate: no out".into()))?);
        let (w0, h0) = canvas(848, 480);
        let width = n("width").map_or(w0, |x| x as usize);
        let height = n("height").map_or(h0, |x| x as usize);
        let frames = match n("frames") {
            Some(f) => frames_for(f as f32 / 24.0),
            None => frames_for(n("seconds").unwrap_or(5.0) as f32),
        };
        let steps = n("steps").map_or(self.steps, |x| x as usize).clamp(1, 200);
        let cfg = n("cfg").map_or(self.cfg, |x| x as f32);
        let shift = n("shift").map_or(self.shift, |x| x as f32);
        let seed = n("seed").map_or(0, |x| x as i64 as u64);
        let negative = s("negative").or(s("negative_prompt")).unwrap_or("");
        let cache = Cache::parse(s("cache").unwrap_or(&self.cache), steps).map_err(Error)?;
        let attn = dit::Attn::parse(Some(s("attention").unwrap_or(&self.attention)), self.dit.tiles).map_err(Error)?;
        // I2V: the picture the clip starts from (the studio's first_frame)
        let image = match s("image").or(s("first_frame")).filter(|p| !p.is_empty()) {
            Some(p) => Some(Picture::decode(&std::fs::read(p).map_err(|e| Error(format!("{p}: {e}")))?).map_err(|e| Error(format!("{p}: {e}")))?),
            None => None,
        };
        let t0 = Instant::now();
        let JobCtl { log, cancel, progress: report } = ctl;
        let mut progress = |stage: &'static str, at: usize, of: usize| -> Result<()> {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error("cancelled".into()));
            }
            if let (Some(p), true) = (report.as_mut(), stage == "denoise") {
                p(at, of);
            }
            Ok(())
        };
        let m = self.make(prompt, negative, image.as_ref(), width, height, frames, steps, cfg, shift, seed, cache, attn, &mut progress, &mut |l| log(l))?;
        // the merge gate's quick run (tooling/gate.sh): the last latents and the frames to hash, and two speeds as its
        // [prompt ...] lines: the DiT's latent tokens a second over the steps, the decoder's frames' pixels a second
        if let Some(p) = s("logits-out") {
            std::fs::write(p, m.latents.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| Error(format!("{p}: {e}")))?;
        }
        if let Some(p) = s("gen-out") {
            std::fs::write(p, m.px.iter().map(|v| (v * 255.0).round() as u8).collect::<Vec<u8>>()).map_err(|e| Error(format!("{p}: {e}")))?;
            log(format!("[prompt {} latent tokens x {} steps in {:.2} s ({:.1} tok/s)]", m.tokens, m.steps, m.denoise_s,
                        (m.tokens * m.steps) as f64 / m.denoise_s));
            log(format!("[prompt {} frames of {}x{} decoded in {:.2} s ({:.1} tok/s)] (the second: thousands of pixels a second)", m.frames, m.width,
                        m.height, m.decode_s, (m.frames * m.width * m.height) as f64 / 1000.0 / m.decode_s));
        }
        // the clip's folder must be there before ffmpeg starts: ffmpeg failing to open its output closes the pipe the
        // frames go down, and the write's SIGPIPE ends the worker instead of answering with an error (2026-11-23: a
        // clip sent with a folder the daemon's container does not have)
        if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| Error(format!("{}: {e} (where the clip goes)", d.display())))?;
        }
        nextsycl_h3::media::write_mp4(&out, &nextsycl_h3::media::Frames { px: &m.px, count: m.frames, height: m.height, width: m.width, fps: 24 }, None,
                                      &mut |l| log(l))
            .map_err(|e| Error(e.0))?;
        Ok(json!({"out": out, "width": m.width, "height": m.height, "frames": m.frames, "fps": 24, "seconds": t0.elapsed().as_secs_f64(), "steps": steps,
                  "cfg": cfg, "seed": seed, "attention": format!("{attn:?}"), "cache": cache.map(|c| format!("deepcache:{},{},{},{}", c.start, c.end, c.interval, c.blocks)), "denoise_s": m.denoise_s, "decode_s": m.decode_s, "peak_gib": m.peak as f64 / GIB}))
    }
}

impl VideoEngine for HunyuanVideo15 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 848, height: 480, seconds: 5.0, fps: 24.0, steps: self.steps as u32, cfg: self.cfg, sampler: Sampler::Euler, schedule: Schedule::Shift,
                   shift: self.shift, upscale: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn jobs(&self) -> &'static [&'static str] {
        &["generate", "check"]
    }
    fn run_job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        match spec.get("kind").and_then(Value::as_str) {
            Some("generate") => self.generate_job(spec, ctl),
            Some("check") => check::job(self, spec, ctl),
            k => Err(Error(format!("{ARCH}: no job {k:?} (generate, check)"))),
        }
    }
    fn device_stats(&self) -> Option<(u64, u64, Option<u64>)> {
        let (total, free) = self.nsd.gpu.memory().ok()?;
        Some((total - free.unwrap_or(total), total, free))
    }
    fn generate(&self, req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> Result<Clip> {
        let d = self.defaults();
        let t0 = Instant::now();
        let frames = frames_for(req.seconds.unwrap_or(d.seconds));
        let cfg = req.cfg.unwrap_or(d.cfg);
        let mut p = |stage: &'static str, at: usize, of: usize| -> Result<()> {
            if cancel() {
                return Err(Error("cancelled".into()));
            }
            progress(Progress { stage, at: at as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() });
            Ok(())
        };
        let negative = req.negative.clone().or_else(|| req.extra.get("negative").cloned()).unwrap_or_default();
        let image = req.keyframes.first().map(|k| k.picture.clone());
        let steps = req.steps.unwrap_or(d.steps) as usize;
        let cache = Cache::parse(req.extra.get("cache").map(String::as_str).unwrap_or(&self.cache), steps).map_err(Error)?;
        let attn = dit::Attn::parse(Some(req.extra.get("attention").map(String::as_str).unwrap_or(&self.attention)), self.dit.tiles).map_err(Error)?;
        let m = self.make(&req.prompt, &negative, image.as_ref(), req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize, frames,
                          steps, cfg, req.shift.unwrap_or(d.shift), req.seed, cache, attn, &mut p, &mut |l| eprintln!("{l}"))?;
        progress(Progress { stage: "mux", at: 0, of: 0, seconds: t0.elapsed().as_secs_f64() });
        nextsycl_h3::media::write_mp4(&req.out, &nextsycl_h3::media::Frames { px: &m.px, count: m.frames, height: m.height, width: m.width, fps: 24 }, None,
                                      &mut |l| eprintln!("{l}"))
            .map_err(|e| Error(e.0))?;
        Ok(Clip { path: req.out.clone(), width: m.width as u32, height: m.height as u32, frames: m.frames as u32, fps: 24.0, audio: false })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (loaded {:.0} s ago; {:.1} GiB of weights)", self.nsd.gpu.name, self.loaded.elapsed().as_secs_f64(),
                     (self.dit.bytes + self.vae.bytes + self.text.lock().ok().and_then(|g| g.as_ref().map(|(t, b)| t.bytes + b.bytes)).unwrap_or(0)) as f64 / GIB)]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_cache_option_is_upstreams_schedule() {
        use super::Cache;
        assert_eq!(Cache::parse("", 50), Ok(None));
        assert_eq!(Cache::parse("off", 50), Ok(None));
        assert_eq!(Cache::parse("deepcache", 50), Ok(Some(Cache { start: 11, end: 45, interval: 4, blocks: 1 })));
        // scaled: 20 steps start at 4.4 -> 4, end at 18
        assert_eq!(Cache::parse("deepcache", 20), Ok(Some(Cache { start: 4, end: 18, interval: 4, blocks: 1 })));
        assert_eq!(Cache::parse("deepcache:5,40,3,2", 50), Ok(Some(Cache { start: 5, end: 40, interval: 3, blocks: 2 })));
        assert!(Cache::parse("deepcache:5,40", 50).is_err());
        assert!(Cache::parse("deepcache:5,40,3,99", 50).is_err());
        assert!(Cache::parse("teacache", 50).is_err());
    }

    #[test]
    fn the_480p_buckets_are_the_pipelines() {
        // 16:9 -> 848 x 480, 9:16 -> 480 x 848, 1:1 -> 640 x 640
        assert_eq!(super::canvas(16, 9), (848, 480));
        assert_eq!(super::canvas(9, 16), (480, 848));
        assert_eq!(super::canvas(640, 640), (640, 640));
        assert_eq!(super::canvas(848, 480), (848, 480));
    }

    #[test]
    fn frames_are_4k_plus_1() {
        assert_eq!(super::frames_for(5.0), 121);
        assert_eq!(super::frames_for(0.375), 9);
        assert_eq!(super::frames_for(1.0), 25);
    }

    #[test]
    fn the_schedule_is_shifted_linspace() {
        let s = super::sigmas(2, 5.0);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 1.0);
        assert!((s[1] - 2.5 / 3.0).abs() < 1e-6);
        assert_eq!(s[2], 0.0);
    }
}
