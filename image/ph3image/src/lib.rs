//! P-H3-Image (PrunaAI) in Rust and SYCL: an image model built on MiniMax H3 - text to image, and edits with up to 8
//! pictures, in 8 steps with no guidance. Diffusers' pipeline (`modular_pipelines/p_h3_image`, commit 84532a05) over
//! H3's own engine (crates/h3: the 50-block denoiser, the text refiner, the token layout - the same as the video
//! engine's) and this crate's text side and VAE decoder:
//!
//! ```text
//!   prompt (as written; an edit: "<Picture i>: " + each picture's vision tokens, then "Edit <Picture 1>. " + the
//!   prompt) -> Qwen3-VL-4B, hidden states 13, 19, 25, 31, 36 -> the text connector -> [T, 5120]     (te.rs)
//!   one sequence [text | the edit's pictures (VAE latents) | 2 x 2 audio rows | the image's 2 x 2 patches]
//!   noise from the seed (torch's draws: each picture's augmentation, the image, the audio) -> Euler over the
//!   checkpoint's 8 sigmas (1 .. 0.528718, 0), x0 = x + sigma v, the conditioning rows held at t = max(t, 0.999)
//!   latents [24, H/16, W/16] -> the transformer decoder, 256-pixel tiles (vae.rs) -> RGB
//! ```
//!
//! The denoiser is any H3 form the video engine reads (ComfyUI's int8 ConvRot, a GGUF of Q8_0 / Q6_K / Q5_K / Q4_K
//! matrices): P-H3-Image's own file, or MiniMax H3's FL2VA one with Pruna's LoRA on the side path (role `lora`).

pub mod check;
pub mod ffi;
pub mod te;
pub mod vae;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_h3::denoiser::{self, unpack_audio, Conditions, Denoiser, Outer, RefIn, Schedule, Shape, TextRefiner};
use nextsycl_h3::device::{Device, Tensor};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::lora::LoraSet;
use nextsycl_h3::noise::Mt19937;
use nextsycl_h3::safetensors::Checkpoint;
use nextsycl_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule as Sched,
                     SizePresets, Step};

pub const ARCH: &str = "p-h3-image";
/// the denoiser (P-H3-Image's, or H3's FL2VA with `lora`), the text encoder (+ connector, + vision tower), its
/// tokenizer.json, the VAE (H3's encoder, P-H3-Image's decoder), Pruna's LoRA for an H3 denoiser
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "vae", "lora"];

/// The checkpoint's sampling grid (modular_model_index.json `sample_sigmas`): taken as it is - the scheduler's shift
/// of 12 is not applied to given sigmas (`shift_provided_sigmas: false`), as ComfyUI's graph samples it at shift 1
pub const SIGMAS: [f32; 9] = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0];
/// canvas sides are multiples of this (the VAE's 16 x the denoiser's 2 x 2 patches)
pub const GRID: u32 = 32;
/// the default canvas: 8 MP square, where the model is at its best
pub const DEFAULT_SIDE: u32 = 2912;
/// an edit's canvas without a size: picture 1's, scaled into 4 .. 8 MP; the other pictures keep theirs within 1 .. 8 MP
pub const CANVAS_MIN_PIXELS: u64 = 4 * 1024 * 1024;
pub const REFERENCE_MIN_PIXELS: u64 = 1024 * 1024;
pub const REFERENCE_MAX_PIXELS: u64 = 8 * 1024 * 1024;
/// the most pictures an edit takes (picture 1 and seven references)
pub const MAX_PICTURES: usize = 8;
/// the largest canvas (a little over the 8 MP sizes; the memory plan is made for it)
pub const MAX_PIXELS: u64 = 8_912_896;
/// the conditioning rows of an edit are mixed with this much of the picture (and presented at max(t, this))
const COND_AUG: f32 = 0.999;
/// the seed an edit's VAE posterior is sampled under, whatever the request's (keyframe_encode_seed)
const ENCODE_SEED: u64 = 42;
/// audio latents per channel denoised beside the image (and discarded)
const AUDIO_T: usize = 2;
/// what an 8 MP picture's work needs on the card beside the weights (the denoiser's stream and buffers, the decoder's
/// tiles), and the room always left free
const WORK_GIB: f64 = 4.5;
const SPARE_GIB: f64 = 1.5;

/// Pruna's recommended sizes: these aspect ratios at 1, 4 and 8 MP (a budget at a ratio, each side rounded to 32: the
/// 8 MP sizes are their table's, 2912 x 2912 ... 3872 x 2176)
const RATIOS: &[&str] = &["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];
const PIXELS: &[u32] = &[1_048_576, 4_194_304, 8_388_608];

const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "threads", env: "NS_PH3_THREADS", value: "N", help: "threads reading the checkpoints at load (default 8)", at: At::Load },
    EngineOption { name: "sage-min-s", env: "NS_PH3_SAGE_MIN_S", value: "TOKENS", help: "the shortest sequence the silo's SageAttention takes (default 1024; shorter: oneDNN's)", at: At::Load },
    EngineOption { name: "vae-batch", env: "NS_PH3_VAE_BATCH", value: "N", help: "the decoder's tiles decoded together (default 16)", at: At::Load },
    EngineOption { name: "te-resident", env: "NS_PH3_TE_RESIDENT", value: "auto|0|1", help: "keep the text encoder on the card between requests (auto: when the card has room for it beside an 8 MP picture's work)", at: At::Load },
    EngineOption { name: "sigmas", env: "", value: "S,S,..,0", help: "another sampling grid than the checkpoint's 8 steps: strictly falling, ending at 0", at: At::Request },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "P-H3-Image (MiniMax H3's 50-block DiT on one frame, Qwen3-VL-4B + text connector, transformer VAE decoder; Rust + SYCL)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(PH3Image::load(files, gpus, o, log)?))
}

/// An H3 error as the image kind's
pub fn h3e(e: nextsycl_h3::Error) -> Error {
    Error(e.0)
}

/// Python's round (half to even) of `side / GRID`, times GRID, at least GRID: diffusers' `round_canvas`
pub fn round_side(side: f64) -> u32 {
    ((side / GRID as f64).round_ties_even() as u32 * GRID).max(GRID)
}

/// A canvas asked for, rounded to the grid
pub fn round_canvas(w: u32, h: u32) -> (u32, u32) {
    (round_side(w as f64), round_side(h as f64))
}

/// A picture's size kept, or scaled at its aspect to `min` or `max` pixels when its area is outside them, the sides
/// rounded to the grid (diffusers' `_size_in_area_range`)
pub fn size_in_area_range(w: u32, h: u32, min: u64, max: u64) -> (u32, u32) {
    let area = w as f64 * h as f64;
    let scale = (area.clamp(min as f64, max as f64) / area).sqrt();
    (round_side(w as f64 * scale), round_side(h as f64 * scale))
}

/// A budget of `pixels` at aspect `ratio` (W / H), each side rounded to the grid (ComfyUI's Resolution Selector)
pub fn size_at(pixels: u64, ratio: f64) -> (u32, u32) {
    let side = |v: f64| ((v / GRID as f64).round() as u32 * GRID).max(GRID);
    (side((pixels as f64 * ratio).sqrt()), side((pixels as f64 / ratio).sqrt()))
}

/// A request's own sampling grid ("S,S,...,0"): strictly falling, from at most 1, ending at 0
pub fn parse_sigmas(s: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse::<f32>().map_err(|_| Error(format!("sigmas {s}: numbers")))).collect::<Result<_>>()?;
    if v.len() < 2 || v.windows(2).any(|w| w[1] >= w[0]) || v[v.len() - 1] != 0.0 || v[0] > 1.0 {
        return Err(Error(format!("sigmas {s}: at least two, strictly falling from at most 1 to 0")));
    }
    Ok(v)
}

/// One Euler step of the reference scheduler (MiniMaxH3Scheduler.step, float32): x0 = x + (1 - t) v with t = 1 - sigma,
/// then x = r x + (1 - r) x0 with r = sigma_next / sigma
pub fn euler(x: &mut [f32], v: &[f32], sigma: f32, next: f32) {
    let t = 1.0 - sigma;
    let s = 1.0 - t;
    let r = next / sigma;
    for (x, v) in x.iter_mut().zip(v) {
        let x0 = *x + s * *v;
        *x = r * *x + (1.0 - r) * x0;
    }
}

pub struct PH3Image {
    gpu: Arc<Gpu>,
    pub dev: Arc<Device>,
    pub ck: Checkpoint,
    pub blocks: nextsycl_h3::dit::Blocks,
    pub outer: Outer,
    pub lora: Option<LoraSet>,
    pub decoder: vae::Decoder,
    /// the text encoder, when it stays on the card between requests
    te: Mutex<Option<te::TextEncoder>>,
    te_keep: bool,
    files: ModelFiles,
    threads: usize,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl PH3Image {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<PH3Image> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let threads = o.setting("NS_PH3_THREADS").and_then(|v| v.parse().ok()).unwrap_or(8);
        for r in ["text-encoder", "tokenizer", "vae"] {
            role(files, r)?;
        }
        // H3's kernels on the kind's queue of the card: the denoiser and the decoder, the text encoder and the silo ordered
        // on the one queue
        // SAFETY: the core's GPU (its queue's owner) is kept in the engine beside the device.
        let dev = unsafe { Device::on_queue(gpu.queue(), gpu.index) }.map_err(h3e)?;
        let ck = Checkpoint::open(role(files, "denoiser")?).map_err(h3e)?;
        let mut blocks = nextsycl_h3::dit::Blocks::load(&dev, &ck, None, threads).map_err(h3e)?;
        // the silo's attention for the long sequences (dist/silo/libnextsycl-ph3image.so; else the shared kernels')
        if let Some(s) = ffi::silo()? {
            // from 1,024 tokens (a 1 MP picture): the shared kernels keep SageAttention for 8,192 and more, oneDNN's bf16
            // attention below; on this model Sage is 15 % faster a step at 1 and 4 MP (docs/ports/p-h3-image.md)
            let min_s = o.setting("NS_PH3_SAGE_MIN_S").and_then(|v| v.parse().ok()).unwrap_or(1024);
            blocks.attention = Some(ffi::attention(&gpu, s, min_s));
            log(format!("{ARCH}: the {} silo's attention", ffi::SILO));
        }
        let outer = Outer::load(&dev, &ck).map_err(h3e)?;
        denoiser::check_fits(&blocks.cfg, &outer).map_err(h3e)?;
        log(format!("{ARCH}: {} blocks ({}) {:.2} GiB in {:.1} s on {}", blocks.blocks.len(), blocks.kquant.map_or("int8".to_string(), |k| format!("{k:?}")),
                    blocks.load_bytes as f64 / (1u64 << 30) as f64, blocks.load_seconds, dev.name()));
        // Pruna's LoRA on an H3 denoiser (and a request's merged LoRAs), on the side path
        let refiners = ck.entries.keys().filter_map(|k| k.strip_prefix("token_refiner.blocks.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        let mut lora: Option<LoraSet> = None;
        let mut paths: Vec<(std::path::PathBuf, f32)> = files.get("lora").map(|p| (p.clone(), 1.0)).into_iter().collect();
        paths.extend(o.merge_loras.iter().map(|l| (l.path.clone(), l.scale)));
        for (p, scale) in paths {
            let set = LoraSet::load(&dev, &Checkpoint::open(&p).map_err(h3e)?, scale, blocks.blocks.len(), refiners).map_err(h3e)?;
            log(format!("LoRA {} x {scale}: {} layers, rank up to {}", p.display(), set.layers, set.max_rank));
            match lora.as_mut() {
                Some(l) => l.stack(set),
                None => lora = Some(set),
            }
        }
        if lora.as_ref().is_some_and(|l| l.max_rank > nextsycl_h3::dit::MAX_LORA_RANK) {
            return Err(Error(format!("{ARCH}: a LoRA of rank above {}", nextsycl_h3::dit::MAX_LORA_RANK)));
        }
        let mut decoder = vae::Decoder::load(&dev, &Checkpoint::open(role(files, "vae")?).map_err(h3e)?, threads).map_err(h3e)?;
        if let Some(b) = o.setting("NS_PH3_VAE_BATCH").and_then(|v| v.parse::<usize>().ok()).filter(|b| *b > 0) {
            decoder.tile_batch = b;
        }
        log(format!("{ARCH}: VAE decoder {:.2} GiB in {:.1} s", decoder.load_bytes as f64 / (1u64 << 30) as f64, decoder.load_seconds));
        // the text encoder stays when the card has room for it beside a picture's work (it is ~5 GiB: the int8 denoiser,
        // the decoder and it do not all fit a 32 GiB card with an 8 MP picture's activations)
        let te_bytes = std::fs::metadata(role(files, "text-encoder")?).map(|m| m.len()).unwrap_or(0) as f64;
        let free = dev.mem_free().unwrap_or(0) as f64;
        let gib = (1u64 << 30) as f64;
        let te_keep = match o.setting("NS_PH3_TE_RESIDENT").as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => free >= te_bytes * 1.15 + (WORK_GIB + SPARE_GIB) * gib,
        };
        log(format!("{ARCH}: {:.1} GiB free; the text encoder {}", free / gib, if te_keep { "stays on the card" } else { "is loaded for each request" }));
        Ok(PH3Image { gpu, dev, ck, blocks, outer, lora, decoder, te: Mutex::new(None), te_keep, files: files.clone(), threads, loaded: Instant::now(),
                      load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()) })
    }

    /// The text encoder, loaded for a request (it is dropped after it, for the denoiser's activations)
    pub fn text_encoder(&self, log: &mut dyn FnMut(String)) -> Result<te::TextEncoder> {
        te::TextEncoder::load(role(&self.files, "text-encoder")?, role(&self.files, "tokenizer")?, &self.gpu, log)
    }

    /// An edit's pictures as conditioning latents [24, h, w] (normalized), each at its size for the VAE: picture 1 at
    /// `first` (the canvas when none was asked for), the others kept within 1 .. 8 MP
    pub fn reference_latents(&self, pics: &[Picture], first: Option<(u32, u32)>) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        let enc = nextsycl_h3::venc::VideoEncoder::load(&self.dev, &Checkpoint::open(role(&self.files, "vae")?).map_err(h3e)?).map_err(h3e)?;
        let (mean, std) = enc.latent_stats();
        let mut out = Vec::with_capacity(pics.len());
        for (i, p) in pics.iter().enumerate() {
            let rgb = p.rgb();
            let (w, h) = match (i, first) {
                (0, Some(s)) => s,
                _ => size_in_area_range(rgb.width, rgb.height, REFERENCE_MIN_PIXELS, REFERENCE_MAX_PIXELS),
            };
            let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
            let px: Vec<f32> = shown.data.iter().map(|v| *v as f32 / 255.0).collect();
            let m = enc.moments_of_picture(&px, h as usize, w as usize, &mut || Ok(())).map_err(h3e)?;
            let (lh, lw) = (h as usize / 16, w as usize / 16);
            let noise = Mt19937::new(ENCODE_SEED).randn(24 * lh * lw);
            out.push((vae::condition_latent(&m, &noise, mean, std), lh, lw));
        }
        Ok(out)
    }

    /// The latents [24, lh, lw] of one picture: the refined text `text` (bfloat16 [T, hidden]) with `tags`, the edit's
    /// conditioning latents `refs`, the draws from `seed`, the grid `sigmas`; `step(i)` after each step (an error stops)
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, text: Tensor, tags: &[i32], refs: &[(Vec<f32>, usize, usize)], lh: usize, lw: usize, seed: u64, sigmas: &[f32],
                  step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let mut g = Mt19937::new(seed);
        // the draws in the pipeline's order: each picture's augmentation noise, then the image's, then the audio's
        let mixed: Vec<RefIn> = refs
            .iter()
            .map(|(lat, h, w)| {
                let n = g.randn(lat.len());
                RefIn::Image { latent: lat.iter().zip(&n).map(|(l, n)| COND_AUG * l + (1.0 - COND_AUG) * n).collect(), h: *h, w: *w }
            })
            .collect();
        let mut xv = g.randn(24 * lh * lw);
        let mut xa = unpack_audio(&g.randn(2 * AUDIO_T * self.outer.audio_c), self.outer.audio_c, AUDIO_T);
        let cond = Conditions { refs: mixed, premixed: true, seed, ..Conditions::default() };
        let shape = Shape { t: 1, h: lh, w: lw, audio_t: AUDIO_T };
        // one grid for the image and the audio (both schedulers get the same sigmas): no shift between them
        let schedule = Schedule { shift_video: 1.0, shift_audio: 1.0, audio_scale: 1.0 };
        let mut d = Denoiser::new(&self.blocks, &self.outer, text, Some(tags.to_vec()), shape, schedule, &cond).map_err(h3e)?;
        d.lora = self.lora.as_ref();
        for i in 0..sigmas.len() - 1 {
            let (vv, va) = d.net(&xv, &xa, sigmas[i], &mut |_| Ok(())).map_err(h3e)?;
            // the engine returns the head's outputs negated; the scheduler takes them as they are
            let neg = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| -x).collect() };
            euler(&mut xv, &neg(vv), sigmas[i], sigmas[i + 1]);
            euler(&mut xa, &neg(va), sigmas[i], sigmas[i + 1]);
            step(i)?;
        }
        Ok(xv)
    }

    /// The conditioning rows [T, 5120] through the denoiser's projection and token refiner: bfloat16 [T, hidden]
    pub fn refine(&self, cond: &[f32]) -> Result<Tensor> {
        let t = cond.len() / te::COND;
        let c = Tensor::from_bytes(&self.dev, DType::BF16, &[t, te::COND], &denoiser::bf16_bytes(cond)).map_err(h3e)?;
        let refiner = TextRefiner::load(&self.dev, &self.ck, self.threads).map_err(h3e)?;
        let out = refiner.run(&self.blocks.cfg, &c, self.lora.as_ref().map(|l| l.refiner.as_slice())).map_err(h3e)?;
        self.dev.wait().map_err(h3e)?;
        Ok(out)
    }
}

impl ImageEngine for PH3Image {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: DEFAULT_SIDE, height: DEFAULT_SIDE, steps: 8, cfg: 1.0, sampler: Sampler::Euler, schedule: Sched::Simple, shift: 1.0 }
    }
    fn edits(&self) -> bool {
        true
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Sched> {
        vec![Sched::Simple]
    }
    fn guidance(&self) -> bool {
        false
    }
    fn grid(&self) -> u32 {
        GRID
    }
    fn shift(&self) -> bool {
        false
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        match req.extra.get("sigmas") {
            Some(s) => s.split(',').count().saturating_sub(1) as u32,
            None => 8,
        }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let pics: Vec<Picture> = match &req.edit {
            None => Vec::new(),
            Some(e) => {
                if !e.condition || e.strength < 1.0 {
                    return Err(Error(format!("{ARCH}: edits read the pictures as references at strength 1 (no img2img: describe the change instead)")));
                }
                if e.mask.is_some() {
                    return Err(Error(format!("{ARCH}: masks are not part of this model's edits (say what to keep instead)")));
                }
                std::iter::once(e.image.clone()).chain(e.refs.iter().cloned()).collect()
            }
        };
        if pics.len() > MAX_PICTURES {
            return Err(Error(format!("{ARCH}: an edit takes at most {MAX_PICTURES} pictures, got {}", pics.len())));
        }
        if req.negative.is_some() || req.cfg.is_some_and(|c| c != 1.0) {
            return Err(Error(format!("{ARCH}: no guidance (no negative prompt, cfg 1): the model is guidance-distilled")));
        }
        // the canvas: as asked (rounded to 32), else 8 MP square; an edit's: picture 1's size within 4 .. 8 MP
        let (w, h, first) = match (req.width, req.height, pics.first()) {
            (Some(w), Some(h), _) => {
                let (w, h) = round_canvas(w, h);
                (w, h, None)
            }
            (None, None, Some(p)) => {
                let (w, h) = size_in_area_range(p.width, p.height, CANVAS_MIN_PIXELS, REFERENCE_MAX_PIXELS);
                (w, h, Some((w, h)))
            }
            (None, None, None) => (DEFAULT_SIDE, DEFAULT_SIDE, None),
            _ => return Err(Error(format!("{ARCH}: a width and a height together, or neither"))),
        };
        if w as u64 * h as u64 > MAX_PIXELS || w < 256 || h < 256 {
            return Err(Error(format!("{w}x{h}: from 256 a side to {:.1} MP (the model is at its best at 8 MP: 2912x2912, 3360x2496, 3872x2176 ...)",
                                     MAX_PIXELS as f64 / 1e6)));
        }
        let sigmas = match req.extra.get("sigmas") {
            Some(s) => parse_sigmas(s)?,
            None => SIGMAS.to_vec(),
        };
        if req.steps.is_some_and(|s| s as usize != sigmas.len() - 1) {
            return Err(Error(format!("{ARCH}: it runs its grid's {} steps (options.sigmas for another grid)", sigmas.len() - 1)));
        }
        let of = (sigmas.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let mut log = |l: String| eprintln!("{l}");
        let cond = {
            let mut slot = self.te.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(self.text_encoder(&mut log)?);
            }
            let c = slot.as_mut().map(|t| t.encode(&req.prompt, &pics)).ok_or_else(|| Error("no text encoder".into()))??;
            if !self.te_keep {
                *slot = None;
            }
            c
        };
        let t_text = t0.elapsed().as_secs_f64();
        let refs = if pics.is_empty() { Vec::new() } else { self.reference_latents(&pics, first)? };
        let t_refs = t0.elapsed().as_secs_f64() - t_text;
        let (lh, lw) = (h as usize / 16, w as usize / 16);
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let t1 = Instant::now();
            let text = self.refine(&cond.rows)?;
            let t_refine = t1.elapsed().as_secs_f64();
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(text, &cond.tags, &refs, lh, lw, req.seed + p as u64, &sigmas, &mut step)?;
            let t_steps = t1.elapsed().as_secs_f64() - t_refine;
            let rgb = self.decoder.decode(&lat, lh, lw, &mut |_, _| Ok(())).map_err(h3e)?;
            log(format!("{ARCH} {w}x{h} picture {p}: text {:.2} s ({} tokens, {} pictures), pictures' latents {:.2} s, refiner {t_refine:.2} s, {} steps {t_steps:.2} s, decode {:.2} s",
                        if p == 0 { t_text } else { 0.0 }, cond.ids.len(), pics.len(), if p == 0 { t_refs } else { 0.0 }, of,
                        t1.elapsed().as_secs_f64() - t_refine - t_steps));
            out.push(Picture { width: w, height: h, channels: 3, data: rgb });
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        let used = self.dev.mem_used() as f64 / (1u64 << 30) as f64;
        vec![format!("{ARCH} in Rust and SYCL on {} ({used:.1} GiB, loaded {:.0} s ago)", self.dev.name(), self.loaded.elapsed().as_secs_f64())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_8_mp_presets_are_prunas_table() {
        // README: 1:1 2912x2912, 4:3 3360x2496, 3:2 3552x2368, 16:9 3872x2176 (and transposed); 4 MP for edits
        assert_eq!(size_at(8_388_608, 1.0), (2912, 2912));
        assert_eq!(size_at(8_388_608, 4.0 / 3.0), (3360, 2496));
        assert_eq!(size_at(8_388_608, 3.0 / 4.0), (2496, 3360));
        assert_eq!(size_at(8_388_608, 3.0 / 2.0), (3552, 2368));
        assert_eq!(size_at(8_388_608, 2.0 / 3.0), (2368, 3552));
        assert_eq!(size_at(8_388_608, 16.0 / 9.0), (3872, 2176));
        assert_eq!(size_at(8_388_608, 9.0 / 16.0), (2176, 3872));
        assert_eq!(size_at(4_194_304, 1.0), (2048, 2048));
        assert_eq!(size_at(4_194_304, 4.0 / 3.0), (2368, 1760));
        assert_eq!(size_at(4_194_304, 3.0 / 2.0), (2496, 1664));
        assert_eq!(size_at(4_194_304, 16.0 / 9.0), (2720, 1536));
        assert_eq!(size_at(4_194_304, 9.0 / 16.0), (1536, 2720));
    }

    #[test]
    fn canvases_round_to_32_as_python_does() {
        assert_eq!(round_canvas(1000, 1000), (992, 992));
        // 1008 / 32 = 31.5: half to even -> 32 x 32
        assert_eq!(round_canvas(1008, 1040), (1024, 1024));
        assert_eq!(round_canvas(10, 20), (32, 32));
        assert_eq!(round_canvas(2912, 3872), (2912, 3872));
    }

    #[test]
    fn an_edit_canvas_is_picture_1_within_4_to_8_mp() {
        // a 1 MP picture goes up to 4 MP
        assert_eq!(size_in_area_range(1024, 1024, CANVAS_MIN_PIXELS, REFERENCE_MAX_PIXELS), (2048, 2048));
        // one inside the range keeps its size (on the grid)
        assert_eq!(size_in_area_range(2368, 1760, CANVAS_MIN_PIXELS, REFERENCE_MAX_PIXELS), (2368, 1760));
        // a 24 MP photo comes down to 8 MP: 6000 x 4000 -> sqrt(8 Mi / 24 M) = 0.5912 -> 3547.2, 2364.8 -> 3552 x 2368
        assert_eq!(size_in_area_range(6000, 4000, CANVAS_MIN_PIXELS, REFERENCE_MAX_PIXELS), (3552, 2368));
        // references within 1 .. 8 MP: a small one goes up to 1 MP
        assert_eq!(size_in_area_range(512, 512, REFERENCE_MIN_PIXELS, REFERENCE_MAX_PIXELS), (1024, 1024));
    }

    #[test]
    fn the_grid_is_the_checkpoint_s_and_falls_to_zero() {
        assert_eq!(SIGMAS.len(), 9);
        assert_eq!((SIGMAS[0], SIGMAS[8]), (1.0, 0.0));
        assert!(SIGMAS.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(parse_sigmas("1, 0.5, 0").unwrap(), vec![1.0, 0.5, 0.0]);
        assert!(parse_sigmas("1,0.5").is_err());
        assert!(parse_sigmas("0.5,0.7,0").is_err());
    }

    #[test]
    fn the_euler_step_lands_on_x0_at_the_last_sigma() {
        // r = 0: x becomes x + sigma v
        let mut x = vec![1.0f32, -2.0];
        euler(&mut x, &[0.5, 1.0], 0.528718, 0.0);
        let s = 1.0 - (1.0 - 0.528718f32);
        assert_eq!(x, vec![1.0 + s * 0.5, -2.0 + s]);
        // sigma 1 -> 0.984753: r x + (1 - r) (x + v)
        let mut y = vec![0.0f32];
        euler(&mut y, &[1.0], 1.0, 0.984753);
        assert!((y[0] - (1.0 - 0.984753)).abs() < 1e-6);
    }

    #[test]
    fn the_connector_reads_the_configs_hidden_states() {
        // text_connector/config.json: hidden_state_indices [13, 19, 25, 31, 36], out 5120
        assert_eq!(te::TAPS, [13, 19, 25, 31, 36]);
        assert_eq!(te::COND, 5120);
    }
}
