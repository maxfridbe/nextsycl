//! MiniMax-H3-Image in Rust and SYCL: MiniMax's image model on H3 - text to image, and edits guided by pictures, in
//! six steps without guidance, up to 16 MP. ComfyUI's graph (`MiniMaxH3ImageModel`, `MiniMaxH3ImageVAE`,
//! `MiniMaxH3ImageConditioning`, comfyanonymous/ComfyUI dba377b9) over H3's own engine (crates/h3: the 50-block
//! denoiser, the text refiner, the token layout, the Qwen3-VL 32B text encoder streamed layer by layer, the vision
//! tower and the VAE encoder - the video engine's code) and this crate's head bank and decoder:
//!
//! ```text
//!   prompt (as written; an edit: "Picture i: " + each picture's vision block, then the prompt) -> Qwen3-VL 32B,
//!   the hidden state after layer 50 [T, 5120] (crates/h3 jobs::encode_pieces) -> projection + token refiner
//!   one sequence [text | the edit's pictures (VAE latents) | the image's 2 x 2 patches]          (net.rs)
//!   noise from the seed (torch's draws) -> Euler over the heads' grid 1, 0.968, 0.888, 0.75, 0.533, 0.232, 0
//!   latents [24, H/16, W/16] -> the 2D transformer decoder, 256-pixel tiles (vae.rs) -> RGB
//! ```
//!
//! The denoiser is any form crates/h3 reads: ComfyUI's int8 ConvRot file (int8 products), or a stable-diffusion.cpp
//! GGUF (Hearthscale's Q5_K_M / Q4_K_M / Q3_K_M: each matrix expanded to bfloat16 for its product; Q3_K read as the
//! same values in Q6_K blocks).
//!
//! LoRAs in ComfyUI's layout ride on the blocks' products (crates/h3's side path: `out += B (A x)` after each of a
//! block's four products, the token refiner's too): the card's (role `lora`, NS_H3I_LORA_STRENGTH), a preset's, and a
//! request's own (stacked while requests keep naming them). The pDMD 2-NFE LoRA's recipe is a card's settings: 2 steps
//! (NS_H3I_STEPS) on ComfyUI's shifted schedule (NS_H3I_SHIFT 12, `net::shifted_sigmas`), the heads still weighted by
//! their grid. An init picture (an img2img / detail pass, or an edit with strength below 1) starts the denoise from its
//! VAE latents at the schedule's first sigma; `live.rs` keeps the text encoder's and the VAE encoder's results for
//! repeated prompts and pictures.

pub mod check;
pub mod ffi;
pub mod live;
pub mod net;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_h3::denoiser::{bf16_bytes, TextRefiner};
use nextsycl_h3::device::{Device, Tensor};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::jobs::{Ctl, TeFiles};
use nextsycl_h3::noise::Mt19937;
use nextsycl_h3::safetensors::Checkpoint;
use nextsycl_h3::vision::{Frames, Piece};
use nextsycl_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, LoraUse, ModelFiles, Picture, Result, Sampler,
                     Schedule as Sched, SizePresets, Step};

pub const ARCH: &str = "minimax-h3-image";
/// the denoiser, the text encoder (H3's Qwen3-VL 32B GGUF), its tokenizer (vocab.json beside merges.txt and
/// tokenizer_config.json), the VAE (the encoder for edits, the image decoder), the vision tower (edits), a LoRA on the
/// blocks (ComfyUI's layout: the pDMD 2-NFE LoRA)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "vae", "vision", "lora"];

/// canvas sides are multiples of this (the VAE's 16 x the denoiser's 2 x 2 patches)
pub const GRID: u32 = 32;
/// the default canvas: 2048 x 2048, the area the model is tuned for (ComfyUI's IMAGE_SIZE)
pub const DEFAULT_SIDE: u32 = 2048;
/// the largest canvas: 16 MP and the 16 MP presets at other aspects (5024 x 3360 is 16.88 M)
pub const MAX_PIXELS: u64 = 4128 * 4096;
/// an edit's pictures: the vision tower sees them at a 1,024 short edge, the VAE within a 1,024 .. 2,048 short edge
pub const REF_SHORT_EDGE: u32 = 1024;
pub const REF_VAE_MAX_SHORT_EDGE: u32 = 2048;
/// the most pictures an edit takes (the model card: up to 16)
pub const MAX_PICTURES: usize = 16;
/// the seed an edit's VAE posterior is sampled under, whatever the request's (ComfyUI's IMAGE_ENCODE_SEED)
const ENCODE_SEED: u64 = 42;
/// the heads' own number of steps
pub const STEPS: usize = 6;

const RATIOS: &[&str] = &["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"];
const PIXELS: &[u32] = &[1_048_576, 4_194_304, 8_388_608, 16_777_216];

const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "threads", env: "NS_H3I_THREADS", value: "N", help: "threads reading the checkpoints at load (default 8)", at: At::Load },
    EngineOption { name: "vae-batch", env: "NS_H3I_VAE_BATCH", value: "N", help: "the decoder's tiles decoded together (default 16)", at: At::Load },
    EngineOption { name: "te-pin", env: "H3_TE_PIN", value: "0|1", help: "keep the text encoder in pinned host memory between requests (default 1 when the host has the room)", at: At::Load },
    EngineOption { name: "sage-min-s", env: "NS_H3I_SAGE_MIN_S", value: "TOKENS", help: "the shortest denoiser sequence the silo's SageAttention takes (default 1024; shorter: oneDNN's)", at: At::Load },
    EngineOption { name: "silo-expand", env: "NS_H3I_SILO_EXPAND", value: "0|1", help: "the silo's GGUF expansion to bfloat16, Q3_K read as stored (default 1; 0: the shared kernels', Q3_K held as Q6_K)", at: At::Load },
    EngineOption { name: "q8-product", env: "NS_H3I_Q8_PRODUCT", value: "0|1", help: "Q8_0: the silo's products read the int8 values (oneDNN weight decompression; default 1; 0: expanded to bfloat16 first)", at: At::Load },
    EngineOption { name: "keep-refiner", env: "NS_H3I_KEEP_REFINER", value: "0|1", help: "keep the token refiner on the card between pictures (default 0: loaded for each picture, 0.08 - 0.17 s; 1: kept, 1.5 GiB more on the card)", at: At::Load },
    EngineOption { name: "profile", env: "NS_H3I_PROFILE", value: "0|1", help: "debug: each step's time by stage on stderr (the queue drained between stages)", at: At::Load },
    EngineOption { name: "sigmas", env: "", value: "S,S,..,0", help: "another sampling grid than the heads' six steps: strictly falling from 1, ending at 0 (below 1 with an init picture: the denoise starts there)", at: At::Request },
    EngineOption { name: "lora-strength", env: "NS_H3I_LORA_STRENGTH", value: "X", help: "the strength of the card's LoRA (role lora; default 1: the pDMD 2-step LoRA's distillation as trained, other values scale it)", at: At::Load },
    EngineOption { name: "steps", env: "NS_H3I_STEPS", value: "N", help: "the default number of steps (default 6: the heads' grid; the pDMD 2-step card and LoRA: 2; a request's steps first)", at: At::Both },
    EngineOption { name: "shift", env: "NS_H3I_SHIFT", value: "S", help: "sample on ComfyUI's shifted flow schedule (MiniMaxH3SigmaShift shift_video S + the simple scheduler) instead of the heads' grid (the pDMD 2-step card and LoRA: 12; a request's shift first)", at: At::Both },
    EngineOption { name: "heads", env: "NS_H3I_HEADS", value: "bank|first", help: "the output heads: the bank weighted by the grid (default, as ComfyUI) or head 0 alone (the diffusers route's single head)", at: At::Both },
    EngineOption { name: "text-cache", env: "NS_H3I_TEXT_CACHE", value: "N", help: "keep the text encoder's result (and the token refiner's) for the last N prompts (+ their pictures): a repeat skips the 32B encoder (default 0: off; the 2-step card: 8)", at: At::Load },
    EngineOption { name: "image-cache", env: "NS_H3I_IMAGE_CACHE", value: "N", help: "keep the VAE latents of the last N pictures (an edit's references, an img2img start) at each size: an edit of the same picture skips the VAE encoder (default 0: off; the 2-step card: 8)", at: At::Load },
    EngineOption { name: "start-sigma", env: "", value: "S", help: "an init picture's denoise starts at sigma S exactly (the demo Space's detail pass: S, 0.943 S, 0.733 S, 0 at 3 steps); without it a strength below 1 is ComfyUI's denoise", at: At::Request },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "MiniMax-H3-Image (H3's 50-block DiT on one frame with a six-head bank, Qwen3-VL 32B, 2D transformer VAE decoder; Rust + SYCL)",
                roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(H3Image::load(files, gpus, o, log)?))
}

/// An H3 error as the image kind's
pub fn h3e(e: nextsycl_h3::Error) -> Error {
    Error(e.0)
}

/// Python's round (half to even) of `side / GRID`, times GRID, at least GRID
pub fn round_side(side: f64) -> u32 {
    ((side / GRID as f64).round_ties_even() as u32 * GRID).max(GRID)
}

/// ComfyUI's `_size_short_edge`: the aspect kept, the short edge brought into [low, high], sides rounded to the grid
pub fn size_short_edge(w: u32, h: u32, low: u32, high: u32) -> (u32, u32) {
    let short = w.min(h) as f64;
    let scale = short.clamp(low as f64, high as f64) / short;
    (round_side(w as f64 * scale), round_side(h as f64 * scale))
}

/// A budget of `pixels` at aspect `ratio` (W / H), each side rounded to the grid
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

/// What a request samples: its grid, whether it starts from its first picture's latents (an img2img / detail pass,
/// or an edit at a strength below 1), whether its pictures condition the model as an edit's references
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub sigmas: Vec<f32>,
    pub init: bool,
    pub condition: bool,
    /// head 0 alone (the request's `heads` option; none: the engine's)
    pub first_head: Option<bool>,
}

/// A request option as a number
fn extra_number<T: std::str::FromStr>(req: &ImageRequest, name: &str) -> Result<Option<T>> {
    req.extra.get(name).map(|v| v.trim().parse::<T>().map_err(|_| Error(format!("{name} {v}: a number")))).transpose()
}

/// A request's plan with the engine's default `steps` and `shift` (none: the heads' grid); the request's own fields
/// first, then its options (`steps`, `shift`: what a request LoRA's card brings, the pDMD LoRA's 2 and 12): `sigmas` as
/// given; else `start-sigma` S with a picture: the demo Space's grid from S; else a strength below 1: ComfyUI's
/// BasicScheduler denoise; else the whole schedule
pub fn plan(req: &ImageRequest, steps: usize, shift: Option<f32>) -> Result<Plan> {
    let steps = extra_number::<usize>(req, "steps")?.unwrap_or(steps);
    let shift = req.shift.or(extra_number::<f32>(req, "shift")?).or(shift);
    if shift.is_some_and(|s| !(0.01..=100.0).contains(&s)) {
        return Err(Error("shift: 0.01 to 100".into()));
    }
    let first_head = match req.extra.get("heads").map(|v| v.trim()) {
        None => None,
        Some("bank") => Some(false),
        Some("first") => Some(true),
        Some(v) => return Err(Error(format!("heads {v}: bank or first"))),
    };
    let condition = req.edit.as_ref().is_some_and(|e| e.condition);
    let strength = req.edit.as_ref().map_or(1.0, |e| e.strength);
    let start = req.extra.get("start-sigma").map(|v| v.trim().parse::<f32>().map_err(|_| Error(format!("start-sigma {v}: a number")))).transpose()?;
    if start.is_some_and(|s| !(s > 0.0 && s <= 1.0)) {
        return Err(Error("start-sigma: above 0, at most 1".into()));
    }
    if start.is_some() && req.edit.is_none() {
        return Err(Error(format!("{ARCH}: start-sigma needs a picture to start from")));
    }
    let init = req.edit.is_some() && (strength < 1.0 || start.is_some());
    if req.edit.is_some() && !condition && !init {
        return Err(Error(format!("{ARCH}: a picture that is not a reference needs a strength below 1 or a start-sigma (an img2img)")));
    }
    let sigmas = match (req.extra.get("sigmas"), start) {
        (Some(_), Some(_)) => return Err(Error("sigmas or start-sigma, not both".into())),
        (Some(s), None) => parse_sigmas(s)?,
        (None, _) => {
            let n = req.steps.map_or(steps, |s| s as usize);
            if !(1..=50).contains(&n) {
                return Err(Error(format!("{ARCH}: {n} steps (1 to 50; the heads are distilled for {STEPS}, the pDMD LoRA for 2)")));
            }
            match start {
                Some(s0) => net::start_grid(s0, &net::schedule(n, shift, 1.0)),
                None => net::schedule(n, shift, if init { strength } else { 1.0 }),
            }
        }
    };
    Ok(Plan { sigmas, init, condition, first_head })
}

/// The text side's result: the conditioning [T, 5120] and per token its modality tag (1 text, 0 a vision token)
pub struct Encoded {
    pub rows: Vec<f32>,
    pub tags: Vec<i32>,
    pub seconds: f64,
}

pub struct H3Image {
    gpu: Arc<Gpu>,
    pub dev: Arc<Device>,
    pub ck: Checkpoint,
    pub blocks: nextsycl_h3::dit::Blocks,
    pub outer: net::Outer,
    pub decoder: vae::Decoder,
    files: ModelFiles,
    threads: usize,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// NS_H3I_PROFILE: each step's time by stage
    pub profile: bool,
    /// the token refiner, loaded once (NS_H3I_KEEP_REFINER=1); None (the default): loaded for each picture
    refiner: Option<TextRefiner>,
    /// the Q8_0 block matrices were rewritten for the silo's product (the shared expansion cannot read them)
    pub q8_repacked: bool,
    /// the LoRAs on the blocks (the card's, a preset's, a request's on top)
    pub loras: Mutex<live::Loras>,
    /// the default steps (NS_H3I_STEPS) and the shifted schedule's shift (NS_H3I_SHIFT; none: the heads' grid)
    pub steps: usize,
    pub shift: Option<f32>,
    /// head 0 alone (NS_H3I_HEADS=first)
    pub first_head: bool,
    /// the text encoder's results by prompt (NS_H3I_TEXT_CACHE), the pictures' latents (NS_H3I_IMAGE_CACHE)
    texts: Mutex<live::Lru<TextEntry>>,
    latents: Mutex<live::Lru<Arc<Vec<f32>>>>,
}

/// An edit's picture as conditioning: its normalized latents [24, lh, lw], lh, lw
pub type RefLatent = (Vec<f32>, usize, usize);

/// A prompt's text-encoder result and the token refiner's output under the LoRAs it was refined with (their key)
struct TextEntry {
    enc: Arc<Encoded>,
    refined: Option<(u64, Tensor)>,
}

/// A setting as a number, refused when it does not read as one
fn number<T: std::str::FromStr>(o: &LoadOptions, name: &str) -> Result<Option<T>> {
    o.setting(name).filter(|v| !v.trim().is_empty()).map(|v| v.trim().parse::<T>().map_err(|_| Error(format!("{name}={v}: a number")))).transpose()
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl H3Image {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<H3Image> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let threads = o.setting("NS_H3I_THREADS").and_then(|v| v.parse().ok()).unwrap_or(8);
        for r in ["text-encoder", "tokenizer", "vae"] {
            role(files, r)?;
        }
        // H3's kernels on the kind's queue of the card (the denoiser, the decoder and the text encoder ordered on it)
        // SAFETY: the core's GPU (its queue's owner) is kept in the engine beside the device.
        let dev = unsafe { Device::on_queue(gpu.queue(), gpu.index) }.map_err(h3e)?;
        let mut ck = Checkpoint::open(role(files, "denoiser")?).map_err(h3e)?;
        let silo = ffi::silo().map_err(|e| Error(e.0))?;
        // the silo's k-quant expansion (NS_H3I_SILO_EXPAND, default on): the block matrices' Q3_K then read as stored
        let expand = silo.filter(|_| o.setting("NS_H3I_SILO_EXPAND").is_none_or(|v| v != "0"));
        if expand.is_some_and(|s| ffi::expands(s, nextsycl_h3::gguf::GType::Q3K)) {
            keep_q3k(&mut ck);
        }
        let mut blocks = nextsycl_h3::dit::Blocks::load(&dev, &ck, None, threads).map_err(h3e)?;
        // the silo's attention (dist/silo/libnextsycl-h3image.so: P-H3-Image's, copied) from NS_H3I_SAGE_MIN_S tokens;
        // without it, or below that, the shared kernels' (SageAttention from 8,192 tokens, oneDNN's below)
        if let Some(s) = silo {
            let min_s = o.setting("NS_H3I_SAGE_MIN_S").and_then(|v| v.parse().ok()).unwrap_or(1024);
            blocks.attention = Some(ffi::attention(&gpu, s, min_s));
            log(format!("{ARCH}: the {} silo's attention from {min_s} tokens", ffi::SILO));
        }
        let on = |name: &str| o.setting(name).is_none_or(|v| v != "0");
        // Q8_0: the silo's products read the int8 values themselves (NS_H3I_Q8_PRODUCT, default on); the other kinds
        // it reads: its expansion, then one product over all the rows (NS_H3I_SILO_EXPAND)
        let mut q8_repacked = false;
        if let (Some(s), Some(kind)) = (silo, blocks.kquant) {
            let q8 = kind == nextsycl_h3::gguf::GType::Q8_0 && on("NS_H3I_Q8_PRODUCT");
            q8_repacked = q8;
            let route = ffi::route(kind, expand.is_some() && ffi::expands(s, kind), q8);
            if route != ffi::Route::Shared {
                let n = if q8 { ffi::q8_repack(&gpu, s, &blocks).map_err(h3e)? } else { 0 };
                blocks.product = Some(ffi::products(&gpu, s, q8, expand.is_some()));
                log(format!("{ARCH}: the {} silo's {kind:?} products ({})", ffi::SILO,
                            if q8 { format!("{n} matrices repacked, int8 decompression, no expansion") } else { "its expansion, one product".into() }));
            }
        }
        let outer = net::Outer::load(&dev, &ck, &blocks.cfg).map_err(h3e)?;
        let form = match ck.entries.values().find(|e| e.name == "blocks.0.mlp.fc1.weight") {
            Some(e) if e.stored.is_some() => format!("{:?} as {:?}", e.stored.unwrap(), e.kquant.unwrap()),
            _ => blocks.kquant.map_or("int8".to_string(), |k| format!("{k:?}")),
        };
        log(format!("{ARCH}: {} blocks ({form}) {:.2} GiB in {:.1} s on {}", blocks.blocks.len(), blocks.load_bytes as f64 / (1u64 << 30) as f64,
                    blocks.load_seconds, dev.name()));
        let mut decoder = vae::Decoder::load(&dev, &Checkpoint::open(role(files, "vae")?).map_err(h3e)?, threads).map_err(h3e)?;
        if let Some(b) = o.setting("NS_H3I_VAE_BATCH").and_then(|v| v.parse::<usize>().ok()).filter(|b| *b > 0) {
            decoder.tile_batch = b;
        }
        log(format!("{ARCH}: VAE decoder {:.2} GiB in {:.1} s; {:.1} GiB used on the card", decoder.load_bytes as f64 / (1u64 << 30) as f64,
                    decoder.load_seconds, dev.mem_used() as f64 / (1u64 << 30) as f64));
        // kept only when asked: it holds 1.5 GiB on the card to save 0.08 - 0.17 s a picture
        let refiner = if o.setting("NS_H3I_KEEP_REFINER").is_some_and(|v| v == "1") {
            let before = dev.mem_used();
            let r = TextRefiner::load(&dev, &ck, threads).map_err(h3e)?;
            log(format!("{ARCH}: token refiner {:.2} GiB, kept", (dev.mem_used() - before) as f64 / (1u64 << 30) as f64));
            Some(r)
        } else {
            None
        };
        let profile = o.setting("NS_H3I_PROFILE").is_some_and(|v| v != "0" && !v.is_empty());
        if profile {
            if let Some(kind) = blocks.kquant {
                log(format!("{ARCH}: profile: {}", expansion_profile(&dev, &gpu, &blocks.cfg, kind, silo).map_err(h3e)?));
            }
        }
        // the card's LoRA (role lora) at NS_H3I_LORA_STRENGTH, then a preset's, on the side path
        let strength: f32 = number(o, "NS_H3I_LORA_STRENGTH")?.unwrap_or(1.0);
        let mut uses: Vec<LoraUse> = files.get("lora").map(|p| LoraUse { name: "lora".into(), path: p.clone(), scale: strength }).into_iter().collect();
        uses.extend(o.merge_loras.iter().cloned());
        let refiners = refiner_blocks(&ck);
        let mut set: Option<nextsycl_h3::lora::LoraSet> = None;
        for l in &uses {
            let before = dev.mem_used();
            let add = nextsycl_h3::lora::LoraSet::load(&dev, &Checkpoint::open(&l.path).map_err(h3e)?, l.scale, blocks.blocks.len(), refiners).map_err(h3e)?;
            if add.max_rank > nextsycl_h3::dit::MAX_LORA_RANK {
                return Err(Error(format!("{ARCH}: {} has rank {} (at most {})", l.path.display(), add.max_rank, nextsycl_h3::dit::MAX_LORA_RANK)));
            }
            log(format!("{ARCH}: LoRA {} x {}: {} layers, rank up to {}, {:.2} GiB on the card", l.path.display(), l.scale, add.layers, add.max_rank,
                        (dev.mem_used() - before) as f64 / (1u64 << 30) as f64));
            match set.as_mut() {
                Some(s) => s.stack(add),
                None => set = Some(add),
            }
        }
        let steps: usize = number(o, "NS_H3I_STEPS")?.unwrap_or(STEPS);
        if !(1..=50).contains(&steps) {
            return Err(Error(format!("NS_H3I_STEPS={steps}: 1 to 50")));
        }
        let shift: Option<f32> = number(o, "NS_H3I_SHIFT")?;
        if shift.is_some_and(|s| !(0.01..=100.0).contains(&s)) {
            return Err(Error("NS_H3I_SHIFT: 0.01 to 100 (ComfyUI's MiniMaxH3SigmaShift range)".into()));
        }
        let first_head = match o.setting("NS_H3I_HEADS").as_deref() {
            None | Some("") | Some("bank") => false,
            Some("first") => true,
            Some(v) => return Err(Error(format!("NS_H3I_HEADS={v}: bank or first"))),
        };
        let texts = live::Lru::new(number(o, "NS_H3I_TEXT_CACHE")?.unwrap_or(0));
        let latents = live::Lru::new(number(o, "NS_H3I_IMAGE_CACHE")?.unwrap_or(0));
        log(format!("{ARCH}: {steps} steps by default on {}{}; caches: {} prompts, {} pictures' latents", match shift {
            Some(s) => format!("the shifted schedule (shift {s})"),
            None => "the heads' grid".into(),
        }, if first_head { ", head 0 alone" } else { "" }, texts.cap(), latents.cap()));
        Ok(H3Image { gpu, dev, ck, blocks, outer, decoder, files: files.clone(), threads, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(),
                     busy: Mutex::new(()), profile, refiner, q8_repacked, loras: Mutex::new(live::Loras::new(set)), steps, shift, first_head,
                     texts: Mutex::new(texts), latents: Mutex::new(latents) })
    }

    /// The default schedule of `steps` steps at `denoise` (1: from noise): the shifted one when the card sets a
    /// shift (or the request does), else the heads' grid
    pub fn sigmas(&self, steps: usize, shift: Option<f32>, denoise: f32) -> Vec<f32> {
        net::schedule(steps, shift.or(self.shift), denoise)
    }

    /// The tokenizer's directory (vocab.json, merges.txt, tokenizer_config.json)
    fn tokenizer_dir(&self) -> Result<PathBuf> {
        let p = role(&self.files, "tokenizer")?;
        Ok(if p.is_file() { p.parent().map(Path::to_path_buf).unwrap_or_else(|| p.to_path_buf()) } else { p.to_path_buf() })
    }

    /// The prompt (and an edit's pictures, RGB at the vision tower's sizes) through H3's text encoder: "Picture i: "
    /// and each picture's vision block ahead of the prompt (ComfyUI's MiniMaxH3ImageConditioning)
    pub fn encode(&self, prompt: &str, pics: &[Picture], log: &mut dyn FnMut(String)) -> Result<Encoded> {
        let mut pieces = Vec::new();
        for (i, p) in pics.iter().enumerate() {
            let rgb = p.rgb();
            let (w, h) = size_short_edge(rgb.width, rgb.height, REF_SHORT_EDGE, REF_SHORT_EDGE);
            let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
            pieces.push(Piece::Text(format!("Picture {}: ", i + 1)));
            pieces.push(Piece::Vision(Frames { frames: vec![shown.data.iter().map(|v| *v as f32 / 255.0).collect()], w: w as usize, h: h as usize }));
        }
        pieces.push(Piece::Text(prompt.to_string()));
        let tok = self.tokenizer_dir()?;
        let visual = self.files.get("vision").map(|p| p.as_path());
        if !pics.is_empty() && visual.is_none() {
            return Err(Error(format!("{ARCH}: an edit needs the vision tower (role vision)")));
        }
        let files = TeFiles { te: role(&self.files, "text-encoder")?, tokenizer: &tok, visual, embeddings: None };
        let cancel = AtomicBool::new(false);
        let mut ctl = Ctl { log, cancel: &cancel, progress: None };
        let (rows, presented, _, seconds) = nextsycl_h3::jobs::encode_pieces(&self.dev, self.threads, &pieces, &files, &mut ctl).map_err(h3e)?;
        Ok(Encoded { rows, tags: presented.tags, seconds })
    }

    /// The conditioning rows [T, 5120] through the denoiser's projection and token refiner (with the LoRAs on the
    /// refiner's blocks): bfloat16 [T, hidden]
    pub fn refine(&self, cond: &[f32]) -> Result<Tensor> {
        let k = self.ck.get("condition_proj.weight").map_err(h3e)?.shape[1];
        let t = cond.len() / k;
        let c = Tensor::from_bytes(&self.dev, DType::BF16, &[t, k], &bf16_bytes(cond)).map_err(h3e)?;
        let loras = self.loras.lock().unwrap_or_else(|p| p.into_inner());
        let lora = loras.set.as_ref().map(|s| s.refiner.as_slice());
        let out = match &self.refiner {
            Some(r) => r.run(&self.blocks.cfg, &c, lora).map_err(h3e)?,
            None => TextRefiner::load(&self.dev, &self.ck, self.threads).map_err(h3e)?.run(&self.blocks.cfg, &c, lora).map_err(h3e)?,
        };
        self.dev.wait().map_err(h3e)?;
        Ok(out)
    }

    /// The text side of a request, from the text cache when it holds the prompt and its pictures: the encoder's result
    /// and the refined text (a copy the picture's run owns); and what was cached (encoder, refiner)
    pub fn text(&self, prompt: &str, pics: &[Picture], log: &mut dyn FnMut(String)) -> Result<(Arc<Encoded>, Tensor, (bool, bool))> {
        let key = live::text_key(prompt, &pics.iter().map(live::picture_key).collect::<Vec<_>>());
        let lkey = self.loras.lock().unwrap_or_else(|p| p.into_inner()).extra_key();
        let mut texts = self.texts.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = texts.get(key) {
            let enc = e.enc.clone();
            if let Some((_, t)) = e.refined.as_ref().filter(|(k, _)| *k == lkey) {
                return Ok((enc, copy(t).map_err(h3e)?, (true, true)));
            }
            let refined = self.refine(&enc.rows)?;
            e.refined = Some((lkey, copy(&refined).map_err(h3e)?));
            return Ok((enc, refined, (true, false)));
        }
        drop(texts);
        let enc = Arc::new(self.encode(prompt, pics, log)?);
        let refined = self.refine(&enc.rows)?;
        let mut texts = self.texts.lock().unwrap_or_else(|p| p.into_inner());
        if texts.cap() > 0 {
            texts.put(key, TextEntry { enc: enc.clone(), refined: Some((lkey, copy(&refined).map_err(h3e)?)) });
        }
        Ok((enc, refined, (false, false)))
    }

    /// H3's VAE encoder (the conv encoder in the VAE file)
    fn encoder(&self) -> Result<nextsycl_h3::venc::VideoEncoder> {
        nextsycl_h3::venc::VideoEncoder::load(&self.dev, &Checkpoint::open(role(&self.files, "vae")?).map_err(h3e)?).map_err(h3e)
    }

    /// A picture's latents [24, h/16, w/16] at `w` x `h` (resized with Lanczos when it is another size): the posterior
    /// sampled under ComfyUI's IMAGE_ENCODE_SEED and normalized, as MiniMaxH3ImageVAE.encode gives them; from the picture
    /// cache when it holds them (`purpose` keys the use); `enc` is loaded on the first miss
    fn latents_of(&self, p: &Picture, w: u32, h: u32, purpose: &str, enc: &mut Option<nextsycl_h3::venc::VideoEncoder>) -> Result<(Arc<Vec<f32>>, bool)> {
        let key = live::latent_key(live::picture_key(p), w, h, purpose);
        if let Some(l) = self.latents.lock().unwrap_or_else(|p| p.into_inner()).get(key) {
            return Ok((l.clone(), true));
        }
        if enc.is_none() {
            *enc = Some(self.encoder()?);
        }
        let e = enc.as_ref().ok_or("no encoder")?;
        let (mean, std) = e.latent_stats();
        let rgb = p.rgb();
        let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
        let px: Vec<f32> = shown.data.iter().map(|v| *v as f32 / 255.0).collect();
        let m = e.moments_of_picture(&px, h as usize, w as usize, &mut || Ok(())).map_err(h3e)?;
        let noise = Mt19937::new(ENCODE_SEED).randn(24 * (h as usize / 16) * (w as usize / 16));
        let l = Arc::new(vae::condition_latent(&m, &noise, mean, std));
        self.latents.lock().unwrap_or_else(|p| p.into_inner()).put(key, l.clone());
        Ok((l, false))
    }

    /// An edit's pictures as conditioning latents [24, h, w] (normalized), each at its VAE size (short edge within
    /// 1,024 .. 2,048)
    pub fn reference_latents(&self, pics: &[Picture]) -> Result<Vec<RefLatent>> {
        let mut enc = None;
        Ok(self.references(pics, &mut enc)?.0)
    }

    /// As `reference_latents`, sharing an encoder; and how many came from the cache
    fn references(&self, pics: &[Picture], enc: &mut Option<nextsycl_h3::venc::VideoEncoder>) -> Result<(Vec<RefLatent>, usize)> {
        let mut out = Vec::with_capacity(pics.len());
        let mut hits = 0;
        for p in pics {
            let (w, h) = size_short_edge(p.width, p.height, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE);
            let (l, hit) = self.latents_of(p, w, h, "ref", enc)?;
            hits += hit as usize;
            out.push((l.as_ref().clone(), h as usize / 16, w as usize / 16));
        }
        Ok((out, hits))
    }

    /// The latents [24, lh, lw] of one picture: the refined text `text` with `tags`, the edit's latents `refs`, the
    /// noise from `seed`, the grid `sigmas`; the start x = s0 noise + (1 - s0) init (ComfyUI's CONST noise scaling at the
    /// first sigma s0; no init: s0 noise); `step(i)` after each step (an error stops)
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, text: Tensor, tags: &[i32], refs: &[(Vec<f32>, usize, usize)], init: Option<&[f32]>, first_head: bool, lh: usize, lw: usize, seed: u64, sigmas: &[f32],
                  step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let mut x = Mt19937::new(seed).randn(net::LATENT_C * lh * lw);
        start_latent(&mut x, init, sigmas[0])?;
        let t_run = Instant::now();
        let loras = self.loras.lock().unwrap_or_else(|p| p.into_inner());
        let mut run = net::Run::new(&self.blocks, &self.outer, text, Some(tags.to_vec()), lh, lw, refs, seed).map_err(h3e)?;
        run.lora = loras.set.as_ref();
        run.first_head = first_head;
        if self.profile {
            self.dev.wait().map_err(h3e)?;
            eprintln!("profile: the picture's buffers and conditioning rows {:.0} ms", t_run.elapsed().as_secs_f64() * 1e3);
            run.profile = true;
        }
        for i in 0..sigmas.len() - 1 {
            let v = run.net(&x, sigmas[i], sigmas[i + 1], &mut |_| Ok(())).map_err(h3e)?;
            net::euler(&mut x, &v, sigmas[i], sigmas[i + 1]);
            step(i)?;
        }
        Ok(x)
    }
}

/// The denoise's start from the noise `x` at the first sigma `s0`: s0 x + (1 - s0) init (ComfyUI's CONST
/// noise_scaling; no init: an empty latent); at s0 = 1 the noise as drawn
pub fn start_latent(x: &mut [f32], init: Option<&[f32]>, s0: f32) -> Result<()> {
    match init {
        Some(l) if l.len() != x.len() => Err(Error(format!("an init latent of {} values for {}", l.len(), x.len()))),
        Some(l) => {
            x.iter_mut().zip(l).for_each(|(n, l)| *n = s0 * *n + (1.0 - s0) * *l);
            Ok(())
        }
        None if s0 != 1.0 => {
            x.iter_mut().for_each(|n| *n *= s0);
            Ok(())
        }
        None => Ok(()),
    }
}

/// A device copy of a tensor
fn copy(t: &Tensor) -> nextsycl_h3::Result<Tensor> {
    let out = Tensor::new(t.buf.device(), t.dtype, &t.shape)?;
    out.copy_rows(0, t, 0, t.shape[0])?;
    Ok(out)
}

/// The token refiner's block count (`token_refiner.blocks.N`)
pub fn refiner_blocks(ck: &Checkpoint) -> usize {
    ck.entries.keys().filter_map(|k| k.strip_prefix("token_refiner.blocks.")).filter_map(|r| r.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1)
}

impl ImageEngine for H3Image {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: DEFAULT_SIDE, height: DEFAULT_SIDE, steps: self.steps as u32, cfg: 1.0, sampler: Sampler::Euler, schedule: Sched::Simple,
                   shift: self.shift.unwrap_or(1.0) }
    }
    fn edits(&self) -> bool {
        true
    }
    fn side_loras(&self) -> bool {
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
        true
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        match plan(req, self.steps, self.shift) {
            Ok(p) => (p.sigmas.len() - 1) as u32,
            Err(_) => req.steps.unwrap_or(self.steps as u32),
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
                if e.mask.is_some() {
                    return Err(Error(format!("{ARCH}: masks are not part of this model's edits (say what to keep instead)")));
                }
                std::iter::once(e.image.clone()).chain(e.refs.iter().cloned()).collect()
            }
        };
        let plan = plan(req, self.steps, self.shift)?;
        if pics.len() > MAX_PICTURES {
            return Err(Error(format!("{ARCH}: an edit takes at most {MAX_PICTURES} pictures, got {}", pics.len())));
        }
        if req.negative.is_some() || req.cfg.is_some_and(|c| c != 1.0) {
            return Err(Error(format!("{ARCH}: no guidance (no negative prompt, cfg 1): the model is distilled")));
        }
        // the canvas: as asked (rounded to 32), else picture 1's VAE size, else 2048 x 2048
        let (w, h) = match (req.width, req.height, pics.first()) {
            (Some(w), Some(h), _) => (round_side(w as f64), round_side(h as f64)),
            (None, None, Some(p)) => size_short_edge(p.width, p.height, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE),
            (None, None, None) => (DEFAULT_SIDE, DEFAULT_SIDE),
            _ => return Err(Error(format!("{ARCH}: a width and a height together, or neither"))),
        };
        if w as u64 * h as u64 > MAX_PIXELS || w < 256 || h < 256 {
            return Err(Error(format!("{w}x{h}: from 256 a side to 16 MP (4096x4096, 5024x3360 ...); the model is tuned for 2048x2048")));
        }
        let sigmas = plan.sigmas;
        let of = (sigmas.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let mut log = |l: String| eprintln!("{l}");
        // a request's own LoRAs on top of the engine's (none: back to the engine's)
        self.loras.lock().unwrap_or_else(|p| p.into_inner())
            .want(&self.dev, &req.loras, self.blocks.blocks.len(), refiner_blocks(&self.ck), nextsycl_h3::dit::MAX_LORA_RANK, &mut log).map_err(h3e)?;
        // the pictures condition the model (an edit) or only start the denoise (an img2img: the prompt alone guides it)
        let shown: &[Picture] = if plan.condition { &pics } else { &[] };
        let (cond, text, (te_hit, refined_hit)) = self.text(&req.prompt, shown, &mut log)?;
        let mut text = Some(text);
        let t_text = t0.elapsed().as_secs_f64();
        let mut enc = None;
        let (refs, ref_hits) = if plan.condition { self.references(&pics, &mut enc)? } else { (Vec::new(), 0) };
        let init = match plan.init {
            true => Some(self.latents_of(&pics[0], w, h, "init", &mut enc)?),
            false => None,
        };
        drop(enc);
        let t_refs = t0.elapsed().as_secs_f64() - t_text;
        let (lh, lw) = (h as usize / 16, w as usize / 16);
        let mut out = Vec::new();
        let n = req.n.max(1);
        for p in 0..n {
            let t1 = Instant::now();
            // the refined text is the same for every picture: a copy for each run but the last
            let mine = match (p + 1 < n, text.take()) {
                (true, Some(t)) => {
                    let c = copy(&t).map_err(h3e)?;
                    text = Some(t);
                    c
                }
                (false, Some(t)) => t,
                (_, None) => return Err(Error("the refined text was used up".into())),
            };
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(mine, &cond.tags, &refs, init.as_ref().map(|(l, _)| l.as_slice()), plan.first_head.unwrap_or(self.first_head), lh, lw, req.seed + p as u64, &sigmas, &mut step)?;
            let t_steps = t1.elapsed().as_secs_f64();
            let rgb = self.decoder.decode(&lat, lh, lw, &mut |_, _| Ok(())).map_err(h3e)?;
            let cached = |hit: bool| if hit { " (cached)" } else { "" };
            log(format!("{ARCH} {w}x{h} picture {p}: text + refiner {:.2} s ({} tokens, {} pictures; encoder{}, refiner{}), pictures' latents {:.2} s ({} of {} cached{}), \
                         {of} steps from sigma {:.3} {t_steps:.2} s, decode {:.2} s",
                        if p == 0 { t_text } else { 0.0 }, cond.tags.len(), shown.len(), cached(te_hit), cached(refined_hit), if p == 0 { t_refs } else { 0.0 },
                        ref_hits + init.as_ref().is_some_and(|(_, h)| *h) as usize, refs.len() + init.is_some() as usize,
                        if init.is_some() { ", one the start" } else { "" }, sigmas[0], t1.elapsed().as_secs_f64() - t_steps));
            out.push(Picture { width: w, height: h, channels: 3, data: rgb });
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        let used = self.dev.mem_used() as f64 / (1u64 << 30) as f64;
        vec![format!("{ARCH} in Rust and SYCL on {} ({used:.1} GiB, loaded {:.0} s ago)", self.gpu_name(), self.loaded.elapsed().as_secs_f64())]
    }
}

/// Whether `name` is one of the denoiser blocks' four matrices (`blocks.<i>.attn.qkv_proj.weight` ...): what the
/// silo expands (the token refiner's go through the shared kernels)
pub fn is_block_matrix(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("blocks.") else { return false };
    let Some((idx, tail)) = rest.split_once('.') else { return false };
    idx.parse::<usize>().is_ok() && ["attn.qkv_proj.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"].contains(&tail)
}

/// The block matrices' Q3_K entries read as the file stores them (110 bytes a block of 256) rather than transcoded
/// to Q6_K (210): for the silo's expansion, which reads Q3_K. The count changed.
pub fn keep_q3k(ck: &mut Checkpoint) -> usize {
    use nextsycl_h3::gguf::GType;
    let mut n = 0;
    for e in ck.entries.values_mut().filter(|e| e.stored == Some(GType::Q3K) && is_block_matrix(&e.name)) {
        e.kquant = Some(GType::Q3K);
        e.stored = None;
        e.bytes = GType::Q3K.bytes(e.shape.iter().product());
        n += 1;
    }
    n
}

/// The k-quant expansion alone (NS_H3I_PROFILE): one block's four matrices expanded to bfloat16 from blocks of `kind`
/// (the bytes are arbitrary: only the time is read), each twice, the second timed - by the shared kernels (Q3_K: its
/// Q6_K form, what they expand) and by the silo when it reads `kind`
fn expansion_profile(dev: &Arc<Device>, gpu: &Gpu, cfg: &nextsycl_h3::dit::Config, kind: nextsycl_h3::gguf::GType, silo: Option<&ffi::Silo>)
                     -> nextsycl_h3::Result<String> {
    use nextsycl_h3::gguf::GType;
    let w = cfg.heads * cfg.head_dim;
    let shapes = [("qkv", 3 * w, cfg.hidden), ("out", cfg.hidden, w), ("fc1", 2 * cfg.ffn, cfg.hidden), ("fc2", cfg.hidden, cfg.ffn)];
    let most = shapes.iter().map(|(_, n, k)| n * k).max().unwrap_or(0);
    let shared_kind = if kind == GType::Q3K { GType::Q6K } else { kind };
    let out = Tensor::new(dev, DType::BF16, &[most])?;
    let src = Tensor::new(dev, DType::U8, &[shared_kind.bytes(most).max(kind.bytes(most))])?;
    let mut lines = Vec::new();
    let mut time = |who: &str, f: &dyn Fn(usize) -> nextsycl_h3::Result<()>| -> nextsycl_h3::Result<()> {
        let (mut parts, mut total) = (Vec::new(), 0.0);
        for (name, n, k) in shapes {
            f(n * k)?;
            dev.wait()?;
            let t = Instant::now();
            f(n * k)?;
            dev.wait()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            total += ms;
            parts.push(format!("{name} {ms:.2}"));
        }
        lines.push(format!("{who} {total:.2} ms a block ({} ms), x {} = {:.0} ms a step", parts.join(", "), cfg.blocks, total * cfg.blocks as f64));
        Ok(())
    };
    time(&format!("shared {shared_kind:?}"), &|n| nextsycl_h3::ops::expand_into(&src, shared_kind, n, &out))?;
    if let Some(s) = silo.filter(|s| ffi::expands(s, kind)) {
        time(&format!("silo {kind:?}"), &|n| ffi::expand_into(gpu, s, &src, kind, n, &out))?;
    }
    Ok(format!("expansion: {}", lines.join("; ")))
}

impl H3Image {
    /// The card the engine runs on (the silo's calls take it)
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn gpu_name(&self) -> String {
        format!("{} (card {})", self.dev.name(), self.gpu.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_size_pictures_as_comfyui_does() {
        // the short edge into [1024, 2048], the aspect kept, sides on the 32 grid (half to even)
        assert_eq!(size_short_edge(512, 768, 1024, 2048), (1024, 1536));
        assert_eq!(size_short_edge(1500, 1000, 1024, 2048), (1536, 1024));
        assert_eq!(size_short_edge(6000, 4000, 1024, 2048), (3072, 2048));
        // the vision tower's copy: a 1,024 short edge
        assert_eq!(size_short_edge(6000, 4000, 1024, 1024), (1536, 1024));
    }

    #[test]
    fn the_presets_reach_16_mp() {
        assert_eq!(size_at(16_777_216, 1.0), (4096, 4096));
        assert_eq!(size_at(4_194_304, 1.0), (2048, 2048));
        assert_eq!(size_at(1_048_576, 1.0), (1024, 1024));
        for r in [1.0, 4.0 / 3.0, 3.0 / 4.0, 1.5, 2.0 / 3.0, 16.0 / 9.0, 9.0 / 16.0] {
            let (w, h) = size_at(16_777_216, r);
            assert!(w as u64 * h as u64 <= MAX_PIXELS && w % 32 == 0 && h % 32 == 0, "{w}x{h}");
        }
    }

    #[test]
    fn only_the_block_matrices_are_expanded_by_the_silo() {
        assert!(is_block_matrix("blocks.0.attn.qkv_proj.weight"));
        assert!(is_block_matrix("blocks.49.mlp.fc2.weight"));
        assert!(!is_block_matrix("blocks.3.attn.q_norm.weight"));
        assert!(!is_block_matrix("blocks.x.mlp.fc1.weight"));
        assert!(!is_block_matrix("txt_refiner.blocks.0.mlp.fc1.weight"));
        assert!(!is_block_matrix("blocks.1.mlp.fc1.weight_scale"));
    }

    #[test]
    fn q3k_block_matrices_are_kept_as_stored() {
        use nextsycl_h3::gguf::GType;
        use nextsycl_h3::safetensors::Entry;
        let entry = |name: &str| Entry { name: name.into(), dtype: DType::U8, shape: vec![512, 256], offset: 0, bytes: GType::Q6K.bytes(512 * 256),
                                         kquant: Some(GType::Q6K), stored: Some(GType::Q3K) };
        let mut ck = Checkpoint { path: "x.gguf".into(), entries: Default::default(), metadata: Default::default(), file_bytes: 0 };
        for n in ["blocks.0.mlp.fc1.weight", "blocks.7.attn.out_proj.weight", "token_refiner.blocks.0.mlp.fc1.weight"] {
            ck.entries.insert(n.into(), entry(n));
        }
        let mut q4 = entry("blocks.1.mlp.fc2.weight");
        (q4.kquant, q4.stored, q4.bytes) = (Some(GType::Q4K), None, GType::Q4K.bytes(512 * 256));
        ck.entries.insert(q4.name.clone(), q4);
        assert_eq!(keep_q3k(&mut ck), 2);
        let e = &ck.entries["blocks.0.mlp.fc1.weight"];
        assert_eq!((e.kquant, e.stored, e.bytes, e.file_bytes()), (Some(GType::Q3K), None, 512 * 110, 512 * 110));
        assert_eq!(e.stored_shape(), vec![512 * 110]);
        // the refiner's stays transcoded (the shared kernels expand it), the Q4_K one is untouched
        let r = &ck.entries["token_refiner.blocks.0.mlp.fc1.weight"];
        assert_eq!((r.kquant, r.stored), (Some(GType::Q6K), Some(GType::Q3K)));
        assert_eq!(ck.entries["blocks.1.mlp.fc2.weight"].kquant, Some(GType::Q4K));
        assert_eq!(keep_q3k(&mut ck), 0);
    }

    #[test]
    fn products_route_by_kind() {
        use ffi::{route, Route};
        use nextsycl_h3::gguf::GType;
        assert_eq!(route(GType::Q8_0, true, true), Route::Q8);
        assert_eq!(route(GType::Q8_0, true, false), Route::Expand);
        assert_eq!(route(GType::Q8_0, false, false), Route::Shared);
        assert_eq!(route(GType::Q4K, true, true), Route::Expand);
        assert_eq!(route(GType::Q6K, false, true), Route::Shared);
    }

    #[test]
    fn the_silo_takes_ggufs_codes() {
        use nextsycl_h3::gguf::GType;
        assert_eq!([GType::Q8_0, GType::Q3K, GType::Q4K, GType::Q5K].map(ffi::gguf_code), [Some(8), Some(11), Some(12), Some(13)]);
        assert_eq!(ffi::gguf_code(GType::Q6K), None);
        // the codes are the GGUF file's own type numbers, as the shared kernels take them
        for k in [GType::Q8_0, GType::Q4K, GType::Q5K] {
            assert_eq!(ffi::gguf_code(k), k.quant_code());
        }
    }

    fn pic() -> Picture {
        Picture { width: 64, height: 64, channels: 3, data: vec![9; 64 * 64 * 3] }
    }

    fn edit(strength: f32, condition: bool) -> Option<nextsycl_image::Edit> {
        Some(nextsycl_image::Edit { image: pic(), refs: Vec::new(), mask: None, strength, condition })
    }

    #[test]
    fn a_request_plans_its_grid() {
        // text to image on the card's 2-step shifted schedule (pDMD: shift 12)
        let req = ImageRequest { prompt: "x".into(), ..Default::default() };
        let p = plan(&req, 2, Some(12.0)).unwrap();
        assert_eq!(p, Plan { sigmas: net::shifted_sigmas(12.0, 2), init: false, condition: false, first_head: None });
        // a request's steps and shift come first; no shift anywhere: the heads' grid
        let p = plan(&ImageRequest { steps: Some(4), shift: Some(3.0), ..req.clone() }, 2, Some(12.0)).unwrap();
        assert_eq!(p.sigmas, net::shifted_sigmas(3.0, 4));
        assert_eq!(plan(&req, 6, None).unwrap().sigmas, net::PDD_SIGMAS.to_vec());
        // a request LoRA's card brings its steps and shift as options (the pDMD LoRA on the plain 6-step card)
        let mut r = req.clone();
        r.extra.insert("steps".into(), "2".into());
        r.extra.insert("shift".into(), "12".into());
        r.extra.insert("heads".into(), "first".into());
        let p = plan(&r, 6, None).unwrap();
        assert_eq!((p.sigmas.clone(), p.first_head), (net::shifted_sigmas(12.0, 2), Some(true)));
        // the request's own fields win over them
        let p = plan(&ImageRequest { steps: Some(3), shift: Some(5.0), ..r.clone() }, 6, None).unwrap();
        assert_eq!(p.sigmas, net::shifted_sigmas(5.0, 3));
        r.extra.insert("heads".into(), "all".into());
        assert!(plan(&r, 6, None).is_err());
        // an edit at strength 1: from noise, the picture a reference
        let p = plan(&ImageRequest { edit: edit(1.0, true), ..req.clone() }, 2, Some(12.0)).unwrap();
        assert!(p.condition && !p.init && p.sigmas[0] == 1.0);
        // an img2img (not a reference) at 0.5: ComfyUI's denoise (3 steps of the 6-step schedule)
        let p = plan(&ImageRequest { edit: edit(0.5, false), steps: Some(3), ..req.clone() }, 2, Some(12.0)).unwrap();
        assert!(!p.condition && p.init);
        assert_eq!(p.sigmas, net::schedule(3, Some(12.0), 0.5));
        // start-sigma: the Space's grid from exactly there, the picture the start
        let mut r = ImageRequest { edit: edit(1.0, false), steps: Some(3), ..req.clone() };
        r.extra.insert("start-sigma".into(), "0.55".into());
        let p = plan(&r, 2, Some(12.0)).unwrap();
        assert!(p.init && !p.condition && (p.sigmas[0] - 0.55).abs() < 1e-6 && p.sigmas.len() == 4);
        // an edit that keeps its picture as a reference and starts from it
        let mut r = ImageRequest { edit: edit(1.0, true), ..req.clone() };
        r.extra.insert("start-sigma".into(), "0.5".into());
        let p = plan(&r, 2, Some(12.0)).unwrap();
        assert!(p.init && p.condition && p.sigmas[0] == 0.5);
    }

    #[test]
    fn a_request_plan_refuses_what_it_cannot_do() {
        let req = ImageRequest { prompt: "x".into(), ..Default::default() };
        // a picture that is neither a reference nor a start
        assert!(plan(&ImageRequest { edit: edit(1.0, false), ..req.clone() }, 2, None).is_err());
        let mut r = req.clone();
        r.extra.insert("start-sigma".into(), "0.5".into());
        assert!(plan(&r, 2, None).is_err(), "start-sigma without a picture");
        let mut r = ImageRequest { edit: edit(1.0, false), ..req.clone() };
        r.extra.insert("start-sigma".into(), "1.5".into());
        assert!(plan(&r, 2, None).is_err());
        r.extra.insert("start-sigma".into(), "0.5".into());
        r.extra.insert("sigmas".into(), "0.5,0".into());
        assert!(plan(&r, 2, None).is_err(), "both grids");
        assert!(plan(&ImageRequest { steps: Some(0), ..req.clone() }, 2, None).is_err());
        assert!(plan(&ImageRequest { steps: Some(51), ..req }, 2, None).is_err());
    }

    #[test]
    fn the_start_mixes_noise_and_picture_at_the_first_sigma() {
        let mut x = vec![1.0f32, -1.0];
        start_latent(&mut x, Some(&[0.5, 0.25]), 0.4).unwrap();
        assert!((x[0] - (0.4 + 0.6 * 0.5)).abs() < 1e-6 && (x[1] - (-0.4 + 0.6 * 0.25)).abs() < 1e-6);
        let mut x = vec![2.0f32];
        start_latent(&mut x, None, 1.0).unwrap();
        assert_eq!(x, vec![2.0]);
        start_latent(&mut x, None, 0.5).unwrap();
        assert_eq!(x, vec![1.0]);
        assert!(start_latent(&mut x, Some(&[1.0, 2.0]), 0.5).is_err());
    }

    #[test]
    fn the_refiner_blocks_are_counted() {
        use nextsycl_h3::safetensors::Entry;
        let mut ck = Checkpoint { path: "x".into(), entries: Default::default(), metadata: Default::default(), file_bytes: 0 };
        for n in ["token_refiner.blocks.0.mlp.fc1.weight", "token_refiner.blocks.1.norm1.weight", "token_refiner.final_norm.weight", "blocks.7.mlp.fc1.weight"] {
            ck.entries.insert(n.into(), Entry { name: n.into(), dtype: DType::BF16, shape: vec![1], offset: 0, bytes: 2, kquant: None, stored: None });
        }
        assert_eq!(refiner_blocks(&ck), 2);
        ck.entries.clear();
        assert_eq!(refiner_blocks(&ck), 0);
    }

    #[test]
    fn a_request_grid_falls_to_zero() {
        assert_eq!(parse_sigmas("1, 0.5, 0").unwrap(), vec![1.0, 0.5, 0.0]);
        assert!(parse_sigmas("1,0.5").is_err());
        assert!(parse_sigmas("0.5,0.7,0").is_err());
    }
}
