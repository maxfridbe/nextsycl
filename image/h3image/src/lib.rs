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

pub mod check;
pub mod ffi;
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
use nextsycl_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule as Sched,
                     SizePresets, Step};

pub const ARCH: &str = "minimax-h3-image";
/// the denoiser, the text encoder (H3's Qwen3-VL 32B GGUF), its tokenizer (vocab.json beside merges.txt and
/// tokenizer_config.json), the VAE (the encoder for edits, the image decoder), the vision tower (edits)
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "tokenizer", "vae", "vision"];

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
    EngineOption { name: "sigmas", env: "", value: "S,S,..,0", help: "another sampling grid than the heads' six steps: strictly falling from 1, ending at 0", at: At::Request },
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
        let ck = Checkpoint::open(role(files, "denoiser")?).map_err(h3e)?;
        let mut blocks = nextsycl_h3::dit::Blocks::load(&dev, &ck, None, threads).map_err(h3e)?;
        // the silo's attention (dist/silo/libnextsycl-h3image.so: P-H3-Image's, copied) from NS_H3I_SAGE_MIN_S tokens;
        // without it, or below that, the shared kernels' (SageAttention from 8,192 tokens, oneDNN's below)
        if let Some(s) = ffi::silo().map_err(|e| Error(e.0))? {
            let min_s = o.setting("NS_H3I_SAGE_MIN_S").and_then(|v| v.parse().ok()).unwrap_or(1024);
            blocks.attention = Some(ffi::attention(&gpu, s, min_s));
            log(format!("{ARCH}: the {} silo's attention from {min_s} tokens", ffi::SILO));
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
        Ok(H3Image { gpu, dev, ck, blocks, outer, decoder, files: files.clone(), threads, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(),
                     busy: Mutex::new(()) })
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

    /// The conditioning rows [T, 5120] through the denoiser's projection and token refiner: bfloat16 [T, hidden]
    pub fn refine(&self, cond: &[f32]) -> Result<Tensor> {
        let k = self.ck.get("condition_proj.weight").map_err(h3e)?.shape[1];
        let t = cond.len() / k;
        let c = Tensor::from_bytes(&self.dev, DType::BF16, &[t, k], &bf16_bytes(cond)).map_err(h3e)?;
        let refiner = TextRefiner::load(&self.dev, &self.ck, self.threads).map_err(h3e)?;
        let out = refiner.run(&self.blocks.cfg, &c, None).map_err(h3e)?;
        self.dev.wait().map_err(h3e)?;
        Ok(out)
    }

    /// An edit's pictures as conditioning latents [24, h, w] (normalized), each at its VAE size (short edge within
    /// 1,024 .. 2,048)
    pub fn reference_latents(&self, pics: &[Picture]) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        let enc = nextsycl_h3::venc::VideoEncoder::load(&self.dev, &Checkpoint::open(role(&self.files, "vae")?).map_err(h3e)?).map_err(h3e)?;
        let (mean, std) = enc.latent_stats();
        let mut out = Vec::with_capacity(pics.len());
        for p in pics {
            let rgb = p.rgb();
            let (w, h) = size_short_edge(rgb.width, rgb.height, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE);
            let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
            let px: Vec<f32> = shown.data.iter().map(|v| *v as f32 / 255.0).collect();
            let m = enc.moments_of_picture(&px, h as usize, w as usize, &mut || Ok(())).map_err(h3e)?;
            let (lh, lw) = (h as usize / 16, w as usize / 16);
            let noise = Mt19937::new(ENCODE_SEED).randn(24 * lh * lw);
            out.push((vae::condition_latent(&m, &noise, mean, std), lh, lw));
        }
        Ok(out)
    }

    /// The latents [24, lh, lw] of one picture: the refined text `text` with `tags`, the edit's latents `refs`, the
    /// noise from `seed`, the grid `sigmas`; `step(i)` after each step (an error stops)
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, text: Tensor, tags: &[i32], refs: &[(Vec<f32>, usize, usize)], lh: usize, lw: usize, seed: u64, sigmas: &[f32],
                  step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let mut x = Mt19937::new(seed).randn(net::LATENT_C * lh * lw);
        let mut run = net::Run::new(&self.blocks, &self.outer, text, Some(tags.to_vec()), lh, lw, refs, seed).map_err(h3e)?;
        for i in 0..sigmas.len() - 1 {
            let v = run.net(&x, sigmas[i], sigmas[i + 1], &mut |_| Ok(())).map_err(h3e)?;
            net::euler(&mut x, &v, sigmas[i], sigmas[i + 1]);
            step(i)?;
        }
        Ok(x)
    }
}

impl ImageEngine for H3Image {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: DEFAULT_SIDE, height: DEFAULT_SIDE, steps: STEPS as u32, cfg: 1.0, sampler: Sampler::Euler, schedule: Sched::Simple, shift: 1.0 }
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
        match (req.extra.get("sigmas"), req.steps) {
            (Some(s), _) => s.split(',').count().saturating_sub(1) as u32,
            (None, Some(s)) => s,
            (None, None) => STEPS as u32,
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
        let sigmas = match (req.extra.get("sigmas"), req.steps) {
            (Some(s), _) => parse_sigmas(s)?,
            (None, Some(s)) if (1..=50).contains(&s) => net::sigmas(s as usize),
            (None, Some(s)) => return Err(Error(format!("{ARCH}: {s} steps (1 to 50; the heads are distilled for {STEPS})"))),
            (None, None) => net::sigmas(STEPS),
        };
        let of = (sigmas.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let mut log = |l: String| eprintln!("{l}");
        let cond = self.encode(&req.prompt, &pics, &mut log)?;
        let t_text = t0.elapsed().as_secs_f64();
        let refs = if pics.is_empty() { Vec::new() } else { self.reference_latents(&pics)? };
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
            log(format!("{ARCH} {w}x{h} picture {p}: text {:.2} s ({} tokens, {} pictures), pictures' latents {:.2} s, refiner {t_refine:.2} s, {of} steps {t_steps:.2} s, decode {:.2} s",
                        if p == 0 { t_text } else { 0.0 }, cond.tags.len(), pics.len(), if p == 0 { t_refs } else { 0.0 },
                        t1.elapsed().as_secs_f64() - t_refine - t_steps));
            out.push(Picture { width: w, height: h, channels: 3, data: rgb });
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        let used = self.dev.mem_used() as f64 / (1u64 << 30) as f64;
        vec![format!("{ARCH} in Rust and SYCL on {} ({used:.1} GiB, loaded {:.0} s ago)", self.gpu_name(), self.loaded.elapsed().as_secs_f64())]
    }
}

impl H3Image {
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
    fn a_request_grid_falls_to_zero() {
        assert_eq!(parse_sigmas("1, 0.5, 0").unwrap(), vec![1.0, 0.5, 0.0]);
        assert!(parse_sigmas("1,0.5").is_err());
        assert!(parse_sigmas("0.5,0.7,0").is_err());
    }
}
