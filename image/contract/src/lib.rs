//! The contract between the image server / command line and the image engines. An engine is one model architecture's
//! whole pipeline - text encoder, denoiser, sampler steps, VAE decode - with its own kernels (`kernels/image/<arch>`)
//! and memory plan. The server drives it only through `ImageEngine`; it is chosen by the architecture its registry
//! entry names (`ImageKind`). Samplers, schedules, LoRAs and pictures are `nextsycl_diffusion`'s.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use nextsycl_core::{Error, Gpu, Result};
pub use nextsycl_diffusion::{LoraUse, Picture, Sampler, Schedule};

/// One generation: the prompt and its settings. `None` fields take the engine's defaults.
#[derive(Clone, Debug, Default)]
pub struct ImageRequest {
    pub prompt: String,
    /// what to steer away from (classifier-free guidance's negative prompt)
    pub negative: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    /// guidance scale (1.0: no guidance - one model call a step)
    pub cfg: Option<f32>,
    pub seed: u64,
    /// pictures to make (seeds seed, seed + 1, ...)
    pub n: u32,
    pub sampler: Option<Sampler>,
    pub schedule: Option<Schedule>,
    pub shift: Option<f32>,
    /// LoRAs for this request only (a preset's LoRAs were merged at load)
    pub loras: Vec<LoraUse>,
    /// an edit: the picture to change and its instructions are `prompt`
    pub edit: Option<Edit>,
    /// keep an alpha channel (models that make transparency)
    pub rgba: bool,
}

/// What an edit starts from
#[derive(Clone, Debug)]
pub struct Edit {
    pub image: Picture,
    /// more reference pictures ("put this chair by the window")
    pub refs: Vec<Picture>,
    /// where the edit may change the picture (white: change)
    pub mask: Option<Picture>,
    /// how far from the original (1.0: as far as a new picture)
    pub strength: f32,
}

/// The engine's own defaults, shown by `info` and used for the request's `None` fields
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f32,
    pub sampler: Sampler,
    pub schedule: Schedule,
    pub shift: f32,
}

/// Progress, reported as the engine goes
#[derive(Clone, Copy, Debug)]
pub struct Step {
    /// which picture of the request (0-based)
    pub picture: u32,
    /// the denoising step done, of `of` (0 of n: the text is encoded)
    pub at: u32,
    pub of: u32,
    /// seconds since the request started
    pub seconds: f64,
}

/// A model's files by role ("transformer", "text-encoder", "vae", ...: the engine says which it needs)
pub type ModelFiles = BTreeMap<String, PathBuf>;

/// How an engine is loaded
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// LoRAs merged into the weights at load (a preset's)
    pub merge_loras: Vec<LoraUse>,
}

/// The runtime one image architecture brings
pub trait ImageEngine: Send + Sync {
    /// its architecture (the registry entry's "arch")
    fn arch(&self) -> &'static str;
    fn defaults(&self) -> Defaults;
    /// whether it takes `ImageRequest::edit`
    fn edits(&self) -> bool {
        false
    }
    /// seconds the load took
    fn load_seconds(&self) -> f64;
    /// make the request's pictures, reporting each step
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step)) -> Result<Vec<Picture>>;
    /// lines for `status`: its GPUs, memory, what is loaded where
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

/// How an engine loads: its files, its GPUs, the options, a log
pub type LoadFn = fn(&ModelFiles, &[Arc<Gpu>], &LoadOptions, &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>>;

/// An engine's registry entry
pub struct ImageKind {
    /// the architectures it serves (a registry entry's "arch", or a GGUF's general.architecture)
    pub archs: &'static [&'static str],
    /// what it is
    pub name: &'static str,
    /// the file roles it needs
    pub roles: &'static [&'static str],
    pub load: LoadFn,
}

/// The entry of `kinds` serving `arch`
pub fn kind_for<'k>(kinds: &'k [ImageKind], arch: &str) -> std::result::Result<&'k ImageKind, String> {
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("image architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}
