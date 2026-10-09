//! The contract between the video server / studio / command line and the video engines. An engine is one model
//! architecture's whole pipeline - text encoder, denoiser, sampler steps, upscaler, video (and audio) decode, the
//! container - with its own kernels (`kernels/video/<arch>`). A clip is a long job, so the engine writes it to a file
//! and reports progress by stage; the queue, the studio and the tools built on clips (speeches, scenes, joins) sit
//! above this contract. Samplers, schedules, LoRAs and pictures are `nextsycl_diffusion`'s.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use nextsycl_core::options::Given;
pub use nextsycl_core::{At, EngineOption, Error, Gpu, Result};
pub use nextsycl_diffusion::{LoraUse, Picture, Sampler, Schedule};

/// One clip: the prompt and its settings. `None` fields take the engine's defaults.
#[derive(Clone, Debug, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seconds: Option<f32>,
    pub steps: Option<u32>,
    pub cfg: Option<f32>,
    pub seed: u64,
    pub sampler: Option<Sampler>,
    pub schedule: Option<Schedule>,
    pub shift: Option<f32>,
    /// upscale the decoded frames by this factor (1.0: none)
    pub upscale: Option<f32>,
    /// LoRAs for this clip only
    pub loras: Vec<LoraUse>,
    /// pictures the clip must pass through
    pub keyframes: Vec<Keyframe>,
    /// a voice or sound the clip's audio follows
    pub audio_ref: Option<PathBuf>,
    /// where the clip goes (an .mp4)
    pub out: PathBuf,
    /// the engine's own options for this clip, by name (`--opt-NAME`, an API request's `options`), checked against
    /// those it declares for requests
    pub extra: Given,
}

/// A picture the clip passes through, at a time
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub at_seconds: f32,
    pub picture: Picture,
    /// how closely (1.0: exactly)
    pub strength: f32,
}

/// The engine's own defaults
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub width: u32,
    pub height: u32,
    pub seconds: f32,
    pub fps: f32,
    pub steps: u32,
    pub cfg: f32,
    pub sampler: Sampler,
    pub schedule: Schedule,
    pub shift: f32,
    pub upscale: f32,
}

/// Progress, reported as the engine goes
#[derive(Clone, Debug)]
pub struct Progress {
    /// "encode", "denoise", "upscale", "decode", "mux"...: the engine's own stage names
    pub stage: &'static str,
    /// the stage's step done, of `of` (both 0 for a stage without steps)
    pub at: u32,
    pub of: u32,
    /// seconds since the clip started
    pub seconds: f64,
}

/// What came out
#[derive(Clone, Debug)]
pub struct Clip {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: f32,
    pub audio: bool,
}

/// A model's files by role ("denoiser", "text-encoder", "vae", "audio-vae", "upscaler", ...)
pub type ModelFiles = BTreeMap<String, PathBuf>;

/// How an engine is loaded
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// LoRAs merged into the weights at load (a preset's)
    pub merge_loras: Vec<LoraUse>,
    /// settings by variable name (a registry entry's `env`, the `--opt-NAME`s given): what the environment would
    /// hold when served; an engine reads a setting here first, then from the environment
    pub settings: std::collections::BTreeMap<String, String>,
}

impl LoadOptions {
    /// A setting: from `settings`, else the environment
    pub fn setting(&self, name: &str) -> Option<String> {
        self.settings.get(name).cloned().or_else(|| std::env::var(name).ok())
    }
}

/// The runtime one video architecture brings
pub trait VideoEngine: Send + Sync {
    fn arch(&self) -> &'static str;
    fn defaults(&self) -> Defaults;
    fn load_seconds(&self) -> f64;
    /// the options it takes beyond the request's fields (its kind's `options`)
    fn options(&self) -> &'static [EngineOption] {
        &[]
    }
    /// make the clip, reporting progress; `cancel()` true stops it at the next step boundary
    fn generate(&self, req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> Result<Clip>;
    /// give the GPUs' memory back (the next clip loads again)
    fn unload(&self) -> Result<()> {
        Ok(())
    }
    /// lines for `status`
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

/// How an engine loads: its files, its GPUs, the options, a log
pub type LoadFn = fn(&ModelFiles, &[Arc<Gpu>], &LoadOptions, &mut dyn FnMut(String)) -> Result<Box<dyn VideoEngine>>;

/// An engine's registry entry
pub struct VideoKind {
    pub archs: &'static [&'static str],
    pub name: &'static str,
    /// the file roles it needs
    pub roles: &'static [&'static str],
    pub load: LoadFn,
    /// the options it takes (`--opt-NAME`: nextsycl_core::options) - at load as the variables they name (in
    /// `LoadOptions::settings` too), per clip in `VideoRequest::extra`
    pub options: &'static [EngineOption],
}

/// The entry of `kinds` serving `arch`
pub fn kind_for<'k>(kinds: &'k [VideoKind], arch: &str) -> std::result::Result<&'k VideoKind, String> {
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("video architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}
