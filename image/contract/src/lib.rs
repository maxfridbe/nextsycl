//! The contract between the image server / command line and the image engines. An engine is one model architecture's
//! whole pipeline - text encoder, denoiser, sampler steps, VAE decode - with its own kernels (`kernels/image/<arch>`)
//! and memory plan. The server drives it only through `ImageEngine`; it is chosen by the architecture its registry
//! entry names (`ImageKind`). Samplers, schedules, LoRAs and pictures are `nextsycl_diffusion`'s.

pub mod layout;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use nextsycl_core::options::Given;
pub use nextsycl_core::{At, EngineOption, Error, Gpu, Result};
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
    /// the engine's own options for this request, by name (`--opt-NAME`, an API request's `options`), checked
    /// against those it declares for requests
    pub extra: Given,
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
    /// the pictures also condition the model as an edit's (its references, read by its vision tower); false: a plain
    /// img2img - the denoise starts from `image` (strength below 1) and the prompt alone guides it
    pub condition: bool,
}

/// One reading (an OCR model's request): pictures in reading order and the instruction
#[derive(Clone, Debug, Default)]
pub struct OcrRequest {
    /// the pages, in order (a single picture: its tiles are read too; several: each page whole)
    pub pages: Vec<Picture>,
    /// the instruction ("document parsing.", "Free OCR.", "<|grounding|>Convert the document to markdown.", ...);
    /// empty: the engine's default; several pages are read under the model's multi-page instruction
    pub prompt: String,
    /// read a single picture's tiles as well as the whole (more detail, more tokens); None: the engine's default
    pub tiles: Option<bool>,
    /// tokens to write at most (0: the engine's default)
    pub max_tokens: u32,
    /// no n-gram of this many tokens repeated within the window (0: off; None: the model's own setting)
    pub ngram: Option<u32>,
    pub ngram_window: Option<u32>,
    /// sampling temperature (0: greedy - the model's own setting for reading)
    pub temperature: f32,
    pub seed: u64,
    /// read the text inside the pictures the model only marks (a second pass a picture); None: the engine's default
    pub figures: Option<bool>,
    /// find columns (blocks side by side, a gutter between) and read each on its own, in reading order; None: the
    /// engine's default
    pub columns: Option<bool>,
    /// several pages: each read on its own (its tiles too), the readings joined with `<PAGE>` marks - else the model's
    /// own multi-page reading in one pass (each page seen whole); None: the engine's default
    pub each_page: Option<bool>,
    /// the engine's own options for this request, by name
    pub extra: Given,
}

/// An instruction an OCR model is trained on: its id, the text, what it reads
#[derive(Clone, Debug)]
pub struct OcrPrompt {
    pub id: &'static str,
    pub prompt: &'static str,
    pub title: &'static str,
}

/// What a reading made
#[derive(Clone, Debug, Default)]
pub struct OcrOutput {
    /// the text as the model wrote it (its layout markup kept: boxes, tables; several pages: a `<PAGE>` mark before each)
    pub text: String,
    /// the instruction it read under (several pages: the model's multi-page one)
    pub prompt: String,
    pub prompt_tokens: u32,
    pub tokens: u32,
    /// seconds to see the pages and read the prompt, then to write
    pub prompt_seconds: f64,
    pub seconds: f64,
}

/// One upscale (an upscaler's request): the picture and the variant's settings. `None` fields take the engine's
/// defaults
#[derive(Clone, Debug)]
pub struct UpscaleRequest {
    pub picture: Picture,
    /// the variant of the family (Real-ESRGAN: realesrgan-x4plus, realesrgan-x4plus-anime, realesr-animevideov3;
    /// waifu2x: models-cunet, models-upconv_7_anime_style_art_rgb, models-upconv_7_photo)
    pub variant: Option<String>,
    /// the factor (Real-ESRGAN: 2, 3, 4 as the variant takes; waifu2x: 1, 2, 4 ... 32)
    pub scale: Option<u32>,
    /// waifu2x's denoise level, -1 (none) to 3
    pub noise: Option<i32>,
    /// the tile's side in input pixels (the picture goes through the model in tiles this big, overlapped)
    pub tile: Option<u32>,
}

/// What an upscale was made with
#[derive(Clone, Debug, Default)]
pub struct UpscaleInfo {
    pub variant: String,
    pub scale: u32,
    pub noise: Option<i32>,
    pub tile: u32,
    pub tiles: u32,
    pub seconds: f64,
}

/// One vectorization (a picture-to-SVG model's request: StarVector). `None` fields take the model's own settings (its
/// generate_im2svg's: 2 beams sampled, top_p 0.9, temperature 1)
#[derive(Clone, Debug)]
pub struct VectorizeRequest {
    pub picture: Picture,
    /// the sequence's length at most, as the model's `max_length` counts it (the picture's embedding rows and the prompt
    /// included); 0: the engine's default
    pub max_length: u32,
    /// beams (1: no beam search)
    pub beams: Option<u32>,
    /// sample (true: the model's own setting) or take the most likely tokens
    pub sample: Option<bool>,
    pub top_p: Option<f32>,
    pub temperature: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: u64,
}

impl Default for VectorizeRequest {
    fn default() -> Self {
        VectorizeRequest { picture: Picture::new(1, 1, 3), max_length: 0, beams: None, sample: None, top_p: None, temperature: None, repetition_penalty: None, seed: 0 }
    }
}

/// What a vectorization made
#[derive(Clone, Debug, Default)]
pub struct VectorizeOutput {
    /// the SVG as the model wrote it ("<svg" and the tokens after it, decoded as its tokenizer decodes): it may stop
    /// short of "</svg>" at the length limit (the caller closes or cuts it)
    pub svg: String,
    /// it ended with "</svg>" (else the length limit stopped it)
    pub closed: bool,
    pub tokens: u32,
    pub prompt_tokens: u32,
    /// seconds to see the picture and read the prompt, then to write
    pub prompt_seconds: f64,
    pub seconds: f64,
}

/// One variant an upscaler offers, for `info` and a page
#[derive(Clone, Debug)]
pub struct UpscaleVariant {
    pub id: &'static str,
    pub about: &'static str,
    pub scales: &'static [u32],
    /// the denoise levels it takes (empty: none)
    pub noise: &'static [i32],
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
    /// settings by name (a registry entry's and its LoRAs' `env`): what the process environment would hold when the
    /// engine is served; an engine reads a setting here first, then from the environment
    pub settings: BTreeMap<String, String>,
}

impl LoadOptions {
    /// A setting: from `settings`, else the environment
    pub fn setting(&self, name: &str) -> Option<String> {
        self.settings.get(name).cloned().or_else(|| std::env::var(name).ok())
    }
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
    /// whether it takes `ImageRequest::loras` (a request's LoRAs on a side path, the weights as loaded); else a server
    /// reloads the model with them merged
    fn side_loras(&self) -> bool {
        false
    }
    /// the samplers it runs (a request naming another is refused)
    fn samplers(&self) -> Vec<Sampler> {
        Sampler::ALL.to_vec()
    }
    /// the schedules it runs
    fn schedules(&self) -> Vec<Schedule> {
        Schedule::ALL.to_vec()
    }
    /// whether it takes guidance (cfg above 1, a negative prompt)
    fn guidance(&self) -> bool {
        true
    }
    /// the options it takes beyond the request's fields (its kind's `options`)
    fn options(&self) -> &'static [EngineOption] {
        &[]
    }
    /// the steps a request runs (its own, else the defaults; an engine whose options change it says so)
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        req.steps.unwrap_or(self.defaults().steps)
    }
    /// seconds the load took
    fn load_seconds(&self) -> f64;
    /// make the request's pictures, reporting each step (an error from `progress` stops it there: a cancel)
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>>;
    /// lines for `status`: its GPUs, memory, what is loaded where
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
    /// whether it reads pictures (an OCR model: `ocr`) rather than makes them
    fn reads(&self) -> bool {
        false
    }
    /// the instructions it reads by (the first is its default), for a page to offer
    fn ocr_prompts(&self) -> Vec<OcrPrompt> {
        Vec::new()
    }
    /// read the request's pages, each piece of text as it is written to `text`
    fn ocr(&self, _req: &OcrRequest, _text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        Err(Error(format!("{} makes pictures; it does not read them", self.arch())))
    }
    /// the variants it upscales with (an upscaler: `upscale`); empty for every other engine
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        Vec::new()
    }
    /// a picture made bigger, reporting each tile done (`Step::at` of `of`)
    fn upscale(&self, _req: &UpscaleRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Picture, UpscaleInfo)> {
        Err(Error(format!("{} does not upscale (an upscaler: real-esrgan, waifu2x)", self.arch())))
    }
    /// whether it writes a picture as SVG (`vectorize`: StarVector)
    fn vectorizes(&self) -> bool {
        false
    }
    /// the picture as SVG, each piece of text as it is written to `text` (an error from `text` stops it: a cancel)
    fn vectorize(&self, _req: &VectorizeRequest, _text: &mut dyn FnMut(&str) -> Result<()>) -> Result<VectorizeOutput> {
        Err(Error(format!("{} does not vectorize pictures (starvector does)", self.arch())))
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
    /// the options it takes (`--opt-NAME`: nextsycl_core::options) - at load as the variables they name (in
    /// `LoadOptions::settings` too), per request in `ImageRequest::extra`
    pub options: &'static [EngineOption],
}

/// The entry of `kinds` serving `arch`
pub fn kind_for<'k>(kinds: &'k [ImageKind], arch: &str) -> std::result::Result<&'k ImageKind, String> {
    kinds.iter().find(|k| k.archs.contains(&arch)).ok_or_else(|| {
        let known: Vec<&str> = kinds.iter().flat_map(|k| k.archs.iter().copied()).collect();
        format!("image architecture {arch:?} has no engine here (known: {})", known.join(", "))
    })
}
