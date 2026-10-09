//! A template image engine: every part of the `nextsycl_image` contract in place and documented, nothing ported. Copy
//! this crate (and `kernels/image/example/`) to port an image model - CONTRIBUTING.md walks through it.
//!
//! What a port fills in, in order:
//! 1. its files and their roles (`ROLES`): read each, checked, so a wrong file fails at load with a reason.
//! 2. its kernels (`kernels/image/<arch>/`, SYCL only) and their C ABI (`<arch>.h`), bound in `ffi.rs`.
//! 3. the pipeline in `generate`: encode the prompt, draw the starting noise from the seed, run the sampler over the
//!    schedule's sigmas (`nextsycl_diffusion::sigmas`) calling the denoiser, decode the latents, return pictures.
//! 4. its registry entry (`kind()`), added to the program's image engines.
//!
//! Until then `load` checks that the image kernel library and the GPU work - it runs the example kernel - and fails
//! with a clear message.

mod ffi;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "example-image";

/// The files it needs, by role
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];

/// This engine's registry entry
pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Example (a template image engine: nothing ported)", roles: ROLES, load }
}

/// The path to the GPU, end to end: this kind's kernel library, this engine's own symbol, a kernel run on `gpu` and
/// its result read back - [1, 2, 3, 4] x 2 (`nextsycl image selftest`)
pub fn selftest(gpu: &Arc<Gpu>) -> Result<Vec<f32>> {
    let k = ffi::api()?;
    let x = DevBuf::from_f32(gpu, &[1.0, 2.0, 3.0, 4.0])?;
    // SAFETY: four floats of device memory on this GPU.
    ffi::check(unsafe { (k.scale)(gpu.raw(), x.fp(), 4, 2.0) }, "the example kernel")?;
    gpu.sync()?;
    let y = x.to_f32()?;
    if y != [2.0, 4.0, 6.0, 8.0] {
        return Err(Error(format!("the example kernel computed {y:?}, not [2, 4, 6, 8]")));
    }
    Ok(y)
}

fn not_ported(what: &str) -> Error {
    Error(format!("the example image engine is a template: {what} is not ported (CONTRIBUTING.md)"))
}

/// Load the model onto `gpus`. A port reads its files here (all of `ROLES`) and plans the memory; the template proves
/// the path to the GPU - the image kernel library, this engine's own symbol, a kernel run and its result - then stops.
fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    for role in ROLES {
        if !files.contains_key(*role) {
            return Err(Error(format!("{ARCH}: no {role} file")));
        }
    }
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let y = selftest(gpu)?;
    log(format!("{ARCH}: the example kernel ran on {}: [1, 2, 3, 4] x 2 = {y:?}", gpu.name));
    Err(not_ported("loading a model"))
}

/// The engine: its weights and plan on its GPUs
pub struct Example {
    loaded: Instant,
}

impl ImageEngine for Example {
    fn arch(&self) -> &'static str {
        ARCH
    }
    /// what a request gets when it leaves a setting out
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: 30, cfg: 4.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 3.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.loaded.elapsed().as_secs_f64()
    }
    /// The pipeline. For each of `req.n` pictures (seed + i): the text encoded (report step 0), the noise drawn, the
    /// sampler stepped over `sigmas(schedule, steps, shift)` with the denoiser (report each step), the latents
    /// decoded to a `Picture` (`req.rgba`: keep alpha).
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step)) -> Result<Vec<Picture>> {
        let d = self.defaults();
        progress(Step { picture: 0, at: 0, of: req.steps.unwrap_or(d.steps), seconds: 0.0 });
        Err(not_ported("generate"))
    }
}
