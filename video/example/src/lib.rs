//! A template video engine: every part of the `nextsycl_video` contract in place and documented, nothing ported. Copy
//! this crate (and `kernels/video/example/`) to port a video model - CONTRIBUTING.md walks through it.
//!
//! What a port fills in, in order:
//! 1. its files and their roles (`ROLES`): read each, checked, so a wrong file fails at load with a reason.
//! 2. its kernels (`kernels/video/<arch>/`, SYCL only) and their C ABI (`<arch>.h`), bound in `ffi.rs`.
//! 3. the pipeline in `generate`: encode the prompt (and keyframes, audio reference), denoise over the schedule's
//!    sigmas, upscale, decode the frames (and audio), write the container to `req.out` - reporting each stage, and
//!    stopping at a step boundary when `cancel()` says so.
//! 4. its registry entry (`kind()`), added to the program's video engines.
//!
//! Until then `load` checks that the video kernel library and the GPU work - it runs the example kernel - and fails
//! with a clear message.

mod ffi;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_video::{Clip, Defaults, Error, LoadOptions, ModelFiles, Progress, Result, Sampler, Schedule, VideoEngine, VideoKind, VideoRequest};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "example-video";

/// The files it needs, by role
pub const ROLES: &[&str] = &["denoiser", "text-encoder", "vae"];

/// This engine's registry entry
pub fn kind() -> VideoKind {
    VideoKind { archs: &[ARCH], name: "Example (a template video engine: nothing ported)", roles: ROLES, load,
                // the options a port takes beyond the request's fields (`--opt-NAME`), each with the variable it sets
                options: &[] }
}

/// The path to the GPU, end to end: this kind's kernel library, this engine's own symbol, a kernel run on `gpu` and
/// its result read back - [1, 2, 3, 4] x 2 (`nextsycl video selftest`)
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
    Error(format!("the example video engine is a template: {what} is not ported (CONTRIBUTING.md)"))
}

/// Load the model onto `gpus`. A port reads its files here (all of `ROLES`) and plans the memory; the template proves
/// the path to the GPU - the video kernel library, this engine's own symbol, a kernel run and its result - then stops.
fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn VideoEngine>> {
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

impl VideoEngine for Example {
    fn arch(&self) -> &'static str {
        ARCH
    }
    /// what a request gets when it leaves a setting out
    fn defaults(&self) -> Defaults {
        Defaults { width: 768, height: 576, seconds: 5.0, fps: 24.0, steps: 8, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 5.0,
                   upscale: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.loaded.elapsed().as_secs_f64()
    }
    /// The pipeline, stage by stage ("encode", "denoise" step by step, "upscale", "decode", "mux"), writing `req.out`.
    fn generate(&self, _req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> Result<Clip> {
        if cancel() {
            return Err(Error("cancelled".into()));
        }
        progress(Progress { stage: "encode", at: 0, of: 0, seconds: 0.0 });
        Err(not_ported("generate"))
    }
}
