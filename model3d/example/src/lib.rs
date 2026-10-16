//! A template 3D engine: every part of the `nextsycl_model3d` contract in place and documented, nothing ported. Copy
//! this crate (and `kernels/model3d/example/`) to port a 3D model - CONTRIBUTING.md walks through it.
//!
//! What a port fills in, in order:
//! 1. its files and their roles (`ROLES`): read each, checked, so a wrong file fails at load with a reason.
//! 2. its kernels (`kernels/model3d/<arch>/`, SYCL only) and their C ABI (`<arch>.h`), bound in `ffi.rs`.
//! 3. what it does (`abilities`) and its pipeline: `generate` - the pictures cut out and encoded, the shape's latents
//!    denoised from the seed, decoded to a surface, meshed, textured (`nextsycl_mesh::tools::{unwrap, rasterize}`),
//!    finished (`nextsycl_model3d::finish`) - or `rig` - the mesh sampled, its skeleton and skin weights (or its
//!    motion) predicted onto it. Report progress as it goes (an error from `progress` is a cancel).
//! 4. its registry entry (`kind()`), added to the program's 3D engines.
//!
//! Until then `load` checks that the 3D kernel library and the GPU work - it runs the example kernel - and fails
//! with a clear message.

mod ffi;

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_model3d::{Abilities, Defaults, Error, LoadOptions, Mesh, MeshRequest, Model3dEngine, Model3dKind, ModelFiles, Result, Step};

/// The architecture this engine serves (a registry entry's "arch")
pub const ARCH: &str = "example-model3d";

/// The files it needs, by role
pub const ROLES: &[&str] = &["dit", "decoder"];

/// This engine's registry entry
pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "Example (a template 3D engine: nothing ported)", roles: ROLES, load,
                  // the options a port takes beyond the request's fields (`--opt-NAME`), each with the variable it sets
                  options: &[] }
}

/// The path to the GPU, end to end: this kind's kernel library, this engine's own symbol, a kernel run on `gpu` and
/// its result read back - [1, 2, 3, 4] x 2 (`nextsycl model3d selftest`)
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
    Error(format!("the example 3D engine is a template: {what} is not ported (CONTRIBUTING.md)"))
}

/// Load the model onto `gpus`. A port reads its files here (all of `ROLES`) and plans the memory; the template proves
/// the path to the GPU - the 3D kernel library, this engine's own symbol, a kernel run and its result - then stops.
fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let missing = nextsycl_model3d::missing(files, ROLES);
    if !missing.is_empty() {
        return Err(Error(format!("{ARCH}: no {} file", missing.join(", "))));
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

impl Model3dEngine for Example {
    fn arch(&self) -> &'static str {
        ARCH
    }
    /// what it does: a generator says how many pictures it takes and whether it paints; a rigger whether it animates
    fn abilities(&self) -> Abilities {
        Abilities { generate: true, texture: false, views: 1, rig: false, animate: false }
    }
    /// what a request gets when it leaves a setting out
    fn defaults(&self) -> Defaults {
        Defaults { steps: 25, cfg: 5.0, resolution: 256, faces: 200_000, texture_size: 2048, frames: 0, fps: 0.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.loaded.elapsed().as_secs_f64()
    }
    /// The pipeline: the pictures encoded (report it), the shape made from the seed (report each step), decoded and
    /// meshed, textured when asked, finished.
    fn generate(&self, _req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        progress(Step { phase: "image", at: 0, of: 1, seconds: 0.0 })?;
        Err(not_ported("generate"))
    }
}
