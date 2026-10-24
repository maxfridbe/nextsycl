//! Hunyuan3D 2.1's paint model on its own: a mesh (from any engine) painted like a picture, without the shape model
//! beside it. The paint weights are ~6.5 GiB, so this is the form a 12 GB card can hold; the full 2.1 engine keeps
//! the shape DiT as well and paints what it has made (`Hunyuan3d::generate`, which also takes a given mesh).

use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{EngineOption, Gpu};
use nextsycl_model3d::{Abilities, Defaults, Error, LoadOptions, Mesh, MeshRequest, Model3dEngine, ModelFiles, Result, Step};

use crate::ops::Ops;
use crate::paint::Paint;
use crate::{paint_from, ARCH_PAINT, OPTIONS};

pub struct Painter {
    ops: Ops,
    paint: Paint,
    settings: LoadOptions,
    loaded: Instant,
    load_seconds: f64,
    weights_bytes: u64,
    busy: std::sync::Mutex<()>,
    /// the card's least free memory seen during the last request (bytes), for `report`
    low: std::sync::atomic::AtomicU64,
}

impl Painter {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Painter> {
        let t0 = Instant::now();
        let ops = Ops::new(gpu)?;
        let paint = paint_from(files, gpu, log)?;
        gpu.sync()?;
        let load_seconds = t0.elapsed().as_secs_f64();
        let weights_bytes = paint.bytes as u64;
        log(format!("{ARCH_PAINT}: loaded on {} in {load_seconds:.1} s ({:.2} GiB of weights)", gpu.name, weights_bytes as f64 / (1u64 << 30) as f64));
        Ok(Painter { ops, paint, settings: o.clone(), loaded: Instant::now(), load_seconds, weights_bytes, busy: std::sync::Mutex::new(()), low: u64::MAX.into() })
    }

    fn note_memory(&self) {
        if let Ok((_, Some(free))) = self.ops.gpu.memory() {
            self.low.fetch_min(free, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Model3dEngine for Painter {
    fn arch(&self) -> &'static str {
        ARCH_PAINT
    }
    fn abilities(&self) -> Abilities {
        // paints a mesh it is given; makes none (generate: false)
        Abilities { generate: false, texture: true, views: 1, rig: false, animate: false }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 15, cfg: 0.0, resolution: 0, faces: 40_000, texture_size: 2048, frames: 0, fps: 0.0 }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn report(&self) -> Vec<String> {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let mut r = vec![format!("{ARCH_PAINT} on {}: {:.2} GiB of weights, up {:.0} s", self.ops.gpu.name, gib(self.weights_bytes), self.loaded.elapsed().as_secs_f64())];
        if let Ok((total, Some(free))) = self.ops.gpu.memory() {
            r.push(format!("card: {:.1} of {:.1} GiB free", gib(free), gib(total)));
            let low = self.low.load(std::sync::atomic::Ordering::Relaxed);
            if low != u64::MAX {
                r.push(format!("the last request: the card at most {:.1} GiB used ({:.1} GiB free at the lowest)", gib(total - low), gib(low)));
            }
        }
        let (k, t, spills) = *self.paint.last_arenas.lock().unwrap();
        if k + t > 0 {
            r.push(format!("the last request's arenas: kept {:.2} GiB, temporaries {:.2} GiB, {spills} spills", gib(k as u64), gib(t as u64)));
        }
        r
    }
    fn generate(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let _one = self.busy.lock().map_err(|_| Error("a previous request failed midway".into()))?;
        self.low.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();
        let mesh = req.mesh.as_ref().ok_or_else(|| Error(format!("{ARCH_PAINT} paints a mesh it is given (mesh or file); it makes none")))?;
        let view = req.views.first().ok_or_else(|| Error("no picture (what the mesh is painted like)".into()))?;
        let steps = match req.extra.get("paint-steps").cloned().or_else(|| self.settings.setting("NS_M3D_HY_PAINT_STEPS")) {
            Some(v) => v.parse::<usize>().map_err(|_| Error(format!("paint-steps: {v}?")))?,
            None => 15,
        };
        let size = req.texture_size.unwrap_or(2048).clamp(256, 8192);
        let mut report = |phase: &'static str, at: u32, of: u32| progress(Step { phase, at, of, seconds: t0.elapsed().as_secs_f64() });
        let given = nextsycl_model3d::finish(mesh.clone(), Some(req.faces.unwrap_or(40_000)));
        let painted = self.paint.paint(&self.ops, &given, &view.image, req.seed, steps, size, None, &mut |ph, a, b| report(ph, a, b))?;
        self.note_memory();
        Ok(nextsycl_model3d::finish(painted, None))
    }
}
