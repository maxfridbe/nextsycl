//! Hi3D v1.0 (hi3d-official/Hi3Dv1.0, built on Sparc3D): a coarse mesh and a picture of the same object in, a
//! high-resolution geometry-only mesh out (recentred, unit radius). The pipeline, as Sparc3DPipeline runs it:
//!
//! ```text
//!   picture -> framed (the alpha box squared on white, padded x 1.2)               prep.rs
//!           -> DINOv2-large tokens at 756 (2,917 x 1,024) | CLIP ViT-L/14's pooled 1,024, each L2-normed, CLIP's
//!              broadcast beside every DINOv2 token: [2917, 2048]                    vision.rs
//!   coarse mesh -> its surface voxels at 64^3 (shrunk 5 % at a time to <= 80,000)  vox.rs
//!   a pass (64 -> 512, then 128 -> 1024, then 192 -> 1536):
//!     noise on the voxels -> 25 Heun updates of the sparse flow transformer (512's at 64, 1024's above), guided
//!       against the zero condition from sigma 0.7 up                                dit.rs, sched.rs
//!     -> denormalized -> the self-pruning decoder: 3 doublings, each child kept by its occupancy logit
//!                                                                                   vae.rs
//!     -> FlexiCubes over the corners' signed distances and offsets, holes filled  extract.rs
//!     -> (another pass: the mesh's surface voxels at the next structure grid)
//!   -> pieces under 0.5 % dropped, decimated to the face target, centred, unit radius  post.rs
//! ```
//!
//! Models load when their stage needs them and go after it (NS_M3D_HI3D_RESIDENT=1 keeps them); `check` compares every
//! stage with reference/hi3d/ref.py's dumps.

pub mod check;
pub mod dit;
pub mod extract;
mod ffi;
pub mod post;
pub mod prep;
mod sconv;
pub mod sched;
pub mod sparse;
mod tables;
pub mod vae;
pub mod vision;
pub mod vox;
pub mod w;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
pub use nextsycl_diffusion::Picture;
use nextsycl_model3d::mesh::Mesh;
use nextsycl_model3d::{Abilities, At, Defaults, EngineOption, Error, LoadOptions, MeshRequest, Model3dEngine, Model3dKind, ModelFiles, Result, Step};
use nextsycl_remover::Remover;

use crate::w::File;

pub const ARCH: &str = "hi3d";

/// The files by role: the eight runtime files of the model repository (the 1024 transformer optional: a model
/// without it refines to 512 only), the two encoders, the shared background remover (optional)
pub const ROLES: &[&str] = &["index?", "scheduler", "vae_config?", "vae", "transformer_512_config", "transformer_512", "transformer_1024_config?",
                             "transformer_1024?", "dino", "clip", "remover?"];

const GIB: f64 = (1u64 << 30) as f64;

/// The refinement ladder: structure grid, voxel budget, extraction grid, face target
pub const LADDER: [(usize, usize, u32, usize); 3] = [(64, 80_000, 512, 1_000_000), (128, 90_000, 1024, 1_000_000), (192, 110_000, 1536, 2_000_000)];
/// Decoded voxels above which another pass is refused (the reference's DECODE_FEATURE_CEILING)
pub const DECODE_CEILING: usize = 12_000_000;
/// DINOv2's tokens
pub const COND_TOKENS: usize = 2917;
pub const COND_CH: usize = 2048;

/// The options it takes (`--opt-NAME`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "resident", env: "NS_M3D_HI3D_RESIDENT", value: "", help: "keep every model loaded between requests (default: each stage loads its model and drops it after)", at: At::Load },
    EngineOption { name: "profile", env: "NS_M3D_HI3D_PROFILE", value: "", help: "each stage's time and the card's memory, on stderr", at: At::Load },
    EngineOption { name: "euler", env: "", value: "", help: "first-order Euler updates instead of the scheduler's Heun (half the transformer calls; not the reference's default)", at: At::Request },
];

/// This engine's registry entry
pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "Hi3D v1.0 (Sparc3D refinement: DINOv2 + CLIP, 1.8B sparse flows, self-pruning decoder, FlexiCubes)", roles: ROLES, load,
                  options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Hi3d::load(files, gpu, o, log)?))
}

/// The kernels on one GPU: the shared diffusion ones and this engine's
pub struct Ctx {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    pub k: ffi::K,
}

impl Ctx {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ctx> {
        Ok(Ctx { gpu: gpu.clone(), nsd: Nsd::new(gpu)?, k: ffi::K::new(gpu)? })
    }
}

/// The scheduler's configuration (scheduler/scheduler_config.json)
#[derive(Clone, Copy, Debug)]
pub struct Sched {
    pub steps: u32,
    pub sigma_min: f32,
    pub time_rescale: f32,
    pub heun: bool,
}

impl Sched {
    pub fn read(path: &Path) -> Result<Sched> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?)
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let class = v["_class_name"].as_str().unwrap_or("FlowMatchHeunScheduler");
        if !["FlowMatchHeunScheduler", "FlowMatchEulerScheduler"].contains(&class) {
            return Err(Error(format!("{}: scheduler {class} (Heun or Euler expected)", path.display())));
        }
        Ok(Sched {
            steps: v["num_inference_steps"].as_u64().unwrap_or(25) as u32,
            sigma_min: v["sigma_min"].as_f64().unwrap_or(0.002) as f32,
            time_rescale: v["time_rescale_factor"].as_f64().unwrap_or(2.0) as f32,
            heun: class == "FlowMatchHeunScheduler",
        })
    }
}

/// A transformer's latent statistics (its config.json: latent_mean, latent_std)
fn stats(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?)
        .map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let vec = |k: &str| -> Vec<f32> { v[k].as_array().into_iter().flatten().filter_map(|x| x.as_f64()).map(|x| x as f32).collect() };
    let (m, s) = (vec("latent_mean"), vec("latent_std"));
    if m.len() != 32 || s.len() != 32 || v["model_channels"].as_u64() != Some(2112) {
        return Err(Error(format!("{}: not a released Sparc3D transformer config (2112 wide, 32 latent statistics)", path.display())));
    }
    Ok((m, s))
}

/// NS_M3D_HI3D_PROFILE on any engine of the process (the loads' times)
static PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A model loaded when its stage needs it, kept when resident
struct Slot<T>(Mutex<Option<Arc<T>>>);

impl<T> Slot<T> {
    fn new() -> Slot<T> {
        Slot(Mutex::new(None))
    }
    fn get(&self, resident: bool, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        let mut g = self.0.lock().unwrap();
        if let Some(t) = g.as_ref() {
            return Ok(t.clone());
        }
        let t0 = Instant::now();
        let t = Arc::new(load()?);
        if PROFILE.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("  loaded a {} in {:.2} s", std::any::type_name::<T>().rsplit("::").next().unwrap_or("model"), t0.elapsed().as_secs_f64());
        }
        if resident {
            *g = Some(t.clone());
        }
        Ok(t)
    }
}

struct Models {
    dino: Slot<vision::Vit>,
    clip: Slot<vision::Vit>,
    t512: Slot<dit::Dit>,
    t1024: Slot<dit::Dit>,
    vae: Slot<vae::Vae>,
    remover: Slot<Remover>,
}

pub struct Hi3d {
    pub files: ModelFiles,
    pub cx: Ctx,
    pub sched: Sched,
    models: Models,
    resident: bool,
    profile: bool,
    load_s: f64,
    run: Mutex<()>,
}

/// A role's file, or why not
fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// What a refinement leaves besides the mesh (the measurements read it)
#[derive(Default, Clone)]
pub struct Trace {
    /// each pass: (structure grid it settled at, voxels, parents, decoded voxels, faces extracted)
    pub passes: Vec<(usize, usize, usize, usize, usize)>,
    /// (stage, seconds, the card's GiB in use when it ended)
    pub stages: Vec<(String, f64, f64)>,
    /// the card's most memory in use during the run (every process's), GiB
    pub peak_gib: f64,
}

/// Watches the card's free memory every 50 ms until dropped: the run's peak
struct Peak {
    stop: Arc<std::sync::atomic::AtomicBool>,
    least: Arc<std::sync::atomic::AtomicU64>,
    total: u64,
    t: Option<std::thread::JoinHandle<()>>,
}

impl Peak {
    fn new(gpu: &Arc<Gpu>) -> Peak {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let least = Arc::new(AtomicU64::new(u64::MAX));
        let total = gpu.memory().map(|m| m.0).unwrap_or(0);
        let (s, l, g) = (stop.clone(), least.clone(), gpu.clone());
        let t = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Ok((_, Some(f))) = g.memory() {
                    l.fetch_min(f, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Peak { stop, least, total, t: Some(t) }
    }
    fn gib(&self) -> f64 {
        let l = self.least.load(std::sync::atomic::Ordering::Relaxed);
        if l == u64::MAX { 0.0 } else { self.total.saturating_sub(l) as f64 / GIB }
    }
}

impl Drop for Peak {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.t.take() {
            let _ = t.join();
        }
    }
}

impl Hi3d {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Hi3d> {
        let t0 = Instant::now();
        let missing = nextsycl_model3d::missing(files, ROLES);
        if !missing.is_empty() {
            return Err(Error(format!("{ARCH}: no {} file", missing.join(", "))));
        }
        if files.contains_key("transformer_1024") != files.contains_key("transformer_1024_config") {
            return Err(Error(format!("{ARCH}: transformer_1024 and its config come together")));
        }
        if let Some(p) = files.get("vae_config") {
            let v = std::fs::read_to_string(p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
            if !v.contains("PTv3MeshDecoder") {
                return Err(Error(format!("{}: not a SparconvVAE / PTv3MeshDecoder config", p.display())));
            }
        }
        let sched = Sched::read(role(files, "scheduler")?)?;
        stats(role(files, "transformer_512_config")?)?;
        let cx = Ctx::new(gpu)?;
        let flag = |n: &str| o.setting(n).is_some_and(|v| v != "0" && !v.is_empty());
        let e = Hi3d {
            files: files.clone(),
            cx,
            sched,
            models: Models { dino: Slot::new(), clip: Slot::new(), t512: Slot::new(), t1024: Slot::new(), vae: Slot::new(), remover: Slot::new() },
            resident: flag("NS_M3D_HI3D_RESIDENT"),
            profile: flag("NS_M3D_HI3D_PROFILE"),
            load_s: 0.0,
            run: Mutex::new(()),
        };
        let f = File::open(role(files, "transformer_512")?, "")?;
        if !f.has("blocks.0.modulation") || !f.has("input_blocks.1.conv1.conv.weight") {
            return Err(Error(format!("{}: not a Sparc3D transformer", f.path())));
        }
        if e.profile {
            PROFILE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let silo = match e.cx.k.silo() {
            Some(s) => format!("silo {}", s.path.display()),
            None => "no silo: its own kernels".to_string(),
        };
        log(format!("{ARCH}: ready on {} (up to {}; models load stage by stage{}; {silo})", gpu.name, e.max_resolution(),
                    if e.resident { ", then stay" } else { "" }));
        Ok(Hi3d { load_s: t0.elapsed().as_secs_f64(), ..e })
    }

    /// The finest extraction grid this model's files reach (1536 is the 1024 transformer's third pass)
    pub fn max_resolution(&self) -> u32 {
        if self.files.contains_key("transformer_1024") { 1536 } else { 512 }
    }

    fn default_resolution(&self) -> u32 {
        if self.files.contains_key("transformer_1024") { 1024 } else { 512 }
    }

    /// Free memory on the card, when the driver says, against what a stage needs (plus 1.5 GiB): the xe driver has
    /// no out-of-memory error, so a stage that would over-commit stops here instead
    fn room(&self, what: &str, bytes: f64) -> Result<()> {
        if let Ok((_, Some(free))) = self.cx.gpu.memory() {
            let need = bytes + 1.5 * GIB;
            if (free as f64) < need {
                return Err(Error(format!("{what}: {:.1} GiB free on {}, {:.1} GiB needed (with 1.5 GiB to spare)", free as f64 / GIB, self.cx.gpu.name,
                                         need / GIB)));
            }
        }
        Ok(())
    }

    pub fn dino(&self) -> Result<Arc<vision::Vit>> {
        self.models.dino.get(self.resident, || {
            self.room("DINOv2", 1.0 * GIB)?;
            vision::Vit::dino(&File::open(role(&self.files, "dino")?, "")?, &self.cx.gpu, prep::DINO_SIZE)
        })
    }

    pub fn clip(&self) -> Result<Arc<vision::Vit>> {
        self.models.clip.get(self.resident, || {
            self.room("CLIP", 1.0 * GIB)?;
            vision::Vit::clip(&File::open(role(&self.files, "clip")?, "vision_model.")?, &self.cx.gpu)
        })
    }

    /// The transformer of a structure grid: 512's at 64, 1024's above
    pub fn transformer(&self, structure: usize) -> Result<Arc<dit::Dit>> {
        let (slot, r) = if structure == 64 { (&self.models.t512, "transformer_512") } else { (&self.models.t1024, "transformer_1024") };
        if !self.files.contains_key(r) {
            return Err(Error(format!("{ARCH}: this model has no {r} (it refines to 512 only)")));
        }
        slot.get(self.resident, || {
            self.room(r, 3.6 * GIB)?;
            let (mean, std) = stats(role(&self.files, &format!("{r}_config"))?)?;
            dit::Dit::load(&File::open(role(&self.files, r)?, "")?, &self.cx.gpu, mean, std)
        })
    }

    pub fn vae(&self) -> Result<Arc<vae::Vae>> {
        self.models.vae.get(self.resident, || {
            self.room("the decoder", 0.4 * GIB)?;
            vae::Vae::load(&File::open(role(&self.files, "vae")?, "")?, &self.cx.gpu)
        })
    }

    fn remover(&self) -> Result<Option<Arc<Remover>>> {
        let Some(p) = self.files.get("remover") else { return Ok(None) };
        Ok(Some(self.models.remover.get(self.resident, || {
            self.room("the background remover", 1.5 * GIB)?;
            Remover::load(p, &self.cx.gpu)
        })?))
    }

    /// The picture as the conditioner frames it (an opaque one cut out first when `cut` and a remover is there)
    pub fn frame(&self, pic: &Picture, cut: bool) -> Result<Picture> {
        let p = if cut && !nextsycl_remover::has_alpha(pic) {
            match self.remover()? {
                Some(r) => r.cut_out(pic)?,
                None => pic.clone(),
            }
        } else {
            pic.clone()
        };
        Ok(prep::frame(&p))
    }

    /// The conditioning tokens [2917, 2048] (float32) of the framed picture: (DINOv2's raw last hidden state, CLIP's
    /// raw pooled output, the condition)
    pub fn condition(&self, framed: &Picture) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (din, cin) = prep::inputs(framed);
        let d = self.dino()?.forward(&self.cx, &din)?;
        let c = self.clip()?.forward(&self.cx, &cin)?;
        if d.len() != COND_TOKENS * 1024 || c.len() != 1024 {
            return Err(Error(format!("the encoders gave {} and {} values ({COND_TOKENS} x 1024 and 1024 expected)", d.len(), c.len())));
        }
        let mut dn = d.clone();
        vision::l2_rows(&mut dn, 1024);
        let mut cn = c.clone();
        vision::l2_rows(&mut cn, 1024);
        let mut cond = Vec::with_capacity(COND_TOKENS * COND_CH);
        for row in dn.chunks(1024) {
            cond.extend_from_slice(row);
            cond.extend_from_slice(&cn);
        }
        Ok((d, c, cond))
    }

    /// One pass's sampling: the latents (denormalized) on `coords` from `noise`, `steps` updates
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, m: &dit::Dit, coords: &[[i32; 3]], cond: &[f32], noise: Vec<f32>, steps: usize, heun: bool, updates: Option<&mut Vec<Vec<f32>>>,
                  mut step: impl FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        let n = coords.len();
        let g = &self.cx.gpu;
        let geo = m.geo(g, coords)?;
        self.room("the transformer's activations", (geo.p * (dit::C * 4 * 4 + dit::C * 2 * 8 + 8448 * 2) + n * 512 * 4 * 6) as f64
            + 24.0 * (COND_TOKENS * 2 * dit::C * 2) as f64)?;
        let ch = DevBuf::new(g, cond.len() * 2)?;
        ch.write(0, &w::half_bytes(cond))?;
        let kvs = m.cache_cross(&self.cx, &ch, COND_TOKENS)?;
        let neg = m.const_cross(&self.cx)?;
        let (sigmas, hats) = sched::schedule(steps, n, self.sched.sigma_min, self.sched.time_rescale);
        let guided = |x: &[f32], t: f32| -> Result<Vec<f32>> {
            let mut v = m.forward(&self.cx, &geo, x, 1000.0 * t, dit::Cross::Cached(&kvs, COND_TOKENS), None)?;
            if let Some(s) = sched::guidance(t) {
                let u = m.forward(&self.cx, &geo, x, 1000.0 * t, dit::Cross::Const(&neg), None)?;
                sched::guide(&mut v, &u, s);
            }
            Ok(v)
        };
        let mut x = noise;
        let mut ups = updates;
        for i in 0..steps {
            let v = guided(&x, hats[i])?;
            let dt = hats[i] - sigmas[i + 1];
            if heun {
                let pred: Vec<f32> = x.iter().zip(&v).map(|(a, b)| a - dt * b).collect();
                let v2 = guided(&pred, sigmas[i + 1])?;
                for ((a, b), c) in x.iter_mut().zip(&v).zip(&v2) {
                    *a -= dt * (0.5 * (b + c));
                }
            } else {
                for (a, b) in x.iter_mut().zip(&v) {
                    *a -= dt * b;
                }
            }
            if let Some(u) = ups.as_deref_mut() {
                u.push(x.clone());
            }
            step(i as u32 + 1, steps as u32)?;
        }
        for r in x.chunks_mut(32) {
            for (c, v) in r.iter_mut().enumerate() {
                *v = *v * m.std[c] + m.mean[c];
            }
        }
        Ok(x)
    }

    /// The whole refinement (the mesh's positions and triangles, the framed picture) to `resolution`, with its trace
    #[allow(clippy::too_many_arguments)]
    pub fn refine(&self, framed: &Picture, pos: &[[f32; 3]], tris: &[[u32; 3]], resolution: u32, steps: usize, heun: bool, seed: u64, faces: Option<usize>,
                  progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Mesh, Trace)> {
        let _run = self.run.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = Instant::now();
        let peak = Peak::new(&self.cx.gpu);
        let target = LADDER.iter().position(|l| l.2 == resolution)
            .ok_or_else(|| Error(format!("resolution {resolution}: 512, 1024 or 1536")))?;
        if resolution > self.max_resolution() {
            return Err(Error(format!("resolution {resolution}: this model reaches {} (it has no transformer_1024)", self.max_resolution())));
        }
        let mut tr = Trace::default();
        let mut mark = Instant::now();
        let used = || self.cx.gpu.memory().ok().and_then(|(t, f)| f.map(|f| t.saturating_sub(f) as f64 / GIB)).unwrap_or(0.0);
        let mut stage = |tr: &mut Trace, name: String| {
            let s = mark.elapsed().as_secs_f64();
            if self.profile {
                eprintln!("  {name}: {s:.2} s, {:.1} GiB on the card", used());
            }
            tr.stages.push((name, s, used()));
            mark = Instant::now();
        };
        let mut report = |phase: &'static str, at: u32, of: u32| progress(Step { phase, at, of, seconds: t0.elapsed().as_secs_f64() });
        report("image", 0, 1)?;
        let (_, _, cond) = self.condition(framed)?;
        if !self.resident {
            self.cx.gpu.sync()?;
        }
        stage(&mut tr, "condition".into());
        let mut rng = nextsycl_diffusion::torch_rng::Mt19937::new(seed);
        let mut cur = extract::Tri { v: pos.to_vec(), f: tris.to_vec() };
        for (li, &(structure, budget, _, _)) in LADDER.iter().enumerate().take(target + 1) {
            report("voxels", li as u32, target as u32 + 1)?;
            let tri: Vec<[u32; 3]> = cur.f.clone();
            let (coords, grid) = vox::adaptive(&cur.v, &tri, structure, budget).map_err(Error)?;
            if grid != structure && self.profile {
                eprintln!("  pass {}: the voxels settled at {grid}^3 (not {structure}^3) for the budget of {budget}", li + 1);
            }
            stage(&mut tr, format!("pass {} voxels", li + 1));
            let m = self.transformer(structure)?;
            let noise = rng.randn(coords.len() * 32);
            let parents = sparse::downsample(&coords).parents.len();
            let lat = self.sample(&m, &coords, &cond, noise, steps, heun, None, |at, of| report(if li == 0 { "pass 1" } else if li == 1 { "pass 2" } else { "pass 3" }, at, of))?;
            drop(m);
            stage(&mut tr, format!("pass {} sampling ({} voxels, {parents} tokens)", li + 1, coords.len()));
            report("decode", li as u32, target as u32 + 1)?;
            let dec = self.vae()?.decode(&self.cx, &coords, &lat, structure, None)?;
            stage(&mut tr, format!("pass {} decode ({} voxels at {}^3)", li + 1, dec.coords.len(), dec.res));
            if li < target && dec.coords.len() > DECODE_CEILING {
                return Err(Error(format!("decoded feature ceiling exceeded before another pass: {} > {DECODE_CEILING}", dec.coords.len())));
            }
            report("mesh", li as u32, target as u32 + 1)?;
            let raw = extract::to_mesh(&dec.coords, dec.res, &dec.sign, &dec.feat, extract::matrix(&coords));
            let filled = extract::fill_holes(&raw);
            if filled.f.is_empty() {
                return Err(Error("mesh extraction produced an empty isosurface".into()));
            }
            tr.passes.push((grid, coords.len(), parents, dec.coords.len(), filled.f.len()));
            stage(&mut tr, format!("pass {} mesh ({} faces)", li + 1, filled.f.len()));
            cur = filled;
        }
        report("finish", 0, 1)?;
        let face_target = faces.unwrap_or(LADDER[target].3);
        let mut out = post::postprocess(&cur, face_target);
        out.compute_normals();
        stage(&mut tr, format!("postprocess ({} faces)", out.triangles()));
        tr.peak_gib = peak.gib();
        Ok((out, tr))
    }
}

impl Model3dEngine for Hi3d {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> Abilities {
        Abilities { refine: true, views: 1, ..Default::default() }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: self.sched.steps, cfg: 0.0, resolution: self.default_resolution(), faces: LADDER[1].3 as u32, texture_size: 0, frames: 0, fps: 0.0 }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let Some(mesh) = req.mesh.as_ref() else {
            return Err(Error(format!("{ARCH} refines a coarse mesh: give one (mesh / file) with the picture of the same object")));
        };
        if req.texture {
            return Err(Error(format!("{ARCH} makes geometry only (texture: false)")));
        }
        let view = req.views.first().ok_or_else(|| Error("a picture of the object: required".into()))?;
        let tris: Vec<[u32; 3]> = mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        if mesh.positions.is_empty() || tris.is_empty() {
            return Err(Error("the coarse mesh has no triangles".into()));
        }
        if mesh.positions.iter().any(|p| p.iter().any(|v| !v.is_finite())) || tris.iter().flatten().any(|i| *i as usize >= mesh.positions.len()) {
            return Err(Error("the coarse mesh has a vertex that is not finite or a face out of range".into()));
        }
        let (lo, hi) = mesh.bounds();
        if (0..3).all(|a| hi[a] - lo[a] == 0.0) {
            return Err(Error("the coarse mesh has no extent".into()));
        }
        let t = &view.image;
        let pic = Picture { width: t.width, height: t.height, channels: 4, data: t.rgba.clone() };
        let framed = self.frame(&pic, req.remove_background.unwrap_or(true))?;
        let resolution = req.resolution.unwrap_or(self.default_resolution());
        let steps = req.steps.unwrap_or(self.sched.steps).clamp(1, 200) as usize;
        let heun = self.sched.heun && !req.extra.contains_key("euler");
        let (m, tr) = self.refine(&framed, &mesh.positions, &tris, resolution, steps, heun, req.seed, req.faces.map(|f| f as usize), progress)?;
        if self.profile {
            for (g, n, p, d, f) in &tr.passes {
                eprintln!("  pass at {g}^3: {n} voxels, {p} tokens, {d} decoded, {f} faces");
            }
            eprintln!("  the card's peak: {:.1} GiB", tr.peak_gib);
        }
        Ok(m)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}: up to {} ({})", self.cx.gpu.name, self.max_resolution(), if self.resident { "models resident" } else { "models load stage by stage" })]
    }
}
