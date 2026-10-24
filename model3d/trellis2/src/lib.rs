//! TRELLIS.2 (microsoft/TRELLIS.2: one picture to a textured mesh) and Pixal3D (its pixel-aligned variant, arch
//! `pixal3d`), from Aero-Ex's GGUF flows and the fp16 decoders. The pipeline, as Trellis2ImageTo3DPipeline runs it:
//!
//! ```text
//!   picture -> the object cut out, cropped, squared, premultiplied (nextsycl-remover: BiRefNet when it is opaque)
//!           -> DINOv3 tokens at 512 (and 1024)                                         dino.rs
//!   noise   -> the sparse-structure flow (dense 16^3, 12 Euler steps, guided)       -> the structure decoder: 64^3
//!              occupancy, max-pooled to 32^3: the voxels                             dit.rs, sampler.rs, ssdec.rs
//!           -> the shape flow on the voxels (512)  [cascade: the shape decoder's subdivisions x4, re-quantized to
//!              res/16 within the token budget, the 1024 shape flow there]            vae.rs (upsample)
//!           -> the texture flow (the shape latent as its concatenated condition)
//!           -> the shape decoder (sparse UNet + subdivisions) -> the dual-grid mesh  vae.rs, mesh.rs
//!              the texture decoder (the shape's subdivisions replayed) -> PBR voxels
//!           -> oriented, decimated to the triangle budget, unwrapped, baked          bake.rs
//! ```
//!
//! The models load stage by stage (each dropped after its stage unless NS_M3D_T2_RESIDENT=1); `check` compares every stage
//! with reference/trellis2/ref.py's dumps.

pub mod bake;
pub mod check;
pub mod dino;
pub mod dit;
mod ffi;
pub mod mesh;
pub mod pixal;
pub mod sampler;
mod sconv;
pub mod sparse;
pub mod ssdec;
pub mod vae;
pub mod w;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{Dt, Nsd};
pub use nextsycl_diffusion::Picture;
use nextsycl_model3d::mesh::{tools, Mesh};
use nextsycl_model3d::{Abilities, At, Defaults, EngineOption, Error, LoadOptions, MeshRequest, Model3dEngine, Model3dKind, ModelFiles, Result, Step};
use nextsycl_remover::Remover;

use crate::sampler::Params;
use crate::w::File;

pub const ARCH: &str = "trellis2";
pub const ARCH_PIXAL: &str = "pixal3d";

/// The files by role (`?`: optional - the 512 flows serve the 512 path and the cascade's first pass, the 1024 ones the
/// cascade's second; a model has the ones its paths need)
pub const ROLES: &[&str] = &["pipeline", "dino", "ss_flow", "ss_dec", "shape_flow_512", "shape_flow_1024?", "shape_dec", "tex_flow_512?", "tex_flow_1024?",
                             "tex_dec", "remover?"];

const GIB: f64 = (1u64 << 30) as f64;

/// The options it takes (`--opt-NAME`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "resident", env: "NS_M3D_T2_RESIDENT", value: "", help: "keep every model loaded between requests (default: each stage loads its model and drops it after)", at: At::Load },
    EngineOption { name: "profile", env: "NS_M3D_T2_PROFILE", value: "", help: "each stage's time and the card's memory, on stderr", at: At::Load },
    EngineOption { name: "max-tokens", env: "", value: "N", help: "the cascade's token budget for the second shape pass (49152): the resolution steps down by 128 until it fits", at: At::Request },
    EngineOption { name: "remesh", env: "", value: "", help: "remesh the surface isotropically before decimating it (slower)", at: At::Request },
    EngineOption { name: "fov", env: "", value: "DEGREES", help: "pixal3d: the picture's horizontal field of view (default 40)", at: At::Request },
    EngineOption { name: "bake-push", env: "", value: "VOXELS", help: "the bake looks for each texel's surface point from this far outside along its normal (default 2)", at: At::Both },
];

/// This engine's registry entries
pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "TRELLIS.2 (DINOv3, 1.3B flows, sparse decoders, dual-grid mesh, PBR bake)", roles: ROLES, load, options: OPTIONS }
}

pub fn kind_pixal() -> Model3dKind {
    Model3dKind { archs: &[ARCH_PIXAL], name: "Pixal3D (TRELLIS.2 with pixel-aligned projected DINOv3 features; manual FOV)", roles: ROLES, load: load_pixal,
                  options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Trellis2::load(ARCH, files, gpu, o, log)?))
}

fn load_pixal(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Trellis2::load(ARCH_PIXAL, files, gpu, o, log)?))
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

/// A seeded standard-normal draw (splitmix64, Box-Muller): the noise of each stage
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (next().max(1e-300), next());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * v).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// pipeline.json: the samplers and the latents' normalisations
#[derive(Clone)]
pub struct Config {
    pub ss: Params,
    pub shape: Params,
    pub tex: Params,
    pub shape_mean: Vec<f32>,
    pub shape_std: Vec<f32>,
    pub tex_mean: Vec<f32>,
    pub tex_std: Vec<f32>,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?)
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let a = &v["args"];
        let vec = |x: &serde_json::Value| -> Result<Vec<f32>> {
            let v: Vec<f32> = x.as_array().into_iter().flatten().filter_map(|n| n.as_f64()).map(|n| n as f32).collect();
            if v.len() != 32 {
                return Err(Error(format!("{}: a normalisation of {} values, not 32", path.display(), v.len())));
            }
            Ok(v)
        };
        Ok(Config {
            ss: Params::from_json(&a["sparse_structure_sampler"]["params"]),
            shape: Params::from_json(&a["shape_slat_sampler"]["params"]),
            tex: Params::from_json(&a["tex_slat_sampler"]["params"]),
            shape_mean: vec(&a["shape_slat_normalization"]["mean"])?,
            shape_std: vec(&a["shape_slat_normalization"]["std"])?,
            tex_mean: vec(&a["tex_slat_normalization"]["mean"])?,
            tex_std: vec(&a["tex_slat_normalization"]["std"])?,
        })
    }
}

/// NS_M3D_T2_PROFILE on any engine of the process (the loads' times)
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

/// The models, by stage
struct Models {
    dino: Slot<dino::Dino>,
    ss_flow: Slot<dit::Dit>,
    ss_dec: Slot<ssdec::SsDecoder>,
    shape512: Slot<dit::Dit>,
    shape1024: Slot<dit::Dit>,
    shape_dec: Slot<vae::Decoder>,
    tex512: Slot<dit::Dit>,
    tex1024: Slot<dit::Dit>,
    tex_dec: Slot<vae::Decoder>,
    remover: Slot<Remover>,
}

pub struct Trellis2 {
    pub arch: &'static str,
    pub files: ModelFiles,
    pub cx: Ctx,
    pub cfg: Config,
    models: Models,
    resident: bool,
    profile: bool,
    loaded: Instant,
    load_s: f64,
    run: Mutex<()>,
}

/// A role's file, or why not
fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// What a generation leaves besides the mesh (the check and the measurements read it)
#[derive(Default)]
pub struct Trace {
    pub res: usize,
    pub tokens: Vec<usize>,
    pub voxels: usize,
    pub stages: Vec<(&'static str, f64, f64)>,
    /// the card's most memory in use during the run (every process's), GiB, when the driver says
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

impl Trellis2 {
    pub fn load(arch: &'static str, files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Trellis2> {
        let t0 = Instant::now();
        let missing = nextsycl_model3d::missing(files, ROLES);
        if !missing.is_empty() {
            return Err(Error(format!("{arch}: no {} file", missing.join(", "))));
        }
        if !files.contains_key("tex_flow_512") && !files.contains_key("tex_flow_1024") {
            return Err(Error(format!("{arch}: no texture flow (tex_flow_512 or tex_flow_1024)")));
        }
        let cfg = Config::read(role(files, "pipeline")?)?;
        let cx = Ctx::new(gpu)?;
        let flag = |n: &str| o.setting(n).is_some_and(|v| v != "0" && !v.is_empty());
        let e = Trellis2 {
            arch,
            files: files.clone(),
            cx,
            cfg,
            models: Models {
                dino: Slot::new(),
                ss_flow: Slot::new(),
                ss_dec: Slot::new(),
                shape512: Slot::new(),
                shape1024: Slot::new(),
                shape_dec: Slot::new(),
                tex512: Slot::new(),
                tex1024: Slot::new(),
                tex_dec: Slot::new(),
                remover: Slot::new(),
            },
            resident: flag("NS_M3D_T2_RESIDENT"),
            profile: flag("NS_M3D_T2_PROFILE"),
            loaded: Instant::now(),
            load_s: 0.0,
            run: Mutex::new(()),
        };
        // the files are opened (and checked) when their stage runs; here only what every path needs is looked at
        let f = File::open(role(files, "ss_flow")?)?;
        if !f.has("blocks.0.modulation") {
            return Err(Error(format!("{}: not a TRELLIS flow model", f.path())));
        }
        if e.profile {
            PROFILE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let silo = match e.cx.k.silo() {
            Some(s) => format!("silo {}", s.path.display()),
            None => "no silo: the shared kernels".to_string(),
        };
        log(format!("{arch}: ready on {} (models load stage by stage{}; {silo})", gpu.name, if e.resident { ", then stay" } else { "" }));
        Ok(Trellis2 { load_s: t0.elapsed().as_secs_f64(), ..e })
    }

    fn file(&self, r: &str) -> Result<File> {
        File::open(role(&self.files, r)?)
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

    fn dit(&self, slot: &Slot<dit::Dit>, r: &str) -> Result<Arc<dit::Dit>> {
        slot.get(self.resident, || {
            self.room(r, 1.6 * GIB)?;
            dit::Dit::load(&self.file(r)?, &self.cx.gpu)
        })
    }

    pub fn dino(&self) -> Result<Arc<dino::Dino>> {
        self.models.dino.get(self.resident, || {
            self.room("DINOv3", 1.4 * GIB)?;
            dino::Dino::load(&self.file("dino")?, &self.cx)
        })
    }
    pub fn ss_flow(&self) -> Result<Arc<dit::Dit>> {
        self.dit(&self.models.ss_flow, "ss_flow")
    }
    pub fn ss_dec(&self) -> Result<Arc<ssdec::SsDecoder>> {
        self.models.ss_dec.get(self.resident, || ssdec::SsDecoder::load(&self.file("ss_dec")?, &self.cx.gpu))
    }
    pub fn shape_flow(&self, hr: bool) -> Result<Arc<dit::Dit>> {
        if hr { self.dit(&self.models.shape1024, "shape_flow_1024") } else { self.dit(&self.models.shape512, "shape_flow_512") }
    }
    pub fn tex_flow(&self, hr: bool) -> Result<Arc<dit::Dit>> {
        let hr = if hr { self.files.contains_key("tex_flow_1024") } else { !self.files.contains_key("tex_flow_512") };
        if hr { self.dit(&self.models.tex1024, "tex_flow_1024") } else { self.dit(&self.models.tex512, "tex_flow_512") }
    }
    pub fn shape_dec(&self) -> Result<Arc<vae::Decoder>> {
        self.models.shape_dec.get(self.resident, || {
            self.room("the shape decoder", 1.0 * GIB)?;
            vae::Decoder::load(&self.file("shape_dec")?, &self.cx.gpu)
        })
    }
    pub fn tex_dec(&self) -> Result<Arc<vae::Decoder>> {
        self.models.tex_dec.get(self.resident, || {
            self.room("the texture decoder", 1.0 * GIB)?;
            vae::Decoder::load(&self.file("tex_dec")?, &self.cx.gpu)
        })
    }
    fn remover(&self) -> Result<Option<Arc<Remover>>> {
        let Some(p) = self.files.get("remover") else { return Ok(None) };
        Ok(Some(self.models.remover.get(self.resident, || {
            self.room("the background remover", 1.5 * GIB)?;
            Remover::load(p, &self.cx.gpu)
        })?))
    }

    /// The object picture from a request's view (cut out when opaque and asked to)
    pub fn prepare(&self, pic: &Picture, cut: bool) -> Result<Picture> {
        let r = if cut && !nextsycl_remover::has_alpha(pic) { self.remover()? } else { None };
        // Pixal3D squares the object with a tenth to spare
        nextsycl_remover::prepare_margin(pic, r.as_deref(), if self.arch == ARCH_PIXAL { 1.1 } else { 1.0 })
    }

    /// The picture's DINOv3 tokens at `size` as the flows read them (half [l, 1024]) and their count
    pub fn cond(&self, pre: &Picture, size: usize) -> Result<(DevBuf, usize)> {
        let d = self.dino()?;
        let f = d.encode(&self.cx, &dino::input(pre, size), size)?;
        let l = f.floats() / dino::DIM;
        let h = DevBuf::new(&self.cx.gpu, f.floats() * 2)?;
        self.cx.k.convert(f.ptr(), Dt::F32, f.floats(), h.ptr(), Dt::F16)?;
        Ok((h, l))
    }

    /// A flow's sampling over tokens at `coords` (rope), starting from `x0` [n, out]; `concat` [n, c] appended to
    /// every call's input (the texture flow's shape latent)
    #[allow(clippy::too_many_arguments)]
    pub fn flow(&self, m: &dit::Dit, p: &Params, coords: &[[i32; 3]], x0: Vec<f32>, cond: &(DevBuf, usize), concat: Option<&[f32]>, proj: Option<&DevBuf>,
                mut step: impl FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        let n = coords.len();
        let g = &self.cx.gpu;
        self.room("a flow's activations", (n * (dit::C * 4 * 3 + dit::C * 2 * 6 + 8192 * 2) + 30 * cond.1 * 2 * dit::C * 2) as f64)?;
        let w = m.work(g, n, cond.1)?;
        let cs = DevBuf::from_f32(g, &dit::rope(coords))?;
        // the condition's keys and values made once for every call; the zero condition's attention is its value bias
        let kvs = m.cache_cross(&self.cx, &cond.0, cond.1)?;
        let neg = m.const_cross(&self.cx)?;
        let negp = proj.map(|p| -> Result<DevBuf> {
            let z = DevBuf::new(g, p.len)?;
            z.fill(0)?;
            Ok(z)
        }).transpose()?;
        let out = m.out_ch;
        if self.profile {
            *m.prof.lock().unwrap() = Some([0.0; 4]);
        }
        let t_flow = Instant::now();
        let r = sampler::sample(x0, p, |x, t, c| {
            let input = match concat {
                Some(cc) => {
                    let k = cc.len() / n;
                    let mut v = Vec::with_capacity(n * (out + k));
                    for i in 0..n {
                        v.extend_from_slice(&x[i * out..(i + 1) * out]);
                        v.extend_from_slice(&cc[i * k..(i + 1) * k]);
                    }
                    v
                }
                None => x.to_vec(),
            };
            let (cross, pj) = if c { (dit::Cross::Cached(&kvs, cond.1), proj) } else { (dit::Cross::Const(&neg), negp.as_ref()) };
            m.forward_with(&self.cx, &w, &input, t, cross, &cs, pj, None)
        }, &mut step);
        if let Some(t) = m.prof.lock().unwrap().take() {
            eprintln!("{}: flow of {n} tokens, {} calls in {:.2} s: self-attention linears {:.2} s, its attention {:.2} s, cross-attention {:.2} s, MLP {:.2} s",
                      self.arch, p.calls(), t_flow.elapsed().as_secs_f64(), t[0], t[1], t[2], t[3]);
        }
        r
    }

    fn mark(&self, tr: &mut Trace, what: &'static str, t0: &mut Instant) {
        let s = t0.elapsed().as_secs_f64();
        let used = self.cx.gpu.memory().ok().and_then(|(t, f)| f.map(|f| (t - f) as f64 / GIB)).unwrap_or(0.0);
        if self.profile {
            eprintln!("{}: {what:<10} {s:>7.2} s   card in use {used:.1} GiB", self.arch);
        }
        tr.stages.push((what, s, used));
        *t0 = Instant::now();
    }

    /// The whole pipeline for a request
    pub fn run(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Mesh, Trace)> {
        let _one = self.run.lock().unwrap();
        let peak = self.profile.then(|| Peak::new(&self.cx.gpu));
        let start = Instant::now();
        let mut tr = Trace::default();
        let mut t0 = Instant::now();
        let mut report = |phase: &'static str, at: u32, of: u32| progress(Step { phase, at, of, seconds: start.elapsed().as_secs_f64() });
        let view = req.views.first().ok_or_else(|| Error("no picture".into()))?;
        let res_req = req.resolution.unwrap_or(1024);
        let cascade = res_req > 512;
        if cascade && (!self.files.contains_key("shape_flow_1024") || !self.files.contains_key("tex_flow_1024")) {
            return Err(Error(format!("{}: resolution {res_req} needs the 1024 flows (shape_flow_1024, tex_flow_1024); this model has the 512 path", self.arch)));
        }
        let steps = req.steps;
        let with = |p: &Params, guided: bool| Params { steps: steps.unwrap_or(p.steps), guidance: if guided { req.cfg.map_or(p.guidance, |c| c as f64) } else { p.guidance }, ..*p };
        let (ssp, shp, txp) = (with(&self.cfg.ss, true), with(&self.cfg.shape, true), with(&self.cfg.tex, false));
        report("image", 0, 1)?;
        let tex = &view.image;
        let pic = Picture { width: tex.width, height: tex.height, channels: 4, data: tex.rgba.clone() };
        let pre = self.prepare(&pic, req.remove_background.unwrap_or(true))?;
        self.mark(&mut tr, "prepare", &mut t0);
        let pix = if self.arch == ARCH_PIXAL { Some(pixal::Camera::from_request(req)?) } else { None };
        let cond512 = self.cond(&pre, 512)?;
        let cond1024 = if cascade { Some(self.cond(&pre, 1024)?) } else { None };
        let pcond = match &pix {
            Some(cam) => Some(pixal::Conds::new(self, &pre, cam, cascade)?),
            None => None,
        };
        if !self.resident {
            drop(self.models.dino.0.lock().unwrap().take());
        }
        report("image", 1, 1)?;
        self.mark(&mut tr, "dino", &mut t0);
        let mut seed = req.seed;
        let mut draw = |n: usize| {
            seed = seed.wrapping_add(0x5851_f42d_4c95_7f2d);
            noise(seed, n)
        };
        // the sparse structure
        let grid = sconv::grid(16);
        let ss = self.ss_flow()?;
        let (c_ss, p_ss) = match &pcond {
            Some(pc) => pc.structure(self, &grid)?,
            None => (None, None),
        };
        let z = self.flow(&ss, &ssp, &grid, draw(grid.len() * 8), c_ss.as_ref().unwrap_or(&cond512), None, p_ss.as_ref(), |i, of| report("structure", i, of))?;
        drop(ss);
        let zc: Vec<f32> = (0..8 * 4096).map(|i| z[(i % 4096) * 8 + i / 4096]).collect();
        let (d, logits) = self.ss_dec()?.decode(&self.cx, &zc)?;
        let mut coords = ssdec::occupied(d, &logits, 32);
        if coords.is_empty() {
            return Err(Error("the sparse structure came out empty (nothing to mesh)".into()));
        }
        tr.tokens.push(coords.len());
        self.mark(&mut tr, "structure", &mut t0);
        // the shape
        let norm = |v: &mut [f32], mean: &[f32], std: &[f32]| v.chunks_exact_mut(32).for_each(|r| r.iter_mut().enumerate().for_each(|(c, x)| *x = *x * std[c] + mean[c]));
        let (c_sh, p_sh) = match &pcond {
            Some(pc) => pc.shape(self, &coords, 512)?,
            None => (None, None),
        };
        let m = self.shape_flow(false)?;
        let mut slat = self.flow(&m, &shp, &coords, draw(coords.len() * 32), c_sh.as_ref().unwrap_or(&cond512), None, p_sh.as_ref(), |i, of| report("shape", i, of))?;
        drop(m);
        norm(&mut slat, &self.cfg.shape_mean, &self.cfg.shape_std);
        let mut res = 512;
        if cascade {
            let up = self.shape_dec()?.run(&self.cx, &coords, &slat, &vae::Plan { guide: None, upsample: Some(4), keep_ups: false, profile: self.profile }, |_, _| Ok(()))?;
            let budget = req.extra.get("max-tokens").and_then(|v| v.parse().ok()).unwrap_or(49152usize);
            let (hc, r) = cascade_coords(&up.coords, 512, res_req as usize, budget);
            res = r;
            coords = hc;
            tr.tokens.push(coords.len());
            self.mark(&mut tr, "upsample", &mut t0);
            let (c_hr, p_hr) = match &pcond {
                Some(pc) => pc.shape(self, &coords, res)?,
                None => (None, None),
            };
            let m = self.shape_flow(true)?;
            slat = self.flow(&m, &shp, &coords, draw(coords.len() * 32), c_hr.as_ref().unwrap_or(cond1024.as_ref().unwrap()), None, p_hr.as_ref(),
                             |i, of| report("shape", i, of))?;
            drop(m);
            norm(&mut slat, &self.cfg.shape_mean, &self.cfg.shape_std);
        }
        tr.res = res;
        self.mark(&mut tr, "shape", &mut t0);
        // the shape decoder first: its mesh is cut from the voxels, oriented, decimated and unwrapped (and the bake's
        // closest-point tree built) on a thread of the CPU while the GPU runs the texture flow and decoder
        let sd = self.shape_dec()?;
        let mut dec = sd.run(&self.cx, &coords, &slat, &vae::Plan { guide: None, upsample: None, keep_ups: false, profile: self.profile }, |i, of| report("decode", i, of))?;
        drop(sd);
        if !self.resident {
            drop(self.models.shape_dec.0.lock().unwrap().take());
        }
        tr.voxels = dec.coords.len();
        self.mark(&mut tr, "decode", &mut t0);
        let tree = std::mem::take(&mut dec.tree);
        let faces = req.faces.unwrap_or(300_000) as usize;
        let size = req.texture_size.unwrap_or(if res <= 512 { 1024 } else { 2048 });
        let remesh = req.extra.contains_key("remesh");
        let mut surface = Mesh::default();
        let (mut m, vox, near) = std::thread::scope(|sc| -> Result<_> {
            let (sf, dc, texture, profile, arch) = (&mut surface, &dec, req.texture, self.profile, self.arch);
            let cpu = sc.spawn(move || {
                let t = Instant::now();
                let mut laps = Vec::new();
                let mut lap = |what: &'static str| laps.push((what, t.elapsed().as_secs_f64()));
                *sf = mesh::dual_grid(&dc.coords, &dc.feats, &dc.nmap, res);
                lap("dual grid");
                tools::orient(sf);
                let surface: &Mesh = sf;
                lap("oriented");
                // a dense surface clustered first to about four times the budget (cells of a few voxels), then decimated
                let cell = (surface.positions.len() as f32 / (2 * faces) as f32).sqrt() / res as f32;
                let mut m = match (remesh, surface.triangles() > 4 * faces) {
                    (true, _) => tools::remesh(surface, None, 3),
                    (false, true) => mesh::cluster(surface, cell),
                    (false, false) => surface.clone(),
                };
                if remesh && m.triangles() > 4 * faces {
                    m = mesh::cluster(&m, cell);
                }
                lap("clustered");
                if m.triangles() > faces {
                    m = tools::decimate(&m, faces);
                }
                lap("decimated");
                tools::orient(&mut m);
                lap("oriented");
                // and the decoded surface's closest points for the bake's texel search
                let near = texture.then(|| {
                    m.compute_normals();
                    m = tools::unwrap(&m, size, 4);
                    lap("unwrapped");
                    let near = tools::closest::Closest::new(&surface.positions, &surface.indices);
                    lap("the closest-point tree");
                    near
                });
                if profile {
                    let laps: Vec<String> = laps.iter().map(|(w, s)| format!("{w} {s:.2}")).collect();
                    eprintln!("{arch}: on the CPU, {} -> {} triangles, by s: {}", surface.triangles(),
                              m.triangles(), laps.join(", "));
                }
                (m, near)
            });
            // the texture latent, then its decoder on the shape decoder's subdivisions
            let vox = if req.texture {
                let sn: Vec<f32> = slat.chunks_exact(32).flat_map(|r| r.iter().enumerate().map(|(c, x)| (x - self.cfg.shape_mean[c]) / self.cfg.shape_std[c]).collect::<Vec<_>>())
                    .collect();
                let (c_tx, p_tx) = match &pcond {
                    Some(pc) => pc.texture(self, &coords, res)?,
                    None => (None, None),
                };
                let tm = self.tex_flow(cascade)?;
                let cond = c_tx.as_ref().unwrap_or(if cascade { cond1024.as_ref().unwrap() } else { &cond512 });
                let mut t = self.flow(&tm, &txp, &coords, draw(coords.len() * 32), cond, Some(&sn), p_tx.as_ref(), |i, of| report("texture", i, of))?;
                drop(tm);
                drop((c_tx, p_tx));
                norm(&mut t, &self.cfg.tex_mean, &self.cfg.tex_std);
                self.mark(&mut tr, "texture", &mut t0);
                let td = self.tex_dec()?;
                let tv = td.run(&self.cx, &coords, &t, &vae::Plan { guide: Some(&tree), upsample: None, keep_ups: false, profile: self.profile }, |i, of| report("decode", i, of))?;
                drop(td);
                if tv.coords != dec.coords {
                    return Err(Error("the texture decoder's voxels are not the shape decoder's".into()));
                }
                let attrs = tv.feats.iter().map(|v| v * 0.5 + 0.5).collect();
                let v = bake::Voxels::new(&tv.coords, attrs, res);
                self.mark(&mut tr, "tex-decode", &mut t0);
                Some(v)
            } else {
                None
            };
            report("mesh", 0, 1)?;
            let (m, near) = cpu.join().map_err(|_| Error("the mesh thread panicked".into()))?;
            report("mesh", 1, 1)?;
            Ok((m, vox, near))
        })?;
        drop((cond512, cond1024, pcond, dec));
        // the time the mesh's thread took past the GPU's work
        self.mark(&mut tr, "mesh", &mut t0);
        if let (Some(vox), Some(near)) = (&vox, &near) {
            report("bake", 0, 1)?;
            m.material = Some(bake::bake(&m, near, vox, size, req.extra.get("bake-push").and_then(|v| v.parse().ok()).unwrap_or(2.0)));
            report("bake", 1, 1)?;
            self.mark(&mut tr, "bake", &mut t0);
        }
        drop(near);
        drop(surface);
        // to glTF's y-up, the picture's view facing +z: TRELLIS's z-up (x, y, z) -> (x, z, -y); Pixal3D's grid is already
        // y-up with the camera on its +z side (its camera looks along the world's +y, the grid's -z)
        if self.arch != ARCH_PIXAL {
            for p in &mut m.positions {
                *p = [p[0], p[2], -p[1]];
            }
        }
        m.normals.clear();
        m.compute_normals();
        if let Some(p) = &peak {
            tr.peak_gib = p.gib();
            eprintln!("{}: total      {:>7.2} s   the card's peak {:.1} GiB in use (every process)", self.arch, start.elapsed().as_secs_f64(), tr.peak_gib);
        }
        Ok((m, tr))
    }
}

/// The cascade's second-pass voxels: the first pass's subdivided coordinates (at `lr` x 16 / 16... the decoder's
/// level 4, `lr` wide) quantized to res / 16, made unique (sorted, as torch.unique), the resolution stepping down by
/// 128 while the count reaches `budget` (never below 1024's floor rule: 1024 is taken whatever its count)
pub fn cascade_coords(up: &[[i32; 3]], lr: usize, res: usize, budget: usize) -> (Vec<[i32; 3]>, usize) {
    let mut r = res;
    loop {
        let g = (r / 16) as f32;
        let mut q: Vec<[i32; 3]> = up.iter().map(|c| c.map(|v| ((v as f32 + 0.5) / lr as f32 * g) as i32)).collect();
        q.sort_unstable();
        q.dedup();
        if q.len() < budget || r == 1024 || r <= 128 {
            return (q, r);
        }
        r -= 128;
    }
}

impl Model3dEngine for Trellis2 {
    fn arch(&self) -> &'static str {
        self.arch
    }
    fn abilities(&self) -> Abilities {
        Abilities { generate: true, texture: true, views: 1, rig: false, animate: false }
    }
    fn defaults(&self) -> Defaults {
        let hr = self.files.contains_key("shape_flow_1024") && self.files.contains_key("tex_flow_1024");
        Defaults { steps: self.cfg.shape.steps, cfg: self.cfg.shape.guidance as f32, resolution: if hr { 1024 } else { 512 }, faces: 300_000,
                   texture_size: if hr { 2048 } else { 1024 }, frames: 0, fps: 0.0 }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s.max(self.loaded.elapsed().as_secs_f64().min(0.0))
    }
    fn generate(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        if req.mesh.is_some() {
            return Err(Error("TRELLIS.2 paints the meshes it makes, not one it is given (hunyuan3d-2.1 or hunyuan3d-2.1-paint paint a mesh)".into()));
        }
        let mut r = req.clone();
        if r.resolution.is_none() {
            r.resolution = Some(self.defaults().resolution);
        }
        Ok(self.run(&r, progress)?.0)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{} on {} ({} models resident)", self.arch, self.cx.gpu.name, if self.resident { "all" } else { "no" })]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cascade_quantizes_and_steps_down() {
        // 512-wide coordinates to 1024 / 16 = 64: (c + 0.5) / 512 * 64 truncated
        let up = vec![[0, 0, 0], [7, 7, 7], [8, 0, 0], [511, 511, 511]];
        let (q, r) = cascade_coords(&up, 512, 1024, 100);
        assert_eq!(r, 1024);
        assert_eq!(q, vec![[0, 0, 0], [1, 0, 0], [63, 63, 63]]);
        // a budget of 2 at 1536: down by 128 until fewer than 2 or 1024
        let (q, r) = cascade_coords(&up, 512, 1536, 2);
        assert_eq!(r, 1024);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn the_noise_is_seeded_and_standard() {
        let a = noise(7, 100_000);
        assert_eq!(a, noise(7, 100_000));
        assert_ne!(a[..10], noise(8, 10)[..]);
        let mean = a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.02, "{mean} {var}");
    }
}
