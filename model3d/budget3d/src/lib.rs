//! Budget3D-Idol and Budget3D-Idol-Turbo (BudgetAny's compute-adaptive fine-tunes of Sperid Labs' Idol): one picture
//! to 3D Gaussian splats, at a token budget. The pipeline, as Budget3D's scripts/sample.py runs it:
//!
//! ```text
//!   picture -> Idol's framing: short side 1024, cut out (BiRefNet) unless it has a real alpha, the alpha eroded,
//!              a square of 1.2 (renders) or 1.7 (photos) x the object, 1024 x 1024 on black        prep.rs
//!           -> DINOv3 ViT-H+/16 tokens [4101, 1280] and FLUX.2 VAE tokens [4101, 128]            dino.rs, vae.rs
//!   noise   -> Idol's flow transformer over the latent [8192, 16] and the camera [5]: 50 Euler steps guided 3
//!              (Turbo: 5 steps, unguided, its phase experts), time shifted by 3; at a budget < 1 the latent runs
//!              as fewer, merged tokens (the layout from the noise, refreshed from the clean estimate)   dit.rs, layout.rs
//!           -> the octree decoder's anchors, the Gaussian decoder's 32 Gaussians an anchor             decoder.rs
//!           -> Splats (y up, the PLY's frame) and the camera the picture was seen from
//! ```
//!
//! The denoiser and the decoders stay loaded; the encoders and the background remover load for a request and go after
//! it (NS_M3D_B3D_RESIDENT=1 keeps them). `check` compares every stage with reference/budget3d/ref.py's dumps.

pub mod check;
pub mod decoder;
pub mod dino;
pub mod dit;
mod ffi;
pub mod layout;
pub mod prep;
pub mod vae;
pub mod w;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
pub use nextsycl_diffusion::Picture;
use nextsycl_model3d::{Abilities, At, Defaults, EngineOption, Error, LoadOptions, MeshRequest, Model3dEngine, Model3dKind, ModelFiles, Result, Splats,
                       Step};
use nextsycl_remover::Remover;

use crate::layout::Layout;
use crate::w::St;

pub const ARCH: &str = "budget3d-idol";

/// The files by role: Budget3D's denoiser (or Turbo's), Idol's own denoiser (its Sobol positions), Idol's two decoders,
/// the gated image encoders (DINOv3 ViT-H+/16, the FLUX.2 VAE: a picture needs both), BiRefNet
pub const ROLES: &[&str] = &["denoiser", "idol", "decoder", "decoder_gs", "dino?", "vae?", "remover?"];

const GIB: f64 = (1u64 << 30) as f64;

/// The options it takes (`--opt-NAME`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "budget", env: "", value: "FRACTION",
                   help: "the share of the 8192 latent tokens the denoiser runs (0.0625 to 1, default 1): the latent's flattest regions, empty space first, are merged into coarser tokens - faster, less fine detail",
                   at: At::Request },
    EngineOption { name: "gaussians", env: "", value: "N", help: "Gaussians in the result (32768 to 1048576, a multiple of 32; default 262144)", at: At::Request },
    EngineOption { name: "mode", env: "", value: "synthetic|real",
                   help: "Idol's crop: synthetic (renders and generated pictures: 1.2 x the object; the default) or real (photographs: 1.7)", at: At::Request },
    EngineOption { name: "crop-margin", env: "", value: "X", help: "the square around the object, as a multiple of its size (overrides mode)", at: At::Request },
    EngineOption { name: "dino", env: "NS_M3D_B3D_DINO", value: "FILE",
                   help: "DINOv3 ViT-H+/16's model.safetensors (gated: facebook/dinov3-vith16plus-pretrain-lvd1689m), when the model's files do not name it", at: At::Load },
    EngineOption { name: "vae", env: "NS_M3D_B3D_VAE", value: "FILE",
                   help: "the FLUX.2 VAE's diffusion_pytorch_model.safetensors (gated: black-forest-labs/FLUX.2-dev, vae/), when the model's files do not name it",
                   at: At::Load },
    EngineOption { name: "resident", env: "NS_M3D_B3D_RESIDENT", value: "", help: "keep the image encoders and the background remover loaded between requests", at: At::Load },
    EngineOption { name: "profile", env: "NS_M3D_B3D_PROFILE", value: "", help: "each stage's time and the card's memory, on stderr", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> Model3dKind {
    Model3dKind { archs: &[ARCH], name: "Budget3D-Idol (DINOv3 ViT-H+ and FLUX.2 VAE tokens, Idol's flow transformer at a token budget, octree and Gaussian decoders: splats)",
                  roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(Budget3d::load(files, gpu, o, log)?))
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

/// `torch.manual_seed(seed)` then `torch.randn` of [8192, 16] and of [5], as the samplers draw them: the latent through
/// the batched Box-Muller (16 values or more), the camera's 5 through PyTorch's scalar path (double precision pairs, the
/// second of each pair kept for the next draw)
pub fn noise(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut g = nextsycl_diffusion::torch_rng::Mt19937::new(seed);
    let z = g.randn(layout::N_SOBOL * dit::IN);
    let mut cam = Vec::with_capacity(dit::CAM);
    let mut cached: Option<f64> = None;
    for _ in 0..dit::CAM {
        if let Some(v) = cached.take() {
            cam.push(v as f32);
            continue;
        }
        let mut u = || {
            let hi = g.next_u32() as u64;
            let lo = g.next_u32() as u64;
            ((hi << 32 | lo) & ((1u64 << 53) - 1)) as f64 * (2f64).powi(-53)
        };
        let (u1, u2) = (u(), u());
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u1;
        cached = Some(r * th.sin());
        cam.push((r * th.cos()) as f32);
    }
    (z, cam)
}

/// Idol's shifted time grid: steps + 1 values from 1 down to 0
pub fn time_grid(steps: usize, shift: f64) -> Vec<f64> {
    (0..=steps).map(|i| {
        let t = 1.0 - i as f64 / steps as f64;
        shift * t / (1.0 + (shift - 1.0) * t)
    }).collect()
}

/// Idol's camera from the sampled camera latent [5]: yaw and pitch (radians), distance, vertical field of view (degrees)
pub fn decode_camera(c: &[f32]) -> [f32; 4] {
    let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let (dx, dy, dz) = if n > 1e-6 { (c[0] / n, c[1] / n, c[2] / n) } else { (0.0, 0.0, 1.0) };
    let (inv, half) = (c[3], c[4]);
    let (radius, fov) = if inv < 1e-3 { (1000.0, 2.0 * (half / 1000.0).atan()) } else { (1.0 / inv, 2.0 * (half * inv).atan()) };
    [dx.atan2(dy), dz.clamp(-1.0, 1.0).asin(), radius, fov.to_degrees()]
}

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
        let t = Arc::new(load()?);
        if resident {
            *g = Some(t.clone());
        }
        Ok(t)
    }
}

/// Watches the card's free memory every 50 ms until dropped: the run's peak (every process's use)
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

/// A request's settings, from its fields and options
#[derive(Clone, Debug)]
pub struct Params {
    pub steps: usize,
    pub cfg: f32,
    pub budget: f64,
    pub gaussians: usize,
    pub margin: f32,
    pub seed: u64,
}

/// What a run leaves besides the splats: each stage's seconds and the card's memory then, the peak
#[derive(Default)]
pub struct Trace {
    pub tokens: usize,
    pub stages: Vec<(&'static str, f64, f64)>,
    pub peak_gib: f64,
}

pub struct Budget3d {
    pub files: ModelFiles,
    pub cx: Ctx,
    pub dit: dit::Dit,
    pub dec: decoder::OctreeDecoder,
    pub gs: decoder::GaussianDecoder,
    dino: Slot<dino::Dino>,
    vae: Slot<vae::VaeEncoder>,
    remover: Slot<Remover>,
    resident: bool,
    profile: bool,
    load_s: f64,
    run: Mutex<()>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// What a picture needs that this model's files lack: the gated encoders
const GATED: &str = "the image encoders are gated on Hugging Face: DINOv3 ViT-H+/16 (facebook/dinov3-vith16plus-pretrain-lvd1689m, the DINOv3 License) and the FLUX.2 VAE (black-forest-labs/FLUX.2-dev, vae/) - accept their terms, download them and register them as this model's dino and vae files";

impl Budget3d {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Budget3d> {
        let t0 = Instant::now();
        let missing = nextsycl_model3d::missing(files, ROLES);
        if !missing.is_empty() {
            return Err(Error(format!("{ARCH}: no {} file", missing.join(", "))));
        }
        let cx = Ctx::new(gpu)?;
        let flag = |n: &str| o.setting(n).is_some_and(|v| v != "0" && !v.is_empty());
        // the gated encoders: the model's files, else the settings that name them (the card's local parts)
        let mut files = files.clone();
        for (r, var) in [("dino", "NS_M3D_B3D_DINO"), ("vae", "NS_M3D_B3D_VAE")] {
            if let Some(p) = o.setting(var).filter(|v| !v.is_empty()).filter(|_| !files.contains_key(r)) {
                if !Path::new(&p).is_file() {
                    return Err(Error(format!("{var}={p}: no such file")));
                }
                files.insert(r.to_string(), p.into());
            }
        }
        let files = &files;
        let dit = dit::Dit::load(&St::open(role(files, "denoiser")?)?, &St::open(role(files, "idol")?)?, &cx, log)?;
        let dec = decoder::OctreeDecoder::load(&St::open(role(files, "decoder")?)?, gpu)?;
        let gs = decoder::GaussianDecoder::load(&St::open(role(files, "decoder_gs")?)?, gpu)?;
        gpu.sync()?;
        let e = Budget3d { files: files.clone(), cx, dit, dec, gs, dino: Slot::new(), vae: Slot::new(), remover: Slot::new(), resident: flag("NS_M3D_B3D_RESIDENT"),
                           profile: flag("NS_M3D_B3D_PROFILE"), load_s: t0.elapsed().as_secs_f64(), run: Mutex::new(()) };
        log(format!("{ARCH}: {} denoiser ({:.2} GB) and the decoders on {} in {:.1} s{}", if e.dit.turbo { "Turbo" } else { "the 50-step" },
                    e.dit.bytes as f64 / 1e9, gpu.name, e.load_s,
                    if files.contains_key("dino") && files.contains_key("vae") { "" } else { "; no image encoders (gated): pictures cannot be turned into splats until they are added" }));
        Ok(e)
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

    /// The request's settings
    pub fn params(&self, req: &MeshRequest) -> Result<Params> {
        let d = self.defaults();
        let num = |k: &str| -> Result<Option<f64>> {
            req.extra.get(k).map(|v| v.parse::<f64>().map_err(|_| Error(format!("{k} {v:?}: a number")))).transpose()
        };
        let budget = num("budget")?.unwrap_or(1.0);
        if !(0.0625..=1.0).contains(&budget) {
            return Err(Error(format!("budget {budget}: 0.0625 to 1 (the share of the 8192 latent tokens)")));
        }
        let gaussians = num("gaussians")?.map_or(262_144, |v| v as usize);
        if !(32_768..=1_048_576).contains(&gaussians) {
            return Err(Error(format!("gaussians {gaussians}: 32768 to 1048576")));
        }
        let margin = match num("crop-margin")? {
            Some(m) if (1.0..=4.0).contains(&m) => m as f32,
            Some(m) => return Err(Error(format!("crop-margin {m}: 1 to 4"))),
            None => prep::margin_of(req.extra.get("mode").map_or("synthetic", String::as_str))?,
        };
        let steps = req.steps.unwrap_or(d.steps) as usize;
        if !(1..=200).contains(&steps) {
            return Err(Error(format!("steps {steps}: 1 to 200")));
        }
        Ok(Params { steps, cfg: if self.dit.turbo { 0.0 } else { req.cfg.unwrap_or(d.cfg) }, budget, gaussians: gaussians / decoder::PER * decoder::PER, margin,
                    seed: req.seed })
    }

    /// The background remover, for the check
    pub fn remover_for_check(&self) -> Result<Option<Arc<Remover>>> {
        self.remover()
    }

    fn remover(&self) -> Result<Option<Arc<Remover>>> {
        let Some(p) = self.files.get("remover") else { return Ok(None) };
        Ok(Some(self.remover.get(self.resident, || {
            self.room("the background remover", 1.5 * GIB)?;
            Remover::load(p, &self.cx.gpu)
        })?))
    }

    /// The picture's two token sets (feature1 [4101, 1280] DINOv3, feature2 [4101, 128] FLUX.2 VAE)
    pub fn encode(&self, pre: &Picture) -> Result<(Vec<f32>, Vec<f32>)> {
        let (Some(dp), Some(vp)) = (self.files.get("dino"), self.files.get("vae")) else {
            return Err(Error(format!("{ARCH}: {GATED}")));
        };
        let x = prep::planes(pre);
        let f1 = {
            let d = self.dino.get(self.resident, || {
                self.room("DINOv3 ViT-H+", 3.6 * GIB)?;
                dino::Dino::load(&St::open(dp)?, &self.cx)
            })?;
            d.encode(&self.cx, &x, prep::SIDE as usize)?
        };
        let f2 = {
            let v = self.vae.get(self.resident, || vae::VaeEncoder::load(&St::open(vp)?, &self.cx.gpu))?;
            self.room("the VAE encoder", 2.0 * GIB)?;
            v.encode(&self.cx.nsd, &self.cx.k, &x, prep::SIDE as usize)?
        };
        Ok((f1, f2))
    }

    /// The sampler from the seed's noise to the latent [8192, 16] and the camera [5], given the picture's tokens
    /// (Budget3D's sample / sample_turbo); `step` hears each step
    pub fn sample(&self, f1: &[f32], f2: &[f32], p: &Params, step: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<(Vec<f32>, Vec<f32>, Layout)> {
        self.sample_with(f1, f2, p, step, &mut |_, _, _, _| {})
    }

    /// `sample`, with each step's latent, camera and layout handed to `on` (the check)
    #[allow(clippy::type_complexity)]
    pub fn sample_with(&self, f1: &[f32], f2: &[f32], p: &Params, step: &mut dyn FnMut(u32, u32) -> Result<()>,
                       on: &mut dyn FnMut(usize, &[f32], &[f32], &Layout)) -> Result<(Vec<f32>, Vec<f32>, Layout)> {
        let cx = &self.cx;
        let n = layout::tokens_of(p.budget);
        let turbo = self.dit.turbo;
        let guided = !turbo && p.cfg > 0.0;
        let l = n + dit::L_COND + 1;
        let rows = if guided { 2 * l } else { l };
        self.room("the denoiser's activations", (rows * dit::D * (4 * 3 + 2 * 6) + rows * 4096 * 2 + 2 * dit::L_COND * dit::D * 4) as f64)?;
        let w = self.dit.work(&cx.gpu, n, if guided { 2 } else { 1 })?;
        let zeros1 = vec![0f32; f1.len()];
        let zeros2 = vec![0f32; f2.len()];
        // the refined condition: guided and unguided, or Turbo's two phases (its context refiner carries the phase's LoRA)
        let ctxs: Vec<DevBuf> = if turbo {
            vec![self.dit.context(cx, &w, f1, f2, 0)?, self.dit.context(cx, &w, f1, f2, 1)?]
        } else if guided {
            vec![self.dit.context(cx, &w, f1, f2, 0)?, self.dit.context(cx, &w, &zeros1, &zeros2, 0)?]
        } else {
            vec![self.dit.context(cx, &w, f1, f2, 0)?]
        };
        let (mut z, mut cam) = noise(p.seed);
        let grid = time_grid(p.steps, 3.0);
        let mut lay: Option<Layout> = None;
        let mut z0 = z.clone();
        for i in 0..p.steps {
            let (t, tp) = (grid[i] as f32, grid[i + 1] as f32);
            // the layout: from the noise at the first step, then from the running clean estimate (base: at step 5
            // only; Turbo: every step)
            if lay.is_none() || turbo || i == 5 {
                let r = if i == 0 { &z } else { &z0 };
                let (s4, s2) = self.dit.octree.detail_scores(r, dit::IN);
                lay = Some(self.dit.octree.build(&s4, &s2, n)?);
            }
            let ly = lay.as_ref().unwrap();
            let ctx: Vec<&DevBuf> = if turbo { vec![&ctxs[dit::phase(true, t)]] } else { ctxs.iter().collect() };
            let (v, vc) = self.dit.velocity(cx, &w, &z, &cam, t, ly, &ctx, p.cfg, None)?;
            for (a, b) in z0.iter_mut().zip(z.iter().zip(&v)) {
                *a = b.0 - t * b.1;
            }
            for (a, b) in z.iter_mut().zip(&v) {
                *a -= (t - tp) * b;
            }
            for (a, b) in cam.iter_mut().zip(&vc) {
                *a -= (t - tp) * b;
            }
            on(i, &z, &cam, ly);
            step(i as u32 + 1, p.steps as u32)?;
        }
        Ok((z, cam, lay.unwrap_or_else(Layout::dense)))
    }

    /// The latent to Gaussians: the octree's anchors (a seeded draw), the Gaussian decoder, activated (y up)
    pub fn decode(&self, z: &[f32], cam: &[f32], gaussians: usize, seed: u64) -> Result<Splats> {
        let lat = decoder::latent(&self.cx.gpu, z)?;
        let mut g = nextsycl_diffusion::torch_rng::Mt19937::new(seed ^ 0x6a09_e667_f3bc_c908);
        let mut rng = || ((g.next_u32() >> 8) as f32) / (1u32 << 24) as f32;
        let anchors = self.dec.anchors(&self.cx, &lat, gaussians / decoder::PER, &mut rng, |_, _, _| {})?;
        let feats = self.gs.features(&self.cx, &lat, &anchors)?;
        let mut s = self.gs.gaussians(&anchors, &feats);
        s.z_up_to_y_up();
        s.camera = Some(decode_camera(cam));
        Ok(s)
    }

    fn mark(&self, tr: &mut Trace, what: &'static str, t0: &mut Instant) {
        let s = t0.elapsed().as_secs_f64();
        let used = self.cx.gpu.memory().ok().and_then(|(t, f)| f.map(|f| (t - f) as f64 / GIB)).unwrap_or(0.0);
        if self.profile {
            eprintln!("{ARCH}: {what:<9} {s:>7.2} s   card in use {used:.1} GiB");
        }
        tr.stages.push((what, s, used));
        *t0 = Instant::now();
    }

    /// The whole pipeline for a request
    pub fn run(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Splats, Trace)> {
        let _one = self.run.lock().unwrap();
        let p = self.params(req)?;
        let peak = Peak::new(&self.cx.gpu);
        let start = Instant::now();
        let mut tr = Trace { tokens: layout::tokens_of(p.budget), ..Default::default() };
        let mut t0 = Instant::now();
        let mut report = |phase: &'static str, at: u32, of: u32| progress(Step { phase, at, of, seconds: start.elapsed().as_secs_f64() });
        let view = req.views.first().ok_or_else(|| Error("no picture".into()))?;
        if !self.files.contains_key("dino") || !self.files.contains_key("vae") {
            return Err(Error(format!("{ARCH}: {GATED}")));
        }
        report("image", 0, 1)?;
        let tex = &view.image;
        let pic = Picture { width: tex.width, height: tex.height, channels: 4, data: tex.rgba.clone() };
        let cut = req.remove_background.unwrap_or(true) && !prep::has_real_alpha(&pic);
        let r = if cut { self.remover()? } else { None };
        let pre = prep::prepare(&pic, r.as_deref(), p.margin)?;
        drop(r);
        self.mark(&mut tr, "prepare", &mut t0);
        let (f1, f2) = self.encode(&pre)?;
        report("image", 1, 1)?;
        self.mark(&mut tr, "encode", &mut t0);
        let (z, cam, _) = self.sample(&f1, &f2, &p, &mut |i, of| report("denoise", i, of))?;
        self.mark(&mut tr, "denoise", &mut t0);
        report("decode", 0, 1)?;
        let s = self.decode(&z, &cam, p.gaussians, p.seed)?;
        report("decode", 1, 1)?;
        self.mark(&mut tr, "decode", &mut t0);
        tr.peak_gib = peak.gib();
        Ok((s, tr))
    }
}

impl Model3dEngine for Budget3d {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn abilities(&self) -> Abilities {
        Abilities { generate: true, texture: false, views: 1, rig: false, animate: false, splats: true }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: if self.dit.turbo { 5 } else { 50 }, cfg: if self.dit.turbo { 0.0 } else { 3.0 }, ..Default::default() }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &MeshRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<nextsycl_model3d::Mesh> {
        Err(Error(format!("{ARCH} makes Gaussian splats (a PLY), not meshes: model3d.generate answers with them")))
    }
    fn splat(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Splats> {
        let (s, tr) = self.run(req, progress)?;
        if self.profile {
            eprintln!("{ARCH}: {} Gaussians, {} tokens, peak {:.1} GiB on the card", s.len(), tr.tokens, tr.peak_gib);
        }
        Ok(s)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}: {} denoiser {:.2} GB resident{}", self.cx.gpu.name, if self.dit.turbo { "Turbo" } else { "50-step" }, self.dit.bytes as f64 / 1e9,
                     if self.files.contains_key("dino") && self.files.contains_key("vae") { "" } else { "; no image encoders (gated)" })]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_grid_is_shifted_toward_noise() {
        let g = time_grid(5, 3.0);
        assert_eq!(g.len(), 6);
        assert_eq!(g[0], 1.0);
        assert_eq!(g[5], 0.0);
        // t = 0.8 -> 3 * 0.8 / (1 + 2 * 0.8)
        assert!((g[1] - 2.4 / 2.6).abs() < 1e-12);
    }

    #[test]
    fn noise_is_torchs() {
        // torch.manual_seed(0); torch.randn(16)[:2] (the latent's first values)
        let (z, cam) = noise(0);
        assert!((z[0] + 1.1258398).abs() < 1e-5 && (z[1] + 1.1523602).abs() < 1e-5);
        assert_eq!(cam.len(), 5);
    }

    #[test]
    fn the_camera_decodes_to_angles() {
        let c = decode_camera(&[0.0, 1.0, 0.0, 0.5, 0.5]);
        assert!(c[0].abs() < 1e-6 && c[1].abs() < 1e-6);
        assert!((c[2] - 2.0).abs() < 1e-6);
        assert!((c[3] - 2.0 * 0.25f32.atan().to_degrees()).abs() < 1e-4);
    }
}
