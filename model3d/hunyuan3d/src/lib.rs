//! Hunyuan3D (Tencent): a mesh from one picture of an object, two architectures on one engine.
//!
//! - **hunyuan3d-2mini** (2mini-turbo): DINOv2-giant at 1,022 px, the Flux-style `Hunyuan3DDiT` (8 double + 16
//!   single blocks, guidance-distilled: no CFG batch) over 3,072 latent tokens, ConsistencyFlowMatchEuler in 5
//!   steps, and the turbo ShapeVAE decoder (FlashVDM's light geometry decoder). Tencent's turbo example samples 512
//!   tokens (the turbo decoder's `latent_shape`, which its pipeline reads); the DiT's own config and ComfyUI use
//!   3,072, and in tests 3,072 keeps detail 512 loses (lettering, creases) for about a second more.
//! - **hunyuan3d-2.1**: DINOv2-Large at 518 px, `HunYuanDiTPlain` (21 blocks, U-ViT skips, MoE in the last 6, the
//!   timestep as a token) over 4,096 tokens, FlowMatchEuler in 50 steps with CFG 5, the ShapeVAE decoder.
//!
//! The pipeline (`generate`): the picture cut out and recentred (`picture`), its DINOv2 tokens (`dino`), the latents
//! denoised from the seed (`flux` / `plain`, `sched`), decoded to occupancy logits on a grid in [-1.01, 1.01]^3 -
//! coarse to fine near the surface (`grid`, `vae`) - meshed by marching cubes at 0 (`nextsycl_mesh::tools::mc`),
//! small floating pieces dropped, finished (decimated to `faces`); with `texture` (2.1), painted by the paint model
//! (`paint`: six views of albedo and metallic-roughness, baked into a UV atlas). The parity of every stage with
//! Tencent's code is `check`'s.

pub mod check;
pub mod dino;
mod ffi;
pub mod flux;
pub mod grid;
pub mod ops;
pub mod paint;
pub mod painter;
pub mod picture;
pub mod prof;
pub mod plain;
pub mod sched;
pub mod vae;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{At, DevBuf, EngineOption, Gpu};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_model3d::{Abilities, Defaults, Error, LoadOptions, Mesh, MeshRequest, Model3dEngine, Model3dKind, ModelFiles, Result, Step};

use crate::ops::{read_f32, Ops, Weights};

pub const ARCH_21: &str = "hunyuan3d-2.1";
pub const ARCH_MINI: &str = "hunyuan3d-2mini";
/// 2.1's paint model alone (no shape model: a mesh from another engine, painted - the 12 GB cards' form)
pub const ARCH_PAINT: &str = "hunyuan3d-2.1-paint";

/// The files by role: `model` (the DiT file with its conditioner and VAE: Tencent's mini-turbo safetensors, or
/// Comfy-Org's 2.1 repackage), `vae` (2mini: the turbo decoder), and the paint model's (2.1's textures: its UNet and
/// VAE checkpoints, DINOv2-giant)
pub const ROLES: &[&str] = &["model?", "vae?", "paint-unet?", "paint-vae?", "paint-dino?", "remover?"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "tokens", env: "NS_M3D_HY_TOKENS", value: "N", help: "latent tokens of the shape (2mini 3072, 2.1 4096; Tencent's turbo example samples 512)", at: At::Both },
    EngineOption { name: "decode", env: "NS_M3D_HY_DECODE", value: "fast|dense", help: "the grid: coarse to fine near the surface (fast) or every point", at: At::Both },
    EngineOption { name: "chunk", env: "NS_M3D_HY_CHUNK", value: "N", help: "points the shape decoder queries at once", at: At::Both },
    EngineOption { name: "paint", env: "NS_M3D_HY_PAINT", value: "0|1", help: "2.1: load the paint model (textures; ~6.5 GiB more) when its files are there", at: At::Load },
    EngineOption { name: "paint-steps", env: "NS_M3D_HY_PAINT_STEPS", value: "N", help: "2.1: the texture's UniPC steps (15)", at: At::Both },
    EngineOption { name: "int8", env: "NS_M3D_HY_INT8", value: "0|1", help: "2.1: the DiT's block matrices as int8 ConvRot (faster steps, 2.7 GiB less VRAM; approximate: see docs/environment.md)", at: At::Load },
    EngineOption { name: "attn", env: "NS_M3D_HY_ATTN", value: "flash|shared", help: "self-attention: the strided flash kernel (half), or the shared kernels' choice (oneDNN; int8 SageAttention from 8,192 rows)", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> Model3dKind {
    Model3dKind {
        archs: &[ARCH_21, ARCH_MINI, ARCH_PAINT],
        name: "Hunyuan3D (2.1: shape and PBR paint; 2mini-turbo: fast shape) - a mesh from a picture",
        roles: ROLES,
        load,
        options: OPTIONS,
    }
}

/// What differs between the two architectures
#[derive(Clone, Copy, Debug)]
pub struct Arch {
    pub name: &'static str,
    /// the picture: the processor's size, the encoder's
    pub proc_size: usize,
    pub enc_size: usize,
    pub tokens: usize,
    pub steps: u32,
    pub cfg: f32,
    pub resolution: u32,
    pub scale_factor: f32,
}

pub const MINI: Arch = Arch { name: ARCH_MINI, proc_size: 1022, enc_size: 1022, tokens: 3072, steps: 5, cfg: 5.0, resolution: 380, scale_factor: 1.018_813_7 };
pub const V21: Arch = Arch { name: ARCH_21, proc_size: 512, enc_size: 518, tokens: 4096, steps: 50, cfg: 5.0, resolution: 384, scale_factor: 1.003_950_6 };

/// The box the shape lives in: [-BOUNDS, BOUNDS]^3
pub const BOUNDS: f32 = 1.01;
const BORDER: f32 = 0.15;
/// The smallest share of the triangles a connected piece keeps (Tencent's FloaterRemover: 0.5 %)
const FLOATERS: f32 = 0.005;

#[allow(clippy::large_enum_variant)] // one a loaded engine
pub enum Dit {
    Flux(flux::Flux),
    Plain(plain::Plain),
}

pub struct Hunyuan3d {
    pub arch: Arch,
    pub ops: Ops,
    pub dino: dino::Dino,
    pub dit: Dit,
    pub vae: vae::Vae,
    loaded: Instant,
    load_seconds: f64,
    settings: LoadOptions,
    weights_bytes: usize,
    /// 2.1's paint model, when its files are there (and NS_M3D_HY_PAINT is not 0)
    pub paint: Option<paint::Paint>,
    /// the shared background remover (BiRefNet, nextsycl-remover), when the entry names its file: an opaque
    /// picture is cut out by it (else by picture.rs's border fallback)
    remover: Option<nextsycl_remover::Remover>,
    busy: Mutex<()>,
    /// the card's least free memory seen during the last request (bytes), for `report`
    low: std::sync::atomic::AtomicU64,
}

/// A seeded standard-normal draw (splitmix64, Box-Muller): the starting noise
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

fn open(files: &ModelFiles, role: &str) -> Result<SafeTensors> {
    let p = files.get(role).ok_or_else(|| Error(format!("no {role} file")))?;
    SafeTensors::open(p).map_err(|e| Error(e.0))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn Model3dEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    if files.get("model").is_none() && files.get("paint-unet").is_some() {
        return Ok(Box::new(painter::Painter::load(files, gpu, o, log)?));
    }
    let model = open(files, "model")?;
    // the architecture by the file's tensors: the Flux-style DiT has double blocks
    let arch = if model.tensor("model.double_blocks.0.img_mod.lin.weight").is_some() { MINI } else { V21 };
    Ok(Box::new(Hunyuan3d::load(arch, files, &model, gpu, o, log)?))
}

/// 2.1's paint model from its three files by role (the UNet's and VAE's are PyTorch pickles, DINOv2-giant's safetensors)
pub fn paint_from(files: &ModelFiles, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<paint::Paint> {
    let open_any = |role: &str| -> Result<SafeTensors> {
        let p = files.get(role).ok_or_else(|| Error(format!("no {role} file")))?;
        let r = if p.extension().is_some_and(|e| e == "safetensors") { SafeTensors::open(p) } else { nextsycl_gguf::pth::open(p) };
        r.map_err(|e| Error(e.0))
    };
    let (u, v, d) = (open_any("paint-unet")?, open_any("paint-vae")?, open_any("paint-dino")?);
    paint::Paint::check_files(&u)?;
    paint::Paint::load(&u, &v, &d, gpu, log)
}

impl Hunyuan3d {
    pub fn load(arch: Arch, files: &ModelFiles, model: &SafeTensors, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Hunyuan3d> {
        let t0 = Instant::now();
        let mut ops = Ops::new(gpu)?;
        ops.flash = match o.setting("NS_M3D_HY_ATTN").as_deref() {
            None | Some("flash") => true,
            Some("shared") => false,
            Some(x) => return Err(Error(format!("NS_M3D_HY_ATTN {x}: flash or shared"))),
        };

        if let Some(s) = ops.silo {
            log(format!("{}: the silo {} (its norms, GEGLU and adds)", arch.name, s.path.display()));
        }
        let dw = Weights::new(model, "conditioner.main_image_encoder.model.", gpu);
        let dino = dino::Dino::load(&dw, arch.enc_size)?;
        log(format!("{}: DINOv2 of {} layers x {} at {} px ({} tokens)", arch.name, dino.layers(), dino.width, dino.size, dino.tokens()));
        let mw = Weights::new(model, "model.", gpu);
        let mut dit = if arch.name == ARCH_MINI { Dit::Flux(flux::Flux::load(&mw)?) } else { Dit::Plain(plain::Plain::load(&mw)?) };
        let mut weights_bytes = 0usize;
        // 2.1's DiT only: 2mini's steps are no faster in int8 (its products are small), only less exact
        if o.setting("NS_M3D_HY_INT8").is_some_and(|v| v == "1") && arch.name == ARCH_21 {
            let t = Instant::now();
            let had = ops.hadamard()?;
            let saved = if let Dit::Plain(p) = &mut dit { p.to_int8(&ops, &had)? } else { 0 };
            weights_bytes = weights_bytes.wrapping_sub(saved);
            log(format!("{}: the DiT's block matrices as int8 ConvRot in {:.1} s ({:.2} GiB less)", arch.name, t.elapsed().as_secs_f64(),
                        saved as f64 / (1u64 << 30) as f64));
        }
        let vae = if arch.name == ARCH_MINI {
            let vf = open(files, "vae").map_err(|e| Error(format!("{}: the turbo decoder: {}", arch.name, e.0)))?;
            vae::Vae::load(&Weights::new(&vf, "", gpu), arch.scale_factor)?
        } else {
            vae::Vae::load(&Weights::new(model, "vae.", gpu), arch.scale_factor)?
        };
        let want_paint = o.setting("NS_M3D_HY_PAINT").is_none_or(|v| v != "0");
        let paint = match (files.get("paint-unet"), files.get("paint-vae"), files.get("paint-dino")) {
            (Some(_), Some(_), Some(_)) if arch.name == ARCH_21 && want_paint => Some(paint_from(files, gpu, log)?),
            _ => None,
        };
        let remover = match files.get("remover") {
            Some(p) => {
                let r = nextsycl_remover::Remover::load(p, gpu).map_err(|e| Error(format!("the background remover: {}", e.0)))?;
                log(format!("{}: the background remover (BiRefNet) loaded", arch.name));
                Some(r)
            }
            None => None,
        };
        gpu.sync()?;
        let weights_bytes = weights_bytes.wrapping_add(dw.bytes.get() + mw.bytes.get() + vae.bytes + paint.as_ref().map_or(0, |p| p.bytes));
        let load_seconds = t0.elapsed().as_secs_f64();
        log(format!("{}: loaded on {} in {load_seconds:.1} s ({:.2} GiB of weights)", arch.name, gpu.name, weights_bytes as f64 / (1u64 << 30) as f64));
        Ok(Hunyuan3d { arch, ops, dino, dit, vae, loaded: Instant::now(), load_seconds, settings: o.clone(), weights_bytes, paint, remover, busy: Mutex::new(()),
                       low: u64::MAX.into() })
    }

    /// Notes the card's free memory now (the low point of a request is in `report`)
    fn note_memory(&self) {
        if let Ok((_, Some(free))) = self.ops.gpu.memory() {
            self.low.fetch_min(free, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// A setting: the request's option, else the load's, else the environment
    fn setting(&self, req: &MeshRequest, name: &str, env: &str) -> Option<String> {
        req.extra.get(name).cloned().or_else(|| self.settings.setting(env))
    }

    /// The DiT's velocity for latents `x` at sigma `t` (guided: CFG for 2.1, the distilled guidance for 2mini)
    fn velocity(&self, st: &mut Run, x: &DevBuf, t: f32, cfg: f32, v: &DevBuf) -> Result<()> {
        match (&self.dit, &mut st.dit) {
            (Dit::Flux(f), RunDit::Flux(sc)) => f.forward(&self.ops, sc, x, t, cfg, v, None),
            (Dit::Plain(p), RunDit::Plain(sc)) => p.forward_cfg(&self.ops, sc, x, t, cfg, v),
            _ => Err(Error("the DiT and its scratch disagree".into())),
        }
    }

    /// The DINOv2 tokens of a picture's pixels (float32 [3, size, size], normalized)
    pub fn condition(&self, pixels: &[f32]) -> Result<DevBuf> {
        self.dino.forward(&self.ops, pixels)
    }

    /// The per-generation state for `tokens` latent rows beside `cond`
    pub fn run_state(&self, cond: &DevBuf, tokens: usize) -> Result<Run> {
        let rows = self.dino.tokens();
        let dit = match &self.dit {
            Dit::Flux(f) => RunDit::Flux(f.scratch(&self.ops, cond, rows, tokens)?),
            Dit::Plain(p) => RunDit::Plain(p.scratch(&self.ops, cond, rows, tokens)?),
        };
        Ok(Run { dit, tokens })
    }

    /// The latents from `noise` (host [tokens, 64]) over `sigmas`, reporting each step
    pub fn sample(&self, st: &mut Run, noise_v: &[f32], sigmas: &[f32], cfg: f32, progress: &mut dyn FnMut(u32, u32) -> Result<()>) -> Result<Vec<f32>> {
        let n = noise_v.len();
        let x = DevBuf::from_f32(&self.ops.gpu, noise_v)?;
        let v = DevBuf::new(&self.ops.gpu, n * 4)?;
        let steps = sigmas.len() - 1;
        for i in 0..steps {
            let (s0, s1) = (sigmas[i], sigmas[i + 1]);
            if s1 != s0 {
                self.velocity(st, &x, s0, cfg, &v)?;
                self.ops.axpy(x.ptr(), v.ptr(), n, s1 - s0)?;
            }
            self.ops.gpu.sync()?;
            prof::mark(&self.ops.nsd, "dit step")?;
            if i == 0 {
                self.note_memory();
            }
            progress(i as u32 + 1, steps as u32)?;
        }
        read_f32(&x, n)
    }

    /// Occupancy logits on the (res + 1)^3 lattice over [-BOUNDS, BOUNDS]^3 (x slowest), every point
    pub fn decode_dense(&self, latents: &[f32], res: usize, chunk: usize, progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let n = latents.len() / self.vae.channels;
        let tokens = self.vae.tokens(&self.ops, latents)?;
        let data = self.vae.data(&self.ops, &tokens, n)?;
        drop(tokens);
        let sc = self.vae.query_scratch(&self.ops, chunk)?;
        let step = 2.0 * BOUNDS / res as f32;
        self.vae.query_lattice(&self.ops, &data, &sc, res + 1, -BOUNDS, step, None, progress)
    }

    /// The same lattice decoded coarse to fine (`grid`): the whole of the coarsest level, then at each finer one
    /// only the points near the surface; `progress(level, levels)`. Also the points queried.
    pub fn decode_fast(&self, latents: &[f32], res: usize, chunk: usize, progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let n = latents.len() / self.vae.channels;
        let tokens = self.vae.tokens(&self.ops, latents)?;
        prof::mark(&self.ops.nsd, "vae transformer")?;
        let data = self.vae.data(&self.ops, &tokens, n)?;
        drop(tokens);
        let sc = self.vae.query_scratch(&self.ops, chunk)?;
        prof::mark(&self.ops.nsd, "vae data")?;
        let lv = grid::levels(res, 63);
        let mut g: Vec<f32> = Vec::new();
        let mut prev = 0;
        let mut queried = 0;
        for (k, &r) in lv.iter().enumerate() {
            let r1 = r + 1;
            let step = 2.0 * BOUNDS / r as f32;
            if k == 0 {
                g = self.vae.query_lattice(&self.ops, &data, &sc, r1, -BOUNDS, step, None, &mut |_, _| Ok(()))?;
                self.note_memory();
                queried += g.len();
                prof::mark(&self.ops.nsd, "decode coarse level")?;
            } else {
                let idx = grid::refine(&g, prev, r1, 0.0, k + 1 == lv.len());
                prof::mark(&self.ops.nsd, "decode refine (CPU)")?;
                let vals = self.vae.query_lattice(&self.ops, &data, &sc, r1, -BOUNDS, step, Some(&idx), &mut |_, _| Ok(()))?;
                queried += idx.len();
                prof::mark(&self.ops.nsd, "decode fine levels")?;
                let mut fine = grid::upsample(&g, prev, r1);
                for (i, v) in idx.iter().zip(vals) {
                    fine[*i as usize] = v;
                }
                g = fine;
                prof::mark(&self.ops.nsd, "decode upsample (CPU)")?;
            }
            if prof::on() {
                eprintln!("decode: level {k} ({r1}^3): {queried} points queried so far");
            }
            prev = r1;
            progress(k + 1, lv.len())?;
        }
        Ok((g, queried))
    }

    /// The mesh of a logit grid: marching cubes at 0, the floaters dropped
    pub fn mesh(logits: &[f32], res: usize) -> Mesh {
        let r1 = res + 1;
        let step = 2.0 * BOUNDS / res as f32;
        let m = nextsycl_mesh::tools::mc::marching_cubes(logits, [r1, r1, r1], 0.0, [-BOUNDS; 3], step);
        let _ = prof::mark_cpu("marching cubes (CPU)");
        let m = nextsycl_mesh::tools::mc::drop_floaters(&m, FLOATERS);
        let _ = prof::mark_cpu("floaters (CPU)");
        m
    }
}

/// One generation's working state
pub struct Run {
    dit: RunDit,
    pub tokens: usize,
}

enum RunDit {
    Flux(flux::Scratch),
    Plain(plain::Scratch),
}

impl Model3dEngine for Hunyuan3d {
    fn arch(&self) -> &'static str {
        self.arch.name
    }
    fn abilities(&self) -> Abilities {
        Abilities { generate: true, texture: self.paint.is_some(), views: 1, rig: false, animate: false }
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: self.arch.steps, cfg: self.arch.cfg, resolution: self.arch.resolution, faces: 40_000, texture_size: 2048, frames: 0, fps: 0.0 }
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }
    fn report(&self) -> Vec<String> {
        let mut r = vec![format!("{} on {}: {:.2} GiB of weights, up {:.0} s", self.arch.name, self.ops.gpu.name,
                                 self.weights_bytes as f64 / (1u64 << 30) as f64, self.loaded.elapsed().as_secs_f64())];
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        if let Ok((total, Some(free))) = self.ops.gpu.memory() {
            r.push(format!("card: {:.1} of {:.1} GiB free", gib(free), gib(total)));
            let low = self.low.load(std::sync::atomic::Ordering::Relaxed);
            if low != u64::MAX {
                r.push(format!("the last request: the card at most {:.1} GiB used ({:.1} GiB free at the lowest)", gib(total - low), gib(low)));
            }
        }
        r
    }
    fn generate(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Mesh> {
        let _one = self.busy.lock().map_err(|_| Error("a previous request failed midway".into()))?;
        self.low.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();
        prof::reset_clock(&self.ops.nsd)?;
        let r = self.generate_timed(req, progress, t0);
        prof::report(&mut |s| eprintln!("{s}"));
        r
    }
}

impl Hunyuan3d {
    fn generate_timed(&self, req: &MeshRequest, progress: &mut dyn FnMut(Step) -> Result<()>, t0: Instant) -> Result<Mesh> {
        let mut report = |phase: &'static str, at: u32, of: u32| progress(Step { phase, at, of, seconds: t0.elapsed().as_secs_f64() });
        let view = req.views.first().ok_or_else(|| Error("no picture".into()))?;
        if let Some(mesh) = &req.mesh {
            // a mesh given: painted as it is, no shape made
            let p = self.paint.as_ref().ok_or_else(|| Error(format!("{} has no paint model loaded (2.1 with its paint files): it cannot paint a mesh", self.arch.name)))?;
            let steps = self.setting(req, "paint-steps", "NS_M3D_HY_PAINT_STEPS").map(|v| v.parse::<usize>().map_err(|_| Error(format!("paint-steps: {v}?")))).transpose()?.unwrap_or(15);
            let size = req.texture_size.unwrap_or(2048).clamp(256, 8192);
            let given = nextsycl_model3d::finish(mesh.clone(), Some(req.faces.unwrap_or(40_000)));
            let painted = p.paint(&self.ops, &given, &view.image, req.seed, steps, size, None, &mut |ph, a, b| report(ph, a, b))?;
            self.note_memory();
            return Ok(nextsycl_model3d::finish(painted, None));
        }
        if req.texture && self.paint.is_none() {
            return Err(Error(format!("{} has no paint model loaded (2.1 with its paint files: paint-unet, paint-vae, paint-dino): ask for the shape alone",
                                     self.arch.name)));
        }
        let num = |name: &str, env: &str| -> Result<Option<usize>> {
            self.setting(req, name, env).map(|v| v.parse::<usize>().map_err(|_| Error(format!("{name}: {v}?")))).transpose()
        };
        let tokens = num("tokens", "NS_M3D_HY_TOKENS")?.unwrap_or(self.arch.tokens);
        let res = req.resolution.unwrap_or(self.arch.resolution) as usize;
        if !(16..=1024).contains(&res) || !(64..=16384).contains(&tokens) {
            return Err(Error(format!("resolution {res} (16 to 1024) or tokens {tokens} (64 to 16384) out of range")));
        }
        let steps = req.steps.unwrap_or(self.arch.steps) as usize;
        let cfg = req.cfg.unwrap_or(self.arch.cfg);
        report("image", 0, 1)?;
        let cut = req.remove_background.unwrap_or(true);
        // an opaque picture cut out by BiRefNet when it is loaded: its mask becomes the alpha the processor reads
        let cut_out;
        let image = match &self.remover {
            Some(r) if cut && view.image.rgba.chunks_exact(4).all(|p| p[3] == 255) => {
                let pic = nextsycl_diffusion::Picture { width: view.image.width, height: view.image.height, channels: 4, data: view.image.rgba.clone() };
                let c = r.cut_out(&pic).map_err(|e| Error(format!("the background remover: {}", e.0)))?;
                cut_out = nextsycl_model3d::Texture { width: c.width, height: c.height, rgba: c.data };
                prof::mark(&self.ops.nsd, "background remover")?;
                &cut_out
            }
            _ => &view.image,
        };
        let (pixels, _note) = picture::prepare(image, cut, self.arch.proc_size, self.arch.enc_size, BORDER)?;
        prof::mark(&self.ops.nsd, "picture (CPU)")?;
        let cond = self.condition(&pixels)?;
        self.ops.gpu.sync()?;
        prof::mark(&self.ops.nsd, "dino")?;
        report("image", 1, 1)?;
        let sigmas = if self.arch.name == ARCH_MINI { sched::consistency(steps, 100) } else { sched::flow_match(steps) };
        let mut st = self.run_state(&cond, tokens)?;
        prof::mark(&self.ops.nsd, "dit context")?;
        let latents = self.sample(&mut st, &noise(req.seed, tokens * self.vae.channels), &sigmas, cfg, &mut |a, b| report("shape", a, b))?;
        drop(st);
        drop(cond);
        let chunk = num("chunk", "NS_M3D_HY_CHUNK")?.unwrap_or_else(|| self.vae.chunk());
        let logits = match self.setting(req, "decode", "NS_M3D_HY_DECODE").as_deref() {
            Some("dense") => self.decode_dense(&latents, res, chunk, &mut |a, b| report("decode", (a * 100 / b.max(1)) as u32, 100))?,
            None | Some("fast") => self.decode_fast(&latents, res, chunk, &mut |a, b| report("decode", a as u32, b as u32))?.0,
            Some(x) => return Err(Error(format!("decode {x}: fast or dense"))),
        };
        // the mesh: marching cubes and the floaters (1 of 2), the decimation (2 of 2)
        report("mesh", 0, 2)?;
        let m = Self::mesh(&logits, res);
        if m.triangles() == 0 {
            return Err(Error("the shape decoded to nothing (no surface in the box)".into()));
        }
        report("mesh", 1, 2)?;
        let out = nextsycl_model3d::finish(m, Some(req.faces.unwrap_or(40_000)));
        prof::mark_cpu("finish (CPU: decimation)")?;
        report("mesh", 2, 2)?;
        match (&self.paint, req.texture) {
            (Some(p), true) => {
                let steps = num("paint-steps", "NS_M3D_HY_PAINT_STEPS")?.unwrap_or(15);
                let size = req.texture_size.unwrap_or(2048).clamp(256, 8192);
                let painted = p.paint(&self.ops, &out, &view.image, req.seed, steps, size, None, &mut |ph, a, b| report(ph, a, b))?;
                self.note_memory();
                let m = nextsycl_model3d::finish(painted, None);
                prof::mark_cpu("paint finish (CPU)")?;
                Ok(m)
            }
            _ => Ok(out),
        }
    }
}
