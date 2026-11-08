//! Z-Image-Turbo (Tongyi-MAI) in Rust and SYCL: a 6B text-to-image model distilled to 8 steps without guidance.
//! diffusers' pipeline (pipelines/z_image/pipeline_z_image.py at 84532a05) over the shared diffusion kernels and this
//! engine's own (kernels/image/zimage); no Python, no ggml.
//!
//! ```text
//!   prompt -> Qwen3's chat template -> Qwen3-4B, the residual stream after layer 35 of every token [L, 2560]
//!   noise [16, H/8, W/8] (seeded) -> Euler over linspace(1, 1/n, n) shifted (shift 3), the DiT's velocity each step
//!   latents -> Flux.1's VAE (16 channels; / 0.3611 + 0.1159) -> RGB
//! ```
//!
//! The DiT is Ming-Image's transformer (Z-Image's own: dit.rs began as image/mingimage's), fed Qwen3-4B's states as
//! its caption. One card holds it all (the DiT made half 11.5 GiB, the encoder's 35 layers 6.6 GiB, the VAE, a
//! picture's activations); with two cards given the encoder goes to the second.

pub mod check;
pub mod dit;
pub mod ffi;
pub mod te;
pub mod vae;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_gguf::Gguf;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, SizePresets, Step};

pub const ARCH: &str = "z-image";
/// the transformer GGUF (ComfyUI's layout), the text encoder (Qwen3-4B, one safetensors file), the VAE (diffusers'
/// AutoencoderKL file), the tokenizer (Qwen3's tokenizer.json)
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae", "tokenizer"];
/// FlowMatchEulerDiscreteScheduler's shift (scheduler_config.json; static)
pub const SHIFT: f32 = 3.0;
/// the steps the Turbo is distilled for (8 NFE: the README's num_inference_steps=9 on the pipeline of its day, 8 on
/// today's, whose levels end at 1/n rather than 0)
pub const STEPS: u32 = 8;
/// a side's limits (multiples of 16, the pipeline's vae_scale_factor * 2)
pub const MIN_SIDE: usize = 256;
pub const MAX_SIDE: usize = 2048;
/// The aspect ratios and pixel budgets of the model's own demo (Tongyi-MAI/Z-Image-Turbo's Space, app.py RES_CHOICES):
/// 1024, 1280 and 1536 squared, 1:1 to 21:9 either way
const RATIOS: &[&str] = &["1:1", "9:7", "7:9", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21"];
const PIXELS: &[u32] = &[1_048_576, 1_638_400, 2_359_296];
/// The demo's sizes (width, height), as listed there
pub const SIZES: &[(u32, u32)] = &[
    (1024, 1024), (1152, 896), (896, 1152), (1152, 864), (864, 1152), (1248, 832), (832, 1248), (1280, 720), (720, 1280), (1344, 576), (576, 1344),
    (1280, 1280), (1440, 1120), (1120, 1440), (1472, 1104), (1104, 1472), (1536, 1024), (1024, 1536), (1536, 864), (864, 1536), (1680, 720), (720, 1680),
    (1536, 1536), (1728, 1344), (1344, 1728), (1728, 1296), (1296, 1728), (1872, 1248), (1248, 1872), (2048, 1152), (1152, 2048), (2016, 864), (864, 2016),
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Z-Image-Turbo (6B single-stream DiT, Qwen3-4B text encoder, Flux VAE; 8 steps; Rust + SYCL)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(ZImage::load(files, gpus, log)?))
}

/// torch.linspace(start, end, n) in float32: from both ends, as torch does
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
}

/// The `steps + 1` levels: get_default_z_image_sigmas (linspace(1, 1 / n, n)) through the scheduler's static shift
/// (s / (1 + (shift - 1) s) times shift, float32), then 0
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let n = steps.max(1);
    let mut out: Vec<f32> = linspace(1.0, 1.0 / n as f32, n).into_iter().map(|s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
    out.push(0.0);
    out
}

/// A size the model takes: multiples of 16, MIN_SIDE to MAX_SIDE a side
pub fn check_size(w: usize, h: usize) -> Result<()> {
    if !w.is_multiple_of(16) || !h.is_multiple_of(16) || w < MIN_SIDE || h < MIN_SIDE || w > MAX_SIDE || h > MAX_SIDE {
        return Err(Error(format!("{w}x{h}: multiples of 16, from {MIN_SIDE} to {MAX_SIDE} a side (the model's own sizes: 1024x1024, 1280x720, 1536x1024 ...)")));
    }
    Ok(())
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

pub struct ZImage {
    pub(crate) te: te::TextEncoder,
    pub(crate) nsd: Nsd,
    pub(crate) dit: dit::Dit,
    pub(crate) vae: vae::Vae,
    gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// the last prompts' captions after the context refiner
    conds: Mutex<Vec<(String, Arc<dit::Prefix>)>>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f std::path::Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

const GIB: f64 = (1u64 << 30) as f64;
/// the room every card keeps free (no out-of-memory error exists: past the card's memory the driver spills)
const MARGIN: u64 = 3 << 29;

impl ZImage {
    /// The DiT and the VAE on `gpus[0]`, the text encoder on `gpus[1]` (or beside them on one card)
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<ZImage> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let te_gpu = gpus.get(1).cloned().unwrap_or_else(|| gpu.clone());
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        if f.architecture() != "lumina2" {
            return Err(Error(format!("{ARCH}: the transformer file is a {} model, not Z-Image's DiT (lumina2)", f.architecture())));
        }
        // the plan: the DiT (float matrices made half: their size as stored or less), the encoder on its card, the VAE
        // and the activations of a 1024 x 1024 picture
        let dit_b: u64 = f.tensors.iter().map(|t| if t.ty.code() == 0 { t.bytes / 2 } else { t.bytes }).sum();
        let te_b = te::plan_bytes(role(files, "text-encoder")?)?;
        let work = dit::work_bytes(4096 + 512) + vae::Vae::work_bytes(1 << 20) + (1u64 << 28);
        let same = Arc::ptr_eq(&gpu, &te_gpu);
        let need = dit_b + work + if same { te_b } else { 0 };
        let free = |g: &Gpu| -> Result<u64> {
            let (total, free) = g.memory()?;
            Ok(free.unwrap_or(total))
        };
        let fr = free(&gpu)?;
        if need + MARGIN > fr {
            return Err(Error(format!("{ARCH}: needs about {:.1} GiB on {} (the DiT {:.1}{}, a 1024 x 1024 picture's work {:.1}) and 1.5 free; {:.1} GiB are free", need as f64 / GIB,
                                     gpu.name, dit_b as f64 / GIB, if same { format!(", the text encoder {:.1}", te_b as f64 / GIB) } else { String::new() },
                                     work as f64 / GIB, fr as f64 / GIB)));
        }
        if !same && te_b + (1u64 << 28) + MARGIN > free(&te_gpu)? {
            return Err(Error(format!("{ARCH}: the text encoder needs about {:.1} GiB on {}; {:.1} GiB are free", te_b as f64 / GIB, te_gpu.name, free(&te_gpu)? as f64 / GIB)));
        }
        let te = te::TextEncoder::load(role(files, "text-encoder")?, role(files, "tokenizer")?, &te_gpu, log)?;
        let nsd = Nsd::new(&gpu)?;
        let dit = dit::Dit::load(&f, &nsd, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        Ok(ZImage { te, nsd, dit, vae, gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()), conds: Mutex::new(Vec::new()) })
    }

    /// The prompt through the text encoder and the context refiner, the last few kept
    pub(crate) fn condition(&self, prompt: &str) -> Result<Arc<dit::Prefix>> {
        if let Some((_, c)) = self.conds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(c.clone());
        }
        let c = Arc::new(self.dit.prefix(&self.nsd, &self.te.encode(prompt)?, None)?);
        let mut list = self.conds.lock().unwrap();
        list.push((prompt.to_string(), c.clone()));
        if list.len() > 8 {
            list.remove(0);
        }
        Ok(c)
    }

    /// Refuses a picture whose activations would not fit beside the weights (the card's free memory less the margin)
    fn check_room(&self, w: usize, h: usize, caption: usize) -> Result<()> {
        let rows = dit::padded(w / 16 * (h / 16)) + dit::padded(caption.max(1));
        let need = dit::work_bytes(rows) + vae::Vae::work_bytes(w * h) + (1u64 << 28);
        let (total, free) = self.gpu.memory()?;
        let free = free.unwrap_or(total);
        if need + MARGIN > free {
            return Err(Error(format!("{w}x{h}: its work needs about {:.1} GiB on {}; {:.1} GiB are free beside the model (a smaller size fits)", need as f64 / GIB,
                                     self.gpu.name, free as f64 / GIB)));
        }
        Ok(())
    }

    /// Latents float32 [16, lh, lw] from the starting noise `start` (same shape): `sig`'s Euler steps over the DiT after
    /// `pre`
    pub(crate) fn sample(&self, pre: &dit::Prefix, start: &[f32], lh: usize, lw: usize, sig: &[f32], step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let hw = (lh / 2, lw / 2);
        let n = hw.0 * hw.1 * dit::PATCH_IN;
        let mods = self.dit.mods(&self.nsd, sig)?;
        let lat = DevBuf::from_f32(gpu, &dit::patchify(start, lh, lw))?;
        for i in 0..sig.len() - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let v = self.dit.velocity(&self.nsd, pre, &lat, hw, &mods[i], None)?;
            // the scheduler's step on the pipeline's minus the model's output
            // SAFETY: n floats each.
            ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, -(sn - s)) }, "euler step")?;
            gpu.sync()?;
            step(i)?;
        }
        Ok(dit::unpatchify(&lat.to_f32()?, lh, lw))
    }
}

impl ImageEngine for ZImage {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: STEPS, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: SHIFT }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// the Turbo is guidance-distilled: no negative prompt, no guidance scale
    fn guidance(&self) -> bool {
        false
    }
    fn size_presets(&self) -> Option<SizePresets> {
        Some(SizePresets { ratios: RATIOS, pixels: PIXELS })
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: text to image only")));
        }
        // guidance 0 is the pipeline's "none" (its scale multiplies pos - neg), 1 the contract's
        if req.negative.as_deref().is_some_and(|n| !n.trim().is_empty()) || req.cfg.is_some_and(|c| c != 0.0 && c != 1.0) {
            return Err(Error(format!("{ARCH}: no guidance (no negative prompt, cfg 1 or 0): the Turbo is distilled to run without it")));
        }
        let (w, h) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        check_size(w, h)?;
        let steps = req.steps.unwrap_or(d.steps) as usize;
        if !(1..=50).contains(&steps) {
            return Err(Error(format!("{steps} steps: from 1 to 50 (the Turbo's own: {STEPS})")));
        }
        let shift = req.shift.unwrap_or(SHIFT);
        if !(shift.is_finite() && shift > 0.0 && shift <= 20.0) {
            return Err(Error(format!("shift {shift}: above 0, at most 20 (the model's: {SHIFT})")));
        }
        let prompt = req.prompt.trim();
        if prompt.is_empty() {
            return Err(Error(format!("{ARCH}: an empty prompt")));
        }
        self.check_room(w, h, self.te.ids(prompt).len())?;
        let sig = sigmas(steps, shift);
        let of = steps as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let pre = self.condition(prompt)?;
        let (lh, lw) = (h / 8, w / 8);
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let start = noise(req.seed + p as u64, dit::LATENT * lh * lw);
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(&pre, &start, lh, lw, &sig, &mut step)?;
            let (rgb, pw, ph) = self.vae.decode(&self.nsd, &lat, lh, lw)?;
            out.push(Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb });
        }
        Ok(out)
    }

    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and SYCL: the DiT and VAE on {}, the text encoder on {} (loaded {:.0} s ago)", self.gpu.name, self.te.gpu.name,
                     self.loaded.elapsed().as_secs_f64())]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_levels_are_the_pipelines() {
        // diffusers 84532a05: retrieve_timesteps(FlowMatchEulerDiscreteScheduler(shift 3), sigmas=get_default_z_image_sigmas(8)),
        // scheduler.sigmas as the reference printed them
        let want = [1.0, 0.95454544, 0.9, 0.8333333, 0.75, 0.64285713, 0.5, 0.3, 0.0];
        let s = super::sigmas(8, super::SHIFT);
        assert_eq!(s.len(), 9);
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{s:?}");
        }
        // one step: from pure noise straight to 0
        assert_eq!(super::sigmas(1, 3.0), vec![1.0, 0.0]);
    }

    #[test]
    fn sizes_are_multiples_of_16_in_range() {
        assert!(super::check_size(1024, 1024).is_ok());
        assert!(super::check_size(1280, 720).is_ok());
        assert!(super::check_size(2048, 1152).is_ok());
        assert!(super::check_size(1000, 1000).is_err());
        assert!(super::check_size(128, 1024).is_err());
        assert!(super::check_size(4096, 1024).is_err());
        // every size of the model's demo is taken
        assert!(super::SIZES.iter().all(|(w, h)| super::check_size(*w as usize, *h as usize).is_ok()));
        assert_eq!(super::SIZES.len(), 33);
    }

    #[test]
    fn a_linspace_ends_where_asked() {
        let v = super::linspace(1.0, 0.125, 8);
        assert_eq!((v[0], v[7]), (1.0, 0.125));
        assert!((v[4] - 0.5).abs() < 1e-7);
    }
}
