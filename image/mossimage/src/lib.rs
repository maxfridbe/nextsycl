//! MOSS-Image-SFT (OpenMOSS) in Rust and SYCL: a 5.1B text-to-image model. OpenMOSS's pipeline
//! (`pipeline_moss_image.py`, the whole model in one file) over the shared diffusion kernels and this engine's own
//! (kernels/image/mossimage); no Python, no ggml.
//!
//! ```text
//!   prompt -> Qwen3.5-9B's chat template -> the encoder's last hidden states of every token [L, 4096]
//!   noise [128, H/16, W/16] (seeded) -> Euler over the snapped shift-5 levels, the DiT's velocity each step, the
//!            guided prediction = uncond + cfg (cond - uncond), the uncond branch skipping block 22
//!   latents -> the FLUX.2 VAE (de-batchnormed, unpatchified to 32 channels) -> RGB
//! ```

pub mod check;
pub mod dit;
pub mod ffi;
pub mod part;
pub mod te;
pub mod vae;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_gguf::Gguf;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

pub const ARCH: &str = "moss-image";
/// the transformer GGUF, the text encoder GGUF (Qwen3.5-9B), the VAE
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];
/// the scheduler's shift and training steps (scheduler_config.json)
const SHIFT: f32 = 5.0;
const TRAIN_STEPS: usize = 1000;
/// the block the unconditional branch leaves out
pub const UNCOND_SKIP: &[usize] = &[22];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "MOSS-Image-SFT (5.1B DiT, Qwen3.5-9B text encoder, FLUX.2 VAE; Rust + SYCL)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(MossImage::load(files, gpus, log)?))
}

/// torch.linspace(start, end, n) in float32: from both ends, as torch does
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
}

/// The scheduler's grid: linspace(1, shift / (total + shift - 1), count), shifted
fn grid(count: usize, shift: f32) -> Vec<f32> {
    let lower = shift / (TRAIN_STEPS as f32 + shift - 1.0);
    linspace(1.0, lower, count).into_iter().map(|v| shift * v / (1.0 + (shift - 1.0) * v)).collect()
}

/// The `steps` levels: the shift-5 grid snapped to the nearest of the 1000 training levels, repeats dropped, then 0
/// (so a run has `len - 1` steps, fewer than asked when two levels snap together)
pub fn sigmas(steps: usize) -> Vec<f32> {
    let mut training = grid(TRAIN_STEPS, 1.0);
    training.reverse();
    let mut out: Vec<f32> = Vec::new();
    for s in grid(steps.max(1), SHIFT) {
        let upper = training.partition_point(|t| *t < s).min(TRAIN_STEPS - 1);
        let lower = upper.saturating_sub(1);
        let nearest = if (s - training[lower]).abs() <= (training[upper] - s).abs() { lower } else { upper };
        if out.last() != Some(&training[nearest]) {
            out.push(training[nearest]);
        }
    }
    out.push(0.0);
    out
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

pub struct MossImage {
    pub(crate) te: te::TextEncoder,
    pub(crate) nsd: Nsd,
    pub(crate) dit: dit::Dit,
    pub(crate) vae: vae::Vae,
    gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    /// the last prompts' conditions (the encoder's rows, float32 [L, 4096]); "" is the guidance branch's
    conds: Mutex<Vec<(String, Arc<dit::Prefix>)>>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f std::path::Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl MossImage {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<MossImage> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        if f.architecture() != "moss_image" {
            return Err(Error(format!("{ARCH}: the transformer file is a {} model, not moss_image", f.architecture())));
        }
        let te = te::TextEncoder::load(role(files, "text-encoder")?, &gpu, log)?;
        let nsd = Nsd::new(&gpu)?;
        let dit = dit::Dit::load(&f, &nsd, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        Ok(MossImage { te, nsd, dit, vae, gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()), conds: Mutex::new(Vec::new()) })
    }

    /// The prompt through the text encoder and the DiT's text embedder, the last few kept
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

    /// Latent tokens float32 [h * w, 128] from the starting noise `start` (same shape): `sig`'s Euler steps over the
    /// DiT after `pre`, and with guidance above 1 the prediction uncond + cfg (cond - uncond) against `npre` with the
    /// uncond branch's blocks left out
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(&self, pre: &dit::Prefix, npre: Option<&dit::Prefix>, cfg: f32, start: &[f32], hw: (usize, usize), sig: &[f32],
                         step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let n = hw.0 * hw.1 * dit::LATENT;
        let mods = self.dit.mods(&self.nsd, sig)?;
        let lat = DevBuf::from_f32(gpu, start)?;
        for i in 0..sig.len() - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            let v = self.dit.velocity(&self.nsd, pre, &lat, hw, &mods[i], &[], None)?;
            if let Some(np) = npre {
                let vu = self.dit.velocity(&self.nsd, np, &lat, hw, &mods[i], UNCOND_SKIP, None)?;
                // SAFETY: n floats each: v = vu + cfg (v - vu)
                ffi::check(unsafe { (k.cfg)(gpu.raw(), v.fp(), v.fp(), vu.fp(), n as i64, cfg) }, "guidance")?;
            }
            // SAFETY: n floats each.
            ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, sn - s) }, "euler step")?;
            gpu.sync()?;
            step(i)?;
        }
        lat.to_f32()
    }
}

impl ImageEngine for MossImage {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: 50, cfg: 3.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: SHIFT }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
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
        let (w, h) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if w < 64 || h < 64 || w > 4096 || h > 4096 || w % 16 != 0 || h % 16 != 0 {
            return Err(Error(format!("{w}x{h}: multiples of 16, from 64 to 4096 a side")));
        }
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.filter(|c| *c >= 0.0).unwrap_or(d.cfg);
        let sig = sigmas(steps);
        let of = (sig.len() - 1) as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let pre = self.condition(req.prompt.trim())?;
        // guidance: the unconditional branch is the empty prompt with block 22 left out
        let npre = if cfg != 1.0 { Some(self.condition("")?) } else { None };
        let hw = (h / 16, w / 16);
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let start = noise(req.seed + p as u64, dit::LATENT * hw.0 * hw.1);
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(&pre, npre.as_deref(), cfg, &start, hw, &sig, &mut step)?;
            let (rgb, pw, ph) = self.vae.decode(&self.nsd, &lat, hw.0, hw.1)?;
            out.push(Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb });
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} in Rust and SYCL on {} (loaded {:.0} s ago)", self.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_levels_end_at_zero_and_fall() {
        let s = super::sigmas(12);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        assert!(s.len() <= 13);
    }

    #[test]
    fn the_ends_of_a_linspace_are_exact() {
        let v = super::linspace(1.0, 0.25, 4);
        assert_eq!((v[0], v[3]), (1.0, 0.25));
    }
}
