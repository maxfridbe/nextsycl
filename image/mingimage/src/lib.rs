//! Ming-Image-0.1-Design (inclusionAI) in Rust and SYCL: a 6B text-to-image model for UI mock-ups, infographics,
//! posters and other text-heavy designs, with transparent (RGBA) output. inclusionAI's pipeline
//! (github.com/inclusionAI/Ming-Image, diffusion/ and modeling_bailingmm2.py) over the shared diffusion kernels and
//! this engine's own (kernels/image/mingimage); no Python, no ggml.
//!
//! ```text
//!   prompt -> the chat template + 256 query tokens -> Ling-Mini-2.0 (MoE) -> the connector -> cap [256, 2560]
//!                                                                         \-> layers 5, 12, 20 -> direct [L, 3840]
//!   noise [16, H/8, W/8] (seeded) -> Euler over FlowMatchEuler's levels (shift 6), the DiT's velocity each step
//!   latents -> the VAE (Wan 2.1's decoder, 4 channels) -> RGBA
//! ```
//!
//! Two cards: the DiT and the VAE on the first (7.3 GB and their activations), the text encoder on the second (9.3
//! GB), every weight resident in its GGUF's type - nothing streamed. Sizes: the pipeline's buckets (1024, 2048) at
//! the closest of their aspect ratios, the picture resized to the size asked (Lanczos) as infer.py does.

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
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

pub const ARCH: &str = "ming-image";
/// the transformer GGUF, the text encoder GGUF (Ling-Mini-2.0 with the connector and the projections), the VAE
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae"];
/// The phrase put before the prompt for a transparent picture (one of the README's fixed RGBA phrases)
pub const RGBA_PHRASE: &str = "RGBA, 4-channel, transparent background";
/// The README's fixed RGBA phrases: a prompt that starts with one keeps it as it is
pub const RGBA_PHRASES: &[&str] = &["带透明通道，4通道RGBA图像", "透明背景，alpha通道，无底图", "抠图素材，背景alpha=0", "孤立主体，透明PNG图层", "不要白底，不要棋盘格，只要透明通道",
    "RGBA, 4-channel, transparent background", "isolated subject, alpha matte, no background", "cutout PNG, alpha=0 outside the object",
    "transparent canvas, not white, not checkerboard", "production RGBA layer for compositing"];
/// FlowMatchEulerDiscreteScheduler's shift (scheduler_config.json: static - the pipeline's dynamic-shift switch is set
/// on a frozen config and never reaches the scheduler)
const SHIFT: f64 = 6.0;

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Ming-Image 0.1 Design (Z-Image's 6B DiT, Ling-Mini-2.0 MoE text encoder, RGBA VAE; Rust + SYCL)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(MingImage::load(files, gpus, log)?))
}

/// The `steps + 1` noise levels: linspace(1, 0, steps) shifted (shift s / (1 + (shift - 1) s)), then 0 - the last
/// step runs from 0 to 0 (a no-op the sampler skips)
pub fn sigmas(steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    let mut out: Vec<f32> = (0..n).map(|i| {
        let s = if n == 1 { 1.0 } else { 1.0 - i as f64 / (n - 1) as f64 };
        (SHIFT * s / (1.0 + (SHIFT - 1.0) * s)) as f32
    }).collect();
    out.push(0.0);
    out
}

/// bailingmm_utils.process_ratio's tables: (height / width, height, width) of the 1024 and 2048 buckets
const R1024: &[(f64, usize, usize)] = &[(0.25, 512, 2048), (0.26, 512, 1984), (0.27, 512, 1920), (0.28, 512, 1856), (0.32, 576, 1792), (0.33, 576, 1728), (0.35, 576, 1664), (0.4, 640, 1600), (0.42, 640, 1536), (0.48, 704, 1472), (0.5, 704, 1408), (0.52, 704, 1344), (0.5581, 768, 1376), (0.5625, 720, 1280), (0.5647, 768, 1360), (0.57, 768, 1344), (0.6, 768, 1280), (0.622, 816, 1312), (0.625, 800, 1280), (0.65, 832, 1280), (0.6582, 832, 1264), (0.6667, 832, 1248), (0.6709, 848, 1264), (0.68, 832, 1216), (0.7013, 864, 1232), (0.72, 832, 1152), (0.7467, 896, 1200), (0.75, 864, 1152), (0.7568, 896, 1184), (0.78, 896, 1152), (0.8, 896, 1120), (0.8056, 928, 1152), (0.82, 896, 1088), (0.88, 960, 1088), (0.94, 960, 1024), (0.9846, 1024, 1040), (1.0, 1024, 1024), (1.07, 1024, 960), (1.13, 1088, 960), (1.21, 1088, 896), (1.2414, 1152, 928), (1.25, 1120, 896), (1.2807, 1168, 912), (1.29, 1152, 896), (1.3333, 1152, 864), (1.3393, 1200, 896), (1.38, 1152, 832), (1.46, 1216, 832), (1.4906, 1264, 848), (1.5, 1248, 832), (1.6, 1280, 800), (1.67, 1280, 768), (1.75, 1344, 768), (1.7708, 1360, 768), (1.7778, 1280, 720), (2.0, 1408, 704), (2.09, 1472, 704), (2.4, 1536, 640), (2.5, 1600, 640), (2.89, 1664, 576), (3.0, 1728, 576), (3.11, 1792, 576), (3.62, 1856, 512), (3.75, 1920, 512), (3.88, 1984, 512), (4.0, 2048, 512)];
const R2048: &[(f64, usize, usize)] = &[(0.25, 1024, 4096), (0.26, 1024, 3968), (0.27, 1024, 3840), (0.28, 1024, 3712), (0.32, 1152, 3584), (0.33, 1152, 3456), (0.35, 1152, 3328), (0.4, 1280, 3200), (0.42, 1280, 3072), (0.48, 1408, 2944), (0.5, 1408, 2816), (0.52, 1408, 2688), (0.5625, 1440, 2560), (0.57, 1536, 2688), (0.6, 1536, 2560), (0.6667, 1664, 2496), (0.68, 1664, 2432), (0.72, 1664, 2304), (0.75, 1824, 2432), (0.78, 1792, 2304), (0.7917, 1824, 2304), (0.8, 1792, 2240), (0.82, 1792, 2176), (0.88, 1920, 2176), (0.94, 1920, 2048), (1.0, 2048, 2048), (1.07, 2048, 1920), (1.13, 2176, 1920), (1.21, 2176, 1792), (1.25, 2240, 1792), (1.2632, 2304, 1824), (1.29, 2304, 1792), (1.3333, 2432, 1824), (1.38, 2304, 1664), (1.46, 2432, 1664), (1.5, 2496, 1664), (1.67, 2560, 1536), (1.75, 2688, 1536), (1.7778, 2560, 1440), (2.0, 2816, 1408), (2.09, 2944, 1408), (2.4, 3072, 1280), (2.5, 3200, 1280), (2.89, 3328, 1152), (3.0, 3456, 1152), (3.11, 3584, 1152), (3.62, 3712, 1024), (3.75, 3840, 1024), (3.88, 3968, 1024), (4.0, 4096, 1024), (0.2941, 1120, 3808), (0.3043, 1120, 3680), (0.3679, 1248, 3392), (0.3846, 1280, 3328), (0.433, 1344, 3104), (0.4574, 1376, 3008), (0.4681, 1408, 3008), (0.5465, 1504, 2752), (0.6296, 1632, 2592), (0.6582, 1664, 2528), (0.7432, 1760, 2368), (0.8551, 1888, 2208), (0.9692, 2016, 2080), (1.0317, 2080, 2016), (1.1695, 2208, 1888), (1.3455, 2368, 1760), (1.5192, 2528, 1664), (1.5882, 2592, 1632), (1.8298, 2752, 1504), (1.913, 2816, 1472), (2.1364, 3008, 1408), (2.186, 3008, 1376), (2.3095, 3104, 1344), (2.6, 3328, 1280), (2.7179, 3392, 1248), (3.2857, 3680, 1120), (3.4, 3808, 1120)];

/// The size the model makes a picture of `w` x `h` at: the bucket nearest sqrt(w h) (1024 or 2048, the smaller on a
/// tie), its entry of the nearest aspect (the first on a tie) - (width, height)
pub fn diffusion_size(w: usize, h: usize) -> (usize, usize) {
    let side = ((w * h) as f64).sqrt();
    let table = if (side - 2048.0).abs() < (side - 1024.0).abs() { R2048 } else { R1024 };
    let r = h as f64 / w as f64;
    let mut best = table[0];
    for e in table {
        if (e.0 - r).abs() < (best.0 - r).abs() {
            best = *e;
        }
    }
    (best.2, best.1)
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

pub struct MingImage {
    pub(crate) te: te::TextEncoder,
    pub(crate) nsd: Nsd,
    pub(crate) dit: dit::Dit,
    pub(crate) vae: vae::Vae,
    gpu: Arc<Gpu>,
    loaded: Instant,
    load_s: f64,
    busy: Mutex<()>,
    conds: Mutex<Vec<(String, Arc<te::Condition>)>>,
}

fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f std::path::Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

impl MingImage {
    /// The DiT and the VAE on `gpus[0]`, the text encoder on `gpus[1]` (or beside them on one card large enough)
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], log: &mut dyn FnMut(String)) -> Result<MingImage> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?.clone();
        let te_gpu = gpus.get(1).cloned().unwrap_or_else(|| gpu.clone());
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let dit_b: u64 = f.tensors.iter().map(|t| t.bytes).sum();
        let (total, free) = gpu.memory()?;
        // the transformer as stored, its float32 modulation linears, the VAE and ~2.5 GiB of activations at 1024 x 1024
        let need = dit_b + (1u64 << 29) + (1u64 << 28) + (5u64 << 29);
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH}: the DiT needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / (1u64 << 30) as f64, gpu.name,
                                      free as f64 / (1u64 << 30) as f64)));
        }
        // the text encoder first on its card: on one card, a full one says so before the DiT is read
        let te = te::TextEncoder::load(role(files, "text-encoder")?, &te_gpu, log)?;
        let nsd = Nsd::new(&gpu)?;
        let dit = dit::Dit::load(&f, &nsd, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, log)?;
        Ok(MingImage { te, nsd, dit, vae, gpu, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64(), busy: Mutex::new(()), conds: Mutex::new(Vec::new()) })
    }

    /// The prompt's condition, the last few kept
    fn condition(&self, prompt: &str) -> Result<Arc<te::Condition>> {
        if let Some((_, c)) = self.conds.lock().unwrap().iter().find(|(p, _)| p == prompt) {
            return Ok(c.clone());
        }
        let c = Arc::new(self.te.encode(prompt)?);
        let mut list = self.conds.lock().unwrap();
        list.push((prompt.to_string(), c.clone()));
        if list.len() > 8 {
            list.remove(0);
        }
        Ok(c)
    }

    /// Latents float32 [16, h, w] from the starting noise `start` (same shape): `sig`'s Euler steps over the DiT
    /// after `pre` (and, with guidance above 1, pos + cfg (pos - neg) against the zero condition `npre`)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(&self, pre: &dit::Prefix, npre: Option<&dit::Prefix>, cfg: f32, start: &[f32], lh: usize, lw: usize, sig: &[f32],
                         step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let k = ffi::api()?;
        let gpu = &self.gpu;
        let hw = (lh / 2, lw / 2);
        let n = hw.0 * hw.1 * dit::PATCH_IN;
        let mods = self.dit.mods(&self.nsd, sig)?;
        let lat = DevBuf::from_f32(gpu, &dit::patchify(start, lh, lw))?;
        for i in 0..sig.len() - 1 {
            let (s, sn) = (sig[i], sig[i + 1]);
            if sn != s {
                let v = self.dit.velocity(&self.nsd, pre, &lat, hw, &mods[i], None)?;
                if let Some(np) = npre {
                    let vn = self.dit.velocity(&self.nsd, np, &lat, hw, &mods[i], None)?;
                    // SAFETY: n floats each: vn = vn - v, then v += -cfg vn (v + cfg (v - vn))
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), vn.fp(), v.fp(), n as i64, -1.0) }, "guidance")?;
                    ffi::check(unsafe { (k.axpy)(gpu.raw(), v.fp(), vn.fp(), n as i64, -cfg) }, "guidance")?;
                }
                // the scheduler's step on the pipeline's minus the model's output
                // SAFETY: n floats each.
                ffi::check(unsafe { (k.axpy)(gpu.raw(), lat.fp(), v.fp(), n as i64, -(sn - s)) }, "euler step")?;
                gpu.sync()?;
            }
            step(i)?;
        }
        Ok(dit::unpatchify(&lat.to_f32()?, lh, lw))
    }
}

impl ImageEngine for MingImage {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: 12, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: SHIFT as f32 }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// above 1: the pipeline's guidance against the zero condition (it has no negative prompt)
    fn guidance(&self) -> bool {
        true
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }

    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let _one = self.busy.lock().unwrap();
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: text to image only (its edit model is Ming-Image-0.1-Design-Layer's)")));
        }
        let (rw, rh) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if rw < 64 || rh < 64 || rw > 4096 || rh > 4096 {
            return Err(Error(format!("{rw}x{rh}: from 64 to 4096 a side")));
        }
        let (dw, dh) = diffusion_size(rw, rh);
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.filter(|c| *c > 0.0).unwrap_or(d.cfg);
        let prompt = req.prompt.trim();
        let text = if req.rgba && !RGBA_PHRASES.iter().any(|p| prompt.starts_with(p)) { format!("{RGBA_PHRASE}, {prompt}") } else { prompt.to_string() };
        let sig = sigmas(steps);
        let of = steps as u32;
        progress(Step { picture: 0, at: 0, of, seconds: t0.elapsed().as_secs_f64() })?;
        let cond = self.condition(&text)?;
        let pre = self.dit.prefix(&self.nsd, &cond.cap, &cond.direct, cond.direct_rows, None)?;
        let npre = if cfg > 1.0 {
            Some(self.dit.prefix(&self.nsd, &vec![0f32; cond.cap.len()], &vec![0f32; cond.direct.len()], cond.direct_rows, None)?)
        } else {
            None
        };
        let (lh, lw) = (dh / 8, dw / 8);
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let start = noise(req.seed + p as u64, dit::LATENT * lh * lw);
            let mut step = |i: usize| progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() });
            let lat = self.sample(&pre, npre.as_ref(), cfg, &start, lh, lw, &sig, &mut step)?;
            let (rgba, pw, ph) = self.vae.decode(&self.nsd, &lat, lh, lw)?;
            let mut pic = Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgba };
            if (pw, ph) != (rw, rh) {
                pic = pic.resize_lanczos(rw as u32, rh as u32);
            }
            if !req.rgba {
                // an opaque picture: over white where the model left any transparency
                let data = pic.data.chunks_exact(4).flat_map(|c| {
                    let a = c[3] as f32 / 255.0;
                    [0, 1, 2].map(|i| (c[i] as f32 * a + 255.0 * (1.0 - a)).round() as u8)
                }).collect();
                pic = Picture { width: pic.width, height: pic.height, channels: 3, data };
            }
            out.push(pic);
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
    fn the_levels_are_the_schedulers() {
        // diffusers' FlowMatchEulerDiscreteScheduler(shift 6) with sigma_min 0, 12 steps (printed from the reference)
        let want = [1.0, 0.98361, 0.96429, 0.94118, 0.91304, 0.87805, 0.83333, 0.77419, 0.69231, 0.57143, 0.375, 0.0, 0.0];
        let s = super::sigmas(12);
        assert_eq!(s.len(), 13);
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{s:?}");
        }
    }

    #[test]
    fn sizes_take_the_pipelines_buckets() {
        assert_eq!(super::diffusion_size(1024, 1024), (1024, 1024));
        assert_eq!(super::diffusion_size(2048, 2048), (2048, 2048));
        // 16:9 at 1024: the table's 1.7778 entry is height / width; 0.5625 is 720 x 1280
        assert_eq!(super::diffusion_size(1280, 720), (1280, 720));
        assert_eq!(super::diffusion_size(1536, 1536), (1024, 1024), "a tie goes to the smaller bucket");
        assert_eq!(super::diffusion_size(1600, 1600), (2048, 2048));
    }
}
