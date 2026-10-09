//! Qwen-Image 2.1 on Intel Arc (SYCL): text to image, the whole pipeline on one GPU.
//!
//! ```text
//!   prompt -> chat template -> Qwen3-VL 8B text tower (int8 ConvRot, nextsycl-qwen3vl) -> last hidden state, the
//!     system turn's tokens dropped
//!   noise [h/16 * w/16, 64] (seeded) -> Euler over the flow-matching sigmas (sched.rs), the DiT's velocity each step
//!     (dit.rs: the text's keys / values computed once, then image rows only)
//!   latents -> VAE decoder (vae.rs) -> RGBA
//! ```
//!
//! The weights are the files the catalog lists (`nextsycl models pull qwen-image-2.1-q8`): the transformer as a GGUF
//! (Q8_0 / BF16 dequantized to half at load), the text encoder as ComfyUI's int8 ConvRot safetensors, the VAE in
//! bf16, the tokenizer's tokenizer.json. `check` compares each stage with the reference's dumps
//! (reference/qwenimage21/ref.py, run on the same quantized files).

pub mod check;
pub mod dit;
mod ffi;
pub mod prof;
pub mod sched;
pub mod vae;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_gguf::Gguf;
use nextsycl_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
use nextsycl_qwen3vl::TextEncoder;

pub const ARCH: &str = "qwen-image-2.1";
pub const ROLES: &[&str] = &["transformer", "text-encoder", "vae", "tokenizer"];

const GIB: f64 = (1u64 << 30) as f64;

/// The prompt's chat turn; the system turn's tokens are dropped from the encoding
const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

/// The VAE's latent statistics (diffusers' vae/config.json latents_mean / latents_std), per channel
#[allow(clippy::approx_constant)] // measured statistics, not -pi / 6
const LATENT_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
const LATENT_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864,
    4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947,
    5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643,
    4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
];

/// The options it takes (`--opt-NAME`: at load the variable named; per request a field of the request's `options`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_QI_INT8", value: "0|1", help: "the DiT's block matrices in int8 ConvRot (~1.4x a step, half the VRAM)", at: At::Load },
    EngineOption { name: "sigmas", env: "NS_QI_SIGMAS", value: "1,X,...", help: "the noise levels from 1 down (a few-step model's own; its step count)", at: At::Both },
    EngineOption { name: "sigma-shift", env: "NS_QI_SIGMA_SHIFT", value: "dynamic|none", help: "the size's shift on those levels, or as given", at: At::Both },
];

/// This engine's registry entry
pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Qwen-Image 2.1 (7B single-stream DiT, Qwen3-VL text encoder, RGBA VAE)", roles: ROLES, load, options: OPTIONS }
}

/// A sigma preset from its two settings (each `None`: not given)
fn preset_of(sigmas: Option<String>, shift: Option<String>) -> Result<Option<Preset>> {
    let Some(v) = sigmas else { return Ok(None) };
    let bad = || Error(format!("sigmas {v}: numbers from 1 down, comma separated"));
    let nodes: Vec<f64> = v.split(',').map(|x| x.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().map_err(|_| bad())?;
    if nodes.is_empty() || nodes.windows(2).any(|w| w[0] <= w[1]) || nodes[0] > 1.0 || *nodes.last().unwrap() <= 0.0 {
        return Err(bad());
    }
    let dynamic = match shift.as_deref() {
        None | Some("dynamic") => true,
        Some("none") => false,
        Some(x) => return Err(Error(format!("sigma-shift {x}: dynamic or none"))),
    };
    Ok(Some(Preset { nodes, dynamic }))
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(QwenImage21::load(files, gpu, o, log)?))
}

/// A few-step model's (or LoRA's) sigma preset: `NS_QI_SIGMAS` (its nodes, from 1 down; 0 is appended) and
/// `NS_QI_SIGMA_SHIFT` (`dynamic`: the size's exponential shift on them, as diffusers' pipeline does; `none`: as given)
#[derive(Clone, Debug)]
pub struct Preset {
    pub nodes: Vec<f64>,
    pub dynamic: bool,
}

pub struct QwenImage21 {
    pub nsd: Nsd,
    pub preset: Option<Preset>,
    pub te: TextEncoder,
    pub dit: dit::Dit,
    pub vae: vae::Vae,
    loaded: Instant,
    load_s: f64,
}

/// A role's file, or why not
fn role<'f>(files: &'f ModelFiles, r: &str) -> Result<&'f Path> {
    files.get(r).map(|p| p.as_path()).ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
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

impl QwenImage21 {
    /// The model on `gpu`, `o.merge_loras` merged into its DiT, its settings from `o` (`NS_QI_INT8`, the sigma preset)
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<QwenImage21> {
        let loras = &o.merge_loras;
        let preset = preset_of(o.setting("NS_QI_SIGMAS"), o.setting("NS_QI_SIGMA_SHIFT"))?;
        if let Some(p) = &preset {
            log(format!("sigmas: a {}-step preset ({})", p.nodes.len(), if p.dynamic { "the size's shift" } else { "as given" }));
        }
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        // the xe driver has no out-of-memory error - an allocation past the card spills to host RAM and can take the
        // machine down - so the plan is checked first: the DiT in half (int8: a byte a weight), the encoder and VAE as stored, the
        // activations of a 1024 x 1024 picture, 1.5 GiB kept free
        let f = Gguf::open(role(files, "transformer")?).map_err(|e| Error(e.0))?;
        let size = |r: &str| -> Result<u64> { std::fs::metadata(role(files, r)?).map(|m| m.len()).map_err(|e| Error(format!("{r}: {e}"))) };
        let int8 = o.setting("NS_QI_INT8").is_some_and(|v| v == "1");
        let dit_b: u64 = f.tensors.iter().map(|t| t.elements() * if int8 { 1 } else { 2 }).sum();
        let need = dit_b + size("text-encoder")? + size("vae")? + (5u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / GIB, gpu.name, free as f64 / GIB)));
        }
        let nsd = Nsd::new(gpu)?;
        let te = TextEncoder::load(role(files, "text-encoder")?, role(files, "tokenizer")?, gpu, log)?;
        let mut deltas = Vec::new();
        for l in loras {
            let d = nextsycl_diffusion::lora::read(&l.path, l.scale).map_err(Error)?;
            log(format!("LoRA {} x {}: {} matrices, rank {}", l.name, l.scale, d.len(), d.first().map_or(0, |x| x.r)));
            deltas.extend(d);
        }
        let dit = dit::Dit::load(&f, &nsd, int8, &deltas, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, (LATENT_MEAN.to_vec(), LATENT_STD.to_vec()), log)?;
        Ok(QwenImage21 { nsd, preset, te, dit, vae, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64() })
    }

    /// The prompt's tokens (the whole chat turn) and how many lead the turn (the system's, dropped)
    pub fn tokens(&self, prompt: &str) -> (Vec<u32>, usize) {
        let ids = self.te.tokenize(&format!("{SYSTEM}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
        let drop = self.te.tokenize(SYSTEM).len();
        (ids, drop)
    }

    /// The text's embedding for the DiT: float32 [T, 4096] on the GPU, T tokens after the system turn
    pub fn encode(&self, prompt: &str) -> Result<(DevBuf, usize)> {
        let (ids, drop) = self.tokens(prompt);
        let all = self.te.encode_ids(&self.nsd, &ids)?;
        let t = ids.len() - drop;
        let d = self.te.hidden;
        let out = DevBuf::f32(&self.nsd.gpu, t * d)?;
        out.copy_within(0, &all, drop * d * 4, t * d * 4)?;
        // `all` is freed on return: the copy first
        self.nsd.gpu.sync()?;
        Ok((out, t))
    }

    /// Denoise `lat` (float32 [h * w, 64], the noise, in place) over `sigmas` after the text `embeds`
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(&self, embeds: &DevBuf, tokens: usize, lat: &DevBuf, hw: (usize, usize), sigmas: &[f32], each: &mut dyn FnMut(usize, &DevBuf) -> Result<()>)
                   -> Result<()> {
        let k = ffi::api()?;
        let pre = self.dit.prefix(&self.nsd, embeds, tokens, self.te.hidden)?;
        let n = hw.0 * hw.1 * dit::LATENT;
        for i in 0..sigmas.len() - 1 {
            prof::reset_clock(&self.nsd)?;
            let v = self.dit.velocity(&self.nsd, &pre, lat, hw, sigmas[i], None)?;
            // SAFETY: n floats each.
            ffi::check(unsafe { (k.axpy)(self.nsd.gpu.raw(), lat.fp(), v.fp(), n as i64, sigmas[i + 1] - sigmas[i]) }, "euler step")?;
            self.nsd.gpu.sync()?;
            each(i, lat)?;
        }
        Ok(())
    }
}

impl ImageEngine for QwenImage21 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        let steps = self.preset.as_ref().map_or(40, |p| p.nodes.len() as u32);
        Defaults { width: 1024, height: 1024, steps, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    /// distilled to run without it
    fn guidance(&self) -> bool {
        false
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn steps_for(&self, req: &ImageRequest) -> u32 {
        match (req.steps, req.extra.get("sigmas")) {
            (Some(s), _) => s,
            (None, Some(v)) => v.split(',').count() as u32,
            (None, None) => self.defaults().steps,
        }
    }
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step)) -> Result<Vec<Picture>> {
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: edits are not ported yet")));
        }
        if req.cfg.is_some_and(|c| c != 1.0) || req.negative.is_some() {
            return Err(Error(format!("{ARCH}: guidance (cfg, a negative prompt) is not ported yet: the model is distilled to run without it")));
        }
        if req.sampler.is_some_and(|s| s != Sampler::Euler) {
            return Err(Error(format!("{ARCH}: only the euler sampler so far")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{ARCH}: LoRAs are not ported yet")));
        }
        let (w, h) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if w % 16 != 0 || h % 16 != 0 {
            return Err(Error(format!("{w}x{h}: the sides are multiples of 16")));
        }
        let steps = req.steps.unwrap_or(d.steps) as usize;
        let hw = (h / 16, w / 16);
        // a request's own sigmas win over the model's preset
        let mine = preset_of(req.extra.get("sigmas").cloned(), req.extra.get("sigma-shift").cloned())?;
        let preset = mine.as_ref().or(self.preset.as_ref());
        let steps = match (&mine, req.steps) {
            (Some(p), None) => p.nodes.len(),
            _ => steps,
        };
        let sig = match preset {
            Some(p) if steps == p.nodes.len() => sched::preset(&p.nodes, hw.0 * hw.1, p.dynamic),
            Some(p) => {
                return Err(Error(format!("this model runs its own {} sigmas (a few-step distill): {steps} steps is not one of its schedules",
                                         p.nodes.len())))
            }
            None => sched::sigmas(steps, hw.0 * hw.1),
        };
        let (embeds, tokens) = self.encode(&req.prompt)?;
        progress(Step { picture: 0, at: 0, of: steps as u32, seconds: t0.elapsed().as_secs_f64() });
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let lat = DevBuf::from_f32(&self.nsd.gpu, &noise(req.seed + p as u64, hw.0 * hw.1 * dit::LATENT))?;
            self.denoise(&embeds, tokens, &lat, hw, &sig, &mut |i, _| {
                progress(Step { picture: p, at: i as u32 + 1, of: steps as u32, seconds: t0.elapsed().as_secs_f64() });
                Ok(())
            })?;
            let tv = Instant::now();
            let (rgba, pw, ph) = self.vae.decode(&self.nsd, &lat.to_f32()?, hw.0, hw.1)?;
            if prof::on() {
                eprintln!("\nvae: {:.2} s", tv.elapsed().as_secs_f64());
                prof::report(&mut |l| eprintln!("{l}"));
            }
            let pic = if req.rgba {
                Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgba }
            } else {
                Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgba.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect() }
            };
            out.push(pic);
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {} (loaded {:.0} s ago)", self.nsd.gpu.name, self.loaded.elapsed().as_secs_f64())]
    }
}
