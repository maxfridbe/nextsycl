//! HunyuanImage 2.1 on Intel Arc (SYCL): text to image, the whole pipeline on one GPU - the base model (50 steps,
//! guidance by a negative prompt) or the MeanFlow-distilled one (8 steps, guidance embedded).
//!
//! ```text
//!   prompt -> the official template -> Qwen2.5-VL-7B (te.rs: hidden_states[-3], the template's 34 tokens dropped)
//!          -> its quoted text "..." -> byT5-small (byt5.rs: the glyph tokens)
//!   noise [h/32 * w/32, 64] (seeded) -> Euler over the static-shift schedule (sched.rs), the DiT's velocity each step
//!     (dit.rs: 20 double + 40 single blocks; the base model: CFG or APG over the negative's pass)
//!   latents -> the 32x VAE decoder (vae.rs) -> RGB
//! ```
//!
//! The files are calcuis' GGUFs (ComfyUI's names): the DiT Q8_0 (`pig`), the text encoder Q4_0 (`dog`), the VAE and
//! byT5 F16, and Qwen2.5-VL's tokenizer.json. `check` compares each stage with reference/hunyuanimage21/ref.py's dumps
//! (the official modules on the same files, float32 on the CPU).

pub mod byt5;
pub mod check;
pub mod dit;
mod ffi;
pub mod prof;
pub mod sched;
pub mod te;
pub mod vae;
mod w;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_image::{At, Defaults, EngineOption, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule,
                     Step};

pub const ARCH: &str = "hunyuanimage-2.1";
pub const ROLES: &[&str] = &["transformer", "text-encoder", "tokenizer", "byt5", "vae"];

const GIB: f64 = (1u64 << 30) as f64;

/// The options it takes (`--opt-NAME`: at load the variable named; per request a field of the request's `options`)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption {
        name: "guidance",
        env: "NS_HI_GUIDANCE",
        value: "apg|cfg",
        help: "the base model's guidance: the pipeline's APG after a few CFG steps (default), or plain CFG throughout",
        at: At::Both,
    },
    EngineOption {
        name: "int8",
        env: "NS_HI_INT8",
        value: "0|1",
        help: "the DiT's products in int8 x int8 (each row of the inputs and of the matrices over its own scale; approximate, faster)",
        at: At::Load,
    },
];

/// This engine's registry entry
pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "HunyuanImage 2.1 (17B DiT, Qwen2.5-VL + byT5 text, 32x VAE; base or MeanFlow-distilled)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(HunyuanImage21::load(files, gpu, o, log)?))
}

/// The base model's guidance
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guidance {
    /// the pipeline's MIX_mode_0: CFG through step 5 (38 with glyph text), then adaptive projected guidance
    Apg,
    /// v_neg + scale (v - v_neg) every step
    Cfg,
}

impl Guidance {
    pub fn parse(v: &str) -> Result<Guidance> {
        match v {
            "apg" => Ok(Guidance::Apg),
            "cfg" => Ok(Guidance::Cfg),
            x => Err(Error(format!("guidance {x}: apg or cfg"))),
        }
    }
}

pub struct HunyuanImage21 {
    pub nsd: Nsd,
    k: ffi::K,
    pub te: te::TextEncoder,
    pub byt5: byt5::Byt5,
    pub dit: dit::Dit,
    pub vae: vae::Vae,
    guidance: Guidance,
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

/// A prompt encoded for the DiT
pub struct Encoded {
    pub text: dit::Text,
    /// whether it quotes text (the glyph path)
    pub glyphs: bool,
}

impl HunyuanImage21 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<HunyuanImage21> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let guidance = Guidance::parse(&o.setting("NS_HI_GUIDANCE").unwrap_or_else(|| "apg".into()))?;
        // the xe driver has no out-of-memory error - an allocation past the card spills to host RAM and can take the
        // machine down - so the plan is checked first: the DiT as stored, 26 of the text encoder's 28 layers, byT5 in
        // float32, the VAE's decoder (half the file), the activations of a 2048 x 2048 picture
        let size = |r: &str| -> Result<u64> { std::fs::metadata(role(files, r)?).map(|m| m.len()).map_err(|e| Error(format!("{r}: {e}"))) };
        let need = size("transformer")? + size("text-encoder")? * 26 / 28 + size("byt5")? * 2 + size("vae")? / 2 + (5u64 << 30);
        let (total, free) = gpu.memory()?;
        let free = free.unwrap_or(total);
        if need > free {
            return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free", need as f64 / GIB, gpu.name, free as f64 / GIB)));
        }
        let nsd = Nsd::new(gpu)?;
        let k = ffi::K::new(gpu)?;
        let f = nextsycl_gguf::Gguf::open(role(files, "transformer")?).map_err(w::ge)?;
        let int8 = o.setting("NS_HI_INT8").is_some_and(|v| v == "1");
        let dit = dit::Dit::load(&f, &nsd, &k, int8, log)?;
        drop(f);
        let te = te::TextEncoder::load(role(files, "text-encoder")?, role(files, "tokenizer")?, &nsd, &k, log)?;
        let byt5 = byt5::Byt5::load(role(files, "byt5")?, &nsd, &k, log)?;
        let vae = vae::Vae::load(role(files, "vae")?, &nsd, &k, log)?;
        log(format!("{ARCH}: {} model, loaded in {:.1} s", if dit.meanflow { "MeanFlow-distilled" } else { "base" }, t0.elapsed().as_secs_f64()));
        Ok(HunyuanImage21 { nsd, k, te, byt5, dit, vae, guidance, loaded: Instant::now(), load_s: t0.elapsed().as_secs_f64() })
    }

    /// The prompt through both text encoders and the DiT's prompt-only layers
    pub fn encode(&self, prompt: &str) -> Result<Encoded> {
        let q = self.te.encode(&self.nsd, prompt)?;
        let glyph = byt5::quoted(prompt);
        let b = match &glyph {
            Some(t) => Some(self.byt5.encode(&self.nsd, &byt5::ids(t))?),
            None => None,
        };
        Ok(Encoded { text: self.dit.text(&self.nsd, &q, b.as_deref())?, glyphs: glyph.is_some() })
    }

    /// Denoise the noise `x` (float32 [h * w, 64]) over `sigmas` with the model's own Euler on the GPU; `neg` the base
    /// model's negative (and the guidance), `guidance` the distilled model's embedded scale. Returns the latents.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(&self, pos: &Encoded, neg: Option<(&Encoded, f32)>, guidance: Option<f32>, x: &[f32], hw: (usize, usize), sigmas: &[f32],
                   mode: Guidance, each: &mut dyn FnMut(usize, &DevBuf) -> Result<()>) -> Result<DevBuf> {
        let nsd = &self.nsd;
        let ts = sched::timesteps(sigmas, self.dit.meanflow);
        let st = self.dit.steps(nsd, &ts, guidance.map(|g| g * 1000.0))?;
        let mut texts = vec![&pos.text];
        if let Some((e, _)) = neg {
            texts.push(&e.text);
        }
        let batch = self.dit.batch(nsd, &texts, hw)?;
        let lat = DevBuf::from_f32(&nsd.gpu, x)?;
        let n = x.len();
        // APG starts later when the glyph path is on: the pipeline sets its OCR flag from the last prompt it encodes
        // (the negative's)
        let mut apg = neg.map(|(e, _)| sched::Apg::new(if e.glyphs { sched::APG_START_OCR } else { sched::APG_START }));
        for i in 0..ts.len() {
            prof::reset_clock(nsd)?;
            let v = self.dit.forward(nsd, &st, i, &batch, &[&lat, &lat], &mut |_, _, _, _| Ok(()))?;
            let v = match (neg, &mut apg) {
                (Some((_, scale)), Some(g)) if mode == Guidance::Apg => {
                    let (c, u) = (v[0].to_f32()?, v[1].to_f32()?);
                    DevBuf::from_f32(&nsd.gpu, &g.guide(i, &c, &u, scale))?
                }
                (Some((_, scale)), _) => {
                    // v_neg + scale (v - v_neg)
                    self.k.axpy(v[0].fp(), v[1].fp(), n, -1.0)?;
                    self.k.axpy(v[1].fp(), v[0].fp(), n, scale)?;
                    v.into_iter().nth(1).unwrap_or_else(|| unreachable!())
                }
                _ => v.into_iter().next().ok_or_else(|| Error("no velocity".into()))?,
            };
            self.k.axpy(lat.fp(), v.fp(), n, sigmas[i + 1] - sigmas[i])?;
            nsd.gpu.sync()?;
            if i == 0 {
                prof::used(nsd, "after the first step");
            }
            each(i, &lat)?;
        }
        Ok(lat)
    }

    /// Denoise with one of the shared samplers (nextsycl_diffusion::samplers: ComfyUI's, on the host): each model call
    /// gets its own timestep's modulations; MeanFlow's next timestep is the schedule's next level when the call is at
    /// one, else the call's own
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_sampled(&self, pos: &Encoded, neg: Option<(&Encoded, f32)>, guidance: Option<f32>, x: Vec<f32>, hw: (usize, usize), sigmas: &[f32],
                           sampler: Sampler, seed: u64, each: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let nsd = &self.nsd;
        let mut texts = vec![&pos.text];
        if let Some((e, _)) = neg {
            texts.push(&e.text);
        }
        let batch = self.dit.batch(nsd, &texts, hw)?;
        let lat = DevBuf::f32(&nsd.gpu, x.len())?;
        let mut model = |x: &[f32], sigma: f32| -> std::result::Result<Vec<f32>, String> {
            let run = || -> Result<Vec<f32>> {
                lat.write(0, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let next = sigmas.iter().position(|s| (*s - sigma).abs() < 1e-6).and_then(|i| sigmas.get(i + 1)).copied();
                let tr = self.dit.meanflow.then(|| next.unwrap_or(sigma) * 1000.0);
                let st = self.dit.steps(nsd, &[(sigma * 1000.0, tr)], guidance.map(|g| g * 1000.0))?;
                let v = self.dit.forward(nsd, &st, 0, &batch, &[&lat, &lat], &mut |_, _, _, _| Ok(()))?;
                let v = match neg {
                    Some((_, scale)) => {
                        let (c, u) = (v[0].to_f32()?, v[1].to_f32()?);
                        c.iter().zip(&u).map(|(c, u)| u + scale * (c - u)).collect::<Vec<f32>>()
                    }
                    None => v[0].to_f32()?,
                };
                Ok(x.iter().zip(&v).map(|(a, b)| a - sigma * b).collect())
            };
            run().map_err(|e| e.0)
        };
        let mut failed: Option<Error> = None;
        let mut draw = 0u64;
        let mut noise_fn = |m: usize| -> Vec<f32> {
            draw += 1;
            noise(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(draw), m)
        };
        let start: Vec<f32> = x.iter().map(|v| v * sigmas[0]).collect();
        let mut step = |i: usize| {
            if failed.is_none() {
                if let Err(e) = each(i) {
                    failed = Some(e);
                }
            }
        };
        let out = nextsycl_diffusion::samplers::sample(sampler, &mut model, start, sigmas, &mut noise_fn, &mut step).map_err(Error)?;
        if let Some(e) = failed {
            return Err(e);
        }
        let last = *sigmas.last().unwrap_or(&0.0);
        Ok(if last > 0.0 { out.iter().map(|v| v / (1.0 - last)).collect() } else { out })
    }
}

impl ImageEngine for HunyuanImage21 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        if self.dit.meanflow {
            Defaults { width: 2048, height: 2048, steps: 8, cfg: 3.25, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 4.0 }
        } else {
            Defaults { width: 2048, height: 2048, steps: 50, cfg: 3.5, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 5.0 }
        }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn samplers(&self) -> Vec<Sampler> {
        Sampler::ALL.to_vec()
    }
    fn schedules(&self) -> Vec<Schedule> {
        Schedule::ALL.to_vec()
    }
    /// the base model: a negative prompt (default empty) and a scale above 1; the distilled model: the scale is
    /// embedded (one pass a step)
    fn guidance(&self) -> bool {
        true
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn generate(&self, req: &ImageRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        let d = self.defaults();
        let t0 = Instant::now();
        if req.edit.is_some() {
            return Err(Error(format!("{ARCH}: text to image only (no edits)")));
        }
        if !req.loras.is_empty() {
            return Err(Error(format!("{ARCH}: LoRAs are not ported yet")));
        }
        let (w, h) = (req.width.unwrap_or(d.width) as usize, req.height.unwrap_or(d.height) as usize);
        if w % 32 != 0 || h % 32 != 0 || w == 0 || h == 0 {
            return Err(Error(format!("{w}x{h}: the sides are multiples of 32")));
        }
        let hw = (h / 32, w / 32);
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.unwrap_or(d.cfg);
        let shift = req.shift.filter(|s| *s > 0.0).unwrap_or(d.shift);
        let sampler = req.sampler.unwrap_or(Sampler::Euler);
        let schedule = req.schedule.unwrap_or(Schedule::Shift);
        let mode = match req.extra.get("guidance") {
            Some(v) => Guidance::parse(v)?,
            None => self.guidance,
        };
        let sig = nextsycl_diffusion::samplers::sigmas_for(sampler, schedule, steps as u32, shift);
        let pos = self.encode(&req.prompt)?;
        // the base model's guidance: a second pass over the negative prompt (the pipeline's default: empty)
        let neg = if !self.dit.guided && cfg > 1.0 { Some(self.encode(req.negative.as_deref().unwrap_or(""))?) } else { None };
        let embedded = self.dit.guided.then_some(cfg);
        progress(Step { picture: 0, at: 0, of: (sig.len() - 1) as u32, seconds: t0.elapsed().as_secs_f64() })?;
        let mut out = Vec::new();
        for p in 0..req.n.max(1) {
            let nz = noise(req.seed + p as u64, hw.0 * hw.1 * dit::LATENT);
            let of = (sig.len() - 1) as u32;
            let negs = neg.as_ref().map(|e| (e, cfg));
            let lat = if sampler == Sampler::Euler && schedule == Schedule::Shift {
                self.denoise(&pos, negs, embedded, &nz, hw, &sig, mode, &mut |i, _| {
                    progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })
                })?.to_f32()?
            } else {
                self.denoise_sampled(&pos, negs, embedded, nz, hw, &sig, sampler, req.seed + p as u64, &mut |i| {
                    progress(Step { picture: p, at: i as u32 + 1, of, seconds: t0.elapsed().as_secs_f64() })
                })?
            };
            let tv = Instant::now();
            let (rgb, pw, ph) = self.vae.decode(&self.nsd, &lat, hw.0, hw.1)?;
            if prof::on() {
                eprintln!("\nvae: {:.2} s", tv.elapsed().as_secs_f64());
                prof::report(&mut |l| eprintln!("{l}"));
            }
            let pic = if req.rgba {
                Picture { width: pw as u32, height: ph as u32, channels: 4, data: rgb.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect() }
            } else {
                Picture { width: pw as u32, height: ph as u32, channels: 3, data: rgb }
            };
            out.push(pic);
        }
        Ok(out)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} ({}) on {} (loaded {:.0} s ago; {:.1} GiB of weights)", if self.dit.meanflow { "distilled" } else { "base" }, self.nsd.gpu.name,
                     self.loaded.elapsed().as_secs_f64(), (self.dit.bytes + self.te.bytes + self.byt5.bytes + self.vae.bytes) as f64 / GIB)]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_noise_is_seeded_and_standard() {
        let a = super::noise(7, 10_000);
        assert_eq!(a, super::noise(7, 10_000));
        assert_ne!(a[..8], super::noise(8, 8)[..]);
        let mean = a.iter().map(|x| *x as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
    }

    #[test]
    fn guidance_modes_parse() {
        assert_eq!(super::Guidance::parse("apg").unwrap(), super::Guidance::Apg);
        assert_eq!(super::Guidance::parse("cfg").unwrap(), super::Guidance::Cfg);
        assert!(super::Guidance::parse("x").is_err());
    }
}
