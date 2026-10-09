//! nextsycl-diffusion: the vocabulary and the host-side math every diffusion engine shares - image and video engines
//! depend on this, never on each other.
//!
//! - `Sampler` and `Schedule`: the names a request picks (`--sampler`, `--schedule`); an engine runs them on its own
//!   denoiser (the solver steps are the engine's: they touch its latents on the GPU).
//! - `sigmas`: the noise levels of a schedule (flow matching: 1 = noise, 0 = image), computed on the host.
//! - `LoraUse`: a LoRA file and its scale, merged at load (a preset) or applied per request.
//! - `Picture`: 8-bit RGB / RGBA pixels, read from and written to PNG.

pub mod picture;
pub mod schedule;

pub use picture::Picture;
pub use schedule::{sigmas, Schedule};

use std::path::PathBuf;

/// How a denoiser's steps are solved
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// first order (the flow models' default)
    Euler,
    /// second order, two model calls a step
    Heun,
    /// DPM-Solver++ (2M): second order from the previous step, one call a step
    DpmPp2m,
    /// DPM-Solver++ (2M) with noise injected each step
    DpmPp2mSde,
    /// UniPC (predictor-corrector, one call a step)
    UniPc,
    /// LCM: for distilled / few-step models
    Lcm,
}

impl Sampler {
    pub const ALL: [Sampler; 6] = [Sampler::Euler, Sampler::Heun, Sampler::DpmPp2m, Sampler::DpmPp2mSde, Sampler::UniPc, Sampler::Lcm];

    /// its name on the command line and in the API
    pub fn name(self) -> &'static str {
        match self {
            Sampler::Euler => "euler",
            Sampler::Heun => "heun",
            Sampler::DpmPp2m => "dpmpp-2m",
            Sampler::DpmPp2mSde => "dpmpp-2m-sde",
            Sampler::UniPc => "unipc",
            Sampler::Lcm => "lcm",
        }
    }

    pub fn parse(s: &str) -> Option<Sampler> {
        Sampler::ALL.into_iter().find(|x| x.name() == s)
    }

    /// model calls a step
    pub fn calls_per_step(self) -> u32 {
        if self == Sampler::Heun { 2 } else { 1 }
    }
}

/// A LoRA: its file and how strongly it applies
#[derive(Clone, Debug, PartialEq)]
pub struct LoraUse {
    /// its id in the registry (or a file name)
    pub name: String,
    pub path: PathBuf,
    pub scale: f32,
}

impl LoraUse {
    /// `name:scale` (the command line's form; the scale 1.0 when left out), its file found by `find`
    pub fn parse(s: &str, find: &dyn Fn(&str) -> Option<PathBuf>) -> Result<LoraUse, String> {
        let (name, scale) = match s.rsplit_once(':') {
            Some((n, x)) => (n, x.parse::<f32>().map_err(|_| format!("--lora {s}: name:scale"))?),
            None => (s, 1.0),
        };
        let path = find(name).ok_or_else(|| format!("no LoRA {name} (nextsycl models search --kind lora)"))?;
        Ok(LoraUse { name: name.to_string(), path, scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samplers_round_trip_by_name() {
        for s in Sampler::ALL {
            assert_eq!(Sampler::parse(s.name()), Some(s));
        }
        assert_eq!(Sampler::parse("nope"), None);
    }

    #[test]
    fn loras_parse() {
        let find = |n: &str| (n == "fast").then(|| PathBuf::from("/m/fast.safetensors"));
        assert_eq!(LoraUse::parse("fast:0.5", &find).unwrap().scale, 0.5);
        assert_eq!(LoraUse::parse("fast", &find).unwrap().scale, 1.0);
        assert!(LoraUse::parse("other:1", &find).is_err());
    }
}
