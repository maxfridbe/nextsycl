//! nextsycl-diffusion: the vocabulary and the host-side math every diffusion engine shares - image and video engines
//! depend on this, never on each other.
//!
//! - `Sampler` and `Schedule`: the names a request picks (`--sampler`, `--schedule`): ComfyUI's samplers and schedulers.
//! - `samplers`: the samplers' steps on the host (ComfyUI's math, flow form), calling the engine's denoiser for x0.
//! - `sigmas`: the noise levels of a schedule (flow matching: 1 = noise, 0 = image), computed on the host.
//! - `LoraUse`: a LoRA file and its scale, merged at load (a preset) or applied per request; `lora`: reading one.
//! - `Picture`: 8-bit RGB / RGBA pixels, read from and written to PNG.
//! - `kernels`: the shared diffusion kernels (kernels/diffusion, H3's) on a GPU - linears (int8 ConvRot too), norms,
//!   RoPE, attention, convolutions - for the engines and the text encoders to compose.

pub mod kernels;
pub mod lora;
pub mod picture;
pub mod samplers;
pub mod schedule;

pub use picture::Picture;
pub use schedule::{sigmas, Schedule};

use std::path::PathBuf;

/// How a denoiser's steps are solved: ComfyUI's sampler suite (`samplers::sample` runs them), by ComfyUI's names
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// first order (the flow models' default)
    Euler,
    /// Euler with noise put back each step (the flow form: `sample_euler_ancestral_RF`)
    EulerAncestral,
    /// second order, two model calls a step
    Heun,
    /// Heun with a third call where it can (three, two, then one call near the end)
    HeunPp2,
    /// exponential Heun in log-SNR on the x0 prediction, two calls a step
    ExpHeun2X0,
    /// the same with noise injected
    ExpHeun2X0Sde,
    /// DPM-Solver-2 steps (two calls); its schedule drops the next-to-last sigma
    Dpm2,
    /// DPM-Solver-2 with noise put back (the flow form); its schedule drops the next-to-last sigma
    Dpm2Ancestral,
    /// linear multistep, order 4
    Lms,
    /// DPM-Solver++(2S) with noise put back (the flow form), two calls a step
    DpmPp2sAncestral,
    /// DPM-Solver++ stochastic, two calls a step
    DpmPpSde,
    /// DPM-Solver++ (2M): second order from the previous step, one call a step
    DpmPp2m,
    /// DPM-Solver++ (2M) with noise injected each step (midpoint)
    DpmPp2mSde,
    /// the same, Heun's correction
    DpmPp2mSdeHeun,
    /// DPM-Solver++ (3M) SDE
    DpmPp3mSde,
    /// DDPM's ancestral step
    Ddpm,
    /// LCM: for distilled / few-step models
    Lcm,
    /// improved PNDM (Adams-Bashforth up to order 4)
    Ipndm,
    /// iPNDM for uneven steps
    IpndmV,
    /// DEIS (order 3, 'tab' coefficients)
    Deis,
    /// RES multistep (second order, exponential)
    ResMultistep,
    /// RES multistep with noise put back
    ResMultistepAncestral,
    /// Euler with a gradient-estimation correction
    GradientEstimation,
    /// ER-SDE-Solver-3
    ErSde,
    /// SEEDS-2 (stochastic, two calls a step)
    Seeds2,
    /// SEEDS-3 (stochastic, three calls a step)
    Seeds3,
    /// ComfyUI's "ddim": Euler (its inpainting noise aside)
    Ddim,
    /// UniPC (predictor-corrector, B(h) = h, one call a step); its schedule drops the next-to-last sigma
    UniPc,
    /// UniPC, B(h) = e^h - 1
    UniPcBh2,
}

impl Sampler {
    pub const ALL: [Sampler; 29] = [
        Sampler::Euler, Sampler::EulerAncestral, Sampler::Heun, Sampler::HeunPp2, Sampler::ExpHeun2X0,
        Sampler::ExpHeun2X0Sde, Sampler::Dpm2, Sampler::Dpm2Ancestral, Sampler::Lms, Sampler::DpmPp2sAncestral,
        Sampler::DpmPpSde, Sampler::DpmPp2m, Sampler::DpmPp2mSde, Sampler::DpmPp2mSdeHeun, Sampler::DpmPp3mSde,
        Sampler::Ddpm, Sampler::Lcm, Sampler::Ipndm, Sampler::IpndmV, Sampler::Deis, Sampler::ResMultistep,
        Sampler::ResMultistepAncestral, Sampler::GradientEstimation, Sampler::ErSde, Sampler::Seeds2, Sampler::Seeds3,
        Sampler::Ddim, Sampler::UniPc, Sampler::UniPcBh2,
    ];

    /// its name on the command line and in the API: ComfyUI's
    pub fn name(self) -> &'static str {
        match self {
            Sampler::Euler => "euler",
            Sampler::EulerAncestral => "euler_ancestral",
            Sampler::Heun => "heun",
            Sampler::HeunPp2 => "heunpp2",
            Sampler::ExpHeun2X0 => "exp_heun_2_x0",
            Sampler::ExpHeun2X0Sde => "exp_heun_2_x0_sde",
            Sampler::Dpm2 => "dpm_2",
            Sampler::Dpm2Ancestral => "dpm_2_ancestral",
            Sampler::Lms => "lms",
            Sampler::DpmPp2sAncestral => "dpmpp_2s_ancestral",
            Sampler::DpmPpSde => "dpmpp_sde",
            Sampler::DpmPp2m => "dpmpp_2m",
            Sampler::DpmPp2mSde => "dpmpp_2m_sde",
            Sampler::DpmPp2mSdeHeun => "dpmpp_2m_sde_heun",
            Sampler::DpmPp3mSde => "dpmpp_3m_sde",
            Sampler::Ddpm => "ddpm",
            Sampler::Lcm => "lcm",
            Sampler::Ipndm => "ipndm",
            Sampler::IpndmV => "ipndm_v",
            Sampler::Deis => "deis",
            Sampler::ResMultistep => "res_multistep",
            Sampler::ResMultistepAncestral => "res_multistep_ancestral",
            Sampler::GradientEstimation => "gradient_estimation",
            Sampler::ErSde => "er_sde",
            Sampler::Seeds2 => "seeds_2",
            Sampler::Seeds3 => "seeds_3",
            Sampler::Ddim => "ddim",
            Sampler::UniPc => "uni_pc",
            Sampler::UniPcBh2 => "uni_pc_bh2",
        }
    }

    /// ComfyUI's name, '-' taken for '_' (the older "dpmpp-2m", "dpmpp-2m-sde"), "unipc" for "uni_pc"
    pub fn parse(s: &str) -> Option<Sampler> {
        let s = s.replace('-', "_");
        let s = if s == "unipc" { "uni_pc".to_string() } else { s };
        Sampler::ALL.into_iter().find(|x| x.name() == s)
    }

    /// model calls a step at most (the last steps can take fewer: `samplers::model_calls` counts a schedule's)
    pub fn calls_per_step(self) -> u32 {
        match self {
            Sampler::HeunPp2 | Sampler::Seeds3 => 3,
            Sampler::Heun | Sampler::ExpHeun2X0 | Sampler::ExpHeun2X0Sde | Sampler::Dpm2 | Sampler::Dpm2Ancestral
            | Sampler::DpmPp2sAncestral | Sampler::DpmPpSde | Sampler::Seeds2 => 2,
            _ => 1,
        }
    }

    /// whether it draws noise as it goes (the caller's seed then matters past the starting latent)
    pub fn draws_noise(self) -> bool {
        matches!(
            self,
            Sampler::EulerAncestral | Sampler::ExpHeun2X0Sde | Sampler::Dpm2Ancestral | Sampler::DpmPp2sAncestral
                | Sampler::DpmPpSde | Sampler::DpmPp2mSde | Sampler::DpmPp2mSdeHeun | Sampler::DpmPp3mSde
                | Sampler::Ddpm | Sampler::Lcm | Sampler::ResMultistepAncestral | Sampler::ErSde | Sampler::Seeds2
                | Sampler::Seeds3
        )
    }

    /// whether ComfyUI computes its schedule one step longer and drops the next-to-last sigma
    /// (`KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS`); `samplers::sigmas_for` does that
    pub fn discards_penultimate_sigma(self) -> bool {
        matches!(self, Sampler::Dpm2 | Sampler::Dpm2Ancestral | Sampler::UniPc | Sampler::UniPcBh2)
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
        // the older names
        assert_eq!(Sampler::parse("dpmpp-2m"), Some(Sampler::DpmPp2m));
        assert_eq!(Sampler::parse("dpmpp-2m-sde"), Some(Sampler::DpmPp2mSde));
        assert_eq!(Sampler::parse("unipc"), Some(Sampler::UniPc));
        assert_eq!(Sampler::parse("lcm"), Some(Sampler::Lcm));
    }

    #[test]
    fn loras_parse() {
        let find = |n: &str| (n == "fast").then(|| PathBuf::from("/m/fast.safetensors"));
        assert_eq!(LoraUse::parse("fast:0.5", &find).unwrap().scale, 0.5);
        assert_eq!(LoraUse::parse("fast", &find).unwrap().scale, 1.0);
        assert!(LoraUse::parse("other:1", &find).is_err());
    }
}
