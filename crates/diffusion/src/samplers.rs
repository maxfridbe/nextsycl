//! ComfyUI's samplers on the host, for flow-matching models: a port of comfy/k_diffusion/sampling.py (with deis.py)
//! and comfy/extra_samplers/uni_pc.py, taking the paths ComfyUI takes for a `CONST` model sampling
//! (x_t = (1 - sigma) x0 + sigma noise, sigma from 1 down to 0).
//!
//! A sampler steps the latent down a schedule's sigmas, asking the engine's denoiser for its x0 prediction
//! (`model(x, sigma)`: for a velocity model, `x - sigma * v`). The latents are small, so they come to the host each
//! step; the math is done in f64 and handed to the model as f32.
//!
//! Noise: `noise(n)` returns n standard normals, the caller seeding it; the samplers that draw take them in ComfyUI's
//! order (one draw per `noise_sampler(...)` call there). ComfyUI's SDE samplers (dpmpp_sde, dpmpp_2m_sde,
//! dpmpp_2m_sde_heun, dpmpp_3m_sde) default to a Brownian tree noise sampler; its output is a unit normal per call
//! (the Brownian increment over the step divided by the square root of the step), which is what they get here - plain
//! normals, not correlated across steps the way a tree's are, so the same seed gives a different (equally valid)
//! picture than ComfyUI's tree would.
//!
//! The defaults are ComfyUI's (eta 1, s_noise 1, the r's, orders and solver types); the `_cfg_pp` variants, the
//! `_gpu` duplicates, dpm_fast, dpm_adaptive, sa_solver and ar_video are not here.

use crate::schedule::{sigmas, Schedule};
use crate::Sampler;

/// ComfyUI's names of the samplers here (`Sampler::ALL` in order)
pub const NAMES: [&str; 29] = [
    "euler", "euler_ancestral", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
    "lms", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_heun", "dpmpp_3m_sde", "ddpm",
    "lcm", "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_ancestral", "gradient_estimation", "er_sde",
    "seeds_2", "seeds_3", "ddim", "uni_pc", "uni_pc_bh2",
];

/// The denoiser: the x0 prediction at (x, sigma)
pub type Model<'a> = dyn FnMut(&[f32], f32) -> Result<Vec<f32>, String> + 'a;
/// n standard normals
pub type Noise<'a> = dyn FnMut(usize) -> Vec<f32> + 'a;

/// What a sampler needs to know of the model's sampling beyond the sigmas
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// the sigma that the log-SNR samplers (dpmpp_sde, dpmpp_2m_sde(_heun), dpmpp_3m_sde, er_sde, seeds_2/3,
    /// exp_heun_2_x0(_sde)) step from in place of a first sigma of 1, where log-SNR is infinite: ComfyUI's
    /// `offset_first_sigma_for_snr`, the model sampling's `percent_to_sigma(1e-4)`
    pub snr_first_sigma: f64,
}

impl Default for Options {
    /// an unshifted flow model's (1 - 1e-4)
    fn default() -> Options {
        Options::for_shift(1.0)
    }
}

impl Options {
    /// for a flow model whose sampling is shifted by `shift` (ComfyUI's ModelSamplingDiscreteFlow / SD3 / AuraFlow
    /// shift; 1 = none)
    pub fn for_shift(shift: f64) -> Options {
        Options { snr_first_sigma: time_snr_shift(shift, 1.0 - 1e-4) }
    }
}

/// ComfyUI's `time_snr_shift`: t moved toward 1 by `shift`
pub fn time_snr_shift(shift: f64, t: f64) -> f64 {
    if shift == 1.0 { t } else { shift * t / (1.0 + (shift - 1.0) * t) }
}

/// The sigmas ComfyUI's KSampler steps `sampler` over: the schedule's, except that dpm_2, dpm_2_ancestral, uni_pc and
/// uni_pc_bh2 get one step more with the next-to-last sigma dropped
pub fn sigmas_for(sampler: Sampler, schedule: Schedule, steps: u32, shift: f32) -> Vec<f32> {
    if !sampler.discards_penultimate_sigma() {
        return sigmas(schedule, steps, shift);
    }
    let mut v = sigmas(schedule, steps + 1, shift);
    if v.len() >= 2 {
        v.remove(v.len() - 2);
    }
    v
}

/// Runs `sampler` down `sigmas` from the latent `x` (already noised to `sigmas[0]`) and returns the result.
/// `progress(i)` is called when step i (of `sigmas.len() - 1`) is done. `Options::default()`'s model sampling.
/// The result is the latent at the last sigma; ComfyUI's KSAMPLER divides that by 1 - last sigma, which is nothing
/// for a schedule ending at 0 and is left to the caller otherwise (uni_pc's own case is kept: see `uni_pc`).
pub fn sample(
    sampler: Sampler, model: &mut Model, x: Vec<f32>, sigmas: &[f32], noise: &mut Noise, progress: &mut dyn FnMut(usize),
) -> Result<Vec<f32>, String> {
    sample_with(sampler, &Options::default(), model, x, sigmas, noise, progress)
}

/// `sample` for a model sampling that `opts` describes
pub fn sample_with(
    sampler: Sampler, opts: &Options, model: &mut Model, x: Vec<f32>, sigmas: &[f32], noise: &mut Noise,
    progress: &mut dyn FnMut(usize),
) -> Result<Vec<f32>, String> {
    if sigmas.len() <= 1 {
        return Ok(x);
    }
    let s: Vec<f64> = sigmas.iter().map(|&v| v as f64).collect();
    let x: V = x.iter().map(|&v| v as f64).collect();
    let mut r = Run { model, noise, progress, n: x.len() };
    let out = match sampler {
        Sampler::Euler | Sampler::Ddim => euler(&mut r, x, &s),
        Sampler::EulerAncestral => euler_ancestral(&mut r, x, &s, 1.0, 1.0),
        Sampler::Heun => heun(&mut r, x, &s),
        Sampler::HeunPp2 => heunpp2(&mut r, x, &s),
        Sampler::ExpHeun2X0 => seeds_2(&mut r, x, &s, opts, 0.0, 0.0, 1.0, Phi::Two),
        Sampler::ExpHeun2X0Sde => seeds_2(&mut r, x, &s, opts, 1.0, 1.0, 1.0, Phi::Two),
        Sampler::Dpm2 => dpm_2(&mut r, x, &s),
        Sampler::Dpm2Ancestral => dpm_2_ancestral(&mut r, x, &s, 1.0, 1.0),
        Sampler::Lms => lms(&mut r, x, &s, 4),
        Sampler::DpmPp2sAncestral => dpmpp_2s_ancestral(&mut r, x, &s, 1.0, 1.0),
        Sampler::DpmPpSde => dpmpp_sde(&mut r, x, &s, opts, 1.0, 1.0, 0.5),
        Sampler::DpmPp2m => dpmpp_2m(&mut r, x, &s),
        Sampler::DpmPp2mSde => dpmpp_2m_sde(&mut r, x, &s, opts, 1.0, 1.0, false),
        Sampler::DpmPp2mSdeHeun => dpmpp_2m_sde(&mut r, x, &s, opts, 1.0, 1.0, true),
        Sampler::DpmPp3mSde => dpmpp_3m_sde(&mut r, x, &s, opts, 1.0, 1.0),
        Sampler::Ddpm => ddpm(&mut r, x, &s),
        Sampler::Lcm => lcm(&mut r, x, &s),
        Sampler::Ipndm => ipndm(&mut r, x, &s, 4),
        Sampler::IpndmV => ipndm_v(&mut r, x, &s, 4),
        Sampler::Deis => deis(&mut r, x, &s, 3),
        Sampler::ResMultistep => res_multistep(&mut r, x, &s, 1.0, 0.0),
        Sampler::ResMultistepAncestral => res_multistep(&mut r, x, &s, 1.0, 1.0),
        Sampler::GradientEstimation => gradient_estimation(&mut r, x, &s, 2.0),
        Sampler::ErSde => er_sde(&mut r, x, &s, opts, 1.0, 3),
        Sampler::Seeds2 => seeds_2(&mut r, x, &s, opts, 1.0, 1.0, 0.5, Phi::One),
        Sampler::Seeds3 => seeds_3(&mut r, x, &s, opts, 1.0, 1.0, 1.0 / 3.0, 2.0 / 3.0),
        Sampler::UniPc => uni_pc(&mut r, x, &s, false),
        Sampler::UniPcBh2 => uni_pc(&mut r, x, &s, true),
    }?;
    Ok(out.into_iter().map(|v| v as f32).collect())
}

/// How many times `sampler` calls the model going down `sigmas` (for a progress bar's length)
pub fn model_calls(sampler: Sampler, sigmas: &[f32]) -> usize {
    let mut calls = 0;
    let mut model = |x: &[f32], _s: f32| {
        calls += 1;
        Ok(vec![0.0; x.len()])
    };
    let mut noise = |n: usize| vec![0.0; n];
    let _ = sample(sampler, &mut model, vec![0.0], sigmas, &mut noise, &mut |_| {});
    calls
}

type V = Vec<f64>;

/// The model, the noise and the progress callback of one run, at f64
struct Run<'a, 'b, 'c, 'd> {
    model: &'a mut Model<'b>,
    noise: &'c mut Noise<'d>,
    progress: &'a mut dyn FnMut(usize),
    n: usize,
}

impl Run<'_, '_, '_, '_> {
    /// the x0 prediction at (x, sigma)
    fn den(&mut self, x: &[f64], sigma: f64) -> Result<V, String> {
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let out = (self.model)(&xf, sigma as f32)?;
        if out.len() != x.len() {
            return Err(format!("the denoiser returned {} values for {}", out.len(), x.len()));
        }
        Ok(out.into_iter().map(|v| v as f64).collect())
    }

    /// the next draw of standard normals, one per latent value
    fn noise(&mut self) -> V {
        let v = (self.noise)(self.n);
        assert_eq!(v.len(), self.n, "noise(n) must return n values");
        v.into_iter().map(|v| v as f64).collect()
    }

    fn done(&mut self, i: usize) {
        (self.progress)(i)
    }
}

/// a x + b y
fn lin(a: f64, x: &[f64], b: f64, y: &[f64]) -> V {
    x.iter().zip(y).map(|(x, y)| a * x + b * y).collect()
}

/// x += c y
fn add(x: &mut [f64], c: f64, y: &[f64]) {
    for (x, y) in x.iter_mut().zip(y) {
        *x += c * y;
    }
}

/// x - y
fn sub(x: &[f64], y: &[f64]) -> V {
    x.iter().zip(y).map(|(x, y)| x - y).collect()
}

/// The Karras ODE derivative (x - x0) / sigma
fn to_d(x: &[f64], den: &[f64], sigma: f64) -> V {
    x.iter().zip(den).map(|(x, d)| (x - d) / sigma).collect()
}

/// ComfyUI's `get_ancestral_step`: the sigma to step down to and the noise to add after (VE form)
fn ancestral_step(from: f64, to: f64, eta: f64) -> (f64, f64) {
    if eta == 0.0 {
        return (to, 0.0);
    }
    let up = (eta * (to * to * (from * from - to * to) / (from * from)).sqrt()).min(to);
    ((to * to - up * up).sqrt(), up)
}

/// The flow ("RF") ancestral step: sigma_down, then alpha_{i+1} / alpha_down and the renoise coefficient
fn rf_step(s0: f64, s1: f64, eta: f64) -> (f64, f64, f64) {
    let sd = s1 * (1.0 + (s1 / s0 - 1.0) * eta);
    let (a1, ad) = (1.0 - s1, 1.0 - sd);
    (sd, a1 / ad, (s1 * s1 - sd * sd * a1 * a1 / (ad * ad)).sqrt())
}

/// Half log-SNR log(alpha / sigma) of a flow sigma: log((1 - sigma) / sigma)
fn half_log_snr(sigma: f64) -> f64 {
    ((1.0 - sigma) / sigma).ln()
}

/// Its inverse: 1 / (1 + e^lambda)
fn snr_sigma(lambda: f64) -> f64 {
    1.0 / (1.0 + lambda.exp())
}

/// ComfyUI's `offset_first_sigma_for_snr`: a first sigma of 1 (or more) moved to `opts.snr_first_sigma`
fn offset_first(s: &[f64], opts: &Options) -> V {
    let mut s = s.to_vec();
    if s.len() > 1 && s[0] >= 1.0 {
        s[0] = opts.snr_first_sigma;
    }
    s
}

/// h phi_2(h) = (e^h - 1 - h) / h
fn ei_h_phi_2(h: f64) -> f64 {
    (h.exp_m1() - h) / h
}

/// torch.lerp
fn lerp(a: f64, b: f64, w: f64) -> f64 {
    a + w * (b - a)
}

/// torch.nan_to_num(v, nan=0): NaN to 0, infinities to the largest finite values
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v.clamp(f64::MIN, f64::MAX) }
}

fn euler(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let d = to_d(&x, &den, s[i]);
        add(&mut x, s[i + 1] - s[i], &d);
        r.done(i);
    }
    Ok(x)
}

/// sample_euler_ancestral_RF
fn euler_ancestral(r: &mut Run, mut x: V, s: &[f64], eta: f64, s_noise: f64) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den;
        } else {
            let (sd, ratio, renoise) = rf_step(s[i], s[i + 1], eta);
            let k = sd / s[i];
            x = lin(k, &x, 1.0 - k, &den);
            if eta > 0.0 {
                let z = r.noise();
                x = lin(ratio, &x, s_noise * renoise, &z);
            }
        }
        r.done(i);
    }
    Ok(x)
}

fn heun(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let d = to_d(&x, &den, s[i]);
        let dt = s[i + 1] - s[i];
        if s[i + 1] == 0.0 {
            add(&mut x, dt, &d);
        } else {
            let x2 = lin(1.0, &x, dt, &d);
            let den2 = r.den(&x2, s[i + 1])?;
            let d2 = to_d(&x2, &den2, s[i + 1]);
            x = x.iter().zip(d.iter().zip(&d2)).map(|(x, (a, b))| x + (a + b) / 2.0 * dt).collect();
        }
        r.done(i);
    }
    Ok(x)
}

fn heunpp2(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    let end = s[s.len() - 1];
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let d = to_d(&x, &den, s[i]);
        let dt = s[i + 1] - s[i];
        if s[i + 1] == end {
            add(&mut x, dt, &d);
        } else if s[i + 2] == end {
            let x2 = lin(1.0, &x, dt, &d);
            let den2 = r.den(&x2, s[i + 1])?;
            let d2 = to_d(&x2, &den2, s[i + 1]);
            let w2 = s[i + 1] / (2.0 * s[0]);
            let w1 = 1.0 - w2;
            x = x.iter().zip(d.iter().zip(&d2)).map(|(x, (a, b))| x + (a * w1 + b * w2) * dt).collect();
        } else {
            let x2 = lin(1.0, &x, dt, &d);
            let den2 = r.den(&x2, s[i + 1])?;
            let d2 = to_d(&x2, &den2, s[i + 1]);
            let x3 = lin(1.0, &x2, s[i + 2] - s[i + 1], &d2);
            let den3 = r.den(&x3, s[i + 2])?;
            let d3 = to_d(&x3, &den3, s[i + 2]);
            let w = 3.0 * s[0];
            let (w2, w3) = (s[i + 1] / w, s[i + 2] / w);
            let w1 = 1.0 - w2 - w3;
            for j in 0..x.len() {
                x[j] += (w1 * d[j] + w2 * d2[j] + w3 * d3[j]) * dt;
            }
        }
        r.done(i);
    }
    Ok(x)
}

fn dpm_2(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let d = to_d(&x, &den, s[i]);
        if s[i + 1] == 0.0 {
            add(&mut x, s[i + 1] - s[i], &d);
        } else {
            let mid = lerp(s[i].ln(), s[i + 1].ln(), 0.5).exp();
            let x2 = lin(1.0, &x, mid - s[i], &d);
            let den2 = r.den(&x2, mid)?;
            let d2 = to_d(&x2, &den2, mid);
            add(&mut x, s[i + 1] - s[i], &d2);
        }
        r.done(i);
    }
    Ok(x)
}

/// sample_dpm_2_ancestral_RF
fn dpm_2_ancestral(r: &mut Run, mut x: V, s: &[f64], eta: f64, s_noise: f64) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let (sd, ratio, renoise) = rf_step(s[i], s[i + 1], eta);
        let d = to_d(&x, &den, s[i]);
        if sd == 0.0 {
            add(&mut x, sd - s[i], &d);
        } else {
            let mid = lerp(s[i].ln(), sd.ln(), 0.5).exp();
            let x2 = lin(1.0, &x, mid - s[i], &d);
            let den2 = r.den(&x2, mid)?;
            let d2 = to_d(&x2, &den2, mid);
            add(&mut x, sd - s[i], &d2);
            let z = r.noise();
            x = lin(ratio, &x, s_noise * renoise, &z);
        }
        r.done(i);
    }
    Ok(x)
}

/// The 4-point Gauss-Legendre rule on [a, b]: exact for the polynomials (degree 3 at most) LMS integrates, as
/// ComfyUI's scipy quad is
fn integrate_poly(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    const X: [f64; 4] = [-0.861_136_311_594_052_6, -0.339_981_043_584_856_3, 0.339_981_043_584_856_3, 0.861_136_311_594_052_6];
    const W: [f64; 4] = [0.347_854_845_137_453_9, 0.652_145_154_862_546_1, 0.652_145_154_862_546_1, 0.347_854_845_137_453_9];
    let (m, h) = ((a + b) / 2.0, (b - a) / 2.0);
    X.iter().zip(W).map(|(x, w)| w * f(m + h * x)).sum::<f64>() * h
}

/// ComfyUI's `linear_multistep_coeff`
fn lms_coeff(order: usize, t: &[f64], i: usize, j: usize) -> f64 {
    let f = |tau: f64| {
        (0..order).filter(|&k| k != j).map(|k| (tau - t[i - k]) / (t[i - j] - t[i - k])).product::<f64>()
    };
    integrate_poly(f, t[i], t[i + 1])
}

fn lms(r: &mut Run, mut x: V, s: &[f64], order: usize) -> Result<V, String> {
    let mut ds: Vec<V> = Vec::new();
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        ds.push(to_d(&x, &den, s[i]));
        if ds.len() > order {
            ds.remove(0);
        }
        if s[i + 1] == 0.0 {
            x = den;
        } else {
            let cur = (i + 1).min(order);
            for (j, d) in ds.iter().rev().enumerate().take(cur) {
                add(&mut x, lms_coeff(cur, s, i, j), d);
            }
        }
        r.done(i);
    }
    Ok(x)
}

/// sample_dpmpp_2s_ancestral_RF
fn dpmpp_2s_ancestral(r: &mut Run, mut x: V, s: &[f64], eta: f64, s_noise: f64) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let (sd, ratio, renoise) = rf_step(s[i], s[i + 1], eta);
        if s[i + 1] == 0.0 {
            let d = to_d(&x, &den, s[i]);
            add(&mut x, sd - s[i], &d);
        } else {
            let sigma_s = if s[i] == 1.0 {
                0.9999
            } else {
                let (ti, td) = (half_log_snr(s[i]), half_log_snr(sd));
                snr_sigma(ti + 0.5 * (td - ti))
            };
            let k = sigma_s / s[i];
            let u = lin(k, &x, 1.0 - k, &den);
            let di = r.den(&u, sigma_s)?;
            let k = sd / s[i];
            x = lin(k, &x, 1.0 - k, &di);
        }
        if s[i + 1] > 0.0 && eta > 0.0 {
            let z = r.noise();
            x = lin(ratio, &x, s_noise * renoise, &z);
        }
        r.done(i);
    }
    Ok(x)
}

fn dpmpp_sde(r: &mut Run, mut x: V, s: &[f64], opts: &Options, eta: f64, s_noise: f64, rr: f64) -> Result<V, String> {
    let s = offset_first(s, opts);
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den;
        } else {
            let (ls, lt) = (half_log_snr(s[i]), half_log_snr(s[i + 1]));
            let h = lt - ls;
            let ls1 = ls + rr * h;
            let fac = 1.0 / (2.0 * rr);
            let sigma_s1 = snr_sigma(ls1);
            let alpha_s = s[i] * ls.exp();
            let alpha_s1 = sigma_s1 * ls1.exp();
            let alpha_t = s[i + 1] * lt.exp();
            // step 1
            let (sd, su) = ancestral_step((-ls).exp(), (-ls1).exp(), eta);
            let h_ = -sd.ln() - ls;
            let mut x2 = lin(alpha_s1 / alpha_s * (-h_).exp(), &x, -alpha_s1 * (-h_).exp_m1(), &den);
            if eta > 0.0 && s_noise > 0.0 {
                let z = r.noise();
                add(&mut x2, alpha_s1 * s_noise * su, &z);
            }
            let den2 = r.den(&x2, sigma_s1)?;
            // step 2
            let (sd, su) = ancestral_step((-ls).exp(), (-lt).exp(), eta);
            let h_ = -sd.ln() - ls;
            let dd = lin(1.0 - fac, &den, fac, &den2);
            x = lin(alpha_t / alpha_s * (-h_).exp(), &x, -alpha_t * (-h_).exp_m1(), &dd);
            if eta > 0.0 && s_noise > 0.0 {
                let z = r.noise();
                add(&mut x, alpha_t * s_noise * su, &z);
            }
        }
        r.done(i);
    }
    Ok(x)
}

fn dpmpp_2m(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    let t = |sigma: f64| -sigma.ln();
    let mut old: Option<V> = None;
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let (ti, tn) = (t(s[i]), t(s[i + 1]));
        let h = tn - ti;
        let k = (-tn).exp() / (-ti).exp();
        x = match &old {
            Some(o) if s[i + 1] != 0.0 => {
                let rr = (ti - t(s[i - 1])) / h;
                let dd = lin(1.0 + 1.0 / (2.0 * rr), &den, -1.0 / (2.0 * rr), o);
                lin(k, &x, -(-h).exp_m1(), &dd)
            }
            _ => lin(k, &x, -(-h).exp_m1(), &den),
        };
        old = Some(den);
        r.done(i);
    }
    Ok(x)
}

fn dpmpp_2m_sde(r: &mut Run, mut x: V, s: &[f64], opts: &Options, eta: f64, s_noise: f64, heun: bool) -> Result<V, String> {
    let s = offset_first(s, opts);
    let mut old: Option<V> = None;
    let (mut h, mut h_last): (Option<f64>, Option<f64>) = (None, None);
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den.clone();
        } else {
            let (ls, lt) = (half_log_snr(s[i]), half_log_snr(s[i + 1]));
            let hh = lt - ls;
            h = Some(hh);
            let h_eta = hh * (eta + 1.0);
            let alpha_t = s[i + 1] * lt.exp();
            x = lin(s[i + 1] / s[i] * (-hh * eta).exp(), &x, alpha_t * -(-h_eta).exp_m1(), &den);
            if let (Some(o), Some(hl)) = (&old, h_last) {
                let rr = hl / hh;
                let c = if heun {
                    alpha_t * (-(-h_eta).exp_m1() / -h_eta + 1.0) * (1.0 / rr)
                } else {
                    0.5 * alpha_t * -(-h_eta).exp_m1() * (1.0 / rr)
                };
                let diff = sub(&den, o);
                add(&mut x, c, &diff);
            }
            if eta > 0.0 && s_noise > 0.0 {
                let z = r.noise();
                add(&mut x, s[i + 1] * (-(-2.0 * hh * eta).exp_m1()).sqrt() * s_noise, &z);
            }
        }
        old = Some(den);
        h_last = h;
        r.done(i);
    }
    Ok(x)
}

fn dpmpp_3m_sde(r: &mut Run, mut x: V, s: &[f64], opts: &Options, eta: f64, s_noise: f64) -> Result<V, String> {
    let s = offset_first(s, opts);
    let (mut den1, mut den2): (Option<V>, Option<V>) = (None, None);
    let (mut h, mut h1, mut h2): (Option<f64>, Option<f64>, Option<f64>) = (None, None, None);
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den.clone();
        } else {
            let (ls, lt) = (half_log_snr(s[i]), half_log_snr(s[i + 1]));
            let hh = lt - ls;
            h = Some(hh);
            let h_eta = hh * (eta + 1.0);
            let alpha_t = s[i + 1] * lt.exp();
            x = lin(s[i + 1] / s[i] * (-hh * eta).exp(), &x, alpha_t * -(-h_eta).exp_m1(), &den);
            let phi_2 = (-h_eta).exp_m1() / h_eta + 1.0;
            if let (Some(hb), Some(hc), Some(d1), Some(d2)) = (h1, h2, &den1, &den2) {
                // DPM-Solver++(3M) SDE
                let (r0, r1) = (hb / hh, hc / hh);
                let phi_3 = phi_2 / h_eta - 0.5;
                for j in 0..x.len() {
                    let d1_0 = (den[j] - d1[j]) / r0;
                    let d1_1 = (d1[j] - d2[j]) / r1;
                    let dd1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1);
                    let dd2 = (d1_0 - d1_1) / (r0 + r1);
                    x[j] += alpha_t * phi_2 * dd1 - alpha_t * phi_3 * dd2;
                }
            } else if let (Some(hb), Some(d1)) = (h1, &den1) {
                // DPM-Solver++(2M) SDE
                let rr = hb / hh;
                for j in 0..x.len() {
                    x[j] += alpha_t * phi_2 * ((den[j] - d1[j]) / rr);
                }
            }
            if eta > 0.0 && s_noise > 0.0 {
                let z = r.noise();
                add(&mut x, s[i + 1] * (-(-2.0 * hh * eta).exp_m1()).sqrt() * s_noise, &z);
            }
        }
        den2 = den1.take();
        den1 = Some(den);
        h2 = h1;
        h1 = h;
        r.done(i);
    }
    Ok(x)
}

/// generic_step_sampler with DDPMSampler_step (ComfyUI uses its VP form on any model sampling)
fn ddpm(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let (sigma, prev) = (s[i], s[i + 1]);
        let eps = to_d(&x, &den, sigma);
        let ac = 1.0 / (sigma * sigma + 1.0);
        let acp = 1.0 / (prev * prev + 1.0);
        let alpha = ac / acp;
        let scale = (1.0 + sigma * sigma).sqrt();
        let mut mu: V = x
            .iter()
            .zip(&eps)
            .map(|(x, e)| (1.0 / alpha).sqrt() * (x / scale - (1.0 - alpha) * e / (1.0 - ac).sqrt()))
            .collect();
        if prev > 0.0 {
            let z = r.noise();
            add(&mut mu, ((1.0 - alpha) * (1.0 - acp) / (1.0 - ac)).sqrt(), &z);
        }
        if prev != 0.0 {
            let k = (1.0 + prev * prev).sqrt();
            mu.iter_mut().for_each(|v| *v *= k);
        }
        x = mu;
        r.done(i);
    }
    Ok(x)
}

/// sample_lcm: x0, then noised back to the next sigma with the flow's noise_scaling
fn lcm(r: &mut Run, mut x: V, s: &[f64]) -> Result<V, String> {
    for i in 0..s.len() - 1 {
        x = r.den(&x, s[i])?;
        if s[i + 1] > 0.0 {
            let z = r.noise();
            x = lin(1.0 - s[i + 1], &x, s[i + 1], &z);
        }
        r.done(i);
    }
    Ok(x)
}

/// The previous derivatives the multistep samplers keep: the last `max_order - 1`, newest last
fn push_history(buf: &mut Vec<V>, d: V, max_order: usize) {
    if buf.len() == max_order - 1 {
        buf.remove(0);
    }
    buf.push(d);
}

fn ipndm(r: &mut Run, mut x: V, s: &[f64], max_order: usize) -> Result<V, String> {
    let mut buf: Vec<V> = Vec::new();
    for i in 0..s.len() - 1 {
        let (t, tn) = (s[i], s[i + 1]);
        let den = r.den(&x, t)?;
        let d = to_d(&x, &den, t);
        let order = max_order.min(i + 1);
        let b = |k: usize| &buf[buf.len() - k];
        if tn == 0.0 {
            x = den;
        } else {
            let dt = tn - t;
            for j in 0..x.len() {
                let slope = match order {
                    1 => d[j],
                    2 => (3.0 * d[j] - b(1)[j]) / 2.0,
                    3 => (23.0 * d[j] - 16.0 * b(1)[j] + 5.0 * b(2)[j]) / 12.0,
                    _ => (55.0 * d[j] - 59.0 * b(1)[j] + 37.0 * b(2)[j] - 9.0 * b(3)[j]) / 24.0,
                };
                x[j] += dt * slope;
            }
        }
        push_history(&mut buf, d, max_order);
        r.done(i);
    }
    Ok(x)
}

fn ipndm_v(r: &mut Run, mut x: V, s: &[f64], max_order: usize) -> Result<V, String> {
    let mut buf: Vec<V> = Vec::new();
    for i in 0..s.len() - 1 {
        let (t, tn) = (s[i], s[i + 1]);
        let den = r.den(&x, t)?;
        let d = to_d(&x, &den, t);
        let order = max_order.min(i + 1);
        if tn == 0.0 {
            x = den;
        } else {
            let hn = tn - t;
            // the coefficients of d and the previous derivatives, newest first
            let c: Vec<f64> = match order {
                1 => vec![1.0],
                2 => {
                    let h1 = t - s[i - 1];
                    vec![(2.0 + hn / h1) / 2.0, -(hn / h1) / 2.0]
                }
                3 => {
                    let (h1, h2) = (t - s[i - 1], s[i - 1] - s[i - 2]);
                    let temp = (1.0 - hn / (3.0 * (hn + h1)) * (hn * (hn + h1)) / (h1 * (h1 + h2))) / 2.0;
                    vec![(2.0 + hn / h1) / 2.0 + temp, -(hn / h1) / 2.0 - (1.0 + h1 / h2) * temp, temp * h1 / h2]
                }
                _ => {
                    let (h1, h2, h3) = (t - s[i - 1], s[i - 1] - s[i - 2], s[i - 2] - s[i - 3]);
                    let temp1 = (1.0 - hn / (3.0 * (hn + h1)) * (hn * (hn + h1)) / (h1 * (h1 + h2))) / 2.0;
                    let temp2 = ((1.0 - hn / (3.0 * (hn + h1))) / 2.0
                        + (1.0 - hn / (2.0 * (hn + h1))) * hn / (6.0 * (hn + h1 + h2)))
                        * (hn * (hn + h1) * (hn + h1 + h2))
                        / (h1 * (h1 + h2) * (h1 + h2 + h3));
                    let q = h1 * (h1 + h2) / (h2 * (h2 + h3));
                    vec![
                        (2.0 + hn / h1) / 2.0 + temp1 + temp2,
                        -(hn / h1) / 2.0 - (1.0 + h1 / h2) * temp1 - (1.0 + (h1 / h2) + q) * temp2,
                        temp1 * h1 / h2 + ((h1 / h2) + q * (1.0 + h2 / h3)) * temp2,
                        -temp2 * q * h1 / h2,
                    ]
                }
            };
            for j in 0..x.len() {
                let mut slope = c[0] * d[j];
                for (k, ck) in c.iter().enumerate().skip(1) {
                    slope += ck * buf[buf.len() - k][j];
                }
                x[j] += hn * slope;
            }
        }
        push_history(&mut buf, d, max_order);
        r.done(i);
    }
    Ok(x)
}

/// DEIS's 'tab' coefficients (comfy/k_diffusion/deis.py): the sigmas taken to the VP time of EDM's schedule
/// (sigma_min 0.002, sigma_max 80, epsilon 1e-3), the exponential integrator's weights of the Lagrange polynomials
/// through the previous times summed on a 10000-point grid, as there. One list per step, current first.
fn deis_coeffs(s: &[f64], max_order: usize) -> Vec<Vec<f64>> {
    // edm2t's constants are float32 tensors there: keep their rounding
    let (smin, smax, eps_s) = (0.002f32, 80.0f32, 1e-3f32);
    let a = (smin * smin + 1.0).ln() / eps_s;
    let b = (smax * smax + 1.0).ln();
    let beta_d = 2.0 * (a - b) / (eps_s - 1.0);
    let beta_min = b - 0.5 * beta_d;
    let (b0, b1) = (beta_min as f64, (beta_d + beta_min) as f64);
    let (bm2, two_bd) = ((beta_min * beta_min) as f64, (2.0 * beta_d) as f64);
    let t: Vec<f64> =
        s.iter().map(|&sig| ((bm2 + two_bd * (sig * sig + 1.0).ln()).sqrt() - beta_min as f64) / beta_d as f64).collect();
    const N: usize = 10000;
    let mut out = Vec::new();
    for i in 0..t.len() - 1 {
        let order = (i + 1).min(max_order);
        // the last step (to sigma 0) runs first order: its integrand is infinite at t = 0
        if order == 1 || s[i + 1] <= 0.0 {
            out.push(Vec::new());
            continue;
        }
        let (tc, tn) = (t[i], t[i + 1]);
        let dtau = (tn - tc) / N as f64;
        let prev: Vec<f64> = (0..order).map(|k| t[i - k]).collect();
        let mut c = vec![0.0f64; order];
        for m in 0..N {
            let tau = tc + (tn - tc) * m as f64 / (N - 1) as f64;
            let alpha = (-0.5 * tau * tau * (b1 - b0) - tau * b0).exp();
            let dlog = -tau * (b1 - b0) - b0;
            let integrand = -0.5 * dlog / (alpha * (1.0 - alpha)).sqrt();
            for (j, cj) in c.iter_mut().enumerate() {
                let poly: f64 =
                    (0..order).filter(|&k| k != j).map(|k| (tau - prev[k]) / (prev[j] - prev[k])).product();
                *cj += integrand * poly;
            }
        }
        out.push(c.into_iter().map(|v| v * dtau).collect());
    }
    out
}

fn deis(r: &mut Run, mut x: V, s: &[f64], max_order: usize) -> Result<V, String> {
    let coeffs = deis_coeffs(s, max_order);
    let mut buf: Vec<V> = Vec::new();
    for i in 0..s.len() - 1 {
        let (t, tn) = (s[i], s[i + 1]);
        let den = r.den(&x, t)?;
        let d = to_d(&x, &den, t);
        let order = if tn <= 0.0 { 1 } else { max_order.min(i + 1) };
        if order == 1 {
            add(&mut x, tn - t, &d);
        } else {
            let c = &coeffs[i];
            for j in 0..x.len() {
                let mut v = c[0] * d[j];
                for (k, ck) in c.iter().enumerate().skip(1) {
                    v += ck * buf[buf.len() - k][j];
                }
                x[j] += v;
            }
        }
        push_history(&mut buf, d, max_order);
        r.done(i);
    }
    Ok(x)
}

/// res_multistep (eta 0) and res_multistep_ancestral (eta 1), not CFG++
fn res_multistep(r: &mut Run, mut x: V, s: &[f64], s_noise: f64, eta: f64) -> Result<V, String> {
    let t = |sigma: f64| -sigma.ln();
    let phi1 = |v: f64| v.exp_m1() / v;
    let phi2 = |v: f64| (phi1(v) - 1.0) / v;
    let mut old: Option<(V, f64)> = None; // the previous x0 and sigma_down
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let (sd, su) = ancestral_step(s[i], s[i + 1], eta);
        match &old {
            Some((od, osd)) if sd != 0.0 => {
                let (ti, t_old, t_next, t_prev) = (t(s[i]), t(*osd), t(sd), t(s[i - 1]));
                let h = t_next - ti;
                let c2 = (t_prev - t_old) / h;
                let (p1, p2) = (phi1(-h), phi2(-h));
                let b1 = nan_to_num(p1 - p2 / c2);
                let b2 = nan_to_num(p2 / c2);
                let e = (-h).exp();
                x = x.iter().zip(den.iter().zip(od)).map(|(x, (d, o))| e * x + h * (b1 * d + b2 * o)).collect();
            }
            _ => {
                let d = to_d(&x, &den, s[i]);
                add(&mut x, sd - s[i], &d);
            }
        }
        if su > 0.0 {
            let z = r.noise();
            add(&mut x, s_noise * su, &z);
        }
        old = Some((den, sd));
        r.done(i);
    }
    Ok(x)
}

fn gradient_estimation(r: &mut Run, mut x: V, s: &[f64], ge_gamma: f64) -> Result<V, String> {
    let mut old_d: Option<V> = None;
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let d = to_d(&x, &den, s[i]);
        let dt = s[i + 1] - s[i];
        if s[i + 1] == 0.0 {
            x = den;
        } else {
            add(&mut x, dt, &d);
            if let Some(o) = &old_d {
                for j in 0..x.len() {
                    x[j] += (ge_gamma - 1.0) * (d[j] - o[j]) * dt;
                }
            }
        }
        old_d = Some(d);
        r.done(i);
    }
    Ok(x)
}

/// ER-SDE-Solver-3 with ComfyUI's default noise scaler x (e^(x^0.3) + 10)
fn er_sde(r: &mut Run, mut x: V, s: &[f64], opts: &Options, s_noise: f64, max_stage: usize) -> Result<V, String> {
    let scaler = |v: f64| v * ((v.powf(0.3)).exp() + 10.0);
    const POINTS: usize = 200;
    let s = offset_first(s, opts);
    // er_lambda = sigma / alpha
    let er: Vec<f64> = s.iter().map(|&v| (-half_log_snr(v)).exp()).collect();
    let mut old: Option<V> = None;
    let mut old_dd: Option<V> = None;
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        let stage = max_stage.min(i + 1);
        if s[i + 1] == 0.0 {
            x = den.clone();
        } else {
            let (els, elt) = (er[i], er[i + 1]);
            let alpha_s = s[i] / els;
            let alpha_t = s[i + 1] / elt;
            let r_alpha = alpha_t / alpha_s;
            let rr = scaler(elt) / scaler(els);
            x = lin(r_alpha * rr, &x, alpha_t * (1.0 - rr), &den);
            if stage >= 2 {
                let o = old.as_ref().expect("the previous step's x0");
                let dt = elt - els;
                let step = -dt / POINTS as f64;
                let pos = |k: usize| elt + k as f64 * step;
                let sum: f64 = (0..POINTS).map(|k| 1.0 / scaler(pos(k))).sum::<f64>() * step;
                let dd: V = den.iter().zip(o).map(|(a, b)| (a - b) / (els - er[i - 1])).collect();
                add(&mut x, alpha_t * (dt + sum * scaler(elt)), &dd);
                if stage >= 3 {
                    let od = old_dd.as_ref().expect("the previous step's derivative");
                    let s_u: f64 = (0..POINTS).map(|k| (pos(k) - els) / scaler(pos(k))).sum::<f64>() * step;
                    let du: V = dd.iter().zip(od).map(|(a, b)| (a - b) / ((els - er[i - 2]) / 2.0)).collect();
                    add(&mut x, alpha_t * (dt * dt / 2.0 + s_u * scaler(elt)), &du);
                }
                old_dd = Some(dd);
            }
            if s_noise > 0.0 {
                let z = r.noise();
                let c = nan_to_num((elt * elt - els * els * rr * rr).sqrt());
                add(&mut x, alpha_t * s_noise * c, &z);
            }
        }
        old = Some(den);
        r.done(i);
    }
    Ok(x)
}

/// seeds_2's second step: the x0 lerp (SEEDS-2) or the phi_2 weights (exponential Heun)
#[derive(Clone, Copy, PartialEq)]
enum Phi {
    One,
    Two,
}

/// SEEDS-2; exp_heun_2_x0 is it with r 1, phi_2 and no noise
#[allow(clippy::too_many_arguments)]
fn seeds_2(r: &mut Run, mut x: V, s: &[f64], opts: &Options, eta: f64, s_noise: f64, rr: f64, phi: Phi) -> Result<V, String> {
    let inject = eta > 0.0 && s_noise > 0.0;
    let s = offset_first(s, opts);
    let fac = 1.0 / (2.0 * rr);
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den;
            r.done(i);
            continue;
        }
        let (ls, lt) = (half_log_snr(s[i]), half_log_snr(s[i + 1]));
        let h = lt - ls;
        let h_eta = h * (eta + 1.0);
        let ls1 = lerp(ls, lt, rr);
        let sigma_s1 = snr_sigma(ls1);
        let alpha_s1 = sigma_s1 * ls1.exp();
        let alpha_t = s[i + 1] * lt.exp();
        // step 1
        let mut x2 = lin(sigma_s1 / s[i] * (-rr * h * eta).exp(), &x, -alpha_s1 * (-rr * h_eta).exp_m1(), &den);
        let mut sde = Vec::new();
        if inject {
            let z = r.noise();
            let c = (-(-2.0 * rr * h * eta).exp_m1()).sqrt();
            sde = z.iter().map(|v| c * v).collect();
            add(&mut x2, sigma_s1 * s_noise, &sde);
        }
        let den2 = r.den(&x2, sigma_s1)?;
        // step 2
        let k = s[i + 1] / s[i] * (-h * eta).exp();
        x = match phi {
            Phi::One => {
                let dd: V = den.iter().zip(&den2).map(|(a, b)| lerp(*a, *b, fac)).collect();
                lin(k, &x, -alpha_t * (-h_eta).exp_m1(), &dd)
            }
            Phi::Two => {
                let b2 = ei_h_phi_2(-h_eta) / rr;
                let b1 = (-h_eta).exp_m1() - b2;
                x.iter().zip(den.iter().zip(&den2)).map(|(x, (a, b))| k * x - alpha_t * (b1 * a + b2 * b)).collect()
            }
        };
        if inject {
            let seg = (rr - 1.0) * h * eta;
            let z = r.noise();
            let c = (-(2.0 * seg).exp_m1()).sqrt();
            for j in 0..x.len() {
                x[j] += (sde[j] * seg.exp() + c * z[j]) * s[i + 1] * s_noise;
            }
        }
        r.done(i);
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn seeds_3(r: &mut Run, mut x: V, s: &[f64], opts: &Options, eta: f64, s_noise: f64, r1: f64, r2: f64) -> Result<V, String> {
    let inject = eta > 0.0 && s_noise > 0.0;
    let s = offset_first(s, opts);
    for i in 0..s.len() - 1 {
        let den = r.den(&x, s[i])?;
        if s[i + 1] == 0.0 {
            x = den;
            r.done(i);
            continue;
        }
        let (ls, lt) = (half_log_snr(s[i]), half_log_snr(s[i + 1]));
        let h = lt - ls;
        let h_eta = h * (eta + 1.0);
        let (ls1, ls2) = (lerp(ls, lt, r1), lerp(ls, lt, r2));
        let (sigma_s1, sigma_s2) = (snr_sigma(ls1), snr_sigma(ls2));
        let alpha_s1 = sigma_s1 * ls1.exp();
        let alpha_s2 = sigma_s2 * ls2.exp();
        let alpha_t = s[i + 1] * lt.exp();
        // step 1
        let mut x2 = lin(sigma_s1 / s[i] * (-r1 * h * eta).exp(), &x, -alpha_s1 * (-r1 * h_eta).exp_m1(), &den);
        let mut sde = Vec::new();
        if inject {
            let z = r.noise();
            let c = (-(-2.0 * r1 * h * eta).exp_m1()).sqrt();
            sde = z.iter().map(|v| c * v).collect();
            add(&mut x2, sigma_s1 * s_noise, &sde);
        }
        let den2 = r.den(&x2, sigma_s1)?;
        // step 2
        let a3_2 = r2 / r1 * ei_h_phi_2(-r2 * h_eta);
        let a3_1 = (-r2 * h_eta).exp_m1() - a3_2;
        let k = sigma_s2 / s[i] * (-r2 * h * eta).exp();
        let mut x3: V =
            x.iter().zip(den.iter().zip(&den2)).map(|(x, (a, b))| k * x - alpha_s2 * (a3_1 * a + a3_2 * b)).collect();
        if inject {
            let seg = (r1 - r2) * h * eta;
            let z = r.noise();
            let c = (-(2.0 * seg).exp_m1()).sqrt();
            for j in 0..x.len() {
                sde[j] = sde[j] * seg.exp() + c * z[j];
            }
            add(&mut x3, sigma_s2 * s_noise, &sde);
        }
        let den3 = r.den(&x3, sigma_s2)?;
        // step 3
        let b3 = ei_h_phi_2(-h_eta) / r2;
        let b1 = (-h_eta).exp_m1() - b3;
        let k = s[i + 1] / s[i] * (-h * eta).exp();
        x = x.iter().zip(den.iter().zip(&den3)).map(|(x, (a, c))| k * x - alpha_t * (b1 * a + b3 * c)).collect();
        if inject {
            let seg = (r2 - 1.0) * h * eta;
            let z = r.noise();
            let c = (-(2.0 * seg).exp_m1()).sqrt();
            for j in 0..x.len() {
                x[j] += (sde[j] * seg.exp() + c * z[j]) * s[i + 1] * s_noise;
            }
        }
        r.done(i);
    }
    Ok(x)
}

/// UniPC's noise schedule over ComfyUI's sigmas (`SigmaConvert`: the VP view of a VE sigma, alpha = 1 / sqrt(1 + s^2))
mod vp {
    pub fn log_mean_coeff(t: f64) -> f64 {
        0.5 * (1.0 / (t * t + 1.0)).ln()
    }
    pub fn alpha(t: f64) -> f64 {
        log_mean_coeff(t).exp()
    }
    pub fn std(t: f64) -> f64 {
        (1.0 - (2.0 * log_mean_coeff(t)).exp()).sqrt()
    }
    pub fn lambda(t: f64) -> f64 {
        let l = log_mean_coeff(t);
        l - 0.5 * (1.0 - (2.0 * l).exp()).ln()
    }
}

/// Solves the small system a x = b (Gaussian elimination, partial pivoting)
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap_or(c);
        a.swap(c, p);
        b.swap(c, p);
        for rr in c + 1..n {
            let f = a[rr][c] / a[c][c];
            let (top, rest) = a.split_at_mut(rr);
            for (v, p) in rest[0][c..].iter_mut().zip(&top[c][c..]) {
                *v -= f * p;
            }
            b[rr] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for c in (0..n).rev() {
        x[c] = (b[c] - (c + 1..n).map(|k| a[c][k] * x[k]).sum::<f64>()) / a[c][c];
    }
    x
}

/// UniPC's data prediction at (x, t): the model sees x scaled back to VE (`predict_eps_sigma`), its noise estimate is
/// turned into x0
fn unipc_x0(r: &mut Run, x: &[f64], t: f64) -> Result<V, String> {
    let k = (t * t + 1.0).sqrt();
    let input: V = x.iter().map(|v| v * k).collect();
    let den = r.den(&input, t)?;
    let (a, sd) = (vp::alpha(t), vp::std(t));
    Ok(x.iter().zip(input.iter().zip(&den)).map(|(x, (i, d))| (x - sd * ((i - d) / t)) / a).collect())
}

/// One multistep UniPC update (`multistep_uni_pc_bh_update`, data prediction) from the previous x0s at the previous
/// times to `t`; with the corrector, also the x0 at the new point
#[allow(clippy::too_many_arguments)]
fn unipc_update(
    r: &mut Run, x: &[f64], models: &[V], ts: &[f64], t: f64, order: usize, corrector: bool, bh2: bool,
) -> Result<(V, Option<V>), String> {
    let t0 = ts[ts.len() - 1];
    let m0 = &models[models.len() - 1];
    let (l0, lt) = (vp::lambda(t0), vp::lambda(t));
    let (sigma0, sigma_t) = (vp::std(t0), vp::std(t));
    let alpha_t = vp::alpha(t);
    let h = lt - l0;
    let mut rks = Vec::new();
    let mut d1s: Vec<V> = Vec::new();
    for i in 1..order {
        let ti = ts[ts.len() - (i + 1)];
        let mi = &models[models.len() - (i + 1)];
        let rk = (vp::lambda(ti) - l0) / h;
        rks.push(rk);
        d1s.push(mi.iter().zip(m0).map(|(a, b)| (a - b) / rk).collect());
    }
    rks.push(1.0);
    let hh = -h;
    let h_phi_1 = hh.exp_m1();
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let mut factorial = 1.0;
    let b_h = if bh2 { hh.exp_m1() } else { hh };
    let mut rm: Vec<Vec<f64>> = Vec::new();
    let mut b = Vec::new();
    for i in 1..=order {
        rm.push(rks.iter().map(|v| v.powi(i as i32 - 1)).collect());
        b.push(h_phi_k * factorial / b_h);
        factorial *= (i + 1) as f64;
        h_phi_k = h_phi_k / hh - 1.0 / factorial;
    }
    let rhos_p = if d1s.is_empty() {
        Vec::new()
    } else if order == 2 {
        vec![0.5]
    } else {
        let sub: Vec<Vec<f64>> = rm[..order - 1].iter().map(|row| row[..order - 1].to_vec()).collect();
        solve(sub, b[..order - 1].to_vec())
    };
    let x_t_: V = x.iter().zip(m0).map(|(x, m)| sigma_t / sigma0 * x - alpha_t * h_phi_1 * m).collect();
    let mut x_t = x_t_.clone();
    for (rho, d1) in rhos_p.iter().zip(&d1s) {
        add(&mut x_t, -alpha_t * b_h * rho, d1);
    }
    if !corrector {
        return Ok((x_t, None));
    }
    let rhos_c = if order == 1 { vec![0.5] } else { solve(rm, b) };
    let model_t = unipc_x0(r, &x_t, t)?;
    let mut x_t = x_t_;
    let last = rhos_c[rhos_c.len() - 1];
    for j in 0..x_t.len() {
        let corr: f64 = rhos_c.iter().zip(&d1s).map(|(rho, d1)| rho * d1[j]).sum();
        x_t[j] -= alpha_t * b_h * (corr + last * (model_t[j] - m0[j]));
    }
    Ok((x_t, Some(model_t)))
}

/// sample_unipc (bh1) / sample_unipc_bh2: multistep, order min(3, steps - 1), lower orders at the end; a last sigma
/// of 0 becomes 0.001 (UniPC's times must stay positive) and ComfyUI's KSAMPLER then divides the result by
/// 1 - 0.001 (its `inverse_noise_scaling` sees the changed sigma), which is kept here
fn uni_pc(r: &mut Run, x: V, s: &[f64], bh2: bool) -> Result<V, String> {
    let mut ts = s.to_vec();
    let last = ts.len() - 1;
    let zero_end = ts[last] == 0.0;
    if zero_end {
        ts[last] = 0.001;
    }
    let steps = ts.len() - 1;
    let order = 3.min(ts.len() as i64 - 2).max(0) as usize;
    let k = (1.0 + ts[0] * ts[0]).sqrt();
    let mut x: V = x.iter().map(|v| v / k).collect();
    let mut models: Vec<V> = Vec::new();
    let mut tprev: Vec<f64> = Vec::new();
    for step_index in 0..steps {
        if step_index == 0 {
            models.push(unipc_x0(r, &x, ts[0])?);
            tprev.push(ts[0]);
        } else if step_index < order {
            let t = ts[step_index];
            let (nx, m) = unipc_update(r, &x, &models, &tprev, t, step_index, true, bh2)?;
            x = nx;
            let m = match m {
                Some(m) => m,
                None => unipc_x0(r, &x, t)?,
            };
            models.push(m);
            tprev.push(t);
        } else {
            let extra = usize::from(step_index == steps - 1);
            for (step, &t) in ts.iter().enumerate().take(step_index + extra + 1).skip(step_index) {
                let step_order = order.min(steps + 1 - step);
                let corrector = step != steps;
                let (nx, m) = unipc_update(r, &x, &models, &tprev, t, step_order, corrector, bh2)?;
                x = nx;
                for i in 0..order - 1 {
                    tprev[i] = tprev[i + 1];
                    models.swap(i, i + 1);
                }
                tprev[order - 1] = t;
                if step < steps {
                    models[order - 1] = match m {
                        Some(m) => m,
                        None => unipc_x0(r, &x, t)?,
                    };
                }
            }
        }
        r.done(step_index);
    }
    let a = vp::alpha(ts[last]);
    let undo = if zero_end { 1.0 - 0.001 } else { 1.0 };
    Ok(x.into_iter().map(|v| v / a / undo).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// reference/samplers/expected.txt, written by reference/samplers/comfy_ref.py from ComfyUI's own code
    struct Expected {
        noise: Vec<f64>,
        x: Vec<f64>,
        sigmas: HashMap<String, Vec<f32>>,
        samplers: Vec<(String, String, Vec<f64>)>,
    }

    fn expected() -> Expected {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/samplers/expected.txt");
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut e = Expected { noise: vec![], x: vec![], sigmas: HashMap::new(), samplers: vec![] };
        let nums = |w: &[&str]| w.iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w[0] {
                "noise" => e.noise = nums(&w[1..]),
                "x" => e.x = nums(&w[1..]),
                "sigmas" => {
                    e.sigmas.insert(w[1].to_string(), nums(&w[2..]).into_iter().map(|v| v as f32).collect());
                }
                "sampler" => e.samplers.push((w[1].to_string(), w[2].to_string(), nums(&w[3..]))),
                _ => {}
            }
        }
        e
    }

    /// the harness's denoiser: tanh(0.9 x + 0.03 i) (1 - sigma) + 0.1 sigma
    fn denoiser(x: &[f32], sigma: f32) -> Result<Vec<f32>, String> {
        let s = sigma as f64;
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| ((0.9 * v as f64 + 0.03 * i as f64).tanh() * (1.0 - s) + 0.1 * s) as f32)
            .collect())
    }

    /// the model sampling each set of the harness runs with
    fn opts(set: &str) -> Options {
        if set == "shift3" { Options::for_shift(3.0) } else { Options::default() }
    }

    #[test]
    fn names_are_comfyuis() {
        assert_eq!(NAMES.len(), Sampler::ALL.len());
        for (n, s) in NAMES.iter().zip(Sampler::ALL) {
            assert_eq!(*n, s.name());
        }
    }

    #[test]
    fn every_sampler_matches_comfyui() {
        let e = expected();
        let mut worst: HashMap<String, f64> = HashMap::new();
        assert_eq!(e.samplers.len(), 3 * NAMES.len(), "expected.txt: one result per sampler and set");
        for (set, name, want) in &e.samplers {
            let sampler = Sampler::parse(name).unwrap_or_else(|| panic!("{name}?"));
            let sig = &e.sigmas[set];
            let mut pos = 0;
            let mut noise = |n: usize| {
                let v: Vec<f32> = e.noise[pos..pos + n].iter().map(|&v| v as f32).collect();
                pos += n;
                v
            };
            let mut steps = 0;
            let x: Vec<f32> = e.x.iter().map(|&v| v as f32).collect();
            let got =
                sample_with(sampler, &opts(set), &mut denoiser, x, sig, &mut noise, &mut |i| steps = i + 1).unwrap();
            assert_eq!(steps, sig.len() - 1, "{name} {set}: progress");
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs())) / scale;
            let w = worst.entry(format!("{name} {set}")).or_insert(0.0);
            *w = w.max(err);
        }
        let mut names: Vec<_> = worst.into_iter().collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (n, w) in &names {
            println!("{n:32} max relative error vs ComfyUI {w:.2e}");
        }
        let bad: Vec<_> = names.iter().filter(|(_, w)| w.is_nan() || *w >= 1e-4).collect();
        assert!(bad.is_empty(), "off ComfyUI by more than 1e-4: {bad:?}");
    }

    #[test]
    fn model_calls_are_counted() {
        let s = crate::sigmas(Schedule::Shift, 10, 3.0);
        assert_eq!(model_calls(Sampler::Euler, &s), 10);
        // Heun's last step to 0 is Euler
        assert_eq!(model_calls(Sampler::Heun, &s), 19);
        assert_eq!(model_calls(Sampler::UniPc, &s), 10);
        for sampler in Sampler::ALL {
            let n = model_calls(sampler, &s);
            assert!(n >= 10 && n <= 10 * sampler.calls_per_step() as usize, "{sampler:?}: {n}");
        }
    }

    #[test]
    fn noise_draws_only_where_comfyui_draws() {
        let s = crate::sigmas(Schedule::Shift, 6, 3.0);
        for sampler in Sampler::ALL {
            let mut draws = 0;
            let mut noise = |n: usize| {
                draws += 1;
                vec![0.5; n]
            };
            sample(sampler, &mut denoiser, vec![0.3; 4], &s, &mut noise, &mut |_| {}).unwrap();
            assert_eq!(draws > 0, sampler.draws_noise(), "{sampler:?}");
        }
    }

    #[test]
    fn discarding_samplers_drop_the_next_to_last_sigma() {
        let plain = sigmas_for(Sampler::Euler, Schedule::Simple, 4, 1.0);
        assert_eq!(plain, crate::sigmas(Schedule::Simple, 4, 1.0));
        let longer = crate::sigmas(Schedule::Simple, 5, 1.0);
        let d = sigmas_for(Sampler::UniPc, Schedule::Simple, 4, 1.0);
        assert_eq!(d.len(), 5);
        assert_eq!(&d[..4], &longer[..4]);
        assert_eq!(d[4], 0.0);
    }
}
