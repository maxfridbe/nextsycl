//! The diffusion head: a frame's acoustic latent (64 values) drawn from noise, conditioned on the language model's
//! last hidden state. Four adaLN layers of 1,536 (RMS norm, shift / scale / gate from SiLU(c), a 4,608 SwiGLU), a final
//! adaLN layer to 64 values; c = cond_proj(hidden) + the timestep's embedding (sinusoids of 256 through a two-layer
//! SiLU MLP). Guidance runs both branches (the script's hidden state and the negative one's) as two rows over the same
//! noisy latent; the v-predictions are mixed by cfg and DPM-Solver++ takes the step (sched.rs), on the device.
//!
//! The modulations depend on the step only through its timestep, so a frame's conditions for every step are made at
//! once and all their modulations are one product (the layers' 14 matrices of 1,536 read once a frame, not once a
//! step). A step is then the input product, four layers of norm / gate-up with SwiGLU / down added in by its gate, the
//! final norm and output product, and the solver's step: nothing read back until the frame's latent is used.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Nsd;

use crate::ops::{fp, Epi, Mat, Ops};
use crate::sched::Dpm;
use crate::weights::Weights;

struct Layer {
    norm: DevBuf,
    /// gate and up, their rows interleaved
    gu: Mat,
    down: Mat,
}

pub struct Head {
    layers: Vec<Layer>,
    noisy: Mat,
    cond: Mat,
    t1: Mat,
    t2: Mat,
    /// every layer's modulation (shift, scale, gate), then the final layer's (shift, scale): [14 x 1,536, 1,536]
    mods: Mat,
    out: Mat,
    pub hidden: usize,
    pub latent: usize,
    ffn: usize,
    eps: f32,
    freq_dim: usize,
    max_period: f64,
}

/// The steps whose modulations one product makes (two rows a step: the product takes at most 20 rows)
const MOD_STEPS: usize = 10;

/// One sampling's work buffers (two rows: the branches)
pub struct Work {
    x_in: DevBuf,
    c0: DevBuf,
    /// every step's conditions [steps * 2, hidden] and modulations [steps * 2, mods.n]
    c: DevBuf,
    m: DevBuf,
    x: DevBuf,
    h: DevBuf,
    act: DevBuf,
    v: DevBuf,
    /// the latent being drawn [latent] and the solver's previous data prediction
    lat: DevBuf,
    prev: DevBuf,
    /// the timesteps' embeddings [steps, hidden] and the schedule they belong to
    temb: DevBuf,
    t_in: DevBuf,
    t_mid: DevBuf,
    steps: usize,
    pub dpm: Dpm,
}

impl Head {
    pub fn load(ops: &Ops, f: &Weights, eps: f32, freq_dim: usize, max_period: f64) -> Result<Head> {
        let p = "model.prediction_head.";
        let mut layers = Vec::new();
        let mut ada = Vec::new();
        let mut h = 0;
        for l in 0.. {
            let n = |s: &str| format!("{p}layers.{l}.{s}");
            if !f.has(&n("norm.weight")) {
                break;
            }
            let a = n("adaLN_modulation.1.weight");
            h = f.shape(&a)?[1];
            ada.push(a);
            layers.push(Layer {
                norm: f.vec(ops, &n("norm.weight"))?,
                gu: f.mat_lm(ops, &[&n("ffn.gate_proj.weight"), &n("ffn.up_proj.weight")], None, true)?,
                down: f.mat(ops, &[&n("ffn.down_proj.weight")], None, false)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers.* in the file"))) };
        let hidden = h;
        let ffn = first.gu.n / 2;
        let fada = format!("{p}final_layer.adaLN_modulation.1.weight");
        let mut parts: Vec<(&str, usize, usize)> = ada.iter().map(|a| (a.as_str(), 0, 3 * hidden)).collect();
        parts.push((&fada, 0, 2 * hidden));
        let mods = f.mat_rows(ops, &parts, hidden, None, false)?;
        let noisy = f.mat(ops, &[&format!("{p}noisy_images_proj.weight")], None, true)?;
        let out = f.mat(ops, &[&format!("{p}final_layer.linear.weight")], None, false)?;
        let t1 = f.mat(ops, &[&format!("{p}t_embedder.mlp.0.weight")], None, true)?;
        if t1.k_in != freq_dim || noisy.n != hidden || out.n != noisy.k_in || mods.n != (3 * layers.len() + 2) * hidden {
            return Err(Error("the diffusion head's timestep embedding, input, modulation or output does not fit".into()));
        }
        Ok(Head {
            noisy,
            cond: f.mat(ops, &[&format!("{p}cond_proj.weight")], None, false)?,
            t1,
            t2: f.mat(ops, &[&format!("{p}t_embedder.mlp.2.weight")], None, false)?,
            mods,
            latent: out.n,
            out,
            layers,
            hidden,
            ffn,
            eps,
            freq_dim,
            max_period,
        })
    }

    /// Bytes a frame reads: the conditions' and modulations' products once, the rest a step
    pub fn frame_bytes(&self, steps: usize) -> usize {
        let step: usize = self.layers.iter().map(|l| l.gu.bytes() + l.down.bytes()).sum::<usize>() + self.noisy.bytes() + self.out.bytes();
        self.cond.bytes() + self.mods.bytes() * steps.div_ceil(MOD_STEPS) + steps * step
    }

    /// The matrices a bench times: the modulations, the first layer's gate / up and down
    pub fn mats(&self) -> [&Mat; 3] {
        [&self.mods, &self.layers[0].gu, &self.layers[0].down]
    }

    pub fn work(&self, ops: &Ops, steps: usize) -> Result<Work> {
        let g = &ops.gpu;
        let (h, s) = (self.hidden, steps.max(1));
        let zeros = |n: usize| -> Result<DevBuf> {
            let b = DevBuf::f32(g, n)?;
            b.fill(0)?;
            Ok(b)
        };
        Ok(Work {
            x_in: zeros(2 * self.noisy.k)?,
            c0: DevBuf::f32(g, 2 * h)?,
            c: DevBuf::f32(g, 2 * s * h)?,
            m: DevBuf::f32(g, 2 * s * self.mods.n)?,
            x: DevBuf::f32(g, 2 * h)?,
            h: DevBuf::f32(g, 2 * h)?,
            act: DevBuf::f32(g, 2 * self.ffn)?,
            v: DevBuf::f32(g, 2 * self.latent)?,
            lat: zeros(64.max(self.latent))?,
            prev: zeros(self.latent)?,
            temb: DevBuf::f32(g, s * h)?,
            t_in: DevBuf::f32(g, self.t1.k)?,
            t_mid: DevBuf::f32(g, h)?,
            steps: 0,
            dpm: Dpm::new(steps),
        })
    }

    /// The sinusoids of a timestep: [cos(t f_i), sin(t f_i)], f_i = exp(-ln(max_period) i / half)
    pub fn sinusoids(&self, t: f32) -> Vec<f32> {
        let half = self.freq_dim / 2;
        let f = |i: usize| (-(self.max_period.ln()) * i as f64 / half as f64).exp() as f32;
        (0..half).map(|i| (t * f(i)).cos()).chain((0..half).map(|i| (t * f(i)).sin())).collect()
    }

    /// The schedule's timestep embeddings, once a schedule
    fn timesteps(&self, ops: &Ops, w: &mut Work) -> Result<()> {
        if w.steps == w.dpm.timesteps.len() {
            return Ok(());
        }
        let h = self.hidden;
        for (i, t) in w.dpm.timesteps.clone().into_iter().enumerate() {
            let mut s = self.sinusoids(t);
            s.resize(self.t1.k, 0.0);
            w.t_in.write(0, &crate::bytes(&s))?;
            ops.gemv(w.t_in.fp(), 1, &self.t1, w.t_mid.fp(), h, false)?;
            ops.silu(w.t_mid.fp(), h)?;
            ops.gemv(w.t_mid.fp(), 1, &self.t2, fp(&w.temb, i * h), h, false)?;
        }
        w.steps = w.dpm.timesteps.len();
        Ok(())
    }

    /// The head's v-prediction for both rows at step i: w.x_in [2, k] (the noisy latent twice, zero-padded), the
    /// step's modulations in w.m -> w.v [2, latent]
    fn forward(&self, ops: &Ops, w: &Work, i: usize) -> Result<()> {
        let (h, nm) = (self.hidden, self.mods.n);
        let m = |at: usize| fp(&w.m, 2 * i * nm + at) as *const f32;
        ops.gemv(w.x_in.fp(), 2, &self.noisy, w.x.fp(), h, false)?;
        for (l, ly) in self.layers.iter().enumerate() {
            let base = 3 * l * h;
            ops.rms_norm(w.x.fp(), 2, h, Some(&ly.norm), self.eps, Some((m(base + h), m(base), nm)), w.h.fp())?;
            ops.gemv_epi(w.h.fp(), 2, h, &ly.gu, w.act.fp(), self.ffn, Epi::Swiglu, None)?;
            ops.gemv_epi(w.act.fp(), 2, self.ffn, &ly.down, w.x.fp(), h, Epi::Res, Some((m(base + 2 * h), nm)))?;
        }
        let base = 3 * self.layers.len() * h;
        ops.rms_norm(w.x.fp(), 2, h, None, self.eps, Some((m(base + h), m(base), nm)), w.h.fp())?;
        ops.gemv(w.h.fp(), 2, &self.out, w.v.fp(), self.latent, false)
    }

    /// A frame's latent, left on the GPU (`latent` reads it): `cond` [2, hidden] (the positive and the negative hidden
    /// state), from `noise` [latent] through the schedule's steps, guided by cfg; `lat` gets the latent too and `un`
    /// (with the latents' scale and bias) the unscaled one, the acoustic decoder's input. `first`: the head's first
    /// step output on both rows (a check's).
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, ops: &Ops, _nsd: &Nsd, w: &mut Work, cond: &DevBuf, noise: &[f32], cfg: f32, mut first: Option<&mut Vec<f32>>,
                  lat: Option<&DevBuf>, un: Option<(&DevBuf, f32, f32)>) -> Result<()> {
        let (h, n, k) = (self.hidden, self.latent, self.noisy.k);
        if noise.len() != n {
            return Err(Error(format!("noise of {} values, {n} expected", noise.len())));
        }
        self.timesteps(ops, w)?;
        let steps = w.dpm.timesteps.len();
        // the noise into the latent and both input rows (queued: views copy through the pinned ring)
        let z = crate::bytes(noise);
        for at in [0, k * 4] {
            w.x_in.view(at, n * 4)?.write_async(0, &z)?;
        }
        w.lat.view(0, n * 4)?.write_async(0, &z)?;
        // the conditions and modulations of every step
        ops.gemv(cond.fp(), 2, &self.cond, w.c0.fp(), h, false)?;
        ops.silu_rows(w.c0.fp(), w.temb.fp(), steps, 2, h, w.c.fp())?;
        for s0 in (0..steps).step_by(MOD_STEPS) {
            let s = (steps - s0).min(MOD_STEPS);
            ops.gemv(fp(&w.c, 2 * s0 * h), 2 * s, &self.mods, fp(&w.m, 2 * s0 * self.mods.n), self.mods.n, false)?;
        }
        for i in 0..steps {
            self.forward(ops, w, i)?;
            if let Some(f) = first.take() {
                *f = w.v.to_f32()?;
            }
            let last = i + 1 == steps;
            let c = w.dpm.coef(i);
            ops.dpm(w.v.fp(), n, cfg, w.lat.fp(), w.prev.fp(), &c, (!last).then_some((w.x_in.fp(), k)), lat.filter(|_| last).map(|l| l.fp()),
                    un.filter(|_| last).map(|(u, s, b)| (u.fp(), s, b)))?;
        }
        Ok(())
    }

    /// The latent the last `sample` drew (waits for it)
    pub fn latent(&self, w: &Work) -> Result<Vec<f32>> {
        Ok(w.lat.to_f32()?[..self.latent].to_vec())
    }
}
