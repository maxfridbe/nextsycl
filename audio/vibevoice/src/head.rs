//! The diffusion head: a frame's acoustic latent (64 values) drawn from noise, conditioned on the language model's
//! last hidden state. Four adaLN layers of 1,536 (RMS norm, shift / scale / gate from SiLU(c), a 4,608 SwiGLU), a final
//! adaLN layer to 64 values; c = cond_proj(hidden) + the timestep's embedding (sinusoids of 256 through a two-layer
//! SiLU MLP). Guidance runs both branches (the script's hidden state and the negative one's) as two rows over the same
//! noisy latent; the v-predictions are mixed by cfg and DPM-Solver++ takes the step (sched.rs) on the host.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::sched::Dpm;
use crate::weights::Weights;

struct Layer {
    norm: DevBuf,
    /// the modulation's three chunks (shift, scale, gate), each [1536, 1536]
    shift: Mat,
    scale: Mat,
    gate: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Head {
    layers: Vec<Layer>,
    noisy: Mat,
    cond: Mat,
    t1: Mat,
    t2: Mat,
    f_shift: Mat,
    f_scale: Mat,
    out: Mat,
    ones: DevBuf,
    rows: DevBuf,
    pub hidden: usize,
    pub latent: usize,
    ffn: usize,
    eps: f32,
    freq_dim: usize,
    max_period: f64,
}

/// One sampling's work buffers (two rows: the branches)
pub struct Work {
    x_in: DevBuf,
    c0: DevBuf,
    c: DevBuf,
    x: DevBuf,
    h: DevBuf,
    sh: DevBuf,
    sc: DevBuf,
    ga: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    y: DevBuf,
    v: DevBuf,
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
        for l in 0.. {
            let n = |s: &str| format!("{p}layers.{l}.{s}");
            if !f.has(&n("norm.weight")) {
                break;
            }
            let ada = n("adaLN_modulation.1.weight");
            let h = f.shape(&ada)?[1];
            let chunk = |i: usize| f.mat_rows(ops, &[(&ada, i * h, (i + 1) * h)], h, None, false);
            layers.push(Layer {
                norm: f.vec(ops, &n("norm.weight"))?,
                shift: chunk(0)?,
                scale: chunk(1)?,
                gate: chunk(2)?,
                gu: f.mat(ops, &[&n("ffn.gate_proj.weight"), &n("ffn.up_proj.weight")], None, false)?,
                down: f.mat(ops, &[&n("ffn.down_proj.weight")], None, false)?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error(format!("no {p}layers.* in the file"))) };
        let hidden = first.shift.n;
        let ffn = first.gu.n / 2;
        let fada = format!("{p}final_layer.adaLN_modulation.1.weight");
        let noisy = f.mat(ops, &[&format!("{p}noisy_images_proj.weight")], None, true)?;
        let out = f.mat(ops, &[&format!("{p}final_layer.linear.weight")], None, false)?;
        let t1 = f.mat(ops, &[&format!("{p}t_embedder.mlp.0.weight")], None, true)?;
        if t1.k_in != freq_dim || noisy.n != hidden || out.n != noisy.k_in {
            return Err(Error("the diffusion head's timestep embedding, input or output does not fit".into()));
        }
        Ok(Head {
            noisy,
            cond: f.mat(ops, &[&format!("{p}cond_proj.weight")], None, false)?,
            t1,
            t2: f.mat(ops, &[&format!("{p}t_embedder.mlp.2.weight")], None, false)?,
            f_shift: f.mat_rows(ops, &[(&fada, 0, hidden)], hidden, None, false)?,
            f_scale: f.mat_rows(ops, &[(&fada, hidden, 2 * hidden)], hidden, None, false)?,
            latent: out.n,
            out,
            ones: DevBuf::from_f32(&ops.gpu, &vec![1.0; hidden])?,
            rows: {
                let b = DevBuf::new(&ops.gpu, 8)?;
                b.write(0, &[0, 0, 0, 0, 1, 0, 0, 0])?;
                b
            },
            layers,
            hidden,
            ffn,
            eps,
            freq_dim,
            max_period,
        })
    }

    pub fn work(&self, ops: &Ops, steps: usize) -> Result<Work> {
        let g = &ops.gpu;
        let h = self.hidden;
        Ok(Work {
            x_in: {
                let b = DevBuf::f32(g, 2 * self.noisy.k)?;
                b.fill(0)?;
                b
            },
            c0: DevBuf::f32(g, 2 * h)?,
            c: DevBuf::f32(g, 2 * h)?,
            x: DevBuf::f32(g, 2 * h)?,
            h: DevBuf::f32(g, 2 * h)?,
            sh: DevBuf::f32(g, 2 * h)?,
            sc: DevBuf::f32(g, 2 * h)?,
            ga: DevBuf::f32(g, 2 * h)?,
            gu: DevBuf::f32(g, 2 * 2 * self.ffn)?,
            act: DevBuf::f32(g, 2 * self.ffn)?,
            y: DevBuf::f32(g, 2 * h)?,
            v: DevBuf::f32(g, 2 * self.latent)?,
            temb: DevBuf::f32(g, steps.max(1) * h)?,
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

    /// The head's v-prediction for both rows: x_in [2, k] (the noisy latent twice, zero-padded), c [2, hidden] the
    /// conditions with the timestep's embedding added (SiLU'd here into `w.c`) -> w.v [2, latent]
    fn forward(&self, ops: &Ops, nsd: &Nsd, w: &Work) -> Result<()> {
        let (h, rows) = (self.hidden, self.rows.ptr() as *const std::ffi::c_void);
        ops.silu(w.c.fp(), 2 * h)?;
        ops.gemv(w.x_in.fp(), 2, &self.noisy, w.x.fp(), h, false)?;
        for l in &self.layers {
            ops.gemv(w.c.fp(), 2, &l.shift, w.sh.fp(), h, false)?;
            ops.gemv(w.c.fp(), 2, &l.scale, w.sc.fp(), h, false)?;
            ops.gemv(w.c.fp(), 2, &l.gate, w.ga.fp(), h, false)?;
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, 2, h, l.norm.ptr(), self.eps, rows, w.sc.ptr(), w.sh.ptr(), w.h.ptr(), Dt::F32)?;
            ops.gemv(w.h.fp(), 2, &l.gu, w.gu.fp(), l.gu.n, false)?;
            nsd.swiglu(w.gu.ptr(), Dt::F32, 2, self.ffn, w.act.ptr(), Dt::F32)?;
            ops.gemv(w.act.fp(), 2, &l.down, w.y.fp(), h, false)?;
            nsd.gate_add(w.x.ptr(), Dt::F32, 2, h, w.y.ptr(), Dt::F32, rows, w.ga.ptr())?;
        }
        ops.gemv(w.c.fp(), 2, &self.f_shift, w.sh.fp(), h, false)?;
        ops.gemv(w.c.fp(), 2, &self.f_scale, w.sc.fp(), h, false)?;
        nsd.rms_norm_mod(w.x.ptr(), Dt::F32, 2, h, self.ones.ptr(), self.eps, rows, w.sc.ptr(), w.sh.ptr(), w.h.ptr(), Dt::F32)?;
        ops.gemv(w.h.fp(), 2, &self.out, w.v.fp(), self.latent, false)
    }

    /// A frame's latent: `cond` [2, hidden] on the GPU (the positive and the negative hidden state), from `noise`
    /// [latent] through the schedule's steps, guided by cfg. `first`: the head's first-step output on both rows (a
    /// check's)
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, ops: &Ops, nsd: &Nsd, w: &mut Work, cond: &DevBuf, noise: &[f32], cfg: f32, mut first: Option<&mut Vec<f32>>) -> Result<Vec<f32>> {
        let (h, lat, k) = (self.hidden, self.latent, self.noisy.k);
        if noise.len() != lat {
            return Err(Error(format!("noise of {} values, {lat} expected", noise.len())));
        }
        self.timesteps(ops, w)?;
        w.dpm.reset();
        ops.gemv(cond.fp(), 2, &self.cond, w.c0.fp(), h, false)?;
        let mut x = noise.to_vec();
        let mut row = vec![0f32; 2 * k];
        for i in 0..w.dpm.timesteps.len() {
            row[..lat].copy_from_slice(&x);
            row[k..k + lat].copy_from_slice(&x);
            w.x_in.write_async(0, &crate::bytes(&row))?;
            w.c.copy_within(0, &w.c0, 0, 2 * h * 4)?;
            ops.add(w.c.fp(), fp(&w.temb, i * h), h)?;
            ops.add(fp(&w.c, h), fp(&w.temb, i * h), h)?;
            self.forward(ops, nsd, w)?;
            ops.gpu.sync()?;
            let v = w.v.to_f32()?;
            if let Some(f) = first.take() {
                *f = v.clone();
            }
            let (pos, neg) = v.split_at(lat);
            let g: Vec<f32> = pos.iter().zip(neg).map(|(p, n)| n + cfg * (p - n)).collect();
            w.dpm.step(&g, &mut x);
        }
        Ok(x)
    }
}
