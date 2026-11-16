//! The MeanFlow head: the next frame's 64-value audio latent from the language model's hidden state, in one step of a
//! small MLP (haru/models/meanflow_head.py). The head predicts the mean velocity u(z, t, h | c, w) over an interval h
//! (noise at t = 0 to data at t = 1); a step is z += h * u. Its condition is the projected hidden state plus four
//! scalar embeddings (t, the interval constant, h and the guidance's 1 - 1/w), each sinusoidal features through a
//! two-layer MLP; those four are constants of a session, made here on the host once a guidance. Then four
//! adaptive-norm layers (shift, scale and a gate from the condition; a SwiGLU feed-forward) and a final adaptive norm
//! and projection.

use std::collections::HashMap;
use std::sync::Mutex;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{fp, Epi, Mat, Ops};
use crate::weights::{Config, Weights};

const FREQ: usize = 256;

/// A scalar's sinusoidal features through fc1, SiLU, fc2 (no biases), on the host
struct Scalar {
    fc1: Vec<f32>,
    fc2: Vec<f32>,
}

impl Scalar {
    fn embed(&self, v: f32, hidden: usize) -> Vec<f32> {
        let half = FREQ / 2;
        let feats: Vec<f32> = (0..FREQ).map(|i| {
            let f = (-(10000f64.ln()) * (i % half) as f64 / half as f64).exp() as f32;
            if i < half { (v * f).cos() } else { (v * f).sin() }
        }).collect();
        let mut a = vec![0f32; hidden];
        for (o, row) in a.iter_mut().zip(self.fc1.chunks_exact(FREQ)) {
            let s: f32 = row.iter().zip(&feats).map(|(w, x)| w * x).sum();
            *o = s / (1.0 + (-s).exp());
        }
        self.fc2.chunks_exact(hidden).map(|row| row.iter().zip(&a).map(|(w, x)| w * x).sum()).collect()
    }
}

struct HeadLayer {
    ada: Mat,
    norm: DevBuf,
    gate_up: Mat,
    down: Mat,
}

pub struct Head {
    hidden: usize,
    latent: usize,
    eps: f32,
    interval: f32,
    pub steps: usize,
    cond: Mat,
    latent_proj: Mat,
    layers: Vec<HeadLayer>,
    final_ada: Mat,
    proj: Mat,
    scalars: [Scalar; 4],
    /// the scalar embeddings' sum for each step, by guidance (its bits)
    consts: Mutex<HashMap<u32, Vec<DevBuf>>>,
    // work
    cv: DevBuf,
    c: DevBuf,
    z: DevBuf,
    x: DevBuf,
    ada: DevBuf,
    h: DevBuf,
    f: DevBuf,
    u: DevBuf,
}

impl Head {
    pub fn load(ops: &Ops, f: &Weights, cfg: &Config) -> Result<Head> {
        let h = cfg.head_hidden;
        if h != cfg.hidden {
            return Err(Error(format!("head.hidden_size {h}: the language model's {} expected", cfg.hidden)));
        }
        let mut layers = Vec::new();
        for l in 0..cfg.head_layers {
            let n = |s: &str| format!("head.layers.{l}.{s}");
            layers.push(HeadLayer {
                ada: f.mat(ops, &[&n("ada.weight")], None, false)?,
                norm: f.vec(ops, &n("norm.weight"), 0.0)?,
                gate_up: f.mat(ops, &[&n("ffn.gate_proj.weight"), &n("ffn.up_proj.weight")], None, true)?,
                down: f.mat(ops, &[&n("ffn.down_proj.weight")], None, false)?,
            });
        }
        let ffn = layers.first().map_or(h, |l| l.down.k);
        let scalar = |p: &str| -> Result<Scalar> {
            let (a, b) = (format!("head.{p}.fc1.weight"), format!("head.{p}.fc2.weight"));
            if f.shape(&a)? != [h, FREQ] || f.shape(&b)? != [h, h] {
                return Err(Error(format!("head.{p}: [{h}, {FREQ}] and [{h}, {h}] expected")));
            }
            Ok(Scalar { fc1: f.all(&a)?, fc2: f.all(&b)? })
        };
        let g = &ops.gpu;
        Ok(Head {
            hidden: h,
            latent: cfg.latent,
            eps: cfg.head_eps,
            interval: cfg.interval,
            steps: cfg.head_steps,
            cond: f.mat(ops, &["head.cond_proj.weight"], None, false)?,
            latent_proj: f.mat(ops, &["head.latent_proj.weight"], None, false)?,
            final_ada: f.mat(ops, &["head.final.ada.weight"], None, false)?,
            proj: f.mat(ops, &["head.final.proj.weight"], None, false)?,
            layers,
            scalars: [scalar("t_embed")?, scalar("dt_embed")?, scalar("h_embed")?, scalar("omega_embed")?],
            consts: Mutex::new(HashMap::new()),
            cv: DevBuf::f32(g, h)?,
            c: DevBuf::f32(g, h)?,
            z: DevBuf::f32(g, cfg.latent.max(4))?,
            x: DevBuf::f32(g, h)?,
            ada: DevBuf::f32(g, 3 * h)?,
            h: DevBuf::f32(g, h)?,
            f: DevBuf::f32(g, ffn)?,
            u: DevBuf::f32(g, cfg.latent.max(4))?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.cond.bytes() + self.latent_proj.bytes() + self.final_ada.bytes() + self.proj.bytes()
            + self.layers.iter().map(|l| l.ada.bytes() + l.gate_up.bytes() + l.down.bytes()).sum::<usize>()
    }

    /// The scalar embeddings' sum for each step at guidance `w` (t = i / steps, the interval constant, h = 1 / steps,
    /// omega = 1 - 1 / w), on the device
    fn consts(&self, ops: &Ops, w: f32) -> Result<Vec<DevBuf>> {
        let mut m = self.consts.lock().map_err(|_| Error("head lock".into()))?;
        if let Some(v) = m.get(&w.to_bits()) {
            return v.iter().map(|b| b.view(0, b.len)).collect();
        }
        let hstep = 1.0 / self.steps as f32;
        let mut v = Vec::new();
        for i in 0..self.steps {
            let parts = [self.scalars[0].embed(i as f32 * hstep, self.hidden), self.scalars[1].embed(self.interval, self.hidden),
                         self.scalars[2].embed(hstep, self.hidden), self.scalars[3].embed(1.0 - 1.0 / w, self.hidden)];
            let sum: Vec<f32> = (0..self.hidden).map(|j| parts.iter().map(|p| p[j]).sum()).collect();
            v.push(DevBuf::from_f32(&ops.gpu, &sum)?);
        }
        let out = v.iter().map(|b| b.view(0, b.len)).collect::<Result<Vec<_>>>()?;
        m.insert(w.to_bits(), v);
        Ok(out)
    }

    /// The normalized latent [latent] from the hidden state at `hidden` (device, `hidden` floats) and `noise`
    /// (z at t = 0), at guidance `w`
    pub fn sample(&self, ops: &Ops, hidden: *const f32, noise: &[f32], w: f32) -> Result<Vec<f32>> {
        let (hd, l) = (self.hidden, self.latent);
        let consts = self.consts(ops, w)?;
        ops.gemv(hidden, 1, hd, &self.cond, self.cv.fp(), hd, Epi::Store, None)?;
        let mut z = noise.to_vec();
        let hstep = 1.0 / self.steps as f32;
        for cst in &consts {
            self.c.copy_within(0, &self.cv, 0, hd * 4)?;
            ops.add(self.c.fp(), cst.fp(), hd)?;
            ops.silu(self.c.fp(), hd)?;
            self.z.write(0, &crate::weights::bytes(&z))?;
            ops.gemv(self.z.fp(), 1, l, &self.latent_proj, self.x.fp(), hd, Epi::Store, None)?;
            for ly in &self.layers {
                ops.gemv(self.c.fp(), 1, hd, &ly.ada, self.ada.fp(), 3 * hd, Epi::Store, None)?;
                ops.rms_norm(self.x.fp(), 1, hd, Some(&ly.norm), self.eps, Some((fp(&self.ada, hd), self.ada.fp(), 0)), self.h.fp())?;
                ops.gemv(self.h.fp(), 1, hd, &ly.gate_up, self.f.fp(), ly.down.k, Epi::Swiglu, None)?;
                ops.gemv(self.f.fp(), 1, ly.down.k, &ly.down, self.x.fp(), hd, Epi::Res, Some((fp(&self.ada, 2 * hd), 0)))?;
            }
            ops.gemv(self.c.fp(), 1, hd, &self.final_ada, self.ada.fp(), 2 * hd, Epi::Store, None)?;
            ops.rms_norm(self.x.fp(), 1, hd, None, self.eps, Some((fp(&self.ada, hd), self.ada.fp(), 0)), self.h.fp())?;
            ops.gemv(self.h.fp(), 1, hd, &self.proj, self.u.fp(), l, Epi::Store, None)?;
            let u = self.u.to_f32()?;
            for (a, b) in z.iter_mut().zip(&u) {
                *a += hstep * b;
            }
        }
        ops.mark(|| "head".into())?;
        Ok(z)
    }
}
