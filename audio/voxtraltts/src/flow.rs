//! The acoustic transformer (upstream's FlowMatchingAudioTransformer): from a frame's hidden state, the semantic code
//! (its head's argmax over 8,192 codes and the two special ids) and the 36 acoustic values by flow matching - an Euler
//! ODE from Gaussian noise over `steps` steps (7 upstream), each step one pass of a 3-layer bidirectional transformer
//! (3,072 wide, 32 / 8 heads of 128, SwiGLU 9,216, RMS ε 1e-5, no positions) over three rows - the projected values,
//! the projected time embedding, the projected hidden state - read at the first row, guided against a pass whose
//! hidden state is zero: v = cfg v_cond + (1 - cfg) v_uncond. Both passes run as one batch of six rows; the whole ODE
//! stays on the GPU (the input projection's 36 columns padded to 256 for the small-batch product).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

/// The acoustic values a frame (and FSQ levels each); the semantic head's special ids ahead of its codes
pub const ACOUSTIC: usize = 36;
pub const LEVELS: usize = 21;
pub const SPECIAL: usize = 2;
/// [EMPTY_AUDIO] and [END_AUDIO] (upstream's AudioSpecialTokens)
pub const EMPTY: usize = 0;
pub const END: usize = 1;
/// the input projection's columns, padded for the product (a multiple of 256)
const PAD: usize = 256;
const HEAD: usize = 128;

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Flow {
    layers: Vec<Layer>,
    norm: DevBuf,
    input: Mat,
    llm: Mat,
    time: Mat,
    out: Mat,
    sem: Mat,
    pub dim: usize,
    heads: usize,
    kv: usize,
    ffn: usize,
    eps: f32,
    /// the semantic codebook's size (8,192): the head's ids past it and the specials are padding
    pub codes: usize,
}

/// One request's work room
pub struct Work {
    x: DevBuf,
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    /// the values [PAD] (36 used), their input projection, the condition's projection, the velocities [6, 36]
    pub lat: DevBuf,
    inp: DevBuf,
    hp: DevBuf,
    vel: DevBuf,
    logits: DevBuf,
    /// the time projections of the schedule's steps [steps, dim] and the step sizes
    tproj: DevBuf,
    dts: Vec<f32>,
}

/// torch.linspace(0, 1, n) in float32 (its halves from either end)
pub fn linspace(n: usize) -> Vec<f32> {
    let step = 1.0f32 / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { i as f32 * step } else { 1.0 - (n - 1 - i) as f32 * step }).collect()
}

/// upstream's TimeEmbedding: [cos(t f_i), sin(t f_i)], f_i = exp(-ln(1e4) i / (d / 2))
pub fn time_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let f: Vec<f32> = (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
    f.iter().map(|f| (t * f).cos()).chain(f.iter().map(|f| (t * f).sin())).collect()
}

/// FSQ: a value clamped to [-1, 1], scaled to the 21 levels and rounded (half to even, as torch's round), plus the
/// special ids
pub fn quantize(lat: &[f32]) -> Vec<i32> {
    lat.iter().map(|v| {
        let s = (v.clamp(-1.0, 1.0) + 1.0) / 2.0 * (LEVELS - 1) as f32;
        s.round_ties_even() as i32 + SPECIAL as i32
    }).collect()
}

/// The semantic code: the head's argmax with [EMPTY_AUDIO] and the padding past the codebook masked ([END_AUDIO] allowed)
pub fn semantic(logits: &[f32], codes: usize) -> usize {
    let mut best = (END, f32::NEG_INFINITY);
    for (i, v) in logits.iter().enumerate().take(SPECIAL + codes) {
        if i != EMPTY && *v > best.1 {
            best = (i, *v);
        }
    }
    best.0
}

/// A host matrix [n, k] (k <= kp) as a half Mat [n, kp], zeros past k
fn padded(ops: &Ops, v: &[f32], n: usize, k: usize, kp: usize) -> Result<Mat> {
    let mut p = vec![0f32; n * kp];
    for r in 0..n {
        p[r * kp..r * kp + k].copy_from_slice(&v[r * k..(r + 1) * k]);
    }
    let stage = DevBuf::from_f32(&ops.gpu, &p)?;
    let w = DevBuf::new(&ops.gpu, n * kp * 2)?;
    ops.to_half(stage.fp(), w.ptr(), n * kp)?;
    ops.gpu.sync()?;
    Ok(Mat { w, n, k: kp })
}

impl Flow {
    pub fn load(ops: &Ops, f: &Weights) -> Result<Flow> {
        let mut layers = Vec::new();
        for l in 0.. {
            let n = |s: &str| format!("fm.blk.{l}.{s}.weight");
            if !f.has(&n("attn_norm")) {
                break;
            }
            layers.push(Layer {
                attn_norm: f.vec(ops, &n("attn_norm"))?,
                ffn_norm: f.vec(ops, &n("ffn_norm"))?,
                qkv: f.mat(ops, &[&n("attn_q"), &n("attn_k"), &n("attn_v")])?,
                o: f.mat(ops, &[&n("attn_output")])?,
                gu: f.mat(ops, &[&n("ffn_gate"), &n("ffn_up")])?,
                down: f.mat(ops, &[&n("ffn_down")])?,
            });
        }
        let Some(first) = layers.first() else { return Err(Error("no fm.blk.* layers in the file".into())) };
        let dim = first.o.n;
        let (heads, kv, ffn) = (first.o.k / HEAD, (first.qkv.n - first.o.k) / 2 / HEAD, first.gu.n / 2);
        let it = f.tensor("fm.input_proj.weight")?;
        if it.shape != [dim as u64, ACOUSTIC as u64] {
            return Err(Error(format!("fm.input_proj: {:?}, [{dim}, {ACOUSTIC}] expected", it.shape)));
        }
        let input = padded(ops, &f.all("fm.input_proj.weight")?, dim, ACOUSTIC, PAD)?;
        let out = f.mat(ops, &["fm.acoustic_output.weight"])?;
        let sem = f.mat(ops, &["fm.semantic_output.weight"])?;
        let codes = f.num("semantic_codebook_size")? as usize;
        if out.n != ACOUSTIC || sem.n < SPECIAL + codes || f.num("acoustic_codebook_size")? as usize != LEVELS || f.has("fm.semantic_output.bias") {
            return Err(Error(format!("the acoustic heads ({} values, {} semantic ids) are not this engine's ({ACOUSTIC} values of {LEVELS} levels, \
                                      {} semantic ids, no bias)", out.n, sem.n, SPECIAL + codes)));
        }
        Ok(Flow {
            norm: f.vec(ops, "fm.norm.weight")?,
            llm: f.mat(ops, &["fm.llm_proj.weight"])?,
            time: f.mat(ops, &["fm.time_proj.weight"])?,
            input,
            out,
            sem,
            layers,
            dim,
            heads,
            kv,
            ffn,
            // upstream's AcousticTransformerArgs.norm_eps (params.json names none)
            eps: 1e-5,
            codes,
        })
    }

    /// Work room for a request whose ODE takes `steps` Euler steps (the time projections made here)
    pub fn work(&self, ops: &Ops, steps: usize) -> Result<Work> {
        let g = &ops.gpu;
        let d = self.dim;
        if !(1..=64).contains(&steps) {
            return Err(Error(format!("{steps} flow steps: 1 to 64")));
        }
        let ts = linspace(steps + 1);
        let emb: Vec<f32> = ts[..steps].iter().flat_map(|t| time_embedding(*t, d)).collect();
        let e = DevBuf::from_f32(g, &emb)?;
        let tproj = DevBuf::f32(g, steps * d)?;
        for r0 in (0..steps).step_by(8) {
            let m = (steps - r0).min(8);
            ops.gemv(fp(&e, r0 * d), m, &self.time, fp(&tproj, r0 * d), d, false)?;
        }
        ops.gpu.sync()?;
        Ok(Work {
            x: DevBuf::f32(g, 6 * d)?,
            hb: DevBuf::f32(g, 6 * d)?,
            qkv: DevBuf::f32(g, 6 * (self.heads + 2 * self.kv) * HEAD)?,
            att: DevBuf::f32(g, 6 * self.heads * HEAD)?,
            gu: DevBuf::f32(g, 6 * 2 * self.ffn)?,
            act: DevBuf::f32(g, 6 * self.ffn)?,
            lat: DevBuf::f32(g, PAD)?,
            inp: DevBuf::f32(g, d)?,
            hp: DevBuf::f32(g, d)?,
            vel: DevBuf::f32(g, 6 * ACOUSTIC)?,
            logits: DevBuf::f32(g, self.sem.n)?,
            tproj,
            dts: ts.windows(2).map(|w| w[1] - w[0]).collect(),
        })
    }

    /// The semantic head's logits for a hidden state (device pointer, `dim` floats)
    pub fn semantic_logits(&self, ops: &Ops, w: &Work, h: *const f32) -> Result<Vec<f32>> {
        ops.gemv(h, 1, &self.sem, w.logits.fp(), self.sem.n, false)?;
        ops.gpu.sync()?;
        w.logits.to_f32()
    }

    #[allow(clippy::too_many_arguments)]
    /// The acoustic values for a hidden state (device) from `noise` [36]: the ODE's result before quantization;
    /// `vel` keeps each step's velocities [2][36] (a check's)
    pub fn acoustic(&self, ops: &Ops, nsd: &Nsd, w: &Work, h: *const f32, noise: &[f32], cfg: f32, mut vel: Option<&mut Vec<Vec<f32>>>)
                    -> Result<Vec<f32>> {
        let (d, q) = (self.dim, self.heads * HEAD);
        let qw = (self.heads + 2 * self.kv) * HEAD;
        let mut lat = vec![0f32; PAD];
        lat[..ACOUSTIC].copy_from_slice(&noise[..ACOUSTIC]);
        w.lat.write(0, &crate::bytes(&lat))?;
        ops.gemv(h, 1, &self.llm, w.hp.fp(), d, false)?;
        let scale = 1.0 / (HEAD as f32).sqrt();
        for (i, dt) in w.dts.iter().enumerate() {
            ops.gemv(w.lat.fp(), 1, &self.input, w.inp.fp(), d, false)?;
            ops.fm_rows(w.inp.fp(), fp(&w.tproj, i * d), w.hp.fp(), d, w.x.fp())?;
            for l in &self.layers {
                nsd.rms_norm_mod(w.x.ptr(), Dt::F32, 6, d, l.attn_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
                ops.gemv(w.hb.fp(), 6, &l.qkv, w.qkv.fp(), qw, false)?;
                // the two passes' three rows each attend among themselves
                for b in 0..2 {
                    let base = fp(&w.qkv, 3 * b * qw);
                    ops.attn_window(base, base.wrapping_add(q), base.wrapping_add(q + self.kv * HEAD), 3, self.heads, self.kv, HEAD, qw, qw, 3, 3, scale,
                                    fp(&w.att, 3 * b * q))?;
                }
                ops.gemv(w.att.fp(), 6, &l.o, w.x.fp(), d, true)?;
                nsd.rms_norm_mod(w.x.ptr(), Dt::F32, 6, d, l.ffn_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
                ops.gemv(w.hb.fp(), 6, &l.gu, w.gu.fp(), 2 * self.ffn, false)?;
                nsd.swiglu(w.gu.ptr(), Dt::F32, 6, self.ffn, w.act.ptr(), Dt::F32)?;
                ops.gemv(w.act.fp(), 6, &l.down, w.x.fp(), d, true)?;
            }
            nsd.rms_norm_mod(w.x.ptr(), Dt::F32, 6, d, self.norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
            ops.gemv(w.hb.fp(), 6, &self.out, w.vel.fp(), ACOUSTIC, false)?;
            if let Some(v) = vel.as_deref_mut() {
                ops.gpu.sync()?;
                let all = w.vel.to_f32()?;
                v.push(all[..ACOUSTIC].iter().chain(&all[3 * ACOUSTIC..4 * ACOUSTIC]).copied().collect());
            }
            // the conditioned pass's first row is row 0, the other's row 3
            ops.cfg_step(w.lat.fp(), w.vel.fp(), fp(&w.vel, 3 * ACOUSTIC), ACOUSTIC, cfg, *dt)?;
        }
        ops.gpu.sync()?;
        Ok(w.lat.to_f32()?[..ACOUSTIC].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_levels_and_the_semantic_mask() {
        let t = linspace(8);
        assert_eq!((t[0], t[7], t.len()), (0.0, 1.0, 8));
        assert!((t[1] - 1.0 / 7.0).abs() < 1e-7);
        let e = time_embedding(0.0, 8);
        assert_eq!(e, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // -1 -> level 0, 1 -> 20, 0 -> 10, clamped beyond; + 2 specials; ties (7.5, 12.5) to even as torch
        assert_eq!(quantize(&[-1.0, 1.0, 0.0, 3.0, -0.25, 0.25]), vec![2, 22, 12, 22, 2 + 8, 2 + 12]);
        // [EMPTY] never, [END] may win, the padding past the codebook never
        assert_eq!(semantic(&[9.0, 1.0, 0.5, 9.5], 1), 1);
        assert_eq!(semantic(&[9.0, 1.0, 2.0, 9.5], 1), 2);
    }
}
