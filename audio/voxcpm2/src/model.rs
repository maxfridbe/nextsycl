//! VoxCPM2's generator (voxcpm's model/voxcpm2.py `_inference`): no audio tokens - each step makes a patch of 4
//! AudioVAE latents (64 wide, 25 a second) by flow matching, conditioned on two language models.
//!
//! ```text
//!   prompt rows: text embeddings (text positions) or the feature encoder's embedding of a patch (audio positions)
//!   base LM (28 layers) -> its output, through the scalar quantizer at audio positions        lm = the last row's
//!   residual LM (8 layers, no positions) over fusion(cat(that, the audio rows' embedding))     res = the last row's
//!   a patch:  DiT (12 layers of 1,024) over [mu_lm, mu_res, t, the previous patch (4), x (4)], 10 Euler steps from
//!             noise, guided 2.0 against mu = 0 (CFG-zero*: the first step none, the scale fitted to each); the stop
//!             head (on lm) says when to end; the patch through the feature encoder (12 layers over [special, 4
//!             latents]) back into both language models
//! ```

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::lm::{long_rope, Cache, Lm};
use crate::ops::{fp, nul, Mat, Ops, Shards};
use crate::talker_bytes as bytes;

/// latents a patch, and their width
pub const PATCH: usize = 4;
pub const FEAT: usize = 64;

/// A float32 linear layer (+ bias), on oneDNN at any row count (the projections whose width the small-batch product
/// does not take)
pub struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    fn load(ops: &Ops, f: &Shards, name: &str, bias: bool) -> Result<Lin> {
        let s = f.shape(&format!("{name}.weight"))?;
        Ok(Lin {
            w: f.dev_f32(ops, &format!("{name}.weight"))?,
            b: if bias { Some(f.dev_f32(ops, &format!("{name}.bias"))?) } else { None },
            n: s[0],
            k: s[1..].iter().product(),
        })
    }

    pub fn apply(&self, nsd: &Nsd, x: *const f32, m: usize, out: *mut f32) -> Result<()> {
        nsd.linear(x.cast(), Dt::F32, m, self.k, self.w.ptr(), self.n, self.b.as_ref().map_or(nul(), |b| b.ptr()), out.cast(), Dt::F32)
    }
}

/// A half matrix with its bias (a projection the small-batch product takes)
pub struct Proj {
    m: Mat,
    b: Option<DevBuf>,
}

impl Proj {
    fn load(ops: &Ops, f: &Shards, name: &str, bias: bool) -> Result<Proj> {
        Ok(Proj { m: f.mat1(ops, &format!("{name}.weight"), false)?, b: if bias { Some(f.dev_f32(ops, &format!("{name}.bias"))?) } else { None } })
    }

    fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32) -> Result<()> {
        self.m.apply(ops, nsd, x, m, Some(xh), None, self.b.as_ref().map_or(nul(), |b| b.ptr()), out)
    }
}

pub struct Model {
    pub base: Lm,
    pub residual: Lm,
    enc: Lm,
    dit: Lm,
    /// the feature encoder's input projection and its special (first) row
    enc_in: Lin,
    enc_special: Vec<f32>,
    enc_to_lm: Proj,
    lm_to_dit: Proj,
    res_to_dit: Proj,
    fusion: Proj,
    fsq_in: Proj,
    fsq_out: Proj,
    fsq_scale: f32,
    stop_proj: Proj,
    stop_head: Proj,
    dit_in: Lin,
    dit_cond: Lin,
    dit_out: Proj,
    time1: Proj,
    time2: Proj,
    /// the delta-time embedding (dt is 0 outside mean mode): constant, made at load
    dt_emb: Vec<f32>,
    pub vocab: usize,
}

/// A generation's buffers
pub struct Session {
    pub base: Cache,
    pub res: Cache,
    enc: Cache,
    dit: Cache,
    /// [rows, 2,048] work: the prompt's rows, a step's
    a: DevBuf,
    b: DevBuf,
    c: DevBuf,
    xh: DevBuf,
    /// the last rows of the language models: lm (after the quantizer for audio), res
    pub lm: DevBuf,
    pub res_h: DevBuf,
    /// mu: [lm_to_dit(lm), res_to_dit(res)]
    mu: DevBuf,
    small: DevBuf,
    rows: usize,
}

fn sinusoid(dim: usize, t: f32) -> Vec<f32> {
    let half = dim / 2;
    let e = (10000f32).ln() / (half as f32 - 1.0);
    let f: Vec<f32> = (0..half).map(|i| 1000.0 * t * (-(i as f32) * e).exp()).collect();
    f.iter().map(|x| x.sin()).chain(f.iter().map(|x| x.cos())).collect()
}

/// The sampler's times: 1 to 0 in n steps, swayed (coefficient 1)
pub fn t_span(n: usize) -> Vec<f32> {
    (0..=n).map(|i| {
        let t = 1.0 - i as f32 / n as f32;
        t + ((std::f32::consts::PI / 2.0 * t).cos() - 1.0 + t)
    }).collect()
}

impl Model {
    pub fn load(ops: &Ops, nsd: &Nsd, f: &Shards, conf: &serde_json::Value, int8: bool, log: &mut dyn FnMut(String)) -> Result<Model> {
        let t0 = std::time::Instant::now();
        let lc = &conf["lm_config"];
        if lc["use_mup"] == true {
            return Err(Error("use_mup checkpoints are not supported".into()));
        }
        let eps = lc["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let theta = lc["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        let (maxp, orig) = (lc["max_position_embeddings"].as_u64().unwrap_or(0), lc["rope_scaling"]["original_max_position_embeddings"].as_u64().unwrap_or(0));
        let key = if maxp > orig { "long_factor" } else { "short_factor" };
        let factors: Vec<f32> = lc["rope_scaling"][key].as_array().into_iter().flatten().filter_map(|v| v.as_f64().map(|x| x as f32)).collect();
        if maxp != orig && orig != 0 {
            return Err(Error("long-RoPE with an attention scale (max positions past the original) is not supported".into()));
        }
        let rope = long_rope(theta, 128, &factors);
        let base = Lm::load(ops, f, "base_lm.", eps, Some(&rope), int8)?;
        let residual = Lm::load(ops, f, "residual_lm.", eps, if conf["residual_lm_no_rope"] == true { None } else { Some(&rope) }, int8)?;
        log(format!("base LM: {} layers of {} ({} / {} heads); residual LM: {} layers, in {:.1} s", base.layers(), base.hidden, base.heads, base.kv,
                    residual.layers(), t0.elapsed().as_secs_f64()));
        let enc = Lm::load(ops, f, "feat_encoder.encoder.", eps, Some(&rope), false)?;
        let dit = Lm::load(ops, f, "feat_decoder.estimator.decoder.", eps, Some(&rope), false)?;
        let p = |n: &str, b: bool| Proj::load(ops, f, n, b);
        let e = "feat_decoder.estimator.";
        let time1 = p(&format!("{e}time_mlp.linear_1"), true)?;
        let time2 = p(&format!("{e}time_mlp.linear_2"), true)?;
        // the delta-time MLP of t = 0, once
        let d1 = p(&format!("{e}delta_time_mlp.linear_1"), true)?;
        let d2 = p(&format!("{e}delta_time_mlp.linear_2"), true)?;
        let h = dit.hidden;
        let xh = DevBuf::new(&ops.gpu, 4 * h * 2)?;
        let (s0, s1) = (DevBuf::from_f32(&ops.gpu, &sinusoid(h, 0.0))?, DevBuf::f32(&ops.gpu, h)?);
        d1.apply(ops, nsd, s0.fp(), 1, &xh, s1.fp())?;
        ops.silu(s1.fp(), h)?;
        d2.apply(ops, nsd, s1.fp(), 1, &xh, s0.fp())?;
        let dt_emb = s0.to_f32()?;
        let m = Model {
            enc_in: Lin::load(ops, f, "feat_encoder.in_proj", true)?,
            enc_special: f.f32("feat_encoder.special_token")?,
            enc_to_lm: p("enc_to_lm_proj", true)?,
            lm_to_dit: p("lm_to_dit_proj", true)?,
            res_to_dit: p("res_to_dit_proj", true)?,
            fusion: p("fusion_concat_proj", true)?,
            fsq_in: p("fsq_layer.in_proj", true)?,
            fsq_out: p("fsq_layer.out_proj", true)?,
            fsq_scale: conf["scalar_quantization_scale"].as_f64().unwrap_or(9.0) as f32,
            stop_proj: p("stop_proj", true)?,
            stop_head: p("stop_head", false)?,
            dit_in: Lin::load(ops, f, &format!("{e}in_proj"), true)?,
            dit_cond: Lin::load(ops, f, &format!("{e}cond_proj"), true)?,
            dit_out: p(&format!("{e}out_proj"), true)?,
            time1,
            time2,
            dt_emb,
            vocab: f.shape("base_lm.embed_tokens.weight")?[0],
            base,
            residual,
            enc,
            dit,
        };
        ops.gpu.sync()?;
        log(format!("local encoder and DiT: {} / {} layers of {}; all in {:.1} s", m.enc.layers(), m.dit.layers(), m.dit.hidden, t0.elapsed().as_secs_f64()));
        Ok(m)
    }

    /// Buffers for a prompt of `prompt` rows (its audio patches among them) and up to `patches` steps
    pub fn session(&self, ops: &Ops, prompt: usize, patches: usize) -> Result<Session> {
        let g = &ops.gpu;
        let h = self.base.hidden;
        let rows = prompt.max(2);
        let t = prompt + patches + 2;
        Ok(Session {
            base: self.base.cache(ops, rows, t)?,
            res: self.residual.cache(ops, rows, t)?,
            enc: self.enc.cache(ops, rows.max(8) * (PATCH + 1), 0)?,
            dit: self.dit.cache(ops, 2 * (2 * PATCH + 3), 0)?,
            a: DevBuf::f32(g, rows * 2 * h)?,
            b: DevBuf::f32(g, rows * 2 * h)?,
            c: DevBuf::f32(g, rows * 2 * h)?,
            xh: DevBuf::new(g, rows * 2 * h * 2)?,
            lm: DevBuf::f32(g, h)?,
            res_h: DevBuf::f32(g, h)?,
            mu: DevBuf::f32(g, 2 * self.dit.hidden)?,
            small: DevBuf::f32(g, 4 * h)?,
            rows,
        })
    }

    /// Patches [n][4 x 64] (host) through the feature encoder and enc_to_lm: [n, 2,048] into `out`
    pub fn encode_patches(&self, ops: &Ops, nsd: &Nsd, s: &Session, patches: &[f32], out: *mut f32) -> Result<()> {
        let eh = self.enc.hidden;
        let per = s.rows.max(8);
        for (ci, chunk) in patches.chunks(per * PATCH * FEAT).enumerate() {
            let m = chunk.len() / (PATCH * FEAT);
            // the latents' rows projected, each patch behind the special row: [m x 5, 1,024]
            let lat = DevBuf::from_f32(&ops.gpu, chunk)?;
            let proj = DevBuf::f32(&ops.gpu, m * PATCH * eh)?;
            self.enc_in.apply(nsd, lat.fp(), m * PATCH, proj.fp())?;
            for i in 0..m {
                s.enc.x.write(i * (PATCH + 1) * eh * 4, &bytes(&self.enc_special))?;
                s.enc.x.copy_within((i * (PATCH + 1) + 1) * eh * 4, &proj, i * PATCH * eh * 4, PATCH * eh * 4)?;
            }
            self.enc.local(ops, nsd, &s.enc, m * (PATCH + 1), PATCH + 1)?;
            // the first row of each, through enc_to_lm
            let cls = DevBuf::f32(&ops.gpu, m * eh)?;
            for i in 0..m {
                cls.copy_within(i * eh * 4, &s.enc.out, i * (PATCH + 1) * eh * 4, eh * 4)?;
            }
            self.enc_to_lm.apply(ops, nsd, cls.fp(), m, &s.xh, fp_at(out, ci * per * self.base.hidden))?;
        }
        Ok(())
    }

    /// The scalar quantizer over n rows of `x` (in place, through s.b)
    fn fsq(&self, ops: &Ops, nsd: &Nsd, s: &Session, x: *mut f32, n: usize) -> Result<()> {
        let d = self.fsq_in.m.n;
        self.fsq_in.apply(ops, nsd, x, n, &s.xh, s.b.fp())?;
        ops.fsq(s.b.fp(), n * d, self.fsq_scale)?;
        self.fsq_out.apply(ops, nsd, s.b.fp(), n, &s.xh, x)
    }

    /// The prompt: rows [L, 2,048] (text embeddings at text positions, zero elsewhere), the audio positions' patches
    /// and mask; the language models filled, lm / res their last rows
    pub fn prefill(&self, ops: &Ops, nsd: &Nsd, s: &Session, text_rows: &[f32], audio: &[bool], patches: &[Vec<f32>]) -> Result<()> {
        let h = self.base.hidden;
        let l = audio.len();
        if l > s.rows {
            return Err(Error(format!("a prompt of {l} rows; the session takes {}", s.rows)));
        }
        // the audio rows' embeddings into c (zero elsewhere), and added to the text rows in a
        s.a.write(0, &bytes(text_rows))?;
        s.c.write(0, &bytes(&vec![0f32; l * h]))?;
        let idx: Vec<usize> = (0..l).filter(|i| audio[*i]).collect();
        if !idx.is_empty() {
            let flat: Vec<f32> = idx.iter().flat_map(|i| patches[*i].iter().copied()).collect();
            self.encode_patches(ops, nsd, s, &flat, s.b.fp())?;
            for (k, i) in idx.iter().enumerate() {
                s.c.copy_within(i * h * 4, &s.b, k * h * 4, h * 4)?;
            }
            ops.add(s.a.fp(), s.c.fp(), l * h)?;
        }
        s.base.x.copy_within(0, &s.a, 0, l * h * 4)?;
        self.base.pass(ops, nsd, &s.base, l, 0)?;
        // its output: quantized at audio positions
        s.a.copy_within(0, &s.base.out, 0, l * h * 4)?;
        for run in runs(audio) {
            self.fsq(ops, nsd, s, fp(&s.a, run.0 * h), run.1 - run.0)?;
        }
        s.lm.copy_within(0, &s.a, (l - 1) * h * 4, h * 4)?;
        // the residual LM over fusion(cat(enc_outputs, audio * feat_embed))
        let cat = DevBuf::f32(&ops.gpu, l * 2 * h)?;
        for i in 0..l {
            cat.copy_within(i * 2 * h * 4, &s.a, i * h * 4, h * 4)?;
            cat.copy_within((i * 2 + 1) * h * 4, &s.c, i * h * 4, h * 4)?;
        }
        self.fusion.apply(ops, nsd, cat.fp(), l, &s.xh, s.res.x.fp())?;
        self.residual.pass(ops, nsd, &s.res, l, 0)?;
        s.res_h.copy_within(0, &s.res.out, (l - 1) * h * 4, h * 4)?;
        ops.gpu.sync()
    }

    /// Whether the stop head says to end (on lm)
    pub fn stop(&self, ops: &Ops, nsd: &Nsd, s: &Session) -> Result<bool> {
        let h = self.base.hidden;
        self.stop_proj.apply(ops, nsd, s.lm.fp(), 1, &s.xh, s.small.fp())?;
        ops.silu(s.small.fp(), h)?;
        self.stop_head.apply(ops, nsd, s.small.fp(), 1, &s.xh, fp(&s.small, h))?;
        let v = s.small.to_f32()?;
        Ok(v[h + 1] > v[h])
    }

    /// mu for the DiT from lm and res
    pub fn mu(&self, ops: &Ops, nsd: &Nsd, s: &Session) -> Result<()> {
        let d = self.dit.hidden;
        self.lm_to_dit.apply(ops, nsd, s.lm.fp(), 1, &s.xh, s.mu.fp())?;
        self.res_to_dit.apply(ops, nsd, s.res_h.fp(), 1, &s.xh, fp(&s.mu, d))
    }

    /// The DiT's velocities at time t for x [64, 4] (host, channels first) and the previous patch `cond` [64, 4]: the
    /// guided (mu) and unguided (mu = 0) ones, each [64, 4]
    fn velocity(&self, ops: &Ops, nsd: &Nsd, s: &Session, x: &[f32], cond_rows: &DevBuf, t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.dit.hidden;
        let seq = 2 * PATCH + 3;
        // the time row: time_mlp(sinusoid(t)) + the delta-time one
        let te = DevBuf::from_f32(&ops.gpu, &sinusoid(d, t))?;
        let t1 = DevBuf::f32(&ops.gpu, d)?;
        self.time1.apply(ops, nsd, te.fp(), 1, &s.xh, t1.fp())?;
        ops.silu(t1.fp(), d)?;
        self.time2.apply(ops, nsd, t1.fp(), 1, &s.xh, te.fp())?;
        ops.add(te.fp(), DevBuf::from_f32(&ops.gpu, &self.dt_emb)?.fp(), d)?;
        // x's rows [4, 64] projected
        let xr: Vec<f32> = (0..PATCH).flat_map(|p| (0..FEAT).map(move |c| x[c * PATCH + p])).collect();
        let xd = DevBuf::from_f32(&ops.gpu, &xr)?;
        let xp = DevBuf::f32(&ops.gpu, PATCH * d)?;
        self.dit_in.apply(nsd, xd.fp(), PATCH, xp.fp())?;
        // [mu_lm, mu_res, t, cond x4, x x4] twice: guided, then mu = 0
        let x0 = &s.dit.x;
        for b in 0..2 {
            let r0 = b * seq * d * 4;
            if b == 0 {
                x0.copy_within(r0, &s.mu, 0, 2 * d * 4)?;
            } else {
                x0.write(r0, &bytes(&vec![0f32; 2 * d]))?;
            }
            x0.copy_within(r0 + 2 * d * 4, &te, 0, d * 4)?;
            x0.copy_within(r0 + 3 * d * 4, cond_rows, 0, PATCH * d * 4)?;
            x0.copy_within(r0 + (3 + PATCH) * d * 4, &xp, 0, PATCH * d * 4)?;
        }
        self.dit.local(ops, nsd, &s.dit, 2 * seq, seq)?;
        // the last 4 rows of each through out_proj
        let o = DevBuf::f32(&ops.gpu, 2 * PATCH * d)?;
        for b in 0..2 {
            o.copy_within(b * PATCH * d * 4, &s.dit.out, (b * seq + 3 + PATCH) * d * 4, PATCH * d * 4)?;
        }
        let v = DevBuf::f32(&ops.gpu, 2 * PATCH * FEAT)?;
        self.dit_out.apply(ops, nsd, o.fp(), 2 * PATCH, &s.xh, v.fp())?;
        let v = v.to_f32()?;
        // rows [4, 64] -> channels first [64, 4]
        let cf = |b: usize| -> Vec<f32> { (0..FEAT).flat_map(|c| (0..PATCH).map(move |p| (b, c, p))).map(|(b, c, p)| v[(b * PATCH + p) * FEAT + c]).collect() };
        Ok((cf(0), cf(1)))
    }

    /// A patch [4 x 64] (rows) by flow matching from `noise` [64, 4], conditioned on mu (s) and the previous patch
    /// `prev` [4 x 64] (rows)
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, ops: &Ops, nsd: &Nsd, s: &Session, noise: &[f32], prev: &[f32], steps: usize, cfg: f32) -> Result<Vec<f32>> {
        let d = self.dit.hidden;
        let cd = DevBuf::from_f32(&ops.gpu, prev)?;
        let cond_rows = DevBuf::f32(&ops.gpu, PATCH * d)?;
        self.dit_cond.apply(nsd, cd.fp(), PATCH, cond_rows.fp())?;
        let ts = t_span(steps);
        let mut x = noise.to_vec();
        let (mut t, mut dt) = (ts[0], ts[0] - ts[1]);
        let zero_init = ((ts.len() as f32 * 0.04) as usize).max(1);
        for step in 1..ts.len() {
            if step > zero_init {
                let (pos, neg) = self.velocity(ops, nsd, s, &x, &cond_rows, t)?;
                // CFG-zero*: the unguided velocity scaled to fit the guided one
                let dot: f32 = pos.iter().zip(&neg).map(|(a, b)| a * b).sum();
                let nn: f32 = neg.iter().map(|b| b * b).sum::<f32>() + 1e-8;
                let st = dot / nn;
                for ((xi, p), n) in x.iter_mut().zip(&pos).zip(&neg) {
                    let v = n * st + cfg * (p - n * st);
                    *xi -= dt * v;
                }
            }
            t -= dt;
            if step < ts.len() - 1 {
                dt = t - ts[step + 1];
            }
        }
        // channels first [64, 4] -> rows [4, 64]
        Ok((0..PATCH).flat_map(|p| (0..FEAT).map(move |c| (c, p))).map(|(c, p)| x[c * PATCH + p]).collect())
    }

    /// A made patch [4 x 64] fed back: through the feature encoder, the base LM a step (then the quantizer), the
    /// residual LM a step over fusion(cat(lm, the patch's embedding)); `pos` its position
    pub fn advance(&self, ops: &Ops, nsd: &Nsd, s: &Session, patch: &[f32], pos: usize) -> Result<()> {
        let h = self.base.hidden;
        self.encode_patches(ops, nsd, s, patch, s.c.fp())?;
        s.base.x.copy_within(0, &s.c, 0, h * 4)?;
        self.base.pass(ops, nsd, &s.base, 1, pos)?;
        s.lm.copy_within(0, &s.base.out, 0, h * 4)?;
        self.fsq(ops, nsd, s, s.lm.fp(), 1)?;
        s.a.copy_within(0, &s.lm, 0, h * 4)?;
        s.a.copy_within(h * 4, &s.c, 0, h * 4)?;
        self.fusion.apply(ops, nsd, s.a.fp(), 1, &s.xh, s.res.x.fp())?;
        self.residual.pass(ops, nsd, &s.res, 1, pos)?;
        s.res_h.copy_within(0, &s.res.out, 0, h * 4)
    }
}

/// A pointer `at` floats past `p`
fn fp_at(p: *mut f32, at: usize) -> *mut f32 {
    p.wrapping_add(at)
}

/// The runs of true in a mask: [start, end)
fn runs(m: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < m.len() {
        if m[i] {
            let s = i;
            while i < m.len() && m[i] {
                i += 1;
            }
            out.push((s, i));
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_runs_and_the_sinusoid() {
        let t = t_span(10);
        assert_eq!(t.len(), 11);
        assert!((t[0] - 1.0).abs() < 1e-6 && t[10].abs() < 1e-6);
        // swayed: t + cos(pi/2 t) - 1 + t, at 0.5: 1 + 0.7071 - 1 = 0.7071
        assert!((t[5] - 0.70710677).abs() < 1e-5);
        assert_eq!(runs(&[false, true, true, false, true]), vec![(1, 3), (4, 5)]);
        let s = sinusoid(8, 0.0);
        assert_eq!(s, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
