//! The language model: Qwen3.5-4B's text layers (transformers' Qwen3_5TextModel), one position a frame. Three of every
//! four layers are Gated DeltaNet layers (a fused projection to queries / keys / values, the gate z and the heads'
//! write strength b and decay a; a causal convolution of 4 with SiLU; the gated delta rule's recurrence over a
//! [128 x 128] state a value head; a gated RMS norm; the output projection), the fourth full attention (q with an
//! output gate, k, v; per-head RMS norm and RoPE on the first quarter of each head; grouped-query attention over the
//! half cache; the gate's sigmoid). Every layer: zero-centred RMS norms (1 + w), a SwiGLU MLP, residuals.
//!
//! Rows go through in groups of up to `ROWS` (a frame: one; a prompt: 16 at a time). The products read the
//! checkpoint's bfloat16 matrices (vv.h's gemv), everything else is float32 on the device; a conversation's state (each
//! DeltaNet layer's convolution rows and recurrence, each attention layer's keys and values) lives in its `State`.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{fp, Epi, Mat, Ops};
use crate::weights::{Config, Weights};

/// The rows one pass takes at most (the products' small-batch form takes 20)
pub const ROWS: usize = 16;
/// Attention's keys are split into at most this many pieces
const PIECES: usize = 64;

enum Mixer {
    Delta {
        /// [qkv (conv_dim) | z (Hv * Dv) | b (Hv) | a (Hv)] x hidden
        proj: Mat,
        conv_w: DevBuf,
        a_log: DevBuf,
        dt_bias: DevBuf,
        norm: DevBuf,
        out: Mat,
    },
    Attn {
        /// [q with gate (Hq * 2D) | k (Hkv * D) | v (Hkv * D)] x hidden
        qkv: Mat,
        qn: DevBuf,
        kn: DevBuf,
        o: Mat,
    },
}

struct Layer {
    in_norm: DevBuf,
    post_norm: DevBuf,
    mixer: Mixer,
    /// gate / up interleaved (Epi::Swiglu)
    gate_up: Mat,
    down: Mat,
}

/// A conversation's place in the model: each DeltaNet layer's convolution rows and recurrence, each attention layer's
/// keys and values (half [Hkv, ctx, D]), and the positions taken
pub struct State {
    conv: Vec<DevBuf>,
    ssm: Vec<DevBuf>,
    kc: Vec<DevBuf>,
    vc: Vec<DevBuf>,
    pub pos: usize,
    pub ctx: usize,
}

impl State {
    pub fn bytes(&self) -> usize {
        self.conv.iter().chain(&self.ssm).chain(&self.kc).chain(&self.vc).map(|b| b.len).sum()
    }
}

/// The work buffers of a pass
struct Work {
    x: DevBuf,
    h: DevBuf,
    proj: DevBuf,
    conv: DevBuf,
    y: DevBuf,
    y2: DevBuf,
    q: DevBuf,
    f: DevBuf,
    part: DevBuf,
    ids: DevBuf,
    lat: DevBuf,
    t1: DevBuf,
    t2: DevBuf,
}

/// An audio stream's projection into the model: fc1, RMS norm, fc2
pub struct Projector {
    fc1: Mat,
    norm: DevBuf,
    fc2: Mat,
}

pub struct Lm {
    pub cfg: Config,
    layers: Vec<Layer>,
    /// the token table, also the output matrix (tied): bfloat16 [vocab, hidden]
    pub embed: Mat,
    norm: DevBuf,
    inv_freq: DevBuf,
    pub model_proj: Projector,
    pub user_proj: Projector,
    w: Work,
    /// the last row's normed hidden state, and the logits
    pub hidden: DevBuf,
    pub logits: DevBuf,
}

impl Lm {
    pub fn load(ops: &Ops, f: &Weights, cfg: &Config, log: &mut dyn FnMut(String)) -> Result<Lm> {
        let c = cfg;
        let (h, vd) = (c.hidden, c.lin_v_heads * c.lin_v_dim);
        if c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.heads / c.kv_heads > 4 || !c.heads.is_multiple_of(c.kv_heads) {
            return Err(Error(format!("haru: attention heads of {} (256 expected), DeltaNet heads of {} / {} (128), {} / {} query / key heads (at most 4 a key \
                                      head)", c.head_dim, c.lin_k_dim, c.lin_v_dim, c.heads, c.kv_heads)));
        }
        let mut layers = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let n = |s: &str| format!("lm.layers.{l}.{s}");
            let mixer = if c.full[l] {
                Mixer::Attn {
                    qkv: f.mat(ops, &[&n("self_attn.q_proj.weight"), &n("self_attn.k_proj.weight"), &n("self_attn.v_proj.weight")], None, false)?,
                    qn: f.vec(ops, &n("self_attn.q_norm.weight"), 1.0)?,
                    kn: f.vec(ops, &n("self_attn.k_norm.weight"), 1.0)?,
                    o: f.mat(ops, &[&n("self_attn.o_proj.weight")], None, false)?,
                }
            } else {
                let cw = n("linear_attn.conv1d.weight");
                let s = f.shape(&cw)?;
                if s != [c.conv_dim(), 1, c.conv_k] {
                    return Err(Error(format!("{cw}: shape {s:?}, [{}, 1, {}] expected", c.conv_dim(), c.conv_k)));
                }
                Mixer::Delta {
                    proj: f.mat(ops, &[&n("linear_attn.in_proj_qkv.weight"), &n("linear_attn.in_proj_z.weight"), &n("linear_attn.in_proj_b.weight"),
                                       &n("linear_attn.in_proj_a.weight")], None, false)?,
                    conv_w: f.vec(ops, &cw, 0.0)?,
                    a_log: f.vec(ops, &n("linear_attn.A_log"), 0.0)?,
                    dt_bias: f.vec(ops, &n("linear_attn.dt_bias"), 0.0)?,
                    norm: f.vec(ops, &n("linear_attn.norm.weight"), 0.0)?,
                    out: f.mat(ops, &[&n("linear_attn.out_proj.weight")], None, false)?,
                }
            };
            layers.push(Layer {
                in_norm: f.vec(ops, &n("input_layernorm.weight"), 1.0)?,
                post_norm: f.vec(ops, &n("post_attention_layernorm.weight"), 1.0)?,
                mixer,
                gate_up: f.mat(ops, &[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")], None, true)?,
                down: f.mat(ops, &[&n("mlp.down_proj.weight")], None, false)?,
            });
            if l % 8 == 7 {
                log(format!("haru: language model layers 0-{l} on {}", ops.gpu.name));
            }
        }
        let embed = f.mat(ops, &["lm.embed_tokens.weight"], None, false)?;
        if embed.n != c.vocab || embed.k != h {
            return Err(Error(format!("lm.embed_tokens.weight: [{}, {}], [{}, {h}] expected", embed.n, embed.k, c.vocab)));
        }
        let proj = |p: &str| -> Result<Projector> {
            Ok(Projector {
                fc1: f.mat(ops, &[&format!("{p}.fc1.weight")], Some(&[&format!("{p}.fc1.bias")]), false)?,
                norm: f.vec(ops, &format!("{p}.norm.weight"), 0.0)?,
                fc2: f.mat(ops, &[&format!("{p}.fc2.weight")], Some(&[&format!("{p}.fc2.bias")]), false)?,
            })
        };
        let half = c.rot / 2;
        let inv: Vec<f32> = (0..half).map(|i| (1.0 / c.theta.powf((2 * i) as f64 / c.rot as f64)) as f32).collect();
        let g = &ops.gpu;
        let qkv_n = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim;
        let proj_n = c.conv_dim() + vd + 2 * c.lin_v_heads;
        let w = Work {
            x: DevBuf::f32(g, ROWS * h)?,
            h: DevBuf::f32(g, ROWS * h)?,
            proj: DevBuf::f32(g, ROWS * proj_n.max(qkv_n))?,
            conv: DevBuf::f32(g, ROWS * c.conv_dim())?,
            y: DevBuf::f32(g, ROWS * vd.max(c.heads * c.head_dim))?,
            y2: DevBuf::f32(g, ROWS * vd.max(c.heads * c.head_dim))?,
            q: DevBuf::f32(g, ROWS * c.heads * c.head_dim)?,
            f: DevBuf::f32(g, ROWS * c.ffn)?,
            part: DevBuf::f32(g, ROWS * c.heads * PIECES * (c.head_dim + 2))?,
            ids: DevBuf::new(g, ROWS * 4)?,
            lat: DevBuf::f32(g, ROWS * c.latent)?,
            t1: DevBuf::f32(g, ROWS * h)?,
            t2: DevBuf::f32(g, ROWS * h)?,
        };
        Ok(Lm {
            cfg: c.clone(),
            layers,
            norm: f.vec(ops, "lm.norm.weight", 1.0)?,
            inv_freq: DevBuf::from_f32(g, &inv)?,
            model_proj: proj("model_audio_proj")?,
            user_proj: proj("user_audio_proj")?,
            hidden: DevBuf::f32(g, ROWS * h)?,
            logits: DevBuf::f32(g, embed.n)?,
            embed,
            w,
        })
    }

    /// Weights on the card (bytes)
    pub fn bytes(&self) -> usize {
        let mut b = self.embed.bytes();
        for l in &self.layers {
            b += l.gate_up.bytes() + l.down.bytes();
            b += match &l.mixer {
                Mixer::Delta { proj, out, conv_w, .. } => proj.bytes() + out.bytes() + conv_w.len,
                Mixer::Attn { qkv, o, .. } => qkv.bytes() + o.bytes(),
            };
        }
        b
    }

    /// The memory a conversation of `ctx` positions holds (bytes)
    pub fn state_bytes(cfg: &Config, ctx: usize) -> usize {
        let delta = cfg.full.iter().filter(|f| !**f).count();
        let full = cfg.layers - delta;
        delta * ((cfg.conv_k - 1) * cfg.conv_dim() * 4 + cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim * 4) + full * 2 * cfg.kv_heads * ctx * cfg.head_dim * 2
    }

    /// A fresh conversation's state of `ctx` positions (zeros)
    pub fn state(&self, ops: &Ops, ctx: usize) -> Result<State> {
        let c = &self.cfg;
        let g = &ops.gpu;
        let (mut conv, mut ssm, mut kc, mut vc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for full in &c.full {
            if *full {
                kc.push(DevBuf::new(g, c.kv_heads * ctx * c.head_dim * 2)?);
                vc.push(DevBuf::new(g, c.kv_heads * ctx * c.head_dim * 2)?);
            } else {
                conv.push(DevBuf::f32(g, (c.conv_k - 1) * c.conv_dim())?);
                ssm.push(DevBuf::f32(g, c.lin_v_heads * c.lin_k_dim * c.lin_v_dim)?);
            }
        }
        let s = State { conv, ssm, kc, vc, pos: 0, ctx };
        self.reset(&s)?;
        Ok(s)
    }

    /// Back to no positions (the recurrences' state zeros; the cache is overwritten as it fills)
    pub fn reset(&self, s: &State) -> Result<()> {
        s.conv.iter().chain(&s.ssm).try_for_each(|b| b.fill(0))
    }

    /// The input rows: `ids`' token rows, plus each audio stream's projection where it has one (`model` / `user`:
    /// (first row, scaled latents [n, latent]) - the latents already in the model's space, (raw + bias) * scale)
    pub fn embed_rows(&self, ops: &Ops, ids: &[u32], model: Option<(usize, &[f32])>, user: Option<(usize, &[f32])>) -> Result<()> {
        let r = ids.len();
        if r == 0 || r > ROWS {
            return Err(Error(format!("{r} rows (1 to {ROWS} a pass)")));
        }
        let c = &self.cfg;
        let ib: Vec<u8> = ids.iter().flat_map(|i| (*i as i32).to_le_bytes()).collect();
        self.w.ids.write(0, &ib)?;
        ops.embed(&self.embed.w, c.hidden, &self.w.ids, r, self.w.x.fp(), c.hidden)?;
        for (p, s) in [(&self.model_proj, model), (&self.user_proj, user)] {
            let Some((r0, lat)) = s else { continue };
            let n = lat.len() / c.latent;
            if n == 0 {
                continue;
            }
            if r0 + n > r {
                return Err(Error(format!("audio rows {r0}..{} of {r}", r0 + n)));
            }
            self.w.lat.write(0, &crate::weights::bytes(lat))?;
            ops.gemv(self.w.lat.fp(), n, c.latent, &p.fc1, self.w.t1.fp(), c.hidden, Epi::Store, None)?;
            ops.rms_norm(self.w.t1.fp(), n, c.hidden, Some(&p.norm), 1e-6, None, self.w.t2.fp())?;
            ops.gemv(self.w.t2.fp(), n, c.hidden, &p.fc2, fp(&self.w.x, r0 * c.hidden), c.hidden, Epi::Acc, None)?;
        }
        Ok(())
    }

    /// The input rows as given (`x` [r, hidden] float32 on the host: a check's own embeddings)
    pub fn set_rows(&self, x: &[f32]) -> Result<()> {
        self.w.x.write(0, &crate::weights::bytes(x))
    }

    /// The input rows (embed_rows / set_rows) through every layer at the state's position; the last row's normed
    /// hidden state into `hidden` (all rows' with `all`), the state one position on a row
    pub fn forward(&self, ops: &Ops, s: &mut State, r: usize, all: bool) -> Result<()> {
        self.forward_tap(ops, s, r, all, &mut |_, _| Ok(()))
    }

    /// `forward`, `tap` given each layer's residual rows after it (a check's view)
    pub fn forward_tap(&self, ops: &Ops, s: &mut State, r: usize, all: bool, tap: &mut dyn FnMut(usize, &DevBuf) -> Result<()>) -> Result<()> {
        let c = &self.cfg;
        let hd = c.hidden;
        if s.pos + r > s.ctx {
            return Err(Error(format!("the conversation is full ({} positions)", s.ctx)));
        }
        let w = &self.w;
        let (mut di, mut ai) = (0, 0);
        for (li, l) in self.layers.iter().enumerate() {
            ops.rms_norm(w.x.fp(), r, hd, Some(&l.in_norm), c.eps, None, w.h.fp())?;
            match &l.mixer {
                Mixer::Delta { proj, conv_w, a_log, dt_bias, norm, out } => {
                    let pn = proj.n;
                    let cd = c.conv_dim();
                    let vd = c.lin_v_heads * c.lin_v_dim;
                    ops.gemv(w.h.fp(), r, hd, proj, w.proj.fp(), pn, Epi::Store, None)?;
                    ops.gdn_conv(w.proj.fp(), pn, r, cd, c.conv_k, conv_w, &s.conv[di], w.conv.fp())?;
                    ops.gdn_rule(w.conv.fp(), cd, fp(&w.proj, cd + vd), pn, a_log, dt_bias, &s.ssm[di], r, c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim,
                                 w.y.fp())?;
                    ops.gated_norm(w.y.fp(), fp(&w.proj, cd), pn, norm, c.eps, r, c.lin_v_heads, c.lin_v_dim, w.y2.fp())?;
                    ops.gemv(w.y2.fp(), r, vd, out, w.x.fp(), hd, Epi::Acc, None)?;
                    di += 1;
                }
                Mixer::Attn { qkv, qn, kn, o } => {
                    let qn_ = qkv.n;
                    let qd = c.heads * c.head_dim;
                    ops.gemv(w.h.fp(), r, hd, qkv, w.proj.fp(), qn_, Epi::Store, None)?;
                    ops.attn_prep(w.proj.fp(), qn_, r, c.heads, c.kv_heads, c.head_dim, c.rot, qn, kn, c.eps, &self.inv_freq, s.pos, w.q.fp(), &s.kc[ai],
                                  &s.vc[ai], s.ctx)?;
                    let n = s.pos + r;
                    let chunk = n.div_ceil(PIECES).max(64);
                    ops.attn(w.q.fp(), &s.kc[ai], &s.vc[ai], r, c.heads, c.kv_heads, c.head_dim, s.ctx, s.pos, chunk, &w.part, fp(&w.proj, c.head_dim), qn_,
                             2 * c.head_dim, w.y.fp())?;
                    ops.gemv(w.y.fp(), r, qd, o, w.x.fp(), hd, Epi::Acc, None)?;
                    ai += 1;
                }
            }
            ops.mark(|| format!("layer {} {}", li % 4, if c.full[li] { "attention" } else { "deltanet" }))?;
            ops.rms_norm(w.x.fp(), r, hd, Some(&l.post_norm), c.eps, None, w.h.fp())?;
            ops.gemv(w.h.fp(), r, hd, &l.gate_up, w.f.fp(), c.ffn, Epi::Swiglu, None)?;
            ops.gemv(w.f.fp(), r, c.ffn, &l.down, w.x.fp(), hd, Epi::Acc, None)?;
            ops.mark(|| "mlp".into())?;
            tap(li, &w.x)?;
        }
        if all {
            ops.rms_norm(w.x.fp(), r, hd, Some(&self.norm), c.eps, None, self.hidden.fp())?;
        } else {
            ops.rms_norm(fp(&w.x, (r - 1) * hd), 1, hd, Some(&self.norm), c.eps, None, self.hidden.fp())?;
        }
        s.pos += r;
        Ok(())
    }

    /// The text logits of row `row` of `hidden` into `logits` (float32: every product sums in float32), read back
    pub fn logits(&self, ops: &Ops, row: usize) -> Result<Vec<f32>> {
        ops.gemv(fp(&self.hidden, row * self.cfg.hidden), 1, self.cfg.hidden, &self.embed, self.logits.fp(), self.embed.n, Epi::Store, None)?;
        ops.mark(|| "logits".into())?;
        self.logits.to_f32()
    }

    /// Row `row` of the residual stream (after a pass: the layers' output before the final norm) - for checks
    pub fn residual(&self, row: usize) -> Result<Vec<f32>> {
        let v = self.w.x.to_f32()?;
        Ok(v[row * self.cfg.hidden..(row + 1) * self.cfg.hidden].to_vec())
    }
}
