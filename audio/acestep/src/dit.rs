//! ACE-Step 1.5's DiT (modeling_acestep_v15_turbo.py) and its condition encoder: what the flow is conditioned on -
//! the caption's text-encoder states projected, the lyrics' embeddings through their encoder (8 layers), the timbre of
//! a reference (its latents through 4 layers, the first row) - packed as lyrics, timbre, caption; and the decoder: 2
//! frames a patch (the context - source latents and chunk masks - beside x_t), 24 layers of AdaLN-modulated attention
//! (a window of 128 patches on even layers, the whole song on odd ones), cross-attention over the conditions (their
//! keys and values made once a song) and a gated SwiGLU MLP, then the velocity out of the patches.

use std::path::Path;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt};
use nextsycl_gguf::safetensors::SafeTensors;
use serde_json::Value;

use crate::codes::Codes;
use crate::nn::{at, read_f32, rope_table, Attn, Layer, Lin, Loader, Mask, Mlp, Ops, Scratch, EPS, HD, M, P};
use crate::sample::LATENT;

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub layers: usize,
    pub lyric_layers: usize,
    pub timbre_layers: usize,
    pub pooler_layers: usize,
    pub text_dim: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub window: usize,
    pub sliding: Vec<bool>,
    pub theta: f64,
    pub pool: usize,
    pub fsq_levels: Vec<u32>,
    pub turbo: bool,
}

impl Config {
    pub fn read(path: &Path) -> Result<Config> {
        let v: Value = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| Error(format!("{}: not JSON", path.display())))?;
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{}: no {k}", path.display())));
        if v["model_type"].as_str() != Some("acestep") {
            return Err(Error(format!("{}: model_type {}, not acestep", path.display(), v["model_type"])));
        }
        let sliding = v["layer_types"].as_array().map(|a| a.iter().map(|t| t.as_str() == Some("sliding_attention")).collect()).unwrap_or_default();
        Ok(Config { hidden: n("hidden_size")?, heads: n("num_attention_heads")?, kv_heads: n("num_key_value_heads")?, layers: n("num_hidden_layers")?,
                    lyric_layers: n("num_lyric_encoder_hidden_layers")?, timbre_layers: n("num_timbre_encoder_hidden_layers")?,
                    pooler_layers: n("num_attention_pooler_hidden_layers")?, text_dim: n("text_hidden_dim")?, in_channels: n("in_channels")?,
                    patch: n("patch_size")?, window: n("sliding_window")?, sliding, theta: v["rope_theta"].as_f64().unwrap_or(1e6), pool: n("pool_window_size")?,
                    fsq_levels: v["fsq_input_levels"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as u32)).collect()).unwrap_or_default(),
                    turbo: v["is_turbo"].as_bool().unwrap_or(false) || v["model_version"].as_str() == Some("turbo") })
    }

    fn mask(&self, i: usize) -> Mask {
        if self.sliding.get(i).copied().unwrap_or(false) { Mask::Window(self.window) } else { Mask::Full }
    }
}

/// A plain RMS norm's weight and an encoder: its input projection, its layers, its final norm
pub struct Encoder {
    pub embed: Lin,
    pub layers: Vec<Layer>,
    pub norm: DevBuf,
}

impl Encoder {
    fn load(l: &Loader, p: &str, n: usize, cfg: &Config) -> Result<Encoder> {
        let layers = (0..n).map(|i| Layer::load(l, &format!("{p}.layers.{i}"), cfg.heads, cfg.kv_heads, cfg.mask(i))).collect::<Result<_>>()?;
        Ok(Encoder { embed: l.lin(&format!("{p}.embed_tokens"), true)?, layers, norm: l.float(&format!("{p}.norm.weight"))? })
    }

    /// rows [m, embed.k] (bfloat16 on the card) through the encoder: float32 [m, hidden] on the card in "eout"
    fn run(&self, ops: &Ops, s: &mut Scratch, x_in: P, m: usize, cfg: &Config) -> Result<M> {
        let c = cfg.hidden;
        let x = s.get(&ops.gpu, "ex", m * c * 4)?;
        ops.linear(x_in, m, &self.embed, x, Dt::F32)?;
        let cs = ops.put_f32(s, "ecs", &rope_table(m, cfg.theta, 0))?;
        for l in &self.layers {
            ops.layer(s, l, x, m, c, cs)?;
        }
        let out = s.get(&ops.gpu, "eout", m * c * 4)?;
        ops.nsd.rms_norm_mod(x, Dt::F32, m, c, self.norm.ptr(), EPS, none(), none(), none(), out, Dt::F32)?;
        Ok(out)
    }
}

/// A DiT layer
struct DitLayer {
    sa_norm: DevBuf,
    sa: Attn,
    ca_norm: DevBuf,
    ca: Attn,
    mlp_norm: DevBuf,
    mlp: Mlp,
    /// scale_shift_table [6, hidden], float32 on the host
    sst: Vec<f32>,
    mask: Mask,
}

/// A timestep embedding's MLP (TimestepEmbedding)
struct TimeEmbed {
    l1: Lin,
    l2: Lin,
    proj: Lin,
}

/// The conditions of one song as the decoder reads them: each layer's cross keys and values (bfloat16 [len, 16 x 128])
pub struct Cond {
    pub len: usize,
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
}

pub struct Dit {
    pub cfg: Config,
    text_projector: Lin,
    lyric: Encoder,
    timbre: Encoder,
    layers: Vec<DitLayer>,
    proj_in: Lin,
    t_embed: TimeEmbed,
    r_embed: TimeEmbed,
    cond_embed: Lin,
    norm_out: DevBuf,
    /// proj_out as a product [patch x 64, hidden], rows (channel, k)
    proj_out: Lin,
    sst_out: Vec<f32>,
    /// the silence latent [frames, 64] (silence_latent.pt): the source of a song made from nothing, the timbre
    /// reference when none is given, the padding of the tokenizer
    pub silence: Vec<f32>,
    /// the audio codes' tokenizer and detokenizer (a cover's hints, the planner's codes)
    pub codes: Codes,
    /// the null condition (classifier-free guidance's unconditional branch: the base model) [hidden]
    pub null_cond: Vec<f32>,
    pub bytes: u64,
}

impl Dit {
    pub fn load(weights: &Path, cfg: Config, silence: Vec<f32>, gpu: &Arc<Gpu>) -> Result<Dit> {
        let st = SafeTensors::open(weights).map_err(|e| Error(e.0))?;
        let l = Loader::new(&st, gpu);
        let (h, heads, kvh) = (cfg.hidden, cfg.heads, cfg.kv_heads);
        let mut layers = Vec::new();
        for i in 0..cfg.layers {
            let p = format!("decoder.layers.{i}");
            let mask = cfg.mask(i);
            layers.push(DitLayer {
                sa_norm: l.float(&format!("{p}.self_attn_norm.weight"))?,
                sa: Attn::load(&l, &format!("{p}.self_attn"), heads, kvh, mask == Mask::Full)?,
                ca_norm: l.float(&format!("{p}.cross_attn_norm.weight"))?,
                ca: Attn::load(&l, &format!("{p}.cross_attn"), heads, kvh, true)?,
                mlp_norm: l.float(&format!("{p}.mlp_norm.weight"))?,
                mlp: Mlp::load(&l, &format!("{p}.mlp"))?,
                sst: l.host(&format!("{p}.scale_shift_table"))?,
                mask,
            });
        }
        let te = |p: &str| -> Result<TimeEmbed> {
            Ok(TimeEmbed { l1: l.lin(&format!("{p}.linear_1"), true)?, l2: l.lin(&format!("{p}.linear_2"), true)?, proj: l.lin(&format!("{p}.time_proj"), true)? })
        };
        // proj_out: ConvTranspose1d [hidden, 64, patch] -> rows (c, k) of a [64 patch, hidden] product
        let w = l.host("decoder.proj_out.1.weight")?;
        let b = l.host("decoder.proj_out.1.bias")?;
        let pk = cfg.patch;
        let mut wt = vec![0f32; LATENT * pk * h];
        for i in 0..h {
            for c in 0..LATENT {
                for k in 0..pk {
                    wt[(c * pk + k) * h + i] = w[(i * LATENT + c) * pk + k];
                }
            }
        }
        let bias: Vec<f32> = (0..LATENT * pk).map(|r| b[r / pk]).collect();
        let proj_out = Lin { w: l.upload(&crate::nn::bf_bytes(&wt))?, b: Some(l.upload(crate::nn::f32_bytes(&bias))?), n: LATENT * pk, k: h };
        let dit = Dit {
            text_projector: l.lin("encoder.text_projector", false)?,
            lyric: Encoder::load(&l, "encoder.lyric_encoder", cfg.lyric_layers, &cfg)?,
            timbre: Encoder::load(&l, "encoder.timbre_encoder", cfg.timbre_layers, &cfg)?,
            layers,
            proj_in: l.lin("decoder.proj_in.1", true)?,
            t_embed: te("decoder.time_embed")?,
            r_embed: te("decoder.time_embed_r")?,
            cond_embed: l.lin("decoder.condition_embedder", true)?,
            norm_out: l.float("decoder.norm_out.weight")?,
            proj_out,
            sst_out: l.host("decoder.scale_shift_table")?,
            silence,
            codes: Codes::load(&l, &cfg)?,
            null_cond: l.host("null_condition_emb")?,
            bytes: 0,
            cfg,
        };
        Ok(Dit { bytes: l.bytes.get(), ..dit })
    }

    /// `n` frames of the silence latent from its start (tiled when the song is longer: `_get_silence_latent_slice`)
    pub fn silence(&self, n: usize) -> Vec<f32> {
        let have = self.silence.len() / LATENT;
        let mut v = Vec::with_capacity(n * LATENT);
        while v.len() < n * LATENT {
            let k = (n - v.len() / LATENT).min(have);
            v.extend_from_slice(&self.silence[..k * LATENT]);
        }
        v
    }

    /// The packed conditions (prepare_condition's encoder_hidden_states): the lyrics through their encoder, the timbre
    /// (row 0 of the reference's encoding), the caption projected - float32 [len, hidden] on the host
    pub fn conditions(&self, ops: &Ops, s: &mut Scratch, text: &[f32], lyrics: &[f32], refer: &[f32]) -> Result<Vec<f32>> {
        let (c, td) = (self.cfg.hidden, self.cfg.text_dim);
        let (lt, ll, lr) = (text.len() / td, lyrics.len() / td, refer.len() / LATENT);
        let g = &ops.gpu;
        let li = ops.put_bf(s, "lyr_in", lyrics)?;
        let lo = self.lyric.run(ops, s, li, ll, &self.cfg)?;
        let lyr = read_f32(s.buf("eout"), ll * c)?;
        let _ = lo;
        let ri = ops.put_bf(s, "ref_in", refer)?;
        self.timbre.run(ops, s, ri, lr, &self.cfg)?;
        let timbre = read_f32(s.buf("eout"), c)?;
        let ti = ops.put_bf(s, "txt_in", text)?;
        let to = s.get(g, "txt_out", lt * c * 4)?;
        ops.linear(ti, lt, &self.text_projector, to, Dt::F32)?;
        let txt = read_f32(s.buf("txt_out"), lt * c)?;
        let mut out = lyr;
        out.extend_from_slice(&timbre);
        out.extend_from_slice(&txt);
        Ok(out)
    }

    /// Each layer's cross keys and values of the packed conditions (the decoder's cache: the same every step)
    pub fn cond(&self, ops: &Ops, s: &mut Scratch, enc: &[f32]) -> Result<Cond> {
        let (c, len) = (self.cfg.hidden, enc.len() / self.cfg.hidden);
        let g = &ops.gpu;
        let ei = ops.put_bf(s, "cond_in", enc)?;
        let e = s.get(g, "cond_e", len * c * 2)?;
        ops.linear(ei, len, &self.cond_embed, e, Dt::BF16)?;
        let (mut k, mut v) = (Vec::new(), Vec::new());
        for l in &self.layers {
            let kw = l.ca.hkv * HD;
            let kb = DevBuf::new(g, len * kw * 2)?;
            let vb = DevBuf::new(g, len * kw * 2)?;
            ops.linear(e, len, &l.ca.k, kb.ptr(), Dt::BF16)?;
            ops.linear(e, len, &l.ca.v, vb.ptr(), Dt::BF16)?;
            ops.qk_norm_rope(kb.ptr(), len, l.ca.hkv, kw, &l.ca.kn, none())?;
            k.push(kb);
            v.push(vb);
        }
        Ok(Cond { len, k, v })
    }

    /// TimestepEmbedding at t: (temb [hidden], timestep_proj [6, hidden]) - the sinusoid (cos, then sin, of t x 1000
    /// at 128 frequencies) through linear_1, SiLU, linear_2 (= temb), SiLU, time_proj
    fn time(&self, ops: &Ops, s: &mut Scratch, te: &TimeEmbed, t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let half = 128;
        let tt = t * 1000.0;
        let mut f = vec![0f32; 2 * half];
        for i in 0..half {
            let freq = (-(10000f32).ln() * i as f32 / half as f32).exp();
            let a = tt * freq;
            f[i] = a.cos();
            f[half + i] = a.sin();
        }
        let silu = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x = *x / (1.0 + (-*x).exp()));
        let g = &ops.gpu;
        let x = ops.put_bf(s, "t_in", &f)?;
        let o = s.get(g, "t_out", te.proj.n * 4)?;
        ops.linear(x, 1, &te.l1, o, Dt::F32)?;
        let mut h = read_f32(s.buf("t_out"), te.l1.n)?;
        silu(&mut h);
        let x = ops.put_bf(s, "t_in", &h)?;
        ops.linear(x, 1, &te.l2, o, Dt::F32)?;
        let temb = read_f32(s.buf("t_out"), te.l2.n)?;
        let mut h = temb.clone();
        silu(&mut h);
        let x = ops.put_bf(s, "t_in", &h)?;
        ops.linear(x, 1, &te.proj, o, Dt::F32)?;
        Ok((temb, read_f32(s.buf("t_out"), te.proj.n)?))
    }

    /// The velocity at t: x_t [frames, 64] beside the context [frames, 128] (source latents, chunk masks), the
    /// conditions' cache; r = t (the turbo model's timestep_r): float32 [frames, 64]
    pub fn velocity(&self, ops: &Ops, s: &mut Scratch, xt: &[f32], ctx: &[f32], t: f32, cond: &Cond) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (c, pk, ci) = (cfg.hidden, cfg.patch, cfg.in_channels);
        let frames = xt.len() / LATENT;
        let ctx_ch = ci - LATENT;
        if ctx.len() != frames * ctx_ch {
            return Err(Error(format!("a context of {} values for {frames} frames of {ctx_ch}", ctx.len())));
        }
        let sp = frames.div_ceil(pk);
        // patches: [sp, (channel, k)] of cat(context, x_t), zero padded to whole patches
        let mut pin = vec![0f32; sp * ci * pk];
        for f in 0..frames {
            let (p, k) = (f / pk, f % pk);
            for ch in 0..ci {
                let v = if ch < ctx_ch { ctx[f * ctx_ch + ch] } else { xt[f * LATENT + ch - ctx_ch] };
                pin[p * ci * pk + ch * pk + k] = v;
            }
        }
        // the modulation: time_embed(t) + time_embed_r(t - r = 0)
        let (te_t, tp_t) = self.time(ops, s, &self.t_embed, t)?;
        let (te_r, tp_r) = self.time(ops, s, &self.r_embed, 0.0)?;
        let temb: Vec<f32> = te_t.iter().zip(&te_r).map(|(a, b)| a + b).collect();
        let tproj: Vec<f32> = tp_t.iter().zip(&tp_r).map(|(a, b)| a + b).collect();
        let g = &ops.gpu;
        let xin = ops.put_bf(s, "d_in", &pin)?;
        let x = s.get(g, "d_x", sp * c * 4)?;
        ops.linear(xin, sp, &self.proj_in, x, Dt::F32)?;
        let cs = ops.put_f32(s, "d_cs", &rope_table(sp, cfg.theta, 0))?;
        let rows = s.get(g, "d_rows", sp * 4)?;
        s.buf("d_rows").fill(0)?;
        let (qw, i2) = (cfg.heads * HD, self.layers[0].mlp.gu.n);
        let h = s.get(g, "d_h", sp * c * 2)?;
        let q = s.get(g, "d_q", sp * qw * 2)?;
        let kk = s.get(g, "d_k", sp * qw * 2)?;
        let v = s.get(g, "d_v", sp * qw * 2)?;
        let o = s.get(g, "d_o", sp * qw * 2)?;
        let y = s.get(g, "d_y", sp * c * 2)?;
        let gu = s.get(g, "d_gu", sp * i2 * 2)?;
        let ff = s.get(g, "d_ff", sp * (i2 / 2) * 2)?;
        let md = s.get(g, "d_mod", 6 * c * 4)?;
        let mf = |j: usize| at(s.buf("d_mod"), j * c * 4) as P;
        let scale = 1.0 / (HD as f32).sqrt();
        for (li, l) in self.layers.iter().enumerate() {
            let m: Vec<f32> = l.sst.iter().zip(&tproj).map(|(a, b)| a + b).collect();
            s.buf("d_mod").write(0, crate::nn::f32_bytes(&m))?;
            let _ = md;
            // self-attention: norm(x) (1 + scale_msa) + shift_msa; x += attn * gate_msa
            ops.nsd.rms_norm_mod(x, Dt::F32, sp, c, l.sa_norm.ptr(), EPS, rows, mf(1), mf(0), h, Dt::BF16)?;
            let a = &l.sa;
            let kvw = a.hkv * HD;
            ops.linear(h, sp, &a.q, q, Dt::BF16)?;
            ops.linear(h, sp, &a.k, kk, Dt::BF16)?;
            ops.linear(h, sp, &a.v, v, Dt::BF16)?;
            ops.qk_norm_rope(q, sp, a.hq, qw, &a.qn, cs)?;
            ops.qk_norm_rope(kk, sp, a.hkv, kvw, &a.kn, cs)?;
            match l.mask {
                Mask::Window(w) => ops.window(q, kk, v, sp, a.hq, a.hkv, w as i64, scale, o)?,
                _ => ops.self_attention(q, kk, v, sp, a.hq, a.hkv, Mask::Full, o)?,
            }
            ops.linear(o, sp, &a.o, y, Dt::BF16)?;
            ops.add(x, y, sp, c, mf(2))?;
            // cross-attention over the conditions: x += attn(norm(x))
            ops.rms(x, sp, c, &l.ca_norm, h)?;
            ops.linear(h, sp, &l.ca.q, q, Dt::BF16)?;
            ops.qk_norm_rope(q, sp, l.ca.hq, qw, &l.ca.qn, none())?;
            ops.nsd.attention_qk(q, sp, qw, cond.k[li].ptr(), cond.v[li].ptr(), cond.len, l.ca.hkv * HD, Dt::BF16, l.ca.hq, HD, o, Dt::BF16)?;
            ops.linear(o, sp, &l.ca.o, y, Dt::BF16)?;
            ops.add(x, y, sp, c, none())?;
            // MLP: norm(x) (1 + c_scale) + c_shift; x += mlp * c_gate
            ops.nsd.rms_norm_mod(x, Dt::F32, sp, c, l.mlp_norm.ptr(), EPS, rows, mf(4), mf(3), h, Dt::BF16)?;
            ops.linear(h, sp, &l.mlp.gu, gu, Dt::BF16)?;
            ops.nsd.swiglu(gu, Dt::BF16, sp, i2 / 2, ff, Dt::BF16)?;
            ops.linear(ff, sp, &l.mlp.down, y, Dt::BF16)?;
            ops.add(x, y, sp, c, mf(5))?;
        }
        // out: norm(x) (1 + scale) + shift (scale_shift_table + temb), then the patches back to frames
        let mut m = vec![0f32; 2 * c];
        for i in 0..c {
            m[i] = self.sst_out[i] + temb[i];
            m[c + i] = self.sst_out[c + i] + temb[i];
        }
        s.buf("d_mod").write(0, crate::nn::f32_bytes(&m))?;
        ops.nsd.rms_norm_mod(x, Dt::F32, sp, c, self.norm_out.ptr(), EPS, rows, mf(1), mf(0), h, Dt::BF16)?;
        let out = s.get(g, "d_out", sp * LATENT * pk * 4)?;
        ops.linear(h, sp, &self.proj_out, out, Dt::F32)?;
        let po = read_f32(s.buf("d_out"), sp * LATENT * pk)?;
        let mut vel = vec![0f32; frames * LATENT];
        for f in 0..frames {
            let (p, k) = (f / pk, f % pk);
            for ch in 0..LATENT {
                vel[f * LATENT + ch] = po[p * LATENT * pk + ch * pk + k];
            }
        }
        Ok(vel)
    }
}
