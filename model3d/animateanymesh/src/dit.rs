//! The rectified-flow MMDiT (`networks/transformer.py`, DyMeshMMDiT): trajectory tokens and the 77 CLIP tokens in one
//! joint attention a block, each stream with its own adaLN-Zero modulation from the time embedding, the attention
//! and MLP weights shared. Run as the sampler needs it: the guided pair (with the prompt, without) as two sequences of
//! one batch.

use nextsycl_core::DevBuf;
use nextsycl_model3d::Result;

use crate::clip::{CTX, WIDTH as CLIP_WIDTH};
use crate::ffi::{Norm, GELU, SILU};
use crate::ops::{Lin, Ln, Ops, Weights};
use crate::prof::mark;

pub const WIDTH: usize = 512;
pub const CHANNELS: usize = 64;
const HEADS: usize = 8;
const LAYERS: usize = 12;

struct Block {
    c_qkv: Lin,
    c_proj: Lin,
    fc: Lin,
    proj: Lin,
    ln_x: Ln,
    ln_text: Ln,
}

pub struct Dit {
    t_fc: Lin,
    t_proj: Lin,
    /// every block's two adaLN linears stacked: [LAYERS * 2 * 6 * WIDTH, WIDTH], block l's x stream first
    adaln: Lin,
    blocks: Vec<Block>,
    ln_pre: Ln,
    ln_pre_text: Ln,
    ln_post: Ln,
    clip_mlp: Lin,
    input_proj: Lin,
    output_proj: Lin,
}

/// The text side of the guided pair, ready to join the trajectory tokens: [2, 77, 512] (the prompt's, then the
/// empty prompt's: zeros through ln_pre_text)
pub struct Text(pub DevBuf);

/// The buffers one forward of `k` trajectory tokens uses (made once a request)
pub struct Work {
    k: usize,
    s: usize,
    x: DevBuf,
    n: DevBuf,
    qkv: DevBuf,
    a: DevBuf,
    p: DevBuf,
    h: DevBuf,
    t: DevBuf,
    mods: DevBuf,
    xin: DevBuf,
    xo: DevBuf,
    scratch: DevBuf,
}

impl Dit {
    pub fn load(w: &Weights) -> Result<Dit> {
        let mut blocks = Vec::new();
        let (mut aw, mut ab) = (Vec::new(), Vec::new());
        for l in 0..LAYERS {
            let p = format!("backbone.resblocks.{l}");
            blocks.push(Block {
                c_qkv: w.lin(&format!("{p}.attn.c_qkv"), 3 * WIDTH, WIDTH, true)?,
                c_proj: w.lin(&format!("{p}.attn.c_proj"), WIDTH, WIDTH, true)?,
                fc: w.lin(&format!("{p}.mlp.c_fc"), 4 * WIDTH, WIDTH, true)?,
                proj: w.lin(&format!("{p}.mlp.c_proj"), WIDTH, 4 * WIDTH, true)?,
                ln_x: w.ln(&format!("{p}.ln_x"), WIDTH)?,
                ln_text: w.ln(&format!("{p}.ln_text"), WIDTH)?,
            });
            for s in ["adaln_x", "adaln_text"] {
                let (sw, vw) = w.host(&format!("{p}.{s}.linear.weight"))?;
                let (sb, vb) = w.host(&format!("{p}.{s}.linear.bias"))?;
                if sw != [6 * WIDTH, WIDTH] || sb != [6 * WIDTH] {
                    return Err(nextsycl_model3d::Error(format!("{p}.{s}: {sw:?} / {sb:?}, not [3072, 512] / [3072]")));
                }
                aw.extend(vw);
                ab.extend(vb);
            }
        }
        let n = LAYERS * 2 * 6 * WIDTH;
        Ok(Dit {
            t_fc: w.lin("time_embed.c_fc", 4 * WIDTH, WIDTH, true)?,
            t_proj: w.lin("time_embed.c_proj", WIDTH, 4 * WIDTH, true)?,
            adaln: Lin { w: DevBuf::from_f32(w.gpu, &aw)?, b: Some(DevBuf::from_f32(w.gpu, &ab)?), n, k: WIDTH },
            blocks,
            ln_pre: w.ln("ln_pre", WIDTH)?,
            ln_pre_text: w.ln("ln_pre_text", WIDTH)?,
            ln_post: w.ln("ln_post", WIDTH)?,
            clip_mlp: w.lin("clip_token_mlp", WIDTH, CLIP_WIDTH, true)?,
            input_proj: w.lin("input_proj", WIDTH, CHANNELS, true)?,
            output_proj: w.lin("output_proj", CHANNELS, WIDTH, true)?,
        })
    }

    /// The guided pair's text tokens from CLIP's last hidden state [77, 768] (the prompt's)
    pub fn text(&self, o: &Ops, clip: &DevBuf) -> Result<Text> {
        let t = o.buf(2 * CTX * WIDTH)?;
        let c = t.view(0, CTX * WIDTH * 4)?;
        let u = t.view(CTX * WIDTH * 4, CTX * WIDTH * 4)?;
        o.linear(clip, CTX, &self.clip_mlp, &c)?;
        o.ln(&c, CTX, WIDTH, 1e-5, self.ln_pre_text.norm(), &c)?;
        // the empty prompt's embedding is multiplied by 0: ln_pre_text of zeros
        u.fill(0)?;
        o.ln(&u, CTX, WIDTH, 1e-5, self.ln_pre_text.norm(), &u)?;
        Ok(Text(t))
    }

    pub fn work(&self, o: &Ops, k: usize) -> Result<Work> {
        let s = k + CTX;
        let m = 2 * s;
        Ok(Work {
            k,
            s,
            x: o.buf(m * WIDTH)?,
            n: o.buf(m * WIDTH)?,
            qkv: o.buf(m * 3 * WIDTH)?,
            a: o.buf(m * WIDTH)?,
            p: o.buf(m * WIDTH)?,
            h: o.buf(m * 4 * WIDTH)?,
            t: o.buf(4 * WIDTH)?,
            mods: o.buf(LAYERS * 2 * 6 * WIDTH)?,
            xin: o.buf(2 * k * WIDTH)?,
            xo: o.buf(2 * k * WIDTH)?,
            // the scores of every (sequence, head, query) at once
            scratch: o.buf((2 * HEADS * s * s).min(64 << 20))?,
        })
    }

    /// The flow of `x` [2k, 64] (rows k..2k the unguided copy) at time `t`: out [2k, 64]
    pub fn forward(&self, o: &Ops, w: &Work, text: &Text, x: &DevBuf, t: f32, out: &DevBuf) -> Result<()> {
        let (k, s) = (w.k, w.s);
        let m = 2 * s;
        // the time embedding and every block's modulations
        let te = timestep_embedding(t, WIDTH);
        w.t.write(0, &f32_bytes(&te))?;
        let h1 = w.h.view(0, 4 * WIDTH * 4)?;
        o.linear(&w.t, 1, &self.t_fc, &h1)?;
        o.k.act(&h1, 4 * WIDTH, GELU)?;
        let temb = w.p.view(0, WIDTH * 4)?;
        o.linear(&h1, 1, &self.t_proj, &temb)?;
        o.k.act(&temb, WIDTH, SILU)?;
        o.linear(&temb, 1, &self.adaln, &w.mods)?;
        mark(&o.nsd, "dit time")?;
        // the input: trajectory tokens through input_proj and ln_pre, the text after them, a sequence each
        o.linear(x, 2 * k, &self.input_proj, &w.xin)?;
        o.ln(&w.xin, 2 * k, WIDTH, 1e-5, self.ln_pre.norm(), &w.xin)?;
        for b in 0..2 {
            w.x.copy_within(b * s * WIDTH * 4, &w.xin, b * k * WIDTH * 4, k * WIDTH * 4)?;
            w.x.copy_within((b * s + k) * WIDTH * 4, &text.0, b * CTX * WIDTH * 4, CTX * WIDTH * 4)?;
        }
        let hd = WIDTH / HEADS;
        for (l, bl) in self.blocks.iter().enumerate() {
            // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp of the x stream, then the text's
            let md = |st: usize, i: usize| w.mods.view(((l * 2 + st) * 6 + i) * WIDTH * 4, WIDTH * 4);
            let (xs, xc, xg, xs2, xc2, xg2) = (md(0, 0)?, md(0, 1)?, md(0, 2)?, md(0, 3)?, md(0, 4)?, md(0, 5)?);
            let (ts, tc, tg, ts2, tc2, tg2) = (md(1, 0)?, md(1, 1)?, md(1, 2)?, md(1, 3)?, md(1, 4)?, md(1, 5)?);
            o.k.ln(&w.x, m, WIDTH, 1e-6, s, k, Norm { scale: Some(&xc), shift: Some(&xs), ..Default::default() },
                   Norm { scale: Some(&tc), shift: Some(&ts), ..Default::default() }, &w.n)?;
            mark(&o.nsd, "dit norm")?;
            o.linear(&w.n, m, &bl.c_qkv, &w.qkv)?;
            mark(&o.nsd, "dit qkv")?;
            // per head: q, k, v of 64 each, side by side (the qkv row viewed as [heads, 3 * 64])
            let qkv_k = w.qkv.view(hd * 4, w.qkv.len - hd * 4)?;
            let qkv_v = w.qkv.view(2 * hd * 4, w.qkv.len - 2 * hd * 4)?;
            o.k.attention((&w.qkv, 3 * WIDTH, 3 * hd), (&qkv_k, 3 * WIDTH, 3 * hd), (&qkv_v, 3 * WIDTH, 3 * hd), 2, s, s, HEADS, hd, hd,
                          1.0 / (hd as f32).sqrt(), false, &w.scratch, (&w.a, WIDTH))?;
            mark(&o.nsd, "dit attention")?;
            o.linear(&w.a, m, &bl.c_proj, &w.p)?;
            mark(&o.nsd, "dit proj")?;
            o.k.gated_add(&w.x, &w.p, m, WIDTH, s, k, Some(&xg), Some(&tg))?;
            o.k.ln(&w.x, m, WIDTH, 1e-5, s, k, Norm { w: Some(&bl.ln_x.w), b: Some(&bl.ln_x.b), scale: Some(&xc2), shift: Some(&xs2) },
                   Norm { w: Some(&bl.ln_text.w), b: Some(&bl.ln_text.b), scale: Some(&tc2), shift: Some(&ts2) }, &w.n)?;
            mark(&o.nsd, "dit norm")?;
            o.linear(&w.n, m, &bl.fc, &w.h)?;
            mark(&o.nsd, "dit mlp up")?;
            o.k.act(&w.h, m * 4 * WIDTH, GELU)?;
            mark(&o.nsd, "dit gelu")?;
            o.linear(&w.h, m, &bl.proj, &w.p)?;
            mark(&o.nsd, "dit mlp down")?;
            o.k.gated_add(&w.x, &w.p, m, WIDTH, s, k, Some(&xg2), Some(&tg2))?;
            mark(&o.nsd, "dit norm")?;
        }
        for b in 0..2 {
            w.xo.copy_within(b * k * WIDTH * 4, &w.x, b * s * WIDTH * 4, k * WIDTH * 4)?;
        }
        o.ln(&w.xo, 2 * k, WIDTH, 1e-5, self.ln_post.norm(), &w.xo)?;
        o.linear(&w.xo, 2 * k, &self.output_proj, out)?;
        mark(&o.nsd, "dit out")
    }
}

/// The sinusoidal embedding of time t (networks/util.py: cosines, then sines; float32 as the reference)
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let c = -(10000f64.ln() as f32);
    let args: Vec<f32> = (0..half).map(|i| t * (c * i as f32 / half as f32).exp()).collect();
    args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
