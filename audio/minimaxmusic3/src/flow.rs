//! The flow-matching stage: the frames' hidden states become Flow-VAE latents (128 channels, 44,100 / 512 a second).
//!
//! The condition encoder mixes each frame's eight hidden states with learned softmax weights (done as the frames are
//! made: `mix`), a 3-tap convolution projects them to 2,048, and nearest-neighbour resampling stretches 25 frames a
//! second onto the latents' 86.13. The transformer (36 blocks of 2,048, 32 heads of 64, the first 32 features of a
//! head rotated, an 8,192 GLU) takes [latents | zeros | condition] (2,304 channels) per latent frame and a leading
//! time-embedding token, and gives the velocity; two passes a step (the condition, zeros) for guidance (1.7).
//!
//! Songs are made in windows of 200 frames, 100 apart: a window's first latents (the previous one's overlap) are
//! blended toward the previous window's at every step and kept as they were at the end, so neighbours agree.
//!
//! ```text
//!   x = concat(lat, 0, cond);  x += conv1x1(x);  h = [temb | proj_in(x)]
//!   block: h += attn(rope(layernorm(h)));  a, g = ff_in(layernorm(h)); h += ff_out(a * silu(g))
//!   v = proj_out(h[1:]);  v += conv1x1(v)
//! ```

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::ar::{bytes, HIDDEN};
use crate::ops::{fp, nul, Mat, Ops, Shards};

pub const LATENT: usize = 128;
const COND: usize = 2048;
const IN: usize = 2 * LATENT + COND;
const DIM: usize = 2048;
const HEADS: usize = 32;
const HEAD: usize = 64;
const ROT: usize = 32;
const FF: usize = 8192;
const FOURIER: usize = 256;
pub const GUIDANCE: f32 = 1.7;
/// the window: 200 frames, 100 apart; the overlap's latents
pub const CHUNK_FRAMES: usize = 200;
pub const CHUNK_HOP: usize = 100;
const OVERLAP: usize = 172;

struct Block {
    n1w: DevBuf,
    n1b: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    qkv: Mat,
    o: Mat,
    /// the gate's rows first, then the value's (the shared swiglu reads silu(first half) * second half)
    ff_in: Mat,
    ff_in_b: DevBuf,
    ff_out: Mat,
    ff_out_b: DevBuf,
}

pub struct Flow {
    /// the condition encoder: the layer mix (softmax'd) and scale, the 3-tap projection float32 [2048, 4096, 3]
    pub mix: Vec<f32>,
    pub mix_scale: f32,
    proj: DevBuf,
    proj_b: DevBuf,
    pre: Mat,
    proj_in: Mat,
    post: Mat,
    proj_out: Mat,
    blocks: Vec<Block>,
    // the time embedding, on the host (one row a step)
    fourier: Vec<f32>,
    t1w: Vec<f32>,
    t1b: Vec<f32>,
    t2w: Vec<f32>,
    t2b: Vec<f32>,
    inv_freq: DevBuf,
    pub bytes: usize,
}

/// A window's buffers, for `l` latent frames
pub struct Work {
    l: usize,
    x: DevBuf,
    xh: DevBuf,
    y: DevBuf,
    h: DevBuf,
    hn: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    ff: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    pub v: DevBuf,
    vt: DevBuf,
}

impl Flow {
    pub fn load(ops: &Ops, cond: &Shards, tr: &Shards, log: &mut dyn FnMut(String)) -> Result<Flow> {
        let t0 = std::time::Instant::now();
        let logits = cond.f32("layer_weight_logits")?;
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        let mix: Vec<f32> = e.iter().map(|x| x / sum).collect();
        let mix_scale = cond.f32("layer_scale")?[0];
        if mix.len() != 8 || cond.shape("proj.weight")? != [COND, HIDDEN, 3] {
            return Err(Error("the condition encoder: not MiniMax Music 3's (8 layers, 4096 -> 2048)".into()));
        }
        let mut blocks = Vec::new();
        let mut bytes = 0;
        for b in 0.. {
            let p = format!("transformer_blocks.{b}.");
            if tr.find(&format!("{p}norm1.weight")).is_err() {
                break;
            }
            let n = |s: &str| format!("{p}{s}");
            let mut fib = tr.f32(&n("ff_in.bias"))?;
            fib.rotate_left(FF);
            let blk = Block {
                n1w: tr.dev_f32(ops, &n("norm1.weight"))?,
                n1b: tr.dev_f32(ops, &n("norm1.bias"))?,
                n2w: tr.dev_f32(ops, &n("norm2.weight"))?,
                n2b: tr.dev_f32(ops, &n("norm2.bias"))?,
                qkv: tr.mat(ops, &[(&n("attn.to_q.weight"), 0, DIM), (&n("attn.to_k.weight"), 0, DIM), (&n("attn.to_v.weight"), 0, DIM)], false)?,
                o: tr.mat1(ops, &n("attn.to_out.0.weight"), false)?,
                ff_in: tr.mat(ops, &[(&n("ff_in.weight"), FF, 2 * FF), (&n("ff_in.weight"), 0, FF)], false)?,
                ff_in_b: DevBuf::from_f32(&ops.gpu, &fib)?,
                ff_out: tr.mat1(ops, &n("ff_out.weight"), false)?,
                ff_out_b: tr.dev_f32(ops, &n("ff_out.bias"))?,
            };
            bytes += blk.qkv.bytes() + blk.o.bytes() + blk.ff_in.bytes() + blk.ff_out.bytes();
            blocks.push(blk);
        }
        if blocks.len() != 36 {
            return Err(Error(format!("the flow transformer: {} blocks, not 36", blocks.len())));
        }
        let inv: Vec<f32> = (0..ROT / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / ROT as f32)).collect();
        let f = Flow {
            mix,
            mix_scale,
            proj: cond.dev_f32(ops, "proj.weight")?,
            proj_b: cond.dev_f32(ops, "proj.bias")?,
            pre: tr.mat1(ops, "preprocess_conv.weight", false)?,
            proj_in: tr.mat1(ops, "proj_in.weight", false)?,
            post: tr.mat1(ops, "postprocess_conv.weight", false)?,
            proj_out: tr.mat1(ops, "proj_out.weight", false)?,
            blocks,
            fourier: tr.f32("time_proj.weight")?,
            t1w: tr.f32("time_embed.linear_1.weight")?,
            t1b: tr.f32("time_embed.linear_1.bias")?,
            t2w: tr.f32("time_embed.linear_2.weight")?,
            t2b: tr.f32("time_embed.linear_2.bias")?,
            inv_freq: DevBuf::from_f32(&ops.gpu, &inv)?,
            bytes,
        };
        ops.gpu.sync()?;
        log(format!("flow transformer (half, {:.1} GiB) in {:.0} s", bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64()));
        Ok(f)
    }

    /// Latent frames for `frames` language-model frames (the condition encoder's resampling)
    pub fn latent_len(frames: usize) -> usize {
        ((frames as f64 * 44100.0 / 24000.0 * 960.0 / 512.0) as usize).max(1)
    }

    /// The window's conditioning: `frames` mixed rows [n, HIDDEN] (device, from row `at`) -> [latent_len(n), COND]
    pub fn condition(&self, ops: &Ops, nsd: &Nsd, frames: &DevBuf, at: usize, n: usize) -> Result<DevBuf> {
        let g = &ops.gpu;
        let t = DevBuf::f32(g, HIDDEN * n)?;
        ops.transpose(fp(frames, at * HIDDEN), n, HIDDEN, t.fp())?;
        let c = DevBuf::f32(g, COND * n)?;
        nsd.conv1d(t.ptr(), 1, HIDDEN, n, self.proj.ptr(), COND, 3, self.proj_b.ptr(), 1, 1, 1, c.ptr(), n)?;
        let lo = Flow::latent_len(n);
        let out = DevBuf::f32(g, lo * COND)?;
        ops.nearest_rows(c.fp(), COND, n, lo, out.fp())?;
        g.sync()?;
        Ok(out)
    }

    pub fn work(&self, ops: &Ops, l: usize) -> Result<Work> {
        let g = &ops.gpu;
        let r = 2 * (l + 1);
        Ok(Work {
            l,
            x: DevBuf::f32(g, 2 * l * IN)?,
            xh: DevBuf::new(g, r * IN.max(FF) * 2)?,
            y: DevBuf::f32(g, 2 * l * IN)?,
            h: DevBuf::f32(g, r * DIM)?,
            hn: DevBuf::new(g, r * DIM * 2)?,
            qkv: DevBuf::new(g, r * 3 * DIM * 2)?,
            att: DevBuf::new(g, r * DIM * 2)?,
            ff: DevBuf::new(g, r * 2 * FF * 2)?,
            act: DevBuf::new(g, r * FF * 2)?,
            tmp: DevBuf::f32(g, r * DIM)?,
            v: DevBuf::f32(g, 2 * l * LATENT)?,
            vt: DevBuf::f32(g, 2 * l * LATENT)?,
        })
    }

    /// The time embedding's row for flow time t (0 noise, 1 data)
    fn temb(&self, t: f32) -> Vec<f32> {
        let half = FOURIER / 2;
        let a = 2.0 * std::f32::consts::PI * t;
        let mut f = vec![0f32; FOURIER];
        for i in 0..half {
            let ang = a * self.fourier[i];
            f[i] = ang.cos();
            f[half + i] = ang.sin();
        }
        let lin = |x: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
            let k = x.len();
            b.iter().enumerate().map(|(o, bo)| bo + w[o * k..(o + 1) * k].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
        };
        let h: Vec<f32> = lin(&f, &self.t1w, &self.t1b).into_iter().map(|x| x / (1.0 + (-x).exp())).collect();
        lin(&h, &self.t2w, &self.t2b)
    }

    /// Both guidance passes' velocities for latents `lat` [l, LATENT] at time t: w.v = [conditional | unconditional]
    pub fn velocity(&self, ops: &Ops, nsd: &Nsd, w: &Work, lat: &DevBuf, cond: &DevBuf, t: f32) -> Result<()> {
        let l = w.l;
        let s = l + 1;
        let r = 2 * s;
        ops.dit_in(lat.fp(), cond.fp(), l, LATENT, COND, w.x.fp())?;
        ops.to_half(w.x.fp(), w.xh.ptr(), 2 * l * IN)?;
        self.pre.apply_half(nsd, w.xh.ptr(), 2 * l, nul(), w.y.ptr(), Dt::F32)?;
        ops.add(w.y.fp(), w.x.fp(), 2 * l * IN)?;
        ops.to_half(w.y.fp(), w.xh.ptr(), 2 * l * IN)?;
        let te = bytes(&self.temb(t));
        for b in 0..2 {
            self.proj_in.apply_half(nsd, w.xh.ptr().wrapping_byte_add(b * l * IN * 2), l, nul(), fp(&w.h, (b * s + 1) * DIM).cast(), Dt::F32)?;
            w.h.write(b * s * DIM * 4, &te)?;
        }
        for blk in &self.blocks {
            nsd.layer_norm(w.h.ptr(), Dt::F32, r, DIM, blk.n1w.ptr(), blk.n1b.ptr(), 1e-5, w.hn.ptr(), Dt::F16)?;
            blk.qkv.apply_half(nsd, w.hn.ptr(), r, nul(), w.qkv.ptr(), Dt::F16)?;
            ops.rope_partial(w.qkv.ptr(), 3 * DIM, r, s, HEADS, HEAD, ROT, &self.inv_freq)?;
            ops.rope_partial(w.qkv.ptr().wrapping_byte_add(DIM * 2), 3 * DIM, r, s, HEADS, HEAD, ROT, &self.inv_freq)?;
            nsd.attention_batch(w.qkv.ptr(), w.qkv.ptr().wrapping_byte_add(DIM * 2), w.qkv.ptr().wrapping_byte_add(2 * DIM * 2), Dt::F16, 2, s, HEADS, HEAD,
                                3 * DIM, w.att.ptr(), Dt::F16)?;
            blk.o.apply_half(nsd, w.att.ptr(), r, nul(), w.tmp.ptr(), Dt::F32)?;
            ops.add(w.h.fp(), w.tmp.fp(), r * DIM)?;
            nsd.layer_norm(w.h.ptr(), Dt::F32, r, DIM, blk.n2w.ptr(), blk.n2b.ptr(), 1e-5, w.hn.ptr(), Dt::F16)?;
            blk.ff_in.apply_half(nsd, w.hn.ptr(), r, blk.ff_in_b.ptr(), w.ff.ptr(), Dt::F16)?;
            nsd.swiglu(w.ff.ptr(), Dt::F16, r, FF, w.act.ptr(), Dt::F16)?;
            blk.ff_out.apply_half(nsd, w.act.ptr(), r, blk.ff_out_b.ptr(), w.tmp.ptr(), Dt::F32)?;
            ops.add(w.h.fp(), w.tmp.fp(), r * DIM)?;
        }
        // the latent rows (the time token dropped) out, then the 1x1 convolution with its shortcut
        for b in 0..2 {
            ops.to_half(fp(&w.h, (b * s + 1) * DIM), w.hn.ptr(), l * DIM)?;
            self.proj_out.apply_half(nsd, w.hn.ptr(), l, nul(), fp(&w.vt, b * l * LATENT).cast(), Dt::F32)?;
        }
        ops.to_half(w.vt.fp(), w.hn.ptr(), 2 * l * LATENT)?;
        self.post.apply_half(nsd, w.hn.ptr(), 2 * l, nul(), w.v.ptr(), Dt::F32)?;
        ops.add(w.v.fp(), w.vt.fp(), 2 * l * LATENT)?;
        Ok(())
    }

    /// The flow times: diffusers' FlowMatchEulerDiscreteScheduler with sigmas linspace(1, 1/steps, steps), inverted
    /// (t = 1 - sigma: 0 noise, 1 data), then 1 appended
    pub fn times(steps: usize) -> Vec<f32> {
        let (a, b) = (1.0f64, 1.0 / steps as f64);
        let mut t: Vec<f32> = (0..steps).map(|i| {
            let s = if steps == 1 { a } else if i == steps - 1 { b } else { a + (b - a) / (steps - 1) as f64 * i as f64 };
            1.0 - s as f32
        }).collect();
        t.push(1.0);
        t
    }

    /// One window: `lat` (its noise, [l, LATENT], in place) denoised over `times` toward `cond`; with `prev` (the
    /// previous window's overlap latents) its first rows blended toward them every step and set to them at the end
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(&self, ops: &Ops, nsd: &Nsd, w: &Work, lat: &DevBuf, cond: &DevBuf, times: &[f32], cfg: f32, prev: Option<(&DevBuf, usize)>,
                   each: &mut dyn FnMut(usize) -> Result<()>) -> Result<()> {
        let l = w.l;
        let n = l * LATENT;
        let noise = match prev {
            Some((_, ov)) => {
                let b = DevBuf::f32(&ops.gpu, ov * LATENT)?;
                b.copy_within(0, lat, 0, ov * LATENT * 4)?;
                Some(b)
            }
            None => None,
        };
        for i in 0..times.len() - 1 {
            let t = times[i];
            if let (Some((p, ov)), Some(nz)) = (prev, &noise) {
                ops.blend(lat.fp(), nz.fp(), p.fp(), ov * LATENT, 1.0 - (1.0 - 1e-6) * t, t)?;
            }
            self.velocity(ops, nsd, w, lat, cond, t)?;
            ops.cfg_step(lat.fp(), w.v.fp(), fp(&w.v, n), n, cfg, times[i + 1] - t)?;
            ops.gpu.sync()?;
            each(i)?;
        }
        if let Some((p, ov)) = prev {
            lat.copy_within(0, p, 0, ov * LATENT * 4)?;
        }
        ops.gpu.sync()
    }

    /// The overlap a window hands the next: its latent rows [start, end) - the reference's (len - 344, len - 172)
    pub fn overlap(l: usize) -> (usize, usize) {
        let a = l.saturating_sub(2 * OVERLAP);
        (a, a.max(l.saturating_sub(OVERLAP)))
    }

    /// The windows' first frames for `frames` frames
    pub fn starts(frames: usize) -> Vec<usize> {
        if frames <= CHUNK_FRAMES {
            vec![0]
        } else {
            (0..frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_windows_and_lengths_are_the_references() {
        assert_eq!(Flow::starts(150), vec![0]);
        assert_eq!(Flow::starts(330), vec![0, 100, 200]);
        assert_eq!(Flow::latent_len(50), 172);
        assert_eq!(Flow::latent_len(200), 689);
        assert_eq!(Flow::overlap(689), (345, 517));
        let t = Flow::times(30);
        assert_eq!(t.len(), 31);
        assert_eq!(t[0], 0.0);
        assert!((t[1] - 1.0 / 30.0).abs() < 1e-6 && t[30] == 1.0);
    }
}
