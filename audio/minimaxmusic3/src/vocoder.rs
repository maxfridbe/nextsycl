//! The Flow-VAE decoder (DAC's): latents [128, L] -> stereo at 44.1 kHz, 512 samples a latent frame. The two
//! channels are decoded as two 64-channel streams (a batch of two): a 1x1 projection to 1,024, a 7-tap convolution
//! to 1,536, four blocks (Snake, a transposed convolution up x8, x8, x4, x2 halving the channels, three residual
//! units of Snake / 7-tap dilated (1, 3, 9) / Snake / 1x1), Snake, a 7-tap convolution to one channel, tanh.
//! The checkpoint's weight-normed convolutions (g, v) are folded at load: w = g v / |v| per output (per input for the
//! transposed ones: their first axis).

use nextsycl_audio::Result;
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Nsd;

use crate::flow::LATENT;
use crate::ops::{Ops, Shards};

const RATIOS: [usize; 4] = [8, 8, 4, 2];
pub const HOP: usize = 512;

struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Unit {
    a1: DevBuf,
    c1: Conv,
    a2: DevBuf,
    c2: Conv,
    dil: usize,
}

struct Block {
    snake: DevBuf,
    up: Conv,
    stride: usize,
    units: Vec<Unit>,
}

pub struct Vocoder {
    inp: Conv,
    conv_in: Conv,
    blocks: Vec<Block>,
    snake_out: DevBuf,
    conv_out: Conv,
}

/// NS_MM3_NSD_CONV=1: the shared plain-loop convolutions (exact float32, slow) instead of oneDNN's
fn nsd_convs() -> bool {
    std::env::var("NS_MM3_NSD_CONV").is_ok_and(|v| v == "1")
}

/// A weight-normed convolution's weight: g [n0, 1, 1] and v [n0, n1, k], normalized over all but the first axis
fn fold(f: &Shards, p: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let g = f.f32(&format!("{p}.weight_g"))?;
    let v = f.f32(&format!("{p}.weight_v"))?;
    let shape = f.shape(&format!("{p}.weight_v"))?;
    let per = v.len() / g.len();
    let mut w = vec![0f32; v.len()];
    for (o, go) in g.iter().enumerate() {
        let row = &v[o * per..(o + 1) * per];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        for (i, x) in row.iter().enumerate() {
            w[o * per + i] = go * x / norm;
        }
    }
    Ok((w, shape))
}

fn conv(ops: &Ops, f: &Shards, p: &str, transposed: bool) -> Result<Conv> {
    let (w, s) = if f.find(&format!("{p}.weight_g")).is_ok() { fold(f, p)? } else { (f.f32(&format!("{p}.weight"))?, f.shape(&format!("{p}.weight"))?) };
    // [Co, Ci, K], or [Ci, Co, K] transposed
    let (ci, co) = if transposed { (s[0], s[1]) } else { (s[1], s[0]) };
    Ok(Conv { w: DevBuf::from_f32(&ops.gpu, &w)?, b: f.dev_f32(ops, &format!("{p}.bias"))?, ci, co, k: s[2] })
}

/// A signal [2, c, l] on the GPU
struct Sig {
    t: DevBuf,
    c: usize,
    l: usize,
}

impl Vocoder {
    pub fn load(ops: &Ops, f: &Shards) -> Result<Vocoder> {
        let mut blocks = Vec::new();
        for (i, stride) in RATIOS.iter().enumerate() {
            let p = format!("blocks.{i}");
            let units = [1, 3, 9].iter().enumerate().map(|(u, dil)| -> Result<Unit> {
                let q = format!("{p}.res_unit{}", u + 1);
                Ok(Unit {
                    a1: f.dev_f32(ops, &format!("{q}.snake1.alpha"))?,
                    c1: conv(ops, f, &format!("{q}.conv1"), false)?,
                    a2: f.dev_f32(ops, &format!("{q}.snake2.alpha"))?,
                    c2: conv(ops, f, &format!("{q}.conv2"), false)?,
                    dil: *dil,
                })
            }).collect::<Result<Vec<_>>>()?;
            blocks.push(Block { snake: f.dev_f32(ops, &format!("{p}.snake1.alpha"))?, up: conv(ops, f, &format!("{p}.conv_t1"), true)?, stride: *stride, units });
        }
        Ok(Vocoder {
            inp: conv(ops, f, "dec_in_proj", false)?,
            conv_in: conv(ops, f, "conv_in", false)?,
            blocks,
            snake_out: f.dev_f32(ops, "snake_out.alpha")?,
            conv_out: conv(ops, f, "conv_out", false)?,
        })
    }

    fn conv(&self, ops: &Ops, nsd: &Nsd, x: &Sig, c: &Conv, dil: usize, pad: usize) -> Result<Sig> {
        let lo = x.l + 2 * pad - dil * (c.k - 1);
        let t = DevBuf::f32(&ops.gpu, 2 * c.co * lo)?;
        if nsd_convs() {
            nsd.conv1d(x.t.ptr(), 2, c.ci, x.l, c.w.ptr(), c.co, c.k, c.b.ptr(), 1, dil, pad, t.ptr(), lo)?;
        } else {
            ops.conv1d(x.t.fp(), 2, c.ci, x.l, &c.w, c.co, c.k, c.b.fp(), 1, dil, pad, t.fp(), lo)?;
        }
        Ok(Sig { t, c: c.co, l: lo })
    }

    fn snake(&self, ops: &Ops, nsd: &Nsd, x: &Sig, a: &DevBuf) -> Result<Sig> {
        let t = DevBuf::f32(&ops.gpu, 2 * x.c * x.l)?;
        nsd.snake(x.t.ptr(), 2, x.c, x.l, a.ptr(), t.ptr())?;
        Ok(Sig { t, c: x.c, l: x.l })
    }

    /// Latents [l, LATENT] (rows) -> the stereo signal [2, l * 512]
    pub fn decode(&self, ops: &Ops, nsd: &Nsd, lat: &DevBuf, l: usize) -> Result<Vec<f32>> {
        // channels first: [128, l] = the two streams [2, 64, l]
        let t = DevBuf::f32(&ops.gpu, LATENT * l)?;
        ops.transpose(lat.fp(), l, LATENT, t.fp())?;
        let mut x = Sig { t, c: LATENT / 2, l };
        x = self.conv(ops, nsd, &x, &self.inp, 1, 0)?;
        x = self.conv(ops, nsd, &x, &self.conv_in, 1, 3)?;
        for b in &self.blocks {
            let y = self.snake(ops, nsd, &x, &b.snake)?;
            let s = b.stride;
            let pad = s.div_ceil(2);
            let lo = (y.l - 1) * s + b.up.k - 2 * pad;
            let t = DevBuf::f32(&ops.gpu, 2 * b.up.co * lo)?;
            if nsd_convs() {
                nsd.conv_transpose1d(y.t.ptr(), 2, b.up.ci, y.l, b.up.w.ptr(), b.up.co, b.up.k, b.up.b.ptr(), s, pad, t.ptr(), lo)?;
            } else {
                ops.conv_transpose1d(y.t.fp(), 2, b.up.ci, y.l, &b.up.w, b.up.co, b.up.k, b.up.b.fp(), s, pad, t.fp(), lo)?;
            }
            x = Sig { t, c: b.up.co, l: lo };
            for u in &b.units {
                let y = self.snake(ops, nsd, &x, &u.a1)?;
                let y = self.conv(ops, nsd, &y, &u.c1, u.dil, 3 * u.dil)?;
                let y = self.snake(ops, nsd, &y, &u.a2)?;
                let y = self.conv(ops, nsd, &y, &u.c2, 1, 0)?;
                ops.add(x.t.fp(), y.t.fp(), 2 * x.c * x.l)?;
            }
        }
        let y = self.snake(ops, nsd, &x, &self.snake_out)?;
        let y = self.conv(ops, nsd, &y, &self.conv_out, 1, 3)?;
        ops.tanh(y.t.fp(), 2 * y.l)?;
        ops.gpu.sync()?;
        y.t.to_f32()
    }

    /// Samples of `l` latent frames
    pub fn samples(l: usize) -> usize {
        l * HOP
    }
}
