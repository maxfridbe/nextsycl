//! The video encoder: pictures [frames, H, W, 3] in [0, 1] -> normalized latents [24, T, H/16, W/16] - what turns a
//! keyframe picture, or the last frames of a previous clip (a motion guide), into the latents the denoiser is
//! conditioned on.
//!
//! A 3-D causal convolution network (ComfyUI `ldm/minimax/vae.py`, `EncoderFCN3D`): six levels of two residual
//! blocks (GroupNorm per frame + SiLU + 3x3x3 convolution, twice), 128 to 1024 channels, downsampling 16x in space
//! and 4x in time. "Causal": along time a convolution only sees the present and the past (two zero frames in
//! front); in space the border is reflected. Then a 1x1x1 convolution, the first 24 of its 48 channels (the
//! mean), normalized by the latent statistics.
//!
//! Pieces as the reference cuts them: 17-frame clips (the last one padded by repeating its last frame; three
//! latent frames dropped from the end of the whole), 256 x 256 pixel tiles cross-faded in latent space.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::host;
use crate::dtype::DType;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

const GROUPS: i64 = 32;
const EPS: f32 = 1e-6;
const SPACE_DOWN: [usize; 6] = [2, 2, 2, 2, 1, 1];
const TIME_DOWN: [usize; 6] = [1, 2, 2, 1, 1, 1];
const CLIP: usize = 17;
const TOKEN_DROP: usize = 3;
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
const RATIO: usize = 16;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

struct C3 {
    /// [co, ci, k, k, k] in the activations' type
    w: Tensor,
    b: Tensor,
    co: usize,
    k: usize,
}

struct Norm {
    w: Tensor,
    b: Tensor,
}

struct Res {
    norm1: Norm,
    conv1: C3,
    norm2: Norm,
    conv2: C3,
    nin: Option<C3>,
}

struct Level {
    blocks: Vec<Res>,
    down: Option<C3>,
    space: usize,
    time: usize,
}

pub struct VideoEncoder {
    dev: Arc<Device>,
    dt: DType,
    conv_in: C3,
    levels: Vec<Level>,
    norm_out: Norm,
    conv_out: C3,
    quant_w: Vec<f32>,
    quant_b: Vec<f32>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// A volume [t, h, w, c] channels-last on the device.
struct V {
    x: Tensor,
    t: usize,
    h: usize,
    w: usize,
    c: usize,
}

fn f32_dev(dev: &Arc<Device>, v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

impl VideoEncoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<VideoEncoder> {
        let dt = DType::F16;
        let conv = |p: &str| -> Result<C3> {
            let e = ck.get(&format!("{p}.weight"))?;
            let w = host(ck, &format!("{p}.weight"))?;
            Ok(C3 {
                w: Tensor::from_bytes(dev, dt, &e.shape, &w.iter().flat_map(|f| crate::dtype::f32_to_f16(*f).to_le_bytes()).collect::<Vec<u8>>())?,
                b: f32_dev(dev, &host(ck, &format!("{p}.bias"))?)?,
                co: e.shape[0],
                k: e.shape[2],
            })
        };
        let norm = |p: &str| -> Result<Norm> { Ok(Norm { w: f32_dev(dev, &host(ck, &format!("{p}.weight"))?)?, b: f32_dev(dev, &host(ck, &format!("{p}.bias"))?)? }) };
        let mut levels = Vec::new();
        for (i, (space, time)) in SPACE_DOWN.iter().zip(TIME_DOWN).enumerate() {
            let mut blocks = Vec::new();
            for j in 0.. {
                let p = format!("encoder.down.{i}.block.{j}");
                if !ck.entries.contains_key(&format!("{p}.conv1.weight")) {
                    break;
                }
                blocks.push(Res {
                    norm1: norm(&format!("{p}.norm1"))?,
                    conv1: conv(&format!("{p}.conv1"))?,
                    norm2: norm(&format!("{p}.norm2"))?,
                    conv2: conv(&format!("{p}.conv2"))?,
                    nin: match ck.entries.contains_key(&format!("{p}.nin_shortcut.weight")) {
                        true => Some(conv(&format!("{p}.nin_shortcut"))?),
                        false => None,
                    },
                });
            }
            let down = match space * time > 1 {
                true => Some(conv(&format!("encoder.down.{i}.downsample.conv"))?),
                false => None,
            };
            levels.push(Level { blocks, down, space: *space, time });
        }
        if levels.iter().any(|l| l.blocks.is_empty()) {
            return Err(Error(format!("{}: no video encoder (the encoder.down.* weights)", ck.path.display())));
        }
        Ok(VideoEncoder {
            dev: dev.clone(),
            dt,
            conv_in: conv("encoder.conv_in")?,
            levels,
            norm_out: norm("encoder.norm_out")?,
            conv_out: conv("encoder.conv_out")?,
            quant_w: host(ck, "quant_conv.weight")?,
            quant_b: host(ck, "quant_conv.bias")?,
            mean: host(ck, "latents_mean")?,
            std: host(ck, "latents_std")?,
        })
    }

    fn vol(&self, t: usize, h: usize, w: usize, c: usize) -> Result<V> {
        Ok(V { x: Tensor::new(&self.dev, self.dt, &[t * h * w, c])?, t, h, w, c })
    }

    /// The causal convolution: pad (zeros in front along time, reflected border in space), then convolve.
    fn conv(&self, v: &V, c: &C3, (front, top, bottom, left, right): (usize, usize, usize, usize, usize), (st, ss): (usize, usize)) -> Result<V> {
        let d = &self.dev;
        let code = self.dt.kernel_code()?;
        let padded;
        let src = if front + top + bottom + left + right > 0 {
            padded = self.vol(v.t + front, v.h + top + bottom, v.w + left + right, v.c)?;
            // SAFETY: device buffers of this device, sized as declared.
            let rc = unsafe {
                (d.api.pad3d)(d.ctx, v.x.buf.ptr(), code, v.t as i64, v.h as i64, v.w as i64, v.c as i64, front as i64, top as i64, bottom as i64,
                              left as i64, right as i64, padded.x.buf.ptr())
            };
            d.check(rc)?;
            &padded
        } else {
            v
        };
        let (to, ho, wo) = ((src.t - c.k) / st + 1, (src.h - c.k) / ss + 1, (src.w - c.k) / ss + 1);
        let out = self.vol(to, ho, wo, c.co)?;
        // SAFETY: as above; w [co, ci, k, k, k].
        let rc = unsafe {
            (d.api.conv3d_ex)(d.ctx, src.x.buf.ptr(), code, src.t as i64, src.h as i64, src.w as i64, src.c as i64, c.w.buf.ptr(), c.co as i64,
                              c.k as i64, c.k as i64, c.k as i64, st as i64, ss as i64, ss as i64, c.b.buf.ptr().cast(), out.x.buf.ptr())
        };
        d.check(rc)?;
        Ok(out)
    }

    /// a 3x3x3 causal convolution with the default padding (one in space, two frames in front)
    fn conv3(&self, v: &V, c: &C3) -> Result<V> {
        let p = c.k / 2;
        self.conv(v, c, (2 * p, p, p, p, p), (1, 1))
    }

    fn norm_silu(&self, v: &V, n: &Norm) -> Result<V> {
        let y = self.vol(v.t, v.h, v.w, v.c)?;
        let d = &self.dev;
        // SAFETY: as in `conv`; the norm vectors float32 [c].
        let rc = unsafe {
            (d.api.group_norm_silu)(d.ctx, v.x.buf.ptr(), self.dt.kernel_code()?, v.t as i64, (v.h * v.w) as i64, v.c as i64, GROUPS, n.w.buf.ptr().cast(),
                                    n.b.buf.ptr().cast(), EPS, std::ptr::null(), std::ptr::null(), y.x.buf.ptr())
        };
        d.check(rc)?;
        Ok(y)
    }

    /// One piece through the network: normalized pixels [t, h, w, 3] -> the 48 moments [t', h/16, w/16, 48].
    fn moments(&self, px: &[f32], t: usize, h: usize, w: usize) -> Result<Moments> {
        let bytes: Vec<u8> = px.iter().flat_map(|f| crate::dtype::f32_to_f16(*f).to_le_bytes()).collect();
        let mut v = V { x: Tensor::from_bytes(&self.dev, self.dt, &[t * h * w, 3], &bytes)?, t, h, w, c: 3 };
        v = self.conv3(&v, &self.conv_in)?;
        for l in &self.levels {
            for b in &l.blocks {
                let hh = self.norm_silu(&v, &b.norm1)?;
                let hh = self.conv3(&hh, &b.conv1)?;
                let sc = match &b.nin {
                    Some(n) => self.conv(&v, n, (0, 0, 0, 0, 0), (1, 1))?,
                    None => v,
                };
                let hh = self.norm_silu(&hh, &b.norm2)?;
                let out = self.conv3(&hh, &b.conv2)?;
                crate::ops::add(&out.x, &sc.x)?;
                v = out;
            }
            if let Some(dc) = &l.down {
                // two zero frames in front; for a stride-2 space, the border reflected on the right and bottom only
                let pad = if l.space == 2 { (2, 0, 1, 0, 1) } else { (2, 0, 0, 0, 0) };
                v = self.conv(&v, dc, pad, (l.time, l.space))?;
            }
        }
        let v = self.norm_silu(&v, &self.norm_out)?;
        let v = self.conv3(&v, &self.conv_out)?;
        let (t2, h2, w2) = (v.t, v.h, v.w);
        Ok((v.x.to_f32()?, t2, h2, w2))
    }

    /// Pixels [frames, H, W, 3] in [0, 1] (H, W multiples of 16) -> normalized latents [24, T, H/16, W/16] and T.
    pub fn encode(&self, px: &[f32], frames: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        if !h.is_multiple_of(RATIO) || !w.is_multiple_of(RATIO) || px.len() != frames * h * w * 3 {
            return Err(Error(format!("encode: {} values for {frames} frames of {w}x{h} (multiples of 16)", px.len())));
        }
        let norm: Vec<f32> = px.iter().enumerate().map(|(i, v)| (v - IMAGENET_MEAN[i % 3]) / IMAGENET_STD[i % 3]).collect();
        let frame = h * w * 3;
        let (lh, lw) = (h / RATIO, w / RATIO);
        // the 48 moments [T, lh, lw, 48] of the whole
        let mut all: Vec<f32> = Vec::new();
        let mut total_t = 0;
        let clips: Vec<(usize, usize)> = if frames == 1 { vec![(0, 1)] } else { (0..frames.div_ceil(CLIP)).map(|i| (i * CLIP, CLIP)).collect() };
        for (start, len) in clips {
            // the clip, its last frame repeated to the full length
            let mut clip = Vec::with_capacity(len * frame);
            for k in 0..len {
                let f = (start + k).min(frames - 1);
                clip.extend_from_slice(&norm[f * frame..(f + 1) * frame]);
            }
            let (m, ct) = self.tiled(&clip, len, h, w, tick)?;
            all.extend(m);
            total_t += ct;
        }
        let keep_t = if frames == 1 { 1 } else { total_t - TOKEN_DROP };
        // the 1x1x1 convolution, the mean (first 24 channels), normalized; to [24, T, lh, lw]
        let n = lh * lw;
        let mut z = vec![0f32; 24 * keep_t * n];
        let first_t = if frames == 1 { total_t - 1 } else { 0 };
        for t in 0..keep_t {
            for i in 0..n {
                let m = &all[((first_t + t) * n + i) * 48..][..48];
                for o in 0..24 {
                    let q = self.quant_b[o] + self.quant_w[o * 48..(o + 1) * 48].iter().zip(m).map(|(a, b)| a * b).sum::<f32>();
                    z[(o * keep_t + t) * n + i] = (q - self.mean[o]) / self.std[o];
                }
            }
        }
        Ok((z, keep_t))
    }

    /// One temporal clip through spatial tiles of 256 px, cross-faded in latent space: [t', lh, lw, 48].
    fn tiled(&self, clip: &[f32], t: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let (ys, yo) = split_tiles(h);
        let (xs, xo) = split_tiles(w);
        let (th, tw) = (h.min(TILE), w.min(TILE));
        let (lh, lw) = (h / RATIO, w / RATIO);
        let mut rows: Vec<Vec<Moments>> = Vec::new();
        let mut tt = 0;
        for &y0 in &ys {
            let mut row = Vec::new();
            for &x0 in &xs {
                let mut tile = Vec::with_capacity(t * th * tw * 3);
                for f in 0..t {
                    for yy in 0..th {
                        let src = ((f * h + y0 + yy) * w + x0) * 3;
                        tile.extend_from_slice(&clip[src..src + tw * 3]);
                    }
                }
                tick()?;
                let m = self.moments(&tile, t, th, tw)?;
                tt = m.1;
                row.push(m);
            }
            rows.push(row);
        }
        // blend and place, in latent units, as the reference's tiled_encode does
        let (lyo, lxo): (Vec<usize>, Vec<usize>) = (yo.iter().map(|o| o / RATIO).collect(), xo.iter().map(|o| o / RATIO).collect());
        let mut out = vec![0f32; tt * lh * lw * 48];
        let mut out_y = 0;
        for (i, row) in rows.iter().enumerate() {
            let mut out_x = 0;
            let mut keep_h = 0;
            for (j, (m, _, mh, mw)) in row.iter().enumerate() {
                let kh = if i + 1 < rows.len() { mh - lyo[i] } else { *mh };
                let kw = if j + 1 < row.len() { mw - lxo[j] } else { *mw };
                for ti in 0..tt {
                    for y in 0..kh {
                        for x in 0..kw {
                            let mut v: [f32; 48] = get(m, (*mh, *mw), ti, y, x).try_into().unwrap();
                            // the top rows fade in from the tile above, the left columns from the tile on the left
                            if i > 0 && y < lyo[i - 1].min(*mh) {
                                let e = lyo[i - 1].min(*mh);
                                let (am, _, ah, aw) = &rows[i - 1][j];
                                let a = get(am, (*ah, *aw), ti, ah - e + y, x);
                                let wb = y as f32 / e as f32;
                                v.iter_mut().zip(a).for_each(|(b, a)| *b = a * (1.0 - wb) + *b * wb);
                            }
                            if j > 0 && x < lxo[j - 1].min(*mw) {
                                let e = lxo[j - 1].min(*mw);
                                let (am, _, ah, aw) = &row[j - 1];
                                // the left neighbour as it was encoded (the reference blends with the raw tiles)
                                let a = get(am, (*ah, *aw), ti, y, aw - e + x);
                                let wb = x as f32 / e as f32;
                                v.iter_mut().zip(a).for_each(|(b, a)| *b = a * (1.0 - wb) + *b * wb);
                            }
                            let dst = ((ti * lh + out_y + y) * lw + out_x + x) * 48;
                            out[dst..dst + 48].copy_from_slice(&v);
                        }
                    }
                }
                out_x += kw;
                keep_h = kh;
            }
            out_y += keep_h;
        }
        Ok((out, tt))
    }
}

/// A tile's moments [t, h, w, 48] and (t, h, w).
type Moments = (Vec<f32>, usize, usize, usize);

/// The 48 moments at (t, y, x) of a tile [t, rh, rw, 48].
fn get(r: &[f32], (rh, rw): (usize, usize), t: usize, y: usize, x: usize) -> &[f32] {
    &r[((t * rh + y) * rw + x) * 48..][..48]
}

/// Tile starts and overlaps along one axis (the decoder's rule).
fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>) {
    if TILE >= len {
        return (vec![0], vec![]);
    }
    let mut n = len.div_ceil(TILE);
    let (mut overlaps, remaining) = loop {
        let o = vec![TILE_OVERLAP_MIN; n - 1];
        let sum: usize = o.iter().sum();
        if TILE * n >= sum + len {
            break (o, TILE * n - sum - len);
        }
        n += 1;
    };
    for i in 0..remaining / RATIO {
        let k = i % (n - 1);
        overlaps[k] += RATIO;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        starts.push(starts.last().unwrap() + TILE - o);
    }
    (starts, overlaps)
}
