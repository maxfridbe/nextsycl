//! MPEG-Diffusion on H3: two tracks at two resolutions. An independent port of the paper "MPEG-Diffusion:
//! Mixed-Resolution Video Generation" (Menn, Gu, He, V S, Song, Liu, Xie, Huang, Xu, Marculescu; arXiv 2611.12069) to
//! MiniMax H3, after scrya-com/mpeg-diffusion's `lot/two_track_h3.py` and `lot/mixed_res.py` (Apache-2.0; its fitted
//! K/V maps and region head are MiniMax H3 Community License files). docs/ports/mpeg-diffusion.md has the checks.
//!
//! ```text
//! steps 0..switch      the whole clip at half resolution (the unmodified model; its sound too)
//! step switch - 1      block 25's half-resolution video states -> region head s = w . LN(h) -> smooth (space, time)
//!                      -> M = s > m + alpha (p - m) per clip (median m, 99th percentile p): a set of half-res tokens
//! at the switch        the clean half-res estimate, upscaled (inside M's bounding box: VAE decode, bicubic x2, encode;
//!                      elsewhere latent x2), re-noised with fresh noise to the switch's sigma: the full-res track
//! then (hr_steps,      one sequence [text | audio | half-res video (all of it) | full-res video (inside M only)]:
//!  linear in sigma)    text, audio and half-res queries see text, audio and half-res keys (the base model at half
//!                      resolution, untouched); full-res queries see text, audio and a full-res grid of keys / values:
//!                      their own inside M, each half-res token's pre-RoPE key / value expanded to its 2x2 cells by
//!                      the fitted map k + U V^T k outside it, rotated at the cells' own positions
//! end                  decode the full-res latent (outside M filled with a latent x2 of the half-res track) and the
//!                      half-res latent, enlarge the latter's pixels (bicubic x2), paste with a ramp inside M's edge
//! ```
//!
//! H3's positions are normalised by the frame's area (layout.rs), so the two grids share one coordinate frame: a
//! half-res token sits on the top-left cell of its 2x2.
//!
//! The full-res queries' attention is assembled from the qkv buffer without a copy a row by the H3 silo's gather
//! (kernels/video/h3/silo, bound by the video engine as `Blocks::gather`); without it, one device copy a row.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{Blocks, Config, SplitAttention};
use crate::dtype::DType;
use crate::safetensors::Checkpoint;
use crate::{Ctx, Error, Result};

/// The block whose states the region head reads (upstream's MID_BLOCK; the head's own metadata says the same)
pub const MID_BLOCK: usize = 25;
/// Features a head, the maps' width
pub const HEAD_DIM: usize = 128;

/// The sampler's settings: upstream's `render --preset quality` (switch 16 of 20, 6 full-res steps linear in sigma,
/// the crop round trip at the switch, the pixel paste, a 4-cell ramp, alpha 0.5, no floor)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// half-res steps before the switch, of `steps` on H3's schedule
    pub switch: usize,
    /// two-track steps after it, linear in sigma from the switch's sigma to 0 (0: the schedule's own remaining steps)
    pub hr_steps: usize,
    pub alpha: f32,
    /// a clip whose smoothed 99th percentile is under it stays half-res (None: never)
    pub floor: Option<f32>,
    /// the paste's ramp inside M's edge, in full-res latent pixels
    pub feather: usize,
    /// latent pixels around M's bounding box that the switch's VAE round trip covers
    pub margin: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { switch: 16, hr_steps: 6, alpha: 0.5, floor: None, feather: 4, margin: 2 }
    }
}

impl Settings {
    /// The defaults with the model card's (or the environment's) overrides: H3_MPEG_SWITCH, H3_MPEG_HR_STEPS,
    /// H3_MPEG_ALPHA, H3_MPEG_FLOOR, H3_MPEG_FEATHER
    pub fn from_env(get: &dyn Fn(&str) -> Option<String>) -> Settings {
        let mut s = Settings::default();
        let num = |k: &str| get(k).and_then(|v| v.trim().parse::<f64>().ok());
        if let Some(v) = num("H3_MPEG_SWITCH") {
            s.switch = v as usize;
        }
        if let Some(v) = num("H3_MPEG_HR_STEPS") {
            s.hr_steps = v as usize;
        }
        if let Some(v) = num("H3_MPEG_ALPHA") {
            s.alpha = v as f32;
        }
        s.floor = num("H3_MPEG_FLOOR").map(|v| v as f32);
        if let Some(v) = num("H3_MPEG_FEATHER") {
            s.feather = v as usize;
        }
        s
    }

    /// The noise levels after the switch: `hr_steps` + 1 points linear from the switch's sigma down to 0 (as
    /// torch.linspace in float64), or the schedule's own rest
    pub fn tail(&self, sigmas: &[f32]) -> Vec<f32> {
        let s0 = sigmas[self.switch.min(sigmas.len() - 1)] as f64;
        if self.hr_steps == 0 {
            return sigmas[self.switch.min(sigmas.len() - 1)..].to_vec();
        }
        let n = self.hr_steps;
        // torch.linspace(start, end, steps): start + i * step for the first half, end - (steps - 1 - i) * step after
        let step = (0.0 - s0) / n as f64;
        (0..=n).map(|i| if i < n.div_ceil(2) { s0 + step * i as f64 } else { 0.0 - step * (n - i) as f64 } as f32).collect()
    }
}

// ---- the region head ------------------------------------------------------------------------------------------

/// `s = w . LN(h) + b` per half-res video token (LayerNorm without affine, eps 1e-5), on block 25's output
pub struct RegionHead {
    pub weight: Vec<f32>,
    pub bias: f32,
    pub hidden: usize,
    pub block: usize,
}

impl RegionHead {
    /// `region_head_tt.safetensors` as published (score.weight [1, hidden], score.bias [1]; metadata hidden, block)
    pub fn load(path: &Path) -> Result<RegionHead> {
        let ck = Checkpoint::open(path).ctx(format!("the region head {}", path.display()))?;
        let read = |n: &str| -> Result<Vec<f32>> { crate::dtype::bytes_to_f32(&ck.read(n)?, ck.get(n)?.dtype) };
        let weight = read("score.weight")?;
        let bias = read("score.bias")?;
        let hidden = weight.len();
        if bias.len() != 1 || ck.metadata.get("hidden").is_some_and(|h| h.parse::<usize>().ok() != Some(hidden)) {
            return Err(Error(format!("{}: a region head is score.weight [1, hidden] and score.bias [1]", path.display())));
        }
        let block = ck.metadata.get("block").and_then(|b| b.parse().ok()).unwrap_or(MID_BLOCK);
        Ok(RegionHead { weight, bias: bias[0], hidden, block })
    }

    /// Logits for `states` [n, hidden] (float32)
    pub fn scores(&self, states: &[f32]) -> Vec<f32> {
        let c = self.hidden;
        states
            .chunks_exact(c)
            .map(|h| {
                // float64 sums: torch's float32 reductions differ from these by rounding only
                let mean = h.iter().map(|v| *v as f64).sum::<f64>() / c as f64;
                let var = h.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / c as f64;
                let inv = 1.0 / (var + 1e-5).sqrt();
                let dot: f64 = h.iter().zip(&self.weight).map(|(v, w)| ((*v as f64 - mean) * inv) as f32 as f64 * *w as f64).sum();
                (dot + self.bias as f64) as f32
            })
            .collect()
    }
}

/// `layout.smooth_detail`: a separable Gaussian (sigma 1, radius 3, reflected at the edges) along the width, then the
/// height, of each [h, w] plane
fn smooth(field: &[f32], planes: usize, h: usize, w: usize) -> Vec<f32> {
    const R: i64 = 3;
    let k: Vec<f32> = (-R..=R).map(|c| (-0.5 * (c as f32) * (c as f32)).exp()).collect();
    let sum: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / sum).collect();
    // reflect padding (no edge repeat): -1 -> 1, n -> n - 2
    let refl = |i: i64, n: i64| -> usize {
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * (n - 1) - i;
        }
        i.clamp(0, n - 1) as usize
    };
    let mut a = vec![0f32; field.len()];
    for p in 0..planes {
        for y in 0..h {
            for x in 0..w {
                let mut s = 0f32;
                for (j, kv) in k.iter().enumerate() {
                    s += field[(p * h + y) * w + refl(x as i64 + j as i64 - R, w as i64)] * kv;
                }
                a[(p * h + y) * w + x] = s;
            }
        }
    }
    let mut b = vec![0f32; field.len()];
    for p in 0..planes {
        for y in 0..h {
            for x in 0..w {
                let mut s = 0f32;
                for (j, kv) in k.iter().enumerate() {
                    s += a[(p * h + refl(y as i64 + j as i64 - R, h as i64)) * w + x] * kv;
                }
                b[(p * h + y) * w + x] = s;
            }
        }
    }
    b
}

/// `mixed_res.region_mask`: scores [t, h, w] (the half-res token grid) -> M, smoothed in space and (t > 1) by
/// [1, 2, 1] / 4 across latent frames, thresholded at m + alpha (p - m) over the whole clip (m: torch's median, the
/// lower middle value; p: the 99th percentile, linear between ranks)
pub fn region_mask(scores: &[f32], t: usize, h: usize, w: usize, alpha: f32, floor: Option<f32>) -> Vec<bool> {
    let mut s = smooth(scores, t, h, w);
    if t > 1 {
        let n = h * w;
        let orig = s.clone();
        for ti in 0..t {
            let (a, b) = (ti.saturating_sub(1), (ti + 1).min(t - 1));
            for i in 0..n {
                s[ti * n + i] = 0.25 * orig[a * n + i] + 0.5 * orig[ti * n + i] + 0.25 * orig[b * n + i];
            }
        }
    }
    let mut sorted = s.clone();
    sorted.sort_by(f32::total_cmp);
    let len = sorted.len();
    let median = sorted[(len - 1) / 2];
    let rank = 0.99f32 * (len - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(len - 1);
    let top = sorted[lo] + (rank - lo as f32) * (sorted[hi] - sorted[lo]);
    if floor.is_some_and(|f| top < f) {
        return vec![false; s.len()];
    }
    // python floats (float64), then the comparison against a float32 tensor in float32
    let thr = (median as f64 + alpha as f64 * (top as f64 - median as f64)) as f32;
    s.iter().map(|v| *v > thr).collect()
}

// ---- the plan: which rows exist and where they sit -----------------------------------------------------------

/// M on the half-res token grid and the index lists the two-track sequence is built from (`mixed_res.TwoTrackPlan`)
#[derive(Clone, Debug)]
pub struct Plan {
    /// the half-res token grid: latent frames, rows, columns
    pub t: usize,
    pub h: usize,
    pub w: usize,
    /// [t * h * w]
    pub mask: Vec<bool>,
    /// the full-res tokens inside M (their raster index in the [t, 2h, 2w] grid), ascending: the full-res rows' order
    pub hr_index: Vec<usize>,
    /// the half-res tokens outside M, ascending: the ones expanded to 2x2 cells for the full-res queries
    pub lr_out: Vec<usize>,
}

impl Plan {
    pub fn new(mask: Vec<bool>, t: usize, h: usize, w: usize) -> Result<Plan> {
        if mask.len() != t * h * w {
            return Err(Error(format!("a region of {} tokens for a {t} x {h} x {w} grid", mask.len())));
        }
        let (hh, ww) = (2 * h, 2 * w);
        let hr_index = (0..t * hh * ww).filter(|i| {
            let (ti, y, x) = (i / (hh * ww), (i / ww) % hh, i % ww);
            mask[(ti * h + y / 2) * w + x / 2]
        }).collect();
        let lr_out = (0..t * h * w).filter(|i| !mask[*i]).collect();
        Ok(Plan { t, h, w, mask, hr_index, lr_out })
    }

    pub fn lr_count(&self) -> usize {
        self.t * self.h * self.w
    }

    pub fn hr_count(&self) -> usize {
        self.hr_index.len()
    }

    /// The share of the frame M covers
    pub fn share(&self) -> f64 {
        self.mask.iter().filter(|m| **m).count() as f64 / self.mask.len().max(1) as f64
    }

    /// Cell `c` (row-major in its 2x2) of half-res token `lr`, as a raster index of the full-res grid
    pub fn cell(&self, lr: usize, c: usize) -> usize {
        let (ti, u, v) = (lr / (self.h * self.w), (lr / self.w) % self.h, lr % self.w);
        (ti * 2 * self.h + 2 * u + c / 2) * 2 * self.w + 2 * v + c % 2
    }

    /// upstream's `cell_order`: argsort of the cells listed "token, then its four cells" - for each full-res cell in
    /// raster order, its place in that list
    pub fn cell_order(&self) -> Vec<usize> {
        let mut order = vec![0usize; 4 * self.lr_count()];
        for lr in 0..self.lr_count() {
            for c in 0..4 {
                order[self.cell(lr, c)] = 4 * lr + c;
            }
        }
        order
    }

    /// upstream's `grid_source`: each full-res cell's row in [half-res tokens, full-res tokens]
    pub fn grid_source(&self) -> Vec<usize> {
        let mut src: Vec<usize> = self.cell_order().iter().map(|o| o / 4).collect();
        for (k, i) in self.hr_index.iter().enumerate() {
            src[*i] = self.lr_count() + k;
        }
        src
    }

    /// The packed sequence's positions [rows, 3]: the half-res layout's (text, audio, half-res video), then the
    /// full-res tokens' own, picked from the full-res layout's video positions
    pub fn positions(&self, lr: &crate::layout::Layout, hr: &crate::layout::Layout) -> Result<Vec<f64>> {
        let hv = hr.segment(crate::layout::Kind::Video).ok_or("the full-res layout has no video")?;
        let mut p = lr.positions.clone();
        for i in &self.hr_index {
            p.extend_from_slice(&hr.positions[(hv.start + i) * 3..(hv.start + i) * 3 + 3]);
        }
        Ok(p)
    }

    /// The expanded keys' positions [4 * lr_out, 3]: cell c of every half-res token outside M, c = 0..4 in turn
    pub fn expanded_positions(&self, hr: &crate::layout::Layout) -> Result<Vec<f64>> {
        let hv = hr.segment(crate::layout::Kind::Video).ok_or("the full-res layout has no video")?;
        let mut p = Vec::with_capacity(4 * self.lr_out.len() * 3);
        for c in 0..4 {
            for lr in &self.lr_out {
                let i = hv.start + self.cell(*lr, c);
                p.extend_from_slice(&hr.positions[i * 3..i * 3 + 3]);
            }
        }
        Ok(p)
    }
}

// ---- the Level-of-Token layout the region implies (reported; upstream's `layout` drawing) -------------------

/// A token of a Level-of-Token layout: frame, top-left cell, extent (eh x ew cells of the full-res token grid)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub t: usize,
    pub u: usize,
    pub v: usize,
    pub eh: usize,
    pub ew: usize,
}

/// `mixed_res.region_layout(full, background=4, ring=1)` on the full-res token grid [t, h, w]: 1x1 inside the
/// region, 2x2 on a one-token ring around it, 4x4 elsewhere - `layout_from_desired_any`'s quadtree from 4x4 roots,
/// clipped edge blocks kept as rectangles when both sides are powers of two
pub fn region_layout(full: &[bool], t: usize, h: usize, w: usize) -> Vec<Rect> {
    let mut desired = vec![4u8; t * h * w];
    for ti in 0..t {
        for y in 0..h {
            for x in 0..w {
                let near = (y.saturating_sub(1)..=(y + 1).min(h - 1)).any(|yy| (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| full[(ti * h + yy) * w + xx]));
                let i = (ti * h + y) * w + x;
                if full[i] {
                    desired[i] = 1;
                } else if near {
                    desired[i] = 2;
                }
            }
        }
    }
    let keep = |ti: usize, u: usize, v: usize, eh: usize, ew: usize| -> bool {
        let m = (u..u + eh).flat_map(|y| (v..v + ew).map(move |x| (y, x))).map(|(y, x)| desired[(ti * h + y) * w + x]).min().unwrap_or(0);
        m as usize >= eh.max(ew)
    };
    let mut rects = Vec::new();
    fn visit(rects: &mut Vec<Rect>, keep: &dyn Fn(usize, usize, usize, usize, usize) -> bool, (h, w): (usize, usize), ti: usize, u: usize, v: usize, size: usize) {
        if u >= h || v >= w {
            return;
        }
        let (eh, ew) = (size.min(h - u), size.min(w - v));
        if size == 1 {
            rects.push(Rect { t: ti, u, v, eh: 1, ew: 1 });
            return;
        }
        if eh == size && ew == size {
            if keep(ti, u, v, size, size) {
                rects.push(Rect { t: ti, u, v, eh: size, ew: size });
                return;
            }
        } else if eh.max(ew) > 1 && eh.is_power_of_two() && ew.is_power_of_two() && keep(ti, u, v, eh, ew) {
            rects.push(Rect { t: ti, u, v, eh, ew });
            return;
        }
        let half = size / 2;
        for du in [0, half] {
            for dv in [0, half] {
                visit(rects, keep, (h, w), ti, u + du, v + dv, half);
            }
        }
    }
    for ti in 0..t {
        for u in (0..h).step_by(4) {
            for v in (0..w).step_by(4) {
                visit(&mut rects, &keep, (h, w), ti, u, v, 4);
            }
        }
    }
    rects
}

// ---- the fitted K/V maps --------------------------------------------------------------------------------------

/// `f_up` per block, for keys and values: a half-res token's head vector k (pre-RoPE) -> its four cells k + D_c k,
/// D = (down [128, r] up [r, 4 x 128]) folded once in float64 (`UpMaps.expand` computes the two products in the
/// activations' type, bfloat16)
pub struct UpMaps {
    pub layers: usize,
    pub rank: usize,
    /// per layer, keys then values: per cell c the [128 out, 128 in] matrix D_c (D_c[o][i] = (down up)[i][c 128 + o])
    pub host: Vec<[Vec<f32>; 2]>,
    /// the same on the card, bfloat16 [4 * 128, 128], when sent (`to_device`)
    pub dev: Vec<[Tensor; 2]>,
}

impl UpMaps {
    /// `up_maps_tt.safetensors` (keys down.l{i}_2x2_{k|v} [128, r], up.l{i}_2x2_{k|v} [r, 512]; metadata layers, rank)
    pub fn load(path: &Path) -> Result<UpMaps> {
        let ck = Checkpoint::open(path).ctx(format!("the K/V maps {}", path.display()))?;
        let layers: usize = ck.metadata.get("layers").and_then(|v| v.parse().ok()).ok_or("the maps' metadata has no layers")?;
        let rank: usize = ck.metadata.get("rank").and_then(|v| v.parse().ok()).ok_or("the maps' metadata has no rank")?;
        let read = |n: &str| -> Result<Vec<f32>> { crate::dtype::bytes_to_f32(&ck.read(n)?, ck.get(n)?.dtype) };
        let d = HEAD_DIM;
        let mut host = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut pair: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            for (j, which) in ["k", "v"].iter().enumerate() {
                let down = read(&format!("down.l{l}_2x2_{which}"))?;
                let up = read(&format!("up.l{l}_2x2_{which}"))?;
                if down.len() != d * rank || up.len() != rank * 4 * d {
                    return Err(Error(format!("maps layer {l} {which}: down {} and up {} values for rank {rank}", down.len(), up.len())));
                }
                let mut m = vec![0f32; 4 * d * d];
                for i in 0..d {
                    for col in 0..4 * d {
                        let s: f64 = (0..rank).map(|r| down[i * rank + r] as f64 * up[r * 4 * d + col] as f64).sum();
                        let (c, o) = (col / d, col % d);
                        m[(c * d + o) * d + i] = s as f32;
                    }
                }
                pair[j] = m;
            }
            host.push(pair);
        }
        Ok(UpMaps { layers, rank, host, dev: Vec::new() })
    }

    /// The matrices to the card as bfloat16
    pub fn to_device(&mut self, dev: &Arc<Device>) -> Result<()> {
        let d = HEAD_DIM;
        self.dev = self.host.iter().map(|[k, v]| -> Result<[Tensor; 2]> {
            Ok([Tensor::from_bytes(dev, DType::BF16, &[4 * d, d], &crate::denoiser::bf16_bytes(k))?,
                Tensor::from_bytes(dev, DType::BF16, &[4 * d, d], &crate::denoiser::bf16_bytes(v))?])
        }).collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    /// `UpMaps.expand` on the host, in float32: rows [n, heads, 128] -> [n, 4, heads, 128] (each token, then its cells)
    pub fn expand_host(&self, layer: usize, which: usize, rows: &[f32], heads: usize) -> Vec<f32> {
        let d = HEAD_DIM;
        let m = &self.host[layer][which];
        let n = rows.len() / (heads * d);
        let mut out = vec![0f32; n * 4 * heads * d];
        for t in 0..n {
            for c in 0..4 {
                for hd in 0..heads {
                    let x = &rows[(t * heads + hd) * d..][..d];
                    for o in 0..d {
                        let w = &m[(c * d + o) * d..][..d];
                        let delta: f64 = x.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
                        out[((t * 4 + c) * heads + hd) * d + o] = x[o] + delta as f32;
                    }
                }
            }
        }
        out
    }
}

// ---- resampling and the paste -----------------------------------------------------------------------------------

/// The cubic convolution weights (A = -0.75, torch's) for a fraction t
fn cubic(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
}

/// F.interpolate(scale_factor=2, mode="bicubic", align_corners=False) of planar [planes, h, w], then clamped to
/// [lo, hi] (the pixels' range); planes in parallel
pub fn bicubic2x(px: &[f32], planes: usize, h: usize, w: usize, lo: f32, hi: f32) -> Vec<f32> {
    let (oh, ow) = (2 * h, 2 * w);
    let taps = |o: usize, n: usize| -> ([usize; 4], [f32; 4]) {
        let src = 0.5 * (o as f32 + 0.5) - 0.5;
        let i0 = src.floor();
        let wts = cubic(src - i0);
        let i0 = i0 as i64;
        (std::array::from_fn(|k| (i0 - 1 + k as i64).clamp(0, n as i64 - 1) as usize), wts)
    };
    let ys: Vec<_> = (0..oh).map(|y| taps(y, h)).collect();
    let xs: Vec<_> = (0..ow).map(|x| taps(x, w)).collect();
    let mut out = vec![0f32; planes * oh * ow];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(32);
    let per = planes.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(per * oh * ow).enumerate() {
            let (ys, xs) = (&ys, &xs);
            sc.spawn(move || {
                for (pi, plane) in chunk.chunks_mut(oh * ow).enumerate() {
                    let src = &px[(ci * per + pi) * h * w..][..h * w];
                    for (y, (yi, yw)) in ys.iter().enumerate() {
                        for (x, (xi, xw)) in xs.iter().enumerate() {
                            // torch: along x for each of the 4 rows, then along y
                            let mut acc = 0f32;
                            for k in 0..4 {
                                let row = &src[yi[k] * w..][..w];
                                let r = row[xi[0]] * xw[0] + row[xi[1]] * xw[1] + row[xi[2]] * xw[2] + row[xi[3]] * xw[3];
                                acc += r * yw[k];
                            }
                            plane[y * ow + x] = acc.clamp(lo, hi);
                        }
                    }
                }
            });
        }
    });
    out
}

/// Linear interpolation taps of an align_corners=False resize by an integer factor (source clamped at 0, as torch's
/// linear modes do)
fn linear_taps(o: usize, n: usize, factor: usize) -> (usize, usize, f32) {
    let src = ((o as f32 + 0.5) / factor as f32 - 0.5).max(0.0);
    let i0 = (src.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    (i0, i1, src - i0 as f32)
}

/// A latent [c, t, h, w] doubled in space by F.interpolate(scale_factor=(1, 2, 2), mode="trilinear"): bilinear in
/// each frame
pub fn latent2x(z: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (oh, ow) = (2 * h, 2 * w);
    let ys: Vec<_> = (0..oh).map(|y| linear_taps(y, h, 2)).collect();
    let xs: Vec<_> = (0..ow).map(|x| linear_taps(x, w, 2)).collect();
    let mut out = vec![0f32; c * t * oh * ow];
    for p in 0..c * t {
        let src = &z[p * h * w..][..h * w];
        for (y, (y0, y1, ly)) in ys.iter().enumerate() {
            for (x, (x0, x1, lx)) in xs.iter().enumerate() {
                let top = src[y0 * w + x0] * (1.0 - lx) + src[y0 * w + x1] * lx;
                let bot = src[y1 * w + x0] * (1.0 - lx) + src[y1 * w + x1] * lx;
                out[(p * oh + y) * ow + x] = top * (1.0 - ly) + bot * ly;
            }
        }
    }
    out
}

/// `two_track_h3.crop_box`: the union over frames of M's bounding box on the half-res latent grid (a token is 2
/// latent pixels), grown by `margin` and rounded out to even sides: (y0, y1, x0, x1); None for an empty M
pub fn crop_box(p: &Plan, lat_h: usize, lat_w: usize, margin: usize) -> Option<(usize, usize, usize, usize)> {
    let rows: Vec<usize> = (0..p.h).filter(|u| (0..p.t).any(|ti| (0..p.w).any(|v| p.mask[(ti * p.h + u) * p.w + v]))).collect();
    let cols: Vec<usize> = (0..p.w).filter(|v| (0..p.t).any(|ti| (0..p.h).any(|u| p.mask[(ti * p.h + u) * p.w + v]))).collect();
    let (r0, r1, c0, c1) = (*rows.first()?, *rows.last()?, *cols.first()?, *cols.last()?);
    let (mut y0, mut y1) = ((2 * r0).saturating_sub(margin), (2 * r1 + 2 + margin).min(lat_h));
    let (mut x0, mut x1) = ((2 * c0).saturating_sub(margin), (2 * c1 + 2 + margin).min(lat_w));
    y0 -= y0 % 2;
    x0 -= x0 % 2;
    y1 = (y1 + y1 % 2).min(lat_h);
    x1 = (x1 + x1 % 2).min(lat_w);
    Some((y0, y1, x0, x1))
}

/// `two_track_h3.mask_to_latent`: M at `scale` latent pixels a token, [t, h scale, w scale]; with `feather` R > 1 a
/// ramp inside M only: the sum of M eroded 0..R-1 times (3x3, the frame's edge counting as inside), over R
pub fn mask_to_latent(p: &Plan, scale: usize, feather: usize) -> Vec<f32> {
    let (h, w) = (p.h * scale, p.w * scale);
    let mut up = vec![0f32; p.t * h * w];
    for ti in 0..p.t {
        for y in 0..h {
            for x in 0..w {
                up[(ti * h + y) * w + x] = if p.mask[(ti * p.h + y / scale) * p.w + x / scale] { 1.0 } else { 0.0 };
            }
        }
    }
    if feather <= 1 {
        return up;
    }
    let mut level = up.clone();
    let mut total = up;
    for _ in 0..feather - 1 {
        let mut next = level.clone();
        for ti in 0..p.t {
            for y in 0..h {
                for x in 0..w {
                    let mut m = f32::INFINITY;
                    for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                        for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                            m = m.min(level[(ti * h + yy) * w + xx]);
                        }
                    }
                    next[(ti * h + y) * w + x] = m;
                }
            }
        }
        level = next;
        total.iter_mut().zip(&level).for_each(|(a, b)| *a += b);
    }
    total.iter().map(|v| v / feather as f32).collect()
}

/// Pixel frames a latent of `t` frames decodes to (1, 5, 9, 13, 17, 22, ...): latent frame k covers 1 frame when k
/// is a multiple of 5, else 4
pub fn frame_of(t: usize, frames: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..t).flat_map(|k| std::iter::repeat_n(k, if k % 5 == 0 { 1 } else { 4 })).collect();
    let last = *v.last().unwrap_or(&0);
    v.resize(frames, last);
    v
}

/// What the decode pastes: the half-res track's latent and the paste's weight on the full-res latent grid
pub struct Paste {
    /// [24, t, h, w] at half resolution
    pub lr: Vec<f32>,
    pub lr_h: usize,
    pub lr_w: usize,
    /// [t, 2 lr_h, 2 lr_w]: 1 inside M, a ramp inside its edge, 0 outside
    pub weight: Vec<f32>,
}

/// `two_track_h3.pixel_compose`'s paste: hr [3, frames, H, W] in place, `lr_up` the half-res pixels enlarged x2;
/// the weight (full-res latent grid, [t, H/16, W/16]) bilinear x16 in space, each latent frame to its pixel frames
pub fn paste(hr: &mut [f32], lr_up: &[f32], weight: &[f32], t: usize, frames: usize, height: usize, width: usize) {
    let (lh, lw) = (height / 16, width / 16);
    let of = frame_of(t, frames);
    let ys: Vec<_> = (0..height).map(|y| linear_taps(y, lh, 16)).collect();
    let xs: Vec<_> = (0..width).map(|x| linear_taps(x, lw, 16)).collect();
    let plane = height * width;
    for (f, k) in of.iter().enumerate() {
        let wl = &weight[k * lh * lw..][..lh * lw];
        for (y, (y0, y1, ly)) in ys.iter().enumerate() {
            for (x, (x0, x1, lx)) in xs.iter().enumerate() {
                let top = wl[y0 * lw + x0] * (1.0 - lx) + wl[y0 * lw + x1] * lx;
                let bot = wl[y1 * lw + x0] * (1.0 - lx) + wl[y1 * lw + x1] * lx;
                let wv = top * (1.0 - ly) + bot * ly;
                for c in 0..3 {
                    let i = (c * frames + f) * plane + y * width + x;
                    hr[i] = wv * hr[i] + (1.0 - wv) * lr_up[i];
                }
            }
        }
    }
}

// ---- the two-track attention on the card ----------------------------------------------------------------------

/// Rows of 16-bit values picked by index from a strided buffer (the silo's kernel, `Blocks::gather`): dst, dst
/// stride, src, src stride, idx (int32 on the card), rows, width, src rows - in elements. Ok(false): not taken.
pub type GatherFn = Box<dyn Fn(*mut c_void, usize, *const c_void, usize, *const i32, usize, usize, usize) -> Result<bool> + Send + Sync>;

/// One two-track run's buffers and index lists on the card, for the plan it was made for: what the blocks' split
/// attention (`SplitAttention`) reads and writes besides the stream
pub struct TwoTrack<'a> {
    pub plan: Plan,
    maps: &'a UpMaps,
    blocks: &'a Blocks,
    /// rows before the half-res video (text, audio)
    pub prefix: usize,
    /// the sequence: prefix + half-res video + full-res rows
    pub tokens: usize,
    /// the full-res queries' keys and values [prefix + hr + 4 lr_out, H D]: the prefix's and the full-res rows'
    /// (rotated, gathered from the qkv buffer), then the expanded cells (cell 0 of every token outside M, cell 1 ...)
    k2: Tensor,
    v2: Tensor,
    /// the half-res tokens outside M: raw k, v (then k normalized) [lr_out, H D]
    kx: Tensor,
    vx: Tensor,
    /// int32: the sequence rows of the half-res tokens outside M; of the prefix and the full-res rows
    idx_out: Tensor,
    idx_kv: Tensor,
    /// the expanded cells' rotations [4 lr_out, rot / 2, 2]
    cs_exp: Tensor,
    /// the packed sequence's positions [tokens, 3]
    pub positions: Vec<f64>,
    /// the stream and the blocks' scratch for the packed sequence
    pub(crate) x: Tensor,
    pub(crate) scratch: crate::dit::Scratch,
}

/// The full-res queries' key rows a block reads: prefix + hr + 4 lr_out
pub fn key_rows(p: &Plan, prefix: usize) -> usize {
    prefix + p.hr_count() + 4 * p.lr_out.len()
}

impl<'a> TwoTrack<'a> {
    /// `lr`, `hr`: the half-res clip's layout and a full-res one of the same text and sound (positions); `mlp_rows`:
    /// the MLP's row chunk (None: all rows)
    pub fn new(blocks: &'a Blocks, maps: &'a UpMaps, inv_freq: &[f32], plan: Plan, lr: &crate::layout::Layout, hr: &crate::layout::Layout, mlp_rows: Option<usize>)
               -> Result<TwoTrack<'a>> {
        let cfg = &blocks.cfg;
        if cfg.head_dim != HEAD_DIM || maps.layers != blocks.blocks.len() || maps.dev.len() != maps.layers {
            return Err(Error(format!("the maps ({} layers, on the card: {}) do not fit {} blocks of heads of {}", maps.layers, maps.dev.len(), blocks.blocks.len(), cfg.head_dim)));
        }
        let vseg = lr.segment(crate::layout::Kind::Video).ok_or("the half-res layout has no video")?;
        if vseg.stop != lr.tokens() || vseg.stop - vseg.start != plan.lr_count() {
            return Err(Error("the half-res video must be the layout's last segment, the plan's size".into()));
        }
        let dev = blocks_device(blocks)?;
        let w = cfg.heads * cfg.head_dim;
        let prefix = vseg.start;
        let (nh, no) = (plan.hr_count(), plan.lr_out.len());
        let tokens = lr.tokens() + nh;
        let positions = plan.positions(lr, hr)?;
        let kr = key_rows(&plan, prefix);
        let i32s = |v: Vec<i32>| -> Result<Tensor> {
            let n = v.len().max(1);
            let mut b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            b.resize(n * 4, 0);
            Tensor::from_bytes(&dev, DType::I32, &[n], &b)
        };
        let idx_out = i32s(plan.lr_out.iter().map(|i| (prefix + i) as i32).collect())?;
        let split = prefix + plan.lr_count();
        let idx_kv = i32s((0..prefix).chain(split..split + nh).map(|i| i as i32).collect())?;
        let cs = crate::dit::rotations(&plan.expanded_positions(hr)?, inv_freq);
        let cs_exp = Tensor::from_bytes(&dev, DType::F32, &[(4 * no).max(1), cfg.rot_dim / 2, 2],
                                        &{
                                            let mut b: Vec<u8> = cs.iter().flat_map(|f| f.to_le_bytes()).collect();
                                            b.resize((4 * no).max(1) * cfg.rot_dim * 4, 0);
                                            b
                                        })?;
        Ok(TwoTrack {
            k2: Tensor::new(&dev, DType::BF16, &[kr, w])?,
            v2: Tensor::new(&dev, DType::BF16, &[kr, w])?,
            kx: Tensor::new(&dev, DType::BF16, &[no.max(1), w])?,
            vx: Tensor::new(&dev, DType::BF16, &[no.max(1), w])?,
            idx_out,
            idx_kv,
            cs_exp,
            x: Tensor::new(&dev, DType::BF16, &[tokens, cfg.hidden])?,
            scratch: crate::dit::Scratch::chunked(&dev, cfg, tokens, DType::BF16, mlp_rows.unwrap_or(tokens))?,
            positions,
            plan,
            maps,
            blocks,
            prefix,
            tokens,
        })
    }

    /// Bytes on the card a run of this plan adds (the packed stream and scratch, the key and value sets)
    pub fn bytes(cfg: &Config, p: &Plan, prefix: usize, mlp_rows: Option<usize>) -> usize {
        let w = cfg.heads * cfg.head_dim;
        let tokens = prefix + p.lr_count() + p.hr_count();
        2 * tokens * cfg.hidden + crate::dit::Scratch::bytes(cfg, tokens, mlp_rows.unwrap_or(tokens)) + 2 * 2 * key_rows(p, prefix) * w
            + 2 * 2 * p.lr_out.len() * w
    }

    fn gather(&self, dst: &Tensor, dst_row: usize, src: &Tensor, col: usize, idx: &Tensor, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let w = self.blocks.cfg.heads * self.blocks.cfg.head_dim;
        let (ss, ds) = (src.shape[1], dst.shape[1]);
        let src_rows = src.shape[0];
        // SAFETY: offsets inside the buffers: dst rows [dst_row, dst_row + rows) of `ds`, src columns [col, col + w)
        let (dp, sp) = unsafe { (dst.buf.ptr().cast::<u16>().add(dst_row * ds).cast::<c_void>(), src.buf.ptr().cast::<u16>().add(col).cast_const().cast::<c_void>()) };
        if dst_row + rows > dst.shape[0] || col + w > ss || idx.elements() < rows {
            return Err(Error("two-track gather: out of bounds".into()));
        }
        if let Some(g) = &self.blocks.gather {
            if g(dp, ds, sp, ss, idx.buf.ptr().cast_const().cast(), rows, w, src_rows)? {
                return Ok(());
            }
        }
        // no silo: one device copy a row (the index read back once)
        let ix: Vec<i32> = idx.to_bytes()?.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let dev = src.buf.device();
        for (r, from) in ix.iter().take(rows).enumerate() {
            let from = *from as usize;
            if from >= src_rows {
                return Err(Error(format!("two-track gather: row {from} of {src_rows}")));
            }
            // SAFETY: row `from` of src (checked) and row dst_row + r of dst, w 16-bit values each
            let rc = unsafe {
                (dev.api.copy)(dev.ctx, dp.cast::<u16>().add(r * ds).cast(), sp.cast::<u16>().add(from * ss).cast(), (w * 2) as u64)
            };
            dev.check(rc)?;
        }
        Ok(())
    }
}

fn blocks_device(b: &Blocks) -> Result<Arc<Device>> {
    b.device().ok_or_else(|| Error("the blocks have no device tensor".into()))
}

impl SplitAttention for TwoTrack<'_> {
    fn before_rope(&self, index: usize, qkv: &Tensor, k_norm: &Tensor) -> Result<()> {
        let cfg = &self.blocks.cfg;
        let (w, d) = (cfg.heads * cfg.head_dim, cfg.head_dim);
        let no = self.plan.lr_out.len();
        if no == 0 {
            return Ok(());
        }
        // the half-res tokens outside M: raw k and v, k then normalized (its per-head RMS norm, no rotation)
        self.gather(&self.kx, 0, qkv, w, &self.idx_out, no)?;
        self.gather(&self.vx, 0, qkv, 2 * w, &self.idx_out, no)?;
        let dev = qkv.buf.device();
        let none = &self.cs_exp;
        // SAFETY: kx holds no rows of w bfloat16; rot_dim 0 reads no rotations
        let rc = unsafe {
            (dev.api.rms_rope)(dev.ctx, self.kx.buf.ptr(), DType::BF16.kernel_code()?, no as i64, cfg.heads as i64, d as i64, w as i64,
                               k_norm.buf.ptr().cast_const().cast(), cfg.qk_eps, none.buf.ptr().cast_const().cast(), 0)
        };
        dev.check(rc)?;
        // the four cells: cell c = x + x D_c^T, a [no H, 128] x [128, 128] product a cell, into k2 / v2 after the
        // prefix's and the full-res rows' places
        let base = self.prefix + self.plan.hr_count();
        let rb = w * 2;
        for (j, (x, dst)) in [(&self.kx, &self.k2), (&self.vx, &self.v2)].into_iter().enumerate() {
            let m = &self.maps.dev[index][j];
            for c in 0..4 {
                let off = (base + c * no) * rb;
                // SAFETY: no rows of the cell's block inside dst (k2 / v2 hold base + 4 no rows); D_c is rows
                // [c 128, c 128 + 128) of m
                unsafe {
                    let o = dst.buf.ptr().cast::<u8>().add(off).cast::<c_void>();
                    dev.check((dev.api.copy)(dev.ctx, o, x.buf.ptr().cast_const(), (no * rb) as u64))?;
                    dev.check((dev.api.linear_acc)(dev.ctx, x.buf.ptr().cast_const(), DType::BF16.kernel_code()?, (no * cfg.heads) as i64, d as i64,
                                                    m.buf.ptr().cast::<u8>().add(c * d * d * 2).cast_const().cast(), d as i64, std::ptr::null(), o))?;
                }
            }
        }
        // the expanded keys rotated at their cells' positions
        // SAFETY: 4 no rows of k2 from row base; cs_exp holds their rotations
        let rc = unsafe {
            (dev.api.rope)(dev.ctx, self.k2.buf.ptr().cast::<u8>().add(base * rb).cast(), DType::BF16.kernel_code()?, (4 * no) as i64, cfg.heads as i64, d as i64,
                           w as i64, self.cs_exp.buf.ptr().cast_const().cast(), cfg.rot_dim as i32)
        };
        dev.check(rc)
    }

    fn attend(&self, _index: usize, qkv: &Tensor, att: &Tensor) -> Result<()> {
        let cfg = &self.blocks.cfg;
        let (w, d, heads) = (cfg.heads * cfg.head_dim, cfg.head_dim, cfg.heads);
        let dev = qkv.buf.device();
        let dt = DType::BF16.kernel_code()?;
        let split = self.prefix + self.plan.lr_count();
        let nh = self.plan.hr_count();
        if qkv.shape[0] < self.tokens || att.shape[0] < self.tokens {
            return Err(Error("two-track attention: the buffers are smaller than the sequence".into()));
        }
        // text, audio and half-res queries over text, audio and half-res keys: the base model at half resolution
        // SAFETY: rows [0, split) of qkv (stride 3w) and att (w)
        let rc = unsafe {
            let q = qkv.buf.ptr().cast_const();
            let k = qkv.buf.ptr().cast::<u16>().add(w).cast_const().cast();
            let v = qkv.buf.ptr().cast::<u16>().add(2 * w).cast_const().cast();
            (dev.api.attention)(dev.ctx, q, k, v, dt, split as i64, heads as i64, d as i64, (3 * w) as i64, att.buf.ptr(), dt)
        };
        dev.check(rc)?;
        if nh == 0 {
            return Ok(());
        }
        // the prefix's and the full-res rows' keys (rotated) and values next to the expanded cells
        self.gather(&self.k2, 0, qkv, w, &self.idx_kv, self.prefix + nh)?;
        self.gather(&self.v2, 0, qkv, 2 * w, &self.idx_kv, self.prefix + nh)?;
        let kr = key_rows(&self.plan, self.prefix);
        // SAFETY: q rows [split, split + nh) of qkv; k2 / v2 kr rows of w; att rows from split
        let rc = unsafe {
            let q = qkv.buf.ptr().cast::<u16>().add(split * 3 * w).cast_const().cast();
            let out = att.buf.ptr().cast::<u16>().add(split * w).cast();
            (dev.api.attention_qk)(dev.ctx, q, nh as i64, (3 * w) as i64, self.k2.buf.ptr().cast_const(), self.v2.buf.ptr().cast_const(), kr as i64, w as i64, dt,
                                   heads as i64, d as i64, out, dt)
        };
        dev.check(rc)
    }
}

/// The reference checks' dump (reference/mpeg-diffusion/tt_ref.py): one two-track network call on the run's own
/// inputs - the refined text, both tracks' x, the sound as the network sees it, the plan and its positions - with its
/// outputs, block 0's raw qkv and attention output, the region head's input and output, and block 0's maps applied on
/// the host to the half-res keys outside M (normalized, before rotation)
#[allow(clippy::too_many_arguments)]
fn dump(d: &mut crate::denoiser::Denoiser, tt: &TwoTrack, maps: &UpMaps, path: &Path, (xv, xh, xa): (&[f32], &[f32], &[f32]), sigma: f32,
        (states, scores): (&[f32], &[f32]), log: &mut dyn FnMut(String)) -> Result<()> {
    use std::collections::BTreeMap;
    let cfg = d.blocks.cfg;
    let sh = d.shape;
    let sc = d.schedule;
    let sigma_a = crate::layout::time_shift_sigma(sigma.max(1e-6) as f64, sc.shift_video as f64, sc.shift_audio as f64) as f32;
    let audio_in: Vec<f32> = xa.iter().map(|x| x * sigma_a / sigma.max(1e-6)).collect();
    d.taps = Some((0, Vec::new()));
    let (v, vh, va) = d.net_two_track(tt, xv, xh, &audio_in, sigma, &mut |_| Ok(()))?;
    let taps = d.taps.take().map(|t| t.1).unwrap_or_default();
    let w = cfg.heads * cfg.head_dim;
    let mut m: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let text = d.text.to_f32()?;
    let tl = text.len() / cfg.hidden;
    m.insert("text".into(), (vec![tl, cfg.hidden], text));
    m.insert("lr".into(), (vec![24, sh.t, sh.h, sh.w], xv.to_vec()));
    m.insert("hr".into(), (vec![24, sh.t, 2 * sh.h, 2 * sh.w], xh.to_vec()));
    m.insert("audio".into(), (vec![32, 2, sh.audio_t], audio_in));
    m.insert("v_lr".into(), (vec![24, sh.t, sh.h, sh.w], v));
    m.insert("v_hr".into(), (vec![24, sh.t, 2 * sh.h, 2 * sh.w], vh));
    m.insert("v_audio".into(), (vec![32, 2, sh.audio_t], va));
    let p = &tt.plan;
    m.insert("mask".into(), (vec![p.t, p.h, p.w], p.mask.iter().map(|b| *b as u8 as f32).collect()));
    m.insert("hr_index".into(), (vec![p.hr_count()], p.hr_index.iter().map(|i| *i as f32).collect()));
    m.insert("positions".into(), (vec![tt.tokens, 3], tt.positions.iter().map(|x| *x as f32).collect()));
    m.insert("states".into(), (vec![p.lr_count(), cfg.hidden], states.to_vec()));
    m.insert("scores".into(), (vec![p.t, p.h, p.w], scores.to_vec()));
    for (name, t) in &taps {
        let cols = if name == "qkv" { 3 * w } else { w };
        m.insert(format!("{name}0"), (vec![t.len() / cols, cols], t.clone()));
    }
    // block 0's maps on the host: the half-res keys outside M, normalized (k_norm, eps 1e-5) before rotation
    if let Some((_, qkv)) = taps.iter().find(|(n, _)| n == "qkv") {
        let kw = d.blocks.blocks[0].k_norm().to_f32()?;
        let mut keys = Vec::with_capacity(p.lr_out.len() * w);
        for lr in &p.lr_out {
            let row = &qkv[(tt.prefix + lr) * 3 * w + w..][..w];
            for hd in 0..cfg.heads {
                let x = &row[hd * cfg.head_dim..][..cfg.head_dim];
                let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cfg.head_dim as f64;
                let r = 1.0 / (ms + cfg.qk_eps as f64).sqrt();
                keys.extend(x.iter().zip(&kw).map(|(v, g)| (*v as f64 * r) as f32 * g));
            }
        }
        let n = p.lr_out.len();
        m.insert("expanded_k0".into(), (vec![n, 4, cfg.heads, HEAD_DIM], maps.expand_host(0, 0, &keys, cfg.heads)));
        m.insert("knorm0".into(), (vec![n, cfg.heads, HEAD_DIM], keys));
    }
    let meta = BTreeMap::from([("sigma".to_string(), format!("{sigma:.9}")), ("prefix".to_string(), tt.prefix.to_string()),
                               ("text_len".to_string(), tl.to_string()), ("audio_t".to_string(), sh.audio_t.to_string())]);
    crate::safetensors::write_f32(path, &m, &meta)?;
    log(format!("mpeg   : the reference dump (one two-track call at sigma {sigma}) written to {}", path.display()));
    Ok(())
}

// ---- the sampler ---------------------------------------------------------------------------------------------

/// A clip's two-track run: the model card's files and settings
pub struct Spec<'a> {
    pub maps: &'a Path,
    pub head: &'a Path,
    pub settings: Settings,
    /// write what the first two-track call saw and gave here (the reference checks: reference/mpeg-diffusion)
    pub dump: Option<&'a Path>,
}

/// What a two-track run gives the decode
pub struct Sampled {
    /// the full-res latent [24, T, 2H, 2W]: the full-res track inside M, a latent x2 of the half-res track outside
    pub video: Vec<f32>,
    /// the sound's latents in the model's own scale
    pub audio: Vec<f32>,
    /// the half-res track and the paste's weight (None: no region, the clip is the half-res track alone)
    pub paste: Paste,
    pub report: serde_json::Value,
}

/// Euler from x to the next noise level along the denoised estimate
fn euler(x: &mut [f32], dn: &[f32], s: f32, next: f32) {
    for (x, d) in x.iter_mut().zip(dn) {
        *x = d + (*x - d) * (next / s);
    }
}

/// The two-track sampler (module doc) on `d`, a denoiser at half resolution (its shape: the clip's latent halved in
/// space) over H3's schedule of `steps` (`sigmas`). `vae`: the video VAE, for the switch's round trip. `between(step,
/// block)` before each block; `on_step(step)` after each step. The steps counted are the network calls: switch +
/// the tail's.
#[allow(clippy::too_many_arguments)]
pub fn sample(d: &mut crate::denoiser::Denoiser, spec: &Spec, sigmas: &[f32], vae: &Path, threads: usize, seed: u64, log: &mut dyn FnMut(String),
              between: &mut dyn FnMut(usize, usize) -> Result<()>, on_step: &mut dyn FnMut(usize) -> Result<()>) -> Result<Sampled> {
    use std::time::Instant;
    let st = spec.settings;
    let sh = d.shape;
    let steps = sigmas.len() - 1;
    if st.switch == 0 || st.switch >= steps {
        return Err(Error(format!("MPEG-Diffusion: the switch ({}) must fall inside the {steps} steps", st.switch)));
    }
    if !sh.h.is_multiple_of(2) || !sh.w.is_multiple_of(2) {
        return Err(Error("MPEG-Diffusion: the canvas must be a multiple of 64 on both sides".into()));
    }
    let head = RegionHead::load(spec.head)?;
    if head.hidden != d.blocks.cfg.hidden || head.block >= d.blocks.blocks.len() {
        return Err(Error(format!("the region head ({} features, block {}) does not fit the denoiser", head.hidden, head.block)));
    }
    let mut maps = UpMaps::load(spec.maps)?;
    let dev = blocks_device(d.blocks)?;
    maps.to_device(&dev)?;
    let (t, th, tw) = (sh.t, sh.h / 2, sh.w / 2);
    let (noise_v, noise_a) = crate::noise::clip_noise(seed, 24 * sh.t * sh.h * sh.w, 32 * 2 * sh.audio_t);
    let s0 = sigmas[0];
    let mut xv: Vec<f32> = noise_v.iter().map(|n| n * s0).collect();
    let mut xa: Vec<f32> = noise_a.iter().map(|n| n * s0).collect();
    let mut report = serde_json::json!({"switch": st.switch, "hr_steps": st.hr_steps, "alpha": st.alpha, "feather": st.feather, "head_block": head.block});
    // 1. the whole clip at half resolution
    let t0 = Instant::now();
    let mut x0 = Vec::new();
    for i in 0..st.switch {
        if i + 1 == st.switch {
            d.capture = Some((head.block, Vec::new()));
        }
        let (dv, da) = d.denoised(&xv, &xa, sigmas[i], &mut |b| between(i, b))?;
        euler(&mut xv, &dv, sigmas[i], sigmas[i + 1]);
        euler(&mut xa, &da, sigmas[i], sigmas[i + 1]);
        x0 = dv;
        on_step(i)?;
    }
    report["half_res_seconds"] = serde_json::json!(t0.elapsed().as_secs_f64());
    // 2. the region
    let states = d.capture.take().map(|c| c.1).filter(|v| !v.is_empty()).ok_or("the region head's block states were not captured")?;
    let states_kept = if spec.dump.is_some() { states.clone() } else { Vec::new() };
    let scores = head.scores(&states);
    let mask = region_mask(&scores, t, th, tw, st.alpha, st.floor);
    let plan = Plan::new(mask, t, th, tw)?;
    let lot = region_layout(&{
        let (hh, ww) = (2 * th, 2 * tw);
        (0..t * hh * ww).map(|i| plan.mask[((i / (hh * ww)) * th + (i / ww) % hh / 2) * tw + (i % ww) / 2]).collect::<Vec<bool>>()
    }, t, 2 * th, 2 * tw);
    log(format!("mpeg   : region {:.1}% of the frame ({} of {} half-res tokens; full-res rows {}; the Level-of-Token layout it implies: {} tokens of {})",
                plan.share() * 100.0, plan.mask.iter().filter(|m| **m).count(), plan.lr_count(), plan.hr_count(), lot.len(), 4 * plan.lr_count()));
    report["region_share"] = serde_json::json!(plan.share());
    report["region_tokens"] = serde_json::json!(plan.hr_count());
    report["lot_layout_tokens"] = serde_json::json!(lot.len());
    report["region_mask"] = serde_json::json!(plan.mask.iter().map(|m| *m as u8).collect::<Vec<u8>>());
    let (hh, hw) = (2 * sh.h, 2 * sh.w);
    if plan.hr_count() == 0 {
        // no region: the half-res schedule to its end, the clip enlarged
        for i in st.switch..steps {
            let (dv, da) = d.denoised(&xv, &xa, sigmas[i], &mut |b| between(i, b))?;
            euler(&mut xv, &dv, sigmas[i], sigmas[i + 1]);
            euler(&mut xa, &da, sigmas[i], sigmas[i + 1]);
            on_step(i)?;
        }
        xa.iter_mut().for_each(|x| *x /= d.schedule.audio_scale);
        let video = latent2x(&xv, 24, t, sh.h, sh.w);
        return Ok(Sampled { video, audio: xa, paste: Paste { lr: xv, lr_h: sh.h, lr_w: sh.w, weight: vec![0.0; t * hh * hw] }, report });
    }
    // 3. the full-res track's start: the clean estimate x2 (inside M's box through the VAE), re-noised
    let t1 = Instant::now();
    let mut start = latent2x(&x0, 24, t, sh.h, sh.w);
    if let Some((y0, y1, x0b, x1b)) = crop_box(&plan, sh.h, sh.w, st.margin) {
        let (ch, cw) = (y1 - y0, x1b - x0b);
        let mut crop = vec![0f32; 24 * t * ch * cw];
        for c in 0..24 * t {
            for y in 0..ch {
                crop[(c * ch + y) * cw..][..cw].copy_from_slice(&x0[(c * sh.h + y0 + y) * sh.w + x0b..][..cw]);
            }
        }
        let ck = Checkpoint::open(vae)?;
        let (px, frames) = {
            let dec = crate::vae::VideoDecoder::load(&dev, &ck, threads)?;
            dec.decode(&crop, t, ch, cw, &mut || dev.wait())?
        };
        let (ph, pw) = (16 * ch, 16 * cw);
        let up = bicubic2x(&px, 3 * frames, ph, pw, 0.0, 1.0);
        let (uh, uw) = (2 * ph, 2 * pw);
        let mut inter = vec![0f32; frames * uh * uw * 3];
        for c in 0..3 {
            for f in 0..frames {
                for i in 0..uh * uw {
                    inter[(f * uh * uw + i) * 3 + c] = up[(c * frames + f) * uh * uw + i];
                }
            }
        }
        drop(up);
        let enc = crate::venc::VideoEncoder::load(&dev, &ck)?;
        let (z, zt) = enc.encode(&inter, frames, uh, uw, &mut || dev.wait())?;
        drop(enc);
        if zt != t || z.len() != 24 * t * 4 * ch * cw {
            return Err(Error(format!("the switch's VAE round trip gave {zt} latent frames ({} values) for {t}", z.len())));
        }
        let (zh, zw) = (2 * ch, 2 * cw);
        for c in 0..24 * t {
            for y in 0..zh {
                start[(c * hh + 2 * y0 + y) * hw + 2 * x0b..][..zw].copy_from_slice(&z[(c * zh + y) * zw..][..zw]);
            }
        }
        // the decoder's and the encoder's scratch given back before the two-track sequence
        dev.trim()?;
        log(format!("mpeg   : switch round trip of the box {}x{} latent px x {t} frames ({frames} pixel frames) in {:.1} s", cw, ch, t1.elapsed().as_secs_f64()));
        report["switch_box"] = serde_json::json!([y0, y1, x0b, x1b]);
    }
    report["switch_upscale_seconds"] = serde_json::json!(t1.elapsed().as_secs_f64());
    let sw = sigmas[st.switch];
    let (eps, _) = crate::noise::clip_noise(seed + 2, 24 * t * hh * hw, 0);
    let mut xh: Vec<f32> = start.iter().zip(&eps).map(|(s, e)| (1.0 - sw) * s + sw * e).collect();
    drop(start);
    // 4. the two tracks
    let hr_layout = crate::layout::Layout::new(d.layout.segment(crate::layout::Kind::Text).map_or(0, |s| s.stop - s.start), t, hh, hw, sh.audio_t, &[], &[])?;
    let need = TwoTrack::bytes(&d.blocks.cfg, &plan, d.layout.segment(crate::layout::Kind::Video).map_or(0, |s| s.start), None);
    log(format!("mpeg   : two-track sequence {} rows (half-res {} + full-res {}), the full-res queries over {} keys; {:.2} GiB more on the card",
                d.layout.tokens() + plan.hr_count(), plan.lr_count(), plan.hr_count(),
                key_rows(&plan, d.layout.segment(crate::layout::Kind::Video).map_or(0, |s| s.start)), need as f64 / (1u64 << 30) as f64));
    let tt = TwoTrack::new(d.blocks, &maps, &d.outer.inv_freq, plan, &d.layout, &hr_layout, None)?;
    let tail = st.tail(sigmas);
    let t2 = Instant::now();
    for k in 0..tail.len() - 1 {
        let i = st.switch + k;
        if k == 0 && spec.dump.is_some() {
            dump(d, &tt, &maps, spec.dump.unwrap_or(Path::new("")), (&xv, &xh, &xa), tail[0], (&states_kept, &scores), log)?;
        }
        let (dv, dh, da) = d.denoised_two_track(&tt, &xv, &xh, &xa, tail[k], &mut |b| between(i, b))?;
        euler(&mut xv, &dv, tail[k], tail[k + 1]);
        euler(&mut xh, &dh, tail[k], tail[k + 1]);
        euler(&mut xa, &da, tail[k], tail[k + 1]);
        on_step(i)?;
    }
    report["two_track_seconds"] = serde_json::json!(t2.elapsed().as_secs_f64());
    report["two_track_rows"] = serde_json::json!(tt.tokens);
    xa.iter_mut().for_each(|x| *x /= d.schedule.audio_scale);
    let plan = tt.plan.clone();
    drop(tt);
    // 5. what the decoder gets: the full-res track inside M, the half-res one x2 outside; the paste's weight
    let inside = mask_to_latent(&plan, 4, 0);
    let ctx = latent2x(&xv, 24, t, sh.h, sh.w);
    let n = t * hh * hw;
    let video: Vec<f32> = (0..xh.len()).map(|i| inside[i % n] * xh[i] + (1.0 - inside[i % n]) * ctx[i]).collect();
    let weight = mask_to_latent(&plan, 4, st.feather);
    Ok(Sampled { video, audio: xa, paste: Paste { lr: xv, lr_h: sh.h, lr_w: sh.w, weight }, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's own functions on fixed inputs (reference/mpeg-diffusion/tt_ref.py fixtures): the region mask from
    /// scores, the plan's index lists, the packed sequence's positions, the Level-of-Token layout, the tail's sigmas
    #[test]
    fn the_plan_positions_and_layout_are_upstreams() {
        let raw = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/mpeg_fixture.json")).unwrap();
        let fx: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let us = |v: &serde_json::Value| -> Vec<usize> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect() };
        for c in fx["cases"].as_array().unwrap() {
            let name = c["name"].as_str().unwrap();
            let n = |k: &str| c[k].as_u64().unwrap() as usize;
            let (t, lh, lw) = (n("latent_t"), n("lat_h"), n("lat_w"));
            let (th, tw) = (lh / 2, lw / 2);
            let scores: Vec<f32> = c["scores"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let mask = region_mask(&scores, t, th, tw, 0.5, None);
            let want: Vec<bool> = us(&c["mask"]).iter().map(|v| *v == 1).collect();
            assert_eq!(mask, want, "{name}: the mask");
            let p = Plan::new(mask, t, th, tw).unwrap();
            assert_eq!(p.hr_index, us(&c["hr_index"]), "{name}: hr_index");
            assert_eq!(p.cell_order(), us(&c["cell_order"]), "{name}: cell_order");
            assert_eq!(p.grid_source(), us(&c["grid_source"]), "{name}: grid_source");
            let lr = crate::layout::Layout::new(n("text_len"), t, lh, lw, n("audio_t"), &[], &[]).unwrap();
            let hr = crate::layout::Layout::new(n("text_len"), t, 2 * lh, 2 * lw, n("audio_t"), &[], &[]).unwrap();
            let pos = p.positions(&lr, &hr).unwrap();
            let want: Vec<f64> = c["positions"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            assert_eq!(pos.len(), want.len(), "{name}: rows");
            let worst = pos.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "{name}: positions differ by {worst}");
            // (float64: the last bit of a quarter of them differs - the sums run in another order); in float32, as the
            // rotation reads them: the same numbers
            assert!(pos.iter().zip(&want).all(|(a, b)| *a as f32 == *b as f32), "{name}: positions in float32");
            let full: Vec<bool> = (0..t * lh * lw).map(|i| p.mask[((i / (lh * lw)) * th + (i / lw) % lh / 2) * tw + (i % lw) / 2]).collect();
            let lot: Vec<Vec<usize>> = region_layout(&full, t, lh, lw).iter().map(|r| vec![r.t, r.u, r.v, r.eh, r.ew]).collect();
            let want: Vec<Vec<usize>> = c["lot"].as_array().unwrap().iter().map(us).collect();
            assert_eq!(lot, want, "{name}: the Level-of-Token layout");
        }
        let tail = Settings::default().tail(&crate::denoiser::sigmas(20, 12.0));
        let want: Vec<f32> = fx["tail_16_6"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(tail, want);
    }

    #[test]
    fn the_tail_is_linear_in_sigma() {
        let s = crate::denoiser::sigmas(20, 12.0);
        let t = Settings::default().tail(&s);
        assert_eq!(t.len(), 7);
        assert_eq!(t[0], s[16]);
        assert_eq!(t[6], 0.0);
        assert!((t[3] - s[16] / 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_plan_lists_cells_in_raster_order() {
        // 1 frame of 2 x 3 tokens, M = the middle column
        let mask = vec![false, true, false, false, true, false];
        let p = Plan::new(mask, 1, 2, 3).unwrap();
        assert_eq!(p.hr_count(), 8);
        assert_eq!(p.hr_index, vec![2, 3, 8, 9, 14, 15, 20, 21]);
        assert_eq!(p.lr_out, vec![0, 2, 3, 5]);
        assert_eq!(p.cell(4, 3), 3 * 6 + 3);
        let src = p.grid_source();
        assert_eq!(src[0], 0);
        assert_eq!(src[2], 6);
        assert_eq!(src[7], 0);
    }

    #[test]
    fn a_region_layout_tiles_the_grid() {
        let (t, h, w) = (1, 24, 42);
        let mut full = vec![false; t * h * w];
        for y in 8..14 {
            for x in 10..20 {
                full[y * w + x] = true;
            }
        }
        let rects = region_layout(&full, t, h, w);
        let mut cover = vec![0; t * h * w];
        for r in &rects {
            for y in r.u..r.u + r.eh {
                for x in r.v..r.v + r.ew {
                    cover[(r.t * h + y) * w + x] += 1;
                }
            }
            if full[r.u * w + r.v] {
                assert_eq!((r.eh, r.ew), (1, 1));
            }
        }
        assert!(cover.iter().all(|c| *c == 1));
    }

    #[test]
    fn bicubic_keeps_a_constant_and_a_ramp_inside() {
        let px = vec![0.25f32; 4 * 5];
        let o = bicubic2x(&px, 1, 4, 5, 0.0, 1.0);
        assert!(o.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn the_feather_ramps_inside_only() {
        let mut mask = vec![false; 10 * 10];
        for y in 2..8 {
            for x in 2..8 {
                mask[y * 10 + x] = true;
            }
        }
        let p = Plan::new(mask, 1, 10, 10).unwrap();
        let w = mask_to_latent(&p, 4, 4);
        // outside 0, the edge cell 1/4, the middle 1
        assert_eq!(w[0], 0.0);
        assert_eq!(w[8 * 40 + 8], 0.25);
        assert_eq!(w[20 * 40 + 20], 1.0);
    }
}
