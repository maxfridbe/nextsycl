//! The decoder: HunyuanVideo-1.5's 3-D causal VAE (the official AutoencoderKLConv3D's Decoder), latents [F, h, w, 32]
//! to RGB frames [4 (F - 1) + 1, 16 h, 16 w, 3]. Channels-last half on the GPU (the pipeline decodes in float16 too),
//! the weights float32 in the file, half on the card.
//!
//! ```text
//!   z = latents / 1.03682
//!   h = conv_in(z) (3x3x3 causal, 32 -> 1024) + z repeated 32 times per channel
//!   middle: resnet, attention (one head of 1024 over every voxel, a frame seeing itself and the frames before), resnet
//!   5 levels (1024, 1024, 512, 256, 128): 3 resnets each; then upsample: levels 0, 1 in time and space (a frame
//!     becomes two - the first one), levels 2, 3 in space: a causal conv to 8 C' (4 C') rearranged, plus the input
//!     repeated per channel and rearranged the same way
//!   RMS norm (over the channels, x sqrt(C) gamma), swish, conv_out (3x3x3 causal, 128 -> 3)
//! resnet: x + conv2(swish(norm2(conv1(swish(norm1(x)))))), a 1x1x1 conv on the shortcut when the width changes
//! causal conv: two copies of the first frame before it, the frame's border repeated around it (replicate padding)
//! ```
//!
//! The pipeline decodes under memory_efficient_context: latents wider or taller than 16 are cut in tiles of 16 x 16
//! latents (256 pixels) every 14, each decoded whole in time, blended over 32 pixels with its upper and left
//! neighbours and cut to 224 - done the same here, the blending on the host.

use std::path::Path;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};

use crate::ffi::K;
use crate::w::{f32_to_f16, St};

pub const SCALING: f32 = 1.03682;
pub const Z: usize = 32;
const CHANNELS: [usize; 5] = [1024, 1024, 512, 256, 128];
/// tiles: 16 latents (256 pixels), a new one every 14 latents, blended over 32 pixels, kept to 224
pub const TILE: usize = 16;
const TILE_STEP: usize = 14;
const BLEND: usize = 32;
const KEEP: usize = 224;
/// the attention's score rows a call takes: at most this many bytes of float32 scores
const SCORE_BYTES: usize = 512 << 20;

/// A convolution: half weights [Co, Ci, k, k, k] (k 3, causal) or [Co, Ci] (k 1: a linear), float32 bias
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Resnet {
    norm1: DevBuf,
    conv1: Conv,
    norm2: DevBuf,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Att {
    norm: DevBuf,
    q: Conv,
    k: Conv,
    v: Conv,
    proj: Conv,
}

struct Level {
    resnets: Vec<Resnet>,
    /// the upsample's conv and whether it doubles time too
    up: Option<(Conv, bool)>,
}

pub struct Vae {
    k: K,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    levels: Vec<Level>,
    norm_out: DevBuf,
    conv_out: Conv,
    pub bytes: usize,
}

/// A channels-last volume of half on the card
pub struct Vol {
    pub b: DevBuf,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Vol {
    fn new(gpu: &std::sync::Arc<nextsycl_core::Gpu>, t: usize, h: usize, w: usize, c: usize) -> Result<Vol> {
        Ok(Vol { b: DevBuf::new(gpu, t * h * w * c * 2)?, t, h, w, c })
    }
    fn n(&self) -> usize {
        self.t * self.h * self.w
    }
}

impl Vae {
    pub fn load(file: &Path, k: &K, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let gpu = &k.gpu;
        let st = St::open(file)?;
        let mut bytes = 0usize;
        let mut conv = |name: &str| -> Result<Conv> {
            let (w, shape) = st.get(&format!("{name}.weight"))?;
            let (co, ci, kk) = match shape.as_slice() {
                [co, ci, a, b, c] if a == b && b == c => (*co, *ci, *a),
                s => return Err(Error(format!("{name}.weight: {s:?}, not a cubic 3-D convolution"))),
            };
            let half: Vec<u8> = w.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect();
            let wb = DevBuf::new(gpu, half.len())?;
            wb.write(0, &half)?;
            let b = DevBuf::from_f32(gpu, &st.vec(&format!("{name}.bias"))?)?;
            bytes += wb.len + b.len;
            Ok(Conv { w: wb, b, ci, co, k: kk })
        };
        let gamma = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &st.vec(&format!("{name}.gamma"))?) };
        let resnet = |p: &str, conv: &mut dyn FnMut(&str) -> Result<Conv>| -> Result<Resnet> {
            Ok(Resnet {
                norm1: gamma(&format!("{p}.norm1"))?,
                conv1: conv(&format!("{p}.conv1.conv"))?,
                norm2: gamma(&format!("{p}.norm2"))?,
                conv2: conv(&format!("{p}.conv2.conv"))?,
                shortcut: if st.has(&format!("{p}.nin_shortcut.weight")) { Some(conv(&format!("{p}.nin_shortcut"))?) } else { None },
            })
        };
        let conv_in = conv("decoder.conv_in.conv")?;
        if conv_in.ci != Z || conv_in.co != CHANNELS[0] {
            return Err(Error(format!("{}: decoder.conv_in {} -> {}: not HunyuanVideo-1.5's VAE", file.display(), conv_in.ci, conv_in.co)));
        }
        let mid = [resnet("decoder.mid.block_1", &mut conv)?, resnet("decoder.mid.block_2", &mut conv)?];
        let att = Att {
            norm: gamma("decoder.mid.attn_1.norm")?,
            q: conv("decoder.mid.attn_1.q")?,
            k: conv("decoder.mid.attn_1.k")?,
            v: conv("decoder.mid.attn_1.v")?,
            proj: conv("decoder.mid.attn_1.proj_out")?,
        };
        let mut levels = Vec::new();
        for (l, ch) in CHANNELS.iter().enumerate() {
            let mut resnets = Vec::new();
            for b in 0.. {
                let p = format!("decoder.up.{l}.block.{b}");
                if !st.has(&format!("{p}.conv1.conv.weight")) {
                    break;
                }
                resnets.push(resnet(&p, &mut conv)?);
            }
            let up = if st.has(&format!("decoder.up.{l}.upsample.conv.conv.weight")) {
                let c = conv(&format!("decoder.up.{l}.upsample.conv.conv"))?;
                let next = CHANNELS.get(l + 1).copied().unwrap_or(*ch);
                let temporal = c.co == 8 * next;
                if !temporal && c.co != 4 * next {
                    return Err(Error(format!("decoder.up.{l}.upsample: {} channels for {next}", c.co)));
                }
                Some((c, temporal))
            } else {
                None
            };
            levels.push(Level { resnets, up });
        }
        let norm_out = gamma("decoder.norm_out")?;
        let conv_out = conv("decoder.conv_out.conv")?;
        log(format!("vae: the decoder ({:.2} GiB, half) on {} in {:.1} s", bytes as f64 / (1u64 << 30) as f64, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(Vae { k: k.clone(), conv_in, mid, att, levels, norm_out, conv_out, bytes })
    }

    /// A causal 3x3x3 convolution (or a 1x1x1 one: a linear over the channels)
    fn conv(&self, nsd: &Nsd, c: &Conv, x: &Vol) -> Result<Vol> {
        let gpu = &nsd.gpu;
        if x.c != c.ci {
            return Err(Error(format!("a convolution of {} channels on {}", c.ci, x.c)));
        }
        let o = Vol::new(gpu, x.t, x.h, x.w, c.co)?;
        if c.k == 1 {
            nsd.linear(x.b.ptr(), Dt::F16, x.n(), c.ci, c.w.ptr(), c.co, c.b.ptr(), o.b.ptr(), Dt::F16)?;
            return Ok(o);
        }
        let p = DevBuf::new(gpu, (x.t + 2) * (x.h + 2) * (x.w + 2) * x.c * 2)?;
        self.k.pad_causal(x.b.ptr(), x.t, x.h, x.w, x.c, p.ptr())?;
        nsd.conv3d_ex(p.ptr(), Dt::F16, (x.t + 2, x.h + 2, x.w + 2), c.ci, c.w.ptr(), c.co, (3, 3, 3), (1, 1, 1), c.b.ptr(), o.b.ptr())?;
        Ok(o)
    }

    fn norm(&self, nsd: &Nsd, x: &Vol, gamma: &DevBuf, silu: bool) -> Result<Vol> {
        let o = Vol::new(&nsd.gpu, x.t, x.h, x.w, x.c)?;
        self.k.rms_channels(x.b.ptr(), x.n(), x.c, gamma.fp(), silu, o.b.ptr())?;
        Ok(o)
    }

    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: Vol) -> Result<Vol> {
        let h = self.norm(nsd, &x, &r.norm1, true)?;
        let h = self.conv(nsd, &r.conv1, &h)?;
        let h = self.norm(nsd, &h, &r.norm2, true)?;
        let h = self.conv(nsd, &r.conv2, &h)?;
        let s = match &r.shortcut {
            Some(c) => self.conv(nsd, c, &x)?,
            None => x,
        };
        self.k.add(h.b.ptr(), s.b.ptr(), h.n() * h.c)?;
        Ok(h)
    }

    /// The middle's attention: one head over every voxel, causal by frame
    fn attention(&self, nsd: &Nsd, x: Vol) -> Result<Vol> {
        let gpu = &nsd.gpu;
        let a = &self.att;
        let n = x.n();
        let c = x.c;
        let hw = x.h * x.w;
        let h = self.norm(nsd, &x, &a.norm, false)?;
        let q = self.conv(nsd, &a.q, &h)?;
        let k = self.conv(nsd, &a.k, &h)?;
        let v = self.conv(nsd, &a.v, &h)?;
        drop(h);
        let vt = DevBuf::new(gpu, n * c * 2)?;
        self.k.transpose(v.b.ptr(), n, c, vt.ptr())?;
        drop(v);
        let o = Vol::new(gpu, x.t, x.h, x.w, c)?;
        // query rows a call: whole frames, the scores within SCORE_BYTES
        let frames = (SCORE_BYTES / (n * 4 * hw)).clamp(1, x.t);
        let rows = frames * hw;
        let s = DevBuf::f32(gpu, rows * n)?;
        let p = DevBuf::new(gpu, rows * n * 2)?;
        let of = DevBuf::f32(gpu, rows * c)?;
        let scale = 1.0 / (c as f32).sqrt();
        let mut r0 = 0;
        while r0 < n {
            let m = rows.min(n - r0);
            // the keys this block of queries sees: every frame up to its last
            let nk = ((r0 + m - 1) / hw + 1) * hw;
            let qp = q.b.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            self.k.linear_f32(qp, m, c, k.b.ptr(), 0, nk, std::ptr::null(), s.fp(), nk)?;
            self.k.softmax_causal(s.fp(), m, nk, r0, hw, scale, p.ptr())?;
            self.k.linear_f32(p.ptr(), m, nk, vt.ptr(), n, c, std::ptr::null(), of.fp(), c)?;
            self.k.to_half(of.fp(), o.b.ptr().cast::<u16>().wrapping_add(r0 * c).cast(), m * c, 1.0)?;
            r0 += m;
        }
        let y = self.conv(nsd, &a.proj, &o)?;
        self.k.add(y.b.ptr(), x.b.ptr(), n * c)?;
        Ok(y)
    }

    /// Latents on the card (half [F, h, w, 32], already divided by the scaling) -> frames half [To, 16h, 16w, 3]
    pub fn decode_vol(&self, nsd: &Nsd, z: Vol, each: &mut dyn FnMut(&str)) -> Result<Vol> {
        let mut x = self.conv(nsd, &self.conv_in, &z)?;
        self.k.rep_add(x.b.ptr(), z.b.ptr(), x.n(), x.c, z.c)?;
        drop(z);
        x = self.resnet(nsd, &self.mid[0], x)?;
        x = self.attention(nsd, x)?;
        x = self.resnet(nsd, &self.mid[1], x)?;
        each("mid");
        for (li, l) in self.levels.iter().enumerate() {
            for r in &l.resnets {
                x = self.resnet(nsd, r, x)?;
            }
            if let Some((c, temporal)) = &l.up {
                let conv = self.conv(nsd, c, &x)?;
                let co = c.co / if *temporal { 8 } else { 4 };
                let o = Vol::new(&nsd.gpu, if *temporal { 2 * x.t - 1 } else { x.t }, 2 * x.h, 2 * x.w, co)?;
                self.k.upsample(conv.b.ptr(), x.b.ptr(), x.t, x.h, x.w, co, x.c, *temporal, o.b.ptr())?;
                x = o;
            }
            nsd.wait()?;
            each(["level 0", "level 1", "level 2", "level 3", "level 4"][li.min(4)]);
        }
        let h = self.norm(nsd, &x, &self.norm_out, true)?;
        drop(x);
        let out = self.conv(nsd, &self.conv_out, &h)?;
        nsd.wait()?;
        Ok(out)
    }

    /// Latents float32 [F, h, w, 32] (as the DiT writes them, not yet divided by the scaling) -> frames float32 [To, H,
    /// W, 3] in [-1, 1] (as the decoder gives them), tiled as the pipeline tiles
    pub fn decode(&self, nsd: &Nsd, lat: &[f32], f: usize, h: usize, w: usize, each: &mut dyn FnMut(String)) -> Result<(Vec<f32>, usize, usize, usize)> {
        if lat.len() != f * h * w * Z {
            return Err(Error(format!("{} latent values for {f} x {h} x {w} x {Z}", lat.len())));
        }
        let to = 4 * (f - 1) + 1;
        let one = |i0: usize, j0: usize, th: usize, tw: usize, each: &mut dyn FnMut(String)| -> Result<Vec<f32>> {
            let mut part = Vec::with_capacity(f * th * tw * Z);
            for t in 0..f {
                for y in i0..i0 + th {
                    let at = ((t * h + y) * w + j0) * Z;
                    part.extend_from_slice(&lat[at..at + tw * Z]);
                }
            }
            let gpu = &nsd.gpu;
            let zf = DevBuf::from_f32(gpu, &part)?;
            let z = Vol::new(gpu, f, th, tw, Z)?;
            self.k.to_half(zf.fp(), z.b.ptr(), part.len(), 1.0 / SCALING)?;
            let v = self.decode_vol(nsd, z, &mut |s| each(format!("tile ({i0}, {j0}): {s}")))?;
            let of = DevBuf::f32(gpu, v.n() * 3)?;
            self.k.to_float(v.b.ptr(), of.fp(), v.n() * 3)?;
            nsd.wait()?;
            of.to_f32()
        };
        if h <= TILE && w <= TILE {
            return Ok((one(0, 0, h, w, each)?, to, 16 * h, 16 * w));
        }
        // spatial_tiled_decode: rows of tiles, each blended with the one above and the one left (in place, as the
        // official loop does: a neighbour is read after its own blending), then cut to 224 and joined
        struct Tile {
            v: Vec<f32>,
            h: usize,
            w: usize,
        }
        let mut rows: Vec<Vec<Tile>> = Vec::new();
        for i in (0..h).step_by(TILE_STEP) {
            let mut row = Vec::new();
            for j in (0..w).step_by(TILE_STEP) {
                let (th, tw) = (TILE.min(h - i), TILE.min(w - j));
                row.push(Tile { v: one(i, j, th, tw, each)?, h: 16 * th, w: 16 * tw });
            }
            rows.push(row);
        }
        for i in 0..rows.len() {
            for j in 0..rows[i].len() {
                if i > 0 {
                    let (above, rest) = rows.split_at_mut(i);
                    let a = &above[i - 1][j];
                    let b = &mut rest[0][j];
                    let e = BLEND.min(a.h).min(b.h);
                    for t in 0..to {
                        for y in 0..e {
                            for x in 0..b.w.min(a.w) {
                                let r = y as f32 / e as f32;
                                for c in 0..3 {
                                    let av = a.v[((t * a.h + a.h - e + y) * a.w + x) * 3 + c];
                                    let bv = &mut b.v[((t * b.h + y) * b.w + x) * 3 + c];
                                    *bv = av * (1.0 - r) + *bv * r;
                                }
                            }
                        }
                    }
                }
                if j > 0 {
                    let (left, rest) = rows[i].split_at_mut(j);
                    let a = &left[j - 1];
                    let b = &mut rest[0];
                    let e = BLEND.min(a.w).min(b.w);
                    for t in 0..to {
                        for y in 0..b.h.min(a.h) {
                            for x in 0..e {
                                let r = x as f32 / e as f32;
                                for c in 0..3 {
                                    let av = a.v[((t * a.h + y) * a.w + a.w - e + x) * 3 + c];
                                    let bv = &mut b.v[((t * b.h + y) * b.w + x) * 3 + c];
                                    *bv = av * (1.0 - r) + *bv * r;
                                }
                            }
                        }
                    }
                }
            }
        }
        let (oh, ow) = (rows.iter().map(|r| r[0].h.min(KEEP)).sum::<usize>(), rows[0].iter().map(|t| t.w.min(KEEP)).sum::<usize>());
        let mut out = vec![0f32; to * oh * ow * 3];
        let mut y0 = 0;
        for row in &rows {
            let mut x0 = 0;
            for tile in row {
                let (kh, kw) = (tile.h.min(KEEP), tile.w.min(KEEP));
                for t in 0..to {
                    for y in 0..kh {
                        let src = ((t * tile.h + y) * tile.w) * 3;
                        let dst = ((t * oh + y0 + y) * ow + x0) * 3;
                        out[dst..dst + kw * 3].copy_from_slice(&tile.v[src..src + kw * 3]);
                    }
                }
                x0 += kw;
            }
            y0 += row[0].h.min(KEEP);
        }
        Ok((out, to, oh, ow))
    }
}
