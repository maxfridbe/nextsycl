//! The video decoder: latents [24, T, H, W] -> pixels [3, frames, 16 H, 16 W] in [0, 1].
//!
//! A transformer, not a stack of convolutions: each latent pixel (one 16x16 patch of 4 frames) is a token, 36 blocks
//! of attention + gated MLP run over the tokens of a tile, and a final linear turns every token back into its 4 x 16
//! x 16 pixels. The same kernels as the denoiser (int8 linears, per-head norm + rotation, fused attention).
//!
//! A clip is decoded in pieces, exactly as the reference does it (ComfyUI `ldm/minimax/vae.py`), because the
//! decoder was trained on pieces of that size: chunks of 5 latent frames (+2 frames of overlap, cross-faded) and
//! tiles of 256 x 256 pixels (16 x 16 latent pixels, overlaps of at least 64 pixels, cross-faded).

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{host, rotations, small};
use crate::dtype::DType;
use crate::ops::{self, Int8Linear, Linear, Rows};
use crate::safetensors::Checkpoint;
use crate::{load, Ctx, Error, Result};

const PREFIX: &str = "decoder.";
const HEADS: usize = 32;
const HEAD_DIM: usize = 64;
const DIM: usize = HEADS * HEAD_DIM;
/// rotation on 3/4 of each head's features: 8 frequencies per axis, 3 axes, a pair each
const ROT_DIM: usize = 48;
const REGISTERS: usize = 4;
const EPS: f32 = 1e-5;
/// pixels per latent pixel, frames per latent frame
const PATCH: usize = 16;
const PATCH_T: usize = 4;
const OUT_C: usize = 3;
const LATENT_C: usize = 24;

// temporal chunking (clip_length 17, token_drop 3)
const CLIP_LENGTH: usize = 17;
const TOKEN_DROP: usize = 3;
const CHUNK_TOKENS: usize = CLIP_LENGTH.div_ceil(PATCH_T); // 5
const TOKEN_OVERLAP: usize = (CHUNK_TOKENS - TOKEN_DROP % CHUNK_TOKENS) % CHUNK_TOKENS; // 2
const FRAME_PRE_PAD: usize = (PATCH_T - CLIP_LENGTH % PATCH_T) % PATCH_T; // 3
const FRAME_OVERLAP: usize = TOKEN_OVERLAP * PATCH_T - FRAME_PRE_PAD; // 5
// spatial tiling, in pixels
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
/// tiles decoded together (they share the linears' matrix products)
const TILE_BATCH: usize = 4;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A linear of the decoder: int8 (the int8_convrot checkpoint) or 16-bit (the fp16 one).
enum Lin {
    I8(Int8Linear),
    F(Linear),
}

impl Lin {
    fn forward(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        match self {
            Lin::I8(l) => l.forward(x, out),
            Lin::F(l) => l.forward(x, out),
        }
    }
}

struct VBlock {
    norm1: Tensor,
    norm2: Tensor,
    /// [1, DIM] float32: the layer scales of the two residual adds
    scale1: Tensor,
    scale2: Tensor,
    /// q, k, v rows regrouped at load: the checkpoint interleaves them per head ([q k v] of head 0, of head 1 ...)
    qkv: Lin,
    out: Lin,
    w1: Lin,
    w2: Lin,
    /// the layer scales are folded into `out` and `w2` (16-bit weights): they accumulate into the stream
    folded: bool,
}

pub struct VideoDecoder {
    dev: Arc<Device>,
    /// the activations' type: bfloat16 for the int8 checkpoint, half for the fp16 one
    act: DType,
    blocks: Vec<VBlock>,
    x_embed: Linear,
    /// [REGISTERS + 1, DIM] in `act`: the register tokens and a zero token, appended after the image tokens
    suffix: Tensor,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_out: Linear,
    ones: Tensor,
    inv_freq: Vec<f32>,
    /// the 1x1x1 convolution before the decoder [24, 24] and its bias; the latent normalization
    pq_w: Vec<f32>,
    pq_b: Vec<f32>,
    lat_mean: Vec<f32>,
    lat_std: Vec<f32>,
    pub load_seconds: f64,
    pub load_bytes: u64,
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

fn act_bytes(v: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::BF16 => v.iter().flat_map(|f| crate::dtype::f32_to_bf16(*f).to_le_bytes()).collect(),
        DType::F16 => v.iter().flat_map(|f| crate::dtype::f32_to_f16(*f).to_le_bytes()).collect(),
        _ => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
    }
}

impl VideoDecoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, threads: usize) -> Result<VideoDecoder> {
        const LIN: [&str; 4] = ["attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"];
        let n = ck.entries.keys().filter_map(|k| k.strip_prefix("decoder.transformer_blocks.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        if n == 0 {
            return Err(Error(format!("{}: no video decoder blocks", ck.path.display())));
        }
        let int8 = ck.quant(&format!("{PREFIX}transformer_blocks.0.attn.to_qkv"))?.is_some();
        let act = if int8 { DType::BF16 } else { DType::F16 };
        let wanted = |name: &str| {
            name.strip_prefix("decoder.transformer_blocks.").and_then(|r| r.split_once('.')).is_some_and(|(_, t)| {
                LIN.iter().any(|l| t == format!("{l}.weight") || t == format!("{l}.weight_scale"))
            })
        };
        let loaded = load::load(dev, ck, threads, wanted).ctx("loading the video decoder")?;
        let mut big = loaded.tensors;
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("{PREFIX}transformer_blocks.{i}");
            let mut lin = |l: &str, regroup: bool| -> Result<Lin> {
                let name = format!("{p}.{l}");
                let mut w = big.remove(&format!("{name}.weight")).ok_or_else(|| Error(format!("{name}.weight was not loaded")))?;
                let mut bias = host(ck, &format!("{name}.bias"))?;
                let mut scale = match int8 {
                    true => Some(big.remove(&format!("{name}.weight_scale")).ok_or_else(|| Error(format!("{name}.weight_scale was not loaded")))?),
                    false => None,
                };
                if regroup {
                    // [h0: q k v][h1: q k v] ... -> [q of every head][k ...][v ...]
                    let order: Vec<usize> = (0..3).flat_map(|part| (0..HEADS).map(move |h| h * 3 + part)).collect();
                    let regroup_rows = |t: &Tensor| -> Result<Tensor> {
                        let out = Tensor::new(dev, t.dtype, &t.shape)?;
                        for (dst, src) in order.iter().enumerate() {
                            out.copy_rows(dst * HEAD_DIM, t, src * HEAD_DIM, HEAD_DIM)?;
                        }
                        Ok(out)
                    };
                    w = regroup_rows(&w)?;
                    if let Some(s) = scale.take() {
                        scale = Some(regroup_rows(&s)?);
                    }
                    bias = order.iter().flat_map(|src| bias[src * HEAD_DIM..(src + 1) * HEAD_DIM].to_vec()).collect();
                }
                let bias = Some(f32_tensor(dev, &[bias.len()], &bias)?);
                Ok(match scale {
                    Some(scale) => {
                        let q = ck.quant(&name)?.ok_or_else(|| Error(format!("{name} has no quantization record")))?;
                        Lin::I8(Int8Linear { weight: w, scale, bias, group: q.convrot.then_some(q.group) })
                    }
                    None => Lin::F(Linear { weight: w, bias }),
                })
            };
            let qkv = lin(LIN[0], true)?;
            let mut out = lin(LIN[1], false)?;
            let w1 = lin(LIN[2], false)?;
            let mut w2 = lin(LIN[3], false)?;
            // 16-bit weights: the layer scales folded into the two linears that feed the stream, so their products
            // add themselves into it (no separate scaled add)
            let folded = match (&mut out, &mut w2) {
                (Lin::F(o), Lin::F(w)) => {
                    o.scale_outputs(&host(ck, &format!("{p}.scale1"))?)?;
                    w.scale_outputs(&host(ck, &format!("{p}.scale2"))?)?;
                    true
                }
                _ => false,
            };
            let s = |k: &str| small(dev, ck, &format!("{p}.{k}"));
            let row = |k: &str| -> Result<Tensor> {
                let v = host(ck, &format!("{p}.{k}"))?;
                f32_tensor(dev, &[1, v.len()], &v)
            };
            blocks.push(VBlock { norm1: s("norm1.weight")?, norm2: s("norm2.weight")?, scale1: row("scale1")?, scale2: row("scale2")?, qkv, out, w1, w2, folded });
        }
        dev.wait()?;
        let lin32 = |name: &str| -> Result<Linear> {
            let w = host(ck, &format!("{PREFIX}{name}.weight"))?;
            let e = ck.get(&format!("{PREFIX}{name}.weight"))?;
            Ok(Linear { weight: f32_tensor(dev, &e.shape, &w)?, bias: Some(small(dev, ck, &format!("{PREFIX}{name}.bias"))?) })
        };
        let mut suffix = host(ck, &format!("{PREFIX}register_tokens"))?;
        suffix.extend(std::iter::repeat_n(0.0, DIM));
        let pq = host(ck, "post_quant_conv.weight")?;
        let inv_freq: Vec<f32> = (0..ROT_DIM / 6).map(|i| 1.0 / 100f32.powf(i as f32 * 6.0 / ROT_DIM as f32)).collect();
        Ok(VideoDecoder {
            dev: dev.clone(),
            act,
            blocks,
            x_embed: lin32("x_embedder")?,
            suffix: Tensor::from_bytes(dev, act, &[REGISTERS + 1, DIM], &act_bytes(&suffix, act))?,
            norm_out_w: small(dev, ck, &format!("{PREFIX}norm_out.weight"))?,
            norm_out_b: small(dev, ck, &format!("{PREFIX}norm_out.bias"))?,
            proj_out: lin32("proj_out")?,
            ones: f32_tensor(dev, &[HEAD_DIM], &[1.0; HEAD_DIM])?,
            inv_freq,
            pq_w: pq,
            pq_b: host(ck, "post_quant_conv.bias")?,
            lat_mean: host(ck, "latents_mean")?,
            lat_std: host(ck, "latents_std")?,
            load_seconds: loaded.seconds,
            load_bytes: loaded.bytes,
        })
    }

    /// Tiles of the same size, decoded together: latents [24, t, h, w] each (denormalized, after the 1x1x1
    /// convolution) -> raw pixels [3, 4t, 16h, 16w] each. The tiles share the linears (one matrix product over all
    /// their tokens: a tile alone, ~1,800 tokens, is a thin product for this card) and keep their own attention.
    fn tiles(&self, zs: &[Vec<f32>], t: usize, h: usize, w: usize) -> Result<Vec<Vec<f32>>> {
        let dev = &self.dev;
        let nb = zs.len();
        let n = t * h * w;
        let s = n + REGISTERS + 1;
        let rows_all = nb * s;
        // tokens: (t, h, w) order, 24 features, tile after tile
        let mut rows = vec![0f32; nb * n * LATENT_C];
        for (k, z) in zs.iter().enumerate() {
            for c in 0..LATENT_C {
                for i in 0..n {
                    rows[(k * n + i) * LATENT_C + c] = z[c * n + i];
                }
            }
        }
        let input = f32_tensor(dev, &[nb * n, LATENT_C], &rows)?;
        let x = Tensor::new(dev, self.act, &[rows_all, DIM])?;
        {
            let e = Tensor::new(dev, self.act, &[nb * n, DIM])?;
            self.x_embed.forward(&input, &e)?;
            for k in 0..nb {
                x.copy_rows(k * s, &e, k * n, n)?;
                x.copy_rows(k * s + n, &self.suffix, 0, REGISTERS + 1)?;
            }
        }
        // positions: cell centres in [-1, 1] per axis, the suffix at 0; angles carry the 2 pi; the same per tile
        let mut pos = Vec::with_capacity(s * 3);
        let c = |i: usize, d: usize| (2.0 * ((i as f64 + 0.5) / d as f64) - 1.0) * std::f64::consts::TAU;
        for ti in 0..t {
            for yi in 0..h {
                for xi in 0..w {
                    pos.extend_from_slice(&[c(ti, t), c(yi, h), c(xi, w)]);
                }
            }
        }
        pos.extend(std::iter::repeat_n(0.0, (REGISTERS + 1) * 3));
        let one = rotations(&pos, &self.inv_freq);
        let all: Vec<f32> = (0..nb).flat_map(|_| one.iter().copied()).collect();
        let cs = f32_tensor(dev, &[rows_all, ROT_DIM / 2, 2], &all)?;
        let zero_rows = Tensor::from_bytes(dev, DType::I32, &[rows_all], &vec![0u8; rows_all * 4])?;

        let hbuf = Tensor::new(dev, self.act, &[rows_all, DIM])?;
        let qkv = Tensor::new(dev, self.act, &[rows_all, 3 * DIM])?;
        let att = Tensor::new(dev, self.act, &[rows_all, DIM])?;
        let att1 = Tensor::new(dev, self.act, &[s, DIM])?;
        let proj = Tensor::new(dev, self.act, &[rows_all, DIM])?;
        let f1 = Tensor::new(dev, self.act, &[rows_all, 2 * 4 * DIM])?;
        let act = Tensor::new(dev, self.act, &[rows_all, 4 * DIM])?;
        let prof = std::env::var_os("NSD_PROFILE").is_some();
        let per_tile = std::env::var("H3_VAE_ATTN_PER_TILE").is_ok_and(|v| v == "1");
        let tb = std::time::Instant::now();
        if prof {
            dev.wait()?;
        }
        // H3S_PROFILE: a wait after every stage, the time summed per stage (slower; for finding where it goes)
        let mut stages: Vec<(&str, f64)> = Vec::new();
        let mut mark = std::time::Instant::now();
        let mut lap = |name: &'static str| -> Result<()> {
            if prof {
                dev.wait()?;
                let dt = mark.elapsed().as_secs_f64();
                match stages.iter_mut().find(|(n, _)| *n == name) {
                    Some(st) => st.1 += dt,
                    None => stages.push((name, dt)),
                }
                mark = std::time::Instant::now();
            }
            Ok(())
        };
        for b in &self.blocks {
            ops::rms_norm_mod(&x, &b.norm1, EPS, None, &hbuf)?;
            lap("norm")?;
            b.qkv.forward(&hbuf, &qkv)?;
            lap("linear qkv")?;
            let all_rows = |i: usize| Rows { t: &qkv, offset: i * DIM, stride: 3 * DIM, tokens: rows_all, heads: HEADS, dim: HEAD_DIM };
            ops::rms_rope(all_rows(0), &self.ones, EPS, &cs, ROT_DIM)?;
            ops::rms_rope(all_rows(1), &self.ones, EPS, &cs, ROT_DIM)?;
            lap("norm + rotation q k")?;
            // every tile's attention in one call (each tile attends to its own tokens only); H3_VAE_ATTN_PER_TILE=1: one
            // call per tile, as before
            if per_tile {
                for k in 0..nb {
                    let part = |i: usize| Rows { t: &qkv, offset: k * s * 3 * DIM + i * DIM, stride: 3 * DIM, tokens: s, heads: HEADS, dim: HEAD_DIM };
                    if nb == 1 {
                        ops::attention(part(0), part(1), part(2), &att)?;
                    } else {
                        ops::attention(part(0), part(1), part(2), &att1)?;
                        att.copy_rows(k * s, &att1, 0, s)?;
                    }
                }
            } else {
                let part = |i: usize| Rows { t: &qkv, offset: i * DIM, stride: 3 * DIM, tokens: nb * s, heads: HEADS, dim: HEAD_DIM };
                ops::attention_batch(part(0), part(1), part(2), nb, &att)?;
            }
            lap("attention")?;
            match (&b.out, b.folded) {
                (Lin::F(l), true) => l.forward_acc(&att, &x)?,
                _ => {
                    b.out.forward(&att, &proj)?;
                    ops::gate_add(&x, &proj, &zero_rows, &b.scale1)?;
                }
            }
            lap("linear out + add")?;
            ops::rms_norm_mod(&x, &b.norm2, EPS, None, &hbuf)?;
            lap("norm")?;
            b.w1.forward(&hbuf, &f1)?;
            lap("linear w1")?;
            ops::swiglu(&f1, &act)?;
            lap("gated activation")?;
            match (&b.w2, b.folded) {
                (Lin::F(l), true) => l.forward_acc(&act, &x)?,
                _ => {
                    b.w2.forward(&act, &proj)?;
                    ops::gate_add(&x, &proj, &zero_rows, &b.scale2)?;
                }
            }
            lap("linear w2 + add")?;
        }
        if prof {
            let sum: f64 = stages.iter().map(|s| s.1).sum();
            for (n, t) in &stages {
                eprintln!("vae stage: {n:22} {:7.1} ms  {:4.1}%", t * 1e3, t / sum * 100.0);
            }
        }
        if prof {
            dev.wait()?;
            eprintln!("vae: {nb} tiles of {s} tokens: blocks {:.3} s", tb.elapsed().as_secs_f64());
        }
        let normed = Tensor::new(dev, DType::F32, &[nb * n, DIM])?;
        {
            let img = Tensor::new(dev, self.act, &[nb * n, DIM])?;
            for k in 0..nb {
                img.copy_rows(k * n, &x, k * s, n)?;
            }
            ops::layer_norm(&img, DIM, Some(&self.norm_out_w), Some(&self.norm_out_b), EPS, &normed)?;
        }
        let per = OUT_C * PATCH_T * PATCH * PATCH;
        let out = Tensor::new(dev, DType::F32, &[nb * n, per])?;
        self.proj_out.forward(&normed, &out)?;
        let o = out.to_f32()?;
        // token (t, h, w) -> pixels (c, t*4 + a, h*16 + y, w*16 + x)
        let (ft, fh, fw) = (t * PATCH_T, h * PATCH, w * PATCH);
        let mut res = Vec::with_capacity(nb);
        for k in 0..nb {
            let o = &o[k * n * per..(k + 1) * n * per];
            let mut px = vec![0f32; OUT_C * ft * fh * fw];
            for ti in 0..t {
                for yi in 0..h {
                    for xi in 0..w {
                        let tok = &o[((ti * h + yi) * w + xi) * per..][..per];
                        for ch in 0..OUT_C {
                            for a in 0..PATCH_T {
                                for y in 0..PATCH {
                                    let dst = ((ch * ft + ti * PATCH_T + a) * fh + yi * PATCH + y) * fw + xi * PATCH;
                                    let src = ((ch * PATCH_T + a) * PATCH + y) * PATCH;
                                    px[dst..dst + PATCH].copy_from_slice(&tok[src..src + PATCH]);
                                }
                            }
                        }
                    }
                }
            }
            res.push(px);
        }
        Ok(res)
    }

    /// Latents [24, t, h, w] (raw, denormalized) through tiling: pixels [3, 4t, 16h, 16w], raw.
    fn tiled(&self, z: &[f32], t: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<Vec<f32>> {
        let (fh, fw, ft) = (h * PATCH, w * PATCH, t * PATCH_T);
        let (ys, yo) = split_tiles(fh);
        let (xs, xo) = split_tiles(fw);
        let tile_px = |len: usize| len.min(TILE);
        let (th, tw) = (tile_px(fh), tile_px(fw));
        let (lh, lw) = (th / PATCH, tw / PATCH);
        // every tile's latents, then the tiles decoded a few at a time
        let mut zs = Vec::with_capacity(ys.len() * xs.len());
        for &y0 in &ys {
            for &x0 in &xs {
                let (ly, lx) = (y0 / PATCH, x0 / PATCH);
                let mut zt = vec![0f32; LATENT_C * t * lh * lw];
                for c in 0..LATENT_C {
                    for ti in 0..t {
                        for yy in 0..lh {
                            let src = ((c * t + ti) * h + ly + yy) * w + lx;
                            let dst = ((c * t + ti) * lh + yy) * lw;
                            zt[dst..dst + lw].copy_from_slice(&z[src..src + lw]);
                        }
                    }
                }
                zs.push(zt);
            }
        }
        let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(zs.len());
        let t0 = std::time::Instant::now();
        for batch in zs.chunks(TILE_BATCH) {
            tick()?;
            decoded.extend(self.tiles(batch, t, lh, lw)?);
        }
        let t_tiles = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let mut decoded = decoded.into_iter();
        let mut canvas = vec![0f32; OUT_C * ft * fh * fw];
        // per row of tiles: the bottom overlap strips of the previous row, per tile column
        let mut row_tails: Vec<Img> = Vec::new();
        let mut out_y = 0;
        for i in 0..ys.len() {
            let mut new_tails = Vec::new();
            let mut left_tail: Option<Img> = None;
            let mut out_x = 0;
            let mut tile_h = 0;
            for j in 0..xs.len() {
                let mut tile = Img { c: OUT_C * ft, h: th, w: tw, v: decoded.next().ok_or("a tile went missing")? };
                if i + 1 < ys.len() {
                    new_tails.push(tile.rows(th - yo[i], th));
                }
                let next_left = (j + 1 < xs.len()).then(|| tile.cols(tw - xo[j], tw));
                if i > 0 {
                    tile = blend(&row_tails[j], &tile, yo[i - 1], Axis::H);
                }
                if j > 0 {
                    tile = blend(left_tail.as_ref().expect("left tail"), &tile, xo[j - 1], Axis::W);
                }
                left_tail = next_left;
                if i + 1 < ys.len() {
                    tile = tile.rows(0, tile.h - yo[i]);
                }
                if j + 1 < xs.len() {
                    tile = tile.cols(0, tile.w - xo[j]);
                }
                for p in 0..tile.c {
                    for yy in 0..tile.h {
                        let dst = (p * fh + out_y + yy) * fw + out_x;
                        canvas[dst..dst + tile.w].copy_from_slice(&tile.v[(p * tile.h + yy) * tile.w..][..tile.w]);
                    }
                }
                tile_h = tile.h;
                out_x += tile.w;
            }
            row_tails = new_tails;
            out_y += tile_h;
        }
        if std::env::var_os("NSD_PROFILE").is_some() {
            eprintln!("vae: {} tiles of {t}x{lh}x{lw}: {t_tiles:.2} s in the tiles, {:.2} s cross-fading", zs.len(), t0.elapsed().as_secs_f64());
        }
        Ok(canvas)
    }

    /// Normalized latents [24, T, H, W] (what the sampler produces) -> pixels [3, frames, 16 H, 16 W] in [0, 1].
    /// `tick` is called before every batch of tiles (cancel checks, progress).
    pub fn decode(&self, z: &[f32], t: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let n = h * w;
        // denormalize, then the 1x1x1 convolution
        let mut zr = vec![0f32; z.len()];
        for c in 0..LATENT_C {
            for i in 0..t * n {
                zr[c * t * n + i] = z[c * t * n + i] * self.lat_std[c] + self.lat_mean[c];
            }
        }
        let mut zp = vec![0f32; z.len()];
        for o in 0..LATENT_C {
            for i in 0..t * n {
                let mut acc = self.pq_b[o];
                for c in 0..LATENT_C {
                    acc += self.pq_w[o * LATENT_C + c] * zr[c * t * n + i];
                }
                zp[o * t * n + i] = acc;
            }
        }
        let (fh, fw) = (h * PATCH, w * PATCH);
        let plane = fh * fw;
        if t == 1 {
            let raw = self.tiled(&zp, 1, h, w, tick)?;
            // the last of the 4 decoded frames
            let mut out = vec![0f32; OUT_C * plane];
            for c in 0..OUT_C {
                out[c * plane..][..plane].copy_from_slice(&raw[(c * PATCH_T + PATCH_T - 1) * plane..][..plane]);
            }
            return Ok((finalize(out, 1, plane), 1));
        }

        // temporal chunks: pad the latent frames, decode 5 + 2 at a time, cross-fade 5 frames
        let total = t + TOKEN_DROP;
        let mut pad = (CHUNK_TOKENS - total % CHUNK_TOKENS) % CHUNK_TOKENS;
        let mut chunks = (total + pad) / CHUNK_TOKENS - 1;
        if chunks < 1 {
            pad += CHUNK_TOKENS;
            chunks += 1;
        }
        let tp = t + pad;
        let frames = frame_count(tp, chunks, pad);
        // padding repeats the last latent frame
        let lat_frame = |ti: usize, c: usize| -> std::ops::Range<usize> {
            let i = ti.min(t - 1);
            (c * t + i) * n..(c * t + i + 1) * n
        };
        let mut out: Vec<Vec<f32>> = Vec::new(); // frames, each [3, plane]
        let mut overlap: Option<Vec<Vec<f32>>> = None;
        let chunk_dec = CHUNK_TOKENS * PATCH_T;
        for ci in 0..chunks {
            let t0 = ci * CHUNK_TOKENS;
            let t1 = (t0 + CHUNK_TOKENS + TOKEN_OVERLAP).min(tp);
            let ct = t1 - t0;
            let mut zc = vec![0f32; LATENT_C * ct * n];
            for c in 0..LATENT_C {
                for k in 0..ct {
                    zc[(c * ct + k) * n..][..n].copy_from_slice(&zp[lat_frame(t0 + k, c)]);
                }
            }
            let raw = self.tiled(&zc, ct, h, w, tick)?;
            let nf = ct * PATCH_T;
            let frame = |f: usize| -> Vec<f32> { (0..OUT_C).flat_map(|c| raw[(c * nf + f) * plane..][..plane].to_vec()).collect() };
            for j in 0..2 {
                let f0 = j * chunk_dec;
                let f1 = (f0 + chunk_dec).min(nf);
                let part: Vec<Vec<f32>> = (f0 + FRAME_PRE_PAD..f1.max(f0 + FRAME_PRE_PAD)).map(frame).collect();
                if j == 0 {
                    let mut part = part;
                    if let Some(prev) = overlap.take() {
                        let k = prev.len().min(part.len()).min(FRAME_OVERLAP);
                        for (i, fr) in part.iter_mut().enumerate().take(k) {
                            let wb = i as f32 / k as f32;
                            let pa = &prev[prev.len() - k + i];
                            for (v, a) in fr.iter_mut().zip(pa) {
                                *v = a * (1.0 - wb) + *v * wb;
                            }
                        }
                    }
                    out.extend(part);
                } else {
                    overlap = Some(part);
                }
            }
            if ci + 1 == chunks {
                if let Some(prev) = overlap.take() {
                    out.extend(prev);
                }
            }
        }
        out.truncate(frames);
        let f = out.len();
        let mut px = vec![0f32; OUT_C * f * plane];
        for (fi, fr) in out.iter().enumerate() {
            for c in 0..OUT_C {
                px[(c * f + fi) * plane..][..plane].copy_from_slice(&fr[c * plane..][..plane]);
            }
        }
        Ok((finalize(px, f, plane), f))
    }
}

/// raw decoder output -> [0, 1]: undo the ImageNet normalization, clamp
fn finalize(mut px: Vec<f32>, frames: usize, plane: usize) -> Vec<f32> {
    for c in 0..OUT_C {
        for v in &mut px[c * frames * plane..(c + 1) * frames * plane] {
            *v = (*v * IMAGENET_STD[c] + IMAGENET_MEAN[c]).clamp(0.0, 1.0);
        }
    }
    px
}

/// The frames a clip of `tp` latent frames (padding included) decodes to, as the reference counts them.
fn frame_count(tp: usize, chunks: usize, pad: usize) -> usize {
    let chunk_dec = CHUNK_TOKENS * PATCH_T;
    let (mut total, mut last_overlap) = (0, 0);
    for i in 0..chunks {
        let t0 = i * CHUNK_TOKENS;
        let t1 = t0 + CHUNK_TOKENS + TOKEN_OVERLAP;
        let len = (t1.min(tp) - t0.min(tp)) * PATCH_T;
        for j in 0..2 {
            let f0 = j * chunk_dec;
            let f1 = (f0 + chunk_dec).min(len);
            let n = f1.saturating_sub(f0).saturating_sub(FRAME_PRE_PAD);
            if j == 0 {
                total += n;
            } else {
                last_overlap = n;
            }
        }
    }
    total += last_overlap;
    // frames that came from the padding
    let tail = CLIP_LENGTH % PATCH_T;
    let pad_frames = if pad == 0 {
        0
    } else if tail == 0 {
        pad * PATCH_T
    } else {
        (0..pad).map(|k| if (tp - pad + k).is_multiple_of(CHUNK_TOKENS) { tail } else { PATCH_T }).sum()
    };
    total - pad_frames
}

/// Tile starts and overlaps along one axis of `len` pixels.
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
    for i in 0..remaining / PATCH {
        let k = i % (n - 1);
        overlaps[k] += PATCH;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        starts.push(starts.last().unwrap() + TILE - o);
    }
    (starts, overlaps)
}

/// Planes (channels x frames) of h x w pixels.
#[derive(Clone)]
struct Img {
    c: usize,
    h: usize,
    w: usize,
    v: Vec<f32>,
}

impl Img {
    fn rows(&self, a: usize, b: usize) -> Img {
        let mut v = Vec::with_capacity(self.c * (b - a) * self.w);
        for p in 0..self.c {
            v.extend_from_slice(&self.v[(p * self.h + a) * self.w..(p * self.h + b) * self.w]);
        }
        Img { c: self.c, h: b - a, w: self.w, v }
    }
    fn cols(&self, a: usize, b: usize) -> Img {
        let mut v = Vec::with_capacity(self.c * self.h * (b - a));
        for r in 0..self.c * self.h {
            v.extend_from_slice(&self.v[r * self.w + a..r * self.w + b]);
        }
        Img { c: self.c, h: self.h, w: b - a, v }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    H,
    W,
}

/// The reference's cross-fade: the last `extent` rows (columns) of `a` into the first of `b`, linearly.
fn blend(a: &Img, b: &Img, extent: usize, axis: Axis) -> Img {
    let (alen, blen) = match axis {
        Axis::H => (a.h, b.h),
        Axis::W => (a.w, b.w),
    };
    let k = extent.min(alen).min(blen);
    let mut out = b.clone();
    for p in 0..b.c {
        for y in 0..b.h {
            for x in 0..b.w {
                let i = match axis {
                    Axis::H => y,
                    Axis::W => x,
                };
                if i >= k {
                    continue;
                }
                let wb = i as f32 / k as f32;
                let (ay, ax) = match axis {
                    Axis::H => (a.h - k + y, x),
                    Axis::W => (y, a.w - k + x),
                };
                let av = a.v[(p * a.h + ay) * a.w + ax];
                let o = &mut out.v[(p * b.h + y) * b.w + x];
                *o = av * (1.0 - wb) + *o * wb;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_follow_the_reference() {
        // what the reference's split_tiles gives: the spare pixels go into the overlaps, 16 at a time
        assert_eq!(split_tiles(256), (vec![0], vec![]));
        assert_eq!(split_tiles(384), (vec![0, 128], vec![128]));
        assert_eq!(split_tiles(288), (vec![0, 32], vec![224]));
        // 1280 px: 5 or 6 tiles leave a gap at 64 px overlaps, so 7, with 128 spare pixels spread over the 6 overlaps
        assert_eq!(split_tiles(1280), (vec![0, 160, 320, 496, 672, 848, 1024], vec![96, 96, 80, 80, 80, 80]));
    }

    #[test]
    fn frame_counts() {
        // 17 latent frames: 56 pixel frames in the reference run (2 s at 24 fps)
        let t = 17;
        let total = t + TOKEN_DROP;
        let pad = (CHUNK_TOKENS - total % CHUNK_TOKENS) % CHUNK_TOKENS;
        let chunks = (total + pad) / CHUNK_TOKENS - 1;
        assert_eq!(frame_count(t + pad, chunks, pad), 56);
        assert_eq!(FRAME_OVERLAP, 5);
    }
}
