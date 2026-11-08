//! The VAE: H3's encoder (crates/h3 `venc`, the conditioning of an edit) and P-H3-Image's own decoder - a transformer
//! over one latent frame (diffusers' PH3ImageViTDecoder):
//!
//! ```text
//!   z = latents * std + mean, then the 1x1 post_quant_conv                       [24, h, w]
//!   tiles of 256 x 256 pixels (16 x 16 latent pixels), overlaps of at least 64 cross-faded (the reference's tiling)
//!   a tile: each latent pixel a token (x_embedder 24 -> 2,048), 4 register tokens and a zero token after them
//!   36 blocks: x += scale1 * attn(rms_norm(x));  x += scale2 * w2(swiglu(w1(rms_norm(x))))    (x float32)
//!     32 heads of 64, q and k RMS-normed, 3-axis rotation on 48 of the 64 (the frame's time fixed at 2 (0.5 / 7) - 1,
//!     height and width the cells' centres in [-1, 1), the suffix tokens at 0; angles 2 pi x the frequencies)
//!   layer norm, proj_out 2,048 -> 3 x 16 x 16 (float32): each token's pixel block; ImageNet's normalization undone
//! ```
//!
//! H3's video decoder is the same network over four frames a token; this one is its own copy for one frame (its time
//! coordinate, a 768-wide output) with the residual stream in float32 as the reference keeps it.

use std::sync::Arc;

use nextsycl_h3::device::{Device, Tensor};
use nextsycl_h3::dit::{host, rotations, small};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::ops::{self, Int8Linear, Linear, Rows};
use nextsycl_h3::safetensors::Checkpoint;
use nextsycl_h3::{load, Ctx, Error, Result};

const PREFIX: &str = "decoder.";
const HEADS: usize = 32;
const HEAD_DIM: usize = 64;
const DIM: usize = HEADS * HEAD_DIM;
/// rotation on 3/4 of each head's features: 8 frequencies per axis, 3 axes, a pair each
const ROT_DIM: usize = 48;
const REGISTERS: usize = 4;
const EPS: f32 = 1e-5;
/// pixels per latent pixel
pub const PATCH: usize = 16;
const OUT_C: usize = 3;
pub const LATENT_C: usize = 24;
/// the frame's time coordinate (diffusers' FRAME_POSITION)
const FRAME_POSITION: f64 = 2.0 * (0.5 / 7.0) - 1.0;
// spatial tiling, in pixels (as H3's decoder)
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A linear of the decoder: 16-bit (the bf16 file) or int8 (the int8_convrot one)
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
}

pub struct Decoder {
    dev: Arc<Device>,
    /// the activations' type going into the products: the weights' (bfloat16), or bfloat16 for int8 weights
    act: DType,
    blocks: Vec<VBlock>,
    x_embed: Linear,
    /// [REGISTERS + 1, DIM] float32: the register tokens and a zero token, appended after the image tokens
    suffix: Tensor,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_out: Linear,
    ones: Tensor,
    inv_freq: Vec<f32>,
    /// the 1x1 convolution before the decoder [24, 24] and its bias; the latent normalization
    pq_w: Vec<f32>,
    pq_b: Vec<f32>,
    pub lat_mean: Vec<f32>,
    pub lat_std: Vec<f32>,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// tiles decoded together (they share the linears' products)
    pub tile_batch: usize,
    /// NS_PH3_PROFILE: seconds by stage
    pub stages: std::sync::Mutex<Vec<(&'static str, f64)>>,
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

/// Tile starts and overlaps along one axis of `len` pixels (the reference's `_split_tiles`: the fewest tiles that cover
/// it with overlaps of at least 64, the spare pixels added to the overlaps 16 at a time)
pub fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>) {
    if TILE >= len {
        return (vec![0], vec![]);
    }
    let mut n = len.div_ceil(TILE);
    while TILE * n < TILE_OVERLAP_MIN * (n - 1) + len {
        n += 1;
    }
    let mut overlaps = vec![TILE_OVERLAP_MIN; n - 1];
    let remaining = TILE * n - TILE_OVERLAP_MIN * (n - 1) - len;
    for i in 0..remaining / PATCH {
        overlaps[i % (n - 1)] += PATCH;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        starts.push(starts.last().unwrap() + TILE - o);
    }
    (starts, overlaps)
}

impl Decoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, threads: usize) -> Result<Decoder> {
        const LIN: [&str; 4] = ["attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2"];
        let n = ck.entries.keys().filter_map(|k| k.strip_prefix("decoder.transformer_blocks.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        if n == 0 {
            return Err(Error(format!("{}: no decoder blocks", ck.path.display())));
        }
        let out_w = ck.get(&format!("{PREFIX}proj_out.weight"))?;
        if out_w.shape[0] != OUT_C * PATCH * PATCH {
            return Err(Error(format!("{}: the decoder writes {} values a token, not a single frame's {} (an H3 video VAE?)", ck.path.display(), out_w.shape[0],
                                     OUT_C * PATCH * PATCH)));
        }
        let int8 = ck.quant(&format!("{PREFIX}transformer_blocks.0.attn.to_qkv"))?.is_some();
        let act = if int8 { DType::BF16 } else { ck.get(&format!("{PREFIX}transformer_blocks.0.attn.to_qkv.weight"))?.dtype };
        if act != DType::BF16 && act != DType::F16 {
            return Err(Error(format!("{}: the decoder's matrices are {act:?}; bfloat16, half or int8 expected", ck.path.display())));
        }
        let wanted = |name: &str| {
            name.strip_prefix("decoder.transformer_blocks.").and_then(|r| r.split_once('.')).is_some_and(|(_, t)| {
                LIN.iter().any(|l| t == format!("{l}.weight") || t == format!("{l}.weight_scale"))
            })
        };
        let loaded = load::load(dev, ck, threads, wanted).ctx("loading the decoder")?;
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
            let out = lin(LIN[1], false)?;
            let w1 = lin(LIN[2], false)?;
            let w2 = lin(LIN[3], false)?;
            let s = |k: &str| small(dev, ck, &format!("{p}.{k}"));
            let row = |k: &str| -> Result<Tensor> {
                let v = host(ck, &format!("{p}.{k}"))?;
                f32_tensor(dev, &[1, v.len()], &v)
            };
            blocks.push(VBlock { norm1: s("norm1.weight")?, norm2: s("norm2.weight")?, scale1: row("scale1")?, scale2: row("scale2")?, qkv, out, w1, w2 });
        }
        dev.wait()?;
        let lin32 = |name: &str| -> Result<Linear> {
            let w = host(ck, &format!("{PREFIX}{name}.weight"))?;
            let e = ck.get(&format!("{PREFIX}{name}.weight"))?;
            Ok(Linear { weight: f32_tensor(dev, &e.shape, &w)?, bias: Some(small(dev, ck, &format!("{PREFIX}{name}.bias"))?) })
        };
        let mut suffix = host(ck, &format!("{PREFIX}register_tokens"))?;
        suffix.extend(std::iter::repeat_n(0.0, DIM));
        let inv_freq: Vec<f32> = (0..ROT_DIM / 6).map(|i| 1.0 / 100f32.powf(i as f32 * 6.0 / ROT_DIM as f32)).collect();
        Ok(Decoder {
            dev: dev.clone(),
            act,
            blocks,
            x_embed: lin32("x_embedder")?,
            suffix: f32_tensor(dev, &[REGISTERS + 1, DIM], &suffix)?,
            norm_out_w: small(dev, ck, &format!("{PREFIX}norm_out.weight"))?,
            norm_out_b: small(dev, ck, &format!("{PREFIX}norm_out.bias"))?,
            proj_out: lin32("proj_out")?,
            ones: f32_tensor(dev, &[HEAD_DIM], &[1.0; HEAD_DIM])?,
            inv_freq,
            pq_w: host(ck, "post_quant_conv.weight")?,
            pq_b: host(ck, "post_quant_conv.bias")?,
            lat_mean: host(ck, "latents_mean")?,
            lat_std: host(ck, "latents_std")?,
            load_seconds: loaded.seconds,
            load_bytes: loaded.bytes,
            tile_batch: 16,
            stages: Default::default(),
        })
    }

    /// Tiles of the same size, decoded together: latents [24, h, w] each (denormalized, after the 1x1 convolution) ->
    /// raw pixels [3, 16h, 16w] each (ImageNet-normalized). The tiles share the linears (one product over all their
    /// tokens) and keep their own attention.
    pub fn tiles(&self, zs: &[Vec<f32>], h: usize, w: usize) -> Result<Vec<Vec<f32>>> {
        let dev = &self.dev;
        let act = self.act;
        let nb = zs.len();
        let n = h * w;
        let s = n + REGISTERS + 1;
        let rows_all = nb * s;
        // tokens: (h, w) order, 24 features, tile after tile
        let mut rows = vec![0f32; nb * n * LATENT_C];
        for (k, z) in zs.iter().enumerate() {
            for c in 0..LATENT_C {
                for i in 0..n {
                    rows[(k * n + i) * LATENT_C + c] = z[c * n + i];
                }
            }
        }
        let input = f32_tensor(dev, &[nb * n, LATENT_C], &rows)?;
        let x = Tensor::new(dev, DType::F32, &[rows_all, DIM])?;
        {
            let e = Tensor::new(dev, DType::F32, &[nb * n, DIM])?;
            self.x_embed.forward(&input, &e)?;
            for k in 0..nb {
                x.copy_rows(k * s, &e, k * n, n)?;
                x.copy_rows(k * s + n, &self.suffix, 0, REGISTERS + 1)?;
            }
        }
        // positions: the frame's time, the cells' centres in [-1, 1) per axis, the suffix at 0; angles carry the 2 pi
        let mut pos = Vec::with_capacity(s * 3);
        let c = |i: usize, d: usize| (2.0 * ((i as f64 + 0.5) / d as f64) - 1.0) * std::f64::consts::TAU;
        for yi in 0..h {
            for xi in 0..w {
                pos.extend_from_slice(&[FRAME_POSITION * std::f64::consts::TAU, c(yi, h), c(xi, w)]);
            }
        }
        pos.extend(std::iter::repeat_n(0.0, (REGISTERS + 1) * 3));
        let one = rotations(&pos, &self.inv_freq);
        let all: Vec<f32> = (0..nb).flat_map(|_| one.iter().copied()).collect();
        let cs = f32_tensor(dev, &[rows_all, ROT_DIM / 2, 2], &all)?;
        let zero_rows = Tensor::from_bytes(dev, DType::I32, &[rows_all], &vec![0u8; rows_all * 4])?;

        let hbuf = Tensor::new(dev, act, &[rows_all, DIM])?;
        let qkv = Tensor::new(dev, act, &[rows_all, 3 * DIM])?;
        let att = Tensor::new(dev, act, &[rows_all, DIM])?;
        let proj = Tensor::new(dev, DType::F32, &[rows_all, DIM])?;
        let f1 = Tensor::new(dev, act, &[rows_all, 2 * 4 * DIM])?;
        let mlp = Tensor::new(dev, act, &[rows_all, 4 * DIM])?;
        // NS_PH3_PROFILE: a wait after each stage, the seconds summed by stage (slower; where the time goes)
        let prof = std::env::var_os("NS_PH3_PROFILE").is_some();
        let mut mark = std::time::Instant::now();
        let mut lap = |name: &'static str| -> Result<()> {
            if prof {
                dev.wait()?;
                let mut st = self.stages.lock().unwrap();
                match st.iter_mut().find(|(n, _)| *n == name) {
                    Some(e) => e.1 += mark.elapsed().as_secs_f64(),
                    None => st.push((name, mark.elapsed().as_secs_f64())),
                }
                mark = std::time::Instant::now();
            }
            Ok(())
        };
        for b in &self.blocks {
            ops::rms_norm_mod(&x, &b.norm1, EPS, None, &hbuf)?;
            lap("norm")?;
            b.qkv.forward(&hbuf, &qkv)?;
            lap("qkv")?;
            let all_rows = |i: usize| Rows { t: &qkv, offset: i * DIM, stride: 3 * DIM, tokens: rows_all, heads: HEADS, dim: HEAD_DIM };
            ops::rms_rope(all_rows(0), &self.ones, EPS, &cs, ROT_DIM)?;
            ops::rms_rope(all_rows(1), &self.ones, EPS, &cs, ROT_DIM)?;
            lap("rope")?;
            ops::attention_batch(all_rows(0), all_rows(1), all_rows(2), nb, &att)?;
            lap("attention")?;
            b.out.forward(&att, &proj)?;
            lap("out")?;
            ops::gate_add(&x, &proj, &zero_rows, &b.scale1)?;
            lap("add")?;
            ops::rms_norm_mod(&x, &b.norm2, EPS, None, &hbuf)?;
            lap("norm")?;
            b.w1.forward(&hbuf, &f1)?;
            lap("w1")?;
            ops::swiglu(&f1, &mlp)?;
            lap("swiglu")?;
            b.w2.forward(&mlp, &proj)?;
            lap("w2")?;
            ops::gate_add(&x, &proj, &zero_rows, &b.scale2)?;
            lap("add")?;
        }
        let normed = Tensor::new(dev, DType::F32, &[nb * n, DIM])?;
        {
            let img = Tensor::new(dev, DType::F32, &[nb * n, DIM])?;
            for k in 0..nb {
                img.copy_rows(k * n, &x, k * s, n)?;
            }
            ops::layer_norm(&img, DIM, Some(&self.norm_out_w), Some(&self.norm_out_b), EPS, &normed)?;
        }
        let per = OUT_C * PATCH * PATCH;
        let out = Tensor::new(dev, DType::F32, &[nb * n, per])?;
        self.proj_out.forward(&normed, &out)?;
        let o = out.to_f32()?;
        // token (h, w) -> pixels (c, h*16 + y, w*16 + x)
        let (fh, fw) = (h * PATCH, w * PATCH);
        let mut res = Vec::with_capacity(nb);
        for k in 0..nb {
            let o = &o[k * n * per..(k + 1) * n * per];
            let mut px = vec![0f32; OUT_C * fh * fw];
            for yi in 0..h {
                for xi in 0..w {
                    let tok = &o[(yi * w + xi) * per..][..per];
                    for ch in 0..OUT_C {
                        for y in 0..PATCH {
                            let dst = (ch * fh + yi * PATCH + y) * fw + xi * PATCH;
                            let src = (ch * PATCH + y) * PATCH;
                            px[dst..dst + PATCH].copy_from_slice(&tok[src..src + PATCH]);
                        }
                    }
                }
            }
            res.push(px);
        }
        Ok(res)
    }

    /// Denormalized latents after the 1x1 convolution `zp` [24, h, w] through the tiles: raw pixels [3, 16h, 16w]
    pub fn tiled(&self, zp: &[f32], h: usize, w: usize, tick: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let (fh, fw) = (h * PATCH, w * PATCH);
        let (ys, yo) = split_tiles(fh);
        let (xs, xo) = split_tiles(fw);
        let (th, tw) = (fh.min(TILE), fw.min(TILE));
        let (lh, lw) = (th / PATCH, tw / PATCH);
        let mut zs = Vec::with_capacity(ys.len() * xs.len());
        for &y0 in &ys {
            for &x0 in &xs {
                let (ly, lx) = (y0 / PATCH, x0 / PATCH);
                let mut zt = vec![0f32; LATENT_C * lh * lw];
                for c in 0..LATENT_C {
                    for yy in 0..lh {
                        let src = (c * h + ly + yy) * w + lx;
                        let dst = (c * lh + yy) * lw;
                        zt[dst..dst + lw].copy_from_slice(&zp[src..src + lw]);
                    }
                }
                zs.push(zt);
            }
        }
        let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(zs.len());
        let t0 = std::time::Instant::now();
        for batch in zs.chunks(self.tile_batch.max(1)) {
            tick(decoded.len(), zs.len())?;
            decoded.extend(self.tiles(batch, lh, lw)?);
        }
        if std::env::var_os("NS_PH3_PROFILE").is_some() {
            let st = std::mem::take(&mut *self.stages.lock().unwrap());
            let sum: f64 = st.iter().map(|s| s.1).sum();
            eprintln!("decoder: {} tiles in {:.2} s ({sum:.2} s in the blocks): {}", zs.len(), t0.elapsed().as_secs_f64(),
                      st.iter().map(|(n, t)| format!("{n} {t:.2}")).collect::<Vec<_>>().join(", "));
        }
        let t1 = std::time::Instant::now();
        // the cross-fades, as the reference's _stitch_tiles: a tile's top rows fade in from the tile above, then its left
        // columns from the tile on the left (both as decoded), then it is cropped by the overlap toward the next
        let tile = |i: usize, j: usize| &decoded[i * xs.len() + j];
        let mut canvas = vec![0f32; OUT_C * fh * fw];
        let mut out_y = 0;
        for i in 0..ys.len() {
            let mut out_x = 0;
            let mut keep_h = th;
            for j in 0..xs.len() {
                let kh = if i + 1 < ys.len() { th - yo[i] } else { th };
                let kw = if j + 1 < xs.len() { tw - xo[j] } else { tw };
                let t = tile(i, j);
                for ch in 0..OUT_C {
                    for y in 0..kh {
                        for xx in 0..kw {
                            let mut v = t[(ch * th + y) * tw + xx];
                            if i > 0 && y < yo[i - 1].min(th) {
                                let e = yo[i - 1].min(th);
                                let a = tile(i - 1, j)[(ch * th + th - e + y) * tw + xx];
                                let wb = y as f32 / e as f32;
                                v = a * (1.0 - wb) + v * wb;
                            }
                            if j > 0 && xx < xo[j - 1].min(tw) {
                                let e = xo[j - 1].min(tw);
                                // the left neighbour as it was decoded (the reference blends with the raw tiles)
                                let a = tile(i, j - 1)[(ch * th + y) * tw + tw - e + xx];
                                let wb = xx as f32 / e as f32;
                                v = a * (1.0 - wb) + v * wb;
                            }
                            canvas[(ch * fh + out_y + y) * fw + out_x + xx] = v;
                        }
                    }
                }
                out_x += kw;
                keep_h = kh;
            }
            out_y += keep_h;
        }
        if std::env::var_os("NS_PH3_PROFILE").is_some() {
            eprintln!("decoder: cross-fading {:.2} s", t1.elapsed().as_secs_f64());
        }
        Ok(canvas)
    }

    /// The 1x1 convolution on denormalized latents [24, n]
    pub fn post_quant(&self, zr: &[f32], n: usize) -> Vec<f32> {
        let mut zp = vec![0f32; zr.len()];
        for o in 0..LATENT_C {
            for i in 0..n {
                let mut acc = self.pq_b[o];
                for c in 0..LATENT_C {
                    acc += self.pq_w[o * LATENT_C + c] * zr[c * n + i];
                }
                zp[o * n + i] = acc;
            }
        }
        zp
    }

    /// Normalized latents [24, h, w] (what the sampler makes) -> the raw decoder output [3, 16h, 16w]
    pub fn decode_raw(&self, z: &[f32], h: usize, w: usize, tick: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let n = h * w;
        let mut zr = vec![0f32; z.len()];
        for c in 0..LATENT_C {
            for i in 0..n {
                zr[c * n + i] = z[c * n + i] * self.lat_std[c] + self.lat_mean[c];
            }
        }
        self.tiled(&self.post_quant(&zr, n), h, w, tick)
    }

    /// Normalized latents [24, h, w] -> RGB bytes [16h, 16w, 3]
    pub fn decode(&self, z: &[f32], h: usize, w: usize, tick: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<u8>> {
        Ok(to_rgb8(&self.decode_raw(z, h, w, tick)?, h * PATCH, w * PATCH))
    }
}

/// The raw decoder output [3, H, W] -> [H, W, 3] bytes: ImageNet's normalization undone, clamped, round(x * 255)
pub fn to_rgb8(raw: &[f32], fh: usize, fw: usize) -> Vec<u8> {
    let plane = fh * fw;
    let mut out = vec![0u8; plane * 3];
    for c in 0..OUT_C {
        for i in 0..plane {
            let v = (raw[c * plane + i] * IMAGENET_STD[c] + IMAGENET_MEAN[c]).clamp(0.0, 1.0);
            out[i * 3 + c] = (v * 255.0).round() as u8;
        }
    }
    out
}

/// An edit's conditioning latent from the encoder's moments [48, h, w] (mean, then log-variance): the posterior
/// sampled under its own generator (`noise`: torch.randn of [24, h, w] after `manual_seed(42)`), rounded to half,
/// normalized - diffusers' `encode_vae_condition`
pub fn condition_latent(moments: &[f32], noise: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    let n = moments.len() / 48;
    let mut z = vec![0f32; 24 * n];
    for c in 0..24 {
        for i in 0..n {
            let (mu, lv) = (moments[c * n + i], moments[(24 + c) * n + i].clamp(-30.0, 20.0));
            let s = mu + (0.5 * lv).exp() * noise[c * n + i];
            let s = nextsycl_h3::dtype::f16_to_f32(nextsycl_h3::dtype::f32_to_f16(s));
            z[c * n + i] = (s - mean[c]) / std[c];
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::split_tiles;

    #[test]
    fn tiles_follow_the_reference() {
        assert_eq!(split_tiles(256), (vec![0], vec![]));
        assert_eq!(split_tiles(384), (vec![0, 128], vec![128]));
        assert_eq!(split_tiles(576), (vec![0, 160, 320], vec![96, 96]));
        // 2912 px (8 MP's square): 15 tiles, 32 spare pixels in the first two overlaps
        let (s, o) = split_tiles(2912);
        assert_eq!(s.len(), 15);
        assert_eq!(o[..3], [80, 80, 64]);
        assert_eq!(s.last().unwrap() + 256, 2912);
    }
}
