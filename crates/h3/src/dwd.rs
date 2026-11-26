//! pi-H3's DWD head ("Decoding Without a Decoder", Lakonik/pi-H3's `*_dwd_adapter`): at the last sampling step the
//! denoiser's own final hidden states of the video rows become the clip's frames, and H3's video decoder never runs.
//! As LakonLab's `DWDHeadMixin` + `SolH3DWDHead` and ComfyUI-piFlow's `MiniMaxH3DWD` run it:
//!
//! ```text
//!   f = LayerNorm(h) * (1 + scale) + shift          (scale, shift) = dwd_norm_out.linear(silu(e(t)))   no affine
//!   x = feature_expand(f)                           each 2x2-latent token -> its 4 Sol tokens (one a latent pixel)
//!   x = the last 8 layers of the Sol-H3 decoder     shifted 5-latent x 16x16 windows + per-block global tokens
//!                                                   (the registers + a zero token), the fast decoder's (fastvae.rs)
//!   px = norm_out, then proj_out (4 frames a latent) or first_frame_proj_out (the one frame of a block's first)
//! ```
//!
//! The frames come out in Sol's ImageNet-normalized RGB, the decoders' own. The norm's scale and shift are one row for
//! all the video rows (one timestep), so they are folded into `feature_expand` once a run: the expansion of the plain
//! layer norm with `W diag(1 + scale)` and `b + W shift` - float32 on the card, the expansion rounded to bfloat16 as
//! the reference rounds its features. The layers run on the H3 video silo's kernels (`fastvae::install`); without the
//! silo there is no head, and a clip is decoded by the VAE from its last latents.

use std::time::Instant;

use crate::device::Tensor;
use crate::dtype::DType;
use crate::fastvae::Sol;
use crate::ops::{self, Linear};
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

/// The Sol decoder's width, and Sol tokens a denoiser token covers (32 x 32 pixels: 2 x 2 latent pixels)
const DIM: usize = 2048;
const EXPAND: usize = 4;
const EPS: f32 = 1e-5;
/// latent frames a native block (the decoder's)
const BLOCK: usize = 5;

pub struct DwdHead {
    sol: Sol,
    /// dwd_norm_out.linear: [2 hidden, t_dim] and its bias, float32 on the host - (scale, shift) of silu(e(t))
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    /// feature_expand: [4 DIM, hidden] and its bias, float32 on the host (a run folds the norm's scale and shift in)
    expand_w: Vec<f32>,
    expand_b: Vec<f32>,
    pub hidden: usize,
    pub t_dim: usize,
    pub load_seconds: f64,
    pub load_bytes: u64,
    /// "int8" or "bfloat16": the layers' matrices as the adapter file has them
    pub form: &'static str,
}

fn f32_of(ck: &Checkpoint, name: &str, bytes: &mut u64) -> Result<(Vec<usize>, Vec<f32>)> {
    let e = ck.get(name)?;
    *bytes += e.bytes as u64;
    Ok((e.shape.clone(), crate::dtype::bytes_to_f32(&ck.read(name)?, e.dtype)?))
}

impl DwdHead {
    /// Whether an adapter file has a DWD head (`dwd_norm_out.*` and `dwd_pixel_refiner.*`)
    pub fn in_file(ck: &Checkpoint) -> bool {
        ck.entries.contains_key("dwd_norm_out.linear.weight") && ck.entries.contains_key("dwd_pixel_refiner.feature_expand.weight")
    }

    /// The head of a DWD adapter (its ComfyUI int8 ConvRot file or the Diffusers float16 one) on GPU `index`, in a context
    /// on nextsycl-core's queue of that GPU; None without the H3 video silo
    pub fn load(index: usize, ck: &Checkpoint, threads: usize) -> Result<Option<DwdHead>> {
        let t0 = Instant::now();
        if ck.entries.keys().any(|k| k.starts_with("dwd_linear_head.") || k.starts_with("first_frame_dwd_linear_head.")) {
            return Err(Error(format!("{}: a DWD head with linear heads (only the Sol pixel refiner is ported)", ck.path.display())));
        }
        let Some((gpu, dev, k)) = Sol::open(index)? else { return Ok(None) };
        let mut bytes = 0u64;
        let sol = Sol::load(gpu, dev, k, ck, "dwd_pixel_refiner.", threads, &mut bytes)?;
        let (ns, norm_w) = f32_of(ck, "dwd_norm_out.linear.weight", &mut bytes)?;
        let (_, norm_b) = f32_of(ck, "dwd_norm_out.linear.bias", &mut bytes)?;
        let (es, expand_w) = f32_of(ck, "dwd_pixel_refiner.feature_expand.weight", &mut bytes)?;
        let (_, expand_b) = f32_of(ck, "dwd_pixel_refiner.feature_expand.bias", &mut bytes)?;
        let (hidden, t_dim) = (ns[0] / 2, ns[1]);
        if es != [EXPAND * DIM, hidden] || norm_b.len() != 2 * hidden || expand_b.len() != EXPAND * DIM {
            return Err(Error(format!("{}: dwd_norm_out {ns:?} / feature_expand {es:?} do not fit a {hidden}-wide denoiser and {EXPAND} Sol tokens of {DIM}",
                                     ck.path.display())));
        }
        let form = if ck.quant("dwd_pixel_refiner.blocks.0.attn.to_qkv")?.is_some() { "int8" } else { "bfloat16" };
        sol.dev.wait()?;
        Ok(Some(DwdHead { sol, norm_w, norm_b, expand_w, expand_b, hidden, t_dim, load_seconds: t0.elapsed().as_secs_f64(), load_bytes: bytes, form }))
    }

    pub fn layers(&self) -> usize {
        self.sol.layers()
    }

    /// The GPU it runs on
    pub fn gpu(&self) -> usize {
        self.sol.gpu()
    }

    /// Bytes on the card a decode of t x h x w latent pixels needs beyond the weights: the fast decoder's plus the folded
    /// expansion and a block's features
    pub fn need_bytes(t: usize, h: usize, w: usize) -> usize {
        crate::fastvae::FastDecoder::need_bytes(t, h, w) + EXPAND * DIM * 4 * 5376 + BLOCK.min(t) * h * w * (5376 + 2 * DIM) * 4
    }

    /// The video rows' final hidden states [t h2 w2, hidden] (float32, the rows' (t, y, x) order; t = 0 a block's first)
    /// and their timestep's curve point `temb` [t_dim] -> frames [3, frames, 32 h2, 32 w2] in [0, 1] and their count.
    /// `tick` between the decoder's temporal windows (cancel checks).
    pub fn decode(&self, hidden: &[f32], temb: &[f32], t: usize, h2: usize, w2: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let c = self.hidden;
        let (h, w) = (2 * h2, 2 * w2);
        if hidden.len() != t * h2 * w2 * c || temb.len() != self.t_dim {
            return Err(Error(format!("dwd: {} hidden values and a {}-value curve point for {t}x{h2}x{w2} tokens of {c}", hidden.len(), temb.len())));
        }
        let dev = &self.sol.dev;
        let k = self.sol.k;
        // the norm's scale and shift at this timestep, folded into the expansion (float64 sums)
        let silu: Vec<f64> = temb.iter().map(|v| *v as f64 / (1.0 + (-*v as f64).exp())).collect();
        let mods: Vec<f64> = (0..2 * c).map(|o| self.norm_b[o] as f64 + self.norm_w[o * self.t_dim..(o + 1) * self.t_dim].iter().zip(&silu).map(|(a, b)| *a as f64 * b).sum::<f64>())
                                     .collect();
        let (scale, shift) = mods.split_at(c);
        let rows = EXPAND * DIM;
        let mut wf = vec![0f32; rows * c];
        let mut bf = vec![0f32; rows];
        std::thread::scope(|s| {
            let per = rows.div_ceil(8);
            for ((wc, bc), r0) in wf.chunks_mut(per * c).zip(bf.chunks_mut(per)).zip((0..rows).step_by(per)) {
                s.spawn(move || {
                    for (i, (wr, bo)) in wc.chunks_mut(c).zip(bc.iter_mut()).enumerate() {
                        let src = &self.expand_w[(r0 + i) * c..(r0 + i + 1) * c];
                        let mut acc = self.expand_b[r0 + i] as f64;
                        for j in 0..c {
                            wr[j] = (src[j] as f64 * (1.0 + scale[j])) as f32;
                            acc += src[j] as f64 * shift[j];
                        }
                        *bo = acc as f32;
                    }
                });
            }
        });
        let f32_tensor = |shape: &[usize], v: &[f32]| Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>());
        let expand = Linear { weight: f32_tensor(&[rows, c], &wf)?, bias: Some(f32_tensor(&[rows], &bf)?) };
        drop((wf, bf));
        // the residual stream, a native block of latent frames at a time: the plain layer norm, the folded expansion
        // (bfloat16 out: [4 n, DIM], each token's 4 Sol tokens), the Sol tokens into (t, y, x) order, widened
        let x = Tensor::new(dev, DType::F32, &[t * h * w, DIM])?;
        {
            let nmax = BLOCK.min(t) * h2 * w2;
            let fin = Tensor::new(dev, DType::F32, &[nmax, c])?;
            let e = Tensor::new(dev, DType::BF16, &[nmax * EXPAND, DIM])?;
            let o = Tensor::new(dev, DType::BF16, &[nmax * EXPAND, DIM])?;
            for f0 in (0..t).step_by(BLOCK) {
                tick()?;
                let f1 = (f0 + BLOCK).min(t);
                let n = (f1 - f0) * h2 * w2;
                let bytes: Vec<u8> = hidden[f0 * h2 * w2 * c..f1 * h2 * w2 * c].iter().flat_map(|f| f.to_le_bytes()).collect();
                // SAFETY: views of live tensors, dropped at the end of the iteration.
                let (fv, ev, ov, xv) = unsafe { (fin.rows_view(0, n)?, e.rows_view(0, n * EXPAND)?, o.rows_view(0, n * EXPAND)?, x.rows_view(f0 * h * w, n * EXPAND)?) };
                fv.buf.write(0, &bytes)?;
                ops::layer_norm(&fv, c, None, None, EPS, &fv)?;
                // SAFETY: the expansion writes [n, 4 DIM] = [4 n, DIM] bfloat16 rows, the size of ev.
                let ew = unsafe { ev.rows_view(0, n * EXPAND)? };
                let ew = Tensor { shape: vec![n, EXPAND * DIM], ..ew };
                expand.forward(&fv, &ew)?;
                // sub-row r = token * 4 + ey * 2 + ex of token (tl, i, j) -> Sol token (tl, 2 i + ey, 2 j + ex)
                let pos: Vec<i32> = (0..n * EXPAND).map(|r| {
                    let (tok, sub) = (r / EXPAND, r % EXPAND);
                    let (tl, rem) = (tok / (h2 * w2), tok % (h2 * w2));
                    let (i, j) = (rem / w2, rem % w2);
                    ((tl * h + 2 * i + sub / 2) * w + 2 * j + sub % 2) as i32
                }).collect();
                let pt = Tensor::from_bytes(dev, DType::I32, &[pos.len()], &pos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                // SAFETY: n * 4 rows of DIM bfloat16 values from ev into ov at the rows the table names (all inside ov).
                k.check(unsafe {
                    (k.rows)(self.sol.queue(), ev.buf.ptr(), DIM as i64, std::ptr::null(), ov.buf.ptr(), DIM as i64, pt.buf.ptr().cast(), (n * EXPAND) as i64, DIM as i64)
                }, "rows (dwd)")?;
                crate::fastvae::widen(dev, &ov, &xv, n * EXPAND)?;
            }
        }
        drop(expand);
        self.sol.run(&x, t, h, w, tick)
    }
}

/// The frames a DWD head makes of t latent frames: 4 a latent, 1 for the first of every 5-latent block
pub fn frames(t: usize) -> usize {
    4 * t - 3 * t.div_ceil(BLOCK)
}

#[cfg(test)]
mod tests {
    #[test]
    fn frames_follow_17k_plus_5() {
        // 22 frames: 7 latents; 243 (10 s): 72; 362: 107
        for (t, f) in [(7, 22), (72, 243), (107, 362), (2, 5)] {
            assert_eq!(super::frames(t), f);
        }
    }
}
