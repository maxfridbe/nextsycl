//! Qwen3-VL's vision tower: pictures to the language model's image tokens (transformers' Qwen3VLVisionModel), from
//! the bf16 tensors in the same file as the text tower (`model.visual.*`; MiniMax H3's 32B: `visual.*` in a file of
//! its own), float32 on the GPU. The 8B and the 32B towers are the same but for the merger's output width.
//!
//! ```text
//!   picture (RGB, sides multiples of 32) -> 16 x 16 patches, each twice in time (3 x 2 x 16 x 16 = 1,536 values,
//!     (x / 255 - 0.5) / 0.5), in 2 x 2 merge-block order
//!   -> linear 1,536 -> 1,152 + the learned 48 x 48 position grid resampled (bilinear, align_corners) to the picture's
//!   -> 27 blocks: x += proj(attention(rope2d(qkv(layernorm(x)))));  x += fc2(gelu_tanh(fc1(layernorm(x))))
//!      16 heads of 72, 2-D RoPE (height and width angles side by side); after blocks 8, 16, 24 a deepstack merger
//!   -> merger: layernorm (1,152), each 2 x 2 block's four tokens as one row of 4,608 -> fc1, gelu, fc2 -> 4,096
//!   deepstack mergers: the four tokens as one row first, then layernorm (4,608), fc1, gelu, fc2 -> 4,096
//! ```

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

pub const PATCH: usize = 16;
pub const MERGE: usize = 2;
const TEMPORAL: usize = 2;
const PATCH_IN: usize = 3 * TEMPORAL * PATCH * PATCH;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;
/// the blocks after which the deepstack mergers read (Qwen3-VL 8B's deepstack_visual_indexes)
const DEEP_AT: [usize; 3] = [8, 16, 24];

struct Block {
    n1w: DevBuf,
    n1b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
    n2w: DevBuf,
    n2b: DevBuf,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
}

struct Merger {
    nw: DevBuf,
    nb: DevBuf,
    /// the norm is over the four tokens' row (deepstack) rather than each token (the final merger)
    post: bool,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
}

pub struct Vision {
    gpu: Arc<Gpu>,
    patch: DevBuf,
    patch_b: DevBuf,
    /// the learned position grid, on the host [side * side, hidden]
    pos: Vec<f32>,
    side: usize,
    blocks: Vec<Block>,
    deep_at: Vec<usize>,
    deep: Vec<Merger>,
    merger: Merger,
    pub hidden: usize,
    heads: usize,
    ffn: usize,
    pub out: usize,
}

/// What a picture gives the language model: its tokens [n, out] and the deepstack features (one [n, out] each)
pub struct Seen {
    pub tokens: DevBuf,
    pub deep: Vec<DevBuf>,
    pub n: usize,
    /// the patch grid (height, width) in 16-pixel patches
    pub grid: (usize, usize),
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// The processor's patches of a picture: RGB float32 [H, W, 3] in 0..255 (sides multiples of 32) -> [n, 1,536] in
/// merge-block order (block row, block column, row in block, column in block), each patch (channel, time, y, x)
pub fn patches(rgb: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, (usize, usize))> {
    patches_frames(&[rgb, rgb], h, w)
}

/// The same for two frames filling the patches' two time steps (a video's frame pair; a picture is its own pair)
pub fn patches_frames(frames: &[&[f32]; TEMPORAL], h: usize, w: usize) -> Result<(Vec<f32>, (usize, usize))> {
    if !h.is_multiple_of(PATCH * MERGE) || !w.is_multiple_of(PATCH * MERGE) || frames.iter().any(|f| f.len() != h * w * 3) {
        return Err(Error(format!("a {w}x{h} picture: the vision tower takes sides that are multiples of {}", PATCH * MERGE)));
    }
    let (gh, gw) = (h / PATCH, w / PATCH);
    let mut out = Vec::with_capacity(gh * gw * PATCH_IN);
    for br in 0..gh / MERGE {
        for bc in 0..gw / MERGE {
            for ir in 0..MERGE {
                for ic in 0..MERGE {
                    let (py, px) = ((br * MERGE + ir) * PATCH, (bc * MERGE + ic) * PATCH);
                    for c in 0..3 {
                        for rgb in frames {
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    out.push((rgb[((py + y) * w + px + x) * 3 + c] / 255.0 - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((out, (gh, gw)))
}

/// Each patch's (row, column), in merge-block order
fn positions(gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(gh * gw);
    for br in 0..gh / MERGE {
        for bc in 0..gw / MERGE {
            for ir in 0..MERGE {
                for ic in 0..MERGE {
                    v.push((br * MERGE + ir, bc * MERGE + ic));
                }
            }
        }
    }
    v
}

/// Bilinear taps (align_corners) of `n` output positions over `side` grid points: (low, high, weight of high)
fn taps(i: usize, n: usize, side: usize) -> (usize, usize, f32) {
    let src = if n > 1 { i as f64 * (side - 1) as f64 / (n - 1) as f64 } else { 0.0 };
    let lo = (src.floor() as usize).min(side - 1);
    let hi = (lo + 1).min(side - 1);
    (lo, hi, (src - lo as f64) as f32)
}

impl Vision {
    /// The tower from the text encoder's file (`model.visual.*`)
    pub fn load(st: &SafeTensors, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        Self::load_from(st, "model.visual.", gpu, log)
    }

    /// The tower from the tensors under `prefix` (`visual.` in H3's file of the 32B tower alone)
    pub fn load_from(st: &SafeTensors, prefix: &str, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vision> {
        let t0 = std::time::Instant::now();
        let up = |n: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &st.f32(&format!("{prefix}{n}")).map_err(ge)?) };
        let shape = |n: &str| -> Result<Vec<u64>> { Ok(st.need(&format!("{prefix}{n}")).map_err(ge)?.shape.clone()) };
        let hidden = shape("patch_embed.proj.bias")?[0] as usize;
        let ffn = shape("blocks.0.mlp.linear_fc1.bias")?[0] as usize;
        let out = shape("merger.linear_fc2.bias")?[0] as usize;
        let n_blocks = (0..).take_while(|i| st.tensor(&format!("{prefix}blocks.{i}.norm1.weight")).is_some()).count();
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let p = |s: &str| format!("blocks.{i}.{s}");
            blocks.push(Block {
                n1w: up(&p("norm1.weight"))?, n1b: up(&p("norm1.bias"))?,
                qkv: up(&p("attn.qkv.weight"))?, qkv_b: up(&p("attn.qkv.bias"))?,
                proj: up(&p("attn.proj.weight"))?, proj_b: up(&p("attn.proj.bias"))?,
                n2w: up(&p("norm2.weight"))?, n2b: up(&p("norm2.bias"))?,
                fc1: up(&p("mlp.linear_fc1.weight"))?, fc1_b: up(&p("mlp.linear_fc1.bias"))?,
                fc2: up(&p("mlp.linear_fc2.weight"))?, fc2_b: up(&p("mlp.linear_fc2.bias"))?,
            });
        }
        let merger = |p: &str, post: bool| -> Result<Merger> {
            Ok(Merger {
                nw: up(&format!("{p}.norm.weight"))?, nb: up(&format!("{p}.norm.bias"))?, post,
                fc1: up(&format!("{p}.linear_fc1.weight"))?, fc1_b: up(&format!("{p}.linear_fc1.bias"))?,
                fc2: up(&format!("{p}.linear_fc2.weight"))?, fc2_b: up(&format!("{p}.linear_fc2.bias"))?,
            })
        };
        let n_deep = (0..).take_while(|i| st.tensor(&format!("{prefix}deepstack_merger_list.{i}.norm.weight")).is_some()).count();
        let deep = (0..n_deep).map(|i| merger(&format!("deepstack_merger_list.{i}"), true)).collect::<Result<Vec<_>>>()?;
        // the config's deepstack_visual_indexes (Qwen3-VL 8B: 8, 16, 24; the file does not carry them)
        let deep_at: Vec<usize> = DEEP_AT.iter().copied().take(n_deep).collect();
        if n_deep != DEEP_AT.len() || n_blocks <= DEEP_AT[DEEP_AT.len() - 1] {
            return Err(Error(format!("the vision tower: {n_blocks} blocks, {n_deep} deepstack mergers - not Qwen3-VL 8B's 27 and 3")));
        }
        let pos = st.f32(&format!("{prefix}pos_embed.weight")).map_err(ge)?;
        let side = ((pos.len() / hidden) as f64).sqrt() as usize;
        let pw = st.f32(&format!("{prefix}patch_embed.proj.weight")).map_err(ge)?;
        if pw.len() != hidden * PATCH_IN {
            return Err(Error(format!("the vision tower's patch embedding is {} values, not {hidden} x {PATCH_IN}", pw.len())));
        }
        let v = Vision {
            gpu: gpu.clone(),
            patch: DevBuf::from_f32(gpu, &pw)?,
            patch_b: up("patch_embed.proj.bias")?,
            pos,
            side,
            blocks,
            deep_at,
            deep,
            merger: merger("merger", false)?,
            hidden,
            heads: hidden / 72,
            ffn,
            out,
        };
        log(format!("qwen3vl vision: {n_blocks} blocks ({hidden} wide), deepstack after {:?}, on {} in {:.1} s", v.deep_at, gpu.name,
                    t0.elapsed().as_secs_f64()));
        Ok(v)
    }

    fn merge(&self, nsd: &Nsd, m: &Merger, x: &DevBuf, n: usize) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let (d, k) = (self.hidden, self.hidden * MERGE * MERGE);
        let rows = n / (MERGE * MERGE);
        let normed = DevBuf::f32(gpu, n * d)?;
        if m.post {
            nsd.layer_norm(x.ptr(), Dt::F32, rows, k, m.nw.ptr(), m.nb.ptr(), EPS, normed.ptr(), Dt::F32)?;
        } else {
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, m.nw.ptr(), m.nb.ptr(), EPS, normed.ptr(), Dt::F32)?;
        }
        let h = DevBuf::f32(gpu, rows * k)?;
        nsd.linear(normed.ptr(), Dt::F32, rows, k, m.fc1.ptr(), k, m.fc1_b.ptr(), h.ptr(), Dt::F32)?;
        nsd.gelu(h.ptr(), Dt::F32, rows * k, false)?;
        let o = DevBuf::f32(gpu, rows * self.out)?;
        nsd.linear(h.ptr(), Dt::F32, rows, k, m.fc2.ptr(), self.out, m.fc2_b.ptr(), o.ptr(), Dt::F32)?;
        nsd.wait()?;
        Ok(o)
    }

    /// A picture's tokens: RGB float32 [H, W, 3] in 0..255
    pub fn see(&self, nsd: &Nsd, rgb: &[f32], hh: usize, ww: usize) -> Result<Seen> {
        let (px, grid) = patches(rgb, hh, ww)?;
        self.see_patches(nsd, &px, grid)
    }

    /// The tokens from the processor's patches [n, 1,536] of a (rows, columns) patch grid
    pub fn see_patches(&self, nsd: &Nsd, px: &[f32], (gh, gw): (usize, usize)) -> Result<Seen> {
        let gpu = &self.gpu;
        let n = gh * gw;
        let d = self.hidden;
        let pos = positions(gh, gw);
        // the learned grid resampled to this one, added to the patches' embedding
        let mut pe = vec![0f32; n * d];
        for (i, (r, c)) in pos.iter().enumerate() {
            let (r0, r1, fr) = taps(*r, gh, self.side);
            let (c0, c1, fc) = taps(*c, gw, self.side);
            for (row, wgt) in [(r0 * self.side + c0, (1.0 - fr) * (1.0 - fc)), (r0 * self.side + c1, (1.0 - fr) * fc), (r1 * self.side + c0, fr * (1.0 - fc)),
                               (r1 * self.side + c1, fr * fc)] {
                for j in 0..d {
                    pe[i * d + j] += wgt * self.pos[row * d + j];
                }
            }
        }
        // 2-D RoPE: angles of the row, then of the column (a quarter of the head each), over the head's two halves
        let hd = d / self.heads;
        let q = hd / 4;
        let inv: Vec<f32> = (0..q).map(|i| (1.0 / THETA.powf((2 * i) as f64 / (hd / 2) as f64)) as f32).collect();
        let mut cs = vec![0f32; n * hd];
        for (t, (r, c)) in pos.iter().enumerate() {
            for i in 0..hd / 2 {
                let a = if i < q { *r as f32 * inv[i] } else { *c as f32 * inv[i - q] };
                cs[t * hd + 2 * i] = a.cos();
                cs[t * hd + 2 * i + 1] = a.sin();
            }
        }
        let cs = DevBuf::from_f32(gpu, &cs)?;
        let input = DevBuf::from_f32(gpu, px)?;
        let x = DevBuf::f32(gpu, n * d)?;
        nsd.linear(input.ptr(), Dt::F32, n, PATCH_IN, self.patch.ptr(), d, self.patch_b.ptr(), x.ptr(), Dt::F32)?;
        let peb = DevBuf::from_f32(gpu, &pe)?;
        nsd.gate_add(x.ptr(), Dt::F32, n, d, peb.ptr(), Dt::F32, none(), none())?;
        let (hb, qkv, att, o) = (DevBuf::f32(gpu, n * d)?, DevBuf::f32(gpu, n * 3 * d)?, DevBuf::f32(gpu, n * d)?, DevBuf::f32(gpu, n * d)?);
        let mlp = DevBuf::f32(gpu, n * self.ffn)?;
        let mut deep = Vec::new();
        let at = |b: &DevBuf, i: usize| -> *mut std::ffi::c_void { b.fp().wrapping_add(i).cast() };
        for (bi, b) in self.blocks.iter().enumerate() {
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, b.n1w.ptr(), b.n1b.ptr(), EPS, hb.ptr(), Dt::F32)?;
            nsd.linear(hb.ptr(), Dt::F32, n, d, b.qkv.ptr(), 3 * d, b.qkv_b.ptr(), qkv.ptr(), Dt::F32)?;
            nsd.rope(qkv.ptr(), Dt::F32, n, self.heads, hd, 3 * d, cs.ptr(), hd)?;
            nsd.rope(at(&qkv, d), Dt::F32, n, self.heads, hd, 3 * d, cs.ptr(), hd)?;
            nsd.attention(qkv.ptr(), at(&qkv, d), at(&qkv, 2 * d), Dt::F32, n, self.heads, hd, 3 * d, att.ptr(), Dt::F32)?;
            nsd.linear(att.ptr(), Dt::F32, n, d, b.proj.ptr(), d, b.proj_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, d, o.ptr(), Dt::F32, none(), none())?;
            nsd.layer_norm(x.ptr(), Dt::F32, n, d, b.n2w.ptr(), b.n2b.ptr(), EPS, hb.ptr(), Dt::F32)?;
            nsd.linear(hb.ptr(), Dt::F32, n, d, b.fc1.ptr(), self.ffn, b.fc1_b.ptr(), mlp.ptr(), Dt::F32)?;
            nsd.gelu(mlp.ptr(), Dt::F32, n * self.ffn, true)?;
            nsd.linear(mlp.ptr(), Dt::F32, n, self.ffn, b.fc2.ptr(), d, b.fc2_b.ptr(), o.ptr(), Dt::F32)?;
            nsd.gate_add(x.ptr(), Dt::F32, n, d, o.ptr(), Dt::F32, none(), none())?;
            if let Some(k) = self.deep_at.iter().position(|l| *l == bi) {
                deep.push(self.merge(nsd, &self.deep[k], &x, n)?);
            }
        }
        let tokens = self.merge(nsd, &self.merger, &x, n)?;
        Ok(Seen { tokens, deep, n: n / (MERGE * MERGE), grid: (gh, gw) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_come_in_merge_blocks() {
        // a 32 x 64 picture: 2 x 4 patches, two merge blocks side by side
        let (h, w) = (32, 64);
        let rgb: Vec<f32> = (0..h * w * 3).map(|i| ((i / 3) % w) as f32).collect();
        let (px, grid) = patches(&rgb, h, w).unwrap();
        assert_eq!(grid, (2, 4));
        assert_eq!(px.len(), 8 * PATCH_IN);
        // the 2nd patch of the 1st block is column 1 (x from 16); the 5th patch (2nd block) column 2 (x from 32)
        let first = |p: usize| px[p * PATCH_IN] * 0.5 * 255.0 + 127.5;
        assert_eq!(first(1), 16.0);
        assert_eq!(first(4), 32.0);
        assert_eq!(positions(2, 4)[4], (0, 2));
        assert_eq!(taps(0, 32, 48), (0, 1, 0.0));
        assert_eq!(taps(31, 32, 48).1, 47);
    }
}
