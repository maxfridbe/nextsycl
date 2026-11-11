//! MiniMax-H3-Image's denoiser around H3's block stack (crates/h3 `dit::Blocks`, `TextRefiner`, `layout`: the same code
//! the video engine runs): ComfyUI's `MiniMaxH3ImageModel` - H3 without the audio stream, on one latent frame, with a
//! bank of six output heads.
//!
//! ```text
//!   x = [ text | the edit's pictures (2 x 2 patches of their latents) | the image's 2 x 2 patches ]     bfloat16
//!   x = H3's 50 blocks (tables per timestep and modality; positions and tags as H3's layout)
//!   v = heads(norm(x) (1 + scale[t]) + shift[t]) on the image's rows, float32; velocity = -v
//!   heads = head 0 + the heads of the grid intervals the step spans, weighted by their share of it (`head_weights`)
//! ```
//!
//! Sampling is Euler over the heads' grid (`PDD_SIGMAS`: six steps), no guidance; another step count reads the grid at
//! even positions (ComfyUI's MiniMaxH3ImageModelSampling, stable-diffusion.cpp's "minimax-h3-image" schedule).

use std::sync::Arc;

use nextsycl_h3::denoiser::{bf16_bytes, patchify, unpatchify};
use nextsycl_h3::device::{Device, Tensor};
use nextsycl_h3::dit::{host, small, Blocks, Config, Scratch, Step};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::layout::{Kind, Layout, RefBlock, Timesteps, VISUAL_COND_TIMESTEP};
use nextsycl_h3::noise::Mt19937;
use nextsycl_h3::ops::{self, Linear, Mod};
use nextsycl_h3::safetensors::Checkpoint;
use nextsycl_h3::{Error, Result};

/// The grid the six heads were distilled on (the checkpoint's `pdd_sigmas`, as ComfyUI's model detection and
/// stable-diffusion.cpp give them)
pub const PDD_SIGMAS: [f32; 7] = [1.0, 0.967_523_7, 0.887_548_5, 0.75, 0.532_773_9, 0.232_008_87, 0.0];
/// latent channels, and the 2 x 2 patch a token covers
pub const LATENT_C: usize = 24;
const PATCH: usize = 2;
/// the final layer runs over the image's rows this many at a time (its float32 intermediates stay small)
const FINAL_CHUNK: usize = 4096;
const FINAL_EPS: f32 = 1e-5;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &f32_bytes(v))
}

/// The weights of the six heads for a step from `sigma` to `next` (ComfyUI's FinalLayer + `_pdd_head`): the grid
/// points nearest to each give the intervals the step spans; head 0 always at 1, every later head of those
/// intervals by its share of their length. On the grid: step 0 is head 0, step i head 0 + head i.
pub fn head_weights(sigma: f32, next: f32) -> [f32; 6] {
    let nearest = |v: f32| (0..PDD_SIGMAS.len()).min_by(|a, b| (PDD_SIGMAS[*a] - v).abs().total_cmp(&(PDD_SIGMAS[*b] - v).abs())).unwrap_or(0);
    let n = PDD_SIGMAS.len() - 1;
    let start = nearest(sigma).min(n - 1);
    let stop = nearest(next).max(start + 1);
    let total: f32 = (start..stop).map(|k| PDD_SIGMAS[k] - PDD_SIGMAS[k + 1]).sum();
    let mut w = [0f32; 6];
    w[0] = 1.0;
    for (k, wk) in w.iter_mut().enumerate().take(stop).skip(start.max(1)) {
        *wk = (PDD_SIGMAS[k] - PDD_SIGMAS[k + 1]) / total;
    }
    w
}

/// The sampling grid of `steps` steps: the heads' own at six; otherwise the grid read at even positions, linearly
/// between its points (stable-diffusion.cpp's MiniMaxH3ImageDenoiser, ComfyUI's model sampling)
pub fn sigmas(steps: usize) -> Vec<f32> {
    if steps == PDD_SIGMAS.len() - 1 {
        return PDD_SIGMAS.to_vec();
    }
    let n = (PDD_SIGMAS.len() - 1) as f32;
    (0..=steps)
        .map(|i| {
            let pos = i as f32 / steps as f32 * n;
            let k = (pos as usize).min(PDD_SIGMAS.len() - 2);
            PDD_SIGMAS[k] + (pos - k as f32) * (PDD_SIGMAS[k + 1] - PDD_SIGMAS[k])
        })
        .collect()
}

/// The schedule ComfyUI's MiniMaxH3SigmaShift (shift_video `shift`) + BasicScheduler "simple" gives at `steps` steps
/// (the pDMD 2-step LoRA's recipe: shift 12, 2 steps -> 1, 0.923, 0): ModelSamplingDiscreteFlow's 1,000-entry table,
/// sigma(t) = shift t / (1 + (shift - 1) t) at t = (i / 1000 * 1000) / 1000 in float32, read by simple_scheduler at
/// index 999 - floor(x * 1000 / steps) for x < steps, then 0. The heads are still weighted by their own grid
/// (`head_weights`), as ComfyUI's FinalLayer does.
pub fn shifted_sigmas(shift: f32, steps: usize) -> Vec<f32> {
    // Python's (alpha - 1) is a double, cast to float32 where it meets the table
    let less = (shift as f64 - 1.0) as f32;
    let table = |i: usize| -> f32 {
        let t = (i as f32 / 1000.0) * 1000.0 / 1000.0;
        if shift == 1.0 {
            t
        } else {
            shift * t / (1.0 + less * t)
        }
    };
    let ss = 1000.0 / steps as f64;
    let mut out: Vec<f32> = (0..steps).map(|x| table(1000 - (x as f64 * ss) as usize)).collect();
    out.push(0.0);
    out
}

/// ComfyUI's BasicScheduler: `steps` steps of a schedule at `denoise` (below 1: the last steps + 1 points of the
/// schedule made for floor(steps / denoise) steps - the img2img start). `shift`: the shifted flow schedule
/// (`shifted_sigmas`); none: the heads' grid (`sigmas`). The denoise is read as the decimal it was written as (a
/// request's 0.6 is 0.6, not float32's 0.60000002, which would make 3 / 0.6 fall to 4).
pub fn schedule(steps: usize, shift: Option<f32>, denoise: f32) -> Vec<f32> {
    let total = if denoise < 1.0 {
        let d = (denoise as f64 * 1e6).round() / 1e6;
        ((steps as f64 / d) as usize).max(steps)
    } else {
        steps
    };
    let full = match shift {
        Some(s) => shifted_sigmas(s, total),
        None => sigmas(total),
    };
    full[full.len() - steps - 1..].to_vec()
}

/// An img2img from `sigma0` on the shifted schedule (the live editor's contract: a strength below 1 is the start sigma):
/// sigma0 taken back to the shift's time t0 = s0 / (shift - (shift - 1) s0), `steps` even steps of t from t0 to 0, each
/// shifted again - 0.55 at shift 12 and 2 steps: 0.55, 0.367, 0
pub fn shifted_from(sigma0: f32, steps: usize, shift: f32) -> Vec<f32> {
    let (s, a) = (sigma0 as f64, shift as f64);
    let t0 = s / (a - (a - 1.0) * s);
    (0..=steps).map(|i| {
        let t = t0 * (1.0 - i as f64 / steps as f64);
        (a * t / (1.0 + (a - 1.0) * t)) as f32
    }).collect()
}

/// The demo Space's detail pass (akhaliq/minimax-h3-image-2step app.py `_refine_grid`): the refiner's own 3-step grid
/// 0.909375, 0.857143, 0.666667 scaled to start at `sigma0`, its ratios kept, then 0. Another step count: `full`, the
/// schedule of that many steps from 1, scaled the same way (the Space only runs 3).
pub fn start_grid(sigma0: f32, full: &[f32]) -> Vec<f32> {
    const BASE: [f32; 3] = [0.909_375, 0.857_143, 0.666_667];
    let steps = full.len() - 1;
    let mut g: Vec<f32> = if steps == BASE.len() {
        BASE.iter().map(|b| sigma0 * (b / BASE[0])).collect()
    } else {
        full[..steps].iter().map(|s| sigma0 * (s / full[0])).collect()
    };
    g.push(0.0);
    g
}

/// One Euler step as ComfyUI's sampler takes it: denoised = x - v sigma, d = (x - denoised) / sigma, x += d (next - sigma)
pub fn euler(x: &mut [f32], v: &[f32], sigma: f32, next: f32) {
    for (x, v) in x.iter_mut().zip(v) {
        let denoised = *x - *v * sigma;
        let d = (*x - denoised) / sigma;
        *x += d * (next - sigma);
    }
}

/// The parts around the blocks: the patch projection in, the final layer and the head bank out, the timestep curve
/// and the rotation frequencies
pub struct Outer {
    video_patch: Linear,
    final_norm: Tensor,
    /// [2 * hidden, t_dim] and its bias: shift, then scale
    final_w: Vec<f32>,
    final_b: Vec<f32>,
    /// the bank: [heads * 96, hidden] and [heads * 96], float32, on the host (blended for each step)
    heads_w: Vec<f32>,
    heads_b: Vec<f32>,
    pub heads: usize,
    pub t_table: Vec<f32>,
    pub inv_freq: Vec<f32>,
}

impl Outer {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, cfg: &Config) -> Result<Outer> {
        if ck.entries.contains_key("audio_patch_proj.weight") {
            return Err(Error(format!("{}: a MiniMax H3 video denoiser (it has an audio stream); MiniMax-H3-Image's has none", ck.path.display())));
        }
        // float32 as stored, or (the Q8_0 GGUF's input projection) a quant expanded on the card
        let pe = ck.get("video_patch_proj.weight")?;
        let weight = match pe.kquant {
            Some(kind) => ops::expand(&Tensor::from_bytes(dev, DType::U8, &[pe.bytes], &ck.read(&pe.name)?)?, kind, &pe.shape, DType::F32)?,
            None => f32_tensor(dev, &pe.shape, &host(ck, "video_patch_proj.weight")?)?,
        };
        let video_patch = Linear { weight, bias: Some(small(dev, ck, "video_patch_proj.bias")?) };
        let per = LATENT_C * PATCH * PATCH;
        let heads_w = host(ck, "final_layer.video_out.weight")?;
        let heads_b = host(ck, "final_layer.video_out.bias")?;
        let heads = heads_b.len() / per;
        if video_patch.inputs() != per || video_patch.outputs() != cfg.hidden || heads * per != heads_b.len() || heads_w.len() != heads_b.len() * cfg.hidden
            || heads != PDD_SIGMAS.len() - 1
        {
            return Err(Error(format!("{}: not MiniMax-H3-Image's shapes (patch projection {:?}, {} head rows): six heads of {per} expected",
                                     ck.path.display(), video_patch.weight.shape, heads_b.len())));
        }
        let final_w = host(ck, "final_layer.adaln_proj.linear.weight")?;
        if final_w.len() != 2 * cfg.hidden * cfg.t_dim {
            return Err(Error(format!("{}: the final layer's table projection has {} values", ck.path.display(), final_w.len())));
        }
        Ok(Outer {
            video_patch,
            final_norm: small(dev, ck, "final_layer.norm.weight")?,
            final_w,
            final_b: host(ck, "final_layer.adaln_proj.linear.bias")?,
            heads_w,
            heads_b,
            heads,
            t_table: host(ck, "adaln_t_table").map_err(|e| Error(format!("{} (only the timestep-curve form of the checkpoint is read)", e.0)))?,
            inv_freq: host(ck, "rope.inv_freq")?,
        })
    }

    /// The final layer's shift and scale tables [R, hidden] for the timestep embeddings [R, t_dim]
    fn final_tables(&self, cfg: &Config, t_emb: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (c, td) = (cfg.hidden, cfg.t_dim);
        let r = t_emb.len() / td;
        let (mut shift, mut scale) = (vec![0f32; r * c], vec![0f32; r * c]);
        for ri in 0..r {
            let t = &t_emb[ri * td..(ri + 1) * td];
            for o in 0..2 * c {
                let v = self.final_b[o] + self.final_w[o * td..(o + 1) * td].iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
                if o < c {
                    shift[ri * c + o] = v;
                } else {
                    scale[ri * c + o - c] = v;
                }
            }
        }
        (shift, scale)
    }

    /// The step's head: head 0 + the weighted later heads (float32, as the reference sums them)
    pub fn head(&self, dev: &Arc<Device>, w: &[f32; 6]) -> Result<Linear> {
        let per = self.heads_b.len() / self.heads;
        let k = self.heads_w.len() / self.heads_b.len();
        let mut wsum = self.heads_w[..per * k].to_vec();
        let mut bsum = self.heads_b[..per].to_vec();
        // the later heads' weighted sum first, then added to head 0 (rows[0] + einsum(...))
        if w[1..].iter().any(|x| *x != 0.0) {
            let (mut ew, mut eb) = (vec![0f32; per * k], vec![0f32; per]);
            for (h, wh) in w.iter().enumerate().skip(1).filter(|(_, x)| **x != 0.0) {
                ew.iter_mut().zip(&self.heads_w[h * per * k..(h + 1) * per * k]).for_each(|(a, b)| *a += wh * b);
                eb.iter_mut().zip(&self.heads_b[h * per..(h + 1) * per]).for_each(|(a, b)| *a += wh * b);
            }
            wsum.iter_mut().zip(&ew).for_each(|(a, b)| *a += b);
            bsum.iter_mut().zip(&eb).for_each(|(a, b)| *a += b);
        }
        Ok(Linear { weight: f32_tensor(dev, &[per, k], &wsum)?, bias: Some(f32_tensor(dev, &[per], &bsum)?) })
    }
}

/// Where a step's time goes (NS_H3I_PROFILE): the queue drained after each stage, milliseconds summed per stage over
/// the blocks; printed once a step. The waits cost a little: the totals are for finding, not for tables.
#[derive(Default)]
pub struct Profile {
    pub stages: Vec<(&'static str, f64)>,
}

impl Profile {
    /// Adds `ms` to `stage` (kept in first-seen order)
    pub fn add(&mut self, stage: &'static str, ms: f64) {
        match self.stages.iter_mut().find(|(n, _)| *n == stage) {
            Some((_, t)) => *t += ms,
            None => self.stages.push((stage, ms)),
        }
    }

    pub fn total(&self) -> f64 {
        self.stages.iter().map(|(_, t)| t).sum()
    }

    /// One line: the total, then each stage and its share
    pub fn line(&self) -> String {
        let total = self.total().max(1e-9);
        let parts: Vec<String> = self.stages.iter().map(|(n, t)| format!("{n} {t:.0} ms ({:.0} %)", 100.0 * t / total)).collect();
        format!("{total:.0} ms: {}", parts.join(", "))
    }
}

/// What ran before a block's tap of this name (crates/h3 `dit::Blocks::block` taps after each stage, in order)
pub fn stage_of(tap: &str) -> &'static str {
    match tap {
        "h1" | "h2" => "norm",
        "qkv" => "qkv",
        "qkv_rotated" => "rope",
        "att" => "attention",
        "attn_out" => "out",
        "x1" | "x2" => "gate",
        "fc1" => "fc1",
        "mlp" => "swiglu+fc2",
        _ => "other",
    }
}

/// A step made ready (`Run::prepare`)
pub struct Prepared {
    pub sigma: f32,
    sigma_in: f32,
    ts: Timesteps,
    t_emb: Vec<f32>,
    step: Step,
}

/// One picture's network: the refined text, the layout, the edit's conditioning rows and the buffers
pub struct Run<'a> {
    pub blocks: &'a Blocks,
    pub outer: &'a Outer,
    pub layout: Layout,
    text: Tensor,
    tags: Option<Vec<i32>>,
    x: Tensor,
    scratch: Scratch,
    cond_rows: Vec<(usize, Tensor)>,
    /// the latent's height and width
    pub h: usize,
    pub w: usize,
    /// each step's time by stage, printed (NS_H3I_PROFILE)
    pub profile: bool,
    /// LoRAs on the blocks' products (the side path: `out += B (A x)` after each), if any
    pub lora: Option<&'a nextsycl_h3::lora::LoraSet>,
    /// head 0 alone at every step (NS_H3I_HEADS=first: the diffusers route, whose converted transformer keeps one head);
    /// false: the bank weighted by the grid, as ComfyUI
    pub first_head: bool,
}

impl<'a> Run<'a> {
    /// `text`: the refined text [L, hidden] (bfloat16); `tags`: per text token its modality (1 text, 0 a vision
    /// token); `refs`: the edit's pictures as normalized latents [24, h, w]; `seed`: the request's (the references'
    /// augmentation noise restarts from it for each, as the reference does)
    #[allow(clippy::too_many_arguments)]
    pub fn new(blocks: &'a Blocks, outer: &'a Outer, text: Tensor, tags: Option<Vec<i32>>, h: usize, w: usize, refs: &[(Vec<f32>, usize, usize)], seed: u64)
               -> Result<Run<'a>> {
        let cfg = &blocks.cfg;
        let dev = text.buf.device().clone();
        if text.dtype != DType::BF16 || text.shape[1] != cfg.hidden {
            return Err(Error(format!("the text tokens must be bfloat16 [L, {}], got {:?} {:?}", cfg.hidden, text.dtype, text.shape)));
        }
        if tags.as_ref().is_some_and(|t| t.len() != text.shape[0] || t.iter().any(|v| !(0..3).contains(v))) {
            return Err(Error("tags: one modality (0, 1 or 2) per text token".into()));
        }
        let blocks_of: Vec<RefBlock> = refs.iter().map(|(_, h, w)| RefBlock::Image { h: *h, w: *w }).collect();
        let layout = Layout::new(text.shape[0], 1, h, w, 0, &[], &blocks_of)?;
        let mut cond_rows = Vec::new();
        let mut segs = layout.segments.iter().filter(|s| s.kind == Kind::RefImg);
        for (lat, rh, rw) in refs {
            if lat.len() != LATENT_C * rh * rw {
                return Err(Error(format!("a reference of {} values for {LATENT_C} x {rh} x {rw}", lat.len())));
            }
            let s = segs.next().ok_or("reference rows missing from the layout")?;
            let mut rows = patchify(lat, LATENT_C, 1, *rh, *rw);
            let noise = Mt19937::new(seed).randn(rows.len());
            let a = VISUAL_COND_TIMESTEP as f32;
            rows.iter_mut().zip(noise).for_each(|(r, n)| *r = a * *r + (1.0 - a) * n);
            cond_rows.push((s.start, embed(&dev, &outer.video_patch, cfg.hidden, &rows)?));
        }
        let s = layout.tokens();
        Ok(Run { x: Tensor::new(&dev, DType::BF16, &[s, cfg.hidden])?, scratch: Scratch::new(&dev, cfg, s, DType::BF16)?, blocks, outer, layout, text, tags, cond_rows, h, w,
                 profile: false, lora: None, first_head: false })
    }

    pub fn tokens(&self) -> usize {
        self.layout.tokens()
    }

    /// Each token's table row and the distinct timesteps at noise level `sigma` (text and image at 1 - sigma, the
    /// edit's pictures at max(1 - sigma, 0.999); a vision token of the text on the video tables)
    pub fn timesteps(&self, sigma: f32) -> Timesteps {
        let mut ts = Timesteps::with_cond(&self.layout, sigma as f64, 1.0, 1.0, VISUAL_COND_TIMESTEP, 1.0);
        if let (Some(tags), Some(seg)) = (&self.tags, self.layout.segment(Kind::Text)) {
            let base = ts.rows[seg.start] - Kind::Text.modality();
            for (r, tag) in ts.rows[seg.start..seg.stop].iter_mut().zip(tags) {
                *r = base + tag;
            }
        }
        ts
    }

    /// The network at noise level `sigma` for a step to `next`: latents [24, h, w] in, the velocity out (the negated
    /// heads, as the reference returns them). `between(i)` before block i (an error stops).
    pub fn net(&mut self, latent: &[f32], sigma: f32, next: f32, between: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let p = self.prepare(sigma)?;
        self.net_prepared(p, latent, next, between)
    }

    /// What a step at noise level `sigma` fixes before the blocks run (the timestep rows, the tables of every block,
    /// the rotations): host work and small uploads (~40 ms; making the next step's on another thread while the card
    /// runs this one was tried: int8 on the B70, 6 steps at 1 / 4 / 8.5 MP, 1.83 / 6.50 / 14.55 s against 1.84 / 6.54 / 14.57 -
    /// no gain, so it is not kept)
    pub fn prepare(&self, sigma: f32) -> Result<Prepared> {
        let cfg = &self.blocks.cfg;
        let dev = self.x.buf.device();
        // the model gets sigma as timestep = sigma * 1000 and divides it again
        let sigma_in = (sigma * 1000.0) / 1000.0;
        let ts = self.timesteps(sigma_in);
        let t_emb = ts.embeddings(&self.outer.t_table, cfg.t_dim);
        let step = Step::new(dev, self.blocks, &ts.rows, &self.layout.positions, &self.outer.inv_freq, &t_emb)?;
        Ok(Prepared { sigma, sigma_in, ts, t_emb, step })
    }

    /// The network for a prepared step (`prepare(sigma)`) to `next`: as `net`
    pub fn net_prepared(&self, p: Prepared, latent: &[f32], next: f32, between: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let cfg = self.blocks.cfg;
        let o = self.outer;
        let dev = self.x.buf.device().clone();
        let Prepared { sigma, sigma_in, ts, t_emb, step } = p;
        let profiling = self.profile;
        let mut prof = Profile::default();
        if profiling {
            dev.wait()?;
        }
        let mut clock = std::time::Instant::now();
        let lap = |p: &mut Profile, c: &mut std::time::Instant, stage: &'static str| -> Result<()> {
            if profiling {
                dev.wait()?;
                p.add(stage, c.elapsed().as_secs_f64() * 1e3);
                *c = std::time::Instant::now();
            }
            Ok(())
        };
        let (tx, img) = (self.layout.segment(Kind::Text).ok_or("no text rows")?, self.layout.segment(Kind::Video).ok_or("no image rows")?);
        let n = img.stop - img.start;
        if latent.len() != LATENT_C * self.h * self.w || n * PATCH * PATCH != self.h * self.w {
            return Err(Error(format!("a latent of {} values for {LATENT_C} x {} x {}", latent.len(), self.h, self.w)));
        }
        let e = embed(&dev, &o.video_patch, cfg.hidden, &patchify(latent, LATENT_C, 1, self.h, self.w))?;
        self.x.copy_rows(tx.start, &self.text, 0, tx.stop - tx.start)?;
        self.x.copy_rows(img.start, &e, 0, n)?;
        for (start, rows) in &self.cond_rows {
            self.x.copy_rows(*start, rows, 0, rows.shape[0])?;
        }
        lap(&mut prof, &mut clock, "embed")?;
        for i in 0..self.blocks.blocks.len() {
            between(i)?;
            let lora = self.lora.and_then(|l| l.blocks.get(i));
            if profiling {
                let mut tap = |name: &str, _: &Tensor| lap(&mut prof, &mut clock, stage_of(name));
                self.blocks.block(i, &self.x, &step, &self.scratch, lora, Some(&mut tap))?;
            } else {
                self.blocks.block(i, &self.x, &step, &self.scratch, lora, None)?;
            }
        }
        let v = self.head_out(&ts, &t_emb, img.start, n, sigma_in, next)?;
        lap(&mut prof, &mut clock, "head")?;
        if profiling {
            eprintln!("profile {} tokens, sigma {sigma:.3}: {}", self.tokens(), prof.line());
        }
        Ok(v)
    }

    /// The final layer and the step's head on the image rows [start, start + n): the velocity [24, h, w]
    pub fn head_out(&self, ts: &Timesteps, t_emb: &[f32], start: usize, n: usize, sigma: f32, next: f32) -> Result<Vec<f32>> {
        let cfg = self.blocks.cfg;
        let o = self.outer;
        let dev = self.x.buf.device().clone();
        let (shift, scale) = o.final_tables(&cfg, t_emb);
        let r = ts.values.len();
        let shift = f32_tensor(&dev, &[r, cfg.hidden], &shift)?;
        let scale = f32_tensor(&dev, &[r, cfg.hidden], &scale)?;
        let head = o.head(&dev, &if self.first_head { [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] } else { head_weights(sigma, next) })?;
        let t_row = ts.video_index as i32;
        let mut out = Vec::with_capacity(n * head.outputs());
        let mut done = 0;
        while done < n {
            let m = FINAL_CHUNK.min(n - done);
            let rows = Tensor::new(&dev, DType::BF16, &[m, cfg.hidden])?;
            rows.copy_rows(0, &self.x, start + done, m)?;
            let table = Tensor::from_bytes(&dev, DType::I32, &[m], &vec![t_row; m].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            let h = Tensor::new(&dev, DType::F32, &[m, cfg.hidden])?;
            ops::rms_norm_mod(&rows, &o.final_norm, FINAL_EPS, Some(&Mod { rows: &table, scale: &scale, shift: &shift }), &h)?;
            let y = Tensor::new(&dev, DType::F32, &[m, head.outputs()])?;
            head.forward(&h, &y)?;
            out.extend(y.to_f32()?);
            done += m;
        }
        let mut v = unpatchify(&out, LATENT_C, 1, self.h, self.w);
        v.iter_mut().for_each(|x| *x = -*x);
        Ok(v)
    }

    /// The stream after the blocks (for checks): bfloat16 [tokens, hidden] as float32
    pub fn stream(&self) -> Result<Vec<f32>> {
        self.x.to_f32()
    }
}

/// Patch rows [n, 96] (float32) through the projection: bfloat16 [n, hidden]
fn embed(dev: &Arc<Device>, lin: &Linear, hidden: usize, rows: &[f32]) -> Result<Tensor> {
    let n = rows.len() / lin.inputs();
    let input = f32_tensor(dev, &[n, lin.inputs()], rows)?;
    let out = Tensor::new(dev, DType::BF16, &[n, hidden])?;
    lin.forward(&input, &out)?;
    Ok(out)
}

/// The refined text as the blocks take it, from float32 rows [L, hidden] (checks feed the reference's)
pub fn text_tensor(dev: &Arc<Device>, rows: &[f32], hidden: usize) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::BF16, &[rows.len() / hidden, hidden], &bf16_bytes(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_the_grid_a_step_takes_head_0_and_its_own() {
        assert_eq!(head_weights(1.0, PDD_SIGMAS[1]), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for i in 1..6 {
            let mut want = [0f32; 6];
            want[0] = 1.0;
            want[i] = 1.0;
            assert_eq!(head_weights(PDD_SIGMAS[i], PDD_SIGMAS[i + 1]), want, "step {i}");
        }
    }

    #[test]
    fn a_long_step_shares_its_heads_by_length() {
        // 3 steps: 1 -> grid[2] -> grid[4] -> 0. Step 0 spans intervals 0 and 1: head 0 at 1, head 1 by its share
        let s = sigmas(3);
        assert_eq!(s, vec![1.0, PDD_SIGMAS[2], PDD_SIGMAS[4], 0.0]);
        let w = head_weights(s[0], s[1]);
        let share = (PDD_SIGMAS[1] - PDD_SIGMAS[2]) / (1.0 - PDD_SIGMAS[2]);
        assert_eq!(w[0], 1.0);
        assert!((w[1] - share).abs() < 1e-6 && w[2..].iter().all(|x| *x == 0.0));
        let w = head_weights(s[1], s[2]);
        assert!((w[2] + w[3] - 1.0).abs() < 1e-6 && w[1] == 0.0 && w[4] == 0.0);
    }

    #[test]
    fn the_grid_is_the_heads_and_other_counts_read_it_linearly() {
        assert_eq!(sigmas(6), PDD_SIGMAS.to_vec());
        let s12 = sigmas(12);
        assert_eq!(s12.len(), 13);
        assert_eq!((s12[0], s12[12]), (1.0, 0.0));
        assert!((s12[1] - (1.0 + PDD_SIGMAS[1]) / 2.0).abs() < 1e-6);
        assert!(s12.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn the_shifted_schedule_is_comfyuis_simple_one() {
        // pDMD's 2 steps at shift 12: 1, sigma(t = 0.5) = 6 / 6.5, 0
        let s = shifted_sigmas(12.0, 2);
        assert_eq!(s.len(), 3);
        assert_eq!((s[0], s[2]), (1.0, 0.0));
        assert!((s[1] - 12.0 * 0.5 / 6.5).abs() < 1e-6, "{s:?}");
        // 3 steps: simple reads index 999 - floor(x * 333.33): t = 1, 0.667, 0.334 (not 2/3, 1/3)
        let s = shifted_sigmas(12.0, 3);
        let at = |t: f32| 12.0 * t / (1.0 + 11.0 * t);
        assert!((s[1] - at(0.667)).abs() < 1e-6 && (s[2] - at(0.334)).abs() < 1e-6, "{s:?}");
        // shift 1 is the plain table
        assert_eq!(shifted_sigmas(1.0, 4), vec![1.0, 0.75, 0.5, 0.25, 0.0]);
        for n in 1..=8 {
            let s = shifted_sigmas(12.0, n);
            assert!(s.len() == n + 1 && s.windows(2).all(|w| w[1] < w[0]));
        }
    }

    #[test]
    fn a_denoise_below_1_starts_partway_down_a_longer_schedule() {
        // ComfyUI's BasicScheduler(simple, 3, denoise 0.5) at shift 12: the last 4 points of the 6-step schedule
        let s = schedule(3, Some(12.0), 0.5);
        let full = shifted_sigmas(12.0, 6);
        assert_eq!(s, full[3..].to_vec());
        assert!((s[0] - 12.0 * 0.5 / 6.5).abs() < 1e-6, "denoise 0.5 starts at sigma 0.923 under shift 12: {s:?}");
        // 3 / 0.6 is 5 in Python's doubles; float32's 0.6 would give 4
        assert_eq!(schedule(3, Some(12.0), 0.6), shifted_sigmas(12.0, 5)[2..].to_vec());
        // no shift: the heads' grid; denoise 1: the whole schedule
        assert_eq!(schedule(6, None, 1.0), PDD_SIGMAS.to_vec());
        assert_eq!(schedule(2, Some(12.0), 1.0), shifted_sigmas(12.0, 2));
        assert_eq!(schedule(2, None, 0.5), sigmas(4)[2..].to_vec());
    }

    #[test]
    fn an_img2img_runs_down_from_its_start_in_the_shifts_time() {
        let g = shifted_from(0.55, 2, 12.0);
        assert_eq!(g.len(), 3);
        assert!((g[0] - 0.55).abs() < 1e-6 && (g[1] - 0.367).abs() < 1e-3 && g[2] == 0.0, "{g:?}");
        // from 1 it is the 2-step schedule itself
        let g = shifted_from(1.0, 2, 12.0);
        assert!((g[0] - 1.0).abs() < 1e-6 && (g[1] - 12.0 * 0.5 / 6.5).abs() < 1e-6);
        assert!(shifted_from(0.3, 4, 12.0).windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn the_spaces_detail_grid_starts_at_sigma0() {
        let g = start_grid(0.5, &shifted_sigmas(12.0, 3));
        assert_eq!(g.len(), 4);
        assert!((g[0] - 0.5).abs() < 1e-7 && g[3] == 0.0);
        assert!((g[1] / g[0] - 0.857_143 / 0.909_375).abs() < 1e-6 && (g[2] / g[0] - 0.666_667 / 0.909_375).abs() < 1e-6);
        let g = start_grid(0.4, &shifted_sigmas(12.0, 2));
        assert_eq!(g.len(), 3);
        assert!((g[0] - 0.4).abs() < 1e-7 && (g[1] - 0.4 * 12.0 * 0.5 / 6.5).abs() < 1e-6 && g[2] == 0.0);
    }

    #[test]
    fn the_profile_sums_stages_in_order() {
        let mut p = Profile::default();
        for tap in ["h1", "qkv", "qkv_rotated", "att", "attn_out", "x1", "h2", "fc1", "mlp", "x2"] {
            p.add(stage_of(tap), 1.0);
        }
        let names: Vec<&str> = p.stages.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["norm", "qkv", "rope", "attention", "out", "gate", "fc1", "swiglu+fc2"]);
        assert_eq!(p.stages[0].1, 2.0);
        assert_eq!(p.total(), 10.0);
        assert!(p.line().starts_with("10 ms: norm 2 ms (20 %)"));
        assert_eq!(stage_of("something else"), "other");
    }

    #[test]
    fn euler_moves_along_the_velocity() {
        // x_next = x + (next - sigma) (x - (x - v sigma)) / sigma = x + (next - sigma) v
        let mut x = vec![1.0f32, -2.0];
        euler(&mut x, &[0.5, 1.0], 0.75, 0.5);
        assert!((x[0] - (1.0 - 0.25 * 0.5)).abs() < 1e-6 && (x[1] - (-2.0 - 0.25)).abs() < 1e-6);
        // the last step lands on the denoised estimate x - v sigma
        let mut y = vec![0.3f32];
        euler(&mut y, &[0.2], 0.232_008_87, 0.0);
        assert!((y[0] - (0.3 - 0.2 * 0.232_008_87)).abs() < 1e-6);
    }
}
