//! Everything around the block stack, and the sampler: from the text conditioning and a starting noise to the
//! finished latents (what the video and audio decoders then turn into pixels and sound).
//!
//! One call of the network (`Denoiser::net`), on video latents [24, T, H, W] and audio latents [32, 2, T_audio]:
//!
//! ```text
//!   video rows = 2x2 patches of the video latent          [T * H/2 * W/2, 96]   (host)
//!   audio rows = one row per stereo channel and frame     [2 * T_audio, 32]     (host)
//!   x = [ text | audio rows . W_audio | video rows . W_video ]                  one sequence, bfloat16
//!   x = 50 blocks (dit.rs)
//!   out = linear(norm(x) * (1 + scale[t]) + shift[t])  on the audio rows and on the video rows, float32
//! ```
//!
//! The text comes from the text encoder (5120 features a token) through a projection and a two-block "refiner" -
//! the same kind of block without the timestep tables or the position rotation - once per clip (`TextRefiner`).
//!
//! The sampler (`sample`) is Euler over a fixed list of noise levels (sigmas, 1 down to 0): the network predicts a
//! velocity, `denoised = x - velocity * sigma`, and each step moves x to the next noise level along it. The audio
//! stream runs on a schedule of its own (less noisy than the video at the same step); the sampler carries it
//! rescaled so one sigma serves both, and the wrapper below converts on the way in and out - this follows the
//! reference pipeline (ComfyUI's MiniMax H3 model) exactly, because the network was trained on it.

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::{host, small, Blocks, Config, Scratch, Step};
use crate::dtype::{f32_to_bf16, DType};
use crate::layout::{time_shift_sigma, Kind, Layout, Timesteps};
use crate::ops::{self, Linear, Mod, Rows};
use crate::safetensors::Checkpoint;
use crate::{load, Ctx, Error, Result};

/// Video latent features per 2x2 patch row, and audio latent features per row.
const PATCH: usize = 2;

/// The final layer runs over the streams' rows this many at a time, so its float32 intermediates stay small.
const FINAL_CHUNK: usize = 4096;

/// A weight as float32, on the device: the int8 checkpoint stores these in float32; a GGUF one in half, or as a
/// k-quant (expanded on the device).
fn f32_weight(dev: &Arc<Device>, ck: &Checkpoint, name: &str) -> Result<Tensor> {
    let e = ck.get(name)?;
    let bytes = ck.read(name)?;
    match e.kquant {
        Some(kind) => ops::expand(&Tensor::from_bytes(dev, DType::U8, &[e.bytes], &bytes)?, kind, &e.shape, DType::F32),
        None => Tensor::from_bytes(dev, DType::F32, &e.shape, &f32_bytes(&crate::dtype::bytes_to_f32(&bytes, e.dtype)?)),
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f32_to_bf16(*f).to_le_bytes()).collect()
}

/// The small parts around the blocks: the patch projections in, the final layer out, the timestep curve and the
/// rotation frequencies. A few MiB; loaded with the blocks and kept.
pub struct Outer {
    video_patch: Linear,
    audio_patch: Linear,
    final_norm: Tensor,
    /// [2 * hidden, t_dim] and its bias: shift, then scale
    final_w: Vec<f32>,
    final_b: Vec<f32>,
    video_out: Linear,
    audio_out: Linear,
    /// [grid, t_dim]: the timestep embedding curve
    pub t_table: Vec<f32>,
    pub inv_freq: Vec<f32>,
    pub video_c: usize,
    pub audio_c: usize,
    pub final_eps: f32,
}

impl Outer {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<Outer> {
        let linear = |name: &str| -> Result<Linear> {
            Ok(Linear { weight: f32_weight(dev, ck, &format!("{name}.weight"))?, bias: Some(small(dev, ck, &format!("{name}.bias"))?) })
        };
        let video_patch = linear("video_patch_proj")?;
        let audio_patch = linear("audio_patch_proj")?;
        if video_patch.weight.dtype != DType::F32 || audio_patch.weight.dtype != DType::F32 {
            return Err(Error("the patch projections are expected in float32".into()));
        }
        let video_c = video_patch.inputs() / (PATCH * PATCH);
        Ok(Outer {
            video_c,
            audio_c: audio_patch.inputs(),
            video_patch,
            audio_patch,
            final_norm: small(dev, ck, "final_layer.norm.weight")?,
            final_w: host(ck, "final_layer.adaln_proj.linear.weight")?,
            final_b: host(ck, "final_layer.adaln_proj.linear.bias")?,
            video_out: linear("final_layer.video_out")?,
            audio_out: linear("final_layer.audio_out")?,
            t_table: host(ck, "adaln_t_table").ctx("only the timestep-curve form of the checkpoint is supported")?,
            inv_freq: host(ck, "rope.inv_freq")?,
            final_eps: 1e-5,
        })
    }

    /// The final layer's shift and scale tables [R, hidden] for the timestep embeddings [R, t_dim].
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
}

struct RefinerBlock {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: Linear,
    out_proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// The text side: projection to the model's width, then two plain blocks over the text tokens alone. About 1.6 GiB
/// of 16-bit weights for a step that runs once per clip, so it is loaded for that and dropped.
pub struct TextRefiner {
    proj: Linear,
    blocks: Vec<RefinerBlock>,
    final_norm: Tensor,
}

impl TextRefiner {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint, threads: usize) -> Result<TextRefiner> {
        const LIN: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];
        let wanted = |name: &str| {
            name == "condition_proj.weight"
                || name.strip_prefix("token_refiner.blocks.").and_then(|r| r.split_once('.')).is_some_and(|(_, t)| LIN.iter().any(|l| t == format!("{l}.weight")))
        };
        let mut big = load::load(dev, ck, threads, wanted).ctx("loading the text refiner")?.tensors;
        // a GGUF checkpoint's k-quant matrices: expanded to bfloat16 here (the int8 checkpoint stores them so)
        for (name, t) in big.iter_mut() {
            let e = ck.get(name)?;
            if let Some(kind) = e.kquant {
                *t = ops::expand(t, kind, &e.shape, DType::BF16)?;
            } else if e.dtype != DType::BF16 {
                return Err(Error(format!("{name}: the text refiner's matrices are expected in bfloat16 or a k-quant, not {:?}", e.dtype)));
            }
        }
        let mut take = |k: String| big.remove(&k).ok_or_else(|| Error(format!("{k} was not loaded")));
        let proj = Linear { weight: take("condition_proj.weight".into())?, bias: Some(small(dev, ck, "condition_proj.bias")?) };
        let n = ck.entries.keys().filter_map(|k| k.strip_prefix("token_refiner.blocks.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("token_refiner.blocks.{i}");
            let mut lin = |l: &str| -> Result<Linear> { Ok(Linear { weight: take(format!("{p}.{l}.weight"))?, bias: None }) };
            blocks.push(RefinerBlock {
                qkv: lin(LIN[0])?,
                out_proj: lin(LIN[1])?,
                fc1: lin(LIN[2])?,
                fc2: lin(LIN[3])?,
                norm1: small(dev, ck, &format!("{p}.norm1.weight"))?,
                norm2: small(dev, ck, &format!("{p}.norm2.weight"))?,
                q_norm: small(dev, ck, &format!("{p}.attn.q_norm.weight"))?,
                k_norm: small(dev, ck, &format!("{p}.attn.k_norm.weight"))?,
            });
        }
        Ok(TextRefiner { proj, blocks, final_norm: small(dev, ck, "token_refiner.final_norm.weight")? })
    }

    /// The text encoder's states [L, text_dim] (in the weights' type) -> the text tokens [L, hidden]. `lora`: LoRAs on
    /// the refiner's blocks, if any.
    pub fn run(&self, cfg: &Config, context: &Tensor, lora: Option<&[crate::lora::BlockLora]>) -> Result<Tensor> {
        let dev = context.buf.device();
        let l = context.shape[0];
        let dt = self.proj.weight.dtype;
        let x = Tensor::new(dev, dt, &[l, cfg.hidden])?;
        self.proj.forward(context, &x)?;
        let w = cfg.heads * cfg.head_dim;
        let s = Scratch::new(dev, cfg, l, dt)?;
        let no_rotation = Tensor::new(dev, DType::F32, &[1])?;
        for (bi, b) in self.blocks.iter().enumerate() {
            let lo = lora.and_then(|l| l.get(bi));
            ops::rms_norm_mod(&x, &b.norm1, cfg.norm_eps, None, &s.h)?;
            b.qkv.forward(&s.h, &s.qkv)?;
            crate::dit::side(lo, 0, &s.h, &s.qkv, &s.lora)?;
            let part = |i: usize| Rows { t: &s.qkv, offset: i * w, stride: 3 * w, tokens: l, heads: cfg.heads, dim: cfg.head_dim };
            ops::rms_rope(part(0), &b.q_norm, cfg.qk_eps, &no_rotation, 0)?;
            ops::rms_rope(part(1), &b.k_norm, cfg.qk_eps, &no_rotation, 0)?;
            ops::attention(part(0), part(1), part(2), &s.att)?;
            b.out_proj.forward(&s.att, &s.proj)?;
            crate::dit::side(lo, 1, &s.att, &s.proj, &s.lora)?;
            ops::add(&x, &s.proj)?;
            ops::rms_norm_mod(&x, &b.norm2, cfg.norm_eps, None, &s.h)?;
            b.fc1.forward(&s.h, &s.fc1)?;
            crate::dit::side(lo, 2, &s.h, &s.fc1, &s.lora)?;
            ops::swiglu(&s.fc1, &s.act)?;
            b.fc2.forward(&s.act, &s.proj)?;
            crate::dit::side(lo, 3, &s.act, &s.proj, &s.lora)?;
            ops::add(&x, &s.proj)?;
        }
        ops::rms_norm_mod(&x, &self.final_norm, cfg.norm_eps, None, &x)?;
        Ok(x)
    }
}

/// Video latent [C, T, H, W] -> rows [T * H/2 * W/2, C * 4]: one row per 2x2 patch, features (c, dy, dx).
pub fn patchify(v: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / PATCH, w / PATCH);
    let mut out = vec![0f32; v.len()];
    for ti in 0..t {
        for yi in 0..hp {
            for xi in 0..wp {
                let row = (ti * hp + yi) * wp + xi;
                for ci in 0..c {
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            out[row * c * 4 + ci * 4 + dy * 2 + dx] = v[((ci * t + ti) * h + yi * 2 + dy) * w + xi * 2 + dx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `patchify`.
pub fn unpatchify(rows: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (hp, wp) = (h / PATCH, w / PATCH);
    let mut out = vec![0f32; rows.len()];
    for ti in 0..t {
        for yi in 0..hp {
            for xi in 0..wp {
                let row = (ti * hp + yi) * wp + xi;
                for ci in 0..c {
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            out[((ci * t + ti) * h + yi * 2 + dy) * w + xi * 2 + dx] = rows[row * c * 4 + ci * 4 + dy * 2 + dx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Audio latent [C, 2, T] -> rows [2 * T, C], channel-major (every frame of the left channel, then the right).
pub fn pack_audio(a: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; a.len()];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ch * t + ti) * c + ci] = a[(ci * 2 + ch) * t + ti];
            }
        }
    }
    out
}

/// The inverse of `pack_audio`.
pub fn unpack_audio(rows: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows.len()];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ci * 2 + ch) * t + ti] = rows[(ch * t + ti) * c + ci];
            }
        }
    }
    out
}

/// The sizes of one clip's latents.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// video latent frames, height, width
    pub t: usize,
    pub h: usize,
    pub w: usize,
    /// audio latent frames
    pub audio_t: usize,
}

/// The noise schedule's two shifts (video, audio) and how the sampler carries the audio.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub shift_video: f32,
    pub shift_audio: f32,
    pub audio_scale: f32,
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule { shift_video: 12.0, shift_audio: 3.0, audio_scale: 4.0 }
    }
}

/// The noise levels of a run of `steps` steps, 1 down to 0 ("simple" schedule over the shifted 1000-level table).
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let table: Vec<f32> = (1..=1000).map(|i| {
        let t = i as f32 / 1000.0;
        shift * t / (1.0 + (shift - 1.0) * t)
    }).collect();
    let stride = table.len() as f64 / steps as f64;
    let mut s: Vec<f32> = (0..steps).map(|x| table[table.len() - 1 - (x as f64 * stride) as usize]).collect();
    s.push(0.0);
    s
}

/// Called before each block with its index (cancel checks, progress).
pub type Between<'a> = &'a mut dyn FnMut(usize) -> Result<()>;

/// A keyframe: latents pinned at a pixel frame of the clip, presented to every step and never denoised. A video
/// keyframe is one latent frame (a picture, or a previous clip's last latent) or several (a moving guide: the
/// last frames of a previous clip); an audio keyframe is a stretch of sound (a previous clip's room tone and
/// voice tail).
pub struct KeyframeIn {
    /// pixel frame of the clip the keyframe starts at (24 fps)
    pub frame_index: usize,
    /// normalized video latents [24, vt, H, W] and vt (the clip's H and W)
    pub video: Option<(Vec<f32>, usize)>,
    /// normalized audio latents [32, 2, rt] and rt
    pub audio: Option<(Vec<f32>, usize)>,
}

/// What a clip is conditioned on besides its text: keyframes, and how much the denoiser is told to trust them.
pub struct Conditions {
    pub keyframes: Vec<KeyframeIn>,
    /// a keyframe's video rows are mixed `aug * latent + (1 - aug) * noise` and presented at timestep
    /// max(t, aug) (0.999 by default; lower: trust a degraded anchor less)
    pub visual_aug: f32,
    /// the same for audio rows (1.0: no noise)
    pub audio_aug: f32,
    /// the sampling seed (the augmentation noise is drawn from it, as the reference does)
    pub seed: u64,
    /// the references (ref2va), in the order the prompt presents them
    pub refs: Vec<RefIn>,
}

/// A reference's normalized latents
pub enum RefIn {
    /// a sound (a voice): [32, 2, t]
    Audio { latent: Vec<f32>, t: usize },
    /// a picture: [24, 1, h, w] at its own size
    Image { latent: Vec<f32>, h: usize, w: usize },
    /// a clip: [24, t, h, w], and its sound [32, 2, rt] when it brings one
    Video { latent: Vec<f32>, t: usize, h: usize, w: usize, audio: Option<(Vec<f32>, usize)> },
}

impl RefIn {
    fn block(&self) -> crate::layout::RefBlock {
        use crate::layout::RefBlock;
        match self {
            RefIn::Audio { t, .. } => RefBlock::Audio { t: *t },
            RefIn::Image { h, w, .. } => RefBlock::Image { h: *h, w: *w },
            RefIn::Video { t, h, w, audio, .. } => RefBlock::Video { t: *t, h: *h, w: *w, audio_t: audio.as_ref().map_or(0, |a| a.1) },
        }
    }
}

impl Default for Conditions {
    fn default() -> Self {
        Conditions {
            keyframes: Vec::new(),
            visual_aug: crate::layout::VISUAL_COND_TIMESTEP as f32,
            audio_aug: crate::layout::AUDIO_COND_TIMESTEP as f32,
            seed: 0,
            refs: Vec::new(),
        }
    }
}

/// One clip's network: the text tokens, the token layout and the buffers, for the block stack loaded in `blocks`.
pub struct Denoiser<'a> {
    pub blocks: &'a Blocks,
    pub outer: &'a Outer,
    pub shape: Shape,
    pub layout: Layout,
    pub schedule: Schedule,
    text: Tensor,
    text_tags: Option<Vec<i32>>,
    x: Tensor,
    scratch: Scratch,
    /// LoRAs on the blocks, if any
    pub lora: Option<&'a crate::lora::LoraSet>,
    /// the keyframes' rows, embedded once: (first row in the sequence, rows)
    cond_rows: Vec<(usize, Tensor)>,
    /// the timesteps conditioning rows are presented at
    cond_t: (f64, f64),
    /// a masked run's token masks
    pub masks: Option<RowMasks>,
}

impl<'a> Denoiser<'a> {
    /// `text`: the refined text tokens [L, hidden]; `text_tags`: per text token, the modality whose tables it uses
    /// (1 = text; the reference marks some tokens otherwise), or `None` for all text.
    pub fn new(blocks: &'a Blocks, outer: &'a Outer, text: Tensor, text_tags: Option<Vec<i32>>, shape: Shape, schedule: Schedule, cond: &Conditions) -> Result<Denoiser<'a>> {
        let cfg = &blocks.cfg;
        let dev = text.buf.device().clone();
        let kfs: Vec<crate::layout::Keyframe> = cond
            .keyframes
            .iter()
            .map(|k| crate::layout::Keyframe { frame_index: k.frame_index as f64, video_latent_t: k.video.as_ref().map(|v| v.1), audio_latent_t: k.audio.as_ref().map(|a| a.1) })
            .collect();
        let refs: Vec<crate::layout::RefBlock> = cond.refs.iter().map(RefIn::block).collect();
        let layout = Layout::new(text.shape[0], shape.t, shape.h, shape.w, shape.audio_t, &kfs, &refs)?;
        // the keyframes' rows: patches (or stereo rows) through the projections once, the augmentation noise drawn
        // from the seed afresh for every keyframe (the reference restarts the same stream each time; audio rows
        // from seed + 1)
        let mut cond_rows = Vec::new();
        {
            let embed = |rows: Vec<f32>, lin: &Linear| -> Result<Tensor> {
                let n = rows.len() / lin.inputs();
                let input = Tensor::from_bytes(&dev, DType::F32, &[n, lin.inputs()], &f32_bytes(&rows))?;
                let out = Tensor::new(&dev, DType::BF16, &[n, cfg.hidden])?;
                lin.forward(&input, &out)?;
                Ok(out)
            };
            let aug = |mut rows: Vec<f32>, a: f32, seed: u64| -> Vec<f32> {
                if a < 1.0 {
                    let noise = crate::noise::Mt19937::new(seed).randn(rows.len());
                    rows.iter_mut().zip(noise).for_each(|(r, n)| *r = a * *r + (1.0 - a) * n);
                }
                rows
            };
            let mut segs = layout.segments.iter().filter(|s| matches!(s.kind, Kind::Cond | Kind::CondAudio | Kind::RefAudio | Kind::RefImg));
            for k in &cond.keyframes {
                if let Some((v, vt)) = &k.video {
                    if v.len() != outer.video_c * vt * shape.h * shape.w {
                        return Err(Error(format!("a video keyframe of {} values for {} x {vt} x {} x {}", v.len(), outer.video_c, shape.h, shape.w)));
                    }
                    let s = segs.next().ok_or("keyframe rows missing from the layout")?;
                    let rows = aug(patchify(v, outer.video_c, *vt, shape.h, shape.w), cond.visual_aug, cond.seed);
                    cond_rows.push((s.start, embed(rows, &outer.video_patch)?));
                }
                if let Some((a, rt)) = &k.audio {
                    if a.len() != outer.audio_c * 2 * rt {
                        return Err(Error(format!("an audio keyframe of {} values for {} x 2 x {rt}", a.len(), outer.audio_c)));
                    }
                    let s = segs.next().ok_or("keyframe rows missing from the layout")?;
                    let rows = aug(pack_audio(a, outer.audio_c, *rt), cond.audio_aug, cond.seed + 1);
                    cond_rows.push((s.start, embed(rows, &outer.audio_patch)?));
                }
            }
            // the references in order (a clip's sound before its pictures), each with the same augmentation
            let sound = |a: &[f32], rt: usize, s: &crate::layout::Segment| -> Result<(usize, Tensor)> {
                if a.len() != outer.audio_c * 2 * rt {
                    return Err(Error(format!("a reference sound of {} values for {} x 2 x {rt}", a.len(), outer.audio_c)));
                }
                Ok((s.start, embed(aug(pack_audio(a, outer.audio_c, rt), cond.audio_aug, cond.seed + 1), &outer.audio_patch)?))
            };
            let picture = |v: &[f32], (t, h, w): (usize, usize, usize), s: &crate::layout::Segment| -> Result<(usize, Tensor)> {
                if v.len() != outer.video_c * t * h * w {
                    return Err(Error(format!("a reference of {} values for {} x {t} x {h} x {w}", v.len(), outer.video_c)));
                }
                Ok((s.start, embed(aug(patchify(v, outer.video_c, t, h, w), cond.visual_aug, cond.seed), &outer.video_patch)?))
            };
            for r in &cond.refs {
                let mut next = || segs.next().ok_or_else(|| Error("reference rows missing from the layout".into()));
                match r {
                    RefIn::Audio { latent, t } if *t > 0 => cond_rows.push(sound(latent, *t, next()?)?),
                    RefIn::Audio { .. } => {}
                    RefIn::Image { latent, h, w } => cond_rows.push(picture(latent, (1, *h, *w), next()?)?),
                    RefIn::Video { latent, t, h, w, audio } => {
                        if let Some((a, rt)) = audio.as_ref().filter(|a| a.1 > 0) {
                            cond_rows.push(sound(a, *rt, next()?)?);
                        }
                        cond_rows.push(picture(latent, (*t, *h, *w), next()?)?);
                    }
                }
            }
        }
        if text.dtype != DType::BF16 || text.shape[1] != cfg.hidden {
            return Err(Error(format!("the text tokens must be bfloat16 [L, {}], got {:?} {:?}", cfg.hidden, text.dtype, text.shape)));
        }
        if text_tags.as_ref().is_some_and(|t| t.len() != text.shape[0] || t.iter().any(|v| !(0..3).contains(v))) {
            return Err(Error("text_tags: one modality (0, 1 or 2) per text token".into()));
        }
        let s = layout.tokens();
        Ok(Denoiser {
            x: Tensor::new(&dev, DType::BF16, &[s, cfg.hidden])?,
            scratch: Scratch::new(&dev, cfg, s, DType::BF16)?,
            blocks,
            outer,
            shape,
            layout,
            schedule,
            text,
            text_tags,
            lora: None,
            cond_rows,
            cond_t: (cond.visual_aug as f64, cond.audio_aug as f64),
            masks: None,
        })
    }

    pub fn tokens(&self) -> usize {
        self.layout.tokens()
    }

    /// Each token's table row and the distinct timesteps at video noise level `sigma`, and per target row (video,
    /// audio) the timestep index the final layer uses.
    fn timesteps(&self, sigma: f32) -> (Timesteps, Vec<i32>, Vec<i32>) {
        let mut ts = Timesteps::with_cond(&self.layout, sigma as f64, self.schedule.shift_video as f64, self.schedule.shift_audio as f64, self.cond_t.0, self.cond_t.1);
        if let (Some(tags), Some(seg)) = (&self.text_tags, self.layout.segment(Kind::Text)) {
            let base = ts.rows[seg.start] - Kind::Text.modality();
            for (r, tag) in ts.rows[seg.start..seg.stop].iter_mut().zip(tags) {
                *r = base + tag;
            }
        }
        let (sv, sa) = (self.layout.segment(Kind::Video).expect("video rows"), self.layout.segment(Kind::Audio).expect("audio rows"));
        let mut fv = vec![ts.video_index as i32; sv.stop - sv.start];
        let mut fa = vec![ts.audio_index as i32; sa.stop - sa.start];
        let Some(m) = &self.masks else { return (ts, fv, fa) };
        // masked rows run at their own timestep: 1 - m * sigma, at most the conditioning timestep (in float32)
        let sigma_v = sigma.max(1e-6);
        let (t_v, t_a) = (ts.values[ts.video_index] as f32, ts.values[ts.audio_index] as f32);
        let pin_v = (t_v as f64).max(self.cond_t.0) as f32;
        let pin_a = (t_a as f64).max(self.cond_t.1) as f32;
        let sigma_a = 1.0 - t_a;
        let rv: Vec<f32> = m.video_rows.iter().map(|m| (1.0 - m * sigma_v).min(pin_v)).collect();
        let ra: Vec<f32> = m.audio_rows.iter().map(|m| (1.0 - m * sigma_a).min(pin_a)).collect();
        let video_masked = m.video_rows.iter().any(|v| *v < 1.0 - 1e-3);
        let audio_masked = m.audio_rows.iter().any(|v| *v < 1.0 - 1e-3);
        let mut values = ts.values.clone();
        if video_masked {
            values.extend(rv.iter().map(|v| *v as f64));
        }
        if audio_masked {
            values.extend(ra.iter().map(|v| *v as f64));
        }
        values.sort_by(f64::total_cmp);
        values.dedup();
        let index = |t: f64| values.iter().position(|v| *v == t).expect("every timestep is in the list") as i32;
        for r in ts.rows.iter_mut() {
            let (old, modality) = (*r / 3, *r % 3);
            *r = index(ts.values[old as usize]) * 3 + modality;
        }
        if video_masked {
            for (i, t) in rv.iter().enumerate() {
                let k = index(*t as f64);
                ts.rows[sv.start + i] = k * 3 + Kind::Video.modality();
                fv[i] = k;
            }
        } else {
            fv.iter_mut().for_each(|f| *f = index(ts.values[ts.video_index]));
        }
        if audio_masked {
            for (i, t) in ra.iter().enumerate() {
                let k = index(*t as f64);
                ts.rows[sa.start + i] = k * 3 + Kind::Audio.modality();
                fa[i] = k;
            }
        } else {
            fa.iter_mut().for_each(|f| *f = index(ts.values[ts.audio_index]));
        }
        ts.video_index = index(ts.values[ts.video_index]) as usize;
        ts.audio_index = index(ts.values[ts.audio_index]) as usize;
        ts.values = values;
        (ts, fv, fa)
    }

    /// The network at video noise level `sigma`: latents in, velocities out (video [C, T, H, W], audio [C, 2, T]),
    /// with the reference's sign (it returns the negated head outputs).
    pub fn net(&mut self, video: &[f32], audio: &[f32], sigma: f32, between: Between) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = self.blocks.cfg;
        let (o, sh) = (self.outer, self.shape);
        let dev = self.x.buf.device().clone();
        let (ts, final_v, final_a) = self.timesteps(sigma);
        let t_emb = ts.embeddings(&o.t_table, cfg.t_dim);
        let step = Step::new(&dev, self.blocks, &ts.rows, &self.layout.positions, &o.inv_freq, &t_emb)?;

        // embed: the patch projections write bfloat16 rows, copied into place after the text
        let seg = |k: Kind| self.layout.segment(k).ok_or_else(|| Error(format!("no {k:?} rows in the layout")));
        let (ts_text, ts_audio, ts_video) = (seg(Kind::Text)?, seg(Kind::Audio)?, seg(Kind::Video)?);
        let embed = |rows: Vec<f32>, n: usize, lin: &Linear| -> Result<Tensor> {
            let input = Tensor::from_bytes(&dev, DType::F32, &[n, lin.inputs()], &f32_bytes(&rows))?;
            let out = Tensor::new(&dev, DType::BF16, &[n, cfg.hidden])?;
            lin.forward(&input, &out)?;
            Ok(out)
        };
        let nv = ts_video.stop - ts_video.start;
        let na = ts_audio.stop - ts_audio.start;
        let ve = embed(patchify(video, o.video_c, sh.t, sh.h, sh.w), nv, &o.video_patch)?;
        let ae = embed(pack_audio(audio, o.audio_c, sh.audio_t), na, &o.audio_patch)?;
        self.x.copy_rows(ts_text.start, &self.text, 0, ts_text.stop - ts_text.start)?;
        self.x.copy_rows(ts_audio.start, &ae, 0, na)?;
        self.x.copy_rows(ts_video.start, &ve, 0, nv)?;
        for (start, rows) in &self.cond_rows {
            self.x.copy_rows(*start, rows, 0, rows.shape[0])?;
        }

        for i in 0..self.blocks.blocks.len() {
            between(i)?;
            let lora = self.lora.and_then(|l| l.blocks.get(i));
            self.blocks.block(i, &self.x, &step, &self.scratch, lora, None)?;
        }

        // final layer, per stream, in chunks of rows
        let (shift, scale) = o.final_tables(&cfg, &t_emb);
        let r = ts.values.len();
        let shift = Tensor::from_bytes(&dev, DType::F32, &[r, cfg.hidden], &f32_bytes(&shift))?;
        let scale = Tensor::from_bytes(&dev, DType::F32, &[r, cfg.hidden], &f32_bytes(&scale))?;
        let head = |start: usize, n: usize, t_rows: &[i32], lin: &Linear| -> Result<Vec<f32>> {
            let mut out = Vec::with_capacity(n * lin.outputs());
            let mut done = 0;
            while done < n {
                let m = FINAL_CHUNK.min(n - done);
                let rows = Tensor::new(&dev, DType::BF16, &[m, cfg.hidden])?;
                rows.copy_rows(0, &self.x, start + done, m)?;
                let table = Tensor::from_bytes(&dev, DType::I32, &[m], &t_rows[done..done + m].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let h = Tensor::new(&dev, DType::F32, &[m, cfg.hidden])?;
                ops::rms_norm_mod(&rows, &o.final_norm, o.final_eps, Some(&Mod { rows: &table, scale: &scale, shift: &shift }), &h)?;
                let y = Tensor::new(&dev, DType::F32, &[m, lin.outputs()])?;
                lin.forward(&h, &y)?;
                out.extend(y.to_f32()?);
                done += m;
            }
            Ok(out)
        };
        let v = head(ts_video.start, nv, &final_v, &o.video_out)?;
        let a = head(ts_audio.start, na, &final_a, &o.audio_out)?;
        let mut v = unpatchify(&v, o.video_c, sh.t, sh.h, sh.w);
        let mut a = unpack_audio(&a, o.audio_c, sh.audio_t);
        v.iter_mut().chain(a.iter_mut()).for_each(|x| *x = -*x);
        Ok((v, a))
    }

    /// The model as the sampler sees it: x (video, and audio as the sampler carries it) at `sigma` -> the denoised
    /// estimates, in the same spaces.
    pub fn denoised(&mut self, xv: &[f32], xa: &[f32], sigma: f32, between: Between) -> Result<(Vec<f32>, Vec<f32>)> {
        let sc = self.schedule;
        // the model receives sigma as timestep = sigma * 1000 and divides again, in float32
        let sigma_v = (sigma * 1000.0 / 1000.0).max(1e-6);
        let sigma_a = time_shift_sigma(sigma_v as f64, sc.shift_video as f64, sc.shift_audio as f64) as f32;
        let carry = sigma_a / sigma_v;
        let audio_in: Vec<f32> = xa.iter().map(|x| x * carry).collect();
        let (mut out_v, mut out_a) = self.net(xv, &audio_in, sigma, between)?;
        if let Some(m) = &self.masks {
            // masked rows predict at mask * sigma: their velocity scaled to match the outer conversion
            let n = m.video_px.len();
            out_v.iter_mut().enumerate().for_each(|(i, v)| *v *= m.video_px[i % n]);
            let at = m.audio_px.len();
            out_a.iter_mut().enumerate().for_each(|(i, v)| *v *= m.audio_px[i % at]);
        }
        let k = 1.0 + (sc.audio_scale - 1.0) * sigma_a;
        let dv = xv.iter().zip(&out_v).map(|(x, o)| x - o * sigma).collect();
        let da = xa.iter().zip(audio_in.iter().zip(&out_a)).map(|(x, (ai, o))| x - ((1.0 - sc.audio_scale) * ai + k * o) * sigma).collect();
        Ok((dv, da))
    }
}

/// A masked run (regenerate part of a clip, or extend one): the source latents, and per latent pixel / audio frame
/// whether to generate it (1) or keep the source (0). As the reference does it (ComfyUI's inpainting sampler with
/// H3's `scale_latent_inpaint` and per-token timesteps): the kept parts are put back, almost clean, before every
/// step and in every estimate; the model sees them at the conditioning timestep and its velocity there is zeroed.
pub struct Inpaint {
    /// normalized video latents [24, T, H, W] and audio latents [32, 2, At] (the model's own scale)
    pub source_v: Vec<f32>,
    pub source_a: Vec<f32>,
    /// [T, H, W] and [At], 1 = generate
    pub mask_v: Vec<f32>,
    pub mask_a: Vec<f32>,
}

/// The masks pooled to the tokens: a video token (2x2 latent pixels) generates if any of its pixels does.
pub struct RowMasks {
    /// per video token row, per audio row (channel-major, both channels alike)
    pub video_rows: Vec<f32>,
    pub audio_rows: Vec<f32>,
    /// the token mask back on the latent pixels [T, H, W]; per audio frame [At]
    pub video_px: Vec<f32>,
    pub audio_px: Vec<f32>,
}

impl RowMasks {
    pub fn new(inp: &Inpaint, sh: Shape) -> RowMasks {
        let (t, h, w) = (sh.t, sh.h, sh.w);
        let (hp, wp) = (h / PATCH, w / PATCH);
        let mut video_rows = vec![0f32; t * hp * wp];
        let mut video_px = vec![0f32; t * h * w];
        for ti in 0..t {
            for y in 0..hp {
                for x in 0..wp {
                    let mut m = 0f32;
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            m = m.max(inp.mask_v[(ti * h + y * 2 + dy) * w + x * 2 + dx]);
                        }
                    }
                    let m = (m * 256.0).ceil() / 256.0;
                    video_rows[(ti * hp + y) * wp + x] = m;
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            video_px[(ti * h + y * 2 + dy) * w + x * 2 + dx] = m;
                        }
                    }
                }
            }
        }
        let audio_px: Vec<f32> = inp.mask_a.iter().map(|m| (m * 256.0).ceil() / 256.0).collect();
        let audio_rows = (0..2).flat_map(|_| audio_px.iter().copied()).collect();
        RowMasks { video_rows, audio_rows, video_px, audio_px }
    }
}

/// Called after each sampler step with its index and the denoised video and audio.
pub type OnStep<'a> = &'a mut dyn FnMut(usize, &[f32], &[f32]) -> Result<()>;

/// Euler sampling from noise (video [C, T, H, W], audio [C, 2, T]) over `sigmas`. `on_step(i, denoised video,
/// denoised audio)` after each step; `between(step, block)` before each block. Returns the latents, the audio back
/// in the model's own scale.
pub fn sample(
    d: &mut Denoiser,
    noise_v: &[f32],
    noise_a: &[f32],
    sigmas: &[f32],
    on_step: OnStep,
    between: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    sample_masked(d, noise_v, noise_a, sigmas, None, on_step, between)
}

/// `sample`, with a masked run when `inpaint` is given (the denoiser's `masks` must be set from it).
pub fn sample_masked(
    d: &mut Denoiser,
    noise_v: &[f32],
    noise_a: &[f32],
    sigmas: &[f32],
    inpaint: Option<&Inpaint>,
    on_step: OnStep,
    between: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let s0 = sigmas[0];
    let scale = d.schedule.audio_scale;
    let mut xv: Vec<f32> = noise_v.iter().map(|n| n * s0).collect();
    let mut xa: Vec<f32> = noise_a.iter().map(|n| n * s0).collect();
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let (dv, da) = estimate(d, &xv, &xa, s, inpaint, noise_v, &mut |b| between(i, b))?;
        on_step(i, &dv, &da)?;
        for (x, dn) in xv.iter_mut().zip(&dv).chain(xa.iter_mut().zip(&da)) {
            let slope = (*x - dn) / s;
            *x += slope * (next - s);
        }
    }
    xa.iter_mut().for_each(|x| *x /= scale);
    Ok((xv, xa))
}

/// The denoised estimates at (x, sigma), as the sampler sees them; with a masked run the kept parts put back (as
/// ComfyUI's inpainting wrapper does around the model) before the model and in its estimates
fn estimate(d: &mut Denoiser, xv: &[f32], xa: &[f32], s: f32, inpaint: Option<&Inpaint>, noise_v: &[f32], between: Between)
            -> Result<(Vec<f32>, Vec<f32>)> {
    let scale = d.schedule.audio_scale;
    match (inpaint, &d.masks) {
        (Some(p), Some(m)) => {
            // the source in the sampler's space (the audio carried x audio_scale)
            let li_a: Vec<f32> = p.source_a.iter().map(|v| v * scale).collect();
            // the kept parts put back almost clean: 0.999 source + 0.001 noise (video), the audio rescaled for the
            // model to see it clean; inside a partly generated token the kept pixels follow x
            let aug = crate::layout::VISUAL_COND_TIMESTEP as f32;
            let (n, at) = (m.video_px.len(), p.mask_a.len());
            let xv2: Vec<f32> = (0..xv.len())
                .map(|j| {
                    let (mk, tok) = (p.mask_v[j % n], m.video_px[j % n]);
                    let mut inj = aug * p.source_v[j] + (1.0 - aug) * noise_v[j];
                    if mk < 1.0 {
                        let wgt = ((tok - mk) / (1.0 - mk).max(1e-6)).clamp(0.0, 1.0);
                        inj += wgt * (xv[j] - inj);
                    }
                    xv[j] * mk + inj * (1.0 - mk)
                })
                .collect();
            let sigma_v = s.max(1e-6);
            let sigma_a = time_shift_sigma(sigma_v as f64, d.schedule.shift_video as f64, d.schedule.shift_audio as f64) as f32;
            let factor = (sigma_v / sigma_a) / scale;
            let xa2: Vec<f32> = (0..xa.len()).map(|j| {
                let mk = p.mask_a[j % at];
                xa[j] * mk + li_a[j] * factor * (1.0 - mk)
            }).collect();
            let (dv, da) = d.denoised(&xv2, &xa2, s, between)?;
            let dv = (0..dv.len()).map(|j| dv[j] * p.mask_v[j % n] + p.source_v[j] * (1.0 - p.mask_v[j % n])).collect::<Vec<f32>>();
            let da = (0..da.len()).map(|j| da[j] * p.mask_a[j % at] + li_a[j] * (1.0 - p.mask_a[j % at])).collect::<Vec<f32>>();
            Ok((dv, da))
        }
        _ => d.denoised(xv, xa, s, between),
    }
}

/// `sample_masked` with any of the shared samplers (ComfyUI's, nextsycl_diffusion::samplers): the video and the
/// carried audio stepped as one vector; `seed` seeds the stochastic samplers' draws. Euler takes the path above.
#[allow(clippy::too_many_arguments)]
pub fn sample_with(
    d: &mut Denoiser,
    sampler: nextsycl_diffusion::Sampler,
    noise_v: &[f32],
    noise_a: &[f32],
    sigmas: &[f32],
    inpaint: Option<&Inpaint>,
    seed: u64,
    on_step: OnStep,
    between: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if sampler == nextsycl_diffusion::Sampler::Euler {
        return sample_masked(d, noise_v, noise_a, sigmas, inpaint, on_step, between);
    }
    let scale = d.schedule.audio_scale;
    let nv = noise_v.len();
    let s0 = sigmas[0];
    let x: Vec<f32> = noise_v.iter().chain(noise_a).map(|n| n * s0).collect();
    let step = std::cell::Cell::new(0usize);
    let last = std::cell::RefCell::new((Vec::new(), Vec::new()));
    let failed = std::cell::RefCell::new(None::<Error>);
    let mut model = |x: &[f32], s: f32| -> std::result::Result<Vec<f32>, String> {
        let i = step.get();
        let (dv, da) = estimate(d, &x[..nv], &x[nv..], s, inpaint, noise_v, &mut |b| between(i, b)).map_err(|e| e.0)?;
        let out = dv.iter().chain(&da).copied().collect();
        *last.borrow_mut() = (dv, da);
        Ok(out)
    };
    let mut rng = crate::rng::Rng::new(seed ^ 0x5eed_5a3d);
    let mut noise = |m: usize| -> Vec<f32> { (0..m).map(|_| rng.normal()).collect() };
    let mut progress = |i: usize| {
        step.set(i + 1);
        if failed.borrow().is_none() {
            let l = last.borrow();
            if let Err(e) = on_step(i, &l.0, &l.1) {
                *failed.borrow_mut() = Some(e);
            }
        }
    };
    let out = nextsycl_diffusion::samplers::sample(sampler, &mut model, x, sigmas, &mut noise, &mut progress).map_err(Error);
    if let Some(e) = failed.into_inner() {
        return Err(e);
    }
    let out = out?;
    let lastsig = *sigmas.last().unwrap_or(&0.0);
    let k = if lastsig > 0.0 { 1.0 / (1.0 - lastsig) } else { 1.0 };
    let xv = out[..nv].iter().map(|v| v * k).collect();
    let xa = out[nv..].iter().map(|v| v * k / scale).collect();
    Ok((xv, xa))
}

/// The outer parts must fit the blocks they were loaded with.
pub fn check_fits(cfg: &Config, o: &Outer) -> Result<()> {
    if o.video_patch.outputs() != cfg.hidden || o.audio_patch.outputs() != cfg.hidden || o.final_w.len() != 2 * cfg.hidden * cfg.t_dim {
        return Err(Error("the patch projections or the final layer do not fit the blocks".into()));
    }
    if o.inv_freq.len() * 6 != cfg.rot_dim {
        return Err(Error("the rotation frequencies do not fit the blocks".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_round_trip() {
        let (c, t, h, w) = (3, 2, 4, 6);
        let v: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify(&v, c, t, h, w);
        // row 0 is the top-left patch of frame 0; its first four features are channel 0's 2x2
        assert_eq!(&rows[..4], &[0.0, 1.0, 6.0, 7.0]);
        assert_eq!(unpatchify(&rows, c, t, h, w), v);
        let a: Vec<f32> = (0..4 * 2 * 5).map(|i| i as f32).collect();
        let r = pack_audio(&a, 4, 5);
        // row 1 = left channel, frame 1: a[c, 0, 1] for c in 0..4
        assert_eq!(&r[4..8], &[1.0, 11.0, 21.0, 31.0]);
        assert_eq!(unpack_audio(&r, 4, 5), a);
    }

    #[test]
    fn sigma_schedule_matches_the_reference() {
        // the reference's 8-step schedule at shift 12 (from a run dump)
        let want = [1.0, 0.988_235_3, 0.972_973, 0.952_381, 0.923_076_9, 0.878_048_8, 0.8, 0.631_578_9, 0.0];
        let got = sigmas(8, 12.0);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
    }
}
