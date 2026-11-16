//! Open-H3-Regenerate-2K-IC (NVlabs' Sana sol-engine; ComfyUI-OpenH3Regenerate2KIC's draft-to-2K workflow): a 768p H3
//! draft clip made again at 2560x1440 - its faces, lip sync, lettering and layout kept - by the Ref2VA denoiser with
//! the draft in its context as `<Video 1>` and an in-context (IC) LoRA that makes the high-resolution version:
//!
//! ```text
//!   draft .mp4 -> 124 frames at 24 fps, cover-fitted to 1344x768 -> H3 VAE encode (a posterior draw, seeded)  z
//!   z -> the 3D-conv latent upscaler (to the target's latent size, whole clip)                               up
//!   "<Video 1>: " + the draft at 2 fps + the official task prompt + "Scene description: " + caption -> text encoder
//!   denoiser (int8 Ref2VA) + HyperFlow 8-step (LoRA + two-time embedding, hyperflow.rs) + LMS Sharp + IC LoRA
//!     reference rows: z (0.999 z + 0.001 noise, at t >= 0.999); audio rows: the fixed silence latent, kept clean
//!     up3: x = (1 - 0.834942) up + 0.834942 noise, Euler on 0.834942, 0.696852, 0.468753, 0 (up4 / hf8: more of the grid)
//!   -> the H3 video decoder at 2560x1440 -> the draft's own sound muxed back, the draft's length kept
//! ```
//!
//! The whole clip is one sequence (~177k tokens at 2K: 133k target rows, 37k reference rows, the draft's vision
//! tokens): the denoiser's blocks stream from the int8 file (`Blocks::load_streamed`), the LoRAs from host memory
//! (`LoraSet::load_to`), the MLP in row chunks (`Conditions::mlp_rows`), so the card holds the activations.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

use serde_json::{json, Value};

use crate::denoiser::{self, Conditions, Denoiser, RefIn, Schedule, Shape, TextRefiner};
use crate::device::Tensor;
use crate::dtype::DType;
use crate::jobs::{self, Ctl, Engine, Latents, TeFiles, Vaes};
use crate::safetensors::Checkpoint;
use crate::vision::Piece;
use crate::{Error, Result};

/// Open-H3-Regenerate-2K-IC's task prompt (infer/ref2va_ic/encode.py)
pub const TASK: &str = "Enhance <Video 1> into a high-quality version of the same video. Preserve the scene, objects, camera motion, timing, and composition \
                        of every frame. Restore fine details and remove degradation. Do not continue or change the scene.";

/// The draft's length on the model's grid (5.17 s), and the fewest real frames it takes (infer/ref2va_ic/media.py)
pub const FRAMES: usize = 124;
pub const MIN_REAL: usize = 113;

/// The output sizes: (draft canvas w, h, computed w, h, rows kept)
pub fn resolution(name: &str) -> Option<(usize, usize, usize, usize, usize)> {
    match name {
        "2k" => Some((1344, 768, 2560, 1440, 1440)),
        "1080p" => Some((1344, 768, 1920, 1088, 1080)),
        _ => None,
    }
}

/// The prompt the text encoder reads after the draft's vision blocks
pub fn prompt(caption: &str) -> String {
    format!("{TASK}\n\nScene description: {}", caption.trim())
}

/// What a regen2k job takes
pub struct RegenSpec<'a> {
    pub draft: &'a Path,
    pub caption: String,
    pub seed: u64,
    /// up3 | up4 | hf8 (the last 3 / 4 / all 8 steps of HyperFlow's grid), or the noise levels themselves
    pub sampler: String,
    pub sigmas: Option<Vec<f32>>,
    /// the draft's canvas, the computed size, the rows kept (centre)
    pub lq: (usize, usize),
    pub target: (usize, usize),
    pub keep_h: usize,
    pub frames: usize,
    pub te: TeFiles<'a>,
    /// the H3 VAE (its encoder and its decoder)
    pub vae: &'a Path,
    pub upscaler: &'a Path,
    pub ic_lora: &'a Path,
    pub hyperflow: &'a Path,
    /// the bf16 checkpoint's time embedder and full AdaLN projections (hyperflow.rs)
    pub adaln: &'a Path,
    /// LMS Sharp and its strength (0: off)
    pub sharp: Option<(&'a Path, f32)>,
    pub silence: &'a Path,
    /// MLP rows at a time
    pub mlp_rows: usize,
    /// a parity dump: the net's inputs and its output at the first level, then stop
    pub check_dump: Option<&'a Path>,
}


/// The silence latent [2, 32, A] -> [32, 2, A] (the denoiser's audio layout), A cut or zero-padded to `at`
fn silence(path: &Path, at: usize) -> Result<Vec<f32>> {
    let ck = Checkpoint::open(path)?;
    let e = ck.get("audio_latents")?.clone();
    let v = crate::dtype::bytes_to_f32(&ck.read("audio_latents")?, e.dtype)?;
    if e.shape.len() != 3 || e.shape[0] != 2 {
        return Err(Error(format!("{}: audio_latents {:?}, [2, C, A] expected", path.display(), e.shape)));
    }
    let (c, a) = (e.shape[1], e.shape[2]);
    let mut out = vec![0f32; c * 2 * at];
    for ch in 0..2 {
        for ci in 0..c {
            for ti in 0..at.min(a) {
                out[(ci * 2 + ch) * at + ti] = v[(ch * c + ci) * a + ti];
            }
        }
    }
    Ok(out)
}

/// The draft: `frames` frames at 24 fps cover-fitted to the canvas, the last real one repeated; (pixels, real frames)
fn read_draft(path: &Path, (w, h): (usize, usize), frames: usize) -> Result<(Vec<f32>, usize)> {
    let (mut px, got) = crate::media::read_frames_at(path, w, h, crate::media::Fit::Cover, Some((24.0, frames)))?;
    if got < MIN_REAL.min(frames) {
        return Err(Error(format!("{}: {got} frames at 24 fps; the model needs at least {MIN_REAL} (about 4.7 s)", path.display())));
    }
    let fsz = w * h * 3;
    let last = px[(got - 1) * fsz..got * fsz].to_vec();
    for _ in got..frames {
        px.extend_from_slice(&last);
    }
    px.truncate(frames * fsz);
    Ok((px, got))
}

/// The run's noise levels: HyperFlow's 8-step grid (shift 12), the last 3 / 4 points or all of it
pub fn sigmas(grid: &[f32], sampler: &str) -> Result<(Vec<f32>, bool)> {
    let n = grid.len();
    match sampler {
        "up3" if n == 9 => Ok((grid[5..].to_vec(), true)),
        "up4" if n == 9 => Ok((grid[4..].to_vec(), true)),
        "hf8" if n == 9 => Ok((grid.to_vec(), false)),
        _ => Err(Error(format!("sampler {sampler}? (up3, up4, hf8; HyperFlow's grid has {n} points)"))),
    }
}

/// A draft -> its 2K regeneration at `out` (.mp4, the draft's sound)
pub fn regen2k(e: &Engine, s: &RegenSpec, out: &Path, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let dev = &e.dev;
    let ((lw, lh), (tw, th)) = (s.lq, s.target);
    if !lw.is_multiple_of(32) || !lh.is_multiple_of(32) || !tw.is_multiple_of(32) || !th.is_multiple_of(32) {
        return Err(Error(format!("the draft canvas {lw}x{lh} and the target {tw}x{th} must be multiples of 32")));
    }
    let (frames, lt, at) = jobs::temporal_shape(s.frames as f64 / 24.0);
    if frames != s.frames {
        return Err(Error(format!("{} frames is not on the model's grid (17k + 5)", s.frames)));
    }
    let mut report = json!({"draft": s.draft.display().to_string(), "target": [tw, th], "frames": frames});
    let mut peak = dev.mem_used();
    let mut note_peak = |d: &crate::device::Device| peak = peak.max(d.mem_used());

    // 1. the draft, its latent and the upscaled start
    let t0 = Instant::now();
    let (px, real) = read_draft(s.draft, (lw, lh), frames)?;
    ctl.say(format!("draft  : {} - {real} real frames, {frames} at {lw}x{lh}", s.draft.display()));
    let (z, zt) = {
        let enc = crate::venc::VideoEncoder::load(dev, &Checkpoint::open(s.vae)?)?;
        let r = enc.encode_with(&px, frames, lh, lw, Some(s.seed), &mut || ctl.check())?;
        note_peak(dev);
        r
    };
    if zt != lt {
        return Err(Error(format!("the draft encoded to {zt} latent frames, the grid says {lt}")));
    }
    let (zh, zw, uh, uw) = (lh / 16, lw / 16, th / 16, tw / 16);
    let encode_s = t0.elapsed().as_secs_f64();
    ctl.say(format!("encode : {frames} frames -> {lt} x {zw}x{zh} latents in {encode_s:.1} s"));
    let t0 = Instant::now();
    let up = {
        let u = crate::upscale::Upscaler::load(dev, &Checkpoint::open(s.upscaler)?)?;
        let scale = ((tw as f32 / lw as f32) + (th as f32 / lh as f32)) / 2.0;
        let cancel = ctl.cancel;
        let (v, _, _) = u.upscale_to(&z, (lt, zh, zw), (uh, uw), scale, false, &mut || {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error("cancelled".into()));
            }
            Ok(())
        })?;
        note_peak(dev);
        v
    };
    let upscale_s = t0.elapsed().as_secs_f64();
    ctl.say(format!("upscale: {zw}x{zh} -> {uw}x{uh} latents in {upscale_s:.1} s"));

    // 2. the presentation: the draft as <Video 1>, then the task prompt
    let mut pieces = vec![Piece::Text("<Video 1>: ".into())];
    pieces.extend(jobs::ref_video_blocks(&px, frames, lw, lh));
    pieces.push(Piece::Text(prompt(&s.caption)));
    drop(px);
    let (ctx, presented, hidden, te_s) = jobs::encode_pieces(dev, e.threads, &pieces, &s.te, ctl)?;
    ctl.check()?;

    // 3. the LoRAs: the blocks' in host memory, the refiner's on the card
    let t0 = Instant::now();
    let nb = e.model.blocks.len();
    let mut lora = crate::lora::LoraSet::load_to(dev, &Checkpoint::open(s.hyperflow)?, 1.0, nb, 2, true)?;
    if let Some((p, strength)) = s.sharp.filter(|(_, st)| *st != 0.0) {
        lora.stack(crate::lora::LoraSet::load_to(dev, &Checkpoint::open(p)?, strength, nb, 2, true)?);
    }
    lora.stack(crate::lora::LoraSet::load_to(dev, &Checkpoint::open(s.ic_lora)?, 1.0, nb, 2, true)?);
    if lora.max_rank > crate::dit::MAX_LORA_RANK {
        return Err(Error(format!("LoRA rank {} is over {}", lora.max_rank, crate::dit::MAX_LORA_RANK)));
    }
    ctl.say(format!("lora   : HyperFlow + {}IC: {} layers, rank up to {}, {:.2} GiB in host memory, read in {:.1} s",
                    if s.sharp.is_some_and(|x| x.1 != 0.0) { "LMS Sharp + " } else { "" }, lora.layers, lora.max_rank,
                    jobs::gib(lora.host_bytes() as u64), t0.elapsed().as_secs_f64()));
    let text = {
        let refiner = TextRefiner::load(dev, &e.ck, e.threads)?;
        let t = Tensor::from_bytes(dev, DType::BF16, &[presented.ids.len(), hidden], &denoiser::bf16_bytes(&ctx))?;
        let r = refiner.run(&e.model.cfg, &t, Some(lora.refiner.as_slice()))?;
        dev.wait()?;
        note_peak(dev);
        r
    };

    // 4. the denoiser: the draft's latent as the reference clip, the upscaled draft as the start, silence kept
    let shape = Shape { t: lt, h: uh, w: uw, audio_t: at };
    let cond = Conditions { seed: s.seed, refs: vec![RefIn::Video { latent: z.clone(), t: lt, h: zh, w: zw, audio: None }], mlp_rows: Some(s.mlp_rows),
                            ..Conditions::default() };
    let tags = presented.has_vision().then(|| presented.tags.clone());
    // what the sequence will hold on the card, before any of it is made: the scratch, the stream, the attention's int8
    // copies and a margin for the kernels' own buffers; refused when the card has less (a clear error, not a spill)
    let cfg = &e.model.cfg;
    let tokens = presented.ids.len() + lt * (zh / 2) * (zw / 2) + 2 * at + lt * (uh / 2) * (uw / 2);
    let need = crate::dit::Scratch::bytes(cfg, tokens, s.mlp_rows) + tokens * cfg.hidden * 4 + tokens * cfg.heads * cfg.head_dim * 3 + (2usize << 30);
    if let Some(free) = dev.mem_free() {
        ctl.say(format!("memory : {tokens} tokens need about {:.1} GiB on the card, {:.1} GiB free", jobs::gib(need as u64), jobs::gib(free)));
        if (need as u64) + (3u64 << 29) > free {
            return Err(Error(format!("{tokens} tokens need about {:.1} GiB on the card and 1.5 GiB to spare; {:.1} GiB are free (a smaller draft or mlp_rows)",
                                     jobs::gib(need as u64), jobs::gib(free))));
        }
    }
    let mut d = Denoiser::new(&e.model, &e.outer, text, tags, shape, Schedule::default(), &cond)?;
    d.lora = Some(&lora);
    let source_a = silence(s.silence, at)?;
    let inpaint = denoiser::Inpaint { source_v: up.clone(), source_a, mask_v: vec![1.0; lt * uh * uw], mask_a: vec![0.0; at] };
    d.masks = Some(denoiser::RowMasks::new(&inpaint, shape));
    let t0 = Instant::now();
    let two = crate::hyperflow::TwoTime::load(s.adaln, &Checkpoint::open(s.hyperflow)?)?;
    let (sigmas, from_up) = match &s.sigmas {
        Some(v) => (v.clone(), v.first().is_some_and(|s0| *s0 < 1.0)),
        None => sigmas(&two.grid, &s.sampler)?,
    };
    let points = crate::hyperflow::points(&sigmas, &mut |sg| d.values_at(sg));
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let prep = two.prepare(&points, &sigmas, Some(&lora.adaln), &e.outer.t_table, e.model.cfg.t_dim, threads)?;
    drop(two);
    ctl.say(format!("time   : HyperFlow's two-time tables for {} points in {:.1} s; sigmas {sigmas:?}", points.len(), t0.elapsed().as_secs_f64()));
    d.hyperflow = Some(&prep);
    let n_v = 24 * lt * uh * uw;
    let (noise_v, noise_a) = crate::noise::clip_noise(s.seed, n_v, 32 * 2 * at);
    let noise_v = match from_up {
        // (1 - s0) up + s0 noise, as the sampler's s0 x noise
        true => noise_v.iter().zip(&up).map(|(n, c)| n + c * (1.0 - sigmas[0]) / sigmas[0]).collect(),
        false => noise_v,
    };
    ctl.say(format!("tokens : {} in the denoiser ({} blocks on the card, the rest streamed)", d.tokens(), e.model.resident()));
    note_peak(dev);

    if let Some(dump) = s.check_dump {
        return check(&mut d, &z, (lt, zh, zw), &noise_v, &inpaint, &sigmas, &ctx, &presented, s.seed, dump, ctl);
    }

    let steps = sigmas.len() - 1;
    let t0 = Instant::now();
    let (v, _) = {
        let cancel = ctl.cancel;
        let mut progress = ctl.progress.take();
        let mut times = Vec::new();
        let mut t_step = Instant::now();
        let devc = dev.clone();
        let r = denoiser::sample_masked(&mut d, &noise_v, &noise_a, &sigmas, Some(&inpaint), &mut |_, _, _| {
            times.push(t_step.elapsed().as_secs_f64());
            t_step = Instant::now();
            Ok(())
        }, &mut |i, b| {
            if b == 0 || b == nb / 2 {
                devc.wait()?;
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error("cancelled".into()));
                }
            }
            if let Some(p) = progress.as_mut() {
                p(i * nb + b, steps * nb);
            }
            Ok(())
        });
        ctl.progress = progress;
        let r = r?;
        ctl.say(format!("sampled: {steps} steps in {:.1} s ({})", t0.elapsed().as_secs_f64(), times.iter().map(|s| format!("{s:.1}")).collect::<Vec<_>>().join(" ")));
        r
    };
    note_peak(dev);
    let sample_s = t0.elapsed().as_secs_f64();
    drop(d);
    drop(lora);
    crate::safetensors::write_f32(&out.with_extension("latents.safetensors"), &BTreeMap::from([
        ("samples.video".to_string(), (vec![1, 24, lt, uh, uw], v.clone())),
    ]), &BTreeMap::new())?;

    // 5. the frames, the draft's length and sound
    let rep = jobs::decode_latents(dev, e.threads, Latents { video: v, t: lt, h: uh, w: uw, audio: None },
                                   &Vaes { video: s.vae, audio: None, upscale: None, pixel: None }, Some(out), None, ctl)?;
    note_peak(dev);
    finish(out, s.draft, real, frames, (tw, th), s.keep_h, ctl)?;
    let total = t_all.elapsed().as_secs_f64();
    for (k, v) in [("encode_seconds", json!(encode_s)), ("upscale_seconds", json!(upscale_s)), ("text_encoder_seconds", json!(te_s)),
                   ("sample_seconds", json!(sample_s)), ("decode_seconds", rep.get("seconds").cloned().unwrap_or(json!(null))),
                   ("total_seconds", json!(total)), ("real_frames", json!(real)), ("peak_gib", json!(jobs::gib(peak))),
                   ("sampler", json!(s.sampler)), ("sigmas", json!(sigmas)), ("width", json!(tw)), ("height", json!(s.keep_h)), ("out", json!(out))] {
        report[k] = v;
    }
    ctl.say(format!("regen  : {} ({tw}x{}, {real} frames, {:.2} s) in {total:.1} s, card peak {:.1} GiB", out.display(), s.keep_h, real as f64 / 24.0, jobs::gib(peak)));
    Ok(report)
}

/// The decoded clip cut to the draft's real frames (and the kept rows), the draft's sound laid under it
fn finish(out: &Path, draft: &Path, real: usize, frames: usize, (w, h): (usize, usize), keep_h: usize, ctl: &mut Ctl) -> Result<()> {
    if real < frames || keep_h < h {
        let tmp = out.with_extension("cut.mp4");
        let mut cmd = std::process::Command::new("ffmpeg");
        cmd.args(["-y", "-loglevel", "error", "-i"]).arg(out).args(["-frames:v", &real.to_string()]);
        if keep_h < h {
            cmd.args(["-vf", &format!("crop={w}:{keep_h}:0:{}", (h - keep_h) / 2)]);
            cmd.args(["-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p"]);
        } else {
            cmd.args(["-c", "copy"]);
        }
        let st = cmd.arg(&tmp).status().map_err(|e| Error(format!("ffmpeg: {e}")))?;
        if !st.success() {
            return Err(Error(format!("{}: ffmpeg could not cut the clip to {real} frames", out.display())));
        }
        std::fs::rename(&tmp, out)?;
    }
    match crate::media::mux_sound(out, draft, 0.0, real as f64 / 24.0) {
        Ok(()) => ctl.say(format!("sound  : the draft's, {:.2} s", real as f64 / 24.0)),
        Err(e) => ctl.say(format!("sound  : none kept ({e})")),
    }
    Ok(())
}

/// The parity dump (reference/regen2k/dit_ref.py): the inputs of the first net call - the reference clip's latent as the
/// rows see it (0.999 z + 0.001 noise, unpatched), x, the silence, the conditioning and its tags - and the velocity the
/// engine gives at the first level
#[allow(clippy::too_many_arguments)]
fn check(d: &mut Denoiser, z: &[f32], (t, h, w): (usize, usize, usize), noise_v: &[f32], inpaint: &denoiser::Inpaint, sigmas: &[f32],
         ctx: &[f32], presented: &crate::vision::Presented, seed: u64, dump: &Path, ctl: &mut Ctl) -> Result<Value> {
    let s0 = sigmas[0];
    let aug = crate::layout::VISUAL_COND_TIMESTEP as f32;
    let rows = denoiser::patchify(z, 24, t, h, w);
    let noise = crate::noise::Mt19937::new(seed).randn(rows.len());
    let mixed: Vec<f32> = rows.iter().zip(&noise).map(|(r, n)| aug * r + (1.0 - aug) * n).collect();
    let anchor = denoiser::unpatchify(&mixed, 24, t, h, w);
    let xv: Vec<f32> = noise_v.iter().map(|n| n * s0).collect();
    let t0 = Instant::now();
    // the model's own call; the kept silence is what the sampler's wrapper hands it
    let sh = d.shape;
    let (v, va) = d.net(&xv, &inpaint.source_a, s0, &mut |_| Ok(()))?;
    ctl.say(format!("check  : one net call at sigma {s0} in {:.1} s", t0.elapsed().as_secs_f64()));
    let hidden = ctx.len() / presented.ids.len();
    let mut extra = BTreeMap::new();
    if let Some(hf) = d.hyperflow {
        let p = hf.points();
        extra.insert("hf.points".to_string(), (vec![p.len(), 2], p.iter().flat_map(|(t, r)| [*t as f32, *r as f32]).collect()));
        for (name, b) in [("hf.block0", Some(0)), ("hf.block49", Some(d.blocks.blocks.len() - 1)), ("hf.final", None)] {
            let rows: Vec<f32> = (0..p.len()).flat_map(|i| hf.row(b, i).to_vec()).collect();
            extra.insert(name.to_string(), (vec![p.len(), rows.len() / p.len()], rows));
        }
    }
    let mut tensors = BTreeMap::from([
        ("anchor".to_string(), (vec![24, t, h, w], anchor)),
        ("z".to_string(), (vec![24, t, h, w], z.to_vec())),
        ("up".to_string(), (vec![24, sh.t, sh.h, sh.w], inpaint.source_v.clone())),
        ("x".to_string(), (vec![24, sh.t, sh.h, sh.w], xv)),
        ("audio".to_string(), (vec![32, 2, sh.audio_t], inpaint.source_a.clone())),
        ("context".to_string(), (vec![presented.ids.len(), hidden], ctx.to_vec())),
        ("tags".to_string(), (vec![presented.tags.len()], presented.tags.iter().map(|x| *x as f32).collect())),
        ("ids".to_string(), (vec![presented.ids.len()], presented.ids.iter().map(|x| *x as f32).collect())),
        ("v".to_string(), (vec![24, sh.t, sh.h, sh.w], v)),
        ("va".to_string(), (vec![32, 2, sh.audio_t], va)),
        ("sigmas".to_string(), (vec![sigmas.len()], sigmas.to_vec())),
    ]);
    tensors.extend(extra);
    crate::safetensors::write_f32(dump, &tensors, &BTreeMap::from([("seed".to_string(), seed.to_string())]))?;
    ctl.say(format!("check  : written {}", dump.display()));
    Ok(json!({"dump": dump.display().to_string()}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_samplers_take_the_end_of_the_grid() {
        let g = [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0];
        assert_eq!(super::sigmas(&g, "up3").unwrap(), (vec![0.834942, 0.696852, 0.468753, 0.0], true));
        assert_eq!(super::sigmas(&g, "up4").unwrap().0.len(), 5);
        assert_eq!(super::sigmas(&g, "hf8").unwrap(), (g.to_vec(), false));
        assert!(super::sigmas(&g, "up5").is_err());
    }

    #[test]
    fn the_prompt_is_the_official_one() {
        assert!(super::prompt(" a fox ").ends_with("Do not continue or change the scene.\n\nScene description: a fox"));
    }
}
