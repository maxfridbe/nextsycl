//! MiniMax Music 3 on Intel Arc (SYCL): lyrics and a description to a song (up to six minutes, 44.1 kHz stereo),
//! the whole pipeline on one GPU - or on two, as the reference runs it: the language model on the first, the flow
//! and the decoder on the second, pipelined (a window's flow starts as soon as its frames are made).
//!
//! ```text
//!   description + lyrics -> the checkpoint's template (prompt.rs) -> tokens, and their classifier-free twin
//!   -> autoregressive stage (ar.rs): an 8B language model draws a semantic code a frame (25 a second), a depth
//!      decoder the seven residual codes; each frame's hidden states mixed into one conditioning row
//!   -> flow matching (flow.rs): 200-frame windows, 100 apart, of Flow-VAE latents from noise (30 Euler steps,
//!      guidance 1.7), each blended into the previous over their overlap
//!   -> the Flow-VAE decoder (vocoder.rs) a window at a time, the overlaps cropped
//! ```
//!
//! The weights are the checkpoint's diffusers files (`nextsycl models pull minimax-music3`): the language model's
//! four bf16 shards (half on the GPU, or int8 with `--opt-int8`), the depth decoder (bf16), the flow transformer and
//! condition encoder and decoder (f32; the transformer in half on the GPU), the tokenizer. `check` compares each
//! stage with the reference's dumps (reference/minimaxmusic3/ref.py: diffusers' own code on the same files).

pub mod ar;
pub mod check;
mod ffi;
pub mod flow;
pub mod ops;
pub mod prompt;
pub mod sample;
pub mod vocoder;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_audio::{shards, At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, LoadOptions, ModelFiles, Result, Step};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_tok::Tokenizer;

use crate::ar::{Ar, HIDDEN};
use crate::flow::{Flow, LATENT};
use crate::ops::{fp, Ops, Shards};
use crate::sample::Rng;
use crate::vocoder::{Vocoder, HOP};

pub const ARCH: &str = "minimax-music3";
/// language-model (4 shards: -1 ... -4), transformer (2: -1, -2), the rest one file each
pub const ROLES: &[&str] = &["language-model*", "depth-decoder", "transformer*", "condition-encoder", "vocoder", "tokenizer"];

const GIB: f64 = (1u64 << 30) as f64;
/// frames a second of the autoregressive stage, and the most it makes (the checkpoint's limits)
pub const FPS: f32 = 25.0;
const MAX_FRAMES: usize = 9000;
const MAX_PROMPT: usize = 5000;
pub const RATE: u32 = 44100;
/// the decoded windows' crops (latent frames): the start of every window but the first, the end of every one but the last
const CROP_LEFT: usize = 86;
const CROP_RIGHT: usize = 344 - 86;

/// The options it takes (`--opt-NAME`: at load the variable named)
pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "int8", env: "NS_MM3_INT8", value: "0|1", help: "the language model and depth decoder in int8 (half the VRAM, ~1.7x the frames a second)", at: At::Load },
    EngineOption { name: "dit-int8", env: "NS_MM3_DIT_INT8", value: "0|1", help: "the flow transformer's block matrices in int8 ConvRot (the card's int8 rate)", at: At::Load },
];

/// This engine's registry entry
pub fn kind() -> AudioKind {
    AudioKind { archs: &[ARCH], name: "MiniMax Music 3 (8B semantic LM + depth decoder, 2.4B flow transformer, Flow-VAE: songs from lyrics)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn AudioEngine>> {
    Ok(Box::new(MiniMaxMusic3::load_on(files, gpus, o, log)?))
}

pub struct MiniMaxMusic3 {
    /// the language model's GPU (and with one GPU everything's)
    pub ops: Ops,
    pub nsd: Nsd,
    pub tok: Tokenizer,
    pub ar: Ar,
    pub flow: Flow,
    pub voc: Vocoder,
    /// the flow's and decoder's GPU when it is a second one
    second: Option<(Ops, Nsd)>,
    loaded: Instant,
    load_s: f64,
}

/// A song's flow and decoder as they go, a window at a time: the previous window's overlap (latents, conditioning),
/// the work buffers, the samples so far
#[derive(Default)]
struct Sounding {
    prev: Option<(DevBuf, DevBuf)>,
    work: Option<flow::Work>,
    out: [Vec<f32>; 2],
}

/// frames a block handed to the second GPU's flow
const HANDOFF: usize = 20;

/// Where the frames' mixed rows go as they are made (the second GPU's flow: a block of rows [n, HIDDEN])
type RowSink<'a> = Option<&'a mut dyn FnMut(&[f32]) -> Result<()>>;

/// What the flow's thread hands the request's thread: progress, or its end
enum FlowMsg {
    Step(Step),
    Done(Result<[Vec<f32>; 2]>),
}

fn role(files: &ModelFiles, r: &str) -> Result<Vec<std::path::PathBuf>> {
    let v = shards(files, r);
    if v.is_empty() {
        return Err(Error(format!("{ARCH}: no {r} file")));
    }
    Ok(v)
}

fn size(paths: &[std::path::PathBuf]) -> u64 {
    paths.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum()
}

impl MiniMaxMusic3 {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<MiniMaxMusic3> {
        MiniMaxMusic3::load_on(files, std::slice::from_ref(gpu), o, log)
    }

    /// On one GPU, or on two: the language model on the first, the flow transformer, condition encoder and decoder on
    /// the second
    pub fn load_on(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<MiniMaxMusic3> {
        let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
        let fgpu = gpus.get(1).filter(|g| g.index != gpu.index);
        let t0 = Instant::now();
        nextsycl_core::use_kind("audio");
        let int8 = o.setting("NS_MM3_INT8").is_some_and(|v| v == "1");
        let (lm, depth, tr) = (role(files, "language-model")?, role(files, "depth-decoder")?, role(files, "transformer")?);
        let (cond, voc, tok) = (role(files, "condition-encoder")?, role(files, "vocoder")?, role(files, "tokenizer")?);
        // the xe driver has no out-of-memory error (an allocation past the card spills to host RAM and can take the
        // machine down), so the plan is checked first: the language model (bf16 on disk; half, or int8: a byte a
        // weight, its embedding table left on disk), the depth decoder, the transformer in half (f32 on disk), a
        // six-minute song's cache (2 GiB) and the work buffers, 1.5 GiB kept free
        let lm_b = size(&lm) - 2 * 200_000 * 4096 * 2;
        let ar_b = (lm_b + size(&depth)) / if int8 { 2 } else { 1 };
        let flow_b = size(&tr) / 2 + size(&cond) + size(&voc);
        // two GPUs: the language model and the cache (2 GiB) and its work (1 GiB) on the first; the flow, the decoder and
        // their work (2 GiB) on the second; 1.5 GiB kept free on each
        let plan: Vec<(&Arc<Gpu>, u64)> = match fgpu {
            None => vec![(gpu, ar_b + flow_b + (5u64 << 30))],
            Some(f) => vec![(gpu, ar_b + (3u64 << 30) + (3u64 << 29)), (f, flow_b + (2u64 << 30) + (3u64 << 29))],
        };
        for (g, need) in &plan {
            let (total, free) = g.memory()?;
            let free = free.unwrap_or(total);
            if *need > free {
                return Err(Error(format!("{ARCH} needs about {:.1} GiB on {}; {:.1} GiB are free{}", *need as f64 / GIB, g.name, free as f64 / GIB,
                                         if int8 { "" } else { " (--opt-int8 1 halves the language model)" })));
            }
        }
        let ops = Ops::new(gpu)?;
        let nsd = Nsd::new(gpu)?;
        let second = match fgpu {
            Some(f) => Some((Ops::new(f)?, Nsd::new(f)?)),
            None => None,
        };
        let (fops, fnsd) = second.as_ref().map_or((&ops, &nsd), |(o, n)| (o, n));
        let tok = Tokenizer::from_hf_json(&tok[0]).map_err(Error)?;
        if tok.id("<|audio_start|>") != Some(151669) {
            return Err(Error(format!("{}: not MiniMax Music 3's tokenizer (no <|audio_start|> at 151669)", tok_path(files))));
        }
        let ar = Ar::load(&ops, Shards::open(&lm)?, &Shards::open(&depth)?, int8, log)?;
        let dit_int8 = o.setting("NS_MM3_DIT_INT8").is_some_and(|v| v == "1");
        let flow = Flow::load(fops, fnsd, &Shards::open(&cond)?, &Shards::open(&tr)?, dit_int8, log)?;
        let voc = Vocoder::load(fops, &Shards::open(&voc)?)?;
        gpu.sync()?;
        fops.gpu.sync()?;
        let load_s = t0.elapsed().as_secs_f64();
        match fgpu {
            Some(f) => log(format!("{ARCH}: the language model on {} (GPU {}), the flow and decoder on {} (GPU {}), in {load_s:.0} s", gpu.name, gpu.index,
                                   f.name, f.index)),
            None => log(format!("{ARCH} on {} in {load_s:.0} s", gpu.name)),
        }
        Ok(MiniMaxMusic3 { ops, nsd, tok, ar, flow, voc, second, loaded: Instant::now(), load_s })
    }

    /// The flow's and decoder's GPU
    fn fo(&self) -> (&Ops, &Nsd) {
        self.second.as_ref().map_or((&self.ops, &self.nsd), |(o, n)| (o, n))
    }

    /// The prompt's ids (checked against the limit) and its twin's
    pub fn tokens(&self, caption: &str, lyrics: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        if caption.trim().is_empty() {
            return Err(Error("the description is empty".into()));
        }
        if lyrics.trim().is_empty() {
            return Err(Error("the lyrics are empty (an instrumental: \"[instrumental]\")".into()));
        }
        let ids = self.tok.encode(&prompt::text(caption, lyrics));
        if ids.len() > MAX_PROMPT {
            return Err(Error(format!("the prompt is {} tokens; the most is {MAX_PROMPT}", ids.len())));
        }
        let unc = prompt::unconditional(&ids);
        Ok((ids, unc))
    }

    /// The autoregressive stage: the frames' mixed conditioning rows [frames, HIDDEN] on the GPU and their count
    pub fn frames(&self, ids: &[u32], unc: &[u32], max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(DevBuf, usize)> {
        self.frames_to(ids, unc, max, seed, t0, progress, None)
    }

    /// `frames`, the mixed rows also handed to `row` as they are made, a block of HANDOFF at a time (to the second
    /// GPU's flow; a read a frame would wait for each frame's pass)
    #[allow(clippy::too_many_arguments)]
    fn frames_to(&self, ids: &[u32], unc: &[u32], max: usize, seed: u64, t0: Instant, progress: &mut dyn FnMut(Step) -> Result<()>,
                 mut row: RowSink) -> Result<(DevBuf, usize)> {
        let (ops, nsd) = (&self.ops, &self.nsd);
        let s = self.ar.session(ops, ids.len(), max + 1)?;
        self.ar.prefill(ops, nsd, &s, ids, unc)?;
        progress(Step { phase: "prompt", at: 1, of: 1, seconds: t0.elapsed().as_secs_f64() })?;
        let frames = DevBuf::f32(&ops.gpu, max * HIDDEN)?;
        let mut rng = Rng::new(seed);
        let mut kept = 0;
        let mut sent = 0;
        let hand = |kept: usize, sent: &mut usize, row: &mut RowSink| -> Result<()> {
            if let Some(r) = row.as_deref_mut() {
                if kept > *sent {
                    let mut b = vec![0u8; (kept - *sent) * HIDDEN * 4];
                    frames.read(*sent * HIDDEN * 4, &mut b)?;
                    let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    r(&v)?;
                    *sent = kept;
                }
            }
            Ok(())
        };
        for f in 0..=max {
            if self.ar.frame(ops, nsd, &s, ids.len() + f, &mut rng, None, None)?.is_none() {
                break;
            }
            if f > 0 {
                ops.mix(s.stage.fp(), 8, HIDDEN, &self.flow.mix, self.flow.mix_scale, fp(&frames, kept * HIDDEN))?;
                kept += 1;
                if kept - sent >= HANDOFF {
                    hand(kept, &mut sent, &mut row)?;
                }
                progress(Step { phase: "tokens", at: kept as u32, of: max as u32, seconds: t0.elapsed().as_secs_f64() })?;
                if kept >= max {
                    break;
                }
            }
        }
        ops.gpu.sync()?;
        hand(kept, &mut sent, &mut row)?;
        if kept == 0 {
            return Err(Error("the model ended the song before its first frame".into()));
        }
        Ok((frames, kept))
    }

    /// The flow stage and the decoder over `n` frames: stereo samples, one Vec a channel. `noise(k, l)`: window k's
    /// starting latents [l, LATENT] (a check's from the reference; else seeded draws)
    #[allow(clippy::too_many_arguments)]
    pub fn sound(&self, frames: &DevBuf, n: usize, steps: usize, cfg: f32, noise: &mut dyn FnMut(usize, usize) -> Result<Vec<f32>>, t0: Instant,
                 progress: &mut dyn FnMut(Step) -> Result<()>, mut keep: Option<&mut Vec<Vec<f32>>>) -> Result<[Vec<f32>; 2]> {
        let starts = Flow::starts(n);
        let mut st = Sounding::default();
        for (k, s0) in starts.iter().enumerate() {
            self.window(&mut st, k, *s0, (s0 + flow::CHUNK_FRAMES).min(n), k + 1 == starts.len(), frames, steps, cfg, noise, t0, starts.len(), progress,
                        keep.as_deref_mut())?;
        }
        Ok(st.out)
    }

    /// Window k (frames [s0, s1); `last`: the song's last) through the flow and the decoder, on the flow's GPU
    /// (`frames` there), its samples added to the song's
    #[allow(clippy::too_many_arguments)]
    fn window(&self, st: &mut Sounding, k: usize, s0: usize, s1: usize, last: bool, frames: &DevBuf, steps: usize, cfg: f32,
              noise: &mut dyn FnMut(usize, usize) -> Result<Vec<f32>>, t0: Instant, windows: usize, progress: &mut dyn FnMut(Step) -> Result<()>,
              keep: Option<&mut Vec<Vec<f32>>>) -> Result<()> {
        let (ops, nsd) = self.fo();
        let times = Flow::times(steps);
        let cond = self.flow.condition(ops, nsd, frames, s0, s1 - s0)?;
        let l = cond.floats() / 2048;
        let mut ov = 0;
        if let Some((pl, pc)) = &st.prev {
            ov = (pl.floats() / LATENT).min(l);
            cond.copy_within(0, pc, 0, ov * 2048 * 4)?;
        }
        let lat = DevBuf::from_f32(&ops.gpu, &noise(k, l)?)?;
        if st.work.as_ref().is_none_or(|w| w.v.floats() != 2 * l * LATENT) {
            // the old window's buffers freed before the new ones are made
            drop(st.work.take());
            st.work = Some(self.flow.work(ops, l)?);
        }
        let w = st.work.as_ref().expect("made above");
        let p = st.prev.as_ref().map(|(pl, _)| (pl, ov));
        self.flow.denoise(ops, nsd, w, &lat, &cond, &times, cfg, p.filter(|(_, o)| *o > 0), &mut |i| {
            progress(Step { phase: "flow", at: (k * steps + i + 1) as u32, of: (windows.max(k + 1) * steps) as u32, seconds: t0.elapsed().as_secs_f64() })
        })?;
        let (a, b) = Flow::overlap(l);
        let pl = DevBuf::f32(&ops.gpu, (b - a) * LATENT)?;
        pl.copy_within(0, &lat, a * LATENT * 4, (b - a) * LATENT * 4)?;
        let pc = DevBuf::f32(&ops.gpu, (b - a) * 2048)?;
        pc.copy_within(0, &cond, a * 2048 * 4, (b - a) * 2048 * 4)?;
        st.prev = Some((pl, pc));
        if let Some(kp) = keep {
            kp.push(lat.to_f32()?);
        }
        let wav = self.voc.decode(ops, nsd, &lat, l)?;
        let len = l * HOP;
        let left = if k == 0 { 0 } else { CROP_LEFT * HOP };
        let right = if last { 0 } else { CROP_RIGHT * HOP };
        for (c, o) in st.out.iter_mut().enumerate() {
            if left + right < len {
                o.extend_from_slice(&wav[c * len + left..c * len + len - right]);
            }
        }
        progress(Step { phase: "decode", at: k as u32 + 1, of: windows.max(k + 1) as u32, seconds: t0.elapsed().as_secs_f64() })
    }

    /// The two-GPU song: the language model's frames on the first GPU, each handed to the second as it is made; a
    /// window's flow and decoder start there once its frames are in (and one more, so whether it is the last is
    /// known), the rest after the last frame. The windows, their noise and their order are `sound`'s
    #[allow(clippy::too_many_arguments)]
    fn pipelined(&self, ids: &[u32], unc: &[u32], max: usize, seed: u64, steps: usize, cfg: f32, t0: Instant,
                 progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<[Vec<f32>; 2]> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc::channel;
        let (fops, _) = self.fo();
        // the rows on the flow's GPU, as they come (None: the frames are done, Some(n) their count)
        let (rtx, rrx) = channel::<std::result::Result<Vec<f32>, usize>>();
        let (mtx, mrx) = channel::<FlowMsg>();
        let stop = AtomicBool::new(false);
        std::thread::scope(|sc| {
            let stop = &stop;
            sc.spawn(move || {
                let r = (|| -> Result<[Vec<f32>; 2]> {
                    let frames = DevBuf::f32(&fops.gpu, max.max(1) * HIDDEN)?;
                    let mut rng = Rng::new(seed.wrapping_add(0x5eed));
                    let mut noise = |_: usize, len: usize| -> Result<Vec<f32>> { Ok((0..len * LATENT).map(|_| rng.normal()).collect()) };
                    let mut report = |s: Step| -> Result<()> {
                        if stop.load(Ordering::Relaxed) {
                            return Err(Error("cancelled".into()));
                        }
                        let _ = mtx.send(FlowMsg::Step(s));
                        Ok(())
                    };
                    let windows = Flow::starts(max).len();
                    let mut st = Sounding::default();
                    let (mut have, mut k) = (0usize, 0usize);
                    let mut total: Option<usize> = None;
                    loop {
                        // the next window when its frames and one more are in, or all of them once the frames are done
                        let s0 = k * flow::CHUNK_HOP;
                        match total {
                            Some(n) => {
                                let starts = Flow::starts(n);
                                if k >= starts.len() {
                                    return Ok(st.out);
                                }
                                self.window(&mut st, k, s0, (s0 + flow::CHUNK_FRAMES).min(n), k + 1 == starts.len(), &frames, steps, cfg, &mut noise, t0,
                                            starts.len(), &mut report, None)?;
                                k += 1;
                                continue;
                            }
                            None if have > s0 + flow::CHUNK_FRAMES => {
                                self.window(&mut st, k, s0, s0 + flow::CHUNK_FRAMES, false, &frames, steps, cfg, &mut noise, t0, windows, &mut report, None)?;
                                k += 1;
                                continue;
                            }
                            None => {}
                        }
                        match rrx.recv() {
                            Ok(Ok(rows)) => {
                                frames.write(have * HIDDEN * 4, &bytes(&rows))?;
                                have += rows.len() / HIDDEN;
                            }
                            Ok(Err(n)) => total = Some(n),
                            Err(_) => return Err(Error("cancelled".into())),
                        }
                    }
                })();
                let _ = mtx.send(FlowMsg::Done(r));
            });
            let forward = |progress: &mut dyn FnMut(Step) -> Result<()>| -> Result<Option<Result<[Vec<f32>; 2]>>> {
                while let Ok(m) = mrx.try_recv() {
                    match m {
                        FlowMsg::Step(s) => progress(s)?,
                        FlowMsg::Done(r) => return Ok(Some(r)),
                    }
                }
                Ok(None)
            };
            let mut early: Option<Result<[Vec<f32>; 2]>> = None;
            let made = {
                let mut send = |row: &[f32]| -> Result<()> {
                    rtx.send(Ok(row.to_vec())).map_err(|_| Error("the flow's thread ended".into()))
                };
                let mut prog = |s: Step| -> Result<()> {
                    progress(s)?;
                    if early.is_none() {
                        early = forward(progress)?;
                        // the flow failed: the frames stop too
                        if let Some(Err(e)) = &early {
                            return Err(Error(e.0.clone()));
                        }
                    }
                    Ok(())
                };
                self.frames_to(ids, unc, max, seed, t0, &mut prog, Some(&mut send))
            };
            let n = match made {
                Ok((_, n)) => n,
                Err(e) => {
                    stop.store(true, Ordering::Relaxed);
                    drop(rtx);
                    return Err(e);
                }
            };
            let _ = rtx.send(Err(n));
            if let Some(r) = early {
                return r;
            }
            loop {
                match mrx.recv() {
                    Ok(FlowMsg::Step(s)) => {
                        if let Err(e) = progress(s) {
                            stop.store(true, Ordering::Relaxed);
                            return Err(e);
                        }
                    }
                    Ok(FlowMsg::Done(r)) => return r,
                    Err(_) => return Err(Error("the flow's thread ended".into())),
                }
            }
        })
    }
}

/// Floats as their bytes
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn tok_path(files: &ModelFiles) -> String {
    files.get("tokenizer").map(|p| p.display().to_string()).unwrap_or_default()
}

impl AudioEngine for MiniMaxMusic3 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { seconds: 60.0, max_seconds: MAX_FRAMES as f32 / FPS, steps: 30, cfg: flow::GUIDANCE, rate: RATE }
    }
    fn lyrics(&self) -> bool {
        true
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, req: &AudioRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        let d = self.defaults();
        let t0 = Instant::now();
        let lyrics = req.lyrics.as_deref().unwrap_or("[instrumental]");
        let (ids, unc) = self.tokens(&req.prompt, lyrics)?;
        let seconds = req.seconds.unwrap_or(d.seconds);
        if seconds <= 0.0 {
            return Err(Error(format!("{seconds} s: a length above zero")));
        }
        let max = ((seconds * FPS) as usize).clamp(1, MAX_FRAMES);
        let steps = req.steps.unwrap_or(d.steps).max(1) as usize;
        let cfg = req.cfg.unwrap_or(d.cfg);
        progress(Step { phase: "prompt", at: 0, of: 1, seconds: 0.0 })?;
        let [l, r] = if self.second.is_some() {
            self.pipelined(&ids, &unc, max, req.seed, steps, cfg, t0, progress)?
        } else {
            let (frames, n) = self.frames(&ids, &unc, max, req.seed, t0, progress)?;
            let mut rng = Rng::new(req.seed.wrapping_add(0x5eed));
            self.sound(&frames, n, steps, cfg, &mut |_, len| Ok((0..len * LATENT).map(|_| rng.normal()).collect()), t0, progress, None)?
        };
        let samples = l.iter().zip(&r).flat_map(|(a, b)| [a.clamp(-1.0, 1.0), b.clamp(-1.0, 1.0)]).collect();
        Ok(Audio { rate: RATE, channels: 2, samples })
    }
    fn report(&self) -> Vec<String> {
        let on = match &self.second {
            Some((o, _)) => format!("{} (the language model) and {} (the flow, the decoder)", self.ops.gpu.name, o.gpu.name),
            None => self.ops.gpu.name.clone(),
        };
        vec![format!("{ARCH} on {on} ({}, loaded {:.0} s ago)", if self.ar.int8 { "int8" } else { "half" }, self.loaded.elapsed().as_secs_f64())]
    }
}

/// Whether `p` is this model's directory (the checkpoint's diffusers layout): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    let list = |sub: &str| -> Vec<std::path::PathBuf> {
        let mut v: Vec<_> = std::fs::read_dir(dir.join(sub)).into_iter().flatten().flatten().map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
        v.sort();
        v
    };
    for (role, sub) in [("language-model", "language_model"), ("transformer", "transformer"), ("depth-decoder", "rvq_depth_decoder"),
                        ("condition-encoder", "condition_encoder"), ("vocoder", "vocoder")] {
        let v = list(sub);
        if v.is_empty() {
            return Err(Error(format!("{}: no {sub}/*.safetensors", dir.display())));
        }
        if role.ends_with("model") || role == "transformer" {
            for (i, p) in v.into_iter().enumerate() {
                f.insert(format!("{role}-{}", i + 1), p);
            }
        } else {
            f.insert(role.to_string(), v[0].clone());
        }
    }
    f.insert("tokenizer".into(), dir.join("tokenizer/tokenizer.json"));
    Ok(f)
}
