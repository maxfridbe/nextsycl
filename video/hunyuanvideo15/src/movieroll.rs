//! MovieRoll (Tencent, yunyangge/MovieRoll; code github Tencent-Hunyuan/MovieRoll 0b5f73e2): long films by rolling
//! diffusion on HunyuanVideo-1.5's DiT (a full fine-tune, the same 8.3B; its files fuse q / k / v), its VAE and its
//! text encoders - this crate's, MovieRoll's three passes in roll.rs.
//!
//! ```text
//!   chunks of 4 latent frames (16 video frames, 0.67 s); text to video: the first chunk 5 (the leftover frame), image
//!   to video: the first chunk the picture's one latent frame (the anchor, cached before the first round)
//!   a window of 5 chunks at rising noise; each round: pop the oldest chunk (clean) -> the film; put it into the KV
//!   cache (t = 0, context noise first) or rebuild the cache (KV recompute: at 18 cached chunks, and at every event
//!   switch) from the first chunk, the middle of every event and the 3 newest; push a fresh noisy chunk; denoise the
//!   window (each chunk at its own step of the ladder, the text of the cleanest chunk's event)
//!   distilled: the ladder [1000, 800, 600, 400, 200] shifted by 5, one Euler step a round, no guidance
//!   base: 50 steps (10 a round), AnnealCFG 1 -> 4 over an empty negative prompt, context noise 0.05
//!   each popped chunk -> the streaming VAE decode (vae.rs stream_*: exact, causal in time) -> ffmpeg as it comes
//! ```
//!
//! The loop is SGLang-MovieRoll's `rolling.loop.run_rolling` (the port upstream ships, which matches its own
//! pipelines), in Rust. Events: an ordered list of prompts split evenly over the film (upstream's even split; a
//! prompt string with `@@@EVENT_BREAK@@@` lines - upstream's rewrite skill's batch form - or a list).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::Picture;
use nextsycl_video::{At, Clip, Defaults, EngineOption, Error, JobCtl, LoadOptions, ModelFiles, Progress, Result, Sampler, Schedule, VideoEngine,
                     VideoKind, VideoRequest};
use serde_json::{json, Value};

use crate::dit::{self, Attn, LATENT};
use crate::roll::{Kv, Pass, RollBuf};
use crate::{Encoded, HunyuanVideo15, GIB};

pub const ARCH: &str = "movieroll";
pub const ROLES: &[&str] = crate::ROLES;
/// upstream's event separator (agent/streaming-video-prompt-rewrite: batch multi-event output)
pub const EVENT_BREAK: &str = "@@@EVENT_BREAK@@@";
pub const CHUNK: usize = 4;
pub const WINDOW: usize = 5;
pub const FULL_HISTORY: usize = 6;
pub const FPS: usize = 24;
/// the longest film: 60 s (what the paper evaluates; 90 chunks)
pub const MAX_SECONDS: f32 = 60.0;

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "mode", env: "NS_MR_MODE", value: "distilled|base", help: "the sampling chain of the card's file: distilled (5 steps) or base (50, AnnealCFG)", at: At::Load },
    EngineOption { name: "context-noise", env: "NS_MR_CONTEXT_NOISE", value: "X", help: "the clean history's context noise sigma (distilled t2v 0.15, i2v 0, base 0.05)", at: At::Both },
    EngineOption { name: "threshold", env: "NS_MR_THRESHOLD", value: "N", help: "cached chunks that trigger a KV recompute (18)", at: At::Both },
    EngineOption { name: "attention", env: "NS_MR_ATTN", value: "KERNEL", help: "the attention: half (default), sage (int8 q and k) or shared", at: At::Both },
    EngineOption { name: "kv", env: "NS_MR_KV", value: "auto|half|int8",
                   help: "the history's KV cache: half, int8 (a scale a row and head: half the memory), auto (half when it fits the card with room for the decode)",
                   at: At::Both },
    EngineOption { name: "image", env: "NS_MR_IMAGE", value: "0|1", help: "load the VAE's encoder (the image-to-video card's: the picture is the first chunk)", at: At::Load },
];

/// What a film keeps its KV cache in
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMode {
    Auto,
    Half,
    Int8,
}

impl KvMode {
    pub fn parse(v: &str) -> std::result::Result<KvMode, String> {
        match v.trim() {
            "" | "auto" => Ok(KvMode::Auto),
            "half" => Ok(KvMode::Half),
            "int8" => Ok(KvMode::Int8),
            x => Err(format!("kv={x}: auto, half or int8")),
        }
    }
}

pub fn kind() -> VideoKind {
    VideoKind { archs: &[ARCH], name: "MovieRoll (HunyuanVideo-1.5 fine-tune: films by rolling diffusion, up to 60 s at 480p, events)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn VideoEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(MovieRoll::load(files, gpu, o, log)?))
}

/// A chain's settings (movieroll/commons/profiles.py)
#[derive(Clone, Debug)]
pub struct Chain {
    pub distilled: bool,
    /// the shifted timesteps (sigma x 1000) and sigmas (one more: 0)
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub steps_per_window: usize,
    pub context_noise: f32,
    pub threshold: usize,
    pub recent_k: usize,
    pub guided: bool,
    pub cfg: (f32, f32),
    /// distilled: recompute at an event switch even when nothing is dropped; the anchor noised like the history
    pub force_at_event: bool,
    pub noise_anchor: bool,
}

impl Chain {
    pub fn distilled(context_noise: f32, threshold: usize) -> Chain {
        let sig: Vec<f32> = [1000f32, 800.0, 600.0, 400.0, 200.0].iter().map(|t| shift(t / 1000.0, 5.0)).collect();
        let mut sigmas = sig.clone();
        sigmas.push(0.0);
        Chain { distilled: true, timesteps: sig.iter().map(|s| s * 1000.0).collect(), sigmas, steps_per_window: 1, context_noise, threshold, recent_k: 3,
                guided: false, cfg: (1.0, 1.0), force_at_event: true, noise_anchor: true }
    }

    pub fn base(steps: usize, context_noise: f32, threshold: usize) -> Chain {
        let s: Vec<f32> = (0..=steps).map(|i| shift(1.0 - i as f32 / steps as f32, 5.0)).collect();
        Chain { distilled: false, timesteps: s[..steps].iter().map(|x| x * 1000.0).collect(), sigmas: s, steps_per_window: steps / WINDOW, context_noise,
                threshold, recent_k: 3, guided: true, cfg: (1.0, 4.0), force_at_event: false, noise_anchor: false }
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }
}

fn shift(s: f32, k: f32) -> f32 {
    k * s / (1.0 + (k - 1.0) * s)
}

/// Each window slot's step index at global step `g` (StreamingSchedulerWrapper / WindowSchedule): slot 0 is the oldest
/// chunk
pub fn step_index(g: usize, steps: usize, window: usize) -> Vec<usize> {
    let spw = steps / window;
    let step = g % steps;
    let offset = (g + 1).saturating_sub(steps).div_ceil(spw) % window;
    (0..window).map(|i| {
        let w = (offset + i) % window;
        (step + steps - (w * spw) % steps) % steps
    }).collect()
}

/// The chunk layout of `latent` frames (latent % 4 == 1): text to video [5, 4, 4, ...], image to video [1, 4, ...]
pub fn chunk_sizes(latent: usize, anchored: bool) -> Result<Vec<usize>> {
    if latent % CHUNK != 1 || latent < 1 + CHUNK {
        return Err(Error(format!("{latent} latent frames: 4n + 1, n >= 1")));
    }
    let first = if anchored { 1 } else { 1 + CHUNK };
    let mut v = vec![first];
    v.extend(std::iter::repeat_n(CHUNK, (latent - first) / CHUNK));
    Ok(v)
}

/// Where each event starts (chunk indices), the events that keep a chunk: geometry.event_split_indices
pub fn event_splits(chunks: usize, events: usize) -> (Vec<usize>, Vec<usize>) {
    if events < 2 {
        return (Vec::new(), vec![0]);
    }
    let splits: Vec<usize> = (1..events).map(|e| chunks * e / events).collect();
    let mut bounds = vec![0];
    bounds.extend(&splits);
    bounds.push(chunks);
    let kept: Vec<usize> = (0..events).filter(|e| bounds[*e] < bounds[e + 1]).collect();
    let dedup = kept[1..].iter().map(|e| splits[e - 1]).collect();
    (dedup, kept)
}

/// The chunks a KV recompute keeps: the first, the middle of every event, the `recent` newest
pub fn recompute_chunks(done: usize, splits: &[usize], recent: usize) -> Vec<usize> {
    let mut s = std::collections::BTreeSet::new();
    s.insert(0);
    let mut b = vec![0];
    b.extend(splits.iter().copied().filter(|x| *x > 0 && *x < done));
    b.push(done);
    for w in b.windows(2) {
        if w[0] < w[1] {
            s.insert((w[0] + w[1] - 1) / 2);
        }
    }
    s.extend(done.saturating_sub(recent)..done);
    s.into_iter().collect()
}

/// The events of a prompt: a JSON list of strings, or lines `@@@EVENT_BREAK@@@` between them, else one
pub fn events_of(prompt: &str) -> Vec<String> {
    let p = prompt.trim();
    if p.starts_with('[') {
        if let Ok(v) = serde_json::from_str::<Vec<String>>(p) {
            let v: Vec<String> = v.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            if !v.is_empty() {
                return v;
            }
        }
    }
    let v: Vec<String> = p.split(EVENT_BREAK).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if v.is_empty() {
        vec![String::new()]
    } else {
        v
    }
}

/// The frames of a film of `seconds`: snapped up to 16n + 1 (n chunks) as the studio snaps them, at least one chunk,
/// at most 60 s
pub fn frames_for(seconds: f32) -> usize {
    let f = ((seconds.clamp(0.0, MAX_SECONDS) * FPS as f32).round() as usize).max(1);
    16 * (f - 1).div_ceil(16).clamp(1, 90) + 1
}

pub struct MovieRoll {
    pub hv: HunyuanVideo15,
    pub mode_distilled: bool,
    context_noise: f32,
    threshold: usize,
    attention: String,
    kv: KvMode,
}

/// The film's pictures as they come: ffmpeg fed decoded frames, an mp4 at the end
pub struct Sink {
    child: Child,
    w: usize,
    h: usize,
    pub frames: usize,
}

impl Sink {
    pub fn open(path: &Path, w: usize, h: usize) -> Result<Sink> {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| Error(format!("{}: {e}", d.display())))?;
        }
        let child = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &FPS.to_string(), "-i", "-",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| Error(format!("starting ffmpeg: {e}")))?;
        Ok(Sink { child, w, h, frames: 0 })
    }

    /// frames float32 [n, h, w, 3] in [-1, 1]
    pub fn push(&mut self, px: &[f32]) -> Result<()> {
        let plane = self.h * self.w * 3;
        let stdin = self.child.stdin.as_mut().ok_or_else(|| Error("ffmpeg has no stdin".into()))?;
        let rgb: Vec<u8> = px.iter().map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8).collect();
        stdin.write_all(&rgb).map_err(|e| Error(format!("writing frames to ffmpeg: {e}")))?;
        self.frames += px.len() / plane;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        let st = self.child.wait().map_err(|e| Error(e.to_string()))?;
        if !st.success() {
            return Err(Error(format!("ffmpeg failed ({st})")));
        }
        Ok(())
    }
}

/// What a film run reports
pub struct Film {
    pub latents: Vec<f32>,
    pub chunks: Vec<usize>,
    pub recomputes: Vec<usize>,
    pub denoise_s: f64,
    pub decode_s: f64,
    pub rounds: usize,
    pub peak: u64,
    pub kv_gib: f64,
}

/// Where a film's noise comes from: the seed's torch-like CPU stream (bfloat16 draws, the distilled chain's), or a
/// reference's dumps (checks)
pub trait NoiseSource {
    /// a fresh noisy chunk float32 [frames h w, 32]
    fn chunk(&mut self, n: usize) -> Result<Vec<f32>>;
    /// a context-noise draw (eps) of n values in the same layout
    fn ctx(&mut self, n: usize) -> Result<Vec<f32>>;
}

/// torch.randn(shape [1, 32, F, h, w], generator, dtype=bfloat16) on the CPU, one generator for the film, in call order
pub struct SeedNoise {
    g: nextsycl_diffusion::torch_rng::Mt19937,
}

impl SeedNoise {
    pub fn new(seed: u64) -> SeedNoise {
        SeedNoise { g: nextsycl_diffusion::torch_rng::Mt19937::new(seed) }
    }

    fn draw(&mut self, n: usize) -> Vec<f32> {
        // [32, rows] channel-first as torch draws it, to channels last
        let v = self.g.randn(n);
        let rows = n / LATENT;
        let bf = |x: f32| -> f32 { f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff_0000) };
        let mut out = vec![0f32; n];
        for c in 0..LATENT {
            for p in 0..rows {
                out[p * LATENT + c] = bf(v[c * rows + p]);
            }
        }
        out
    }
}

impl NoiseSource for SeedNoise {
    fn chunk(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.draw(n))
    }
    fn ctx(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.draw(n))
    }
}

/// Each chunk leaving the window: (its index in the film, its latents float32 [f h w, 32], f)
pub type ChunkOut<'a> = &'a mut dyn FnMut(usize, &[f32], usize) -> Result<()>;

/// Every model call of a film, for a check against a reference's (kind, the latents in, the velocity out)
pub type OpTap<'a> = &'a mut dyn FnMut(&str, &[f32], Option<&[f32]>) -> Result<()>;

impl MovieRoll {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<MovieRoll> {
        let setting = |name: &str| o.setting(name).filter(|v| !v.is_empty());
        let mode = setting("NS_MR_MODE").map_or_else(|| "distilled".to_string(), |v| v.to_string());
        if mode != "distilled" && mode != "base" {
            return Err(Error(format!("NS_MR_MODE={mode}: distilled or base")));
        }
        let num = |name: &str, d: f32| -> Result<f32> { setting(name).map_or(Ok(d), |v| v.parse().map_err(|_| Error(format!("{name}={v}: a number")))) };
        let context_noise = num("NS_MR_CONTEXT_NOISE", if mode == "base" { 0.05 } else { 0.15 })?;
        let threshold = num("NS_MR_THRESHOLD", 18.0)? as usize;
        let attention = setting("NS_MR_ATTN").map_or_else(String::new, |v| v.to_string());
        let kv = KvMode::parse(&setting("NS_MR_KV").unwrap_or_default()).map_err(Error)?;
        let image = setting("NS_MR_IMAGE").is_some_and(|v| v == "1");
        if !(0.0..1.0).contains(&context_noise) || !(2..=40).contains(&threshold) {
            return Err(Error(format!("{ARCH}: context noise {context_noise} (0 to 1), threshold {threshold} (2 to 40)")));
        }
        let hv = HunyuanVideo15::load_with(files, gpu, o, image, log)?;
        log(format!("{ARCH}: {mode}, context noise {context_noise}, recompute at {threshold} chunks"));
        Ok(MovieRoll { hv, mode_distilled: mode == "distilled", context_noise, threshold, attention, kv })
    }

    fn chain(&self, steps: Option<usize>, context_noise: f32, threshold: usize) -> Result<Chain> {
        if self.mode_distilled {
            if steps.is_some_and(|s| s != 5) {
                return Err(Error(format!("{ARCH}: the distilled chain has its fixed 5-step ladder")));
            }
            Ok(Chain::distilled(context_noise, threshold))
        } else {
            let s = steps.unwrap_or(50);
            if s == 0 || !s.is_multiple_of(WINDOW) || s > 100 {
                return Err(Error(format!("{ARCH}: base steps {s}: a multiple of {WINDOW}, at most 100")));
            }
            Ok(Chain::base(s, context_noise, threshold))
        }
    }

    /// A film: `events` encoded, `anchor` the image-to-video picture's latent (float32 [h w, 32]), `latent_frames`
    /// 4n + 1, each popped chunk handed to `out(chunk index, latents float32 [f h w, 32], frames)`
    #[allow(clippy::too_many_arguments)]
    pub fn film(&self, chain: &Chain, events: &[Encoded], negative: Option<&Encoded>, anchor: Option<&[f32]>, latent_frames: usize, hw: (usize, usize),
                attn: Attn, kv_mode: KvMode, noise: &mut dyn NoiseSource, progress: &mut dyn FnMut(usize, usize) -> Result<()>,
                out: ChunkOut, tap: Option<OpTap>, log: &mut dyn FnMut(String)) -> Result<Film> {
        let hv = &self.hv;
        let nsd = &hv.nsd;
        let gpu = &nsd.gpu;
        let tpf = hw.0 * hw.1;
        let sizes = chunk_sizes(latent_frames, anchor.is_some())?;
        let (splits, kept) = event_splits(sizes.len(), events.len());
        if kept.len() != events.len() {
            return Err(Error(format!("{ARCH}: {} events for {} chunks: every event needs a chunk", events.len(), sizes.len())));
        }
        let mut to_event = vec![0usize; sizes.len()];
        for (e, b) in splits.iter().enumerate() {
            for c in to_event.iter_mut().skip(*b) {
                *c = e + 1;
            }
        }
        let steps = chain.steps();
        // step rows: the ladder's timesteps, then 0 (the clean history's)
        let mut ts: Vec<(f32, Option<f32>)> = chain.timesteps.iter().map(|t| (*t, None)).collect();
        ts.push((0.0, None));
        let st = hv.dit.steps(nsd, &ts)?;
        let zero_row = steps;
        // the cache: the full-history blocks up to the threshold's chunks (or what an event-heavy recompute keeps: the
        // first chunk, every event's middle, the newest), the sink-recent ones the sink and one recent chunk; a
        // recompute's pass holds the chunks it keeps
        let keep = 1 + events.len() + chain.recent_k;
        let full_frames = (1 + CHUNK) + (keep.max(chain.threshold) - 1) * CHUNK;
        let recompute_frames = (1 + CHUNK) + (keep - 1) * CHUNK;
        let sr_frames = (1 + CHUNK) + CHUNK;
        let win_frames = WINDOW * CHUNK + 1;
        if full_frames + win_frames > crate::roll::MAX_FRAMES {
            return Err(Error(format!("{ARCH}: {} events and a threshold of {} need {} cached frames: past the RoPE grid", events.len(), chain.threshold,
                                     full_frames)));
        }
        let tq = events.iter().chain(negative).map(|e| e.text.tq).max().unwrap_or(1);
        let tt = events.iter().chain(negative).map(|e| e.text.tokens()).max().unwrap_or(1);
        // the room first (the driver has no out-of-memory error, and a card nearly full hangs the box): the cache, the
        // pass's buffers (its rows x 44 KB, the attention's keys and values x 8 KB) and 1.5 GiB to spare
        let blocks = hv.dit.doubles.len();
        let rows = win_frames.max(recompute_frames) * tpf + tt;
        let buf_need = rows * (dit::DIM * 4 + dit::DIM * 2 * 3 + dit::DIM * 3 * 2 + dit::FFN * 2) + ((full_frames + win_frames) * tpf + tt) * dit::DIM * 4;
        // besides: what a pass allocates as it goes (the final layer's rows, the RoPE tables) and the streaming decode of a
        // latent frame - 5.3 GiB over the cache and buffers measured at 864 x 480 on the B65, so 5.5 - and 1.5 GiB never touched
        let reserve = (11u64 << 29) + (3u64 << 29);
        let free = gpu.memory().ok().and_then(|m| m.1);
        let need = |int8: bool| (Kv::need(blocks, FULL_HISTORY, tpf, full_frames, sr_frames, int8) + buf_need) as u64 + reserve;
        let int8 = match kv_mode {
            KvMode::Half => false,
            KvMode::Int8 => true,
            KvMode::Auto => free.is_some_and(|f| need(false) > f),
        };
        if let Some(f) = free {
            if need(int8) > f {
                return Err(Error(format!("{ARCH}: the film's cache ({}) and buffers need {:.1} GiB with 7 for the decode and to spare; {:.1} GiB are free on {}",
                                         if int8 { "int8" } else { "half" }, need(int8) as f64 / GIB, f as f64 / GIB, gpu.name)));
            }
        }
        let mut kv = Kv::new(gpu, blocks, FULL_HISTORY, tpf, full_frames, sr_frames, int8)?;
        let rb = RollBuf::new(&hv.dit, gpu, tpf, win_frames.max(recompute_frames), full_frames + win_frames, tt, tq)?;
        let kv_gib = (kv.bytes() + rb.bytes()) as f64 / GIB;
        log(format!("{ARCH}: {} chunks {:?}, {} events (switches at chunks {:?}), {} steps, the cache ({}) and buffers {:.1} GiB", sizes.len(),
                    &sizes[..sizes.len().min(3)], events.len(), splits, steps, if kv.int8() { "int8" } else { "half" }, kv_gib));
        let ctx = |clean: &[f32], noise: &mut dyn NoiseSource| -> Result<Vec<f32>> {
            if chain.context_noise <= 0.0 {
                return Ok(clean.to_vec());
            }
            let e = noise.ctx(clean.len())?;
            let s = chain.context_noise;
            Ok(clean.iter().zip(&e).map(|(c, e)| (1.0 - s) * c + s * e).collect())
        };
        let mut tap = tap;
        let mut call = |name: &str, x: &[f32], v: Option<&[f32]>| -> Result<()> {
            if let Some(t) = tap.as_mut() {
                t(name, x, v)?;
            }
            Ok(())
        };
        let zero_spans = |frames: &[usize]| -> Vec<(usize, usize)> { frames.iter().map(|f| (*f, zero_row)).collect() };
        let mut finalized: Vec<Vec<f32>> = Vec::new();
        let mut recomputes = Vec::new();
        let mut sizes = sizes;
        let t0 = Instant::now();
        let decode_s = std::cell::Cell::new(0.0f64);
        let mut emit = |i: usize, lat: &[f32], f: usize, finalized: &mut Vec<Vec<f32>>| -> Result<()> {
            let t = Instant::now();
            out(i, lat, f)?;
            finalized.push(lat.to_vec());
            decode_s.set(decode_s.get() + t.elapsed().as_secs_f64());
            Ok(())
        };
        if let Some(a) = anchor {
            if a.len() != tpf * LATENT || sizes[0] != 1 {
                return Err(Error(format!("{ARCH}: the anchor is one latent frame")));
            }
            emit(0, a, 1, &mut finalized)?;
            let x = if chain.noise_anchor { ctx(a, noise)? } else { a.to_vec() };
            call("encode", &x, None)?;
            let xb = DevBuf::from_f32(gpu, &x)?;
            hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &zero_spans(&[1]), hw, Pass::Encode, attn)?;
            sizes.remove(0);
            to_event.remove(0);
        }
        let num = sizes.len();
        if num == 0 {
            return Err(Error("nothing to generate".into()));
        }
        // the window: chunks in order, float32 [frames h w, 32] each
        let mut queue: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut decoded = 0;
        let mut g = 0;
        let mut rounds = 0;
        for round in 0..num + WINDOW {
            rounds += 1;
            if round >= WINDOW {
                let (f, clean) = queue.remove(0);
                let last = decoded == num - 1;
                let idx = finalized.len();
                emit(idx, &clean, f, &mut finalized)?;
                decoded += 1;
                progress(decoded, num)?;
                if last {
                    break;
                }
                // into the cache: a plain encode, or a rebuild at the threshold or an event switch
                let done = finalized.len();
                let switch = !splits.is_empty() && splits.contains(&done);
                let mut rebuild = switch || kv.chunks() >= chain.threshold;
                let mut chosen = Vec::new();
                if rebuild {
                    chosen = recompute_chunks(done, &splits, chain.recent_k);
                    if chosen.len() >= done && !(switch && chain.force_at_event) {
                        rebuild = false;
                    }
                }
                if rebuild {
                    let frames: Vec<usize> = chosen.iter().map(|i| finalized[*i].len() / (tpf * LATENT)).collect();
                    let hist: Vec<f32> = chosen.iter().flat_map(|i| finalized[*i].iter().copied()).collect();
                    let x = ctx(&hist, noise)?;
                    call("recompute", &x, None)?;
                    let xb = DevBuf::from_f32(gpu, &x)?;
                    hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &zero_spans(&frames), hw, Pass::Recompute, attn)?;
                    recomputes.push(done);
                    log(format!("{ARCH}: KV recompute at {done} chunks ({}): kept {chosen:?}", if switch { "event switch" } else { "threshold" }));
                } else {
                    let x = ctx(&clean, noise)?;
                    call("encode", &x, None)?;
                    let xb = DevBuf::from_f32(gpu, &x)?;
                    hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &zero_spans(&[f]), hw, Pass::Encode, attn)?;
                }
            }
            let f = if round < num { sizes[round] } else { sizes[num - 1] };
            queue.push((f, noise.chunk(f * tpf * LATENT)?));
            let event = &events[to_event[decoded.min(num - 1)]];
            for _ in 0..chain.steps_per_window {
                let idx = step_index(g, steps, WINDOW);
                let k = queue.len();
                let spans: Vec<(usize, usize)> = queue.iter().zip(&idx[..k]).map(|((f, _), s)| (*f, *s)).collect();
                let lat: Vec<f32> = queue.iter().flat_map(|(_, l)| l.iter().copied()).collect();
                let lb = DevBuf::from_f32(gpu, &lat)?;
                let tsi = spans[0].1;
                let mut v = hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &lb, &spans, hw, Pass::Denoise { text: &event.text, tsi }, attn)?
                    .ok_or_else(|| Error("no velocity".into()))?
                    .to_f32()?;
                if let (true, Some(neg)) = (chain.guided, negative) {
                    // AnnealCFG: v_neg + w(t) (v - v_neg), w linear in each chunk's timestep
                    let vn = hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &lb, &spans, hw, Pass::Denoise { text: &neg.text, tsi }, attn)?
                        .ok_or_else(|| Error("no velocity".into()))?
                        .to_f32()?;
                    let mut r = 0;
                    for (f, s) in &spans {
                        let w = chain.cfg.0 + (chain.cfg.1 - chain.cfg.0) * chain.timesteps[*s] / 1000.0;
                        let rows = f * tpf * LATENT;
                        for (p, n) in v[r..r + rows].iter_mut().zip(&vn[r..r + rows]) {
                            *p = n + w * (*p - n);
                        }
                        r += rows;
                    }
                }
                call("denoise", &lat, Some(&v))?;
                // Euler, chunk by chunk at its own sigma
                let mut r = 0;
                for ((f, l), s) in queue.iter_mut().zip(&idx[..k]) {
                    let dt = chain.sigmas[s + 1] - chain.sigmas[*s];
                    let rows = *f * tpf * LATENT;
                    for (x, d) in l.iter_mut().zip(&v[r..r + rows]) {
                        *x += d * dt;
                    }
                    r += rows;
                }
                g += 1;
            }
        }
        let decode_s = decode_s.get();
        let denoise_s = t0.elapsed().as_secs_f64() - decode_s;
        let chunks: Vec<usize> = finalized.iter().map(|c| c.len() / (tpf * LATENT)).collect();
        Ok(Film { latents: finalized.concat(), chunks, recomputes, denoise_s, decode_s, rounds, peak: 0, kv_gib })
    }
}

impl MovieRoll {
    fn attn(&self, v: Option<&str>) -> Result<Attn> {
        Attn::parse(Some(v.unwrap_or(&self.attention)), self.hv.dit.tiles).map_err(Error)
    }

    /// The film's latent canvas from a request's size (480 x 864 or 864 x 480, MovieRoll's two) and seconds
    fn canvas(width: Option<usize>, height: Option<usize>) -> (usize, usize) {
        match (width, height) {
            (Some(w), Some(h)) if h > w => (480, 864),
            _ => (864, 480),
        }
    }

    /// The image-to-video anchor: the picture scaled to cover the canvas, cropped in the middle (Lanczos), the VAE
    /// encoder's mode x its scaling (utils/image_condition.py encode_input_image)
    fn anchor(&self, p: &Picture, w: usize, h: usize) -> Result<Vec<f32>> {
        let rgb = p.rgb();
        let px = crate::vision::cover(&rgb.data, rgb.width as usize, rgb.height as usize, w, h)?;
        self.hv.vae.encode(&self.hv.nsd, &px, h, w)
    }

    /// A film to `out`: every chunk decoded as it leaves the window (the streaming VAE, one latent frame a call)
    #[allow(clippy::too_many_arguments)]
    pub fn make(&self, prompt: &str, image: Option<&Picture>, width: Option<usize>, height: Option<usize>, seconds: f32, seed: u64, steps: Option<usize>,
                context_noise: Option<f32>, threshold: Option<usize>, attn: Attn, kv: KvMode, out: &Path, keep: Option<&mut Vec<f32>>,
                progress: &mut dyn FnMut(usize, usize) -> Result<()>, log: &mut dyn FnMut(String)) -> Result<(Film, usize, usize, usize)> {
        let hv = &self.hv;
        let (w, h) = Self::canvas(width, height);
        let hw = (h / 16, w / 16);
        let frames = frames_for(seconds);
        let latent = (frames - 1) / 4 + 1;
        let chain = self.chain(steps, context_noise.unwrap_or(self.context_noise), threshold.unwrap_or(self.threshold))?;
        let t0 = Instant::now();
        hv.peak.store(0, std::sync::atomic::Ordering::Relaxed);
        let texts = events_of(prompt);
        let mut events = Vec::with_capacity(texts.len());
        for t in &texts {
            events.push(hv.encode(t, None)?);
        }
        let negative = if chain.guided { Some(hv.encode("", None)?) } else { None };
        // the text encoders' 4.2 GiB back to the card for the film (the next prompt loads them again, ~5 s)
        hv.unload_text()?;
        let anchor = match image {
            Some(p) => Some(self.anchor(p, w, h)?),
            None => None,
        };
        log(format!("{ARCH}: {w}x{h}, {frames} frames ({} s, {latent} latent), {} event(s), seed {seed}, attention {attn:?}; text in {:.1} s",
                    frames as f32 / FPS as f32, events.len(), t0.elapsed().as_secs_f64()));
        let mut sink = Sink::open(out, w, h)?;
        hv.vae.stream_begin()?;
        let mut keep = keep;
        let mut noise = SeedNoise::new(seed);
        let tpf = hw.0 * hw.1;
        let film = {
            let mut emit = |_: usize, lat: &[f32], f: usize| -> Result<()> {
                if let Some(k) = keep.as_mut() {
                    k.extend_from_slice(lat);
                }
                for i in 0..f {
                    let (px, _) = hv.vae.stream_chunk(&hv.nsd, &lat[i * tpf * LATENT..(i + 1) * tpf * LATENT], 1, hw.0, hw.1)?;
                    sink.push(&px)?;
                }
                hv.note();
                Ok(())
            };
            self.film(&chain, &events, negative.as_ref(), anchor.as_deref(), latent, hw, attn, kv, &mut noise, progress, &mut emit, None, log)
        };
        hv.vae.stream_end()?;
        let mut film = film?;
        let written = sink.frames;
        sink.finish()?;
        film.peak = hv.peak.load(std::sync::atomic::Ordering::Relaxed);
        Ok((film, written, w, h))
    }

    fn generate_job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        let s = |k: &str| spec.get(k).and_then(Value::as_str);
        let n = |k: &str| spec.get(k).and_then(Value::as_f64);
        let prompt = match spec.get("prompt") {
            Some(Value::Array(a)) => serde_json::to_string(a).map_err(|e| Error(e.to_string()))?,
            Some(Value::String(p)) => p.clone(),
            _ => return Err(Error("generate: no prompt".into())),
        };
        let out = PathBuf::from(s("out").ok_or_else(|| Error("generate: no out".into()))?);
        let seconds = n("seconds").map_or(10.0, |x| x as f32);
        if !(0.0..=MAX_SECONDS).contains(&seconds) {
            return Err(Error(format!("{ARCH}: seconds {seconds}: up to {MAX_SECONDS}")));
        }
        let image = match s("image").or(s("first_frame")).filter(|p| !p.is_empty()) {
            Some(p) => Some(Picture::decode(&std::fs::read(p).map_err(|e| Error(format!("{p}: {e}")))?).map_err(|e| Error(format!("{p}: {e}")))?),
            None => None,
        };
        let attn = self.attn(s("attention"))?;
        let kv = s("kv").map_or(Ok(self.kv), KvMode::parse).map_err(Error)?;
        let t0 = Instant::now();
        let JobCtl { log, cancel, progress: report } = ctl;
        let mut progress = |at: usize, of: usize| -> Result<()> {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error("cancelled".into()));
            }
            if let Some(p) = report.as_mut() {
                p(at, of);
            }
            Ok(())
        };
        let mut lat = Vec::new();
        let gate = s("logits-out").is_some();
        let (film, frames, w, h) = self.make(&prompt, image.as_ref(), n("width").map(|x| x as usize), n("height").map(|x| x as usize), seconds,
                                             n("seed").map_or(0, |x| x as i64 as u64), n("steps").map(|x| x as usize), n("context_noise").map(|x| x as f32),
                                             n("threshold").map(|x| x as usize), attn, kv, &out, if gate { Some(&mut lat) } else { None }, &mut progress,
                                             &mut |l| log(l))?;
        let total = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: {} chunks in {} rounds: denoise {:.1} s ({:.2} s a chunk), decode {:.1} s, {frames} frames in {total:.1} s; recomputes at {:?}; \
                     the card's peak {:.1} GiB (cache + buffers {:.1} GiB)", film.chunks.len(), film.rounds, film.denoise_s,
                    film.denoise_s / film.chunks.len().max(1) as f64, film.decode_s, film.recomputes, film.peak as f64 / GIB, film.kv_gib));
        if let Some(p) = s("logits-out") {
            std::fs::write(p, lat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| Error(format!("{p}: {e}")))?;
            log(format!("[prompt {} latent chunks in {:.2} s ({:.1} tok/s)]", film.chunks.len(), film.denoise_s,
                        (film.latents.len() / LATENT) as f64 / film.denoise_s));
        }
        Ok(json!({"out": out, "width": w, "height": h, "frames": frames, "fps": FPS, "seconds": total, "chunks": film.chunks.len(), "rounds": film.rounds,
                  "recomputes": film.recomputes, "denoise_s": film.denoise_s, "decode_s": film.decode_s, "peak_gib": film.peak as f64 / GIB}))
    }
}

/// A reference's noise (reference/movieroll/ref.py: noise_NNN.npy, ctxeps_NNN.npy in draw order)
struct DumpNoise {
    dir: PathBuf,
    noise: usize,
    ctx: usize,
}

impl NoiseSource for DumpNoise {
    fn chunk(&mut self, n: usize) -> Result<Vec<f32>> {
        let (_, v) = crate::check::read_npy(&self.dir.join(format!("noise_{:03}.npy", self.noise)))?;
        self.noise += 1;
        if v.len() != n {
            return Err(Error(format!("noise draw {}: {} values, {n} wanted", self.noise - 1, v.len())));
        }
        Ok(v)
    }
    fn ctx(&mut self, n: usize) -> Result<Vec<f32>> {
        let (_, v) = crate::check::read_npy(&self.dir.join(format!("ctxeps_{:03}.npy", self.ctx)))?;
        self.ctx += 1;
        if v.len() != n {
            return Err(Error(format!("context noise draw {}: {} values, {n} wanted", self.ctx - 1, v.len())));
        }
        Ok(v)
    }
}

impl MovieRoll {
    /// The film of reference/movieroll/ref.py's roll.json made here with its noise: every model call's input and
    /// velocity against the reference's, then the film's latents; and the seed's noise stream against its draws
    fn check_job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        use crate::check::{compare, read_npy};
        let dir = PathBuf::from(spec.get("dir").and_then(Value::as_str).ok_or_else(|| Error("check: no dir".into()))?);
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("roll.json")).map_err(|e| Error(format!("roll.json: {e}")))?)
            .map_err(|e| Error(format!("roll.json: {e}")))?;
        let log = &mut *ctl.log;
        let hv = &self.hv;
        let lat = meta["latent"].as_array().ok_or_else(|| Error("roll.json: no latent".into()))?;
        let (lf, h, w) = (lat[0].as_u64().unwrap_or(0) as usize, lat[1].as_u64().unwrap_or(0) as usize, lat[2].as_u64().unwrap_or(0) as usize);
        let mut events = Vec::new();
        for e in meta["events"].as_array().into_iter().flatten() {
            let d = PathBuf::from(e.as_str().unwrap_or(""));
            let (_, q) = read_npy(&d.join("embeds.npy"))?;
            let b = d.join("byt5.npy");
            let b = if b.exists() { Some(read_npy(&b)?.1) } else { None };
            events.push(Encoded { text: hv.dit.text(&hv.nsd, &q, b.as_deref(), None)? });
        }
        let chain = Chain::distilled(meta["context_noise"].as_f64().unwrap_or(0.15) as f32, meta["threshold"].as_u64().unwrap_or(18) as usize);
        // the seed's own stream against the reference's draws
        let mut sn = SeedNoise::new(meta["seed"].as_u64().unwrap_or(0));
        let (_, n0) = read_npy(&dir.join("noise_000.npy"))?;
        let (rel, mx, _) = compare(&sn.chunk(n0.len())?, &n0);
        log(format!("seed noise against the reference's first draw: rel {rel:.2e} max {mx:.2e}{}", if mx == 0.0 { " (the same draws)" } else { "" }));
        let ops = meta["ops"].as_array().cloned().unwrap_or_default();
        let mut at = 0usize;
        let mut rows = Vec::new();
        let mut lines = Vec::new();
        let mut worst = 0f64;
        let mut tap = |kind: &str, x: &[f32], v: Option<&[f32]>| -> Result<()> {
            let op = ops.get(at).ok_or_else(|| Error(format!("call {at} ({kind}) past the reference's {}", ops.len())))?;
            if op["op"].as_str() != Some(kind) {
                return Err(Error(format!("call {at}: {kind} here, {} in the reference", op["op"])));
            }
            let base = dir.join(format!("op_{at:03}_{kind}"));
            let (_, xi) = read_npy(&PathBuf::from(format!("{}_in.npy", base.display())))?;
            let (ri, _, _) = compare(x, &xi);
            let mut line = format!("{at:3} {kind:<9} in rel {ri:.2e}");
            let mut rv = 0.0;
            if let Some(v) = v {
                let (_, want) = read_npy(&PathBuf::from(format!("{}_v.npy", base.display())))?;
                let (r, m, c) = compare(v, &want);
                rv = r;
                line += &format!("  velocity rel {r:.2e} max {m:.2e} cos {c:.6}");
            }
            worst = worst.max(rv);
            rows.push(json!({"op": at, "kind": kind, "in_rel": ri, "v_rel": rv}));
            lines.push(line);
            at += 1;
            Ok(())
        };
        // teacher-forced: every call of the reference made here on the reference's own input (the cache built from
        // the reference's encodes): each op's own error, with nothing carried from the ops before
        let kv = spec.get("kv").and_then(Value::as_str).map_or(Ok(self.kv), KvMode::parse).map_err(Error)?;
        let teacher = self.replay(&dir, &ops, &events, &chain, (h, w), self.attn(spec.get("attention").and_then(Value::as_str))?, kv == KvMode::Int8, log)?;
        let mut noise = DumpNoise { dir: dir.clone(), noise: 0, ctx: 0 };
        let mut sink = |_: usize, _: &[f32], _: usize| -> Result<()> { Ok(()) };
        let attn = self.attn(spec.get("attention").and_then(Value::as_str))?;
        let film = self.film(&chain, &events, None, None, lf, (h, w), attn, kv, &mut noise, &mut |_, _| Ok(()), &mut sink, Some(&mut tap), log)?;
        for l in lines {
            log(l);
        }
        let (_, want) = read_npy(&dir.join("latents.npy"))?;
        let (r, m, c) = compare(&film.latents, &want);
        log(format!("film latents: rel {r:.2e} max {m:.2e} cos {c:.6}; recomputes here {:?}, in the reference {}", film.recomputes, meta["recomputes"]));
        Ok(json!({"ops": rows, "teacher": teacher, "latents_rel": r, "latents_cos": c, "worst_velocity_rel": worst, "recomputes": film.recomputes}))
    }

    /// reference/movieroll/ref.py's calls one by one on its own inputs (roll.json's ops): each denoise's velocity
    /// against the reference's
    #[allow(clippy::too_many_arguments)]
    fn replay(&self, dir: &Path, ops: &[Value], events: &[Encoded], chain: &Chain, hw: (usize, usize), attn: Attn, int8: bool, log: &mut dyn FnMut(String))
              -> Result<Vec<Value>> {
        use crate::check::{compare, read_npy};
        let hv = &self.hv;
        let (nsd, gpu) = (&hv.nsd, &hv.nsd.gpu);
        let tpf = hw.0 * hw.1;
        let mut ts: Vec<(f32, Option<f32>)> = chain.timesteps.iter().map(|t| (*t, None)).collect();
        ts.push((0.0, None));
        let st = hv.dit.steps(nsd, &ts)?;
        let zero = chain.steps();
        let row = |t: f64| -> Result<usize> {
            if t.abs() < 1e-3 {
                return Ok(zero);
            }
            chain.timesteps.iter().position(|x| (*x as f64 - t).abs() < 1e-2).ok_or_else(|| Error(format!("replay: timestep {t} not on the ladder")))
        };
        let full_frames = 5 + (chain.threshold.max(1 + events.len() + chain.recent_k) - 1) * CHUNK;
        let mut kv = Kv::new(gpu, hv.dit.doubles.len(), FULL_HISTORY, tpf, full_frames, 2 * CHUNK + 1, int8)?;
        let tt = events.iter().map(|e| e.text.tokens()).max().unwrap_or(1);
        let tq = events.iter().map(|e| e.text.tq).max().unwrap_or(1);
        let rb = RollBuf::new(&hv.dit, gpu, tpf, full_frames.max(WINDOW * CHUNK + 1), full_frames + WINDOW * CHUNK + 1, tt, tq)?;
        let mut out = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            let kind = op["op"].as_str().unwrap_or("");
            let (_, x) = read_npy(&dir.join(format!("op_{i:03}_{kind}_in.npy")))?;
            let xb = DevBuf::from_f32(gpu, &x)?;
            match kind {
                "denoise" => {
                    // chunks: runs of frames at one timestep
                    let t: Vec<f64> = op["t"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
                    let mut spans: Vec<(usize, usize)> = Vec::new();
                    for v in &t {
                        let r = row(*v)?;
                        match spans.last_mut() {
                            Some((f, s)) if *s == r => *f += 1,
                            _ => spans.push((1, r)),
                        }
                    }
                    let e = op["event"].as_u64().unwrap_or(0) as usize;
                    let text = &events.get(e).ok_or_else(|| Error(format!("replay: event {e}")))?.text;
                    let v = hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &spans, hw, Pass::Denoise { text, tsi: spans[0].1 }, attn)?
                        .ok_or_else(|| Error("no velocity".into()))?
                        .to_f32()?;
                    let (_, want) = read_npy(&dir.join(format!("op_{i:03}_denoise_v.npy")))?;
                    let (r, m, c) = compare(&v, &want);
                    log(format!("teacher {i:3} denoise  chunks {:?} cache {} frames: velocity rel {r:.2e} max {m:.2e} cos {c:.6}",
                                spans.iter().map(|s| s.0).collect::<Vec<_>>(), kv.frames(0)));
                    out.push(json!({"op": i, "v_rel": r, "cos": c}));
                }
                "encode" => {
                    let f = op["frames"].as_u64().unwrap_or(0) as usize;
                    hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &[(f, zero)], hw, Pass::Encode, attn)?;
                }
                "recompute" => {
                    let fr: Vec<(usize, usize)> = op["chunks"].as_array().into_iter().flatten().filter_map(Value::as_u64).map(|f| (f as usize, zero)).collect();
                    hv.dit.roll_pass(nsd, &st, &rb, &mut kv, &xb, &fr, hw, Pass::Recompute, attn)?;
                }
                k => return Err(Error(format!("replay: op {k}"))),
            }
        }
        Ok(out)
    }
}

impl VideoEngine for MovieRoll {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 864, height: 480, seconds: 10.0, fps: FPS as f32, steps: if self.mode_distilled { 5 } else { 50 }, cfg: 1.0, sampler: Sampler::Euler,
                   schedule: Schedule::Shift, shift: 5.0, upscale: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.hv.load_s
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn jobs(&self) -> &'static [&'static str] {
        &["generate", "check"]
    }
    fn run_job(&self, spec: &Value, ctl: &mut JobCtl) -> Result<Value> {
        match spec.get("kind").and_then(Value::as_str) {
            Some("generate") => self.generate_job(spec, ctl),
            Some("check") => self.check_job(spec, ctl),
            k => Err(Error(format!("{ARCH}: no job {k:?} (generate, check)"))),
        }
    }
    fn device_stats(&self) -> Option<(u64, u64, Option<u64>)> {
        let (total, free) = self.hv.nsd.gpu.memory().ok()?;
        Some((total - free.unwrap_or(total), total, free))
    }
    fn generate(&self, req: &VideoRequest, progress: &mut dyn FnMut(Progress), cancel: &dyn Fn() -> bool) -> Result<Clip> {
        let t0 = Instant::now();
        let image = req.keyframes.first().map(|k| k.picture.clone());
        let attn = self.attn(req.extra.get("attention").map(String::as_str))?;
        let mut p = |at: usize, of: usize| -> Result<()> {
            if cancel() {
                return Err(Error("cancelled".into()));
            }
            progress(Progress { stage: "denoise", at: at as u32, of: of as u32, seconds: t0.elapsed().as_secs_f64() });
            Ok(())
        };
        let num = |k: &str| req.extra.get(k).and_then(|v| v.parse::<f64>().ok());
        let kv = req.extra.get("kv").map_or(Ok(self.kv), |v| KvMode::parse(v)).map_err(Error)?;
        let (_, frames, w, h) = self.make(&req.prompt, image.as_ref(), req.width.map(|x| x as usize), req.height.map(|x| x as usize),
                                          req.seconds.unwrap_or(10.0).min(MAX_SECONDS), req.seed, req.steps.map(|x| x as usize),
                                          num("context-noise").map(|x| x as f32), num("threshold").map(|x| x as usize), attn, kv, &req.out, None, &mut p,
                                          &mut |l| eprintln!("{l}"))?;
        Ok(Clip { path: req.out.clone(), width: w as u32, height: h as u32, frames: frames as u32, fps: FPS as f32, audio: false })
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} ({}) on {}", if self.mode_distilled { "distilled" } else { "base" }, self.hv.nsd.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_schedule_is_upstreams() {
        // distilled: one step a slot; round 0 the new chunk at 0; round 4 the oldest at 4; round 5 the queue rolled
        assert_eq!(step_index(0, 5, 5)[0], 0);
        assert_eq!(&step_index(1, 5, 5)[..2], &[1, 0]);
        assert_eq!(step_index(4, 5, 5), vec![4, 3, 2, 1, 0]);
        assert_eq!(step_index(5, 5, 5), vec![4, 3, 2, 1, 0]);
        // base: 50 steps, 10 a slot
        assert_eq!(&step_index(9, 50, 5)[..1], &[9]);
        assert_eq!(&step_index(10, 50, 5)[..2], &[10, 0]);
        assert_eq!(step_index(49, 50, 5), vec![49, 39, 29, 19, 9]);
        assert_eq!(step_index(50, 50, 5), vec![40, 30, 20, 10, 0]);
    }

    #[test]
    fn chunks_events_and_recompute_choices_are_upstreams() {
        assert_eq!(chunk_sizes(29, false).unwrap(), vec![5, 4, 4, 4, 4, 4, 4]);
        assert_eq!(chunk_sizes(29, true).unwrap(), vec![1, 4, 4, 4, 4, 4, 4, 4]);
        assert!(chunk_sizes(28, false).is_err());
        assert_eq!(event_splits(7, 2), (vec![3], vec![0, 1]));
        assert_eq!(event_splits(7, 1), (vec![], vec![0]));
        assert_eq!(recompute_chunks(10, &[], 3), vec![0, 4, 7, 8, 9]);
        assert_eq!(recompute_chunks(5, &[3, 8], 3), vec![0, 1, 2, 3, 4]);
        assert_eq!(frames_for(10.0), 241);
        assert_eq!(frames_for(60.0), 1441);
        assert_eq!(frames_for(0.1), 17);
        assert_eq!(frames_for(9.0), 225);
        assert_eq!(frames_for(240.0 / 24.0), 241);
        assert_eq!(events_of("a @@@EVENT_BREAK@@@ b"), vec!["a", "b"]);
        assert_eq!(events_of(r#"["a", "b", " "]"#), vec!["a", "b"]);
        assert_eq!(events_of("one"), vec!["one"]);
    }

    #[test]
    fn the_distilled_ladder_is_shifted() {
        let c = Chain::distilled(0.15, 18);
        assert_eq!(c.steps(), 5);
        assert!((c.timesteps[0] - 1000.0).abs() < 1e-3);
        assert!((c.sigmas[1] - shift(0.8, 5.0)).abs() < 1e-6);
        assert_eq!(*c.sigmas.last().unwrap(), 0.0);
    }
}
