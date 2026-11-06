//! FlashVSR v1.1 (OpenImagingLab; Apache-2.0) in Rust and SYCL: one-step diffusion video super-resolution, a whole
//! clip as one stream. FlashVSR's tiny pipeline (diffsynth/pipelines/flashvsr_tiny.py with v1.1's `Causal_LQ4x_Proj`)
//! without Python; a port of nextcuda's engine (image/flashvsr there) to Intel's cards:
//!
//! ```text
//!   frames -> Pillow bicubic x scale -> mirrored out to multiples of 128 ([-1, 1])
//!   chunk 0: frames 0-24 -> LQ projection (6 latent frames) + noise -> DiT (one step, its KV ring filled) -> latents
//!            -> tiny decoder with frames 0-20 as its condition -> frames 0-20
//!   chunk k: frames 8k+17 .. 8k+24 -> 2 latent frames -> DiT (attending to the cached 3 pairs) -> frames 8k+13 .. 8k+20
//!   each frame: the colour fix (AdaIN per channel to the upscaled input frame, the reference's), cut back to the
//!   picture, bytes
//! ```
//!
//! The stream has no length limit: the KV ring rolls (the last 3 latent pairs), the convolutions keep their last
//! frames, and the rotary positions are relative. The clip's last frame is repeated past its end (4 to 11 times, as the
//! reference pads) so every frame comes out. One card by default; the DiT's blocks can be split over two
//! (`NS_FVSR_GPUS`, `NS_FVSR_SPLIT`), the cards pipelined. A size whose stream does not fit a card runs in tiles of the
//! upscaled frame (x4: 2 x 2 of 1792 x 1024), each its own stream over segments of frames, the overlaps blended.
//!
//! Source: `dit.rs` the DiT and its block-sparse attention, `lq.rs` the LQ projection, `tcd.rs` the decoder,
//! `resize.rs` the frames' preparation, `check.rs` the check against reference/flashvsr/ref.py.

pub mod check;
pub mod dit;
pub mod ffi;
pub mod lq;
pub mod prof;
pub mod resize;
pub mod tcd;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{At, DevBuf, EngineOption, Gpu, HostBuf};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step,
    UpscaleInfo, UpscaleRequest, UpscaleVariant};

use crate::dit::{Chunk, Ends, Rope, Stage, DIM};
use crate::ffi::{check as ck, f32_bytes, g};
use crate::lq::LqProj;
use crate::resize::Geometry;
use crate::tcd::Tcd;

pub const ARCH: &str = "flashvsr";
/// the DiT (its safetensors), the LQ projection and the decoder (their PyTorch checkpoints), the prompt's embedding
pub const ROLES: &[&str] = &["dit", "lq-proj", "decoder", "prompt"];

pub const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "gpus", env: "NS_FVSR_GPUS", value: "N[,M]", help: "the cards, by index (default: the server's one; \"0,1\": the DiT's blocks split over both, pipelined)",
                   at: At::Load },
    EngineOption { name: "split", env: "NS_FVSR_SPLIT", value: "N,M", help: "the DiT's 30 blocks over the cards, in order (default 30 on one, 14,16 on two)",
                   at: At::Load },
    EngineOption { name: "decoder", env: "NS_FVSR_DECODER", value: "first|last", help: "the card the decoder runs on, with two (default first)", at: At::Load },
    EngineOption { name: "sparse_ratio", env: "", value: "R", help: "the attention's top-k as FlashVSR's sparse_ratio (default 1.5: faster than 2.0, as steady)",
                   at: At::Request },
    EngineOption { name: "local_range", env: "", value: "N", help: "the attention's local window in 128-pixel blocks, odd (default 11: steadier; 9 sharper)",
                   at: At::Request },
    EngineOption { name: "color_fix", env: "", value: "0|1", help: "match each frame's colour to the upscaled input frame (AdaIN, default 1)", at: At::Request },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "FlashVSR v1.1 video super-resolution (Wan 2.1 1.3B, one step, streaming; Rust + SYCL)", roles: ROLES, load, options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(FlashVsr::load(files, gpus, o, log)?))
}

const VARIANTS: &[UpscaleVariant] = &[UpscaleVariant {
    id: "flashvsr-v1.1",
    about: "real footage, faces and skin: one-step diffusion over the whole clip as one stream (temporally steady), 2x to 4x",
    scales: &[2, 3, 4],
    noise: &[],
}];

/// What a run takes
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub scale: f64,
    pub seed: u64,
    pub sparse_ratio: f64,
    pub local_range: usize,
    pub color_fix: bool,
    /// the latent frame the stream's first one is, for the noise (a tile's or a segment's stream inside a clip)
    pub lat_base: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { scale: 2.0, seed: 0, sparse_ratio: 1.5, local_range: 11, color_fix: true, lat_base: 0 }
    }
}

impl Settings {
    /// The settings of a request (its variant, scale, seed and options)
    pub fn of(variant: Option<&str>, scale: Option<u32>, seed: Option<u64>, extra: &nextsycl_image::Given) -> Result<Settings> {
        if let Some(v) = variant.filter(|v| !v.is_empty() && *v != "flashvsr" && *v != VARIANTS[0].id) {
            return Err(Error(format!("{v}: flashvsr has {}", VARIANTS[0].id)));
        }
        let scale = scale.unwrap_or(2);
        if !(2..=4).contains(&scale) {
            return Err(Error(format!("flashvsr makes 2x to 4x, not {scale}x")));
        }
        let mut s = Settings { scale: scale as f64, seed: seed.unwrap_or(0), ..Settings::default() };
        if let Some(v) = extra.get("sparse_ratio") {
            s.sparse_ratio = v.trim().parse().ok().filter(|x: &f64| (0.25..=8.0).contains(x)).ok_or_else(|| Error(format!("sparse_ratio {v}: 0.25 to 8")))?;
        }
        if let Some(v) = extra.get("local_range") {
            s.local_range = v.trim().parse().ok().filter(|x: &usize| (3..=31).contains(x) && x % 2 == 1).ok_or_else(|| Error(format!("local_range {v}: odd, 3 to 31")))?;
        }
        if let Some(v) = extra.get("color_fix") {
            s.color_fix = !matches!(v.trim(), "0" | "false" | "no" | "off");
        }
        Ok(s)
    }
}

/// A check's view into a run (check.rs): the noise to use, the tensors to compare
pub trait Hooks {
    /// latent frame t's noise [16, h8, w8] (None: the engine's own)
    fn noise(&mut self, _t: usize, _h8: usize, _w8: usize) -> Option<Vec<f32>> {
        None
    }
    /// chunk `c`'s tensor `name` (lq, b0, b15, b29, lat), on its card
    fn tensor(&mut self, _c: usize, _name: &str, _v: &DevBuf, _rows: usize) -> Result<()> {
        Ok(())
    }
    /// frame i as the decoder made it (half [th, tw, 3] in about [0, 1], before the colour fix)
    fn decoded(&mut self, _i: usize, _v: &DevBuf) -> Result<()> {
        Ok(())
    }
    /// whether it reads tensors (a check): then the cards take turns on this thread, never pipelined
    fn active(&self) -> bool {
        false
    }
}

struct NoHooks;
impl Hooks for NoHooks {}

/// A frame read: prepared, or being prepared on its own thread
enum Slot {
    Ready(Vec<u8>),
    Pending(std::thread::JoinHandle<Vec<u8>>),
}

impl Slot {
    fn ready(&mut self) -> Result<&Vec<u8>> {
        if let Slot::Pending(_) = self {
            let Slot::Pending(h) = std::mem::replace(self, Slot::Ready(Vec::new())) else { unreachable!() };
            *self = Slot::Ready(h.join().map_err(|_| Error("flashvsr: a frame's preparation failed".into()))?);
        }
        match self {
            Slot::Ready(v) => Ok(v),
            Slot::Pending(_) => unreachable!(),
        }
    }
}

/// Kept free on a card beside a run (the box spills and hangs past its memory)
pub const MARGIN: u64 = 3 << 29;
/// What the runtime and oneDNN's scratch take beside the engine's own buffers (counted against a card's total)
pub const RUNTIME: u64 = 1 << 30;

/// Whether a run needing `need` bytes more fits a card with `total` bytes, `free` of them free now: Err a plain error
/// when it never could (the engine's weights `weights` held too), Err no room when it could on an emptier card
pub fn room_for(card: &str, what: &str, need: u64, weights: u64, total: u64, free: u64) -> Result<()> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if need + weights + MARGIN + RUNTIME > total {
        return Err(Error(format!("flashvsr: {what} needs {:.1} GiB on {card} beside its {:.1} GiB of weights, and the card has {:.1} in all: a smaller scale or source",
                                 gib(need), gib(weights), gib(total))));
    }
    if need + MARGIN > free {
        return Err(Error::no_room(format!("flashvsr: {what} needs {:.1} GiB more on {card} and it has {:.1} free", gib(need + MARGIN - free), gib(free))));
    }
    Ok(())
}

/// A chunk for the cards after the first: its blocks, the head and the step
struct Job {
    chunk: usize,
    lat0: usize,
    c: Chunk,
    cs: Vec<f32>,
    noise: Option<Vec<Vec<f32>>>,
}

/// What the back card's thread tells the first: x has been taken, chunk n's latents are made
enum Msg {
    Copied,
    Lat(usize),
}

/// The cards after the first: their blocks, the head and the step (with one card, the head alone, on it)
struct Back<'a> {
    ends: &'a Ends,
    stages: &'a mut [Stage],
    xs: &'a [DevBuf],
    css: &'a [DevBuf],
    x0: &'a DevBuf,
    noise: &'a DevBuf,
    hbuf: &'a DevBuf,
    pred: &'a DevBuf,
    lat: &'a DevBuf,
    first: Arc<Gpu>,
    last: Arc<Gpu>,
    /// wait for the last card after the head (the latents go to another thread or card; else the decoder's work just
    /// follows on the same queue)
    sync_end: bool,
    seed: u64,
    lat_base: usize,
    h8: usize,
    w8: usize,
}

impl Back<'_> {
    fn rows(&self, job: &Job) -> usize {
        job.c.rows()
    }

    /// x from the first card into the second's buffer
    fn copy_in(&self, job: &Job, staging: &mut [u8]) -> Result<()> {
        if let Some(x1) = self.xs.first() {
            let src = self.x0.view(0, self.rows(job) * DIM * 4)?;
            x1.copy_from_peer(&src, staging)?;
        }
        Ok(())
    }

    /// the blocks after the first card's, the head and the step: latents = noise - output, in `lat`
    fn run(&mut self, job: &Job, staging: &mut [u8], grab: &mut dyn FnMut(usize, &str, &DevBuf, usize) -> Result<()>) -> Result<()> {
        let k = ffi::api()?;
        let (f, rows, h8, w8) = (2 * job.c.groups, self.rows(job), self.h8, self.w8);
        for (i, st) in self.stages.iter_mut().enumerate() {
            if i > 0 {
                let src = self.xs[i - 1].view(0, rows * DIM * 4)?;
                self.xs[i].copy_from_peer(&src, staging)?;
            }
            self.css[i].view(0, self.css[i].len)?.write_async(Rope::hw_len(self.h8 / 2, self.w8 / 2) * 4, f32_bytes(&job.cs))?;
            st.run(&self.xs[i], &self.css[i], &job.c, &mut |layer, x| {
                if [0, 15, 29].contains(&layer) {
                    grab(job.chunk, &format!("b{layer}"), x, rows)?;
                }
                Ok(())
            })?;
        }
        let x = self.xs.last().unwrap_or(self.x0);
        if !Arc::ptr_eq(&self.first, &self.last) {
            write_noise(self.noise, &self.last, job.noise.as_deref(), f, job.lat0 + self.lat_base, h8, w8, self.seed)?;
        }
        let gl = g(&self.last);
        // SAFETY: the buffers hold rows x 1536 / 64 values and 16 x f x h8 x w8 latents
        ck(unsafe { (k.ln)(gl, x.fp(), rows as i64, DIM as i64, 1e-6, self.ends.head_sc.fp(), self.ends.head_sh.fp(), self.hbuf.ptr()) }, "head norm")?;
        self.ends.head.float(&self.last, self.hbuf.ptr(), rows, self.pred.fp())?;
        ck(unsafe { (k.unpatchify_sub)(gl, self.pred.fp(), self.noise.fp(), f as i64, h8 as i64, w8 as i64, self.lat.fp()) }, "step")?;
        grab(job.chunk, "lat", self.lat, 16 * f * h8 * w8)?;
        if self.sync_end {
            self.last.sync()?;
        }
        Ok(())
    }
}

/// The decoder and the colour fix on their card: a chunk's latents to frames, bytes, out. The condition frames go up
/// and the bytes come back through pinned host memory, copied by the queue in its order (the copy engines' full rate;
/// kernels reading the pinned memory in place over PCIe made a frame 30 % slower), so a chunk waits for its card once,
/// at its end - not a copy and a wait a frame
struct Dec<'a> {
    tcd: &'a mut Tcd,
    /// the condition frames' bytes, up to 24 [th, tw, 3]; the made frames' bytes, up to 24 [sh, sw, 3]
    pin_cond: HostBuf,
    pin_out: HostBuf,
    /// their copies on the card
    cond_u8: &'a DevBuf,
    out_u8: &'a DevBuf,
    cond: &'a DevBuf,
    scratch: &'a DevBuf,
    ms: &'a DevBuf,
    gpu: Arc<Gpu>,
    geo: Geometry,
    color_fix: bool,
}

impl Dec<'_> {
    /// Chunk pc's f latent frames (lat, on this card) with their condition frames `bytes` (prepared [th, tw, 3] each:
    /// 24 for chunk 0, 8 after): each real frame (before the clip's end, `ended`) made and handed to `emit`; how many
    #[allow(clippy::too_many_arguments)]
    fn chunk(&mut self, lat: &DevBuf, pc: usize, f: usize, bytes: &[u8], ended: Option<usize>, emit: &mut dyn FnMut(Picture) -> Result<()>,
             hooks: &mut dyn Hooks) -> Result<usize> {
        let k = ffi::api()?;
        let geo = self.geo;
        let gd = g(&self.gpu);
        let fsize = geo.th * geo.tw * 3;
        let nc = bytes.len() / fsize;
        // (the card is done with these pinned bytes: the last chunk ended with a wait for it)
        self.pin_cond.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        self.cond_u8.copy_within(0, &self.pin_cond.device_view(0, bytes.len())?, 0, bytes.len())?;
        // SAFETY (the calls below): the buffers hold what they read and write (sized by the run).
        ck(unsafe { (k.u8_to_half)(gd, self.cond_u8.ptr() as *const u8, (nc * fsize) as i64, self.cond.ptr()) }, "cond")?;
        let pre = if pc == 0 { 0 } else { 8 * pc + 13 };
        let front = if pc == 0 { 3 } else { 0 };
        let real = |i: usize| ended.is_none_or(|n| i < n);
        let mut idx = pre;
        let fb = geo.sw * geo.sh * 3;
        let mut made = 0usize;
        let (cond, scratch, ms, gpu, color_fix, out_all) = (self.cond, self.scratch, self.ms, self.gpu.clone(), self.color_fix, self.out_u8);
        for t in 0..f {
            let cp = unsafe { (cond.ptr() as *mut u8).add(4 * t * fsize * 2) };
            self.tcd.frame(lat, f, t, cp as *const _, &mut |dec| {
                let i = idx;
                idx += 1;
                if !real(i) {
                    return Ok(());
                }
                hooks.decoded(i, dec)?;
                let lqf = unsafe { (cond.ptr() as *mut u8).add((i - pre + front) * fsize * 2) };
                let out_u8 = out_all.view(made * fb, fb)?;
                made += 1;
                prof::timed(&gpu, "colour fix + bytes", 0.0, (geo.sw * geo.sh * 3 * 4) as f64, || {
                    let msp = if color_fix {
                        ck(unsafe { (k.adain)(gd, dec.ptr(), lqf as *const _, geo.tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64,
                                              scratch.fp(), ms.fp()) }, "colour")?;
                        ms.fp() as *const f32
                    } else {
                        std::ptr::null()
                    };
                    ck(unsafe { (k.to_u8)(gd, dec.ptr(), geo.tw as i64, geo.left as i64, geo.top as i64, geo.sw as i64, geo.sh as i64, msp,
                                          out_u8.ptr() as *mut u8) }, "bytes")
                })?;
                Ok(())
            })?;
        }
        // the chunk's one wait: its frames' bytes copied down to pinned memory
        if made > 0 {
            self.pin_out.device_view(0, made * fb)?.copy_within(0, self.out_u8, 0, made * fb)?;
        }
        self.gpu.sync()?;
        for j in 0..made {
            let mut pic = Picture::new(geo.sw as u32, geo.sh as u32, 3);
            pic.data.copy_from_slice(&self.pin_out.as_slice()[j * fb..(j + 1) * fb]);
            emit(pic)?;
        }
        Ok(made)
    }
}

/// A chunk's noise [16, f, h8, w8] into buf: given (a check's, per latent frame [16, h8, w8]) or the engine's own
#[allow(clippy::too_many_arguments)]
fn write_noise(buf: &DevBuf, gpu: &Arc<Gpu>, given: Option<&[Vec<f32>]>, f: usize, lat0: usize, h8: usize, w8: usize, seed: u64) -> Result<()> {
    let k = ffi::api()?;
    for t in 0..f {
        for c in 0..16 {
            match given {
                Some(v) => buf.write(((c * f + t) * h8 * w8) * 4, f32_bytes(&v[t][c * h8 * w8..(c + 1) * h8 * w8]))?,
                None => {
                    // SAFETY: plane (c, t) of buf
                    let p = unsafe { buf.fp().add((c * f + t) * h8 * w8) };
                    ck(unsafe { (k.randn)(g(gpu), p, (h8 * w8) as i64, seed, (((lat0 + t) * 16 + c) * h8 * w8) as u64) }, "noise")?;
                }
            }
        }
    }
    Ok(())
}

struct Inner {
    stages: Vec<Stage>,
    lq: LqProj,
    tcd: Tcd,
    /// the decoder's card (an index into the stages' cards)
    dec: usize,
}

pub struct FlashVsr {
    pub gpus: Vec<Arc<Gpu>>,
    ends: Ends,
    inner: Mutex<Inner>,
    load_secs: f64,
    /// each card's weights (bytes)
    pub weight_bytes: Vec<usize>,
}

impl Drop for FlashVsr {
    fn drop(&mut self) {
        // the weights go: their oneDNN reorders must not be found by a later buffer at the same address
        if let Ok(k) = ffi::api() {
            for gpu in &self.gpus {
                let _ = gpu.sync();
                // SAFETY: a live GPU handle.
                unsafe { (k.dnn_clear)(g(gpu)) };
            }
        }
    }
}

fn pth_table(path: &Path) -> Result<SafeTensors> {
    nextsycl_gguf::pthst::open(path).map_err(|e| Error(e.0))
}

/// The cards a load runs on: NS_FVSR_GPUS by index, else the server's
fn cards(o: &LoadOptions, gpus: &[Arc<Gpu>]) -> Result<Vec<Arc<Gpu>>> {
    match o.setting("NS_FVSR_GPUS").filter(|v| !v.trim().is_empty()) {
        Some(v) => v.split(',').map(|i| i.trim().parse::<usize>().map_err(|_| Error(format!("NS_FVSR_GPUS={v}: card numbers, comma separated"))).and_then(Gpu::open))
            .collect(),
        None => Ok(gpus.to_vec()),
    }
}

/// The DiT's blocks a card: NS_FVSR_SPLIT, else 30 on one card, 14 / 16 on two - the first (the B70) also holds the LQ
/// projection and the decoder (measured at 1728 x 960: 17 / 13 0.116 s a frame, the B65 66 % busy; 15 / 15 0.110;
/// 14 / 16 0.107, both 86 %)
pub fn split(spec: Option<&str>, cards: usize) -> Result<Vec<usize>> {
    let counts: Vec<usize> = match spec {
        Some(s) if !s.trim().is_empty() => s.split(',').map(|v| v.trim().parse::<usize>().map_err(|_| Error(format!("NS_FVSR_SPLIT={s}: block counts")))).collect::<Result<_>>()?,
        _ if cards >= 2 => vec![14, 16],
        _ => vec![dit::LAYERS],
    };
    if counts.iter().sum::<usize>() != dit::LAYERS || counts.len() > cards.max(1) || counts.contains(&0) {
        return Err(Error(format!("NS_FVSR_SPLIT: {counts:?} is not {} blocks over at most {cards} card(s)", dit::LAYERS)));
    }
    Ok(counts)
}

impl FlashVsr {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<FlashVsr> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        let gpus = cards(o, gpus)?;
        if gpus.is_empty() {
            return Err(Error("flashvsr needs a GPU".into()));
        }
        let k = ffi::api()?;
        for gpu in &gpus {
            // SAFETY: a live GPU handle (a reload's buffers may reuse an address an earlier engine's reorders know).
            ck(unsafe { (k.dnn_clear)(g(gpu)) }, "oneDNN's caches")?;
        }
        let path = |role: &str| files.get(role).cloned().ok_or_else(|| Error(format!("flashvsr: no {role} file (the card names dit, lq-proj, decoder, prompt)")));
        let st = SafeTensors::open(&path("dit")?).map_err(|e| Error(e.0))?;
        let counts = split(o.setting("NS_FVSR_SPLIT").as_deref(), gpus.len())?;
        let gpus: Vec<Arc<Gpu>> = gpus[..counts.len()].to_vec();
        let (t, tmod) = dit::time_embedding(&st)?;
        let prompt = nextsycl_gguf::pth::Pth::open(&path("prompt")?).map_err(|e| Error(e.0))?;
        let name = prompt.tensors.keys().next().cloned().ok_or_else(|| Error("flashvsr: the prompt file holds no tensor".into()))?;
        let ctx = prompt.f32(&name).map_err(|e| Error(e.0))?;
        if ctx.is_empty() || ctx.len() % dit::TEXT != 0 {
            return Err(Error(format!("flashvsr: the prompt is {} values, not rows of {}", ctx.len(), dit::TEXT)));
        }
        let ctx = dit::text_embedding(&st, &gpus[0], &ctx)?;
        let mut stages = Vec::new();
        let mut from = 0;
        for (i, n) in counts.iter().enumerate() {
            stages.push(Stage::load(&st, &gpus[i], from, from + n, &tmod, &ctx, log)?);
            from += n;
        }
        let last = gpus.last().unwrap().clone();
        let ends = Ends::load(&st, &gpus[0], &last, &t)?;
        let lq = LqProj::load(&pth_table(&path("lq-proj")?)?, &gpus[0])?;
        let dec = match o.setting("NS_FVSR_DECODER").as_deref().unwrap_or("first") {
            "first" | "" => 0,
            "last" => gpus.len() - 1,
            d => return Err(Error(format!("NS_FVSR_DECODER={d}: first or last"))),
        };
        let tcd = Tcd::load(&pth_table(&path("decoder")?)?, &gpus[dec])?;
        let mut weight_bytes: Vec<usize> = stages.iter().map(|s| s.weight_bytes).collect();
        weight_bytes[0] += lq.bytes + ends.patch.bytes();
        *weight_bytes.last_mut().unwrap() += ends.head.bytes();
        weight_bytes[dec] += tcd.bytes;
        for gpu in &gpus {
            gpu.sync()?;
        }
        log(format!("flashvsr: loaded in {:.1} s ({})", t0.elapsed().as_secs_f64(),
                    weight_bytes.iter().zip(&gpus).map(|(b, g)| format!("{:.2} GiB on {}", *b as f64 / (1u64 << 30) as f64, g.name)).collect::<Vec<_>>().join(", ")));
        Ok(FlashVsr { gpus, ends, inner: Mutex::new(Inner { stages, lq, tcd, dec }), load_secs: t0.elapsed().as_secs_f64(), weight_bytes })
    }

    /// The memory a frame size takes on each card beyond the weights (bytes), and whether each card has it made
    fn need(inner: &Inner, geo: &Geometry) -> Vec<(usize, bool)> {
        let (hp, wp) = (geo.th / 16, geo.tw / 16);
        let nb = (hp / 8) * (wp / 8);
        let mut v: Vec<(usize, bool)> = inner.stages.iter().map(|s| (s.need_bytes(nb), s.is_prepared(nb))).collect();
        v[0].0 += LqProj::need_bytes(hp, wp) + Self::stream_bytes(geo);
        v[inner.dec].0 += Tcd::need_bytes(geo.th / 8, geo.tw / 8);
        v
    }

    /// The run's own buffers beside the stages' (frames, tokens, noise, latents: the first and last cards')
    fn stream_bytes(geo: &Geometry) -> usize {
        let (th, tw) = (geo.th, geo.tw);
        let rows0 = 3 * (th / 16) * (tw / 16);
        let fsize = th * tw * 3;
        4 * fsize * 3 + 6 * rows0 / 3 * DIM * 4 + 2 * 16 * 6 * (th / 8) * (tw / 8) * 4 + rows0 * (64 * 6 + DIM * 4 + 128 * 4 + DIM * 2) + 24 * fsize * 3
    }

    /// Every card's per-size buffers let go (the weights stay); NS_FVSR_KEEP=1 keeps them for the next clip of the
    /// same size
    pub fn release(&self) -> Result<()> {
        if std::env::var("NS_FVSR_KEEP").is_ok_and(|v| v.trim() == "1") {
            return Ok(());
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        for gpu in &self.gpus {
            gpu.sync()?;
        }
        for s in inner.stages.iter_mut() {
            s.release();
        }
        inner.lq.release();
        inner.tcd.release();
        Ok(())
    }

    /// Whether a size's stream fits the cards (else it runs in tiles)
    pub fn fits(&self, geo: &Geometry) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        Self::need(&inner, geo).iter().zip(&self.weight_bytes).zip(&self.gpus).all(|(((n, _), w), gpu)| {
            let total = gpu.memory().map(|m| m.0).unwrap_or(0);
            (*n + *w) as u64 + MARGIN + RUNTIME <= total
        })
    }

    /// A clip through the stream: frames from `next` (RGB [h, w, 3], or already prepared [th, tw, 3] when `prepared`),
    /// each made frame (the picture's part, [sh, sw, 3]) to `emit`
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, geo: Geometry, set: &Settings, prepared: bool, next: &mut dyn FnMut() -> Result<Option<Vec<u8>>>, emit: &mut dyn FnMut(Picture) -> Result<()>,
               hooks: &mut dyn Hooks) -> Result<u32> {
        let k = ffi::api()?;
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let inner = &mut *guard;
        let (th, tw) = (geo.th, geo.tw);
        let (h8, w8, hp, wp) = (th / 8, tw / 8, th / 16, tw / 16);
        let (nbh, nbw) = (hp / 8, wp / 8);
        let nb = nbh * nbw;
        // the memory first: refused here, before anything is made, when a card cannot hold this size now (no room:
        // retryable) or ever (a plain error)
        let what = format!("{} x {} ({} x {} inside)", geo.sw, geo.sh, tw, th);
        for (((n, made), w), gpu) in Self::need(inner, &geo).into_iter().zip(&self.weight_bytes).zip(&self.gpus) {
            if let Ok((total, Some(free))) = gpu.memory() {
                room_for(&gpu.name, &what, if made { 0 } else { n as u64 }, *w as u64, total, free)?;
            }
        }
        for s in inner.stages.iter_mut() {
            s.prepare(nb)?;
        }
        inner.lq.reset(hp, wp)?;
        inner.tcd.reset(h8, w8)?;
        let topk = dit::topk(nb, th, tw, set.sparse_ratio);
        let first_gpu = self.gpus[0].clone();
        let last_gpu = self.gpus.last().unwrap().clone();
        let dec_gpu = self.gpus[inner.dec].clone();
        let fsize = th * tw * 3;
        // the frames on the host: [base, base + len) of the clip, each prepared on a thread of its own as it is read (a
        // chunk ahead, so the bicubic runs beside the cards' work); the clip's count once it has ended
        let mut frames: VecDeque<Slot> = VecDeque::new();
        let mut base = 0usize;
        let mut ended: Option<usize> = None;
        let mut last: Vec<u8> = Vec::new();
        let mut read = |want: usize, frames: &mut VecDeque<Slot>, base: usize, ended: &mut Option<usize>| -> Result<()> {
            while ended.is_none() && base + frames.len() < want {
                match next()? {
                    Some(f) if prepared => frames.push_back(Slot::Ready(f)),
                    Some(f) => frames.push_back(Slot::Pending(std::thread::spawn(move || resize::prepare(&f, &geo)))),
                    None => *ended = Some(base + frames.len()),
                }
            }
            Ok(())
        };
        // frame i prepared (waited for); past the end, the clip's last
        let get = |i: usize, frames: &mut VecDeque<Slot>, base: usize, last: &mut Vec<u8>| -> Result<Vec<u8>> {
            let at = if i >= base && i - base < frames.len() { i - base } else if !frames.is_empty() { frames.len() - 1 } else { return Ok(last.clone()) };
            let is_last = at == frames.len() - 1;
            let p = frames[at].ready()?;
            if p.len() != fsize {
                return Err(Error(format!("flashvsr: a frame of {} bytes, {} expected", p.len(), fsize)));
            }
            if is_last {
                last.clone_from(p);
            }
            Ok(p.clone())
        };
        // buffers: up to 4 frames of bytes and halves, the LQ tokens, noise, x and the latents
        // the LQ projection's frames: 8 calls' slots of 4 frames in pinned memory the kernel reads in place, used in turn (a
        // chunk has at most 7 calls, and every chunk's end waits for the card: a slot is free when it comes round again)
        let mut pin_lq = HostBuf::new(&first_gpu, 8 * 4 * fsize)?;
        let mut lq_slot = 0usize;
        let frames_u8 = DevBuf::new(&first_gpu, 4 * fsize)?;
        let frames_h = DevBuf::new(&first_gpu, 4 * fsize * 2)?;
        let lqtok = DevBuf::new(&first_gpu, 6 * hp * wp * DIM * 4)?;
        let rows0 = 3 * nb * 128;
        let noise0 = DevBuf::new(&first_gpu, 16 * 6 * h8 * w8 * 4)?;
        let patches = DevBuf::new(&first_gpu, rows0 * 64 * 2)?;
        let mut xs: Vec<DevBuf> = Vec::new();
        let mut css: Vec<DevBuf> = Vec::new();
        for s in &inner.stages {
            xs.push(DevBuf::new(&s.gpu, rows0 * DIM * 4)?);
            // the RoPE tables: the grid's patches' once, the chunk's frames' each chunk
            let rope = DevBuf::new(&s.gpu, Rope::len(hp, wp) * 4)?;
            rope.write(0, f32_bytes(&Rope::hw(hp, wp)))?;
            css.push(rope);
        }
        let noise_l = if Arc::ptr_eq(&first_gpu, &last_gpu) { noise0.view(0, noise0.len)? } else { DevBuf::new(&last_gpu, 16 * 6 * h8 * w8 * 4)? };
        // the head's normed rows: on one card in the LQ tokens' buffer (used up by then: they are added before block 0)
        let hbuf = if Arc::ptr_eq(&first_gpu, &last_gpu) { lqtok.view(0, rows0 * DIM * 2)? } else { DevBuf::new(&last_gpu, rows0 * DIM * 2)? };
        let pred = DevBuf::new(&last_gpu, rows0 * 64 * 4)?;
        let lat = DevBuf::new(&last_gpu, 16 * 6 * h8 * w8 * 4)?;
        let lat_d = if Arc::ptr_eq(&dec_gpu, &last_gpu) { lat.view(0, lat.len)? } else { DevBuf::new(&dec_gpu, lat.len)? };
        let mut lat_staging = vec![0u8; if Arc::ptr_eq(&dec_gpu, &last_gpu) { 0 } else { lat.len }];

        let cond = DevBuf::new(&dec_gpu, 24 * fsize * 2)?;
        let scratch = DevBuf::new(&dec_gpu, 2 * 256 * 6 * 4)?;
        let ms = DevBuf::new(&dec_gpu, 12 * 4)?;

        let mut staging = vec![0u8; if inner.stages.len() > 1 { rows0 * DIM * 4 } else { 0 }];
        let gf = g(&first_gpu);
        let ns = inner.stages.len();
        // the cards overlap when the decoder is on the first: the last card runs chunk k's blocks while the first decodes
        // chunk k - 1 and starts chunk k + 1 (not when profiling - the marks wait for both cards - nor for a check, whose
        // hooks read the tensors on this thread)
        let pipelined = ns > 1 && inner.dec == 0 && !prof::on() && !hooks.active() && std::env::var("NS_FVSR_PIPELINE").map_or(true, |v| v != "0");
        let Inner { stages, lq, tcd, .. } = inner;
        let (front, rest) = stages.split_at_mut(1);
        let stage0 = &mut front[0];
        let (x0, xs_b) = xs.split_first().unwrap();
        let (cs0, css_b) = css.split_first().unwrap();
        let back = Back { ends: &self.ends, stages: rest, xs: xs_b, css: css_b, x0, noise: &noise_l, hbuf: &hbuf, pred: &pred, lat: &lat, last: last_gpu.clone(),
                              first: first_gpu.clone(), sync_end: pipelined || !Arc::ptr_eq(&dec_gpu, &last_gpu), seed: set.seed, lat_base: set.lat_base, h8, w8 };
        // the decoder: chunk pc's latents (in lat_d) with its frames as the condition, its frames out
        let cond_u8 = DevBuf::new(&dec_gpu, 24 * fsize)?;
        let out_u8 = DevBuf::new(&dec_gpu, 24 * geo.sw * geo.sh * 3)?;
        let mut dec = Dec { tcd, pin_cond: HostBuf::new(&dec_gpu, 24 * fsize)?, pin_out: HostBuf::new(&dec_gpu, 24 * geo.sw * geo.sh * 3)?, cond_u8: &cond_u8,
                            out_u8: &out_u8, cond: &cond,
                            scratch: &scratch, ms: &ms, gpu: dec_gpu.clone(), geo, color_fix: set.color_fix };
        let mut made = 0usize;
        let mut low = vec![u64::MAX; self.gpus.len()];
        std::thread::scope(|sc| -> Result<()> {
            let (jtx, jrx) = std::sync::mpsc::channel::<Job>();
            let (rtx, rrx) = std::sync::mpsc::channel::<std::result::Result<Msg, String>>();
            let (atx, arx) = std::sync::mpsc::channel::<()>();
            let mut inline = None;
            if pipelined {
                let mut staging = std::mem::take(&mut staging);
                let mut back = back;
                sc.spawn(move || {
                    let mut owed = false;
                    while let Ok(job) = jrx.recv() {
                        let r = (|| -> Result<()> {
                            back.copy_in(&job, &mut staging)?;
                            rtx.send(Ok(Msg::Copied)).map_err(|_| Error("stopped".into()))?;
                            // the latents' buffer is free once the first card has taken the last ones
                            if owed {
                                arx.recv().map_err(|_| Error("stopped".into()))?;
                            }
                            back.run(&job, &mut staging, &mut |_, _, _, _| Ok(()))?;
                            owed = true;
                            rtx.send(Ok(Msg::Lat(job.chunk))).map_err(|_| Error("stopped".into()))
                        })();
                        if let Err(e) = r {
                            let _ = rtx.send(Err(e.0));
                            return;
                        }
                    }
                });
            } else {
                inline = Some(back);
            }
            // the back card's messages, kept until asked for (its "latents made" can come before "x taken" of the next)
            let mut early: Vec<Msg> = Vec::new();
            let mut wait = |want: &dyn Fn(&Msg) -> bool| -> Result<()> {
                if let Some(i) = early.iter().position(want) {
                    early.remove(i);
                    return Ok(());
                }
                loop {
                    match rrx.recv() {
                        Ok(Ok(m)) if want(&m) => return Ok(()),
                        Ok(Ok(m)) => early.push(m),
                        Ok(Err(e)) => return Err(Error(e)),
                        Err(_) => return Err(Error("flashvsr: the second card's thread stopped".into())),
                    }
                }
            };
            let mut pending: Option<(usize, usize)> = None;
            let mut chunk = 0usize;
            loop {
                let hi = if chunk == 0 { 25 } else { 8 * chunk + 25 };
                // a chunk ahead: the next chunk's frames prepare while this one runs
                read(hi + 8, &mut frames, base, &mut ended)?;
                let done = match ended {
                    Some(0) => return Err(Error("flashvsr: the clip has no frames".into())),
                    Some(n) => chunk + 2 >= (n + 3).div_ceil(8).max(3),
                    None => false,
                };
                if !done {
                    let (f, lat0, calls): (usize, usize, Vec<(usize, usize)>) = if chunk == 0 {
                        (6, 0, (0..7).map(|i| (if i == 0 { 0 } else { 4 * i - 3 }, 4 * i + 1)).collect())
                    } else {
                        (2, 4 + 2 * chunk, vec![(8 * chunk + 17, 8 * chunk + 21), (8 * chunk + 21, 8 * chunk + 25)])
                    };
                    let rows = f / 2 * nb * 128;
                    // SAFETY (the kernel calls below): every buffer holds what the call reads or writes, sized above.
                    // ---- the LQ projection
                    let mut got = 0;
                    for (a, b) in calls {
                        let n = b - a;
                        let mut bytes = Vec::with_capacity(n * fsize);
                        for i in a..b {
                            bytes.extend_from_slice(&get(i, &mut frames, base, &mut last)?);
                        }
                        let at = (lq_slot % 8) * 4 * fsize;
                        lq_slot += 1;
                        pin_lq.as_mut_slice()[at..at + bytes.len()].copy_from_slice(&bytes);
                        frames_u8.copy_within(0, &pin_lq.device_view(at, bytes.len())?, 0, bytes.len())?;
                        ck(unsafe { (k.u8_to_half)(gf, frames_u8.ptr() as *const u8, (n * fsize) as i64, frames_h.ptr()) }, "frames")?;
                        let out = lqtok.view(got * hp * wp * DIM * 4, hp * wp * DIM * 4)?;
                        if lq.stream(&frames_h, n, &out)? {
                            got += 1;
                        }
                    }
                    if got != f {
                        return Err(Error(format!("flashvsr: the LQ projection made {got} latent frames, {f} expected")));
                    }
                    hooks.tensor(chunk, "lq", &lqtok, f * hp * wp)?;
                    // ---- the noise (a check's from its hooks, else the engine's own) and the tokens
                    let given: Option<Vec<Vec<f32>>> = (0..f).map(|t| hooks.noise(lat0 + t, h8, w8)).collect();
                    write_noise(&noise0, &first_gpu, given.as_deref(), f, lat0 + set.lat_base, h8, w8, set.seed)?;
                    ck(unsafe { (k.patchify)(gf, noise0.fp(), f as i64, h8 as i64, w8 as i64, patches.ptr()) }, "patches")?;
                    self.ends.patch.float(&first_gpu, patches.ptr(), rows, x0.fp())?;
                    ck(unsafe { (k.add_lq)(gf, x0.fp(), lqtok.fp(), f as i64, hp as i64, wp as i64, DIM as i64) }, "lq residual")?;
                    let job = Job { chunk, lat0, c: Chunk::of(chunk, nbh, nbw, topk, set.local_range), cs: Rope::time(f, lat0), noise: given };
                    // ---- the first card's blocks
                    cs0.view(0, cs0.len)?.write_async(Rope::hw_len(hp, wp) * 4, f32_bytes(&job.cs))?;
                    stage0.run(x0, cs0, &job.c, &mut |layer, x| {
                        if [0, 15, 29].contains(&layer) {
                            hooks.tensor(chunk, &format!("b{layer}"), x, rows)?;
                        }
                        Ok(())
                    })?;
                    // (the first card's memory at its fullest: the chunk's activations beside the ring)
                    if let Ok((_, Some(free))) = first_gpu.memory() {
                        low[0] = low[0].min(free);
                    }
                    // ---- the other cards' blocks, the head and the step: here, or on the back card's thread
                    match inline.as_mut() {
                        Some(b) => {
                            b.copy_in(&job, &mut staging)?;
                            b.run(&job, &mut staging, &mut |c, name, v, n| hooks.tensor(c, name, v, n))?;
                        }
                        None => {
                            jtx.send(job).map_err(|_| Error("flashvsr: the second card's thread stopped".into()))?;
                            wait(&|m| matches!(m, Msg::Copied))?;
                        }
                    }
                }
                // the chunk before (pipelined), or this one: its latents to the decoder, its frames out
                let finish = if pipelined { pending.take() } else if done { None } else { Some((chunk, if chunk == 0 { 6 } else { 2 })) };
                if let Some((pc, f)) = finish {
                    if pipelined {
                        wait(&|m| matches!(m, Msg::Lat(c) if *c == pc))?;
                    }
                    if !Arc::ptr_eq(&dec_gpu, &last_gpu) {
                        let src = lat.view(0, 16 * f * h8 * w8 * 4)?;
                        lat_d.copy_from_peer(&src, &mut lat_staging)?;
                    }
                    if pipelined {
                        let _ = atx.send(());
                    }
                    // its condition: the frames it makes (chunk 0: 0-20 with frame 0 three times in front)
                    let (pre, cur) = if pc == 0 { (0, 21) } else { (8 * pc + 13, 8 * pc + 21) };
                    let mut bytes = Vec::with_capacity((cur - pre + 3) * fsize);
                    if pc == 0 {
                        for _ in 0..3 {
                            bytes.extend_from_slice(&get(0, &mut frames, base, &mut last)?);
                        }
                    }
                    for i in pre..cur {
                        bytes.extend_from_slice(&get(i, &mut frames, base, &mut last)?);
                    }
                    made += dec.chunk(&lat_d, pc, f, &bytes, ended, emit, hooks)?;
                    // each card's least free memory so far (the run's peak, for its log line)
                    for (l, gpu) in low.iter_mut().zip(&self.gpus) {
                        if let Ok((_, Some(free))) = gpu.memory() {
                            *l = (*l).min(free);
                        }
                    }
                }
                if done {
                    break;
                }
                if pipelined {
                    pending = Some((chunk, if chunk == 0 { 6 } else { 2 }));
                }
                // the frames no later chunk (nor the chunk still to decode) reads
                let keep = match pending {
                    Some((0, _)) => 0,
                    Some((pc, _)) => 8 * pc + 13,
                    None => 8 * (chunk + 1) + 13,
                };
                while base < keep && !frames.is_empty() {
                    frames.pop_front();
                    base += 1;
                }
                chunk += 1;
            }
            drop(jtx);
            Ok(())
        })?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!("flashvsr: {} x {} stream on {} card(s){}: at most in use {}", geo.sw, geo.sh, self.gpus.len(), if pipelined { ", pipelined" } else { "" },
                  low.iter().zip(&self.gpus).map(|(l, g)| {
                      let total = g.memory().map(|m| m.0).unwrap_or(0);
                      format!("{} {:.1} of {:.1} GiB", g.name, if *l == u64::MAX { 0.0 } else { gib(total - (*l).min(total)) }, gib(total))
                  }).collect::<Vec<_>>().join(", "));
        Ok(made as u32)
    }
}

/// A tile's largest side (1728 x 960 out, 1792 x 1024 inside: the platform's x2) and the tiles' overlap
const TILE_W: usize = 1792;
const TILE_H: usize = 1024;
const TILE_OVERLAP: usize = 128;
/// A tiled run's segments: frames made a pass, and the frames each tile's stream runs before them to fill its memory
const SEG: usize = 192;
const WARM: usize = 32;

/// Tiles along one side: (tile size, offsets)
pub fn tiles_along(total: usize, max: usize) -> (usize, Vec<usize>) {
    if total <= max {
        return (total, vec![0]);
    }
    let n = (total - TILE_OVERLAP).div_ceil(max - TILE_OVERLAP);
    let size = (total + (n - 1) * TILE_OVERLAP).div_ceil(n).div_ceil(128) * 128;
    let size = size.min(max);
    let offs = (0..n).map(|i| if n == 1 { 0 } else { i * (total - size) / (n - 1) }).collect();
    (size, offs)
}

/// Tile (x0, y0)'s frame `p` [tile_h, tile_w, 3] blended into `dst` [th, tw, 3]: the overlaps with the tiles before
/// (left, above) linearly, the rest copied
#[allow(clippy::too_many_arguments)]
pub fn blend_tile(dst: &mut [u8], tw: usize, p: &[u8], tile_w: usize, tile_h: usize, x0: usize, y0: usize, ovx: usize, ovy: usize) {
    for r in 0..tile_h {
        let ay = if r < ovy { (r as f32 + 0.5) / ovy as f32 } else { 1.0 };
        let o = ((y0 + r) * tw + x0) * 3;
        let q = r * tile_w * 3;
        // the columns blended: the left overlap, or the whole row inside the top one; the rest copied as they are
        let mixed = if ay < 1.0 { tile_w } else { ovx.min(tile_w) };
        for c in 0..mixed {
            let ax = if c < ovx { (c as f32 + 0.5) / ovx as f32 } else { 1.0 };
            let wgt = ax.min(ay);
            for ch in 0..3 {
                let i = o + c * 3 + ch;
                let (d, v) = (dst[i] as f32, p[q + c * 3 + ch] as f32);
                dst[i] = (d + (v - d) * wgt).round() as u8;
            }
        }
        dst[o + mixed * 3..o + tile_w * 3].copy_from_slice(&p[q + mixed * 3..q + tile_w * 3]);
    }
}

impl FlashVsr {
    /// A clip too big for a card whole: in tiles of the upscaled frame (each its own stream), over segments of SEG
    /// frames, each tile's stream starting WARM frames before its segment (their frames thrown away: the KV ring and the
    /// convolutions' memory filled as in one stream) and running 8 past it (the frames it looks ahead to). The noise is
    /// the clip's by latent frame, so the tiles agree; their overlaps are blended linearly. (nextcuda's scheme.)
    pub fn run_tiled(&self, geo: Geometry, set: &Settings, next: &mut dyn FnMut() -> Result<Option<Vec<u8>>>, emit: &mut dyn FnMut(Picture) -> Result<()>)
        -> Result<u32> {
        let (tw, th) = (geo.tw, geo.th);
        let (tile_w, xs) = tiles_along(tw, TILE_W);
        let (tile_h, ys) = tiles_along(th, TILE_H);
        let tgeo = Geometry { w: tile_w, h: tile_h, sw: tile_w, sh: tile_h, tw: tile_w, th: tile_h, left: 0, top: 0 };
        if !self.fits(&tgeo) {
            return Err(Error(format!("flashvsr: a {tile_w} x {tile_h} tile does not fit the card(s)")));
        }
        let fsize = tw * th * 3;
        let mut frames: VecDeque<Vec<u8>> = VecDeque::new();
        let mut base = 0usize;
        let mut ended: Option<usize> = None;
        let mut made = 0usize;
        let mut s = 0usize;
        // a finished segment's frames, handed out one a tile frame while the next segment runs
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
        let crop = |f: &Vec<u8>| -> Picture {
            let mut pic = Picture::new(geo.sw as u32, geo.sh as u32, 3);
            for r in 0..geo.sh {
                let at = ((geo.top + r) * tw + geo.left) * 3;
                pic.data[r * geo.sw * 3..(r + 1) * geo.sw * 3].copy_from_slice(&f[at..at + geo.sw * 3]);
            }
            pic
        };
        loop {
            // the segment [s, e), its stream from a, its frames up to e + 8 on hand (prepared on threads)
            let mut todo: Vec<std::thread::JoinHandle<Vec<u8>>> = Vec::new();
            while ended.is_none() && base + frames.len() + todo.len() < s + SEG + 8 {
                match next()? {
                    Some(f) => todo.push(std::thread::spawn(move || resize::prepare(&f, &geo))),
                    None => ended = Some(base + frames.len() + todo.len()),
                }
            }
            for h in todo {
                frames.push_back(h.join().map_err(|_| Error("flashvsr: a frame's preparation failed".into()))?);
            }
            let have = base + frames.len();
            let e = (s + SEG).min(ended.unwrap_or(usize::MAX)).min(have);
            if e <= s {
                for f in pending.drain(..) {
                    emit(crop(&f))?;
                    made += 1;
                }
                break;
            }
            let a = s.saturating_sub(WARM);
            let stop = (e + 8).min(have);
            let mut seg: Vec<Vec<u8>> = vec![vec![0u8; fsize]; e - s];
            for (j, &y0) in ys.iter().enumerate() {
                for (i, &x0) in xs.iter().enumerate() {
                    let mut idx = a;
                    let mut out = a;
                    let mut st = *set;
                    st.lat_base = a / 4;
                    let fr = &frames;
                    let ovx = if i > 0 { xs[i - 1] + tile_w - x0 } else { 0 };
                    let ovy = if j > 0 { ys[j - 1] + tile_h - y0 } else { 0 };
                    self.run(tgeo, &st, true, &mut || {
                        if idx >= stop {
                            return Ok(None);
                        }
                        let src = &fr[idx - base];
                        let mut t = vec![0u8; tile_w * tile_h * 3];
                        for r in 0..tile_h {
                            let at = ((y0 + r) * tw + x0) * 3;
                            t[r * tile_w * 3..(r + 1) * tile_w * 3].copy_from_slice(&src[at..at + tile_w * 3]);
                        }
                        idx += 1;
                        Ok(Some(t))
                    }, &mut |p| {
                        let kf = out;
                        out += 1;
                        if kf < s || kf >= e {
                            return Ok(());
                        }
                        blend_tile(&mut seg[kf - s], tw, &p.data, tile_w, tile_h, x0, y0, ovx, ovy);
                        // the segment before, a frame at a time beside this work (its frames are not held back for a burst)
                        if let Some(f) = pending.pop_front() {
                            emit(crop(&f))?;
                            made += 1;
                        }
                        Ok(())
                    }, &mut NoHooks)?;
                }
            }
            for f in pending.drain(..) {
                emit(crop(&f))?;
                made += 1;
            }
            pending.extend(seg);
            s = e;
            // the next segment's warm-up onward stays
            let keep = s.saturating_sub(WARM);
            while base < keep && !frames.is_empty() {
                frames.pop_front();
                base += 1;
            }
        }
        Ok(made as u32)
    }

    /// Whether this request's size runs in tiles (NS_FVSR_TILE=1 / 0 force it)
    fn tiled(&self, geo: &Geometry) -> bool {
        match std::env::var("NS_FVSR_TILE").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => !self.fits(geo),
        }
    }
}

/// A frame as RGB bytes of the clip's size (alpha dropped)
fn rgb(p: Picture, w: u32, h: u32) -> Result<Vec<u8>> {
    if (p.width, p.height) != (w, h) {
        return Err(Error(format!("a frame of {}x{} in a clip of {w}x{h}", p.width, p.height)));
    }
    Ok(match p.channels {
        3 => p.data,
        4 => p.data.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        1 => p.data.iter().flat_map(|&v| [v, v, v]).collect(),
        c => return Err(Error(format!("a picture of {c} channels"))),
    })
}

impl ImageEngine for FlashVsr {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("flashvsr upscales clips (video.upscale) and pictures (image.upscale); it does not draw".into()))
    }
    fn report(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.stages.iter().map(|s| format!("flashvsr on {}: blocks {}-{}{}", s.gpu.name, s.from, s.to - 1,
                                            if Arc::ptr_eq(&s.gpu, &self.gpus[inner.dec]) { ", the decoder" } else { "" })).collect()
    }
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        VARIANTS.to_vec()
    }
    /// A picture as a still clip (FlashVSR needs a clip: the frame repeated to its 25), its first frame back
    fn upscale(&self, req: &UpscaleRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Picture, UpscaleInfo)> {
        let mut once = Some(req.picture.clone());
        let mut got = None;
        let info = self.upscale_clip(req, &mut || Ok(once.take()), &mut |p| {
            if got.is_none() {
                got = Some(p);
            }
            Ok(())
        })?;
        progress(Step { picture: 0, at: 1, of: 1, seconds: info.seconds })?;
        Ok((got.ok_or_else(|| Error("flashvsr made no frame".into()))?, info))
    }
    fn upscales_clips(&self) -> bool {
        true
    }
    fn upscale_clip(&self, req: &UpscaleRequest, next: &mut dyn FnMut() -> Result<Option<Picture>>, put: &mut dyn FnMut(Picture) -> Result<()>)
        -> Result<UpscaleInfo> {
        let t0 = Instant::now();
        let set = Settings::of(req.variant.as_deref(), req.scale, req.seed, &req.extra)?;
        let (w, h) = (req.picture.width, req.picture.height);
        let geo = Geometry::new(w as usize, h as usize, set.scale);
        let mut next_rgb = || -> Result<Option<Vec<u8>>> { next()?.map(|p| rgb(p, w, h)).transpose() };
        let tiled = self.tiled(&geo);
        let n = if tiled { self.run_tiled(geo, &set, &mut next_rgb, put) } else { self.run(geo, &set, false, &mut next_rgb, put, &mut NoHooks) };
        // the clip's buffers let go (the KV ring, the decoder's past, the projection's frames: 13-17 GiB at 1728 x 960)
        // so an idle server leaves its cards to the other models (another upscaler on the same card answered no_room
        // beside it); the next clip makes them again in a few ms
        let released = self.release();
        let n = n?;
        released?;
        let seconds = t0.elapsed().as_secs_f64();
        eprintln!("flashvsr: {w}x{h} -> {}x{}{}, {n} frames in {seconds:.1} s ({:.3} s a frame) on {} card(s); sparse_ratio {}, local_range {}", geo.sw, geo.sh,
                  if tiled { " in tiles" } else { "" }, seconds / n.max(1) as f64, self.gpus.len(), set.sparse_ratio, set.local_range);
        for line in self.gpus.iter().flat_map(|g| prof::table(g)) {
            eprintln!("{line}");
        }
        Ok(UpscaleInfo { variant: VARIANTS[0].id.into(), scale: set.scale as u32, noise: None, tile: 0, tiles: n, seconds })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_film_sizes_pad_to_whole_windows() {
        let g = Geometry::new(864, 480, 2.0);
        assert_eq!((g.sw, g.sh, g.tw, g.th), (1728, 960, 1792, 1024));
        let g = Geometry::new(1280, 704, 2.0);
        assert_eq!((g.sw, g.sh, g.tw, g.th, g.left, g.top), (2560, 1408, 2560, 1408, 0, 0));
    }

    #[test]
    fn x4_runs_as_two_by_two_tiles() {
        let g = Geometry::new(864, 480, 4.0);
        assert_eq!((g.tw, g.th), (3456, 1920));
        let (w, xs) = tiles_along(g.tw, TILE_W);
        let (h, ys) = tiles_along(g.th, TILE_H);
        assert_eq!((w, h, xs.len(), ys.len()), (1792, 1024, 2, 2));
        // the tiles cover the frame and overlap by at least the blend band
        assert_eq!(xs.last().unwrap() + w, g.tw);
        assert_eq!(ys.last().unwrap() + h, g.th);
        assert!(xs[0] + w - xs[1] >= TILE_OVERLAP && ys[0] + h - ys[1] >= TILE_OVERLAP);
    }

    #[test]
    fn a_blend_fades_across_the_overlap() {
        // two tiles of 4 x 1 in a frame of 6 x 1, overlapping by 2: the second's values ramp in
        let mut dst = vec![0u8; 6 * 3];
        blend_tile(&mut dst, 6, &[100u8; 12], 4, 1, 0, 0, 0, 0);
        blend_tile(&mut dst, 6, &[200u8; 12], 4, 1, 2, 0, 2, 0);
        let px: Vec<u8> = dst.chunks(3).map(|c| c[0]).collect();
        assert_eq!(px, vec![100, 100, 125, 175, 200, 200]);
    }

    #[test]
    fn a_full_card_is_no_room_and_a_size_no_card_takes_is_not() {
        let gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
        // 14 GiB more beside 3.6 of weights, on an idle 31.9 GiB card
        assert!(room_for("B70", "1728 x 960", gib(14.0), gib(3.6), gib(31.9), gib(28.0)).is_ok());
        // another process holds 20 GiB: retryable
        let e = room_for("B70", "1728 x 960", gib(14.0), gib(3.6), gib(31.9), gib(8.0)).unwrap_err();
        assert!(e.is_no_room(), "{e}");
        // more than the card has in all: change the request, not wait
        let e = room_for("B70", "5120 x 2816", gib(60.0), gib(3.6), gib(31.9), gib(28.0)).unwrap_err();
        assert!(!e.is_no_room(), "{e}");
    }

    #[test]
    fn settings_take_the_platforms_keys() {
        let mut x = nextsycl_image::Given::new();
        x.insert("sparse_ratio".into(), "2.0".into());
        x.insert("local_range".into(), "9".into());
        x.insert("color_fix".into(), "0".into());
        let s = Settings::of(Some("flashvsr-v1.1"), Some(4), Some(7), &x).unwrap();
        assert_eq!((s.scale, s.seed, s.sparse_ratio, s.local_range, s.color_fix), (4.0, 7, 2.0, 9, false));
        let d = Settings::of(None, None, None, &Default::default()).unwrap();
        assert_eq!((d.scale, d.seed, d.sparse_ratio, d.local_range, d.color_fix), (2.0, 0, 1.5, 11, true));
        assert!(Settings::of(None, Some(1), None, &Default::default()).is_err());
        x.insert("local_range".into(), "10".into());
        assert!(Settings::of(None, None, None, &x).is_err());
        assert!(Settings::of(Some("seedvr2-3b"), None, None, &Default::default()).is_err());
    }

    #[test]
    fn the_platforms_sizes_fit_one_card_at_x2() {
        // the stream's memory beside the weights (3.04 GiB measured) on a 31.9 GiB card (the B70, the B65)
        let total = (31.9 * (1u64 << 30) as f64) as u64;
        let weights = (3.04 * (1u64 << 30) as f64) as u64;
        for (w, h) in [(864, 480), (1280, 704)] {
            let geo = Geometry::new(w, h, 2.0);
            let (hp, wp) = (geo.th / 16, geo.tw / 16);
            let need = Stage::need_bytes_for(dit::LAYERS, (hp / 8) * (wp / 8)) + LqProj::need_bytes(hp, wp) + Tcd::need_bytes(geo.th / 8, geo.tw / 8)
                + FlashVsr::stream_bytes(&geo);
            assert!(room_for("B70", "x2", need as u64, weights, total, total).is_ok(), "{w} x {h}: {:.1} GiB", need as f64 / (1u64 << 30) as f64);
        }
    }

    #[test]
    fn the_split_is_thirty_blocks() {
        assert_eq!(split(None, 1).unwrap(), vec![30]);
        assert_eq!(split(None, 2).unwrap(), vec![14, 16]);
        assert_eq!(split(Some("20,10"), 2).unwrap(), vec![20, 10]);
        assert!(split(Some("20,11"), 2).is_err());
        assert!(split(Some("10,10,10"), 2).is_err());
    }
}
