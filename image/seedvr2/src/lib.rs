//! SeedVR2 (ByteDance-Seed, Apache-2.0) in Rust and SYCL: one-step diffusion restoration and upscaling of video, the
//! 3B and 7B NaDiTs over a causal 3-D VAE. The official inference (projects/inference_seedvr2_3b.py) on the shared
//! diffusion kernels and this engine's own (kernels/image/seedvr2); no Python.
//!
//! ```text
//!   frames -> bicubic (antialiased) to the size asked, a centre crop to multiples of 16, [-1, 1]
//!          -> the VAE encoder (vae.rs): the condition latent [t, h, w, 16] (its mean, times 0.9152)
//!   noise (seeded) | condition | 1 -> the DiT at t = 1000 (dit.rs): v; Euler's one step to the end: x0 = noise - v
//!   x0 -> the VAE decoder -> frames -> the wavelet colour fix against the resized input -> RGB
//! ```
//!
//! A clip goes through in chunks of `chunk` frames (1 + 4 n; a shorter last one is padded with its last frame, as the
//! reference pads a clip), each chunk the whole reference pipeline on its own - numz's ComfyUI node does the same -,
//! neighbouring chunks overlapping by `overlap` frames, crossfaded over the whole overlap. The DiT's windows join a chunk's
//! frames in time (four to eight latent frames a window), which is what keeps a chunk steady; the overlap carries it
//! across the chunks' seams.

pub mod check;
pub mod dit;
pub mod ffi;
pub mod prof;
pub mod vae;
pub mod window;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{At, DevBuf, EngineOption, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step,
    UpscaleInfo, UpscaleRequest, UpscaleVariant};

use crate::vae::Vol;

pub const ARCH: &str = "seedvr2";
/// the VAE, the positive prompt's embedding (bfloat16 [L, 5120]: pos_emb.pt's storage), and a DiT a variant
pub const ROLES: &[&str] = &["vae", "text", "dit-3b", "dit-7b"];
/// the text encoder's width (the embedding's rows)
const TEXT: usize = 5120;

const VARIANTS: &[UpscaleVariant] = &[
    UpscaleVariant { id: "seedvr2-3b", about: "SeedVR2 3B: one-step restoration with natural detail, temporally steady (chunks of frames together)",
                     scales: &[1, 2, 3, 4], noise: &[] },
    UpscaleVariant { id: "seedvr2-7b", about: "SeedVR2 7B: the larger DiT, finer detail, about twice the 3B's time", scales: &[1, 2, 3, 4], noise: &[] },
];

const OPTIONS: &[EngineOption] = &[
    EngineOption { name: "chunk", env: "NS_SVR_CHUNK", value: "1 + 4 n frames", help: "the frames that go through the model together (default 33)",
                   at: At::Load },
    EngineOption { name: "overlap", env: "NS_SVR_OVERLAP", value: "frames", help: "the frames neighbouring chunks share, crossfaded (default 4)",
                   at: At::Load },
    EngineOption { name: "seed", env: "NS_SVR_SEED", value: "a whole number", help: "the noise's seed (default 666, the reference's)", at: At::Load },
];

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "SeedVR2 video restoration and upscaling (3B / 7B NaDiT, causal 3-D VAE; Rust + SYCL)", roles: ROLES, load,
                options: OPTIONS }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(SeedVr2::load(files, gpus, o, log)?))
}

/// Each card's lowest free memory seen at the engine's largest allocations (a clip's peak, for its log line)
static LOW: [AtomicU64; 8] = [const { AtomicU64::new(u64::MAX) }; 8];

/// Notes so far (the driver is asked one in 16, or each with NS_SVR_MEMLOG: a query is not free)
static NOTES: AtomicU64 = AtomicU64::new(0);

/// The card's free memory now, the lowest kept (NS_SVR_MEMLOG: each one printed, with where)
pub(crate) fn note(gpu: &Gpu, what: &str) {
    let log = std::env::var_os("NS_SVR_MEMLOG").is_some();
    if !log && !NOTES.fetch_add(1, Ordering::Relaxed).is_multiple_of(16) {
        return;
    }
    if let Ok((total, Some(free))) = gpu.memory() {
        LOW[gpu.index % 8].fetch_min(free, Ordering::Relaxed);
        if log {
            eprintln!("seedvr2 memory: {:.2} GiB in use at {what}", (total - free.min(total)) as f64 / (1u64 << 30) as f64);
        }
    }
}

/// The wall clock in seconds (the chunk lines' stamps, for a timeline of a run)
fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// A seeded standard-normal draw (splitmix64, Box-Muller): the starting noise
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (next().max(1e-300), next());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * v).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Python's round (halves to even)
fn round_even(x: f64) -> i64 {
    let r = x.round();
    (if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }) as i64
}

/// The model's picture size for a source of w x h at `scale`: the reference's area resize (each side times the
/// scale, rounded), then the centre crop to multiples of 16 (torchvision's: the offset rounded). (resized w, h, crop x,
/// y, out w, h)
pub fn geometry(w: u32, h: u32, scale: f64) -> (usize, usize, usize, usize, usize, usize) {
    let (rw, rh) = (round_even(w as f64 * scale).max(16) as usize, round_even(h as f64 * scale).max(16) as usize);
    let (ow, oh) = (rw - rw % 16, rh - rh % 16);
    (rw, rh, round_even((rw - ow) as f64 / 2.0) as usize, round_even((rh - oh) as f64 / 2.0) as usize, ow, oh)
}

/// The card memory a chunk of frames this size takes beyond what the loaded engine holds (GiB): the decoder's kept
/// frames and the full-resolution slices grow with the pixels (24.5 GiB peak at 1728x960, the VAE's ~1 GiB in it)
pub fn chunk_gib(rw: usize, rh: usize) -> f64 {
    23.5 * (rw * rh) as f64 / (1728.0 * 960.0)
}

/// Kept free on a card beside a run (the box spills and hangs past its memory)
const MARGIN_GIB: f64 = 1.5;

/// Whether a chunk of `rw` x `rh` fits beside what holds the card now: Err no room (it fits an empty card; the same
/// call may work later) or a size no card of this one's memory takes
pub fn room_for(card: &str, rw: usize, rh: usize, total_gib: f64, free_gib: f64) -> Result<()> {
    let need = chunk_gib(rw, rh) + MARGIN_GIB;
    // the engine's own (the VAE, the runtime) is held on an empty card too
    if need + 1.5 > total_gib {
        return Err(Error(format!("seedvr2: {rw} x {rh} needs {need:.1} GiB on {card} and it has {total_gib:.1} in all: a smaller scale or source")));
    }
    if need > free_gib {
        return Err(Error::no_room(format!("seedvr2: {rw} x {rh} needs {:.1} GiB more on {card} and it has {free_gib:.1} free", need - free_gib)));
    }
    Ok(())
}

/// The earlier chunk's weights over `n` overlapping frames: a raised cosine across the whole overlap (4 frames: 0.90,
/// 0.65, 0.35, 0.10). ComfyUI-SeedVR2's Hann window fades over the middle third only, which for its 4-frame overlap
/// is a cut between the second and third frame: a jump in the detail at every seam (measured on an H3 clip: the face
/// changed 2-3x as much there as between other frames)
pub fn crossfade(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + 0.5 * (std::f32::consts::PI * (i + 1) as f32 / (n + 1) as f32).cos()).collect()
}

/// A chunk's frame count: 1 + 4 n (a frame or more)
fn chunk_len(n: usize) -> usize {
    if n <= 1 { 1 } else { 1 + 4 * (n - 1).div_ceil(4) }
}

/// One card's share: its kernels' context, its VAE and the DiT it holds (in pinned host memory, up for each call)
pub struct Card {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    pub vae: vae::Vae,
    /// the DiT loaded and its variant (one at a time: the 7B's is 16 GB)
    dit: Mutex<Option<(&'static str, Arc<dit::Dit>)>>,
    /// the last clip shapes' windows (keyed by the DiT and the shape)
    shapes: Mutex<Vec<ShapeKey>>,
}

/// A kept clip shape: (the DiT, t, h, w) and its windows
type ShapeKey = ((usize, usize, usize, usize), Arc<dit::Shape>);

/// A chunk for a card: its frames (RGB bytes of the source's size, 1 + 4 n of them), where it sits in the clip
struct Job {
    index: usize,
    bytes: Vec<u8>,
    frames: usize,
}

pub struct SeedVr2 {
    /// the cards it runs on (the registry entry's GPUs): a clip's chunks go to whichever is free
    pub cards: Vec<Card>,
    /// the positive prompt's embedding, bfloat16 [L, 5120], and L
    text: Vec<u16>,
    text_len: usize,
    /// each variant's DiT file
    dits: BTreeMap<&'static str, PathBuf>,
    pub chunk: usize,
    pub overlap: usize,
    pub seed: u64,
    load_s: f64,
    busy: Mutex<()>,
}

fn number(o: &LoadOptions, name: &str, default: u64) -> Result<u64> {
    match o.setting(name) {
        None => Ok(default),
        Some(v) => v.trim().parse().map_err(|_| Error(format!("{name}={v}: a whole number"))),
    }
}

impl SeedVr2 {
    pub fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<SeedVr2> {
        let t0 = Instant::now();
        nextsycl_core::use_kind("image");
        if gpus.is_empty() {
            return Err(Error("no GPU".into()));
        }
        let role = |r: &str| files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let raw = std::fs::read(role("text")?).map_err(|e| Error(format!("the text embedding: {e}")))?;
        if raw.is_empty() || raw.len() % (TEXT * 2) != 0 {
            return Err(Error(format!("the text embedding is {} bytes: not bfloat16 rows of {TEXT} (pos_emb.pt's data/0)", raw.len())));
        }
        let text: Vec<u16> = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        let text_len = text.len() / TEXT;
        let mut dits = BTreeMap::new();
        for v in VARIANTS {
            if let Some(p) = files.get(&format!("dit-{}", &v.id[8..])) {
                dits.insert(v.id, p.clone());
            }
        }
        if dits.is_empty() {
            return Err(Error(format!("{ARCH}: no DiT (roles dit-3b, dit-7b)")));
        }
        let chunk = number(o, "NS_SVR_CHUNK", 33)? as usize;
        if chunk == 0 || !(chunk - 1).is_multiple_of(4) {
            return Err(Error(format!("NS_SVR_CHUNK={chunk}: 1 + 4 n frames")));
        }
        let overlap = number(o, "NS_SVR_OVERLAP", 4)? as usize;
        if overlap >= chunk {
            return Err(Error(format!("NS_SVR_OVERLAP={overlap}: fewer than the chunk's {chunk} frames")));
        }
        let vae_file = role("vae")?;
        // NS_SVR_GPUS: the cards to run on, by index (a server is handed one; a clip's chunks can go to several)
        let mine: Vec<Arc<Gpu>> = match o.setting("NS_SVR_GPUS").filter(|v| !v.trim().is_empty()) {
            Some(v) => v.split(',').map(|i| i.trim().parse::<usize>().map_err(|_| Error(format!("NS_SVR_GPUS={v}: card numbers, comma separated")))
                .and_then(Gpu::open)).collect::<Result<_>>()?,
            None => gpus.to_vec(),
        };
        // NS_SVR_MIN_FREE_GIB: no load on a card with less free than that (another process holding it: a run beside it
        // would push the card past its memory)
        if let Some(min) = o.setting("NS_SVR_MIN_FREE_GIB").and_then(|v| v.trim().parse::<f64>().ok()).filter(|m| *m > 0.0) {
            for gpu in &mine {
                if let Ok((_, Some(free))) = gpu.memory() {
                    let gib = free as f64 / (1u64 << 30) as f64;
                    if gib < min {
                        return Err(Error(format!("{}: {gib:.1} GiB free, less than NS_SVR_MIN_FREE_GIB={min}", gpu.name)));
                    }
                }
            }
        }
        let mut cards = Vec::new();
        for gpu in &mine {
            cards.push(Card { gpu: gpu.clone(), nsd: Nsd::new(gpu)?, vae: vae::Vae::load(&vae_file, gpu, log)?, dit: Mutex::new(None),
                              shapes: Mutex::new(Vec::new()) });
        }
        let e = SeedVr2 { cards, text, text_len, dits, chunk, overlap, seed: number(o, "NS_SVR_SEED", 666)?, load_s: 0.0, busy: Mutex::new(()) };
        // the first variant's DiT now (the 3B when there is one); another on its first request
        e.dits_for(e.dits.keys().next().expect("checked"))?;
        Ok(SeedVr2 { load_s: t0.elapsed().as_secs_f64(), ..e })
    }

    /// A card's DiT of a variant, loaded (the other let go first)
    pub fn dit_for(&self, card: &Card, variant: &'static str, log: &mut dyn FnMut(String)) -> Result<Arc<dit::Dit>> {
        let mut g = card.dit.lock().unwrap();
        if let Some((v, d)) = g.as_ref() {
            if *v == variant {
                return Ok(d.clone());
            }
        }
        let path = self.dits.get(variant).ok_or_else(|| Error(format!("{variant}: its DiT is not here (pulled: {})",
                                                                      self.dits.keys().copied().collect::<Vec<_>>().join(", "))))?;
        *g = None;
        let d = Arc::new(dit::Dit::load(path, &self.text, self.text_len, &card.nsd, log)?);
        *g = Some((variant, d.clone()));
        Ok(d)
    }

    /// Every card's DiT of a variant, loaded side by side (each card's own pinned copy)
    fn dits_for(&self, variant: &'static str) -> Result<Vec<Arc<dit::Dit>>> {
        std::thread::scope(|sc| {
            let hs: Vec<_> = self.cards.iter().map(|c| sc.spawn(move || self.dit_for(c, variant, &mut |m| eprintln!("{m}")))).collect();
            hs.into_iter().map(|h| h.join().map_err(|_| Error("a DiT load panicked".into()))?).collect()
        })
    }

    fn variant(&self, req: &UpscaleRequest) -> Result<&'static str> {
        match req.variant.as_deref() {
            None | Some("") => Ok(self.cards[0].dit.lock().unwrap().as_ref().map(|(v, _)| *v).unwrap_or(VARIANTS[0].id)),
            Some(v) => VARIANTS.iter().map(|x| x.id).find(|x| *x == v)
                .ok_or_else(|| Error(format!("variant {v}: seedvr2-3b or seedvr2-7b"))),
        }
    }
}

impl Card {
    /// The windows of a latent clip shape for this card's DiT: made once and kept (a clip's chunks are all one shape
    /// but the last; the rotary tables take the host a while)
    fn shape(&self, dit: &Arc<dit::Dit>, t: usize, h: usize, w: usize) -> Result<Arc<dit::Shape>> {
        let mut g = self.shapes.lock().unwrap();
        if let Some((_, s)) = g.iter().find(|(k, _)| *k == (Arc::as_ptr(dit) as usize, t, h, w)) {
            return Ok(s.clone());
        }
        let s = Arc::new(dit.shape(t, h, w)?);
        g.truncate(2);
        g.insert(0, ((Arc::as_ptr(dit) as usize, t, h, w), s.clone()));
        Ok(s)
    }

    /// One chunk through the whole pipeline: frames of RGB bytes (the source's size, 1 + 4 n of them) -> the made
    /// frames, RGB bytes of the out size
    pub fn chunk(&self, dit: &Arc<dit::Dit>, rgb: &[u8], frames: usize, src: (u32, u32), scale: f64, seed: u64) -> Result<Vec<Vec<u8>>> {
        let gpu = &self.gpu;
        let nsd = &self.nsd;
        let k = ffi::api()?;
        let (rw, rh, cx, cy, ow, oh) = geometry(src.0, src.1, scale);
        let w0 = wall();
        let up = DevBuf::new(gpu, rgb.len())?;
        prof::timed(gpu, "frames up (host -> card)", 0.0, rgb.len() as f64, || up.write(0, rgb))?;
        let input = Vol::new(gpu, frames, oh, ow, 3)?;
        prof::timed(gpu, "resize (bicubic, antialiased)", 0.0, (rgb.len() + 4 * frames * src.1 as usize * ow * 3 * 2 + frames * oh * ow * 3 * 2) as f64, || {
            // SAFETY: frames of the source's size in, frames of oh x ow x 3 out.
            ffi::check(unsafe {
                (k.resize)(gpu.raw(), up.ptr().cast(), frames as i64, src.1 as i64, src.0 as i64, rh as i64, rw as i64, cy as i64, cx as i64, oh as i64,
                           ow as i64, input.buf.ptr())
            }, "resize")
        })?;
        // the resize reads it: done before it goes (a free does not wait for the queue)
        nsd.wait()?;
        drop(up);
        let t0 = Instant::now();
        let cond = self.vae.encode(nsd, &input)?;
        let t_enc = t0.elapsed().as_secs_f64();
        vae::settle(nsd)?;
        let (t, h, w) = (1 + (frames - 1) / 4, oh / 8, ow / 8);
        let n = noise(seed, cond.len());
        let s = self.shape(dit, t, h, w)?;
        let t1 = Instant::now();
        let mut v = dit.forward(nsd, &s, &n, &cond, &mut |_, _, _| Ok(()))?;
        // a call that came back non-finite (now and then, cause open: see docs/ports/seedvr2.md) is made again, each
        // block's output checked on the way to say where it starts; the second one is used when it is whole
        if v.iter().any(|x| !x.is_finite()) {
            let mut first = None;
            let again = dit.forward(nsd, &s, &n, &cond, &mut |i, xv, xt| {
                if first.is_none() && (xv.to_f32()?.iter().any(|x| !x.is_finite()) || xt.to_f32()?.iter().any(|x| !x.is_finite())) {
                    first = Some(i);
                }
                Ok(())
            })?;
            let ok = again.iter().all(|x| x.is_finite());
            let (bad, at) = dit.verify()?;
            eprintln!("seedvr2 chunk: the DiT's output was non-finite; made again: {} (non-finite from block {}); its weights on the card: {bad} bytes \
                       differ from the host's{}", if ok { "whole" } else { "non-finite again" }, first.map_or("none".to_string(), |b| b.to_string()),
                      at.map(|(f, t)| format!(", the first at {f} ({t})")).unwrap_or_default());
            if !ok {
                return Err(Error("the DiT's output is non-finite twice".into()));
            }
            v = again;
        }
        // the VAE's slices at full resolution want the room: the DiT's weights go back up for the next chunk's call
        dit.evict();
        let t_dit = t1.elapsed().as_secs_f64();
        // a check that nothing went non-finite (a black chunk would be one)
        let bad = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
        if bad(&cond) + bad(&v) > 0 {
            eprintln!("seedvr2 chunk: NON-FINITE values: {} in the condition latent, {} in the DiT's output", bad(&cond), bad(&v));
        }
        let x0: Vec<f32> = n.iter().zip(&v).map(|(a, b)| a - b).collect();
        let mut out = Vec::with_capacity(frames);
        let fb = oh * ow * 3;
        let tmp = DevBuf::f32(gpu, 2 * 5 * fb)?;
        self.vae.decode(nsd, &x0, t, h, w, &mut |dec| {
            let f0 = out.len();
            let nf = dec.t.min(frames - f0);
            let fixed = Vol::new(gpu, nf, oh, ow, 3)?;
            let style = input.buf.ptr().cast::<u16>().wrapping_add(f0 * fb).cast();
            let bytes = DevBuf::new(gpu, nf * fb)?;
            // five blur passes over float32 frames, the two pictures read, the result and the bytes written
            prof::timed(gpu, "colour fix + to RGB bytes", 0.0, (nf * fb) as f64 * (2.0 * 2.0 + 4.0 * 2.0 + 5.0 * 9.0 * 4.0 + 2.0 + 1.0), || unsafe {
                // SAFETY: nf frames of oh x ow x 3 in each, the scratch 2 x 5 frames of floats (a slice is at most 5).
                ffi::check((k.colorfix)(gpu.raw(), dec.buf.ptr(), style, nf as i64, oh as i64, ow as i64, tmp.fp(), fixed.buf.ptr()), "colour fix")?;
                ffi::check((k.to_rgb8)(gpu.raw(), fixed.buf.ptr(), (nf * fb) as i64, bytes.ptr().cast()), "to rgb")
            })?;
            nsd.wait()?;
            let mut host = vec![0u8; nf * fb];
            prof::timed(gpu, "frames down (card -> host)", 0.0, (nf * fb) as f64, || bytes.read(0, &mut host))?;
            let black = host.iter().step_by(97).filter(|&&b| b == 0).count() * 97;
            if black * 10 > host.len() * 9 {
                eprintln!("seedvr2 chunk: frames {f0}.. came out black (the colour fix's input non-finite?)");
            }
            out.extend(host.chunks_exact(fb).map(|f| f.to_vec()));
            Ok(())
        })?;
        vae::settle(nsd)?;
        let t_all = t0.elapsed().as_secs_f64();
        eprintln!("seedvr2 chunk: {frames} frames at {ow}x{oh} on {} (gpu {}): encode {t_enc:.1} s, dit {t_dit:.1} s (its weights up included), decode {:.1} s; \
                   wall {w0:.1} (encode from {:.1}, dit from {:.1}, decode from {:.1}) to {:.1}", gpu.name, gpu.index, t_all - t_enc - t_dit,
                  w0 + (wall() - w0 - t_all), w0 + (wall() - w0 - t_all) + t_enc, w0 + (wall() - w0 - t_all) + t_enc + t_dit, wall());
        for line in prof::table(gpu) {
            eprintln!("{line}");
        }
        Ok(out)
    }
}

impl ImageEngine for SeedVr2 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn options(&self) -> &'static [EngineOption] {
        OPTIONS
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error("seedvr2 restores and upscales pictures and clips (image.upscale, video.upscale); it does not draw them".into()))
    }
    fn report(&self) -> Vec<String> {
        self.cards.iter().map(|c| {
            let d = c.dit.lock().unwrap();
            format!("seedvr2 on {}: {}; chunks of {} frames, {} shared", c.gpu.name,
                    d.as_ref().map(|(v, x)| format!("{v} ({})", x.what)).unwrap_or_else(|| "no DiT loaded".into()), self.chunk, self.overlap)
        }).collect()
    }
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        VARIANTS.iter().filter(|v| self.dits.contains_key(v.id)).cloned().collect()
    }
    fn upscale(&self, req: &UpscaleRequest, progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<(Picture, UpscaleInfo)> {
        let mut once = Some(req.picture.clone());
        let mut got = None;
        let info = self.upscale_clip(req, &mut || Ok(once.take()), &mut |p| {
            got = Some(p);
            Ok(())
        })?;
        progress(Step { picture: 0, at: 1, of: 1, seconds: info.seconds })?;
        Ok((got.ok_or_else(|| Error("no picture made".into()))?, info))
    }
    fn upscales_clips(&self) -> bool {
        true
    }
    fn upscale_clip(&self, req: &UpscaleRequest, next: &mut dyn FnMut() -> Result<Option<Picture>>, put: &mut dyn FnMut(Picture) -> Result<()>)
        -> Result<UpscaleInfo> {
        let t0 = Instant::now();
        let _busy = self.busy.lock().unwrap();
        LOW.iter().for_each(|l| l.store(u64::MAX, Ordering::Relaxed));
        let variant = self.variant(req)?;
        let scale = req.scale.unwrap_or(2).clamp(1, 4) as f64;
        let dits = self.dits_for(variant)?;
        let (sw, sh) = (req.picture.width, req.picture.height);
        let (rw, rh, _, _, ow, oh) = geometry(sw, sh, scale);
        // every card the chunks go to has the room now, or no room (retryable) before a frame is read
        for card in &self.cards {
            if let Ok((total, Some(free))) = card.gpu.memory() {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                room_for(&card.gpu.name, rw, rh, gib(total), gib(free))?;
            }
        }
        let fb_in = sw as usize * sh as usize * 3;
        // a frame as RGB bytes of the source's size (alpha dropped)
        let rgb = |p: Picture| -> Result<Vec<u8>> {
            if (p.width, p.height) != (sw, sh) {
                return Err(Error(format!("a frame of {}x{} in a clip of {sw}x{sh}", p.width, p.height)));
            }
            Ok(match p.channels {
                3 => p.data,
                4 => p.data.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
                1 => p.data.iter().flat_map(|&v| [v, v, v]).collect(),
                c => return Err(Error(format!("a picture of {c} channels"))),
            })
        };
        let pic = |data: Vec<u8>| Picture { width: ow as u32, height: oh as u32, channels: 3, data };
        let step = self.chunk - self.overlap;
        // One worker a card takes the chunks as they come (a free card the next one); this thread reads the frames,
        // makes the chunks, and fades and hands out the made frames in order - so the host's work and the cards' run
        // side by side. At most one chunk waits beyond those running.
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let job_rx = Mutex::new(job_rx);
        let (res_tx, res_rx) = std::sync::mpsc::channel::<(usize, Result<Vec<Vec<u8>>>)>();
        let chunks = std::thread::scope(|sc| -> Result<u32> {
            for (card, dit) in self.cards.iter().zip(&dits) {
                let (job_rx, res_tx) = (&job_rx, res_tx.clone());
                let seed = self.seed;
                sc.spawn(move || loop {
                    let job = job_rx.lock().unwrap().recv();
                    let Ok(job) = job else { break };
                    let r = card.chunk(dit, &job.bytes, job.frames, (sw, sh), scale, seed.wrapping_add(job.index as u64));
                    if res_tx.send((job.index, r)).is_err() {
                        break;
                    }
                });
            }
            drop(res_tx);
            let mut run = || -> Result<u32> {
                // the frames read and not yet a chunk's; each chunk's (frames, shared, kept); the made frames of the last
                // chunk's overlap, not yet handed out
                let mut pending: Vec<Vec<u8>> = Vec::new();
                let mut meta: Vec<(usize, usize, usize)> = Vec::new();
                let mut tail: Vec<Vec<u8>> = Vec::new();
                let mut made_by: BTreeMap<usize, Vec<Vec<u8>>> = BTreeMap::new();
                let (mut ended, mut done, mut running, mut emitted) = (false, false, 0usize, 0usize);
                loop {
                    while !done && running < self.cards.len() + 1 {
                        while !ended && pending.len() < self.chunk {
                            match next()? {
                                Some(p) => pending.push(rgb(p)?),
                                None => ended = true,
                            }
                        }
                        // the frames this chunk adds (the first `overlap` repeat the last chunk's)
                        let shared = if meta.is_empty() { 0 } else { self.overlap.min(pending.len()) };
                        if pending.len() <= shared {
                            done = true;
                            break;
                        }
                        let real = pending.len().min(self.chunk);
                        let len = chunk_len(real);
                        let mut bytes = Vec::with_capacity(len * fb_in);
                        for i in 0..len {
                            bytes.extend_from_slice(&pending[i.min(real - 1)]);
                        }
                        let last = ended && pending.len() <= self.chunk;
                        meta.push((real, shared, if last { 0 } else { self.overlap.min(real) }));
                        job_tx.send(Job { index: meta.len() - 1, bytes, frames: len }).map_err(|_| Error("the cards' workers ended".into()))?;
                        running += 1;
                        if last {
                            done = true;
                        } else {
                            pending.drain(..step.min(pending.len()));
                        }
                    }
                    if running == 0 {
                        break;
                    }
                    let (i, r) = res_rx.recv().map_err(|_| Error("the cards' workers ended".into()))?;
                    running -= 1;
                    made_by.insert(i, r?);
                    // the chunks made so far, in order: the overlap faded from the earlier chunk, the rest handed out
                    while let Some(mut made) = made_by.remove(&emitted) {
                        let (real, shared, keep) = meta[emitted];
                        made.truncate(real);
                        let w = crossfade(shared);
                        for (j, prev) in tail.drain(..).enumerate() {
                            let a = w[j];
                            put(pic(prev.iter().zip(&made[j]).map(|(&p, &c)| (p as f32 * a + c as f32 * (1.0 - a)).round() as u8).collect()))?;
                        }
                        let n = made.len();
                        for f in made.drain(shared..n - keep) {
                            put(pic(f))?;
                        }
                        // made is now the overlap's frames (handed out above) and the `keep` the next chunk fades from
                        tail = made.split_off(shared);
                        emitted += 1;
                    }
                }
                for f in tail.drain(..) {
                    put(pic(f))?;
                }
                Ok(meta.len() as u32)
            };
            let r = run();
            // the workers end when the jobs do (a running chunk is finished first)
            drop(job_tx);
            r
        })?;
        let seconds = t0.elapsed().as_secs_f64();
        let use_of = |c: &Card| {
            let total = c.gpu.memory().map(|m| m.0).unwrap_or(0);
            let low = LOW[c.gpu.index % 8].load(Ordering::Relaxed);
            format!("{} {:.1} of {:.1} GiB", c.gpu.name, if low == u64::MAX { 0.0 } else { (total - low.min(total)) as f64 / (1u64 << 30) as f64 },
                    total as f64 / (1u64 << 30) as f64)
        };
        eprintln!("seedvr2: {variant} {sw}x{sh} -> {ow}x{oh}, {chunks} chunks of {} frames on {} card(s) in {seconds:.1} s; at most in use: {}", self.chunk,
                  self.cards.len(), self.cards.iter().map(use_of).collect::<Vec<_>>().join(", "));
        Ok(UpscaleInfo { variant: variant.into(), scale: scale as u32, noise: None, tile: self.chunk as u32, tiles: chunks, seconds })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_film_size_is_twice_the_source() {
        assert_eq!(geometry(864, 480, 2.0), (1728, 960, 0, 0, 1728, 960));
        // a side that is not a multiple of 16 is cropped about its centre
        assert_eq!(geometry(500, 300, 2.0), (1000, 600, 4, 4, 992, 592));
    }

    #[test]
    fn chunks_are_one_plus_four_n_frames() {
        assert_eq!((chunk_len(1), chunk_len(2), chunk_len(5), chunk_len(6), chunk_len(33)), (1, 5, 5, 9, 33));
    }

    #[test]
    fn the_crossfade_spans_the_overlap() {
        let w = crossfade(4);
        let want = [0.9045, 0.6545, 0.3455, 0.0955];
        assert!(w.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4), "{w:?}");
        // symmetric: the later chunk's weights are the earlier's reversed
        let w = crossfade(5);
        assert!((0..5).all(|i| (w[i] + w[4 - i] - 1.0).abs() < 1e-6) && (w[2] - 0.5).abs() < 1e-6, "{w:?}");
        assert!(crossfade(1).len() == 1 && (crossfade(1)[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_full_card_is_no_room_and_a_size_no_card_takes_is_not() {
        // the film size beside an idle card, then beside 10 GiB another process holds
        assert!(room_for("B70", 1728, 960, 31.9, 30.0).is_ok());
        let e = room_for("B70", 1728, 960, 31.9, 20.0).unwrap_err();
        assert!(e.is_no_room() && e.0.contains("needs 5.0 GiB more on B70 and it has 20.0 free"), "{e}");
        // 1280x704 x2: more than the card has in all - change the request, not wait
        let e = room_for("B70", 2560, 1408, 31.9, 30.0).unwrap_err();
        assert!(!e.is_no_room(), "{e}");
    }
}
