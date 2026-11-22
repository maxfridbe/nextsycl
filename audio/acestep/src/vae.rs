//! Oobleck (diffusers' AutoencoderOobleck, Stable Audio's VAE), ACE-Step 1.5's: 48 kHz stereo to 64 latent channels at
//! 25 Hz (strides 2, 4, 4, 6, 10) and back. Weight-normed 1-D convolutions (folded at load: w = g v / |v| over all
//! but the first axis), the snake with its own beta, residual units at dilations 1, 3, 9. float32 throughout. The
//! convolutions are the silo's (kernels/audio/acestep/silo: one oneDNN primitive each, float32 accumulation) when it is
//! there, else the shared kernels' plain loops (nsd_conv1d, many times slower). A song is decoded in windows of 256
//! latent frames with 64 more each side, the overlap discarded - the reference handler's tiled_decode, whose windows
//! these are.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::none;
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi;
use crate::nn::{f32_bytes, read_f32, Loader, Ops, Scratch, M, P};
use crate::sample::LATENT;

/// the reference's decode windows (latent frames): a core of 256 - 2 x 64, 64 either side (its CPU run's and a card
/// with 12 GB free; `auto_chunk`)
pub const CHUNK: usize = 256;
pub const OVERLAP: usize = 64;
/// samples a latent frame (2 x 4 x 4 x 6 x 10)
pub const HOP: usize = 1920;
/// the reference's encode windows (samples): 30 s, 2 s of overlap each side, on a card of more than 8 GB; 15 s on one
/// of less - here too when the 30 s window's activations (3 x 128 channels at the audio rate, 2.2 GB) do not fit beside
/// what is loaded
pub const ENC_CHUNK: usize = 48_000 * 30;
pub const ENC_CHUNK_SMALL: usize = 48_000 * 15;
pub const ENC_OVERLAP: usize = 48_000 * 2;
pub const RATE: u32 = 48_000;

struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    ci: usize,
    co: usize,
    k: usize,
}

struct Snake {
    alpha: DevBuf,
    inv_beta: DevBuf,
    c: usize,
}

struct Res {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
    dil: usize,
}

struct DecBlock {
    s: Snake,
    up: Conv,
    stride: usize,
    res: [Res; 3],
}

struct EncBlock {
    res: [Res; 3],
    s: Snake,
    down: Conv,
    stride: usize,
}

pub struct Vae {
    d_conv1: Conv,
    d_blocks: Vec<DecBlock>,
    d_snake: Snake,
    d_conv2: Conv,
    e_conv1: Option<Conv>,
    e_blocks: Vec<EncBlock>,
    e_snake: Option<Snake>,
    e_conv2: Option<Conv>,
    pub bytes: u64,
    gpu: Arc<Gpu>,
    /// the silo's convolutions (None: the shared kernels')
    silo: Option<&'static ffi::Silo>,
    peak: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64),
}

impl Drop for Vae {
    fn drop(&mut self) {
        if let Some(k) = self.silo {
            // the silo's copies of these weights (keyed by their addresses) go before the weights do
            // SAFETY: a live GPU handle.
            let _ = unsafe { (k.release)(self.gpu.raw()) };
        }
    }
}

/// NS_ACE_VAE_PROFILE=1: each VAE operation waited for and timed, by kind (`take_profile`)
fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NS_ACE_VAE_PROFILE").is_ok_and(|v| v.trim() == "1"))
}

#[allow(clippy::type_complexity)]
static PROFILE: Mutex<Vec<(String, f64, usize)>> = Mutex::new(Vec::new());

/// `f` timed under `what` when profiling (the queue drained before and after it), else just run
fn timed<T>(ops: &Ops, what: &dyn Fn() -> String, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !profiling() {
        return f();
    }
    ops.gpu.sync()?;
    let t0 = Instant::now();
    let r = f()?;
    ops.gpu.sync()?;
    let dt = t0.elapsed().as_secs_f64();
    let name = what();
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += dt;
            e.2 += 1;
        }
        None => p.push((name, dt, 1)),
    }
    Ok(r)
}

/// The profile gathered since the last call (operation kind, seconds, calls), the costliest first
pub fn take_profile() -> Vec<(String, f64, usize)> {
    let mut p = std::mem::take(&mut *PROFILE.lock().unwrap_or_else(|e| e.into_inner()));
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    p
}

fn conv(l: &Loader, p: &str, transposed: bool) -> Result<Conv> {
    let gname = format!("{p}.weight_g");
    let (w, shape) = if l.has(&gname) {
        let g = l.host(&gname)?;
        let v = l.host(&format!("{p}.weight_v"))?;
        let shape = l.st.need(&format!("{p}.weight_v")).map_err(|e| Error(e.0))?.shape.clone();
        let d0 = shape[0] as usize;
        let rest = v.len() / d0;
        let mut w = vec![0f32; v.len()];
        for i in 0..d0 {
            let row = &v[i * rest..(i + 1) * rest];
            let n = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
            for j in 0..rest {
                w[i * rest + j] = g[i] * row[j] / n;
            }
        }
        (w, shape)
    } else {
        (l.host(&format!("{p}.weight"))?, l.st.need(&format!("{p}.weight")).map_err(|e| Error(e.0))?.shape.clone())
    };
    let b = if l.has(&format!("{p}.bias")) { Some(l.float(&format!("{p}.bias"))?) } else { None };
    // a convolution [co, ci, k]; a transposed one [ci, co, k]
    let (ci, co) = if transposed { (shape[0] as usize, shape[1] as usize) } else { (shape[1] as usize, shape[0] as usize) };
    Ok(Conv { w: l.upload(f32_bytes(&w))?, b, ci, co, k: shape[2] as usize })
}

fn snake(l: &Loader, p: &str) -> Result<Snake> {
    let a = l.host(&format!("{p}.alpha"))?;
    let b = l.host(&format!("{p}.beta"))?;
    let alpha: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let inv: Vec<f32> = b.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect();
    Ok(Snake { c: alpha.len(), alpha: l.upload(f32_bytes(&alpha))?, inv_beta: l.upload(f32_bytes(&inv))? })
}

fn res(l: &Loader, p: &str, dil: usize) -> Result<Res> {
    Ok(Res { s1: snake(l, &format!("{p}.snake1"))?, c1: conv(l, &format!("{p}.conv1"), false)?, s2: snake(l, &format!("{p}.snake2"))?,
             c2: conv(l, &format!("{p}.conv2"), false)?, dil })
}

impl Vae {
    /// `strides`: the config's downsampling_ratios
    pub fn load(weights: &Path, strides: &[usize], gpu: &Arc<Gpu>, with_encoder: bool) -> Result<Vae> {
        let st = SafeTensors::open(weights).map_err(|e| Error(e.0))?;
        let l = Loader::new(&st, gpu);
        let n = strides.len();
        let mut d_blocks = Vec::new();
        for (i, s) in strides.iter().rev().enumerate() {
            let p = format!("decoder.block.{i}");
            d_blocks.push(DecBlock { s: snake(&l, &format!("{p}.snake1"))?, up: conv(&l, &format!("{p}.conv_t1"), true)?, stride: *s,
                                     res: [res(&l, &format!("{p}.res_unit1"), 1)?, res(&l, &format!("{p}.res_unit2"), 3)?, res(&l, &format!("{p}.res_unit3"), 9)?] });
        }
        let (mut e_conv1, mut e_snake, mut e_conv2, mut e_blocks) = (None, None, None, Vec::new());
        if with_encoder {
            e_conv1 = Some(conv(&l, "encoder.conv1", false)?);
            for (i, s) in strides.iter().enumerate() {
                let p = format!("encoder.block.{i}");
                e_blocks.push(EncBlock { res: [res(&l, &format!("{p}.res_unit1"), 1)?, res(&l, &format!("{p}.res_unit2"), 3)?, res(&l, &format!("{p}.res_unit3"), 9)?],
                                         s: snake(&l, &format!("{p}.snake1"))?, down: conv(&l, &format!("{p}.conv1"), false)?, stride: *s });
            }
            e_snake = Some(snake(&l, "encoder.snake1")?);
            e_conv2 = Some(conv(&l, "encoder.conv2", false)?);
        }
        let _ = n;
        // the silo unless NS_ACE_VAE_SILO=0 (the shared kernels: the A/B's other side)
        let silo = match std::env::var("NS_ACE_VAE_SILO").is_ok_and(|v| v.trim() == "0") {
            true => None,
            false => ffi::silo().map_err(|e| Error(e.0))?.filter(|k| {
                // SAFETY: a live GPU handle.
                unsafe { (k.supported)(gpu.raw()) == 1 }
            }),
        };
        Ok(Vae { d_conv1: conv(&l, "decoder.conv1", false)?, d_blocks, d_snake: snake(&l, "decoder.snake1")?, d_conv2: conv(&l, "decoder.conv2", false)?,
                 e_conv1, e_blocks, e_snake, e_conv2, bytes: l.bytes.get(), gpu: gpu.clone(), silo, peak: Default::default() })
    }

    /// whose convolutions these are: "silo" (oneDNN) or "shared" (the plain loops)
    pub fn convs(&self) -> &'static str {
        if self.silo.is_some() { "silo" } else { "shared" }
    }

    /// What the silo holds on the card now (bytes): its copies of the weights, its scratchpad (none without the silo)
    pub fn silo_held(&self) -> (u64, u64) {
        let Some(k) = self.silo else { return (0, 0) };
        let (mut w, mut s) = (0i64, 0i64);
        // SAFETY: a live GPU handle, out-pointers to locals.
        let _ = unsafe { (k.held)(self.gpu.raw(), &mut w, &mut s) };
        (w.max(0) as u64, s.max(0) as u64)
    }

    /// The end of a pass: the silo's scratchpad and its copies of the weights given back (the card holds the DiT's work
    /// next; the next pass makes them again, milliseconds); the most it held kept for `silo_peak`
    fn trim(&self) {
        if let Some(k) = self.silo {
            let (w, s) = self.silo_held();
            self.peak.0.fetch_max(w, std::sync::atomic::Ordering::Relaxed);
            self.peak.1.fetch_max(s, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: a live GPU handle.
            let _ = unsafe { (k.trim)(self.gpu.raw()) };
        }
    }

    /// The most the silo held at the end of a pass yet (bytes): its copies of the weights, its scratchpad
    pub fn silo_peak(&self) -> (u64, u64) {
        (self.peak.0.load(std::sync::atomic::Ordering::Relaxed), self.peak.1.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn has_encoder(&self) -> bool {
        self.e_conv1.is_some()
    }

    fn snake(ops: &Ops, s: &Snake, x: P, l: usize, out: M) -> Result<()> {
        timed(ops, &|| format!("snake c{}", s.c), || {
            // SAFETY: x and out hold s.c x l floats.
            ffi::check(unsafe { (ops.k.snake)(ops.gpu.raw(), x.cast(), s.c as i64, l as i64, s.alpha.fp(), s.inv_beta.fp(), out.cast()) }, "snake")
        })
    }

    /// x [c.ci, l] through a convolution (`transposed`: c is a transposed one, dil 1) into out: its length
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, ops: &Ops, c: &Conv, transposed: bool, x: P, l: usize, stride: usize, dil: usize, pad: usize, out: M) -> Result<usize> {
        let lo = if transposed { (l - 1) * stride - 2 * pad + c.k } else { (l + 2 * pad - dil * (c.k - 1) - 1) / stride + 1 };
        let bias = c.b.as_ref().map_or(none(), |b| b.ptr() as P);
        let what = || format!("{} k{} {}->{}{}", if transposed { "up" } else if stride > 1 { "down" } else { "conv" }, c.k, c.ci, c.co,
                              if dil > 1 { format!(" dil{dil}") } else { String::new() });
        timed(ops, &what, || match self.silo {
            Some(k) => ffi::check(
                // SAFETY: x holds c.ci x l floats, out c.co x lo, the weights and bias are c's.
                unsafe { (k.conv1d)(ops.gpu.raw(), transposed as i32, x.cast(), c.ci as i64, l as i64, c.w.fp(), c.co as i64, c.k as i64, bias.cast(),
                                    stride as i64, dil as i64, pad as i64, out.cast(), lo as i64) },
                "the VAE's convolution"),
            None if transposed => ops.nsd.conv_transpose1d(x, 1, c.ci, l, c.w.ptr(), c.co, c.k, bias, stride, pad, out, lo),
            None => ops.nsd.conv1d(x, 1, c.ci, l, c.w.ptr(), c.co, c.k, bias, stride, dil, pad, out, lo),
        })?;
        Ok(lo)
    }

    /// x [c.ci, l] through c, a transposed convolution (pad ceil(stride / 2)), into out: its length. tmp (`tmp_bytes`,
    /// free during the call) holds the silo's phases.
    #[allow(clippy::too_many_arguments)]
    fn up(&self, ops: &Ops, c: &Conv, x: P, l: usize, stride: usize, tmp: M, tmp_bytes: usize, out: M) -> Result<usize> {
        let Some(k) = self.silo else { return self.conv(ops, c, true, x, l, stride, 1, stride.div_ceil(2), out) };
        let pad = stride.div_ceil(2);
        let lo = (l - 1) * stride - 2 * pad + c.k;
        let bias = c.b.as_ref().map_or(none(), |b| b.ptr() as P);
        timed(ops, &|| format!("up k{} {}->{}", c.k, c.ci, c.co), || {
            ffi::check(
                // SAFETY: x holds c.ci x l floats, out c.co x lo, tmp tmp_bytes; the weights and bias are c's.
                unsafe { (k.conv_transpose1d)(ops.gpu.raw(), x.cast(), c.ci as i64, l as i64, c.w.fp(), c.co as i64, c.k as i64, bias.cast(),
                                              stride as i64, pad as i64, tmp.cast(), (tmp_bytes / 4) as i64, out.cast(), lo as i64) },
                "the VAE's transposed convolution")
        })?;
        Ok(lo)
    }

    /// a residual unit over x [c, l] in place (a, b: scratch of c x l)
    #[allow(clippy::too_many_arguments)]
    fn res(&self, ops: &Ops, r: &Res, x: M, l: usize, a: M, b: M) -> Result<()> {
        Self::snake(ops, &r.s1, x, l, a)?;
        self.conv(ops, &r.c1, false, a, l, 1, r.dil, 3 * r.dil, b)?;
        Self::snake(ops, &r.s2, b, l, b)?;
        self.conv(ops, &r.c2, false, b, l, 1, 1, 0, a)?;
        timed(ops, &|| format!("residual add c{}", r.c1.co), || {
            ops.nsd.gate_add(x, nextsycl_diffusion::kernels::Dt::F32, 1, r.c1.co * l, a, nextsycl_diffusion::kernels::Dt::F32, none(), none())
        })
    }

    /// Latents [frames, 64] (frames-major, as the DiT makes them) of one window to stereo [2, frames x 1920]
    fn decode_window(&self, ops: &Ops, s: &mut Scratch, z: &[f32]) -> Result<Vec<f32>> {
        let frames = z.len() / LATENT;
        let g = &ops.gpu;
        // the biggest activation: 128 channels at the audio rate; the input 2048 at 25 Hz (and a little more: the silo's
        // phases of the last upsampling, in the free buffer, are a few samples longer than its output)
        let big = frames * HOP * 128 * 4 + (1 << 20);
        let bufs = [s.get(g, "v0", big)?, s.get(g, "v1", big)?, s.get(g, "v2", big)?];
        // [frames, 64] -> [64, frames]
        let mut zt = vec![0f32; z.len()];
        for f in 0..frames {
            for c in 0..LATENT {
                zt[c * frames + f] = z[f * LATENT + c];
            }
        }
        let zin = timed(ops, &|| "upload".into(), || ops.put_f32(s, "vz", &zt))?;
        let (mut x, a, mut b) = (bufs[0], bufs[1], bufs[2]);
        let mut l = self.conv(ops, &self.d_conv1, false, zin, frames, 1, 1, 3, x)?;
        for blk in &self.d_blocks {
            Self::snake(ops, &blk.s, x, l, a)?;
            // x is free until the swap: the silo's scratch
            l = self.up(ops, &blk.up, a, l, blk.stride, x, big, b)?;
            std::mem::swap(&mut x, &mut b);
            for r in &blk.res {
                self.res(ops, r, x, l, a, b)?;
            }
        }
        Self::snake(ops, &self.d_snake, x, l, a)?;
        let lo = self.conv(ops, &self.d_conv2, false, a, l, 1, 1, 3, b)?;
        let _ = lo;
        let name = if b == bufs[0] { "v0" } else if b == bufs[1] { "v1" } else { "v2" };
        timed(ops, &|| "read back".into(), || read_f32(s.buf(name), 2 * l))
    }

    /// The reference's decode window for the memory the card has free (`_get_auto_decode_chunk_size`: 512 frames from
    /// 24 GB, 384 from 16, 256 from 12, else 128), what this scratch holds counted as free
    pub fn auto_chunk(ops: &Ops, s: &Scratch) -> usize {
        let free = ops.gpu.free_or_guess().map_or(0, |f| f + s.bytes() as u64) as f64 / (1u64 << 30) as f64;
        if free >= 24.0 { 512 } else if free >= 16.0 { 384 } else if free >= 12.0 { 256 } else { 128 }
    }

    /// The whole song's latents [frames, 64] to stereo [2, frames x 1920] (channel-major), in the reference's windows of
    /// `chunk` frames (the overlap of 64 halved until a core is left, as `_tiled_decode_inner`)
    pub fn decode(&self, ops: &Ops, s: &mut Scratch, z: &[f32], chunk: usize, progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let r = self.decode_windows(ops, s, z, chunk, progress);
        self.trim();
        r
    }

    fn decode_windows(&self, ops: &Ops, s: &mut Scratch, z: &[f32], chunk: usize, progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let frames = z.len() / LATENT;
        let mut overlap = OVERLAP;
        while chunk <= 2 * overlap && overlap > 4 {
            overlap /= 2;
        }
        if frames <= chunk {
            progress(0, 1)?;
            let w = self.decode_window(ops, s, z)?;
            progress(1, 1)?;
            return Ok(w);
        }
        let stride = chunk - 2 * overlap;
        let steps = frames.div_ceil(stride);
        let total = frames * HOP;
        let mut out = vec![0f32; 2 * total];
        let mut written = 0;
        for i in 0..steps {
            progress(i, steps)?;
            let core0 = i * stride;
            let core1 = (core0 + stride).min(frames);
            let w0 = core0.saturating_sub(overlap);
            let w1 = (core1 + overlap).min(frames);
            let wav = self.decode_window(ops, s, &z[w0 * LATENT..w1 * LATENT])?;
            let wl = wav.len() / 2;
            let (t0, t1) = ((core0 - w0) * HOP, wl - (w1 - core1) * HOP);
            for ch in 0..2 {
                let seg = &wav[ch * wl + t0..ch * wl + t1];
                out[ch * total + written..ch * total + written + seg.len()].copy_from_slice(seg);
            }
            written += t1 - t0;
        }
        progress(steps, steps)?;
        Ok(out)
    }

    /// One window of stereo [2, n] (channel-major, 48 kHz) through the encoder: (mean, scale) [n / 1920, 64] each,
    /// frames-major (the posterior's std is softplus(scale) + 1e-4)
    fn encode_window(&self, ops: &Ops, s: &mut Scratch, wav: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let (Some(c1), Some(sn), Some(c2)) = (&self.e_conv1, &self.e_snake, &self.e_conv2) else {
            return Err(Error("the VAE was loaded without its encoder".into()));
        };
        let n = wav.len() / 2;
        let g = &ops.gpu;
        let big = n * 128 * 4 + (1 << 20);
        let bufs = [s.get(g, "v0", big)?, s.get(g, "v1", big)?, s.get(g, "v2", big)?];
        let xin = timed(ops, &|| "upload".into(), || ops.put_f32(s, "vw", wav))?;
        let (mut x, a, mut b) = (bufs[0], bufs[1], bufs[2]);
        let mut l = self.conv(ops, c1, false, xin, n, 1, 1, 3, x)?;
        for blk in &self.e_blocks {
            for r in &blk.res {
                self.res(ops, r, x, l, a, b)?;
            }
            Self::snake(ops, &blk.s, x, l, a)?;
            l = self.conv(ops, &blk.down, false, a, l, blk.stride, 1, blk.stride.div_ceil(2), b)?;
            std::mem::swap(&mut x, &mut b);
        }
        Self::snake(ops, sn, x, l, a)?;
        let lo = self.conv(ops, c2, false, a, l, 1, 1, 1, b)?;
        let name = if b == bufs[0] { "v0" } else if b == bufs[1] { "v1" } else { "v2" };
        // [128, lo]: the mean, then the scale
        let both = timed(ops, &|| "read back".into(), || read_f32(s.buf(name), c2.co * lo))?;
        let (mut mean, mut scale) = (vec![0f32; lo * LATENT], vec![0f32; lo * LATENT]);
        for f in 0..lo {
            for c in 0..LATENT {
                mean[f * LATENT + c] = both[c * lo + f];
                scale[f * LATENT + c] = both[(LATENT + c) * lo + f];
            }
        }
        Ok((mean, scale))
    }

    /// The means alone of one window (the check's: the reference dumps its first window's)
    pub fn encode_mean(&self, ops: &Ops, s: &mut Scratch, wav: &[f32]) -> Result<Vec<f32>> {
        let r = self.encode_window(ops, s, wav);
        self.trim();
        Ok(r?.0)
    }

    /// Stereo [2, n] (channel-major, 48 kHz) to latents [n / 1920, 64] (frames-major), as the handler's tiled_encode:
    /// windows of 30 s (2 s of overlap each side discarded, cores of 26 s), each its posterior's sample (mean + std
    /// noise, from `noise`) - or its mean when `noise` is None
    pub fn encode(&self, ops: &Ops, s: &mut Scratch, wav: &[f32], noise: Option<&mut dyn FnMut() -> f32>,
                  progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let r = self.encode_windows(ops, s, wav, noise, progress);
        self.trim();
        r
    }

    fn encode_windows(&self, ops: &Ops, s: &mut Scratch, wav: &[f32], mut noise: Option<&mut dyn FnMut() -> f32>,
                      progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let n = wav.len() / 2;
        let mut draw = |mean: Vec<f32>, scale: Vec<f32>| -> Vec<f32> {
            match noise.as_mut() {
                Some(r) => mean.iter().zip(&scale).map(|(m, sc)| {
                    // softplus as torch's (x past 20: x itself)
                    let std = if *sc > 20.0 { *sc } else { sc.exp().ln_1p() } + 1e-4;
                    m + std * r()
                }).collect(),
                None => mean,
            }
        };
        let window = |a: usize, b: usize| -> Vec<f32> {
            let mut w = Vec::with_capacity(2 * (b - a));
            w.extend_from_slice(&wav[a..b]);
            w.extend_from_slice(&wav[n + a..n + b]);
            w
        };
        // the window: 30 s unless its buffers do not fit in what the card has free (beside what this scratch holds); the
        // silo's convolutions also want a scratchpad of about one activation and the encoder's weights in their layout
        let silo = self.silo.is_some();
        let need = |c: usize| 3 * c * 128 * 4 + (512 << 20) + if silo { c * 128 * 4 + (384 << 20) } else { 0 };
        let free = ops.gpu.memory().ok().and_then(|m| m.1).map_or(u64::MAX, |f| f + s.bytes() as u64);
        let chunk = if (free as usize) < need(ENC_CHUNK) { ENC_CHUNK_SMALL } else { ENC_CHUNK };
        if n <= chunk {
            progress(0, 1)?;
            let (m, sc) = self.encode_window(ops, s, wav)?;
            progress(1, 1)?;
            return Ok(draw(m, sc));
        }
        // _tiled_encode_offload_cpu: the latent rate from the first window, cores written in turn
        let stride = chunk - 2 * ENC_OVERLAP;
        let steps = n.div_ceil(stride);
        let mut out = Vec::new();
        let mut df = HOP as f64;
        for i in 0..steps {
            progress(i, steps)?;
            let core0 = i * stride;
            let core1 = (core0 + stride).min(n);
            let w0 = core0.saturating_sub(ENC_OVERLAP);
            let w1 = (core1 + ENC_OVERLAP).min(n);
            let (m, sc) = self.encode_window(ops, s, &window(w0, w1))?;
            let z = draw(m, sc);
            let len = z.len() / LATENT;
            if i == 0 {
                df = (w1 - w0) as f64 / len as f64;
            }
            let trim0 = ((core0 - w0) as f64 / df).round() as usize;
            let trim1 = ((w1 - core1) as f64 / df).round() as usize;
            let end = if trim1 > 0 { len.saturating_sub(trim1) } else { len };
            out.extend_from_slice(&z[trim0.min(end) * LATENT..end * LATENT]);
        }
        progress(steps, steps)?;
        // the reference's buffer holds round(n / df) frames: the cores past it are cut
        let total = (n as f64 / df).round() as usize;
        out.truncate(total.min(out.len() / LATENT) * LATENT);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_windows_cover_the_song_once() {
        // the reference's window arithmetic: cores of 128 frames, 64 more each side, trimmed
        let frames = 750usize;
        let stride = super::CHUNK - 2 * super::OVERLAP;
        let mut covered = 0;
        for i in 0..frames.div_ceil(stride) {
            let c0 = i * stride;
            let c1 = (c0 + stride).min(frames);
            assert!(c0.saturating_sub(super::OVERLAP) <= c0 && (c1 + super::OVERLAP).min(frames) >= c1);
            covered += c1 - c0;
        }
        assert_eq!(covered, frames);
    }
}
