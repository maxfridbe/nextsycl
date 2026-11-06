//! The VAE (the reference's models/video_vae_v3, s8_c16_t4_inflation_sd3.yaml): a causal 3-D autoencoder, 8x in
//! space and 4x in time, 16 latent channels. bfloat16 channels-last volumes [T, H, W, C] on the GPU.
//!
//! ```text
//!   encode  conv_in 3 -> 128; four down blocks (128, 256, 512, 512) of two resnets, the first three halving the
//!           picture, the second and third also the time; mid (resnet, attention, resnet); norm, SiLU, conv_out -> 32
//!           (mean | log variance: the mean is the latent)
//!   decode  conv_in 16 -> 512; mid; four up blocks (512, 512, 256, 128) of three resnets, the first three doubling
//!           the picture (a 1x1x1 conv to 4 or 8 times the channels and a shuffle), the first two also the time;
//!           norm, SiLU, conv_out -> RGB
//!   resnet  x + conv2(silu(norm2(conv1(silu(norm1(x)))))), 3x3x3 convolutions, a 1x1x1 shortcut when the width changes
//!   norms   GroupNorm 32, eps 1e-6, statistics per frame; the mid attention one head over each frame's pixels
//! ```
//!
//! Every convolution is causal in time: the frames before the first are the first repeated (twice for 3 taps), and a
//! clip goes through in slices - the encoder 1 + 4 frames, then 4 at a time; the decoder 2 latent frames, then 1 -
//! each convolution keeping the last frames of its input for the next slice (the reference's memory state), so a
//! slice's output is the whole clip's.

use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::{f16_to_f32, SafeTensors};

use crate::dit::bf16;
use crate::{ffi, prof};

const EPS: f32 = 1e-6;
const GROUPS: usize = 32;
/// the latent's scale (the config's scaling_factor)
pub const SCALE: f32 = 0.9152;
pub const LATENT: usize = 16;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A channels-last volume [t, h, w, c] of bfloat16 on the card
pub struct Vol {
    pub buf: std::mem::ManuallyDrop<DevBuf>,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

// A free does not wait for the queue (sycl::free): a large buffer let go while a queued kernel still reads it can be
// unmapped under it. A volume let go is kept until its card's next wait (`settle`: the slices end with one), then
// freed - a wait on every one drained the queue thousands of times a chunk (the B70 a quarter idle). Meanwhile a new
// volume of the same size takes a kept one: the queue is in order, so the kernels that still read it run before any
// that writes it again - no wait, no allocation (a slice's volumes come in a few sizes, over and over). `relieve`
// settles only when the kept ones pass NS_SVR_POOL_GIB (2), so the card waits a few times a slice, not at every resnet.
static LATER: [Mutex<Vec<DevBuf>>; 8] = [const { Mutex::new(Vec::new()) }; 8];

impl Drop for Vol {
    fn drop(&mut self) {
        // SAFETY: the buffer is taken once, here, and the volume is not used after.
        let b = unsafe { std::mem::ManuallyDrop::take(&mut self.buf) };
        LATER[b.gpu().index % 8].lock().unwrap().push(b);
    }
}

/// The card's queue drained, then the volumes let go since the last time freed
pub fn settle(nsd: &Nsd) -> Result<()> {
    nsd.wait()?;
    let gone = std::mem::take(&mut *LATER[nsd.gpu.index % 8].lock().unwrap());
    drop(gone);
    Ok(())
}

/// The volumes let go held past NS_SVR_POOL_GIB (default 2): settled (a wait, then freed); else nothing
pub fn relieve(nsd: &Nsd) -> Result<()> {
    let limit = std::env::var("NS_SVR_POOL_GIB").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(2.0);
    let held: usize = LATER[nsd.gpu.index % 8].lock().unwrap().iter().map(|b| b.len).sum();
    if held as f64 > limit * (1u64 << 30) as f64 {
        settle(nsd)?;
    }
    Ok(())
}

impl Vol {
    pub fn new(gpu: &Arc<Gpu>, t: usize, h: usize, w: usize, c: usize) -> Result<Vol> {
        let len = (t * h * w * c * 2).max(2);
        // a let-go volume of the size (see LATER), else a new one
        let kept = {
            let mut l = LATER[gpu.index % 8].lock().unwrap();
            l.iter().position(|b| b.len == len).map(|i| l.swap_remove(i))
        };
        let buf = match kept {
            Some(b) => b,
            None => DevBuf::new(gpu, len)?,
        };
        Ok(Vol { buf: std::mem::ManuallyDrop::new(buf), t, h, w, c })
    }
    fn frame_bytes(&self) -> usize {
        self.h * self.w * self.c * 2
    }
    fn px(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// A causal convolution: its weights, and the frames it keeps for the next slice
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: [usize; 3],
    s: [usize; 3],
    /// zeros above, below, left, right
    pad: [usize; 4],
    /// the first frame's copies in front when nothing is kept (twice the reference's temporal padding)
    front: usize,
    /// the frames kept: kernel depth less the stride
    keep: usize,
    /// the kept frames, where they are (a volume and a frame of it: held, not copied)
    memory: Mutex<Vec<(Arc<Vol>, usize)>>,
}

impl Conv {
    fn reset(&self) {
        self.memory.lock().unwrap().clear();
    }
}

struct Norm {
    w: DevBuf,
    b: DevBuf,
}

struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Att {
    norm: Norm,
    /// q, k, v, out: bfloat16 [c, c] and float32 biases
    w: [DevBuf; 4],
    b: [DevBuf; 4],
}

struct Mid {
    r0: Resnet,
    att: Att,
    r1: Resnet,
}

struct Down {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

struct Up {
    resnets: Vec<Resnet>,
    /// the 1x1x1 conv to (sx sy sz) times the channels, the shuffle's time factor, the conv after
    up: Option<(Conv, usize, Conv)>,
}

pub struct Vae {
    gpu: Arc<Gpu>,
    enc_in: Conv,
    downs: Vec<Down>,
    enc_mid: Mid,
    enc_norm: Norm,
    enc_out: Conv,
    dec_in: Conv,
    dec_mid: Mid,
    ups: Vec<Up>,
    dec_norm: Norm,
    dec_out: Conv,
}

impl Vae {
    pub fn load(file: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<Vae> {
        let t0 = std::time::Instant::now();
        let st = SafeTensors::open(file).map_err(ge)?;
        let vec = |name: &str| -> Result<DevBuf> { DevBuf::from_f32(gpu, &st.f32(name).map_err(ge)?) };
        let mat = |name: &str| -> Result<(DevBuf, Vec<usize>)> {
            let t = st.need(name).map_err(ge)?;
            let raw = st.read(t).map_err(ge)?;
            let bytes: Vec<u8> = match t.dtype.as_str() {
                "F16" => raw.chunks_exact(2).flat_map(|c| bf16(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).to_le_bytes()).collect(),
                "BF16" => raw,
                "F32" => raw.chunks_exact(4).flat_map(|c| bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()).collect(),
                d => return Err(Error(format!("{name}: {d} weights"))),
            };
            let b = DevBuf::new(gpu, bytes.len())?;
            b.write(0, &bytes)?;
            Ok((b, t.shape.iter().map(|&v| v as usize).collect()))
        };
        // a convolution: its kernel from the weight's shape; stride and causal front by role
        let conv = |p: &str, s: [usize; 3], pad: [usize; 4]| -> Result<Conv> {
            let (w, sh) = mat(&format!("{p}.weight"))?;
            if sh.len() != 5 {
                return Err(Error(format!("{p}.weight: {sh:?}, not a 3-D convolution")));
            }
            let k = [sh[2], sh[3], sh[4]];
            Ok(Conv { w, b: vec(&format!("{p}.bias"))?, ci: sh[1], co: sh[0], k, s, pad, front: k[0] - 1, keep: k[0] - s[0], memory: Mutex::new(Vec::new()) })
        };
        let same = |p: &str| -> Result<Conv> {
            let c = conv(p, [1, 1, 1], [0; 4])?;
            let ph = c.k[1] / 2;
            Ok(Conv { pad: [ph, ph, ph, ph], ..c })
        };
        let norm = |p: &str| -> Result<Norm> { Ok(Norm { w: vec(&format!("{p}.weight"))?, b: vec(&format!("{p}.bias"))? }) };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: norm(&format!("{p}.norm1"))?,
                conv1: same(&format!("{p}.conv1"))?,
                norm2: norm(&format!("{p}.norm2"))?,
                conv2: same(&format!("{p}.conv2"))?,
                shortcut: if st.tensor(&format!("{p}.conv_shortcut.weight")).is_some() { Some(same(&format!("{p}.conv_shortcut"))?) } else { None },
            })
        };
        let att = |p: &str| -> Result<Att> {
            let (mut w, mut b) = (Vec::new(), Vec::new());
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                w.push(mat(&format!("{p}.{n}.weight"))?.0);
                b.push(vec(&format!("{p}.{n}.bias"))?);
            }
            Ok(Att { norm: norm(&format!("{p}.group_norm"))?, w: w.try_into().map_err(|_| Error("attention".into()))?,
                     b: b.try_into().map_err(|_| Error("attention".into()))? })
        };
        let mid = |p: &str| -> Result<Mid> {
            Ok(Mid { r0: resnet(&format!("{p}.mid_block.resnets.0"))?, att: att(&format!("{p}.mid_block.attentions.0"))?, r1: resnet(&format!("{p}.mid_block.resnets.1"))? })
        };
        let count = |p: &str| (0..).take_while(|j| st.tensor(&format!("{p}.resnets.{j}.conv1.weight")).is_some()).count();
        let mut downs = Vec::new();
        for i in 0..4 {
            let p = format!("encoder.down_blocks.{i}");
            let n = count(&p);
            let dp = format!("{p}.downsamplers.0.conv");
            let down = if st.tensor(&format!("{dp}.weight")).is_some() {
                // stride 2 in each axis the kernel spans (time only when it has 3 taps); zeros right and bottom
                let c = conv(&dp, [1, 2, 2], [0, 1, 0, 1])?;
                let s = [if c.k[0] == 3 { 2 } else { 1 }, 2, 2];
                Some(Conv { s, keep: c.k[0] - s[0], ..c })
            } else {
                None
            };
            downs.push(Down { resnets: (0..n).map(|j| resnet(&format!("{p}.resnets.{j}"))).collect::<Result<_>>()?, down });
        }
        let mut ups = Vec::new();
        for i in 0..4 {
            let p = format!("decoder.up_blocks.{i}");
            let n = count(&p);
            let up = if st.tensor(&format!("{p}.upsamplers.0.upscale_conv.weight")).is_some() {
                let u = conv(&format!("{p}.upsamplers.0.upscale_conv"), [1, 1, 1], [0; 4])?;
                let r = u.co / u.ci;
                Some((u, r / 4, same(&format!("{p}.upsamplers.0.conv"))?))
            } else {
                None
            };
            ups.push(Up { resnets: (0..n).map(|j| resnet(&format!("{p}.resnets.{j}"))).collect::<Result<_>>()?, up });
        }
        let vae = Vae {
            gpu: gpu.clone(),
            enc_in: same("encoder.conv_in")?,
            downs,
            enc_mid: mid("encoder")?,
            enc_norm: norm("encoder.conv_norm_out")?,
            enc_out: same("encoder.conv_out")?,
            dec_in: same("decoder.conv_in")?,
            dec_mid: mid("decoder")?,
            ups,
            dec_norm: norm("decoder.conv_norm_out")?,
            dec_out: same("decoder.conv_out")?,
        };
        if vae.enc_in.ci != 3 || vae.enc_out.co != 2 * LATENT || vae.dec_in.ci != LATENT || vae.dec_out.co != 3 {
            return Err(Error(format!("{}: not SeedVR2's VAE (RGB to 2 x 16 latent channels and back)", file.display())));
        }
        log(format!("seedvr2 vae: {} down and {} up blocks on {} in {:.1} s", vae.downs.len(), vae.ups.len(), gpu.name, t0.elapsed().as_secs_f64()));
        Ok(vae)
    }

    /// A new clip: no convolution keeps frames from the last one
    pub fn reset(&self) {
        let r = |r: &Resnet| {
            r.conv1.reset();
            r.conv2.reset();
            if let Some(s) = &r.shortcut {
                s.reset();
            }
        };
        for c in [&self.enc_in, &self.enc_out, &self.dec_in, &self.dec_out] {
            c.reset();
        }
        for m in [&self.enc_mid, &self.dec_mid] {
            r(&m.r0);
            r(&m.r1);
        }
        for d in &self.downs {
            d.resnets.iter().for_each(r);
            if let Some(c) = &d.down {
                c.reset();
            }
        }
        for u in &self.ups {
            u.resnets.iter().for_each(r);
            if let Some((a, _, b)) = &u.up {
                a.reset();
                b.reset();
            }
        }
    }

    /// One causal convolution of x (its kept frames, or the first frame repeated, in front). The silo's taps read the
    /// frames where they are; the engine's own path copies them into one volume first
    fn conv(&self, nsd: &Nsd, c: &Conv, x: Arc<Vol>) -> Result<Vol> {
        self.conv_to(nsd, c, x, None)
    }

    /// The same, its result added to `onto` (a resnet's residual: the silo's taps sum onto it in the convolution; the
    /// engine's own path adds after)
    fn conv_to(&self, nsd: &Nsd, c: &Conv, x: Arc<Vol>, onto: Option<Vol>) -> Result<Vol> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        if x.c != c.ci {
            return Err(Error(format!("a convolution of {} channels given {}", c.ci, x.c)));
        }
        if c.k == [1, 1, 1] {
            // a pointwise convolution is a linear layer
            let out = Vol::new(gpu, x.t, x.h, x.w, c.co)?;
            let (m, kk, n) = (x.px() as f64, c.ci as f64, c.co as f64);
            // SAFETY: the silo's entry point, on this GPU's context.
            let silo = ffi::silo()?.filter(|sl| unsafe { (sl.supported)(gpu.raw()) } == 1);
            prof::timed(gpu, "vae conv 1x1x1 (linear)", 2.0 * m * kk * n, 2.0 * (m * kk + kk * n + m * n), || match silo {
                // SAFETY: x holds px x ci bfloat16, out px x co, the weight co x ci, the bias co floats.
                Some(sl) => ffi::check(unsafe { (sl.linear)(gpu.raw(), x.buf.ptr(), x.px() as i64, c.ci as i64, c.w.ptr(), c.co as i64, c.b.fp(), out.buf.ptr()) },
                                       "silo linear"),
                None => nsd.linear(x.buf.ptr(), Dt::BF16, x.px(), c.ci, c.w.ptr(), c.co, c.b.ptr(), out.buf.ptr(), Dt::BF16),
            })?;
            return Ok(out);
        }
        let mut mem = c.memory.lock().unwrap();
        let fb = x.frame_bytes();
        // the padded sequence: the kept frames (or the first frame repeated), then x's
        let mut seq: Vec<(Arc<Vol>, usize)> = if mem.is_empty() { (0..c.front).map(|_| (x.clone(), 0)).collect() } else { mem.clone() };
        seq.extend((0..x.t).map(|i| (x.clone(), i)));
        let st = seq.len();
        if st < c.k[0] || !(st - c.k[0]).is_multiple_of(c.s[0]) {
            return Err(Error(format!("a causal convolution of {} taps, stride {}, given {st} frames", c.k[0], c.s[0])));
        }
        let to = (st - c.k[0]) / c.s[0] + 1;
        let ho = (x.h + c.pad[0] + c.pad[1] - c.k[1]) / c.s[1] + 1;
        let wo = (x.w + c.pad[2] + c.pad[3] - c.k[2]) / c.s[2] + 1;
        let silo = ffi::silo()?.filter(|_| c.s == [1, 1, 1] && c.pad[0] == c.pad[1] && c.pad[2] == c.pad[3]);
        // SAFETY: the silo's entry point, on this GPU's context.
        let silo = silo.filter(|sl| unsafe { (sl.supported)(gpu.raw()) } == 1);
        let onto = onto.filter(|o| (o.t, o.h, o.w, o.c) == (to, ho, wo, c.co));
        let fused = onto.is_some() && silo.is_some();
        let (out, rest) = match onto {
            Some(o) if fused => (o, None),
            o => (Vol::new(gpu, to, ho, wo, c.co)?, o),
        };
        crate::note(gpu, &format!("a conv {}x{}x{} {} -> {}", st, x.h, x.w, c.ci, c.co));
        let taps = (c.k[0] * c.k[1] * c.k[2]) as f64;
        let flops = 2.0 * (to * ho * wo) as f64 * c.co as f64 * c.ci as f64 * taps;
        let what = format!("vae conv {}x{}x{} {}x{} {}->{}", c.k[0], c.k[1], c.k[2], x.w, x.h, c.ci, c.co);
        if let Some(sl) = silo {
            let f = if fused { sl.conv_taps_acc } else { sl.conv_taps };
            let ptrs: Vec<*const c_void> = seq.iter().map(|(v, i)| v.buf.ptr().cast::<u8>().wrapping_add(i * fb).cast_const().cast()).collect();
            // each tap reads its frames once and writes (or adds to) the output
            let bytes = 2.0 * c.k[0] as f64 * ((to * x.h * x.w * c.ci) as f64 + 2.0 * (to * ho * wo * c.co) as f64);
            prof::timed(gpu, &format!("{what} (silo taps)"), flops, bytes, || {
                // SAFETY: st frame pointers of h x w x ci bfloat16 each, out to x ho x wo x co, w the weight's [co, ci, k...].
                ffi::check(unsafe {
                    f(gpu.raw(), ptrs.as_ptr(), st as i64, x.h as i64, x.w as i64, c.ci as i64, c.w.ptr(), c.co as i64, c.k[0] as i64,
                                   c.k[1] as i64, c.k[2] as i64, c.pad[0] as i64, c.pad[2] as i64, c.b.fp(), out.buf.ptr())
                }, "silo conv taps")
            })?;
        } else {
            let src = Vol::new(gpu, st, x.h, x.w, x.c)?;
            prof::timed(gpu, "vae causal frames (copies)", 0.0, 2.0 * (st * fb) as f64, || {
                let front = st - x.t;
                for (i, (v, f)) in seq[..front].iter().enumerate() {
                    src.buf.copy_within(i * fb, &v.buf, f * fb, fb)?;
                }
                src.buf.copy_within(front * fb, &x.buf, 0, x.t * fb)
            })?;
            let bytes = 2.0 * ((src.px() * c.ci) as f64 + (to * ho * wo * c.co) as f64 + c.co as f64 * c.ci as f64 * taps);
            prof::timed(gpu, &what, flops, bytes, || {
                // SAFETY: src holds t x h x w x ci, out to x ho x wo x co bfloat16 values, w the weight's [co, ci, k...].
                ffi::check(unsafe {
                    (k.conv3d)(gpu.raw(), src.buf.ptr(), src.t as i64, src.h as i64, src.w as i64, c.ci as i64, c.w.ptr(), c.co as i64, c.k[0] as i64,
                               c.k[1] as i64, c.k[2] as i64, c.s[0] as i64, c.s[1] as i64, c.s[2] as i64, c.pad[0] as i64, c.pad[1] as i64, c.pad[2] as i64,
                               c.pad[3] as i64, c.b.fp(), out.buf.ptr())
                }, "conv3d")
            })?;
        }
        *mem = seq.split_off(st - c.keep);
        drop(mem);
        if let Some(r) = rest {
            // the residual not fused: added after
            prof::timed(gpu, "vae residual add", 0.0, 6.0 * (out.px() * out.c) as f64, || {
                // SAFETY: out and r hold the same number of bfloat16 values.
                ffi::check(unsafe { (k.add)(gpu.raw(), out.buf.ptr(), r.buf.ptr(), (out.px() * out.c) as i64) }, "residual add")
            })?;
        }
        Ok(out)
    }

    /// group norm (and SiLU) of x, a new volume
    fn norm(&self, x: &Vol, n: &Norm, silu: bool) -> Result<Vol> {
        let k = ffi::api()?;
        let out = Vol::new(&self.gpu, x.t, x.h, x.w, x.c)?;
        // two reads and a write
        // the silo's when it is there (channels in eights)
        // SAFETY: the silo's entry point, on this GPU's context.
        let f = match ffi::silo()?.filter(|sl| x.c.is_multiple_of(8) && unsafe { (sl.supported)(self.gpu.raw()) } == 1) {
            Some(sl) => sl.group_norm,
            None => k.group_norm,
        };
        prof::timed(&self.gpu, "vae group norm", 0.0, 6.0 * (x.px() * x.c) as f64, || {
            // SAFETY: t frames of h x w x c bfloat16 in and out.
            ffi::check(unsafe {
                f(self.gpu.raw(), x.buf.ptr(), x.t as i64, (x.h * x.w) as i64, x.c as i64, GROUPS as i64, n.w.fp(), n.b.fp(), EPS, silu as i32,
                               out.buf.ptr())
            }, "group norm")
        })?;
        Ok(out)
    }

    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: Vol) -> Result<Vol> {
        let a = self.norm(&x, &r.norm1, true)?;
        let a = self.conv(nsd, &r.conv1, Arc::new(a))?;
        let a = self.norm(&a, &r.norm2, true)?;
        let short = match &r.shortcut {
            Some(c) => self.conv(nsd, c, Arc::new(x))?,
            None => x,
        };
        // the second convolution summed onto the residual (x is no other's: conv1 read its norm, the shortcut keeps
        // nothing)
        let out = self.conv_to(nsd, &r.conv2, Arc::new(a), Some(short))?;
        relieve(nsd)?;
        Ok(out)
    }

    /// The mid attention, one frame at a time: x + out(attention(q, k, v of group_norm(x))), one head of every channel
    /// over the frame's pixels - two GEMMs and a softmax a block of query rows; v made transposed (its bias added after:
    /// each row of probabilities sums to 1)
    fn attention(&self, nsd: &Nsd, a: &Att, x: Vol) -> Result<Vol> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let (px, c) = (x.h * x.w, x.c);
        let n = self.norm(&x, &a.norm, false)?;
        let (q, kk, vt) = (DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, px * c * 2)?, DevBuf::new(gpu, c * px * 2)?);
        let rows_max = ((256usize << 20) / (px * 4)).clamp(1, px);
        let scores = DevBuf::f32(gpu, rows_max * px)?;
        let probs = DevBuf::new(gpu, rows_max * px * 2)?;
        crate::note(gpu, "the vae attention");
        let o = DevBuf::new(gpu, px * c * 2)?;
        let y = DevBuf::new(gpu, px * c * 2)?;
        let scale = 1.0 / (c as f32).sqrt();
        let at = |b: &DevBuf, e: usize| -> *mut c_void { b.ptr().cast::<u16>().wrapping_add(e).cast() };
        // q, k, v, out: 4 px c c each; scores and the weighted sum: 2 px px c each
        let (pf, cf) = (px as f64, c as f64);
        let flops = x.t as f64 * (8.0 * pf * cf * cf + 4.0 * pf * pf * cf);
        let bytes = x.t as f64 * (2.0 * 8.0 * pf * cf + 6.0 * pf * pf);
        prof::timed(gpu, "vae attention (a frame's pixels)", flops, bytes, || {
        for f in 0..x.t {
            let nf = at(&n.buf, f * px * c);
            nsd.linear(nf, Dt::BF16, px, c, a.w[0].ptr(), c, a.b[0].ptr(), q.ptr(), Dt::BF16)?;
            nsd.linear(nf, Dt::BF16, px, c, a.w[1].ptr(), c, a.b[1].ptr(), kk.ptr(), Dt::BF16)?;
            nsd.linear(a.w[2].ptr(), Dt::BF16, c, c, nf, px, none(), vt.ptr(), Dt::BF16)?;
            let mut r0 = 0;
            while r0 < px {
                let rows = rows_max.min(px - r0);
                nsd.linear(at(&q, r0 * c), Dt::BF16, rows, c, kk.ptr(), px, none(), scores.ptr(), Dt::F32)?;
                // SAFETY: scores holds rows x px floats, probs as many bfloat16.
                ffi::check(unsafe { (k.softmax_rows)(gpu.raw(), scores.fp(), rows as i64, px as i64, scale, probs.ptr()) }, "softmax")?;
                nsd.linear(probs.ptr(), Dt::BF16, rows, px, vt.ptr(), c, a.b[2].ptr(), at(&o, r0 * c), Dt::BF16)?;
                r0 += rows;
            }
            nsd.linear(o.ptr(), Dt::BF16, px, c, a.w[3].ptr(), c, a.b[3].ptr(), y.ptr(), Dt::BF16)?;
            // SAFETY: a frame of px x c bfloat16 in x and y.
            ffi::check(unsafe { (k.add)(gpu.raw(), at(&x.buf, f * px * c), y.ptr(), (px * c) as i64) }, "residual add")?;
        }
        Ok(())
        })?;
        // (a wait: q, k, v and the scores are plain buffers, freed as this returns)
        settle(nsd)?;
        Ok(x)
    }

    fn mid(&self, nsd: &Nsd, m: &Mid, x: Vol) -> Result<Vol> {
        let x = self.resnet(nsd, &m.r0, x)?;
        let x = self.attention(nsd, &m.att, x)?;
        self.resnet(nsd, &m.r1, x)
    }

    /// x through `f` `per` frames at a time, the outputs joined: the same as all at once (the convolutions keep their
    /// last frames from piece to piece, the norms and the attention are per frame), with a few frames' memory at the
    /// full resolution instead of a slice's
    fn pieces(&self, x: Vol, per: usize, f: &mut dyn FnMut(Vol) -> Result<Vol>) -> Result<Vol> {
        if x.t <= per {
            return f(x);
        }
        let fb = x.frame_bytes();
        let mut outs = Vec::new();
        for t0 in (0..x.t).step_by(per) {
            let n = per.min(x.t - t0);
            let p = Vol::new(&self.gpu, n, x.h, x.w, x.c)?;
            p.buf.copy_within(0, &x.buf, t0 * fb, n * fb)?;
            outs.push(f(p)?);
        }
        drop(x);
        let (t, o) = (outs.iter().map(|o| o.t).sum(), &outs[0]);
        let all = Vol::new(&self.gpu, t, o.h, o.w, o.c)?;
        let ob = o.frame_bytes();
        let mut at = 0;
        for o in &outs {
            all.buf.copy_within(at * ob, &o.buf, 0, o.t * ob)?;
            at += o.t;
        }
        Ok(all)
    }

    /// The frames a piece at full resolution: those whose widest volume (`channels` there) fits in about 1 GiB
    /// (NS_SVR_PIECE: that many, a check's)
    fn per(h: usize, w: usize, channels: usize) -> usize {
        if let Some(n) = std::env::var("NS_SVR_PIECE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0) {
            return n;
        }
        ((1usize << 30) / (h * w * channels * 2).max(1)).max(1)
    }

    /// One slice through the encoder: frames [t, H, W, 3] in [-1, 1] -> [t', H / 8, W / 8, 32] (mean | log variance).
    /// Its full-resolution part (conv_in and the first block, whose downsampler halves only the picture) a piece at a time
    pub fn encode_slice(&self, nsd: &Nsd, x: Vol) -> Result<Vol> {
        let (d0, rest) = self.downs.split_first().ok_or_else(|| Error("no down blocks".into()))?;
        let per = Self::per(x.h, x.w, self.enc_in.co);
        let mut x = self.pieces(x, per, &mut |p| {
            let mut x = self.conv(nsd, &self.enc_in, Arc::new(p))?;
            for r in &d0.resnets {
                x = self.resnet(nsd, r, x)?;
            }
            match &d0.down {
                Some(c) if c.k[0] == 1 => self.conv(nsd, c, Arc::new(x)),
                Some(_) => Err(Error("the first down block halves the time: its pieces would not line up".into())),
                None => Ok(x),
            }
        })?;
        for d in rest {
            for r in &d.resnets {
                x = self.resnet(nsd, r, x)?;
            }
            if let Some(c) = &d.down {
                x = self.conv(nsd, c, Arc::new(x))?;
            }
        }
        let x = self.mid(nsd, &self.enc_mid, x)?;
        let x = self.norm(&x, &self.enc_norm, true)?;
        let out = self.conv(nsd, &self.enc_out, Arc::new(x))?;
        settle(nsd)?;
        Ok(out)
    }

    /// One up block (its resnets, then its upsampler: the 1x1x1 conv, the shuffle, the conv)
    fn up(&self, nsd: &Nsd, u: &Up, mut x: Vol, first: bool) -> Result<Vol> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        for r in &u.resnets {
            x = self.resnet(nsd, r, x)?;
        }
        let Some((a, sz, c)) = &u.up else { return Ok(x) };
        let y = self.conv(nsd, a, Arc::new(x))?;
        let drop_second = first && *sz == 2;
        let s = Vol::new(gpu, y.t * sz - drop_second as usize, y.h * 2, y.w * 2, c.ci)?;
        prof::timed(gpu, "vae upsample shuffle", 0.0, 4.0 * (s.px() * s.c) as f64, || {
            // SAFETY: y holds t x h x w x (4 sz c) values, s the shuffle's output.
            ffi::check(unsafe {
                (k.shuffle_up)(gpu.raw(), y.buf.ptr(), y.t as i64, y.h as i64, y.w as i64, c.ci as i64, 2, 2, *sz as i64, drop_second as i32, s.buf.ptr())
            }, "upsample")
        })?;
        drop(y);
        self.conv(nsd, c, Arc::new(s))
    }

    /// One slice through the decoder: latents [t, h, w, 16] (already divided by the scale) -> frames [t', 8 h, 8 w, 3];
    /// first: the clip's first slice (the upsamplers drop the first latent frame's duplicate). Everything after the
    /// last upsampler that doubles the time (the half and full resolution) a piece at a time
    pub fn decode_slice(&self, nsd: &Nsd, z: Vol, first: bool) -> Result<Vol> {
        let x = self.conv(nsd, &self.dec_in, Arc::new(z))?;
        let mut x = self.mid(nsd, &self.dec_mid, x)?;
        let split = self.ups.iter().rposition(|u| u.up.as_ref().is_some_and(|(_, sz, _)| *sz == 2)).map_or(0, |i| i + 1);
        for u in &self.ups[..split] {
            x = self.up(nsd, u, x, first)?;
        }
        // the widest volume after: the first spatial upsampler's input times 4 at twice the side
        let per = Self::per(x.h * 2, x.w * 2, self.ups.get(split).map_or(x.c, |u| u.up.as_ref().map_or(x.c, |(a, _, _)| a.co / 4)));
        let out = self.pieces(x, per, &mut |p| {
            let mut x = p;
            for u in &self.ups[split..] {
                x = self.up(nsd, u, x, first)?;
            }
            let x = self.norm(&x, &self.dec_norm, true)?;
            self.conv(nsd, &self.dec_out, Arc::new(x))
        })?;
        settle(nsd)?;
        Ok(out)
    }

    /// A clip of 1 + 4 k frames [T, H, W, 3] (H, W multiples of 8) to its latent mean float32 [1 + k, H / 8, W / 8, 16]
    /// times the scale (the DiT's condition): in slices of 1 + 4 frames, then 4
    pub fn encode(&self, nsd: &Nsd, x: &Vol) -> Result<Vec<f32>> {
        self.reset();
        let fb = x.frame_bytes();
        let mut out = Vec::new();
        let mut f0 = 0;
        while f0 < x.t {
            let n = if f0 == 0 { 5.min(x.t) } else { 4.min(x.t - f0) };
            let s = Vol::new(&self.gpu, n, x.h, x.w, x.c)?;
            s.buf.copy_within(0, &x.buf, f0 * fb, n * fb)?;
            let h = self.encode_slice(nsd, s)?;
            let mut raw = vec![0u8; h.px() * h.c * 2];
            h.buf.read(0, &mut raw)?;
            for px in raw.chunks_exact(h.c * 2) {
                out.extend(px[..LATENT * 2].chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) * SCALE));
            }
            f0 += n;
        }
        self.reset();
        Ok(out)
    }

    /// The latent float32 [t, h, w, 16] (the DiT's, times the scale) to frames [1 + 4 (t - 1), 8 h, 8 w, 3] in about
    /// [-1, 1]: in slices of 2 latent frames, then 1; `frames` gets each slice's frames as they are made
    pub fn decode(&self, nsd: &Nsd, z: &[f32], t: usize, h: usize, w: usize, frames: &mut dyn FnMut(Vol) -> Result<()>) -> Result<()> {
        self.reset();
        let per = h * w * LATENT;
        let mut i = 0;
        while i < t {
            let n = if i == 0 { 2.min(t) } else { 1 };
            let bits: Vec<u8> = z[i * per..(i + n) * per].iter().flat_map(|v| bf16(v / SCALE).to_le_bytes()).collect();
            let s = Vol::new(&self.gpu, n, h, w, LATENT)?;
            s.buf.write(0, &bits)?;
            frames(self.decode_slice(nsd, s, i == 0)?)?;
            i += n;
        }
        self.reset();
        Ok(())
    }
}
