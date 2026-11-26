//! Open-H3-Regenerate-2K's fast video decoder (`vae_fast/fast_h3_vae_v2_24l.safetensors`; the release's
//! `infer/ref2va_ic/fast_vae/decoder.py`, from NVIDIA's StreamRender-H3 work): the H3 decoder's transformer pruned to 24
//! of its 36 layers and distilled to decode a whole clip in one piece - no spatial tiles, no temporal chunks cross-faded:
//!
//! ```text
//!   time   native 5-latent blocks (17 frames); attention inside temporal windows of 5 latent frames, odd layers shifted
//!          by 2 (Video-Swin style), so neighbouring blocks talk inside the network
//!   space  16 x 16-token windows (256 px), odd layers shifted by 8
//!   global every block owns 5 tokens (the 4 registers + a zero token): they attend to every token of their block, and
//!          every window attends to the global tokens of the blocks it overlaps
//!   rope   window-local ids on all three axes (any length decodes the same way)
//!   out    per token 3 x 4 x 16 x 16 pixels; the first latent of every block keeps the last of its 4 frames
//! ```
//!
//! On the card the clip's residual stream (float32, one row a token) stays whole and each layer runs a temporal window at
//! a time (at 2560x1440: 72,000 tokens): its norm, qkv, the window attention (the windows' rows gathered into variable-
//! length sequences - ARK's kernel on sycl-tla, the silo's `varlen`), the projection and the MLP, written back in place.
//! A window's keys are its own tokens and the global tokens of its blocks, both made before the window changes anything,
//! so this is the release's layer exactly. The global tokens' attention over their whole block takes the block's keys a
//! temporal window at a time into a running softmax (the silo's `glob`), and the global rows are finished when every
//! window has run. bfloat16 matrices and activations, the residual in float32, the head in float32 (the release's
//! reference engine). Its kernels beyond the shared ones are the H3 video engine's silo (kernels/video/h3/silo), bound by
//! the engine and handed over with `install`; without them the job keeps the full H3 decoder.
//!
//! The layers and the pixel head are `Sol`, shared with pi-H3's DWD head (dwd.rs): the last 8 layers of the same
//! decoder fed from the denoiser's last hidden states instead of the latents, in bfloat16 or int8 ConvRot.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::device::{Device, Tensor};
use crate::dit::rotations;
use crate::dtype::DType;
use crate::ops::{self, Int8Linear, Linear, Rows};
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

const HEADS: usize = 32;
const HEAD_DIM: usize = 64;
const DIM: usize = HEADS * HEAD_DIM;
const ROT_DIM: usize = 48;
const EPS: f32 = 1e-5;
const LATENT_C: usize = 24;
const PATCH: usize = 16;
const PATCH_T: usize = 4;
const OUT_C: usize = 3;
/// the native block (latent frames), the window (latent pixels), the odd layers' shifts, frames the first latent of a
/// block drops
const BLOCK: usize = 5;
const WINDOW: usize = 16;
const T_SHIFT: usize = 2;
const TRIM: usize = 3;
/// global tokens a block: 4 registers + 1
const GLOBALS: usize = 5;
/// MLP rows at a time (its two widest buffers)
const MLP_ROWS: usize = 16384;

/// The silo's functions (kernels/video/h3/silo/silo.h), bound by the H3 video engine (video/h3/src/silo.rs)
#[allow(clippy::type_complexity)]
pub struct Kernels {
    pub error: unsafe extern "C" fn() -> *const c_char,
    pub supported: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub rows: unsafe extern "C" fn(*mut c_void, *const c_void, i64, *const i32, *mut c_void, i64, *const i32, i64, i64) -> c_int,
    pub varlen: unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void, *mut c_void, i64, *const i32, *const i32, *const i32, i64, i64,
                                     i64, i64, i64) -> c_int,
    pub glob_init: unsafe extern "C" fn(*mut c_void, *mut f32, i64) -> c_int,
    pub glob: unsafe extern "C" fn(*mut c_void, *const c_void, i64, *const c_void, *const c_void, i64, i64, i64, i64, *mut f32, *mut f32, i64) -> c_int,
    pub glob_out: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, *mut c_void, i64) -> c_int,
}

static KERNELS: OnceLock<Kernels> = OnceLock::new();

/// The engine's silo functions, once (the first install stays)
pub fn install(k: Kernels) {
    let _ = KERNELS.set(k);
}

/// The silo's functions, when the engine found its silo
pub fn kernels() -> Option<&'static Kernels> {
    KERNELS.get()
}

impl Kernels {
    pub(crate) fn check(&self, rc: c_int, what: &str) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: the silo's NUL-terminated thread-local message.
        Err(Error(format!("{what}: {}", unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy())))
    }
}

/// A linear of a layer: bfloat16 (the fast decoder's float32 file, a DWD adapter's float16 one) or int8 ConvRot (a DWD
/// adapter's ComfyUI int8 file)
enum Lin {
    F(Linear),
    I8(Int8Linear),
}

impl Lin {
    fn forward(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        match self {
            Lin::F(l) => l.forward(x, out),
            Lin::I8(l) => l.forward(x, out),
        }
    }
}

struct Layer {
    n1: Tensor,
    n2: Tensor,
    s1: Tensor,
    s2: Tensor,
    qkv: Lin,
    out: Lin,
    w1: Lin,
    w2: Lin,
}

/// The decoder's layers and its pixel head, on a context on nextsycl-core's queue of one GPU (the silo's kernels run
/// there): the residual stream of a clip's t x h x w tokens in, frames out. The fast decoder runs all 24; pi-H3's DWD head
/// the last 8 with its own weights and a one-frame head for the first latent of every block.
pub(crate) struct Sol {
    /// the GPU whose queue the context and the silo's kernels share (held as long as the context)
    gpu: Arc<nextsycl_core::Gpu>,
    pub(crate) dev: Arc<Device>,
    pub(crate) k: &'static Kernels,
    layers: Vec<Layer>,
    /// the global tokens' start (registers + a zero token), bfloat16-rounded, float32 [GLOBALS, DIM]
    g0: Vec<f32>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    /// [3 x 4 x 16 x 16, DIM]: a latent frame's 4 frames
    proj_out: Linear,
    /// [3 x 1 x 16 x 16, DIM]: the first latent frame of a block's one frame (the DWD head); without it the first
    /// latent's frame is the last of `proj_out`'s 4 (the decoder drops 3)
    first_out: Option<Linear>,
    ones: Tensor,
    inv_freq: Vec<f32>,
}

pub struct FastDecoder {
    sol: Sol,
    /// de-normalization, the 1x1x1 convolution and the embedding folded into one [DIM, 24] float32 map
    embed: Linear,
    pub load_seconds: f64,
    pub load_bytes: u64,
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

fn i32_tensor(dev: &Arc<Device>, v: &[i32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::I32, &[v.len().max(1)], &if v.is_empty() { vec![0u8; 4] } else { v.iter().flat_map(|f| f.to_le_bytes()).collect() })
}

/// float32 values -> bfloat16 bytes, `threads` at a time
fn bf16_bytes(v: &[f32], threads: usize) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 2];
    let per = v.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (a, b) in v.chunks(per).zip(out.chunks_mut(per * 2)) {
            s.spawn(move || {
                for (f, o) in a.iter().zip(b.chunks_exact_mut(2)) {
                    o.copy_from_slice(&crate::dtype::f32_to_bf16(*f).to_le_bytes());
                }
            });
        }
    });
    out
}

/// A pointer `elems` 16-bit elements into a tensor
fn at(t: &Tensor, elems: usize) -> *mut c_void {
    // SAFETY: callers keep `elems` inside the tensor (the shapes are the decoder's own).
    unsafe { t.buf.ptr().cast::<u16>().add(elems).cast() }
}

/// One temporal window of a layer's partition: its frames, the native blocks it overlaps, and its spatial windows as
/// variable-length sequences
struct Chunk {
    t0: usize,
    t1: usize,
    blocks: Vec<usize>,
    /// token i of the chunk (t, y, x order from t0) -> its row in the window-ordered queries / in the key pool
    qpos: Tensor,
    kpos: Tensor,
    /// the global rows copied into the key pool: from row gsrc[i] of the global qkv to row gdst[i]
    gsrc: Tensor,
    gdst: Tensor,
    globals: usize,
    cu_q: Tensor,
    cu_k: Tensor,
    windows: usize,
    max_q: usize,
    max_k: usize,
    total_k: usize,
}

/// The temporal windows of a layer with shifts (s, st) over t x h x w tokens (decoder.py `_geometry`), host side
struct ChunkHost {
    t0: usize,
    t1: usize,
    blocks: Vec<usize>,
    qpos: Vec<i32>,
    kpos: Vec<i32>,
    gsrc: Vec<i32>,
    gdst: Vec<i32>,
    cu_q: Vec<i32>,
    cu_k: Vec<i32>,
}

fn geometry(t: usize, h: usize, w: usize, s: usize, st: usize) -> Vec<ChunkHost> {
    let (tp, hp, wp) = ((t + st).div_ceil(BLOCK) * BLOCK, (h + s).div_ceil(WINDOW) * WINDOW, (w + s).div_ceil(WINDOW) * WINDOW);
    let span = |i: usize, size: usize, shift: usize, len: usize| ((i * size).saturating_sub(shift), ((i + 1) * size - shift).min(len));
    let hw = h * w;
    let mut out = Vec::new();
    for j in 0..tp / BLOCK {
        let (t0, t1) = span(j, BLOCK, st, t);
        if t1 <= t0 {
            continue;
        }
        let mut blocks = vec![t0 / BLOCK];
        if (t1 - 1) / BLOCK != t0 / BLOCK {
            blocks.push((t1 - 1) / BLOCK);
        }
        let n = (t1 - t0) * hw;
        let (mut qpos, mut kpos) = (vec![0i32; n], vec![0i32; n]);
        let (mut cu_q, mut cu_k, mut gsrc, mut gdst) = (vec![0i32], vec![0i32], Vec::new(), Vec::new());
        for i in 0..hp / WINDOW {
            let (y0, y1) = span(i, WINDOW, s, h);
            if y1 <= y0 {
                continue;
            }
            for k in 0..wp / WINDOW {
                let (x0, x1) = span(k, WINDOW, s, w);
                if x1 <= x0 {
                    continue;
                }
                let (bq, bk) = (*cu_q.last().unwrap(), *cu_k.last().unwrap());
                let mut c = 0i32;
                for tt in t0..t1 {
                    for yy in y0..y1 {
                        for xx in x0..x1 {
                            let tok = (tt - t0) * hw + yy * w + xx;
                            qpos[tok] = bq + c;
                            kpos[tok] = bk + c;
                            c += 1;
                        }
                    }
                }
                for (jj, b) in blocks.iter().enumerate() {
                    for sl in 0..GLOBALS {
                        gsrc.push((b * GLOBALS + sl) as i32);
                        gdst.push(bk + c + (jj * GLOBALS + sl) as i32);
                    }
                }
                cu_q.push(bq + c);
                cu_k.push(bk + c + (blocks.len() * GLOBALS) as i32);
            }
        }
        out.push(ChunkHost { t0, t1, blocks, qpos, kpos, gsrc, gdst, cu_q, cu_k });
    }
    out
}

/// (cos, sin) for one temporal window's worth of window-local positions: [BLOCK * h * w, ROT_DIM / 2, 2] - local time
/// lt, row, column; a chunk starting at local time lt0 takes rows from lt0 * h * w
fn rope_table(h: usize, w: usize, s: usize, inv_freq: &[f32]) -> Vec<f32> {
    let c = |i: usize, n: usize| (2.0 * ((i as f64 + 0.5) / n as f64) - 1.0) * std::f64::consts::TAU;
    let mut pos = Vec::with_capacity(BLOCK * h * w * 3);
    for lt in 0..BLOCK {
        for y in 0..h {
            for x in 0..w {
                pos.extend_from_slice(&[c(lt, BLOCK), c((y + s) % WINDOW, WINDOW), c((x + s) % WINDOW, WINDOW)]);
            }
        }
    }
    rotations(&pos, inv_freq)
}

/// What `Sol::open` gives: the GPU (its queue held), a context on that queue, the silo's kernels
pub(crate) type Opened = (Arc<nextsycl_core::Gpu>, Arc<Device>, &'static Kernels);

/// [h0: q k v][h1: q k v] ... -> [q of every head][k ...][v ...] (as the full decoder loads it): the source row of each
/// output row
fn qkv_rows() -> Vec<usize> {
    (0..3).flat_map(|part| (0..HEADS).map(move |h| h * 3 + part)).flat_map(|src| src * HEAD_DIM..(src + 1) * HEAD_DIM).collect()
}

/// The rows of a row-major [rows, width] buffer of `size`-byte elements in `order`
fn rows_in(v: &[u8], size: usize, width: usize, order: &[usize]) -> Vec<u8> {
    let row = width * size;
    order.iter().flat_map(|r| v[r * row..(r + 1) * row].iter().copied()).collect()
}

impl Sol {
    /// A context on nextsycl-core's queue of GPU `index` and the silo's kernels: None without the silo (or when its
    /// kernels do not run on that GPU)
    pub(crate) fn open(index: usize) -> Result<Option<Opened>> {
        let Some(k) = kernels() else { return Ok(None) };
        let gpu = nextsycl_core::Gpu::open(index).map_err(|e| Error(e.0))?;
        // SAFETY: the plain check of a live queue.
        if unsafe { (k.supported)(gpu.queue()) } != 1 {
            return Ok(None);
        }
        // SAFETY: the GPU's live queue, held beside the context (`gpu` lives as long as the Sol that keeps it).
        let dev = unsafe { Device::on_queue(gpu.queue(), index) }?;
        Ok(Some((gpu, dev, k)))
    }

    /// The layers under `prefix` (`blocks.N.*`, or a DWD adapter's `dwd_pixel_refiner.blocks.N.*`), the final norm, the
    /// projection(s) and the registers, from `ck`: float matrices as bfloat16, int8 ConvRot ones as they are. `bytes`
    /// counts what was read.
    pub(crate) fn load(gpu: Arc<nextsycl_core::Gpu>, dev: Arc<Device>, k: &'static Kernels, ck: &Checkpoint, prefix: &str, threads: usize, bytes: &mut u64)
                       -> Result<Sol> {
        let counted = std::cell::Cell::new(0u64);
        let read = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let e = ck.get(name)?;
            counted.set(counted.get() + e.bytes as u64);
            Ok((e.shape.clone(), crate::dtype::bytes_to_f32(&ck.read(name)?, e.dtype)?))
        };
        let blocks = format!("{prefix}blocks.");
        let n = ck.entries.keys().filter_map(|k| k.strip_prefix(blocks.as_str())?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        if n == 0 {
            return Err(Error(format!("{}: no decoder layers ({blocks}N.*)", ck.path.display())));
        }
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("{blocks}{i}");
            let lin = |l: &str, regroup: bool| -> Result<Lin> {
                let name = format!("{p}.{l}");
                let (_, mut b) = read(&format!("{name}.bias"))?;
                let order = regroup.then(qkv_rows);
                if let Some(o) = &order {
                    b = o.iter().map(|r| b[*r]).collect();
                }
                let bias = Some(f32_tensor(&dev, &[b.len()], &b)?);
                if let Some(q) = ck.quant(&name)? {
                    // int8 [N, K] with a scale a row: the rows regrouped on the host, the bytes as they are
                    let (we, se) = (ck.get(&format!("{name}.weight"))?, ck.get(&format!("{name}.weight_scale"))?);
                    if we.dtype != DType::I8 || se.dtype != DType::F32 {
                        return Err(Error(format!("{name}: int8 weights with float32 scales expected")));
                    }
                    let (shape, kk) = (we.shape.clone(), we.shape[1]);
                    let (mut w, mut s) = (ck.read(&format!("{name}.weight"))?, ck.read(&format!("{name}.weight_scale"))?);
                    counted.set(counted.get() + (w.len() + s.len()) as u64);
                    if let Some(o) = &order {
                        w = rows_in(&w, 1, kk, o);
                        s = rows_in(&s, 4, 1, o);
                    }
                    return Ok(Lin::I8(Int8Linear { weight: Tensor::from_bytes(&dev, DType::I8, &shape, &w)?, scale: Tensor::from_bytes(&dev, DType::F32, &[shape[0]], &s)?,
                                                   bias, group: q.convrot.then_some(q.group) }));
                }
                let (shape, mut w) = read(&format!("{name}.weight"))?;
                if let Some(o) = &order {
                    let kk = shape[1];
                    w = o.iter().flat_map(|r| w[r * kk..(r + 1) * kk].iter().copied()).collect();
                }
                Ok(Lin::F(Linear { weight: Tensor::from_bytes(&dev, DType::BF16, &shape, &bf16_bytes(&w, threads))?, bias }))
            };
            let qkv = lin("attn.to_qkv", true)?;
            let out = lin("attn.to_out", false)?;
            let w1 = lin("ff.w1", false)?;
            let w2 = lin("ff.w2", false)?;
            let vec1 = |k: &str| -> Result<Tensor> {
                let (_, v) = read(&format!("{p}.{k}"))?;
                f32_tensor(&dev, &[1, v.len()], &v)
            };
            layers.push(Layer { n1: vec1("norm1.weight")?, n2: vec1("norm2.weight")?, s1: vec1("scale1")?, s2: vec1("scale2")?, qkv, out, w1, w2 });
        }
        let (_, reg) = read(&format!("{prefix}register_tokens"))?;
        let mut g0: Vec<f32> = reg.iter().map(|v| crate::dtype::bf16_to_f32(crate::dtype::f32_to_bf16(*v))).collect();
        g0.resize(GLOBALS * DIM, 0.0);
        let (_, nw) = read(&format!("{prefix}norm_out.weight"))?;
        let (_, nb) = read(&format!("{prefix}norm_out.bias"))?;
        let lin32 = |name: &str| -> Result<Linear> {
            let (ps, pw) = read(&format!("{prefix}{name}.weight"))?;
            let (_, pb) = read(&format!("{prefix}{name}.bias"))?;
            Ok(Linear { weight: f32_tensor(&dev, &ps, &pw)?, bias: Some(f32_tensor(&dev, &[pb.len()], &pb)?) })
        };
        let proj_out = lin32("proj_out")?;
        let first_out = match ck.entries.contains_key(&format!("{prefix}first_frame_proj_out.weight")) {
            true => Some(lin32("first_frame_proj_out")?),
            false => None,
        };
        let per = OUT_C * PATCH * PATCH;
        if proj_out.outputs() != per * PATCH_T || first_out.as_ref().is_some_and(|f| f.outputs() != per) {
            return Err(Error(format!("{}: a pixel head of {} (first frame {:?}) outputs, {} / {per} expected", ck.path.display(), proj_out.outputs(),
                                     first_out.as_ref().map(|f| f.outputs()), per * PATCH_T)));
        }
        *bytes += counted.get();
        let inv_freq: Vec<f32> = (0..ROT_DIM / 6).map(|i| 1.0 / 100f32.powf(i as f32 * 6.0 / ROT_DIM as f32)).collect();
        Ok(Sol {
            norm_out_w: f32_tensor(&dev, &[DIM], &nw)?,
            norm_out_b: f32_tensor(&dev, &[DIM], &nb)?,
            proj_out,
            first_out,
            ones: f32_tensor(&dev, &[HEAD_DIM], &[1.0; HEAD_DIM])?,
            g0,
            layers,
            inv_freq,
            k,
            dev,
            gpu,
        })
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The GPU it runs on (its index)
    pub(crate) fn gpu(&self) -> usize {
        self.gpu.index
    }

    /// nextsycl-core's queue of that GPU, the silo's kernels' first argument
    pub(crate) fn queue(&self) -> *mut c_void {
        self.gpu.queue()
    }
}

impl FastDecoder {
    /// The fast decoder's weights (float32 in the release; bfloat16 on the card) on GPU `index`, in a context on
    /// nextsycl-core's queue of that GPU (the silo's kernels run there); the latent statistics from the H3 VAE's
    /// checkpoint `vae` (the release takes them from its config: the same numbers). None when the engine has no silo.
    pub fn load(index: usize, path: &Path, vae: &Path, threads: usize) -> Result<Option<FastDecoder>> {
        let t0 = Instant::now();
        let Some((gpu, dev, k)) = Sol::open(index)? else { return Ok(None) };
        let ck = Checkpoint::open(path)?;
        let mut bytes = 0u64;
        let sol = Sol::load(gpu, dev.clone(), k, &ck, "", threads, &mut bytes)?;
        let mut read = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let e = ck.get(name)?;
            bytes += e.bytes as u64;
            Ok((e.shape.clone(), crate::dtype::bytes_to_f32(&ck.read(name)?, e.dtype)?))
        };
        // e = We (Wc (z std + mean) + bc) + be = A z + c
        let (_, wc) = read("post_quant_conv.weight")?;
        let (_, bc) = read("post_quant_conv.bias")?;
        let (_, we) = read("x_embedder.weight")?;
        let (_, be) = read("x_embedder.bias")?;
        let stats = Checkpoint::open(vae)?;
        let (mean, std) = (crate::dit::host(&stats, "latents_mean")?, crate::dit::host(&stats, "latents_std")?);
        if mean.len() != LATENT_C || std.len() != LATENT_C {
            return Err(Error(format!("{}: latents_mean / latents_std of {LATENT_C} channels expected", vae.display())));
        }
        let mut a = vec![0f32; DIM * LATENT_C];
        let mut c = vec![0f32; DIM];
        let inner: Vec<f64> = (0..LATENT_C).map(|o| bc[o] as f64 + (0..LATENT_C).map(|i| wc[o * LATENT_C + i] as f64 * mean[i] as f64).sum::<f64>()).collect();
        for d in 0..DIM {
            for i in 0..LATENT_C {
                a[d * LATENT_C + i] = ((0..LATENT_C).map(|o| we[d * LATENT_C + o] as f64 * wc[o * LATENT_C + i] as f64).sum::<f64>() * std[i] as f64) as f32;
            }
            c[d] = (be[d] as f64 + (0..LATENT_C).map(|o| we[d * LATENT_C + o] as f64 * inner[o]).sum::<f64>()) as f32;
        }
        let embed = Linear { weight: f32_tensor(&dev, &[DIM, LATENT_C], &a)?, bias: Some(f32_tensor(&dev, &[DIM], &c)?) };
        dev.wait()?;
        Ok(Some(FastDecoder { sol, embed, load_seconds: t0.elapsed().as_secs_f64(), load_bytes: bytes }))
    }

    pub fn layers(&self) -> usize {
        self.sol.layers()
    }

    /// The GPU it runs on (its index)
    pub fn gpu(&self) -> usize {
        self.sol.gpu()
    }

    /// The bytes on the card a decode of t x h x w latents needs beyond the weights (the residual stream, the window's
    /// buffers): for the job's room check
    pub fn need_bytes(t: usize, h: usize, w: usize) -> usize {
        let (n, nmax) = (t * h * w, BLOCK.min(t) * h * w);
        let wins = (h + WINDOW).div_ceil(WINDOW) * (w + WINDOW).div_ceil(WINDOW);
        n * DIM * 4 + nmax * DIM * 2 * 9 + 2 * (nmax + wins * 2 * GLOBALS) * DIM * 2 + MLP_ROWS * 4 * DIM * 2 * 3 + h * w * OUT_C * PATCH_T * PATCH * PATCH * 4
            + (1usize << 30)
    }

    /// Normalized latents [24, t, h, w] (t = 0 a block's first) -> pixels [3, frames, 16 h, 16 w] in [0, 1] and the frame
    /// count (4 t less 3 a block). `tick` between temporal windows (cancel checks).
    pub fn decode(&self, z: &[f32], t: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let dev = &self.sol.dev;
        let hw = h * w;
        let n_all = t * hw;
        let nmax = BLOCK.min(t) * hw;
        // the residual stream: the tokens
        let x = Tensor::new(dev, DType::F32, &[n_all, DIM])?;
        {
            let mut rows = vec![0f32; nmax * LATENT_C];
            let e = Tensor::new(dev, DType::BF16, &[nmax, DIM])?;
            for f0 in (0..t).step_by(BLOCK) {
                let f1 = (f0 + BLOCK).min(t);
                let n = (f1 - f0) * hw;
                for c in 0..LATENT_C {
                    for i in 0..n {
                        rows[i * LATENT_C + c] = z[c * n_all + f0 * hw + i];
                    }
                }
                let input = f32_tensor(dev, &[n, LATENT_C], &rows[..n * LATENT_C])?;
                // SAFETY: views of live tensors, dropped before them.
                let (ev, xv) = unsafe { (e.rows_view(0, n)?, x.rows_view(f0 * hw, n)?) };
                self.embed.forward(&input, &ev)?;
                // the embedding rounded to bfloat16 (the release's), widened into the stream
                widen(dev, &ev, &xv, n)?;
            }
        }
        self.sol.run(&x, t, h, w, tick)
    }
}

/// bfloat16 rows [n, DIM] -> float32 rows [n, DIM] (both on `dev`), and a wait
pub(crate) fn widen(dev: &Device, src: &Tensor, dst: &Tensor, n: usize) -> Result<()> {
    // SAFETY: bfloat16 [n, DIM] -> float32 [n, DIM], both on this device.
    let rc = unsafe { (dev.api.dequant)(dev.ctx, src.buf.ptr().cast_const(), 30, (n * DIM) as i64, dst.buf.ptr(), DType::F32.kernel_code()?) };
    dev.check(rc)?;
    dev.wait()
}

impl Sol {
    /// The layers and the head on the residual stream `x` (float32 [t h w, DIM], tokens in (t, y, x) order; t = 0 a
    /// block's first), in place -> pixels [3, frames, 16 h, 16 w] in [0, 1] and the frame count (4 t less 3 a block).
    /// `tick` between temporal windows (cancel checks).
    pub(crate) fn run(&self, x: &Tensor, t: usize, h: usize, w: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<(Vec<f32>, usize)> {
        let dev = &self.dev;
        let k = self.k;
        let qu = self.gpu.queue();
        let hw = h * w;
        let nb = t.div_ceil(BLOCK);
        let gs = nb * GLOBALS;
        let prof = std::env::var_os("NSD_PROFILE").is_some();
        let t_start = Instant::now();
        // the partitions of the even and odd layers, their windows' index tables on the card
        let mut parts: Vec<((usize, usize), Vec<Chunk>)> = Vec::new();
        let mut nmax = 0;
        let mut kmax = 0;
        let mut gmax = 1;
        for sh in [(0, 0), (WINDOW / 2, T_SHIFT)] {
            let mut chunks = Vec::new();
            for c in geometry(t, h, w, sh.0, sh.1) {
                let lens = |cu: &[i32]| cu.windows(2).map(|p| (p[1] - p[0]) as usize).max().unwrap_or(0);
                let n = (c.t1 - c.t0) * hw;
                nmax = nmax.max(n);
                kmax = kmax.max(*c.cu_k.last().unwrap() as usize);
                gmax = gmax.max(c.gsrc.len());
                chunks.push(Chunk {
                    t0: c.t0,
                    t1: c.t1,
                    windows: c.cu_q.len() - 1,
                    max_q: lens(&c.cu_q),
                    max_k: lens(&c.cu_k),
                    total_k: *c.cu_k.last().unwrap() as usize,
                    globals: c.gsrc.len(),
                    qpos: i32_tensor(dev, &c.qpos)?,
                    kpos: i32_tensor(dev, &c.kpos)?,
                    gsrc: i32_tensor(dev, &c.gsrc)?,
                    gdst: i32_tensor(dev, &c.gdst)?,
                    cu_q: i32_tensor(dev, &c.cu_q)?,
                    cu_k: i32_tensor(dev, &c.cu_k)?,
                    blocks: c.blocks,
                });
            }
            parts.push((sh, chunks));
        }
        let wmax = parts.iter().flat_map(|p| p.1.iter().map(|c| c.windows)).max().unwrap_or(1);
        let cu_zero = i32_tensor(dev, &vec![0; wmax + 1])?;
        let cs = [f32_tensor(dev, &[BLOCK * hw, ROT_DIM], &rope_table(h, w, 0, &self.inv_freq))?,
                  f32_tensor(dev, &[BLOCK * hw, ROT_DIM], &rope_table(h, w, WINDOW / 2, &self.inv_freq))?];
        let cs_g = f32_tensor(dev, &[gs, ROT_DIM], &rotations(&vec![0.0; gs * 3], &self.inv_freq))?;

        if x.dtype != DType::F32 || x.elements() != t * hw * DIM {
            return Err(Error(format!("sol: a stream {:?} {:?} for {t}x{h}x{w} tokens", x.dtype, x.shape)));
        }
        // the global rows beside the stream
        let g = f32_tensor(dev, &[gs, DIM], &(0..nb).flat_map(|_| self.g0.iter().copied()).collect::<Vec<f32>>())?;

        // a window's buffers
        let hbuf = Tensor::new(dev, DType::BF16, &[nmax.max(gs), DIM])?;
        let qkv = Tensor::new(dev, DType::BF16, &[nmax, 3 * DIM])?;
        let qw = Tensor::new(dev, DType::BF16, &[nmax, DIM])?;
        let kw = Tensor::new(dev, DType::BF16, &[kmax, DIM])?;
        let vw = Tensor::new(dev, DType::BF16, &[kmax, DIM])?;
        let ow = Tensor::new(dev, DType::BF16, &[nmax, DIM])?;
        let o = Tensor::new(dev, DType::BF16, &[nmax.max(gs), DIM])?;
        let y = Tensor::new(dev, DType::BF16, &[nmax.max(gs), DIM])?;
        let f1 = Tensor::new(dev, DType::BF16, &[MLP_ROWS, 8 * DIM])?;
        let act = Tensor::new(dev, DType::BF16, &[MLP_ROWS, 4 * DIM])?;
        let zero_rows = i32_tensor(dev, &vec![0; nmax.max(gs)])?;
        let qkv_g = Tensor::new(dev, DType::BF16, &[gs, 3 * DIM])?;
        let state = Tensor::new(dev, DType::F32, &[gs * HEADS, HEAD_DIM + 2])?;
        let splits = (BLOCK * hw).div_ceil(256);
        let scratch = Tensor::new(dev, DType::F32, &[splits * HEADS * GLOBALS, HEAD_DIM + 2])?;
        let scratch_n = (splits * HEADS * GLOBALS * (HEAD_DIM + 2)) as i64;
        let mut stages: Vec<(&str, f64)> = Vec::new();
        let mut mark = Instant::now();
        let mut lap = |name: &'static str| -> Result<()> {
            if prof {
                dev.wait()?;
                let dt = mark.elapsed().as_secs_f64();
                match stages.iter_mut().find(|(n, _)| *n == name) {
                    Some(s) => s.1 += dt,
                    None => stages.push((name, dt)),
                }
                mark = Instant::now();
            }
            Ok(())
        };
        // the MLP of rows xs (float32, in place), a slice of MLP_ROWS at a time
        let mlp = |xs: &Tensor, n: usize, l: &Layer| -> Result<()> {
            for r0 in (0..n).step_by(MLP_ROWS) {
                let m = MLP_ROWS.min(n - r0);
                // SAFETY: views of live tensors, dropped at the end of the iteration.
                let (xm, hm, fm, am, ym, zr) = unsafe { (xs.rows_view(r0, m)?, hbuf.rows_view(0, m)?, f1.rows_view(0, m)?, act.rows_view(0, m)?, y.rows_view(0, m)?, zero_rows.rows_view(0, m)?) };
                ops::rms_norm_mod(&xm, &l.n2, EPS, None, &hm)?;
                l.w1.forward(&hm, &fm)?;
                ops::swiglu(&fm, &am)?;
                l.w2.forward(&am, &ym)?;
                ops::gate_add(&xm, &ym, &zr, &l.s2)?;
            }
            Ok(())
        };
        let n_layers = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            let odd = li % 2 == 1;
            let (_, chunks) = &parts[odd as usize];
            let st = if odd { T_SHIFT } else { 0 };
            let last = li + 1 == n_layers;
            // the global rows' q, k, v (from the stream before this layer)
            {
                // SAFETY: views of live tensors.
                let hg = unsafe { hbuf.rows_view(0, gs)? };
                ops::rms_norm_mod(&g, &l.n1, EPS, None, &hg)?;
                l.qkv.forward(&hg, &qkv_g)?;
                for part in 0..2 {
                    ops::rms_rope(Rows { t: &qkv_g, offset: part * DIM, stride: 3 * DIM, tokens: gs, heads: HEADS, dim: HEAD_DIM }, &self.ones, EPS, &cs_g, ROT_DIM)?;
                }
                // SAFETY: the state [gs * HEADS, 66] float32 on this queue.
                k.check(unsafe { (k.glob_init)(qu, state.buf.ptr().cast(), (gs * HEADS) as i64) }, "glob_init")?;
            }
            lap("globals in")?;
            for c in chunks {
                tick()?;
                let n = (c.t1 - c.t0) * hw;
                // SAFETY: views of live tensors, dropped at the end of the iteration.
                let (xc, hc, qc, owc, oc, yc, zc) = unsafe {
                    (x.rows_view(c.t0 * hw, n)?, hbuf.rows_view(0, n)?, qkv.rows_view(0, n)?, ow.rows_view(0, n)?, o.rows_view(0, n)?, y.rows_view(0, n)?,
                     zero_rows.rows_view(0, n)?)
                };
                ops::rms_norm_mod(&xc, &l.n1, EPS, None, &hc)?;
                l.qkv.forward(&hc, &qc)?;
                lap("norm + qkv")?;
                // SAFETY: a view of the live table.
                let csv = unsafe { cs[odd as usize].rows_view((c.t0 + st) % BLOCK * hw, n)? };
                for part in 0..2 {
                    ops::rms_rope(Rows { t: &qc, offset: part * DIM, stride: 3 * DIM, tokens: n, heads: HEADS, dim: HEAD_DIM }, &self.ones, EPS, &csv, ROT_DIM)?;
                }
                lap("qk norm + rope")?;
                // the global rows' attention over this window's part of their blocks
                if !last {
                    for &b in &c.blocks {
                        let (fa, fb) = (c.t0.max(b * BLOCK), c.t1.min(b * BLOCK + BLOCK));
                        let off = (fa - c.t0) * hw * 3 * DIM;
                        // SAFETY: rows inside qc and qkv_g; the state's block and the scratch sized above.
                        k.check(unsafe {
                            (k.glob)(qu, at(&qkv_g, b * GLOBALS * 3 * DIM), (3 * DIM) as i64, at(&qc, off + DIM), at(&qc, off + 2 * DIM), (3 * DIM) as i64,
                                     ((fb - fa) * hw) as i64, HEADS as i64, GLOBALS as i64, state.buf.ptr().cast::<f32>().add(b * GLOBALS * HEADS * (HEAD_DIM + 2)),
                                     scratch.buf.ptr().cast(), scratch_n)
                        }, "glob")?;
                    }
                    lap("global attention")?;
                }
                // the windows: queries in window order, keys and values with their blocks' global rows
                // SAFETY: n rows of qc into qw / kw / vw by the chunk's tables (their sizes are its geometry's).
                unsafe {
                    k.check((k.rows)(qu, at(&qc, 0), (3 * DIM) as i64, std::ptr::null(), qw.buf.ptr(), DIM as i64, c.qpos.buf.ptr().cast(), n as i64, DIM as i64), "rows q")?;
                    k.check((k.rows)(qu, at(&qc, DIM), (3 * DIM) as i64, std::ptr::null(), kw.buf.ptr(), DIM as i64, c.kpos.buf.ptr().cast(), n as i64, DIM as i64), "rows k")?;
                    k.check((k.rows)(qu, at(&qc, 2 * DIM), (3 * DIM) as i64, std::ptr::null(), vw.buf.ptr(), DIM as i64, c.kpos.buf.ptr().cast(), n as i64, DIM as i64), "rows v")?;
                    k.check((k.rows)(qu, at(&qkv_g, DIM), (3 * DIM) as i64, c.gsrc.buf.ptr().cast(), kw.buf.ptr(), DIM as i64, c.gdst.buf.ptr().cast(), c.globals as i64,
                                     DIM as i64), "rows kg")?;
                    k.check((k.rows)(qu, at(&qkv_g, 2 * DIM), (3 * DIM) as i64, c.gsrc.buf.ptr().cast(), vw.buf.ptr(), DIM as i64, c.gdst.buf.ptr().cast(),
                                     c.globals as i64, DIM as i64), "rows vg")?;
                }
                lap("gather")?;
                // SAFETY: the packed windows and their tables, all on this queue.
                k.check(unsafe {
                    (k.varlen)(qu, qw.buf.ptr(), kw.buf.ptr(), vw.buf.ptr(), owc.buf.ptr(), c.windows as i64, c.cu_q.buf.ptr().cast(), c.cu_k.buf.ptr().cast(),
                               cu_zero.buf.ptr().cast(), c.max_q as i64, c.max_k as i64, n as i64, c.total_k as i64, HEADS as i64)
                }, "varlen")?;
                lap("window attention")?;
                // back to token order
                // SAFETY: n rows of owc by the chunk's query table into oc.
                k.check(unsafe { (k.rows)(qu, owc.buf.ptr(), DIM as i64, c.qpos.buf.ptr().cast(), oc.buf.ptr(), DIM as i64, std::ptr::null(), n as i64, DIM as i64) },
                        "rows o")?;
                l.out.forward(&oc, &yc)?;
                ops::gate_add(&xc, &yc, &zc, &l.s1)?;
                lap("out + add")?;
                mlp(&xc, n, l)?;
                lap("mlp")?;
            }
            // the global rows, once every window has given its keys (the last layer's are never read)
            if !last {
                for b in 0..nb {
                    // their own keys: the block's global rows
                    // SAFETY: rows of qkv_g; the state's block.
                    k.check(unsafe {
                        (k.glob)(qu, at(&qkv_g, b * GLOBALS * 3 * DIM), (3 * DIM) as i64, at(&qkv_g, b * GLOBALS * 3 * DIM + DIM), at(&qkv_g, b * GLOBALS * 3 * DIM + 2 * DIM),
                                 (3 * DIM) as i64, GLOBALS as i64, HEADS as i64, GLOBALS as i64, state.buf.ptr().cast::<f32>().add(b * GLOBALS * HEADS * (HEAD_DIM + 2)),
                                 scratch.buf.ptr().cast(), scratch_n)
                    }, "glob (own)")?;
                    // SAFETY: the state's block -> GLOBALS rows of o.
                    k.check(unsafe {
                        (k.glob_out)(qu, state.buf.ptr().cast::<f32>().add(b * GLOBALS * HEADS * (HEAD_DIM + 2)), HEADS as i64, GLOBALS as i64, at(&o, b * GLOBALS * DIM),
                                     DIM as i64)
                    }, "glob_out")?;
                }
                // SAFETY: views of live tensors.
                let (og, yg, zg) = unsafe { (o.rows_view(0, gs)?, y.rows_view(0, gs)?, zero_rows.rows_view(0, gs)?) };
                l.out.forward(&og, &yg)?;
                ops::gate_add(&g, &yg, &zg, &l.s1)?;
                mlp(&g, gs, l)?;
                lap("globals out")?;
            }
            if prof {
                dev.wait()?;
                eprintln!("fastvae: layer {li} done at {:.2} s", t_start.elapsed().as_secs_f64());
            }
        }
        drop((qkv, qw, kw, vw, ow, o, y, f1, act, qkv_g, state, scratch, hbuf));
        let t_blocks = t_start.elapsed().as_secs_f64();

        // the head, a latent frame at a time: layer norm, the float32 projection, the patches into frames
        let frames = t * PATCH_T - TRIM * nb;
        let (fh, fw) = (h * PATCH, w * PATCH);
        let plane = fh * fw;
        let mut px = vec![0f32; OUT_C * frames * plane];
        let normed = Tensor::new(dev, DType::F32, &[hw, DIM])?;
        let head = Tensor::new(dev, DType::F32, &[hw, OUT_C * PATCH_T * PATCH * PATCH])?;
        let head1 = match &self.first_out {
            Some(_) => Some(Tensor::new(dev, DType::F32, &[hw, OUT_C * PATCH * PATCH])?),
            None => None,
        };
        for ti in 0..t {
            tick()?;
            // SAFETY: a view of the live stream.
            let xv = unsafe { x.rows_view(ti * hw, hw)? };
            ops::layer_norm(&xv, DIM, Some(&self.norm_out_w), Some(&self.norm_out_b), EPS, &normed)?;
            let lt = ti % BLOCK;
            // the first latent of a block: its one frame from its own projection (DWD), or the last of the four
            let (v, nt, pt0) = match (&self.first_out, &head1) {
                (Some(f), Some(h1)) if lt == 0 => {
                    f.forward(&normed, h1)?;
                    (h1.to_f32()?, 1, PATCH_T - 1)
                }
                _ => {
                    self.proj_out.forward(&normed, &head)?;
                    (head.to_f32()?, PATCH_T, 0)
                }
            };
            let per = OUT_C * nt * PATCH * PATCH;
            for j in 0..nt {
                let pt = pt0 + j;
                if lt == 0 && pt < TRIM {
                    continue;
                }
                let fo = ti * PATCH_T + pt - TRIM * (ti / BLOCK + 1);
                for yi in 0..h {
                    for xi in 0..w {
                        let tok = &v[(yi * w + xi) * per..][..per];
                        for ch in 0..OUT_C {
                            for py in 0..PATCH {
                                let dst = (ch * frames + fo) * plane + (yi * PATCH + py) * fw + xi * PATCH;
                                let src = (ch * nt + j) * PATCH * PATCH + py * PATCH;
                                px[dst..dst + PATCH].copy_from_slice(&tok[src..src + PATCH]);
                            }
                        }
                    }
                }
            }
        }
        if prof {
            let sum: f64 = stages.iter().map(|s| s.1).sum();
            for (n, s) in &stages {
                eprintln!("fastvae stage: {n:18} {:8.1} ms  {:4.1}%", s * 1e3, s / sum * 100.0);
            }
            eprintln!("fastvae: {t}x{h}x{w} latents: blocks {t_blocks:.2} s, head {:.2} s", t_start.elapsed().as_secs_f64() - t_blocks);
        }
        Ok((crate::vae::finalize(px, frames, plane), frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every token is in exactly one window of a partition; a window's keys are its tokens and its blocks' global rows
    #[test]
    fn the_windows_cover_every_token_once() {
        for (t, h, w) in [(37, 90, 160), (7, 20, 36), (12, 17, 33), (2, 16, 16)] {
            for (s, st) in [(0, 0), (8, 2)] {
                let chunks = geometry(t, h, w, s, st);
                assert_eq!(chunks.first().map(|c| c.t0), Some(0));
                assert_eq!(chunks.last().map(|c| c.t1), Some(t));
                for p in chunks.windows(2) {
                    assert_eq!(p[0].t1, p[1].t0);
                }
                for c in &chunks {
                    let n = (c.t1 - c.t0) * h * w;
                    let mut q: Vec<i32> = c.qpos.clone();
                    q.sort();
                    assert_eq!(q, (0..n as i32).collect::<Vec<_>>(), "{t}x{h}x{w} shift {s}");
                    assert_eq!(*c.cu_q.last().unwrap() as usize, n);
                    assert_eq!(*c.cu_k.last().unwrap() as usize, n + (c.cu_q.len() - 1) * c.blocks.len() * GLOBALS);
                    assert!(c.t1 - c.t0 <= BLOCK && c.blocks.len() <= 2);
                    if st == 0 {
                        assert_eq!(c.blocks.len(), 1);
                    }
                }
            }
        }
        // shifted: the first temporal window is 3 latent frames, then 5 across two blocks
        let c = geometry(37, 90, 160, 8, 2);
        assert_eq!((c[0].t0, c[0].t1, c[1].t0, c[1].t1), (0, 3, 3, 8));
        assert_eq!(c[1].blocks, vec![0, 1]);
        // 2560x1440: 6 x 10 windows unshifted, 7 x 11 shifted
        assert_eq!(geometry(37, 90, 160, 0, 0)[0].cu_q.len() - 1, 60);
        assert_eq!(geometry(37, 90, 160, 8, 2)[0].cu_q.len() - 1, 77);
    }

    #[test]
    fn frames_per_clip() {
        // 37 latent frames (124 pixel frames: a 5.17 s draft), 72 (243), 107 (362)
        for (t, f) in [(37, 124), (72, 243), (107, 362)] {
            assert_eq!(t * PATCH_T - TRIM * t.div_ceil(BLOCK), f);
        }
    }
}
