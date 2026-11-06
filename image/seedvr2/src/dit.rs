//! The NaDiT (the reference's models/dit_v2 for the 3B, models/dit for the 7B): one call at t = 1000 turns the noise
//! and the low-resolution latent into the velocity whose step is the restored latent. bfloat16 weights and
//! activations (the reference runs under bfloat16 autocast), the residual streams in float32.
//!
//! ```text
//!   vid  [noise 16 | condition 16 | 1] a latent pixel, 1 x 2 x 2 patches -> vid_in (132 -> D)
//!   txt  the fixed positive prompt's embedding [L, 5120] -> txt_in (5120 -> D)
//!   emb  t = 1000 -> sinusoid 256 -> MLP -> 6 D: per block a shift, scale and gate for attention and the MLP
//!   block (32 for the 3B, 36 for the 7B), video and text side by side (MMDiT; the 3B's from block 10 on with one set
//!         of weights for both):
//!     h = rms(x) * scale + shift; qkv; q, k: per-head RMS norm, rotary (3B: mmrope3d, the text too; 7B: the video's
//!     window positions in [-1, 1]); attention within each window of the video tokens plus every text token (the
//!     windows: window.rs, shifted on odd blocks), the text's outputs the mean over the windows; x += out * gate;
//!     the MLP (3B: SwiGLU, 7B: GELU tanh) the same way; the 3B's last block modulates only the video (its text goes
//!     into attention normed alone, and is not read after it)
//!   out  (3B: rms norm with weight, then the first block's attention shift and scale - the reference's cache key
//!         collision - plus its own) -> vid_out (D -> 64) -> the 16 channels of each latent pixel
//! ```

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use nextsycl_core::{DevBuf, Error, Gpu, HostBuf, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::{f16_to_f32, SafeTensors};

use crate::{ffi, prof, window};

const EPS: f32 = 1e-5;
const HEAD: usize = 128;
/// the latent's channels; the DiT's input: noise, the condition and its mask
pub const LATENT: usize = 16;
const VID_IN: usize = 2 * LATENT + 1;
/// the timestep of the one step
const T: f32 = 1000.0;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// float32 to bfloat16 bits, rounding to nearest even (as PyTorch's cast)
pub fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The rotary positions of the two architectures
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rope {
    /// the 3B's mmrope3d: rotary_embedding_torch "lang" frequencies (theta 10000) over 42 features an axis; a window's
    /// video token at (L + t, h, w) for a text of L tokens, text token i at (i, i, i)
    Mm { freqs: usize },
    /// the 7B's: "pixel" frequencies (1 to 128, times pi) over 20 features an axis, a window's token at linspace(-1, 1)
    /// along each of its sides; the text unrotated
    Pixel { freqs: usize },
}

impl Rope {
    fn freqs(self) -> Vec<f32> {
        match self {
            // 1 / theta ^ (arange(0, dim, 2) / dim), dim = 2 freqs
            Rope::Mm { freqs } => (0..freqs).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / (2 * freqs) as f32)).collect(),
            // linspace(1, max_freq / 2, dim / 2) * pi
            Rope::Pixel { freqs } => linspace(1.0, 128.0, freqs).into_iter().map(|f| f * std::f32::consts::PI).collect(),
        }
    }

    /// the rotated pairs a head (three axes)
    pub fn pairs(self) -> usize {
        match self {
            Rope::Mm { freqs } | Rope::Pixel { freqs } => 3 * freqs,
        }
    }
}

/// torch.linspace(start, end, n) in float32: from both ends, as torch does
pub fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 }).collect()
}

/// A matrix's place in the weights' slab (bytes from its start)
#[derive(Clone, Copy)]
struct W(usize);

/// One side's (video or text) weights of a block, with its modulation for t = 1000 folded in
struct Branch {
    qkv: W,
    out_w: W,
    out_b: DevBuf,
    norm_q: DevBuf,
    norm_k: DevBuf,
    /// SwiGLU: [gate; in] [2 hidden, D]; GELU: [hidden, D]
    mlp_in: W,
    mlp_in_b: Option<DevBuf>,
    mlp_out: W,
    mlp_out_b: Option<DevBuf>,
    /// attention's shift, scale, gate; the MLP's (float32 [D] each)
    m: [DevBuf; 6],
}

struct Block {
    vid: Arc<Branch>,
    txt: Arc<Branch>,
    shifted: bool,
    /// the text goes into attention normed but not modulated (the 3B's last block: its ada is vid_only)
    txt_plain: bool,
}

/// A clip shape's windows, ready for the GPU: per window layout (plain, shifted) the gathers and the rotary phases
pub struct Layout {
    /// rows of the packed qkv: each window's video tokens, then the text (index N + l)
    fwd: DevBuf,
    /// each video token's row in the packed attention output
    rev: DevBuf,
    /// each window's first text row
    base: DevBuf,
    windows: usize,
    rows: usize,
    /// runs of consecutive windows of one length: (first row, length, windows)
    runs: Vec<(usize, usize, usize)>,
    /// the video tokens' rotary phases [N, P, 2] (by token, its own window's position)
    cs: DevBuf,
}

/// A latent clip's shapes and windows
pub struct Shape {
    pub t: usize,
    pub h: usize,
    pub w: usize,
    layouts: [Layout; 2],
}

impl Shape {
    /// tokens: the 1 x 2 x 2 patches
    pub fn tokens(&self) -> usize {
        self.t * (self.h / 2) * (self.w / 2)
    }
}

pub struct Dit {
    gpu: Arc<Gpu>,
    pub dim: usize,
    heads: usize,
    hidden: usize,
    swiglu: bool,
    rope: Rope,
    blocks: Vec<Block>,
    vid_in_w: W,
    vid_in_b: DevBuf,
    /// the text after txt_in, float32 [L, D]
    txt0: DevBuf,
    pub txt_len: usize,
    /// the text's rotary phases [L, P, 2] (the 3B)
    txt_cs: Option<DevBuf>,
    /// the output norm's weight, shift and scale (the 3B)
    out_norm: Option<[DevBuf; 3]>,
    vid_out_w: W,
    vid_out_b: DevBuf,
    /// the matrices (bfloat16, one after another) in pinned host memory, and on the card while a call runs: the VAE's
    /// slices at full resolution want the card's room (the 7B's 16 GB would not leave it), so the slab goes up for each
    /// call (~25 GB/s) and is let go after
    host: HostBuf,
    dev: Mutex<Option<DevBuf>>,
    /// each matrix's first byte in the slab and its tensors (to name a byte that went wrong)
    places: Vec<(usize, String)>,
    /// a modulation that leaves the norm as it is (scale 1, shift 0)
    ones: DevBuf,
    zeros: DevBuf,
    pub what: String,
}

/// The t = 1000 embedding [6 D]: the diffusers sinusoid (sin first, no shift) through emb_in's MLP, float32
fn time_embedding(st: &SafeTensors, dim: usize) -> Result<Vec<f32>> {
    let half = 128;
    let mut x: Vec<f32> = (0..2 * half)
        .map(|i| {
            let f = (-(10000f32.ln()) * (i % half) as f32 / half as f32).exp() * T;
            if i < half { f.sin() } else { f.cos() }
        })
        .collect();
    let lin = |x: &[f32], p: &str, silu: bool| -> Result<Vec<f32>> {
        let w = st.f32(&format!("{p}.weight")).map_err(ge)?;
        let b = st.f32(&format!("{p}.bias")).map_err(ge)?;
        let k = x.len();
        Ok((0..b.len())
            .map(|o| {
                let v = b[o] + w[o * k..(o + 1) * k].iter().zip(x).map(|(a, c)| a * c).sum::<f32>();
                if silu { v / (1.0 + (-v).exp()) } else { v }
            })
            .collect())
    };
    x = lin(&x, "emb_in.proj_in", true)?;
    x = lin(&x, "emb_in.proj_hid", true)?;
    let e = lin(&x, "emb_in.proj_out", false)?;
    if e.len() != 6 * dim {
        return Err(Error(format!("emb_in makes {} values, not 6 x {dim}", e.len())));
    }
    Ok(e)
}

/// Threads that fill a slab (reads of the file and the conversion to bfloat16, or a copy)
fn fill_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16)
}

/// `dst` = `src` (the same length), copied by several threads
fn fill_copy(dst: &mut [u8], src: &[u8]) {
    let piece = dst.len().div_ceil(fill_threads()).max(1);
    std::thread::scope(|sc| {
        for (d, s) in dst.chunks_mut(piece).zip(src.chunks(piece)) {
            sc.spawn(move || d.copy_from_slice(s));
        }
    });
}

/// Each tensor of `parts` (its name, its first byte in `dst`) read and converted to bfloat16 at its place, the tensors
/// shared out between threads (each reads its own: positioned reads of the file)
fn fill_convert(dst: &mut [u8], parts: &[(String, usize)], st: &SafeTensors) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // each part's slice of dst, handed out whole: the places are disjoint (laid out one after another)
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by_key(|&i| parts[i].1);
    let mut slices: Vec<Option<&mut [u8]>> = (0..parts.len()).map(|_| None).collect();
    let mut rest = dst;
    let mut at = 0usize;
    for &i in &order {
        let t = st.need(&parts[i].0).map_err(ge)?;
        let len = t.elements() as usize * 2;
        let skip = parts[i].1.checked_sub(at).ok_or_else(|| Error(format!("{}: overlaps the tensor before it in the slab", parts[i].0)))?;
        if skip + len > rest.len() {
            return Err(Error(format!("{}: past the slab's end", parts[i].0)));
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let (mine, tail) = tail.split_at_mut(len);
        slices[i] = Some(mine);
        rest = tail;
        at = parts[i].1 + len;
    }
    let jobs: Vec<Mutex<Option<&mut [u8]>>> = slices.into_iter().map(Mutex::new).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..fill_threads())
            .map(|_| {
                sc.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= parts.len() {
                            return Ok(());
                        }
                        let out = jobs[i].lock().unwrap().take().expect("each part once");
                        let n = &parts[i].0;
                        let t = st.need(n).map_err(ge)?;
                        let raw = st.read(t).map_err(ge)?;
                        match t.dtype.as_str() {
                            "BF16" => out.copy_from_slice(&raw),
                            "F16" => out.chunks_exact_mut(2).zip(raw.chunks_exact(2))
                                .for_each(|(o, c)| o.copy_from_slice(&bf16(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).to_le_bytes())),
                            "F32" => out.chunks_exact_mut(2).zip(raw.chunks_exact(4))
                                .for_each(|(o, c)| o.copy_from_slice(&bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())),
                            d => return Err(Error(format!("{n}: {d} weights (F16, BF16 or F32 expected)"))),
                        }
                    }
                })
            })
            .collect();
        hs.into_iter().try_for_each(|h| h.join().map_err(|_| Error("a slab filler panicked".into()))?)
    })
}

impl Dit {
    /// The matrices' slab in pinned host memory (for another card's load of the same file)
    pub fn host_slab(&self) -> &[u8] {
        self.host.as_slice()
    }

    /// `from`: another card's slab of this file (`host_slab()`), copied instead of read and converted again
    pub fn load(path: &std::path::Path, txt: &[u16], txt_len: usize, nsd: &Nsd, from: Option<&[u8]>, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(path).map_err(ge)?;
        let shape = |n: &str| -> Result<Vec<usize>> { Ok(st.need(n).map_err(ge)?.shape.iter().map(|&v| v as usize).collect()) };
        let vin = shape("vid_in.proj.weight")?;
        let dim = vin[0];
        if vin[1] != 4 * VID_IN || dim % HEAD != 0 {
            return Err(Error(format!("{}: vid_in takes {} features (a SeedVR2 NaDiT takes {})", path.display(), vin[1], 4 * VID_IN)));
        }
        let layers = (0..).take_while(|i| st.tensor(&format!("blocks.{i}.attn.proj_qkv.vid.weight")).is_some()
            || st.tensor(&format!("blocks.{i}.attn.proj_qkv.all.weight")).is_some()).count();
        let swiglu = st.tensor("blocks.0.mlp.vid.proj_in_gate.weight").is_some();
        let rope = match shape("blocks.0.attn.rope.rope.freqs")?[0] {
            21 => Rope::Mm { freqs: 21 },
            10 => Rope::Pixel { freqs: 10 },
            n => return Err(Error(format!("{}: {n} rotary frequencies (the 3B has 21, the 7B 10)", path.display()))),
        };
        let hidden = shape("blocks.0.mlp.vid.proj_in.weight")?[0];
        let emb = time_embedding(&st, dim)?;
        // a tensor as bfloat16 bytes
        let bytes = |n: &str| -> Result<Vec<u8>> {
            let t = st.need(n).map_err(ge)?;
            let raw = st.read(t).map_err(ge)?;
            Ok(match t.dtype.as_str() {
                "F16" => raw.chunks_exact(2).flat_map(|c| bf16(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).to_le_bytes()).collect(),
                "BF16" => raw,
                "F32" => raw.chunks_exact(4).flat_map(|c| bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()).collect(),
                d => return Err(Error(format!("{n}: {d} weights (F16, BF16 or F32 expected)"))),
            })
        };
        // a matrix's place in the slab (the tensors named one after another), filled once every place is known
        let plan: RefCell<Vec<(Vec<String>, usize)>> = RefCell::new(Vec::new());
        let size = Cell::new(0usize);
        let mat = |names: &[String]| -> Result<W> {
            let at = size.get();
            for n in names {
                size.set(size.get() + st.need(n).map_err(ge)?.elements() as usize * 2);
            }
            plan.borrow_mut().push((names.to_vec(), at));
            Ok(W(at))
        };
        let vec = |n: &str| -> Result<DevBuf> { DevBuf::from_f32(&gpu, &st.f32(n).map_err(ge)?) };
        let branch = |i: usize, side: &str| -> Result<Branch> {
            let p = |s: &str| format!("blocks.{i}.{s}");
            let ada = |l: usize, g: usize, name: &str| -> Result<DevBuf> {
                let param = st.f32(&p(&format!("ada.{side}.{name}"))).map_err(ge)?;
                // emb is "(d l g)": d outermost, then the layer (attn, mlp), then shift / scale / gate
                DevBuf::from_f32(&gpu, &(0..dim).map(|d| emb[d * 6 + l * 3 + g] + param[d]).collect::<Vec<_>>())
            };
            let (mlp_in, mlp_in_b, mlp_out_b) = if swiglu {
                (mat(&[p(&format!("mlp.{side}.proj_in_gate.weight")), p(&format!("mlp.{side}.proj_in.weight"))])?, None, None)
            } else {
                (mat(&[p(&format!("mlp.{side}.proj_in.weight"))])?, Some(vec(&p(&format!("mlp.{side}.proj_in.bias")))?),
                 Some(vec(&p(&format!("mlp.{side}.proj_out.bias")))?))
            };
            Ok(Branch {
                qkv: mat(&[p(&format!("attn.proj_qkv.{side}.weight"))])?,
                out_w: mat(&[p(&format!("attn.proj_out.{side}.weight"))])?,
                out_b: vec(&p(&format!("attn.proj_out.{side}.bias")))?,
                norm_q: vec(&p(&format!("attn.norm_q.{side}.weight")))?,
                norm_k: vec(&p(&format!("attn.norm_k.{side}.weight")))?,
                mlp_in,
                mlp_in_b,
                mlp_out: mat(&[p(&format!("mlp.{side}.proj_out.weight"))])?,
                mlp_out_b,
                m: [ada(0, 0, "attn_shift")?, ada(0, 1, "attn_scale")?, ada(0, 2, "attn_gate")?, ada(1, 0, "mlp_shift")?, ada(1, 1, "mlp_scale")?,
                    ada(1, 2, "mlp_gate")?],
            })
        };
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let (vid, txt) = if st.tensor(&format!("blocks.{i}.attn.proj_qkv.all.weight")).is_some() {
                let b = Arc::new(branch(i, "all")?);
                (b.clone(), b)
            } else {
                (Arc::new(branch(i, "vid")?), Arc::new(branch(i, "txt")?))
            };
            // models/dit_v2 (the 3B) leaves the last block's text out of its ada (vid_only); models/dit (the 7B) does not
            blocks.push(Block { vid, txt, shifted: i % 2 == 1, txt_plain: i == layers - 1 && matches!(rope, Rope::Mm { .. }) });
        }
        // the text through txt_in once: it is the same every call
        let tin = shape("txt_in.weight")?;
        if txt.len() != txt_len * tin[1] {
            return Err(Error(format!("the text embedding has {} values, not {txt_len} x {}", txt.len(), tin[1])));
        }
        let tb = DevBuf::new(&gpu, txt.len() * 2)?;
        tb.write(0, &txt.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let txt0 = DevBuf::f32(&gpu, txt_len * dim)?;
        let tw = DevBuf::new(&gpu, tin[0] * tin[1] * 2)?;
        tw.write(0, &bytes("txt_in.weight")?)?;
        let tbias = vec("txt_in.bias")?;
        nsd.linear(tb.ptr(), Dt::BF16, txt_len, tin[1], tw.ptr(), dim, tbias.ptr(), txt0.ptr(), Dt::F32)?;
        nsd.wait()?;
        let txt_cs = match rope {
            Rope::Mm { freqs } => {
                let f = rope.freqs();
                let mut cs = Vec::with_capacity(txt_len * 3 * freqs * 2);
                for i in 0..txt_len {
                    for j in 0..3 * freqs {
                        let a = i as f32 * f[j % freqs];
                        cs.extend([(a as f64).cos() as f32, (a as f64).sin() as f32]);
                    }
                }
                Some(DevBuf::from_f32(&gpu, &cs)?)
            }
            Rope::Pixel { .. } => None,
        };
        let out_norm = if st.tensor("vid_out_norm.weight").is_some() {
            let (sh, sc) = (st.f32("vid_out_ada.out_shift").map_err(ge)?, st.f32("vid_out_ada.out_scale").map_err(ge)?);
            Some([vec("vid_out_norm.weight")?, DevBuf::from_f32(&gpu, &(0..dim).map(|d| emb[d * 6] + sh[d]).collect::<Vec<_>>())?,
                  DevBuf::from_f32(&gpu, &(0..dim).map(|d| emb[d * 6 + 1] + sc[d]).collect::<Vec<_>>())?])
        } else {
            None
        };
        let (vid_in_w, vid_out_w) = (mat(&["vid_in.proj.weight".into()])?, mat(&["vid_out.proj.weight".into()])?);
        let mut host = HostBuf::new(&gpu, size.get())?;
        let planned = plan.take();
        let places: Vec<(usize, String)> = planned.iter().map(|(n, at)| (*at, n.join(" + "))).collect();
        match from.filter(|f| f.len() == host.len) {
            // another card's slab of the same file: the same layout, copied
            Some(f) => fill_copy(host.as_mut_slice(), f),
            None => {
                let mut parts = Vec::new();
                for (names, at) in planned {
                    let mut at = at;
                    for n in names {
                        let t = st.need(&n).map_err(ge)?;
                        parts.push((n, at));
                        at += t.elements() as usize * 2;
                    }
                }
                fill_convert(host.as_mut_slice(), &parts, &st)?;
            }
        }
        let what = format!("{} blocks of {dim} ({} heads), {} MLP of {hidden}, {}", layers, dim / HEAD, if swiglu { "SwiGLU" } else { "GELU" },
                           match rope { Rope::Mm { .. } => "mmrope3d", Rope::Pixel { .. } => "window rope" });
        let dit = Dit {
            gpu: gpu.clone(),
            dim,
            heads: dim / HEAD,
            hidden,
            swiglu,
            rope,
            blocks,
            vid_in_w,
            vid_in_b: vec("vid_in.proj.bias")?,
            txt0,
            txt_len,
            txt_cs,
            out_norm,
            vid_out_w,
            host,
            dev: Mutex::new(None),
            places,
            vid_out_b: vec("vid_out.proj.bias")?,
            ones: DevBuf::from_f32(&gpu, &vec![1.0; dim])?,
            zeros: DevBuf::from_f32(&gpu, &vec![0.0; dim])?,
            what,
        };
        log(format!("seedvr2 dit: {} on {} in {:.1} s", dit.what, gpu.name, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The weights' bytes (the slab's)
    pub fn bytes(&self) -> usize {
        self.host.len
    }

    /// The slab on the card (uploaded when it is not), held for a call
    fn slab(&self) -> Result<std::sync::MutexGuard<'_, Option<DevBuf>>> {
        let mut g = self.dev.lock().unwrap();
        if g.is_none() {
            let d = DevBuf::new(&self.gpu, self.host.len)?;
            // in pieces of 512 MiB: one copy of several GB has come out short (a DiT gone non-finite now and then)
            const PIECE: usize = 512 << 20;
            let host = self.host.as_slice();
            prof::timed(&self.gpu, "dit weights up (pinned host -> card, PCIe)", 0.0, self.host.len as f64, || {
                (0..host.len()).step_by(PIECE).try_for_each(|at| d.write(at, &host[at..(at + PIECE).min(host.len())]))
            })?;
            // NS_SVR_VERIFY: the card's copy read back and compared
            if std::env::var_os("NS_SVR_VERIFY").is_some() {
                let mut back = vec![0u8; PIECE];
                let mut bad = 0usize;
                for at in (0..host.len()).step_by(PIECE) {
                    let n = PIECE.min(host.len() - at);
                    d.read(at, &mut back[..n])?;
                    bad += back[..n].iter().zip(&host[at..at + n]).filter(|(a, b)| a != b).count();
                }
                eprintln!("seedvr2 dit weights on {}: {} of {} bytes differ from the host's", self.gpu.name, bad, host.len());
            }
            *g = Some(d);
        }
        Ok(g)
    }

    /// The card's copy against the host's: the bytes that differ and the first of them with its tensor (a check after
    /// a call went non-finite)
    pub fn verify(&self) -> Result<(usize, Option<(usize, String)>)> {
        let g = self.dev.lock().unwrap();
        let Some(d) = g.as_ref() else { return Ok((0, None)) };
        const PIECE: usize = 512 << 20;
        let host = self.host.as_slice();
        let mut back = vec![0u8; PIECE];
        let (mut bad, mut first) = (0usize, None);
        for at in (0..host.len()).step_by(PIECE) {
            let n = PIECE.min(host.len() - at);
            d.read(at, &mut back[..n])?;
            for (i, (a, b)) in back[..n].iter().zip(&host[at..at + n]).enumerate() {
                if a != b {
                    bad += 1;
                    first.get_or_insert(at + i);
                }
            }
        }
        let named = first.map(|f| {
            let t = self.places.iter().rev().find(|(at, _)| *at <= f).map(|(at, n)| format!("{n} +{}", f - at)).unwrap_or_default();
            (f, t)
        });
        Ok((bad, named))
    }

    /// The card's copy let go (the next call uploads it again)
    pub fn evict(&self) {
        *self.dev.lock().unwrap() = None;
    }

    /// The windows of a latent clip of t x h x w latent pixels (h, w even)
    pub fn shape(&self, t: usize, h: usize, w: usize) -> Result<Shape> {
        let (ph, pw) = (h / 2, w / 2);
        let n = t * ph * pw;
        let l = self.txt_len;
        let f = self.rope.freqs();
        let p = self.rope.pairs();
        let layout = |shifted: bool| -> Result<Layout> {
            let wins = window::windows(t, ph, pw, shifted);
            let mut fwd: Vec<i32> = Vec::with_capacity(n + wins.len() * l);
            let mut rev = vec![0i32; n];
            let mut base = Vec::with_capacity(wins.len());
            let mut cs = vec![0f32; n * p * 2];
            let mut runs: Vec<(usize, usize, usize)> = Vec::new();
            for win in &wins {
                let start = fwd.len();
                let [(t0, t1), (h0, h1), (w0, w1)] = *win;
                let (pt, phh, pww): (Vec<f32>, Vec<f32>, Vec<f32>) = match self.rope {
                    // positions along each axis of the window: (L + t, h, w) in whole steps
                    Rope::Mm { .. } => ((0..t1 - t0).map(|i| (l + i) as f32).collect(), (0..h1 - h0).map(|i| i as f32).collect(),
                                        (0..w1 - w0).map(|i| i as f32).collect()),
                    Rope::Pixel { .. } => (linspace(-1.0, 1.0, t1 - t0), linspace(-1.0, 1.0, h1 - h0), linspace(-1.0, 1.0, w1 - w0)),
                };
                for (a, tt) in (t0..t1).enumerate() {
                    for (b, hh) in (h0..h1).enumerate() {
                        for (c, ww) in (w0..w1).enumerate() {
                            let tok = (tt * ph + hh) * pw + ww;
                            rev[tok] = fwd.len() as i32;
                            fwd.push(tok as i32);
                            let nf = p / 3;
                            for j in 0..p {
                                let pos = [pt[a], phh[b], pww[c]][j / nf];
                                let ang = pos * f[j % nf];
                                cs[(tok * p + j) * 2] = (ang as f64).cos() as f32;
                                cs[(tok * p + j) * 2 + 1] = (ang as f64).sin() as f32;
                            }
                        }
                    }
                }
                base.push(fwd.len() as i32);
                fwd.extend((0..l).map(|i| (n + i) as i32));
                let len = fwd.len() - start;
                match runs.last_mut() {
                    Some(r) if r.1 == len => r.2 += 1,
                    _ => runs.push((start, len, 1)),
                }
            }
            let gpu = &self.gpu;
            let ints = |v: &[i32]| -> Result<DevBuf> {
                let b = DevBuf::new(gpu, v.len() * 4)?;
                b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                Ok(b)
            };
            Ok(Layout { rows: fwd.len(), windows: wins.len(), fwd: ints(&fwd)?, rev: ints(&rev)?, base: ints(&base)?, runs, cs: DevBuf::from_f32(gpu, &cs)? })
        };
        Ok(Shape { t, h, w, layouts: [layout(false)?, layout(true)?] })
    }

    /// The velocity of one call: noise and cond float32 [t, h, w, 16] -> v float32 [t, h, w, 16]. `taps`: the
    /// blocks after which the video tokens float32 [N, D] are handed out (a check's)
    pub fn forward(&self, nsd: &Nsd, s: &Shape, noise: &[f32], cond: &[f32], taps: &mut dyn FnMut(usize, &DevBuf, &DevBuf) -> Result<()>) -> Result<Vec<f32>> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let slab = self.slab()?;
        let base = slab.as_ref().expect("uploaded").ptr().cast::<u8>();
        let w = |m: W| -> *const c_void { base.wrapping_add(m.0).cast() };
        let (d, l, n) = (self.dim, self.txt_len, s.tokens());
        let (ph, pw) = (s.h / 2, s.w / 2);
        if noise.len() != s.t * s.h * s.w * LATENT || cond.len() != noise.len() {
            return Err(Error("the noise and the condition must be t x h x w x 16".into()));
        }
        // the patches: "(T t) (H h) (W w) c -> T H W (t h w c)" with t = 1, c = noise | cond | mask
        let mut x = vec![0u16; n * 4 * VID_IN];
        for tt in 0..s.t {
            for y in 0..ph {
                for xx in 0..pw {
                    let tok = (tt * ph + y) * pw + xx;
                    for (q, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                        let px = (tt * s.h + 2 * y + dy) * s.w + 2 * xx + dx;
                        let o = tok * 4 * VID_IN + q * VID_IN;
                        for c in 0..LATENT {
                            x[o + c] = bf16(noise[px * LATENT + c]);
                            x[o + LATENT + c] = bf16(cond[px * LATENT + c]);
                        }
                        x[o + 2 * LATENT] = bf16(1.0);
                    }
                }
            }
        }
        let xin = DevBuf::new(gpu, x.len() * 2)?;
        xin.write(0, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let xv = DevBuf::f32(gpu, n * d)?;
        nsd.linear(xin.ptr(), Dt::BF16, n, 4 * VID_IN, w(self.vid_in_w), d, self.vid_in_b.ptr(), xv.ptr(), Dt::F32)?;
        let xt = DevBuf::f32(gpu, l * d)?;
        xt.copy_within(0, &self.txt0, 0, l * d * 4)?;
        let i3 = 3 * d;
        let rows_max = s.layouts.iter().map(|y| y.rows).max().unwrap_or(0);
        let hb = DevBuf::new(gpu, n * d * 2)?;
        let htb = DevBuf::new(gpu, l * d * 2)?;
        let qkv = DevBuf::new(gpu, n * i3 * 2)?;
        let qkvt = DevBuf::new(gpu, l * i3 * 2)?;
        let packed = DevBuf::new(gpu, rows_max * i3 * 2)?;
        let att = DevBuf::new(gpu, rows_max * d * 2)?;
        let av = DevBuf::new(gpu, n * d * 2)?;
        let at = DevBuf::new(gpu, l * d * 2)?;
        let ov = DevBuf::new(gpu, n * d * 2)?;
        let ot = DevBuf::new(gpu, l * d * 2)?;
        let up = if self.swiglu { 2 * self.hidden } else { self.hidden };
        let u = DevBuf::new(gpu, n * up * 2)?;
        let su = DevBuf::new(gpu, n * self.hidden * 2)?;
        crate::note(gpu, "the dit's buffers");
        let p = self.rope.pairs() as i64;
        let last = self.blocks.len() - 1;
        // a GEMM's FLOPs and bytes (bf16 in, out of the given width)
        let mm = |m: usize, kk: usize, nn: usize| -> (f64, f64) {
            let (m, kk, nn) = (m as f64, kk as f64, nn as f64);
            (2.0 * m * kk * nn, 2.0 * (m * kk + kk * nn + m * nn))
        };
        let rowwise = |m: usize, c: usize| 8.0 * (m * c) as f64; // a float32 row read, a bfloat16 row written, and back
        // the attention in bfloat16 on the silo (as the reference); else the shared one in half (q, k, v made half for it)
        let silo = ffi::silo()?;
        let half = silo.is_none() as i32;
        let qkv_dt = if silo.is_some() { Dt::BF16 } else { Dt::F16 };
        for (bi, b) in self.blocks.iter().enumerate() {
            let y = &s.layouts[b.shifted as usize];
            prof::timed(gpu, "dit norm + modulation", 0.0, rowwise(n + l, d), || {
                // SAFETY (every call below): the buffers hold the rows and columns named (sized above for this shape).
                unsafe {
                    ffi::check((k.norm_mod)(gpu.raw(), xv.fp(), n as i64, d as i64, std::ptr::null(), 1e-5, b.vid.m[1].fp(), b.vid.m[0].fp(), hb.ptr()), "norm")?;
                    let (sc, sh) = if b.txt_plain { (self.ones.fp(), self.zeros.fp()) } else { (b.txt.m[1].fp(), b.txt.m[0].fp()) };
                    ffi::check((k.norm_mod)(gpu.raw(), xt.fp(), l as i64, d as i64, std::ptr::null(), 1e-5, sc, sh, htb.ptr()), "norm")
                }
            })?;
            let (f, by) = mm(n + l, d, i3);
            prof::timed(gpu, "dit linear qkv", f, by, || {
                // q, k, v in half: the shared attention reads half in place (bfloat16 it copies to half first)
                nsd.linear(hb.ptr(), Dt::BF16, n, d, w(b.vid.qkv), i3, none(), qkv.ptr(), qkv_dt)?;
                nsd.linear(htb.ptr(), Dt::BF16, l, d, w(b.txt.qkv), i3, none(), qkvt.ptr(), qkv_dt)
            })?;
            prof::timed(gpu, "dit qk norm, rope, window gathers", 0.0, 2.0 * 2.0 * ((n + l) * 2 * d + y.rows * i3 + (y.rows + n) * d) as f64, || unsafe {
                ffi::check((k.qk_norm_rope)(gpu.raw(), qkv.ptr(), n as i64, self.heads as i64, HEAD as i64, i3 as i64, b.vid.norm_q.fp(), b.vid.norm_k.fp(), EPS,
                                            y.cs.fp(), p, half), "qk norm")?;
                ffi::check((k.qk_norm_rope)(gpu.raw(), qkvt.ptr(), l as i64, self.heads as i64, HEAD as i64, i3 as i64, b.txt.norm_q.fp(), b.txt.norm_k.fp(),
                                            EPS, self.txt_cs.as_ref().map(|c| c.fp() as *const f32).unwrap_or(std::ptr::null()), p, half), "qk norm")?;
                ffi::check((k.gather)(gpu.raw(), qkv.ptr(), n as i64, i3 as i64, qkvt.ptr(), i3 as i64, y.fwd.ptr().cast(), y.rows as i64, i3 as i64,
                                      packed.ptr(), 0), "windows")
            })?;
            // 4 S^2 D a head and window (q k^T and the weighted sum)
            let af: f64 = y.runs.iter().map(|&(_, len, count)| 4.0 * (len * len * d) as f64 * count as f64).sum();
            prof::timed(gpu, "dit window attention", af, 2.0 * (y.rows * (i3 + d)) as f64, || {
                for &(r0, len, count) in &y.runs {
                    let q = packed.ptr().cast::<u16>().wrapping_add(r0 * i3);
                    let o = att.ptr().cast::<u16>().wrapping_add(r0 * d).cast::<c_void>();
                    match silo {
                        // SAFETY: count windows of len packed rows (stride i3) from r0, out the same rows of d.
                        Some(sl) => ffi::check(unsafe {
                            (sl.attention)(gpu.raw(), q.cast(), q.wrapping_add(d).cast(), q.wrapping_add(2 * d).cast(), count as i64, len as i64,
                                           self.heads as i64, HEAD as i64, i3 as i64, o)
                        }, "silo attention")?,
                        None => nsd.attention_batch(q.cast::<c_void>(), q.wrapping_add(d).cast::<c_void>(), q.wrapping_add(2 * d).cast::<c_void>(),
                                                    Dt::F16, count, len, self.heads, HEAD, i3, o, Dt::F16)?,
                    }
                }
                Ok(())
            })?;
            prof::timed(gpu, "dit qk norm, rope, window gathers", 0.0, 2.0 * 2.0 * (n * d + y.windows * l * d) as f64, || unsafe {
                ffi::check((k.gather)(gpu.raw(), att.ptr(), y.rows as i64, d as i64, att.ptr(), d as i64, y.rev.ptr().cast(), n as i64, d as i64, av.ptr(), half), "windows back")?;
                ffi::check((k.text_mean)(gpu.raw(), att.ptr(), y.base.ptr().cast(), y.windows as i64, l as i64, d as i64, at.ptr(), half), "text mean")
            })?;
            // the last block's text is never read again
            let tl = if bi == last { 0 } else { l };
            let (f, by) = mm(n + tl, d, d);
            prof::timed(gpu, "dit linear out", f, by, || {
                nsd.linear(av.ptr(), Dt::BF16, n, d, w(b.vid.out_w), d, b.vid.out_b.ptr(), ov.ptr(), Dt::BF16)?;
                if bi != last {
                    nsd.linear(at.ptr(), Dt::BF16, l, d, w(b.txt.out_w), d, b.txt.out_b.ptr(), ot.ptr(), Dt::BF16)?;
                }
                Ok(())
            })?;
            prof::timed(gpu, "dit gates + residual", 0.0, 10.0 * ((n + tl) * d) as f64, || unsafe {
                ffi::check((k.gate_add)(gpu.raw(), xv.fp(), ov.ptr(), n as i64, d as i64, b.vid.m[2].fp()), "gate")?;
                if bi != last {
                    ffi::check((k.gate_add)(gpu.raw(), xt.fp(), ot.ptr(), l as i64, d as i64, b.txt.m[2].fp()), "gate")?;
                }
                Ok(())
            })?;
            let sides: &[(&Branch, &DevBuf, usize)] = if bi == last { &[(&b.vid, &xv, n)] } else { &[(&b.vid, &xv, n), (&b.txt, &xt, l)] };
            for &(br, xs, m) in sides {
                prof::timed(gpu, "dit norm + modulation", 0.0, rowwise(m, d), || unsafe {
                    ffi::check((k.norm_mod)(gpu.raw(), xs.fp(), m as i64, d as i64, std::ptr::null(), 1e-5, br.m[4].fp(), br.m[3].fp(), hb.ptr()), "norm")
                })?;
                let (f1, b1) = mm(m, d, up);
                let (f2, b2) = mm(m, self.hidden, d);
                prof::timed(gpu, "dit mlp (linears + activation)", f1 + f2, b1 + b2 + 6.0 * (m * up) as f64, || {
                    if self.swiglu {
                        nsd.linear(hb.ptr(), Dt::BF16, m, d, w(br.mlp_in), up, none(), u.ptr(), Dt::BF16)?;
                        nsd.swiglu(u.ptr(), Dt::BF16, m, self.hidden, su.ptr(), Dt::BF16)?;
                        nsd.linear(su.ptr(), Dt::BF16, m, self.hidden, w(br.mlp_out), d, none(), ov.ptr(), Dt::BF16)
                    } else {
                        let bias = br.mlp_in_b.as_ref().map(|b| b.ptr() as *const c_void).unwrap_or(none());
                        nsd.linear(hb.ptr(), Dt::BF16, m, d, w(br.mlp_in), up, bias, u.ptr(), Dt::BF16)?;
                        nsd.gelu(u.ptr(), Dt::BF16, m * self.hidden, true)?;
                        let bias = br.mlp_out_b.as_ref().map(|b| b.ptr() as *const c_void).unwrap_or(none());
                        nsd.linear(u.ptr(), Dt::BF16, m, self.hidden, w(br.mlp_out), d, bias, ov.ptr(), Dt::BF16)
                    }
                })?;
                prof::timed(gpu, "dit gates + residual", 0.0, 10.0 * (m * d) as f64, || unsafe {
                    ffi::check((k.gate_add)(gpu.raw(), xs.fp(), ov.ptr(), m as i64, d as i64, br.m[5].fp()), "gate")
                })?;
            }
            taps(bi, &xv, &xt)?;
        }
        // the output: (3B) its norm and modulation, then the 64 = 2 x 2 x 16 of each patch
        unsafe {
            match &self.out_norm {
                Some([w, sh, sc]) => ffi::check((k.norm_mod)(gpu.raw(), xv.fp(), n as i64, d as i64, w.fp(), EPS, sc.fp(), sh.fp(), hb.ptr()), "out norm")?,
                None => ffi::check((k.to_bf16)(gpu.raw(), xv.fp(), hb.ptr(), (n * d) as i64), "to bf16")?,
            }
        }
        let vo = DevBuf::f32(gpu, n * 4 * LATENT)?;
        nsd.linear(hb.ptr(), Dt::BF16, n, d, w(self.vid_out_w), 4 * LATENT, self.vid_out_b.ptr(), vo.ptr(), Dt::F32)?;
        nsd.wait()?;
        let o = vo.to_f32()?;
        // "T H W (t h w c) -> (T t) (H h) (W w) c"
        let mut v = vec![0f32; s.t * s.h * s.w * LATENT];
        for tt in 0..s.t {
            for y in 0..ph {
                for xx in 0..pw {
                    let tok = (tt * ph + y) * pw + xx;
                    for (q, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                        let px = (tt * s.h + 2 * y + dy) * s.w + 2 * xx + dx;
                        v[px * LATENT..(px + 1) * LATENT].copy_from_slice(&o[tok * 4 * LATENT + q * LATENT..tok * 4 * LATENT + (q + 1) * LATENT]);
                    }
                }
            }
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between two bfloat16 values: to the even one (1.0)
        assert_eq!(bf16(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(bf16(1.0 + 3.0 / 256.0), 0x3f82);
    }

    #[test]
    fn the_rope_frequencies_are_rotary_embedding_torchs() {
        let f = Rope::Mm { freqs: 21 }.freqs();
        assert_eq!(f.len(), 21);
        assert!((f[0] - 1.0).abs() < 1e-7 && (f[1] - 1.0 / 10000f32.powf(2.0 / 42.0)).abs() < 1e-7);
        let p = Rope::Pixel { freqs: 10 }.freqs();
        assert!((p[0] - std::f32::consts::PI).abs() < 1e-6 && (p[9] - 128.0 * std::f32::consts::PI).abs() < 1e-3);
        assert_eq!(linspace(-1.0, 1.0, 1), vec![-1.0]);
        assert_eq!(linspace(-1.0, 1.0, 3), vec![-1.0, 0.0, 1.0]);
    }
}
