//! The denoiser: Z-Image's single-stream transformer (diffusers' `ZImageTransformer2DModel`,
//! transformer_z_image.py): 30 blocks of 3,840 features (30 heads x 128), two noise-refiner blocks over the image's
//! tokens and two context-refiner blocks over the caption's. Ming-Image's DiT (image/mingimage/src/dit.rs) is the same
//! transformer and this began as a copy of it; what differs is what comes in - the caption is Qwen3-4B's hidden states
//! (no query tokens, no direct condition) - and that both sequences are padded to a multiple of 32 with the model's pad
//! tokens, which every attention sees (the pipeline's batch of one has no mask).
//!
//! ```text
//!   image tokens: the latents' 2 x 2 patches [N, 64] -> x_embedder; rows N..N' the x pad token -> noise refiner (modulated)
//!   caption:      rmsnorm + linear of Qwen3-4B's states [L, 2560]; rows L..L' the cap pad token -> context refiner
//!                 (not modulated; once a prompt)
//!   [image (N') | caption (L')] -> 30 blocks -> the image's N rows: layernorm * (1 + scale), linear -> [N, 64]
//!   block: x += tanh(g1) * rmsnorm(attention(rmsnorm(x) * (1 + s1)))
//!          x += tanh(g2) * rmsnorm(w2(silu(w1 h) * w3 h)),  h = rmsnorm(x) * (1 + s2)
//! ```
//!
//! N', L': rounded up to 32. s1 g1 s2 g2: the block's adaLN linear of the timestep's embedding (no SiLU before it), the
//! same for every token. Positions: caption row j at (1 + j, 0, 0), its pad rows included; the image's at (L' + 1, row,
//! column), its pad rows at (0, 0, 0). The RoPE pairs features (2i, 2i + 1); the shared kernel rotates (i, 64 + i), so q's and k's rows
//! (and their norms) are permuted at load.
//!
//! The file is ComfyUI's layout as GGUF (`lumina2`: q | k | v fused, `attention.out`, `x_embedder`, `final_layer`). A
//! float matrix (BF16, F16, F32) is made half on the card at load and multiplied whole (oneDNN); a quantized one (Q8_0,
//! K-quants) stays as stored for the shared quantized product. Activations: the residual stream float32, the
//! products' inputs half; q | k | v are made half at 1/16 and the gated product at 1/256 (|k| 5.6e3 and
//! |silu(w1 h) w3 h| 6.5e4 at the first step of the README's prompt), the products after them float32 into an rms norm,
//! whose epsilon takes the scale (the norm is otherwise blind to it).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;

pub const DIM: usize = 3840;
pub const HEADS: usize = 30;
pub const HEAD: usize = 128;
pub const FFN: usize = 10240;
/// a patch's features: 2 x 2 x 16 latent channels
pub const PATCH_IN: usize = 64;
pub const LATENT: usize = 16;
/// the caption's Qwen3-4B features
pub const CAP: usize = 2560;
const EPS: f32 = 1e-5;
const ADALN: usize = 256;
const AXES: [usize; 3] = [32, 48, 48];
const THETA: f64 = 256.0;
pub const SEQ_MULTI_OF: usize = 32;
/// the timestep's scale (config.json t_scale)
const T_SCALE: f32 = 1000.0;

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// `n` rounded up to the sequences' multiple
pub fn padded(n: usize) -> usize {
    n.div_ceil(SEQ_MULTI_OF) * SEQ_MULTI_OF
}

/// The DiT's file in ComfyUI's layout: a GGUF (float or quantized matrices), or ComfyUI's int8 ConvRot safetensors
/// (every block matrix int8 with a scale per row, its inputs rotated by a normalized Hadamard matrix in groups of 256;
/// the rest float32)
pub enum Source {
    Gguf(Gguf),
    St(SafeTensors),
}

/// A tensor's stored type: a GGUF type (a safetensors float maps to its GGUF twin), or int8 ConvRot
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dty {
    G(GType),
    I8,
}

/// The ConvRot group (comfy_quant's convrot_groupsize)
const GROUP: usize = 256;

impl Source {
    pub fn open(path: &std::path::Path) -> Result<Source> {
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut std::fs::File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?, &mut magic)
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Ok(if &magic == b"GGUF" { Source::Gguf(Gguf::open(path).map_err(ge)?) } else { Source::St(SafeTensors::open(path).map_err(ge)?) })
    }

    pub fn path(&self) -> std::path::PathBuf {
        match self {
            Source::Gguf(f) => f.paths[0].clone(),
            Source::St(s) => s.path.clone(),
        }
    }

    /// What it is, for a log line and the checks: "lumina2" (a GGUF's architecture) or "int8 ConvRot"
    pub fn what(&self) -> String {
        match self {
            Source::Gguf(f) => f.architecture().to_string(),
            Source::St(_) => "int8 ConvRot".into(),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match self {
            Source::Gguf(f) => f.tensor(name).is_some(),
            Source::St(s) => s.tensor(name).is_some(),
        }
    }

    /// (stored type, shape outermost first)
    fn info(&self, name: &str) -> Result<(Dty, Vec<u64>)> {
        let none = || Error(format!("{}: no tensor {name}", self.path().display()));
        match self {
            Source::Gguf(f) => f.tensor(name).map(|t| (Dty::G(t.ty), t.shape.clone())).ok_or_else(none),
            Source::St(s) => {
                let t = s.tensor(name).ok_or_else(none)?;
                let d = match t.dtype.as_str() {
                    "F32" => Dty::G(GType::F32),
                    "F16" => Dty::G(GType::F16),
                    "BF16" => Dty::G(GType::BF16),
                    "I8" => Dty::I8,
                    d => return Err(Error(format!("{name}: {d} is not read"))),
                };
                Ok((d, t.shape.clone()))
            }
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        match self {
            Source::Gguf(f) => f.read(f.tensor(name).ok_or_else(|| Error(format!("no tensor {name}")))?).map_err(ge),
            Source::St(s) => s.read(s.need(name).map_err(ge)?).map_err(ge),
        }
    }

    /// Bytes its matrices take on a card as this engine keeps them (float ones made half)
    pub fn card_bytes(&self) -> u64 {
        match self {
            Source::Gguf(f) => f.tensors.iter().map(|t| if t.ty.code() == 0 { t.bytes / 2 } else { t.bytes }).sum(),
            Source::St(s) => s.tensors.iter().map(|t| if t.dtype == "F32" { t.bytes / 2 } else { t.bytes }).sum(),
        }
    }

    /// An int8 matrix's scales (float32, one per row) after checking it is ConvRot in groups of 256
    fn int8_scales(&self, name: &str) -> Result<Vec<f32>> {
        let Source::St(s) = self else { return Err(Error(format!("{name}: int8 outside a safetensors file"))) };
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let meta = s.read(s.need(&format!("{base}.comfy_quant")).map_err(ge)?).map_err(ge)?;
        let meta = String::from_utf8_lossy(&meta).replace(' ', "");
        if !meta.contains("\"format\":\"int8_tensorwise\"") || !meta.contains("\"convrot\":true") || !meta.contains(&format!("\"convrot_groupsize\":{GROUP}")) {
            return Err(Error(format!("{name}: {meta}, not int8 ConvRot in groups of {GROUP}")));
        }
        s.f32(&format!("{name}_scale")).map_err(ge)
    }
}

/// v . H in place for each group of 256 of v, H the matrix `hadamard(256)` makes, as four 4-point passes (H is
/// H4 (x) H4 (x) H4 (x) H4 / 16, digit by digit in base 4)
fn hadamard_rows(v: &mut [f32]) {
    const H4: [[f32; 4]; 4] = [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]];
    for g in v.chunks_exact_mut(GROUP) {
        let mut stride = 1;
        while stride < GROUP {
            for base in 0..GROUP {
                if (base / stride) % 4 != 0 {
                    continue;
                }
                let x = [g[base], g[base + stride], g[base + 2 * stride], g[base + 3 * stride]];
                for (j, col) in (0..4).map(|j| (j, base + j * stride)) {
                    g[col] = (0..4).map(|i| x[i] * H4[i][j]).sum();
                }
            }
            stride *= 4;
        }
        g.iter_mut().for_each(|x| *x /= 16.0);
    }
}

/// The normalized regular Hadamard matrix of size g: H4 (x) H4 (x) ... / sqrt(g) (comfy-kitchen's ConvRot; symmetric
/// and its own inverse)
#[cfg(test)]
fn hadamard(g: usize) -> Vec<f32> {
    const H4: [[i32; 4]; 4] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];
    let norm = 1.0 / (g as f32).sqrt();
    let mut h = vec![0f32; g * g];
    for i in 0..g {
        for j in 0..g {
            let (mut a, mut b, mut s, mut sign) = (i, j, 1, 1);
            while s < g {
                sign *= H4[a % 4][b % 4];
                a /= 4;
                b /= 4;
                s *= 4;
            }
            h[i * g + j] = sign as f32 * norm;
        }
    }
    h
}

/// A matrix on the card: half (made from a float file's), as stored (GGUF-quantized), or int8 ConvRot with a scale per
/// row (the activations quantized per row on the fly: the card's int8 rate)
pub(crate) enum Weights {
    Half(DevBuf),
    Stored { w: DevBuf, qtype: u32 },
    Int8 { w: DevBuf, s: DevBuf },
}

/// A matrix of `rows` outputs over `k` inputs
pub(crate) struct Part {
    pub(crate) w: Weights,
    pub(crate) rows: usize,
    pub(crate) k: usize,
    /// the factor folded into the matrix (its outputs come scaled by it)
    pub(crate) scale: f32,
}

impl Part {
    /// Whether its outputs come scaled (a factor folded into the half matrix at load)
    pub(crate) fn scaled(&self) -> bool {
        self.scale != 1.0
    }

    /// out [m, rows] = x [m, k] (half) . W^T + bias
    pub(crate) fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, bias: Option<&DevBuf>, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let b = bias.map_or(none(), |b| b.ptr().cast_const());
        match &self.w {
            Weights::Half(h) => nsd.linear(x, Dt::F16, m, self.k, h.ptr(), self.rows, b, out, out_dt),
            Weights::Stored { w, qtype } => nsd.linear_q(x, Dt::F16, m, self.k, w.ptr(), *qtype, self.rows, b, out, out_dt, 0),
            Weights::Int8 { w, s } => nsd.int8_linear(x, Dt::F16, m, self.k, w.ptr(), self.rows, s.ptr(), self.rows, b, out, out_dt, GROUP),
        }
    }

    /// Bytes on the card
    pub(crate) fn bytes(&self) -> u64 {
        match &self.w {
            Weights::Half(h) | Weights::Stored { w: h, .. } => h.len as u64,
            Weights::Int8 { w, s } => (w.len + s.len) as u64,
        }
    }
}

/// Matrices from `f` stacked by rows (q | k | v, w1 | w3: one product each) onto the card; `perm`: rows within each
/// head of the first `perm` rows reordered for the shared RoPE kernel; `scale`: folded into a float matrix when it is
/// made half (a power of two: exact but for values that fall into half's subnormals) or into an int8 one's row scales;
/// a GGUF-quantized one is kept as stored
pub(crate) fn part(f: &Source, nsd: &Nsd, names: &[String], perm: usize, scale: f32) -> Result<Part> {
    let gpu = &nsd.gpu;
    let k = ffi::api()?;
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    let (mut rows, mut kk, mut ty) = (0usize, 0usize, None);
    for name in names {
        let (d, shape) = f.info(name)?;
        if let Dty::G(g) = d {
            if ![0, 1, 8, 12, 13, 14, 30].contains(&g.code()) {
                return Err(Error(format!("{name}: {} weights are not read yet", g.name())));
            }
        }
        if ty.is_some_and(|x| x != d) || (kk != 0 && kk != shape[1] as usize) {
            return Err(Error(format!("{name}: its type or width differs from the matrix it is stacked with")));
        }
        ty = Some(d);
        kk = shape[1] as usize;
        rows += shape[0] as usize;
        bytes.extend(f.read(name)?);
        if d == Dty::I8 {
            scales.extend(f.int8_scales(name)?);
        }
    }
    let ty = ty.ok_or_else(|| Error("no matrix named".into()))?;
    if perm > 0 {
        let rb = bytes.len() / rows;
        let (mut out, mut sout) = (bytes.clone(), scales.clone());
        for r in 0..perm {
            let (h, j) = (r / HEAD, r % HEAD);
            let src = h * HEAD + rope_perm(j);
            out[r * rb..(r + 1) * rb].copy_from_slice(&bytes[src * rb..(src + 1) * rb]);
            if !scales.is_empty() {
                sout[r] = scales[src];
            }
        }
        bytes = out;
        scales = sout;
    }
    let stored = DevBuf::new(gpu, bytes.len())?;
    stored.write(0, &bytes)?;
    let n = rows * kk;
    let ty = match ty {
        Dty::G(g) => g,
        Dty::I8 => {
            if !kk.is_multiple_of(GROUP) || scales.len() != rows {
                return Err(Error(format!("{}: int8 rows of {kk} (not a multiple of {GROUP}) or {} scales for {rows} rows", names[0], scales.len())));
            }
            let s: Vec<f32> = scales.iter().map(|v| v * scale).collect();
            return Ok(Part { w: Weights::Int8 { w: stored, s: DevBuf::from_f32(gpu, &s)? }, rows, k: kk, scale });
        }
    };
    let w = match ty {
        GType::F16 if scale == 1.0 => Weights::Half(stored),
        GType::BF16 if scale == 1.0 => {
            let h = DevBuf::new(gpu, n * 2)?;
            nsd.dequant(stored.ptr(), 30, n, h.ptr(), Dt::F16)?;
            nsd.wait()?;
            Weights::Half(h)
        }
        GType::F32 | GType::F16 | GType::BF16 => {
            // float32 first (scaled there), then half
            let f = match ty {
                GType::F32 => stored,
                GType::F16 => {
                    let f = DevBuf::f32(gpu, n)?;
                    // SAFETY: n halves in, n floats out.
                    ffi::check(unsafe { (k.to_float)(gpu.raw(), stored.ptr(), f.fp(), n as i64) }, "to float")?;
                    f
                }
                _ => {
                    let f = DevBuf::f32(gpu, n)?;
                    nsd.dequant(stored.ptr(), 30, n, f.ptr(), Dt::F32)?;
                    f
                }
            };
            if scale != 1.0 {
                nsd.scale(f.ptr(), n, scale)?;
            }
            let h = DevBuf::new(gpu, n * 2)?;
            // SAFETY: n floats in, n halves out.
            ffi::check(unsafe { (k.to_half)(gpu.raw(), f.fp(), h.ptr(), n as i64) }, "to half")?;
            nsd.wait()?;
            Weights::Half(h)
        }
        q => Weights::Stored { w: stored, qtype: q.code() },
    };
    let scale = if matches!(w, Weights::Half(_)) { scale } else { 1.0 };
    Ok(Part { w, rows, k: kk, scale })
}

/// A float vector or matrix (any stored type) as float32 on the host; GGUF-quantized ones expanded on the card, int8
/// ConvRot ones on the host (q s, un-rotated)
pub(crate) fn host_vec(f: &Source, nsd: &Nsd, name: &str) -> Result<Vec<f32>> {
    let (d, shape) = f.info(name)?;
    let b = f.read(name)?;
    let g = match d {
        Dty::G(g) => g,
        Dty::I8 => {
            let s = f.int8_scales(name)?;
            let k = shape.get(1).copied().unwrap_or(1) as usize;
            if !k.is_multiple_of(GROUP) {
                return Err(Error(format!("{name}: int8 rows of {k}, not a multiple of {GROUP}")));
            }
            // w = (q s) . H over each group (the file's rows are w . H, and H is its own inverse)
            let mut out: Vec<f32> = b.chunks_exact(k).zip(&s).flat_map(|(row, sc)| row.iter().map(move |q| *q as i8 as f32 * sc)).collect();
            hadamard_rows(&mut out);
            return Ok(out);
        }
    };
    Ok(match g {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        q => {
            let n: usize = shape.iter().product::<u64>() as usize;
            let src = DevBuf::new(&nsd.gpu, b.len())?;
            src.write(0, &b)?;
            let out = DevBuf::f32(&nsd.gpu, n)?;
            nsd.dequant(src.ptr(), q.code(), n, out.ptr(), Dt::F32)?;
            nsd.wait()?;
            out.to_f32()?
        }
    })
}

struct Block {
    qkv: Part,
    o: Part,
    w13: Part,
    w2: Part,
    norm_q: DevBuf,
    norm_k: DevBuf,
    n1: DevBuf,
    n2: DevBuf,
    f1: DevBuf,
    f2: DevBuf,
    /// the adaLN linear float32 [4 * 3840, 256] and its bias (the modulated blocks)
    ada: Option<(DevBuf, DevBuf)>,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    x_w: DevBuf,
    x_b: DevBuf,
    x_pad: Vec<f32>,
    cap_norm: DevBuf,
    cap_w: Part,
    cap_b: DevBuf,
    cap_pad: Vec<f32>,
    t1: (Vec<f32>, Vec<f32>),
    t2: (Vec<f32>, Vec<f32>),
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    layers: Vec<Block>,
    /// the final layer: its adaLN (after a SiLU) on the host, its linear float32 on the card
    final_ada: (Vec<f32>, Vec<f32>),
    final_w: DevBuf,
    final_b: DevBuf,
    /// bytes of weights on the card
    pub bytes: u64,
    /// the silo's fused passes, when built and this GPU runs them at the DiT's sizes (None: this engine's own kernels)
    silo: Option<&'static ffi::Silo>,
}

/// Where a block sits in its sequence: its modulation, whether its input norm is already in the pass's half buffer (the
/// block before made it), and the block after it in the same sequence (whose input norm its last pass makes)
struct At<'a> {
    md: Option<&'a Mod>,
    normed: bool,
    next: Option<(&'a Block, Option<&'a Mod>)>,
}

/// One block's modulation for a step: scale1, tanh(gate1), scale2, tanh(gate2), float32 [3840] each
pub struct Mod {
    s1: DevBuf,
    g1: DevBuf,
    s2: DevBuf,
    g2: DevBuf,
}

/// A step's modulations: the noise refiner's, the layers', the final layer's scale
pub struct StepMods {
    nr: Vec<Mod>,
    layers: Vec<Mod>,
    final_scale: DevBuf,
}

/// The caption's rows after the context refiner (float32 [L', 3840], the pad token's rows included) and their RoPE
/// table, once a prompt
pub struct Prefix {
    pub rows: usize,
    pub x: DevBuf,
    cs: Vec<f32>,
}

/// Intermediates kept for a check
#[derive(Default)]
pub struct Grabs {
    /// set: a GPU wait after each part of every block, the seconds by part in `prof` (where a forward's time goes)
    pub profile: bool,
    pub prof: Vec<(&'static str, f64)>,
    pub x_emb: Vec<f32>,
    pub nr_out: Vec<f32>,
    pub cap_emb: Vec<f32>,
    pub cr_out: Vec<f32>,
    pub l0_out: Vec<f32>,
    pub l29_out: Vec<f32>,
}

/// q / k row order: pair (2i, 2i + 1) -> (i, 64 + i)
fn rope_perm(j: usize) -> usize {
    if j < HEAD / 2 {
        2 * j
    } else {
        2 * (j - HEAD / 2) + 1
    }
}

/// (cos, sin) per token and pair for positions (frame, height, width), in the permuted order; as the reference's table
/// (float64 frequencies times the position, made float32 before the cosine)
fn rope_table(idx: &[[i64; 3]]) -> Vec<f32> {
    let mut cs = vec![0f32; idx.len() * HEAD];
    for (t, p) in idx.iter().enumerate() {
        let mut pair = 0;
        for (axis, dim) in AXES.iter().enumerate() {
            for i in 0..dim / 2 {
                let inv = 1.0 / THETA.powf(2.0 * i as f64 / *dim as f64);
                let a = (p[axis] as f64 * inv) as f32;
                cs[t * HEAD + 2 * pair] = a.cos();
                cs[t * HEAD + 2 * pair + 1] = a.sin();
                pair += 1;
            }
        }
    }
    cs
}

/// y = W x + b on the host (W [n, k] row-major)
fn host_linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * (*c as f64)).sum::<f64>() as f32 + bi).collect()
}

fn silu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
}

/// The timestep the model is given at level `s`, as the pipeline computes it in float32: (1000 - 1000 s) / 1000
pub fn model_t(s: f32) -> f32 {
    (1000.0 - s * 1000.0) / 1000.0
}

/// The sinusoid of 256 for timestep `t` (TimestepEmbedder.timestep_embedding of t * t_scale): [cos | sin], float32 as
/// torch computes it
pub fn sinusoid(t: f32) -> Vec<f32> {
    let half = ADALN / 2;
    let t = t * T_SCALE;
    let mut e = vec![0f32; ADALN];
    for i in 0..half {
        let freq = (-(10000f64.ln() as f32) * i as f32 / half as f32).exp();
        let a = t * freq;
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e
}

impl Dit {
    pub fn load(f: &Source, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        if !f.has("x_embedder.weight") || !f.has("layers.0.attention.qkv.weight") {
            return Err(Error(format!("{}: not Z-Image's DiT in ComfyUI's layout (no x_embedder / layers.0.attention.qkv)", f.path().display())));
        }
        let vecd = |name: &str, permute: bool| -> Result<DevBuf> {
            let v = host_vec(f, nsd, name)?;
            let v = if permute { (0..v.len()).map(|j| v[rope_perm(j)]).collect() } else { v };
            DevBuf::from_f32(&gpu, &v)
        };
        let block = |p: &str, modulated: bool| -> Result<Block> {
            Ok(Block {
                qkv: part(f, nsd, &[format!("{p}attention.qkv.weight")], 2 * DIM, ffi::QKV_SCALE)?,
                o: part(f, nsd, &[format!("{p}attention.out.weight")], 0, 1.0)?,
                w13: part(f, nsd, &[format!("{p}feed_forward.w1.weight"), format!("{p}feed_forward.w3.weight")], 0, 1.0)?,
                w2: part(f, nsd, &[format!("{p}feed_forward.w2.weight")], 0, 1.0)?,
                norm_q: vecd(&format!("{p}attention.q_norm.weight"), true)?,
                norm_k: vecd(&format!("{p}attention.k_norm.weight"), true)?,
                n1: vecd(&format!("{p}attention_norm1.weight"), false)?,
                n2: vecd(&format!("{p}attention_norm2.weight"), false)?,
                f1: vecd(&format!("{p}ffn_norm1.weight"), false)?,
                f2: vecd(&format!("{p}ffn_norm2.weight"), false)?,
                ada: if modulated {
                    Some((vecd(&format!("{p}adaLN_modulation.0.weight"), false)?, vecd(&format!("{p}adaLN_modulation.0.bias"), false)?))
                } else {
                    None
                },
            })
        };
        let count = |pre: &str| (0..).take_while(|i| f.has(&format!("{pre}{i}.attention.qkv.weight"))).count();
        let (nl, nr, nc) = (count("layers."), count("noise_refiner."), count("context_refiner."));
        let mut dit = Dit {
            gpu: gpu.clone(),
            x_w: vecd("x_embedder.weight", false)?,
            x_b: vecd("x_embedder.bias", false)?,
            x_pad: host_vec(f, nsd, "x_pad_token")?,
            cap_norm: vecd("cap_embedder.0.weight", false)?,
            cap_w: part(f, nsd, &["cap_embedder.1.weight".to_string()], 0, 1.0)?,
            cap_b: vecd("cap_embedder.1.bias", false)?,
            cap_pad: host_vec(f, nsd, "cap_pad_token")?,
            t1: (host_vec(f, nsd, "t_embedder.mlp.0.weight")?, host_vec(f, nsd, "t_embedder.mlp.0.bias")?),
            t2: (host_vec(f, nsd, "t_embedder.mlp.2.weight")?, host_vec(f, nsd, "t_embedder.mlp.2.bias")?),
            noise_refiner: (0..nr).map(|i| block(&format!("noise_refiner.{i}."), true)).collect::<Result<_>>()?,
            context_refiner: (0..nc).map(|i| block(&format!("context_refiner.{i}."), false)).collect::<Result<_>>()?,
            layers: (0..nl).map(|i| block(&format!("layers.{i}."), true)).collect::<Result<_>>()?,
            final_ada: (host_vec(f, nsd, "final_layer.adaLN_modulation.1.weight")?, host_vec(f, nsd, "final_layer.adaLN_modulation.1.bias")?),
            final_w: vecd("final_layer.linear.weight", false)?,
            final_b: vecd("final_layer.linear.bias", false)?,
            bytes: 0,
            // SAFETY: a query of the GPU's sub-group sizes.
            silo: ffi::silo()?.filter(|s| unsafe { (s.supported)(gpu.raw(), DIM as i64, HEAD as i64) } == 1),
        };
        if dit.cap_w.k != CAP || dit.x_pad.len() != DIM || dit.cap_pad.len() != DIM {
            return Err(Error(format!("{}: caption features {} (Z-Image's: {CAP}), pad tokens {} / {}", f.path().display(), dit.cap_w.k, dit.x_pad.len(),
                                     dit.cap_pad.len())));
        }
        dit.bytes = dit.noise_refiner.iter().chain(&dit.context_refiner).chain(&dit.layers).map(|b| b.qkv.bytes() + b.o.bytes() + b.w13.bytes() + b.w2.bytes()).sum::<u64>()
            + dit.cap_w.bytes();
        log(format!("z-image dit: {nl} blocks, {nr} + {nc} refiners, {:.1} GiB on {} in {:.1} s ({})", dit.bytes as f64 / (1u64 << 30) as f64, gpu.name,
                    t0.elapsed().as_secs_f64(), if dit.silo.is_some() { "the silo's fused passes" } else { "no silo: this engine's own kernels" }));
        Ok(dit)
    }

    /// The blocks of a sequence with where each sits (the silo's passes norm the next block's input at the end of one)
    fn walk<'a>(&self, blocks: &'a [Block], mods: Option<&'a [Mod]>) -> Vec<(&'a Block, At<'a>)> {
        let md = |i: usize| mods.map(|m| &m[i]);
        (0..blocks.len()).map(|i| {
            let fused = self.silo.is_some();
            (&blocks[i], At { md: md(i), normed: fused && i > 0, next: if fused { blocks.get(i + 1).map(|b| (b, md(i + 1))) } else { None } })
        }).collect()
    }

    /// The timestep's embedding at level `s`: the sinusoid, linear, SiLU, linear
    pub fn t_emb(&self, s: f32) -> Vec<f32> {
        let mut h = host_linear(&self.t1.0, &self.t1.1, &sinusoid(model_t(s)));
        silu(&mut h);
        host_linear(&self.t2.0, &self.t2.1, &h)
    }

    /// Each step's modulations for the levels `sigmas` (the steps run: all but the last level), one product a step
    /// (one over all the steps' rows sums in a different order from call to call)
    pub fn mods(&self, nsd: &Nsd, sigmas: &[f32]) -> Result<Vec<StepMods>> {
        let gpu = &self.gpu;
        let steps = sigmas.len().saturating_sub(1);
        let tanh = |x: &[f32]| x.iter().map(|v| v.tanh()).collect::<Vec<f32>>();
        let out_b = DevBuf::f32(gpu, 4 * DIM)?;
        let mut out = Vec::with_capacity(steps);
        for s in &sigmas[..steps] {
            let emb = self.t_emb(*s);
            let e = DevBuf::from_f32(gpu, &emb)?;
            let one = |b: &Block| -> Result<Mod> {
                let (w, bias) = b.ada.as_ref().ok_or_else(|| Error("an unmodulated block".into()))?;
                nsd.linear(e.ptr(), Dt::F32, 1, ADALN, w.ptr(), 4 * DIM, bias.fp().cast(), out_b.ptr(), Dt::F32)?;
                let r = out_b.to_f32()?;
                Ok(Mod {
                    s1: DevBuf::from_f32(gpu, &r[..DIM])?,
                    g1: DevBuf::from_f32(gpu, &tanh(&r[DIM..2 * DIM]))?,
                    s2: DevBuf::from_f32(gpu, &r[2 * DIM..3 * DIM])?,
                    g2: DevBuf::from_f32(gpu, &tanh(&r[3 * DIM..]))?,
                })
            };
            let mut se = emb.clone();
            silu(&mut se);
            let fs = host_linear(&self.final_ada.0, &self.final_ada.1, &se);
            out.push(StepMods {
                nr: self.noise_refiner.iter().map(one).collect::<Result<_>>()?,
                layers: self.layers.iter().map(one).collect::<Result<_>>()?,
                final_scale: DevBuf::from_f32(gpu, &fs)?,
            });
        }
        Ok(out)
    }

    /// One block over x float32 [m, 3840] in place, at `at` in its sequence (None modulation: an unmodulated refiner).
    /// With the silo, its residual steps are one fused pass each and the last also norms the next block's input
    #[allow(clippy::too_many_arguments)]
    fn block(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, cs: &DevBuf, at: &At, t: &Tmp) -> Result<()> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let qw = 3 * DIM;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        let nul = std::ptr::null::<f32>();
        let mods = |md: Option<&Mod>| match md {
            Some(v) => (v.s1.fp().cast_const(), v.g1.fp().cast_const(), v.s2.fp().cast_const(), v.g2.fp().cast_const()),
            None => (nul, nul, nul, nul),
        };
        let (s1, g1, s2, g2) = mods(at.md);
        if !at.normed {
            // SAFETY: x holds m x 3840 floats, t.h as many halves; the weights and scales 3840 floats.
            ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), m as i64, DIM as i64, b.n1.fp(), EPS, s1, t.h.ptr()) }, "norm")?;
            t.lap(nsd, "norms and gates")?;
        }
        // q | k | v half at QKV_SCALE (folded into a float file's matrix at load, else the float32 product scaled into
        // half): q and k are rms-normed after (their scale goes), v's comes back through the attention and the output
        // linear, and the norm after that (attention_norm2) is blind to it too - only the epsilons take the scale
        if b.qkv.scaled() {
            b.qkv.apply(nsd, t.h.ptr(), m, None, t.qkv.ptr(), Dt::F16)?;
            t.lap(nsd, "q k v product")?;
        } else {
            b.qkv.apply(nsd, t.h.ptr(), m, None, t.of.ptr(), Dt::F32)?;
            t.lap(nsd, "q k v product")?;
            // SAFETY: of holds m x qw floats, qkv as many halves.
            ffi::check(unsafe { (k.to_half_rows)(gpu.raw(), t.of.fp(), m as i64, qw as i64, ffi::QKV_SCALE, t.qkv.ptr(), qw as i64) }, "q k v")?;
            t.lap(nsd, "q k v to half")?;
        }
        // (the norm of q s with eps s^2 is the norm of q with eps: the scale is exact)
        let eps_qkv = EPS * ffi::QKV_SCALE * ffi::QKV_SCALE;
        match self.silo {
            // SAFETY: qkv holds m rows of qw halves; the norms 128 floats, cs m x 128.
            Some(sl) => ffi::check(unsafe { (sl.qk_rope)(gpu.raw(), t.qkv.ptr(), m as i64, HEADS as i64, HEAD as i64, qw as i64, DIM as i64, b.norm_q.fp(), b.norm_k.fp(),
                                                         eps_qkv, cs.fp()) }, "q k norm + rope")?,
            None => {
                nsd.rms_rope(t.qkv.ptr(), Dt::F16, m, HEADS, HEAD, qw, b.norm_q.ptr(), eps_qkv, cs.ptr(), HEAD)?;
                nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, m, HEADS, HEAD, qw, b.norm_k.ptr(), eps_qkv, cs.ptr(), HEAD)?;
            }
        }
        t.lap(nsd, "q k norm + rope")?;
        nsd.attention_qk(t.qkv.ptr(), m, qw, half_at(&t.qkv, DIM), half_at(&t.qkv, 2 * DIM), m, qw, Dt::F16, HEADS, HEAD, t.att.ptr(), Dt::F16)?;
        t.lap(nsd, "attention")?;
        b.o.apply(nsd, t.att.ptr(), m, None, t.of.ptr(), Dt::F32)?;
        t.lap(nsd, "out product")?;
        match self.silo {
            // SAFETY: of and x hold m x 3840 floats, t.h as many halves; the weights and scales 3840 floats.
            Some(sl) => ffi::check(unsafe { (sl.norm_gate_norm)(gpu.raw(), t.of.fp(), x.fp(), m as i64, DIM as i64, b.n2.fp(), eps_qkv, g1, b.f1.fp(), EPS, s2,
                                                                t.h.ptr()) }, "norm, gate, norm")?,
            None => {
                // SAFETY: as above.
                ffi::check(unsafe { (k.rms_mod)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, b.n2.fp(), eps_qkv, std::ptr::null(), t.h.ptr()) }, "norm")?;
                ffi::check(unsafe { (k.gate_add)(gpu.raw(), x.fp(), m as i64, DIM as i64, t.h.ptr(), g1) }, "gate")?;
                ffi::check(unsafe { (k.rms_mod)(gpu.raw(), x.fp(), m as i64, DIM as i64, b.f1.fp(), EPS, s2, t.h.ptr()) }, "norm")?;
            }
        }
        t.lap(nsd, "norms and gates")?;
        b.w13.apply(nsd, t.h.ptr(), m, None, t.gp.ptr(), Dt::F16)?;
        t.lap(nsd, "w1 | w3 product")?;
        match self.silo {
            // SAFETY: gp holds m x 2 FFN halves, act m x FFN.
            Some(sl) => ffi::check(unsafe { (sl.swiglu)(gpu.raw(), t.gp.ptr(), m as i64, FFN as i64, ffi::ACT_SCALE, t.act.ptr()) }, "swiglu")?,
            None => ffi::check(unsafe { (k.swiglu_scaled)(gpu.raw(), t.gp.ptr(), m as i64, FFN as i64, ffi::ACT_SCALE, t.act.ptr()) }, "swiglu")?,
        }
        t.lap(nsd, "swiglu")?;
        b.w2.apply(nsd, t.act.ptr(), m, None, t.of.ptr(), Dt::F32)?;
        t.lap(nsd, "w2 product")?;
        let eps_act = EPS * ffi::ACT_SCALE * ffi::ACT_SCALE;
        match self.silo {
            Some(sl) => {
                // the next block's input norm in the same pass, when there is a next block in this sequence
                let (wb, sb, out) = match at.next {
                    Some((nb, nmd)) => (nb.n1.fp().cast_const(), mods(nmd).0, t.h.ptr()),
                    None => (nul, nul, std::ptr::null_mut()),
                };
                // SAFETY: as above; out NULL: x alone.
                ffi::check(unsafe { (sl.norm_gate_norm)(gpu.raw(), t.of.fp(), x.fp(), m as i64, DIM as i64, b.f2.fp(), eps_act, g2, wb, EPS, sb, out) },
                           "norm, gate, norm")?;
            }
            None => {
                // SAFETY: as above.
                ffi::check(unsafe { (k.rms_mod)(gpu.raw(), t.of.fp(), m as i64, DIM as i64, b.f2.fp(), eps_act, std::ptr::null(), t.h.ptr()) }, "norm")?;
                ffi::check(unsafe { (k.gate_add)(gpu.raw(), x.fp(), m as i64, DIM as i64, t.h.ptr(), g2) }, "gate")?;
            }
        }
        t.lap(nsd, "norms and gates")
    }

    /// Rows `from..to` of x float32 [., 3840] set to `v` (the pad token)
    fn pad_rows(x: &DevBuf, from: usize, to: usize, v: &[f32]) -> Result<()> {
        if to > from {
            let rows: Vec<f32> = v.iter().copied().cycle().take((to - from) * DIM).collect();
            let bytes: Vec<u8> = rows.iter().flat_map(|f| f.to_le_bytes()).collect();
            x.write(from * DIM * 4, &bytes)?;
        }
        Ok(())
    }

    /// The caption's rows: Qwen3-4B's states (float32 [L, 2560]) embedded, padded to L' with the pad token, through the
    /// context refiner
    pub fn prefix(&self, nsd: &Nsd, cap: &[f32], grabs: Option<&mut Grabs>) -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let l = cap.len() / CAP;
        if l == 0 {
            return Err(Error("an empty caption".into()));
        }
        let c = padded(l);
        let cf = DevBuf::from_f32(gpu, cap)?;
        let ch = DevBuf::new(gpu, cap.len() * 2)?;
        // SAFETY: l x 2560 values; the weight 2560 floats.
        ffi::check(unsafe { (k.rms_mod)(gpu.raw(), cf.fp(), l as i64, CAP as i64, self.cap_norm.fp(), EPS, std::ptr::null(), ch.ptr()) }, "norm")?;
        let x = DevBuf::f32(gpu, c * DIM)?;
        self.cap_w.apply(nsd, ch.ptr(), l, Some(&self.cap_b), x.ptr(), Dt::F32)?;
        nsd.wait()?;
        Self::pad_rows(&x, l, c, &self.cap_pad)?;
        // every caption row at (1 + j, 0, 0), its padding too: the reference's grid for the caption is its padded length
        // (_pad_with_ids then appends origin rows for the padding, which the table's cut to the rows drops)
        let idx: Vec<[i64; 3]> = (0..c).map(|j| [1 + j as i64, 0, 0]).collect();
        let cs_host = rope_table(&idx);
        let cs = DevBuf::from_f32(gpu, &cs_host)?;
        let t = Tmp::new(gpu, c)?;
        let mut grabs = grabs;
        if let Some(g) = grabs.as_deref_mut() {
            g.cap_emb = x.to_f32()?;
        }
        for (b, at) in self.walk(&self.context_refiner, None) {
            self.block(nsd, b, &x, c, &cs, &at, &t)?;
        }
        nsd.wait()?;
        if let Some(g) = grabs {
            g.cr_out = x.to_f32()?;
        }
        Ok(Prefix { rows: c, x, cs: cs_host })
    }

    /// The model's output for the image's patches `lat` (float32 [hh * ww, 64], row-major patches of (2, 2, 16)) at a
    /// step with modulations `md`: float32 [hh * ww, 64] (the reference's, before the pipeline negates it)
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), md: &StepMods, mut grabs: Option<&mut Grabs>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let n = hw.0 * hw.1;
        let np = padded(n);
        let c = pre.rows;
        let s = np + c;
        let x = DevBuf::f32(gpu, s * DIM)?;
        nsd.linear(lat.ptr(), Dt::F32, n, PATCH_IN, self.x_w.ptr(), DIM, self.x_b.fp().cast(), x.ptr(), Dt::F32)?;
        nsd.wait()?;
        Self::pad_rows(&x, n, np, &self.x_pad)?;
        // positions: the image at the frame after the caption, rows and columns from 0; its padding at the origin
        let frame = (c + 1) as i64;
        let idx: Vec<[i64; 3]> = (0..np).map(|i| if i < n { [frame, (i / hw.1) as i64, (i % hw.1) as i64] } else { [0, 0, 0] }).collect();
        let mut cs_all = rope_table(&idx);
        let cs_img = DevBuf::from_f32(gpu, &cs_all)?;
        let t = Tmp::new(gpu, s)?;
        if let Some(g) = grabs.as_deref_mut() {
            g.x_emb = x.to_f32()?[..np * DIM].to_vec();
            if g.profile {
                nsd.wait()?;
                *t.prof.borrow_mut() = Some((std::time::Instant::now(), Vec::new()));
            }
        }
        for (b, at) in self.walk(&self.noise_refiner, Some(&md.nr)) {
            self.block(nsd, b, &x, np, &cs_img, &at, &t)?;
        }
        if let Some(g) = grabs.as_deref_mut() {
            nsd.wait()?;
            g.nr_out = x.to_f32()?[..np * DIM].to_vec();
        }
        // [image | caption]
        x.copy_within(np * DIM * 4, &pre.x, 0, c * DIM * 4)?;
        cs_all.extend_from_slice(&pre.cs);
        let cs = DevBuf::from_f32(gpu, &cs_all)?;
        let last = self.layers.len().saturating_sub(1);
        for (i, (b, at)) in self.walk(&self.layers, Some(&md.layers)).into_iter().enumerate() {
            self.block(nsd, b, &x, s, &cs, &at, &t)?;
            if let Some(g) = grabs.as_deref_mut() {
                if i == 0 || i == last {
                    nsd.wait()?;
                    let v = x.to_f32()?;
                    if i == 0 {
                        g.l0_out = v;
                    } else {
                        g.l29_out = v;
                    }
                }
            }
        }
        // the final layer on the image's rows
        // SAFETY: n x 3840 values each way; the scale 3840 floats.
        ffi::check(unsafe { (k.ln_mod)(gpu.raw(), x.fp(), n as i64, DIM as i64, 1e-6, md.final_scale.fp(), t.h.ptr()) }, "final norm")?;
        let hf = DevBuf::f32(gpu, n * DIM)?;
        ffi::check(unsafe { (k.to_float)(gpu.raw(), t.h.ptr(), hf.fp(), (n * DIM) as i64) }, "to float")?;
        let v = DevBuf::f32(gpu, n * PATCH_IN)?;
        nsd.linear(hf.ptr(), Dt::F32, n, DIM, self.final_w.ptr(), PATCH_IN, self.final_b.fp().cast(), v.ptr(), Dt::F32)?;
        nsd.wait()?;
        if let (Some(g), Some((_, acc))) = (grabs, t.prof.borrow_mut().take()) {
            g.prof = acc;
        }
        Ok(v)
    }

    /// Each block product alone at `rows` tokens, warm, timed over a few runs: (what, seconds a product, its FLOPs)
    pub fn gemm_probe(&self, nsd: &Nsd, rows: usize) -> Result<Vec<(&'static str, f64, f64)>> {
        let b = &self.layers[0];
        let x = DevBuf::new(&self.gpu, rows * FFN * 2)?;
        x.fill(0)?;
        let out = DevBuf::f32(&self.gpu, rows * 2 * FFN)?;
        let mut res = Vec::new();
        for (what, p, dt) in [("q | k | v", &b.qkv, Dt::F32), ("out", &b.o, Dt::F32), ("w1 | w3", &b.w13, Dt::F16), ("w2", &b.w2, Dt::F32)] {
            p.apply(nsd, x.ptr(), rows, None, out.ptr(), dt)?;
            nsd.wait()?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                p.apply(nsd, x.ptr(), rows, None, out.ptr(), dt)?;
            }
            nsd.wait()?;
            res.push((what, t0.elapsed().as_secs_f64() / 5.0, 2.0 * (rows * p.rows * p.k) as f64));
        }
        Ok(res)
    }
}

/// A profile in progress: the clock since the last lap, the seconds by part
type Prof = (std::time::Instant, Vec<(&'static str, f64)>);

/// A pass's buffers, for up to `rows` tokens
struct Tmp {
    /// when profiling: the clock and the seconds by part
    prof: std::cell::RefCell<Option<Prof>>,
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    of: DevBuf,
    gp: DevBuf,
    act: DevBuf,
}

impl Tmp {
    /// When profiling: the GPU waited for, the seconds since the last lap added to `name`'s
    fn lap(&self, nsd: &Nsd, name: &'static str) -> Result<()> {
        if let Some((t0, acc)) = self.prof.borrow_mut().as_mut() {
            nsd.wait()?;
            let s = t0.elapsed().as_secs_f64();
            *t0 = std::time::Instant::now();
            match acc.iter_mut().find(|(n, _)| *n == name) {
                Some(e) => e.1 += s,
                None => acc.push((name, s)),
            }
        }
        Ok(())
    }

    fn new(gpu: &Arc<Gpu>, rows: usize) -> Result<Tmp> {
        Ok(Tmp {
            prof: std::cell::RefCell::new(None),
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            of: DevBuf::f32(gpu, rows * 3 * DIM)?,
            gp: DevBuf::new(gpu, rows * 2 * FFN * 2)?,
            act: DevBuf::new(gpu, rows * FFN * 2)?,
        })
    }
}

/// Bytes of a pass's buffers for `rows` tokens (the memory plan)
pub fn work_bytes(rows: usize) -> u64 {
    (rows * (DIM * 2 + 3 * DIM * 2 + DIM * 2 + 3 * DIM * 4 + 2 * FFN * 2 + FFN * 2 + 2 * DIM * 4)) as u64
}

/// The latents [16, H, W] (H, W even) as the DiT's patches [H/2 * W/2, 64]: feature (ph * 2 + pw) * 16 + c
pub fn patchify(lat: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let mut out = vec![0f32; hh * ww * PATCH_IN];
    for y in 0..hh {
        for x in 0..ww {
            for ph in 0..2 {
                for pw in 0..2 {
                    for c in 0..LATENT {
                        out[(y * ww + x) * PATCH_IN + (ph * 2 + pw) * LATENT + c] = lat[c * h * w + (2 * y + ph) * w + 2 * x + pw];
                    }
                }
            }
        }
    }
    out
}

/// The inverse: patches [H/2 * W/2, 64] to latents [16, H, W]
pub fn unpatchify(p: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (h / 2, w / 2);
    let mut out = vec![0f32; LATENT * h * w];
    for y in 0..hh {
        for x in 0..ww {
            for ph in 0..2 {
                for pw in 0..2 {
                    for c in 0..LATENT {
                        out[c * h * w + (2 * y + ph) * w + 2 * x + pw] = p[(y * ww + x) * PATCH_IN + (ph * 2 + pw) * LATENT + c];
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn patches_go_and_come_back() {
        let (h, w) = (6, 4);
        let lat: Vec<f32> = (0..16 * h * w).map(|i| i as f32).collect();
        let p = super::patchify(&lat, h, w);
        assert_eq!(p.len(), 3 * 2 * 64);
        // patch (0, 0): (ph 0, pw 1, c 0) is pixel (0, 1) of channel 0
        assert_eq!(p[16], 1.0);
        // (ph 1, pw 0, c 2): pixel (1, 0) of channel 2
        assert_eq!(p[2 * 16 + 2], (2 * h * w + w) as f32);
        assert_eq!(super::unpatchify(&p, h, w), lat);
    }

    #[test]
    fn rope_pairs_follow_the_axes_and_pads_stay_put() {
        let cs = super::rope_table(&[[3, 0, 0], [0, 5, 0], [0, 0, 0]]);
        // token 0: the frame axis's first pair turns by 3, the height pairs not at all
        assert!((cs[0] - 3f32.cos()).abs() < 1e-6 && (cs[1] - 3f32.sin()).abs() < 1e-6);
        assert!((cs[2 * 16] - 1.0).abs() < 1e-6);
        // token 1: the first height pair (pair 16) turns by 5
        assert!((cs[128 + 2 * 16] - 5f32.cos()).abs() < 1e-6);
        // a pad token at the origin: no turn anywhere
        assert!(cs[256..].chunks(2).all(|p| p[0] == 1.0 && p[1] == 0.0));
    }

    #[test]
    fn the_fast_hadamard_is_the_matrix() {
        // v . H by the four 4-point passes against the dense matrix, on two groups; H . H = I (it undoes itself)
        let g = super::GROUP;
        let h = super::hadamard(g);
        let v: Vec<f32> = (0..2 * g).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let mut fast = v.clone();
        super::hadamard_rows(&mut fast);
        for (k, chunk) in v.chunks(g).enumerate() {
            for j in 0..g {
                let dense: f32 = (0..g).map(|i| chunk[i] * h[i * g + j]).sum();
                assert!((fast[k * g + j] - dense).abs() < 1e-4, "group {k}, column {j}");
            }
        }
        super::hadamard_rows(&mut fast);
        assert!(fast.iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn sequences_pad_to_32() {
        assert_eq!(super::padded(101), 128);
        assert_eq!(super::padded(4096), 4096);
        // 1280 x 720: 80 x 45 patches
        assert_eq!(super::padded(80 * 45), 3616);
    }

    #[test]
    fn the_timestep_is_the_pipelines() {
        // (1000 - 1000 s) / 1000 in float32; the sinusoid's first pair is cos / sin of t * 1000
        assert_eq!(super::model_t(1.0), 0.0);
        assert!((super::model_t(0.25) - 0.75).abs() < 1e-7);
        let e = super::sinusoid(0.5);
        assert!((e[0] - 500f32.cos()).abs() < 1e-6 && (e[128] - 500f32.sin()).abs() < 1e-6);
    }
}
