//! The denoiser: MiniMax H3's block stack (crates/h3 `dit`: the same block, the same kernels) held this engine's way, and
//! what MiniMax-H3-Image has around it - no audio stream, and a bank of six output heads.
//!
//! ```text
//!   x = [ text | the edit's pictures | the image's 2 x 2 patches ]        one sequence, bfloat16
//!   a block (H3's):  h = norm(x) (1 + scale_a[row]) + shift_a[row];  q, k, v = linear(h);  q, k: per-head norm + rotation
//!                    x += linear(attention(q, k, v)) gate_a[row]
//!                    h = norm(x) (1 + scale_m[row]) + shift_m[row];  x += linear(silu(u) w) gate_m[row], [u | w] = linear(h)
//!   out = heads(norm(x) (1 + scale[t]) + shift[t]) on the image's rows, float32: head 0, plus the heads of the grid
//!         intervals the step spans (offsets from head 0), weighted by their share of the step
//! ```
//!
//! A block's four matrices are held in one of three forms (`Mat`): int8 as ComfyUI's int8 ConvRot file stores them; a
//! GGUF's quantized blocks expanded to bfloat16 for each product (H3's way: the card holds the file's size, the product
//! runs at the 16-bit rate - the exact form of the quantized weights); or the quantized blocks turned into int8 once at
//! load by the silo (rotated as ConvRot's and rounded to 8 bits a row: the card holds a byte a weight and every product
//! runs on the matrix engine at the int8 rate).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use nextsycl_h3::device::{Device, Tensor};
use nextsycl_h3::dit::{rotations, AttentionFn};
use nextsycl_h3::dtype::DType;
use nextsycl_h3::layout::{Kind, Layout, RefBlock, Timesteps, VISUAL_COND_TIMESTEP};
use nextsycl_h3::ops::{self, Int8Linear, Linear, Mod, Rows};

use crate::ffi::Silo;
use crate::weights::{Quant, Weights};
use crate::{h3e, Error, Result};

/// The grid the six heads were distilled on (the checkpoint's `pdd_sigmas`; ComfyUI's and stable-diffusion.cpp's)
pub const PDD_SIGMAS: [f32; 7] = [1.0, 0.967_523_7, 0.887_548_5, 0.75, 0.532_773_9, 0.232_008_87, 0.0];
const PATCH: usize = 2;
const FINAL_CHUNK: usize = 4096;
const LINEARS: [&str; 4] = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"];

/// How a GGUF's quantized block matrices are multiplied
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// expanded to bfloat16 for each product (the quantized weights exactly)
    Exact,
    /// turned into int8 rows at load (the silo): int8 products
    Int8,
}

/// The sizes, from the file's shapes
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rot_dim: usize,
    pub modalities: usize,
    pub t_dim: usize,
    pub blocks: usize,
    pub refiners: usize,
    pub text_dim: usize,
    /// latent channels of the image (24) and the output heads in the bank (6)
    pub latent_c: usize,
    pub heads_out: usize,
}

pub const NORM_EPS: f32 = 1e-5;

fn count(w: &Weights, prefix: &str) -> usize {
    w.ck.entries.keys().filter_map(|k| k.strip_prefix(prefix)?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1)
}

impl Config {
    pub fn read(w: &Weights) -> Result<Config> {
        let one = |name: &str| -> Result<usize> { Ok(w.ck.get(name).map_err(h3e)?.shape[0]) };
        let hidden = one("blocks.0.norm1.weight")?;
        let head_dim = one("blocks.0.attn.q_norm.weight")?;
        let (_, qk) = w.shape2("blocks.0.attn.out_proj.weight")?;
        let (_, ffn) = w.shape2("blocks.0.mlp.fc2.weight")?;
        let adaln = w.ck.get("blocks.0.adaln_proj.linear.weight").map_err(h3e)?.shape.clone();
        let (patch_out, patch_in) = w.shape2("video_patch_proj.weight")?;
        let (out_rows, _) = w.shape2("final_layer.video_out.weight")?;
        let (_, text_dim) = w.shape2("condition_proj.weight")?;
        if w.ck.entries.contains_key("audio_patch_proj.weight") {
            return Err(Error(format!("{}: a MiniMax H3 video denoiser (it has an audio stream); MiniMax-H3-Image's has none", w.ck.path.display())));
        }
        if !w.ck.entries.contains_key("adaln_t_table") {
            return Err(Error(format!("{}: only the timestep-curve form (adaln_t_table) is read", w.ck.path.display())));
        }
        if patch_out != hidden || !patch_in.is_multiple_of(PATCH * PATCH) || adaln.len() != 2 || !adaln[0].is_multiple_of(6 * hidden) || !qk.is_multiple_of(head_dim) {
            return Err(Error(format!("{}: the shapes are not an H3 denoiser's", w.ck.path.display())));
        }
        let latent_c = patch_in / (PATCH * PATCH);
        if !out_rows.is_multiple_of(patch_in) || out_rows / patch_in != PDD_SIGMAS.len() - 1 {
            return Err(Error(format!("{}: {} output rows for {patch_in} a head: MiniMax-H3-Image's six-head bank is expected", w.ck.path.display(), out_rows)));
        }
        Ok(Config {
            hidden,
            heads: qk / head_dim,
            head_dim,
            ffn,
            rot_dim: one("rope.inv_freq")? * 6,
            modalities: adaln[0] / (6 * hidden),
            t_dim: adaln[1],
            blocks: count(w, "blocks."),
            refiners: count(w, "token_refiner.blocks."),
            text_dim,
            latent_c,
            heads_out: out_rows / patch_in,
        })
    }
}

/// A block matrix on the card
pub enum Mat {
    I8(Int8Linear),
    /// a GGUF's blocks as stored: [N, K] of `q`
    K { blocks: Tensor, q: Quant, n: usize, k: usize },
}

impl Mat {
    pub fn size(&self) -> (usize, usize) {
        match self {
            Mat::I8(l) => (l.outputs(), l.inputs()),
            Mat::K { n, k, .. } => (*n, *k),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Mat::I8(l) => l.weight.buf.len() + l.scale.buf.len(),
            Mat::K { blocks, .. } => blocks.buf.len(),
        }
    }
}

struct Block {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    mats: [Mat; 4],
    /// [6 * hidden * modalities, t_dim] and its bias (empty for a refiner block: no timestep tables)
    adaln_w: Vec<f32>,
    adaln_b: Vec<f32>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_tensor(dev: &Arc<Device>, shape: &[usize], v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, shape, &f32_bytes(v)).map_err(h3e)
}

/// The block's six tables for the timestep embeddings [R, t_dim]: shift, scale, gate of the attention half, then of the
/// MLP half; each [R * modalities, hidden], row = timestep index * modalities + modality (H3's `Block::tables`)
pub fn tables(cfg: &Config, adaln_w: &[f32], adaln_b: &[f32], t_emb: &[f32]) -> [Vec<f32>; 6] {
    let (c, m, td) = (cfg.hidden, cfg.modalities, cfg.t_dim);
    let r = t_emb.len() / td;
    let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0f32; r * m * c]);
    for ri in 0..r {
        let t = &t_emb[ri * td..(ri + 1) * td];
        for mi in 0..m {
            for (ci, table) in out.iter_mut().enumerate() {
                let o0 = mi * 6 * c + ci * c;
                let row = &mut table[(ri * m + mi) * c..(ri * m + mi + 1) * c];
                for (i, v) in row.iter_mut().enumerate() {
                    let o = o0 + i;
                    *v = adaln_b[o] + adaln_w[o * td..(o + 1) * td].iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
    }
    out
}

/// The weights of the output heads for a step from `sigma` to `next` on the six-head bank's grid: head 0 (the base)
/// always, and each head whose grid interval the step spans by its share of the step's length (ComfyUI's `_pdd_head`).
/// On the grid itself: step 0 is head 0 alone, step i is head 0 + head i.
pub fn head_weights(sigma: f32, next: f32) -> [f32; 6] {
    let nearest = |v: f32| (0..PDD_SIGMAS.len()).min_by(|a, b| (PDD_SIGMAS[*a] - v).abs().total_cmp(&(PDD_SIGMAS[*b] - v).abs())).unwrap_or(0);
    let n = PDD_SIGMAS.len() - 1;
    let start = nearest(sigma).min(n - 1);
    let stop = nearest(next).max(start + 1);
    let mut w = [0f32; 6];
    w[0] = 1.0;
    let total = PDD_SIGMAS[start] - PDD_SIGMAS[stop];
    for (k, wk) in w.iter_mut().enumerate().take(stop).skip(start.max(1)) {
        *wk = (PDD_SIGMAS[k] - PDD_SIGMAS[k + 1]) / total;
    }
    w
}

/// The sampling grid of `steps` steps: the heads' own at six, else the grid read at even positions (ComfyUI's
/// `MiniMaxH3ImageModelSampling.sigma`: linear in the grid index)
pub fn sigmas(steps: usize) -> Vec<f32> {
    let n = (PDD_SIGMAS.len() - 1) as f32;
    (0..=steps)
        .map(|i| {
            let pos = (1.0 - i as f32 / steps as f32).clamp(0.0, 1.0) * n;
            // the grid ascending (index 0 = sigma 0), as the reference stores it
            let g = |k: usize| PDD_SIGMAS[PDD_SIGMAS.len() - 1 - k];
            let k = (pos.floor() as usize).min(PDD_SIGMAS.len() - 2);
            g(k) + (pos - k as f32) * (g(k + 1) - g(k))
        })
        .collect()
}

/// Video latent [C, H, W] -> rows [H/2 * W/2, C * 4]: one row per 2x2 patch, features (c, dy, dx)
pub fn patchify(v: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    nextsycl_h3::denoiser::patchify(v, c, 1, h, w)
}

pub fn unpatchify(rows: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    nextsycl_h3::denoiser::unpatchify(rows, c, 1, h, w)
}

/// Seconds by stage of the blocks (NS_H3I_PROFILE: a wait after each stage)
#[derive(Default)]
pub struct Profile {
    pub on: bool,
    stages: std::sync::Mutex<Vec<(&'static str, f64)>>,
}

impl Profile {
    fn add(&self, name: &'static str, s: f64) {
        let mut st = self.stages.lock().unwrap_or_else(|p| p.into_inner());
        match st.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += s,
            None => st.push((name, s)),
        }
    }
    /// The stages' seconds since the last call, as one line
    pub fn take(&self) -> String {
        let st = std::mem::take(&mut *self.stages.lock().unwrap_or_else(|p| p.into_inner()));
        let sum: f64 = st.iter().map(|s| s.1).sum();
        format!("{sum:.2} s: {}", st.iter().map(|(n, t)| format!("{n} {t:.2}")).collect::<Vec<_>>().join(", "))
    }
}

/// The denoiser on the card
pub struct Stack {
    pub cfg: Config,
    dev: Arc<Device>,
    blocks: Vec<Block>,
    refiner: Vec<Block>,
    refiner_norm: Tensor,
    cond_proj: Linear,
    video_patch: Linear,
    final_norm: Tensor,
    final_w: Vec<f32>,
    final_b: Vec<f32>,
    /// the six heads [6 * 96, hidden] and their biases, on the host: blended for each step
    heads_w: Vec<f32>,
    heads_b: Vec<f32>,
    pub t_table: Vec<f32>,
    pub inv_freq: Vec<f32>,
    /// the bfloat16 buffers the quantized matrices expand into, by shape (the exact form)
    wbuf: BTreeMap<(usize, usize), Linear>,
    pub silo: Option<&'static Silo>,
    gpu: Arc<nextsycl_core::Gpu>,
    /// a silo's attention for the long sequences (else the shared kernels')
    pub attention: Option<AttentionFn>,
    pub quant: Option<Quant>,
    pub form: Form,
    pub load_seconds: f64,
    pub convert_seconds: f64,
    pub file_bytes: u64,
    pub card_bytes: u64,
    pub profile: Profile,
}

/// What a step fixes for every block
pub struct Step {
    tokens: usize,
    rows: Tensor,
    cs: Tensor,
    tables: Vec<[Tensor; 6]>,
}

/// The buffers a block writes between its kernels
pub struct Scratch {
    h: Tensor,
    qkv: Tensor,
    att: Tensor,
    proj: Tensor,
    fc1: Tensor,
    act: Tensor,
}

impl Scratch {
    pub fn new(dev: &Arc<Device>, cfg: &Config, tokens: usize) -> Result<Scratch> {
        let w = cfg.heads * cfg.head_dim;
        let t = |c: usize| Tensor::new(dev, DType::BF16, &[tokens, c]).map_err(h3e);
        Ok(Scratch { h: t(cfg.hidden)?, qkv: t(3 * w)?, att: t(w)?, proj: t(cfg.hidden)?, fc1: t(2 * cfg.ffn)?, act: t(cfg.ffn)? })
    }
}

impl Stack {
    /// Loads the denoiser: the blocks, the text refiner and the parts around them. `form`: how a GGUF's matrices are
    /// multiplied (the int8 file is int8 whatever it says)
    pub fn load(gpu: &Arc<nextsycl_core::Gpu>, dev: &Arc<Device>, w: &Weights, form: Form, silo: Option<&'static Silo>, threads: usize,
                log: &mut dyn FnMut(String)) -> Result<Stack> {
        let cfg = Config::read(w)?;
        let quant = w.block_quant();
        if let Some(q) = quant {
            if q.shared().is_none() && silo.is_none() {
                return Err(Error(format!("{} matrices are expanded by this engine's silo (dist/silo/libnextsycl-{}.so), which is not there", q.name(), crate::ffi::SILO)));
            }
            if form == Form::Int8 && silo.is_none() {
                return Err(Error(format!("int8 products of a GGUF's matrices need this engine's silo (dist/silo/libnextsycl-{}.so)", crate::ffi::SILO)));
            }
        }
        let wanted = |name: &str| -> bool {
            let tail = match (name.strip_prefix("blocks."), name.strip_prefix("token_refiner.blocks.")) {
                (Some(r), _) | (_, Some(r)) => r.split_once('.').map(|x| x.1),
                _ => None,
            };
            tail.is_some_and(|t| LINEARS.iter().any(|l| t == format!("{l}.weight") || t == format!("{l}.weight_scale")))
        };
        let loaded = nextsycl_h3::load::load(dev, &w.ck, threads, wanted).map_err(h3e)?;
        let mut big = loaded.tensors;
        let t0 = Instant::now();
        let mut card = 0u64;
        let small = |name: &str| -> Result<Tensor> {
            let v = w.host(name)?;
            f32_tensor(dev, &[v.len()], &v)
        };
        let mut assemble = |p: &str, tables: bool| -> Result<Block> {
            let mut mats = Vec::with_capacity(4);
            for l in LINEARS {
                let name = format!("{p}.{l}");
                let wn = format!("{name}.weight");
                let t = big.remove(&wn).ok_or_else(|| Error(format!("{wn} was not loaded")))?;
                let m = match w.quants.get(&wn) {
                    Some((q, shape)) => {
                        let (n, k) = (shape[0], shape[1]);
                        match (form, silo) {
                            (Form::Int8, Some(s)) => Mat::I8(s.to_int8(gpu, dev, &t, *q, n, k)?),
                            _ => Mat::K { blocks: t, q: *q, n, k },
                        }
                    }
                    None => {
                        let q = w.ck.quant(&name).map_err(h3e)?.ok_or_else(|| Error(format!("{name}: neither an int8 ConvRot matrix nor a GGUF quant")))?;
                        let scale = big.remove(&format!("{name}.weight_scale")).ok_or_else(|| Error(format!("{name}.weight_scale was not loaded")))?;
                        Mat::I8(Int8Linear { weight: t, scale, bias: None, group: q.convrot.then_some(q.group) })
                    }
                };
                card += m.bytes() as u64;
                mats.push(m);
            }
            let mats: [Mat; 4] = mats.try_into().map_err(|_| Error("four matrices a block".into()))?;
            let (adaln_w, adaln_b) = match tables {
                true => (w.host(&format!("{p}.adaln_proj.linear.weight"))?, w.host(&format!("{p}.adaln_proj.linear.bias"))?),
                false => (Vec::new(), Vec::new()),
            };
            Ok(Block {
                mats,
                norm1: small(&format!("{p}.norm1.weight"))?,
                norm2: small(&format!("{p}.norm2.weight"))?,
                q_norm: small(&format!("{p}.attn.q_norm.weight"))?,
                k_norm: small(&format!("{p}.attn.k_norm.weight"))?,
                adaln_w,
                adaln_b,
            })
        };
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for i in 0..cfg.blocks {
            blocks.push(assemble(&format!("blocks.{i}"), true)?);
        }
        let mut refiner = Vec::with_capacity(cfg.refiners);
        for i in 0..cfg.refiners {
            refiner.push(assemble(&format!("token_refiner.blocks.{i}"), false)?);
        }
        dev.wait().map_err(h3e)?;
        let convert_seconds = t0.elapsed().as_secs_f64();
        // the exact form's expansion buffers: one per shape
        let mut wbuf = BTreeMap::new();
        for m in blocks.iter().chain(&refiner).flat_map(|b| b.mats.iter()) {
            if let Mat::K { n, k, .. } = m {
                if !wbuf.contains_key(&(*n, *k)) {
                    wbuf.insert((*n, *k), Linear { weight: Tensor::new(dev, DType::BF16, &[*n, *k]).map_err(h3e)?, bias: None });
                    card += (*n * *k * 2) as u64;
                }
            }
        }
        // a matrix outside the blocks: float as stored, or a GGUF quant expanded on the host (the Q8_0 file's input
        // projection)
        let matrix = |name: &str| -> Result<(Vec<f32>, usize, usize)> {
            let (n, k) = w.shape2(name)?;
            let v = match w.quants.get(name) {
                Some((q, _)) => crate::weights::dequant(*q, &w.ck.read(name).map_err(h3e)?),
                None => w.host(name)?,
            };
            Ok((v, n, k))
        };
        let linear = |name: &str, dt: DType| -> Result<Linear> {
            let (v, n, k) = matrix(&format!("{name}.weight"))?;
            let weight = match dt {
                DType::BF16 => Tensor::from_bytes(dev, DType::BF16, &[n, k], &nextsycl_h3::denoiser::bf16_bytes(&v)).map_err(h3e)?,
                _ => f32_tensor(dev, &[n, k], &v)?,
            };
            Ok(Linear { weight, bias: Some(small(&format!("{name}.bias"))?) })
        };
        let (heads_w, _, _) = matrix("final_layer.video_out.weight")?;
        let s = Stack {
            cfg,
            dev: dev.clone(),
            blocks,
            refiner,
            refiner_norm: small("token_refiner.final_norm.weight")?,
            cond_proj: linear("condition_proj", DType::BF16)?,
            video_patch: linear("video_patch_proj", DType::F32)?,
            final_norm: small("final_layer.norm.weight")?,
            final_w: w.host("final_layer.adaln_proj.linear.weight")?,
            final_b: w.host("final_layer.adaln_proj.linear.bias")?,
            heads_w,
            heads_b: w.host("final_layer.video_out.bias")?,
            t_table: w.host("adaln_t_table")?,
            inv_freq: w.host("rope.inv_freq")?,
            wbuf,
            silo,
            gpu: gpu.clone(),
            attention: None,
            quant,
            form: if quant.is_some() { form } else { Form::Int8 },
            load_seconds: loaded.seconds,
            convert_seconds,
            file_bytes: loaded.bytes,
            card_bytes: card,
            profile: Profile { on: std::env::var_os("NS_H3I_PROFILE").is_some(), ..Profile::default() },
        };
        if s.final_w.len() != 2 * cfg.hidden * cfg.t_dim || s.inv_freq.len() * 6 != cfg.rot_dim || s.heads_b.len() != cfg.heads_out * cfg.latent_c * 4 {
            return Err(Error("the final layer or the rotation frequencies do not fit the blocks".into()));
        }
        log(format!("{}: {} blocks + {} refiner blocks, {} -> {}, {:.2} GiB read in {:.1} s{}, {:.2} GiB on the card", crate::ARCH, cfg.blocks, cfg.refiners,
                    quant.map_or("int8 ConvRot".to_string(), |q| q.name().to_string()),
                    match (quant, s.form) { (Some(_), Form::Exact) => "bfloat16 a product", _ => "int8 products" }, loaded.bytes as f64 / (1u64 << 30) as f64,
                    loaded.seconds, if quant.is_some() && s.form == Form::Int8 { format!(", to int8 in {convert_seconds:.1} s") } else { String::new() },
                    card as f64 / (1u64 << 30) as f64));
        Ok(s)
    }

    /// `out = linear(x)` with one of the matrices
    fn mm(&self, m: &Mat, x: &Tensor, out: &Tensor) -> Result<()> {
        match m {
            Mat::I8(l) => l.forward(x, out).map_err(h3e),
            Mat::K { blocks, q, n, k } => {
                let lin = self.wbuf.get(&(*n, *k)).ok_or_else(|| Error("a quantized matrix without its expansion buffer".into()))?;
                match (q.shared(), self.silo) {
                    // the silo's expansion where it is built (every type, Q3_K among them); else H3's shared one
                    (_, Some(s)) => s.expand(&self.gpu, blocks, *q, n * k, &lin.weight)?,
                    (Some(g), None) => ops::expand_into(blocks, g, n * k, &lin.weight).map_err(h3e)?,
                    (None, None) => return Err(Error(format!("{} needs the silo", q.name()))),
                }
                lin.forward(x, out).map_err(h3e)
            }
        }
    }

    fn lap(&self, name: &'static str, mark: &mut Instant) -> Result<()> {
        if self.profile.on {
            self.dev.wait().map_err(h3e)?;
            self.profile.add(name, mark.elapsed().as_secs_f64());
            *mark = Instant::now();
        }
        Ok(())
    }

    /// One block on the stream `x` [S, hidden], in place. `m`: the timestep tables (none: a refiner block);
    /// `cs`, `rot_dim`: the rotation (0: none)
    #[allow(clippy::too_many_arguments)]
    fn run_block(&self, b: &Block, x: &Tensor, tokens: usize, rows: Option<&Tensor>, t: Option<&[Tensor; 6]>, cs: &Tensor, rot_dim: usize, s: &Scratch) -> Result<()> {
        let cfg = &self.cfg;
        let mut mark = Instant::now();
        let md = |scale: usize, shift: usize| match (rows, t) {
            (Some(r), Some(t)) => Some(Mod { rows: r, scale: &t[scale], shift: &t[shift] }),
            _ => None,
        };
        let gated = |other: &Tensor, gate: usize| -> Result<()> {
            match (rows, t) {
                (Some(r), Some(t)) => ops::gate_add(x, other, r, &t[gate]).map_err(h3e),
                _ => ops::add(x, other).map_err(h3e),
            }
        };
        ops::rms_norm_mod(x, &b.norm1, NORM_EPS, md(1, 0).as_ref(), &s.h).map_err(h3e)?;
        self.lap("norm", &mut mark)?;
        self.mm(&b.mats[0], &s.h, &s.qkv)?;
        self.lap("qkv", &mut mark)?;
        let w = cfg.heads * cfg.head_dim;
        let part = |i: usize| Rows { t: &s.qkv, offset: i * w, stride: 3 * w, tokens, heads: cfg.heads, dim: cfg.head_dim };
        ops::rms_rope(part(0), &b.q_norm, NORM_EPS, cs, rot_dim).map_err(h3e)?;
        ops::rms_rope(part(1), &b.k_norm, NORM_EPS, cs, rot_dim).map_err(h3e)?;
        self.lap("rope", &mut mark)?;
        if !self.attention.as_ref().map_or(Ok(false), |f| f(part(0), part(1), part(2), &s.att)).map_err(h3e)? {
            ops::attention(part(0), part(1), part(2), &s.att).map_err(h3e)?;
        }
        self.lap("attention", &mut mark)?;
        self.mm(&b.mats[1], &s.att, &s.proj)?;
        self.lap("out", &mut mark)?;
        gated(&s.proj, 2)?;
        self.lap("add", &mut mark)?;
        ops::rms_norm_mod(x, &b.norm2, NORM_EPS, md(4, 3).as_ref(), &s.h).map_err(h3e)?;
        self.lap("norm", &mut mark)?;
        self.mm(&b.mats[2], &s.h, &s.fc1)?;
        self.lap("fc1", &mut mark)?;
        ops::swiglu(&s.fc1, &s.act).map_err(h3e)?;
        self.lap("swiglu", &mut mark)?;
        self.mm(&b.mats[3], &s.act, &s.proj)?;
        self.lap("fc2", &mut mark)?;
        gated(&s.proj, 5)?;
        self.lap("add", &mut mark)
    }

    /// The text encoder's states [L, text_dim] -> the text tokens: bfloat16 [L, hidden] (the projection, the refiner's
    /// blocks - no timestep, no rotation - and its final norm)
    pub fn refine(&self, cond: &[f32]) -> Result<Tensor> {
        let cfg = &self.cfg;
        let l = cond.len() / cfg.text_dim;
        if l == 0 || cond.len() != l * cfg.text_dim {
            return Err(Error(format!("{} conditioning values are not rows of {}", cond.len(), cfg.text_dim)));
        }
        let dev = &self.dev;
        let c = Tensor::from_bytes(dev, DType::BF16, &[l, cfg.text_dim], &nextsycl_h3::denoiser::bf16_bytes(cond)).map_err(h3e)?;
        let x = Tensor::new(dev, DType::BF16, &[l, cfg.hidden]).map_err(h3e)?;
        self.cond_proj.forward(&c, &x).map_err(h3e)?;
        let s = Scratch::new(dev, cfg, l)?;
        let none = f32_tensor(dev, &[1], &[0.0])?;
        for b in &self.refiner {
            self.run_block(b, &x, l, None, None, &none, 0, &s)?;
        }
        ops::rms_norm_mod(&x, &self.refiner_norm, NORM_EPS, None, &x).map_err(h3e)?;
        self.dev.wait().map_err(h3e)?;
        // the refiner's stages are not the steps'
        if self.profile.on {
            let _ = self.profile.take();
        }
        Ok(x)
    }

    fn step(&self, rows: &[i32], positions: &[f64], t_emb: &[f32]) -> Result<Step> {
        let cfg = &self.cfg;
        let dev = &self.dev;
        let r = t_emb.len() / cfg.t_dim * cfg.modalities;
        if let Some(bad) = rows.iter().find(|x| **x < 0 || **x as usize >= r) {
            return Err(Error(format!("step: table row {bad} of {r}")));
        }
        let cs = rotations(positions, &self.inv_freq);
        let mut tabs = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let t = tables(cfg, &b.adaln_w, &b.adaln_b, t_emb);
            let mut d = Vec::with_capacity(6);
            for v in &t {
                d.push(f32_tensor(dev, &[r, cfg.hidden], v)?);
            }
            tabs.push(d.try_into().map_err(|_| Error("six tables a block".into()))?);
        }
        Ok(Step {
            tokens: rows.len(),
            rows: Tensor::from_bytes(dev, DType::I32, &[rows.len()], &rows.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(h3e)?,
            cs: f32_tensor(dev, &[rows.len(), cfg.rot_dim / 2, 2], &cs)?,
            tables: tabs,
        })
    }

    /// The six heads blended for a step: [96, hidden] and its bias
    pub fn head(&self, sigma: f32, next: f32) -> (Vec<f32>, Vec<f32>) {
        let per = self.cfg.latent_c * PATCH * PATCH;
        let c = self.cfg.hidden;
        let hw = head_weights(sigma, next);
        let (mut w, mut b) = (vec![0f32; per * c], vec![0f32; per]);
        for (k, wk) in hw.iter().enumerate().filter(|(_, w)| **w != 0.0) {
            w.iter_mut().zip(&self.heads_w[k * per * c..(k + 1) * per * c]).for_each(|(o, v)| *o += wk * v);
            b.iter_mut().zip(&self.heads_b[k * per..(k + 1) * per]).for_each(|(o, v)| *o += wk * v);
        }
        (w, b)
    }
}

/// An edit's picture as the denoiser gets it: normalized latents [24, h, w]
pub struct Reference {
    pub latent: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// One picture's network: the text tokens, the layout and the buffers
pub struct Denoiser<'a> {
    pub stack: &'a Stack,
    pub layout: Layout,
    pub lh: usize,
    pub lw: usize,
    text: Tensor,
    tags: Vec<i32>,
    x: Tensor,
    scratch: Scratch,
    /// the references' rows, embedded once: (first row, rows)
    refs: Vec<(usize, Tensor)>,
    /// called before each block with its index (cancel checks)
    pub taps: Option<Tap<'a>>,
}

/// Called with a stage's name and its tensor (the parity check's taps: "x_in", "blk_N")
pub type Tap<'a> = Box<dyn FnMut(&str, &Tensor) -> Result<()> + 'a>;

impl<'a> Denoiser<'a> {
    /// `text`: the refined text tokens [L, hidden]; `tags`: each token's modality (1 text, 0 a picture's vision tokens);
    /// `refs`: the edit's pictures, each mixed `0.999 latent + 0.001 noise` here with noise drawn afresh from `seed`
    /// (as the reference draws it: one stream per picture, over its patch rows)
    pub fn new(stack: &'a Stack, text: Tensor, tags: &[i32], refs: &[Reference], lh: usize, lw: usize, seed: u64) -> Result<Denoiser<'a>> {
        let cfg = &stack.cfg;
        let dev = &stack.dev;
        if text.dtype != DType::BF16 || text.shape[1] != cfg.hidden || tags.len() != text.shape[0] || tags.iter().any(|t| !(0..3).contains(t)) {
            return Err(Error(format!("the text tokens: bfloat16 [L, {}] with a modality tag each", cfg.hidden)));
        }
        let blocks: Vec<RefBlock> = refs.iter().map(|r| RefBlock::Image { h: r.h, w: r.w }).collect();
        let layout = Layout::new(text.shape[0], 1, lh, lw, 0, &[], &blocks).map_err(h3e)?;
        let mut rows = Vec::new();
        for (r, seg) in refs.iter().zip(layout.segments.iter().filter(|s| s.kind == Kind::RefImg)) {
            if r.latent.len() != cfg.latent_c * r.h * r.w {
                return Err(Error(format!("a reference of {} values for {} x {} x {}", r.latent.len(), cfg.latent_c, r.h, r.w)));
            }
            let mut p = patchify(&r.latent, cfg.latent_c, r.h, r.w);
            let aug = VISUAL_COND_TIMESTEP as f32;
            let noise = nextsycl_h3::noise::Mt19937::new(seed).randn(p.len());
            p.iter_mut().zip(noise).for_each(|(v, n)| *v = aug * *v + (1.0 - aug) * n);
            rows.push((seg.start, stack.embed(&p)?));
        }
        let s = layout.tokens();
        Ok(Denoiser {
            x: Tensor::new(dev, DType::BF16, &[s, cfg.hidden]).map_err(h3e)?,
            scratch: Scratch::new(dev, cfg, s)?,
            stack,
            layout,
            lh,
            lw,
            text,
            tags: tags.to_vec(),
            refs: rows,
            taps: None,
        })
    }

    pub fn tokens(&self) -> usize {
        self.layout.tokens()
    }

    /// Each token's table row at noise level `sigma`, the distinct timesteps, and the image rows' timestep index
    pub fn timesteps(&self, sigma: f32) -> (Timesteps, usize) {
        // one stream, no shift: the audio stream's timestep falls on the image's
        let mut ts = Timesteps::with_cond(&self.layout, sigma as f64, 1.0, 1.0, VISUAL_COND_TIMESTEP, 1.0);
        if let Some(seg) = self.layout.segment(Kind::Text) {
            if seg.stop > seg.start {
                let base = ts.rows[seg.start] - Kind::Text.modality();
                for (r, tag) in ts.rows[seg.start..seg.stop].iter_mut().zip(&self.tags) {
                    *r = base + tag;
                }
            }
        }
        let vi = ts.video_index;
        (ts, vi)
    }

    /// The network at noise level `sigma` of a step to `next`: latents [24, lh, lw] in, the velocity out (the reference's
    /// sign: the head's output negated). `between(block)` before each block.
    pub fn net(&mut self, latent: &[f32], sigma: f32, next: f32, between: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let st = self.stack;
        let cfg = &st.cfg;
        let dev = &st.dev;
        let (ts, vi) = self.timesteps(sigma);
        let t_emb = ts.embeddings(&st.t_table, cfg.t_dim);
        let step = st.step(&ts.rows, &self.layout.positions, &t_emb)?;
        let seg = |k: Kind| self.layout.segment(k).ok_or_else(|| Error(format!("no {k:?} rows in the layout")));
        let (stext, simg) = (seg(Kind::Text)?, seg(Kind::Video)?);
        let n = simg.stop - simg.start;
        let e = st.embed(&patchify(latent, cfg.latent_c, self.lh, self.lw))?;
        self.x.copy_rows(stext.start, &self.text, 0, stext.stop - stext.start).map_err(h3e)?;
        self.x.copy_rows(simg.start, &e, 0, n).map_err(h3e)?;
        for (start, rows) in &self.refs {
            self.x.copy_rows(*start, rows, 0, rows.shape[0]).map_err(h3e)?;
        }
        if let Some(t) = self.taps.as_mut() {
            t("x_in", &self.x)?;
        }
        for (i, b) in st.blocks.iter().enumerate() {
            between(i)?;
            st.run_block(b, &self.x, step.tokens, Some(&step.rows), Some(&step.tables[i]), &step.cs, cfg.rot_dim, &self.scratch)?;
            if let Some(t) = self.taps.as_mut() {
                t(&format!("blk_{i}"), &self.x)?;
            }
        }
        // the final layer: norm, the timestep's scale and shift, the step's blend of the heads; in chunks of rows
        let (c, td) = (cfg.hidden, cfg.t_dim);
        let r = ts.values.len();
        let (mut shift, mut scale) = (vec![0f32; r * c], vec![0f32; r * c]);
        for ri in 0..r {
            let t = &t_emb[ri * td..(ri + 1) * td];
            for o in 0..2 * c {
                let v = st.final_b[o] + st.final_w[o * td..(o + 1) * td].iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
                if o < c {
                    shift[ri * c + o] = v;
                } else {
                    scale[ri * c + o - c] = v;
                }
            }
        }
        let shift = f32_tensor(dev, &[r, c], &shift)?;
        let scale = f32_tensor(dev, &[r, c], &scale)?;
        let per = cfg.latent_c * PATCH * PATCH;
        let (hw, hb) = st.head(sigma, next);
        let head = Linear { weight: f32_tensor(dev, &[per, c], &hw)?, bias: Some(f32_tensor(dev, &[per], &hb)?) };
        let mut out = Vec::with_capacity(n * per);
        let mut done = 0;
        while done < n {
            let m = FINAL_CHUNK.min(n - done);
            let rows = Tensor::new(dev, DType::BF16, &[m, c]).map_err(h3e)?;
            rows.copy_rows(0, &self.x, simg.start + done, m).map_err(h3e)?;
            let table = Tensor::from_bytes(dev, DType::I32, &[m], &(vi as i32).to_le_bytes().repeat(m)).map_err(h3e)?;
            let h = Tensor::new(dev, DType::F32, &[m, c]).map_err(h3e)?;
            ops::rms_norm_mod(&rows, &st.final_norm, NORM_EPS, Some(&Mod { rows: &table, scale: &scale, shift: &shift }), &h).map_err(h3e)?;
            let y = Tensor::new(dev, DType::F32, &[m, per]).map_err(h3e)?;
            head.forward(&h, &y).map_err(h3e)?;
            out.extend(y.to_f32().map_err(h3e)?);
            done += m;
        }
        let mut v = unpatchify(&out, cfg.latent_c, self.lh, self.lw);
        v.iter_mut().for_each(|x| *x = -*x);
        Ok(v)
    }
}

impl Stack {
    /// Patch rows [n, 96] (float32) through the input projection: bfloat16 [n, hidden]
    fn embed(&self, rows: &[f32]) -> Result<Tensor> {
        let k = self.video_patch.inputs();
        let n = rows.len() / k;
        let input = f32_tensor(&self.dev, &[n, k], rows)?;
        let out = Tensor::new(&self.dev, DType::BF16, &[n, self.cfg.hidden]).map_err(h3e)?;
        self.video_patch.forward(&input, &out).map_err(h3e)?;
        Ok(out)
    }
}

/// One Euler step of the flow: x += v (next - sigma) (ComfyUI's sampler on `denoised = x - sigma v`)
pub fn euler(x: &mut [f32], v: &[f32], sigma: f32, next: f32) {
    let dt = next - sigma;
    x.iter_mut().zip(v).for_each(|(x, v)| *x += v * dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_s_steps_take_head_0_and_their_own() {
        // step 0: the base head alone; step i: the base and offset i
        for i in 0..6 {
            let w = head_weights(PDD_SIGMAS[i], PDD_SIGMAS[i + 1]);
            let mut want = [0f32; 6];
            want[0] = 1.0;
            want[i] = 1.0;
            assert_eq!(w, want, "step {i}");
        }
    }

    #[test]
    fn a_step_over_two_intervals_blends_their_heads_by_length() {
        // 3 steps: 1 -> 0.8875 -> 0.5328 -> 0: step 1 spans intervals 2 and 3
        let w = head_weights(PDD_SIGMAS[2], PDD_SIGMAS[4]);
        let total = PDD_SIGMAS[2] - PDD_SIGMAS[4];
        assert_eq!(w[0], 1.0);
        assert!((w[2] - (PDD_SIGMAS[2] - PDD_SIGMAS[3]) / total).abs() < 1e-6);
        assert!((w[3] - (PDD_SIGMAS[3] - PDD_SIGMAS[4]) / total).abs() < 1e-6);
        assert!((w[2] + w[3] - 1.0).abs() < 1e-6);
        assert_eq!((w[1], w[4], w[5]), (0.0, 0.0, 0.0));
        // the first step over two intervals: head 0 is the base, interval 1's offset by its share
        let w = head_weights(PDD_SIGMAS[0], PDD_SIGMAS[2]);
        assert_eq!(w[0], 1.0);
        assert!((w[1] - (PDD_SIGMAS[1] - PDD_SIGMAS[2]) / (PDD_SIGMAS[0] - PDD_SIGMAS[2])).abs() < 1e-6);
        // a sigma off the grid goes to its nearest point; the last point steps to the end
        assert_eq!(head_weights(0.96, 0.89), head_weights(PDD_SIGMAS[1], PDD_SIGMAS[2]));
        assert_eq!(head_weights(0.0, 0.0), head_weights(PDD_SIGMAS[5], PDD_SIGMAS[6]));
    }

    #[test]
    fn six_steps_are_the_grid_and_other_counts_read_it_evenly() {
        let s = sigmas(6);
        for (a, b) in s.iter().zip(PDD_SIGMAS) {
            assert!((a - b).abs() < 1e-6, "{s:?}");
        }
        // 3 steps: every second point
        let s = sigmas(3);
        for (a, b) in s.iter().zip([PDD_SIGMAS[0], PDD_SIGMAS[2], PDD_SIGMAS[4], PDD_SIGMAS[6]]) {
            assert!((a - b).abs() < 1e-6, "{s:?}");
        }
        // 12 steps: the grid's points and their midpoints, falling to 0
        let s = sigmas(12);
        assert_eq!(s.len(), 13);
        assert!((s[1] - (PDD_SIGMAS[0] + PDD_SIGMAS[1]) / 2.0).abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[1] < w[0]) && s[12] == 0.0);
    }

    #[test]
    fn euler_moves_along_the_velocity() {
        let mut x = vec![1.0f32, -2.0];
        euler(&mut x, &[0.5, 1.0], 1.0, 0.75);
        assert_eq!(x, vec![1.0 - 0.125, -2.0 - 0.25]);
        // the last step lands on x - sigma v
        let mut y = vec![3.0f32];
        euler(&mut y, &[2.0], 0.25, 0.0);
        assert_eq!(y, vec![2.5]);
    }

    #[test]
    fn the_tables_are_the_projection_of_the_timestep() {
        let cfg = Config { hidden: 2, heads: 1, head_dim: 2, ffn: 2, rot_dim: 0, modalities: 3, t_dim: 2, blocks: 1, refiners: 0, text_dim: 2, latent_c: 1, heads_out: 6 };
        // [6 * hidden * modalities, t_dim] = [36, 2]: row o = (o, 1), bias o / 10
        let w: Vec<f32> = (0..36).flat_map(|o| [o as f32, 1.0]).collect();
        let b: Vec<f32> = (0..36).map(|o| o as f32 / 10.0).collect();
        let t = tables(&cfg, &w, &b, &[1.0, 2.0, 0.0, 1.0]);
        // table 2 (gate_a), timestep 0, modality 1, feature 1: o = 1 * 12 + 2 * 2 + 1 = 17 -> 17 * 1 + 2 + 1.7
        assert_eq!(t[2][(3 + 1) * 2 - 6 + 1], 17.0 + 2.0 + 1.7);
        // timestep 1 (0, 1): 1 + bias
        assert_eq!(t[2][(3 + 1) * 2 + 1], 1.0 + 1.7);
        assert_eq!(t[0].len(), 2 * 3 * 2);
    }
}
