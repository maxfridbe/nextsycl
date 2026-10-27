//! The denoiser: MOSS-Image's DiT (OpenMOSS `pipeline_moss_image.py`, `MOSSImageTransformer`): 42 blocks of 3,072
//! features (24 heads x 128) over one stream, the image's tokens followed by the prompt's, the matrices kept on the
//! card in the GGUF's types.
//!
//! ```text
//!   image: the latents [H*W, 128] -> video_embedder (Linear 128 -> 3072)
//!   text:  the encoder's hidden states [L, 4096] -> Linear 4096 -> 3072 -> 2 text blocks (biases, fused qkv, gated FFN)
//!   [image | text] -> 42 blocks -> the image's rows: rmsnorm * (1 + scale) + shift, Linear 3072 -> 128
//!   block: x += o(attention(rmsnorm1(x) * (1 + scale_a) + shift_a)) * gate_a
//!          x += down(situ(up, gate)) * gate_f,  (up, gate) = ffn_up_gate(rmsnorm2(x) * (1 + scale_f) + shift_f)
//! ```
//!
//! The six modulation vectors (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f) are one linear of the timestep's
//! embedding, shared by every block. RoPE: 4 axes (frame, text position, height, width) of 16 bands, applied half-split
//! (feature j with j + 64), image tokens at (0, text length, h - (H - 1)/2, w - (W - 1)/2), text at (0, i, 0, 0).

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_diffusion::lora::{self, Delta};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;
use crate::part::{f16_bits, host_vec, Part};

pub const DIM: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD: usize = 128;
pub const FFN: usize = 8192;
/// the latent channels: a token's features
pub const LATENT: usize = 128;
pub const TEXT_DIM: usize = 4096;
const FREQ: usize = 256;
const EPS: f32 = 1e-6;
const BANDS: usize = HEAD / 8;

/// A block's matrices expanded to half on the card (NS_MOSS_EXPAND): the same values the per-forward expansion gives
struct Half {
    /// q, k and v side by side: [3 x 3072, 3072]
    qkv: DevBuf,
    o: DevBuf,
    up_gate: DevBuf,
    down: DevBuf,
}

struct Block {
    half: Option<Half>,
    q: Part,
    k: Part,
    v: Part,
    o: Part,
    up_gate: Part,
    down: Part,
    q_norm: DevBuf,
    k_norm: DevBuf,
    norm1: DevBuf,
    norm2: DevBuf,
}

struct TextBlock {
    qkv: Part,
    qkv_b: DevBuf,
    proj: Part,
    proj_b: DevBuf,
    gate: Part,
    gate_b: DevBuf,
    up: Part,
    up_b: DevBuf,
    down: Part,
    down_b: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    norm1: DevBuf,
    norm2: DevBuf,
}

pub struct Dit {
    gpu: Arc<Gpu>,
    video: Part,
    video_b: DevBuf,
    text: Part,
    text_b: DevBuf,
    text_blocks: Vec<TextBlock>,
    layers: Vec<Block>,
    final_norm: DevBuf,
    final_linear: Part,
    /// the timestep's MLP on the host: (weight [3072, 256], bias), (weight [3072, 3072], bias)
    t1: (Vec<f32>, Vec<f32>),
    t2: (Vec<f32>, Vec<f32>),
    /// the shared modulation and the head's, float32 on the card: weight [N, 3072] and bias
    mod_w: DevBuf,
    mod_b: DevBuf,
    head_w: DevBuf,
    head_b: DevBuf,
    /// the RoPE bands (rope.bands: 1 / 10000^(i/16))
    bands: Vec<f32>,
    /// a block's biggest matrix expanded to half (the products below read it whole, one launch a matrix)
    wide: DevBuf,
    /// the last forwards' rotary tables and buffers by shape (the cond and the uncond branch alternate): a forward's
    /// set-up is ~10 ms of host trigonometry and allocation
    setups: std::sync::Mutex<Vec<Setup>>,
    /// seconds the host took to queue the last forward, before it waited for the GPU (f64 bits)
    queued: std::sync::atomic::AtomicU64,
    pub bytes: u64,
}

/// A step's modulation: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f float32 [3072] each, then the head's
/// shift and scale
pub struct StepMod {
    pub blocks: DevBuf,
    pub head: DevBuf,
}

/// The prompt's rows after the text embedder, once a prompt (float32 [L, 3072]) and its RoPE table
pub struct Prefix {
    pub rows: usize,
    pub x: DevBuf,
}

/// Seconds a pass spends in each op (a check's `bench` stage: the queue is drained at each mark, so the sum runs longer
/// than the pass)
pub struct Prof {
    last: std::time::Instant,
    pub acc: Vec<(&'static str, f64)>,
}

impl Prof {
    pub fn new() -> Prof {
        Prof { last: std::time::Instant::now(), acc: Vec::new() }
    }

    fn mark(&mut self, nsd: &Nsd, name: &'static str) -> Result<()> {
        nsd.wait()?;
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        match self.acc.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += dt,
            None => self.acc.push((name, dt)),
        }
        Ok(())
    }
}

impl Default for Prof {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediates kept for a check
#[derive(Default)]
pub struct Grabs {
    pub emb: Vec<f32>,
    pub txt: Vec<f32>,
    /// (block index, the whole stream after it)
    pub blocks: Vec<(usize, Vec<f32>)>,
    /// block 0 taken apart: (name, values)
    pub probe: Vec<(&'static str, Vec<f32>)>,
    /// every block's op times, when asked for
    pub prof: Option<Prof>,
}

/// y = W x + b on the host (W [n, k] row-major)
fn host_linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k).zip(b).map(|(row, bi)| row.iter().zip(x).map(|(a, c)| (*a as f64) * (*c as f64)).sum::<f64>() as f32 + bi).collect()
}

fn silu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp());
}

/// The RoPE table for tokens at `coords` (r, f, h, w): (cos, sin) per token and feature pair j (axis j / 16, band
/// j % 16), the phase in float32 as the reference makes it
pub fn rope_table(coords: &[[f32; 4]], bands: &[f32]) -> Vec<f32> {
    let mut cs = vec![0f32; coords.len() * HEAD];
    for (t, c) in coords.iter().enumerate() {
        for j in 0..HEAD / 2 {
            let phase = c[j / BANDS] * bands[j % BANDS];
            let (s, co) = (phase as f64).sin_cos();
            cs[t * HEAD + 2 * j] = co as f32;
            cs[t * HEAD + 2 * j + 1] = s as f32;
        }
    }
    cs
}

/// The image's tokens (h x w, row-major) then the prompt's `text` tokens: their coordinates
pub fn coords(h: usize, w: usize, text: usize) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(h * w + text);
    for y in 0..h {
        for x in 0..w {
            out.push([0.0, text as f32, y as f32 - (h as f32 - 1.0) / 2.0, x as f32 - (w as f32 - 1.0) / 2.0]);
        }
    }
    for i in 0..text {
        out.push([0.0, i as f32, 0.0, 0.0]);
    }
    out
}

/// Expands the blocks' matrices to half on the card while the card has room (NS_MOSS_EXPAND: `auto` (default) keeps
/// 8 GiB free after each block, `0` never, `1` all of them): a forward then skips the expansion, ~9 % of it.
/// `loras`: updates merged into the expanded matrices (W += B . A, one rounding to half), which need every block
/// expanded; one that matches no block matrix is an error
fn expand(mut layers: Vec<Block>, nsd: &Nsd, loras: &[Delta], log: &mut dyn FnMut(String)) -> Result<Vec<Block>> {
    const RESERVE: u64 = 8 << 30;
    // what a LoRA's merge leaves free, as the card's guard: 1.5 GiB
    const KEEP: u64 = 3 << 29;
    let mut mode = std::env::var("NS_MOSS_EXPAND").unwrap_or_else(|_| "auto".into());
    if !loras.is_empty() {
        if mode == "0" {
            return Err(Error("LoRAs are merged into the expanded matrices: NS_MOSS_EXPAND=0 leaves none".into()));
        }
        mode = "lora".into();
    }
    if mode == "0" {
        return Ok(layers);
    }
    let gpu = nsd.gpu.clone();
    let t0 = std::time::Instant::now();
    let mut done = 0;
    let by_key: std::collections::BTreeMap<String, Vec<&Delta>> = loras.iter().fold(Default::default(), |mut m, d| {
        m.entry(lora::key(&d.target)).or_default().push(d);
        m
    });
    let merged = std::cell::RefCell::new(std::collections::BTreeSet::new());
    // the updates a module names, added into its rows of the half matrix at `w`
    let merge = |w: *mut std::ffi::c_void, p: &Part, name: &str| -> Result<()> {
        let key = lora::key(name);
        for d in by_key.get(&key).into_iter().flatten() {
            if d.n != p.rows || d.k != p.k {
                return Err(Error(format!("LoRA {}: {}x{}, the matrix is {}x{}", d.target, d.n, d.k, p.rows, p.k)));
            }
            let up: Vec<u8> = d.b.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect();
            let down: Vec<u8> = (0..d.k).flat_map(|c| (0..d.r).map(move |j| (c, j))).flat_map(|(c, j)| f16_bits(d.a[j * d.k + c]).to_le_bytes()).collect();
            let (bd, ad) = (DevBuf::new(&gpu, up.len())?, DevBuf::new(&gpu, down.len())?);
            bd.write(0, &up)?;
            ad.write(0, &down)?;
            nsd.linear_acc(bd.ptr(), Dt::F16, d.n, d.r, ad.ptr(), d.k, w)?;
            nsd.wait()?;
            merged.borrow_mut().insert(key.clone());
        }
        Ok(())
    };
    for (i, b) in layers.iter_mut().enumerate() {
        let need = (WIDE_VALUES + 3 * DIM * DIM + DIM * DIM + DIM * FFN) as u64 * 2;
        if mode != "1" {
            let (_, free) = gpu.memory()?;
            // the quantized copies it frees count: the block's matrices
            let freed: u64 = [&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down].iter().map(|p| p.stored()).sum();
            if mode == "lora" {
                if free.is_none_or(|f| f + freed < need + KEEP) {
                    return Err(Error(format!("LoRAs need every block expanded to half: block {i} does not fit on {} (1.5 GiB kept free)", gpu.name)));
                }
            } else if free.is_some_and(|f| f + freed < need + RESERVE) || free.is_none() {
                break;
            }
        }
        let pre = format!("layers.{i}.proj.");
        let half = |parts: &[(&Part, &str)]| -> Result<DevBuf> {
            let n: usize = parts.iter().map(|(p, _)| p.rows * p.k).sum();
            let out = DevBuf::new(&gpu, n * 2)?;
            let mut at = 0;
            for (p, name) in parts {
                let w: *mut std::ffi::c_void = out.ptr().cast::<u16>().wrapping_add(at).cast();
                nsd.dequant(p.w.ptr(), p.qtype, p.rows * p.k, w, Dt::F16)?;
                merge(w, p, &format!("{pre}{name}"))?;
                at += p.rows * p.k;
            }
            nsd.wait()?;
            Ok(out)
        };
        let h = Half { qkv: half(&[(&b.q, "q"), (&b.k, "k"), (&b.v, "v")])?, o: half(&[(&b.o, "o")])?, up_gate: half(&[(&b.up_gate, "ffn_up_gate")])?,
                       down: half(&[(&b.down, "ffn_down")])? };
        // the quantized copies go: one byte stands in for each
        for p in [&mut b.q, &mut b.k, &mut b.v, &mut b.o, &mut b.up_gate, &mut b.down] {
            p.w = DevBuf::new(&gpu, 1)?;
        }
        b.half = Some(h);
        done += 1;
    }
    let missed: std::collections::BTreeSet<&str> = loras.iter().filter(|d| !merged.borrow().contains(&lora::key(&d.target))).map(|d| d.target.as_str()).collect();
    if !missed.is_empty() {
        let few: Vec<&str> = missed.iter().take(4).copied().collect();
        return Err(Error(format!("LoRA modules this engine does not merge ({} of them): {}", missed.len(), few.join(", "))));
    }
    log(format!("moss-image dit: {done} of {} blocks expanded to half on the card in {:.1} s", layers.len(), t0.elapsed().as_secs_f64()));
    if !loras.is_empty() {
        log(format!("moss-image dit: {} LoRA updates merged into {} matrices", loras.len(), merged.borrow().len()));
    }
    Ok(layers)
}

impl Dit {
    /// `loras`: updates merged into the block matrices at load
    pub fn load(f: &Gguf, nsd: &Nsd, loras: &[Delta], log: &mut dyn FnMut(String)) -> Result<Dit> {
        let gpu = nsd.gpu.clone();
        let t0 = std::time::Instant::now();
        let tensor = |name: &str| f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())));
        let part = |name: &str| Part::load(f, &gpu, name);
        let vecd = |name: &str| crate::part::vecd(f, &gpu, name);
        // a matrix as float32 on the card (the modulation linears: a few steps' rows at a time)
        let dense = |name: &str| -> Result<DevBuf> {
            let t = tensor(name)?;
            let n = t.elements() as usize;
            match t.ty {
                GType::F32 | GType::BF16 | GType::F16 => DevBuf::from_f32(&gpu, &host_vec(f, name)?),
                _ => {
                    let p = part(name)?;
                    let out = DevBuf::f32(&gpu, n)?;
                    nsd.dequant(p.w.ptr(), p.qtype, n, out.ptr(), Dt::F32)?;
                    nsd.wait()?;
                    Ok(out)
                }
            }
        };
        let count = |pre: &str| (0..).take_while(|i| f.tensor(&format!("{pre}{i}.proj.q.weight")).is_some()).count();
        let nl = count("layers.");
        if nl == 0 || f.tensor("video_embedder.weight").is_none() {
            return Err(Error(format!("{}: not MOSS-Image's DiT (no layers / video_embedder)", f.paths[0].display())));
        }
        let block = |i: usize| -> Result<Block> {
            let p = format!("layers.{i}.");
            Ok(Block {
                half: None,
                q: part(&format!("{p}proj.q.weight"))?,
                k: part(&format!("{p}proj.k.weight"))?,
                v: part(&format!("{p}proj.v.weight"))?,
                o: part(&format!("{p}proj.o.weight"))?,
                up_gate: part(&format!("{p}proj.ffn_up_gate.weight"))?,
                down: part(&format!("{p}proj.ffn_down.weight"))?,
                q_norm: vecd(&format!("{p}q_norm.weight"))?,
                k_norm: vecd(&format!("{p}k_norm.weight"))?,
                norm1: vecd(&format!("{p}norm1.weight"))?,
                norm2: vecd(&format!("{p}norm2.weight"))?,
            })
        };
        let text_block = |i: usize| -> Result<TextBlock> {
            let p = format!("text_embedder.blocks.{i}.");
            Ok(TextBlock {
                qkv: part(&format!("{p}qkv.weight"))?,
                qkv_b: vecd(&format!("{p}qkv.bias"))?,
                proj: part(&format!("{p}proj.weight"))?,
                proj_b: vecd(&format!("{p}proj.bias"))?,
                gate: part(&format!("{p}ffn_gate.weight"))?,
                gate_b: vecd(&format!("{p}ffn_gate.bias"))?,
                up: part(&format!("{p}ffn_up.weight"))?,
                up_b: vecd(&format!("{p}ffn_up.bias"))?,
                down: part(&format!("{p}ffn_down.weight"))?,
                down_b: vecd(&format!("{p}ffn_down.bias"))?,
                q_norm: vecd(&format!("{p}q_norm.weight"))?,
                k_norm: vecd(&format!("{p}k_norm.weight"))?,
                norm1: vecd(&format!("{p}norm1.weight"))?,
                norm2: vecd(&format!("{p}norm2.weight"))?,
            })
        };
        let bands = host_vec(f, "rope.bands")?;
        if bands.len() != BANDS {
            return Err(Error(format!("rope.bands has {} values, not {BANDS}", bands.len())));
        }
        let dit = Dit {
            gpu: gpu.clone(),
            video: part("video_embedder.weight")?,
            video_b: vecd("video_embedder.bias")?,
            text: part("text_embedder.proj.weight")?,
            text_b: vecd("text_embedder.proj.bias")?,
            text_blocks: (0..2).map(text_block).collect::<Result<_>>()?,
            layers: expand((0..nl).map(block).collect::<Result<_>>()?, nsd, loras, log)?,
            final_norm: vecd("final_norm_video.weight")?,
            final_linear: part("final_linear_video.weight")?,
            t1: (host_vec(f, "time_embedding.0.weight")?, host_vec(f, "time_embedding.0.bias")?),
            t2: (host_vec(f, "time_embedding.2.weight")?, host_vec(f, "time_embedding.2.bias")?),
            mod_w: dense("shared_block_modulation.weight")?,
            mod_b: vecd("shared_block_modulation.bias")?,
            head_w: dense("video_head_modulation.weight")?,
            head_b: vecd("video_head_modulation.bias")?,
            bands,
            wide: DevBuf::new(&gpu, WIDE_VALUES * 2)?,
            queued: 0.into(),
            setups: Default::default(),
            bytes: f.tensors.iter().map(|t| t.bytes).sum(),
        };
        log(format!("moss-image dit: {nl} blocks on {} in {:.1} s", gpu.name, t0.elapsed().as_secs_f64()));
        Ok(dit)
    }

    /// The silo's passes, when it is built and this GPU runs them at the DiT's sizes
    fn silo(&self) -> Result<Option<&'static ffi::Silo>> {
        // SAFETY: a live GPU handle; plain integers.
        Ok(ffi::silo()?.filter(|s| unsafe { (s.supported)(self.gpu.raw(), DIM as i64, HEAD as i64) } == 1))
    }

    /// Seconds the host took to queue the last forward (it then waited for the GPU)
    pub fn queued(&self) -> f64 {
        f64::from_bits(self.queued.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The timestep's embedding for `sigma`: the scheduler's timestep (sigma x 1000, float32) as the sinusoid
    /// [cos | sin] of 256 (float64), Linear, SiLU, Linear
    fn t_emb(&self, sigma: f32) -> Vec<f32> {
        let t = (sigma * 1000.0) as f64;
        let mut e = vec![0f32; FREQ];
        for i in 0..FREQ / 2 {
            let a = t * 10000f64.powf(-(i as f64) / (FREQ / 2) as f64);
            e[i] = a.cos() as f32;
            e[FREQ / 2 + i] = a.sin() as f32;
        }
        let mut h = host_linear(&self.t1.0, &self.t1.1, &e);
        silu(&mut h);
        host_linear(&self.t2.0, &self.t2.1, &h)
    }

    /// Each step's modulations for the levels `sigmas` (the steps run: all but the last level)
    pub fn mods(&self, nsd: &Nsd, sigmas: &[f32]) -> Result<Vec<StepMod>> {
        let gpu = &self.gpu;
        let steps = sigmas.len().saturating_sub(1);
        let embs: Vec<Vec<f32>> = sigmas[..steps].iter().map(|s| self.t_emb(*s)).collect();
        let e = DevBuf::from_f32(gpu, &embs.concat())?;
        let blocks = DevBuf::f32(gpu, steps * 6 * DIM)?;
        let head = DevBuf::f32(gpu, steps * 2 * DIM)?;
        // a step a product: one batch of rows is not summed in the same order from call to call (~1e-6, which the
        // sampler's rounding then turns into a different picture each run)
        for i in 0..steps {
            let row = |b: &DevBuf, w: usize| -> *mut std::ffi::c_void { b.ptr().cast::<f32>().wrapping_add(i * w).cast() };
            nsd.linear(row(&e, DIM), Dt::F32, 1, DIM, self.mod_w.ptr(), 6 * DIM, self.mod_b.fp().cast(), row(&blocks, 6 * DIM), Dt::F32)?;
            nsd.linear(row(&e, DIM), Dt::F32, 1, DIM, self.head_w.ptr(), 2 * DIM, self.head_b.fp().cast(), row(&head, 2 * DIM), Dt::F32)?;
        }
        nsd.wait()?;
        let (bv, hv) = (blocks.to_f32()?, head.to_f32()?);
        (0..steps).map(|i| Ok(StepMod { blocks: DevBuf::from_f32(gpu, &bv[i * 6 * DIM..(i + 1) * 6 * DIM])?, head: DevBuf::from_f32(gpu, &hv[i * 2 * DIM..(i + 1) * 2 * DIM])? })).collect()
    }

    /// The prompt's hidden states (float32 [L, 4096]) through the text embedder: Linear and two text blocks
    pub fn prefix(&self, nsd: &Nsd, ctx: &[f32], grabs: Option<&mut Grabs>) -> Result<Prefix> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let l = ctx.len() / TEXT_DIM;
        let cf = DevBuf::from_f32(gpu, ctx)?;
        let ch = DevBuf::new(gpu, ctx.len() * 2)?;
        // SAFETY: l x 4096 floats in, as many halves out.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), cf.fp(), ch.ptr(), ctx.len() as i64) }, "to half")?;
        let x = DevBuf::f32(gpu, l * DIM)?;
        self.text.apply(nsd, ch.ptr(), l, Some(&self.text_b), x.ptr(), Dt::F32, 0)?;
        // the text blocks' rope: the prompt's rows alone, at (0, i, 0, 0)
        let cs = DevBuf::from_f32(gpu, &rope_table(&(0..l).map(|i| [0.0, i as f32, 0.0, 0.0]).collect::<Vec<_>>(), &self.bands))?;
        let t = Tmp::new(gpu, l, true)?;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        for b in &self.text_blocks {
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, b.norm1.ptr(), EPS, none(), none(), none(), t.h.ptr(), Dt::F16)?;
            b.qkv.apply(nsd, t.h.ptr(), l, Some(&b.qkv_b), t.qkv.ptr(), Dt::F16, 0)?;
            nsd.rms_rope(t.qkv.ptr(), Dt::F16, l, HEADS, HEAD, 3 * DIM, b.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, l, HEADS, HEAD, 3 * DIM, b.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.attention_qk(t.qkv.ptr(), l, 3 * DIM, half_at(&t.qkv, DIM), half_at(&t.qkv, 2 * DIM), l, 3 * DIM, Dt::F16, HEADS, HEAD, t.att.ptr(), Dt::F16)?;
            b.proj.apply(nsd, t.att.ptr(), l, Some(&b.proj_b), t.of.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, t.of.ptr(), Dt::F32, none(), none())?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, l, DIM, b.norm2.ptr(), EPS, none(), none(), none(), t.h.ptr(), Dt::F16)?;
            b.up.apply(nsd, t.h.ptr(), l, Some(&b.up_b), t.hid.ptr(), Dt::F32, 0)?;
            b.gate.apply(nsd, t.h.ptr(), l, Some(&b.gate_b), t.gate.ptr(), Dt::F32, 0)?;
            // SAFETY: l x 8192 floats each way, the activation l x 8192 halves.
            ffi::check(unsafe { (k.situ)(gpu.raw(), t.hid.fp(), FFN as i64, t.gate.fp(), FFN as i64, l as i64, FFN as i64, t.act.ptr(), FFN as i64) }, "situ")?;
            b.down.apply(nsd, t.act.ptr(), l, Some(&b.down_b), t.of.ptr(), Dt::F32, 0)?;
            nsd.gate_add(x.ptr(), Dt::F32, l, DIM, t.of.ptr(), Dt::F32, none(), none())?;
        }
        nsd.wait()?;
        if let Some(g) = grabs {
            g.txt = x.to_f32()?;
        }
        Ok(Prefix { rows: l, x })
    }

    /// One block over x float32 [m, 3072] in place; `md`: the step's modulation
    #[allow(clippy::too_many_arguments)]
    fn block(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, cs: &DevBuf, md: &StepMod, t: &Tmp, mut probe: Option<&mut Vec<(&'static str, Vec<f32>)>>,
             mut prof: Option<&mut Prof>, carry: &mut bool, flush: bool) -> Result<()> {
        let mut lap = |name: &'static str| -> Result<()> { prof.as_deref_mut().map_or(Ok(()), |p| p.mark(nsd, name)) };
        lap("(other)")?;
        let gpu = &self.gpu;
        let k = ffi::api()?;
        // a probe's snapshot of a half buffer (n values) or a float32 one
        let snap = |probe: &mut Option<&mut Vec<(&'static str, Vec<f32>)>>, name: &'static str, buf: &DevBuf, n: usize, half: bool| -> Result<()> {
            let Some(p) = probe.as_deref_mut() else { return Ok(()) };
            nsd.wait()?;
            if half {
                let f = DevBuf::f32(gpu, n)?;
                // SAFETY: n halves in, n floats out.
                ffi::check(unsafe { (k.to_float)(gpu.raw(), buf.ptr(), f.fp(), n as i64) }, "probe")?;
                nsd.wait()?;
                p.push((name, f.to_f32()?));
            } else {
                p.push((name, buf.to_f32()?[..n].to_vec()));
            }
            Ok(())
        };
        snap(&mut probe, "x0", x, m * DIM, false)?;
        let qw = 3 * DIM;
        let half_at = |buf: &DevBuf, n: usize| -> *mut std::ffi::c_void { buf.ptr().cast::<u16>().wrapping_add(n).cast() };
        let at = |i: usize| -> *const std::ffi::c_void { md.blocks.fp().wrapping_add(i * DIM).cast() };
        let (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f) = (at(0), at(1), at(2), at(3), at(4), at(5));
        let silo = self.silo()?;
        let gp = |p: *const std::ffi::c_void| p;
        if let Some(sl) = silo {
            // the previous block's product joins x here (gated), in the same pass as this norm
            let (r, gt) = if *carry { (t.of.fp().cast_const().cast(), gp(gate_f)) } else { (std::ptr::null(), std::ptr::null()) };
            // SAFETY: x and of are m x 3072 floats, the modulation rows 3072 floats, h m x 3072 halves.
            ffi::check(unsafe { (sl.add_norm)(gpu.raw(), x.fp(), r, gt, m as i64, DIM as i64, b.norm1.ptr(), EPS, scale_a, shift_a, t.h.ptr()) }, "add norm")?;
            *carry = false;
        } else {
            if *carry {
                nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, none(), gate_f)?;
                *carry = false;
            }
            nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm1.ptr(), EPS, t.rows.ptr(), scale_a, shift_a, t.h.ptr(), Dt::F16)?;
        }
        lap("norm")?;
        snap(&mut probe, "y1", &t.h, m * DIM, true)?;
        // q, k and v side by side (expanded now or at load): one product writes the packed rows
        let qkv_w = match &b.half {
            Some(h) => h.qkv.ptr(),
            None => {
                for (i, p) in [&b.q, &b.k, &b.v].into_iter().enumerate() {
                    nsd.dequant(p.w.ptr(), p.qtype, p.rows * p.k, half_at(&self.wide, i * DIM * DIM), Dt::F16)?;
                }
                self.wide.ptr()
            }
        };
        nsd.linear(t.h.ptr(), Dt::F16, m, DIM, qkv_w, qw, none(), t.qkv.ptr(), Dt::F16)?;
        lap("qkv gemm")?;
        if probe.is_some() {
            // v before anything else (a probe's snapshot is of the packed rows: v is column block 2)
            nsd.wait()?;
            let f = DevBuf::f32(gpu, m * qw)?;
            // SAFETY: m x 3 x 3072 halves in, as many floats out.
            ffi::check(unsafe { (k.to_float)(gpu.raw(), t.qkv.ptr(), f.fp(), (m * qw) as i64) }, "probe")?;
            nsd.wait()?;
            let all = f.to_f32()?;
            let col = |c: usize| -> Vec<f32> { (0..m).flat_map(|r| all[r * qw + c * DIM..r * qw + (c + 1) * DIM].to_vec()).collect() };
            if let Some(p) = probe.as_deref_mut() {
                p.push(("v", col(2)));
            }
        }
        if let Some(sl) = silo {
            // SAFETY: m rows of 3 x 3072 halves (q at column 0, k at 3072), the rotary table m x 128 floats.
            ffi::check(unsafe { (sl.qk_rope)(gpu.raw(), t.qkv.ptr(), m as i64, HEADS as i64, HEAD as i64, qw as i64, DIM as i64, b.q_norm.ptr(), b.k_norm.ptr(), EPS, cs.ptr()) }, "qk rope")?;
        } else {
            nsd.rms_rope(t.qkv.ptr(), Dt::F16, m, HEADS, HEAD, qw, b.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
            nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, m, HEADS, HEAD, qw, b.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
        }
        lap("qk norm rope")?;
        nsd.attention_qk(t.qkv.ptr(), m, qw, half_at(&t.qkv, DIM), half_at(&t.qkv, 2 * DIM), m, qw, Dt::F16, HEADS, HEAD, t.att.ptr(), Dt::F16)?;
        lap("attention")?;
        snap(&mut probe, "att", &t.att, m * DIM, true)?;
        b.o.apply_whole(nsd, b.half.as_ref().map(|h| &h.o), &self.wide, t.att.ptr(), m, t.of.ptr(), Dt::F32)?;
        lap("o gemm")?;
        if let (Some(sl), false) = (silo, probe.is_some()) {
            // SAFETY: as the first norm's.
            ffi::check(unsafe { (sl.add_norm)(gpu.raw(), x.fp(), t.of.fp().cast_const().cast(), gate_a, m as i64, DIM as i64, b.norm2.ptr(), EPS, scale_f, shift_f, t.h.ptr()) }, "add norm")?;
            lap("norm")?;
        } else {
            nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, none(), gate_a)?;
            lap("gate add")?;
            snap(&mut probe, "x1", x, m * DIM, false)?;
            nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm2.ptr(), EPS, t.rows.ptr(), scale_f, shift_f, t.h.ptr(), Dt::F16)?;
            lap("norm")?;
        }
        if let Some(sl) = silo {
            // (hidden | gate) in half side by side, the soft-clipped product straight from them
            b.up_gate.apply_whole(nsd, b.half.as_ref().map(|h| &h.up_gate), &self.wide, t.h.ptr(), m, t.hid.ptr(), Dt::F16)?;
            lap("up gate gemm")?;
            let hp = t.hid.ptr().cast_const();
            // SAFETY: m rows of 2 x 8192 halves (hidden first, gate 8192 later); the activation m x 8192 halves.
            ffi::check(unsafe { (sl.situ)(gpu.raw(), hp, hp.cast::<u16>().wrapping_add(FFN).cast(), 2 * FFN as i64, m as i64, FFN as i64, t.act.ptr(), FFN as i64) }, "situ")?;
        } else {
            // (hidden | gate) float32 side by side, then the soft-clipped product in half
            b.up_gate.apply_whole(nsd, b.half.as_ref().map(|h| &h.up_gate), &self.wide, t.h.ptr(), m, t.hid.ptr(), Dt::F32)?;
            lap("up gate gemm")?;
            // SAFETY: m rows of 2 x 8192 floats (hidden first, gate 8192 later); the activation m x 8192 halves.
            ffi::check(unsafe { (k.situ)(gpu.raw(), t.hid.fp(), 2 * FFN as i64, t.hid.fp().wrapping_add(FFN), 2 * FFN as i64, m as i64, FFN as i64, t.act.ptr(), FFN as i64) }, "situ")?;
        }
        lap("situ")?;
        snap(&mut probe, "act", &t.act, m * FFN, true)?;
        b.down.apply_whole(nsd, b.half.as_ref().map(|h| &h.down), &self.wide, t.act.ptr(), m, t.of.ptr(), Dt::F32)?;
        lap("down gemm")?;
        if silo.is_some() && !flush && probe.is_none() {
            *carry = true;      // joins x in the next norm
        } else {
            nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, none(), gate_f)?;
        }
        lap("gate add")?;
        snap(&mut probe, "x2", x, m * DIM, false)
    }

    /// Block 0's fused up / gate product at `m` rows, timed three ways (a check's `bench`): the whole call, the
    /// weights' expansion to half alone, and the half product on the expanded copy
    pub fn gemm_probe(&self, nsd: &Nsd, m: usize) -> Result<Vec<(&'static str, f64)>> {
        let gpu = &self.gpu;
        let b = &self.layers[0];
        if b.half.is_some() {
            // the quantized copies are gone (NS_MOSS_EXPAND=0 to time them)
            return Ok(Vec::new());
        }
        let (n, kk) = (b.up_gate.rows, b.up_gate.k);
        let x = DevBuf::new(gpu, m * kk * 2)?;
        x.fill(0x3c)?;
        let out = DevBuf::new(gpu, m * n * 2)?;
        let wh = DevBuf::new(gpu, n * kk * 2)?;
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            f()?;
            nsd.wait()?;
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                f()?;
            }
            nsd.wait()?;
            Ok(t0.elapsed().as_secs_f64() / 5.0)
        };
        let whole = time(&mut || b.up_gate.apply(nsd, x.ptr(), m, None, out.ptr(), Dt::F16, 0))?;
        let deq = time(&mut || nsd.dequant(b.up_gate.w.ptr(), b.up_gate.qtype, n * kk, wh.ptr(), Dt::F16))?;
        let prod = time(&mut || nsd.linear(x.ptr(), Dt::F16, m, kk, wh.ptr(), n, none(), out.ptr(), Dt::F16))?;
        // the down product (8192 -> 3072) on its expanded copy: float32 out and half out
        let d = &b.down;
        let xd = DevBuf::new(gpu, m * d.k * 2)?;
        xd.fill(0x3c)?;
        let od = DevBuf::new(gpu, m * d.rows * 4)?;
        nsd.dequant(d.w.ptr(), d.qtype, d.rows * d.k, wh.ptr(), Dt::F16)?;
        let down32 = time(&mut || nsd.linear(xd.ptr(), Dt::F16, m, d.k, wh.ptr(), d.rows, none(), od.ptr(), Dt::F32))?;
        let down16 = time(&mut || nsd.linear(xd.ptr(), Dt::F16, m, d.k, wh.ptr(), d.rows, none(), od.ptr(), Dt::F16))?;
        Ok(vec![("whole call", whole), ("expansion alone", deq), ("half product alone", prod), ("down, f32 out", down32), ("down, half out", down16)])
    }

    /// The silo's fused passes at `m` rows, each timed alone with the effective bandwidth (a check's `bench`): (name,
    /// seconds, bytes moved)
    pub fn silo_probe(&self, nsd: &Nsd, m: usize) -> Result<Vec<(&'static str, f64, f64)>> {
        let Some(sl) = self.silo()? else { return Ok(Vec::new()) };
        let gpu = &self.gpu;
        let b = &self.layers[0];
        let t = Tmp::new(gpu, m, false)?;
        let x = DevBuf::f32(gpu, m * DIM)?;
        x.fill(0)?;
        let cs = DevBuf::from_f32(gpu, &rope_table(&coords(1, m, 0), &self.bands))?;
        let ones = DevBuf::from_f32(gpu, &vec![1f32; DIM])?;
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            f()?;
            nsd.wait()?;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                f()?;
            }
            nsd.wait()?;
            Ok(t0.elapsed().as_secs_f64() / 10.0)
        };
        let (mf, df, ff) = (m as f64, DIM as f64, FFN as f64);
        let mut out = Vec::new();
        // a forward's set-up outside the blocks: the rotary table (host trigonometry, upload) and the buffers
        let t0 = std::time::Instant::now();
        let side = (m as f64).sqrt() as usize;
        let tab = DevBuf::from_f32(gpu, &rope_table(&coords(side, side, 32), &self.bands))?;
        out.push(("set-up: rope table", t0.elapsed().as_secs_f64(), 0.0));
        let t0 = std::time::Instant::now();
        let tmp = Tmp::new(gpu, m, false)?;
        drop((tab, tmp));
        out.push(("set-up: buffers", t0.elapsed().as_secs_f64(), 0.0));
        let g = gpu.raw();
        let tn = time(&mut || ffi::check(unsafe { (sl.add_norm)(g, x.fp(), t.of.fp().cast_const().cast(), ones.ptr().cast_const(), m as i64, DIM as i64, b.norm1.ptr(), EPS, ones.ptr().cast_const(), ones.ptr().cast_const(), t.h.ptr()) }, "add norm"))?;
        out.push(("add + norm + modulation", tn, mf * df * (4.0 * 3.0 + 2.0)));
        let tq = time(&mut || ffi::check(unsafe { (sl.qk_rope)(g, t.qkv.ptr(), m as i64, HEADS as i64, HEAD as i64, (3 * DIM) as i64, DIM as i64, b.q_norm.ptr(), b.k_norm.ptr(), EPS, cs.ptr()) }, "qk rope"))?;
        out.push(("q k norm + rope", tq, mf * df * 2.0 * 2.0 * 2.0));
        let hp = t.hid.ptr().cast_const();
        let ts = time(&mut || ffi::check(unsafe { (sl.situ)(g, hp, hp.cast::<u16>().wrapping_add(FFN).cast(), 2 * FFN as i64, m as i64, FFN as i64, t.act.ptr(), FFN as i64) }, "situ"))?;
        out.push(("soft-clipped gate", ts, mf * ff * (2.0 * 2.0 + 2.0)));
        Ok(out)
    }

    /// The model's output for the image's tokens `lat` (float32 [h * w, 128], row-major tokens) at a step with
    /// modulation `md`, the prompt after `pre`, the blocks in `skip` left out: float32 [h * w, 128] (the reference's
    /// output transposed)
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), md: &StepMod, skip: &[usize], mut grabs: Option<&mut Grabs>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
        let started = std::time::Instant::now();
        let n = hw.0 * hw.1;
        let c = pre.rows;
        let s = n + c;
        let x = DevBuf::f32(gpu, s * DIM)?;
        let lh = DevBuf::new(gpu, n * LATENT * 2)?;
        // SAFETY: n x 128 floats in, as many halves out.
        ffi::check(unsafe { (k.to_half)(gpu.raw(), lat.fp(), lh.ptr(), (n * LATENT) as i64) }, "to half")?;
        self.video.apply(nsd, lh.ptr(), n, Some(&self.video_b), x.ptr(), Dt::F32, 0)?;
        if let Some(g) = grabs.as_deref_mut() {
            nsd.wait()?;
            g.emb = x.to_f32()?[..n * DIM].to_vec();
        }
        // [image | text]
        x.copy_within(n * DIM * 4, &pre.x, 0, c * DIM * 4)?;
        let key = (hw.0, hw.1, c);
        let cached = {
            let mut v = self.setups.lock().unwrap();
            v.iter().position(|e| e.key == key).map(|i| v.remove(i))
        };
        let Setup { cs, t, .. } = match cached {
            Some(e) => e,
            None => Setup { key, cs: DevBuf::from_f32(gpu, &rope_table(&coords(hw.0, hw.1, c), &self.bands))?, t: Tmp::new(gpu, s, false)? },
        };
        if let Some(g) = grabs.as_deref_mut() {
            g.probe.push(("mod", md.blocks.to_f32()?));
            g.probe.push(("head", md.head.to_f32()?));
            g.probe.push(("rope", cs.to_f32()?));
        }
        // whether the last block's product is still to be added to x (gated): the next norm does it
        let mut carry = false;
        for (i, b) in self.layers.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            let snap = grabs.is_some() && [0, 1, 21, 22, 41].contains(&i);
            let (pr, pf) = match grabs.as_deref_mut() {
                Some(g) => (if i == 0 { Some(&mut g.probe) } else { None }, g.prof.as_mut()),
                None => (None, None),
            };
            self.block(nsd, b, &x, s, &cs, md, &t, pr, pf, &mut carry, snap)?;
            if let Some(g) = grabs.as_deref_mut() {
                if [0, 1, 21, 22, 41].contains(&i) {
                    nsd.wait()?;
                    g.blocks.push((i, x.to_f32()?));
                }
            }
        }
        // the head on the image's rows: rmsnorm * (1 + scale) + shift, then Linear 3072 -> 128
        let (head_scale, head_shift) = (md.head.fp().wrapping_add(DIM).cast_const().cast(), md.head.ptr().cast_const());
        match self.silo()? {
            Some(sl) if carry => {
                // the last block's product joins x for the whole token run, then the head's norm on the image's rows
                let gate_f = md.blocks.fp().wrapping_add(5 * DIM).cast_const().cast();
                // SAFETY: x and of are s x 3072 floats; the head's rows are the image's first n.
                ffi::check(unsafe { (sl.add_norm)(gpu.raw(), x.fp(), t.of.fp().cast_const().cast(), gate_f, s as i64, DIM as i64, self.final_norm.ptr(), EPS, head_scale, head_shift, t.h.ptr()) }, "add norm")?;
            }
            _ => nsd.rms_norm_mod(x.ptr(), Dt::F32, n, DIM, self.final_norm.ptr(), EPS, t.rows.ptr(), head_scale, head_shift, t.h.ptr(), Dt::F16)?,
        }
        let v = DevBuf::f32(gpu, n * LATENT)?;
        self.final_linear.apply(nsd, t.h.ptr(), n, None, v.ptr(), Dt::F32, 0)?;
        // kept for the next forward of this shape, while the buffers are small (two shapes at most)
        if s <= MAX_CACHED_ROWS {
            let mut v = self.setups.lock().unwrap();
            v.push(Setup { key, cs, t });
            if v.len() > 2 {
                v.remove(0);
            }
        }
        self.queued.store(started.elapsed().as_secs_f64().to_bits(), std::sync::atomic::Ordering::Relaxed);
        nsd.wait()?;
        Ok(v)
    }
}

const PAD_ROWS: usize = 256;
/// Rows (tokens) up to which a forward's buffers are kept for the next one (4128 at 1024 x 1024: ~1 GB each)
const MAX_CACHED_ROWS: usize = 8192;

/// A forward's rotary table and buffers, for the shape `key` (picture rows, picture columns, prompt tokens)
struct Setup {
    key: (usize, usize, usize),
    cs: DevBuf,
    t: Tmp,
}
/// Values of the biggest block matrix (the fused up / gate: 16384 x 3072)
const WIDE_VALUES: usize = 2 * FFN * DIM;

/// A pass's buffers, for up to `rows` tokens
struct Tmp {
    /// the modulation row every token uses (row 0): int32 zeros
    rows: DevBuf,
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    of: DevBuf,
    hid: DevBuf,
    gate: DevBuf,
    act: DevBuf,
}

impl Tmp {
    fn new(gpu: &Arc<Gpu>, rows: usize, gate: bool) -> Result<Tmp> {
        let zero = DevBuf::new(gpu, rows * 4)?;
        zero.write(0, &vec![0u8; rows * 4])?;
        // zeroed, and a block of rows past the end: a kernel that reads a little beyond the last row meets zeros
        let rows = rows + PAD_ROWS;
        let t = Tmp {
            rows: zero,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            of: DevBuf::f32(gpu, rows * DIM)?,
            // the fused (hidden | gate) rows of a block, or the text blocks' hidden alone
            hid: DevBuf::f32(gpu, rows * 2 * FFN)?,
            // the text blocks' second half (the DiT's blocks have the fused matrix)
            gate: DevBuf::f32(gpu, if gate { rows * FFN } else { 1 })?,
            act: DevBuf::new(gpu, rows * FFN * 2)?,
        };
        for b in [&t.h, &t.qkv, &t.att, &t.of, &t.hid, &t.gate, &t.act] {
            b.fill(0)?;
        }
        gpu.sync()?;
        Ok(t)
    }
}

/// Latents [128, H, W] (channel-major) as tokens [H * W, 128]
pub fn tokens(lat: &[f32], hw: usize) -> Vec<f32> {
    let mut out = vec![0f32; lat.len()];
    for c in 0..LATENT {
        for p in 0..hw {
            out[p * LATENT + c] = lat[c * hw + p];
        }
    }
    out
}

/// The inverse: tokens [H * W, 128] to latents [128, H, W]
pub fn planes(tok: &[f32], hw: usize) -> Vec<f32> {
    let mut out = vec![0f32; tok.len()];
    for c in 0..LATENT {
        for p in 0..hw {
            out[c * hw + p] = tok[p * LATENT + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn tokens_go_and_come_back() {
        let hw = 6;
        let lat: Vec<f32> = (0..super::LATENT * hw).map(|i| i as f32).collect();
        let t = super::tokens(&lat, hw);
        // token 1, channel 2 is plane 2's second value
        assert_eq!(t[super::LATENT + 2], (2 * hw + 1) as f32);
        assert_eq!(super::planes(&t, hw), lat);
    }

    #[test]
    fn rope_axes_follow_the_coordinates() {
        let bands: Vec<f32> = (0..16).map(|i| 1.0 / 10000f32.powf(i as f32 / 16.0)).collect();
        // an image token at (0, 5, -1.5, 2): the frame axis (pairs 0..16) is still, the text axis turns by 5 x band
        let cs = super::rope_table(&[[0.0, 5.0, -1.5, 2.0]], &bands);
        assert!((cs[0] - 1.0).abs() < 1e-7 && cs[1].abs() < 1e-7);
        assert!((cs[2 * 16] - 5f32.cos()).abs() < 1e-6 && (cs[2 * 16 + 1] - 5f32.sin()).abs() < 1e-6);
        // the height axis, band 0: -1.5
        assert!((cs[2 * 32] - 1.5f32.cos()).abs() < 1e-6 && (cs[2 * 32 + 1] + 1.5f32.sin()).abs() < 1e-6);
        let c = super::coords(2, 2, 3);
        assert_eq!(c.len(), 7);
        assert_eq!(c[0], [0.0, 3.0, -0.5, -0.5]);
        assert_eq!(c[6], [0.0, 2.0, 0.0, 0.0]);
    }
}
