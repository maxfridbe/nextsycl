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
use nextsycl_gguf::{GType, Gguf};

use crate::ffi;
use crate::part::{host_vec, Part};

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

struct Block {
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
    pub bytes: u64,
}

/// A step's modulation: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f float32 [3072] each, then the head's
/// shift and scale
pub struct StepMod {
    blocks: DevBuf,
    head: DevBuf,
}

/// The prompt's rows after the text embedder, once a prompt (float32 [L, 3072]) and its RoPE table
pub struct Prefix {
    pub rows: usize,
    pub x: DevBuf,
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

impl Dit {
    pub fn load(f: &Gguf, nsd: &Nsd, log: &mut dyn FnMut(String)) -> Result<Dit> {
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
            layers: (0..nl).map(block).collect::<Result<_>>()?,
            final_norm: vecd("final_norm_video.weight")?,
            final_linear: part("final_linear_video.weight")?,
            t1: (host_vec(f, "time_embedding.0.weight")?, host_vec(f, "time_embedding.0.bias")?),
            t2: (host_vec(f, "time_embedding.2.weight")?, host_vec(f, "time_embedding.2.bias")?),
            mod_w: dense("shared_block_modulation.weight")?,
            mod_b: vecd("shared_block_modulation.bias")?,
            head_w: dense("video_head_modulation.weight")?,
            head_b: vecd("video_head_modulation.bias")?,
            bands,
            bytes: f.tensors.iter().map(|t| t.bytes).sum(),
        };
        log(format!("moss-image dit: {nl} blocks on {} in {:.1} s", gpu.name, t0.elapsed().as_secs_f64()));
        Ok(dit)
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
        nsd.linear(e.ptr(), Dt::F32, steps, DIM, self.mod_w.ptr(), 6 * DIM, self.mod_b.fp().cast(), blocks.ptr(), Dt::F32)?;
        let head = DevBuf::f32(gpu, steps * 2 * DIM)?;
        nsd.linear(e.ptr(), Dt::F32, steps, DIM, self.head_w.ptr(), 2 * DIM, self.head_b.fp().cast(), head.ptr(), Dt::F32)?;
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
        let t = Tmp::new(gpu, l)?;
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
    fn block(&self, nsd: &Nsd, b: &Block, x: &DevBuf, m: usize, cs: &DevBuf, md: &StepMod, t: &Tmp, mut probe: Option<&mut Vec<(&'static str, Vec<f32>)>>) -> Result<()> {
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
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm1.ptr(), EPS, t.rows.ptr(), scale_a, shift_a, t.h.ptr(), Dt::F16)?;
        snap(&mut probe, "y1", &t.h, m * DIM, true)?;
        for (i, p) in [&b.q, &b.k, &b.v].into_iter().enumerate() {
            p.apply(nsd, t.h.ptr(), m, None, half_at(&t.qkv, i * DIM), Dt::F16, qw)?;
        }
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
        nsd.rms_rope(t.qkv.ptr(), Dt::F16, m, HEADS, HEAD, qw, b.q_norm.ptr(), EPS, cs.ptr(), HEAD)?;
        nsd.rms_rope(half_at(&t.qkv, DIM), Dt::F16, m, HEADS, HEAD, qw, b.k_norm.ptr(), EPS, cs.ptr(), HEAD)?;
        nsd.attention_qk(t.qkv.ptr(), m, qw, half_at(&t.qkv, DIM), half_at(&t.qkv, 2 * DIM), m, qw, Dt::F16, HEADS, HEAD, t.att.ptr(), Dt::F16)?;
        snap(&mut probe, "att", &t.att, m * DIM, true)?;
        b.o.apply(nsd, t.att.ptr(), m, None, t.of.ptr(), Dt::F32, 0)?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, none(), gate_a)?;
        snap(&mut probe, "x1", x, m * DIM, false)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, m, DIM, b.norm2.ptr(), EPS, t.rows.ptr(), scale_f, shift_f, t.h.ptr(), Dt::F16)?;
        // (hidden | gate) float32 side by side, then the soft-clipped product in half
        b.up_gate.apply(nsd, t.h.ptr(), m, None, t.hid.ptr(), Dt::F32, 2 * FFN)?;
        // SAFETY: m rows of 2 x 8192 floats (hidden first, gate 8192 later); the activation m x 8192 halves.
        ffi::check(unsafe { (k.situ)(gpu.raw(), t.hid.fp(), 2 * FFN as i64, t.hid.fp().wrapping_add(FFN), 2 * FFN as i64, m as i64, FFN as i64, t.act.ptr(), FFN as i64) }, "situ")?;
        snap(&mut probe, "act", &t.act, m * FFN, true)?;
        b.down.apply(nsd, t.act.ptr(), m, None, t.of.ptr(), Dt::F32, 0)?;
        nsd.gate_add(x.ptr(), Dt::F32, m, DIM, t.of.ptr(), Dt::F32, none(), gate_f)?;
        snap(&mut probe, "x2", x, m * DIM, false)
    }

    /// The model's output for the image's tokens `lat` (float32 [h * w, 128], row-major tokens) at a step with
    /// modulation `md`, the prompt after `pre`, the blocks in `skip` left out: float32 [h * w, 128] (the reference's
    /// output transposed)
    #[allow(clippy::too_many_arguments)]
    pub fn velocity(&self, nsd: &Nsd, pre: &Prefix, lat: &DevBuf, hw: (usize, usize), md: &StepMod, skip: &[usize], mut grabs: Option<&mut Grabs>) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api()?;
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
        let cs = DevBuf::from_f32(gpu, &rope_table(&coords(hw.0, hw.1, c), &self.bands))?;
        let t = Tmp::new(gpu, s)?;
        if let Some(g) = grabs.as_deref_mut() {
            g.probe.push(("mod", md.blocks.to_f32()?));
            g.probe.push(("head", md.head.to_f32()?));
            g.probe.push(("rope", cs.to_f32()?));
        }
        for (i, b) in self.layers.iter().enumerate() {
            if skip.contains(&i) {
                continue;
            }
            let pr = if i == 0 { grabs.as_deref_mut().map(|g| &mut g.probe) } else { None };
            self.block(nsd, b, &x, s, &cs, md, &t, pr)?;
            if let Some(g) = grabs.as_deref_mut() {
                if [0, 1, 21, 22, 41].contains(&i) {
                    nsd.wait()?;
                    g.blocks.push((i, x.to_f32()?));
                }
            }
        }
        // the head on the image's rows: rmsnorm * (1 + scale) + shift, then Linear 3072 -> 128
        nsd.rms_norm_mod(x.ptr(), Dt::F32, n, DIM, self.final_norm.ptr(), EPS, t.rows.ptr(), md.head.fp().wrapping_add(DIM).cast(), md.head.ptr().cast_const(), t.h.ptr(), Dt::F16)?;
        let v = DevBuf::f32(gpu, n * LATENT)?;
        self.final_linear.apply(nsd, t.h.ptr(), n, None, v.ptr(), Dt::F32, 0)?;
        nsd.wait()?;
        Ok(v)
    }
}

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
    fn new(gpu: &Arc<Gpu>, rows: usize) -> Result<Tmp> {
        let zero = DevBuf::new(gpu, rows * 4)?;
        zero.write(0, &vec![0u8; rows * 4])?;
        Ok(Tmp {
            rows: zero,
            h: DevBuf::new(gpu, rows * DIM * 2)?,
            qkv: DevBuf::new(gpu, rows * 3 * DIM * 2)?,
            att: DevBuf::new(gpu, rows * DIM * 2)?,
            of: DevBuf::f32(gpu, rows * DIM)?,
            // the fused (hidden | gate) rows of a block, or the text blocks' hidden alone
            hid: DevBuf::f32(gpu, rows * 2 * FFN)?,
            gate: DevBuf::f32(gpu, rows * FFN)?,
            act: DevBuf::new(gpu, rows * FFN * 2)?,
        })
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
