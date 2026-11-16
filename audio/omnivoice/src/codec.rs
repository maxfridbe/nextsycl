//! The Higgs Audio v2 codec (transformers' HiggsAudioV2TokenizerModel; the tokenizer GGUF): 24 kHz sound and 25 frames a
//! second of 8 codes of 1,024.
//!
//! ```text
//!   decode  codes [8, T] -> each codebook's 64-wide row, its projection out (64 -> 1,024), summed, fc2 (1,024 -> 256)
//!           (folded at load into one product of the eight rows side by side: [T, 512] -> [T, 256])
//!           -> DAC's decoder: conv 7 (256 -> 1,024), five blocks (Snake, a transposed conv up by 8, 5, 4, 2, 3 - its
//!           output padding stride % 2 -, three residual units of dilations 1, 3, 9), Snake, conv 7 (32 -> 1), no tanh
//!   encode  a recording at 24 kHz (a whole number of 960-sample frames)
//!           -> semantic: 16 kHz (torchaudio's resampler), 160 zeros each side, HuBERT (7 convs, 12 post-norm layers),
//!              the mean of its 13 hidden states, every other frame, the semantic encoder (convs and ELU residuals)
//!           -> acoustic: DAC's encoder (conv 7, five blocks down by 8, 5, 4, 2, 3, Snake, conv 3 -> 256)
//!           -> [acoustic; semantic] (1,024) -> fc -> the residual quantizer: per codebook its projection in (1,024 ->
//!              64), the nearest of its 1,024 rows, the row's projection out taken from the residual
//! ```
//!
//! Float32 throughout, as the reference (the file's half convolutions expanded at load).

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Dt;

use crate::ops::{bytes, fp, read_f32, Ops};
use crate::weights::Weights;

pub const RATE: u32 = 24000;
pub const SEMANTIC_RATE: u32 = 16000;
/// samples a frame
pub const HOP: usize = 960;
const ARCH: &str = "omnivoice";

struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    co: usize,
    ci: usize,
    k: usize,
}

impl Conv {
    /// `name.weight` [co, ci, k] (a transposed one: [ci, co, k]), `name.bias` when the file has it
    fn load(ops: &Ops, w: &Weights, name: &str, transposed: bool) -> Result<Conv> {
        let s = w.shape(&format!("{name}.weight"))?;
        let s = match s.len() {
            3 => s,
            // [ci, k]: a kernel of one output channel (DAC's last conv: the file drops the unit dimension)
            2 if !transposed => vec![1, s[0], s[1]],
            _ => return Err(Error(format!("{name}.weight: shape {s:?}, a convolution's 3 dimensions expected"))),
        };
        let (co, ci) = if transposed { (s[1], s[0]) } else { (s[0], s[1]) };
        // a convolution without a bias gets zeros: the oneDNN wrapper (mm3's conv1d_lr) keeps one primitive a shape,
        // made with or without a bias by its first call - a biased convolution of the same shape after a bias-less one
        // (the semantic encoder's k3 convolutions) would run without its bias
        let b = if w.has(&format!("{name}.bias")) { w.dev(ops, &format!("{name}.bias"))? } else { Weights::upload(ops, &vec![0f32; co])? };
        let b = Some(b);
        Ok(Conv { w: w.dev(ops, &format!("{name}.weight"))?, b, co, ci, k: s[2] })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, ops: &Ops, x: *const f32, l: usize, stride: usize, dil: usize, pl: usize, pr: usize, out: *mut f32) -> Result<usize> {
        ops.conv(x, self.ci, l, &self.w, self.co, self.k, self.b.as_ref(), stride, dil, pl, pr, out)
    }
}

/// DAC's residual unit: x += conv1x1(snake(conv7_dil(snake(x))))
struct Res {
    a1: DevBuf,
    c1: Conv,
    a2: DevBuf,
    c2: Conv,
    dil: usize,
}

impl Res {
    fn load(ops: &Ops, w: &Weights, p: &str, dil: usize) -> Result<Res> {
        Ok(Res {
            a1: w.dev(ops, &format!("{p}.snake1.alpha"))?,
            c1: Conv::load(ops, w, &format!("{p}.conv1"), false)?,
            a2: w.dev(ops, &format!("{p}.snake2.alpha"))?,
            c2: Conv::load(ops, w, &format!("{p}.conv2"), false)?,
            dil,
        })
    }

    fn run(&self, ops: &Ops, x: &DevBuf, l: usize, t1: &DevBuf, t2: &DevBuf) -> Result<()> {
        let c = self.c1.ci;
        ops.snake(x.fp(), c, l, &self.a1, t1.fp())?;
        let pad = 3 * self.dil;
        self.c1.run(ops, t1.fp(), l, 1, self.dil, pad, pad, t2.fp())?;
        ops.snake(t2.fp(), c, l, &self.a2, t1.fp())?;
        self.c2.run(ops, t1.fp(), l, 1, 1, 0, 0, t2.fp())?;
        ops.add(x.fp(), t2.fp(), c * l)
    }
}

struct DecBlock {
    a: DevBuf,
    up: Conv,
    res: Vec<Res>,
    stride: usize,
}

struct EncBlock {
    res: Vec<Res>,
    a: DevBuf,
    down: Conv,
    stride: usize,
}

/// The semantic encoder's residual unit: x += conv1x1(elu(conv3(elu(x)))), no biases
struct SemRes {
    c1: Conv,
    c2: Conv,
    dil: usize,
}

struct HubertLayer {
    qkv: DevBuf,
    qkv_b: DevBuf,
    out: DevBuf,
    out_b: DevBuf,
    ln: (DevBuf, DevBuf),
    up: DevBuf,
    up_b: DevBuf,
    down: DevBuf,
    down_b: DevBuf,
    fln: (DevBuf, DevBuf),
}

/// The encoder's parts (a recording's codes): HuBERT, the semantic encoder, DAC's encoder, fc
struct Encoder {
    // HuBERT
    feat: Vec<(Conv, usize)>,
    gn: (DevBuf, DevBuf),
    proj_ln: (DevBuf, DevBuf),
    proj: DevBuf,
    proj_b: DevBuf,
    pos: Conv,
    pos_groups: usize,
    enc_ln: (DevBuf, DevBuf),
    layers: Vec<HubertLayer>,
    heads: usize,
    eps: f32,
    // the semantic encoder
    sem_in: Conv,
    sem_blocks: Vec<(Vec<SemRes>, Conv)>,
    // DAC's encoder
    ac_in: Conv,
    ac_blocks: Vec<EncBlock>,
    ac_a: DevBuf,
    ac_out: Conv,
    fc: DevBuf,
    fc_b: DevBuf,
    /// the quantizer's projections in (host): [64, 1,024] and bias [64] a codebook
    q_in: Vec<(Vec<f32>, Vec<f32>)>,
    /// and out [1,024, 64], bias [1,024]
    q_out: Vec<(Vec<f32>, Vec<f32>)>,
}

pub struct Codec {
    /// the codebooks' rows (host): [1,024, 64] each
    embed: Vec<Vec<f32>>,
    /// the eight projections out and fc2 folded: [256, 8 * 64], bias [256]
    front: DevBuf,
    front_b: DevBuf,
    dec_in: Conv,
    blocks: Vec<DecBlock>,
    dec_a: DevBuf,
    dec_out: Conv,
    enc: Option<Encoder>,
    pub codebooks: usize,
    pub codes: usize,
    dim: usize,
}

/// What a check keeps of an encode
#[derive(Default)]
pub struct EncTaps {
    pub sem16: Vec<f32>,
    pub semantic: Vec<f32>,
    pub esem: Vec<f32>,
    pub eac: Vec<f32>,
    pub fc: Vec<f32>,
}

fn pair(ops: &Ops, w: &Weights, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((w.dev(ops, &format!("{p}.weight"))?, w.dev(ops, &format!("{p}.bias"))?))
}

impl Codec {
    pub fn load(ops: &Ops, w: &Weights, encoder: bool) -> Result<Codec> {
        let ratios = w.nums("omnivoice.acoustic.upsampling_ratios")?;
        if ratios.iter().product::<usize>() != HOP || w.num("omnivoice.sample_rate")? as u32 != RATE {
            return Err(Error(format!("{ARCH}: a codec of {ratios:?} at {} Hz, not Higgs Audio v2's 960 samples at 24 kHz", w.num("omnivoice.sample_rate")?)));
        }
        let codes = w.num("omnivoice.codebook_size")? as usize;
        let dim = w.num("omnivoice.codebook_dim")? as usize;
        let mut codebooks = 0;
        while w.has(&format!("quantizer.quantizers.{codebooks}.codebook.embed")) {
            codebooks += 1;
        }
        let mut embed = Vec::new();
        let mut q_in = Vec::new();
        let mut q_out = Vec::new();
        for i in 0..codebooks {
            let p = format!("quantizer.quantizers.{i}");
            embed.push(w.f32(ops, &format!("{p}.codebook.embed"))?);
            q_in.push((w.f32(ops, &format!("{p}.project_in.weight"))?, w.f32(ops, &format!("{p}.project_in.bias"))?));
            q_out.push((w.f32(ops, &format!("{p}.project_out.weight"))?, w.f32(ops, &format!("{p}.project_out.bias"))?));
        }
        let hidden = q_out[0].1.len();
        // fc2 . [P_0 | ... | P_7] and fc2 . sum b_i + b2, on the GPU: x = fc2 [256, 1,024], W^T = [P_0 | ...]^T [512, 1,024]
        let fc2 = w.f32(ops, "fc2.weight")?;
        let fc2_b = w.f32(ops, "fc2.bias")?;
        let ac = fc2_b.len();
        let k = codebooks * dim;
        let mut pt = vec![0f32; k * hidden];
        for (i, (p, _)) in q_out.iter().enumerate() {
            for r in 0..hidden {
                for j in 0..dim {
                    pt[(i * dim + j) * hidden + r] = p[r * dim + j];
                }
            }
        }
        let x = Weights::upload(ops, &fc2)?;
        let wt = Weights::upload(ops, &pt)?;
        let front = DevBuf::f32(&ops.gpu, ac * k)?;
        ops.linear_f(x.fp(), ac, hidden, &wt, k, None, front.fp())?;
        ops.gpu.sync()?;
        let mut bsum = vec![0f64; hidden];
        for (_, b) in &q_out {
            bsum.iter_mut().zip(b).for_each(|(s, v)| *s += *v as f64);
        }
        let front_b: Vec<f32> = (0..ac).map(|r| (fc2_b[r] as f64 + (0..hidden).map(|c| fc2[r * hidden + c] as f64 * bsum[c]).sum::<f64>()) as f32).collect();
        let dec_in = Conv::load(ops, w, "acoustic_decoder.conv1", false)?;
        let mut blocks = Vec::new();
        for (b, s) in ratios.iter().enumerate() {
            let p = format!("acoustic_decoder.block.{b}");
            blocks.push(DecBlock {
                a: w.dev(ops, &format!("{p}.snake1.alpha"))?,
                up: Conv::load(ops, w, &format!("{p}.conv_t1"), true)?,
                res: [1, 3, 9].iter().enumerate().map(|(i, d)| Res::load(ops, w, &format!("{p}.res_unit{}", i + 1), *d)).collect::<Result<_>>()?,
                stride: *s,
            });
        }
        let enc = if encoder { Some(Encoder::load(ops, w)?) } else { None };
        Ok(Codec {
            embed,
            front,
            front_b: Weights::upload(ops, &front_b)?,
            dec_in,
            blocks,
            dec_a: w.dev(ops, "acoustic_decoder.snake1.alpha")?,
            dec_out: Conv::load(ops, w, "acoustic_decoder.conv2", false)?,
            enc,
            codebooks,
            codes,
            dim,
        })
    }

    /// The semantic encoder alone over HuBERT's features [fs, hid] (a check's): [hid, fs]
    pub fn semantic_encoder(&self, ops: &Ops, sem: &[f32], fs: usize, taps: Option<&mut Vec<Vec<f32>>>) -> Result<Vec<f32>> {
        let e = self.enc.as_ref().ok_or_else(|| Error(format!("{ARCH}: the codec was loaded without its encoder")))?;
        let b = e.sem_encode(ops, sem, fs, taps)?;
        ops.gpu.sync()?;
        read_f32(&b, 0, sem.len())
    }

    pub fn encodes(&self) -> bool {
        self.enc.is_some()
    }

    /// Codes [codebooks][t] (codebook-major) to 24 kHz samples (t * 960)
    pub fn decode(&self, ops: &Ops, codes: &[i32], t: usize) -> Result<Vec<f32>> {
        if codes.len() != self.codebooks * t || t == 0 {
            return Err(Error(format!("{} codes for {t} frames of {} codebooks", codes.len(), self.codebooks)));
        }
        let (g, k) = (&ops.gpu, self.codebooks * self.dim);
        let mut x = vec![0f32; t * k];
        for f in 0..t {
            for (i, e) in self.embed.iter().enumerate() {
                let c = (codes[i * t + f].clamp(0, self.codes as i32 - 1)) as usize;
                x[f * k + i * self.dim..f * k + (i + 1) * self.dim].copy_from_slice(&e[c * self.dim..(c + 1) * self.dim]);
            }
        }
        let xin = DevBuf::f32(g, t * k)?;
        xin.write(0, &bytes(&x))?;
        let ac = self.dec_in.ci;
        let rows = DevBuf::f32(g, t * ac)?;
        ops.linear_f(xin.fp(), t, k, &self.front, ac, Some(&self.front_b), rows.fp())?;
        let biggest = self.blocks.iter().scan((self.dec_in.co, t), |(c, l), b| {
            *l *= b.stride;
            *c = b.up.co;
            Some(*c * *l)
        }).max().unwrap_or(0).max(self.dec_in.co * t);
        let (a, b1, b2) = (DevBuf::f32(g, biggest)?, DevBuf::f32(g, biggest)?, DevBuf::f32(g, biggest)?);
        ops.transpose(rows.fp(), t, ac, b1.fp())?;
        let mut l = self.dec_in.run(ops, b1.fp(), t, 1, 1, 3, 3, a.fp())?;
        for blk in &self.blocks {
            ops.snake(a.fp(), blk.up.ci, l, &blk.a, b1.fp())?;
            let s = blk.stride;
            let pad = s.div_ceil(2);
            l = blk.up.run_t(ops, b1.fp(), l, s, pad, pad - s % 2, a.fp())?;
            for r in &blk.res {
                r.run(ops, &a, l, &b1, &b2)?;
            }
        }
        ops.snake(a.fp(), self.dec_out.ci, l, &self.dec_a, b1.fp())?;
        let n = self.dec_out.run(ops, b1.fp(), l, 1, 1, 3, 3, b2.fp())?;
        g.sync()?;
        read_f32(&b2, 0, n)
    }

    /// A recording at 24 kHz (a whole number of frames) to its codes [codebooks][t] (codebook-major) and t
    pub fn encode(&self, ops: &Ops, wav: &[f32], taps: Option<&mut EncTaps>) -> Result<(Vec<i32>, usize)> {
        let e = self.enc.as_ref().ok_or_else(|| Error(format!("{ARCH}: the codec was loaded without its encoder")))?;
        e.encode(ops, self, wav, taps)
    }
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn run_t(&self, ops: &Ops, x: *const f32, l: usize, stride: usize, pl: usize, pr: usize, out: *mut f32) -> Result<usize> {
        ops.conv_t(x, self.ci, l, &self.w, self.co, self.k, self.b.as_ref(), stride, pl, pr, out)
    }
}

impl Encoder {
    fn load(ops: &Ops, w: &Weights) -> Result<Encoder> {
        let kernel = w.nums("omnivoice.semantic.conv_kernel")?;
        let stride = w.nums("omnivoice.semantic.conv_stride")?;
        let feat = (0..kernel.len()).map(|i| Ok((Conv::load(ops, w, &format!("semantic_model.feature_extractor.conv_layers.{i}.conv"), false)?, stride[i])))
            .collect::<Result<Vec<_>>>()?;
        if feat.iter().zip(&kernel).any(|((c, _), k)| c.k != *k) {
            return Err(Error("HuBERT's feature convolutions do not fit the file's kernel sizes".into()));
        }
        let hid = w.num("omnivoice.semantic.hidden_size")? as usize;
        let heads = w.num("omnivoice.semantic.num_attention_heads")? as usize;
        let n = w.num("omnivoice.semantic.num_hidden_layers")? as usize;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let p = format!("semantic_model.encoder.layers.{i}");
            let a = |x: &str| format!("{p}.attention.{x}");
            let mut qkv = Vec::new();
            let mut qkv_b = Vec::new();
            for m in ["q_proj", "k_proj", "v_proj"] {
                qkv.extend(w.f32(ops, &a(&format!("{m}.weight")))?);
                qkv_b.extend(w.f32(ops, &a(&format!("{m}.bias")))?);
            }
            layers.push(HubertLayer {
                qkv: Weights::upload(ops, &qkv)?,
                qkv_b: Weights::upload(ops, &qkv_b)?,
                out: w.dev(ops, &a("out_proj.weight"))?,
                out_b: w.dev(ops, &a("out_proj.bias"))?,
                ln: pair(ops, w, &format!("{p}.layer_norm"))?,
                up: w.dev(ops, &format!("{p}.feed_forward.intermediate_dense.weight"))?,
                up_b: w.dev(ops, &format!("{p}.feed_forward.intermediate_dense.bias"))?,
                down: w.dev(ops, &format!("{p}.feed_forward.output_dense.weight"))?,
                down_b: w.dev(ops, &format!("{p}.feed_forward.output_dense.bias"))?,
                fln: pair(ops, w, &format!("{p}.final_layer_norm"))?,
            });
        }
        let mut sem_blocks = Vec::new();
        let mut b = 0;
        while w.has(&format!("encoder_semantic.conv_blocks.{b}.conv.weight")) {
            let p = format!("encoder_semantic.conv_blocks.{b}");
            let mut res = Vec::new();
            let mut r = 0;
            while w.has(&format!("{p}.res_units.{r}.conv1.weight")) {
                // block_dilations [1, 1]
                res.push(SemRes { c1: Conv::load(ops, w, &format!("{p}.res_units.{r}.conv1"), false)?, c2: Conv::load(ops, w, &format!("{p}.res_units.{r}.conv2"), false)?,
                                  dil: 1 });
                r += 1;
            }
            sem_blocks.push((res, Conv::load(ops, w, &format!("{p}.conv"), false)?));
            b += 1;
        }
        let ratios = w.nums("omnivoice.acoustic.downsampling_ratios")?;
        let mut ac_blocks = Vec::new();
        for (i, s) in ratios.iter().enumerate() {
            let p = format!("acoustic_encoder.block.{i}");
            ac_blocks.push(EncBlock {
                res: [1, 3, 9].iter().enumerate().map(|(j, d)| Res::load(ops, w, &format!("{p}.res_unit{}", j + 1), *d)).collect::<Result<_>>()?,
                a: w.dev(ops, &format!("{p}.snake1.alpha"))?,
                down: Conv::load(ops, w, &format!("{p}.conv1"), false)?,
                stride: *s,
            });
        }
        let pos = Conv::load(ops, w, "semantic_model.encoder.pos_conv_embed.conv", false)?;
        let pos_groups = w.num("omnivoice.semantic.num_conv_pos_embedding_groups")? as usize;
        if pos.ci * pos_groups != hid {
            return Err(Error(format!("HuBERT's positional convolution: {} channels a group of {pos_groups}, the model is {hid} wide", pos.ci)));
        }
        let mut q_in = Vec::new();
        let mut q_out = Vec::new();
        let mut i = 0;
        while w.has(&format!("quantizer.quantizers.{i}.project_in.weight")) {
            let p = format!("quantizer.quantizers.{i}");
            q_in.push((w.f32(ops, &format!("{p}.project_in.weight"))?, w.f32(ops, &format!("{p}.project_in.bias"))?));
            q_out.push((w.f32(ops, &format!("{p}.project_out.weight"))?, w.f32(ops, &format!("{p}.project_out.bias"))?));
            i += 1;
        }
        Ok(Encoder {
            feat,
            gn: pair(ops, w, "semantic_model.feature_extractor.conv_layers.0.layer_norm")?,
            proj_ln: pair(ops, w, "semantic_model.feature_projection.layer_norm")?,
            proj: w.dev(ops, "semantic_model.feature_projection.projection.weight")?,
            proj_b: w.dev(ops, "semantic_model.feature_projection.projection.bias")?,
            pos,
            pos_groups,
            enc_ln: pair(ops, w, "semantic_model.encoder.layer_norm")?,
            layers,
            heads,
            eps: w.num("omnivoice.semantic.layer_norm_eps")? as f32,
            sem_in: Conv::load(ops, w, "encoder_semantic.conv", false)?,
            sem_blocks,
            ac_in: Conv::load(ops, w, "acoustic_encoder.conv1", false)?,
            ac_blocks,
            ac_a: w.dev(ops, "acoustic_encoder.snake1.alpha")?,
            ac_out: Conv::load(ops, w, "acoustic_encoder.conv2", false)?,
            fc: w.dev(ops, "fc.weight")?,
            fc_b: w.dev(ops, "fc.bias")?,
            q_in,
            q_out,
        })
    }

    /// HuBERT over 16 kHz samples (padded): the mean of its hidden states [f, hidden] on the GPU, and f
    fn hubert(&self, ops: &Ops, x16: &[f32]) -> Result<(DevBuf, usize)> {
        let (g, nsd) = (&ops.gpu, &ops.nsd);
        let ch = self.feat[0].0.co;
        let mut l = x16.len();
        let mut lens = Vec::new();
        for (c, s) in &self.feat {
            l = (l - c.k) / s + 1;
            lens.push(l);
        }
        let f = l;
        if f == 0 {
            return Err(Error("the recording is too short for HuBERT".into()));
        }
        let first = ch * lens[0];
        let (a, b) = (DevBuf::f32(g, first)?, DevBuf::f32(g, first)?);
        let inp = DevBuf::f32(g, x16.len())?;
        inp.write(0, &bytes(x16))?;
        let mut l = x16.len();
        let mut src = inp.fp();
        for (i, (c, s)) in self.feat.iter().enumerate() {
            let dst = if i % 2 == 0 { a.fp() } else { b.fp() };
            l = c.run(ops, src, l, *s, 1, 0, 0, dst)?;
            if i == 0 {
                ops.chan_norm(dst, c.co, l, &self.gn.0, &self.gn.1, 1e-5, dst)?;
            }
            nsd.gelu(dst.cast(), Dt::F32, c.co * l, false)?;
            src = dst;
        }
        let hid = self.proj_b.len / 4;
        // [ch, f] -> [f, ch], its layer norm, the projection
        let t = DevBuf::f32(g, f * ch)?;
        ops.transpose(src, ch, f, t.fp())?;
        nsd.layer_norm(t.ptr(), Dt::F32, f, ch, self.proj_ln.0.ptr(), self.proj_ln.1.ptr(), self.eps, t.ptr(), Dt::F32)?;
        let h = DevBuf::f32(g, f * hid)?;
        ops.linear_f(t.fp(), f, ch, &self.proj, hid, Some(&self.proj_b), h.fp())?;
        // the positional convolution: groups of channels, k = 128, padding 64 each side, the last frame dropped, GELU
        let ht = DevBuf::f32(g, f * hid)?;
        ops.transpose(h.fp(), f, hid, ht.fp())?;
        let pc = self.pos.k;
        let po = DevBuf::f32(g, hid * (f + 1))?;
        let gw = self.pos.ci;
        for grp in 0..self.pos_groups {
            let w = self.pos.w.view(grp * gw * gw * pc * 4, gw * gw * pc * 4)?;
            let bias = self.pos.b.as_ref().map(|b| b.view(grp * gw * 4, gw * 4)).transpose()?;
            ops.conv(fp(&ht, grp * gw * f), gw, f, &w, gw, pc, bias.as_ref(), 1, 1, pc / 2, pc / 2, fp(&po, grp * gw * (f + 1)))?;
        }
        // drop the last frame: [hid, f + 1] -> rows [f + 1, hid], the first f
        let pt = DevBuf::f32(g, (f + 1) * hid)?;
        ops.transpose(po.fp(), hid, f + 1, pt.fp())?;
        nsd.gelu(pt.ptr(), Dt::F32, f * hid, false)?;
        ops.add(h.fp(), pt.fp(), f * hid)?;
        nsd.layer_norm(h.ptr(), Dt::F32, f, hid, self.enc_ln.0.ptr(), self.enc_ln.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
        let mean = DevBuf::f32(g, f * hid)?;
        mean.copy_within(0, &h, 0, f * hid * 4)?;
        let qkv = DevBuf::f32(g, f * 3 * hid)?;
        let att = DevBuf::f32(g, f * hid)?;
        let tmp = DevBuf::f32(g, f * hid)?;
        let inter = self.layers[0].up_b.len / 4;
        let up = DevBuf::f32(g, f * inter)?;
        let d = hid / self.heads;
        for ly in &self.layers {
            ops.linear_f(h.fp(), f, hid, &ly.qkv, 3 * hid, Some(&ly.qkv_b), qkv.fp())?;
            nsd.attention(qkv.ptr(), fp(&qkv, hid).cast(), fp(&qkv, 2 * hid).cast(), Dt::F32, f, self.heads, d, 3 * hid, att.ptr(), Dt::F32)?;
            ops.linear_f(att.fp(), f, hid, &ly.out, hid, Some(&ly.out_b), tmp.fp())?;
            ops.add(h.fp(), tmp.fp(), f * hid)?;
            nsd.layer_norm(h.ptr(), Dt::F32, f, hid, ly.ln.0.ptr(), ly.ln.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
            ops.linear_f(h.fp(), f, hid, &ly.up, inter, Some(&ly.up_b), up.fp())?;
            nsd.gelu(up.ptr(), Dt::F32, f * inter, false)?;
            ops.linear_f(up.fp(), f, inter, &ly.down, hid, Some(&ly.down_b), tmp.fp())?;
            ops.add(h.fp(), tmp.fp(), f * hid)?;
            nsd.layer_norm(h.ptr(), Dt::F32, f, hid, ly.fln.0.ptr(), ly.fln.1.ptr(), self.eps, h.ptr(), Dt::F32)?;
            ops.add(mean.fp(), h.fp(), f * hid)?;
        }
        nsd.scale(mean.ptr(), f * hid, 1.0 / (self.layers.len() + 1) as f32)?;
        Ok((mean, f))
    }

    /// The semantic encoder over HuBERT's features [fs, hid] (rows): its output [hid, fs] on the GPU
    fn sem_encode(&self, ops: &Ops, sem: &[f32], fs: usize, mut taps: Option<&mut Vec<Vec<f32>>>) -> Result<DevBuf> {
        let (g, hid) = (&ops.gpu, self.proj_b.len / 4);
        let rows = DevBuf::f32(g, fs * hid)?;
        rows.write(0, &bytes(sem))?;
        let (s0, s1, s2) = (DevBuf::f32(g, fs * hid)?, DevBuf::f32(g, fs * hid)?, DevBuf::f32(g, fs * hid)?);
        ops.transpose(rows.fp(), fs, hid, s1.fp())?;
        self.sem_in.run(ops, s1.fp(), fs, 1, 1, 1, 1, s0.fp())?;
        let mut tap = |b: &DevBuf| -> Result<()> {
            if let Some(t) = taps.as_deref_mut() {
                ops.gpu.sync()?;
                t.push(read_f32(b, 0, hid * fs)?);
            }
            Ok(())
        };
        tap(&s0)?;
        for (res, conv) in &self.sem_blocks {
            for r in res {
                ops.elu(s0.fp(), hid * fs, s1.fp())?;
                r.c1.run(ops, s1.fp(), fs, 1, r.dil, r.dil, r.dil, s2.fp())?;
                ops.elu(s2.fp(), hid * fs, s2.fp())?;
                r.c2.run(ops, s2.fp(), fs, 1, 1, 0, 0, s1.fp())?;
                ops.add(s0.fp(), s1.fp(), hid * fs)?;
                tap(&s0)?;
            }
            conv.run(ops, s0.fp(), fs, 1, 1, 1, 1, s1.fp())?;
            s0.copy_within(0, &s1, 0, hid * fs * 4)?;
            tap(&s0)?;
        }
        Ok(s0)
    }

    fn encode(&self, ops: &Ops, c: &Codec, wav: &[f32], mut taps: Option<&mut EncTaps>) -> Result<(Vec<i32>, usize)> {
        let g = &ops.gpu;
        if wav.is_empty() || !wav.len().is_multiple_of(HOP) {
            return Err(Error(format!("{} samples: a whole number of {HOP}-sample frames to encode", wav.len())));
        }
        // the semantic features: 16 kHz, 160 zeros each side, HuBERT, every other frame
        let mut x16 = vec![0f32; 160];
        x16.extend(crate::post::resample(wav, RATE, SEMANTIC_RATE));
        x16.extend(std::iter::repeat_n(0f32, 160));
        if let Some(k) = taps.as_deref_mut() {
            k.sem16 = x16.clone();
        }
        let (mean, f) = self.hubert(ops, &x16)?;
        let hid = self.proj_b.len / 4;
        g.sync()?;
        let all = read_f32(&mean, 0, f * hid)?;
        let fs = f.div_ceil(2);
        let mut sem = Vec::with_capacity(fs * hid);
        for r in (0..f).step_by(2) {
            sem.extend_from_slice(&all[r * hid..(r + 1) * hid]);
        }
        if let Some(k) = taps.as_deref_mut() {
            k.semantic = sem.clone();
        }
        let s0 = self.sem_encode(ops, &sem, fs, None)?;
        // the acoustic features: DAC's encoder (480 zeros each side when its length would not match the semantic one's)
        let t = wav.len() / HOP;
        let pad = if t == fs { 0 } else { HOP / 2 };
        let mut w24 = vec![0f32; pad];
        w24.extend_from_slice(wav);
        w24.extend(std::iter::repeat_n(0f32, pad));
        let mut l = w24.len();
        let mut biggest = self.ac_in.co * l;
        for b in &self.ac_blocks {
            biggest = biggest.max(b.down.ci * l);
            l = (l + 2 * b.stride.div_ceil(2) - 2 * b.stride) / b.stride + 1;
        }
        if l != fs {
            return Err(Error(format!("the acoustic encoder makes {l} frames, the semantic one {fs}")));
        }
        let (a, b1, b2) = (DevBuf::f32(g, biggest)?, DevBuf::f32(g, biggest)?, DevBuf::f32(g, biggest)?);
        b1.write(0, &bytes(&w24))?;
        let mut l = self.ac_in.run(ops, b1.fp(), w24.len(), 1, 1, 3, 3, a.fp())?;
        for blk in &self.ac_blocks {
            for r in &blk.res {
                r.run(ops, &a, l, &b1, &b2)?;
            }
            ops.snake(a.fp(), blk.down.ci, l, &blk.a, b1.fp())?;
            let p = blk.stride.div_ceil(2);
            l = blk.down.run(ops, b1.fp(), l, blk.stride, 1, p, p, a.fp())?;
        }
        ops.snake(a.fp(), self.ac_out.ci, l, &self.ac_a, b1.fp())?;
        let ac = self.ac_out.co;
        // [acoustic (ac); semantic (hid)] -> rows [fs, ac + hid] -> fc
        let cat = DevBuf::f32(g, (ac + hid) * fs)?;
        self.ac_out.run(ops, b1.fp(), l, 1, 1, 1, 1, cat.fp())?;
        cat.copy_within(ac * fs * 4, &s0, 0, hid * fs * 4)?;
        if let Some(k) = taps.as_deref_mut() {
            g.sync()?;
            k.eac = read_f32(&cat, 0, ac * fs)?;
            k.esem = read_f32(&cat, ac * fs, hid * fs)?;
        }
        let width = ac + hid;
        let ct = DevBuf::f32(g, width * fs)?;
        ops.transpose(cat.fp(), width, fs, ct.fp())?;
        let emb = DevBuf::f32(g, width * fs)?;
        ops.linear_f(ct.fp(), fs, width, &self.fc, width, Some(&self.fc_b), emb.fp())?;
        g.sync()?;
        let mut res = read_f32(&emb, 0, width * fs)?;
        if let Some(k) = taps {
            // [fs, width] -> [width, fs], as the reference keeps it
            k.fc = (0..width).flat_map(|ci| (0..fs).map(move |r| (ci, r))).map(|(ci, r)| res[r * width + ci]).collect();
        }
        // the residual quantizer, on the host
        let nq = c.codebooks;
        let mut codes = vec![0i32; nq * fs];
        for q in 0..nq {
            let (wi, bi) = &self.q_in[q];
            let (wo, bo) = &self.q_out[q];
            let e = &c.embed[q];
            let dim = bi.len();
            let found = par_rows(fs, |r| {
                let x = &res[r * width..(r + 1) * width];
                let z: Vec<f32> = (0..dim).map(|j| bi[j] + dot(&wi[j * width..(j + 1) * width], x)).collect();
                let zz: f32 = z.iter().map(|v| v * v).sum();
                let mut best = (f32::NEG_INFINITY, 0usize);
                for (ci, row) in e.chunks_exact(dim).enumerate() {
                    // as transformers: -(|z|^2 - 2 z.e + |e|^2), the first maximum
                    let ee: f32 = row.iter().map(|v| v * v).sum();
                    let d = -(zz - 2.0 * dot(&z, row) + ee);
                    if d > best.0 {
                        best = (d, ci);
                    }
                }
                best.1
            });
            for (r, ci) in found.iter().enumerate() {
                codes[q * fs + r] = *ci as i32;
                let row = &e[ci * dim..(ci + 1) * dim];
                let x = &mut res[r * width..(r + 1) * width];
                for (o, xo) in x.iter_mut().enumerate() {
                    *xo -= bo[o] + dot(&wo[o * dim..(o + 1) * dim], row);
                }
            }
        }
        Ok((codes, fs))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// f(r) for r < n over the host's threads
fn par_rows<F: Fn(usize) -> usize + Sync>(n: usize, f: F) -> Vec<usize> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).clamp(1, 16);
    let per = n.div_ceil(threads).max(1);
    let mut out = vec![0usize; n];
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, o) in chunk.iter_mut().enumerate() {
                    *o = f(i * per + j);
                }
            });
        }
    });
    out
}
