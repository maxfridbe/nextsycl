//! Raw-reference cloning: a recording to the codec's own codes [frames, 10], which the backbone reads as the reference
//! prefix (the checkpoint's default speaker mode). Two models: the frozen sixteen-layer Wav2Vec2-BERT semantic teacher
//! (`ssl-teacher.gguf`, `w2vbert_ssl`) and the dual reference encoder (`zac-encoder.gguf`, `mossdual_encoder`).
//!
//! ```text
//!   44.1 kHz audio, cut in lanes of 300 frames (20 s) with 150 frames of left context
//!   acoustic: 44160 Hz (the file's polyphase kernel) -> causal conv, two Snake residual blocks with strided convs
//!     (x2, x4) -> patch-down 23 and four x2 MOSS stages -> [300, 1024]
//!   semantic: the lane at 16 kHz -> SeamlessM4T fbank (host) -> the teacher's layer 16 -> pooled onto the 300 frames
//!     -> the semantic encoder (projection, 4 ConvNeXt) -> RVQ 0
//!   residual: acoustic - the semantic head's prediction of RVQ 0's latent -> RVQ 1..9
//! ```
//!
//! Origin: Zyphra's ggml server's zac-encoder.cpp / ssl-encoder.cpp and their MLX port
//! (github.com/Zyphra/ZONOS3-Mini, MIT). The reference runs the production BF16 autocast recipe (operands and tensor
//! boundaries rounded to bfloat16); this port runs float32 throughout, so a code at an RVQ near-tie can differ.

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::codec::{convnext, moss_layer, moss_weights, Block, MossLayer};
use crate::ops::{e, nulf, Mat, Ops, Weights};
use crate::spk::{fft, polyphase, resample_any};

/// NS_Z3_ENC_DUMP=<dir>: the first lane's stages as .npy, named as zac-encoder-cli --dump names them
fn dump(name: &str, shape: &[usize], v: &[f32]) {
    if let Ok(d) = std::env::var("NS_Z3_ENC_DUMP") {
        let _ = crate::check::npy(&std::path::Path::new(&d).join(format!("lane.0.{name}.npy")), shape, &v[..shape.iter().product::<usize>()]);
    }
}

/// a channels-first [c, l] buffer dumped as [l, c]
fn dump_cf(ops: &Ops, name: &str, x: &DevBuf, c: usize, l: usize) -> Result<()> {
    if std::env::var("NS_Z3_ENC_DUMP").is_err() {
        return Ok(());
    }
    ops.sync()?;
    let v = x.to_f32().map_err(e)?;
    let mut t = vec![0f32; c * l];
    for ci in 0..c {
        for li in 0..l {
            t[li * c + ci] = v[ci * l + li];
        }
    }
    dump(name, &[l, c], &t);
    Ok(())
}

fn dump_dev(ops: &Ops, name: &str, x: &DevBuf, t: usize, c: usize) -> Result<()> {
    if std::env::var("NS_Z3_ENC_DUMP").is_err() {
        return Ok(());
    }
    ops.sync()?;
    dump(name, &[t, c], &x.to_f32().map_err(e)?);
    Ok(())
}

const LANE: usize = 300;
const CONTEXT: usize = 150;
const HOP: usize = 2940;
const IHOP: usize = 2944;
/// the longest recording encoded (the registry's audio_max_seconds for a reference)
pub const MAX_SECONDS: usize = 60;

struct Lin {
    m: Mat,
    b: Option<DevBuf>,
}

fn lin(ops: &Ops, w: &Weights, p: &str, bias: bool) -> Result<Lin> {
    Ok(Lin { m: w.mat(ops, &format!("{p}.weight"))?, b: if bias { Some(w.dev(ops, &format!("{p}.bias"))?) } else { None } })
}

impl Lin {
    fn run(&self, ops: &Ops, x: *const f32, t: usize, out: *mut f32) -> Result<()> {
        ops.linear(x, t, &self.m, self.b.as_ref().map_or(nulf(), |b| b.fp() as *const f32), out)
    }
}

struct Ln {
    w: DevBuf,
    b: DevBuf,
}

fn ln(ops: &Ops, w: &Weights, p: &str) -> Result<Ln> {
    Ok(Ln { w: w.dev(ops, &format!("{p}.weight"))?, b: w.dev(ops, &format!("{p}.bias"))? })
}

struct SslLayer {
    ffn1_ln: Ln,
    ffn1_up: Lin,
    ffn1_down: Lin,
    ffn2_ln: Ln,
    ffn2_up: Lin,
    ffn2_down: Lin,
    attn_ln: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    dist: DevBuf,
    conv_ln: Ln,
    pw1: Lin,
    dw: DevBuf,
    dw_ln: Ln,
    pw2: Lin,
    final_ln: Ln,
}

/// The Wav2Vec2-BERT teacher (layer 16) and its SeamlessM4T frontend
pub struct SslTeacher {
    window: Vec<f64>,
    mel: Vec<f64>,
    resample: Vec<f64>,
    n_mels: usize,
    flen: usize,
    hop: usize,
    nfft: usize,
    stride: usize,
    eps: f32,
    preemph: f64,
    floor: f64,
    pad_value: f32,
    d: usize,
    heads: usize,
    rel: (usize, usize),
    kernel: usize,
    fp_ln: Ln,
    fp_proj: Lin,
    layers: Vec<SslLayer>,
    half: DevBuf,
    ones: DevBuf,
    /// the production BF16 recipe (the codes the backbone was trained on come from it), else float32 throughout
    pub bf: bool,
}

impl SslTeacher {
    pub fn load(ops: &Ops, w: &Weights) -> Result<SslTeacher> {
        if w.g.architecture() != "w2vbert_ssl" {
            return Err(Error(format!("{}: not the Wav2Vec2-BERT teacher (w2vbert_ssl)", w.name())));
        }
        let k = |s: &str| format!("w2vbert_ssl.{s}");
        let n_layer = w.u(&k("n_layer"))?;
        let d = w.u(&k("n_embd"))?;
        let mut layers = Vec::new();
        for l in 0..n_layer {
            let p = |x: &str| format!("enc.{l}.{x}");
            // the conv module's depthwise taps: [C, K] as stored
            layers.push(SslLayer {
                ffn1_ln: ln(ops, w, &p("ffn1_ln"))?,
                ffn1_up: lin(ops, w, &p("ffn1.up"), true)?,
                ffn1_down: lin(ops, w, &p("ffn1.down"), true)?,
                ffn2_ln: ln(ops, w, &p("ffn2_ln"))?,
                ffn2_up: lin(ops, w, &p("ffn2.up"), true)?,
                ffn2_down: lin(ops, w, &p("ffn2.down"), true)?,
                attn_ln: ln(ops, w, &p("attn_ln"))?,
                q: lin(ops, w, &p("attn.q"), true)?,
                k: lin(ops, w, &p("attn.k"), true)?,
                v: lin(ops, w, &p("attn.v"), true)?,
                o: lin(ops, w, &p("attn.o"), true)?,
                dist: w.dev(ops, &p("attn.dist"))?,
                conv_ln: ln(ops, w, &p("conv_ln"))?,
                pw1: lin(ops, w, &p("conv.pw1"), false)?,
                dw: w.dev(ops, &p("conv.dw.weight"))?,
                dw_ln: ln(ops, w, &p("conv.dw_ln"))?,
                pw2: lin(ops, w, &p("conv.pw2"), false)?,
                final_ln: ln(ops, w, &p("final_ln"))?,
            });
        }
        Ok(SslTeacher {
            window: w.f64("frontend.window")?,
            mel: w.f64("frontend.mel_filters")?,
            resample: w.f64("frontend.resample_44100")?,
            n_mels: w.u(&k("n_mels"))?,
            flen: w.u(&k("frame_length"))?,
            hop: w.u(&k("hop_length"))?,
            nfft: w.u(&k("fft_length"))?,
            stride: w.u(&k("stride"))?,
            eps: w.f(&k("eps"))? as f32,
            preemph: w.f(&k("preemphasis"))?,
            floor: w.f(&k("mel_floor"))?,
            pad_value: w.f(&k("padding_value"))? as f32,
            heads: w.u(&k("n_head"))?,
            rel: (w.u(&k("rel_left"))?, w.u(&k("rel_right"))?),
            kernel: w.u(&k("conv_kernel"))?,
            fp_ln: ln(ops, w, "fp.ln")?,
            fp_proj: lin(ops, w, "fp.proj", true)?,
            layers,
            half: ops.from_f32(&vec![0.5f32; d])?,
            ones: ops.from_f32(&vec![1.0f32; d])?,
            d,
            bf: false,
        })
    }

    /// The BF16 recipe: the matrices and the relative-position table rounded to bfloat16 (norms, biases and the
    /// depthwise taps stay float32)
    pub fn use_bf16(&mut self, ops: &Ops) -> Result<()> {
        let mut mats = vec![&self.fp_proj.m];
        for l in &self.layers {
            mats.extend([&l.ffn1_up.m, &l.ffn1_down.m, &l.ffn2_up.m, &l.ffn2_down.m, &l.q.m, &l.k.m, &l.v.m, &l.o.m, &l.pw1.m, &l.pw2.m]);
            ops.bf16(l.dist.fp(), l.dist.floats())?;
        }
        for m in mats {
            ops.bf16(m.w.fp(), m.n * m.k)?;
        }
        self.bf = true;
        Ok(())
    }

    /// 16 kHz audio -> (features [T, stride n_mels], valid [T]): the SeamlessM4T fbank as ssl-encoder.cpp computes it
    pub fn features_16k(&self, wav: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let (fl, hop, nfft, nm) = (self.flen, self.hop, self.nfft, self.n_mels);
        let nf = if wav.len() < fl { 0 } else { 1 + (wav.len() - fl) / hop };
        if nf < 2 {
            return Err(Error("the recording is too short for the teacher's fbank".into()));
        }
        let bins = nfft / 2 + 1;
        let mut fb = vec![0f32; nf * nm];
        let (mut re, mut im) = (vec![0f64; nfft], vec![0f64; nfft]);
        for f in 0..nf {
            let mut z: Vec<f64> = (0..fl).map(|i| wav[f * hop + i] as f64 * 32768.0).collect();
            let mean = z.iter().sum::<f64>() / fl as f64;
            z.iter_mut().for_each(|x| *x -= mean);
            for i in (1..fl).rev() {
                z[i] -= self.preemph * z[i - 1];
            }
            z[0] *= 1.0 - self.preemph;
            re.iter_mut().for_each(|x| *x = 0.0);
            im.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..fl {
                re[i] = z[i] * self.window[i];
            }
            fft(&mut re, &mut im);
            for m in 0..nm {
                let mut e = 0f64;
                for b in 0..bins {
                    let (r, i) = (re[b] as f32 as f64, im[b] as f32 as f64);
                    e += (r * r + i * i) * self.mel[b * nm + m];
                }
                fb[f * nm + m] = e.max(self.floor).ln() as f32;
            }
        }
        // per-mel mean and (unbiased) deviation over the frames
        for m in 0..nm {
            let mean = (0..nf).map(|f| fb[f * nm + m] as f64).sum::<f64>() / nf as f64;
            let var = (0..nf).map(|f| (fb[f * nm + m] as f64 - mean).powi(2)).sum::<f64>() / (nf - 1) as f64;
            let sd = (var + 1e-7).sqrt();
            for f in 0..nf {
                fb[f * nm + m] = ((fb[f * nm + m] as f64 - mean) / sd) as f32;
            }
        }
        let t = nf.div_ceil(self.stride);
        let mut feats = vec![self.pad_value; t * self.stride * nm];
        feats[..nf * nm].copy_from_slice(&fb);
        let mut valid = vec![1f32; t];
        if nf % self.stride != 0 {
            valid[t - 1] = 0.0;
        }
        Ok((feats, valid, t))
    }

    /// Mono 44.1 kHz audio -> the layer-16 hidden states [T, d]
    pub fn encode(&self, ops: &Ops, pcm44100: &[f32]) -> Result<(Vec<f32>, usize)> {
        let wav16 = polyphase(pcm44100, &self.resample, 441, 160, 17);
        let (feats, valid, t) = self.features_16k(&wav16)?;
        let d = self.d;
        dump("ssl.features", &[t, feats.len() / t], &feats);
        let fin = feats.len() / t;
        let x0 = ops.from_f32(&feats)?;
        let vd = ops.from_f32(&valid)?;
        let n = |c: usize| ops.f32(t * c);
        let (x, a, b, q, k, v) = (n(d)?, n(d)?, n(d)?, n(d)?, n(d)?, n(d)?);
        let big = n(4 * d)?;
        let zero_invalid = |buf: &DevBuf, c: usize| -> Result<()> {
            for (i, ok) in valid.iter().enumerate() {
                if *ok == 0.0 {
                    buf.write(i * c * 4, &vec![0u8; c * 4]).map_err(e)?;
                }
            }
            Ok(())
        };
        let fbuf = n(fin)?;
        // the BF16 recipe (bf): a matmul's input and its output (after the bias) rounded; the first layer's residual
        // stream too (ssl_teacher.py)
        let bf = self.bf;
        let r = |p: *mut f32, m: usize| if bf { ops.bf16(p, m) } else { Ok(()) };
        let lin = |l: &Lin, x: &DevBuf, out: &DevBuf| -> Result<()> {
            r(x.fp(), t * l.m.k)?;
            l.run(ops, x.fp(), t, out.fp())?;
            r(out.fp(), t * l.m.n)
        };
        r(x0.fp(), t * fin)?;
        ops.layer_norm(x0.fp(), t, fin, self.fp_ln.w.fp(), self.fp_ln.b.fp(), self.eps, fbuf.fp())?;
        lin(&self.fp_proj, &fbuf, &x)?;
        zero_invalid(&x, d)?;
        dump_dev(ops, "ssl.hidden_0", &x, t, d)?;
        let scale = 1.0 / ((d / self.heads) as f32).sqrt();
        let ffn = |l1: &Ln, up: &Lin, down: &Lin| -> Result<()> {
            ops.layer_norm(x.fp(), t, d, l1.w.fp(), l1.b.fp(), self.eps, a.fp())?;
            lin(up, &a, &big)?;
            ops.silu(big.fp(), t * up.m.n)?;
            lin(down, &big, &b)?;
            ops.scale_add(x.fp(), b.fp(), self.half.fp(), t, d)
        };
        for (li, ly) in self.layers.iter().enumerate() {
            let res = |x: &DevBuf| if li == 0 { r(x.fp(), t * d) } else { Ok(()) };
            ffn(&ly.ffn1_ln, &ly.ffn1_up, &ly.ffn1_down)?;
            res(&x)?;
            ops.layer_norm(x.fp(), t, d, ly.attn_ln.w.fp(), ly.attn_ln.b.fp(), self.eps, a.fp())?;
            r(a.fp(), t * d)?;
            lin(&ly.q, &a, &q)?;
            lin(&ly.k, &a, &k)?;
            lin(&ly.v, &a, &v)?;
            ops.attn_rel(q.fp(), k.fp(), v.fp(), d, t, self.heads, d / self.heads, ly.dist.fp(), self.rel.0, self.rel.1, vd.fp(), scale, bf, a.fp())?;
            lin(&ly.o, &a, &b)?;
            ops.scale_add(x.fp(), b.fp(), self.ones.fp(), t, d)?;
            res(&x)?;
            ops.layer_norm(x.fp(), t, d, ly.conv_ln.w.fp(), ly.conv_ln.b.fp(), self.eps, a.fp())?;
            zero_invalid(&a, d)?;
            lin(&ly.pw1, &a, &big)?;
            ops.glu(big.fp(), t, d, a.fp())?;
            r(a.fp(), t * d)?;
            ops.dwconv_causal(a.fp(), t, d, ly.dw.fp(), self.kernel, nulf(), b.fp())?;
            r(b.fp(), t * d)?;
            ops.layer_norm(b.fp(), t, d, ly.dw_ln.w.fp(), ly.dw_ln.b.fp(), self.eps, a.fp())?;
            ops.silu(a.fp(), t * d)?;
            lin(&ly.pw2, &a, &b)?;
            ops.scale_add(x.fp(), b.fp(), self.ones.fp(), t, d)?;
            res(&x)?;
            ffn(&ly.ffn2_ln, &ly.ffn2_up, &ly.ffn2_down)?;
            res(&x)?;
            ops.layer_norm(x.fp(), t, d, ly.final_ln.w.fp(), ly.final_ln.b.fp(), self.eps, a.fp())?;
            x.copy_within(0, &a, 0, t * d * 4).map_err(e)?;
            if li < 2 {
                dump_dev(ops, &format!("ssl.hidden_{}", li + 1), &x, t, d)?;
            }
        }
        Ok((x.to_f32().map_err(e)?[..t * d].to_vec(), t))
    }
}

/// A channels-first causal convolution: weight [co, ci, k] on the card
struct Conv {
    w: DevBuf,
    b: DevBuf,
    co: usize,
    ci: usize,
    k: usize,
}

/// x rounded to the nearest bfloat16 value (ties to even), as the reference's bf16_round
fn bf16h(x: f32) -> f32 {
    let u = x.to_bits();
    if u & 0x7f80_0000 == 0x7f80_0000 {
        return x;
    }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// The BF16 recipe's weights: every matrix, bias and tap outside the norms holds a bfloat16 value
fn round_dev(ops: &Ops, b: &DevBuf) -> Result<()> {
    ops.bf16(b.fp(), b.floats())
}

fn round_block(ops: &Ops, b: &Block) -> Result<()> {
    for d in [&b.dw, &b.dw_b, &b.pw1_b, &b.pw2_b, &b.pw1.w, &b.pw2.w] {
        round_dev(ops, d)?;
    }
    Ok(())
}

fn conv(ops: &Ops, w: &Weights, p: &str) -> Result<Conv> {
    let s = w.shape(&format!("{p}.weight"))?; // [K, OC, IC]
    let (k, co, ci) = (s[0], s[1], s[2]);
    let v = w.f32(&format!("{p}.weight"))?;
    let mut t = vec![0f32; co * ci * k];
    for kk in 0..k {
        for o in 0..co {
            for i in 0..ci {
                t[(o * ci + i) * k + kk] = v[(kk * co + o) * ci + i];
            }
        }
    }
    let b: Vec<f32> = w.f32(&format!("{p}.bias"))?.into_iter().map(bf16h).collect();
    t.iter_mut().for_each(|x| *x = bf16h(*x));
    Ok(Conv { w: ops.from_f32(&t)?, b: ops.from_f32(&b)?, co, ci, k })
}

struct Snake {
    alpha: DevBuf,
    inv: DevBuf,
}

fn snake(ops: &Ops, w: &Weights, p: &str) -> Result<Snake> {
    Ok(Snake { alpha: w.dev(ops, &format!("{p}.alpha"))?, inv: w.dev(ops, &format!("{p}.inv_alpha"))? })
}

struct ResUnit {
    s1: Snake,
    c1: Conv,
    s2: Snake,
    c2: Conv,
    dil: usize,
}

struct DownBlock {
    res: Vec<ResUnit>,
    snake: Snake,
    conv: Conv,
    stride: usize,
}

struct EncStage {
    inp: Mat,
    layers: Vec<MossLayer>,
    out: Mat,
    d: usize,
    heads: usize,
    context: usize,
    patch: usize,
}

/// One RVQ codebook (host): the input / output projections and its 8-dimensional codes
struct Quantizer {
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    cb: Vec<f32>,
    cbn: Vec<f32>,
    cbn2: Vec<f32>,
    c: usize,
}

impl Quantizer {
    fn load(w: &Weights, p: &str) -> Result<Quantizer> {
        let cb = w.f32(&format!("{p}.codebook"))?;
        let mut cbn = vec![0f32; cb.len()];
        let mut cbn2 = vec![0f32; cb.len() / 8];
        for (i, r) in cb.chunks_exact(8).enumerate() {
            let n = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for j in 0..8 {
                cbn[i * 8 + j] = r[j] / n;
            }
            cbn2[i] = cbn[i * 8..i * 8 + 8].iter().map(|x| x * x).sum();
        }
        let in_w = w.f32(&format!("{p}.in_proj.weight"))?;
        Ok(Quantizer { c: in_w.len() / 8, in_w, in_b: w.f32(&format!("{p}.in_proj.bias"))?, out_w: w.f32(&format!("{p}.out_proj.weight"))?,
                       out_b: w.f32(&format!("{p}.out_proj.bias"))?, cb, cbn, cbn2 })
    }

    /// x [t, c] -> (quantized [t, c], ids [t]); the first index on a tie
    fn run(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<u32>) {
        let c = self.c;
        let mut q = vec![0f32; t * c];
        let mut ids = vec![0u32; t];
        for r in 0..t {
            let xr = &x[r * c..(r + 1) * c];
            let mut z = [0f32; 8];
            for (j, zj) in z.iter_mut().enumerate() {
                *zj = self.in_b[j] + xr.iter().zip(&self.in_w[j * c..(j + 1) * c]).map(|(a, b)| a * b).sum::<f32>();
            }
            let nz = z.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let n: Vec<f32> = z.iter().map(|v| v / nz).collect();
            let nn: f32 = n.iter().map(|v| v * v).sum();
            let mut best = (f32::INFINITY, 0usize);
            for k in 0..self.cbn2.len() {
                let dot: f32 = (0..8).map(|j| n[j] * self.cbn[k * 8 + j]).sum();
                let d = (nn - 2.0 * dot) + self.cbn2[k];
                if d < best.0 {
                    best = (d, k);
                }
            }
            ids[r] = best.1 as u32;
            let code = &self.cb[best.1 * 8..best.1 * 8 + 8];
            for o in 0..c {
                q[r * c + o] = self.out_b[o] + (0..8).map(|j| code[j] * self.out_w[o * 8 + j]).sum::<f32>();
            }
        }
        (q, ids)
    }
}

/// The reference encoder with its teacher
pub struct RefEncoder {
    pub ssl: SslTeacher,
    inc: Conv,
    blocks: Vec<DownBlock>,
    stages: Vec<EncStage>,
    sem_in: Mat,
    sem_in_b: DevBuf,
    sem_enc: Vec<Block>,
    sem_head: Vec<Block>,
    sem_out: Mat,
    sem_out_b: DevBuf,
    rvq: Vec<Quantizer>,
    resample: Vec<f64>,
    rs: (usize, usize, usize),
    window_ratio: f32,
    ln_eps: f32,
    cn_eps: f32,
    ones: DevBuf,
    /// the production BF16 recipe (NS_Z3_ENC_F32=1: float32 activations and matrices, a diagnostic: the acoustic
    /// convolutions keep their bfloat16 weights)
    bf: bool,
}

/// A ConvNeXt block of the encoder file (its depthwise taps stored [K, C]: made [C, K] here)
fn enc_block(ops: &Ops, w: &Weights, p: &str) -> Result<Block> {
    let mut bytes = 0;
    let mut b = crate::codec::block(ops, w, p, &mut bytes)?;
    let s = w.shape(&format!("{p}.dw.weight"))?; // [K, C]
    let v = w.f32(&format!("{p}.dw.weight"))?;
    let (k, c) = (s[0], s[1]);
    let t: Vec<f32> = (0..c * k).map(|i| v[(i % k) * c + i / k]).collect();
    b.dw = ops.from_f32(&t)?;
    b.k = k;
    Ok(b)
}

impl RefEncoder {
    pub fn load(ops: &Ops, enc: &Weights, ssl: &Weights, log: &mut dyn FnMut(String)) -> Result<RefEncoder> {
        if enc.g.architecture() != "mossdual_encoder" {
            return Err(Error(format!("{}: not the dual reference encoder (mossdual_encoder)", enc.name())));
        }
        let k = |s: &str| format!("mossdual_encoder.{s}");
        if (enc.u(&k("lane_frames"))?, enc.u(&k("context_frames"))?, enc.u(&k("hop"))?, enc.u(&k("internal_hop"))?) != (LANE, CONTEXT, HOP, IHOP) {
            return Err(Error(format!("{}: an encoder recipe of other lanes or hops", enc.name())));
        }
        let mut blocks = Vec::new();
        for b in 0..2 {
            let p = format!("enc.b{b}");
            let mut res = Vec::new();
            for (r, dil) in [1usize, 3, 9].into_iter().enumerate() {
                let q = format!("{p}.res{r}");
                res.push(ResUnit { s1: snake(ops, enc, &format!("{q}.snake1"))?, c1: conv(ops, enc, &format!("{q}.conv1"))?,
                                   s2: snake(ops, enc, &format!("{q}.snake2"))?, c2: conv(ops, enc, &format!("{q}.conv2"))?, dil });
            }
            blocks.push(DownBlock { res, snake: snake(ops, enc, &format!("{p}.snake"))?, conv: conv(ops, enc, &format!("{p}.conv"))?, stride: if b == 0 { 2 } else { 4 } });
        }
        let cols = |key: &str| enc.us(&k(&format!("stage.{key}")));
        let (dm, nh, nl, ctx, patch) = (cols("d_model")?, cols("n_head")?, cols("n_layer")?, cols("context")?, cols("patch")?);
        let mut stages = Vec::new();
        for s in 0..dm.len() {
            let p = format!("stage.{s}");
            let layers = (0..nl[s]).map(|l| moss_weights(ops, enc, &format!("{p}.l{l}"))).collect::<Result<Vec<_>>>()?;
            stages.push(EncStage { inp: enc.mat(ops, &format!("{p}.in.weight"))?, layers, out: enc.mat(ops, &format!("{p}.out.weight"))?, d: dm[s], heads: nh[s],
                                   context: ctx[s], patch: patch[s] });
        }
        let depth = enc.u(&k("semantic_depth"))?;
        let bf = std::env::var("NS_Z3_ENC_F32").map_or(true, |v| v != "1");
        let mut teacher = SslTeacher::load(ops, ssl)?;
        if bf {
            teacher.use_bf16(ops)?;
        }
        let r = RefEncoder {
            ssl: teacher,
            inc: conv(ops, enc, "enc.in")?,
            blocks,
            stages,
            sem_in: enc.mat(ops, "sem_enc.in.weight")?,
            sem_in_b: enc.dev(ops, "sem_enc.in.bias")?,
            sem_enc: (0..depth).map(|b| enc_block(ops, enc, &format!("sem_enc.{b}"))).collect::<Result<Vec<_>>>()?,
            sem_head: (0..depth).map(|b| enc_block(ops, enc, &format!("sem_head.{b}"))).collect::<Result<Vec<_>>>()?,
            sem_out: enc.mat(ops, "sem_head.out.weight")?,
            sem_out_b: enc.dev(ops, "sem_head.out.bias")?,
            rvq: (0..10).map(|c| Quantizer::load(enc, &format!("rvq.{c}"))).collect::<Result<Vec<_>>>()?,
            resample: enc.f64("resample.kernel")?,
            rs: (enc.u(&k("resample.orig"))?, enc.u(&k("resample.new"))?, enc.u(&k("resample.width"))?),
            window_ratio: enc.f(&k("window_ratio"))? as f32,
            ln_eps: enc.f(&k("ln_eps"))? as f32,
            cn_eps: enc.f(&k("convnext_eps"))? as f32,
            ones: ops.from_f32(&vec![1.0f32; 4096])?,
            bf,
        };
        if bf {
            for st in &r.stages {
                round_dev(ops, &st.inp.w)?;
                round_dev(ops, &st.out.w)?;
                for ly in &st.layers {
                    ly.round_bf16(ops)?;
                }
            }
            for d in [&r.sem_in.w, &r.sem_in_b, &r.sem_out.w, &r.sem_out_b] {
                round_dev(ops, d)?;
            }
            for b in r.sem_enc.iter().chain(&r.sem_head) {
                round_block(ops, b)?;
            }
        }
        log(format!("zonos3: reference encoder and SSL teacher loaded ({} MOSS stages, {})", r.stages.len(), if bf { "BF16 recipe" } else { "float32" }));
        Ok(r)
    }

    fn r(&self, ops: &Ops, p: *mut f32, n: usize) -> Result<()> {
        if self.bf {
            ops.bf16(p, n)?;
        }
        Ok(())
    }

    /// A causal convolution of channels-first x [ci, l]: left padded (k - 1) dil + 1 - stride, out [co, lo]; BF16:
    /// its input rounded, its output before and after the bias (zac-encoder.cpp encoder_conv)
    fn causal(&self, ops: &Ops, c: &Conv, x: &DevBuf, l: usize, dil: usize, stride: usize, pad: &DevBuf, out: &DevBuf) -> Result<usize> {
        let left = (c.k - 1) * dil + 1 - stride;
        let lp = l + left;
        pad.fill(0).map_err(e)?;
        pad.copy_2d(left * 4, lp * 4, x, 0, l * 4, l * 4, c.ci).map_err(e)?;
        self.r(ops, pad.fp(), c.ci * lp)?;
        let lo = ops.conv1d(pad.fp(), c.ci, lp, c.w.fp(), c.co, c.k, nulf(), stride, dil, out.fp())?;
        self.r(ops, out.fp(), c.co * lo)?;
        ops.add_rows(out.fp(), c.b.fp(), c.co, lo)?;
        self.r(ops, out.fp(), c.co * lo)?;
        Ok(lo)
    }

    /// A stage projection: BF16 operands, the output rounded
    fn proj(&self, ops: &Ops, x: &DevBuf, t: usize, m: &Mat, out: &DevBuf) -> Result<()> {
        self.r(ops, x.fp(), t * m.k)?;
        ops.linear(x.fp(), t, m, nulf(), out.fp())?;
        self.r(ops, out.fp(), t * m.n)
    }

    /// One lane's acoustic latent [300, 1024] from its 44160 Hz samples (LANE x 2944)
    fn acoustic(&self, ops: &Ops, lane: &[f32]) -> Result<Vec<f32>> {
        let l0 = lane.len();
        // the widest activation is 64 channels at the full rate (the strided blocks keep channels x length)
        let buf = |n: usize| ops.f32(n);
        let (x, y, z, pad) = (buf(64 * l0)?, buf(64 * l0)?, buf(64 * l0)?, buf(64 * l0 + 128 * 64)?);
        let input = ops.from_f32(lane)?;
        let mut l = self.causal(ops, &self.inc, &input, l0, 1, 1, &pad, &x)?;
        let mut c = self.inc.co;
        dump_cf(ops, "enc.in", &x, c, l)?;
        for (bi, b) in self.blocks.iter().enumerate() {
            for (ri, r) in b.res.iter().enumerate() {
                ops.snake_cf(x.fp(), c, l, r.s1.alpha.fp(), r.s1.inv.fp(), y.fp())?;
                self.causal(ops, &r.c1, &y, l, r.dil, 1, &pad, &z)?;
                ops.snake_cf(z.fp(), c, l, r.s2.alpha.fp(), r.s2.inv.fp(), y.fp())?;
                self.causal(ops, &r.c2, &y, l, 1, 1, &pad, &z)?;
                ops.add(x.fp(), z.fp(), c * l)?;
                self.r(ops, x.fp(), c * l)?;
                dump_cf(ops, &format!("enc.b{bi}.res{ri}"), &x, c, l)?;
            }
            ops.snake_cf(x.fp(), c, l, b.snake.alpha.fp(), b.snake.inv.fp(), y.fp())?;
            l = self.causal(ops, &b.conv, &y, l, 1, b.stride, &pad, &x)?;
            c = b.conv.co;
            dump_cf(ops, &format!("enc.b{bi}.conv"), &x, c, l)?;
        }
        // channels-first [c, l] -> [l, c]
        ops.swap01(x.fp(), c, l, 1, y.fp())?;
        let (mut cur, mut a, mut bb) = (y, x, z);
        let mut t = l;
        let mut width = c;
        for (si, st) in self.stages.iter().enumerate() {
            ops.patch_down(cur.fp(), t, width, st.patch, a.fp())?;
            t /= st.patch;
            std::mem::swap(&mut cur, &mut a);
            self.proj(ops, &cur, t, &st.inp, &a)?;
            std::mem::swap(&mut cur, &mut a);
            dump_dev(ops, &format!("stage.{si}.in"), &cur, t, st.d)?;
            let big = ops.f32(t * (3 * st.d).max(st.layers.first().map_or(0, |ly| ly.ffn1.n)))?;
            for (li, ly) in st.layers.iter().enumerate() {
                moss_layer(ops, ly, &cur, t, st.d, st.heads, st.context, self.ln_eps, &a, &bb, &big, self.bf)?;
                if li == 0 {
                    dump_dev(ops, &format!("stage.{si}.l0"), &cur, t, st.d)?;
                }
            }
            self.proj(ops, &cur, t, &st.out, &a)?;
            std::mem::swap(&mut cur, &mut a);
            width = st.out.n;
            dump_dev(ops, &format!("stage.{si}"), &cur, t, width)?;
        }
        let _ = &mut bb;
        if t != LANE {
            return Err(Error(format!("the acoustic encoder made {t} frames of a lane, {LANE} expected")));
        }
        Ok(cur.to_f32().map_err(e)?[..t * width].to_vec())
    }

    /// The teacher's T frames pooled onto the lane's 300 by box windows (zac-encoder.cpp pool_ssl)
    fn pool(&self, ssl: &[f32], t: usize, c: usize) -> Vec<f32> {
        let mut prefix = vec![0f32; (t + 1) * c];
        for i in 0..t {
            for j in 0..c {
                prefix[(i + 1) * c + j] = prefix[i * c + j] + ssl[i * c + j];
            }
        }
        let ratio = t as f32 / LANE as f32;
        let half = 0.5 * self.window_ratio * ratio;
        let mut out = vec![0f32; LANE * c];
        for f in 0..LANE {
            let centre = (f as f32 + 0.5) * ratio;
            let (lft, rgt) = ((centre - half).clamp(0.0, t as f32), (centre + half).clamp(0.0, t as f32));
            let (li, ri) = (lft.floor() as usize, rgt.floor() as usize);
            let (lf, rf) = (lft - li as f32, rgt - ri as f32);
            for j in 0..c {
                let a = prefix[li * c + j] + ssl[li.min(t - 1) * c + j] * lf;
                let b = prefix[ri * c + j] + ssl[ri.min(t - 1) * c + j] * rf;
                out[f * c + j] = (b - a) / (rgt - lft);
            }
        }
        out
    }

    /// x [LANE, c] through ConvNeXt blocks (and the projections around them)
    fn semantic(&self, ops: &Ops, x: &[f32], encode: bool) -> Result<Vec<f32>> {
        let t = LANE;
        let inp = ops.from_f32(x)?;
        let c = 1024usize;
        let (cur, a, b, xr) = (ops.f32(t * c)?, ops.f32(t * c)?, ops.f32(t * c)?, ops.f32(t * c)?);
        let blocks = if encode { &self.sem_enc } else { &self.sem_head };
        let d = 768;
        let big = ops.f32(t * blocks.iter().map(|b| b.pw1.n).max().unwrap_or(d))?;
        // a WNConv / output projection: rounded before and after its bias
        let projection = |x: &DevBuf, m: &Mat, bias: &DevBuf, out: &DevBuf| -> Result<()> {
            self.r(ops, x.fp(), t * m.k)?;
            ops.linear(x.fp(), t, m, nulf(), out.fp())?;
            self.r(ops, out.fp(), t * m.n)?;
            ops.add_vec(out.fp(), t, m.n, bias.fp())?;
            self.r(ops, out.fp(), t * m.n)
        };
        if encode {
            projection(&inp, &self.sem_in, &self.sem_in_b, &cur)?;
        } else {
            cur.copy_within(0, &inp, 0, t * d * 4).map_err(e)?;
        }
        for blk in blocks {
            if self.bf {
                self.convnext_bf(ops, blk, &cur, t, d, &xr, &a, &b, &big, encode)?;
            } else {
                convnext(ops, blk, &cur, t, d, self.cn_eps, &self.ones, &a, &b, &big)?;
            }
        }
        if encode {
            return Ok(cur.to_f32().map_err(e)?[..t * d].to_vec());
        }
        projection(&cur, &self.sem_out, &self.sem_out_b, &a)?;
        Ok(a.to_f32().map_err(e)?[..t * self.sem_out.n].to_vec())
    }

    /// A causal ConvNeXt block in the BF16 recipe (codec_decoder.py convnext_block with rnd; round_out: the semantic
    /// encoder's blocks round their output)
    fn convnext_bf(&self, ops: &Ops, blk: &Block, x: &DevBuf, t: usize, c: usize, xr: &DevBuf, a: &DevBuf, b: &DevBuf, big: &DevBuf, round_out: bool)
                   -> Result<()> {
        let r = |p: *mut f32, n: usize| ops.bf16(p, n);
        xr.copy_within(0, x, 0, t * c * 4).map_err(e)?;
        r(xr.fp(), t * c)?;
        ops.dwconv_causal(xr.fp(), t, c, blk.dw.fp(), blk.k, nulf(), a.fp())?;
        r(a.fp(), t * c)?;
        ops.add_vec(a.fp(), t, c, blk.dw_b.fp())?;
        r(a.fp(), t * c)?;
        ops.layer_norm(a.fp(), t, c, blk.ln_w.fp(), blk.ln_b.fp(), self.cn_eps, b.fp())?;
        r(b.fp(), t * c)?;
        ops.linear(b.fp(), t, &blk.pw1, blk.pw1_b.fp(), big.fp())?;
        r(big.fp(), t * blk.pw1.n)?;
        ops.gelu(big.fp(), t * blk.pw1.n, false)?;
        r(big.fp(), t * blk.pw1.n)?;
        ops.linear(big.fp(), t, &blk.pw2, blk.pw2_b.fp(), a.fp())?;
        r(a.fp(), t * c)?;
        ops.scale_add(x.fp(), a.fp(), self.ones.fp(), t, c)?;
        if round_out {
            r(x.fp(), t * c)?;
        }
        Ok(())
    }

    /// Mono audio (any rate) -> ordinary codes [frames, 10], frames = ceil(samples / 2940) of its first 60 s
    pub fn encode(&self, ops: &Ops, samples: &[f32], rate: u32) -> Result<Vec<[u32; 10]>> {
        let pcm = if rate == 44100 { samples.to_vec() } else { resample_any(samples, rate, 44100) };
        let limit = MAX_SECONDS * 44100 / HOP * HOP;
        let audio = &pcm[..pcm.len().min(limit)];
        if audio.is_empty() || audio.iter().any(|x| !x.is_finite()) {
            return Err(Error("the recording must be finite and nonempty".into()));
        }
        let n = audio.len();
        let frames = n.div_ceil(HOP);
        let (orig, new, width) = self.rs;
        let mut internal = polyphase(audio, &self.resample, orig, new, width);
        internal.resize(frames * IHOP, 0.0);
        let mut codes = vec![[0u32; 10]; frames];
        let mut start = 0;
        while start < frames {
            let context = if start == 0 { 0 } else { CONTEXT };
            let count = (frames - start).min(LANE - context);
            let left = start - context;
            let mut lane = vec![0f32; LANE * IHOP];
            lane[..(context + count) * IHOP].copy_from_slice(&internal[left * IHOP..(left + context + count) * IHOP]);
            let mut teacher = vec![0f32; LANE * HOP];
            let t0 = left * HOP;
            let tn = (n - t0).min((context + count) * HOP);
            teacher[..tn].copy_from_slice(&audio[t0..t0 + tn]);
            let (ssl, tt) = self.ssl.encode(ops, &teacher)?;
            let acoustic = self.acoustic(ops, &lane)?;
            let pooled = self.pool(&ssl, tt, self.ssl.d);
            let semantic = self.semantic(ops, &pooled, true)?;
            let (q1, ids0) = self.rvq[0].run(&semantic, LANE);
            let pred = self.semantic(ops, &q1, false)?;
            if start == 0 {
                dump("ssl", &[tt, self.ssl.d], &ssl);
                dump("pooled", &[LANE, self.ssl.d], &pooled);
                dump("semantic", &[LANE, 768], &semantic);
                dump("q1", &[LANE, 768], &q1);
                dump("semantic_prediction", &[LANE, pred.len() / LANE], &pred);
                dump("acoustic", &[LANE, acoustic.len() / LANE], &acoustic);
            }
            let mut residual: Vec<f32> = acoustic.iter().zip(&pred).map(|(a, p)| a - p).collect();
            let mut all = vec![ids0];
            for q in &self.rvq[1..] {
                let (qr, ids) = q.run(&residual, LANE);
                residual.iter_mut().zip(&qr).for_each(|(r, v)| *r -= v);
                all.push(ids);
            }
            for f in 0..count {
                for (cb, ids) in all.iter().enumerate() {
                    codes[start + f][cb] = ids[context + f];
                }
            }
            start += count;
        }
        Ok(codes)
    }
}

/// A shared, lazily loaded encoder
pub type Shared = Arc<RefEncoder>;
