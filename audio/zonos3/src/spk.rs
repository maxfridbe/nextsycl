//! The speaker encoder (`mect_spk`, ownmect_s2: MECT-B2 + ASTP, 192-d): a recording to the embedding the backbone's
//! speaker row projects. The frontend runs on the host in float64 (16 kHz by the GGUF's polyphase kernel, Kaldi fbank:
//! x32768, 25 / 10 ms frames, the frame's mean removed, pre-emphasis 0.97, the stored Hamming window, a 512-point
//! power spectrum, 80 mel banks, a log floor; then the mean over time removed); the network on the card, channels-last
//! [F, T, C]: a 3x3 stem, four stages of MECT blocks (grouped 3x3 residual CNNs, then the flattened features through a
//! projection, a positional depthwise convolution, 4-head attention and a 4-expert dense-softmax FFN, added back),
//! the stages' features fused by learned weights, attentive statistics pooling, a batch norm and a 192-d projection.
//!
//! Origin: Zyphra's `zonos3.models.mect_speaker` as the ggml server's spk-encoder.cpp and the MLX port run it
//! (github.com/Zyphra/ZONOS3-Mini, MIT; the model Apache-2.0). The embedding is not L2-normalized.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{nulf, Mat, Ops, Weights};

pub const ARCH: &str = "mect_spk";
const HIDDEN: usize = 64;
const FEAT: usize = 2560;
/// per stage, per block: (cnn depth, stride F, stride T) - the rest comes from the weights
const MECT_B2: [&[(usize, usize, usize)]; 4] = [
    &[(3, 1, 1), (3, 1, 1)],
    &[(3, 2, 1), (3, 1, 2), (3, 1, 2), (3, 1, 2)],
    &[(3, 2, 1), (3, 1, 2), (3, 1, 2), (3, 1, 2), (3, 1, 2)],
    &[(3, 2, 1), (2, 1, 2), (2, 1, 2)],
];

struct Grouped {
    w: DevBuf,
    co: usize,
    a: usize,
}

struct Affine {
    scale: DevBuf,
    shift: DevBuf,
}

struct ResBlock {
    conv1: Grouped,
    pw1: Mat,
    pw1_b: DevBuf,
    bn1: Affine,
    conv2: Grouped,
    pw2: Mat,
    pw2_b: DevBuf,
    bn2: Affine,
    sc: Option<(Mat, Affine)>,
}

struct Block {
    cnn: Vec<ResBlock>,
    reduce: Option<(Mat, DevBuf, Affine)>,
    sf: usize,
    st: usize,
    rproj: (Mat, DevBuf),
    rproj_ln: (DevBuf, DevBuf),
    pos_dw: (DevBuf, DevBuf, usize),
    pos_bn: Affine,
    pos_pw: (Mat, DevBuf),
    q: (Mat, DevBuf),
    k: (Mat, DevBuf),
    v: (Mat, DevBuf),
    o: (Mat, DevBuf),
    ln1: (DevBuf, DevBuf),
    gate: Mat,
    experts: Mat,
    ffn_out: (Mat, DevBuf),
    ln2: (DevBuf, DevBuf),
    eproj: (Mat, DevBuf),
}

pub struct SpeakerEncoder {
    window: Vec<f64>,
    mel: Vec<f64>,
    resample: Vec<f64>,
    n_mels: usize,
    flen: usize,
    fshift: usize,
    nfft: usize,
    heads: usize,
    preemph: f64,
    scale: f64,
    floor: f64,
    ln_eps: f32,
    var_floor: f32,
    max_seconds: usize,
    head_w: DevBuf,
    head_b: DevBuf,
    head_bn: Affine,
    stages: Vec<Vec<Block>>,
    dense: DevBuf,
    l1: (Mat, DevBuf),
    l2: (Mat, DevBuf),
    bn: (Vec<f32>, Vec<f32>),
    linear: Vec<f32>,
    pub dim: usize,
}

fn affine(ops: &Ops, w: &Weights, p: &str) -> Result<Affine> {
    Ok(Affine { scale: w.dev(ops, &format!("{p}.scale"))?, shift: w.dev(ops, &format!("{p}.shift"))? })
}

fn lin(ops: &Ops, w: &Weights, p: &str) -> Result<(Mat, DevBuf)> {
    Ok((w.mat(ops, &format!("{p}.weight"))?, w.dev(ops, &format!("{p}.bias"))?))
}

fn ln(ops: &Ops, w: &Weights, p: &str) -> Result<(DevBuf, DevBuf)> {
    Ok((w.dev(ops, &format!("{p}.weight"))?, w.dev(ops, &format!("{p}.bias"))?))
}

/// The slots w0, w1 ... of a grouped convolution ([Co, 1, 3, 3] each) as one [Co, a, 3, 3]
fn grouped(ops: &Ops, w: &Weights, p: &str) -> Result<Grouped> {
    let mut slots = Vec::new();
    while w.has(&format!("{p}.w{}", slots.len())) {
        slots.push(w.f32(&format!("{p}.w{}", slots.len()))?);
    }
    if slots.is_empty() {
        return Err(Error(format!("{}: no {p}.w0", w.name())));
    }
    let a = slots.len();
    let co = slots[0].len() / 9;
    let mut v = vec![0f32; co * a * 9];
    for (j, s) in slots.iter().enumerate() {
        for o in 0..co {
            v[(o * a + j) * 9..(o * a + j) * 9 + 9].copy_from_slice(&s[o * 9..o * 9 + 9]);
        }
    }
    Ok(Grouped { w: ops.from_f32(&v)?, co, a })
}

/// A 1x1 convolution's weight [Co, Ci, 1, 1] as a matrix [Co, Ci]
fn pw(ops: &Ops, w: &Weights, name: &str) -> Result<Mat> {
    let s = w.shape(name)?;
    let v = w.f32(name)?;
    Ok(Mat { w: ops.from_f32(&v)?, n: s[0], k: s[1] })
}

/// An in-place iterative radix-2 FFT of re / im (length a power of two)
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (s + k, s + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// torchaudio's polyphase resampler with a kernel [new, 2 width + orig]: out[n new + i] = sum_k K[i][k] x[n orig + k -
/// width], ceil(new len / orig) samples, accumulated in float64
pub fn polyphase(x: &[f32], kern: &[f64], orig: usize, new: usize, width: usize) -> Vec<f32> {
    let l = x.len();
    let total = (new * l).div_ceil(orig);
    let taps = 2 * width + orig;
    (0..total).map(|o| {
        let (n, i) = (o / new, o % new);
        let kr = &kern[i * taps..(i + 1) * taps];
        let mut s = 0f64;
        for (j, k) in kr.iter().enumerate() {
            let src = (n * orig + j) as isize - width as isize;
            if src >= 0 && (src as usize) < l {
                s += k * x[src as usize] as f64;
            }
        }
        s as f32
    }).collect()
}

/// A windowed-sinc resampler for rates the model's own kernels do not cover (Hann, 32 zero crossings)
pub fn resample_any(x: &[f32], rate: u32, to: u32) -> Vec<f32> {
    if rate == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / rate as f64;
    let cut = ratio.min(1.0);
    let zc = 32.0;
    let half = (zc / cut).ceil() as isize;
    let n = (x.len() as f64 * ratio).round() as usize;
    (0..n).map(|i| {
        let center = i as f64 / ratio;
        let c0 = center.floor() as isize;
        let mut s = 0.0;
        for j in c0 - half..=c0 + half {
            if j < 0 || j as usize >= x.len() {
                continue;
            }
            let d = (j as f64 - center) * cut;
            if d.abs() >= zc {
                continue;
            }
            let sinc = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d) };
            s += x[j as usize] as f64 * sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / zc).cos()) * cut;
        }
        s as f32
    }).collect()
}

impl SpeakerEncoder {
    pub fn load(ops: &Ops, w: &Weights, log: &mut dyn FnMut(String)) -> Result<SpeakerEncoder> {
        if w.g.architecture() != ARCH || w.s(&format!("{ARCH}.backbone"))? != "MECT_B2" {
            return Err(Error(format!("{}: not a MECT-B2 speaker encoder ({ARCH})", w.name())));
        }
        let k = |s: &str| format!("{ARCH}.{s}");
        let to64 = |v: Vec<f32>| v.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let mut stages = Vec::new();
        for (s, cfg) in MECT_B2.iter().enumerate() {
            let mut blocks = Vec::new();
            for (b, (depth, sf, st)) in cfg.iter().enumerate() {
                let p = |x: &str| format!("fe.b{}.{b}.{x}", s + 1);
                let mut cnn = Vec::new();
                for r in 0..*depth {
                    let q = |x: &str| p(&format!("cnn.{r}.{x}"));
                    let sc = if w.has(&q("sc.weight")) { Some((pw(ops, w, &q("sc.weight"))?, affine(ops, w, &q("sc.bn"))?)) } else { None };
                    cnn.push(ResBlock {
                        conv1: grouped(ops, w, &q("conv1"))?,
                        pw1: pw(ops, w, &q("pw1.weight"))?,
                        pw1_b: w.dev(ops, &q("pw1.bias"))?,
                        bn1: affine(ops, w, &q("bn1"))?,
                        conv2: grouped(ops, w, &q("conv2"))?,
                        pw2: pw(ops, w, &q("pw2.weight"))?,
                        pw2_b: w.dev(ops, &q("pw2.bias"))?,
                        bn2: affine(ops, w, &q("bn2"))?,
                        sc,
                    });
                }
                let reduce = if w.has(&p("reduce.weight")) {
                    Some((pw(ops, w, &p("reduce.weight"))?, w.dev(ops, &p("reduce.bias"))?, affine(ops, w, &p("reduce.bn"))?))
                } else {
                    None
                };
                blocks.push(Block {
                    cnn,
                    reduce,
                    sf: *sf,
                    st: *st,
                    rproj: lin(ops, w, &p("rproj"))?,
                    rproj_ln: ln(ops, w, &p("rproj.ln"))?,
                    pos_dw: (w.dev(ops, &p("pos.dw.weight"))?, w.dev(ops, &p("pos.dw.bias"))?, w.shape(&p("pos.dw.weight"))?[1]),
                    pos_bn: affine(ops, w, &p("pos.bn"))?,
                    pos_pw: lin(ops, w, &p("pos.pw"))?,
                    q: lin(ops, w, &p("attn.q"))?,
                    k: lin(ops, w, &p("attn.k"))?,
                    v: lin(ops, w, &p("attn.v"))?,
                    o: lin(ops, w, &p("attn.o"))?,
                    ln1: ln(ops, w, &p("ln1"))?,
                    gate: w.mat(ops, &p("moe.gate.weight"))?,
                    experts: w.mat(ops, &p("moe.experts.weight"))?,
                    ffn_out: lin(ops, w, &p("ffn_out"))?,
                    ln2: ln(ops, w, &p("ln2"))?,
                    eproj: lin(ops, w, &p("eproj"))?,
                });
            }
            stages.push(blocks);
        }
        let head = w.f32("fe.head.conv.weight")?;
        let e = SpeakerEncoder {
            window: to64(w.f32("frontend.window")?),
            mel: to64(w.f32("frontend.mel_banks")?),
            resample: to64(w.f32("frontend.resample_44100")?),
            n_mels: w.u(&k("n_mels"))?,
            flen: w.u(&k("frame_length"))?,
            fshift: w.u(&k("frame_shift"))?,
            nfft: w.u(&k("n_fft"))?,
            heads: w.u(&k("n_heads"))?,
            preemph: w.f(&k("preemphasis"))?,
            scale: w.f(&k("waveform_scale"))?,
            floor: w.f(&k("log_floor"))?,
            ln_eps: w.f(&k("ln_eps"))? as f32,
            var_floor: w.f(&k("pool_var_floor"))? as f32,
            max_seconds: w.u(&k("max_seconds"))?,
            head_w: ops.from_f32(&head)?,
            head_b: w.dev(ops, "fe.head.conv.bias")?,
            head_bn: affine(ops, w, "fe.head.bn")?,
            stages,
            dense: w.dev(ops, "fe.dense_w")?,
            l1: lin(ops, w, "pool.l1")?,
            l2: lin(ops, w, "pool.l2")?,
            bn: (w.f32("bn.scale")?, w.f32("bn.shift")?),
            linear: w.f32("linear.weight")?,
            dim: w.u(&k("embedding_length"))?,
        };
        if e.nfft & (e.nfft - 1) != 0 || e.window.len() != e.flen || e.mel.len() != e.n_mels * (e.nfft / 2 + 1) {
            return Err(Error(format!("{}: an inconsistent fbank frontend", w.name())));
        }
        log(format!("{ARCH}: speaker encoder ({}-d) loaded", e.dim));
        Ok(e)
    }

    /// Kaldi fbank of 16 kHz audio (at most max_seconds) with the mean over time removed: [T, n_mels]
    pub fn fbank(&self, wav16: &[f32]) -> Result<(Vec<f32>, usize)> {
        let wav = &wav16[..wav16.len().min(self.max_seconds * 16000)];
        if wav.len() < self.flen {
            return Err(Error("the recording is shorter than 25 ms".into()));
        }
        let t = 1 + (wav.len() - self.flen) / self.fshift;
        let bins = self.nfft / 2 + 1;
        let mut fb = vec![0f32; t * self.n_mels];
        let (mut re, mut im) = (vec![0f64; self.nfft], vec![0f64; self.nfft]);
        for f in 0..t {
            let fr: Vec<f64> = (0..self.flen).map(|i| wav[f * self.fshift + i] as f64 * self.scale).collect();
            let mean = fr.iter().sum::<f64>() / self.flen as f64;
            let fr: Vec<f64> = fr.iter().map(|x| x - mean).collect();
            re.iter_mut().for_each(|x| *x = 0.0);
            im.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..self.flen {
                let prev = if i == 0 { fr[0] } else { fr[i - 1] };
                re[i] = (fr[i] - self.preemph * prev) * self.window[i];
            }
            fft(&mut re, &mut im);
            for m in 0..self.n_mels {
                let row = &self.mel[m * bins..(m + 1) * bins];
                let e: f64 = (0..bins).map(|b| (re[b] * re[b] + im[b] * im[b]) * row[b]).sum();
                fb[f * self.n_mels + m] = e.max(self.floor).ln() as f32;
            }
        }
        for m in 0..self.n_mels {
            let mean = (0..t).map(|f| fb[f * self.n_mels + m] as f64).sum::<f64>() / t as f64;
            for f in 0..t {
                let i = f * self.n_mels + m;
                fb[i] = (fb[i] as f64 - mean) as f32;
            }
        }
        Ok((fb, t))
    }

    /// A recording (mono, any rate) to its speaker embedding [dim]
    pub fn embed(&self, ops: &Ops, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
        let pcm = if rate == 44100 { samples.to_vec() } else { resample_any(samples, rate, 44100) };
        let pcm = &pcm[..pcm.len().min(self.max_seconds * 44100 + 44100)];
        let wav16 = polyphase(pcm, &self.resample, 441, 160, 17);
        let (fb, t) = self.fbank(&wav16)?;
        self.embed_fbank(ops, &fb, t)
    }

    /// The network over a mean-removed fbank [T, n_mels]
    pub fn embed_fbank(&self, ops: &Ops, fb: &[f32], t: usize) -> Result<Vec<f32>> {
        let nm = self.n_mels;
        // the widest activation (a dry run of the shapes): the first stage's expanded channels over every mel
        let mut cap = t * FEAT;
        let mut fdry = nm;
        for stage in &self.stages {
            for b in stage {
                for (r, rb) in b.cnn.iter().enumerate() {
                    if r == 0 {
                        fdry = (fdry - 1) / b.sf + 1;
                    }
                    cap = cap.max(fdry * t * rb.conv1.co.max(rb.pw1.n).max(rb.pw2.n).max(rb.conv2.co));
                }
            }
        }
        let bufs = (0..5).map(|_| ops.f32(cap)).collect::<Result<Vec<_>>>()?;
        let one64 = ops.from_f32(&[1.0f32; HIDDEN])?;
        let zero64 = ops.from_f32(&[0.0f32; HIDDEN])?;
        let ones = ops.from_f32(&vec![1.0f32; FEAT])?;
        let zeros = ops.from_f32(&vec![0.0f32; FEAT])?;
        let fin = ops.f32(t * nm)?;
        // [T, mels] -> [mels, T, 1]
        let mut xt = vec![0f32; t * nm];
        for f in 0..t {
            for m in 0..nm {
                xt[m * t + f] = fb[f * nm + m];
            }
        }
        let fin_b: Vec<u8> = xt.iter().flat_map(|v| v.to_le_bytes()).collect();
        fin.write(0, &fin_b).map_err(crate::ops::e)?;
        let c0 = self.head_bn.scale.floats();
        let x = ops.f32(t * FEAT)?;
        ops.conv3x3(fin.fp(), nm, t, 1, self.head_w.fp(), c0, 1, 1, 1, self.head_b.fp(), bufs[0].fp())?;
        ops.affine(bufs[0].fp(), nm * t, c0, self.head_bn.scale.fp(), self.head_bn.shift.fp(), false, nulf(), x.fp())?;
        let xf = ops.f32(t * FEAT)?;
        ops.swap01(x.fp(), nm, t, c0, xf.fp())?;
        let (mut fdim, mut cdim) = (nm, c0);
        // per-block scratch on the flattened rows
        let rows = |n: usize| ops.f32(t * n);
        let (x2, h, a1, a2, at) = (rows(FEAT)?, rows(HIDDEN)?, rows(HIDDEN)?, rows(HIDDEN)?, rows(HIDDEN)?);
        let (qb, kb, vb) = (rows(HIDDEN)?, rows(HIDDEN)?, rows(HIDDEN)?);
        let (ex, r4, e2) = (rows(4 * HIDDEN)?, rows(4)?, rows(FEAT)?);
        let mut feats = Vec::new();
        for stage in &self.stages {
            for b in stage {
                // the residual CNN, strided by the block's first
                let (mut f, mut tt, mut c) = (fdim, t, cdim);
                let mut cur_is_x = true;
                let mut cur = 3usize;  // bufs[3] holds the running y once the first res block ran
                for (r, rb) in b.cnn.iter().enumerate() {
                    let (sf, st) = if r == 0 { (b.sf, b.st) } else { (1, 1) };
                    let src = if cur_is_x { x.fp() } else { bufs[cur].fp() };
                    let (fo, to) = ((f - 1) / sf + 1, (tt - 1) / st + 1);
                    let co1 = rb.conv1.co;
                    ops.conv3x3(src, f, tt, c, rb.conv1.w.fp(), co1, rb.conv1.a, sf, st, nulf(), bufs[0].fp())?;
                    ops.linear(bufs[0].fp(), fo * to, &rb.pw1, rb.pw1_b.fp(), bufs[1].fp())?;
                    let c1 = rb.pw1.n;
                    ops.affine(bufs[1].fp(), fo * to, c1, rb.bn1.scale.fp(), rb.bn1.shift.fp(), true, nulf(), bufs[1].fp())?;
                    ops.conv3x3(bufs[1].fp(), fo, to, c1, rb.conv2.w.fp(), rb.conv2.co, rb.conv2.a, 1, 1, nulf(), bufs[0].fp())?;
                    ops.linear(bufs[0].fp(), fo * to, &rb.pw2, rb.pw2_b.fp(), bufs[1].fp())?;
                    let c2 = rb.pw2.n;
                    let out = if cur == 3 && !cur_is_x { 4 } else { 3 };
                    let sc = match &rb.sc {
                        Some((m, bn)) => {
                            ops.subsample(src, f, tt, c, sf, st, bufs[0].fp())?;
                            ops.linear(bufs[0].fp(), fo * to, m, nulf(), bufs[2].fp())?;
                            ops.affine(bufs[2].fp(), fo * to, c2, bn.scale.fp(), bn.shift.fp(), false, nulf(), bufs[2].fp())?;
                            bufs[2].fp() as *const f32
                        }
                        None => {
                            if (fo, to, c2) != (f, tt, c) {
                                return Err(Error("a residual block without a shortcut changes its shape".into()));
                            }
                            src as *const f32
                        }
                    };
                    ops.affine(bufs[1].fp(), fo * to, c2, rb.bn2.scale.fp(), rb.bn2.shift.fp(), true, sc, bufs[out].fp())?;
                    cur = out;
                    cur_is_x = false;
                    (f, tt, c) = (fo, to, c2);
                }
                if let Some((m, bias, bn)) = &b.reduce {
                    ops.linear(bufs[cur].fp(), f * tt, m, bias.fp(), bufs[0].fp())?;
                    ops.affine(bufs[0].fp(), f * tt, m.n, bn.scale.fp(), bn.shift.fp(), false, nulf(), bufs[0].fp())?;
                    cur = 0;
                    c = m.n;
                }
                if f * c != FEAT {
                    return Err(Error(format!("a MECT block's features are {} wide, {FEAT} expected", f * c)));
                }
                // flattened [T2, F*C]
                ops.swap01(bufs[cur].fp(), f, tt, c, x2.fp())?;
                let t2 = tt;
                ops.linear(x2.fp(), t2, &b.rproj.0, b.rproj.1.fp(), a1.fp())?;
                ops.layer_norm(a1.fp(), t2, HIDDEN, b.rproj_ln.0.fp(), b.rproj_ln.1.fp(), self.ln_eps, h.fp())?;
                // positional convolution: depthwise (centred, zero padded), BN, exact GELU, pointwise; added
                ops.dwconv_centred(h.fp(), t2, HIDDEN, b.pos_dw.0.fp(), b.pos_dw.2, b.pos_dw.1.fp(), a1.fp())?;
                ops.affine(a1.fp(), t2, HIDDEN, b.pos_bn.scale.fp(), b.pos_bn.shift.fp(), false, nulf(), a1.fp())?;
                ops.gelu(a1.fp(), t2 * HIDDEN, false)?;
                ops.linear(a1.fp(), t2, &b.pos_pw.0, b.pos_pw.1.fp(), a2.fp())?;
                ops.affine(a2.fp(), t2, HIDDEN, one64.fp(), zero64.fp(), false, h.fp(), h.fp())?;
                // attention (4 heads of 16, all frames), then LN(h + attn)
                ops.linear(h.fp(), t2, &b.q.0, b.q.1.fp(), qb.fp())?;
                ops.linear(h.fp(), t2, &b.k.0, b.k.1.fp(), kb.fp())?;
                ops.linear(h.fp(), t2, &b.v.0, b.v.1.fp(), vb.fp())?;
                let hd = HIDDEN / self.heads;
                ops.attention(qb.fp(), HIDDEN, t2, kb.fp(), vb.fp(), HIDDEN, t2, self.heads, self.heads, hd, 1.0 / (hd as f32).sqrt(), 0, false, 0, at.fp())?;
                ops.linear(at.fp(), t2, &b.o.0, b.o.1.fp(), a1.fp())?;
                ops.affine(a1.fp(), t2, HIDDEN, one64.fp(), zero64.fp(), false, h.fp(), a1.fp())?;
                ops.layer_norm(a1.fp(), t2, HIDDEN, b.ln1.0.fp(), b.ln1.1.fp(), self.ln_eps, h.fp())?;
                // the dense-softmax token MoE: softmax(gate), each expert's NewGELU, mixed, NewGELU, out; LN(h + it)
                ops.linear(h.fp(), t2, &b.gate, nulf(), r4.fp())?;
                let ne = b.gate.n;
                ops.softmax(r4.fp(), t2, ne)?;
                ops.linear(h.fp(), t2, &b.experts, nulf(), ex.fp())?;
                ops.gelu(ex.fp(), t2 * b.experts.n, true)?;
                ops.mix_experts(ex.fp(), r4.fp(), t2, ne, HIDDEN, a1.fp())?;
                ops.gelu(a1.fp(), t2 * HIDDEN, true)?;
                ops.linear(a1.fp(), t2, &b.ffn_out.0, b.ffn_out.1.fp(), a2.fp())?;
                ops.affine(a2.fp(), t2, HIDDEN, one64.fp(), zero64.fp(), false, h.fp(), a2.fp())?;
                ops.layer_norm(a2.fp(), t2, HIDDEN, b.ln2.0.fp(), b.ln2.1.fp(), self.ln_eps, h.fp())?;
                // back to the features: eproj(h) + x2, repeated to T when the CNN strided time, + the block's input
                ops.linear(h.fp(), t2, &b.eproj.0, b.eproj.1.fp(), e2.fp())?;
                ops.affine(e2.fp(), t2, FEAT, ones.fp(), zeros.fp(), false, x2.fp(), e2.fp())?;
                ops.repeat_rows(e2.fp(), b.st, t, FEAT, xf.fp(), x2.fp())?;
                xf.copy_within(0, &x2, 0, t * FEAT * 4).map_err(crate::ops::e)?;
                // the next block's input: [F, T, C] again
                ops.swap01(xf.fp(), t, f, c, x.fp())?;
                (fdim, cdim) = (f, c);
            }
            let keep = ops.f32(t * FEAT)?;
            keep.copy_within(0, &xf, 0, t * FEAT * 4).map_err(crate::ops::e)?;
            feats.push(keep);
        }
        if feats.len() != 4 {
            return Err(Error("MECT-B2 has four stages".into()));
        }
        let fused = ops.f32(t * FEAT)?;
        ops.fuse4([feats[0].fp(), feats[1].fp(), feats[2].fp(), feats[3].fp()], self.dense.fp(), t, FEAT, fused.fp())?;
        let p1 = ops.f32(t * self.l1.0.n)?;
        ops.linear(fused.fp(), t, &self.l1.0, self.l1.1.fp(), p1.fp())?;
        ops.tanh(p1.fp(), t * self.l1.0.n)?;
        let att = ops.f32(t * FEAT)?;
        ops.linear(p1.fp(), t, &self.l2.0, self.l2.1.fp(), att.fp())?;
        let pooled = ops.f32(2 * FEAT)?;
        ops.astp(fused.fp(), att.fp(), t, FEAT, self.var_floor, pooled.fp())?;
        let p = pooled.to_f32().map_err(crate::ops::e)?;
        let z: Vec<f32> = p.iter().enumerate().map(|(i, v)| v * self.bn.0[i] + self.bn.1[i]).collect();
        let n = z.len();
        let emb: Vec<f32> = (0..self.dim).map(|o| (0..n).map(|i| self.linear[o * n + i] as f64 * z[i] as f64).sum::<f64>() as f32).collect();
        if emb.iter().any(|v| !v.is_finite()) {
            return Err(Error("the speaker encoder made a non-finite embedding".into()));
        }
        Ok(emb)
    }

}
