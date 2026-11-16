//! The ZAC codec's decoder (`mossdual_istft`): ten codes a frame at 15 Hz to 44.1 kHz. The semantic code's table and
//! causal ConvNeXt head plus the nine residual tables, five windowed-causal MOSS transformer stages with patch
//! upsampling (15 -> 240 Hz), the multi-resolution iSTFT head (ConvNeXt trunk; log magnitude and phase for n_fft 736
//! and 2208, hop 184; the inverse real DFT as one product with its basis, overlap-add over the squared window's
//! envelope), the two branches' FIR mixer at 44160 Hz and the exact 44160 -> 44100 resampler.
//!
//! Origin: Zyphra's `zonos3.codec.istft_head_a_20261002` as the ggml server's zac.cpp and the MLX port run it
//! (github.com/Zyphra/ZONOS3-Mini, MIT; the model Apache-2.0): over a whole utterance the streaming decoder equals one
//! causal pass, which is what `decode` runs.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;

use crate::ops::{fp, nulf, Mat, Ops, Weights};

pub const ARCH: &str = "mossdual_istft";

pub struct Block {
    pub(crate) dw: DevBuf,
    pub(crate) k: usize,
    pub(crate) dw_b: DevBuf,
    pub(crate) ln_w: DevBuf,
    pub(crate) ln_b: DevBuf,
    pub(crate) pw1: Mat,
    pub(crate) pw1_b: DevBuf,
    pub(crate) pw2: Mat,
    pub(crate) pw2_b: DevBuf,
}

pub struct MossLayer {
    n1_w: DevBuf,
    n1_b: DevBuf,
    qkv: Mat,
    wo: Mat,
    attn_scale: DevBuf,
    n2_w: DevBuf,
    n2_b: DevBuf,
    pub(crate) ffn1: Mat,
    ffn2: Mat,
    ffn_scale: DevBuf,
}

struct Stage {
    inp: Mat,
    layers: Vec<MossLayer>,
    out: Option<Mat>,
    d: usize,
    heads: usize,
    context: usize,
    patch: usize,
}

/// One MOSS transformer layer in place on x [t, d] (stage positions 0..t): pre-LN windowed-causal attention (adjacent-pair
/// RoPE, keys within `context` before each query) and a GELU FFN, each added through its layer scale. `a`, `b` [t, d] and
/// `big` [t, max(3d, ffn)] scratch
#[allow(clippy::too_many_arguments)]
/// One MOSS layer in place on x [t, d]; `bf`: the reference encoder's BF16 recipe (every matmul's operands and output,
/// the rotated q and k and the attention output rounded to bfloat16; codec_decoder.py moss_layer's `rnd`)
pub fn moss_layer(ops: &Ops, ly: &MossLayer, x: &DevBuf, t: usize, d: usize, heads: usize, context: usize, ln_eps: f32, a: &DevBuf, b: &DevBuf, big: &DevBuf,
                  bf: bool) -> Result<()> {
    let r = |p: *mut f32, n: usize| if bf { ops.bf16(p, n) } else { Ok(()) };
    let hd = d / heads;
    ops.layer_norm(x.fp(), t, d, ly.n1_w.fp(), ly.n1_b.fp(), ln_eps, a.fp())?;
    r(a.fp(), t * d)?;
    ops.linear(a.fp(), t, &ly.qkv, nulf(), big.fp())?;
    r(big.fp(), t * 3 * d)?;
    ops.rope(big.fp(), 3 * d, t, heads, hd, 10000.0, 0)?;
    ops.rope(fp(big, d), 3 * d, t, heads, hd, 10000.0, 0)?;
    r(big.fp(), t * 3 * d)?;
    ops.attention(big.fp(), 3 * d, t, fp(big, d), fp(big, 2 * d), 3 * d, t, heads, heads, hd, (hd as f32).powf(-0.5), 0, true, context, a.fp())?;
    r(a.fp(), t * d)?;
    ops.linear(a.fp(), t, &ly.wo, nulf(), b.fp())?;
    r(b.fp(), t * d)?;
    ops.scale_add(x.fp(), b.fp(), ly.attn_scale.fp(), t, d)?;
    ops.layer_norm(x.fp(), t, d, ly.n2_w.fp(), ly.n2_b.fp(), ln_eps, a.fp())?;
    r(a.fp(), t * d)?;
    ops.linear(a.fp(), t, &ly.ffn1, nulf(), big.fp())?;
    r(big.fp(), t * ly.ffn1.n)?;
    ops.gelu(big.fp(), t * ly.ffn1.n, false)?;
    r(big.fp(), t * ly.ffn1.n)?;
    ops.linear(big.fp(), t, &ly.ffn2, nulf(), b.fp())?;
    r(b.fp(), t * d)?;
    ops.scale_add(x.fp(), b.fp(), ly.ffn_scale.fp(), t, d)
}

impl MossLayer {
    /// The matrices' values rounded to bfloat16 (the reference encoder's BF16 operands)
    pub fn round_bf16(&self, ops: &Ops) -> Result<()> {
        for m in [&self.qkv, &self.wo, &self.ffn1, &self.ffn2] {
            ops.bf16(m.w.fp(), m.n * m.k)?;
        }
        Ok(())
    }
}

/// A MOSS layer's weights (`<p>.norm1` ... `<p>.ffn_scale`)
pub fn moss_weights(ops: &Ops, w: &Weights, p: &str) -> Result<MossLayer> {
    let q = |x: &str| format!("{p}.{x}");
    Ok(MossLayer {
        n1_w: w.dev(ops, &q("norm1.weight"))?,
        n1_b: w.dev(ops, &q("norm1.bias"))?,
        qkv: w.mat(ops, &q("qkv.weight"))?,
        wo: w.mat(ops, &q("wo.weight"))?,
        attn_scale: w.dev(ops, &q("attn_scale"))?,
        n2_w: w.dev(ops, &q("norm2.weight"))?,
        n2_b: w.dev(ops, &q("norm2.bias"))?,
        ffn1: w.mat(ops, &q("ffn1.weight"))?,
        ffn2: w.mat(ops, &q("ffn2.weight"))?,
        ffn_scale: w.dev(ops, &q("ffn_scale"))?,
    })
}

/// x += pw2(gelu(pw1(LN(causal dwconv(x))))): `a`, `b` [t, c] and `big` [t, hidden] scratch
#[allow(clippy::too_many_arguments)]
pub fn convnext(ops: &Ops, blk: &Block, x: &DevBuf, t: usize, c: usize, eps: f32, ones: &DevBuf, a: &DevBuf, b: &DevBuf, big: &DevBuf) -> Result<()> {
    ops.dwconv_causal(x.fp(), t, c, blk.dw.fp(), blk.k, blk.dw_b.fp(), a.fp())?;
    ops.layer_norm(a.fp(), t, c, blk.ln_w.fp(), blk.ln_b.fp(), eps, b.fp())?;
    ops.linear(b.fp(), t, &blk.pw1, blk.pw1_b.fp(), big.fp())?;
    ops.gelu(big.fp(), t * blk.pw1.n, false)?;
    ops.linear(big.fp(), t, &blk.pw2, blk.pw2_b.fp(), a.fp())?;
    ops.scale_add(x.fp(), a.fp(), ones.fp(), t, c)
}

pub struct Decoder {
    /// the decoder's id (`mossdual_istft.head.variant`: keep2_plain, full_mr)
    pub variant: String,
    pub hop: usize,
    pub rate: u32,
    pub ncb: usize,
    pub codebook: usize,
    pub headroom_db: f32,
    sem: DevBuf,
    sem_b: DevBuf,
    sem_blocks: Vec<Block>,
    sem_out: Mat,
    sem_out_b: DevBuf,
    res: Vec<DevBuf>,
    res_b: DevBuf,
    stages: Vec<Stage>,
    head: Vec<Block>,
    head_ln_w: DevBuf,
    head_ln_b: DevBuf,
    head_out: [(Mat, DevBuf); 2],
    /// per branch: n_fft, the inverse DFT's basis [n, n + 2], the window [n]
    branches: Vec<(usize, Mat, DevBuf)>,
    head_hop: usize,
    maxlog: f32,
    mixer: DevBuf,
    mixer_k: usize,
    rs: (usize, usize, usize),
    rs_kernel: DevBuf,
    ln_eps: f32,
    cn_eps: f32,
    lat: usize,
    sem_dim: usize,
    ones: DevBuf,
    pub bytes: usize,
}

pub fn block(ops: &Ops, w: &Weights, p: &str, bytes: &mut usize) -> Result<Block> {
    let pw1 = w.mat(ops, &format!("{p}.pw1.weight"))?;
    let pw2 = w.mat(ops, &format!("{p}.pw2.weight"))?;
    *bytes += pw1.bytes() + pw2.bytes();
    Ok(Block {
        k: w.shape(&format!("{p}.dw.weight"))?[1],
        dw: w.dev(ops, &format!("{p}.dw.weight"))?,
        dw_b: w.dev(ops, &format!("{p}.dw.bias"))?,
        ln_w: w.dev(ops, &format!("{p}.norm.weight"))?,
        ln_b: w.dev(ops, &format!("{p}.norm.bias"))?,
        pw1,
        pw1_b: w.dev(ops, &format!("{p}.pw1.bias"))?,
        pw2,
        pw2_b: w.dev(ops, &format!("{p}.pw2.bias"))?,
    })
}

/// The inverse real DFT of n points as a matrix [n, 2 (n / 2 + 1)] over (real | imaginary) bins: x[t] = sum over
/// bins of (re c cos - im c sin)(2 pi k t / n) / n, c = 1 for the DC and Nyquist bins, else 2
pub fn irfft_basis(n: usize) -> Vec<f32> {
    let bins = n / 2 + 1;
    let mut b = vec![0f32; n * 2 * bins];
    for t in 0..n {
        for k in 0..bins {
            let c = if k == 0 || (n.is_multiple_of(2) && k == bins - 1) { 1.0 } else { 2.0 };
            let a = 2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
            b[t * 2 * bins + k] = (c * a.cos() / n as f64) as f32;
            b[t * 2 * bins + bins + k] = (-c * a.sin() / n as f64) as f32;
        }
    }
    b
}

impl Decoder {
    pub fn load(ops: &Ops, w: &Weights, log: &mut dyn FnMut(String)) -> Result<Decoder> {
        if w.g.architecture() != ARCH {
            return Err(Error(format!("{}: general.architecture is {}, not {ARCH} (the ZONOS3 MR-iSTFT decoder)", w.name(), w.g.architecture())));
        }
        let k = |s: &str| format!("{ARCH}.{s}");
        let (rate, hop, ihop, ncb) = (w.u(&k("sample_rate"))?, w.u(&k("hop"))?, w.u(&k("internal_hop"))?, w.u(&k("n_codebooks"))?);
        if (rate, hop, ihop, ncb) != (44100, 2940, 2944, 10) {
            return Err(Error(format!("{}: a decoder of {rate} Hz, hop {hop} / {ihop}, {ncb} codebooks; this engine runs 44100, 2940 / 2944, 10", w.name())));
        }
        let mut bytes = 0usize;
        let sem_depth = w.u(&k("sem_head.depth"))?;
        let head_depth = w.u(&k("head.depth"))?;
        let sem_blocks = (0..sem_depth).map(|b| block(ops, w, &format!("sem_head.{b}"), &mut bytes)).collect::<Result<Vec<_>>>()?;
        let head = (0..head_depth).map(|b| block(ops, w, &format!("head.{b}"), &mut bytes)).collect::<Result<Vec<_>>>()?;
        let cols = |key: &str| w.us(&k(&format!("stage.{key}")));
        let (din, dout, dm, nh, nl, ctx, patch) = (cols("in_dim")?, cols("out_dim")?, cols("d_model")?, cols("n_head")?, cols("n_layer")?, cols("context")?,
                                                   cols("patch")?);
        let mut stages = Vec::new();
        for s in 0..din.len() {
            let p = format!("stage.{s}");
            let mut layers = Vec::new();
            for l in 0..nl[s] {
                let q = |x: &str| format!("{p}.l{l}.{x}");
                let ly = MossLayer {
                    n1_w: w.dev(ops, &q("norm1.weight"))?,
                    n1_b: w.dev(ops, &q("norm1.bias"))?,
                    qkv: w.mat(ops, &q("qkv.weight"))?,
                    wo: w.mat(ops, &q("wo.weight"))?,
                    attn_scale: w.dev(ops, &q("attn_scale"))?,
                    n2_w: w.dev(ops, &q("norm2.weight"))?,
                    n2_b: w.dev(ops, &q("norm2.bias"))?,
                    ffn1: w.mat(ops, &q("ffn1.weight"))?,
                    ffn2: w.mat(ops, &q("ffn2.weight"))?,
                    ffn_scale: w.dev(ops, &q("ffn_scale"))?,
                };
                bytes += ly.qkv.bytes() + ly.wo.bytes() + ly.ffn1.bytes() + ly.ffn2.bytes();
                layers.push(ly);
            }
            let inp = w.mat(ops, &format!("{p}.in.weight"))?;
            let out = if dout[s] > 0 { Some(w.mat(ops, &format!("{p}.out.weight"))?) } else { None };
            bytes += inp.bytes() + out.as_ref().map_or(0, Mat::bytes);
            if dm[s] % nh[s] != 0 || dm[s] / nh[s] != 64 {
                return Err(Error(format!("{}: stage {s}'s heads are {} wide; this engine's attention takes 64", w.name(), dm[s] / nh[s].max(1))));
            }
            stages.push(Stage { inp, layers, out, d: dm[s], heads: nh[s], context: ctx[s], patch: patch[s] });
        }
        let nfft = w.us(&k("head.n_fft"))?;
        let mut branches = Vec::new();
        for (i, n) in nfft.iter().enumerate() {
            let basis = irfft_basis(*n);
            let m = Mat { w: ops.from_f32(&basis)?, n: *n, k: 2 * (n / 2 + 1) };
            bytes += m.bytes();
            branches.push((*n, m, w.dev(ops, &format!("head.window.{i}"))?));
        }
        let res = (0..ncb - 1).map(|i| w.dev(ops, &format!("res.{i}.table"))).collect::<Result<Vec<_>>>()?;
        let sem = w.dev(ops, "sem.table")?;
        bytes += sem.len + res.iter().map(|r| r.len).sum::<usize>();
        let sem_out = w.mat(ops, "sem_head.out.weight")?;
        let head_out = [(w.mat(ops, "head.out.0.weight")?, w.dev(ops, "head.out.0.bias")?), (w.mat(ops, "head.out.1.weight")?, w.dev(ops, "head.out.1.bias")?)];
        let d = Decoder {
            variant: w.s(&k("head.variant"))?,
            hop,
            rate: rate as u32,
            ncb,
            codebook: w.u(&k("codebook_size"))?,
            headroom_db: w.f(&k("headroom_db"))? as f32,
            sem_b: w.dev(ops, "sem.bias")?,
            sem,
            sem_blocks,
            sem_out_b: w.dev(ops, "sem_head.out.bias")?,
            lat: sem_out.n,
            sem_dim: sem_out.k,
            sem_out,
            res,
            res_b: w.dev(ops, "res.bias")?,
            stages,
            head,
            head_ln_w: w.dev(ops, "head.norm.weight")?,
            head_ln_b: w.dev(ops, "head.norm.bias")?,
            head_out,
            branches,
            head_hop: w.u(&k("head.hop"))?,
            maxlog: w.f(&k("head.max_log_magnitude"))? as f32,
            mixer_k: w.shape("head.mixer.weight")?[1],
            mixer: w.dev(ops, "head.mixer.weight")?,
            rs: (w.u(&k("resample.orig"))?, w.u(&k("resample.new"))?, w.u(&k("resample.width"))?),
            rs_kernel: w.dev(ops, "resample.kernel")?,
            ln_eps: w.f(&k("ln_eps"))? as f32,
            cn_eps: w.f(&k("convnext_eps"))? as f32,
            ones: ops.from_f32(&vec![1.0f32; 4096])?,
            bytes,
        };
        log(format!("{ARCH}: decoder {} ({} stages), {:.0} MiB on the card", d.variant, d.stages.len(), d.bytes as f64 / (1u64 << 20) as f64));
        Ok(d)
    }

    /// x += pw2(gelu(pw1(LN(dwconv(x))))): `a`, `b` [t, c] and `big` [t, hidden] scratch
    fn convnext(&self, ops: &Ops, blk: &Block, x: &DevBuf, t: usize, c: usize, a: &DevBuf, b: &DevBuf, big: &DevBuf) -> Result<()> {
        ops.dwconv_causal(x.fp(), t, c, blk.dw.fp(), blk.k, blk.dw_b.fp(), a.fp())?;
        ops.layer_norm(a.fp(), t, c, blk.ln_w.fp(), blk.ln_b.fp(), self.cn_eps, b.fp())?;
        ops.linear(b.fp(), t, &blk.pw1, blk.pw1_b.fp(), big.fp())?;
        ops.gelu(big.fp(), t * blk.pw1.n, false)?;
        ops.linear(big.fp(), t, &blk.pw2, blk.pw2_b.fp(), a.fp())?;
        ops.scale_add(x.fp(), a.fp(), self.ones.fp(), t, c)
    }

    /// Ordinary codes [frames, ncb] -> float32 PCM [frames * hop] at 44.1 kHz, before the output gain
    pub fn decode(&self, ops: &Ops, codes: &[[u32; 10]]) -> Result<Vec<f32>> {
        let t0 = codes.len();
        if t0 == 0 {
            return Ok(Vec::new());
        }
        let ids: Vec<i32> = codes.iter().flat_map(|r| r.iter().map(|c| (*c as usize).min(self.codebook - 1) as i32)).collect();
        let ids_d = ops.from_i32(&ids)?;
        let up: usize = self.stages.iter().map(|s| s.patch).product();
        let tmax = t0 * up;
        // the widest rows each buffer holds at each rate
        let (mut big_n, mut buf_n, mut t) = (0usize, t0 * self.lat.max(self.sem_dim), t0);
        big_n = big_n.max(t0 * self.sem_blocks.first().map_or(0, |b| b.pw1.n));
        for st in &self.stages {
            let ffn = st.layers.first().map_or(0, |l| l.ffn1.n);
            big_n = big_n.max(t * (3 * st.d).max(ffn));
            buf_n = buf_n.max(t * st.d.max(st.inp.k).max(st.out.as_ref().map_or(0, |o| o.n)));
            t *= st.patch;
        }
        let spec_w = self.branches.iter().map(|b| 2 * (b.0 / 2 + 1)).max().unwrap_or(0);
        big_n = big_n.max(tmax * self.head.first().map_or(0, |b| b.pw1.n)).max(tmax * spec_w);
        buf_n = buf_n.max(tmax * spec_w).max(tmax * self.head_ln_w.floats());
        let big = ops.f32(big_n)?;
        let x = ops.f32(buf_n)?;
        let h = ops.f32(buf_n)?;
        let o = ops.f32(buf_n)?;
        // the semantic code's table and head, then the residual codes
        let (sd, lat) = (self.sem_dim, self.lat);
        ops.lookup(&ids_d, t0, self.ncb, 0, self.sem.fp(), sd, self.sem_b.fp(), false, x.fp())?;
        for b in &self.sem_blocks {
            self.convnext(ops, b, &x, t0, sd, &h, &o, &big)?;
        }
        ops.linear(x.fp(), t0, &self.sem_out, self.sem_out_b.fp(), h.fp())?;
        for (c, tab) in self.res.iter().enumerate() {
            ops.lookup(&ids_d, t0, self.ncb, c + 1, tab.fp(), lat, nulf(), true, h.fp())?;
        }
        ops.add_vec(h.fp(), t0, lat, self.res_b.fp())?;
        // the stages: x in h on entry
        let (mut cur, mut a, b) = (h, x, o);
        let mut t = t0;
        for st in &self.stages {
            ops.linear(cur.fp(), t, &st.inp, nulf(), a.fp())?;
            std::mem::swap(&mut cur, &mut a);
            let d = st.d;
            for ly in &st.layers {
                moss_layer(ops, ly, &cur, t, d, st.heads, st.context, self.ln_eps, &a, &b, &big, false)?;
            }
            let mut width = d;
            if let Some(out) = &st.out {
                ops.linear(cur.fp(), t, out, nulf(), a.fp())?;
                std::mem::swap(&mut cur, &mut a);
                width = out.n;
            }
            if st.patch > 1 {
                ops.patch_up(cur.fp(), t, width / st.patch, st.patch, a.fp())?;
                std::mem::swap(&mut cur, &mut a);
                t *= st.patch;
            }
        }
        // the head's ConvNeXt trunk, its norm and the two branches' spectra
        let c = self.head_ln_w.floats();
        for blk in &self.head {
            self.convnext(ops, blk, &cur, t, c, &a, &b, &big)?;
        }
        ops.layer_norm(cur.fp(), t, c, self.head_ln_w.fp(), self.head_ln_b.fp(), self.cn_eps, a.fp())?;
        std::mem::swap(&mut cur, &mut a);
        let samples = t * self.head_hop;
        let s0 = ops.f32(samples)?;
        let s1 = ops.f32(samples)?;
        for (i, (n, basis, win)) in self.branches.iter().enumerate() {
            let (m, bias) = &self.head_out[i];
            let bins = n / 2 + 1;
            ops.linear(cur.fp(), t, m, bias.fp(), big.fp())?;
            ops.spec(big.fp(), t, bins, self.maxlog, a.fp())?;
            let frames = ops.f32(t * n)?;
            ops.linear(a.fp(), t, basis, nulf(), frames.fp())?;
            ops.ola(frames.fp(), t, *n, self.head_hop, win.fp(), if i == 0 { s0.fp() } else { s1.fp() })?;
        }
        let mixed = ops.f32(samples)?;
        ops.mixer(s0.fp(), s1.fp(), samples, self.mixer.fp(), self.mixer_k, mixed.fp())?;
        let (orig, nw, width) = self.rs;
        let total = (nw * samples).div_ceil(orig);
        let out = ops.f32(total)?;
        ops.resample(mixed.fp(), samples, self.rs_kernel.fp(), orig, nw, width, out.fp(), total)?;
        let mut pcm = out.to_f32().map_err(crate::ops::e)?;
        pcm.truncate(total);
        pcm.resize(t0 * self.hop, 0.0);
        Ok(pcm)
    }
}
