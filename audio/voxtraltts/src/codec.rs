//! The Voxtral codec's decoder (upstream's VoxtralTTSAudioTokenizer._forward_decoder): a frame's 37 codes as 292
//! values - the semantic code's row of the VQ codebook [8,192 x 256] and the 36 FSQ levels mapped to [-1, 1] - then a
//! causal convolution (292 -> 1,024, k 3, its 2 values ahead repeated), four stages of two transformer layers (1,024
//! wide, 8 heads of 128, RMS ε 0.01, q / k RMS norms over the whole row ε 1e-6, causal ALiBi attention over windows of
//! 2, 4, 8 and 16 keys back, SwiGLU 4,096, layer scales) with a transposed convolution (k 4, stride 2, 2 trimmed from
//! the end) after the first three, and a causal convolution (1,024 -> 240, k 7, its 6 values ahead mirrored): 240
//! samples a frame at 24 kHz. As upstream's decode_helper_batch_async, a run is decoded in chunks of 375 frames, each
//! on its own.

use nextsycl_audio::{Error, Result};
use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::flow::{ACOUSTIC, LEVELS, SPECIAL};
use crate::ops::{fp, Mat, Ops};
use crate::weights::Weights;

/// upstream's decode chunk (frames)
pub const CHUNK: usize = 375;
const HEAD: usize = 128;

struct Layer {
    attn_norm: DevBuf,
    ffn_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    attn_scale: DevBuf,
    ffn_scale: DevBuf,
    qkv: Mat,
    o: Mat,
    gu: Mat,
    down: Mat,
}

pub struct Codec {
    /// the semantic codebook [codes, sdim] (host)
    cb: Vec<f32>,
    pub codes: usize,
    sdim: usize,
    dim: usize,
    heads: usize,
    ffn: usize,
    eps: f32,
    qk_eps: f32,
    conv0: DevBuf,
    k0: usize,
    /// the transposed convolutions [dim, dim, k] and their strides
    ups: Vec<(DevBuf, usize, usize)>,
    stages: Vec<Vec<Layer>>,
    windows: Vec<usize>,
    out: DevBuf,
    kout: usize,
    /// samples a frame
    pub patch: usize,
}

fn ints(s: &str) -> Result<Vec<usize>> {
    s.split(',').map(|x| x.trim().parse().map_err(|_| Error(format!("{s:?}: a list of whole numbers expected")))).collect()
}

impl Codec {
    pub fn load(ops: &Ops, f: &Weights) -> Result<Codec> {
        let g = &ops.gpu;
        let strides = ints(&f.text("codec.conv_strides").unwrap_or_else(|| "1,2,2,2".into()))?;
        let kernels = ints(&f.text("codec.conv_kernels").unwrap_or_else(|| "3,4,4,4".into()))?;
        let lengths = ints(&f.text("codec.tfm_lengths").unwrap_or_else(|| "2,2,2,2".into()))?;
        let (dim, heads, sdim, patch) = (f.num("codec.dim")? as usize, f.num("codec.n_heads")? as usize, f.num("codec.semantic_dim")? as usize,
                                         f.num("codec.patch_size")? as usize);
        if strides.len() != 4 || strides[0] != 1 || kernels.len() != 4 || lengths.len() != 4 || f.num("codec.head_dim")? as usize != HEAD
            || f.num("codec.acoustic_dim")? as usize != ACOUSTIC || f.num("codec.n_kv_heads")? as usize != heads {
            return Err(Error(format!("a codec of strides {strides:?}, kernels {kernels:?}, stages {lengths:?}: not the one this engine decodes \
                                      (1,2,2,2 / 3,4,4,4 / 2,2,2,2; heads of {HEAD}, {ACOUSTIC} acoustic values)")));
        }
        let codes = f.num("semantic_codebook_size")? as usize;
        let cb = f.all("codec.semantic_cb.weight")?;
        if cb.len() != codes * sdim {
            return Err(Error(format!("codec.semantic_cb: {} values, [{codes}, {sdim}] expected", cb.len())));
        }
        let conv = |name: &str| -> Result<(DevBuf, Vec<u64>)> { Ok((DevBuf::from_f32(g, &f.all(name)?)?, f.tensor(name)?.shape.clone())) };
        let (conv0, s0) = conv("codec.dec.conv.0.weight")?;
        if s0 != [dim as u64, (sdim + ACOUSTIC) as u64, kernels[0] as u64] {
            return Err(Error(format!("codec.dec.conv.0: {s0:?}")));
        }
        let mut ups = Vec::new();
        for i in 1..4 {
            let (w, s) = conv(&format!("codec.dec.conv.{i}.weight"))?;
            if s != [dim as u64, dim as u64, kernels[i] as u64] {
                return Err(Error(format!("codec.dec.conv.{i}: {s:?}")));
            }
            ups.push((w, kernels[i], strides[i]));
        }
        let mut stages = Vec::new();
        for (s, n) in lengths.iter().enumerate() {
            let mut layers = Vec::new();
            for l in 0..*n {
                let p = |x: &str| format!("codec.dec.tfm.{s}.blk.{l}.{x}");
                layers.push(Layer {
                    attn_norm: f.vec(ops, &p("attn_norm.weight"))?,
                    ffn_norm: f.vec(ops, &p("ffn_norm.weight"))?,
                    q_norm: f.vec(ops, &p("q_norm.weight"))?,
                    k_norm: f.vec(ops, &p("k_norm.weight"))?,
                    attn_scale: f.vec(ops, &p("attn_scale"))?,
                    ffn_scale: f.vec(ops, &p("ffn_scale"))?,
                    qkv: f.mat(ops, &[&p("attn_q.weight"), &p("attn_k.weight"), &p("attn_v.weight")])?,
                    o: f.mat(ops, &[&p("attn_o.weight")])?,
                    gu: f.mat(ops, &[&p("ffn_gate.weight"), &p("ffn_up.weight")])?,
                    down: f.mat(ops, &[&p("ffn_down.weight")])?,
                });
            }
            stages.push(layers);
        }
        let ffn = stages[0][0].gu.n / 2;
        let (out, so) = conv("codec.output.weight")?;
        if so[0] as usize != patch || so[1] as usize != dim {
            return Err(Error(format!("codec.output: {so:?}")));
        }
        // the windows (upstream: attn_sliding_window_size 16 halved by each of the encoder's three downsamplings, then
        // doubled by each upsampling here)
        let w0 = f.num("codec.attn_window")? as usize;
        let up = strides[1..].iter().product::<usize>();
        let windows: Vec<usize> = (0..4).map(|s| w0 / up * strides[1..=s].iter().product::<usize>()).collect();
        Ok(Codec { cb, codes, sdim, dim, heads, ffn, eps: f.num("codec.norm_eps")? as f32, qk_eps: f.num("codec.qk_norm_eps")? as f32, conv0, k0: kernels[0],
                   ups, stages, windows, out, kout: so[2] as usize, patch })
    }

    /// A chunk's 292 values a frame, channels first [292, T], from codes as the model emits them (the specials' 2 on
    /// each)
    pub fn embed(&self, frames: &[Vec<i32>]) -> Vec<f32> {
        let t = frames.len();
        let mut e = vec![0f32; (self.sdim + ACOUSTIC) * t];
        for (i, f) in frames.iter().enumerate() {
            let s = (f[0] - SPECIAL as i32).clamp(0, self.codes as i32 - 1) as usize;
            for d in 0..self.sdim {
                e[d * t + i] = self.cb[s * self.sdim + d];
            }
            for a in 0..ACOUSTIC {
                let c = (f[1 + a] - SPECIAL as i32).clamp(0, LEVELS as i32 - 1);
                e[(self.sdim + a) * t + i] = c as f32 * 2.0 / (LEVELS - 1) as f32 - 1.0;
            }
        }
        e
    }

    /// The speech of frames [T][37] (24 kHz), chunk by chunk; `progress` after each
    pub fn decode(&self, ops: &Ops, nsd: &Nsd, frames: &[Vec<i32>], progress: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        if frames.iter().any(|f| f.len() != 1 + ACOUSTIC) {
            return Err(Error(format!("frames of {} codes expected", 1 + ACOUSTIC)));
        }
        let g = &ops.gpu;
        let longest = frames.len().clamp(1, CHUNK);
        let up = self.ups.iter().map(|u| u.2).product::<usize>();
        let rows = longest * up;
        let (d, c0) = (self.dim, self.sdim + ACOUSTIC);
        let widest = (3 * d).max(2 * self.ffn).max(c0 * (longest + self.k0)).max(self.patch * rows).max(d * (rows + self.kout));
        let a = DevBuf::f32(g, widest)?;
        let b = DevBuf::f32(g, widest)?;
        let x = DevBuf::f32(g, rows * d)?;
        let w = Work {
            hb: DevBuf::f32(g, rows * d)?,
            qkv: DevBuf::f32(g, rows * 3 * d)?,
            att: DevBuf::f32(g, rows * d)?,
            gu: DevBuf::f32(g, rows * 2 * self.ffn)?,
            act: DevBuf::f32(g, rows * self.ffn)?,
            tmp: DevBuf::f32(g, rows * d)?,
            xh: DevBuf::new(g, rows * (3 * d).max(2 * self.ffn).max(d) * 2)?,
        };
        let mut out = Vec::with_capacity(frames.len() * self.patch);
        let chunks = frames.len().div_ceil(CHUNK);
        for (ci, chunk) in frames.chunks(CHUNK).enumerate() {
            let t = chunk.len();
            a.write(0, &crate::bytes(&self.embed(chunk)))?;
            // the first convolution's k - 1 values ahead: the first frame repeated
            ops.pad_left(a.fp(), c0, t, self.k0 - 1, false, b.fp())?;
            ops.conv(b.fp(), c0, t + self.k0 - 1, &self.conv0, d, self.k0, a.fp())?;
            ops.transpose(a.fp(), d, t, x.fp())?;
            let mut l = t;
            for (s, layers) in self.stages.iter().enumerate() {
                for layer in layers {
                    self.layer(ops, nsd, layer, &x, &w, l, self.windows[s])?;
                }
                if let Some((wt, k, stride)) = self.ups.get(s) {
                    ops.transpose(x.fp(), l, d, a.fp())?;
                    ops.conv_up(a.fp(), d, l, wt, d, *k, *stride, b.fp())?;
                    l *= stride;
                    ops.transpose(b.fp(), d, l, x.fp())?;
                }
            }
            // the output convolution's 6 values ahead mirrored, then [240, L] -> samples (frame-major)
            ops.transpose(x.fp(), l, d, a.fp())?;
            ops.pad_left(a.fp(), d, l, self.kout - 1, true, b.fp())?;
            ops.conv(b.fp(), d, l + self.kout - 1, &self.out, self.patch, self.kout, a.fp())?;
            ops.transpose(a.fp(), self.patch, l, b.fp())?;
            g.sync()?;
            let mut v = vec![0u8; l * self.patch * 4];
            b.read(0, &mut v)?;
            out.extend(v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
            progress(ci + 1, chunks)?;
        }
        Ok(out)
    }

    /// One transformer layer over rows x [l, dim], in place
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ops: &Ops, nsd: &Nsd, ly: &Layer, x: &DevBuf, w: &Work, l: usize, window: usize) -> Result<()> {
        let d = self.dim;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, d, ly.attn_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
        ly.qkv.apply(ops, nsd, w.hb.fp(), l, &w.xh, w.qkv.fp())?;
        ops.rms_rows(w.qkv.fp(), 3 * d, l, d, &ly.q_norm, self.qk_eps)?;
        ops.rms_rows(fp(&w.qkv, d), 3 * d, l, d, &ly.k_norm, self.qk_eps)?;
        ops.alibi_attn(w.qkv.fp(), fp(&w.qkv, d), fp(&w.qkv, 2 * d), l, self.heads, HEAD, 3 * d, 3 * d, window, w.att.fp())?;
        ly.o.apply(ops, nsd, w.att.fp(), l, &w.xh, w.tmp.fp())?;
        ops.scale_add(x.fp(), w.tmp.fp(), &ly.attn_scale, l, d)?;
        nsd.rms_norm_mod(x.ptr(), Dt::F32, l, d, ly.ffn_norm.ptr(), self.eps, none(), none(), none(), w.hb.ptr(), Dt::F32)?;
        ly.gu.apply(ops, nsd, w.hb.fp(), l, &w.xh, w.gu.fp())?;
        nsd.swiglu(w.gu.ptr(), Dt::F32, l, self.ffn, w.act.ptr(), Dt::F32)?;
        ly.down.apply(ops, nsd, w.act.fp(), l, &w.xh, w.tmp.fp())?;
        ops.scale_add(x.fp(), w.tmp.fp(), &ly.ffn_scale, l, d)
    }
}

struct Work {
    hb: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    tmp: DevBuf,
    xh: DevBuf,
}
