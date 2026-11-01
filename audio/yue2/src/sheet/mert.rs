//! SheetSage2's encoder on the card: MERT-v2-FullSong (modeling_mert2.py) with SheetSage2's attention adapters merged
//! (the file's attention matrices are the merged ones: no LoRA at run time), the softmax mix of its 25 states and the
//! projection to the decoder's width (modeling_sheetsage2.py `get_audio_features`).
//!
//! ```text
//!   mel [30000, 128] (mel.rs, 100 a second)
//!   subsampling: three ConvNeXt stages - 128 (3 layers), LN + conv k2 s2 -> 512 (4), LN + conv k2 s2 -> 1024 (5): 25 a second
//!     a layer: x += down(GRN(gelu(up(LN(dwconv7(x))))))       GRN: x n gamma + beta + x, n the L2 norm along time / its channel mean
//!   24 conformer blocks of 1024 (16 heads of 64, rotate-half RoPE base 10000 from position 0 of the window, no mask):
//!     x += ffn1(LN x) / 2;  x += attn(LN x);  x += pw2(gelu(LN(dwconv31(glu(pw1(LN x))))));  x += ffn2(LN x) / 2;  x = LN x
//!   mixed = sum softmax(layer_w)[i] state_i  (state 0 the subsampler's output, i the blocks')   memory = proj(mixed) [7500, 512]
//! ```
//!
//! The window is always the full 300 s (a shorter recording zero-padded on the right: upstream encodes the silence too,
//! and its GRN and attention see it), so the encoder runs on 7500 frames every time. The weights are the file's half
//! matrices as stored (converted from bf16: the same values the reference's bf16 autocast multiplies); the residual
//! stream, the norms and the mix in float32, the products' inputs half, accumulated in float32.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

use crate::ffi::{self, Api};
use crate::lm::{at, host_vec, Scratch, M, P};
use crate::sheet::mel::{Frontend, MELS};

pub const WIDTH: usize = 1024;
pub const BLOCKS: usize = 24;
pub const HEADS: usize = 16;
pub const HEAD: usize = 64;
pub const FFN: usize = 4096;
pub const OUT: usize = 512;
/// frames of a window: 300 s at 25 a second
pub const FRAMES: usize = 7500;
const STAGES: [(usize, usize, usize); 3] = [(128, 128, 3), (128, 512, 4), (512, 1024, 5)];
const SUB_EPS: f32 = 1e-6;
const EPS: f32 = 1e-5;

/// A matrix as stored (half [n, k]) and its bias
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// A layer norm's weight and bias
pub struct Norm {
    pub w: DevBuf,
    pub b: DevBuf,
}

struct Cnx {
    dw: DevBuf,
    dw_b: DevBuf,
    norm: Norm,
    up: Lin,
    grn_w: DevBuf,
    grn_b: DevBuf,
    down: Lin,
}

struct Stage {
    /// the resampling: its norm and the k2 s2 convolution as a product over two frames ([cout, 2 cin], half)
    resample: Option<(Norm, Lin)>,
    c: usize,
    layers: Vec<Cnx>,
}

struct Ffn {
    norm: Norm,
    up: Lin,
    down: Lin,
}

struct Block {
    ffn1: Ffn,
    attn_norm: Norm,
    qkv: Lin,
    o: Lin,
    conv_norm: Norm,
    pw1: Lin,
    dw: DevBuf,
    dw_norm: Norm,
    pw2: Lin,
    ffn2: Ffn,
    final_norm: Norm,
}

pub struct Mert {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static Api,
    pub front: Frontend,
    stages: Vec<Stage>,
    blocks: Vec<Block>,
    /// softmax(layer_w): the 25 states' weights
    pub mix: Vec<f32>,
    proj: Lin,
    /// RoPE's (cos, sin) for the window's frames, float32 [7500, 32, 2]
    cs: DevBuf,
    pub bytes: u64,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// float32 rounded to bfloat16 (to nearest even), as float32
pub fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return x;
    }
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// float32 values as half bits (round to nearest even)
pub fn to_half(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| half_bits(*x).to_le_bytes()).collect()
}

fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// The file's tensors on the card, counted
pub struct Up<'a> {
    pub f: &'a Gguf,
    pub gpu: &'a Arc<Gpu>,
    pub bytes: u64,
}

impl Up<'_> {
    pub fn raw(&mut self, b: &[u8]) -> Result<DevBuf> {
        let d = DevBuf::new(self.gpu, b.len())?;
        d.write(0, b)?;
        self.bytes += b.len() as u64;
        Ok(d)
    }

    pub fn vec(&mut self, name: &str, n: usize) -> Result<DevBuf> {
        let v = host_vec(self.f, name)?;
        if v.len() != n {
            return Err(Error(format!("{name}: {} values, {n} expected", v.len())));
        }
        self.raw(crate::f32_bytes(&v))
    }

    pub fn norm(&mut self, p: &str, n: usize) -> Result<Norm> {
        Ok(Norm { w: self.vec(&format!("{p}.weight"), n)?, b: self.vec(&format!("{p}.bias"), n)? })
    }

    /// A half matrix [n, k] as the file stores it (F16), with its bias when it has one
    pub fn lin(&mut self, p: &str, n: usize, k: usize, bias: bool) -> Result<Lin> {
        let name = format!("{p}.weight");
        let t = self.f.tensor(&name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.f.paths[0].display())))?;
        if t.ty != GType::F16 || t.elements() != (n * k) as u64 || t.shape.last().copied() != Some(k as u64) {
            return Err(Error(format!("{name}: {} {:?}, half [{n}, {k}] expected", t.ty.name(), t.shape)));
        }
        let w = self.raw(&self.f.read(t).map_err(ge)?)?;
        let b = if bias { Some(self.vec(&format!("{p}.bias"), n)?) } else { None };
        Ok(Lin { w, b, n, k })
    }
}


impl Mert {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>) -> Result<Mert> {
        let mut u = Up { f, gpu, bytes: 0 };
        let front = Frontend::new(&host_vec(f, "enc.mel.stft_window")?, &host_vec(f, "enc.mel.filterbank")?, &host_vec(f, "enc.mel.mean")?, &host_vec(f, "enc.mel.std")?)
            .map_err(Error)?;
        let mut stages = Vec::new();
        for (si, (cin, c, depth)) in STAGES.iter().copied().enumerate() {
            let p = format!("enc.sub.{si}");
            let resample = if si == 0 {
                None
            } else {
                // torch's [cout, cin, 2] as the product over a pair of frames: W'[o, j cin + c] = w[o, c, j]
                let w = host_vec(f, &format!("{p}.down_conv.weight"))?;
                if w.len() != c * cin * 2 {
                    return Err(Error(format!("{p}.down_conv.weight: {} values, [{c}, {cin}, 2] expected", w.len())));
                }
                let mut wp = vec![0f32; c * 2 * cin];
                for o in 0..c {
                    for ci in 0..cin {
                        for j in 0..2 {
                            wp[o * 2 * cin + j * cin + ci] = w[o * cin * 2 + ci * 2 + j];
                        }
                    }
                }
                let lin = Lin { w: u.raw(&to_half(&wp))?, b: Some(u.vec(&format!("{p}.down_conv.bias"), c)?), n: c, k: 2 * cin };
                Some((u.norm(&format!("{p}.down_norm"), cin)?, lin))
            };
            let mut layers = Vec::new();
            for li in 0..depth {
                let q = format!("{p}.cnx.{li}");
                layers.push(Cnx { dw: u.vec(&format!("{q}.dw.weight"), c * 7)?, dw_b: u.vec(&format!("{q}.dw.bias"), c)?, norm: u.norm(&format!("{q}.norm"), c)?,
                                  up: u.lin(&format!("{q}.pw_up"), 4 * c, c, true)?, grn_w: u.vec(&format!("{q}.grn.weight"), 4 * c)?,
                                  grn_b: u.vec(&format!("{q}.grn.bias"), 4 * c)?, down: u.lin(&format!("{q}.pw_down"), c, 4 * c, true)? });
            }
            stages.push(Stage { resample, c, layers });
        }
        let mut blocks = Vec::new();
        for bi in 0..BLOCKS {
            let p = format!("enc.blk.{bi}");
            let ffn = |u: &mut Up, t: &str| -> Result<Ffn> {
                Ok(Ffn { norm: u.norm(&format!("{p}.{t}_norm"), WIDTH)?, up: u.lin(&format!("{p}.{t}_up"), FFN, WIDTH, true)?,
                         down: u.lin(&format!("{p}.{t}_down"), WIDTH, FFN, true)? })
            };
            // q | k | v stacked: one product
            let mut w = Vec::new();
            let mut b = Vec::new();
            for x in ["attn_q", "attn_k", "attn_v"] {
                let t = f.tensor(&format!("{p}.{x}.weight")).ok_or_else(|| Error(format!("no {p}.{x}.weight")))?;
                if t.ty != GType::F16 || t.elements() != (WIDTH * WIDTH) as u64 {
                    return Err(Error(format!("{p}.{x}.weight: half [1024, 1024] expected")));
                }
                w.extend(f.read(t).map_err(ge)?);
                b.extend(host_vec(f, &format!("{p}.{x}.bias"))?);
            }
            let qkv = Lin { w: u.raw(&w)?, b: Some(u.raw(crate::f32_bytes(&b))?), n: 3 * WIDTH, k: WIDTH };
            blocks.push(Block { ffn1: ffn(&mut u, "ffn1")?, attn_norm: u.norm(&format!("{p}.attn_norm"), WIDTH)?, qkv, o: u.lin(&format!("{p}.attn_output"), WIDTH, WIDTH, true)?,
                                conv_norm: u.norm(&format!("{p}.conv_norm"), WIDTH)?, pw1: u.lin(&format!("{p}.conv_pw1"), 2 * WIDTH, WIDTH, false)?,
                                dw: u.vec(&format!("{p}.conv_dw.weight"), WIDTH * 31)?, dw_norm: u.norm(&format!("{p}.conv_dw_norm"), WIDTH)?,
                                pw2: u.lin(&format!("{p}.conv_pw2"), WIDTH, WIDTH, false)?, ffn2: ffn(&mut u, "ffn2")?,
                                final_norm: u.norm(&format!("{p}.final_norm"), WIDTH)? });
        }
        let lw = host_vec(f, "enc.layer_w")?;
        if lw.len() != BLOCKS + 1 {
            return Err(Error(format!("enc.layer_w: {} values, 25 expected", lw.len())));
        }
        // softmax in float32, as torch.softmax of the float32 parameter
        let mx = lw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = lw.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        let mix = e.iter().map(|v| v / s).collect();
        let proj = u.lin("enc.proj", OUT, WIDTH, true)?;
        // RoPE as upstream's transcription runs it. RotaryEmbedding makes its (cos, sin) table once and caches it, and
        // the first call is under transcribe's bf16 autocast: torch.einsum (an autocast-to-bf16 op) multiplies the
        // positions and inv_freq rounded to bf16, its product rounded to bf16, and the cosines and sines of those bf16
        // angles are bf16 too. A position past 256 is then a multiple of 2 or more (7499 is 7488), the angles of the
        // fast frequencies are far from position * inv_freq - and the model's whole transcription (every reference
        // token) is made with that table; with exact float32 angles the memory is another (cosine 0.934 to upstream's,
        // measured on the test recording). So the table here is upstream's, bf16 step by bf16 step.
        let inv: Vec<f32> = (0..HEAD / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / HEAD as f32)).collect();
        // NS_YUE2_SS2_ANGLES=fp32: float32 angles, for a check against a reference run without bf16 autocast
        let exact_angles = std::env::var("NS_YUE2_SS2_ANGLES").is_ok_and(|v| v.trim() == "fp32");
        let mut cs = Vec::with_capacity(FRAMES * HEAD);
        for t in 0..FRAMES {
            for f in &inv {
                if exact_angles {
                    let a = t as f32 * f;
                    cs.push(a.cos());
                    cs.push(a.sin());
                    continue;
                }
                let a = bf16(bf16(t as f32) * bf16(*f));
                cs.push(bf16((a as f64).cos() as f32));
                cs.push(bf16((a as f64).sin() as f32));
            }
        }
        let cs = u.raw(crate::f32_bytes(&cs))?;
        let bytes = u.bytes;
        Ok(Mert { nsd: Nsd::new(gpu)?, k: ffi::api()?, gpu: gpu.clone(), front, stages, blocks, mix, proj, cs, bytes })
    }

    fn g(&self) -> M {
        self.gpu.raw()
    }

    fn ln(&self, x: P, m: usize, c: usize, n: &Norm, eps: f32, out: M, dt: Dt) -> Result<()> {
        self.nsd.layer_norm(x, Dt::F32, m, c, n.w.ptr(), n.b.ptr(), eps, out, dt)
    }

    fn lin(&self, x: P, m: usize, l: &Lin, out: M, dt: Dt) -> Result<()> {
        self.nsd.linear(x, Dt::F16, m, l.k, l.w.ptr(), l.n, l.b.as_ref().map_or(none(), |b| b.ptr().cast_const()), out, dt)
    }

    /// float32 -> half (n values)
    fn half(&self, x: P, n: usize, out: M) -> Result<()> {
        // SAFETY: x holds n floats, out n halves.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x.cast(), n as i64, out) }, "half")
    }

    fn axpy(&self, x: M, y: P, n: usize, s: f32) -> Result<()> {
        // SAFETY: x and y hold n floats.
        ffi::check(unsafe { (self.k.axpy)(self.g(), x.cast(), x.cast_const().cast(), y.cast(), n as i64, s) }, "axpy")
    }

    fn dwconv(&self, x: P, t: usize, c: usize, w: &DevBuf, k: usize, b: Option<&DevBuf>, out: M) -> Result<()> {
        // SAFETY: x and out [t, c] floats, w [c, k], b [c].
        ffi::check(unsafe { (self.k.dwconv)(self.g(), x.cast(), t as i64, c as i64, w.fp(), k as i64, b.map_or(std::ptr::null(), |b| b.fp()), out.cast()) }, "dwconv")
    }

    /// The features of one window (mel [frames, 128] for 300 s: 30000 frames) to the decoder's memory, float32 [7500,
    /// 512] on the card; `tap` sees the states on the way (its name: "sub", "block{i}", "mixed"), float32 [7500, width]
    pub fn encode(&self, s: &mut Scratch, mel: &[f32], tap: &mut dyn FnMut(&str, &DevBuf, usize) -> Result<()>) -> Result<DevBuf> {
        let g = &self.gpu;
        let mut t = mel.len() / MELS;
        if t != 4 * FRAMES {
            return Err(Error(format!("{t} mel frames: a window is {} (300 s)", 4 * FRAMES)));
        }
        // the subsampler: x float32 [t, c]
        let mut x = DevBuf::from_f32(g, mel)?;
        for st in &self.stages {
            if let Some((n, conv)) = &st.resample {
                let cin = conv.k / 2;
                let h = s.get(g, "lnh", t * cin * 2)?;
                self.ln(x.ptr(), t, cin, n, SUB_EPS, h, Dt::F16)?;
                t /= 2;
                let y = DevBuf::f32(g, t * st.c)?;
                self.lin(h, t, conv, y.ptr(), Dt::F32)?;
                x = y;
            }
            let c = st.c;
            for l in &st.layers {
                let d = s.get(g, "dw", t * c * 4)?;
                self.dwconv(x.ptr(), t, c, &l.dw, 7, Some(&l.dw_b), d)?;
                let h = s.get(g, "lnh", t * c * 2)?;
                self.ln(d, t, c, &l.norm, SUB_EPS, h, Dt::F16)?;
                let up = s.get(g, "up", t * 4 * c * 4)?;
                self.lin(h, t, &l.up, up, Dt::F32)?;
                self.nsd.gelu(up, Dt::F32, t * 4 * c, false)?;
                let uh = s.get(g, "uh", t * 4 * c * 2)?;
                let part = s.get(g, "grn", (4 * c + 1) * 4)?;
                // SAFETY: up [t, 4c] floats, the weights [4c], uh [t, 4c] half, part 4c + 1 floats.
                ffi::check(unsafe { (self.k.grn)(self.g(), up.cast(), t as i64, (4 * c) as i64, l.grn_w.fp(), l.grn_b.fp(), uh, part.cast()) }, "grn")?;
                let o = s.get(g, "o", t * c * 4)?;
                self.lin(uh, t, &l.down, o, Dt::F32)?;
                self.axpy(x.ptr(), o, t * c, 1.0)?;
            }
        }
        tap("sub", &x, WIDTH)?;
        let n = t * WIDTH;
        let mixed = DevBuf::f32(g, n)?;
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.axpy)(self.g(), mixed.fp(), x.fp(), x.fp(), n as i64, self.mix[0] - 1.0) }, "mix")?;
        let hh = s.get(g, "hh", t * FFN * 2)?;
        let up = s.get(g, "up", t * FFN * 4)?;
        let o = s.get(g, "o", n * 4)?;
        let qkv = s.get(g, "qkv", t * 3 * WIDTH * 2)?;
        let att = s.get(g, "att", n * 2)?;
        let glu = s.get(g, "glu", n * 4)?;
        let d = s.get(g, "dw", n * 4)?;
        for (bi, b) in self.blocks.iter().enumerate() {
            let ffn = |f: &Ffn| -> Result<()> {
                self.ln(x.ptr(), t, WIDTH, &f.norm, EPS, hh, Dt::F16)?;
                self.lin(hh, t, &f.up, up, Dt::F32)?;
                self.nsd.gelu(up, Dt::F32, t * FFN, false)?;
                self.half(up, t * FFN, hh)?;
                self.lin(hh, t, &f.down, o, Dt::F32)?;
                self.axpy(x.ptr(), o, n, 0.5)
            };
            ffn(&b.ffn1)?;
            // attention: q | k | v half [t, 3072], RoPE on q and k in place
            self.ln(x.ptr(), t, WIDTH, &b.attn_norm, EPS, hh, Dt::F16)?;
            self.lin(hh, t, &b.qkv, qkv, Dt::F16)?;
            self.nsd.rope(qkv, Dt::F16, t, HEADS, HEAD, 3 * WIDTH, self.cs.ptr(), HEAD)?;
            self.nsd.rope(at(qkv, WIDTH * 2), Dt::F16, t, HEADS, HEAD, 3 * WIDTH, self.cs.ptr(), HEAD)?;
            self.nsd.attention(qkv, at(qkv, WIDTH * 2), at(qkv, 2 * WIDTH * 2), Dt::F16, t, HEADS, HEAD, 3 * WIDTH, att, Dt::F16)?;
            self.lin(att, t, &b.o, o, Dt::F32)?;
            self.axpy(x.ptr(), o, n, 1.0)?;
            // the convolution module
            self.ln(x.ptr(), t, WIDTH, &b.conv_norm, EPS, hh, Dt::F16)?;
            self.lin(hh, t, &b.pw1, up, Dt::F32)?;
            // SAFETY: up [t, 2048] floats, glu [t, 1024].
            ffi::check(unsafe { (self.k.glu)(self.g(), up.cast(), t as i64, WIDTH as i64, glu.cast()) }, "glu")?;
            self.dwconv(glu, t, WIDTH, &b.dw, 31, None, d)?;
            self.ln(d, t, WIDTH, &b.dw_norm, EPS, d, Dt::F32)?;
            self.nsd.gelu(d, Dt::F32, n, false)?;
            self.half(d, n, hh)?;
            self.lin(hh, t, &b.pw2, o, Dt::F32)?;
            self.axpy(x.ptr(), o, n, 1.0)?;
            ffn(&b.ffn2)?;
            self.ln(x.ptr(), t, WIDTH, &b.final_norm, EPS, x.ptr(), Dt::F32)?;
            self.axpy(mixed.ptr(), x.ptr(), n, self.mix[bi + 1])?;
            tap(&format!("block{bi}"), &x, WIDTH)?;
        }
        tap("mixed", &mixed, WIDTH)?;
        self.half(mixed.ptr(), n, hh)?;
        let memory = DevBuf::f32(g, t * OUT)?;
        self.lin(hh, t, &self.proj, memory.ptr(), Dt::F32)?;
        Ok(memory)
    }
}
