//! The engine's kernels (ffi.rs and nsd's) as checked calls on device pointers, its matrices (float32 on the card,
//! multiplied by oneMKL through nsd_linear) and the GGUF tensors they come from.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::path::Path;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf, Tensor, Value};

use crate::ffi;

type F = *const f32;
type W = *mut f32;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// A null float pointer (an absent weight or bias)
pub fn nulf() -> F {
    std::ptr::null()
}

pub fn e(x: nextsycl_core::Error) -> Error {
    Error(x.0)
}

/// The engine's kernels on one GPU
pub struct Ops {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static ffi::Api,
}

macro_rules! call {
    ($self:ident, $f:ident, $what:literal $(, $a:expr)*) => {
        // SAFETY: the caller's buffers hold the sizes named (see z3.h).
        ffi::check(unsafe { ($self.k.$f)($self.gpu.raw() $(, $a)*) }, $what).map_err(e)
    };
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), nsd: Nsd::new(gpu).map_err(e)?, k: ffi::api().map_err(e)? })
    }

    pub fn f32(&self, n: usize) -> Result<DevBuf> {
        DevBuf::f32(&self.gpu, n.max(1)).map_err(e)
    }

    pub fn from_f32(&self, v: &[f32]) -> Result<DevBuf> {
        if v.is_empty() {
            return self.f32(1);
        }
        DevBuf::from_f32(&self.gpu, v).map_err(e)
    }

    pub fn from_i32(&self, v: &[i32]) -> Result<DevBuf> {
        let b = DevBuf::new(&self.gpu, (v.len() * 4).max(4)).map_err(e)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if !bytes.is_empty() {
            b.write(0, &bytes).map_err(e)?;
        }
        Ok(b)
    }

    pub fn sync(&self) -> Result<()> {
        self.gpu.sync().map_err(e)
    }

    /// out [m, n] = x [m, k] . W^T (+ bias [n])
    pub fn linear(&self, x: F, m: usize, w: &Mat, bias: F, out: W) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        self.nsd.linear(x.cast(), Dt::F32, m, w.k, w.w.ptr(), w.n, bias.cast(), out.cast(), Dt::F32).map_err(e)
    }

    pub fn layer_norm(&self, x: F, m: usize, c: usize, w: F, b: F, eps: f32, out: W) -> Result<()> {
        self.nsd.layer_norm(x.cast(), Dt::F32, m, c, w.cast(), b.cast(), eps, out.cast(), Dt::F32).map_err(e)
    }

    /// GELU in place: the erf form, or the tanh one
    pub fn gelu(&self, x: W, n: usize, tanh: bool) -> Result<()> {
        self.nsd.gelu(x.cast(), Dt::F32, n, tanh).map_err(e)
    }

    pub fn embed(&self, audio: &DevBuf, ncb: usize, va: usize, text: &DevBuf, vt: usize, ids: &DevBuf, rows: usize, d: usize, out: W) -> Result<()> {
        call!(self, embed, "embeddings", audio.ptr(), ncb as i64, va as i64, text.ptr(), vt as i64, ids.ptr().cast(), rows as i64, d as i64, out)
    }

    pub fn rms(&self, x: F, xs: usize, rows: usize, c: usize, w: F, eps: f32, out: W) -> Result<()> {
        call!(self, rms, "rms norm", x, xs as i64, rows as i64, c as i64, w, eps, out)
    }

    pub fn attnres_score(&self, src: F, rows: usize, d: usize, q: F, nq: usize, eps: f32, scores: W) -> Result<()> {
        call!(self, attnres_score, "attnres scores", src, rows as i64, d as i64, q, nq as i64, eps, scores)
    }

    pub fn attnres_mix(&self, bank: F, scores: F, s: usize, cap: usize, rows: usize, d: usize, nq: usize, qi: usize, out: W) -> Result<()> {
        call!(self, attnres_mix, "attnres mix", bank, scores, s as i64, cap as i64, rows as i64, d as i64, nq as i64, qi as i64, out)
    }

    pub fn add_vec(&self, x: W, rows: usize, c: usize, v: F) -> Result<()> {
        call!(self, add_vec, "add", x, rows as i64, c as i64, v)
    }

    pub fn gdn_conv(&self, proj: F, rows: usize, p: usize, c: usize, w: F, k: usize, hist: W, out: W) -> Result<()> {
        call!(self, gdn_conv, "gdn conv", proj, rows as i64, p as i64, c as i64, w, k as i64, hist, out)
    }

    pub fn gdn_prep(&self, conv: F, proj: F, rows: usize, p: usize, ab0: usize, h: usize, s: usize, a_log: F, dt_bias: F, q: W, k: W, v: W, decay: W,
                    beta: W) -> Result<()> {
        call!(self, gdn_prep, "gdn prep", conv, proj, rows as i64, p as i64, ab0 as i64, h as i64, s as i64, a_log, dt_bias, q, k, v, decay, beta)
    }

    pub fn gdn_scan(&self, q: F, k: F, v: F, decay: F, beta: F, rows: usize, h: usize, s: usize, state: W, y: W) -> Result<()> {
        call!(self, gdn_scan, "gdn scan", q, k, v, decay, beta, rows as i64, h as i64, s as i64, state, y)
    }

    pub fn gdn_out(&self, y: F, proj: F, p: usize, z0: usize, rows: usize, h: usize, s: usize, w: F, eps: f32, out: W) -> Result<()> {
        call!(self, gdn_out, "gdn out", y, proj, p as i64, z0 as i64, rows as i64, h as i64, s as i64, w, eps, out)
    }

    pub fn head_rms(&self, x: W, xs: usize, rows: usize, h: usize, d: usize, scale: F, eps: f32) -> Result<()> {
        call!(self, head_rms, "head rms", x, xs as i64, rows as i64, h as i64, d as i64, scale, eps)
    }

    pub fn rope(&self, x: W, xs: usize, rows: usize, h: usize, d: usize, theta: f32, pos0: usize) -> Result<()> {
        call!(self, rope, "rope", x, xs as i64, rows as i64, h as i64, d as i64, theta, pos0 as i64)
    }

    pub fn attention(&self, q: F, qs: usize, rows: usize, k: F, v: F, kvs: usize, keys: usize, hq: usize, hk: usize, d: usize, scale: f32, pos0: usize,
                     causal: bool, window: usize, out: W) -> Result<()> {
        call!(self, attention, "attention", q, qs as i64, rows as i64, k, v, kvs as i64, keys as i64, hq as i64, hk as i64, d as i64, scale, pos0 as i64,
              causal as i32, window as i64, out)
    }

    pub fn head_gate(&self, x: W, gate: F, rows: usize, h: usize, d: usize) -> Result<()> {
        call!(self, head_gate, "head gate", x, gate, rows as i64, h as i64, d as i64)
    }

    pub fn softcap(&self, x: W, n: usize, cap: f32) -> Result<()> {
        call!(self, softcap, "softcap", x, n as i64, cap)
    }

    pub fn silu_mul(&self, a: F, b: F, out: W, n: usize) -> Result<()> {
        call!(self, silu_mul, "silu mul", a, b, out, n as i64)
    }

    pub fn topk(&self, logits: F, bias: F, rows: usize, ne: usize, k: usize, idx: &DevBuf, w: W) -> Result<()> {
        call!(self, topk, "router", logits, bias, rows as i64, ne as i64, k as i64, idx.ptr().cast(), w)
    }

    pub fn moe_up(&self, x: F, rows: usize, l: usize, gate: &DevBuf, up: &DevBuf, idx: &DevBuf, k: usize, f: usize, h: W) -> Result<()> {
        call!(self, moe_up, "experts up", x, rows as i64, l as i64, gate.ptr(), up.ptr(), idx.ptr().cast(), k as i64, f as i64, h)
    }

    pub fn moe_down(&self, h: F, rows: usize, k: usize, f: usize, down: &DevBuf, idx: &DevBuf, w: F, l: usize, out: W) -> Result<()> {
        call!(self, moe_down, "experts down", h, rows as i64, k as i64, f as i64, down.ptr(), idx.ptr().cast(), w, l as i64, out)
    }

    /// the edit LoRA's expert projections t [rows k, r] = down[e] . x (per_slot: x a (row, slot)'s, else its row's)
    pub fn moe_lora_t(&self, x: F, rows: usize, inn: usize, idx: &DevBuf, k: usize, per_slot: bool, down: &DevBuf, r: usize, t: W) -> Result<()> {
        call!(self, moe_lora_t, "experts lora", x, rows as i64, inn as i64, idx.ptr().cast(), k as i64, per_slot as i64, down.fp(), r as i64, t)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_up_lora(&self, x: F, rows: usize, l: usize, gate: &DevBuf, up: &DevBuf, idx: &DevBuf, k: usize, f: usize, tg: F, tu: F, upg: &DevBuf, upu: &DevBuf,
                       r: usize, h: W) -> Result<()> {
        call!(self, moe_up_lora, "experts up (lora)", x, rows as i64, l as i64, gate.ptr(), up.ptr(), idx.ptr().cast(), k as i64, f as i64, tg, tu, upg.fp(),
              upu.fp(), r as i64, h)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_lora(&self, h: F, rows: usize, k: usize, f: usize, down: &DevBuf, idx: &DevBuf, w: F, l: usize, td: F, upd: &DevBuf, r: usize, out: W)
                         -> Result<()> {
        call!(self, moe_down_lora, "experts down (lora)", h, rows as i64, k as i64, f as i64, down.ptr(), idx.ptr().cast(), w, l as i64, td, upd.fp(), r as i64,
              out)
    }

    pub fn lookup(&self, codes: &DevBuf, t: usize, ncol: usize, col: usize, table: F, c: usize, bias: F, acc: bool, out: W) -> Result<()> {
        call!(self, lookup, "lookup", codes.ptr().cast(), t as i64, ncol as i64, col as i64, table, c as i64, bias, acc as i32, out)
    }

    pub fn dwconv_causal(&self, x: F, t: usize, c: usize, w: F, k: usize, bias: F, out: W) -> Result<()> {
        call!(self, dwconv_causal, "causal depthwise conv", x, t as i64, c as i64, w, k as i64, bias, out)
    }

    pub fn dwconv_centred(&self, x: F, t: usize, c: usize, w: F, k: usize, bias: F, out: W) -> Result<()> {
        call!(self, dwconv_centred, "depthwise conv", x, t as i64, c as i64, w, k as i64, bias, out)
    }

    pub fn scale_add(&self, x: W, y: F, scale: F, r: usize, c: usize) -> Result<()> {
        call!(self, scale_add, "scaled add", x, y, scale, r as i64, c as i64)
    }

    pub fn patch_up(&self, x: F, t: usize, c: usize, h: usize, out: W) -> Result<()> {
        call!(self, patch_up, "patch up", x, t as i64, c as i64, h as i64, out)
    }

    pub fn spec(&self, y: F, f: usize, bins: usize, maxlog: f32, ri: W) -> Result<()> {
        call!(self, spec, "spectrum", y, f as i64, bins as i64, maxlog, ri)
    }

    pub fn ola(&self, frames: F, f: usize, n: usize, hop: usize, win: F, out: W) -> Result<()> {
        call!(self, ola, "overlap-add", frames, f as i64, n as i64, hop as i64, win, out)
    }

    pub fn mixer(&self, x0: F, x1: F, l: usize, w: F, k: usize, out: W) -> Result<()> {
        call!(self, mixer, "mixer", x0, x1, l as i64, w, k as i64, out)
    }

    pub fn resample(&self, x: F, l: usize, kern: F, orig: usize, nw: usize, width: usize, out: W, total: usize) -> Result<()> {
        call!(self, resample, "resample", x, l as i64, kern, orig as i64, nw as i64, width as i64, out, total as i64)
    }

    pub fn conv3x3(&self, x: F, fi: usize, ti: usize, ci: usize, w: F, co: usize, a: usize, sf: usize, st: usize, bias: F, out: W) -> Result<()> {
        call!(self, conv3x3, "conv3x3", x, fi as i64, ti as i64, ci as i64, w, co as i64, a as i64, sf as i64, st as i64, bias, out)
    }

    pub fn affine(&self, x: F, n: usize, c: usize, scale: F, shift: F, relu: bool, add: F, out: W) -> Result<()> {
        call!(self, affine, "affine", x, n as i64, c as i64, scale, shift, relu as i32, add, out)
    }

    pub fn subsample(&self, x: F, f: usize, t: usize, c: usize, sf: usize, st: usize, out: W) -> Result<()> {
        call!(self, subsample, "subsample", x, f as i64, t as i64, c as i64, sf as i64, st as i64, out)
    }

    pub fn swap01(&self, x: F, a: usize, b: usize, c: usize, out: W) -> Result<()> {
        call!(self, swap01, "swap", x, a as i64, b as i64, c as i64, out)
    }

    pub fn repeat_rows(&self, x: F, rep: usize, t: usize, c: usize, add: F, out: W) -> Result<()> {
        call!(self, repeat_rows, "repeat rows", x, rep as i64, t as i64, c as i64, add, out)
    }

    pub fn softmax(&self, x: W, n: usize, c: usize) -> Result<()> {
        call!(self, softmax, "softmax", x, n as i64, c as i64)
    }

    pub fn mix_experts(&self, x: F, r: F, t: usize, ne: usize, j: usize, out: W) -> Result<()> {
        call!(self, mix_experts, "mix experts", x, r, t as i64, ne as i64, j as i64, out)
    }

    pub fn fuse4(&self, x: [F; 4], w: F, t: usize, c: usize, out: W) -> Result<()> {
        call!(self, fuse4, "fuse", x[0], x[1], x[2], x[3], w, t as i64, c as i64, out)
    }

    pub fn astp(&self, x: F, att: F, t: usize, c: usize, floor: f32, out: W) -> Result<()> {
        call!(self, astp, "pooling", x, att, t as i64, c as i64, floor, out)
    }

    pub fn tanh(&self, x: W, n: usize) -> Result<()> {
        call!(self, tanh, "tanh", x, n as i64)
    }

    pub fn snake_cf(&self, x: F, c: usize, l: usize, alpha: F, inv_alpha: F, out: W) -> Result<()> {
        call!(self, snake_cf, "snake", x, c as i64, l as i64, alpha, inv_alpha, out)
    }

    pub fn add(&self, x: W, y: F, n: usize) -> Result<()> {
        call!(self, add, "add", x, y, n as i64)
    }

    pub fn patch_down(&self, x: F, t: usize, c: usize, h: usize, out: W) -> Result<()> {
        call!(self, patch_down, "patch down", x, t as i64, c as i64, h as i64, out)
    }

    pub fn glu(&self, x: F, t: usize, c: usize, out: W) -> Result<()> {
        call!(self, glu, "glu", x, t as i64, c as i64, out)
    }

    pub fn silu(&self, x: W, n: usize) -> Result<()> {
        call!(self, silu, "silu", x, n as i64)
    }

    pub fn attn_rel(&self, q: F, k: F, v: F, qs: usize, t: usize, h: usize, d: usize, dist: F, left: usize, right: usize, valid: F, scale: f32, bf: bool,
                    out: W) -> Result<()> {
        call!(self, attn_rel, "relative attention", q, k, v, qs as i64, t as i64, h as i64, d as i64, dist, left as i64, right as i64, valid, scale,
              bf as std::os::raw::c_int, out)
    }

    /// x rounded to bfloat16 values in place (the reference encoder's BF16 recipe)
    pub fn bf16(&self, x: W, n: usize) -> Result<()> {
        call!(self, bf16, "bf16 round", x, n as i64)
    }

    /// LACI's ACI mask on a decode row: `n` masked heads (`spec` on the card: head, lo, hi absolute key positions)
    /// recomputed over `keys` keys with the text keys [ts, te) outside their window blocked
    pub fn attn_window(&self, q: F, k: F, v: F, kvs: usize, keys: usize, hq: usize, hk: usize, hd: usize, scale: f32, spec: &DevBuf, n: usize, ts: usize,
                       te: usize, out: W) -> Result<()> {
        call!(self, attn_window, "windowed attention", q, k, v, kvs as i64, keys as i64, hq as i64, hk as i64, hd as i64, scale, spec.ptr().cast::<i32>() as *const i32,
              n as i64, ts as i64, te as i64, out)
    }

    /// full attention, heads of up to 128: q, k, v [t, h hd] rows qs apart
    pub fn attn_full(&self, q: F, k: F, v: F, qs: usize, t: usize, h: usize, hd: usize, scale: f32, out: W) -> Result<()> {
        call!(self, attn_full, "attention (full)", q, k, v, qs as i64, t as i64, h as i64, hd as i64, scale, out)
    }

    /// rotate-half RoPE from host tables cos / sin [rows, d] on x [rows, h d] (rows xs apart)
    pub fn rope_tab(&self, x: W, xs: usize, rows: usize, h: usize, d: usize, cs: F, sn: F) -> Result<()> {
        call!(self, rope_tab, "rope (tables)", x, xs as i64, rows as i64, h as i64, d as i64, cs, sn)
    }

    /// ConvTranspose1d's scatter: cols [t, k, c] -> y [to, c] (+ bias), stride s, padding p
    pub fn col2im(&self, cols: F, t: usize, k: usize, c: usize, s: usize, p: usize, bias: F, to: usize, y: W) -> Result<()> {
        call!(self, col2im, "col2im", cols, t as i64, k as i64, c as i64, s as i64, p as i64, bias, to as i64, y)
    }

    /// channels-first x [rows, cols] += bias[row]
    pub fn add_rows(&self, x: W, bias: F, rows: usize, cols: usize) -> Result<()> {
        call!(self, add_rows, "row bias", x, bias, rows as i64, cols as i64)
    }

    /// A 1-D convolution of channels-first x [ci, l] with w [co, ci, k] (nsd's: unfold + SGEMM), no padding: out [co, lo]
    pub fn conv1d(&self, x: F, ci: usize, l: usize, w: F, co: usize, k: usize, bias: F, stride: usize, dil: usize, out: W) -> Result<usize> {
        let lo = (l - dil * (k - 1) - 1) / stride + 1;
        self.nsd.conv1d(x.cast(), 1, ci, l, w.cast(), co, k, bias.cast(), stride, dil, 0, out.cast(), lo).map_err(e)?;
        Ok(lo)
    }
}

/// A matrix [n, k] on the card in float32
pub struct Mat {
    pub w: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    pub fn bytes(&self) -> usize {
        self.w.len
    }
}

/// IEEE half bits to float
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, ex, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = if ex == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut ex = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                ex -= 1;
            }
            (s << 31) | (ex << 23) | ((m & 0x3ff) << 13)
        }
    } else if ex == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((ex + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// A tensor's bytes as float32 (F32, F16, BF16, Q8_0, Q4_0)
pub fn to_f32(t: &Tensor, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match t.ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::Q8_0 => b.chunks_exact(34).flat_map(|blk| {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            blk[2..].iter().map(move |q| *q as i8 as f32 * d)
        }).collect(),
        GType::Q4_0 => b.chunks_exact(18).flat_map(|blk| {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let q = &blk[2..];
            (0..32).map(move |j| (if j < 16 { (q[j] & 15) as i32 } else { (q[j - 16] >> 4) as i32 } - 8) as f32 * d)
        }).collect(),
        ty => return Err(Error(format!("{}: type {}, not one this engine reads (F32, F16, BF16, Q8_0, Q4_0)", t.name, ty.name()))),
    })
}

/// A GGUF file of this engine's (the backbone, a decoder, the speaker encoder, the steering bundle)
pub struct Weights {
    pub g: Gguf,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        Ok(Weights { g: Gguf::open(path).map_err(|x| Error(x.0))? })
    }

    pub fn name(&self) -> String {
        self.g.paths[0].display().to_string()
    }

    pub fn find(&self, name: &str) -> Result<&Tensor> {
        self.g.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.name())))
    }

    pub fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    /// row-major, outermost first
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.find(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        to_f32(t, &self.g.read(t).map_err(|x| Error(x.0))?)
    }

    /// A tensor as float64: F64 as stored (the teacher's frontend tables), any other type through float32
    pub fn f64(&self, name: &str) -> Result<Vec<f64>> {
        let t = self.find(name)?;
        let b = self.g.read(t).map_err(|x| Error(x.0))?;
        if t.ty == GType::F64 {
            return Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or_default())).collect());
        }
        Ok(to_f32(t, &b)?.into_iter().map(|x| x as f64).collect())
    }

    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.find(name)?;
        self.g.read(t).map_err(|x| Error(x.0))
    }

    /// A tensor as float32 on the card
    pub fn dev(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        ops.from_f32(&self.f32(name)?)
    }

    /// A tensor's bytes on the card as stored (a quantized expert stack, a half table)
    pub fn dev_raw(&self, ops: &Ops, name: &str) -> Result<(DevBuf, GType)> {
        let t = self.find(name)?;
        let b = self.g.read(t).map_err(|x| Error(x.0))?;
        let d = DevBuf::new(&ops.gpu, b.len()).map_err(e)?;
        d.write(0, &b).map_err(e)?;
        Ok((d, t.ty))
    }

    /// A matrix [n, k] (a tensor of two dimensions, or more with the leading ones its rows) as float32 on the card
    pub fn mat(&self, ops: &Ops, name: &str) -> Result<Mat> {
        let s = self.shape(name)?;
        let k = *s.last().unwrap_or(&1);
        let n = s.iter().rev().skip(1).product::<usize>().max(1);
        let v = self.f32(name)?;
        Ok(Mat { w: ops.from_f32(&v)?, n, k })
    }

    pub fn meta(&self, key: &str) -> Result<&Value> {
        self.g.meta(key).ok_or_else(|| Error(format!("{}: no {key}", self.name())))
    }

    pub fn u(&self, key: &str) -> Result<usize> {
        self.meta(key)?.as_u64().map(|v| v as usize).ok_or_else(|| Error(format!("{}: {key} is not a count", self.name())))
    }

    pub fn f(&self, key: &str) -> Result<f64> {
        self.meta(key)?.as_f64().ok_or_else(|| Error(format!("{}: {key} is not a number", self.name())))
    }

    pub fn s(&self, key: &str) -> Result<String> {
        self.meta(key)?.as_str().map(str::to_string).ok_or_else(|| Error(format!("{}: {key} is not text", self.name())))
    }

    pub fn us(&self, key: &str) -> Result<Vec<usize>> {
        let a = self.meta(key)?.as_array().ok_or_else(|| Error(format!("{}: {key} is not a list", self.name())))?;
        a.iter().map(|v| v.as_u64().map(|x| x as usize).ok_or_else(|| Error(format!("{}: {key} holds a non-count", self.name())))).collect()
    }

    pub fn fs(&self, key: &str) -> Result<Vec<f64>> {
        let a = self.meta(key)?.as_array().ok_or_else(|| Error(format!("{}: {key} is not a list", self.name())))?;
        a.iter().map(|v| v.as_f64().ok_or_else(|| Error(format!("{}: {key} holds a non-number", self.name())))).collect()
    }

    pub fn ss(&self, key: &str) -> Result<Vec<String>> {
        let a = self.meta(key)?.as_array().ok_or_else(|| Error(format!("{}: {key} is not a list", self.name())))?;
        Ok(a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
    }
}

/// A float pointer, or null, for an optional buffer
pub fn opt(b: &Option<DevBuf>) -> F {
    b.as_ref().map_or(nulf(), |b| b.fp() as F)
}

pub fn nul() -> *const std::ffi::c_void {
    none()
}
