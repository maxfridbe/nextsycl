//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: the file's Q8_0 blocks on the
//! GPU as int8 codes and half scales (the decode's product reads them as they are), or half (an F16 file's); a prompt's
//! wide products expand a matrix to float32 for oneDNN's GEMM.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// The engine's kernels on one GPU
#[derive(Clone)]
pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), k: ffi::api()? })
    }

    fn g(&self) -> *mut c_void {
        self.gpu.raw()
    }

    /// Floats of scratch `attn` needs
    pub fn attn_scratch(&self, b: usize, s: usize, hq: usize, d: usize, p0: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.attn_scratch)(b as i64, s as i64, hq as i64, d as i64, p0 as i64) as usize }
    }

    /// Causal attention against a half cache [b, hkv, t, d] holding positions [0, p0 + s)
    pub fn attn(&self, q: *const f32, qs: usize, kc: &DevBuf, vc: &DevBuf, b: usize, s: usize, hq: usize, hkv: usize, d: usize, t: usize, p0: usize,
                out: *mut f32, part: *mut f32) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.attn)(self.g(), q, qs as i64, kc.ptr(), vc.ptr(), b as i64, s as i64, hq as i64, hkv as i64, d as i64, t as i64, p0 as i64,
                                          out, part) }, "attention")
    }

    pub fn kv_store(&self, k: *const f32, v: *const f32, ks: usize, b: usize, s: usize, hkv: usize, d: usize, t: usize, p0: usize, kc: &DevBuf,
                    vc: &DevBuf) -> Result<()> {
        // SAFETY: as attn.
        ffi::check(unsafe { (self.k.kv_store)(self.g(), k, v, ks as i64, b as i64, s as i64, hkv as i64, d as i64, t as i64, p0 as i64, kc.ptr(), vc.ptr()) },
                   "kv store")
    }

    /// Per-head RMS norm (none: no weight) and RoPE on interleaved pairs from a (cos, sin) table, positions p0 + row % s
    pub fn norm_rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, d: usize, w: Option<&DevBuf>, eps: f32, cs: &DevBuf, p0: usize)
                     -> Result<()> {
        // SAFETY: rows of h heads x d floats, xs apart; w d floats; cs (p0 + s) x d floats.
        ffi::check(unsafe { (self.k.norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, d as i64, w.map_or(std::ptr::null(), |w| w.fp()),
                                               eps, cs.fp(), p0 as i64) }, "norm + rope")
    }

    /// out (copies rows, ldo apart) = sum of the table's rows `idx` (at most 8): table half [R, d]
    pub fn embed(&self, table: &DevBuf, d: usize, idx: &[i32], out: *mut f32, ldo: usize, copies: usize) -> Result<()> {
        // SAFETY: table rows of d halfs (the indices in range: the caller's), out copies rows.
        ffi::check(unsafe { (self.k.embed)(self.g(), table.ptr(), d as i64, idx.as_ptr(), idx.len() as i32, 1.0, out, ldo as i64, copies as i32) }, "embed")
    }

    pub fn to_half(&self, x: *const f32, out: *mut c_void, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    pub fn tanh(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.tanh)(self.g(), x, n as i64) }, "tanh")
    }

    /// A causal 1-D convolution on oneDNN: x [b, ci, l] -> out [b, co, l], w [co, ci, k] ((k - 1) * dil zeros ahead;
    /// the weight kept alive: its reordered copy is cached by its pointer)
    pub fn conv_causal(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, dil: usize, out: *mut f32) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, 1, dil as i64, ((k - 1) * dil) as i64, 0,
                                               out, l as i64) }, "conv1d")
    }

    /// A transposed one, its end trimmed to l * stride (fish-speech's CausalTransConvNet): w [ci, co, k]
    pub fn conv_up(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, out: *mut f32) -> Result<()> {
        // SAFETY: as conv_causal.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 1, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1, 0, (k - stride) as i64,
                                               out, (l * stride) as i64) }, "conv_transpose1d")
    }

    /// A strided causal convolution (fish-speech's CausalConvNet): (k - stride) zeros ahead, behind as many as the last
    /// window needs; out [co, ceil(l / stride)]; its length returned
    pub fn conv_strided(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, out: *mut f32)
                        -> Result<usize> {
        let pt = k - stride;
        let lo = l.div_ceil(stride);
        let extra = (lo - 1) * stride + k - pt - l;
        // SAFETY: as conv_causal.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1, pt as i64,
                                               extra as i64, out, lo as i64) }, "conv1d (strided)")?;
        Ok(lo)
    }

    /// A depthwise causal convolution of x [c, l] (k - 1 zeros ahead): w [c, k], bias [c]
    pub fn dwconv(&self, x: *const f32, c: usize, l: usize, w: &DevBuf, bias: &DevBuf, k: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x and out [c, l], w [c, k], bias c floats.
        ffi::check(unsafe { (self.k.dwconv)(self.g(), x, c as i64, l as i64, w.fp(), bias.fp(), k as i64, out) }, "depthwise conv")
    }

    /// Causal attention over a window: query i sees keys (i - w, i]
    pub fn window_attn(&self, q: *const f32, k: *const f32, v: *const f32, l: usize, h: usize, d: usize, qs: usize, kvs: usize, w: usize, out: *mut f32)
                       -> Result<()> {
        // SAFETY: q, k, v rows of h x d floats qs / kvs apart, out [l, h x d].
        ffi::check(unsafe { (self.k.window_attn)(self.g(), q, k, v, l as i64, h as i64, d as i64, qs as i64, kvs as i64, w as i64, out) }, "window attention")
    }

    /// x [r, c] += scale[c] * y [r, c]
    pub fn scale_add(&self, x: *mut f32, y: *const f32, scale: &DevBuf, r: usize, c: usize) -> Result<()> {
        // SAFETY: x and y [r, c], scale c floats.
        ffi::check(unsafe { (self.k.scale_add)(self.g(), x, y, scale.fp(), r as i64, c as i64) }, "scaled add")
    }
}

/// A matrix's weights on the GPU: half [n, k], or Q8_0 as int8 codes [n, k] and half scales [n, k / 32]
pub enum Store {
    Half(DevBuf),
    Q8 { q: DevBuf, s: DevBuf },
}

/// A matrix [n, k] on the GPU
pub struct Mat {
    pub w: Store,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    /// Bytes a float32 copy of it takes (the staging a wide product expands it into)
    pub fn f32_bytes(&self) -> usize {
        self.n * self.k * 4
    }

    /// out[m] (+)= x[m] . W^T, m <= 8 rows, out rows `ldo` apart (k a multiple of 256): bound by reading W once
    pub fn gemv(&self, ops: &Ops, x: *const f32, m: usize, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        match &self.w {
            // SAFETY: the caller's buffers hold the sizes named.
            Store::Half(w) => ffi::check(unsafe { (ops.k.gemv)(ops.g(), x, m as i64, self.k as i64, w.ptr(), 0, std::ptr::null(), self.n as i64, std::ptr::null(),
                                                               out, ldo as i64, acc as i32) }, "gemv"),
            // SAFETY: as above.
            Store::Q8 { q, s } => ffi::check(unsafe { (ops.k.gemv_q8)(ops.g(), x, m as i64, self.k as i64, q.ptr().cast(), s.ptr(), self.n as i64, out, ldo as i64,
                                                                    acc as i32) }, "gemv (q8_0)"),
        }
    }

    /// out [m, n] f32 = x [m, k] f32 . W^T: a few rows by the decode's product, more by oneDNN's float32 GEMM with the
    /// matrix expanded into `wf` (a float32 buffer of n * k). Not half's GEMM: some prompts' activations pass half's
    /// range (a down projection's input of tens of thousands: inf in half, the prompt's logits wrong)
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, wf: Option<&DevBuf>, out: *mut f32) -> Result<()> {
        if m <= 8 && self.k.is_multiple_of(256) {
            return self.gemv(ops, x, m, out, self.n, false);
        }
        let wf = wf.filter(|b| b.len >= self.f32_bytes()).ok_or_else(|| Error(format!("no float32 staging for a [{}, {}] matrix", self.n, self.k)))?;
        match &self.w {
            // SAFETY: w n * k halfs, wf n * k floats.
            Store::Half(w) => ffi::check(unsafe { (ops.k.half_float)(ops.g(), w.ptr(), (self.n * self.k) as i64, wf.fp()) }, "half to float")?,
            // SAFETY: q [n, k], s [n, k / 32], wf n * k floats.
            Store::Q8 { q, s } => ffi::check(unsafe { (ops.k.dequant_q8)(ops.g(), q.ptr().cast(), s.ptr(), self.n as i64, self.k as i64, wf.fp()) }, "dequant q8_0")?,
        }
        nsd.linear(x.cast(), Dt::F32, m, self.k, wf.ptr(), self.n, none(), out.cast(), Dt::F32)
    }
}

/// A float32 linear layer [n, k] (+ bias) on oneDNN: the codec's (float32 throughout, as the reference)
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    pub fn apply(&self, nsd: &Nsd, x: *const f32, m: usize, out: *mut f32) -> Result<()> {
        nsd.linear(x.cast(), Dt::F32, m, self.k, self.w.ptr(), self.n, self.b.as_ref().map_or(none(), |b| b.ptr()), out.cast(), Dt::F32)
    }
}

/// fish-speech's rotary table (precompute_freqs_cis): for each position and pair i < d / 2, (cos, sin) of position *
/// base^(-2i / d), each rounded to bfloat16 as upstream keeps them (the slow, fast and codec models alike)
pub fn rope_table(positions: usize, d: usize, base: f64) -> Vec<f32> {
    // in float32, as torch computes 1 / base ** (arange(0, d, 2) / d)
    let freqs: Vec<f32> = (0..d / 2).map(|i| 1.0 / (base as f32).powf((2 * i) as f32 / d as f32)).collect();
    let mut t = Vec::with_capacity(positions * d);
    for p in 0..positions {
        for f in &freqs {
            // torch.outer in float32, then polar's cos / sin, then .to(bfloat16)
            let a = p as f32 * f;
            t.push(bf16_round(a.cos()));
            t.push(bf16_round(a.sin()));
        }
    }
    t
}

/// A float rounded to the nearest bfloat16 (ties to even), as float
pub fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotary_table_is_upstreams_bfloat16() {
        let t = rope_table(3, 4, 10000.0);
        assert_eq!(t.len(), 12);
        // position 0: cos 1, sin 0 for every pair
        assert_eq!(&t[..4], &[1.0, 0.0, 1.0, 0.0]);
        // position 1, pair 0: angle 1 -> cos 0.5403 -> bf16 0.5390625, sin 0.8415 -> 0.83984375
        assert_eq!((t[4], t[5]), (0.5390625, 0.83984375));
        // a bfloat16 step at 1 is 1/128: halfway ties go to the even mantissa
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(bf16_round(1.0 + 1.0 / 200.0), 1.0 + 1.0 / 128.0);
    }
}
