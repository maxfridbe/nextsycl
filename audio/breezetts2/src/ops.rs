//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half on the GPU, expanded at
//! load from the GGUF's blocks (weights.rs).
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

type P = *const c_void;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// A null pointer (an absent bias or norm)
pub fn nul() -> P {
    none()
}

/// The engine's kernels on one GPU
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

    /// out[m] (+)= x[m] . W^T, m < M <= 8, out rows `ldo` apart (K a multiple of 256)
    pub fn gemv(&self, x: *const f32, m: usize, w: &Mat, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, w.k as i64, w.w.ptr(), 0, std::ptr::null(), w.n as i64, std::ptr::null(), out, ldo as i64,
                                          acc as i32) }, "gemv")
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

    /// Qwen3's per-head RMS norm and RoPE (heads of 128)
    pub fn qk_norm_rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, w: &DevBuf, eps: f32, inv: &DevBuf, p0: usize) -> Result<()> {
        // SAFETY: rows of h heads x 128 floats, xs apart; w 128 floats, inv 64.
        ffi::check(unsafe { (self.k.qk_norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, w.fp(), eps, inv.fp(), p0 as i64) }, "q/k norm + rope")
    }

    /// Per-head RMS norm (none: no weight) and RoPE from a frequency table, any head size up to 256
    pub fn head_norm_rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, d: usize, w: Option<&DevBuf>, eps: f32, inv: &DevBuf, p0: usize)
                          -> Result<()> {
        // SAFETY: rows of h heads x d floats, xs apart; w d floats, inv d / 2.
        ffi::check(unsafe { (self.k.head_norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, d as i64, w.map_or(std::ptr::null(), |w| w.fp()),
                                                    eps, inv.fp(), p0 as i64) }, "head norm + rope")
    }

    /// Bidirectional attention, each query over keys (i - left, i + right)
    pub fn attn_window(&self, q: *const f32, k: *const f32, v: *const f32, l: usize, hq: usize, hkv: usize, d: usize, qs: usize, kvs: usize, left: usize,
                       right: usize, scale: f32, out: *mut f32) -> Result<()> {
        // SAFETY: q rows of hq x d floats qs apart, k / v rows of hkv x d kvs apart, out [l, hq x d].
        ffi::check(unsafe { (self.k.attn_window)(self.g(), q, k, v, l as i64, hq as i64, hkv as i64, d as i64, qs as i64, kvs as i64, left as i64, right as i64,
                                                 scale, out) }, "windowed attention")
    }

    /// out [r, f] = gelu_tanh(gate) * up from gu [r, 2f]
    pub fn gelu_mul(&self, gu: *const f32, r: usize, f: usize, out: *mut f32) -> Result<()> {
        // SAFETY: gu [r, 2f], out [r, f] floats.
        ffi::check(unsafe { (self.k.gelu_mul)(self.g(), gu, r as i64, f as i64, out) }, "gelu mul")
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
}

/// A matrix [n, k] on the GPU in half
pub struct Mat {
    pub w: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    /// out [m, n] f32 = x [m, k] f32 . W^T: a few rows by the engine's own product (bound by reading W once; k a
    /// multiple of 256), more - or any k - by oneDNN's half GEMM (x converted into `xh`, a half buffer of m * k)
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32) -> Result<()> {
        if m <= 8 && self.k.is_multiple_of(256) {
            return ops.gemv(x, m, self, out, self.n, false);
        }
        if xh.len < m * self.k * 2 {
            return Err(Error(format!("a half staging buffer of {} bytes for a [{m}, {}] product", xh.len, self.k)));
        }
        ops.to_half(x, xh.ptr(), m * self.k)?;
        nsd.linear(xh.ptr(), Dt::F16, m, self.k, self.w.ptr(), self.n, none(), out.cast(), Dt::F32)
    }
}
