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

    /// lat += dt * (vu + cfg * (vc - vu)) over n values: the guided Euler step
    pub fn cfg_step(&self, lat: *mut f32, vc: *const f32, vu: *const f32, n: usize, cfg: f32, dt: f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.cfg_step)(self.g(), lat, vc, vu, n as i64, cfg, dt) }, "euler step")
    }

    /// out [c, r] = x [r, c] transposed
    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r * c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    /// A 1-D convolution on oneDNN without padding: x [ci, l] -> out [co, l - k + 1] (w [co, ci, k], kept by oneDNN: alive
    /// and unchanged)
    pub fn conv(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, out: *mut f32) -> Result<()> {
        // SAFETY: the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, std::ptr::null(), 1, 1, 0, 0, out,
                                               (l + 1 - k) as i64) }, "conv1d")
    }

    /// A transposed 1-D convolution, k - stride trimmed from the end (upstream's CausalConvTranspose1d, trim ratio 1):
    /// x [ci, l] -> out [co, l * stride] (w [ci, co, k])
    pub fn conv_up(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, stride: usize, out: *mut f32) -> Result<()> {
        // SAFETY: the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 1, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, std::ptr::null(), stride as i64, 1, 0,
                                               (k - stride) as i64, out, (l * stride) as i64) }, "conv_transpose1d")
    }

    /// x [r, c] += scale[c] * y [r, c]
    pub fn scale_add(&self, x: *mut f32, y: *const f32, scale: &DevBuf, r: usize, c: usize) -> Result<()> {
        // SAFETY: r * c floats each, scale c.
        ffi::check(unsafe { (self.k.scale_add)(self.g(), x, y, scale.fp(), r as i64, c as i64) }, "scaled add")
    }

    /// Causal ALiBi attention over keys [i - w, i]: q / k / v rows of h heads x d, qs / kvs apart; out [l, h x d]
    pub fn alibi_attn(&self, q: *const f32, k: *const f32, v: *const f32, l: usize, h: usize, d: usize, qs: usize, kvs: usize, w: usize, out: *mut f32)
                      -> Result<()> {
        // SAFETY: the rows named.
        ffi::check(unsafe { (self.k.alibi_attn)(self.g(), q, k, v, l as i64, h as i64, d as i64, qs as i64, kvs as i64, w as i64, out) }, "alibi attention")
    }

    /// RMS norm of `rows` rows of c values, xs apart, in place (weight [c])
    pub fn rms_rows(&self, x: *mut f32, xs: usize, rows: usize, c: usize, w: &DevBuf, eps: f32) -> Result<()> {
        // SAFETY: rows rows of c floats xs apart, w c floats.
        ffi::check(unsafe { (self.k.rms_rows)(self.g(), x, xs as i64, rows as i64, c as i64, w.fp(), eps) }, "row norm")
    }

    /// out [c, p + l] = x [c, l] padded ahead: `reflect` false repeats the first value, true mirrors (p < l)
    pub fn pad_left(&self, x: *const f32, c: usize, l: usize, p: usize, reflect: bool, out: *mut f32) -> Result<()> {
        // SAFETY: c * l and c * (p + l) floats.
        ffi::check(unsafe { (self.k.pad_left)(self.g(), x, c as i64, l as i64, p as i64, reflect as i32, out) }, "pad")
    }

    /// The flow transformer's six input rows [6, d] from the input projection, the time projection and the condition
    pub fn fm_rows(&self, inp: *const f32, tproj: *const f32, hp: *const f32, d: usize, out: *mut f32) -> Result<()> {
        // SAFETY: d floats each, out 6 d.
        ffi::check(unsafe { (self.k.fm_rows)(self.g(), inp, tproj, hp, d as i64, out) }, "flow rows")
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
