//! The engine's kernels (ffi.rs) and the shared diffusion kernels it uses, as checked calls on device pointers.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_audio::Result;
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};

use crate::ffi;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// A pointer `at` bytes into a buffer
pub fn at(b: &DevBuf, bytes: usize) -> *mut c_void {
    b.ptr().wrapping_byte_add(bytes)
}

/// Floats as their bytes
pub fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Int32s as their bytes
pub fn ibytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// A device buffer's floats [at, at + n)
pub fn read_f32(b: &DevBuf, at: usize, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0u8; n * 4];
    b.read(at * 4, &mut v)?;
    Ok(v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// The kernels on one GPU: the engine's own and the shared diffusion context
pub struct Ops {
    pub gpu: Arc<Gpu>,
    pub nsd: Nsd,
    k: &'static ffi::Api,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), nsd: Nsd::new(gpu)?, k: ffi::api()? })
    }

    fn g(&self) -> *mut c_void {
        self.gpu.raw()
    }

    /// out[rows[i]] (rows `ldo` floats apart) = sum over the codebooks of table[ids[i][c] + c * v]; table [c * v, d]
    pub fn audio_embed(&self, table: &DevBuf, c: usize, v: usize, d: usize, ids: &DevBuf, rows: &DevBuf, n: usize, out: *mut f32, ldo: usize) -> Result<()> {
        // SAFETY: table [c * v, d] floats, ids [n, c] and rows [n] int32s, out rows in range (the caller's).
        ffi::check(unsafe { (self.k.audio_embed)(self.g(), table.fp(), c as i64, v as i64, d as i64, ids.ptr().cast(), rows.ptr().cast(), n as i64, out, ldo as i64) },
                   "audio embeddings")
    }

    /// [q | k | v] (hq, hkv, hkv heads of d halfs) -> [q | k' | v'] (hq heads each), r rows
    pub fn expand_qkv(&self, qkv: *const c_void, r: usize, hq: usize, hkv: usize, d: usize, out: *mut c_void) -> Result<()> {
        // SAFETY: qkv [r, (hq + 2 hkv) d] halfs, out [r, 3 hq d] halfs.
        ffi::check(unsafe { (self.k.expand_qkv)(self.g(), qkv, r as i64, hq as i64, hkv as i64, d as i64, out) }, "expand qkv")
    }

    /// The step's best code and its log probability per (codebook, position): logits [2 t, c * v]
    pub fn score(&self, logits: *const f32, t: usize, c: usize, v: usize, mask: usize, guidance: f32, tok: &DevBuf, score: &DevBuf) -> Result<()> {
        // SAFETY: logits [2 t, c v] floats, tok / score c t each.
        ffi::check(unsafe { (self.k.score)(self.g(), logits, t as i64, c as i64, v as i64, mask as i64, guidance, tok.ptr().cast(), score.fp()) }, "scores")
    }

    /// GroupNorm with a group a channel over x [c, l]
    pub fn chan_norm(&self, x: *const f32, c: usize, l: usize, w: &DevBuf, b: &DevBuf, eps: f32, out: *mut f32) -> Result<()> {
        // SAFETY: x and out [c, l], w and b c floats.
        ffi::check(unsafe { (self.k.chan_norm)(self.g(), x, c as i64, l as i64, w.fp(), b.fp(), eps, out) }, "channel norm")
    }

    pub fn elu(&self, x: *const f32, n: usize, out: *mut f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.elu)(self.g(), x, n as i64, out) }, "elu")
    }

    /// A 1-D convolution on oneDNN: x [ci, l] -> out [co, lo], w [co, ci, k] (kept alive: its reordered copy is cached
    /// by its pointer), `pl` zeros ahead and `pr` behind
    pub fn conv(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: Option<&DevBuf>, stride: usize, dil: usize, pl: usize, pr: usize,
                out: *mut f32) -> Result<usize> {
        let lo = (l + pl + pr - dil * (k - 1) - 1) / stride + 1;
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias.map_or(std::ptr::null(), |b| b.fp()),
                                               stride as i64, dil as i64, pl as i64, pr as i64, out, lo as i64) }, "conv1d")?;
        Ok(lo)
    }

    /// A transposed one: x [ci, l], w [ci, co, k]; `pl` cut from the front and `pr` from the end of the full
    /// (l - 1) stride + k
    pub fn conv_t(&self, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: Option<&DevBuf>, stride: usize, pl: usize, pr: usize,
                  out: *mut f32) -> Result<usize> {
        let lo = (l - 1) * stride + k - pl - pr;
        // SAFETY: as conv.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 1, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias.map_or(std::ptr::null(), |b| b.fp()),
                                               stride as i64, 1, pl as i64, pr as i64, out, lo as i64) }, "conv_transpose1d")?;
        Ok(lo)
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    /// out [c, r] = x [r, c]
    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    pub fn to_half(&self, x: *const f32, out: *mut c_void, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    /// Snake over x [c, l] (alpha [c]): x + sin^2(alpha x) / alpha
    pub fn snake(&self, x: *const f32, c: usize, l: usize, alpha: &DevBuf, out: *mut f32) -> Result<()> {
        self.nsd.snake(x.cast(), 1, c, l, alpha.ptr(), out.cast())
    }

    /// out [m, n] = x [m, k] . W^T (+ bias), half in, `out_dt` out
    pub fn linear_h(&self, x: *const c_void, m: usize, k: usize, w: &DevBuf, n: usize, out: *mut c_void, out_dt: Dt) -> Result<()> {
        self.nsd.linear(x, Dt::F16, m, k, w.ptr(), n, none(), out, out_dt)
    }

    /// out [m, n] = x [m, k] . W^T + bias, float32
    pub fn linear_f(&self, x: *const f32, m: usize, k: usize, w: &DevBuf, n: usize, bias: Option<&DevBuf>, out: *mut f32) -> Result<()> {
        self.nsd.linear(x.cast(), Dt::F32, m, k, w.ptr(), n, bias.map_or(none(), |b| b.ptr()), out.cast(), Dt::F32)
    }
}
