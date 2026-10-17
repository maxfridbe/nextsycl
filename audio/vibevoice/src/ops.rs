//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half on the GPU (with an
//! optional float32 bias), expanded at load from the GGUF's blocks (weights.rs).
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

    /// out[m] (+)= x[m] . W^T (+ bias), m < M <= 20, out rows `ldo` apart
    pub fn gemv(&self, x: *const f32, m: usize, w: &Mat, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        self.gemv_epi(x, m, w.k, w, out, ldo, if acc { Epi::Acc } else { Epi::Store }, None)
    }

    /// out[m] (epi) x[m] . W^T + bias for m < M <= 20 rows of x `ldx` floats apart (a window of a longer signal, a
    /// convolution's), out rows `ldo` apart; `gamma` the layer scale of Epi::Res
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_epi(&self, x: *const f32, m: usize, ldx: usize, w: &Mat, out: *mut f32, ldo: usize, epi: Epi, gamma: Option<&DevBuf>) -> Result<()> {
        let bias = w.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32);
        let scales = w.scales.as_ref().map_or(std::ptr::null(), |s| s.ptr() as *const c_void);
        let gamma = gamma.map_or(std::ptr::null(), |g| g.fp() as *const f32);
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, ldx as i64, w.k as i64, w.w.ptr(), w.wt as i32, scales, w.n as i64, bias, epi as i32, gamma, out,
                                           ldo as i64) }, "gemv")
    }

    /// RMS norm of the rows of x [m, c] into out: x / rms * w (w None: 1); with `modulate` (scale, shift, their rows
    /// `lds` apart): then v * (1 + scale) + shift
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(&self, x: *const f32, m: usize, c: usize, w: Option<&DevBuf>, eps: f32, modulate: Option<(*const f32, *const f32, usize)>, out: *mut f32)
                    -> Result<()> {
        let (sc, sh, lds) = modulate.unwrap_or((std::ptr::null(), std::ptr::null(), 0));
        // SAFETY: x and out m * c floats; w c; scale / shift m rows of c, lds apart.
        ffi::check(unsafe { (self.k.rms_norm)(self.g(), x, m as i64, c as i64, w.map_or(std::ptr::null(), |w| w.fp() as *const f32), eps, sc, sh, lds as i64, out) },
                   "rms norm")
    }

    /// A decode step's RoPE and key / value store: rows of `qkv` (`ws` apart), row r of sequence `seqs[r]` (its
    /// cache's keys, values, positions) at the position in `pos` (device int32 [rows])
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv(&self, qkv: *mut f32, ws: usize, pos: &DevBuf, hq: usize, hkv: usize, d: usize, inv: &DevBuf, seqs: &[(&DevBuf, &DevBuf, usize)]) -> Result<()> {
        let (a, b) = (seqs[0], *seqs.get(1).unwrap_or(&seqs[0]));
        // SAFETY: qkv rows of (hq + 2 hkv) d floats; the caches [hkv, t, d] half; pos rows int32.
        ffi::check(unsafe { (self.k.rope_kv)(self.g(), qkv, ws as i64, seqs.len() as i64, pos.ptr() as *const i32, hq as i64, hkv as i64, d as i64, inv.fp(),
                                             a.0.ptr(), a.1.ptr(), a.2 as i64, b.0.ptr(), b.1.ptr(), b.2 as i64) }, "rope + kv store")
    }

    /// A decode step's attention: row r of sequence `seqs[r]` up to its position in `pos`; the keys in `pieces`
    /// pieces (`part`: rows * hq * pieces * 130 floats); out [rows, hq * 128]
    #[allow(clippy::too_many_arguments)]
    pub fn attn_dec(&self, q: *const f32, qs: usize, pos: &DevBuf, seqs: &[(&DevBuf, &DevBuf, usize)], hq: usize, hkv: usize, pieces: usize, out: *mut f32,
                    part: &DevBuf) -> Result<()> {
        let (a, b) = (seqs[0], *seqs.get(1).unwrap_or(&seqs[0]));
        if part.len < seqs.len() * hq * pieces * 130 * 4 {
            return Err(Error(format!("attention scratch of {} bytes for {} rows in {pieces} pieces", part.len, seqs.len())));
        }
        // SAFETY: as rope_kv; part checked above.
        ffi::check(unsafe { (self.k.attn_dec)(self.g(), q, qs as i64, seqs.len() as i64, pos.ptr() as *const i32, a.0.ptr(), a.1.ptr(), a.2 as i64, b.0.ptr(),
                                              b.1.ptr(), b.2 as i64, hq as i64, hkv as i64, 128, pieces as i64, out, part.fp()) }, "attention")
    }

    /// out [m, c] = silu(gate) * up of x [m, 2c] holding interleaved (gate, up) pairs
    pub fn swiglu_pairs(&self, x: *const f32, m: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x m * 2c floats, out m * c.
        ffi::check(unsafe { (self.k.swiglu_pairs)(self.g(), x, m as i64, c as i64, out) }, "swiglu")
    }

    /// Floats of scratch `attn` needs
    pub fn attn_scratch(&self, b: usize, s: usize, hq: usize, d: usize, p0: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.attn_scratch)(b as i64, s as i64, hq as i64, d as i64, p0 as i64) as usize }
    }

    /// Causal attention against a half cache [b, hkv, t, d] holding positions [0, p0 + s) (six query heads a key head)
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

    /// RoPE (rotate-half) from a frequency table on rows of h heads x d, row r at position p0 + r % s
    pub fn rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, d: usize, inv: &DevBuf, p0: usize) -> Result<()> {
        // SAFETY: rows of h heads x d floats, xs apart; inv d / 2 floats.
        ffi::check(unsafe { (self.k.head_norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, d as i64, std::ptr::null(), 0.0, inv.fp(),
                                                    p0 as i64) }, "rope")
    }

    pub fn to_half(&self, x: *const f32, out: *mut c_void, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    /// x += y (n floats)
    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    /// out [c, r] = x [r, c]
    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r * c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    /// A 1-D convolution of one signal x [ci, l] (oneDNN): w [co, ci, k] (transposed: [ci, co, k]) float32, pad_l /
    /// pad_r zeros (a transposed one: outputs cut from its start / end); out [co, lo]
    pub fn conv(&self, transposed: bool, x: *const f32, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: &DevBuf, stride: usize, pad_l: usize,
                pad_r: usize, out: *mut f32, lo: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named; the weight stays alive and unchanged (oneDNN keeps its
        // reordered copy by its address).
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), transposed as i32, x, 1, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias.fp(), stride as i64, 1,
                                               pad_l as i64, pad_r as i64, out, lo as i64) }, "conv1d")
    }

    /// x [r, c] += scale[c] * y [r, c]
    pub fn scale_add(&self, x: *mut f32, y: *const f32, scale: &DevBuf, r: usize, c: usize) -> Result<()> {
        // SAFETY: r * c floats each, scale c.
        ffi::check(unsafe { (self.k.scale_add)(self.g(), x, y, scale.fp(), r as i64, c as i64) }, "scale add")
    }

    /// x = silu(x) over n
    pub fn silu(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.silu)(self.g(), x, n as i64) }, "silu")
    }

    /// The depthwise causal convolution over l rows of c channels after the k - 1 rows of `ctx`
    pub fn dwconv(&self, x: *const f32, ctx: &DevBuf, l: usize, c: usize, w: &DevBuf, bias: &DevBuf, k: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x and out l * c floats, ctx (k - 1) * c, w c * k, bias c.
        ffi::check(unsafe { (self.k.dwconv)(self.g(), x, ctx.fp(), l as i64, c as i64, w.fp(), bias.fp(), k as i64, out) }, "depthwise conv")
    }

    /// ctx [p, c] becomes the last p rows of [ctx; x] (x [l, c])
    pub fn ctx_push(&self, ctx: &DevBuf, p: usize, x: *const f32, l: usize, c: usize) -> Result<()> {
        // SAFETY: ctx p * c floats, x l * c.
        ffi::check(unsafe { (self.k.ctx_push)(self.g(), ctx.fp(), p as i64, x, l as i64, c as i64) }, "context push")
    }
}

/// What a product does with its sums (vv.h's NS_VV_*)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Epi {
    Store = 0,
    Acc = 1,
    /// rows 2i, 2i + 1 are gate i and up i: out[i] = silu(gate) * up
    Swiglu = 2,
    Gelu = 3,
    /// out += gamma * v
    Res = 4,
}

/// A matrix [n, k] on the GPU: half (wt 0), bfloat16 (1) or Q8_0 (2: int8 codes [n, k], `scales` half [n, k / 32]),
/// its bias float32 [n] (or none). `k` may be padded with zero columns to a multiple of 256: the input rows are then
/// `k` wide, zero past `k_in`.
pub struct Mat {
    pub w: DevBuf,
    pub wt: u8,
    pub scales: Option<DevBuf>,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
    pub k_in: usize,
}

impl Mat {
    /// Bytes its weights take on the card
    pub fn bytes(&self) -> usize {
        self.w.len + self.scales.as_ref().map_or(0, |s| s.len)
    }

    /// out [m, n] f32 = x [m, k] f32 . W^T + bias: a few rows by the engine's own product (bound by reading W once), more
    /// by oneDNN's half GEMM (x converted into `xh`, a half buffer of m * k)
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32) -> Result<()> {
        self.apply_with(ops, nsd, x, m, xh, out, None)
    }

    /// As apply; a Q8_0 matrix's wide product expands it into `deq` (half [n, k]) first
    #[allow(clippy::too_many_arguments)]
    pub fn apply_with(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: &DevBuf, out: *mut f32, deq: Option<&DevBuf>) -> Result<()> {
        if m <= 8 && self.k.is_multiple_of(8) {
            return ops.gemv(x, m, self, out, self.n, false);
        }
        if xh.len < m * self.k * 2 {
            return Err(Error(format!("a half staging buffer of {} bytes for a [{m}, {}] product", xh.len, self.k)));
        }
        let w = match self.wt {
            0 => self.w.ptr(),
            2 => {
                let d = deq.filter(|d| d.len >= self.n * self.k * 2)
                    .ok_or_else(|| Error(format!("a Q8_0 [{}, {}] product of {m} rows needs room to expand it", self.n, self.k)))?;
                let sc = self.scales.as_ref().ok_or_else(|| Error("a Q8_0 matrix without its scales".into()))?;
                // SAFETY: codes n * k bytes, scales n * k / 32 halves, d n * k halves (checked).
                ffi::check(unsafe { (ops.k.dequant_q8)(ops.g(), self.w.ptr(), sc.ptr(), self.n as i64, self.k as i64, d.ptr()) }, "Q8_0 expand")?;
                d.ptr()
            }
            t => return Err(Error(format!("a wide product of weights of type {t}"))),
        };
        ops.to_half(x, xh.ptr(), m * self.k)?;
        let bias = self.bias.as_ref().map_or(none(), |b| b.ptr() as *const c_void);
        nsd.linear(xh.ptr(), Dt::F16, m, self.k, w, self.n, bias, out.cast(), Dt::F32)
    }
}
