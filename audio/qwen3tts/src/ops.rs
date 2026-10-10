//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half or int8 (a scale per
//! row), read from the checkpoint's safetensors (bf16 / f32) and converted on the GPU. (The MiniMax Music engine's
//! ops, with the codec's kernels beside them.)
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

use nextsycl_audio::{Error, Result};
use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::{SafeTensors, StTensor};

use crate::ffi;

type P = *const c_void;
type M = *mut c_void;

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

    fn g(&self) -> M {
        self.gpu.raw()
    }

    /// out[m] (+)= x[m] . W^T (* scale + bias), m < M <= 8, out rows `ldo` apart
    pub fn gemv(&self, x: *const f32, m: usize, k: usize, w: &Mat, bias: P, out: *mut f32, ldo: usize, acc: bool) -> Result<()> {
        let (wt, sc) = match &w.scale {
            None => (0, std::ptr::null()),
            Some(s) => (1, s.fp() as *const f32),
        };
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gemv)(self.g(), x, m as i64, k as i64, w.w.ptr(), wt, sc, w.n as i64, bias.cast(), out, ldo as i64, acc as i32) }, "gemv")
    }

    /// Floats of scratch `attn` needs
    pub fn attn_scratch(&self, b: usize, s: usize, hq: usize, d: usize, p0: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe { (self.k.attn_scratch)(b as i64, s as i64, hq as i64, d as i64, p0 as i64) as usize }
    }

    pub fn attn(&self, q: *const f32, qs: usize, kc: &DevBuf, vc: &DevBuf, b: usize, s: usize, hq: usize, hkv: usize, d: usize, t: usize, p0: usize,
                out: *mut f32, part: *mut f32) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named; the cache [b, hkv, t, d] half.
        ffi::check(unsafe { (self.k.attn)(self.g(), q, qs as i64, kc.ptr(), vc.ptr(), b as i64, s as i64, hq as i64, hkv as i64, d as i64, t as i64, p0 as i64,
                                          out, part) }, "attention")
    }

    pub fn kv_store(&self, k: *const f32, v: *const f32, ks: usize, b: usize, s: usize, hkv: usize, d: usize, t: usize, p0: usize, kc: &DevBuf,
                    vc: &DevBuf) -> Result<()> {
        // SAFETY: as attn.
        ffi::check(unsafe { (self.k.kv_store)(self.g(), k, v, ks as i64, b as i64, s as i64, hkv as i64, d as i64, t as i64, p0 as i64, kc.ptr(), vc.ptr()) },
                   "kv store")
    }

    pub fn qk_norm_rope(&self, x: *mut f32, xs: usize, rows: usize, s: usize, h: usize, w: &DevBuf, eps: f32, inv: &DevBuf, p0: usize) -> Result<()> {
        // SAFETY: rows of h heads x 128 floats, xs apart; w 128 floats, inv 64.
        ffi::check(unsafe { (self.k.qk_norm_rope)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, w.fp(), eps, inv.fp(), p0 as i64) }, "q/k norm + rope")
    }

    pub fn embed(&self, table: &DevBuf, d: usize, idx: &[i32], scale: f32, out: *mut f32, ldo: usize, copies: usize) -> Result<()> {
        // SAFETY: table rows of d halfs (the indices in range: the caller's), out copies rows.
        ffi::check(unsafe { (self.k.embed)(self.g(), table.ptr(), d as i64, idx.as_ptr(), idx.len() as i32, scale, out, ldo as i64, copies as i32) }, "embed")
    }

    pub fn to_half(&self, x: *const f32, out: M, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    pub fn bf16_to_float(&self, x: P, out: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.bf16_to_float)(self.g(), x.cast(), out, n as i64) }, "bf16 to float")
    }

    pub fn quant_rows(&self, w: *const f32, n: usize, k: usize, q: M, scale: *mut f32) -> Result<()> {
        // SAFETY: w [n, k] floats, q [n, k] bytes, scale n floats.
        ffi::check(unsafe { (self.k.quant_rows)(self.g(), w, n as i64, k as i64, q.cast(), scale) }, "int8 rows")
    }

    pub fn dequant_rows(&self, q: P, scale: *const f32, n: usize, k: usize, out: M) -> Result<()> {
        // SAFETY: q [n, k] bytes, scale n floats, out [n, k] halfs.
        ffi::check(unsafe { (self.k.dequant_rows)(self.g(), q.cast(), scale, n as i64, k as i64, out) }, "int8 rows to half")
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    /// A causal 1-D convolution on oneDNN: x [b, ci, l] -> out [b, co, l], w [co, ci, k] ((k - 1) * dil zeros ahead;
    /// the weight kept alive: its reordered copy is cached by its pointer)
    pub fn conv_causal(&self, x: *const f32, b: usize, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, dil: usize, out: *mut f32)
                       -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, b as i64, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, 1, dil as i64,
                                               ((k - 1) * dil) as i64, 0, out, l as i64) }, "conv1d")
    }

    /// A transposed one, its end trimmed to l * stride: w [ci, co, k]
    pub fn conv_up(&self, x: *const f32, b: usize, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, out: *mut f32)
                   -> Result<()> {
        // SAFETY: as conv_causal.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 1, x, b as i64, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1, 0,
                                               (k - stride) as i64, out, (l * stride) as i64) }, "conv_transpose1d")
    }

    pub fn snake_beta(&self, x: *const f32, c: usize, l: usize, la: &DevBuf, lb: &DevBuf, out: *mut f32) -> Result<()> {
        // SAFETY: x and out [c, l] floats, la and lb c floats.
        ffi::check(unsafe { (self.k.snake_beta)(self.g(), x, c as i64, l as i64, la.fp(), lb.fp(), out) }, "snake")
    }

    pub fn dwconv(&self, x: *const f32, c: usize, l: usize, w: &DevBuf, bias: &DevBuf, k: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x and out [c, l], w [c, k], bias c floats.
        ffi::check(unsafe { (self.k.dwconv)(self.g(), x, c as i64, l as i64, w.fp(), bias.fp(), k as i64, out) }, "depthwise conv")
    }

    pub fn window_attn(&self, q: *const f32, k: *const f32, v: *const f32, l: usize, h: usize, d: usize, qs: usize, kvs: usize, w: usize, out: *mut f32)
                       -> Result<()> {
        // SAFETY: q, k, v rows of h x d floats qs / kvs apart, out [l, h x d].
        ffi::check(unsafe { (self.k.window_attn)(self.g(), q, k, v, l as i64, h as i64, d as i64, qs as i64, kvs as i64, w as i64, out) }, "window attention")
    }

    pub fn rope(&self, x: *mut f32, xs: usize, l: usize, h: usize, d: usize, theta: f32, p0: usize) -> Result<()> {
        // SAFETY: rows of h x d floats, xs apart.
        ffi::check(unsafe { (self.k.rope)(self.g(), x, xs as i64, l as i64, h as i64, d as i64, theta, p0 as i64) }, "rope")
    }

    pub fn scale_add(&self, x: *mut f32, y: *const f32, scale: &DevBuf, r: usize, c: usize) -> Result<()> {
        // SAFETY: x and y [r, c], scale c floats.
        ffi::check(unsafe { (self.k.scale_add)(self.g(), x, y, scale.fp(), r as i64, c as i64) }, "scaled add")
    }

    pub fn silu(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.silu)(self.g(), x, n as i64) }, "silu")
    }

    pub fn elu(&self, x: *const f32, n: usize, out: *mut f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.elu)(self.g(), x, n as i64, out) }, "elu")
    }

    /// A strided causal convolution as Mimi's: (k - stride) zeros ahead, behind as many as the last window needs;
    /// out [b, co, ceil(l / stride)]; its length returned
    pub fn conv_strided(&self, x: *const f32, b: usize, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, out: *mut f32)
                        -> Result<usize> {
        let pt = k - stride;
        let frames = (l + pt).saturating_sub(k).div_ceil(stride);
        let extra = frames * stride + k - pt - l;
        let lo = frames + 1;
        // SAFETY: as conv_causal.
        ffi::check(unsafe { (self.k.conv1d_lr)(self.g(), 0, x, b as i64, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, 1,
                                               pt as i64, extra as i64, out, lo as i64) }, "conv1d")?;
        Ok(lo)
    }

    pub fn clamp(&self, x: *mut f32, n: usize, lo: f32, hi: f32) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.clamp)(self.g(), x, n as i64, lo, hi) }, "clamp")
    }
}

/// A matrix [n, k] on the GPU: half, or int8 with a scale per row
pub struct Mat {
    pub w: DevBuf,
    pub scale: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Mat {
    /// out [m, n] f32 = x [m, k] f32 . W^T + bias: a few rows by the engine's own product (bound by reading W once,
    /// int8 weights against float activations), more by oneDNN's half GEMM (x converted into `xh`, a half buffer of
    /// m * k; an int8 matrix first expanded to half in `wh`, n * k halfs - quantizing the activations too, oneDNN's
    /// int8 path, loses Qwen3's outlier features)
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&self, ops: &Ops, nsd: &Nsd, x: *const f32, m: usize, xh: Option<&DevBuf>, wh: Option<&DevBuf>, bias: P, out: *mut f32) -> Result<()> {
        if m <= 8 {
            return ops.gemv(x, m, self.k, self, bias, out, self.n, false);
        }
        let xh = xh.ok_or_else(|| Error("a half staging buffer for a wide product".into()))?;
        ops.to_half(x, xh.ptr(), m * self.k)?;
        let w = match &self.scale {
            None => self.w.ptr(),
            Some(s) => {
                let wh = wh.ok_or_else(|| Error("a half matrix buffer for a wide int8 product".into()))?;
                ops.dequant_rows(self.w.ptr(), s.fp(), self.n, self.k, wh.ptr())?;
                wh.ptr()
            }
        };
        nsd.linear(xh.ptr(), Dt::F16, m, self.k, w, self.n, bias, out.cast(), Dt::F32)
    }

    pub fn bytes(&self) -> usize {
        self.w.len + self.scale.as_ref().map_or(0, |s| s.len)
    }
}

/// A model's safetensors shards, its tensors found by name across them
pub struct Shards {
    files: Vec<SafeTensors>,
}

impl Shards {
    pub fn open(paths: &[PathBuf]) -> Result<Shards> {
        if paths.is_empty() {
            return Err(Error("no files".into()));
        }
        let files = paths.iter().map(|p| SafeTensors::open(p).map_err(|e| Error(e.0))).collect::<Result<Vec<_>>>()?;
        Ok(Shards { files })
    }

    pub fn find(&self, name: &str) -> Result<(&SafeTensors, &StTensor)> {
        self.files.iter().find_map(|f| f.tensor(name).map(|t| (f, t))).ok_or_else(|| {
            Error(format!("{}: no tensor {name}", self.files[0].path.parent().map(|p| p.display().to_string()).unwrap_or_default()))
        })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.find(name)?.1.shape.iter().map(|d| *d as usize).collect())
    }

    /// A tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (f, _) = self.find(name)?;
        f.f32(name).map_err(|e| Error(e.0))
    }

    /// Rows [r0, r1) of a 2-D (or more: rows of the rest) bf16 / f32 tensor as float32 on the host
    pub fn rows_f32(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let (f, t) = self.find(name)?;
        let row: usize = t.shape.iter().skip(1).product::<u64>() as usize;
        let es = elem(t)?;
        let mut b = vec![0u8; (r1 - r0) * row * es];
        f.read_into(t, (r0 * row * es) as u64, &mut b).map_err(|e| Error(e.0))?;
        Ok(match t.dtype.as_str() {
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            _ => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        })
    }

    /// Rows [r0, r1) of a tensor onto the GPU as float32 at `dst` (f32 offset `at`), through `stage` (bytes, at least
    /// the rows' size)
    fn rows_to(&self, ops: &Ops, name: &str, r0: usize, r1: usize, stage: &DevBuf, dst: &DevBuf, at: usize) -> Result<usize> {
        let (f, t) = self.find(name)?;
        let row: usize = t.shape.iter().skip(1).product::<u64>() as usize;
        let es = elem(t)?;
        let n = (r1 - r0) * row;
        let mut b = vec![0u8; n * es];
        f.read_into(t, (r0 * row * es) as u64, &mut b).map_err(|e| Error(e.0))?;
        if es == 4 {
            dst.write(at * 4, &b)?;
        } else {
            stage.write(0, &b)?;
            ops.bf16_to_float(stage.ptr(), fp(dst, at), n)?;
            ops.gpu.sync()?;
        }
        Ok(n)
    }

    /// A tensor as float32 on the GPU
    pub fn dev_f32(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        let (_, t) = self.find(name)?;
        let n = t.elements() as usize;
        let out = DevBuf::f32(&ops.gpu, n)?;
        let stage = DevBuf::new(&ops.gpu, n * 2)?;
        self.rows_to(ops, name, 0, t.shape.first().copied().unwrap_or(1) as usize, &stage, &out, 0)?;
        Ok(out)
    }

    /// A matrix from row ranges of tensors, stacked: [(name, r0, r1)], every part `k` wide; half, or int8 rows
    pub fn mat(&self, ops: &Ops, parts: &[(&str, usize, usize)], int8: bool) -> Result<Mat> {
        let mut k = 0;
        let mut n = 0;
        let mut biggest = 0;
        for (name, r0, r1) in parts {
            let s = self.shape(name)?;
            let kk: usize = s.iter().skip(1).product();
            if k != 0 && kk != k {
                return Err(Error(format!("{name}: {kk} wide, the matrix {k}")));
            }
            if *r1 > s[0] || r0 >= r1 {
                return Err(Error(format!("{name}: rows {r0}..{r1} of {}", s[0])));
            }
            k = kk;
            n += r1 - r0;
            biggest = biggest.max((r1 - r0) * kk);
        }
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * if int8 { 1 } else { 2 })?;
        let scale = if int8 { Some(DevBuf::f32(gpu, n)?) } else { None };
        let stage = DevBuf::new(gpu, biggest * 2)?;
        let f = DevBuf::f32(gpu, biggest)?;
        let mut row = 0;
        for (name, r0, r1) in parts {
            let cnt = self.rows_to(ops, name, *r0, *r1, &stage, &f, 0)?;
            match &scale {
                None => ops.to_half(f.fp(), w.ptr().wrapping_byte_add(row * k * 2), cnt)?,
                Some(s) => ops.quant_rows(f.fp(), r1 - r0, k, w.ptr().wrapping_byte_add(row * k), fp(s, row))?,
            }
            ops.gpu.sync()?;
            row += r1 - r0;
        }
        Ok(Mat { w, scale, n, k })
    }

    /// A whole tensor as a matrix
    pub fn mat1(&self, ops: &Ops, name: &str, int8: bool) -> Result<Mat> {
        let r = self.shape(name)?[0];
        self.mat(ops, &[(name, 0, r)], int8)
    }
}

fn elem(t: &StTensor) -> Result<usize> {
    match t.dtype.as_str() {
        "BF16" => Ok(2),
        "F32" => Ok(4),
        d => Err(Error(format!("{}: {d}, not bf16 or f32", t.name))),
    }
}

/// A null pointer (an absent bias)
pub fn nul() -> P {
    none()
}
