//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half or int8 (a scale per
//! row), read from the checkpoint's safetensors shards (bf16 / f32) and converted on the GPU.
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

    pub fn rope_partial(&self, x: M, xs: usize, rows: usize, s: usize, h: usize, d: usize, rot: usize, inv: &DevBuf) -> Result<()> {
        // SAFETY: half rows of h heads x d, xs apart; inv rot / 2 floats.
        ffi::check(unsafe { (self.k.rope_partial)(self.g(), x, xs as i64, rows as i64, s as i64, h as i64, d as i64, rot as i64, inv.fp()) }, "rope")
    }

    pub fn embed(&self, table: &DevBuf, d: usize, idx: &[i32], scale: f32, out: *mut f32, ldo: usize, copies: usize) -> Result<()> {
        // SAFETY: table rows of d halfs (the indices in range: the caller's), out copies rows.
        ffi::check(unsafe { (self.k.embed)(self.g(), table.ptr(), d as i64, idx.as_ptr(), idx.len() as i32, scale, out, ldo as i64, copies as i32) }, "embed")
    }

    pub fn mix(&self, h: *const f32, n: usize, d: usize, w: &[f32], scale: f32, out: *mut f32) -> Result<()> {
        // SAFETY: h n rows of d floats, out d floats; w is read on the host.
        ffi::check(unsafe { (self.k.mix)(self.g(), h, n as i32, d as i64, w.as_ptr(), scale, out) }, "mix")
    }

    pub fn to_half(&self, x: *const f32, out: M, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x, out, n as i64) }, "to half")
    }

    pub fn bf16_to_half(&self, x: P, out: M, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.bf16_to_half)(self.g(), x.cast(), out, n as i64) }, "bf16 to half")
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

    pub fn dit_in(&self, lat: *const f32, cond: *const f32, l: usize, c: usize, cc: usize, out: *mut f32) -> Result<()> {
        // SAFETY: lat [l, c], cond [l, cc], out [2, l, 2c + cc] floats.
        ffi::check(unsafe { (self.k.dit_in)(self.g(), lat, cond, l as i64, c as i64, cc as i64, out) }, "the flow transformer's input")
    }

    pub fn cfg_step(&self, lat: *mut f32, vc: *const f32, vu: *const f32, n: usize, cfg: f32, dt: f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.cfg_step)(self.g(), lat, vc, vu, n as i64, cfg, dt) }, "euler step")
    }

    pub fn blend(&self, x: *mut f32, p: *const f32, q: *const f32, n: usize, a: f32, b: f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.blend)(self.g(), x, p, q, n as i64, a, b) }, "blend")
    }

    pub fn nearest_rows(&self, x: *const f32, c: usize, li: usize, lo: usize, out: *mut f32) -> Result<()> {
        // SAFETY: x [c, li], out [lo, c] floats.
        ffi::check(unsafe { (self.k.nearest_rows)(self.g(), x, c as i64, li as i64, lo as i64, out) }, "nearest")
    }

    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    /// A 1-D convolution on oneDNN: x [b, ci, l] -> out [b, co, lo], w [co, ci, k] (kept alive: its reordered copy is cached)
    pub fn conv1d(&self, x: *const f32, b: usize, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize, dil: usize,
                  pad: usize, out: *mut f32, lo: usize) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.conv1d)(self.g(), x, b as i64, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64, dil as i64,
                                            pad as i64, out, lo as i64) }, "conv1d")
    }

    /// The transposed one: w [ci, co, k]
    pub fn conv_transpose1d(&self, x: *const f32, b: usize, ci: usize, l: usize, w: &DevBuf, co: usize, k: usize, bias: *const f32, stride: usize,
                            pad: usize, out: *mut f32, lo: usize) -> Result<()> {
        // SAFETY: as conv1d.
        ffi::check(unsafe { (self.k.conv_transpose1d)(self.g(), x, b as i64, ci as i64, l as i64, w.fp(), co as i64, k as i64, bias, stride as i64,
                                                      pad as i64, out, lo as i64) }, "conv_transpose1d")
    }

    pub fn tanh(&self, x: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.tanh)(self.g(), x, n as i64) }, "tanh")
    }
}

/// A matrix [n, k] on the GPU: half, or int8 with a scale per row
pub struct Mat {
    pub w: DevBuf,
    pub scale: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
    /// int8 ConvRot: the rows were rotated by the normalized Hadamard matrix in groups of this many inputs (0: none)
    pub group: usize,
}

/// ConvRot's group: 256 inputs
pub const GROUP: usize = 256;

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

    /// out [m, n] (f32 or half) = x [m, k] half . W^T + bias by oneDNN: a half matrix, or an int8 ConvRot one (the
    /// activations rotated and quantized per row on the fly, the card's int8 rate)
    pub fn apply_half(&self, nsd: &Nsd, xh: P, m: usize, bias: P, out: M, out_dt: Dt) -> Result<()> {
        match &self.scale {
            None => nsd.linear(xh, Dt::F16, m, self.k, self.w.ptr(), self.n, bias, out, out_dt),
            Some(s) if self.group > 0 => nsd.int8_linear(xh, Dt::F16, m, self.k, self.w.ptr(), self.n, s.ptr(), self.n, bias, out, out_dt, self.group),
            Some(_) => Err(Error("apply_half: a half or ConvRot matrix".into())),
        }
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
        Ok(Mat { w, scale, n, k, group: 0 })
    }

    /// `mat`'s rows as int8 ConvRot: float32 on the GPU, rotated along their inputs by the normalized Hadamard
    /// matrix `had` [GROUP, GROUP] (W . H, a GEMM), quantized per row
    pub fn mat_convrot(&self, ops: &Ops, nsd: &Nsd, parts: &[(&str, usize, usize)], had: &DevBuf) -> Result<Mat> {
        let gpu = &ops.gpu;
        let mut k = 0;
        let mut n = 0;
        let mut biggest = 0;
        for (name, r0, r1) in parts {
            k = self.shape(name)?.iter().skip(1).product();
            n += r1 - r0;
            biggest = biggest.max((r1 - r0) * k);
        }
        if k % GROUP != 0 {
            return Err(Error(format!("{}: {k} inputs, not a multiple of {GROUP}", parts[0].0)));
        }
        let f = DevBuf::f32(gpu, n * k)?;
        let stage = DevBuf::new(gpu, biggest * 2)?;
        let mut row = 0;
        for (name, r0, r1) in parts {
            self.rows_to(ops, name, *r0, *r1, &stage, &f, row * k)?;
            row += r1 - r0;
        }
        let rot = DevBuf::f32(gpu, n * k)?;
        nsd.linear(f.ptr(), Dt::F32, n * k / GROUP, GROUP, had.ptr(), GROUP, none(), rot.ptr(), Dt::F32)?;
        let w = DevBuf::new(gpu, n * k)?;
        let scale = DevBuf::f32(gpu, n)?;
        ops.quant_rows(rot.fp(), n, k, w.ptr(), scale.fp())?;
        gpu.sync()?;
        Ok(Mat { w, scale: Some(scale), n, k, group: GROUP })
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
