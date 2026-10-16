//! The codec's kernels (ffi.rs) as checked calls on device pointers, and its weights: the speech tokenizer's
//! safetensors (float32; bf16 widened on the GPU), each tensor found by name.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_gguf::safetensors::{SafeTensors, StTensor};

use crate::ffi;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// The codec's kernels on one GPU
#[derive(Clone)]
pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { gpu: gpu.clone(), k: ffi::api()? })
    }

    fn g(&self) -> *mut std::ffi::c_void {
        self.gpu.raw()
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
    }

    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        // SAFETY: r x c floats each.
        ffi::check(unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) }, "transpose")
    }

    pub fn bf16_to_float(&self, x: *const u16, out: *mut f32, n: usize) -> Result<()> {
        // SAFETY: n values each.
        ffi::check(unsafe { (self.k.bf16_to_float)(self.g(), x, out, n as i64) }, "bf16 to float")
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

    pub fn elu(&self, x: *const f32, n: usize, out: *mut f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.elu)(self.g(), x, n as i64, out) }, "elu")
    }

    pub fn clamp(&self, x: *mut f32, n: usize, lo: f32, hi: f32) -> Result<()> {
        // SAFETY: n floats.
        ffi::check(unsafe { (self.k.clamp)(self.g(), x, n as i64, lo, hi) }, "clamp")
    }
}

/// The speech tokenizer's safetensors, its tensors found by name
pub struct Weights {
    file: SafeTensors,
}

fn elem(t: &StTensor) -> Result<usize> {
    match t.dtype.as_str() {
        "BF16" => Ok(2),
        "F32" => Ok(4),
        d => Err(Error(format!("{}: {d}, not bf16 or f32", t.name))),
    }
}

impl Weights {
    pub fn open(path: &Path) -> Result<Weights> {
        Ok(Weights { file: SafeTensors::open(path).map_err(|e| Error(e.0))? })
    }

    pub fn find(&self, name: &str) -> Result<&StTensor> {
        self.file.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.file.path.display())))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.find(name)?.shape.iter().map(|d| *d as usize).collect())
    }

    /// A tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        self.find(name)?;
        self.file.f32(name).map_err(|e| Error(e.0))
    }

    /// A tensor as float32 on the GPU
    pub fn dev_f32(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        let t = self.find(name)?;
        let n = t.elements() as usize;
        let es = elem(t)?;
        let b = self.file.read(t).map_err(|e| Error(e.0))?;
        let out = DevBuf::f32(&ops.gpu, n)?;
        if es == 4 {
            out.write(0, &b)?;
        } else {
            let stage = DevBuf::new(&ops.gpu, n * 2)?;
            stage.write(0, &b)?;
            ops.bf16_to_float(stage.ptr().cast(), out.fp(), n)?;
            ops.gpu.sync()?;
        }
        Ok(out)
    }
}
