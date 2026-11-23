//! Hunyuan's text encoders, shared by HunyuanImage 2.1 (image/hunyuanimage21) and HunyuanVideo-1.5
//! (video/hunyuanvideo15): both take Qwen2.5-VL-7B's hidden_states[-3] (layer 25's output, no final norm) after a
//! template, and Glyph-SDXL-v2's byT5-small for the text a prompt quotes. They differ only in the template, the crop
//! and the quote rules (`te::Template`, `byt5::Quotes`).
//!
//! The kernels they need beyond the shared diffusion ones (`Kern`) are each kind library's own: an engine hands its
//! kernels in (the image library's hi21, the video library's hv15 - the same code; a kind's library holds only its
//! own engines' kernels).

pub mod byt5;
pub mod te;

use std::ffi::c_void;

use nextsycl_core::{Error, Result};
use nextsycl_gguf::{GType, Gguf, Tensor};

/// The engine's kernels the encoders call (hi21.h / hv15.h): queued on the GPU's queue, sizes the caller's to hold
pub trait Kern: Send + Sync {
    /// n values of Q4_0 blocks -> half
    fn q4_0_half(&self, src: *const c_void, n: usize, out: *mut c_void) -> Result<()>;
    /// half = float32 x s, n values
    fn to_half(&self, x: *const f32, out: *mut c_void, n: usize, s: f32) -> Result<()>;
    /// out [m, n] float32 = x [m, k] half . w^T (w [n, k] half) + bias, accumulated and written in float32
    #[allow(clippy::too_many_arguments)]
    fn linear_f32(&self, x: *const c_void, m: usize, k: usize, w: *const c_void, n: usize, bias: *const f32, out: *mut f32) -> Result<()>;
    /// attention over short float32 sequences: scores * scale (+ bias [hq, l, l]), causal or not
    #[allow(clippy::too_many_arguments)]
    fn attn_small(&self, q: *const f32, qs: usize, k: *const f32, v: *const f32, kvs: usize, l: usize, hq: usize, hkv: usize, d: usize, scale: f32,
                  bias: *const f32, causal: bool, out: *mut f32) -> Result<()>;
    /// out [m, c] float32 = gelu_tanh(x[:, :c]) * x[:, c:] (x float32 [m, 2c])
    fn geglu_f32(&self, x: *const c_void, m: usize, c: usize, out: *mut c_void) -> Result<()>;
}

pub(crate) fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A tensor of `f` by name, or why not
pub(crate) fn tensor<'a>(f: &'a Gguf, name: &str) -> Result<&'a Tensor> {
    f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))
}

fn f16(b: [u8; 2]) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes(b))
}

/// Bytes of a float tensor as float32 (F32, F16, BF16)
pub(crate) fn floats(name: &str, ty: GType, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16([c[0], c[1]])).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        other => return Err(Error(format!("{name}: {} where a float tensor was expected", other.name()))),
    })
}

/// A float tensor on the host
pub(crate) fn host(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = tensor(f, name)?;
    floats(name, t.ty, &f.read(t).map_err(ge)?)
}

/// The values of a run of Q4_0 blocks (an fp16 scale, 16 bytes of nibbles a 32 values) on the host: a token's
/// embedding row
pub fn q4_0_row(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 18 * 32);
    for blk in b.chunks_exact(18) {
        let d = f16([blk[0], blk[1]]);
        let mut v = [0f32; 32];
        for j in 0..16 {
            v[j] = d * ((blk[2 + j] & 0xF) as i32 - 8) as f32;
            v[j + 16] = d * ((blk[2 + j] >> 4) as i32 - 8) as f32;
        }
        out.extend_from_slice(&v);
    }
    out
}

/// Matrices of `f` stacked by rows: (raw bytes, rows, columns, type)
pub(crate) fn stacked(f: &Gguf, names: &[String]) -> Result<(Vec<u8>, usize, usize, GType)> {
    let mut raw = Vec::new();
    let (mut n, mut k, mut ty) = (0usize, 0usize, None);
    for name in names {
        let t = tensor(f, name)?;
        if ty.is_some_and(|x| x != t.ty) {
            return Err(Error(format!("{name}: {} beside {:?}", t.ty.name(), ty)));
        }
        ty = Some(t.ty);
        let cols = *t.shape.last().unwrap_or(&0) as usize;
        let rows = t.elements() as usize / cols.max(1);
        if k != 0 && k != cols {
            return Err(Error(format!("{name}: {cols} columns beside {k}")));
        }
        k = cols;
        n += rows;
        raw.extend(f.read(t).map_err(ge)?);
    }
    Ok((raw, n, k, ty.ok_or_else(|| Error("no tensors".into()))?))
}
