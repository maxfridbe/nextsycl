//! Weights as the GGUF stores them, on the card: a matrix kept in its type (the shared quantized linear reads it), small
//! vectors as float32.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf};

pub fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A matrix as stored, on the card: `rows` outputs of `k` inputs
pub struct Part {
    pub w: DevBuf,
    pub qtype: u32,
    pub rows: usize,
    pub k: usize,
}

impl Part {
    pub fn load(f: &Gguf, gpu: &Arc<Gpu>, name: &str) -> Result<Part> {
        let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
        let code = t.ty.code();
        if ![0, 1, 8, 12, 13, 14, 30].contains(&code) {
            return Err(Error(format!("{name}: {} weights are not read yet", t.ty.name())));
        }
        let (rows, k) = (t.shape[0] as usize, t.shape[1] as usize);
        let bytes = f.read(t).map_err(ge)?;
        let w = DevBuf::new(gpu, bytes.len())?;
        w.write(0, &bytes)?;
        Ok(Part { w, qtype: code, rows, k })
    }

    /// out [m, rows] = x [m, k] (half) . W^T + bias, `ldo` the out's row stride (0: rows)
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&self, nsd: &Nsd, x: *const std::ffi::c_void, m: usize, bias: Option<&DevBuf>, out: *mut std::ffi::c_void, out_dt: Dt, ldo: usize) -> Result<()> {
        let b = bias.map_or(none(), |b| b.ptr().cast_const());
        nsd.linear_q(x, Dt::F16, m, self.k, self.w.ptr(), self.qtype, self.rows, b, out, out_dt, ldo)
    }
}

/// A float tensor (float32, bfloat16 or half) on the host as float32
pub fn host_vec(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))?;
    let b = f.read(t).map_err(ge)?;
    Ok(match t.ty {
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{name}: a {} tensor where float was expected", other.name()))),
    })
}

/// A float tensor on the card as float32
pub fn vecd(f: &Gguf, gpu: &Arc<Gpu>, name: &str) -> Result<DevBuf> {
    DevBuf::from_f32(gpu, &host_vec(f, name)?)
}
