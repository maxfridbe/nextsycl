//! Weights: the safetensors files (the flow transformers' fp16 torso and float32 edges, the decoder's float32, DINOv2's
//! and CLIP's float32). Linear layers and convolutions on the GPU as half (fp16 values as stored; float32 rounded to
//! half once at load), or float32 where the reference keeps a layer in float32 and the product is small (the
//! transformer's input and output layers); small tensors on the host as float32.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::{Epi, K};

pub fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

fn f16(b: [u8; 2]) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes(b))
}

/// A safetensors file and the prefix its names are read under
pub struct File {
    pub st: SafeTensors,
    pub prefix: String,
}

impl File {
    pub fn open(path: &std::path::Path, prefix: &str) -> Result<File> {
        Ok(File { st: SafeTensors::open(path).map_err(ge)?, prefix: prefix.to_string() })
    }

    pub fn path(&self) -> String {
        self.st.path.display().to_string()
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.tensor(&format!("{}{name}", self.prefix)).is_some()
    }

    /// (shape, dtype name, bytes) of a tensor
    pub fn raw(&self, name: &str) -> Result<(Vec<usize>, String, Vec<u8>)> {
        let full = format!("{}{name}", self.prefix);
        let t = self.st.tensor(&full).ok_or_else(|| Error(format!("{}: no tensor {full}", self.path())))?;
        Ok((t.shape.iter().map(|d| *d as usize).collect(), t.dtype.clone(), self.st.read(t).map_err(ge)?))
    }

    /// A tensor as float32 on the host, with its shape
    pub fn host(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, ty, b) = self.raw(name)?;
        Ok((shape, floats(name, &ty, &b)?))
    }

    pub fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.host(name)?.1)
    }

    /// A small float32 vector on the GPU
    pub fn dev(&self, gpu: &Arc<Gpu>, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(gpu, &self.vec(name)?)
    }
}

/// Bytes of a tensor as float32
pub fn floats(name: &str, ty: &str, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match ty {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => b.chunks_exact(2).map(|c| f16([c[0], c[1]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        other => return Err(Error(format!("{name}: {other} is not read here"))),
    })
}

/// half bits of a float32 (round to nearest even)
pub fn to_half(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
}

/// float32 values as half bytes
pub fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_half(*x).to_le_bytes()).collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A half buffer from float32 values
pub fn dev_half(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len().max(1) * 2)?;
    b.write(0, &half_bytes(v))?;
    Ok(b)
}

/// An int32 buffer
pub fn dev_i32(gpu: &Arc<Gpu>, v: &[i32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len().max(1) * 4)?;
    b.write(0, &i32_bytes(v))?;
    Ok(b)
}

/// Half values back as float32
pub fn read_half(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    b.read(0, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|c| f16([c[0], c[1]])).collect())
}

/// How a linear layer's matrix is kept on the GPU
pub enum Wt {
    Half(DevBuf),
    F32(DevBuf),
}

/// A linear layer: out [m, n] = x [m, k] W^T + bias
pub struct Lin {
    pub w: Wt,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    /// `p.weight` ([n, k], or a convolution's [n, ...] flattened in its stored order) and `p.bias` when the file has
    /// it; `f32` keeps the matrix in float32 (a float32 product), else half
    pub fn load(f: &File, gpu: &Arc<Gpu>, p: &str, f32: bool) -> Result<Lin> {
        let (shape, ty, raw) = f.raw(&format!("{p}.weight"))?;
        let n = shape[0];
        let k = shape[1..].iter().product::<usize>();
        let bias = if f.has(&format!("{p}.bias")) { Some(DevBuf::from_f32(gpu, &f.vec(&format!("{p}.bias"))?)?) } else { None };
        let w = if (ty == "F16" && !f32) || (ty == "F32" && f32) {
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            if f32 { Wt::F32(b) } else { Wt::Half(b) }
        } else {
            let v = floats(p, &ty, &raw)?;
            if f32 { Wt::F32(DevBuf::from_f32(gpu, &v)?) } else { Wt::Half(dev_half(gpu, &v)?) }
        };
        Ok(Lin { w, bias, n, k })
    }

    /// From host values (n x k) and a bias
    pub fn from_host(gpu: &Arc<Gpu>, w: &[f32], bias: Option<&[f32]>, n: usize, k: usize, f32: bool) -> Result<Lin> {
        let w = if f32 { Wt::F32(DevBuf::from_f32(gpu, w)?) } else { Wt::Half(dev_half(gpu, w)?) };
        Ok(Lin { w, bias: bias.map(|b| DevBuf::from_f32(gpu, b)).transpose()?, n, k })
    }

    pub fn bytes(&self) -> usize {
        match &self.w {
            Wt::Half(b) | Wt::F32(b) => b.len,
        }
    }

    fn bias_p(&self) -> *const f32 {
        self.bias.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }

    /// out [m, n] = x [m, k] W^T + bias: x half (float32 for a float32 matrix), out in half or (accumulated and written
    /// in) float32
    pub fn apply(&self, cx: &crate::Ctx, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        self.apply_epi(&cx.nsd, &cx.k, x, m, out, out_dt, Epi::Plain, None)
    }

    /// `apply` with an epilogue: the silo's product with it fused when the silo takes it (one pass), else the product
    /// and then hi3d.h's pass (a residual's product goes through `tmp`, float32 [m, n], first)
    #[allow(clippy::too_many_arguments)]
    pub fn apply_epi(&self, nsd: &Nsd, k8: &K, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt, epi: Epi,
                     tmp: Option<&DevBuf>) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if let Wt::Half(w) = &self.w {
            if k8.linear_epi(x, m, self.k, w.ptr(), self.n, self.bias_p(), out, out_dt, epi)? {
                return Ok(());
            }
        }
        match epi {
            Epi::Plain => {}
            Epi::Gelu | Epi::GeluErf => {
                self.plain(nsd, k8, x, m, out, out_dt)?;
                return nsd.gelu(out, out_dt, m * self.n, epi == Epi::Gelu);
            }
            Epi::Silu => {
                self.plain(nsd, k8, x, m, out, out_dt)?;
                return k8.act(out, out_dt, m * self.n, 0);
            }
            Epi::Resid(gate) => {
                let t = tmp.filter(|t| t.len >= m * self.n * 4).ok_or_else(|| Error("a residual product without its float32 buffer".into()))?;
                self.plain(nsd, k8, x, m, t.ptr(), Dt::F32)?;
                return k8.gate_add(out.cast(), m, self.n, t.ptr(), Dt::F32, gate);
            }
        }
        self.plain(nsd, k8, x, m, out, out_dt)
    }

    /// hi3d.h's (and the shared nsd_linear's) product
    fn plain(&self, nsd: &Nsd, k8: &K, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        match &self.w {
            Wt::F32(w) => nsd.linear(x, Dt::F32, m, self.k, w.ptr(), self.n, self.bias_p().cast(), out, out_dt),
            Wt::Half(w) if out_dt == Dt::F32 => k8.linear_f32(x, m, self.k, w.ptr(), self.n, self.bias_p(), out.cast()),
            Wt::Half(w) => nsd.linear(x, Dt::F16, m, self.k, w.ptr(), self.n, self.bias_p().cast(), out, Dt::F16),
        }
    }
}

/// A null device pointer
pub fn null() -> *const std::ffi::c_void {
    none()
}

/// A host matrix [n, k] and bias (the timestep embedding, the modulations)
pub struct HostLin {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl HostLin {
    pub fn load(f: &File, p: &str) -> Result<HostLin> {
        let (sh, w) = f.host(&format!("{p}.weight"))?;
        if sh.len() != 2 {
            return Err(Error(format!("{p}.weight: {sh:?}, not a matrix")));
        }
        Ok(HostLin { n: sh[0], k: sh[1], w, b: f.vec(&format!("{p}.bias"))? })
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n).map(|o| {
            let row = &self.w[o * self.k..(o + 1) * self.k];
            self.b[o] + row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }
}

pub fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rounds_to_nearest_even() {
        assert_eq!(to_half(1.0), 0x3c00);
        assert_eq!(to_half(-2.0), 0xc000);
        for v in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 3.25] {
            let back = nextsycl_gguf::safetensors::f16_to_f32(to_half(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
    }
}
