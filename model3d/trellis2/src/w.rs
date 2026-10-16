//! Weights: the flows' GGUFs (ComfyUI-GGUF style: arch "flux", the original names, `adaLN_modulation.1` kept as
//! `guidance_in` and a few others Flux-named - `names` tries both) and the decoders' / DINOv3's safetensors. Linear
//! layers on the GPU as stored (Q8_0 blocks, expanded to half into a scratch buffer just before each product), as half,
//! or as float32; small tensors on the host as float32.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_gguf::{GType, Gguf};

use crate::ffi::K;

pub fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

fn f16(b: [u8; 2]) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes(b))
}

/// The Flux names ComfyUI-GGUF gave some of TRELLIS's tensors (Aero-Ex's files), tried when the original is absent
const FLUX: &[(&str, &str)] = &[("t_embedder.", "time_in."), ("adaLN_modulation.1.", "guidance_in."), ("input_layer.", "img_in."), ("out_layer.", "final_layer.")];

/// A model file: a GGUF (the flows) or a safetensors (the decoders, DINOv3)
pub enum File {
    Gguf(Gguf),
    St(SafeTensors),
}

impl File {
    pub fn open(path: &std::path::Path) -> Result<File> {
        if path.extension().is_some_and(|e| e == "gguf") {
            Ok(File::Gguf(Gguf::open(path).map_err(ge)?))
        } else {
            Ok(File::St(SafeTensors::open(path).map_err(ge)?))
        }
    }

    pub fn path(&self) -> String {
        match self {
            File::Gguf(g) => g.paths[0].display().to_string(),
            File::St(s) => s.path.display().to_string(),
        }
    }

    /// The name this file stores `name` under (the original, or its Flux alias), if any
    pub fn name(&self, name: &str) -> Option<String> {
        let has = |n: &str| match self {
            File::Gguf(g) => g.tensor(n).is_some(),
            File::St(s) => s.tensor(n).is_some(),
        };
        if has(name) {
            return Some(name.to_string());
        }
        FLUX.iter().find_map(|(o, f)| name.strip_prefix(o).map(|rest| format!("{f}{rest}")).filter(|n| has(n)))
    }

    pub fn has(&self, name: &str) -> bool {
        self.name(name).is_some()
    }

    /// (shape, type, bytes) of a tensor
    pub fn raw(&self, name: &str) -> Result<(Vec<usize>, GType, Vec<u8>)> {
        let n = self.name(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.path())))?;
        match self {
            File::Gguf(g) => {
                let t = g.tensor(&n).unwrap();
                Ok((t.shape.iter().map(|d| *d as usize).collect(), t.ty, g.read(t).map_err(ge)?))
            }
            File::St(s) => {
                let t = s.tensor(&n).unwrap();
                let ty = match t.dtype.as_str() {
                    "F32" => GType::F32,
                    "F16" => GType::F16,
                    "BF16" => GType::BF16,
                    d => return Err(Error(format!("{n}: {d}, not a float type"))),
                };
                Ok((t.shape.iter().map(|d| *d as usize).collect(), ty, s.read(t).map_err(ge)?))
            }
        }
    }

    /// A tensor as float32 on the host (F32, F16, BF16, Q8_0), with its shape
    pub fn host(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, ty, b) = self.raw(name)?;
        Ok((shape, floats(name, ty, &b)?))
    }

    pub fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.host(name)?.1)
    }
}

/// Bytes of a tensor as float32
pub fn floats(name: &str, ty: GType, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16([c[0], c[1]])).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GType::Q8_0 => {
            let mut o = Vec::with_capacity(b.len() / 34 * 32);
            for blk in b.chunks_exact(34) {
                let d = f16([blk[0], blk[1]]);
                o.extend(blk[2..].iter().map(|q| d * (*q as i8) as f32));
            }
            o
        }
        other => return Err(Error(format!("{name}: {} is not read here", other.name()))),
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

/// A half buffer from float32 values
pub fn dev_half(gpu: &Arc<Gpu>, v: &[f32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len() * 2)?;
    b.write(0, &half_bytes(v))?;
    Ok(b)
}

/// How a linear layer's matrix is kept on the GPU
pub enum Wt {
    /// Q8_0 blocks as stored, expanded to half before each product
    Q8(DevBuf),
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

/// How to keep a matrix
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Q8_0 as stored, float types as half
    Stored,
    Half,
    F32,
}

impl Lin {
    /// `p.weight` ([n, k], or a convolution's [n, ...] flattened) and `p.bias` when the file has it
    pub fn load(f: &File, gpu: &Arc<Gpu>, p: &str, keep: Keep) -> Result<Lin> {
        let (shape, ty, raw) = f.raw(&format!("{p}.weight"))?;
        let n = shape[0];
        let k = shape[1..].iter().product::<usize>();
        let bias = if f.has(&format!("{p}.bias")) { Some(DevBuf::from_f32(gpu, &f.vec(&format!("{p}.bias"))?)?) } else { None };
        let w = if ty == GType::Q8_0 && keep == Keep::Stored {
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            Wt::Q8(b)
        } else {
            let v = floats(p, ty, &raw)?;
            if keep == Keep::F32 { Wt::F32(DevBuf::from_f32(gpu, &v)?) } else { Wt::Half(dev_half(gpu, &v)?) }
        };
        Ok(Lin { w, bias, n, k })
    }

    /// From host values (n x k) and a bias
    pub fn from_host(gpu: &Arc<Gpu>, w: &[f32], bias: Option<&[f32]>, n: usize, k: usize, keep: Keep) -> Result<Lin> {
        let w = if keep == Keep::F32 { Wt::F32(DevBuf::from_f32(gpu, w)?) } else { Wt::Half(dev_half(gpu, w)?) };
        Ok(Lin { w, bias: bias.map(|b| DevBuf::from_f32(gpu, b)).transpose()?, n, k })
    }

    pub fn bytes(&self) -> usize {
        match &self.w {
            Wt::Q8(b) | Wt::Half(b) | Wt::F32(b) => b.len,
        }
    }

    fn bias_p(&self) -> *const f32 {
        self.bias.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }

    /// The matrix in half, ready for products: its own buffer, or expanded into `scratch` (n x k halves at least)
    pub fn ready(&self, nsd: &Nsd, scratch: &DevBuf) -> Result<*const std::ffi::c_void> {
        Ok(match &self.w {
            Wt::Q8(b) => {
                if scratch.len < self.n * self.k * 2 {
                    return Err(Error(format!("a scratch of {} bytes for a {} x {} matrix", scratch.len, self.n, self.k)));
                }
                nsd.dequant(b.ptr(), GType::Q8_0.code(), self.n * self.k, scratch.ptr(), Dt::F16)?;
                scratch.ptr()
            }
            Wt::Half(b) => b.ptr(),
            Wt::F32(_) => return Err(Error("a float32 matrix has no half product".into())),
        })
    }

    /// out [m, n] = x [m, k] (half) W^T + bias, in half or (accumulated and written in) float32
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if let Wt::F32(w) = &self.w {
            return nsd.linear(x, Dt::F32, m, self.k, w.ptr(), self.n, self.bias_p().cast(), out, out_dt);
        }
        let w = self.ready(nsd, scratch)?;
        if out_dt == Dt::F32 {
            k8.linear_f32(x, m, self.k, w, self.n, self.bias_p(), out.cast())
        } else {
            nsd.linear(x, Dt::F16, m, self.k, w, self.n, self.bias_p().cast(), out, Dt::F16)
        }
    }
}

/// A null device pointer
pub fn null() -> *const std::ffi::c_void {
    none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rounds_to_nearest_even() {
        for v in [0.0f32, 1.0, -2.5, 65504.0, 1e-5, 3.25, -0.000061, 1e9] {
            let h = to_half(v);
            let back = nextsycl_gguf::safetensors::f16_to_f32(h);
            if v.abs() < 65504.0 {
                assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
            } else if v > 65504.0 {
                assert!(back.is_infinite());
            }
        }
        assert_eq!(to_half(1.0), 0x3c00);
        assert_eq!(to_half(-2.0), 0xc000);
    }

    #[test]
    fn q8_0_blocks_read() {
        // one block: scale 0.5, values -2, 3
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        b[2] = (-2i8) as u8;
        b[3] = 3;
        let v = floats("t", GType::Q8_0, &b).unwrap();
        assert_eq!(v.len(), 32);
        assert_eq!((v[0], v[1], v[2]), (-1.0, 1.5, 0.0));
    }
}
