//! Weights: safetensors files (Budget3D's denoiser, Idol's decoders and Sobol positions, the encoders), linear layers on
//! the GPU as half (the transformer blocks) or float32 (what Budget3D keeps in float32: the RePo layers, the refiners,
//! the heads), small tensors on the host as float32. Idol-turbo's adapted matrices are stored as `<name>.base.weight`
//! beside their LoRA experts: `name` finds either.

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::K;

pub fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A safetensors file
pub struct St(pub SafeTensors);

impl St {
    pub fn open(path: &Path) -> Result<St> {
        Ok(St(SafeTensors::open(path).map_err(ge)?))
    }
    pub fn path(&self) -> String {
        self.0.path.display().to_string()
    }
    /// The name `name` is stored under: itself, or (an Idol-turbo matrix) with `.base` before its last part
    pub fn name(&self, name: &str) -> Option<String> {
        if self.0.tensor(name).is_some() {
            return Some(name.to_string());
        }
        let (a, b) = name.rsplit_once('.')?;
        let alt = format!("{a}.base.{b}");
        self.0.tensor(&alt).is_some().then_some(alt)
    }
    pub fn has(&self, name: &str) -> bool {
        self.name(name).is_some()
    }
    /// A tensor as float32 on the host
    pub fn vec(&self, name: &str) -> Result<Vec<f32>> {
        let n = self.name(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.path())))?;
        self.0.f32(&n).map_err(ge)
    }
    /// A tensor's shape
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let n = self.name(name).ok_or_else(|| Error(format!("{}: no tensor {name}", self.path())))?;
        Ok(self.0.tensor(&n).unwrap().shape.iter().map(|d| *d as usize).collect())
    }
    /// A tensor of exactly `n` values
    pub fn vec_n(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let v = self.vec(name)?;
        if v.len() != n {
            return Err(Error(format!("{}: {name} has {} values, not {n}", self.path(), v.len())));
        }
        Ok(v)
    }
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

/// An int32 buffer
pub fn dev_i32(gpu: &Arc<Gpu>, v: &[i32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len() * 4)?;
    b.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(b)
}

/// A linear layer: out [m, n] = x [m, k] W^T + bias; W half (`half`) or float32
pub struct Lin {
    pub w: DevBuf,
    pub half: bool,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    /// `p.weight` ([n, k], or a convolution's [n, ...] flattened) and `p.bias` when the file has it
    pub fn load(f: &St, gpu: &Arc<Gpu>, p: &str, half: bool) -> Result<Lin> {
        let shape = f.shape(&format!("{p}.weight"))?;
        let (n, k) = (shape[0], shape[1..].iter().product::<usize>());
        let w = f.vec(&format!("{p}.weight"))?;
        let b = if f.has(&format!("{p}.bias")) { Some(f.vec(&format!("{p}.bias"))?) } else { None };
        Lin::from_host(gpu, &w, b.as_deref(), n, k, half)
    }

    /// From host values (n x k) and a bias
    pub fn from_host(gpu: &Arc<Gpu>, w: &[f32], bias: Option<&[f32]>, n: usize, k: usize, half: bool) -> Result<Lin> {
        if w.len() != n * k {
            return Err(Error(format!("a {n} x {k} matrix from {} values", w.len())));
        }
        let w = if half { dev_half(gpu, w)? } else { DevBuf::from_f32(gpu, w)? };
        Ok(Lin { w, half, bias: bias.map(|b| DevBuf::from_f32(gpu, b)).transpose()?, n, k })
    }

    pub fn bytes(&self) -> usize {
        self.w.len + self.bias.as_ref().map_or(0, |b| b.len)
    }

    pub fn bias_p(&self) -> *const f32 {
        self.bias.as_ref().map_or(std::ptr::null(), |b| b.fp())
    }

    /// out [m, n] = x [m, k] W^T + bias: a half matrix takes half x and writes half (`out_dt` F16) or float32
    /// (accumulated in float32); a float32 matrix takes and writes float32
    pub fn apply(&self, nsd: &Nsd, k8: &K, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if !self.half {
            if out_dt != Dt::F32 {
                return Err(Error("a float32 matrix writes float32".into()));
            }
            return nsd.linear(x, Dt::F32, m, self.k, self.w.ptr(), self.n, self.bias_p().cast(), out, Dt::F32);
        }
        if out_dt == Dt::F32 {
            k8.linear_f32(x, m, self.k, self.w.ptr(), self.n, self.bias_p(), out.cast())
        } else {
            nsd.linear(x, Dt::F16, m, self.k, self.w.ptr(), self.n, self.bias_p().cast(), out, Dt::F16)
        }
    }
}

/// A small linear layer applied on the host (the time and level embeddings, the camera's MLP, the heads' last rows)
pub struct HostLin {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl HostLin {
    pub fn load(f: &St, p: &str) -> Result<HostLin> {
        let shape = f.shape(&format!("{p}.weight"))?;
        let (n, k) = (shape[0], shape[1..].iter().product::<usize>());
        let b = if f.has(&format!("{p}.bias")) { f.vec_n(&format!("{p}.bias"), n)? } else { vec![0.0; n] };
        Ok(HostLin { w: f.vec_n(&format!("{p}.weight"), n * k)?, b, n, k })
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

/// GELU's tanh approximation
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

/// The sinusoidal embedding of a scalar (Idol's SinusoidalEmbedder before its MLP): [cos(t f_i), sin(t f_i)] with
/// f_i = period^(-i / half) (times 2 pi when `two_pi`)
pub fn sinusoid(t: f32, dim: usize, period: f32, two_pi: bool) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let f = (-(period as f64).ln() * i as f64 / half as f64).exp() as f32;
        let mut a = t * f;
        if two_pi {
            a *= 2.0 * std::f32::consts::PI;
        }
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_rounds_to_nearest_even() {
        assert_eq!(to_half(1.0), 0x3c00);
        assert_eq!(to_half(-2.0), 0xc000);
        let h = to_half(0.1);
        assert!((nextsycl_gguf::safetensors::f16_to_f32(h) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn sinusoids_are_cos_then_sin() {
        let e = sinusoid(0.0, 8, 10000.0, false);
        assert_eq!(e, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let e = sinusoid(1.0, 4, 10000.0, false);
        // f_0 = 1, f_1 = 10000^(-1/2)
        assert!((e[0] - 1f32.cos()).abs() < 1e-6 && (e[2] - 1f32.sin()).abs() < 1e-6);
        assert!((e[1] - 0.01f32.cos()).abs() < 1e-6);
    }
}
