//! Weights from the GGUF files: vectors on the host or the GPU as float32, and linear layers kept as stored (Q8_0 or
//! Q4_0 blocks, expanded to half into a scratch buffer just before their product) or as float32 / half.

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::{GType, Gguf, Tensor};

use crate::ffi::K;

pub fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A tensor of `f` by name, or why not
pub fn tensor<'a>(f: &'a Gguf, name: &str) -> Result<&'a Tensor> {
    f.tensor(name).ok_or_else(|| Error(format!("{}: no tensor {name}", f.paths[0].display())))
}

fn f16(b: [u8; 2]) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes(b))
}

/// Bytes of a float tensor as float32 (F32, F16, BF16)
pub fn floats(name: &str, ty: GType, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match ty {
        GType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GType::F16 => b.chunks_exact(2).map(|c| f16([c[0], c[1]])).collect(),
        GType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        other => return Err(Error(format!("{name}: {} where a float tensor was expected", other.name()))),
    })
}

/// A float tensor on the host
pub fn host(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = tensor(f, name)?;
    floats(name, t.ty, &f.read(t).map_err(ge)?)
}

/// The values of one Q4_0 block run (an fp16 scale, 16 bytes of nibbles a 32 values) on the host: a token's
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

/// How a linear layer's matrix is kept on the GPU
pub enum Wt {
    /// Q8_0 or Q4_0 blocks as stored, expanded to half before each use
    Blocks(DevBuf, GType),
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

/// How to keep a matrix: as stored (quantized blocks; float types as half), as float32, or as half
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    Stored,
    F32,
    Half,
}

/// The row order of a q / k projection whose RoPE pairs features (2i, 2i + 1) of each 128-wide head, for kernels that
/// rotate (i, 64 + i): new row j of a head reads old row perm(j)
pub fn rope_perm(j: usize) -> usize {
    if j < 64 {
        2 * j
    } else {
        2 * (j - 64) + 1
    }
}

/// A row range of a matrix whose rows are permuted per head (`rope_perm`)
#[derive(Clone, Copy)]
pub struct Perm {
    pub from: usize,
    pub to: usize,
}

fn permuted(i: usize, perms: &[Perm]) -> usize {
    for p in perms {
        if i >= p.from && i < p.to {
            let r = i - p.from;
            return p.from + (r / 128) * 128 + rope_perm(r % 128);
        }
    }
    i
}

impl Lin {
    /// The matrices `names` stacked by rows (and their biases `bias`, when given), rows in `perms` reordered per head
    #[allow(clippy::too_many_arguments)]
    pub fn load(f: &Gguf, nsd: &Nsd, k8: &K, names: &[&str], bias: &[&str], perms: &[Perm], keep: Keep) -> Result<Lin> {
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
        let ty = ty.ok_or_else(|| Error("no tensors".into()))?;
        let gpu = &nsd.gpu;
        if !perms.is_empty() {
            let rb = raw.len() / n;
            let mut out = vec![0u8; raw.len()];
            for r in 0..n {
                let s = permuted(r, perms);
                out[r * rb..(r + 1) * rb].copy_from_slice(&raw[s * rb..(s + 1) * rb]);
            }
            raw = out;
        }
        let bias = if bias.is_empty() {
            None
        } else {
            let mut v = Vec::with_capacity(n);
            for b in bias {
                v.extend(host(f, b)?);
            }
            if v.len() != n {
                return Err(Error(format!("{}: a bias of {} for {n} rows", bias[0], v.len())));
            }
            let v: Vec<f32> = (0..n).map(|r| v[permuted(r, perms)]).collect();
            Some(DevBuf::from_f32(gpu, &v)?)
        };
        let quant = matches!(ty, GType::Q8_0 | GType::Q4_0);
        let w = if quant && keep == Keep::Stored {
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            Wt::Blocks(b, ty)
        } else if quant {
            // expanded once at load
            let b = DevBuf::new(gpu, raw.len())?;
            b.write(0, &raw)?;
            let dt = if keep == Keep::Half { Dt::F16 } else { Dt::F32 };
            let o = DevBuf::new(gpu, n * k * if dt == Dt::F32 { 4 } else { 2 })?;
            expand(nsd, k8, &b, ty, n * k, &o, dt)?;
            nsd.wait()?;
            if keep == Keep::Half { Wt::Half(o) } else { Wt::F32(o) }
        } else {
            let v = floats(names[0], ty, &raw)?;
            if keep == Keep::F32 {
                Wt::F32(DevBuf::from_f32(gpu, &v)?)
            } else {
                let o = DevBuf::f32(gpu, n * k)?;
                o.write(0, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                let h = DevBuf::new(gpu, n * k * 2)?;
                k8.to_half(o.fp(), h.ptr(), n * k, 1.0)?;
                gpu.sync()?;
                Wt::Half(h)
            }
        };
        Ok(Lin { w, bias, n, k })
    }

    /// bytes on the GPU
    pub fn bytes(&self) -> usize {
        let w = match &self.w {
            Wt::Blocks(b, _) | Wt::Half(b) | Wt::F32(b) => b.len,
        };
        w + self.bias.as_ref().map_or(0, |b| b.len)
    }

    /// whether its products take half inputs
    pub fn half(&self) -> bool {
        !matches!(self.w, Wt::F32(_))
    }

    /// The matrix in half, ready for products: its own buffer, or expanded into `scratch` (n x k halves at least)
    pub fn ready(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf) -> Result<*const std::ffi::c_void> {
        Ok(match &self.w {
            Wt::Blocks(b, ty) => {
                expand(nsd, k8, b, *ty, self.n * self.k, scratch, Dt::F16)?;
                scratch.ptr()
            }
            Wt::Half(b) | Wt::F32(b) => b.ptr(),
        })
    }

    /// out [m, n] (out_dt) = x [m, k] W^T + bias, with W from `ready` (x half, or float32 for a float32 matrix)
    pub fn gemm(&self, nsd: &Nsd, w: *const std::ffi::c_void, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let dt = if self.half() { Dt::F16 } else { Dt::F32 };
        let bias = self.bias.as_ref().map_or(none(), |b| b.ptr() as *const _);
        nsd.linear(x, dt, m, self.k, w, self.n, bias, out, out_dt)
    }

    /// out [m, n] float32 = x [m, k] float32 W^T + bias with the matrix in float32 (Q8_0 expanded into `scratch`, n x k
    /// floats at least): the modulations, whose inputs and outputs stay float32
    pub fn apply_f32(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void) -> Result<()> {
        let w = match &self.w {
            Wt::Blocks(b, ty) => {
                expand(nsd, k8, b, *ty, self.n * self.k, scratch, Dt::F32)?;
                scratch.ptr()
            }
            Wt::F32(b) => b.ptr(),
            Wt::Half(_) => return Err(Error("a half matrix has no float32 product here".into())),
        };
        let bias = self.bias.as_ref().map_or(none(), |b| b.ptr() as *const _);
        nsd.linear(x, Dt::F32, m, self.k, w, self.n, bias, out, Dt::F32)
    }

    /// `ready` then `gemm`
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let w = self.ready(nsd, k8, scratch)?;
        self.gemm(nsd, w, x, m, out, out_dt)
    }
}

/// n values of quantized blocks into `dt` (half or float32)
fn expand(nsd: &Nsd, k8: &K, src: &DevBuf, ty: GType, n: usize, out: &DevBuf, dt: Dt) -> Result<()> {
    let size = if dt == Dt::F32 { 4 } else { 2 };
    if out.len < n * size {
        return Err(Error(format!("expanding {n} values into {} bytes", out.len)));
    }
    match (ty, dt) {
        (GType::Q8_0, _) => nsd.dequant(src.ptr(), GType::Q8_0.code(), n, out.ptr(), dt),
        (GType::Q4_0, Dt::F16) => k8.q4_0_half(src.ptr(), n, out.ptr()),
        (other, _) => Err(Error(format!("{} blocks into {dt:?} are not read", other.name()))),
    }
}
