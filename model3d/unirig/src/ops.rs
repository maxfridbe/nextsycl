//! The building blocks: weights read from the PyTorch checkpoints onto the GPU, linear layers (the shared diffusion
//! kernels' float32 oneDNN product) and this engine's own kernels, on one GPU.

use std::sync::Arc;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_gguf::pth::Pth;
use nextsycl_model3d::{Error, Result};

use crate::ffi::K;

/// A linear layer: weight [n, k] (PyTorch's [out, in]) and an optional bias [n]
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// A layer norm's weight and bias
pub struct Ln {
    pub w: DevBuf,
    pub b: DevBuf,
}


/// One checkpoint's tensors under a prefix, uploaded as asked
pub struct Weights<'a> {
    pub pth: &'a Pth,
    pub prefix: String,
    pub gpu: &'a Arc<Gpu>,
}

impl Weights<'_> {
    /// A tensor (prefixed) on the host: (shape, values)
    pub fn host(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let full = format!("{}{name}", self.prefix);
        let t = self.pth.need(&full).map_err(|e| Error(e.0))?;
        Ok((t.shape.clone(), self.pth.f32(&full).map_err(|e| Error(e.0))?))
    }

    /// A tensor of `shape` on the GPU (a wrong shape: an error naming it)
    pub fn dev(&self, name: &str, shape: &[usize]) -> Result<DevBuf> {
        let (s, v) = self.host(name)?;
        if s != shape {
            return Err(Error(format!("{}: {}{name} is {s:?}, not {shape:?}", self.pth.path.display(), self.prefix)));
        }
        DevBuf::from_f32(self.gpu, &v)
    }

    /// `name.weight` [n, k] and, when `bias`, `name.bias` [n]
    pub fn lin(&self, name: &str, n: usize, k: usize, bias: bool) -> Result<Lin> {
        Ok(Lin {
            w: self.dev(&format!("{name}.weight"), &[n, k])?,
            b: if bias { Some(self.dev(&format!("{name}.bias"), &[n])?) } else { None },
            n,
            k,
        })
    }

    pub fn ln(&self, name: &str, c: usize) -> Result<Ln> {
        Ok(Ln { w: self.dev(&format!("{name}.weight"), &[c])?, b: self.dev(&format!("{name}.bias"), &[c])? })
    }
}

/// The kernels on the engine's GPU
pub struct Ops {
    pub nsd: Nsd,
    pub k: K,
    pub gpu: Arc<Gpu>,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { nsd: Nsd::new(gpu)?, k: K::new(gpu)?, gpu: gpu.clone() })
    }

    /// A buffer of n floats
    pub fn buf(&self, n: usize) -> Result<DevBuf> {
        DevBuf::f32(&self.gpu, n)
    }

    /// out [m, l.n] = x [m, l.k] . W^T + b
    pub fn linear(&self, x: &DevBuf, m: usize, l: &Lin, out: &DevBuf) -> Result<()> {
        if x.floats() < m * l.k || out.floats() < m * l.n {
            return Err(Error(format!("linear [{m}, {}] -> [{m}, {}]: buffers of {} and {} floats", l.k, l.n, x.floats(), out.floats())));
        }
        let b = l.b.as_ref().map_or(std::ptr::null(), |b| b.ptr() as *const _);
        self.nsd.linear(x.ptr(), Dt::F32, m, l.k, l.w.ptr(), l.n, b, out.ptr(), Dt::F32)
    }

    /// out [m, c] = layer norm of x [m, c] (affine `n` when given), eps 1e-5
    pub fn ln(&self, x: &DevBuf, m: usize, c: usize, n: Option<&Ln>, out: &DevBuf) -> Result<()> {
        self.k.ln(x, c, m, c, 1e-5, n.map(|n| &n.w), n.map(|n| &n.b), out)
    }
}

/// A seeded standard-normal draw (splitmix64, Box-Muller)
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (r.unit().max(1e-300), r.unit());
        let m = (-2.0 * u.ln()).sqrt();
        out.push((m * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((m * (std::f64::consts::TAU * v).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// splitmix64
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Bytes of an int32 slice (for an upload)
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// An int32 buffer holding `v`
pub fn dev_i32(gpu: &Arc<Gpu>, v: &[i32]) -> Result<DevBuf> {
    let b = DevBuf::new(gpu, v.len().max(1) * 4)?;
    if !v.is_empty() {
        b.write(0, &i32_bytes(v))?;
    }
    Ok(b)
}
