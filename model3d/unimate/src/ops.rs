//! The building blocks: weights read from the checkpoints into a named store (host float32), uploaded as the layers
//! take them; linear layers (the shared diffusion kernels' float32 oneDNN product) and this engine's own kernels.

use std::collections::BTreeMap;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_model3d::{Error, Result};

use crate::ffi::K;

/// A linear layer: weight [n, k] (PyTorch's [out, in]) and an optional bias [n]
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// A host copy of a linear layer (for the few small products done on the CPU)
pub struct HostLin {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl HostLin {
    /// out [m, n] = x [m, k] W^T + b (sums over k in order, float32, as a plain loop)
    pub fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * self.n];
        for r in 0..m {
            let xr = &x[r * self.k..(r + 1) * self.k];
            for o in 0..self.n {
                let w = &self.w[o * self.k..(o + 1) * self.k];
                let mut s = 0f32;
                for (a, b) in xr.iter().zip(w) {
                    s += a * b;
                }
                out[r * self.n + o] = s + self.b.get(o).copied().unwrap_or(0.0);
            }
        }
        out
    }
}

/// Tensors by name on the host, taken as the layers are built
pub struct Store {
    pub map: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    pub gpu: Arc<Gpu>,
}

impl Store {
    pub fn new(gpu: &Arc<Gpu>) -> Store {
        Store { map: BTreeMap::new(), gpu: gpu.clone() }
    }

    /// A tensor (removed from the store) of `shape`
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (s, v) = self.map.remove(name).ok_or_else(|| Error(format!("the checkpoint has no {name}")))?;
        if s != shape {
            return Err(Error(format!("{name} is {s:?}, not {shape:?}")));
        }
        Ok(v)
    }

    pub fn dev(&mut self, name: &str, shape: &[usize]) -> Result<DevBuf> {
        let v = self.take(name, shape)?;
        DevBuf::from_f32(&self.gpu, &v)
    }

    /// `name.weight` [n, k] and, when `bias`, `name.bias` [n]
    pub fn lin(&mut self, name: &str, n: usize, k: usize, bias: bool) -> Result<Lin> {
        Ok(Lin {
            w: self.dev(&format!("{name}.weight"), &[n, k])?,
            b: if bias { Some(self.dev(&format!("{name}.bias"), &[n])?) } else { None },
            n,
            k,
        })
    }

    /// A weight [n, k] cut into row ranges (one Lin each; `sizes` sum to n), with the bias cut alike (an in_proj)
    pub fn lin_split(&mut self, weight: &str, bias: Option<&str>, k: usize, sizes: &[usize]) -> Result<Vec<Lin>> {
        let n: usize = sizes.iter().sum();
        let w = self.take(weight, &[n, k])?;
        let b = bias.map(|b| self.take(b, &[n])).transpose()?;
        let mut at = 0;
        let mut out = Vec::new();
        for &s in sizes {
            out.push(Lin {
                w: DevBuf::from_f32(&self.gpu, &w[at * k..(at + s) * k])?,
                b: b.as_ref().map(|b| DevBuf::from_f32(&self.gpu, &b[at..at + s])).transpose()?,
                n: s,
                k,
            });
            at += s;
        }
        Ok(out)
    }

    pub fn host_lin(&mut self, name: &str, n: usize, k: usize, bias: bool) -> Result<HostLin> {
        Ok(HostLin {
            w: self.take(&format!("{name}.weight"), &[n, k])?,
            b: if bias { self.take(&format!("{name}.bias"), &[n])? } else { Vec::new() },
            n,
            k,
        })
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

    pub fn up(&self, v: &[f32]) -> Result<DevBuf> {
        DevBuf::from_f32(&self.gpu, v)
    }

    /// out [m, l.n] = x [m, l.k] . W^T + b
    pub fn linear(&self, x: &DevBuf, m: usize, l: &Lin, out: &DevBuf) -> Result<()> {
        if x.floats() < m * l.k || out.floats() < m * l.n {
            return Err(Error(format!("linear [{m}, {}] -> [{m}, {}]: buffers of {} and {} floats", l.k, l.n, x.floats(), out.floats())));
        }
        let b = l.b.as_ref().map_or(std::ptr::null(), |b| b.ptr() as *const _);
        self.nsd.linear(x.ptr(), Dt::F32, m, l.k, l.w.ptr(), l.n, b, out.ptr(), Dt::F32)
    }
}

/// A seeded standard-normal draw (splitmix64, Box-Muller): the flow's starting noise
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng(seed);
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
