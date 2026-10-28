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
    /// the weight as bfloat16 (device memory), for the silo's decode product: only when every value is exactly representable
    pub bf: Option<DevBuf>,
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
    /// whether the silo's bfloat16 decode product exists (the weights then keep a bf16 copy where they can)
    pub silo: bool,
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
            bf: None,
        })
    }

    pub fn ln(&self, name: &str, c: usize) -> Result<Ln> {
        Ok(Ln { w: self.dev(&format!("{name}.weight"), &[c])?, b: self.dev(&format!("{name}.bias"), &[c])? })
    }

    /// The weights `names` (each `.weight` [n_i, k]) stacked row-wise into one layer [sum n_i, k] (no bias): one product for
    /// what the model computes as several
    pub fn lin_cat(&self, names: &[&str], ns: &[usize], k: usize) -> Result<Lin> {
        let mut all = Vec::new();
        for (name, &n) in names.iter().zip(ns) {
            let (s, v) = self.host(&format!("{name}.weight"))?;
            if s != [n, k] {
                return Err(Error(format!("{}: {}{name}.weight is {s:?}, not {:?}", self.pth.path.display(), self.prefix, [n, k])));
            }
            all.extend(v);
        }
        Ok(Lin { w: DevBuf::from_f32(self.gpu, &all)?, b: None, n: ns.iter().sum(), k, bf: None })
    }

    /// `lin_cat` that also keeps the weight as bfloat16 (TokenRig stores its decoder so, so it is exact; any value that is not
    /// leaves no bf16 copy) when the silo has the decode product
    pub fn lin_cat_bf16(&self, names: &[&str], ns: &[usize], k: usize) -> Result<Lin> {
        let mut all = Vec::new();
        for (name, &n) in names.iter().zip(ns) {
            let (s, v) = self.host(&format!("{name}.weight"))?;
            if s != [n, k] {
                return Err(Error(format!("{}: {}{name}.weight is {s:?}, not {:?}", self.pth.path.display(), self.prefix, [n, k])));
            }
            all.extend(v);
        }
        let bf = if self.silo { bf16_of(self.gpu, &all)? } else { None };
        Ok(Lin { w: DevBuf::from_f32(self.gpu, &all)?, b: None, n: ns.iter().sum(), k, bf })
    }

    /// `name.weight` [n, k] without a bias, with a bfloat16 copy where the silo can use one (`lin_cat_bf16`)
    pub fn lin_bf16(&self, name: &str, n: usize, k: usize) -> Result<Lin> {
        self.lin_cat_bf16(&[name], &[n], k)
    }

    /// A one-dimensional tensor on the GPU (`name`, as given: a norm's weight)
    pub fn vec(&self, name: &str, n: usize) -> Result<DevBuf> {
        self.dev(name, &[n])
    }
}

/// The values as bfloat16 on the GPU, when each is exactly a bfloat16 (its low 16 bits zero)
fn bf16_of(gpu: &Arc<Gpu>, v: &[f32]) -> Result<Option<DevBuf>> {
    if v.iter().any(|x| x.to_bits() & 0xffff != 0) {
        return Ok(None);
    }
    let bytes: Vec<u8> = v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let b = DevBuf::new(gpu, bytes.len())?;
    b.write(0, &bytes)?;
    Ok(Some(b))
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
        if let (Some(bf), None) = (&l.bf, &l.b) {
            if self.k.gemv(x, m, bf, l.n, l.k, out)? {
                return Ok(());
            }
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
