//! Weights from the official safetensors (float32): the DiT's large matrices quantized to Q8_0 as they load (ggml's
//! quantize_row_q8_0_ref: blocks of 32 along a row over an fp16 scale amax / 127, values rounded half away from zero) -
//! the silo multiplies them as int8 x its half scales -, everything else float32.

use std::path::Path;

use nextsycl_core::{DevBuf, Error, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;

use crate::ffi::{Act, Mm, K, T};

pub fn se(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A safetensors file of float tensors
pub struct St(pub SafeTensors);

impl St {
    pub fn open(path: &Path) -> Result<St> {
        Ok(St(SafeTensors::open(path).map_err(se)?))
    }

    pub fn has(&self, name: &str) -> bool {
        self.0.tensor(name).is_some()
    }

    /// A tensor as float32 and its shape
    pub fn get(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self.0.need(name).map_err(se)?;
        let shape = t.shape.iter().map(|d| *d as usize).collect();
        Ok((self.0.f32(name).map_err(se)?, shape))
    }

    pub fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.get(name)?.0)
    }
}

/// IEEE half from float32, rounded to nearest even (as PyTorch's .half())
pub fn f32_to_f16(v: f32) -> u16 {
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

/// Q8_0 blocks of `v` (a multiple of 32 values): ggml's quantize_row_q8_0_ref
pub fn q8_0(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() / 32 * 34];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(16);
    let blocks = v.len() / 32;
    let per = blocks.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * 34).enumerate() {
            let src = &v[ci * per * 32..];
            s.spawn(move || {
                for (b, o) in chunk.chunks_exact_mut(34).enumerate() {
                    let x = &src[b * 32..b * 32 + 32];
                    let amax = x.iter().fold(0f32, |m, a| m.max(a.abs()));
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    o[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                    for (j, a) in x.iter().enumerate() {
                        o[2 + j] = ((a * id).round() as i32).clamp(-128, 127) as i8 as u8;
                    }
                }
            });
        }
    });
    out
}

/// How a matrix is kept on the card
pub enum Wt {
    /// Q8_0 split for the silo's products: int8 [n, k] and half scales [k / 32, n]
    I8 { w: DevBuf, s: DevBuf },
    /// Q8_0 blocks: the modulations (expanded to float32 for their products), or every matrix without the silo
    Blocks(DevBuf),
    F32(DevBuf),
}

/// A linear layer: out [m, n] = x [m, k] W^T + bias
pub struct Lin {
    pub w: Wt,
    pub bias: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// What a product does with its result
#[derive(Clone, Copy)]
pub enum Post {
    /// written as half
    Half,
    /// GELU (tanh), written as half
    Gelu,
    /// out (float32) += gate (float32 [n]) x the product: the gated residual
    Gate(*const f32),
}

/// How to keep a matrix
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Q8_0, split for the silo when it is there
    Q8,
    /// Q8_0 blocks (float32 products)
    Q8Blocks,
    F32,
}

impl Lin {
    /// Matrices `names` stacked by rows (and their biases), rows reordered by `perm` (new row r reads old row perm(r))
    pub fn load(st: &St, k8: &K, names: &[String], biases: &[String], perm: Option<&dyn Fn(usize) -> usize>, keep: Keep) -> Result<Lin> {
        let gpu = &k8.gpu;
        let mut v = Vec::new();
        let (mut n, mut k) = (0usize, 0usize);
        for name in names {
            let (x, shape) = st.get(name)?;
            let cols: usize = shape.iter().skip(1).product();
            if k != 0 && cols != k {
                return Err(Error(format!("{name}: {cols} columns beside {k}")));
            }
            k = cols;
            n += shape.first().copied().unwrap_or(0);
            v.extend(x);
        }
        let mut b = Vec::new();
        for name in biases {
            b.extend(st.vec(name)?);
        }
        if !biases.is_empty() && b.len() != n {
            return Err(Error(format!("{}: a bias of {} for {n} rows", biases[0], b.len())));
        }
        if let Some(p) = perm {
            let old = std::mem::take(&mut v);
            v = (0..n).flat_map(|r| old[p(r) * k..(p(r) + 1) * k].iter().copied()).collect();
            if !b.is_empty() {
                b = (0..n).map(|r| b[p(r)]).collect();
            }
        }
        let bias = if b.is_empty() { None } else { Some(DevBuf::from_f32(gpu, &b)?) };
        let w = match keep {
            Keep::F32 => Wt::F32(DevBuf::from_f32(gpu, &v)?),
            Keep::Q8 | Keep::Q8Blocks => {
                if !k.is_multiple_of(32) {
                    return Err(Error(format!("{}: {k} columns are not whole Q8_0 blocks", names[0])));
                }
                let q = q8_0(&v);
                drop(v);
                let blocks = DevBuf::new(gpu, q.len())?;
                blocks.write(0, &q)?;
                if keep == Keep::Q8 && k8.silo().is_some() {
                    let w = DevBuf::new(gpu, n * k)?;
                    let s = DevBuf::new(gpu, n * k / 32 * 2)?;
                    k8.q8_split(blocks.ptr(), n, k, w.ptr(), s.ptr())?;
                    gpu.sync()?;
                    Wt::I8 { w, s }
                } else {
                    Wt::Blocks(blocks)
                }
            }
        };
        Ok(Lin { w, bias, n, k })
    }

    /// bytes on the card
    pub fn bytes(&self) -> usize {
        let w = match &self.w {
            Wt::I8 { w, s } => w.len + s.len,
            Wt::Blocks(b) | Wt::F32(b) => b.len,
        };
        w + self.bias.as_ref().map_or(0, |b| b.len)
    }

    fn bias_ptr(&self) -> *const f32 {
        self.bias.as_ref().map_or(std::ptr::null(), |b| b.fp() as *const f32)
    }

    /// out [m, n] float32 = x [m, k] float32 W^T + bias, a Q8_0 matrix expanded into `scratch` (n x k floats at least)
    pub fn apply_f32(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void) -> Result<()> {
        let w = match &self.w {
            Wt::Blocks(b) => {
                if scratch.len < self.n * self.k * 4 {
                    return Err(Error("the float32 scratch is too small".into()));
                }
                k8.q8_0_float(b.ptr(), self.n * self.k, scratch.fp())?;
                scratch.ptr()
            }
            Wt::F32(b) => b.ptr(),
            Wt::I8 { .. } => return Err(Error("a split matrix has no float32 product here".into())),
        };
        let bias = self.bias.as_ref().map_or(none(), |b| b.ptr() as *const _);
        nsd.linear(x, Dt::F32, m, self.k, w, self.n, bias, out, Dt::F32)
    }

    /// The DiT's product: x half [m, k] (rows lda apart) -> out (rows ldo apart) as `post` says. On the silo straight
    /// from the int8 matrix; without it the blocks expanded to half in `scratch` (n x k halves), then oneDNN.
    #[allow(clippy::too_many_arguments)]
    pub fn mm(&self, nsd: &Nsd, k8: &K, scratch: &DevBuf, tmp: &DevBuf, x: *const std::ffi::c_void, lda: usize, m: usize, out: *mut std::ffi::c_void,
              ldo: usize, post: Post) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let (out_dt, p, gate) = match post {
            Post::Half => (1, 0, std::ptr::null()),
            Post::Gelu => (1, 1, std::ptr::null()),
            Post::Gate(g) => (0, 2, g),
        };
        match &self.w {
            Wt::I8 { w, s } => k8.mm(&Mm {
                x,
                lda: lda as i64,
                x_bs: 0,
                b: 1,
                m: m as i64,
                k: self.k as i64,
                k0: 0,
                kn: self.k as i64,
                n: self.n as i64,
                w: w.ptr() as *const i8,
                s: s.ptr(),
                bias: self.bias_ptr(),
                out,
                ldo: ldo as i64,
                o_bs: 0,
                out_dt,
                post: p,
                gate,
                wscale: std::ptr::null(),
                xq: std::ptr::null_mut(),
                xs: std::ptr::null_mut(),
            }),
            Wt::Blocks(b) => {
                // the shared path: the matrix in half, the rows contiguous (lda = k, ldo = n here)
                if lda != self.k || ldo != self.n {
                    return Err(Error("the shared products take contiguous rows".into()));
                }
                if scratch.len < self.n * self.k * 2 || tmp.len < m * self.n * 4 {
                    return Err(Error("the product's scratch is too small".into()));
                }
                nsd.dequant(b.ptr(), nextsycl_gguf::GType::Q8_0.code(), self.n * self.k, scratch.ptr(), Dt::F16)?;
                match post {
                    Post::Half | Post::Gelu => {
                        k8.linear_f32(x, m, self.k, scratch.ptr(), 0, self.n, self.bias_ptr(), tmp.fp(), 0)?;
                        if let Post::Gelu = post {
                            k8.act(tmp.ptr(), T::F32, m * self.n, Act::GeluTanh)?;
                        }
                        k8.to_half(tmp.fp(), out, m * self.n, 1.0)
                    }
                    Post::Gate(g) => {
                        k8.linear_f32(x, m, self.k, scratch.ptr(), 0, self.n, self.bias_ptr(), tmp.fp(), 0)?;
                        nsd.gate_add(out, Dt::F32, m, self.n, tmp.ptr(), Dt::F32, zero_rows(&k8.gpu, m)?.ptr(), g.cast())
                    }
                }
            }
            Wt::F32(_) => Err(Error("a float32 matrix in a half product".into())),
        }
    }
}

/// Row indexes all zero (the gate kernels' row table: every row the same gate)
fn zero_rows(gpu: &std::sync::Arc<nextsycl_core::Gpu>, m: usize) -> Result<DevBuf> {
    let z = DevBuf::new(gpu, m * 4)?;
    z.fill(0)?;
    Ok(z)
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_rounds_as_pytorch() {
        for (v, h) in [(1.0f32, 0x3c00u16), (-2.0, 0xc000), (65504.0, 0x7bff), (1e-8, 0), (0.1, 0x2e66), (3.140625, 0x4248), (6.1035156e-5, 0x0400),
                       (1e6, 0x7c00), (2.9802322e-8, 0x0000), (5.9604645e-8, 0x0001)] {
            assert_eq!(super::f32_to_f16(v), h, "{v}");
        }
    }

    #[test]
    fn q8_0_blocks_are_ggmls() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.37).collect();
        let q = super::q8_0(&v);
        assert_eq!(q.len(), 68);
        let d = nextsycl_gguf::safetensors::f16_to_f32(u16::from_le_bytes([q[0], q[1]]));
        assert!((d - 20.0 * 0.37 / 127.0).abs() < 1e-5);
        assert_eq!(q[2] as i8, -127);
        // round half away from zero
        assert_eq!(super::q8_0(&[[2.5f32; 1].as_slice(), &[127.0; 31]].concat())[2] as i8, 3);
    }
}
