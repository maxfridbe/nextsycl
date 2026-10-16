//! The engine's working layer: its kernels (ffi.rs) and the shared diffusion kernels as checked calls on device
//! pointers, its linear layers (half weights, float32 biases) and how they are read from a safetensors file.
//!
//! Activations are half where a matrix product reads them and float32 where they accumulate (the residual streams,
//! the latents). The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the
//! calls are not `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::Arc;

use nextsycl_core::{Arena, DevBuf, Error, Gpu, Result};
use nextsycl_diffusion::kernels::{none, Dt, Nsd};
use nextsycl_gguf::safetensors::{f16_to_f32, SafeTensors};

use crate::ffi;

pub type P = *const c_void;
pub type M = *mut c_void;

/// `at` halves into a buffer
pub fn hp(b: &DevBuf, at: usize) -> M {
    b.ptr().cast::<u16>().wrapping_add(at).cast()
}

/// `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// IEEE half from float32, rounded to nearest even (overflow to infinity)
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return sign | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half_m = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half_m & 1 == 1) { half_m + 1 } else { half_m };
        return sign | r as u16;
    }
    let half_m = m >> 13;
    let rem = m & 0x1fff;
    let mut out = ((e2 as u32) << 10) | half_m;
    if rem > 0x1000 || (rem == 0x1000 && half_m & 1 == 1) {
        out += 1;
    }
    sign | out as u16
}

/// A linear layer: half weights [n, k], a float32 bias [n] or none
pub struct Lin {
    pub w: DevBuf,
    pub b: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

impl Lin {
    pub fn bias(&self) -> P {
        self.b.as_ref().map_or(none(), |b| b.ptr() as P)
    }
    pub fn bytes(&self) -> usize {
        self.w.len + self.b.as_ref().map_or(0, |b| b.len)
    }
}

/// A host matrix of half bits [n, k], rows to be picked or reordered before it goes up
pub struct HostMat {
    pub v: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

impl HostMat {
    /// rows `rows` of it, in that order
    pub fn rows(&self, rows: impl IntoIterator<Item = usize>) -> HostMat {
        let mut v = Vec::new();
        let mut n = 0;
        for r in rows {
            v.extend_from_slice(&self.v[r * self.k..(r + 1) * self.k]);
            n += 1;
        }
        HostMat { v, n, k: self.k }
    }
    /// columns [c0, c1) of every row
    pub fn cols(&self, c0: usize, c1: usize) -> HostMat {
        let v = (0..self.n).flat_map(|r| self.v[r * self.k + c0..r * self.k + c1].iter().copied()).collect();
        HostMat { v, n: self.n, k: c1 - c0 }
    }
    /// this matrix's rows, then `o`'s
    pub fn stack(mut self, o: &HostMat) -> HostMat {
        assert_eq!(self.k, o.k);
        self.v.extend_from_slice(&o.v);
        self.n += o.n;
        self
    }
}

/// Reading one model's tensors (a name prefix inside a file) onto a GPU
pub struct Weights<'a> {
    pub st: &'a SafeTensors,
    pub prefix: String,
    pub gpu: Arc<Gpu>,
    /// bytes uploaded so far
    pub bytes: std::cell::Cell<usize>,
}

impl<'a> Weights<'a> {
    pub fn new(st: &'a SafeTensors, prefix: &str, gpu: &Arc<Gpu>) -> Weights<'a> {
        Weights { st, prefix: prefix.to_string(), gpu: gpu.clone(), bytes: 0.into() }
    }
    fn name(&self, n: &str) -> String {
        format!("{}{n}", self.prefix)
    }
    pub fn has(&self, n: &str) -> bool {
        self.st.tensor(&self.name(n)).is_some()
    }
    pub fn shape(&self, n: &str) -> Result<Vec<usize>> {
        Ok(self.st.need(&self.name(n)).map_err(|e| Error(e.0))?.shape.iter().map(|&d| d as usize).collect())
    }
    /// A tensor as float32 on the host
    pub fn host_f32(&self, n: &str) -> Result<Vec<f32>> {
        self.st.f32(&self.name(n)).map_err(|e| Error(e.0))
    }
    /// A matrix (or any tensor, as [dim 0, the rest]) as half bits on the host
    pub fn host_mat(&self, n: &str) -> Result<HostMat> {
        let full = self.name(n);
        let t = self.st.need(&full).map_err(|e| Error(e.0))?;
        let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
        let rows = *shape.first().unwrap_or(&1);
        let k = shape.iter().skip(1).product::<usize>().max(1);
        let v = match t.dtype.as_str() {
            "F16" => self.st.read(t).map_err(|e| Error(e.0))?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            _ => self.host_f32(n)?.into_iter().map(f32_to_f16).collect(),
        };
        Ok(HostMat { v, n: rows, k })
    }
    /// A float32 vector on the GPU
    pub fn vec(&self, n: &str) -> Result<DevBuf> {
        let v = self.host_f32(n)?;
        self.bytes.set(self.bytes.get() + v.len() * 4);
        DevBuf::from_f32(&self.gpu, &v)
    }
    pub fn vec_opt(&self, n: &str) -> Result<Option<DevBuf>> {
        if self.has(n) { self.vec(n).map(Some) } else { Ok(None) }
    }
    /// A host matrix and its bias onto the GPU
    pub fn upload(&self, m: &HostMat, bias: Option<Vec<f32>>) -> Result<Lin> {
        let w = DevBuf::new(&self.gpu, m.v.len() * 2)?;
        // SAFETY: plain u16s viewed as their bytes.
        let bytes = unsafe { std::slice::from_raw_parts(m.v.as_ptr().cast::<u8>(), m.v.len() * 2) };
        w.write(0, bytes)?;
        let b = match bias {
            Some(b) => {
                if b.len() != m.n {
                    return Err(Error(format!("a bias of {} for {} rows", b.len(), m.n)));
                }
                Some(DevBuf::from_f32(&self.gpu, &b)?)
            }
            None => None,
        };
        self.bytes.set(self.bytes.get() + w.len + b.as_ref().map_or(0, |b| b.len));
        Ok(Lin { w, b, n: m.n, k: m.k })
    }
    /// `<n>.weight` and `<n>.bias` (when there is one), as stored
    pub fn lin(&self, n: &str) -> Result<Lin> {
        let m = self.host_mat(&format!("{n}.weight"))?;
        let b = if self.has(&format!("{n}.bias")) { Some(self.host_f32(&format!("{n}.bias"))?) } else { None };
        self.upload(&m, b)
    }
    /// A bias as float32 on the host, or zeros of `n` when the layer has none
    pub fn bias_or_zero(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        if self.has(name) { self.host_f32(name) } else { Ok(vec![0.0; n]) }
    }
}

/// The engine's kernels on one GPU
pub struct Ops {
    pub nsd: Nsd,
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    /// self-attention by the strided flash kernel (the shared kernels' cross-attention path, half precision, the
    /// rows read in place) at every length; false: the shared choice (oneDNN's fused attention under 8,192 rows -
    /// q, k, v copied out per head first - SageAttention's int8 q and k from there)
    pub flash: bool,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ops> {
        Ok(Ops { nsd: Nsd::new(gpu)?, gpu: gpu.clone(), k: ffi::api()?, flash: true })
    }

    fn g(&self) -> M {
        self.gpu.raw()
    }

    /// out [m, n] (out_dt) = x [m, k] half . W^T + b
    pub fn lin(&self, l: &Lin, x: P, m: usize, out: M, out_dt: Dt) -> Result<()> {
        self.nsd.linear(x, Dt::F16, m, l.k, l.w.ptr(), l.n, l.bias(), out, out_dt)
    }

    /// out [m, n] half += x [m, k] half . W^T (no bias)
    pub fn lin_acc(&self, l: &Lin, x: P, m: usize, out: M) -> Result<()> {
        self.nsd.linear_acc(x, Dt::F16, m, l.k, l.w.ptr(), l.n, out)
    }

    /// out [m, c] = (layernorm(x) * w + b) * (1 + scale) + shift (each float32 [c] or null); x and out half or float32
    pub fn ln(&self, x: P, x_half: bool, m: usize, c: usize, eps: f32, w: P, b: P, scale: P, shift: P, out: M, out_half: bool) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.ln)(self.g(), x, x_half as i32, m as i64, c as i64, eps, w.cast(), b.cast(), scale.cast(), shift.cast(), out, out_half as i32) },
                   "layer norm")
    }

    /// Per-head RMS (`rms`) or layer norm of half rows `stride` apart, in place
    pub fn head_norm(&self, x: M, m: usize, h: usize, d: usize, stride: usize, w: P, b: P, eps: f32, rms: bool) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.head_norm)(self.g(), x, m as i64, h as i64, d as i64, stride as i64, w.cast(), b.cast(), eps, rms as i32) }, "head norm")
    }

    pub fn silu(&self, x: P, out: M, n: usize) -> Result<()> {
        // SAFETY: n floats in, n halves out.
        ffi::check(unsafe { (self.k.silu)(self.g(), x.cast(), out, n as i64) }, "silu")
    }
    pub fn to_half(&self, x: P, out: M, n: usize) -> Result<()> {
        // SAFETY: n floats in, n halves out.
        ffi::check(unsafe { (self.k.to_half)(self.g(), x.cast(), out, n as i64) }, "to half")
    }
    pub fn to_float(&self, x: P, out: M, n: usize) -> Result<()> {
        // SAFETY: n halves in, n floats out.
        ffi::check(unsafe { (self.k.to_float)(self.g(), x, out.cast(), n as i64) }, "to float")
    }
    /// x float32 [m, c] += y (half or float32) * gate (float32 [c] or null)
    pub fn add(&self, x: M, y: P, y_half: bool, m: usize, c: usize, gate: P) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.add)(self.g(), x.cast(), y, y_half as i32, m as i64, c as i64, gate.cast()) }, "add")
    }
    /// x += a * y, float32
    pub fn axpy(&self, x: M, y: P, n: usize, a: f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.axpy)(self.g(), x.cast(), y.cast(), n as i64, a) }, "axpy")
    }
    /// out = u + s (c - u), float32
    pub fn cfg(&self, out: M, c: P, u: P, n: usize, s: f32) -> Result<()> {
        // SAFETY: n floats each.
        ffi::check(unsafe { (self.k.cfg)(self.g(), out.cast(), c.cast(), u.cast(), n as i64, s) }, "cfg")
    }
    /// The query embedding of n points (`pts` float32 [n, 3]), or of lattice points (`idx` int32 [n] or the run from
    /// `start`) of an r1^3 lattice at lo + i * step; out half [n, 3 (2 nf + 1)]
    pub fn fourier(&self, pts: P, idx: P, start: usize, n: usize, r1: usize, lo: f32, step: f32, nf: usize, pi: bool, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.fourier)(self.g(), pts.cast(), idx.cast(), start as i64, n as i64, r1 as i64, lo, step, nf as i32, pi as i32, out) },
                   "fourier")
    }
    /// softmax over e experts per row, the k largest: idx int32 [m, k], w float32 [m, k]
    pub fn route(&self, logits: P, m: usize, e: usize, k: usize, idx: M, w: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.route)(self.g(), logits.cast(), m as i64, e as i64, k as i64, idx.cast(), w.cast()) }, "route")
    }
    pub fn gather(&self, x: P, idx: P, n: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.gather)(self.g(), x, idx.cast(), n as i64, c as i64, out) }, "gather")
    }
    pub fn scatter_add(&self, y: P, idx: P, w: P, n: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.scatter_add)(self.g(), y, idx.cast(), w.cast(), n as i64, c as i64, out.cast()) }, "scatter add")
    }
    /// GroupNorm (G groups) of f frames of n pixels x c channels (half, channels last), SiLU after when `silu`;
    /// `stats`: float32 scratch [f * g * 2]
    pub fn group_norm(&self, x: P, f: usize, n: usize, c: usize, g: usize, w: P, b: P, eps: f32, silu: bool, stats: M, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.group_norm)(self.g(), x, f as i64, n as i64, c as i64, g as i64, w.cast(), b.cast(), eps, silu as i32, stats.cast(), out) },
                   "group norm")
    }
    /// out [m, f] = x[:, :f] * gelu(x[:, f:]), half
    pub fn geglu(&self, x: P, m: usize, f: usize, out: M) -> Result<()> {
        // SAFETY: x holds [m, 2f], out [m, f].
        ffi::check(unsafe { (self.k.geglu)(self.g(), x, m as i64, f as i64, out) }, "geglu")
    }
    /// x [n, h, w, c] half zero-padded by (top, bottom, left, right) into out
    pub fn pad2d(&self, x: P, n: usize, h: usize, w: usize, c: usize, pad: (usize, usize, usize, usize), out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.pad2d)(self.g(), x, n as i64, h as i64, w as i64, c as i64, pad.0 as i64, pad.1 as i64, pad.2 as i64, pad.3 as i64, out) },
                   "pad")
    }
    /// out [m, ca + cb] = [a | b], half
    pub fn concat(&self, a: P, ca: usize, b: P, cb: usize, m: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.concat)(self.g(), a, ca as i64, b, cb as i64, m as i64, out) }, "concat")
    }
    /// out [n, 2h, 2w, c] = nearest 2x of x [n, h, w, c], half
    pub fn up2(&self, x: P, n: usize, h: usize, w: usize, c: usize, out: M) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        ffi::check(unsafe { (self.k.up2)(self.g(), x, n as i64, h as i64, w as i64, c as i64, out) }, "up2")
    }
    /// x half += s * y half
    pub fn add_half(&self, x: M, y: P, n: usize, s: f32) -> Result<()> {
        // SAFETY: n halves each.
        ffi::check(unsafe { (self.k.add_half)(self.g(), x, y, n as i64, s) }, "add (half)")
    }

    /// GELU in place over n halves: exact (erf) or the tanh approximation
    pub fn gelu(&self, x: M, n: usize, tanh: bool) -> Result<()> {
        self.nsd.gelu(x, Dt::F16, n, tanh)
    }
    /// Self-attention over s rows of h heads x d (q, k, v rows `stride` apart, half); out [s, h d] half
    pub fn attention(&self, q: P, k: P, v: P, s: usize, h: usize, d: usize, stride: usize, out: M) -> Result<()> {
        if self.flash {
            return self.nsd.attention_qk(q, s, stride, k, v, s, stride, Dt::F16, h, d, out, Dt::F16);
        }
        self.nsd.attention(q, k, v, Dt::F16, s, h, d, stride, out, Dt::F16)
    }

    /// b sequences of s rows each, one after the other (q, k, v rows `stride` apart); out [b s, h d] half
    pub fn attention_batch(&self, q: P, k: P, v: P, b: usize, s: usize, h: usize, d: usize, stride: usize, out: M) -> Result<()> {
        if self.flash {
            for i in 0..b {
                let at = |p: P| p.cast::<u16>().wrapping_add(i * s * stride).cast();
                self.nsd.attention_qk(at(q), s, stride, at(k), at(v), s, stride, Dt::F16, h, d, out.cast::<u16>().wrapping_add(i * s * h * d).cast(), Dt::F16)?;
            }
            return Ok(());
        }
        self.nsd.attention_batch(q, k, v, Dt::F16, b, s, h, d, stride, out, Dt::F16)
    }
    /// sq query rows (qs apart) over skv key / value rows (kvs apart); out [sq, h d] half
    pub fn cross_attention(&self, q: P, sq: usize, qs: usize, k: P, v: P, skv: usize, kvs: usize, h: usize, d: usize, out: M) -> Result<()> {
        self.nsd.attention_qk(q, sq, qs, k, v, skv, kvs, Dt::F16, h, d, out, Dt::F16)
    }
}

/// A temporary of `n` halves from an arena
pub fn halves(a: &Arena, n: usize) -> Result<DevBuf> {
    a.bytes(n * 2)
}

/// Device floats to the host
pub fn read_f32(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0u8; n * 4];
    b.read(0, &mut v)?;
    Ok(v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Device halves to host floats
pub fn read_half(b: &DevBuf, n: usize) -> Result<Vec<f32>> {
    let mut v = vec![0u8; n * 2];
    b.read(0, &mut v)?;
    Ok(v.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
}

/// Host floats into a device buffer at float `at`
pub fn write_f32(b: &DevBuf, at: usize, v: &[f32]) -> Result<()> {
    // SAFETY: plain floats viewed as their bytes.
    b.write(at * 4, unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest_even() {
        for &(f, h) in &[(0.0f32, 0u16), (1.0, 0x3c00), (-2.0, 0xc000), (65504.0, 0x7bff), (1e6, 0x7c00), (5.960_464_5e-8, 1), (0.1, 0x2e66)] {
            assert_eq!(f32_to_f16(f), h, "{f}");
        }
        for i in 0..0x7c00u16 {
            assert_eq!(f32_to_f16(f16_to_f32(i)), i);
        }
    }
}
