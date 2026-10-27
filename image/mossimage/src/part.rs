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

impl Part {
    /// Bytes the matrix takes on the card as stored
    pub fn stored(&self) -> u64 {
        let (blk, bytes) = match self.qtype {
            0 => (1, 4),
            1 | 30 => (1, 2),
            8 => (32, 34),
            12 => (256, 144),
            13 => (256, 176),
            _ => (256, 210),
        };
        (self.rows * self.k / blk * bytes) as u64
    }

    /// out [m, rows] = x [m, k] (half) . W^T with W expanded whole into `wide` (or `kept`, already expanded) (at least rows x k halves) and one
    /// product straight into `out`: the shared quantized product expands a slice of rows at a time and moves its
    /// results into their columns, which on the DiT's matrices costs a third again on the product's own time
    #[allow(clippy::too_many_arguments)]
    pub fn apply_whole(&self, nsd: &Nsd, kept: Option<&DevBuf>, wide: &DevBuf, x: *const std::ffi::c_void, m: usize, out: *mut std::ffi::c_void, out_dt: Dt) -> Result<()> {
        let w = match kept {
            Some(h) => h.ptr(),
            None => {
                nsd.dequant(self.w.ptr(), self.qtype, self.rows * self.k, wide.ptr(), Dt::F16)?;
                wide.ptr()
            }
        };
        nsd.linear(x, Dt::F16, m, self.k, w, self.rows, none(), out, out_dt)
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

/// A float32 as half's bits (round to nearest even; past half's range, infinity)
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // a subnormal: the implicit bit joins the mantissa, shifted right
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rem > mid || (rem == mid && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
    sign | (half + up as u32) as u16
}

#[cfg(test)]
mod tests {
    use super::f16_bits;

    #[test]
    fn half_bits_match_known_values() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e9), 0x7c00);
        assert_eq!(f16_bits(f32::from_bits(0x3880_0000)), 0x0400); // 2^-14, half's smallest normal
        assert_eq!(f16_bits(f32::from_bits(0x3800_0000)), 0x0200); // 2^-15: a subnormal
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00); // the tie goes to even
        assert_eq!(f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn half_bits_round_trip_through_the_reader() {
        for bits in [0x0001u16, 0x03ff, 0x0400, 0x3555, 0x7bff, 0x8400, 0xb800] {
            let x = nextsycl_gguf::safetensors::f16_to_f32(bits);
            assert_eq!(f16_bits(x), bits);
        }
    }
}
