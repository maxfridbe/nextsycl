//! Element types, and the 16-bit float conversions the host needs (the device does its own).

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I32,
    I64,
    F64,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::I8 | DType::U8 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    /// The safetensors header's name for the type.
    pub fn parse(s: &str) -> Result<DType> {
        Ok(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I8" => DType::I8,
            "U8" => DType::U8,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "F64" => DType::F64,
            other => return Err(Error(format!("unsupported tensor type {other}"))),
        })
    }

    /// The kernel library's code for a floating-point type (`H3S_F32` ...).
    pub fn kernel_code(self) -> Result<i32> {
        match self {
            DType::F32 => Ok(crate::sys::F32),
            DType::F16 => Ok(crate::sys::F16),
            DType::BF16 => Ok(crate::sys::BF16),
            other => Err(Error(format!("{other:?} is not a floating-point tensor type"))),
        }
    }
}

/// bfloat16 is the top half of a float32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round to nearest, ties to even (what the device and PyTorch do).
pub fn f32_to_bf16(f: f32) -> u16 {
    let mut u = f.to_bits();
    if u & 0x7f80_0000 == 0x7f80_0000 {
        return (u >> 16) as u16; // inf / nan: truncate
    }
    u = u.wrapping_add(0x7fff + ((u >> 16) & 1));
    (u >> 16) as u16
}

/// IEEE half -> float32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalize
            let shift = m.leading_zeros() - 21;
            let m = (m << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (31, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// float32 -> IEEE half, round to nearest even (overflow to infinity, small values to subnormals or zero).
pub fn f32_to_f16(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32;
    let man = u & 0x7f_ffff;
    if exp == 0xff {
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
        // subnormal: the implicit bit joins the mantissa, shifted into place with rounding
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut h = m >> shift;
        if rest > half || (rest == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1; // a carry into the exponent is the right rounding (up to infinity)
    }
    sign | h as u16
}

/// Bytes of a tensor as float32 values (F32, F16 and BF16 only).
pub fn bytes_to_f32(bytes: &[u8], dt: DType) -> Result<Vec<f32>> {
    Ok(match dt {
        DType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        DType::BF16 => bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        DType::F16 => bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(Error(format!("{other:?} is not a floating-point tensor type"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.140625, -1234.0, 1e-20] {
            let r = bf16_to_f32(f32_to_bf16(v));
            assert!((r - v).abs() <= v.abs() * 0.004, "{v} -> {r}");
        }
        // ties go to even
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn f16_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0), 0.0);
    }
}
