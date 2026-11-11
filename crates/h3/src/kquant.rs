//! Q3_K matrices (MiniMax-H3-Image's Q3_K_M file) as Q6_K, exactly.
//!
//! A Q3_K block of 256 values holds a half-precision `d`, sixteen 6-bit scales `s` (one per 16 values, used as
//! `s - 32`, so -32 .. 31) and 3-bit values `q` (two low bits and a high-bit mask, used as -4 .. 3): a value is
//! `d * (s - 32) * q`. A Q6_K block holds `d`, sixteen int8 scales (one per 16 values) and 6-bit values used as
//! `q - 32` (-32 .. 31): `d * scale * (q - 32)`. So every Q3_K block is a Q6_K block with the same `d`, `scale = s - 32`
//! and `q6 = q3 + 32`: the same 256 values, computed the same way (`d * scale` first, then times the value, in float32),
//! at 210 bytes instead of 110. The shared kernels expand Q6_K; Q3_K they do not read.
//!
//! (ggml's block layouts: `block_q3_K` = hmask[32], qs[64], scales[12], d; `block_q6_K` = ql[128], qh[64], scales[16]
//! (int8), d.)

use crate::{Error, Result};

pub const Q3K_BYTES: usize = 110;
pub const Q6K_BYTES: usize = 210;
const QK: usize = 256;

/// One Q3_K block's sixteen scales, unpacked from 12 bytes and centred (ggml's `kmask` shuffle): -32 .. 31
pub fn q3k_scales(packed: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let word = |i: usize| u32::from_le_bytes([packed[4 * i], packed[4 * i + 1], packed[4 * i + 2], packed[4 * i + 3]]);
    let (a0, a1, tmp) = (word(0), word(1), word(2));
    let aux = [
        (a0 & KMASK2) | ((tmp & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut s = [0i8; 16];
    for (i, w) in aux.iter().enumerate() {
        for (k, b) in w.to_le_bytes().iter().enumerate() {
            s[4 * i + k] = *b as i8 - 32;
        }
    }
    s
}

/// One Q3_K block's 256 values as integers -4 .. 3 (before the scales), in order
pub fn q3k_values(block: &[u8]) -> [i8; QK] {
    let (hmask, qs) = (&block[0..32], &block[32..96]);
    let mut v = [0i8; QK];
    for c in 0..2 {
        for j in 0..4 {
            let bit = 1u8 << (4 * c + j);
            for half in 0..2 {
                for l in 0..16 {
                    let i = half * 16 + l;
                    let low = ((qs[32 * c + i] >> (2 * j)) & 3) as i8;
                    v[128 * c + 32 * j + i] = low - if hmask[i] & bit != 0 { 0 } else { 4 };
                }
            }
        }
    }
    v
}

/// Q3_K blocks -> the same values as Q6_K blocks
pub fn q3k_to_q6k(src: &[u8]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(Q3K_BYTES) {
        return Err(Error(format!("{} bytes of Q3_K blocks: not a multiple of {Q3K_BYTES}", src.len())));
    }
    let n = src.len() / Q3K_BYTES;
    let mut out = vec![0u8; n * Q6K_BYTES];
    for (b, o) in src.chunks_exact(Q3K_BYTES).zip(out.chunks_exact_mut(Q6K_BYTES)) {
        let q = q3k_values(b);
        let s = q3k_scales(&b[96..108]);
        let q6 = |e: usize| (q[e] + 32) as u8;
        let (ql, rest) = o.split_at_mut(128);
        let (qh, rest) = rest.split_at_mut(64);
        let (sc, d) = rest.split_at_mut(16);
        for c in 0..2 {
            for l in 0..32 {
                let (a, bb, cc, dd) = (q6(128 * c + l), q6(128 * c + 32 + l), q6(128 * c + 64 + l), q6(128 * c + 96 + l));
                ql[64 * c + l] = (a & 0xf) | ((cc & 0xf) << 4);
                ql[64 * c + 32 + l] = (bb & 0xf) | ((dd & 0xf) << 4);
                qh[32 * c + l] = (a >> 4) | ((bb >> 4) << 2) | ((cc >> 4) << 4) | ((dd >> 4) << 6);
            }
        }
        for (x, v) in sc.iter_mut().zip(s) {
            *x = v as u8;
        }
        d.copy_from_slice(&b[108..110]);
    }
    Ok(out)
}

/// Host expansion of Q3_K blocks (ggml's `dequantize_row_q3_K`), for checks
pub fn dequant_q3k(src: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() / Q3K_BYTES * QK);
    for b in src.chunks_exact(Q3K_BYTES) {
        let d = crate::dtype::f16_to_f32(u16::from_le_bytes([b[108], b[109]]));
        let (q, s) = (q3k_values(b), q3k_scales(&b[96..108]));
        out.extend((0..QK).map(|e| (d * s[e / 16] as f32) * q[e] as f32));
    }
    out
}

/// Host expansion of Q6_K blocks (ggml's `dequantize_row_q6_K`), for checks
pub fn dequant_q6k(src: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() / Q6K_BYTES * QK);
    for b in src.chunks_exact(Q6K_BYTES) {
        let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
        let d = crate::dtype::f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let mut y = [0f32; QK];
        for c in 0..2 {
            let (ql, qh, sc) = (&ql[64 * c..], &qh[32 * c..], &sc[8 * c..]);
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[128 * c + l] = d * sc[is] as i8 as f32 * q1 as f32;
                y[128 * c + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[128 * c + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[128 * c + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
        }
        out.extend_from_slice(&y);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(n: usize, seed: u64) -> Vec<u8> {
        let mut r = crate::rng::Rng::new(seed);
        let mut v: Vec<u8> = (0..n * Q3K_BYTES).map(|_| (r.next_u64() >> 24) as u8).collect();
        // finite, ordinary d values (a half between 2^-10 and 2^-3, either sign)
        for b in v.chunks_exact_mut(Q3K_BYTES) {
            let d = crate::dtype::f32_to_f16((b[0] as f32 + 1.0) / 2048.0 * if b[1] & 1 == 1 { -1.0 } else { 1.0 });
            b[108..110].copy_from_slice(&d.to_le_bytes());
        }
        v
    }

    #[test]
    fn the_scales_unpack_as_ggml_s_shuffle() {
        // byte j < 8: low nibble of packed[j] (j < 8) or high nibble of packed[j - 8], top bits from packed[8 + j % 4]
        let packed: Vec<u8> = (0..12u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
        let s = q3k_scales(&packed);
        for (j, v) in s.iter().enumerate() {
            let low = if j < 8 { packed[j] & 0xf } else { packed[j - 8] >> 4 };
            let high = (packed[8 + j % 4] >> (2 * (j / 4))) & 3;
            assert_eq!(*v as i32, (low | (high << 4)) as i32 - 32, "scale {j}");
        }
    }

    #[test]
    fn a_hand_made_block_reads_as_written() {
        // every scale 1 (stored 33), every low pair 3, the high bit set on value 0 only -> value 0: 3, the rest 3 - 4
        let mut b = vec![0u8; Q3K_BYTES];
        b[0] = 1; // hmask[0] bit 0: value 0 (chunk 0, j 0, l 0)
        b[32..96].fill(0xff);
        // scale 33 everywhere: low nibble 1, high bits 2 (33 = 0b10_0001)
        b[96..104].fill(0x11);
        b[104..108].fill(0xaa);
        b[108..110].copy_from_slice(&crate::dtype::f32_to_f16(0.5).to_le_bytes());
        let v = dequant_q3k(&b);
        assert_eq!(v[0], 0.5 * 3.0);
        assert!(v[1..].iter().all(|x| *x == -0.5));
        assert_eq!(dequant_q6k(&q3k_to_q6k(&b).unwrap()), v);
    }

    #[test]
    fn q6k_holds_every_q3k_value_bit_for_bit() {
        let src = blocks(64, 7);
        let a = dequant_q3k(&src);
        let b = dequant_q6k(&q3k_to_q6k(&src).unwrap());
        assert_eq!(a.len(), 64 * 256);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        // the extremes are reached: -32 x -4 = 128 and 31 x -4 = -124 are in range of the int8 scale and 6-bit value
        assert!(a.iter().any(|x| *x != 0.0));
        assert!(q3k_to_q6k(&src[..100]).is_err());
    }
}
