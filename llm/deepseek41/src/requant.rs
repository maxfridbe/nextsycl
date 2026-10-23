//! Q8_0 (Q6_K, F16) -> Q4_K on the host (NS_DS41_DENSE_Q4K=1): the Unsloth file keeps attention and the shared expert in Q8_0 (8.5 bits a
//! weight), a decode token reads all of them; in q4_K (4.5 bits) they are half the bytes and the engine's fused q4_K kernels
//! (q_a | kv, the grouped o_a in one launch) apply. The quantizer is ggml's quantize_row_q4_K_ref (no importance matrix): per
//! 32 values a scale and a minimum found by a least-squares search over 21 candidate scales, the 8 scales and minimums of a
//! block in 6 bits against the block's largest.

/// An f16's bits as a float
fn f16_to_f32(h: u16) -> f32 {
    nextsycl_gguf::safetensors::f16_to_f32(h)
}

/// A float as an f16's bits, rounded to nearest even
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x7F_FFFF;
    if e == 0xFF {
        return sign | 0x7C00 | if m != 0 { 0x200 } else { 0 };
    }
    let e = e - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let (h, rest) = (m >> shift, m & ((1 << shift) - 1));
        let half = 1u32 << (shift - 1);
        let mut h = h;
        if rest > half || (rest == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let (mut h, rest) = (((e as u32) << 10) | (m >> 13), m & 0x1FFF);
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn nearest_int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// ggml's make_qkx2_quants: n values to L in 0..=nmax as `scale * L + min` (min <= 0), the error weighted; returns
/// (scale, -min)
fn make_qkx2_quants(n: usize, nmax: i32, x: &[f32], weights: &[f32], l: &mut [u8], laux: &mut [u8], rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32) {
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let diff = scale * q as f32 + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let (w, qf) = (weights[i], q as f32);
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                mad += weights[i] * diff * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml's get_scale_min_k4
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// One 256-value block as a 144-byte q4_K block
fn quantize_block(x: &[f32], out: &mut [u8]) {
    let (mut l, mut laux) = ([0u8; 256], [0u8; 32]);
    let (mut scales, mut mins) = ([0f32; 8], [0f32; 8]);
    let mut weights = [0f32; 32];
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..8 {
        let xs = &x[32 * j..32 * j + 32];
        let sum_x2: f32 = xs.iter().map(|v| v * v).sum();
        let av_x = (sum_x2 / 32.0).sqrt();
        for i in 0..32 {
            weights[i] = av_x + xs[i].abs();
        }
        let (s, m) = make_qkx2_quants(32, 15, xs, &weights, &mut l[32 * j..32 * j + 32], &mut laux, -1.0, 0.1, 20);
        scales[j] = s;
        mins[j] = m;
        max_scale = max_scale.max(s);
        max_min = max_min.max(m);
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc = [0u8; 12];
    for j in 0..8 {
        let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
        let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    let d = f32_to_f16(max_scale / 63.0);
    let dmin = f32_to_f16(max_min / 63.0);
    out[0..2].copy_from_slice(&d.to_le_bytes());
    out[2..4].copy_from_slice(&dmin.to_le_bytes());
    out[4..16].copy_from_slice(&sc);
    // the levels again against the stored (6-bit, f16) scales
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, &sc);
        let dd = f16_to_f32(d) * s as f32;
        if dd == 0.0 {
            continue;
        }
        let dm = f16_to_f32(dmin) * m as f32;
        for ii in 0..32 {
            l[32 * j + ii] = nearest_int((x[32 * j + ii] + dm) / dd).clamp(0, 15) as u8;
        }
    }
    for j in (0..256).step_by(64) {
        for k in 0..32 {
            out[16 + j / 2 + k] = l[j + k] | (l[j + k + 32] << 4);
        }
    }
}

/// What to_q4k reads
#[derive(Clone, Copy, PartialEq)]
pub enum From {
    Q8_0,
    Q6K,
    /// the XQ file's Engram projection
    F16,
}

/// A 256-value block of Q8_0 (8 x 34 bytes: an f16 scale, 32 int8), Q6_K (210 bytes) or F16 (512 bytes) as floats
fn dequant_block(from: From, b: &[u8], x: &mut [f32; 256]) {
    if from == From::F16 {
        for (i, v) in x.iter_mut().enumerate() {
            *v = f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
        }
        return;
    }
    if from == From::Q8_0 {
        for s in 0..8 {
            let blk = &b[s * 34..s * 34 + 34];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for i in 0..32 {
                x[s * 32 + i] = d * (blk[2 + i] as i8) as f32;
            }
        }
        return;
    }
    // ql[128], qh[64], scales[16] (int8), d
    let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
    for n in 0..2 {
        let (ql, qh, sc) = (&b[n * 64..], &b[128 + n * 32..], &b[192 + n * 8..]);
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            x[n * 128 + l] = d * (sc[is] as i8) as f32 * q1 as f32;
            x[n * 128 + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            x[n * 128 + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            x[n * 128 + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Q8_0, Q6_K or F16 blocks -> q4_K blocks, `n` values (a multiple of 256), on all cores
pub fn to_q4k(src: &[u8], n: usize, from: From) -> Vec<u8> {
    let sb = match from {
        From::Q8_0 => 272,
        From::Q6K => 210,
        From::F16 => 512,
    };
    assert!(n.is_multiple_of(256) && src.len() == n / 256 * sb);
    let nb = n / 256;
    let mut out = vec![0u8; nb * 144];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).min(32);
    let per = nb.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ti, chunk) in out.chunks_mut(per * 144).enumerate() {
            sc.spawn(move || {
                let mut x = [0f32; 256];
                for (bi, ob) in chunk.chunks_mut(144).enumerate() {
                    let b = ti * per + bi;
                    dequant_block(from, &src[b * sb..(b + 1) * sb], &mut x);
                    quantize_block(&x, ob);
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// q4_K blocks dequantized as ggml's dequantize_row_q4_K
    fn deq(b: &[u8]) -> Vec<f32> {
        let (d, dmin) = (f16_to_f32(u16::from_le_bytes([b[0], b[1]])), f16_to_f32(u16::from_le_bytes([b[2], b[3]])));
        let mut y = vec![0f32; 256];
        for j in 0..4 {
            let (s0, m0) = scale_min_k4(2 * j, &b[4..16]);
            let (s1, m1) = scale_min_k4(2 * j + 1, &b[4..16]);
            for l in 0..32 {
                let q = b[16 + 32 * j + l];
                y[64 * j + l] = d * s0 as f32 * (q & 0xF) as f32 - dmin * m0 as f32;
                y[64 * j + 32 + l] = d * s1 as f32 * (q >> 4) as f32 - dmin * m1 as f32;
            }
        }
        y
    }

    #[test]
    fn q4k_of_q8_0_is_close() {
        // smooth + noisy data through Q8_0 and back through q4_K: the error of 4-bit levels (a quarter of the range)
        let n = 512;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0 * (1.0 + (i / 32) as f32 * 0.1)).collect();
        let mut src = Vec::new();
        for blk in x.chunks(32) {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            src.extend(f32_to_f16(d).to_le_bytes());
            for v in blk {
                src.push(((v / d).round() as i8) as u8);
            }
        }
        let q = to_q4k(&src, n, From::Q8_0);
        assert_eq!(q.len(), 2 * 144);
        let y: Vec<f32> = q.chunks(144).flat_map(deq).collect();
        let err = (x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.iter().map(|a| a * a).sum::<f32>()).sqrt();
        assert!(err < 0.08, "relative error {err}");
    }
}
