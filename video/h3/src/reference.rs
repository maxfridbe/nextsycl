//! The same arithmetic on the CPU, in float32 and exact integers: what a kernel's output is checked against.
//! Slow on purpose - no tricks, so it can be read against the definition in `kernels/h3sycl.h`.

/// The normalized regular Hadamard matrix of size `g` (a power of 4): the Kronecker power of
/// `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, divided by sqrt(g). Row-major.
pub fn hadamard(g: usize) -> Vec<f32> {
    const H4: [[i32; 4]; 4] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];
    let norm = 1.0 / (g as f32).sqrt();
    let mut h = vec![0f32; g * g];
    for i in 0..g {
        for j in 0..g {
            let (mut a, mut b, mut s, mut sign) = (i, j, 1, 1);
            while s < g {
                sign *= H4[a % 4][b % 4];
                a /= 4;
                b /= 4;
                s *= 4;
            }
            h[i * g + j] = sign as f32 * norm;
        }
    }
    h
}

/// An int8 linear layer on the host: int8 weights [n, k], 1 or n scales, an optional bias of n values, and the
/// rotation group size when the weights were stored rotated.
pub struct Layer<'a> {
    pub w: &'a [i8],
    pub n: usize,
    pub k: usize,
    pub wscale: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub group: Option<usize>,
}

/// `int8_linear` as defined in h3sycl.h. x [m, k] float32; the result is [m, n].
pub fn int8_linear(x: &[f32], m: usize, layer: &Layer) -> Vec<f32> {
    let Layer { w, n, k, wscale, bias, group } = *layer;
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let had = group.map(hadamard);
    let mut out = vec![0f32; m * n];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(m.max(1));
    let rows_per = m.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(rows_per * n).enumerate() {
            let had = had.as_deref();
            s.spawn(move || {
                let mut xr = vec![0f32; k];
                let mut q = vec![0i32; k];
                for (ri, orow) in chunk.chunks_mut(n).enumerate() {
                    let r = ci * rows_per + ri;
                    let row = &x[r * k..(r + 1) * k];
                    match (had, group) {
                        (Some(h), Some(g)) => {
                            for (gi, grp) in row.chunks(g).enumerate() {
                                for j in 0..g {
                                    let mut acc = 0f32;
                                    for (i, v) in grp.iter().enumerate() {
                                        acc += v * h[i * g + j];
                                    }
                                    xr[gi * g + j] = acc;
                                }
                            }
                        }
                        _ => xr.copy_from_slice(row),
                    }
                    let scale = (xr.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0).max(1e-30);
                    for (qi, v) in q.iter_mut().zip(&xr) {
                        *qi = (v / scale).round_ties_even().clamp(-128.0, 127.0) as i32;
                    }
                    for (o, (wrow, oi)) in w.chunks(k).zip(orow.iter_mut()).enumerate() {
                        let acc: i32 = wrow.iter().zip(&q).map(|(a, b)| *a as i32 * b).sum();
                        let ws = if wscale.len() == 1 { wscale[0] } else { wscale[o] };
                        *oi = acc as f32 * scale * ws + bias.map_or(0.0, |b| b[o]);
                    }
                }
            });
        }
    });
    out
}

/// How close two results are: the relative error `|a - b| / |b|` and the cosine of the angle between them.
pub fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d2, mut a2, mut b2, mut ab) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        d2 += (x - y) * (x - y);
        a2 += x * x;
        b2 += y * y;
        ab += x * y;
    }
    ((d2 / b2.max(1e-300)).sqrt(), ab / (a2.sqrt() * b2.sqrt()).max(1e-300))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_orthonormal() {
        for g in [4usize, 16, 64] {
            let h = hadamard(g);
            for i in 0..g {
                for j in 0..g {
                    let dot: f32 = (0..g).map(|c| h[i * g + c] * h[j * g + c]).sum();
                    assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5, "g={g} rows {i},{j}: {dot}");
                }
            }
        }
    }

    #[test]
    fn int8_linear_matches_a_plain_linear_within_quantization() {
        // weights that are exactly int8 * scale, so the only error is the 8-bit rounding of the activations
        let (m, k, n) = (3usize, 64usize, 5usize);
        let mut rng = crate::rng::Rng::new(7);
        let x: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let w: Vec<i8> = (0..n * k).map(|_| ((rng.next_u64() % 255) as i32 - 127) as i8).collect();
        let ws = [0.01f32];
        let got = int8_linear(&x, m, &Layer { w: &w, n, k, wscale: &ws, bias: None, group: None });
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for o in 0..n {
                want[r * n + o] = (0..k).map(|c| x[r * k + c] * w[o * k + c] as f32 * ws[0]).sum();
            }
        }
        let (rel, cos) = compare(&got, &want);
        assert!(rel < 0.02 && cos > 0.999, "rel {rel} cos {cos}");
    }
}
