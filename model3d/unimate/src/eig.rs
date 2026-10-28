//! Symmetric eigenproblems (cyclic Jacobi, float64) for the skeleton's Laplacian and the closest rotation of a matrix.

/// Eigenvalues (ascending) and eigenvectors of the symmetric n x n matrix `a` (row-major); vector k is column k of the
/// returned matrix (`v[i * n + k]`)
pub fn eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0f64;
        for i in 0..n {
            for j in 0..i {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| m[a * n + a].total_cmp(&m[b * n + b]));
    let vals = idx.iter().map(|&i| m[i * n + i]).collect();
    let mut vecs = vec![0f64; n * n];
    for (new, &old) in idx.iter().enumerate() {
        for i in 0..n {
            vecs[i * n + new] = v[i * n + old];
        }
    }
    (vals, vecs)
}

/// The rotation closest to the 3 x 3 matrix `m` (row-major; Frobenius norm, proper rotations only): the quaternion
/// (w, x, y, z) of Horn's method, the top eigenvector of a 4 x 4 matrix, as a matrix
pub fn closest_rotation(m: &[f64; 9]) -> [f64; 9] {
    // S_ab = m_ba: maximize tr(R^T m)
    let s = |a: usize, b: usize| m[b * 3 + a];
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (s(0, 0), s(0, 1), s(0, 2), s(1, 0), s(1, 1), s(1, 2), s(2, 0), s(2, 1), s(2, 2));
    let n = [
        sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
        syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
        szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
        sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
    ];
    let (_, v) = eigh(&n, 4);
    let (w, x, y, z) = (v[3], v[7], v[11], v[15]);
    let l = (w * w + x * x + y * y + z * z).sqrt();
    let (w, x, y, z) = (w / l, x / l, y / l, z / l);
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_graph_has_cosine_modes() {
        // the Laplacian of a path of 4 nodes: eigenvalues 0, 2 - sqrt2, 2, 2 + sqrt2 (unnormalized)
        let l = [1.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 1.0];
        let (w, v) = eigh(&l, 4);
        for (got, want) in w.iter().zip([0.0, 2.0 - 2f64.sqrt(), 2.0, 2.0 + 2f64.sqrt()]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        // A v = w v
        for k in 0..4 {
            for i in 0..4 {
                let av: f64 = (0..4).map(|j| l[i * 4 + j] * v[j * 4 + k]).sum();
                assert!((av - w[k] * v[i * 4 + k]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn the_closest_rotation_of_a_noisy_rotation_is_it() {
        // 40 degrees about (1, 2, 3)/|.|
        let (ax, ay, az) = (1.0f64 / 14f64.sqrt(), 2.0 / 14f64.sqrt(), 3.0 / 14f64.sqrt());
        let (c, s) = (40f64.to_radians().cos(), 40f64.to_radians().sin());
        let r = [
            c + ax * ax * (1.0 - c), ax * ay * (1.0 - c) - az * s, ax * az * (1.0 - c) + ay * s,
            ay * ax * (1.0 - c) + az * s, c + ay * ay * (1.0 - c), ay * az * (1.0 - c) - ax * s,
            az * ax * (1.0 - c) - ay * s, az * ay * (1.0 - c) + ax * s, c + az * az * (1.0 - c),
        ];
        let mut noisy = r;
        for (i, x) in noisy.iter_mut().enumerate() {
            *x = *x * 1.7 + 0.02 * ((i as f64 * 1.3).sin());
        }
        let got = closest_rotation(&noisy);
        for i in 0..9 {
            assert!((got[i] - r[i]).abs() < 0.03, "{i}: {} vs {}", got[i], r[i]);
        }
        // an exact rotation comes back unchanged
        let same = closest_rotation(&r);
        for i in 0..9 {
            assert!((same[i] - r[i]).abs() < 1e-9);
        }
    }
}
