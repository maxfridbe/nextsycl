//! The sampler's output as an animation (unimate/utils/motion_utils.py, rotation_conversions.py): the features
//! de-normalized, the root's path from its velocities and facing, each joint's rotation as the closest rotation to the
//! sum of its children's 6D rotations (the model's slot j holds the rotation that turns the bone from j's parent to j,
//! which is joint parent(j)'s own), as quaternions.

use crate::eig::closest_rotation;
use crate::net::{FEAT, T};

type M3 = [f64; 9];

/// 6D (two columns, Gram-Schmidt) to a rotation matrix with columns x, y, z; degenerate input gives the identity
pub fn matrix_6d(c: &[f64]) -> M3 {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let (a, b) = ([c[0], c[1], c[2]], [c[3], c[4], c[5]]);
    let (na, nb) = (norm(a), norm(b));
    let ok = c.iter().all(|v| v.is_finite()) && na > 1e-6 && nb > 1e-6 && norm(cross([a[0] / na, a[1] / na, a[2] / na], [b[0] / nb, b[1] / nb, b[2] / nb])) > 1e-6;
    let (a, b) = if ok { (a, b) } else { ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) };
    let na = norm(a);
    let x = [a[0] / na, a[1] / na, a[2] / na];
    let z = cross(x, b);
    let nz = norm(z);
    let z = [z[0] / nz, z[1] / nz, z[2] / nz];
    let y = cross(z, x);
    [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
}

fn mul(a: &M3, b: &M3) -> M3 {
    let mut o = [0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    o
}

fn transpose(a: &M3) -> M3 {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// A rotation matrix as a quaternion (w, x, y, z), w >= 0 (Shepperd's method)
pub fn quaternion(m: &M3) -> [f64; 4] {
    let (d0, d1, d2) = (m[0], m[4], m[8]);
    let cand = [1.0 + d0 + d1 + d2, 1.0 + d0 - d1 - d2, 1.0 - d0 + d1 - d2, 1.0 - d0 - d1 + d2];
    let b = (0..4).fold(0, |b, i| if cand[i] > cand[b] { i } else { b });
    let s = 2.0 * cand[b].max(1e-12).sqrt();
    let (m21m12, m02m20, m10m01) = (m[7] - m[5], m[2] - m[6], m[3] - m[1]);
    let (m21p12, m02p20, m10p01) = (m[7] + m[5], m[2] + m[6], m[3] + m[1]);
    let q = match b {
        0 => [s / 4.0, m21m12 / s, m02m20 / s, m10m01 / s],
        1 => [m21m12 / s, s / 4.0, m10p01 / s, m02p20 / s],
        2 => [m02m20 / s, m10p01 / s, s / 4.0, m21p12 / s],
        _ => [m10m01 / s, m02p20 / s, m21p12 / s, s / 4.0],
    };
    let l = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q = q.map(|v| v / l);
    if q[0] < 0.0 { q.map(|v| -v) } else { q }
}

/// The decoded motion
pub struct Decoded {
    /// [T][n][12] de-normalized features
    pub features: Vec<f32>,
    /// [T][n] local rotations (w, x, y, z), in the frame asked for
    pub quats: Vec<[f32; 4]>,
    /// [T] the root's path in the canonical frame (x, height, z), the scale of the canonical frame
    pub root: Vec<[f32; 3]>,
}

/// The sampler's `x` [71, 12, 60] for a skeleton of `n` joints with the model's `parents`, de-normalized with
/// `mean` / `std` ([71, 12]); the rotations conjugated into the frame whose map to the canonical one is `to_canon`
pub fn decode(x: &[f32], n: usize, parents: &[i32], mean: &[f64], std: &[f64], to_canon: Option<&[[f64; 3]; 3]>) -> Decoded {
    let mut feat = vec![0f64; T * n * FEAT];
    for t in 0..T {
        for j in 0..n {
            for c in 0..FEAT {
                // the reference de-normalizes in float32
                let v = x[(j * FEAT + c) * T + t] * (std[j * FEAT + c] as f32) + mean[j * FEAT + c] as f32;
                feat[(t * n + j) * FEAT + c] = f64::from(v);
            }
        }
    }
    // the root: facing from channels 3..9, velocity in 9 and 11, height in 1
    let mut root = vec![[0f64; 3]; T];
    let mut acc = [0f64; 3];
    for t in 0..T {
        let f = &feat[t * n * FEAT..t * n * FEAT + FEAT];
        let r = matrix_6d(&f[3..9]);
        let mut v = [0f64; 3];
        if t > 0 {
            let p = &feat[(t - 1) * n * FEAT..(t - 1) * n * FEAT + FEAT];
            v = [p[9], 0.0, p[11]];
        }
        // R^T v
        let w = [0, 1, 2].map(|i| (0..3).map(|j| r[j * 3 + i] * v[j]).sum::<f64>());
        for k in 0..3 {
            acc[k] += w[k];
        }
        root[t] = [acc[0], f[1], acc[2]];
    }
    let mut quats = Vec::with_capacity(T * n);
    let mut has_child = vec![false; n];
    for &p in parents.iter().take(n) {
        if p >= 0 {
            has_child[p as usize] = true;
        }
    }
    for t in 0..T {
        let hml: Vec<M3> = (0..n).map(|j| matrix_6d(&feat[(t * n + j) * FEAT + 3..(t * n + j) * FEAT + 9])).collect();
        let mut sum = vec![[0f64; 9]; n];
        for j in 1..n {
            let p = parents[j] as usize;
            for k in 0..9 {
                sum[p][k] += hml[j][k];
            }
        }
        for j in 0..n {
            let mut m = if has_child[j] { closest_rotation(&sum[j]) } else { [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
            if let Some(r) = to_canon {
                let rm: M3 = [r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]];
                m = mul(&mul(&transpose(&rm), &m), &rm);
            }
            quats.push(quaternion(&m).map(|v| v as f32));
        }
    }
    Decoded { features: feat.iter().map(|v| *v as f32).collect(), quats, root: root.iter().map(|r| r.map(|v| v as f32)).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::J;

    #[test]
    fn six_d_and_quaternions() {
        let id = matrix_6d(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(id, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(quaternion(&id), [1.0, 0.0, 0.0, 0.0]);
        // 90 degrees about Z: x -> y, y -> -x
        let rz = matrix_6d(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]);
        let q = quaternion(&rz);
        assert!((q[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12 && (q[3] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        // degenerate input is the identity
        assert_eq!(matrix_6d(&[0.0; 6]), id);
    }

    #[test]
    fn a_still_skeleton_decodes_to_identity_and_a_straight_path() {
        let n = 3;
        let parents = [-1, 0, 1];
        let (mean, std) = (vec![0f64; J * FEAT], vec![1f64; J * FEAT]);
        let mut x = vec![0f32; J * FEAT * T];
        for j in 0..n {
            for t in 0..T {
                for (c, v) in [(3, 1.0), (7, 1.0)] {
                    x[(j * FEAT + c) * T + t] = v;
                }
            }
        }
        // the root moves 0.5 along z every frame, 1.0 high
        for t in 0..T {
            x[(11) * T + t] = 0.5;
            x[T + t] = 1.0;
        }
        let d = decode(&x, n, &parents, &mean, &std, None);
        assert!(d.quats.iter().all(|q| (q[0] - 1.0).abs() < 1e-6));
        assert_eq!(d.root[0], [0.0, 1.0, 0.0]);
        assert!((d.root[10][2] - 5.0).abs() < 1e-5);
    }
}
