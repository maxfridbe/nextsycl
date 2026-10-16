//! The shape's grid decoded coarse to fine (FlashVDM's hierarchical volume decoding): the whole lattice at the
//! coarsest level (the resolution halved until it would fall under 63), then at each finer level only the points
//! near the surface - the coarse points where the sign changes between neighbours or the logit is small (|x| < 0.95),
//! dilated, carried to the finer lattice and dilated there again. Tencent leaves the other points NaN; here they take
//! the coarser grid's value, trilinearly - they are far from the surface, so only their sign matters, and a complete
//! grid keeps marching cubes from closing the band's inner edge into a second shell.

/// The resolutions from coarse to fine (cells a side): `res`, res / 2 ... while at least `min`
pub fn levels(res: usize, min: usize) -> Vec<usize> {
    let mut r = vec![];
    let mut x = res;
    if x < min {
        r.push(x);
    }
    while x >= min {
        r.push(x);
        x /= 2;
    }
    r.reverse();
    r
}

/// Points of an n^3 lattice (x slowest) set where `mask` is, grown by one in every direction (a 3 x 3 x 3 box)
pub fn dilate(mask: &[bool], n: usize) -> Vec<bool> {
    // separable: along z, then y, then x
    let mut a = mask.to_vec();
    let at = |x: usize, y: usize, z: usize| (x * n + y) * n + z;
    for axis in 0..3 {
        let src = a.clone();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let i = at(x, y, z);
                    if src[i] {
                        continue;
                    }
                    let c = [x, y, z][axis];
                    let nb = |d: i64| -> bool {
                        let k = c as i64 + d;
                        if k < 0 || k >= n as i64 {
                            return false;
                        }
                        let mut p = [x, y, z];
                        p[axis] = k as usize;
                        src[at(p[0], p[1], p[2])]
                    };
                    a[i] = nb(-1) || nb(1);
                }
            }
        }
    }
    a
}

/// The coarse points near the surface: a neighbour (of 6) on the other side of `level`, or |value - level| < `near`
pub fn near_surface(v: &[f32], n: usize, level: f32, near: f32) -> Vec<bool> {
    let at = |x: usize, y: usize, z: usize| (x * n + y) * n + z;
    let mut out = vec![false; n * n * n];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let i = at(x, y, z);
                let s = v[i] > level;
                let mut m = (v[i] - level).abs() < near;
                for (dx, dy, dz) in [(-1i64, 0i64, 0i64), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                    let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                    if a < 0 || b < 0 || c < 0 || a >= n as i64 || b >= n as i64 || c >= n as i64 {
                        continue;
                    }
                    m |= (v[at(a as usize, b as usize, c as usize)] > level) != s;
                }
                out[i] = m;
            }
        }
    }
    out
}

/// A coarse lattice (nc points a side) resampled trilinearly onto a fine one (nf a side) spanning the same box
pub fn upsample(v: &[f32], nc: usize, nf: usize) -> Vec<f32> {
    let s = (nc - 1) as f32 / (nf - 1).max(1) as f32;
    let taps: Vec<(usize, usize, f32)> = (0..nf).map(|j| {
        let c = j as f32 * s;
        let i = (c.floor() as usize).min(nc - 1);
        (i, (i + 1).min(nc - 1), c - i as f32)
    }).collect();
    let at = |x: usize, y: usize, z: usize| v[(x * nc + y) * nc + z];
    let mut out = vec![0f32; nf * nf * nf];
    for (x, &(x0, x1, fx)) in taps.iter().enumerate() {
        for (y, &(y0, y1, fy)) in taps.iter().enumerate() {
            for (z, &(z0, z1, fz)) in taps.iter().enumerate() {
                let l = |a: f32, b: f32, f: f32| a + (b - a) * f;
                let c00 = l(at(x0, y0, z0), at(x0, y0, z1), fz);
                let c01 = l(at(x0, y1, z0), at(x0, y1, z1), fz);
                let c10 = l(at(x1, y0, z0), at(x1, y0, z1), fz);
                let c11 = l(at(x1, y1, z0), at(x1, y1, z1), fz);
                out[(x * nf + y) * nf + z] = l(l(c00, c01, fy), l(c10, c11, fy), fx);
            }
        }
    }
    out
}

/// The fine points to query after a coarse grid: (their indices on the fine lattice, x slowest)
pub fn refine(coarse: &[f32], nc: usize, nf: usize, level: f32, last: bool) -> Vec<i32> {
    let mut m = near_surface(coarse, nc, level, 0.95);
    if !last {
        m = dilate(&m, nc);
    }
    let mut f = vec![false; nf * nf * nf];
    let map = |i: usize| ((i as f64 * (nf - 1) as f64 / (nc - 1) as f64).round() as usize).min(nf - 1);
    for x in 0..nc {
        for y in 0..nc {
            for z in 0..nc {
                if m[(x * nc + y) * nc + z] {
                    f[(map(x) * nf + map(y)) * nf + map(z)] = true;
                }
            }
        }
    }
    for _ in 0..if last { 2 } else { 1 } {
        f = dilate(&f, nf);
    }
    f.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as i32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_halve_down_to_the_minimum() {
        assert_eq!(levels(380, 63), vec![95, 190, 380]);
        assert_eq!(levels(384, 63), vec![96, 192, 384]);
        assert_eq!(levels(48, 63), vec![48]);
    }

    #[test]
    fn a_band_around_a_sphere_and_its_fill() {
        // a sphere's signed distance, coarse 17^3 -> fine 33^3: the refined points hug the surface and the rest
        // keep their sign from the coarse grid
        let field = |n: usize| -> Vec<f32> {
            let mut v = Vec::new();
            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        let p = [x, y, z].map(|i| -1.0 + 2.0 * i as f32 / (n - 1) as f32);
                        v.push(0.6 - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
                    }
                }
            }
            v
        };
        let (c, f) = (field(17), field(33));
        let idx = refine(&c.iter().map(|x| x * 10.0).collect::<Vec<_>>(), 17, 33, 0.0, true);
        assert!(!idx.is_empty() && idx.len() < 33 * 33 * 33 / 2);
        // every fine cell crossing the surface has all its corners refined
        let set: std::collections::HashSet<i32> = idx.iter().copied().collect();
        let n = 33;
        for x in 0..n - 1 {
            for y in 0..n - 1 {
                for z in 0..n - 1 {
                    let corners: Vec<usize> = (0..8).map(|k| ((x + (k & 1)) * n + y + ((k >> 1) & 1)) * n + z + ((k >> 2) & 1)).collect();
                    let signs: Vec<bool> = corners.iter().map(|&i| f[i] > 0.0).collect();
                    if signs.iter().any(|&s| s != signs[0]) {
                        assert!(corners.iter().all(|i| set.contains(&(*i as i32))), "cell {x} {y} {z}");
                    }
                }
            }
        }
        let up = upsample(&c, 17, 33);
        for (i, (a, b)) in up.iter().zip(&f).enumerate() {
            if !set.contains(&(i as i32)) {
                assert_eq!(*a > 0.0, *b > 0.0);
            }
        }
    }

    #[test]
    fn dilation_grows_a_point_to_a_box() {
        let n = 5;
        let mut m = vec![false; n * n * n];
        m[(2 * n + 2) * n + 2] = true;
        let d = dilate(&m, n);
        assert_eq!(d.iter().filter(|b| **b).count(), 27);
    }
}
