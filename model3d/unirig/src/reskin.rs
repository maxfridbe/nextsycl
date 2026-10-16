//! The samples' weights carried to the mesh's vertices (src/system/skin.py, `reskin`: sample_method median, alpha 2,
//! threshold 0.03): each vertex takes the median of its 7 nearest samples' weights; one pass of diffusion over the
//! mesh's edges, on weights summed up the skeleton, only from hotter to cooler neighbours; weights below 0.03 dropped
//! where a larger one remains; renormalized.
//!
//! The diffusion is UniRig's numpy, including its `neighbor_skin[to] += ...`: with repeated targets numpy keeps the
//! last edge's value (no sum), so a vertex hears from one neighbour - the last edge into it - and that is mirrored.

/// k nearest of `pts` to `q` (brute force over a grid's nearby cells; exact): indices, nearest first
struct Knn {
    cell: f32,
    map: std::collections::HashMap<[i32; 3], Vec<u32>>,
}

impl Knn {
    fn new(pts: &[[f32; 3]]) -> Knn {
        // about 4 points a cell
        let (lo, hi) = pts.iter().fold(([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]), |(l, h), p| {
            ([l[0].min(p[0]), l[1].min(p[1]), l[2].min(p[2])], [h[0].max(p[0]), h[1].max(p[1]), h[2].max(p[2])])
        });
        let vol: f32 = (0..3).map(|c| (hi[c] - lo[c]).max(1e-3)).product();
        let cell = (vol * 4.0 / pts.len().max(1) as f32).cbrt().max(1e-6);
        let mut map: std::collections::HashMap<[i32; 3], Vec<u32>> = std::collections::HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(p.map(|x| (x / cell).floor() as i32)).or_default().push(i as u32);
        }
        Knn { cell, map }
    }

    fn query(&self, pts: &[[f32; 3]], q: [f32; 3], k: usize) -> Vec<u32> {
        let k = k.min(pts.len());
        if k == 0 {
            return Vec::new();
        }
        let c = q.map(|x| (x / self.cell).floor() as i32);
        let d2 = |i: u32| {
            let p = pts[i as usize];
            let (x, y, z) = ((p[0] - q[0]) as f64, (p[1] - q[1]) as f64, (p[2] - q[2]) as f64);
            x * x + y * y + z * z
        };
        let mut best: Vec<(f64, u32)> = Vec::new();
        for r in 0i32.. {
            // the shell of cells r away
            for a in -r..=r {
                for b in -r..=r {
                    for e in -r..=r {
                        if a.abs().max(b.abs()).max(e.abs()) != r {
                            continue;
                        }
                        if let Some(v) = self.map.get(&[c[0] + a, c[1] + b, c[2] + e]) {
                            best.extend(v.iter().map(|&i| (d2(i), i)));
                        }
                    }
                }
            }
            if best.len() >= k {
                best.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                // a point beyond this shell is at least r cells away
                let bound = (r as f64 * self.cell as f64).powi(2);
                if best[k - 1].0 <= bound || r > 1 << 12 {
                    break;
                }
            }
            if r > 1 << 12 {
                break;
            }
        }
        best.truncate(k);
        best.into_iter().map(|x| x.1).collect()
    }
}

/// The weights [V, J] of the mesh (`v`, `f`) from the samples' `sw` [S, J] at `pts`
pub fn reskin(pts: &[[f32; 3]], sw: &[f32], v: &[[f32; 3]], f: &[[u32; 3]], parents: &[Option<usize>]) -> Vec<f32> {
    let j = parents.len();
    let n = v.len();
    let knn = Knn::new(pts);
    let mut skin = vec![0f32; n * j];
    for (i, p) in v.iter().enumerate() {
        let near = knn.query(pts, *p, 7);
        for c in 0..j {
            let mut vals: Vec<f32> = near.iter().map(|&s| sw[s as usize * j + c]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let m = vals.len();
            skin[i * j + c] = if m % 2 == 1 { vals[m / 2] } else { (vals[m / 2 - 1] + vals[m / 2]) / 2.0 };
        }
    }
    // weights summed up the skeleton (children last to first)
    let mut sum = skin.clone();
    for i in (0..j).rev() {
        if let Some(p) = parents[i] {
            for r in 0..n {
                sum[r * j + p] += sum[r * j + i];
            }
        }
    }
    // the directed edges, in numpy's order: (0,1), (1,2), (2,0) of every face, then each reversed
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(f.len() * 6);
    for k in 0..3 {
        edges.extend(f.iter().map(|t| (t[k], t[(k + 1) % 3])));
    }
    let fwd = edges.len();
    for e in 0..fwd {
        let (a, b) = edges[e];
        edges.push((b, a));
    }
    let mut nb = vec![0f32; n * j];
    let mut co = vec![0f32; n * j];
    for &(a, b) in &edges {
        let (pa, pb) = (v[a as usize], v[b as usize]);
        let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let dis = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let c = (-dis * 2.0).exp();
        for k in 0..j {
            let hot = sum[b as usize * j + k] < sum[a as usize * j + k];
            let m = if hot { 1.0 } else { 0.0 };
            // the last edge into b wins (numpy's fancy-index +=)
            nb[b as usize * j + k] = sum[a as usize * j + k] * c * m;
            co[b as usize * j + k] = c * m;
        }
    }
    for i in 0..n * j {
        sum[i] = (sum[i] + nb[i]) / (1.0 + co[i]);
    }
    for i in 0..j {
        if let Some(p) = parents[i] {
            for r in 0..n {
                sum[r * j + p] -= sum[r * j + i];
            }
        }
    }
    let mut out = sum;
    for row in out.chunks_exact_mut(j) {
        let s: f32 = row.iter().sum();
        row.iter_mut().for_each(|x| *x /= s);
    }
    // small weights dropped where a larger one remains
    for row in out.chunks_exact_mut(j) {
        if row.iter().any(|x| *x >= 0.03) {
            row.iter_mut().filter(|x| **x < 0.03).for_each(|x| *x = 0.0);
        }
        let s: f32 = row.iter().sum();
        row.iter_mut().for_each(|x| *x /= s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_samples_are_found() {
        let pts: Vec<[f32; 3]> = (0..1000).map(|i| [(i % 10) as f32, ((i / 10) % 10) as f32, (i / 100) as f32]).collect();
        let k = Knn::new(&pts);
        let got = k.query(&pts, [4.1, 4.0, 4.0], 7);
        assert_eq!(got[0], 444);
        assert_eq!(got.len(), 7);
        // the six axis neighbours of 444 and 445 are next: all within distance 1.1
        for i in got {
            let p = pts[i as usize];
            assert!(((p[0] - 4.1).powi(2) + (p[1] - 4.0).powi(2) + (p[2] - 4.0).powi(2)).sqrt() < 1.2);
        }
    }

    #[test]
    fn weights_stay_normalized() {
        // a strip of two triangles; samples on its corners, two bones (1 child of 0)
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let f = vec![[0, 1, 2], [0, 2, 3]];
        let sw = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = reskin(&v, &sw, &v, &f, &[None, Some(0)]);
        for r in w.chunks(2) {
            assert!((r[0] + r[1] - 1.0).abs() < 1e-5);
        }
    }
}
