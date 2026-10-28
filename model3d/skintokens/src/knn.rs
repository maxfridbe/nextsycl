//! The samples' weights carried to the mesh's vertices (Asset.from_data of the demo): each vertex takes the weights of its 8
//! nearest samples, weighted by 1 / (distance + 1e-8), normalized. The nearest neighbours come from a uniform grid.

/// The `k` nearest of `pts` to each of `queries`: (index, distance), nearest first, `k` a query (fewer points: the points
/// repeat their last)
pub fn nearest(pts: &[[f32; 3]], queries: &[[f32; 3]], k: usize) -> Vec<(u32, f32)> {
    let n = pts.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in pts.iter().chain(queries) {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let g = ((n as f64 / 3.0).cbrt().ceil() as usize).clamp(1, 160);
    let ext = [0, 1, 2].map(|c| (hi[c] - lo[c]).max(1e-9));
    let size = [0, 1, 2].map(|c| ext[c] / g as f32);
    let cs = size[0].min(size[1]).min(size[2]);
    let cell = |p: &[f32; 3]| [0, 1, 2].map(|c| (((p[c] - lo[c]) / size[c]) as usize).min(g - 1));
    let mut start = vec![0u32; g * g * g + 1];
    for p in pts {
        let c = cell(p);
        start[(c[2] * g + c[1]) * g + c[0] + 1] += 1;
    }
    for i in 0..g * g * g {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut order = vec![0u32; n];
    for (i, p) in pts.iter().enumerate() {
        let c = cell(p);
        let at = (c[2] * g + c[1]) * g + c[0];
        order[fill[at] as usize] = i as u32;
        fill[at] += 1;
    }
    let one = |q: &[f32; 3]| -> Vec<(u32, f32)> {
        let c = cell(q);
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        let mut r = 0usize;
        loop {
            let (x0, x1) = (c[0].saturating_sub(r), (c[0] + r).min(g - 1));
            let (y0, y1) = (c[1].saturating_sub(r), (c[1] + r).min(g - 1));
            let (z0, z1) = (c[2].saturating_sub(r), (c[2] + r).min(g - 1));
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        // only the shell of Chebyshev radius r
                        if r > 0 && x.abs_diff(c[0]).max(y.abs_diff(c[1])).max(z.abs_diff(c[2])) != r {
                            continue;
                        }
                        let at = (z * g + y) * g + x;
                        for &i in &order[start[at] as usize..start[at + 1] as usize] {
                            let p = pts[i as usize];
                            let d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
                            if best.len() < k || d2 < best[best.len() - 1].0 {
                                let pos = best.partition_point(|b| b.0 <= d2);
                                best.insert(pos, (d2, i));
                                best.truncate(k);
                            }
                        }
                    }
                }
            }
            // everything outside the shells searched is at least r cells away
            if best.len() == k && best[k - 1].0.sqrt() <= r as f32 * cs {
                break;
            }
            if r >= g {
                break;
            }
            r += 1;
        }
        let mut out: Vec<(u32, f32)> = best.iter().map(|b| (b.1, b.0.sqrt())).collect();
        while out.len() < k {
            out.push(*out.last().unwrap_or(&(0, 0.0)));
        }
        out
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    let per = queries.len().div_ceil(threads).max(1);
    let mut out = vec![(0u32, 0f32); queries.len() * k];
    std::thread::scope(|s| {
        for (qs, os) in queries.chunks(per).zip(out.chunks_mut(per * k)) {
            let one = &one;
            s.spawn(move || {
                for (q, o) in qs.iter().zip(os.chunks_mut(k)) {
                    o.copy_from_slice(&one(q));
                }
            });
        }
    });
    out
}

/// The weights `skin` [N, J] of the samples `pts` carried to `targets`: the `k` nearest samples blended by inverse distance
/// -> [V, J]
pub fn carry(pts: &[[f32; 3]], skin: &[f32], j: usize, targets: &[[f32; 3]], k: usize) -> Vec<f32> {
    let nn = nearest(pts, targets, k);
    let mut out = vec![0f32; targets.len() * j];
    for (v, row) in nn.chunks_exact(k).enumerate() {
        let w: Vec<f32> = row.iter().map(|(_, d)| 1.0 / (d + 1e-8)).collect();
        let z: f32 = w.iter().sum();
        for ((i, _), w) in row.iter().zip(&w) {
            let s = &skin[*i as usize * j..(*i as usize + 1) * j];
            for (o, x) in out[v * j..(v + 1) * j].iter_mut().zip(s) {
                *o += x * (w / z);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_match_brute_force() {
        let mut r = crate::ops::Rng::new(9);
        let pts: Vec<[f32; 3]> = (0..2000).map(|_| [r.unit() as f32, r.unit() as f32, r.unit() as f32]).collect();
        let qs: Vec<[f32; 3]> = (0..50).map(|_| [r.unit() as f32 * 1.2 - 0.1, r.unit() as f32, r.unit() as f32]).collect();
        let got = nearest(&pts, &qs, 8);
        for (qi, q) in qs.iter().enumerate() {
            let mut all: Vec<(f32, usize)> = pts.iter().enumerate().map(|(i, p)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2), i)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            for k in 0..8 {
                assert!((got[qi * 8 + k].1 - all[k].0.sqrt()).abs() < 1e-5, "query {qi} neighbour {k}");
            }
        }
    }

    #[test]
    fn a_vertex_on_a_sample_takes_its_weights() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let skin = [1.0, 0.0, 0.0, 1.0];
        let w = carry(&pts, &skin, 2, &[[0.0, 0.0, 0.0]], 2);
        assert!(w[0] > 0.999 && w[1] < 1e-3);
    }
}
