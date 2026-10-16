//! The point choices of the joint encoders' PointNet++ layers, on the host (they are tens of joints): farthest point
//! sampling (torch_cluster's `fps`, started at the first point - torch_cluster starts at a random one, which makes
//! RootNet and BoneNet vary run to run), radius neighbourhoods as PyTorch Geometric's PointConv sees them, and
//! k-nearest interpolation (`knn_interpolate`).

fn d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// ceil(ratio * n) points by farthest point sampling from point 0 (the first farthest on ties)
pub fn fps(pos: &[[f32; 3]], ratio: f32) -> Vec<usize> {
    let n = pos.len();
    if n == 0 {
        return Vec::new();
    }
    let k = ((n as f32 * ratio).ceil() as usize).clamp(1, n);
    let mut out = vec![0usize];
    let mut dist: Vec<f32> = pos.iter().map(|p| d2(*p, pos[0])).collect();
    while out.len() < k {
        let mut best = 0;
        for i in 1..n {
            if dist[i] > dist[best] {
                best = i;
            }
        }
        out.push(best);
        for i in 0..n {
            dist[i] = dist[i].min(d2(pos[i], pos[best]));
        }
    }
    out
}

/// Each sampled point's sources for PointConv: the points strictly within r of it (the nearest `max` of them), less
/// the one whose index equals the target's own, plus that one (PointConv removes and adds self-loops by index, the
/// sources and targets indexed apart)
pub fn radius_edges(pos: &[[f32; 3]], idx: &[usize], r: f32, max: usize) -> Vec<Vec<usize>> {
    idx.iter().enumerate().map(|(i, &c)| {
        let mut nb: Vec<(f32, usize)> = pos.iter().enumerate().map(|(j, p)| (d2(*p, pos[c]), j)).filter(|(d, _)| *d < r * r).collect();
        nb.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        nb.truncate(max);
        let mut out: Vec<usize> = nb.into_iter().map(|x| x.1).filter(|&j| j != i).collect();
        out.push(i);
        out
    }).collect()
}

/// Features x [src, c] at `dst` points: the k nearest sources weighted by 1 / squared distance (at least 1e-16)
pub fn knn_interpolate(x: &[f32], c: usize, src: &[[f32; 3]], dst: &[[f32; 3]], k: usize) -> Vec<f32> {
    let mut out = vec![0f32; dst.len() * c];
    for (i, p) in dst.iter().enumerate() {
        let mut nb: Vec<(f32, usize)> = src.iter().enumerate().map(|(j, q)| (d2(*q, *p), j)).collect();
        nb.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        nb.truncate(k);
        let mut wsum = 0f32;
        for (d, j) in &nb {
            let w = 1.0 / d.max(1e-16);
            wsum += w;
            for (o, v) in out[i * c..(i + 1) * c].iter_mut().zip(&x[j * c..(j + 1) * c]) {
                *o += w * v;
            }
        }
        for o in &mut out[i * c..(i + 1) * c] {
            *o /= wsum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn farthest_points_spread() {
        let p = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        assert_eq!(fps(&p, 0.5), vec![0, 2]);
        assert_eq!(fps(&p, 0.999), vec![0, 2, 3, 1]);
    }

    #[test]
    fn neighbourhoods_and_interpolation() {
        let p = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // targets: points 2 and 0 (indices 0 and 1 among the targets)
        let e = radius_edges(&p, &[2, 0], 0.5, 64);
        // target 0 is point 2: its neighbour is itself (index 2), plus source 0 by index
        assert_eq!(e[0], vec![2, 0]);
        // target 1 is point 0: neighbours 0 and 1, less source 1 (its own index), plus 1
        assert_eq!(e[1], vec![0, 1]);
        let x = [1.0, 3.0];
        let y = knn_interpolate(&x, 1, &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &[[0.5, 0.0, 0.0]], 2);
        // weights 1 / 0.25 and 1 / 2.25: (4 + 3 / 2.25) / (4 + 1 / 2.25)
        assert!((y[0] - (4.0 + 3.0 / 2.25) / (4.0 + 1.0 / 2.25)).abs() < 1e-5);
    }
}
