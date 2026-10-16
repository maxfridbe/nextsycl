//! The skin model's prior (`src/data/vertex_group.py`, VertexGroupVoxelSkin), on the CPU: the mesh normalized into
//! [-1, 1] by its own box, a solid voxelization on a 196^3 grid, a graph of the mesh's edges, the voxels' axis
//! neighbours (weighted 3x) and links from voxels to their nearest vertex, shortest paths from each joint's nearest
//! node, and each vertex's weight for a joint from its distance d: 1 / ((1 - a) d + a d^2)^2, normalized (a = 0.5).
//!
//! UniRig voxelizes with pyrender: six orthographic depth renders, a voxel inside when it lies between the nearest
//! surfaces seen from both sides along at least two of the three axes. Here the same tests run on rays through the
//! pixel centres (each triangle rasterized into each axis's grid of rays, the nearest hit from either side kept),
//! with UniRig's index conventions - including its z test, which reads the voxel mirrored in z (it tests height
//! c(k) but places the voxel at -c(k)).

use std::collections::{BinaryHeap, HashMap};

/// The prior: [N, J] weights for the vertices `v` (any frame: normalized here) of triangles `f` and the joints `joints`
/// (the same frame)
pub fn voxel_skin(v: &[[f32; 3]], f: &[[u32; 3]], joints: &[[f32; 3]], grid: usize) -> Vec<f32> {
    let n = v.len();
    let jn = joints.len();
    if n == 0 || jn == 0 {
        return vec![0.0; n * jn];
    }
    // normalized by the vertices' box (float32, as numpy)
    let (c, s) = crate::prep::normalization(v.iter().copied());
    let nv: Vec<[f32; 3]> = v.iter().map(|p| crate::prep::apply(*p, c, s)).collect();
    let nj: Vec<[f32; 3]> = joints.iter().map(|p| crate::prep::apply(*p, c, s)).collect();
    let (vox, cells) = voxelize_cells(&nv, f, grid);
    let d = distances(&nv, f, &vox, &cells, &nj, grid);
    weights(&d, n, &nv, &nj)
}

/// A voxel's centre coordinate on the grid: t * 2 / g - 1 + 1 / g (float32)
fn centre(t: usize, g: usize) -> f32 {
    (t as f32 * 2.0 / g as f32 - 1.0) + (1.0f64 / g as f64) as f32
}

/// Nearest hits along one axis: for each ray (u, v) of the grid (the two other axes, pixel centres), the smallest and
/// largest coordinate where a triangle covers it (+inf / -inf: none)
fn hits(v: &[[f32; 3]], f: &[[u32; 3]], axis: usize, g: usize) -> (Vec<f32>, Vec<f32>) {
    let (ua, va) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let mut lo = vec![f32::INFINITY; g * g];
    let mut hi = vec![f32::NEG_INFINITY; g * g];
    let px = |x: f32| (x + 1.0) * g as f32 / 2.0 - 0.5;
    for t in f {
        let p = [v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]];
        let (u, w) = ([p[0][ua], p[1][ua], p[2][ua]], [p[0][va], p[1][va], p[2][va]]);
        let area = (u[1] - u[0]) * (w[2] - w[0]) - (u[2] - u[0]) * (w[1] - w[0]);
        if area == 0.0 {
            continue;
        }
        let u0 = px(u.iter().copied().fold(f32::INFINITY, f32::min)).ceil().max(0.0) as usize;
        let u1 = px(u.iter().copied().fold(f32::NEG_INFINITY, f32::max)).floor().min(g as f32 - 1.0);
        let w0 = px(w.iter().copied().fold(f32::INFINITY, f32::min)).ceil().max(0.0) as usize;
        let w1 = px(w.iter().copied().fold(f32::NEG_INFINITY, f32::max)).floor().min(g as f32 - 1.0);
        if u1 < 0.0 || w1 < 0.0 {
            continue;
        }
        for a in u0..=u1 as usize {
            let cu = centre(a, g);
            for b in w0..=w1 as usize {
                let cw = centre(b, g);
                // barycentric coordinates of the ray in the triangle's projection
                let l1 = ((cu - u[0]) * (w[2] - w[0]) - (u[2] - u[0]) * (cw - w[0])) / area;
                let l2 = ((u[1] - u[0]) * (cw - w[0]) - (cu - u[0]) * (w[1] - w[0])) / area;
                let l0 = 1.0 - l1 - l2;
                if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                    continue;
                }
                let d = l0 * p[0][axis] + l1 * p[1][axis] + l2 * p[2][axis];
                let k = a * g + b;
                lo[k] = lo[k].min(d);
                hi[k] = hi[k].max(d);
            }
        }
    }
    (lo, hi)
}

/// The solid voxels: their centres (UniRig's convention: (c(i), c(j), c(g - 1 - k)))
pub fn voxelize(v: &[[f32; 3]], f: &[[u32; 3]], g: usize) -> Vec<[f32; 3]> {
    voxelize_cells(v, f, g).0
}

/// The solid voxels: their centres, and each one's cell on the grid (i, j, g - 1 - k: where its centre is)
fn voxelize_cells(v: &[[f32; 3]], f: &[[u32; 3]], g: usize) -> (Vec<[f32; 3]>, Vec<[u16; 3]>) {
    let (xl, xh) = hits(v, f, 0, g); // rays along x, indexed (y, z)
    let (yl, yh) = hits(v, f, 1, g); // along y, indexed (x, z)
    let (zl, zh) = hits(v, f, 2, g); // along z, indexed (x, y)
    let mut out = Vec::new();
    let mut cells = Vec::new();
    for i in 0..g {
        let ci = centre(i, g);
        for j in 0..g {
            let cj = centre(j, g);
            for k in 0..g {
                let ck = centre(k, g);
                // the voxel's place in z, and the rays through it
                let kz = g - 1 - k;
                let gz = centre(kz, g);
                let mx = { let r = j * g + kz; xl[r] <= ci && ci <= xh[r] };
                let my = { let r = i * g + kz; yl[r] <= cj && cj <= yh[r] };
                let mz = { let r = i * g + j; zl[r] <= ck && ck <= zh[r] };
                if (mx && my) || (mx && mz) || (my && mz) {
                    out.push([ci, cj, gz]);
                    cells.push([i as u16, j as u16, kz as u16]);
                }
            }
        }
    }
    (out, cells)
}

fn d3(a: [f32; 3], b: [f32; 3]) -> f64 {
    let (x, y, z) = ((a[0] - b[0]) as f64, (a[1] - b[1]) as f64, (a[2] - b[2]) as f64);
    (x * x + y * y + z * z).sqrt()
}

/// A uniform grid of points for nearest-neighbour queries within a few cells (a coordinate's cell: (x / cell).floor()):
/// dense over the points' box when that is small, else the occupied cells sorted; each cell's points in their order
struct Cells {
    cell: f32,
    lo: [i32; 3],
    dim: [usize; 3],
    /// dense: every cell of the box; sparse: the occupied cells' keys, sorted
    keys: Option<Vec<[i32; 3]>>,
    start: Vec<u32>,
    pts: Vec<u32>,
}

impl Cells {
    fn new(pts: &[[f32; 3]], cell: f32) -> Cells {
        let keys: Vec<[i32; 3]> = pts.iter().map(|p| Self::key(*p, cell)).collect();
        if keys.is_empty() {
            return Cells { cell, lo: [0; 3], dim: [0; 3], keys: None, start: vec![0], pts: Vec::new() };
        }
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for k in &keys {
            for c in 0..3 {
                lo[c] = lo[c].min(k[c]);
                hi[c] = hi[c].max(k[c]);
            }
        }
        let dim = [0, 1, 2].map(|c| (hi[c] as i64 - lo[c] as i64) as usize + 1);
        let total = dim[0].saturating_mul(dim[1]).saturating_mul(dim[2]);
        if total > (16 << 20).max(8 * pts.len()) {
            let mut order: Vec<u32> = (0..pts.len() as u32).collect();
            order.sort_by_key(|&i| (keys[i as usize], i));
            let mut uk: Vec<[i32; 3]> = Vec::new();
            let mut start = Vec::new();
            for (r, &i) in order.iter().enumerate() {
                if uk.last() != Some(&keys[i as usize]) {
                    uk.push(keys[i as usize]);
                    start.push(r as u32);
                }
            }
            start.push(order.len() as u32);
            return Cells { cell, lo, dim, keys: Some(uk), start, pts: order };
        }
        let at = |k: &[i32; 3]| ((k[0] - lo[0]) as usize * dim[1] + (k[1] - lo[1]) as usize) * dim[2] + (k[2] - lo[2]) as usize;
        let mut start = vec![0u32; total + 1];
        for k in &keys {
            start[at(k) + 1] += 1;
        }
        for i in 0..total {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut out = vec![0u32; keys.len()];
        for (i, k) in keys.iter().enumerate() {
            let c = at(k);
            out[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        Cells { cell, lo, dim, keys: None, start, pts: out }
    }
    fn key(p: [f32; 3], cell: f32) -> [i32; 3] {
        p.map(|x| (x / cell).floor() as i32)
    }
    /// The points of one cell
    fn of(&self, q: [i32; 3]) -> &[u32] {
        let at = match &self.keys {
            Some(k) => match k.binary_search(&q) {
                Ok(i) => i,
                Err(_) => return &[],
            },
            None => {
                let r = [0, 1, 2].map(|c| q[c] as i64 - self.lo[c] as i64);
                if (0..3).any(|c| r[c] < 0 || r[c] >= self.dim[c] as i64) {
                    return &[];
                }
                (r[0] as usize * self.dim[1] + r[1] as usize) * self.dim[2] + r[2] as usize
            }
        };
        &self.pts[self.start[at] as usize..self.start[at + 1] as usize]
    }
    /// Points within the 3 x 3 x 3 cells around p
    fn near(&self, p: [f32; 3]) -> impl Iterator<Item = u32> + '_ {
        let k = Self::key(p, self.cell);
        (-1..=1).flat_map(move |a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| [k[0] + a, k[1] + b, k[2] + c])))
            .flat_map(move |q| self.of(q)).copied()
    }
}

/// The threads the CPU stages use
fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(32)
}

/// [f(0), f(1), ... f(n - 1)], computed on the machine's threads
fn par_map<T: Send + Clone + Default>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let per = n.div_ceil(threads()).max(4096);
    std::thread::scope(|sc| {
        for (c, ch) in out.chunks_mut(per).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (k, slot) in ch.iter_mut().enumerate() {
                    *slot = f(c * per + k);
                }
            });
        }
    });
    out
}

/// No edge (a voxel's missing neighbour or link)
const NONE: u32 = u32::MAX;

/// Shortest paths from each joint (its nearest node) over the graph of the mesh and the voxels (`cells`: each voxel's
/// place on the grid): [J, N] for the mesh's vertices (inf: unreachable)
pub fn distances(v: &[[f32; 3]], f: &[[u32; 3]], vox: &[[f32; 3]], cells: &[[u16; 3]], joints: &[[f32; 3]], g: usize) -> Vec<Vec<f64>> {
    let n = v.len();
    let m = vox.len();
    let h = 2.0 / g as f32;
    let reach = h as f64 * 1.001;
    // The graph is the reference's: directed edges summed when repeated (scipy's csr_matrix), then undirected as
    // the smaller of both directions. Voxel to voxel: the axis neighbours on the grid (the only voxels within
    // reach), 3x their distance, the same both ways; each voxel to its nearest vertex within reach, one way. Neither
    // kind repeats, so they go into the adjacency as they are; the vertices' own edges through a map.
    let mut grid = vec![NONE; g * g * g];
    for (i, c) in cells.iter().enumerate() {
        grid[(c[0] as usize * g + c[1] as usize) * g + c[2] as usize] = i as u32;
    }
    let vnb: Vec<[(u32, f64); 6]> = par_map(m, |i| {
        let c = cells[i];
        let mut out = [(NONE, 0.0); 6];
        for (s, (axis, step)) in [(0, -1i32), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)].into_iter().enumerate() {
            let t = c[axis] as i32 + step;
            if t < 0 || t >= g as i32 {
                continue;
            }
            let mut q = c;
            q[axis] = t as u16;
            let o = grid[(q[0] as usize * g + q[1] as usize) * g + q[2] as usize];
            if o != NONE {
                let d = d3(vox[i], vox[o as usize]);
                if d > 0.0 && d < reach {
                    out[s] = ((n + o as usize) as u32, d * 3.0);
                }
            }
        }
        out
    });
    drop(grid);
    // each voxel to its nearest vertex, when within reach
    let mc = Cells::new(v, h);
    let link: Vec<(u32, f64)> = par_map(m, |i| {
        let p = vox[i];
        match mc.near(p).map(|q| (d3(p, v[q as usize]), q)).min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))) {
            Some((d, q)) if d > 0.0 && d < reach => (q, d),
            _ => (NONE, 0.0),
        }
    });
    drop(mc);
    // between vertices: those closer than 1e-5 (at most 3 a vertex), and the mesh's edges (float32 lengths, as numpy)
    let mut w: HashMap<(u32, u32), f64> = HashMap::new();
    let mut add = |a: usize, b: usize, d: f64| *w.entry((a as u32, b as u32)).or_insert(0.0) += d;
    let lk = 1e-5f32;
    let lc = Cells::new(v, lk);
    for (i, p) in v.iter().enumerate() {
        let mut near: Vec<(f64, u32)> = lc.near(*p).filter(|&q| q as usize != i).map(|q| (d3(*p, v[q as usize]), q)).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d, q) in near.into_iter().take(3) {
            if d > 0.0 && d < lk as f64 * 1.001 {
                add(i, q as usize, d);
            }
        }
    }
    drop(lc);
    for t in f {
        for k in 0..3 {
            let (a, b) = (t[k] as usize, t[(k + 1) % 3] as usize);
            let e = [v[a][0] - v[b][0], v[a][1] - v[b][1], v[a][2] - v[b][2]];
            let l = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
            add(a, b, l as f64);
        }
    }
    let mut und: HashMap<(u32, u32), f64> = HashMap::with_capacity(w.len());
    for (&(a, b), &d) in &w {
        let key = (a.min(b), a.max(b));
        let e = und.entry(key).or_insert(f64::INFINITY);
        *e = e.min(d);
    }
    drop(w);
    // the adjacency (CSR)
    let total = n + m;
    let mut deg = vec![0usize; total];
    for &(a, b) in und.keys() {
        deg[a as usize] += 1;
        deg[b as usize] += 1;
    }
    for (i, (nb, &(q, _))) in vnb.iter().zip(&link).enumerate() {
        deg[n + i] += nb.iter().filter(|e| e.0 != NONE).count();
        if q != NONE {
            deg[n + i] += 1;
            deg[q as usize] += 1;
        }
    }
    let mut start = vec![0usize; total + 1];
    for i in 0..total {
        start[i + 1] = start[i] + deg[i];
    }
    drop(deg);
    let mut fill = start.clone();
    let mut adj = vec![(0u32, 0f64); start[total]];
    let mut put = |a: usize, e: (u32, f64)| {
        adj[fill[a]] = e;
        fill[a] += 1;
    };
    for (&(a, b), &d) in &und {
        put(a as usize, (b, d));
        put(b as usize, (a, d));
    }
    for (i, (nb, &(q, d))) in vnb.iter().zip(&link).enumerate() {
        for &e in nb.iter().filter(|e| e.0 != NONE) {
            put(n + i, e);
        }
        if q != NONE {
            put(n + i, (q, d));
            put(q as usize, ((n + i) as u32, d));
        }
    }
    drop((vnb, link, und));
    // each joint's nearest node, vertices and voxels together
    let nodes: Vec<usize> = joints.iter().map(|j| {
        let mut best = (f64::INFINITY, 0usize);
        for (i, p) in v.iter().chain(vox.iter()).enumerate() {
            let d = d3(*j, *p);
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }).collect();
    // Dijkstra from each, in parallel
    let threads = threads().min(joints.len().max(1));
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); joints.len()];
    std::thread::scope(|sc| {
        let chunks: Vec<(usize, &mut [Vec<f64>])> = {
            let per = joints.len().div_ceil(threads).max(1);
            out.chunks_mut(per).enumerate().map(|(i, c)| (i * per, c)).collect()
        };
        for (j0, ch) in chunks {
            let (start, adj, nodes) = (&start, &adj, &nodes);
            sc.spawn(move || {
                for (k, slot) in ch.iter_mut().enumerate() {
                    *slot = dijkstra(start, adj, nodes[j0 + k], total)[..n].to_vec();
                }
            });
        }
    });
    out
}

#[derive(PartialEq)]
struct Item(f64, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Shortest distances from `src` over a CSR graph (start [n + 1], adj (node, weight))
pub fn dijkstra(start: &[usize], adj: &[(u32, f64)], src: usize, n: usize) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    d[src] = 0.0;
    heap.push(Item(0.0, src as u32));
    while let Some(Item(du, u)) = heap.pop() {
        if du > d[u as usize] {
            continue;
        }
        for &(v, w) in &adj[start[u as usize]..start[u as usize + 1]] {
            let nd = du + w;
            if nd < d[v as usize] {
                d[v as usize] = nd;
                heap.push(Item(nd, v));
            }
        }
    }
    d
}

/// The weights from the distances [J][N]: unreachable vertices take their 3 nearest joints' straight distances,
/// every other inf the largest finite distance; d at least 1e-6; (1 / (0.5 d + 0.5 d^2))^2, normalized a vertex
fn weights(d: &[Vec<f64>], n: usize, v: &[[f32; 3]], joints: &[[f32; 3]]) -> Vec<f32> {
    let jn = joints.len();
    let mut d: Vec<Vec<f64>> = d.to_vec();
    let k = jn.min(3);
    for i in 0..n {
        if d.iter().all(|r| r[i].is_infinite()) {
            let mut near: Vec<(f64, usize)> = joints.iter().enumerate().map(|(j, p)| (d3(v[i], *p), j)).collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (dist, j) in near.into_iter().take(k) {
                d[j][i] = dist;
            }
        }
    }
    let max = d.iter().flatten().copied().filter(|x| x.is_finite()).fold(0f64, f64::max);
    let alpha = 0.5;
    let mut out = vec![0f32; n * jn];
    for i in 0..n {
        let s: Vec<f64> = (0..jn).map(|j| {
            let x = if d[j][i].is_finite() { d[j][i] } else { max }.max(1e-6);
            let y = 1.0 / ((1.0 - alpha) * x + alpha * x * x);
            y * y
        }).collect();
        let tot: f64 = s.iter().sum();
        for j in 0..jn {
            let w = s[j] / tot;
            out[i * jn + j] = if w.is_finite() { w as f32 } else { 0.0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed box of 12 triangles from lo to hi
    fn cube(lo: f32, hi: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let v: Vec<[f32; 3]> = (0..8).map(|i| [if i & 1 == 0 { lo } else { hi }, if i & 2 == 0 { lo } else { hi }, if i & 4 == 0 { lo } else { hi }]).collect();
        let f = vec![[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]];
        (v, f)
    }

    #[test]
    fn a_cube_is_filled() {
        let (v, f) = cube(-0.5, 0.5);
        let vox = voxelize(&v, &f, 16);
        // the voxels of centres inside [-0.5, 0.5]: 8 a side (16 cells over [-1, 1])
        assert_eq!(vox.len(), 8 * 8 * 8);
        assert!(vox.iter().all(|p| p.iter().all(|x| x.abs() < 0.5)));
    }

    #[test]
    fn shortest_paths_on_a_line() {
        // 0 - 1 - 2 with weights 1 and 2, and 3 alone
        let start = vec![0, 1, 3, 4, 4];
        let adj = vec![(1, 1.0), (0, 1.0), (2, 2.0), (1, 2.0)];
        let d = dijkstra(&start, &adj, 0, 4);
        assert_eq!(d[..3], [0.0, 1.0, 3.0]);
        assert!(d[3].is_infinite());
    }

    #[test]
    fn weights_favour_the_nearer_joint() {
        let (v, f) = cube(-0.5, 0.5);
        let w = voxel_skin(&v, &f, &[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 16);
        assert_eq!(w.len(), 16);
        // vertex 0 is joint 0's, vertex 7 joint 1's; rows sum to 1
        assert!(w[0] > 0.9 && w[7 * 2 + 1] > 0.9);
        for r in w.chunks(2) {
            assert!((r[0] + r[1] - 1.0).abs() < 1e-5);
        }
    }
}
