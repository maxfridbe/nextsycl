//! RigNet's geometry, on the CPU: the normalization (quick_start.normalize_obj), the mesh's edges, the surface
//! geodesic distances (common_ops.calc_surface_geodesic: Poisson-disk samples, a 5-nearest graph, Dijkstra from every
//! sample), the geodesic edges (gen_dataset.get_geo_edges), the voxelization the joint and bone checks read (binvox
//! -d 88 in the reference) and the volumetric geodesic distances from vertices to bones (visibility by rays from each
//! bone to each vertex, geodesic detours for the hidden ones: compute_volumetric_geodesic and
//! quick_start.calc_geodesic_matrix).

use std::collections::BinaryHeap;

use nextsycl_model3d::mesh::{add, cross, dot, length, scale, sub};

/// splitmix64: the sampling's and the geodesic edges' choices, from the request's seed
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

/// The threads the CPU stages use
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16)
}

/// Scaled to height 1 on its longest side, its feet (lowest y) at 0, centred in x and z: (positions, pivot, scale);
/// the original is p / scale + pivot
pub fn normalize(v: &[[f32; 3]]) -> (Vec<[f32; 3]>, [f32; 3], f32) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in v {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let s = 1.0 / dims.iter().fold(0f32, |a, b| a.max(*b)).max(1e-12);
    let pivot = [(lo[0] + hi[0]) / 2.0, lo[1], (lo[2] + hi[2]) / 2.0];
    (v.iter().map(|p| scale(sub(*p, pivot), s)).collect(), pivot, s)
}

/// The mesh's edges, both ways, each once (gen_dataset.get_tpl_edges), as (source, target)
pub fn tpl_edges(n: usize, faces: &[[u32; 3]]) -> Vec<(u32, u32)> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in faces {
        for k in 0..3 {
            for m in 0..3 {
                if f[k] != f[m] {
                    adj[f[k] as usize].push(f[m]);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (v, a) in adj.iter_mut().enumerate() {
        a.sort_unstable();
        a.dedup();
        out.extend(a.iter().map(|&n| (v as u32, n)));
    }
    out
}

/// Area-weighted vertex normals (open3d's compute_vertex_normals)
pub fn vertex_normals(v: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut n = vec![[0f32; 3]; v.len()];
    for f in faces {
        let c = cross(sub(v[f[1] as usize], v[f[0] as usize]), sub(v[f[2] as usize], v[f[0] as usize]));
        for &i in f {
            n[i as usize] = add(n[i as usize], c);
        }
    }
    n.into_iter().map(|x| {
        let l = length(x);
        if l > 0.0 { scale(x, 1.0 / l) } else { x }
    }).collect()
}

/// A uniform grid of points for radius queries
struct Grid {
    lo: [f32; 3],
    cell: f32,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn new(p: &[[f32; 3]], cell: f32) -> Grid {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for q in p {
            for c in 0..3 {
                lo[c] = lo[c].min(q[c]);
                hi[c] = hi[c].max(q[c]);
            }
        }
        let cell = cell.max(1e-6);
        let dims = [0, 1, 2].map(|c| (((hi[c] - lo[c]) / cell) as usize + 1).clamp(1, 256));
        let mut g = Grid { lo, cell, dims, cells: vec![Vec::new(); dims[0] * dims[1] * dims[2]] };
        for (i, q) in p.iter().enumerate() {
            let k = g.key(*q);
            g.cells[k].push(i as u32);
        }
        g
    }
    fn at(&self, x: f32, c: usize) -> usize {
        (((x - self.lo[c]) / self.cell) as isize).clamp(0, self.dims[c] as isize - 1) as usize
    }
    fn key(&self, q: [f32; 3]) -> usize {
        (self.at(q[2], 2) * self.dims[1] + self.at(q[1], 1)) * self.dims[0] + self.at(q[0], 0)
    }
    /// The points within r of q (r at most a few cells)
    fn near(&self, p: &[[f32; 3]], q: [f32; 3], r: f32, out: &mut Vec<u32>) {
        out.clear();
        let reach = (r / self.cell).ceil() as isize;
        let c = [0, 1, 2].map(|k| self.at(q[k], k) as isize);
        for z in (c[2] - reach).max(0)..=(c[2] + reach).min(self.dims[2] as isize - 1) {
            for y in (c[1] - reach).max(0)..=(c[1] + reach).min(self.dims[1] as isize - 1) {
                for x in (c[0] - reach).max(0)..=(c[0] + reach).min(self.dims[0] as isize - 1) {
                    for &i in &self.cells[(z as usize * self.dims[1] + y as usize) * self.dims[0] + x as usize] {
                        let d = sub(p[i as usize], q);
                        if dot(d, d) <= r * r {
                            out.push(i);
                        }
                    }
                }
            }
        }
    }
}

/// Points spread evenly over the surface with their normals (open3d's sample_points_poisson_disk: 5 n area-weighted
/// samples, then Yuksel's weighted sample elimination down to n)
pub fn poisson_disk(v: &[[f32; 3]], vn: &[[f32; 3]], faces: &[[u32; 3]], n: usize, rng: &mut Rng) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let areas: Vec<f64> = faces.iter().map(|f| {
        0.5 * length(cross(sub(v[f[1] as usize], v[f[0] as usize]), sub(v[f[2] as usize], v[f[0] as usize]))) as f64
    }).collect();
    let total: f64 = areas.iter().sum();
    if faces.is_empty() || total <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let mut cum = Vec::with_capacity(areas.len());
    let mut s = 0.0;
    for a in &areas {
        s += a;
        cum.push(s);
    }
    let m = 5 * n;
    let (mut pts, mut nrm) = (Vec::with_capacity(m), Vec::with_capacity(m));
    for _ in 0..m {
        let t = rng.unit() * total;
        let fi = cum.partition_point(|c| *c < t).min(faces.len() - 1);
        let f = faces[fi];
        let (r1, r2) = (rng.unit().sqrt() as f32, rng.unit() as f32);
        let (a, b, c) = (1.0 - r1, r1 * (1.0 - r2), r1 * r2);
        let at = |x: &[[f32; 3]]| add(add(scale(x[f[0] as usize], a), scale(x[f[1] as usize], b)), scale(x[f[2] as usize], c));
        pts.push(at(v));
        let q = at(vn);
        let l = length(q);
        nrm.push(if l > 0.0 { scale(q, 1.0 / l) } else { q });
    }
    if n >= m {
        return (pts, nrm);
    }
    // weighted sample elimination
    let (alpha, beta, gamma) = (8.0f64, 0.5f64, 1.5f64);
    let ratio = n as f64 / m as f64;
    let r_max = 2.0 * ((total / n as f64) / (2.0 * 3f64.sqrt())).sqrt();
    let r_min = r_max * beta * (1.0 - ratio.powf(gamma));
    let wf = |d2: f32| -> f64 {
        let d = (d2 as f64).sqrt().max(r_min);
        (1.0 - d / r_max).max(0.0).powf(alpha)
    };
    let grid = Grid::new(&pts, r_max as f32);
    let mut nb = Vec::new();
    let mut neighbours: Vec<Vec<u32>> = Vec::with_capacity(m);
    let mut w = vec![0f64; m];
    for i in 0..m {
        grid.near(&pts, pts[i], r_max as f32, &mut nb);
        let list: Vec<u32> = nb.iter().copied().filter(|&j| j as usize != i).collect();
        w[i] = list.iter().map(|&j| { let d = sub(pts[j as usize], pts[i]); wf(dot(d, d)) }).sum();
        neighbours.push(list);
    }
    #[derive(PartialEq)]
    struct E(f64, u32);
    impl Eq for E {}
    impl PartialOrd for E {
        fn partial_cmp(&self, o: &E) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for E {
        fn cmp(&self, o: &E) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
        }
    }
    let mut heap: BinaryHeap<E> = (0..m).map(|i| E(w[i], i as u32)).collect();
    let mut alive = vec![true; m];
    let mut left = m;
    while left > n {
        let Some(E(wt, i)) = heap.pop() else { break };
        let i = i as usize;
        if !alive[i] || wt != w[i] {
            continue;
        }
        alive[i] = false;
        left -= 1;
        for &j in &neighbours[i] {
            let j = j as usize;
            if alive[j] {
                let d = sub(pts[j], pts[i]);
                w[j] -= wf(dot(d, d));
                heap.push(E(w[j], j as u32));
            }
        }
    }
    let keep: Vec<usize> = (0..m).filter(|&i| alive[i]).collect();
    (keep.iter().map(|&i| pts[i]).collect(), keep.iter().map(|&i| nrm[i]).collect())
}

/// Shortest paths from `src` over an undirected weighted graph (adjacency lists)
pub fn dijkstra(adj: &[Vec<(u32, f32)>], src: usize, dist: &mut [f64]) {
    dist.fill(f64::INFINITY);
    dist[src] = 0.0;
    #[derive(PartialEq)]
    struct E(f64, u32);
    impl Eq for E {}
    impl PartialOrd for E {
        fn partial_cmp(&self, o: &E) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for E {
        fn cmp(&self, o: &E) -> std::cmp::Ordering {
            o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
        }
    }
    let mut h = BinaryHeap::new();
    h.push(E(0.0, src as u32));
    while let Some(E(d, u)) = h.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &(v, w) in &adj[u as usize] {
            let nd = d + w as f64;
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                h.push(E(nd, v));
            }
        }
    }
}

/// The surface geodesic distances between the vertices: each vertex's nearest sample's row of the samples' matrix
pub struct Geodesic {
    pub s: usize,
    /// [s, s], float32
    pub dist: Vec<f32>,
    /// each vertex's sample
    pub nn: Vec<u32>,
}

impl Geodesic {
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.dist[self.nn[i] as usize * self.s + self.nn[j] as usize]
    }
    /// The whole vertex matrix [n, n] (the reference's surface_geodesic)
    pub fn matrix(&self) -> Vec<f32> {
        let n = self.nn.len();
        (0..n * n).map(|k| self.get(k / n, k % n)).collect()
    }
}

/// common_ops.calc_surface_geodesic: samples linked to their 5 nearest (when the normals are not opposed: cosine >
/// -0.5) by their distance, every pair's shortest path (no path: the straight distance + 8), each vertex taking its
/// nearest sample's
pub fn surface_geodesic(v: &[[f32; 3]], pts: &[[f32; 3]], nrm: &[[f32; 3]]) -> Geodesic {
    let s = pts.len();
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); s];
    for p in 0..s {
        let mut d: Vec<(f32, u32)> = (0..s).map(|q| (length(sub(pts[q], pts[p])), q as u32)).collect();
        d.select_nth_unstable_by(6.min(s - 1), |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut near: Vec<(f32, u32)> = d[..7.min(s)].to_vec();
        near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let np = length(nrm[p]);
        for &(dq, q) in near.iter().skip(1).take(5) {
            let nq = length(nrm[q as usize]);
            let cos = dot(nrm[q as usize], nrm[p]) / (nq * np + 1e-10);
            if cos > -0.5 {
                adj[p].push((q, dq));
                adj[q as usize].push((p as u32, dq));
            }
        }
    }
    // an edge stored both ways, once each (the reference's sparse matrix keeps one value a pair)
    for a in &mut adj {
        a.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
        a.dedup_by_key(|x| x.0);
    }
    let mut dist = vec![0f32; s * s];
    let nt = threads();
    let chunk = s.div_ceil(nt).max(1);
    std::thread::scope(|sc| {
        for (t, rows) in dist.chunks_mut(chunk * s).enumerate() {
            let adj = &adj;
            sc.spawn(move || {
                let mut d = vec![0f64; s];
                for (r, row) in rows.chunks_mut(s).enumerate() {
                    let p = t * chunk + r;
                    dijkstra(adj, p, &mut d);
                    for (q, o) in row.iter_mut().enumerate() {
                        *o = if d[q].is_finite() { d[q] as f32 } else { 8.0 + length(sub(pts[q], pts[p])) };
                    }
                }
            });
        }
    });
    let nn = v.iter().map(|x| {
        let mut best = (f32::INFINITY, 0u32);
        for (q, p) in pts.iter().enumerate() {
            let d = sub(*p, *x);
            let d = dot(d, d);
            if d < best.0 {
                best = (d, q as u32);
            }
        }
        best.1
    }).collect();
    Geodesic { s, dist, nn }
}

/// gen_dataset.get_geo_edges: each vertex to the vertices within geodesic 0.06 of it (itself apart), 10 of them at
/// random when there are more; (source, target)
pub fn geo_edges(g: &Geodesic, rng: &mut Rng) -> Vec<(u32, u32)> {
    let n = g.nn.len();
    let mut out = Vec::new();
    for i in 0..n {
        let mut ball: Vec<u32> = (0..n).filter(|&j| j != i && g.get(i, j) <= 0.06).map(|j| j as u32).collect();
        if ball.len() > 10 {
            for k in 0..10 {
                let r = k + rng.below(ball.len() - k);
                ball.swap(k, r);
            }
            ball.truncate(10);
        }
        out.extend(ball.into_iter().map(|j| (i as u32, j)));
    }
    out
}

/// A solid voxelization (binvox -d 88's role): the grid over the bounding cube (from the box's lowest corner, the
/// longest side), a voxel filled when the surface passes through it or when lines along at least two of the three
/// axes through its centre cross the surface an odd number of times before it. binvox's own rules (its parity
/// voting, its exact surface test) differ in detail.
pub struct Vox {
    pub dims: usize,
    pub translate: [f32; 3],
    pub scale: f32,
    /// [x][y][z]
    pub data: Vec<bool>,
}

impl Vox {
    pub fn at(&self, x: usize, y: usize, z: usize) -> bool {
        self.data[(x * self.dims + y) * self.dims + z]
    }

    /// utils/mst_utils.inside_check: the voxel by rounding (half to even, as numpy) its scaled position
    pub fn inside(&self, p: [f32; 3]) -> bool {
        let d = self.dims as i64;
        let vc = [0, 1, 2].map(|c| (((p[c] - self.translate[c]) as f64 / self.scale as f64) * d as f64).round_ties_even() as i64);
        if vc.iter().any(|&x| x < 0 || x >= d) {
            return false;
        }
        self.at(vc[0] as usize, vc[1] as usize, vc[2] as usize)
    }
}

pub fn voxelize(v: &[[f32; 3]], faces: &[[u32; 3]], dims: usize) -> Vox {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in v {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let s = (0..3).map(|c| hi[c] - lo[c]).fold(0f32, f32::max).max(1e-9);
    let vs = s / dims as f32;
    let idx = |x: usize, y: usize, z: usize| (x * dims + y) * dims + z;
    let mut surf = vec![false; dims * dims * dims];
    let cellof = |p: [f32; 3]| [0, 1, 2].map(|c| (((p[c] - lo[c]) / vs) as isize).clamp(0, dims as isize - 1) as usize);
    for f in faces {
        let (a, b, c) = (v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]);
        let steps = ((length(sub(b, a)).max(length(sub(c, a))).max(length(sub(c, b))) / (0.25 * vs)).ceil() as usize).clamp(1, 4096);
        for i in 0..=steps {
            for j in 0..=steps - i {
                let (u, w) = (i as f32 / steps as f32, j as f32 / steps as f32);
                let p = add(a, add(scale(sub(b, a), u), scale(sub(c, a), w)));
                let k = cellof(p);
                surf[idx(k[0], k[1], k[2])] = true;
            }
        }
    }
    // parity along each axis: the crossings of the line through each column of voxel centres
    let mut votes = vec![0u8; dims * dims * dims];
    for axis in 0..3 {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut cols: Vec<Vec<f32>> = vec![Vec::new(); dims * dims];
        let centre = |i: usize, c: usize| lo[c] + (i as f32 + 0.5) * vs;
        // the columns nudged off the voxel centres by an irrational fraction, so none runs exactly along an edge or
        // through a vertex (where a crossing would count twice or not at all)
        let (nu, nw) = (vs * 1e-3 * 0.618_034, vs * 1e-3 * 0.414_213_6);
        for f in faces {
            let p = [v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]];
            let (umin, umax) = (p.iter().map(|q| q[u]).fold(f32::INFINITY, f32::min), p.iter().map(|q| q[u]).fold(f32::NEG_INFINITY, f32::max));
            let (wmin, wmax) = (p.iter().map(|q| q[w]).fold(f32::INFINITY, f32::min), p.iter().map(|q| q[w]).fold(f32::NEG_INFINITY, f32::max));
            let i0 = (((umin - lo[u] - nu) / vs - 0.5).ceil().max(0.0)) as usize;
            let j0 = (((wmin - lo[w] - nw) / vs - 0.5).ceil().max(0.0)) as usize;
            let i1 = ((((umax - lo[u] - nu) / vs - 0.5).floor()) as isize).min(dims as isize - 1);
            let j1 = ((((wmax - lo[w] - nw) / vs - 0.5).floor()) as isize).min(dims as isize - 1);
            if i1 < 0 || j1 < 0 {
                continue;
            }
            for i in i0..=i1 as usize {
                for j in j0..=j1 as usize {
                    let (cu, cw) = (centre(i, u) + nu, centre(j, w) + nw);
                    // the point (cu, cw) in the triangle's projection: barycentric
                    let d = (p[1][u] - p[0][u]) * (p[2][w] - p[0][w]) - (p[2][u] - p[0][u]) * (p[1][w] - p[0][w]);
                    if d.abs() < 1e-20 {
                        continue;
                    }
                    let l1 = ((cu - p[0][u]) * (p[2][w] - p[0][w]) - (p[2][u] - p[0][u]) * (cw - p[0][w])) / d;
                    let l2 = ((p[1][u] - p[0][u]) * (cw - p[0][w]) - (cu - p[0][u]) * (p[1][w] - p[0][w])) / d;
                    if l1 >= 0.0 && l2 >= 0.0 && l1 + l2 <= 1.0 {
                        let a = p[0][axis] + l1 * (p[1][axis] - p[0][axis]) + l2 * (p[2][axis] - p[0][axis]);
                        cols[i * dims + j].push(a);
                    }
                }
            }
        }
        for i in 0..dims {
            for j in 0..dims {
                let col = &mut cols[i * dims + j];
                col.sort_by(f32::total_cmp);
                for k in 0..dims {
                    let ck = centre(k, axis);
                    if col.partition_point(|x| *x < ck) % 2 == 1 {
                        let mut c = [0usize; 3];
                        c[axis] = k;
                        c[u] = i;
                        c[w] = j;
                        votes[idx(c[0], c[1], c[2])] += 1;
                    }
                }
            }
        }
    }
    let data = surf.iter().zip(&votes).map(|(s, v)| *s || *v >= 2).collect();
    Vox { dims, translate: lo, scale: s, data }
}

/// The nearest point of each bone (segments [a | b], 6 floats a bone) to each vertex: (origins, distances), bone-major
/// (geometric_proc.compute_volumetric_geodesic.pts2line)
pub fn pts2line(pts: &[[f32; 3]], bones: &[[f32; 6]]) -> (Vec<[f32; 3]>, Vec<f32>) {
    let mut org = Vec::with_capacity(pts.len() * bones.len());
    let mut dist = Vec::with_capacity(pts.len() * bones.len());
    for b in bones {
        let (a, e) = ([b[0], b[1], b[2]], [b[3], b[4], b[5]]);
        let l2 = dot(sub(e, a), sub(e, a));
        for p in pts {
            let o = if l2.abs() < 1e-8 {
                a
            } else {
                let t = (dot(sub(*p, a), sub(e, a)) / l2).clamp(0.0, 1.0);
                add(a, scale(sub(e, a), t))
            };
            org.push(o);
            dist.push(length(sub(o, *p)));
        }
    }
    (org, dist)
}

/// Möller-Trumbore: the ray's parameter at the triangle, edges included (with a small tolerance)
fn ray_tri(o: [f32; 3], d: [f32; 3], a: [f32; 3], e1: [f32; 3], e2: [f32; 3]) -> Option<f32> {
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let t0 = sub(o, a);
    let u = dot(t0, p) * inv;
    const EPS: f32 = 1e-6;
    if !(-EPS..=1.0 + EPS).contains(&u) {
        return None;
    }
    let q = cross(t0, e1);
    let w = dot(d, q) * inv;
    if w < -EPS || u + w > 1.0 + EPS {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t >= -1e-6).then_some(t)
}

/// compute_volumetric_geodesic.calc_pts2bone_visible_mat: each ray (origin -> end) sees its end when the nearest
/// surface along it is the end itself (within 1e-4) or nothing is hit
pub fn visible(v: &[[f32; 3]], faces: &[[u32; 3]], origins: &[[f32; 3]], ends: &[[f32; 3]]) -> Vec<bool> {
    let tris: Vec<([f32; 3], [f32; 3], [f32; 3])> = faces.iter().map(|f| {
        let a = v[f[0] as usize];
        (a, sub(v[f[1] as usize], a), sub(v[f[2] as usize], a))
    }).collect();
    let mut out = vec![false; origins.len()];
    let nt = threads();
    let chunk = origins.len().div_ceil(nt).max(1);
    std::thread::scope(|sc| {
        for (t, part) in out.chunks_mut(chunk).enumerate() {
            let tris = &tris;
            sc.spawn(move || {
                for (k, o) in part.iter_mut().enumerate() {
                    let r = t * chunk + k;
                    let d = add(sub(ends[r], origins[r]), [1e-15; 3]);
                    let len = length(d);
                    let mut min_hit = f32::INFINITY;
                    for (a, e1, e2) in tris {
                        if let Some(h) = ray_tri(origins[r], d, *a, *e1, *e2) {
                            min_hit = min_hit.min(h.max(0.0) * len);
                        }
                    }
                    if !min_hit.is_finite() {
                        min_hit = len;
                    }
                    *o = (min_hit - len).abs() < 1e-4;
                }
            });
        }
    });
    out
}

/// numpy's percentile (linear interpolation)
pub fn percentile(v: &mut [f32], q: f64) -> f32 {
    v.sort_by(f32::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    let a = v[i] as f64;
    let b = v[(i + 1).min(v.len() - 1)] as f64;
    (a + (b - a) * f) as f32
}

/// quick_start.calc_geodesic_matrix (without its subsampling): [n, bones], the straight distance from each vertex to
/// each bone it sees, else the geodesic detour to the nearest vertex that sees it
pub fn volumetric_geodesic(v: &[[f32; 3]], faces: &[[u32; 3]], bones: &[[f32; 6]], g: &Geodesic) -> Vec<f32> {
    let (org, dist) = pts2line(v, bones);
    let ends: Vec<[f32; 3]> = (0..bones.len()).flat_map(|_| v.iter().copied()).collect();
    let vis = visible(v, faces, &org, &ends);
    geodesic_by_visibility(v.len(), bones.len(), &dist, vis, g)
}

/// The rest of `volumetric_geodesic` given the rays' visibility (bone-major, as `pts2line`'s distances)
pub fn geodesic_by_visibility(n: usize, nb: usize, dist: &[f32], mut vis: Vec<bool>, g: &Geodesic) -> Vec<f32> {
    // visible points much farther than most are not trusted
    for b in 0..nb {
        let mut d: Vec<f32> = (0..n).filter(|&i| vis[b * n + i]).map(|i| dist[b * n + i]).collect();
        if d.is_empty() {
            continue;
        }
        let th = percentile(&mut d, 15.0);
        for i in 0..n {
            if dist[b * n + i] > 1.3 * th {
                vis[b * n + i] = false;
            }
        }
    }
    let mut out = vec![0f32; n * nb];
    for b in 0..nb {
        let seen: Vec<usize> = (0..n).filter(|&i| vis[b * n + i]).collect();
        for i in 0..n {
            out[i * nb + b] = if seen.is_empty() || vis[b * n + i] { dist[b * n + i] } else { 0.0 };
        }
        if seen.is_empty() {
            continue;
        }
        for r in 0..n {
            if vis[b * n + r] {
                continue;
            }
            let (mut best, mut at) = (f32::INFINITY, seen[0]);
            for &s in &seen {
                let d = g.get(r, s);
                if d < best {
                    best = d;
                    at = s;
                }
            }
            out[r * nb + b] = if best.is_infinite() { 8.0 + dist[b * n + r] } else { best + dist[b * n + at] };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dijkstra_takes_the_short_way() {
        // 0 - 1 - 2 with 0.5 each, and 0 - 2 with 2
        let adj = vec![vec![(1, 0.5), (2, 2.0)], vec![(0, 0.5), (2, 0.5)], vec![(1, 0.5), (0, 2.0)], vec![]];
        let mut d = vec![0.0; 4];
        dijkstra(&adj, 0, &mut d);
        assert_eq!(&d[..3], &[0.0, 0.5, 1.0]);
        assert!(d[3].is_infinite());
    }

    fn cube() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let v: Vec<[f32; 3]> = (0..8).map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32]).collect();
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let mut f = Vec::new();
        for q in quads {
            f.push([q[0], q[1], q[2]]);
            f.push([q[0], q[2], q[3]]);
        }
        (v, f)
    }

    #[test]
    fn a_cube_voxelizes_solid() {
        let (v, f) = cube();
        let vox = voxelize(&v, &f, 16);
        assert_eq!(vox.scale, 1.0);
        assert!(vox.data.iter().all(|x| *x));
        assert!(vox.inside([0.5, 0.5, 0.5]));
        assert!(!vox.inside([2.0, 0.5, 0.5]));
    }

    #[test]
    fn rays_see_through_empty_space_only() {
        let (v, f) = cube();
        // from the centre to a corner: nothing in between
        let vis = visible(&v, &f, &[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], &[[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]]);
        assert!(vis[0]);
        // to a point outside: the wall is in between
        assert!(!vis[1]);
        let mut x = [3.0, 1.0, 2.0, 4.0];
        assert_eq!(percentile(&mut x, 50.0), 2.5);
    }

    #[test]
    fn geodesics_follow_the_surface() {
        // samples on a line, normals up: each linked to its neighbours
        let pts: Vec<[f32; 3]> = (0..20).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();
        let nrm = vec![[0.0, 1.0, 0.0]; 20];
        let g = surface_geodesic(&[[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]], &pts, &nrm);
        assert!((g.get(0, 1) - 1.9).abs() < 1e-4);
        let mut rng = Rng::new(1);
        let e = geo_edges(&g, &mut rng);
        assert!(e.is_empty());
        let (n, p, s) = normalize(&[[0.0, 2.0, 0.0], [2.0, 4.0, 1.0]]);
        assert_eq!((p, s), ([1.0, 2.0, 0.5], 0.5));
        assert_eq!(n[1], [0.5, 1.0, 0.25]);
    }

    #[test]
    fn poisson_samples_cover_the_surface() {
        let (v, f) = cube();
        let n = vertex_normals(&v, &f);
        let (p, q) = poisson_disk(&v, &n, &f, 200, &mut Rng::new(3));
        assert_eq!(p.len(), 200);
        assert_eq!(q.len(), 200);
        // spread: no two samples very close
        let mut min = f32::INFINITY;
        for i in 0..p.len() {
            for j in 0..i {
                min = min.min(length(sub(p[i], p[j])));
            }
        }
        assert!(min > 0.01, "{min}");
    }
}
