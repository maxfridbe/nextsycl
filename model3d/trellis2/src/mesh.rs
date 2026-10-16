//! The flexible dual grid to a mesh (o_voxel's flexible_dual_grid_to_mesh, inference): each voxel is a vertex at
//! (coords + 2 sigmoid(h[0..3]) - 0.5) / res - 0.5; for each voxel and axis whose edge logit h[3 + axis] > 0, the
//! four voxels around that edge make a quad when all four exist, split on the diagonal whose split weights
//! (softplus(h[6])) multiply larger. The quads' winding carries no inside or outside, so the mesh is oriented after.

use nextsycl_model3d::mesh::{cross, dot, sub, tools, Mesh};

/// The voxels around each axis' edge, relative to the voxel (o_voxel's edge_neighbor_voxel_offset)
const QUAD: [[[i32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch's softplus (beta 1, threshold 20)
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// The dual grid's mesh: voxels `coords` at resolution `res`, their decoder outputs `h` [n, 7], the level's
/// neighbour map [n, 27] (the quads' neighbours are among its taps); unused vertices dropped, unoriented
pub fn dual_grid(coords: &[[i32; 3]], h: &[f32], nmap: &[i32], res: usize) -> Mesh {
    let n = coords.len();
    let vs = 1.0 / res as f32;
    let mut m = Mesh { positions: Vec::with_capacity(n), ..Default::default() };
    for (i, c) in coords.iter().enumerate() {
        let r = &h[i * 7..i * 7 + 3];
        m.positions.push([0, 1, 2].map(|a| (c[a] as f32 + (2.0 * sigmoid(r[a]) - 0.5)) * vs - 0.5));
    }
    let tap = |o: [i32; 3]| ((o[0] + 1) * 9 + (o[1] + 1) * 3 + (o[2] + 1)) as usize;
    let sw: Vec<f32> = (0..n).map(|i| softplus(h[i * 7 + 6])).collect();
    for i in 0..n {
        for (a, quad) in QUAD.iter().enumerate() {
            if h[i * 7 + 3 + a] <= 0.0 {
                continue;
            }
            let q = quad.map(|o| nmap[i * 27 + tap(o)]);
            if q.iter().any(|v| *v < 0) {
                continue;
            }
            let q = q.map(|v| v as u32);
            let s02 = sw[q[0] as usize] * sw[q[2] as usize];
            let s13 = sw[q[1] as usize] * sw[q[3] as usize];
            if s02 > s13 {
                m.indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
            } else {
                m.indices.extend([q[0], q[1], q[3], q[3], q[1], q[2]]);
            }
        }
    }
    tools::compact(&mut m);
    m
}

/// Vertex clustering: the vertices of each cube of side `cell` merged at their mean, the triangles that fold dropped
/// (and repeats of one) - a fast first cut of a dense mesh before the quadric decimation, which then starts from a
/// few times its budget instead of millions of triangles. Windings are kept.
pub fn cluster(m: &Mesh, cell: f32) -> Mesh {
    use std::collections::HashMap;
    let q = |v: f32| (v / cell).floor() as i64;
    let mut at: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut sum: Vec<[f64; 4]> = Vec::new();
    let map: Vec<u32> = m.positions.iter().map(|p| {
        let k = (q(p[0]), q(p[1]), q(p[2]));
        let id = *at.entry(k).or_insert_with(|| {
            sum.push([0.0; 4]);
            (sum.len() - 1) as u32
        });
        let s = &mut sum[id as usize];
        s[0] += p[0] as f64;
        s[1] += p[1] as f64;
        s[2] += p[2] as f64;
        s[3] += 1.0;
        id
    }).collect();
    let mut out = Mesh { positions: sum.iter().map(|s| [(s[0] / s[3]) as f32, (s[1] / s[3]) as f32, (s[2] / s[3]) as f32]).collect(), ..Default::default() };
    let mut seen = std::collections::HashSet::new();
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]);
        if a == b || b == c || a == c {
            continue;
        }
        // one triangle a vertex triple, whatever its rotation or winding
        let mut k = [a, b, c];
        k.sort_unstable();
        if seen.insert(k) {
            out.indices.extend([a, b, c]);
        }
    }
    tools::compact(&mut out);
    out
}

/// The simplified mesh's small open patches turned to agree with what surrounds them. `tools::orient` joins
/// triangles across manifold edges and turns each such region by its signed volume, which means nothing for the small
/// open patches the clustering and the decimation leave between non-manifold edges: one triangle in a few hundred
/// came out facing inward (holes in a back-face-culled view, a normal pointing in for the bake's texel search). Here
/// each open region smaller than a hundredth of the mesh, the larger first, turns by a vote of its triangles: does
/// each normal agree with the normals of the triangles of other regions it shares a vertex with (area-weighted)? The
/// large and the closed regions keep their windings. Returns the triangles flipped.
#[allow(dead_code)]
pub fn orient_patches(m: &mut Mesh) -> usize {
    let nt = m.indices.len() / 3;
    if nt == 0 {
        return 0;
    }
    // the regions: triangles joined across edges two triangles share running opposite ways (consistent windings); a
    // region with any other edge (open, non-manifold or inconsistent) is open
    let mut e: Vec<(u64, u32, bool)> = Vec::with_capacity(nt * 3);
    for t in 0..nt {
        for k in 0..3 {
            let (a, b) = (m.indices[t * 3 + k], m.indices[t * 3 + (k + 1) % 3]);
            e.push((((a.min(b) as u64) << 32) | a.max(b) as u64, t as u32, a < b));
        }
    }
    e.sort_unstable_by_key(|x| x.0);
    let mut parent: Vec<u32> = (0..nt as u32).collect();
    fn root(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut rim = vec![false; nt];
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j].0 == e[i].0 {
            j += 1;
        }
        if j - i == 2 && e[i].2 != e[i + 1].2 {
            let (a, b) = (root(&mut parent, e[i].1), root(&mut parent, e[i + 1].1));
            if a != b {
                parent[a as usize] = b;
            }
        } else {
            for x in &e[i..j] {
                rim[x.1 as usize] = true;
            }
        }
        i = j;
    }
    drop(e);
    let region: Vec<u32> = (0..nt as u32).map(|t| root(&mut parent, t)).collect();
    let mut size = vec![0u32; nt];
    let mut open = vec![false; nt];
    for t in 0..nt {
        size[region[t] as usize] += 1;
        open[region[t] as usize] |= rim[t];
    }
    let small = (nt / 100).max(1) as u32;
    let mut patches: Vec<u32> = (0..nt as u32).filter(|&r| size[r as usize] > 0 && size[r as usize] < small && open[r as usize]).collect();
    if patches.is_empty() {
        return 0;
    }
    patches.sort_by_key(|&r| std::cmp::Reverse(size[r as usize]));
    // each patch's triangles, and each vertex's triangles
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); nt];
    for t in 0..nt {
        let r = region[t] as usize;
        if size[r] < small && open[r] {
            members[r].push(t as u32);
        }
    }
    let nv = m.positions.len();
    let mut start = vec![0u32; nv + 1];
    for &v in &m.indices {
        start[v as usize + 1] += 1;
    }
    for v in 0..nv {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut vt = vec![0u32; m.indices.len()];
    for (k, &v) in m.indices.iter().enumerate() {
        vt[fill[v as usize] as usize] = (k / 3) as u32;
        fill[v as usize] += 1;
    }
    let normal = |m: &Mesh, t: usize| {
        let p = [0, 1, 2].map(|k| m.positions[m.indices[t * 3 + k] as usize]);
        cross(sub(p[1], p[0]), sub(p[2], p[0]))
    };
    let mut flipped = 0;
    for r in patches {
        let mut vote = 0f64;
        for &t in &members[r as usize] {
            let n = normal(m, t as usize);
            for k in 0..3 {
                let v = m.indices[t as usize * 3 + k] as usize;
                for &u in &vt[start[v] as usize..start[v + 1] as usize] {
                    if region[u as usize] != r {
                        vote += dot(n, normal(m, u as usize)) as f64;
                    }
                }
            }
        }
        if vote < 0.0 {
            for &t in &members[r as usize] {
                m.indices.swap(t as usize * 3 + 1, t as usize * 3 + 2);
                flipped += 1;
            }
        }
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::neighbours;

    /// A sphere of radius r (voxels) on a res grid as a dual grid: the voxels around the surface, each vertex at its
    /// centre, an axis' flag where the sphere's surface crosses that voxel's edge
    fn sphere_grid(res: i32, r: f32) -> (Vec<[i32; 3]>, Vec<f32>) {
        let c = res as f32 / 2.0;
        let inside = |p: [f32; 3]| ((p[0] - c).powi(2) + (p[1] - c).powi(2) + (p[2] - c).powi(2)).sqrt() < r;
        let mut coords = Vec::new();
        let mut h = Vec::new();
        for x in 0..res {
            for y in 0..res {
                for z in 0..res {
                    // the voxel's edges' far corners (y + 1, z + 1) along x, and so on
                    let corner = |dx: i32, dy: i32, dz: i32| inside([(x + dx) as f32, (y + dy) as f32, (z + dz) as f32]);
                    let flags = [corner(0, 1, 1) != corner(1, 1, 1), corner(1, 0, 1) != corner(1, 1, 1), corner(1, 1, 0) != corner(1, 1, 1)];
                    let near = (0..8).map(|k| corner(k & 1, (k >> 1) & 1, k >> 2)).collect::<Vec<_>>();
                    if near.iter().all(|v| *v) || near.iter().all(|v| !*v) {
                        continue;
                    }
                    coords.push([x, y, z]);
                    // vertex at the centre (sigmoid(0) = 0.5 -> 0.5), flags as logits, equal split weights
                    h.extend([0.0, 0.0, 0.0]);
                    h.extend(flags.map(|f| if f { 1.0 } else { -1.0 }));
                    h.push(0.0);
                }
            }
        }
        (coords, h)
    }

    #[test]
    fn a_voxel_sphere_meshes_closed_and_orients_outward() {
        let (coords, h) = sphere_grid(24, 7.3);
        let nm = neighbours(&coords);
        let mut m = dual_grid(&coords, &h, &nm, 24);
        assert!(m.triangles() > 500, "{} triangles", m.triangles());
        assert!(m.validate().is_ok());
        // every edge shared by exactly two triangles
        let e = tools::edge_counts(&m.indices);
        assert!(e.values().all(|c| *c == 2), "an open or non-manifold edge");
        tools::orient(&mut m);
        // the volume of the oriented mesh: positive, near the sphere's (r / res)^3 4/3 pi
        let mut vol = 0f64;
        for t in m.indices.chunks(3) {
            let p = |i: u32| m.positions[i as usize].map(|v| v as f64);
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
        }
        let want = 4.0 / 3.0 * std::f64::consts::PI * (7.3f64 / 24.0).powi(3);
        assert!(vol > 0.0 && (vol - want).abs() < 0.25 * want, "volume {vol} for {want}");
    }

    #[test]
    fn a_quad_splits_on_the_heavier_diagonal() {
        // four voxels around one x edge
        let coords = vec![[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]];
        let mut h = vec![0f32; 4 * 7];
        h[3] = 1.0;                      // voxel 0's x edge
        h[2 * 7 + 6] = 5.0;              // voxel 2 weighs most: diagonal 0-2
        // the triangles as voxel numbers (the vertices are compacted in the order the triangles use them)
        let voxels = |m: &Mesh| -> Vec<usize> {
            let at = |c: [i32; 3]| c.map(|v| (v as f32 + 0.5) * 0.5 - 0.5);
            m.indices.iter().map(|&i| coords.iter().position(|c| at(*c) == m.positions[i as usize]).unwrap()).collect()
        };
        let m = dual_grid(&coords, &h, &neighbours(&coords), 2);
        assert_eq!(voxels(&m), vec![0, 1, 2, 0, 2, 3]);
        h[2 * 7 + 6] = -5.0;
        h[7 + 6] = 5.0;                  // voxel 1: diagonal 1-3
        let m = dual_grid(&coords, &h, &neighbours(&coords), 2);
        assert_eq!(voxels(&m), vec![0, 1, 3, 3, 1, 2]);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p nextsycl-model3d-trellis2 -- --ignored --nocapture decimation_time`
    #[test]
    #[ignore]
    fn decimation_time() {
        let res: i32 = 900;
        let c = res as f32 / 2.0;
        let r = res as f32 * 0.4;
        let mut coords = Vec::new();
        let mut h = Vec::new();
        let inside = |p: [f32; 3]| {
            let d = ((p[0] - c).powi(2) + (p[1] - c).powi(2) + (p[2] - c).powi(2)).sqrt();
            d < r + 3.0 * (p[0] * 0.3).sin()
        };
        for x in 0..res {
            for y in 0..res {
                for z in 0..res {
                    let corner = |dx: i32, dy: i32, dz: i32| inside([(x + dx) as f32, (y + dy) as f32, (z + dz) as f32]);
                    let near: Vec<bool> = (0..8).map(|k| corner(k & 1, (k >> 1) & 1, k >> 2)).collect();
                    if near.iter().all(|v| *v) || near.iter().all(|v| !*v) {
                        continue;
                    }
                    coords.push([x, y, z]);
                    let flags = [corner(0, 1, 1) != corner(1, 1, 1), corner(1, 0, 1) != corner(1, 1, 1), corner(1, 1, 0) != corner(1, 1, 1)];
                    h.extend([0.0, 0.0, 0.0]);
                    h.extend(flags.map(|f| if f { 1.0 } else { -1.0 }));
                    h.push(0.0);
                }
            }
        }
        let nm = crate::sparse::neighbours(&coords);
        let t = std::time::Instant::now();
        let mut m = dual_grid(&coords, &h, &nm, res as usize);
        eprintln!("dual grid: {} triangles, {:.2} s", m.triangles(), t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        tools::orient(&mut m);
        eprintln!("orient {:.2} s", t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let cell = (m.positions.len() as f32 / 600_000.0).sqrt() / res as f32;
        let m = cluster(&m, cell);
        eprintln!("cluster -> {} triangles, {:.2} s", m.triangles(), t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let d = tools::decimate(&m, 300_000);
        eprintln!("decimate -> {} triangles, {:.2} s", d.triangles(), t.elapsed().as_secs_f64());
    }
}
