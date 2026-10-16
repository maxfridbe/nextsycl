//! The CPU tools a 3D pipeline runs around its models: `weld` (one vertex a position), `decimate` (quadric error
//! edge collapse to a triangle budget), `remesh` (isotropic: even triangles of a target edge length) and `uv`
//! (a fresh atlas: charts by normal, unfolded and packed, the old texture baked across).

pub mod closest;
pub mod decimate;
pub mod remesh;
pub mod uv;

use std::collections::HashMap;

use crate::Mesh;

pub use decimate::decimate;
pub use remesh::remesh;
pub use uv::{bake, unwrap};

/// One vertex a position (within `eps`, on a grid of that size): the triangles kept, the ones that fold to a line
/// or a point dropped; normals, UVs and the skin go (they belong to the seams welded away), the first vertex's
/// colour stays. Returns the map from old vertices to new.
pub fn weld(m: &Mesh, eps: f32) -> (Mesh, Vec<u32>) {
    let q = |x: f32| (x / eps.max(1e-12)).round() as i64;
    let mut at: HashMap<[i64; 3], u32> = HashMap::new();
    let mut out = Mesh { material: m.material.clone(), ..Default::default() };
    let mut map = Vec::with_capacity(m.positions.len());
    for (i, p) in m.positions.iter().enumerate() {
        let k = [q(p[0]), q(p[1]), q(p[2])];
        let id = *at.entry(k).or_insert_with(|| {
            out.positions.push(*p);
            if let Some(c) = m.colors.get(i) {
                out.colors.push(*c);
            }
            (out.positions.len() - 1) as u32
        });
        map.push(id);
    }
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]);
        if a != b && b != c && a != c {
            out.indices.extend([a, b, c]);
        }
    }
    (out, map)
}

/// Triangles' edges with how many triangles share each (1: a boundary)
pub fn edge_counts(indices: &[u32]) -> HashMap<(u32, u32), u32> {
    let mut e = HashMap::new();
    for t in indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *e.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    e
}

/// Drops vertices no triangle uses (and remaps the indices); every per-vertex attribute follows
pub fn compact(m: &mut Mesh) {
    let mut used = vec![u32::MAX; m.positions.len()];
    let mut n = 0u32;
    for &i in &m.indices {
        if used[i as usize] == u32::MAX {
            used[i as usize] = n;
            n += 1;
        }
    }
    if n as usize == m.positions.len() && used.iter().enumerate().all(|(i, &u)| u as usize == i) {
        return;
    }
    fn keep<T: Copy>(v: &mut Vec<T>, used: &[u32], n: u32) {
        if v.is_empty() {
            return;
        }
        let mut out = vec![v[0]; n as usize];
        for (i, &u) in used.iter().enumerate() {
            if u != u32::MAX {
                out[u as usize] = v[i];
            }
        }
        *v = out;
    }
    keep(&mut m.positions, &used, n);
    keep(&mut m.normals, &used, n);
    keep(&mut m.uvs, &used, n);
    keep(&mut m.colors, &used, n);
    if let Some(s) = &mut m.skin {
        keep(&mut s.weights, &used, n);
    }
    if let Some(a) = &mut m.animation {
        for f in &mut a.frames {
            keep(f, &used, n);
        }
    }
    for i in &mut m.indices {
        *i = used[*i as usize];
    }
}

/// The triangles wound consistently, outward: across every edge two triangles share (a manifold edge) the
/// neighbour is flipped to run the edge the other way, region by region from a seed; then each connected region whose
/// signed volume is negative is turned inside out. For meshes whose winding carries no meaning (a dual grid's quads,
/// marching cubes without signs). Returns how many triangles were flipped.
pub fn orient(m: &mut Mesh) -> usize {
    let nt = m.indices.len() / 3;
    if nt == 0 {
        return 0;
    }
    // every edge as (undirected key, triangle, forward?), sorted: an edge of two entries is manifold
    let mut e: Vec<(u64, u32, bool)> = Vec::with_capacity(nt * 3);
    for t in 0..nt {
        for k in 0..3 {
            let (a, b) = (m.indices[t * 3 + k], m.indices[t * 3 + (k + 1) % 3]);
            e.push((((a.min(b) as u64) << 32) | a.max(b) as u64, t as u32, a < b));
        }
    }
    e.sort_unstable_by_key(|x| x.0);
    // each triangle's neighbours across its manifold edges, and whether the pair runs the edge the same way
    let mut adj: Vec<Vec<(u32, bool)>> = vec![Vec::new(); nt];
    let mut i = 0;
    while i < e.len() {
        let mut j = i + 1;
        while j < e.len() && e[j].0 == e[i].0 {
            j += 1;
        }
        if j - i == 2 {
            let (x, y) = (e[i], e[i + 1]);
            let same = x.2 == y.2;
            adj[x.1 as usize].push((y.1, same));
            adj[y.1 as usize].push((x.1, same));
        }
        i = j;
    }
    drop(e);
    let mut flip = vec![false; nt];
    let mut seen = vec![false; nt];
    let mut stack = Vec::new();
    let pos = |i: u32| {
        let q = m.positions[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    for seed in 0..nt {
        if seen[seed] {
            continue;
        }
        let mut region = vec![seed as u32];
        seen[seed] = true;
        stack.push(seed as u32);
        while let Some(t) = stack.pop() {
            for &(n, same) in &adj[t as usize] {
                if !seen[n as usize] {
                    seen[n as usize] = true;
                    // the same direction along a shared edge means opposite windings: flip the neighbour
                    flip[n as usize] = flip[t as usize] ^ same;
                    region.push(n);
                    stack.push(n);
                }
            }
        }
        // the region's signed volume as it would be wound
        let mut vol = 0f64;
        for &t in &region {
            let t = t as usize;
            let mut v = [m.indices[t * 3], m.indices[t * 3 + 1], m.indices[t * 3 + 2]];
            if flip[t] {
                v.swap(1, 2);
            }
            let (a, b, c) = (pos(v[0]), pos(v[1]), pos(v[2]));
            vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        if vol < 0.0 {
            for &t in &region {
                flip[t as usize] = !flip[t as usize];
            }
        }
    }
    let mut flipped = 0;
    for (t, f) in flip.iter().enumerate() {
        if *f {
            m.indices.swap(t * 3 + 1, t * 3 + 2);
            flipped += 1;
        }
    }
    flipped
}

#[cfg(test)]
pub(crate) mod testing {
    use crate::Mesh;

    /// A UV sphere: `rings` x `segments`, unit radius, closed (poles welded), outward winding
    pub fn sphere(rings: usize, segments: usize) -> Mesh {
        let mut m = Mesh::default();
        m.positions.push([0.0, 1.0, 0.0]);
        for r in 1..rings {
            let th = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..segments {
                let ph = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
                m.positions.push([th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]);
            }
        }
        m.positions.push([0.0, -1.0, 0.0]);
        let ring = |r: usize, s: usize| (1 + (r - 1) * segments + s % segments) as u32;
        let south = (m.positions.len() - 1) as u32;
        for s in 0..segments {
            m.indices.extend([0, ring(1, s + 1), ring(1, s)]);
            m.indices.extend([south, ring(rings - 1, s), ring(rings - 1, s + 1)]);
        }
        for r in 1..rings - 1 {
            for s in 0..segments {
                let (a, b, c, d) = (ring(r, s), ring(r, s + 1), ring(r + 1, s), ring(r + 1, s + 1));
                m.indices.extend([a, b, d, a, d, c]);
            }
        }
        m
    }

    /// Every edge shared by exactly two triangles, each way once (a closed, consistently wound surface)
    pub fn is_closed_manifold(m: &Mesh) -> bool {
        let mut dir = std::collections::HashMap::new();
        for t in m.indices.chunks_exact(3) {
            for k in 0..3 {
                *dir.entry((t[k], t[(k + 1) % 3])).or_insert(0u32) += 1;
            }
        }
        dir.iter().all(|(&(a, b), &n)| n == 1 && dir.get(&(b, a)) == Some(&1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weld_joins_split_corners() {
        let mut m = testing::sphere(8, 12);
        // split every triangle into its own corners
        let p: Vec<[f32; 3]> = m.indices.iter().map(|&i| m.positions[i as usize]).collect();
        m.indices = (0..p.len() as u32).collect();
        m.positions = p;
        let (w, map) = weld(&m, 1e-5);
        assert_eq!(w.positions.len(), testing::sphere(8, 12).positions.len());
        assert_eq!(map.len(), m.positions.len());
        assert!(testing::is_closed_manifold(&w));
    }

    #[test]
    fn orient_turns_scrambled_triangles_outward() {
        let good = testing::sphere(10, 16);
        let mut m = good.clone();
        // every third triangle turned around, then the whole surface inside out
        for t in (0..m.triangles()).step_by(3) {
            m.indices.swap(t * 3 + 1, t * 3 + 2);
        }
        for t in 0..m.triangles() {
            m.indices.swap(t * 3 + 1, t * 3 + 2);
        }
        orient(&mut m);
        assert!(testing::is_closed_manifold(&m));
        // as wound as the outward sphere, triangle by triangle
        assert_eq!(m.indices.chunks(3).zip(good.indices.chunks(3)).filter(|(a, b)| a != b).count(), 0);
    }

    #[test]
    fn compact_drops_unused() {
        let mut m = testing::sphere(6, 8);
        m.positions.push([9.0, 9.0, 9.0]);
        let n = m.positions.len();
        compact(&mut m);
        assert_eq!(m.positions.len(), n - 1);
        assert!(m.validate().is_ok());
    }
}
