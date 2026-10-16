//! Marching cubes: the surface where a sampled field crosses a level, as a welded, consistently wound triangle mesh
//! (what a shape generator's occupancy or SDF grid becomes), and `drop_floaters` to clear the small pieces such a
//! surface carries.
//!
//! The case table is not copied from a paper: each of the 256 corner patterns is triangulated when first needed by
//! walking the cube's faces. On a face, every run of inside corners (going round it counter-clockwise, seen from
//! outside) is cut off by one segment, from the edge where the run begins to the edge where it ends - so an ambiguous
//! face (inside corners on a diagonal) always separates them. Both cubes sharing a face see the same corners and so
//! the same segments (in opposite directions), which makes the surface watertight and its winding consistent; each
//! cube's segments chain into closed loops, fanned into triangles that face away from the inside. A fan's chords must
//! not lie in a face of the cube (the neighbour across it could draw the same chord: an edge of four triangles), so
//! the fan starts at a vertex whose chords all cross the cube's inside, or, where no vertex does, at a new vertex in
//! the loop's middle.

use std::sync::OnceLock;

use crate::Mesh;

/// Corner i of a cell sits at (i & 1, i >> 1 & 1, i >> 2 & 1)
fn corner(i: usize) -> [usize; 3] {
    [i & 1, (i >> 1) & 1, (i >> 2) & 1]
}

/// The 12 edges as (corner, corner, axis), the first corner the lower one along the axis
fn edges() -> [(usize, usize, usize); 12] {
    let mut e = [(0, 0, 0); 12];
    let mut n = 0;
    for axis in 0..3 {
        for a in 0..8 {
            if (a >> axis) & 1 == 0 {
                e[n] = (a, a | (1 << axis), axis);
                n += 1;
            }
        }
    }
    e
}

/// One loop of crossed edges, in order, and the vertex its fan starts at (None: a new vertex in its middle)
struct Loop {
    edges: Vec<u8>,
    apex: Option<usize>,
}

/// Per corner pattern (bit i: corner i inside), its loops
fn table() -> &'static Vec<Vec<Loop>> {
    static T: OnceLock<Vec<Vec<Loop>>> = OnceLock::new();
    T.get_or_init(|| {
        let es = edges();
        let edge_of = |a: usize, b: usize| es.iter().position(|&(x, y, _)| (x, y) == (a.min(b), a.max(b))).expect("an edge") as u8;
        // the six faces' corners, counter-clockwise seen from outside
        let mut faces = Vec::new();
        for axis in 0..3 {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            for side in 0..2 {
                let at = |du: usize, dv: usize| (side << axis) | (du << u) | (dv << v);
                let mut cyc = [at(0, 0), at(1, 0), at(1, 1), at(0, 1)];
                // the cycle turns about e_u x e_v = +e_axis (u, v cyclic after axis); outward is -e_axis on side 0
                if side == 0 {
                    cyc.reverse();
                }
                faces.push(cyc);
            }
        }
        (0..256usize).map(|case| {
            let inside = |c: usize| (case >> c) & 1 == 1;
            // next[e]: the segment leaving crossing edge e (where it is entered) on its face
            let mut next = [u8::MAX; 12];
            for f in &faces {
                for k in 0..4 {
                    let (a, b) = (f[k], f[(k + 1) % 4]);
                    if inside(a) || !inside(b) {
                        continue;
                    }
                    // a run of inside corners begins at b: find where it ends
                    let mut j = (k + 1) % 4;
                    while inside(f[(j + 1) % 4]) {
                        j = (j + 1) % 4;
                    }
                    next[edge_of(a, b) as usize] = edge_of(f[j], f[(j + 1) % 4]);
                }
            }
            // two edges of one face
            let share = |a: u8, b: u8| {
                let (ea, eb) = (es[a as usize], es[b as usize]);
                faces.iter().any(|f| [ea.0, ea.1, eb.0, eb.1].iter().all(|c| f.contains(c)))
            };
            let mut loops = Vec::new();
            let mut seen = [false; 12];
            for start in 0..12 {
                if next[start] == u8::MAX || seen[start] {
                    continue;
                }
                let mut lp = vec![start as u8];
                seen[start] = true;
                let mut e = next[start] as usize;
                while e != start {
                    seen[e] = true;
                    lp.push(e as u8);
                    e = next[e] as usize;
                }
                let n = lp.len();
                let apex = (0..n).find(|&k| (2..n - 1).all(|d| !share(lp[k], lp[(k + d) % n])));
                loops.push(Loop { edges: lp, apex });
            }
            loops
        }).collect()
    })
}

/// The surface where `values` (a [nx, ny, nz] grid, x slowest, z fastest) crosses `level`, the inside where it is
/// above. Grid point (i, j, k) sits at origin + (i, j, k) * step. Triangles are counter-clockwise seen from outside,
/// vertices shared between the cells (one a crossed grid edge, placed on it by linear interpolation).
pub fn marching_cubes(values: &[f32], dims: [usize; 3], level: f32, origin: [f32; 3], step: f32) -> Mesh {
    let [nx, ny, nz] = dims;
    assert_eq!(values.len(), nx * ny * nz, "marching_cubes: {} values for {nx} x {ny} x {nz}", values.len());
    let mut m = Mesh::default();
    if nx < 2 || ny < 2 || nz < 2 {
        return m;
    }
    let t = table();
    let es = edges();
    let at = |x: usize, y: usize, z: usize| values[(x * ny + y) * nz + z];
    // the vertex on each grid edge, by its lower grid point: two planes of x (this cell layer's and the next)
    let plane = ny * nz * 3;
    let mut planes = [vec![u32::MAX; plane], vec![u32::MAX; plane]];
    let mut v = [0f32; 8];
    for x in 0..nx - 1 {
        for y in 0..ny - 1 {
            for z in 0..nz - 1 {
                let mut case = 0usize;
                for (i, vi) in v.iter_mut().enumerate() {
                    let c = corner(i);
                    *vi = at(x + c[0], y + c[1], z + c[2]);
                    if *vi > level {
                        case |= 1 << i;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                let mut ids = [u32::MAX; 12];
                for &e in t[case].iter().flat_map(|l| &l.edges) {
                    let e = e as usize;
                    if ids[e] != u32::MAX {
                        continue;
                    }
                    let (a, b, axis) = es[e];
                    let ca = corner(a);
                    let (p, slot) = (ca[0], ((y + ca[1]) * nz + z + ca[2]) * 3 + axis);
                    let id = planes[p][slot];
                    ids[e] = if id != u32::MAX {
                        id
                    } else {
                        let f = ((level - v[a]) / (v[b] - v[a])).clamp(0.0, 1.0);
                        let mut pos = [(x + ca[0]) as f32, (y + ca[1]) as f32, (z + ca[2]) as f32];
                        pos[axis] += f;
                        m.positions.push([origin[0] + pos[0] * step, origin[1] + pos[1] * step, origin[2] + pos[2] * step]);
                        let n = (m.positions.len() - 1) as u32;
                        planes[p][slot] = n;
                        n
                    };
                }
                for l in &t[case] {
                    let vs: Vec<u32> = l.edges.iter().map(|&e| ids[e as usize]).collect();
                    let n = vs.len();
                    match l.apex {
                        Some(k) => {
                            for d in 1..n - 1 {
                                m.indices.extend([vs[k], vs[(k + d) % n], vs[(k + d + 1) % n]]);
                            }
                        }
                        None => {
                            let mut c = [0f32; 3];
                            for &i in &vs {
                                c = crate::add(c, m.positions[i as usize]);
                            }
                            m.positions.push(crate::scale(c, 1.0 / n as f32));
                            let ci = (m.positions.len() - 1) as u32;
                            for d in 0..n {
                                m.indices.extend([ci, vs[d], vs[(d + 1) % n]]);
                            }
                        }
                    }
                }
            }
        }
        planes.swap(0, 1);
        planes[1].fill(u32::MAX);
    }
    m
}

/// The mesh without its small connected pieces: those with fewer than `ratio` of all the triangles (by shared
/// vertices); unused vertices dropped
pub fn drop_floaters(m: &Mesh, ratio: f32) -> Mesh {
    let n = m.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for t in m.indices.chunks_exact(3) {
        let a = find(&mut parent, t[0]);
        for &o in &t[1..] {
            let b = find(&mut parent, o);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut count = std::collections::HashMap::new();
    let roots: Vec<u32> = m.indices.chunks_exact(3).map(|t| find(&mut parent, t[0])).collect();
    for r in &roots {
        *count.entry(*r).or_insert(0usize) += 1;
    }
    let min = (ratio * m.triangles() as f32).ceil() as usize;
    let mut out = m.clone();
    out.indices = m.indices.chunks_exact(3).zip(&roots).filter(|(_, r)| count[*r] >= min).flat_map(|(t, _)| t.iter().copied()).collect();
    super::compact(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::testing::is_closed_manifold;
    use crate::{length, normalize, sub};

    fn field(n: usize, f: impl Fn([f32; 3]) -> f32) -> (Vec<f32>, [f32; 3], f32) {
        let (lo, step) = (-1.2f32, 2.4 / (n - 1) as f32);
        let mut v = Vec::with_capacity(n * n * n);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    v.push(f([lo + x as f32 * step, lo + y as f32 * step, lo + z as f32 * step]));
                }
            }
        }
        (v, [lo; 3], step)
    }

    #[test]
    fn every_case_closes_its_loops() {
        let t = table();
        assert!(t[0].is_empty() && t[255].is_empty());
        for (c, loops) in t.iter().enumerate().skip(1).take(254) {
            assert!(!loops.is_empty() && loops.iter().all(|l| l.edges.len() >= 3), "case {c}");
        }
        // one corner in: one triangle; its complement too
        assert_eq!(t[1].len(), 1);
        assert_eq!(t[1][0].edges.len(), 3);
        assert_eq!(t[254][0].edges.len(), 3);
    }

    #[test]
    fn a_sphere_is_a_closed_manifold_on_the_sphere() {
        let n = 40;
        let (v, o, s) = field(n, |p| 0.8 - length(p));
        let m = marching_cubes(&v, [n, n, n], 0.0, o, s);
        assert!(m.triangles() > 1000);
        assert!(m.validate().is_ok());
        assert!(is_closed_manifold(&m));
        // every vertex on the sphere, every triangle facing out
        for p in &m.positions {
            assert!((length(*p) - 0.8).abs() < 0.01, "{p:?}");
        }
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| m.positions[t[k] as usize]);
            let nrm = crate::cross(sub(b, a), sub(c, a));
            let centre = crate::scale(crate::add(crate::add(a, b), c), 1.0 / 3.0);
            assert!(crate::dot(nrm, normalize(centre)) > 0.0);
        }
        // Euler characteristic of a sphere
        let edges = crate::tools::edge_counts(&m.indices).len() as i64;
        assert_eq!(m.positions.len() as i64 - edges + m.triangles() as i64, 2);
    }

    #[test]
    fn a_noisy_field_stays_watertight() {
        // every ambiguous configuration shows up in a random field; the box's border kept outside
        let n = 24;
        let mut s = 12345u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut v = vec![-1f32; n * n * n];
        for x in 1..n - 1 {
            for y in 1..n - 1 {
                for z in 1..n - 1 {
                    v[(x * n + y) * n + z] = rnd();
                }
            }
        }
        let m = marching_cubes(&v, [n, n, n], 0.0, [0.0; 3], 1.0);
        assert!(m.triangles() > 1000);
        assert!(is_closed_manifold(&m));
    }

    #[test]
    fn floaters_go() {
        let n = 40;
        let (v, o, s) = field(n, |p| (0.6 - length(p)).max(0.12 - length(sub(p, [0.95, 0.95, 0.95]))));
        let m = marching_cubes(&v, [n, n, n], 0.0, o, s);
        let kept = drop_floaters(&m, 0.05);
        assert!(kept.triangles() < m.triangles());
        assert!(kept.positions.iter().all(|p| length(*p) < 0.7));
        assert!(is_closed_manifold(&kept));
    }
}
