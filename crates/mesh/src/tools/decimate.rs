//! Decimation by quadric error (Garland & Heckbert): every edge's collapse priced by the summed squared distances to
//! its vertices' planes, the cheapest collapsed first to the point that minimises it, until the triangle budget is
//! met. Boundary edges carry a perpendicular plane (the outline keeps its shape); a collapse that would flip a
//! triangle or pinch the surface (a link condition) is refused. Positions and colours are kept; UVs, normals and a
//! skin are dropped (re-UV with `uv`, normals with `Mesh::compute_normals`).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::{cross, dot, normalize, sub, Mesh};

#[derive(Clone, Copy, Default)]
struct Q([f64; 10]);

impl Q {
    fn plane(n: [f64; 3], d: f64, w: f64) -> Q {
        let [a, b, c] = n;
        Q([a * a * w, a * b * w, a * c * w, a * d * w, b * b * w, b * c * w, b * d * w, c * c * w, c * d * w, d * d * w])
    }
    fn add(self, o: Q) -> Q {
        let mut r = self;
        for i in 0..10 {
            r.0[i] += o.0[i];
        }
        r
    }
    fn eval(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let [x, y, z] = p;
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y
            + q[7] * z * z + 2.0 * q[8] * z + q[9]
    }
    /// The point minimising it, when its 3x3 part is well conditioned
    fn optimum(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        let m = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let scale = m[0][0].abs() + m[1][1].abs() + m[2][2].abs();
        if det.abs() < 1e-9 * scale.powi(3).max(1e-30) {
            return None;
        }
        let b = [-q[3], -q[6], -q[8]];
        let solve = |col: usize| {
            let mut a = m;
            for r in 0..3 {
                a[r][col] = b[r];
            }
            (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])) / det
        };
        Some([solve(0), solve(1), solve(2)])
    }
}

struct Cand {
    cost: f64,
    a: u32,
    b: u32,
    stamp: (u32, u32),
    to: [f32; 3],
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.total_cmp(&self.cost) // a min-heap
    }
}

fn f64v(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

/// The mesh reduced to at most `target` triangles (welded first, so split seams do not stop it)
pub fn decimate(input: &Mesh, target: usize) -> Mesh {
    let (bb_lo, bb_hi) = input.bounds();
    let diag = crate::length(sub(bb_hi, bb_lo)).max(1e-6);
    let (mut m, _) = super::weld(input, diag * 1e-6);
    if m.triangles() <= target {
        super::compact(&mut m);
        return m;
    }
    let nv = m.positions.len();
    let mut tris: Vec<[u32; 3]> = m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    let mut alive = vec![true; tris.len()];
    let mut vt: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (i, t) in tris.iter().enumerate() {
        for &v in t {
            vt[v as usize].push(i as u32);
        }
    }
    let mut q = vec![Q::default(); nv];
    let face_n = |p: &[[f32; 3]], t: [u32; 3]| cross(sub(p[t[1] as usize], p[t[0] as usize]), sub(p[t[2] as usize], p[t[0] as usize]));
    for t in &tris {
        let n = face_n(&m.positions, *t);
        let area = crate::length(n) as f64 * 0.5;
        if area <= 0.0 {
            continue;
        }
        let nn = f64v(normalize(n));
        let d = -(nn[0] * m.positions[t[0] as usize][0] as f64 + nn[1] * m.positions[t[0] as usize][1] as f64 + nn[2] * m.positions[t[0] as usize][2] as f64);
        let pq = Q::plane(nn, d, area);
        for &v in t {
            q[v as usize] = q[v as usize].add(pq);
        }
    }
    // the outline: a plane through each boundary edge, perpendicular to its triangle, weighted heavily
    let edges = super::edge_counts(&m.indices);
    for t in &tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if edges.get(&(a.min(b), a.max(b))) == Some(&1) {
                let e = sub(m.positions[b as usize], m.positions[a as usize]);
                let n = normalize(cross(e, face_n(&m.positions, *t)));
                let nn = f64v(n);
                let pa = f64v(m.positions[a as usize]);
                let d = -(nn[0] * pa[0] + nn[1] * pa[1] + nn[2] * pa[2]);
                let w = (crate::length(e) as f64).powi(2) * 1000.0;
                let pq = Q::plane(nn, d, w);
                q[a as usize] = q[a as usize].add(pq);
                q[b as usize] = q[b as usize].add(pq);
            }
        }
    }
    let mut stamp = vec![0u32; nv];
    let mut gone = vec![false; nv];
    let price = |q: &[Q], p: &[[f32; 3]], a: u32, b: u32| -> (f64, [f32; 3]) {
        let s = q[a as usize].add(q[b as usize]);
        let (pa, pb) = (f64v(p[a as usize]), f64v(p[b as usize]));
        let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
        let mut best = (s.eval(mid), mid);
        for c in [pa, pb].into_iter().chain(s.optimum()) {
            // the optimum only when it stays near the edge (a far solution is a degenerate quadric's)
            let far = (0..3).map(|i| (c[i] - mid[i]).powi(2)).sum::<f64>().sqrt() > 2.0 * (0..3).map(|i| (pa[i] - pb[i]).powi(2)).sum::<f64>().sqrt() + 1e-12;
            if far {
                continue;
            }
            let e = s.eval(c);
            if e < best.0 {
                best = (e, c);
            }
        }
        (best.0.max(0.0), [best.1[0] as f32, best.1[1] as f32, best.1[2] as f32])
    };
    let mut heap = BinaryHeap::new();
    for &(a, b) in edges.keys() {
        let (cost, to) = price(&q, &m.positions, a, b);
        heap.push(Cand { cost, a, b, stamp: (0, 0), to });
    }
    let mut live = tris.len();
    let neighbours = |vt: &[Vec<u32>], tris: &[[u32; 3]], alive: &[bool], v: u32| -> HashSet<u32> {
        vt[v as usize].iter().filter(|&&t| alive[t as usize]).flat_map(|&t| tris[t as usize]).filter(|&x| x != v).collect()
    };
    while live > target {
        let Some(c) = heap.pop() else { break };
        let (a, b) = (c.a, c.b);
        if gone[a as usize] || gone[b as usize] || stamp[a as usize] != c.stamp.0 || stamp[b as usize] != c.stamp.1 {
            continue;
        }
        let na = neighbours(&vt, &tris, &alive, a);
        if !na.contains(&b) {
            continue;
        }
        let nb = neighbours(&vt, &tris, &alive, b);
        // the link condition: the edge's two ends share exactly the vertices opposite it (2 inside, 1 on a boundary)
        let shared = na.intersection(&nb).count();
        let opposite = vt[a as usize].iter().filter(|&&t| alive[t as usize] && tris[t as usize].contains(&b)).count();
        if shared != opposite {
            continue;
        }
        // no triangle around either end may flip or collapse to a sliver
        let mut ok = true;
        for &v in &[a, b] {
            for &t in &vt[v as usize] {
                let tr = tris[t as usize];
                if !alive[t as usize] || (tr.contains(&a) && tr.contains(&b)) {
                    continue;
                }
                let before = face_n(&m.positions, tr);
                let moved = tr.map(|x| if x == v { 0 } else { 1 });
                let pts: Vec<[f32; 3]> = tr.iter().zip(moved).map(|(&x, k)| if k == 0 { c.to } else { m.positions[x as usize] }).collect();
                let after = cross(sub(pts[1], pts[0]), sub(pts[2], pts[0]));
                let (lb, la) = (crate::length(before), crate::length(after));
                if la < 1e-12 * diag * diag || dot(before, after) < 0.2 * lb * la {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        if !ok {
            continue;
        }
        // collapse b into a
        m.positions[a as usize] = c.to;
        q[a as usize] = q[a as usize].add(q[b as usize]);
        gone[b as usize] = true;
        let moved: Vec<u32> = std::mem::take(&mut vt[b as usize]);
        for t in moved {
            if !alive[t as usize] {
                continue;
            }
            let tr = &mut tris[t as usize];
            if tr.contains(&a) {
                alive[t as usize] = false;
                live -= 1;
            } else {
                for x in tr.iter_mut() {
                    if *x == b {
                        *x = a;
                    }
                }
                vt[a as usize].push(t);
            }
        }
        vt[a as usize].retain(|&t| alive[t as usize]);
        stamp[a as usize] += 1;
        for n in neighbours(&vt, &tris, &alive, a) {
            let (cost, to) = price(&q, &m.positions, a, n);
            heap.push(Cand { cost, a, b: n, stamp: (stamp[a as usize], stamp[n as usize]), to });
        }
    }
    m.indices = tris.iter().zip(&alive).filter(|(_, &l)| l).flat_map(|(t, _)| *t).collect();
    super::compact(&mut m);
    m
}

#[cfg(test)]
mod tests {
    use super::super::testing::{is_closed_manifold, sphere};
    use super::*;

    #[test]
    fn a_sphere_halves_and_stays_a_sphere() {
        let m = sphere(40, 60);
        let before = m.triangles();
        let d = decimate(&m, before / 10);
        assert!(d.triangles() <= before / 10, "{} triangles", d.triangles());
        assert!(d.triangles() > before / 20);
        assert!(is_closed_manifold(&d), "the result has holes or folds");
        // still on the sphere: every vertex near radius 1
        let worst = d.positions.iter().map(|p| (crate::length(*p) - 1.0).abs()).fold(0f32, f32::max);
        assert!(worst < 0.05, "a vertex {worst} off the surface");
    }

    #[test]
    fn under_budget_is_left_alone() {
        let m = sphere(6, 8);
        assert_eq!(decimate(&m, 10_000).triangles(), m.triangles());
    }
}
