//! Closest points on a triangle mesh: a uniform grid of the triangles' boxes, searched in growing shells around
//! the query until no nearer cell can hold a better triangle. What remesh projects onto and bake samples from.

use crate::{add, dot, scale, sub};

pub struct Closest<'m> {
    pos: &'m [[f32; 3]],
    tris: Vec<[u32; 3]>,
    lo: [f32; 3],
    cell: f32,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

/// The nearest point to `p` on triangle (a, b, c) and its barycentric weights
pub fn on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    // Ericson, Real-Time Collision Detection 5.1.5
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(sub(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (add(a, add(scale(ab, v), scale(ac, w))), [1.0 - v - w, v, w])
}

impl<'m> Closest<'m> {
    pub fn new(pos: &'m [[f32; 3]], indices: &[u32]) -> Closest<'m> {
        let tris: Vec<[u32; 3]> = indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in pos {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        if pos.is_empty() {
            lo = [0.0; 3];
            hi = [1.0; 3];
        }
        let ext = sub(hi, lo);
        let vol = ext.iter().map(|x| x.max(1e-6)).product::<f32>();
        // about 2 triangles a cell
        let cell = (vol / (tris.len().max(1) as f32 / 2.0)).cbrt().max(1e-6);
        let dims = [0, 1, 2].map(|c| ((ext[c] / cell).ceil() as usize).clamp(1, 512));
        let mut cells = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        let at = |x: f32, c: usize| (((x - lo[c]) / cell) as isize).clamp(0, dims[c] as isize - 1) as usize;
        for (i, t) in tris.iter().enumerate() {
            let ps = t.map(|v| pos[v as usize]);
            let l = [0, 1, 2].map(|c| at(ps.iter().map(|p| p[c]).fold(f32::INFINITY, f32::min), c));
            let h = [0, 1, 2].map(|c| at(ps.iter().map(|p| p[c]).fold(f32::NEG_INFINITY, f32::max), c));
            for z in l[2]..=h[2] {
                for y in l[1]..=h[1] {
                    for x in l[0]..=h[0] {
                        cells[(z * dims[1] + y) * dims[0] + x].push(i as u32);
                    }
                }
            }
        }
        Closest { pos, tris, lo, cell, dims, cells }
    }

    /// The nearest point on the mesh: (the point, its triangle, barycentric weights); None without triangles
    pub fn query(&self, p: [f32; 3]) -> Option<([f32; 3], usize, [f32; 3])> {
        if self.tris.is_empty() {
            return None;
        }
        let at = |c: usize| (((p[c] - self.lo[c]) / self.cell).floor() as isize).clamp(0, self.dims[c] as isize - 1);
        let home = [at(0), at(1), at(2)];
        let mut best: Option<(f32, [f32; 3], usize, [f32; 3])> = None;
        let maxr = *self.dims.iter().max().unwrap() as isize;
        for r in 0..=maxr {
            // a shell of cells r away; once the best is nearer than the shell's inner distance, stop
            if let Some(b) = &best {
                let inner = (r as f32 - 1.0).max(0.0) * self.cell;
                if b.0 <= inner * inner {
                    break;
                }
            }
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        let (x, y, z) = (home[0] + dx, home[1] + dy, home[2] + dz);
                        if x < 0 || y < 0 || z < 0 || x >= self.dims[0] as isize || y >= self.dims[1] as isize || z >= self.dims[2] as isize {
                            continue;
                        }
                        for &t in &self.cells[(z as usize * self.dims[1] + y as usize) * self.dims[0] + x as usize] {
                            let tr = self.tris[t as usize];
                            let (q, bc) = on_triangle(p, self.pos[tr[0] as usize], self.pos[tr[1] as usize], self.pos[tr[2] as usize]);
                            let d = dot(sub(q, p), sub(q, p));
                            if best.as_ref().is_none_or(|b| d < b.0) {
                                best = Some((d, q, t as usize, bc));
                            }
                        }
                    }
                }
            }
        }
        best.map(|b| (b.1, b.2, b.3))
    }

    pub fn triangle(&self, t: usize) -> [u32; 3] {
        self.tris[t]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_point_on_a_sphere() {
        let m = super::super::testing::sphere(20, 30);
        let c = Closest::new(&m.positions, &m.indices);
        for p in [[3.0, 0.1, 0.2], [0.0, -0.5, 0.0], [0.3, 0.3, -0.3]] {
            let (q, _, bc) = c.query(p).unwrap();
            assert!((crate::length(q) - 1.0).abs() < 0.02);
            // the same direction as p
            assert!(dot(crate::normalize(q), crate::normalize(p)) > 0.99);
            assert!((bc.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        }
    }
}
