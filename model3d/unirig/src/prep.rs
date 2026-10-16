//! The mesh as UniRig takes it (src/data/extract.py, augment.py, sampler.py): turned to Blender's frame (Z up), its
//! vertices merged as trimesh merges them (positions equal to 8 decimals), decimated past 50,000 triangles, vertex
//! normals weighted by the corners' angles and face normals (trimesh's), centred on its box and scaled into [-1, 1];
//! then sampled: up to 8,192 vertices at random and the rest of the points on the surface, triangles drawn by their
//! squared area (UniRig's `sample_surface` weighs them so), each point with a normal (its vertex's, or its face's).

use nextsycl_model3d::mesh::{self, tools, Mesh};

use crate::ops::Rng;

/// glTF's frame (Y up) to Blender's (Z up), as its importers turn it
pub fn to_blender(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[2], p[1]]
}

pub fn from_blender(p: [f32; 3]) -> [f32; 3] {
    [p[0], p[2], -p[1]]
}

/// The mesh the stages share (Blender's frame, not yet normalized)
pub struct Shape {
    pub v: Vec<[f32; 3]>,
    pub f: Vec<[u32; 3]>,
    pub vn: Vec<[f32; 3]>,
    pub fn_: Vec<[f32; 3]>,
    /// each input vertex's vertex here (None: dropped by a decimation - then by the nearest)
    pub map: Vec<u32>,
    pub decimated: bool,
}

/// Faces' normals (unit; a degenerate face: zeros, as trimesh's)
pub fn face_normals(v: &[[f32; 3]], f: &[[u32; 3]]) -> Vec<[f32; 3]> {
    f.iter().map(|t| {
        let n = mesh::cross(mesh::sub(v[t[1] as usize], v[t[0] as usize]), mesh::sub(v[t[2] as usize], v[t[0] as usize]));
        let l = mesh::length(n);
        if l > 0.0 { mesh::scale(n, 1.0 / l) } else { [0.0; 3] }
    }).collect()
}

/// Vertex normals: the faces' normals weighted by each corner's angle (trimesh's `weighted_vertex_normals`)
pub fn vertex_normals(v: &[[f32; 3]], f: &[[u32; 3]], fnrm: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f64; 3]; v.len()];
    for (t, n) in f.iter().zip(fnrm) {
        for k in 0..3 {
            let (a, b, c) = (v[t[k] as usize], v[t[(k + 1) % 3] as usize], v[t[(k + 2) % 3] as usize]);
            let (u, w) = (mesh::normalize(mesh::sub(b, a)), mesh::normalize(mesh::sub(c, a)));
            let ang = mesh::dot(u, w).clamp(-1.0, 1.0).acos() as f64;
            for d in 0..3 {
                acc[t[k] as usize][d] += ang * n[d] as f64;
            }
        }
    }
    acc.into_iter().map(|a| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if l > 0.0 { [(a[0] / l) as f32, (a[1] / l) as f32, (a[2] / l) as f32] } else { [0.0; 3] }
    }).collect()
}

/// The user's mesh as UniRig's extract leaves it
pub fn shape(m: &Mesh, max_faces: usize) -> Shape {
    let b = Mesh { positions: m.positions.iter().map(|p| to_blender(*p)).collect(), indices: m.indices.clone(), ..Default::default() };
    // trimesh merges positions equal to 8 decimal places
    let (mut w, map) = tools::weld(&b, 1e-8);
    tools::compact(&mut w);
    let (w, map, decimated) = if w.triangles() > max_faces {
        let d = tools::decimate(&w, max_faces);
        let c = tools::closest::Closest::new(&d.positions, &d.indices);
        // each original vertex goes to the nearest corner of its closest triangle
        let map2: Vec<u32> = b.positions.iter().map(|p| match c.query(*p) {
            Some((_, t, bc)) => {
                let tri = c.triangle(t);
                let k = (0..3).max_by(|&i, &j| bc[i].total_cmp(&bc[j])).unwrap_or(0);
                tri[k]
            }
            None => 0,
        }).collect();
        (d, map2, true)
    } else {
        (w, map, false)
    };
    let f: Vec<[u32; 3]> = w.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    let fn_ = face_normals(&w.positions, &f);
    let vn = vertex_normals(&w.positions, &f, &fn_);
    Shape { v: w.positions, f, vn, fn_, map, decimated }
}

/// The normalization into [-1, 1]: (centre, scale) of the box of `pts` (scale: half its longest side)
pub fn normalization(pts: impl Iterator<Item = [f32; 3]>) -> ([f32; 3], f32) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in pts {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let c = [0, 1, 2].map(|i| (hi[i] + lo[i]) / 2.0);
    let s = (0..3).map(|i| (hi[i] - lo[i]) / 2.0).fold(0f32, f32::max);
    (c, if s > 0.0 { s } else { 1.0 })
}

pub fn apply(p: [f32; 3], c: [f32; 3], s: f32) -> [f32; 3] {
    [(p[0] - c[0]) / s, (p[1] - c[1]) / s, (p[2] - c[2]) / s]
}

/// Samples of a mesh: points, normals, and for each, (triangle or vertex, barycentric weights) to carry vertex data
pub struct Samples {
    pub pts: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    /// (the vertex, None) or (the triangle, its weights)
    pub at: Vec<(u32, Option<[f32; 3]>)>,
}

/// SamplerMix: up to `vertex_samples` vertices in random order, then surface points to `total`
pub fn sample(v: &[[f32; 3]], vn: &[[f32; 3]], f: &[[u32; 3]], fnrm: &[[f32; 3]], total: usize, vertex_samples: usize, rng: &mut Rng) -> Samples {
    let mut out = Samples { pts: Vec::with_capacity(total), nrm: Vec::with_capacity(total), at: Vec::with_capacity(total) };
    let nv = vertex_samples.min(v.len());
    let mut perm: Vec<usize> = (0..v.len()).collect();
    for i in 0..nv {
        let j = i + ((rng.unit() * (v.len() - i) as f64) as usize).min(v.len() - i - 1);
        perm.swap(i, j);
    }
    for &i in &perm[..nv] {
        out.pts.push(v[i]);
        out.nrm.push(vn[i]);
        out.at.push((i as u32, None));
    }
    // triangles by their squared area (the cross product's squared length), as UniRig weighs them
    let mut cum = Vec::with_capacity(f.len());
    let mut acc = 0f64;
    for t in f {
        let n = mesh::cross(mesh::sub(v[t[1] as usize], v[t[0] as usize]), mesh::sub(v[t[2] as usize], v[t[0] as usize]));
        acc += mesh::dot(n, n) as f64;
        cum.push(acc);
    }
    for _ in nv..total {
        let r = rng.unit() * acc;
        let ti = cum.partition_point(|c| *c < r).min(f.len().saturating_sub(1));
        let (mut a, mut b) = (rng.unit() as f32, rng.unit() as f32);
        if a + b > 1.0 {
            a = 1.0 - a;
            b = 1.0 - b;
        }
        let t = f[ti];
        let (p0, p1, p2) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
        out.pts.push([0, 1, 2].map(|c| p0[c] + (p1[c] - p0[c]) * a + (p2[c] - p0[c]) * b));
        out.nrm.push(fnrm[ti]);
        out.at.push((ti as u32, Some([1.0 - a - b, a, b])));
    }
    out
}

impl Samples {
    /// Per-vertex data `x` [V, j] carried to the samples [n, j] (a vertex's own, or the triangle's barycentric blend)
    pub fn carry(&self, f: &[[u32; 3]], x: &[f32], j: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.at.len() * j);
        for (i, w) in &self.at {
            match w {
                None => out.extend_from_slice(&x[*i as usize * j..(*i as usize + 1) * j]),
                Some(w) => {
                    let t = f[*i as usize];
                    for c in 0..j {
                        let v0 = x[t[0] as usize * j + c];
                        out.push(v0 + (x[t[1] as usize * j + c] - v0) * w[1] + (x[t[2] as usize * j + c] - v0) * w[2]);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_and_samples_of_a_quad() {
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let f = vec![[0, 1, 2], [0, 2, 3]];
        let fnrm = face_normals(&v, &f);
        assert_eq!(fnrm[0], [0.0, 0.0, 1.0]);
        let vn = vertex_normals(&v, &f, &fnrm);
        assert!(vn.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
        let mut r = Rng::new(5);
        let s = sample(&v, &vn, &f, &fnrm, 100, 2, &mut r);
        assert_eq!(s.pts.len(), 100);
        assert!(s.pts.iter().all(|p| p[0] >= -1e-6 && p[0] <= 1.0 + 1e-6 && p[2] == 0.0));
        // carried data: x at the vertices -> the samples' x
        let xs: Vec<f32> = v.iter().map(|p| p[0]).collect();
        let c = s.carry(&f, &xs, 1);
        for (p, x) in s.pts.iter().zip(&c) {
            assert!((p[0] - x).abs() < 1e-5);
        }
        let (cen, sc) = normalization(v.iter().copied());
        assert_eq!((cen, sc), ([0.5, 0.5, 0.0], 0.5));
    }
}
