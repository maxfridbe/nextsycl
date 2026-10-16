//! The mesh as the model takes it (test_drive.py and utils/mesh_utils.py): turned to Blender's frame (Z up: what the
//! reference's Blender import gives and the model was trained in), identical vertices merged (sorted, as
//! `torch.unique` sorts them: the order matters, farthest point sampling starts at the first), degenerate faces
//! dropped, recentred and scaled into [-1, 1], padded to 4096 rows; and the way back to the input's vertices.

use nextsycl_model3d::Mesh;

/// The model's view of a mesh
pub struct Prepared {
    /// merged vertices, normalized, Blender's frame
    pub verts: Vec<[f32; 3]>,
    /// merged faces (degenerate ones dropped)
    pub faces: Vec<[u32; 3]>,
    /// each input vertex's merged vertex
    pub map: Vec<u32>,
    /// the normalization: p_model = (p_blender - center) / den
    pub center: [f32; 3],
    pub den: f32,
}

/// glTF's frame (Y up) to Blender's (Z up), as its importers turn it
pub fn to_blender(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[2], p[1]]
}

pub fn from_blender(p: [f32; 3]) -> [f32; 3] {
    [p[0], p[2], -p[1]]
}

/// A float as a sort key where -0 and 0 are one value
fn key(x: f32) -> f32 {
    x + 0.0
}

pub fn prepare(m: &Mesh) -> Prepared {
    let pos: Vec<[f32; 3]> = m.positions.iter().map(|p| to_blender(*p).map(key)).collect();
    // torch.unique(dim=0, sorted=True): rows in lexicographic order, equal rows one
    let mut order: Vec<u32> = (0..pos.len() as u32).collect();
    let cmp = |a: &[f32; 3], b: &[f32; 3]| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    order.sort_by(|&a, &b| cmp(&pos[a as usize], &pos[b as usize]));
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut map = vec![0u32; pos.len()];
    for &i in &order {
        let p = pos[i as usize];
        if verts.last() != Some(&p) {
            verts.push(p);
        }
        map[i as usize] = (verts.len() - 1) as u32;
    }
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for t in m.indices.chunks_exact(3) {
        let f = [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]];
        if f[0] == f[1] || f[0] == f[2] || f[1] == f[2] {
            continue;
        }
        let mut s = f;
        s.sort();
        if seen.insert(s) {
            faces.push(s);
        }
    }
    faces.sort();
    // recentred on the box's middle, scaled by the largest coordinate (float32, as the reference)
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in &verts {
        for c in 0..3 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let center = [0, 1, 2].map(|c| (hi[c] + lo[c]) / 2.0);
    let mut vmax = 0f32;
    for p in &mut verts {
        for c in 0..3 {
            p[c] -= center[c];
            vmax = vmax.max(p[c].abs());
        }
    }
    let den = vmax + 1e-8;
    for p in &mut verts {
        for x in p.iter_mut() {
            *x /= den;
        }
    }
    Prepared { verts, faces, map, center, den }
}

impl Prepared {
    /// The neighbour graph of the faces' edges (both ways, each once, ascending) over `rows` rows (padding: none):
    /// (rowptr [rows + 1], cols)
    pub fn csr(&self, rows: usize) -> (Vec<i32>, Vec<i32>) {
        let mut adj: Vec<Vec<i32>> = vec![Vec::new(); rows];
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k] as usize, f[(k + 1) % 3] as usize);
                adj[a].push(b as i32);
                adj[b].push(a as i32);
            }
        }
        let mut rowptr = vec![0i32];
        let mut cols = Vec::new();
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
            cols.extend_from_slice(a);
            rowptr.push(cols.len() as i32);
        }
        (rowptr, cols)
    }

    /// A position of the model's frame back in the input's
    pub fn unmap(&self, p: [f32; 3]) -> [f32; 3] {
        from_blender([0, 1, 2].map(|c| p[c] * self.den + self.center[c]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_sorts_and_drops_degenerate_faces() {
        // two triangles sharing an edge, with the shared corners split; a third collapses onto a line
        let m = Mesh {
            positions: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            indices: vec![0, 1, 2, 3, 5, 4, 1, 3, 4],
            ..Default::default()
        };
        let p = prepare(&m);
        assert_eq!(p.verts.len(), 4);
        assert_eq!(p.faces.len(), 2);
        // the map takes split corners to one vertex, and the order is lexicographic in Blender's frame
        assert_eq!(p.map[1], p.map[3]);
        for w in p.verts.windows(2) {
            assert!(w[0] < w[1]);
        }
        // the round trip
        for (i, q) in m.positions.iter().enumerate() {
            let back = p.unmap(p.verts[p.map[i] as usize]);
            for c in 0..3 {
                assert!((back[c] - q[c]).abs() < 1e-6);
            }
        }
        let (rowptr, cols) = p.csr(6);
        assert_eq!(rowptr.len(), 7);
        assert_eq!(*rowptr.last().unwrap() as usize, cols.len());
        assert_eq!(cols.len(), 10); // 5 edges, both ways
        assert_eq!(rowptr[5], rowptr[4]); // padding rows: no neighbours
    }
}
