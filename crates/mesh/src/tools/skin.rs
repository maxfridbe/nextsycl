//! Skin weights between meshes: a rigger that works on a resampled or remeshed copy of the user's mesh hands its
//! weights back to the original vertices here (`transfer`: each vertex's closest point on the copy, its triangle's
//! weights blended by the barycentric coordinates), and keeps the four largest a vertex for glTF (`top4`).

use super::closest::Closest;

/// Dense weights `w` [V, j] of the mesh (`pos`, `indices`) carried to `targets`: [T, j], each row the barycentric
/// blend of the weights at the closest point's triangle (rows of a mesh without triangles: the nearest vertex's)
pub fn transfer(pos: &[[f32; 3]], indices: &[u32], w: &[f32], j: usize, targets: &[[f32; 3]]) -> Vec<f32> {
    let mut out = vec![0f32; targets.len() * j];
    if pos.is_empty() || j == 0 {
        return out;
    }
    let c = Closest::new(pos, indices);
    for (t, p) in targets.iter().enumerate() {
        let row = &mut out[t * j..(t + 1) * j];
        match c.query(*p) {
            Some((_, tri, bc)) => {
                let v = c.triangle(tri);
                for k in 0..3 {
                    let src = &w[v[k] as usize * j..(v[k] as usize + 1) * j];
                    for (o, s) in row.iter_mut().zip(src) {
                        *o += bc[k] * s;
                    }
                }
            }
            None => {
                let near = (0..pos.len()).min_by(|&a, &b| {
                    let d = |i: usize| crate::dot(crate::sub(pos[i], *p), crate::sub(pos[i], *p));
                    d(a).total_cmp(&d(b))
                }).unwrap_or(0);
                row.copy_from_slice(&w[near * j..(near + 1) * j]);
            }
        }
    }
    out
}

/// The four largest weights of each row of `w` [V, j], renormalized to sum 1 (a row of zeros: all on joint 0)
pub fn top4(w: &[f32], j: usize) -> Vec<[(u16, f32); 4]> {
    if j == 0 {
        return Vec::new();
    }
    w.chunks_exact(j).map(|row| {
        let mut idx: Vec<usize> = (0..j).collect();
        idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let mut out = [(0u16, 0f32); 4];
        for (k, &i) in idx.iter().take(4).enumerate() {
            out[k] = (i as u16, row[i].max(0.0));
        }
        let s: f32 = out.iter().map(|x| x.1).sum();
        if s > 0.0 {
            for x in &mut out {
                x.1 /= s;
            }
        } else {
            out[0] = (0, 1.0);
        }
        out
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_follow_the_closest_point() {
        // a quad of two triangles; joint 0 at x = 0, joint 1 at x = 2
        let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]];
        let idx = [0, 1, 2, 0, 2, 3];
        let w = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let t = transfer(&pos, &idx, &w, 2, &[[0.5, 1.0, 0.3], [2.0, 0.0, -1.0]]);
        // x = 0.5 of 2: a quarter of the way to joint 1
        assert!((t[0] - 0.75).abs() < 1e-5 && (t[1] - 0.25).abs() < 1e-5);
        assert!((t[2] - 0.0).abs() < 1e-5 && (t[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top4_keeps_the_largest() {
        let w = [0.1, 0.4, 0.05, 0.2, 0.25, 0.0];
        let t = top4(&w, 6);
        assert_eq!(t[0][0].0, 1);
        assert_eq!(t[0][1].0, 4);
        assert_eq!(t[0][2].0, 3);
        assert_eq!(t[0][3].0, 0);
        assert!((t.iter().flat_map(|r| r.iter()).map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(top4(&[0.0, 0.0], 2)[0][0], (0, 1.0));
    }
}
