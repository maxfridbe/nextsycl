//! sparc3d.sparcubes.postprocess: the pieces holding under 0.5 % of the faces dropped (faces joined by an edge exactly
//! two of them share, trimesh's face_adjacency; the largest piece always kept), simplified to the face target (here the
//! mesh crate's quadric decimation; the reference uses pyfqmr, aggressiveness 5, borders kept), centred on the
//! vertices' mean and scaled so the farthest vertex is at radius 1.

use std::collections::HashMap;

use nextsycl_model3d::mesh::{tools, Mesh};

use crate::extract::Tri;

/// The faces in pieces of at least `ratio` of them (and the largest), vertices compacted in their old order
pub fn filter_components(m: &Tri, ratio: f64) -> Tri {
    let nf = m.f.len();
    let mut parent: Vec<u32> = (0..nf as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    // each edge's faces: (how many, the first two)
    let mut by_edge: HashMap<(u32, u32), (u32, u32, u32)> = HashMap::with_capacity(nf * 2);
    for (fi, t) in m.f.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let e = by_edge.entry((a.min(b), a.max(b))).or_insert((0, fi as u32, 0));
            if e.0 == 1 {
                e.2 = fi as u32;
            }
            e.0 += 1;
        }
    }
    for (cnt, f0, f1) in by_edge.values() {
        if *cnt == 2 {
            let (x, y) = (find(&mut parent, *f0), find(&mut parent, *f1));
            if x != y {
                parent[x as usize] = y;
            }
        }
    }
    drop(by_edge);
    let roots: Vec<u32> = (0..nf as u32).map(|f| find(&mut parent, f)).collect();
    let mut size: HashMap<u32, usize> = HashMap::new();
    for r in &roots {
        *size.entry(*r).or_default() += 1;
    }
    let largest = size.iter().max_by_key(|(r, s)| (**s, std::cmp::Reverse(**r))).map(|(r, _)| *r).unwrap_or(0);
    let keep: Vec<bool> = roots.iter().map(|r| *r == largest || size[r] as f64 / nf as f64 >= ratio).collect();
    let faces: Vec<[u32; 3]> = m.f.iter().zip(&keep).filter(|(_, k)| **k).map(|(t, _)| *t).collect();
    let mut used = vec![false; m.v.len()];
    for t in &faces {
        for v in t {
            used[*v as usize] = true;
        }
    }
    let mut remap = vec![u32::MAX; m.v.len()];
    let mut v = Vec::new();
    for (i, u) in used.iter().enumerate() {
        if *u {
            remap[i] = v.len() as u32;
            v.push(m.v[i]);
        }
    }
    Tri { v, f: faces.iter().map(|t| t.map(|i| remap[i as usize])).collect() }
}

/// The whole postprocess: components, decimation to `target` faces, centred, unit radius
pub fn postprocess(m: &Tri, target: usize) -> Mesh {
    let kept = filter_components(m, 0.005);
    let mesh = Mesh { positions: kept.v, indices: kept.f.iter().flatten().copied().collect(), ..Default::default() };
    let mut out = if mesh.triangles() > target { tools::decimate(&mesh, target) } else { mesh };
    let n = out.positions.len().max(1) as f64;
    let mut c = [0f64; 3];
    for p in &out.positions {
        for a in 0..3 {
            c[a] += p[a] as f64;
        }
    }
    let c = c.map(|x| x / n);
    let r = out.positions.iter().map(|p| ((p[0] as f64 - c[0]).powi(2) + (p[1] as f64 - c[1]).powi(2) + (p[2] as f64 - c[2]).powi(2)).sqrt())
        .fold(0f64, f64::max)
        .max(1e-12);
    for p in &mut out.positions {
        for a in 0..3 {
            p[a] = ((p[a] as f64 - c[a]) / r) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pieces_go_and_the_rest_is_unit() {
        // a big piece of 200 triangles (a strip) and a lone triangle (0.5 %: under 0.5 % of 201 it goes)
        let mut v = Vec::new();
        let mut f = Vec::new();
        for i in 0..=100 {
            v.push([i as f32, 0.0, 0.0]);
            v.push([i as f32, 1.0, 0.0]);
        }
        for i in 0..100u32 {
            f.push([2 * i, 2 * i + 2, 2 * i + 1]);
            f.push([2 * i + 1, 2 * i + 2, 2 * i + 3]);
        }
        let b = v.len() as u32;
        v.extend([[50.0, 5.0, 0.0], [51.0, 5.0, 0.0], [50.0, 6.0, 0.0]]);
        f.push([b, b + 1, b + 2]);
        let k = filter_components(&Tri { v, f }, 0.005);
        assert_eq!(k.f.len(), 200);
        assert_eq!(k.v.len(), 202);
        let m = postprocess(&k, 1_000_000);
        let r = m.positions.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).fold(0f32, f32::max);
        assert!((r - 1.0).abs() < 1e-5);
    }
}
