//! The decoded voxels to a triangle mesh, as sparc3d.sparcubes does it (CPU):
//!
//! 1. `point_to_mesh`: the voxels sorted by (x, y, z); each voxel's 8 corners' values - the sign probability
//!    (sigmoid of its logit), the distance, the offset (x 0.5) - averaged over the voxels sharing a corner; the distance
//!    takes its sign from the averaged probability (>= 0.5: outside); the corners moved by their offsets and mapped
//!    by the latent grid's matrix (`Point._build_matrix`: centred on the latents' mean, scaled to their radius);
//! 2. FlexiCubes' dual marching cubes without its learned weights (sparc3d's copy): the cubes whose corners change
//!    sign, their case (the ambiguous ones resolved against the neighbour the table names), one to four dual vertices
//!    a cube at the mean of its edges' zero crossings, a quad around every crossing edge shared by four such cubes
//!    (winding toward the positive side), split on a fixed diagonal;
//! 3. `fill_mesh_holes`: every boundary loop of up to 10,000 edges closed by ear clipping (float32, as numpy does it).

use std::collections::HashMap;

use crate::tables::{CHECK, DMC, NUM_VD};

/// A triangle mesh
#[derive(Clone, Default)]
pub struct Tri {
    pub v: Vec<[f32; 3]>,
    pub f: Vec<[u32; 3]>,
}

const CORNERS: [[i32; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const EDGES: [usize; 24] = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4];

/// Point._build_matrix of the latent voxels: (scale, centre) with x' = (x - centre) / scale
pub fn matrix(latent: &[[i32; 3]]) -> (f32, [f32; 3]) {
    let n = latent.len() as f32;
    let mut c = [0f32; 3];
    for p in latent {
        for a in 0..3 {
            c[a] += p[a] as f32;
        }
    }
    let c = c.map(|v| v / n);
    let s = latent.iter().map(|p| {
        let d = [p[0] as f32 - c[0], p[1] as f32 - c[1], p[2] as f32 - c[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }).fold(0f32, f32::max) + 1e-8;
    (s, c)
}

fn vkey(c: [i32; 3]) -> u64 {
    ((c[0] as u64) << 42) | ((c[1] as u64) << 21) | c[2] as u64
}

/// The mesh of the decoded voxels (coords at `res`; sign [n, 8] logits; feat [n, 32] activated) with the latent
/// grid's matrix
pub fn to_mesh(coords: &[[i32; 3]], res: usize, sign: &[f32], feat: &[f32], mat: (f32, [f32; 3])) -> Tri {
    let n = coords.len();
    let r = res as i64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| (coords[i][0] as i64 * r + coords[i][1] as i64) * r + coords[i][2] as i64);
    let cubes_at: Vec<[i32; 3]> = order.iter().map(|&i| coords[i]).collect();
    // the unique corners, sorted (torch.unique over rows), and each cube's 8
    let mut keys: Vec<u64> = Vec::with_capacity(n * 8);
    for c in &cubes_at {
        for k in CORNERS {
            keys.push(vkey([c[0] + k[0], c[1] + k[1], c[2] + k[2]]));
        }
    }
    let mut uniq = keys.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let cube: Vec<u32> = keys.iter().map(|k| uniq.binary_search(k).unwrap() as u32).collect();
    drop(keys);
    let nv = uniq.len();
    // per corner [sign prob, sdf, dx, dy, dz] averaged at the shared vertices
    let mut acc = vec![[0f32; 5]; nv];
    let mut cnt = vec![0u32; nv];
    for (ci, &i) in order.iter().enumerate() {
        for k in 0..8 {
            let v = cube[ci * 8 + k] as usize;
            let s = 1.0 / (1.0 + (-sign[i * 8 + k]).exp());
            let f = &feat[i * 32..i * 32 + 32];
            let a = &mut acc[v];
            a[0] += s;
            a[1] += f[k];
            for j in 0..3 {
                a[2 + j] += f[8 + 3 * k + j] * 0.5;
            }
            cnt[v] += 1;
        }
    }
    let mut pos = vec![[0f32; 3]; nv];
    let mut sdf = vec![0f32; nv];
    for v in 0..nv {
        let c = cnt[v] as f32;
        let a = acc[v].map(|x| x / c);
        let k = uniq[v];
        let g = [(k >> 42) as f32, ((k >> 21) & 0x1f_ffff) as f32, (k & 0x1f_ffff) as f32];
        sdf[v] = if a[0] >= 0.5 { a[1].abs() } else { -a[1].abs() };
        for j in 0..3 {
            pos[v][j] = (g[j] + a[2 + j]) / mat.0 - mat.1[j] / mat.0;
        }
    }
    flexicubes(&pos, &sdf, &cube, &cubes_at, res)
}

/// FlexiCubes (sparc3d's: no learned weights, sparse case lookup): (vertices, faces)
pub fn flexicubes(pos: &[[f32; 3]], sdf: &[f32], cube: &[u32], at: &[[i32; 3]], res: usize) -> Tri {
    let nc = at.len();
    let occ = |v: u32| sdf[v as usize] < 0.0;
    // the surface cubes and their cases
    let mut surf = Vec::new();
    let mut case = Vec::new();
    for c in 0..nc {
        let mut id = 0usize;
        let mut cnt = 0;
        for k in 0..8 {
            if occ(cube[c * 8 + k]) {
                id |= 1 << k;
                cnt += 1;
            }
        }
        if cnt > 0 && cnt < 8 {
            surf.push(c);
            case.push(id);
        }
    }
    if surf.is_empty() {
        return Tri::default();
    }
    // the ambiguous cases: resolved when the neighbour the table names is an ambiguous cube too (its own neighbour
    // inside the grid)
    let r = res as i32;
    let inside = |p: [i32; 3]| p.iter().all(|v| *v >= 0 && *v < r);
    let mut problem: HashMap<[i32; 3], usize> = HashMap::new();
    let mut checks = Vec::new();
    for (si, &c) in surf.iter().enumerate() {
        let ck = CHECK[case[si]];
        if ck[0] == 1 {
            let adj = [at[c][0] + ck[1] as i32, at[c][1] + ck[2] as i32, at[c][2] + ck[3] as i32];
            if inside(adj) {
                problem.insert(at[c], si);
                checks.push((si, adj, ck[4] as usize));
            }
        }
    }
    let mut newcase = case.clone();
    for (si, adj, to) in checks {
        if problem.contains_key(&adj) {
            newcase[si] = to;
        }
    }
    let case = newcase;
    // the edges of the surface cubes: unique by their two corners (sorted), counted, the crossing ones numbered
    let ns = surf.len();
    let mut all: Vec<(u64, u32)> = Vec::with_capacity(ns * 12);
    for (si, &c) in surf.iter().enumerate() {
        for e in 0..12 {
            let (a, b) = (cube[c * 8 + EDGES[2 * e]] as u64, cube[c * 8 + EDGES[2 * e + 1]] as u64);
            all.push(((a << 32) | b, (si * 12 + e) as u32));
        }
    }
    all.sort_unstable();
    let mut edge_of = vec![0u32; ns * 12]; // the unique edge of each cube edge
    let mut uniq: Vec<u64> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for (k, slot) in &all {
        if uniq.last() != Some(k) {
            uniq.push(*k);
            counts.push(0);
        }
        *counts.last_mut().unwrap() += 1;
        edge_of[*slot as usize] = (uniq.len() - 1) as u32;
    }
    drop(all);
    let crossing: Vec<bool> = uniq.iter().map(|k| occ((k >> 32) as u32) != occ((k & 0xffff_ffff) as u32)).collect();
    let mut surf_id = vec![-1i64; uniq.len()];
    let mut sedges: Vec<(u32, u32)> = Vec::new();
    for (u, k) in uniq.iter().enumerate() {
        if crossing[u] {
            surf_id[u] = sedges.len() as i64;
            sedges.push(((k >> 32) as u32, (k & 0xffff_ffff) as u32));
        }
    }
    // the zero crossing on each surface edge
    let ue: Vec<[f32; 3]> = sedges.iter().map(|&(a, b)| {
        let (s0, s1) = (sdf[a as usize], sdf[b as usize]);
        let (w0, w1) = (s1, -s0);
        let den = w0 + w1;
        let (x0, x1) = (pos[a as usize], pos[b as usize]);
        [(x0[0] * w0 + x1[0] * w1) / den, (x0[1] * w0 + x1[1] * w1) / den, (x0[2] * w0 + x1[2] * w1) / den]
    }).collect();
    // dual vertices: the cubes grouped by how many they emit (1, 2, 3, 4), each the mean of its edges' crossings
    let mut vd: Vec<[f32; 3]> = Vec::new();
    // (a cube edge no dual vertex lists stays 0, as the reference's zero-initialised map)
    let mut vd_of = vec![0u32; ns * 12];
    for num in 1..=4u8 {
        for si in 0..ns {
            if NUM_VD[case[si]] != num {
                continue;
            }
            for group in DMC[case[si]].iter().take(num as usize) {
                let id = vd.len() as u32;
                let mut s = [0f32; 3];
                let mut m = 0f32;
                for &e in group {
                    if e < 0 {
                        continue;
                    }
                    let slot = si * 12 + e as usize;
                    let sid = surf_id[edge_of[slot] as usize];
                    if sid < 0 {
                        continue;
                    }
                    let p = ue[sid as usize];
                    for j in 0..3 {
                        s[j] += p[j];
                    }
                    m += 1.0;
                    vd_of[slot] = id;
                }
                vd.push(if m > 0.0 { s.map(|x| x / m) } else { s });
            }
        }
    }
    // quads: every crossing edge shared by 4 surface cubes, its dual vertices in cube order
    let mut quads: Vec<(i64, usize)> = Vec::new();
    for (slot, &u) in edge_of.iter().enumerate() {
        let u = u as usize;
        if counts[u] == 4 && crossing[u] {
            quads.push((surf_id[u], slot));
        }
    }
    quads.sort_by_key(|q| q.0);
    let mut flipped = Vec::new();
    let mut kept = Vec::new();
    for q in quads.chunks_exact(4) {
        let ids = [vd_of[q[0].1], vd_of[q[1].1], vd_of[q[2].1], vd_of[q[3].1]];
        let (a, _) = sedges[q[0].0 as usize];
        if sdf[a as usize] > 0.0 {
            flipped.push([ids[0], ids[1], ids[3], ids[2]]);
        } else {
            kept.push([ids[2], ids[3], ids[1], ids[0]]);
        }
    }
    let mut f = Vec::with_capacity(2 * (flipped.len() + kept.len()));
    for q in flipped.iter().chain(&kept) {
        f.push([q[0], q[1], q[3]]);
        f.push([q[3], q[1], q[2]]);
    }
    Tri { v: vd, f }
}

/// fill_mesh_holes: each boundary loop (half-edges whose edge occurs once, chained) of 3 .. 10,000 vertices closed by
/// ear clipping, the triangles appended
pub fn fill_holes(m: &Tri) -> Tri {
    let nf = m.f.len();
    let mut directed: Vec<(u32, u32)> = Vec::with_capacity(nf * 3);
    for k in 0..3 {
        for t in &m.f {
            directed.push((t[k], t[(k + 1) % 3]));
        }
    }
    let mut count: HashMap<(u32, u32), u32> = HashMap::with_capacity(directed.len());
    for &(a, b) in &directed {
        *count.entry((a.min(b), a.max(b))).or_default() += 1;
    }
    let boundary: Vec<(u32, u32)> = directed.iter().copied().filter(|&(a, b)| count[&(a.min(b), a.max(b))] == 1).collect();
    drop(count);
    let mut outgoing: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &(s, _)) in boundary.iter().enumerate() {
        outgoing.entry(s).or_default().push(i);
    }
    let mut visited = vec![false; boundary.len()];
    let mut out = m.clone();
    for i in 0..boundary.len() {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let start = boundary[i].0;
        let mut lp = vec![start, boundary[i].1];
        while *lp.last().unwrap() != start {
            let mut next = None;
            for &j in outgoing.get(lp.last().unwrap()).map(|v| v.as_slice()).unwrap_or(&[]) {
                if !visited[j] {
                    next = Some(j);
                    break;
                }
            }
            match next {
                Some(j) => {
                    visited[j] = true;
                    lp.push(boundary[j].1);
                }
                None => break,
            }
        }
        if lp[0] == *lp.last().unwrap() && lp.len() > 3 {
            lp.pop();
            if lp.len() >= 3 && lp.len() <= 10000 {
                out.f.extend(ear_clip(&m.v, &lp));
            }
        }
    }
    out
}

fn ear_clip(v: &[[f32; 3]], lp: &[u32]) -> Vec<[u32; 3]> {
    let mut pts: Vec<[f32; 3]> = lp.iter().map(|&i| v[i as usize]).collect();
    let mut idx: Vec<u32> = lp.to_vec();
    let mut faces = Vec::new();
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f32; 3]| dot(a, a).sqrt();
    loop {
        let l = idx.len();
        if l == 3 {
            faces.push([idx[2], idx[1], idx[0]]);
            break;
        }
        let t: Vec<[f32; 3]> = (0..l).map(|i| {
            let d = sub(pts[(i + 1) % l], pts[i]);
            let n = norm(d) + 1e-6;
            d.map(|x| x / n)
        }).collect();
        let mut center = [0f32; 3];
        for p in &pts {
            for j in 0..3 {
                center[j] += p[j];
            }
        }
        let center = center.map(|x| x / l as f32);
        let mut best = (f32::NEG_INFINITY, 0usize);
        for i in 0..l {
            let tp = t[(i + l - 1) % l];
            let nn = cross(tp.map(|x| -x), t[i]);
            let pp = sub(pts[(i + 1) % l], center);
            let qq = sub(pts[i], center);
            let s = dot(cross(pp, qq), nn);
            let sign = if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { 0.0 };
            let a = (norm(nn) * sign).atan2(-dot(tp, t[i]));
            let score = if a == 0.0 { f32::INFINITY } else { 1.0 / a };
            if score > best.0 {
                best = (score, i);
            }
        }
        let ear = best.1;
        let (i0, i2) = ((ear + 1) % l, (ear + l - 1) % l);
        faces.push([idx[i0], idx[ear], idx[i2]]);
        pts.remove(ear);
        idx.remove(ear);
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_of_voxels_makes_a_closed_surface() {
        // a 4^3 block of voxels with the corners inside a sphere negative: a closed surface around it
        let mut coords = Vec::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    coords.push([x, y, z]);
                }
            }
        }
        let n = coords.len();
        // every corner's logit and distance from the corner's distance to the centre (2, 2, 2), radius 1.2
        let mut sign = vec![0f32; n * 8];
        let mut feat = vec![0f32; n * 32];
        for (i, c) in coords.iter().enumerate() {
            for (k, o) in CORNERS.iter().enumerate() {
                let p = [c[0] + o[0], c[1] + o[1], c[2] + o[2]].map(|v| v as f32 - 2.0);
                let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.2;
                sign[i * 8 + k] = d * 10.0;
                feat[i * 32 + k] = d.tanh();
            }
        }
        let m = to_mesh(&coords, 8, &sign, &feat, (1.0, [0.0; 3]));
        assert!(!m.f.is_empty());
        // closed: every edge in two triangles
        let mut e: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &m.f {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *e.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(e.values().all(|c| *c == 2), "an open or non-manifold edge");
        let filled = fill_holes(&m);
        assert_eq!(filled.f.len(), m.f.len());
    }

    #[test]
    fn a_square_hole_is_closed() {
        let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let m = Tri { v: v.clone(), f: vec![] };
        assert!(fill_holes(&m).f.is_empty());
        let faces = ear_clip(&v, &[0, 1, 2, 3]);
        assert_eq!(faces.len(), 2);
    }
}
