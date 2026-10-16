//! Isotropic remeshing (Botsch & Kobbelt): evenly sized triangles of a target edge length over the same surface.
//! Each pass splits edges longer than 4/3 of it, collapses those shorter than 4/5, flips edges toward valence 6,
//! smooths tangentially and projects the vertices back onto the input. Boundary vertices stay put. Positions only
//! (and colours, from the nearest input point): re-UV the result with `uv`.

use std::collections::{HashMap, HashSet};

use super::closest::Closest;
use crate::{add, cross, dot, length, normalize, scale, sub, Mesh};

struct M {
    pos: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
}

impl M {
    fn vt(&self) -> Vec<Vec<u32>> {
        let mut vt = vec![Vec::new(); self.pos.len()];
        for (i, t) in self.tris.iter().enumerate() {
            for &v in t {
                vt[v as usize].push(i as u32);
            }
        }
        vt
    }
    /// edge -> the triangles on it
    fn edges(&self) -> HashMap<(u32, u32), Vec<u32>> {
        let mut e: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (i, t) in self.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                e.entry((a.min(b), a.max(b))).or_default().push(i as u32);
            }
        }
        e
    }
    fn boundary(&self) -> Vec<bool> {
        let mut b = vec![false; self.pos.len()];
        for ((x, y), ts) in self.edges() {
            if ts.len() != 2 {
                b[x as usize] = true;
                b[y as usize] = true;
            }
        }
        b
    }
    fn normal(&self, t: [u32; 3]) -> [f32; 3] {
        cross(sub(self.pos[t[1] as usize], self.pos[t[0] as usize]), sub(self.pos[t[2] as usize], self.pos[t[0] as usize]))
    }
}

/// Every edge longer than `hi` split at its middle, all at once (a triangle with 1, 2 or 3 split edges becomes 2, 3
/// or 4); the surface stays manifold
fn split(m: &mut M, hi: f32) -> usize {
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    for &(a, b) in m.edges().keys() {
        if length(sub(m.pos[a as usize], m.pos[b as usize])) > hi {
            m.pos.push(scale(add(m.pos[a as usize], m.pos[b as usize]), 0.5));
            mid.insert((a, b), (m.pos.len() - 1) as u32);
        }
    }
    if mid.is_empty() {
        return 0;
    }
    let get = |a: u32, b: u32| mid.get(&(a.min(b), a.max(b))).copied();
    let mut out = Vec::with_capacity(m.tris.len() * 2);
    for &t in &m.tris {
        let ms = [get(t[0], t[1]), get(t[1], t[2]), get(t[2], t[0])];
        match ms.iter().filter(|x| x.is_some()).count() {
            0 => out.push(t),
            1 => {
                let k = ms.iter().position(|x| x.is_some()).unwrap();
                let (a, b, c, x) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3], ms[k].unwrap());
                out.extend([[a, x, c], [x, b, c]]);
            }
            2 => {
                // the unsplit edge is k: c, a around it; split points on the other two
                let k = ms.iter().position(|x| x.is_none()).unwrap();
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let (xb, xc) = (ms[(k + 1) % 3].unwrap(), ms[(k + 2) % 3].unwrap()); // on b-c and c-a
                out.extend([[xc, xb, c], [a, b, xb], [a, xb, xc]]);
            }
            _ => {
                let (x, y, z) = (ms[0].unwrap(), ms[1].unwrap(), ms[2].unwrap());
                out.extend([[t[0], x, z], [x, t[1], y], [z, y, t[2]], [x, y, z]]);
            }
        }
    }
    m.tris = out;
    mid.len()
}

/// Edges shorter than `lo` collapsed to their middle where that makes no edge longer than `hi`, flips nothing and
/// keeps the surface a manifold; boundary vertices are not moved
fn collapse(m: &mut M, lo: f32, hi: f32) -> usize {
    let bnd = m.boundary();
    let mut vt = m.vt();
    let mut alive = vec![true; m.tris.len()];
    let mut gone = vec![false; m.pos.len()];
    let mut short: Vec<(f32, u32, u32)> = m.edges().keys().map(|&(a, b)| (length(sub(m.pos[a as usize], m.pos[b as usize])), a, b))
        .filter(|e| e.0 < lo).collect();
    short.sort_by(|x, y| x.0.total_cmp(&y.0));
    let ring = |vt: &[Vec<u32>], tris: &[[u32; 3]], alive: &[bool], v: u32| -> HashSet<u32> {
        vt[v as usize].iter().filter(|&&t| alive[t as usize]).flat_map(|&t| tris[t as usize]).filter(|&x| x != v).collect()
    };
    let mut n = 0;
    for (_, a, b) in short {
        if gone[a as usize] || gone[b as usize] || bnd[a as usize] || bnd[b as usize] {
            continue;
        }
        let (ra, rb) = (ring(&vt, &m.tris, &alive, a), ring(&vt, &m.tris, &alive, b));
        if !ra.contains(&b) || ra.intersection(&rb).count() != 2 || length(sub(m.pos[a as usize], m.pos[b as usize])) >= lo {
            continue;
        }
        let to = scale(add(m.pos[a as usize], m.pos[b as usize]), 0.5);
        if ra.iter().chain(rb.iter()).any(|&x| x != a && x != b && length(sub(m.pos[x as usize], to)) > hi) {
            continue;
        }
        let flips = [a, b].iter().any(|&v| vt[v as usize].iter().any(|&t| {
            let tr = m.tris[t as usize];
            if !alive[t as usize] || (tr.contains(&a) && tr.contains(&b)) {
                return false;
            }
            let before = m.normal(tr);
            let p = tr.map(|x| if x == v { to } else { m.pos[x as usize] });
            let after = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            dot(before, after) <= 0.1 * length(before) * length(after)
        }));
        if flips {
            continue;
        }
        m.pos[a as usize] = to;
        gone[b as usize] = true;
        for t in std::mem::take(&mut vt[b as usize]) {
            if !alive[t as usize] {
                continue;
            }
            if m.tris[t as usize].contains(&a) {
                alive[t as usize] = false;
            } else {
                for x in m.tris[t as usize].iter_mut() {
                    if *x == b {
                        *x = a;
                    }
                }
                vt[a as usize].push(t);
            }
        }
        n += 1;
    }
    m.tris = m.tris.iter().zip(&alive).filter(|(_, &l)| l).map(|(t, _)| *t).collect();
    n
}

/// Edges flipped where that brings the four vertices' valences nearer 6 (4 on a boundary)
fn flip(m: &mut M) -> usize {
    let bnd = m.boundary();
    let mut val = vec![0i32; m.pos.len()];
    let edges = m.edges();
    for &(a, b) in edges.keys() {
        val[a as usize] += 1;
        val[b as usize] += 1;
    }
    let want = |v: u32| if bnd[v as usize] { 4 } else { 6 };
    let mut have: HashSet<(u32, u32)> = edges.keys().copied().collect();
    let mut n = 0;
    let mut touched = vec![false; m.tris.len()];
    let mut list: Vec<_> = edges.into_iter().filter(|(_, ts)| ts.len() == 2).collect();
    list.sort_by_key(|(e, _)| *e);
    for ((a, b), ts) in list {
        let (t0, t1) = (ts[0] as usize, ts[1] as usize);
        if touched[t0] || touched[t1] {
            continue;
        }
        let opp = |t: [u32; 3]| *t.iter().find(|&&x| x != a && x != b).unwrap();
        let (c, d) = (opp(m.tris[t0]), opp(m.tris[t1]));
        if c == d || have.contains(&(c.min(d), c.max(d))) {
            continue;
        }
        let dev = |v: &[i32]| -> i32 { [a, b, c, d].iter().map(|&x| (v[x as usize] - want(x)).abs()).sum() };
        let before = dev(&val);
        let mut v2 = val.clone();
        v2[a as usize] -= 1;
        v2[b as usize] -= 1;
        v2[c as usize] += 1;
        v2[d as usize] += 1;
        if dev(&v2) >= before {
            continue;
        }
        // the new pair, wound as the old: t0 has a->b (in its order) or b->a
        let tr0 = m.tris[t0];
        let k = tr0.iter().position(|&x| x == a).unwrap();
        let (x, y) = if tr0[(k + 1) % 3] == b { (a, b) } else { (b, a) };
        let n0 = [c, x, d];
        let n1 = [d, y, c];
        let ok = |t: [u32; 3]| dot(m.normal(t), m.normal(tr0)) > 0.0;
        if !ok(n0) || !ok(n1) {
            continue;
        }
        m.tris[t0] = n0;
        m.tris[t1] = n1;
        touched[t0] = true;
        touched[t1] = true;
        have.remove(&(a.min(b), a.max(b)));
        have.insert((c.min(d), c.max(d)));
        val = v2;
        n += 1;
    }
    n
}

/// Each inner vertex moved toward its neighbours' centroid, in its tangent plane only, then back onto the surface
fn smooth_and_project(m: &mut M, onto: &Closest, lambda: f32) {
    let bnd = m.boundary();
    let mut sum = vec![[0f32; 3]; m.pos.len()];
    let mut cnt = vec![0f32; m.pos.len()];
    let mut nrm = vec![[0f32; 3]; m.pos.len()];
    for &t in &m.tris {
        let n = m.normal(t);
        for k in 0..3 {
            let (a, b) = (t[k] as usize, t[(k + 1) % 3] as usize);
            sum[a] = add(sum[a], m.pos[b]);
            cnt[a] += 1.0;
            nrm[a] = add(nrm[a], n);
        }
    }
    for v in 0..m.pos.len() {
        if bnd[v] || cnt[v] == 0.0 {
            continue;
        }
        let n = normalize(nrm[v]);
        let d = sub(scale(sum[v], 1.0 / cnt[v]), m.pos[v]);
        let tangential = sub(d, scale(n, dot(d, n)));
        let p = add(m.pos[v], scale(tangential, lambda));
        m.pos[v] = onto.query(p).map_or(p, |q| q.0);
    }
}

/// The surface remeshed with edges near `edge` long (None: the input's mean edge length), over `passes` passes
pub fn remesh(input: &Mesh, edge: Option<f32>, passes: usize) -> Mesh {
    let (lo_b, hi_b) = input.bounds();
    let diag = length(sub(hi_b, lo_b)).max(1e-6);
    let (welded, _) = super::weld(input, diag * 1e-6);
    let mean = {
        let e = super::edge_counts(&welded.indices);
        let s: f32 = e.keys().map(|&(a, b)| length(sub(welded.positions[a as usize], welded.positions[b as usize]))).sum();
        s / e.len().max(1) as f32
    };
    let target = edge.unwrap_or(mean).max(diag * 1e-4);
    let onto = Closest::new(&input.positions, &input.indices);
    let mut m = M { pos: welded.positions.clone(), tris: welded.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect() };
    for _ in 0..passes {
        split(&mut m, target * 4.0 / 3.0);
        collapse(&mut m, target * 4.0 / 5.0, target * 4.0 / 3.0);
        let mut tmp = Mesh { positions: m.pos.clone(), indices: m.tris.iter().flatten().copied().collect(), ..Default::default() };
        super::compact(&mut tmp);
        m = M { pos: tmp.positions, tris: tmp.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect() };
        flip(&mut m);
        smooth_and_project(&mut m, &onto, 0.5);
    }
    let mut out = Mesh { positions: m.pos, indices: m.tris.into_iter().flatten().collect(), material: input.material.clone(), ..Default::default() };
    super::compact(&mut out);
    if !input.colors.is_empty() {
        out.colors = out.positions.iter().map(|&p| {
            onto.query(p).map_or([1.0; 4], |(_, t, bc)| {
                let tr = onto.triangle(t);
                let mut c = [0f32; 4];
                for k in 0..3 {
                    for (ch, x) in c.iter_mut().enumerate() {
                        *x += bc[k] * input.colors[tr[k] as usize][ch];
                    }
                }
                c
            })
        }).collect();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::testing::{is_closed_manifold, sphere};
    use super::*;

    #[test]
    fn a_stretched_sphere_comes_out_even() {
        // a UV sphere's triangles are long at the equator and thin at the poles
        let m = sphere(16, 64);
        let r = remesh(&m, Some(0.15), 6);
        assert!(is_closed_manifold(&r), "holes or folds after remeshing");
        let e = super::super::edge_counts(&r.indices);
        let lens: Vec<f32> = e.keys().map(|&(a, b)| length(sub(r.positions[a as usize], r.positions[b as usize]))).collect();
        let mean = lens.iter().sum::<f32>() / lens.len() as f32;
        let within = lens.iter().filter(|&&l| l > 0.15 * 0.5 && l < 0.15 * 1.6).count() as f32 / lens.len() as f32;
        assert!((mean - 0.15).abs() < 0.04, "mean edge {mean}");
        assert!(within > 0.9, "only {within} of the edges near the target");
        let worst = r.positions.iter().map(|p| (length(*p) - 1.0).abs()).fold(0f32, f32::max);
        assert!(worst < 0.03, "a vertex {worst} off the surface");
    }
}
