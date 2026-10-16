//! The generated views baked into the mesh's UV atlas, as Tencent's MeshRender.back_project ("back_sample") and
//! fast_bake_texture do it: every texel's surface point projected into each view; kept where the view sees it (its
//! depth within 3e-3 of the view's z-buffer), where the surface faces the camera (cos > cos 75 degrees) and away from
//! the view's silhouette and depth edges (an 8-pixel margin at 2,048 px); the views blended by their weight x cos^4,
//! a view that adds less than 1 % new texels skipped. Texels no view saw are filled from the mesh: the seen texels
//! give their vertices colours, those spread over the mesh's edges to the vertices without one, and an unseen texel
//! takes its triangle's vertex colours; the gutter between charts is grown from its neighbours.
//!
//! Tencent's depth edges come from OpenCV's Canny on the 8-bit normalized depth; here they are the pixels whose 8-bit
//! depth differs from a neighbour's by more than a threshold (the same purpose: no colour across occlusion edges).

use nextsycl_mesh::tools::uv::{grow, texels, Texel};
use nextsycl_mesh::{cross, normalize, sub, Mesh, Texture};

use super::render::{view, Scene, WEIGHTS};

const DEPTH_TOLERANCE: f32 = 3e-3;
const BAKE_EXP: i32 = 4;
/// Tencent's Canny thresholds are 30 / 80 on 8-bit depth; a jump of this many levels between neighbours is an edge
const DEPTH_EDGE: f32 = 30.0;

/// One view's bake: per texel of the list, its colour and its weight (0: not seen)
struct Baked {
    rgb: Vec<[f32; 3]>,
    w: Vec<f32>,
}

/// A box test over an s x s mask: whether any pixel within `r` is set (separable)
fn any_within(m: &[bool], s: usize, r: usize) -> Vec<bool> {
    let pass = |src: &[bool], horiz: bool| -> Vec<bool> {
        let mut out = vec![false; s * s];
        for y in 0..s {
            // a running count over the window
            let mut count = 0usize;
            let at = |i: usize| if horiz { src[y * s + i] } else { src[i * s + y] };
            for i in 0..=r.min(s - 1) {
                count += at(i) as usize;
            }
            for i in 0..s {
                let idx = if horiz { y * s + i } else { i * s + y };
                out[idx] = count > 0;
                if i + r + 1 < s {
                    count += at(i + r + 1) as usize;
                }
                if i >= r {
                    count -= at(i - r) as usize;
                }
            }
        }
        out
    };
    let h = pass(m, true);
    pass(&h, false)
}

/// The views' colours (8-bit values [size, size, 3], rows as rendered) at the texels
fn back_project(s: &Scene, tex: &[Texel], image: &[f32], elev: f64, azim: f64, size: usize) -> Baked {
    let v = view(s, elev, azim, size);
    let r = &v.raster;
    let n = size * size;
    // camera-space face normals, depth, the view's cosine
    let fnrm: Vec<[f32; 3]> = s.tri.iter().map(|t| {
        let p = &v.camera_pos;
        normalize(cross(sub(p[t[1] as usize], p[t[0] as usize]), sub(p[t[2] as usize], p[t[0] as usize])))
    }).collect();
    let mut depth = vec![0f32; n];
    let mut cos = vec![0f32; n];
    let mut vis = vec![false; n];
    let cos_thres = 75f32.to_radians().cos();
    for k in 0..n {
        let f = r.face[k];
        if f == 0 {
            continue;
        }
        let t = s.tri[(f - 1) as usize];
        let b = r.bary[k];
        depth[k] = (0..3).map(|j| b[j] * v.camera_pos[t[j] as usize][2]).sum();
        let c = -fnrm[(f - 1) as usize][2];
        cos[k] = if c < cos_thres { 0.0 } else { c };
        vis[k] = true;
    }
    // the depth's 8-bit picture over the visible pixels, and its edges
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for k in 0..n {
        if vis[k] {
            lo = lo.min(depth[k]);
            hi = hi.max(depth[k]);
        }
    }
    let d8: Vec<f32> = (0..n).map(|k| if vis[k] { ((depth[k] - lo) / (hi - lo).max(1e-12) * 255.0).floor() } else { 0.0 }).collect();
    let mut edge = vec![false; n];
    for y in 0..size {
        for x in 0..size {
            let k = y * size + x;
            let mut e = false;
            if x + 1 < size {
                e |= (d8[k] - d8[k + 1]).abs() > DEPTH_EDGE;
            }
            if y + 1 < size {
                e |= (d8[k] - d8[k + size]).abs() > DEPTH_EDGE;
            }
            edge[k] = e;
        }
    }
    // the unreliable margin: 2 / 512 of the view's size around the silhouette and the depth edges
    let rad = 2 * size / 512;
    let invis: Vec<bool> = vis.iter().map(|v| !v).collect();
    let near_bg = any_within(&invis, size, rad);
    let near_edge = any_within(&edge, size, rad);
    for k in 0..n {
        if near_bg[k] || near_edge[k] {
            cos[k] = 0.0;
        }
    }
    let (p00, p11) = (v.cam.proj[0][0], v.cam.proj[1][1]);
    let mv = v.cam.mv;
    let fs = size as f32;
    let mut out = Baked { rgb: vec![[0.0; 3]; tex.len()], w: vec![0.0; tex.len()] };
    for (i, t) in tex.iter().enumerate() {
        let p = s.place(t.pos);
        let cam = [0, 1, 2].map(|r| mv[r][0] * p[0] + mv[r][1] * p[1] + mv[r][2] * p[2] + mv[r][3]);
        let (x, y, z) = (cam[0] * p00, cam[1] * p11, cam[2]);
        if !(-1.0..=1.0).contains(&x) || !(-1.0..=1.0).contains(&y) {
            continue;
        }
        let ix = (((x * 0.5 + 0.5) * fs) as i64).clamp(0, size as i64 - 1) as usize;
        let iy = (((y * 0.5 + 0.5) * fs) as i64).clamp(0, size as i64 - 1) as usize;
        let k = iy * size + ix;
        if (z - depth[k]).abs() >= DEPTH_TOLERANCE || !vis[k] || cos[k] <= 0.0 {
            continue;
        }
        let wx = (x * 0.5 + 0.5) * fs - ix as f32;
        let wy = (y * 0.5 + 0.5) * fs - iy as f32;
        let (xr, yr) = ((ix + 1).min(size - 1), (iy + 1).min(size - 1));
        let px = |yy: usize, xx: usize, c: usize| image[(yy * size + xx) * 3 + c];
        out.rgb[i] = [0, 1, 2].map(|c| {
            ((px(iy, ix, c) * (1.0 - wx) + px(iy, xr, c) * wx) * (1.0 - wy) + (px(yr, ix, c) * (1.0 - wx) + px(yr, xr, c) * wx) * wy) / 255.0
        });
        out.w[i] = cos[k];
    }
    out
}

/// The six views (8-bit values [size, size, 3]) baked into an atlas of `tsize` texels a side over the mesh's UVs:
/// RGB in [0, 1] per texel and whether a view saw it
pub fn bake(m: &Mesh, s: &Scene, views: &[Vec<f32>], size: usize, tsize: u32) -> (Vec<[f32; 3]>, Vec<bool>, Vec<Texel>) {
    let tex = texels(m, tsize);
    let n = (tsize * tsize) as usize;
    let mut merge = vec![[0f32; 3]; n];
    let mut trust = vec![0f32; n];
    for (vi, img) in views.iter().enumerate() {
        let b = back_project(s, &tex, img, super::render::ELEVS[vi], super::render::AZIMS[vi], size);
        // the texture's own (a texel two triangles cover keeps the last), weighted
        let mut tw = vec![0f32; n];
        let mut tc = vec![[0f32; 3]; n];
        for (i, t) in tex.iter().enumerate() {
            tw[t.index] = WEIGHTS[vi] * b.w[i].powi(BAKE_EXP);
            tc[t.index] = b.rgb[i];
        }
        let seen = tw.iter().filter(|w| **w > 0.0).count();
        let painted = tw.iter().zip(&trust).filter(|(w, t)| **w > 0.0 && **t > 0.0).count();
        if seen > 0 && painted as f64 / seen as f64 > 0.99 {
            continue;
        }
        for k in 0..n {
            if tw[k] > 0.0 {
                for c in 0..3 {
                    merge[k][c] += tc[k][c] * tw[k];
                }
                trust[k] += tw[k];
            }
        }
    }
    let rgb = merge.iter().zip(&trust).map(|(m, t)| m.map(|v| v / t.max(1e-8))).collect();
    (rgb, trust.iter().map(|t| *t > 1e-8).collect(), tex)
}

/// The atlas made whole: the unseen texels from the mesh's vertex colours (spread from the seen ones), the gutter
/// grown; an RGBA texture
pub fn inpaint(m: &Mesh, rgb: &[[f32; 3]], seen: &[bool], tex: &[Texel], tsize: u32) -> Texture {
    let s = tsize as usize;
    let nv = m.positions.len();
    // vertex colours: the mean of the seen texels nearest each vertex (barycentric weight > 0.5)
    let mut vc = vec![[0f64; 3]; nv];
    let mut vw = vec![0f64; nv];
    for t in tex {
        if !seen[t.index] {
            continue;
        }
        let tri = &m.indices[t.tri * 3..t.tri * 3 + 3];
        for j in 0..3 {
            let b = t.bary[j] as f64;
            if b > 0.5 {
                for c in 0..3 {
                    vc[tri[j] as usize][c] += b * rgb[t.index][c] as f64;
                }
                vw[tri[j] as usize] += b;
            }
        }
    }
    let mut known: Vec<bool> = vw.iter().map(|w| *w > 0.0).collect();
    let mut col: Vec<[f32; 3]> = (0..nv).map(|i| if known[i] { vc[i].map(|v| (v / vw[i]) as f32) } else { [0.5; 3] }).collect();
    // spread over the edges: an unknown vertex takes the distance-weighted mean of its known neighbours, until none
    // is left (or none can be reached)
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for t in m.indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            nb[a as usize].push(b);
            nb[b as usize].push(a);
        }
    }
    // vertices at one position (the atlas's seams) share what they know
    let (welded, map) = nextsycl_mesh::tools::weld(m, 1e-6);
    let mut at_pos: Vec<Vec<u32>> = vec![Vec::new(); welded.positions.len()];
    for (i, &w) in map.iter().enumerate() {
        at_pos[w as usize].push(i as u32);
    }
    for group in &at_pos {
        if group.len() > 1 {
            for &a in group {
                for &b in group {
                    if a != b {
                        nb[a as usize].push(b);
                    }
                }
            }
        }
    }
    for _ in 0..nv {
        let mut changed = false;
        let mut next = known.clone();
        for v in 0..nv {
            if known[v] {
                continue;
            }
            let (mut acc, mut wsum) = ([0f32; 3], 0f32);
            for &u in &nb[v] {
                if known[u as usize] {
                    let d = nextsycl_mesh::length(sub(m.positions[v], m.positions[u as usize])).max(1e-6);
                    let w = 1.0 / d;
                    for c in 0..3 {
                        acc[c] += col[u as usize][c] * w;
                    }
                    wsum += w;
                }
            }
            if wsum > 0.0 {
                col[v] = acc.map(|a| a / wsum);
                next[v] = true;
                changed = true;
            }
        }
        known = next;
        if !changed {
            break;
        }
    }
    let mut acc = vec![[0f32; 4]; s * s];
    let mut hit = vec![false; s * s];
    for t in tex {
        let c = if seen[t.index] {
            rgb[t.index]
        } else {
            let tri = &m.indices[t.tri * 3..t.tri * 3 + 3];
            [0, 1, 2].map(|k| (0..3).map(|j| t.bary[j] * col[tri[j] as usize][k]).sum())
        };
        acc[t.index] = [c[0], c[1], c[2], 1.0];
        hit[t.index] = true;
    }
    grow(&mut acc, &mut hit, s, 8);
    let mut out = Texture::new(tsize, tsize);
    for (px, c) in out.rgba.chunks_exact_mut(4).zip(&acc) {
        for (o, v) in px.iter_mut().zip(c) {
            *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_test_reaches_its_radius() {
        let s = 9;
        let mut m = vec![false; s * s];
        m[4 * s + 4] = true;
        let r = any_within(&m, s, 2);
        assert_eq!(r.iter().filter(|b| **b).count(), 25);
        assert!(r[2 * s + 2] && r[6 * s + 6] && !r[s + 4]);
    }
}
