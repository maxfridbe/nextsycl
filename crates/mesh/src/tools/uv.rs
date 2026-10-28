//! A fresh UV atlas and textures baked into it. `unwrap`: charts grown from the largest triangles while their
//! normals stay within ~60 degrees of the chart's (so each projects flat onto its own plane without folding), laid
//! out by shelf packing in a square with a gutter, the vertices split along the chart seams. `rasterize`: every
//! texel a triangle covers in UV space, its surface point and normal handed to a sampler (an engine's voxel colours,
//! a paint model's views); `bake`: the old surface's texture or colours carried onto the new atlas through the
//! closest point. Empty texels next to filled ones are grown into (the gutter), so filtering never reads black.

use super::closest::Closest;
use crate::{add, cross, dot, length, normalize, scale, sub, Material, Mesh, Texture};

/// Charts by region growing: each triangle's chart index
fn charts(m: &Mesh, cos_limit: f32) -> (Vec<usize>, usize) {
    let tris: Vec<[u32; 3]> = m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    let fn_: Vec<[f32; 3]> = tris.iter().map(|t| cross(sub(m.positions[t[1] as usize], m.positions[t[0] as usize]), sub(m.positions[t[2] as usize], m.positions[t[0] as usize]))).collect();
    let mut by_edge: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let mut order: Vec<usize> = (0..tris.len()).collect();
    order.sort_by(|&a, &b| length(fn_[b]).total_cmp(&length(fn_[a])));
    let mut chart = vec![usize::MAX; tris.len()];
    let mut n = 0;
    for seed in order {
        if chart[seed] != usize::MAX {
            continue;
        }
        let mut axis = fn_[seed];
        let mut stack = vec![seed];
        chart[seed] = n;
        while let Some(f) = stack.pop() {
            let t = tris[f];
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                for &g in &by_edge[&(a.min(b), a.max(b))] {
                    if chart[g] != usize::MAX {
                        continue;
                    }
                    let gn = fn_[g];
                    if length(gn) == 0.0 || dot(normalize(gn), normalize(axis)) >= cos_limit {
                        chart[g] = n;
                        axis = add(axis, gn);
                        stack.push(g);
                    }
                }
            }
        }
        n += 1;
    }
    (chart, n)
}

/// The mesh with a new atlas in `uvs` (vertices split at chart seams; normals and colours follow, the skin and the
/// animation too); `size` and `gutter` (texels) set the spacing between charts
pub fn unwrap(m: &Mesh, size: u32, gutter: u32) -> Mesh {
    let (chart, nc) = charts(m, 0.5);
    let tris: Vec<[u32; 3]> = m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    // each chart's plane: its area-weighted normal, two axes across it
    let mut axis = vec![[0f32; 3]; nc];
    for (f, t) in tris.iter().enumerate() {
        let n = cross(sub(m.positions[t[1] as usize], m.positions[t[0] as usize]), sub(m.positions[t[2] as usize], m.positions[t[0] as usize]));
        axis[chart[f]] = add(axis[chart[f]], n);
    }
    let basis: Vec<([f32; 3], [f32; 3])> = axis.iter().map(|a| {
        let n = normalize(*a);
        let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = normalize(cross(helper, n));
        (u, cross(n, u))
    }).collect();
    // new vertices: one per (chart, old vertex)
    let mut out = Mesh { material: m.material.clone(), ..Default::default() };
    let mut key: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
    let mut flat: Vec<[f32; 2]> = Vec::new();
    let mut of_chart: Vec<usize> = Vec::new();
    let mut src: Vec<u32> = Vec::new();
    for (f, t) in tris.iter().enumerate() {
        let c = chart[f];
        for &v in t {
            let id = *key.entry((c, v)).or_insert_with(|| {
                let p = m.positions[v as usize];
                flat.push([dot(p, basis[c].0), dot(p, basis[c].1)]);
                of_chart.push(c);
                src.push(v);
                (flat.len() - 1) as u32
            });
            out.indices.push(id);
        }
    }
    out.positions = src.iter().map(|&v| m.positions[v as usize]).collect();
    if !m.normals.is_empty() {
        out.normals = src.iter().map(|&v| m.normals[v as usize]).collect();
    }
    if !m.colors.is_empty() {
        out.colors = src.iter().map(|&v| m.colors[v as usize]).collect();
    }
    if let Some(s) = &m.skin {
        out.skin = Some(crate::Skin { joints: s.joints.clone(), weights: src.iter().map(|&v| s.weights[v as usize]).collect(), motion: s.motion.clone() });
    }
    if let Some(a) = &m.animation {
        out.animation = Some(crate::Morph { fps: a.fps, frames: a.frames.iter().map(|f| src.iter().map(|&v| f[v as usize]).collect()).collect() });
    }
    // chart boxes, then shelves in a square grown until they fit
    let mut lo = vec![[f32::INFINITY; 2]; nc];
    let mut hi = vec![[f32::NEG_INFINITY; 2]; nc];
    for (i, p) in flat.iter().enumerate() {
        let c = of_chart[i];
        for k in 0..2 {
            lo[c][k] = lo[c][k].min(p[k]);
            hi[c][k] = hi[c][k].max(p[k]);
        }
    }
    let dim: Vec<[f32; 2]> = (0..nc).map(|c| [(hi[c][0] - lo[c][0]).max(1e-9), (hi[c][1] - lo[c][1]).max(1e-9)]).collect();
    let area: f32 = dim.iter().map(|d| d[0] * d[1]).sum();
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| dim[b][1].total_cmp(&dim[a][1]));
    let mut side = area.sqrt() * 1.15;
    let place = loop {
        let pad = side * gutter as f32 / size.max(1) as f32;
        let (mut x, mut y, mut row_h) = (pad, pad, 0f32);
        let mut at = vec![[0f32; 2]; nc];
        let mut fits = true;
        for &c in &order {
            if x + dim[c][0] + pad > side {
                x = pad;
                y += row_h + pad;
                row_h = 0.0;
            }
            if x + dim[c][0] + pad > side {
                fits = false; // a chart wider than the square
                break;
            }
            at[c] = [x, y];
            x += dim[c][0] + pad;
            row_h = row_h.max(dim[c][1]);
        }
        if fits && y + row_h + pad <= side {
            break at;
        }
        side *= 1.05;
    };
    out.uvs = flat.iter().enumerate().map(|(i, p)| {
        let c = of_chart[i];
        [(p[0] - lo[c][0] + place[c][0]) / side, (p[1] - lo[c][1] + place[c][1]) / side]
    }).collect();
    // keep each triangle's UV winding the same as its 3D one (a mirrored chart reads its texture backwards)
    out
}

/// A texel a triangle covers in UV space: its index in the texture (row-major, v down), the triangle, the
/// barycentric weights of its centre, and the surface point and normal there
#[derive(Clone, Copy, Debug)]
pub struct Texel {
    pub index: usize,
    pub tri: usize,
    pub bary: [f32; 3],
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// Every texel a triangle covers in UV space (sampled at the texel centre; a texel two triangles cover is listed
/// for each)
pub fn texels(m: &Mesh, size: u32) -> Vec<Texel> {
    let s = size as usize;
    let mut out = Vec::new();
    let tri_n = |t: &[u32]| normalize(cross(sub(m.positions[t[1] as usize], m.positions[t[0] as usize]), sub(m.positions[t[2] as usize], m.positions[t[0] as usize])));
    for (ti, t) in m.indices.chunks_exact(3).enumerate() {
        let uv = [0, 1, 2].map(|k| { let u = m.uvs[t[k] as usize]; [u[0] * size as f32, u[1] * size as f32] });
        let x0 = uv.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let x1 = (uv.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let y0 = uv.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let y1 = (uv.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
        if area.abs() < 1e-12 {
            continue;
        }
        let fnrm = tri_n(t);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let w = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / area;
                let (b0, b1, b2) = (w(uv[1], uv[2]), w(uv[2], uv[0]), w(uv[0], uv[1]));
                if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
                    continue;
                }
                let pos = add(add(scale(m.positions[t[0] as usize], b0), scale(m.positions[t[1] as usize], b1)), scale(m.positions[t[2] as usize], b2));
                let nrm = if m.normals.is_empty() { fnrm } else {
                    normalize(add(add(scale(m.normals[t[0] as usize], b0), scale(m.normals[t[1] as usize], b1)), scale(m.normals[t[2] as usize], b2)))
                };
                out.push(Texel { index: y * s + x, tri: ti, bary: [b0, b1, b2], pos, normal: nrm });
            }
        }
    }
    out
}

/// Every texel a triangle covers in UV space (sampled at the texel centre), with the surface point and normal
/// there, handed to `sample`; the texture it fills (texels no triangle covers grown from their neighbours)
pub fn rasterize(m: &Mesh, size: u32, mut sample: impl FnMut([f32; 3], [f32; 3]) -> [f32; 4]) -> Texture {
    let s = size as usize;
    let mut acc = vec![[0f32; 4]; s * s];
    let mut hit = vec![false; s * s];
    for t in texels(m, size) {
        acc[t.index] = sample(t.pos, t.normal);
        hit[t.index] = true;
    }
    grow(&mut acc, &mut hit, s, 8);
    let mut t = Texture::new(size, size);
    for (px, c) in t.rgba.chunks_exact_mut(4).zip(&acc) {
        for (o, v) in px.iter_mut().zip(c) {
            *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    t
}

/// Empty texels (of an s x s texture) next to filled ones take their mean, `passes` times: a texture's gutter
pub fn grow(acc: &mut [[f32; 4]], hit: &mut [bool], s: usize, passes: usize) {
    for _ in 0..passes {
        let (prev, was) = (acc.to_vec(), hit.to_vec());
        for y in 0..s {
            for x in 0..s {
                if was[y * s + x] {
                    continue;
                }
                let mut sum = [0f32; 4];
                let mut n = 0.0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < s && (ny as usize) < s && was[ny as usize * s + nx as usize] {
                        let c = prev[ny as usize * s + nx as usize];
                        for k in 0..4 {
                            sum[k] += c[k];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    acc[y * s + x] = sum.map(|v| v / n);
                    hit[y * s + x] = true;
                }
            }
        }
    }
}

/// `target` (with an atlas) textured from `source`: its base colour texture through its UVs, else its vertex
/// colours, at the closest point of the source surface; metallic-roughness too when the source has it
pub fn bake(target: &Mesh, source: &Mesh, size: u32) -> Result<Material, String> {
    if target.uvs.len() != target.positions.len() {
        return Err("the target has no atlas (unwrap it first)".into());
    }
    let near = Closest::new(&source.positions, &source.indices);
    let src_mat = source.material.clone().unwrap_or_default();
    let lerp_uv = |t: [u32; 3], bc: [f32; 3]| -> [f32; 2] {
        let mut u = [0f32; 2];
        for k in 0..3 {
            let s = source.uvs[t[k] as usize];
            u[0] += bc[k] * s[0];
            u[1] += bc[k] * s[1];
        }
        u
    };
    let has_uv = source.uvs.len() == source.positions.len();
    let color = |p: [f32; 3]| -> [f32; 4] {
        let Some((_, ti, bc)) = near.query(p) else { return [1.0; 4] };
        let t = near.triangle(ti);
        match (&src_mat.base_color, has_uv) {
            (Some(tex), true) => {
                let c = tex.sample(lerp_uv(t, bc)[0], lerp_uv(t, bc)[1]);
                [0, 1, 2, 3].map(|k| c[k] * src_mat.base_color_factor[k])
            }
            _ if !source.colors.is_empty() => {
                let mut c = [0f32; 4];
                for k in 0..3 {
                    for (ch, x) in c.iter_mut().enumerate() {
                        *x += bc[k] * source.colors[t[k] as usize][ch];
                    }
                }
                c
            }
            _ => src_mat.base_color_factor,
        }
    };
    let base = rasterize(target, size, |p, _| color(p));
    let mr = match (&src_mat.metallic_roughness, has_uv) {
        (Some(tex), true) => Some(rasterize(target, size, |p, _| {
            near.query(p).map_or([0.0, 1.0, 0.0, 1.0], |(_, ti, bc)| { let u = lerp_uv(near.triangle(ti), bc); tex.sample(u[0], u[1]) })
        })),
        _ => None,
    };
    Ok(Material { base_color: Some(base), metallic_roughness: mr, normal: None, base_color_factor: [1.0; 4], metallic: src_mat.metallic, roughness: src_mat.roughness })
}

#[cfg(test)]
mod tests {
    use super::super::testing::sphere;
    use super::*;

    #[test]
    fn an_atlas_covers_every_triangle_without_overlap() {
        let mut m = sphere(24, 48);
        m.compute_normals();
        let u = unwrap(&m, 512, 2);
        assert_eq!(u.triangles(), m.triangles());
        assert!(u.uvs.iter().all(|t| (0.0..=1.0).contains(&t[0]) && (0.0..=1.0).contains(&t[1])));
        // no texel claimed by two triangles: rasterize triangle ids
        let s = 512usize;
        let mut owner = vec![u32::MAX; s * s];
        let mut clashes = 0;
        for (ti, t) in u.indices.chunks_exact(3).enumerate() {
            let uv = [0, 1, 2].map(|k| { let p = u.uvs[t[k] as usize]; [p[0] * s as f32, p[1] * s as f32] });
            let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
            for y in 0..s {
                for x in 0..s {
                    let p = [x as f32 + 0.5, y as f32 + 0.5];
                    let w = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / area;
                    if w(uv[1], uv[2]) > 0.01 && w(uv[2], uv[0]) > 0.01 && w(uv[0], uv[1]) > 0.01 {
                        if owner[y * s + x] != u32::MAX {
                            clashes += 1;
                        }
                        owner[y * s + x] = ti as u32;
                    }
                }
            }
        }
        assert_eq!(clashes, 0, "charts overlap in the atlas");
    }

    #[test]
    fn colours_bake_across() {
        let mut m = sphere(16, 32);
        // red above the equator, blue below
        m.colors = m.positions.iter().map(|p| if p[1] > 0.0 { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] }).collect();
        let u = unwrap(&m, 256, 2);
        let mat = bake(&u, &m, 256).unwrap();
        let tex = mat.base_color.unwrap();
        // the top pole's texel is red
        let top = u.positions.iter().position(|p| p[1] > 0.999).unwrap();
        let c = tex.sample(u.uvs[top][0], u.uvs[top][1]);
        assert!(c[0] > 0.8 && c[2] < 0.2, "{c:?}");
    }
}
