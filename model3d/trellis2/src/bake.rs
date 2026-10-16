//! The texture decoder's voxels baked onto the final mesh: each texel's surface point (snapped to the decoded surface,
//! as the reference snaps to its BVH) samples the voxel attributes trilinearly - flex_gemm's grid_sample_3d: voxel c
//! centred at c + 0.5 in grid units g = (p + 0.5) res, the eight cells around g weighted 1 - |g - c - 0.5| per axis,
//! missing ones left out and the rest renormalized - into a base colour texture and a metallic-roughness one
//! (glTF's packing: roughness in G, metallic in B).

use nextsycl_model3d::mesh::tools::closest::Closest;
use nextsycl_model3d::mesh::{add, cross, normalize, scale, sub, Material, Mesh, Texture};

use crate::sparse::CoordMap;

/// Voxel attributes at a resolution: base colour (3), metallic, roughness, alpha - in [0, 1]
pub struct Voxels {
    map: CoordMap,
    attrs: Vec<f32>,
    pub res: usize,
}

pub const ATTRS: usize = 6;

impl Voxels {
    pub fn new(coords: &[[i32; 3]], attrs: Vec<f32>, res: usize) -> Voxels {
        Voxels { map: CoordMap::new(coords), attrs, res }
    }

    /// The attributes at a point of [-0.5, 0.5]^3
    pub fn sample(&self, p: [f32; 3]) -> [f32; ATTRS] {
        let g = p.map(|v| (v + 0.5) * self.res as f32);
        let base = g.map(|v| (v - 0.5).floor() as i32);
        let mut out = [0f32; ATTRS];
        let mut wsum = 0f32;
        for i in 0..8 {
            let c = [base[0] + (i & 1), base[1] + ((i >> 1) & 1), base[2] + ((i >> 2) & 1)];
            if c.iter().any(|v| *v < 0 || *v >= self.res as i32) {
                continue;
            }
            let Some(row) = self.map.get(c) else { continue };
            let w = (0..3).map(|a| 1.0 - (g[a] - c[a] as f32 - 0.5).abs()).product::<f32>();
            wsum += w;
            for (o, v) in out.iter_mut().zip(&self.attrs[row as usize * ATTRS..(row as usize + 1) * ATTRS]) {
                *o += w * v;
            }
        }
        let k = 1.0 / wsum.max(1e-12);
        out.map(|v| v * k)
    }
}

/// The base colour and metallic-roughness textures of `mesh` (with an atlas) from the voxels, each texel's point
/// snapped onto the decoded mesh (before it was simplified: `near`, its closest-point tree) first. The texels are found as
/// `tools::uv::rasterize` finds them (texel centres inside a triangle in UV space; the barycentric surface point and
/// normal), sampled on every thread, the empty ones grown from their neighbours.
pub fn bake(mesh: &Mesh, near: &Closest, vox: &Voxels, size: u32, push_voxels: f32) -> Material {
    // the nearest point of the decoded surface, looked for from two voxels outside along the texel's normal: on a
    // thin shell the simplified surface sits between the two sheets, and the outer one is the one seen
    let push = push_voxels / vox.res as f32;
    let s = size as usize;
    let texels = texels(mesh, size);
    let mut attrs = vec![[0f32; ATTRS]; texels.len()];
    crate::sparse::par_rows(&mut attrs, 1, |r0, chunk| {
        for (i, a) in chunk.iter_mut().enumerate() {
            let (_, p, n) = texels[r0 + i];
            let q = [p[0] + n[0] * push, p[1] + n[1] * push, p[2] + n[2] * push];
            *a = vox.sample(near.query(q).map_or(p, |(x, _, _)| x));
        }
    });
    // both textures' texels together (base colour RGBA, then metallic-roughness RGBA) and whether each is covered
    let mut cells = vec![Cell::default(); s * s];
    for ((t, _, _), a) in texels.iter().zip(&attrs) {
        cells[*t] = Cell { v: [a[0], a[1], a[2], 1.0, 0.0, a[4], a[3], 1.0], hit: true };
    }
    dilate(&mut cells, s, 8);
    let tex = |off: usize| {
        let mut t = Texture::new(size, size);
        for (px, c) in t.rgba.chunks_exact_mut(4).zip(&cells) {
            for (o, v) in px.iter_mut().zip(&c.v[off..off + 4]) {
                *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        t
    };
    Material { base_color: Some(tex(0)), metallic_roughness: Some(tex(4)), normal: None, base_color_factor: [1.0; 4], metallic: 1.0, roughness: 1.0 }
}

/// A texel of both textures: base colour RGBA, metallic-roughness RGBA; covered (or grown into) yet
#[derive(Clone, Copy, Default)]
struct Cell {
    v: [f32; 8],
    hit: bool,
}

/// Every texel a triangle covers (its centre inside, in UV space): (texel index, surface point, normal), the
/// triangles in order (a later one's texel after an earlier one's), found on every thread
fn texels(m: &Mesh, size: u32) -> Vec<(usize, [f32; 3], [f32; 3])> {
    let nt = m.indices.len() / 3;
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(32);
    let per = nt.div_ceil(threads).max(4096);
    let parts: Vec<Vec<(usize, [f32; 3], [f32; 3])>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt).step_by(per).map(|t0| sc.spawn(move || texels_of(m, size, t0, (t0 + per).min(nt)))).collect();
        hs.into_iter().map(|h| h.join().expect("a texel thread panicked")).collect()
    });
    parts.concat()
}

/// `texels` of triangles t0 .. t1
fn texels_of(m: &Mesh, size: u32, t0: usize, t1: usize) -> Vec<(usize, [f32; 3], [f32; 3])> {
    let s = size as usize;
    let mut out = Vec::new();
    for t in m.indices[t0 * 3..t1 * 3].chunks_exact(3) {
        let uv = [0, 1, 2].map(|k| {
            let u = m.uvs[t[k] as usize];
            [u[0] * size as f32, u[1] * size as f32]
        });
        let x0 = uv.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let x1 = (uv.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let y0 = uv.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let y1 = (uv.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil() as usize).min(s);
        let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
        if area.abs() < 1e-12 {
            continue;
        }
        let p = [0, 1, 2].map(|k| m.positions[t[k] as usize]);
        let fnrm = normalize(cross(sub(p[1], p[0]), sub(p[2], p[0])));
        for y in y0..y1 {
            for x in x0..x1 {
                let c = [x as f32 + 0.5, y as f32 + 0.5];
                let w = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / area;
                let (b0, b1, b2) = (w(uv[1], uv[2]), w(uv[2], uv[0]), w(uv[0], uv[1]));
                if b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4 {
                    continue;
                }
                let pos = add(add(scale(p[0], b0), scale(p[1], b1)), scale(p[2], b2));
                let nrm = if m.normals.len() == m.positions.len() {
                    let n = [0, 1, 2].map(|k| m.normals[t[k] as usize]);
                    normalize(add(add(scale(n[0], b0), scale(n[1], b1)), scale(n[2], b2)))
                } else {
                    fnrm
                };
                out.push((y * s + x, pos, nrm));
            }
        }
    }
    out
}

/// Empty texels next to filled ones take their mean, `passes` times (the seams' gutters); rows on every thread
fn dilate(cells: &mut [Cell], s: usize, passes: usize) {
    for _ in 0..passes {
        let prev = cells.to_vec();
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(32);
        let rows = s.div_ceil(threads).max(1);
        let pass = |y0: usize, chunk: &mut [Cell]| {
            for (yy, row) in chunk.chunks_exact_mut(s).enumerate() {
                let y = y0 + yy;
                for (x, c) in row.iter_mut().enumerate() {
                    if c.hit {
                        continue;
                    }
                    let mut sum = [0f32; 8];
                    let mut n = 0.0;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < s && (ny as usize) < s {
                            let p = &prev[ny as usize * s + nx as usize];
                            if p.hit {
                                sum.iter_mut().zip(p.v).for_each(|(a, b)| *a += b);
                                n += 1.0;
                            }
                        }
                    }
                    if n > 0.0 {
                        *c = Cell { v: sum.map(|v| v / n), hit: true };
                    }
                }
            }
        };
        std::thread::scope(|sc| {
            for (i, chunk) in cells.chunks_mut(rows * s).enumerate() {
                let pass = &pass;
                sc.spawn(move || pass(i * rows, chunk));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilinear_samples_renormalize_over_present_voxels() {
        // two voxels along x at res 4: colours 0 and 1 in the first channel
        let coords = vec![[1, 1, 1], [2, 1, 1]];
        let mut attrs = vec![0f32; 2 * ATTRS];
        attrs[ATTRS] = 1.0;
        let v = Voxels::new(&coords, attrs, 4);
        let centre = |c: [i32; 3]| c.map(|x| (x as f32 + 0.5) / 4.0 - 0.5);
        // at a voxel's centre: its own value
        assert!((v.sample(centre([1, 1, 1]))[0] - 0.0).abs() < 1e-6);
        assert!((v.sample(centre([2, 1, 1]))[0] - 1.0).abs() < 1e-6);
        // half way: the mean, though the other six corners are empty (renormalized, not darkened)
        let mid = [(centre([1, 1, 1])[0] + centre([2, 1, 1])[0]) / 2.0, centre([1, 1, 1])[1] + 0.05, centre([1, 1, 1])[2]];
        assert!((v.sample(mid)[0] - 0.5).abs() < 1e-5, "{}", v.sample(mid)[0]);
        // far from every voxel: zero
        assert_eq!(v.sample([0.45, 0.45, 0.45])[0], 0.0);
    }
}
