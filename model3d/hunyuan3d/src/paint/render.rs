//! The paint model's view of the mesh, as Tencent's MeshRender makes it (CPU): the mesh moved into the renderer's
//! frame - (-x, z, -y), centred, scaled so its farthest point is 0.575 from the centre - six orthographic cameras
//! (1.2 wide, 1.45 away; azimuths 0 / 90 / 180 / 270 and straight down and up), and custom_rasterizer's z-buffer
//! (pixel centres at (ndc / 2 + 1/2) (size - 1) + 1/2, depth quantized to 2^18 levels with the face index beside it,
//! rows bottom to top, no culling). From it: the normal maps (face normals in the renderer's frame, (n + 1) / 2, white
//! background), the position maps (1/2 - p / 1.15, interpolated, white background), and the depth, visibility and
//! camera-space normals the bake reads.

use nextsycl_mesh::{cross, normalize, sub, Mesh};

pub const ELEVS: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0];
pub const AZIMS: [f64; 6] = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0];
pub const WEIGHTS: [f32; 6] = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05];
const DISTANCE: f64 = 1.45;
const ORTHO: f64 = 1.2;
pub const SCALE_FACTOR: f32 = 1.15;
const MAXINT: i64 = 2147483647;

/// The mesh in the renderer's frame
pub struct Scene {
    pub pos: Vec<[f32; 3]>,
    pub tri: Vec<[u32; 3]>,
    /// mesh -> renderer: p' = (A p - centre) * scale, A = (-x, z, -y)
    pub centre: [f32; 3],
    pub scale: f32,
}

/// The mesh frame's point in the renderer's frame (before centring and scaling)
pub fn axes(p: [f32; 3]) -> [f32; 3] {
    [-p[0], p[2], -p[1]]
}

impl Scene {
    pub fn new(m: &Mesh) -> Scene {
        let a: Vec<[f32; 3]> = m.positions.iter().map(|&p| axes(p)).collect();
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &a {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let centre = [0, 1, 2].map(|k| (lo[k] + hi[k]) / 2.0);
        let far = a.iter().map(|p| nextsycl_mesh::length(sub(*p, centre))).fold(0f32, f32::max) * 2.0;
        let scale = SCALE_FACTOR / far.max(1e-12);
        let pos = a.iter().map(|p| [0, 1, 2].map(|k| (p[k] - centre[k]) * scale)).collect();
        Scene { pos, tri: m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect(), centre, scale }
    }

    /// A mesh-frame point in the renderer's frame
    pub fn place(&self, p: [f32; 3]) -> [f32; 3] {
        let a = axes(p);
        [0, 1, 2].map(|k| (a[k] - self.centre[k]) * self.scale)
    }
}

/// A camera: world -> camera (4 x 4), and the orthographic projection
pub struct Camera {
    pub mv: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

/// Tencent's get_mv_matrix (elevation negated, azimuth + 90, z up) and orthographic projection
pub fn camera(elev: f64, azim: f64) -> Camera {
    let (e, a) = ((-elev).to_radians(), (azim + 90.0).to_radians());
    let c = [DISTANCE * e.cos() * a.cos(), DISTANCE * e.cos() * a.sin(), DISTANCE * e.sin()];
    let n = |v: [f64; 3]| { let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); [v[0] / l, v[1] / l, v[2] / l] };
    let x = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let look = n([-c[0], -c[1], -c[2]]);
    let right = n(x(look, [0.0, 0.0, 1.0]));
    let up = n(x(right, look));
    // c2w columns: right, up, -look; w2c = its inverse
    let rows = [right, up, [-look[0], -look[1], -look[2]]];
    let mut mv = [[0f32; 4]; 4];
    for (i, r) in rows.iter().enumerate() {
        for k in 0..3 {
            mv[i][k] = r[k] as f32;
        }
        mv[i][3] = -(r[0] * c[0] + r[1] * c[1] + r[2] * c[2]) as f32;
    }
    mv[3][3] = 1.0;
    let (l, r, b, t, near, far) = (-ORTHO / 2.0, ORTHO / 2.0, -ORTHO / 2.0, ORTHO / 2.0, 0.1f64, 100.0f64);
    let mut proj = [[0f32; 4]; 4];
    proj[0][0] = (2.0 / (r - l)) as f32;
    proj[1][1] = (2.0 / (t - b)) as f32;
    proj[2][2] = (-2.0 / (far - near)) as f32;
    proj[0][3] = (-(r + l) / (r - l)) as f32;
    proj[1][3] = (-(t + b) / (t - b)) as f32;
    proj[2][3] = (-(far + near) / (far - near)) as f32;
    proj[3][3] = 1.0;
    Camera { mv, proj }
}

fn mul(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3] * v[3])
}

/// A rasterized view: per pixel (row-major, rows bottom up) its face (+1; 0 none) and barycentric weights
pub struct Raster {
    pub size: usize,
    pub face: Vec<u32>,
    pub bary: Vec<[f32; 3]>,
}

fn area2(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])
}

/// custom_rasterizer's rasterize_image (its CPU path), square: clip-space vertices [x, y, z, w], faces
pub fn rasterize(clip: &[[f32; 4]], tri: &[[u32; 3]], size: usize) -> Raster {
    let s1 = (size - 1) as f32;
    let px: Vec<[f32; 3]> = clip.iter().map(|v| [(v[0] / v[3] * 0.5 + 0.5) * s1 + 0.5, (0.5 + 0.5 * v[1] / v[3]) * s1 + 0.5, v[2] / v[3] * 0.49999 + 0.5]).collect();
    let mut z = vec![MAXINT * MAXINT + (MAXINT - 1); size * size];
    for (i, t) in tri.iter().enumerate() {
        let (a, b, c) = (px[t[0] as usize], px[t[1] as usize], px[t[2] as usize]);
        let ar = area2([a[0], a[1]], [b[0], b[1]], [c[0], c[1]]);
        if ar == 0.0 {
            continue;
        }
        let inv = 1.0 / ar;
        let (x0, x1) = (a[0].min(b[0]).min(c[0]) as i64, a[0].max(b[0]).max(c[0]));
        let (y0, y1) = (a[1].min(b[1]).min(c[1]) as i64, a[1].max(b[1]).max(c[1]));
        let mut x = x0;
        while (x as f32) < x1 + 1.0 {
            if x >= 0 && (x as usize) < size {
                let mut y = y0;
                while (y as f32) < y1 + 1.0 {
                    if y >= 0 && (y as usize) < size {
                        let p = [x as f32 + 0.5, y as f32 + 0.5];
                        let beta = area2([a[0], a[1]], p, [c[0], c[1]]) * inv;
                        let gamma = area2([a[0], a[1]], [b[0], b[1]], p) * inv;
                        let alpha = 1.0 - beta - gamma;
                        if (0.0..=1.0).contains(&alpha) && (0.0..=1.0).contains(&beta) && (0.0..=1.0).contains(&gamma) {
                            let depth = alpha * a[2] + beta * b[2] + gamma * c[2];
                            let tok = (depth * (2 << 17) as f32) as i32 as i64 * MAXINT + i as i64 + 1;
                            let k = y as usize * size + x as usize;
                            z[k] = z[k].min(tok);
                        }
                    }
                    y += 1;
                }
            }
            x += 1;
        }
    }
    let mut face = vec![0u32; size * size];
    let mut bary = vec![[0f32; 3]; size * size];
    for k in 0..size * size {
        let f = z[k] % MAXINT;
        if f == MAXINT - 1 {
            continue;
        }
        face[k] = f as u32;
        let t = tri[(f - 1) as usize];
        let p = [(k % size) as f32 + 0.5, (k / size) as f32 + 0.5];
        let (a, b, c) = (px[t[0] as usize], px[t[1] as usize], px[t[2] as usize]);
        let ar = area2([a[0], a[1]], [b[0], b[1]], [c[0], c[1]]);
        if ar == 0.0 {
            bary[k] = [-1.0; 3];
            continue;
        }
        let beta = area2([a[0], a[1]], p, [c[0], c[1]]) / ar;
        let gamma = area2([a[0], a[1]], [b[0], b[1]], p) / ar;
        let w = [clip[t[0] as usize][3], clip[t[1] as usize][3], clip[t[2] as usize][3]];
        let bb = [(1.0 - beta - gamma) / w[0], beta / w[1], gamma / w[2]];
        let sum = bb[0] + bb[1] + bb[2];
        bary[k] = bb.map(|v| v / sum);
    }
    Raster { size, face, bary }
}

/// A view of the scene: its camera-space positions, clip positions and raster
pub struct View {
    pub cam: Camera,
    pub camera_pos: Vec<[f32; 3]>,
    pub raster: Raster,
}

pub fn view(s: &Scene, elev: f64, azim: f64, size: usize) -> View {
    let cam = camera(elev, azim);
    let pc: Vec<[f32; 4]> = s.pos.iter().map(|p| mul(&cam.mv, [p[0], p[1], p[2], 1.0])).collect();
    let clip: Vec<[f32; 4]> = pc.iter().map(|p| mul(&cam.proj, *p)).collect();
    let raster = rasterize(&clip, &s.tri, size);
    View { cam, camera_pos: pc.iter().map(|p| [p[0] / p[3], p[1] / p[3], p[2] / p[3]]).collect(), raster }
}

fn face_normal(p: &[[f32; 3]], t: [u32; 3]) -> [f32; 3] {
    normalize(cross(sub(p[t[1] as usize], p[t[0] as usize]), sub(p[t[2] as usize], p[t[0] as usize])))
}

/// The normal map (face normals in the renderer's frame) as 8-bit values [size, size, 3] (rows bottom up)
pub fn normal_map(s: &Scene, v: &View) -> Vec<f32> {
    let r = &v.raster;
    let normals: Vec<[f32; 3]> = s.tri.iter().map(|&t| face_normal(&s.pos, t)).collect();
    r.face.iter().flat_map(|&f| {
        let n = if f > 0 { normals[(f - 1) as usize] } else { [1.0; 3] };
        n.map(|x| if f > 0 { ((x + 1.0) * 0.5 * 255.0).floor() } else { 255.0 })
    }).collect()
}

/// The position map (1/2 - p / 1.15, interpolated) as 8-bit values [size, size, 3]
pub fn position_map(s: &Scene, v: &View) -> Vec<f32> {
    let r = &v.raster;
    r.face.iter().zip(&r.bary).flat_map(|(&f, b)| {
        if f == 0 {
            return [255.0; 3];
        }
        let t = s.tri[(f - 1) as usize];
        [0, 1, 2].map(|k| {
            let val: f32 = (0..3).map(|j| b[j] * (0.5 - s.pos[t[j] as usize][k] / SCALE_FACTOR)).sum();
            (val * 255.0).clamp(0.0, 255.0).floor()
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_front_camera_looks_down_minus_y() {
        // azimuth 0: the camera sits at +y (the mesh's front, +z, in the renderer's frame) looking back at the origin
        let c = camera(0.0, 0.0);
        let p = mul(&c.mv, [0.0, 0.0, 0.0, 1.0]);
        assert!((p[2] + 1.45).abs() < 1e-5, "{p:?}");
        let q = mul(&c.mv, [0.0, 1.0, 0.0, 1.0]);
        assert!(q[2] > p[2]);
        // the renderer's +z (the mesh's up) is the camera's up
        let u = mul(&c.mv, [0.0, 0.0, 1.0, 1.0]);
        assert!((u[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_square_fills_its_pixels() {
        // two triangles over the whole clip square
        let clip = [[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0]];
        let r = rasterize(&clip, &[[0, 1, 2], [0, 2, 3]], 8);
        assert!(r.face.iter().all(|&f| f > 0));
        for b in &r.bary {
            assert!((b[0] + b[1] + b[2] - 1.0).abs() < 1e-5);
        }
    }
}
