//! 3D Gaussian splats: what an image-to-splat engine makes (Budget3D-Idol), the standard 3DGS PLY they travel as, and
//! a small CPU rasterizer for checks and thumbnails (the web page draws them with its own WebGL2 renderer).
//!
//! Values are kept activated, in the PLY's frame - 3DGS's, y down (OpenCV / COLMAP: the object's up is -y; Idol's
//! save_ply writes that, and 3DGS viewers expect it): positions, the degree-0 spherical-harmonic colour (`dc`:
//! colour = 0.5 + 0.2821 dc), opacity in [0, 1], scales (standard deviations along the rotated axes) and unit
//! rotation quaternions (w, x, y, z). The PLY stores what 3DGS viewers read: opacity as its logit, scales as logs.

/// SH degree 0's constant
pub const SH_C0: f32 = 0.282_094_8;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Splats {
    pub positions: Vec<[f32; 3]>,
    pub dc: Vec<[f32; 3]>,
    pub opacity: Vec<f32>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    /// the camera the picture was seen from, when the engine predicts it: yaw and pitch (radians, about the object's
    /// up axis and from its equator), distance, vertical field of view (degrees)
    pub camera: Option<[f32; 4]>,
}

impl Splats {
    pub fn with_capacity(n: usize) -> Splats {
        Splats { positions: Vec::with_capacity(n), dc: Vec::with_capacity(n), opacity: Vec::with_capacity(n), scales: Vec::with_capacity(n),
                 rotations: Vec::with_capacity(n), camera: None }
    }
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    pub fn push(&mut self, pos: [f32; 3], dc: [f32; 3], opacity: f32, scale: [f32; 3], rot: [f32; 4]) {
        self.positions.push(pos);
        self.dc.push(dc);
        self.opacity.push(opacity);
        self.scales.push(scale);
        self.rotations.push(rot);
    }

    /// From a z-up frame to the PLY's (y down: up is -y): +90 degrees about x, (x, y, z) -> (x, -z, y), each rotation turned
    /// with it (Idol's save_ply)
    pub fn z_up_to_y_up(&mut self) {
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let a = [h, h, 0.0, 0.0];
        for p in &mut self.positions {
            *p = [p[0], -p[2], p[1]];
        }
        for q in &mut self.rotations {
            *q = quat_mul(a, *q);
        }
    }

    /// The standard 3DGS PLY (binary little-endian): x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3, opacity
    /// as its logit and scales as logs (clamped so a saturated value stays finite)
    pub fn to_ply(&self) -> Vec<u8> {
        let n = self.len();
        let names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
        let mut out = format!("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n").into_bytes();
        for nm in names {
            out.extend_from_slice(format!("property float {nm}\n").as_bytes());
        }
        out.extend_from_slice(b"end_header\n");
        out.reserve(n * names.len() * 4);
        let eps = 1e-6f32;
        for i in 0..n {
            let o = self.opacity[i].clamp(eps, 1.0 - eps);
            let s = self.scales[i];
            let p = self.positions[i];
            let d = self.dc[i];
            let q = self.rotations[i];
            let row = [p[0], p[1], p[2], 0.0, 0.0, 0.0, d[0], d[1], d[2], (o / (1.0 - o)).ln(), s[0].max(eps).ln(), s[1].max(eps).ln(), s[2].max(eps).ln(),
                       q[0], q[1], q[2], q[3]];
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// A 3DGS PLY (binary little-endian floats; other properties, such as higher spherical harmonics, are skipped)
    pub fn from_ply(b: &[u8]) -> Result<Splats, String> {
        let end = b.windows(11).position(|w| w == b"end_header\n").ok_or("not a PLY (no end_header)")? + 11;
        let head = std::str::from_utf8(&b[..end]).map_err(|_| "a PLY header that is not text")?;
        if !head.starts_with("ply") || !head.contains("format binary_little_endian") {
            return Err("a binary little-endian PLY is read here".into());
        }
        let mut n = 0usize;
        let mut props = Vec::new();
        let mut in_vertex = false;
        for line in head.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w.as_slice() {
                ["element", "vertex", c] => {
                    n = c.parse().map_err(|_| "a bad vertex count")?;
                    in_vertex = true;
                }
                ["element", ..] => in_vertex = false,
                ["property", "float", name] | ["property", "float32", name] if in_vertex => props.push(name.to_string()),
                ["property", ..] if in_vertex => return Err(format!("{line}: float properties only")),
                _ => {}
            }
        }
        let stride = props.len() * 4;
        if b.len() < end + n * stride {
            return Err(format!("{} vertices of {} bytes do not fit the file", n, stride));
        }
        let at = |name: &str| props.iter().position(|p| p == name).ok_or_else(|| format!("no property {name}"));
        let ix: Vec<usize> = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
            .iter().map(|p| at(p)).collect::<Result<_, _>>()?;
        let mut s = Splats::with_capacity(n);
        for i in 0..n {
            let row = &b[end + i * stride..end + (i + 1) * stride];
            let f = |k: usize| {
                let o = ix[k] * 4;
                f32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]])
            };
            let mut q = [f(10), f(11), f(12), f(13)];
            let nq = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            q.iter_mut().for_each(|v| *v /= nq);
            s.push([f(0), f(1), f(2)], [f(3), f(4), f(5)], 1.0 / (1.0 + (-f(6)).exp()), [f(7).exp(), f(8).exp(), f(9).exp()], q);
        }
        Ok(s)
    }

    /// The bounding box of the splats that show (opacity over 1/255): (min, max)
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for (p, o) in self.positions.iter().zip(&self.opacity) {
            if *o > 1.0 / 255.0 {
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
        }
        (lo, hi)
    }
}

/// Hamilton product a * b of (w, x, y, z) quaternions
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// A picture of the splats from `eye` looking at `target` (`up` the world's up), vertical field of view `fov`
/// degrees, on a background of `bg`: the 3DGS image formation (EWA projection with its 0.3 px low-pass, alpha =
/// min(0.99, o exp(-q / 2)) dropped under 1/255, front to back in depth order) on the CPU
#[allow(clippy::too_many_arguments)]
pub fn render(s: &Splats, eye: [f32; 3], target: [f32; 3], up: [f32; 3], fov: f32, w: u32, h: u32, bg: [u8; 3]) -> crate::Texture {
    let f_ = norm(sub(target, eye));
    let r_ = norm(cross(f_, up));
    let d_ = cross(f_, r_);
    let (wf, hf) = (w as f32, h as f32);
    let fl = 0.5 * hf / (fov.to_radians() * 0.5).tan();
    struct G {
        u: f32,
        v: f32,
        z: f32,
        con: [f32; 3],
        rad: f32,
        o: f32,
        c: [f32; 3],
    }
    let mut gs = Vec::new();
    for i in 0..s.len() {
        if s.opacity[i] < 1.0 / 255.0 {
            continue;
        }
        let p = sub(s.positions[i], eye);
        let pc = [dot(p, r_), dot(p, d_), dot(p, f_)];
        if pc[2] < 0.02 {
            continue;
        }
        let rm = quat_rot(s.rotations[i]);
        let sc = s.scales[i];
        // the covariance in the camera's frame: (R_cam M)(R_cam M)^T, M = R diag(s)
        let rows = [r_, d_, f_];
        let mut a = [[0f32; 3]; 3];
        for (ri, rr) in rows.iter().enumerate() {
            for k in 0..3 {
                a[ri][k] = (0..3).map(|j| rr[j] * rm[j][k]).sum::<f32>() * sc[k];
            }
        }
        let cov = |x: usize, y: usize| (0..3).map(|k| a[x][k] * a[y][k]).sum::<f32>();
        let (x, y, z) = (pc[0], pc[1], pc[2]);
        let j = [[fl / z, 0.0, -fl * x / (z * z)], [0.0, fl / z, -fl * y / (z * z)]];
        let mut c2 = [[0f32; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                c2[r][c] = (0..3).map(|p| (0..3).map(|q| j[r][p] * cov(p, q) * j[c][q]).sum::<f32>()).sum();
            }
        }
        let (ca, cb, cc) = (c2[0][0] + 0.3, c2[0][1], c2[1][1] + 0.3);
        let det = (ca * cc - cb * cb).max(1e-12);
        let mid = 0.5 * (ca + cc);
        let lam = mid + (mid * mid - det).max(0.1).sqrt();
        let rad = (3.0 * lam.sqrt()).ceil();
        let (u, v) = (fl * x / z + 0.5 * wf, fl * y / z + 0.5 * hf);
        if u + rad < 0.0 || u - rad > wf || v + rad < 0.0 || v - rad > hf {
            continue;
        }
        let dc = s.dc[i];
        gs.push(G { u, v, z, con: [cc / det, -cb / det, ca / det], rad, o: s.opacity[i],
                    c: [(0.5 + SH_C0 * dc[0]).clamp(0.0, 1.0), (0.5 + SH_C0 * dc[1]).clamp(0.0, 1.0), (0.5 + SH_C0 * dc[2]).clamp(0.0, 1.0)] });
    }
    gs.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));
    let np = (w * h) as usize;
    let mut col = vec![[0f32; 3]; np];
    let mut tr = vec![1f32; np];
    for g in &gs {
        let (x0, x1) = (((g.u - g.rad).floor().max(0.0)) as u32, ((g.u + g.rad).ceil().min(wf - 1.0)).max(0.0) as u32);
        let (y0, y1) = (((g.v - g.rad).floor().max(0.0)) as u32, ((g.v + g.rad).ceil().min(hf - 1.0)).max(0.0) as u32);
        for py in y0..=y1 {
            for px in x0..=x1 {
                let k = (py * w + px) as usize;
                if tr[k] < 1e-4 {
                    continue;
                }
                let (dx, dy) = (px as f32 + 0.5 - g.u, py as f32 + 0.5 - g.v);
                let power = -0.5 * (g.con[0] * dx * dx + g.con[2] * dy * dy) - g.con[1] * dx * dy;
                if power > 0.0 {
                    continue;
                }
                let al = (g.o * power.exp()).min(0.99);
                if al < 1.0 / 255.0 {
                    continue;
                }
                let w = al * tr[k];
                for (o, c) in col[k].iter_mut().zip(g.c) {
                    *o += w * c;
                }
                tr[k] *= 1.0 - al;
            }
        }
    }
    let mut t = crate::Texture::new(w, h);
    for k in 0..np {
        for c in 0..3 {
            t.rgba[k * 4 + c] = ((col[k][c] + tr[k] * bg[c] as f32 / 255.0).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        t.rgba[k * 4 + 3] = 255;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one() -> Splats {
        let mut s = Splats::default();
        s.push([0.0, 0.0, 0.0], [1.0, -1.0, 0.0], 0.9, [0.1, 0.05, 0.02], [1.0, 0.0, 0.0, 0.0]);
        s.push([0.2, 0.1, -0.1], [0.0, 0.0, 0.0], 0.5, [0.02, 0.02, 0.02], [0.5f32.sqrt(), 0.5f32.sqrt(), 0.0, 0.0]);
        s
    }

    #[test]
    fn a_ply_round_trips() {
        let s = one();
        let b = s.to_ply();
        assert!(b.starts_with(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"));
        let r = Splats::from_ply(&b).unwrap();
        assert_eq!(r.len(), 2);
        for i in 0..2 {
            assert!((r.opacity[i] - s.opacity[i]).abs() < 1e-5);
            for c in 0..3 {
                assert!((r.scales[i][c] - s.scales[i][c]).abs() < 1e-6);
                assert_eq!(r.positions[i][c], s.positions[i][c]);
            }
        }
        assert!(Splats::from_ply(b"ply\nformat ascii 1.0\nend_header\n").is_err());
    }

    #[test]
    fn z_up_turns_to_y_up() {
        let mut s = one();
        s.positions[0] = [1.0, 2.0, 3.0];
        s.z_up_to_y_up();
        assert_eq!(s.positions[0], [1.0, -3.0, 2.0]);
        // the identity rotation becomes the turn itself
        let q = s.rotations[0];
        assert!((q[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6 && (q[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn a_splat_shows_in_the_middle_of_the_picture() {
        let mut s = Splats::default();
        s.push([0.0, 0.0, 0.0], [1.5, 1.5, 1.5], 0.95, [0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0]);
        let t = render(&s, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 40.0, 32, 32, [0, 0, 0]);
        let px = |x: usize, y: usize| t.rgba[(y * 32 + x) * 4];
        assert!(px(16, 16) > 150, "the centre is lit: {}", px(16, 16));
        assert_eq!(px(0, 0), 0, "the corner is background");
    }
}
