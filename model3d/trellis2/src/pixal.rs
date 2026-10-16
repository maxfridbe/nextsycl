//! Pixal3D's conditioning (TencentARC/Pixal3D, "proj" mode): every block cross-attends only to DINOv3's CLS and 4
//! register tokens, and adds `proj_linear` of DINOv3 patch features sampled where its token's grid point projects into
//! the picture - a pinhole camera of the picture's field of view (Blender's sensor of 32 mm), looking along +y from
//! (0, -d, 0) with z up, d placing the unit grid's half-width at the picture's edge (the reference's distance_from_fov
//! at 512 pixels); the grid's points linspace(-1, 1, R) rotated (x, -z, y) and halved; bilinear grid_sample,
//! align_corners=False, border padding. Each stage has its own extractor: the structure 512 px on its 16^3 grid, the
//! 512 shape pass 512 px on 32^3, the 1024 shape and texture passes 1024 px on (res / 16)^3.
//!
//! The shape and texture stages were trained on [low-res patches | NAF-upsampled patches] (2048 channels). NAF (a
//! learned, image-guided upsampler) is not ported: its half takes the low-res samples again - the fallback the
//! ComfyUI route allows, at some loss of the finest pixel-aligned detail.

use nextsycl_core::DevBuf;
use nextsycl_diffusion::kernels::Dt;
use nextsycl_diffusion::Picture;
use nextsycl_model3d::{Error, MeshRequest, Result};

use crate::{dino, Trellis2};

/// Blender's default field of view (a 35 mm lens on a 32 mm sensor): the reference's default camera_angle_x
pub const DEFAULT_FOV: f64 = 0.8575560450553894;

/// The camera the picture was taken with
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// horizontal field of view, radians
    pub fov: f32,
    pub distance: f32,
}

impl Camera {
    /// A field of view (radians) and the distance the reference derives from it
    pub fn new(fov: f64) -> Camera {
        // distance_from_fov((-1, 0, 0), target (0, 511), mesh_scale 1, 512 px): f_pixels * xw / x_ndc - yw
        let f_px = (16.0f32 / ((fov / 2.0) as f32).tan()) * 512.0 / 32.0;
        let distance = f_px * -0.5 / -256.0;
        Camera { fov: fov as f32, distance }
    }

    /// From a request's `fov` (degrees), else the reference's default
    pub fn from_request(req: &MeshRequest) -> Result<Camera> {
        let deg = req.extra.get("fov").map(|v| v.parse::<f64>().map_err(|_| Error(format!("fov: {v}?")))).transpose()?;
        let fov = deg.map_or(DEFAULT_FOV, |d| d.to_radians());
        if !(0.01..3.0).contains(&fov) {
            return Err(Error(format!("fov: {:.1} degrees is not a camera's", fov.to_degrees())));
        }
        Ok(Camera::new(fov))
    }

    /// The grid_sample coordinates ([-1, 1], x then y) of grid point (i, j, k) of an r^3 grid, seen in a picture of
    /// `px` pixels
    pub fn uv(&self, c: [i32; 3], r: usize, px: usize) -> [f32; 2] {
        let lin = |i: i32| -1.0 + 2.0 * i as f32 / (r - 1).max(1) as f32;
        let (x, y, z) = (lin(c[0]), lin(c[1]), lin(c[2]));
        // rotated (x, -z, y), halved; the camera's frame: x_cam = x', y_cam = z', z_cam = -y' - d
        let p = [x * 0.5, -z * 0.5, y * 0.5];
        let (xc, yc, zc) = (p[0], p[2], -p[1] - self.distance);
        let f = (16.0 / (self.fov / 2.0).tan()) * px as f32 / 32.0;
        let xn = f * xc / (-zc + 1e-8);
        let yn = f * yc / (-zc + 1e-8);
        let (xp, yp) = (xn + px as f32 / 2.0, -yn + px as f32 / 2.0);
        [(xp + 0.5) / px as f32 * 2.0 - 1.0, (yp + 0.5) / px as f32 * 2.0 - 1.0]
    }
}

/// A stage's condition: the global tokens (half [5, 1024], 5) and the projected features (half [n, proj_ch])
pub type Cond = (Option<(DevBuf, usize)>, Option<DevBuf>);

/// DINOv3's features of the picture at 512 (and 1024 for the cascade): [5 + p, 1024] float32 on the GPU
pub struct Conds {
    f512: DevBuf,
    f1024: Option<DevBuf>,
    cam: Camera,
}

impl Conds {
    pub fn new(e: &Trellis2, pre: &Picture, cam: &Camera, cascade: bool) -> Result<Conds> {
        let d = e.dino()?;
        let f512 = d.encode(&e.cx, &dino::input(pre, 512), 512)?;
        let f1024 = if cascade { Some(d.encode(&e.cx, &dino::input(pre, 1024), 1024)?) } else { None };
        Ok(Conds { f512, f1024, cam: *cam })
    }

    /// The global tokens and the features projected at `coords` of an r^3 grid from the picture's `size` features,
    /// `wide` (the NAF stages): [low-res | low-res again] (2048), else 1024
    fn project(&self, e: &Trellis2, size: usize, coords: &[[i32; 3]], r: usize, wide: bool) -> Result<Cond> {
        let f = if size == 512 { &self.f512 } else { self.f1024.as_ref().ok_or_else(|| Error("pixal3d: no 1024 features".into()))? };
        let g = &e.cx.gpu;
        let np = size / 16;
        let n = coords.len();
        let uv: Vec<f32> = coords.iter().flat_map(|c| self.cam.uv(*c, r, size)).collect();
        let uv = DevBuf::from_f32(g, &uv)?;
        let w = if wide { 2 * dino::DIM } else { dino::DIM };
        let out = DevBuf::f32(g, n * w)?;
        // SAFETY: the patch tokens start after CLS and the 4 registers, rows of 1024.
        let patches = unsafe { f.fp().add(5 * dino::DIM) };
        e.cx.k.grid_sample2d(patches, np, np, dino::DIM, uv.fp(), n, out.fp(), w)?;
        if wide {
            // SAFETY: the second half of each row of out.
            e.cx.k.grid_sample2d(patches, np, np, dino::DIM, uv.fp(), n, unsafe { out.fp().add(dino::DIM) }, w)?;
        }
        let proj = DevBuf::new(g, n * w * 2)?;
        e.cx.k.convert(out.ptr(), Dt::F32, n * w, proj.ptr(), Dt::F16)?;
        let glob = DevBuf::new(g, 5 * dino::DIM * 2)?;
        e.cx.k.convert(f.ptr(), Dt::F32, 5 * dino::DIM, glob.ptr(), Dt::F16)?;
        Ok((Some((glob, 5)), Some(proj)))
    }

    pub fn structure(&self, e: &Trellis2, grid: &[[i32; 3]]) -> Result<Cond> {
        self.project(e, 512, grid, 16, false)
    }
    pub fn shape(&self, e: &Trellis2, coords: &[[i32; 3]], res: usize) -> Result<Cond> {
        if res <= 512 { self.project(e, 512, coords, 32, true) } else { self.project(e, 1024, coords, res / 16, true) }
    }
    pub fn texture(&self, e: &Trellis2, coords: &[[i32; 3]], res: usize) -> Result<Cond> {
        self.project(e, 1024, coords, res / 16, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_edge_lands_on_the_picture_edge() {
        let cam = Camera::new(DEFAULT_FOV);
        // the grid point (-1, 0 depth, 0 height) on the 512 picture's left edge (pixel 0's left side + half a pixel)
        let uv = cam.uv([0, 8, 8], 17, 512);
        assert!((uv[0] - (0.5 / 512.0 * 2.0 - 1.0)).abs() < 1e-4, "{uv:?}");
        // the centre column's middle: the picture's centre
        let c = cam.uv([8, 8, 8], 17, 512);
        assert!((c[0] - 1.0 / 512.0).abs() < 1e-5 && (c[1] - 1.0 / 512.0).abs() < 1e-5, "{c:?}");
        // up (grid y, the third world axis) is up in the picture: smaller row
        assert!(cam.uv([8, 16, 8], 17, 512)[1] < c[1]);
    }
}
