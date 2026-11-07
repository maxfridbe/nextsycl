//! Background removal as an image engine (`image.remove_background`): a picture in, the picture as RGBA (its own RGB,
//! straight alpha) and its mask out, both at the picture's size.
//!
//! - `birefnet`: the foundation's remover (`nextsycl-remover`: ZhengPeng7/BiRefNet, Swin-L, float32 on the GPU), the
//!   one the 3D engines cut their pictures out with - its own preparation and mask (TRELLIS.2's rembg.BiRefNet).
//! - `isnet`: IS-Net general-use as rembg runs it (`isnet.rs`), its preparation and mask rembg's own: Pillow's Lanczos
//!   to 1024, divided by the picture's largest value, less 0.5; the prediction scaled to its own [min, max], 8 bits,
//!   Lanczos back to the picture's size.
//!
//! `post_process_mask` is rembg's post_process_mask for either: an opening (OpenCV's 3 x 3 ellipse, a cross), OpenCV's
//! 5 x 5 Gaussian (sigma 2, borders reflected 101), a threshold at 127 - the platform asks rembg for it. After
//! nextcuda's image/rembg (the same API, the same checks).

pub mod check;
pub mod ffi;
pub mod isnet;
pub mod onnx;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};
pub use nextsycl_remover::Remover;

pub const ARCH_BIREFNET: &str = "birefnet";
pub const ARCH_ISNET: &str = "isnet";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH_BIREFNET, ARCH_ISNET], name: "background removal: BiRefNet, IS-Net general-use (Rust + SYCL)", roles: &["remover"], load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().cloned().ok_or_else(|| Error("a background remover needs a GPU".into()))?;
    if let Some(p) = files.get("isnet") {
        let t0 = Instant::now();
        let net = isnet::IsNet::load(&onnx::Model::open(p)?, &gpu)?;
        let secs = t0.elapsed().as_secs_f64();
        log(format!("isnet: loaded in {secs:.1} s on {} ({:.0} MB)", gpu.name, net.bytes as f64 / 1e6));
        return Ok(Box::new(IsNetEngine { net: Mutex::new(net), load_secs: secs }));
    }
    let path: PathBuf = files.get("remover").or_else(|| files.values().next()).cloned()
        .ok_or_else(|| Error("birefnet: no remover file (the card names it)".into()))?;
    let r = Remover::load(&path, &gpu)?;
    log(format!("birefnet: loaded in {:.1} s on {}", r.load_seconds, gpu.name));
    Ok(Box::new(BiRefNet { load_secs: r.load_seconds, r: Mutex::new(r) }))
}

/// The picture's RGB with the mask as its alpha: straight, not premultiplied (rembg's naive_cutout multiplies; this
/// keeps the colours as they were)
pub fn with_alpha(pic: &Picture, mask: &Picture) -> Picture {
    let rgb = pic.rgb();
    Picture { width: pic.width, height: pic.height, channels: 4,
              data: rgb.data.chunks_exact(3).zip(&mask.data).flat_map(|(p, a)| [p[0], p[1], p[2], *a]).collect() }
}

/// What both engines say a request to draw
fn no_drawing(arch: &str) -> Error {
    Error(format!("{arch} removes backgrounds (image.remove_background); it does not draw"))
}

fn no_defaults() -> Defaults {
    Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
}

pub struct BiRefNet {
    r: Mutex<Remover>,
    load_secs: f64,
}

impl ImageEngine for BiRefNet {
    fn arch(&self) -> &'static str {
        ARCH_BIREFNET
    }
    fn defaults(&self) -> Defaults {
        no_defaults()
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(no_drawing(ARCH_BIREFNET))
    }
    fn removes_background(&self) -> bool {
        true
    }
    fn remove_background(&self, pic: &Picture, post_process: bool) -> Result<(Picture, Picture)> {
        let r = self.r.lock().unwrap_or_else(|p| p.into_inner());
        let mut mask = r.mask(pic)?;
        if post_process {
            mask = post_process_mask(&mask);
        }
        Ok((with_alpha(pic, &mask), mask))
    }
}

pub struct IsNetEngine {
    net: Mutex<isnet::IsNet>,
    load_secs: f64,
}

impl IsNetEngine {
    /// rembg's input: Pillow's Lanczos to 1024 x 1024, divided by its largest value, less 0.5 (std 1), channels-last -
    /// in float64 as numpy does it, then float32
    pub fn input(pic: &Picture) -> Vec<f32> {
        let r = pic.rgb().resize_lanczos(isnet::SIDE as u32, isnet::SIDE as u32);
        let mx = (*r.data.iter().max().unwrap_or(&0) as f64).max(1e-6);
        r.data.iter().map(|&v| (v as f64 / mx - 0.5) as f32).collect()
    }

    /// rembg's mask from the prediction: scaled to its [min, max] (float32), times 255 truncated, Lanczos to w x h
    pub fn mask_of(pred: &[f32], w: u32, h: u32) -> Picture {
        let (mi, ma) = pred.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let d = ma - mi;
        let m = Picture { width: isnet::SIDE as u32, height: isnet::SIDE as u32, channels: 1,
                          data: pred.iter().map(|&v| if d > 0.0 { (((v - mi) / d) * 255.0) as u8 } else { 0 }).collect() };
        m.resize_lanczos(w, h)
    }
}

impl ImageEngine for IsNetEngine {
    fn arch(&self) -> &'static str {
        ARCH_ISNET
    }
    fn defaults(&self) -> Defaults {
        no_defaults()
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(no_drawing(ARCH_ISNET))
    }
    fn removes_background(&self) -> bool {
        true
    }
    fn remove_background(&self, pic: &Picture, post_process: bool) -> Result<(Picture, Picture)> {
        let net = self.net.lock().unwrap_or_else(|p| p.into_inner());
        let pred = net.forward(&Self::input(pic))?;
        let mut mask = Self::mask_of(&pred, pic.width, pic.height);
        if post_process {
            mask = post_process_mask(&mask);
        }
        Ok((with_alpha(pic, &mask), mask))
    }
}

/// cv2.morphologyEx(MORPH_OPEN) with the 3 x 3 ellipse (a cross): erode, then dilate; outside the picture takes no
/// part (OpenCV's default morphology border)
pub fn open3(d: &[u8], w: usize, h: usize) -> Vec<u8> {
    let pass = |d: &[u8], min: bool| -> Vec<u8> {
        let mut o = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut v = d[y * w + x];
                let mut take = |u: u8| v = if min { v.min(u) } else { v.max(u) };
                if x > 0 {
                    take(d[y * w + x - 1]);
                }
                if x + 1 < w {
                    take(d[y * w + x + 1]);
                }
                if y > 0 {
                    take(d[(y - 1) * w + x]);
                }
                if y + 1 < h {
                    take(d[(y + 1) * w + x]);
                }
                o[y * w + x] = v;
            }
        }
        o
    };
    pass(&pass(d, true), false)
}

/// BORDER_REFLECT_101 (OpenCV's default): -1 -> 1, n -> n - 2; a side of one pixel is that pixel
fn reflect101(i: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    let mut i = i;
    // a kernel wider than the side reflects more than once
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * n - 2 - i };
    }
    i as usize
}

/// cv2.GaussianBlur(5 x 5, sigma 2, BORDER_REFLECT_101) of an 8-bit picture as OpenCV computes it: the 1-D kernel in
/// fixed point (8 fraction bits, its sum 256), the rows then the columns in integers, rounded once at the end
pub fn blur5(d: &[u8], w: usize, h: usize) -> Vec<u8> {
    let k: Vec<f64> = (-2i64..=2).map(|i| (-((i * i) as f64) / (2.0 * 4.0)).exp()).collect();
    let ks: f64 = k.iter().sum();
    let kf: Vec<i64> = k.iter().map(|v| (v / ks * 256.0).round() as i64).collect();
    let (wi, hi) = (w as i64, h as i64);
    let mut tmp = vec![0i64; w * h];
    for y in 0..h {
        let row = &d[y * w..(y + 1) * w];
        for x in 0..wi {
            tmp[y * w + x as usize] = (0..5).map(|t| kf[t] * row[reflect101(x + t as i64 - 2, wi)] as i64).sum();
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..hi {
        let rows: Vec<usize> = (0..5).map(|t| reflect101(y + t as i64 - 2, hi) * w).collect();
        for x in 0..w {
            let s: i64 = (0..5).map(|t| kf[t] * tmp[rows[t] + x]).sum();
            out[y as usize * w + x] = ((s + (1 << 15)) >> 16).clamp(0, 255) as u8;
        }
    }
    out
}

/// rembg's post_process_mask: the opening, the blur, then 0 below 127, else 255
pub fn post_process_mask(m: &Picture) -> Picture {
    let (w, h) = (m.width as usize, m.height as usize);
    let b = blur5(&open3(&m.data, w, h), w, h);
    Picture { width: m.width, height: m.height, channels: 1, data: b.iter().map(|&v| if v < 127 { 0 } else { 255 }).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // fixtures from OpenCV 5.0.0 (opencv-python-headless, rembg 2.0.67's environment): rembg's kernel
    // getStructuringElement(MORPH_ELLIPSE, (3, 3)), morphologyEx(MORPH_OPEN), GaussianBlur((5, 5), sigmaX=2, sigmaY=2,
    // borderType=BORDER_DEFAULT); 8 x 6 pictures, rows of 8
    const W: usize = 8;
    const H: usize = 6;

    fn ramp() -> Vec<u8> {
        (0..H).flat_map(|y| (0..W).map(move |x| ((x * 37 + y * 91 + x * y * 13) % 256) as u8)).collect()
    }

    fn edge() -> Vec<u8> {
        (0..H).flat_map(|y| (0..W).map(move |x| if x > 2 { ((x * 3 + y * 5 + (x * y) % 3) * 9 % 256) as u8 } else { (250 - y * 7) as u8 })).collect()
    }

    #[test]
    fn the_opening_is_opencvs_cross() {
        let want: [u8; W * H] = [0, 37, 74, 74, 85, 85, 85, 3, 37, 52, 52, 74, 35, 85, 48, 48, 52, 52, 52, 65, 65, 48, 48, 48, 17, 52, 65, 65, 65, 65, 48, 37,
                                 45, 45, 30, 95, 95, 41, 43, 43, 45, 45, 95, 95, 95, 95, 43, 43];
        assert_eq!(open3(&ramp(), W, H), want);
        let want: [u8; W * H] = [243, 243, 243, 81, 108, 135, 162, 162, 243, 243, 236, 126, 162, 162, 162, 162, 236, 236, 229, 171, 162, 162, 162, 41, 229,
                                 229, 222, 216, 162, 14, 41, 68, 222, 222, 216, 5, 41, 77, 86, 122, 215, 215, 215, 50, 77, 86, 122, 122];
        assert_eq!(open3(&edge(), W, H), want);
    }

    #[test]
    fn the_blur_reflects_its_borders_101() {
        // every pixel within two of an edge reads reflected ones: the whole of an 8 x 6 picture
        let want: [u8; W * H] = [126, 133, 131, 139, 138, 136, 130, 135, 119, 129, 130, 140, 141, 136, 131, 138, 121, 130, 130, 139, 139, 138, 133, 136, 129,
                                 139, 139, 143, 139, 138, 128, 130, 123, 133, 135, 139, 136, 140, 131, 128, 113, 126, 132, 137, 137, 139, 132, 130];
        assert_eq!(blur5(&ramp(), W, H), want);
        // sides shorter than the kernel (3 x 2): reflected more than once
        assert_eq!(blur5(&[0, 200, 90, 255, 10, 140], 3, 2), [114, 114, 114, 113, 110, 115]);
        assert_eq!(blur5(&[77], 1, 1), [77]);
    }

    #[test]
    fn post_process_thresholds_at_127() {
        let m = Picture { width: W as u32, height: H as u32, channels: 1, data: edge() };
        let blurred: [u8; W * H] = [239, 223, 200, 178, 161, 147, 150, 151, 238, 223, 202, 177, 156, 138, 137, 135, 233, 218, 195, 167, 142, 122, 119, 115, 227,
                                    211, 187, 156, 129, 108, 105, 101, 223, 202, 173, 138, 108, 90, 92, 91, 221, 200, 169, 131, 99, 81, 83, 81];
        assert_eq!(blur5(&open3(&m.data, W, H), W, H), blurred);
        let want: Vec<u8> = blurred.iter().map(|&v| if v < 127 { 0 } else { 255 }).collect();
        let pp = post_process_mask(&m);
        assert_eq!((pp.width, pp.height, pp.channels), (W as u32, H as u32, 1));
        assert_eq!(pp.data, want);
        // 127 itself is kept, 126 is not
        let flat = |v: u8| post_process_mask(&Picture { width: 5, height: 5, channels: 1, data: vec![v; 25] }).data;
        assert_eq!((flat(127)[12], flat(126)[12]), (255, 0));
    }

    #[test]
    fn the_cut_out_is_straight_alpha() {
        let pic = Picture { width: 2, height: 1, channels: 3, data: vec![200, 100, 50, 10, 20, 30] };
        let mask = Picture { width: 2, height: 1, channels: 1, data: vec![128, 0] };
        let rgba = with_alpha(&pic, &mask);
        // the colours as they were (rembg's naive_cutout would have 100, 50, 25 and 0, 0, 0)
        assert_eq!((rgba.channels, rgba.data), (4, vec![200, 100, 50, 128, 10, 20, 30, 0]));
        // a picture with alpha of its own: its alpha replaced by the mask, its colours kept
        let pic = Picture { width: 1, height: 1, channels: 4, data: vec![1, 2, 3, 9] };
        assert_eq!(with_alpha(&pic, &Picture { width: 1, height: 1, channels: 1, data: vec![255] }).data, vec![1, 2, 3, 255]);
    }

    #[test]
    fn the_mask_scales_the_prediction_to_its_range() {
        let side = isnet::SIDE;
        let pred: Vec<f32> = (0..side * side).map(|i| if i % side < side / 2 { 0.25 } else { 0.75 }).collect();
        let m = IsNetEngine::mask_of(&pred, side as u32, side as u32);
        assert_eq!((m.data[0], m.data[side - 1]), (0, 255));
        // a flat prediction has no range: no object
        assert!(IsNetEngine::mask_of(&vec![0.5; side * side], 4, 4).data.iter().all(|&v| v == 0));
    }
}
