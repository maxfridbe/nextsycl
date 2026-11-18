//! Depth Anything V2 Small as an image engine (`video.control`'s depth, `image depth`): a picture in, its relative
//! depth out - the network (`net.rs`, DINOv2-small + the DPT head) and its picture handling:
//!
//! - **in**: DPTImageProcessor's size rule (518 x 518, keep the aspect ratio, sides multiples of 14: the scale nearest
//!   to 1 on either side is applied to both and each side rounded to a multiple of 14), Pillow's bicubic (antialiased,
//!   as `Picture::resize_bicubic`), ImageNet mean and std;
//! - **out**: the head's ReLU output (disparity: larger is nearer) resized to the asked size, bilinear with half-pixel
//!   centres as torch's `interpolate`; `depth8` scales a set of maps by their common [min, max] to 8 bits, truncated
//!   (ComfyUI's depth preprocessors scale a picture by its own range; a clip uses one range for all its frames, so the
//!   depth does not breathe from frame to frame).
//!
//! License: the Small weights are Apache-2.0; Base, Large and Giant are CC-BY-NC-4.0 and are not served.

pub mod check;
pub mod ffi;
pub mod net;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::Gpu;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step};

pub const ARCH: &str = "depth-anything-v2";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "depth estimation: Depth Anything V2 Small (Rust + SYCL)", roles: &["model"], load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().cloned().ok_or_else(|| Error("a depth estimator needs a GPU".into()))?;
    let path = files.get("model").or_else(|| files.values().next()).cloned().ok_or_else(|| Error("depth-anything-v2: no model file (the card names it)".into()))?;
    let t0 = Instant::now();
    let net = net::DepthNet::load(&path, &gpu)?;
    let secs = t0.elapsed().as_secs_f64();
    log(format!("depthanything: loaded in {secs:.1} s on {} ({:.0} MB)", gpu.name, net.bytes as f64 / 1e6));
    Ok(Box::new(Depth { net: Mutex::new(net), load_secs: secs }))
}

/// DPTImageProcessor's output size for a picture of w x h: (width, height)
pub fn input_size(w: u32, h: u32) -> (usize, usize) {
    const TARGET: f64 = 518.0;
    const MULTIPLE: f64 = net::PATCH as f64;
    let (sh, sw) = (TARGET / h as f64, TARGET / w as f64);
    let s = if (1.0 - sw).abs() < (1.0 - sh).abs() { (sw, sw) } else { (sh, sh) };
    // Python's round() is half to even
    let round_even = |v: f64| -> f64 {
        let r = v.round();
        if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
    };
    let side = |scale: f64, n: u32| ((round_even(scale * n as f64 / MULTIPLE) * MULTIPLE) as usize).max(net::PATCH);
    (side(s.1, w), side(s.0, h))
}

/// The network's input for a picture: [h, w, 3] float32 and (w, h)
pub fn prepare(pic: &Picture) -> (Vec<f32>, usize, usize) {
    let rgb = pic.rgb();
    let (w, h) = input_size(rgb.width, rgb.height);
    let r = rgb.resize_bicubic(w as u32, h as u32);
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let x = r.data.chunks_exact(3).flat_map(|p| (0..3).map(move |c| ((p[c] as f32 / 255.0) - MEAN[c]) / STD[c])).collect();
    (x, w, h)
}

pub use nextsycl_image::depth8;

/// A grey map as the RGB picture a control video holds
pub fn grey_rgb(w: u32, h: u32, grey: &[u8]) -> Picture {
    Picture { width: w, height: h, channels: 3, data: grey.iter().flat_map(|&v| [v, v, v]).collect() }
}

pub struct Depth {
    net: Mutex<net::DepthNet>,
    load_secs: f64,
}

impl ImageEngine for Depth {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 0, height: 0, steps: 1, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Simple, shift: 1.0 }
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(&self, _req: &ImageRequest, _p: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error(format!("{ARCH} estimates depth (video.control, image depth); it does not draw")))
    }
    fn estimates_depth(&self) -> bool {
        true
    }
    fn depth(&self, pic: &Picture) -> Result<Vec<f32>> {
        if pic.width == 0 || pic.height == 0 {
            return Err(Error("depth: an empty picture".into()));
        }
        let (x, w, h) = prepare(pic);
        self.net.lock().unwrap_or_else(|p| p.into_inner()).depth_at(&x, h, w, pic.width as usize, pic.height as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_size_is_dpts_rule() {
        // reference/depthanything/ref.py (transformers 5.22's DPTImageProcessor): these pictures' sizes
        assert_eq!(input_size(640, 360), (518, 294));
        assert_eq!(input_size(518, 518), (518, 518));
        assert_eq!(input_size(300, 500), (308, 518));
        // the scale nearest to 1 wins: 1920 x 1080 is scaled by 518 / 1080 (0.48) or 518 / 1920 (0.27) - the former
        assert_eq!(input_size(1920, 1080), (924, 518));
        // a tiny picture is scaled up, not left small
        assert_eq!(input_size(64, 64), (518, 518));
        // a very thin one keeps at least one patch
        assert_eq!(input_size(5000, 4).1, 14);
    }

    #[test]
    fn prepare_normalizes_like_imagenet() {
        let p = Picture { width: 518, height: 518, channels: 3, data: vec![255; 518 * 518 * 3] };
        let (x, w, h) = prepare(&p);
        assert_eq!((w, h, x.len()), (518, 518, 518 * 518 * 3));
        assert!((x[0] - (1.0 - 0.485) / 0.229).abs() < 1e-6 && (x[2] - (1.0 - 0.406) / 0.225).abs() < 1e-6);
    }

    #[test]
    fn a_grey_map_is_three_equal_channels() {
        let g = grey_rgb(2, 1, &[7, 200]);
        assert_eq!(g.data, vec![7, 7, 7, 200, 200, 200]);
    }
}
