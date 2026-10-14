//! The page as the vision tower sees it - the processor's `smart_resize` and patches (image_processing_paddleocr_vl.py),
//! with PIL's resampling (`Picture::resize_bicubic`):
//!
//! ```text
//!   sides rounded to multiples of 28 (a 2 x 2 block of 14-pixel patches), the area kept within 112896 .. 1003520
//!   pixels (scaled down / up, aspect kept), resized bicubic
//!   values: x / 255, then (x - 0.5) / 0.5
//!   patches row by row, each [3][14][14] (channel, y, x) - a stride-14 convolution as a linear
//! ```

use nextsycl_image::Picture;

pub const PATCH: u32 = 14;
pub const FACTOR: u32 = 28;
pub const MIN_PIXELS: u32 = 112896;
pub const MAX_PIXELS: u32 = 1003520;

/// What the tower sees of a page
pub struct Seen {
    /// patch rows and columns (even)
    pub gh: usize,
    pub gw: usize,
    /// [gh * gw][3 * 14 * 14]
    pub patches: Vec<f32>,
}

/// Python's round(): halves to even
fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// smart_resize: the (height, width) the page is resized to
pub fn smart_resize(height: u32, width: u32, max_pixels: u32) -> (u32, u32) {
    let f = FACTOR as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f {
        w = round(w * f / h);
        h = f;
    }
    if w < f {
        h = round(h * f / w);
        w = f;
    }
    let mut hb = round(h / f) * f;
    let mut wb = round(w / f) * f;
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = (h / beta / f).floor() * f;
        wb = (w / beta / f).floor() * f;
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    (hb.max(f) as u32, wb.max(f) as u32)
}

pub fn see(page: &Picture, max_pixels: u32) -> Seen {
    let p = page.rgb();
    let (h, w) = smart_resize(p.height, p.width, max_pixels);
    let r = if (w, h) == (p.width, p.height) { p } else { p.resize_bicubic(w, h) };
    let (gh, gw, ps) = ((h / PATCH) as usize, (w / PATCH) as usize, PATCH as usize);
    let k = 3 * ps * ps;
    let mut patches = vec![0f32; gh * gw * k];
    for py in 0..gh {
        for px in 0..gw {
            let o = &mut patches[(py * gw + px) * k..(py * gw + px + 1) * k];
            for y in 0..ps {
                for x in 0..ps {
                    let at = ((py * ps + y) * w as usize + px * ps + x) * 3;
                    for c in 0..3 {
                        o[(c * ps + y) * ps + x] = (r.data[at + c] as f32 / 255.0 - 0.5) / 0.5;
                    }
                }
            }
        }
    }
    Seen { gh, gw, patches }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_as_the_processor_does() {
        // a letter page at 150 dpi (1275 x 1650): over the cap, scaled down - the values the processor computes
        assert_eq!(smart_resize(1650, 1275, MAX_PIXELS), (1120, 868));
        // small: scaled up to the floor
        assert_eq!(smart_resize(100, 200, MAX_PIXELS), (252, 476));
    }
}
