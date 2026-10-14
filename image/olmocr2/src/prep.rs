//! The page as olmOCR's tower sees it - olmOCR's rendering (the longest side 1288 pixels), then Qwen2.5-VL's processor
//! and its window index (transformers' Qwen2_5_VL image processor and `get_window_index`), with PIL's resampling:
//!
//! ```text
//!   the page resized so its longest side is 1288 (bicubic)
//!   smart_resize: sides rounded to multiples of 28, the area within 3136 .. 12845056 pixels
//!   values: (x / 255 - mean) / std, CLIP's
//!   patches of 14 x 14, a 2 x 2 block of them (a unit: one image token after the merger) together, (c, y, x) each
//!   units in windows of 4 x 4 (112 pixels) - window by window, row by row inside - for the blocks that attend
//!   within their window; the merger's rows go back to row order after the tower
//! ```

use nextsycl_image::Picture;

pub const LONGEST: u32 = 1288;
pub const PATCH: usize = 14;
pub const FACTOR: u32 = 28;
pub const MIN_PIXELS: u32 = 3136;
pub const MAX_PIXELS: u32 = 12845056;
/// a window's side in units (112 pixels / 28)
pub const WINDOW: usize = 4;

/// What the tower sees of a page
pub struct Seen {
    /// patch rows and columns (even)
    pub gh: usize,
    pub gw: usize,
    /// [n][3 * 14 * 14] in window order
    pub patches: Vec<f32>,
    /// each patch's (row, column) in the patch grid, window order
    pub pos: Vec<(u32, u32)>,
    /// each patch's window: the patches [lo, hi)
    pub lo: Vec<i32>,
    pub hi: Vec<i32>,
    /// for each unit in row order, its row in window order
    pub back: Vec<i32>,
}

fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// smart_resize: the (height, width) the picture is resized to
pub fn smart_resize(height: u32, width: u32) -> (u32, u32) {
    let f = FACTOR as f64;
    let (h, w) = (height as f64, width as f64);
    let mut hb = (round(h / f) * f).max(f);
    let mut wb = (round(w / f) * f).max(f);
    if hb * wb > MAX_PIXELS as f64 {
        let beta = (h * w / MAX_PIXELS as f64).sqrt();
        hb = ((h / beta / f).floor() * f).max(f);
        wb = ((w / beta / f).floor() * f).max(f);
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    (hb as u32, wb as u32)
}

/// The page with its longest side `LONGEST` (olmOCR renders its pages so)
pub fn fit(page: &Picture) -> Picture {
    let p = page.rgb();
    let s = LONGEST as f64 / p.width.max(p.height) as f64;
    let (w, h) = ((p.width as f64 * s).round().max(1.0) as u32, (p.height as f64 * s).round().max(1.0) as u32);
    if (w, h) == (p.width, p.height) { p } else { p.resize_bicubic(w, h) }
}

/// A fitted page's values as the tower takes them: resized (smart_resize), (x / 255 - mean) / std, planes [3][H][W]
pub fn planes(page: &Picture, mean: [f32; 3], std: [f32; 3]) -> (Vec<f32>, usize, usize) {
    let p = page.rgb();
    let (h, w) = smart_resize(p.height, p.width);
    let r = if (w, h) == (p.width, p.height) { p } else { p.resize_bicubic(w, h) };
    let n = (w * h) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in r.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    (out, h as usize, w as usize)
}

/// A fitted page's patches, window order; `mean` / `std` the tower's
pub fn see(page: &Picture, mean: [f32; 3], std: [f32; 3]) -> Seen {
    let (pl, h, w) = planes(page, mean, std);
    see_planes(&pl, h, w)
}

/// Planes [3][h][w] (sides multiples of 28) as patches, window order
pub fn see_planes(pl: &[f32], h: usize, w: usize) -> Seen {
    let (gh, gw) = (h / PATCH, w / PATCH);
    let (uh, uw) = (gh / 2, gw / 2);
    // the units, window by window
    let mut order = Vec::with_capacity(uh * uw);
    let mut windows = Vec::new();
    for wy in 0..uh.div_ceil(WINDOW) {
        for wx in 0..uw.div_ceil(WINDOW) {
            let start = order.len();
            for y in wy * WINDOW..((wy + 1) * WINDOW).min(uh) {
                for x in wx * WINDOW..((wx + 1) * WINDOW).min(uw) {
                    order.push((y, x));
                }
            }
            windows.push((start, order.len()));
        }
    }
    let k = 3 * PATCH * PATCH;
    let n = gh * gw;
    let mut patches = vec![0f32; n * k];
    let mut pos = Vec::with_capacity(n);
    let mut lo = vec![0i32; n];
    let mut hi = vec![0i32; n];
    let mut back = vec![0i32; uh * uw];
    for &(a, b) in &windows {
        for (u, &(uy, ux)) in order.iter().enumerate().take(b).skip(a) {
            back[uy * uw + ux] = u as i32;
            for s in 0..4 {
                let (py, px) = (2 * uy + s / 2, 2 * ux + s % 2);
                let i = u * 4 + s;
                pos.push((py as u32, px as u32));
                lo[i] = (a * 4) as i32;
                hi[i] = (b * 4) as i32;
                let o = &mut patches[i * k..(i + 1) * k];
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let at = (py * PATCH + y) * w + px * PATCH + x;
                        for c in 0..3 {
                            o[(c * PATCH + y) * PATCH + x] = pl[c * h * w + at];
                        }
                    }
                }
            }
        }
    }
    Seen { gh, gw, patches, pos, lo, hi, back }
}

/// PIL's transpose(ROTATE_90 / 180 / 270): counter-clockwise by `deg`
pub fn rotate(p: &Picture, deg: u32) -> Picture {
    let c = p.channels as usize;
    let (w, h) = (p.width as usize, p.height as usize);
    let (nw, nh) = if deg % 180 == 90 { (h, w) } else { (w, h) };
    let mut out = Picture::new(nw as u32, nh as u32, p.channels);
    for y in 0..nh {
        for x in 0..nw {
            // the source of output (x, y)
            let (sx, sy) = match deg % 360 {
                90 => (w - 1 - y, x),
                180 => (w - 1 - x, h - 1 - y),
                270 => (y, h - 1 - x),
                _ => (x, y),
            };
            let (d, s) = ((y * nw + x) * c, (sy * w + sx) * c);
            out.data[d..d + c].copy_from_slice(&p.data[s..s + c]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_as_the_processor_does() {
        // a letter page fitted to 1288: 911 x 1288 -> 924 x 1288 (llama.cpp's reference saw the same)
        assert_eq!(smart_resize(1288, 911), (1288, 924));
    }

    #[test]
    fn windows_of_four_units() {
        // 10 x 12 patches: 5 x 6 units, windows 4 x 4 / 4 x 2 / 1 x 4 / 1 x 2
        let p = Picture::new(12 * 14, 10 * 14, 3);
        let s = see(&p, [0.0; 3], [1.0; 3]);
        assert_eq!(s.pos.len(), 120);
        assert_eq!((s.lo[0], s.hi[0]), (0, 64));
        assert_eq!((s.lo[64], s.hi[64]), (64, 96));
        // unit (0, 4) is the first of the second window
        assert_eq!(s.back[4], 16);
        assert_eq!(s.pos[64], (0, 8));
    }

    #[test]
    fn rotates_counter_clockwise() {
        // a 2 x 1 picture [a b] turned 90 counter-clockwise: b over a
        let p = Picture { width: 2, height: 1, channels: 1, data: vec![1, 2] };
        let r = rotate(&p, 90);
        assert_eq!((r.width, r.height, r.data.clone()), (1, 2, vec![2, 1]));
    }
}
