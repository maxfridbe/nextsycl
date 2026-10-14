//! The pictures as the model sees them - the reference's `infer` / `infer_multi` (modeling_unlimitedocr.py) step by
//! step, with PIL's resampling (`Picture::resize_bicubic`):
//!
//! ```text
//!   one picture, tiles on ("gundam"): the global view - ImageOps.pad to 1024 x 1024 (aspect kept, grey 127 around)
//!     - and, when a side is over 640, tiles of 640: the grid of 2 .. 32 tiles closest to the picture's aspect
//!     (dynamic_preprocess), the picture resized to it and cut, row by row
//!   several pictures (pages), or tiles off: each page's global view only
//!   values: x / 255, then (x - 0.5) / 0.5, channel planes [3][H][W]
//! ```
//!
//! The image tokens the decoder gets: the tiles first (a row of tiles' patch rows side by side, a newline after each),
//! then each page's global view (16 rows of 16 and a newline), then the view separator.

use nextsycl_image::Picture;

pub const BASE: u32 = 1024;
pub const TILE: u32 = 640;
pub const MIN_TILES: u32 = 2;
pub const MAX_TILES: u32 = 32;
/// ImageOps.pad's colour: tuple(int(x * 255) for x in mean) with mean 0.5
pub const PAD: u8 = 127;

/// A view's values for the vision tower: [3][side][side]
pub struct View {
    pub side: u32,
    pub planes: Vec<f32>,
}

/// What the tower sees of a request
pub struct Seen {
    /// the tiles, row by row (none: the global views alone)
    pub tiles: Vec<View>,
    /// the tile grid (columns, rows)
    pub grid: (u32, u32),
    /// one global view a page
    pub globals: Vec<View>,
}

fn rgb(p: &Picture) -> Picture {
    match p.channels {
        3 => p.clone(),
        // PIL's convert("RGB") drops alpha (no compositing)
        4 => Picture { width: p.width, height: p.height, channels: 3, data: p.data.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect() },
        1 => Picture { width: p.width, height: p.height, channels: 3, data: p.data.iter().flat_map(|v| [*v, *v, *v]).collect() },
        _ => Picture { width: p.width, height: p.height, channels: 3, data: p.data.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0]]).collect() },
    }
}

/// Python's round(): halves to even
fn round(x: f64) -> u32 {
    x.round_ties_even() as u32
}

/// ImageOps.pad(image, (side, side), color=127): resized to fit (contain), centred on grey
pub fn pad(p: &Picture, side: u32) -> Picture {
    let (w, h) = (p.width as f64, p.height as f64);
    let (nw, nh) = if (w / h - 1.0).abs() < f64::EPSILON {
        (side, side)
    } else if w > h {
        (side, round(h / w * side as f64))
    } else {
        (round(w / h * side as f64), side)
    };
    let r = if (nw, nh) == (p.width, p.height) { p.clone() } else { p.resize_bicubic(nw, nh) };
    let mut out = Picture { width: side, height: side, channels: 3, data: vec![PAD; (side * side * 3) as usize] };
    let (x0, y0) = (round((side - nw) as f64 * 0.5), round((side - nh) as f64 * 0.5));
    for y in 0..nh as usize {
        let src = &r.data[y * nw as usize * 3..(y + 1) * nw as usize * 3];
        let at = ((y0 as usize + y) * side as usize + x0 as usize) * 3;
        out.data[at..at + src.len()].copy_from_slice(src);
    }
    out
}

/// dynamic_preprocess's grid: the (columns, rows) of MIN..=MAX tiles closest to the picture's aspect (a tie: the
/// later one when the picture's area is over half the grid's)
pub fn grid(w: u32, h: u32) -> (u32, u32) {
    let mut ratios: Vec<(u32, u32)> = Vec::new();
    for n in MIN_TILES..=MAX_TILES {
        for i in 1..=n {
            for j in 1..=n {
                if i * j >= MIN_TILES && i * j <= MAX_TILES && !ratios.contains(&(i, j)) {
                    ratios.push((i, j));
                }
            }
        }
    }
    // sorted by tile count (equal counts never tie on aspect: the same count and ratio is the same grid)
    ratios.sort_by_key(|r| r.0 * r.1);
    let aspect = w as f64 / h as f64;
    let area = w as f64 * h as f64;
    let mut best = (1, 1);
    let mut best_diff = f64::INFINITY;
    for &(i, j) in &ratios {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * (TILE * TILE * i * j) as f64 {
            best = (i, j);
        }
    }
    best
}

/// A picture's values as the tower takes them: (x / 255 - 0.5) / 0.5, planes [3][H][W]
pub fn planes(p: &Picture) -> Vec<f32> {
    let n = (p.width * p.height) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in p.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    out
}

/// The views of a request's pages: one page with `tiles` - its global view and (a side over 640) its tiles; else
/// each page's global view
pub fn see(pages: &[Picture], tiles: bool) -> Seen {
    let pages: Vec<Picture> = pages.iter().map(rgb).collect();
    let globals = pages.iter().map(|p| View { side: BASE, planes: planes(&pad(p, BASE)) }).collect();
    let mut seen = Seen { tiles: Vec::new(), grid: (1, 1), globals };
    if let [p] = pages.as_slice() {
        if tiles && (p.width > TILE || p.height > TILE) {
            let (gw, gh) = grid(p.width, p.height);
            let r = p.resize_bicubic(TILE * gw, TILE * gh);
            for i in 0..gw * gh {
                let (tx, ty) = ((i % gw) * TILE, (i / gw) * TILE);
                let mut t = Picture::new(TILE, TILE, 3);
                for y in 0..TILE as usize {
                    let src = ((ty as usize + y) * r.width as usize + tx as usize) * 3;
                    t.data[y * TILE as usize * 3..(y + 1) * TILE as usize * 3].copy_from_slice(&r.data[src..src + TILE as usize * 3]);
                }
                seen.tiles.push(View { side: TILE, planes: planes(&t) });
            }
            seen.grid = (gw, gh);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_as_the_reference_picks_them() {
        // an A4 page at 150 dpi: 2 x 3; a wide strip; a letter page at 150 dpi (the values the reference computes)
        assert_eq!(grid(1240, 1754), (2, 3));
        assert_eq!(grid(3000, 700), (9, 2));
        assert_eq!(grid(1275, 1650), (3, 4));
    }

    #[test]
    fn pad_centres_on_grey() {
        let mut p = Picture::new(4, 2, 3);
        p.data.fill(255);
        let q = pad(&p, 8);
        // 8 x 4 picture in the middle rows 2..6
        assert_eq!(q.data[0], PAD);
        assert_eq!(q.data[(2 * 8) * 3], 255);
        assert_eq!(q.data[(6 * 8) * 3], PAD);
    }
}
