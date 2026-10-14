//! Columns found before the model reads a page: the model reads a page as one whole, so blocks set side by side
//! (two school years of a transcript, a two-column article) come back as one wide table with their rows interleaved.
//! A gutter - a strip with no ink, at least two characters wide, running down a long stretch of the page - splits
//! the page into what is above it, its left and right sides, and what is below; what is above and below is split
//! again the same way, the sides are not (inside a block, the gaps between a table's columns look like gutters too),
//! and each region is read on its own, in reading order.
//!
//! ```text
//!   ink: grey < 160;  lines: runs of rows with ink;  in each line the x ranges free of ink for >= GAP pixels
//!   a gutter at x: the longest run of consecutive lines (blank ones count) free at x, by the height it spans;
//!     the best x of the page's middle 70% wins (the tallest; ties: the nearest the middle)
//!   kept when it spans >= 25% of the page's text height, with ink on both sides in >= 4 of its lines
//! ```

use nextsycl_image::Picture;

/// A region of the page to read, in pixels: x, y, width, height
pub type Region = (u32, u32, u32, u32);

const INK: u8 = 160;

/// The page's grey levels and size
struct Grey {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn grey(p: &Picture) -> Grey {
    let c = p.channels as usize;
    let px = p.data.chunks_exact(c).map(|v| if c >= 3 { ((v[0] as u32 * 30 + v[1] as u32 * 59 + v[2] as u32 * 11) / 100) as u8 } else { v[0] }).collect();
    Grey { w: p.width as usize, h: p.height as usize, px }
}

/// Each text line of a region (rows y0..y1) and, per x, whether that x sits in a gap of >= `gap` pixels free of ink
struct Line {
    y0: usize,
    y1: usize,
    free: Vec<bool>,
}

fn lines(g: &Grey, (x0, y0, w, h): (usize, usize, usize, usize), gap: usize) -> Vec<Line> {
    let row_ink = |y: usize| (x0..x0 + w).any(|x| g.px[y * g.w + x] < INK);
    let mut out = Vec::new();
    let mut y = y0;
    while y < y0 + h {
        if !row_ink(y) {
            y += 1;
            continue;
        }
        let s = y;
        while y < y0 + h && row_ink(y) {
            y += 1;
        }
        // a column of the line is free when none of its rows has ink there; free runs shorter than the gap are not
        let col: Vec<bool> = (x0..x0 + w).map(|x| (s..y).all(|yy| g.px[yy * g.w + x] >= INK)).collect();
        let mut free = vec![false; w];
        let mut i = 0;
        while i < w {
            if !col[i] {
                i += 1;
                continue;
            }
            let a = i;
            while i < w && col[i] {
                i += 1;
            }
            if i - a >= gap {
                free[a..i].fill(true);
            }
        }
        out.push(Line { y0: s, y1: y, free });
    }
    out
}

/// The best gutter of a region: (x in the region, first line, last line) - lines as indices into `ls`
fn gutter(ls: &[Line], w: usize, min_span: usize) -> Option<(usize, usize, usize)> {
    let mut best: Option<(usize, usize, usize, usize)> = None; // (span, x, a, b)
    let mid = w / 2;
    for x in w * 15 / 100..w * 85 / 100 {
        let mut a = 0;
        while a < ls.len() {
            if !ls[a].free[x] {
                a += 1;
                continue;
            }
            let mut b = a;
            while b + 1 < ls.len() && ls[b + 1].free[x] {
                b += 1;
            }
            let span = ls[b].y1 - ls[a].y0;
            // ink on both sides of x in enough of its lines
            let both = ls[a..=b].iter().filter(|l| l.free[..x].iter().any(|f| !f) && l.free[x..].iter().any(|f| !f)).count();
            if span >= min_span && both >= 4 {
                let better = match best {
                    None => true,
                    Some((s, bx, _, _)) => span > s || (span == s && x.abs_diff(mid) < bx.abs_diff(mid)),
                };
                if better {
                    best = Some((span, x, a, b));
                }
            }
            a = b + 1;
        }
    }
    let (_, x, a, b) = best?;
    // the middle of the free strip around x, common to the run's lines
    let (mut l, mut r) = (x, x);
    while l > 0 && ls[a..=b].iter().all(|ln| ln.free[l - 1]) {
        l -= 1;
    }
    while r + 1 < w && ls[a..=b].iter().all(|ln| ln.free[r + 1]) {
        r += 1;
    }
    Some(((l + r) / 2, a, b))
}

/// The regions of a page in reading order: the whole page when no gutter is found
pub fn regions(p: &Picture) -> Vec<Region> {
    let g = grey(p);
    let mut out = Vec::new();
    split(&g, (0, 0, g.w, g.h), 0, &mut out);
    out.into_iter().map(|(x, y, w, h)| (x as u32, y as u32, w as u32, h as u32)).collect()
}

fn split(g: &Grey, r: (usize, usize, usize, usize), depth: usize, out: &mut Vec<(usize, usize, usize, usize)>) {
    let (x0, y0, w, h) = r;
    // two characters at the page's scale: 1.2% of the page's width, at least 8 pixels
    let gap = (g.w * 12 / 1000).max(8);
    let ls = lines(g, r, gap);
    if ls.is_empty() {
        return;
    }
    let text_h = ls.last().map(|l| l.y1).unwrap_or(y0) - ls[0].y0;
    let found = if depth < 2 && w > 200 { gutter(&ls, w, (text_h / 4).max(1)) } else { None };
    let Some((x, a, b)) = found else {
        out.push(r);
        return;
    };
    let top = ls[a].y0;
    let bottom = ls[b].y1;
    // above the gutter, its two sides, below it - each split again
    if a > 0 {
        split(g, (x0, y0, w, top - y0), depth + 1, out);
    }
    // the sides whole: a block's own table columns are not split apart
    out.push((x0, top, x, bottom - top));
    out.push((x0 + x, top, w - x, bottom - top));
    if b + 1 < ls.len() {
        split(g, (x0, bottom, w, y0 + h - bottom), depth + 1, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A white page with dark boxes (text lines) at the given (x, y, w, h)
    fn page(w: u32, h: u32, boxes: &[(u32, u32, u32, u32)]) -> Picture {
        let mut p = Picture::new(w, h, 3);
        p.data.fill(255);
        for &(bx, by, bw, bh) in boxes {
            for y in by..by + bh {
                for x in bx..bx + bw {
                    let i = ((y * w + x) * 3) as usize;
                    p.data[i..i + 3].fill(0);
                }
            }
        }
        p
    }

    #[test]
    fn one_column_stays_whole() {
        let lines: Vec<_> = (0..20).map(|i| (50, 50 + i * 30, 900, 12)).collect();
        assert_eq!(regions(&page(1000, 800, &lines)).len(), 1);
    }

    #[test]
    fn two_columns_under_a_heading() {
        // a heading across, then 20 lines in two columns with a 40-pixel gutter at 480..520
        let mut b = vec![(50, 20, 900, 14)];
        for i in 0..20 {
            b.push((50, 60 + i * 30, 430, 12));
            b.push((520, 60 + i * 30, 430, 12));
        }
        let r = regions(&page(1000, 700, &b));
        assert_eq!(r.len(), 3, "{r:?}");
        // the heading first, then the left column, then the right
        assert!(r[0].1 == 0 && r[0].2 == 1000);
        assert!(r[1].0 == 0 && r[1].2 > 470 && r[1].2 < 530);
        assert!(r[2].0 > 470);
    }
}
