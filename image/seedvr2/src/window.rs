//! The DiT's attention windows (the reference's models/dit_v2/window.py, the same for both sizes): the video tokens
//! of a latent clip [t, h, w] (after the 1 x 2 x 2 patches) cut into windows sized as if the picture were 720p -
//! 45 x 80 patches over 3 x 3 windows, the time in 4 - and, on every second block, the same windows shifted by half.
//! Each window's tokens attend to each other and to the whole text.

/// One window: its [start, end) along t, h and w
pub type Window = [(usize, usize); 3];

/// Python's round (halves to even), as the reference sizes the windows
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// The window sizes (wt, wh, ww): the picture scaled to 45 x 80 patches, cut into 3 x 3; the time (at most 30 latent
/// frames counted) into 4
fn sizes(t: usize, h: usize, w: usize) -> (usize, usize, usize) {
    let scale = (45.0 * 80.0 / (h * w) as f64).sqrt();
    let (rh, rw) = (round_even(h as f64 * scale), round_even(w as f64 * scale));
    ((t.min(30) as f64 / 4.0).ceil() as usize, (rh / 3.0).ceil() as usize, (rw / 3.0).ceil() as usize)
}

/// The windows of a latent clip of t x h x w patches, in the reference's order (w outermost, then h, then t); shifted:
/// the "swin" blocks' (half a window along each axis the window does not cover whole)
pub fn windows(t: usize, h: usize, w: usize, shifted: bool) -> Vec<Window> {
    let (wt, wh, ww) = sizes(t, h, w);
    let axis = |n: usize, size: usize| -> Vec<(usize, usize)> {
        if !shifted {
            return (0..n.div_ceil(size)).map(|i| (i * size, ((i + 1) * size).min(n))).filter(|(a, b)| b > a).collect();
        }
        // shift by half a window when the window is shorter than the axis: one more window, the first and last cut
        let s = if size < n { 0.5 } else { 0.0 };
        let count = if s > 0.0 { ((n as f64 - s) / size as f64).ceil() as usize + 1 } else { 1 };
        (0..count)
            .map(|i| {
                let a = (((i as f64 - s) * size as f64) as i64).max(0) as usize;
                let b = (((i as f64 - s + 1.0) * size as f64) as i64).min(n as i64).max(0) as usize;
                (a, b)
            })
            .filter(|(a, b)| b > a)
            .collect()
    };
    let (ts, hs, ws) = (axis(t, wt), axis(h, wh), axis(w, ww));
    let mut out = Vec::with_capacity(ts.len() * hs.len() * ws.len());
    for &x in &ws {
        for &y in &hs {
            for &z in &ts {
                out.push([z, y, x]);
            }
        }
    }
    out
}

/// A window's token count
pub fn len(win: &Window) -> usize {
    win.iter().map(|(a, b)| b - a).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a clip's shape, shifted, and the reference's window count, first and last
    type Case = ((usize, usize, usize), bool, usize, Window, Window);

    // the reference's slices (windows.json from reference/seedvr2/ref.py): count, first, last
    #[test]
    fn windows_are_the_references() {
        let cases: &[Case] = &[
            ((3, 14, 24), false, 3, [(0, 1), (0, 14), (0, 24)], [(2, 3), (0, 14), (0, 24)]),
            ((6, 60, 108), false, 48, [(0, 2), (0, 15), (0, 27)], [(4, 6), (45, 60), (81, 108)]),
            ((6, 60, 108), true, 100, [(0, 1), (0, 7), (0, 13)], [(5, 6), (52, 60), (94, 108)]),
            ((1, 60, 108), true, 25, [(0, 1), (0, 7), (0, 13)], [(0, 1), (52, 60), (94, 108)]),
            ((9, 45, 80), false, 27, [(0, 3), (0, 15), (0, 27)], [(6, 9), (30, 45), (54, 80)]),
            ((9, 45, 80), true, 64, [(0, 1), (0, 7), (0, 13)], [(7, 9), (37, 45), (67, 80)]),
            ((2, 37, 50), false, 18, [(0, 1), (0, 18), (0, 24)], [(1, 2), (36, 37), (48, 50)]),
            ((2, 37, 50), true, 18, [(0, 1), (0, 9), (0, 12)], [(1, 2), (27, 37), (36, 50)]),
        ];
        for &((t, h, w), shifted, n, first, last) in cases {
            let v = windows(t, h, w, shifted);
            assert_eq!((v.len(), v[0], *v.last().unwrap()), (n, first, last), "{t}x{h}x{w} shifted {shifted}");
            // every token in exactly one window
            assert_eq!(v.iter().map(len).sum::<usize>(), t * h * w);
        }
        // the second window of the shifted 6 x 60 x 108: [1, 3) in time (the windows run over t first)
        assert_eq!(windows(6, 60, 108, true)[1], [(1, 3), (0, 7), (0, 13)]);
    }

    #[test]
    fn python_rounds_halves_to_even() {
        assert_eq!((round_even(2.5), round_even(3.5), round_even(-2.5), round_even(2.4)), (2.0, 4.0, -2.0, 2.0));
    }
}
