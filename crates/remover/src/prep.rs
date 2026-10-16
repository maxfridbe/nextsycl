//! The object picture an image-to-3D model wants, as TRELLIS.2's `preprocess_image` makes it (byte for byte: PIL's
//! Lanczos, the crop's rounding, numpy's float32 premultiplication): the long side at most 1024, the background cut
//! out when the picture has no alpha of its own, cropped to the box of alpha > 0.8 around its centre, square,
//! RGB premultiplied by alpha (black behind).

use nextsycl_core::Result;
use nextsycl_diffusion::Picture;

use crate::Remover;

/// Whether a picture carries an alpha of its own (RGBA with a pixel not opaque)
pub fn has_alpha(p: &Picture) -> bool {
    p.channels == 4 && p.data.chunks_exact(4).any(|px| px[3] != 255)
}

/// Python's round(): halves to even
fn round_half_even(v: f64) -> i64 {
    let f = v.floor();
    let d = v - f;
    if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) {
        f as i64 + 1
    } else {
        f as i64
    }
}

/// The square, premultiplied object picture (RGB); `remover` cuts an opaque picture out (without one, an opaque
/// picture is taken whole)
pub fn prepare(pic: &Picture, remover: Option<&Remover>) -> Result<Picture> {
    prepare_margin(pic, remover, 1.0)
}

/// `prepare` with the square `margin` times the object's box (Pixal3D's 1.1)
pub fn prepare_margin(pic: &Picture, remover: Option<&Remover>, margin: f64) -> Result<Picture> {
    let alpha = has_alpha(pic);
    let max = pic.width.max(pic.height) as f64;
    let scale = (1024.0 / max).min(1.0);
    let (w, h) = ((pic.width as f64 * scale) as u32, (pic.height as f64 * scale) as u32);
    let mut p = if scale >= 1.0 {
        pic.clone()
    } else if pic.channels == 4 {
        resize_rgba(pic, w, h)
    } else {
        pic.resize_lanczos(w, h)
    };
    p = if alpha {
        p.rgba()
    } else if let Some(r) = remover {
        r.cut_out(&p)?
    } else {
        p.rgba()
    };
    Ok(crop_square(&p, margin))
}

/// PIL's Lanczos resize of an RGBA picture: through premultiplied RGBa (its Convert.c rounding both ways)
pub fn resize_rgba(p: &Picture, w: u32, h: u32) -> Picture {
    let pre = Picture { data: p.data.chunks_exact(4).flat_map(|px| {
        let a = px[3] as u32;
        let m = |v: u8| {
            let t = v as u32 * a + 128;
            (((t >> 8) + t) >> 8) as u8
        };
        [m(px[0]), m(px[1]), m(px[2]), px[3]]
    }).collect(), ..p.clone() };
    let mut r = pre.resize_lanczos(w, h);
    for px in r.data.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a != 0 && a != 255 {
            for v in &mut px[..3] {
                *v = ((255 * *v as u32) / a).min(255) as u8;
            }
        }
    }
    r
}

/// The alpha > 0.8 box made square around its centre, `margin` times its side (PIL's crop: rounded corners, zeros
/// outside), premultiplied
pub fn crop_square(p: &Picture, margin: f64) -> Picture {
    let (w, h) = (p.width as i64, p.height as i64);
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for y in 0..h {
        for x in 0..w {
            if p.data[((y * w + x) * 4 + 3) as usize] as f64 > 0.8 * 255.0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 {
        // nothing solid: the whole picture
        (x0, y0, x1, y1) = (0, 0, w - 1, h - 1);
    }
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = (((x1 - x0).max(y1 - y0)) as f64 * margin) as i64 / 2;
    let (bx0, by0) = (round_half_even(cx - half as f64), round_half_even(cy - half as f64));
    let (bx1, by1) = (round_half_even(cx + half as f64), round_half_even(cy + half as f64));
    let (ow, oh) = ((bx1 - bx0).max(0), (by1 - by0).max(0));
    let mut out = Picture::new(ow as u32, oh as u32, 3);
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = (bx0 + x, by0 + y);
            if sx < 0 || sy < 0 || sx >= w || sy >= h {
                continue;
            }
            let s = ((sy * w + sx) * 4) as usize;
            let a = p.data[s + 3] as f32 / 255.0;
            for c in 0..3 {
                let v = p.data[s + c] as f32 / 255.0 * a;
                out.data[((y * ow + x) * 3) as usize + c] = (v * 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_pythons() {
        assert_eq!(round_half_even(10.5), 10);
        assert_eq!(round_half_even(11.5), 12);
        assert_eq!(round_half_even(-0.5), 0);
        assert_eq!(round_half_even(2.4), 2);
    }

    #[test]
    fn the_crop_is_square_and_premultiplied() {
        // a 10 x 6 picture, an opaque 4 x 2 block at (3, 2), half-transparent elsewhere
        let mut p = Picture::new(10, 6, 4);
        for (i, px) in p.data.chunks_exact_mut(4).enumerate() {
            let (x, y) = (i % 10, i / 10);
            let solid = (3..7).contains(&x) && (2..4).contains(&y);
            px.copy_from_slice(&[200, 100, 50, if solid { 255 } else { 128 }]);
        }
        let o = crop_square(&p, 1.0);
        // box x 3..6, y 2..3: centre (4.5, 2.5), size 3 -> (3.5, 1.5, 5.5, 3.5) rounded (4, 2, 6, 4)
        assert_eq!((o.width, o.height), (2, 2));
        assert_eq!(&o.data[..3], &[200, 100, 50]);
        assert!(has_alpha(&p));
    }
}
