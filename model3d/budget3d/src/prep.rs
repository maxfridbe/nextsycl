//! Idol's framing of a picture (idol/preprocess.py): the short side to 1024 (Lanczos); the object cut out by BiRefNet
//! unless the picture has a real alpha; the alpha eroded by a pixel (a 3 x 3 minimum: no background-coloured fringe);
//! a square of `margin` x the object's bounding box around its centre (1.2 for renders and generated pictures, 1.7
//! for photos); 1024 x 1024 (Lanczos, premultiplied as PIL resizes RGBA); composited on black.

use nextsycl_diffusion::Picture;
use nextsycl_model3d::{Error, Result};
use nextsycl_remover::Remover;

/// The side the encoders see
pub const SIDE: u32 = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The crop margins of Idol's two modes
pub fn margin_of(mode: &str) -> Result<f32> {
    match mode {
        "synthetic" => Ok(1.2),
        "real" => Ok(1.7),
        m => Err(Error(format!("mode {m:?}: synthetic (renders, generated pictures) or real (photographs)"))),
    }
}

/// Has the picture an alpha below 255 anywhere
pub fn has_real_alpha(p: &Picture) -> bool {
    p.channels == 4 && p.data.chunks_exact(4).any(|px| px[3] < 255)
}

/// Bilinear resampling with align_corners=True of a float map [h, w, c] to [oh, ow, c]
fn bilinear_ac(x: &[f32], h: usize, w: usize, c: usize, oh: usize, ow: usize) -> Vec<f32> {
    let mut out = vec![0f32; oh * ow * c];
    let sy = if oh > 1 { (h - 1) as f32 / (oh - 1) as f32 } else { 0.0 };
    let sx = if ow > 1 { (w - 1) as f32 / (ow - 1) as f32 } else { 0.0 };
    for y in 0..oh {
        let fy = y as f32 * sy;
        let y0 = (fy.floor() as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let ly = fy - y0 as f32;
        for xx in 0..ow {
            let fx = xx as f32 * sx;
            let x0 = (fx.floor() as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let lx = fx - x0 as f32;
            for ch in 0..c {
                let v = |yy: usize, xq: usize| x[(yy * w + xq) * c + ch];
                out[(y * ow + xx) * c + ch] = (1.0 - ly) * ((1.0 - lx) * v(y0, x0) + lx * v(y0, x1)) + ly * ((1.0 - lx) * v(y1, x0) + lx * v(y1, x1));
            }
        }
    }
    out
}

/// BiRefNet's alpha as Idol's remove_background makes it: the picture to 1024 x 1024 (bilinear, corners aligned),
/// ImageNet-normalized; the sigmoid back to the picture's size the same way, as bytes (truncated)
pub fn birefnet_alpha(r: &Remover, rgb: &Picture) -> Result<Vec<u8>> {
    let (w, h) = (rgb.width as usize, rgb.height as usize);
    let f: Vec<f32> = rgb.data.iter().map(|v| *v as f32 / 255.0).collect();
    let s = nextsycl_remover::SIDE;
    let mut x = bilinear_ac(&f, h, w, 3, s, s);
    for px in x.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] = (px[c] - MEAN[c]) / STD[c];
        }
    }
    let (logits, _) = r.logits(&x, false)?;
    let sig: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let back = bilinear_ac(&sig, s, s, 1, h, w);
    Ok(back.iter().map(|a| (a.clamp(0.0, 1.0) * 255.0) as u8).collect())
}

/// The picture the encoders see: RGB 1024 x 1024. `remover`: BiRefNet, for pictures without a real alpha (none: the
/// picture is taken whole)
pub fn prepare(pic: &Picture, remover: Option<&Remover>, margin: f32) -> Result<Picture> {
    let (w, h) = (pic.width as f32, pic.height as f32);
    let scale = SIDE as f32 / w.min(h);
    let (nw, nh) = (((w * scale).round() as u32).max(1), ((h * scale).round() as u32).max(1));
    let rgba = pic.rgba();
    let (rgb, mut alpha) = if has_real_alpha(pic) {
        // PIL resizes RGBA premultiplied
        let pm = premultiply(&rgba).resize_lanczos(nw, nh);
        unpremultiply(&pm)
    } else {
        let rgb = pic.rgb().resize_lanczos(nw, nh);
        let a = match remover {
            Some(r) => birefnet_alpha(r, &rgb)?,
            None => vec![255u8; (nw * nh) as usize],
        };
        (rgb, a)
    };
    alpha = erode(&alpha, nw as usize, nh as usize);
    let (iw, ih) = (nw as usize, nh as usize);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..ih {
        for x in 0..iw {
            if alpha[y * iw + x] > 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 == usize::MAX {
        return Err(Error("no object found in the picture (the alpha is empty)".into()));
    }
    let (cx, cy) = ((x0 + x1) as f32 / 2.0, (y0 + y1) as f32 / 2.0);
    let half = (x1 - x0).max(y1 - y0) as f32 / 2.0 * margin;
    // PIL's crop box: int() of each edge (toward zero); outside the picture is transparent
    let (l, t, r, b) = ((cx - half) as i64, (cy - half) as i64, (cx + half) as i64, (cy + half) as i64);
    let (cw, ch) = ((r - l).max(1) as usize, (b - t).max(1) as usize);
    let mut crop = Picture { width: cw as u32, height: ch as u32, channels: 4, data: vec![0u8; cw * ch * 4] };
    for y in 0..ch {
        let sy = t + y as i64;
        if sy < 0 || sy >= ih as i64 {
            continue;
        }
        for x in 0..cw {
            let sx = l + x as i64;
            if sx < 0 || sx >= iw as i64 {
                continue;
            }
            let (s, d) = ((sy as usize * iw + sx as usize), (y * cw + x));
            crop.data[d * 4..d * 4 + 3].copy_from_slice(&rgb.data[s * 3..s * 3 + 3]);
            crop.data[d * 4 + 3] = alpha[s];
        }
    }
    // the square to 1024 (premultiplied, as PIL resizes RGBA), then on black: the premultiplied colour
    let pm = premultiply(&crop).resize_lanczos(SIDE, SIDE);
    Ok(Picture { width: SIDE, height: SIDE, channels: 3, data: pm.data.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect() })
}

/// RGBA with its colour multiplied by alpha (rounded, as PIL's RGBa)
fn premultiply(p: &Picture) -> Picture {
    Picture { width: p.width, height: p.height, channels: 4, data: p.data.chunks_exact(4).flat_map(|px| {
        let a = px[3] as u32;
        let m = |c: u8| ((c as u32 * a + 127) / 255) as u8;
        [m(px[0]), m(px[1]), m(px[2]), px[3]]
    }).collect() }
}

/// Premultiplied RGBA back to (RGB, alpha)
fn unpremultiply(p: &Picture) -> (Picture, Vec<u8>) {
    let mut rgb = Vec::with_capacity(p.data.len() / 4 * 3);
    let mut a = Vec::with_capacity(p.data.len() / 4);
    for px in p.data.chunks_exact(4) {
        let al = px[3] as u32;
        let u = |c: u8| (c as u32 * 255 + al / 2).checked_div(al).map_or(0, |v| v.min(255) as u8);
        rgb.extend([u(px[0]), u(px[1]), u(px[2])]);
        a.push(px[3]);
    }
    (Picture { width: p.width, height: p.height, channels: 3, data: rgb }, a)
}

/// A 3 x 3 minimum (PIL's MinFilter(3): the edges repeated)
fn erode(a: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut o = vec![0u8; a.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = 255u8;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    m = m.min(a[yy * w + xx]);
                }
            }
            o[y * w + x] = m;
        }
    }
    o
}

/// The picture as the encoders' input: [3, 1024, 1024] planes in [0, 1]
pub fn planes(p: &Picture) -> Vec<f32> {
    let n = (p.width * p.height) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in p.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * n + i] = px[c] as f32 / 255.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_object_is_squared_and_centred_on_black() {
        // a 200 x 100 picture, a red 20 x 20 square at (150, 40) on a transparent background
        let mut p = Picture { width: 200, height: 100, channels: 4, data: vec![0u8; 200 * 100 * 4] };
        for y in 40..60 {
            for x in 150..170 {
                p.data[(y * 200 + x) * 4..(y * 200 + x) * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        let o = prepare(&p, None, 1.2).unwrap();
        assert_eq!((o.width, o.height, o.channels), (1024, 1024, 3));
        let px = |x: usize, y: usize| &o.data[(y * 1024 + x) * 3..(y * 1024 + x) * 3 + 3];
        assert!(px(512, 512)[0] > 200 && px(512, 512)[1] < 30, "the object fills the middle: {:?}", px(512, 512));
        assert_eq!(px(5, 5), &[0, 0, 0], "the margin is black");
        assert!(prepare(&Picture { width: 4, height: 4, channels: 4, data: vec![0; 64] }, None, 1.2).is_err());
    }

    #[test]
    fn erosion_takes_the_neighbourhood_minimum() {
        let a = vec![255, 255, 255, 255, 0, 255, 255, 255, 255];
        assert_eq!(erode(&a, 3, 3), vec![0; 9]);
        let b = vec![9u8; 16];
        assert_eq!(erode(&b, 4, 4), b);
    }

    #[test]
    fn corners_aligned_bilinear_keeps_the_ends() {
        let x = vec![0.0, 1.0];
        let o = bilinear_ac(&x, 1, 2, 1, 1, 5);
        assert_eq!(o, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
