//! The picture as sparc3d's prepare_image and ImageConditioner take it: the square around the alpha's bounding box
//! (PIL's getbbox: alpha > 0; transparent where it runs past the picture), composited over white (PIL's paste with
//! the alpha as mask: 8-bit, rounded), padded to 1.2 times its side on white, centred; then DINOv2's input (torch's
//! bilinear resize to 756, align_corners false, no antialias, of the 8-bit values / 255) and CLIP's (PIL's Lanczos to
//! 224, 8-bit), each normalized by its mean and deviation. A picture without alpha is taken whole.

use nextsycl_diffusion::Picture;

pub const DINO_SIZE: usize = 756;
pub const CLIP_SIZE: usize = 224;
const DINO_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DINO_STD: [f32; 3] = [0.229, 0.224, 0.225];
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// PIL's DIV255 (rounded division by 255 of an 8-bit product sum)
fn div255(a: u32) -> u8 {
    let t = a + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// The framed square, RGB on white
pub fn frame(pic: &Picture) -> Picture {
    let rgba = pic.rgba();
    let (w, h) = (rgba.width as i64, rgba.height as i64);
    let alpha = |x: i64, y: i64| rgba.data[((y * w + x) * 4 + 3) as usize];
    // the alpha's bounding box (exclusive right / lower), else the whole picture
    let (mut l, mut u, mut r, mut d) = (w, h, -1i64, -1i64);
    for y in 0..h {
        for x in 0..w {
            if alpha(x, y) != 0 {
                l = l.min(x);
                u = u.min(y);
                r = r.max(x + 1);
                d = d.max(y + 1);
            }
        }
    }
    let (l, u, bw, bh) = if r < 0 { (0, 0, w, h) } else { (l, u, r - l, d - u) };
    let side = bw.max(bh);
    let (l, u) = (l - (side - bw) / 2, u - (side - bh) / 2);
    // the crop over white (outside the picture: transparent, so white)
    let mut sq = vec![255u8; (side * side * 3) as usize];
    for y in 0..side {
        for x in 0..side {
            let (sx, sy) = (l + x, u + y);
            if sx < 0 || sy < 0 || sx >= w || sy >= h {
                continue;
            }
            let i = ((sy * w + sx) * 4) as usize;
            let a = rgba.data[i + 3] as u32;
            for c in 0..3 {
                let o = ((y * side + x) * 3 + c) as usize;
                sq[o] = div255(sq[o] as u32 * (255 - a) + rgba.data[i + c as usize] as u32 * a);
            }
        }
    }
    let big = (side as f64 * 1.2) as i64;
    let off = ((big - side) / 2, (big - side) / 2);
    let mut out = Picture { width: big as u32, height: big as u32, channels: 3, data: vec![255u8; (big * big * 3) as usize] };
    for y in 0..side {
        let (ty, sy) = (y + off.1, y);
        if ty < 0 || ty >= big {
            continue;
        }
        for x in 0..side {
            let tx = x + off.0;
            if tx < 0 || tx >= big {
                continue;
            }
            for c in 0..3 {
                out.data[((ty * big + tx) * 3 + c) as usize] = sq[((sy * side + x) * 3 + c) as usize];
            }
        }
    }
    out
}

/// torch's F.interpolate(bilinear, align_corners=False) of [3, h, w] values to [3, m, m]
fn bilinear(src: &[f32], h: usize, w: usize, m: usize) -> Vec<f32> {
    let axis = |n: usize| -> Vec<(usize, usize, f32)> {
        let scale = n as f32 / m as f32;
        (0..m).map(|o| {
            let s = (scale * (o as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (s as usize).min(n - 1);
            let i1 = if i0 < n - 1 { i0 + 1 } else { i0 };
            (i0, i1, s - i0 as f32)
        }).collect()
    };
    let (ys, xs) = (axis(h), axis(w));
    let mut out = vec![0f32; 3 * m * m];
    for c in 0..3 {
        let p = &src[c * h * w..(c + 1) * h * w];
        for (oy, &(y0, y1, ly)) in ys.iter().enumerate() {
            for (ox, &(x0, x1, lx)) in xs.iter().enumerate() {
                let top = (1.0 - lx) * p[y0 * w + x0] + lx * p[y0 * w + x1];
                let bot = (1.0 - lx) * p[y1 * w + x0] + lx * p[y1 * w + x1];
                out[c * m * m + oy * m + ox] = (1.0 - ly) * top + ly * bot;
            }
        }
    }
    out
}

fn planes(p: &Picture) -> Vec<f32> {
    let (w, h) = (p.width as usize, p.height as usize);
    let mut v = vec![0f32; 3 * w * h];
    for i in 0..w * h {
        for c in 0..3 {
            v[c * w * h + i] = p.data[i * p.channels as usize + c] as f32 / 255.0;
        }
    }
    v
}

fn normalize(v: &mut [f32], n: usize, mean: [f32; 3], std: [f32; 3]) {
    for c in 0..3 {
        for x in &mut v[c * n * n..(c + 1) * n * n] {
            *x = (*x - mean[c]) / std[c];
        }
    }
}

/// (DINOv2's input [3, 756, 756], CLIP's [3, 224, 224]) of the framed picture
pub fn inputs(framed: &Picture) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (framed.width as usize, framed.height as usize);
    let mut dino = bilinear(&planes(framed), h, w, DINO_SIZE);
    normalize(&mut dino, DINO_SIZE, DINO_MEAN, DINO_STD);
    let mut clip = planes(&framed.resize_lanczos(CLIP_SIZE as u32, CLIP_SIZE as u32));
    normalize(&mut clip, CLIP_SIZE, CLIP_MEAN, CLIP_STD);
    (dino, clip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_the_alpha_box_on_white() {
        // a 10 x 6 picture, opaque red in x 2..6, y 1..3 (a 4 x 2 box): a 4-square, then 4.8 -> 4 wide padded
        let mut p = Picture::new(10, 6, 4);
        for y in 1..3 {
            for x in 2..6 {
                let i = (y * 10 + x) * 4;
                p.data[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        let f = frame(&p);
        assert_eq!((f.width, f.height, f.channels), (4, 4, 3));
        // the box is centred vertically in its square: rows 1, 2 red, 0, 3 white
        assert_eq!(&f.data[0..3], &[255, 255, 255]);
        assert_eq!(&f.data[4 * 3..4 * 3 + 3], &[255, 0, 0]);
        assert_eq!(div255(255 * 255), 255);
        assert_eq!(div255(0), 0);
    }

    #[test]
    fn bilinear_keeps_a_constant() {
        let v = vec![0.25f32; 3 * 5 * 7];
        assert!(bilinear(&v, 5, 7, 9).iter().all(|x| (x - 0.25).abs() < 1e-6));
    }
}
