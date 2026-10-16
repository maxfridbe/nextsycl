//! The picture as Hunyuan3D's conditioner takes it (Tencent's ImageProcessorV2, then the encoder's own transform):
//! the object cut out by its alpha, recentred on a square canvas with a 15 % border (area resampling), composited on
//! white, resized to the processor's size (bicubic), then to the encoder's (bilinear) and normalized with ImageNet's
//! mean and deviation.
//!
//! A picture without transparency is cut out first. The shared background remover (BiRefNet) is not in this engine
//! yet (`remove_background` is its hook); until then an opaque picture's background is taken to be the near-uniform
//! colour along its border: the pixels of about that colour reachable from the border are cleared. A picture whose
//! border is not near-uniform is taken whole.

use nextsycl_model3d::{Error, Result, Texture};

/// An RGBA picture as float32 [h, w, 4] in [0, 255]
#[derive(Clone)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Img {
    pub fn from_texture(t: &Texture) -> Img {
        Img { w: t.width as usize, h: t.height as usize, px: t.rgba.iter().map(|&b| b as f32).collect() }
    }
}

/// How far two colours are apart at most for the fallback to call them the same background (per channel, of 255)
const BG_TOLERANCE: f32 = 24.0;

/// The background cut out of an opaque picture: the hook for the shared remover (BiRefNet). For now the fallback
/// described above; the reason it took (or did not take) the background away
pub fn remove_background(img: &mut Img) -> String {
    let (w, h) = (img.w, img.h);
    if w < 3 || h < 3 {
        return "too small to cut out".into();
    }
    let at = |x: usize, y: usize| (y * w + x) * 4;
    let border: Vec<usize> = (0..w).flat_map(|x| [at(x, 0), at(x, h - 1)]).chain((0..h).flat_map(|y| [at(0, y), at(w - 1, y)])).collect();
    // the border's median colour, and how much of the border is near it
    let mut key = [0f32; 3];
    for (c, k) in key.iter_mut().enumerate() {
        let mut v: Vec<f32> = border.iter().map(|&i| img.px[i + c]).collect();
        v.sort_by(f32::total_cmp);
        *k = v[v.len() / 2];
    }
    let near = |px: &[f32], i: usize| (0..3).all(|c| (px[i + c] - key[c]).abs() <= BG_TOLERANCE);
    let share = border.iter().filter(|&&i| near(&img.px, i)).count() as f32 / border.len() as f32;
    if share < 0.8 {
        return format!("the border is not one colour ({:.0} % near its median): the picture taken whole", share * 100.0);
    }
    // flood from the border through the pixels near that colour
    let mut bg = vec![false; w * h];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for x in 0..w {
        stack.push((x, 0));
        stack.push((x, h - 1));
    }
    for y in 0..h {
        stack.push((0, y));
        stack.push((w - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        let i = y * w + x;
        if bg[i] || !near(&img.px, i * 4) {
            continue;
        }
        bg[i] = true;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    let n = bg.iter().filter(|b| **b).count();
    if n == w * h {
        return "the picture is all background: taken whole".into();
    }
    for (i, b) in bg.iter().enumerate() {
        if *b {
            img.px[i * 4 + 3] = 0.0;
        }
    }
    format!("background ({:.0}, {:.0}, {:.0}) cut from the border: {:.0} % of the picture", key[0], key[1], key[2], 100.0 * n as f32 / (w * h) as f32)
}

/// Area resampling (OpenCV's INTER_AREA when shrinking: each output pixel the mean of the input it covers; bilinear
/// when growing) of [h, w, c] floats
pub fn resize_area(src: &[f32], w: usize, h: usize, c: usize, ow: usize, oh: usize) -> Vec<f32> {
    if ow >= w && oh >= h {
        return resize_linear(src, w, h, c, ow, oh);
    }
    let (sx, sy) = (w as f64 / ow as f64, h as f64 / oh as f64);
    let spans = |n: usize, s: f64, lim: usize| -> Vec<Vec<(usize, f64)>> {
        (0..n).map(|o| {
            let (a, b) = (o as f64 * s, ((o + 1) as f64 * s).min(lim as f64));
            let mut v = Vec::new();
            let mut i = a.floor() as usize;
            while (i as f64) < b && i < lim {
                let cover = (b.min(i as f64 + 1.0) - a.max(i as f64)).max(0.0);
                if cover > 0.0 {
                    v.push((i, cover / (b - a)));
                }
                i += 1;
            }
            v
        }).collect()
    };
    let (xs, ys) = (spans(ow, sx, w), spans(oh, sy, h));
    let mut out = vec![0f32; ow * oh * c];
    for (oy, yv) in ys.iter().enumerate() {
        for (ox, xv) in xs.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0f64;
                for &(y, wy) in yv {
                    for &(x, wx) in xv {
                        acc += wy * wx * src[(y * w + x) * c + ch] as f64;
                    }
                }
                out[(oy * ow + ox) * c + ch] = acc as f32;
            }
        }
    }
    out
}

/// Bilinear resampling (half-pixel centres, edges clamped) of [h, w, c] floats
pub fn resize_linear(src: &[f32], w: usize, h: usize, c: usize, ow: usize, oh: usize) -> Vec<f32> {
    let tap = |o: usize, n: usize, on: usize| -> (usize, usize, f32) {
        let s = ((o as f32 + 0.5) * n as f32 / on as f32 - 0.5).max(0.0);
        let i = (s.floor() as usize).min(n - 1);
        (i, (i + 1).min(n - 1), s - i as f32)
    };
    let mut out = vec![0f32; ow * oh * c];
    for oy in 0..oh {
        let (y0, y1, fy) = tap(oy, h, oh);
        for ox in 0..ow {
            let (x0, x1, fx) = tap(ox, w, ow);
            for ch in 0..c {
                let p = |x: usize, y: usize| src[(y * w + x) * c + ch];
                let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
                let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
                out[(oy * ow + ox) * c + ch] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    out
}

/// Bicubic resampling (OpenCV's INTER_CUBIC: A = -0.75, half-pixel centres, edges clamped) of [h, w, c] floats
pub fn resize_cubic(src: &[f32], w: usize, h: usize, c: usize, ow: usize, oh: usize) -> Vec<f32> {
    let weights = |t: f32| -> [f32; 4] {
        let a = -0.75f32;
        let c1 = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        let c2 = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
    };
    let taps = |on: usize, n: usize| -> Vec<(i64, [f32; 4])> {
        (0..on).map(|o| {
            let s = (o as f32 + 0.5) * n as f32 / on as f32 - 0.5;
            let f = s.floor();
            (f as i64, weights(s - f))
        }).collect()
    };
    let (tx, ty) = (taps(ow, w), taps(oh, h));
    let cx = |i: i64| i.clamp(0, w as i64 - 1) as usize;
    let cy = |i: i64| i.clamp(0, h as i64 - 1) as usize;
    let mut out = vec![0f32; ow * oh * c];
    for (oy, (fy, wy)) in ty.iter().enumerate() {
        for (ox, (fx, wx)) in tx.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0f32;
                for (j, wyj) in wy.iter().enumerate() {
                    let y = cy(fy - 1 + j as i64);
                    for (i, wxi) in wx.iter().enumerate() {
                        acc += wyj * wxi * src[(y * w + cx(fx - 1 + i as i64)) * c + ch];
                    }
                }
                out[(oy * ow + ox) * c + ch] = acc;
            }
        }
    }
    out
}

/// The object recentred on a square white canvas of the picture's longer side, `border` of it left empty
/// (ImageProcessorV2.recenter): RGB floats in [0, 255] (whole values, as the uint8 it makes), and the side
pub fn recenter(img: &Img, border: f32) -> Result<(Vec<f32>, usize)> {
    let (w, h) = (img.w, img.h);
    let (mut y0, mut y1, mut x0, mut x1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..h {
        for x in 0..w {
            if img.px[(y * w + x) * 4 + 3] > 0.0 {
                y0 = y0.min(y);
                y1 = y1.max(y);
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
        }
    }
    if y0 == usize::MAX || y1 == y0 || x1 == x0 {
        return Err(Error("the picture has no object in it (all transparent, or a line)".into()));
    }
    // as numpy slices it: [y0, y1) x [x0, x1)
    let (ch, cw) = (y1 - y0, x1 - x0);
    let size = w.max(h);
    let desired = (size as f32 * (1.0 - border)) as usize;
    let scale = desired as f32 / ch.max(cw) as f32;
    let (h2, w2) = (((ch as f32 * scale) as usize).max(1), ((cw as f32 * scale) as usize).max(1));
    let crop: Vec<f32> = (y0..y1).flat_map(|y| (x0..x1).flat_map(move |x| (0..4).map(move |c| (y, x, c)))).map(|(y, x, c)| img.px[(y * w + x) * 4 + c]).collect();
    let small = resize_area(&crop, cw, ch, 4, w2, h2);
    let (oy, ox) = ((size - h2) / 2, (size - w2) / 2);
    let mut out = vec![255f32; size * size * 3];
    for y in 0..h2 {
        for x in 0..w2 {
            let s = &small[(y * w2 + x) * 4..(y * w2 + x) * 4 + 4];
            let a = (s[3].round().clamp(0.0, 255.0)) / 255.0;
            for c in 0..3 {
                let v = s[c].round().clamp(0.0, 255.0) * a + 255.0 * (1.0 - a);
                out[((oy + y) * size + ox + x) * 3 + c] = v.clamp(0.0, 255.0).trunc();
            }
        }
    }
    Ok((out, size))
}

/// The conditioner's input: [3, enc, enc] normalized, from the picture (`cut`: whether to cut an opaque one out);
/// also a line on what was done to it
pub fn prepare(t: &Texture, cut: bool, proc_size: usize, enc_size: usize, border: f32) -> Result<(Vec<f32>, String)> {
    let mut img = Img::from_texture(t);
    let opaque = img.px.chunks_exact(4).all(|p| p[3] >= 255.0);
    let note = if !opaque {
        "its alpha taken as the outline".to_string()
    } else if cut {
        remove_background(&mut img)
    } else {
        "taken whole (opaque, background kept)".to_string()
    };
    let (canvas, side) = recenter(&img, border)?;
    let p = resize_cubic(&canvas, side, side, 3, proc_size, proc_size);
    // the processor's uint8, then the encoder's [0, 1]
    let p: Vec<f32> = p.into_iter().map(|v| v.round().clamp(0.0, 255.0) / 255.0).collect();
    let p = if enc_size == proc_size { p } else { resize_linear(&p, proc_size, proc_size, 3, enc_size, enc_size) };
    let n = enc_size;
    let mut planes = vec![0f32; 3 * n * n];
    for i in 0..n * n {
        for c in 0..3 {
            planes[c * n * n + i] = p[i * 3 + c];
        }
    }
    crate::dino::normalize(&mut planes, n);
    Ok((planes, note))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(w: u32, h: u32, opaque: bool) -> Texture {
        let mut t = Texture::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - w as f32 / 2.0).powi(2) + (y as f32 - h as f32 / 2.0).powi(2)).sqrt();
                let i = ((y * w + x) * 4) as usize;
                let inside = d < w.min(h) as f32 / 4.0;
                let px = if inside { [200, 30, 30, 255] } else if opaque { [90, 140, 90, 255] } else { [0, 0, 0, 0] };
                t.rgba[i..i + 4].copy_from_slice(&px);
            }
        }
        t
    }

    #[test]
    fn resizes_keep_constants() {
        let v = vec![7.0f32; 10 * 6 * 2];
        for r in [resize_area(&v, 10, 6, 2, 4, 3), resize_linear(&v, 10, 6, 2, 13, 9), resize_cubic(&v, 10, 6, 2, 13, 9)] {
            assert!(r.iter().all(|x| (x - 7.0).abs() < 1e-4));
        }
        // area: 2 x 2 blocks averaged
        let r = resize_area(&[0.0, 2.0, 4.0, 6.0], 2, 2, 1, 1, 1);
        assert_eq!(r, vec![3.0]);
    }

    #[test]
    fn the_object_fills_its_share_of_the_canvas() {
        let (rgb, side) = recenter(&Img::from_texture(&disc(200, 100, false)), 0.15).unwrap();
        assert_eq!(side, 200);
        // the disc (50 px across) grows to 170 px, centred; the corners white
        let red = |x: usize, y: usize| rgb[(y * side + x) * 3] > 150.0 && rgb[(y * side + x) * 3 + 1] < 100.0;
        assert!(red(100, 100) && red(20, 100) && red(179, 100) && !red(10, 100) && !red(190, 100));
        assert_eq!(&rgb[0..3], &[255.0, 255.0, 255.0]);
    }

    #[test]
    fn an_opaque_border_colour_is_cut() {
        let mut img = Img::from_texture(&disc(64, 64, true));
        let why = remove_background(&mut img);
        assert!(why.contains("cut"), "{why}");
        assert_eq!(img.px[3], 0.0);
        assert_eq!(img.px[(32 * 64 + 32) * 4 + 3], 255.0);
        let (px, _) = prepare(&disc(64, 64, true), true, 64, 70, 0.15).unwrap();
        assert_eq!(px.len(), 3 * 70 * 70);
    }
}
