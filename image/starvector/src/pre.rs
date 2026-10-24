//! The picture as the model sees it, exactly as StarVector's processor makes it from what the platform sends: the
//! picture composited on white (PIL's `alpha_composite`, its integer arithmetic), padded to a square with white
//! (torchvision's `pad`, fill 255), resized to 224 x 224 bicubically (PIL's `Image.resize` with BICUBIC: its separable
//! fixed-point resampling, a = -0.5, the support widened by the scale), then `ToTensor` and CLIP's normalisation.
//! Pure Rust on the host: the same bytes PIL gives, so the encoder starts from the reference's very pixels.

use nextsycl_image::Picture;

/// The side the encoder takes
pub const SIZE: usize = 224;
pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_1];

/// RGBA bytes of any picture (gray, gray + alpha, RGB, RGBA), as PIL's convert("RGBA")
fn rgba(p: &Picture) -> Vec<u8> {
    let n = p.width as usize * p.height as usize;
    let c = p.channels as usize;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let px = &p.data[i * c..i * c + c];
        let (r, g, b, a) = match c {
            1 => (px[0], px[0], px[0], 255),
            2 => (px[0], px[0], px[0], px[1]),
            3 => (px[0], px[1], px[2], 255),
            _ => (px[0], px[1], px[2], px[3]),
        };
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

/// PIL's ImagingAlphaComposite of `src` over opaque white, then convert("RGB") (AlphaComposite.c: PRECISION_BITS 7,
/// SHIFTFORDIV255)
pub fn flatten_on_white(p: &Picture) -> Vec<u8> {
    const PB: u32 = 7;
    let div255 = |a: u32| ((a >> 8) + a) >> 8;
    let src = rgba(p);
    let mut out = Vec::with_capacity(src.len() / 4 * 3);
    for px in src.chunks_exact(4) {
        let sa = px[3] as u32;
        if sa == 0 {
            out.extend_from_slice(&[255, 255, 255]);
            continue;
        }
        let blend = 255 * (255 - sa);
        let outa255 = sa * 255 + blend;
        let coef1 = sa * 255 * 255 * (1 << PB) / outa255;
        let coef2 = 255 * (1 << PB) - coef1;
        for &c in &px[..3] {
            let t = c as u32 * coef1 + 255 * coef2;
            out.push((div255(t + (0x80 << PB)) >> PB) as u8);
        }
    }
    out
}

/// torchvision's pad to a square with white (left, top: half the difference rounded down)
pub fn pad_square(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, usize) {
    let m = w.max(h);
    let (l, t) = ((m - w) / 2, (m - h) / 2);
    let mut out = vec![255u8; m * m * 3];
    for y in 0..h {
        out[((y + t) * m + l) * 3..((y + t) * m + l + w) * 3].copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    (out, m)
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// PIL's precompute_coeffs + normalize_coeffs_8bpc (Resample.c): per output index (xmin, the integer weights)
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    const PB: i32 = 22;
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let mut k: Vec<f64> = (0..xmax).map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                for v in &mut k {
                    *v /= ww;
                }
            }
            let ki = k.iter().map(|&v| if v < 0.0 { (-0.5 + v * (1 << PB) as f64) as i32 } else { (0.5 + v * (1 << PB) as f64) as i32 }).collect();
            (xmin, ki)
        })
        .collect()
}

fn clip8(v: i32) -> u8 {
    (v >> 22).clamp(0, 255) as u8
}

/// PIL's bicubic resize of RGB bytes [h, w] to [oh, ow]: horizontal, then vertical, each 8-bit
pub fn resize(rgb: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let horiz = if ow != w {
        let k = coeffs(w, ow);
        let mut out = vec![0u8; h * ow * 3];
        for y in 0..h {
            for (xx, (xmin, ks)) in k.iter().enumerate() {
                for c in 0..3 {
                    let mut ss: i32 = 1 << 21;
                    for (x, kv) in ks.iter().enumerate() {
                        ss += rgb[(y * w + xmin + x) * 3 + c] as i32 * kv;
                    }
                    out[(y * ow + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        out
    } else {
        rgb.to_vec()
    };
    if oh == h {
        return horiz;
    }
    let k = coeffs(h, oh);
    let mut out = vec![0u8; oh * ow * 3];
    for (yy, (ymin, ks)) in k.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let mut ss: i32 = 1 << 21;
                for (y, kv) in ks.iter().enumerate() {
                    ss += horiz[((ymin + y) * ow + x) * 3 + c] as i32 * kv;
                }
                out[(yy * ow + x) * 3 + c] = clip8(ss);
            }
        }
    }
    out
}

/// The encoder's input [3, 224, 224] (channels first, normalised) from a picture as the platform sends it
pub fn pixels(p: &Picture) -> Vec<f32> {
    let flat = flatten_on_white(p);
    let (sq, m) = pad_square(&flat, p.width as usize, p.height as usize);
    let small = resize(&sq, m, m, SIZE, SIZE);
    let mut out = vec![0f32; 3 * SIZE * SIZE];
    for i in 0..SIZE * SIZE {
        for c in 0..3 {
            out[c * SIZE * SIZE + i] = (small[i * 3 + c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_under_transparency_and_opaque_untouched() {
        let p = Picture { width: 3, height: 1, channels: 4, data: vec![10, 20, 30, 255, 200, 100, 0, 0, 0, 0, 0, 128] };
        let f = flatten_on_white(&p);
        assert_eq!(&f[..3], &[10, 20, 30], "opaque: as it was");
        assert_eq!(&f[3..6], &[255, 255, 255], "transparent: white");
        // half black over white: PIL gives 127
        assert_eq!(f[6], 127);
    }

    #[test]
    fn a_square_pads_nothing_and_a_wide_one_pads_top_and_bottom() {
        let (sq, m) = pad_square(&[1, 2, 3, 4, 5, 6], 2, 1);
        assert_eq!(m, 2);
        assert_eq!(&sq[..6], &[1, 2, 3, 4, 5, 6][..], "(2 - 1) / 2 = 0 rows on top: the extra row goes below");
        assert_eq!(&sq[6..], &[255; 6]);
        let (sq, _) = pad_square(&[9, 9, 9], 1, 1);
        assert_eq!(sq, vec![9, 9, 9]);
    }

    #[test]
    fn a_flat_picture_resizes_to_itself_and_weights_sum_to_one() {
        let rgb = vec![77u8; 50 * 30 * 3];
        assert!(resize(&rgb, 50, 30, 224, 224).iter().all(|&v| v == 77));
        for (_, k) in coeffs(1000, 224) {
            let s: i64 = k.iter().map(|&v| v as i64).sum();
            assert!((s - (1 << 22)).abs() <= k.len() as i64, "{s}");
        }
    }
}
