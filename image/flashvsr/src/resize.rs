//! A frame made ready for FlashVSR, on the host: upscaled by Pillow's bicubic (its `Image.resize(..., BICUBIC)`,
//! the reference's), arithmetic for arithmetic - 22-bit fixed-point coefficients, a horizontal pass then a vertical one,
//! each rounded to bytes - then mirrored out (numpy's 'reflect') to the next multiple of 128 on each side, split evenly.

/// Pillow's bicubic filter (a = -0.5)
fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Each output position's first input and its integer coefficients (Pillow's precompute_coeffs + normalize_coeffs_8bpc)
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, Vec<i32>)>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let n = xmax - xmin;
        let mut k: Vec<f64> = (0..n).map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
        let ww: f64 = k.iter().sum();
        if ww != 0.0 {
            k.iter_mut().for_each(|v| *v /= ww);
        }
        let ki = k.iter().map(|&v| if v < 0.0 { (-0.5 + v * (1u64 << PRECISION_BITS) as f64) as i32 } else { (0.5 + v * (1u64 << PRECISION_BITS) as f64) as i32 }).collect();
        out.push((xmin, ki));
    }
    (ksize, out)
}

fn clip8(v: i64) -> u8 {
    if v >= (1i64 << PRECISION_BITS) << 8 {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

/// rgb [h, w, 3] -> [oh, ow, 3], Pillow's bicubic (rows in bands over the cores: the same arithmetic, in parallel)
pub fn bicubic_resize(rgb: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let (_, cx) = coeffs(w, ow);
    let (_, cy) = coeffs(h, oh);
    // a few bands: several frames are prepared at once, each on its own thread
    let threads = 4;
    // horizontal, over the rows the vertical pass reads (all of them when upscaling)
    let mut tmp = vec![0u8; h * ow * 3];
    let band = h.div_ceil(threads);
    std::thread::scope(|sc| {
        for (bi, chunk) in tmp.chunks_mut(band * ow * 3).enumerate() {
            let cx = &cx;
            sc.spawn(move || {
                for (r, out) in chunk.chunks_mut(ow * 3).enumerate() {
                    let y = bi * band + r;
                    let row = &rgb[y * w * 3..(y + 1) * w * 3];
                    for (x, (x0, k)) in cx.iter().enumerate() {
                        for c in 0..3 {
                            let mut s: i64 = 1 << (PRECISION_BITS - 1);
                            for (i, kv) in k.iter().enumerate() {
                                s += row[(x0 + i) * 3 + c] as i64 * *kv as i64;
                            }
                            out[x * 3 + c] = clip8(s);
                        }
                    }
                }
            });
        }
    });
    let mut out = vec![0u8; oh * ow * 3];
    let band = oh.div_ceil(threads);
    std::thread::scope(|sc| {
        for (bi, chunk) in out.chunks_mut(band * ow * 3).enumerate() {
            let (cy, tmp) = (&cy, &tmp);
            sc.spawn(move || {
                for (r, orow) in chunk.chunks_mut(ow * 3).enumerate() {
                    let (y0, k) = &cy[bi * band + r];
                    for x in 0..ow {
                        for c in 0..3 {
                            let mut s: i64 = 1 << (PRECISION_BITS - 1);
                            for (i, kv) in k.iter().enumerate() {
                                s += tmp[((y0 + i) * ow + x) * 3 + c] as i64 * *kv as i64;
                            }
                            orow[x * 3 + c] = clip8(s);
                        }
                    }
                }
            });
        }
    });
    out
}

/// The geometry of a clip's frames: the upscaled size, the padded size (multiples of 128) and the picture's place in it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub w: usize,
    pub h: usize,
    pub sw: usize,
    pub sh: usize,
    pub tw: usize,
    pub th: usize,
    pub left: usize,
    pub top: usize,
}

impl Geometry {
    pub fn new(w: usize, h: usize, scale: f64) -> Geometry {
        let sw = (w as f64 * scale).round() as usize;
        let sh = (h as f64 * scale).round() as usize;
        let tw = sw.div_ceil(128) * 128;
        let th = sh.div_ceil(128) * 128;
        Geometry { w, h, sw, sh, tw, th, left: (tw - sw) / 2, top: (th - sh) / 2 }
    }
}

/// numpy 'reflect' index (the edge not repeated)
fn reflect(i: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    let p = 2 * (n - 1);
    let mut j = i.rem_euclid(p);
    if j >= n {
        j = p - j;
    }
    j as usize
}

/// A frame [h, w, 3] upscaled and mirrored out: [th, tw, 3]
pub fn prepare(rgb: &[u8], g: &Geometry) -> Vec<u8> {
    let up = if g.sw == g.w && g.sh == g.h { rgb.to_vec() } else { bicubic_resize(rgb, g.w, g.h, g.sw, g.sh) };
    if g.tw == g.sw && g.th == g.sh {
        return up;
    }
    let mut out = vec![0u8; g.th * g.tw * 3];
    for y in 0..g.th {
        let sy = reflect(y as i64 - g.top as i64, g.sh as i64);
        for x in 0..g.tw {
            let sx = reflect(x as i64 - g.left as i64, g.sw as i64);
            out[(y * g.tw + x) * 3..(y * g.tw + x) * 3 + 3].copy_from_slice(&up[(sy * g.sw + sx) * 3..(sy * g.sw + sx) * 3 + 3]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_sum_to_one() {
        let (_, c) = coeffs(864, 1728);
        for (_, k) in &c {
            let s: i64 = k.iter().map(|&v| v as i64).sum();
            assert!((s - (1 << PRECISION_BITS)).abs() <= 4, "{s}");
        }
    }

    #[test]
    fn a_flat_picture_stays_flat() {
        let src = vec![77u8; 10 * 6 * 3];
        let out = bicubic_resize(&src, 10, 6, 20, 12);
        assert!(out.iter().all(|&v| v == 77));
    }

    #[test]
    fn mirrored_like_numpy() {
        // numpy.pad([0, 1, 2, 3], 2, 'reflect') == [2, 1, 0, 1, 2, 3, 2, 1]
        let got: Vec<usize> = (-2..6).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![2, 1, 0, 1, 2, 3, 2, 1]);
        let g = Geometry::new(864, 480, 2.0);
        assert_eq!((g.sw, g.sh, g.tw, g.th, g.left, g.top), (1728, 960, 1792, 1024, 32, 32));
    }
}
