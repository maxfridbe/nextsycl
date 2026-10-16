//! Pillow's resampling (Image.resize), as Tencent's paint pipeline resizes its maps and pictures: separable,
//! antialiased (the filter stretched by the scale when shrinking), bicubic with a = -0.5 or bilinear, rounded to 8
//! bits after each pass; RGBA premultiplied around it. On 8-bit pictures [h, w, c] held as floats (whole values).

/// The filters Pillow resamples with
#[derive(Clone, Copy)]
pub enum Filter {
    Bicubic,
    Bilinear,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bicubic => 2.0,
            Filter::Bilinear => 1.0,
        }
    }
    fn at(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bicubic => {
                let a = -0.5;
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Filter::Bilinear => (1.0 - x).max(0.0),
        }
    }
}

/// Each output index's (first input index, weights)
fn taps(inn: usize, out: usize, f: Filter) -> Vec<(usize, Vec<f64>)> {
    let scale = inn as f64 / out as f64;
    let fscale = scale.max(1.0);
    let support = f.support() * fscale;
    (0..out).map(|o| {
        let center = (o as f64 + 0.5) * scale;
        let lo = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let hi = ((center + support + 0.5).floor() as i64).min(inn as i64) as usize;
        let mut w: Vec<f64> = (lo..hi).map(|x| f.at((x as f64 - center + 0.5) / fscale)).collect();
        let s: f64 = w.iter().sum();
        if s != 0.0 {
            for v in &mut w {
                *v /= s;
            }
        }
        (lo, w)
    }).collect()
}

/// [h, w, c] 8-bit values resized to [oh, ow, c]
pub fn resize(src: &[f32], w: usize, h: usize, c: usize, ow: usize, oh: usize, f: Filter) -> Vec<f32> {
    let r8 = |v: f64| v.round().clamp(0.0, 255.0) as f32;
    let horiz = if ow == w { src.to_vec() } else {
        let t = taps(w, ow, f);
        let mut o = vec![0f32; h * ow * c];
        for y in 0..h {
            for (x, (lo, ws)) in t.iter().enumerate() {
                for ch in 0..c {
                    let mut acc = 0f64;
                    for (k, wk) in ws.iter().enumerate() {
                        acc += wk * src[(y * w + lo + k) * c + ch] as f64;
                    }
                    o[(y * ow + x) * c + ch] = r8(acc);
                }
            }
        }
        o
    };
    if oh == h {
        return horiz;
    }
    let t = taps(h, oh, f);
    let mut o = vec![0f32; oh * ow * c];
    for (y, (lo, ws)) in t.iter().enumerate() {
        for x in 0..ow {
            for ch in 0..c {
                let mut acc = 0f64;
                for (k, wk) in ws.iter().enumerate() {
                    acc += wk * horiz[((lo + k) * ow + x) * c + ch] as f64;
                }
                o[(y * ow + x) * c + ch] = r8(acc);
            }
        }
    }
    o
}

/// An RGBA picture resized as Pillow does (premultiplied by alpha around the resampling)
pub fn resize_rgba(src: &[f32], w: usize, h: usize, ow: usize, oh: usize, f: Filter) -> Vec<f32> {
    let pre: Vec<f32> = src.chunks_exact(4).flat_map(|p| {
        let a = p[3];
        [(p[0] * a / 255.0).round(), (p[1] * a / 255.0).round(), (p[2] * a / 255.0).round(), a]
    }).collect();
    let r = resize(&pre, w, h, 4, ow, oh, f);
    r.chunks_exact(4).flat_map(|p| {
        let a = p[3];
        if a == 0.0 { [0.0, 0.0, 0.0, 0.0] } else { [(p[0] * 255.0 / a).round().min(255.0), (p[1] * 255.0 / a).round().min(255.0), (p[2] * 255.0 / a).round().min(255.0), a] }
    }).collect()
}

/// RGB on white through the alpha (Image.paste with the alpha as the mask), 8-bit values
pub fn on_white(rgba: &[f32]) -> Vec<f32> {
    rgba.chunks_exact(4).flat_map(|p| {
        let m = p[3];
        [0, 1, 2].map(|k| ((p[k] * m + 255.0 * (255.0 - m)) / 255.0).round())
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_stay_and_sizes_follow() {
        let v = vec![200f32; 16 * 8 * 3];
        let r = resize(&v, 16, 8, 3, 4, 2, Filter::Bicubic);
        assert_eq!(r.len(), 4 * 2 * 3);
        assert!(r.iter().all(|x| *x == 200.0));
        let up = resize(&v, 16, 8, 3, 20, 10, Filter::Bilinear);
        assert!(up.iter().all(|x| *x == 200.0));
        // a 4:1 shrink of a step averages across it
        let mut s = vec![0f32; 8];
        s[4..].fill(255.0);
        let r = resize(&s, 8, 1, 1, 2, 1, Filter::Bicubic);
        assert!(r[0] < 30.0 && r[1] > 225.0, "{r:?}");
    }

    #[test]
    fn white_under_transparency() {
        assert_eq!(on_white(&[10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0, 255.0]), vec![255.0, 255.0, 255.0, 10.0, 20.0, 30.0]);
    }
}
