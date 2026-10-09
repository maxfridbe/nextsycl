//! Pictures on the host: 8-bit RGB or RGBA, row-major, top row first - what an image engine returns and what an edit or
//! a video keyframe takes. PNG in and out, with text chunks for the settings that made a picture.

use std::path::Path;

/// 8-bit pixels: `channels` 3 (RGB) or 4 (RGBA)
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub data: Vec<u8>,
}

impl Picture {
    pub fn new(width: u32, height: u32, channels: u8) -> Picture {
        Picture { width, height, channels, data: vec![0; width as usize * height as usize * channels as usize] }
    }

    /// A picture from a file's bytes: PNG, JPEG or WebP (8-bit RGB, or RGBA when it has alpha)
    pub fn decode(bytes: &[u8]) -> Result<Picture, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("not a picture this reads (PNG, JPEG, WebP): {e}"))?;
        Ok(if img.color().has_alpha() {
            let r = img.to_rgba8();
            Picture { width: r.width(), height: r.height(), channels: 4, data: r.into_raw() }
        } else {
            let r = img.to_rgb8();
            Picture { width: r.width(), height: r.height(), channels: 3, data: r.into_raw() }
        })
    }

    /// A PNG file (any bit depth and color type; 16-bit is reduced to 8, grey and palette expanded)
    pub fn read_png(path: &Path) -> Result<Picture, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut dec = png::Decoder::new(std::io::BufReader::new(file));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut r = dec.read_info().map_err(|e| format!("{}: {e}", path.display()))?;
        let mut buf = vec![0; r.output_buffer_size()];
        let info = r.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        buf.truncate(info.buffer_size());
        let (w, h) = (info.width, info.height);
        let data: Vec<u8> = match info.color_type {
            png::ColorType::Rgb | png::ColorType::Rgba => buf,
            png::ColorType::Grayscale => buf.iter().flat_map(|g| [*g, *g, *g]).collect(),
            png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
            png::ColorType::Indexed => return Err(format!("{}: a palette PNG was not expanded", path.display())),
        };
        let channels = (data.len() / (w as usize * h as usize).max(1)) as u8;
        Ok(Picture { width: w, height: h, channels, data })
    }

    /// A PNG's tEXt chunks (the settings `write_png` stored), without decoding its pixels
    pub fn png_text(path: &Path) -> Result<Vec<(String, String)>, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let r = png::Decoder::new(std::io::BufReader::new(file)).read_info().map_err(|e| format!("{}: {e}", path.display()))?;
        let info = r.info();
        Ok(info.uncompressed_latin1_text.iter().map(|t| (t.keyword.clone(), t.text.clone())).collect())
    }

    /// Write as PNG, with `text` as tEXt chunks (the prompt, seed, steps... that made it)
    pub fn write_png(&self, path: &Path, text: &[(&str, String)]) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.encode_png(std::io::BufWriter::new(file), text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The PNG's bytes (as `write_png` writes them)
    pub fn png_bytes(&self, text: &[(&str, String)]) -> Result<Vec<u8>, String> {
        let mut v = Vec::new();
        self.encode_png(&mut v, text)?;
        Ok(v)
    }

    fn encode_png(&self, out: impl std::io::Write, text: &[(&str, String)]) -> Result<(), String> {
        let mut enc = png::Encoder::new(out, self.width, self.height);
        enc.set_color(if self.channels == 4 { png::ColorType::Rgba } else { png::ColorType::Rgb });
        enc.set_depth(png::BitDepth::Eight);
        for (k, v) in text {
            enc.add_text_chunk(k.to_string(), v.clone()).map_err(|e| e.to_string())?;
        }
        let mut w = enc.write_header().map_err(|e| e.to_string())?;
        w.write_image_data(&self.data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Picture;

    #[test]
    fn png_round_trip() {
        let mut p = Picture::new(3, 2, 4);
        for (i, b) in p.data.iter_mut().enumerate() {
            *b = (i * 11) as u8;
        }
        let path = std::env::temp_dir().join(format!("nextsycl-picture-{}.png", std::process::id()));
        p.write_png(&path, &[("prompt", "a test".into())]).unwrap();
        let q = Picture::read_png(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(p, q);
    }
}

/// PIL's Lanczos-3 kernel
fn lanczos(x: f64) -> f64 {
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    if x.abs() < 3.0 { sinc(x) * sinc(x / 3.0) } else { 0.0 }
}

/// PIL's 8-bit resampling coefficients for one axis (ImagingResample: support 3 x the scale when shrinking, fixed point
/// with 22 fraction bits): for each output position its first input index and weights
fn coeffs(inp: usize, out: usize) -> Vec<(usize, Vec<i64>)> {
    const BITS: u32 = 22;
    let scale = inp as f64 / out as f64;
    let fs = scale.max(1.0);
    let support = 3.0 * fs;
    (0..out).map(|xx| {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(inp as i64) as usize - xmin;
        let w: Vec<f64> = (0..xmax).map(|x| lanczos((x as f64 + xmin as f64 - center + 0.5) / fs)).collect();
        let total: f64 = w.iter().sum();
        let k = w.iter().map(|v| {
            let v = if total != 0.0 { v / total } else { *v } * (1u64 << BITS) as f64;
            if v < 0.0 { (v - 0.5) as i64 } else { (v + 0.5) as i64 }
        }).collect();
        (xmin, k)
    }).collect()
}

impl Picture {
    /// Resized to `w` x `h` as PIL's Image.resize with LANCZOS does it (8 bits a channel: the horizontal pass first,
    /// rounded to bytes, then the vertical one)
    pub fn resize_lanczos(&self, w: u32, h: u32) -> Picture {
        let (iw, ih, c) = (self.width as usize, self.height as usize, self.channels as usize);
        let (w, h) = (w as usize, h as usize);
        if (w, h) == (iw, ih) {
            return self.clone();
        }
        let clip = |v: i64| ((v + (1 << 21)) >> 22).clamp(0, 255) as u8;
        // horizontal
        let mut tmp = vec![0u8; ih * w * c];
        let hc = coeffs(iw, w);
        for y in 0..ih {
            for (x, (x0, k)) in hc.iter().enumerate() {
                for ch in 0..c {
                    let s: i64 = k.iter().enumerate().map(|(i, kv)| kv * self.data[(y * iw + x0 + i) * c + ch] as i64).sum();
                    tmp[(y * w + x) * c + ch] = clip(s);
                }
            }
        }
        let mut out = vec![0u8; h * w * c];
        let vc = coeffs(ih, h);
        for (y, (y0, k)) in vc.iter().enumerate() {
            for x in 0..w {
                for ch in 0..c {
                    let s: i64 = k.iter().enumerate().map(|(i, kv)| kv * tmp[((y0 + i) * w + x) * c + ch] as i64).sum();
                    out[(y * w + x) * c + ch] = clip(s);
                }
            }
        }
        Picture { width: w as u32, height: h as u32, channels: self.channels, data: out }
    }

    /// As RGBA (an opaque alpha added to RGB)
    pub fn rgba(&self) -> Picture {
        match self.channels {
            4 => self.clone(),
            3 => Picture { width: self.width, height: self.height, channels: 4, data: self.data.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect() },
            1 => Picture { width: self.width, height: self.height, channels: 4, data: self.data.iter().flat_map(|v| [*v, *v, *v, 255]).collect() },
            _ => Picture { width: self.width, height: self.height, channels: 4, data: self.data.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect() },
        }
    }
}

#[cfg(test)]
mod resize_tests {
    use super::*;

    #[test]
    fn lanczos_keeps_flat_colour_and_size() {
        let p = Picture { width: 7, height: 5, channels: 3, data: vec![200; 7 * 5 * 3] };
        let r = p.resize_lanczos(3, 2);
        assert_eq!((r.width, r.height), (3, 2));
        assert!(r.data.iter().all(|v| *v == 200));
        let u = p.resize_lanczos(14, 10);
        assert!(u.data.iter().all(|v| *v == 200));
    }
}

