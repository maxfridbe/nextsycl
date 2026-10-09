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

    /// Write as PNG, with `text` as tEXt chunks (the prompt, seed, steps... that made it)
    pub fn write_png(&self, path: &Path, text: &[(&str, String)]) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        enc.set_color(if self.channels == 4 { png::ColorType::Rgba } else { png::ColorType::Rgb });
        enc.set_depth(png::BitDepth::Eight);
        for (k, v) in text {
            enc.add_text_chunk(k.to_string(), v.clone()).map_err(|e| e.to_string())?;
        }
        let mut w = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
        w.write_image_data(&self.data).map_err(|e| format!("{}: {e}", path.display()))
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
