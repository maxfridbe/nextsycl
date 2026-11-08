//! The text side: P-H3-Image's Qwen3-VL-4B (ComfyUI's int8 ConvRot file, crates/qwen3vl) read after layers 13, 19, 25,
//! 31 and the final norm, and the text connector in the same file (diffusers' PH3ImageTextConnector) that maps those
//! five states to the 5,120 features H3's denoiser reads:
//!
//! ```text
//!   x = [hs13 | hs19 | hs25 | hs31 | hs36]            [T, 5 x 2,560]
//!   cond = ((x - input_mean) / input_std) . W^T * output_std + output_mean          W [5,120, 12,800], float32
//!   cond[0] = sink_embedding                          (the first token is an attention sink, the same for every prompt)
//! ```
//!
//! The standardizations are folded into W and a bias at load (one product). An edit's pictures go through the tower
//! in the same file first: `"<Picture i>: "`, the picture's vision tokens, then the instruction after `"Edit <Picture
//! 1>. "` - no chat template, no special tokens beyond the vision ones (diffusers' PH3ImageEditTextEncoderStep).

use std::path::Path;
use std::sync::Arc;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::{Dt, Nsd};
use nextsycl_gguf::safetensors::SafeTensors;
use nextsycl_image::{Error, Picture, Result};
use nextsycl_qwen3vl::vision::Seen;

/// The hidden states the connector reads (diffusers' hidden_state_indices: transformers' numbering, 36 = after the
/// final norm)
pub const TAPS: [usize; 5] = [13, 19, 25, 31, 36];
/// The conditioning's width (the denoiser's text_dim)
pub const COND: usize = 5120;
/// A picture is shown to the text encoder at about this many pixels, its sides multiples of 32 (vision_pixels)
pub const VISION_PIXELS: u64 = 1_000_000;

/// The modality tags of the conditioning's rows: text 1, a picture's vision tokens 0 (H3 tags them as video)
pub const TAG_TEXT: i32 = 1;
pub const TAG_VISION: i32 = 0;

pub struct TextEncoder {
    pub enc: nextsycl_qwen3vl::TextEncoder,
    pub nsd: Nsd,
    gpu: Arc<Gpu>,
    /// the folded projection [COND, 5 * hidden] and its bias [COND], float32 on the GPU
    w: DevBuf,
    b: DevBuf,
    sink: Vec<f32>,
}

/// What the text encoder made of a request: the conditioning rows float32 [T, COND] and each row's modality tag
pub struct Cond {
    pub rows: Vec<f32>,
    pub tags: Vec<i32>,
    pub ids: Vec<u32>,
}

fn ge(e: nextsycl_gguf::Error) -> Error {
    Error(e.0)
}

/// A picture's size for the text encoder: about `pixels` at its own aspect ratio, the sides multiples of `multiple`;
/// one already on that grid within 20 % of the area keeps its size (diffusers' `_size_at_area`)
pub fn size_at_area(w: u32, h: u32, pixels: u64, multiple: u32) -> (u32, u32) {
    let area = w as f64 * h as f64;
    if w.is_multiple_of(multiple) && h.is_multiple_of(multiple) && 0.8 * pixels as f64 <= area && area <= 1.2 * pixels as f64 {
        return (w, h);
    }
    let ratio = w as f64 / h as f64;
    let side = |v: f64| ((v / multiple as f64).round_ties_even() as u32 * multiple).max(multiple);
    (side((pixels as f64 * ratio).sqrt()), side((pixels as f64 / ratio).sqrt()))
}

impl TextEncoder {
    pub fn load(file: &Path, tokenizer: &Path, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<TextEncoder> {
        // the vision tower only when a picture comes (text to image keeps its 1.6 GiB off the card)
        let enc = nextsycl_qwen3vl::TextEncoder::load_with(file, tokenizer, gpu, false, log)?;
        if enc.layers() < TAPS[TAPS.len() - 1] {
            return Err(Error(format!("{}: {} layers, the connector reads after layer {}", file.display(), enc.layers(), TAPS[TAPS.len() - 1])));
        }
        let st = SafeTensors::open(file).map_err(ge)?;
        let f = |n: &str| st.f32(&format!("text_connector.{n}")).map_err(ge);
        let (im, is, om, os, sink) = (f("input_mean")?, f("input_std")?, f("output_mean")?, f("output_std")?, f("sink_embedding")?);
        let mut w = f("proj.weight")?;
        let k = TAPS.len() * enc.hidden;
        if w.len() != COND * k || im.len() != k || om.len() != COND || sink.len() != COND {
            return Err(Error(format!("{}: the text connector is not [{COND}, {k}]", file.display())));
        }
        // W' = diag(os) W diag(1 / is), b = om - W' im (both standardizations in the one product)
        let mut b = vec![0f32; COND];
        for (o, row) in w.chunks_exact_mut(k).enumerate() {
            let mut acc = 0f64;
            for (j, v) in row.iter_mut().enumerate() {
                *v = *v / is[j] * os[o];
                acc += *v as f64 * im[j] as f64;
            }
            b[o] = om[o] - acc as f32;
        }
        let nsd = Nsd::new(gpu)?;
        Ok(TextEncoder { w: DevBuf::from_f32(gpu, &w)?, b: DevBuf::from_f32(gpu, &b)?, sink, enc, nsd, gpu: gpu.clone() })
    }

    /// The conditioning of a text-to-image prompt (encoded as written) or of an edit (`pictures`: picture 1 first,
    /// each resized for the tower here)
    pub fn encode(&mut self, prompt: &str, pictures: &[Picture]) -> Result<Cond> {
        if !pictures.is_empty() {
            self.enc.load_vision(&mut |l| eprintln!("{l}"))?;
        }
        let tok = &self.enc.tok;
        let mut ids: Vec<u32> = Vec::new();
        let mut tags: Vec<i32> = Vec::new();
        let mut seen: Vec<Seen> = Vec::new();
        if !pictures.is_empty() {
            let vision = self.enc.vision.as_ref().ok_or_else(|| Error("the text encoder's file has no vision tower (an edit needs it)".into()))?;
            let special = |s: &str| tok.id(s).ok_or_else(|| Error(format!("the tokenizer has no {s}")));
            let (vs, pad, ve) = (special("<|vision_start|>")?, special("<|image_pad|>")?, special("<|vision_end|>")?);
            for (i, p) in pictures.iter().enumerate() {
                let rgb = p.rgb();
                let (w, h) = size_at_area(rgb.width, rgb.height, VISION_PIXELS, 32);
                let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
                let px: Vec<f32> = shown.data.iter().map(|v| *v as f32).collect();
                let s = vision.see(&self.nsd, &px, h as usize, w as usize)?;
                let label = tok.encode(&format!("<Picture {}>: ", i + 1));
                tags.extend(std::iter::repeat_n(TAG_TEXT, label.len()));
                ids.extend(label);
                ids.push(vs);
                ids.extend(std::iter::repeat_n(pad, s.n));
                ids.push(ve);
                tags.extend(std::iter::repeat_n(TAG_VISION, s.n + 2));
                seen.push(s);
            }
        }
        let prompt = match pictures.is_empty() || prompt.trim_start().starts_with("Edit <Picture 1>.") || prompt.trim_start().starts_with("subject_definitions:") {
            true => prompt.to_string(),
            false => format!("Edit <Picture 1>. {prompt}"),
        };
        let text = tok.encode(&prompt);
        tags.extend(std::iter::repeat_n(TAG_TEXT, text.len()));
        ids.extend(text);
        let rows = self.cond_of(&ids, &seen.iter().collect::<Vec<_>>())?;
        Ok(Cond { rows, tags, ids })
    }

    /// The five hidden states of `ids` (with `seen` at their image tokens) through the connector: [T, COND]
    pub fn cond_of(&self, ids: &[u32], seen: &[&Seen]) -> Result<Vec<f32>> {
        let hs = self.enc.encode_taps(&self.nsd, ids, seen, &TAPS, true)?;
        self.connect(&hs, ids.len())
    }

    /// The connector over hidden states [T, hidden] each (in TAPS order)
    pub fn connect(&self, hs: &[DevBuf], t: usize) -> Result<Vec<f32>> {
        let d = self.enc.hidden;
        let k = hs.len() * d;
        let x = DevBuf::f32(&self.gpu, t * k)?;
        for (i, h) in hs.iter().enumerate() {
            x.copy_2d(i * d * 4, k * 4, h, 0, d * 4, d * 4, t)?;
        }
        let out = DevBuf::f32(&self.gpu, t * COND)?;
        self.nsd.linear(x.ptr(), Dt::F32, t, k, self.w.ptr(), COND, self.b.ptr(), out.ptr(), Dt::F32)?;
        self.nsd.wait()?;
        let mut rows = out.to_f32()?;
        rows[..COND].copy_from_slice(&self.sink);
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::size_at_area;

    #[test]
    fn pictures_are_shown_at_about_a_megapixel_on_the_32_grid() {
        // on the grid within 20 %: kept
        assert_eq!(size_at_area(1024, 1024, 1_000_000, 32), (1024, 1024));
        assert_eq!(size_at_area(1152, 896, 1_000_000, 32), (1152, 896));
        // a 4 MP square: sqrt(1e6) = 1000 -> 31.25 x 32 -> 992
        assert_eq!(size_at_area(2048, 2048, 1_000_000, 32), (992, 992));
        // 16:9 at 8 MP: 1333.3 / 32 = 41.67 -> 1344; 750 / 32 = 23.4 -> 736
        assert_eq!(size_at_area(3872, 2176, 1_000_000, 32), (1344, 736));
        // never below one grid step
        assert_eq!(size_at_area(4000, 10, 1_000_000, 32), (20000, 64));
    }
}
