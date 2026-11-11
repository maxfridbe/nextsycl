//! An image model's own text encoder as a describer (`qwen3vl-describe`): Qwen-Image 2.1's Qwen3-VL-8B-Instruct and
//! P-H3-Image's Qwen3-VL-4B are whole language models in their int8 ConvRot files - every layer, the final norm, the
//! head (the 4B's tied to its token embedding) and the vision tower - so the same file that conditions the image model
//! can look at a picture and write the prompt for it (image.describe `describer: native`, docs/api/image-describe.md).
//! A describer card (model-cards/image/*-describe.yml) names the image model's own files; nothing is downloaded.
//!
//! It serves the image kind's reading contract (`ocr`: pictures in, text out): the picture seen by the vision tower,
//! the chat template of Qwen3-VL-Instruct (a system turn when the request gives one - `extra["system"]` -, the picture
//! then the instruction in the user turn, no thinking), and `nextsycl_qwen3vl::chat::Writer` writing the answer.
//!
//! ```text
//!   <|im_start|>system\n{system}<|im_end|>\n
//!   <|im_start|>user\n<|vision_start|><|image_pad|> x n<|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n
//! ```

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_diffusion::kernels::Nsd;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, OcrOutput, OcrPrompt, OcrRequest, Picture, Result, Sampler,
                     Schedule, Step};
use nextsycl_qwen3vl::chat::{WriteOptions, Writer};
use nextsycl_qwen3vl::TextEncoder;

pub const ARCH: &str = "qwen3vl-describe";
pub const ROLES: &[&str] = &["text-encoder", "tokenizer"];
/// The picture's largest area as the tower sees it (pixels): ~1,024 positions, image.describe's default
pub const PIXELS: u64 = 1_048_576;
/// and at most (image.describe's max_pixels)
pub const MAX_PIXELS: u64 = 4_194_304;
/// tokens written when the request says none
pub const MAX_TOKENS: u32 = 512;
pub const DEFAULT_PROMPT: &str = "Describe this picture in detail.";

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "Qwen3-VL text encoder as a describer (an image model's own: pictures to text)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<nextsycl_core::Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    let path = |r: &str| files.get(r).ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
    Ok(Box::new(Describer::load(path("text-encoder")?, path("tokenizer")?, gpu, log)?))
}

pub struct Describer {
    pub nsd: Nsd,
    pub enc: TextEncoder,
    pub writer: Writer,
    stop: Vec<u32>,
    load_s: f64,
    work: Mutex<()>,
}

/// A picture's size for the tower: its own when its sides are multiples of 32 and it is at most `pixels`, else the
/// nearest such at its aspect within `pixels` (at least 32 a side)
pub fn tower_size(w: u32, h: u32, pixels: u64) -> (u32, u32) {
    let fit = |v: f64| ((v / 32.0).floor() as u32).max(1) * 32;
    let area = w as u64 * h as u64;
    let s = if area > pixels { (pixels as f64 / area as f64).sqrt() } else { 1.0 };
    (fit(w as f64 * s), fit(h as f64 * s))
}

/// The prompt's text around the picture's tokens: (before, after)
pub fn template(system: Option<&str>, instruction: &str) -> (String, String) {
    let sys = system.filter(|s| !s.trim().is_empty()).map(|s| format!("<|im_start|>system\n{s}<|im_end|>\n")).unwrap_or_default();
    (format!("{sys}<|im_start|>user\n<|vision_start|>"), format!("<|vision_end|>{instruction}<|im_end|>\n<|im_start|>assistant\n"))
}

impl Describer {
    pub fn load(te: &std::path::Path, tok: &std::path::Path, gpu: &Arc<nextsycl_core::Gpu>, log: &mut dyn FnMut(String)) -> Result<Describer> {
        let t0 = Instant::now();
        let nsd = Nsd::new(gpu)?;
        let enc = TextEncoder::load_with(te, tok, gpu, true, log)?;
        if enc.vision.is_none() {
            return Err(Error(format!("{}: no vision tower (model.visual.*): it cannot see a picture", te.display())));
        }
        let writer = Writer::new(&enc).map_err(|e| Error(format!("{}: {}", te.display(), e.0)))?;
        let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| enc.tok.id(t)).collect();
        if stop.is_empty() {
            return Err(Error("the tokenizer has no <|im_end|>".into()));
        }
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("{ARCH}: {} layers, head {} ({} tokens), on {} in {load_s:.1} s", enc.layers(), if writer.tied { "tied to the embedding" } else { "lm_head" },
                    writer.vocab, gpu.name));
        Ok(Describer { nsd, enc, writer, stop, load_s, work: Mutex::new(()) })
    }

    /// One picture described under `instruction` (and a system turn), the text as it is written to `text`
    pub fn describe(&self, page: &Picture, system: Option<&str>, instruction: &str, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let _w = self.work.lock().unwrap();
        let t0 = Instant::now();
        let rgb = page.rgb();
        let (w, h) = tower_size(rgb.width, rgb.height, MAX_PIXELS);
        let shown = if (w, h) == (rgb.width, rgb.height) { rgb } else { rgb.resize_lanczos(w, h) };
        let px: Vec<f32> = shown.data.iter().map(|v| *v as f32).collect();
        let vision = self.enc.vision.as_ref().ok_or_else(|| Error("no vision tower".into()))?;
        let seen = vision.see(&self.nsd, &px, h as usize, w as usize)?;
        let pad = self.enc.tok.id("<|image_pad|>").ok_or_else(|| Error("the tokenizer has no <|image_pad|>".into()))?;
        let (before, after) = template(system, instruction);
        let mut ids = self.enc.tokenize(&before);
        ids.extend(std::iter::repeat_n(pad, seen.n));
        ids.extend(self.enc.tokenize(&after));
        let o = WriteOptions { max_tokens: if req.max_tokens == 0 { MAX_TOKENS } else { req.max_tokens } as usize, temperature: req.temperature, top_p: 0.95,
                               seed: req.seed, stop: self.stop.clone() };
        let seen_s = t0.elapsed().as_secs_f64();
        let mut shown_len = 0usize;
        let out = self.writer.write(&self.enc, &self.nsd, &ids, &[&seen], &o, &mut |so_far| {
            text(&so_far[shown_len..]);
            shown_len = so_far.len();
            true
        })?;
        Ok(OcrOutput { text: out.text, prompt: instruction.to_string(), prompt_tokens: out.prompt_tokens as u32, tokens: out.ids.len() as u32,
                       prompt_seconds: seen_s + out.prompt_seconds, seconds: out.write_seconds })
    }
}

impl ImageEngine for Describer {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: 1024, height: 1024, steps: 0, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error(format!("{ARCH} describes pictures; it does not make them (image.describe describer native)")))
    }
    fn reads(&self) -> bool {
        true
    }
    fn ocr(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let page = match req.pages.as_slice() {
            [p] => p,
            [] => return Err(Error("no picture to describe".into())),
            _ => return Err(Error(format!("{} pictures: {ARCH} describes one at a time", req.pages.len()))),
        };
        let instruction = if req.prompt.trim().is_empty() { DEFAULT_PROMPT } else { req.prompt.trim() };
        self.describe(page, req.extra.get("system").map(String::as_str), instruction, req, text)
    }
    fn ocr_prompts(&self) -> Vec<OcrPrompt> {
        vec![OcrPrompt { id: "describe", prompt: DEFAULT_PROMPT, title: "Describe the picture" }]
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH}: {} layers, {} tokens, vision tower, on {}", self.enc.layers(), self.writer.vocab, self.nsd.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_is_shown_on_the_towers_grid_within_its_pixels() {
        assert_eq!(tower_size(1024, 768, PIXELS), (1024, 768));
        assert_eq!(tower_size(1000, 750, PIXELS), (992, 736));
        let (w, h) = tower_size(4000, 3000, PIXELS);
        assert!(w % 32 == 0 && h % 32 == 0 && (w * h) as u64 <= PIXELS && (w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.03, "{w}x{h}");
        assert_eq!(tower_size(10, 10, PIXELS), (32, 32));
    }

    #[test]
    fn the_template_is_qwen3_vl_instructs() {
        let (a, b) = template(Some("You write prompts."), "Write the prompt.");
        assert_eq!(a, "<|im_start|>system\nYou write prompts.<|im_end|>\n<|im_start|>user\n<|vision_start|>");
        assert_eq!(b, "<|vision_end|>Write the prompt.<|im_end|>\n<|im_start|>assistant\n");
        assert!(!template(None, "x").0.contains("system"));
        assert!(!template(Some("  "), "x").0.contains("system"));
    }
}
