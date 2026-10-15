//! olmOCR-2 (allenai/olmOCR-2-7B-1025, a Qwen2.5-VL-7B fine-tune): a page in, its text out as olmOCR writes it -
//! Markdown, tables as HTML, equations as LaTeX, figures as `![alt](page_x_y_w_h.png)` - after a front matter of what
//! the model saw (language, whether the page is upright, table, diagram). From two GGUFs: the decoder (`qwen2vl`,
//! Q8_0) and the projector (`mmproj`, F16). The reading follows olmOCR's pipeline (olmocr/pipeline.py) and
//! transformers' Qwen2.5-VL:
//!
//! ```text
//!   the page rendered with its longest side 1288 (prep.rs)
//!   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}
//!   <|vision_start|> <image tokens> <|vision_end|><|im_end|>\n<|im_start|>assistant\n
//!   positions Qwen2-VL's (the image's tokens at (s, s + row, s + column), the text after from s + the larger side)
//!   greedy until <|im_end|>; a page the model says is turned (is_rotation_valid: false, rotation_correction r) is
//!   turned by r (counter-clockwise, as PIL's ROTATE_r) and read again, once
//! ```

pub mod check;
pub mod decoder;
mod ffi;
pub mod prep;
pub mod vision;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_gguf::Gguf;
use nextsycl_image::layout;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, OcrOutput, OcrPrompt, OcrRequest, Picture, Result, Sampler,
                     Schedule, Step};
use nextsycl_tok::Tokenizer;

pub const ARCH: &str = "olmocr2";
pub const ARCHS: &[&str] = &[ARCH, "qwen2vl"];
pub const ROLES: &[&str] = &["model", "mmproj"];
pub const IMAGE_TOKEN: u32 = 151655;
/// olmOCR's own instruction (olmocr/prompts/prompts.py: build_no_anchoring_v4_yaml_prompt)
pub const DEFAULT_PROMPT: &str = "Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\nIf there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\nReturn your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.";
/// olmOCR's MAX_TOKENS
pub const MAX_TOKENS: u32 = 8000;

pub fn kind() -> ImageKind {
    ImageKind { archs: ARCHS, name: "olmOCR-2 (pages to Markdown: a Qwen2.5-VL-7B fine-tune)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(OlmOcr::load(files, gpu, log)?))
}

pub struct OlmOcr {
    pub nsd: Nsd,
    pub vision: vision::Vision,
    pub decoder: decoder::Decoder,
    pub tok: Tokenizer,
    eos: u32,
    load_s: f64,
    work: Mutex<()>,
}

/// get_rope_index for one image (an `h` x `w` grid of units)
pub fn positions(ids: &[u32], h: usize, w: usize) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(ids.len());
    let mut next = 0u32;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == IMAGE_TOKEN {
            let s = next;
            for r in 0..h as u32 {
                for c in 0..w as u32 {
                    out.push([s, s + r, s + c]);
                }
            }
            i += h * w;
            next = s + h.max(w) as u32;
        } else {
            out.push([next; 3]);
            next += 1;
            i += 1;
        }
    }
    out
}

/// The front matter's values (between the first two `---` lines) and the text after it
pub fn front_matter(text: &str) -> (Vec<(String, String)>, &str) {
    let t = text.trim_start();
    let Some(rest) = t.strip_prefix("---\n") else { return (Vec::new(), text) };
    let Some(end) = rest.find("\n---") else { return (Vec::new(), text) };
    let vals = rest[..end].lines().filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
    let body = &rest[end + 4..];
    (vals, body.strip_prefix('\n').unwrap_or(body))
}

/// The turn a reading asks for: rotation_correction when is_rotation_valid is false
fn turn(text: &str) -> u32 {
    let (v, _) = front_matter(text);
    let get = |k: &str| v.iter().find(|(a, _)| a == k).map(|(_, b)| b.to_lowercase());
    if get("is_rotation_valid").as_deref() == Some("false") {
        get("rotation_correction").and_then(|r| r.parse::<u32>().ok()).filter(|r| [90, 180, 270].contains(r)).unwrap_or(0)
    } else {
        0
    }
}

/// A runaway: the tail one unit of at most 32 tokens over and over (16 times and 64 tokens at least); the length to keep
fn looping(seq: &[u32]) -> Option<usize> {
    for p in 1..=32usize {
        let tail = (16 * p).max(64);
        if seq.len() < tail + p {
            continue;
        }
        let mut s = seq.len() - tail;
        if (s..seq.len()).all(|i| seq[i] == seq[i - p]) {
            while s > p && seq[s - 1] == seq[s - 1 - p] {
                s -= 1;
            }
            return Some(s);
        }
    }
    None
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn pick(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f64> = logits.iter().map(|v| (((v - mx) / temperature) as f64).exp()).collect();
    let mut r = rng.next() * p.iter().sum::<f64>();
    for (i, v) in p.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}

fn add(all: &mut OcrOutput, r: &OcrOutput) {
    all.prompt = r.prompt.clone();
    all.prompt_tokens += r.prompt_tokens;
    all.tokens += r.tokens;
    all.prompt_seconds += r.prompt_seconds;
    all.seconds += r.seconds;
}

impl OlmOcr {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<OlmOcr> {
        let t0 = Instant::now();
        let path = |r: &str| files.get(r).ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let fm = Gguf::open(path("model")?).map_err(|e| Error(e.0))?;
        if fm.architecture() != "qwen2vl" {
            return Err(Error(format!("{}: a {} model, not qwen2vl", path("model")?.display(), fm.architecture())));
        }
        let fp = Gguf::open(path("mmproj")?).map_err(|e| Error(e.0))?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_gguf(&fm).map_err(Error)?;
        let eos = fm.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).unwrap_or(151645) as u32;
        let vision = vision::Vision::load(&fp, gpu, log)?;
        let decoder = decoder::Decoder::load(&fm, gpu, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("olmocr2: loaded on {} in {load_s:.1} s", gpu.name));
        Ok(OlmOcr { nsd, vision, decoder, tok, eos, load_s, work: Mutex::new(()) })
    }

    pub fn prompt_tokens(&self, prompt: &str, n_image: usize) -> Vec<u32> {
        let mut ids = self.tok.encode(&format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|vision_start|>"));
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, n_image));
        ids.extend(self.tok.encode("<|vision_end|><|im_end|>\n<|im_start|>assistant\n"));
        ids
    }

    pub fn embed(&self, ids: &[u32], img: &DevBuf) -> Result<DevBuf> {
        let e = self.decoder.g.n_embd;
        let text: Vec<u32> = ids.iter().map(|&t| if t == IMAGE_TOKEN { 0 } else { t }).collect();
        let x = self.decoder.embed(&self.nsd, &text)?;
        if let Some(i) = ids.iter().position(|&t| t == IMAGE_TOKEN) {
            let n = ids[i..].iter().take_while(|&&t| t == IMAGE_TOKEN).count();
            x.copy_within(i * e * 4, img, 0, n * e * 4)?;
        }
        Ok(x)
    }

    pub fn read(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let _w = self.work.lock().unwrap();
        if req.pages.is_empty() {
            return Err(Error("no pages to read".into()));
        }
        if req.pages.len() == 1 {
            return self.read_page(&req.pages[0], req, text);
        }
        let mut all = OcrOutput::default();
        for page in &req.pages {
            text("<PAGE>");
            let r = self.read_page(page, req, text)?;
            text("\n");
            all.text.push_str(&format!("<PAGE>{}\n", r.text));
            add(&mut all, &r);
        }
        Ok(all)
    }

    /// One page: by its columns (layout.rs) when asked and found, else whole
    fn read_page(&self, page: &Picture, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let regions = if req.columns.unwrap_or(true) { layout::regions(page) } else { Vec::new() };
        if regions.len() <= 1 {
            return self.read_one(page, req, text);
        }
        let mut all = OcrOutput::default();
        for (x, y, w, h) in regions {
            let c = page.channels as usize;
            let mut crop = Picture::new(w, h, page.channels);
            for yy in 0..h as usize {
                let src = ((y as usize + yy) * page.width as usize + x as usize) * c;
                crop.data[yy * w as usize * c..(yy + 1) * w as usize * c].copy_from_slice(&page.data[src..src + w as usize * c]);
            }
            let r = self.read_one(&crop, req, text)?;
            text("\n");
            let t = r.text.trim();
            if !t.is_empty() {
                all.text.push_str(t);
                all.text.push('\n');
            }
            add(&mut all, &r);
        }
        Ok(all)
    }

    /// One picture, its front matter dropped; turned and read again when the model says it is turned
    fn read_one(&self, page: &Picture, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let prompt = if req.prompt.trim().is_empty() { DEFAULT_PROMPT } else { req.prompt.trim() };
        let fitted = prep::fit(page);
        let mut out = self.write(&fitted, prompt, req, text, true)?;
        let r = turn(&out.text);
        if r != 0 {
            text(&format!("\n\n(the page is turned: read again, turned {r} degrees)\n\n"));
            let again = self.write(&prep::rotate(&fitted, r), prompt, req, text, true)?;
            let mut a = again;
            a.prompt_tokens += out.prompt_tokens;
            a.tokens += out.tokens;
            a.prompt_seconds += out.prompt_seconds;
            a.seconds += out.seconds;
            out = a;
        }
        out.text = front_matter(&out.text).1.to_string();
        Ok(out)
    }

    /// The image tokens of a fitted page and its unit grid (rows, columns)
    pub fn image_tokens(&self, page: &Picture, tap: &mut dyn FnMut(&str, &DevBuf) -> nextsycl_core::Result<()>) -> Result<(DevBuf, usize, usize)> {
        let seen = prep::see(page, self.vision.mean, self.vision.std);
        let img = self.vision.see(&self.nsd, &seen, tap)?;
        Ok((img, seen.gh / 2, seen.gw / 2))
    }

    /// One pass: see, read the prompt, write (`stream`: each piece to `text` as written)
    fn write(&self, page: &Picture, prompt: &str, req: &OcrRequest, text: &mut dyn FnMut(&str), stream: bool) -> Result<OcrOutput> {
        let t0 = Instant::now();
        let (img, h, w) = self.image_tokens(page, &mut |_, _| Ok(()))?;
        let ids = self.prompt_tokens(prompt, h * w);
        let pos = positions(&ids, h, w);
        let x = self.embed(&ids, &img)?;
        self.nsd.wait()?;
        drop(img);
        let max = if req.max_tokens == 0 { MAX_TOKENS } else { req.max_tokens } as usize;
        let mut cache = self.decoder.cache(ids.len() + max)?;
        let mut logits = self.decoder.pass(&self.nsd, &mut cache, &x, &pos, &mut |_, _| Ok(()))?;
        drop(x);
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let mut next_pos = pos.last().map_or(0, |p| p[0].max(p[1]).max(p[2]) + 1);
        let mut out_ids: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        let mut rng = Rng(req.seed ^ 0x6f6c6d6f);
        let t1 = Instant::now();
        while out_ids.len() < max {
            let next = pick(&logits, req.temperature, &mut rng);
            if next == self.eos {
                break;
            }
            out_ids.push(next);
            if let Some(keep) = looping(&out_ids) {
                out_ids.truncate(keep);
                break;
            }
            if stream {
                // the body as it is written: past the front matter (once it has closed)
                let bytes = self.tok.decode_bytes(&out_ids);
                let valid = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                let so_far = std::str::from_utf8(&bytes[..valid]).unwrap_or("");
                let start = if so_far.trim_start().starts_with("---") {
                    let t0 = so_far.len() - so_far.trim_start().len();
                    so_far[t0 + 3..].find("\n---").map(|e| (t0 + 3 + e + 4 + 1).min(valid))
                } else if !so_far.trim_start().is_empty() && !"---".starts_with(so_far.trim_start()) {
                    Some(0)
                } else {
                    None
                };
                if let Some(st) = start {
                    shown = shown.max(st);
                    if valid > shown {
                        text(&so_far[shown..valid]);
                        shown = valid;
                    }
                }
            }
            if out_ids.len() == max {
                break;
            }
            logits = self.decoder.step(&self.nsd, &mut cache, next, next_pos)?;
            next_pos += 1;
        }
        let all = String::from_utf8_lossy(&self.tok.decode_bytes(&out_ids)).into_owned();
        Ok(OcrOutput { text: all, prompt: prompt.to_string(), prompt_tokens: ids.len() as u32, tokens: out_ids.len() as u32, prompt_seconds,
                       seconds: t1.elapsed().as_secs_f64() })
    }
}

impl ImageEngine for OlmOcr {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: prep::LONGEST, height: prep::LONGEST, steps: 0, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error(format!("{ARCH} reads pages; it does not make pictures (image.ocr)")))
    }
    fn reads(&self) -> bool {
        true
    }
    fn ocr(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        self.read(req, text)
    }
    fn ocr_prompts(&self) -> Vec<OcrPrompt> {
        vec![OcrPrompt { id: "page", prompt: DEFAULT_PROMPT, title: "Page → Markdown (the model's own)" }]
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH}: decoder {} layers, {:.1} GiB of Q8_0, on {}", self.decoder.g.layers, self.decoder.weight_bytes as f64 / (1u64 << 30) as f64,
                     self.nsd.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_read_and_dropped() {
        let t = "---\nprimary_language: en\nis_rotation_valid: False\nrotation_correction: 90\nis_table: False\nis_diagram: False\n---\nHello";
        let (v, body) = front_matter(t);
        assert_eq!(v.len(), 5);
        assert_eq!(body, "Hello");
        assert_eq!(turn(t), 90);
        assert_eq!(turn("---\nis_rotation_valid: True\nrotation_correction: 0\n---\nx"), 0);
        assert_eq!(front_matter("no matter").1, "no matter");
    }

    #[test]
    fn positions_as_qwen2_vl() {
        let ids = [1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2];
        let p = positions(&ids, 2, 2);
        assert_eq!(p[4], [1, 2, 2]);
        assert_eq!(p[5], [3, 3, 3]);
    }
}
