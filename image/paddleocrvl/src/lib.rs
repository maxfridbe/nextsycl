//! PaddleOCR-VL (PaddlePaddle/PaddleOCR-VL-1.6): a page, or an element cut from one, in - its text out: plain text,
//! a table, a formula (LaTeX), a chart (a table of its values), a seal, or text with its place (spotting), each by its
//! instruction. From two GGUFs - the decoder (`paddleocr`: ERNIE-4.5 0.3B, BF16) and the projector (`mmproj`: the
//! SigLIP NaViT tower and the 2 x 2 merge, BF16) - and the original repo's tokenizer.json (the GGUF's vocabulary is
//! SentencePiece's; transformers reads the BPE one). The reading follows transformers' implementation
//! (modeling_paddleocr_vl.py, its chat template):
//!
//! ```text
//!   <|begin_of_sentence|>User: <|IMAGE_START|> <image tokens> <|IMAGE_END|>{instruction}\nAssistant:\n
//!   image tokens: the page at its own size (prep.rs), a token a 28 x 28 block; positions Qwen2-VL's (the text before
//!   one apart; the image's tokens at (start, start + row, start + column); the text after from start + the larger
//!   side)
//!   then greedy tokens until </s>
//! ```

pub mod check;
pub mod decoder;
mod ffi;
pub mod format;
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

/// The architecture (a registry entry may name the GGUF's too)
pub const ARCH: &str = "paddleocr-vl";
pub const ARCHS: &[&str] = &[ARCH, "paddleocr"];
/// Its files: the decoder, the projector and the original tokenizer.json
pub const ROLES: &[&str] = &["model", "mmproj", "tokenizer"];
pub const IMAGE_TOKEN: u32 = 100295;
pub const DEFAULT_PROMPT: &str = "OCR:";
pub const TABLE_PROMPT: &str = "Table Recognition:";
/// Spotting reads finer: the page may be larger
pub const SPOTTING_MAX_PIXELS: u32 = 1605632;
pub const MAX_TOKENS: u32 = 4096;

pub fn kind() -> ImageKind {
    ImageKind { archs: ARCHS, name: "PaddleOCR-VL (pages, tables, formulas, charts, seals: a SigLIP NaViT tower, an ERNIE-4.5 0.3B decoder)", roles: ROLES, load,
                options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(PaddleOcrVl::load(files, gpu, log)?))
}

pub struct PaddleOcrVl {
    pub nsd: Nsd,
    pub vision: vision::Vision,
    pub decoder: decoder::Decoder,
    pub tok: Tokenizer,
    eos: u32,
    load_s: f64,
    /// one reading at a time
    work: Mutex<()>,
}

/// A small generator for sampling (splitmix64)
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

/// The tokens that would complete an n-gram already in the last `window` tokens of `seq`
fn banned(seq: &[u32], n: usize, window: usize) -> Vec<u32> {
    if n == 0 || seq.len() < n {
        return Vec::new();
    }
    let start = seq.len().saturating_sub(window);
    let end = seq.len() + 1 - n;
    if end <= start {
        return Vec::new();
    }
    let prefix = &seq[seq.len() - (n - 1)..];
    let mut out: Vec<u32> = (start..end).filter(|&i| &seq[i..i + n - 1] == prefix).map(|i| seq[i + n - 1]).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// A runaway: the written tokens' tail one unit of at most 32 tokens over and over (16 times and 64 tokens at least);
/// the length to keep (the tail cut to one copy of the unit). Greedy reading falls into these on pages it cannot place
/// (a column of marks, a ruled blank) and would write them to the end of its budget.
fn looping(seq: &[u32]) -> Option<usize> {
    for p in 1..=32usize {
        let tail = (16 * p).max(64);
        if seq.len() < tail + p {
            continue;
        }
        let mut s = seq.len() - tail;
        if (s..seq.len()).all(|i| seq[i] == seq[i - p]) {
            // back to where the repeating began
            while s > p && seq[s - 1] == seq[s - 1 - p] {
                s -= 1;
            }
            return Some(s);
        }
    }
    None
}

fn pick(logits: &mut [f32], ban: &[u32], temperature: f32, rng: &mut Rng) -> u32 {
    for b in ban {
        if let Some(v) = logits.get_mut(*b as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    if temperature <= 0.0 {
        // the first of the largest (torch.argmax)
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

/// get_rope_index for one image: (time, row, column) of each token - the text one apart, the image's tokens (an
/// `h` x `w` grid of merged blocks) at (s, s + row, s + column) for the image's first index s, the text after it from
/// s + max(h, w)
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

impl PaddleOcrVl {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<PaddleOcrVl> {
        let t0 = Instant::now();
        let path = |r: &str| files.get(r).ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let fm = Gguf::open(path("model")?).map_err(|e| Error(e.0))?;
        if fm.architecture() != "paddleocr" {
            return Err(Error(format!("{}: a {} model, not paddleocr", path("model")?.display(), fm.architecture())));
        }
        let fp = Gguf::open(path("mmproj")?).map_err(|e| Error(e.0))?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_hf_json(path("tokenizer")?).map_err(Error)?;
        let eos = fm.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
        let vision = vision::Vision::load(&fp, gpu, &nsd, log)?;
        let decoder = decoder::Decoder::load(&fm, gpu, &nsd, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("paddleocrvl: loaded on {} in {load_s:.1} s", gpu.name));
        Ok(PaddleOcrVl { nsd, vision, decoder, tok, eos, load_s, work: Mutex::new(()) })
    }

    /// The prompt's tokens around `n_image` image tokens
    pub fn prompt_tokens(&self, prompt: &str, n_image: usize) -> Vec<u32> {
        let mut ids = self.tok.encode("<|begin_of_sentence|>User: <|IMAGE_START|>");
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, n_image));
        ids.extend(self.tok.encode(&format!("<|IMAGE_END|>{prompt}\nAssistant:\n")));
        ids
    }

    /// The prompt's embeddings: the text's rows, the image tokens' rows from `img`
    pub fn embed(&self, ids: &[u32], img: &DevBuf) -> Result<DevBuf> {
        let e = vision::OUT;
        let text: Vec<u32> = ids.iter().map(|&t| if t == IMAGE_TOKEN { 0 } else { t }).collect();
        let x = self.decoder.embed(&self.nsd, &text)?;
        if let Some(i) = ids.iter().position(|&t| t == IMAGE_TOKEN) {
            let n = ids[i..].iter().take_while(|&&t| t == IMAGE_TOKEN).count();
            x.copy_within(i * e * 4, img, 0, n * e * 4)?;
        }
        Ok(x)
    }

    /// A reading (one at a time): several pages each on its own, joined with <PAGE> marks; a page by its columns
    pub fn read(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let _w = self.work.lock().unwrap();
        if req.pages.is_empty() {
            return Err(Error("no pages to read".into()));
        }
        if req.pages.len() == 1 {
            return self.read_page(req, text);
        }
        let mut all = OcrOutput::default();
        for page in &req.pages {
            text("<PAGE>");
            let one = OcrRequest { pages: vec![page.clone()], extra: Default::default(), ..req.clone() };
            let r = self.read_page(&one, text)?;
            text("\n");
            all.text.push_str(&format!("<PAGE>{}\n", r.text));
            add(&mut all, &r);
        }
        Ok(all)
    }

    /// One page: its columns found (layout.rs) and each read on its own - or, columns off or none found, whole
    fn read_page(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let page = &req.pages[0];
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
                all.text.push_str(&format::to_page(t, (x, y, w, h), (page.width, page.height)));
                all.text.push('\n');
            }
            add(&mut all, &r);
        }
        Ok(all)
    }

    /// The image tokens of a page and its patch grid's merged size (rows, columns)
    pub fn image_tokens(&self, page: &Picture, max_pixels: u32, tap: &mut dyn FnMut(&str, &DevBuf) -> nextsycl_core::Result<()>) -> Result<(DevBuf, usize, usize)> {
        let seen = prep::see(page, max_pixels);
        let img = self.vision.see(&self.nsd, &seen.patches, seen.gh, seen.gw, tap)?;
        Ok((img, seen.gh / 2, seen.gw / 2))
    }

    /// One pass over one picture: see it, read the prompt, write
    /// One picture read; a plain reading that ran away (a grid of marks it cannot place - a transcript's grade table)
    /// read again as a table, the table in place of where the reading began to repeat
    fn read_one(&self, page: &Picture, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let prompt = if req.prompt.trim().is_empty() { DEFAULT_PROMPT } else { req.prompt.trim() };
        let (mut out, looped) = self.write(page, prompt, req, text)?;
        if !looped || prompt != DEFAULT_PROMPT || req.extra.get("rescue").is_some_and(|v| v == "false") {
            return Ok(out);
        }
        let (tab, tab_looped) = self.write(page, TABLE_PROMPT, req, &mut |_| {})?;
        if tab_looped || !tab.text.contains("<table") {
            return Ok(out);
        }
        let merged = rescue(&out.text, &tab.text);
        text(&merged[merged.len().min(out.text.len())..]);
        out.text = merged;
        out.tokens += tab.tokens;
        out.prompt_tokens += tab.prompt_tokens;
        out.prompt_seconds += tab.prompt_seconds;
        out.seconds += tab.seconds;
        Ok(out)
    }

    /// One pass: see the picture, read the prompt, write (and whether it ran away)
    fn write(&self, page: &Picture, prompt: &str, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<(OcrOutput, bool)> {
        let t0 = Instant::now();
        let max_pixels = if prompt.starts_with("Spotting") { SPOTTING_MAX_PIXELS } else { prep::MAX_PIXELS };
        let (img, h, w) = self.image_tokens(page, max_pixels, &mut |_, _| Ok(()))?;
        let ids = self.prompt_tokens(prompt, h * w);
        let pos = positions(&ids, h, w);
        let x = self.embed(&ids, &img)?;
        // the copy out of the image tokens first (sycl::free does not wait for queued work)
        self.nsd.wait()?;
        drop(img);
        let max = if req.max_tokens == 0 { MAX_TOKENS } else { req.max_tokens } as usize;
        let mut cache = self.decoder.cache(ids.len() + max)?;
        let mut logits = self.decoder.pass(&self.nsd, &mut cache, &x, &pos, &mut |_, _| Ok(()))?;
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let mut next_pos = pos.last().map_or(0, |p| p[0].max(p[1]).max(p[2]) + 1);
        let n = req.ngram.unwrap_or(0) as usize;
        let window = req.ngram_window.unwrap_or(128) as usize;
        let mut seq = ids.clone();
        let mut out_ids: Vec<u32> = Vec::new();
        let mut looped = false;
        let mut shown = 0usize;
        let mut rng = Rng(req.seed ^ 0x706f6372);
        // tables and spotted text are written in the model's own tokens: they come out converted, at the end
        let stream = !(prompt.starts_with("Table") || prompt.starts_with("Chart") || prompt.starts_with("Spotting"));
        let t1 = Instant::now();
        while out_ids.len() < max {
            let ban = banned(&seq, n, window);
            let next = pick(&mut logits, &ban, req.temperature, &mut rng);
            if next == self.eos {
                break;
            }
            seq.push(next);
            out_ids.push(next);
            if let Some(keep) = looping(&out_ids) {
                out_ids.truncate(keep);
                looped = true;
                break;
            }
            // the text so far, up to its last whole character
            let bytes = self.tok.decode_bytes(&out_ids);
            let valid = match std::str::from_utf8(&bytes) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if stream && valid > shown {
                text(std::str::from_utf8(&bytes[shown..valid]).unwrap_or(""));
                shown = valid;
            }
            if out_ids.len() == max {
                break;
            }
            logits = self.decoder.step(&self.nsd, &mut cache, next, next_pos)?;
            next_pos += 1;
        }
        let mut all = String::from_utf8_lossy(&self.tok.decode_bytes(&out_ids)).into_owned();
        if !stream {
            all = format::spots_to_det(&format::otsl_to_html(&all));
        }
        if all.len() > shown {
            text(&all[shown..]);
        }
        Ok((OcrOutput { text: all, prompt: prompt.to_string(), prompt_tokens: ids.len() as u32, tokens: out_ids.len() as u32, prompt_seconds,
                        seconds: t1.elapsed().as_secs_f64() }, looped))
    }
}

/// A runaway plain reading and the table read instead: the reading up to the line the table begins with (its first
/// cell's text; else the whole reading), then the table
fn rescue(plain: &str, table: &str) -> String {
    let first = table.split("<td").nth(1).and_then(|c| c.split_once('>')).map(|(_, c)| c.split("</td>").next().unwrap_or("").trim().to_string())
        .unwrap_or_default();
    let mut head = String::new();
    for line in plain.lines() {
        let l = line.trim();
        if !first.is_empty() && !l.is_empty() && first.starts_with(l) {
            break;
        }
        head.push_str(line);
        head.push('\n');
    }
    format!("{}\n\n{table}\n", head.trim_end())
}

fn add(all: &mut OcrOutput, r: &OcrOutput) {
    all.prompt = r.prompt.clone();
    all.prompt_tokens += r.prompt_tokens;
    all.tokens += r.tokens;
    all.prompt_seconds += r.prompt_seconds;
    all.seconds += r.seconds;
}

impl ImageEngine for PaddleOcrVl {
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
        Err(Error(format!("{ARCH} reads pages; it does not make pictures (image.ocr)")))
    }
    fn reads(&self) -> bool {
        true
    }
    fn ocr(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        self.read(req, text)
    }
    fn ocr_prompts(&self) -> Vec<OcrPrompt> {
        vec![
            OcrPrompt { id: "ocr", prompt: DEFAULT_PROMPT, title: "Text (the model's own)" },
            OcrPrompt { id: "table", prompt: "Table Recognition:", title: "A table" },
            OcrPrompt { id: "formula", prompt: "Formula Recognition:", title: "A formula → LaTeX" },
            OcrPrompt { id: "chart", prompt: "Chart Recognition:", title: "A chart → its table" },
            OcrPrompt { id: "seal", prompt: "Seal Recognition:", title: "A seal or stamp" },
            OcrPrompt { id: "spotting", prompt: "Spotting:", title: "Text with its place" },
        ]
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH}: decoder {} layers, vision {} of {}, on {}", self.decoder.g.layers, self.vision.heads, self.vision.c, self.nsd.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_runaway_is_cut() {
        // text, then "1 \n" over and over: kept up to one copy past the text
        let mut s = vec![5, 6, 7];
        for _ in 0..40 {
            s.extend([1, 2]);
        }
        let keep = looping(&s).unwrap();
        assert!(keep <= 3 + 2 * 2, "{keep}");
        assert_eq!(&s[..3], &[5, 6, 7]);
        assert!(looping(&[1, 2, 3, 4, 5]).is_none());
    }

    #[test]
    fn a_table_rescues_a_runaway() {
        let t = rescue("TRANSCRIPT\nName: Jo\nYEAR\n2008\nB B 1.0\n", "<table><tr><td>YEAR</td><td>2008</td></tr></table>");
        assert_eq!(t, "TRANSCRIPT\nName: Jo\n\n<table><tr><td>YEAR</td><td>2008</td></tr></table>\n");
    }

    #[test]
    fn positions_as_qwen2_vl() {
        // two text tokens, a 2 x 3 image, two text tokens
        let ids = [1, 2, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 3, 4];
        let p = positions(&ids, 2, 3);
        assert_eq!(p[1], [1, 1, 1]);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[7], [2, 3, 4]);
        assert_eq!(p[8], [5, 5, 5]);
        assert_eq!(p[9], [6, 6, 6]);
    }
}
