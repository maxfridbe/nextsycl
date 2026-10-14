//! Unlimited-OCR (baidu/Unlimited-OCR, the DeepSeek-OCR architecture): pages in, their text out - as Markdown and HTML
//! tables, with boxes when asked (`<|grounding|>`). From two GGUFs: the decoder (`deepseek2-ocr`, any of its quants)
//! and the projector (`mmproj-*`, F16: SAM-B, CLIP-L and the linear into the decoder). Checked stage by stage against
//! llama.cpp (`check`); the reading follows the reference's `infer` / `infer_multi` (modeling_unlimitedocr.py):
//!
//! ```text
//!   [BOS] <image tokens> <prompt>   (the plain template: "<image>" + the instruction)
//!   image tokens: the tiles' (one picture, a side over 640), each page's global view, the view separator - prep.rs
//!   then greedy (or sampled) tokens, no 35-gram repeated within the last 128 tokens (1024 for several pages),
//!   until the end of sentence
//! ```

pub mod check;
pub mod decoder;
pub mod layout;
mod ffi;
pub mod prep;
pub mod vision;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_core::{DevBuf, Gpu};
use nextsycl_diffusion::kernels::Nsd;
use nextsycl_gguf::Gguf;
use nextsycl_image::{Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, OcrOutput, OcrRequest, Picture, Result, Sampler,
                     Schedule, Step};
use nextsycl_tok::Tokenizer;
use vision::CopyF;

/// The architecture (the decoder GGUF's general.architecture; a registry entry may say either)
pub const ARCH: &str = "unlimited-ocr";
pub const ARCHS: &[&str] = &[ARCH, "deepseek2-ocr"];
/// Its files: the decoder and the projector
pub const ROLES: &[&str] = &["model", "mmproj"];
/// The reference's image token and the instruction it is trained on
pub const IMAGE_TOKEN: u32 = 128815;
pub const DEFAULT_PROMPT: &str = "document parsing.";
/// The instruction for several pages (the reference's infer_multi): the model reads every page under it, each after a
/// `<PAGE>` mark - under a single page's instruction it reads the first page (or nothing) and stops
pub const MULTI_PROMPT: &str = "Multi page parsing.";
/// What a picture inside a page is read with (its text alone)
pub const FIGURE_PROMPT: &str = "Free OCR.";
pub const NGRAM: u32 = 35;
pub const NGRAM_WINDOW: u32 = 128;
pub const NGRAM_WINDOW_PAGES: u32 = 1024;
pub const MAX_TOKENS: u32 = 8192;
/// the global view's patch rows (1024 / 16 / 4) and a tile's (640 / 16 / 4)
const GQ: usize = 16;
const TQ: usize = 10;

pub fn kind() -> ImageKind {
    ImageKind { archs: ARCHS, name: "Unlimited-OCR (pages to Markdown: SAM-B + CLIP-L vision, a DeepSeek-V2 MoE decoder)", roles: ROLES, load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(UnlimitedOcr::load(files, gpu, log)?))
}

pub struct UnlimitedOcr {
    pub nsd: Nsd,
    pub vision: vision::Vision,
    pub decoder: decoder::Decoder,
    pub tok: Tokenizer,
    eos: u32,
    load_s: f64,
    /// one reading at a time
    work: Mutex<()>,
}

/// Python's no-repeat rule (SlidingWindowNoRepeatNgramProcessor): the tokens that would complete an n-gram already in
/// the window's last `window` tokens of `seq`
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

/// A reading of a crop with its boxes moved to the page: [x1, y1, x2, y2] in thousandths of the crop (at `x0`, `y0`,
/// `cw` x `ch` pixels of a `w` x `h` page) to thousandths of the page
fn to_page(text: &str, (x0, y0, cw, ch): (u32, u32, u32, u32), (w, h): (u32, u32)) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("<|det|>") {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let Some(end) = tail.find("<|/det|>") else {
            out.push_str(tail);
            return out;
        };
        let inner = &tail["<|det|>".len()..end];
        let moved = inner.split_once('[').and_then(|(label, b)| {
            let n: Vec<f64> = b.trim_end_matches(']').split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let [a, b, c, d] = n[..] else { return None };
            let px = |v: f64, o: u32, s: u32, side: u32| ((o as f64 + v / 999.0 * s as f64) / side as f64 * 999.0).round().clamp(0.0, 999.0) as u32;
            Some(format!("{label}[{}, {}, {}, {}]", px(a, x0, cw, w), px(b, y0, ch, h), px(c, x0, cw, w), px(d, y0, ch, h)))
        });
        out.push_str("<|det|>");
        out.push_str(moved.as_deref().unwrap_or(inner));
        out.push_str("<|/det|>");
        rest = &tail[end + "<|/det|>".len()..];
    }
    out.push_str(rest);
    out
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

impl UnlimitedOcr {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<UnlimitedOcr> {
        let t0 = Instant::now();
        let path = |r: &str| files.get(r).ok_or_else(|| Error(format!("{ARCH}: no {r} file")));
        let fm = Gguf::open(path("model")?).map_err(|e| Error(e.0))?;
        if fm.architecture() != "deepseek2-ocr" {
            return Err(Error(format!("{}: a {} model, not deepseek2-ocr", path("model")?.display(), fm.architecture())));
        }
        let fp = Gguf::open(path("mmproj")?).map_err(|e| Error(e.0))?;
        let nsd = Nsd::new(gpu)?;
        let tok = Tokenizer::from_gguf(&fm).map_err(Error)?;
        let eos = fm.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let vision = vision::Vision::load(&fp, gpu, log)?;
        let decoder = decoder::Decoder::load(&fm, gpu, &nsd, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("unlimitedocr: loaded on {} in {load_s:.1} s", gpu.name));
        Ok(UnlimitedOcr { nsd, vision, decoder, tok, eos, load_s, work: Mutex::new(()) })
    }

    /// The prompt's tokens: [BOS], the text before the image, `n_image` image tokens, the text after
    pub fn prompt_tokens(&self, prompt: &str, n_image: usize) -> Vec<u32> {
        let p = if prompt.contains("<image>") { prompt.to_string() } else { format!("<image>{prompt}") };
        let (before, after) = p.split_once("<image>").unwrap_or(("", &p));
        let mut ids = vec![0u32];
        ids.extend(self.tok.encode(before));
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, n_image));
        ids.extend(self.tok.encode(after));
        ids
    }

    /// The pages' image tokens [n][1280] on the GPU, in the reference's order (prep.rs); `tap` sees the vision stages
    pub fn image_tokens(&self, seen: &prep::Seen, tap: &mut dyn FnMut(&str, &DevBuf) -> nextsycl_core::Result<()>) -> Result<(DevBuf, usize)> {
        let e = vision::OUT;
        let (gw, gh) = (seen.grid.0 as usize, seen.grid.1 as usize);
        let n_tiles = if seen.tiles.is_empty() { 0 } else { (TQ * gw + 1) * TQ * gh };
        let n = n_tiles + seen.globals.len() * ((GQ + 1) * GQ + 1);
        let gpu = &self.nsd.gpu;
        let out = DevBuf::f32(gpu, n * e)?;
        let nl = DevBuf::from_f32(gpu, &self.vision.newline)?;
        let sep = DevBuf::from_f32(gpu, &self.vision.separator)?;
        let mut row = 0usize;
        if !seen.tiles.is_empty() {
            // the tiles a few at a time (memory), then woven: a tile row's patch rows side by side, a newline each
            let mut feats = Vec::new();
            for chunk in seen.tiles.chunks(8) {
                let planes: Vec<f32> = chunk.iter().flat_map(|v| v.planes.iter().copied()).collect();
                feats.push(self.vision.see(&self.nsd, &planes, chunk.len(), prep::TILE as usize, tap)?);
            }
            let tile = |i: usize| (&feats[i / 8], (i % 8) * TQ * TQ);
            for r in 0..gh {
                for y in 0..TQ {
                    for c in 0..gw {
                        let (buf, base) = tile(r * gw + c);
                        out.copy_within_f(row * e, buf, (base + y * TQ) * e, TQ * e)?;
                        row += TQ;
                    }
                    out.copy_within_f(row * e, &nl, 0, e)?;
                    row += 1;
                }
            }
        }
        for v in &seen.globals {
            let f = self.vision.see(&self.nsd, &v.planes, 1, prep::BASE as usize, tap)?;
            for y in 0..GQ {
                out.copy_within_f(row * e, &f, y * GQ * e, GQ * e)?;
                row += GQ;
                out.copy_within_f(row * e, &nl, 0, e)?;
                row += 1;
            }
            out.copy_within_f(row * e, &sep, 0, e)?;
            row += 1;
        }
        debug_assert_eq!(row, n);
        // the newline, separator and features go when this returns: the copies from them first
        self.nsd.wait()?;
        Ok((out, n))
    }

    /// The prompt's embeddings: the text's rows, the image tokens' rows from `img`
    pub fn embed(&self, ids: &[u32], img: &DevBuf) -> Result<DevBuf> {
        let e = vision::OUT;
        let text: Vec<u32> = ids.iter().map(|&t| if t == IMAGE_TOKEN { 0 } else { t }).collect();
        let x = self.decoder.embed(&self.nsd, &text)?;
        let mut k = 0usize;
        let mut i = 0usize;
        while i < ids.len() {
            if ids[i] == IMAGE_TOKEN {
                let j = (i..ids.len()).find(|&j| ids[j] != IMAGE_TOKEN).unwrap_or(ids.len());
                x.copy_within_f(i * e, img, k * e, (j - i) * e)?;
                k += j - i;
                i = j;
            } else {
                i += 1;
            }
        }
        Ok(x)
    }

    /// A reading: see the pages, read the prompt, write until the end (or max tokens), each piece to `text`
    /// A reading (one at a time); then, `figures` on, each picture the model only marked (an empty `image` block of a
    /// size that can hold text) read on its own and its text put in the block
    pub fn read(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let _w = self.work.lock().unwrap();
        if req.pages.len() > 1 && req.each_page.unwrap_or(true) {
            // page by page, each as a reading of its own (tiles, figures), joined with <PAGE> marks
            let mut all = OcrOutput::default();
            for page in &req.pages {
                text("<PAGE>");
                let one = OcrRequest { pages: vec![page.clone()], each_page: Some(false), extra: Default::default(), ..req.clone() };
                let mut r = self.read_page(&one, text)?;
                if one.figures.unwrap_or(true) {
                    self.read_figures(&one, &mut r)?;
                }
                text("\n");
                all.text.push_str(&format!("<PAGE>{}\n", r.text));
                all.prompt = r.prompt;
                all.prompt_tokens += r.prompt_tokens;
                all.tokens += r.tokens;
                all.prompt_seconds += r.prompt_seconds;
                all.seconds += r.seconds;
            }
            return Ok(all);
        }
        let mut out = if req.pages.len() == 1 { self.read_page(req, text)? } else { self.read_one(req, text)? };
        if req.figures.unwrap_or(true) {
            self.read_figures(req, &mut out)?;
        }
        Ok(out)
    }

    /// One page: its columns found (layout.rs) and each region read on its own, their boxes moved to the page - or,
    /// columns off or none found, the page read whole
    fn read_page(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let page = &req.pages[0];
        let regions = if req.columns.unwrap_or(true) { layout::regions(page) } else { Vec::new() };
        if regions.len() <= 1 {
            return self.read_one(req, text);
        }
        let mut all = OcrOutput::default();
        for (x, y, w, h) in regions {
            let c = page.channels as usize;
            let mut crop = Picture::new(w, h, page.channels);
            for yy in 0..h as usize {
                let src = ((y as usize + yy) * page.width as usize + x as usize) * c;
                crop.data[yy * w as usize * c..(yy + 1) * w as usize * c].copy_from_slice(&page.data[src..src + w as usize * c]);
            }
            let one = OcrRequest { pages: vec![crop], columns: Some(false), figures: Some(false), each_page: Some(false), extra: Default::default(), ..req.clone() };
            let r = self.read_one(&one, text)?;
            text("\n");
            let t = r.text.trim();
            if !t.is_empty() {
                all.text.push_str(&to_page(t, (x, y, w, h), (page.width, page.height)));
                all.text.push('\n');
            }
            all.prompt = r.prompt;
            all.prompt_tokens += r.prompt_tokens;
            all.tokens += r.tokens;
            all.prompt_seconds += r.prompt_seconds;
            all.seconds += r.seconds;
        }
        Ok(all)
    }

    /// The empty picture blocks of a reading, cropped from their pages and read with "Free OCR."
    fn read_figures(&self, req: &OcrRequest, out: &mut OcrOutput) -> Result<()> {
        let t0 = Instant::now();
        // the pages' texts: one, or each after a <PAGE> mark
        let parts: Vec<&str> = if out.text.contains("<PAGE>") { out.text.split("<PAGE>").skip(1).collect() } else { vec![out.text.as_str()] };
        let mut done: Vec<String> = Vec::new();
        let mut extra = 0u32;
        let mut any = false;
        for (pi, part) in parts.iter().enumerate() {
            let Some(page) = req.pages.get(pi) else {
                done.push(part.to_string());
                continue;
            };
            let mut s = String::with_capacity(part.len());
            let mut rest: &str = part;
            while let Some(at) = rest.find("<|det|>image [") {
                s.push_str(&rest[..at]);
                let tail = &rest[at..];
                let end = tail.find("<|/det|>").map(|e| e + "<|/det|>".len()).unwrap_or(tail.len());
                let tag = &tail[..end];
                s.push_str(tag);
                rest = &tail[end..];
                // only a block with nothing written after it (up to the line's end)
                let line_end = rest.find('\n').unwrap_or(rest.len());
                if !rest[..line_end].trim().is_empty() {
                    continue;
                }
                let nums: Vec<u32> = tag.trim_start_matches("<|det|>image [").trim_end_matches("]<|/det|>").split(',')
                    .filter_map(|v| v.trim().parse().ok()).collect();
                let [x1, y1, x2, y2] = nums[..] else { continue };
                // the reference's own crop (x / 999 of the page), a margin of 1% around
                let (w, h) = (page.width as f64, page.height as f64);
                let px = |v: u32, side: f64, d: f64| ((v as f64 / 999.0 + d) * side).clamp(0.0, side) as u32;
                let (cx1, cy1, cx2, cy2) = (px(x1, w, -0.01), px(y1, h, -0.01), px(x2, w, 0.01), px(y2, h, 0.01));
                let (cw, ch) = (cx2.saturating_sub(cx1), cy2.saturating_sub(cy1));
                // a picture big enough to hold text: 96 pixels a side and 2% of the page
                if cw < 96 || ch < 96 || (cw as f64 * ch as f64) < 0.02 * w * h {
                    continue;
                }
                let c = page.channels as usize;
                let mut crop = Picture::new(cw, ch, page.channels);
                for y in 0..ch as usize {
                    let src = ((cy1 as usize + y) * page.width as usize + cx1 as usize) * c;
                    crop.data[y * cw as usize * c..(y + 1) * cw as usize * c].copy_from_slice(&page.data[src..src + cw as usize * c]);
                }
                let sub = OcrRequest { pages: vec![crop], prompt: FIGURE_PROMPT.into(), tiles: Some(true), max_tokens: req.max_tokens.clamp(0, 4096),
                                       ngram: req.ngram, ngram_window: None, temperature: 0.0, seed: req.seed, figures: Some(false), each_page: Some(false),
                                       columns: Some(false), extra: Default::default() };
                let r = self.read_one(&sub, &mut |_| {})?;
                extra += r.tokens;
                let t = r.text.trim();
                if !t.is_empty() {
                    any = true;
                    // the picture's own blocks after it, their boxes moved from the crop's thousandths to the page's
                    s.push('\n');
                    s.push_str(&to_page(t, (cx1, cy1, cw, ch), (page.width, page.height)));
                }
            }
            s.push_str(rest);
            done.push(s);
        }
        if any {
            out.text = if out.text.contains("<PAGE>") { done.iter().map(|p| format!("<PAGE>{p}")).collect() } else { done.concat() };
        }
        out.tokens += extra;
        out.seconds += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// One pass: see the pages, read the prompt, write
    fn read_one(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        let t0 = Instant::now();
        if req.pages.is_empty() {
            return Err(Error("no pages to read".into()));
        }
        let several = req.pages.len() > 1;
        let seen = prep::see(&req.pages, req.tiles.unwrap_or(true));
        let (img, n_img) = self.image_tokens(&seen, &mut |_, _| Ok(()))?;
        let prompt = if several && !req.prompt.contains("Multi page") {
            MULTI_PROMPT
        } else if req.prompt.trim().is_empty() {
            DEFAULT_PROMPT
        } else {
            req.prompt.as_str()
        };
        let ids = self.prompt_tokens(prompt, n_img);
        let x = self.embed(&ids, &img)?;
        // the copies out of the image tokens first (sycl::free does not wait for queued work)
        self.nsd.wait()?;
        drop(img);
        let mut cache = self.decoder.cache(ids.len())?;
        let mut logits = self.decoder.pass(&self.nsd, &mut cache, &x, ids.len(), &mut |_, _| Ok(()))?;
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let n = req.ngram.unwrap_or(NGRAM) as usize;
        let window = req.ngram_window.unwrap_or(if several { NGRAM_WINDOW_PAGES } else { NGRAM_WINDOW }) as usize;
        let max = if req.max_tokens == 0 { MAX_TOKENS } else { req.max_tokens } as usize;
        let mut seq = ids.clone();
        let mut out_ids: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        let mut rng = Rng(req.seed ^ 0x6f63725f);
        let t1 = Instant::now();
        while out_ids.len() < max {
            let ban = banned(&seq, n, window);
            let next = pick(&mut logits, &ban, req.temperature, &mut rng);
            if next == self.eos {
                break;
            }
            seq.push(next);
            out_ids.push(next);
            // the text so far, up to its last whole character
            let bytes = self.tok.decode_bytes(&out_ids);
            let valid = match std::str::from_utf8(&bytes) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > shown {
                text(std::str::from_utf8(&bytes[shown..valid]).unwrap_or(""));
                shown = valid;
            }
            if out_ids.len() == max {
                break;
            }
            cache.written += 1;
            let xe = self.decoder.embed(&self.nsd, &[next])?;
            logits = self.decoder.pass(&self.nsd, &mut cache, &xe, 1, &mut |_, _| Ok(()))?;
        }
        let all = String::from_utf8_lossy(&self.tok.decode_bytes(&out_ids)).into_owned();
        if all.len() > shown {
            text(&all[shown..]);
        }
        Ok(OcrOutput { text: all, prompt: prompt.to_string(), prompt_tokens: ids.len() as u32, tokens: out_ids.len() as u32, prompt_seconds, seconds: t1.elapsed().as_secs_f64() })
    }
}

impl ImageEngine for UnlimitedOcr {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: prep::BASE, height: prep::BASE, steps: 0, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 1.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step)) -> Result<Vec<Picture>> {
        Err(Error(format!("{ARCH} reads pages; it does not make pictures (image.ocr)")))
    }
    fn reads(&self) -> bool {
        true
    }
    fn ocr(&self, req: &OcrRequest, text: &mut dyn FnMut(&str)) -> Result<OcrOutput> {
        self.read(req, text)
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH}: decoder {} layers, {:.1} GiB of weights read, on {}", self.decoder.g.layers, self.decoder.loaded_bytes as f64 / (1u64 << 30) as f64,
                     self.nsd.gpu.name)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_boxes_move_to_the_page() {
        // a crop of the right half of a 1000 x 1000 page: its whole width is the page's 500..1000
        let t = to_page("<|det|>text [0, 0, 999, 999]<|/det|>hello", (500, 0, 500, 1000), (1000, 1000));
        assert_eq!(t, "<|det|>text [500, 0, 999, 999]<|/det|>hello");
    }

    #[test]
    fn ngram_bans_the_repeat() {
        // 3-grams: "1 2 3" seen; the tail "1 2" would repeat it with 3
        let seq = [1, 2, 3, 4, 1, 2];
        assert_eq!(banned(&seq, 3, 128), vec![3]);
        assert!(banned(&seq, 3, 2).is_empty());
        assert!(banned(&seq, 0, 128).is_empty());
    }
}
