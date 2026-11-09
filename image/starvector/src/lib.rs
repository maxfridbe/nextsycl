//! StarVector-1B im2svg (ServiceNow, Apache-2.0 on BigCode's OpenRAIL-M StarCoder): a picture written as SVG code.
//! The picture is flattened on white and seen at 224 x 224 by a CLIP ViT-L/14; an adapter makes its 257 rows the
//! first embeddings of StarCoder-1B, which writes the SVG after the prompt "<svg" until "</svg>" or the length limit -
//! as the starvector package's generate_im2svg: 2 beams sampled, top_p 0.9 (the model's own settings; greedy, more
//! beams or plain sampling on request). Rust and SYCL: the shared kernels and the upscaler's sigmoid; the picture's
//! preprocessing (PIL's arithmetic) and the decoding (transformers' beam search) on the host.
//!
//! An SVG the length limit stopped short is finished here (`close`): cut back to its last whole piece, every open element
//! closed, "</svg>" last, checked to parse as XML - the answer says it was `truncated`.

pub mod check;
pub mod close;
pub mod ffi;
pub mod gen;
pub mod model;
pub mod pre;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextsycl_image::{Defaults, Error, Gpu, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule, Step,
                     VectorizeOutput, VectorizeRequest};

pub const ARCH: &str = "starvector";
/// The longest sequence there is: StarCoder's 8192 positions (the picture's 257 rows, the prompt and the SVG's tokens);
/// a request's max_length is held to it
pub const MAX_LENGTH_CAP: u32 = model::POSITIONS as u32;
/// The shortest default length (a request that names no max_length gets at least this)
pub const DEFAULT_MIN_LENGTH: u32 = 4096;

/// The length a request that names none is held to, from how much drawing the picture holds (`pre::edges`): the
/// model wrote 913 tokens for a flat logo of 4,677 edge pixels (0.2 a pixel) and ends on its own well before any limit
/// - so an allowance of a token an edge pixel (five times that) on top of the prefix, never under DEFAULT_MIN_LENGTH,
/// never over the cap. A picture that fills the cap (13,000 edge pixels: a busy burst of wedges) gets all of it; the
/// time is bounded by the cap (about 1 minute at 120 tokens/s).
pub fn default_max_length(edges: usize) -> u32 {
    let prefix = (model::QUERY + 6) as u64;
    (prefix + edges as u64).clamp(DEFAULT_MIN_LENGTH as u64, MAX_LENGTH_CAP as u64) as u32
}


/// tokens between two decodes of the text written so far (a piece for the caller's stream)
const PIECE: usize = 64;
/// beams at most (the logits buffer's rows)
pub const MAX_BEAMS: u32 = 8;

pub fn kind() -> ImageKind {
    ImageKind { archs: &[ARCH], name: "StarVector-1B im2svg: a picture as SVG (CLIP ViT-L/14 + StarCoder-1B, Rust + SYCL)", roles: &["model", "model_2", "tokenizer"], load, options: &[] }
}

fn load(files: &ModelFiles, gpus: &[Arc<Gpu>], _o: &LoadOptions, log: &mut dyn FnMut(String)) -> Result<Box<dyn ImageEngine>> {
    let gpu = gpus.first().ok_or_else(|| Error("no GPU".into()))?;
    Ok(Box::new(StarVector::load(files, gpu, log)?))
}

fn role(files: &ModelFiles, r: &str) -> Result<PathBuf> {
    files.get(r).cloned().ok_or_else(|| Error(format!("{ARCH}: no {r} file")))
}

/// A checkpoint's directory as downloaded (its two shards and tokenizer.json): its files by role
pub fn files_in(dir: &Path) -> Result<ModelFiles> {
    let mut f = ModelFiles::new();
    for (r, name) in [("model", "model-00001-of-00002.safetensors"), ("model_2", "model-00002-of-00002.safetensors"), ("tokenizer", "tokenizer.json")] {
        let p = dir.join(name);
        if !p.is_file() {
            return Err(Error(format!("{}: no {name} (a StarVector checkpoint's directory)", dir.display())));
        }
        f.insert(r.into(), p);
    }
    Ok(f)
}

pub struct StarVector {
    pub model: Mutex<model::Model>,
    pub vocab: text::Vocab,
    pub gpu: Arc<Gpu>,
    load_s: f64,
}

/// The decoder as gen::decode drives it: the prefix's logits, then a step a token
struct Run<'a> {
    m: &'a mut model::Model,
    first: Option<Vec<f32>>,
    len: usize,
}

impl gen::Lm for Run<'_> {
    fn vocab(&self) -> usize {
        self.m.vocab
    }
    fn first(&mut self) -> Result<Vec<f32>> {
        self.first.take().ok_or_else(|| Error("the prefix's logits were taken".into()))
    }
    fn step(&mut self, from: &[usize], tokens: &[u32]) -> Result<Vec<f32>> {
        let out = self.m.step(from, tokens, self.len)?;
        self.len += 1;
        Ok(out)
    }
}

impl StarVector {
    pub fn load(files: &ModelFiles, gpu: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<StarVector> {
        let t0 = Instant::now();
        let vocab = text::Vocab::open(&role(files, "tokenizer")?)?;
        let shards = vec![role(files, "model")?, role(files, "model_2")?];
        let model = model::Model::load(&shards, gpu, model::QUERY + vocab.prompt.len() + 8, log)?;
        let load_s = t0.elapsed().as_secs_f64();
        log(format!("starvector: loaded in {load_s:.1} s ({:.2} GiB of weights)", model::Model::weight_bytes(model.vocab) as f64 / (1u64 << 30) as f64));
        Ok(StarVector { model: Mutex::new(model), vocab, gpu: gpu.clone(), load_s })
    }

    /// The prefix's embeddings: the adapter's 257 rows and the prompt's tokens
    fn prefix(&self, m: &model::Model, adapter: &[f32]) -> Vec<f32> {
        let mut x = adapter.to_vec();
        for &t in &self.vocab.prompt {
            x.extend(m.token_row(t));
        }
        x
    }

    /// The settings a request asks for, the model's own where it says nothing
    pub fn settings(&self, req: &VectorizeRequest, edges: usize) -> Result<gen::Settings> {
        let beams = req.beams.unwrap_or(2);
        if !(1..=MAX_BEAMS).contains(&beams) {
            return Err(Error(format!("beams {beams}: 1 to {MAX_BEAMS}")));
        }
        let prefix = model::QUERY + self.vocab.prompt.len();
        let max_length = if req.max_length == 0 { default_max_length(edges) } else { req.max_length } as usize;
        // max_length counts the prefix (transformers: inputs_embeds' length comes off it); StarCoder has 8192 positions
        let max_new = max_length.min(model::POSITIONS).saturating_sub(prefix);
        if max_new == 0 {
            return Err(Error(format!("max_length {max_length}: at least {} (the picture's {} rows and the prompt come first)", prefix + 1, model::QUERY)));
        }
        let top_p = req.top_p.unwrap_or(0.9);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(Error(format!("top_p {top_p}: above 0, at most 1")));
        }
        Ok(gen::Settings {
            beams: beams as usize,
            sample: req.sample.unwrap_or(true),
            top_p,
            temperature: req.temperature.unwrap_or(1.0),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            max_new,
            eos: text::EOS,
            stop: self.vocab.stop.clone(),
            seed: req.seed,
        })
    }
}

impl ImageEngine for StarVector {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults { width: pre::SIZE as u32, height: pre::SIZE as u32, steps: 0, cfg: 1.0, sampler: Sampler::Euler, schedule: Schedule::Shift, shift: 0.0 }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn generate(&self, _req: &ImageRequest, _progress: &mut dyn FnMut(Step) -> Result<()>) -> Result<Vec<Picture>> {
        Err(Error(format!("{ARCH} writes pictures as SVG (image.vectorize); it does not make them")))
    }
    fn report(&self) -> Vec<String> {
        vec![format!("{ARCH} on {}", self.gpu.name)]
    }
    fn vectorizes(&self) -> bool {
        true
    }
    fn vectorize(&self, req: &VectorizeRequest, text: &mut dyn FnMut(&str) -> Result<()>) -> Result<VectorizeOutput> {
        let t0 = Instant::now();
        let pixels = pre::pixels(&req.picture);
        let s = self.settings(req, pre::edges(&pixels))?;
        let mut m = self.model.lock().unwrap_or_else(|p| p.into_inner());
        let (_enc, adapter) = m.encode(&pixels)?;
        let prefix = self.prefix(&m, &adapter);
        let first = m.prefill(&prefix, s.beams)?;
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let len = prefix.len() / model::D;
        let mut run = Run { m: &mut m, first: Some(first), len };
        let mut shown = 0usize;
        let vocab = &self.vocab;
        let out = gen::decode(&mut run, &s, &mut |lead| {
            // called every token (the caller's cancel check rides on it); the text is decoded every PIECE tokens - the leading
            // beam's text as it grows (with beams it may change course: what is new of it, then) - as decoding the whole
            // text a token made the host work grow with the square of the length
            if lead.len() % PIECE != 0 {
                return text("");
            }
            let txt = vocab.decode(&[vocab.prompt.as_slice(), lead].concat());
            let piece = if txt.len() >= shown && txt.is_char_boundary(shown) { txt[shown..].to_string() } else { String::new() };
            shown = txt.len();
            text(&piece)
        })?;
        let wrote = vocab.decode(&[vocab.prompt.as_slice(), out.tokens.as_slice()].concat());
        // what the limit cut short is closed; what the model closed itself is checked and kept
        let fin = close::finish(&wrote);
        Ok(VectorizeOutput {
            svg: fin.svg,
            closed: fin.valid,
            truncated: !out.stopped,
            repaired: fin.repaired,
            max_length: (s.max_new + model::QUERY + vocab.prompt.len()) as u32,
            tokens: out.tokens.len() as u32,
            prompt_tokens: len as u32,
            prompt_seconds,
            seconds: t0.elapsed().as_secs_f64() - prompt_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_length_follows_the_drawing_between_the_floor_and_the_cap() {
        assert_eq!(default_max_length(0), DEFAULT_MIN_LENGTH);
        assert_eq!(default_max_length(100), DEFAULT_MIN_LENGTH, "a plain icon gets the floor");
        let logo = default_max_length(4677);
        assert!(logo > DEFAULT_MIN_LENGTH && logo < MAX_LENGTH_CAP, "{logo}");
        assert!(default_max_length(5000) <= default_max_length(6000), "more drawing, more room");
        assert_eq!(default_max_length(100_000), MAX_LENGTH_CAP);
        assert_eq!(MAX_LENGTH_CAP, 8192);
    }

    #[test]
    fn edges_count_the_pixel_pairs_that_differ() {
        let flat = vec![0.5f32; 3 * pre::SIZE * pre::SIZE];
        assert_eq!(pre::edges(&flat), 0);
        // the left half black, the right half white: one vertical edge, SIZE pairs
        let mut split = flat.clone();
        for c in 0..3 {
            for y in 0..pre::SIZE {
                for x in 0..pre::SIZE {
                    split[c * pre::SIZE * pre::SIZE + y * pre::SIZE + x] = if x < pre::SIZE / 2 { -2.0 } else { 2.0 };
                }
            }
        }
        assert_eq!(pre::edges(&split), pre::SIZE);
        // a mild difference (under 24 grey levels) is not an edge
        let mut soft = flat;
        soft[0] += 0.05;
        assert_eq!(pre::edges(&soft), 0);
    }
}
