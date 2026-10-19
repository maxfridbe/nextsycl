//! The vision sidecar (`DeepSeek-V4.1-Flash.vision.gguf`) on the host side: a picture's plan and patches as the
//! reference's image_processor.py makes them, its place in a prompt, and the images a server has handed in.
//!
//! An image is a span of positions in the prompt, `[start] ([image] x w, [newline]) x h [end]`, whose input rows are
//! the sidecar's (its three learned rows and the ViT + aligner's rows in reading order) instead of token embeddings.
//! The span's ids are this engine's own (`IMG_BIT` set, past the vocabulary): the image's key and the position in its
//! span. The same image gives the same ids, so a prompt cache keyed by ids stays right; the rows are made when a pass
//! first reads the span (`Ds41::image_rows`, on the GPU) and kept with the image.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nextsycl_diffusion::Picture;
use nextsycl_gguf::{GType, Gguf, Value};

use crate::{Error, ModelResult as Result};

/// An image span's ids: this bit set, the image's key (21 bits) above the position in its span (10 bits)
pub const IMG_BIT: u32 = 0x8000_0000;
const POS_BITS: u32 = 10;
const KEY_MASK: u32 = (1 << 21) - 1;
/// images kept (their patches, then their rows once made)
const KEEP: usize = 8;

pub fn is_image(id: u32) -> bool {
    id & IMG_BIT != 0
}
pub fn key_of(id: u32) -> u32 {
    (id >> POS_BITS) & KEY_MASK
}
pub fn pos_of(id: u32) -> usize {
    (id & ((1 << POS_BITS) - 1)) as usize
}

/// What a position of a span holds
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slot {
    Start,
    End,
    Newline,
    /// the aligner's row this many into the image's rows
    Row(usize),
}

/// The sidecar's settings (its metadata; image_processor.py's arguments)
#[derive(Clone, Debug)]
pub struct Cfg {
    pub layers: usize,
    pub dim: usize,
    pub heads: usize,
    pub ff: usize,
    pub patch: usize,
    pub theta: f32,
    pub ratio: usize,
    pub max_tokens: usize,
    pub min_pixels: u64,
    /// the placeholder's id in the vocabulary, `<｜deepseek_image｜>`
    pub image_token: u32,
}

/// A picture's sizes: the ViT's patch grid, the LLM's token grid after the aligner, the pixels it is padded to
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub vh: usize,
    pub vw: usize,
    pub lh: usize,
    pub lw: usize,
    pub bh: usize,
    pub bw: usize,
}

impl Plan {
    /// the span's positions: start, h rows of (w images, a newline), end
    pub fn span(&self) -> usize {
        self.lh * (self.lw + 1) + 2
    }
    pub fn slot(&self, i: usize) -> Slot {
        if i == 0 {
            return Slot::Start;
        }
        if i + 1 == self.span() {
            return Slot::End;
        }
        let (r, c) = ((i - 1) / (self.lw + 1), (i - 1) % (self.lw + 1));
        if c == self.lw { Slot::Newline } else { Slot::Row(r * self.lw + c) }
    }
}

/// image_processor.py's llm_grid
fn llm_grid(bh: usize, bw: usize, p: usize, r: usize) -> (usize, usize) {
    ((bh / p).div_ceil(r), (bw / p).div_ceil(r))
}

/// image_processor.py's solve_resize_ratio: the largest aspect-keeping size whose grid fits max tokens
fn solve_resize_ratio(height: f64, width: f64, p: usize, r: usize, max: usize) -> (usize, usize) {
    let ratio = height / width;
    let max_w = ((max as f64 - 2.0) / ratio + 0.25).sqrt() - 0.5;
    let max_h = max_w * ratio;
    let cell = p * r;
    if max_w < 1.0 {
        return ((max - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (max - 3) * cell);
    }
    let beta = (max_w.floor() * cell as f64 / width).min(max_h.floor() * cell as f64 / height);
    (((height * beta / p as f64).floor() as usize) * p, ((width * beta / p as f64).floor() as usize) * p)
}

/// image_processor.py's plan_image_grid (no wide-image ratio: the sidecar sets none)
pub fn plan(width: u32, height: u32, c: &Cfg) -> Plan {
    let p = c.patch;
    let (mut w, mut h) = (width as f64, height as f64);
    let area = width as u64 * height as u64;
    if area > 0 && area < c.min_pixels {
        let k = (c.min_pixels as f64 / area as f64).sqrt();
        w = (w * k).trunc();
        h = (h * k).trunc();
    }
    let (mut bw, mut bh) = (((w / p as f64).ceil() as usize) * p, ((h / p as f64).ceil() as usize) * p);
    let (mut lh, mut lw) = llm_grid(bh, bw, p, c.ratio);
    if lh * (lw + 1) + 2 > c.max_tokens {
        (bh, bw) = solve_resize_ratio(h, w, p, c.ratio, c.max_tokens);
        (lh, lw) = llm_grid(bh, bw, p, c.ratio);
    }
    Plan { vh: bh / p, vw: bw / p, lh, lw, bh, bw }
}

/// float32 rounded to bfloat16 (nearest, ties to even) and back: the reference's pixels are bf16
fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// ImageOps.pad(picture, (bw, bh), color 127): resized (bicubic, as PIL) to fit, centred on gray
fn pad(pic: &Picture, bw: usize, bh: usize) -> Picture {
    let (w, h) = (pic.width as usize, pic.height as usize);
    let (im, dest) = (w as f64 / h as f64, bw as f64 / bh as f64);
    if im == dest {
        return if (w, h) == (bw, bh) { pic.clone() } else { pic.resize_bicubic(bw as u32, bh as u32) };
    }
    let mut out = Picture { width: bw as u32, height: bh as u32, channels: 3, data: vec![127; bw * bh * 3] };
    let (rw, rh, x0, y0) = if im > dest {
        let nh = (h as f64 / w as f64 * bw as f64).round_ties_even() as usize;
        (bw, nh, 0, ((bh as f64 - nh as f64) * 0.5).round_ties_even() as usize)
    } else {
        let nw = (w as f64 / h as f64 * bh as f64).round_ties_even() as usize;
        (nw, bh, ((bw as f64 - nw as f64) * 0.5).round_ties_even() as usize, 0)
    };
    let r = if (rw, rh) == (w, h) { pic.clone() } else { pic.resize_bicubic(rw as u32, rh as u32) };
    for y in 0..rh.min(bh) {
        let (d, s) = (((y0 + y) * bw + x0) * 3, y * rw * 3);
        let n = rw.min(bw - x0) * 3;
        out.data[d..d + n].copy_from_slice(&r.data[s..s + n]);
    }
    out
}

/// A picture's ViT input: [vh * vw][3 * p * p] (a patch's channels, then its rows, then its columns), the pixels
/// (x / 255 - 0.5) / 0.5 rounded to bf16 as the reference's
pub fn patches(pic: &Picture, pl: &Plan, p: usize) -> Vec<f32> {
    let img = pad(&pic.rgb(), pl.bw, pl.bh);
    let px = |y: usize, x: usize, c: usize| bf16_round((img.data[(y * pl.bw + x) * 3 + c] as f32 / 255.0 - 0.5) / 0.5);
    let mut out = Vec::with_capacity(pl.vh * pl.vw * 3 * p * p);
    for i in 0..pl.vh {
        for j in 0..pl.vw {
            for c in 0..3 {
                for y in 0..p {
                    for x in 0..p {
                        out.push(px(i * p + y, j * p + x, c));
                    }
                }
            }
        }
    }
    out
}

/// An image handed in: its plan, its patches until its rows are made, then its rows ([lh * lw][n_embd])
pub struct Img {
    pub hash: u64,
    pub plan: Plan,
    pub patches: Mutex<Option<Arc<Vec<f32>>>>,
    pub rows: Mutex<Option<Arc<Vec<f32>>>>,
}

/// The images held by key, and their keys oldest first
type Held = (HashMap<u32, Arc<Img>>, Vec<u32>);

/// The sidecar: its file, settings, learned rows and routing biases, the images handed in
pub struct Vision {
    pub file: Gguf,
    pub cfg: Cfg,
    /// the placeholder's text (the vocabulary's entry for its id), what a chat template puts in an image's place
    pub marker: String,
    /// the span's start, end and newline rows [n_embd]
    pub start: Vec<f32>,
    pub end: Vec<f32>,
    pub newline: Vec<f32>,
    /// the routers' biases for image positions, by layer
    pub bias_vl: HashMap<u64, Arc<Vec<f32>>>,
    images: Mutex<Held>,
}

/// NS_DS41_VISION: the sidecar's file (default: the *.vision.gguf beside the model; 0: none)
pub fn sidecar(model: &Path) -> Option<PathBuf> {
    match std::env::var("NS_DS41_VISION") {
        Ok(v) if v == "0" => None,
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            let dir = model.parent()?;
            let mut v: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".vision.gguf"))).collect();
            v.sort();
            v.into_iter().next()
        }
    }
}

/// A tensor's values as float32 (F32, F16 or BF16)
pub fn floats(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("the vision sidecar has no {name}")))?;
    let raw = f.read(t).map_err(|e| Error(e.0))?;
    Ok(match t.ty {
        GType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        GType::BF16 => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        GType::F16 => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        ty => return Err(Error(format!("{name}: {} (the sidecar's tensors are F32, F16 or BF16)", ty.name()))),
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1F) as u32, (h & 0x3FF) as u32);
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // subnormal: normalise
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
        (31, _) => (s << 31) | 0x7F80_0000 | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Vision {
    pub fn open(path: &Path, n_layer: u64, n_embd: usize) -> Result<Vision> {
        let file = Gguf::open(path).map_err(|e| Error(format!("the vision sidecar {}: {}", path.display(), e.0)))?;
        if file.arch_meta("sidecar").and_then(Value::as_str) != Some("vision") {
            return Err(Error(format!("{}: not a vision sidecar (deepseek4.sidecar)", path.display())));
        }
        let u = |k: &str| file.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| Error(format!("the vision sidecar: {k} is missing")));
        let cfg = Cfg {
            layers: u("vision.block_count")? as usize,
            dim: u("vision.embedding_length")? as usize,
            heads: u("vision.attention.head_count")? as usize,
            ff: u("vision.feed_forward_length")? as usize,
            patch: u("vision.patch_size")? as usize,
            theta: file.arch_meta("vision.rope.freq_base").and_then(Value::as_f64).unwrap_or(10000.0) as f32,
            ratio: u("vision.downsample_ratio")? as usize,
            max_tokens: u("vision.max_image_tokens")? as usize,
            min_pixels: u("vision.min_pixels")?,
            image_token: u("image_token_id")? as u32,
        };
        if cfg.dim / cfg.heads != 64 || cfg.max_tokens > 1 << POS_BITS {
            return Err(Error(format!("the vision sidecar: heads of {} (64 here) and spans of {} positions (at most {})", cfg.dim / cfg.heads,
                                     cfg.max_tokens, 1 << POS_BITS)));
        }
        let row = |n: &str| -> Result<Vec<f32>> {
            let v = floats(&file, n)?;
            if v.len() != n_embd {
                return Err(Error(format!("{n}: {} values, the model's rows {n_embd}", v.len())));
            }
            Ok(v)
        };
        let (start, end, newline) = (row("v.image_start.weight")?, row("v.image_end.weight")?, row("v.image_newline.weight")?);
        let mut bias_vl = HashMap::new();
        for l in 0..n_layer {
            let name = format!("blk.{l}.exp_probs_b_vl.bias");
            if file.tensor(&name).is_some() {
                bias_vl.insert(l, Arc::new(floats(&file, &name)?));
            }
        }
        Ok(Vision { file, cfg, marker: "<｜deepseek_image｜>".into(), start, end, newline, bias_vl, images: Mutex::new((HashMap::new(), Vec::new())) })
    }

    /// A picture's span of ids (PNG, JPEG or WebP bytes): registered for the passes that read it
    pub fn tokens(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let pic = Picture::decode(bytes).map_err(Error)?;
        if pic.width == 0 || pic.height == 0 {
            return Err(Error("an empty picture".into()));
        }
        let pl = plan(pic.width, pic.height, &self.cfg);
        let hash = fnv(bytes) ^ ((pic.width as u64) << 32 | pic.height as u64);
        let mut g = self.images.lock().unwrap();
        // the key: the hash's low bits, the next free one past an image with another hash
        let mut key = (hash as u32) & KEY_MASK;
        loop {
            match g.0.get(&key) {
                Some(im) if im.hash != hash => key = (key + 1) & KEY_MASK,
                _ => break,
            }
        }
        g.0.entry(key).or_insert_with(|| {
            let px = patches(&pic, &pl, self.cfg.patch);
            Arc::new(Img { hash, plan: pl, patches: Mutex::new(Some(Arc::new(px))), rows: Mutex::new(None) })
        });
        g.1.retain(|k| *k != key);
        g.1.push(key);
        while g.1.len() > KEEP {
            let old = g.1.remove(0);
            g.0.remove(&old);
        }
        Ok((0..pl.span()).map(|i| IMG_BIT | (key << POS_BITS) | i as u32).collect())
    }

    /// The image an id of a span names
    pub fn image(&self, id: u32) -> Result<Arc<Img>> {
        self.images.lock().unwrap().0.get(&key_of(id)).cloned()
            .ok_or_else(|| Error("an image span whose image is no longer held (more than 8 images since): send it again".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        Cfg { layers: 32, dim: 1024, heads: 16, ff: 2816, patch: 14, theta: 10000.0, ratio: 3, max_tokens: 1024, min_pixels: 544 * 544, image_token: 129264 }
    }

    #[test]
    fn plans_as_the_processor() {
        // small: scaled up to the minimum pixels, 544 x 544 -> 39 x 39 patches, 13 x 13 tokens
        let p = plan(100, 100, &cfg());
        assert_eq!((p.vh, p.vw, p.lh, p.lw), (39, 39, 13, 13));
        assert_eq!(p.span(), 13 * 14 + 2);
        // large: shrunk to fit 1024 tokens
        let p = plan(4000, 3000, &cfg());
        assert!(p.span() <= 1024, "{p:?}");
        assert_eq!((p.lh, p.lw), (p.vh.div_ceil(3), p.vw.div_ceil(3)));
        // the slots in reading order
        let p = Plan { vh: 6, vw: 6, lh: 2, lw: 2, bh: 84, bw: 84 };
        let s: Vec<Slot> = (0..p.span()).map(|i| p.slot(i)).collect();
        assert_eq!(s, vec![Slot::Start, Slot::Row(0), Slot::Row(1), Slot::Newline, Slot::Row(2), Slot::Row(3), Slot::Newline, Slot::End]);
    }

    #[test]
    fn ids_carry_key_and_place() {
        let id = IMG_BIT | (0x1ABCD << POS_BITS) | 517;
        assert!(is_image(id));
        assert_eq!((key_of(id), pos_of(id)), (0x1ABCD, 517));
        assert!(!is_image(129264));
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
    }
}
