//! The vision tower (the model's `mmproj` GGUF: `clip.projector_type qwen3vl_merger`, Qwen3-VL's ViT) on the host
//! side: a picture's size, its patches and positions as transformers' Qwen3-VL processor and model make them, its
//! place in a prompt, and the pictures a server has handed in. The tower itself runs in the engine
//! (`Qwen35::image_rows`):
//!
//! ```text
//!   patches 16 x 16 (two frames of the same picture: the two temporal kernels summed), a 2 x 2 unit's four in a row
//!   + the learned positions (a 48 x 48 grid, bilinear to the picture's, corners aligned)
//!   27 blocks of 1152: layer norm -> q, k, v (bias) -> 2D rotary positions (heads of 72: 18 frequencies of the
//!     patch's row, 18 of its column, NEOX halves; base 10000) -> attention over the whole picture (16 heads) -> out
//!     (bias) -> residual; layer norm -> GELU MLP of 4304 (tanh form, biases) -> residual            (eps 1e-6)
//!   merger: layer norm -> a unit's 4 patches into one row of 4608 -> linear -> GELU (erf) -> linear to 5120
//! ```
//!
//! A picture is a span of the prompt between `<|vision_start|>` and `<|vision_end|>`, a position a merged unit in
//! reading order, whose input rows are the tower's. Its rotary positions are Qwen3-VL's: the span's (t, row, column)
//! from its first position, and the text after it goes on from there plus the longer side (`Plan::advance`), not the
//! span's length. The span's ids are this engine's own (`IMG_BIT` set, past the vocabulary): the picture's key and
//! the position in its span. The same picture gives the same ids, so a prompt cache keyed by ids stays right.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nextsycl_diffusion::Picture;
use nextsycl_gguf::{GType, Gguf, Value};
use nextsycl_llm::{Error, Result};

/// An image span's ids: this bit set, the picture's key (18 bits) above the position in its span (13 bits)
pub const IMG_BIT: u32 = 0x8000_0000;
const POS_BITS: u32 = 13;
const KEY_MASK: u32 = (1 << 18) - 1;
/// pictures kept (their patches, then their rows once made)
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

/// The tower's settings (the mmproj file's metadata; the processor's limits)
#[derive(Clone, Debug)]
pub struct Cfg {
    pub layers: usize,
    pub dim: usize,
    pub heads: usize,
    pub ff: usize,
    pub patch: usize,
    pub merge: usize,
    /// the rows it makes (the language model's width)
    pub out: usize,
    pub eps: f32,
    /// the learned positions' grid side
    pub grid: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub min_pixels: u64,
    pub max_pixels: u64,
}

/// A picture's sizes: the pixels it is resized to, its patch grid, its span's grid (2 x 2 patches a position)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub w: usize,
    pub h: usize,
    pub gh: usize,
    pub gw: usize,
    pub lh: usize,
    pub lw: usize,
}

impl Plan {
    /// the span's positions (merged units)
    pub fn span(&self) -> usize {
        self.lh * self.lw
    }
    /// how far the text's rotary position moves past the span (Qwen3-VL's get_rope_index: the longer side)
    pub fn advance(&self) -> usize {
        self.lh.max(self.lw)
    }
    /// the span's position i as (row, column)
    pub fn rc(&self, i: usize) -> (usize, usize) {
        (i / self.lw, i % self.lw)
    }
    /// patches
    pub fn n(&self) -> usize {
        self.gh * self.gw
    }
}

/// transformers' smart_resize: sides multiples of `f`, near the picture's, its area within [min, max] pixels
pub fn smart_resize(width: u32, height: u32, f: usize, min: u64, max: u64) -> (usize, usize) {
    let (w, h, fl) = (width as f64, height as f64, f as f64);
    let mut hb = (f as f64).max((h / fl).round_ties_even() * fl);
    let mut wb = (f as f64).max((w / fl).round_ties_even() * fl);
    if hb * wb > max as f64 {
        let beta = (h * w / max as f64).sqrt();
        hb = fl.max((h / beta / fl).floor() * fl);
        wb = fl.max((w / beta / fl).floor() * fl);
    } else if hb * wb < min as f64 {
        let beta = (min as f64 / (h * w)).sqrt();
        hb = (h * beta / fl).ceil() * fl;
        wb = (w * beta / fl).ceil() * fl;
    }
    (wb as usize, hb as usize)
}

pub fn plan(width: u32, height: u32, c: &Cfg) -> Plan {
    let (w, h) = smart_resize(width, height, c.patch * c.merge, c.min_pixels, c.max_pixels);
    let (gh, gw) = (h / c.patch, w / c.patch);
    Plan { w, h, gh, gw, lh: gh / c.merge, lw: gw / c.merge }
}

/// Patch k's place in the grid: the patches go a 2 x 2 unit at a time (the merger takes four rows in a row), the
/// units in reading order
pub fn patch_yx(pl: &Plan, m: usize, k: usize) -> (usize, usize) {
    let per = m * m;
    let (u, i) = (k / per, k % per);
    let (uy, ux) = (u / pl.lw, u % pl.lw);
    (uy * m + i / m, ux * m + i % m)
}

/// A picture's tower input: [n][3 * p * p] (a patch's channels, then its rows, then its columns), the pixels
/// (x / 255 - mean) / std, the picture resized (bicubic, as PIL) to the plan's size
pub fn patches(pic: &Picture, pl: &Plan, c: &Cfg) -> Vec<f32> {
    let img = pic.rgb();
    let img = if (img.width as usize, img.height as usize) == (pl.w, pl.h) { img } else { img.resize_bicubic(pl.w as u32, pl.h as u32) };
    let p = c.patch;
    let mut out = Vec::with_capacity(pl.n() * 3 * p * p);
    for k in 0..pl.n() {
        let (py, px) = patch_yx(pl, c.merge, k);
        for ch in 0..3 {
            for y in 0..p {
                for x in 0..p {
                    let v = img.data[((py * p + y) * pl.w + px * p + x) * 3 + ch] as f32 / 255.0;
                    out.push((v - c.mean[ch]) / c.std[ch]);
                }
            }
        }
    }
    out
}

/// The learned positions for the plan's grid in patch order: [n][dim], each the bilinear mix (corners aligned) of
/// the four nearest of the `grid` x `grid` table (transformers' fast_pos_embed_interpolate)
pub fn positions(table: &[f32], pl: &Plan, c: &Cfg) -> Vec<f32> {
    let (g, d) = (c.grid, c.dim);
    let idx = |n: usize, i: usize| if n <= 1 { 0.0 } else { i as f64 * (g - 1) as f64 / (n - 1) as f64 };
    let mut out = vec![0f32; pl.n() * d];
    for k in 0..pl.n() {
        let (y, x) = patch_yx(pl, c.merge, k);
        let (fy, fx) = (idx(pl.gh, y), idx(pl.gw, x));
        let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
        let (y1, x1) = ((y0 + 1).min(g - 1), (x0 + 1).min(g - 1));
        let (dy, dx) = ((fy - y0 as f64) as f32, (fx - x0 as f64) as f32);
        let w = [(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx];
        let rows = [y0 * g + x0, y0 * g + x1, y1 * g + x0, y1 * g + x1];
        let o = &mut out[k * d..(k + 1) * d];
        for (wi, r) in w.iter().zip(rows) {
            for (a, b) in o.iter_mut().zip(&table[r * d..(r + 1) * d]) {
                *a += wi * b;
            }
        }
    }
    out
}

/// The rotary table of the tower's attention: [n][head / 2] (cos, sin) - the first quarter of a head's frequencies
/// turn by the patch's row, the second by its column (base 10000)
pub fn rotary(pl: &Plan, c: &Cfg) -> Vec<f32> {
    let half = c.dim / c.heads / 2;
    let q = half / 2;
    let mut cs = vec![0f32; pl.n() * half * 2];
    for k in 0..pl.n() {
        let (y, x) = patch_yx(pl, c.merge, k);
        for i in 0..half {
            let (p, j) = if i < q { (y as f64, i) } else { (x as f64, i - q) };
            let a = p / 10000f64.powf(2.0 * j as f64 / half as f64);
            cs[(k * half + i) * 2] = a.cos() as f32;
            cs[(k * half + i) * 2 + 1] = a.sin() as f32;
        }
    }
    cs
}

/// A picture handed in: its plan, its patches until its rows are made, then its rows ([lh * lw][out])
pub struct Img {
    pub hash: u64,
    pub plan: Plan,
    pub patches: Mutex<Option<Arc<Vec<f32>>>>,
    pub rows: Mutex<Option<Arc<Vec<f32>>>>,
}

/// The pictures held by key, and their keys oldest first
type Held = (HashMap<u32, Arc<Img>>, Vec<u32>);

/// The tower's file, settings and learned positions, the span's marker tokens, the pictures handed in
pub struct Vision {
    pub file: Gguf,
    pub cfg: Cfg,
    /// the learned positions [grid * grid][dim]
    pub table: Vec<f32>,
    /// `<|vision_start|>`, `<|vision_end|>`: the real tokens around a span
    pub start: u32,
    pub end: u32,
    /// what a chat template puts in a picture's place (`<|image_pad|>`): replaced by start, the span, end
    pub marker: String,
    images: Mutex<Held>,
}

/// NS_Q35_VISION: the mmproj file (default: an mmproj*.gguf beside the model, else in its parent folder; 0: none)
pub fn sidecar(model: &Path) -> Option<PathBuf> {
    match std::env::var("NS_Q35_VISION") {
        Ok(v) if v == "0" => None,
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            let dir = model.parent()?;
            let find = |d: &Path| -> Option<PathBuf> {
                let mut v: Vec<PathBuf> = std::fs::read_dir(d).ok()?.flatten().map(|e| e.path()).filter(|p| {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
                }).collect();
                // F16 before BF16 / F32 (the order the names sort in)
                v.sort();
                v.into_iter().next()
            };
            find(dir)
        }
    }
}

/// A tensor's values as float32 (F32, F16 or BF16)
pub fn floats(f: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("the vision file has no {name}")))?;
    let raw = f.read(t).map_err(|e| Error(e.0))?;
    Ok(match t.ty {
        GType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        GType::BF16 => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        GType::F16 => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        ty => return Err(Error(format!("{name}: {} (the vision file's tensors are F32, F16 or BF16)", ty.name()))),
    })
}

/// A tensor's values as half (F16 as stored; F32 / BF16 rounded)
pub fn halves(f: &Gguf, name: &str) -> Result<Vec<u8>> {
    let t = f.tensor(name).ok_or_else(|| Error(format!("the vision file has no {name}")))?;
    if t.ty == GType::F16 {
        return f.read(t).map_err(|e| Error(e.0)).map(|r| r.to_vec());
    }
    Ok(floats(f, name)?.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect())
}

pub fn f16_to_f32(h: u16) -> f32 {
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

/// float32 to half (nearest, ties to even; no subnormals kept below 2^-24)
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if e >= 31 {
        return s | 0x7C00 | if (b & 0x7FFF_FFFF) > 0x7F80_0000 { 0x200 } else { 0 };
    }
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1 << (shift - 1);
        let r = m & ((1 << shift) - 1);
        let mut v = (m >> shift) as u16;
        if r > half || (r == half && v & 1 == 1) {
            v += 1;
        }
        return s | v;
    }
    let mut v = ((e as u32) << 10 | (m >> 13)) as u16;
    let r = m & 0x1FFF;
    if r > 0x1000 || (r == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    s | v
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A token's id by its text in the model's vocabulary
fn token_id(model: &Gguf, text: &str) -> Option<u32> {
    model.meta("tokenizer.ggml.tokens")?.as_array()?.iter().position(|t| t.as_str() == Some(text)).map(|i| i as u32)
}

/// NS_Q35_IMAGE_TOKENS: the most positions a picture takes (default 4,096 - llama.cpp's; larger pictures are scaled
/// down to fit)
pub fn max_tokens() -> usize {
    std::env::var("NS_Q35_IMAGE_TOKENS").ok().and_then(|v| v.parse().ok()).filter(|v| *v >= 4).unwrap_or(4096).min(1 << POS_BITS)
}

impl Vision {
    /// The tower's file for `model` (its vocabulary names the span's markers); `max_tokens` the most positions a picture
    /// may take here
    pub fn open(path: &Path, model: &Gguf, n_embd: usize, max_tokens: usize) -> Result<Vision> {
        let file = Gguf::open(path).map_err(|e| Error(format!("the vision file {}: {}", path.display(), e.0)))?;
        if file.meta("clip.projector_type").and_then(Value::as_str) != Some("qwen3vl_merger") {
            return Err(Error(format!("{}: not Qwen3-VL's vision tower (clip.projector_type qwen3vl_merger)", path.display())));
        }
        let u = |k: &str| file.meta(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| Error(format!("the vision file: {k} is missing")));
        let three = |k: &str| -> [f32; 3] {
            file.meta(k).and_then(|v| v.as_array()).map(|a| {
                let v: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                if v.len() == 3 { [v[0], v[1], v[2]] } else { [0.5; 3] }
            }).unwrap_or([0.5; 3])
        };
        let (patch, merge) = (u("clip.vision.patch_size")?, u("clip.vision.spatial_merge_size").unwrap_or(2));
        let unit = (patch * merge * patch * merge) as u64;
        let dim = u("clip.vision.embedding_length")?;
        let table = floats(&file, "v.position_embd.weight")?;
        let grid = ((table.len() / dim) as f64).sqrt() as usize;
        let cfg = Cfg {
            layers: u("clip.vision.block_count")?,
            dim,
            heads: u("clip.vision.attention.head_count")?,
            ff: u("clip.vision.feed_forward_length")?,
            patch,
            merge,
            out: u("clip.vision.projection_dim")?,
            eps: file.meta("clip.vision.attention.layer_norm_epsilon").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            grid,
            mean: three("clip.vision.image_mean"),
            std: three("clip.vision.image_std"),
            // the processor's least area (shortest_edge 65,536 pixels), the most: max_tokens positions
            min_pixels: 65_536u64.min(max_tokens as u64 * unit),
            max_pixels: max_tokens as u64 * unit,
        };
        if grid * grid * dim != table.len() || merge != 2 || cfg.out != n_embd || !cfg.dim.is_multiple_of(cfg.heads) || !(cfg.dim / cfg.heads).is_multiple_of(4) {
            return Err(Error(format!("the vision file: a {grid} x {grid} position grid of {dim}, merge {merge}, rows of {} (the model's {n_embd}), heads of {}",
                                     cfg.out, cfg.dim / cfg.heads.max(1))));
        }
        if file.meta("clip.vision.is_deepstack_layers").and_then(|v| v.as_array()).is_some_and(|a| a.iter().any(|x| matches!(x, Value::Bool(true)))) {
            return Err(Error("the vision file has deepstack layers (not read here)".into()));
        }
        let start = token_id(model, "<|vision_start|>").ok_or("the vocabulary has no <|vision_start|>")?;
        let end = token_id(model, "<|vision_end|>").ok_or("the vocabulary has no <|vision_end|>")?;
        token_id(model, "<|image_pad|>").ok_or("the vocabulary has no <|image_pad|>")?;
        Ok(Vision { file, cfg, table, start, end, marker: "<|image_pad|>".into(), images: Mutex::new((HashMap::new(), Vec::new())) })
    }

    /// A picture's ids (PNG, JPEG or WebP bytes): `<|vision_start|>`, its span, `<|vision_end|>`; the picture registered
    /// for the passes that read it
    pub fn tokens(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        let pic = Picture::decode(bytes).map_err(Error)?;
        if pic.width == 0 || pic.height == 0 {
            return Err(Error("an empty picture".into()));
        }
        let pl = plan(pic.width, pic.height, &self.cfg);
        let hash = fnv(bytes) ^ ((pic.width as u64) << 32 | pic.height as u64);
        let mut g = self.images.lock().unwrap();
        // the key: the hash's low bits, the next free one past a picture with another hash
        let mut key = (hash as u32) & KEY_MASK;
        loop {
            match g.0.get(&key) {
                Some(im) if im.hash != hash => key = (key + 1) & KEY_MASK,
                _ => break,
            }
        }
        g.0.entry(key).or_insert_with(|| {
            let px = patches(&pic, &pl, &self.cfg);
            Arc::new(Img { hash, plan: pl, patches: Mutex::new(Some(Arc::new(px))), rows: Mutex::new(None) })
        });
        g.1.retain(|k| *k != key);
        g.1.push(key);
        while g.1.len() > KEEP {
            let old = g.1.remove(0);
            g.0.remove(&old);
        }
        let mut ids = Vec::with_capacity(pl.span() + 2);
        ids.push(self.start);
        ids.extend((0..pl.span()).map(|i| IMG_BIT | (key << POS_BITS) | i as u32));
        ids.push(self.end);
        Ok(ids)
    }

    /// The picture an id of a span names
    pub fn image(&self, id: u32) -> Result<Arc<Img>> {
        self.images.lock().unwrap().0.get(&key_of(id)).cloned()
            .ok_or_else(|| Error("an image span whose picture is no longer held (more than 8 pictures since): send it again".into()))
    }
}

/// The rotary positions (t, row, column) of a pass's rows from `kv0` (the session's position) with the text's offset
/// `shift` (its rotary position = its place less shift): each image position (row, column) from its span's first
/// position; the offset after each span that ends in the pass grows by the span's length less its advance. The new
/// offset with them.
pub fn rope_rows(tokens: &[u32], kv0: usize, shift: usize, plan_of: &dyn Fn(u32) -> Result<Plan>) -> Result<(Vec<i32>, usize)> {
    let mut out = Vec::with_capacity(tokens.len() * 3);
    let mut shift = shift;
    for (i, &t) in tokens.iter().enumerate() {
        let kv = kv0 + i;
        if !is_image(t) {
            let p = (kv - shift) as i32;
            out.extend([p, p, p]);
            continue;
        }
        let pl = plan_of(t)?;
        let k = pos_of(t);
        if k >= pl.span() {
            return Err(Error(format!("an image position {k} past its span of {}", pl.span())));
        }
        // the span's first position, as text would have it
        let base = (kv - k - shift) as i32;
        let (r, c) = pl.rc(k);
        out.extend([base, base + r as i32, base + c as i32]);
        if k + 1 == pl.span() {
            shift += pl.span() - pl.advance();
        }
    }
    Ok((out, shift))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        Cfg { layers: 27, dim: 1152, heads: 16, ff: 4304, patch: 16, merge: 2, out: 5120, eps: 1e-6, grid: 48, mean: [0.5; 3], std: [0.5; 3],
              min_pixels: 65_536, max_pixels: 4096 * 1024 }
    }

    #[test]
    fn resizes_as_the_processor() {
        // aligned sizes stay; the area is kept within the limits, the sides multiples of 32
        assert_eq!(smart_resize(640, 480, 32, 65_536, 4096 * 1024), (640, 480));
        assert_eq!(smart_resize(100, 50, 32, 65_536, 4096 * 1024), (384, 192));
        let (w, h) = smart_resize(4000, 3000, 32, 65_536, 4096 * 1024);
        assert!(w * h <= 4096 * 1024 && w % 32 == 0 && h % 32 == 0, "{w} x {h}");
        let p = plan(640, 480, &cfg());
        assert_eq!((p.gh, p.gw, p.lh, p.lw, p.span(), p.advance()), (30, 40, 15, 20, 300, 20));
    }

    #[test]
    fn patches_go_a_unit_at_a_time() {
        let p = Plan { w: 128, h: 64, gh: 4, gw: 8, lh: 2, lw: 4 };
        let yx: Vec<(usize, usize)> = (0..8).map(|k| patch_yx(&p, 2, k)).collect();
        assert_eq!(yx, vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(patch_yx(&p, 2, 16), (2, 0));
    }

    #[test]
    fn rotary_positions_follow_qwen3_vl() {
        let p = Plan { w: 64, h: 128, gh: 8, gw: 4, lh: 4, lw: 2 };
        let plan_of = |_: u32| Ok(p);
        // text, start, a 4 x 2 span, end, text: the span from position 2, the text after it at 2 + 4
        let mut toks = vec![10, 11];
        toks.extend((0..8u32).map(|i| IMG_BIT | (5 << POS_BITS) | i));
        toks.extend([12, 13]);
        let (pos, shift) = rope_rows(&toks, 0, 0, &plan_of).unwrap();
        assert_eq!(&pos[0..6], &[0, 0, 0, 1, 1, 1]);
        assert_eq!(&pos[6..9], &[2, 2, 2]);
        assert_eq!(&pos[9..12], &[2, 2, 3]);
        assert_eq!(&pos[27..30], &[2, 5, 3]);
        assert_eq!(&pos[30..36], &[6, 6, 6, 7, 7, 7]);
        assert_eq!(shift, 8 - 4);
        // the same span read in two passes gives the same positions
        let (a, s1) = rope_rows(&toks[..5], 0, 0, &plan_of).unwrap();
        let (b, s2) = rope_rows(&toks[5..], 5, s1, &plan_of).unwrap();
        assert_eq!([a, b].concat(), pos);
        assert_eq!(s2, shift);
    }

    #[test]
    fn ids_carry_key_and_place() {
        let id = IMG_BIT | (0x2ABCD << POS_BITS) | 4095;
        assert!(is_image(id));
        assert_eq!((key_of(id), pos_of(id)), (0x2ABCD, 4095));
        assert_eq!(f16_to_f32(f32_to_f16(1.5)), 1.5);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
    }
}
