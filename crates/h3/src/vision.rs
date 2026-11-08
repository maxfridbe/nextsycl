//! Pictures and clips for the text encoder (ComfyUI `text_encoders/minimax.py`): the prompt is presented as pieces of
//! text and vision blocks, never chat-templated -
//!
//! ```text
//!   t2va    <prompt>
//!   fl2va   "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>      (the keyframes' pictures)
//!   ref2va  per reference, in order:  picture  "<Picture i>: " <vision block>
//!                                     sound    "<Audio j>: "                               (never enters the encoder)
//!                                     clip     "<Video k>: " then per two frames at 2 fps "<T.T seconds>" <vision block>
//!           then <prompt>
//! ```
//!
//! A vision block is `<|vision_start|>`, the vision tower's tokens, `<|vision_end|>`; its tokens sit on the M-RoPE
//! grid (time fixed, height and width along the merged patch grid) and get the tower's deepstack features added after
//! the encoder's layers 0, 1 and 2. The denoiser sees the whole block as video (modality tag 0), the rest as text (1).
//!
//! The tower (Qwen3-VL 32B's, 27 blocks, `visual.*` in a file of its own) runs from crates/qwen3vl in a context of its
//! own on the engine's card, loaded for the clip and freed before the text encoder streams in.

use std::path::Path;
use std::sync::Arc;

use nextsycl_qwen3vl::vision::{patches_frames, Vision, MERGE, PATCH};

use crate::{Error, Result};

pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
/// the vision tokens' placeholder id (their rows come from the tower, not the embedding table)
pub const IMAGE_PAD: u32 = 151655;
/// an empty presentation is one pad token, as the reference encodes it
pub const PAD: u32 = 151643;

/// A picture for the tower: one frame (a picture) or two (a clip's two frames in one block), RGB [h, w, 3] in [0, 1],
/// sides multiples of 32
pub struct Frames {
    pub frames: Vec<Vec<f32>>,
    pub w: usize,
    pub h: usize,
}

pub enum Piece {
    Text(String),
    Vision(Frames),
}

/// A vision block as the encoder receives it: where its tokens start in the ids, its merged grid (rows, columns), the
/// tokens [n, hidden] and the deepstack features (one [n, hidden] each), float32 on the host
pub struct Seen {
    pub at: usize,
    pub grid: (usize, usize),
    pub tokens: Vec<f32>,
    pub deep: Vec<Vec<f32>>,
}

/// The ids, the denoiser's modality tag per id, the vision blocks, and the embeddings' rows (`embedding:NAME` in the
/// text: rows of the encoder's input [n, hidden] that stand in for tokens)
pub struct Presented {
    pub ids: Vec<u32>,
    pub tags: Vec<i32>,
    pub seen: Vec<Seen>,
    pub embeds: Vec<(usize, Vec<f32>)>,
}

/// A text piece split as ComfyUI's tokenizer splits it (sd1_clip.SDTokenizer): at every `embedding:` that follows
/// white space; an embedding's name is the next word (cut at `<` or `[`, retried without trailing commas), and the
/// words after it are text again (joined by single spaces)
enum Seg {
    Text(String),
    Embed(Vec<f32>, String),
}

const EMBEDDING: &str = "embedding:";

/// An embedding by name from the directory: `NAME`, `NAME.safetensors`; its rows for the 32B encoder (key
/// `qwen3vl_32b`, [n, 5120]) as float32
fn load_embed(dir: &Path, name: &str) -> Option<Vec<f32>> {
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return None;
    }
    let p = [dir.join(name), dir.join(format!("{name}.safetensors"))].into_iter().find(|p| p.is_file())?;
    let st = nextsycl_gguf::safetensors::SafeTensors::open(&p).ok()?;
    let key = if st.tensor("qwen3vl_32b").is_some() { "qwen3vl_32b".to_string() } else { st.tensors.first()?.name.clone() };
    st.f32(&key).ok()
}

fn split_embeddings(text: &str, dir: Option<&Path>, log: &mut dyn FnMut(String)) -> Vec<Seg> {
    let Some(dir) = dir else { return vec![Seg::Text(text.to_string())] };
    // split before each `embedding:` preceded by white space (the white space stays with the text before it)
    let mut words = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while let Some(off) = text[i..].find(EMBEDDING) {
        let at = i + off;
        if at > 0 && text[..at].chars().next_back().is_some_and(char::is_whitespace) {
            words.push(&text[start..at]);
            start = at;
        }
        i = at + EMBEDDING.len();
    }
    words.push(&text[start..]);
    let mut out = Vec::new();
    for w in words.into_iter().filter(|w| !w.is_empty()) {
        let Some(rest) = w.strip_prefix(EMBEDDING) else {
            out.push(Seg::Text(w.to_string()));
            continue;
        };
        let rest = rest.trim_matches('\n');
        let mut parts = rest.split_whitespace();
        let mut name = parts.next().unwrap_or("").to_string();
        let mut leftover = parts.collect::<Vec<_>>().join(" ");
        if let Some(b) = name.find(['<', '[']) {
            leftover = format!("{}{}", &name[b..], if leftover.is_empty() { String::new() } else { format!(" {leftover}") });
            name.truncate(b);
        }
        let mut embed = load_embed(dir, &name);
        if embed.is_none() {
            let stripped = name.trim_matches(',');
            if stripped.len() < name.len() {
                embed = load_embed(dir, stripped);
                leftover = format!("{} {leftover}", &name[stripped.len()..]);
                name = stripped.to_string();
            }
        }
        match embed {
            Some(e) => out.push(Seg::Embed(e, name)),
            None => log(format!("prompt : embedding:{name} does not exist, ignoring")),
        }
        if !leftover.is_empty() {
            out.push(Seg::Text(leftover));
        }
    }
    out
}

impl Presented {
    pub fn has_vision(&self) -> bool {
        !self.seen.is_empty()
    }
}

/// The pieces as ids (each text piece tokenized on its own, as the reference does), the vision blocks through the
/// tower (`visual`: its file; `gpu`: the card)
pub fn present(pieces: &[Piece], tok: &crate::tokenizer::Tokenizer, visual: Option<&Path>, embeddings: Option<&Path>, gpu: usize,
               log: &mut dyn FnMut(String)) -> Result<Presented> {
    let mut ids = Vec::new();
    let mut tags = Vec::new();
    let mut blocks: Vec<(usize, &Frames)> = Vec::new();
    let mut embeds = Vec::new();
    for p in pieces {
        match p {
            Piece::Text(s) if s.is_empty() => {}
            Piece::Text(s) => {
                for seg in split_embeddings(s, embeddings, log) {
                    match seg {
                        Seg::Text(t) => {
                            let t = tok.encode(&t)?;
                            tags.extend(std::iter::repeat_n(1, t.len()));
                            ids.extend(t);
                        }
                        Seg::Embed(rows, name) => {
                            let n = rows.len() / 5120;
                            log(format!("prompt : embedding:{name} = {n} rows"));
                            embeds.push((ids.len(), rows));
                            ids.extend(std::iter::repeat_n(PAD, n));
                            tags.extend(std::iter::repeat_n(1, n));
                        }
                    }
                }
            }
            Piece::Vision(f) => {
                if !f.h.is_multiple_of(PATCH * MERGE) || !f.w.is_multiple_of(PATCH * MERGE) || f.frames.is_empty() || f.frames.len() > 2 {
                    return Err(Error(format!("a {}x{} vision block of {} frames: sides multiples of 32, one or two frames", f.w, f.h, f.frames.len())));
                }
                let n = (f.h / PATCH / MERGE) * (f.w / PATCH / MERGE);
                ids.push(VISION_START);
                blocks.push((ids.len(), f));
                ids.extend(std::iter::repeat_n(IMAGE_PAD, n));
                ids.push(VISION_END);
                tags.extend(std::iter::repeat_n(0, n + 2));
            }
        }
    }
    if ids.is_empty() {
        ids.push(PAD);
        tags.push(1);
    }
    let mut seen = Vec::new();
    if !blocks.is_empty() {
        let path = visual.ok_or("pictures for the text encoder need its vision tower (the model's \"visual\" file)")?;
        let t0 = std::time::Instant::now();
        nextsycl_core::use_kind("video");
        let g: Arc<nextsycl_core::Gpu> = nextsycl_core::Gpu::open(gpu).map_err(|e| Error(e.0))?;
        let nsd = nextsycl_diffusion::kernels::Nsd::new(&g).map_err(|e| Error(e.0))?;
        let st = nextsycl_gguf::safetensors::SafeTensors::open(path).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))?;
        let v = Vision::load_from(&st, "visual.", &g, &mut |l| log(l)).map_err(|e| Error(e.0))?;
        for (at, f) in blocks {
            let rgb: Vec<Vec<f32>> = f.frames.iter().map(|x| x.iter().map(|v| v * 255.0).collect()).collect();
            let pair = [rgb[0].as_slice(), rgb[rgb.len() - 1].as_slice()];
            let (px, grid) = patches_frames(&pair, f.h, f.w).map_err(|e| Error(e.0))?;
            let s = v.see_patches(&nsd, &px, grid).map_err(|e| Error(e.0))?;
            nsd.wait().map_err(|e| Error(e.0))?;
            let host = |b: &nextsycl_core::DevBuf| b.to_f32().map_err(|e| Error(e.0));
            seen.push(Seen { at, grid: (grid.0 / MERGE, grid.1 / MERGE), tokens: host(&s.tokens)?, deep: s.deep.iter().map(host).collect::<Result<_>>()? });
        }
        log(format!("vision : {} block(s) through the tower in {:.1} s ({} tokens)", seen.len(), t0.elapsed().as_secs_f64(),
                    seen.iter().map(|s| s.grid.0 * s.grid.1).sum::<usize>()));
    }
    Ok(Presented { ids, tags, seen, embeds })
}

/// Each id's (time, height, width) position: text counts on; a vision block's tokens share the time of its first
/// token and spread over its grid, and the text after it resumes past the larger side (ComfyUI qwen_vl.py
/// qwen2vl_mrope_position_ids)
pub fn positions(ids: usize, seen: &[Seen]) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(ids);
    let mut p = 0f64;
    let mut i = 0;
    let mut blocks = seen.iter().peekable();
    while i < ids {
        if let Some(s) = blocks.peek().filter(|s| s.at == i) {
            let (lh, lw) = s.grid;
            for r in 0..lh {
                for c in 0..lw {
                    out.push([p, p + r as f64, p + c as f64]);
                }
            }
            i += lh * lw;
            p += lh.max(lw) as f64;
            blocks.next();
            continue;
        }
        out.push([p, p, p]);
        p += 1.0;
        i += 1;
    }
    out
}

/// The rotary table [ids, 64, 2] (cos, sin) for the interleaved M-RoPE of Qwen3-VL (sections 24 / 20 / 20: pair j
/// turns by the height position when j % 3 == 1 and by the width position when j % 3 == 2, below pair 60; by the
/// time position otherwise)
pub fn rope_table(pos: &[[f64; 3]], head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half).map(|j| 1.0 / theta.powf((2 * j) as f32 / head_dim as f32)).collect();
    let mut cs = Vec::with_capacity(pos.len() * half * 2);
    for p in pos {
        for (j, f) in inv.iter().enumerate() {
            let axis = if j < 60 && j % 3 == 1 { 1 } else if j < 60 && j % 3 == 2 { 2 } else { 0 };
            let a = p[axis] as f32 * f;
            cs.extend_from_slice(&[a.cos(), a.sin()]);
        }
    }
    cs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_follow_comfyui() {
        // 3 text ids, a 2 x 3 block, 2 text ids: the block at time 3, the text after it at 3 + 3
        let seen = vec![Seen { at: 3, grid: (2, 3), tokens: vec![], deep: vec![] }];
        let p = positions(11, &seen);
        assert_eq!(p[2], [2.0, 2.0, 2.0]);
        assert_eq!(p[3], [3.0, 3.0, 3.0]);
        assert_eq!(p[5], [3.0, 3.0, 5.0]);
        assert_eq!(p[6], [3.0, 4.0, 3.0]);
        assert_eq!(p[8], [3.0, 4.0, 5.0]);
        assert_eq!(p[9], [6.0, 6.0, 6.0]);
        assert_eq!(p[10], [7.0, 7.0, 7.0]);
    }

    #[test]
    fn embeddings_split_as_comfyui_does() {
        let dir = std::env::temp_dir().join(format!("h3-emb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows: Vec<f32> = (0..2 * 5120).map(|i| i as f32).collect();
        let t = std::collections::BTreeMap::from([("qwen3vl_32b".to_string(), (vec![2, 5120], rows))]);
        crate::safetensors::write_f32(&dir.join("e1.safetensors"), &t, &std::collections::BTreeMap::new()).unwrap();
        let show = |text: &str| -> Vec<String> {
            split_embeddings(text, Some(&dir), &mut |_| {}).into_iter().map(|s| match s {
                Seg::Text(t) => format!("T[{t}]"),
                Seg::Embed(r, n) => format!("E[{n}:{}]", r.len() / 5120),
            }).collect()
        };
        assert_eq!(show("a fox embedding:e1 runs  far"), ["T[a fox ]", "E[e1:2]", "T[runs far]"]);
        assert_eq!(show("embedding:e1, then"), ["E[e1:2]", "T[, then]"]);
        assert_eq!(show("x embedding:e1<b> y"), ["T[x ]", "E[e1:2]", "T[<b> y]"]);
        // not after white space: plain text; an unknown name: left out
        assert_eq!(show("x,embedding:e1 y"), ["T[x,embedding:e1 y]"]);
        assert_eq!(show("x embedding:nope y"), ["T[x ]", "T[y]"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn text_alone_is_plain_rope() {
        let p = positions(4, &[]);
        let t = rope_table(&p, 128, 5e6);
        // position 2, pair 5: angle 2 / theta^(10/128)
        let a = 2.0f32 / 5e6f32.powf(10.0 / 128.0);
        assert!((t[(2 * 64 + 5) * 2] - a.cos()).abs() < 1e-6);
    }
}
