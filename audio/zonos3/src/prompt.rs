//! The prompt: the rows the backbone reads before it speaks, each ten audio columns and a text column.
//!
//! ```text
//!   [speaker row | reference prefix (frames + 9 rows, delayed)] [loudness] [fmax] [speaking rate]
//!   BOS (SuperBPE ids + 4) EOS  [silence rows]
//! ```
//!
//! Origin: Zyphra's `zonos3.tts.prompt` and `zonos3.tts.superbpe` (github.com/Zyphra/ZONOS3-Mini, MIT): the speaker
//! and condition rows are all-pad rows whose embeddings the model replaces (a projected speaker embedding, a bucket's
//! row); the reference prefix holds a recording's codes in the delay pattern (codebook c of frame t on row t + c);
//! the silence prefill is the codec's silence frame in the same pattern, its first rows only.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;

use crate::lm::Config;

/// How the voice comes into the prompt
#[derive(Clone, Debug)]
pub enum Voice {
    /// no voice: the model's own
    None,
    /// a 192-d speaker embedding (a preset's, or a recording's from the speaker encoder)
    Embedding(Vec<f32>),
    /// a recording's codes [frames, 10] (a preset's .codes.npy): the reference prefix
    Reference(Vec<[u32; 10]>),
}

/// A prompt and where its conditioning rows are
#[derive(Clone, Debug)]
pub struct Prompt {
    /// rows of ncb + 1 ids
    pub ids: Vec<u32>,
    pub rows: usize,
    /// the speaker row's position and its embedding (not yet projected)
    pub speaker: Option<(usize, Vec<f32>)>,
    /// (row, bucket) for each condition table, in order
    pub buckets: Vec<(usize, usize)>,
    /// the silence rows' audio columns (the decoder's first frames)
    pub prefill: Vec<[u32; 10]>,
    /// whether the last row holds audio (the reference's cold-start rule)
    pub audio_prefix: bool,
}

/// The longest reference prefix (60 s of codes, the registry's reference_max_frames)
pub const REFERENCE_MAX_FRAMES: usize = 900;

/// SuperBPE's convention: BOS, the tokenizer's ids + offset, EOS
pub fn text_ids(tok: &Tokenizer, c: &Config, text: &str) -> Vec<u32> {
    let mut v = vec![c.bos];
    v.extend(tok.encode_ordinary(text).into_iter().map(|i| i + c.offset));
    v.push(c.eos);
    v
}

/// bisect_right over a condition's edges: the bucket of a value
pub fn bucket(edges: &[f64], value: f64) -> usize {
    edges.iter().take_while(|e| **e <= value).count()
}

/// The prompt of `text` in `voice`: `buckets` per condition table (None: unknown), `silence` prefill frames
pub fn build(c: &Config, text_ids: &[u32], voice: &Voice, buckets: &[Option<usize>], silence: usize) -> Result<Prompt> {
    let ncb = c.ncb;
    let w = ncb + 1;
    let pad = c.pad;
    let tpad = c.text_vocab;
    let mut ids = Vec::new();
    let row = |ids: &mut Vec<u32>, audio: &[u32], text: u32| {
        ids.extend_from_slice(audio);
        ids.push(text);
    };
    let pads = vec![pad; ncb];
    let mut speaker = None;
    match voice {
        Voice::None => {}
        Voice::Embedding(v) => {
            if v.len() != c.speaker_dim {
                return Err(Error(format!("a speaker embedding of {} values; this model takes {}", v.len(), c.speaker_dim)));
            }
            speaker = Some((0, v.clone()));
            row(&mut ids, &pads, tpad);
        }
        Voice::Reference(codes) => {
            let codes = &codes[..codes.len().min(REFERENCE_MAX_FRAMES)];
            if codes.is_empty() {
                return Err(Error("a reference of no frames".into()));
            }
            if codes.iter().flatten().any(|v| *v as usize >= c.codebook) {
                return Err(Error(format!("reference codes outside the codebook (0..{})", c.codebook)));
            }
            for t in 0..codes.len() + ncb - 1 {
                let a: Vec<u32> = (0..ncb).map(|cb| if t >= cb && t - cb < codes.len() { codes[t - cb][cb] } else { pad }).collect();
                row(&mut ids, &a, tpad);
            }
        }
    }
    let mut bk = Vec::new();
    for (i, n) in c.cond_counts.iter().enumerate() {
        let b = buckets.get(i).copied().flatten().unwrap_or(*n).min(*n);
        bk.push((ids.len() / w, b));
        row(&mut ids, &pads, tpad);
    }
    for t in text_ids {
        row(&mut ids, &pads, *t);
    }
    let mut prefill = Vec::new();
    if silence > 0 {
        if c.silence.len() != ncb {
            return Err(Error("the backbone carries no silence frame".into()));
        }
        for t in 0..silence {
            let a: Vec<u32> = (0..ncb).map(|cb| if t >= cb { c.silence[cb] } else { pad }).collect();
            let mut f = [0u32; 10];
            f[..ncb.min(10)].copy_from_slice(&a[..ncb.min(10)]);
            prefill.push(f);
            row(&mut ids, &a, tpad);
        }
    }
    let rows = ids.len() / w;
    let audio_prefix = ids[(rows - 1) * w..(rows - 1) * w + ncb].iter().any(|v| *v != pad);
    Ok(Prompt { ids, rows, speaker, buckets: bk, prefill, audio_prefix })
}

/// A .npy file's array: (shape, values as f64) for little-endian int8..int64, uint8..uint64, float32 / float64,
/// C order
pub fn read_npy(b: &[u8]) -> Result<(Vec<usize>, Vec<f64>)> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(Error("not a .npy file".into()));
    }
    let (hlen, start) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = std::str::from_utf8(&b[start..start + hlen]).map_err(|_| Error("a .npy header that is not text".into()))?;
    let field = |k: &str| -> Option<&str> {
        let i = head.find(&format!("'{k}'"))?;
        Some(head[i + k.len() + 2..].trim_start_matches([':', ' ']))
    };
    let descr = field("descr").and_then(|s| s.strip_prefix('\'')).and_then(|s| s.split('\'').next()).ok_or_else(|| Error("a .npy header without descr".into()))?;
    if field("fortran_order").is_some_and(|s| s.starts_with("True")) {
        return Err(Error("a Fortran-order .npy array".into()));
    }
    let shape_s = field("shape").and_then(|s| s.strip_prefix('(')).and_then(|s| s.split(')').next()).ok_or_else(|| Error("a .npy header without shape".into()))?;
    let shape: Vec<usize> = shape_s.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect();
    let n: usize = shape.iter().product();
    let data = &b[start + hlen..];
    let (size, f): (usize, fn(&[u8]) -> f64) = match descr {
        "<f4" => (4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<f8" => (8, |c| f64::from_le_bytes(c.try_into().expect("8 bytes"))),
        "<i8" => (8, |c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64),
        "<i4" => (4, |c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<i2" => (2, |c| i16::from_le_bytes([c[0], c[1]]) as f64),
        "|i1" => (1, |c| c[0] as i8 as f64),
        "<u8" => (8, |c| u64::from_le_bytes(c.try_into().expect("8 bytes")) as f64),
        "<u4" => (4, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        "<u2" => (2, |c| u16::from_le_bytes([c[0], c[1]]) as f64),
        "|u1" => (1, |c| c[0] as f64),
        d => return Err(Error(format!("a .npy array of {d}"))),
    };
    if data.len() < n * size {
        return Err(Error("a .npy file shorter than its shape".into()));
    }
    Ok((shape, data[..n * size].chunks_exact(size).map(f).collect()))
}

/// A preset's codes (.codes.npy: [frames, 10] integers)
pub fn read_codes(b: &[u8]) -> Result<Vec<[u32; 10]>> {
    let (shape, v) = read_npy(b)?;
    if shape.len() != 2 || shape[1] != 10 {
        return Err(Error(format!("reference codes of shape {shape:?}; [frames, 10] expected")));
    }
    Ok(v.chunks_exact(10).map(|r| {
        let mut f = [0u32; 10];
        for (i, x) in r.iter().enumerate() {
            f[i] = x.max(0.0) as u32;
        }
        f
    }).collect())
}

/// A preset's speaker embedding (.embedding.npy: [192] floats)
pub fn read_embedding(b: &[u8]) -> Result<Vec<f32>> {
    let (shape, v) = read_npy(b)?;
    if shape.iter().product::<usize>() != v.len() || v.is_empty() {
        return Err(Error("an empty speaker embedding".into()));
    }
    Ok(v.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            layers: 1, ctx: 64, dim: 4, head_dim: 32, heads: 1, kv_heads: 1, ffn: 4, experts: 1, topk: 1, ncb: 10, codebook: 4096, audio_vocab: 4098,
            eoa: 4096, pad: 4097, text_vocab: 8196, text_rows: 8197, gdn_heads: 1, gdn_dim: 128, gdn_kernel: 4, speaker_dim: 3,
            cond_names: vec!["a".into(), "b".into()], cond_counts: vec![10, 30], cond_edges: vec![vec![], vec![]], theta: 1e4, eps: 1e-5,
            qk_eps: 1e-6, softcap: 15.0, gdn_period: 4, silence: (100..110).collect(), tokenizer: String::new(), spec: String::new(), bos: 2, eos: 3, offset: 4,
        }
    }

    #[test]
    fn the_prompt_layout() {
        let c = config();
        let p = build(&c, &[2, 9, 3], &Voice::Embedding(vec![0.0; 3]), &[Some(4), None], 3).unwrap();
        // speaker, 2 conditions, 3 text, 3 silence
        assert_eq!(p.rows, 9);
        assert_eq!(p.speaker.as_ref().unwrap().0, 0);
        assert_eq!(p.buckets, vec![(1, 4), (2, 30)]);
        let row = |r: usize| &p.ids[r * 11..r * 11 + 11];
        assert_eq!(row(3)[10], 2);
        assert!(row(3)[..10].iter().all(|v| *v == 4097));
        // silence: row t holds codebooks 0..=t
        assert_eq!(row(6)[0], 100);
        assert_eq!(row(6)[1], 4097);
        assert_eq!(row(8)[2], 102);
        assert_eq!(row(8)[3], 4097);
        assert_eq!(row(8)[10], 8196);
        assert!(p.audio_prefix);
        assert_eq!(p.prefill.len(), 3);
    }

    #[test]
    fn the_reference_prefix_is_delayed() {
        let c = config();
        let codes: Vec<[u32; 10]> = (0..2).map(|t| std::array::from_fn(|cb| (t * 10 + cb) as u32)).collect();
        let p = build(&c, &[2, 3], &Voice::Reference(codes), &[], 0).unwrap();
        // 2 frames + 9 rows, 2 conditions, 2 text
        assert_eq!(p.rows, 11 + 2 + 2);
        let row = |r: usize| &p.ids[r * 11..r * 11 + 11];
        assert_eq!(row(0)[0], 0);
        assert_eq!(row(0)[1], 4097);
        assert_eq!(row(1)[1], 1);
        assert_eq!(row(1)[0], 10);
        assert_eq!(row(10)[9], 19);
        assert_eq!(row(10)[8], 4097);
        assert!(!p.audio_prefix);
    }

    #[test]
    fn buckets_bisect_right() {
        let e = [1.0, 2.0, 3.0];
        assert_eq!(bucket(&e, 0.5), 0);
        assert_eq!(bucket(&e, 1.0), 1);
        assert_eq!(bucket(&e, 2.5), 2);
        assert_eq!(bucket(&e, 9.0), 3);
    }

    #[test]
    fn npy_codes_and_embeddings() {
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        let h = "{'descr': '<i8', 'fortran_order': False, 'shape': (1, 10), }";
        let mut head = h.to_string();
        while !(10 + head.len() + 1).is_multiple_of(64) {
            head.push(' ');
        }
        head.push('\n');
        b.extend((head.len() as u16).to_le_bytes());
        b.extend(head.as_bytes());
        for i in 0..10i64 {
            b.extend((i * 7).to_le_bytes());
        }
        let c = read_codes(&b).unwrap();
        assert_eq!(c, vec![[0, 7, 14, 21, 28, 35, 42, 49, 56, 63]]);
        assert!(read_embedding(&b).is_ok());
        assert!(read_npy(b"nope").is_err());
    }
}
