//! The prompt's encoder: CLIP ViT-L/14's text tower as Stable Diffusion 1.5 ships it (inside the RF checkpoint) and
//! its tokenizer (byte-level BPE with an end-of-word mark, lower-cased, 77 tokens padded with the end token).

use std::collections::HashMap;
use std::path::Path;

use nextsycl_core::DevBuf;
use nextsycl_model3d::{Error, Result};

use crate::ffi::QUICK_GELU;
use crate::ops::{Lin, Ln, Ops, Weights};

pub const CTX: usize = 77;
pub const WIDTH: usize = 768;
const HEADS: usize = 12;
const LAYERS: usize = 12;
const VOCAB: usize = 49408;
const BOS: u32 = 49406;
const EOS: u32 = 49407;

/// CLIP's tokenizer
pub struct ClipTok {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_char: [char; 256],
}

/// GPT-2's map of bytes to printable characters (CLIP's byte_encoder)
fn bytes_to_chars() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        out[*b as usize] = char::from_u32(*c).unwrap_or('?');
    }
    out
}

impl ClipTok {
    /// From `vocab.json` and `merges.txt` (the first 48,894 merges, as CLIP reads them)
    pub fn open(vocab: &Path, merges: &Path) -> Result<ClipTok> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(vocab).map_err(|e| Error(format!("{}: {e}", vocab.display())))?)
            .map_err(|e| Error(format!("{}: {e}", vocab.display())))?;
        let vocab_map: HashMap<String, u32> = v.as_object().ok_or_else(|| Error(format!("{}: not a JSON object", vocab.display())))?
            .iter().filter_map(|(k, x)| Some((k.clone(), x.as_u64()? as u32))).collect();
        if vocab_map.len() != VOCAB {
            return Err(Error(format!("{}: {} tokens, CLIP has {VOCAB}", vocab.display(), vocab_map.len())));
        }
        let text = std::fs::read_to_string(merges).map_err(|e| Error(format!("{}: {e}", merges.display())))?;
        let ranks = text.lines().skip(1).take(49152 - 256 - 2).enumerate().filter_map(|(i, l)| {
            let (a, b) = l.split_once(' ')?;
            Some(((a.to_string(), b.to_string()), i))
        }).collect();
        Ok(ClipTok { vocab: vocab_map, ranks, byte_char: bytes_to_chars() })
    }

    /// One word's BPE pieces (its last symbol marked with </w>)
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(l) = syms.last_mut() {
            l.push_str("</w>");
        }
        loop {
            let best = (0..syms.len().saturating_sub(1))
                .filter_map(|i| self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())).map(|r| (*r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let (a, b) = (syms[i].clone(), syms[i + 1].clone());
            // every occurrence of the pair, left to right
            let mut out = Vec::with_capacity(syms.len());
            let mut k = 0;
            while k < syms.len() {
                if k + 1 < syms.len() && syms[k] == a && syms[k + 1] == b {
                    out.push(format!("{a}{b}"));
                    k += 2;
                } else {
                    out.push(syms[k].clone());
                    k += 1;
                }
            }
            syms = out;
        }
        syms
    }

    /// The words CLIP's pattern finds: contractions, runs of letters, single digits, runs of anything else that is
    /// not a space
    fn words(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let cs: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c == '\'' {
                let rest: String = cs[i + 1..cs.len().min(i + 3)].iter().collect();
                if let Some(t) = ["ll", "re", "ve", "s", "t", "m", "d"].iter().find(|t| rest.starts_with(**t)) {
                    out.push(format!("'{t}"));
                    i += 1 + t.len();
                    continue;
                }
            }
            if c.is_alphabetic() {
                let s = i;
                while i < cs.len() && cs[i].is_alphabetic() {
                    i += 1;
                }
                out.push(cs[s..i].iter().collect());
            } else if c.is_numeric() {
                out.push(c.to_string());
                i += 1;
            } else {
                let s = i;
                while i < cs.len() && !cs[i].is_whitespace() && !cs[i].is_alphabetic() && !cs[i].is_numeric() {
                    i += 1;
                }
                out.push(cs[s..i].iter().collect());
            }
        }
        out
    }

    /// The prompt as CLIP's 77 tokens: start, the text's (at most 75), end, then end tokens as padding
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![BOS];
        for w in Self::words(&clean) {
            let mapped: String = w.bytes().map(|b| self.byte_char[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                ids.push(*self.vocab.get(&piece).unwrap_or(&EOS));
            }
        }
        ids.truncate(CTX - 1);
        ids.push(EOS);
        ids.resize(CTX, EOS);
        ids
    }
}

struct Layer {
    ln1: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

/// The text tower
pub struct Clip {
    tok_emb: DevBuf,
    pos_emb: DevBuf,
    layers: Vec<Layer>,
    final_ln: Ln,
}

impl Clip {
    pub fn load(w: &Weights) -> Result<Clip> {
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("encoder.layers.{l}");
            layers.push(Layer {
                ln1: w.ln(&format!("{p}.layer_norm1"), WIDTH)?,
                q: w.lin(&format!("{p}.self_attn.q_proj"), WIDTH, WIDTH, true)?,
                k: w.lin(&format!("{p}.self_attn.k_proj"), WIDTH, WIDTH, true)?,
                v: w.lin(&format!("{p}.self_attn.v_proj"), WIDTH, WIDTH, true)?,
                o: w.lin(&format!("{p}.self_attn.out_proj"), WIDTH, WIDTH, true)?,
                ln2: w.ln(&format!("{p}.layer_norm2"), WIDTH)?,
                fc1: w.lin(&format!("{p}.mlp.fc1"), 4 * WIDTH, WIDTH, true)?,
                fc2: w.lin(&format!("{p}.mlp.fc2"), WIDTH, 4 * WIDTH, true)?,
            });
        }
        Ok(Clip {
            tok_emb: w.dev("embeddings.token_embedding.weight", &[VOCAB, WIDTH])?,
            pos_emb: w.dev("embeddings.position_embedding.weight", &[CTX, WIDTH])?,
            layers,
            final_ln: w.ln("final_layer_norm", WIDTH)?,
        })
    }

    /// The last hidden state [77, 768] of `ids` (77 tokens)
    pub fn encode(&self, o: &Ops, ids: &[u32]) -> Result<DevBuf> {
        if ids.len() != CTX {
            return Err(Error(format!("CLIP takes {CTX} tokens, not {}", ids.len())));
        }
        let idx: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = crate::ops::dev_i32(&o.gpu, &idx)?;
        let x = o.buf(CTX * WIDTH)?;
        o.k.gather(&self.tok_emb, &idx, CTX, WIDTH, &x)?;
        o.k.gated_add(&x, &self.pos_emb, CTX, WIDTH, 1, 1, None, None)?;
        let (h, q, k, v, a, t, f) = (o.buf(CTX * WIDTH)?, o.buf(CTX * WIDTH)?, o.buf(CTX * WIDTH)?, o.buf(CTX * WIDTH)?, o.buf(CTX * WIDTH)?,
                                     o.buf(CTX * WIDTH)?, o.buf(CTX * 4 * WIDTH)?);
        let scratch = o.buf(HEADS * CTX * CTX)?;
        let hd = WIDTH / HEADS;
        for l in &self.layers {
            o.ln(&x, CTX, WIDTH, 1e-5, l.ln1.norm(), &h)?;
            o.linear(&h, CTX, &l.q, &q)?;
            o.linear(&h, CTX, &l.k, &k)?;
            o.linear(&h, CTX, &l.v, &v)?;
            o.k.attention((&q, WIDTH, hd), (&k, WIDTH, hd), (&v, WIDTH, hd), 1, CTX, CTX, HEADS, hd, hd, 1.0 / (hd as f32).sqrt(), true, &scratch,
                          (&a, WIDTH))?;
            o.linear(&a, CTX, &l.o, &t)?;
            o.k.gated_add(&x, &t, CTX, WIDTH, 1, 1, None, None)?;
            o.ln(&x, CTX, WIDTH, 1e-5, l.ln2.norm(), &h)?;
            o.linear(&h, CTX, &l.fc1, &f)?;
            o.k.act(&f, CTX * 4 * WIDTH, QUICK_GELU)?;
            o.linear(&f, CTX, &l.fc2, &t)?;
            o.k.gated_add(&x, &t, CTX, WIDTH, 1, 1, None, None)?;
        }
        o.ln(&x, CTX, WIDTH, 1e-5, self.final_ln.norm(), &h)?;
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_split_as_clips_pattern() {
        assert_eq!(ClipTok::words("a dragon's wings, 2 flaps!!"), vec!["a", "dragon", "'s", "wings", ",", "2", "flaps", "!!"]);
        assert_eq!(ClipTok::words("x12"), vec!["x", "1", "2"]);
    }

    #[test]
    fn bpe_merges_by_rank() {
        let mut ranks = HashMap::new();
        ranks.insert(("a".to_string(), "b".to_string()), 1);
        ranks.insert(("ab".to_string(), "c</w>".to_string()), 0);
        ranks.insert(("b".to_string(), "c</w>".to_string()), 2);
        let t = ClipTok { vocab: HashMap::new(), ranks, byte_char: bytes_to_chars() };
        // (a, b) merges first (rank 1 < 2), then (ab, c</w>)
        assert_eq!(t.bpe("abc"), vec!["abc</w>"]);
        assert_eq!(bytes_to_chars()[b'a' as usize], 'a');
        assert_eq!(bytes_to_chars()[b' ' as usize], '\u{120}');
    }
}
