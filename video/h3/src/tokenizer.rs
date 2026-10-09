//! The text encoder's tokenizer: Qwen2's byte-level BPE, as the reference uses it (transformers' `Qwen2Tokenizer`
//! through ComfyUI's MiniMax H3 tokenizer): no chat template, no start or end token - the prompt's own tokens.
//!
//! 1. NFC-normalize the text;
//! 2. split out the special tokens (written literally in the text, e.g. `<|caption_start|>`);
//! 3. cut the rest into pieces with Qwen's pattern (words with their leading space or punctuation, single digits,
//!    runs of punctuation, newlines, spaces);
//! 4. each piece's UTF-8 bytes, written as GPT-2's printable stand-in characters, merged pair by pair in the order
//!    of `merges.txt` until no listed pair is left; the pieces' ids from `vocab.json`.

use std::collections::HashMap;
use std::path::Path;

use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::{Ctx, Error, Result};

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The tokens the MiniMax H3 tokenizer adds to Qwen's, with their fixed ids.
const MINIMAX_EXTRA: [(&str, u32); 7] = [
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
];

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// literal text -> id, longest first
    special: Vec<(String, u32)>,
    pattern: Regex,
    byte_char: [char; 256],
}

/// GPT-2's map from bytes to printable characters: the printable Latin-1 bytes stand for themselves, the rest are
/// moved to U+0100 and up.
fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut n = 0;
    for b in 0..256u32 {
        out[b as usize] = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

impl Tokenizer {
    /// From a directory with `vocab.json`, `merges.txt` and `tokenizer_config.json` (the repository's `tokenizer/`).
    pub fn load(dir: &Path) -> Result<Tokenizer> {
        let vocab: HashMap<String, u32> =
            serde_json::from_slice(&std::fs::read(dir.join("vocab.json")).ctx("reading vocab.json")?).ctx("parsing vocab.json")?;
        let merges = std::fs::read_to_string(dir.join("merges.txt")).ctx("reading merges.txt")?;
        let mut ranks = HashMap::new();
        for (i, line) in merges.lines().filter(|l| !l.starts_with("#version") && !l.is_empty()).enumerate() {
            let (a, b) = line.split_once(' ').ok_or_else(|| Error(format!("merges.txt line {}: {line:?}", i + 1)))?;
            ranks.insert((a.to_string(), b.to_string()), i);
        }
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json")).ctx("reading tokenizer_config.json")?).ctx("parsing tokenizer_config.json")?;
        let mut special: Vec<(String, u32)> = cfg["added_tokens_decoder"]
            .as_object()
            .ok_or("tokenizer_config.json has no added_tokens_decoder")?
            .iter()
            .filter_map(|(id, t)| Some((t["content"].as_str()?.to_string(), id.parse().ok()?)))
            .collect();
        special.extend(MINIMAX_EXTRA.iter().map(|(s, id)| (s.to_string(), *id)));
        special.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        Ok(Tokenizer { vocab, ranks, special, pattern: Regex::new(PATTERN).map_err(|e| Error(e.to_string()))?, byte_char: bytes_to_unicode() })
    }

    /// The token ids of `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let text: String = text.nfc().collect();
        let mut out = Vec::new();
        let mut rest = text.as_str();
        while !rest.is_empty() {
            // the earliest special token (the longest at that position)
            let next = self.special.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id))).min_by_key(|(p, l, _)| (*p, std::cmp::Reverse(*l)));
            match next {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out)?;
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.pattern.find_iter(text) {
            let piece = m.map_err(|e| Error(e.to_string()))?.as_str();
            let mut parts: Vec<String> = piece.bytes().map(|b| self.byte_char[b as usize].to_string()).collect();
            // merge the best-ranked neighbouring pair until none is listed
            loop {
                let best = parts.windows(2).enumerate().filter_map(|(i, w)| self.ranks.get(&(w[0].clone(), w[1].clone())).map(|r| (*r, i))).min();
                let Some((_, i)) = best else { break };
                let merged = format!("{}{}", parts[i], parts[i + 1]);
                parts.splice(i..i + 2, [merged]);
            }
            for p in parts {
                out.push(*self.vocab.get(&p).ok_or_else(|| Error(format!("the piece {p:?} is not in the vocabulary")))?);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_tokenizer() {
        // the tokenizer's files (vocab.json, merges.txt, tokenizer_config.json: the catalog's minimax-h3/tokenizer)
        let Ok(dir) = std::env::var("NS_TEST_H3_TOKENIZER") else {
            eprintln!("skipped: NS_TEST_H3_TOKENIZER names no tokenizer directory");
            return;
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tok = Tokenizer::load(Path::new(&dir)).unwrap();
        // the reference's ids (reference/tokens.py): two prompts and a set of awkward strings
        let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("tests/data/tokens_reference.json")).unwrap()).unwrap();
        let mut bad = Vec::new();
        for (name, case) in fx.as_object().unwrap() {
            let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let text = case["text"].as_str().unwrap();
            let mut got = tok.encode(text).unwrap();
            if got.is_empty() {
                got.push(151643); // the reference's single pad token for an empty prompt
            }
            if got != want {
                bad.push(format!("{name} {text:?}: got {got:?}, want {want:?}"));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
