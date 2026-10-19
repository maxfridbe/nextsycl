//! The text encoder's tokenizer: Qwen2's byte-level BPE, as the reference uses it (transformers' `Qwen2Tokenizer`
//! through ComfyUI's MiniMax H3 tokenizer): no chat template, no start or end token - the prompt's own tokens.
//!
//! 1. NFC-normalize the text;
//! 2. split out the special tokens (written literally in the text, e.g. `<|caption_start|>`);
//! 3. cut the rest into pieces with Qwen's pattern (words with their leading space or punctuation, single digits,
//!    runs of punctuation, newlines, spaces);
//! 4. each piece's UTF-8 bytes, written as GPT-2's printable stand-in characters, merged pair by pair in the order
//!    of `merges.txt` until no listed pair is left; the pieces' ids from `vocab.json`.
//!
//! Steps 2-4 are the foundation's Qwen2 tokenizer (`nextsycl_tok::Tokenizer::from_vocab_merges`), with the tokens
//! H3 adds; the normalization is here (that crate has no Unicode tables).

use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

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

pub struct Tokenizer(nextsycl_tok::Tokenizer);

impl Tokenizer {
    /// From a directory with `vocab.json`, `merges.txt` and `tokenizer_config.json` (the repository's `tokenizer/`).
    pub fn load(dir: &Path) -> Result<Tokenizer> {
        let t = nextsycl_tok::Tokenizer::from_vocab_merges(dir).map_err(Error)?;
        if t.id("<|endoftext|>").is_none() {
            return Err(Error(format!("{}: tokenizer_config.json lists no added tokens (<|endoftext|>)", dir.display())));
        }
        Ok(Tokenizer(t.with_special(&MINIMAX_EXTRA)))
    }

    /// The token ids of `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let text: String = text.nfc().collect();
        Ok(self.0.encode(&text))
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
