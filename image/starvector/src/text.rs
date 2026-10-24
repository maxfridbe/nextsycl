//! StarVector's text: the prompt it writes after ("<svg"), the stop sequence ("</svg>"), and the decoding of what it
//! wrote as its tokenizer (transformers' slow GPT2Tokenizer, StarCoder's byte-level BPE) decodes it with
//! `skip_special_tokens`: the special tokens (StarCoder's 19 and [PAD]) dropped, the three tokens StarVector adds
//! (<svg-start>, <image-start>, <caption-start>) as pieces of their own joined by spaces, bytes as UTF-8 with
//! replacement, then `clean_up_tokenization_spaces` (on in its tokenizer_config).

use std::path::Path;

use nextsycl_image::{Error, Result};
use nextsycl_tok::Tokenizer;

pub const PROMPT: &str = "<svg";
pub const STOP: &str = "</svg>";
/// StarCoder's own vocabulary; StarVector's four tokens come after it
pub const BASE_VOCAB: u32 = 49152;
pub const PAD: u32 = 49152;
pub const EOS: u32 = 0;
/// StarCoder's special tokens (<|endoftext|>, the fim and repository tokens)
const SPECIALS: u32 = 19;

pub struct Vocab {
    pub tok: Tokenizer,
    pub prompt: Vec<u32>,
    pub stop: Vec<u32>,
}

impl Vocab {
    pub fn open(tokenizer_json: &Path) -> Result<Vocab> {
        let tok = Tokenizer::from_hf_json(tokenizer_json).map_err(Error)?;
        let prompt = tok.encode(PROMPT);
        let stop = tok.encode(STOP);
        if prompt.is_empty() || stop.is_empty() {
            return Err(Error(format!("{}: \"<svg\" or \"</svg>\" encodes to nothing", tokenizer_json.display())));
        }
        Ok(Vocab { tok, prompt, stop })
    }

    fn special(id: u32) -> bool {
        id < SPECIALS || id == PAD
    }

    /// The text of `ids` as the model's tokenizer decodes it (skip_special_tokens, clean-up on)
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut run: Vec<u32> = Vec::new();
        for &id in ids {
            if Self::special(id) {
                continue;
            }
            if id > PAD {
                // an added token: a piece of its own
                if !run.is_empty() {
                    parts.push(self.tok.decode(&run));
                    run.clear();
                }
                parts.push(self.tok.tokens.get(id as usize).cloned().unwrap_or_default());
            } else {
                run.push(id);
            }
        }
        if !run.is_empty() {
            parts.push(self.tok.decode(&run));
        }
        clean_up(&parts.join(" "))
    }
}

/// transformers' clean_up_tokenization
pub fn clean_up(s: &str) -> String {
    s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't")
        .replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clean_up_is_transformers() {
        assert_eq!(clean_up("a , b . c ' d"), "a, b. c'd");
        assert_eq!(clean_up("<path d=\"M 1 .5\"/>"), "<path d=\"M 1.5\"/>", "it changes SVG too, as the model's own decode does");
    }

    #[test]
    fn starvectors_prompt_and_stop() {
        let Ok(p) = std::env::var("NS_TEST_STARVECTOR_TOKENIZER") else { return };
        let v = Vocab::open(Path::new(&p)).unwrap();
        assert_eq!(v.prompt, vec![46, 3672]);
        assert_eq!(v.stop, vec![377, 3672, 48]);
        assert_eq!(v.decode(&[46, 3672, 0, 49152]), "<svg");
    }
}
