//! A request's prompt, as mistral-common's encode_speech_request builds it and vllm-omni's tts_preprocess fills it:
//!
//! ```text
//!   <s> [BEGIN_AUDIO] [AUDIO] x N [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
//! ```
//!
//! with the preset voice's N embedding rows (voice.<name> in the GGUF: upstream's voice_embedding/<name>.pt, each a
//! reference frame's 37 code embeddings summed) at the [AUDIO] positions and the token embedding everywhere else. The
//! text is Tekken (tiktoken BPE) from the vocabulary CrispASR keeps in the GGUF.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;

use crate::weights::Weights;

/// The special tokens the prompt names, by their text in Tekken's specials
pub const BOS: &str = "<s>";
pub const AUDIO: &str = "[AUDIO]";
pub const BEGIN_AUDIO: &str = "[BEGIN_AUDIO]";
pub const TEXT_TO_AUDIO: &str = "[NEXT_AUDIO_TEXT]";
pub const AUDIO_TO_TEXT: &str = "[REPEAT_AUDIO_TEXT]";

/// The ids of the prompt's special tokens
#[derive(Clone, Copy, Debug)]
pub struct Specials {
    pub bos: u32,
    pub audio: u32,
    pub begin_audio: u32,
    pub text_to_audio: u32,
    pub audio_to_text: u32,
}

impl Specials {
    pub fn of(names: &[String]) -> Result<Specials> {
        let id = |s: &str| names.iter().position(|n| n == s).map(|i| i as u32).ok_or_else(|| Error(format!("the tokenizer has no {s}")));
        Ok(Specials { bos: id(BOS)?, audio: id(AUDIO)?, begin_audio: id(BEGIN_AUDIO)?, text_to_audio: id(TEXT_TO_AUDIO)?, audio_to_text: id(AUDIO_TO_TEXT)? })
    }
}

/// The prompt's ids for a voice of `voice` rows (its rows go where [AUDIO] is)
pub fn ids(tok: &Tokenizer, sp: &Specials, text: &str, voice: usize) -> Vec<u32> {
    let mut v = vec![sp.bos, sp.begin_audio];
    v.extend(std::iter::repeat_n(sp.audio, voice));
    v.push(sp.text_to_audio);
    v.extend(tok.encode(text));
    v.push(sp.audio_to_text);
    v.push(sp.begin_audio);
    v
}

/// CrispASR's serialization of tekken.json's vocabulary (tokenizer.tekken.vocab_tensor: each piece a little-endian u16
/// length and its bytes, stored one byte a float32): the pieces in rank order
pub fn pieces(blob: &[f32]) -> Result<Vec<Vec<u8>>> {
    let b: Vec<u8> = blob.iter().map(|v| *v as u8).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if i + 2 > b.len() {
            return Err(Error("tokenizer.tekken.vocab_tensor: a piece's length cut off".into()));
        }
        let n = u16::from_le_bytes([b[i], b[i + 1]]) as usize;
        if i + 2 + n > b.len() {
            return Err(Error("tokenizer.tekken.vocab_tensor: a piece cut off".into()));
        }
        out.push(b[i + 2..i + 2 + n].to_vec());
        i += 2 + n;
    }
    Ok(out)
}

/// The tokenizer from the GGUF: Tekken's specials and pieces, the first vocab_size - specials in use (the table's
/// rows: tekken.json lists 150,000 pieces, the model embeds 131,072 ids)
pub fn tokenizer(w: &Weights) -> Result<(Tokenizer, Specials)> {
    let specials = w.texts("tokenizer.tekken.specials");
    let vocab = w.num("llm.vocab_size")? as usize;
    if specials.is_empty() || vocab <= specials.len() {
        return Err(Error("the GGUF carries no Tekken vocabulary (tokenizer.tekken.*)".into()));
    }
    let p = pieces(&w.all("tokenizer.tekken.vocab_tensor")?)?;
    let tok = Tokenizer::from_tekken(&p, &specials, vocab - specials.len()).map_err(Error)?;
    Ok((tok, Specials::of(&specials)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_and_the_vocabulary_blob() {
        let blob: Vec<f32> = [2u8, 0, b'h', b'i', 1, 0, b'x', 0, 0].iter().map(|b| *b as f32).collect();
        assert_eq!(pieces(&blob).unwrap(), vec![b"hi".to_vec(), b"x".to_vec(), vec![]]);
        assert!(pieces(&[5.0, 0.0, 1.0]).is_err());
        let names: Vec<String> = ["<unk>", "<s>", "</s>"].iter().map(|s| s.to_string()).collect::<Vec<_>>().into_iter()
            .chain((3..24).map(|i| format!("<SPECIAL_{i}>"))).chain([AUDIO, BEGIN_AUDIO].map(String::from))
            .chain((26..35).map(|i| format!("<SPECIAL_{i}>"))).chain([AUDIO_TO_TEXT, TEXT_TO_AUDIO].map(String::from)).collect();
        let sp = Specials::of(&names).unwrap();
        // mistral-common's ids: <s> 1, [AUDIO] 24, [BEGIN_AUDIO] 25, [REPEAT_AUDIO_TEXT] 35, [NEXT_AUDIO_TEXT] 36
        assert_eq!((sp.bos, sp.audio, sp.begin_audio, sp.audio_to_text, sp.text_to_audio), (1, 24, 25, 35, 36));
        let pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let tok = Tokenizer::from_tekken(&pieces, &names, 256).unwrap();
        let v = ids(&tok, &sp, "a", 2);
        assert_eq!(v, vec![1, 25, 24, 24, 36, names.len() as u32 + b'a' as u32, 35, 25]);
    }
}
