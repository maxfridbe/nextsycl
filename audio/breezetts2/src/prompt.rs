//! A request's prompts, as upstream's templates build them (breeze_infer/templates.py) - one a guidance branch:
//!
//! ```text
//!   plain      [<bos>[S0]text]
//!   design     [<bos>[S0]<ins_bos>instruction<ins_eos>text]                        guided against plain
//!   clone      [<bos>[S0]ref_text] [the recording's frames, <|audio_eos|>] [<bos>[S0]text]
//!   direction  [<bos>[S0]ref_text] [the recording's frames, <|audio_eos|>] [<bos>[S0]<ins_bos>instruction<ins_eos>text]
//!                                                                                  guided against clone
//! ```
//!
//! A text segment is its own sequence through the text encoder (text.rs), projected to the backbone's width; a
//! recording's frame is the sum of its 16 codes' embeddings (codebook i's code c: row c + 2,051 i of the shared
//! table), the end of the recording the sum of code 0's.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;

/// A piece of a prompt
#[derive(Clone, Debug)]
pub enum Segment {
    Text(String),
    /// a recording's frames [F][16], followed by the end-of-audio row
    Audio(Vec<Vec<i32>>),
}

/// What a request asks for, checked
#[derive(Clone, Debug)]
pub struct Spec {
    pub text: String,
    pub instruction: Option<String>,
    /// a recording's transcript and frames
    pub reference: Option<(String, Vec<Vec<i32>>)>,
    /// guidance (used with an instruction: design and direction have a branch without it)
    pub cfg: f32,
}

pub const SPEAKER: &str = "[S0]";

impl Spec {
    /// The template's name (upstream's)
    pub fn template(&self) -> &'static str {
        match (&self.reference, &self.instruction) {
            (Some(_), Some(_)) => "ref_edit_tata",
            (Some(_), None) => "ref_clone_tata",
            (None, Some(_)) => "tts_instruction",
            (None, None) => "tts_plain",
        }
    }

    fn segments(&self, instruction: Option<&str>) -> Vec<Segment> {
        let text = match instruction {
            Some(i) => format!("{SPEAKER}<ins_bos>{i}<ins_eos>{}", self.text),
            None => format!("{SPEAKER}{}", self.text),
        };
        match &self.reference {
            Some((rt, codes)) => vec![Segment::Text(format!("{SPEAKER}{rt}")), Segment::Audio(codes.clone()), Segment::Text(text)],
            None => vec![Segment::Text(text)],
        }
    }

    /// The branches' prompts: the conditioned one, and with guidance the one without the instruction
    pub fn branches(&self) -> Vec<Vec<Segment>> {
        let mut b = vec![self.segments(self.instruction.as_deref())];
        if self.guided() {
            b.push(self.segments(None));
        }
        b
    }

    pub fn guided(&self) -> bool {
        self.instruction.is_some() && self.cfg != 1.0
    }
}

/// A text segment's ids: <bos>, then the text (its special tokens - [S0], <ins_bos> ... - as their ids)
pub fn text_ids(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let bos = tok.id("<bos>").ok_or_else(|| Error("the tokenizer has no <bos>".into()))?;
    let mut v = vec![bos];
    v.extend(tok.encode(text));
    Ok(v)
}

/// A prompt's token ids (<|AUDIO|> a frame, <|audio_eos|> after a recording), as upstream's input_ids
pub fn ids(tok: &Tokenizer, segs: &[Segment]) -> Result<Vec<u32>> {
    let (audio, eos) = (tok.id("<|AUDIO|>").ok_or_else(|| Error("no <|AUDIO|>".into()))?, tok.id("<|audio_eos|>").ok_or_else(|| Error("no <|audio_eos|>".into()))?);
    let mut v = Vec::new();
    for s in segs {
        match s {
            Segment::Text(t) => v.extend(text_ids(tok, t)?),
            Segment::Audio(f) => {
                v.extend(std::iter::repeat_n(audio, f.len()));
                v.push(eos);
            }
        }
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_and_branches() {
        let mut s = Spec { text: "hi".into(), instruction: None, reference: None, cfg: 4.0 };
        assert_eq!(s.template(), "tts_plain");
        assert_eq!(s.branches().len(), 1);
        s.instruction = Some("calm".into());
        assert_eq!(s.template(), "tts_instruction");
        let b = s.branches();
        assert_eq!(b.len(), 2);
        assert!(matches!(&b[0][0], Segment::Text(t) if t == "[S0]<ins_bos>calm<ins_eos>hi"));
        assert!(matches!(&b[1][0], Segment::Text(t) if t == "[S0]hi"));
        s.reference = Some(("ref".into(), vec![vec![0; 16]]));
        assert_eq!(s.template(), "ref_edit_tata");
        let b = s.branches();
        assert_eq!(b[1].len(), 3);
        assert!(matches!(&b[1][2], Segment::Text(t) if t == "[S0]hi"));
        s.cfg = 1.0;
        assert_eq!(s.branches().len(), 1);
    }
}
