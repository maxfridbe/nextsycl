//! A request's prompt, as fish-speech's generate_long builds its Conversation (fish_speech/models/text2semantic/
//! inference.py, conversation.py, content_sequence.py):
//!
//! ```text
//!   <|im_start|>system\n convert the provided text to speech <|im_end|>\n                          (no recording)
//!   <|im_start|>system\n convert the provided text to speech reference to the following:\n\nText:\n
//!       <|speaker:0|>TRANSCRIPT \n\nSpeech:\n [the recording's frames] <|im_end|>\n                (a clone)
//!   <|im_start|>user\n TEXT <|im_end|>\n <|im_start|>assistant\n<|voice|>   -> the frames, ended by <|im_end|>
//! ```
//!
//! A text with speaker turns (`<|speaker:N|>`) is read a batch of turns at a time (up to five speakers and 200 bytes):
//! each batch a user turn, the frames made for it an assistant turn the next batch's prompt carries on from. Each part
//! is tokenized on its own, as upstream's TextParts are; a frame's row is its semantic id (151,678 + its first code)
//! and its ten codes.
//!
//! Instructions are S2 Pro's inline tags - free-form words in square brackets where they apply (`[whisper]`,
//! `[excited tone]`, `[pause]`); a request's `instructions` become one at the start of each turn.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;
use unicode_normalization::UnicodeNormalization;

/// A prompt row: a token id, and for a frame its codes (the slow model sums their embeddings)
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub token: u32,
    pub codes: Option<Vec<i32>>,
}

/// What a request asks for, checked
#[derive(Clone, Debug)]
pub struct Spec {
    pub text: String,
    pub instruction: Option<String>,
    /// a recording's transcript and frames [F][10]
    pub reference: Option<(String, Vec<Vec<i32>>)>,
}

/// upstream's batching of speaker turns (group_turns_into_batches(max_speakers=5, max_bytes=chunk_length 200))
const MAX_SPEAKERS: usize = 5;
const MAX_BYTES: usize = 200;

/// A speaker tag at the start of `s`: its length (`<|speaker:N|>`)
fn speaker_tag(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("<|speaker:")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    (digits > 0 && rest[digits..].starts_with("|>")).then_some("<|speaker:".len() + digits + 2)
}

/// upstream's split_text_by_speaker: the turns, each starting with its tag (text before the first tag is dropped, as
/// upstream drops it); none without tags
pub fn turns(text: &str) -> Vec<String> {
    let mut starts = Vec::new();
    for (i, _) in text.match_indices("<|speaker:") {
        if speaker_tag(&text[i..]).is_some() {
            starts.push(i);
        }
    }
    starts.iter().enumerate().map(|(n, &s)| {
        let e = starts.get(n + 1).copied().unwrap_or(text.len());
        // the tag and what follows it, stripped at both ends (as upstream: (part + parts[i + 1]).strip())
        text[s..e].trim().to_string()
    }).collect()
}

/// upstream's group_turns_into_batches: turns joined by newlines, a new batch past five speakers or 200 bytes
pub fn batches(turns: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut bytes = 0;
    for t in turns {
        let tb = t.len();
        if cur.len() >= MAX_SPEAKERS || (bytes + tb > MAX_BYTES && !cur.is_empty()) {
            out.push(cur.join("\n"));
            cur = vec![t];
            bytes = tb;
        } else {
            cur.push(t);
            bytes += tb;
        }
    }
    if !cur.is_empty() {
        out.push(cur.join("\n"));
    }
    out
}

impl Spec {
    /// The instruction as an inline tag ("[...]"; brackets the request already has are kept)
    fn tag(&self) -> Option<String> {
        self.instruction.as_ref().map(|i| {
            let i = i.trim();
            if i.starts_with('[') && i.ends_with(']') { i.to_string() } else { format!("[{i}]") }
        })
    }

    /// The text's batches, each a user turn: the turns' batches with speaker tags, else the whole text; the
    /// instruction's tag at the start of each turn
    pub fn batches(&self) -> Vec<String> {
        let tag = self.tag();
        let ts = turns(&self.text);
        if ts.is_empty() {
            let t = self.text.trim();
            return vec![match &tag {
                Some(g) => format!("{g} {t}"),
                None => t.to_string(),
            }];
        }
        let tagged: Vec<String> = ts.iter().map(|t| match (&tag, speaker_tag(t)) {
            (Some(g), Some(n)) => format!("{}{g} {}", &t[..n], t[n..].trim_start()),
            _ => t.clone(),
        }).collect();
        batches(&tagged)
    }
}

/// The tokenizer's ids of the prompt's special tokens
pub struct Ids {
    pub im_start: u32,
    pub im_end: u32,
    pub voice: u32,
    pub newline: u32,
    pub semantic0: u32,
}

impl Ids {
    pub fn of(tok: &Tokenizer) -> Result<Ids> {
        let id = |t: &str| tok.id(t).ok_or_else(|| Error(format!("the tokenizer has no {t}: not S2 Pro's")));
        let newline = tok.encode("\n");
        if newline.len() != 1 {
            return Err(Error("the tokenizer reads a newline as several ids: not S2 Pro's".into()));
        }
        Ok(Ids { im_start: id("<|im_start|>")?, im_end: id("<|im_end|>")?, voice: id("<|voice|>")?, newline: newline[0], semantic0: id("<|semantic:0|>")? })
    }
}

/// A part's ids (NFC-normalized first, as the checkpoint's tokenizer does)
pub fn text(tok: &Tokenizer, s: &str) -> Vec<Row> {
    tok.encode(&s.nfc().collect::<String>()).into_iter().map(|token| Row { token, codes: None }).collect()
}

/// Frames as rows: the semantic id of the first code, and the codes
pub fn frames(ids: &Ids, f: &[Vec<i32>]) -> Vec<Row> {
    f.iter().map(|c| Row { token: ids.semantic0 + c[0].max(0) as u32, codes: Some(c.clone()) }).collect()
}

/// The system turn (with a recording: its transcript and frames)
pub fn system(tok: &Tokenizer, ids: &Ids, reference: Option<&(String, Vec<Vec<i32>>)>) -> Vec<Row> {
    let mut r = text(tok, "<|im_start|>system\n");
    match reference {
        Some((t, codes)) => {
            r.extend(text(tok, "convert the provided text to speech reference to the following:\n\nText:\n"));
            let t = t.trim();
            r.extend(text(tok, &if t.contains("<|speaker:") { t.to_string() } else { format!("<|speaker:0|>{t}") }));
            r.extend(text(tok, "\n\nSpeech:\n"));
            r.extend(frames(ids, codes));
        }
        None => r.extend(text(tok, "convert the provided text to speech")),
    }
    r.extend(end(ids));
    r
}

/// The end of a turn: <|im_end|>\n
pub fn end(ids: &Ids) -> Vec<Row> {
    [ids.im_end, ids.newline].iter().map(|&token| Row { token, codes: None }).collect()
}

/// A user turn and the assistant's start: the rows after which the frames come
pub fn user(tok: &Tokenizer, ids: &Ids, batch: &str) -> Vec<Row> {
    let mut r = text(tok, "<|im_start|>user\n");
    r.extend(text(tok, batch));
    r.extend(end(ids));
    r.extend(text(tok, "<|im_start|>assistant\n"));
    r.push(Row { token: ids.voice, codes: None });
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_turns_split_and_batch_as_upstream() {
        assert!(turns("plain text").is_empty());
        let t = turns("intro <|speaker:0|> Hi there. <|speaker:1|>Hello!<|speaker:x|> not a tag");
        assert_eq!(t, vec!["<|speaker:0|> Hi there.".to_string(), "<|speaker:1|>Hello!<|speaker:x|> not a tag".to_string()]);
        let b = batches(&t);
        assert_eq!(b, vec!["<|speaker:0|> Hi there.\n<|speaker:1|>Hello!<|speaker:x|> not a tag".to_string()]);
        // a sixth turn, or one past 200 bytes, starts a new batch
        let six: Vec<String> = (0..6).map(|i| format!("<|speaker:{}|>a", i % 2)).collect();
        assert_eq!(batches(&six).len(), 2);
        let long = vec!["x".repeat(150), "y".repeat(100)];
        assert_eq!(batches(&long).len(), 2);
    }

    #[test]
    fn an_instruction_is_an_inline_tag_on_each_turn() {
        let mut s = Spec { text: "Hello.".into(), instruction: Some("whisper".into()), reference: None };
        assert_eq!(s.batches(), vec!["[whisper] Hello.".to_string()]);
        s.instruction = Some("[sad]".into());
        s.text = "<|speaker:0|>Hi. <|speaker:1|> Bye.".into();
        assert_eq!(s.batches(), vec!["<|speaker:0|>[sad] Hi.\n<|speaker:1|>[sad] Bye.".to_string()]);
        s.instruction = None;
        assert_eq!(s.batches(), vec!["<|speaker:0|>Hi.\n<|speaker:1|> Bye.".to_string()]);
    }
}
