//! A request's prompt, as upstream's processor builds it (the -hf checkpoint's chat template; upstream's
//! VibeVoiceProcessor the same): a system line, the voices (each speaker's recording as its frames), the script a line
//! a turn, then the cue to speak:
//!
//! ```text
//!    Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n
//!    Voice input:\n                                                       (only with recordings)
//!    Speaker 0:<audio start><audio> x F0<audio end>\n
//!    Speaker 1:<audio start><audio> x F1<audio end>\n
//!    Text input:\n
//!    Speaker 0: Welcome back to the show.\n
//!    Speaker 1: Thanks for having me.\n
//!    Speech output:\n<audio start>
//! ```
//!
//! A script's speakers are its "Speaker N:" lines, numbered from 0 (upstream's parser: numbers from 1 are made
//! 0-based); a line without a speaker carries on the one before, and a text without any is speaker 0's. Recording i is
//! speaker i's voice. The text between two special tokens is one piece for the tokenizer, as the template's string
//! is split at its special tokens.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;

pub const SYSTEM: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

/// One turn of the script
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub speaker: usize,
    pub text: String,
}

/// The script's turns (see the module's notes)
pub fn parse(script: &str) -> Vec<Line> {
    let mut lines: Vec<Line> = Vec::new();
    for raw in script.replace("\\n", "\n").lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        match speaker_of(line) {
            Some((s, t)) => lines.push(Line { speaker: s, text: t.trim().to_string() }),
            None => match lines.last_mut() {
                Some(l) => l.text = format!("{} {line}", l.text).trim().to_string(),
                None => lines.push(Line { speaker: 0, text: line.to_string() }),
            },
        }
    }
    // numbered from 1 (none is 0): one down
    if lines.iter().all(|l| l.speaker > 0) {
        lines.iter_mut().for_each(|l| l.speaker -= 1);
    }
    lines
}

/// "Speaker N: text" (any case, spaces around the number and the colon) -> (N, text)
fn speaker_of(line: &str) -> Option<(usize, &str)> {
    let rest = line.get(..7).filter(|p| p.eq_ignore_ascii_case("speaker")).map(|_| &line[7..])?;
    let rest = rest.trim_start();
    let digits = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    if digits == 0 {
        return None;
    }
    let n = rest[..digits].parse().ok()?;
    let after = rest[digits..].trim_start().strip_prefix(':')?;
    Some((n, after))
}

/// The special tokens' ids (the checkpoint's config)
#[derive(Clone, Copy, Debug)]
pub struct Specials {
    pub audio_bos: u32,
    pub audio_eos: u32,
    pub audio: u32,
}

/// Each voiced speaker's place in the prompt, in prompt order: (speaker, first position, frames)
pub type Placed = Vec<(usize, usize, usize)>;

/// The prompt's token ids, the audio token where a voice's frame goes; `voices[s]`: speaker s's frames (None: no
/// recording). Returns the ids and where the voices go.
pub fn ids(tok: &Tokenizer, sp: Specials, lines: &[Line], voices: &[Option<usize>]) -> Result<(Vec<u32>, Placed)> {
    if lines.is_empty() {
        return Err(Error("the script is empty".into()));
    }
    let mut ids = Vec::new();
    let mut text = String::from(SYSTEM);
    let mut placed = Vec::new();
    // the voiced speakers in the order they first speak (the template's)
    let mut voiced: Vec<usize> = Vec::new();
    for l in lines {
        if voices.get(l.speaker).copied().flatten().is_some() && !voiced.contains(&l.speaker) {
            voiced.push(l.speaker);
        }
    }
    if !voiced.is_empty() {
        text.push_str(" Voice input:\n");
        for s in voiced {
            let frames = voices[s].unwrap_or(0);
            text.push_str(&format!(" Speaker {s}:"));
            ids.extend(tok.encode(&text));
            ids.push(sp.audio_bos);
            placed.push((s, ids.len(), frames));
            ids.extend(std::iter::repeat_n(sp.audio, frames));
            ids.push(sp.audio_eos);
            text = "\n".to_string();
        }
    }
    text.push_str(" Text input:\n");
    for l in lines {
        text.push_str(&format!(" Speaker {}: {}\n", l.speaker, l.text));
    }
    text.push_str(" Speech output:\n");
    ids.extend(tok.encode(&text));
    ids.push(sp.audio_bos);
    Ok((ids, placed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(s: usize, t: &str) -> Line {
        Line { speaker: s, text: t.into() }
    }

    #[test]
    fn scripts_parse_as_upstream() {
        assert_eq!(parse("Speaker 1: Hello there.\nSpeaker 2:  Hi!\n"), vec![l(0, "Hello there."), l(1, "Hi!")]);
        // already 0-based: kept
        assert_eq!(parse("speaker 0: a\\nSPEAKER 1 : b"), vec![l(0, "a"), l(1, "b")]);
        // a line without a speaker carries on; plain text is speaker 0's
        assert_eq!(parse("Speaker 1: one\ntwo\n\nSpeaker 2: three"), vec![l(0, "one two"), l(1, "three")]);
        assert_eq!(parse("Just say this."), vec![l(0, "Just say this.")]);
        // "Speakers" and "Speaker:" are text
        assert_eq!(parse("Speaker: x"), vec![l(0, "Speaker: x")]);
        assert!(parse("  \n ").is_empty());
    }
}
