//! The prompt: the checkpoint's special-token template around the cleaned description and the normalized lyrics
//! (diffusers' `_clean_caption` / `_normalize_lyrics`, line for line - even whitespace changes the song), and its
//! classifier-free twin (every token but the first and the last two replaced by `<|audio_cfg|>`).

use fancy_regex::{Captures, Regex};
use std::sync::OnceLock;

pub const AUDIO_CFG: u32 = 151654;

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("a valid pattern"))
}

fn sub(r: &Regex, text: &str, with: &dyn Fn(&Captures) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for c in r.captures_iter(text).flatten() {
        let m = c.get(0).expect("the whole match");
        out.push_str(&text[last..m.start()]);
        out.push_str(&with(&c));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn group1(c: &Captures) -> String {
    c.get(1).map_or(String::new(), |m| m.as_str().to_string())
}

/// Python's str.splitlines(): every line break it knows, no trailing empty line
fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let b: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < b.len() {
        let (at, c) = b[i];
        if matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&s[start..at]);
            let mut next = at + c.len_utf8();
            if c == '\r' && i + 1 < b.len() && b[i + 1].1 == '\n' {
                next += 1;
                i += 1;
            }
            start = next;
        }
        i += 1;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `_clean_caption`: `<|key value|>` tags as "key is value", markdown's headings, bullets, emphasis and rules gone
pub fn clean_caption(caption: &str) -> String {
    static TAG: OnceLock<Regex> = OnceLock::new();
    static HEAD: OnceLock<Regex> = OnceLock::new();
    static BUL: OnceLock<Regex> = OnceLock::new();
    static STAR: OnceLock<Regex> = OnceLock::new();
    static BOLD: OnceLock<Regex> = OnceLock::new();
    static EM: OnceLock<Regex> = OnceLock::new();
    static RULE: OnceLock<Regex> = OnceLock::new();
    static NL: OnceLock<Regex> = OnceLock::new();
    let text = sub(re(&TAG, r"<\|([^|]*)\|>"), caption, &|c| {
        let inner = group1(c);
        let inner = inner.trim();
        match inner.split_once(char::is_whitespace) {
            Some((a, b)) => format!("{a} is {}", b.trim_start()),
            None => inner.to_string(),
        }
    });
    let mut lines = Vec::new();
    for line in splitlines(&text) {
        let mut line = sub(re(&HEAD, r"^\s{0,3}#{1,6}\s+"), line, &|_| String::new());
        line = sub(re(&BUL, r"^\s*[*+-]\s+"), &line, &|_| String::new());
        line = sub(re(&STAR, r"^\s*\*\s+"), &line, &|_| String::new());
        while line.contains("**") {
            let updated = sub(re(&BOLD, r"\*\*([^*]+)\*\*"), &line, &group1);
            if updated == line {
                break;
            }
            line = updated;
        }
        line = sub(re(&EM, r"(?<!\*)\*([^*\n]+)\*(?!\*)"), &line, &group1);
        lines.push(line.trim_end().to_string());
    }
    let text = lines.join("\n");
    let text = sub(re(&RULE, r"(?m)^\s*[-*_]{3,}\s*$"), &text, &|_| String::new());
    let text = text.replace("• ", "").replace("    ", "");
    sub(re(&NL, r"\n{2,}"), &text, &|_| "\n".to_string())
}

/// `_normalize_lyrics`: a line's leading structure tags kept alone (the rest of that line dropped), tags on lines of
/// their own and lower case, "[start]" first
pub fn normalize_lyrics(lyrics: &str) -> String {
    static LEAD: OnceLock<Regex> = OnceLock::new();
    static TAGS: OnceLock<Regex> = OnceLock::new();
    let lead = re(&LEAD, r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)");
    let out: Vec<String> = lyrics.split('\n').map(|line| match lead.captures(line).ok().flatten() {
        Some(c) => group1(&c).trim().to_string(),
        None => line.to_string(),
    }).collect();
    let text = out.join("\n").replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n");
    let text = sub(re(&TAGS, r"\[([^\]]+)\]"), &text, &|c| format!("[{}]", group1(c).to_lowercase()));
    format!("[start]\n{text}")
}

/// The template the language model was trained on
pub fn text(caption: &str, lyrics: &str) -> String {
    format!("<|im_start|><|caption_start|>{}<|caption_end|><|lyrics_start|>{}<|lyrics_end|><|im_end|><|audio_start|>", clean_caption(caption),
            normalize_lyrics(lyrics))
}

/// The classifier-free twin of a prompt's ids
pub fn unconditional(ids: &[u32]) -> Vec<u32> {
    let n = ids.len();
    ids.iter().enumerate().map(|(i, id)| if i >= 1 && i + 2 < n { AUDIO_CFG } else { *id }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captions_lose_markdown_and_tags() {
        assert_eq!(clean_caption("## Genre\n- **Blues** rock\n* *slow*\n---\n<|bpm 92|>\n\n\nend"), "Genre\nBlues rock\nslow\nbpm is 92\nend");
        assert_eq!(clean_caption("a • b    c"), "a bc");
    }

    #[test]
    fn lyrics_keep_tags_on_their_own_lines() {
        assert_eq!(normalize_lyrics("[Verse] dropped\nline one [Chorus] two\n  [Intro][Solo]  x"), "[start]\n[verse]\nline one\n[chorus]\ntwo\n[intro][solo]");
        assert_eq!(unconditional(&[1, 2, 3, 4, 5]), vec![1, AUDIO_CFG, AUDIO_CFG, 4, 5]);
    }
}
