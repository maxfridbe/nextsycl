//! The text side, as upstream builds it (omnivoice/models/omnivoice.py, omnivoice/utils/{text,duration,voice_design,
//! lang_map}.py): the prompt's tokens, the voice-design attributes, the language, the length estimate and the long
//! text's chunks.
//!
//! The prompt (`_prepare_inference_inputs`): `[<|denoise|> when there is a recording] <|lang_start|>LANG<|lang_end|>
//! <|instruct_start|>ATTRIBUTES<|instruct_end|>` (`None` for a missing one), then `<|text_start|>TRANSCRIPT TEXT
//! <|text_end|>` with each non-verbal tag (`[laughter]` ...) tokenized on its own; each token repeated over the eight
//! codebooks; then the recording's codes and the target's mask codes.

use std::collections::HashMap;

use fancy_regex::Regex;
use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;
use unicode_normalization::UnicodeNormalization;

/// The non-verbal tags the model was trained on
pub const EVENTS: &[&str] = &[
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]", "[surprise-ah]",
    "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
];

/// upstream's docs/lang_id_name_map.tsv: id, name, ISO 639-3, hours of training data
const LANGUAGES: &str = include_str!("languages.tsv");

/// The voice-design attributes by category (English, Chinese); a category's items exclude each other
const GENDER: &[(&str, &str)] = &[("male", "男"), ("female", "女")];
const AGE: &[(&str, &str)] = &[("child", "儿童"), ("teenager", "少年"), ("young adult", "青年"), ("middle-aged", "中年"), ("elderly", "老年")];
const PITCH: &[(&str, &str)] = &[("very low pitch", "极低音调"), ("low pitch", "低音调"), ("moderate pitch", "中音调"), ("high pitch", "高音调"),
                                 ("very high pitch", "极高音调")];
const STYLE: &[(&str, &str)] = &[("whisper", "耳语")];
/// English only
pub const ACCENTS: &[&str] = &["american accent", "british accent", "australian accent", "chinese accent", "canadian accent", "indian accent", "korean accent",
                               "portuguese accent", "russian accent", "japanese accent"];
/// Chinese only
pub const DIALECTS: &[&str] = &["河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"];

/// The special tokens' ids (the file's metadata)
#[derive(Clone, Debug)]
pub struct Ids {
    pub denoise: u32,
}

pub fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn has_cjk(s: &str) -> bool {
    s.chars().any(is_cjk)
}

/// The languages: lower-case name -> id, and the ids
pub struct Languages {
    by_name: HashMap<String, String>,
    ids: Vec<String>,
    pub names: Vec<String>,
}

impl Languages {
    pub fn new() -> Languages {
        let mut by_name = HashMap::new();
        let mut ids = Vec::new();
        let mut names = Vec::new();
        for line in LANGUAGES.lines().skip(1) {
            let mut f = line.split('\t');
            let (Some(id), Some(name)) = (f.next(), f.next()) else { continue };
            by_name.insert(name.to_lowercase(), id.to_string());
            ids.push(id.to_string());
            names.push(name.to_lowercase());
        }
        Languages { by_name, ids, names }
    }

    /// upstream's _resolve_language: an id as it is, a name (any case) to its id; none, "none", "auto" or one it does
    /// not know: None (the model guesses from the text)
    pub fn resolve(&self, l: Option<&str>) -> Option<String> {
        let l = l?.trim();
        if l.is_empty() || l.eq_ignore_ascii_case("none") || l.eq_ignore_ascii_case("auto") {
            return None;
        }
        if self.ids.iter().any(|i| i == l) {
            return Some(l.to_string());
        }
        self.by_name.get(&l.to_lowercase()).cloned()
    }

    pub fn known(&self, l: &str) -> bool {
        self.resolve(Some(l)).is_some()
    }
}

impl Default for Languages {
    fn default() -> Self {
        Languages::new()
    }
}

/// upstream's _resolve_instruct: comma-separated attributes (`,` or `，`), each one it knows (any case), at most one a
/// category, English accents and Chinese dialects not together; all in Chinese when the text has Chinese (or a dialect
/// is named) and no accent is, else in English
pub fn resolve_instruct(instruct: &str, text: &str) -> Result<Option<String>> {
    let s = instruct.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let all: Vec<&str> = GENDER.iter().chain(AGE).chain(PITCH).chain(STYLE).flat_map(|(e, z)| [*e, *z]).chain(ACCENTS.iter().copied()).chain(DIALECTS.iter().copied())
        .collect();
    let mut items = Vec::new();
    let mut unknown = Vec::new();
    for raw in s.split([',', '，']).map(str::trim).filter(|x| !x.is_empty()) {
        let n = raw.to_lowercase();
        if all.contains(&n.as_str()) {
            items.push(n);
        } else {
            unknown.push(raw.to_string());
        }
    }
    if !unknown.is_empty() {
        return Err(Error(format!("instructions: {} not a voice attribute OmniVoice knows. It takes comma-separated attributes, at most one of each \
                                  kind: gender (male, female), age (child, teenager, young adult, middle-aged, elderly), pitch (very low pitch, low pitch, \
                                  moderate pitch, high pitch, very high pitch), whisper, an English accent ({}), a Chinese dialect ({}) - or their Chinese \
                                  names",
                                 unknown.iter().map(|u| format!("{u:?}")).collect::<Vec<_>>().join(", "), ACCENTS.join(", "), DIALECTS.join("，"))));
    }
    let dialect = items.iter().any(|n| n.ends_with('话'));
    let accent = items.iter().any(|n| n.contains(" accent"));
    if dialect && accent {
        return Err(Error("instructions: a Chinese dialect and an English accent together (dialects are for Chinese speech, accents for English)".into()));
    }
    let zh = if dialect { true } else if accent { false } else { has_cjk(text) };
    let pairs: Vec<(&str, &str)> = GENDER.iter().chain(AGE).chain(PITCH).chain(STYLE).copied().collect();
    let items: Vec<String> = items.into_iter().map(|n| {
        match pairs.iter().find(|(e, z)| *e == n || *z == n) {
            Some((e, z)) => if zh { z.to_string() } else { e.to_string() },
            None => n,
        }
    }).collect();
    for cat in [GENDER, AGE, PITCH, STYLE] {
        let hits: Vec<&String> = items.iter().filter(|n| cat.iter().any(|(e, z)| e == n || z == n)).collect();
        if hits.len() > 1 {
            return Err(Error(format!("instructions: {} are of the same kind (one each of gender, age, pitch, style, accent, dialect)",
                                     hits.iter().map(|h| format!("{h:?}")).collect::<Vec<_>>().join(" and "))));
        }
    }
    for set in [ACCENTS, DIALECTS] {
        let hits: Vec<&String> = items.iter().filter(|n| set.contains(&n.as_str())).collect();
        if hits.len() > 1 {
            return Err(Error(format!("instructions: {} are of the same kind (one each of gender, age, pitch, style, accent, dialect)",
                                     hits.iter().map(|h| format!("{h:?}")).collect::<Vec<_>>().join(" and "))));
        }
    }
    let sep = if items.iter().any(|n| has_cjk(n)) { "，" } else { ", " };
    Ok(Some(items.join(sep)))
}

/// upstream's _combine_text: the transcript and the text joined by a space, line breaks dropped, Chinese parentheses
/// made ASCII, runs of spaces and tabs one space, no spaces next to Chinese characters
pub fn combine(ref_text: Option<&str>, text: &str) -> String {
    let full = match ref_text.filter(|r| !r.is_empty()) {
        Some(r) => format!("{} {}", r.trim(), text.trim()),
        None => text.trim().to_string(),
    };
    let full: String = full.chars().filter(|c| *c != '\r' && *c != '\n').map(|c| match c {
        '\u{ff08}' => '(',
        '\u{ff09}' => ')',
        c => c,
    }).collect();
    let mut out = String::with_capacity(full.len());
    let mut space = false;
    for c in full.chars() {
        if c == ' ' || c == '\t' {
            space = true;
            continue;
        }
        if space {
            out.push(' ');
            space = false;
        }
        out.push(c);
    }
    if space {
        out.push(' ');
    }
    // spaces after or before a Chinese character go
    let cs: Vec<char> = out.chars().collect();
    let mut res = String::with_capacity(out.len());
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_whitespace() {
            let mut j = i;
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            let after_cjk = i > 0 && is_cjk(cs[i - 1]);
            let before_cjk = j < cs.len() && is_cjk(cs[j]);
            if !(after_cjk || before_cjk) {
                res.extend(&cs[i..j]);
            }
            i = j;
            continue;
        }
        res.push(cs[i]);
        i += 1;
    }
    res
}

/// upstream's add_punctuation: a full stop at the end of a transcript without one (Chinese's for Chinese)
pub fn add_punctuation(t: &str) -> String {
    let t = t.trim();
    if t.is_empty() {
        return String::new();
    }
    const END: &[char] = &[';', ':', ',', '.', '!', '?', '…', ')', ']', '}', '"', '\'', '“', '”', '‘', '’', '；', '：', '，', '。', '！', '？', '、', '）', '】'];
    if t.ends_with(END) {
        return t.to_string();
    }
    format!("{t}{}", if has_cjk(t) { "。" } else { "." })
}

/// The tokenizer: the base GGUF's vocabulary (Qwen2's byte-level BPE and split) with its added tokens matched in the
/// text, the text NFC-normalized first (as the checkpoint's tokenizer.json)
pub struct Text {
    pub tok: Tokenizer,
    pub ids: Ids,
    nonverbal: Regex,
}

impl Text {
    pub fn new(g: &nextsycl_gguf::Gguf) -> Result<Text> {
        let tok = Tokenizer::from_gguf_as(g, "qwen2").map_err(|e| Error(format!("the base GGUF's tokenizer: {e}")))?;
        // Qwen's added tokens (151643 <|endoftext|> to the file's last) are matched in the text as they are
        let first = tok.id("<|endoftext|>").ok_or_else(|| Error("the base GGUF's vocabulary has no <|endoftext|>: not Qwen's".into()))?;
        let n = g.meta("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map_or(0, |a| a.len()) as u32;
        let names: Vec<(String, u32)> = (first..n).filter_map(|i| {
            let s = tok.decode(&[i]);
            (!s.is_empty()).then_some((s, i))
        }).collect();
        let extra: Vec<(&str, u32)> = names.iter().map(|(s, i)| (s.as_str(), *i)).collect();
        let tok = tok.with_special(&extra);
        let id = |k: &str| -> Result<u32> {
            g.meta(&format!("omnivoice.special.{k}")).and_then(|v| v.as_u64()).map(|v| v as u32).ok_or_else(|| Error(format!("the base GGUF has no omnivoice.special.{k}")))
        };
        for (k, s) in [("denoise", "<|denoise|>"), ("lang_start", "<|lang_start|>"), ("text_start", "<|text_start|>"), ("text_end", "<|text_end|>")] {
            if tok.id(s) != Some(id(k)?) {
                return Err(Error(format!("the base GGUF's {s} is not its omnivoice.special.{k}")));
            }
        }
        let nonverbal = Regex::new(&format!("\\[({})\\]", EVENTS.iter().map(|e| e.trim_matches(['[', ']'])).collect::<Vec<_>>().join("|")))
            .map_err(|e| Error(e.to_string()))?;
        Ok(Text { tok, ids: Ids { denoise: id("denoise")? }, nonverbal })
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        self.tok.encode(&s.nfc().collect::<String>())
    }

    /// The style tokens: `<|denoise|>` (with a recording), the language, the attributes
    pub fn style(&self, denoise: bool, lang: Option<&str>, instruct: Option<&str>) -> Vec<u32> {
        let mut s = String::new();
        if denoise {
            s.push_str("<|denoise|>");
        }
        s.push_str(&format!("<|lang_start|>{}<|lang_end|><|instruct_start|>{}<|instruct_end|>", lang.unwrap_or("None"), instruct.unwrap_or("None")));
        self.encode(&s)
    }

    /// The text tokens: `<|text_start|>` the combined text `<|text_end|>`, each non-verbal tag tokenized on its own
    pub fn text(&self, ref_text: Option<&str>, text: &str) -> Vec<u32> {
        let wrapped = format!("<|text_start|>{}<|text_end|>", combine(ref_text, text));
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.nonverbal.find_iter(&wrapped).flatten() {
            if m.start() > last {
                out.extend(self.encode(&wrapped[last..m.start()]));
            }
            out.extend(self.encode(m.as_str()));
            last = m.end();
        }
        if last < wrapped.len() {
            out.extend(self.encode(&wrapped[last..]));
        }
        out
    }
}

/// upstream's RuleDurationEstimator: a character's speaking weight by script (1.0 a Latin letter)
pub struct Durations {
    mark: Regex,
    punct: Regex,
    space: Regex,
    digit: Regex,
}

/// (the last code point of a block, its weight)
const RANGES: &[(u32, f32)] = &[
    (0x02AF, 1.0), (0x03FF, 1.0), (0x052F, 1.0), (0x058F, 1.0), (0x05FF, 1.5), (0x077F, 1.5), (0x089F, 1.5), (0x08FF, 1.5), (0x097F, 1.8), (0x09FF, 1.8),
    (0x0A7F, 1.8), (0x0AFF, 1.8), (0x0B7F, 1.8), (0x0BFF, 1.8), (0x0C7F, 1.8), (0x0CFF, 1.8), (0x0D7F, 1.8), (0x0DFF, 1.8), (0x0EFF, 1.5), (0x0FFF, 1.8),
    (0x109F, 1.8), (0x10FF, 1.0), (0x11FF, 2.5), (0x137F, 3.0), (0x139F, 3.0), (0x13FF, 1.0), (0x167F, 1.0), (0x169F, 1.0), (0x16FF, 1.0), (0x171F, 1.0),
    (0x173F, 1.0), (0x175F, 1.0), (0x177F, 1.0), (0x17FF, 1.8), (0x18AF, 1.0), (0x18FF, 1.0), (0x194F, 1.8), (0x19DF, 1.8), (0x19FF, 1.8), (0x1A1F, 1.8),
    (0x1AAF, 1.8), (0x1B7F, 1.8), (0x1BBF, 1.8), (0x1BFF, 1.8), (0x1C4F, 1.8), (0x1C7F, 1.8), (0x1C8F, 1.0), (0x1CBF, 1.0), (0x1CCF, 1.8), (0x1CFF, 1.8),
    (0x1D7F, 1.0), (0x1DBF, 1.0), (0x1DFF, 1.0), (0x1EFF, 1.0), (0x309F, 2.2), (0x30FF, 2.2), (0x312F, 3.0), (0x318F, 2.5), (0x9FFF, 3.0), (0xA4CF, 3.0),
    (0xA4FF, 1.0), (0xA63F, 1.0), (0xA69F, 1.0), (0xA6FF, 1.0), (0xA7FF, 1.0), (0xA82F, 1.8), (0xA87F, 1.0), (0xA8DF, 1.8), (0xA8FF, 1.8), (0xA92F, 1.8),
    (0xA95F, 1.8), (0xA97F, 2.5), (0xA9DF, 1.8), (0xA9FF, 1.8), (0xAA5F, 1.8), (0xAA7F, 1.8), (0xAADF, 1.8), (0xAAFF, 1.8), (0xAB2F, 3.0), (0xAB6F, 1.0),
    (0xABBF, 1.0), (0xABFF, 1.8), (0xD7AF, 2.5), (0xFAFF, 3.0), (0xFDFF, 1.5), (0xFE6F, 1.0), (0xFEFF, 1.5), (0xFFEF, 1.0),
];

impl Durations {
    pub fn new() -> Durations {
        let re = |p: &str| Regex::new(p).expect("a Unicode category pattern");
        Durations { mark: re(r"^\p{M}$"), punct: re(r"^[\p{P}\p{S}]$"), space: re(r"^\p{Z}$"), digit: re(r"^\p{N}$") }
    }

    fn weight(&self, c: char) -> f32 {
        let code = c as u32;
        if c.is_ascii_alphabetic() {
            return 1.0;
        }
        if c == ' ' {
            return 0.2;
        }
        if code == 0x0640 {
            return 0.0;
        }
        let s = c.to_string();
        let is = |r: &Regex| r.is_match(&s).unwrap_or(false);
        if is(&self.mark) {
            return 0.0;
        }
        if is(&self.punct) {
            return 0.5;
        }
        if is(&self.space) {
            return 0.2;
        }
        if is(&self.digit) {
            return 3.5;
        }
        match RANGES.iter().find(|(end, _)| code <= *end) {
            Some((_, w)) => *w,
            None if code > 0x20000 => 3.0,
            None => 1.0,
        }
    }

    pub fn weight_of(&self, s: &str) -> f64 {
        s.chars().map(|c| self.weight(c) as f64).sum()
    }

    /// The target's frames for `text` at the pace of `ref_text` spoken in `ref_frames` (short ones boosted toward 50)
    pub fn estimate(&self, text: &str, ref_text: &str, ref_frames: f64) -> f64 {
        if ref_frames <= 0.0 || ref_text.is_empty() {
            return 0.0;
        }
        let rw = self.weight_of(ref_text);
        if rw == 0.0 {
            return 0.0;
        }
        let est = self.weight_of(text) / (rw / ref_frames);
        if est < 50.0 { 50.0 * (est / 50.0).powf(1.0 / 3.0) } else { est }
    }

    /// upstream's _estimate_target_tokens: with a recording at its pace, else at "Nice to meet you." in 25 frames
    pub fn target_frames(&self, text: &str, ref_text: Option<&str>, ref_frames: Option<usize>, speed: f64) -> usize {
        let (rt, rf) = match (ref_text.filter(|r| !r.is_empty()), ref_frames) {
            (Some(r), Some(f)) => (r, f as f64),
            _ => ("Nice to meet you.", 25.0),
        };
        let mut est = self.estimate(text, rt, rf);
        if speed > 0.0 && speed != 1.0 {
            est /= speed;
        }
        (est as usize).max(1)
    }
}

impl Default for Durations {
    fn default() -> Self {
        Durations::new()
    }
}

/// upstream's chunk_text_punctuation: sentences split at punctuation (not after a known abbreviation), merged up to
/// `chunk_len` characters, a chunk under `min_len` merged into its neighbour
pub fn chunks(text: &str, chunk_len: usize, min_len: usize) -> Vec<String> {
    const SPLIT: &[char] = &['.', ',', ';', ':', '!', '?', '。', '，', '；', '：', '！', '？'];
    const CLOSING: &[char] = &['"', '\'', '“', '”', '‘', '’', '）', ']', '》', '>', '」', '】'];
    const ABBR: &[&str] = &["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.",
                            "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.",
                            "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def."];
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    for c in text.chars() {
        if cur.is_empty() && !sentences.is_empty() && (SPLIT.contains(&c) || CLOSING.contains(&c)) {
            sentences.last_mut().expect("a sentence").push(c);
            continue;
        }
        cur.push(c);
        if SPLIT.contains(&c) {
            let mut abbr = false;
            if c == '.' {
                let s: String = cur.iter().collect();
                if let Some(last) = s.split_whitespace().last() {
                    abbr = ABBR.contains(&last);
                }
            }
            if !abbr {
                sentences.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    let mut merged: Vec<Vec<char>> = Vec::new();
    let mut chunk: Vec<char> = Vec::new();
    for s in sentences {
        if chunk.len() + s.len() <= chunk_len {
            chunk.extend(s);
        } else {
            if !chunk.is_empty() {
                merged.push(chunk);
            }
            chunk = s;
        }
    }
    if !chunk.is_empty() {
        merged.push(chunk);
    }
    let first_short = merged.first().is_some_and(|c| c.len() < min_len);
    let mut fin: Vec<Vec<char>> = Vec::new();
    for (i, c) in merged.into_iter().enumerate() {
        if i == 1 && first_short {
            fin.last_mut().expect("the first chunk").extend(c);
        } else if c.len() >= min_len || fin.is_empty() {
            fin.push(c);
        } else {
            fin.last_mut().expect("a chunk").extend(c);
        }
    }
    fin.into_iter().map(|c| c.into_iter().collect::<String>().trim().to_string()).filter(|s| !s.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_resolve_as_upstream() {
        assert_eq!(resolve_instruct("Female, Young Adult, british accent", "Hello").unwrap().unwrap(), "female, young adult, british accent");
        // Chinese text: the attributes in Chinese, full-width commas
        assert_eq!(resolve_instruct("male, low pitch", "你好").unwrap().unwrap(), "男，低音调");
        // a dialect makes it Chinese, an accent English
        assert_eq!(resolve_instruct("female，四川话", "hello").unwrap().unwrap(), "女，四川话");
        assert!(resolve_instruct("male, female", "x").is_err());
        assert!(resolve_instruct("sounds happy", "x").is_err());
        assert!(resolve_instruct("british accent, 四川话", "x").is_err());
        assert_eq!(resolve_instruct("  ", "x").unwrap(), None);
    }

    #[test]
    fn languages_and_text_as_upstream() {
        let l = Languages::new();
        assert_eq!(l.names.len(), 646);
        assert_eq!(l.resolve(Some("English")).as_deref(), Some("en"));
        assert_eq!(l.resolve(Some("en")).as_deref(), Some("en"));
        assert_eq!(l.resolve(Some("auto")), None);
        assert_eq!(l.resolve(Some("klingon")), None);
        assert_eq!(combine(Some(" Hi there "), "How\n are  you"), "Hi there How are you");
        assert_eq!(combine(None, "你好 世界（一）"), "你好世界(一)");
        assert_eq!(add_punctuation("hello"), "hello.");
        assert_eq!(add_punctuation("你好"), "你好。");
        assert_eq!(add_punctuation("done!"), "done!");
    }

    #[test]
    fn durations_follow_upstreams_weights() {
        let d = Durations::new();
        // "Nice to meet you." = 13 letters + 3 spaces + 1 full stop = 13 + 0.6 + 0.5
        assert!((d.weight_of("Nice to meet you.") - 14.1).abs() < 1e-6);
        assert!((d.weight_of("你好，世界！") - (4.0 * 3.0 + 2.0 * 0.5)).abs() < 1e-6);
        assert!((d.weight_of("12") - 7.0).abs() < 1e-6);
        // a long text at the default pace: weight / (14.1 / 25)
        let t = "a".repeat(100);
        assert_eq!(d.target_frames(&t, None, None, 1.0), (100.0 / (14.1 / 25.0)) as usize);
        // a short one boosted toward 50
        assert!(d.target_frames("Hi.", None, None, 1.0) > 20);
    }

    #[test]
    fn long_text_splits_at_sentences() {
        let c = chunks("Dr. Smith went home. It was late, very late! Then he slept.", 25, 3);
        assert_eq!(c, vec!["Dr. Smith went home.", "It was late, very late!", "Then he slept."]);
    }
}
