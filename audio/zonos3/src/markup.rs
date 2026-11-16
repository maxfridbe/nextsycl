//! The text a request says, as ZONOS3-Mini reads it: readable inline markup turned into the model's own markers, the
//! text normalized (Zyphra's zyphra-textnorm, the grammar of the request's language), then SuperBPE stretch by stretch.
//!
//! ```text
//!   [pause] [pause 0.25|0.5|1|2]   a pause of about that many seconds after a word (the nearest of the model's four
//!                                  buckets, 0.25 / 0.5 / 1 / 2 s): U+E003..U+E006
//!   [start]                        (first, once) speech from the first frame, no silence prefill: U+E002
//!   [ipa təmeɪtoʊ]                 a word said as the IPA written (the model's 136 phone symbols; stress marks,
//!                                  syllable dots and spaces dropped): U+E000 IPA U+E001
//!   [steer ID=DOSE, ...]words[/steer]  hidden-space steering over these words alone (per-word: emotions.*, mood.*,
//!                                  delivery.*, expressive.*, disfluency.*, emphasis_nv_nodur.emphasis), ramped in and
//!                                  out over 0.2 s
//! ```
//!
//! Any other bracketed text is ordinary text (said as written), as Zyphra's runtime treats angle-bracket tags. The
//! model's own markers (U+E000..U+E006) typed into the text pass through as they are.
//!
//! Origin: Zyphra's `zonos3.tts.superbpe.payload_ids`, `zonos3.tokenizer.textnorm` and `zonos3.tts.steering`
//! (`resolve_controls`' span schedule, `schedule_delta`) - github.com/Zyphra/ZONOS3-Mini, MIT. The markup syntax is
//! nextsycl's own.

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;

use crate::lm::Config;

pub const GROUP_OPEN: char = '\u{E000}';
pub const GROUP_CLOSE: char = '\u{E001}';
pub const START: char = '\u{E002}';
/// The markup the server's speech text pass leaves as written (AudioEngine::markup): the spans parse() reads, as the
/// card's annotations[].keep state them
pub const KEEP: &[&str] = &[r"\[pause(?: [0-9.]+s?)?\]", r"\[start\]", r"\[ipa [^\]]*\]", r"\[steer [^\]]*\]|\[/steer\]"];
pub const PAUSE0: char = '\u{E003}';
/// span boundaries while the text is normalized (not the model's: removed before encoding)
const SPAN_OPEN: char = '\u{E00B}';
const SPAN_CLOSE: char = '\u{E00C}';
/// the pause buckets' nominal lengths, seconds (U+E003 + i)
pub const PAUSE_SECONDS: [f32; 4] = [0.2475, 0.495, 0.9899, 1.9799];
/// frames a span ramps in and out over (0.2 s)
pub const RAMP_FRAMES: f32 = 3.0;
/// frames a character, unsteered (15 frames a second, ~15 characters a second)
pub const FRAMES_PER_CHAR: f32 = 1.0;

/// The phoneme / start / pause rows a backbone has (its text_tokenizer_spec)
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub phone_open: Option<u32>,
    pub phone_close: u32,
    pub symbols: Vec<String>,
    pub start: Option<u32>,
    pub pauses: Vec<u32>,
}

impl Spec {
    pub fn read(spec_json: &str) -> Spec {
        let j: serde_json::Value = serde_json::from_str(spec_json).unwrap_or_default();
        let mut s = Spec::default();
        if let Some(p) = j.get("phonemes").filter(|p| p.is_object()) {
            s.phone_open = p["open_id"].as_u64().map(|v| v as u32);
            s.phone_close = p["close_id"].as_u64().unwrap_or(0) as u32;
            s.symbols = p["symbols"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
        }
        s.start = j["start"]["id"].as_u64().map(|v| v as u32);
        s.pauses = j["pauses"]["ids"].as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
        s
    }
}

/// A control span: the characters of the spoken text it covers and its doses
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub controls: Vec<(String, f32)>,
}

/// Markup read: the text with the model's markers, its spans (boundaries as SPAN_OPEN / SPAN_CLOSE) and whether it
/// asked for the start marker
#[derive(Clone, Debug, Default)]
pub struct Parsed {
    pub text: String,
    pub spans: Vec<Vec<(String, f32)>>,
    pub start: bool,
}

fn parse_controls(s: &str) -> Result<Vec<(String, f32)>> {
    crate::steer::parse_controls(s)
}

/// Readable markup to the model's markers. Unknown bracket tags stay text.
pub fn parse(text: &str) -> Result<Parsed> {
    let mut out = Parsed::default();
    let mut rest = text;
    let mut open = false;
    let mut first = true;
    while let Some(i) = rest.find('[') {
        let Some(j) = rest[i..].find(']') else { break };
        let tag = &rest[i + 1..i + j];
        let low = tag.trim().to_lowercase();
        let before = &rest[..i];
        let mut handled = true;
        if low == "pause" || low.starts_with("pause ") {
            let secs: f32 = low.strip_prefix("pause").unwrap_or("").trim().trim_end_matches('s').parse().unwrap_or(0.5);
            let b = PAUSE_SECONDS.iter().enumerate().min_by(|a, c| (a.1.ln() - secs.max(0.05).ln()).abs().total_cmp(&(c.1.ln() - secs.max(0.05).ln()).abs()))
                .map_or(1, |x| x.0);
            out.text.push_str(before.trim_end());
            out.text.push(char::from_u32(PAUSE0 as u32 + b as u32).unwrap_or(PAUSE0));
            if !rest[i + j + 1..].starts_with(char::is_whitespace) && !rest[i + j + 1..].is_empty() {
                out.text.push(' ');
            }
        } else if low == "start" {
            if !first || !before.trim().is_empty() || out.start {
                return Err(Error("[start] goes first in the text, once".into()));
            }
            out.start = true;
        } else if let Some(ipa) = tag.trim().strip_prefix("ipa ").or_else(|| tag.trim().strip_prefix("IPA ")) {
            out.text.push_str(before);
            out.text.push(GROUP_OPEN);
            out.text.extend(ipa.chars().filter(|c| !matches!(c, 'ˈ' | 'ˌ' | '.' | ' ' | '/')));
            out.text.push(GROUP_CLOSE);
        } else if let Some(c) = low.strip_prefix("steer ").map(|_| &tag.trim()[6..]) {
            if open {
                return Err(Error("[steer ...] spans do not nest: close one with [/steer] first".into()));
            }
            out.text.push_str(before);
            out.spans.push(parse_controls(c)?);
            // normalize() wraps it as a one-character group, so the normalizer keeps it where it is
            out.text.push(SPAN_OPEN);
            open = true;
        } else if low == "/steer" {
            if !open {
                return Err(Error("[/steer] without a [steer ...]".into()));
            }
            out.text.push_str(before);
            out.text.push(SPAN_CLOSE);
            open = false;
        } else {
            handled = false;
        }
        if handled {
            rest = &rest[i + j + 1..];
        } else {
            out.text.push_str(&rest[..i + 1]);
            rest = &rest[i + 1..];
        }
        first = false;
    }
    out.text.push_str(rest);
    if open {
        return Err(Error("a [steer ...] span is not closed with [/steer]".into()));
    }
    if out.start {
        out.text.insert(0, START);
    }
    Ok(out)
}

/// A request's language as zyphra-textnorm's grammar code: a code (en, en_us, de-DE) or an English name; auto / none:
/// the script of the text decides (Latin: English)
pub fn grammar_language(language: Option<&str>, text: &str) -> String {
    let l = language.unwrap_or("").trim().to_lowercase();
    let names = [
        ("english", "en"), ("german", "de"), ("spanish", "es"), ("french", "fr"), ("italian", "it"), ("portuguese", "pt"), ("japanese", "ja"),
        ("chinese", "zh"), ("mandarin", "zh"), ("korean", "ko"), ("dutch", "nl"), ("polish", "pl"), ("turkish", "tr"), ("ukrainian", "uk"),
        ("czech", "cs"), ("danish", "da"), ("finnish", "fi"), ("greek", "el"), ("norwegian", "nb"), ("romanian", "ro"), ("slovak", "sk"),
        ("bulgarian", "bg"), ("hebrew", "he"), ("thai", "th"), ("tagalog", "tl"), ("malay", "ms"), ("georgian", "ka"), ("tamil", "ta"),
        ("gujarati", "gu"), ("kannada", "kn"), ("malayalam", "ml"), ("punjabi", "pa"), ("odia", "or"), ("russian", "ru"), ("swedish", "sv"),
        ("hungarian", "hu"), ("arabic", "ar"), ("vietnamese", "vi"), ("indonesian", "id"), ("hindi", "hi"), ("bengali", "bn"), ("telugu", "te"),
        ("marathi", "mr"), ("urdu", "ur"),
    ];
    if let Some((_, c)) = names.iter().find(|(n, _)| *n == l) {
        return c.to_string();
    }
    let code = l.split(['_', '-']).next().unwrap_or("").to_string();
    let code = if code == "no" { "nb".to_string() } else { code };
    if zyphra_textnorm::supported(&code) {
        return code;
    }
    // by script
    let count = |f: &dyn Fn(char) -> bool| text.chars().filter(|c| f(*c)).count();
    let scripts: [(&str, usize); 9] = [
        ("ja", count(&|c| ('\u{3040}'..='\u{30FF}').contains(&c))),
        ("zh", count(&|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))),
        ("ko", count(&|c| ('\u{AC00}'..='\u{D7AF}').contains(&c))),
        ("ru", count(&|c| ('\u{0400}'..='\u{04FF}').contains(&c))),
        ("hi", count(&|c| ('\u{0900}'..='\u{097F}').contains(&c))),
        ("ar", count(&|c| ('\u{0600}'..='\u{06FF}').contains(&c))),
        ("he", count(&|c| ('\u{0590}'..='\u{05FF}').contains(&c))),
        ("th", count(&|c| ('\u{0E00}'..='\u{0E7F}').contains(&c))),
        ("el", count(&|c| ('\u{0370}'..='\u{03FF}').contains(&c))),
    ];
    // kana marks Japanese even beside kanji
    if scripts[0].1 > 0 {
        return "ja".into();
    }
    scripts.iter().filter(|s| s.1 > 0).max_by_key(|s| s.1).map_or("en".to_string(), |s| s.0.to_string())
}

/// The normalizer over marked text: line breaks collapsed, the grammar of `lang`, the single markers (start, pauses,
/// span boundaries) wrapped as one-character groups so they stay where they are (as Zyphra's TTSTextNormalizer does)
pub fn normalize(text: &str, lang: &str) -> String {
    let single = |c: char| (START..='\u{E00C}').contains(&c);
    let mut wrapped = String::new();
    let mut chars = text.chars().peekable();
    let mut prev: Option<char> = None;
    while let Some(c) = chars.next() {
        let already = prev == Some(GROUP_OPEN) && chars.peek() == Some(&GROUP_CLOSE);
        if single(c) && !already {
            wrapped.extend([GROUP_OPEN, c, GROUP_CLOSE]);
        } else {
            wrapped.push(c);
        }
        prev = Some(c);
    }
    let out = if zyphra_textnorm::supported(lang) { zyphra_textnorm::normalize(&wrapped, lang) } else { zyphra_textnorm::collapse_line_breaks(&wrapped) };
    // unwrap the single markers again
    let v: Vec<char> = out.chars().collect();
    let mut s = String::new();
    let mut i = 0;
    while i < v.len() {
        if v[i] == GROUP_OPEN && i + 2 < v.len() && single(v[i + 1]) && v[i + 2] == GROUP_CLOSE {
            s.push(v[i + 1]);
            i += 3;
        } else {
            s.push(v[i]);
            i += 1;
        }
    }
    s
}

/// The span boundaries out of the text: the text the model reads and each span's character range in it
pub fn take_spans(text: &str, controls: &[Vec<(String, f32)>]) -> Result<(String, Vec<Span>)> {
    let mut out = String::new();
    let mut spans = Vec::new();
    let mut n = 0usize;
    let mut open: Option<usize> = None;
    for c in text.chars() {
        match c {
            SPAN_OPEN => open = Some(n),
            SPAN_CLOSE => {
                let s = open.take().ok_or_else(|| Error("a steer span lost its start in normalization".into()))?;
                let k = spans.len();
                let ctl = controls.get(k).cloned().ok_or_else(|| Error("more steer spans than given".into()))?;
                if n > s {
                    spans.push(Span { start: s, end: n, controls: ctl });
                } else {
                    // an empty span still takes its place in the order
                    spans.push(Span { start: s, end: s, controls: Vec::new() });
                }
            }
            _ => {
                out.push(c);
                n += 1;
            }
        }
    }
    spans.retain(|s| s.end > s.start && !s.controls.is_empty());
    Ok((out, spans))
}

/// The text ids (BOS, payload, EOS), as Zyphra's `superbpe.text_ids`: the start marker, the pause markers and the
/// phoneme groups as their rows, the stretches between them SuperBPE-encoded one by one
pub fn text_ids(tok: &Tokenizer, c: &Config, spec: &Spec, text: &str) -> Result<Vec<u32>> {
    let mut v = vec![c.bos];
    v.extend(payload(tok, c, spec, text)?);
    v.push(c.eos);
    Ok(v)
}

fn bpe(tok: &Tokenizer, c: &Config, s: &str) -> Vec<u32> {
    tok.encode_ordinary(s).into_iter().map(|i| i + c.offset).collect()
}

fn payload(tok: &Tokenizer, c: &Config, spec: &Spec, text: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    let mut stretch = String::new();
    let mut chars = text.chars().peekable();
    let mut at = 0usize;
    let flush = |s: &mut String, ids: &mut Vec<u32>| {
        if !s.is_empty() {
            ids.extend(bpe(tok, c, s));
            s.clear();
        }
    };
    while let Some(ch) = chars.next() {
        match ch {
            START => {
                let id = spec.start.ok_or_else(|| Error("this backbone has no start token (the edit backbone, backbone-ptq-edit.gguf, has one)".into()))?;
                if at != 0 {
                    return Err(Error("the start marker goes first, once".into()));
                }
                ids.push(id);
            }
            c0 if (PAUSE0..='\u{E00A}').contains(&c0) => {
                let k = c0 as usize - PAUSE0 as usize;
                let id = *spec.pauses.get(k).ok_or_else(|| Error(format!("pause marker U+{:04X} outside the backbone's {} pause buckets", c0 as u32, spec.pauses.len())))?;
                flush(&mut stretch, &mut ids);
                ids.push(id);
            }
            GROUP_OPEN => {
                let open = spec.phone_open.ok_or_else(|| Error("this backbone has no phoneme tokens (the edit backbone, backbone-ptq-edit.gguf, has them)".into()))?;
                flush(&mut stretch, &mut ids);
                ids.push(open);
                let mut closed = false;
                for p in chars.by_ref() {
                    if p == GROUP_CLOSE {
                        closed = true;
                        break;
                    }
                    let i = spec.symbols.iter().position(|s| s.chars().eq(std::iter::once(p)))
                        .ok_or_else(|| Error(format!("IPA symbol {p:?} is not one of the model's phones")))?;
                    ids.push(open + 2 + i as u32);
                }
                if !closed {
                    return Err(Error("an IPA group is not closed".into()));
                }
                ids.push(spec.phone_close);
            }
            GROUP_CLOSE => return Err(Error("an IPA group closes without opening".into())),
            _ => stretch.push(ch),
        }
        at += 1;
    }
    flush(&mut stretch, &mut ids);
    Ok(ids)
}

/// The span schedule (Zyphra's resolve_controls): each span's frames by character position at `rate` frames a
/// character, a span's own characters stretched by its pace factor
pub fn schedule(spans: &[Span], n_chars: usize, rate: f32, factor: &dyn Fn(&[(String, f32)]) -> f32) -> Vec<(f32, f32)> {
    let mut per = vec![rate; n_chars];
    for s in spans {
        let f = factor(&s.controls);
        for p in per.iter_mut().take(s.end.min(n_chars)).skip(s.start) {
            *p = rate * f;
        }
    }
    let mut at = vec![0f32; n_chars + 1];
    for i in 0..n_chars {
        at[i + 1] = at[i] + per[i];
    }
    spans.iter().map(|s| (at[s.start.min(n_chars)].round(), at[s.end.min(n_chars)].round())).collect()
}

/// A span's weight at generated frame `f`: 1 inside [start, end), ramping to 0 over RAMP_FRAMES either side
pub fn span_weight(start: f32, end: f32, f: f32) -> f32 {
    let r = RAMP_FRAMES.max(1.0);
    ((f - (start - r)) / r).min(((end + r) - f) / r).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_becomes_the_models_markers() {
        let p = parse("[start]Hello there.[pause 1] Say [ipa təˈmeɪ.toʊ] [steer emotions.happy=3]with joy[/steer] now [laugh].").unwrap();
        assert!(p.start);
        assert!(p.text.starts_with(START));
        assert!(p.text.contains('\u{E005}'));
        assert!(p.text.contains("\u{E000}təmeɪtoʊ\u{E001}"));
        assert!(p.text.contains("[laugh]"), "an unknown tag is text");
        assert_eq!(p.spans, vec![vec![("emotions.happy".to_string(), 3.0)]]);
        let (t, spans) = take_spans(&p.text, &p.spans).unwrap();
        assert_eq!(spans.len(), 1);
        let said: String = t.chars().skip(spans[0].start).take(spans[0].end - spans[0].start).collect();
        assert_eq!(said, "with joy");
        assert!(parse("[steer x=1]open").is_err());
        assert!(parse("Hi [start]").is_err());
    }

    /// The spans the server keeps are the card's annotations[].keep, and they keep every markup form whole
    #[test]
    fn the_kept_spans_are_the_cards() {
        let card = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../model-cards/audio/zonos3-mini.yml")).unwrap();
        for k in KEEP {
            assert!(card.contains(&format!("keep: '{k}'")), "the card states keep: '{k}'");
        }
        let res: Vec<fancy_regex::Regex> = KEEP.iter().map(|k| fancy_regex::Regex::new(k).unwrap()).collect();
        for e in ["[pause]", "[pause 2]", "[pause 0.25]", "[start]", "[ipa təmɑːtoʊ]", "[steer delivery.whisper=4]", "[/steer]"] {
            assert!(res.iter().any(|r| r.find(e).unwrap().is_some_and(|m| m.as_str() == e)), "{e} kept whole");
        }
        // and what the server's pass hands on (the kept spans set apart by spaces) parses
        parse("[start] It is. [pause 2] A [ipa təmɑːtoʊ] , [steer delivery.whisper=4] quiet [/steer] now. [pause]").unwrap();
        assert!(!res.iter().any(|r| r.is_match("[laugh] (laugh) <laugh/>").unwrap()), "no tags it does not read");
    }

    #[test]
    fn pauses_take_the_nearest_bucket() {
        for (s, want) in [("[pause]", '\u{E004}'), ("[pause 0.2]", '\u{E003}'), ("[pause 2]", '\u{E006}'), ("[pause 0.8s]", '\u{E005}')] {
            assert!(parse(&format!("a{s} b")).unwrap().text.contains(want), "{s}");
        }
    }

    #[test]
    fn markers_and_spans_survive_normalization() {
        let p = parse("It costs $5.[pause] [steer delivery.whisper=3]Keep it 2 yourself[/steer] ok").unwrap();
        let n = normalize(&p.text, "en");
        assert!(n.contains("five dollars"), "{n}");
        assert!(n.contains('\u{E004}'));
        let (t, spans) = take_spans(&n, &p.spans).unwrap();
        let said: String = t.chars().skip(spans[0].start).take(spans[0].end - spans[0].start).collect();
        assert!(said.to_lowercase().contains("keep it"), "{said}");
        assert!(!t.contains('\u{E00B}'));
    }

    #[test]
    fn languages_by_name_code_or_script() {
        assert_eq!(grammar_language(Some("English"), ""), "en");
        assert_eq!(grammar_language(Some("de_DE"), ""), "de");
        assert_eq!(grammar_language(Some("auto"), "Привет, мир"), "ru");
        assert_eq!(grammar_language(None, "こんにちは世界"), "ja");
        assert_eq!(grammar_language(None, "Hello"), "en");
    }

    #[test]
    fn the_schedule_places_spans_by_characters() {
        let spans = vec![Span { start: 10, end: 20, controls: vec![("delivery.tempo".into(), 2.0)] }];
        let s = schedule(&spans, 30, 1.0, &|_| 0.5);
        assert_eq!(s, vec![(10.0, 15.0)]);
        assert_eq!(span_weight(10.0, 15.0, 12.0), 1.0);
        assert_eq!(span_weight(10.0, 15.0, 7.0), 0.0);
        assert!((span_weight(10.0, 15.0, 8.5) - 0.5).abs() < 1e-6);
    }
}
