//! The tokens around the model, on the host: the special ones and the languages (from tokenizer.json and
//! generation_config.json), the logit rules of openai-whisper's greedy decoding (suppress_blank, suppress_tokens,
//! ApplyTimestampRules), and a window's tokens cut into segments by their timestamp pairs (transcribe's loop).

use std::path::Path;

use nextsycl_audio::{Error, Result};
use nextsycl_tok::Tokenizer;
use serde_json::Value;

/// seconds a timestamp token stands for, and the encoder positions a second (frames / 2)
pub const TIME_PRECISION: f64 = 0.02;
pub const TOKENS_PER_SECOND: f64 = 50.0;

/// openai-whisper's language names by code (tokenizer.LANGUAGES), in the vocabulary's order
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"), ("zh", "chinese"), ("de", "german"), ("es", "spanish"), ("ru", "russian"), ("ko", "korean"), ("fr", "french"),
    ("ja", "japanese"), ("pt", "portuguese"), ("tr", "turkish"), ("pl", "polish"), ("ca", "catalan"), ("nl", "dutch"), ("ar", "arabic"),
    ("sv", "swedish"), ("it", "italian"), ("id", "indonesian"), ("hi", "hindi"), ("fi", "finnish"), ("vi", "vietnamese"), ("he", "hebrew"),
    ("uk", "ukrainian"), ("el", "greek"), ("ms", "malay"), ("cs", "czech"), ("ro", "romanian"), ("da", "danish"), ("hu", "hungarian"),
    ("ta", "tamil"), ("no", "norwegian"), ("th", "thai"), ("ur", "urdu"), ("hr", "croatian"), ("bg", "bulgarian"), ("lt", "lithuanian"),
    ("la", "latin"), ("mi", "maori"), ("ml", "malayalam"), ("cy", "welsh"), ("sk", "slovak"), ("te", "telugu"), ("fa", "persian"),
    ("lv", "latvian"), ("bn", "bengali"), ("sr", "serbian"), ("az", "azerbaijani"), ("sl", "slovenian"), ("kn", "kannada"), ("et", "estonian"),
    ("mk", "macedonian"), ("br", "breton"), ("eu", "basque"), ("is", "icelandic"), ("hy", "armenian"), ("ne", "nepali"), ("mn", "mongolian"),
    ("bs", "bosnian"), ("kk", "kazakh"), ("sq", "albanian"), ("sw", "swahili"), ("gl", "galician"), ("mr", "marathi"), ("pa", "punjabi"),
    ("si", "sinhala"), ("km", "khmer"), ("sn", "shona"), ("yo", "yoruba"), ("so", "somali"), ("af", "afrikaans"), ("oc", "occitan"),
    ("ka", "georgian"), ("be", "belarusian"), ("tg", "tajik"), ("sd", "sindhi"), ("gu", "gujarati"), ("am", "amharic"), ("yi", "yiddish"),
    ("lo", "lao"), ("uz", "uzbek"), ("fo", "faroese"), ("ht", "haitian creole"), ("ps", "pashto"), ("tk", "turkmen"), ("nn", "nynorsk"),
    ("mt", "maltese"), ("sa", "sanskrit"), ("lb", "luxembourgish"), ("my", "myanmar"), ("bo", "tibetan"), ("tl", "tagalog"), ("mg", "malagasy"),
    ("as", "assamese"), ("tt", "tatar"), ("haw", "hawaiian"), ("ln", "lingala"), ("ha", "hausa"), ("ba", "bashkir"), ("jw", "javanese"),
    ("su", "sundanese"), ("yue", "cantonese"),
];

/// openai-whisper's other names for some (TO_LANGUAGE_CODE's aliases)
const ALIASES: &[(&str, &str)] = &[
    ("burmese", "my"), ("valencian", "ca"), ("flemish", "nl"), ("haitian", "ht"), ("letzeburgesch", "lb"), ("pushto", "ps"), ("panjabi", "pa"),
    ("moldavian", "ro"), ("moldovan", "ro"), ("sinhalese", "si"), ("castilian", "es"), ("mandarin", "zh"),
];

/// The languages written without spaces between words: their words are cut at character boundaries
pub const UNSPACED: &[&str] = &["zh", "ja", "th", "lo", "my", "yue"];

/// The vocabulary's special tokens and the generation settings
pub struct Vocab {
    pub tok: Tokenizer,
    pub eot: u32,
    pub sot: u32,
    pub translate: u32,
    pub transcribe: u32,
    pub sot_prev: u32,
    pub no_speech: u32,
    pub no_timestamps: u32,
    /// <|0.00|>: timestamps are this and above
    pub timestamp_begin: u32,
    /// (code, token) in the vocabulary's order
    pub languages: Vec<(String, u32)>,
    /// suppressed throughout (the generation config's suppress_tokens: symbols and the specials)
    pub suppress: Vec<u32>,
    /// suppressed at the first token sampled (" " and <|endoftext|>)
    pub begin_suppress: Vec<u32>,
    /// the decoder's (layer, head) whose cross-attention follows the words in time
    pub alignment_heads: Vec<(usize, usize)>,
    /// the latest first timestamp (1.0 s: index 50)
    pub max_initial_timestamp: u32,
    pub vocab_size: usize,
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().into_iter().flatten().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()
}

impl Vocab {
    /// tokenizer.json and generation_config.json (`decoder_layers`: for the default alignment heads when the config
    /// names none - the last half of the decoder's layers)
    pub fn open(tokenizer: &Path, generation: &Path, decoder_layers: usize, heads: usize, vocab_size: usize) -> Result<Vocab> {
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        let g: Value = std::fs::read(generation).ok().and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| Error(format!("{}: not JSON", generation.display())))?;
        let need = |t: &str| tok.id(t).ok_or_else(|| Error(format!("{}: no token {t}", tokenizer.display())));
        let (eot, sot) = (need("<|endoftext|>")?, need("<|startoftranscript|>")?);
        let (translate, transcribe) = (need("<|translate|>")?, need("<|transcribe|>")?);
        let no_timestamps = need("<|notimestamps|>")?;
        let timestamp_begin = need("<|0.00|>")?;
        let sot_prev = need("<|startofprev|>")?;
        // large-v3's <|nospeech|>; the older vocabularies' <|nocaptions|>
        let no_speech = tok.id("<|nospeech|>").or_else(|| tok.id("<|nocaptions|>")).ok_or_else(|| Error("no <|nospeech|> token".into()))?;
        // the language tokens: <|xx|> between <|startoftranscript|> and <|translate|>
        let languages: Vec<(String, u32)> = (sot + 1..translate).filter_map(|i| {
            let t = tok.tokens.get(i as usize)?;
            Some((t.strip_prefix("<|")?.strip_suffix("|>")?.to_string(), i))
        }).collect();
        if languages.is_empty() {
            return Err(Error(format!("{}: no language tokens", tokenizer.display())));
        }
        let mut alignment_heads: Vec<(usize, usize)> = g["alignment_heads"].as_array().into_iter().flatten()
            .filter_map(|p| Some((p.get(0)?.as_u64()? as usize, p.get(1)?.as_u64()? as usize))).collect();
        if alignment_heads.is_empty() {
            alignment_heads = (decoder_layers / 2..decoder_layers).flat_map(|l| (0..heads).map(move |h| (l, h))).collect();
        }
        let mut suppress = ids(&g["suppress_tokens"]);
        // openai-whisper's own additions (already in the config's list for large-v3: kept once)
        for t in [translate, transcribe, sot, sot_prev, need("<|startoflm|>")?, no_speech] {
            if !suppress.contains(&t) {
                suppress.push(t);
            }
        }
        let mut begin_suppress = ids(&g["begin_suppress_tokens"]);
        for t in tok.encode(" ").into_iter().chain([eot]) {
            if !begin_suppress.contains(&t) {
                begin_suppress.push(t);
            }
        }
        let max_initial_timestamp = g["max_initial_timestamp_index"].as_u64().unwrap_or(50) as u32;
        Ok(Vocab { tok, eot, sot, translate, transcribe, sot_prev, no_speech, no_timestamps, timestamp_begin, languages, suppress, begin_suppress,
                   alignment_heads, max_initial_timestamp, vocab_size })
    }

    /// A language as asked (a code, a name, an alias, any case) as its code; None: "auto" or nothing
    pub fn language(&self, asked: Option<&str>) -> Result<Option<String>> {
        let Some(a) = asked.map(|a| a.trim().to_lowercase()).filter(|a| !a.is_empty() && a != "auto" && a != "unknown") else { return Ok(None) };
        let code = if self.languages.iter().any(|(c, _)| *c == a) {
            a.clone()
        } else if let Some((c, _)) = LANGUAGES.iter().find(|(_, n)| *n == a) {
            c.to_string()
        } else if let Some((_, c)) = ALIASES.iter().find(|(n, _)| *n == a) {
            c.to_string()
        } else {
            return Err(Error(format!("language {a}: not one Whisper writes (a code or a name: en, english, de, german, ...)")));
        };
        if !self.languages.iter().any(|(c, _)| *c == code) {
            return Err(Error(format!("language {code}: not in this model's vocabulary")));
        }
        Ok(Some(code))
    }

    pub fn language_token(&self, code: &str) -> Option<u32> {
        self.languages.iter().find(|(c, _)| c == code).map(|(_, t)| *t)
    }

    /// The prompt's start: <|startoftranscript|> <|lang|> <|transcribe|> (or <|translate|>)
    pub fn sot_sequence(&self, language: &str, translate: bool) -> Vec<u32> {
        let mut v = vec![self.sot];
        if let Some(t) = self.language_token(language) {
            v.push(t);
        }
        v.push(if translate { self.translate } else { self.transcribe });
        v
    }

    /// The text of tokens, timestamps and specials left out
    pub fn text(&self, tokens: &[u32]) -> String {
        let t: Vec<u32> = tokens.iter().copied().filter(|t| *t < self.eot).collect();
        self.tok.decode(&t)
    }
}

/// openai-whisper's logit filters for greedy decoding, in its order: SuppressBlank, SuppressTokens,
/// ApplyTimestampRules. `sampled`: the tokens sampled so far in this window (after the prompt).
pub fn apply_rules(logits: &mut [f32], sampled: &[u32], v: &Vocab, max_initial_timestamp: Option<u32>) {
    let ninf = f32::NEG_INFINITY;
    let n = logits.len();
    let tb = v.timestamp_begin as usize;
    if sampled.is_empty() {
        for t in &v.begin_suppress {
            if (*t as usize) < n {
                logits[*t as usize] = ninf;
            }
        }
    }
    for t in &v.suppress {
        if (*t as usize) < n {
            logits[*t as usize] = ninf;
        }
    }
    // timestamps
    if (v.no_timestamps as usize) < n {
        logits[v.no_timestamps as usize] = ninf;
    }
    let ts = |t: u32| t >= v.timestamp_begin;
    let last_was = sampled.last().is_some_and(|t| ts(*t));
    let penultimate_was = sampled.len() < 2 || ts(sampled[sampled.len() - 2]);
    if last_was {
        if penultimate_was {
            // a pair closed: text next
            logits[tb..].fill(ninf);
        } else {
            // a timestamp after text: another timestamp (the pair's end) or the end
            logits[..v.eot as usize].fill(ninf);
        }
    }
    if let Some(last) = sampled.iter().rev().find(|t| ts(**t)) {
        // timestamps do not go back; the one after a pair's end is past it
        let floor = if last_was && !penultimate_was { *last } else { *last + 1 };
        let floor = (floor as usize).min(n);
        logits[tb..floor].fill(ninf);
    }
    if sampled.is_empty() {
        // a window starts with a timestamp, at most max_initial_timestamp in
        logits[..tb].fill(ninf);
        if let Some(m) = max_initial_timestamp {
            let last_allowed = tb + m as usize;
            if last_allowed + 1 < n {
                logits[last_allowed + 1..].fill(ninf);
            }
        }
    }
    // a timestamp when the timestamps together are likelier than any text token
    let lse = |s: &[f32]| -> f64 {
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if m == f32::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        m as f64 + s.iter().map(|x| ((*x - m) as f64).exp()).sum::<f64>().ln()
    };
    let all = lse(logits);
    let ts_lp = lse(&logits[tb..]) - all;
    let text_max = logits[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64 - all;
    if ts_lp > text_max {
        logits[..tb].fill(ninf);
    }
}

/// The greedy choice and its log-probability (of the filtered logits, as openai-whisper sums them)
pub fn pick(logits: &[f32]) -> (u32, f64) {
    let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, x) in logits.iter().enumerate() {
        if *x > bv {
            bv = *x;
            best = i;
        }
    }
    let lse = bv as f64 + logits.iter().map(|x| ((*x - bv) as f64).exp()).sum::<f64>().ln();
    (best as u32, bv as f64 - lse)
}

/// The probability of `token` under the softmax of raw logits
pub fn prob(logits: &[f32], token: u32) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = logits.iter().map(|x| ((*x - m) as f64).exp()).sum();
    ((logits[token as usize] - m) as f64).exp() / z
}

/// A window's segments: (start, end in seconds from the window, its tokens timestamps included)
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    pub start: f64,
    pub end: f64,
    pub tokens: Vec<u32>,
}

/// transcribe's cut of a window's tokens (its <|endoftext|> dropped) into segments, and how many frames to move on:
/// pairs of timestamps close segments; a window ending on a single timestamp is done whole, else the next window
/// starts at its last timestamp. `segment_frames`: the window's frames that hold sound (at most 3000).
pub fn cut(tokens: &[u32], v: &Vocab, segment_frames: usize) -> (Vec<Cut>, usize) {
    let ts = |t: u32| t >= v.timestamp_begin;
    let pos = |t: u32| (t - v.timestamp_begin) as f64 * TIME_PRECISION;
    let n = tokens.len();
    let single_ending = n >= 2 && !ts(tokens[n - 2]) && ts(tokens[n - 1]);
    let mut consecutive: Vec<usize> = (1..n).filter(|&i| ts(tokens[i - 1]) && ts(tokens[i])).collect();
    let mut out = Vec::new();
    if !consecutive.is_empty() {
        if single_ending {
            consecutive.push(n);
        }
        let mut last = 0;
        for &c in &consecutive {
            let s = &tokens[last..c];
            out.push(Cut { start: pos(s[0]), end: pos(*s.last().unwrap()), tokens: s.to_vec() });
            last = c;
        }
        let advance = if single_ending {
            segment_frames
        } else {
            // the last timestamp: 2 frames a position
            (tokens[last - 1] - v.timestamp_begin) as usize * 2
        };
        (out, advance)
    } else {
        let duration = segment_frames as f64 / 100.0;
        let end = match tokens.iter().rev().find(|t| ts(**t)) {
            Some(t) if *t != v.timestamp_begin => pos(*t),
            _ => duration,
        };
        if !tokens.is_empty() {
            out.push(Cut { start: 0.0, end, tokens: tokens.to_vec() });
        }
        (out, segment_frames)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The real vocabulary (NS_TEST_WHISPER_TOKENIZER, NS_TEST_WHISPER_GENERATION: the model's tokenizer.json and
    /// generation_config.json); None without them
    pub fn vocab() -> Option<Vocab> {
        let (t, g) = (std::env::var("NS_TEST_WHISPER_TOKENIZER").ok()?, std::env::var("NS_TEST_WHISPER_GENERATION").ok()?);
        Some(Vocab::open(Path::new(&t), Path::new(&g), 4, 20, 51866).unwrap())
    }

    /// openai-whisper's greedy decoding of a window (NS_TEST_WHISPER_REF: a dump of reference/whisper/ref.py) is
    /// these rules over its own logits, token for token: logits.npy row r is the decoder after token r of tokens.npy
    #[test]
    fn greedy_under_the_rules_is_openais() {
        let (Some(v), Ok(dir)) = (vocab(), std::env::var("NS_TEST_WHISPER_REF")) else { return };
        let mut checked = 0;
        for clip in ["short", "long", "song"] {
            let d = Path::new(&dir).join(clip);
            let Ok((_, toks)) = crate::check::read_npy(&d.join("tokens.npy")) else { continue };
            let toks: Vec<u32> = toks.iter().map(|x| *x as u32).collect();
            let (_, l) = crate::check::read_npy(&d.join("logits.npy")).unwrap();
            let vs = v.vocab_size;
            let begin = 3;
            for i in 0..=toks.len() - begin {
                let mut row: Vec<f32> = l[(begin - 1 + i) * vs..(begin + i) * vs].iter().map(|x| *x as f32).collect();
                apply_rules(&mut row, &toks[begin..begin + i], &v, Some(v.max_initial_timestamp));
                let want = toks.get(begin + i).copied().unwrap_or(v.eot);
                assert_eq!(pick(&row).0, want, "{clip}: step {i}");
            }
            // and the window's segments: openai's first window cut, its seek
            let sampled = &toks[begin..];
            let (cuts, _) = cut(sampled, &v, 3000);
            assert!(!cuts.is_empty() && cuts.iter().all(|c| c.end >= c.start), "{clip}");
            checked += 1;
        }
        assert_eq!(checked, 3, "the three clips");
    }

    #[test]
    fn the_vocabulary_of_large_v3() {
        let Some(v) = vocab() else { return };
        assert_eq!((v.eot, v.sot, v.translate, v.transcribe, v.no_timestamps, v.timestamp_begin), (50257, 50258, 50359, 50360, 50364, 50365));
        assert_eq!(v.no_speech, 50363);
        assert_eq!(v.languages.len(), 100);
        // the vocabulary's order is openai-whisper's
        for ((c, t), (want, _)) in v.languages.iter().zip(LANGUAGES) {
            assert_eq!(c, want, "token {t}");
        }
        assert_eq!(v.alignment_heads, vec![(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)]);
        assert_eq!(v.sot_sequence("de", false), vec![50258, 50261, 50360]);
        assert_eq!(v.language(Some("German")).unwrap().as_deref(), Some("de"));
        assert_eq!(v.language(Some("mandarin")).unwrap().as_deref(), Some("zh"));
        assert_eq!(v.language(Some("auto")).unwrap(), None);
        assert!(v.language(Some("klingon")).is_err());
        assert!(v.begin_suppress.contains(&220) && v.begin_suppress.contains(&v.eot));
    }

    /// A small stand-in: text tokens below 10, eot 10, timestamps from 20 (<|0.00|> .. <|0.58|>)
    fn toy() -> Vocab {
        let tj = r#"{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2}, "merges": []}, "added_tokens": [], "pre_tokenizer": {"type": "ByteLevel"}}"#;
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!("nswhisper-toy-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("t.json"), tj).unwrap();
        let tok = Tokenizer::from_hf_json(&d.join("t.json")).unwrap();
        Vocab { tok, eot: 10, sot: 11, translate: 13, transcribe: 14, sot_prev: 15, no_speech: 16, no_timestamps: 17, timestamp_begin: 20,
                languages: vec![("en".into(), 12)], suppress: vec![3], begin_suppress: vec![2, 10], alignment_heads: vec![], max_initial_timestamp: 5,
                vocab_size: 50 }
    }

    #[test]
    fn a_window_starts_with_an_early_timestamp() {
        let v = toy();
        let mut l = vec![0f32; 50];
        l[0] = 9.0; // text is likelier, but a window starts with a timestamp
        apply_rules(&mut l, &[], &v, Some(v.max_initial_timestamp));
        assert!(l[..20].iter().all(|x| *x == f32::NEG_INFINITY));
        assert!(l[20..=25].iter().all(|x| x.is_finite()) && l[26..].iter().all(|x| *x == f32::NEG_INFINITY));
    }

    #[test]
    fn timestamps_come_in_pairs_and_never_go_back() {
        let v = toy();
        // after <|0.06|> text: text or a timestamp past 0.06 (text likelier here, so the timestamps do not win)
        let mut l = vec![0f32; 50];
        l[0] = 9.0;
        apply_rules(&mut l, &[23, 0], &v, None);
        assert!(l[20..24].iter().all(|x| *x == f32::NEG_INFINITY) && l[24].is_finite() && l[0].is_finite());
        // a timestamp after text closes the pair: another timestamp or the end, no text
        let mut l = vec![0f32; 50];
        l[10] = 9.0;
        apply_rules(&mut l, &[23, 0, 25], &v, None);
        assert!(l[..10].iter().all(|x| *x == f32::NEG_INFINITY) && l[10].is_finite(), "eot allowed");
        assert!(l[24].is_infinite() && l[25].is_finite(), "the pair's second may repeat the first");
        // a closed pair: text next
        let mut l = vec![0f32; 50];
        l[0] = 9.0;
        apply_rules(&mut l, &[23, 0, 25, 25], &v, None);
        assert!(l[20..].iter().all(|x| *x == f32::NEG_INFINITY) && l[0].is_finite());
        // no notimestamps, no suppressed token, ever
        assert!(l[17].is_infinite() && l[3].is_infinite());
    }

    #[test]
    fn timestamps_win_when_together_likelier() {
        let v = toy();
        let mut l = vec![-5f32; 50];
        l[0] = 0.0;
        for x in l[20..30].iter_mut() {
            *x = -0.5; // each below text, together above it
        }
        apply_rules(&mut l, &[23, 0], &v, None);
        assert!(l[..20].iter().all(|x| *x == f32::NEG_INFINITY));
        assert_eq!(pick(&l).0, 24, "the first allowed: past the last timestamp");
    }

    #[test]
    fn a_window_cut_into_segments() {
        let v = toy();
        // <|0.00|> a b <|0.10|><|0.10|> b <|0.30|>: two segments, a single timestamp ending: the whole window
        let t = [20, 0, 1, 25, 25, 1, 35];
        let (c, adv) = cut(&t, &v, 3000);
        assert_eq!(c, vec![Cut { start: 0.0, end: 0.1, tokens: vec![20, 0, 1, 25] }, Cut { start: 0.1, end: 0.3, tokens: vec![25, 1, 35] }]);
        assert_eq!(adv, 3000);
        // a pair at the end: the next window starts at its timestamp (2 frames a position)
        let t = [20, 0, 25, 25, 1, 30, 30];
        let (c, adv) = cut(&t, &v, 3000);
        assert_eq!(c.len(), 2);
        assert_eq!(adv, 20);
        // no pair: one segment to the last timestamp
        let (c, adv) = cut(&[20, 0, 1, 27], &v, 1234);
        assert_eq!(c, vec![Cut { start: 0.0, end: 0.14, tokens: vec![20, 0, 1, 27] }]);
        assert_eq!(adv, 1234);
    }
}
