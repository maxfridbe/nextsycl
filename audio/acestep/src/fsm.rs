//! The planner's constrained decoding (acestep/constrained_logits_processor.py MetadataConstrainedLogitsProcessor):
//! a state machine over the tokens that holds the LM to its training format -
//!
//! ```text
//!   <think>\nbpm: N\ncaption: ...\nduration: N\nkeyscale: K\nlanguage: L\ntimesignature: T\n</think>  then audio codes
//! ```
//!
//! fixed strings forced a token at a time (the longest prefix that is one token), numbers / keys / languages from
//! prefix trees of their own tokenizations, the caption free (no backticks, no audio codes, 512 tokens at most) until
//! the line after it starts a field, the user's fields injected token by token; the codes phase allows audio codes and
//! the end alone, the end held back until the song's codes are all there and then forced. The genres field and the
//! "understand" phase are not used by the platform and are not here.

use std::collections::{HashMap, VecDeque};

use nextsycl_tok::Tokenizer;

pub const MAX_AUDIO_CODE: u32 = 63_999;
pub const BPM: (u32, u32) = (30, 300);
pub const DURATION: (u32, u32) = (10, 600);
pub const TIME_SIGNATURES: [u32; 4] = [2, 3, 4, 6];
pub const LANGUAGES: [&str; 51] = ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is",
                                    "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta",
                                    "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown"];

/// The LM's tokenizer as the processor uses it: plain text (no "<": no added token can be in it) through a copy
/// without the 65 thousand added tokens, which the full one scans for at every call
pub struct LmTok {
    pub full: Tokenizer,
    plain: Tokenizer,
}

impl LmTok {
    pub fn from_hf_json(path: &std::path::Path) -> Result<LmTok, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let full = Tokenizer::from_hf_json_text(&text)?;
        let mut j: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        j["added_tokens"] = serde_json::Value::Array(Vec::new());
        let plain = Tokenizer::from_hf_json_text(&j.to_string())?;
        Ok(LmTok { full, plain })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.contains('<') { self.full.encode(text) } else { self.plain.encode(text) }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.full.decode(ids)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum State {
    ThinkTag,
    NewlineAfterThink,
    BpmName,
    BpmValue,
    CaptionName,
    CaptionValue,
    DurationName,
    DurationValue,
    KeyscaleName,
    KeyscaleValue,
    LanguageName,
    LanguageValue,
    TimesigName,
    TimesigValue,
    ThinkEndTag,
    Codes,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Bpm,
    Caption,
    Duration,
    Keyscale,
    Language,
    Timesig,
}

impl Field {
    fn name(self) -> &'static str {
        match self {
            Field::Bpm => "bpm",
            Field::Caption => "caption",
            Field::Duration => "duration",
            Field::Keyscale => "keyscale",
            Field::Language => "language",
            Field::Timesig => "timesignature",
        }
    }
}

type Tree = HashMap<Vec<u32>, Vec<u32>>;

/// The processor's tables (built once a model: the tokenizations it constrains by)
pub struct Tables {
    newline: u32,
    digits: [u32; 10],
    backtick: Option<u32>,
    eos: u32,
    /// token -> audio code (0 .. 63999), for the vocabulary
    pub code_of: Vec<Option<u32>>,
    think_end: Vec<u32>,
    bpm: Tree,
    duration: Tree,
    timesig: Tree,
    keyscale: Tree,
    language: Tree,
    fixed: HashMap<State, &'static str>,
}

fn last(v: Vec<u32>) -> Option<u32> {
    v.last().copied()
}

impl Tables {
    pub fn new(tok: &LmTok, vocab: usize) -> Result<Tables, String> {
        let newline = last(tok.encode("\n")).ok_or("no newline token")?;
        let mut digits = [0u32; 10];
        for (d, slot) in digits.iter_mut().enumerate() {
            *slot = last(tok.encode(&d.to_string())).ok_or("no digit token")?;
        }
        let eos = tok.full.id("<|im_end|>").ok_or("no <|im_end|> token")?;
        let mut code_of = vec![None; vocab];
        for (i, slot) in code_of.iter_mut().enumerate() {
            let t = tok.full.tokens.get(i).map(String::as_str).unwrap_or("");
            if let Some(n) = t.strip_prefix("<|audio_code_").and_then(|r| r.strip_suffix("|>")).and_then(|n| n.parse::<u32>().ok()) {
                if n <= MAX_AUDIO_CODE {
                    *slot = Some(n);
                }
            }
        }
        // a value's tokens after its field's "name:" (tokenized whole, with the space, as the model sees it)
        let tree = |field: &str, values: &[String], first_note: bool| -> Tree {
            let ctx = tok.encode(&format!("{field}:"));
            let mut t: Tree = HashMap::new();
            for v in values {
                let full = tok.encode(&format!("{field}: {v}"));
                if full.len() < ctx.len() || full[..ctx.len()] != ctx[..] {
                    continue;
                }
                let val = &full[ctx.len()..];
                if val.is_empty() {
                    continue;
                }
                if first_note {
                    let s = tok.decode(&val[..1]);
                    if !s.trim_start().chars().next().is_some_and(|c| "ABCDEFG".contains(c.to_ascii_uppercase())) {
                        continue;
                    }
                }
                for i in 0..=val.len() {
                    let e = t.entry(val[..i].to_vec()).or_default();
                    let next = if i < val.len() { val[i] } else { newline };
                    if !e.contains(&next) {
                        e.push(next);
                    }
                }
            }
            t
        };
        let nums = |a: u32, b: u32| (a..=b).map(|v| v.to_string()).collect::<Vec<_>>();
        let mut keys = Vec::new();
        for n in ["A", "B", "C", "D", "E", "F", "G"] {
            for a in ["", "#", "b", "♯", "♭"] {
                for m in ["major", "minor"] {
                    keys.push(format!("{n}{a} {m}"));
                }
            }
        }
        let fixed = HashMap::from([(State::ThinkTag, "<think>"), (State::NewlineAfterThink, "\n"), (State::BpmName, "bpm:"), (State::CaptionName, "caption:"),
                                   (State::DurationName, "duration:"), (State::KeyscaleName, "keyscale:"), (State::LanguageName, "language:"),
                                   (State::TimesigName, "timesignature:"), (State::ThinkEndTag, "</think>")]);
        Ok(Tables {
            newline,
            digits,
            backtick: last(tok.encode("`")),
            eos,
            code_of,
            think_end: tok.encode("</think>"),
            bpm: tree("bpm", &nums(BPM.0, BPM.1), false),
            duration: tree("duration", &nums(DURATION.0, DURATION.1), false),
            timesig: tree("timesignature", &TIME_SIGNATURES.iter().map(|v| v.to_string()).collect::<Vec<_>>(), false),
            keyscale: tree("keyscale", &keys, true),
            language: tree("language", &LANGUAGES.iter().map(|v| v.to_string()).collect::<Vec<_>>(), false),
            fixed,
        })
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }
}

/// Which phase the processor runs (the handler's generation_phase)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Cot,
    Codes,
}

pub struct Fsm<'a> {
    t: &'a Tables,
    tok: &'a LmTok,
    pub state: State,
    phase: Phase,
    skip_caption: bool,
    skip_language: bool,
    stop_at_reasoning: bool,
    user: HashMap<&'static str, String>,
    target_duration: Option<f64>,
    pub target_codes: Option<usize>,
    pub codes_count: usize,
    position: usize,
    acc: String,
    acc_ids: Vec<u32>,
    caption_after_newline: bool,
    caption_tokens: usize,
    caption_ending: bool,
    pending_field: String,
    queue: VecDeque<u32>,
    current_user: Option<Field>,
    /// the fixed strings' allowed tokens by (state, position): each an encode or more to find
    fixed_cache: HashMap<(State, usize), Vec<u32>>,
}

fn whitelist(scores: &mut [f32], allowed: &[u32]) {
    let keep: Vec<(u32, f32)> = allowed.iter().filter(|a| (**a as usize) < scores.len()).map(|a| (*a, scores[*a as usize])).collect();
    scores.fill(f32::NEG_INFINITY);
    for (a, v) in keep {
        scores[a as usize] = v;
    }
}

impl<'a> Fsm<'a> {
    /// `user`: the user's fields by name (bpm, caption, duration, keyscale, language, timesignature)
    #[allow(clippy::too_many_arguments)]
    pub fn new(t: &'a Tables, tok: &'a LmTok, phase: Phase, target_duration: Option<f64>, user: HashMap<&'static str, String>, stop_at_reasoning: bool,
               skip_caption: bool, skip_language: bool) -> Fsm<'a> {
        Fsm {
            t,
            tok,
            state: State::ThinkTag,
            phase,
            skip_caption,
            skip_language,
            stop_at_reasoning,
            user,
            target_codes: target_duration.filter(|d| *d > 0.0).map(|d| (d * 5.0) as usize),
            target_duration,
            codes_count: 0,
            position: 0,
            acc: String::new(),
            acc_ids: Vec::new(),
            caption_after_newline: false,
            caption_tokens: 0,
            caption_ending: false,
            pending_field: String::new(),
            queue: VecDeque::new(),
            current_user: None,
            fixed_cache: HashMap::new(),
        }
    }

    fn next_field(&self, f: Field) -> State {
        let order = [Field::Bpm, Field::Caption, Field::Duration, Field::Keyscale, Field::Language, Field::Timesig];
        let i = order.iter().position(|x| *x == f).unwrap_or(order.len());
        for g in order.iter().skip(i + 1) {
            match g {
                Field::Caption if self.skip_caption => continue,
                Field::Language if self.skip_language => continue,
                Field::Bpm => return State::BpmName,
                Field::Caption => return State::CaptionName,
                Field::Duration => return State::DurationName,
                Field::Keyscale => return State::KeyscaleName,
                Field::Language => return State::LanguageName,
                Field::Timesig => return State::TimesigName,
            }
        }
        State::ThinkEndTag
    }

    fn next_state(&self, s: State) -> Option<State> {
        Some(match s {
            State::ThinkTag => State::NewlineAfterThink,
            State::NewlineAfterThink => State::BpmName,
            State::BpmName => State::BpmValue,
            State::BpmValue => self.next_field(Field::Bpm),
            State::CaptionName if !self.skip_caption => State::CaptionValue,
            State::CaptionValue if !self.skip_caption => self.next_field(Field::Caption),
            State::DurationName => State::DurationValue,
            State::DurationValue => self.next_field(Field::Duration),
            State::KeyscaleName => State::KeyscaleValue,
            State::KeyscaleValue => self.next_field(Field::Keyscale),
            State::LanguageName if !self.skip_language => State::LanguageValue,
            State::LanguageValue if !self.skip_language => self.next_field(Field::Language),
            State::TimesigName => State::TimesigValue,
            State::TimesigValue => State::ThinkEndTag,
            State::ThinkEndTag => State::Codes,
            State::Codes => State::Completed,
            _ => return None,
        })
    }

    fn transition(&mut self) {
        if let Some(n) = self.next_state(self.state) {
            self.state = n;
            self.position = 0;
            self.acc.clear();
            self.acc_ids.clear();
            self.caption_after_newline = false;
            self.caption_tokens = 0;
            self.caption_ending = false;
            self.pending_field.clear();
        }
    }

    /// `_get_allowed_tokens_for_fixed_string`: the longest prefix of what is left that is one token (else the first
    /// tokens of its prefixes that match it)
    fn fixed_allowed(&mut self) -> Vec<u32> {
        let key = (self.state, self.position);
        if let Some(v) = self.fixed_cache.get(&key) {
            return v.clone();
        }
        let fixed = self.t.fixed[&self.state];
        let remaining: String = fixed.chars().skip(self.position).collect();
        let chars: Vec<char> = remaining.chars().collect();
        let mut out = Vec::new();
        for end in (1..=chars.len()).rev() {
            let p: String = chars[..end].iter().collect();
            let t = self.tok.encode(&p);
            if t.len() == 1 {
                out = t;
                break;
            }
        }
        if out.is_empty() && !chars.is_empty() {
            let mut best: Vec<(u32, usize)> = Vec::new();
            for end in 1..=chars.len().min(19) {
                let p: String = chars[..end].iter().collect();
                if let Some(first) = self.tok.encode(&p).first().copied() {
                    let d = self.tok.decode(&[first]).trim_start().to_lowercase();
                    let np = p.trim_start().to_lowercase();
                    if d.starts_with(&np) || np.starts_with(&d) {
                        match best.iter_mut().find(|(t, _)| *t == first) {
                            Some(b) => b.1 = b.1.max(end),
                            None => best.push((first, end)),
                        }
                    }
                }
            }
            best.sort_by(|a, b| b.1.cmp(&a.1));
            out = best.into_iter().map(|(t, _)| t).collect();
        }
        self.fixed_cache.insert(key, out.clone());
        out
    }

    fn user_value(&mut self, f: Field) -> Option<Vec<u32>> {
        let v = self.user.get(f.name())?;
        Some(self.tok.encode(&format!(" {v}\n")))
    }

    /// A field's user value starts its injection (its first token whitelisted); true when it did
    fn inject(&mut self, f: Field, scores: &mut [f32]) -> bool {
        if !self.queue.is_empty() {
            return false;
        }
        let fresh = match f {
            Field::Caption => self.acc.is_empty(),
            _ => self.acc_ids.is_empty(),
        };
        if !fresh {
            return false;
        }
        match self.user_value(f) {
            Some(v) if !v.is_empty() => {
                whitelist(scores, &v[..1]);
                self.queue = v.into();
                self.current_user = Some(f);
                true
            }
            _ => false,
        }
    }

    /// The scores of the next token constrained (the processor's __call__, before top-p and the temperature);
    /// `input`: the sequence so far (the codes phase looks for </think> in it)
    pub fn process(&mut self, input: &[u32], scores: &mut [f32]) {
        if self.state == State::Completed {
            return;
        }
        if self.phase == Phase::Codes && self.state == State::ThinkTag && !self.t.think_end.is_empty()
            && input.windows(self.t.think_end.len()).any(|w| w == self.t.think_end.as_slice()) {
            self.state = State::Codes;
            self.codes_count = 0;
        }
        if self.state == State::Codes {
            let eos = self.t.eos as usize;
            for (i, s) in scores.iter_mut().enumerate() {
                if i != eos && self.t.code_of.get(i).is_none_or(|c| c.is_none()) {
                    *s = f32::NEG_INFINITY;
                }
            }
            if let Some(target) = self.target_codes {
                if self.codes_count < target {
                    scores[eos] = f32::NEG_INFINITY;
                } else {
                    let e = scores[eos];
                    scores.fill(f32::NEG_INFINITY);
                    scores[eos] = e;
                }
            }
            return;
        }
        self.process_single(scores);
    }

    fn process_single(&mut self, scores: &mut [f32]) {
        if let Some(n) = self.queue.front().copied() {
            whitelist(scores, &[n]);
            return;
        }
        let nl = self.t.newline;
        let eos = self.t.eos;
        if self.t.fixed.contains_key(&self.state) {
            let fixed_len = self.t.fixed[&self.state].chars().count();
            let allowed = self.fixed_allowed();
            if !allowed.is_empty() {
                if self.state == State::ThinkEndTag && self.stop_at_reasoning && fixed_len - self.position.min(fixed_len) <= 10 {
                    whitelist(scores, &[eos]);
                    return;
                }
                whitelist(scores, &allowed);
            } else {
                if self.state == State::ThinkEndTag && self.stop_at_reasoning {
                    whitelist(scores, &[eos]);
                    return;
                }
                self.transition();
                if self.t.fixed.contains_key(&self.state) {
                    return;
                }
                scores.fill(0.0);
                self.process_single(scores);
            }
            return;
        }
        let tree_allowed = |tree: &Tree, ids: &[u32]| -> (Vec<u32>, bool) {
            match tree.get(ids) {
                Some(v) => (v.clone(), v.contains(&nl)),
                None => (Vec::new(), false),
            }
        };
        match self.state {
            State::BpmValue => {
                if self.inject(Field::Bpm, scores) {
                    return;
                }
                let (mut allowed, done) = tree_allowed(&self.t.bpm, &self.acc_ids);
                if done {
                    allowed.push(nl);
                }
                whitelist(scores, &allowed);
            }
            State::CaptionValue => {
                if self.inject(Field::Caption, scores) {
                    return;
                }
                if self.caption_after_newline {
                    let top = scores.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
                    let text = self.tok.decode(&[top as u32]);
                    self.caption_after_newline = false;
                    if text.chars().next().is_some_and(|c| c != ' ' && c != '\t') {
                        self.caption_ending = true;
                        self.pending_field.clear();
                        return;
                    }
                }
                if self.caption_ending {
                    return;
                }
                if let Some(b) = self.t.backtick {
                    scores[b as usize] = f32::NEG_INFINITY;
                }
                for (i, s) in scores.iter_mut().enumerate() {
                    if self.t.code_of.get(i).is_some_and(|c| c.is_some()) {
                        *s = f32::NEG_INFINITY;
                    }
                }
                if self.caption_tokens >= 512 {
                    whitelist(scores, &[nl]);
                }
            }
            State::DurationValue => {
                if self.inject(Field::Duration, scores) {
                    return;
                }
                if let Some(d) = self.target_duration {
                    let target = (d as i64).to_string();
                    let at = self.acc.chars().count();
                    match target.chars().nth(at).and_then(|c| c.to_digit(10)) {
                        Some(dg) => whitelist(scores, &[self.t.digits[dg as usize]]),
                        None => whitelist(scores, &[nl]),
                    }
                } else {
                    let (mut allowed, done) = tree_allowed(&self.t.duration, &self.acc_ids);
                    if done {
                        allowed.push(nl);
                    }
                    whitelist(scores, &allowed);
                }
            }
            State::KeyscaleValue => {
                if self.inject(Field::Keyscale, scores) {
                    return;
                }
                let (allowed, done) = tree_allowed(&self.t.keyscale, &self.acc_ids);
                if done || allowed.is_empty() {
                    whitelist(scores, &[nl]);
                } else {
                    whitelist(scores, &allowed);
                }
            }
            State::LanguageValue => {
                if self.inject(Field::Language, scores) {
                    return;
                }
                if self.acc_ids.is_empty() {
                    // the most likely language's first token, the rest of it forced
                    match self.t.language.get(&Vec::new()).filter(|c| !c.is_empty()) {
                        Some(c) => {
                            let top = c.iter().copied().fold((c[0], f32::NEG_INFINITY), |b, t| if scores[t as usize] > b.1 { (t, scores[t as usize]) } else { b }).0;
                            whitelist(scores, &[top]);
                        }
                        None => whitelist(scores, &[nl]),
                    }
                } else {
                    let (allowed, done) = tree_allowed(&self.t.language, &self.acc_ids);
                    if done || allowed.is_empty() {
                        whitelist(scores, &[nl]);
                    } else {
                        whitelist(scores, &allowed);
                    }
                }
            }
            State::TimesigValue => {
                if self.inject(Field::Timesig, scores) {
                    return;
                }
                let (allowed, done) = tree_allowed(&self.t.timesig, &self.acc_ids);
                if done {
                    whitelist(scores, &[nl]);
                } else {
                    whitelist(scores, &allowed);
                }
            }
            _ => {}
        }
    }

    /// After a token is chosen (update_state)
    pub fn update(&mut self, token: u32) {
        match self.state {
            State::Completed => return,
            State::Codes => {
                self.codes_count += 1;
                return;
            }
            _ => {}
        }
        if self.queue.pop_front().is_some() {
            if self.queue.is_empty() {
                let f = self.current_user.take();
                match f {
                    Some(f) => {
                        self.state = self.next_field(f);
                        self.position = 0;
                        self.acc.clear();
                        self.acc_ids.clear();
                    }
                    None => self.transition(),
                }
            }
            return;
        }
        let text = self.tok.decode(&[token]);
        let nl = self.t.newline;
        if let Some(fixed) = self.t.fixed.get(&self.state) {
            self.position += text.chars().count();
            if self.position >= fixed.chars().count() {
                if self.state == State::ThinkEndTag && self.stop_at_reasoning {
                    self.state = State::Completed;
                    self.position = 0;
                    self.acc.clear();
                    self.acc_ids.clear();
                } else {
                    self.transition();
                }
            }
            return;
        }
        match self.state {
            State::BpmValue | State::DurationValue | State::TimesigValue => {
                if token == nl {
                    self.transition();
                } else {
                    self.acc_ids.push(token);
                    let t = text.trim();
                    if !t.is_empty() && t.chars().all(|c| c.is_numeric()) {
                        self.acc.push_str(t);
                    }
                }
            }
            State::CaptionValue => {
                self.caption_tokens += 1;
                self.acc.push_str(&text);
                self.caption_after_newline = text.contains('\n');
                if self.caption_ending {
                    self.pending_field.push_str(&text);
                    if text.contains(':') {
                        let name = self.pending_field.trim().trim_end_matches(':').trim().to_lowercase();
                        let to = match name.as_str() {
                            "duration" => Some(State::DurationValue),
                            "keyscale" => Some(State::KeyscaleValue),
                            "language" => Some(State::LanguageValue),
                            "timesignature" => Some(State::TimesigValue),
                            _ => None,
                        };
                        match to {
                            Some(s) => {
                                self.state = s;
                                self.position = 0;
                                self.acc.clear();
                                self.acc_ids.clear();
                                self.caption_ending = false;
                                self.pending_field.clear();
                            }
                            None => {
                                self.caption_ending = false;
                                self.pending_field.clear();
                                self.transition();
                            }
                        }
                    }
                }
            }
            State::KeyscaleValue | State::LanguageValue => {
                if token == nl {
                    self.transition();
                } else {
                    self.acc_ids.push(token);
                    self.acc.push_str(&text);
                }
            }
            _ => {}
        }
    }
}
