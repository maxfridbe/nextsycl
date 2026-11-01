//! SheetSage2's symbolic vocabulary (tokenization_sheetsage2.py, schema v1), its event grammar (generation_sheetsage2.py
//! `PromptGrammarState`: which tokens may come next, the mask of the greedy decode) and the sequence <-> events
//! conversions (`decode_sequence`, `encode_decoded_sequence`).
//!
//! ```text
//!   0 pad  1 sos  2 eos  3 out  | 4.. the prompts (capacity 256: timestamp 4, downbeat_meter 5, structure 6, key 7,
//!   chord_majmin 8, chord_full 9, melody_vocal 10, melody_full 11) | 257 subbeat shifts (0..256) | 30000 times (10 ms
//!   over the 300 s window) | 192 meters (numerator 1..32 outer, denominator 1..32 inner) | 256 eighth-note positions |
//!   23 structure labels | 24 keys (12 major, 12 minor) | 25 major / minor chords | 361 full chords | 256 pitches (128 a
//!   melody track) | 24 durations                                                         31678 tokens, fingerprint 5ba3325af0344c7f
//! ```
//!
//! A sequence is `sos, the prompts, out`, then events, then `eos`: an event is one or more subbeat shifts (the steps
//! since the last one, a subbeat a 16th of a beat... the model's own grid) and its fields in the schema's order -
//! timestamp, rhythm (a meter and / or an eighth position), structure, key, chord, melody (pitch [duration] pairs).

pub const PAD: u32 = 0;
pub const SOS: u32 = 1;
pub const EOS: u32 = 2;
pub const OUT: u32 = 3;
pub const PROMPT0: u32 = 4;
pub const PROMPT_NAMES: [&str; 8] = ["timestamp", "downbeat_meter", "structure", "key", "chord_majmin", "chord_full", "melody_vocal", "melody_full"];
/// each prompt's output field (an index into FIELDS)
const PROMPT_FIELD: [usize; 8] = [0, 1, 2, 3, 4, 4, 5, 5];
/// transcribe's prompts (FULL_TASK_PROMPTS): every field, the full chords, both melodies
pub const FULL: [&str; 6] = ["timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full"];
pub const FIELDS: [&str; 6] = ["timestamp", "rhythm", "structure", "key", "chord", "melody"];
pub const F_TIME: usize = 0;
pub const F_RHYTHM: usize = 1;
pub const F_STRUCT: usize = 2;
pub const F_KEY: usize = 3;
pub const F_CHORD: usize = 4;
pub const F_MELODY: usize = 5;

pub const MAX_SHIFT: u32 = 256;
pub const SHIFT0: u32 = PROMPT0 + 256;
pub const TIME0: u32 = SHIFT0 + MAX_SHIFT + 1;
pub const TIME_HZ: u32 = 100;
/// time tokens: the window's 300 s at 100 a second
pub const N_TIME: u32 = 30_000;
pub const METER0: u32 = TIME0 + N_TIME;
pub const METER_DENS: [u32; 6] = [1, 2, 4, 8, 16, 32];
pub const N_METER: u32 = 32 * 6;
pub const EIGHTH0: u32 = METER0 + N_METER;
pub const STRUCT0: u32 = EIGHTH0 + 256;
pub const STRUCTURES: [&str; 23] = ["silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus", "interlude", "fade-out", "loop", "rap",
                                    "preshot", "irregular", "instrumental", "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
                                    "development", "variation", "pre-outro"];
pub const KEY0: u32 = STRUCT0 + STRUCTURES.len() as u32;
pub const MAJMIN0: u32 = KEY0 + 24;
pub const FULLCHORD0: u32 = MAJMIN0 + 25;
pub const N_FULLCHORD: u32 = 361;
pub const PITCH0: u32 = FULLCHORD0 + N_FULLCHORD;
pub const DUR0: u32 = PITCH0 + 256;
pub const DURATIONS: [u32; 24] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];
pub const VOCAB: usize = (DUR0 + DURATIONS.len() as u32) as usize;

pub const SHARPS: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const QUALITIES: [&str; 15] = ["maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4", "sus4(b7)", "maj6", "min6"];

fn inversions(q: &str) -> &'static [&'static str] {
    match q {
        "maj" => &["/2", "/3", "/5"],
        "min" => &["/2", "/b3", "/5"],
        "maj7" => &["/3", "/5", "/7"],
        "min7" => &["/b3", "/5", "/b7"],
        "7" => &["/3", "/5", "/b7"],
        _ => &[],
    }
}

/// FULL_CHORD_VOCABULARY: "N", then quality outer, root middle, inversion inner (the plain chord last)
pub fn full_chords() -> Vec<String> {
    let mut v = vec!["N".to_string()];
    for q in QUALITIES {
        for r in SHARPS {
            for inv in inversions(q).iter().chain(std::iter::once(&"")) {
                v.push(format!("{r}:{q}{inv}"));
            }
        }
    }
    v
}

pub fn majmin_chords() -> Vec<String> {
    let mut v = vec!["N".to_string()];
    v.extend(SHARPS.iter().map(|r| format!("{r}:maj")));
    v.extend(SHARPS.iter().map(|r| format!("{r}:min")));
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Pad,
    Sos,
    Eos,
    Out,
    Prompt,
    Shift,
    Time,
    Meter,
    Eighth,
    Structure,
    Key,
    MajMin,
    FullChord,
    Pitch,
    Duration,
}

/// A token's type (token_type; None: past the vocabulary, or an unused prompt slot)
pub fn ty(t: u32) -> Option<Ty> {
    Some(match t {
        _ if (SHIFT0..TIME0).contains(&t) => Ty::Shift,
        _ if (TIME0..METER0).contains(&t) => Ty::Time,
        _ if (METER0..EIGHTH0).contains(&t) => Ty::Meter,
        _ if (EIGHTH0..STRUCT0).contains(&t) => Ty::Eighth,
        _ if (STRUCT0..KEY0).contains(&t) => Ty::Structure,
        _ if (KEY0..MAJMIN0).contains(&t) => Ty::Key,
        _ if (MAJMIN0..FULLCHORD0).contains(&t) => Ty::MajMin,
        _ if (FULLCHORD0..PITCH0).contains(&t) => Ty::FullChord,
        _ if (PITCH0..DUR0).contains(&t) => Ty::Pitch,
        _ if (DUR0..VOCAB as u32).contains(&t) => Ty::Duration,
        _ if (PROMPT0..PROMPT0 + PROMPT_NAMES.len() as u32).contains(&t) => Ty::Prompt,
        PAD => Ty::Pad,
        SOS => Ty::Sos,
        EOS => Ty::Eos,
        OUT => Ty::Out,
        _ => return None,
    })
}

/// The output field a payload token belongs to
pub fn field_of(t: Ty) -> Option<usize> {
    Some(match t {
        Ty::Time => F_TIME,
        Ty::Meter | Ty::Eighth => F_RHYTHM,
        Ty::Structure => F_STRUCT,
        Ty::Key => F_KEY,
        Ty::MajMin | Ty::FullChord => F_CHORD,
        Ty::Pitch | Ty::Duration => F_MELODY,
        _ => return None,
    })
}

pub fn prompt_id(name: &str) -> Option<u32> {
    PROMPT_NAMES.iter().position(|p| *p == name).map(|i| PROMPT0 + i as u32)
}

/// The prompts in the schema's order, one a sampling group (normalize_prompts)
pub fn normalize(prompts: &[&str]) -> Result<Vec<u32>, String> {
    let mut ids: Vec<u32> = Vec::new();
    for p in prompts {
        let n = p.trim().trim_start_matches("<|").trim_end_matches("|>");
        let id = prompt_id(n).ok_or_else(|| format!("unknown prompt {p:?}"))?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids.sort();
    for w in ids.windows(2) {
        if PROMPT_FIELD[(w[0] - PROMPT0) as usize] == PROMPT_FIELD[(w[1] - PROMPT0) as usize] {
            return Err(format!("prompts {} and {} are mutually exclusive", PROMPT_NAMES[(w[0] - PROMPT0) as usize], PROMPT_NAMES[(w[1] - PROMPT0) as usize]));
        }
    }
    if ids.is_empty() {
        return Err("at least one task prompt is required".into());
    }
    Ok(ids)
}

/// sos, the prompts, out
pub fn prompt_prefix(prompts: &[u32]) -> Vec<u32> {
    let mut v = vec![SOS];
    v.extend_from_slice(prompts);
    v.push(OUT);
    v
}

/// A shift as tokens (256 at most each)
pub fn shift_tokens(mut shift: u64) -> Vec<u32> {
    let mut v = Vec::new();
    while shift > MAX_SHIFT as u64 {
        v.push(SHIFT0 + MAX_SHIFT);
        shift -= MAX_SHIFT as u64;
    }
    v.push(SHIFT0 + shift as u32);
    v
}

/// The meter (numerator, denominator) of a meter token
pub fn meter(t: u32) -> (u32, u32) {
    let i = t - METER0;
    (i / 6 + 1, METER_DENS[(i % 6) as usize])
}

/// The key a key token names, in the canonical spelling (normalize_key_name of the tokenizer's sharp name)
pub fn key_name(t: u32) -> String {
    let id = (t - KEY0) as usize;
    crate::sheet::spell::normalize_key(&format!("{}:{}", SHARPS[id % 12], if id >= 12 { "minor" } else { "major" }))
}

/// The grammar of the greedy decode (PromptGrammarState): the tokens allowed next, from what came since `out`
#[derive(Clone, Debug)]
pub struct Grammar {
    pub events: u32,
    in_shift: bool,
    shift_run: u32,
    payload: u32,
    last_field: i32,
    incomplete: Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Incomplete {
    None,
    RhythmAfterMeter,
    MelodyAfterPitch,
}

impl Default for Grammar {
    fn default() -> Self {
        Grammar { events: 0, in_shift: true, shift_run: 0, payload: 0, last_field: -1, incomplete: Incomplete::None }
    }
}

impl Grammar {
    /// The allowed tokens as [start, end) ranges
    pub fn allowed(&self) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        if self.payload > 0 {
            v.push((EOS, EOS + 1));
        }
        if (self.payload > 0 || self.in_shift) && self.shift_run < 4 {
            v.push((SHIFT0, TIME0));
        }
        match self.incomplete {
            Incomplete::RhythmAfterMeter => {
                v.push((EIGHTH0, STRUCT0));
                return v;
            }
            Incomplete::MelodyAfterPitch => {
                v.push((DUR0, VOCAB as u32));
                v.push((PITCH0, DUR0));
                return v;
            }
            Incomplete::None => {}
        }
        let lf = self.last_field;
        if lf < 0 {
            v.push((TIME0, METER0));
        }
        if lf < 1 {
            v.push((METER0, EIGHTH0));
            v.push((EIGHTH0, STRUCT0));
        }
        if lf < 2 {
            v.push((STRUCT0, KEY0));
        }
        if lf < 3 {
            v.push((KEY0, MAJMIN0));
        }
        if lf < 4 {
            v.push((FULLCHORD0, PITCH0));
        }
        if lf <= 5 {
            v.push((PITCH0, DUR0));
        }
        v
    }

    /// The step's token taken; true at eos
    pub fn update(&mut self, t: u32) -> Result<bool, String> {
        if t == EOS {
            return Ok(true);
        }
        let k = ty(t).ok_or_else(|| format!("token {t} is outside the vocabulary"))?;
        if k == Ty::Shift {
            if !self.in_shift && self.payload > 0 {
                self.events += 1;
                self.payload = 0;
                self.last_field = -1;
                self.incomplete = Incomplete::None;
            }
            self.in_shift = true;
            self.shift_run += 1;
            return Ok(false);
        }
        self.in_shift = false;
        self.shift_run = 0;
        self.payload += 1;
        let (lf, inc) = match k {
            Ty::Time => (0, Incomplete::None),
            Ty::Meter => (1, Incomplete::RhythmAfterMeter),
            Ty::Eighth => (1, Incomplete::None),
            Ty::Structure => (2, Incomplete::None),
            Ty::Key => (3, Incomplete::None),
            Ty::FullChord => (4, Incomplete::None),
            Ty::Pitch => (5, Incomplete::MelodyAfterPitch),
            Ty::Duration => (5, Incomplete::None),
            other => return Err(format!("unexpected prompt token type {other:?}")),
        };
        self.last_field = lf;
        self.incomplete = inc;
        Ok(false)
    }

    /// The greedy pick over `logits` (the masked argmax: the first of the largest)
    pub fn pick(&self, logits: &[f32]) -> u32 {
        let (mut best, mut at) = (f32::NEG_INFINITY, u32::MAX);
        let mut ranges = self.allowed();
        ranges.sort();
        for (a, b) in ranges {
            for t in a..b.min(logits.len() as u32) {
                let v = logits[t as usize];
                if v > best || at == u32::MAX {
                    best = v;
                    at = t;
                }
            }
        }
        at
    }
}

/// A note of a melody field
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub pitch: u32,
    pub track: u32,
    pub duration_bin: u32,
    pub duration_steps: u32,
    /// the stitched note's end (seconds of the song), once its window is placed
    pub end_time: Option<f64>,
}

/// One decoded event: its subbeat and its fields' tokens (by FIELDS index); once stitched, its time in the song
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Event {
    pub subbeat: i64,
    pub tokens: [Vec<u32>; 6],
    /// seconds from the song's start (stitched_window_events)
    pub time: Option<f64>,
    pub window_index: usize,
    pub window_start: f64,
    pub source_subbeat: i64,
    pub global_subbeat: i64,
    /// the melody's notes with their ends (set when stitched)
    pub notes: Vec<Note>,
}

impl Event {
    pub fn has(&self, f: usize) -> bool {
        !self.tokens[f].is_empty()
    }

    /// the timestamp value: the window-local seconds of its time token
    pub fn timestamp(&self) -> Option<f64> {
        self.tokens[F_TIME].first().map(|t| (t - TIME0) as f64 / TIME_HZ as f64)
    }

    /// the rhythm value: (meter, eighth position)
    pub fn rhythm(&self) -> (Option<(u32, u32)>, Option<u32>) {
        let (mut m, mut e) = (None, None);
        for t in &self.tokens[F_RHYTHM] {
            match ty(*t) {
                Some(Ty::Meter) => m = Some(meter(*t)),
                Some(Ty::Eighth) => e = Some(t - EIGHTH0),
                _ => {}
            }
        }
        (m, e)
    }

    pub fn structure(&self) -> Option<&'static str> {
        self.tokens[F_STRUCT].first().map(|t| STRUCTURES[(t - STRUCT0) as usize])
    }

    pub fn key(&self) -> Option<String> {
        self.tokens[F_KEY].first().map(|t| key_name(*t))
    }

    pub fn chord(&self, full: &[String], majmin: &[String]) -> Option<String> {
        self.tokens[F_CHORD].first().map(|t| if ty(*t) == Some(Ty::MajMin) { majmin[(t - MAJMIN0) as usize].clone() } else { full[(t - FULLCHORD0) as usize].clone() })
    }

    /// the melody's notes from its tokens (a pitch, then its duration when one follows; bin 0 otherwise)
    pub fn melody(&self) -> Vec<Note> {
        let f = &self.tokens[F_MELODY];
        let mut v = Vec::new();
        let mut i = 0;
        while i < f.len() {
            let p = f[i] - PITCH0;
            let mut bin = 0;
            if i + 1 < f.len() && ty(f[i + 1]) == Some(Ty::Duration) {
                bin = f[i + 1] - DUR0;
                i += 2;
            } else {
                i += 1;
            }
            v.push(Note { pitch: p % 128, track: (p >= 128) as u32, duration_bin: bin, duration_steps: DURATIONS[bin as usize], end_time: None });
        }
        v
    }
}

/// A decoded sequence (decode_sequence's dict)
#[derive(Clone, Debug, Default)]
pub struct Decoded {
    pub prompts: Vec<u32>,
    pub events: Vec<Event>,
    pub has_eos: bool,
}

/// decode_sequence: the events of a sequence, strict (every rule upstream checks) or not (an empty event or a token of
/// a field no prompt asked for kept going)
pub fn decode_sequence(tokens: &[u32], strict: bool) -> Result<Decoded, String> {
    let mut tokens = tokens.to_vec();
    while tokens.last() == Some(&PAD) {
        tokens.pop();
    }
    if tokens.first() != Some(&SOS) {
        return Err("sequence must begin with <|sos|>".into());
    }
    let out_at = tokens.iter().skip(1).position(|t| *t == OUT).map(|p| p + 1).ok_or("sequence is missing <|out|>")?;
    let mut prompts = Vec::new();
    for t in &tokens[1..out_at] {
        if ty(*t) != Some(Ty::Prompt) {
            return Err(format!("token {t} is not a prompt token"));
        }
        prompts.push(*t);
    }
    if strict {
        let names: Vec<&str> = prompts.iter().map(|p| PROMPT_NAMES[(p - PROMPT0) as usize]).collect();
        if normalize(&names)? != prompts {
            return Err("prompt tokens are not in canonical schema order".into());
        }
    }
    let active: Vec<usize> = prompts.iter().map(|p| PROMPT_FIELD[(p - PROMPT0) as usize]).collect();
    let mut events = Vec::new();
    let mut pos = out_at + 1;
    let mut step: i64 = 0;
    let mut saw_eos = false;
    let tyof = |t: u32| ty(t).ok_or_else(|| format!("token {t} is outside vocabulary size {VOCAB}"));
    while pos < tokens.len() {
        let t = tokens[pos];
        if t == EOS {
            saw_eos = true;
            pos += 1;
            break;
        }
        if tyof(t)? != Ty::Shift {
            return Err(format!("event at token index {pos} has no subbeat shift"));
        }
        let mut shift = 0i64;
        while pos < tokens.len() && tyof(tokens[pos])? == Ty::Shift {
            shift += (tokens[pos] - SHIFT0) as i64;
            pos += 1;
        }
        step += shift;
        let mut fields: [Vec<u32>; 6] = Default::default();
        while pos < tokens.len() {
            let t = tokens[pos];
            let k = tyof(t)?;
            if k == Ty::Shift || t == EOS {
                break;
            }
            let f = field_of(k).ok_or_else(|| format!("token {t} ({k:?}) has no field in schema v1"))?;
            if strict && !active.contains(&f) {
                return Err(format!("token {t} belongs to inactive output field {:?}", FIELDS[f]));
            }
            fields[f].push(t);
            pos += 1;
        }
        if fields.iter().all(|f| f.is_empty()) {
            if strict {
                return Err(format!("empty event at subbeat {step}"));
            }
            continue;
        }
        if strict {
            for (f, vals) in fields.iter().enumerate() {
                if vals.is_empty() {
                    continue;
                }
                let types: Vec<Ty> = vals.iter().map(|v| ty(*v).unwrap_or(Ty::Pad)).collect();
                match f {
                    F_TIME if types != [Ty::Time] => return Err("timestamp event must contain exactly one time token".into()),
                    F_RHYTHM if types != [Ty::Eighth] && types != [Ty::Meter, Ty::Eighth] => return Err(format!("invalid rhythm payload: {types:?}")),
                    F_STRUCT | F_KEY | F_CHORD if vals.len() != 1 => return Err(format!("field {:?} must contain exactly one token", FIELDS[f])),
                    F_MELODY => {
                        let mut i = 0;
                        while i < types.len() {
                            if types[i] != Ty::Pitch {
                                return Err("melody payload must contain pitch tokens with optional duration".into());
                            }
                            i += if i + 1 < types.len() && types[i + 1] == Ty::Duration { 2 } else { 1 };
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut e = Event { subbeat: step, tokens: fields, ..Default::default() };
        e.notes = e.melody();
        events.push(e);
    }
    if strict && !saw_eos {
        return Err("sequence is missing <|eos|>".into());
    }
    if strict && pos != tokens.len() {
        return Err("non-padding tokens follow <|eos|>".into());
    }
    Ok(Decoded { prompts, events, has_eos: saw_eos })
}

/// decode_generated_tokens: strict, else (for the two errors upstream recovers from) the lenient reading and the
/// strict one's complaint as a warning
pub fn decode_generated(tokens: &[u32]) -> Result<(Decoded, Option<String>), String> {
    match decode_sequence(tokens, true) {
        Ok(d) => Ok((d, None)),
        Err(e) if e.contains("empty event at subbeat") || e.contains("belongs to inactive output field") => Ok((decode_sequence(tokens, false)?, Some(e))),
        Err(e) => Err(e),
    }
}

/// encode_decoded_sequence: the events back to tokens (sorted by subbeat), eos when `has_eos`
pub fn encode_decoded(d: &Decoded) -> Result<Vec<u32>, String> {
    let names: Vec<&str> = d.prompts.iter().map(|p| PROMPT_NAMES[(p - PROMPT0) as usize]).collect();
    let mut out = prompt_prefix(&normalize(&names)?);
    let mut prev = 0i64;
    for e in &d.events {
        if e.subbeat < prev {
            return Err("events must be sorted by non-decreasing subbeat".into());
        }
        out.extend(shift_tokens((e.subbeat - prev) as u64));
        prev = e.subbeat;
        for f in &e.tokens {
            out.extend_from_slice(f);
        }
    }
    if d.has_eos {
        out.push(EOS);
    }
    Ok(out)
}

/// A token as upstream describes it (tokens.txt)
pub fn describe(t: u32) -> String {
    match ty(t) {
        Some(Ty::Prompt) => format!("<|{}|>", PROMPT_NAMES[(t - PROMPT0) as usize]),
        Some(Ty::Shift) => format!("<subbeat_shift_{}>", t - SHIFT0),
        Some(Ty::Time) => format!("<time_{:.2}s>", (t - TIME0) as f64 / TIME_HZ as f64),
        Some(Ty::Meter) => {
            let (n, d) = meter(t);
            format!("<meter_{n}/{d}>")
        }
        Some(Ty::Eighth) => format!("<eighth_pos_{}>", t - EIGHTH0),
        Some(Ty::Structure) => format!("<structure_{}>", STRUCTURES[(t - STRUCT0) as usize]),
        Some(Ty::Key) => format!("<key_{}>", key_name(t)),
        Some(Ty::MajMin) => format!("<chord_majmin_{}>", majmin_chords()[(t - MAJMIN0) as usize]),
        Some(Ty::FullChord) => format!("<chord_full_{}>", full_chords()[(t - FULLCHORD0) as usize]),
        Some(Ty::Pitch) => format!("<pitch_{}_track_{}>", (t - PITCH0) % 128, ((t - PITCH0) >= 128) as u32),
        Some(Ty::Duration) => format!("<duration_{}>", t - DUR0),
        Some(Ty::Pad) => "<|pad|>".into(),
        Some(Ty::Sos) => "<|sos|>".into(),
        Some(Ty::Eos) => "<|eos|>".into(),
        Some(Ty::Out) => "<|out|>".into(),
        None => format!("<{t}?>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vocabulary_closes_where_upstreams_does() {
        assert_eq!(VOCAB, 31678);
        assert_eq!(full_chords().len(), N_FULLCHORD as usize);
        assert_eq!(full_chords()[1], "C:maj/2");
        assert_eq!(prompt_prefix(&normalize(&FULL).unwrap()), vec![1, 4, 5, 6, 7, 9, 11, 3]);
        assert_eq!(shift_tokens(600), vec![SHIFT0 + 256, SHIFT0 + 256, SHIFT0 + 88]);
        assert!(normalize(&["chord_full", "chord_majmin"]).is_err());
    }
}
