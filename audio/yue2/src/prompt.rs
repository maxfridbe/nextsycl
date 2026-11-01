//! The prompts and the token ids, as upstream's protocol.py writes them (yue2-native-v1): the instruction of the cot
//! mode, the style under `[Tags]`, the lyrics under `[Lyrics]`, after an EOD; the score between ABC_START and ABC_END;
//! MUSIC_START before the semantic codes. The text is NFC-normalized, then tiktoken's `encode_ordinary`: a special
//! token's text in the lyrics is read as plain characters.

use nextsycl_tok::Tokenizer;
use unicode_normalization::UnicodeNormalization;

pub const EOD: u32 = 151_643;
pub const ABC_START: u32 = 151_847;
pub const ABC_END: u32 = 151_848;
pub const MUSIC_START: u32 = 151_851;
pub const MUSIC_END: u32 = 151_852;
pub const CODEC_OFFSET: u32 = 151_853;
pub const CODEC_SIZE: u32 = 32_768;

/// What the model plans before the music (protocol.py's INSTRUCTIONS)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cot {
    /// no score: the codes from the words alone
    Off,
    /// a melody-only score first (the covers' mode)
    Melody,
    /// a chord-annotated score first (the default)
    Full,
}

impl Cot {
    pub fn parse(s: &str) -> Option<Cot> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" | "false" => Some(Cot::Off),
            "melody" => Some(Cot::Melody),
            "full" | "" => Some(Cot::Full),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Cot::Off => "off",
            Cot::Melody => "melody",
            Cot::Full => "full",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Cot::Off => "Generate music with codec tokens from the given conditions.",
            Cot::Melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
            Cot::Full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        }
    }

    /// the semantic phase's guidance when the request names none (SongRequest.guidance)
    pub fn default_cfg(self) -> f32 {
        if self == Cot::Off { 1.01 } else { 1.0 }
    }
}

/// The text encoding upstream's tokenizer makes (NFC, no special tokens read out of the text)
pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode_ordinary(&text.nfc().collect::<String>())
}

/// The request's text (SongRequest.text)
pub fn text(cot: Cot, style: &str, lyrics: &str) -> String {
    format!("{}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n", cot.instruction())
}

/// The first phase's prompt (token_prefixes with no score yet): cot off ends at MUSIC_START, else at ABC_START (the
/// score is drawn after it)
pub fn prefix(tok: &Tokenizer, cot: Cot, style: &str, lyrics: &str) -> Vec<u32> {
    let mut v = vec![EOD];
    v.extend(encode(tok, &text(cot, style, lyrics)));
    match cot {
        Cot::Off => v.extend([ABC_START, ABC_END, MUSIC_START]),
        _ => v.push(ABC_START),
    }
    v
}

/// The semantic phase's prompt: the first's with its score closed (token_prefixes with the score's ids)
pub fn semantic(first: &[u32], cot: Cot, abc: &[u32]) -> Vec<u32> {
    let mut v = first.to_vec();
    if cot != Cot::Off {
        v.extend_from_slice(abc);
        v.extend([ABC_END, MUSIC_START]);
    }
    v
}

/// The classifier-free twin (negative_prefix): the instruction alone, the same score
pub fn negative(tok: &Tokenizer, cot: Cot, abc: &[u32]) -> Vec<u32> {
    let mut v = vec![EOD];
    v.extend(encode(tok, cot.instruction()));
    if cot == Cot::Off {
        v.push(MUSIC_START);
    } else {
        v.push(ABC_START);
        v.extend_from_slice(abc);
        v.extend([ABC_END, MUSIC_START]);
    }
    v
}

/// chunk_ranges: the song's frames in chunks a context holds beside the prompt (whole songs are one in practice)
pub fn chunks(frames: usize, prompt: usize, context: usize) -> Vec<(usize, usize)> {
    let size = ((context.saturating_sub(prompt + 3)) / 2).min(context);
    if size == 0 {
        return Vec::new();
    }
    (0..frames).step_by(size).map(|a| (a, (a + size).min(frames))).collect()
}

/// Whether a voice's lines hold a note (a pitch letter outside the chord symbols' quotes; rests are z / Z / x)
fn has_notes(lines: &[&str]) -> bool {
    lines.iter().any(|l| {
        let mut quoted = false;
        l.chars().any(|c| {
            if c == '"' {
                quoted = !quoted;
            }
            !quoted && matches!(c, 'A'..='G' | 'a'..='g')
        })
    })
}

/// A cover's score with its tune where YuE2 sings it: SheetSage2 writes a recording's lead line into the `Vocal` voice
/// when someone sings it and into `Ins` when an instrument plays it; YuE2 sings the `Vocal` voice (its training's
/// scores had the sung line there - over an all-rest Vocal voice it makes an instrumental). When the Vocal voice has no
/// note and Ins has, the two voices' lines are exchanged (bar for bar: their durations match); else the score is kept.
/// Whether it exchanged them is the second value.
pub fn sung_lead(abc: &str) -> (String, bool) {
    let lines: Vec<&str> = abc.lines().collect();
    // the body's blocks: a line "V: Vocal" or "V: Ins" alone, then the lines up to the next "V:" or "%" line
    let mut blocks: Vec<(usize, usize, &str)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        let voice = match t {
            "V: Vocal" | "V:Vocal" => Some("Vocal"),
            "V: Ins" | "V:Ins" => Some("Ins"),
            _ => None,
        };
        if let Some(v) = voice {
            let mut j = i + 1;
            while j < lines.len() && !lines[j].trim_start().starts_with("V:") && !lines[j].trim_start().starts_with('%') {
                j += 1;
            }
            blocks.push((i + 1, j, v));
            i = j;
        } else {
            i += 1;
        }
    }
    let of = |v: &str| -> Vec<&str> { blocks.iter().filter(|b| b.2 == v).flat_map(|b| lines[b.0..b.1].iter().copied()).collect() };
    if has_notes(&of("Vocal")) || !has_notes(&of("Ins")) {
        return (abc.to_string(), false);
    }
    // the voices' blocks pair up in order (the score writes each passage's Vocal block, then its Ins block)
    let vocal: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == "Vocal").collect();
    let ins: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == "Ins").collect();
    if vocal.len() != ins.len() {
        return (abc.to_string(), false);
    }
    let mut body: std::collections::BTreeMap<usize, (usize, Vec<&str>)> = std::collections::BTreeMap::new();
    for (v, n) in vocal.iter().zip(&ins) {
        body.insert(v.0, (v.1, lines[n.0..n.1].to_vec()));
        body.insert(n.0, (n.1, lines[v.0..v.1].to_vec()));
    }
    let mut out = Vec::with_capacity(lines.len());
    let mut k = 0;
    while k < lines.len() {
        match body.get(&k) {
            Some((end, repl)) => {
                out.extend(repl.iter().copied());
                k = *end;
            }
            None => {
                out.push(lines[k]);
                k += 1;
            }
        }
    }
    let mut s = out.join("\n");
    if abc.ends_with('\n') {
        s.push('\n');
    }
    (s, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunks_cover_the_frames() {
        assert_eq!(chunks(9000, 3000, 24576), vec![(0, 9000)]);
        let c = chunks(9000, 10000, 24576);
        assert_eq!(c.first(), Some(&(0, 7286)));
        assert_eq!(c.last(), Some(&(7286, 9000)));
        assert!(chunks(10, 24575, 24576).is_empty());
    }

    #[test]
    fn an_instrumental_lead_is_sung() {
        let abc = "X:1\nV: Vocal clef=treble name=\"Vocal Melody\"\nV: Ins clef=treble name=\"Ins Melody\"\nK:C\n% intro\nV: Vocal\nZ2|\nV: Ins\nG4A4|c8|\n% verse\nV: Vocal\n\"C\"z8|\nV: Ins\n\"C\"e8|\n";
        let (s, swapped) = sung_lead(abc);
        assert!(swapped);
        assert_eq!(s, "X:1\nV: Vocal clef=treble name=\"Vocal Melody\"\nV: Ins clef=treble name=\"Ins Melody\"\nK:C\n% intro\nV: Vocal\nG4A4|c8|\nV: Ins\nZ2|\n% verse\nV: Vocal\n\"C\"e8|\nV: Ins\n\"C\"z8|\n");
        // a sung lead stays where it is
        let (t, again) = sung_lead(&s);
        assert!(!again && t == s);
    }

    #[test]
    fn the_modes_parse() {
        assert_eq!(Cot::parse("Melody"), Some(Cot::Melody));
        assert_eq!(Cot::parse(""), Some(Cot::Full));
        assert_eq!(Cot::parse("OFF"), Some(Cot::Off));
        assert_eq!(Cot::parse("chords"), None);
    }
}
