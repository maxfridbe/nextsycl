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

/// frames two neighbouring chunks of the flow share (10 s), and where in them the song passes from one to the next: a
/// crossfade of `FADE` frames (2 s) from `FADE_AT` frames (4 s) in. Each chunk's flow is solved alone, so cut end to end
/// the latents step at the cut (2026-11-23, 600 s songs cut at 362.9 s: the step between the two frames at the 97th / 92nd
/// percentile of the song's frame-to-frame steps, B70 / B65); crossfaded, the step in the middle of the fade sat at the
/// 3rd. The join also keeps the frames next to a chunk's own LATENT_START / LATENT_END out of the song
pub const OVERLAP: usize = 250;
pub const FADE_AT: usize = 100;
pub const FADE: usize = 50;

/// The song's frames in chunks a context holds beside the prompt (one up to (24576 - prompt - 3) / 2 frames: ~7 minutes
/// beside a short prompt; a ten-minute song is two). Upstream's chunk_ranges cut them end to end; here each next chunk
/// starts `OVERLAP` frames before the last one ends (end to end only when a chunk is too short to share that much)
pub fn chunks(frames: usize, prompt: usize, context: usize) -> Vec<(usize, usize)> {
    let size = ((context.saturating_sub(prompt + 3)) / 2).min(context);
    if size == 0 {
        return Vec::new();
    }
    let overlap = if size > 2 * OVERLAP { OVERLAP } else { 0 };
    let mut v = Vec::new();
    let mut a = 0;
    loop {
        let b = (a + size).min(frames);
        v.push((a, b));
        if b >= frames {
            return v;
        }
        a = b - overlap;
    }
}

/// The chunks' latents ([b - a, width] each, in `chunks` order) as one song's [frames, width]: where two share frames,
/// the first alone up to `FADE_AT` frames into the shared ones, then a linear crossfade of `FADE` frames, then the next
pub fn join(frames: usize, width: usize, parts: &[((usize, usize), Vec<f32>)]) -> Vec<f32> {
    let mut out = vec![0f32; frames * width];
    // the weight of part i at frame t: rising from its shared start (if any), falling where the next one rises
    let rise = |i: usize, t: usize| -> f32 {
        if i == 0 {
            return 1.0;
        }
        let (a, prev_b) = (parts[i].0 .0, parts[i - 1].0 .1);
        if prev_b <= a {
            return 1.0;
        }
        let (at, n) = if prev_b - a >= FADE_AT + FADE { (FADE_AT, FADE) } else { (0, prev_b - a) };
        let (s, e) = (a + at, a + at + n);
        if t < s {
            0.0
        } else if t >= e {
            1.0
        } else {
            (t - s) as f32 / n as f32 + 0.5 / n as f32
        }
    };
    for (i, ((a, b), lat)) in parts.iter().enumerate() {
        for t in *a..*b {
            let w = rise(i, t) * if i + 1 < parts.len() && t >= parts[i + 1].0 .0 { 1.0 - rise(i + 1, t) } else { 1.0 };
            if w == 0.0 {
                continue;
            }
            for (o, x) in out[t * width..(t + 1) * width].iter_mut().zip(&lat[(t - a) * width..(t - a + 1) * width]) {
                *o += w * x;
            }
        }
    }
    out
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

/// The body's voice blocks: (first line, end line, voice) for each "V: Vocal" or "V: Ins" line alone, its lines up to the
/// next "V:" or "%" line
fn voice_blocks(lines: &[&str]) -> Vec<(usize, usize, &'static str)> {
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let voice = match lines[i].trim() {
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
    blocks
}

/// The score with some line ranges replaced (start -> (end, the new lines)), its final newline kept
fn rewrite(abc: &str, lines: &[&str], body: &std::collections::BTreeMap<usize, (usize, Vec<String>)>) -> String {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut k = 0;
    while k < lines.len() {
        match body.get(&k) {
            Some((end, repl)) => {
                out.extend(repl.iter().cloned());
                k = *end;
            }
            None => {
                out.push(lines[k].to_string());
                k += 1;
            }
        }
    }
    let mut s = out.join("\n");
    if abc.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// The lead moved from voice `from` to voice `to` when `to` has no note and `from` has: the two voices' lines are
/// exchanged (bar for bar: their durations match); else the score is kept. Whether it exchanged them is the second value.
fn moved_lead(abc: &str, from: &str, to: &str) -> (String, bool) {
    let lines: Vec<&str> = abc.lines().collect();
    let blocks = voice_blocks(&lines);
    let of = |v: &str| -> Vec<&str> { blocks.iter().filter(|b| b.2 == v).flat_map(|b| lines[b.0..b.1].iter().copied()).collect() };
    if has_notes(&of(to)) || !has_notes(&of(from)) {
        return (abc.to_string(), false);
    }
    // the voices' blocks pair up in order (the score writes each passage's Vocal block, then its Ins block)
    let a: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == from).collect();
    let b: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == to).collect();
    if a.len() != b.len() {
        return (abc.to_string(), false);
    }
    let own = |r: &[&str]| r.iter().map(|l| l.to_string()).collect::<Vec<_>>();
    let mut body = std::collections::BTreeMap::new();
    for (x, y) in a.iter().zip(&b) {
        body.insert(x.0, (x.1, own(&lines[y.0..y.1])));
        body.insert(y.0, (y.1, own(&lines[x.0..x.1])));
    }
    (rewrite(abc, &lines, &body), true)
}

/// A cover's score with its tune where YuE2 sings it: SheetSage2 writes a recording's lead line into the `Vocal` voice
/// when someone sings it and into `Ins` when an instrument plays it; YuE2 sings the `Vocal` voice (its training's
/// scores had the sung line there - over an all-rest Vocal voice it makes an instrumental). When the Vocal voice has no
/// note and Ins has, the two voices' lines are exchanged (bar for bar: their durations match); else the score is kept.
/// Whether it exchanged them is the second value.
pub fn sung_lead(abc: &str) -> (String, bool) {
    moved_lead(abc, "Ins", "Vocal")
}

/// A line of ABC with its notes made rests of the same length: a pitch (its accidentals, letter and octave marks) or a
/// bracketed chord of pitches becomes `z`, ties go (a rest is not tied); chord symbols ("C"), lengths, bar lines and
/// the rest stay
fn rested(line: &str) -> String {
    let c: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < c.len() {
        match c[i] {
            '"' => {
                // a chord symbol (or annotation), as it is
                let j = c[i + 1..].iter().position(|&x| x == '"').map_or(c.len(), |p| i + 2 + p);
                out.extend(&c[i..j]);
                i = j;
            }
            '[' if c.get(i + 1).is_some_and(|x| matches!(x, '^' | '_' | '=' | 'A'..='G' | 'a'..='g')) => {
                // a chord of notes: one rest
                let j = c[i..].iter().position(|&x| x == ']').map_or(c.len(), |p| i + p + 1);
                out.push('z');
                i = j;
            }
            '^' | '_' | '=' | 'A'..='G' | 'a'..='g' => {
                // a note: its accidentals, letter and octave marks
                let mut j = i;
                while j < c.len() && matches!(c[j], '^' | '_' | '=') {
                    j += 1;
                }
                if j < c.len() && c[j].is_ascii_alphabetic() && matches!(c[j].to_ascii_uppercase(), 'A'..='G') {
                    j += 1;
                    while j < c.len() && matches!(c[j], '\'' | ',') {
                        j += 1;
                    }
                    out.push('z');
                } else {
                    out.extend(&c[i..j]);
                }
                i = j.max(i + 1);
            }
            '-' => i += 1,
            x => {
                out.push(x);
                i += 1;
            }
        }
    }
    out
}

/// A cover's score as an instrumental: the lead where an instrument plays it (SheetSage2's `Ins` voice) and the `Vocal`
/// voice YuE2 sings all rests - the score as SheetSage2 writes an instrumental recording, which YuE2 plays (measured on
/// a Jingle Bells band recording: jazz, metal and synthwave with no words Whisper heard). Passage by passage (each
/// Vocal block with the Ins block after it): a lead in Vocal alone moves to Ins, so the tune is kept where SheetSage2
/// heard it as sung (a violin's line, 2026-11-11: some passages of a Paganini caprice); with a line in both the Vocal
/// notes become rests of the same length.
pub fn played_lead(abc: &str) -> String {
    let lines: Vec<&str> = abc.lines().collect();
    let blocks = voice_blocks(&lines);
    let vocal: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == "Vocal").collect();
    let ins: Vec<&(usize, usize, &str)> = blocks.iter().filter(|b| b.2 == "Ins").collect();
    let notes = |b: &(usize, usize, &str)| has_notes(&lines[b.0..b.1]);
    let own = |b: &(usize, usize, &str)| lines[b.0..b.1].iter().map(|l| l.to_string()).collect::<Vec<_>>();
    let rest = |b: &(usize, usize, &str)| lines[b.0..b.1].iter().map(|l| rested(l)).collect::<Vec<_>>();
    let mut body = std::collections::BTreeMap::new();
    if vocal.len() == ins.len() {
        for (v, n) in vocal.iter().zip(&ins) {
            if !notes(v) {
                continue;
            }
            if notes(n) {
                body.insert(v.0, (v.1, rest(v)));
            } else {
                // the bars match (the voices' durations are the same): the lines exchanged
                body.insert(v.0, (v.1, own(n)));
                body.insert(n.0, (n.1, own(v)));
            }
        }
    } else {
        // blocks that do not pair up: nothing moved, every Vocal note a rest
        for v in vocal.iter().filter(|v| notes(v)) {
            body.insert(v.0, (v.1, rest(v)));
        }
    }
    if body.is_empty() {
        return abc.to_string();
    }
    rewrite(abc, &lines, &body)
}

/// The lyrics of an instrumental cover: the score's sections (its "% intro", "% verse" ... lines, in order) as empty
/// section tags - every passage an instrumental one, as an empty [Intro] is in a song's lyrics. No section: [Instrumental]
pub fn bare_sections(abc: &str) -> String {
    let tags: Vec<String> = abc.lines().filter_map(|l| l.trim().strip_prefix("% ")).map(str::trim)
        .filter(|l| crate::sheet::tokens::STRUCTURES.contains(l) && *l != "silence")
        .map(|l| format!("[{}]", l.split(' ').map(|w| w.split('-').map(|p| {
            let mut c = p.chars();
            c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
        }).collect::<Vec<_>>().join("-")).collect::<Vec<_>>().join(" ")))
        .collect();
    if tags.is_empty() {
        return "[Instrumental]\n".into();
    }
    tags.join("\n\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunks_cover_the_frames() {
        assert_eq!(chunks(9000, 3000, 24576), vec![(0, 9000)]);
        let c = chunks(9000, 10000, 24576);
        assert_eq!(c, vec![(0, 7286), (7036, 9000)]);
        assert!(chunks(10, 24575, 24576).is_empty());
        // ten minutes beside a 3000-token prompt: two chunks sharing OVERLAP frames
        assert_eq!(chunks(15000, 3000, 24576), vec![(0, 10786), (10536, 15000)]);
        // a chunk too short to share: end to end
        assert_eq!(chunks(1000, 24576 - 3 - 800, 24576), vec![(0, 400), (400, 800), (800, 1000)]);
        // every frame is in a chunk, each chunk within the context beside the prompt
        for (f, p) in [(15000, 4519), (10500, 6000), (15000, 9000), (3, 100)] {
            let c = chunks(f, p, 24576);
            assert_eq!((c[0].0, c.last().unwrap().1), (0, f));
            assert!(c.windows(2).all(|w| w[1].0 < w[0].1 && w[1].0 > w[0].0));
            assert!(c.iter().all(|(a, b)| p + 2 * (b - a) + 3 <= 24576));
        }
    }

    #[test]
    fn the_chunks_join_with_a_crossfade() {
        // one chunk: as it is
        let one = vec![((0, 4), vec![1.0, 2.0, 3.0, 4.0])];
        assert_eq!(join(4, 1, &one), vec![1.0, 2.0, 3.0, 4.0]);
        // end to end: each its own
        assert_eq!(join(4, 1, &[((0, 2), vec![1.0, 1.0]), ((2, 4), vec![2.0, 2.0])]), vec![1.0, 1.0, 2.0, 2.0]);
        // sharing OVERLAP frames: the first alone up to FADE_AT in, a FADE-frame ramp, then the second; the weights sum to one
        let (n, a) = (1000, 700);
        let parts = [((0, a + OVERLAP), vec![1.0; a + OVERLAP]), ((a, n), vec![3.0; n - a])];
        let j = join(n, 1, &parts);
        assert!(j[..a + FADE_AT].iter().all(|x| *x == 1.0));
        assert!(j[a + FADE_AT + FADE..].iter().all(|x| *x == 3.0));
        let ramp = &j[a + FADE_AT..a + FADE_AT + FADE];
        assert!(ramp.windows(2).all(|w| w[1] > w[0]) && ramp.iter().all(|x| *x > 1.0 && *x < 3.0));
        // two channels a frame stay together
        let two = join(2, 2, &[((0, 2), vec![1.0, 2.0, 3.0, 4.0])]);
        assert_eq!(two, vec![1.0, 2.0, 3.0, 4.0]);
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
    fn an_instrumental_cover_plays_its_lead() {
        // the lead in Vocal alone moves to Ins
        let sung = "X:1\nV: Vocal clef=treble name=\"Vocal Melody\"\nV: Ins clef=treble name=\"Ins Melody\"\nK:C\n% intro\nV: Vocal\nG4A4|c8|\nV: Ins\nZ2|\n% verse\nV: Vocal\n\"C\"e8|\nV: Ins\n\"C\"z8|\n";
        let played = "X:1\nV: Vocal clef=treble name=\"Vocal Melody\"\nV: Ins clef=treble name=\"Ins Melody\"\nK:C\n% intro\nV: Vocal\nZ2|\nV: Ins\nG4A4|c8|\n% verse\nV: Vocal\n\"C\"z8|\nV: Ins\n\"C\"e8|\n";
        assert_eq!(played_lead(sung), played);
        // as transcribed (the lead in Ins already): kept
        assert_eq!(played_lead(played), played);
        // a line in both: the Vocal voice's notes rests of the same length, its chord symbols kept
        let both = "K:C\n% verse\nV: Vocal\n\"Am\"^c'2-c/2 [ceg]4 _B,3/2|\nV: Ins\nE8|\n";
        assert_eq!(played_lead(both), "K:C\n% verse\nV: Vocal\n\"Am\"z2z/2 z4 z3/2|\nV: Ins\nE8|\n");
        // passage by passage: the lead in Ins in one, in Vocal alone in the next (moved, not lost), in both in a third
        let mixed = "K:Dm\n% intro\nV: Vocal\nZ2|\nV: Ins\nD4F4|A8|\n% verse\nV: Vocal\nd4f4|a8|\nV: Ins\nZ2|\n% outro\nV: Vocal\nA8|\nV: Ins\nD8|\n";
        assert_eq!(played_lead(mixed), "K:Dm\n% intro\nV: Vocal\nZ2|\nV: Ins\nD4F4|A8|\n% verse\nV: Vocal\nZ2|\nV: Ins\nd4f4|a8|\n% outro\nV: Vocal\nz8|\nV: Ins\nD8|\n");
        assert!(!has_notes(&["\"Am\"z2z/2 z4 z3/2|"]));
    }

    #[test]
    fn an_instrumental_cover_has_bare_sections() {
        let abc = "K:C\n% intro\nV: Vocal\nZ2|\n% pre-chorus and chorus\nV: Vocal\nZ2|\n% ss2\n% outro\nV: Vocal\nZ|\n";
        assert_eq!(bare_sections(abc), "[Intro]\n\n[Pre-Chorus And Chorus]\n\n[Outro]\n");
        assert_eq!(bare_sections("K:C\nV: Vocal\nZ2|\n"), "[Instrumental]\n");
    }

    #[test]
    fn the_modes_parse() {
        assert_eq!(Cot::parse("Melody"), Some(Cot::Melody));
        assert_eq!(Cot::parse(""), Some(Cot::Full));
        assert_eq!(Cot::parse("OFF"), Some(Cot::Off));
        assert_eq!(Cot::parse("chords"), None);
    }
}
