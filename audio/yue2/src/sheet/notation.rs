//! The stitched events to ABC: exports_sheetsage2.py's `export_result` (its ABC half: the melody notes, the beats, the
//! key / chord / structure intervals, the chords spelled for the local key) and notation_sheetsage2.py whole (the beat
//! grid, the measures inferred from the downbeats, the voices on a grid of four subbeats a beat, the two-voice ABC and
//! its validation), as upstream runs them.
//!
//! Upstream hands the notes to the notation through a MIDI file in memory (pretty_midi, 960 ticks a beat at 120 bpm):
//! the round trip rounds every onset and end to a tick of 1/1920 s (Python's round, half to even) and drops a note
//! whose two ends land on one tick. `midi_quantize` does the same, so the grid sees the same times.
//!
//! `melody_only` (covers): the chords are never read; the chord timeline is "N" throughout, so held notes and rests
//! are written as their own segments and no chord symbol appears - both melody voices (Vocal, Ins) are kept.

use std::collections::BTreeMap;

use crate::sheet::spell::{correct_chord_rows, normalize_key};
use crate::sheet::tokens::{full_chords, majmin_chords, Event, F_CHORD, F_KEY, F_RHYTHM, F_STRUCT};

const SUBBEAT_DIVISION: i64 = 4;
const VOICES: [&str; 2] = ["Vocal", "Ins"];
const SUPPORTED: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];
const LETTERS: &[u8; 7] = b"CDEFGAB";
const SHARP_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_NAMES: [&str; 12] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

type R<T> = Result<T, String>;

fn key_signature(k: &str) -> Option<i32> {
    Some(match k {
        "C" | "Am" => 0,
        "G" | "Em" => 1,
        "D" | "Bm" => 2,
        "A" | "F#m" => 3,
        "E" | "C#m" => 4,
        "B" | "G#m" => 5,
        "F#" | "D#m" => 6,
        "C#" | "A#m" => 7,
        "F" | "Dm" => -1,
        "Bb" | "Gm" => -2,
        "Eb" | "Cm" => -3,
        "Ab" | "Fm" => -4,
        "Db" | "Bbm" => -5,
        "Gb" | "Ebm" => -6,
        "Cb" | "Abm" => -7,
        _ => return None,
    })
}

/// _KEY_RELATIVE_PITCH_NAMES: each pitch class's name in a key of `n` sharps (flats below 0)
fn key_relative_name(n: i32, pc: usize) -> Option<&'static str> {
    const T: [[&str; 12]; 15] = [
        ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"],
        ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        ["B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        ["B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        ["B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
    ];
    T.get((n + 7) as usize).map(|r| r[pc])
}

fn natural_pc(l: char) -> i32 {
    match l {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        _ => 11,
    }
}

/// _pitch_class: (pitch class, letter, accidental) of a spelling like "F#", "Bbb"
fn pitch_class(root: &str) -> R<(i32, char, String)> {
    let mut cs = root.chars();
    let l = cs.next().filter(|c| ('A'..='G').contains(c)).ok_or_else(|| format!("Invalid pitch spelling {root:?}"))?;
    let acc: String = cs.collect();
    let ok = acc.is_empty() || acc == "#" || acc == "##" || acc == "b" || acc == "bb";
    if !ok {
        return Err(format!("Invalid pitch spelling {root:?}"));
    }
    let off = acc.matches('#').count() as i32 - acc.matches('b').count() as i32;
    Ok(((natural_pc(l) + off).rem_euclid(12), l, acc))
}

fn portable_pitch_name(root: &str, preserve_double: bool) -> R<String> {
    let (pc, _, acc) = pitch_class(root)?;
    if preserve_double || acc.chars().count() <= 1 {
        return Ok(root.to_string());
    }
    Ok(if acc.starts_with('#') { SHARP_NAMES[pc as usize] } else { FLAT_NAMES[pc as usize] }.to_string())
}

fn bass_degree_to_pitch(root: &str, degree: &str) -> R<String> {
    if pitch_class(degree).is_ok() {
        return portable_pitch_name(degree, true);
    }
    let digits_at = degree.find(|c: char| c.is_ascii_digit()).ok_or_else(|| format!("Invalid chord bass degree {degree:?}"))?;
    let (dacc, dnum) = degree.split_at(digits_at);
    let okacc = dacc.is_empty() || dacc == "#" || dacc == "##" || dacc == "b" || dacc == "bb";
    let d: i32 = dnum.parse().map_err(|_| format!("Invalid chord bass degree {degree:?}"))?;
    if !okacc || !(1..=13).contains(&d) || dnum.starts_with('0') {
        return Err(format!("Invalid chord bass degree {degree:?}"));
    }
    let (rpc, rl, racc) = pitch_class(root)?;
    let scale = [0, 2, 4, 5, 7, 9, 11];
    let mut iv = scale[((d - 1) % 7) as usize] + 12 * ((d - 1) / 7);
    iv += dacc.matches('#').count() as i32 - dacc.matches('b').count() as i32;
    let target = (rpc + iv).rem_euclid(12);
    let li = (LETTERS.iter().position(|c| *c as char == rl).unwrap_or(0) as i32 + d - 1) % 7;
    let tl = LETTERS[li as usize] as char;
    let diff = (target - natural_pc(tl) + 6).rem_euclid(12) - 6;
    if (-2..=2).contains(&diff) {
        return Ok(format!("{tl}{}", ["bb", "b", "", "#", "##"][(diff + 2) as usize]));
    }
    let names = if format!("{racc}{dacc}").contains('#') { SHARP_NAMES } else { FLAT_NAMES };
    Ok(names[target as usize].to_string())
}

fn quality_abc(q: &str) -> Option<&'static str> {
    Some(match q {
        "maj" => "",
        "min" => "m",
        "dim" => "dim",
        "aug" => "aug",
        "7" => "7",
        "maj7" => "maj7",
        "min7" => "m7",
        "dim7" => "dim7",
        "hdim7" => "m7b5",
        "sus4" => "sus4",
        "sus2" => "sus2",
        "maj6" => "6",
        "min6" => "m6",
        "sus4(b7)" => "7sus4",
        "minmaj7" => "m(maj7)",
        _ => return None,
    })
}

/// chord_symbol_to_abc: "A:min7/b3" as "Am7/C" (None: no chord)
pub fn chord_symbol_to_abc(chord: &str) -> R<Option<String>> {
    let chord = chord.trim();
    if chord == "N" || chord == "X" || chord == "?" {
        return Ok(None);
    }
    let (root, desc) = chord.split_once(':').ok_or_else(|| format!("Chord {chord:?} is missing the ':' quality separator"))?;
    let (q, bass) = match desc.split_once('/') {
        Some((q, b)) => (q, Some(b)),
        None => (desc, None),
    };
    let qa = quality_abc(q).ok_or_else(|| format!("Unsupported chord quality {q:?} in {chord:?}; refusing to rewrite it as major"))?;
    let mut text = portable_pitch_name(root, true)? + qa;
    if let Some(b) = bass.filter(|b| !b.is_empty()) {
        text += "/";
        text += &bass_degree_to_pitch(root, b)?;
    }
    Ok(Some(text))
}

/// key_symbol_to_abc: "Eb:major" as "Eb", "A:minor" as "Am" (a name the key signatures know)
pub fn key_symbol_to_abc(key: &str) -> R<String> {
    let key = key.trim();
    let (root, minor) = if let Some((r, m)) = key.split_once(':') {
        if m != "major" && m != "minor" {
            return Err(format!("Unsupported key mode {m:?} in {key:?}"));
        }
        (r.to_string(), m == "minor")
    } else if let Some(r) = key.strip_suffix('m') {
        (r.to_string(), true)
    } else {
        (key.to_string(), false)
    };
    let m = if minor { "m" } else { "" };
    let (pc, _, acc) = pitch_class(&root)?;
    let cand = portable_pitch_name(&root, false)? + m;
    if key_signature(&cand).is_some() {
        return Ok(cand);
    }
    let flat = acc.contains('b');
    let names = if flat { FLAT_NAMES } else { SHARP_NAMES };
    let mut cand = format!("{}{m}", names[pc as usize]);
    if key_signature(&cand).is_none() {
        let other = if flat { SHARP_NAMES } else { FLAT_NAMES };
        cand = format!("{}{m}", other[pc as usize]);
    }
    if key_signature(&cand).is_none() {
        return Err(format!("Cannot encode portable ABC key for {key:?}"));
    }
    Ok(cand)
}

fn key_accidentals(key: &str) -> R<[i32; 7]> {
    let n = key_signature(key).ok_or_else(|| format!("Unsupported ABC key signature {key:?}"))?;
    let mut a = [0; 7];
    let order = if n > 0 { b"FCGDAEB" } else { b"BEADGCF" };
    for l in &order[..n.unsigned_abs() as usize] {
        a[LETTERS.iter().position(|c| c == l).unwrap_or(0)] = if n > 0 { 1 } else { -1 };
    }
    Ok(a)
}

/// note_to_abc: the key-relative spelling, an accidental written only where the bar's state changes
fn note_to_abc(note: i64, key_acc: &[i32; 7], bar_acc: &mut BTreeMap<usize, i32>) -> R<String> {
    let count: i32 = key_acc.iter().sum();
    let name = key_relative_name(count, note.rem_euclid(12) as usize).ok_or_else(|| format!("Unsupported key signature accidental count {count}"))?;
    let letter = name.chars().next().unwrap_or('C');
    let acc = &name[1..];
    let num = match acc {
        "" => 0,
        "#" => 1,
        "##" => 2,
        "b" => -1,
        _ => -2,
    };
    let mut octave = (note - 60).div_euclid(12);
    if note.rem_euclid(12) == 11 && num == -1 {
        octave += 1;
    } else if note.rem_euclid(12) == 0 && num == 1 {
        octave -= 1;
    }
    let si = LETTERS.iter().position(|c| *c as char == letter).unwrap_or(0);
    let cur = *bar_acc.get(&si).unwrap_or(&key_acc[si]);
    let mut text = String::new();
    if cur != num {
        bar_acc.insert(si, num);
        text.push_str(match num {
            -2 => "__",
            -1 => "_",
            0 => "=",
            1 => "^",
            _ => "^^",
        });
    }
    if octave > 0 {
        text.push(letter.to_ascii_lowercase());
        for _ in 1..octave {
            text.push('\'');
        }
    } else {
        text.push(letter);
        for _ in 0..(-octave) {
            text.push(',');
        }
    }
    Ok(text)
}

#[derive(Clone, Debug)]
struct Beat {
    time: f64,
    id: i64,
    numerator: i64,
    denominator: i64,
}

#[derive(Clone, Debug)]
struct Measure {
    index: usize,
    start_beat: usize,
    end_beat: usize,
    numerator: i64,
    denominator: i64,
    notated_numerator: Option<i64>,
    notated_denominator: Option<i64>,
    pad_before: bool,
}

impl Measure {
    fn start_t(&self) -> usize {
        self.start_beat * SUBBEAT_DIVISION as usize
    }
    fn end_t(&self) -> usize {
        self.end_beat * SUBBEAT_DIVISION as usize
    }
    fn abc_num(&self) -> i64 {
        self.notated_numerator.filter(|n| *n != 0).unwrap_or(self.numerator)
    }
    fn abc_den(&self) -> i64 {
        self.notated_denominator.filter(|n| *n != 0).unwrap_or(self.denominator)
    }
}

/// _mode_with_first_tiebreak
fn mode_first(v: &[i64]) -> i64 {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for x in v {
        *counts.entry(*x).or_default() += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    *v.iter().find(|x| counts[x] == max).unwrap_or(&0)
}

/// infer_measures (meter_conflict "infer"): the measures from the actual downbeats
fn infer_measures(beats: &[Beat]) -> R<(Vec<Measure>, Vec<String>)> {
    let down: Vec<usize> = beats.iter().enumerate().filter(|(_, b)| b.id == 1).map(|(i, _)| i).collect();
    if down.is_empty() {
        return Err("No downbeat (beat ID 1) exists in the beat lab".into());
    }
    let mut spans: Vec<(usize, usize, bool, bool)> = Vec::new();
    if down[0] > 0 {
        spans.push((0, down[0], true, false));
    }
    for w in down.windows(2) {
        spans.push((w[0], w[1], false, false));
    }
    if *down.last().unwrap_or(&0) < beats.len() - 1 {
        spans.push((*down.last().unwrap_or(&0), beats.len() - 1, false, true));
    }
    if spans.is_empty() {
        return Err("No positive-length measure exists between downbeats".into());
    }
    let mut diag = Vec::new();
    let mut ms = Vec::new();
    for (mi, (s, e, pickup, partial)) in spans.into_iter().enumerate() {
        let ev = &beats[s..e];
        let n = ev.len() as i64;
        if n < 1 {
            return Err(format!("Measure {mi}: empty downbeat span"));
        }
        let ids: Vec<i64> = ev.iter().map(|b| b.id).collect();
        if ids.iter().enumerate().any(|(i, x)| *x != ids[0] + i as i64) {
            return Err(format!("Measure {mi}: non-consecutive beat IDs {ids:?}"));
        }
        if !pickup && ids[0] != 1 {
            return Err(format!("Measure {mi}: full measure does not start at beat ID 1"));
        }
        let dens: Vec<i64> = ev.iter().map(|b| b.denominator).collect();
        let den = mode_first(&dens);
        let nums: Vec<i64> = ev.iter().map(|b| b.numerator).collect();
        let declared = mode_first(&nums);
        let num_conflict = nums.iter().any(|x| *x != n);
        let den_conflict = dens.iter().any(|x| *x != den);
        let all_same = nums.iter().all(|x| *x == nums[0]);
        let pad_final = partial && all_same && !den_conflict && declared >= n;
        if pad_final && declared > n {
            diag.push(format!("measure {mi}: padded final {n}/{den} span to declared {declared}/{den} with trailing rest"));
        } else if num_conflict {
            diag.push(format!("measure {mi}: inferred {n}/{den} from downbeat span; declared numerators were {nums:?}"));
        }
        if den_conflict {
            diag.push(format!("measure {mi}: placed denominator {den} at the measure boundary; row declarations were {dens:?}"));
        }
        ms.push(Measure { index: mi, start_beat: s, end_beat: e, numerator: n, denominator: den, notated_numerator: Some(if pad_final { declared } else { n }),
                          notated_denominator: None, pad_before: false });
    }
    if ms.len() >= 2 {
        let (first, next) = (&ms[0], &ms[1]);
        let fd = first.numerator as f64 / first.denominator as f64;
        let nd = next.abc_num() as f64 / next.abc_den() as f64;
        if fd < nd {
            let (nn, ndn) = (next.abc_num(), next.abc_den());
            diag.push(format!("measure 0: padded leading {}/{} span to {nn}/{ndn} with preceding rest", first.numerator, first.denominator));
            let f = &mut ms[0];
            f.notated_numerator = Some(nn);
            f.notated_denominator = Some(ndn);
            f.pad_before = true;
        }
    }
    Ok((ms, diag))
}

struct Score {
    measures: Vec<Measure>,
    times: Vec<f64>,
    quarters: Vec<f64>,
    dens: Vec<i64>,
    keys: Vec<String>,
    chords: Vec<String>,
    structures: Vec<(usize, String)>,
    voices: [Vec<i64>; 2],
}

/// _build_grid: four subbeats a beat interval (np.linspace's points), their quarter positions and denominators
fn build_grid(beats: &[Beat], ms: &[Measure]) -> R<(Vec<f64>, Vec<f64>, Vec<i64>)> {
    let mut idens = vec![0i64; beats.len() - 1];
    for m in ms {
        for d in &mut idens[m.start_beat..m.end_beat] {
            *d = m.denominator;
        }
    }
    if idens.contains(&0) {
        return Err("Downbeat spans do not cover every beat interval".into());
    }
    let (mut times, mut dens, mut qs) = (Vec::new(), Vec::new(), vec![0.0f64]);
    let mut q = 0.0f64;
    for i in 0..beats.len() - 1 {
        let (a, b) = (beats[i].time, beats[i + 1].time);
        let step = (b - a) / SUBBEAT_DIVISION as f64;
        for j in 0..SUBBEAT_DIVISION {
            times.push(j as f64 * step + a);
            dens.push(idens[i]);
        }
        let qstep = 4.0 / idens[i] as f64 / SUBBEAT_DIVISION as f64;
        for _ in 0..SUBBEAT_DIVISION {
            q += qstep;
            qs.push(q);
        }
    }
    times.push(beats[beats.len() - 1].time);
    dens.push(idens[idens.len() - 1]);
    Ok((times, qs, dens))
}

fn boundaries(times: &[f64]) -> Vec<f64> {
    times.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

/// np.searchsorted(bounds, t) (side left)
fn quantize(bounds: &[f64], t: f64) -> usize {
    bounds.partition_point(|b| *b < t)
}

fn fill_intervals(rows: &[(f64, f64, String)], times: &[f64], default: &str) -> R<Vec<String>> {
    let b = boundaries(times);
    let n = times.len();
    let mut out = vec![default.to_string(); n];
    for (a, e, v) in rows {
        let s = quantize(&b, *a).min(n - 1);
        let t = quantize(&b, *e).min(n - 1);
        if t <= s {
            return Err(format!("Interval {a:.6}-{e:.6} ({v}) is shorter than the ABC subbeat grid"));
        }
        for x in &mut out[s..t] {
            x.clone_from(v);
        }
    }
    if n > 1 {
        out[n - 1] = out[n - 2].clone();
    }
    Ok(out)
}

/// _notes_to_arr: a voice on the grid (0 rest, pitch * 2 + 2 held, + 1 at its onset)
fn notes_to_arr(notes: &[(f64, f64, i64)], times: &[f64], voice: &str) -> R<Vec<i64>> {
    let b = boundaries(times);
    let n = times.len();
    let mut out = vec![0i64; n];
    let mut sorted = notes.to_vec();
    sorted.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2)));
    for (s, e, p) in sorted {
        let a = quantize(&b, s).min(n - 1);
        let z = quantize(&b, e).min(n - 1);
        if z <= a {
            return Err(format!("{voice}: MIDI note pitch={p} at {s:.6}-{e:.6} cannot be represented on the decoded subbeat grid"));
        }
        if out[a..z].iter().any(|x| *x != 0) {
            return Err(format!("{voice}: overlapping quantized melody notes at subbeats {a}:{z}"));
        }
        let sus = p * 2 + 2;
        for x in &mut out[a..z] {
            *x = sus;
        }
        out[a] = sus + 1;
    }
    Ok(out)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

fn unit_denominator(s: &Score) -> R<i64> {
    let mut l = 1i64;
    for m in &s.measures {
        for d in [m.denominator, m.abc_den()] {
            let v = d * SUBBEAT_DIVISION;
            l = l / gcd(l, v) * v;
        }
    }
    if l > 1024 {
        return Err(format!("Required ABC unit length 1/{l} is unreasonably small"));
    }
    Ok(l)
}

fn abc_units(m: &Measure, u: i64) -> i64 {
    m.abc_num() * u / m.abc_den()
}

fn padding_units(m: &Measure, u: i64) -> i64 {
    abc_units(m, u) - m.numerator * u / m.denominator
}

fn duration_units(s: &Score, a: usize, b: usize, u: i64) -> R<i64> {
    let mut n = 0;
    for d in &s.dens[a..b] {
        let div = d * SUBBEAT_DIVISION;
        if u % div != 0 {
            return Err(format!("ABC L:1/{u} cannot express a 1/{div} subbeat exactly"));
        }
        n += u / div;
    }
    Ok(n)
}

fn estimate_tempo(s: &Score) -> R<f64> {
    let secs = s.times[s.times.len() - 1] - s.times[0];
    let qn = s.quarters[s.quarters.len() - 1] - s.quarters[0];
    if secs <= 0.0 || qn <= 0.0 {
        return Err("Cannot estimate tempo from a zero-duration score".into());
    }
    Ok(qn / secs * 60.0)
}

fn continues_pitch(v: i64, next: i64) -> bool {
    v > 0 && next == (v / 2 - 1) * 2 + 2
}

fn same_segment(v: i64, next: i64) -> bool {
    if v == 0 {
        next == 0
    } else {
        next == (v / 2 - 1) * 2 + 2
    }
}

fn split_duration(d: i64) -> R<Vec<i64>> {
    if d <= 0 {
        return Err(format!("Cannot serialize non-positive duration {d}"));
    }
    let mut out = Vec::new();
    let mut r = d;
    while r != 0 {
        if SUPPORTED.contains(&r) {
            out.push(r);
            break;
        }
        let c = SUPPORTED.iter().copied().filter(|v| *v < r).max().ok_or_else(|| format!("Duration {d} cannot be split into representable ABC values"))?;
        out.push(c);
        r -= c;
    }
    Ok(out)
}

fn render_tokens(prefix: &str, note: &str, d: i64, tie_out: bool) -> R<Vec<String>> {
    let chunks = split_duration(d)?;
    let n = chunks.len();
    Ok(chunks.into_iter().enumerate().map(|(i, c)| {
        let cont = note != "z" && (i + 1 < n || tie_out);
        format!("{}{note}{}{}", if i == 0 { prefix } else { "" }, if c == 1 { String::new() } else { c.to_string() }, if cont { "-" } else { "" })
    }).collect())
}

fn render_measure(s: &Score, vi: usize, m: &Measure, u: i64) -> R<String> {
    let voice = &s.voices[vi];
    let show = vi == 0;
    let mut bar_acc: BTreeMap<usize, i32> = BTreeMap::new();
    let mut cur_key = s.keys[m.start_t()].clone();
    let mut key_acc = key_accidentals(&cur_key)?;
    let mut parts: Vec<String> = Vec::new();
    let pad = padding_units(m, u);
    if pad < 0 {
        return Err(format!("Measure {}: notated meter is shorter than its decoded span", m.index));
    }
    let mut lead = if m.pad_before { pad } else { 0 };
    let mut trail = if m.pad_before { 0 } else { pad };
    let (st, et) = (m.start_t(), m.end_t());
    let mut t = st;
    while t < et {
        let mut cps = vec![et];
        if let Some(p) = (t + 1..et).find(|p| !same_segment(voice[t], voice[*p])) {
            cps.push(p);
        }
        if let Some(p) = (t + 1..et).find(|p| s.keys[*p] != s.keys[p - 1]) {
            cps.push(p);
        }
        if show {
            if let Some(p) = (t + 1..et).find(|p| s.chords[*p] != s.chords[p - 1]) {
                cps.push(p);
            }
        }
        let next = *cps.iter().min().unwrap_or(&et);
        let mut prefix = String::new();
        let key = &s.keys[t];
        if t > st && *key != cur_key {
            cur_key = key.clone();
            key_acc = key_accidentals(&cur_key)?;
            bar_acc.clear();
            prefix += &format!("[K:{cur_key}]");
        }
        if show && (t == st || s.chords[t] != s.chords[t - 1]) {
            if let Some(c) = chord_symbol_to_abc(&s.chords[t])? {
                prefix += &format!("\"{c}\"");
            }
        }
        let v = voice[t];
        let note = if v == 0 { "z".to_string() } else { note_to_abc(v / 2 - 1, &key_acc, &mut bar_acc)? };
        let mut d = duration_units(s, t, next, u)?;
        if t == st && lead != 0 {
            if v == 0 && prefix.is_empty() {
                d += lead;
            } else {
                parts.extend(render_tokens("", "z", lead, false)?);
            }
            lead = 0;
        }
        if v == 0 && next == et && trail != 0 {
            d += trail;
            trail = 0;
        }
        if d <= 0 {
            return Err(format!("Non-positive ABC duration at subbeats {t}:{next}"));
        }
        let tie = v > 0 && next < voice.len() && continues_pitch(v, voice[next]);
        parts.extend(render_tokens(&prefix, &note, d, tie)?);
        t = next;
    }
    if lead != 0 {
        return Err(format!("Measure {}: leading rest padding was not serialized", m.index));
    }
    if trail != 0 {
        parts.extend(render_tokens("", "z", trail, false)?);
    }
    Ok(parts.concat())
}

/// An element of a serialized measure (_MUSIC_ELEMENT_RE): a quoted chord, an inline key, a note or rest
#[derive(Debug, PartialEq)]
enum El {
    Quoted(String),
    Key(String),
    Note { note: String, duration: String, tie: bool },
}

/// re.finditer over _MUSIC_ELEMENT_RE: (start, end, element)
fn elements(s: &str) -> Vec<(usize, usize, El)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            if let Some(j) = s[i + 1..].find('"') {
                out.push((i, i + 2 + j, El::Quoted(s[i + 1..i + 1 + j].to_string())));
                i = i + 2 + j;
                continue;
            }
        }
        if s[i..].starts_with("[K:") {
            if let Some(j) = s[i + 3..].find(']') {
                if j > 0 {
                    out.push((i, i + 4 + j, El::Key(s[i + 3..i + 3 + j].to_string())));
                    i = i + 4 + j;
                    continue;
                }
            }
        }
        let mut j = i;
        while j < b.len() && matches!(b[j], b'_' | b'=' | b'^') {
            j += 1;
        }
        if j < b.len() && matches!(b[j], b'A'..=b'G' | b'a'..=b'g' | b'z') {
            j += 1;
            while j < b.len() && matches!(b[j], b',' | b'\'') {
                j += 1;
            }
            let note = s[i..j].to_string();
            let d0 = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let duration = s[d0..j].to_string();
            let tie = j < b.len() && b[j] == b'-';
            if tie {
                j += 1;
            }
            out.push((i, j, El::Note { note, duration, tie }));
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

fn compressible_rest(m: &str) -> bool {
    let mut cursor = 0;
    let mut saw = false;
    for (a, z, e) in elements(m) {
        if !m[cursor..a].is_empty() {
            return false;
        }
        cursor = z;
        match e {
            El::Quoted(_) | El::Key(_) => return false,
            El::Note { note, tie, .. } => {
                saw = true;
                if note != "z" || tie {
                    return false;
                }
            }
        }
    }
    saw && cursor == m.len()
}

fn render_group(s: &Score, vi: usize, ms: &[&Measure], u: i64) -> R<String> {
    let rendered: Vec<String> = ms.iter().map(|m| render_measure(s, vi, m, u)).collect::<R<_>>()?;
    let mut out = String::new();
    let mut i = 0;
    while i < rendered.len() {
        if !compressible_rest(&rendered[i]) {
            out += &rendered[i];
            out.push('|');
            i += 1;
            continue;
        }
        let mut e = i + 1;
        while e < rendered.len() && compressible_rest(&rendered[e]) {
            e += 1;
        }
        let n = e - i;
        out += &format!("Z{}|", if n > 1 { n.to_string() } else { String::new() });
        i = e;
    }
    Ok(out)
}

struct Group {
    measures: Vec<usize>,
    labels: Vec<String>,
    meter_changed: bool,
    key_changed: bool,
}

fn measure_groups(s: &Score) -> Vec<Group> {
    let first = &s.measures[0];
    let mut meter = (first.abc_num(), first.abc_den());
    let mut key = s.keys[first.start_t()].clone();
    let mut structure = String::new();
    let mut groups: Vec<Group> = Vec::new();
    for (mi, m) in s.measures.iter().enumerate() {
        let mt = (m.abc_num(), m.abc_den());
        let k = &s.keys[m.start_t()];
        let meter_changed = mt != meter;
        let key_changed = *k != key;
        let mut labels = Vec::new();
        for (t, l) in &s.structures {
            if !(m.start_t() <= *t && *t < m.end_t()) {
                continue;
            }
            let clean = l.split_whitespace().collect::<Vec<_>>().join(" ");
            if !clean.is_empty() && clean != structure {
                labels.push(clean.clone());
                structure = clean;
            }
        }
        let start = groups.last().is_none_or(|g| g.measures.len() >= 4) || meter_changed || key_changed || !labels.is_empty();
        if start {
            groups.push(Group { measures: vec![mi], labels, meter_changed, key_changed });
        } else if let Some(g) = groups.last_mut() {
            g.measures.push(mi);
        }
        meter = mt;
        key = s.keys[m.end_t() - 1].clone();
    }
    groups
}

fn score_to_abc(s: &Score) -> R<String> {
    let u = unit_denominator(s)?;
    let first = &s.measures[0];
    let mut lines = vec!["X:1".to_string(), "T:".to_string(), format!("M:{}/{}", first.abc_num(), first.abc_den()), format!("L:1/{u}"),
                         format!("Q:1/4={}", estimate_tempo(s)?.round_ties_even() as i64), "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"".to_string(),
                         "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"".to_string(), format!("K:{}", s.keys[first.start_t()])];
    for g in measure_groups(s) {
        lines.extend(g.labels.iter().map(|l| format!("% {l}")));
        let f = &s.measures[g.measures[0]];
        let ms: Vec<&Measure> = g.measures.iter().map(|i| &s.measures[*i]).collect();
        for (vi, v) in VOICES.iter().enumerate() {
            lines.push(format!("V: {v}"));
            if g.meter_changed {
                lines.push(format!("M:{}/{}", f.abc_num(), f.abc_den()));
            }
            if g.key_changed {
                lines.push(format!("K:{}", s.keys[f.start_t()]));
            }
            lines.push(render_group(s, vi, &ms, u)?);
        }
    }
    let text = lines.join("\n") + "\n";
    validate(&text, s)?;
    Ok(text)
}

/// _parse_music_measure: the measure's chords and inline keys at their positions (in units), its duration checked
#[allow(clippy::type_complexity)]
fn parse_measure(body: &str, expected: i64, ctx: &str) -> R<(Vec<(i64, String)>, Vec<(i64, String)>)> {
    if body == "Z" {
        return Ok((Vec::new(), Vec::new()));
    }
    let (mut pos, mut cursor) = (0i64, 0usize);
    let (mut quoted, mut keys) = (Vec::new(), Vec::new());
    for (a, z, e) in elements(body) {
        if !body[cursor..a].trim().is_empty() {
            return Err(format!("{ctx}: unsupported serialized ABC tokens {:?}", &body[cursor..a]));
        }
        cursor = z;
        match e {
            El::Quoted(q) => quoted.push((pos, q)),
            El::Key(k) => keys.push((pos, k)),
            El::Note { note, duration, tie } => {
                if tie && note == "z" {
                    return Err(format!("{ctx}: a rest cannot be tied"));
                }
                let d: i64 = if duration.is_empty() { 1 } else { duration.parse().unwrap_or(0) };
                if !SUPPORTED.contains(&d) {
                    return Err(format!("{ctx}: duration {d} is not parser-representable"));
                }
                pos += d;
            }
        }
    }
    if !body[cursor..].trim().is_empty() {
        return Err(format!("{ctx}: unsupported serialized ABC tokens {:?}", &body[cursor..]));
    }
    if pos != expected {
        return Err(format!("{ctx}: duration {pos} does not match meter duration {expected}"));
    }
    let b = body.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'-' && (i == 0 || b[i - 1].is_ascii_whitespace() || b[i - 1] == b'|') && b.get(i + 1).is_some_and(|c| b"_=^ABCDEFGabcdefg".contains(c)) {
            return Err(format!("{ctx}: tie is written before its second note"));
        }
    }
    Ok((quoted, keys))
}

#[allow(clippy::type_complexity)]
fn parse_voice_group(lines: &[&str], mut c: usize, voice: &str, gi: usize) -> R<(usize, BTreeMap<String, String>, Vec<String>)> {
    let want = format!("V: {voice}");
    if lines.get(c) != Some(&want.as_str()) {
        return Err(format!("Group {gi}: expected {want}, got {:?}", lines.get(c).unwrap_or(&"<end>")));
    }
    c += 1;
    let mut fields = BTreeMap::new();
    while c < lines.len() && (lines[c].starts_with("M:") || lines[c].starts_with("K:")) {
        let (n, v) = lines[c].split_once(':').unwrap_or(("", ""));
        if fields.contains_key(n) {
            return Err(format!("Group {gi} {voice}: repeated {n}: field"));
        }
        fields.insert(n.to_string(), v.to_string());
        c += 1;
    }
    let music = *lines.get(c).ok_or_else(|| format!("Group {gi} {voice}: missing music line"))?;
    if ["V:", "M:", "K:", "%"].iter().any(|p| music.starts_with(p)) {
        return Err(format!("Group {gi} {voice}: invalid music line {music:?}"));
    }
    c += 1;
    let split: Vec<&str> = music.split('|').collect();
    if split.last().is_some_and(|l| !l.trim().is_empty()) {
        return Err(format!("Group {gi} {voice}: music line must end with a barline"));
    }
    let bars: Vec<&str> = split[..split.len() - 1].iter().map(|b| b.trim()).collect();
    if bars.iter().any(|b| b.is_empty()) {
        return Err(format!("Group {gi} {voice}: empty serialized measure"));
    }
    let mut out = Vec::new();
    for b in bars {
        let count = b.strip_prefix('Z').filter(|r| r.is_empty() || (r.len() == 1 && (b'1'..=b'4').contains(&r.as_bytes()[0])));
        match count {
            None => out.push(b.to_string()),
            Some("1") => return Err(format!("Group {gi} {voice}: Z1 must be written as Z")),
            Some(r) => {
                let n: usize = if r.is_empty() { 1 } else { r.parse().unwrap_or(1) };
                out.extend(std::iter::repeat_n("Z".to_string(), n));
            }
        }
    }
    if !(1..=4).contains(&out.len()) {
        return Err(format!("Group {gi} {voice}: expected 1-4 semantic measures"));
    }
    Ok((c, fields, out))
}

fn expected_chords(s: &Score, m: &Measure, u: i64) -> R<Vec<(i64, String)>> {
    let lead = if m.pad_before { padding_units(m, u) } else { 0 };
    let mut v = Vec::new();
    for t in m.start_t()..m.end_t() {
        if t != m.start_t() && s.chords[t] == s.chords[t - 1] {
            continue;
        }
        if let Some(c) = chord_symbol_to_abc(&s.chords[t])? {
            v.push((lead + duration_units(s, m.start_t(), t, u)?, c));
        }
    }
    Ok(v)
}

fn expected_keys(s: &Score, m: &Measure, u: i64) -> R<Vec<(i64, String)>> {
    let lead = if m.pad_before { padding_units(m, u) } else { 0 };
    let mut v = Vec::new();
    for t in m.start_t() + 1..m.end_t() {
        if s.keys[t] != s.keys[t - 1] {
            v.push((lead + duration_units(s, m.start_t(), t, u)?, s.keys[t].clone()));
        }
    }
    Ok(v)
}

/// validate_serialized_abc: the invariants upstream checks before it lets an ABC out
fn validate(text: &str, s: &Score) -> R<()> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.first() != Some(&"X:1") {
        return Err("ABC must start with X:1".into());
    }
    if lines.get(1) != Some(&"T:") {
        return Err("ABC title must be fixed as empty T:".into());
    }
    if lines.iter().any(|l| l.starts_with("%abc-") || l.starts_with("I:abc-creator")) {
        return Err("ABC must not contain version or creator metadata".into());
    }
    if text.contains("%%MIDI gchordoff") || text.contains("% ss2") {
        return Err("ABC must not contain %%MIDI gchordoff or % ss2 metadata".into());
    }
    let voice_ids: Vec<&str> = lines.iter().filter_map(|l| {
        if l.starts_with("V: Vocal ") {
            Some("Vocal")
        } else if l.starts_with("V: Ins ") {
            Some("Ins")
        } else {
            None
        }
    }).collect();
    if voice_ids != VOICES {
        return Err(format!("Expected fixed Vocal/Ins voice definitions, got {voice_ids:?}"));
    }
    let ins_at = lines.iter().position(|l| l.starts_with("V: Ins "));
    let hk = lines.iter().enumerate().position(|(i, l)| i > 0 && l.starts_with("K:") && ins_at.is_some_and(|h| h < i)).ok_or("ABC header K: field is missing")?;
    let first = &s.measures[0];
    let want_m = format!("M:{}/{}", first.abc_num(), first.abc_den());
    let hm: Vec<&str> = lines[..=hk].iter().copied().filter(|l| l.starts_with("M:")).collect();
    if hm != [want_m.as_str()] {
        return Err(format!("ABC header meters {hm:?} != [{want_m:?}]"));
    }
    let want_k = format!("K:{}", s.keys[first.start_t()]);
    let hks: Vec<&str> = lines[..=hk].iter().copied().filter(|l| l.starts_with("K:")).collect();
    if hks != [want_k.as_str()] {
        return Err(format!("ABC header keys {hks:?} != [{want_k:?}]"));
    }
    let u = unit_denominator(s)?;
    let mut c = hk + 1;
    for (gi, g) in measure_groups(s).iter().enumerate() {
        let mut labels = Vec::new();
        while c < lines.len() && lines[c].starts_with("% ") {
            labels.push(lines[c][2..].trim().to_string());
            c += 1;
        }
        if labels != g.labels {
            return Err(format!("Group {gi}: structure labels {labels:?} != {:?}", g.labels));
        }
        let (c1, vf, vb) = parse_voice_group(&lines, c, "Vocal", gi)?;
        let (c2, inf, ib) = parse_voice_group(&lines, c1, "Ins", gi)?;
        c = c2;
        if vf != inf {
            return Err(format!("Group {gi}: meter/key changes must be scoped to both voices"));
        }
        let f = &s.measures[g.measures[0]];
        let mut want = BTreeMap::new();
        if g.meter_changed {
            want.insert("M".to_string(), format!("{}/{}", f.abc_num(), f.abc_den()));
        }
        if g.key_changed {
            want.insert("K".to_string(), s.keys[f.start_t()].clone());
        }
        if vf != want {
            return Err(format!("Group {gi}: fields {vf:?} != required changes {want:?}"));
        }
        if vb.len() != g.measures.len() || ib.len() != g.measures.len() {
            return Err(format!("Group {gi}: both voices must contain {} measures", g.measures.len()));
        }
        for (bi, mi) in g.measures.iter().enumerate() {
            let m = &s.measures[*mi];
            let dur = abc_units(m, u);
            for (voice, bars) in [("Vocal", &vb), ("Ins", &ib)] {
                let (q, k) = parse_measure(&bars[bi], dur, &format!("measure {} {voice}", m.index))?;
                let ek = expected_keys(s, m, u)?;
                if k != ek {
                    return Err(format!("Measure {} {voice}: inline keys {k:?} != {ek:?}", m.index));
                }
                if voice == "Vocal" {
                    let ec = expected_chords(s, m, u)?;
                    if q != ec {
                        return Err(format!("Measure {}: chord symbols {q:?} != {ec:?}", m.index));
                    }
                } else if !q.is_empty() {
                    return Err(format!("Measure {}: chords must only be in Vocal", m.index));
                }
            }
        }
    }
    if c != lines.len() {
        return Err(format!("Unexpected trailing ABC body lines: {:?}", &lines[c..(c + 5).min(lines.len())]));
    }
    Ok(())
}

/// The seconds a MIDI tick is (pretty_midi at its default 120 bpm, 960 ticks a beat)
const TICK: f64 = 60.0 / (120.0 * 960.0);

/// A time through pretty_midi's write and read: its tick (Python's round: half to even), back to seconds
fn midi_tick(t: f64) -> i64 {
    (t / TICK).round_ties_even() as i64
}

/// interval_rows: (start, end, value) of the events carrying `value`, each to the next one's start (the last to
/// `duration`), the empty ones dropped
fn interval_rows(events: &[Event], value: &dyn Fn(&Event) -> Option<String>, duration: f64) -> Vec<(f64, f64, String)> {
    let mut rows: Vec<(f64, f64, String)> = events.iter().filter_map(|e| value(e).map(|v| (e.time.unwrap_or(0.0), 0.0, v))).collect();
    for i in 0..rows.len() {
        rows[i].1 = if i + 1 < rows.len() { rows[i + 1].0 } else { duration };
    }
    rows.into_iter().filter(|r| r.1 > r.0).collect()
}

/// np.median of a slice
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// What the notation made: the ABC, or why there is none (upstream's abc_error), and its diagnostics
#[derive(Clone, Debug, Default)]
pub struct Sheet {
    pub abc: Option<String>,
    pub error: Option<String>,
    pub diagnostics: Vec<String>,
    pub measures: usize,
    pub notes: usize,
}

/// export_result's ABC (paper false): the stitched events of the whole song, its duration in seconds
pub fn export(events: &[Event], duration: f64, melody_only: bool) -> Sheet {
    let mut sh = Sheet::default();
    // the melody notes: [start, end, pitch, track], each clipped to the song
    let mut notes: Vec<(f64, f64, i64, u32)> = Vec::new();
    for e in events {
        let start = e.time.unwrap_or(0.0);
        for n in &e.notes {
            let end = duration.min(n.end_time.unwrap_or(start));
            if end > start {
                notes.push((start, end, n.pitch as i64, n.track));
            }
        }
    }
    notes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));
    sh.notes = notes.len();
    match abc(events, duration, &notes, melody_only, &mut sh.diagnostics) {
        Ok((text, measures)) => {
            sh.abc = Some(text);
            sh.measures = measures;
        }
        Err(e) => sh.error = Some(e),
    }
    sh
}

fn abc(events: &[Event], duration: f64, notes: &[(f64, f64, i64, u32)], melody_only: bool, diag: &mut Vec<String>) -> R<(String, usize)> {
    // rhythm_rows: the beats (time, beat number, meter)
    let mut beats: Vec<Beat> = Vec::new();
    let mut meter: Option<(u32, u32)> = None;
    for e in events {
        let (m, eighth) = e.rhythm();
        if !e.has(F_RHYTHM) {
            continue;
        }
        meter = m.or(meter);
        if let (Some(ei), Some((num, den))) = (eighth, meter) {
            // the model's eighth-note position, as a beat of the meter's denominator
            let p = ei as i64 * den as i64;
            if p % 8 != 0 {
                return Err(format!("Eighth position {ei} is off the {num}/{den} beat grid"));
            }
            let p = p / 8;
            if !(0..num as i64).contains(&p) {
                return Err(format!("Eighth position {ei} is outside meter ({num}, {den})"));
            }
            beats.push(Beat { time: e.time.unwrap_or(0.0), id: p + 1, numerator: num as i64, denominator: den as i64 });
        }
    }
    if beats.len() < 2 {
        return Err("At least two decoded beats are required for ABC".into());
    }
    // the last beat is the score's end boundary: the final tempo continued to the audio's (or the last note's) end
    let tail: Vec<f64> = beats[beats.len().saturating_sub(9)..].windows(2).map(|w| w[1].time - w[0].time).collect();
    let period = median(&tail);
    if period.is_nan() || period <= 0.0 {
        return Err("Decoded beats must increase in time".into());
    }
    let end = duration.max(notes.iter().map(|n| n.1).fold(0.0, f64::max));
    while beats[beats.len() - 1].time < end - 1e-6 {
        let p = beats[beats.len() - 1].clone();
        beats.push(Beat { time: p.time + period, id: p.id % p.numerator + 1, numerator: p.numerator, denominator: p.denominator });
    }
    let (b0, bn) = (beats[0].time, beats[beats.len() - 1].time);
    let clip = |rows: Vec<(f64, f64, String)>| -> Vec<(f64, f64, String)> {
        rows.into_iter().filter(|(a, b, _)| *b > b0 && *a < bn).map(|(a, b, v)| (b0.max(a), bn.min(b), v)).collect()
    };
    let fulls = full_chords();
    let mm = majmin_chords();
    let key_rows = interval_rows(events, &|e: &Event| e.has(F_KEY).then(|| e.key().unwrap_or_default()), duration);
    let chord_rows = interval_rows(events, &|e: &Event| e.has(F_CHORD).then(|| e.chord(&fulls, &mm).unwrap_or_default()), duration);
    let chords = clip(correct_chord_rows(&chord_rows, &key_rows)?);
    let keys = clip(key_rows.iter().map(|(a, b, k)| (*a, *b, normalize_key(k))).collect());
    let structures = clip(interval_rows(events, &|e: &Event| e.has(F_STRUCT).then(|| e.structure().unwrap_or_default().to_string()), duration));
    if key_rows.is_empty() {
        return Err("No key was decoded; cannot construct a keyed ABC score".into());
    }
    // notation_notes: each track monophonic (a note clipped at the next onset)
    let mut clean: Vec<(f64, f64, i64, u32)> = Vec::new();
    for track in [0u32, 1] {
        let mut ordered: Vec<(f64, f64, i64, u32)> = notes.iter().copied().filter(|n| n.3 == track).collect();
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.total_cmp(&b.1)));
        for i in 0..ordered.len() {
            if i + 1 < ordered.len() && ordered[i].1 > ordered[i + 1].0 + 1e-6 {
                ordered[i].1 = ordered[i + 1].0;
                diag.push(format!("notation only: clipped track {track} note at {:.3} to next onset", ordered[i].0));
            }
            if ordered[i].1 > ordered[i].0 + 1e-6 {
                clean.push(ordered[i]);
            }
        }
    }
    // through the MIDI file: the ticks' times, a note of no ticks dropped
    let mut voices: [Vec<(f64, f64, i64)>; 2] = Default::default();
    for (s, e, p, tr) in &clean {
        let (a, z) = (midi_tick(*s), midi_tick(*e));
        if z > a {
            voices[*tr as usize].push((TICK * a as f64, TICK * z as f64, *p));
        }
    }
    // build_rebuilt_abc_score_from_data: the rows parsed and checked as upstream reads them
    for w in beats.windows(2) {
        if w[1].time <= w[0].time {
            return Err("beats: beat times must be strictly increasing".into());
        }
    }
    for b in &beats {
        if b.id < 1 || b.numerator < 1 || b.denominator < 1 || b.denominator & (b.denominator - 1) != 0 {
            return Err(format!("beats: invalid beat row {b:?}"));
        }
    }
    let check_rows = |rows: &[(f64, f64, String)], what: &str| -> R<()> {
        let mut prev: Option<f64> = None;
        for (a, b, _) in rows {
            if b <= a {
                return Err(format!("{what}: end must be after start"));
            }
            if prev.is_some_and(|p| *a < p - 1e-6) {
                return Err(format!("{what}: overlapping intervals"));
            }
            prev = Some(*b);
        }
        Ok(())
    };
    check_rows(&keys, "keys")?;
    check_rows(&structures, "structures")?;
    let keys: Vec<(f64, f64, String)> = keys.iter().map(|(a, b, k)| Ok((*a, *b, key_symbol_to_abc(k)?))).collect::<R<_>>()?;
    if keys.is_empty() {
        return Err("keys: at least one key interval is required".into());
    }
    let structures: Vec<(f64, f64, String)> = structures.iter().map(|(a, b, l)| (*a, *b, l.trim().to_string())).collect();
    let chords: Vec<(f64, f64, String)> = if melody_only {
        Vec::new()
    } else {
        check_rows(&chords, "chords")?;
        for (_, _, c) in &chords {
            chord_symbol_to_abc(c)?;
        }
        chords.iter().map(|(a, b, c)| (*a, *b, c.trim().to_string())).collect()
    };
    let (measures, d) = infer_measures(&beats)?;
    diag.extend(d);
    let (times, quarters, dens) = build_grid(&beats, &measures)?;
    let v0 = notes_to_arr(&voices[0], &times, "Vocal")?;
    let v1 = notes_to_arr(&voices[1], &times, "Ins")?;
    let key_arr = fill_intervals(&keys, &times, &keys[0].2)?;
    let chord_arr = if melody_only { vec!["N".to_string(); times.len()] } else { fill_intervals(&chords, &times, "N")? };
    let b = boundaries(&times);
    let st: Vec<(usize, String)> = structures.iter().map(|(a, _, l)| (quantize(&b, *a).min(times.len() - 1), l.clone())).collect();
    let score = Score { measures, times, quarters, dens, keys: key_arr, chords: chord_arr, structures: st, voices: [v0, v1] };
    let text = score_to_abc(&score)?;
    Ok((text, score.measures.len()))
}

