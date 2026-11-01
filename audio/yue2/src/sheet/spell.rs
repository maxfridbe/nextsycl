//! Key and chord spelling (chord_spelling_sheetsage2.py, "ported from SheetSage2 LM"): a key's canonical name
//! (`normalize_key_name`: D#:major is Eb:major, A#:minor Bb:minor), and a chord's root spelled for the local key
//! (`correct_chord_spelling`: of the seven letters the root could be written on, the one whose chord tones sit closest
//! to the key on the circle of fifths - D#/G#/A# become Eb/Ab/Bb in C minor). Pitches, qualities and inversions are
//! never changed; only the root's letter and accidentals.

const KEY_MAP: [[&str; 12]; 7] = [
    ["C:major", "Db:major", "D:major", "Eb:major", "E:major", "F:major", "F#:major", "G:major", "Ab:major", "A:major", "Bb:major", "B:major"],
    ["D:dorian", "Eb:dorian", "E:dorian", "F:dorian", "F#:dorian", "G:dorian", "G#:dorian", "A:dorian", "Bb:dorian", "B:dorian", "C:dorian", "C#:dorian"],
    ["E:phrygian", "F:phrygian", "F#:phrygian", "G:phrygian", "G#:phrygian", "A:phrygian", "A#:phrygian", "B:phrygian", "C:phrygian", "C#:phrygian", "D:phrygian",
     "D#:phrygian"],
    ["F:lydian", "Gb:lydian", "G:lydian", "Ab:lydian", "A:lydian", "Bb:lydian", "B:lydian", "C:lydian", "Db:lydian", "D:lydian", "Eb:lydian", "E:lydian"],
    ["G:mixolydian", "Ab:mixolydian", "A:mixolydian", "Bb:mixolydian", "B:mixolydian", "C:mixolydian", "C#:mixolydian", "D:mixolydian", "Eb:mixolydian",
     "E:mixolydian", "F:mixolydian", "F#:mixolydian"],
    ["A:minor", "Bb:minor", "B:minor", "C:minor", "C#:minor", "D:minor", "D#:minor", "E:minor", "F:minor", "F#:minor", "G:minor", "G#:minor"],
    ["B:locrian", "C:locrian", "C#:locrian", "D:locrian", "D#:locrian", "E:locrian", "E#:locrian", "F#:locrian", "G:locrian", "G#:locrian", "A:locrian", "A#:locrian"],
];
const MODE_NAMES: [&str; 7] = ["major", "dorian", "phrygian", "lydian", "mixolydian", "minor", "locrian"];
const MODE_STARTS: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
const NOTE_NAMES: &[u8; 7] = b"CDEFGAB";
const DEFAULT_SPELLING: [&str; 12] = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7"];
const CIRCLE_OF_FIFTH_INV: [i32; 7] = [1, 3, 5, 0, 2, 4, 6];
/// circle-of-fifths offsets of the major scale's degrees 1-7 from the tonic
const MAJOR_SCALE_FIFTHS: [i32; 7] = [0, 2, 4, -1, 1, 3, 5];

/// The qualities' chroma templates (the root marked 2), as upstream's QUALITIES
fn quality(q: &str) -> Option<[u8; 12]> {
    Some(match q {
        "maj" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "min" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "aug" => [2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "dim" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "sus4" => [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        "sus4(b7)" => [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "sus4(b7,9)" => [2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "sus2" => [2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "7" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min7" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "minmaj7" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        "maj6" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "min6" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "9" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj9" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min9" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "7(b9)" => [2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "7(#9)" => [2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj6(9)" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "min6(9)" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "maj(9)" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "min(9)" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "maj(11)" => [2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        "min(11)" => [2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        "11" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        "maj9(11)" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        "min11" => [2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "13" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "maj13" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "min13" => [2, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "dim7" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "hdim7" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        _ => return None,
    })
}

/// mir_eval.chord.pitch_class_to_semitone: a letter and its sharps / flats, mod 12 (None: not a pitch name)
pub fn semitone(pc: &str) -> Option<i32> {
    let mut s = 0i32;
    for (i, c) in pc.chars().enumerate() {
        match (i, c) {
            (0, l) => s = [0, 2, 4, 5, 7, 9, 11][NOTE_NAMES.iter().position(|n| *n as char == l)?],
            (_, '#') => s += 1,
            (_, 'b') => s -= 1,
            _ => return None,
        }
    }
    Some(s.rem_euclid(12))
}

/// normalize_key_name: the key in SheetSage2 LM's canonical spelling
pub fn normalize_key(key: &str) -> String {
    let Some((tonic, mode)) = key.split_once(':') else { return key.to_string() };
    let (Some(m), Some(t)) = (MODE_NAMES.iter().position(|x| *x == mode), semitone(tonic)) else { return key.to_string() };
    KEY_MAP[m][(t - MODE_STARTS[m]).rem_euclid(12) as usize].to_string()
}

/// a scale degree ("b3", "#5", "bb7") as (degree - 1, offset)
fn degree(sd: &str) -> (i32, i32) {
    let sharps = sd.chars().filter(|c| *c == '#').count() as i32;
    let flats = sd.chars().filter(|c| *c == 'b').count() as i32;
    let n: i32 = sd.trim_matches(|c| c == '#' || c == 'b').parse().unwrap_or(1);
    let off = if sd.starts_with('#') { sharps } else if sd.starts_with('b') { -flats } else { 0 };
    (n - 1, off)
}

/// The quality's tones as scale degrees of the root's own major scale (get_chord_spelling_table)
fn spelling_table(q: &str) -> Option<Vec<(i32, i32)>> {
    let chroma = quality(q)?;
    let mut v = Vec::new();
    for (i, c) in chroma.iter().enumerate() {
        if *c > 0 {
            let s = if q.contains("#9") && i == 3 {
                "#2"
            } else if q.contains("dim7") && i == 9 {
                "bb7"
            } else {
                DEFAULT_SPELLING[i]
            };
            v.push(degree(s));
        }
    }
    Some(v)
}

/// spell_chord_tones: every tone's (letter, accidental) from the root's
fn spell_tones(root: (i32, i32), q: &str) -> Option<Vec<(i32, i32)>> {
    let iv = spelling_table(q)?;
    let root_pos = CIRCLE_OF_FIFTH_INV[root.0 as usize] + root.1 * 7;
    Some(iv.iter().map(|(d, o)| {
        let letter = (root.0 + d).rem_euclid(7);
        let tone_pos = root_pos + MAJOR_SCALE_FIFTHS[*d as usize] + o * 7;
        (letter, (tone_pos - CIRCLE_OF_FIFTH_INV[letter as usize]).div_euclid(7))
    }).collect())
}

/// score_spelling_under_key: how far the tones sit from the key's on the circle of fifths (the root counted twice)
fn score(sp: &[(i32, i32)], key: (i32, i32)) -> f64 {
    let key_pos = CIRCLE_OF_FIFTH_INV[key.0 as usize] + key.1 * 7;
    let rel: Vec<i32> = sp.iter().map(|(l, a)| ((CIRCLE_OF_FIFTH_INV[l.rem_euclid(7) as usize] + a * 7) - (key_pos + 2)).abs().saturating_sub(3).max(0)).collect();
    rel.iter().sum::<i32>() as f64 + rel.first().copied().unwrap_or(0) as f64
}

/// note_name_to_tuple
fn note_tuple(n: &str) -> (i32, i32) {
    let l = NOTE_NAMES.iter().position(|c| *c as char == n.chars().next().unwrap_or('C')).unwrap_or(0) as i32;
    let off = if n.ends_with('#') { n.matches('#').count() as i32 } else if n.ends_with('b') { -(n.matches('b').count() as i32) } else { 0 };
    (l, off)
}

/// correct_chord_spelling: the chord's root spelled for `key` (inversions kept as they are)
pub fn correct_chord(label: &str, key: &str) -> Result<String, String> {
    if label == "N" || label == "X" {
        return Ok(label.to_string());
    }
    let (kt, km) = key.split_once(':').ok_or_else(|| format!("key {key:?}"))?;
    let km = MODE_NAMES.iter().position(|m| *m == km).ok_or_else(|| format!("key mode {km:?}"))?;
    let scale = (semitone(kt).ok_or_else(|| format!("key {key:?}"))? - MODE_STARTS[km]).rem_euclid(12);
    let key_sp = note_tuple(KEY_MAP[0][scale as usize].split(':').next().unwrap_or("C"));
    let (body, inversion) = match label.split_once('/') {
        Some((b, i)) => (b, i),
        None => (label, ""),
    };
    let (root, q) = body.split_once(':').ok_or_else(|| format!("chord {label:?}"))?;
    let rs = semitone(root).ok_or_else(|| format!("chord root {root:?}"))?;
    let off = |l: usize| (rs - semitone(&(NOTE_NAMES[l] as char).to_string()).unwrap_or(0) + 6).rem_euclid(12) - 6;
    let mut best = (f64::INFINITY, 0usize);
    for l in 0..7 {
        let sp = spell_tones((l as i32, off(l)), q).ok_or_else(|| format!("chord quality {q:?}"))?;
        let s = score(&sp, key_sp);
        if s < best.0 {
            best = (s, l);
        }
    }
    let a = off(best.1);
    let mut r = (NOTE_NAMES[best.1] as char).to_string();
    r.push_str(&if a < 0 { "b".repeat((-a) as usize) } else { "#".repeat(a as usize) });
    Ok(if inversion.is_empty() { format!("{r}:{q}") } else { format!("{r}:{q}/{inversion}") })
}

/// correct_chord_rows: each chord spelled for the key at its midpoint (the decoded keys' intervals; none: kept)
pub fn correct_chord_rows(chords: &[(f64, f64, String)], keys: &[(f64, f64, String)]) -> Result<Vec<(f64, f64, String)>, String> {
    if keys.is_empty() {
        return Ok(chords.to_vec());
    }
    let bounds: Vec<f64> = keys[..keys.len() - 1].iter().map(|k| k.1).collect();
    chords.iter().map(|(a, b, l)| {
        let mid = (a + b) / 2.0;
        let i = bounds.partition_point(|x| *x < mid);
        Ok((*a, *b, correct_chord(l, &keys[i].2)?))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_chords_are_spelled_as_upstream_spells_them() {
        assert_eq!(normalize_key("D#:major"), "Eb:major");
        assert_eq!(normalize_key("A#:minor"), "Bb:minor");
        assert_eq!(normalize_key("C#:major"), "Db:major");
        assert_eq!(correct_chord("D#:maj", "C:minor").unwrap(), "Eb:maj");
        assert_eq!(correct_chord("G#:maj", "C:minor").unwrap(), "Ab:maj");
        assert_eq!(correct_chord("C:maj/3", "E:major").unwrap(), "C:maj/3");
        assert_eq!(correct_chord("A#:maj", "G:major").unwrap(), "Bb:maj");
    }
}
